import networkx as nx
from collections import OrderedDict
import itertools
import math
import torch
import torch.nn as nn

import quantlib.graphs as qg
from quantlib.graphs.grrules.dporules import DPORule
from quantlib.graphs.grrules import Seeker
from quantlib.graphs.graphs import __KERNEL_PARTITION__, __NODE_ID_FORMAT__, PyTorchNode

import quantlib.algorithms as qa

from .folding import foldsteinqconvbnste, foldconvbnste, foldsteinqconvbn


__all__ = [
    'FoldSTEINQConvBNSTETypeARule',
    'FoldSTEINQConvBNSTETypeBRule',
    'FoldConvBNSTERule',
    'FoldSTEINQConvBNRule',
]


class FoldSTEINQConvBNSTETypeARule(DPORule):  # w/o max pooling

    def __init__(self, gamma_int_bits=10, gamma_frac_bits=17, beta_int_bits=8, beta_frac_bits=0):

        self._gamma_int_bits  = gamma_int_bits
        self._gamma_frac_bits = gamma_frac_bits
        self._beta_int_bits   = beta_int_bits
        self._beta_frac_bits  = beta_frac_bits

        # Nodes of the interface
        K_types = OrderedDict()
        K_types.update({'HPTout': qg.graphs.HelperOutputPrecisionTunnel.__name__})
        K_types.update({'HPTin':  qg.graphs.HelperInputPrecisionTunnel.__name__})
        K_types = OrderedDict([('/'.join(['K-term', k]), v) for k, v in K_types.items()])

        # Nodes in the core template graph
        LK_types = OrderedDict()
        LK_types.update({'STEin':     qa.ste.STEActivation.__name__})
        LK_types.update({'Conv':      qa.inq.INQConv2d.__name__})
        LK_types.update({'BatchNorm': nn.BatchNorm2d.__name__})
        LK_types.update({'ReLU':      nn.ReLU.__name__})
        LK_types.update({'STEout':    qa.ste.STEActivation.__name__})
        LK_types = OrderedDict([('/'.join(['L-term', k]), v) for k, v in LK_types.items()])

        # Nodes in the core replacement graph
        RK_types = OrderedDict()
        RK_types.update({'TWConv':   nn.Conv2d.__name__})
        RK_types.update({'XPAffine': nn.Conv2d.__name__})
        RK_types.update({'S&C':      qg.graphs.ShiftAndClip.__name__})
        RK_types = OrderedDict([('/'.join(['R-term', k]), v) for k, v in RK_types.items()])

        K_node_IDs  = list(K_types.keys())
        LK_node_IDs = list(LK_types.keys())
        RK_node_IDs = list(RK_types.keys())

        # define the template graph L [L-term]
        L_node_IDs = [K_node_IDs[0]] + LK_node_IDs + [K_node_IDs[-1]]
        self.L = nx.DiGraph()
        # Define arcs between nodes in full template graph
        self.L.add_edges_from({(u, v) for u, v in zip(L_node_IDs[:-1], L_node_IDs[1:])})

        # Here, graph is only operation nodes
        # Necessary for seeker
        nx.set_node_attributes(self.L, {vL: __KERNEL_PARTITION__ for vL in set(self.L.nodes)}, 'bipartite')
        nx.set_node_attributes(self.L, {**K_types, **LK_types}, 'type')

        # define the context (sub-)graph K [K-term]
        VK = set(K_node_IDs)  # precision tunnel nodes define the context graph
        self.K = self.L.subgraph(VK)

        # define the template (sub-)graph L\K
        VLK = set(self.L.nodes).difference(set(self.K.nodes))
        self.LK = self.L.subgraph(VLK)

        # define the replacement (sub-)graph R\K ["gluing" R\K to K yields the graph R, i.e., the R-term]
        self.RK = nx.DiGraph()
        self.RK.add_edges_from({(u, v) for u, v in zip(RK_node_IDs[:-1], RK_node_IDs[1:])})
        nx.set_node_attributes(self.RK, {vRK: __KERNEL_PARTITION__ for vRK in set(self.RK.nodes)}, 'bipartite')
        nx.set_node_attributes(self.RK, RK_types, 'type')

        # define the arcs that go from the vertices of K to those of R\K, and viceversa
        E_K2RK = {(K_node_IDs[0], RK_node_IDs[0])}
        E_RK2K = {(RK_node_IDs[-1], K_node_IDs[-1])}
        E_K2RK2K = E_K2RK | E_RK2K
        # disintegrate `E_K2RK` and `E_RK2K` along fibres to speed up rule application
        # A fibre is kind of like fixing one argument of a two input one output function and looking at all possible outputs
        self.F_K2RK = {vK: set(arc for arc in E_K2RK if arc[0] == vK) for vK in set(self.K.nodes)}
        self.F_RK2K = {vK: set(arc for arc in E_RK2K if arc[1] == vK) for vK in set(self.K.nodes)}

        # # glue together the (sub-)graphs L\K and R\K along the vertices of K
        # self.S = nx.compose(self.L, self.RK)
        # self.S.add_edges_from(E_K2RK2K)

        # since the GRR's L-term has been modified, rebuild the seeker
        self.seeker = Seeker(self.L)

        # this machinery can generate always-new identifiers for different rule applications
        self._counter = itertools.count()

    def _get_rule_count(self):
        rule_count = ''.join(['FINQBNSTETA', __NODE_ID_FORMAT__.format(next(self._counter))])
        return rule_count

    def core(self, HI, g, nodes_dict):

        # generate the substitute (sub-)graph J\I
        rule_count = self._get_rule_count()
        g_RK2JI = {vRK: '_'.join([rule_count, vRK.replace('R-term/', '')]) for vRK in set(self.RK.nodes)}
        JI = nx.relabel_nodes(self.RK, g_RK2JI, copy=True)

        # get pointers to the old modules;
        # these pointers will enable two actions:
        #   1. extracting the arguments required to perform the folding
        #   2. extracting the parameters to instantiate the new modules
        g_L2H = {vL: vH for vH, vL in g.items()}
        mstein  = nodes_dict[g_L2H['/'.join(['L-term', 'STEin'])]].nobj
        minq2d  = nodes_dict[g_L2H['/'.join(['L-term', 'Conv'])]].nobj
        mbn2d   = nodes_dict[g_L2H['/'.join(['L-term', 'BatchNorm'])]].nobj
        msteout = nodes_dict[g_L2H['/'.join(['L-term', 'STEout'])]].nobj

        # fold
        weight, gamma, beta = foldsteinqconvbnste(mstein.num_levels, mstein.abs_max_value,
                                                  minq2d.weight_frozen,
                                                  mbn2d.running_mean, mbn2d.running_var, mbn2d.eps, mbn2d.weight, mbn2d.bias,
                                                  msteout.num_levels, msteout.abs_max_value,
                                                  gamma_int_bits=self._gamma_int_bits, gamma_frac_bits=self._gamma_frac_bits,
                                                  beta_int_bits=self._beta_int_bits, beta_frac_bits=self._beta_frac_bits)

        # build the new modules
        mtwconv = nn.Conv2d(minq2d.in_channels, minq2d.out_channels, minq2d.kernel_size,
                            stride=minq2d.stride, padding=minq2d.padding, dilation=minq2d.dilation, groups=minq2d.groups,
                            bias=minq2d.bias is not None).to(torch.device('cpu'))
        mtwconv.weight.data = weight

        mxpaffine = nn.Conv2d(minq2d.out_channels, minq2d.out_channels, 1,
                              stride=1, padding=0, groups=minq2d.out_channels,
                              bias=True).to(torch.device('cpu'))
        mxpaffine.weight.data = gamma
        mxpaffine.bias.data   = beta

        msandc = qg.graphs.ShiftAndClip(n_bits=math.ceil(math.log(msteout.num_levels, 2)),
                                        shift=self._gamma_frac_bits,
                                        signed=True, only_positive=True).to(torch.device('cpu'))

        # register the newly created nodes
        vJI_2_ptnode = {}
        vJI_2_ptnode[g_RK2JI['/'.join(['R-term', 'TWConv'])]]   = PyTorchNode(mtwconv)
        vJI_2_ptnode[g_RK2JI['/'.join(['R-term', 'XPAffine'])]] = PyTorchNode(mxpaffine)
        vJI_2_ptnode[g_RK2JI['/'.join(['R-term', 'S&C'])]]      = PyTorchNode(msandc)

        return JI, vJI_2_ptnode

    # G: Full/original graph
    # nodes_dict: Mapping between node identifiers of G and actual underlying objects
    # g: One instance of all occurences of the template in G, i.e. one application point for the replacement rule -> one morphism
    def apply(self, G, nodes_dict, g):

        # create new containers
        G = G.copy()
        # Dictionary mapping of node identifiers to a payload
        # keys in nodes_dict should be the same as G.nodes
        nodes_dict = {**nodes_dict}

        # characterise the match graph H
        # Occurence of template in the graph
        # SPMATTEO: Some assumptions to discuss
        VI = {vH for vH, vL in g.items() if vL in set(self.K.nodes)} # Occurence of context
        VHI = {vH for vH, vL in g.items() if vL not in set(self.K.nodes)} # Occurence of core template
        HI = G.subgraph(VHI) # HI is the subgraph induced by the set of nodes VHI

        # generate the substitute (sub-)graph J\I (completely detached from G)
        # Instantiate blueprint of the replacement graph
        JI, vJI_2_ptnode = self.core(HI, g, nodes_dict)

        # add the substitute (sub-)graph J\I to the main graph G
        G = nx.compose(G, JI) # G now has two connected but 'independent' subgraphs
        nodes_dict.update(vJI_2_ptnode) # Add new payloads from substitute graph

        # glue the substitute (sub-)graph J\I to the interface (sub-)graph I
        JI2RK_morphisms = Seeker(self.RK).get_morphisms(JI)
        assert len(JI2RK_morphisms) == 1
        g_JI2RK = JI2RK_morphisms[0]
        g_RK2JI = {vRK: vJI for vJI, vRK in g_JI2RK.items()}
        for vI in VI: # for each node in the interface subgraph of G
            vK = g[vI]
            G.add_edges_from({(vI, g_RK2JI[vRK]) for (_, vRK) in self.F_K2RK[vK]}) # incoming interface connections from G to substitute graph
            G.add_edges_from({(g_RK2JI[vRK], vI) for (vRK, _) in self.F_RK2K[vK]}) # outcoming interface connections from substitute graph to G
            # the new modules are fully integerized, so the precision tunnel should not embed integer numbers in floating point numbers
            # Specific to integer arithmetic transformation -> No relation to graph editing, per-se
            if nodes_dict[vI].ntype == qg.graphs.HelperOutputPrecisionTunnel.__name__:
                nodes_dict[vI] = PyTorchNode(qg.graphs.HelperOutputPrecisionTunnel(1.0))
            elif nodes_dict[vI].ntype == qg.graphs.HelperInputPrecisionTunnel.__name__:
                nodes_dict[vI] = PyTorchNode(qg.graphs.HelperInputPrecisionTunnel(1.0))
            else:
                raise TypeError  # interface nodes should be objects of class `qg.graphs.HelperPrecisionTunnel` only

        # discard the match (sub-)graph H\I
        # Assumption: removing a node also removes all arcs pointing to or from that node
        G.remove_nodes_from(set(HI.nodes))

        # Remove the payload, i.e. underying objects, accordingly
        for vHI in VHI:
            del nodes_dict[vHI]

        return G, nodes_dict

    def seek(self, G, nodes_dict):
        gs = self.seeker.get_morphisms(G)
        return gs


class FoldSTEINQConvBNSTETypeBRule(DPORule):  # w/o max pooling

    def __init__(self, gamma_int_bits=10, gamma_frac_bits=17, beta_int_bits=8, beta_frac_bits=0):

        self._gamma_int_bits  = gamma_int_bits
        self._gamma_frac_bits = gamma_frac_bits
        self._beta_int_bits   = beta_int_bits
        self._beta_frac_bits  = beta_frac_bits

        K_types = OrderedDict()
        K_types.update({'HPTout': qg.graphs.HelperOutputPrecisionTunnel.__name__})
        K_types.update({'HPTin':  qg.graphs.HelperInputPrecisionTunnel.__name__})
        K_types = OrderedDict([('/'.join(['K-term', k]), v) for k, v in K_types.items()])

        LK_types = OrderedDict()
        LK_types.update({'STEin':     qa.ste.STEActivation.__name__})
        LK_types.update({'Conv':      qa.inq.INQConv2d.__name__})
        LK_types.update({'BatchNorm': nn.BatchNorm2d.__name__})
        LK_types.update({'ReLU':      nn.ReLU.__name__})
        LK_types.update({'MaxPool':   nn.MaxPool2d.__name__})
        LK_types.update({'STEout':    qa.ste.STEActivation.__name__})
        LK_types = OrderedDict([('/'.join(['L-term', k]), v) for k, v in LK_types.items()])

        RK_types = OrderedDict()
        RK_types.update({'TWConv':   nn.Conv2d.__name__})
        RK_types.update({'XPAffine': nn.Conv2d.__name__})
        RK_types.update({'S&C':      qg.graphs.ShiftAndClip.__name__})
        RK_types.update({'MaxPool':  nn.MaxPool2d.__name__})
        RK_types = OrderedDict([('/'.join(['R-term', k]), v) for k, v in RK_types.items()])

        K_node_IDs  = list(K_types.keys())
        LK_node_IDs = list(LK_types.keys())
        RK_node_IDs = list(RK_types.keys())

        # define the template graph L [L-term]
        L_node_IDs = [K_node_IDs[0]] + LK_node_IDs + [K_node_IDs[-1]]
        self.L = nx.DiGraph()
        self.L.add_edges_from({(u, v) for u, v in zip(L_node_IDs[:-1], L_node_IDs[1:])})
        nx.set_node_attributes(self.L, {vL: __KERNEL_PARTITION__ for vL in set(self.L.nodes)}, 'bipartite')
        nx.set_node_attributes(self.L, {**K_types, **LK_types}, 'type')

        # define the context (sub-)graph K [K-term]
        VK = set(K_node_IDs)  # precision tunnel nodes define the context graph
        self.K = self.L.subgraph(VK)

        # define the template (sub-)graph L\K
        VLK = set(self.L.nodes).difference(set(self.K.nodes))
        self.LK = self.L.subgraph(VLK)

        # define the replacement (sub-)graph R\K ["gluing" R\K to K yields the graph R, i.e., the R-term]
        self.RK = nx.DiGraph()
        self.RK.add_edges_from({(u, v) for u, v in zip(RK_node_IDs[:-1], RK_node_IDs[1:])})
        nx.set_node_attributes(self.RK, {vRK: __KERNEL_PARTITION__ for vRK in set(self.RK.nodes)}, 'bipartite')
        nx.set_node_attributes(self.RK, RK_types, 'type')

        # define the arcs that go from the vertices of K to those of R\K, and viceversa
        E_K2RK = {(K_node_IDs[0], RK_node_IDs[0])}
        E_RK2K = {(RK_node_IDs[-1], K_node_IDs[-1])}
        E_K2RK2K = E_K2RK | E_RK2K
        # disintegrate `E_K2RK` and `E_RK2K` along fibres to speed up rule application
        self.F_K2RK = {vK: set(arc for arc in E_K2RK if arc[0] == vK) for vK in set(self.K.nodes)}
        self.F_RK2K = {vK: set(arc for arc in E_RK2K if arc[1] == vK) for vK in set(self.K.nodes)}

        # # glue together the (sub-)graphs L\K and R\K along the vertices of K
        # self.S = nx.compose(self.L, self.RK)
        # self.S.add_edges_from(E_K2RK2K)

        # since the GRR's L-term has been modified, rebuild the seeker
        self.seeker = Seeker(self.L)

        # this machinery can generate always-new identifiers for different rule applications
        self._counter = itertools.count()

    def _get_rule_count(self):
        rule_count = ''.join(['FINQBNSTETB', __NODE_ID_FORMAT__.format(next(self._counter))])
        return rule_count

    def core(self, HI, g, nodes_dict):

        # generate the substitute (sub-)graph J\I
        rule_count = self._get_rule_count()
        g_RK2JI = {vRK: '_'.join([rule_count, vRK.replace('R-term/', '')]) for vRK in set(self.RK.nodes)}
        JI = nx.relabel_nodes(self.RK, g_RK2JI, copy=True)

        # get pointers to the old modules;
        # these pointers will enable two actions:
        #   1. extracting the arguments required to perform the folding
        #   2. extracting the parameters to instantiate the new modules
        g_L2H = {vL: vH for vH, vL in g.items()}
        mstein  = nodes_dict[g_L2H['/'.join(['L-term', 'STEin'])]].nobj
        minq2d  = nodes_dict[g_L2H['/'.join(['L-term', 'Conv'])]].nobj
        mbn2d   = nodes_dict[g_L2H['/'.join(['L-term', 'BatchNorm'])]].nobj
        msteout = nodes_dict[g_L2H['/'.join(['L-term', 'STEout'])]].nobj
        mmxpold = nodes_dict[g_L2H['/'.join(['L-term', 'MaxPool'])]].nobj

        # fold
        weight, gamma, beta = foldsteinqconvbnste(mstein.num_levels, mstein.abs_max_value,
                                                  minq2d.weight_frozen,
                                                  mbn2d.running_mean, mbn2d.running_var, mbn2d.eps, mbn2d.weight, mbn2d.bias,
                                                  msteout.num_levels, msteout.abs_max_value,
                                                  gamma_int_bits=self._gamma_int_bits, gamma_frac_bits=self._gamma_frac_bits,
                                                  beta_int_bits=self._beta_int_bits, beta_frac_bits=self._beta_frac_bits)

        # build the new modules
        mtwconv = nn.Conv2d(minq2d.in_channels, minq2d.out_channels, minq2d.kernel_size,
                            stride=minq2d.stride, padding=minq2d.padding, dilation=minq2d.dilation, groups=minq2d.groups,
                            bias=minq2d.bias is not None).to(torch.device('cpu'))
        mtwconv.weight.data = weight

        mxpaffine = nn.Conv2d(minq2d.out_channels, minq2d.out_channels, 1,
                              stride=1, padding=0, groups=minq2d.out_channels,
                              bias=True).to(torch.device('cpu'))
        mxpaffine.weight.data = gamma
        mxpaffine.bias.data   = beta

        msandc = qg.graphs.ShiftAndClip(n_bits=math.ceil(math.log(msteout.num_levels, 2)),
                                        shift=self._gamma_frac_bits,
                                        signed=True, only_positive=True).to(torch.device('cpu'))

        mmxpnew = nn.MaxPool2d(kernel_size=mmxpold.kernel_size, stride=mmxpold.stride, padding=mmxpold.padding)

        # register the newly created nodes
        vJI_2_ptnode = {}
        vJI_2_ptnode[g_RK2JI['/'.join(['R-term', 'TWConv'])]]   = PyTorchNode(mtwconv)
        vJI_2_ptnode[g_RK2JI['/'.join(['R-term', 'XPAffine'])]] = PyTorchNode(mxpaffine)
        vJI_2_ptnode[g_RK2JI['/'.join(['R-term', 'S&C'])]]      = PyTorchNode(msandc)
        vJI_2_ptnode[g_RK2JI['/'.join(['R-term', 'MaxPool'])]]  = PyTorchNode(mmxpnew)

        return JI, vJI_2_ptnode

    def apply(self, G, nodes_dict, g):

        # create new containers
        G = G.copy()
        nodes_dict = {**nodes_dict}

        # characterise the match graph H
        VI = {vH for vH, vL in g.items() if vL in set(self.K.nodes)}
        VHI = {vH for vH, vL in g.items() if vL not in set(self.K.nodes)}
        HI = G.subgraph(VHI)

        # generate the substitute (sub-)graph J\I
        JI, vJI_2_ptnode = self.core(HI, g, nodes_dict)

        # add the substitute (sub-)graph J\I to the main graph G
        G = nx.compose(G, JI)
        nodes_dict.update(vJI_2_ptnode)

        # glue the substitute (sub-)graph J\I to the interface (sub-)graph I
        JI2RK_morphisms = Seeker(self.RK).get_morphisms(JI)
        assert len(JI2RK_morphisms) == 1
        g_JI2RK = JI2RK_morphisms[0]
        g_RK2JI = {vRK: vJI for vJI, vRK in g_JI2RK.items()}
        for vI in VI:
            vK = g[vI]
            G.add_edges_from({(vI, g_RK2JI[vRK]) for (_, vRK) in self.F_K2RK[vK]})
            G.add_edges_from({(g_RK2JI[vRK], vI) for (vRK, _) in self.F_RK2K[vK]})
            # the new modules are fully integerized, so the precision tunnel should not embed integer numbers in floating point numbers
            if nodes_dict[vI].ntype == qg.graphs.HelperOutputPrecisionTunnel.__name__:
                nodes_dict[vI] = PyTorchNode(qg.graphs.HelperOutputPrecisionTunnel(1.0))
            elif nodes_dict[vI].ntype == qg.graphs.HelperInputPrecisionTunnel.__name__:
                nodes_dict[vI] = PyTorchNode(qg.graphs.HelperInputPrecisionTunnel(1.0))
            else:
                raise TypeError  # interface nodes should be objects of class `qg.graphs.HelperPrecisionTunnel` only

        # discard the match (sub-)graph H\I
        G.remove_nodes_from(set(HI.nodes))
        for vHI in VHI:
            del nodes_dict[vHI]

        return G, nodes_dict

    def seek(self, G, nodes_dict):
        gs = self.seeker.get_morphisms(G)
        return gs


class FoldConvBNSTERule(DPORule):

    def __init__(self):

        K_types = OrderedDict()
        K_types.update({'HI': qg.graphs.HelperInput.__name__})
        K_types.update({'HPTin':  qg.graphs.HelperInputPrecisionTunnel.__name__})
        K_types = OrderedDict([('/'.join(['K-term', k]), v) for k, v in K_types.items()])

        LK_types = OrderedDict()
        LK_types.update({'Conv':      nn.Conv2d.__name__})
        LK_types.update({'BatchNorm': nn.BatchNorm2d.__name__})
        LK_types.update({'ReLU':      nn.ReLU.__name__})
        LK_types.update({'STE':       qa.ste.STEActivation.__name__})
        LK_types = OrderedDict([('/'.join(['L-term', k]), v) for k, v in LK_types.items()])

        RK_types = OrderedDict()
        RK_types.update({'Conv': nn.Conv2d.__name__})
        RK_types.update({'F&C':  qg.graphs.FloorAndClip.__name__})
        RK_types = OrderedDict([('/'.join(['R-term', k]), v) for k, v in RK_types.items()])

        K_node_IDs  = list(K_types.keys())
        LK_node_IDs = list(LK_types.keys())
        RK_node_IDs = list(RK_types.keys())

        # define the template graph L [L-term]
        L_node_IDs = [K_node_IDs[0]] + LK_node_IDs + [K_node_IDs[-1]]
        self.L = nx.DiGraph()
        self.L.add_edges_from({(u, v) for u, v in zip(L_node_IDs[:-1], L_node_IDs[1:])})
        nx.set_node_attributes(self.L, {vL: __KERNEL_PARTITION__ for vL in set(self.L.nodes)}, 'bipartite')
        nx.set_node_attributes(self.L, {**K_types, **LK_types}, 'type')

        # define the context (sub-)graph K [K-term]
        VK = set(K_node_IDs)  # precision tunnel nodes define the context graph
        self.K = self.L.subgraph(VK)

        # define the template (sub-)graph L\K
        VLK = set(self.L.nodes).difference(set(self.K.nodes))
        self.LK = self.L.subgraph(VLK)

        # define the replacement (sub-)graph R\K ["gluing" R\K to K yields the graph R, i.e., the R-term]
        self.RK = nx.DiGraph()
        self.RK.add_edges_from({(u, v) for u, v in zip(RK_node_IDs[:-1], RK_node_IDs[1:])})
        nx.set_node_attributes(self.RK, {vRK: __KERNEL_PARTITION__ for vRK in set(self.RK.nodes)}, 'bipartite')
        nx.set_node_attributes(self.RK, RK_types, 'type')

        # define the arcs that go from the vertices of K to those of R\K, and viceversa
        E_K2RK = {(K_node_IDs[0], RK_node_IDs[0])}
        E_RK2K = {(RK_node_IDs[-1], K_node_IDs[-1])}
        E_K2RK2K = E_K2RK | E_RK2K
        # disintegrate `E_K2RK` and `E_RK2K` along fibres to speed up rule application
        self.F_K2RK = {vK: set(arc for arc in E_K2RK if arc[0] == vK) for vK in set(self.K.nodes)}
        self.F_RK2K = {vK: set(arc for arc in E_RK2K if arc[1] == vK) for vK in set(self.K.nodes)}

        # # glue together the (sub-)graphs L\K and R\K along the vertices of K
        # self.S = nx.compose(self.L, self.RK)
        # self.S.add_edges_from(E_K2RK2K)

        # since the GRR's L-term has been modified, rebuild the seeker
        self.seeker = Seeker(self.L)

        # this machinery can generate always-new identifiers for different rule applications
        self._counter = itertools.count()

    def _get_rule_count(self):
        rule_count = ''.join(['FCBNSTE', __NODE_ID_FORMAT__.format(next(self._counter))])
        return rule_count

    def core(self, HI, g, nodes_dict):

        # generate the substitute (sub-)graph J\I
        rule_count = self._get_rule_count()
        g_RK2JI = {vRK: '_'.join([rule_count, vRK.replace('R-term/', '')]) for vRK in set(self.RK.nodes)}
        JI = nx.relabel_nodes(self.RK, g_RK2JI, copy=True)

        # get pointers to the old modules;
        # these pointers will enable two actions:
        #   1. extracting the arguments required to perform the folding
        #   2. extracting the parameters to instantiate the new modules
        g_L2H = {vL: vH for vH, vL in g.items()}
        mconvold = nodes_dict[g_L2H['/'.join(['L-term', 'Conv'])]].nobj
        mbn2d    = nodes_dict[g_L2H['/'.join(['L-term', 'BatchNorm'])]].nobj
        mste     = nodes_dict[g_L2H['/'.join(['L-term', 'STE'])]].nobj

        # fold
        weight, bias = foldconvbnste(mconvold.weight,
                                     mbn2d.running_mean, mbn2d.running_var, mbn2d.eps, mbn2d.weight, mbn2d.bias,
                                     mste.num_levels, mste.abs_max_value)

        # build the new modules
        mconvnew = nn.Conv2d(mconvold.in_channels, mconvold.out_channels, mconvold.kernel_size,
                             stride=mconvold.stride, padding=mconvold.padding, dilation=mconvold.dilation, groups=mconvold.groups,
                             bias=True).to(torch.device('cpu'))
        mconvnew.weight.data = weight
        mconvnew.bias.data   = bias

        mfandc = qg.graphs.FloorAndClip(n_bits=math.ceil(math.log(mste.num_levels, 2)),
                                        signed=True, only_positive=True).to(torch.device('cpu'))

        # register the newly created nodes
        vJI_2_ptnode = {}
        vJI_2_ptnode[g_RK2JI['/'.join(['R-term', 'Conv'])]]   = PyTorchNode(mconvnew)
        vJI_2_ptnode[g_RK2JI['/'.join(['R-term', 'F&C'])]]      = PyTorchNode(mfandc)

        return JI, vJI_2_ptnode

    def apply(self, G, nodes_dict, g):

        # create new containers
        G = G.copy()
        nodes_dict = {**nodes_dict}

        # characterise the match graph H
        VI = {vH for vH, vL in g.items() if vL in set(self.K.nodes)}
        VHI = {vH for vH, vL in g.items() if vL not in set(self.K.nodes)}
        HI = G.subgraph(VHI)

        # generate the substitute (sub-)graph J\I
        JI, vJI_2_ptnode = self.core(HI, g, nodes_dict)

        # add the substitute (sub-)graph J\I to the main graph G
        G = nx.compose(G, JI)
        nodes_dict.update(vJI_2_ptnode)

        # glue the substitute (sub-)graph J\I to the interface (sub-)graph I
        JI2RK_morphisms = Seeker(self.RK).get_morphisms(JI)
        assert len(JI2RK_morphisms) == 1
        g_JI2RK = JI2RK_morphisms[0]
        g_RK2JI = {vRK: vJI for vJI, vRK in g_JI2RK.items()}
        for vI in VI:
            vK = g[vI]
            G.add_edges_from({(vI, g_RK2JI[vRK]) for (_, vRK) in self.F_K2RK[vK]})
            G.add_edges_from({(g_RK2JI[vRK], vI) for (vRK, _) in self.F_RK2K[vK]})
            # the new modules are fully integerized, so the precision tunnel should not embed integer numbers in floating point numbers
            if nodes_dict[vI].ntype == qg.graphs.HelperInput.__name__:
                pass
            elif nodes_dict[vI].ntype == qg.graphs.HelperInputPrecisionTunnel.__name__:
                nodes_dict[vI] = PyTorchNode(qg.graphs.HelperInputPrecisionTunnel(1.0))
            else:
                raise TypeError  # interface nodes should be objects of class `qg.graphs.HelperPrecisionTunnel` only

        # discard the match (sub-)graph H\I
        G.remove_nodes_from(set(HI.nodes))
        for vHI in VHI:
            del nodes_dict[vHI]

        return G, nodes_dict

    def seek(self, G, nodes_dict):
        gs = self.seeker.get_morphisms(G)
        return gs


class FoldSTEINQConvBNRule(DPORule):

    def __init__(self):

        K_types = OrderedDict()
        K_types.update({'HI':       qg.graphs.HelperOutputPrecisionTunnel.__name__})
        K_types.update({'MaxPool':  nn.MaxPool2d.__name__})
        K_types = OrderedDict([('/'.join(['K-term', k]), v) for k, v in K_types.items()])

        LK_types = OrderedDict()
        LK_types.update({'STE':       qa.ste.STEActivation.__name__})
        LK_types.update({'INQConv':   qa.inq.INQConv2d.__name__})
        LK_types.update({'BatchNorm': nn.BatchNorm2d.__name__})
        LK_types.update({'ReLU':      nn.ReLU.__name__})
        LK_types = OrderedDict([('/'.join(['L-term', k]), v) for k, v in LK_types.items()])

        RK_types = OrderedDict()
        RK_types.update({'Conv': nn.Conv2d.__name__})
        RK_types.update({'ReLU': nn.ReLU.__name__})
        RK_types = OrderedDict([('/'.join(['R-term', k]), v) for k, v in RK_types.items()])

        K_node_IDs  = list(K_types.keys())
        LK_node_IDs = list(LK_types.keys())
        RK_node_IDs = list(RK_types.keys())

        # define the template graph L [L-term]
        L_node_IDs = [K_node_IDs[0]] + LK_node_IDs + [K_node_IDs[-1]]
        self.L = nx.DiGraph()
        self.L.add_edges_from({(u, v) for u, v in zip(L_node_IDs[:-1], L_node_IDs[1:])})
        nx.set_node_attributes(self.L, {vL: __KERNEL_PARTITION__ for vL in set(self.L.nodes)}, 'bipartite')
        nx.set_node_attributes(self.L, {**K_types, **LK_types}, 'type')

        # define the context (sub-)graph K [K-term]
        VK = set(K_node_IDs)  # precision tunnel nodes define the context graph
        self.K = self.L.subgraph(VK)

        # define the template (sub-)graph L\K
        VLK = set(self.L.nodes).difference(set(self.K.nodes))
        self.LK = self.L.subgraph(VLK)

        # define the replacement (sub-)graph R\K ["gluing" R\K to K yields the graph R, i.e., the R-term]
        self.RK = nx.DiGraph()
        self.RK.add_edges_from({(u, v) for u, v in zip(RK_node_IDs[:-1], RK_node_IDs[1:])})
        nx.set_node_attributes(self.RK, {vRK: __KERNEL_PARTITION__ for vRK in set(self.RK.nodes)}, 'bipartite')
        nx.set_node_attributes(self.RK, RK_types, 'type')

        # define the arcs that go from the vertices of K to those of R\K, and viceversa
        E_K2RK = {(K_node_IDs[0], RK_node_IDs[0])}
        E_RK2K = {(RK_node_IDs[-1], K_node_IDs[-1])}
        E_K2RK2K = E_K2RK | E_RK2K
        # disintegrate `E_K2RK` and `E_RK2K` along fibres to speed up rule application
        self.F_K2RK = {vK: set(arc for arc in E_K2RK if arc[0] == vK) for vK in set(self.K.nodes)}
        self.F_RK2K = {vK: set(arc for arc in E_RK2K if arc[1] == vK) for vK in set(self.K.nodes)}

        # # glue together the (sub-)graphs L\K and R\K along the vertices of K
        # self.S = nx.compose(self.L, self.RK)
        # self.S.add_edges_from(E_K2RK2K)

        # since the GRR's L-term has been modified, rebuild the seeker
        self.seeker = Seeker(self.L)

        # this machinery can generate always-new identifiers for different rule applications
        self._counter = itertools.count()

    def _get_rule_count(self):
        rule_count = ''.join(['FSTEINQBN', __NODE_ID_FORMAT__.format(next(self._counter))])
        return rule_count

    def core(self, HI, g, nodes_dict):

        # generate the substitute (sub-)graph J\I
        rule_count = self._get_rule_count()
        g_RK2JI = {vRK: '_'.join([rule_count, vRK.replace('R-term/', '')]) for vRK in set(self.RK.nodes)}
        JI = nx.relabel_nodes(self.RK, g_RK2JI, copy=True)

        # get pointers to the old modules;
        # these pointers will enable two actions:
        #   1. extracting the arguments required to perform the folding
        #   2. extracting the parameters to instantiate the new modules
        g_L2H = {vL: vH for vH, vL in g.items()}
        mste     = nodes_dict[g_L2H['/'.join(['L-term', 'STE'])]].nobj
        minq2d   = nodes_dict[g_L2H['/'.join(['L-term', 'INQConv'])]].nobj
        mbn2d    = nodes_dict[g_L2H['/'.join(['L-term', 'BatchNorm'])]].nobj
        mreluold = nodes_dict[g_L2H['/'.join(['L-term', 'ReLU'])]].nobj

        # fold
        weight, bias = foldsteinqconvbn(mste.num_levels, mste.abs_max_value,
                                        minq2d.weight_frozen,
                                        mbn2d.running_mean, mbn2d.running_var, mbn2d.eps, mbn2d.weight, mbn2d.bias)

        # build the new modules
        mconv = nn.Conv2d(minq2d.in_channels, minq2d.out_channels, minq2d.kernel_size,
                          stride=minq2d.stride, padding=minq2d.padding, dilation=minq2d.dilation, groups=minq2d.groups,
                          bias=True).to(torch.device('cpu'))
        mconv.weight.data = weight
        mconv.bias.data   = bias

        mrelunew = nn.ReLU(inplace=True)

        # register the newly created nodes
        vJI_2_ptnode = {}
        vJI_2_ptnode[g_RK2JI['/'.join(['R-term', 'Conv'])]] = PyTorchNode(mconv)
        vJI_2_ptnode[g_RK2JI['/'.join(['R-term', 'ReLU'])]] = PyTorchNode(mrelunew)

        return JI, vJI_2_ptnode

    def apply(self, G, nodes_dict, g):

        # create new containers
        G = G.copy()
        nodes_dict = {**nodes_dict}

        # characterise the match graph H
        VI = {vH for vH, vL in g.items() if vL in set(self.K.nodes)}
        VHI = {vH for vH, vL in g.items() if vL not in set(self.K.nodes)}
        HI = G.subgraph(VHI)

        # generate the substitute (sub-)graph J\I
        JI, vJI_2_ptnode = self.core(HI, g, nodes_dict)

        # add the substitute (sub-)graph J\I to the main graph G
        G = nx.compose(G, JI)
        nodes_dict.update(vJI_2_ptnode)

        # glue the substitute (sub-)graph J\I to the interface (sub-)graph I
        JI2RK_morphisms = Seeker(self.RK).get_morphisms(JI)
        assert len(JI2RK_morphisms) == 1
        g_JI2RK = JI2RK_morphisms[0]
        g_RK2JI = {vRK: vJI for vJI, vRK in g_JI2RK.items()}
        for vI in VI:
            vK = g[vI]
            G.add_edges_from({(vI, g_RK2JI[vRK]) for (_, vRK) in self.F_K2RK[vK]})
            G.add_edges_from({(g_RK2JI[vRK], vI) for (vRK, _) in self.F_RK2K[vK]})
            # the new modules are fully integerized, so the precision tunnel should not embed integer numbers in floating point numbers
            if nodes_dict[vI].ntype == qg.graphs.HelperOutputPrecisionTunnel.__name__:
                nodes_dict[vI] = PyTorchNode(qg.graphs.HelperOutputPrecisionTunnel(1.0))
            elif nodes_dict[vI].ntype == nn.MaxPool2d.__name__:
                pass
            else:
                raise TypeError  # interface nodes should be objects of class `qg.graphs.HelperPrecisionTunnel` only

        # discard the match (sub-)graph H\I
        G.remove_nodes_from(set(HI.nodes))
        for vHI in VHI:
            del nodes_dict[vHI]

        return G, nodes_dict

    def seek(self, G, nodes_dict):
        gs = self.seeker.get_morphisms(G)
        return gs
