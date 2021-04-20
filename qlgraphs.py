import torch

import quantlib.graphs as qg


loader = qg.edit.Loader('ImageNet', 'VGG', {})

onnxg = qg.graphs.ONNXGraph(loader.net, torch.ones(1, 3, 224, 224).to('cuda'))
onnxe = qg.editor.Editor(onnxg)
onnxe.startup()
for mod_name, rho in qg.grrules.load_rescoping_rules(modules=['AdaptiveAvgPool2d', 'ViewFlattenNd', 'INQConv2d', 'STEActivation']).items():
    print("Applying rule {} to `nn.Module`s of type {} ...".format(type(rho), mod_name))
    onnxe.set_grr(rho)
    onnxe.edit()
onnxe.shutdown()

pytorchg = qg.graphs.PyTorchGraph(loader.net, onnxg)
pytorche = qg.editor.Editor(pytorchg, onlykernel=True)
pytorche.startup()
# add helper nodes
pytorche.set_grr(qg.grrules.AddInputNodeRule())
pytorche.edit(gs=pytorche.seek(VIs=[{'O000000'}]))
pytorche.set_grr(qg.grrules.AddOutputNodeRule())
pytorche.edit(gs=pytorche.seek(VIs=[{'O000074'}]))
pytorche.set_grr(qg.grrules.AddPrecisionTunnelRule('STEActivation'))
pytorche.edit()
# # remove helper nodes
# pytorche.set_grr(qg.grrules.RemovePrecisionTunnelRule())
# pytorche.edit()
# pytorche.set_grr(qg.grrules.RemoveOutputNodeRule())
# pytorche.edit()
# pytorche.set_grr(qg.grrules.RemoveInputNodeRule())
# pytorche.edit()
