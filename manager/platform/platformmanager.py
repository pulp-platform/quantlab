# 
# platformmanager.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2021 ETH Zurich.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

import socket
import multiprocessing
import os
import torch

from manager import QUANTLAB_PREFIX


__all__ = [
    'PlatformManager',
]


_MASTER_PROC_RANK = 0

_CPUS_CAP         = 2  # maximum 2 - 1 "helper" processes can be used to pre-process data points on CPU-only machine (the "main" process should anyway not be able to keep up with data preparation)
_CPUS_PER_GPU_CAP = 5  # maximum 5 - 1 "helper" processes for each GPU can be used to pre-process data points


class PlatformManager(object):

    def __init__(self):
        """The entity that manages the engineering aspects of a ML experiment.

        As much as it is useful to think to machine learning systems in purely
        functional terms, training them efficiently requires to consider some
        details about the platform supporting the computation. These details
        include hardware, e.g., whether the machine is equipped with both CPUs
        and GPUs or just CPUs, and operating system information; possibly,
        also the run-time properties of concurrent computing paradigms such as
        the message passing interface (MPI) should be considered.

        Deep neural networks are intrinsically parallel both at the
        intra-neuron-level (different synapsis of a neuron can weight their
        respective inputs in parallel) and at the intra-layer-level (multiple
        neurons can perform their computations in parallel). Sometimes, DNNs
        exhibit concurrency even at the network-level; for instance, in
        ResNets, the layers of different branches in the same layer block can
        be executed concurrently. DNNs also have data-independent control
        flows, and the mini-batch stochastic gradient descent optimisation
        algorithm allows to compute learning signals (i.e., gradients) with
        respect to multiple points in parallel. These properties make DNNs
        ideal fits for vector processors, parallel processors, and even for
        distributed computing systems. Given the highly symmetric nature of
        the computations that they entail, a simple but effective way to
        accelerate inference and training of DNNs is exploiting all these
        opportunities of *data parallelism*.

        Apart from single-CPU and single-GPU flows, QuantLab supports several
        *data parallel* computing configurations to speed up your experiments;
        note, though, that QuantLab does NOT support *model parallelism*.
        QuantLab can support both single-process and multi-process
        experiments. In the case of single-process experiments, QuantLab
        supports single-CPU, single-GPU, and multi-GPU experiments. In the
        case of multi-process experiments, QuantLab supports both single-node
        and multi-node (i.e., distributed) runs; in both cases, each process
        can run on either single-CPU or single-GPU, but not on multiple GPUs.
        Multi-process (and therefore possibly distributed) experiments are
        supported via Horovod; if Horovod is not installed on your platform,
        then QuantLab can only run single-process (and therefore necessarily
        non-distributed) experiments. If Horovod is installed on your
        platform, QuantLab can run single-process experiments both without
        Horovod support (in single-CPU, single-GPU, and multi-GPU settings)
        and with Horovod support (in single-CPU and single-GPU mode only);
        this last possibility is not recommended, though.

        Since data parallel computations can be trivially load-balanced (e.g.,
        by assigning the same amount of data points to each worker), QuantLab
        does NOT implement complex load balancing algorithms. Instead, it
        assumes that the hardware infrastructure is *homogeneous*, and will
        trivially assign the same workload to each process or sub-process. By
        *homogeneous* we mean the following:
          * in single-process, multi-GPU configurations we assume that all the
            GPUs are of the same model;
          * in multi-process configurations we assume that all the processors
            and the GPUs on the underlying computing nodes are of the same
            model; we also assume that each process has equally fast access to
            the disk (so that the data can be fetched at the same speed).

        To summarise, the following scenarios are supported:
          1. single-process, single-node, single-CPU, w/o Horovod (laptop/PC)                     - almost surely the slowest configuration (deep learning w/o GPUs?);
          2. single-process, single-node, single-GPU, w/o Horovod (laptop/PC)                     - okay for small networks and data sets;
          3. single-process, single-node, multi-GPU,  w/o Horovod (workstation)                   - okay for local prototyping;
          4. single-process, single-node, single-CPU, w/  Horovod (laptop/PC/workstation/cluster) - (?) Horovod is apparently unnecessary: see scenario 1;
          5. single-process, single-node, single-GPU, w/  Horovod (laptop/PC/workstation/cluster) - (?) Horovod is apparently unnecessary: see scenario 2;
          6. multi-process,  single-node, single-CPU, w/  Horovod (laptop/PC/workstation/cluster) - likely slower than scenario 3; possibly also than scenario 2 (this depends on the GPU model);
          7. multi-process,  single-node, single-GPU, w/  Horovod (laptop/PC/workstation/cluster) - (?) Horovod is apparently unnecessary: see scenario 3;
          8. multi-process,  multi-node,  single-CPU, w/  Horovod (cluster)                       - depending on the number of nodes, might be slower than scenario 3 (this depends on the GPU model);
          9. multi-process,  multi-node,  single-GPU, w/  Horovod (cluster)                       - likely the fastest configuration.

        Attributes:
            hostname (str): the name of the machine on which the Python
                interpreter that built the ``PlatformManager`` is running; the
                *host* machine.
            host_ip (str): the host's IP address.
            n_cpus (int): the number of CPUs on the host.
            n_gpus (int): the number of GPUs on the host; this reflects the
                visibility of devices as filtered by the environment variable
                ``CUDA_VISIBLE_DEVICES``.
            device (:obj:`torch.device`): the device to which the current
                process will be pinned.
            hvd (None or :obj:`types.ModuleType`): if activated, the Horovod
                Python module (PyTorch version).
            global_rank (int): the identifier of the current process in the
                MPI communicator spawned by Horovod.
            global_size (int): the number of sibling processes in the MPI
                communicator spawned by Horovod.
            local_rank (int): the identifier of the current process with
                respect to its siblings running on the same computing node.
            local_size (int): the number of sibling processes assigned by
                Horovod to the same computing node as the current process.
            master_rank (int): the identifier of the master process; this
                information is necessary for proper synchronisation of
                multi-process runs.
            is_master (None or bool): whether the PyTorch process is the
                master process; this information is necessary for proper
                synchronisation of multi-process runs.

        """

        # get host machine information
        self.hostname = socket.gethostname()
        try:
            self.host_ip = socket.gethostbyname(self.hostname)
        except socket.gaierror:
            # When this method is created inside a Python `subprocess` (e.g.,
            # during the `configure` DoE flow) on my machine (MacBook Air,
            # Retina, 13", 2020 - macOS Big Sur 11.1), the hostname can not be
            # resolved properly:
            #
            #     https://stackoverflow.com/a/43549848
            #
            # With this `try-except` construct I can work around the issue,
            # although the reported IP won't be relevant since it will very
            # likely by `127.0.0.1`.
            self.host_ip = socket.gethostbyname('localhost')
        self.n_cpus   = multiprocessing.cpu_count()
        self.n_gpus   = torch.cuda.device_count() if torch.cuda.is_available() else 0
        assert self.n_gpus <= self.n_cpus  # max one GPU per core
        self.device   = None

        # who am I?
        self.pid = os.getpid()

        # pin the process to a device
        if self.n_gpus > 0:
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device('cpu')

        # Horovod information
        self.hvd         = None
        self.global_rank = None
        self.global_size = None
        self.local_rank  = None
        self.local_size  = None

        # master-worker information
        self.master_rank = _MASTER_PROC_RANK
        self.is_master   = None

    @property
    def is_horovod_run(self) -> bool:
        return self.hvd is not None

    @property
    def is_singleproc_horovod_run(self) -> bool:
        return self.is_horovod_run and self.global_size == 1  # using Horovod for single-process runs seems unuseful, but we accept it

    @property
    def is_multiproc_horovod_run(self) -> bool:
        return self.is_horovod_run and self.global_size > 1

    @property
    def is_distributed_run(self) -> bool:
        return self.is_multiproc_horovod_run and (self.global_size != self.local_size)

    @property
    def is_nndataparallel_run(self) -> bool:
        return (not self.is_horovod_run) and (self.n_gpus > 1)

    def startup(self, horovod: bool) -> None:
        """Compute how to optimise QuantLab runs on this platform.

        Args:
            horovod: whether to use Horovod.

        """

        # startup horovod?
        if horovod:

            import horovod.torch as hvd

            # register and start up horovod
            self.hvd = hvd
            self.hvd.init()  # why Horovod vs. PyTorch's distributed data parallel (DDP)? https://github.com/horovod/horovod/issues/1973

            # get process and MPI communicator information
            self.global_rank = self.hvd.rank()
            self.global_size = self.hvd.size()
            self.local_rank  = self.hvd.local_rank()
            self.local_size  = self.hvd.local_size()

            # identify master process
            self.is_master = self.global_rank == self.master_rank

        # which is the main device?
        if self.n_gpus > 0:
            if self.is_horovod_run:
                assert self.local_size <= self.n_gpus  # each GPU should be used at max by one process
                torch.cuda.set_device(self.local_rank)
            self.device = torch.cuda.current_device()
        else:
            if self.is_horovod_run:
                assert self.local_size <= self.n_cpus  # each CPU should be used at max one process
            self.device = torch.device('cpu')

    @property
    def num_workers(self) -> int:
        """Compute the optimal number of helper processes for pre-processing."""

        if self.n_gpus > 0:
            if self.is_horovod_run:
                effective_n_cpus = (self.local_size * self.n_cpus) // self.n_gpus
                n_cpus_per_process = min(effective_n_cpus, _CPUS_PER_GPU_CAP)
            else:  # this is a single-process (hence, single-node), multi-GPU run
                n_cpus_per_process = min(self.n_cpus, self.n_gpus * (_CPUS_PER_GPU_CAP - 1) + 1)

        else:
            if self.is_horovod_run:
                n_cpus_per_process = min(self.n_cpus // self.local_size, _CPUS_CAP)
            else:
                n_cpus_per_process = min(self.n_cpus, _CPUS_CAP)

        num_workers = n_cpus_per_process - 1  # worker processes are just "helpers" to the "main" process, to which an entire core should be reserved
        return num_workers

    @property
    def pin_memory(self) -> bool:
        return True if (self.is_horovod_run and self.n_gpus > 0) else False

    def show(self) -> None:

        message  = QUANTLAB_PREFIX + "Node:   <hostname:   {:15s}>\n".format(self.hostname)
        message += QUANTLAB_PREFIX + "        <IP address: {:15s}>\n".format(self.host_ip)
        message += QUANTLAB_PREFIX + "        #CPUs:  {:3d}\n".format(self.n_cpus)
        message += QUANTLAB_PREFIX + "        #GPUs:  {:3d}\n".format(self.n_gpus)
        message += QUANTLAB_PREFIX + "        Device: {}\n".format(self.device)
        message += QUANTLAB_PREFIX + "\n"

        if self.is_multiproc_horovod_run:
            message += QUANTLAB_PREFIX + "MPI:    <Process PID: {:6d}>\n".format(self.pid)
            message += QUANTLAB_PREFIX + "        Global rank: {:3d}/{:3d}\n".format(self.global_rank, self.global_size)
            message += QUANTLAB_PREFIX + "        Local rank:  {:3d}/{:3d}\n".format(self.local_rank, self.local_size)
            message += QUANTLAB_PREFIX + "\n"

        message += QUANTLAB_PREFIX + "DataLoader: # Workers:  {}\n".format(self.num_workers)
        message += QUANTLAB_PREFIX + "            Pin memory? {}\n".format(self.pin_memory)
        message += QUANTLAB_PREFIX

        # synched flush to console; in multi-process runs, this should avoid weird interleavings of lines output by different processes
        print(message)

