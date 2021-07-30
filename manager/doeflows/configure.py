import importlib
import subprocess

from .experimentaldesign import ExperimentalDesignLogger


def configure(args):

    logger = ExperimentalDesignLogger(args.problem, args.topology, args.exp_design)
