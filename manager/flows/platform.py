# -*- coding: utf-8 -*-
from manager.platform import PlatformManager


def platform(args):

    platform = PlatformManager()
    platform.startup(horovod=args.horovod)

    platform.show()
