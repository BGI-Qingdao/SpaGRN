#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 21 Mar 2023 15:41
# @Author: Yao LI

import logging


class GetLogger:
    def __init__(self, path, clevel = logging.DEBUG, Flevel = logging.DEBUG):
        self.logger = logging.getLogger('spaGRN')
        self.logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "[%(asctime)s][%(name)s][%(process)d][%(thread)d][%(module)s][%(lineno)d][%(levelname)s]: %(message)s",
            '%Y-%m-%d %H:%M:%S')
        # set CMD logging
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(clevel)
        # set file logging
        fh = logging.FileHandler(path)
        fh.setFormatter(fmt)
        fh.setLevel(Flevel)
        self.logger.addHandler(sh)
        self.logger.addHandler(fh)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def cri(self, message):
        self.logger.critical(message)


# logger = GetLogger('spaGRN.log', logging.ERROR, logging.DEBUG)
