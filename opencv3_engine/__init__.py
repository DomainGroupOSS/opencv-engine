__author__ = 'konstantin.burov'
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

__version__ = '3.0.5'

try:
    from opencv3_engine.engine import Engine  # NOQA
except ImportError:
    logging.exception('Could not import opencv_engine. Probably due to setup.py installing it.')
