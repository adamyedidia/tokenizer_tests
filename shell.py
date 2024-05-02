#!/usr/bin/env python3
import logging
import IPython

from token_tester import *  # noqa

__name__ = '__ipython__'
logger = logging.getLogger(__name__)

# Starts an ipython shell with access to the variables in this local scope (the imports)
IPython.start_ipython(user_ns=locals())
