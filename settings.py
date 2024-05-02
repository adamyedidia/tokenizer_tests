import logging
import os

logger = logging.getLogger(__name__)

OPENAI_SECRET_KEY = ""

from local_settings import *

if not os.getenv('OPENAI_API_KEY') and OPENAI_SECRET_KEY:
    logger.info('Setting OpenAI API key...')
    os.environ['OPENAI_API_KEY'] = OPENAI_SECRET_KEY
else:
    logger.warn('No OpenAI API key found. Please set OPENAI_SECRET_KEY in local_settings.py')
