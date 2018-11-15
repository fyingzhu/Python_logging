# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:48:19 2018

@author: 017253
"""

from config import logger

def sum_(a,b):
    logger.debug('debug message')
    logger.info('info message')
    logger.warn('warn message')
    logger.error('error message')
    logger.critical('critical message')
    return a+b

sum_(1,2)