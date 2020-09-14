# -*- coding: utf-8 -*- 
"""
Project: bert_distill
Creator: shijundai
Create time: 2020-09-14 17:31
IDE: PyCharm
Introduction:
"""
import logging
from logging.handlers import TimedRotatingFileHandler

def LogInit(name):
    LOG_FILE = 'log/{}.all.log'.format(name)
    ERROR_LOG_FILE = 'log/ClickBaitONNX.err.log'
    LOG_FORMAT_STR = '%(asctime)s.%(msecs)03d %(levelname)s %(process)d : %(message)s'
    LOG_FORMAT = logging.Formatter(LOG_FORMAT_STR)

    my_logger = logging.getLogger("all")
    my_logger.setLevel(level=logging.INFO)

    handler = TimedRotatingFileHandler(LOG_FILE, when='D', interval=1, backupCount=7)
    handler.setLevel(logging.INFO)
    handler.setFormatter(LOG_FORMAT)
    my_logger.addHandler(handler)

    # handler1 = logging.FileHandler(ERROR_LOG_FILE)
    handler1 = TimedRotatingFileHandler(ERROR_LOG_FILE, when='D', interval=1, backupCount=7)
    handler1.setLevel(logging.ERROR)
    handler1.setFormatter(LOG_FORMAT)
    my_logger.addHandler(handler1)
    return my_logger