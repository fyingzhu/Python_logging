# -*- coding: utf-8 -*-
"""
工程中我们使用一个名字为config.py的Python模块用来保存全局的配置，
由于logging在工程中每个源代码文件都可能用到，
因此我们把logging模块在config.py中生成一个实例，
这样其它模块只需要引用这个实例就可以了。
在其它模块中，我们使用这样的语句引用logger对象：
# from config import logger
"""

import logging
import logging.config


logging.config.fileConfig('logging.conf')
logger = logging.getLogger('cisdi')