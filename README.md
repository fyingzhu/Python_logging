# Python_logging
Python logging模块使用配置文件记录日志

## 一、功能说明：
    - 1、以时间命名日志，默认为按每天的日期命名，可配置；
    - 2、设置日志级别开关，可配置；


## 二、使用说明：

    工程中我们使用一个名字为config.py的Python模块用来保存全局的配置，
    由于logging在工程中每个源代码文件都可能用到，
    因此我们把logging模块在config.py中生成一个实例，
    这样其它模块只需要引用这个实例就可以了。
    在其它模块中，我们使用这样的语句引用logger对象：
    
    ```from config import logger```

    需要记录日志的时候，只需要使用logger.error()，logger.debug()类似的语句就好了。

> 注意：logging模块是线程安全的。


## 三、配置文件logging.conf说明：

[loggers]
cisdi 为一个实例，可自行修改。

[handler_timedrt]
    按时间回滚记录日志。
    args=('log/cisdi' + '.log', 'midnight', 1, 0) 解析如下：

函数：
    TimedRotatingFileHandler(filename, when='h', interval=1, backupCount=0, encoding=None, delay=False, utc=False)
参数说明：
    when:按照哪种时间单位滚动（可选S-按秒，M-按分钟，H-按小时，D-按天，W0-W6-按指定的星期几，midnight-在凌晨）
    interval=t： 表示间隔时间为t
    backupCount=n，表示保存最近n份记录，0表示不删除历史。

[formatters] 
    定义输出格式。
