import datetime
import json
import os
import re
import ast
import math


def load_logs(filepath):
    """Loads all lines from log file into list of dicts"""

    #logs_array = []
    with open(filepath, 'r') as handle:
        lines = handle.readlines()
        for log in lines:
            #if not re.match(r'^\s*$', log):
            #log = log.replace("'", "\"")
            #log = log.replace("False", "\"False\"")
            #log = log.replace("True", "\"True\"")
            log = ast.literal_eval(log)
            #logs_array.append(log)
    return log


def update_log(logfile, log, update):
    with open(logfile, 'w') as handle:
        handle.write('{}\n'.format({**log, **update}))
        """
        for log in logs_array:
            
            if log['id'] == id:
                #print({**log, **update})
                handle.write('{}\n'.format({**log, **update}))
            else:
                handle.write('{}\n'.format(log))
        """


def check_if_log_exists(filepath):
    exists = False
    if (os.path.isfile(filepath) is True):
        exists = True
    return exists


def create_log(func):
    def wrapper(*args, **kwargs):
        
        logfile = './logs/log_{}'.format(args[1]['id'])
        result = func(*args, **kwargs)
        exists = check_if_log_exists(logfile)
        if exists is False:
            with open(logfile, 'w') as handle:
                today = datetime.datetime.now()
                args[1]['date'] = '{}-{}-{} {}:{}'.format(today.day, today.month, today.year,  today.hour,  today.minute)
                handle.write('{}\n'.format(args[1]))
        # else:
        #    raise ValueError('Log already exists')
        
        return result

    return wrapper


def elapsedtime_log(func):
    def wrapper(*args, **kwargs):
        
        #logs_array = load_logs()
        logfile = './logs/log_{}'.format(args[0].opts['id'])
        exists = check_if_log_exists(logfile)
        if exists is False:
            raise ValueError('Log ID does not exist.')
        
        time1 = datetime.datetime.now()
        result = func(*args, **kwargs)
        time2 = datetime.datetime.now()
        elapsed_time = time2 - time1
        minutes = divmod(elapsed_time.total_seconds(), 60)[0]
        
        log = load_logs(logfile)
        update = {func.__name__+'-minutes': minutes}
        update_log(logfile, log, update)
        #with open(logfile, 'w') as handle:
        #    args[0].opts[func.__name__+'-minutes'] = minutes
        #    handle.write(str(args[0].opts)+'\n')
        return result

    return wrapper


def update_result_log(func):
    def wrapper(*args, **kwargs):
        
        #logs_array = load_logs()
        logfile = './logs/log_{}'.format(args[0].opts['id'])
        exists = check_if_log_exists(logfile)
        if exists is False:
            raise ValueError('Log ID does not exist.')
        
        log = load_logs(logfile)
        
        result = func(*args, **kwargs)  
        result = float(result)
        if math.isnan(result) is True:
            result = 'nan'
        update = {args[2]: result}
        update_log(logfile, log, update)
        
        return result

    return wrapper