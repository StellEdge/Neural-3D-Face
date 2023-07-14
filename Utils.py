import os
import time
from datetime import datetime,date,timedelta
DATA_DIR = "./data"

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
# def xavier(param):
#     init.xavier_uniform(param)
#
#
# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         xavier(m.weight.data)
#         m.bias.data.zero_()

def timelog(s):
    print(timelog_str() + ' ' + s)

def timelog_str():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    today = date.today()
    d1 = today.strftime("%Y/%m/%d/")
    return d1+current_time

def timesave_str():
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S_")
    today = date.today()
    d1 = today.strftime("%Y_%m_%d_")
    return d1+current_time

def create_temp_dir(name=''):

    str_time = timesave_str()
    data_path = os.path.join(DATA_DIR, 'tmp')
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    tmp_path = os.path.join(data_path, 'train_' + str_time + name)
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    print('Temp data path: ' + tmp_path)
    tmp_path_images = os.path.join(tmp_path, 'images')
    if not os.path.exists(tmp_path_images):
        os.mkdir(tmp_path_images)
    print('Temp data - images path: ' + tmp_path_images)
    tmp_path_models = os.path.join(tmp_path, 'models')
    if not os.path.exists(tmp_path_models):
        os.mkdir(tmp_path_models)
    print('Temp data - images path: ' + tmp_path_models)
    return tmp_path,tmp_path_images, tmp_path_models

def get_model_size(model,name):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(name+' model size: {:.3f}MB'.format(size_all_mb))