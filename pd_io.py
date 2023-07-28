# -*- encoding: utf-8 -*-
'''
@File    :   pd_io.py
@Time    :   2023/02/16 15:17:51
@Author  :   
@Version :   1.0
@Contact :   
@Desc    :   文件、目录相关函数处理
'''
import os
import re
import random
import errno
import yaml
import h5py
import numpy as np
import time
from configobj import ConfigObj
from datetime import datetime
from contextlib import contextmanager
import pdb



def get_files(path) :
    """
    获取目录下文件
    """
    files=os.listdir(path)
    files.sort()
    return files

def  check_path(path):
    """
    生成对应目录
    """
    if not os.path.exists(path):
        os.makedirs(path)

def load_file(file):
    '''
    读取文件
    '''
    with  open(file,'w') as f:
        lines=f.readlines()
    return lines

def makeYamlCfg(yaml_dict, oFile):
    # 投影程序需要的配置文件名称
    cfgPath = os.path.dirname(oFile)
    if not os.path.isdir(cfgPath):
        os.makedirs(cfgPath)
    with open(oFile, 'w') as stream:
            yaml.dump(yaml_dict, stream, default_flow_style=False)


def loadYamlCfg(iFile):

    if not os.path.isfile(iFile):
        print("No yaml: %s" % (iFile))
        return None
    try:
        with open(iFile, 'r') as stream:
            plt_cfg = yaml.load(stream)

        # check yaml
        if 'chan' in plt_cfg and 'plot' in plt_cfg and 'days' in plt_cfg:
            pass
        else:
            print ('plt yaml lack some keys')
            plt_cfg = None
    except:
        print('plt yaml not valid: %s' % iFile)
        plt_cfg = None

    return plt_cfg


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def FindFile(ipath, pat):
    '''
    # 要查找符合文件正则的文件
    '''
    FileLst = []

    if not os.path.isdir(ipath):
        return FileLst
    Lst = sorted(os.listdir(ipath), reverse=False)
    for Line in Lst:
        FullPath = os.path.join(ipath, Line)
        if os.path.isdir(FullPath):
            continue
        # 如果是文件则进行正则判断，并把符合规则的所有文件保存到List中
        elif os.path.isfile(FullPath):
            FileName = Line
            m = re.match(pat, FileName)
            if m:
                FileLst.append(FullPath)
    if len(FileLst) == 0:
        return FileLst
    else:
        return FileLst


def find_file(path, reg):
    '''
    path: 要遍历的目录
    reg: 符合条件的文件
    '''
    FileLst = []
    try:
        lst = os.walk(path)
        for root, dirs, files in lst:
            for name in files:
                try:
                    m = re.match(reg, name)
                except Exception as e:
                    continue
                if m:
                    FileLst.append(os.path.join(root, name))
    except Exception as e:
        print (str(e))

    return sorted(FileLst)


def path_replace_ymd(path, ymd):
    '''
    path:替换路径中的日期 ,path中%YYYY%MM%DD%JJJ 等关键字会被ymd日期实例
    ymd: yyyymmdd  (20180101)
    '''
    # 转成datetime类型
    ymd = datetime.strptime(ymd, '%Y%m%d')
    yy = ymd.strftime('%Y')
    mm = ymd.strftime('%m')
    dd = ymd.strftime('%d')
    jj = ymd.strftime('%j')
    path = path.replace('%YYYY', yy)
    path = path.replace('%MM', mm)
    path = path.replace('%DD', dd)
    path = path.replace('%JJJ', jj)

    return  path


def load_yaml_config(in_file):
    """
    加载 Yaml 文件
    :param in_file:
    :return: Yaml 类
    """
    if not os.path.isfile(in_file):
        print ("File is not exist: {}".format(in_file))
        return None
    try:
        with open(in_file, 'r') as stream:
            yaml_data = yaml.load(stream)
    except IOError as why:
        print (why)
        print ("Load yaml file error.")
        yaml_data = None

    return yaml_data


def is_none(*args):
    """
    判断传入的变量中是否有 None
    :param args:
    :return:
    """
    has_none = False
    for arg in args:
        if arg is None:
            has_none = True
    return has_none


def copy_attrs_h5py(pre_object, out_object):
    """
    复制 file、dataset 或者 group 的属性
    :param pre_object: 被复制属性的 dataset 或者 group
    :param out_object: 复制属性的 dataset 或者 group
    :return:
    """
    for akey in pre_object.attrs.keys():
        out_object.attrs[akey] = pre_object.attrs[akey]


def read_dataset_hdf5(file_path, set_name):
    """
    读取 hdf5 文件，返回一个 numpy 多维数组
    :param file_path: (unicode)文件路径
    :param set_name: (str or list)表的名字
    :return: 如果传入的表名字是一个字符串，返回 numpy.ndarray
             如果传入的表名字是一个列表，返回一个字典，key 是表名字，
             value 是 numpy.ndarry
    """
    if isinstance(set_name, str):
        if os.path.isfile(file_path):
            file_h5py = h5py.File(file_path, 'r')
            data = file_h5py.get(set_name)[:]
            dataset = np.array(data)
            file_h5py.close()
            return dataset
        else:
            raise ValueError('value error: file_path')
    elif isinstance(set_name, list):
        datasets = {}
        if os.path.isfile(file_path):
            file_h5py = h5py.File(file_path, 'r')
            for name in set_name:
                data = file_h5py.get(name)[:]
                dataset = np.array(data)
                datasets[name] = dataset
            file_h5py.close()
            return datasets
        else:
            raise ValueError('value error: file_path')
    else:
        raise ValueError('value error: set_name')


def attrs2dict(attrs):
    """
    将一个 HDF5 attr 类转为 Dict 类
    :return:
    """
    d = {}
    for k, v in attrs.items():
        d[k] = v
    return d


class Config(object):
    """
    加载配置文件
    """

    def __init__(self, config_file):
        """
        初始化
        """
        self.error = False

        self.config_file = config_file  # config 文件路径
        self.config_data = None  # 加载到的配置数据

        # load config file
        self.load_config_file()
        # load config data
        self.load_config_data()

    def load_config_file(self):
        """
        尝试加载配置文件
        :return:
        """
        try:
            self.load_yaml_file()
        except Exception:
            self.load_cfg_file()
        finally:
            if len(self.config_data) == 0:
                self.error = True
                print ("Load config file error: {}".format(self.config_file))

    def load_yaml_file(self):
        """
        加载 yaml 文件内容
        :return:
        """
        with open(self.config_file, 'r') as stream:
            self.config_data = yaml.load(stream)

    def load_cfg_file(self):
        """
        加载 config 文件内容
        :return:
        """
        self.config_data = ConfigObj(self.config_file)

    def load_config_data(self):
        """
        读取配置数据
        :return:
        """
        self._load_config_data(self.config_data)

    def _load_config_data(self, config_data, prefix=""):
        """
        读取配置数据，动态创建属性值
        :return:
        """
        if self.error:
            return
        for k in config_data:
            data = config_data[k]
            attr = prefix + k.lower()
            self.__dict__[attr] = data
            if isinstance(data, dict):
                next_prefix = attr + "_"
                self._load_config_data(data, prefix=next_prefix)


@contextmanager
def progress_lock(max_wait_time=5):
    try:
        sleep_time = 0
        lock = "progress.lock"
        while True:
            if os.path.isfile(lock):
                if sleep_time > max_wait_time:
                    try:
                        os.remove(lock)
                        break
                    except:
                        continue
                else:
                    random_number = random.random() * 0.1
                    sleep_time += random_number

                time.sleep(random_number)
            else:
                break
        with open(lock, "w"):
            pass
        yield
    finally:
        try:
            os.remove(lock)
        except:
            pass




if __name__ == '__main__':

    get_files()
