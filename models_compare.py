#!/usr/bin/python3

from xmlrpc.client import Boolean
import numpy as np
import sys, os
import argparse
from xml.dom.minidom import parse
import xml.dom.minidom
import json

os_path = os.path.abspath(os.path.dirname(os.getcwd()))


class Model:
    def __init__(self, path=os.path.join(os_path, 'model')):
        self.path = path
        self.model = []

    def file_path_walk(self):
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if ".xml" in file:
                    self.model.append(os.path.join(root, file))
                    print("files", files)
        return self.model

    def write_into_txt(self):
        with open("model_name.txt","w+") as f:
            for i in self.model:
                f.write(i + '\n')


if __name__ == "__main__":
    model = Model()
    model_path = model.file_path_walk()
    model.write_into_txt()
    path = '/home/ruiqi/openVINO/model/nfs/models_cache/resnet-50/caffe/caffe/FP32/1/dldt'
    A = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if ".xml" in file:
                A.append(file)
    print("finish")
