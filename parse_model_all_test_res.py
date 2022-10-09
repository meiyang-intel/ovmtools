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
    def __init__(self, path=os.getcwd()):
        self.fp32_path = [os.path.join(path, 'brg.f32.log.layer.csv'), os.path.join(path, 'jit.f32.log.layer.csv')]
        self.int8_path = [os.path.join(path, 'brg.i8.log.layer.csv'), os.path.join(path, 'jit.i8.log.layer.csv')]
        self.fp32_model = []
        self.int8_model = []

    def get_file(self):
        self.fp32_model = self.read_file(self.fp32_path)
        self.write_into_txt(self.fp32_model, "fp32_model.txt")
        self.int8_model = self.read_file(self.int8_path)
        self.write_into_txt(self.int8_model, "int8_model.txt")

    def read_file(self, path):
        model_list = []
        for file in path:
            with open(file, "r") as fp32_f:
                data = fp32_f.readlines()
                model_list = self.parse_model(data[1:])
        return model_list

    @staticmethod
    def parse_model(data):
        model_list = []
        for val in data:
            model_tmp = val.split(",")[0]
            model_list.append(model_tmp)
        return list(set(model_list))

    @staticmethod
    def write_into_txt(data, name):
        with open(name, "w+") as f:
            for i in data:
                f.write(i + '\n')


if __name__ == "__main__":
    model = Model()
    model.get_file()
    print("finish")
