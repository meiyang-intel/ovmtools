import sys, os
import argparse
import os.path
import json
import csv
import subprocess


class Analyze(object):
    def __init__(self, cpu, input_path, output_csv):
        self.cpu = cpu
        self.input = input_path
        self.output = output_csv
        self.model = []
        self.data = []

    def __get_file__(self):
        for root, _, files in os.walk(self.input):
            for file in files:
                self.model.append(os.path.join(root, file))

    def get_data(self):
        self.__get_file__()
        for i in self.model:
            json_data = self.read_file(i)
            self.analyze_data(json_data)
        self.__write_csv__()

    @staticmethod
    def read_file(model):
        with open(model, 'r') as json_file:
            json_data = json.load(json_file)
        return json_data

    def analyze_data(self, json_data):
        res = []
        model_name = json_data["model_name"]
        print(model_name)
        for layer_id in range(len(json_data) - 1):
            res_layer = []
            print(layer_id)
            layer_data = json_data['layer' + str(layer_id)]
            layer_name = layer_data["layer_name"]
            layer_type = layer_data["brgconv"]["node_type"] if layer_data["brgconv"]["node_type"] \
                else layer_data["jit"]["node_type"]
            brg_info = self.__exec_info__(layer_data, "brgconv")
            jit_info = self.__exec_info__(layer_data, "jit")
            brg_cmd = brg_info[1]
            jit_cmd = jit_info[1]
            if brg_cmd != 'NA' or jit_cmd != 'NA':
                brg_cmd, jit_cmd = self.__check_invalid_attr(brg_info[1], jit_info[1])
            if brg_info[1] != "NA" and "Convolution" in layer_type:
                brg_average_t, brg_min_t = self.__run_benchdnn__(brg_cmd)
            else:
                brg_average_t = 0
                brg_min_t = 0
            if jit_info[1] != "NA" and "Convolution" in layer_type:
                jit_average_t, jit_min_t = self.__run_benchdnn__(jit_cmd)
            else:
                jit_average_t = 0
                jit_min_t = 0

            brg_average_t *= 1000
            brg_min_t *= 1000
            jit_average_t *= 1000
            jit_min_t *= 1000
            diff_json = self.__get_time_dif__(brg_info[2], jit_info[2])
            diff_average = self.__get_time_dif__(brg_average_t, jit_average_t)
            diff_min = self.__get_time_dif__(brg_min_t, jit_min_t)

            if "Convolution" in layer_type:
                res_layer += [model_name] + [layer_name] + [layer_type] + \
                             [brg_info[0], brg_cmd, brg_info[2]] + \
                             [str(brg_average_t)] + [str(brg_min_t)] + \
                             [jit_info[0], jit_cmd, jit_info[2]] + \
                             [str(jit_average_t)] + [str(jit_min_t)] + \
                             [str(diff_json)] + [str(diff_average)] + [str(diff_min)]
                res.append(res_layer)
        res_sorted = sorted(res, key=lambda x:x[5])
        self.data += res_sorted

    @staticmethod
    def __exec_info__(layer_data, exec_type):
        info_lis = ["onednn_cmd", "benchdnn_cmd", "time"]
        exec_info = []
        for i in info_lis:
            if i == "time":
                exec_info.append(layer_data[exec_type][i] if layer_data[exec_type][i] else 0)
            else:
                exec_info.append(layer_data[exec_type][i].replace('\n', '') if layer_data[exec_type][i] else "NA")
        return exec_info

    def __check_invalid_attr(self, brg_cmd_org, jit_cmd_org):
        invalid_dnn_list = ["eltwise_hsigmoid", "eltwise_round_half_to_even", "eltwise_round_half_away_from_zero",
                            "depthwise_scale_shift", "depthwise_prelu",
                            "quantization_quantize_dequantize", "quantization_quantize", "binarization_depthwise"]
        brg_cmd = brg_cmd_org
        jit_cmd = jit_cmd_org
        for i in invalid_dnn_list:
            if i in brg_cmd_org or i in jit_cmd_org:
                print("invalid attr: ", i)
                brg_cmd = self.__del_attr__(brg_cmd_org)
                jit_cmd = self.__del_attr__(jit_cmd_org)
        return brg_cmd, jit_cmd

    def __del_attr__(self, cmd_org):
        cmd_tmp = cmd_org.split(" ")
        cmd = " ".join(i for i in cmd_tmp if "attr-post-ops" not in i)
        return cmd

    def __run_benchdnn__(self, cmd):
        average_t = "0"
        min_t = "0"
        status, result = subprocess.getstatusoutput("cd ../openvino/bin/intel64/Release && numactl -C " + self.cpu + " -m 0 -- " + cmd)
        # assert status == 0, "Error running benchdnn"
        print("cmd: ", cmd)
        if status != 0:
            print("Error running benchdnn")
            return 0, 0
        tmp = result.split('\n')[-1].split(" ")
        min_t = float(tmp[2].split(":")[-1])
        average_t = float(tmp[-1].split(":")[-1])
        return average_t, min_t

    @staticmethod
    def __get_time_dif__(brg_time, no_brg_time):
        if no_brg_time:
            return (brg_time - no_brg_time) / no_brg_time
        else:
            return "NA"

    def __write_csv__(self):
        with open(self.output, mode='w', newline='', encoding='utf8') as f_csv:
            cf = csv.writer(f_csv)
            cf.writerow(['model_name', 'layer_name', 'layer_type',
                         'brg_verbose', 'brg_benchdnn_cmd', 'brg_time',
                         'brg_benchdnn_time_average', 'brg_benchdnn_time_min',
                         'no_brg_verbose', 'no_brg_cmd', 'no_brg_time',
                         'no_brg_benchdnn_time_average', 'no_brg_benchdnn_time_min',
                         '(brg_time - no_brg_time)/no_brg time',
                         '(brg_benchdnn_time_average - no_brg_benchdnn_time_average)/no_brg_benchdnn_time_average',
                         '(brg_benchdnn_time_min - no_brg_benchdnn_time_min)/no_brg_benchdnn_time_min'])
            for i in self.data:
                cf.writerow(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cpus", help="the core number of cpu", type=str)
    parser.add_argument("-i", "--input", help="the folder of json data", type=str)
    parser.add_argument("-o", "--output", help="write into csv", default="result.csv", type=str)
    args = parser.parse_args()
    data = Analyze(args.cpus, args.input, args.output)
    data.get_data()
