import os
import create_compare_csv as create_compare_csv
import os.path
import json
import csv
import subprocess


class CompareModel():
    def __init__(self, exec_graph_A, exec_graph_B, model, log_file_A, log_file_B, prefixA, prefixB,
                 reportA, reportB, output_file):
        self.exec_graph_A = exec_graph_A
        self.exec_graph_B = exec_graph_B
        self.model = model
        self.log_file_A = log_file_A
        self.log_file_B = log_file_B
        self.prefixA = prefixA
        self.prefixB = prefixB
        self.reportA = reportA
        self.reportB = reportB
        self.output_file = output_file

    def run_create_compare_csv_tool(self):
        out_path = os.path.join(self.output_file, "json_data")
        result = create_compare_csv.main(self.exec_graph_A, self.exec_graph_B, self.model,
                                         self.log_file_A, self.log_file_B, self.prefixA, self.prefixB,
                                         self.reportA, self.reportB, out_path)
        return result


class Analyze(object):
    def __init__(self, cpu, benchdnn_flag, input_path, output_csv, binA_prefix, binB_prefix, binA, binB=""):
        self.cpu = cpu
        self.benchdnn_flag = benchdnn_flag
        self.input = input_path
        self.output = output_csv
        self.model = []
        self.data = []
        self.prefixA = binA_prefix
        self.prefixB = binB_prefix
        self.binA = binA
        self.binB = binB if binB else binA

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
            layer_type = layer_data[self.prefixA]["node_type"] if layer_data[self.prefixA]["node_type"] \
                else layer_data[self.prefixB]["node_type"]
            prefixA_info = self.__exec_info__(layer_data, self.prefixA)
            prefixB_info = self.__exec_info__(layer_data, self.prefixB)
            prefixA_cmd = prefixA_info[1]
            prefixB_cmd = prefixB_info[1]
            if prefixA_cmd != 'NA' or prefixB_cmd != 'NA':
                prefixA_cmd, prefixB_cmd = self.__check_invalid_attr(prefixA_info[1], prefixB_info[1])
            if prefixA_info[1] != "NA":
                prefixA_average_t, prefixA_min_t = self.__run_benchdnn__(prefixA_cmd, "a")
            else:
                prefixA_average_t = 0
                prefixA_min_t = 0
            if prefixB_info[1] != "NA":
                prefixB_average_t, prefixB_min_t = self.__run_benchdnn__(prefixB_cmd, "b")
            else:
                prefixB_average_t = 0
                prefixB_min_t = 0

            prefixA_average_t *= 1000
            prefixA_min_t *= 1000
            prefixB_average_t *= 1000
            prefixB_min_t *= 1000
            diff_json = self.__get_time_dif__(prefixA_info[2], prefixB_info[2])
            diff_average = self.__get_time_dif__(prefixA_average_t, prefixB_average_t)
            diff_min = self.__get_time_dif__(prefixA_min_t, prefixB_min_t)

            res_layer += [model_name] + [layer_name] + [layer_type] + \
                         [prefixA_info[0], prefixA_cmd, prefixA_info[2]] + \
                         [str(prefixA_average_t)] + [str(prefixA_min_t)] + \
                         [prefixB_info[0], prefixB_cmd, prefixB_info[2]] + \
                         [str(prefixB_average_t)] + [str(prefixB_min_t)] + \
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

    def __check_invalid_attr(self, prefixA_cmd_org, prefixB_cmd_org):
        invalid_dnn_list = ["eltwise_hsigmoid", "eltwise_round_half_to_even", "eltwise_round_half_away_from_zero",
                            "depthwise_scale_shift", "depthwise_prelu",
                            "quantization_quantize_dequantize", "quantization_quantize", "binarization_depthwise"]
        prefixA_cmd = prefixA_cmd_org
        prefixB_cmd = prefixB_cmd_org
        for i in invalid_dnn_list:
            if i in prefixA_cmd_org or i in prefixB_cmd_org:
                print("invalid attr: ", i)
                prefixA_cmd = self.__del_attr__(prefixA_cmd_org)
                prefixB_cmd = self.__del_attr__(prefixB_cmd_org)
        return prefixA_cmd, prefixB_cmd

    def __del_attr__(self, cmd_org):
        cmd_tmp = cmd_org.split(" ")
        cmd = " ".join(i for i in cmd_tmp if "attr-post-ops" not in i)
        return cmd

    def __run_benchdnn__(self, cmd, flag):
        average_t = "0"
        min_t = "0"
        if not self.benchdnn_flag:
            return 0, 0
        if flag == "a":
            path = "cd " + self.binA
            status, result = subprocess.getstatusoutput(path + "&& numactl -C " + self.cpu + " -m 0 -- " + cmd)
        elif flag == "b":
            path = "cd " + self.binB
            status, result = subprocess.getstatusoutput(path + "&& numactl -C " + self.cpu + " -m 0 -- " + cmd)
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
    def __get_time_dif__(prefixA_time, prefixB_time):
        if prefixB_time:
            return (prefixA_time - prefixB_time) / prefixB_time
        else:
            return "NA"

    def __write_csv__(self):
        with open(self.output, mode='w', newline='', encoding='utf8') as f_csv:
            cf = csv.writer(f_csv)
            cf.writerow(['model_name', 'layer_name', 'layer_type',
                         f'{self.prefixA}', f'{self.prefixA}_benchdnn_cmd', f'{self.prefixA}_time',
                         f'{self.prefixA}_benchdnn_time_average', f'{self.prefixA}_benchdnn_time_min',
                         f'{self.prefixB}', f'{self.prefixB}_cmd', f'{self.prefixB}_time',
                         f'{self.prefixB}_benchdnn_time_average', f'{self.prefixB}_benchdnn_time_min',
                         f'({self.prefixA}_time - {self.prefixB}_time)/{self.prefixB}_time',
                         f'({self.prefixA}_benchdnn_time_average - {self.prefixB}_benchdnn_time_average)/{self.prefixB}_benchdnn_time_average',
                         f'({self.prefixA}_benchdnn_time_min - {self.prefixB}_benchdnn_time_min)/{self.prefixB}_benchdnn_time_min'])
            for i in self.data:
                cf.writerow(i)

