import os
import argparse
import os.path
import json
import csv
import subprocess


class MultiBenchmark(object):
    def __init__(self, arg):
        self.args = arg
        self.res = []

    def multi_run_benchmark(self):
        i = 1
        while i <= 50:
            __, result = subprocess.getstatusoutput("cd ../openvino/bin/intel64/Release && "
                                                    "./benchmark_app " + self.args)
            self.res.append(float(result.split('\n')[-1].split(" ")[-2]))
            i += 1
        avg_res = sum(self.res)/50
        return avg_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-arg", "--arguments", help="arguments", type=str)
    args = parser.parse_args()
    data = MultiBenchmark(args.arguments)
    fps = data.multi_run_benchmark()
    print("FPS: ", fps)
