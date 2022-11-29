import os
from pathlib import Path
import subprocess
from Filter import FilterModel
import configparser


class Process:
    def __init__(self, config):
        self.config = config
        self.fp32_model = self.config['FP32']['model_path']
        self.int8_model = self.config['INT8']['model_path']
        self.filter = eval(self.config['Filter']['need_filter'])
        self.benchdnn = eval(self.config['Mode']['need_run_benchdnn'])
        self.compare_bin = eval(self.config['Mode']['diff_bin'])
        self.compare_env = eval(self.config['Mode']['diff_env'])
        self.logA_detail_file = ""
        self.logB_detail_file = ""
        self.outA_file = ""
        self.outB_file = ""
        self.compare_max_file = ""
        self.filter_layer_file = ""
        self.cpus = self.config['Basic']['cpus']
        self.node = self.config['Basic']['node']
        self.script_path = os.getcwd()
        self.compare_max_lst = []
        self.filter_model_list = []

    def get_models(self):
        self.fp32_model = Path(self.fp32_model).rglob('*.xml') if os.path.isdir(self.fp32_model) else self.fp32_model
        self.int8_model = Path(self.int8_model).rglob('*.xml') if os.path.isdir(self.int8_model) else self.int8_model

    def del_old_data_file(self, save_path, prefixA, prefixB):
        files = [f'{save_path}/{prefixA}_detail.log', f'{save_path}/{prefixA}.log',
                 f'{save_path}/{prefixA}.vs.{prefixB}.max.csv',
                 f'{save_path}/{prefixB}_detail.log', f'{save_path}/{prefixB}.log',
                 f'{save_path}/{prefixA}.layer.csv']
        for file in files:
            if (os.path.exists(file)):
                os.remove(file)

    def openfile(self, save_path, binA_prefix, binB_prefix):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.logA_detail_file = open(f'{save_path}/{binA_prefix}_detail.log', 'w')
        self.outA_file = open(f'{save_path}/{binA_prefix}.log', 'w')
        self.logB_detail_file = open(f'{save_path}/{binB_prefix}_detail.log', 'w')
        self.outB_file = open(f'{save_path}/{binB_prefix}.log', 'w')
        if self.filter:
            self.compare_max_file = open(f'{save_path}/{binA_prefix}.vs.{binB_prefix}.max.csv','w')
            compare_filter_A_B_file_title = 'model,' + f'{binA_prefix}' + ',' + f'{binB_prefix}' + ',' + "ratio (A-B)/B," + 'geomean\n'
            self.compare_max_file.write(compare_filter_A_B_file_title)
            self.filter_layer_file = open(f'{save_path}/{binA_prefix}.layer.csv','w')
            layer_title = f'name,{binA_prefix}_fps,{binB_prefix}_fps,ratio,delta(ms),layer1,time1(ms),,,,,,,,,,\n'
            self.filter_layer_file.write(layer_title)

    def process_file(self, precision):
        save_path = f'{self.script_path}/{self.config[precision]["save_folder"]}'
        binA_prefix = self.config[precision]['binA_prefix']
        binB_prefix = self.config[precision]['binB_prefix']
        self.del_old_data_file(save_path, binA_prefix, binB_prefix)
        self.openfile(save_path, binA_prefix, binB_prefix)
        return save_path

    def closefile(self):
        self.logA_detail_file.close()
        self.outA_file.close()
        self.logB_detail_file.close()
        self.outB_file.close()
        if self.filter:
            self.compare_max_file.close()

    def process_benmark(self):
        if self.fp32_model:
            self.compare_max_lst = []
            self.filter_model_list = []
            save_path = self.process_file(precision="FP32")
            self.choose_bins_env(self.fp32_model, "FP32", save_path)
        if self.int8_model:
            self.compare_max_lst = []
            self.filter_model_list = []
            save_path = self.process_file(precision="INT8")
            self.choose_bins_env(self.int8_model, "INT8", save_path)

    def choose_bins_env(self, models, precision, save_path):
        binA_prefix = self.config[precision]['binA_prefix']
        binB_prefix = self.config[precision]['binB_prefix']
        common_args = self.config[precision]['common_args']
        binA = self.config['BIN']['binA']
        binB = self.config['BIN']['binB']
        envA = self.config['Mode']['envA']
        envB = self.config['Mode']['envB']
        if self.filter:
            compare_max_lst = []
            filter_model_list = []
        for idx, path in enumerate(models):
            model_path = os.path.normpath(str(path))
            print(f'{idx:3d} {model_path}... ', end='')
            benchmark_log_A = []
            fps_resultA = []
            benchmark_log_B = []
            fps_resultB = []
            if self.compare_bin and self.compare_env:
                benchmark_log_A, fps_resultA = self.run_benmark_app(binA, envA, model=model_path, prefix=binA_prefix,
                                                                    common_args=common_args,
                                                                    log_file=self.logA_detail_file,
                                                                    out_file=self.outA_file,
                                                                    report_folder=f'{save_path}/a')
                benchmark_log_B, fps_resultB = self.run_benmark_app(binB, envB, model=model_path, prefix=binB_prefix,
                                                                    common_args=common_args,
                                                                    log_file=self.logB_detail_file,
                                                                    out_file=self.outB_file,
                                                                    report_folder=f'{save_path}/b')
            elif self.compare_bin and not self.compare_env:
                benchmark_log_A, fps_resultA = self.run_benmark_app(binA, model=model_path, prefix=binA_prefix,
                                                                    common_args=common_args,
                                                                    log_file=self.logA_detail_file,
                                                                    out_file=self.outA_file,
                                                                    report_folder=f'{save_path}/a')
                benchmark_log_B, fps_resultB = self.run_benmark_app(binB, model=model_path, prefix=binB_prefix,
                                                                    common_args=common_args,
                                                                    log_file=self.logB_detail_file,
                                                                    out_file=self.outB_file,
                                                                    report_folder=f'{save_path}/b')
            elif not self.compare_bin and self.compare_env:
                benchmark_log_A, fps_resultA = self.run_benmark_app(binA, envA, model=model_path, prefix=binA_prefix,
                                                                    common_args=common_args,
                                                                    log_file=self.logA_detail_file,
                                                                    out_file=self.outA_file,
                                                                    report_folder=f'{save_path}/a')
                benchmark_log_B, fps_resultB = self.run_benmark_app(binA, envB, model=model_path, prefix=binB_prefix,
                                                                    common_args=common_args,
                                                                    log_file=self.logB_detail_file,
                                                                    out_file=self.outB_file,
                                                                    report_folder=f'{save_path}/b')
            else:
                benchmark_log_A, fps_resultA = self.run_benmark_app(binA, model=model_path, prefix=binA_prefix,
                                                                    common_args=common_args,
                                                                    log_file=self.logA_detail_file,
                                                                    out_file=self.outA_file,
                                                                    report_folder=f'{save_path}/a')
            if self.filter and benchmark_log_B:
                self.get_filter_data(benchmark_log_A, benchmark_log_B, fps_resultA, fps_resultB, precision, f'{save_path}/a', f'{save_path}/b')

        if self.filter:
            self.write_into_max_csv(self.compare_max_file, self.compare_max_lst)
        self.closefile()

    def run_benmark_app(self, bin, env="", model="", prefix="", common_args="", log_file="", out_file="", report_folder = ""):
        # os.chdir(bin)
        print("bin: ", bin)
        if env: para, val = self.process_env(env)
        result = []
        benchmark_log_list = []
        if not os.path.exists(report_folder):
            os.makedirs(report_folder)
        for i in range(3):
            log_file.write(f'========================= {bin} testing {model}...\n')
            if env: os.environ[para] = val
            bin_cmd = 'numactl -C ' + self.cpus + ' -m ' + self.node + f' {bin}/benchmark_app' + \
                      ' -m ' + model + " " + common_args + ' -report_folder=' + report_folder
            print("cmd: ", bin_cmd)
            outputA = subprocess.run(bin_cmd, shell=True, capture_output=True)
            out = outputA.stdout.decode()
            # print("benchmark detail log: ", out)
            log_file.write(out)
            if outputA.returncode == 0:
                fps = out.split('Throughput')[-1]
                fps = fps.split()[1]
                line = f'{fps} {model}'
                result.append(line)
                print(f'{fps}', end=' ')
            else:
                line = f'-1 {model}'
                result.append(line)
            out_file.write(line + '\n')
            out_file.flush()
        print(' ')
        # benchmark_log_list.append([model, out])
        if self.compare_bin:
            pass  # filter
        print("finish")
        return out, result

    def process_env(self, env):
        para, val = env.split("=")
        return para, val

    def get_filter_data(self, benchmark_log_A, benchmark_log_B, fps_resultA, fps_resultB, precision, reportA, reportB):
        model_filter = FilterModel(benchmark_log_A, benchmark_log_B, fps_resultA, fps_resultB, self.config)
        results_lst, result_sort_sets = model_filter.filter_res()
        self.compare_max_lst.append(results_lst)
        save_path = f'{self.script_path}/{self.config[precision]["save_folder"]}'
        if result_sort_sets:
            self.filter_model_list.append(result_sort_sets)
            layer_result = model_filter.run_compare_tool(result_sort_sets, save_path, reportA, reportB)
            self.write_layer_csv(result_sort_sets, layer_result)

    def write_layer_csv(self, result_sort, layer_result):
        name = result_sort[0]
        binA_fps = result_sort[1]
        binB_fps = result_sort[2]
        ratio = (binA_fps-binB_fps)*100/binB_fps
        self.filter_layer_file.write(f'{name},{binA_fps},{binB_fps},{ratio:.2f}%,{(-1/binA_fps+1/binB_fps)*1000},')
        result = sorted(layer_result, key=lambda x: x[1])
        for (n, t) in result:
            if t < 0:
                self.filter_layer_file.write(f'{n},{t},')
        self.filter_layer_file.write('\n')

    def write_benchamrk_temp_log(self, file, benchmark_log):
        with open(file, 'w') as f:
            f.write(benchmark_log)

    def write_into_max_csv(self, file, data):
        file.write("\n".join(data))


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("config.ini")
    A = Process(config)
    A.get_models()
    A.process_benmark()