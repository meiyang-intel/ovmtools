import env_config
import os
from pathlib import Path
import subprocess
import Filter

class Mode:
    def __init__(self):
        self.filter = env_config.need_filter
        self.benchdnn = env_config.need_run_benchdnn
        self.compare_bin = env_config.diff_bin
        self.compare_env = env_config.diff_env
        self.fp32_model = env_config.fp32_model_path
        self.fp32_args = env_config.fp32_common_args
        self.int8_model = env_config.int8_model_path
        self.int8_args = env_config.int8_common_args

    def get_models(self):
        self.fp32_model = Path(self.fp32_model).rglob('*.xml') if os.path.isdir(self.fp32_model) else self.fp32_model
        self.int8_model = Path(self.int8_model).rglob('*.xml') if os.path.isdir(self.int8_model) else self.int8_model

    def process_benmark(self):
        if self.fp32_model:
            self.choose_bins_env(self.fp32_model, env_config.fp32_binA_prefix,
                                 env_config.fp32_binB_prefix, env_config.fp32_common_args)
        if self.int8_model:
            self.choose_bins_env(self.int8_model, env_config.int8_binA_prefix,
                                 env_config.int8_binB_prefix, env_config.int8_common_args)

    def choose_bins_env(self, models, binA_prefix, binB_prefix, common_args):
        logA_file = open(f'{os.getcwd()}/{binA_prefix}_detail.log', 'w')
        outA_file = open(f'{os.getcwd()}/{binA_prefix}.log', 'w')
        logB_file = open(f'{os.getcwd()}/{binB_prefix}_detail.log', 'w')
        outB_file = open(f'{os.getcwd()}/{binB_prefix}.log', 'w')
        for idx, path in enumerate(models):
            model_path = os.path.normpath(str(path))
            print(f'{idx:3d} {model_path}... ', end='')
            benchmark_log_listA = [] 
            fps_resultA = []
            benchmark_log_listB = []
            fps_resultB = []
            if self.compare_bin and self.compare_env:
                benchmark_log_listA, fps_resultA = self.run_benmark_app(env_config.binA, env_config.envA, model=model_path,
                                                                      prefix=binA_prefix, common_args=common_args,
                                                                      log_file=logA_file, out_file=outA_file)
                benchmark_log_listB, fps_resultB = self.run_benmark_app(env_config.binB, env_config.envB, model=model_path,
                                                                      prefix=binB_prefix, common_args=common_args,
                                                                      log_file=logB_file, out_file=outB_file)
            elif self.compare_bin and not self.compare_env:
                benchmark_log_listA, fps_resultA = self.run_benmark_app(env_config.binA, model=model_path, 
                                                                      prefix=binA_prefix, common_args=common_args, 
                                                                      log_file=logA_file, out_file=outA_file)
                benchmark_log_listB, fps_resultB = self.run_benmark_app(env_config.binB, model=model_path,
                                                                      prefix=binB_prefix, common_args=common_args, 
                                                                      log_file=logB_file, out_file=outB_file)
            elif not self.compare_bin and self.compare_env:
                benchmark_log_listA, fps_resultA = self.run_benmark_app(env_config.binA, env_config.envA, model=model_path,
                                                                      prefix=binA_prefix, common_args=common_args,
                                                                      log_file=logA_file, out_file=outA_file)
                benchmark_log_listB, fps_resultB = self.run_benmark_app(env_config.binA, env_config.envB, model=model_path,
                                                                      prefix=binB_prefix, common_args=common_args,
                                                                      log_file=logB_file, out_file=outB_file)
            else:
                benchmark_log_listA, fps_resultA = self.run_benmark_app(env_config.binA, model=model_path, 
                                                                      prefix=binA_prefix, common_args=common_args, 
                                                                      log_file=logA_file, out_file=outA_file)
            if self.filter:
                Filter(benchmark_log_listA, benchmark_log_listB, fps_resultA, fps_resultB)
        
        logA_file.close()
        logB_file.close()


    def run_benmark_app(self, bin, env="", model="", prefix="", common_args="", log_file="", out_file=""):
        os.chdir(bin)
        print("bin: ", bin)
        para, val = self.process_env(env)
        result = []
        benchmark_log_listA = []
        for i in range(3):
            log_file.write(f'========================= {bin} testing {model}...\n')
            os.environ[para] = val
            # bin_cmd = 'numactl -C ' + env_config.cpus + ' -m ' + str(env_config.node) + f' {bin}/benchmark_app'
            # outputA = subprocess.run([bin_cmd, '-m', model_path] + common_args, capture_output=True)
            outputA = subprocess.run('numactl -C ' + env_config.cpus + ' -m ' + str(env_config.node) +
                                     f' {bin}/benchmark_app' + ' -m ' + model + common_args + ' -report_folder=./b',
                                     shell=True, capture_output=True)
            out = outputA.stdout.decode()
            # print("benchmark detail log: ", out)
            log_file.write(out)
            if outputA.returncode == 0:
                fps = out.split('Throughput')[-1]
                fps = fps.split()[1]
                line = f'{fps} {model}'
                result.append(line)
                print(f'{fps}', end=' '),
            else:
                line = f'-1 {model}'
                result.append(line)
            out_file.write(line + '\n')
            out_file.flush()
        print(' ')
        benchmark_log_listA.append([model, out])
        if self.compare_bin:
            pass  # filter
        print("finish")
        return benchmark_log_listA, result

    def process_env(self, env):
        para, val = env.split("=")
        return para, val


if __name__ == '__main__':
    A = Mode()
    A.get_models()
    A.process_benmark()