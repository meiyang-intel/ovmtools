#!/bin/bash

### Step1 Running all_test.sh
# Change the model_dir and common_args
cpus=0,1,2,3
node=0
model_dir=`pwd`/../model/inn_nfs_share/cv_bench_cache/try_builds_cache/sk_13sept_75models_22.2/
bin_dir=`pwd`/../openvino/bin/intel64/Release

mkdir -p ./a
mkdir -p ./b
export ONEDNN_VERBOSE=0
common_args=" -t 5 -nireq=1 -nstreams 1 -nthreads 4 -pc -infer_precision f32 -json_stats -report_type=detailed_counters"

# fp32
new_log=brg.f32
base_log=jit.f32

# without brg
export USE_BRG=0
numactl -C $cpus -m $node -- python3 all_test.py $bin_dir $base_log $model_dir $common_args -report_folder=./b
# with brg
export USE_BRG=1
numactl -C $cpus -m $node -- python3 all_test.py $bin_dir $new_log $model_dir $common_args -report_folder=./a
numactl -C $cpus -m $node -- python3 all_postprocess.py $new_log.log $base_log.log -0.05 $bin_dir default_check $common_args
numactl -C $cpus -m $node -- python3 all_postprocess.py $base_log.log $new_log.log -0.05 $bin_dir check_fast $common_args

#i8
model_dir=`pwd`/../model/inn_nfs_share/cv_bench_cache/sk_13sept_75models_22.2_int8
new_log=brg.i8
base_log=jit.i8

# without brg
export USE_BRG=0
numactl -C $cpus -m $node -- python3 all_test.py $bin_dir $base_log $model_dir $common_args -report_folder=./b
# with brg
export USE_BRG=1
numactl -C $cpus -m $node -- python3 all_test.py $bin_dir $new_log $model_dir $common_args -report_folder=./a
numactl -C $cpus -m $node -- python3 all_postprocess.py $new_log.log $base_log.log -0.05 $bin_dir default_check $common_args
numactl -C $cpus -m $node -- python3 all_postprocess.py $base_log.log $new_log.log -0.05 $bin_dir check_fast $common_args

### Step2 Collect model name
# model name in brg.xxx.log.layer.csv and jit.xxx.log.layer.csv
python3 parse_model_all_test_res.py
# Get the model name files, fp32_model.txt and int8_model.txt

### Step 3 Collect comparing benchmark data of these models
# Input the file name of fp32 output
# Input the file name of int8 output
binA=`pwd`/../openvino/bin/intel64/Release

mkdir -p ./enable_brgconv
mkdir -p ./disable_brgconv
mkdir -p ./all_test_output_fp32
mkdir -p ./all_test_output_int8

export OV_CPU_DEBUG_LOG='CreatePrimitives;conv.cpp'
export VERBOSE_CONVERT=`pwd`/../openvino/src/plugins/intel_cpu/thirdparty/onednn/scripts/verbose_converter
export DNNL_MAX_CPU_ISA=AVX512_CORE_VNNI

#fp32
model_list_path=`pwd`/fp32_model.txt
for line in `cat ${model_list_path}`
do
  echo $line
  export USE_BRG=1
  LD_LIBRARY_PATH=${binA}/lib ONEDNN_VERBOSE=0 taskset -c $cpus numactl -C $cpus -m $node -- /usr/bin/time -v ${binA}/benchmark_app -m ${line} $common_args -exec_graph_path ./enable_brgconv/exec_graph_enable_brgconv.xml -report_folder=./enable_brgconv |& tee ./enable_brgconv/pc_enable_brgconv.txt
  export USE_BRG=0
  LD_LIBRARY_PATH=${binA}/lib ONEDNN_VERBOSE=0 taskset -c $cpus numactl -C $cpus -m $node -- /usr/bin/time -v ${binA}/benchmark_app -m ${line} $common_args -exec_graph_path ./disable_brgconv/exec_graph_disable_brgconv.xml -report_folder=./disable_brgconv |& tee ./disable_brgconv/pc_disable_brgconv.txt
  python3 compare_py_brgconv_100.py ./enable_brgconv/pc_enable_brgconv.txt ./disable_brgconv/pc_disable_brgconv.txt -m ${line} -output_file ./all_test_output_fp32
done

#int8
model_list_path=`pwd`/int8_model.txt
for line in `cat ${model_list_path}`
do
  echo $line
  export USE_BRG=1
  LD_LIBRARY_PATH=${binA}/lib ONEDNN_VERBOSE=0 taskset -c $cpus numactl -C $cpus -m $node -- /usr/bin/time -v ${binA}/benchmark_app -m ${line} $common_args -exec_graph_path ./enable_brgconv/exec_graph_enable_brgconv.xml -report_folder=./enable_brgconv |& tee ./enable_brgconv/pc_enable_brgconv.txt
  export USE_BRG=0
  LD_LIBRARY_PATH=${binA}/lib ONEDNN_VERBOSE=0 taskset -c $cpus numactl -C $cpus -m $node -- /usr/bin/time -v ${binA}/benchmark_app -m ${line} $common_args -exec_graph_path ./disable_brgconv/exec_graph_disable_brgconv.xml -report_folder=./disable_brgconv |& tee ./disable_brgconv/pc_disable_brgconv.txt
  python3 compare_py_brgconv_100.py ./enable_brgconv/pc_enable_brgconv.txt ./disable_brgconv/pc_disable_brgconv.txt -m ${line} -output_file ./all_test_output_int8
done

### Step4 Change the result of Step3 to csv
# Collect convolution layer data
python3 analyze_conv_flexCPU.py -c $cpus -i ./all_test_output_fp32/ -o all_test_result_fp32_conv.csv
python3 analyze_conv_flexCPU.py -c $cpus -i ./all_test_output_int8/ -o all_test_result_int8_conv.csv