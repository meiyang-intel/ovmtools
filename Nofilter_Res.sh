#!/bin/bash

# Change the model_dir and common_args
cpus=4,5,6,7
node=0
model_dir=`pwd`/../model/inn_nfs_share/cv_bench_cache/try_builds_cache/sk_13sept_75models_22.2/
model_int8_dir=`pwd`/../model/inn_nfs_share/cv_bench_cache/sk_13sept_75models_22.2_int8/

export ONEDNN_VERBOSE=0
common_args=" -t 5 -nireq=1 -nstreams 1 -nthreads 4 -pc -infer_precision f32 -json_stats -report_type=detailed_counters"

### Step1 Collect model name
# Get the model name files, no_filter_fp32_model.txt and no_filter_int8_model.txt
find $model_dir -name *.xml > no_filter_fp32_model.txt
find $model_int8 -name *.xml > no_filter_int8_model.txt

### Step 2 Collect comparing benchmark data of these models
# Input the file name of fp32 output
# Input the file name of int8 output
binA=`pwd`/../openvino/bin/intel64/Release

mkdir -p ./no_filter_enable_brgconv
mkdir -p ./no_filter_disable_brgconv
mkdir -p ./no_filter_output_fp32
mkdir -p ./no_filter_output_int8

export OV_CPU_DEBUG_LOG='CreatePrimitives;conv.cpp'
export VERBOSE_CONVERT=`pwd`/../openvino/src/plugins/intel_cpu/thirdparty/onednn/scripts/verbose_converter
export DNNL_MAX_CPU_ISA=AVX512_CORE_VNNI

#fp32
model_list_path=`pwd`/no_filter_fp32_model.txt
for line in `cat ${model_list_path}`
do
  echo $line
  export USE_BRG=1
  LD_LIBRARY_PATH=${binA}/lib ONEDNN_VERBOSE=0 taskset -c $cpus numactl -C $cpus -m $node -- /usr/bin/time -v ${binA}/benchmark_app -m ${line} $common_args -exec_graph_path ./no_filter_enable_brgconv/no_filter_exec_graph_enable_brgconv.xml -report_folder=./no_filter_enable_brgconv |& tee ./no_filter_enable_brgconv/no_filter_pc_enable_brgconv.txt
  export USE_BRG=0
  LD_LIBRARY_PATH=${binA}/lib ONEDNN_VERBOSE=0 taskset -c $cpus numactl -C $cpus -m $node -- /usr/bin/time -v ${binA}/benchmark_app -m ${line} $common_args -exec_graph_path ./no_filter_disable_brgconv/no_filter_exec_graph_disable_brgconv.xml -report_folder=./no_filter_disable_brgconv |& tee ./no_filter_disable_brgconv/no_filter_pc_disable_brgconv.txt
  python3 compare_py_brgconv_100.py ./no_filter_enable_brgconv/no_filter_pc_enable_brgconv.txt ./no_filter_disable_brgconv/no_filter_pc_disable_brgconv.txt ./no_filter_enable_brgconv/no_filter_exec_graph_enable_brgconv.xml ./no_filter_disable_brgconv/no_filter_exec_graph_disable_brgconv.xml -m ${line} -output_file ./no_filter_output_fp32
done

#int8
model_list_path=`pwd`/no_filter_int8_model.txt
for line in `cat ${model_list_path}`
do
  echo $line
  export USE_BRG=1
  LD_LIBRARY_PATH=${binA}/lib ONEDNN_VERBOSE=0 taskset -c $cpus numactl -C $cpus -m $node -- /usr/bin/time -v ${binA}/benchmark_app -m ${line} $common_args -exec_graph_path ./no_filter_enable_brgconv/no_filter_exec_graph_enable_brgconv.xml -report_folder=./no_filter_enable_brgconv |& tee ./no_filter_enable_brgconv/no_filter_pc_enable_brgconv.txt
  export USE_BRG=0
  LD_LIBRARY_PATH=${binA}/lib ONEDNN_VERBOSE=0 taskset -c $cpus numactl -C $cpus -m $node -- /usr/bin/time -v ${binA}/benchmark_app -m ${line} $common_args -exec_graph_path ./no_filter_disable_brgconv/no_filter_exec_graph_disable_brgconv.xml -report_folder=./no_filter_disable_brgconv |& tee ./no_filter_disable_brgconv/no_filter_pc_disable_brgconv.txt
  python3 compare_py_brgconv_100.py ./no_filter_enable_brgconv/no_filter_pc_enable_brgconv.txt ./no_filter_disable_brgconv/no_filter_pc_disable_brgconv.txt ./no_filter_enable_brgconv/no_filter_exec_graph_enable_brgconv.xml ./no_filter_disable_brgconv/no_filter_exec_graph_disable_brgconv.xml -m ${line} -output_file ./no_filter_output_int8
done

### Step3 Change the result of Step2 to csv
# Collect convolution layer data
python3 analyze_conv_flexCPU.py -c $cpus -i ./no_filter_output_fp32/ -o no_filter_result_fp32_conv.csv
python3 analyze_conv_flexCPU.py -c $cpus -i ./no_filter_output_int8/ -o no_filter_result_int8_conv.csv