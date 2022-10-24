#!/bin/bash

# Change the model_dir and common_args
cpus=4,5,6,7
node=0

export ONEDNN_VERBOSE=0
common_args=" -d CPU -nstreams 1 -niter 100 -json_stats -report_type=detailed_counters"

binA=`pwd`/../openvino/bin/intel64/Release

rm -rf single_a
rm -rf single_b
rm -rf single_output

mkdir -p ./single_a
mkdir -p ./single_b
mkdir -p ./single_output

export OV_CPU_DEBUG_LOG='CreatePrimitives;conv.cpp'
export VERBOSE_CONVERT=`pwd`/../openvino/src/plugins/intel_cpu/thirdparty/onednn/scripts/verbose_converter
export DNNL_MAX_CPU_ISA=AVX512_CORE_VNNI

line=$1
echo $line
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
LD_LIBRARY_PATH=${binA}/lib ONEDNN_VERBOSE=0 taskset -c $cpus numactl -C $cpus -m $node -- /usr/bin/time -v ${binA}/benchmark_app -m ${line} $common_args -exec_graph_path ./single_a/single_a_exec_graph.xml -report_folder=./single_a |& tee ./single_a/single_a_pc.txt
export DNNL_MAX_CPU_ISA=AVX512_CORE_VNNI
LD_LIBRARY_PATH=${binA}/lib ONEDNN_VERBOSE=0 taskset -c $cpus numactl -C $cpus -m $node -- /usr/bin/time -v ${binA}/benchmark_app -m ${line} $common_args -exec_graph_path ./single_b/single_b_exec_graph.xml -report_folder=./single_b |& tee ./single_b/single_b_pc.txt
python3 compare_py_brgconv_100.py ./single_a/single_a_pc.txt ./single_b/single_b_pc.txt ./single_a/single_a_exec_graph.xml ./single_b/single_b_exec_graph.xml -m ${line} -output_file ./single_output

python3 analyze_data.py -i ./single_output/ -o single_result.csv
