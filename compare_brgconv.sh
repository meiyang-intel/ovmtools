#!/bin/bash

binA=`pwd`/../openvino/bin/intel64/Release

args=" -t 5 -nireq=1 -nstreams 1 -nthreads 4 -pc -infer_precision f32 -json_stats -report_type=detailed_counters"
nfs_models_cache=~/models/nfs/models_cache
mkdir -p ./enable_brgconv
mkdir -p ./disable_brgconv

cores=4,5,6,7
node=0

#models=`find ${nfs_models_cache} -name *.xml`
export OV_CPU_DEBUG_LOG='CreatePrimitives;conv.cpp'
export VERBOSE_CONVERT=`pwd`/../openvino/src/plugins/intel_cpu/thirdparty/onednn/scripts/verbose_converter
export DNNL_MAX_CPU_ISA=AVX512_CORE_VNNI
model_list_path="/home/ruiqi/openVINO/model/modelName.txt"
for line in `cat ${model_list_path}`
do
  echo $line
  export USE_BRG=1
  LD_LIBRARY_PATH=${binA}/lib ONEDNN_VERBOSE=0 taskset -c $cores numactl -C $cores -m $node -- /usr/bin/time -v ${binA}/benchmark_app -m ${line} $args -exec_graph_path ./enable_brgconv/exec_graph_enable_brgconv.xml -report_folder=./enable_brgconv |& tee ./enable_brgconv/pc_enable_brgconv.txt
  export USE_BRG=0
  LD_LIBRARY_PATH=${binA}/lib ONEDNN_VERBOSE=0 taskset -c $cores numactl -C $cores -m $node -- /usr/bin/time -v ${binA}/benchmark_app -m ${line} $args -exec_graph_path ./disable_brgconv/exec_graph_disable_brgconv.xml -report_folder=./disable_brgconv |& tee ./disable_brgconv/pc_disable_brgconv.txt
  python3 compare_py_brgconv.py ./enable_brgconv/pc_enable_brgconv.txt ./disable_brgconv/pc_disable_brgconv.txt -m ${line}
done

#export USE_BRG=1
#LD_LIBRARY_PATH=${binA}/lib ONEDNN_VERBOSE=0 taskset -c $cores numactl -C $cores -m $node -- /usr/bin/time -v ${binA}/benchmark_app -m $1 $args -exec_graph_path ./enable_brgconv/exec_graph_enable_brgconv.xml -report_folder=./enable_brgconv |& tee ./enable_brgconv/pc_enable_brgconv.txt
#export USE_BRG=0
#LD_LIBRARY_PATH=${binA}/lib ONEDNN_VERBOSE=0 taskset -c $cores numactl -C $cores -m $node -- /usr/bin/time -v ${binA}/benchmark_app -m $1 $args -exec_graph_path ./disable_brgconv/exec_graph_disable_brgconv.xml -report_folder=./disable_brgconv |& tee ./disable_brgconv/pc_disable_brgconv.txt

#echo A = ${binA}
#echo B = ${binA}

#python3 compare_py_brgconv.py ./enable_brgconv/pc_enable_brgconv.txt ./disable_brgconv/pc_disable_brgconv.txt -m $model