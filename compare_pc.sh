#!/bin/bash

binA=`pwd`/../openvino/bin/intel64/Release
binB=~/openvino/bin-new2.6B/intel64/Release
binB=$binA

args=" -t 5 -nireq=1 -nstreams 1 -nthreads 4 -pc -infer_precision f32 -json_stats -report_type=detailed_counters"
nfs_models_cache=~/models/nfs/models_cache
mkdir -p ./a
mkdir -p ./b

cores=4,5,6,7
node=0

#models=`find ${nfs_models_cache} -name *.xml`
export OV_CPU_DEBUG_LOG='CreatePrimitives;conv.cpp;deconv.cpp'
export VERBOSE_CONVERT=`pwd`/../openvino/src/plugins/intel_cpu/thirdparty/onednn/scripts/verbose_converter
USE_BRG=0 LD_LIBRARY_PATH=${binB}/lib ONEDNN_VERBOSE=0 taskset -c $cores numactl -C $cores -m $node -- /usr/bin/time -v ${binB}/benchmark_app -m $1 $args -exec_graph_path exec_graph_B.xml -report_folder=./b |& tee pcB.txt
USE_BRG=1 LD_LIBRARY_PATH=${binA}/lib ONEDNN_VERBOSE=0 taskset -c $cores numactl -C $cores -m $node -- /usr/bin/time -v ${binA}/benchmark_app -m $1 $args -exec_graph_path exec_graph_A.xml -report_folder=./a |& tee pcA.txt

echo A = ${binA}
echo B = ${binB}

python3 compare_vis.py pcA.txt pcB.txt