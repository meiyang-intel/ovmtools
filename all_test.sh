cpus=4,5,6,7
node=0
model_dir=`pwd`/../f32_models/
bin_dir=`pwd`/../openvino/bin/intel64/Release

mkdir -p ./a
mkdir -p ./b
export ONEDNN_VERBOSE=0
# fp32
new_log=brg.f32
base_log=jit.f32
common_args="-infer_precision=f32 -t=5 -b=1 -nireq=1 -nstreams=1 -nthreads=4 -json_stats -report_type=detailed_counters"
# without brg
export USE_BRG=0
numactl -C $cpus -m $node -- python3 all_test.py $bin_dir $base_log $model_dir $common_args -report_folder=./b
# with brg
export USE_BRG=1
numactl -C $cpus -m $node -- python3 all_test.py $bin_dir $new_log $model_dir $common_args -report_folder=./a
numactl -C $cpus -m $node -- python3 all_postprocess.py $new_log.log $base_log.log -0.05 $bin_dir default_check $common_args
numactl -C $cpus -m $node -- python3 all_postprocess.py $base_log.log $new_log.log -0.05 $bin_dir check_fast $common_args 

#i8
model_dir=`pwd`/../i8_models/
new_log=brg.i8
base_log=jit.i8
common_args="-t=5 -b=1 -nireq=1 -nstreams=1 -nthreads=4 -json_stats -report_type=detailed_counters"
# without brg
export USE_BRG=0
numactl -C $cpus -m $node -- python3 all_test.py $bin_dir $base_log $model_dir $common_args -report_folder=./b
# with brg
export USE_BRG=1
numactl -C $cpus -m $node -- python3 all_test.py $bin_dir $new_log $model_dir $common_args -report_folder=./a
numactl -C $cpus -m $node -- python3 all_postprocess.py $new_log.log $base_log.log -0.05 $bin_dir default_check $common_args
numactl -C $cpus -m $node -- python3 all_postprocess.py $base_log.log $new_log.log -0.05 $bin_dir check_fast $common_args 
