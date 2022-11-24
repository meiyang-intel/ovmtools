import os


cpus="4,5"
node=0
binB="/home/ruiqiyang/disk/openVINO/openvino/bin/intel64/Release/"
binA="/home/ruiqiyang/disk/openVINO/test_cc_openvino/openvino/bin/intel64/Debug/"
threshold = "-0.05"
# FP32
# multi-models: If you want to test all models in a folder, pls enter the path to the folder
# such as /home/model/cv_bench_cache/try_builds_cache/sk_8oct_100models_22.3.0-8234-c83ad806
# single-models: If you only want to test single model, pls enter the absolute path to the model's xml file
# such as /home/sk_8oct_100models_22.3.0-8234-c83ad806/resnet-50/caffe/caffe/FP32/1/dldt/resnet-50.xml
fp32_model_path="/home/ruiqiyang/disk/openVINO/ovmtools/fp32_test_model"
fp32_save_path=os.path.join(os.getcwd(), "fp32_output")
# prefixes included in the output file
fp32_binA_prefix="brg.f32" # -> using binA or using envA
fp32_binB_prefix="jit.f32" # -> using binB or using envB
fp32_common_args=" -infer_precision=f32 -hint none -t=5 -b=1 -nireq=1 -nstreams=1 -nthreads=4 -json_stats -report_type=detailed_counters"

# INT8
int8_model_path="/home/ruiqiyang/disk/openVINO/ovmtools/int8_test_model"
int8_save_path=os.path.join(os.getcwd(), "int8_output")
int8_binA_prefix="brg.i8"
int8_binB_prefix="jit.i8"
int8_common_args=fp32_common_args

# Mode
# Filtering models with reduced FPS
need_filter=True
# Running benchdnn cmd in Linux cmd
need_run_benchdnn=True
# Compare the same model data using different binaries.
diff_bin=True
# Compare different env, such as BRG
diff_env=True
envA="USE_BRG=1" # split by =
envB="USE_BRG=0" # split by =