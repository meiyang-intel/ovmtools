# OpenVINO model tools


## Performance regression test for CPU plugin

## mount IR cache by nfs

```bash

sudo apt-get install nfs-common

mkdir nfs
sudo mount -t nfs 10.67.108.173:/home/vsi/nfs_share ./nfs

```

## configuration profile -- config.ini
***Tips: If you do not use some non-bool parameters, please set them to empty, do not delete keywords. Such as binB="". If you 
want to set bool parameters, please use `True` or `False`. Pay attention to capitalization.***

### 1. Support single model in the same environment. No comparison

If you only want to display the situation under the fixed environment variables of a model, please configure as follows
```
[Mode]
# There is no comparison, only use one binary(binA).
single=True
```

### 2. Support for comparing the performance results of two versions of binary
```
[Mode]
single=False
diff_bin=True

# If you want to filter data
[Filter]
need_filter=True
threshold=-0.05

# If you don't want to filter model, you want to show all the model data
[Filter]
need_filter=True
threshold=0
```
### 3. Support for comparing the performance results of two different environments
```
[Mode]
single=False
# Compare different env, such as BRG
diff_env=True

# split by =
envA=USE_BRG=1
envB=USE_BRG=0
```
 - using the same binary
```
[BIN]
binA = /home/openvino/bin/intel64/Debug/
binB = ""
```
 - using different binaries
```
# binA corresponds to envA
# binB corresponds to envB
[BIN]
binA = /home/1/openvino/bin/intel64/Debug/ 
binB = /home/2/openvino/bin/intel64/Debug/
 ```
### 4. Some parameters
`need_run_benchdnn`:  If you want to run the benchdnn command in the linux command interface, 
please set this parameter to True. The premise is that there is a print verbose in the benchmark_app

## run tests
After setting the config.ini
```
python3 main.py
```

