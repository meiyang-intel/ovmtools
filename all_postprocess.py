import os
import sys
import subprocess
import compare_vis as compare_vis

# new_log base_log low_thresh ov_bench_dir
new_log = sys.argv[1]
base_log = sys.argv[2]
thresh = float(sys.argv[3])
ov_bench_dir = sys.argv[4]
check_good = True if sys.argv[5] == 'check_fast' else False
args = ['-pc']
if len(sys.argv) > 6:
    args += sys.argv[6:]

args = [str(i) for i in args]

def median(path):
    with open(path, 'r') as f:
        c = f.readlines()
    results = []
    results_data = []
    for i in range(len(c) // 3):
        v0 = float(c[0 + 3 * i].split()[0])
        v1 = float(c[1 + 3 * i].split()[0])
        v2 = float(c[2 + 3 * i].split()[0])
        #v = v0 + v1 + v2 - max([v0, v1, v2]) - min([v0, v1, v2])
        v = max([v0, v1, v2])
        results.append(f'{v:.2f},{c[0 + 3 * i].split()[1]}')
        results_data.append((v, c[0 + 3 * i].split()[1]))
    with open(f'{path}.max.csv', 'w') as f:
        f.write('\n'.join(results))
    return results_data
# 1, filter the performance data
result_new = median(new_log)
result_base = median(base_log)

# 2, filter the desired sets
assert(len(result_new) == len(result_base))
result_sets = []
for i in range(len(result_new)):
    if result_new[i][0] < 0 or result_base[i][0] < 0:
        print(f'{result_new[i][1]} has no fps, skipped.')
        continue
    if (result_new[i][0] - result_base[i][0]) / result_base[i][0] < thresh:
        # idx, name, new_fps, base_fps
        result_sets.append((i, result_new[i][1], result_new[i][0], result_base[i][0]))

# 3, run compare tool
detail_f = open(f'{new_log}.layer.csv', 'w')
detail_f.write('name,new_fps,base_fps,ratio,delta(ms),layer1,time1(ms),,,,,,,,,,\n')
for (i, name, new_fps, base_fps) in result_sets:
    os.environ['USE_BRG'] = '0'
    outputA = subprocess.run([f'{ov_bench_dir}/benchmark_app', '-m', name] + args + ['-report_folder=./b'], capture_output=True)
    out = outputA.stdout.decode()
    if outputA.returncode == 0:    
        with open('test1.log', 'w') as f:
            f.write(out)
    else:
        raise Exception(f'exec bench error, model:{name}\n error: {out}')
    os.environ['USE_BRG'] = '1'
    outputA = subprocess.run([f'{ov_bench_dir}/benchmark_app', '-m', name] + args + ['-report_folder=./a'], capture_output=True)
    out = outputA.stdout.decode()
    if outputA.returncode == 0:
        with open('test2.log', 'w') as f:
            f.write(out)
    else:
        raise Exception(f'exec bench error, model:{name}\n error: {out}')
    result = compare_vis.show_compare_result('test2.log', 'test1.log')
    detail_f.write(f'{name},{new_fps},{base_fps},{(new_fps-base_fps)*100/base_fps:.2f}%,{(-1/new_fps+1/base_fps)*1000},')
    if not check_good:
        result = sorted(result, key=lambda x: x[1])
        for (n, t) in result:
            if t < 0:
                detail_f.write(f'{n},{t},')
    else:
        result = sorted(result, key=lambda x: x[1], reverse=True)
        for (n, t) in result:
            if t > 0:
                detail_f.write(f'{n},{t},')
    detail_f.write('\n')