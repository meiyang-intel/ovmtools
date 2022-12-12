#!/usr/bin/python3

import sys, os
import argparse
import os.path
import json

my_verbose_converter = None
if 'VERBOSE_CONVERT' in os.environ:
    sys.path.append(os.environ['VERBOSE_CONVERT'])
    try:
        import verbose_converter

        my_verbose_converter = verbose_converter.convert
    except Exception as e:
        print(e)
        pass

exec_graph_A = ''
exec_graph_B = ''
args = None


def find_layout(exec_graph, name):
    if (name.endswith("...")):
        tag = 'originalLayersNames="{}'.format(name.rstrip("..."))
    else:
        tag = 'originalLayersNames="{}'.format(name)
    layout = 'outputLayouts="'
    found = 0
    ret = "?"
    for l in exec_graph:
        if tag in l:
            sl = l[l.index(layout) + len(layout):]
            found += 1
            ret = sl[0:sl.index('"')]

    if found > 1:
        ret = "?"
    return ret


pc_log_start_tag = "[ INFO ] Performance counts for 0-th infer request:"
pc_log_end_tag = "Total time:"


def analyse(log_file, json_dir):
    pc_by_type = {}
    pc_by_node = {}
    statis_by_type = {}

    stat = []
    verbose_by_name = {}
    layers = None
    if os.path.isfile(f'{json_dir}/benchmark_detailed_counters_report.json'):
        with open(f'{json_dir}/benchmark_detailed_counters_report.json', "r") as f:
            layers = json.loads(f.read())

    def append_to_result(run, layer_type, realTime, cpuTime, execType, name):
        if (run == "NOT_RUN"):
            return
        node_type = layer_type + "_" + execType

        if node_type.startswith('Convolution_brgconv_avx512'):
            node_type = node_type.replace('brgconv', 'brg.jit')
        elif node_type.startswith('Convolution_jit_avx512'):
            node_type = node_type.replace('jit', 'brg.jit')
        elif node_type.startswith('Convolution_jit_gemm_FP32'):
            node_type = 'Convolution_brg.jit_avx512_FP32'
        if node_type.startswith('GroupConvolution_ref_any_FP32'):
            node_type = 'GroupConvolution_any.brg.jit.gemm_FP32'
        elif node_type.startswith('GroupConvolution_brgconv_avx512') and node_type.endswith('_FP32'):
            node_type = 'GroupConvolution_any.brg.jit.gemm_FP32'
        elif node_type.startswith('GroupConvolution_jit_avx512_FP32'):
            node_type = 'GroupConvolution_any.brg.jit.gemm_FP32'
        elif node_type.startswith('GroupConvolution_jit_gemm_FP32'):
            node_type = 'GroupConvolution_any.brg.jit.gemm_FP32'

        if not node_type in pc_by_type:
            pc_by_type[node_type] = [0, 0]  # cnt, total
        if layer_type not in statis_by_type:
            statis_by_type[layer_type] = [0, 0]

        pc_by_type[node_type][0] += 1
        pc_by_type[node_type][1] += int(realTime)
        statis_by_type[layer_type][0] += 1
        statis_by_type[layer_type][1] += int(realTime)

        pc_by_node[name] = [int(realTime), layer_type, execType]

    with open(log_file, "r") as f:
        start = False
        for l in f.readlines():
            if 'verbose##' in l:
                items = l.split('##')
                verbose_by_name[items[1]] = items[2].strip()
                continue
            if l.startswith("	Percent of CPU this job got") or \
                    l.startswith("	Maximum resident set size (kbytes)") or \
                    l.startswith("	User time (seconds)"):
                stat.append(l.strip("\t"))
                continue
            if l.startswith("[ INFO ] 	Average:") or \
                    l.startswith("[ INFO ] Throughput:") or \
                    l.startswith("[ INFO ]    Average:"):
                stat.append(l[9:].strip(" ").strip("\t"))
                continue

            if l == '\n' or layers: continue
            if l.startswith(pc_log_start_tag):
                start = True
                continue
            if start:
                if l.startswith(pc_log_end_tag):
                    start = False
            if start:
                name = l[:30].rstrip(" ")
                run, _, layer_type, _, realTime, _, cpuTime, _, execType = l[30:].split()
                append_to_result(run, layer_type, realTime, cpuTime, execType, name)

    if layers:
        for layer in layers['detailed_performance'][0]['nodes']:
            append_to_result(layer['status'], layer['node_type'], \
                             layer['real_time'] * 1000, layer['cpu_time'] * 1000, layer['exec_type'], layer['name'])

    pc_by_node = sorted(pc_by_node.items(), key=lambda d: d[1][0], reverse=True)
    pc_by_type = sorted(pc_by_type.items(), key=lambda d: d[1][1], reverse=True)
    statis_by_type = sorted(statis_by_type.items(), key=lambda d: d[1][1], reverse=True)
    return pc_by_node, pc_by_type, stat, verbose_by_name, statis_by_type


def smart_val(v):
    if abs(v) > 1000000:
        return "{:.1f}M".format(v/1000000)
    if abs(v) > 1000:
        return "{:.1f}K".format(v/1000)
    return v


def show_compare_result(log_file_A, log_file_B, all_dict, prefixA, prefixB, reportA, reportB):
    pc_by_node0, pc_by_type0, stat0, verbose_by_name0, statis_by_type0 = analyse(log_file_A, reportA)
    if prefixB:
        pc_by_node1, pc_by_type1, stat1, verbose_by_name1, statis_by_type1 = analyse(log_file_B, reportB)
    else:
        pc_by_node1 = []
        pc_by_type1 = []
        stat1 = []
        verbose_by_name1 = ""
        statis_by_type1 = []
    print("{}   :    {}".format(log_file_A, log_file_B))

    print("*********************************************************")
    print("*                   comparing by node                   *")
    print("*********************************************************")
    # collect all type names
    all_names = list(set([t for t, _ in pc_by_node0])|set([t for t, _ in pc_by_node1]))

    total_time0 = total_time1 = 0
    layerid = 0
    for name in all_names:
        layer_dict={}
        layer_info = []
        A_info = []
        B_info = []
        time_diff = 0
        info0, time0, layer0, exectype0, layout0 = get_exec_info(pc_by_node0, name)  # prefixA
        info1, time1, layer1, exectype1, layout1 = get_exec_info(pc_by_node1, name)  # prefixB
        layer_info = [layerid, name]
        A_info = [layer0, exectype0, layout0, time0]
        B_info = [layer1, exectype1, layout1, time1]
        node_type = ""
        show_verbose= True
        if node_type in info0 or node_type in info1:
            total_time0 += time0
            total_time1 += time1
            time_diff = smart_val(time1 - time0)
            print("{:>6} {:>50}  {:<50}  {}".format(time_diff, info0, info1, name))

            if show_verbose:
                verbose0, benchdnn0 = get_print_info(name, verbose_by_name0)
                verbose1, benchdnn1 = get_print_info(name, verbose_by_name1)
                A_info += [verbose0, benchdnn0]
                B_info += [verbose1, benchdnn1]
        layerid += 1
        dict_data = JsonData(prefixA, prefixB)
        layer_dict = dict_data.layer(layer_info, A_info, B_info, time_diff)
        all_dict.update(layer_dict)


    print("")
    print("{:>6} {:>50}   {:<50}   {}".format(smart_val(total_time1 - total_time0), total_time0,total_time1, "Totals"))

    print("")
    for i in range(len(stat0)):
        s0 = stat0[i].rstrip("\n").rstrip("\r")
        if stat1: s1 = stat1[i].rstrip("\n").rstrip("\r")
        else: s1 = "None"
        print("{:>50}   {:<50} ".format(s0, s1))

    return all_dict


def get_exec_info(pc_by_node, name):
    def find(pclist, type_name):
        for name, v in pclist:
            if name == type_name:
                return v
        return None
    v0 = find(pc_by_node, name)
    if v0:
        time0, layer0, exectype = v0
        layout = find_layout(exec_graph_A, name)
        info0 = "{}_{}_{} {:6.1f}".format(layer0, exectype, layout, time0)
    else:
        time0 = 0
        info0 = "---"
        layer0 = ""
        exectype = ""
        layout = ""
    return info0, time0, layer0, exectype, layout


def get_print_info(name, verbose_by_name0):
    verbose = ""
    benchdnn = ""
    if name in verbose_by_name0:
        verbose = 'onednn_verbose,exec,' + verbose_by_name0[name]
        print(verbose)
        all_verbose = \
f'''
onednn_verbose,info,prim_template:operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time
{verbose},1.7478
'''
        if my_verbose_converter:
            status, output = my_verbose_converter(verbose_level=0, parser='oneDNN',
                                                  input=all_verbose.splitlines(), action='generate',
                                                  generator='benchdnn', split_output=False, agg_keys=None)
            if output != None:
                for key, value in output.items():
                    benchdnn = f"./benchdnn --fix-times-per-prb=100 --mode=p {value}"
                    print(f"./benchdnn --fix-times-per-prb=100 --mode=p {value}", end='')
    return verbose, benchdnn


def write_json_data(all_dict, output_path, model):
    name = get_output_name(output_path, model)
    json_str = json.dumps(all_dict, indent=2)
    with open(os.path.join(output_path, name + ".json"), "w+") as f:
        f.write(json_str)


def get_output_name(output_path, model):
    path = model.split("/")
    name = path[-1].split(".")[0]
    while (os.path.exists(os.path.join(output_path, name + ".json"))):
        name = name + "_1"
    return name


# layer_info = [layer_id, layer_name]
# A_info = [node_type, exec_type, layout, time, onednn_cmd, benchdnn_cmd]
# B_info = [node_type, exec_type, layout, time, onednn_cmd, benchdnn_cmd]
# time_diff the time difference between brgconv and jit


class JsonData:
    def __init__(self, prefixA, prefixB):
        self.jsonData = {}
        self.prefixA = prefixA
        self.prefixB = prefixB

    def layer(self, layer_info, A_info, B_info, time_diff):
        layer_id = "layer" + str(layer_info[0])
        self.jsonData[layer_id] = {}
        self.jsonData[layer_id]["layer_name"] = layer_info[1]
        self.jsonData[layer_id][self.prefixA] = self.exec_node(A_info)
        self.jsonData[layer_id][self.prefixB] = self.exec_node(B_info)
        self.jsonData[layer_id]["time_difference"] = time_diff
        return self.jsonData

    @staticmethod
    def exec_node(info):
        exec_dict = {}
        exec_dict["node_type"] = info[0]
        exec_dict["exec_type"] = info[1]
        exec_dict["layout"] = info[2]
        exec_dict["time"] = info[3]
        exec_dict["onednn_cmd"] = info[4]
        exec_dict["benchdnn_cmd"] = info[5]
        return exec_dict


def main(exec_graph_A, exec_graph_B, model, log_file_A, log_file_B, prefixA, prefixB, reportA, reportB, output_file):
    my_verbose_converter = None
    if 'VERBOSE_CONVERT' in os.environ:
        sys.path.append(os.environ['VERBOSE_CONVERT'])
        try:
            import verbose_converter

            my_verbose_converter = verbose_converter.convert
        except Exception as e:
            print(e)
            pass

    with open(exec_graph_A or './a/exec_graph_A.xml') as f:
        print("prefixA: ", exec_graph_A)
        exec_graph_A = f.readlines()

    if prefixB:
        with open(exec_graph_B or './b/exec_graph_B.xml') as f:
            print("prefixB: ", exec_graph_B)
            exec_graph_B = f.readlines()
    else:
        exec_graph_B = ""

    all_dict = {"model_name": model}
    data_cmp = show_compare_result(log_file_A or './testA.log', log_file_B or './testB.log',
                                   all_dict, prefixA, prefixB, reportA, reportB)
    write_json_data(data_cmp, output_file or './output', model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--node_cnt", help="number of nodes to show", default=1000, type=int)
    parser.add_argument("-t", "--node_type", help="node type filter", default="", type=str)
    parser.add_argument("log_file_A", nargs="?")
    parser.add_argument("log_file_B", nargs="?")
    parser.add_argument("exec_graph_A", nargs="?")
    parser.add_argument("exec_graph_B", nargs="?")
    parser.add_argument("-s", "--show_verbose", default=True, help="show onednn verbose",
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("-m", "--model", default="", type=str)
    parser.add_argument("-output_file", default="", type=str)
    parser.add_argument("-rA", "--reportA", help="report folderA", default="./a", type=str)
    parser.add_argument("-rB", "--reportB", help="report folderB", default="./b", type=str)
    parser.add_argument("-pA", "--prefixA", help="prefixA", default="brg", type=str)
    parser.add_argument("-pB", "--prefixB", help="prefixB", default="jit", type=str)
    args = parser.parse_args()

    main(args.exec_graph_A, args.exec_graph_B, args.model, args.log_file_A, args.log_file_B,
         args.prefixA, args.prefixB, args.reportA, args.reportB, args.output_file)

    print("finish")
