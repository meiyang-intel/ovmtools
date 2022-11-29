import compare_vis as compare_vis


class FilterModel():
    def __init__(self, logA, logB, fpsA, fpsB, config):
        self.logA = logA
        self.logB = logB
        self.fpsA = fpsA
        self.fpsB = fpsB
        self.config = config

    def median(self, fps_res):
        v0 = float(fps_res[0].split()[0])
        v1 = float(fps_res[1].split()[0])
        v2 = float(fps_res[2].split()[0])
        # v = v0 + v1 + v2 - max([v0, v1, v2]) - min([v0, v1, v2])
        v = max([v0, v1, v2])
        results_data = (v, fps_res[0].split()[1])
        return results_data

    def filter_res(self):
        fpsA_median = self.median(self.fpsA)
        fpsB_median = self.median(self.fpsB)
        ratio = (fpsA_median[0] - fpsB_median[0]) / fpsB_median[0]
        geomean = fpsA_median[0] / fpsB_median[0]
        results_lst = ','.join([f'{fpsA_median[1]},{fpsA_median[0]:.2f},{fpsB_median[0]:.2f}, {ratio:.2f}, {geomean:.2f}'])
        result_sort_sets = []
        # model,prefixA,prefixB,ratio (A-B)/B,geomean
        if fpsA_median[0] < 0 or fpsB_median[0] < 0:
            print(f'{fpsA_median[1]} has no fps, skipped.')
        if (fpsA_median[0] - fpsB_median[0]) / fpsB_median[0] < float(self.config['Filter']['threshold']):
            # model_name, prefixA_fps, prefixB_fps
            result_sort_sets = (fpsA_median[1], fpsA_median[0], fpsB_median[0])
        return results_lst, result_sort_sets

    def write_benchamrk_temp_log(self, file, benchmark_log):
        with open(file, 'w') as f:
            f.write(benchmark_log)

    def run_compare_tool(self, result_sort_sets, save_path, reportA, reportB):
        self.write_benchamrk_temp_log(f'{save_path}/testA.log', self.logA)
        self.write_benchamrk_temp_log(f'{save_path}/testB.log', self.logB)
        result = compare_vis.show_compare_result(f'{save_path}/testA.log', f'{save_path}/testB.log', reportA, reportB)
        return result
