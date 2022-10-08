@set binA=%cd%\../openvino/bin/intel64/RelWithDebInfo
rem @set binB=%cd%\../openvino-tmp/bin/intel64/RelWithDebInfo
@set binB=%binA%
@set VERBOSE_CONVERT=%cd%/../openvino/src/plugins/intel_cpu/thirdparty/onednn/scripts/verbose_converter

@set args=-niter 100 -b 1 -nireq=1 -nstreams 1 -nthreads 4 -pc -infer_precision f32 -json_stats -report_type=detailed_counters
@md %cd%\a
@md %cd%\b

@set cur=%cd%
@set ONEDNN_VERBOSE=2
@cd %binB%
@set OV_CPU_DEBUG_LOG=CreatePrimitives;conv.cpp;deconv.cpp
@set USE_BRG=0
benchmark_app -m %1 %args% -exec_graph_path %cur%\exec_graph_B.xml -report_folder=%cur%/b > %cur%\pcB.txt
@cd %binA%
@set USE_BRG=1
benchmark_app -m %1 %args% -exec_graph_path %cur%\exec_graph_A.xml -report_folder=%cur%/a > %cur%\pcA.txt
@cd %cur%

@python compare_vis.py pcA.txt pcB.txt