import os
import torch
try:
    import torch_npu
    npu_available = True
except:
    npu_available = False
    print(f"NPU Not Available")

__all__ = ['profiling']

class Profiling():
    def __init__(self, wait=1, warmup=1, active=1, repeat=1, skip_first=0):
        self.profiling_dir = os.getenv("PROFILING_DIR", "./prof")
        if int(os.getenv("PROFILING_LEVEL", "0")) == 1:
            self.profiling_level = torch_npu.profiler.ProfilerLevel.Level1
        elif int(os.getenv("PROFILING_LEVEL", "0")) == 2:
            self.profiling_level = torch_npu.profiler.ProfilerLevel.Level2
        else:
            self.profiling_level = torch_npu.profiler.ProfilerLevel.Level0
        self.profiling_python_stack = True if int(os.getenv("PROFILING_PYTHON_STACK", "0")) == 1 else False
        
        print(f"Profiling level: {self.profiling_level}")
        print(f"Profiling python stack: {self.profiling_python_stack}")
        print(f"Profiling dir: {self.profiling_dir}")

        if npu_available:
            self._npu_init(wait=wait, warmup=warmup, active=active, repeat=repeat, skip_first=skip_first)
        else:
            self._gpu_init()


    def _gpu_init(self):
        self._prof = None
        pass
    
    def _npu_init(self, wait=1, warmup=1, active=1, repeat=1, skip_first=0):
        if os.getenv("PROFILING_ENABLE", "0") == "1":
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                export_type=torch_npu.profiler.ExportType.Text,
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=self.profiling_level,
                l2_cache=False,
                data_simplification=False
            )
            self._prof = torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                with_stack=self.profiling_python_stack,
                record_shapes=True,
                profile_memory=False,
                schedule=torch_npu.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat, skip_first=skip_first),
                experimental_config=experimental_config,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(self.profiling_dir)
            )
        else:
            self._prof = None

    def start(self):
        if self._prof:
            self._prof.__enter__()
        else:
            print('Start Profiling Failed. Profiling not init')

    def step(self):
        if self._prof:
            self._prof.step()
        else:
            print('Profiling Step Failed. Profiling not init')

    def stop(self):
        if self._prof:
            self._prof.__exit__(None, None, None)
        else:
            print('Stop Profiling Failed. Profiling not init')