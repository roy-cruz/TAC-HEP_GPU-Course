# COMP
CXX      := g++
NVCC     := nvcc
CXXFLAGS := -O3 -g -I ./hh
NVCCFLAGS := -O3 -I ./hh
ALPAKA_INCLUDE := $(HOME)/public/alpaka/include
ALPAKA_FLAGS   := -x cu -expt-relaxed-constexpr -std=c++20 -O3 -g \
                  -I $(ALPAKA_INCLUDE) -D ALPAKA_ACC_GPU_CUDA_ENABLED \
                  -I ./hh -Wno-deprecated-declarations

OBJS := objs

CPU_EXE          := $(OBJS)/stencil_mult_cpu
CUDA_EXPLICIT_EXE := $(OBJS)/stencil_mult_explicit
CUDA_MANAGED_EXE  := $(OBJS)/stencil_mult_managed
CUDA_OPT_EXE      := $(OBJS)/stencil_mult_opt
ALPAKA_EXE        := $(OBJS)/stencil_mult_alpaka

all: $(CPU_EXE) $(CUDA_EXPLICIT_EXE) $(CUDA_MANAGED_EXE) $(CUDA_OPT_EXE) $(ALPAKA_EXE)

$(CPU_EXE): ex1-cpu/stencil_mult.cpp | $(OBJS)
	$(CXX) $< -o $@ $(CXXFLAGS)

$(CUDA_EXPLICIT_EXE): ex2-cuda/stencil_mult_explicit.cu | $(OBJS)
	$(NVCC) $< -o $@ $(NVCCFLAGS)

$(CUDA_MANAGED_EXE): ex2-cuda/stencil_mult_managed.cu | $(OBJS)
	$(NVCC) $< -o $@ $(NVCCFLAGS)

$(CUDA_OPT_EXE): ex3-cudaopt/stencil_mult_opt.cu | $(OBJS)
	$(NVCC) $< -o $@ $(NVCCFLAGS)

$(ALPAKA_EXE): ex4-alpaka/stencil_mult_alpaka.cpp | $(OBJS)
	$(NVCC) $< -o $@ $(ALPAKA_FLAGS)

$(OBJS):
	mkdir -p $(OBJS)

# PROF
PROF_DIR := profiling
VTUNE_RESULT := $(PROF_DIR)/cpu_hotspots

PROFILE_TARGET_cpu      := $(CPU_EXE)
PROFILE_TARGET_explicit := $(CUDA_EXPLICIT_EXE)
PROFILE_TARGET_managed  := $(CUDA_MANAGED_EXE)
PROFILE_TARGET_opt      := $(CUDA_OPT_EXE)
PROFILE_TARGET_alpaka   := $(ALPAKA_EXE)

PROFILE_TOOL_cpu      := vtune
PROFILE_TOOL_explicit := nsys
PROFILE_TOOL_managed  := nsys
PROFILE_TOOL_opt      := nsys
PROFILE_TOOL_alpaka   := nsys

profile: $(PROF_DIR)
ifeq ($(PROFILE_TARGET_$(name)),)
	$(error Unknown profile target '$(name)'. Valid options: cpu explicit managed opt alpaka)
endif

	@echo "Profiling target: $(name)"
	@echo "Executable: $(PROFILE_TARGET_$(name))"
	@echo "Profiler: $(PROFILE_TOOL_$(name))"

ifeq ($(PROFILE_TOOL_$(name)),vtune)
	vtune -collect hotspots -result-dir $(PROF_DIR)/vtune_$(name) $(PROFILE_TARGET_$(name))
	vtune -report hotspots -result-dir $(PROF_DIR)/vtune_$(name) -format=csv > $(PROF_DIR)/profile_$(name).csv
else
	nsys profile --trace=cuda,osrt -o $(PROF_DIR)/nsys_$(name) $(PROFILE_TARGET_$(name))

	@echo "Exporting CSV for $(name)..."

	nsys stats --report osrt_sum --format csv \
	    -o $(PROF_DIR)/profile_$(name)_osrt.csv \
	    $(PROF_DIR)/nsys_$(name).nsys-rep

	nsys stats --report cuda_api_sum --format csv \
	    -o $(PROF_DIR)/profile_$(name)_apisum.csv \
	    $(PROF_DIR)/nsys_$(name).nsys-rep

	nsys stats --report cuda_gpu_kern_sum --format csv \
	    -o $(PROF_DIR)/profile_$(name)_kernsum.csv \
	    $(PROF_DIR)/nsys_$(name).nsys-rep

	nsys stats --report cuda_gpu_mem_time_sum --format csv \
	    -o $(PROF_DIR)/profile_$(name)_memsum.csv \
	    $(PROF_DIR)/nsys_$(name).nsys-rep
endif

	@echo "Profiling for $(name) complete."

$(PROF_DIR):
	mkdir -p $(PROF_DIR)

run_all_profiles: clean_all all $(PROF_DIR)
	$(MAKE) profile name=cpu
	$(MAKE) profile name=explicit
	$(MAKE) profile name=managed
	$(MAKE) profile name=opt
	$(MAKE) profile name=alpaka
	
# CLEAN
clean_objs:
	rm -rf $(OBJS)

clean_prof:
	rm -rf $(PROF_DIR)

clean_all:
	rm -rf $(OBJS) $(PROF_DIR)

.PHONY: all clean