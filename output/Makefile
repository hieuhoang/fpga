
FPGA_INCLUDE_FLAGS:=$(shell aocl compile-config)
FPGA_LINK_FLAGS:=$(shell aocl link-config)

all: emu rtl 

emu: 
	$(MAKE) -C kernels emu
	g++ -std=c++11 main.cpp kernel.cpp debug-devices.cpp fpga-matrix.cpp host-matrix.cpp -lboost_timer -lboost_chrono -lboost_system $(FPGA_INCLUDE_FLAGS) $(FPGA_LINK_FLAGS) -Wl,--no-as-needed -lalteracl -lelf

rtl: 
	$(MAKE) -C kernels rtl

fpga: 
	$(MAKE) -C kernels fpga
	g++ -std=c++11 main.cpp kernel.cpp debug-devices.cpp fpga-matrix.cpp host-matrix.cpp -lboost_timer -lboost_chrono -lboost_system $(FPGA_INCLUDE_FLAGS) $(FPGA_LINK_FLAGS) -Wl,--no-as-needed -lalteracl -lelf

emu-cuda:
	$(MAKE) -C kernels emu
	nvcc -std=c++11 -c cuda-code.cu -o cuda-code.o
	g++ -std=c++11 main.cpp kernel.cpp debug-devices.cpp fpga-matrix.cpp host-matrix.cpp cuda-code.o -lboost_timer -lboost_chrono -lboost_system -lcuda -lcudart -lcublas $(FPGA_INCLUDE_FLAGS) $(FPGA_LINK_FLAGS) -Wl,--no-as-needed -lalteracl -laltera_s5_ref_mmd -lelf -DUSE_CUDA

emu-nocl:
	nvcc -std=c++11 -c cuda-code.cu -o cuda-code.o
	g++ -std=c++11 main.cpp host-matrix.cpp cuda-code.o -lboost_timer -lboost_chrono -lboost_system -lcuda -lcudart -lcublas -DUSE_CUDA -DNO_CL 
