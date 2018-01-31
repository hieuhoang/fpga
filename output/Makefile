
FPGA_INCLUDE_FLAGS:=$(shell aocl compile-config)
FPGA_LINK_FLAGS:=$(shell aocl link-config)

all: emu rtl 

emu: 
	$(MAKE) -C kernels emu
	g++ -std=c++11 main.cpp kernel.cpp debug-devices.cpp fpga-matrix.cpp host-matrix.cpp $(FPGA_INCLUDE_FLAGS) $(FPGA_LINK_FLAGS) -Wl,--no-as-needed -lalteracl -lelf

rtl: 
	$(MAKE) -C kernels rtl

fpga: 
	$(MAKE) -C kernels fpga
	g++ -std=c++11 main.cpp kernel.cpp debug-devices.cpp fpga-matrix.cpp host-matrix.cpp $(FPGA_INCLUDE_FLAGS) $(FPGA_LINK_FLAGS) -Wl,--no-as-needed -lalteracl -lelf

emu-cuda:
	$(MAKE) -C kernels emu
	nvcc -std=c++11 -c cuda-code.cu -o cuda-code.o
	g++ -std=c++11 main.cpp kernel.cpp debug-devices.cpp fpga-matrix.cpp host-matrix.cpp cuda-code.o -lcuda -lcudart -lcublas $(FPGA_INCLUDE_FLAGS) $(FPGA_LINK_FLAGS) -Wl,--no-as-needed -lalteracl -laltera_s5_ref_mmd -lelf -DUSE_CUDA

