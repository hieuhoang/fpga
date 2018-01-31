
FPGA_INCLUDE_FLAGS=`aocl compile-config`
FPGA_LINK_FLAGS=`aocl link-config`

# no cuda
#g++ -std=c++11 main.cpp kernel.cpp debug-devices.cpp matrix.cpp host-matrix.cpp $FPGA_INCLUDE_FLAGS $FPGA_LINK_FLAGS -Wl,--no-as-needed -lalteracl -lelf

# with cuda
nvcc -std=c++11 -c cuda-code.cu -o cuda-code.o
g++ -std=c++11 main.cpp kernel.cpp debug-devices.cpp matrix.cpp host-matrix.cpp cuda-code.o -lcuda -lcudart -lcublas -I/home/hieu/intelFPGA/17.1/hld/host/include -L/home/hieu/intelFPGA/17.1/hld/board/s5_ref/linux64/lib -L/home/hieu/intelFPGA/17.1/hld/host/linux64/lib -Wl,--no-as-needed -lalteracl -laltera_s5_ref_mmd -lelf -DUSE_CUDA

