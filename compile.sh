g++ -std=c++11 main.cpp kernel.cpp debug-devices.cpp matrix.cpp host_matrix.cpp -I/home/hieu/intelFPGA/17.1/hld/host/include -L/home/hieu/intelFPGA/17.1/hld/board/s5_ref/linux64/lib -L/home/hieu/intelFPGA/17.1/hld/host/linux64/lib -Wl,--no-as-needed -lalteracl -laltera_s5_ref_mmd -lelf


