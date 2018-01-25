g++ -std=c++11 main.cpp kernel.cpp -I/home/hieu/intelFPGA/17.1/hld/host/include -L/home/hieu/intelFPGA/17.1/hld/board/a10_ref/linux64/lib -L/home/hieu/intelFPGA/17.1/hld/host/linux64/lib -Wl,--no-as-needed -lalteracl -laltera_a10_ref_mmd -lelf

