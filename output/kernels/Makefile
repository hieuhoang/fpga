all: emu rtl 

emu: 
	aoc -march=emulator -v -DEMULATOR=1 OutputLayer.cl -o fpga.aocx

rtl: 
	aoc -c -v -fpc -fp-relaxed OutputLayer.cl

fpga: 
	aoc -v -fpc -fp-relaxed OutputLayer.cl -profile -incremental
