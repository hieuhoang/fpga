all: emu rtl 

emu: 
	aoc -march=emulator -v -DEMULATOR=1 OutputLayer.cl -o OutputLayer.aocx

rtl: 
	aoc -c -v -fpc -fp-relaxed OutputLayer.cl -o OutputLayer.aoco

fpga: 
	aoc -v -fpc -fp-relaxed OutputLayer.cl -o OutputLayer.aocx -profile -incremental

