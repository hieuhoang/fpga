#pragma once

#ifndef EMULATOR
#define EMULATOR 0
#endif

#define VOCABSIZE 384 //85120 //384  //good multiple of 16 and 128
#define LAYER_DIM 512 // assuming to be multiple of 16
#define MAXBATCH 1000

#define P 16 //should be multiple 16 for B loading logic to work
#define TILECOUNT (VOCABSIZE / P) //VOCABSIZE will be a good multiple of P 

#define WLOADTIME ((P * LAYER_DIM) >> 4) //using float16
#define BLOADTIME (P >> 4) //using float16

struct MaxY
{
	float value;
  unsigned index;
};

