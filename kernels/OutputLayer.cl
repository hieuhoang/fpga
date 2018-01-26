#ifndef EMULATOR
#define EMULATOR 0
#endif

#define VOCABSIZE 384  //good multiple of 16 and 128
#define LAYER_DIM 512 // assuming to be multiple of 16
#define MAXBATCH 1000

#define P 16 //should be multiple 16 for B loading logic to work
#define TILECOUNT (VOCABSIZE / P) //VOCABSIZE will be a good multiple of P 

#define WLOADTIME ((P * LAYER_DIM) >> 4) //using float16
#define BLOADTIME (P >> 4) //using float16

struct MaxY_type
{
	float MaxVal;
    unsigned index;
};

__attribute__((max_global_work_dim(0)))
__kernel void OutputLayer_float(
				__global float * restrict W,
				__global float * restrict X,
				__global float * restrict B,
				__global struct MaxY_type * restrict MaxY, 
				unsigned batchsize
                  )
{
#if EMULATOR == 1
    printf("OpenCL: OutputLayer_float, batchsize=%d \n",batchsize);
#endif

	__global volatile float16* restrict ddr_access_pointer;
	__global volatile struct MaxY_type* restrict MaxYpointer;
	__global volatile float16* restrict Wpointer_prev;
	__global volatile float16* restrict Bpointer_prev;

    float16 Wlocal[LAYER_DIM>>4][P]  __attribute__((numbanks(P), bankwidth(64))); //64 bytes = float16 bytesize
    float Blocal[P];


	Wpointer_prev = (__global volatile float16 *)W;
	Bpointer_prev = (__global volatile float16 *)B;
	
	MaxYpointer = (__global volatile struct MaxY_type *)MaxY;
	struct MaxY_type MaxYlocal[MAXBATCH];
	#pragma unroll 1
	for (short mi=0; mi < batchsize; mi++){
		MaxYlocal[mi].MaxVal=-MAXFLOAT;
		MaxYlocal[mi].index=VOCABSIZE+100; //you can get rid of this later
	}
#if EMULATOR == 1
    printf("OpenCL: MaxY values initialized to %f \n",-MAXFLOAT);
#endif	
	
	for (unsigned tile=0; tile < TILECOUNT; tile++) {
#if EMULATOR == 1
    printf("OpenCL: tile=%d \n",tile);
#endif
		ddr_access_pointer = (__global volatile float16 *)Wpointer_prev;
	
		unsigned wr_index=0;
		//fetch W and B to local
		for (unsigned i=0; i < (WLOADTIME + BLOADTIME); i++) {
	
			float16 temp_val = *ddr_access_pointer;
			if (i < WLOADTIME) {
				//#pragma unroll 
				//for (char u=0; u < 16; u++) {
				//	Wlocal[wr_index >> 5][(wr_index & 0x1F)*16+u]=temp_val[u]; // good for LAYER_DIM 512 (512/16=32)
				//}
				Wlocal[wr_index & 0x1F][wr_index >> 5]=temp_val;
				wr_index++;
			}
			else {
				#pragma unroll 
				for (char u=0; u < 16; u++) {
					Blocal[wr_index*16+u]=temp_val[u]; // good for P as a multiple of 16
				}		
				wr_index++;
			}
			ddr_access_pointer++;
		
			if (i==(WLOADTIME-1)) { //we should keep track of W for the next batch
				Wpointer_prev = ddr_access_pointer;
				ddr_access_pointer = (__global volatile float16 *)Bpointer_prev; //would byte aligning be a problem?
				wr_index = 0;
			}
		}
		
		//do the matrix multiplication of tile with X
		//read 16 numbers from a column, multiply with 16 numbers from a row of W to make one Y output and then add b and write back
		__global volatile float16* restrict Xpointer;
		Xpointer = (__global volatile float16 *)X;
		for (unsigned xj=0; xj < batchsize; xj++) { 
#if EMULATOR == 1
      //printf("OpenCL: xj=%d \n",xj);
#endif
			float ylocal[P]; //non-initialized
			for (short pr=0; pr < P; pr++) { 
				ylocal[pr]=0.0f;
			}
			//#pragma max_concurreny 1
			for (unsigned xi=0; xi < LAYER_DIM>>4; xi++) { //read 16 numbers at a time
				float16 xval= *Xpointer;
				#pragma unroll
				for (short pi=0; pi < P; pi++) { 
					#pragma unroll
					for (char u=0; u < 16; u++) {
						ylocal[pi] += xval[u]*Wlocal[xi][pi][u];
					}
				}
				Xpointer++;
			}
			//now you have P instances of Y elements ready from the same column xj
			float16 yaddb[P>>4];
			#pragma unroll 1
			for (short pb=0; pb < P>>4; pb++) {
#if EMULATOR == 1
        //printf("OpenCL: pb=%d \n",pb);
#endif
				#pragma unroll
				for (char u=0; u < 16; u++) {
					yaddb[pb][u]=ylocal[pb*16+u] + Blocal[pb*16+u];
				}				 
				//*Ypointer = yaddb[pb];
				//Ypointer++;
				#pragma unroll 1
				for (char u=0; u < 16; u++) {
					if (yaddb[pb][u] > MaxYlocal[xj].MaxVal) {
						MaxYlocal[xj].MaxVal = yaddb[pb][u];
						MaxYlocal[xj].index  = tile * P + pb*16 + u;
					}
				}
			}
		}
	
	}
		

}
	
 

