#include <cstdint>
#include "stdio.h"
#include "math.h"
__device__ float avgLum;
__global__ void tonemap_kernel(std::uint32_t* out, const float* in, int width, int height, float exposure)
{
	
	int x = (blockIdx.x*blockDim.x + threadIdx.x)*4;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
//	if(x==0 and y==0)
//		printf("Hello CUda\n");
	if(y< height && x <= width*4-4){
		float input_tone[3] = {in[y*width*4+x] * exposure,in[y*4*width+x+1]*exposure,in[y*4*width+x+2]*exposure};
		float output_tone[3];
		output_tone[0] = input_tone[0]*(0.9036*input_tone[0] + 0.018)/(input_tone[0]*(0.8748*input_tone[0]+0.354)+0.14);
		output_tone[1] = input_tone[1]*(0.9036*input_tone[1] + 0.018)/(input_tone[1]*(0.8748*input_tone[1]+0.354)+0.14);
		output_tone[2] = input_tone[2]*(0.9036*input_tone[2] + 0.018)/(input_tone[2]*(0.8748*input_tone[2]+0.354)+0.14);
		
		output_tone[0] = (output_tone[0] <= 0.0031308 ? 12.92 * output_tone[0] : 1.055 * powf(output_tone[0],1/2.4) - 0.055);
		output_tone[1] = (output_tone[1] <= 0.0031308 ? 12.92 * output_tone[1] : 1.055 * powf(output_tone[1],1/2.4) - 0.055);
		output_tone[2] = (output_tone[2] <= 0.0031308 ? 12.92 * output_tone[2] : 1.055 * powf(output_tone[2],1/2.4) - 0.055);

		output_tone[0] = (output_tone[0]>1 ? 255 : rintf(255*output_tone[0]));
		output_tone[1] = (output_tone[1]>1? 255 : rintf(255*output_tone[1]));
		output_tone[2] = (output_tone[2]>1? 255 : rintf(255*output_tone[2]));

		out[y*width+x/4] = 0 |(std::uint8_t)(output_tone[2]) <<16 | (std::uint8_t)(output_tone[1]) <<8 | (std::uint8_t)(output_tone[0]);
	}
	
	
}
__global__ void getAvgLum(const float* in, float exposure){
	__shared__ float ins[16*16];
	int col = (blockIdx.x*blockDim.x + threadIdx.x)*4;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	if(row< height && col <= width*4-4){
		ins[threadIdx.y*blockDim.x+threadIdx.x] = logf(0.2126*in[row*4*width+col]+0.7152*in[row*4*width+col+1]+0.0722*in[row*4*width+col+2]);
		unsigned int t = threadIdx.y*blockDim.x+threadIdx.x;
		for(unsigned int stride = blockDim.y*blockDim.x>>1;stride>0;stride>>=1){
			__syncthreads();
			if(t<stride)
				ins[t] += ins[t+stride];
		}
		if(t==0)
			avgLum *= expf(ins[t]/(height*width));
		

	}
}


void tonemap(std::uint32_t* out, const float* in, int width, int height, float exposure, float brightpass_threshold)
{	dim3 blockSize (32,32,1);
	dim3 gridSize (width/blockSize.x+1,height/blockSize.y+1,1);
//	printf("before tonemapkernel\n");	
	tonemap_kernel<<<gridSize, blockSize>>>(out, in, width, height, exposure);
	cudaDeviceSynchronize();
//	printf("after tonemapkernel\n");
}
void getAvgLum(const float* in,int width,int height,float exposure){
	float value = 1;
	throw_error(cudaMemcpyToSymbol(avgLum,&value,sizeof(float)));
	dim3 blockSize (16,16,1);
	dim3 gridSize (width/blockSize.x+1,height/blockSize.y+1,1);
	getAvgLum<<<gridSize, blockSize>>>(d_input_image.get());

	cudaDeviceSynchronize();
	float result = 0;
	throw_error(CudaMemcpyFromSymbol(&result,avgLum,sizeof(float)));
	std::cout<<'result '<<result<<std::endl;
	float exposure_host =0;
	throw_error(CudaMemcpyFromSymbol(&exposure_host,exposure,sizeof(float)));
	exposure_host = exposure_host*0.18/result;
	throw_error(cudaMemcpyToSymbol(exposure,&exposure_host,sizeof(float)));
}


