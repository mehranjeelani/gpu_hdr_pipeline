#include <utils/CUDA/error.h>
#include <utils/CUDA/memory.h>
#include "iostream"
#include "HDRPipeline.h"
__device__ float avgLum; 
__global__  void getAvgLum(const float* in);

HDRPipeline::HDRPipeline(int width, int height)
	: frame_width(width),
	  frame_height(height),
	  d_input_image(CUDA::malloc<float>(width * height * 4)),
	  d_output_image(CUDA::malloc_zeroed<std::uint32_t>(width * height))
{
}


void tonemap(std::uint32_t* out, const float* in, int width, int height, float exposure, float brightpass_threshold);

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
void HDRPipeline::process(cudaArray_t out, cudaArray_t in, float exposure, float brightpass_threshold)
{
	throw_error(cudaMemcpy2DFromArray(d_input_image.get(), frame_width * 16U, in, 0, 0, frame_width * 16U, frame_height, cudaMemcpyDeviceToDevice));
	std::cout<<'Hello WOrld'<<std::endl;
	float value = 1;
	throw_error(cudaMemcpyToSymbol(avgLum,&value,sizeof(float)));
	dim3 blockSize (16,16,1);
	dim3 gridSize (frame_width/blockSize.x+1,frame_height/blockSize.y+1,1);
	getAvgLum<<<gridSize, blockSize>>>(d_input_image.get());

	cudaDeviceSynchronize();
	float result = 0;
	throw_error(CudaMemcpyFromSymbol(&result,avgLum,sizeof(float)));
	std::cout<<'result '<<result<<std::endl;
	float exposure_host =0;
	throw_error(CudaMemcpyFromSymbol(&exposure_host,exposure,sizeof(float)));
	exposure_host = exposure_host*0.18/result;
	throw_error(cudaMemcpyToSymbol(exposure,&exposure_host,sizeof(float)));
	tonemap(d_output_image.get(), d_input_image.get(), frame_width, frame_height, exposure, brightpass_threshold);


	throw_error(cudaMemcpy2DToArray(out, 0, 0, d_output_image.get(), frame_width * 4U, frame_width * 4U, frame_height, cudaMemcpyDeviceToDevice));
}
