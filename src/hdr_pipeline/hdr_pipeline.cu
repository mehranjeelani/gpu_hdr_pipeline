#include <cstdint>
#include "stdio.h"
#include "math.h"
#include <utils/CUDA/error.h>
#include "iostream"
#include "bloom_kernel.h"
__device__ float avgLum=0;
__constant__ float c_bloom_kernel[63]; 
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
		
		output_tone[0] = (output_tone[0]>1 ? 255 : int(255*output_tone[0]));
		output_tone[1] = (output_tone[1]>1? 255 : int(255*output_tone[1]));
		output_tone[2] = (output_tone[2]>1? 255 : int(255*output_tone[2]));
		
		out[y*width+x/4] = 0 |(std::uint8_t)(output_tone[2]) <<16 | (std::uint8_t)(output_tone[1]) <<8 | (std::uint8_t)(output_tone[0]);
		/*
		output_tone[0] = (input_tone[0] <= 0.0031308 ? 12.92 * input_tone[0] : 1.055 * powf(input_tone[0],1/2.4) - 0.055);
		output_tone[1] = (input_tone[1] <= 0.0031308 ? 12.92 * input_tone[1] : 1.055 * powf(input_tone[1],1/2.4) - 0.055);
		output_tone[2] = (input_tone[2] <= 0.0031308 ? 12.92 * input_tone[2] : 1.055 * powf(input_tone[2],1/2.4) - 0.055);
		
		output_tone[0] = (output_tone[0]>1 ? 255 : rintf(255*output_tone[0]));
		output_tone[1] = (output_tone[1]>1? 255 : rintf(255*output_tone[1]));
		output_tone[2] = (output_tone[2]>1? 255 : rintf(255*output_tone[2]));
		
		out[y*width+x/4] = 0 |(std::uint8_t)(output_tone[2]) <<16 | (std::uint8_t)(output_tone[1]) <<8 | (std::uint8_t)(output_tone[0]);
		*/
		
	}
	
	
}

__global__ void getAvgLum_kernel(const float* in,int width,int height){
	/*
	__shared__ float ins[16 * 16];
	int col = (blockIdx.x*blockDim.x + threadIdx.x)*4;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	if(row< height && col <= width*4-4){
		ins[threadIdx.y*blockDim.x+threadIdx.x] = logf(0.2126*in[row*4*width+col]+0.7152*in[row*4*width+col+1]+0.0722*in[row*4*width+col+2]);
		__syncthreads();
		unsigned int t = threadIdx.y*blockDim.x+threadIdx.x;
		for(unsigned int stride = blockDim.y*blockDim.x>>1;stride>0;stride>>=1){
			__syncthreads();
			if(t<stride)
				ins[t] += ins[t+stride];
		}
		if (t == 0)
			atomicAdd(&avgLum, ins[t]);
			
		

	}*/
	// getting ever slightly brighter image with the above approach even though that is faster
	int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < height && col <= width * 4 - 4) {
		float val = logf(0.2126 * in[row * 4 * width + col] + 0.7152 * in[row * 4 * width + col + 1] + 0.0722 * in[row * 4 * width + col + 2]);
		atomicAdd(&avgLum, val);
	}
}

__global__ void brightPass_kernel(float* out, const float* in, int width, int height, float exposure,float brightpass_threshold)
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
		
		
		output_tone[0] = ((output_tone[0]-0.8*brightpass_threshold)/(0.2*brightpass_threshold));
		output_tone[1] = ((output_tone[1]-0.8*brightpass_threshold)/(0.2*brightpass_threshold));
		output_tone[2] = ((output_tone[2]-0.8*brightpass_threshold)/(0.2*brightpass_threshold));
		
		output_tone[0] = output_tone[0]>0 ? output_tone[0] : 0;
		output_tone[1] = output_tone[1]>0 ? output_tone[1] : 0;
		output_tone[2] = output_tone[2]>0 ? output_tone[2] : 0;

		output_tone[0] = output_tone[0]<1 ? output_tone[0] : 1;
		output_tone[1] = output_tone[1]<1 ? output_tone[1] : 1;
		output_tone[2] = output_tone[2]<1 ? output_tone[2] : 1;

		out[y*width*4+x] = output_tone[0]*output_tone[0]*in[y*width*4+x];
		out[y*width*4+x+1] = output_tone[1]*output_tone[1]*in[y*width*4+x+1];
		out[y*width*4+x+2] = output_tone[2]*output_tone[2]*in[y*width*4+x+2];
		out[y*width*4+x+3] = 0;
	}
	
}
__global__ void convolution_x_kernel(float* convolution_x_image,float* brightPass_image,int width,int height){
	int x = (blockIdx.x*blockDim.x + threadIdx.x);
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if(y< height && x < width*4){
		convolution_x_image[y*width*4+x] = 0;
		
		for(int i = -31;i<32;i++){
			if(x+i*4>=0 && x+i*4<width*4)
				convolution_x_image[y*width*4+x] += brightPass_image[y*width*4+x+i*4]*c_bloom_kernel[i+31];
				

		}
		
	}
}
__global__ void convolution_image_kernel(float* convolution_image,float* convolution_x_image,const float* in,float* brightPass_image,int width,int height){
	int x = (blockIdx.x*blockDim.x + threadIdx.x);
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if(y< height && x < width*4){
		convolution_image[y*width*4+x] = 0;
		
		for(int i = -31;i<32;i++){
			if(y+i>=0 && y+i<height)
				convolution_image[y*width*4+x] += convolution_x_image[(y+i)*width*4+x]*c_bloom_kernel[i+31];
				

		}
		
		convolution_image[y*width*4+x] = convolution_image[y*width*4+x]+in[y*width*4+x]-brightPass_image[y*width*4+x];

	}
}	


void tonemap(std::uint32_t* out, const float* in, int width, int height, float exposure, float brightpass_threshold)
{
	//std::cout<<"bloom kernel "<<bloom_kernel[0]<<std::endl;
	dim3 blockSize (32,32,1);
	dim3 gridSize (width/blockSize.x+1,height/blockSize.y+1,1);
	//throw_error(cudaMemcpyToSymbol(c_bloom_kernel,&bl,sizeof(float)));

	tonemap_kernel<<<gridSize, blockSize>>>(out, in, width, height, exposure);
	cudaDeviceSynchronize();

}
float getAvgLum(const float* in,int width,int height,float exposure){
 	
	dim3 blockSize (16,16,1);
	dim3 gridSize (width/blockSize.x+1,height/blockSize.y+1,1);
	getAvgLum_kernel<<<gridSize, blockSize>>>(in,width,height);
	cudaDeviceSynchronize();
	float result;
	throw_error(cudaMemcpyFromSymbol(&result,avgLum,sizeof(float)));
	exposure = exposure * 0.18 / exp(result/(height*width));
	return exposure;
}
void brightPass(float* out, const float* in, int width, int height, float exposure, float brightpass_threshold)
{
	dim3 blockSize (32,32,1);
	dim3 gridSize (width/blockSize.x+1,height/blockSize.y+1,1);
	brightPass_kernel<<<gridSize, blockSize>>>(out, in, width, height, exposure,brightpass_threshold);
	cudaDeviceSynchronize();

}
void convolution(float* convolution_image,float* convolution_x_image,float*  brightPass_image,const float* in, int width,int height){
	cudaMemcpyToSymbol(c_bloom_kernel, &bloom_kernel, 63 * sizeof(float));
	dim3 blockSize (32,32,1);
	dim3 gridSize (width*4/blockSize.x+1,height/blockSize.y+1,1);
	convolution_x_kernel<<<gridSize,blockSize>>>(convolution_x_image,brightPass_image,width,height);
	cudaDeviceSynchronize();
	convolution_image_kernel<<<gridSize,blockSize>>>(convolution_image,convolution_x_image,in,brightPass_image,width,height);
	cudaDeviceSynchronize();
}