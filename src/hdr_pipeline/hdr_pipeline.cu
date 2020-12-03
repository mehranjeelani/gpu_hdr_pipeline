#include <cstdint>
#include "stdio.h"
#include "math.h"

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

	/*
	output_tone[0] = compute(in[y*width*4+x]*exposure,width,height);
	output_tone[1] = compute(in[y*width*4+x+1] * exposure,width,height);
	output_tone[2] = compute(in[y*width*4+x+2] * exposure,width,height);
	//std::uint8_t scaled_output[3];
	*/
	/*
	for(int i=0;i<3;i++){
		output_tone[i] = input_tone[i]*(0.9036*input_tone[i] + 0.018)/(input_tone[i]*(0.8748*input_tone[i]+0.354)+0.14);
		if(output_tone[i] <= 0.0031308)
			output_tone[i] = 12.92 * output_tone[i];
		else
			output_tone[i] = 1.055 * powf(output_tone[i],1/2.4) - 0.055;
		output_tone[i] = (output_tone[i]>1) ? 255 : rintf(255*output_tone[i]);
			
	}
	*/
	//out[y*width+x/4] = 0 |(std::uint8_t)(output_tone[2]) <<16 | (std::uint8_t)(output_tone[1]) <<8 | (std::uint8_t)(output_tone[0]);
	/*
	out[y*width+x/4] = (std::uint8_t)(output_tone[2]) <<16;
	out[y*width+x/4] =  (std::uint8_t)(output_tone[1])<<8;
	out[y*width+x/4] =  (std::uint8_t)(output_tone[0]);
//	float output_tone = input_tone*(0.9036*input_tone + 0.018)/(input_tone*(0.8748*input_tone+0.354)+0.14);
//	float output_gamma;
//	if(output_tone <= 0.0031308)
//		output_gamma = 12.92 * output_tone;
//	else
//		output_gamma = 1.055 * pow(output_tone,1/2.4) - 0.055;
//	std::uint8_t scaled_value = (output_gamma>1) ? 255 : 255*output_gamma;
	*/
		
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

