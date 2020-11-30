#include <cstdint>
#include "stdio.h"
#include <math.h>

__global__ void tonemap_kernel(std::uint32_t* out, const float* in, int width, int height, float exposure)
{
	
	int x = (blockIdx.x*blockDim.x + threadIdx.x)*4;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if(y< height && x <= width*4-4){
	float input_tone[3] = {in[y*width*4+x] * exposure,in[y*4*width+x+1]*exposure,in[y*4*width+x+2]*exposure};
	float output_tone[3];
	std::uint8_t scaled_output[3];
	for(int i=0;i<3;i++){
		output_tone[i] = input_tone[i]*(0.9036*input_tone[i] + 0.018)/(input_tone[i]*(0.8748*input_tone[i]+0.354)+0.14);
		if(output_tone[i] <= 0.0031308)
			output_tone[i] = 12.92 * output_tone[i];
		else
			output_tone[i] = 1.055 * pow(output_tone[i],1/2.4) - 0.055;
		scaled_output[i] = (output_tone[i]>1) ? 255 : 255*output_tone[i];
			
	}
	out[y*width+x/4] = 0;
	out[y*width+x/4] |= scaled_output[2] <<16;
	out[y*width+x/4] |= scaled_output[1]<<8;
	out[y*width+x/4] |= scaled_output[0];
//	float output_tone = input_tone*(0.9036*input_tone + 0.018)/(input_tone*(0.8748*input_tone+0.354)+0.14);
//	float output_gamma;
//	if(output_tone <= 0.0031308)
//		output_gamma = 12.92 * output_tone;
//	else
//		output_gamma = 1.055 * pow(output_tone,1/2.4) - 0.055;
//	std::uint8_t scaled_value = (output_gamma>1) ? 255 : 255*output_gamma;
		
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
