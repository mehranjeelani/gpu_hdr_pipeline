#include <utils/CUDA/error.h>
#include <utils/CUDA/memory.h>

#include "HDRPipeline.h"
//#include "bloom_kernel.h"

HDRPipeline::HDRPipeline(int width, int height)
	: frame_width(width),
	  frame_height(height),
	  d_input_image(CUDA::malloc<float>(width * height * 4)),
	  d_brightPass_image(CUDA::malloc<float>(width * height * 4)),
	  d_convolution_x_image(CUDA::malloc<float>(width * height * 4)),
	  d_convolution_image(CUDA::malloc<float>(width * height * 4)),
	  d_output_image(CUDA::malloc_zeroed<std::uint32_t>(width * height))
{
}


void tonemap(std::uint32_t* out, const float* in, int width, int height, float exposure, float brightpass_threshold);
void brightPass(float* out, const float* in, int width, int height, float exposure, float brightpass_threshold);
float getAvgLum(const float* in,int width,int height,float exposure);
void convolution(float* convolution_image,float* convolution_x_image,float*  brightPass_image,const float* in, int width,int height);


void HDRPipeline::process(cudaArray_t out, cudaArray_t in, float exposure, float brightpass_threshold)
{	
	throw_error(cudaMemcpy2DFromArray(d_input_image.get(), frame_width * 16U, in, 0, 0, frame_width * 16U, frame_height, cudaMemcpyDeviceToDevice));

	float exposured = getAvgLum(d_input_image.get(),frame_width,frame_height,exposure);
	brightPass(d_brightPass_image.get(), d_input_image.get(), frame_width, frame_height, exposured, brightpass_threshold);
	convolution(d_convolution_image.get(),d_convolution_x_image.get(),d_brightPass_image.get(),d_input_image.get(),frame_width,frame_height);
	
	tonemap(d_output_image.get(), d_convolution_image.get(), frame_width, frame_height, exposured, brightpass_threshold);


	throw_error(cudaMemcpy2DToArray(out, 0, 0, d_output_image.get(), frame_width * 4U, frame_width * 4U, frame_height, cudaMemcpyDeviceToDevice));
}