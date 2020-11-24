#include <iostream>
#include <iomanip>

#include <cuda_runtime_api.h>

#include <utils/image.h>
#include <utils/pfm.h>
#include <utils/png.h>
#include <utils/CUDA/error.h>
#include <utils/CUDA/memory.h>
#include <utils/CUDA/array.h>
#include <utils/CUDA/event.h>

#include "envmap.h"
#include "HDRPipeline.h"


void run(const char* envmap_path, float exposure, float brightpass_threshold, int test_runs)
{
	auto envmap = load_envmap(envmap_path, false);

	int image_width = static_cast<int>(width(envmap));
	int image_height = static_cast<int>(height(envmap));

	auto hdr_frame = CUDA::create_array(width(envmap), height(envmap), { 32, 32, 32, 32, cudaChannelFormatKindFloat });
	auto ldr_frame = CUDA::create_array(width(envmap), height(envmap), { 8, 8, 8, 8, cudaChannelFormatKindUnsigned });

	throw_error(cudaMemcpy2DToArray(hdr_frame.get(), 0, 0, data(envmap), image_width * 16U, image_width * 16U, image_height, cudaMemcpyHostToDevice));

	HDRPipeline pipeline(image_width, image_height);


	auto pipeline_begin = CUDA::create_event();
	auto pipeline_end = CUDA::create_event();

	float pipeline_time = 0.0f;

	for (int i = 0; i < test_runs; ++i)
	{
		throw_error(cudaEventRecord(pipeline_begin.get()));
		pipeline.process(ldr_frame.get(), hdr_frame.get(), exposure, brightpass_threshold);
		throw_error(cudaEventRecord(pipeline_end.get()));

		throw_error(cudaEventSynchronize(pipeline_end.get()));

		pipeline_time += CUDA::elapsed_time(pipeline_begin.get(), pipeline_end.get());
	}

	std::cout << "------------------------------------------------------------------------\n" << std::setprecision(2) << std::fixed <<
	             "avg time: " << pipeline_time / test_runs << " ms\n";


	image2D<std::uint32_t> output(image_width, image_height);
	throw_error(cudaMemcpy2DFromArray(data(output), width(output) * 4U, ldr_frame.get(), 0, 0, image_width * 4U, image_height, cudaMemcpyDeviceToHost));

	PNG::saveImageR8G8B8("output.png", output);
}
