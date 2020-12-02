#include <limits>
#include <iostream>

#include <cuda_runtime_api.h>

#include <utils/CUDA/error.h>

#include "HDRDemo.h"


int HDRDemo::add_vertex(const math::float3& position, const math::float3& normal)
{
	int i = static_cast<int>(size(vertices)) / 6;
	vertices.push_back(position.x);
	vertices.push_back(position.y);
	vertices.push_back(position.z);
	vertices.push_back(normal.x);
	vertices.push_back(normal.y);
	vertices.push_back(normal.z);

	if (position.x < bb_min.x)
		bb_min.x = position.x;
	if (position.x > bb_max.x)
		bb_max.x = position.x;

	if (position.y < bb_min.y)
		bb_min.y = position.y;
	if (position.y > bb_max.y)
		bb_max.y = position.y;

	if (position.z < bb_min.z)
		bb_min.z = position.z;
	if (position.z > bb_max.z)
		bb_max.z = position.z;

	return i;
}

void HDRDemo::add_triangle(int v_1, int v_2, int v_3)
{
	indices.push_back(v_1);
	indices.push_back(v_2);
	indices.push_back(v_3);
}

void HDRDemo::add_model(const std::filesystem::path& path)
{
	std::cout << "reading " << path << '\n' << std::flush;

	if (indices.empty())
	{
		bb_min = { std::numeric_limits<float>::max(),  std::numeric_limits<float>::max(),  std::numeric_limits<float>::max() };
		bb_max = { -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max() };
	}

	OBJ::readTriangles(*this, path);
	std::cout << '\n';
}

void HDRDemo::run(const std::filesystem::path& output_file, const std::filesystem::path& envmap, int cuda_device, float exposure_value, float brightpass_threshold, int test_runs)
{
	cudaDeviceProp props;
	throw_error(cudaGetDeviceProperties(&props, cuda_device));
	std::cout << "using cuda device " << cuda_device << ":\n"
	            "\t" << props.name << "\n"
	            "\tcompute capability " << props.major << "." << props.minor << " @ " << std::setprecision(1) << std::fixed << props.clockRate / 1000.0f << " MHz\n"
	            "\t" << props.multiProcessorCount << " multiprocessors\n"
	            "\t" << props.totalGlobalMem / (1024U * 1024U) << " MiB global memory  " << props.sharedMemPerMultiprocessor / 1024 << " KiB shared memory\n" << std::flush;

	throw_error(cudaSetDevice(cuda_device));


	void run(const std::filesystem::path& output_file, const std::filesystem::path& envmap, const float* vertex_data, int num_vertices, const std::uint32_t* index_data, int num_indices, const math::float3& bb_min, const math::float3& bb_max, float exposure, float brightpass_threshold, int test_runs);

	float exposure = std::exp2(exposure_value);

	run(output_file, envmap, data(vertices), static_cast<int>(size(vertices) / 6), data(indices), static_cast<int>(size(indices)), bb_min, bb_max, exposure, brightpass_threshold, test_runs);
}
