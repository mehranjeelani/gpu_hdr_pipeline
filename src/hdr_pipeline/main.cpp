#include <cstdint>
#include <cmath>
#include <array>
#include <iostream>
#include <iomanip>
#include <exception>

#include <cuda_runtime_api.h>

#include <utils/argparse.h>
#include <utils/CUDA/error.h>

using namespace std::literals;


namespace
{
	std::ostream& print_usage(std::ostream& out)
	{
		return out << R"""(usage: hdr_pipeline [{options}] <input-file>
	options:
	  --device <i>           use CUDA device <i>, default: 0
	  --exposure <v>         set exposure value to <v>, default: 0.0
	  --brightpass <v>       set brightpass threshold to <v>, default: 0.9
	  --test-runs <N>        average timings over <N> test runs, default: 1
)""";
	}
}

int main(int argc, char* argv[])
{
	try
	{
		const char* envmap_path = nullptr;
		int cuda_device = 0;
		float exposure_value = 0.0f;
		float brightpass_threshold = 0.9f;
		int test_runs = 1;

		for (const char* const* a = argv + 1; *a; ++a)
		{
			if (!argparse::parseIntArgument(cuda_device, a, "--device"sv))
			if (!argparse::parseFloatArgument(exposure_value, a, "--exposure"sv))
			if (!argparse::parseFloatArgument(brightpass_threshold, a, "--brightpass"sv))
			if (!argparse::parseIntArgument(test_runs, a, "--test-runs"))
				envmap_path = *a;
		}

		if (!envmap_path)
			throw argparse::usage_error("expected input file");

		cudaDeviceProp props;
		throw_error(cudaGetDeviceProperties(&props, cuda_device));
		std::cout << "using cuda device " << cuda_device << ":\n"
		            "\t" << props.name << "\n"
		            "\tcompute capability " << props.major << "." << props.minor << " @ " << std::setprecision(1) << std::fixed << props.clockRate / 1000.0f << " MHz\n"
		            "\t" << props.multiProcessorCount << " multiprocessors\n"
		            "\t" << props.totalGlobalMem / (1024U * 1024U) << " MiB global memory  " << props.sharedMemPerMultiprocessor / 1024 << " KiB shared memory\n" << std::endl;

		throw_error(cudaSetDevice(cuda_device));


		void run(const char* envmap_path, float exposure_value, float brightpass_threshold, int test_runs);

		float exposure = std::exp2(exposure_value);

		run(envmap_path, exposure, brightpass_threshold, test_runs);
	}
	catch (const argparse::usage_error& e)
	{
		std::cerr << "ERROR: " << e.what() << '\n' << print_usage;
		return -127;
	}
	catch (const std::exception& e)
	{
		std::cerr << "ERROR: " << e.what() << '\n';
		return -1;
	}
	catch (...)
	{
		std::cerr << "ERROR: unknown exception\n";
		return -128;
	}

	return 0;
}
