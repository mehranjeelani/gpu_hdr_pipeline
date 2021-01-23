#include <utility>
#include <exception>
#include <iostream>
#include <iomanip>
#include <filesystem>

#include <utils/argparse.h>

#include "ParticleDemo.h"

using namespace std::literals;


namespace
{
	std::ostream& print_usage(std::ostream& out)
	{
		return out << R"""(usage: particles [{options}] <file>
	options:
	  -o <file>              save replay to <file>
	  --device <i>           use CUDA device <i>, default: 0
	  --timestep <dt>        simulation time step <dt> s, default: 0.01 s
	  -N <N>                 run <N> simulation frames, default: 10000
	  --frozen               start simulation frozen, default: false
)""";
	}
}

int main(int argc, char* argv[])
{
	try
	{
		ParticleDemo demo;
		int cuda_device = 0;
		float dt = 0.01f;
		int N = 1000;
		bool frozen = false;
		std::filesystem::path particles_file;
		std::filesystem::path output_file;

		for (const char* const* a = argv + 1; *a; ++a)
		{
			if (!argparse::parseIntArgument(cuda_device, a, "--device"sv))
			if (!argparse::parseFloatArgument(dt, a, "--timestep"sv))
			if (!argparse::parseIntArgument(N, a, "-N"sv))
			if (argparse::parseBoolFlag(a, "--frozen"sv))
				frozen = true;
			if (const char* str = argparse::parseStringArgument(a, "-o"sv))
				output_file = str;
			else
				particles_file = *a;
		}

		if (particles_file.empty())
			throw argparse::usage_error("expected input file");

		demo.run(std::move(output_file), particles_file, N, dt, cuda_device);
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
