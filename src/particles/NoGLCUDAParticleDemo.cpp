#include <optional>
#include <iostream>
#include <iomanip>
#include <filesystem>

#include <utils/dynamic_library.h>

#include <utils/CUDA/error.h>
#include <utils/CUDA/device.h>

#include <utils/math/vector.h>

#include "particle_system_state.h"

#include "ParticleSystemLoader.h"
#include "CUDAParticles.h"
#include "ParticleDemo.h"


namespace
{
	CUDAParticles load(const std::filesystem::path& path)
	{
		class SceneBuilder : private virtual ParticleSystemBuilder, private virtual ParticleReplayBuilder
		{
			std::optional<CUDAParticles> particles;

			void add_particle_simulation(std::size_t num_particles, std::unique_ptr<float[]> position, std::unique_ptr<std::uint32_t[]> color, const ParticleSystemParameters& params) override
			{
				static auto module = particle_system_module("particle_system");
				particles.emplace(module.create_instance(num_particles, &position[0] + 0 * num_particles, &position[0] + 1 * num_particles, &position[0] + 2 * num_particles, &position[0] + 3 * num_particles, &color[0], params), num_particles);
			}

			ParticleReplayBuilder& add_particle_replay(std::size_t num_particles, std::unique_ptr<float[]> position, std::unique_ptr<std::uint32_t[]> color, const ParticleSystemParameters& params) override
			{
				throw std::runtime_error("particle replay not implemented yet");
				return *this;
			}

			void add_frame(std::chrono::nanoseconds dt, float* positions, const std::uint32_t* colors) override
			{
			}

		public:
			CUDAParticles load(const std::filesystem::path& path)
			{
				load_particles(*this, path);

				if (!particles.has_value())
					throw std::runtime_error("file did not contain a particle system");

				return std::move(*particles);
			}
		};

		SceneBuilder scene_builder;
		return scene_builder.load(path);
	}

	std::ostream& pad(std::ostream& out, int n)
	{
		for (int i = n; i > 0; --i) out.put(' ');
		return out;
	}
}

void ParticleDemo::run(std::filesystem::path output_file, const std::filesystem::path& input_file, int N, float dt, int cuda_device)
{
	CUDA::print_device_properties(std::cout, cuda_device) << '\n' << '\n' << std::flush;
	throw_error(cudaSetDevice(cuda_device));

	auto particles = load(input_file);

	//if (output_file.empty())
	//	(output_file = input_file).replace_extension(".particlereplay");

	//std::ofstream out(output_file);

	//if (!out)
	//	throw std::runtime_error("failed to open output file \"" + output_file.string() + '"');


	//ParticleReplayWriter writer(out);

	float particles_time = 0.0f;

	std::cout << '\n' << N << " frame(s):\n";

	int padding = static_cast<int>(std::log10(N));
	int next_padding_shift = 10;

	for (int i = 0; i < N; ++i)
	{
		auto t = particles.update(1, dt);

		if ((i + 1) >= next_padding_shift)
		{
			--padding;
			next_padding_shift *= 10;
		}

		pad(std::cout, padding) << "t_" << (i + 1) << ": " << std::setprecision(2) << std::fixed << t << " ms\n" << std::flush;

		particles_time += t;
	}

	std::cout << "avg time: " << std::setprecision(2) << std::fixed << particles_time / N << " ms\n" << std::flush;
}
