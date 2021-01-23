#include <utility>
#include <stdexcept>
#include <iostream>

#include <utils/CUDA/error.h>
#include <utils/CUDA/device.h>

#include "particle_system_state.h"

#include "ParticleSystemLoader.h"
#include "GLCUDAParticles.h"
#include "GLParticleDemo.h"


std::tuple<std::unique_ptr<GLScene>, math::float3, math::float3> load_scene(const std::filesystem::path& path, int cuda_device)
{
	CUDA::print_device_properties(std::cout, cuda_device) << '\n' << '\n' << std::flush;
	throw_error(cudaSetDevice(cuda_device));

	class SceneBuilder : private virtual ParticleSystemBuilder, private virtual ParticleReplayBuilder
	{
		std::unique_ptr<GLScene> scene;
		math::float3 bb_min;
		math::float3 bb_max;

		void add_particle_simulation(std::size_t num_particles, std::unique_ptr<float[]> position, std::unique_ptr<std::uint32_t[]> color, const ParticleSystemParameters& params) override
		{
			static auto module = particle_system_module("particle_system");
			auto particles = module.create_instance(num_particles, &position[0] + 0 * num_particles, &position[0] + 1 * num_particles, &position[0] + 2 * num_particles, &position[0] + 3 * num_particles, &color[0], params);
			scene = std::make_unique<GLCUDAParticles>(std::move(particles), num_particles, std::move(position), std::move(color));
			bb_min = { params.bb_min[0], params.bb_min[1], params.bb_min[2] };
			bb_max = { params.bb_max[0], params.bb_max[1], params.bb_max[2] };
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
		std::tuple<std::unique_ptr<GLScene>, math::float3, math::float3> load(const std::filesystem::path& path)
		{
			load_particles(*this, path);
			return { std::move(scene), bb_min, bb_max };
		}
	};

	SceneBuilder scene_builder;
	return scene_builder.load(path);
}
