#include <utils/dynamic_library.h>

#include <utils/CUDA/error.h>

#include "CUDAParticles.h"


CUDAParticles::CUDAParticles(particle_system_instance particles, std::size_t num_particles)
	: particles(std::move(particles)),
	  num_particles(num_particles),
	  position_buffer(CUDA::malloc<float>(4 * num_particles)),
	  color_buffer(CUDA::malloc<std::uint32_t>(num_particles)),
	  particles_begin(CUDA::create_event()),
	  particles_end(CUDA::create_event())
{
}

float CUDAParticles::update(int steps, float dt)
{
	throw_error(cudaEventRecord(particles_begin.get()));

	for (int i = 0; i < steps; ++i)
		particles.update(position_buffer.get(), color_buffer.get(), dt);

	throw_error(cudaEventRecord(particles_end.get()));

	throw_error(cudaEventSynchronize(particles_end.get()));

	return CUDA::elapsed_time(particles_begin.get(), particles_end.get());
}
