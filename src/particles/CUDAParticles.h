#ifndef INCLUDED_CUDA_PARTICLES
#define INCLUDED_CUDA_PARTICLES

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

#include <utils/CUDA/memory.h>
#include <utils/CUDA/event.h>

#include "particle_system_module.h"

#include "ParticleSystemLoader.h"


class CUDAParticles
{
	particle_system_instance particles;

	std::size_t num_particles;

	CUDA::unique_ptr<float> position_buffer;
	CUDA::unique_ptr<std::uint32_t> color_buffer;

	CUDA::unique_event particles_begin;
	CUDA::unique_event particles_end;

public:
	CUDAParticles(particle_system_instance particles, std::size_t num_particles);

	float update(int steps, float dt);
};

#endif // INCLUDED_CUDA_PARTICLES
