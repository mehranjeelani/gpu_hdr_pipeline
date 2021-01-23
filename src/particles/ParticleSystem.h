#ifndef INCLUDED_PARTICLE_SYSTEM
#define INCLUDED_PARTICLE_SYSTEM

#pragma once

#include <cuda_runtime_api.h>


#include "particle_system_module.h"


class ParticleSystem
{
	const std::size_t num_particles;
	const ParticleSystemParameters params;

public:
	ParticleSystem(std::size_t num_particles, const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color, const ParticleSystemParameters& params);

	void reset(const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color);
	void update(float* position, std::uint32_t* color, float dt);
};

#endif // INCLUDED_PARTICLE_SIMULATION
