#ifndef INCLUDED_PARTICLE_SYSTEM
#define INCLUDED_PARTICLE_SYSTEM

#pragma once

#include <cuda_runtime_api.h>
#include<cstddef>
#include<cstdint>
#include <thrust/device_ptr.h>
#include "particle_system_module.h"


class ParticleSystem
{
	const std::size_t num_particles;
	const ParticleSystemParameters params;
	float* currentPos;
	float* prevPos;
	int* grid;
	int* keys;
	int* values;
	//thrust::device_ptr<int> keys;
	//thrust::device_ptr<int> values;
	int* cellStart;
	int* cellEnd;
	std::uint32_t* particleColor;
	float* acceleration;
	

public:
	ParticleSystem(std::size_t num_particles, const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color, const  ParticleSystemParameters& params);

	void reset(const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color);
	void update(float* position, std::uint32_t* color, float dt);
};

#endif // INCLUDED_PARTICLE_SIMULATION
