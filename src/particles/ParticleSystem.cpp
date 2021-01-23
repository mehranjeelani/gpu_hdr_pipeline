#include "ParticleSystem.h"


ParticleSystem::ParticleSystem(std::size_t num_particles, const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color, const ParticleSystemParameters& params)
	: num_particles(num_particles)
	, params(params)
{
	reset(x, y, z, r, color);
}

void ParticleSystem::reset(const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color)
{
	// TODO: reset particle system to the given state
}

void ParticleSystem::update(float* position, std::uint32_t* color, float dt)
{
	// TODO: update particle system by timestep dt (in seconds)
	//       position and color are device pointers to write-only buffers to receive the result
}
