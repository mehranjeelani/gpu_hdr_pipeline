#include "ParticleSystem.h"
#include<iostream>

ParticleSystem::ParticleSystem(std::size_t num_particles, const float* x, const float* y, const float* z, 
							const float* r, const std::uint32_t* color, const ParticleSystemParameters& params)
	: num_particles(num_particles)
	, params(params)
{
	// std::cout<<"In constructor"<<std::endl;
	void *prev,*c,*curr;
	cudaMalloc(&curr, num_particles*4*sizeof(float));//+ num_particles * sizeof(std::uint32_t));
	currentPos = static_cast<float*>(curr);
	cudaMalloc(&prev, num_particles*4*sizeof(float));
	prevPos = static_cast<float*>(prev);
	cudaMalloc(&c, num_particles*sizeof(std::uint32_t));//+ num_particles * sizeof(std::uint32_t));
	particleColor = static_cast<std::uint32_t*>(c);
	// std::cout<<"calling reset"<<std::endl;
	reset(x, y, z, r, color);
	

}

void ParticleSystem::reset(const float* x, const float* y, const float* z, const float* r, 
							const std::uint32_t* color)
{
	// TODO: reset particle system to the given state
	//std::cout<<"In reset"<<std::endl;
	//std::cout<<"in reset"<<std::endl;
	
	cudaMemcpy(currentPos + 0 * num_particles, x, num_particles * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(currentPos + 1 * num_particles, y, num_particles * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(currentPos + 2 * num_particles, z, num_particles * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(currentPos + 3 * num_particles, r, num_particles * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(particleColor, color, num_particles * sizeof(std::uint32_t),cudaMemcpyHostToDevice);
	cudaMemcpy(prevPos, currentPos, 4 * num_particles * sizeof(float),cudaMemcpyHostToHost);
	//std::cout<<"leaving reset"<<std::endl;
	//std::cout<<"y cordinate of first particle in reset "<<y[0]<<std::endl;
}
void update_particles(float* position, std::uint32_t* color, float* prevPos, 
					float* currentPos,std::uint32_t* particleColor, std::size_t num_particles,
					const ParticleSystemParameters params,float dt);

void ParticleSystem::update(float* position, std::uint32_t* color, float dt)
{
	// TODO: update particle system by timestep dt (in seconds)
	//       position and color are device pointers to write-only buffers to receive the result
	// update_particles(position, std::uint32_t* color, float* input, std::size_t num_particles){
	//std::cout<<"in update and will call update_particles"<<std::endl;
	//std::cout<<"y cordinate of first particle in update "<<currentPos[1 * num_particles + 0]<<std::endl;
	update_particles(position, color, prevPos, currentPos, particleColor, num_particles, params, dt);
	// std::cout<<"leaving update"<<std::endl;
}
