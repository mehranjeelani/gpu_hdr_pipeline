#include "ParticleSystem.h"
#include<cstddef>
#include<cstdint>
#include<iostream>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include<thrust/device_malloc.h>
#include<cmath>
ParticleSystem::ParticleSystem(std::size_t num_particles, const float* x, const float* y, const float* z, 
							const float* r, const std::uint32_t* color, const ParticleSystemParameters& params)
	: num_particles(num_particles)
	, params(params)
{
	// std::cout<<"In constructor"<<std::endl;
	void *prev,*c,*curr,*k,*v,*cS,*cE,*a;
	cudaMalloc(&curr, num_particles*4*sizeof(float));//+ num_particles * sizeof(std::uint32_t));
	currentPos = static_cast<float*>(curr);
	cudaMalloc(&prev, num_particles*4*sizeof(float));
	prevPos = static_cast<float*>(prev);
	cudaMalloc(&c, num_particles*sizeof(std::uint32_t));//+ num_particles * sizeof(std::uint32_t));
	particleColor = static_cast<std::uint32_t*>(c);
	cudaMalloc(&a, 3*num_particles*sizeof(float));//+ num_particles * sizeof(std::uint32_t));
	acceleration = static_cast<float*>(a);
	int N_x = ceil((params.bb_max[0] - params.bb_min[0])/(2*params.max_particle_radius));
	int N_y = ceil((params.bb_max[1] - params.bb_min[1])/(2*params.max_particle_radius));
	int N_z = ceil((params.bb_max[2] - params.bb_min[2])/(2*params.max_particle_radius));
	cudaMalloc(&cS, N_x*N_y*N_z*sizeof(int));
	cellStart = static_cast<int*>(cS);
	cudaMalloc(&cE, N_x*N_y*N_z*sizeof(int));
	cellEnd = static_cast<int*>(cE);
	//cudaMalloc(&k, num_particles*sizeof(int));
	//keys = static_cast<int*>(k);
	keys = thrust::device_malloc<int>(num_particles);
	values = thrust::device_malloc<int>(num_particles);
	//cudaMalloc(&v, num_particles*sizeof(int));
	//values = static_cast<int*>(v);
	// std::cout<<"calling reset"<<std::endl;
	reset(x, y, z, r, color);

}

void ParticleSystem::reset(const float* x, const float* y, const float* z, const float* r, 
							const std::uint32_t* color)
{
	// TODO: reset particle system to the given state
	// std::cout<<"In reset"<<std::endl;
	// std::cout<<"in reset"<<std::endl;
	
	cudaMemcpy(currentPos + 0 * num_particles, x, num_particles * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(currentPos + 1 * num_particles, y, num_particles * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(currentPos + 2 * num_particles, z, num_particles * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(currentPos + 3 * num_particles, r, num_particles * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(particleColor, color, num_particles * sizeof(std::uint32_t),cudaMemcpyHostToDevice);
	cudaMemcpy(prevPos, currentPos, 4 * num_particles * sizeof(float),cudaMemcpyHostToHost);
	// std::cout<<"leaving reset"<<std::endl;
	// std::cout<<"y cordinate of first particle in reset "<<y[0]<<std::endl;
	//cudaMemset(keys, 0, num_particles*sizeof(int));
	cudaMemset(acceleration, 0, 3*num_particles*sizeof(float));
	//cudaMemset(values, 0, num_particles*sizeof(int));
	int N_x = floor((params.bb_max[0] - params.bb_min[0])/(2*params.max_particle_radius))+1;
	int N_y = floor((params.bb_max[1] - params.bb_min[1])/(2*params.max_particle_radius))+1;
	int N_z = floor((params.bb_max[2] - params.bb_min[2])/(2*params.max_particle_radius))+1;
	cudaMemset(cellStart, -1,  N_x*N_y*N_z*sizeof(int));
	cudaMemset(cellEnd, -1,  N_x*N_y*N_z*sizeof(int));
	//thrust::fill(keys, keys + num_particles, (int) 0);
	//thrust::fill(values, values + num_particles, (int) 0);

}
void update_particles(float* position, std::uint32_t* color, float* prevPos, 
					float* currentPos,std::uint32_t* particleColor, std::size_t num_particles,
					 const ParticleSystemParameters params,float dt,thrust::device_ptr<int> keys,thrust::device_ptr<int> values,int* cellStart,int* cellEnd,float* acceleration);

void ParticleSystem::update(float* position, std::uint32_t* color, float dt)
{
	// TODO: update particle system by timestep dt (in seconds)
	//       position and color are device pointers to write-only buffers to receive the result
	// update_particles(position, std::uint32_t* color, float* input, std::size_t num_particles){
	// std::cout<<"in update and will call update_particles"<<std::endl;
	// std::cout<<"y cordinate of first particle in update "<<currentPos[1 * num_particles + 0]<<std::endl;
	update_particles(position, color, prevPos, currentPos, particleColor, num_particles, params, dt,keys,values,cellStart,cellEnd,acceleration);
	// std::cout<<"leaving update"<<std::endl;
}
