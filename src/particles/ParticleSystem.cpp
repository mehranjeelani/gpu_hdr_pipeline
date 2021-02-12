#include "ParticleSystem.h"
#include<cstddef>
#include<cstdint>
#include<iostream>
#include<cmath>
ParticleSystem::ParticleSystem(std::size_t num_particles, const float* x, const float* y, const float* z, 
							const float* r, const std::uint32_t* color, const ParticleSystemParameters& params)
	: num_particles(num_particles)
	, params(params)
{
	//printf("Max Particle Radius is %f \n",params.max_particle_radius);
	void *prev,*c,*curr,*cS,*cE,*a;
	cudaMalloc(&curr, num_particles*4*sizeof(float));
	currentPos = static_cast<float*>(curr);
	cudaMalloc(&prev, num_particles*4*sizeof(float));
	prevPos = static_cast<float*>(prev);
	cudaMalloc(&c, num_particles*sizeof(std::uint32_t));
	particleColor = static_cast<std::uint32_t*>(c);
	cudaMalloc(&a, 3*num_particles*sizeof(float));
	acceleration = static_cast<float*>(a);
	int N_x = ceil((params.bb_max[0] - params.bb_min[0])/(cellSize));
	int N_y = ceil((params.bb_max[1] - params.bb_min[1])/(cellSize));
	int N_z = ceil((params.bb_max[2] - params.bb_min[2])/(cellSize));
	cudaMalloc(&cS, N_x*N_y*N_z*sizeof(int));
	cellStart = static_cast<int*>(cS);
	cudaMalloc(&cE, N_x*N_y*N_z*sizeof(int));
	cellEnd = static_cast<int*>(cE);
	cudaMalloc((void **) &keys, num_particles * sizeof(int));
	cudaMalloc((void **) &values, num_particles * sizeof(int));
	reset(x, y, z, r, color);

}

void ParticleSystem::reset(const float* x, const float* y, const float* z, const float* r, 
							const std::uint32_t* color)
{
	
	
	cudaMemcpy(currentPos + 0 * num_particles, x, num_particles * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(currentPos + 1 * num_particles, y, num_particles * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(currentPos + 2 * num_particles, z, num_particles * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(currentPos + 3 * num_particles, r, num_particles * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(particleColor, color, num_particles * sizeof(std::uint32_t),cudaMemcpyHostToDevice);
	cudaMemcpy(prevPos, currentPos, 4 * num_particles * sizeof(float),cudaMemcpyHostToHost);
	cudaMemset(acceleration, 0, 3*num_particles*sizeof(float));
	int N_x = floor((params.bb_max[0] - params.bb_min[0])/(cellSize))+1;
	int N_y = floor((params.bb_max[1] - params.bb_min[1])/(cellSize))+1;
	int N_z = floor((params.bb_max[2] - params.bb_min[2])/(cellSize))+1;
	cudaMemset(cellStart, 0,  N_x*N_y*N_z*sizeof(int));
	cudaMemset(cellEnd, 0,  N_x*N_y*N_z*sizeof(int));
	

}
void update_particles(float* position, std::uint32_t* color, float* prevPos, 
					float* currentPos,std::uint32_t* particleColor, std::size_t num_particles,
					 const ParticleSystemParameters params,float dt,int* keys,int* values,int* cellStart,int* cellEnd,float* acceleration);

void ParticleSystem::update(float* position, std::uint32_t* color, float dt)
{
	
	update_particles(position, color, prevPos, currentPos, particleColor, num_particles, params, dt,keys,values,cellStart,cellEnd,acceleration);

}
