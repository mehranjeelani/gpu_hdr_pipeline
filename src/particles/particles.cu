#include<cstddef>
#include<cstdint>
#include "ParticleSystem.h"
#include<iostream>

__global__ void update_kernel(float* position, std::uint32_t* color, float* prevPos, float* currentPos,
                            std::uint32_t* particleColor, std::size_t num_particles,
                            const ParticleSystemParameters params,float dt)
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid == 0 )
        printf("value in update kernel of y coordinate of particle 0 before update is %f\n",
            currentPos[1 * num_particles + tid]);


    if(tid < num_particles)
    {
        float radius = currentPos[3 * num_particles + tid];

        for(int i = 0; i < 3; i++ )
        {
            position[tid * 4 + i] = 2 * currentPos[i * num_particles + tid] -  prevPos[i * num_particles + tid] 
                                    + dt * dt * params.gravity[i] ;
            int change = 0;
            while( position[tid * 4 + i] > params.bb_max[i] - radius ||
                    position[tid * 4 + i] < params.bb_min[i] + radius)
            {    
                if( position[tid * 4 + i] > params.bb_max[i] - radius)
                {  
                    position[tid * 4 + i] = (params.bb_max[i] - radius)  * (params.bounce + 1) 
                                            - position[tid * 4 + i] * params.bounce;
                    if(change == 0)
                    {
                        currentPos[i * num_particles + tid] = (params.bb_max[i] - radius)  * (params.bounce + 1) 
                                                             - currentPos[i * num_particles + tid] * params.bounce;
                        change = 1;
                    }
                }
                if( position[tid * 4 + i] < params.bb_min[i] + radius)
                {
                    position[tid * 4 + i] = (params.bb_min[i] + radius)  * (params.bounce + 1) 
                                            - position[tid * 4 + i] * params.bounce;
                    if(change == 0)
                    {
                        currentPos[i * num_particles + tid] = (params.bb_min[i] + radius)  * (params.bounce + 1) 
                                                                - currentPos[i * num_particles + tid] * params.bounce;
                        change = 1;
                    }
                }

            }
            prevPos[i * num_particles + tid] = currentPos[i * num_particles + tid];
            currentPos[i * num_particles + tid] = position[tid * 4 + i];

            
        }
        prevPos[3 * num_particles + tid] = currentPos[3 * num_particles + tid];
        position[tid * 4 + 3] = currentPos[3 * num_particles + tid];
        color[tid] =  particleColor[tid];

    }

    if(tid == 0 ){
        printf("value in update kernel of y coordinate of particle 0 after update is %f\n",position[tid * 4 + 1]);
        
    }
    }

void update_particles(float* position, std::uint32_t* color, float* prevPos, 
                    float* currentPos,std::uint32_t* particleColor, std::size_t num_particles,
                    const ParticleSystemParameters params,float dt){
    printf("In update particles\n");
    dim3 blockSize (1024,1,1);
    dim3 gridSize (num_particles/blockSize.x+1,1,1);
    update_kernel<<<gridSize,blockSize>>>(position, color, prevPos, currentPos, particleColor,
                                        num_particles,params,dt);
    //std::cout<<"Params bounce = "<<params.bounce<<std::endl;
    //std::cout<<"Params gravity = "<<params.gravity[1]<<std::endl;
    cudaDeviceSynchronize();
    printf("leaving update particles\n");
}