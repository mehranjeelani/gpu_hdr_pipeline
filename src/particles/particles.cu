#include<cstddef>
#include<cstdint>
#include "ParticleSystem.h"
#include<iostream>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
__global__ void update_kernel(float* position, std::uint32_t* color, float* prevPos, float* currentPos,
                            std::uint32_t* particleColor, std::size_t num_particles,
                            const ParticleSystemParameters params,float dt,float* acceleration)
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    /***
    if(tid == 0 )
        printf("value in update kernel of y coordinate of particle 0 before update is %f\n",
            currentPos[1 * num_particles + tid]);

    ***/
    if(tid < num_particles)
    {
        float radius = currentPos[3 * num_particles + tid];

        for(int i = 0; i < 3; i++ )
        {
            position[tid * 4 + i] = 2 * currentPos[i * num_particles + tid] -  prevPos[i * num_particles + tid] 
                                    + dt * dt * (params.gravity[i]+acceleration[i * num_particles + tid]) ;
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
    /**
    if(tid == 0 ){
        printf("value in update kernel of y coordinate of particle 0 after update is %f\n",position[tid * 4 + 1]);
        
    }
    **/
    }


    //*********************************************************************************
    

__global__ void calcHash(float* currentPos,std::size_t num_particles,
                                                const ParticleSystemParameters params,
                                                int* keys,int* values,int N_x,int N_y,int N_z)
{
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if(tid<num_particles){

           
            int cell_x = floor((currentPos[0 * num_particles + tid]-params.bb_min[0])/(2*params.max_particle_radius));
            int cell_y = floor((currentPos[1 * num_particles + tid]-params.bb_min[1])/(2*params.max_particle_radius));
            int cell_z = floor((currentPos[2 * num_particles + tid] - params.bb_min[2])/(2*params.max_particle_radius));
            int cell_index = cell_y*N_x*N_z + cell_z*N_x + cell_x;
            keys[tid] = cell_index;
            values[tid] = tid;
        }
        
    }

//*************************************************************************************************************
 
__global__ void printFunction(int* a,int* b,std::size_t num_particles){
    for(int i=0;i<1;i++)
        printf("cellStart[%d] = %d \t cellEnd[%d] = %d \n",622,a[622],i,b[622]);
        
 }

 //**********************************************************************************************************
 __global__ void sort(int* keys,int* values,std::size_t num_particles){
    thrust::sort_by_key(thrust::device,keys, keys + num_particles, values);
}
//***********************************************************************************************************
__global__ void findCellStartEnd(int* keys,int* cellStart,int* cellEnd,std::size_t num_particles){
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<num_particles){
        int cellIndex;
        if(tid==0){
            cellIndex = keys[tid];
            cellStart[cellIndex] = tid;
        }
        else{
            if(keys[tid] != keys[tid-1]){
                cellIndex = keys[tid];
                cellStart[cellIndex] = tid;
            }
        }
        if(tid == num_particles-1){
            cellIndex = keys[tid];
            cellEnd[cellIndex] = tid;
        }
        else{
            if(keys[tid] != keys[tid+1]){
                cellIndex = keys[tid];
                cellEnd[cellIndex] = tid;
            }
        }
            
    }
}  

//*********************************************************************************************************** 

__global__ void resolveCollission(float* currentPos,float* prevPos,std::size_t num_particles,const ParticleSystemParameters params,
                                  float dt, int* keys, int* values, int* cellStart,int* cellEnd,int N_x,int N_y,int N_z,float* acceleration,std::uint32_t* color)
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<num_particles){
        int cell_x = floor((currentPos[0 * num_particles + tid]-params.bb_min[0])/(2*params.max_particle_radius));
        int cell_y = floor((currentPos[1 * num_particles + tid]-params.bb_min[1])/(2*params.max_particle_radius));
        int cell_z = floor((currentPos[2 * num_particles + tid] - params.bb_min[2])/(2*params.max_particle_radius));
        int cell_index = cell_y*N_x*N_z + cell_z*N_x + cell_x;
        float Force_x,Force_y,Force_z;
        Force_x = Force_y = Force_z = 0;
        for(int i = -1;i<2;i++){
            if(cell_x+i<0 || cell_x+i>=N_x)
                continue;
            for(int j = -1;j<2;j++){
                if(cell_y+j<0 || cell_y+j>=N_y)
                    continue;
                for(int k = -1;k<2;k++){
                    if(cell_z+k<0 || cell_z+k>=N_z)
                        continue;
                    int neighbor_cell = (cell_y+j)*N_x*N_z + (cell_z+k)*N_x + cell_x+i;
                    int startIndex = cellStart[neighbor_cell];
                    int endIndex = cellEnd[neighbor_cell];
                    for(int p=startIndex;p<=endIndex;p++){
                        int particleId = values[p];
                        float r_a = currentPos[3 * num_particles + tid];
                        float r_b = currentPos[3 * num_particles + particleId];
                        float x_a = currentPos[0 * num_particles + tid];
                        float x_b = currentPos[0 * num_particles + particleId];
                        float y_a = currentPos[1 * num_particles + tid];
                        float y_b = currentPos[1 * num_particles + particleId];
                        float z_a = currentPos[2 * num_particles + tid];
                        float z_b = currentPos[2 * num_particles + particleId];
                        float distance = sqrt(pow(x_a - x_b,2)+pow(y_a - y_b,2) + pow(z_a-z_b,2));
                        //printf("dt is %f",dt);
                        if(distance < r_a + r_b && tid != particleId){
                            
                            
                            float vx_a =  (x_a - prevPos[0 * num_particles + tid])/dt;
                            float vy_a =  (y_a - prevPos[1 * num_particles + tid])/dt;
                            float vz_a =  (z_a - prevPos[2 * num_particles + tid])/dt;
                            float vx_b =  (x_b - prevPos[0 * num_particles + particleId])/dt;
                            float vy_b =  (y_b - prevPos[1 * num_particles + particleId])/dt;
                            float vz_b =  (z_b - prevPos[2 * num_particles + particleId])/dt;
                            float dab_x = (x_b-x_a)/distance;
                            float dab_y = (y_b-y_a)/distance;
                            float dab_z = (z_b-z_a)/distance;
                            float dotProduct = (vx_b - vx_a)*dab_x + (vy_b - vy_a)*dab_y + (vz_b - vz_a)*dab_z;
                            float tvx = (vx_b - vx_a) - dab_x*dotProduct;
                            float tvy = (vy_b - vy_a) - dab_y*dotProduct;
                            float tvz = (vz_b - vz_a) - dab_z*dotProduct;
                            float fsx = -params.coll_spring *(r_a+r_b-distance)*dab_x;
                            float fsy = -params.coll_spring *(r_a+r_b-distance)*dab_y;
                            float fsz = -params.coll_spring *(r_a+r_b-distance)*dab_z;
                            float fdx = params.coll_damping*(vx_b - vx_a);
                            float fdy = params.coll_damping*(vy_b - vy_a);
                            float fdz = params.coll_damping*(vz_b - vz_a);
                            float ftx = params.coll_shear*tvx;
                            float fty = params.coll_shear*tvy;
                            float ftz = params.coll_shear*tvz;
                            Force_x = Force_x + fsx+fdx+ftx;
                            Force_y = Force_y+fsy+fdy+fty;
                            Force_z = Force_z + fsz+fdz+ftz;
                            

                        }
                    }
                }
            }
        }
        
        acceleration[0 * num_particles + tid] = Force_x;
        acceleration[1 * num_particles + tid] = Force_y;
        acceleration[2 * num_particles + tid] = Force_z;
        

    }



}

//***********************************************************************************************************
void update_particles(float* position, std::uint32_t* color, float* prevPos, 
                    float* currentPos,std::uint32_t* particleColor, std::size_t num_particles,
                    const ParticleSystemParameters params,float dt,int* keys,int* values,int* cellStart,int* cellEnd,float* acceleration)
{
                        // printf("In update particles\n");
    dim3 blockSize (128,1,1);
    dim3 gridSize (num_particles/blockSize.x+1,1,1);
    update_kernel<<<gridSize,blockSize>>>(position, color, prevPos, currentPos, particleColor,
                                        num_particles,params,dt,acceleration);
    // std::cout<<"Params bounce = "<<params.bounce<<std::endl;
    // std::cout<<"Params gravity = "<<params.gravity[1]<<std::endl;
    cudaDeviceSynchronize();
    int N_x = floor((params.bb_max[0] - params.bb_min[0])/(2*params.max_particle_radius))+1;
    int N_y = floor((params.bb_max[1] - params.bb_min[1])/(2*params.max_particle_radius))+1;
    int N_z = floor((params.bb_max[2] - params.bb_min[2])/(2*params.max_particle_radius))+1;
    calcHash<<<gridSize,blockSize>>>(currentPos,num_particles,params,keys,values,N_x,N_y,N_z);
    cudaDeviceSynchronize();
    //printf("Before Sort\n");
    //printFunction<<<(1,1,1),(1,1,1)>>>(keys,values,num_particles);
    //cudaDeviceSynchronize();
    //printf("After Sort\n");
    sort<<<(1,1,1),(1,1,1)>>>(keys,values,num_particles);
    cudaDeviceSynchronize();
    findCellStartEnd<<<gridSize,blockSize>>>(keys,cellStart,cellEnd,num_particles);
    cudaDeviceSynchronize();
    //printFunction<<<(1,1,1),(1,1,1)>>>(cellStart,cellEnd,num_particles);
    //cudaDeviceSynchronize();
    resolveCollission<<<gridSize,blockSize>>>(currentPos,prevPos,num_particles,params,dt,keys,values,cellStart,cellEnd,N_x,N_y,N_z,acceleration,color);
    cudaDeviceSynchronize();    
    // printf("leaving update particles\n");
}