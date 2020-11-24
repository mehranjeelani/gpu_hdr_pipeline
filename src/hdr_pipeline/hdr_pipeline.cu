#include <cstdint>



__global__ void tonemap_kernel(std::uint32_t* out, const float* in, int width, int height, float exposure)
{
}

void tonemap(std::uint32_t* out, const float* in, int width, int height, float exposure, float brightpass_threshold)
{
	tonemap_kernel<<<1, 1>>>(out, in, width, height, exposure);
}
