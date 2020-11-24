#ifndef INCLUDED_UTILS_CUDA_GRAPHICS_INTEROP
#define INCLUDED_UTILS_CUDA_GRAPHICS_INTEROP

#pragma once

#include <cstddef>
#include <memory>
#include <array>

#include <cuda_runtime_api.h>

#include "error.h"


namespace CUDA
{
	namespace graphics
	{
		struct ResourceDeleter
		{
			void operator()(cudaGraphicsResource_t res) const
			{
				cudaGraphicsUnregisterResource(res);
			}
		};

		using unique_resource = std::unique_ptr<cudaGraphicsResource, ResourceDeleter>;


		template <std::size_t N>
		class mapped_resources
		{
			std::array<cudaGraphicsResource_t, N> res;

		public:
			template <typename... Args>
			mapped_resources(Args&&... args)
				: res { std::forward<Args>(args)... }
			{
				static_assert(sizeof...(args) == N);
				throw_error(cudaGraphicsMapResources(static_cast<int>(std::size(res)), std::data(res)));
			}

			~mapped_resources()
			{
				cudaGraphicsUnmapResources(static_cast<int>(std::size(res)), std::data(res));
			}

			decltype(auto) operator[](std::size_t i) const noexcept { return res[i]; }
			auto begin() const noexcept { return std::begin(res); }
			auto end() const noexcept { return std::end(res); }
		};

		template <typename... Args>
		[[nodiscard]] auto map_resources(Args&&... args)
		{
			return mapped_resources<sizeof...(args)> { std::forward<Args>(args)... };
		}


		inline auto get_mapped_buffer(cudaGraphicsResource_t resource)
		{
			struct
			{
				void* ptr;
				std::size_t size;
			} mapped_buffer;

			throw_error(cudaGraphicsResourceGetMappedPointer(&mapped_buffer.ptr, &mapped_buffer.size, resource));

			return mapped_buffer;
		}

		inline cudaArray_t get_mapped_array(cudaGraphicsResource_t resource, unsigned int array_index = 0U, unsigned int mip_level = 0U)
		{
			cudaArray_t arr;
			throw_error(cudaGraphicsSubResourceGetMappedArray(&arr, resource, array_index, mip_level));
			return arr;
		}
	}
}

#endif  // INCLUDED_UTILS_CUDA_GRAPHICS_INTEROP
