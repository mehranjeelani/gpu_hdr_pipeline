#ifndef INCLUDED_UTILS_CAMERA
#define INCLUDED_UTILS_CAMERA

#pragma once

#include <cstddef>

#include <utils/math/matrix.h>


struct Camera
{
	static constexpr std::size_t uniform_buffer_size = (6 * 4 * 4 + 3 + 1) * 4U;
	static constexpr std::size_t uniform_buffer_alignment = 16U;

	virtual std::byte* writeUniformBuffer(std::byte* dest, float aspect) const = 0;

protected:
	Camera() = default;
	Camera(const Camera&) = default;
	Camera(Camera&&) = default;
	Camera& operator =(const Camera&) = default;
	Camera& operator =(Camera&&) = default;
	~Camera() = default;
};

#endif  // INCLUDED_UTILS_CAMERA
