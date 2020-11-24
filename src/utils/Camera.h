#ifndef INCLUDED_UTILS_CAMERA
#define INCLUDED_UTILS_CAMERA

#pragma once

#include <utils/math/matrix.h>


struct Camera
{
	struct UniformBuffer
	{
		alignas(16) math::float4x4 V;
		alignas(16) math::float4x4 V_inv;
		alignas(16) math::float4x4 P;
		alignas(16) math::float4x4 P_inv;
		alignas(16) math::float4x4 PV;
		alignas(16) math::float4x4 PV_inv;
		alignas(16) math::float3 position;
	};

	virtual void writeUniformBuffer(UniformBuffer* buffer, float aspect) const = 0;

protected:
	Camera() = default;
	Camera(const Camera&) = default;
	Camera(Camera&&) = default;
	Camera& operator =(const Camera&) = default;
	Camera& operator =(Camera&&) = default;
	~Camera() = default;
};

#endif  // INCLUDED_UTILS_CAMERA
