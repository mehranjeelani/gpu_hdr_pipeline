#include "PerspectiveCamera.h"


namespace
{
	math::float4x4 proj(float fov, float aspect, float near, float far)
	{
		const float s2 = 1.0f / std::tan(fov * 0.5f);
		const float s1 = s2 / aspect;
		//const float z1 = (far + near) / (far - near);
		//const float z2 = -2.0f * near * far / (far - near);
		const float z1 = 1.0f;
		const float z2 = -2.0f * near;

		return math::float4x4(s1, 0.0f, 0.0f, 0.0f,
		                      0.0f, s2, 0.0f, 0.0f,
		                      0.0f, 0.0f, z1, z2,
		                      0.0f, 0.0f, 1.0f, 0.0f);
	}
}

PerspectiveCamera::PerspectiveCamera(float fov, float near, float far)
	: fov(fov),
	  near(near),
	  far(far),
	  navigator(nullptr)
{
}

void PerspectiveCamera::attach(const Navigator* navigator)
{
	PerspectiveCamera::navigator = navigator;
}

void PerspectiveCamera::writeUniformBuffer(UniformBuffer* buffer, float aspect) const
{
	if (navigator)
	{
		navigator->writeWorldToLocalTransform(&buffer->V);
		navigator->writeLocalToWorldTransform(&buffer->V_inv);
		navigator->writePosition(&buffer->position);
	}
	else
	{
		buffer->V = math::identity<math::float4x4>;
		buffer->V_inv = math::identity<math::float4x4>;
		buffer->position = math::float3(0.0f, 0.0f, 0.0f);
	}

	buffer->P = proj(fov, aspect, near, far);
	buffer->P_inv = inverse(buffer->P);
	buffer->PV = buffer->P * buffer->V;
	buffer->PV_inv = buffer->V_inv * buffer->P_inv;
}
