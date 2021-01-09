#include <utility>
#include <algorithm>

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

	constexpr std::size_t align(std::size_t offset, std::size_t alignment)
	{
		return (offset + alignment - 1) / alignment * alignment;
	}

	template <typename Arg, typename... Args>
	std::byte* write_std140(std::byte* dest, Arg&& arg, Args&&... args);

	std::byte* write_std140(std::byte* dest, const float value)
	{
		static_assert(sizeof(float) == 4U);
		auto src = reinterpret_cast<const std::byte*>(&value);
		return std::copy(src, src + 4, dest);
	}

	std::byte* write_std140(std::byte* dest, const math::float3 value)
	{
		return write_std140(dest, value.x, value.y, value.z);
	}

	std::byte* write_std140(std::byte* dest, const math::float4 value)
	{
		return write_std140(dest, value.x, value.y, value.z, value.w);
	}

	std::byte* write_std140(std::byte* dest, const math::float4x4 M)
	{
		return write_std140(dest,
			math::float4 { M._11, M._12, M._13, M._14 },
			math::float4 { M._21, M._22, M._23, M._24 },
			math::float4 { M._31, M._32, M._33, M._34 },
			math::float4 { M._41, M._42, M._43, M._44 });
	}

	template <typename Arg, typename... Args>
	std::byte* write_std140(std::byte* dest, Arg&& arg, Args&&... args)
	{
		static_assert(sizeof...(Args) != 0);
		return write_std140(write_std140(dest, std::forward<Arg>(arg)), std::forward<Args>(args)...);
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

std::byte* PerspectiveCamera::writeUniformBuffer(std::byte* dest, float aspect) const
{
	auto V = navigator ? navigator->world_to_local_transform() : math::identity<math::float4x4>;
	auto V_inv = navigator ? navigator->local_to_world_transform() : math::identity<math::float4x4>;
	auto position = navigator ? navigator->position() : math::float3(0.0f, 0.0f, 0.0f);

	auto P = proj(fov, aspect, near, far);
	auto P_inv = inverse(P);
	auto PV = P * V;
	auto PV_inv = V_inv * P_inv;

	return write_std140(dest, V, V_inv, P, P_inv, PV, PV_inv, position, aspect);
}
