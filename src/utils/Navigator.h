#ifndef INCLUDED_UTILS_NAVIGATOR
#define INCLUDED_UTILS_NAVIGATOR

#pragma once

#include <utils/math/matrix.h>


struct Navigator
{
	virtual void reset() = 0;
	virtual math::float4x4 world_to_local_transform() const = 0;
	virtual math::float4x4 local_to_world_transform() const = 0;
	virtual math::float3 position() const = 0;

protected:
	Navigator() = default;
	Navigator(const Navigator&) = default;
	Navigator(Navigator&&) = default;
	Navigator& operator =(const Navigator&) = default;
	Navigator& operator =(Navigator&&) = default;
	~Navigator() = default;
};

#endif  // INCLUDED_UTILS_NAVIGATOR
