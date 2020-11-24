#ifndef INCLUDED_UTILS_NAVIGATOR
#define INCLUDED_UTILS_NAVIGATOR

#pragma once

#include <utils/math/matrix.h>


struct Navigator
{
	virtual void reset() = 0;
	virtual void writeWorldToLocalTransform(math::float4x4* M) const = 0;
	virtual void writeLocalToWorldTransform(math::float4x4* M) const = 0;
	virtual void writePosition(math::float3* p) const = 0;

protected:
	Navigator() = default;
	Navigator(const Navigator&) = default;
	Navigator(Navigator&&) = default;
	Navigator& operator =(const Navigator&) = default;
	Navigator& operator =(Navigator&&) = default;
	~Navigator() = default;
};

#endif  // INCLUDED_UTILS_NAVIGATOR
