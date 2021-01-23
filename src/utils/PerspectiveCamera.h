#ifndef INCLUDED_UTILS_PERSPECTIVE_CAMERA
#define INCLUDED_UTILS_PERSPECTIVE_CAMERA

#pragma once

#include <utils/math/vector.h>

#include "Camera.h"
#include "Navigator.h"


class PerspectiveCamera : public virtual Camera
{
	float fov;
	float near;
	float far;
	const Navigator* navigator;

public:
	PerspectiveCamera(float fov, float near, float far);

	std::byte* writeUniformBuffer(std::byte* dest, float aspect) const override;

	void attach(const Navigator* navigator);
};

#endif  // INCLUDED_UTILS_PERSPECTIVE_CAMERA
