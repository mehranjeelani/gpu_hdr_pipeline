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

	void attach(const Navigator* navigator);
	void writeUniformBuffer(UniformBuffer* buffer, float aspect) const override;
};

#endif  // INCLUDED_UTILS_PERSPECTIVE_CAMERA
