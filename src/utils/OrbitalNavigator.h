#ifndef INCLUDED_UTILS_ORBITAL_NAVIGATOR
#define INCLUDED_UTILS_ORBITAL_NAVIGATOR

#pragma once

#include <GL/platform/InputHandler.h>

#include "Navigator.h"


class OrbitalNavigator : public Navigator, public virtual GL::platform::MouseInputHandler
{
	math::float3 u;
	math::float3 v;
	math::float3 w;
	math::float3 position;

	math::int2 last_pos;
	unsigned int drag;

	float phi;
	float theta;
	float radius;
	math::float3 lookat;

	math::float3 initial_lookat;
	float initial_phi;
	float initial_theta;
	float initial_radius;

	void rotateH(float dphi);
	void rotateV(float dtheta);
	void zoom(float dr);
	void pan(float u, float v);
	void update();

public:
	OrbitalNavigator(float phi, float theta, float radius, const math::float3& lookat);

	void reset() override;
	void writeWorldToLocalTransform(math::float4x4* M) const override;
	void writeLocalToWorldTransform(math::float4x4* M) const override;
	void writePosition(math::float3* p) const override;

	void buttonDown(GL::platform::Button button, int x, int y, GL::platform::Window*) override;
	void buttonUp(GL::platform::Button button, int x, int y, GL::platform::Window*) override;
	void mouseMove(int x, int y, GL::platform::Window*) override;
	void mouseWheel(int delta, GL::platform::Window*) override;
};

#endif  // INCLUDED_UTILS_ORBITAL_NAVIGATOR
