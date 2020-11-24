#include "OrbitalNavigator.h"


OrbitalNavigator::OrbitalNavigator(float phi, float theta, float radius, const math::float3& lookat)
	: phi(phi),
	  theta(theta),
	  radius(radius),
	  lookat(lookat),
	  initial_phi(this->phi),
	  initial_theta(this->theta),
	  initial_radius(this->radius),
	  initial_lookat(this->lookat),
	  drag(0U)
{
	update();
}

void OrbitalNavigator::rotateH(float dphi)
{
	phi = math::fmod(phi + dphi, 2.0f * math::pi<float>);
}

void OrbitalNavigator::rotateV(float dtheta)
{
	theta = math::fmod(theta + dtheta, 2.0f * math::pi<float>);
}

void OrbitalNavigator::zoom(float dr)
{
	radius += dr;
}

void OrbitalNavigator::pan(float x, float y)
{
	lookat += x * u + y * v;
}

void OrbitalNavigator::update()
{
	float cp = math::cos(phi);
	float sp = math::sin(phi);
	float ct = math::cos(theta);
	float st = math::sin(theta);

	w = -math::float3(ct * cp, st, ct * sp);
	v = math::float3(-st * cp, ct, -st * sp);
	u = cross(v, w);

	position = lookat - radius * w;
}

void OrbitalNavigator::reset()
{
	lookat = initial_lookat;
	phi = initial_phi;
	theta = initial_theta;
	radius = initial_radius;
	update();
}

void OrbitalNavigator::writeWorldToLocalTransform(math::float4x4* M) const
{
	*M = {
		u.x, u.y, u.z, -dot(u, position),
		v.x, v.y, v.z, -dot(v, position),
		w.x, w.y, w.z, -dot(w, position),
		0.0f, 0.0f, 0.0f, 1.0f
	};
}

void OrbitalNavigator::writeLocalToWorldTransform(math::float4x4* M) const
{
	*M = {
		u.x, v.x, w.x, position.x,
		u.y, v.y, w.y, position.y,
		u.z, v.z, w.z, position.z,
		0.0f, 0.0f, 0.0f, 1.0f
	};
}

void OrbitalNavigator::writePosition(math::float3* p) const
{
	*p = OrbitalNavigator::position;
}


void OrbitalNavigator::buttonDown(GL::platform::Button button, int x, int y, GL::platform::Window*)
{
	drag |= static_cast<unsigned int>(button);
	last_pos = math::int2(x, y);
}

void OrbitalNavigator::buttonUp(GL::platform::Button button, int x, int y, GL::platform::Window*)
{
	drag &= ~static_cast<unsigned int>(button);
}

void OrbitalNavigator::mouseMove(int x, int y, GL::platform::Window*)
{
	if (drag)
	{
		math::int2 pos = math::int2(x, y);
		math::int2 d = pos - last_pos;

		if (drag & static_cast<unsigned int>(GL::platform::Button::LEFT))
		{
			rotateH(-d.x * 0.012f);
			rotateV(d.y * 0.012f);
		}

		if (drag & static_cast<unsigned int>(GL::platform::Button::RIGHT))
		{
			math::int2 absd = { math::abs(d.x), math::abs(d.y) };

			float dr = ((absd.y > absd.x) ? (d.y < 0 ? 1.0f : -1.0f) : (d.x > 0.0f ? 1.0f : -1.0f)) * math::sqrt(static_cast<float>(d.x * d.x + d.y * d.y)) * 0.012f * radius;
			zoom(dr);
		}

		if (drag & static_cast<unsigned int>(GL::platform::Button::MIDDLE))
		{
			pan(-d.x * radius * 0.007f, d.y * radius * 0.007f);
		}

		update();

		last_pos = pos;
	}
}

void OrbitalNavigator::mouseWheel(int delta, GL::platform::Window*)
{
	zoom(delta * 0.007f * radius);
	update();
}
