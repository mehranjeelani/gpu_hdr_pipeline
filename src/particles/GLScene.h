#ifndef INCLUDED_GLSCENE
#define INCLUDED_GLSCENE

#pragma once

#include <GL/buffer.h>

#include <utils/Camera.h>


class GLScene
{
	const Camera* camera = nullptr;

	GL::Buffer camera_uniform_buffer;

protected:
	virtual void draw() const = 0;

	GLScene();

public:
	virtual ~GLScene() = default;

	void attach(const Camera* navigator);

	virtual void reset() = 0;
	virtual float update(int steps, float dt) = 0;
	void draw(int viewport_width, int viewport_height);
};

#endif  // INCLUDED_GLSCENE
