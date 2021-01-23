#ifndef INCLUDED_GL_PARTICLE_REPLAY
#define INCLUDED_GL_PARTICLE_REPLAY

#pragma once

#include <cstdint>

#include <GL/gl.h>

#include <GL/vertex_array.h>
#include <GL/buffer.h>


class GLParticleReplay
{
	GLsizei num_particles;

	GL::VertexArray vao;

	GL::Buffer particle_position_buffer;
	GL::Buffer particle_color_buffer;

public:
	GLParticleReplay();

	void draw();
};

#endif  // INCLUDED_GL_PARTICLE_REPLAY
