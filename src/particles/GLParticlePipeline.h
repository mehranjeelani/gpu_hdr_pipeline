#ifndef INCLUDED_GL_PARTICLE_PIPELINE
#define INCLUDED_GL_PARTICLE_PIPELINE

#pragma once

#include <cstdint>

#include <GL/gl.h>

#include <GL/shader.h>
#include <GL/vertex_array.h>
#include <GL/buffer.h>


class GLParticlePipeline
{
	GL::Program particle_prog;

public:
	GLParticlePipeline();

	void draw(GLsizei num_particles, GLuint vao) const;
};

#endif  // INCLUDED_GL_PARTICLE_PIPELINE
