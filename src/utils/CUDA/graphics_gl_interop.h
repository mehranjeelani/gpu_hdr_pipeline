#ifndef INCLUDED_UTILS_CUDA_GRAPHICS_GL_INTEROP
#define INCLUDED_UTILS_CUDA_GRAPHICS_GL_INTEROP

#pragma once

#include <cuda_gl_interop.h>

#include "error.h"
#include "graphics_interop.h"


namespace CUDA
{
	namespace graphics
	{
		inline unique_resource register_GL_buffer(GLuint buffer, unsigned int flags)
		{
			cudaGraphicsResource_t res;
			throw_error(cudaGraphicsGLRegisterBuffer(&res, buffer, flags));
			return unique_resource(res);
		}

		inline unique_resource register_GL_image(GLuint image, GLenum target, unsigned int flags)
		{
			cudaGraphicsResource_t res;
			throw_error(cudaGraphicsGLRegisterImage(&res, image, target, flags));
			return unique_resource(res);
		}
	}
}

#endif  // INCLUDED_UTILS_CUDA_GRAPHICS_GL_INTEROP
