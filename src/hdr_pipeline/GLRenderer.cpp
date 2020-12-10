#include <cstdint>
#include <memory>
#include <sstream>
#include <iostream>
#include <iomanip>

#include <GL/error.h>

#include <cuda_gl_interop.h>

#include <utils/CUDA/error.h>
#include <utils/CUDA/graphics_gl_interop.h>

#include "GLScene.h"
#include "GLRenderer.h"
#include "HDRPipeline.h"

using namespace std::literals;


#ifdef WIN32
extern "C" __declspec(dllexport) DWORD NvOptimusEnablement = 1U;
#endif

namespace
{
	const char vs[] = R"""(
#version 430

out vec2 t;

void main()
{
	vec2 p = vec2((gl_VertexID & 0x2) * 0.5f, (gl_VertexID & 0x1));
	gl_Position = vec4(p * 4.0f - 1.0f, 0.0f, 1.0f);
	t = 2.0f * p;
}
)""";

	const char fs[] = R"""(
#version 430

layout(location = 0) uniform sampler2D color_buffer;

in vec2 t;

layout(location = 0) out vec4 color;

void main()
{
	color = texture(color_buffer, t);
	//color = vec4(t, 0.0f, 1.0f);
}
)""";

#ifdef WIN32
	void APIENTRY debug_output_callback(GLenum /*source*/, GLenum /*type*/, GLuint /*id*/, GLenum /*severity*/, GLsizei /*length*/, const GLchar* message, const void* /*userParam*/)
	{
		OutputDebugStringA(message);
		OutputDebugStringA("\n");
	}
#else
	void debug_output_callback(GLenum /*source*/, GLenum /*type*/, GLuint /*id*/, GLenum /*severity*/, GLsizei /*length*/, const GLchar* message, const void* /*userParam*/)
	{
		std::cerr << message << '\n';
	}
#endif
}

GLRenderer::GLRenderer(std::string title, int width, int height, float exposure, float brightpass_threshold)
	: window(title.c_str(), 1024, 1024 * height / width),
	  context(window.createContext(4, 3, true)),
	  ctx(context, window),
	  exposure(exposure),
	  brightpass_threshold(brightpass_threshold),
	  title(std::move(title))
{
	std::cout << "\nOpenGL on " << glGetString(GL_RENDERER) << '\n' << std::flush;

	glDebugMessageCallback(debug_output_callback, nullptr);
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);


	glEnable(GL_FRAMEBUFFER_SRGB);

	auto vs = GL::compileVertexShader(::vs);
	auto fs = GL::compileFragmentShader(::fs);

	glAttachShader(prog, vs);
	glAttachShader(prog, fs);
	GL::linkProgram(prog);

	window.attach(this);
	GL::throw_error();
}

void GLRenderer::resize(int width, int height, GL::platform::Window*)
{
	framebuffer_width = width;
	framebuffer_height = height;

	hdr_buffer_resource.reset();
	ldr_buffer_resource.reset();

	hdr_buffer = GL::createTexture2D(framebuffer_width, framebuffer_height, 1, GL_RGBA32F);
	ldr_buffer = GL::createTexture2D(framebuffer_width, framebuffer_height, 1, GL_SRGB8_ALPHA8);
	depth_buffer = GL::createRenderbuffer(framebuffer_width, height, GL_DEPTH_COMPONENT32);

	hdr_buffer_resource = CUDA::graphics::register_GL_image(hdr_buffer, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
	ldr_buffer_resource = CUDA::graphics::register_GL_image(ldr_buffer, GL_TEXTURE_2D, 0U);

	pipeline.emplace(framebuffer_width, framebuffer_height);

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
	glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, hdr_buffer, 0);
	glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buffer);
}

void GLRenderer::render()
{
	if (framebuffer_width <= 0 || framebuffer_height <= 0)
		return;

	glViewport(0, 0, framebuffer_width, framebuffer_height);

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);

	if (scene)
		scene->draw(framebuffer_width, framebuffer_height);

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

	{
		auto mapped_resources = CUDA::graphics::map_resources(hdr_buffer_resource.get(), ldr_buffer_resource.get());

		cudaArray_t hdr_buffer_array = CUDA::graphics::get_mapped_array(hdr_buffer_resource.get());
		cudaArray_t ldr_buffer_array = CUDA::graphics::get_mapped_array(ldr_buffer_resource.get());

		if (pipeline)
		{
			throw_error(cudaEventRecord(pipeline_begin.get()));
			pipeline->process(ldr_buffer_array, hdr_buffer_array, exposure, brightpass_threshold);
			throw_error(cudaEventRecord(pipeline_end.get()));

			throw_error(cudaEventSynchronize(pipeline_end.get()));

			pipeline_time += CUDA::elapsed_time(pipeline_begin.get(), pipeline_end.get());

			++frame_count;

			if (auto now = std::chrono::steady_clock::now(); next_fps_tick <= now)
			{
				auto dt = std::chrono::duration<float>(now - next_fps_tick + 1s).count();

				std::ostringstream str;
				str << title << " @ t = " << std::setprecision(2) << std::fixed << pipeline_time / frame_count << " ms";

				window.title(str.str().c_str());

				next_fps_tick = now + 1s;
				frame_count = 0;
				pipeline_time = 0.0f;
			}
		}
	}

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glBindVertexArray(vao);
	glUseProgram(prog);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, ldr_buffer);
	glBindSampler(0, sampler);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 3);
	GL::throw_error();

	ctx.swapBuffers();
}

void GLRenderer::attach(GL::platform::MouseInputHandler* mouse_input)
{
	window.attach(mouse_input);
}

void GLRenderer::attach(GL::platform::KeyboardInputHandler* keyboard_input)
{
	window.attach(keyboard_input);
}

void GLRenderer::attach(const GLScene* scene)
{
	this->scene = scene;
}
