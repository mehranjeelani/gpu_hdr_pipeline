#include <iterator>

#include <GL/error.h>

#include <utils/pfm.h>
#include <utils/Camera.h>

#include "GLScene.h"


namespace
{
#define CAMERA R"""(
struct Camera
{
	mat4x4 V;
	mat4x4 V_inv;
	mat4x4 P;
	mat4x4 P_inv;
	mat4x4 PV;
	mat4x4 PV_inv;
	vec3 position;
};

layout(std140, row_major, binding = 0) uniform CameraUniformBuffer
{
	Camera camera;
};
)"""

#define LATLONG R"""(
vec2 lat_long(vec3 d)
{
	const float pi = 3.14159265358979f;
	float lambda = atan(d.x, d.z) * (0.5f / pi) + 0.5f;
	float phi = normalize(d).y * 0.5f + 0.5f;
	return vec2(lambda, phi);
}
)"""

	const char vs_env[] = R"""(
#version 430
)"""
CAMERA
R"""(

out vec3 d;

void main()
{
	vec2 p = vec2((gl_VertexID & 0x2) * 0.5f, (gl_VertexID & 0x1));
	gl_Position = vec4(p * 4.0f - 1.0f, 1.0f, 1.0f);

	vec4 p_far = camera.PV_inv * gl_Position;
	vec4 p_near = camera.PV_inv * vec4(gl_Position.xy, -1.0f, 1.0f);

	d = p_far.xyz / p_far.w - p_near.xyz / p_near.w;
	//d = vec3(2.0f * p, 0.0f);
}
)""";

	const char fs_env[] = R"""(
#version 430

layout(location = 0) uniform sampler2D envmap;

in vec3 d;

layout(location = 0) out vec4 color;

)"""
LATLONG
R"""(

void main()
{
	color = texture(envmap, lat_long(d));
	//color = texture(envmap, d.xy);
	//color = vec4(d, 1.0f);
}
)""";

	const char vs_model[] = R"""(
#version 430
)"""
CAMERA
R"""(

layout(location = 0) in vec3 v_p;
layout(location = 1) in vec3 v_n;

out vec3 a_n;

void main()
{
	gl_Position = camera.PV * vec4(v_p, 1.0f);
	a_n = (vec4(v_n, 0.0f) * camera.V_inv).xyz;
}
)""";

	const char fs_model[] = R"""(
#version 430

layout(location = 0) uniform vec4 albedo;

in vec3 a_n;

layout(location = 0) out vec4 color;

void main()
{
	vec3 n = normalize(a_n);

	//float lambert = 1.0f;
	//float lambert = max(dot(n, l), 0.0f);

	//color = vec4(albedo.rgb * lambert, albedo.a);
	//color = vec4(-n.zzz, 1.0f);
	color = vec4(n, 1.0f);
	//color = vec4(t, 0.0f, 1.0f);
}
)""";
}

GLScene::GLScene(const Camera& camera, const image2D<std::array<float, 4>>& env)
	: camera(camera)
{
	{
		auto vs = GL::compileVertexShader(::vs_env);
		auto fs = GL::compileFragmentShader(::fs_env);
		glAttachShader(prog_env, vs);
		glAttachShader(prog_env, fs);
		GL::linkProgram(prog_env);
	}

	{
		auto vs = GL::compileVertexShader(::vs_model);
		auto fs = GL::compileFragmentShader(::fs_model);
		glAttachShader(prog_model, vs);
		glAttachShader(prog_model, fs);
		GL::linkProgram(prog_model);
	}


	glBindBuffer(GL_UNIFORM_BUFFER, camera_uniform_buffer);
	glBufferStorage(GL_UNIFORM_BUFFER, sizeof(Camera::UniformBuffer), nullptr, GL_DYNAMIC_STORAGE_BIT);
	GL::throw_error();


	{
		glBindTexture(GL_TEXTURE_2D, envmap);
		glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA16F, static_cast<GLsizei>(width(env)), static_cast<GLsizei>(height(env)));
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, static_cast<GLsizei>(width(env)), static_cast<GLsizei>(height(env)), GL_RGBA, GL_FLOAT, data(env));
	}
	GL::throw_error();


	//const float vertices[] = {
	// -1.0f, 1.0f,-1.0f, 0.0f, 0.0f,-1.0f,
	// -1.0f,-1.0f,-1.0f, 0.0f, 0.0f,-1.0f,
	// 1.0f, 1.0f,-1.0f, 0.0f, 0.0f,-1.0f,
	// 1.0f,-1.0f,-1.0f, 0.0f, 0.0f,-1.0f,

	// 1.0f, 1.0f,-1.0f, 1.0f, 0.0f, 0.0f,
	// 1.0f,-1.0f,-1.0f, 1.0f, 0.0f, 0.0f,
	// 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
	// 1.0f,-1.0f, 1.0f, 1.0f, 0.0f, 0.0f,

	// 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
	// 1.0f,-1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
	// -1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
	// -1.0f,-1.0f, 1.0f, 0.0f, 0.0f, 1.0f,

	// -1.0f, 1.0f, 1.0f, -1.0f, 0.0f, 0.0f,
	// -1.0f,-1.0f, 1.0f, -1.0f, 0.0f, 0.0f,
	// -1.0f, 1.0f,-1.0f, -1.0f, 0.0f, 0.0f,
	// -1.0f,-1.0f,-1.0f, -1.0f, 0.0f, 0.0f,

	// -1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f,
	// -1.0f, 1.0f,-1.0f, 0.0f, 1.0f, 0.0f,
	// 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f,
	// 1.0f, 1.0f,-1.0f, 0.0f, 1.0f, 0.0f,

	// -1.0f,-1.0f,-1.0f, 0.0f,-1.0f, 0.0f,
	// -1.0f,-1.0f, 1.0f, 0.0f,-1.0f, 0.0f,
	// 1.0f,-1.0f,-1.0f, 0.0f,-1.0f, 0.0f,
	// 1.0f,-1.0f, 1.0f, 0.0f,-1.0f, 0.0f
	//};

	//const GLuint indices[] = {
	//	 0,  1,  2,  1,  3,  2,
	//	 4,  5,  6,  5,  7,  6,
	//	 8,  9, 10,  9, 11, 10,
	//	12, 13, 14, 13, 15, 14,
	//	16, 17, 18, 17, 19, 18,
	//	20, 21, 22, 21, 23, 22
	//};

	//glBindVertexArray(vao_model);
	//glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
	//glBufferStorage(GL_ARRAY_BUFFER, sizeof(vertices), vertices, 0U);
	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
	//glBufferStorage(GL_ELEMENT_ARRAY_BUFFER, sizeof(num_indices), indices, 0U);

	//glBindVertexBuffer(0U, vertex_buffer, 0U, 24U);
	//glEnableVertexAttribArray(0U);
	//glEnableVertexAttribArray(1U);
	//glVertexAttribFormat(0U, 3, GL_FLOAT, GL_FALSE, 0U);
	//glVertexAttribFormat(1U, 3, GL_FLOAT, GL_FALSE, 12U);
	//glVertexAttribBinding(0U, 0U);
	//glVertexAttribBinding(1U, 0U);
	//GL::throw_error();

	//num_indices = static_cast<GLsizei>(std::size(indices));
}

void GLScene::draw(int framebuffer_width, int framebuffer_height) const
{
	Camera::UniformBuffer camera_params;
	camera.writeUniformBuffer(&camera_params, framebuffer_width * 1.0f / framebuffer_height);
	glBindBufferBase(GL_UNIFORM_BUFFER, 0U, camera_uniform_buffer);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(Camera::UniformBuffer), &camera_params);

	//glEnable(GL_CULL_FACE);
	//glDisable(GL_CULL_FACE);
	//glFrontFace(GL_CCW);

	glClearColor(0.6f, 0.7f, 1.0f, 1.0f);
	glClearDepth(1.0f);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	glDisable(GL_DEPTH_TEST);
	glDepthMask(GL_FALSE);

	glBindVertexArray(vao_env);
	glUseProgram(prog_env);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, envmap);
	glUniform1i(0, 0);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 3);
	GL::throw_error();


	//glEnable(GL_DEPTH_TEST);
	//glDepthMask(GL_TRUE);

	//glBindVertexArray(vao_model);
	//glUseProgram(prog_model);
	//glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, nullptr);
	//GL::throw_error();
}
