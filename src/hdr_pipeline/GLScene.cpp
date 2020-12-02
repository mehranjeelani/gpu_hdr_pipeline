#include <cmath>

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

	vec4 p_2 = camera.PV_inv * vec4(gl_Position.xy, 0.0f, 1.0f);
	vec4 p_1 = camera.PV_inv * vec4(gl_Position.xy, -1.0f, 1.0f);

	d = p_2.xyz / p_2.w - p_1.xyz / p_1.w;
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
}
)""";

	const char vs_model[] = R"""(
#version 430
)"""
CAMERA
R"""(

layout(location = 0) in vec3 v_p;
layout(location = 1) in vec3 v_n;

out vec3 a_p;
out vec3 a_n;

void main()
{
	gl_Position = camera.PV * vec4(v_p, 1.0f);
	a_p = v_p;
	//a_n = (vec4(v_n, 0.0f) * camera.V_inv).xyz;
	a_n = v_n;
}
)""";

	const char fs_model[] = R"""(
#version 430
)"""
CAMERA
LATLONG
R"""(

layout(location = 0) uniform sampler2D envmap;
layout(location = 1) uniform vec3 albedo = vec3(0.1f, 0.1f, 0.1f);

in vec3 a_p;
in vec3 a_n;

layout(location = 0) out vec4 color;

void main()
{
	vec3 n = normalize(a_n);
	vec3 v = normalize(camera.position - a_p);
	vec3 r = reflect(-v, n);

	float lambert = max(dot(n, v), 0.0f);

	float bla = 1.0f - lambert;
	float bla2 = bla * bla;

	const float R_0 = 0.14f;

	float R = R_0 + (1 - R_0) * bla2 * bla2 * bla;

	color = vec4(R * texture(envmap, lat_long(r)).rgb + (1.0f - R) * albedo * lambert, 1.0f);
}
)""";
}

GLScene::GLScene(const Camera& camera, const image2D<std::array<float, 4>>& env, const float* vertex_data, GLsizei num_vertices, const std::uint32_t* index_data, GLsizei num_indices)
	: camera(camera), num_indices(num_indices)
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
		glTexStorage2D(GL_TEXTURE_2D, /*static_cast<GLsizei>(std::log2(std::max(width(env), height(env)))) +*/ 1, GL_RGBA16F, static_cast<GLsizei>(width(env)), static_cast<GLsizei>(height(env)));
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, static_cast<GLsizei>(width(env)), static_cast<GLsizei>(height(env)), GL_RGBA, GL_FLOAT, data(env));
		//glGenerateMipmap(GL_TEXTURE_2D);

		//glSamplerParameteri(envmap_sampler, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
		//glSamplerParameteri(envmap_sampler, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
		//glSamplerParameteri(envmap_sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	}
	GL::throw_error();

	if (num_indices)
	{
		glBindVertexArray(vao_model);
		glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
		glBufferStorage(GL_ARRAY_BUFFER, num_vertices * 6 * 4U, vertex_data, 0U);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
		glBufferStorage(GL_ELEMENT_ARRAY_BUFFER, num_indices * 4U, index_data, 0U);

		glBindVertexBuffer(0U, vertex_buffer, 0U, 24U);
		glEnableVertexAttribArray(0U);
		glEnableVertexAttribArray(1U);
		glVertexAttribFormat(0U, 3, GL_FLOAT, GL_FALSE, 0U);
		glVertexAttribFormat(1U, 3, GL_FLOAT, GL_FALSE, 12U);
		glVertexAttribBinding(0U, 0U);
		glVertexAttribBinding(1U, 0U);
		GL::throw_error();
	}
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
	//glBindSampler(0, envmap_sampler);
	glUniform1i(0, 0);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 3);
	GL::throw_error();


	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);

	glBindVertexArray(vao_model);
	glUseProgram(prog_model);
	glUniform1i(0, 0);
	glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, nullptr);
	GL::throw_error();
}
