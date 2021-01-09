#version 430

#include <camera>


layout(points) in;
layout(triangle_strip, max_vertices = 4) out;


in VS_OUTPUT {
	vec4 position;
	vec4 color;
} vs_in[];


out vec2 pos;
out vec4 albedo;

void main()
{
	vec4 p = camera.PV * vec4(vs_in[0].position.xyz, 1.0f);

	vec2 s = vec2(vs_in[0].position.w / camera.aspect, vs_in[0].position.w);

	albedo = vs_in[0].color;

	gl_Position = p + vec4(-s.x, -s.y, 0.0f, 0.0f);
	pos = vec2(-1.0f, -1.0f);
	EmitVertex();
	gl_Position = p + vec4( s.x, -s.y, 0.0f, 0.0f);
	pos = vec2( 1.0f, -1.0f);
	EmitVertex();
	gl_Position = p + vec4(-s.x,  s.y, 0.0f, 0.0f);
	pos = vec2(-1.0f,  1.0f);
	EmitVertex();
	gl_Position = p + vec4( s.x,  s.y, 0.0f, 0.0f);
	pos = vec2( 1.0f,  1.0f);
	EmitVertex();
}
