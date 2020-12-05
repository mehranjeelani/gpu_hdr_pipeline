#ifndef INCLUDED_UTILS_OBJ
#define INCLUDED_UTILS_OBJ

#pragma once

#include <stdexcept>
#include <filesystem>

#include <utils/math/vector.h>


namespace OBJ
{
	class parse_error : std::runtime_error
	{
		using std::runtime_error::runtime_error;
	};


	struct MeshSink
	{
		virtual int add_vertex(const math::float3& position, const math::float3& normal) = 0;
		virtual void add_triangle(int v_1, int v_2, int v_3) = 0;

	protected:
		MeshSink() = default;
		MeshSink(const MeshSink&) = default;
		MeshSink(MeshSink&&) = default;
		MeshSink& operator =(const MeshSink&) = default;
		MeshSink& operator =(MeshSink&&) = default;
		~MeshSink() = default;
	};

	MeshSink& readTriangles(MeshSink& sink, const char* begin, const char* end, const char* name);
	MeshSink& readTriangles(MeshSink& sink, const std::filesystem::path& path);
}

#endif  // INCLUDED_UTILS_OBJ
