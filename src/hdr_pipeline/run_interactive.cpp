#include <cstdint>
#include <iostream>

#include <GL/platform/Application.h>

#include <utils/PerspectiveCamera.h>
#include <utils/OrbitalNavigator.h>
#include <utils/image.h>

#include "envmap.h"
#include "InputHandler.h"
#include "GLRenderer.h"
#include "GLScene.h"
#include "HDRPipeline.h"


void run(const std::filesystem::path& output_file, const std::filesystem::path& envmap_path, const float* vertex_data, int num_vertices, const std::uint32_t* index_data, int num_indices, const math::float3& bb_min, const math::float3& bb_max, float exposure, float brightpass_threshold, int test_runs)
{
	if (test_runs != 1)
		std::cerr << "\nWARNING: test-runs parameter ignored in interactive mode\n";

	auto envmap = load_envmap(envmap_path, true);

	PerspectiveCamera camera(60.0f * math::pi<float> / 180.0f, 0.1f, 100.0f);
	OrbitalNavigator navigator(-math::pi<float> / 2, 0.0f, length(bb_max - bb_min) * 1.5f, (bb_min + bb_max) * 0.5f);

	camera.attach(&navigator);

	InputHandler input_handler(navigator);

	GLRenderer renderer(envmap_path.string(), static_cast<int>(width(envmap)), static_cast<int>(height(envmap)), exposure, brightpass_threshold);

	GLScene scene(camera, envmap, vertex_data, static_cast<GLsizei>(num_vertices), index_data, static_cast<GLsizei>(num_indices));

	renderer.attach(static_cast<GL::platform::MouseInputHandler*>(&input_handler));
	renderer.attach(static_cast<GL::platform::KeyboardInputHandler*>(&input_handler));
	renderer.attach(&scene);

	GL::platform::run(renderer);
}
