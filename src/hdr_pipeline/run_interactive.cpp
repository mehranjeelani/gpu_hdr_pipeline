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


void run(const char* envmap_path, float exposure, float brightpass_threshold, int test_runs)
{
	if (test_runs != 1)
		std::cerr << "WARNING: test-runs parameter ignored in interactive mode\n";

	auto envmap = load_envmap(envmap_path, true);

	PerspectiveCamera camera(60.0f * math::pi<float> / 180.0f, 0.1f, 100.0f);
	OrbitalNavigator navigator(-math::pi<float> / 2, 0.0f, 10.0f, {0.0f, 0.0f, 0.0f});

	camera.attach(&navigator);

	InputHandler input_handler(navigator);

	GLRenderer renderer(static_cast<int>(width(envmap)), static_cast<int>(height(envmap)), exposure, brightpass_threshold);

	GLScene scene(camera, envmap);

	renderer.attach(static_cast<GL::platform::MouseInputHandler*>(&input_handler));
	renderer.attach(static_cast<GL::platform::KeyboardInputHandler*>(&input_handler));
	renderer.attach(&scene);

	GL::platform::run(renderer);
}
