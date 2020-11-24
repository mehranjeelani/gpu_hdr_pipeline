#include <GL/platform/Application.h>
#include <GL/platform/DefaultDisplayHandler.h>


namespace GL::platform
{
	void DefaultDisplayHandler::close(Window*)
	{
		GL::platform::quit();
	}

	void DefaultDisplayHandler::destroy(Window*)
	{
	}

	void DefaultDisplayHandler::move(int, int, Window*)
	{
	}

	void DefaultDisplayHandler::resize(int, int, Window*)
	{
	}
}
