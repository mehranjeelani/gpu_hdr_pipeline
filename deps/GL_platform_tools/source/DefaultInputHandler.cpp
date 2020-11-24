#include <GL/platform/DefaultInputHandler.h>


namespace GL::platform
{
	void DefaultMouseInputHandler::buttonDown(Button, int, int, Window*)
	{
	}

	void DefaultMouseInputHandler::buttonUp(Button, int, int, Window*)
	{
	}

	void DefaultMouseInputHandler::mouseMove(int, int, Window*)
	{
	}

	void DefaultMouseInputHandler::mouseWheel(int, Window*)
	{
	}


	void DefaultKeyboardInputHandler::keyDown(Key, Window*)
	{
	}

	void DefaultKeyboardInputHandler::keyUp(Key, Window*)
	{
	}


	void DefaultConsoleHandler::command(const char*, std::size_t)
	{
	}
}
