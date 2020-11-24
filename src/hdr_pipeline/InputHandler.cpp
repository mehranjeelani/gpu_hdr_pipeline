#include "InputHandler.h"


InputHandler::InputHandler(OrbitalNavigator& navigator)
	: navigator(navigator)
{
}

void InputHandler::keyDown(GL::platform::Key, GL::platform::Window*)
{
}

void InputHandler::keyUp(GL::platform::Key, GL::platform::Window*)
{
}

void InputHandler::buttonDown(GL::platform::Button button, int x, int y, GL::platform::Window* window)
{
	navigator.buttonDown(button, x, y, window);
}

void InputHandler::buttonUp(GL::platform::Button button, int x, int y, GL::platform::Window* window)
{
	navigator.buttonUp(button, x, y, window);
}

void InputHandler::mouseMove(int x, int y, GL::platform::Window* window)
{
	navigator.mouseMove(x, y, window);
}

void InputHandler::mouseWheel(int d, GL::platform::Window* window)
{
	navigator.mouseWheel(d, window);
}
