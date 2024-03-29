#ifndef INCLUDED_X11_GL_WINDOW
#define INCLUDED_X11_GL_WINDOW

#pragma once

#include <memory>
#include <stdexcept>

#include <x11/platform.h>

#include <GL/platform/DisplayHandler.h>
#include <GL/platform/InputHandler.h>
#include <GL/platform/Renderer.h>

#include "x11_ptr.h"
#include "X11Display.h"
#include "X11GLContext.h"
#include "X11WindowHandle.h"
#include "X11ColormapHandle.h"


namespace X11::GL
{
	class Window
	{
		friend class WindowContextScopeState;
		friend void run(::GL::platform::Renderer& renderer, ::GL::platform::ConsoleHandler* console_handler);

		GLXFBConfig fb_config;
		std::unique_ptr<XVisualInfo, X11::deleter> vi;

		ColormapHandle colormap;
		WindowHandle window;

		::GL::platform::DisplayHandler* display_handler = nullptr;
		::GL::platform::MouseInputHandler* mouse_handler = nullptr;
		::GL::platform::KeyboardInputHandler* keyboard_handler = nullptr;

		void handleEvent(const XEvent& event);

	public:
		Window(const Window&) = delete;
		Window& operator =(const Window&) = delete;

		Window(const char* title, int width, int height, int depth_buffer_bits = 0, int stencil_buffer_bits = 0, bool stereo = false);
		~Window();

		void title(const char* title);

		void resize(int width, int height);

		Context createContext(int version_major, int version_minor, bool debug = false);

		void attach(::GL::platform::DisplayHandler* display_handler);
		void attach(::GL::platform::MouseInputHandler* mouse_handler);
		void attach(::GL::platform::KeyboardInputHandler* keyboard_handler);
	};


	class WindowContextScopeState
	{
	protected:
		static ::Display* display(Window& window)
		{
			extern X11::Display display;
			return display;
		}

		static GLXDrawable drawable(Window& window)
		{
			return window.window;
		}

	public:
		WindowContextScopeState(Window& window)
		{
		}
	};

	template <>
	struct SurfaceTypeTraits<Window>
	{
		using ContextScopeState = WindowContextScopeState;
	};
}

#endif  // INCLUDED_X11_GL_WINDOW
