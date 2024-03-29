#ifndef INCLUDED_WIN32_GL_CONTEXT
#define INCLUDED_WIN32_GL_CONTEXT

#pragma once

#include <utility>
#include <memory>

#include <GL/gl.h>

#include <win32/platform.h>
#include <win32/error.h>
#include <win32/handle.h>
#include <win32/unique_handle.h>
#include <win32/glcore.h>


namespace Win32::GL
{
	struct GLContextHandleSpace : NullableHandleSpace<HGLRC, nullptr>
	{
		void close(HGLRC context) const
		{
			assertClosed(wglDeleteContext(context));
		}
	};

	using unique_hglrc = unique_handle<GLContextHandleSpace>;

	void setPixelFormat(HDC hdc, int depth_buffer_bits, int stencil_buffer_bits, bool stereo = false);
	unique_hglrc createContext(HDC hdc, int version_major, int version_minor, bool debug = false);


	struct glcoreContextDestroyDeleter
	{
		void operator ()(const glcoreContext* ctx)
		{
			glcoreContextDestroy(ctx);
		}
	};

	using unique_glcoreContext = std::unique_ptr<const glcoreContext, glcoreContextDestroyDeleter>;


	class Context
	{
		template <class SurfaceType>
		friend class context_scope;

		unique_hglrc hglrc;
		unique_glcoreContext ctx;

	public:
		Context(const Context&) = delete;
		Context& operator =(const Context&) = delete;

		Context(HDC hdc, int version_major, int version_minor, bool debug = false);

		Context(Context&& c)
			: hglrc(std::move(c.hglrc)),
			  ctx(std::move(c.ctx))
		{
		}

		Context& operator =(Context&& c)
		{
			using std::swap;
			swap(hglrc, c.hglrc);
			swap(ctx, c.ctx);
			return *this;
		}
	};


	template <class SurfaceType>
	struct SurfaceTypeTraits;

	template <class SurfaceType>
	class context_scope : private SurfaceTypeTraits<SurfaceType>::ContextScopeState
	{
		HDC hdc;
		HGLRC hglrc;
		const glcoreContext* ctx;

		HDC hdc_restore;
		HGLRC hglrc_restore;
		const glcoreContext* ctx_restore;

		void makeCurrent()
		{
			if (wglMakeCurrent(hdc, hglrc) != TRUE)
				Win32::throw_last_error();
			glcoreContextMakeCurrent(ctx);
		}

		using SurfaceTypeTraits<SurfaceType>::ContextScopeState::openHDC;
		using SurfaceTypeTraits<SurfaceType>::ContextScopeState::closeHDC;

	public:
		context_scope(const context_scope&) = delete;
		context_scope& operator =(const context_scope&) = delete;

		context_scope(Context& context, SurfaceType& surface)
			: SurfaceTypeTraits<SurfaceType>::ContextScopeState(surface),
			  hdc(openHDC()),
			  hglrc(context.hglrc),
			  ctx(context.ctx.get()),
			  hdc_restore(wglGetCurrentDC()),
			  hglrc_restore(wglGetCurrentContext()),
			  ctx_restore(glcoreContextGetCurrent())
		{
			makeCurrent();
		}

		~context_scope()
		{
			if (wglMakeCurrent(hdc_restore, hglrc_restore) != TRUE)
				Win32::throw_last_error();
			glcoreContextMakeCurrent(ctx_restore);
			closeHDC(hdc);
		}

		void bind(Context& context, SurfaceType& surface)
		{
			typename SurfaceTypeTraits<SurfaceType>::ContextScopeState old_state(surface);

			HDC old_hdc = hdc;

			using std::swap;
			swap(*this, old_state);
			hdc = openHDC();

			makeCurrent();

			old_state.releaseDC(old_hdc);
		}

		void setSwapInterval(int interval)
		{
			wglSwapIntervalEXT(interval);
		}

		void swapBuffers()
		{
			SwapBuffers(hdc);
		}
	};
}

#endif  // INCLUDED_WIN32_GL_CONTEXT
