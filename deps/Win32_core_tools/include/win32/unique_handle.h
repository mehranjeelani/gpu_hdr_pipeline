#ifndef INCLUDED_WIN32_UNIQUE_HANDLE
#define INCLUDED_WIN32_UNIQUE_HANDLE

#pragma once

#include <utility>


namespace Win32
{
	template <class HandleSpace>
	class unique_handle : HandleSpace
	{
	public:
		using handle_type = typename HandleSpace::handle_type;
		using handle_space_type = HandleSpace;

	private:
		handle_type h;

		void close(handle_type handle) noexcept
		{
			if (!HandleSpace::isNull(handle))
				HandleSpace::close(handle);
		}

	public:
		unique_handle(const unique_handle&) = delete;
		unique_handle& operator =(const unique_handle&) = delete;
		
		unique_handle() noexcept
			: h(HandleSpace::null())
		{
		}

		explicit unique_handle(handle_type handle) noexcept
			: h(handle)
		{
		}

		unique_handle(handle_type handle, const HandleSpace& handle_space) noexcept
			: HandleSpace(handle_space),
			  h(handle)
		{
		}

		unique_handle(handle_type handle, HandleSpace&& handle_space) noexcept
			: HandleSpace(std::move(handle_space)),
			  h(handle)
		{
		}

		unique_handle(unique_handle&& handle) noexcept
			: HandleSpace(std::move(static_cast<HandleSpace&&>(handle))),
			  h(handle.h)
		{
			handle.h = HandleSpace::null();
		}

		~unique_handle()
		{
			close(h);
		}

		operator handle_type() const noexcept { return h; }

		unique_handle& operator =(unique_handle&& handle) noexcept
		{
			using std::swap;
			swap(*this, handle);
			return *this;
		}

		handle_type release() noexcept
		{
			handle_type temp = h;
			h = HandleSpace::null();
			return temp;
		}

		void reset(handle_type handle = HandleSpace::null()) noexcept
		{
			using std::swap;
			swap(this->h, handle);
			close(handle);
		}

		friend void swap(unique_handle& a, unique_handle& b) noexcept
		{
			using std::swap;
			swap(a.h, b.h);
		}
	};
}

#endif  // INCLUDED_WIN32_UNIQUE_HANDLE
