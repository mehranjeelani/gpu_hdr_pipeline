#ifndef INCLUDED_WIN32_HANDLE
#define INCLUDED_WIN32_HANDLE

#pragma once

#include <cassert>

#include "platform.h"


namespace Win32
{
	inline void assertClosed([[maybe_unused]] BOOL closed)
	{
		assert(closed);
	}

	template <typename T, T NULL_VALUE>
	struct NullableHandleSpace
	{
		using handle_type = T;

		handle_type null() const
		{
			return NULL_VALUE;
		}

		bool isNull(handle_type h) const
		{
			return h == NULL_VALUE;
		}
	};

	struct InvalidHandleValueNullHandleSpace
	{
		using handle_type = HANDLE;

		constexpr handle_type null() const
		{
			return INVALID_HANDLE_VALUE;
		}

		constexpr bool isNull(handle_type h) const
		{
			return h == INVALID_HANDLE_VALUE;
		}
	};

	template <typename NullHandleSpace>
	struct KernelObjectHandleSpace : NullHandleSpace
	{
		void close(HANDLE h) const
		{
			assertClosed(CloseHandle(h));
		}
	};

	HANDLE duplicateHandle(HANDLE source_process, HANDLE source_handle, HANDLE target_process, DWORD desired_access, BOOL inheritable, DWORD options = 0U);

	DWORD getHandleInformation(HANDLE h);
	void setHandleInformation(HANDLE h, DWORD mask, DWORD flags);
}

#endif  // INCLUDED_WIN32_HANDLE
