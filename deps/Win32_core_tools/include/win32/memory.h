#ifndef INCLUDED_WIN32_MEMORY
#define INCLUDED_WIN32_MEMORY

#pragma once

#include <memory>

#include "platform.h"
#include "handle.h"
#include "unique_handle.h"


namespace Win32
{
	struct HeapHandleSpace : NullableHandleSpace<HANDLE, nullptr>
	{
		void close(HANDLE heap) const
		{
			assertClosed(HeapDestroy(heap));
		}
	};

	using unique_heap = unique_handle<HeapHandleSpace>;

	unique_heap createHeap(DWORD options, SIZE_T initial_size, SIZE_T maximum_size = 0U);


	class HeapFreeDeleter
	{
		HANDLE heap;

	public:
		HeapFreeDeleter(HANDLE heap)
			: heap(heap)
		{
		}

		void operator ()(void* p) const
		{
			assertClosed(HeapFree(heap, 0U, p));
		}
	};

	struct DefaultHeapDeleter
	{
		void operator ()(void* p) const
		{
			assertClosed(HeapFree(GetProcessHeap(), 0U, p));
		}
	};

	template <typename T, typename Del = DefaultHeapDeleter>
	using unique_heap_ptr = std::unique_ptr<T, Del>;


	struct HGLOBALHandleSpace : NullableHandleSpace<HGLOBAL, nullptr>
	{
		void close(HGLOBAL hmem) const
		{
			assertClosed(GlobalFree(hmem) == 0);
		}
	};

	using unique_hglobal = unique_handle<HGLOBALHandleSpace>;


	struct HLOCALHandleSpace : NullableHandleSpace<HLOCAL, nullptr>
	{
		void close(HLOCAL hmem) const
		{
			assertClosed(LocalFree(hmem) == 0);
		}
	};

	using unique_hlocal = unique_handle<HLOCALHandleSpace>;
}

#endif  // INCLUDED_WIN32_MODULE_HANDLE
