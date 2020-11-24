#ifndef INCLUDED_WIN32_EVENT_HANDLE
#define INCLUDED_WIN32_EVENT_HANDLE

#pragma once

#include "platform.h"
#include "handle.h"
#include "unique_handle.h"


namespace Win32
{
	struct EventHandleSpace : KernelObjectHandleSpace<NullableHandleSpace<HANDLE, nullptr>> {};

	using unique_hevent = unique_handle<EventHandleSpace>;

	unique_hevent createEvent();
}

#endif  // INCLUDED_WIN32_EVENT_HANDLE
