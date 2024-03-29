#ifndef INCLUDED_WIN32_PLATFORM
#define INCLUDED_WIN32_PLATFORM

#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>

#undef NOMINMAX
#undef WIN32_LEAN_AND_MEAN
#undef NEAR
#undef FAR
#undef near
#undef far

#endif  // INCLUDED_WIN32_PLATFORM
