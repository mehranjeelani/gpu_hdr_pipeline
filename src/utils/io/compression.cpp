#include <zlib.h>

#include "compression.h"

using namespace std::literals;


void* zalloc(voidpf opaque, uInt items, uInt size)
{
	return ::operator new(items * size);
}

void zfree(voidpf opaque, voidpf address)
{
	return ::operator delete(address);
}

std::string_view zlib_error_name(int res)
{
	switch (res)
	{
	case Z_NEED_DICT:
		return "Z_NEED_DICT"sv;
	case Z_ERRNO:
		return "Z_ERRNO"sv;
	case Z_STREAM_ERROR:
		return "Z_STREAM_ERROR"sv;
	case Z_DATA_ERROR:
		return "Z_DATA_ERROR"sv;
	case Z_MEM_ERROR:
		return "Z_MEM_ERROR"sv;
	case Z_BUF_ERROR:
		return "Z_BUF_ERROR"sv;
	case Z_VERSION_ERROR:
		return "Z_VERSION_ERROR"sv;
	}

	return "unknown zlib error code"sv;
}
