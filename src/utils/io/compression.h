#ifndef INCLUDED_UTILS_IO_COMPRESSION
#define INCLUDED_UTILS_IO_COMPRESSION

#pragma once

#include <stdexcept>
#include <memory>
#include <fstream>
#include <string_view>
#include <iostream>

#include <zlib.h>


void* zalloc(voidpf opaque, uInt items, uInt size);
void zfree(voidpf opaque, voidpf address);

std::string_view zlib_error_name(int res);

inline int zlib_throw_error(int res)
{
	if (res < 0)
		throw std::runtime_error(std::string(zlib_error_name(res)));
	return res;
}


class zlib_writer
{
	z_stream stream;

	Bytef buffer[4096];

	std::ostream& file;


	int produce(int flush = Z_NO_FLUSH)
	{
		stream.avail_out = sizeof(buffer);
		stream.next_out = buffer;

		int ret = zlib_throw_error(deflate(&stream, flush));

		if (!file.write(reinterpret_cast<const char*>(buffer), stream.next_out - buffer))
			throw std::runtime_error("failed to write to particles file");

		return ret;
	}

public:
	zlib_writer(std::ostream& file)
		: file(file)
	{
		stream.zalloc = &zalloc;
		stream.zfree = &zfree;
		stream.opaque = nullptr;

		zlib_throw_error(deflateInit(&stream, Z_BEST_COMPRESSION));
	}

	zlib_writer(zlib_writer&) = delete;
	zlib_writer& operator =(zlib_writer&) = delete;

	template <typename T>
	void operator ()(const T* data, std::size_t size)
	{
		static_assert(std::is_trivially_copyable_v<T>);

		stream.avail_in = size * sizeof(T);
		stream.next_in = reinterpret_cast<const Bytef*>(data);

		while (stream.avail_in)
			produce();
	}

	template <typename T>
	void operator ()(const T& data)
	{
		static_assert(std::is_trivially_copyable_v<T>);
		operator ()(reinterpret_cast<const std::byte*>(&data), sizeof(data));
	}

	~zlib_writer()
	{
		while (produce(Z_FINISH) != Z_STREAM_END);
		deflateEnd(&stream);
	}
};

class zlib_reader
{
	z_stream stream;

	Bytef buffer[4096];

	std::istream& file;

	int consume(int flush = Z_NO_FLUSH)
	{
		if (stream.avail_in == 0)
		{
			if (!file.read(reinterpret_cast<char*>(buffer), sizeof(buffer)) && !file.eof())
				throw std::runtime_error("failed to read from particles file");

			stream.avail_in = file.gcount();
			stream.next_in = buffer;
		}

		return zlib_throw_error(inflate(&stream, flush));
	}

public:
	zlib_reader(std::istream& file)
		: file(file)
	{
		stream.zalloc = &zalloc;
		stream.zfree = &zfree;
		stream.opaque = nullptr;
		stream.avail_in = 0;

		zlib_throw_error(inflateInit(&stream));
	}

	template <typename T>
	void operator ()(T* data, std::size_t size)
	{
		static_assert(std::is_trivially_copyable_v<T>);

		stream.avail_out = size * sizeof(T);
		stream.next_out = reinterpret_cast<Bytef*>(data);

		while (stream.avail_out)
			consume();
	}

	template <typename T>
	void operator ()(T& data)
	{
		static_assert(std::is_trivially_copyable_v<T>);
		operator ()(reinterpret_cast<std::byte*>(&data), sizeof(data));
	}

	~zlib_reader()
	{
		while (stream.avail_in-- > 0)
			file.putback(*stream.next_in++);

		inflateEnd(&stream);
	}
};

#endif // INCLUDED_UTILS_IO_COMPRESSION
