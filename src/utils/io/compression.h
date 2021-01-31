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

	int produce(std::ostream& file, int flush = Z_NO_FLUSH)
	{
		stream.avail_out = sizeof(buffer);
		stream.next_out = buffer;

		int ret = zlib_throw_error(deflate(&stream, flush));

		if (!file.write(reinterpret_cast<const char*>(buffer), stream.next_out - buffer))
			throw std::runtime_error("failed to write to particles file");

		return ret;
	}

public:
	zlib_writer()
	{
		stream.zalloc = &zalloc;
		stream.zfree = &zfree;
		stream.opaque = nullptr;

		zlib_throw_error(deflateInit(&stream, Z_BEST_COMPRESSION));
	}

	zlib_writer(zlib_writer&) = delete;
	zlib_writer& operator =(zlib_writer&) = delete;

	template <typename T>
	void operator ()(std::ostream& file, const T* data, std::size_t size)
	{
		static_assert(std::is_trivially_copyable_v<T>);

		stream.avail_in = size * sizeof(T);
		stream.next_in = reinterpret_cast<const Bytef*>(data);

		while (stream.avail_in)
			produce(file);
	}

	template <typename T>
	void operator ()(std::ostream& file, const T& data)
	{
		operator ()(file, &data, 1);
	}

	std::ostream& finish(std::ostream& file)
	{
		while (produce(file, Z_FINISH) != Z_STREAM_END);
		return file;
	}

	~zlib_writer()
	{
		deflateEnd(&stream);
	}
};

class zlib_reader
{
	z_stream stream;

	Bytef buffer[4096];

	bool more = true;

	void consume(std::istream& file, int flush = Z_NO_FLUSH)
	{
		if (stream.avail_in == 0)
		{
			if (!file.read(reinterpret_cast<char*>(buffer), sizeof(buffer)))
				if (!file.eof() || file.gcount() == 0)
					throw std::runtime_error("failed to read from particles file");

			stream.avail_in = file.gcount();
			stream.next_in = buffer;
		}

		if (zlib_throw_error(inflate(&stream, flush)) == Z_STREAM_END)
		{
			while (stream.avail_in-- > 0)
				file.putback(*stream.next_in++);

			more = false;
		}
	}

public:
	zlib_reader()
	{
		stream.zalloc = &zalloc;
		stream.zfree = &zfree;
		stream.opaque = nullptr;
		stream.avail_in = 0;

		zlib_throw_error(inflateInit(&stream));
	}

	template <typename T>
	void operator ()(T* data, std::size_t size, std::istream& file)
	{
		static_assert(std::is_trivially_copyable_v<T>);

		stream.avail_out = size * sizeof(T);
		stream.next_out = reinterpret_cast<Bytef*>(data);

		while (stream.avail_out)
			consume(file);
	}

	template <typename T>
	void operator ()(T& data, std::istream& file)
	{
		(*this)(&data, 1, file);
	}

	template <typename T>
	T read(std::istream& file)
	{
		T data;
		operator ()(data, file);
		return data;
	}

	explicit operator bool() const { return more; }

	~zlib_reader()
	{
		inflateEnd(&stream);
	}
};

#endif // INCLUDED_UTILS_IO_COMPRESSION
