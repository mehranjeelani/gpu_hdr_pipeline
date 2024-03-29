#ifndef INCLUDED_UTILS_IO
#define INCLUDED_UTILS_IO

#pragma once

#include <type_traits>
#include <cstddef>
#include <iostream>


template <typename T>
T read(std::istream& in)
{
	static_assert(std::is_trivially_copyable_v<T>);

	T value;
	in.read(reinterpret_cast<char*>(&value), sizeof(T));
	return value;
}

template <typename T>
std::istream& read(T& value, std::istream& in)
{
	static_assert(std::is_trivially_copyable_v<T>);
	return in.read(reinterpret_cast<char*>(&value), sizeof(T));
}

template <typename T>
std::istream& read(T* values, std::istream& in, std::size_t count)
{
	static_assert(std::is_trivially_copyable_v<T>);
	return in.read(reinterpret_cast<char*>(values), sizeof(T) * count);
}

template <typename T, std::size_t N>
std::istream& read(T(&values)[N], std::istream& in)
{
	static_assert(std::is_trivially_copyable_v<T>);
	return in.read(reinterpret_cast<char*>(values), sizeof(values));
}

template <typename T>
std::ostream& write(std::ostream& out, T value)
{
	static_assert(std::is_trivially_copyable_v<T>);
	return out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
std::ostream& write(std::ostream& out, const T* values, std::size_t count)
{
	static_assert(std::is_trivially_copyable_v<T>);
	return out.write(reinterpret_cast<const char*>(values), sizeof(T) * count);
}

template <typename T, std::size_t N>
std::ostream& write(std::ostream& out, const T(&values)[N])
{
	static_assert(std::is_trivially_copyable_v<T>);
	return out.write(reinterpret_cast<const char*>(values), sizeof(values));
}

#endif  // INCLUDED_UTILS_IO
