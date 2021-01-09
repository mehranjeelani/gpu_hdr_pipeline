#ifndef INCLUDED_DYNAMIC_LIBRARY
#define INCLUDED_DYNAMIC_LIBRARY

#pragma once

#include <stdexcept>
#include <string>
#include <memory>
#include <filesystem>

#ifdef _WIN32
#include <type_traits>
#include <win32/platform.h>
#include <win32/error.h>
#else
#include <dlfcn.h>
#endif


struct library_deleter
{
#ifdef _WIN32
	void operator ()(HMODULE module) const
	{
		FreeLibrary(module);
	}
#else
	void operator ()(void* module) const
	{
		dlclose(module);
	}
#endif
};

using unique_library =
#ifdef _WIN32
	std::unique_ptr<std::remove_pointer_t<HMODULE>, library_deleter>;
#else
	std::unique_ptr<void, library_deleter>;
#endif


inline auto load_library(const std::filesystem::path& path)
{
#ifdef _WIN32
	auto module = LoadLibraryW(path.c_str());

	if (!module)
		Win32::throw_last_error();

	return unique_library(module);
#else
	auto module = dlopen(path.c_str(), RTLD_NOW);

	if (!module)
		module = dlopen(std::filesystem::path(path).replace_extension(".so").c_str(), RTLD_NOW);

	if (!module)
		throw std::runtime_error("failed to load \"" + path.string() + '"');

	return unique_library(module);
#endif
}

template <typename T>
#ifdef _WIN32
T* lookup_symbol(HMODULE module, const char* name)
{
	auto sym = reinterpret_cast<T*>(GetProcAddress(module, name));
#else
T* lookup_symbol(void* module, const char* name)
{
	auto sym = reinterpret_cast<T*>(dlsym(module, name));
#endif

	using namespace std::literals;

	if (!sym)
		throw std::runtime_error("could not find symbol \""s + name + '"');

	return sym;
}

#endif  // INCLUDED_DYNAMIC_LIBRARY
