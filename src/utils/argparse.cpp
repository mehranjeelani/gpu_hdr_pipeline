//#include <charconv>
#include <cstring>

#include "argparse.h"


namespace
{
	bool compareOption(const char* arg, std::string_view option)
	{
		return std::strncmp(arg, option.data(), option.length()) == 0;
	}
}

namespace argparse
{
	bool parseBoolFlag(const char* const*& argv, std::string_view option)
	{
		if (!compareOption(*argv, option))
			return false;

		++argv;
		return true;
	}

	bool parseStringArgument(const char*& value, const char* const*& argv, std::string_view option)
	{
		if (!compareOption(*argv, option))
			return false;

		const char* startptr = *argv + option.length();

		if (*startptr)
		{
			value = startptr;
			return true;
		}

		startptr = *++argv;

		if (!*startptr)
			throw usage_error("expected argument");

		value = startptr;
		return true;
	}

	bool parseIntArgument(int& value, const char* const*& argv, std::string_view option)
	{
		if (!compareOption(*argv, option))
			return false;

		const char* startptr = *argv + option.length();

		if (!*startptr)
		{
			startptr = *++argv;
			if (!*startptr)
				throw usage_error("expected integer argument");
		}

		char* endptr = nullptr;

		int v = std::strtol(startptr, &endptr, 10);

		if (*endptr)
			throw usage_error("argument is not an integer");

		value = v;
		return true;
	}

	bool parseFloatArgument(float& value, const char* const*& argv, std::string_view option)
	{
		if (!compareOption(*argv, option))
			return false;

		const char* startptr = *argv + option.length();

		if (!*startptr)
		{
			startptr = *++argv;
			if (!*startptr)
				throw usage_error("expected float argument");
		}

		char* endptr = nullptr;

		float v = std::strtof(startptr, &endptr);

		if (*endptr)
			throw usage_error("argument is not a float");

		value = v;
		return true;
	}
}
