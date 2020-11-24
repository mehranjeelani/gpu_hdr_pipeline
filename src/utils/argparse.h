#ifndef INCLUDED_ARGPARSE
#define INCLUDED_ARGPARSE

#include <string_view>
#include <stdexcept>


namespace argparse
{
	struct usage_error : std::runtime_error
	{
		using std::runtime_error::runtime_error;
	};


	bool parseBoolFlag(const char* const*& argv, std::string_view option);
	bool parseStringArgument(const char*& value, const char* const*& argv, std::string_view option);
	bool parseIntArgument(int& value, const char* const*& argv, std::string_view option);
	bool parseFloatArgument(float& value, const char* const*& argv, std::string_view option);
}

#endif  // INCLUDED_ARGPARSE
