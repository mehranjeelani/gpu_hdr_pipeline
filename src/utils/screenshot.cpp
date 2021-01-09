#include <ctime>
#include <string_view>
#include <filesystem>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>

using namespace std::literals;


namespace
{
	std::filesystem::path build_screenshot_name(std::filesystem::path path, std::tm* time, int i)
	{
		std::ostringstream suffix;
		suffix << '_'
			<< time->tm_year + 1900 << '-' << std::setfill('0') << std::setw(2)
			<< time->tm_mon + 1 << '-' << std::setw(2)
			<< time->tm_mday << 'T' << std::setw(2)
			<< time->tm_hour << std::setw(2)
			<< time->tm_min << std::setw(2)
			<< time->tm_sec;

		if (i != 0)
			suffix << '_' << i;

		suffix << path.extension().string();

		return path.replace_extension() += suffix.str();
	}
}

std::ofstream open_screenshot_file(const std::filesystem::path& base_path)
{
	auto timestamp = std::time(nullptr);
	auto time = std::localtime(&timestamp);

	for (int i = 0; ; ++i)
	{
		auto filename = build_screenshot_name(base_path, time, i);

		std::ofstream file(filename, std::ios::binary);

		if (!file)
			std::cerr << "WARNING: failed to open "sv << filename << '\n';

		if (file.tellp() != 0)
			continue;

		return file;
	}
}
