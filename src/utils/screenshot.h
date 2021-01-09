#ifndef INCLUDED_UTILS_SCREENSHOT
#define INCLUDED_UTILS_SCREENSHOT

#pragma once

#include <fstream>
#include <filesystem>


std::ofstream open_screenshot_file(const std::filesystem::path& base_path);

#endif  // INCLUDED_UTILS_SCREENSHOT
