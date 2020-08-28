#pragma once

#include "chroma-renderer/core/types/Image.h"

#include <string>

std::string getDateTime();
bool saveImage(const std::string& path, Image* img);