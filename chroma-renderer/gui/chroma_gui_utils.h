#pragma once

#include "chroma-renderer/core/types/image.h"

#include <string>

std::string getDateTime();
bool saveImage(const std::string& path, Image* img);