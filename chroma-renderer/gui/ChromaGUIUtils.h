#pragma once

#include "chroma-renderer/core/types/Image.h"

#include <string>

const std::string getDateTime();
bool saveImage(std::string path, Image* img);