#pragma once

#include "chroma-renderer/core/scene/Scene.h"

#include <functional>
namespace ModelImporter
{

bool import(const std::string& fileName, Scene& s);
void importcbscene(const std::string& fileName, Scene& s, const std::function<void()>& cb);

} // namespace ModelImporter