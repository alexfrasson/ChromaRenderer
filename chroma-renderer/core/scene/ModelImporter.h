#pragma once

#include "chroma-renderer/core/scene/Scene.h"

#include <functional>

namespace model_importer
{

bool import(const std::string& file_name, Scene& s);
void importcbscene(const std::string& file_name, Scene& s, const std::function<void()>& cb);

} // namespace ModelImporter