#pragma once

#include "chroma-renderer/core/renderer/ChromaRenderer.h"

namespace chromagui
{

void mainMenu(ChromaRenderer* cr);
void dockSpace();
bool settingsWindow(ChromaRenderer* cr);
bool viewportWindow(ChromaRenderer* cr);
bool materialsWindow(ChromaRenderer* cr);

bool renderGui(ChromaRenderer* cr);

} // namespace chromagui