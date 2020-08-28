#pragma once

#include "chroma-renderer/core/renderer/ChromaRenderer.h"

namespace chromagui
{

void MainMenu(ChromaRenderer* cr);
void DockSpace();
bool SettingsWindow(ChromaRenderer* cr);
bool ViewportWindow(ChromaRenderer* cr);
bool MaterialsWindow(ChromaRenderer* cr);

bool RenderGui(ChromaRenderer* cr);

} // namespace chromagui