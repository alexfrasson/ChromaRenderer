#pragma once

#include "chroma-renderer/core/renderer/ChromaRenderer.h"

class ChromaGui
{
  public:
    static void MainMenu(ChromaRenderer* cr);
    static void DockSpace();
    static bool SettingsWindow(ChromaRenderer* cr);
    static bool ViewportWindow(ChromaRenderer* cr);
    static bool MaterialsWindow(ChromaRenderer* cr);

    static bool RenderGui(ChromaRenderer* cr);

  private:
    ChromaGui() = delete;
    ~ChromaGui() = delete;
};