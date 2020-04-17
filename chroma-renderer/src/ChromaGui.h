#pragma once

#include <ChromaRenderer.h>
#include <glfw/glfw3.h>

class ChromaGui
{
  public:
    static void MainMenu(GLFWwindow* window, ChromaRenderer* cr);
    static void DockSpace();
    static bool SettingsWindow(ChromaRenderer* cr);
    static bool ViewportWindow(ChromaRenderer* cr);
    static bool MaterialsWindow(ChromaRenderer* cr);

    static bool RenderGui(GLFWwindow* window, ChromaRenderer* cr);

  private:
    ChromaGui() = delete;
    ~ChromaGui() = delete;
};