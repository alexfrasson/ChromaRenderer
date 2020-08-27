#include "chroma-renderer/gui/ChromaGui.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <argparse.hpp>
#include <glad/glad.h>
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

void term_func()
{
    std::cout << "Terminate!" << std::endl;
    auto eptr = std::current_exception();

    // if (eptr != nullptr)
    {
        try
        {
            std::rethrow_exception(eptr);
        }
        catch (const std::exception& e)
        {
            std::cout << "Unhandled exception1!" << std::endl;
            std::cout << e.what() << std::endl;
        }
        catch (...)
        {
            std::cout << "Unhandled exception2!" << std::endl;
        }
    }

    std::cout << "Terminate: done" << std::endl;
    std::abort();
}

GLFWwindow* g_window;
const char* g_glsl_version;

std::unique_ptr<ChromaRenderer> cr;

#ifndef WM_DPICHANGED
#define WM_DPICHANGED 0x02E0 // From Windows SDK 8.1+ headers // NOLINT
#endif

void CherryTheme()
{
    auto HI = [](const float v) { return ImVec4(0.502f, 0.075f, 0.256f, v); };
    auto MED = [](const float v) { return ImVec4(0.455f, 0.198f, 0.301f, v); };
    auto LOW = [](const float v) { return ImVec4(0.232f, 0.201f, 0.271f, v); };
    auto BG = [](const float v) { return ImVec4(0.200f, 0.220f, 0.270f, v); };
    auto TEXT_COLOR = [](const float v) { return ImVec4(0.860f, 0.930f, 0.890f, v); };

    auto& style = ImGui::GetStyle();
    style.Colors[ImGuiCol_Text] = TEXT_COLOR(0.78f);
    style.Colors[ImGuiCol_TextDisabled] = TEXT_COLOR(0.28f);
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.13f, 0.14f, 0.17f, 1.00f);
    // style.Colors[ImGuiCol_ChildWindowBg] = BG(0.58f);
    style.Colors[ImGuiCol_PopupBg] = BG(0.9f);
    style.Colors[ImGuiCol_Border] = ImVec4(0.31f, 0.31f, 1.00f, 0.00f);
    style.Colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    style.Colors[ImGuiCol_FrameBg] = BG(1.00f);
    style.Colors[ImGuiCol_FrameBgHovered] = MED(0.78f);
    style.Colors[ImGuiCol_FrameBgActive] = MED(1.00f);
    style.Colors[ImGuiCol_TitleBg] = LOW(1.00f);
    style.Colors[ImGuiCol_TitleBgActive] = HI(1.00f);
    style.Colors[ImGuiCol_TitleBgCollapsed] = BG(0.75f);
    style.Colors[ImGuiCol_MenuBarBg] = BG(0.47f);
    style.Colors[ImGuiCol_ScrollbarBg] = BG(1.00f);
    style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.09f, 0.15f, 0.16f, 1.00f);
    style.Colors[ImGuiCol_ScrollbarGrabHovered] = MED(0.78f);
    style.Colors[ImGuiCol_ScrollbarGrabActive] = MED(1.00f);
    style.Colors[ImGuiCol_CheckMark] = ImVec4(0.71f, 0.22f, 0.27f, 1.00f);
    style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.47f, 0.77f, 0.83f, 0.14f);
    style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.71f, 0.22f, 0.27f, 1.00f);
    style.Colors[ImGuiCol_Button] = ImVec4(0.47f, 0.77f, 0.83f, 0.14f);
    style.Colors[ImGuiCol_ButtonHovered] = MED(0.86f);
    style.Colors[ImGuiCol_ButtonActive] = MED(1.00f);
    style.Colors[ImGuiCol_Header] = MED(0.76f);
    style.Colors[ImGuiCol_HeaderHovered] = MED(0.86f);
    style.Colors[ImGuiCol_HeaderActive] = HI(1.00f);
    style.Colors[ImGuiCol_ResizeGrip] = ImVec4(0.47f, 0.77f, 0.83f, 0.04f);
    style.Colors[ImGuiCol_ResizeGripHovered] = MED(0.78f);
    style.Colors[ImGuiCol_ResizeGripActive] = MED(1.00f);
    style.Colors[ImGuiCol_PlotLines] = TEXT_COLOR(0.63f);
    style.Colors[ImGuiCol_PlotLinesHovered] = MED(1.00f);
    style.Colors[ImGuiCol_PlotHistogram] = TEXT_COLOR(0.63f);
    style.Colors[ImGuiCol_PlotHistogramHovered] = MED(1.00f);
    style.Colors[ImGuiCol_TextSelectedBg] = MED(0.43f);
    // [...]
    style.Colors[ImGuiCol_ModalWindowDarkening] = BG(0.73f);

    style.WindowPadding = ImVec2(6, 4);
    style.WindowRounding = 0.0f;
    style.FramePadding = ImVec2(5, 2);
    style.FrameRounding = 3.0f;
    style.ItemSpacing = ImVec2(7, 1);
    style.ItemInnerSpacing = ImVec2(1, 1);
    style.TouchExtraPadding = ImVec2(0, 0);
    style.IndentSpacing = 6.0f;
    style.ScrollbarSize = 12.0f;
    style.ScrollbarRounding = 16.0f;
    style.GrabMinSize = 20.0f;
    style.GrabRounding = 2.0f;

    style.WindowTitleAlign.x = 0.50f;

    style.Colors[ImGuiCol_Border] = ImVec4(0.539f, 0.479f, 0.255f, 0.162f);
    style.FrameBorderSize = 0.0f;
    style.WindowBorderSize = 1.0f;
}

bool InitializeImGui(GLFWwindow* window, const char* glsl_version)
{
    // Setup Dear ImGui binding
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    (void)io;

    io.ConfigFlags = 0;

    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls // NOLINT
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;     // Enable Docking // NOLINT
    // io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows
    // io.ConfigFlags |= ImGuiConfigFlags_ViewportsNoTaskBarIcons;
    // io.ConfigFlags |= ImGuiConfigFlags_ViewportsNoMerge;
    // io.ConfigFlags |= ImGuiConfigFlags_DpiEnableScaleFonts;     // FIXME-DPI: THIS CURRENTLY DOESN'T WORK AS
    // EXPECTED. DON'T USE IN USER APP! io.ConfigFlags |= ImGuiConfigFlags_DpiEnableScaleViewports; // FIXME-DPI

    if (!ImGui_ImplGlfw_InitForOpenGL(window, true))
    {
        std::cerr << "Failed to initialize 'ImGui_ImplGlfw_InitForOpenGL'" << std::endl;
        return false;
    }

    if (!ImGui_ImplOpenGL3_Init(glsl_version))
    {
        std::cerr << "Failed to initialize 'ImGui_ImplOpenGL3_Init'" << std::endl;
        return false;
    }

    io.ConfigWindowsResizeFromEdges = true;
    io.ConfigDockingWithShift = false;

    // Setup style
    ImGui::GetStyle().TabRounding = 0.0f;
    ImGui::GetStyle().ScrollbarRounding = 0.0f;
    ImGui::GetStyle().WindowRounding = 0.0f;

    // Load Fonts
    // - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use
    // ImGui::PushFont()/PopFont() to select them.
    // - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
    // - If the file cannot be loaded, the function will return nullptr. Please handle those errors in your application
    // (e.g. use an assertion, or display an error and quit).
    // - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling
    // ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
    // - Read 'misc/fonts/README.txt' for more instructions and details.
    // - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double
    // backslash \\ !
    // io.Fonts->AddFontDefault();
    io.Fonts->AddFontFromFileTTF("./chroma-renderer/resources/fonts/Roboto-Medium.ttf", 16.0f);
    // io.Fonts->AddFontFromFileTTF("../resources/DroidSans.ttf", 16.0f);
    // io.Fonts->AddFontFromFileTTF("../resources/Cousine-Regular.ttf", 16.0f);
    // ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, nullptr,
    // io.Fonts->GetGlyphRangesJapanese()); IM_ASSERT(font != nullptr);

    // ImGui::StyleColorsDark();
    // ImGui::StyleColorsClassic();
    CherryTheme();

    return true;
}

void ImGuiCleanup()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

static void glfw_error_callback(int error, const char* description)
{
    std::cerr << "Glfw Error " << error << ": " << description << std::endl;
}

bool InitGLFW()
{
    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (glfwInit() == GLFW_FALSE)
    {
        return false;
    }

    // Decide GL+GLSL versions
#if __APPLE__
    // GL 3.2 + GLSL 150
    glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);           // Required on Mac
#else
    // GL 3.0 + GLSL 130
    g_glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

    // Create window with graphics context
    g_window = glfwCreateWindow(1920, 1080, "ChromaRenderer", nullptr, nullptr);
    if (g_window == nullptr)
    {
        return false;
    }
    glfwMakeContextCurrent(g_window);
    // glfwSwapInterval(1); // Enable vsync
    glfwSwapInterval(0);

    return true;
}

void MainLoop()
{
    while (glfwWindowShouldClose(g_window) == 0)
    {
        ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your
        // inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those
        // two flags.
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code
        // to learn more about Dear ImGui!).
        // ImGui::ShowDemoWindow();

        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
        {
            bool somethingChanged = chromagui::RenderGui(cr.get());

            cr->update();

            if (cr->isRunning() && somethingChanged)
            {
                cr->stopRender();
                cr->startRender();
                cr->update();
            }
        }

        // Rendering
        ImGui::Render();
        int display_w{0};
        int display_h{0};
        glfwMakeContextCurrent(g_window);
        glfwGetFramebufferSize(g_window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwMakeContextCurrent(g_window);
        glfwSwapBuffers(g_window);
    }
}

bool ValidateArgs(argparse::ArgumentParser& program)
{
    if (const auto& scene_file_path = program.present<std::string>("-s"))
    {
        if (!fs::exists(*scene_file_path))
        {
            std::cerr << "File '" << *scene_file_path << "' does not exist." << std::endl;
            return false;
        }
    }

    if (const auto& env_map_file_path = program.present<std::string>("-e"))
    {
        if (!fs::exists(*env_map_file_path))
        {
            std::cerr << "File '" << *env_map_file_path << "' does not exist." << std::endl;
            return false;
        }
    }

    return true;
}

int main(const int argc, const char** argv) // NOLINT(bugprone-exception-escape)
{
    std::set_terminate(term_func);

    argparse::ArgumentParser program("chroma-renderer");
    program.add_argument("-r", "--render")
        .help("Start rendering immediately.")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("-s", "--scene").help("Scene file path.");
    program.add_argument("-e", "--env_map").help("Environemnt map file path.");

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err)
    {
        std::cout << err.what() << std::endl;
        std::cout << program;
        exit(0);
    }

    if (!ValidateArgs(program))
    {
        return 1;
    }

    if (!InitGLFW())
    {
        return 1;
    }

    if (gladLoadGL() == 0)
    {
        std::cerr << "Failed to initialize OpenGL loader!" << std::endl;
        return 1;
    }

    InitializeImGui(g_window, g_glsl_version);

    cr = std::make_unique<ChromaRenderer>();

    if (const auto& scene_file_path = program.present<std::string>("-s"))
    {
        cr->importScene(*scene_file_path);
    }

    if (const auto& env_map_file_path = program.present<std::string>("-e"))
    {
        cr->importEnviromentMap(*env_map_file_path);
    }

    if (program["-r"] == true)
    {
        cr->startRender();
    }

    MainLoop();

    cr.reset();

    ImGuiCleanup();

    glfwDestroyWindow(g_window);
    glfwTerminate();

    return 0;
}
