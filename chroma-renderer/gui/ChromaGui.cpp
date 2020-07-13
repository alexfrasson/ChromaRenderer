#include "chroma-renderer/gui/ChromaGui.h"
#include "chroma-renderer/gui/ChromaGUIUtils.h"

#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <nfd.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/rotate_vector.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>

constexpr auto RADTODEGREE = 57.295779513082320876798154814105f;
constexpr auto DEGREETORAD = 0.01745329251994329576923690768489f;

namespace fs = std::filesystem;

ImVec2 mainMenuBarSize;

const int max_frames = 32;
std::vector<float> frameTimes(max_frames);
int currentFrameTimeIndex = 0;

float movementSpeed = 30.0f;

void ChromaGui::MainMenu(GLFWwindow* /*window*/, ChromaRenderer* cr)
{
    if (ImGui::BeginMainMenuBar())
    {
        mainMenuBarSize = ImGui::GetWindowSize();

        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::BeginMenu("Import"))
            {
                static std::string path;

                if (ImGui::MenuItem("Scene...",
                                    nullptr,
                                    nullptr,
                                    !(cr->getState() == ChromaRenderer::State::LOADINGSCENE)))
                {
                    path.clear();

                    nfdchar_t* outPath = NULL;
                    nfdresult_t result = NFD_OpenDialog("obj,dae,gltf", NULL, &outPath);
                    if (result == NFD_OKAY)
                    {
                        path = std::string(outPath);

                        if (fs::exists(path))
                        {
                            if (cr->isRunning())
                                cr->stopRender();

                            std::cout << path << std::endl;
                            cr->importScene(path);
                        }
                        free(outPath);
                    }
                    // else if (result == NFD_CANCEL)
                    // {
                    //     puts("User pressed cancel.");
                    // }
                    // else
                    // {
                    //     printf("Error: %s\n", NFD_GetError());
                    // }
                }
                if (ImGui::MenuItem("Enviroment Map..."))
                {
                    path.clear();

                    nfdchar_t* outPath = NULL;
                    nfdresult_t result = NFD_OpenDialog("hdr,jpg,png", NULL, &outPath);
                    if (result == NFD_OKAY)
                    {
                        path = std::string(outPath);

                        if (fs::exists(path))
                        {
                            bool wasRendering = cr->isRunning();
                            if (wasRendering)
                                cr->stopRender();

                            std::cout << path << std::endl;
                            cr->importEnviromentMap(path);

                            if (wasRendering)
                                cr->startRender();
                        }
                        free(outPath);
                    }
                    // else if (result == NFD_CANCEL)
                    // {
                    //     puts("User pressed cancel.");
                    // }
                    // else
                    // {
                    //     printf("Error: %s\n", NFD_GetError());
                    // }
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}

void ChromaGui::DockSpace()
{
    static bool opt_fullscreen_persistant = true;
    static ImGuiDockNodeFlags opt_flags =
        ImGuiDockNodeFlags_None; // | ImGuiDockNodeFlags_PassthruInEmptyNodes | ImGuiDockNodeFlags_RenderWindowBg;
    bool opt_fullscreen = opt_fullscreen_persistant;

    // We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
    // because it would be confusing to have two docking targets within each others.
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
    if (opt_fullscreen)
    {
        ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->Pos);
        ImGui::SetNextWindowSize(viewport->Size);
        ImGui::SetNextWindowViewport(viewport->ID);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                        ImGuiWindowFlags_NoMove;
        window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
    }

    // When using ImGuiDockNodeFlags_RenderWindowBg or ImGuiDockNodeFlags_InvisibleDockspace, DockSpace() will render
    // our background and handle the pass-thru hole, so we ask Begin() to not render a background.
    // if (opt_flags & ImGuiDockNodeFlags_RenderWindowBg)
    //	ImGui::SetNextWindowBgAlpha(0.0f);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("DockSpace Demo", nullptr, window_flags);
    ImGui::PopStyleVar();

    if (opt_fullscreen)
        ImGui::PopStyleVar(2);

    // Dockspace
    ImGuiIO& io = ImGui::GetIO();
    if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable)
    {
        ImGuiID dockspace_id = ImGui::GetID("MyDockspace");
        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), opt_flags);

        static bool should_init_layout = true;
        if (should_init_layout)
        {
            should_init_layout = false;

            ImGui::DockBuilderRemoveNode(dockspace_id);
            ImGui::DockBuilderAddNode(dockspace_id, opt_flags | ImGuiDockNodeFlags_DockSpace);
            ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);

            ImGuiID dock_main_id = dockspace_id; // This variable will track the document node, however we are not using
                                                 // it here as we aren't docking anything into it.
            ImGuiID dock_id_left = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, 0.16f, NULL, &dock_main_id);
            ImGuiID dock_id_right =
                ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Right, 0.20f, NULL, &dock_main_id);
            ImGuiID dock_id_left_bottom =
                ImGui::DockBuilderSplitNode(dock_id_left, ImGuiDir_Down, 0.20f, NULL, &dock_id_left);

            ImGui::DockBuilderDockWindow("Settings", dock_id_left);
            ImGui::DockBuilderDockWindow("Material Editor", dock_id_right);
            ImGui::DockBuilderDockWindow("EnvMapDebug", dock_main_id);
            ImGui::DockBuilderDockWindow("Viewport", dock_main_id);
            ImGui::DockBuilderDockWindow("Debug", dock_id_left_bottom);
            ImGui::DockBuilderFinish(dockspace_id);
        }
    }

    ImGui::End();
}

bool ChromaGui::MaterialsWindow(ChromaRenderer* cr)
{
    bool somethingChanged = false;

    // ImGui::SetNextWindowSize(ImVec2(300, 500), ImGuiCond_::ImGuiCond_Once);
    if (!ImGui::Begin("Material Editor"))
    {
        ImGui::End();
        return false;
    }

    // ShowHelpMarker("This example shows how you may implement a property editor using two columns.\nAll objects/fields
    // data are dummies here.\nRemember that in many simple cases, you can use ImGui::SameLine(xxx) to position\nyour
    // cursor horizontally instead of using the Columns() API.");

    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 2));
    ImGui::Columns(2);
    ImGui::Separator();

    struct funcs
    {
        static bool ShowDummyObject(Material& mat, int uid)
        {
            bool somethingChanged = false;

            ImGui::PushID(
                uid); // Use object uid as identifier. Most commonly you could also use the object pointer as a base ID.
            ImGui::AlignTextToFramePadding(); // Text and Tree nodes are less high than regular widgets, here we add
                                              // vertical spacing to make the tree lines equal high.
            bool node_open = ImGui::TreeNodeEx(mat.name.c_str());

            ImGui::NextColumn();
            ImGui::AlignTextToFramePadding();
            // ImGui::Text("my sailor is rich");

            ImGui::NextColumn();
            if (node_open)
            {
                ImGui::PushID(0);
                ImGui::AlignTextToFramePadding();
                ImGui::TreeNodeEx("KD",
                                  ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                      ImGuiTreeNodeFlags_Bullet);

                ImGui::NextColumn();
                ImGui::PushItemWidth(-1);
                if (ImGui::ColorEdit3("", &mat.kd.r))
                    somethingChanged = true;

                ImGui::PopItemWidth();
                ImGui::NextColumn();
                ImGui::PopID();

                ImGui::PushID(1);
                ImGui::AlignTextToFramePadding();
                ImGui::TreeNodeEx("KE",
                                  ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                      ImGuiTreeNodeFlags_Bullet);

                ImGui::NextColumn();
                ImGui::PushItemWidth(-1);
                if (ImGui::ColorEdit3("", &mat.ke.r))
                    somethingChanged = true;

                ImGui::PopItemWidth();
                ImGui::NextColumn();
                ImGui::PopID();

                ImGui::PushID(2);
                ImGui::AlignTextToFramePadding();
                ImGui::TreeNodeEx("Transparency",
                                  ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                      ImGuiTreeNodeFlags_Bullet);

                ImGui::NextColumn();
                ImGui::PushItemWidth(-1);
                if (ImGui::ColorEdit3("", &mat.transparent.r))
                    somethingChanged = true;

                ImGui::PopItemWidth();
                ImGui::NextColumn();
                ImGui::PopID();

                ImGui::TreePop();
            }
            ImGui::PopID();

            return somethingChanged;
        }
    };

    // Iterate dummy objects with dummy members (all the same data)
    for (size_t i = 0; i < cr->scene.materials.size(); i++)
        if (funcs::ShowDummyObject(cr->scene.materials[i], (int)i))
            somethingChanged = true;
    // for (int obj_i = 0; obj_i < 3; obj_i++)
    //	funcs::ShowDummyObject("Object", obj_i);

    ImGui::Columns(1);
    ImGui::Separator();
    ImGui::PopStyleVar();
    ImGui::End();

    return somethingChanged;
}

bool ChromaGui::SettingsWindow(ChromaRenderer* cr)
{
    bool somethingChanged = false;

    RendererSettings rs = cr->getSettings();

    // ImGui::SetNextWindowPos(glm::vec2(0, mainMenuBarSize.y), ImGuiCond_::ImGuiCond_Once);
    ImGui::Begin("Settings");
    {
        ImGui::Text("Average %.2f ms (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::Text("Current %.2f ms (%.1f FPS)", ImGui::GetIO().DeltaTime * 1000.0f, 1.0f / ImGui::GetIO().DeltaTime);

        ImGui::PushItemWidth(ImGui::GetContentRegionAvailWidth());
        ImGui::PlotLines("",
                         &frameTimes[0],
                         (int)frameTimes.size(),
                         currentFrameTimeIndex,
                         "",
                         0.0f,
                         150.0f,
                         ImVec2(0, 75));
        ImGui::PopItemWidth();

        // Settings
        {
            ImGui::PushItemWidth(175);

            if (!cr->isRunning())
            {
                ImGui::DragInt("Width", &rs.width, 1, 50, 8192);
                ImGui::DragInt("Height", &rs.height, 1, 50, 8192);
                ImGui::DragInt("Samples/pixel", &rs.samplesperpixel, 1, 1, 99999);
            }

            float hfov = rs.horizontalFOV * RADTODEGREE;
            if (ImGui::DragFloat("HFov", &hfov, 1, 5, 360))
            {
                rs.horizontalFOV = hfov * DEGREETORAD;
                cr->scene.camera.horizontalFOV(rs.horizontalFOV);
                somethingChanged = true;
            }

            ImGui::DragFloat("Apperture", &cr->scene.camera.apperture, 0.01f, 0.0001f, 1000.0f);
            ImGui::DragFloat("Shutter time", &cr->scene.camera.shutterTime, 0.01f, 0.0001f, 1000.0f);
            ImGui::DragFloat("ISO", &cr->scene.camera.iso, 1.0f, 1.0f, 6000.0f);
            ImGui::Checkbox("Adjust exposure", &cr->post_processor.adjustExposure);
            ImGui::Checkbox("Tonemapping", &cr->post_processor.tonemapping);
            ImGui::Checkbox("Linear to sRGB", &cr->post_processor.linearToSrbg);

            ImGui::DragFloat("Movement Speed", &movementSpeed, 1.0f, 0.0f, 10000.0f);

            ImGui::PopItemWidth();
        }

        if (rs != cr->getSettings())
            cr->setSettings(rs);

        if (!(cr->getState() == ChromaRenderer::State::LOADINGSCENE) &&
            ImGui::Button(cr->isRunning() ? "Stop" : "Render"))
        {
            if (cr->isRunning())
                cr->stopRender();
            else
            {
                cr->startRender(rs);
            }
        }

        if (ImGui::Button("Save"))
        {
            saveImage("image.bmp", &cr->final_target);
        }
    }
    ImGui::End();

    return somethingChanged;
}

ImVec2 GetAvailableRegionForImage(const float aspect_ratio)
{
    const ImVec2 availableRegion = ImGui::GetContentRegionAvail();
    const float windowAspectRatio = availableRegion.x / (float)availableRegion.y;

    float height;
    float width;

    if (windowAspectRatio < aspect_ratio)
    {
        width = availableRegion.x - 2;
        height = width * 1.0f / aspect_ratio;
        ImGui::SetCursorPosY(ImGui::GetCursorPos().y + (availableRegion.y - height) / 2.0f);
    }
    else
    {
        height = availableRegion.y - 2;
        width = height * aspect_ratio;
        ImGui::SetCursorPosX(ImGui::GetCursorPos().x + (availableRegion.x - width) / 2.0f);
    }

    return ImVec2(width, height);
}

void DrawImage(const Image& img, const bool flip_vert = false)
{
    const ImVec2 img_region = GetAvailableRegionForImage(img.getAspectRatio());
    ImGui::Image((ImTextureID)img.textureID,
                 img_region,
                 ImVec2(0.0f, flip_vert ? 0.0f : 1.0f),
                 ImVec2(1.0f, flip_vert ? 1.0f : 0.0f),
                 ImColor(255, 255, 255, 255),
                 ImColor(255, 255, 255, 200));
}

bool ChromaGui::ViewportWindow(ChromaRenderer* cr)
{
    bool somethingChanged = false;

    ImGui::Begin("Viewport");
    {
        ImGui::Text("%.3f MRays/sec", cr->cudaPathTracer.getInstantRaysPerSec() * 0.000001f);
        ImGui::SameLine();
        ImGui::ProgressBar(cr->cudaPathTracer.getProgress(),
                           ImVec2(-1, 0),
                           (std::to_string(cr->cudaPathTracer.getFinishedSamples()) + std::string("/") +
                            std::to_string(cr->cudaPathTracer.getTargetSamplesPerPixel()))
                               .c_str());

        const ImVec2 img_available_region = ImGui::GetContentRegionAvail();
        RendererSettings rs = cr->getSettings();
        rs.width = std::max(static_cast<int>(img_available_region.x), 10);
        rs.height = std::max(static_cast<int>(img_available_region.y), 10);

        if (rs != cr->getSettings())
        {
            cr->setSettings(rs);
            somethingChanged = true;
        }

        DrawImage(cr->final_target);

        if (cr->isRunning() && ImGui::IsWindowHovered(ImGuiHoveredFlags_::ImGuiHoveredFlags_None))
        {
            if (ImGui::IsMouseDragging(1))
            {
                const float lookSens = 0.4f;

                if (ImGui::GetIO().KeysDown[GLFW_KEY_W])
                {
                    cr->scene.camera.eye += cr->scene.camera.forward * ImGui::GetIO().DeltaTime * movementSpeed;
                    somethingChanged = true;
                }
                if (ImGui::GetIO().KeysDown[GLFW_KEY_S])
                {
                    cr->scene.camera.eye -= cr->scene.camera.forward * ImGui::GetIO().DeltaTime * movementSpeed;
                    somethingChanged = true;
                }
                if (ImGui::GetIO().KeysDown[GLFW_KEY_D])
                {
                    cr->scene.camera.eye += cr->scene.camera.right * ImGui::GetIO().DeltaTime * movementSpeed;
                    somethingChanged = true;
                }
                if (ImGui::GetIO().KeysDown[GLFW_KEY_A])
                {
                    cr->scene.camera.eye -= cr->scene.camera.right * ImGui::GetIO().DeltaTime * movementSpeed;
                    somethingChanged = true;
                }

                if (ImGui::GetIO().MouseDelta.x != 0 || ImGui::GetIO().MouseDelta.y != 0)
                {
                    glm::vec2 angle = glm::vec2(ImGui::GetIO().MouseDelta.x, ImGui::GetIO().MouseDelta.y) *
                                      ImGui::GetIO().DeltaTime * lookSens;

                    cr->scene.camera.forward = glm::normalize(rotateY(cr->scene.camera.forward, -angle.x));
                    cr->scene.camera.right = glm::normalize(glm::cross(cr->scene.camera.forward, glm::vec3(0, -1, 0)));
                    cr->scene.camera.up = -glm::normalize(glm::cross(cr->scene.camera.forward, cr->scene.camera.right));

                    cr->scene.camera.forward =
                        glm::normalize(glm::rotate(cr->scene.camera.forward, angle.y, cr->scene.camera.right));
                    cr->scene.camera.right = glm::normalize(glm::cross(cr->scene.camera.forward, glm::vec3(0, 1, 0)));
                    cr->scene.camera.up = -glm::normalize(glm::cross(cr->scene.camera.forward, cr->scene.camera.right));

                    somethingChanged = true;
                }
            }
        }
    }
    ImGui::End();

    ImGui::Begin("Debug");
    ImGui::LabelText("Image", "Image (%d, %d)", (int)cr->final_target.getWidth(), (int)cr->final_target.getHeight());
    ImGui::End();

    return somethingChanged;
}

void EnvMapDebug(ChromaRenderer* cr)
{
    ImGui::Begin("EnvMapDebug");

    DrawImage(cr->env_map, true);

    ImGui::End();
}

bool ChromaGui::RenderGui(GLFWwindow* window, ChromaRenderer* cr)
{
    bool somethingChanged = false;

    currentFrameTimeIndex = (currentFrameTimeIndex + 1) % (int)frameTimes.size();
    frameTimes[currentFrameTimeIndex] = ImGui::GetIO().DeltaTime * 1000.0f;

    MainMenu(window, cr);

    DockSpace();

    // ImGui::ShowDemoWindow();

    if (MaterialsWindow(cr))
    {
        cr->cudaPathTracer.setMaterials(cr->scene.materials);
        somethingChanged = true;
    }

    if (SettingsWindow(cr))
        somethingChanged = true;

    EnvMapDebug(cr);

    if (ViewportWindow(cr))
        somethingChanged = true;

    return somethingChanged;
}