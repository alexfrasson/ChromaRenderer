#pragma once

#define GLFW_EXPOSE_NATIVE_WIN32

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <string>

#if defined(_WIN32) || defined(WIN32)
#include <Commdlg.h> //Open file dialog
#include <Windows.h>

bool importModelDialog(std::string& path, GLFWwindow* window)
{
    OPENFILENAME ofn;
    TCHAR szFile[MAX_PATH];

    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFile = szFile;
    ofn.lpstrFile[0] = '\0';
    ofn.hwndOwner = glfwGetWin32Window(window);
    ofn.nMaxFile = sizeof(szFile);
    // The Assimp 3.0 library provides importers for a lot of file formats, including:- 3DS- BLEND - DAE (Collada)-
    // IFC-STEP - ASE- DXF- HMP- MD2
    //- MD3 - MD5- MDC- MDL- NFF- PLY- STL- X - OBJ - SMD- LWO - LXO - LWS - XML - TER - AC3D - MS3D
    // A buffer containing pairs of null-terminated filter strings. The last string in the buffer must be terminated by
    // two NULL characters.
    ofn.lpstrFilter = TEXT("Supported "
                           "Formats\0*.3DS;*.BLEND;*.DAE;*.ASE;*.DXF;*.HMP;*.MD2;*.MD3;*.MD5;*.MDC;*.MDL;*.NFF;*.PLY;*."
                           "STL;*.X;*.OBJ;*.SMD;*.LWO;*.LXO;*.LWS;*.XML;*.TER;*.AC3D;*.MS3D\0");
    ofn.nFilterIndex = 0;
    ofn.lpstrInitialDir = NULL;
    ofn.lpstrFileTitle = NULL;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

    if (GetOpenFileName(&ofn))
    {
        path = ofn.lpstrFile;
        return true;
    }
    return false;
}
bool exportImageDialog(std::string& path, GLFWwindow* window)
{
    OPENFILENAME ofn;
    TCHAR szFile[MAX_PATH];

    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFile = szFile;
    // ofn.lpstrFile = "untitled.bmp";
    ofn.lpstrFile[0] = '\0';
    ofn.hwndOwner = glfwGetWin32Window(window);
    ofn.nMaxFile = sizeof(szFile);
    // A buffer containing pairs of null-terminated filter strings. The last string in the buffer must be terminated by
    // two NULL characters.
    ofn.lpstrFilter = TEXT("BMP(.bmp)\0*.bmp\0");
    ofn.lpstrDefExt = "bmp";
    ofn.nFilterIndex = 0;
    ofn.lpstrInitialDir = NULL;
    ofn.lpstrFileTitle = NULL;
    ofn.Flags = OFN_PATHMUSTEXIST;

    if (GetSaveFileName(&ofn))
    {
        path = ofn.lpstrFile;
        return true;
    }
    return false;
}
#else
bool importModelDialog(std::string& path, sf::WindowHandle handle)
{
}
bool exportImageDialog(std::string& path, sf::WindowHandle handle)
{
}
#endif