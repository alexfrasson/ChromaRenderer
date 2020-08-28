#include "chroma-renderer/gui/ChromaGUIUtils.h"

#include <ctime>
#include <iostream>

// Get current date/time, the format is YYYY-MM-DD-HH:mm:ss
std::string getDateTime()
{
    time_t now = time(nullptr);
    tm tstruct{};
    char buf[80];
    tstruct = *localtime(&now);
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d-%H-%M-%S", &tstruct); // NOLINT

    return static_cast<char*>(buf);
}

bool saveImage(const std::string& path, Image* img)
{
    if (img == nullptr)
    {
        std::cerr << "nullptr img pointer." << std::endl;
        return false;
    }
    if (img->getBuffer() == nullptr)
    {
        std::cerr << "nullptr buffer pointer." << std::endl;
        return false;
    }

    img->readDataFromOpenGLTexture();

    unsigned char bmpfileheader[14] = {'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0};
    unsigned char bmpinfoheader[40] = {40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0};

    std::uint32_t filesize = 54 + 3 * img->getWidth() * img->getHeight();
    bmpfileheader[2] = (unsigned char)(filesize);
    bmpfileheader[3] = (unsigned char)(filesize >> 8u);
    bmpfileheader[4] = (unsigned char)(filesize >> 16u);
    bmpfileheader[5] = (unsigned char)(filesize >> 24u);

    bmpinfoheader[4] = (unsigned char)(img->getWidth());
    bmpinfoheader[5] = (unsigned char)(img->getWidth() >> 8u);
    bmpinfoheader[6] = (unsigned char)(img->getWidth() >> 16u);
    bmpinfoheader[7] = (unsigned char)(img->getWidth() >> 24u);
    bmpinfoheader[8] = (unsigned char)(img->getHeight());
    bmpinfoheader[9] = (unsigned char)(img->getHeight() >> 8u);
    bmpinfoheader[10] = (unsigned char)(img->getHeight() >> 16u);
    bmpinfoheader[11] = (unsigned char)(img->getHeight() >> 24u);

    // To bgr
    std::vector<unsigned char> bgr;
    bgr.assign(img->getWidth() * img->getHeight() * 3, 0);
    for (std::uint32_t i = 0; i < img->getWidth(); i++)
    {
        for (std::uint32_t j = 0; j < img->getHeight(); j++)
        {
            bgr[(img->getWidth() * j + i) * 3 + 0] = (unsigned char)img->getBuffer()[(img->getWidth() * j + i) * 4 + 2];
            bgr[(img->getWidth() * j + i) * 3 + 1] = (unsigned char)img->getBuffer()[(img->getWidth() * j + i) * 4 + 1];
            bgr[(img->getWidth() * j + i) * 3 + 2] = (unsigned char)img->getBuffer()[(img->getWidth() * j + i) * 4 + 0];
        }
    }

    FILE* f{fopen(path.c_str(), "wb")};
    if (f == nullptr)
    {
        std::cerr << "Unable to create file " << path << std::endl;
        return false;
    }

    fwrite(static_cast<unsigned char*>(bmpfileheader), 1, 14, f);
    fwrite(static_cast<unsigned char*>(bmpinfoheader), 1, 40, f);

    unsigned char bmppad[3] = {0, 0, 0};
    for (int i = static_cast<int>(img->getHeight()) - 1; i >= 0; i--)
    {
        fwrite(bgr.data() + (img->getWidth() * (img->getHeight() - static_cast<std::uint32_t>(i) - 1) * 3),
               3,
               img->getWidth(),
               f);
        fwrite(static_cast<unsigned char*>(bmppad), 1, (4 - (img->getWidth() * 3) % 4) % 4, f);
    }

    fclose(f); // NOLINT

    return true;
}
