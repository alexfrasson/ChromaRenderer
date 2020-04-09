#include <ChromaGUIUtils.h>

#include <ctime>
#include <iostream>

// Get current date/time, the format is YYYY-MM-DD-HH:mm:ss
const std::string getDateTime()
{
    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d-%H-%M-%S", &tstruct);

    return buf;
}

bool saveImage(std::string path, Image* img)
{
    if (img == NULL)
    {
        std::cerr << "NULL img pointer." << std::endl;
        return false;
    }
    if (img->getBuffer() == NULL)
    {
        std::cerr << "NULL buffer pointer." << std::endl;
        return false;
    }

    img->readDataFromOpenGLTexture();

    unsigned char bmpfileheader[14] = {'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0};
    unsigned char bmpinfoheader[40] = {40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0};
    unsigned char bmppad[3] = {0, 0, 0};
    int filesize = 54 + 3 * img->getWidth() * img->getHeight();

    bmpfileheader[2] = (unsigned char)(filesize);
    bmpfileheader[3] = (unsigned char)(filesize >> 8);
    bmpfileheader[4] = (unsigned char)(filesize >> 16);
    bmpfileheader[5] = (unsigned char)(filesize >> 24);

    bmpinfoheader[4] = (unsigned char)(img->getWidth());
    bmpinfoheader[5] = (unsigned char)(img->getWidth() >> 8);
    bmpinfoheader[6] = (unsigned char)(img->getWidth() >> 16);
    bmpinfoheader[7] = (unsigned char)(img->getWidth() >> 24);
    bmpinfoheader[8] = (unsigned char)(img->getHeight());
    bmpinfoheader[9] = (unsigned char)(img->getHeight() >> 8);
    bmpinfoheader[10] = (unsigned char)(img->getHeight() >> 16);
    bmpinfoheader[11] = (unsigned char)(img->getHeight() >> 24);

    FILE* f = NULL;
    // std::string name("C:\\rendered-");
    // name.append(dateTime());
    // name.append(".bmp");
    f = fopen(path.c_str(), "wb");
    if (f == NULL)
    {
        std::cerr << "Unable to create file " << path << std::endl;
        return false;
    }
    fwrite(bmpfileheader, 1, 14, f);
    fwrite(bmpinfoheader, 1, 40, f);
    // for (int i = 0; i < img->height; i++)

    // To bgr
    unsigned char* bgr = new unsigned char[img->getWidth() * img->getHeight() * 3];

    for (int i = 0; i < img->getWidth(); i++)
    {
        for (int j = 0; j < img->getHeight(); j++)
        {
            bgr[(img->getWidth() * j + i) * 3 + 0] = (unsigned char)img->getBuffer()[(img->getWidth() * j + i) * 4 + 2];
            bgr[(img->getWidth() * j + i) * 3 + 1] = (unsigned char)img->getBuffer()[(img->getWidth() * j + i) * 4 + 1];
            bgr[(img->getWidth() * j + i) * 3 + 2] = (unsigned char)img->getBuffer()[(img->getWidth() * j + i) * 4 + 0];
        }
    }

    for (int i = img->getHeight() - 1; i >= 0; i--)
    {
        // fwrite(img->buffer + (img->width*(img->height - i - 1) * 3), 3, img->width, f);
        fwrite(bgr + (img->getWidth() * (img->getHeight() - i - 1) * 3), 3, img->getWidth(), f);
        fwrite(bmppad, 1, (4 - (img->getWidth() * 3) % 4) % 4, f);
    }
    fclose(f);

    delete[] bgr;

    return true;
}
