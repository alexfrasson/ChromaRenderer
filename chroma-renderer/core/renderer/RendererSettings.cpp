#include "chroma-renderer/core/renderer/RendererSettings.h"

#include <algorithm>
#include <thread>

RendererSettings::RendererSettings() : width(640), height(480), horizontalFOV(1.0), samplesperpixel(10000)
{
}

bool RendererSettings::operator==(const RendererSettings& rs) const
{
    return width == rs.width && height == rs.height && samplesperpixel == rs.samplesperpixel &&
           horizontalFOV == rs.horizontalFOV;
}

bool RendererSettings::operator!=(const RendererSettings& rs) const
{
    return !((*this) == rs);
}