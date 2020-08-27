#include "chroma-renderer/core/renderer/RendererSettings.h"

bool RendererSettings::operator==(const RendererSettings& rs) const
{
    return width == rs.width && height == rs.height && samplesperpixel == rs.samplesperpixel &&
           horizontalFOV == rs.horizontalFOV;
}

bool RendererSettings::operator!=(const RendererSettings& rs) const
{
    return !((*this) == rs);
}