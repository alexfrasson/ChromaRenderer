#include "chroma-renderer/core/renderer/RendererSettings.h"
#include "chroma-renderer/core/utility/floating_point_equality.h"

bool RendererSettings::operator==(const RendererSettings& rs) const
{
    return width == rs.width && height == rs.height && samplesperpixel == rs.samplesperpixel &&
           almostEquals(horizontalFOV, rs.horizontalFOV);
}

bool RendererSettings::operator!=(const RendererSettings& rs) const
{
    return !((*this) == rs);
}