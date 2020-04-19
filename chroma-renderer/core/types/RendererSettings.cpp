#include "chroma-renderer/core/types/RendererSettings.h"

#include <algorithm>
#include <thread>

RendererSettings::RendererSettings()
    : width(640), height(480), horizontalFOV(1.0), enviromentLightColor(0.8, 0.8, 0.8), enviromentLightIntensity(1), nthreads(2),
      supersampling(true), samplesperpixel(1000), boundingboxtest(true), shadowray(false)
{
    nthreads = std::max(std::thread::hardware_concurrency(), 2u);
}

bool RendererSettings::operator==(const RendererSettings& rs) const
{
    return width == rs.width && height == rs.height && nthreads == rs.nthreads && supersampling == rs.supersampling &&
           samplesperpixel == rs.samplesperpixel && boundingboxtest == rs.boundingboxtest &&
           shadowray == rs.shadowray && horizontalFOV == rs.horizontalFOV &&
           enviromentLightColor == rs.enviromentLightColor && enviromentLightIntensity == rs.enviromentLightIntensity;
}

bool RendererSettings::operator!=(const RendererSettings& rs) const
{
    return !((*this) == rs);
}