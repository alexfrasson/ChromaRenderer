#include <RendererSettings.h>

#include <algorithm>
#include <thread>

RendererSettings::RendererSettings()
    : width(640), height(480), nthreads(2), enviromentLightColor(0.8, 0.8, 0.8), enviromentLightIntensity(1),
      supersampling(true), samplesperpixel(1000), boundingboxtest(true), shadowray(false), horizontalFOV(1.0)
{
    nthreads = std::max(std::thread::hardware_concurrency(), 2u);
}

const bool RendererSettings::operator==(const RendererSettings& rs)
{
    return width == rs.width && height == rs.height && nthreads == rs.nthreads && supersampling == rs.supersampling &&
           samplesperpixel == rs.samplesperpixel && boundingboxtest == rs.boundingboxtest &&
           shadowray == rs.shadowray && horizontalFOV == rs.horizontalFOV &&
           enviromentLightColor == rs.enviromentLightColor && enviromentLightIntensity == rs.enviromentLightIntensity;
}

const bool RendererSettings::operator!=(const RendererSettings& rs)
{
    return !((*this) == rs);
}