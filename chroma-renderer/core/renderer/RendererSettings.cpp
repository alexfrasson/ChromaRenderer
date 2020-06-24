#include "chroma-renderer/core/renderer/RendererSettings.h"

#include <algorithm>
#include <thread>

RendererSettings::RendererSettings()
    : width(640), height(480), horizontalFOV(1.0), nthreads(2), supersampling(true), samplesperpixel(10000),
      boundingboxtest(true), shadowray(false)
{
    nthreads = std::max(std::thread::hardware_concurrency(), 2u);
}

bool RendererSettings::operator==(const RendererSettings& rs) const
{
    return width == rs.width && height == rs.height && nthreads == rs.nthreads && supersampling == rs.supersampling &&
           samplesperpixel == rs.samplesperpixel && boundingboxtest == rs.boundingboxtest &&
           shadowray == rs.shadowray && horizontalFOV == rs.horizontalFOV;
}

bool RendererSettings::operator!=(const RendererSettings& rs) const
{
    return !((*this) == rs);
}