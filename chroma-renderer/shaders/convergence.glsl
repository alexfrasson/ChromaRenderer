#version 450 core

#extension GL_NV_gpu_shader5 : enable

layout(local_size_x = 16, local_size_y = 16) in;

// layout(binding = 0) uniform atomic_uint count;

// layout(rgba32f) uniform image2D lastRenderedBuffer;
layout(rgba32f) uniform image2D srcImage;
layout(rgba32f) uniform image2D dstImage;

// uniform bool readFromFboTex0;
uniform float enviromentLightIntensity;

//
// Neutral tonemapping (Hable/Hejl/Frostbite)
// Input is linear RGB
//
vec3 NeutralCurve(vec3 x, float a, float b, float c, float d, float e, float f)
{
    return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f;
}

vec3 NeutralTonemap(vec3 x)
{
    // Tonemap
    float a = 0.2;
    float b = 0.29;
    float c = 0.24;
    float d = 0.272;
    float e = 0.02;
    float f = 0.3;
    float whiteLevel = 5.3;
    float whiteClip = 1.0;

    vec3 whiteScale = vec3(1.0) / NeutralCurve(vec3(whiteLevel), a, b, c, d, e, f);
    x = NeutralCurve(x * whiteScale, a, b, c, d, e, f);
    x *= whiteScale;

    // Post-curve white point adjustment
    x /= whiteClip.xxx;

    return x;
}

void main()
{
    ivec2 size = imageSize(srcImage);

    if (gl_GlobalInvocationID.x >= size.x || gl_GlobalInvocationID.y >= size.y)
        return;

    ivec2 texcoord = ivec2(gl_GlobalInvocationID.xy);

    vec4 snap = imageLoad(srcImage, texcoord);
    vec3 color = (snap.xyz / snap.w) * enviromentLightIntensity;

    // color = pow(color.xyz, vec3(1.0 / 2.2));

    color = max(color, 0.0);

    color = NeutralTonemap(color);

    imageStore(dstImage, texcoord, vec4(color, 1.0));

    // uint32_t inCount;
    // uint32_t sampleCount;
    // bool converged;

    ////if (!unpackData(snap.a, inCount, sampleCount, converged))
    ////	return;

    // bool snapValid = unpackData(snap.a, inCount, sampleCount, converged);

    // vec3 snapAvgColor = snap.rgb / sampleCount;

    // vec4 c = imageLoad(lastRenderedBuffer, texcoord);

    // if (!unpackData(c.a, inCount, sampleCount, converged) && !snapValid)
    //	return;
    //
    // if (snapValid)
    //{
    //	vec3 currAvgColor = c.rgb / sampleCount;

    //	float diff = length(currAvgColor - snapAvgColor);
    //	converged = diff < convergenceThreshold;

    //	#ifdef SHOW_HEATMAP
    //	if (!converged)
    //		atomicCounterIncrement(count);
    //	#endif
    //}

    // vec4 outValue;
    // outValue.a = packData(inCount, sampleCount, converged);
    // outValue.rgb = c.rgb;

    // imageStore(imgSnapshot, texcoord, outValue);
    // imageStore(lastRenderedBuffer, texcoord, outValue);
}