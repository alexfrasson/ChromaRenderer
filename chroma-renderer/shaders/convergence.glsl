#version 450 core

#extension GL_NV_gpu_shader5 : enable

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f) uniform image2D srcImage;
layout(rgba32f) uniform image2D dstImage;

uniform float apperture;
uniform float shutterTime;
uniform float iso;
uniform bool tonemapping;
uniform bool linearToSrbg;
uniform bool adjustExposure;

// Neutral tonemapping (Hable/Hejl/Frostbite)
// https://64.github.io/tonemapping/
// http://filmicworlds.com/blog/filmic-tonemapping-operators/
// http://filmicworlds.com/blog/minimal-color-grading-tools/
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

// https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
float computeEV100(float aperture, float shutterTime, float ISO)
{
    // EV number is defined as:
    // 2^ EV_s = N^2 / t and EV_s = EV_100 + log2 (S /100)
    // This gives
    // EV_s = log2 (N^2 / t)
    // EV_100 + log2 (S /100) = log2 (N^2 / t)
    // EV_100 = log2 (N^2 / t) - log2 (S /100)
    // EV_100 = log2 (N^2 / t . 100 / S)
    return log2(sqrt(aperture) / shutterTime * 100.0 / ISO);
}

float computeEV100FromAvgLuminance(float avgLuminance)
{
    // We later use the middle gray at 12.7% in order to have
    // a middle gray at 18% with a sqrt (2) room for specular highlights
    // But here we deal with the spot meter measuring the middle gray
    // which is fixed at 12.5 for matching standard camera
    // constructor settings (i.e. calibration constant K = 12.5)
    // Reference : http :// en. wikipedia . org / wiki / Film_speed
    return log2(avgLuminance * 100.0 / 12.5);
}

float convertEV100ToExposure(float EV100)
{
    // Compute the maximum luminance possible with H_sbs sensitivity
    // maxLum = 78 / ( S * q ) * N^2 / t
    // = 78 / ( S * q ) * 2^ EV_100
    // = 78 / (100 * 0.65) * 2^ EV_100
    // = 1.2 * 2^ EV
    // Reference : http :// en. wikipedia . org / wiki / Film_speed
    float maxLuminance = 1.2 * pow(2.0, EV100);
    return 1.0 / maxLuminance;
}

vec3 accurateLinearToSRGB(vec3 linearCol)
{
    vec3 sRGBLo = linearCol * 12.92;
    vec3 sRGBHi = (pow(abs(linearCol), vec3(1.0 / 2.4)) * 1.055) - 0.055;
    // hlsl: float3 sRGB = ( linearCol <= 0.0031308) ? sRGBLo : sRGBHi ;
    vec3 sRGB = mix(sRGBHi, sRGBLo, lessThanEqual(linearCol, vec3(0.0031308)));
    return sRGB;
}

void main()
{
    const ivec2 size = imageSize(srcImage);
    if (gl_GlobalInvocationID.x >= size.x || gl_GlobalInvocationID.y >= size.y)
    {
        return;
    }

    const ivec2 texcoord = ivec2(gl_GlobalInvocationID.xy);

    const vec4 snap = imageLoad(srcImage, texcoord);
    vec3 color = snap.xyz;

    if (adjustExposure)
    {
        const float EV100 = computeEV100(apperture, shutterTime, iso);
        const float exposure = convertEV100ToExposure(EV100);
        color *= exposure;
    }

    if (tonemapping)
    {
        color = NeutralTonemap(color);
    }

    if (linearToSrbg)
    {
        color = accurateLinearToSRGB(color);
    }

    imageStore(dstImage, texcoord, vec4(color, 1.0));
}