#pragma once

class Color
{
  public:
    Color() = default;

    Color(float red, float green, float blue) : r(red), g(green), b(blue)
    {
    }

    explicit Color(float c) : r(c), g(c), b(c)
    {
    }

    Color operator*(float f) const
    {
        return {r * f, g * f, b * f};
    }

    Color operator+(const Color& rhs) const
    {
        return {r + rhs.r, g + rhs.g, b + rhs.b};
    }

    Color operator*(const Color& rhs) const
    {
        return {r * rhs.r, g * rhs.g, b * rhs.b};
    }

    Color operator/(const float& f) const
    {
        return {r / f, g / f, b / f};
    }

    void operator+=(const Color& rhs)
    {
        r += rhs.r;
        g += rhs.g;
        b += rhs.b;
    }

    void operator+=(float rhs)
    {
        r += rhs;
        g += rhs;
        b += rhs;
    }

    void operator/=(float rhs)
    {
        r /= rhs;
        g /= rhs;
        b /= rhs;
    }

    static const Color kBlack;
    static const Color kBlue;
    static const Color kGreen;
    static const Color kRed;
    static const Color kWhite;
    static const Color kGray;

    float r{1.0f}, g{1.0f}, b{1.0f};
};