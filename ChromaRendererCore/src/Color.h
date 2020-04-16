#pragma once

class Color
{
  public:
    float r, g, b;

    Color() : r(0.5f), g(0.5f), b(0.5f)
    {
    }

    Color(float r, float g, float b) : r(r), g(g), b(b)
    {
    }

    Color(float c) : r(c), g(c), b(c)
    {
    }

    Color operator*(float f) const
    {
        return Color(r * f, g * f, b * f);
    }

    void operator+=(const Color& rhs)
    {
        r += rhs.r;
        g += rhs.g;
        b += rhs.b;
    }

    Color operator+(const Color& rhs) const
    {
        return Color(r + rhs.r, g + rhs.g, b + rhs.b);
    }

    Color operator*(const Color& rhs) const
    {
        return Color(r * rhs.r, g * rhs.g, b * rhs.b);
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

    Color operator/(float& f) const
    {
        return Color(r / f, g / f, b / f);
    }

    Color operator/(int f) const
    {
        return Color(r / f, g / f, b / f);
    }

    static const Color BLACK;
    static const Color BLUE;
    static const Color GREEN;
    static const Color RED;
    static const Color WHITE;
    static const Color GRAY;
};