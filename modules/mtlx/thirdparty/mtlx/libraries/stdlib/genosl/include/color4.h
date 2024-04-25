// Open Shading Language : Copyright (c) 2009-2017 Sony Pictures Imageworks Inc., et al.
// https://github.com/imageworks/OpenShadingLanguage/blob/master/LICENSE

#pragma once
#define COLOR4_H


// color4 is a color + alpha
struct color4
{
    color rgb;
    float a;
};



//
// For color4, define math operators to match color
//

color4 __operator__neg__(color4 a)
{
    return color4(-a.rgb, -a.a);
}

color4 __operator__add__(color4 a, color4 b)
{
    return color4(a.rgb + b.rgb, a.a + b.a);
}

color4 __operator__add__(color4 a, int b)
{
    return a + color4(color(b), b);
}

color4 __operator__add__(color4 a, float b)
{
    return a + color4(color(b), b);
}

color4 __operator__add__(int a, color4 b)
{
    return color4(color(a), a) + b;
}

color4 __operator__add__(float a, color4 b)
{
    return color4(color(a), a) + b;
}

color4 __operator__sub__(color4 a, color4 b)
{
    return color4(a.rgb - b.rgb, a.a - b.a);
}

color4 __operator__sub__(color4 a, int b)
{
    return a - color4(color(b), b);
}

color4 __operator__sub__(color4 a, float b)
{
    return a - color4(color(b), b);
}

color4 __operator__sub__(int a, color4 b)
{
    return color4(color(a), a) - b;
}

color4 __operator__sub__(float a, color4 b)
{
    return color4(color(a), a) - b;
}

color4 __operator__mul__(color4 a, color4 b)
{
    return color4(a.rgb * b.rgb, a.a * b.a);
}

color4 __operator__mul__(color4 a, int b)
{
    return a * color4(color(b), b);
}

color4 __operator__mul__(color4 a, float b)
{
    return a * color4(color(b), b);
}

color4 __operator__mul__(int a, color4 b)
{
    return color4(color(a), a) * b;
}

color4 __operator__mul__(float a, color4 b)
{
    return color4(color(a), a) * b;
}

color4 __operator__div__(color4 a, color4 b)
{
    return color4(a.rgb / b.rgb, a.a / b.a);
}

color4 __operator__div__(color4 a, int b)
{
    float b_inv = 1.0/b;
    return a * color4(color(b_inv), b_inv);
}

color4 __operator__div__(color4 a, float b)
{
    float b_inv = 1.0/b;
    return a * color4(color(b_inv), b_inv);
}

color4 __operator_div__(int a, color4 b)
{
    return color4(color(a), a) / b;
}

color4 __operator__div__(float a, color4 b)
{
    return color4(color(a), a) / b;
}

int __operator__eq__(color4 a, color4 b)
{
    return (a.rgb == b.rgb) && (a.a == b.a);
}

int __operator__ne__(color4 a, color4 b)
{
    return (a.rgb != b.rgb) || (a.a != b.a);
}



//
// For color4, define most of the stdosl functions to match color
//

color4 abs(color4 a)
{
    return color4(abs(a.rgb), abs(a.a));
}

color4 ceil(color4 a)
{
    return color4(ceil(a.rgb), ceil(a.a));
}

color4 round(color4 a)
{
    return color4(round(a.rgb), round(a.a));
}

color4 floor(color4 a)
{
    return color4(floor(a.rgb), floor(a.a));
}

color4 sqrt(color4 a)
{
    return color4(sqrt(a.rgb), sqrt(a.a));
}

color4 exp(color4 a)
{
    return color4(exp(a.rgb), exp(a.a));
}

color4 log(color4 a)
{
    return color4(log(a.rgb), log(a.a));
}

color4 log2(color4 a)
{
    return color4(log2(a.rgb), log2(a.a));
}

color4 mix(color4 a, color4 b, float x )
{
    return color4(mix(a.rgb, b.rgb, x),
                  mix(a.a, b.a, x));
}

color4 mix(color4 a, color4 b, color4 x )
{
    return color4(mix(a.rgb, b.rgb, x.rgb),
                  mix(a.a, b.a, x.a));
}

float dot(color4 a, color b)
{
    return dot(a.rgb, b);
}

color4 smoothstep(color4 edge0, color4 edge1, color4 c)
{
    return color4(smoothstep(edge0.rgb, edge1.rgb, c.rgb),
                  smoothstep(edge0.a, edge1.a, c.a));
}

color4 smoothstep(float edge0, float edge1, color4 c)
{
    return smoothstep(color4(color(edge0), edge0), color4(color(edge1), edge1), c);
}

color4 clamp(color4 c, color4 minval, color4 maxval)
{
    return color4(clamp(c.rgb, minval.rgb, maxval.rgb),
                  clamp(c.a, minval.a, maxval.a));
}

color4 clamp(color4 c, float minval, float maxval)
{
    return clamp(c, color4(color(minval), minval), color4(color(maxval), maxval));
}

color4 max(color4 a, color4 b)
{
    return color4(max(a.rgb, b.rgb),
                  max(a.a, b.a));
}

color4 max(color4 a, float b)
{
    return color4(max(a.rgb, b),
                  max(a.a, b));
}

color4 min(color4 a, color4 b)
{
    return color4(min(a.rgb, b.rgb),
                  min(a.a, b.a));
}

color4 min(color4 a, float b)
{
    return color4(min(a.rgb, b),
                  min(a.a, b));
}

color4 mod(color4 a, color4 b)
{
    return color4(mod(a.rgb, b.rgb),
                  mod(a.a, b.a));
}

color4 mod(color4 a, int b)
{
    return mod(a, color4(color(b), b));
}

color4 mod(color4 a, float b)
{
    return mod(a, color4(color(b), b));
}

color4 fmod(color4 a, color4 b)
{
    return color4(fmod(a.rgb, b.rgb),
                  fmod(a.a, b.a));
}

color4 fmod(color4 a, int b)
{
    return fmod(a, color4(color(b), b));
}

color4 fmod(color4 a, float b)
{
    return fmod(a, color4(color(b), b));
}

color4 pow(color4 base, color4 power)
{
    return color4(pow(base.rgb, power.rgb),
                  pow(base.a, power.a));
}

color4 pow(color4 base, float power)
{
    return color4(pow(base.rgb, power),
                  pow(base.a, power));
}

color4 sign(color4 a)
{
    return color4(sign(a.rgb),
                  sign(a.a));
}

color4 sin(color4 a)
{
    return color4(sin(a.rgb),
                  sin(a.a));
}

color4 cos(color4 a)
{
    return color4(cos(a.rgb),
                  cos(a.a));
}

color4 tan(color4 a)
{
    return color4(tan(a.rgb),
                  tan(a.a));
}

color4 asin(color4 a)
{
    return color4(asin(a.rgb),
                  asin(a.a));
}

color4 acos(color4 a)
{
    return color4(acos(a.rgb),
                  acos(a.a));
}

color4 atan2(color4 a, float f)
{
    return color4(atan2(a.rgb, f),
                  atan2(a.a, f));
}

color4 atan2(color4 a, color4 b)
{
    return color4(atan2(a.rgb, b.rgb),
                  atan2(a.a, b.a));
}


color4 transformc (string fromspace, string tospace, color4 C)
{
    return color4 (transformc (fromspace, tospace, C.rgb), C.a);
}
