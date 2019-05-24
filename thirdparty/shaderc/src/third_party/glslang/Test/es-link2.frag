#version 100

varying mediump vec4 varyingColor;

mediump vec4 calculateColor()
{
    return varyingColor * 0.5;
}
