void mx_luminance_color4(vec4 _in, vec3 lumacoeffs, out vec4 result)
{
    result = vec4(vec3(dot(_in.rgb, lumacoeffs)), _in.a);
}
