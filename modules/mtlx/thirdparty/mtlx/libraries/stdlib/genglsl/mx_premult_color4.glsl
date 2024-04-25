void mx_premult_color4(vec4 _in, out vec4 result)
{
    result = vec4(_in.rgb * _in.a, _in.a);
}
