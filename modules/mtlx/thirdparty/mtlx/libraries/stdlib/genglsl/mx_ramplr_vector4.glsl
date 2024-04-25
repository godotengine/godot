void mx_ramplr_vector4(vec4 valuel, vec4 valuer, vec2 texcoord, out vec4 result)
{
    result = mix (valuel, valuer, clamp(texcoord.x, 0.0, 1.0) );
}
