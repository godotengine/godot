void mx_ramplr_vector2(vec2 valuel, vec2 valuer, vec2 texcoord, out vec2 result)
{
    result = mix (valuel, valuer, clamp(texcoord.x, 0.0, 1.0) );
}
