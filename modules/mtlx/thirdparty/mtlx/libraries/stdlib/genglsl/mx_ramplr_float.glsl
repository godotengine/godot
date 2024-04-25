void mx_ramplr_float(float valuel, float valuer, vec2 texcoord, out float result)
{
    result = mix (valuel, valuer, clamp(texcoord.x, 0.0, 1.0) );
}
