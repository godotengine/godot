void mx_ramptb_float(float valuet, float valueb, vec2 texcoord, out float result)
{
    result = mix (valuet, valueb, clamp(texcoord.y, 0.0, 1.0) );
}
