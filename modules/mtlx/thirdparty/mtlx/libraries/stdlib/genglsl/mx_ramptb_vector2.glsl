void mx_ramptb_vector2(vec2 valuet, vec2 valueb, vec2 texcoord, out vec2 result)
{
    result = mix (valuet, valueb, clamp(texcoord.y, 0.0, 1.0) );
}
