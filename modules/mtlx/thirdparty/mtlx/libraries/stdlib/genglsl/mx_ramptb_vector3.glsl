void mx_ramptb_vector3(vec3 valuet, vec3 valueb, vec2 texcoord, out vec3 result)
{
    result = mix (valuet, valueb, clamp(texcoord.y, 0.0, 1.0) );
}
