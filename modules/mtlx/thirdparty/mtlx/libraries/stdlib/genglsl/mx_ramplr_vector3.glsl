void mx_ramplr_vector3(vec3 valuel, vec3 valuer, vec2 texcoord, out vec3 result)
{
    result = mix (valuel, valuer, clamp(texcoord.x, 0.0, 1.0) );
}
