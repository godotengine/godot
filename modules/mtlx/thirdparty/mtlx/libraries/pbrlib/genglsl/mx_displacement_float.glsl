void mx_displacement_float(float disp, float scale, out displacementshader result)
{
    result.offset = vec3(disp);
    result.scale = scale;
}
