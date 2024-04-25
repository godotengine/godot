void mx_mix_surfaceshader(surfaceshader fg, surfaceshader bg, float w, out surfaceshader returnshader)
{
    returnshader.color = mix(bg.color, fg.color, w);
    returnshader.transparency = mix(bg.transparency, fg.transparency, w);
}
