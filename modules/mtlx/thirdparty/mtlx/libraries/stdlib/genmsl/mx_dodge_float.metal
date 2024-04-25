void mx_dodge_float(float fg, float bg, float mixval, out float result)
{
    if (abs(1.0 - fg) < M_FLOAT_EPS)
    {
        result = 0.0;
        return;
    }
    result = mixval*(bg / (1.0 - fg)) + ((1.0-mixval)*bg);
}
