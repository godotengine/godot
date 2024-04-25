void mx_burn_float(float fg, float bg, float mixval, out float result)
{
    if (abs(fg) < M_FLOAT_EPS)
    {
        result = 0.0;
        return;
    }
    result = mixval*(1.0 - ((1.0 - bg) / fg)) + ((1.0-mixval)*bg);
}
