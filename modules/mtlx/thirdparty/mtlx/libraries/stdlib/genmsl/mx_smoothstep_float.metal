void mx_smoothstep_float(float val, float low, float high, out float result)
{
    if (val <= low)
        result = 0.0;
    else if (val >= high)
        result = 1.0;
    else
        result = smoothstep(low, high, val);
}
