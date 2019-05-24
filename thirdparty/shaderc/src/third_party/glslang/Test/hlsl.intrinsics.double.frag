
float PixelShaderFunction(double inDV1a, double inDV1b, double inDV1c,
                          double2 inDV2, double3 inDV3, double4 inDV4,
                          uint inU1a, uint inU1b)
{
    double  r00 = fma(inDV1a, inDV1b, inDV1c);
    double  r01 = asdouble(inU1a, inU1b);

    return 0.0;
}

