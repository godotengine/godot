int PixelShaderFunction(
    int a1,
    int a2,
    int a3,
    int a4
    ) : COLOR0
{
    return (a1 * a2 + a3 << a4) + (a1 << a2 + a3 * a4);
}
