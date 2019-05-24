float4 PixelShaderFunction(float4 input) : COLOR0
{
    while (any(input != input)) { return input; }
    while (false) ;
    [unroll] while (false) { }
    while ((false)) { }
}
