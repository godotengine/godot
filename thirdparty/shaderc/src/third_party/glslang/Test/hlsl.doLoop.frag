float4 PixelShaderFunction(float input) : COLOR0
{
    [unroll] do {} while (false);
    [unroll] do {;} while (false);
    do { return (float4)input; } while (input > 2.0);
    do ++input; while (input < 10.0);
    do while (++input < 10.0); while (++input < 10.0); // nest while inside do-while
    return (float4)input;
}
