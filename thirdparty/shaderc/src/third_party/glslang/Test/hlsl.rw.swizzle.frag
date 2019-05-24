RWTexture2D<float3> rwtx;
RWBuffer<float3> buf;

float3 SomeValue() { return float3(1,2,3); }

float4 main() : SV_Target0
{
    int2 tc2 = { 0, 0 };
    int tc = 0;

    // Test swizzles and partial updates of L-values when writing to buffers and writable textures.
    rwtx[tc2].zyx = float3(1,2,3);     // full swizzle, simple RHS
    rwtx[tc2].zyx = SomeValue();       // full swizzle, complex RHS
    rwtx[tc2].zyx = 2;                 // full swizzle, modify op

    // Partial updates not yet supported.
    // Partial values, which will use swizzles.
    // buf[tc].yz = 42;                 // partial swizzle, simple RHS
    // buf[tc].yz = SomeValue().x;      // partial swizzle, complex RHS
    // buf[tc].yz += 43;                // partial swizzle, modify op

    // // Partial values, which will use index.
    // buf[tc].y = 44;                  // single index, simple RHS
    // buf[tc].y = SomeValue().x;       // single index, complex RHS
    // buf[tc].y += 45;                 // single index, modify op

    return 0.0;
}
