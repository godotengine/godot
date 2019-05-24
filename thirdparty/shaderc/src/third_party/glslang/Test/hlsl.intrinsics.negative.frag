float PixelShaderFunctionS(float inF0, float inF1, float inF2, int inI0)
{
    // AllMemoryBarrier();              // TODO: expected error: invalid in fragment stage
    // AllMemoryBarrierWithGroupSync(); // TODO: expected error: invalid in fragment stage
    asdouble(inF0, inF1);                     // expected error: only integer inputs
    CheckAccessFullyMapped(3.0);              // expected error: only valid on integers
    countbits(inF0);                          // expected error: only integer inputs
    cross(inF0, inF1);                        // expected error: only on float3 inputs
    D3DCOLORtoUBYTE4(inF0);                   // expected error: only on float4 inputs
    determinant(inF0);                        // expected error: only valid on mats
    // DeviceMemoryBarrierWithGroupSync();      // TODO: expected error: only valid in compute stage
    f16tof32(inF0);                           // expected error: only integer inputs
    firstbithigh(inF0);                       // expected error: only integer inputs
    firstbitlow(inF0);                        // expected error: only integer inputs
    // fma(inF0, inF1, inF2); // TODO: this might auto-promote: need to check against FXC
    // InterlockedAdd(inI0, inI0, 3);            // expected error: last parameter is out TODO: accepted even though marked as out in proto generator
    // InterlockedAnd(inI0, inI0, 3);            // expected error: last parameter is out TODO: accepted even though marked as out i    // InterlockedMax(inI0, inI0, 3);            // expected error: last parameter is out TODO: accepted even though marked as out in proto generator
    // InterlockedMin(inI0, inI0, 3);            // expected error: last parameter is out TODO: accepted even though marked as out in proto generator
    // InterlockedOor(inI0, inI0, 3);            // expected error: last parameter is out TODO: accepted even though marked as out in proto generator
    // InterlockedXor(inI0, inI0, 3);            // expected error: last parameter is out TODO: accepted even though marked as out in proto generator
    // GroupMemoryBarrier();               // TODO: expected error: invalid in fragment stage
    // GroupMemoryBarrierWithGroupSync();  // TODO: expected error: invalid in fragment stage
    length(inF0);                             // expected error: invalid on scalars
    msad4(inF0, float2(0), float4(0));        // expected error: only integer inputs
    normalize(inF0);                          // expected error: invalid on scalars
    reflect(inF0, inF1);                      // expected error: invalid on scalars
    refract(inF0, inF1, inF2);                // expected error: invalid on scalars
    refract(float2(0), float2(0), float2(0)); // expected error: last parameter only scalar
    reversebits(inF0);                        // expected error: only integer inputs
    transpose(inF0);                          // expected error: only valid on mats

    return 0.0;
}

float1 PixelShaderFunction1(float1 inF0, float1 inF1, float1 inF2, int1 inI0)
{
    // TODO: ... add when float1 prototypes are generated

    GetRenderTargetSamplePosition(inF0); // expected error: only integer inputs

    return 0.0;
}

float2 PixelShaderFunction2(float2 inF0, float2 inF1, float2 inF2, int2 inI0)
{
    asdouble(inF0, inF1);         // expected error: only integer inputs
    CheckAccessFullyMapped(inF0); // expected error: only valid on scalars
    countbits(inF0);              // expected error: only integer inputs
    cross(inF0, inF1);            // expected error: only on float3 inputs
    D3DCOLORtoUBYTE4(inF0);       // expected error: only on float4 inputs
    determinant(inF0);            // expected error: only valid on mats
    f16tof32(inF0);               // expected error: only integer inputs
    firstbithigh(inF0);           // expected error: only integer inputs
    firstbitlow(inF0);            // expected error: only integer inputs
    // fma(inF0, inF1, inF2); // TODO: this might auto-promote: need to check against FXC
    reversebits(inF0);            // expected error: only integer inputs
    transpose(inF0);              // expected error: only valid on mats

    return float2(1,2);
}

float3 PixelShaderFunction3(float3 inF0, float3 inF1, float3 inF2, int3 inI0)
{
    CheckAccessFullyMapped(inF0);  // expected error: only valid on scalars
    countbits(inF0);            // expected error: only integer inputs
    D3DCOLORtoUBYTE4(inF0);     // expected error: only on float4 inputs
    determinant(inF0);          // expected error: only valid on mats
    f16tof32(inF0);             // expected error: only integer inputs
    firstbithigh(inF0);         // expected error: only integer inputs
    firstbitlow(inF0);          // expected error: only integer inputs
    // fma(inF0, inF1, inF2); // TODO: this might auto-promote: need to check against FXC
    reversebits(inF0);          // expected error: only integer inputs
    transpose(inF0);            // expected error: only valid on mats


    return float3(1,2,3);
}

float4 PixelShaderFunction(float4 inF0, float4 inF1, float4 inF2, int4 inI0)
{
    CheckAccessFullyMapped(inF0); // expected error: only valid on scalars
    countbits(inF0);              // expected error: only integer inputs
    cross(inF0, inF1);            // expected error: only on float3 inputs
    determinant(inF0);            // expected error: only valid on mats
    f16tof32(inF0);               // expected error: only integer inputs
    firstbithigh(inF0);           // expected error: only integer inputs
    firstbitlow(inF0);            // expected error: only integer inputs
    // fma(inF0, inF1, inF2); // TODO: this might auto-promote: need to check against FXC
    reversebits(inF0);            // expected error: only integer inputs
    transpose(inF0);              // expected error: only valid on mats

    return float4(1,2,3,4);
}

// TODO: FXC doesn't accept this with (), but glslang doesn't accept it without.
#define MATFNS() \
    countbits(inF0);          \
    D3DCOLORtoUBYTE4(inF0);   \
    cross(inF0, inF1);        \
    f16tof32(inF0);           \
    firstbithigh(inF0);       \
    firstbitlow(inF0);        \
    reversebits(inF0);        \
    length(inF0);             \
    noise(inF0);              \
    normalize(inF0);          \
    reflect(inF0, inF1);      \
    refract(inF0, inF1, 1.0); \
    reversebits(inF0);        \
    

// TODO: turn on non-square matrix tests when protos are available.

float2x2 PixelShaderFunction2x2(float2x2 inF0, float2x2 inF1, float2x2 inF2)
{
    // TODO: FXC doesn't accept this with (), but glslang doesn't accept it without.
    MATFNS()

    return float2x2(2,2,2,2);
}

float3x3 PixelShaderFunction3x3(float3x3 inF0, float3x3 inF1, float3x3 inF2)
{
    // TODO: FXC doesn't accept this with (), but glslang doesn't accept it without.
    MATFNS()

    return float3x3(3,3,3,3,3,3,3,3,3);
}

float4x4 PixelShaderFunction4x4(float4x4 inF0, float4x4 inF1, float4x4 inF2)
{
    // TODO: FXC doesn't accept this with (), but glslang doesn't accept it without.
    MATFNS()

    return float4x4(4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4);
}
