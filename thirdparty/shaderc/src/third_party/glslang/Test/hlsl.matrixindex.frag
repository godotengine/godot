uniform int      idx;
uniform float3x2 um;

struct PS_OUTPUT
{
    float4 Color : SV_Target0;
};

PS_OUTPUT main()
{
    // matrices of 3 rows, 2 columns (regardless of row vs col major storage)
    const float3x2 m1 = { { 10, 11 },  // row-wise initialization
                          { 12, 13 },
                          { 14, 15 } }; 

    const float3x2 m2 = { 20, 21, 22, 23, 24, 25 };  // component-wise matrix initialization is allowed
    const float3x2 m3 = { 30, 31, 33, 33, 34, 35 };  // component-wise matrix initialization is allowed

    // These can be observed in the AST post-const folding to ensure we obtain the right value,
    // as given in comments to the right of each line.  Note that the first indirection into a
    // matrix returns a row vector.
    float e1_00 = m1[0][0]; // 10
    float e1_01 = m1[0][1]; // 11
    float e1_10 = m1[1][0]; // 12
    float e1_11 = m1[1][1]; // 13
    float e1_20 = m1[2][0]; // 14
    float e1_21 = m1[2][1]; // 15

    float e2_00 = m2[0][0]; // 20
    float e2_01 = m2[0][1]; // 21
    float e2_10 = m2[1][0]; // 22
    float e2_11 = m2[1][1]; // 23
    float e2_20 = m2[2][0]; // 24
    float e2_21 = m2[2][1]; // 25

    // float e3a_00 = m3._m00; // TODO... also as an lvalue for a non-const matrix
    // float e3b_00 = m3._11;  // TODO... also as an lvalue for a non-const matrix

    float2 r0a = m1[0];  // row0: 10,11: types must match: constant index into constant
    float2 r1a = m1[1];  // row1: 12,13: ...
    float2 r2a = m1[2];  // row2: 14,15: ...

    float2 r0b = m2[idx]; // types should match: variable index into constant
    float2 r0c = um[idx]; // types should match: variable index into variable

    PS_OUTPUT psout;
    psout.Color = e2_11; // 23
    return psout;
}
