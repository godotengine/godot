void ShaderFunction(float inf) : COLOR0
{
    float3x4 m;

    // tests that convert to non-matrix swizzles

    m._34  = 1.0; // AST should have a normal component select
    m._m23 = 2.0; // same code
    m[2][3] = 2.0; // same code

    m._11_12_13_14 = float4(3.0);      // AST should have normal column selection (first row)
    m._m10_m11_m12_m13 = float4(3.0);  // AST should have normal column selection (second row)
    m[1] = float4(3.0);                // same code

    // tests that stay as matrix swizzles

    float3 f3;
    m._11_22_23 = f3;
    m._21_12_31 = float3(5.0);
    m._11_12_21 = 2 * f3;

    // r-value
    f3 = m._21_12_31;
}

float3x3 createMat3x3(float3 a, float3 b, float3 c)
{
    float3x3 m;
    m._11_21_31 = a;
    m._12_22_32 = b;
    m._13_23_33 = c;
    return m;
}
