
void Fn1(float4x4 p) { }

float4 main() : SV_TARGET
{
    const float4x4 mat1c = 0.20;
    const float4x4 mat2c = {2, 2.1, 2.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const float4x4 mat3c = (float4x4)float1(0.1);

    float4x4 mat1 = 0.25;
    float4x4 mat2 = {3, 3.1, 3.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    float4x4 mat3 = (float4x4)0.375;
    // float4x4 mat5 = (float4x4)Fn2(); // TODO: enable when compex rvalue handling is in place

    float4x4 mat4;
    mat4 = 0.75;
    mat4 = float4x4(4, 4.1, 4.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    mat4 = (float4x4)0.5;

    mat4 *= 0.75;
    mat4 += 0.75;
    mat4 -= 0.5;
    mat4 /= 2.0;

    Fn1(5.0); // test calling fn accepting matrix with scalar type

    return mat1c[0] + mat3c[0] + mat1[1] + mat4[2];
}
