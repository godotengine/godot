#pragma pack_matrix(row_major)

struct MyBuffer1
{
    column_major float4x4 mat1;
    row_major    float4x4 mat2;
    /*floating*/ float4x4 mat3;
};

#pragma pack_matrix(column_major)

struct MyBuffer2
{
    column_major float4x4 mat1;
    row_major    float4x4 mat2;
    /*floating*/ float4x4 mat3;
};

#pragma pack_matrix(random_string_foo)

cbuffer Example
{
    MyBuffer1 g_MyBuffer1;
    MyBuffer2 g_MyBuffer2;
    column_major float4x4 mat1a;
};

float4 main() : SV_Target0
{
    return 
        g_MyBuffer1.mat1[0] + g_MyBuffer1.mat2[0] + g_MyBuffer1.mat3[0] +
        g_MyBuffer2.mat1[0] + g_MyBuffer2.mat2[0] + g_MyBuffer2.mat3[0];
}
