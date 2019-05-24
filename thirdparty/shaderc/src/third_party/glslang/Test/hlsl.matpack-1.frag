struct MyBuffer1
{
    column_major float4x4 mat1;
    row_major    float4x4 mat2;
    float4 vec1;
    float  foo;
};

struct MyBuffer2
{
    row_major float4x4 mat1;
    float4 vec1;
};

cbuffer Example
{
    MyBuffer1 g_MyBuffer1;
    MyBuffer2 g_MyBuffer2;
    column_major float4x4 mat1a;
};

float4 main() : SV_Target0
{
    return mul(g_MyBuffer1.mat1, g_MyBuffer1.vec1) +
           mul(g_MyBuffer2.mat1, g_MyBuffer2.vec1);
}

