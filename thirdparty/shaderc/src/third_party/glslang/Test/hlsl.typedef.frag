typedef float4 myVec4;

float4 ShaderFunction(float4 input, int ii) : COLOR0
{
    typedef int myInt;
    myVec4 a1 = myVec4(1.0);
    myInt i = 2;
    typedef myInt myInt2;
    myInt2 j = ii;
    return input * a1 + myVec4(i + j);
}
