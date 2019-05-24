struct OutParam {
    float2 v;
    int2 i;
};

void fun(out OutParam op)
{
    op.v = float2(0.4);
    op.i = int2(7);
}

float4 PixelShaderFunction(float4 input, out float4 out1, out OutParam out2, out OutParam out3) : COLOR0
{
    out1 = input;
    out2.v = 2.0;
    out2.i = 3;
    OutParam local;
    local.v = 12.0;
    local.i = 13;
    fun(out3);

    return out1;
}
