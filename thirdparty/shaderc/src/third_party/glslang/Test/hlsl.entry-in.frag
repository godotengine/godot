struct InParam {
    float2 v;
    float4 fragCoord : SV_POSITION;
    int2 i2;
};

float fun(InParam p)
{
    return p.v.y + p.fragCoord.x;
}

float4 PixelShaderFunction(InParam i) : COLOR0
{
    InParam local;
    local = i;
    float ret1 = fun(local);
    float ret2 = fun(i);

    return local.fragCoord * ret1 * ret2;
}
