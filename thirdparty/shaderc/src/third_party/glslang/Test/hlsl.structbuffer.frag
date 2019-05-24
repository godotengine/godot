struct sb_t
{
    float3 color;
    bool   test;
    bool   test2;
}; // stride = 20

StructuredBuffer<sb_t>  sbuf : register(t10);
StructuredBuffer<float> sbuf2;

float4 main(uint pos : FOO) : SV_Target0
{
    sb_t mydata = sbuf.Load(pos);

    uint size;
    uint stride;
    sbuf.GetDimensions(size, stride);

    if (sbuf[pos].test)
        return float4(sbuf[pos].color + sbuf2[pos], 0);
    else
        return mydata.color.x + size + stride;
}
