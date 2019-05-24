struct sb_t
{
    float3 color;
    bool   test;
};


RWStructuredBuffer<sb_t>  sbuf;
RWStructuredBuffer<float> sbuf2;

float4 main(uint pos : FOO) : SV_Target0
{
    sbuf2[pos+1] = 42;

    uint size;
    uint stride;
    sbuf.GetDimensions(size, stride);

    if (sbuf[pos].test)
        return float4(sbuf[pos].color + sbuf2[pos], 0);
    else
        return size + stride;
}
