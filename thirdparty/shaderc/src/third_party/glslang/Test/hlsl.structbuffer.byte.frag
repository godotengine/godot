
ByteAddressBuffer sbuf;

float4 main(uint pos : FOO) : SV_Target0
{
    uint size;
    sbuf.GetDimensions(size);

    return sbuf.Load(pos) +
        float4(sbuf.Load2(pos+4), 0, 0) +
        float4(sbuf.Load3(pos+8), 0) +
        sbuf.Load4(pos+12);
}
