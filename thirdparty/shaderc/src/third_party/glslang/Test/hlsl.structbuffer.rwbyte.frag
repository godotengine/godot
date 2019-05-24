
RWByteAddressBuffer sbuf;

float4 main(uint pos : FOO) : SV_Target0
{
    uint size;
    sbuf.GetDimensions(size);

    sbuf.Store(pos,  sbuf.Load(pos));
    sbuf.Store2(pos, sbuf.Load2(pos));
    sbuf.Store3(pos, sbuf.Load3(pos));
    sbuf.Store4(pos, sbuf.Load4(pos));

    return sbuf.Load(pos);
}
