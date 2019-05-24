
RWByteAddressBuffer sbuf;

float4 main(uint pos : FOO) : SV_Target0
{
    uint u;

    sbuf.InterlockedAdd(8, 1);
    sbuf.InterlockedAdd(8, 1, u);
    sbuf.InterlockedAnd(8, 1);
    sbuf.InterlockedAnd(8, 1, u);
    sbuf.InterlockedCompareExchange(8, 1, 2, u);
    // sbuf.InterlockedCompareStore(8, 1, 2); // TODO: ...
    sbuf.InterlockedExchange(8, 1, u);
    sbuf.InterlockedMax(8, 1);
    sbuf.InterlockedMax(8, 1, u);
    sbuf.InterlockedMin(8, 1);
    sbuf.InterlockedMin(8, 1, u);
    sbuf.InterlockedOr(8, 1);
    sbuf.InterlockedOr(8, 1, u);
    sbuf.InterlockedXor(8, 1);
    sbuf.InterlockedXor(8, 1, u);

    return sbuf.Load(pos);
}
