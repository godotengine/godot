RWStructuredBuffer<uint4>      sbuf_rw_i;
RWStructuredBuffer<uint4>      sbuf_rw_d;

RWStructuredBuffer<uint4>      sbuf_rw_nocounter; // doesn't use inc or dec

float4 main(uint pos : FOO) : SV_Target0
{
    uint4 result = 0;

    sbuf_rw_i[7];
    sbuf_rw_d[7];

    sbuf_rw_nocounter[5] = 2;

    uint c1 = sbuf_rw_i.IncrementCounter();
    uint c2 = sbuf_rw_d.DecrementCounter();

    return float4(result.x, result.y, c1, c2);
}
