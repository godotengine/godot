AppendStructuredBuffer<float4>  sbuf_a;
ConsumeStructuredBuffer<float4> sbuf_c;

AppendStructuredBuffer<float4>  sbuf_unused;

float4 main(uint pos : FOO) : SV_Target0
{
    sbuf_a.Append(float4(1,2,3,4));

    return sbuf_c.Consume();
}
