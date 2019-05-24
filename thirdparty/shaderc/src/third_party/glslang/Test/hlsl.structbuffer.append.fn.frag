
// float4 Fn1(ConsumeStructuredBuffer<float4> arg_c)
// {
//     return arg_c.Consume();
// }

float4 Fn2(AppendStructuredBuffer<float4> arg_a, ConsumeStructuredBuffer<float4> arg_c)
{
    arg_a.Append(float4(1,2,3,4));
    return arg_c.Consume();
}

AppendStructuredBuffer<float4>  sbuf_a;
ConsumeStructuredBuffer<float4> sbuf_c;

AppendStructuredBuffer<float4>  sbuf_unused;

float4 main(uint pos : FOO) : SV_Target0
{
    // Fn1(sbuf_c);

    return Fn2(sbuf_a, sbuf_c);
}
