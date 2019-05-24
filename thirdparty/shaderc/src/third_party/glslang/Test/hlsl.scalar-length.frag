float4 main() : SV_Target0
{
    float4 test = { 0, 1, 2, 3 };

    return length(test.a);
}
