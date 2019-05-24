
StructuredBuffer<uint4>  sbuf : register(t10);

uint4 get(in StructuredBuffer<uint4> sb, uint bufferOffset)
{
    return sb[bufferOffset];
}

void set(in RWStructuredBuffer<uint4> sb, uint bufferOffset, uint4 data)
{
    sb[bufferOffset] = data;
}

RWStructuredBuffer<uint4> sbuf2;

// Not shared, because of type difference.
StructuredBuffer<uint3>  sbuf3 : register(t12);

float4 main(uint pos : FOO) : SV_Target0
{
    set(sbuf2, 2, get(sbuf, 3));

    return 0;
}
