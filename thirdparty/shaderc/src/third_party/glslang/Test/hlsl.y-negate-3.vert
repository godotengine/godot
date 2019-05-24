// Test Y negation from entry point out parameter

float4 position;

struct VS_OUT {
    float4 pos : SV_Position;
    int somethingelse;
};

VS_OUT main()
{
    VS_OUT vsout;

    vsout.pos = position;
    vsout.somethingelse = 42;

    return vsout;
}
