struct S {
    float f;
    float4 pos : SV_Position;
};

float4 main(int i, S input[3]) : COLOR0
{
    S a[3];
    input = a;

    return a[1].pos;
}
