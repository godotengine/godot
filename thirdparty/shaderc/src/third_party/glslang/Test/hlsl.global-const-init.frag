
cbuffer CB {
    float4 foo;
};

static const float4 bar = foo; // test const (in the immutable sense) initializer from non-const.

static const float2 a1[2] = { { 1, 2 }, { foo.x, 4 } }; // not entirely constant
static const float2 a2[2] = { { 5, 6 }, { 7, 8 } };     // entirely constant

float4 main() : SV_Target0
{
    return bar;
}
