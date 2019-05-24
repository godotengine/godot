struct S {
    float2 f;
};

[[vk::binding(1)]]
StructuredBuffer<S> buffer1;

[[vk::binding(3, 2)]]
StructuredBuffer<S> buffer3;

[[vk::input_attachment_index(4)]]
Texture2D<float4> attach;

[[vk::constant_id(13)]] const int ci = 11;

[[vk::push_constant]] cbuffer pcBuf { int a; };

[[vk::location(7)]] float4
main([[vk::location(8)]] float4 input: A) : B
{
    return input + attach.Load(float2(0.5));// * a;
}
