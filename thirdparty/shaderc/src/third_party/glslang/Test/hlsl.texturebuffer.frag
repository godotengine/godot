
struct Data {
    float4  f;
    int4    i;
};

TextureBuffer<Data> TextureBuffer_var : register(t0);

tbuffer tbuf2 {
    float4 f2;
    int4   i2;
};

float4 main(float4 pos : SV_POSITION) : SV_TARGET
{
    return TextureBuffer_var.f + f2;
}
