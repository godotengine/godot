Texture2D tex[2];

struct Packed {
    int a;
    Texture2D     membTex[2];
    int b;
};

float4 main(float4 pos : POSITION) : SV_POSITION
{
    Packed packed;

    packed.membTex = tex;

    return pos;
}