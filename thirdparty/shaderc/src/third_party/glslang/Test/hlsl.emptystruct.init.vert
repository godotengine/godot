struct Test { };

static const Test Test_Empty;

float4 main(in uint vertexIndex : VERTEXID) : VS_OUT_POSITION
{
    return 0;
}
