struct PSInput
{
    float4 Pos      : SV_POSITION;
    float2 TexCoord : TEXCOORD;
    float3 TerrainPos : TERRAINPOS;
    uint VertexID : VertexID;
};

typedef PSInput foo_t[2][3];

[maxvertexcount(4)]
void main(point uint v[1] : VertexID, inout TriangleStream<PSInput> OutputStream)
{
    foo_t Verts;

    PSInput Out = (PSInput) 0;

    for (int x=0; x<2; ++x)
        for (int y=0; y<2; ++y)
            Verts[x][y] = Out;
}
