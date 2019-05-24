
struct VertexData {
    float4 position : POSITION;
    float4 color    : COLOR0;
    float2 uv       : TEXCOORD0;
};

struct PS_IN {
    float4 position : SV_POSITION;
    float4 color    : COLOR0;
    float2 uv       : TEXCOORD0;
};

[maxvertexcount(4)]
void main(line VertexData vin[2], inout TriangleStream<PS_IN> outStream)
{
    PS_IN vout;

    vout.color = vin[1].color;
    vout.uv = vin[1].uv;
    vout.position = vin[1].position;
    outStream.Append(vout);
}
