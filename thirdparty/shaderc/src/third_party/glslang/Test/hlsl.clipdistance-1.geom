struct S {
    float4 pos     : SV_Position;
    float2 clip    : SV_ClipDistance0;
};

[maxvertexcount(3)]
void main(triangle in float4 pos[3] : SV_Position, 
          triangle in uint VertexID[3] : VertexID,
          inout LineStream<S> OutputStream,
          triangle in float4 clip[3] : SV_ClipDistance)   // externally: an array 3 of array 4 (not vec4!) of float.
{
    S s;

    s.pos = pos[0];
    s.clip = clip[0].xy;

    OutputStream.Append(s);
}

