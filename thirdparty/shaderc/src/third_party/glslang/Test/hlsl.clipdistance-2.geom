struct S {
    float4 pos     : SV_Position;
    float2 clip[2] : SV_ClipDistance0;
};

[maxvertexcount(3)]
void main(triangle in float4 pos[3] : SV_Position, 
          triangle in uint VertexID[3] : VertexID,
          inout LineStream<S> OutputStream,
          triangle in float2 clip[3][2] : SV_ClipDistance) // externally: an array 3 of array 4 of float.
{
    S s;

    s.pos = pos[0];
    s.clip[0] = clip[0][0];
    s.clip[1] = clip[0][1];

    OutputStream.Append(s);
}
