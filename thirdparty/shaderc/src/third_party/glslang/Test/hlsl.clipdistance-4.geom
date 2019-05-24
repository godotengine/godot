struct S {
    float4 pos     : SV_Position;
    float2 clip0   : SV_ClipDistance0;  // clip0 and clip1 form an array of float[4] externally.
    float2 clip1   : SV_ClipDistance1;
};

[maxvertexcount(3)]
void main(triangle in float4 pos[3] : SV_Position, 
          triangle in uint VertexID[3] : VertexID,
          inout LineStream<S> OutputStream,
          triangle in float2 clip0[3] : SV_ClipDistance0,  // test input arrayed semantic vars
          triangle in float2 clip1[3] : SV_ClipDistance1)
{
    S s;

    s.pos = pos[0];
    s.clip0 = clip0[0];
    s.clip1 = clip1[0];

    OutputStream.Append(s);
}
