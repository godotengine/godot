struct S {
    float clip0 : SV_Position;
    float clip0 : SV_ClipDistance0;
    float cull0 : SV_CullDistance0;
    uint vpai   : SV_ViewportArrayIndex;
    uint rtai   : SV_RenderTargetArrayIndex;
    int ii      : SV_InstanceID;
};

[maxvertexcount(4)]
void main(triangle in uint VertexID[3] : VertexID,
       inout LineStream<S> OutputStream)
{
    S s;
    OutputStream.Append(s);
}
