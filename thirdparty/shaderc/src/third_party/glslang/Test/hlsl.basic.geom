struct PSInput
{
    float  myfloat    : SOME_SEMANTIC;
    int    something  : ANOTHER_SEMANTIC;
};

struct nametest {
    int Append;        // these are valid names even though they are also method names.
    int RestartStrip;  // ...
};

[maxvertexcount(4)]
void main(triangle in uint VertexID[3] : VertexID,
          triangle uint test[3] : FOO, 
          inout LineStream<PSInput> OutputStream)
{
    PSInput Vert;

    Vert.myfloat    = test[0] + test[1] + test[2];
    Vert.something  = VertexID[0];

    OutputStream.Append(Vert);
    OutputStream.Append(Vert);
    OutputStream.RestartStrip();
}
