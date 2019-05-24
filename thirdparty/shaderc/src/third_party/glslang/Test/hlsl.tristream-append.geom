struct GSPS_INPUT
{
};

// Test Append() method appearing before declaration of entry point's stream output.

void EmitVertex(in GSPS_INPUT output, inout TriangleStream<GSPS_INPUT> TriStream)
{
    TriStream.Append( output );
}

[maxvertexcount(3)]
void main( triangle GSPS_INPUT input[3], inout TriangleStream<GSPS_INPUT> TriStream )
{
    EmitVertex(input[0], TriStream);
    EmitVertex(input[1], TriStream);
    EmitVertex(input[2], TriStream);
}
