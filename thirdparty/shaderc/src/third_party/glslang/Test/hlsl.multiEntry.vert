Buffer<float4> Position;

float4 FakeEntrypoint(uint Index : SV_VERTEXID) : SV_POSITION
{ 
    return Position.Load(Index);
}

float4 RealEntrypoint(uint Index : SV_VERTEXID) : SV_POSITION
{ 
    return FakeEntrypoint(Index);
}