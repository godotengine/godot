
// Test trivial case for structure splitting: the IN and OUT structs have ONLY an interstage IO.
// This should fall back to flattening, and not produce any empty structures.

struct VS_INPUT
{
    float4 Pos_in : SV_Position;
};

struct VS_OUTPUT
{
    float4 Pos : SV_Position;
};

VS_OUTPUT main(VS_INPUT vsin, float4 Pos_loose : SV_Position)
{
    VS_OUTPUT vsout;

    vsout.Pos = vsin.Pos_in + Pos_loose;

    return vsout;
}
