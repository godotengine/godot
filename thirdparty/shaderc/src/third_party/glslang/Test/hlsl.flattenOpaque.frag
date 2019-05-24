struct os {
    sampler2D s2D;
};

struct os2 {
    sampler2D s2D;
    Texture2D tex;
};

Texture2D tex;
os s;
os2 s2;

float4 osCall1(os s)
{
    return tex.Sample(s.s2D, float2(0.2, 0.3));
}

float4 osCall2(os s, float2 f2)
{
    return tex.Sample(s.s2D, f2);
}

float4 os2Call1(os2 s)
{
    return s.tex.Sample(s.s2D, float2(0.2, 0.3));
}

float4 os2Call2(os2 s, float2 f2)
{
    return s.tex.Sample(s.s2D, f2);
}

float4 main() : SV_TARGET0
{
    return osCall1(s) +
           osCall2(s, float2(0.2, 0.3)) +
           os2Call1(s2) +
           os2Call2(s2, float2(0.2, 0.3));
}
