struct OS {
    SamplerState ss;
    float a;
    Texture2D tex;
};

SamplerState gss;
SamplerState gss2;
Texture2D gtex;

float4 osCall(OS s)
{
    return s.a * s.tex.Sample(s.ss, float2(0.2, 0.3));
}

float4 main() : SV_TARGET0
{
    OS os;
    os.ss = gss2;
    os.ss = gss;
    os.tex = gtex;
    os.a = 3.0;

    // this should give an error
    //SamplerState localss;
    //localss = gss2;

    return osCall(os);
}
