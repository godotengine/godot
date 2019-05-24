struct PSInput {
  float interp;
  uint no_interp;
};

struct PSOutput {
    float4 o1 : SV_TARGET2;
    float4 o2 : SV_TARGET1;
};

PSOutput main(PSInput input : INPUT, out float4 po : SV_TARGET0)
{
    PSOutput pso;
    pso.o1 = float4(float(input.no_interp), input.interp, 0, 1);
    pso.o2 = 1;
    po = 0;

    return pso;
}