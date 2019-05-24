struct PSInput {
  float interp;
  uint no_interp;
};

void main(PSInput input : INPUT, out float4 out1 : SV_TARGET1, out float4 out2 : SV_TARGET3)
{
    out1 = 1;
    out2 = 0;
}