struct PSInput {
  float interp;
  uint no_interp;
};

float4 main(PSInput input : INPUT) : SV_TARGET
{
  return float4(float(input.no_interp), input.interp, 0, 1);
}