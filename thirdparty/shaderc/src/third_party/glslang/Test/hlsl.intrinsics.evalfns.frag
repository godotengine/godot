
void main(float inF1, float2 inF2, float3 inF3, float4 inF4, int2 inI2) : COLOR
{
    EvaluateAttributeSnapped(inF1, int2(8,15));
    EvaluateAttributeSnapped(inF2, int2(0,1));
    EvaluateAttributeSnapped(inF3, int2(3,10));
    EvaluateAttributeSnapped(inF4, int2(7,8));

    EvaluateAttributeSnapped(inF1, inI2);
}
