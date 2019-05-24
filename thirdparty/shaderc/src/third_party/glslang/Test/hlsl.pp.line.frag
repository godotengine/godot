
#line 1

struct PS_OUTPUT
{
    float4 Color : SV_Target0;
    float  Depth : SV_Depth;
};

#line 2

PS_OUTPUT main()
{
   PS_OUTPUT psout;

#line 123 "SomeFile.frag"

   int thisLineIs = __LINE__;  // gets 124

   psout.Color = float4(thisLineIs, 0, 0, 1);
   psout.Depth = 1.0;

   return psout;
}
