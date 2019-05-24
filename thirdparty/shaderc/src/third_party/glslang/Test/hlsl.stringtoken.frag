
struct PS_OUTPUT
{
    float4 Color : SV_Target0;
};

Texture2D TestTexture <
    string TestAttribute1 = "TestAttribute";
    string TestAttribute2 = "false";
    int    TestAttribute3 = 3;
>;

uniform float4 TestUF <string StrValue = "foo";>;

PS_OUTPUT main()
{
   PS_OUTPUT psout;
   psout.Color = float4(0,0,0,1);
   return psout;
}
