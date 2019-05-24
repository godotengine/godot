static float2 i = float2(1.0, 2.0);

struct type1
{
    void setmem(float4 m) { memVar = m; }
    void seti(int si) { i = si; }
    float4 memVar;
    float4 memFun(float4 a) : SV_Position
    {
        return i * a + memVar;
    }
    int memFun(int a) : SV_Position
    {
        return i + a - memVar.z;
    }
    int i;
};

static float2 j = i;

struct type2
{
    float2 memFun() { return i; }
};

float4 main() : SV_Target0
{
   type1 test;
   test.setmem(float4(2.0,2.0,2.0,2.0));
   test.seti(17);
   float4 f4 = float4(1.0,1.0,1.0,1.0);
   f4 += test.memFun(float4(5.0f,5.0f,5.0f,5.0f));
   f4 += test.memFun(7);
   return f4;
}
