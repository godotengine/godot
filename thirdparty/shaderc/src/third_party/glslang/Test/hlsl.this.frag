static float2 var = float2(1.0, 2.0);

struct type1
{
    int memFun1(int3 var)
    {
        return var.z + this.var + var2;
    }
    int memFun2(int a)
    {
        int3 var = int3(1,2,3);
        return var.z + (int)bar.y + this.var2;
    }
    float2 bar;
    int var;
    int var2;
};

float4 main() : SV_Target0
{
   type1 T;
   T.bar = var;
   T.var = 7;
   T.var2 = 9;
   int i = T.memFun1(int3(10,11,12));
   i += T.memFun2(17);

   return float4(i,i,i,i);
}
