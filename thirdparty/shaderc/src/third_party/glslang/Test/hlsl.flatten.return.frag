
struct PS_OUTPUT
{
    float4 color : SV_Target0;
    float other_struct_member1;
    float other_struct_member2;
    float other_struct_member3;
};

PS_OUTPUT Func1()
{
    return PS_OUTPUT(float4(1,1,1,1), 2, 3, 4);
}

PS_OUTPUT main()
{
    return Func1();
}
