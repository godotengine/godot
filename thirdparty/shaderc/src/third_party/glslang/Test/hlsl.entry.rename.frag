
struct PS_OUTPUT
{
    float4 Color : SV_Target0;
};

void not_the_entry_point() { }
int also_not_the_entry_point;

PS_OUTPUT main()
{
    PS_OUTPUT psout;
    psout.Color = 0;
    return psout;
}
