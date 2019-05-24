struct PS_OUTPUT
{
    float4 Color : SV_Target0;
};

uniform int    ival;
uniform int4   ival4;
uniform float  fval;
uniform float4 fval4;

PS_OUTPUT main()
{
    if (ival && fval);
    if (ival || fval);

    PS_OUTPUT psout;
    psout.Color = 1.0;
    return psout;
}


