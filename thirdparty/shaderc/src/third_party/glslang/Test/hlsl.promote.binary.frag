struct PS_OUTPUT
{
    float4 Color : SV_Target0;
};

uniform bool   bval;
uniform bool4  bval4;
uniform int    ival;
uniform int4   ival4;
uniform float  fval;
uniform float4 fval4;

PS_OUTPUT main()
{
    ival  % fval;
    ival4 % fval4;

    bval  % fval;
    bval4 % fval4;

    int l_int = 1;
    l_int %= fval;

    PS_OUTPUT psout;
    psout.Color = 0;
    return psout;
}

