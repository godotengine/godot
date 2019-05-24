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
    !ival;  // scalar int
    !ival4; // vector int

    !fval;  // scalar float
    !fval4; // vector float
    
    if (ival);
    if (fval);
    if (!ival);
    if (!fval);

    PS_OUTPUT psout;
    psout.Color = 1.0;
    return psout;
}


