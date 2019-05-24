struct PS_OUTPUT { float4 color : SV_Target0; };

PS_OUTPUT main()
{
    // Evaluates to a sequence: 3, 4, 5, 6, 7, 8, and a float2(9,10), float2(11,12) sequence
    (int(3));
    (int(3) + int(1));
    (int(3) + int(1) + int(1));
    (((int(6))));
    (int(7.0));
    ((int((2)) ? 8 : 8));
    (float2(9, 10), float2(11, 12));

    PS_OUTPUT ps_output;
    ps_output.color = 1.0;
    return ps_output;
}
