void PixelShaderFunction(float4 input) : COLOR0
{
    [unroll];
    [];
    [][][];
    [unroll(4)];
    [allow_uav_condition];
    [unroll(4)] [allow_uav_condition];
    [  loop  ];
    [fastopt];
    [branch] if (0);
    [flatten];
}
