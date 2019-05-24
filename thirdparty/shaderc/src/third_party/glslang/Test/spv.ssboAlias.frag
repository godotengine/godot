AppendStructuredBuffer<uint> Buf1 : register(u1);
AppendStructuredBuffer<uint> Buf2 : register(u2);
AppendStructuredBuffer<uint> Buf3 : register(u1);

float4 main() : SV_Target
{
	Buf1.Append(10u);
	Buf2.Append(20u);
	return float4(1.0, 3.0, 5.0, 1.0);
}