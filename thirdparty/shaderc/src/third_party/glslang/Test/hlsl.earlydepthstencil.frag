RWTexture2D<uint> Values;

struct InputStruct {
	float4 Position : SV_POSITION;
};

[earlydepthstencil]
uint main(InputStruct input) : SV_Target {
	uint oldVal;
	InterlockedExchange(Values[uint2(input.Position.x, input.Position.y)], 1.0, oldVal);
	return oldVal;
}
