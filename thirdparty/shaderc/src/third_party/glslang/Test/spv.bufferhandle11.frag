#version 450
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_8bit_storage : enable
#extension GL_EXT_buffer_reference : enable

layout(std140, binding = 0) buffer AcBlock { highp uint ac_numPassed; };

layout(std140, buffer_reference) buffer Block
{
	uint8_t var;
};
layout (push_constant, std430) uniform PC {
	Block block;
};

bool compare_uint8_t  (highp uint a, highp uint b)    { return a == b; }

void main (void)
{
	bool allOk = true;
	allOk = allOk && compare_uint8_t(uint(block.var), 7u);
	if (allOk)
		ac_numPassed++;

	block.var = uint8_t(9u);
}