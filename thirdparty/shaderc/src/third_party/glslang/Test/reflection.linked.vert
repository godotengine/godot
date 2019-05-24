#version 440 core

layout(binding = 0, std140) uniform ubo_block {
	float unused_uniform;
	float shared_uniform;
	float vsonly_uniform;
	float fsonly_uniform;
} ubo;

in float vertin;

out float vertout;

void main()
{
    vertout = vertin;
    vertout += ubo.shared_uniform;
    vertout += ubo.vsonly_uniform;
}
