#version 440 core

layout(binding = 0, std140) uniform ubo_block {
	float unused_uniform;
	float shared_uniform;
	float vsonly_uniform;
	float fsonly_uniform;
} ubo;

in float vertout;

out float fragout;

void main()
{
    fragout = vertout;
    fragout += ubo.shared_uniform;
    fragout += ubo.fsonly_uniform;
}
