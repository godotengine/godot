#version 450
#extension GL_NV_sample_mask_override_coverage : enable
in vec4 color;
layout(override_coverage) out int gl_SampleMask[];
void main() {
    gl_SampleMask[0] = int(0xFFFFFFFF);
}