#version 330

#extension GL_OVR_multiview : enable

layout(num_views = 2) in;

void main() {
    gl_Position = vec4(gl_ViewID_OVR, 0, 0, 0);
}
