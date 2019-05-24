#version 450

#extension GL_EXT_multiview : enable

out vec4 color;

void main() {
    color = vec4(gl_ViewIndex, 0, 0, 0);
}
