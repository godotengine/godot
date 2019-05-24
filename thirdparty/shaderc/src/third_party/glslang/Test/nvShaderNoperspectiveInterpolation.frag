#version 300 es

precision mediump float;

noperspective in vec4 bad; // ERROR

#extension GL_NV_shader_noperspective_interpolation : enable

noperspective in vec4 color;

out vec4 fragColor;

void main() {
    fragColor = color;
}