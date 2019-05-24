#version 450

#extension GL_EXT_fragment_invocation_density : require

layout (location = 0) out vec2 FragSize;
layout (location = 2) out int FragInvocationCount;

void main () {
    FragSize = gl_FragSizeEXT;
    FragInvocationCount = gl_FragInvocationCountEXT;
}
