#version 450

// try using a fragment-only extension in a vertex shader
#extension GL_EXT_fragment_invocation_density : require

layout (location = 0) out uvec2 FragSize;
layout (location = 2) out int FragInvocationCount;

void main () {
    FragSize = gl_FragSizeEXT;
    FragInvocationCount = gl_FragInvocationCountEXT;
}
