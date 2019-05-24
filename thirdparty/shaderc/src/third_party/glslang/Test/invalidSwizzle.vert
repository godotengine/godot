#version 420

void f();
uniform sampler2D s;

void main() {
    vec2 v = s.rr; // Swizzles do not apply to samplers
    f().xx; // Scalar swizzle does not apply to void
    f().xy; // Vector swizzle does not apply either
}