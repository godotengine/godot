#version 120

attribute vec3 v3;

uniform mat3x2 m32;

const mat2x4 m24 = mat2x4(1.0, 2.0, 
                          3.0, 4.0,
                          3.0, 4.0,
                          3.0, 4.0, 5.0);          // ERROR, too many arguments

void main()
{
    mat2x3 m23;
    vec3 a, b;

    a = v3 * m23;      // ERROR, type mismatch
    b = m32 * v3;      // ERROR, type mismatch
    m23.xy;            // ERROR, can't use .

    gl_Position = vec4(m23 * m32 * v3, m24[2][4]);  // ERROR, 2 and 4 are out of range
    m23 *= m23;        // ERROR, right side needs to be square
    m23 *= m32;        // ERROR, left columns must match right rows
}
