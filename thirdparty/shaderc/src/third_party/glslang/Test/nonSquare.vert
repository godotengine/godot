#version 120

attribute vec3 v3;
attribute vec4 v4;

uniform mat3x2 m32;

const vec2 cv2 = vec2(10.0, 20.0);
const mat2x4 m24 = mat2x4(3.0);
const mat4x2 m42 = mat4x2(1.0, 2.0, 
                          3.0, 4.0,
                          5.0, 6.0, 
                          7.0, 8.0);

void main()
{
    mat2x3 m23;
    vec2 a, b;

    a = v3 * m23;
    b = m32 * v3;

    gl_Position = vec4(m23 * m32 * v3, m24[1][3]) + 
                  (m24 * m42) * v4 + cv2 * m42 + m24 * cv2 + vec4(cv2[1], cv2.x, m42[2][1], m42[2][0]);
}
