#version 150

in mat3 colorTransform;
in vec3 Color;
in mat4 m, n;

in mat4x3 um43;
in mat3x4 un34;
in mat2 um2;
in mat3 um3;
in mat4 um4;

in vec4 v;

in vec3 u;

out vec4 FragColor;

void main()
{
    mat3x4 m34 = outerProduct(v, u);

    m34 += mat3x4(4.3);

    FragColor = vec4(Color, 1.0);
    FragColor *= vec4(FragColor * m34, 1.0);

    m34 *= v.x;

    mat4 m44 = mat4(un34);

    m44 += m34 * um43;

    FragColor += (-m44) * v;

    FragColor *= matrixCompMult(m44, m44);

    m34 = transpose(um43);
    FragColor *= vec4(FragColor * m34, 1.0);
    FragColor *= vec4(determinant(um4));
    mat2 inv = inverse(um2);
    FragColor *= vec4(inv[0][0], inv[1][0], inv[0][1], inv[1][1]);
    mat3 inv3 = inverse(um3);
    FragColor *= vec4(inv3[2][1]);

    mat4 inv4 = inverse(um4);
    FragColor *= inv4;

    FragColor = vec4(FragColor * matrixCompMult(un34, un34), FragColor.w);
}
