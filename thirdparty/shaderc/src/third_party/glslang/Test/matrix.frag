#version 130

//#define TEST_POST_110

uniform mat3 colorTransform;
varying vec3 Color;
uniform mat4 m, n;

#ifdef TEST_POST_110
uniform mat4x3 um43;
uniform mat3x4 un34;
#else
uniform mat4 um43;
uniform mat4 un34;
#endif

varying vec4 v;

#ifdef TEST_POST_110
varying vec3 u;
#else
varying vec4 u;
#endif

void main()
{
    gl_FragColor = vec4(un34[1]);
    gl_FragColor += vec4(Color * colorTransform, 1.0);

    if (m != n)
        gl_FragColor += v;
   else {
        gl_FragColor += m * v;
        gl_FragColor += v * (m - n);
   }
    
#ifdef TEST_POST_110
    mat3x4 m34 = outerProduct(v, u);
    m34 += mat4(v.x);
    m34 += mat4(u, u.x, u, u.x, u, u.x, u.x);
#else
    mat4 m34 = mat4(v.x*u.x, v.x*u.y, v.x*u.z, v.x*u.w, 
                    v.y*u.x, v.y*u.y, v.y*u.z, v.y*u.w, 
                    v.z*u.x, v.z*u.y, v.z*u.z, v.z*u.w, 
                    v.w*u.x, v.w*u.y, v.w*u.z, v.w*u.w);
    m34 += mat4(v.x);
    m34 += mat4(u, u.x, u, u.x, u, u.x, u.x);

#endif

    if (m34 == un34)
        gl_FragColor += m34 * u;
    else
        gl_FragColor += (un34 * um43) * v;
}
