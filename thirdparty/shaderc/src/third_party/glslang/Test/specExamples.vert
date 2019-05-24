#version 430

#extension GL_3DL_array_objects : enable

out Vertex {
    vec4 Position;  // API transform/feedback will use “Vertex.Position”
    vec2 Texture;
} Coords;           // shader will use “Coords.Position”

out Vertex2 {
    vec4 Color;     // API will use “Color”
};

uniform Transform {  // API uses “Transform[2]” to refer to instance 2
    mat4           ModelViewMatrix;
    mat4           ModelViewProjectionMatrix;
    vec4           a[];  // array will get implicitly sized
    float          Deformation;
} transforms[4];

layout(location = 3) in vec4 normal;
layout(location = 6) in vec4 colors[3];
layout(location = 9) in mat4 transforms2[2];

layout(location = 3) struct S {
    vec3 a1;
    mat2 b;
    vec4 c[2];
} s;

layout(triangles, invocations = 6) in;

layout(lines) in;    // legal for Color2, input size is 2, matching Color2

layout(triangle_strip, max_vertices = 60) out;  // order does not matter
layout(max_vertices = 60) out;      // redeclaration okay
layout(triangle_strip) out;         // redeclaration okay
//layout(points) out;                 // error, contradicts triangle_strip
//layout(max_vertices = 30) out;      // error, contradicts 60

layout(stream = 1) out;

layout(stream=1) out;             // default is now stream 1
out vec4 var1;                    // var1 gets default stream (1)
layout(stream=2) out Block1 {     // "Block1" belongs to stream 2
    layout(stream=2) vec4 var2;   // redundant block member stream decl
    layout(stream=3) vec2 var3;   // ILLEGAL (must match block stream)
    vec3 var4;                    // belongs to stream 2
};
layout(stream=0) out;             // default is now stream 0
out vec4 var5;                    // var5 gets default stream (0)
out Block2 {                      // "Block2" gets default stream (0)
    vec4 var6;
};
layout(stream=3) out vec4 var7;   // var7 belongs to stream 3

layout(shared, column_major) uniform;
layout(shared, column_major) buffer;

layout(row_major, column_major)

layout(shared, row_major) uniform; // default is now shared and row_major

layout(std140) uniform Transform2 { // layout of this block is std140
    mat4 M1;                       // row_major
    layout(column_major) mat4 M2;  // column major
    mat3 N1;                       // row_major
};

layout(column_major) uniform T3 {  // shared and column_major
    mat4 M13;                      // column_major
    layout(row_major) mat4 m14;    // row major
    mat3 N12;                      // column_major
};

// in one compilation unit...
layout(binding=3) uniform sampler2D s17; // s bound to unit 3

// in another compilation unit...
uniform sampler2D s17;                   // okay, s still bound at 3

// in another compilation unit...
//layout(binding=4) uniform sampler2D s; // ERROR: contradictory bindings

layout (binding = 2, offset = 4) uniform atomic_uint a2;

layout (binding = 2) uniform atomic_uint bar;

layout (binding = 2, offset = 4) uniform atomic_uint;

layout (binding = 2) uniform atomic_uint bar; // offset is 4
layout (offset = 8) uniform atomic_uint bar23;  // error, no default binding

layout (binding=3, offset=4) uniform atomic_uint a2; // offset = 4
layout (binding=2) uniform atomic_uint b2;           // offset = 0
layout (binding=3) uniform atomic_uint c2;           // offset = 8
layout (binding=2) uniform atomic_uint d2;           // offset = 4

//layout (offset=4)                // error, must include binding
//layout (binding=1, offset=0)  a; // okay
//layout (binding=2, offset=0)  b; // okay
//layout (binding=1, offset=0)  c; // error, offsets must not be shared
//                                 //        between a and c
//layout (binding=1, offset=2)  d; // error, overlaps offset 0 of a

flat  in vec4 gl_FrontColor;  // input to geometry shader, no “gl_in[]”
flat out vec4 gl_FrontColor;  // output from geometry shader

invariant gl_Position;   // make existing gl_Position be invariant

out vec3 ColorInv;
invariant ColorIvn;      // make existing Color be invariant

invariant centroid out vec3 Color4;
precise out vec4 position;

out vec3 Color5;
precise Color5;            // make existing Color be precise
in vec4 a, b, c, d;
precise out vec4 v;

coherent buffer Block {
    readonly vec4 member1;
    vec4 member2;
};

buffer Block2a {
    coherent readonly vec4 member1A;
    coherent vec4 member2A;
};

shared vec4 shv;

vec4 funcA(restrict image2D a)   {  }

vec4 funcB(image2D a)            {  }
layout(rgba32f) uniform image2D img1;
layout(rgba32f) coherent uniform image2D img2;

float func(float e, float f, float g, float h)
{
    return (e*f) + (g*h);            // no constraint on order or 
                                     // operator consistency
}

float func2(float e, float f, float g, float h)
{
    precise float result = (e*f) + (g*h);  // ensures same precision for
                                           // the two multiplies
    return result;
}

float func3(float i, float j, precise out float k)
{
    k = i * i + j;                   // precise, due to <k> declaration
}

void main()
{
    vec3 r = vec3(a * b);           // precise, used to compute v.xyz
    vec3 s = vec3(c * d);           // precise, used to compute v.xyz
    v.xyz = r + s;                          // precise                      
    v.w = (a.w * b.w) + (c.w * d.w);        // precise
    v.x = func(a.x, b.x, c.x, d.x);         // values computed in func()
                                            // are NOT precise
    v.x = func2(a.x, b.x, c.x, d.x);        // precise!
    func3(a.x * b.x, c.x * d.x, v.x);       // precise!
        
    funcA(img1);              // OK, adding "restrict" is allowed
    funcB(img2);              // illegal, stripping "coherent" is not

    {
        struct light {
            float intensity;
            vec3 position;
        };

        light lightVar = light(3.0, vec3(1.0, 2.0, 3.0));
    }
    {
        const float c[3] = float[3](5.0, 7.2, 1.1);
        const float d[3] = float[](5.0, 7.2, 1.1);

        float g;
        float a[5] = float[5](g, 1, g, 2.3, g);
        float b[3];

        b = float[3](g, g + 1.0, g + 2.0);
    }
    {
        vec4 b[2] = { vec4(1.0), vec4(1.0) };
        vec4[3][2](b, b, b);        // constructor
        vec4[][2](b, b, b);         // constructor, valid, size deduced
        vec4[3][](b, b, b);         // compile-time error, invalid type constructed
    }
}
