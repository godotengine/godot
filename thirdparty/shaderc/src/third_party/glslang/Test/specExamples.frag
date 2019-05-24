#version 430

#extension GL_3DL_array_objects : enable

int  a = 0xffffffff;  // 32 bits, a gets the value -1
int  b = 0xffffffffU; // ERROR: can't convert uint to int
uint c = 0xffffffff;  // 32 bits, c gets the value 0xFFFFFFFF
uint d = 0xffffffffU; // 32 bits, d gets the value 0xFFFFFFFF
int  e = -1;          // the literal is "1", then negation is performed,
                      //   and the resulting non-literal 32-bit signed 
                      //   bit pattern of 0xFFFFFFFF is assigned, giving e 
                      //   the value of -1.
uint f = -1u;         // the literal is "1u", then negation is performed,
                      //   and the resulting non-literal 32-bit unsigned 
                      //   bit pattern of 0xFFFFFFFF is assigned, giving f 
                      //   the value of 0xFFFFFFFF.
int  g = 3000000000;  // a signed decimal literal taking 32 bits,
                      //   setting the sign bit, g gets -1294967296
int  h = 0xA0000000;  // okay, 32-bit signed hexadecimal
int  i = 5000000000;  // ERROR: needs more than 32 bits
int  j = 0xFFFFFFFFF; // ERROR: needs more that 32 bits
int  k = 0x80000000;  // k gets -2147483648 == 0x80000000
int  l = 2147483648;  // l gets -2147483648 (the literal set the sign bit)

float fa, fb = 1.5;     // single-precision floating-point
double fc, fd = 2.0LF;  // double-precision floating-point

vec2 texcoord1, texcoord2;
vec3 position;
vec4 myRGBA;
ivec2 textureLookup;
bvec3 less;

mat2 mat2D;
mat3 optMatrix;
mat4 view, projection;
mat4x4 view;  // an alternate way of declaring a mat4
mat3x2 m;     // a matrix with 3 columns and 2 rows
dmat4 highPrecisionMVP;
dmat2x4 dm;

struct light {
    float intensity;
    vec3 position;
} lightVar;

struct S { float f; };

struct T {
	//S;              // Error: anonymous structures disallowed
	//struct { ... }; // Error: embedded structures disallowed
	S s;            // Okay: nested structures with name are allowed
};

float frequencies[3];	
uniform vec4 lightPosition[4];
light lights[];
const int numLights = 2;
light lights[numLights];

in vec3 normal;
centroid in vec2 TexCoord;
invariant centroid in vec4 Color;
noperspective in float temperature;
flat in vec3 myColor;
noperspective centroid in vec2 myTexCoord;

uniform vec4 lightPosition;
uniform vec3 color = vec3(0.7, 0.7, 0.2);  // value assigned at link time

in Material {
    smooth in vec4 Color1; // legal, input inside in block
    smooth vec4 Color2;    // legal, 'in' inherited from 'in Material'
    vec2 TexCoordA;        // legal, TexCoord is an input
    uniform float Atten;   // illegal, mismatched  storage qualifier

};

in Light {
    vec4 LightPos;
    vec3 LightColor;
};
in ColoredTexture {
    vec4 Color;
    vec2 TexCoord;        
} Materiala;           // instance name
vec3 Color;            // different Color than Material.Color

in vec4 gl_FragCoord;     // redeclaration that changes nothing is allowed

// All the following are allowed redeclaration that change behavior
layout(origin_upper_left) in vec4 gl_FragCoord;
layout(pixel_center_integer) in vec4 gl_FragCoord;
layout(origin_upper_left, pixel_center_integer) in vec4 gl_FragCoord;

layout(early_fragment_tests) in;

// compute shader:
layout (local_size_x = 32, local_size_y = 32) in;
layout (local_size_x = 8) in;

layout(location = 3) out vec4 color;
layout(location = 3, index = 1) out vec4 factor;
layout(location = 2) out vec4 colors[3];

layout (depth_greater) out float gl_FragDepth;

// redeclaration that changes nothing is allowed
out float gl_FragDepth;

// assume it may be modified in any way
layout (depth_any) out float gl_FragDepth;

// assume it may be modified such that its value will only increase
layout (depth_greater) out float gl_FragDepth;

// assume it may be modified such that its value will only decrease
layout (depth_less) out float gl_FragDepth;

// assume it will not be modified
layout (depth_unchanged) out float gl_FragDepth;

in vec4 gl_Color;             // predeclared by the fragment language
flat  in vec4 gl_Color;       // redeclared by user to be flat


float[5] foo(float[5]) 
{
    return float[5](3.4, 4.2, 5.0, 5.2, 1.1);
}

precision highp float;
precision highp int;
precision mediump int;
precision highp float;

void main()
{
    {
		float a[5] = float[5](3.4, 4.2, 5.0, 5.2, 1.1);
	}
	{
		float a[5] = float[](3.4, 4.2, 5.0, 5.2, 1.1);  // same thing
	}
    {
	    vec4 a[3][2];  // size-3 array of size-2 array of vec4
		vec4[2] a1[3];  // size-3 array of size-2 array of vec4
		vec4[3][2] a2;  // size-3 array of size-2 array of vec4
		vec4 b[2] = vec4[2](vec4(0.0), vec4(0.1));
		vec4[3][2] a3 = vec4[3][2](b, b, b);        // constructor
		void foo(vec4[3][2]);  // prototype with unnamed parameter
		vec4 a4[3][2] = {vec4[2](vec4(0.0), vec4(1.0)),   
						 vec4[2](vec4(0.0), vec4(1.0)),   
						 vec4[2](vec4(0.0), vec4(1.0)) };
    }
	{
		float a[5];
		{
			float b[] = a;  // b is explicitly size 5
		}
		{
			float b[5] = a; // means the same thing
		}
		{
			float b[] = float[](1,2,3,4,5);  // also explicitly sizes to 5
		}
		a.length();  // returns 5 
	}
    {
		vec4 a[3][2];
		a.length();     // this is 3
		a[x].length();  // this is 2
    }
	// for an array b containing a member array a:
	b[++x].a.length();    // b is never dereferenced, but “++x” is evaluated

	// for an array s of a shader storage object containing a member array a:
	s[x].a.length();      // s is dereferenced; x needs to be a valid index
	//
	//All of the following declarations result in a compile-time error.
	//float a[2] = { 3.4, 4.2, 5.0 };         // illegal
	//vec2 b = { 1.0, 2.0, 3.0 };             // illegal
	//mat3x3 c = { vec3(0.0), vec3(1.0), vec3(2.0), vec3(3.0) };    // illegal
	//mat2x2 d = { 1.0, 0.0, 0.0, 1.0 };      // illegal, can't flatten nesting
	//struct {
	//	float a;
	//	int   b;
	//} e = { 1.2, 2, 3 };                    // illegal

    struct {
        float a;
        int   b;
    } e = { 1.2, 2 };             // legal, all types match

    struct {
        float a;
        int   b;
    } e = { 1, 3 };               // legal, first initializer is converted

    //All of the following declarations result in a compile-time error.
    //int a = true;                           // illegal
    //vec4 b[2] = { vec4(0.0), 1.0 };         // illegal
    //mat4x2 c = { vec3(0.0), vec3(1.0) };    // illegal

    //struct S1 {
    //    vec4 a;
    //    vec4 b;
    //};

    //struct {
    //    float s;
    //    float t;
    //} d[] = { S1(vec4(0.0), vec4(1.1)) };   // illegal

    {
        float a[] = float[](3.4, 4.2, 5.0, 5.2, 1.1);
        float b[] = { 3.4, 4.2, 5.0, 5.2, 1.1 };
        float c[] = a;                          // c is explicitly size 5
        float d[5] = b;                         // means the same thing
    }
    {
        const vec3 zAxis = vec3 (0.0, 0.0, 1.0);
        const float ceiling = a + b; // a and b not necessarily constants
    }
    {
        in vec4 position;
        in vec3 normal;
        in vec2 texCoord[4];
    }
    {
        lowp float color;
        out mediump vec2 P;
        lowp ivec2 foo(lowp mat3);
        highp mat4 m;
    }

}
