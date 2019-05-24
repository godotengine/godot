#version 400

uniform float u;

int foo(int a, const int b, in int c, const in int d, out int e, inout int f)
{
    int sum = a + b + c + d + f; // no e, it is out only
	// sum should be 47 now

	a *= 64;
	// no b, it is read only
	c *= 64;
	// no d, it is read only
	e = 64 * 16; // e starts undefined
	f *= 64;

	sum += a + 64 * b + c + 64 * d + e + f; // everything has a value now, totaling of 64(1+2+4+8+16+32) = 64*63 = 4032
	// sum should be 4032 + 47  = 4079
	
	return sum;
}

int foo2(float a, vec3 b, out int r)
{
    r = int(3.0 * a);
    return int(5.0 * b.y);
}

int foo3()
{
    if (u > 3.2) {
        discard;
        return 1000000;
    }

    return 2000000;
}

void main()
{
    int e;
	int t = 2;
	struct s {
	    ivec4 t;
	} f;
	f.t.y = 32;

    // test the different qualifers
    int color = foo(1, 2, t+t, 8, e, f.t.y);

	color += 128 * (e + f.t.y); // right side should be 128(64(16 + 32)) = 393216
	// sum should be 4079 + 393216 = 397295
    
    // test conversions
    float arg;
    float ret;
    ret = foo2(4, ivec3(1,2,3), arg);  // ret = 10, param = 12.0
    color += int(ret + arg); // adds 22, for total of 397317

    color += foo3();         // theoretically, add 2000000, for total of 2397317

    gl_FragColor = vec4(color);
}

vec3 m(vec2);
void aggCall()
{
    float F;
    m(ivec2(F));  // test input conversion of single argument that's an aggregate; other function tests in 120.vert
}

vec4 badConv()
{
    return u;     // ERROR, can change scalar to vector
}