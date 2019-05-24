#version 400

uniform sampler2D samp2D;
in vec2 coord;
in vec4 color;

struct s1 {
    int i;
    float f;
};

struct s2 {
    int i;
    float f;
	s1 s1_1;
	vec4 bleh;
};

struct s3 {
	s2 s2_1;
    int i;
    float f;
	s1 s1_1;
};


flat in s1 foo;
flat in s2 foo2;
flat in s3 foo3;


flat in int condition;

void main()
{
	s2 locals2;
	s3 locals3;
	float localFArray[16];
	int localIArray[8];

	locals2 = foo3.s2_1;

	if (foo3.s2_1.i > 0) {
		locals2.s1_1.f = 1.0;
		localFArray[4] = coord.x;
		localIArray[2] = foo3.s2_1.i;
	} else {
		locals2.s1_1.f = coord.x;
		localFArray[4] = 1.0;
		localIArray[2] = 0;
	}

	if (localIArray[2] == 0)
		++localFArray[4];

 	float localArray[16];
	int x = 5;
	localArray[x] = coord.x;

	float[16] a;

	for (int i = 0; i < 16; i++)
		a[i] = 0.0;
	
	if (condition == 1)
		a = localArray;
	
	locals2.bleh = color;
	locals2.bleh.z = coord.y;

	gl_FragColor = locals2.bleh * (localFArray[4] + locals2.s1_1.f + localArray[x] + a[x]) * texture(samp2D, coord);
}
