#version 100

int f(int a, int b, int c)
{
	int a = b;  // ERROR, redefinition

    {
		float a = float(a) + 1.0;
    }

	return a;
}

int f(int a, int b, int c);  // okay to redeclare

bool b;
float b(int a);      // ERROR: redefinition

float c(int a);
bool c;              // ERROR: redefinition

float f;             // ERROR: redefinition
float tan;           // okay, built-in is in an outer scope
float sin(float x);  // ERROR: can't redefine built-in functions
float cos(float x)   // ERROR: can't redefine built-in functions
{
	return 1.0;
}
bool radians(bool x) // okay, can overload built-in functions
{
    return true;
}

invariant gl_Position;

void main()
{
    int g();    // ERROR: no local function declarations
	g();

    float sin;  // okay
	sin;
    sin(0.7);  // ERROR, use of hidden function
    f(1,2,3);

    float f;    // hides f()
    f = 3.0;

    gl_Position = vec4(f);

    for (int f = 0; f < 10; ++f)
        ++f;

    int x = 1;
    { 
        float x = 2.0, /* 2nd x visible here */ y = x; // y is initialized to 2
        int z = z; // ERROR: z not previously defined.
    }
    {
        int x = x; // x is initialized to '1'
    }

    struct S 
    { 
        int x; 
    };
    {
        S S = S(0); // 'S' is only visible as a struct and constructor 
        S.x;        // 'S' is now visible as a variable
    }

    int degrees;
    degrees(3.2);  // ERROR, use of hidden built-in function
}

varying struct SSS { float f; } s; // ERROR
