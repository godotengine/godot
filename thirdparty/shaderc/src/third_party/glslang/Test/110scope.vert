#version 110

int f(int a, int b, int c)
{
	int a = b;  // ERROR, redefinition

    {
		float a = float(a) + 1.0; // okay
    }

	return a;
}

int f(int a, int b, int c);  // okay to redeclare

bool b;
float b(int a);      // okay, b and b() are different

float c(int a);
bool c;              // okay, and c() are different

float f;             // okay f and f() are different
float tan;           // okay, hides built-in function
float sin(float x);  // okay, can redefine built-in functions
float cos(float x)   // okay, can redefine built-in functions
{
	return 1.0;
}
bool radians(bool x) // okay, can overload built-in functions
{
    return true;
}

int gi = f(1,2,3);  // ERROR, can't call user-defined function from global scope

void main()
{
    int g();    // okay
    g();

    float sin; // okay
    sin;
    sin(0.7);  // okay
    f(1,2,3);

    float f;
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
    degrees(3.2);

    {
        S s;
        s.x = 3;
        struct S {   // okay, hides S
            bool b;
        };
        S t;
        t.b = true;
        struct S {    // ERROR, redefinition of struct S
            float f;
        };
    }
}
