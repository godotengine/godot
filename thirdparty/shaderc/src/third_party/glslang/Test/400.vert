#version 400 core

in double d;   // ERROR, no doubles
in dvec3 d3;   // ERROR, no doubles
in dmat4 dm4;  // ERROR, no doubles

// function selection under type conversion
void foo1(double a, uint b)  {}
void foo1(double a, int b)   {}
void foo1(double a, float b) {}
void foo1(double a, double b){}

void foo2(double a, float b) {}
void foo2(double a, double b){}

void foo3(double a, float b) {}
void foo3(float a, double b) {}

void ftd(  int,  float, double) {}
void ftd( uint,  float, double) {}
void ftd(float, double, double) {}

void main()
{
    double d;
	uint u;
	int i;
	float f;

	foo1(d, d);
	foo1(d, u);
	foo1(d, i);
	foo1(d, f);

	foo1(f, d);
	foo1(f, u);
	foo1(f, i);
	foo1(f, f);

	foo1(u, d);
	foo1(u, u);
	foo1(u, i);
	foo1(u, f);

	foo1(i, d);
	foo1(i, u);
	foo1(i, i);
	foo1(i, f);

	foo2(d, d);
	foo2(d, u);
	foo2(d, i);
	foo2(d, f);

	foo2(f, d);
	foo2(f, u);
	foo2(f, i);
	foo2(f, f);

	foo2(u, d);
	foo2(u, u);
	foo2(u, i);
	foo2(u, f);

	foo2(i, d);
	foo2(i, u);
	foo2(i, i);
	foo2(i, f);

	foo3(d, d);  // ERROR, no match
	foo3(d, u);
	foo3(d, i);
	foo3(d, f);

	foo3(f, d);
	foo3(f, u); // ERROR, ambiguous
	foo3(f, i); // ERROR, ambiguous
	foo3(f, f); // ERROR, ambiguous

	foo3(u, d);
	foo3(u, u); // ERROR, ambiguous
	foo3(u, i); // ERROR, ambiguous
	foo3(u, f); // ERROR, ambiguous

	foo3(i, d);
	foo3(i, u); // ERROR, ambiguous
	foo3(i, i); // ERROR, ambiguous
	foo3(i, f); // ERROR, ambiguous

	ftd(i, f, f);
	ftd(u, f, f);
}

void itf(int, float, int);
void itf(int, double, int);

void tf()
{
    double d;
	uint u;
	int i;
	float f;
	
	itf(i, i, i);
	itf(i, u, i);
}
