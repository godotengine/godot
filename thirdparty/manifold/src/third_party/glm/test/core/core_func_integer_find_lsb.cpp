#include <glm/glm.hpp>
#include <cstdio>
#include <cstdlib>     //To define "exit", req'd by XLC.
#include <ctime>

int nlz(unsigned x)
{
	int pop(unsigned x);

	x = x | (x >> 1);
	x = x | (x >> 2);
	x = x | (x >> 4);
	x = x | (x >> 8);
	x = x | (x >>16);
	return pop(~x);
}

int pop(unsigned x)
{
	x = x - ((x >> 1) & 0x55555555);
	x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
	x = (x + (x >> 4)) & 0x0F0F0F0F;
	x = x + (x << 8);
	x = x + (x << 16);
	return x >> 24;
}

int ntz1(unsigned x)
{
	return 32 - nlz(~x & (x-1));
}

int ntz2(unsigned x)
{
	return pop(~x & (x - 1));
}

int ntz3(unsigned x)
{
	int n;

	if (x == 0) return(32);
	n = 1;
	if ((x & 0x0000FFFF) == 0) {n = n +16; x = x >>16;}
	if ((x & 0x000000FF) == 0) {n = n + 8; x = x >> 8;}
	if ((x & 0x0000000F) == 0) {n = n + 4; x = x >> 4;}
	if ((x & 0x00000003) == 0) {n = n + 2; x = x >> 2;}
	return n - (x & 1);
}

int ntz4(unsigned x)
{
	unsigned y;
	int n;

	if (x == 0) return 32;
	n = 31;
	y = x <<16;  if (y != 0) {n = n -16;  x = y;}
	y = x << 8;  if (y != 0) {n = n - 8;  x = y;}
	y = x << 4;  if (y != 0) {n = n - 4;  x = y;}
	y = x << 2;  if (y != 0) {n = n - 2;  x = y;}
	y = x << 1;  if (y != 0) {n = n - 1;}
	return n;
}

int ntz4a(unsigned x)
{
	unsigned y;
	int n;

	if (x == 0) return 32;
	n = 31;
	y = x <<16;  if (y != 0) {n = n -16;  x = y;}
	y = x << 8;  if (y != 0) {n = n - 8;  x = y;}
	y = x << 4;  if (y != 0) {n = n - 4;  x = y;}
	y = x << 2;  if (y != 0) {n = n - 2;  x = y;}
	n = n - ((x << 1) >> 31);
	return n;
}

int ntz5(char x)
{
	if (x & 15) {
		if (x & 3) {
			if (x & 1) return 0;
			else return 1;
		}
		else if (x & 4) return 2;
		else return 3;
	}
	else if (x & 0x30) {
		if (x & 0x10) return 4;
		else return 5;
	}
	else if (x & 0x40) return 6;
	else if (x) return 7;
	else return 8;
}

int ntz6(unsigned x)
{
	int n;

	x = ~x & (x - 1);
	n = 0;				// n = 32;
	while(x != 0)
	{					// while (x != 0) {
		n = n + 1;		//    n = n - 1;
		x = x >> 1;		//    x = x + x;
	}					// }
	return n;			// return n;
}

int ntz6a(unsigned x)
{
	int n = 32;

	while (x != 0) {
		n = n - 1;
		x = x + x;
	}
	return n;
}

/* Dean Gaudet's algorithm. To be most useful there must be a good way
to evaluate the C "conditional expression" (a?b:c construction) without
branching. The result of a?b:c is b if a is true (nonzero), and c if a
is false (0).
   For example, a compare to zero op that sets a target GPR to 1 if the
operand is 0, and to 0 if the operand is nonzero, will do it. With this
instruction, the algorithm is entirely branch-free. But the most
interesting thing about it is the high degree of parallelism. All six
lines with conditional expressions can be executed in parallel (on a
machine with sufficient computational units).
   Although the instruction count is 30 measured statically, it could
execute in only 10 cycles on a machine with sufficient parallelism.
   The first two uses of y can instead be x, which would increase the
useful parallelism on most machines (the assignments to y, bz, and b4
could then all run in parallel). */

int ntz7(unsigned x)
{
	unsigned y, bz, b4, b3, b2, b1, b0;

	y = x & -x;               // Isolate rightmost 1-bit.
	bz = y ? 0 : 1;           // 1 if y = 0.
	b4 = (y & 0x0000FFFF) ? 0 : 16;
	b3 = (y & 0x00FF00FF) ? 0 : 8;
	b2 = (y & 0x0F0F0F0F) ? 0 : 4;
	b1 = (y & 0x33333333) ? 0 : 2;
	b0 = (y & 0x55555555) ? 0 : 1;
	return bz + b4 + b3 + b2 + b1 + b0;
}

// This file has divisions by zero to test isnan
#if GLM_COMPILER & GLM_COMPILER_VC
#	pragma warning(disable : 4800)
#endif

int ntz7_christophe(unsigned x)
{
	unsigned y, bz, b4, b3, b2, b1, b0;

	y = x & -x;               // Isolate rightmost 1-bit.
	bz = unsigned(!bool(y));           // 1 if y = 0.
	b4 = unsigned(!bool(y & 0x0000FFFF)) * 16;
	b3 = unsigned(!bool(y & 0x00FF00FF)) * 8;
	b2 = unsigned(!bool(y & 0x0F0F0F0F)) * 4;
	b1 = unsigned(!bool(y & 0x33333333)) * 2;
	b0 = unsigned(!bool(y & 0x55555555)) * 1;
	return bz + b4 + b3 + b2 + b1 + b0;
}

/* Below is David Seal's algorithm, found at
http://www.ciphersbyritter.com/NEWS4/BITCT.HTM Table
entries marked "u" are unused. 6 ops including a
multiply, plus an indexed load. */

#define u 99
int ntz8(unsigned x)
{
	static char table[64] =
		{32, 0, 1,12, 2, 6, u,13,   3, u, 7, u, u, u, u,14,
		10, 4, u, u, 8, u, u,25,   u, u, u, u, u,21,27,15,
		31,11, 5, u, u, u, u, u,   9, u, u,24, u, u,20,26,
		30, u, u, u, u,23, u,19,  29, u,22,18,28,17,16, u};

	x = (x & -x)*0x0450FBAF;
	return table[x >> 26];
}

/* Seal's algorithm with multiply expanded.
9 elementary ops plus an indexed load. */

int ntz8a(unsigned x)
{
	static char table[64] =
		{32, 0, 1,12, 2, 6, u,13,   3, u, 7, u, u, u, u,14,
		10, 4, u, u, 8, u, u,25,   u, u, u, u, u,21,27,15,
		31,11, 5, u, u, u, u, u,   9, u, u,24, u, u,20,26,
		30, u, u, u, u,23, u,19,  29, u,22,18,28,17,16, u};

	x = (x & -x);
	x = (x << 4) + x;    // x = x*17.
	x = (x << 6) + x;    // x = x*65.
	x = (x << 16) - x;   // x = x*65535.
	return table[x >> 26];
}

/* Reiser's algorithm. Three ops including a "remainder,"
plus an indexed load. */

int ntz9(unsigned x)
{
	static char table[37] = {
		32,  0,  1, 26,  2, 23, 27,
		u,  3, 16, 24, 30, 28, 11,  u, 13,  4,
		7, 17,  u, 25, 22, 31, 15, 29, 10, 12,
		6,  u, 21, 14,  9,  5, 20,  8, 19, 18};

	x = (x & -x)%37;
	return table[x];
}

/* Using a de Bruijn sequence. This is a table lookup with a 32-entry
table. The de Bruijn sequence used here is
                0000 0100 1101 0111 0110 0101 0001 1111,
obtained from Danny Dube's October 3, 1997, posting in
comp.compression.research. Thanks to Norbert Juffa for this reference. */

int ntz10(unsigned x) {

   static char table[32] =
     { 0, 1, 2,24, 3,19, 6,25,  22, 4,20,10,16, 7,12,26,
      31,23,18, 5,21, 9,15,11,  30,17, 8,14,29,13,28,27};

   if (x == 0) return 32;
   x = (x & -x)*0x04D7651F;
   return table[x >> 27];
}

/* Norbert Juffa's code, answer to exercise 1 of Chapter 5 (2nd ed). */

#define SLOW_MUL
int ntz11 (unsigned int n) {

   static unsigned char tab[32] =
   {   0,  1,  2, 24,  3, 19, 6,  25,
      22,  4, 20, 10, 16,  7, 12, 26,
      31, 23, 18,  5, 21,  9, 15, 11,
      30, 17,  8, 14, 29, 13, 28, 27
   };
   unsigned int k;
   n = n & (-n);        /* isolate lsb */
   printf("n = %d\n", n);
#if defined(SLOW_MUL)
   k = (n << 11) - n;
   k = (k <<  2) + k;
   k = (k <<  8) + n;
   k = (k <<  5) - k;
#else
   k = n * 0x4d7651f;
#endif
   return n ? tab[k>>27] : 32;
}

int errors;
void error(int x, int y) {
   errors = errors + 1;
   std::printf("Error for x = %08x, got %d\n", x, y);
}

int main()
{
#	ifdef NDEBUG

	int i, m, n;
	static unsigned test[] = {0,32, 1,0, 2,1, 3,0, 4,2, 5,0, 6,1,  7,0,
		8,3, 9,0, 16,4, 32,5, 64,6, 128,7, 255,0, 256,8, 512,9, 1024,10,
		2048,11, 4096,12, 8192,13, 16384,14, 32768,15, 65536,16,
		0x20000,17, 0x40000,18, 0x80000,19, 0x100000,20, 0x200000,21,
		0x400000,22, 0x800000,23, 0x1000000,24, 0x2000000,25,
		0x4000000,26, 0x8000000,27, 0x10000000,28, 0x20000000,29,
		0x40000000,30, 0x80000000,31, 0xFFFFFFF0,4, 0x3000FF00,8,
		0xC0000000,30, 0x60000000,29, 0x00011000, 12};

	std::size_t const Count = 1000;

	n = sizeof(test)/4;

	std::clock_t TimestampBeg = 0;
	std::clock_t TimestampEnd = 0;

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (ntz1(test[i]) != test[i+1]) error(test[i], ntz1(test[i]));}
	TimestampEnd = std::clock();

	std::printf("ntz1: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (ntz2(test[i]) != test[i+1]) error(test[i], ntz2(test[i]));}
	TimestampEnd = std::clock();

	std::printf("ntz2: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (ntz3(test[i]) != test[i+1]) error(test[i], ntz3(test[i]));}
	TimestampEnd = std::clock();

	std::printf("ntz3: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (ntz4(test[i]) != test[i+1]) error(test[i], ntz4(test[i]));}
	TimestampEnd = std::clock();

	std::printf("ntz4: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (ntz4a(test[i]) != test[i+1]) error(test[i], ntz4a(test[i]));}
	TimestampEnd = std::clock();

	std::printf("ntz4a: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for(std::size_t k = 0; k < Count; ++k)
	for(i = 0; i < n; i += 2)
	{
		m = test[i+1];
		if(m > 8)
			m = 8;
		if(ntz5(static_cast<char>(test[i])) != m)
			error(test[i], ntz5(static_cast<char>(test[i])));
	}
	TimestampEnd = std::clock();

	std::printf("ntz5: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (ntz6(test[i]) != test[i+1]) error(test[i], ntz6(test[i]));}
	TimestampEnd = std::clock();

	std::printf("ntz6: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (ntz6a(test[i]) != test[i+1]) error(test[i], ntz6a(test[i]));}
	TimestampEnd = std::clock();

	std::printf("ntz6a: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (ntz7(test[i]) != test[i+1]) error(test[i], ntz7(test[i]));}
	TimestampEnd = std::clock();

	std::printf("ntz7: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (ntz7_christophe(test[i]) != test[i+1]) error(test[i], ntz7(test[i]));}
	TimestampEnd = std::clock();

	std::printf("ntz7_christophe: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (ntz8(test[i]) != test[i+1]) error(test[i], ntz8(test[i]));}
	TimestampEnd = std::clock();

	std::printf("ntz8: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (ntz8a(test[i]) != test[i+1]) error(test[i], ntz8a(test[i]));}
	TimestampEnd = std::clock();

	std::printf("ntz8a: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (ntz9(test[i]) != test[i+1]) error(test[i], ntz9(test[i]));}
	TimestampEnd = std::clock();

	std::printf("ntz9: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (ntz10(test[i]) != test[i+1]) error(test[i], ntz10(test[i]));}
	TimestampEnd = std::clock();

	std::printf("ntz10: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	if (errors == 0)
		std::printf("Passed all %d cases.\n", static_cast<int>(sizeof(test)/8));

#	endif//NDEBUG
}
