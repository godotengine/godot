#include <glm/glm.hpp>
#include <cstdio>
#include <cstdlib>     // To define "exit", req'd by XLC.
#include <ctime>

#define LE 1            // 1 for little-endian, 0 for big-endian.

int pop(unsigned x) {
   x = x - ((x >> 1) & 0x55555555);
   x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
   x = (x + (x >> 4)) & 0x0F0F0F0F;
   x = x + (x << 8);
   x = x + (x << 16);
   return x >> 24;
}

int nlz1(unsigned x) {
   int n;

   if (x == 0) return(32);
   n = 0;
   if (x <= 0x0000FFFF) {n = n +16; x = x <<16;}
   if (x <= 0x00FFFFFF) {n = n + 8; x = x << 8;}
   if (x <= 0x0FFFFFFF) {n = n + 4; x = x << 4;}
   if (x <= 0x3FFFFFFF) {n = n + 2; x = x << 2;}
   if (x <= 0x7FFFFFFF) {n = n + 1;}
   return n;
}

int nlz1a(unsigned x) {
   int n;

/* if (x == 0) return(32); */
   if (static_cast<int>(x) <= 0) return (~x >> 26) & 32;
   n = 1;
   if ((x >> 16) == 0) {n = n +16; x = x <<16;}
   if ((x >> 24) == 0) {n = n + 8; x = x << 8;}
   if ((x >> 28) == 0) {n = n + 4; x = x << 4;}
   if ((x >> 30) == 0) {n = n + 2; x = x << 2;}
   n = n - (x >> 31);
   return n;
}
// On basic Risc, 12 to 20 instructions.

int nlz2(unsigned x) {
   unsigned y;
   int n;

   n = 32;
   y = x >>16;  if (y != 0) {n = n -16;  x = y;}
   y = x >> 8;  if (y != 0) {n = n - 8;  x = y;}
   y = x >> 4;  if (y != 0) {n = n - 4;  x = y;}
   y = x >> 2;  if (y != 0) {n = n - 2;  x = y;}
   y = x >> 1;  if (y != 0) return n - 2;
   return n - x;
}

// As above but coded as a loop for compactness:
// 23 to 33 basic Risc instructions.
int nlz2a(unsigned x) {
   unsigned y;
   int n, c;

   n = 32;
   c = 16;
   do {
      y = x >> c;  if (y != 0) {n = n - c;  x = y;}
      c = c >> 1;
   } while (c != 0);
   return n - x;
}

int nlz3(int x) {
   int y, n;

   n = 0;
   y = x;
L: if (x < 0) return n;
   if (y == 0) return 32 - n;
   n = n + 1;
   x = x << 1;
   y = y >> 1;
   goto L;
}

int nlz4(unsigned x) {
   int y, m, n;

   y = -(x >> 16);      // If left half of x is 0,
   m = (y >> 16) & 16;  // set n = 16.  If left half
   n = 16 - m;          // is nonzero, set n = 0 and
   x = x >> m;          // shift x right 16.
                        // Now x is of the form 0000xxxx.
   y = x - 0x100;       // If positions 8-15 are 0,
   m = (y >> 16) & 8;   // add 8 to n and shift x left 8.
   n = n + m;
   x = x << m;

   y = x - 0x1000;      // If positions 12-15 are 0,
   m = (y >> 16) & 4;   // add 4 to n and shift x left 4.
   n = n + m;
   x = x << m;

   y = x - 0x4000;      // If positions 14-15 are 0,
   m = (y >> 16) & 2;   // add 2 to n and shift x left 2.
   n = n + m;
   x = x << m;

   y = x >> 14;         // Set y = 0, 1, 2, or 3.
   m = y & ~(y >> 1);   // Set m = 0, 1, 2, or 2 resp.
   return n + 2 - m;
}

int nlz5(unsigned x) {
   int pop(unsigned x);

   x = x | (x >> 1);
   x = x | (x >> 2);
   x = x | (x >> 4);
   x = x | (x >> 8);
   x = x | (x >>16);
   return pop(~x);
}

/* The four programs below are not valid ANSI C programs.  This is
because they refer to the same storage locations as two different types.
However, they work with xlc/AIX, gcc/AIX, and gcc/NT.  If you try to
code them more compactly by declaring a variable xx to be "double," and
then using

   n = 1054 - (*((unsigned *)&xx + LE) >> 20);

then you are violating not only the rule above, but also the ANSI C
rule that pointer arithmetic can be performed only on pointers to
array elements.
   When coded with the above statement, the program fails with xlc,
gcc/AIX, and gcc/NT, at some optimization levels.
   BTW, these programs use the "anonymous union" feature of C++, not
available in C. */

int nlz6(unsigned k)
{
	union {
		unsigned asInt[2];
		double asDouble;
	};
	int n;

	asDouble = static_cast<double>(k) + 0.5;
	n = 1054 - (asInt[LE] >> 20);
	return n;
}

int nlz7(unsigned k)
{
	union {
		unsigned asInt[2];
		double asDouble;
	};
	int n;

	asDouble = static_cast<double>(k);
	n = 1054 - (asInt[LE] >> 20);
	n = (n & 31) + (n >> 9);
	return n;
}

   /* In single qualifier, round-to-nearest mode, the basic method fails for:
   k = 0, k = 01FFFFFF, 03FFFFFE <= k <= 03FFFFFF,
                        07FFFFFC <= k <= 07FFFFFF,
                        0FFFFFF8 <= k <= 0FFFFFFF,
                                   ...
                        7FFFFFC0 <= k <= 7FFFFFFF.
                        FFFFFF80 <= k <= FFFFFFFF.
   For k = 0 it gives 158, and for the other values it is too low by 1. */

int nlz8(unsigned k)
{
	union {
		unsigned asInt;
		float asFloat;
	};
	int n;

	k = k & ~(k >> 1);           /* Fix problem with rounding. */
	asFloat = static_cast<float>(k) + 0.5f;
	n = 158 - (asInt >> 23);
	return n;
}

/* The example below shows how to make a macro for nlz.  It uses an
extension to the C and C++ languages that is provided by the GNU C/C++
compiler, namely, that of allowing statements and declarations in
expressions (see "Using and Porting GNU CC", by Richard M. Stallman
(1998).  The underscores are necessary to protect against the
possibility that the macro argument will conflict with one of its local
variables, e.g., NLZ(k). */

int nlz9(unsigned k)
{
	union {
		unsigned asInt;
		float asFloat;
	};
	int n;

	k = k & ~(k >> 1);           /* Fix problem with rounding. */
	asFloat = static_cast<float>(k);
	n = 158 - (asInt >> 23);
	n = (n & 31) + (n >> 6);     /* Fix problem with k = 0. */
	return n;
}

/* Below are three nearly equivalent programs for computing the number
of leading zeros in a word. This material is not in HD, but may be in a
future edition.
   Immediately below is Robert Harley's algorithm, found at the
comp.arch newsgroup entry dated 7/12/96, pointed out to me by Norbert
Juffa.
   Table entries marked "u" are unused. 14 ops including a multiply,
plus an indexed load.
   The smallest multiplier that works is 0x045BCED1 = 17*65*129*513 (all
of form 2**k + 1). There are no multipliers of three terms of the form
2**k +- 1 that work, with a table size of 64 or 128. There are some,
with a table size of 64, if you precede the multiplication with x = x -
(x >> 1), but that seems less elegant. There are also some if you use a
table size of 256, the smallest is 0x01033CBF = 65*255*1025 (this would
save two instructions in the form of this algorithm with the
multiplication expanded into shifts and adds, but the table size is
getting a bit large). */

#define u 99
int nlz10(unsigned x)
{
	static char table[64] =
		{32,31, u,16, u,30, 3, u,  15, u, u, u,29,10, 2, u,
		u, u,12,14,21, u,19, u,   u,28, u,25, u, 9, 1, u,
		17, u, 4, u, u, u,11, u,  13,22,20, u,26, u, u,18,
		5, u, u,23, u,27, u, 6,   u,24, 7, u, 8, u, 0, u};

	x = x | (x >> 1);		// Propagate leftmost
	x = x | (x >> 2);		// 1-bit to the right.
	x = x | (x >> 4);
	x = x | (x >> 8);
	x = x | (x >>16);
	x = x*0x06EB14F9;		// Multiplier is 7*255**3.
	return table[x >> 26];
}

/* Harley's algorithm with multiply expanded.
19 elementary ops plus an indexed load. */

int nlz10a(unsigned x)
{
	static char table[64] =
		{32,31, u,16, u,30, 3, u,  15, u, u, u,29,10, 2, u,
		u, u,12,14,21, u,19, u,   u,28, u,25, u, 9, 1, u,
		17, u, 4, u, u, u,11, u,  13,22,20, u,26, u, u,18,
		5, u, u,23, u,27, u, 6,   u,24, 7, u, 8, u, 0, u};

	x = x | (x >> 1);    // Propagate leftmost
	x = x | (x >> 2);    // 1-bit to the right.
	x = x | (x >> 4);
	x = x | (x >> 8);
	x = x | (x >> 16);
	x = (x << 3) - x;    // Multiply by 7.
	x = (x << 8) - x;    // Multiply by 255.
	x = (x << 8) - x;    // Again.
	x = (x << 8) - x;    // Again.
	return table[x >> 26];
}

/* Julius Goryavsky's version of Harley's algorithm.
17 elementary ops plus an indexed load, if the machine
has "and not." */

int nlz10b(unsigned x)
{
	static char table[64] =
		{32,20,19, u, u,18, u, 7,  10,17, u, u,14, u, 6, u,
		u, 9, u,16, u, u, 1,26,   u,13, u, u,24, 5, u, u,
		u,21, u, 8,11, u,15, u,   u, u, u, 2,27, 0,25, u,
		22, u,12, u, u, 3,28, u,  23, u, 4,29, u, u,30,31};

	x = x | (x >> 1);    // Propagate leftmost
	x = x | (x >> 2);    // 1-bit to the right.
	x = x | (x >> 4);
	x = x | (x >> 8);
	x = x & ~(x >> 16);
	x = x*0xFD7049FF;    // Activate this line or the following 3.
	// x = (x << 9) - x;    // Multiply by 511.
	// x = (x << 11) - x;   // Multiply by 2047.
	// x = (x << 14) - x;   // Multiply by 16383.
	return table[x >> 26];
}

int errors;
void error(int x, int y)
{
	errors = errors + 1;
	std::printf("Error for x = %08x, got %d\n", x, y);
}

int main()
{
#	ifdef NDEBUG

	int i, n;
	static unsigned test[] = {0,32, 1,31, 2,30, 3,30, 4,29, 5,29, 6,29,
		7,29, 8,28, 9,28, 16,27, 32,26, 64,25, 128,24, 255,24, 256,23,
		512,22, 1024,21, 2048,20, 4096,19, 8192,18, 16384,17, 32768,16,
		65536,15, 0x20000,14, 0x40000,13, 0x80000,12, 0x100000,11,
		0x200000,10, 0x400000,9, 0x800000,8, 0x1000000,7, 0x2000000,6,
		0x4000000,5, 0x8000000,4, 0x0FFFFFFF,4, 0x10000000,3,
		0x3000FFFF,2, 0x50003333,1, 0x7FFFFFFF,1, 0x80000000,0,
		0xFFFFFFFF,0};
	std::size_t const Count = 1000;

	n = sizeof(test)/4;

	std::clock_t TimestampBeg = 0;
	std::clock_t TimestampEnd = 0;

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (nlz1(test[i]) != test[i+1]) error(test[i], nlz1(test[i]));}
	TimestampEnd = std::clock();

	std::printf("nlz1: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (nlz1a(test[i]) != test[i+1]) error(test[i], nlz1a(test[i]));}
	TimestampEnd = std::clock();

	std::printf("nlz1a: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (nlz2(test[i]) != test[i+1]) error(test[i], nlz2(test[i]));}
	TimestampEnd = std::clock();

	std::printf("nlz2: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (nlz2a(test[i]) != test[i+1]) error(test[i], nlz2a(test[i]));}
	TimestampEnd = std::clock();

	std::printf("nlz2a: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (nlz3(test[i]) != test[i+1]) error(test[i], nlz3(test[i]));}
	TimestampEnd = std::clock();

	std::printf("nlz3: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (nlz4(test[i]) != test[i+1]) error(test[i], nlz4(test[i]));}
	TimestampEnd = std::clock();

	std::printf("nlz4: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (nlz5(test[i]) != test[i+1]) error(test[i], nlz5(test[i]));}
	TimestampEnd = std::clock();

	std::printf("nlz5: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (nlz6(test[i]) != test[i+1]) error(test[i], nlz6(test[i]));}
	TimestampEnd = std::clock();

	std::printf("nlz6: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (nlz7(test[i]) != test[i+1]) error(test[i], nlz7(test[i]));}
	TimestampEnd = std::clock();

	std::printf("nlz7: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (nlz8(test[i]) != test[i+1]) error(test[i], nlz8(test[i]));}
	TimestampEnd = std::clock();

	std::printf("nlz8: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (nlz9(test[i]) != test[i+1]) error(test[i], nlz9(test[i]));}
	TimestampEnd = std::clock();

	std::printf("nlz9: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (nlz10(test[i]) != test[i+1]) error(test[i], nlz10(test[i]));}
	TimestampEnd = std::clock();

	std::printf("nlz10: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (nlz10a(test[i]) != test[i+1]) error(test[i], nlz10a(test[i]));}
	TimestampEnd = std::clock();

	std::printf("nlz10a: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (nlz10b(test[i]) != test[i+1]) error(test[i], nlz10b(test[i]));}
	TimestampEnd = std::clock();

	std::printf("nlz10b: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	if (errors == 0)
		std::printf("Passed all %d cases.\n", static_cast<int>(sizeof(test)/8));

#	endif//NDEBUG
}
