// This has the programs for computing the number of 1-bits
// in a word, or byte, etc.
// Max line length is 57, to fit in hacker.book.
#include <cstdio>
#include <cstdlib>     //To define "exit", req'd by XLC.
#include <ctime>

unsigned rotatel(unsigned x, int n)
{
	if (static_cast<unsigned>(n) > 63) { std::printf("rotatel, n out of range.\n"); std::exit(1);}
	return (x << n) | (x >> (32 - n));
}

int pop0(unsigned x)
{
	x = (x & 0x55555555) + ((x >> 1) & 0x55555555);
	x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
	x = (x & 0x0F0F0F0F) + ((x >> 4) & 0x0F0F0F0F);
	x = (x & 0x00FF00FF) + ((x >> 8) & 0x00FF00FF);
	x = (x & 0x0000FFFF) + ((x >>16) & 0x0000FFFF);
	return x;
}

int pop1(unsigned x)
{
	x = x - ((x >> 1) & 0x55555555);
	x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
	x = (x + (x >> 4)) & 0x0F0F0F0F;
	x = x + (x >> 8);
	x = x + (x >> 16);
	return x & 0x0000003F;
}
/* Note: an alternative to the last three executable lines above is:
   return x*0x01010101 >> 24;
if your machine has a fast multiplier (suggested by Jari Kirma). */

int pop2(unsigned x)
{
	unsigned n;

	n = (x >> 1) & 033333333333;       // Count bits in
	x = x - n;                         // each 3-bit
	n = (n >> 1) & 033333333333;       // field.
	x = x - n;
	x = (x + (x >> 3)) & 030707070707; // 6-bit sums.
	return x%63;                       // Add 6-bit sums.
}

/* An alternative to the "return" statement above is:
   return ((x * 0404040404) >> 26) +  // Add 6-bit sums.
           (x >> 30);
which runs faster on most machines (suggested by Norbert Juffa). */

int pop3(unsigned x)
{
	unsigned n;

	n = (x >> 1) & 0x77777777;        // Count bits in
	x = x - n;                        // each 4-bit
	n = (n >> 1) & 0x77777777;        // field.
	x = x - n;
	n = (n >> 1) & 0x77777777;
	x = x - n;
	x = (x + (x >> 4)) & 0x0F0F0F0F;  // Get byte sums.
	x = x*0x01010101;                 // Add the bytes.
	return x >> 24;
}

int pop4(unsigned x)
{
	int n;

	n = 0;
	while (x != 0) {
		n = n + 1;
		x = x & (x - 1);
	}
	return n;
}

int pop5(unsigned x)
{
	int i, sum;

	// Rotate and sum method        // Shift right & subtract

	sum = x;                     // sum = x;
	for (i = 1; i <= 31; i++) {  // while (x != 0) {
		x = rotatel(x, 1);        //    x = x >> 1;
		sum = sum + x;            //    sum = sum - x;
	}                            // }
	return -sum;                 // return sum;
}

int pop5a(unsigned x)
{
	int sum;

	// Shift right & subtract

	sum = x;
	while (x != 0) {
		x = x >> 1;
		sum = sum - x;
	}
	return sum;
}

int pop6(unsigned x)
{ // Table lookup.
	static char table[256] = {
		0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
		1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
		1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,

		1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
		3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,

		1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
		3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,

		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
		3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
		3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
		4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};

	return table[x         & 0xFF] +
			table[(x >>  8) & 0xFF] +
			table[(x >> 16) & 0xFF] +
			table[(x >> 24)];
}

// The following works only for 8-bit quantities.
int pop7(unsigned x)
{
	x = x*0x08040201;    // Make 4 copies.
	x = x >> 3;          // So next step hits proper bits.
	x = x & 0x11111111;  // Every 4th bit.
	x = x*0x11111111;    // Sum the digits (each 0 or 1).
	x = x >> 28;         // Position the result.
	return x;
}

// The following works only for 7-bit quantities.
int pop8(unsigned x)
{
	x = x*0x02040810;    // Make 4 copies, left-adjusted.
	x = x & 0x11111111;  // Every 4th bit.
	x = x*0x11111111;    // Sum the digits (each 0 or 1).
	x = x >> 28;         // Position the result.
	return x;
}

// The following works only for 15-bit quantities.
int pop9(unsigned x)
{
	unsigned long long y;
	y = x * 0x0002000400080010ULL;
	y = y & 0x1111111111111111ULL;
	y = y * 0x1111111111111111ULL;
	y = y >> 60;
	return static_cast<int>(y);
}

int errors;
void error(int x, int y)
{
	errors = errors + 1;
	std::printf("Error for x = %08x, got %08x\n", x, y);
}

int main()
{
#	ifdef NDEBUG

	int i, n;
	static unsigned test[] = {0,0, 1,1, 2,1, 3,2, 4,1, 5,2, 6,2, 7,3,
		8,1, 9,2, 10,2, 11,3, 12,2, 13,3, 14,3, 15,4, 16,1, 17,2,
		0x3F,6, 0x40,1, 0x41,2, 0x7f,7, 0x80,1, 0x81,2, 0xfe,7, 0xff,8,
		0x4000,1, 0x4001,2, 0x7000,3, 0x7fff,15,
		0x55555555,16, 0xAAAAAAAA, 16, 0xFF000000,8, 0xC0C0C0C0,8,
		0x0FFFFFF0,24, 0x80000000,1, 0xFFFFFFFF,32};

	std::size_t const Count = 1000000;

	n = sizeof(test)/4;

	std::clock_t TimestampBeg = 0;
	std::clock_t TimestampEnd = 0;

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (pop0(test[i]) != test[i+1]) error(test[i], pop0(test[i]));}
	TimestampEnd = std::clock();

	std::printf("pop0: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (pop1(test[i]) != test[i+1]) error(test[i], pop1(test[i]));}
	TimestampEnd = std::clock();

	std::printf("pop1: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (pop2(test[i]) != test[i+1]) error(test[i], pop2(test[i]));}
	TimestampEnd = std::clock();

	std::printf("pop2: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (pop3(test[i]) != test[i+1]) error(test[i], pop3(test[i]));}
	TimestampEnd = std::clock();

	std::printf("pop3: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (pop4(test[i]) != test[i+1]) error(test[i], pop4(test[i]));}
	TimestampEnd = std::clock();

	std::printf("pop4: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (pop5(test[i]) != test[i+1]) error(test[i], pop5(test[i]));}
	TimestampEnd = std::clock();

	std::printf("pop5: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (pop5a(test[i]) != test[i+1]) error(test[i], pop5a(test[i]));}
	TimestampEnd = std::clock();

	std::printf("pop5a: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if (pop6(test[i]) != test[i+1]) error(test[i], pop6(test[i]));}
	TimestampEnd = std::clock();

	std::printf("pop6: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if ((test[i] & 0xffffff00) == 0)
		if (pop7(test[i]) != test[i+1]) error(test[i], pop7(test[i]));}
	TimestampEnd = std::clock();

	std::printf("pop7: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if ((test[i] & 0xffffff80) == 0)
		if (pop8(test[i]) != test[i+1]) error(test[i], pop8(test[i]));}
	TimestampEnd = std::clock();

	std::printf("pop8: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	TimestampBeg = std::clock();
	for (std::size_t k = 0; k < Count; ++k)
	for (i = 0; i < n; i += 2) {
		if ((test[i] & 0xffff8000) == 0)
		if (pop9(test[i]) != test[i+1]) error(test[i], pop9(test[i]));}
	TimestampEnd = std::clock();

	std::printf("pop9: %d clocks\n", static_cast<int>(TimestampEnd - TimestampBeg));

	if (errors == 0)
		std::printf("Passed all %d cases.\n", static_cast<int>(sizeof(test)/8));

#	endif//NDEBUG
}
