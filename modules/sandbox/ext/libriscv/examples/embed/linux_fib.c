#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

static long fib(uint64_t n, uint64_t acc, uint64_t prev)
{
	if (n == 0)
		return acc;
	else
		return fib(n - 1, prev + acc, acc);
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		fprintf(stderr, "%s: Missing argument: n\n", argv[0]);
		return -1;
	}
	const int n = atoi(argv[1]);
	printf("fib(%d) = %lu\n", n, fib(n, 0, 1));
}
