static long fib(long n, long acc, long prev)
{
	if (n == 0)
		return acc;
	else
		return fib(n - 1, prev + acc, acc);
}

static inline long syscall(long n, long arg0) {
	register long a0 asm("a0") = arg0;
	register long syscall_id asm("a7") = n;

	__asm__ volatile ("scall" : "+r"(a0) : "r"(syscall_id));

	return a0;
}

void _start()
{
	const volatile long n = 256000000;
	syscall(93, fib(n, 0, 1));
}
