#pragma once

#define SYSCALL_WRITE  64
#define SYSCALL_EXIT   93

extern "C"
{
	__attribute__((noreturn))
	void _exit(int);
	long write(int fd, const void*, unsigned long);
}


inline long syscall(long n)
{
	register long a0 asm("a0");
	register long syscall_id asm("a7") = n;

	asm volatile ("scall" : "=r"(a0) : "r"(syscall_id));

	return a0;
}

inline long syscall(long n, long arg0)
{
	register long a0 asm("a0") = arg0;
	register long syscall_id asm("a7") = n;

	asm volatile ("scall" : "+r"(a0) : "r"(syscall_id));

	return a0;
}

inline long syscall(long n, long arg0, long arg1)
{
	register long a0 asm("a0") = arg0;
	register long a1 asm("a1") = arg1;
	register long syscall_id asm("a7") = n;

	asm volatile ("scall" : "+r"(a0) : "r"(a1), "r"(syscall_id));

	return a0;
}

inline long syscall(long n, long arg0, long arg1, long arg2)
{
	register long a0 asm("a0") = arg0;
	register long a1 asm("a1") = arg1;
	register long a2 asm("a2") = arg2;
	register long syscall_id asm("a7") = n;

	asm volatile ("scall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(syscall_id));

	return a0;
}
