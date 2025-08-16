#pragma once

#ifndef NATIVE_SYSCALLS_BASE
#define NATIVE_SYSCALLS_BASE    1  /* libc starts at 1 */
#endif
#ifndef THREAD_SYSCALLS_BASE
#define THREAD_SYSCALLS_BASE   30  /* Threads start at 30 */
#endif

#define SYSCALL_MALLOC    (NATIVE_SYSCALLS_BASE+0)
#define SYSCALL_CALLOC    (NATIVE_SYSCALLS_BASE+1)
#define SYSCALL_REALLOC   (NATIVE_SYSCALLS_BASE+2)
#define SYSCALL_FREE      (NATIVE_SYSCALLS_BASE+3)
#define SYSCALL_MEMINFO   (NATIVE_SYSCALLS_BASE+4)

#define SYSCALL_MEMCPY    (NATIVE_SYSCALLS_BASE+5)
#define SYSCALL_MEMSET    (NATIVE_SYSCALLS_BASE+6)
#define SYSCALL_MEMMOVE   (NATIVE_SYSCALLS_BASE+7)
#define SYSCALL_MEMCMP    (NATIVE_SYSCALLS_BASE+8)

#define SYSCALL_STRLEN    (NATIVE_SYSCALLS_BASE+10)
#define SYSCALL_STRCMP    (NATIVE_SYSCALLS_BASE+11)

#define SYSCALL_BACKTRACE (NATIVE_SYSCALLS_BASE+19)

#define SYSCALL_WRITE  64
#define SYSCALL_EXIT   93

inline long syscall1(long n)
{
	register long a0 asm("a0");
	register long syscall_id asm("a7") = n;

	asm volatile ("scall" : "+r"(a0) : "r"(syscall_id));

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

inline long syscall(long n, long arg0, long arg1, long arg2, long arg3)
{
	register long a0 asm("a0") = arg0;
	register long a1 asm("a1") = arg1;
	register long a2 asm("a2") = arg2;
	register long a3 asm("a3") = arg3;
	register long syscall_id asm("a7") = n;

	asm volatile ("scall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a3), "r"(syscall_id));

	return a0;
}

inline long
syscall(long n, long arg0, long arg1, long arg2, long arg3, long arg4)
{
	register long a0 asm("a0") = arg0;
	register long a1 asm("a1") = arg1;
	register long a2 asm("a2") = arg2;
	register long a3 asm("a3") = arg3;
	register long a4 asm("a4") = arg4;
	register long syscall_id asm("a7") = n;

	asm volatile ("scall"
		: "+r"(a0) : "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(syscall_id));

	return a0;
}

inline long
syscall(long n, long arg0, long arg1, long arg2, long arg3, long arg4, long arg5)
{
	register long a0 asm("a0") = arg0;
	register long a1 asm("a1") = arg1;
	register long a2 asm("a2") = arg2;
	register long a3 asm("a3") = arg3;
	register long a4 asm("a4") = arg4;
	register long a5 asm("a5") = arg5;
	// NOTE: only 16 regs in RV32E instruction set
	register long syscall_id asm("a7") = n;

	asm volatile ("scall"
		: "+r"(a0) : "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(syscall_id));

	return a0;
}

inline long
syscall(long n, long arg0, long arg1, long arg2,
		long arg3, long arg4, long arg5, long arg6)
{
	register long a0 asm("a0") = arg0;
	register long a1 asm("a1") = arg1;
	register long a2 asm("a2") = arg2;
	register long a3 asm("a3") = arg3;
	register long a4 asm("a4") = arg4;
	register long a5 asm("a5") = arg5;
	register long a6 asm("a6") = arg6;
	register long syscall_id asm("a7") = n;

	asm volatile ("scall"
		: "+r"(a0) : "r"(a1), "r"(a2), "r"(a3),
			"r"(a4), "r"(a5), "r"(a6), "r"(syscall_id));

	return a0;
}

/** Pointer-parameter syscalls **/

inline long psyscall(long n, const void* arg0)
{
	asm ("" ::: "memory");
	return syscall(n, (long) arg0);
}

inline long psyscall(long n, const void* arg0, const void* arg1)
{
	asm ("" ::: "memory");
	return syscall(n, (long) arg0, (long) arg1);
}

inline long psyscall(long n, const void* arg0, long arg1)
{
	asm ("" ::: "memory");
	return syscall(n, (long) arg0, arg1);
}

inline long psyscall(long n, const void* arg0, long arg1, long arg2)
{
	asm ("" ::: "memory");
	return syscall(n, (long) arg0, arg1, arg2);
}

inline long psyscall(long n, const void* arg0, long arg1, const void* arg2)
{
	asm ("" ::: "memory");
	return syscall(n, (long) arg0, arg1, (long) arg2);
}

inline long psyscall(long n, const void* arg0, const void* arg1, const void* arg2)
{
	asm ("" ::: "memory");
	return syscall(n, (long) arg0, (long) arg1, (long) arg2);
}

/** Floating-point System calls */

inline float
fsyscallf(long n, float farg0)
{
	register float fa0 asm("fa0") = farg0;
	register long syscall_id asm("a7") = n;
	float retval;

	asm volatile ("scall"
	 	: "+f"(fa0) : "r"(syscall_id) : "a0");
	retval = fa0;

	return retval;
}

inline float
fsyscallf(long n, float farg0, float farg1)
{
	register float fa0 asm("fa0") = farg0;
	register float fa1 asm("fa1") = farg1;
	register long syscall_id asm("a7") = n;

	asm volatile ("scall"
	 	: "+f"(fa0) : "f"(fa1), "r"(syscall_id) : "a0");

	return fa0;
}

inline float
fsyscallf(long n, float farg0, float farg1, float farg2)
{
	register float fa0 asm("fa0") = farg0;
	register float fa1 asm("fa1") = farg1;
	register float fa2 asm("fa2") = farg2;
	register long syscall_id asm("a7") = n;

	asm volatile ("scall"
	 	: "+f"(fa0) : "f"(fa1), "f"(fa2), "r"(syscall_id) : "a0");

	return fa0;
}
