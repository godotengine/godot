/**
 * Accelerated heap using syscalls
 *
**/
#pragma once
#include <cstddef>
#include <include/syscall.hpp>

struct MemInfo {
	size_t bytes_free;
	size_t bytes_used;
	size_t chunks_used;
};

inline void* sys_malloc(std::size_t size) {
	register void*   ret asm("a0");
	register size_t  a0  asm("a0") = size;
	register long syscall_id asm("a7") = SYSCALL_MALLOC;

	asm volatile ("ecall"
	:	"=m"(*(char(*)[size]) ret), "=r"(ret)
	:	"r"(a0), "r"(syscall_id));
	return ret;
}
inline void* sys_calloc(size_t, size_t);
inline void* sys_realloc(void*, size_t);
inline void  sys_free(void* ptr)
{
	register void*  a0  asm("a0") = ptr;
	register long syscall_id asm("a7") = SYSCALL_FREE;

	asm volatile ("ecall"
	:
	:	"r"(a0), "r"(syscall_id));
}

inline int sys_meminfo(void* ptr, size_t len)
{
	return psyscall(SYSCALL_MEMINFO, ptr, len);
}
