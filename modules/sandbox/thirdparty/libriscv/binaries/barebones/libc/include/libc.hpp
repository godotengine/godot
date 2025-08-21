#pragma once
#include <cstddef>
#include <cstdint>
#define _NOTHROW __attribute__ ((__nothrow__))

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((noreturn))
void panic(const char* reason);

#ifdef __GLIBC__
#include <stdlib.h>
#include <string.h>
#else
extern void* memset(void* dest, int ch, size_t size);
extern void* memcpy(void* dest, const void* src, size_t size);
extern void* memmove(void* dest, const void* src, size_t size);
extern int   memcmp(const void* ptr1, const void* ptr2, size_t n);
extern char*  strcpy(char* dst, const char* src);
extern size_t strlen(const char* str);
extern int    strcmp(const char* str1, const char* str2);
extern char*  strcat(char* dest, const char* src);

extern void* malloc(size_t) _NOTHROW;
extern void* calloc(size_t, size_t) _NOTHROW;
extern void  free(void*) _NOTHROW;
#endif

#define STDIN_FILENO  0
#define STDOUT_FILENO 1
#define STDERR_FILENO 2

#ifdef __cplusplus
}
#endif

extern "C" long sys_write(const void* data, size_t len);

inline void put_string(const char* string)
{
	(void) sys_write(string, __builtin_strlen(string));
}
