#include <stdio.h>
// A macro that creates a varadic callable host function
// Drawback: Floats are promoted to doubles
#define STR(x) #x
#define CREATE_HOST_FUNCTION(INDEX, ...) \
	__asm__(".pushsection .text\n"	\
		".global call_host_function" #INDEX "\n"	\
		"call_host_function" #INDEX ":\n"	\
		"    li a7, " STR(INDEX) "\n"	\
		"    ecall\n"	\
		"    ret\n"	\
		".popsection\n");	\
	extern long call_host_function ## INDEX(__VA_ARGS__);

// Create three host functions indexed 500, 501, and 502
struct Strings {
	unsigned long count;
	const char *strings[32];
};
CREATE_HOST_FUNCTION(500, struct Strings*);
struct Buffer {
	unsigned long count;
	char buffer[256];
	unsigned long another_count;
	char *another_buffer;
};
CREATE_HOST_FUNCTION(501, struct Buffer*);
typedef void (*HostFunction)(const char*);
CREATE_HOST_FUNCTION(502, HostFunction);

// For the third host function, we'll use inline assembly to make
// a blazing fast vec3 normalize function
struct vec3 {
	float x, y, z;
};
static inline struct vec3 normalize(struct vec3 *v)
{
	// Assign X, Y, Z to the first three argument floating-point registers (FA0, FA1, FA2)
	register float x __asm__ ("fa0") = v->x;
	register float y __asm__ ("fa1") = v->y;
	register float z __asm__ ("fa2") = v->z;
	// Assign the system call number to A7
	register int  a7 __asm__ ("a7")  = 503;
	// Perform the system call using inline assembly
	// +f means that it's both an input and an output register (the host modifies it)
	__asm__("ecall"
		: "+f" (x), "+f" (y), "+f" (z)
		: "r" (a7));
	// This should boil down to extremely efficient instructions
	return (struct vec3){x, y, z};
}

static void my_function(const char *str)
{
	printf("Host says: %s\n", str);
	fflush(stdout);
}

int main()
{
	printf("Hello, Micro RISC-V World!\n");

	// Call the host function that prints strings
	struct Strings vec = {2, {"Hello", "World"}};
	call_host_function500(&vec);

	// Call the host function that modifies a buffer
	struct Buffer buf;
	char another_buf[256];
	buf.another_count = sizeof(another_buf);
	buf.another_buffer = another_buf;
	call_host_function501(&buf);
	printf("Buffer: %s\n", buf.buffer);
	printf("Another Buffer: %s\n", buf.another_buffer);

	// Call a host function that takes a function pointer
	call_host_function502(&my_function);

	// Call a host function that normalizes a vector
	// We'll use a simple 0, 3, 0 which can be easily verified as 0, 1, 0
	struct vec3 v = {0.0f, 3.0f, 0.0f};
	v = normalize(&v);
	printf("Normalized vector: %.1f, %.1f, %.1f\n", v.x, v.y, v.z);
}
