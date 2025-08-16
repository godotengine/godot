#include <stdio.h>

#include <libriscv.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
typedef unsigned long gaddr_t;

static void error_callback(void *opaque, int type, const char *msg, long data)
{
	fprintf(stderr, "Error: %s (data: 0x%lX)\n", msg, data);
}

static void stdout_callback(void *opaque, const char *msg, unsigned len)
{
	printf("[libriscv] stdout: %.*s", (int)len, msg);
}

// A host function that can be called from the guest program
static void host_function_500(RISCVMachine *m)
{
	printf("Hello from host function 0!\n");
	RISCVRegisters *regs = libriscv_get_registers(m);

	// Get a zero-copy view of Strings in guest memory
	struct Strings {
		gaddr_t count;
		gaddr_t strings[32];
	};
	struct Strings *strings = LIBRISCV_VIEW_ARG(m, regs, 0, struct Strings);

	// For each string up to count, read it from guest memory and print it
	for (size_t i = 0; i < strings->count; i++) {
		unsigned len;
		const char *str = libriscv_memstring(m, strings->strings[i], 256, &len);
		printf("  %s\n", str);
	}
}

static void host_function_501(RISCVMachine *m)
{
	printf("Hello from host function 1!\n");
	RISCVRegisters *regs = libriscv_get_registers(m);

	// Get a zero-copy view of Strings in guest memory
	struct Buffers {
		gaddr_t count;
		char    buffer[256];            // An inline buffer
		gaddr_t another_count;
		gaddr_t another_buffer_address; // A pointer to a buffer somewhere in guest memory
	};
	struct Buffers *buf = LIBRISCV_VIEW_WRITABLE_ARG(m, regs, 0, struct Buffers);

	// Write a string to the buffer in guest memory
	strcpy(buf->buffer, "Hello from host function 1!");
	buf->count = strlen(buf->buffer);

	// The "another" buffer has a count and then a guest pointer to the buffer
	// In order to get a writable pointer to that buffer, we can use memarray<T>():
	char* another_buf = LIBRISCV_VIEW_WRITABLE_ARRAY(m, char, buf->another_buffer_address, buf->another_count);
	// Let's check if the buffer is large enough to hold the string
	static const char str[] = "Another buffer from host function 1!";
	if (sizeof(str) > buf->another_count) {
		printf("Another buffer is too small to hold the string!\n");
		return;
	}
	// Copy the string to the buffer
	memcpy(another_buf, str, sizeof(str));
	// Update the count of the buffer
	buf->another_count = sizeof(str)-1;
}

// A host function that stores a guest-side function pointer
static gaddr_t g_host_functions_addr = 0;
static void host_function_502(RISCVMachine *m)
{
	RISCVRegisters *regs = libriscv_get_registers(m);

	// Get the function pointer argument as a guest address
	g_host_functions_addr = LIBRISCV_ARG_REGISTER(regs, 0); // A0
}

// A host function that deals with floating-point numbers
// As an example we take in X, Y, Z and normalize them, then
// return the result in the same registers.
static void host_function_503(RISCVMachine *m)
{
	RISCVRegisters *regs = libriscv_get_registers(m);

	// Get the floating-point arguments
	float x = LIBRISCV_FP32_ARG_REG(regs, 0);
	float y = LIBRISCV_FP32_ARG_REG(regs, 1);
	float z = LIBRISCV_FP32_ARG_REG(regs, 2);

	// Normalize the vector
	const float len = sqrtf(x*x + y*y + z*z);
	if (len > 0) {
		const float inv_len = 1.0f / len;
		x *= inv_len;
		y *= inv_len;
		z *= inv_len;
	}

	// Return the result
	LIBRISCV_FP32_ARG_REG(regs, 0) = x;
	LIBRISCV_FP32_ARG_REG(regs, 1) = y;
	LIBRISCV_FP32_ARG_REG(regs, 2) = z;
}

int main(int argc, char **argv)
{
	char *program;
	size_t size;
	
	size = libriscv_load_binary_file(argv[1], &program);
	if (size < 0)
	{
		fprintf(stderr, "Error loading file %s. Check that it exists?\n", argv[1]);
		return 1;
	}

	// Register custom system call handlers
	libriscv_set_syscall_handler(500, host_function_500);
	libriscv_set_syscall_handler(501, host_function_501);
	libriscv_set_syscall_handler(502, host_function_502);
	libriscv_set_syscall_handler(503, host_function_503);

	RISCVOptions options;
	libriscv_set_defaults(&options);
	options.stdout = stdout_callback;
	options.error = error_callback;
	// Setting up arguments is a requirement for a Linux environment
	char *argv2[] = {"program", NULL};
	options.argv = (const char **)argv2;
	options.argc = 1;

	RISCVMachine *machine = libriscv_new(program, size, &options);
	if (machine == NULL)
	{
		fprintf(stderr, "Error creating machine\n");
		return 1;
	}

	if (libriscv_run(machine, ~0))
	{
		fprintf(stderr, "Error running main() function\n");
		return 1;
	}

	// Call the host function that takes a function pointer
	if (g_host_functions_addr != 0)
	{
		if (libriscv_setup_vmcall(machine, g_host_functions_addr) == 0)
		{
			RISCVRegisters *regs = libriscv_get_registers(machine);

			// Add a string argument to the function
			static const char hello[] = "Hello from a callback function!";
			uint64_t strva = libriscv_stack_push(machine, regs, hello, sizeof(hello));
			LIBRISCV_ARG_REGISTER(regs, 0) = strva;

			if (libriscv_run(machine, ~0))
			{
				fprintf(stderr, "Error running callback function\n");
				return 1;
			}
		} else {
			fprintf(stderr, "Could not jump to function at 0x%lX\n", (long)g_host_functions_addr);
		}
	} else {
		fprintf(stderr, "Host function 2 was not called!!?\n");
	}

	libriscv_delete(machine);
	free(program);
	printf("Done\n");
	return 0;
}
