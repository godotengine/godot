#include <libriscv.h>

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static struct timespec time_now();
static int64_t nanodiff(struct timespec start_time, struct timespec end_time);
static char *read_file(const char *filename, size_t *len);

static void error_callback(void *opaque, int type, const char *msg, long data)
{
	fprintf(stderr, "Error: %s (data: 0x%lX)\n", msg, data);
}

static void stdout_callback(void *opaque, const char *msg, unsigned len)
{
	printf("[libriscv] stdout: %.*s", (int)len, msg);
}

static void my_exit(RISCVMachine *m)
{
	RISCVRegisters *regs = libriscv_get_registers(m);
	#define REG_A0   10

	printf("Exit called! Status=%ld\n", regs->r[REG_A0]);
	libriscv_stop(m);
}

static void make_vm_function_call(RISCVMachine *m, const char *function);

int main(int argc, char **argv)
{
	if (argc < 2) {
		fprintf(stderr, "Usage: %s [RISC-V elf file]\n", argv[0]);
		exit(1);
	}

    size_t size = 0;
	char *buffer = read_file(argv[1], &size);

	/* Create guest program arguments from argv[2...] */
	const unsigned g_argc = argc - 1;
	const char **g_argv = calloc(1 + g_argc, sizeof(char *));
	g_argv[0] = "my_program";
	for (unsigned i = 1; i < g_argc; i++)
		g_argv[i] = argv[i + 1];

	/* RISC-V machine options */
	RISCVOptions options;
	libriscv_set_defaults(&options);
	options.max_memory = 4ULL << 30; // 4 GiB
	options.argc = g_argc;
	options.argv = g_argv;
	options.error = error_callback;
	options.stdout = stdout_callback;
	options.opaque = NULL;
	options.strict_sandbox = 0;

	/* RISC-V machine */
	RISCVMachine *m = libriscv_new(buffer, size, &options);
	if (!m) {
		fprintf(stderr, "Failed to initialize the RISC-V machine!\n");
		exit(1);
	}

	/* A custom exit system call handler. WARNING: POSIX threads will not work right! */
	libriscv_set_syscall_handler(93, my_exit);

	/* Add some allowed files that covers most dynamic executables. */
	static const char *libs[] = {
		"libdl.so.2",
		"libm.so.6",
		"libgcc_s.so.1",
		"libc.so.6",
		"libatomic.so.1",
		"libstdc++.so.6",
		"libresolv.so.2",
		"libnss_dns.so.2",
		"libnss_files.so.2"
	};
	for (unsigned i = 0; i < sizeof(libs)/sizeof(libs[0]); i++)
		libriscv_allow_file(m, libs[i]);

	struct timespec start_time = time_now();

	/* RISC-V execution, timing out after 5bn instructions */
	const int res = libriscv_run(m, 5000000000ull);
	if (res < 0) {
		fprintf(stderr, "Error during execution: %s\n", libriscv_strerror(res));
		exit(1);
	}

	struct timespec end_time = time_now();

	const int64_t retval = libriscv_return_value(m);
	const uint64_t icount = libriscv_instruction_counter(m);
	const int64_t nanos = nanodiff(start_time, end_time);

	printf(">>> Program exited, exit code = %" PRId64 " (0x%" PRIX64 ")\n",
		retval, (uint64_t)retval);
	printf("Instructions executed: %" PRIu64 "  Runtime: %.3fms  Insn/s: %.0fmi/s\n",
		icount, nanos/1e6,
		icount / (nanos * 1e-3));


	make_vm_function_call(m, "test");
	make_vm_function_call(m, "test");
	make_vm_function_call(m, "test");

	libriscv_delete(m);
}

/**
 * Make a VM function call into the program, step by step!
 **/
void make_vm_function_call(RISCVMachine *m, const char *function)
{
	/* Find the address of a function from the ELF symbol table */
	const uint64_t vaddr = libriscv_address_of(m, function);
	/* Only make the function call if "test" is a visible symbol */
	if (vaddr == 0x0)
		return;

	/* Begin a VM function call */
	printf("\n*** Starting a VM function call to %s at 0x%lX\n",
		function, (long)vaddr);

	if (libriscv_setup_vmcall(m, vaddr) == 0) {
		/**
		 * Put some arguments in machine registers:
		 * 1. An integer in arg0
		 * 2. A string in arg1
		 * In order for the program to read the string, it needs to be
		 * copied into the programs virtual memory. The easiest way to
		 * do that is to push it on the stack.
		**/
		RISCVRegisters *regs = libriscv_get_registers(m);
		/* Place an integer in the first argument (a0) register */
		LIBRISCV_ARG_REGISTER(regs, 0) = 123;
		/* Put a string (with terminating zero, due to sizeof) on the stack */
		static const char hello[] = "Hello VM-call World!";
		uint64_t strva = libriscv_stack_push(m, regs, hello, sizeof(hello));
		/* Place the _strings address_ in the second argument (a1) register */
		LIBRISCV_ARG_REGISTER(regs, 1) = strva;
		/* Begin execution, with max 1bn instruction count. */
		libriscv_run(m, 1000000000ull);

		printf("*** VM function call return value: %ld\n",
			libriscv_return_value(m));
	} else {
		fprintf(stderr,
			"Could not jump to function at 0x%lX\n", (long)vaddr);
	}
}

struct timespec time_now()
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t;
}
int64_t nanodiff(struct timespec start_time, struct timespec end_time)
{
	return (end_time.tv_sec - start_time.tv_sec) * (int64_t)1e9 + (end_time.tv_nsec - start_time.tv_nsec);
}
char *read_file(const char *filename, size_t *size)
{
    FILE* f = fopen(filename, "rb");
    if (f == NULL) {
		fprintf(stderr, "Could not open file: %s\n", filename);
		exit(1);
	}

    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);

	char *buffer = malloc(*size);
    if (*size != fread(buffer, 1, *size, f))
    {
        fclose(f);
		fprintf(stderr, "Could not read file: %s\n", filename);
		exit(1);
    }
    fclose(f);
	return buffer;
}
