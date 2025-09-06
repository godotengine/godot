## libriscv C API

The API is fairly new and currently only executes 64-bit RISC-V programs.

There is a [demo program](/c/test/test.c) showing how the C API can be used.

[![C API test project](https://github.com/fwsGonzo/libriscv/actions/workflows/capi.yml/badge.svg)](https://github.com/fwsGonzo/libriscv/actions/workflows/capi.yml)

### Example code

```c
RISCVOptions options;
libriscv_set_defaults(&options);
options.max_memory = 4ULL << 30; /* 4 GiB */
options.strict_sandbox = 1; /* Disable files and sockets. */

/* Guest program arguments. */
const char * guest_args[2] = {"my_program", "1234"};
options.argc = 2;
options.argv = guest_args;

/* Create a new RISC-V machine. The program must out-live the machine. */
RISCVMachine *m = libriscv_new(buffer, size, &options);
if (!m) {
	fprintf(stderr, "Failed to initialize the RISC-V machine!\n");
	exit(1);
}

/* Execute the RISC-V program now, but time out execution at 5B instructions. */
const int res = libriscv_run(m, 5000000000ull);
if (res < 0) {
	fprintf(stderr, "Error during execution: %s\n", libriscv_strerror(res));
	exit(1);
}

/* Free the program. */
libriscv_delete(m);
```

You can find a more advanced example of C API usage in the [advanced examples project](../examples/advanced/src/example.c). In that example we are defining host functions and calling them.

### API header

The current [C API header](/c/libriscv.h) is a work in progress. It covers many basic operations, but can only load 64-bit RISC-V programs for now.

```c
#define RISCV_ERROR_TYPE_GENERAL_EXCEPTION  -1
#define RISCV_ERROR_TYPE_MACHINE_EXCEPTION  -2
#define RISCV_ERROR_TYPE_MACHINE_TIMEOUT    -3
typedef void (*riscv_error_func_t)(void *opaque, int type, const char *msg, long data);

typedef void (*riscv_stdout_func_t)(void *opaque, const char *msg, unsigned size);

typedef struct {
	uint64_t max_memory;
	uint32_t stack_size;
	int      strict_sandbox;  /* No file or socket permissions */
	unsigned     argc;        /* Program arguments */
	const char **argv;
	riscv_error_func_t error; /* Error callback */
	riscv_stdout_func_t stdout; /* Stdout callback */
	void *opaque;             /* User-provided pointer */
} RISCVOptions;

/* Fill out default values. */
void libriscv_set_defaults(RISCVOptions *options);

/* Create a new 64-bit RISC-V machine from an ELF binary. The binary must out-live the machine. */
RISCVMachine *libriscv_new(const void *elf_prog, unsigned elf_size, RISCVOptions *o);

/* Free a RISC-V machine created using libriscv_new. */
int libriscv_delete(RISCVMachine *m);


/* Start execution at current PC, with the given instruction limit. 0 on success.
   When an error occurs, the negative value is one of the RISCV_ERROR_ enum values. */
LIBRISCVAPI int libriscv_run(RISCVMachine *m, uint64_t instruction_limit);

/* Add a host-side filepath that can be opened by the guest program. Sandbox must be disabled. */
LIBRISCVAPI void libriscv_allow_file(RISCVMachine *m, const char *path);

/* Returns a string describing a negative return value. */
LIBRISCVAPI const char * libriscv_strerror(int return_value);

/* Return current value of the return value register A0. */
LIBRISCVAPI int64_t libriscv_return_value(RISCVMachine *m);

/* Return symbol address or NULL if not found. */
LIBRISCVAPI uint64_t libriscv_address_of(RISCVMachine *m, const char *name);

/* Return the opaque value provided during machine creation. */
LIBRISCVAPI void * libriscv_opaque(RISCVMachine *m);

/*** View and modify the RISC-V emulator state ***/

typedef union {
	float   f32[2];
	double  f64;
} RISCVFloat;

typedef struct {
	uint64_t  pc;
	uint64_t  r[32];
	uint32_t  fcsr;
	RISCVFloat fr[32];
} RISCVRegisters;

/* Retrieve the internal registers of the RISC-V machine. Changing PC is dangerous. */
LIBRISCVAPI RISCVRegisters * libriscv_get_registers(RISCVMachine *m);

/* Change the PC register safely. PC can be changed before running and during system calls. */
LIBRISCVAPI int libriscv_jump(RISCVMachine *m, uint64_t address);

/* Copy memory in and out of the RISC-V machine. */
LIBRISCVAPI int libriscv_copy_to_guest(RISCVMachine *m, uint64_t dst, const void *src, unsigned len);
LIBRISCVAPI int libriscv_copy_from_guest(RISCVMachine *m, void *dst, uint64_t src, unsigned len);

/* View a zero-terminated string from readable memory of at most maxlen length. The string is read-only.
   On success, set *length and return a pointer to the string, zero-copy. Otherwise, return null. */
LIBRISCVAPI const char * libriscv_memstring(RISCVMachine *m, uint64_t src, unsigned maxlen, unsigned *length);

/* View a slice of readable (but not guaranteed writable) memory from src to src + length.
   On success, returns a pointer to the memory. Otherwise, returns null. */
LIBRISCVAPI const char * libriscv_memview(RISCVMachine *m, uint64_t src, unsigned length);

/* View a slice of readable and writable memory from src to src + length.
   On success, returns a pointer to the writable memory. Otherwise, returns null. */
LIBRISCVAPI char * libriscv_writable_memview(RISCVMachine *m, uint64_t src, unsigned length);

/* Stops execution normally. Only possible from a system call and EBREAK. */
LIBRISCVAPI void libriscv_stop(RISCVMachine *m);

/* Return current instruction counter value. */
LIBRISCVAPI uint64_t libriscv_instruction_counter(RISCVMachine *m);

/* Return a *pointer* to the instruction max counter. */
LIBRISCVAPI uint64_t * libriscv_max_counter_pointer(RISCVMachine *m);

/*** RISC-V system call handling ***/

typedef void (*riscv_syscall_handler_t)(RISCVMachine *m);

/* Install a custom system call handler. */
LIBRISCVAPI int libriscv_set_syscall_handler(unsigned num, riscv_syscall_handler_t);

/* Triggers a CPU exception. Only safe to call from a system call. Will end execution. */
LIBRISCVAPI void libriscv_trigger_exception(RISCVMachine *m, unsigned exception, uint64_t data);

/*** RISC-V VM function calls ***/

/* Make preparations for a VM function call. Returns 0 on success. */
LIBRISCVAPI int libriscv_setup_vmcall(RISCVMachine *m, uint64_t address);

/* Stack realignment helper. */
#define LIBRISCV_REALIGN_STACK(regs)  ((regs)->r[2] & ~0xFLL)

/* Register function or system call argument helper. */
#define LIBRISCV_ARG_REGISTER(regs, n)  (regs)->r[10 + (n)]

/* Put data on the current stack, with maintained 16-byte alignment. */
uint64_t libriscv_stack_push(RISCVMachine *m, RISCVRegisters *regs, const char *data, unsigned len);

```

### VM function calls

See [the C API test program](/c/test/test.c) for an example of a VM function call. It requires the symbol 'test' to be visible in the ELF program. The [64-bit newlib example](/binaries/newlib64/src/hello_world.cpp) will have a 'test' function.

The current API is fairly open in order to be low latency. It should be possible to macroize the function calls, or at least make them more dynamic, so that they appear more like a function call API-wise. For now, tools not policies.

### VM preemption

In order to preempt, store all registers and the max instruction counter, then VM call the preemption address. Once back from the call, restore registers and pass the old max instruction counter to `libriscv_run()`:

```c
void my_system_call(RISCVMachine *m)
{
	RISCVRegisters *regs = libriscv_get_registers(m);
	uint64_t *max_ptr = libriscv_max_counter_pointer(m);

	/* Store registers and max counter */
	RISCVRegisters temp_regs = *regs;
	uint64_t  temp_max = *max_ptr;

	/* Make a VM function call somewhere else */
	my_vmcall(m, other_function_address);

	/* Restore registers and max counter */
	*regs = temp;
	*max_ptr = temp_max;

	/* Set system call return register a0 */
	LIBRISCV_ARG_REGISTER(regs, 0) = 0;
}

```
