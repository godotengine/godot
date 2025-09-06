# Creating your own environment

If you want a completely freestanding environment in your embedded program you will need to do a few things in order to call into a C function properly and use both stack and static storage.

## Setup

You will want to avoid using a low address as the initial stack value, as it could mean that some stack pointer value will be evaluated as 0x0 (null) with some bad luck, which could mysteriously fail one of your own checks or asserts. Additionally, the machine automatically makes the zero-page unreadable on start to help you catch accesses to the zero-page, which are typically bugs. It's fine to start the stack at 0x0 though, as the address will wrap around and start pushing bytes at the top of the address space, at least in 32-bit.

```C++
machine.cpu.reg(riscv::REG_SP) = 0x0;
```

Additionally, compile your binary with `-nostdlib -nostdinc` and possibly `-ffreestanding` if you don't have an embedded compiler.

## Programming the startup

From now on, all the example code is going to be implemented inside the guest binary, in the ELF entry function which is always named `_start` and is a C function. In C++ you could write it like this:

```C++
extern "C"
void _start()
{
	// startup code here
}
```

The first thing you must do is setting the GP register to the absolute address of `__global_pointer`. The only way to do that is to disable relaxation:

```C++
asm volatile
("   .option push 				\t\n\
	 .option norelax 			\t\n\
	 1:auipc gp, %pcrel_hi(__global_pointer$) \t\n\
	 addi  gp, gp, %pcrel_lo(1b) \t\n\
	.option pop					\t\n\
");
// make sure all accesses to static memory happen after:
asm volatile("" ::: "memory");
```

Now that we have access to static storage, we can clear .bss which is the area of memory used by zero-initialized variables:
```C++
extern char __bss_start;
extern char __BSS_END__;
for (char* bss = &__bss_start; bss < &__BSS_END__; bss++) {
	*bss = 0;
}
```

Memory is initially zero in the emulator, and so the BSS zeroing can be skipped.

After this you might want to initialize your heap, if you have one. If not, consider getting a tiny heap implementation from an open source project. Perhaps also initialize some early standard out (stdout) facility so that you can get feedback from subsystems that print errors during initialization.

Next up is calling global constructors, which while not common in C is very common in C++ and other languages, and doesn't contribute much to the binary size:

```C++
extern void(*__init_array_start [])();
extern void(*__init_array_end [])();
int count = __init_array_end - __init_array_start;
for (int i = 0; i < count; i++) {
	__init_array_start[i]();
}
```
Now you are done initializing the absolute minimal C/C++ freestanding environment. Calling main is as simple as:

```C++
extern int main(int, char**);

// geronimo!
_exit(main(0, nullptr));
```

Here we mandate that you must implement `int main()` or get an undefined reference, and also the almost-mandatory `_exit` system call wrapper. You can implement `_exit` like this:

```C++
#define SYSCALL_EXIT   93

extern "C" {
	__attribute__((noreturn))
	void _exit(int status) {
		syscall(SYSCALL_EXIT, status);
		__builtin_unreachable();
	}
}
```

You will need to handle the EXIT system call on the outside of the machine as well, to stop the machine. If you don't handle the EXIT system call and stop the machine, it will continue executing instructions past the function, which does not return. A one-argument system call can be implemented like this:

```C++
template <int W>
void syscall_exit(riscv::Machine<W>& machine)
{
	printf(">>> Program exited, exit code = %d\n", machine.template sysarg<int> (0));
	machine.stop();
}
```
And installed as a 32-bit system call handler like this:
```C++
machine.install_syscall_handler(93, syscall_exit<riscv::RISCV32>);
```

The machine instruction processing loop will stop running immediately after this system call has been invoked.

Finally, to make a system call with one (1) argument from the guest environment you could do something like this (in C++):
```C++
inline long syscall(long n, long arg0)
{
	register long a0 asm("a0") = arg0;
	register long syscall_id asm("a7") = n;

	asm volatile ("ecall" : "+r"(a0) : "r"(syscall_id));

	return a0;
}
```

All integer and pointer arguments are in the a0 to a6 registers, which adds up to 7 arguments in total. The return value of the system call is written back into a0. If you want to create a custom system call that fills some values into a struct, you should allocate room for that struct inside the guest, and just pass the pointer to that struct as one of the arguments to the system call.

If you have done all this you should now have the absolute minimum C and C++ freestanding environment up and running. Have fun!

## Putting it all together

Have a look at [start.cpp](/binaries/micro/src/start.cpp) for the micro example project.

# Main arguments

On Linux, main() can take several arguments:

```
int main(int argc, char** argc, char** envp);
```

Here, envp is the pointer to the environment variables. It is a list of strings that ends with a NULL value.
