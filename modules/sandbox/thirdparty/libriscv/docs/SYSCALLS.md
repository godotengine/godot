# System Calls

System calls are services that the program running inside the emulator can request from the host to function properly, and be able to do useful things other than just calculations. For example, the only way to be able to print text in *your* terminal from inside the virtual machine is to request the system to do that, and then hope that it does! The host system is under no obligation to do anything, especially if it doesn't seem like a good idea to do!

If you use `printf("Hello world!\n");` in your program, it will likely cause a call to the `write` system call wrapper. This wrapper then does the actual system call itself, which boils down to setting up the arguments and then executing an `ECALL` instruction. At that point the emulator itself will stop executing the virtual machine (the guest) and instead handle this system call. When it's done handling the system call it will continue running the virtual machine. It is completely fine to think of the virtual machine as being paused during system calls. The return value of the system call in register A0 usually indicates whether or not it succeeded or not.

## System call numbers

Linux-specific system call numbers are taken from `linux-headers/include/asm-generic/unistd.h` in the riscv-gnu-toolchain. You can also make up your own system calls, and even how to do a system call (the ABI). [List of system call numbers](https://github.com/riscv-collab/riscv-gnu-toolchain/blob/master/linux-headers/include/asm-generic/unistd.h).

Example:
```
#define __NR_getcwd 17
```
[getcwd()](http://man7.org/linux/man-pages/man2/getcwd.2.html) is system call number 17, and returns the current working directory. You can also run `man getcwd` to get the same page up in a terminal.

## Standard library system calls

When running a RISC-V program in the emulator, you might see messages about unhandled system calls as long as you provide a callback function to the machine by setting `machine.on_unhandled_syscall = my_function;`. Unhandled system calls return `-ENOSYS` by default. To implement missing system calls, you have to set a handler for it.

These are system calls executed by the C and C++ standard libraries, and some of them are not optional. For example, there is no graceful way to stop running a Linux program without implementing `exit` (93) or `exit_group` (94).

If you want to see the stdout output from your hello world, you will also want to implement either `write` or `writev` (depending on the C library).

## Handling a system call

Let's start with an example of handling exit:
```C++
template <int W>
void syscall_exit(Machine<W>& machine)
{
	// Get the first (and only) argument as a 32-bit integer
	auto [exit_code] = machine.template sysargs <int> ();
	// Do something with the exit_code argument
	printf("The machine exited with code: %d\n", exit_code);
	// The exit system call makes the machine stop running
	machine.stop();
}
```
Our exit system call handler extracts the exit status code from the first argument to the system call, prints it to stdout and then stops the machine. Installing the system call handler in a machine is straight-forward:

```C++
machine.install_syscall_handler(93, syscall_exit<RISCV64>);
machine.install_syscall_handler(94, machine.syscall_handlers.at(93));
```
Here we installed a 64-bit system call handler for both `exit` (93) and `exit_group` (94).

```C++
Machine<RISCV64>::install_syscall_handler(93, syscall_exit<RISCV64>);
```
System call handlers are static by design to avoid system call setup overhead when creating many machines.

Stopping the machine in practice just means exiting the `Machine::simulate()` loop.

Be careful about modifying registers during system calls, as it may cause problems
in the simulated program. The program may only be expecting you to modify register A0 (return value) or modifying memory pointed to by a system call argument.

## Handling an advanced system call

If we want stdout from the VM printed in our terminal, we should handle `write`:

```C++
#include <unistd.h>

template <int W>
void syscall_write(Machine<W>& machine)
{
	const auto [fd, address, len] =
		machine.template sysargs <int, address_type<W>, address_type<W>> ();
	// We only accept standard output pipes, for now :)
	if (fd == 1 || fd == 2) {
		char buffer[1024];
		const size_t len_g = std::min(sizeof(buffer), len);
		machine.memory.memcpy_out(buffer, address, len_g);
		// Write buffer to our terminal!
		machine.set_result_or_error(write(fd, buffer, len_g));
		return;
	}
	machine.set_result(-EBADF);
}
```
Here we extract 3 arguments, `int fd, void* buffer, size_t len`, looks familiar? We have to make sure fd is one of the known standard pipes, otherwise the VM could start writing to real files open in the host process!

The return value of a call into a kernel is usually a success or error indication, and the way to set an error is to negate a POSIX error code. Success is often 0, however in this case the return value is the bytes written. To make sure we pass on errno properly, we use the helper function `machine.set_result_or_error()`. It takes care of handling the common error case for us.

## Zero-copy write

`write` can be implemented in a zero-copy manner:

```C++
#include <unistd.h>

template <int W>
void syscall_write(Machine<W>& machine)
{
	const auto [fd, address, len] =
		machine.template sysargs <int, address_type<W>, address_type<W>> ();
	// We only accept standard output pipes, for now :)
	if (fd == 1 || fd == 2) {
		// Zero-copy buffers pointing into guest memory
		riscv::vBuffer buffers[16];
		size_t cnt =
			machine.memory.gather_buffers_from_range(16, buffers, address, len);
		// We could use writev here, but we will just print it instead
		for (size_t i = 0; i < cnt; i++) {
			machine.print(buffers[i].ptr, buffers[i].len);
		}
		machine.set_result(len);
		return;
	}
	machine.set_result(-EBADF);
}
```
`gather_buffers_from_range` will fill an iovec-like array of structs up until the given number of buffers. We can then use that array to print or forward the data without copying anything.

`gather_buffers_from_range` will concatenate sequential parts of guest memory, and very often even just 16 gather-buffers are enough to cover ~99% of cases. This is especially the case if the read-write arena is enabled. One should not think of a single buffer as page-sized, but rather sequential memory up until the next buffer.

## Memory helper methods

A fictive system call that has a single string as system call argument can be implemented in a variety of ways:

1. Retrieve buffer address and length, then copy data into a host buffer.
> Note: This function will throw an exception under all circumstances if it cannot complete successfully. Page permissions apply, but not invalid lengths. If the length is 256GB, `copy_from_guest()` *will* attempt to copy that, and will only fail when running out of memory. All memory operations in _libriscv_ are strictly bounds-checked, but if there really is 256GB of memory in the guest, the copy operation will try to copy all 256GB out.

```C++
template <int W>
void syscall_string(Machine<W>& machine)
{
	const auto [address, len] =
		machine.template sysargs <address_type<W>, address_type<W>> ();

	// Create a buffer and copy into it. Page protections apply.
	std::vector<uint8_t> buffer(len);
	machine.copy_from_guest(buffer.data(), address, len);
}
```

The helpers `machine.copy_from_guest` and `machine.copy_to_guest` work in all configurations, and in all settings. They are, as mentioned, dutifully going to copy every byte you requested, even if it's a large amount of bytes. So remember to check!

2. Fill an array of iovec-like structs with the guest buffer address and length. The buffers will contain host pointers and safe lengths, and can be passed directly to readv/writev.
> Note: This function will throw an exception under all circumstances if it cannot complete successfully. Page permissions apply. The operation is unbounded, meaning that if, for example, we attempt to fill iovec buffers with 32GB of memory, and that memory is sequential in the guest, it only needs 1 iovec entry to represent that, and so it *will* return a single buffer that is 32GB long. It will only fail if there are not enough buffers to represent the entire data.

```C++
template <int W>
void syscall_string(Machine<W>& machine)
{
	const auto [address, len] =
		machine.template sysargs <address_type<W>, address_type<W>> ();

	riscv::vBuffer buffers[16];
	size_t cnt =
		machine.memory.gather_buffers_from_range(16, buffers, address, len);
	const ssize_t res =
		writev(1, (struct iovec *)&buffers[0], cnt);
}
```

3. Directly read a zero-terminated string. Page protections apply.
> Note: This function will throw an exception under all circumstances if it cannot complete successfully. Page permissions apply. The operation is bounded by a second argument `memstring(addr, maxlen)` that limits the operation to by default 16MB. This acts as a preventative measure against invalid strings, and simplifies API usage.
```C++
template <int W>
void syscall_string(Machine<W>& machine)
{
	const auto [address] =
		machine.template sysargs <address_type<W>> ();

	const auto string = machine.memory.memstring(address);
}
```
It uses `strnlen` from your C++ library under the hood, making it very effective. It returns a `std::string`, which benefits from SSO. If allocations are to be avoided, look at `memview` instead.

4. Using `std::string` directly is a shortcut for example 3, shown above. The same rules apply.
```C++
template <int W>
void syscall_string(Machine<W>& machine)
{
	const auto [string] =
		machine.template sysargs <std::string> (); // Consumes 1 register
}
```


5. Resolve an address and a length (2 registers) to a `std::string_view`.
> Note: This function will throw an exception under all circumstances if it cannot complete successfully. Page permissions *do not* apply, but read-write arena rules apply (eg. cannot write to read-only program area). The operation has a default hard 16MB limit (third argument to `memview(addr, len, maxlen)`). This acts as a defensive measure against invalid lengths, and simplifies API usage.
```C++
template <int W>
void syscall_string(Machine<W>& machine)
{
	const auto [address, len] =
		machine.template sysargs <address_type<W>, address_type<W>> ();

	const auto strview = machine.memory.memview(address, len);
}
```
Using a `std::string_view` is only possible when memory is sequential, and requires 2 registers (address and length). It is sequential by default. No memory is copied, making this a preferred and very fast operation.

6. Using `std::string_view` directly is a shortcut for example 5, shown above. The same rules apply.
```C++
template <int W>
void syscall_string(Machine<W>& machine)
{
	const auto [view] =
		machine.template sysargs <std::string_view> (); // Consumes 2 registers
}
```

## Other examples

1. Read a struct by value.
> Note: This function will throw an exception under all circumstances if it cannot complete successfully. Page permissions apply. Uses `memcpy_out(&t, addr, sizeof(T))` behind the scenes.

```C++
template <int W>
void syscall_struct(Machine<W>& machine)
{
	struct MyStruct {
		std::array<int, 44> mydata;
		char buffer[64];
	};
	const auto [mystruct] =
		machine.template sysargs <MyStruct> (); // Consumes 1 register (the address)
}
```

2. Get a pointer `T*` to a struct.
> Note: This function will throw an exception under all circumstances if it cannot complete successfully. Page permissions *do not* apply, instead read-write arena rules apply. Uses `memarray<T> (addr, 1)` behind the scenes.

```C++
template <int W>
void syscall_struct(Machine<W>& machine)
{
	struct MyStruct {
		std::array<int, 44> mydata;
		char buffer[64];
	};
	const auto [mystruct_ptr] =
		machine.template sysargs <MyStruct*> (); // Consumes 1 register (the address)
}
```
Notice how the type is _a pointer_. If instead you want a fixed-size span of T, you can use `std::span<T, N>` or use a pointer to a fixed-size std::array: `std::array<T, N>*`. Not all platforms you might want to support will have span support.

3. Get a dynamic N-element span of struct (`std::span<T>`).
> Note: This function will throw an exception under all circumstances if it cannot complete successfully. Page permissions *do not* apply, instead read-write arena rules apply. Uses `memspan<T> (addr, n)` behind the scenes.

```C++
template <int W>
void syscall_struct(Machine<W>& machine)
{
	struct MyStruct {
		int value;
	};
	const auto [span] =
		machine.template sysargs <std::span<MyStruct>> (); // Consumes 2 registers
}
```

4. Get a pointer to an N-element array of struct (`std::array<T, N>*`).
> Note: This function will throw an exception under all circumstances if it cannot complete successfully. Page permissions *do not* apply, instead read-write arena rules apply. Uses `memarray<T, N> (addr)` behind the scenes.

```C++
template <int W>
void syscall_struct(Machine<W>& machine)
{
	struct MyStruct {
		int value;
	};
	const auto [mystruct_ptr] =
		machine.template sysargs <std::array<MyStruct, 44>*> (); // Consumes 1 register (the address)
}
```
The entire array is guaranteed to be accessible for reading and writing. No memory is copied.


5. Getting writable buffers for `readv()`.

> Note: This function will throw an exception under all circumstances if it cannot complete successfully. Page permissions apply. The function creates writable pages for the entire segment, and returns `cnt` iovec entries that can be passed directly to `readv()`.

```C++
	riscv::vBuffer buffers[16];
	size_t cnt =
		machine.memory.gather_writable_buffers_from_range(16, buffers, address, len);
	const ssize_t res =
		readv(1, (struct iovec *)&buffers[0], cnt);
```

Using `gather_writable_buffers_from_range` we can let the Linux kernel block and read into the guests memory until completion.


## Communicating the other way

While the example above handles a copy from the guest- to the host-system, the other way around is the best way to handle queries. For example, the `getcwd()` function requires passing a buffer and a length:

```C++
// this is normal C++
std::array<char, PATH_MAX> buffer;
char* b = getcwd(buffer.data(), buffer.size());
assert(b != nullptr);

printf("The current working directory is: %s\n", buffer.data());
```

To handle this system call, we will need to copy into the guest:

```C++
#include <unistd.h>

template <int W>
void syscall_getcwd(Machine<W>& machine)
{
	const auto [address, len] =
		machine.template sysargs <address_type<W>, address_type<W>> ();
	// make something up! :)
	const char path[] = "/home/vmguest";
	// we only accept lengths of at least sizeof(path)
	if (len >= sizeof(path)) {
		machine.copy_to_guest(address, path, sizeof(path));
		machine.set_result(address);
		return; // ^ this way will copy the terminating zero as well!
	}
	// for unacceptable values we return null
	machine.set_result(0);
}
```

If in doubt, just use `address_type<W>` for the syscall argument, and it will be the same size as a register, which all system call arguments are anyway.

## The RISC-V system call ABI

On RISC-V a system call has its own instruction: `ECALL`. A system call can have up to 7 arguments and has 1 return value. The arguments are in registers A0-A6, in that order, and the return value is written into A0 before giving back control to the guest. A7 contains the system call number. These are all integer/pointer registers.

For 32-bit, every 64-bit integer argument will use 2 registers. For example, the system call `uint64_t my_syscall(uint64_t a, uint64_t b, uint64_t c)` would use 6 integer registers for its arguments (A0-A5), and 2 return registers (A0, A1).

Floating-point arguments can be in FA0-FA7, however they are rarely (if ever) used for system calls.

To pass larger data around, the guest should allocate buffers of the appropriate size and pass the address of the buffers as arguments to the system call.

## Custom system calls

If you are doing a low-latency implementation of the emulator in eg. a game engine, it's good to know that it is possible to make up your own system calls such that the latency is minimal. This type of latency is so low that it is regularly 5-50x faster than other solutions, however it can sometimes be hard to get right.

Let's take as an example a system call that normalizes a f32 vec2. In order to lower latency, we will take x and y as float arguments in FA0 and FA1, modify them and return them in the same registers.

```C++
template <int W>
void api_vector_normalize(Machine<W>& machine)
{
	auto [dx, dy] = machine.sysargs<float, float>();
	glm::vec2 vec = glm::normalize(glm::vec2(dx, dy));
	machine.set_result(vec.x, vec.y);
}
```

Now, how can you invoke such a system call from your program? It can be quite complicated if you are not used to inline assembly, but let's try. First we need to figure out how to pass floats to a system call. Then, we need to specify that the same registers have been modified during the system call. Let's implement `sys_vec2_normalize`:

```C++
inline Vector2 Vector2::normalized() const noexcept {
	register float x asm("fa0") = this->x;
	register float y asm("fa1") = this->y;
	register int syscall asm("a7") = ECALL_VEC2_NORMALIZED;

	__asm__ volatile("ecall"
					 : "+f"(x), "+f"(y)
					 : "r"(syscall));
	return {x, y};
}
```

What we are doing here is pinning x and y to fa0 and fa1 using the register keyword. We also pin the system call number for this operation to a7. After that we execute a system call using the `ecall` pseudo-instruction. Now, how does just executing an instruction suddenly mean that the registers are set properly at the start, and then how does it know that fa0 and fa1 was modified after?

The answer is in the rest of the inline assembly. The first part is the raw assembly, just `ecall`. Then comes the outputs, then inputs and finally clobbers (which are not present here). So, `ecall` is raw assembly, `"+f"(x), "+f"(y)` are outputs, and `"r"(syscall)` is an input. Shouldn't x and y also be an input? Yes, they are because we do `+f` instead of just `f`. `r` is a general-purpose register, while `f` is a floating-point register. `+f` means that it's an input *and* an output. Basically, it changed after the raw assembly completed.

How does this turn into a low-latency operation? Well, imagine this:
```C++
myvec = myvec.normalized()
```
The compiler will see that `x` and `y` went into `fa0` and `fa1`, and then the registers got modified. After that we assign this result directly into our `x` and `y` again. This means that the compiler will not move anything around in registers at all.

The only way to improve on this is to do more in the system calls. And that is always an option because system calls are free to call. They cost ~2ns to call, and you can pass anything and return anything. Creativity is the only limit here.

What if you don't have performance issues and you just want to do these calculations directly in the program running in the emulator? That is completely fine. RISC-V has dedicated instructions for square-root and division. It's not going to become a bottleneck. The reason I am showing this example is just to hammer home that it is easy to make custom system calls for your own needs, and that you don't even need to use inline assembly - but should you want that chefs-kiss performance, go for it.

## Heavy computations

What if you want to script something complicated like erosion calculations? Can't you do that in the emulator? The answer is yes, you can, but the actual processing should be done in system calls using a library that specifically does those operations using a fast method. This is true regardless of the emulator in question, even gold-standard emulators with "near-native" (*cough*) performance. The reason is that these kinds of heavy calculations often benefit greatly from SIMD operations, which you often get from specialized libraries that does it all for you. Implementing your own slow erosion calculations is only going to create an artificial bottleneck.

That said, portability is always a concern. If you generate embeddable binary translation and activate it, and the performance is acceptable, then that's great. In that case you might also want to avoid too many system calls in the middle of it, as binary translation can be close to native performance within a single function, as long as it doesn't have to leave or jump around too much.


## Special note on EBREAK

The `EBREAK` instruction is handled as a system call in this emulator, specifically it uses the system call number `riscv::SYSCALL_EBREAK`, which at the time of writing is put at the end of the system call table (N-1), however it can be changed in the common header or by setting a global define `RISCV_SYSCALL_EBREAK_NR` to a specific number.

`EBREAK` is very convenient for debugging purposes, as adding it somewhere in the code is very simple: `asm("ebreak");`.

NOTE: Be careful of `__builtin_trap()`, as it erroneously assumes that you are never returning, and the compiler will stop producing instructions after it.

## Special note on read-write arena rules

The read-write arena is by default enabled, and splits the address space into 3 zones: Inaccessible, read-only and read-write. The area before the ELF program becomes inaccessible, such as the zero page (and consequently 0x0). The read-only area of the ELF becomes read-only (rodata sections). And finally, the .data, .bss and the heap becomes read-write until the end of configured memory.

This is purely an optimization, and it maintains the sandbox. It can be disabled in order to get full paging support for the entire address space. When disabled, some helper functions become unavailable as the address space is no longer guaranteed to be sequential, but other opportunities arise in their place. For example, forking is fully supported, native heap allocator that only allocates sequential data, taking over page allocation in order to use custom arenas etc.
