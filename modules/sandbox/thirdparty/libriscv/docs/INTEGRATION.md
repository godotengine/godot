# Integration

Integrating libriscv into your project is fairly straight-forward, but employing it as a low-latency scripting solution is more difficult. There is an example of how to do this in my [RVScript repository](https://github.com/fwsGonzo/rvscript). There is also a simpler version of this in this repostiryo, [the gamedev examples](/examples/gamedev).

This document explains how to integrate _libriscv_ into your projects.

## Embedding libriscv

You add libriscv primarily through CMake. Add it like so:

```sh
mkdir -p ext
cd ext
git submodule add git@github.com:fwsGonzo/libriscv.git
```

Now create a CMakeLists.txt in the ext folder:

```cmake
# libriscv in 64-bit
option(RISCV_32I "" OFF)
option(RISCV_64I "" ON)

add_subdirectory(libriscv/lib)
# We need to make room for our own system calls, as well as
# the classic Linux system calls (0-500). So ours are from 500-600.
target_compile_definitions(riscv PUBLIC RISCV_SYSCALLS_MAX=600)
```

We can now use this ext/CMakeLists.txt from our root CMakeLists.txt:

```cmake
add_subdirectory(ext)

...
target_link_libraries(myprogram riscv)
```

libriscv is now accessible in the code:

```C++
#include <libriscv/machine.hpp>

int main()
{
    riscv::Machine<riscv::RISCV64> machine(binary_vec_u8, options);
}
```

You can see how RVScript does the same thing [here](https://github.com/fwsGonzo/rvscript/blob/master/ext/CMakeLists.txt).

The engine subfolder is adding libriscv [here](https://github.com/fwsGonzo/rvscript/blob/master/engine/CMakeLists.txt#L46).

Note that you can also install libriscv through packaging, eg. `libriscv` on AUR.

## Configuring libriscv

_libriscv_ has good defaults, but has a variety of configuration options that alters its behavior and even performance. Some options are declared experimental, and not appropriate for sandboxing.

_libriscv_ is primarily configured using CMake options:

> RISCV_DEBUG
- Enable extra debugging features, such as verbose jumps. _libriscv_ does not ordinarily need this enabled.

> RISCV_EXT_A
- Enable atomic instructions (A-extension)

> RISCV_EXT_C
- Enable compressed instructions (C-extension)

> RISCV_EXT_V
- Enable vector instructions (V-extension)

> RISCV_32I
- Enable 32-bit RISC-V emulation

> RISCV_64I
- Enable 64-bit RISC-V emulation

> RISCV_128I
- Enable 128-bit RISC-V emulation

> RISCV_FCSR
- Enable floating-point rounding mode emulation, as well as extra NaN-handling.

> RISCV_EXPERIMENTAL
- Enable or reveal experimental features. Extra options revealed must be separately enabled to take effect.

> RISCV_MEMORY_TRAPS
- Enable traps on pages. Pages with traps must have caching disabled, and may not lie inside the memory arena (if enabled). Pages with traps that lie outside of the memory arena can be repeatedly triggered using reads, writes and jumps.

> RISCV_BINARY_TRANSLATION
- Enable high-performance emulation using binary translation.

> RISCV_LIBTCC
- Enable JIT-compilation using libtcc. Binary translation must also be enabled.

> RISCV_LIBTCC_DISTRO_PACKAGE
- When RISCV_LIBTCC is enabled, an option to use libtcc from a distro package is available. When enabled, `libtcc.a` is used directly and must be in the search path. When disabled, a CMake version of libtcc is fetched from a remote Git repository.

> RISCV_FLAT_RW_ARENA
- Enable high-performance memory operations using a flat read-write arena. The guest address space is separated into 4 parts: 1. The area starting at zero up to the beginning of the ELF program is made invalid. 2. The area starting from the ELF to the end of .rodata is made read-only. The .data section and up to the end of the arena is made read+write. And finally, outside of the arena uses virtual paging, where page protections apply.

> RISCV_THREADED
- Enable threaded dispatch, using computed goto. Fastest dispatch method. When threaded and tailcall are both disabled, fall back to switch-based dispatch.

> RISCV_TAILCALL_DISPATCH
- Enable dispatch using musttail. Clang only. Faster than threaded for simple loops, but on real programs it is always a bit slower.

> RISCV_ENCOMPASSING_ARENA
- Create an N-bit address space where all memory operations must reside. All memory accesses outside of this address space is inaccessible.

> RISCV_ENCOMPASSING_ARENA_BITS
- When RISCV_ENCOMPASSING_ARENA is enabled, this option sets the number of bits each memory address has, effectively making up the size of the address space. For example, 32-bits is a 4GB address space, and 30 is a 1GB address space. 32-bits is most likely the fastest setting. The entire address space is mapped out at construction. Address masking is used to avoid bounds-checking and speculation issues. Experimental feature.

The fastest configuration is:
1. Use 32-bit RISC-V for fast instruction dispatch, or 64-bit RISC-V for higher memory bandwidth
2. Disable C-extension, unless your RISC-V programs use it
3. Always enable flat read-write arena
4. Enable experimental + 32-bit encompassing arena
5. Enable binary translation (or use embedded source files)
6. Disable execution timeout (use CPU::simulate_inaccurate)
7. Enable link-time optimization

Although this is the fastest known configuration, one should use the one that is most convenient.

## Machine Options

The Machine constructor has many options, and we will go through each one.

> memory_max
- Set the maximum amount (upper limit) of memory a guest program can consume. Inside this memory a guest program can do anything it wants to, however it may never access memory outside of this area. If you give the guest 8GB of memory, it is possible it will only end up using 100MB. Only memory that is written to will use physical memory on your machine.

> stack_size
- Set the initial stack size for the main thread. This is a simple mmap allocation. Think of it as `stack = machine.memory.mmap_allocate(stack_size)`. It does not extend guest memory, nor does it touch memory.

> load_program
- When enabled, the binary provided to Machine will be loaded as an ELF program. Default: true.

> protect_segments
- When enabled, the protection bits in the ELF segments of a loaded ELF program will be applied to the pages they are loaded to. Default: true.

> allow_write_exec_segment
- Allow loading a segment with write+execute at the same time. When not enabled, any W+E segment will throw an exception, preventing Machine construction. Default: false.

> enforce_exec_only
- Only allow execute-only segments. An executable segment with read- or write-permissions will cause an exception, preventing Machine construction. Default: false.

> ignore_text_section
- Some programs have executable code outside of the .text section, which is unfortunate. Setting this to true allows loading these programs. Default: false.

> verbose_loader
- Verbose logging to stdout when loading a program. Default: false.

> minimal_fork
- When forking a Machine into another, do not loan any pages, leaving the new fork blank. In order for the new machine to work, pages must now be loaned on-demand using callbacks. Default: false.

> use_memory_arena
- Pre-allocate all guest memory using mmap. All pages will be backed by the arena, making guest memory sequential and improving performance. 

> use_shared_execute_segments
- Share matching execute between all machines automatically. Thread-safe. Default: true.

> default_exit_function
- When making calls into the VM, an exit function is created by default that stops the machine. It is possible to override this with your own.

> page_fault_handler
- A callback which gets called when the Machine needs memory for a certain address. This facilitates sharing, custom arenas, avoiding zeroing memory and so on. A default page fault handler is normally created that constructs pages backed by the arena (if enabled) or simple memory allocations.

> translate_enabled
- Binary translation yields performance improvements to individual execute segments. For example, a dynamic executable might have 3 execute segments (1. the dynamic linker, 2. libc, 3. your program). When _libriscv_ is configured with binary translation, and whenever a new execute segment is about to be executed on, it will try to look for an existing translation first. In order to match with an existing translation, the hash of the translation must match both the emulators configuration and the execute segment it was produced from. When enabled, _libriscv_ will use binary translation according to these rules:
1. If `translate_enable_embedded` is enabled, and embedded binary translation has self-registered, use this first, if there is a matching execute segment hash. This translation is never loaded in the background, and is applied instantly. It is the most efficient translation, anIfd supports all platforms (even those without dynamic linking).
2. If no embedded translation is found, attempt to load a translation from a shared object. This is done by checking the file system for a filename built from `translation_prefix` and `translation_suffix`. Once found, it is dynamically loaded and applied. It is applied instantly and is never loaded in the background.
3. If no translation was loaded and libtcc is enabled, perform binary translation using libtcc right now.
4. If `translate_invoke_compiler` is enabled, and there are no translations to be found, one can be generated using a system compiler. This is done by compiling the C99 binary translation using the CC environment variable. After the compilation finishes, it will be loaded and applied.
5. If `translate_background_callback` is set, background compilation can be performed from the user-provided callback. After background compilation is completed, the results are loaded and live-patched in a thread-safe manner.
6. If `translation_cache` is enabled, the final shared object will be kept in the file system, so that it may be reused later. Default: true

> translate_trace
- When enabled, trace information is generated during binary translation execution. Very spammy. Default: false

> translate_timing
- When enabled, verbose timing information will be printed to stdout during the binary translation process, showing the time spent in each sub-system. Default: false

> translation_use_arena
- When enabled, the binary translator will make use of the memory arena. Default: true

> translate_ignore_instruction_limit
- When enabled, instruction counting is not performed during binary translation, and execution can only stop using another external method. This slightly improves performance. Default: false

> translate_use_register_caching
- When enabled, Machine registers will be put into local stack variables in the binary translation, and loaded and stored more efficiently than unoptimized code. This improves code compiled with -O0, or code produced using simpler compilers like TCC. Default: Enabled with libtcc, otherwise disabled.

> cross_compile
- A vector of cross-compilation methods. Each method is invoked during binary translation, as needed. If an output already exists, skip. A method can be to produce embeddable source files, while another method can be a cross-compiler invocation. Windows-compatible MinGW .dll's can be cross-compiled from Linux.


## Compiling a RISC-V program

Note that compiling your own RISC-V compiler is completely optional. _libriscv_ is fully compatible with any local RISC-V compilers in your packaging system, and compatible with most if not all systems languages (C/C++, Zig, Rust, ...).

Further, just using your distributions local RISC-V cross-compiler is recommended. If, however, you want to compile your own RISC-V toolchain [have a look at our guide](/docs/NEWLIB.md).

Your distributions RISC-V cross-compiler is typically installed like this:

```sh
sudo apt install gcc-12-riscv64-linux-gnu g++-12-riscv64-linux-gnu
```

## Compiling a basic program

Using this simple C program:
```C
#include <stdio.h>
#define STR(x) #x

__attribute__((used, retain))
int my_function(int arg)
{
	printf("Hello " STR(__FUNC__) " World! Arg=%d\n", arg);
	return arg;
}
int main()
{
	printf("Hello World!\n");
}
```

We can compile it like so:
```sh
riscv64-linux-gnu-gcc-12 -static -O2 myprogram.cpp -o myprogram
```

We generally compile statically, in order for everything (all dependencies) to be available to us inside the program. The program will be self-contained. Although dynamic executables are supported, some whitelisting is needed in order to allow the sandbox to dynamically load and link shared libraries. This is why we prefer static linking over other mechanisms.

Now we can run through `main()` and we can also make a function call to `my_function`:

```C++
#include <libriscv/machine.hpp>

int main()
{
	// Create 64-bit RISC-V machine using loaded program
    riscv::Machine<riscv::RISCV64> machine(binary_vec_u8);

	// Add POSIX system call interfaces (no filesystem or network access)
	machine().setup_linux_syscalls(false, false);
	machine().setup_posix_threads();

	// setup program argv *after* setting new stack pointer
	machine().setup_linux({"my_program", "arg0"}, {"LC_ALL=C"});

	// Run through main()
	try {
		machine().simulate();
	} catch (const std::exception& e) {
		fprintf(stderr, "Exception: %s\n", e.what());
	}

	// Call a function (as long as it's in the symbol table)
	int ret = machine().vmcall("my_function", 123);

	// Forward return value from function
	return ret;
}
```

Note: If you strip the program, you cannot call even retained functions. Use a linker option to strip all symbols except the ones you care about instead from a text file: `-Wl,--retain-symbols-file=symbols.txt`. Alternatively, only strip debug symbols. Debug information is often the largest contributor to file size.

# Simple system-call implementation

For most people, just using a simple system call scheme that doesn't require much scaffolding will be good enough. So, have a look at the [gamedev example](/examples/gamedev) where this is done.


# Advanced features

These features are already implemented in [RVScript](https://github.com/fwsGonzo/rvscript), but I am briefly detailing how it works and how to implement it here.

## Dynamic calls (guest-side)

In order for the script to be useful we can't only focus on making function calls into the sandboxed program. We also want to make calls from the program and back into the host (eg. game engine) in order to ask for stuff, or ask the game engine to do something. For example to create a timer.

Dynamic calls are an integral part of a low-friction scripting framework, but they require a bit of work to integrate. The best way to understand how they are generated and then used in the script, is to read the code:

The [python script](https://github.com/fwsGonzo/rvscript/blob/master/programs/dyncalls/generate.py) that reads [dynamic_calls.json](https://github.com/fwsGonzo/rvscript/blob/master/programs/dynamic_calls.json) and outputs callable functions and inline assembly variants.

In order to re-generate the API every time dynamic_calls.json is changed, we use a simple call to [add_dependencies()](https://github.com/fwsGonzo/rvscript/blob/master/programs/micro/micro.cmake#L58)

In order to rebuild the program each time the API changes, we add the [generated sources to the build list](https://github.com/fwsGonzo/rvscript/blob/master/programs/micro/micro.cmake#L51-L56). Notice how they are explicitly marked as GENERATED.

Once all the sources are generated, the dynamic call API can be [included in the guest programs](https://github.com/fwsGonzo/rvscript/blob/master/programs/micro/api/api.h#L7).

And finally, we can [use all the dynamic calls](https://github.com/fwsGonzo/rvscript/blob/master/programs/micro/api/api_impl.h#L149) we specified in the JSON file. By using, I mean implementing a helper wrapper function in the program running inside the sandbox.

The dynamic call python script will generate the exact function written down. For example, for creating a timer the dynamic call signature is `"int sys_timer_periodic (float, float, timer_callback, void*, size_t)"`. That means, inside the sandbox you can now use `sys_timer_periodic`, however it's not a nice API on its own. Let's write a helper function for it:

```C++
#include <dyncall_api.h> // timer_callback, sys_timer_periodic

struct Timer {
	using TimerCallback = void (*)(Timer);

	/// @brief Create a timer that calls a function after the given seconds,
	/// then periodically gets called again after the given period (also in seconds).
	static Timer periodic(float seconds, float period, TimerCallback callback)
	{
		return {sys_timer_periodic(seconds, period, [] (int id, void* cb) {
			// Cast cb pointer to our callback type, and construct a timer from the ID as arg
			((TimerCallback)cb)(Timer(id));
		}, callback, sizeof(callback))};
	}

	int id;
};
```
Using this tiny wrapper and without any fancy std::function-like types we have created a wrapper for timer creation. We can now use it like so:

```C++
auto t = Timer::periodic(5.0f, [] (Timer t) {
	print("Hello from timer ", t.id, "!\n");
});

```
Also, using the periodic wrapper function we can create many more helper functions, like oneshot timers:

```C++
struct Timer {
	using TimerCallback = void (*)(Timer);

	static Timer periodic(float seconds, float period, TimerCallback callback) { ... }

	static Timer periodic(float period, TimerCallback callback) {
		return periodic(0.0f, seconds, callback);
	}

	static Timer oneshot(float seconds, TimerCallback callback) {
		return periodic(seconds, 0.0f, callback);
	}

	int id;
};
```
Now we have a decent Timer API inside the sandbox.


## Implementing a dynamic call handler (host-side)

This part is intentionally very low-friction. Adding dynamic calls means assigning a callback to a (string) function definition:

```C++
	Script::set_dynamic_calls({
		{"Timer::stop", "void sys_timer_stop (int)",
		 [](Script& script)
		 {
			 // Stop timer
			 const auto [timer_id] = script.machine().sysargs<int>();
			 timers.stop(timer_id);
		 }},
		{"Timer::periodic", "int sys_timer_periodic (float, float, timer_callback, void*, size_t)",
		 [](Script& script)
		 {
			 // Periodic timer
			 auto& machine = script.machine();
			 const auto [time, peri, addr, data, size]
				 = machine.sysargs<float, float, gaddr_t, gaddr_t, gaddr_t>();

			 auto capture = CaptureStorage::get(machine, data, size);

			 int id = timers.periodic(
				 time, peri,
				 [addr = (gaddr_t)addr, capture, script = &script](int id)
				 {
					 script->call(addr, id, capture);
				 });
			 machine.set_result(id);
		 }},
	});
```
The friendler `Timer::stop` and `Timer::periodic` is only used when an exception happens in order to make errors more readable.


# Safety and predictability

Dynamic call implementations in the host and the table in the guest program identify each others only using the function definition strings (and only that): `"void sys_timer_stop (int)"` and `"int sys_timer_periodic (float, float, timer_callback, void*, size_t)"`, in this case.

If any of the definitions change, they will no longer find each other, and you will be notified if anyone tries to call an unhandled dynamic call. So if there is a mismatch in the definitions between the program and the host engine, they won't be able to see each others, but you will be able to see what they are trying to do when it fails.

It is designed this way to catch:

- Mismatching arguments, even mismatching argument names
- Being able to write out which functions are missing/unimplemented
- If an exception is called when handling a dynamic call, we can print the name and the definition
- Avoid collisions

It's definitely a very verbose API, however that pays off when integrating this and when debugging later on.
