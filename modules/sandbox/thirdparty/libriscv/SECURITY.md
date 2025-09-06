# Security Guarantees

libriscv provides a safe sandbox that guests can not escape from, short of vulnerabilities in custom system calls installed by the host. This includes the virtual machine and the native helper libraries. Do not use binary translation in production at this time. Do not use linux filesystem or socket system calls in production at this time.

libriscv provides termination guarantees and default resource limits - code should not be able to exhaust CPU or RAM resources on the system during initialization or execution. If blocking calls are used during system calls, use socket timeouts or timers + signals to cancel.

libriscv can be loaded with any program from any untrusted source. Any program or combination of instructions that is able to violate the sandbox integrity, including high resource usage, is considered a vulnerabilty.

# Safe Configuration

In order for libriscv to operate as a proper sandbox, one must adhere to these strict rules:

- The filesystem and network socket features need to be disabled.
  - For the C++ API this applies only if you call `machine.setup_linux_syscalls(true, true);`. Set each argument to false in order to disable files and sockets respectively. If you do not need Linux system calls, simply do not call `setup_linux_syscalls` at all, and consider the lighter variants instead.
  - For the C API the strict_sandbox mode should be enabled, which disables files and sockets.
- When instantiating a machine, make sure the max memory limit is suitably low.
- When initializing a program or calling a function, use a low instruction limit.
- Disable automatic printing to stdout using `machine.set_printer()`.
- Handle unknown system calls (which by default prints to stdout) with an empty callback.
- Consider replacing the default `rdtime` callback with your own that uses milliseconds or even tens of milliseconds instead of the default microsecond granularity.

Example configuration:
```c++
	using machine_t = riscv::Machine<MARCH>;

	riscv::MachineOptions<MARCH> options {
		.memory_max = MAX_MEMORY, .stack_size = STACK_SIZE
	};
	m_machine.reset(new machine_t(m_binary, options));
	machine().setup_argv({"my_program"});

	machine().set_printer([](const machine_t&, const char* p, size_t len) {
		// do nothing
	});
	machine().on_unhandled_syscall = [](machine_t& machine, size_t num) {
		// do nothing
	}

	// MAX_INSTRUCTIONS should be suitably low
	machine().simulate(MAX_INSTRUCTIONS);
```

After this setup, one can start adding system calls, but only as needed.

# Reporting a Vulnerability

You can report security bugs to fwsGonzo directly at fwsgonzo at hotmail dot com.
