Debugging with libriscv
================

Debugging with *libriscv* can be as complex or simplistic as one wants, depending on what you have to work with. If you don't have any symbols, and you have advanced knowledge of RISC-V you can step through the program instruction by instruction. This was the method used when developing the RISC-V emulator, but it's not always super helpful when debugging a normal mostly-working program in a robust emulator.

There are three main methods to debugging. One is using the built-in debugging facilities. Another is stepping through the program yourself manually, and checking for any conditions you are interested in, using the emulators state. And the third option is connecting with GDB remotely, which is what you are after if you are just debugging a normal program. Importantly, GDB gives you good introspection of the environment when using any modern language with debuginfo support.

## Debugging with the emulator itself

This method is platform-independent and works everywhere, but you will want to do it from a terminal so you can type commands. It allows you to step through the program instruction by instruction and trap on execute, reads and writes. Doing that, however, requires you to program this behavior. libriscv is fundamentally a library with a flexible, programmable emulator.

You can enable the RISCV_DEBUG CMake option if you want, but it is entirely optional. It will enable extra debugging features in the machine.

```C++
	DebugMachine debug { machine };
	// Print all instructions one by one
	debug.verbose_instructions = true;
	// Break immediately
	debug.print_and_pause();

	try {
		debug.simulate();
	} catch (riscv::MachineException& me) {
		printf(">>> Machine exception %d: %s (data: 0x%lX)\n",
				me.type(), me.what(), long(me.data()));
		debug.print_and_pause();
	} catch (std::exception& e) {
		printf(">>> General exception: %s\n", e.what());
		debug.print_and_pause();
	}
```
An example of how to use the built-in CLI to step through instruction by instruction.

## Debugging manually with libriscv

By simulating a single instruction using `CPU::step_one()` we can programmatically apply any conditions we want:

```C++
DebugMachine debug { machine };

debug.simulate([] (auto& debug) {
	if (debug.machine.cpu.reg(10) == 0x1234)
		debug.print_and_pause();
});

```
This will step through the code one instruction at a time until register A0 is 0x1234, then break into the debugging CLI and print location and registers.

## Debugging remotely with GDB

To get a GDB capable of debugging RISC-V you will need to install your distros equivalent of `gdb-multiarch`. It will have RISC-V support built in, and it will detect the architecture of ELF programs you load in it.

```
sudo apt install gdb-multiarch
```

Once you have a RISC-V aware GDB you can start it with `gdb-multiarch my.elf`. Once inside GDB execute `target remote :2159` to connect when the emulator is waiting for a debugger.

```C++
#include <libriscv/rsp_server.hpp>
...
template <int W>
void gdb_listen(uint16_t port, Machine<W>& machine)
{
	printf("GDB server is listening on localhost:%u\n", port);
	riscv::RSP<W> server { machine, port };
	auto client = server.accept();
	if (client != nullptr) {
		printf("GDB connected\n");
		while (client->process_one());
	}
	// Finish the *remainder* of the program
	if (!machine.stopped())
		machine.simulate(/* machine.max_instructions() */);
}
```

Remember to build your RISC-V program with `-ggdb3 -O0`, otherwise you will not get complete information during the debugging session.

The RSP client will automatically feed new instruction limits to the machine when you type continue in GDB. The limit is still in place in order for you to not lose interactivity with GDB, in case there are infinite loops.

## Debugging remotely using program breakpoints

One powerful option is opening up for a remote debugger on-demand. To do this you need to implement a system call that simply does what the previous section does: Opening up a port for a remote debugger. The difference is that you do it during the system call, so that you can debug things like failed assertions and other should-not-get-here things. You can open up a debugger under any condition. GDB will resume from where the program stopped.

In other words, call the `gdb_listen` function above during anytime you want to have a look at what's going on.

The most likely system call candidate for this behavior is for handling EBREAK interruptions. The emulator defines `RISCV_SYSCALL_EBREAK_NR` by default to `RISCV_SYSCALLS_MAX-1`, but it can be overridden. Reaching EBREAK is always handled as a system call in the emulator.

To avoid having to repeat yourself, create a GDB script to automatically connect and enter TUI mode:
```
target remote :2159
layout next
```
Then run `gdb -x myscript.gdb`.

Good luck!
