/**************************************************************************/
/*  sandbox_exception.cpp                                                 */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "sandbox.h"

#include "core/error/error_macros.h"
#include "core/string/print_string.h"
#include "cpp/script_cpp.h"
#include "elf/script_elf.h"
#include <charconv>
#ifdef __linux__
#include <libriscv/rsp_server.hpp>

#include <unistd.h>
static const uint16_t RSP_PORT = 2159;

static const char *getenv_with_default(const char *str, const char *defval) {
	const char *value = getenv(str);
	if (value)
		return value;
	return defval;
}
#endif

static constexpr bool VERBOSE_EXCEPTIONS = false;

static inline String to_hex(gaddr_t value) {
	char str[20] = { 0 };
	char *end = std::to_chars(std::begin(str), std::end(str), value, 16).ptr;
	return String::utf8(str, int64_t(end - str));
}

void Sandbox::handle_exception(gaddr_t address) {
	riscv::Memory<RISCV_ARCH>::Callsite callsite = machine().memory.lookup(address);
	// If the callsite is not found, try to use the cache to find the address
	if (callsite.address == 0x0) {
		callsite.address = address;
		auto it = m_lookup.find(address);
		if (it != m_lookup.end()) {
			const auto u8str = it->second.name.utf8();
			callsite = riscv::Memory<RISCV_ARCH>::Callsite{
				.name = std::string(u8str.ptr(), u8str.length()),
				.address = it->second.address,
				.offset = 0x0,
				.size = 0
			};
		}
	}
	print_line(String("[") + get_name() + "] Exception when calling:\n  " + String(callsite.name.c_str()) + " (0x" + to_hex(callsite.address) + ")\nBacktrace:");

	this->m_exceptions++;
	Sandbox::m_global_exceptions++;

	if (m_machine->memory.binary().empty()) {
		ERR_PRINT("No binary loaded. Remember to assign a program to the Sandbox!");
		return;
	}

	this->print_backtrace(address);

	try {
		throw; // re-throw
	} catch (const riscv::MachineTimeoutException &e) {
		this->handle_timeout(address);
		return; // NOTE: might wanna stay
	} catch (const riscv::MachineException &e) {
		const String instr(machine().cpu.current_instruction_to_string().c_str());
		const String regs(machine().cpu.registers().to_string().c_str());

		print_line(String("\nException: ") + e.what() + "  (data: " + to_hex(e.data()) + ")\n>>> " + instr + "\n>>> Machine registers:\n[PC\t" + to_hex(machine().cpu.pc()) + "] " + regs + "\n");
	} catch (const std::exception &e) {
		print_line(String("\nMessage: ") + e.what() + "\n\n");
		ERR_PRINT(("Exception: " + std::string(e.what())).c_str());
	}

	String elfpath = "";
#if defined(__linux__) || defined(__APPLE__)
	// Docker container debugging functionality has been removed
	// Source line information is no longer available through Docker
	Ref<ELFScript> script = this->get_program();
	if (!script.is_null()) {
		elfpath = get_program()->get_path();
		print_line("Exception in Sandbox - source line debugging unavailable (Docker support removed)");
	}
#endif

	if constexpr (VERBOSE_EXCEPTIONS) {
		print_line(String("Program page: ") + machine().memory.get_page_info(machine().cpu.pc()).c_str());
		print_line(String("Stack page: ") + machine().memory.get_page_info(machine().cpu.reg(2)).c_str());
	}

#ifdef __linux__
	if (getenv("GDB")) {
		const bool oneFrameUp = false;
		const uint16_t port = RSP_PORT;

		if (0 == fork()) {
			char scrname[64];
			strncpy(scrname, "/tmp/dbgscript-XXXXXX", sizeof(scrname));
			const int fd = mkstemp(scrname);
			if (fd < 0) {
				throw std::runtime_error("Unable to create script for debugging");
			}

			const std::string debugscript =
					// Delete the script file (after GDB closes it)
					"shell unlink " + std::string(scrname) + "\n"
															 // Load the original file used by the script
															 "file " +
					std::string(getenv("GDB")) + "\n"
												 // Connect remotely to the given port @port
												 "target remote localhost:" +
					std::to_string(port) + "\n"
										   // Enable the fancy TUI
										   "layout next\nlayout next\n"
										   // Disable pagination for the message
										   "set pagination off\n"
										   // Print the message given by the caller
										   "echo Remote debugging session started\n" +
					"\n"
					// Go up one step from the syscall wrapper (which can fail)
					+ std::string(oneFrameUp ? "up\n" : "");

			ssize_t len = write(fd, debugscript.c_str(), debugscript.size());
			if (len < (ssize_t)debugscript.size()) {
				throw std::runtime_error(
						"Unable to write script file for debugging");
			}
			close(fd);

			const char *argv[] = { getenv_with_default("GDBPATH", "/usr/bin/gdb-multiarch"), "-x",
				scrname, nullptr };
			// XXX: This is not kosher, but GDB is open-source, safe and let's not
			// pretend that anyone downloads gdb-multiarch from a website anyway.
			// There is a finite list of things we should pass to GDB to make it
			// behave well, but I haven't been able to find the right combination.
			extern char **environ;
			if (-1 == execve(argv[0], (char *const *)argv, environ)) {
				throw std::runtime_error(
						"Unable to start gdb-multiarch for debugging");
			}
		} // child

		riscv::RSP<RISCV_ARCH> server{ this->machine(), port };
		auto client = server.accept();
		if (client != nullptr) {
			printf("GDB connected\n");
			// client->set_verbose(true);
			while (client->process_one())
				;
		}
	} // if getenv("GDB")
#endif
} // handle_exception()

void Sandbox::handle_timeout(gaddr_t address) {
	this->m_timeouts++;
	Sandbox::m_global_timeouts++;
	auto callsite = machine().memory.lookup(address);
	print_line(String("Sandbox: Timeout for '") + callsite.name.c_str() + "' (Timeouts: " + itos(m_timeouts) + ")\n");
}

void Sandbox::print_backtrace(const gaddr_t addr) {
	machine().memory.print_backtrace(
			[](std::string_view line) {
				String line_str(std::string(line).c_str());
				print_line("-> " + line_str);
			});
	auto origin = machine().memory.lookup(addr);
	String name(origin.name.c_str());
	print_line("-> [-] 0x" + to_hex(origin.address) + " + 0x" + to_hex(origin.offset) + ": " + name);
}
