#include <libriscv/machine.hpp>
#include <libriscv/debug.hpp>
#include <libriscv/rsp_server.hpp>
#include <inttypes.h>
#include <chrono>
#include <thread>
#include "settings.hpp"
#if __has_include(<unistd.h>)
#include <fcntl.h>
#endif
#if defined(__linux__)
#include <sys/stat.h>
#include <sys/mman.h>
#endif
static inline std::vector<uint8_t> load_file(const std::string &);
static constexpr uint64_t MAX_MEMORY = (riscv::encompassing_Nbit_arena == 0) ? uint64_t(4000) << 20 : uint64_t(1) << riscv::encompassing_Nbit_arena;
static const std::string DYNAMIC_LINKER = "/usr/riscv64-linux-gnu/lib/ld-linux-riscv64-lp64d.so.1";
//#define NODEJS_WORKAROUND

struct Arguments {
	bool verbose = false;
	bool quit = false;
	bool accurate = false;
	bool debug = false;
	bool singlestep = false;
	bool gdb = false;
	bool silent = false;
	bool timing = false;
	bool trace = false;
	bool no_translate = false;
	bool translate_regcache = riscv::libtcc_enabled; // Default: Register caching w/libtcc
	bool translate_future = true;
	bool mingw = false;
	bool from_start = false;
	bool sandbox = false;
	bool execute_only = false;
	bool ignore_text = false;
	bool background = riscv::libtcc_enabled; // Run binary translation in background thread
	bool proxy_mode = false;  // Proxy mode for system calls
	uint64_t fuel = 30'000'000'000ULL; // Default: Timeout after ~30bn instructions
	uint64_t max_memory = 0;
	std::vector<std::string> allowed_files;
	std::string output_file;
	std::string call_function;
	std::string jump_hints_file;
};

#ifdef HAVE_GETOPT_LONG
#include <getopt.h>

static const struct option long_options[] = {
	{"help", no_argument, 0, 'h'},
	{"verbose", no_argument, 0, 'v'},
	{"quit", no_argument, 0, 'Q'},
	{"accurate", no_argument, 0, 'a'},
	{"debug", no_argument, 0, 'd'},
	{"single-step", no_argument, 0, '1'},
	{"fuel", required_argument, 0, 'f'},
	{"memory", required_argument, 0, 'm'},
	{"gdb", no_argument, 0, 'g'},
	{"silent", no_argument, 0, 's'},
	{"timing", no_argument, 0, 't'},
	{"trace", no_argument, 0, 'T'},
	{"no-translate", no_argument, 0, 'n'},
	{"no-translate-future", no_argument, 0, 'N'},
	{"translate-regcache", no_argument, 0, 'R'},
	{"no-translate-regcache", no_argument, 0, 1000},
	{"jump-hints", required_argument, 0, 'J'},
	{"background", no_argument, 0, 'B'},
	{"no-background", no_argument, 0, 1001},
	{"mingw", no_argument, 0, 'M'},
	{"output", required_argument, 0, 'o'},
	{"from-start", no_argument, 0, 'F'},
	{"sandbox", no_argument, 0, 'S'},
	{"proxy", no_argument, 0, 'P'},
	{"allow", required_argument, 0, 'A'},
	{"execute-only", no_argument, 0, 'X'},
	{"ignore-text", no_argument, 0, 'I'},
	{"call", required_argument, 0, 'c'},
	{0, 0, 0, 0}
};

static void print_help(const char* name)
{
	printf("Usage: %s [options] <program> [args]\n", name);
	printf("Options:\n"
		"  -h, --help         Print this help message\n"
		"  -v, --verbose      Enable verbose loader output\n"
		"  -Q, --quit         Quit after loading the program (to produce eg. binary translations)\n"
		"  -a, --accurate     Accurate instruction counting\n"
		"  -d, --debug        Enable CLI debugger\n"
		"  -1, --single-step  One instruction at a time, enabling exact exceptions\n"
		"  -f, --fuel amt     Set max instructions until program halts\n"
		"  -m, --memory amt   Set max memory size in MiB (default: 4096 MiB)\n"
		"  -g, --gdb          Start GDB server on port 2159\n"
		"  -s, --silent       Suppress program completion information\n"
		"  -t, --timing       Enable timing information in binary translator\n"
		"  -T, --trace        Enable tracing in binary translator\n"
		"  -n, --no-translate Disable binary translation\n"
		"  -N, --no-translate-future Disable binary translation of non-initial segments\n"
		"  -R, --translate-regcache Enable register caching in binary translator\n"
		"      --no-translate-regcache Disable register caching in binary translator\n"
		"  -J, --jump-hints file  Load jump location hints from file, unless empty then record instead\n"
		"  -B  --background   Run binary translation in background w/live-patching\n"
		"      --no-background Disable background binary translation\n"
		"  -M, --mingw        Cross-compile for Windows (MinGW)\n"
		"  -o, --output file  Output embeddable binary translated code (C99)\n"
		"  -F, --from-start   Start debugger from the beginning (_start)\n"
		"  -S  --sandbox      Enable strict sandbox\n"
		"  -P, --proxy        Enable proxy mode, allowing access to all files (disabling the sandbox)\n"
		"  -A, --allow file   Allow file to be opened by the guest\n"
		"  -X, --execute-only Enforce execute-only segments (no read/write)\n"
		"  -I, --ignore-text  Ignore .text section, and use segments only\n"
		"  -c, --call func    Call a function after loading the program\n"
		"\n"
	);
	printf("libriscv v%d.%d is compiled with:\n"
#ifdef RISCV_32I
		"-  32-bit RISC-V support (RV32GB)\n"
#endif
#ifdef RISCV_64I
		"-  64-bit RISC-V support (RV64GB)\n"
#endif
#ifdef RISCV_128I
		"-  128-bit RISC-V support (RV128G)\n"
#endif
#ifdef RISCV_EXT_A
		"-  A: Atomic extension is enabled\n"
#endif
#ifdef RISCV_EXT_C
		"-  C: Compressed extension is enabled\n"
#endif
#ifdef RISCV_EXT_V
		"-  V: Vector extension is enabled\n"
#endif
#if defined(RISCV_BINARY_TRANSLATION) && defined(RISCV_LIBTCC)
		"-  Binary translation is enabled (libtcc)\n"
#elif defined(RISCV_BINARY_TRANSLATION)
		"-  Binary translation is enabled\n"
#endif
#ifdef RISCV_DEBUG
		"-  Extra debugging features are enabled\n"
#endif
#ifdef RISCV_FLAT_RW_ARENA
		"-  Flat sequential memory arena is enabled\n"
#endif
#ifdef RISCV_ENCOMPASSING_ARENA_BITS
#define _STR(x) #x
#define STR(x) _STR(x)
		"-  " STR(RISCV_ENCOMPASSING_ARENA_BITS) "-bit masked address space is enabled (experimental)\n"
#endif
		"\n",
		RISCV_VERSION_MAJOR, RISCV_VERSION_MINOR
	);
}

static int parse_arguments(int argc, const char** argv, Arguments& args)
{
	int c;
	while ((c = getopt_long(argc, (char**)argv, "hvQad1f:m:gstTnNRJ:BMo:FSPA:XIc:", long_options, nullptr)) != -1)
	{
		switch (c)
		{
			case 'h': print_help(argv[0]); return 0;
			case 'v': args.verbose = true; break;
			case 'Q': args.quit = true; break;
			case 'a': args.accurate = true; break;
			case 'd': args.debug = true; break;
			case '1': args.singlestep = true; break;
			case 'f': break;
			case 'g': args.gdb = true; break;
			case 's': args.silent = true; break;
			case 't': args.timing = true; break;
			case 'T': args.trace = true; break;
			case 'n': args.no_translate = true; break;
			case 'N': args.translate_future = false; break;
			case 'R': args.translate_regcache = true; break;
			case 'J': break;
			case 'B': args.background = true; break;
			case 'M': args.mingw = true; break;
			case 'o': break;
			case 'F': args.from_start = true; break;
			case 'S': args.sandbox = true; break;
			case 'P': args.proxy_mode = true; break;
			case 'A': args.allowed_files.push_back(optarg); break;
			case 'X': args.execute_only = true; break;
			case 'I': args.ignore_text = true; break;
			case 1000: args.translate_regcache = false; break;
			case 1001: args.background = false; break;
			case 'm': // --memory
				if (optarg) {
					char* endptr;
					args.max_memory = strtoull(optarg, &endptr, 10);
					if (*endptr != '\0' || args.max_memory == 0) {
						fprintf(stderr, "Invalid memory size: %s\n", optarg);
						return -1;
					}
					args.max_memory <<= 20; // Convert MiB to bytes
				} else {
					fprintf(stderr, "Memory size must be specified\n");
					return -1;
				}
				break;
			case 'c': break;
			default:
				fprintf(stderr, "Unknown option: %c\n", c);
				return -1;
		}

		if (c == 'f') {
			char* endptr;
			args.fuel = strtoull(optarg, &endptr, 10);
			if (*endptr != '\0') {
				fprintf(stderr, "Invalid number: %s\n", optarg);
				return -1;
			}
			if (args.fuel == 0) {
				args.fuel = UINT64_MAX;
				args.accurate = false; // It will run forever anyway
			} else {
				args.accurate = true;
			}
			if (args.verbose) {
				printf("* Fuel set to %" PRIu64 "\n", args.fuel);
			}
		} else if (c == 'o') {
			args.output_file = optarg;
			if (args.verbose) {
				printf("* Output file prefix set to %s\n", args.output_file.c_str());
			}
		} else if (c == 'c') {
			args.call_function = optarg;
			if (args.verbose) {
				printf("* Function to VMCall: %s\n", args.call_function.c_str());
			}
		} else if (c == 'J') {
			args.jump_hints_file = optarg;
			if (args.verbose) {
				printf("* Jump hints file: %s\n", args.jump_hints_file.c_str());
			}
		}
	}

	if (optind >= argc) {
		print_help(argv[0]);
		return -1;
	}

	return optind;
}

#endif

template <int W>
static void run_sighandler(riscv::Machine<W>&);

template <int W>
static void run_program(
	const Arguments& cli_args,
	const std::string_view binary,
	const bool is_dynamic,
	const std::vector<std::string>& args)
{
	if (cli_args.mingw && (!riscv::binary_translation_enabled || riscv::libtcc_enabled)) {
		fprintf(stderr, "Error: Full binary translation must be enabled for MinGW cross-compilation\n");
		exit(1);
	}

	std::vector<riscv::MachineTranslationOptions> cc;
	if (cli_args.mingw) {
		cc.push_back(riscv::MachineTranslationCrossOptions{});
	}
	if (!cli_args.output_file.empty()) {
		cc.push_back(riscv::MachineTranslationEmbeddableCodeOptions{cli_args.output_file});
	}

	auto options = std::make_shared<riscv::MachineOptions<W>>(riscv::MachineOptions<W>{
		.memory_max = cli_args.max_memory ? cli_args.max_memory : MAX_MEMORY,
		.enforce_exec_only = cli_args.execute_only,
		.ignore_text_section = cli_args.ignore_text,
		.verbose_loader = cli_args.verbose,
		.use_shared_execute_segments = false, // We are only creating one machine, disabling this can enable some optimizations
#ifdef NODEJS_WORKAROUND
		.ebreak_locations = {
			"pthread_rwlock_rdlock", "pthread_rwlock_wrlock" // Live-patch locations
		},
#endif
#ifdef RISCV_BINARY_TRANSLATION
		.translate_enabled = !cli_args.no_translate,
		.translate_future_segments = cli_args.translate_future,
		.translate_trace = cli_args.trace,
		.translate_timing = cli_args.timing,
		.translate_ignore_instruction_limit = !cli_args.accurate, // Press Ctrl+C to stop
		.translate_use_register_caching = cli_args.translate_regcache,
		.translate_automatic_nbit_address_space = false,
		.translate_unsafe_remove_checks = cli_args.proxy_mode, // Proxy mode disabled sandboxing
		.record_slowpaths_to_jump_hints = !cli_args.jump_hints_file.empty(),
#ifdef _WIN32
		.translation_prefix = "translations/rvbintr-",
		.translation_suffix = ".dll",
#else
		.translator_jump_hints = load_jump_hints<W>(cli_args.jump_hints_file, cli_args.verbose),
		.translate_background_callback = cli_args.background ?
			[] (auto& compilation_step) {
				std::thread([compilation_step = std::move(compilation_step)] {
					compilation_step();
				}).detach();
			} : std::function<void(std::function<void()>&)>(nullptr),
		.cross_compile = std::move(cc),
#endif
#endif
	});

	// Create a RISC-V machine with the binary as input program
	auto st0 = std::chrono::high_resolution_clock::now();
	riscv::Machine<W> machine { binary, *options };
	if (cli_args.verbose) {
		auto st1 = std::chrono::high_resolution_clock::now();
		printf("* Loaded in %.3f ms\n", std::chrono::duration<double, std::milli>(st1 - st0).count());
	}

	// Remember the options for later in case background compilation is enabled,
	// if new execute segments need to be decoded and so on. Basically all future
	// operations that need to know the options. This is optional.
	machine.set_options(std::move(options));

	if (cli_args.quit) { // Quit after instantiating the machine
		return;
	}

	// A helper system call to ask for symbols that is possibly only known at runtime
	// Used by testing executables
	riscv::address_type<W> symbol_function = 0;
	machine.set_userdata(&symbol_function);
	machine.install_syscall_handler(500,
		[] (auto& machine) {
			auto [addr] = machine.template sysargs<riscv::address_type<W>>();
			auto& symfunc = *machine.template get_userdata<decltype(symbol_function)>();
			symfunc = addr;
			printf("Introduced to symbol function: 0x%" PRIX64 "\n", uint64_t(addr));
		});

	// Enable stdin when in proxy mode
	if (cli_args.proxy_mode) {
		machine.set_stdin(
		[](const riscv::Machine<W> &machine, char *buf, size_t size) -> long {
			return read(0, buf, size);
		});
	}

	if constexpr (full_linux_guest)
	{
		std::vector<std::string> env = {
			"LC_CTYPE=C", "LC_ALL=C", "RUST_BACKTRACE=full"
		};
		machine.setup_linux(args, env);
		// Linux system to open files and access internet
		machine.setup_linux_syscalls();
		machine.fds().permit_filesystem = !cli_args.sandbox;
		machine.fds().permit_sockets    = !cli_args.sandbox;
		if (cli_args.proxy_mode) {
			if (cli_args.sandbox)
				fprintf(stderr, "Warning: Proxy mode is enabled, but sandbox is also enabled\n");
			machine.fds().permit_filesystem = true;
			machine.fds().permit_sockets    = true;
			machine.fds().proxy_mode = true; // Proxy mode for system calls (no more sandbox)
#ifndef _WIN32
			char buf[4096];
			machine.fds().cwd = getcwd(buf, sizeof(buf));
#endif
		}
		// Rewrite certain links to masquerade and simplify some interactions (eg. /proc/self/exe)
		machine.fds().filter_readlink = [&] (void* user, std::string& path) {
			if (path == "/proc/self/exe") {
				path = machine.fds().cwd + "/program";
				return true;
			}
			if (path == "/proc/self/fd/1" || path == "/proc/self/fd/2") {
				return true;
			}
			fprintf(stderr, "Guest wanted to readlink: %s (denied)\n", path.c_str());
			return false;
		};
		// Only allow opening certain file paths. The void* argument is
		// the user-provided pointer set in the RISC-V machine.
		machine.fds().filter_open = [=] (void* user, std::string& path) {
			(void) user;
			if (path == "/etc/hostname"
				|| path == "/etc/hosts"
				|| path == "/etc/nsswitch.conf"
				|| path == "/etc/host.conf"
				|| path == "/etc/resolv.conf")
				return true;
			if (path == "/dev/urandom")
				return true;
			if (path == "/program") { // Fake program path
				path = args.at(0); // Sneakily open the real program instead
				return true;
			}
			if (path == "/etc/ssl/certs/ca-certificates.crt")
				return true;

			// Paths that are allowed to be opened
			static const std::string sandbox_libdir  = "/lib/riscv64-linux-gnu/";
			// The real path to the libraries (on the host system)
			static const std::string real_libdir = "/usr/riscv64-linux-gnu/lib/";
			// The dynamic linker and libraries we allow
			static const std::vector<std::string> libs = {
				"libdl.so.2", "libm.so.6", "libgcc_s.so.1", "libc.so.6", "libatomic.so.1",
				"libstdc++.so.6", "libresolv.so.2", "libnss_dns.so.2", "libnss_files.so.2"
			};

			if (path.find(sandbox_libdir) == 0) {
				// Find the library name
				auto lib = path.substr(sandbox_libdir.size());
				if (std::find(libs.begin(), libs.end(), lib) == libs.end()) {
					if (cli_args.verbose) {
						fprintf(stderr, "Guest wanted to open: %s (denied)\n", path.c_str());
					}
					return false;
				} else if (cli_args.verbose) {
					fprintf(stderr, "Guest wanted to open: %s (allowed)\n", path.c_str());
				}
				// Construct new path
				path = real_libdir + path.substr(sandbox_libdir.size());
				return true;
			}

			if (is_dynamic && args.size() > 1 && path == args.at(1)) {
				return true;
			}
			if (cli_args.proxy_mode) {
				return true;
			}
			for (const auto& allowed : cli_args.allowed_files) {
				if (path == allowed) {
					return true;
				}
			}
			if (cli_args.verbose) {
				fprintf(stderr, "Guest wanted to open: %s (denied)\n", path.c_str());
			}
			return false;
		};
		// multi-threading
		machine.setup_posix_threads();
	}
	else if constexpr (newlib_mini_guest)
	{
		// the minimum number of syscalls needed for malloc and C++ exceptions
		machine.setup_newlib_syscalls(true);
		machine.fds().permit_filesystem = !cli_args.sandbox;
		machine.setup_argv(args);
		machine.on_unhandled_syscall =
		[] (riscv::Machine<W>& machine, size_t num) {
			if (num == 1024) { // newlib_open()
#include "newlib_open.hpp"
			}
			fprintf(stderr, "Unhandled syscall: %zu\n", num);
		};
		machine.fds().filter_open = [=] (void*, std::string& path) {
			if (cli_args.proxy_mode) {
				return true;
			}
			for (const auto& allowed : cli_args.allowed_files) {
				if (path == allowed) {
					return true;
				}
			}
			if (cli_args.verbose) {
				fprintf(stderr, "Guest wanted to open: %s (denied)\n", path.c_str());
			}
			return false;
		};
	}
	else if constexpr (micro_guest)
	{
		// This guest has accelerated libc functions, which
		// are provided as system calls
		// See: tests/unit/native.cpp and tests/unit/include/native_libc.h
		constexpr size_t heap_size = 6ULL << 20; // 6MB
		auto heap = machine.memory.mmap_allocate(heap_size);

		machine.setup_native_heap(470, heap, heap_size);
		machine.setup_native_memory(475);
		machine.setup_native_threads(490);

		machine.setup_newlib_syscalls();
		machine.setup_argv(args);
	}
	else {
		fprintf(stderr, "Unknown emulation mode! Exiting...\n");
		exit(1);
	}

	// A CLI debugger used with --debug or DEBUG=1
	riscv::DebugMachine debug { machine };

	if (cli_args.debug)
	{
		// Print all instructions by default
		const bool vi = true;
		// With --verbose we also print register values after
		// every instruction.
		const bool vr = cli_args.verbose;

		auto main_address = machine.address_of("main");
		if (cli_args.from_start || main_address == 0x0) {
			debug.verbose_instructions = vi;
			debug.verbose_registers = vr;
			// Without main() this is a custom or stripped program,
			// so we break immediately.
			debug.print_and_pause();
		} else {
			// Automatic breakpoint at main() to help debug certain programs
			debug.breakpoint(main_address,
			[vi, vr] (auto& debug) {
				auto& cpu = debug.machine.cpu;
				// Remove the breakpoint to speed up debugging
				debug.erase_breakpoint(cpu.pc());
				debug.verbose_instructions = vi;
				debug.verbose_registers = vr;
				printf("\n*\n* Entered main() @ 0x%" PRIX64 "\n*\n", uint64_t(cpu.pc()));
				debug.print_and_pause();
			});
		}
	}

	auto t0 = std::chrono::high_resolution_clock::now();
	try {
		// If you run the emulator with --gdb or GDB=1, you can connect
		// with gdb-multiarch using target remote localhost:2159.
		if (cli_args.gdb) {
			printf("GDB server is listening on localhost:2159\n");
			riscv::RSP<W> server { machine, 2159 };
			auto client = server.accept();
			if (client != nullptr) {
				printf("GDB is connected\n");
				while (client->process_one());
			}
			if (!machine.stopped()) {
				// Run remainder of program
				machine.simulate(cli_args.fuel);
			}
		} else if (cli_args.debug) {
			// CLI debug simulation
			debug.simulate();
		} else if (cli_args.singlestep) {
			// Single-step precise simulation
			machine.set_max_instructions(~0ULL);
			machine.cpu.simulate_precise();
		} else {
#ifdef NODEJS_WORKAROUND
			// In order to get NodeJS to work we need to live-patch deadlocked rwlocks
			// This is a temporary workaround until the issue is found and fixed.
			static const auto rw_rdlock = machine.address_of("pthread_rwlock_rdlock");
			static const auto rw_wrlock = machine.address_of("pthread_rwlock_wrlock");
			machine.install_syscall_handler(riscv::SYSCALL_EBREAK,
			[] (auto& machine)
			{
				auto& cpu = machine.cpu;
				if (cpu.pc() == rw_rdlock || cpu.pc() == rw_wrlock) {
					// Execute 2 instruction and step over them
					cpu.step_one(false);
					cpu.step_one(false);
					// Check for deadlock
					if (cpu.reg(14) == cpu.reg(15)) {
						// Deadlock detected, avoid branch (beq a4, a5) and reset the lock
						cpu.reg(14) = 0xFF;
						machine.memory.template write<uint32_t>(cpu.reg(10), 0);
					}
				} else {
					throw riscv::MachineException(riscv::UNHANDLED_SYSCALL, "EBREAK instruction", cpu.pc());
				}
			});
#endif // NODEJS_WORKAROUND

			// Normal RISC-V simulation
			if (cli_args.accurate)
				machine.simulate(cli_args.fuel);
			else {
				// Simulate until it eventually stops (or user interrupts)
				machine.cpu.simulate_inaccurate(machine.cpu.pc());
			}
		}
	} catch (const riscv::MachineException& me) {
		printf("%s\n", machine.cpu.current_instruction_to_string().c_str());
		printf(">>> Machine exception %d: %s (data: 0x%" PRIX64 ")\n",
				me.type(), me.what(), me.data());
		printf("%s\n", machine.cpu.registers().to_string().c_str());
		machine.memory.print_backtrace(
			[] (std::string_view line) {
				printf("-> %.*s\n", (int)line.size(), line.begin());
			});
		if (me.type() == riscv::UNIMPLEMENTED_INSTRUCTION || me.type() == riscv::MISALIGNED_INSTRUCTION) {
			printf(">>> Is an instruction extension disabled?\n");
			printf(">>> A-extension: %d  C-extension: %d  V-extension: %d\n",
				riscv::atomics_enabled, riscv::compressed_enabled, riscv::vector_extension);
		}
		if (cli_args.debug)
			debug.print_and_pause();
		else
			run_sighandler(machine);
	} catch (std::exception& e) {
		printf(">>> Exception: %s\n", e.what());
		machine.memory.print_backtrace(
			[] (std::string_view line) {
				printf("-> %.*s\n", (int)line.size(), line.begin());
			});
		if (cli_args.debug)
			debug.print_and_pause();
		else
			run_sighandler(machine);
	}

	auto t1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> runtime = t1 - t0;

	if (!cli_args.silent) {
		const auto retval = machine.return_value();
		printf(">>> Program exited, exit code = %" PRId64 " (0x%" PRIX64 ")\n",
			int64_t(retval), uint64_t(retval));
		if (cli_args.accurate)
		printf("Instructions executed: %" PRIu64 "  Runtime: %.3fms  Insn/s: %.0fmi/s\n",
			machine.instruction_counter(), runtime.count()*1000.0,
			machine.instruction_counter() / (runtime.count() * 1e6));
		else
		printf("Runtime: %.3fms   (Use --accurate for instruction counting)\n",
			runtime.count()*1000.0);
		printf("Pages in use: %zu (%" PRIu64 " kB virtual memory, total %" PRIu64 " kB)\n",
			machine.memory.pages_active(),
			machine.memory.pages_active() * riscv::Page::size() / uint64_t(1024),
			machine.memory.memory_usage_total() / uint64_t(1024));
	}

	if (!cli_args.call_function.empty())
	{
		auto addr = machine.address_of(cli_args.call_function);
		if (addr == 0 && symbol_function != 0) {
			addr = machine.vmcall(symbol_function, cli_args.call_function);
		}
		if (addr != 0) {
			printf("Calling function %s @ 0x%lX\n", cli_args.call_function.c_str(), long(addr));
			machine.vmcall(addr);
		} else {
			printf("Error: Function %s not found, not able to call\n", cli_args.call_function.c_str());
		}
	}

#ifdef RISCV_BINARY_TRANSLATION
	if (!cli_args.jump_hints_file.empty()) {
		const auto jump_hints = machine.memory.gather_jump_hints();
		if (jump_hints.size() > machine.options().translator_jump_hints.size()) {
			store_jump_hints<W>(cli_args.jump_hints_file, jump_hints);
			if (cli_args.verbose)
				printf("%zu jump hints were saved to %s\n",
					jump_hints.size(), cli_args.jump_hints_file.c_str());
		}
	}
#endif
}

int main(int argc, const char** argv)
{
	Arguments cli_args;
#ifdef HAVE_GETOPT_LONG
	const int optind = parse_arguments(argc, argv, cli_args);
	if (optind < 0)
		return 1;
	else if (optind == 0)
		return 0;
	// Skip over the parsed arguments
	argc -= optind;
	argv += optind;
#else
	if (argc < 2) {
		fprintf(stderr, "Provide RISC-V binary as argument!\n");
		exit(1);
	}
	// Skip over the program name
	argc -= 1;
	argv += 1;

	// Environment variables can be used to control the emulator
	cli_args.verbose = getenv("VERBOSE") != nullptr;
	cli_args.debug = getenv("DEBUG") != nullptr;
	cli_args.gdb = getenv("GDB") != nullptr;
	cli_args.silent = getenv("SILENT") != nullptr;
	cli_args.timing = getenv("TIMING") != nullptr;
	cli_args.trace = getenv("TRACE") != nullptr;
	cli_args.no_translate = getenv("NO_TRANSLATE") != nullptr;
	cli_args.mingw = getenv("MINGW") != nullptr;
	cli_args.from_start = getenv("FROM_START") != nullptr;

#endif

	std::vector<std::string> args;
	for (int i = 0; i < argc; i++) {
		args.push_back(argv[i]);
	}
	const std::string& filename = args.front();

	using ElfHeader = typename riscv::Elf<4>::Header;

	try {
		std::vector<uint8_t> vbin;
#if !defined(__linux__)
		// Use load_file for non-Posix systems
		vbin = load_file(filename);
		if (vbin.size() < sizeof(ElfHeader)) {
			fprintf(stderr, "ELF binary was too small to be usable!\n");
			exit(1);
		}
		std::string_view binary { (const char*)vbin.data(), vbin.size() };
#else
		// Use mmap for Posix systems, not sure if Apple supports this
		std::string_view binary;
		int fd = open(filename.c_str(), O_RDONLY);
		if (fd < 0) {
			fprintf(stderr, "Could not open file: %s\n", filename.c_str());
			exit(1);
		}
		struct stat st;
		if (fstat(fd, &st) < 0) {
			fprintf(stderr, "Could not stat file: %s\n", filename.c_str());
			exit(1);
		}
		if (st.st_size < sizeof(ElfHeader)) {
			fprintf(stderr, "ELF binary was too small to be usable!\n");
			exit(1);
		}
		void* ptr = mmap(nullptr, st.st_size, PROT_READ, MAP_FILE|MAP_PRIVATE|MAP_NORESERVE, fd, 0);
		if (ptr == MAP_FAILED) {
			fprintf(stderr, "Could not mmap file: %s\n", filename.c_str());
			exit(1);
		}
		binary = { (const char*)ptr, size_t(st.st_size) };
#endif

		bool is_dynamic = false;
		if (binary[4] == riscv::ELFCLASS64) {
			std::string_view interpreter;
			std::tie(is_dynamic, interpreter) =
				riscv::Elf<8>::is_dynamic(std::string_view(binary.data(), binary.size()));

			if (is_dynamic) {
				// Load the dynamic linker shared object
				if (interpreter.empty()) {
					interpreter = DYNAMIC_LINKER;
				}
				try {
					vbin = load_file(std::string(interpreter));
				} catch (const std::exception& e) {
					vbin = load_file(DYNAMIC_LINKER);
				}
				binary = { (const char*)vbin.data(), vbin.size() };
				// Insert program name as argv[1]
				args.insert(args.begin() + 1, args.at(0));
				// Set dynamic linker to argv[0]
				args.at(0) = interpreter;
			}
		}

		if (binary[4] == riscv::ELFCLASS64)
#ifdef RISCV_64I
			run_program<riscv::RISCV64> (cli_args, binary, is_dynamic, args);
#else
			throw riscv::MachineException(riscv::FEATURE_DISABLED, "64-bit not currently enabled");
#endif
		else if (binary[4] == riscv::ELFCLASS32)
#ifdef RISCV_32I
			run_program<riscv::RISCV32> (cli_args, binary, is_dynamic, args);
#else
			throw riscv::MachineException(riscv::FEATURE_DISABLED, "32-bit not currently enabled");
#endif
		else
			throw riscv::MachineException(riscv::INVALID_PROGRAM, "Unknown ELF class", binary[4]);
	} catch (const std::exception& e) {
		printf("Exception: %s\n", e.what());
	}

	return 0;
}

template <int W>
void run_sighandler(riscv::Machine<W>& machine)
{
	constexpr int SIG_SEGV = 11;
	auto& action = machine.sigaction(SIG_SEGV);
	if (action.is_unset())
		return;

	auto handler = action.handler;
	action.handler = 0x0; // Avoid re-triggering(?)

	machine.stack_push(machine.cpu.reg(riscv::REG_RA));
	machine.cpu.reg(riscv::REG_RA) = machine.cpu.pc();
	machine.cpu.reg(riscv::REG_ARG0) = 11; /* SIGSEGV */
	try {
		machine.cpu.jump(handler);
		machine.simulate(60'000);
	} catch (...) {}

	action.handler = handler;
}

#include <stdexcept>
#include <unistd.h>
#include <fstream>
std::vector<uint8_t> load_file(const std::string& filename)
{
	std::ifstream file(filename, std::ios::in | std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("Could not open file: " + filename);
	}

	return std::vector<uint8_t> (std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

template <int W>
std::vector<riscv::address_type<W>> load_jump_hints(const std::string& filename, bool verbose)
{
	std::vector<riscv::address_type<W>> hints;
	if (filename.empty())
		return hints;

	std::ifstream file(filename);
	if (!file.is_open()) {
		if (verbose)
			fprintf(stderr, "Could not open jump hints file: %s\n", filename.c_str());
		return hints;
	}

	std::string line;
	while (std::getline(file, line)) {
		if (line.empty() || line[0] == '#') continue;
		// Parse hex address from line
		hints.push_back(std::stoull(line, nullptr, 16));
		//printf("Jump hint: 0x%lX\n", long(hints.back()));
	}
	return hints;
}

template <int W>
void store_jump_hints(const std::string& filename, const std::vector<riscv::address_type<W>>& hints)
{
	std::ofstream file(filename);
	if (!file.is_open()) {
		fprintf(stderr, "Could not open jump hints file for writing: %s\n", filename.c_str());
		return;
	}

	for (auto addr : hints) {
		file << "0x" << std::hex << addr << std::endl;
	}
}
