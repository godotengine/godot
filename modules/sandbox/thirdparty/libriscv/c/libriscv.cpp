#include "libriscv.h"

#include <libriscv/util/load_binary_file.hpp>
#include <libriscv/machine.hpp>
using namespace riscv;
static const std::vector<std::string> env = {"LC_CTYPE=C", "LC_ALL=C", "USER=groot"};

#define RISCV_ARCH  RISCV64
#define MACHINE(x) ((Machine<RISCV_ARCH> *)x)
#define ERROR_CALLBACK(m, type, msg, data) \
	if (auto *usr = m->template get_userdata<UserData> (); usr->error != nullptr) \
		usr->error(usr->opaque, type, msg, data);
#define USERDATA(m) (*m->template get_userdata<UserData> ())

static std::vector<std::string> fill(unsigned count, const char* const* args) {
	std::vector<std::string> v;
	v.reserve(count);
	for (unsigned i = 0; i < count; i++)
		v.push_back(args[i]);
	return v;
}

struct UserData {
	riscv_error_func_t error = nullptr;
	riscv_stdout_func_t stdout = nullptr;
	void *opaque = nullptr;
	std::vector<std::string> allowed_files;
};

extern "C"
void libriscv_set_defaults(RISCVOptions *options)
{
	MachineOptions<RISCV_ARCH> mo;

	options->max_memory = mo.memory_max;
	options->stack_size = mo.stack_size;
	options->strict_sandbox = true;
	options->argc = 0;
}

extern "C"
RISCVMachine *libriscv_new(const void *elf_prog, unsigned elf_length, RISCVOptions *options)
{
	MachineOptions<RISCV_ARCH> mo {
		.memory_max = options->max_memory,
		.stack_size = options->stack_size,
	};
	UserData *u = nullptr;
	try {
		auto view = std::string_view{(const char *)elf_prog, size_t(elf_length)};

		auto* m = new Machine<RISCV_ARCH> { view, mo };
		u = new UserData {
			.error = options->error, .stdout = options->stdout, .opaque = options->opaque
		};
		m->set_userdata(u);
		m->set_printer([] (auto& m, const char* data, size_t size) {
			auto& userdata = (*m.template get_userdata<UserData> ());
			if (userdata.stdout)
				userdata.stdout(userdata.opaque, data, size);
			else
				printf("%.*s", (int)size, data);
		});
		Machine<RISCV_ARCH>::on_unhandled_syscall = [] (auto& m, size_t num) {
			ERROR_CALLBACK((&m), RISCV_ERROR_TYPE_MACHINE_EXCEPTION, "Unknown system call", num);
		};

		std::vector<std::string> args;
		if (options->argc > 0) {
			args = fill(options->argc, options->argv);
		} else {
			args.push_back("./program"); // We need at least one argument
		}
		m->setup_linux(args, env);
		m->setup_linux_syscalls();
		m->setup_posix_threads();
		m->fds().permit_filesystem = !options->strict_sandbox;
		m->fds().permit_sockets = !options->strict_sandbox;
		// TODO: File permissions
		if (!options->strict_sandbox) {
			if (m->memory.is_dynamic_executable()) {
				// Since it's dynamic, the first argument (the program) is the dynamic linker
				// we'll treat the first argument as the program path, and automatically allow it
				USERDATA(m).allowed_files.push_back(args.at(0));
			}
			m->fds().filter_open = [=] (void* user, std::string& path) {
				(void) user;
				if (path == "/dev/urandom")
					return true;
				if (path == "/program") { // Fake program path
					path = args.at(0); // Sneakily open the real program instead
					return true;
				}

				// Paths that are allowed to be opened
				static const std::string sandbox_libdir  = "/lib/riscv64-linux-gnu/";
				// The real path to the libraries (on the host system)
				static const std::string real_libdir = "/usr/riscv64-linux-gnu/lib/";
				// The dynamic linker and libraries we allow
				auto& allowed_files = USERDATA(m).allowed_files;

				if (path.find(sandbox_libdir) == 0) {
					// Find the library name
					auto lib = path.substr(sandbox_libdir.size());
					for (const std::string& allowed_lib : allowed_files) {
						if (lib == allowed_lib) {
							// Construct new path
							path = real_libdir + path.substr(sandbox_libdir.size());
							return true;
						}
					}
				}

				if (m->memory.is_dynamic_executable() && args.size() > 1 && path == args.at(1)) {
					return true;
				}

				for (const auto& allowed : allowed_files) {
					if (path == allowed) {
						return true;
					}
				}
				return false;
			};
		}

		return (RISCVMachine *)m;
	}
	catch (const MachineException& me)
	{
		if (options->error)
			options->error(options->opaque, RISCV_ERROR_TYPE_MACHINE_EXCEPTION, me.what(), me.data());
		delete u;
		return NULL;
	}
	catch (const std::exception& e)
	{
		if (options->error)
			options->error(options->opaque, RISCV_ERROR_TYPE_GENERAL_EXCEPTION, e.what(), 0);
		delete u;
		return NULL;
	}
}

extern "C"
int libriscv_delete(RISCVMachine *m)
{
	try {
		delete MACHINE(m)->get_userdata<UserData> ();
		delete MACHINE(m);
		return 0;
	}
	catch (...)
	{
		return -1;
	}
}

extern "C"
int libriscv_run(RISCVMachine *m, uint64_t instruction_limit)
{
	try {
		auto& machine = *MACHINE(m);
		if (instruction_limit == 0) {
			machine.cpu.simulate_inaccurate(machine.cpu.pc());
			return machine.instruction_limit_reached() ? -RISCV_ERROR_TYPE_MACHINE_TIMEOUT : 0;
		}
		else {
			return machine.simulate<false>(instruction_limit) ? 0 : -RISCV_ERROR_TYPE_MACHINE_TIMEOUT;
		}
	} catch (const MachineTimeoutException& tmo) {
		ERROR_CALLBACK(MACHINE(m), RISCV_ERROR_TYPE_MACHINE_TIMEOUT, tmo.what(), tmo.data());
		return RISCV_ERROR_TYPE_MACHINE_TIMEOUT;
	} catch (const MachineException& me) {
		ERROR_CALLBACK(MACHINE(m), RISCV_ERROR_TYPE_MACHINE_EXCEPTION, me.what(), me.data());
		return RISCV_ERROR_TYPE_MACHINE_EXCEPTION;
	} catch (const std::exception& e) {
		ERROR_CALLBACK(MACHINE(m), RISCV_ERROR_TYPE_GENERAL_EXCEPTION, e.what(), 0);
		return RISCV_ERROR_TYPE_GENERAL_EXCEPTION;
	}
}
extern "C"
const char * libriscv_strerror(int return_value)
{
	switch (return_value) {
	case 0:
		return "No error";
	case RISCV_ERROR_TYPE_MACHINE_TIMEOUT:
		return "Timed out";
	case RISCV_ERROR_TYPE_MACHINE_EXCEPTION:
		return "Machine exception";
	case RISCV_ERROR_TYPE_GENERAL_EXCEPTION:
		return "General exception";
	default:
		return "Unknown error";
	}
}
extern "C"
void libriscv_stop(RISCVMachine *m)
{
	MACHINE(m)->stop();
}

extern "C"
int64_t libriscv_return_value(RISCVMachine *m)
{
	return MACHINE(m)->return_value();
}

extern "C"
uint64_t libriscv_instruction_counter(RISCVMachine *m)
{
	return MACHINE(m)->instruction_counter();
}
extern "C"
uint64_t * libriscv_max_counter_pointer(RISCVMachine *m)
{
	return &MACHINE(m)->get_counters().second;
}

extern "C"
int libriscv_instruction_limit_reached(RISCVMachine *m)
{
	return MACHINE(m)->instruction_limit_reached();
}

extern "C"
uint64_t libriscv_address_of(RISCVMachine *m, const char *name)
{
	try {
		return ((Machine<RISCV_ARCH> *)m)->address_of(name);
	}
	catch (...) {
		return 0x0;
	}
}

extern "C"
void * libriscv_opaque(RISCVMachine *m)
{
	return MACHINE(m)->get_userdata<UserData> ()->opaque;
}

extern "C"
void libriscv_allow_file(RISCVMachine *m, const char *path)
{
	USERDATA(MACHINE(m)).allowed_files.push_back(path);
}

extern "C"
int libriscv_set_syscall_handler(unsigned idx, riscv_syscall_handler_t handler)
{
	try {
		Machine<RISCV_ARCH>::syscall_handlers.at(idx) = Machine<RISCV_ARCH>::syscall_t(handler);
		return 0;
	}
	catch (...) {
		return RISCV_ERROR_TYPE_GENERAL_EXCEPTION;
	}
}

extern "C"
void libriscv_set_result_register(RISCVMachine *m, int64_t value)
{
	MACHINE(m)->set_result(value);
}
extern "C"
RISCVRegisters * libriscv_get_registers(RISCVMachine *m)
{
	return (RISCVRegisters *)&MACHINE(m)->cpu.registers();
}
extern "C"
int libriscv_jump(RISCVMachine *m, uint64_t address)
{
	try {
		MACHINE(m)->cpu.jump(address);
		return 0;
	} catch (const MachineException& me) {
		ERROR_CALLBACK(MACHINE(m), RISCV_ERROR_TYPE_MACHINE_EXCEPTION, me.what(), me.data());
		return RISCV_ERROR_TYPE_MACHINE_EXCEPTION;
	} catch (const std::exception& e) {
		ERROR_CALLBACK(MACHINE(m), RISCV_ERROR_TYPE_GENERAL_EXCEPTION, e.what(), 0);
	}
	return RISCV_ERROR_TYPE_GENERAL_EXCEPTION;
}
extern "C"
int libriscv_setup_vmcall(RISCVMachine *m, uint64_t address)
{
	try {
		auto* machine = MACHINE(m);
		machine->cpu.reset_stack_pointer();
		machine->setup_call();
		machine->cpu.jump(address);
		return 0;
	} catch (const MachineException& me) {
		ERROR_CALLBACK(MACHINE(m), RISCV_ERROR_TYPE_MACHINE_EXCEPTION, me.what(), me.data());
		return RISCV_ERROR_TYPE_MACHINE_EXCEPTION;
	} catch (const std::exception& e) {
		ERROR_CALLBACK(MACHINE(m), RISCV_ERROR_TYPE_GENERAL_EXCEPTION, e.what(), 0);
	}
	return RISCV_ERROR_TYPE_GENERAL_EXCEPTION;
}

extern "C"
int libriscv_copy_to_guest(RISCVMachine *m, uint64_t dst, const void *src, unsigned len)
{
	try {
		MACHINE(m)->copy_to_guest(dst, src, len);
		return 0;
	} catch (const MachineException& me) {
		ERROR_CALLBACK(MACHINE(m), RISCV_ERROR_TYPE_MACHINE_EXCEPTION, me.what(), me.data());
		return RISCV_ERROR_TYPE_MACHINE_EXCEPTION;
	} catch (const std::exception& e) {
		ERROR_CALLBACK(MACHINE(m), RISCV_ERROR_TYPE_GENERAL_EXCEPTION, e.what(), 0);
	}
	return RISCV_ERROR_TYPE_GENERAL_EXCEPTION;
}
extern "C"
int libriscv_copy_from_guest(RISCVMachine *m, void* dst, uint64_t src, unsigned len)
{
	try {
		MACHINE(m)->copy_from_guest(dst, src, len);
		return 0;
	} catch (const MachineException& me) {
		ERROR_CALLBACK(MACHINE(m), RISCV_ERROR_TYPE_MACHINE_EXCEPTION, me.what(), me.data());
		return RISCV_ERROR_TYPE_MACHINE_EXCEPTION;
	} catch (const std::exception& e) {
		ERROR_CALLBACK(MACHINE(m), RISCV_ERROR_TYPE_GENERAL_EXCEPTION, e.what(), 0);
	}
	return RISCV_ERROR_TYPE_GENERAL_EXCEPTION;
}

extern "C"
const char * libriscv_memstring(RISCVMachine *m, uint64_t src, unsigned maxlen, unsigned* length)
{
	if (length == nullptr)
		return nullptr;
	char *result = nullptr;

	try {
		const auto view = MACHINE(m)->memory.memstring_view(src, maxlen);
		*length = view.size();
		return view.data();
	} catch (const MachineException& me) {
		ERROR_CALLBACK(MACHINE(m), RISCV_ERROR_TYPE_MACHINE_EXCEPTION, me.what(), me.data());
	} catch (const std::exception& e) {
		ERROR_CALLBACK(MACHINE(m), RISCV_ERROR_TYPE_GENERAL_EXCEPTION, e.what(), 0);
	}

	if (result)
		std::free(result);
	*length = 0;
	return nullptr;
}

extern "C"
const char * libriscv_memview(RISCVMachine *m, uint64_t src, unsigned length)
{
	try {
		auto buffer = MACHINE(m)->memory.memview(src, length);
		return buffer.data();
	} catch (const MachineException& me) {
		ERROR_CALLBACK(MACHINE(m), RISCV_ERROR_TYPE_MACHINE_EXCEPTION, me.what(), me.data());
	} catch (const std::exception& e) {
		ERROR_CALLBACK(MACHINE(m), RISCV_ERROR_TYPE_GENERAL_EXCEPTION, e.what(), 0);
	}
	return nullptr;
}

extern "C"
char * libriscv_writable_memview(RISCVMachine *m, uint64_t src, unsigned length)
{
	try {
		auto buffer = MACHINE(m)->memory.writable_memview(src, length);
		return (char *)buffer.data();
	} catch (const MachineException& me) {
		ERROR_CALLBACK(MACHINE(m), RISCV_ERROR_TYPE_MACHINE_EXCEPTION, me.what(), me.data());
	} catch (const std::exception& e) {
		ERROR_CALLBACK(MACHINE(m), RISCV_ERROR_TYPE_GENERAL_EXCEPTION, e.what(), 0);
	}
	return nullptr;
}

extern "C"
void libriscv_trigger_exception(RISCVMachine *m, unsigned exception, uint64_t data)
{
	MACHINE(m)->cpu.trigger_exception(exception, data);
}

extern "C"
int libriscv_load_binary_file(const char *filename, char **data)
{
	if (filename == NULL || data == NULL) {
		return -1;
	}

	std::string filename_cpp(filename);
	std::vector<uint8_t> loaded_file = load_binary_file(filename_cpp);
	size_t size = loaded_file.size() * sizeof(char);

	*data = (char *) malloc(size);
	if (*data == nullptr) {
		return -1;
	}

	std::copy(loaded_file.begin(), loaded_file.end(), *data);

	return size;
}
