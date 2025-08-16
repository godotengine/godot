#include <fmt/core.h>
#include <cmath>
#include <libriscv/machine.hpp>
#include <libriscv/util/load_binary_file.hpp>
using namespace riscv;
static constexpr int MARCH = riscv::RISCV64;
using RiscvMachine = riscv::Machine<MARCH>;
using gaddr_t = riscv::address_type<MARCH>;

int main(int argc, char** argv)
{
	if (argc < 2) {
		fmt::print("Usage: {} [program file] [arguments ...]\n", argv[0]);
		return -1;
	}
	// A function address we will call later
	static gaddr_t g_host_functions_addr = 0;

	// Register a host function that can be called from the guest program
	RiscvMachine::install_syscall_handler(500, [](RiscvMachine& machine) {
		fmt::print("Hello from host function 0!\n");

		// Get a zero-copy view of Strings in guest memory
		struct Strings {
			gaddr_t count;
			gaddr_t strings[32];
		};
		auto [vec] = machine.sysargs<Strings*>();

		// For each string up to count, read it from guest memory and print it
		for (size_t i = 0; i < vec->count; i++) {
			std::string str = machine.memory.memstring(vec->strings[i]);
			fmt::print("  {}\n", str);
		}
	});

	// Register a two-way host function that modifies guest memory
	RiscvMachine::install_syscall_handler(501, [](RiscvMachine& machine) {
		fmt::print("Hello from host function 1!\n");

		// Get a zero-copy view of Strings in guest memory
		struct Buffer {
			gaddr_t count;
			char    buffer[256];            // An inline buffer
			gaddr_t another_count;
			gaddr_t another_buffer_address; // A pointer to a buffer somewhere in guest memory
		};
		auto [buf] = machine.sysargs<Buffer*>();

		// Write a string to the buffer in guest memory
		strcpy(buf->buffer, "Hello from host function 1!");
		buf->count = strlen(buf->buffer);

		// The "another" buffer has a count and then a guest pointer to the buffer
		// In order to get a writable pointer to that buffer, we can use memarray<T>():
		char* another_buf = machine.memory.memarray<char>(buf->another_buffer_address, buf->another_count);
		// Let's check if the buffer is large enough to hold the string
		const std::string str = "Another buffer from host function 1!";
		if (str.size() > buf->another_count) {
			fmt::print("Another buffer is too small to hold the string!\n");
			return;
		}
		// Copy the string to the buffer
		strcpy(another_buf, str.c_str());
		another_buf[str.size()] = '\0';
		// Update the count of the buffer
		buf->another_count = str.size();
	});

	// Register a host function that takes a function pointer
	RiscvMachine::install_syscall_handler(502, [](RiscvMachine& machine) {
		// Get the function pointer argument as a guest address
		auto [fn] = machine.sysargs<gaddr_t>();

		// Set our host function address so we can call it later
		g_host_functions_addr = fn;
	});

	// Register a host function that normalizes a vec3
	RiscvMachine::install_syscall_handler(503, [](RiscvMachine& machine) {
		// Get the vec3 argument as a guest address
		auto [x, y, z] = machine.sysargs<float, float, float>();

		// Normalize the vector
		float len = sqrtf(x * x + y * y + z * z);
		if (len > 0) {
			float inv_len = 1.0f / len;
			x *= inv_len;
			y *= inv_len;
			z *= inv_len;
		}

		machine.set_result(x, y, z);
	});

	// Create a new machine
	std::vector<uint8_t> program = load_binary_file(argv[1]);
	RiscvMachine machine(program);

	// Setup the machine configuration and syscall interface
	machine.setup_linux({"program"}, {"LC_CTYPE=C", "LC_ALL=C", "USER=groot"});
	// Add POSIX system call interfaces (no filesystem or network access)
	machine.setup_linux_syscalls(false, false);
	machine.setup_posix_threads();

	// Run the machine
	try {
		machine.simulate();
	} catch (const std::exception& e) {
		fmt::print("Exception: {}\n", e.what());
	}

	// Call the host function that takes a function pointer
	if (g_host_functions_addr != 0) {
		fmt::print("Calling host function 2...\n");
		machine.vmcall(g_host_functions_addr, "Hello From A Function Callback!");
	} else {
		fmt::print("Host function 2 was not called!!?\n");
	}

	return 0;
}


