#include <catch2/catch_test_macros.hpp>
#include <libriscv/machine.hpp>
extern std::vector<uint8_t> load_file(const std::string& filename);
static const uint64_t MAX_MEMORY = 680ul << 20; /* 680MB */
static const uint64_t MAX_INSTRUCTIONS = 10'000'000ul;
static const std::string cwd {SRCDIR};
using namespace riscv;

TEST_CASE("Golang Hello World", "[Verify]")
{
	const auto binary = load_file(cwd + "/elf/golang-riscv64-hello-world");

	riscv::Machine<RISCV64> machine { binary, { .memory_max = MAX_MEMORY } };
	// We need to install Linux system calls for maximum gucciness
	machine.setup_linux_syscalls();
	machine.fds().permit_filesystem = true;
	machine.fds().permit_sockets = false;
	machine.fds().filter_open = [] (void* user, const std::string& path) {
		(void) user; (void) path;
		return false;
	};
	// multi-threading
	machine.setup_posix_threads();
	// We need to create a Linux environment for runtimes to work well
	machine.setup_linux(
		{"golang-riscv64-hello-world"},
		{"LC_TYPE=C", "LC_ALL=C", "USER=root"});

	struct State {
		bool output_is_hello_world = false;
	} state;
	machine.set_userdata(&state);
	machine.set_printer([] (const auto& m, const char* data, size_t size) {
		auto* state = m.template get_userdata<State> ();
		std::string text{data, data + size};
		state->output_is_hello_world = (text == "hello world");
	});

	// Run for at most X instructions before giving up
	machine.simulate(MAX_INSTRUCTIONS);

	REQUIRE(machine.return_value() == 0);
	REQUIRE(state.output_is_hello_world);
}

TEST_CASE("Zig Hello World", "[Verify]")
{
	const auto binary = load_file(cwd + "/elf/zig-riscv64-hello-world");

	riscv::Machine<RISCV64> machine { binary, { .memory_max = MAX_MEMORY } };
	// Install Linux system calls
	machine.setup_linux_syscalls();
	// Create a Linux environment for runtimes to work well
	machine.setup_linux(
		{"zig-riscv64-hello-world"},
		{"LC_TYPE=C", "LC_ALL=C", "USER=root"});

	struct State {
		std::string text;
	} state;
	machine.set_userdata(&state);
	machine.set_printer([] (const auto& m, const char* data, size_t size) {
		auto* state = m.template get_userdata<State> ();
		state->text.append(data, data + size);
	});

	// Run for at most X instructions before giving up
	machine.simulate(MAX_INSTRUCTIONS);

	REQUIRE(machine.return_value() == 0);
	REQUIRE(state.text == "Hello, world!\n");
}

TEST_CASE("Rust Hello World", "[Verify]")
{
	const auto binary = load_file(cwd + "/elf/rust-riscv64-hello-world");

	riscv::Machine<RISCV64> machine { binary, { .memory_max = MAX_MEMORY } };
	// Install Linux system calls
	machine.setup_linux_syscalls();
	// Create a Linux environment for runtimes to work well
	machine.setup_linux(
		{"rust-riscv64-hello-world"},
		{"LC_TYPE=C", "LC_ALL=C", "USER=root"});

	struct State {
		std::string text;
	} state;
	machine.set_userdata(&state);
	machine.set_printer([] (const auto& m, const char* data, size_t size) {
		auto* state = m.template get_userdata<State> ();
		state->text.append(data, data + size);
	});

	// Run for at most X instructions before giving up
	machine.simulate(MAX_INSTRUCTIONS);

	REQUIRE(machine.return_value() == 0);
	REQUIRE(state.text == "Hello World!\n");
}

TEST_CASE("RV32 Newlib with B-ext Hello World", "[Verify]")
{
	const auto binary = load_file(cwd + "/elf/newlib-rv32gb-hello-world");

	riscv::Machine<RISCV32> machine { binary, { .memory_max = MAX_MEMORY } };
	// Install Linux system calls
	machine.setup_linux_syscalls();
	// Create a Linux environment for runtimes to work well
	machine.setup_linux(
		{"newlib-rv32gb-hello-world"},
		{"LC_TYPE=C", "LC_ALL=C", "USER=root"});

	struct State {
		std::string text;
	} state;
	machine.set_userdata(&state);
	machine.set_printer([] (const auto& m, const char* data, size_t size) {
		auto* state = m.template get_userdata<State> ();
		state->text.append(data, data + size);
	});

	// Run for at most X instructions before giving up
	machine.simulate(MAX_INSTRUCTIONS);

	// Sadly, main() returns int which gets sign-extended to all bits set
	// which kind of defeats the purpose of testing ROL... oh well.
	REQUIRE(machine.return_value() == 666);
	REQUIRE(state.text.find("[confronted]") != std::string::npos);
	REQUIRE(state.text.find("[problem]") != std::string::npos);
	REQUIRE(state.text.find("[regular]") != std::string::npos);
	REQUIRE(state.text.find("[expressions]") != std::string::npos);
	REQUIRE(state.text.find("[problems]") != std::string::npos);
	REQUIRE(state.text.find("Caught exception: Hello Exceptions!") != std::string::npos);
}

TEST_CASE("RV64 Newlib with B-ext Hello World", "[Verify]")
{
	const auto binary = load_file(cwd + "/elf/newlib-rv64gb-hello-world");

	riscv::Machine<RISCV64> machine { binary, { .memory_max = MAX_MEMORY } };
	// Install Linux system calls
	machine.setup_linux_syscalls();
	// Create a Linux environment for runtimes to work well
	machine.setup_linux(
		{"newlib-rv64gb-hello-world"},
		{"LC_TYPE=C", "LC_ALL=C", "USER=root"});

	struct State {
		std::string text;
	} state;
	machine.set_userdata(&state);
	machine.set_printer([] (const auto& m, const char* data, size_t size) {
		auto* state = m.template get_userdata<State> ();
		state->text.append(data, data + size);
	});

	// Run for at most X instructions before giving up
	machine.simulate(MAX_INSTRUCTIONS);

	// Sadly, main() returns int which gets sign-extended to all bits set
	// which kind of defeats the purpose of testing ROL... oh well.
	REQUIRE(machine.return_value() == 666);
	REQUIRE(state.text.find("[confronted]") != std::string::npos);
	REQUIRE(state.text.find("[problem]") != std::string::npos);
	REQUIRE(state.text.find("[regular]") != std::string::npos);
	REQUIRE(state.text.find("[expressions]") != std::string::npos);
	REQUIRE(state.text.find("[problems]") != std::string::npos);
	REQUIRE(state.text.find("Caught exception: Hello Exceptions!") != std::string::npos);
}

TEST_CASE("TinyCC dynamic fib", "[Verify]")
{
	const auto binary = load_file(cwd + "/elf/tinycc-rv64g-fib");

	riscv::Machine<RISCV64> machine { binary, { .memory_max = MAX_MEMORY } };
	// Install Linux system calls
	machine.setup_linux_syscalls();
	// Create a Linux environment for runtimes to work well
	machine.setup_linux(
		{"tinycc-rv64g-fib"},
		{"LC_TYPE=C", "LC_ALL=C", "USER=groot"});

	// Run for at most X instructions before giving up
	machine.simulate(MAX_INSTRUCTIONS);

	REQUIRE(machine.return_value() == 75025); // fib(25)
}

TEST_CASE("RV32 Execute-Only Hello World", "[Verify]")
{
	const auto binary = load_file(cwd + "/elf/riscv32gb-execute-only");
	constexpr uint64_t MAX_MEMORY = 128ul << 20; /* 128MB */

	riscv::Machine<RISCV32> machine{binary, {
		.memory_max = MAX_MEMORY,
		.enforce_exec_only = true,
	}};
	machine.setup_newlib_syscalls();
	machine.setup_argv(
		{"riscv32gb-execute-only"}, {});

	machine.setup_native_heap(580,
		machine.memory.mmap_allocate(0x1800000), 0x1800000);
	machine.setup_native_memory(585);

	struct State {
		std::string text;
	} state;
	machine.set_userdata(&state);

	machine.install_syscall_handler(502,
	[] (auto& machine) {
		auto [buf, count] = machine.template sysargs<uint32_t, uint32_t>();
		auto view = machine.memory.memview(buf, count);
		printf("%.*s", (int) view.size(), view.data());

		auto* state = machine.template get_userdata<State>();
		state->text.append((const char*) view.data(), view.size());
	});

	machine.simulate(MAX_INSTRUCTIONS);

	REQUIRE(machine.return_value() == 123);
	REQUIRE(state.text == "Hello, World!");
}
