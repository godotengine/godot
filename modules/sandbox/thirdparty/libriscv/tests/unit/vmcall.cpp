#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <libriscv/machine.hpp>
extern std::vector<uint8_t> build_and_load(const std::string& code,
	const std::string& args = "-O2 -static", bool cpp = false);
static const uint64_t MAX_MEMORY = 8ul << 20; /* 8MB */
static const uint64_t MAX_INSTRUCTIONS = 10'000'000ul;
using namespace riscv;

TEST_CASE("VM function call", "[VMCall]")
{
	struct State {
		bool output_is_hello_world = false;
	} state;
	const auto binary = build_and_load(R"M(
	extern long write(int, const void*, unsigned long);
	__attribute__((used, retain))
	void hello() {
		write(1, "Hello World!", 12);
	}

	int main() {
		return 666;
	})M");

	riscv::Machine<RISCV64> machine { binary, { .memory_max = MAX_MEMORY } };
	// We need to install Linux system calls for maximum gucciness
	machine.setup_linux_syscalls();
	// We need to create a Linux environment for runtimes to work well
	machine.setup_linux(
		{"vmcall"},
		{"LC_TYPE=C", "LC_ALL=C", "USER=root"});

	machine.set_userdata(&state);
	machine.set_printer([] (const auto& m, const char* data, size_t size) {
		auto* state = m.template get_userdata<State> ();
		std::string text{data, data + size};
		state->output_is_hello_world = (text == "Hello World!");
	});
	// Run for at most X instructions before giving up
	machine.simulate(MAX_INSTRUCTIONS);

	REQUIRE(machine.return_value<int>() == 666);
	REQUIRE(!state.output_is_hello_world);

	const auto hello_address = machine.address_of("hello");
	REQUIRE(hello_address != 0x0);

	// Execute guest function
	machine.vmcall(hello_address);

	// Now hello world should have been printed
	REQUIRE(state.output_is_hello_world);
}

TEST_CASE("VM call return values", "[VMCall]")
{
	const auto binary = build_and_load(R"M(
	__attribute__((used, retain))
	const char* hello() {
		return "Hello World!";
	}

	static struct Data {
		int val1;
		int val2;
		float f1;
	} data = {.val1 = 1, .val2 = 2, .f1 = 3.0f};
	__attribute__((used, retain))
	extern struct Data* structs() {
		return &data;
	}

	int main() {
		return 666;
	})M");

	riscv::Machine<RISCV64> machine { binary, { .memory_max = MAX_MEMORY } };
	// We need to install Linux system calls for maximum gucciness
	machine.setup_linux_syscalls();
	// We need to create a Linux environment for runtimes to work well
	machine.setup_linux(
		{"vmcall"},
		{"LC_TYPE=C", "LC_ALL=C", "USER=root"});

	const auto hello_address = machine.address_of("hello");
	REQUIRE(hello_address != 0x0);

	// Test returning a string
	machine.vmcall(hello_address);
	REQUIRE(machine.return_value<std::string>() == "Hello World!");

	const auto structs_address = machine.address_of("structs");
	REQUIRE(structs_address != 0x0);

	// Test returning a structure
	struct Data {
		int val1;
		int val2;
		float f1;
	};
	machine.vmcall(structs_address);

	const auto data = machine.return_value<Data>();
	REQUIRE(data.val1 == 1);
	REQUIRE(data.val2 == 2);
	REQUIRE(data.f1 == 3.0f);

	// Test returning a pointer to a structure
	const auto* data_ptr = machine.return_value<Data*>();
	REQUIRE(data_ptr->val1 == 1);
	REQUIRE(data_ptr->val2 == 2);
	REQUIRE(data_ptr->f1 == 3.0f);
}

TEST_CASE("VM call enum values", "[VMCall]")
{
	const auto binary = build_and_load(R"M(
	#include <assert.h>
	int do_syscall(int value) {
		register long         a0 __asm__("a0") = value;
		register long syscall_id __asm__("a7") = 0;

		__asm__ volatile ("ecall" : "+r"(a0) : "r"(syscall_id));
		return a0;
	}
	__attribute__((used, retain))
	int mycall(int value) {
		assert(value == 1);
		return do_syscall(value);
	}

	int main() {
		return 666;
	})M");

	riscv::Machine<RISCV64> machine { binary, { .memory_max = MAX_MEMORY } };
	machine.setup_linux_syscalls();
	machine.setup_linux(
		{"vmcall"},
		{"LC_TYPE=C", "LC_ALL=C", "USER=root"});

	// Test Enum values
	enum class MyEnum : int {
		Hello = 1,
		World = 2,
	};

	machine.install_syscall_handler(0,
	[] (auto& machine) {
		auto [value] = machine.template sysargs <MyEnum> ();
		REQUIRE(value == MyEnum::Hello);
		machine.set_result(MyEnum::World);
	});

	machine.vmcall("mycall", MyEnum::Hello);
	REQUIRE(machine.return_value<MyEnum>() == MyEnum::World);
}

TEST_CASE("VM function call in fork", "[VMCall]")
{
	// The global variable 'value' should get
	// forked as value=1. We assert this, then
	// we set value=0. New forks should continue
	// to see value=1 as they are forked from the
	// main VM where value is still 0.
	const auto binary = build_and_load(R"M(
	#include <assert.h>
	#include <string.h>
	extern long write(int, const void*, unsigned long);
	static int value = 0;

	__attribute__((used, retain))
	void hello() {
		assert(value == 1);
		value = 0;
		write(1, "Hello World!", 12);
	}

	__attribute__((used, retain))
	int str(const char *arg) {
		assert(strcmp(arg, "Hello") == 0);
		return 1;
	}

	struct Data {
		int val1;
		int val2;
		float f1;
	};
	__attribute__((used, retain))
	int structs(struct Data *data) {
		assert(data->val1 == 1);
		assert(data->val2 == 2);
		assert(data->f1 == 3.0f);
		return 2;
	}

	__attribute__((used, retain))
	int ints(long i1, long i2, long i3) {
		assert(i1 == 123);
		assert(i2 == 456);
		assert(i3 == 456);
		return 3;
	}

	__attribute__((used, retain))
	int fps(float f1, double d1) {
		assert(f1 == 1.0f);
		assert(d1 == 2.0);
		return 4;
	}

	int main() {
		value = 1;
		return 666;
	})M");

	riscv::Machine<RISCV64> machine { binary, {
		.memory_max = MAX_MEMORY,
		.use_memory_arena = false,
	} };
	machine.setup_linux_syscalls();
	machine.setup_linux(
		{"vmcall"},
		{"LC_TYPE=C", "LC_ALL=C", "USER=root"});

	machine.simulate(MAX_INSTRUCTIONS);
	REQUIRE(machine.return_value<int>() == 666);

	// Test many forks
	for (size_t i = 0; i < 10; i++)
	{
		riscv::Machine<RISCV64> fork { machine, {
#ifdef RISCV_BINARY_TRANSLATION
			.use_memory_arena = false,
#endif
		} };
		REQUIRE(fork.memory.uses_flat_memory_arena() == false);

		fork.set_printer([] (const auto&, const char* data, size_t size) {
			std::string text{data, data + size};
			REQUIRE(text == "Hello World!");
		});

		const auto hello_address = fork.address_of("hello");
		REQUIRE(hello_address != 0x0);

		// Execute guest function
		fork.vmcall(hello_address);

		int res1 = fork.vmcall("str", "Hello");
		REQUIRE(res1 == 1);

		res1 = fork.vmcall("str", std::string("Hello"));
		REQUIRE(res1 == 1);

		std::string hello = "Hello";
		const std::string& ref = hello;

		res1 = fork.vmcall("str", ref);
		REQUIRE(res1 == 1);

		struct {
			int v1 = 1;
			int v2 = 2;
			float f1 = 3.0f;
		} data;
		int res2 = fork.vmcall("structs", data);
		REQUIRE(res2 == 2);

		long intval = 456;
		long& intref = intval;

		int res3 = fork.vmcall("ints", 123L, intref, (long&&)intref);
		REQUIRE(res3 == 3);

		int res4 = fork.vmcall("fps", 1.0f, 2.0);
		REQUIRE(res4 == 4);

		// XXX: Binary translation currently "remembers" that arena
		// was enabled, and will not disable it for the fork.
		if constexpr (riscv::flat_readwrite_arena && riscv::binary_translation_enabled)
			return;
	}
}

TEST_CASE("VM call and preemption", "[VMCall]")
{
	struct State {
		bool output_is_hello_world = false;
	} state;
	const auto binary = build_and_load(R"M(
	extern long write(int, const void*, unsigned long);
	long syscall1(long n, long arg0) {
		register long a0 __asm__("a0") = arg0;
		register long syscall_id __asm__("a7") = n;

		__asm__ volatile ("scall" : "+r"(a0) : "r"(syscall_id));

		return a0;
	}

	__attribute__((used, retain))
	long start() {
		syscall1(500, 1234567);
		return 1;
	}
	__attribute__((used, retain))
	void preempt(int arg) {
		write(1, "Hello World!", arg);
	}

	int main() {
		syscall1(500, 1234567);
		return 666;
	})M");

	riscv::Machine<RISCV64> machine { binary, { .memory_max = MAX_MEMORY } };
	machine.setup_linux_syscalls();
	machine.setup_linux(
		{"vmcall"},
		{"LC_TYPE=C", "LC_ALL=C", "USER=root"});

	machine.set_userdata(&state);
	machine.set_printer([] (const auto& m, const char* data, size_t size) {
		auto* state = m.template get_userdata<State> ();
		std::string text{data, data + size};
		state->output_is_hello_world = (text == "Hello World!");
	});

	machine.install_syscall_handler(500,
	[] (auto& machine) {
		auto [arg0] = machine.template sysargs <int> ();
		REQUIRE(arg0 == 1234567);

		const auto func = machine.address_of("preempt");
		REQUIRE(func != 0x0);

		machine.preempt(15'000ull, func, strlen("Hello World!"));
	});

	REQUIRE(!state.output_is_hello_world);

	machine.simulate(MAX_INSTRUCTIONS);

	REQUIRE(state.output_is_hello_world);
	REQUIRE(machine.return_value<int>() == 666);

	for (int i = 0; i < 10; i++)
	{
		state.output_is_hello_world = false;

		const auto func = machine.address_of("start");
		REQUIRE(func != 0x0);

		// Execute guest function
		machine.vmcall<15'000ull>(func);
		REQUIRE(machine.return_value<int>() == 1);

		// Now hello world should have been printed
		REQUIRE(state.output_is_hello_world);
	}
}

TEST_CASE("VM call and STOP instruction", "[VMCall]")
{
	struct State {
		bool output_is_hello_world = false;
	} state;
	const auto binary = build_and_load(R"M(
	extern long write(int, const void*, unsigned long);
	long syscall1(long n, long arg0) {
		register long a0 __asm__("a0") = arg0;
		register long syscall_id __asm__("a7") = n;

		__asm__ volatile ("scall" : "+r"(a0) : "r"(syscall_id));

		return a0;
	}
	void return_fast1(long retval)
	{
		register long a0 __asm__("a0") = retval;

		__asm__ volatile (".insn i SYSTEM, 0, x0, x0, 0x7ff" :: "r"(a0));
		__builtin_unreachable();
	}

	__attribute__((used, retain))
	long start() {
		syscall1(500, 1234567);
		return_fast1(1234);
		return 5678;
	}
	__attribute__((used, retain))
	long preempt(int arg) {
		write(1, "Hello World!", arg);
		return_fast1(777);
	}

	int main() {
		syscall1(500, 1234567);
		return_fast1(777);
		return 666;
	})M");

	riscv::Machine<RISCV64> machine { binary, { .memory_max = MAX_MEMORY } };
	machine.setup_linux_syscalls();
	machine.setup_linux(
		{"vmcall"},
		{"LC_TYPE=C", "LC_ALL=C", "USER=root"});

	machine.set_userdata(&state);
	machine.set_printer([] (const auto& m, const char* data, size_t size) {
		auto* state = m.template get_userdata<State> ();
		std::string text{data, data + size};
		state->output_is_hello_world = (text == "Hello World!");
	});

	machine.install_syscall_handler(500,
	[] (auto& machine) {
		auto [arg0] = machine.template sysargs <int> ();
		REQUIRE(arg0 == 1234567);

		const auto func = machine.address_of("preempt");
		REQUIRE(func != 0x0);

		auto result = machine.preempt(15'000ull, func, strlen("Hello World!"));
		REQUIRE(result == 777);
	});

	REQUIRE(!state.output_is_hello_world);

	machine.simulate(MAX_INSTRUCTIONS);

	REQUIRE(state.output_is_hello_world);
	REQUIRE(machine.return_value<int>() == 777);

	for (int i = 0; i < 10; i++)
	{
		state.output_is_hello_world = false;

		const auto func = machine.address_of("start");
		REQUIRE(func != 0x0);

		// Execute guest function
		machine.vmcall<15'000ull>(func);
		REQUIRE(machine.return_value<int>() == 1234);

		// Now hello world should have been printed
		REQUIRE(state.output_is_hello_world);
	}
}

TEST_CASE("VM call with arrays and vectors", "[VMCall]")
{
	const auto binary = build_and_load(R"M(
	__attribute__((used, retain))
	int pass_iarray(const int* data, unsigned size) {
		if (size != 3)
			return 0;
		if (data[0] != 1 || data[1] != 2 || data[2] != 3)
			return 0;
		return 1;
	}

	__attribute__((used, retain))
	int pass_farray(const float* data, unsigned size) {
		if (size != 3)
			return 0;
		if (data[0] != 1.0f || data[1] != 2.0f || data[2] != 3.0f)
			return 0;
		return 1;
	}

	struct Data {
		int val1;
		int val2;
		float f1;
	};
	__attribute__((used, retain))
	int pass_struct(const struct Data* data, unsigned size) {
		if (size != 3)
			return 0;
		if (data[0].val1 != 1 || data[0].val2 != 2 || data[0].f1 != 3.0f)
			return 0;
		if (data[1].val1 != 4 || data[1].val2 != 5 || data[1].f1 != 6.0f)
			return 0;
		if (data[2].val1 != 7 || data[2].val2 != 8 || data[2].f1 != 9.0f)
			return 0;
		return 1;
	}

	int main() {
		return 666;
	})M");

	riscv::Machine<RISCV64> machine { binary, { .memory_max = MAX_MEMORY } };
	machine.setup_linux_syscalls();
	machine.setup_linux(
		{"vmcall"},
		{"LC_TYPE=C", "LC_ALL=C", "USER=root"});

	machine.simulate(MAX_INSTRUCTIONS);
	REQUIRE(machine.return_value<int>() == 666);

	// Test passing an integer array
	std::array<int, 3> iarray = {1, 2, 3};
	// The array is pushed on the stack, so it becomes a sequential pointer argument
	int res1 = machine.vmcall("pass_iarray", iarray, iarray.size());
	REQUIRE(res1 == 1);

	// A const-reference to an array should also work
	const std::array<int, 3>& array_ref = iarray;
	int res2 = machine.vmcall("pass_iarray", array_ref, array_ref.size());
	REQUIRE(res2 == 1);

	// Test passing a float array
	std::array<float, 3> farray = {1.0f, 2.0f, 3.0f};
	int res3 = machine.vmcall("pass_farray", farray, farray.size());
	REQUIRE(res3 == 1);

	// Test passing a vector
	struct Data {
		int val1;
		int val2;
		float f1;
	};
	std::vector<Data> vec = {
		{1, 2, 3.0f},
		{4, 5, 6.0f},
		{7, 8, 9.0f},
	};
	// The vector is pushed on the stack, so it becomes a sequential pointer argument
	int res4 = machine.vmcall("pass_struct", vec, vec.size());
	REQUIRE(res4 == 1);

	// A const-reference to a vector should also work
	const std::vector<Data>& vec_ref = vec;
	int res5 = machine.vmcall("pass_struct", vec_ref, vec_ref.size());
	REQUIRE(res5 == 1);
}
