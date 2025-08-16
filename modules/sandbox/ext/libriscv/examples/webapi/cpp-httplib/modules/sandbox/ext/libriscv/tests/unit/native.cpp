#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <libriscv/machine.hpp>
#include <libriscv/native_heap.hpp>
extern std::vector<uint8_t> build_and_load(const std::string& code,
	const std::string& args = "-O2 -static", bool cpp = false);
static const uint64_t MAX_INSTRUCTIONS = 10'000'000ul;
static const std::string cwd {SRCDIR};
static bool is_zig() {
	const char* rcc = getenv("RCC");
	if (rcc == nullptr)
		return false;
	return std::string(rcc).find("zig") != std::string::npos;
}
using namespace riscv;

static const int HEAP_SYSCALLS_BASE	  = 470;
static const int MEMORY_SYSCALLS_BASE = 475;
static const int THREADS_SYSCALL_BASE = 490;

using CppString = GuestStdString<RISCV64>;
template <typename T>
using CppVector = GuestStdVector<RISCV64, T>;
using ScopedCppString = ScopedArenaObject<RISCV64, CppString>;
template <typename T>
using ScopedCppVector = ScopedArenaObject<RISCV64, CppVector<T>>;

template <int W>
static void setup_native_system_calls(riscv::Machine<W>& machine)
{
	// Syscall-backed heap
	constexpr size_t heap_size = 65536;
	auto heap = machine.memory.mmap_allocate(heap_size);

	machine.setup_native_heap(HEAP_SYSCALLS_BASE, heap, heap_size);
	machine.setup_native_memory(MEMORY_SYSCALLS_BASE);
	machine.setup_native_threads(THREADS_SYSCALL_BASE);
}

TEST_CASE("Activate native helper syscalls", "[Native]")
{
	const auto binary = build_and_load(R"M(
	#include <stdlib.h>
	#include <stdio.h>
	int main(int argc, char** argv)
	{
		const char *hello = (const char*)atol(argv[1]);
		printf("%s\n", hello);
		return 666;
	})M");

	riscv::Machine<RISCV64> machine { binary };
	machine.setup_linux_syscalls();

	setup_native_system_calls(machine);

	// Allocate string on heap
	static const std::string hello = "Hello World!";
	auto addr = machine.arena().malloc(64);
	machine.copy_to_guest(addr, hello.data(), hello.size()+1);

	// Pass string address to guest as main argument
	machine.setup_linux(
		{"native", std::to_string(addr)},
		{"LC_TYPE=C", "LC_ALL=C", "USER=root"});

	// Catch output from machine
	struct State {
		bool output_is_hello_world = false;
	} state;

	machine.set_userdata(&state);
	machine.set_printer([] (const auto& m, const char* data, size_t size) {
		auto* state = m.template get_userdata<State> ();
		std::string text{data, data + size};
		// musl writev:
		state->output_is_hello_world = state->output_is_hello_world || (text == "Hello World!");
		// glibc write:
		state->output_is_hello_world = state->output_is_hello_world || (text == "Hello World!\n");
	});

	// Run simulation
	machine.simulate(MAX_INSTRUCTIONS);

	REQUIRE(machine.return_value() == 666);
	REQUIRE(state.output_is_hello_world);
}

TEST_CASE("Use native helper syscalls", "[Native]")
{
	const auto binary = build_and_load(R"M(
	#include <include/native_libc.h>
	#include <stdlib.h>
	#include <stdio.h>
	int main()
	{
		char* hello = malloc(13);
		memcpy(hello, "Hello World!", 13);
		hello = realloc(hello, 128);
		printf("%s\n", hello);
		free(hello);
		return 666;
	})M", "-O2 -static -I" + cwd);

	riscv::Machine<RISCV64> machine { binary };

	setup_native_system_calls(machine);

	machine.setup_linux_syscalls();
	machine.setup_linux(
		{"native"},
		{"LC_TYPE=C", "LC_ALL=C", "USER=root"});

	// Catch output from machine
	struct State {
		bool output_is_hello_world = false;
	} state;

	machine.set_userdata(&state);
	machine.set_printer([] (const auto& m, const char* data, size_t size) {
		auto* state = m.template get_userdata<State> ();
		std::string text{data, data + size};
		// musl writev:
		state->output_is_hello_world = state->output_is_hello_world || (text == "Hello World!");
		// glibc write:
		state->output_is_hello_world = state->output_is_hello_world || (text == "Hello World!\n");
	});

	// Run simulation
	machine.simulate(MAX_INSTRUCTIONS);

	REQUIRE(machine.return_value() == 666);
	REQUIRE(state.output_is_hello_world);
}

TEST_CASE("Free unknown causes exception", "[Native]")
{
	const auto binary = build_and_load(R"M(
	#include <include/native_libc.h>
	int main()
	{
		free((void *)0x1234);
		return 666;
	})M", "-O2 -static -I" + cwd);

	riscv::Machine<RISCV64> machine { binary };
	setup_native_system_calls(machine);

	machine.setup_linux_syscalls();
	machine.setup_linux(
		{"native"},
		{"LC_TYPE=C", "LC_ALL=C", "USER=root"});

	bool error = false;
	try {
		machine.simulate(MAX_INSTRUCTIONS);
	} catch (const std::exception& e) {
		// Libtcc does not forward the real exception (instead throws a generic SYSTEM_CALL_FAILED)
		if constexpr (!libtcc_enabled)
			REQUIRE(std::string(e.what()) == "Possible double-free for freed pointer");
		error = true;
	}
	REQUIRE(error);
}

TEST_CASE("VM calls with std::string and std::vector", "[Native]")
{
	if (is_zig()) // We don't support libc++ std::string yet
		return;

	const auto binary = build_and_load(R"M(
	#include <string>
	#include <vector>
	#include <cassert>

	void* operator new(size_t size) {
		return malloc(size);
	}
	void operator delete(void* ptr) {
		free(ptr);
	}

	extern "C" __attribute__((used, retain))
	void test(std::string& str,
		const std::vector<int>& ints,
		const std::vector<std::string>& strings)
	{
		std::string result = "Hello, " + str + "! Integers:";
		for (auto i : ints)
			result += " " + std::to_string(i);
		result += " Strings:";
		for (const auto& s : strings)
			result += " " + s;
		str = result;
	}

	struct Data {
		int a, b, c, d;
	};

	extern "C" __attribute__((used, retain))
	void test2(Data* data) {
		assert(data->a == 1);
		assert(data->b == 2);
		assert(data->c == 3);
		assert(data->d == 4);
		data->a = 5;
		data->b = 6;
		data->c = 7;
		data->d = 8;
	}

	extern "C" __attribute__((used, retain))
	int test3(std::vector<std::vector<int>>& vec) {
		assert(vec.size() == 2);
		assert(vec[0].size() == 3);
		assert(vec[1].size() == 2);
		assert(vec[0][0] == 1);
		assert(vec[0][1] == 2);
		assert(vec[0][2] == 3);
		assert(vec[1][0] == 4);
		assert(vec[1][1] == 5);

		vec.at(1).push_back(666);
		return 666;
	}

	int main() {
		return 666;
	})M", "-O2 -static -x c " + cwd + "/include/native_libc.h -x c++ ", true);

	riscv::Machine<RISCV64> machine { binary };
	setup_native_system_calls(machine);
	machine.setup_linux_syscalls();
	machine.setup_linux(
		{"vmcall"},
		{"LC_TYPE=C", "LC_ALL=C", "USER=root"});

	machine.simulate(MAX_INSTRUCTIONS);
	REQUIRE(machine.return_value<int>() == 666);

	// Define the test6 function, which has a std::string& argument in the guest,
	// and a std::vector<int>& and std::vector<std::string>&. The stack is writable,
	// so the guest can choose whether or not to use const references.
	const unsigned allocs_before = machine.arena().allocation_counter() - machine.arena().deallocation_counter();

	for (int i = 0; i < 10; i++) {
		// Create a GuestStdString object with a string
		ScopedCppString str(machine);
		REQUIRE(str->empty());
		str = "C++ World ..SSO..";
		REQUIRE(str->to_string(machine) == "C++ World ..SSO..");

		// Create a GuestStdVector object with a vector of integers
		ScopedCppVector<int> ivec(machine);
		REQUIRE(ivec->empty());
		ivec = std::vector<int>{ 1, 2, 3 };
		REQUIRE(ivec->size() == 3);
		ivec->assign(machine, std::vector<int>{ 1, 2, 3, 4, 5 });
		REQUIRE(ivec->size() == 5);

		// Create a vector of strings using a specialization for std::string
		ScopedCppVector<CppString> svec(machine,
			std::vector<std::string>{ "Hello,", "World!", "This string is long :)" });
		REQUIRE(svec->size() == 3);

		machine.vmcall("test", str, ivec, svec);

		// Check that the string was modified
		REQUIRE(str->to_string(machine) == "Hello, C++ World ..SSO..! Integers: 1 2 3 4 5 Strings: Hello, World! This string is long :)");
	}

	// Check that the number of active allocations is the same as before the test
	const unsigned allocs_now = machine.arena().allocation_counter() -
		machine.arena().deallocation_counter();
	REQUIRE(allocs_now == allocs_before);

	// Test the second function
	for (int i = 0; i < 10; i++) {
		// Scoped arena objects are guest-heap allocated, which means we can read back data
		// from the guest after the function call
		struct Data {
			int a, b, c, d;
		};
		ScopedArenaObject<RISCV64, Data> data(machine, Data{1, 2, 3, 4});

		machine.vmcall("test2", data);

		// Check that the struct was modified
		REQUIRE(data->a == 5);
		REQUIRE(data->b == 6);
		REQUIRE(data->c == 7);
		REQUIRE(data->d == 8);
	}

	const unsigned allocs_after2 = machine.arena().allocation_counter() -
		machine.arena().deallocation_counter();
	REQUIRE(allocs_after2 == allocs_before);

	// Test the third function
	for (int i = 0; i < 10; i++) {
		ScopedCppVector<CppVector<int>> vec(machine);
		vec->push_back(machine, std::vector<int>{1, 2, 3});
		vec->push_back(machine, std::vector<int>{4, 5});
		REQUIRE(vec->size() == 2);
		REQUIRE(vec->capacity() >= 2);
		vec->clear(machine);
		REQUIRE(vec->empty());
		REQUIRE(vec->capacity() >= 2);
		vec->push_back(machine, std::vector<int>{1, 2, 3});
		vec->push_back(machine, std::vector<int>{4, 5});
		REQUIRE(vec->size() == 2);
		// Using reserve increases the capacity, but not the size
		vec->reserve(machine, 16);
		REQUIRE(vec->capacity() >= 16);
		REQUIRE(vec->size() == 2);
		// Check that the vectors were correctly initialized
		REQUIRE(vec->at(machine, 0).size() == 3);
		REQUIRE(vec->at(machine, 1).size() == 2);
		REQUIRE(vec->at(machine, 0).at(machine, 0) == 1);
		REQUIRE(vec->at(machine, 0).at(machine, 1) == 2);
		REQUIRE(vec->at(machine, 0).at(machine, 2) == 3);
		REQUIRE(vec->at(machine, 1).at(machine, 0) == 4);
		REQUIRE(vec->at(machine, 1).at(machine, 1) == 5);

		const int ret = machine.vmcall("test3", vec);

		// Check that the function returned the expected value
		REQUIRE(ret == 666);
		REQUIRE(vec->size() == 2);
		// We modified the second vector, adding an element
		REQUIRE(vec->at(machine, 1).size() == 3);
		REQUIRE(vec->at(machine, 1).at(machine, 2) == 666);

		// Test iterators (slightly more complex)
		size_t count = 0;
		auto begin = vec->begin(machine);
		auto end = vec->end(machine);
		for (auto it = begin; it != end; ++it) {
			auto& v = *it;
			for (size_t i = 0; i < v.size(); i++) {
				count += v.at(machine, i);
			}
		}
		REQUIRE(count == 1 + 2 + 3 + 4 + 5 + 666);
	}

	const unsigned allocs_after3 = machine.arena().allocation_counter() -
		machine.arena().deallocation_counter();
	REQUIRE(allocs_after3 == allocs_before);
}
