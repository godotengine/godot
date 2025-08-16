#include "server.hpp"

#include <libriscv/machine.hpp>
#include <libriscv/threads.hpp>
using namespace httplib;

// Avoid endless loops, code that takes too long and excessive memory usage
static const uint64_t MAX_BINARY       = 32'000'000UL;
static const uint64_t MAX_INSTRUCTIONS = 36'000'000UL;
static const uint64_t MAX_MEMORY       = 32UL * 1024 * 1024;
static const size_t   NUM_SAMPLES      = 50;

static const std::vector<std::string> env = {
	"LC_CTYPE=C", "LC_ALL=C", "USER=groot"
};

extern uint64_t micros_now();
extern uint64_t monotonic_micros_now();

static void
protected_execute(const Request& req, Response& res, const ContentReader& creader)
{
	std::vector<uint8_t> binary;
	creader([&] (const char* data, size_t data_length) {
		if (binary.size() + data_length > MAX_BINARY) return false;
		binary.insert(binary.end(), data, data + data_length);
		return true;
	});
	if (binary.empty()) {
		res.status = 400;
		res.set_header("X-Error", "Empty binary");
	}

	// go-time: create machine, execute code
	const riscv::MachineOptions<riscv::RISCV64> options {
		.memory_max = MAX_MEMORY
	};
	riscv::Machine<riscv::RISCV64> machine { binary, options };

	machine.setup_linux({"program"}, env);
	machine.setup_linux_syscalls();
	machine.setup_posix_threads();

	struct BenchmarkState {
		bool benchmark = false;
		uint64_t begin_ic = 0;
		uint64_t bench_time = 0;
		uint64_t bench_ic = 0;
		uint64_t first = 0;
		std::vector<uint64_t> samples;
		std::string output;
	} state;
	machine.set_userdata(&state);

	machine.set_printer(
	[] (auto& machine, const char* text, size_t len) {
		auto* state = machine.template get_userdata<BenchmarkState>();

		state->output.append(text, len);
	});

	// Stop (pause) the machine when he hit a trap/break instruction
	machine.install_syscall_handler(riscv::SYSCALL_EBREAK,
	[] (auto& machine) {
		auto* state = machine.template get_userdata<BenchmarkState>();
		const auto addr = machine.template sysarg<uint64_t>(0);

		if (state->benchmark) throw std::runtime_error("Already benchmarking");
		state->benchmark = true;

		auto pf = machine.get_printer();
		machine.set_printer([] (auto&, const char*, size_t) {});
		uint64_t ic = machine.instruction_counter();

		asm("" : : : "memory");
		const uint64_t t0 = micros_now();
		asm("" : : : "memory");

		machine.preempt(~0ULL, addr);

		asm("" : : : "memory");
		const uint64_t t1 = micros_now();
		asm("" : : : "memory");

		machine.set_printer(pf);
		state->benchmark = false;
		if (state->begin_ic == 0) {
			state->begin_ic = machine.instruction_counter();
			state->bench_time = t0;
			state->bench_ic = machine.instruction_counter() - ic;
			state->first = t1 - t0;
		} else {
			state->samples.push_back(t1 - t0);
		}
	});

	// Execute until we have hit a break
	const uint64_t st0 = micros_now();
	asm("" : : : "memory");
	try {
		machine.simulate(MAX_INSTRUCTIONS);
	} catch (std::exception& e) {
		res.set_header("X-Exception", e.what());
	}
	asm("" : : : "memory");
	const uint64_t st2 = micros_now();
	asm("" : : : "memory");
	res.set_header("X-Startup-Time", std::to_string(state.bench_time - st0));
	res.set_header("X-Total-Time", std::to_string(st2 - st0));
	res.set_header("X-Startup-Instructions", std::to_string(state.begin_ic));
	// Cache for 2 seconds (the benchmark results)
	res.set_header("Cache-Control", "max-age=2");

	auto& samples = state.samples;
	if (!samples.empty()) {
		std::sort(samples.begin(), samples.end());
		const uint64_t lowest = samples[0];
		const uint64_t median = samples[samples.size() / 2];
		const uint64_t highest = samples[samples.size()-1];
		res.set_header("X-Runtime-Samples", std::to_string(samples.size()));
		res.set_header("X-Runtime-Lowest", std::to_string(lowest));
		res.set_header("X-Runtime-Median", std::to_string(median));
		res.set_header("X-Runtime-Highest", std::to_string(highest));
	}

	// Instruction count of first benchmark
	res.set_header("X-Runtime-First", std::to_string(state.first));
	res.set_header("X-Instruction-Count", std::to_string(state.bench_ic));

	res.set_header("X-Binary-Size", std::to_string(binary.size()));
	const size_t active_mem = machine.memory.pages_active() * 4096;
	res.set_header("X-Memory-Usage", std::to_string(active_mem));
	res.set_header("X-Memory-Max", std::to_string(MAX_MEMORY));
	res.set_content(state.output, "text/plain");

	// A0 is both a return value and first argument, matching
	// any calls to exit()
	const int exit_code = machine.cpu.reg(10);
	res.status = 200;
	res.set_header("X-Exit-Code", std::to_string(exit_code));
}

void execute(const Request& req, Response& res, const ContentReader& creader)
{
	try {
		protected_execute(req, res, creader);
	} catch (std::exception& e) {
		res.status = 200;
		res.set_header("X-Error", e.what());
	}
}
