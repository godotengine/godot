#include <cstdint>

namespace riscv
{
    template <int W> struct Machine;

	// In fastsim mode the instruction counter becomes a register
	// the function, and we only update m_counter in Machine on exit
	// When binary translation is enabled we cannot do this optimization.
	struct InstrCounter
	{
		InstrCounter(uint64_t icounter, uint64_t maxcounter)
		  : m_counter(icounter),
			m_max(maxcounter)
		{}
		~InstrCounter() = default;

		template <int W>
		void apply(Machine<W>& machine) {
			machine.set_instruction_counter(m_counter);
			machine.set_max_instructions(m_max);
		}
		template <int W>
		void apply_counter(Machine<W>& machine) {
			machine.set_instruction_counter(m_counter);
		}
		// Used by binary translator to compensate for its own function already being counted
		// TODO: Account for this inside the binary translator instead. Very minor impact.
		template <int W>
		void apply_counter_minus_1(Machine<W>& machine) {
			machine.set_instruction_counter(m_counter-1);
			machine.set_max_instructions(m_max);
		}
		template <int W>
		void retrieve_max_counter(Machine<W>& machine) {
			m_max     = machine.max_instructions();
		}
		template <int W>
		void retrieve_counters(Machine<W>& machine) {
			m_counter = machine.instruction_counter();
			m_max     = machine.max_instructions();
		}

		uint64_t value() const noexcept {
			return m_counter;
		}
		uint64_t max() const noexcept {
			return m_max;
		}
		void stop() noexcept {
			m_max = 0; // This stops the machine
		}
		void set_counters(uint64_t value, uint64_t max) {
			m_counter = value;
			m_max     = max;
		}
		void increment_counter(uint64_t cnt) {
			m_counter += cnt;
		}
		bool overflowed() const noexcept {
			return m_counter >= m_max;
		}
	private:
		uint64_t m_counter;
		uint64_t m_max;
	};
} // riscv
