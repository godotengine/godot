
// Use a trick to access the Machine directly on g++/clang, Linux-only for now
#if (defined(__GNUG__) || defined(__clang__)) && defined(__linux__)
template <int W> RISCV_ALWAYS_INLINE inline
Machine<W>& CPU<W>::machine() noexcept { return *reinterpret_cast<Machine<W>*> (this); }
template <int W> RISCV_ALWAYS_INLINE inline
const Machine<W>& CPU<W>::machine() const noexcept { return *reinterpret_cast<const Machine<W>*> (this); }
#else
template <int W> RISCV_ALWAYS_INLINE inline
Machine<W>& CPU<W>::machine() noexcept { return this->m_machine; }
template <int W> RISCV_ALWAYS_INLINE inline
const Machine<W>& CPU<W>::machine() const noexcept { return this->m_machine; }
#endif

template <int W> RISCV_ALWAYS_INLINE inline
Memory<W>& CPU<W>::memory() noexcept { return machine().memory; }
template <int W> RISCV_ALWAYS_INLINE inline
const Memory<W>& CPU<W>::memory() const noexcept { return machine().memory; }

template <int W>
inline CPU<W>::CPU(Machine<W>& machine)
	: m_machine { machine }, m_exec(empty_execute_segment().get())
{
}
template <int W>
inline void CPU<W>::reset_stack_pointer() noexcept
{
	// initial stack location
	this->reg(2) = machine().memory.stack_initial();
}

template<int W>
inline void CPU<W>::jump(const address_t dst)
{
	// it's possible to jump to a misaligned address
	if constexpr (!compressed_enabled) {
		if (UNLIKELY(dst & 0x3)) {
			trigger_exception(MISALIGNED_INSTRUCTION, dst);
		}
	} else {
		if (UNLIKELY(dst & 0x1)) {
			trigger_exception(MISALIGNED_INSTRUCTION, dst);
		}
	}
	this->registers().pc = dst;
}

template<int W>
inline void CPU<W>::aligned_jump(const address_t dst) noexcept
{
	this->registers().pc = dst;
}

template<int W>
inline void CPU<W>::increment_pc(int delta) noexcept
{
	registers().pc += delta;
}
