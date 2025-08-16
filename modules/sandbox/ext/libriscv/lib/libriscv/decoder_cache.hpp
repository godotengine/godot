#pragma once
#include "common.hpp"
#include "types.hpp"
#include <unordered_map>
#include <vector>

namespace riscv {

template <int W>
struct DecoderData {
	using Handler = instruction_handler<W>;

	uint8_t  m_bytecode;
	uint8_t  m_handler;
#ifdef RISCV_EXT_COMPRESSED
	uint16_t idxend  : 8;
	uint16_t icount  : 8;
#else
	uint16_t idxend;
#endif

	uint32_t instr;

	// Switch-based and threaded simulation uses bytecodes.
	RISCV_ALWAYS_INLINE
	auto get_bytecode() const noexcept {
		return this->m_bytecode;
	}
	void set_bytecode(uint16_t num) noexcept {
		this->m_bytecode = num;
	}

	void set_handler(Instruction<W> insn) noexcept {
		this->set_insn_handler(insn.handler);
	}
	void set_insn_handler(instruction_handler<W> ih) noexcept {
		this->m_handler = handler_index_for(ih);
	}
	void set_invalid_handler() noexcept {
		this->m_handler = 0;
	}
	bool is_invalid_handler() const noexcept {
		return this->m_handler == 0;
	}

	// Used by live-patching to set both bytecode and handler index.
	void set_atomic_bytecode_and_handler(uint8_t bytecode, uint8_t handler_idx) noexcept {
		// XXX: Assumes little-endian
		*(uint16_t* )&m_bytecode = ( handler_idx << 8 ) | bytecode;
	}

	RISCV_ALWAYS_INLINE
	auto block_bytes() const noexcept {
		return idxend * (compressed_enabled ? 2 : 4);
	}
	RISCV_ALWAYS_INLINE
	auto instruction_count() const noexcept {
#ifdef RISCV_EXT_COMPRESSED
		return icount;
#else
		return idxend + 1;
#endif
	}

	bool operator==(const DecoderData<W>& other) const noexcept {
		return m_bytecode == other.m_bytecode &&
			m_handler == other.m_handler &&
			idxend == other.idxend &&
			instr == other.instr;
	}

	static size_t handler_index_for(Handler new_handler);
	static Handler* get_handlers() noexcept {
		return &instr_handlers[0];
	}

	void atomic_overwrite(const DecoderData<W>& other) noexcept {
		static_assert(sizeof(DecoderData<W>) == 8, "DecoderData size mismatch");
		*(uint64_t*)this = *(uint64_t*)&other;
	}
private:
	static inline std::array<Handler, 256> instr_handlers;
	static inline std::size_t handler_count = 0;
	static inline std::unordered_map<Handler, size_t> handler_cache;
};

template <int W>
struct DecoderCache
{
	static constexpr size_t DIVISOR = (compressed_enabled) ? 2 : 4;
	static constexpr unsigned SHIFT = (compressed_enabled) ? 1 : 2;

	inline auto& get(size_t idx) noexcept {
		return cache[idx];
	}

	inline auto* get_base() noexcept {
		return &cache[0];
	}

	std::array<DecoderData<W>, PageSize / DIVISOR> cache;
};

}
