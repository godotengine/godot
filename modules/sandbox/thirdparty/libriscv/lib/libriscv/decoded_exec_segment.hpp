#pragma once
#include <memory>
#include "types.hpp"
#include <mutex>
#include <condition_variable>
#include <unordered_set>

namespace riscv
{
	template<int W> struct DecoderCache;
	template<int W> struct DecoderData;

	// A fully decoded execute segment
	template <int W>
	struct DecodedExecuteSegment
	{
		using address_t = address_type<W>;

		bool is_within(address_t addr, size_t len = 2) const noexcept {
			address_t addr_end;
#ifdef _MSC_VER
			addr_end = addr + len;
			return addr >= m_vaddr_begin && addr_end <= m_vaddr_end && (addr_end > addr);
#else
			if (!__builtin_add_overflow(addr, len, &addr_end))
				return addr >= m_vaddr_begin && addr_end <= m_vaddr_end;
#endif
			return false;
		}

		auto* exec_data(address_t pc = 0) const noexcept {
			return m_exec_pagedata.get() - m_exec_pagedata_base + pc;
		}

		address_t exec_begin() const noexcept { return m_vaddr_begin; }
		address_t exec_end() const noexcept { return m_vaddr_end; }
		address_t pagedata_base() const noexcept { return m_exec_pagedata_base; }

		auto* decoder_cache() noexcept { return m_exec_decoder; }
		auto* decoder_cache() const noexcept { return m_exec_decoder; }
		auto* decoder_cache_base() const noexcept { return m_decoder_cache.get(); }
		size_t decoder_cache_size() const noexcept { return m_decoder_cache_size; }

		auto* create_decoder_cache(DecoderCache<W>* cache, size_t size) {
			m_decoder_cache.reset(cache);
			m_decoder_cache_size = size;
			return m_decoder_cache.get();
		}
		void set_decoder(DecoderData<W>* dec) { m_exec_decoder = dec; }

		size_t size_bytes() const noexcept {
			return sizeof(*this) + m_exec_pagedata_size + m_decoder_cache_size; // * sizeof(DecoderCache<W>);
		}
		bool empty() const noexcept { return m_exec_pagedata_size == 0; }

		DecodedExecuteSegment() = default;
		DecodedExecuteSegment(address_t pbase, size_t len, address_t vaddr, size_t exlen);
		DecodedExecuteSegment(DecodedExecuteSegment&&);
		~DecodedExecuteSegment();

		size_t threaded_rewrite(size_t bytecode, address_t pc, rv32i_instruction& instr);

		uint32_t crc32c_hash() const noexcept { return m_crc32c_hash; }
		void set_crc32c_hash(uint32_t hash) { m_crc32c_hash = hash; }

#ifdef RISCV_BINARY_TRANSLATION
		bool is_binary_translated() const noexcept { return !m_translator_mappings.empty(); }
		bool is_libtcc() const noexcept { return m_is_libtcc; }
		void* binary_translation_so() const { return m_bintr_dl; }
		void set_binary_translated(void* dl, bool is_libtcc) const { m_bintr_dl = dl; m_is_libtcc = is_libtcc; }
		uint32_t translation_hash() const { return m_bintr_hash; }
		void set_translation_hash(uint32_t hash) { m_bintr_hash = hash; }
		auto& create_mappings(size_t mappings) { m_translator_mappings.resize(mappings); return m_translator_mappings; }
		void set_mapping(unsigned i, bintr_block_func<W> handler) { m_translator_mappings.at(i) = handler; }
		bintr_block_func<W> mapping_at(unsigned i) const { return m_translator_mappings.at(i); }
		bintr_block_func<W> unchecked_mapping_at(unsigned i) const { return m_translator_mappings[i]; }
		size_t translator_mappings() const noexcept { return m_translator_mappings.size(); }
		auto* patched_decoder_cache() noexcept { return m_patched_exec_decoder; }
		void set_patched_decoder_cache(std::unique_ptr<DecoderCache<W>[]> cache, DecoderData<W>* dec)
			{ m_patched_decoder_cache = std::move(cache); m_patched_exec_decoder = dec; }

		void set_record_slowpaths(bool do_record) { m_do_record_slowpaths = do_record; }
		bool is_recording_slowpaths() const noexcept { return m_do_record_slowpaths; }
		void wait_for_compilation_complete() {
			std::unique_lock<std::mutex> lock(m_background_compilation_mutex);
			m_background_compilation_cv.wait(lock, [this]{ return !m_is_background_compiling; });
		}
		bool is_background_compiling() const noexcept { return m_is_background_compiling; }
		void set_background_compiling(bool is_bg) {
			std::lock_guard<std::mutex> lock(m_background_compilation_mutex);
			if (m_is_background_compiling && !is_bg) {
				m_background_compilation_cv.notify_all();
			}
			m_is_background_compiling = is_bg;
		}
#ifdef RISCV_DEBUG
		void insert_slowpath_address(address_t addr) { m_slowpath_addresses.insert(addr); }
		auto& slowpath_addresses() const noexcept { return m_slowpath_addresses; }
#endif
#else
		bool is_binary_translated() const noexcept { return false; }
		bool is_libtcc() const noexcept { return false; }
#endif

		bool is_execute_only() const noexcept { return m_is_execute_only; }
		void set_execute_only(bool is_xo) { m_is_execute_only = is_xo; }

		bool is_likely_jit() const noexcept { return m_is_likely_jit; }
		void set_likely_jit(bool is_jit) { m_is_likely_jit = is_jit; }

		bool is_stale() const noexcept { return m_is_stale; }
		void set_stale(bool is_stale) { m_is_stale = is_stale; }

	private:
		address_t m_vaddr_begin = 0;
		address_t m_vaddr_end   = 0;
		DecoderData<W>* m_exec_decoder = nullptr;

		// The flat execute segment is used to execute
		// the CPU::simulate_precise function in order to
		// support debugging, as well as when producing
		// the decoder cache
		size_t    m_exec_pagedata_size = 0;
		address_t m_exec_pagedata_base = 0;
		std::unique_ptr<uint8_t[]> m_exec_pagedata = nullptr;

		// Decoder cache is used to run bytecode simulation at a high speed
		size_t          m_decoder_cache_size = 0;
		std::unique_ptr<DecoderCache<W>[]> m_decoder_cache = nullptr;

#ifdef RISCV_BINARY_TRANSLATION
		std::vector<bintr_block_func<W>> m_translator_mappings;
		std::unique_ptr<DecoderCache<W>[]> m_patched_decoder_cache = nullptr;
		DecoderData<W>* m_patched_exec_decoder = nullptr;
		mutable void* m_bintr_dl = nullptr;
#ifdef RISCV_DEBUG
		std::unordered_set<address_t> m_slowpath_addresses;
#endif
		uint32_t m_bintr_hash = 0x0; // CRC32-C of the execute segment + compiler options
#endif
		uint32_t m_crc32c_hash = 0x0; // CRC32-C of the execute segment
		bool m_is_execute_only = false;
#ifdef RISCV_BINARY_TRANSLATION
		bool m_do_record_slowpaths = false;
		mutable bool m_is_libtcc = false;
		bool m_is_background_compiling = false;
		mutable std::mutex m_background_compilation_mutex;
		std::condition_variable m_background_compilation_cv;
#endif
		// High-memory execute segments are likely to be JIT'd, and needs to
		// be nuked when attempting to re-use the segment
		bool m_is_likely_jit = false;
		bool m_is_stale = false;
	};

	template <int W>
	inline DecodedExecuteSegment<W>::DecodedExecuteSegment(
		address_t pbase, size_t len, address_t exaddr, size_t exlen)
	{
		m_vaddr_begin = exaddr;
		m_vaddr_end   = exaddr + exlen;
		// Prevent zero-length allocation warning by ensuring minimum size
		const size_t alloc_len = (len > 0) ? len : 1;
		m_exec_pagedata.reset(new uint8_t[alloc_len]);
		m_exec_pagedata_size = len;
		m_exec_pagedata_base = pbase;
	}

	template <int W>
	inline DecodedExecuteSegment<W>::DecodedExecuteSegment(DecodedExecuteSegment&& other)
	{
		m_vaddr_begin = other.m_vaddr_begin;
		m_vaddr_end   = other.m_vaddr_end;
		m_exec_decoder = other.m_exec_decoder;
		other.m_exec_decoder = nullptr;

		m_exec_pagedata_size = other.m_exec_pagedata_size;
		m_exec_pagedata_base = other.m_exec_pagedata_base;
		m_exec_pagedata = std::move(other.m_exec_pagedata);

		m_decoder_cache_size = other.m_decoder_cache_size;
		m_decoder_cache = std::move(other.m_decoder_cache);

#ifdef RISCV_BINARY_TRANSLATION
		m_translator_mappings = std::move(other.m_translator_mappings);
		m_bintr_dl = other.m_bintr_dl;
		other.m_bintr_dl = nullptr;
		m_bintr_hash = other.m_bintr_hash;
		m_is_libtcc = other.m_is_libtcc;
		m_patched_decoder_cache = std::move(other.m_patched_decoder_cache);
		m_patched_exec_decoder = other.m_patched_exec_decoder;
#endif
	}

	template <int W>
	inline DecodedExecuteSegment<W>::~DecodedExecuteSegment()
	{
#ifdef RISCV_BINARY_TRANSLATION
		extern void  dylib_close(void* dylib, bool is_libtcc);
		if (m_bintr_dl)
			dylib_close(m_bintr_dl, m_is_libtcc);
		m_bintr_dl = nullptr;
		// Wait for any background compilation to finish
		wait_for_compilation_complete();
#endif
	}

} // riscv
