#pragma once
#include "elf.hpp"
#include "page.hpp"
#include <cassert>
#include <cstring>
#include <string_view>
#include <unordered_map>
#include "decoded_exec_segment.hpp"
#include "mmap_cache.hpp"
#include "util/buffer.hpp" // <string>
#include "util/function.hpp"
#if RISCV_SPAN_AVAILABLE
#include <span>
#endif

namespace riscv
{
	template<int W> struct Machine;
	struct vBuffer { char* ptr; size_t len; };

	template<int W>
	struct alignas(RISCV_MACHINE_ALIGNMENT) Memory
	{
		using address_t = address_type<W>;
		using mmio_cb_t = Page::mmio_cb_t;
		using page_fault_cb_t = riscv::Function<Page&(Memory&, address_t, bool)>;
		using page_readf_cb_t = riscv::Function<const Page&(const Memory&, address_t)>;
		using page_write_cb_t = riscv::Function<void(Memory&, address_t, Page&)>;
		static constexpr address_t BRK_MAX      = RISCV_BRK_MEMORY_SIZE; // Default BRK size
		static constexpr address_t DYLINK_BASE  = 0x40000; // Dynamic link base address
		static constexpr address_t RWREAD_BEGIN = 0x1000; // Default rw-arena rodata start

		template <typename T>
		T read(address_t src);

		template <typename T>
		T& writable_read(address_t src);

		template <typename T>
		void write(address_t dst, T value);

		template <typename T>
		void write_paging(address_t dst, T value);

		void memset(address_t dst, uint8_t value, size_t len);
		void memcpy(address_t dst, const void* src, size_t);
		void memcpy(address_t dst, Machine<W>& srcm, address_t src, address_t len);
		void memcpy_out(void* dst, address_t src, size_t) const;
		// Copy between overlapping memory regions. Only available when flat_readwrite_arena is enabled!
		bool try_memmove(address_t dst, address_t src, size_t len);
		// Compare bounded memory
		int memcmp(address_t p1, address_t p2, size_t len) const;
		int memcmp(const void* p1, address_t p2, size_t len) const;
		// Perform the equivalent of MADV_DONTNEED on memory region
		void memdiscard(address_t dst, size_t len, bool ignore_protections);
		/* Fill an array of buffers pointing to complete guest virtual [addr, len].
		   Throws an exception if there was a protection violation.
		   Returns the number of buffers filled, or an exception if not enough. */
		size_t gather_buffers_from_range(size_t cnt, vBuffer[], address_t addr, size_t len) const;
		size_t gather_writable_buffers_from_range(size_t cnt, vBuffer[], address_t addr, size_t len);
		// Gather fragmented virtual memory into a buffer abstraction that can output
		// to a vector, a string and check sequentiality.
		riscv::Buffer membuffer(address_t addr, size_t len, size_t maxlen = 16ul << 20) const;

		/// @brief View known-sequential read/write virtual memory (or throw exception)
		/// @param addr The address to start reading from
		/// @param len The number of bytes to read
		/// @param maxlen The maximum number of bytes to read
		/// @return A string_view of the readable memory
		/// @note Do not attempt to write to the memory region, as it may be read-only
		std::string_view memview(address_t addr, size_t len, size_t maxlen = 16ul << 20) const;

		/// @brief View known-sequential writable virtual memory (or throw exception)
		/// @param addr The address to start reading from
		/// @param len The number of bytes to read
		/// @param maxlen The maximum number of bytes to read
		/// @return A string_view of the readable/writable memory
		std::string_view writable_memview(address_t addr, size_t len, size_t maxlen = 16ul << 20) const;

#ifdef RISCV_SPAN_AVAILABLE
		// View known-sequential virtual memory as array of T with given number of elements (or throw exception)
		// This always uses writable_memview() under the hood to ensure the memory is writable
		template <typename T>
		std::span<T> memspan(address_t addr, size_t elements, size_t maxlen = 16ul << 20) const;
		template <typename T, size_t N>
		std::span<T, N> memspan(address_t addr, size_t maxlen = 16ul << 20) const { return memspan<T>(addr, N, maxlen).template first<N>(); }
#endif

		/// @brief View a fixed-size array of T from a known-sequential virtual memory region
		/// @tparam T The type of the array elements. If T is const, the memory is read-only.
		/// @param addr The address to start reading from
		/// @return A pointer to the array, or nullptr if the memory region is not sequential or too large
		template <typename T, size_t N>
		std::array<T, N>* memarray(address_t addr) const;

		/// @brief View a dynamically-sized array of T from a known-sequential virtual memory region
		/// @tparam T The type of the array elements. If T is const, the memory is read-only.
		/// @param addr The address to start reading from
		/// @param len The number of elements to read
		/// @param maxbytes The maximum number of bytes to read
		/// @return A pointer to the array, or nullptr if the memory region is not sequential or too large
		template <typename T>
		T* memarray(address_t addr, size_t len, size_t maxbytes = 16ul << 20) const;

		// Try viewing a dynamically-sized array of T from a known-sequential virtual memory region
		// Returns nullptr if the memory region is not sequential or too large
		template <typename T>
		T* try_memarray(address_t addr, size_t len, size_t maxbytes = 16ul << 20) const;
		// Read zero-terminated string directly from guests memory, into an owning std::string
		std::string memstring(address_t addr, size_t maxlen = 16384) const;
		size_t strlen(address_t addr, size_t maxlen = 16384) const;
		// Read zero-terminated string directly from guests memory, returning a non-owning string_view
		std::string_view memstring_view(address_t addr, size_t maxlen = 16384) const;

		// Returns the ELF entry/start address (the first instruction)
		address_t start_address() const noexcept { return this->m_start_address; }
		// Returns the current initial stack pointer (unrelated to SP register)
		address_t stack_initial() const noexcept { return this->m_stack_address; }
		void set_stack_initial(address_t addr) { this->m_stack_address = addr; }
		// Returns the address used for exiting (returning from) a vmcall()
		address_t exit_address() const noexcept;
		void      set_exit_address(address_t new_exit);
		// The initial heap address (*not* the current heap maximum)
		address_t heap_address() const noexcept { return this->m_heap_address; }
		// Simple memory mapping implementation
		auto& mmap_cache() noexcept { return m_mmap_cache; }
		address_t mmap_start() const noexcept { return this->m_heap_address + BRK_MAX; }
		const address_t& mmap_address() const noexcept { return m_mmap_address; }
		address_t& mmap_address() noexcept { return m_mmap_address; }
		// Allocate at least writable bytes through mmap(), and return the page-aligned address
		address_t mmap_allocate(address_t bytes);
		// Attempts to relax a previous call to mmap_allocate(), freeing space at the end
		bool mmap_relax(address_t addr, address_t size, address_t new_size);
		// Unmap a memory range
		bool mmap_unmap(address_t addr, address_t size);


		Machine<W>& machine() noexcept { return this->m_machine; }
		const Machine<W>& machine() const noexcept { return this->m_machine; }
		bool is_forked() const noexcept { return !this->m_original_machine; }

#ifdef RISCV_EXT_ATOMICS
		auto& atomics() noexcept { return this->m_atomics; }
		const auto& atomics() const noexcept { return this->m_atomics; }
#endif // RISCV_EXT_ATOMICS

		// Symbol table and section lookup functions
		address_t resolve_address(std::string_view sym) const;
		address_t resolve_section(const char* name) const;
		// Basic backtraces and symbol lookups
		struct Callsite {
			std::string name = "(null)";
			address_t   address = 0x0;
			uint32_t    offset  = 0x0;
			size_t      size    = 0;
		};
		Callsite lookup(address_t) const;
		void print_backtrace(std::function<void(std::string_view)>, bool ra = true) const;

		// Get list of all symbols in the binary
		std::vector<const char*> all_symbols() const;
		// Get list of all unmangled symbols in the binary that starts with a given prefix
		// If the prefix is empty, all unmangled functions are returned
		std::vector<std::string_view> all_unmangled_function_symbols(const std::string& prefix = "") const;
		// Get list of all comments in the ELF binary
		std::vector<std::string_view> elf_comments() const;

		// Counts all the memory used by the machine, execute segments, pages, etc.
		uint64_t memory_usage_total() const noexcept;
		// Helpers for memory usage
		size_t pages_active() const noexcept { return m_pages.size(); }
		size_t owned_pages_active() const noexcept;
		// Page handling
		const auto& pages() const noexcept { return m_pages; }
		auto& pages() noexcept { return m_pages; }
		const Page& get_page(address_t) const;
		const Page& get_exec_pageno(address_t npage) const; // throws
		const Page& get_pageno(address_t npage) const;
		const Page& get_readable_pageno(address_t npage) const;
		Page& create_writable_pageno(address_t npage, bool initialize = true);
		void  set_page_attr(address_t, size_t len, PageAttributes);
		void set_pageno_attr(address_t pageno, PageAttributes);
		std::string get_page_info(address_t addr) const;
		static inline address_t page_number(const address_t address) noexcept {
			return address / Page::size();
		}
		CachedPage<W, const PageData>& rdcache() const noexcept {
			return m_rd_cache;
		}
		CachedPage<W, PageData>& wrcache() noexcept {
			return m_wr_cache;
		}
		const PageData& cached_readable_page(address_t, size_t) const;
		PageData& cached_writable_page(address_t);
		// Page creation & destruction
		template <typename... Args>
		Page& allocate_page(address_t page, Args&& ...);
		void  invalidate_cache(address_t pageno, Page*) const noexcept;
		void  invalidate_reset_cache() const noexcept;
		void  free_pages(address_t, size_t len);
		bool  free_pageno(address_t pageno);

		// Event for writing to unused/unknown memory
		// The old handler is returned, so it can be restored later.
		page_fault_cb_t set_page_fault_handler(page_fault_cb_t h) {
			auto old_handler = std::move(m_page_fault_handler);
			this->m_page_fault_handler = h;
			return old_handler;
		}

		// Event for reading unused/unknown memory
		// The old handler is returned, so it can be restored later.
		page_readf_cb_t set_page_readf_handler(page_readf_cb_t h) {
			auto old_handler = std::move(m_page_readf_handler);
			this->m_page_readf_handler = h;
			return old_handler;
		}
		void reset_page_readf_handler() { this->m_page_readf_handler = default_page_read; }

		// Event for writes on copy-on-write pages
		void set_page_write_handler(page_write_cb_t h) { this->m_page_write_handler = h; }
		static void default_page_write(Memory&, address_t, Page& page);
		static const Page& default_page_read(const Memory&, address_t);
		// NOTE: use print_and_pause() to immediately break!
		void trap(address_t page_addr, mmio_cb_t callback);
		// shared pages (regular pages will have priority!)
		Page&  install_shared_page(address_t pageno, const Page&);
		// create pages for non-owned (shared) memory with given attributes
		void insert_non_owned_memory(
			address_t dst, void* src, size_t size, PageAttributes = {});

		// Custom execute segment, returns page base, final size and execute segment pointer
		std::shared_ptr<DecodedExecuteSegment<W>>& exec_segment_for(address_t vaddr);
		const std::shared_ptr<DecodedExecuteSegment<W>>& exec_segment_for(address_t vaddr) const;
		DecodedExecuteSegment<W>& create_execute_segment(const MachineOptions<W>&, const void* data, address_t addr, size_t len, bool is_initial, bool is_likely_jit = false);
		size_t execute_segments_count() const noexcept { return m_exec.size(); }
		// Evict all execute segments, also disabling the main execute segment
		void evict_execute_segments();
		void evict_execute_segment(DecodedExecuteSegment<W>&);
#ifdef RISCV_BINARY_TRANSLATION
		std::vector<address_t> gather_jump_hints() const;
#endif

		const auto& binary() const noexcept { return m_binary; }
		void reset();
		bool is_dynamic_executable() const noexcept { return this->m_is_dynamic; }
		address_t elf_base_address(address_t offset) const;

		bool uses_flat_memory_arena() const noexcept { return riscv::flat_readwrite_arena && this->m_arena.data != nullptr; }
		bool uses_Nbit_encompassing_arena() const noexcept { return riscv::encompassing_Nbit_arena != 0 && this->m_arena.data != nullptr; }
		void* memory_arena_ptr() const noexcept { return (void *)this->m_arena.data; }
		auto& memory_arena_ptr_ref() const noexcept { return this->m_arena.data; }
		size_t memory_arena_size() const noexcept { return this->m_arena.pages * Page::size(); }
		address_t memory_arena_read_boundary() const noexcept { return this->m_arena.read_boundary; }
		address_t memory_arena_write_boundary() const noexcept { return this->m_arena.write_boundary; }
		address_t initial_rodata_end() const noexcept { return this->m_arena.initial_rodata_end; }

		// Serializes the current memory state to an existing vector
		// Returns the final size of the serialized state
		size_t serialize_to(std::vector<uint8_t>& vec) const;
		// Returns memory to a previously stored state
		void deserialize_from(const std::vector<uint8_t>&, const SerializedMachine<W>&);

		Memory(Machine<W>&, std::string_view, MachineOptions<W>);
		Memory(Machine<W>&, const Machine<W>&, MachineOptions<W>);
		~Memory();
	private:
		void clear_all_pages();
		void initial_paging();
		[[noreturn]] static void protection_fault(address_t);
		// Helpers
		template <typename T>
		static void foreach_helper(T& mem, address_t addr, size_t len,
			std::function<void(T&, address_t, const uint8_t*, size_t)> callback);
		template <typename T>
		static void memview_helper(T& mem, address_t addr, size_t len,
			std::function<void(T&, const uint8_t*, size_t)> callback);
		// ELF stuff
		using Elf = typename riscv::Elf<W>;
		template <typename T> T* elf_offset(size_t ofs) const {
			if (ofs + sizeof(T) >= ofs && ofs + sizeof(T) < m_binary.size())
				return (T*) &m_binary[ofs];
#if __cpp_exceptions
			throw MachineException(INVALID_PROGRAM, "Invalid ELF offset", ofs);
#else
			std::abort();
#endif

		}
		const auto* elf_header() const {
			return elf_offset<const typename Elf::Header> (0);
		}
		const typename Elf::SectionHeader* section_by_name(const std::string& name) const;
		void dynamic_linking(const typename Elf::Header&);
		void relocate_section(const char* section_name, const char* symtab);
		const typename Elf::Sym* resolve_symbol(std::string_view name) const;
		const typename Elf::Sym* elf_sym_index(const typename Elf::SectionHeader* shdr, uint32_t symidx) const;
		// ELF loader
		void binary_loader(const MachineOptions<W>&);
		void binary_load_ph(const MachineOptions<W>&, const typename Elf::ProgramHeader*, address_t vaddr);
		void serialize_execute_segment(const MachineOptions<W>&, const typename Elf::ProgramHeader*, address_t vaddr);
		void generate_decoder_cache(const MachineOptions<W>&, std::shared_ptr<DecodedExecuteSegment<W>>&, bool is_initial);
		// Machine copy-on-write fork
		void machine_loader(const Machine<W>&, const MachineOptions<W>&);

		address_t m_start_address = 0;
		address_t m_stack_address = 0;
		address_t m_exit_address  = 0;
		address_t m_mmap_address  = 0;
		address_t m_heap_address  = 0;

		Machine<W>& m_machine;

		mutable CachedPage<W, const PageData> m_rd_cache;
		mutable CachedPage<W, PageData> m_wr_cache;

		std::unordered_map<address_t, Page> m_pages;

		const bool m_original_machine;
		bool m_is_dynamic = false;

		const std::string_view m_binary;

		// Memory map cache
		MMapCache<W> m_mmap_cache;

		page_fault_cb_t m_page_fault_handler = nullptr;
		page_write_cb_t m_page_write_handler = default_page_write;
		page_readf_cb_t m_page_readf_handler = default_page_read;

#ifdef RISCV_EXT_ATOMICS
		AtomicMemory<W> m_atomics;
#endif

		// Execute segments
		std::vector<std::shared_ptr<DecodedExecuteSegment<W>>> m_exec;
		std::shared_ptr<DecodedExecuteSegment<W>>& next_execute_segment();

		// Linear arena at start of memory (mmap-backed)
		struct {
			PageData* data = nullptr;
			address_t read_boundary = 0;
			address_t write_boundary = 0;
			address_t initial_rodata_end = 0;
			size_t    pages = 0;
		} m_arena;

		friend struct CPU<W>;
	};
#include "memory_inline.hpp"
#include "memory_inline_pages.hpp"
#include "memory_helpers_paging.hpp"
}
