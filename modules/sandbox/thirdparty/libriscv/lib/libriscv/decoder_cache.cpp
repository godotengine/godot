#include "machine.hpp"
#include "decoder_cache.hpp"
#include "instruction_list.hpp"
#include "internal_common.hpp"
#include "rvc.hpp"
#include "safe_instr_loader.hpp"
#include "threaded_rewriter.cpp"
#include "threaded_bytecodes.hpp"
#include "util/crc32.hpp"
#include <inttypes.h>
#include <mutex>
#include <string>
#include <unordered_set>
//#define ENABLE_TIMINGS
struct SegmentKey {
	uint64_t pc;
	uint32_t crc;
	uint64_t arena_size = 0;

	template <int W>
	static SegmentKey from(const riscv::DecodedExecuteSegment<W>& segment, uint64_t arena_size) {
		SegmentKey key;
		key.pc = uint64_t(segment.exec_begin());
		key.crc = segment.crc32c_hash();
		key.arena_size = arena_size;
		return key;
	}

	bool operator==(const SegmentKey& other) const {
		return pc == other.pc && crc == other.crc;
	}
	bool operator<(const SegmentKey& other) const {
		return pc < other.pc || (pc == other.pc && crc < other.crc);
	}
};
namespace std {
	template <>
	struct hash<SegmentKey> {
		size_t operator()(const SegmentKey& key) const {
			return key.pc ^ key.crc ^ key.arena_size;
		}
	};
}

namespace riscv
{
	static constexpr bool VERBOSE_DECODER = false;
	static std::mutex handler_idx_mutex;
#ifdef ENABLE_TIMINGS
	static inline timespec time_now();
	static inline long nanodiff(timespec, timespec);
	#define TIME_POINT(x) \
		[[maybe_unused]] timespec x;  \
		if (true) {                   \
			asm("" : : : "memory");   \
			x = time_now();           \
			asm("" : : : "memory");   \
		}
#else
	#define TIME_POINT(x) /* x */
#endif

	template <int W>
	struct SharedExecuteSegments {
		SharedExecuteSegments() = default;
		SharedExecuteSegments(const SharedExecuteSegments&) = delete;
		SharedExecuteSegments& operator=(const SharedExecuteSegments&) = delete;
		using key_t = SegmentKey;

		struct Segment {
			std::shared_ptr<DecodedExecuteSegment<W>> segment;
			std::mutex mutex;

			std::shared_ptr<DecodedExecuteSegment<W>> get() {
				std::lock_guard<std::mutex> lock(mutex);
				return segment;
			}

			void unlocked_set(std::shared_ptr<DecodedExecuteSegment<W>> new_segment) {
				this->segment = std::move(new_segment);
			}
		};

		// Remove a segment if it is the last reference
		void remove_if_unique(key_t key) {
			std::lock_guard<std::mutex> lock(mutex);
			// We are not able to remove the Segment itself, as the mutex
			// may be locked by another thread. We can, however, lock the
			// Segments mutex and set the segment to nullptr.
			auto it = m_segments.find(key);
			if (it != m_segments.end()) {
				std::scoped_lock segment_lock(it->second.mutex);
				if (it->second.segment.use_count() == 1)
					it->second.segment = nullptr;
			}
		}

		auto& get_segment(key_t key) {
			std::scoped_lock lock(mutex);
			auto& entry = m_segments[key];
			return entry;
		}

	private:
		std::unordered_map<key_t, Segment> m_segments;
		std::mutex mutex;
	};
	template <int W>
	static SharedExecuteSegments<W> shared_execute_segments;

	template <int W>
	static bool is_regular_compressed(uint16_t instr) {
		const rv32c_instruction ci { instr };
		#define CI_CODE(x, y) ((x << 13) | (y))
		switch (ci.opcode()) {
		case CI_CODE(0b001, 0b01):
			if constexpr (W >= 8) return true; // C.ADDIW
			return false; // C.JAL 32-bit
		case CI_CODE(0b101, 0b01): // C.JMP
		case CI_CODE(0b110, 0b01): // C.BEQZ
		case CI_CODE(0b111, 0b01): // C.BNEZ
			return false;
		case CI_CODE(0b100, 0b10): { // VARIOUS
				const bool topbit = ci.whole & (1 << 12);
				if (!topbit && ci.CR.rd != 0 && ci.CR.rs2 == 0) {
					return false; // C.JR rd
				} else if (topbit && ci.CR.rd != 0 && ci.CR.rs2 == 0) {
					return false; // C.JALR ra, rd+0
				} // TODO: Handle C.EBREAK
				return true;
			}
		default:
			return true;
		}
	}

	template <int W>
	struct DecoderEntryAndCount {
		DecoderData<W>* entry;
		int count;
	};

	template <int W>
	static inline void fill_entries(
		const std::array<DecoderEntryAndCount<W>, 256>& block_array,
		size_t block_array_count, address_type<W> block_pc, address_type<W> current_pc)
	{
		const unsigned last_count = block_array[block_array_count - 1].count;
		unsigned count = (current_pc - block_pc) >> 1;
		count -= last_count;
		if (count > 255)
			throw MachineException(INVALID_PROGRAM, "Too many non-branching instructions in a row");

		for (size_t i = 0; i < block_array_count; i++) {
			const DecoderEntryAndCount<W>& tuple = block_array[i];
			DecoderData<W>* entry = tuple.entry;
			const int length = tuple.count;

			// Ends at instruction *before* last PC
			entry->idxend = count;
			entry->icount = block_array_count - i;

			if constexpr (VERBOSE_DECODER) {
				fprintf(stderr, "Block 0x%lX has %u instructions\n", block_pc, count);
			}

			count -= length;
		}
	}

	template <int W>
	static void realize_fastsim(
		address_type<W> base_pc, address_type<W> last_pc,
		const uint8_t* exec_segment, DecoderData<W>* exec_decoder)
	{
#ifdef RISCV_BINARY_TRANSLATION
		const auto translator_op = RV32I_BC_TRANSLATOR;
#endif

		if constexpr (compressed_enabled)
		{
			if (UNLIKELY(base_pc >= last_pc))
				throw MachineException(INVALID_PROGRAM, "The execute segment has an overflow");
			if (UNLIKELY(base_pc & 0x1))
				throw MachineException(INVALID_PROGRAM, "The execute segment is misaligned");

			// Go through entire executable segment and measure lengths
			// Record entries while looking for jumping instruction, then
			// fill out data and opcode lengths previous instructions.
			std::array<DecoderEntryAndCount<W>, 256> block_array;
			address_type<W> pc = base_pc;
			while (pc < last_pc) {
				size_t block_array_count = 0;
				const address_type<W> block_pc = pc;
				DecoderData<W>* entry = &exec_decoder[pc / DecoderCache<W>::DIVISOR];
				const AlignedLoad16* iptr  = (AlignedLoad16*)&exec_segment[pc];
				const AlignedLoad16* iptr_begin = iptr;
				while (true) {
					const unsigned length = iptr->length();
					const int count = length >> 1;

					// Record the instruction
					block_array[block_array_count++] = { entry, count };

					// Make sure PC does not overflow
#ifdef _MSC_VER
					if (pc + length < pc)
						throw MachineException(INVALID_PROGRAM, "PC overflow during execute segment decoding");
#else
					[[maybe_unused]] address_type<W> pc2;
					if (UNLIKELY(__builtin_add_overflow(pc, length, &pc2)))
						throw MachineException(INVALID_PROGRAM, "PC overflow during execute segment decoding");
#endif
					pc += length;

					// If ending up crossing last_pc, it's an invalid block although
					// it could just be garbage, so let's force-end with an invalid instruction.
					if (UNLIKELY(pc > last_pc)) {
						entry->m_bytecode = 0; // Invalid instruction
						entry->m_handler = 0;
						break;
					}

					// All opcodes that can modify PC
					if (length == 2)
					{
						if (!is_regular_compressed<W>(iptr->half()))
							break;
					} else {
						const unsigned opcode = iptr->opcode();
						if (opcode == RV32I_BRANCH || opcode == RV32I_SYSTEM
							|| opcode == RV32I_JAL || opcode == RV32I_JALR)
							break;
					}
				#ifdef RISCV_BINARY_TRANSLATION
					if (entry->get_bytecode() == translator_op)
						break;
				#endif

					// A last test for the last instruction, which should have been a block-ending
					// instruction. Since it wasn't we must force-end the block here.
					if (UNLIKELY(pc >= last_pc)) {
						entry->m_bytecode = 0; // Invalid instruction
						entry->m_handler = 0;
						break;
					}

					iptr += count;

					// Too large blocks are likely malicious (although could be many empty pages)
					if (UNLIKELY(iptr - iptr_begin >= 255)) {
						// NOTE: Reinsert original instruction, as long sequences will lead to
						// PC becoming desynched, as it doesn't get increased.
						// We use a new block-ending fallback function handler instead.
						rv32i_instruction instruction = read_instruction(exec_segment, pc - length, last_pc);
						entry->set_bytecode(RV32I_BC_FUNCBLOCK);
						entry->set_invalid_handler(); // Resolve lazily
						entry->instr = instruction.whole;
						break;
					}

					entry += count;
				}
				if constexpr (VERBOSE_DECODER) {
					fprintf(stderr, "Block 0x%lX to 0x%lX\n", block_pc, pc);
				}

				if (UNLIKELY(block_array_count == 0))
					throw MachineException(INVALID_PROGRAM, "Encountered empty block after measuring");

				fill_entries(block_array, block_array_count, block_pc, pc);
			}
		} else { // !compressed_enabled
			// Count distance to next branching instruction backwards
			// and fill in idxend for all entries along the way.
			// This is for uncompressed instructions, which are always
			// 32-bits in size. We can use the idxend value for
			// instruction counting.
			unsigned idxend = 0;
			address_type<W> pc = last_pc - 4;
			// NOTE: The last check avoids overflow
			while (pc >= base_pc && pc < last_pc)
			{
				const rv32i_instruction instruction = read_instruction(
					exec_segment, pc, last_pc);
				DecoderData<W>& entry = exec_decoder[pc / DecoderCache<W>::DIVISOR];
				const unsigned opcode = instruction.opcode();

				// All opcodes that can modify PC and stop the machine
				if (opcode == RV32I_BRANCH || opcode == RV32I_SYSTEM
					|| opcode == RV32I_JAL || opcode == RV32I_JALR)
					idxend = 0;
			#ifdef RISCV_BINARY_TRANSLATION
				if (entry.get_bytecode() == translator_op)
					idxend = 0;
			#endif
				if (UNLIKELY(idxend == 65535)) {
					// It's a long sequence of instructions, so end block here.
					entry.set_bytecode(RV32I_BC_FUNCBLOCK);
					entry.set_invalid_handler(); // Resolve lazily
					entry.instr = instruction.whole;
					idxend = 0;
				}

				// Ends at *one instruction before* the block ends
				entry.idxend = idxend;
				// Increment after, idx becomes block count - 1
				idxend ++;

				pc -= 4;
			}
		}
	}

	// The decoder cache is a sequential array of DecoderData<W> entries
	// each of which (currently) serves a dual purpose of enabling
	// threaded dispatch (m_bytecode) and fallback to callback function
	// (m_handler). This enables high-speed emulation, precise simulation,
	// CLI debugging and remote GDB debugging without rebuilding the emulator.
	//
	// The decoder cache covers all pages that the execute segment belongs
	// in, so that all legal jumps (based on page +exec permission) will
	// result in correct execution (including invalid instructions).
	//
	// The goal of the decoder cache is to allow uninterrupted execution
	// with minimal bounds-checking, while also enabling accurate
	// instruction counting.
	template <int W> RISCV_INTERNAL
	void Memory<W>::generate_decoder_cache(
		[[maybe_unused]] const MachineOptions<W>& options,
		std::shared_ptr<DecodedExecuteSegment<W>>& shared_segment, [[maybe_unused]] bool is_initial)
	{
		TIME_POINT(t0);
		auto& exec = *shared_segment;
		if (exec.exec_end() < exec.exec_begin())
			throw MachineException(INVALID_PROGRAM, "Execute segment was invalid");

		const auto pbase = exec.pagedata_base();
		const auto addr  = exec.exec_begin();
		const auto len   = exec.exec_end() - exec.exec_begin();
		constexpr size_t PMASK = Page::size()-1;
		// We need to allocate room for at least one more decoder cache entry.
		// This is because jump and branch instructions don't check PC after
		// not branching. The last entry is an invalid instruction.
		const size_t prelen  = addr - pbase;
		const size_t midlen  = len + prelen + 4; // Extra entry
		const size_t plen = (midlen + PMASK) & ~PMASK;
		//printf("generate_decoder_cache: Addr 0x%X Len %zx becomes 0x%X->0x%X PRE %zx MIDDLE %zu TOTAL %zu\n",
		//	addr, len, pbase, pbase + plen, prelen, midlen, plen);

		const size_t n_pages = plen / Page::size();
		if (n_pages == 0) {
			throw MachineException(INVALID_PROGRAM,
				"Program produced empty decoder cache");
		}
		// Here we allocate the decoder cache which is page-sized
		auto* decoder_cache = exec.create_decoder_cache(
			new DecoderCache<W> [n_pages], n_pages);
		// Clear the decoder cache! (technically only needed when binary translation is enabled)
		std::memset(decoder_cache, 0, n_pages * sizeof(DecoderCache<W>));
		// Get a base address relative pointer to the decoder cache
		// Eg. exec_decoder[pbase] is the first entry in the decoder cache
		// so that PC with a simple shift can be used as a direct index.
		auto* exec_decoder = 
			decoder_cache[0].get_base() - pbase / DecoderCache<W>::DIVISOR;
		exec.set_decoder(exec_decoder);

		DecoderData<W> invalid_op;
		invalid_op.set_handler(this->machine().cpu.decode({0}));
		if (UNLIKELY(invalid_op.m_handler != 0)) {
			throw MachineException(INVALID_PROGRAM,
				"The invalid instruction did not have the index zero", invalid_op.m_handler);
		}

		// PC-relative pointer to instruction bits
		auto* exec_segment = exec.exec_data();
		TIME_POINT(t1);

#ifdef RISCV_BINARY_TRANSLATION
		// We do not support binary translation for RV128I
		// Also, avoid binary translation for execute segments that are likely JIT-compiled
		const bool allow_translation = is_initial || options.translate_future_segments;
		if (allow_translation && !exec.is_likely_jit()) {
			// Attempt to load binary translation
			// Also, fill out the binary translation SO filename for later
			std::string bintr_filename;
			int result = machine().cpu.load_translation(options, &bintr_filename, exec);
			const bool must_translate = result > 0;
			if (must_translate)
			{
				machine().cpu.try_translate(
					options, bintr_filename, shared_segment);
			}
		}
	#endif

		// When compressed instructions are enabled, many decoder
		// entries are illegal because they are between instructions.
		bool was_full_instruction = true;

		/* Generate all instruction pointers for executable code.
		   Cannot step outside of this area when pregen is enabled,
		   so it's fine to leave the boundries alone. */
		TIME_POINT(t2);
		address_type<W> dst = addr;
		const address_type<W> end_addr = addr + len;
		for (; dst < addr + len;)
		{
			auto& entry = exec_decoder[dst / DecoderCache<W>::DIVISOR];
			entry.m_handler = 0;
			entry.idxend = 0;

			// Load unaligned instruction from execute segment
			const auto instruction = read_instruction(
				exec_segment, dst, end_addr);
			rv32i_instruction rewritten = instruction;

#ifdef RISCV_BINARY_TRANSLATION
			// Translator activation uses a special bytecode
			// but we must still validate the mapping index.
			if (entry.get_bytecode() == RV32I_BC_TRANSLATOR && entry.is_invalid_handler() && entry.instr < exec.translator_mappings()) {
				if constexpr (compressed_enabled) {
					dst += 2;
					if (was_full_instruction) {
						was_full_instruction = (instruction.length() == 2);
					} else {
						was_full_instruction = true;
					}
				} else
					dst += 4;
				continue;
			}
#endif // RISCV_BINARY_TRANSLATION

			if (!compressed_enabled || was_full_instruction) {
				// Cache the (modified) instruction bits
				auto bytecode = CPU<W>::computed_index_for(instruction);
				// Threaded rewrites are **always** enabled
				bytecode = exec.threaded_rewrite(bytecode, dst, rewritten);
				entry.set_bytecode(bytecode);
				entry.instr = rewritten.whole;
			} else {
				// WARNING: If we don't ignore this instruction,
				// it will get *wrong* idxend values, and cause *invalid jumps*
				entry.m_handler = 0;
				entry.set_bytecode(0);
				// ^ Must be made invalid, even if technically possible to jump to!
			}
			if constexpr (VERBOSE_DECODER) {
				if (entry.get_bytecode() >= RV32I_BC_BEQ && entry.get_bytecode() <= RV32I_BC_BGEU) {
					fprintf(stderr, "Detected branch bytecode at 0x%lX\n", dst);
				}
				if (entry.get_bytecode() == RV32I_BC_BEQ_FW || entry.get_bytecode() == RV32I_BC_BNE_FW) {
					fprintf(stderr, "Detected forward branch bytecode at 0x%lX\n", dst);
				}
			}

			// Increment PC after everything
			if constexpr (compressed_enabled) {
				// With compressed we always step forward 2 bytes at a time
				dst += 2;
				if (was_full_instruction) {
					// For it to be a full instruction again,
					// the length needs to match.
					was_full_instruction = (instruction.length() == 2);
				} else {
					// If it wasn't a full instruction last time, it
					// will for sure be one now.
					was_full_instruction = true;
				}
			} else
				dst += 4;
		}
		// Make sure the last entry is an invalid instruction
		// This simplifies many other sub-systems
		auto& entry = exec_decoder[(addr + len) / DecoderCache<W>::DIVISOR];
		entry.set_bytecode(0);
		entry.m_handler = 0;
		entry.idxend = 0;
		TIME_POINT(t3);

		realize_fastsim<W>(addr, dst, exec_segment, exec_decoder);

		// Debugging: EBREAK locations
		for (auto& loc : options.ebreak_locations) {
			address_type<W> ebreak_addr = 0;
			if (std::holds_alternative<address_type<W>>(loc))
				ebreak_addr = std::get<address_type<W>>(loc);
			else
				ebreak_addr = machine().address_of(std::get<std::string>(loc));

			if (ebreak_addr != 0x0 && ebreak_addr >= exec.exec_begin() && ebreak_addr < exec.exec_end()) {
				CPU<W>::install_ebreak_for(exec, ebreak_addr);
				if (options.verbose_loader) {
					printf("libriscv: Added ebreak location at 0x%" PRIx64 "\n", uint64_t(ebreak_addr));
				}
			}
		}

		TIME_POINT(t4);
#ifdef ENABLE_TIMINGS
		const long t1t0 = nanodiff(t0, t1);
		const long t2t1 = nanodiff(t1, t2);
		const long t3t2 = nanodiff(t2, t3);
		const long t3t4 = nanodiff(t3, t4);
		printf("libriscv: Decoder cache allocation took %ld ns\n", t1t0);
		if constexpr (binary_translation_enabled)
			printf("libriscv: Decoder cache bintr activation took %ld ns\n", t2t1);
		printf("libriscv: Decoder cache generation took %ld ns\n", t3t2);
		printf("libriscv: Decoder cache realization took %ld ns\n", t3t4);
		printf("libriscv: Decoder cache totals: %ld us\n", nanodiff(t0, t4) / 1000);
#endif
	}

	template <int W> RISCV_INTERNAL
	size_t DecoderData<W>::handler_index_for(Handler new_handler)
	{
		std::scoped_lock lock(handler_idx_mutex);

		auto it = handler_cache.find(new_handler);
		if (it != handler_cache.end())
			return it->second;

		if (UNLIKELY(handler_count >= instr_handlers.size()))
			throw MachineException(INVALID_PROGRAM, "Too many instruction handlers");
		instr_handlers[handler_count] = new_handler;
		const size_t idx = handler_count++;
		handler_cache.emplace(new_handler, idx);
		return idx;
	}

	// An execute segment contains a sequential array of raw instruction bits
	// belonging to a set of sequential pages with +exec permission.
	// It also contains a decoder cache that is produced from this instruction data.
	// It is not strictly necessary to store the raw instruction bits, however, it
	// enables step by step simulation as well as CLI- and remote debugging without
	// rebuilding the emulator.
	// Crucially, because of page alignments and 4 extra bytes, the necessary checks
	// when reading from the execute segment is reduced. You can always read 4 bytes
	// no matter where you are in the segment, a whole instruction unchecked.
	template <int W> RISCV_INTERNAL
	DecodedExecuteSegment<W>& Memory<W>::create_execute_segment(
		const MachineOptions<W>& options, const void *vdata, address_type<W> vaddr, size_t exlen, bool is_initial, bool is_likely_jit)
	{
		if (UNLIKELY(exlen % (compressed_enabled ? 2 : 4)))
			throw MachineException(INVALID_PROGRAM, "Misaligned execute segment length");

		constexpr address_type<W> PMASK = Page::size()-1;
		const address_type<W> pbase = vaddr & ~PMASK;
		const size_t prelen  = vaddr - pbase;
		// Make 4 bytes of extra room to avoid having to validate 4-byte reads
		// when reading at 2 bytes before the end of the execute segment.
		const size_t midlen  = exlen + prelen + 2; // Extra room for reads
		const size_t plen = (midlen + PMASK) & ~PMASK;
		// Because postlen uses midlen, we end up zeroing the extra 4 bytes in the end
		const size_t postlen = plen - midlen;
		//printf("Addr 0x%X Len %zx becomes 0x%X->0x%X PRE %zx MIDDLE %zu POST %zu TOTAL %zu\n",
		//	vaddr, exlen, pbase, pbase + plen, prelen, exlen, postlen, plen);
		if (UNLIKELY(prelen > plen || prelen + exlen > plen)) {
			throw MachineException(INVALID_PROGRAM, "Segment virtual base was bogus");
		}
#ifdef _MSC_VER
		if (UNLIKELY(pbase + plen < pbase))
			throw MachineException(INVALID_PROGRAM, "Segment virtual base was bogus");
#else
		[[maybe_unused]] address_type<W> pbase2;
		if (UNLIKELY(__builtin_add_overflow(pbase, plen, &pbase2)))
			throw MachineException(INVALID_PROGRAM, "Segment virtual base was bogus");
#endif
		// Create the whole executable memory range
		auto current_exec = std::make_shared<DecodedExecuteSegment<W>>(pbase, plen, vaddr, exlen);

		auto* exec_data = current_exec->exec_data(pbase);
		// This is a zeroed prologue in order to be able to use whole pages
		std::memset(&exec_data[0],      0,     prelen);
		// This is the actual instruction bytes
		std::memcpy(&exec_data[prelen], vdata, exlen);
		// This memset() operation will end up zeroing the extra 4 bytes
		std::memset(&exec_data[prelen + exlen], 0,   postlen);

		// Create CRC32-C hash of the execute segment
		const uint32_t hash = crc32c(exec_data, current_exec->exec_end() - current_exec->exec_begin());

		// Get a free slot to reference the execute segment
		auto& free_slot = this->next_execute_segment();


		if (options.use_shared_execute_segments)
		{
			// We have to key on the base address of the execute segment as well as the hash
			const SegmentKey key{uint64_t(current_exec->exec_begin()), hash, memory_arena_size()};

			// In order to prevent others from creating the same execute segment
			// we need to lock the shared execute segments mutex.
			auto& segment = shared_execute_segments<W>.get_segment(key);
			std::scoped_lock lock(segment.mutex);

			if (segment.segment != nullptr) {
				free_slot = segment.segment;
				return *free_slot;
			}

			// We need to create a new execute segment, as there is no shared
			// execute segment with the same hash.
			free_slot = std::move(current_exec);
			free_slot->set_likely_jit(is_likely_jit);
#if defined(RISCV_BINARY_TRANSLATION) && defined(RISCV_DEBUG)
			free_slot->set_record_slowpaths(options.record_slowpaths_to_jump_hints && !is_likely_jit);
#endif
			// Store the hash in the decoder cache
			free_slot->set_crc32c_hash(hash);

			this->generate_decoder_cache(options, free_slot, is_initial);

			// Share the execute segment
			shared_execute_segments<W>.get_segment(key).unlocked_set(free_slot);
		}
		else
		{
			free_slot = std::move(current_exec);
			free_slot->set_likely_jit(is_likely_jit);
#if defined(RISCV_BINARY_TRANSLATION) && defined(RISCV_DEBUG)
			free_slot->set_record_slowpaths(options.record_slowpaths_to_jump_hints && !is_likely_jit);
#endif
			// Store the hash in the decoder cache
			free_slot->set_crc32c_hash(hash);

			this->generate_decoder_cache(options, free_slot, is_initial);
		}

		return *free_slot;
	}

	template <int W>
	std::shared_ptr<DecodedExecuteSegment<W>>& Memory<W>::next_execute_segment()
	{
		if (LIKELY(m_exec.size() < RISCV_MAX_EXECUTE_SEGS)) {
			m_exec.push_back(nullptr);
			return m_exec.back();
		}
		throw MachineException(INVALID_PROGRAM, "Max execute segments reached");
	}

	template <int W>
	const std::shared_ptr<DecodedExecuteSegment<W>>& Memory<W>::exec_segment_for(address_type<W> vaddr) const
	{
		return const_cast<Memory<W>*>(this)->exec_segment_for(vaddr);
	}

	template <int W>
	void Memory<W>::evict_execute_segments()
	{
		// destructor could throw, so let's invalidate early
		machine().cpu.set_execute_segment(*CPU<W>::empty_execute_segment());

		while (!m_exec.empty()) {
			try {
				auto& segment = m_exec.back();
				if (segment) {
					const SegmentKey key = SegmentKey::from(*segment, memory_arena_size());
					segment = nullptr;
					shared_execute_segments<W>.remove_if_unique(key);
				}
				m_exec.pop_back();
			} catch (...) {
				// Ignore exceptions
			}
		}
	}

	template <int W>
	void Memory<W>::evict_execute_segment(DecodedExecuteSegment<W>& segment)
	{
		const SegmentKey key = SegmentKey::from(segment, memory_arena_size());
		for (auto& seg : m_exec) {
			if (seg.get() == &segment) {
				seg = nullptr;
				if (&seg == &m_exec.back())
					m_exec.pop_back();
				break;
			}
		}
		shared_execute_segments<W>.remove_if_unique(key);
	}

#ifdef RISCV_BINARY_TRANSLATION
	template <int W>
	std::vector<address_type<W>> Memory<W>::gather_jump_hints() const
	{
		std::vector<address_type<W>> result;
#  ifdef RISCV_DEBUG
		std::unordered_set<address_type<W>> addresses;
		for (auto addr : machine().options().translator_jump_hints)
			addresses.insert(addr);
		for (auto& segment : m_exec) {
			if (segment) {
				if (segment->is_recording_slowpaths()) {
					for (auto addr : segment->slowpath_addresses())
						addresses.insert(addr);
				}
			}
		}
		for (auto addr : addresses)
			result.push_back(addr);
#  endif // RISCV_DEBUG
		return result;
	}
#endif

#ifdef ENABLE_TIMINGS
	timespec time_now()
	{
		timespec t;
		clock_gettime(CLOCK_MONOTONIC, &t);
		return t;
	}
	long nanodiff(timespec start_time, timespec end_time)
	{
		return (end_time.tv_sec - start_time.tv_sec) * (long)1e9 + (end_time.tv_nsec - start_time.tv_nsec);
	}
#endif

	INSTANTIATE_32_IF_ENABLED(DecoderData);
	INSTANTIATE_32_IF_ENABLED(Memory);
	INSTANTIATE_64_IF_ENABLED(DecoderData);
	INSTANTIATE_64_IF_ENABLED(Memory);
	INSTANTIATE_128_IF_ENABLED(DecoderData);
	INSTANTIATE_128_IF_ENABLED(Memory);
} // riscv
