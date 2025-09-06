#include <libriscv/machine.hpp>

#include "internal_common.hpp"
#ifdef __GNUG__
#define RISCV_PACKED __attribute__((packed))
#else
#define RISCV_PACKED /**/
#endif

namespace riscv
{
	static const uint64_t MAGiC_V4LUE = 0x9c36ab9301aed873;
	template <int W>
	struct SerializedMachine
	{
		using address_t = address_type<W>;

		uint64_t magic;
		uint32_t n_pages;
		uint32_t n_datapages;
		uint16_t reg_size;
		uint16_t page_size;
		uint16_t attr_size;
		uint16_t serp_size;
		uint16_t reserved;
		uint16_t cpu_offset;
		uint32_t mem_offset;

		Registers<W> registers;
		uint64_t     counter;

		address_t start_address = 0;
		address_t stack_address = 0;
		address_t mmap_address  = 0;
		address_t heap_address  = 0;
		address_t exit_address  = 0;
	};
	struct SerializedPage
	{
		uint64_t addr;
		PageAttributes attr;
		bool is_cow_page = false;
		uint8_t padding[3] {0};
	};

	template <int W>
	size_t Machine<W>::serialize_to(std::vector<uint8_t>& vec) const
	{
		const size_t before = vec.size();

		unsigned datapage_count = 0;
		for (const auto& it : memory.pages()) {
			if (!it.second.is_cow_page()) datapage_count++;
		}

		const SerializedMachine<W> header {
			.magic    = MAGiC_V4LUE,
			.n_pages  = (unsigned) memory.pages().size(),
			.n_datapages = datapage_count,
			.reg_size = sizeof(Registers<W>),
			.page_size = Page::size(),
			.attr_size = sizeof(PageAttributes),
			.serp_size = sizeof(SerializedPage),
			.reserved = 0,
			.cpu_offset = sizeof(SerializedMachine<W>),
			.mem_offset = sizeof(SerializedMachine<W>) + 0x0,

			.registers = cpu.registers(),
			.counter   = this->instruction_counter(),

			.start_address = memory.start_address(),
			.stack_address = memory.stack_initial(),
			.mmap_address  = memory.mmap_address(),
			.heap_address  = memory.heap_address(),
			.exit_address  = memory.exit_address(),
		};
		const auto* hptr = (const uint8_t*) &header;
		vec.insert(vec.end(), hptr, hptr + sizeof(header));
		this->cpu.serialize_to(vec);
		this->memory.serialize_to(vec);

		const size_t after = vec.size();
		return after - before;
	}
	template <int W>
	void CPU<W>::serialize_to(std::vector<uint8_t>& /* vec */) const
	{
	}
	template <int W>
	size_t Memory<W>::serialize_to(std::vector<uint8_t>& vec) const
	{
		const size_t before = vec.size();
		if (this->m_arena.pages > 0 && riscv::flat_readwrite_arena) {
			throw MachineException(
				FEATURE_DISABLED, "Serialize is incompatible with flat read-write arena");
		}

		const size_t est_page_bytes =
			this->m_pages.size() * (sizeof(SerializedPage) + sizeof(PageData));
		vec.reserve(vec.size() + est_page_bytes);

		for (const auto& it : this->m_pages)
		{
			const auto& page = it.second;

			// XXX: 128-bit addresses not taken into account
			SerializedPage spage {
				.addr = static_cast<uint64_t>(it.first),
				.attr = page.attr,
				.is_cow_page = page.is_cow_page(),
			};
			// Make all pages owned from now on
			spage.attr.is_cow = false;
			spage.attr.non_owning = false;

			// Serialize page attributes
			auto* sptr = (const uint8_t*) &spage;
			vec.insert(vec.end(), sptr, sptr + sizeof(SerializedPage));

			// The zero-page (and other guard pages) may not have data
			if (page.is_cow_page())
				continue;

			// Serialize page data
			vec.insert(vec.end(), page.data(), page.data() + sizeof(PageData));
		}

		const size_t after = vec.size();
		return after - before;
	}

	template <int W>
	int Machine<W>::deserialize_from(const std::vector<uint8_t>& vec)
	{
		if (vec.size() < sizeof(SerializedMachine<W>)) {
			return -1;
		}
		const auto& header = *(const SerializedMachine<W>*) vec.data();
		if (header.magic != MAGiC_V4LUE)
			return -1;
		if (header.reg_size != sizeof(Registers<W>))
			return -2;
		if (header.page_size != Page::size())
			return -3;
		if (header.attr_size != sizeof(PageAttributes))
			return -4;
		if (header.serp_size != sizeof(SerializedPage))
			return -5;
		this->m_counter = header.counter;
		this->m_max_counter = 0;
		cpu.deserialize_from(vec, header);
		memory.deserialize_from(vec, header);
		return 0;
	}
	template <int W>
	void CPU<W>::deserialize_from(const std::vector<uint8_t>& /* vec */,
					const SerializedMachine<W>& state)
	{
		// restore CPU registers and counters
		this->m_regs = state.registers;
		this->m_exec = CPU::empty_execute_segment().get();
	}
	template <int W>
	void Memory<W>::deserialize_from(const std::vector<uint8_t>& vec,
					const SerializedMachine<W>& state)
	{
		this->m_start_address = state.start_address;
		this->m_stack_address = state.stack_address;
		this->m_mmap_address  = state.mmap_address;
		this->m_heap_address  = state.heap_address;
		this->m_exit_address  = state.exit_address;

#ifdef RISCV_EXT_ATOMICS
		this->m_atomics = {};
#endif

		const size_t page_bytes =
			  state.n_pages * sizeof(SerializedPage)
			+ state.n_datapages * sizeof(PageData);
		if (vec.size() < state.mem_offset + page_bytes) {
			throw MachineException(INVALID_PROGRAM, "Serialized machine state was invalid");
		}

		// completely reset the paging system as
		// all pages will be completely replaced
		this->clear_all_pages();
		this->evict_execute_segments();

		size_t off = state.mem_offset;
		for (size_t p = 0; p < state.n_pages; p++)
		{
			const SerializedPage page = *(SerializedPage*) &vec[off];
			off += sizeof(SerializedPage);

			PageAttributes new_attr = page.attr;
			// Pages with data
			if (!page.is_cow_page) {
				Page* new_page = nullptr;
				if (page.addr < this->m_arena.pages)
				{
					// Create new non-owning arena page
					new_attr.non_owning = true;
					auto result = m_pages.try_emplace(
						page.addr,
						new_attr, &this->m_arena.data[page.addr]
					);
					new_page = &result.first->second;
				}
				else
				{
					// Create new uninitialized page
					auto result = m_pages.try_emplace(
						page.addr,
						new_attr, PageData::UNINITIALIZED
					);
					new_page = &result.first->second;
				}
				// Copy unaligned data into new PageData
				const auto* data = &vec[off];
				std::copy(data, data + sizeof(PageData), new_page->data());
				off += sizeof(PageData);
			} else {
				// Pages without data
				m_pages.try_emplace(
					page.addr,
					new_attr, Page::cow_page().m_page.get()
				);
			}
		}
		// page tables have been changed
		this->invalidate_reset_cache();
	}

	INSTANTIATE_32_IF_ENABLED(Machine);
	INSTANTIATE_64_IF_ENABLED(Machine);
	INSTANTIATE_128_IF_ENABLED(Machine);

	INSTANTIATE_32_IF_ENABLED(CPU);
	INSTANTIATE_64_IF_ENABLED(CPU);
	INSTANTIATE_128_IF_ENABLED(CPU);

	INSTANTIATE_32_IF_ENABLED(Memory);
	INSTANTIATE_64_IF_ENABLED(Memory);
	INSTANTIATE_128_IF_ENABLED(Memory);
}
