#include "machine.hpp"
#include "internal_common.hpp"

namespace riscv
{
	template <int W>
	address_type<W> Memory<W>::mmap_allocate(address_t bytes)
	{
		// Bytes rounded up to nearest PageSize.
		const address_t result = this->m_mmap_address;
		this->m_mmap_address += (bytes + PageMask) & ~address_t{PageMask};
		return result;
	}

	template <int W>
	bool Memory<W>::mmap_relax(address_t addr, address_t size, address_t new_size)
	{
		// Undo or relax the last mmap allocation. Returns true if successful.
		if (this->m_mmap_address == addr + size && new_size <= size) {
			this->m_mmap_address = (addr + new_size + PageMask) & ~address_t{PageMask};
			return true;
		}
		return false;
	}

	template <int W>
	bool Memory<W>::mmap_unmap(address_t addr, address_t size)
	{
		size = (size + PageMask) & ~address_t{PageMask};
		const bool relaxed = this->mmap_relax(addr, size, 0u);
		if (relaxed)
		{
			// If relaxation happened, invalidate intersecting cache entries.
			this->mmap_cache().invalidate(addr, size);
		}
		else if (addr >= this->mmap_start())
		{
			// If relaxation didn't happen, put in the cache for later.
			this->mmap_cache().insert(addr, size);
		}
		return relaxed;
	}

	INSTANTIATE_32_IF_ENABLED(Memory);
	INSTANTIATE_64_IF_ENABLED(Memory);
	INSTANTIATE_128_IF_ENABLED(Memory);
} // riscv
