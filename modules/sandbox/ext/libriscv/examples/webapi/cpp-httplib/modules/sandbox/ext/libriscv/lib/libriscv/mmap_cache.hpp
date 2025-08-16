#include <array>
#include <cstdint>
#include <vector>
#include "types.hpp"

namespace riscv
{
	template <int W>
	struct MMapCache
	{
		using address_t = address_type<W>;

		struct Range {
			address_t addr = 0x0;
			address_t size = 0u;

			constexpr bool empty() const noexcept { return size == 0u; }
			// Invalidate if one of the ranges is in the other (both ways!)
			constexpr bool within(address_t mem, address_t memsize) const noexcept {
				return ((this->addr >= mem) && (this->addr + this->size <= mem + memsize))
					|| ((mem >= this->addr) && (mem + memsize <= this->addr + this->size));
			}
			constexpr bool equals(address_t mem, address_t memsize) const noexcept {
				return (this->addr == mem) && (this->addr + this->size == mem + memsize);
			}
		};

		Range find(address_t size)
		{
			auto it = m_lines.begin();
			while (it != m_lines.end())
			{
				auto& r = *it;
				if (!r.empty())
				{
					if (r.size >= size) {
						const Range result { r.addr, size };
						if (r.size > size) {
							r.addr += size;
							r.size -= size;
						} else {
							m_lines.erase(it);
						}
						return result;
					}
				}
				++it;
			}
			return Range{};
		}

		void invalidate(address_t addr, address_t size)
		{
			auto it = m_lines.begin();
			while (it != m_lines.end())
			{
				const auto r = *it;
				if (r.within(addr, size))
				{
					bool equals = r.equals(addr, size);
					it = m_lines.erase(it);
					if (equals) return;
				}
				else ++it;
			}
		}

		void insert(address_t addr, address_t size)
		{
			/* Extend the back range? */
			if (!m_lines.empty()) {
				if (m_lines.back().addr + m_lines.back().size == addr) {
					m_lines.back().size += size;
					return;
				}
			}

			m_lines.push_back({addr, size});
		}

	private:
		std::vector<Range> m_lines {};
	};

} // riscv
