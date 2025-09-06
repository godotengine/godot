#pragma once

template <int W> inline
void Memory<W>::memset(address_t dst, uint8_t value, size_t len)
{
	while (len > 0)
	{
		const size_t offset = dst & (Page::size()-1); // offset within page
		const size_t size = std::min(Page::size() - offset, len);
		auto& page = this->create_writable_pageno(dst / Page::size(), size != Page::size());

		std::memset(page.data() + offset, value, size);

		dst += size;
		len -= size;
	}
}

template <int W> inline
void Memory<W>::memcpy(address_t dst, const void* vsrc, size_t len)
{
	auto* src = (uint8_t*) vsrc;
	while (len != 0)
	{
		const size_t offset = dst & (Page::size()-1); // offset within page
		const size_t size = std::min(Page::size() - offset, len);
		auto& page = this->create_writable_pageno(dst / Page::size(), size != Page::size());

		std::copy(src, src + size, page.data() + offset);

		dst += size;
		src += size;
		len -= size;
	}
}

template <int W> inline
void Memory<W>::memcpy_out(void* vdst, address_t src, size_t len) const
{
	auto* dst = (uint8_t*) vdst;
	while (len != 0)
	{
		const size_t offset = src & (Page::size()-1);
		const size_t size = std::min(Page::size() - offset, len);
		const auto& page = this->get_page(src);
		if (UNLIKELY(!page.attr.read))
			protection_fault(src);

		std::copy(page.data() + offset, page.data() + offset + size, dst);

		dst += size;
		src += size;
		len -= size;
	}
}

template <int W> inline
bool Memory<W>::try_memmove(address_t dst, address_t src, size_t len)
{
	if (dst == src || len == 0)
		return true;

	if constexpr (flat_readwrite_arena) {
		if (LIKELY(dst + len < memory_arena_size() && dst + len > dst &&
			src + len < memory_arena_size() && src + len > src)) {
			char* p_src = &((char *)m_arena.data)[src];
			char* p_dest = &((char *)m_arena.data)[dst];
			std::memmove(p_dest, p_src, len);
			return true;
		}
	}
	// Fast-path memmove requires flat-readwrite-arena
	return false;
}

template <int W> inline
std::string Memory<W>::memstring(address_t addr, const size_t max_len) const
{
	std::string result;
	address_t pageno = page_number(addr);
	// fast-path
	{
		const size_t offset = addr & (Page::size()-1);
		const Page& page = this->get_readable_pageno(pageno);

		const char* start = (const char*) &page.data()[offset];
		const char* pgend = (const char*) &page.data()[std::min(Page::size(), offset + max_len)];
		//
		const char* reader = start + strnlen(start, pgend - start);
		result.append(start, reader);
		// early exit
		if (LIKELY(reader < pgend)) {
			return result;
		}
	}
	// slow-path: cross page-boundary
	while (result.size() < max_len)
	{
		const size_t max_bytes = std::min(Page::size(), max_len - result.size());
		pageno ++;
		const Page& page = this->get_readable_pageno(pageno);

		const char* start = (const char*) page.data();
		const char* endptr = (const char*) &page.data()[max_bytes];

		const char* reader = start + strnlen(start, max_bytes);
		result.append(start, reader);
		// if we didn't stop at the page border, we must be done
		if (reader < endptr)
			return result;
	}
	return result;
}

template <int W> inline
std::string_view Memory<W>::memstring_view(address_t addr, size_t max_len) const
{
	if constexpr (flat_readwrite_arena) {
		if (LIKELY(addr < memory_arena_size())) {
			auto* begin = &((const char *)m_arena.data)[RISCV_SPECSAFE(addr)];
			// limit to the end of the arena
			max_len = std::min(max_len, size_t(memory_arena_size() - addr));
			// limit to the end of the string
			const size_t len = strnlen(begin, max_len);
			return {begin, len};
		}
	}

	protection_fault(addr);
}

template <int W> inline
riscv::Buffer Memory<W>::membuffer(address_t addr,
	const size_t datalen, const size_t maxlen) const
{
	riscv::Buffer result;

	if (UNLIKELY(datalen > maxlen))
		protection_fault(addr);

	if constexpr (flat_readwrite_arena) {
		if (LIKELY(addr + datalen < memory_arena_size() && addr + datalen > addr)) {
			auto* begin = &((char *)memory_arena_ptr())[RISCV_SPECSAFE(addr)];
			result.append_page(begin, datalen);
			return result;
		}
	}

	address_t pageno = page_number(addr);
	const Page& page = this->get_readable_pageno(pageno);

	const size_t offset = addr & (Page::size()-1);
	auto* start = (const char*) &page.data()[offset];
	const size_t page_bytes = std::min(Page::size() - offset, datalen);

	result.append_page(start, page_bytes);
	// slow-path: cross page-boundary
	while (result.size() < datalen)
	{
		const size_t max_bytes = std::min(Page::size(), datalen - result.size());
		pageno ++;
		const Page& slow_page = this->get_readable_pageno(pageno);

		result.append_page((const char*) slow_page.data(), max_bytes);
	}
	return result;
}

template <int W> inline
std::string_view Memory<W>::memview(address_t addr, size_t len, size_t maxlen) const
{
	if (UNLIKELY(len > maxlen))
		protection_fault(addr);

	if (len == 0)
		return {};

	if constexpr (flat_readwrite_arena) {
		if (LIKELY(addr + len - RWREAD_BEGIN < memory_arena_read_boundary() && addr < addr + len)) {
			auto* begin = &((const char *)m_arena.data)[RISCV_SPECSAFE(addr)];
			return {begin, len};
		}
	}

	// Fallback: Try gathering a single buffer, which throws OUT_OF_MEMORY if it fails
	std::array<vBuffer, 1> buffers{};
	gather_buffers_from_range(1, buffers.data(), addr, len);
	return {(const char *)buffers[0].ptr, buffers[0].len};
}

template <int W> inline
std::string_view Memory<W>::writable_memview(address_t addr, size_t len, size_t maxlen) const
{
	if (UNLIKELY(len > maxlen))
		protection_fault(addr);

	if (len == 0)
		return {};

	if constexpr (flat_readwrite_arena) {
		if (LIKELY(addr + len - initial_rodata_end() < memory_arena_write_boundary() && addr < addr + len)) {
			char* begin = &((char *)m_arena.data)[RISCV_SPECSAFE(addr)];
			return {begin, len};
		}
	}

	// Fallback: Try gathering a single buffer, which throws OUT_OF_MEMORY if it fails
	std::array<vBuffer, 1> buffers{};
	const_cast<Memory<W>*> (this)->gather_writable_buffers_from_range(1, buffers.data(), addr, len);
	return {(char *)buffers[0].ptr, buffers[0].len};
}

template <int W>
template <typename T, size_t N>
std::array<T, N>* Memory<W>::memarray(address_t addr) const
{
	// Special case: empty array - NOT DEREFERENCABLE
	if (N != 0 && addr % alignof(T) != 0)
		protection_fault(addr);

	std::string_view view;
	if constexpr (std::is_const_v<T>) {
		view = memview(addr, sizeof(T) * N);
	} else {
		view = writable_memview(addr, sizeof(T) * N);
	}
	if (view.size() != sizeof(T) * N)
		protection_fault(addr);

	return (std::array<T, N>*) view.data();
}

template <int W>
template <typename T>
T* Memory<W>::memarray(address_t addr, size_t count, size_t maxbytes) const
{
	// Special case: empty array - NOT DEREFERENCABLE
	if (count != 0 && addr % alignof(T) != 0)
		protection_fault(addr);

	std::string_view view;
	// When T* is const, we can use plain memview
	if constexpr (std::is_const_v<T>) {
		view = memview(addr, count * sizeof(T), maxbytes);
	} else { // Otherwise, we need to use writable_memview
		view = writable_memview(addr, count * sizeof(T), maxbytes);
	}
	if (view.size() != count * sizeof(T))
		protection_fault(addr);

	return (T*) view.data();
}

template <int W>
template <typename T>
T* Memory<W>::try_memarray(address_t addr, size_t count, size_t maxbytes) const
{
	if (count == 0)
		return nullptr;

	const size_t len = count * sizeof(T);
	if (UNLIKELY(len > maxbytes))
		protection_fault(addr);

	if (addr % alignof(T) != 0)
		protection_fault(addr);

	if constexpr (flat_readwrite_arena && std::is_const_v<T>) {
		if (LIKELY(addr + len - RWREAD_BEGIN < memory_arena_read_boundary() && addr < addr + len)) {
			const char* begin = &((const char *)m_arena.data)[RISCV_SPECSAFE(addr)];
			return (T*) begin;
		}
	} else if constexpr (flat_readwrite_arena) {
		if (LIKELY(addr + len - initial_rodata_end() < memory_arena_write_boundary() && addr < addr + len)) {
			char* begin = &((char *)m_arena.data)[RISCV_SPECSAFE(addr)];
			return (T*) begin;
		}
	}

	// TODO: Add try_gather_buffers_from_range
	return nullptr;
}

#ifdef RISCV_SPAN_AVAILABLE

template <int W>
template <typename T> inline
std::span<T> Memory<W>::memspan(address_t addr, size_t count, size_t maxlen) const
{
	if (addr % alignof(T) != 0)
		protection_fault(addr);

	if (count == 0)
		return {};

	if constexpr (std::is_const_v<T>) {
		auto view = memview(addr, count * sizeof(T), maxlen);
		if (view.size() == count * sizeof(T) && uintptr_t(view.data()) % alignof(T) == 0)
			return {(const T *)view.data(), count};
	} else {
		auto view = writable_memview(addr, count * sizeof(T), maxlen);
		if (view.size() == count * sizeof(T) && uintptr_t(view.data()) % alignof(T) == 0)
			return {(T *)view.data(), count};
	}

	// It's too dangerous to return an empty span here
	protection_fault(addr);
}

#endif // RISCV_SPAN_AVAILABLE

template <int W> inline
size_t Memory<W>::strlen(address_t addr, size_t maxlen) const
{
	size_t len = 0;

	do {
		const address_t offset = addr & (Page::size()-1);
		const address_t pageno = page_number(addr);
		const Page& page = this->get_readable_pageno(pageno);

		const char* start = (const char*) &page.data()[offset];
		const size_t max_bytes = Page::size() - offset;
		const size_t thislen = strnlen(start, max_bytes);
		len += thislen;
		if (thislen != max_bytes) break;
		addr += len;
	} while (len < maxlen);

	return (len <= maxlen) ? len : maxlen;
}

template <int W> inline
int Memory<W>::memcmp(address_t p1, address_t p2, size_t len) const
{
	if (UNLIKELY(p1 + len < p1))
		protection_fault(p1);
	if (UNLIKELY(p2 + len < p2))
		protection_fault(p2);

	// NOTE: fast implementation if no pointer crosses page boundary
	const auto pageno1 = this->page_number(p1);
	const auto pageno2 = this->page_number(p2);
	if (pageno1 == ((p1 + len-1) / Page::size()) &&
		pageno2 == ((p2 + len-1) / Page::size())) {
		auto& page1 = this->get_readable_pageno(pageno1);
		auto& page2 = this->get_readable_pageno(pageno2);

		const uint8_t* s1 = page1.data() + p1 % Page::SIZE;
		const uint8_t* s2 = page2.data() + p2 % Page::SIZE;
		return std::memcmp(s1, s2, len);
	}
	else // slow path (optimizable)
	{
		uint8_t v1 = 0;
		uint8_t v2 = 0;
		while (len > 0) {
			const auto slow_pageno1 = this->page_number(p1);
			const auto slow_pageno2 = this->page_number(p2);
			auto& page1 = this->get_readable_pageno(slow_pageno1);
			auto& page2 = this->get_readable_pageno(slow_pageno2);

			v1 = page1.data()[p1 % Page::SIZE];
			v2 = page2.data()[p2 % Page::SIZE];
			if (v1 != v2) break;
			p1++;
			p2++;
			len--;
		}
		return len == 0 ? 0 : (v1 - v2);
	}
}
template <int W> inline
int Memory<W>::memcmp(const void* ptr1, address_t p2, size_t len) const
{
	if (UNLIKELY(p2 + len < p2))
		protection_fault(p2);

	const char* s1 = (const char*) ptr1;
	// NOTE: fast implementation if no pointer crosses page boundary
	const auto pageno2 = this->page_number(p2);
	if (pageno2 == ((p2 + len-1) / Page::size())) {
		auto& page2 = this->get_readable_pageno(pageno2);

		const uint8_t* s2 = page2.data() + p2 % Page::SIZE;
		return std::memcmp(s1, s2, len);
	}
	else // slow path (optimizable)
	{
		uint8_t v2 = 0;
		while (len > 0) {
			const auto slow_pageno2 = this->page_number(p2);
			auto& page2 = this->get_readable_pageno(slow_pageno2);

			v2 = page2.data()[p2 % Page::SIZE];
			if (*s1 != v2) break;
			s1++;
			p2++;
			len--;
		}
		return len == 0 ? 0 : (*s1 - v2);
	}
}

template <int W> inline
void Memory<W>::memcpy(
	address_t dst, Machine<W>& srcm, address_t src, address_t len)
{
	if constexpr (riscv::flat_readwrite_arena) {
		// Fast-path: Find the entire source and destination buffers in the memory arena
		if (uint8_t* srcptr = srcm.memory.template try_memarray<uint8_t> (src, len)) {
			if (uint8_t* dstptr = this->template try_memarray<uint8_t> (dst, len)) {
				std::memcpy(dstptr, srcptr, len);
				return;
			}
			this->memcpy(dst, srcptr, len);
			return;
		}
	}
	if ((dst & (W-1)) == (src & (W-1))) {
		while ((src & (W-1)) != 0 && len > 0) {
			this->template write<uint8_t> (dst++,
				srcm.memory.template read<uint8_t> (src++));
			len --;
		}
		while (len >= 4*W) {
			if constexpr (riscv::flat_readwrite_arena) {
				// Fast-path: Find the entire source buffer in the memory arena using memarray()
				if (uint8_t* srcptr = srcm.memory.template try_memarray<uint8_t> (src, len)) {
					this->memcpy(dst, srcptr, len);
					return;
				}
			}
			this->template write<address_t> (dst + 0,
				srcm.memory.template read<address_t> (src + 0));
			this->template write<address_t> (dst + 1*W,
				srcm.memory.template read<address_t> (src + 1*W));
			this->template write<address_t> (dst + 2*W,
				srcm.memory.template read<address_t> (src + 2*W));
			this->template write<address_t> (dst + 3*W,
				srcm.memory.template read<address_t> (src + 3*W));
			dst += 4*W; src += 4*W; len -= 4*W;
		}
		while (len >= W) {
			this->template write<address_t> (dst,
				srcm.memory.template read<address_t> (src));
			dst += W; src += W; len -= W;
		}
	}
	while (len > 0) {
		this->template write<uint8_t> (dst++,
			srcm.memory.template read<uint8_t> (src++));
		len --;
	}
}

template <int W> inline
size_t Memory<W>::gather_buffers_from_range(
	size_t cnt, vBuffer buffers[], address_t addr, size_t len) const
{
	size_t index = 0;
	vBuffer* last = nullptr;
	while (len != 0 && index < cnt)
	{
		const size_t offset = addr & (Page::SIZE-1);
		const size_t size = std::min(Page::SIZE - offset, len);
		auto& page = get_readable_pageno(page_number(addr));

		auto* ptr = (char*) &page.data()[offset];
		if (last && ptr == last->ptr + last->len) {
			last->len += size;
		} else {
			last = &buffers[index];
			last->ptr = ptr;
			last->len = size;
			index ++;
		}
		addr += size;
		len -= size;
	}
	if (UNLIKELY(len != 0)) {
		machine().cpu.trigger_exception(OUT_OF_MEMORY, len);
	}
	return index;
}

template <int W> inline
size_t Memory<W>::gather_writable_buffers_from_range(
	size_t cnt, vBuffer buffers[], address_t addr, size_t len)
{
	size_t index = 0;
	vBuffer* last = nullptr;
	while (len != 0 && index < cnt)
	{
		const size_t offset = addr & (Page::SIZE-1);
		const size_t size = std::min(Page::SIZE - offset, len);
		auto& page = create_writable_pageno(page_number(addr));

		auto* ptr = (char*) &page.data()[offset];
		if (last && ptr == last->ptr + last->len) {
			last->len += size;
		} else {
			last = &buffers[index];
			last->ptr = ptr;
			last->len = size;
			index ++;
		}
		addr += size;
		len -= size;
	}
	if (UNLIKELY(len != 0)) {
		machine().cpu.trigger_exception(OUT_OF_MEMORY, index);
	}
	return index;
}
