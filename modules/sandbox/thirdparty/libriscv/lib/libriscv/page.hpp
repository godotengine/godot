#pragma once
#include "common.hpp"
#include "types.hpp"
#include <cassert>
#include <memory>
#include <array>

namespace riscv {

struct PageAttributes
{
	bool read = true;
	bool write = true;
	bool exec = false;
	bool is_cow = false;
	bool non_owning = false;
	bool dont_fork = false;
	mutable bool cacheable = true;
	uint8_t user_defined = 0; /* Use this for yourself */

	int to_prot() const noexcept {
		return this->read | (this->write << 1) | (this->exec << 2);
	}

	constexpr bool is_cacheable() const noexcept {
		// Cacheable only makes sense when memory traps are enabled
		if constexpr (memory_traps_enabled)
			return cacheable;
		else
			return true;
	}
	bool is_default() const noexcept {
		constexpr PageAttributes def {};
		return this->read == def.read && this->write == def.write && this->exec == def.exec;
	}
	void apply_regular_attributes(PageAttributes other) {
		const auto no = this->non_owning;
		*this = other;
		this->non_owning = no;
	}
};

struct PageData {
	std::array<uint8_t, PageSize> buffer8;

	template <typename T>
	inline T& aligned_read(uint32_t offset) const
	{
		if constexpr (memory_alignment_check) {
			if (offset % sizeof(T))
#if __cpp_exceptions
				throw MachineException(INVALID_ALIGNMENT, "Misaligned read", offset);
#else
				std::abort();
#endif
		}
		return *(T*) &buffer8[offset];
	}

	template <typename T>
	inline void aligned_write(uint32_t offset, T value)
	{
		if constexpr (memory_alignment_check) {
			if (offset % sizeof(T))
#if __cpp_exceptions
				throw MachineException(INVALID_ALIGNMENT, "Misaligned write", offset);
#else
				std::abort();
#endif
		}
		*(T*) &buffer8[offset] = value;
	}

	PageData() noexcept : buffer8{} {}
	PageData(const PageData& other) noexcept : buffer8{other.buffer8} {}
	PageData(const std::array<uint8_t, PageSize>& data) noexcept : buffer8{data} {}
	enum Initialization { INITIALIZED, UNINITIALIZED };
	PageData(Initialization i) noexcept { if (i == INITIALIZED) buffer8 = {}; }
};

struct Page
{
	static constexpr unsigned SIZE = PageSize;

	using mmio_cb_t = std::function<void(Page&, uint32_t, int, int64_t)>;

	// create a new blank page
	Page() { m_page.reset(new PageData {}); };
	// create a new possibly uninitialized page
	Page(PageData::Initialization i) { m_page.reset(new PageData {i}); };
	// copy another page (or data)
	Page(const PageAttributes& a, const PageData& d = {})
		: attr(a), m_page(new PageData{d}) { attr.non_owning = false; }
	Page(Page&& other) noexcept
		: attr(other.attr), m_page(std::move(other.m_page)) {}
	Page& operator= (Page&& other) noexcept {
		attr = other.attr;
		m_page = std::move(other.m_page);
		return *this;
	}
	// create a page that doesn't own this memory
	Page(const PageAttributes& a, PageData* data);
	// don't try to free non-owned page memory
	~Page() {
		if (attr.non_owning) m_page.release();
	}

	PageData& page() noexcept { return *m_page; }
	const PageData& page() const noexcept { return *m_page; }

	std::string to_string() const;

	uint8_t* data() noexcept {
		return page().buffer8.data();
	}
	const uint8_t* data() const noexcept {
		return page().buffer8.data();
	}
	void new_data(PageData* data, bool data_owned);

	static constexpr size_t size() noexcept {
		return SIZE;
	}

	bool is_cow_page() const noexcept { return this->data() == cow_page().data(); }

	static const Page& cow_page() noexcept;
	static const Page& guard_page() noexcept;
	static const Page& host_page() noexcept;

	/* Transform a CoW-page to an owned writable page */
	void make_writable()
	{
		if (m_page != nullptr)
		{
			auto* new_data = new PageData {*m_page};
			if (attr.non_owning) m_page.release();
			m_page.reset(new_data);
		} else {
			m_page.reset(new PageData {});
		}
		attr.write = true;
		attr.is_cow = false;
		attr.non_owning = false;
	}
	void write_to_another(PageData* other)
	{
		if (this->m_page != nullptr)
		{
			other->buffer8 = m_page->buffer8;
			if (attr.non_owning) m_page.release();
		}
		m_page.reset(other);
		attr.write = true;
		attr.is_cow = false;
		attr.non_owning = true;
	}

	// Loan a page from somewhere else, that will not be
	// deleted here. There is no ref-counting mechanism, and
	// the memory is ultimately owned by the master page.
	void loan(const Page& master_page) {
		this->attr = master_page.attr;
		this->attr.non_owning = true;
		this->m_page.reset(master_page.m_page.get());
	}

	// this combination has been benchmarked to be faster than
	// page-aligning the PageData struct and putting it first
	PageAttributes attr;
	std::unique_ptr<PageData> m_page;

	bool has_trap() const noexcept { return m_trap != nullptr; }
	// NOTE: Setting a trap makes the page uncacheable
	bool set_trap(mmio_cb_t newtrap) const;
	void trap(uint32_t offset, int mode, int64_t value) const;
	static int trap_mode(int mode) noexcept { return mode & 0xF000; }
	static int trap_size(int mode) noexcept { return mode & 0x0FFF; }

	mutable mmio_cb_t m_trap = nullptr;
};

inline Page::Page(const PageAttributes& a, PageData* data)
	: attr(a)
{
	if (UNLIKELY(data == nullptr))
#if __cpp_exceptions
		throw MachineException(ILLEGAL_OPERATION, "Tried to create a page with no page data");
#else
		std::abort();
#endif
	attr.non_owning = true;
	m_page.reset(data);
}

inline void Page::new_data(PageData* data, bool data_owned)
{
	if (this->attr.non_owning)
		this->m_page.release();
	this->m_page.reset(data);
	this->attr.non_owning = !data_owned;
}

inline void Page::trap(uint32_t offset, int mode, int64_t value) const
{
	this->m_trap((Page&) *this, offset, mode, value);
}
inline bool Page::set_trap(mmio_cb_t newtrap) const {
	if constexpr (memory_traps_enabled) {
		// Setting a trap makes the page uncacheable and vice versa
		// This is done so that reads and writes always trigger the
		// slow-path that allows trapping.
		this->attr.cacheable = (newtrap == nullptr);
		this->m_trap = newtrap;
		return true;
	} else {
		(void) newtrap;
		return false;
	}
}

inline std::string Page::to_string() const
{
	return "Readable: " + std::string(attr.read ? "[x]" : "[ ]") +
		"  Writable: " + std::string(attr.write ? "[x]" : "[ ]") +
		"  Executable: " + std::string(attr.exec ? "[x]" : "[ ]");
}

// Helper class for caching pages
template <int W, typename T> struct CachedPage {
	address_type<W> pageno = (address_type<W>)-1;
	T* page = nullptr;

	void reset() { pageno = (address_type<W>)-1; page = nullptr; }
};

}
