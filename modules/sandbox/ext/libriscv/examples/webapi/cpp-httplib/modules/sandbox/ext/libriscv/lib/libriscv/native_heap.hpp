//
// C++ Header-Only Separate Address-Space Allocator
// by fwsGonzo, originally based on allocator written in C by Snaipe
//
#pragma once
#include "common.hpp"
#include <cstddef>
#include <cassert>
#include <deque>
#include <unordered_map>
#include "util/function.hpp"

namespace riscv
{
struct Arena;

struct ArenaChunk
{
	using PointerType = uint32_t;

	ArenaChunk() = default;
	ArenaChunk(ArenaChunk* n, ArenaChunk* p, size_t s, bool f, PointerType d)
		: next(n), prev(p), size(s), free(f), data(d) {}

	ArenaChunk* next = nullptr;
	ArenaChunk* prev = nullptr;
	size_t size = 0;
	bool   free = false;
	PointerType data = 0;

	ArenaChunk* find_used(PointerType ptr);
	ArenaChunk* find_free(size_t size);
	void merge_next(Arena&);
	void split_next(Arena&, size_t size);
	void subsume_next(Arena&, size_t extra);
};

struct Arena
{
	static constexpr size_t ALIGNMENT = 16u;
	using PointerType = ArenaChunk::PointerType;
	using ReallocResult = std::tuple<PointerType, size_t>;
	using unknown_realloc_func_t = Function<ReallocResult(PointerType, size_t)>;
	using unknown_free_func_t = Function<int(PointerType, ArenaChunk *)>;

	/// @brief Construct an arena that manages allocations for a given memory range.
	/// @param base The base address of the memory range.
	/// @param end  The end address of the memory range.
	Arena(PointerType base, PointerType end);

	/// @brief Transfer allocations from another arena.
	/// @param other The arena to transfer allocations from.
	/// @note The other arena is left unchanged, allowing for multiple transfers.
	Arena(const Arena& other);

	/// @brief Allocate memory from the arena.
	/// @param size The size of the allocation.
	/// @return Address to the allocated memory range, or 0 if allocation failed.
	/// @note The memory range is not guaranteed to be zeroed. 8-byte alignment is guaranteed.
	PointerType malloc(size_t size);

	/// @brief Reallocate memory from the arena.
	/// @param ptr The allocation to resize.
	/// @param newsize The new size of the allocation.
	/// @return Pointer to the reallocated memory range, or 0 if reallocation failed.
	/// @note Memory is not moved here, only the allocation itself. An implementor
	/// should copy the data from the old pointer to the new pointer if necessary.
	ReallocResult realloc(PointerType old, size_t size);

	/// @brief Get the size of an allocation.
	/// @param src The pointer to the memory range.
	/// @param allow_free Whether to allow querying the size of a free chunk.
	/// @return The size of the memory range, or 0 if the pointer is invalid.
	size_t      size(PointerType src, bool allow_free = false);

	/// @brief Free a previous allocation.
	/// @param src The pointer to the memory range.
	/// @return 0 if the memory range was successfully freed, or -1 if the pointer is invalid.
	signed int  free(PointerType);

	/// @brief Attempt to allocate a fully sequential memory range,
	/// unless the arena is flat, in which case all memory is sequential.
	/// Alignment is currently ignored, but 8-byte alignment is guaranteed.
	/// @param size The size of the memory range to allocate.
	/// @param alignment The alignment of the returned address.
	/// @param arena_is_flat Whether the arena is flat. A configuration option.
	/// @return Pointer to the allocated memory range, or 0 if allocation failed.
	/// @note The memory range is not guaranteed to be zeroed.
	/// @note If an excessive amount of chunks are allocated, an exception is thrown.
	PointerType seq_alloc_aligned(size_t size, size_t alignment, bool arena_is_flat = riscv::flat_readwrite_arena);

	size_t bytes_free() const;
	size_t bytes_used() const;
	size_t chunks_used() const noexcept { return m_chunks.size(); }

	void set_max_chunks(unsigned new_max) { this->m_max_chunks = new_max; }

	unsigned allocation_counter() const noexcept { return m_allocation_counter; }
	unsigned deallocation_counter() const noexcept { return m_deallocation_counter; }

	void transfer(Arena& dest) const;

	void on_unknown_free(unknown_free_func_t func) {
		m_free_unknown_chunk = std::move(func);
	}
	void on_unknown_realloc(unknown_realloc_func_t func) {
		m_realloc_unknown_chunk = std::move(func);
	}

	/** Internal usage **/
	inline ArenaChunk& base_chunk() {
		return m_base_chunk;
	}
	template <typename... Args>
	ArenaChunk* new_chunk(Args&&... args);
	void   free_chunk(ArenaChunk*);
	ArenaChunk* find_chunk(PointerType ptr);

	static size_t word_align(size_t size) {
		return (size + (ALIGNMENT-1)) & ~(ALIGNMENT-1);
	}
	static size_t fixup_size(size_t size) {
		// The minimum allocation is 8 bytes
		return std::max(ALIGNMENT, word_align(size));
	}
private:
	void internal_free(ArenaChunk* ch);
	void foreach(Function<void(const ArenaChunk&)>) const;
	ArenaChunk* begin_find_used(PointerType ptr);

	std::deque<ArenaChunk> m_chunks;
	std::vector<ArenaChunk*> m_free_chunks;
#ifdef ENABLE_ARENA_CHUNK_MAP
	std::unordered_map<PointerType, ArenaChunk*> m_used_chunk_map;
#endif
	ArenaChunk  m_base_chunk;
	unsigned    m_max_chunks = 4'000u;
	unsigned    m_allocation_counter = 0u;
	unsigned    m_deallocation_counter = 0u;

	unknown_free_func_t m_free_unknown_chunk
		= [] (auto, auto*) { return -1; };
	unknown_realloc_func_t m_realloc_unknown_chunk
		= [] (auto, auto) { return ReallocResult{0, 0}; };
	friend struct ArenaChunk;
};

inline ArenaChunk* Arena::begin_find_used(PointerType ptr)
{
#ifdef ENABLE_ARENA_CHUNK_MAP
	auto it = m_used_chunk_map.find(ptr);
	if (it != m_used_chunk_map.end())
		return it->second;
	return nullptr;
#else
	return base_chunk().find_used(ptr);
#endif
}

// find exact free chunk that matches ptr
inline ArenaChunk* ArenaChunk::find_used(PointerType ptr)
{
	ArenaChunk* ch = this;
	while (ch != nullptr) {
		if (!ch->free && ch->data == ptr)
			return ch;
		ch = ch->next;
	}
	return nullptr;
}
// find free chunk that has at least given size
inline ArenaChunk* ArenaChunk::find_free(size_t size)
{
    ArenaChunk* ch = this;
	while (ch != nullptr) {
		if (ch->free && ch->size >= size)
			return ch;
		ch = ch->next;
	}
	return nullptr;
}
// merge this and next into this chunk
inline void ArenaChunk::merge_next(Arena& arena)
{
	ArenaChunk* freech = this->next;
	this->size += freech->size;
	this->next = freech->next;
	if (this->next) {
		this->next->prev = this;
	}
	arena.free_chunk(freech);
}

inline void ArenaChunk::subsume_next(Arena& arena, size_t newlen)
{
	assert(this->size < newlen);
	ArenaChunk* ch = this->next;
	assert(ch);

	if (this->size + ch->size < newlen)
		return;

	const size_t subsume = newlen - this->size;
	ch->size -= subsume;
	ch->data += subsume;
	this->size = newlen;

	// Free the next chunk if we ate all of it
	if (ch->size == 0) {
		this->next = ch->next;
		if (this->next) {
			this->next->prev = this;
		}
		arena.free_chunk(ch);
	}
}

inline void ArenaChunk::split_next(Arena& arena, size_t size)
{
	// Only split if the new chunk would not be empty
	if (this->size > size)
	{
		ArenaChunk* newch = arena.new_chunk(
			this->next,
			this,
			this->size - size,
			true, // free
			this->data + (PointerType) size
		);
		if (this->next) {
			this->next->prev = newch;
		}
		this->next = newch;
	} else {
		// If the new chunk would be empty, connect distant chunks instead
		if (this->prev && this->next && this->prev->free && this->next->free) {
			this->prev->next = this->next;
			this->next->prev = this->prev;
		} else if (this->prev && this->prev->free) {
			this->prev->next = nullptr;
		} else if (this->next && this->next->free) {
			this->next->prev = nullptr;
		}
	}
	this->size = size;
}

template <typename... Args>
inline ArenaChunk* Arena::new_chunk(Args&&... args)
{
	if (UNLIKELY(m_free_chunks.empty())) {
		if (m_chunks.size() >= this->m_max_chunks)
			throw MachineException(INVALID_PROGRAM, "Too many arena chunks", this->m_max_chunks);

		m_chunks.emplace_back(std::forward<Args>(args)...);
		return &m_chunks.back();
	}
	auto* chunk = m_free_chunks.back();
	m_free_chunks.pop_back();
	return new (chunk) ArenaChunk {std::forward<Args>(args)...};
}
inline void Arena::free_chunk(ArenaChunk* chunk)
{
	m_free_chunks.push_back(chunk);
}
inline ArenaChunk* Arena::find_chunk(PointerType ptr)
{
	for (auto& chunk : m_chunks) {
		if (!chunk.free && chunk.data == ptr)
			return &chunk;
	}
	return nullptr;
}

inline void Arena::internal_free(ArenaChunk* ch)
{
	this->m_deallocation_counter++;
#ifdef ENABLE_ARENA_CHUNK_MAP
	this->m_used_chunk_map.erase(ch->data);
#endif
	ch->free = true;
	// merge chunks ahead and behind us
	if (ch->next && ch->next->free) {
		ch->merge_next(*this);
	}
	if (ch->prev && ch->prev->free) {
		ch = ch->prev;
		ch->merge_next(*this);
	}
}

inline Arena::PointerType Arena::malloc(size_t size)
{
	const size_t length = fixup_size(size);
	ArenaChunk* ch = base_chunk().find_free(length);
	this->m_allocation_counter++;

	if (ch != nullptr) {
		ch->split_next(*this, length);
		ch->free = false;

#ifdef ENABLE_ARENA_CHUNK_MAP
		this->m_used_chunk_map.insert_or_assign(ch->data, ch);
#endif
		return ch->data;
	}
	return 0;
}

// Guarantee that allocation is sequential in memory
// when accessed outside of emulation. A single page
// has fully sequential memory within itself, so we can
// always allocate sequential memory within a single page.
inline Arena::PointerType Arena::seq_alloc_aligned(size_t size, size_t alignment, bool arena_is_flat)
{
	(void)alignment;

	if (arena_is_flat) {
		return malloc(size);
	}

	// XXX: Alignment is ignored for now,
	// but 16-byte alignment is guaranteed.
	const size_t objectsize = fixup_size(size);
	this->m_allocation_counter++;
	if (objectsize > RISCV_PAGE_SIZE)
		throw MachineException(INVALID_PROGRAM, "Requested sequential allocation too large", objectsize);

	ArenaChunk* ch = &base_chunk();
restart_seq_alloc_search:
	// Find memory that can always cover the object sequentially
	ch = ch->find_free(objectsize);

	if (ch != nullptr) {
		// Check if data + size is on the same page
		if ((ch->data & ~(RISCV_PAGE_SIZE-1)) !=
			((ch->data + objectsize - 1) & ~(RISCV_PAGE_SIZE-1)))
		{
			// The next page boundary
			const PointerType boundary = (ch->data + objectsize - 1) & ~(RISCV_PAGE_SIZE-1);
			if (boundary < ch->data)
				throw MachineException(INVALID_PROGRAM, "Page boundary overflow", boundary);
			// Figure out the size until the boundary
			const size_t remaining = boundary - ch->data;
			// If the chunks new size would be too small, find a new chunk instead
			if (ch->size - remaining < objectsize) {
				if (ch->next == nullptr) {
					// We ran out of arena space
					return 0;
				}
				ch = ch->next;
				goto restart_seq_alloc_search;
			}
			// Split at the page boundary
			ch->split_next(*this, remaining);
			ch = ch->next;
		}

		ch->split_next(*this, objectsize);
		ch->free = false;
#ifdef ENABLE_ARENA_CHUNK_MAP
		this->m_used_chunk_map.insert_or_assign(ch->data, ch);
#endif
		return ch->data;
	}
	return 0;
}

inline Arena::ReallocResult
	Arena::realloc(PointerType ptr, size_t newsize)
{
	if (ptr == 0x0) // Regular malloc
		return {malloc(newsize), 0};

	ArenaChunk* ch = this->begin_find_used(ptr);
	if (UNLIKELY(ch == nullptr || ch->free)) {
		// Realloc failure handler
		return m_realloc_unknown_chunk(ptr, newsize);
	}

	newsize = fixup_size(newsize);
	if (ch->size >= newsize) // Already long enough?
		return {ch->data, 0};

	// We return the old length to aid memcpy
	const size_t old_len = ch->size;
	// Try to eat from the next chunk
	if (ch->next && ch->next->free) {
		ch->subsume_next(*this, newsize);
		if (ch->size >= newsize)
			return {ch->data, 0};
	}

	// Fallback to malloc, then free the old chunk
	ptr = malloc(newsize);
	if (ptr != 0x0) {
		this->internal_free(ch);
		return {ptr, old_len};
	}

	return {0x0, 0x0};
}

inline size_t Arena::size(PointerType ptr, bool allow_free)
{
	ArenaChunk* ch = this->begin_find_used(ptr);
	if (UNLIKELY(ch == nullptr || (ch->free && !allow_free)))
		return 0;
	return ch->size;
}

inline int Arena::free(PointerType ptr)
{
	ArenaChunk* ch = this->begin_find_used(ptr);
	if (UNLIKELY(ch == nullptr || ch->free))
		return m_free_unknown_chunk(ptr, ch);

	this->internal_free(ch);
	return 0;
}

inline Arena::Arena(PointerType arena_base, PointerType arena_end)
{
	m_base_chunk.size = arena_end - arena_base;
	m_base_chunk.data = arena_base;
	m_base_chunk.free = true;
}

inline void Arena::foreach(Function<void(const ArenaChunk&)> callback) const
{
	const ArenaChunk* ch = &this->m_base_chunk;
    while (ch != nullptr) {
		callback(*ch);
		ch = ch->next;
	}
}

inline size_t Arena::bytes_free() const
{
	size_t size = 0;
	foreach([&size] (const ArenaChunk& chunk) {
		if (chunk.free) size += chunk.size;
	});
	return size;
}
inline size_t Arena::bytes_used() const
{
	size_t size = 0;
	foreach([&size] (const ArenaChunk& chunk) {
		if (!chunk.free) size += chunk.size;
	});
	return size;
}

inline Arena::Arena(const Arena& other)
{
	other.transfer(*this);
}

inline void Arena::transfer(Arena& dest) const
{
	dest.m_base_chunk = m_base_chunk;
	dest.m_chunks.clear();
	dest.m_free_chunks.clear();

	ArenaChunk* last = &dest.m_base_chunk;

	const ArenaChunk* chunk = m_base_chunk.next;
	while (chunk != nullptr)
	{
		dest.m_chunks.push_back(*chunk);
		auto& new_chunk = dest.m_chunks.back();
		new_chunk.prev = last;
		new_chunk.next = nullptr;
		last->next = &new_chunk;
		/* New last before next iteration */
		last = &new_chunk;

		chunk = chunk->next;
	}
}

} // namespace riscv
