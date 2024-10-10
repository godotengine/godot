/*
 * Copyright 2019-2021 Hans-Kristian Arntzen
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * At your option, you may choose to accept this material under either:
 *  1. The Apache License, Version 2.0, found at <http://www.apache.org/licenses/LICENSE-2.0>, or
 *  2. The MIT License, found at <http://opensource.org/licenses/MIT>.
 */

#ifndef SPIRV_CROSS_CONTAINERS_HPP
#define SPIRV_CROSS_CONTAINERS_HPP

#include "spirv_cross_error_handling.hpp"
#include <algorithm>
#include <exception>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <stack>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef SPIRV_CROSS_NAMESPACE_OVERRIDE
#define SPIRV_CROSS_NAMESPACE SPIRV_CROSS_NAMESPACE_OVERRIDE
#else
#define SPIRV_CROSS_NAMESPACE spirv_cross
#endif

namespace SPIRV_CROSS_NAMESPACE
{
#ifndef SPIRV_CROSS_FORCE_STL_TYPES
// std::aligned_storage does not support size == 0, so roll our own.
template <typename T, size_t N>
class AlignedBuffer
{
public:
	T *data()
	{
#if defined(_MSC_VER) && _MSC_VER < 1900
		// MSVC 2013 workarounds, sigh ...
		// Only use this workaround on MSVC 2013 due to some confusion around default initialized unions.
		// Spec seems to suggest the memory will be zero-initialized, which is *not* what we want.
		return reinterpret_cast<T *>(u.aligned_char);
#else
		return reinterpret_cast<T *>(aligned_char);
#endif
	}

private:
#if defined(_MSC_VER) && _MSC_VER < 1900
	// MSVC 2013 workarounds, sigh ...
	union
	{
		char aligned_char[sizeof(T) * N];
		double dummy_aligner;
	} u;
#else
	alignas(T) char aligned_char[sizeof(T) * N];
#endif
};

template <typename T>
class AlignedBuffer<T, 0>
{
public:
	T *data()
	{
		return nullptr;
	}
};

// An immutable version of SmallVector which erases type information about storage.
template <typename T>
class VectorView
{
public:
	T &operator[](size_t i) SPIRV_CROSS_NOEXCEPT
	{
		return ptr[i];
	}

	const T &operator[](size_t i) const SPIRV_CROSS_NOEXCEPT
	{
		return ptr[i];
	}

	bool empty() const SPIRV_CROSS_NOEXCEPT
	{
		return buffer_size == 0;
	}

	size_t size() const SPIRV_CROSS_NOEXCEPT
	{
		return buffer_size;
	}

	T *data() SPIRV_CROSS_NOEXCEPT
	{
		return ptr;
	}

	const T *data() const SPIRV_CROSS_NOEXCEPT
	{
		return ptr;
	}

	T *begin() SPIRV_CROSS_NOEXCEPT
	{
		return ptr;
	}

	T *end() SPIRV_CROSS_NOEXCEPT
	{
		return ptr + buffer_size;
	}

	const T *begin() const SPIRV_CROSS_NOEXCEPT
	{
		return ptr;
	}

	const T *end() const SPIRV_CROSS_NOEXCEPT
	{
		return ptr + buffer_size;
	}

	T &front() SPIRV_CROSS_NOEXCEPT
	{
		return ptr[0];
	}

	const T &front() const SPIRV_CROSS_NOEXCEPT
	{
		return ptr[0];
	}

	T &back() SPIRV_CROSS_NOEXCEPT
	{
		return ptr[buffer_size - 1];
	}

	const T &back() const SPIRV_CROSS_NOEXCEPT
	{
		return ptr[buffer_size - 1];
	}

	// Makes it easier to consume SmallVector.
#if defined(_MSC_VER) && _MSC_VER < 1900
	explicit operator std::vector<T>() const
	{
		// Another MSVC 2013 workaround. It does not understand lvalue/rvalue qualified operations.
		return std::vector<T>(ptr, ptr + buffer_size);
	}
#else
	// Makes it easier to consume SmallVector.
	explicit operator std::vector<T>() const &
	{
		return std::vector<T>(ptr, ptr + buffer_size);
	}

	// If we are converting as an r-value, we can pilfer our elements.
	explicit operator std::vector<T>() &&
	{
		return std::vector<T>(std::make_move_iterator(ptr), std::make_move_iterator(ptr + buffer_size));
	}
#endif

	// Avoid sliced copies. Base class should only be read as a reference.
	VectorView(const VectorView &) = delete;
	void operator=(const VectorView &) = delete;

protected:
	VectorView() = default;
	T *ptr = nullptr;
	size_t buffer_size = 0;
};

// Simple vector which supports up to N elements inline, without malloc/free.
// We use a lot of throwaway vectors all over the place which triggers allocations.
// This class only implements the subset of std::vector we need in SPIRV-Cross.
// It is *NOT* a drop-in replacement in general projects.
template <typename T, size_t N = 8>
class SmallVector : public VectorView<T>
{
public:
	SmallVector() SPIRV_CROSS_NOEXCEPT
	{
		this->ptr = stack_storage.data();
		buffer_capacity = N;
	}

	template <typename U>
	SmallVector(const U *arg_list_begin, const U *arg_list_end) SPIRV_CROSS_NOEXCEPT : SmallVector()
	{
		auto count = size_t(arg_list_end - arg_list_begin);
		reserve(count);
		for (size_t i = 0; i < count; i++, arg_list_begin++)
			new (&this->ptr[i]) T(*arg_list_begin);
		this->buffer_size = count;
	}

	template <typename U>
	SmallVector(std::initializer_list<U> init) SPIRV_CROSS_NOEXCEPT : SmallVector(init.begin(), init.end())
	{
	}

	template <typename U, size_t M>
	explicit SmallVector(const U (&init)[M]) SPIRV_CROSS_NOEXCEPT : SmallVector(init, init + M)
	{
	}

	SmallVector(SmallVector &&other) SPIRV_CROSS_NOEXCEPT : SmallVector()
	{
		*this = std::move(other);
	}

	SmallVector &operator=(SmallVector &&other) SPIRV_CROSS_NOEXCEPT
	{
		clear();
		if (other.ptr != other.stack_storage.data())
		{
			// Pilfer allocated pointer.
			if (this->ptr != stack_storage.data())
				free(this->ptr);
			this->ptr = other.ptr;
			this->buffer_size = other.buffer_size;
			buffer_capacity = other.buffer_capacity;
			other.ptr = nullptr;
			other.buffer_size = 0;
			other.buffer_capacity = 0;
		}
		else
		{
			// Need to move the stack contents individually.
			reserve(other.buffer_size);
			for (size_t i = 0; i < other.buffer_size; i++)
			{
				new (&this->ptr[i]) T(std::move(other.ptr[i]));
				other.ptr[i].~T();
			}
			this->buffer_size = other.buffer_size;
			other.buffer_size = 0;
		}
		return *this;
	}

	SmallVector(const SmallVector &other) SPIRV_CROSS_NOEXCEPT : SmallVector()
	{
		*this = other;
	}

	SmallVector &operator=(const SmallVector &other) SPIRV_CROSS_NOEXCEPT
	{
		if (this == &other)
			return *this;

		clear();
		reserve(other.buffer_size);
		for (size_t i = 0; i < other.buffer_size; i++)
			new (&this->ptr[i]) T(other.ptr[i]);
		this->buffer_size = other.buffer_size;
		return *this;
	}

	explicit SmallVector(size_t count) SPIRV_CROSS_NOEXCEPT : SmallVector()
	{
		resize(count);
	}

	~SmallVector()
	{
		clear();
		if (this->ptr != stack_storage.data())
			free(this->ptr);
	}

	void clear() SPIRV_CROSS_NOEXCEPT
	{
		for (size_t i = 0; i < this->buffer_size; i++)
			this->ptr[i].~T();
		this->buffer_size = 0;
	}

	void push_back(const T &t) SPIRV_CROSS_NOEXCEPT
	{
		reserve(this->buffer_size + 1);
		new (&this->ptr[this->buffer_size]) T(t);
		this->buffer_size++;
	}

	void push_back(T &&t) SPIRV_CROSS_NOEXCEPT
	{
		reserve(this->buffer_size + 1);
		new (&this->ptr[this->buffer_size]) T(std::move(t));
		this->buffer_size++;
	}

	void pop_back() SPIRV_CROSS_NOEXCEPT
	{
		// Work around false positive warning on GCC 8.3.
		// Calling pop_back on empty vector is undefined.
		if (!this->empty())
			resize(this->buffer_size - 1);
	}

	template <typename... Ts>
	void emplace_back(Ts &&... ts) SPIRV_CROSS_NOEXCEPT
	{
		reserve(this->buffer_size + 1);
		new (&this->ptr[this->buffer_size]) T(std::forward<Ts>(ts)...);
		this->buffer_size++;
	}

	void reserve(size_t count) SPIRV_CROSS_NOEXCEPT
	{
		if ((count > (std::numeric_limits<size_t>::max)() / sizeof(T)) ||
		    (count > (std::numeric_limits<size_t>::max)() / 2))
		{
			// Only way this should ever happen is with garbage input, terminate.
			std::terminate();
		}

		if (count > buffer_capacity)
		{
			size_t target_capacity = buffer_capacity;
			if (target_capacity == 0)
				target_capacity = 1;

			// Weird parens works around macro issues on Windows if NOMINMAX is not used.
			target_capacity = (std::max)(target_capacity, N);

			// Need to ensure there is a POT value of target capacity which is larger than count,
			// otherwise this will overflow.
			while (target_capacity < count)
				target_capacity <<= 1u;

			T *new_buffer =
			    target_capacity > N ? static_cast<T *>(malloc(target_capacity * sizeof(T))) : stack_storage.data();

			// If we actually fail this malloc, we are hosed anyways, there is no reason to attempt recovery.
			if (!new_buffer)
				std::terminate();

			// In case for some reason two allocations both come from same stack.
			if (new_buffer != this->ptr)
			{
				// We don't deal with types which can throw in move constructor.
				for (size_t i = 0; i < this->buffer_size; i++)
				{
					new (&new_buffer[i]) T(std::move(this->ptr[i]));
					this->ptr[i].~T();
				}
			}

			if (this->ptr != stack_storage.data())
				free(this->ptr);
			this->ptr = new_buffer;
			buffer_capacity = target_capacity;
		}
	}

	void insert(T *itr, const T *insert_begin, const T *insert_end) SPIRV_CROSS_NOEXCEPT
	{
		auto count = size_t(insert_end - insert_begin);
		if (itr == this->end())
		{
			reserve(this->buffer_size + count);
			for (size_t i = 0; i < count; i++, insert_begin++)
				new (&this->ptr[this->buffer_size + i]) T(*insert_begin);
			this->buffer_size += count;
		}
		else
		{
			if (this->buffer_size + count > buffer_capacity)
			{
				auto target_capacity = this->buffer_size + count;
				if (target_capacity == 0)
					target_capacity = 1;
				if (target_capacity < N)
					target_capacity = N;

				while (target_capacity < count)
					target_capacity <<= 1u;

				// Need to allocate new buffer. Move everything to a new buffer.
				T *new_buffer =
				    target_capacity > N ? static_cast<T *>(malloc(target_capacity * sizeof(T))) : stack_storage.data();

				// If we actually fail this malloc, we are hosed anyways, there is no reason to attempt recovery.
				if (!new_buffer)
					std::terminate();

				// First, move elements from source buffer to new buffer.
				// We don't deal with types which can throw in move constructor.
				auto *target_itr = new_buffer;
				auto *original_source_itr = this->begin();

				if (new_buffer != this->ptr)
				{
					while (original_source_itr != itr)
					{
						new (target_itr) T(std::move(*original_source_itr));
						original_source_itr->~T();
						++original_source_itr;
						++target_itr;
					}
				}

				// Copy-construct new elements.
				for (auto *source_itr = insert_begin; source_itr != insert_end; ++source_itr, ++target_itr)
					new (target_itr) T(*source_itr);

				// Move over the other half.
				if (new_buffer != this->ptr || insert_begin != insert_end)
				{
					while (original_source_itr != this->end())
					{
						new (target_itr) T(std::move(*original_source_itr));
						original_source_itr->~T();
						++original_source_itr;
						++target_itr;
					}
				}

				if (this->ptr != stack_storage.data())
					free(this->ptr);
				this->ptr = new_buffer;
				buffer_capacity = target_capacity;
			}
			else
			{
				// Move in place, need to be a bit careful about which elements are constructed and which are not.
				// Move the end and construct the new elements.
				auto *target_itr = this->end() + count;
				auto *source_itr = this->end();
				while (target_itr != this->end() && source_itr != itr)
				{
					--target_itr;
					--source_itr;
					new (target_itr) T(std::move(*source_itr));
				}

				// For already constructed elements we can move-assign.
				std::move_backward(itr, source_itr, target_itr);

				// For the inserts which go to already constructed elements, we can do a plain copy.
				while (itr != this->end() && insert_begin != insert_end)
					*itr++ = *insert_begin++;

				// For inserts into newly allocated memory, we must copy-construct instead.
				while (insert_begin != insert_end)
				{
					new (itr) T(*insert_begin);
					++itr;
					++insert_begin;
				}
			}

			this->buffer_size += count;
		}
	}

	void insert(T *itr, const T &value) SPIRV_CROSS_NOEXCEPT
	{
		insert(itr, &value, &value + 1);
	}

	T *erase(T *itr) SPIRV_CROSS_NOEXCEPT
	{
		std::move(itr + 1, this->end(), itr);
		this->ptr[--this->buffer_size].~T();
		return itr;
	}

	void erase(T *start_erase, T *end_erase) SPIRV_CROSS_NOEXCEPT
	{
		if (end_erase == this->end())
		{
			resize(size_t(start_erase - this->begin()));
		}
		else
		{
			auto new_size = this->buffer_size - (end_erase - start_erase);
			std::move(end_erase, this->end(), start_erase);
			resize(new_size);
		}
	}

	void resize(size_t new_size) SPIRV_CROSS_NOEXCEPT
	{
		if (new_size < this->buffer_size)
		{
			for (size_t i = new_size; i < this->buffer_size; i++)
				this->ptr[i].~T();
		}
		else if (new_size > this->buffer_size)
		{
			reserve(new_size);
			for (size_t i = this->buffer_size; i < new_size; i++)
				new (&this->ptr[i]) T();
		}

		this->buffer_size = new_size;
	}

private:
	size_t buffer_capacity = 0;
	AlignedBuffer<T, N> stack_storage;
};

// A vector without stack storage.
// Could also be a typedef-ed to std::vector,
// but might as well use the one we have.
template <typename T>
using Vector = SmallVector<T, 0>;

#else // SPIRV_CROSS_FORCE_STL_TYPES

template <typename T, size_t N = 8>
using SmallVector = std::vector<T>;
template <typename T>
using Vector = std::vector<T>;
template <typename T>
using VectorView = std::vector<T>;

#endif // SPIRV_CROSS_FORCE_STL_TYPES

// An object pool which we use for allocating IVariant-derived objects.
// We know we are going to allocate a bunch of objects of each type,
// so amortize the mallocs.
class ObjectPoolBase
{
public:
	virtual ~ObjectPoolBase() = default;
	virtual void deallocate_opaque(void *ptr) = 0;
};

template <typename T>
class ObjectPool : public ObjectPoolBase
{
public:
	explicit ObjectPool(unsigned start_object_count_ = 16)
	    : start_object_count(start_object_count_)
	{
	}

	template <typename... P>
	T *allocate(P &&... p)
	{
		if (vacants.empty())
		{
			unsigned num_objects = start_object_count << memory.size();
			T *ptr = static_cast<T *>(malloc(num_objects * sizeof(T)));
			if (!ptr)
				return nullptr;

			vacants.reserve(num_objects);
			for (unsigned i = 0; i < num_objects; i++)
				vacants.push_back(&ptr[i]);

			memory.emplace_back(ptr);
		}

		T *ptr = vacants.back();
		vacants.pop_back();
		new (ptr) T(std::forward<P>(p)...);
		return ptr;
	}

	void deallocate(T *ptr)
	{
		ptr->~T();
		vacants.push_back(ptr);
	}

	void deallocate_opaque(void *ptr) override
	{
		deallocate(static_cast<T *>(ptr));
	}

	void clear()
	{
		vacants.clear();
		memory.clear();
	}

protected:
	Vector<T *> vacants;

	struct MallocDeleter
	{
		void operator()(T *ptr)
		{
			::free(ptr);
		}
	};

	SmallVector<std::unique_ptr<T, MallocDeleter>> memory;
	unsigned start_object_count;
};

template <size_t StackSize = 4096, size_t BlockSize = 4096>
class StringStream
{
public:
	StringStream()
	{
		reset();
	}

	~StringStream()
	{
		reset();
	}

	// Disable copies and moves. Makes it easier to implement, and we don't need it.
	StringStream(const StringStream &) = delete;
	void operator=(const StringStream &) = delete;

	template <typename T, typename std::enable_if<!std::is_floating_point<T>::value, int>::type = 0>
	StringStream &operator<<(const T &t)
	{
		auto s = std::to_string(t);
		append(s.data(), s.size());
		return *this;
	}

	// Only overload this to make float/double conversions ambiguous.
	StringStream &operator<<(uint32_t v)
	{
		auto s = std::to_string(v);
		append(s.data(), s.size());
		return *this;
	}

	StringStream &operator<<(char c)
	{
		append(&c, 1);
		return *this;
	}

	StringStream &operator<<(const std::string &s)
	{
		append(s.data(), s.size());
		return *this;
	}

	StringStream &operator<<(const char *s)
	{
		append(s, strlen(s));
		return *this;
	}

	template <size_t N>
	StringStream &operator<<(const char (&s)[N])
	{
		append(s, strlen(s));
		return *this;
	}

	std::string str() const
	{
		std::string ret;
		size_t target_size = 0;
		for (auto &saved : saved_buffers)
			target_size += saved.offset;
		target_size += current_buffer.offset;
		ret.reserve(target_size);

		for (auto &saved : saved_buffers)
			ret.insert(ret.end(), saved.buffer, saved.buffer + saved.offset);
		ret.insert(ret.end(), current_buffer.buffer, current_buffer.buffer + current_buffer.offset);
		return ret;
	}

	void reset()
	{
		for (auto &saved : saved_buffers)
			if (saved.buffer != stack_buffer)
				free(saved.buffer);
		if (current_buffer.buffer != stack_buffer)
			free(current_buffer.buffer);

		saved_buffers.clear();
		current_buffer.buffer = stack_buffer;
		current_buffer.offset = 0;
		current_buffer.size = sizeof(stack_buffer);
	}

private:
	struct Buffer
	{
		char *buffer = nullptr;
		size_t offset = 0;
		size_t size = 0;
	};
	Buffer current_buffer;
	char stack_buffer[StackSize];
	SmallVector<Buffer> saved_buffers;

	void append(const char *s, size_t len)
	{
		size_t avail = current_buffer.size - current_buffer.offset;
		if (avail < len)
		{
			if (avail > 0)
			{
				memcpy(current_buffer.buffer + current_buffer.offset, s, avail);
				s += avail;
				len -= avail;
				current_buffer.offset += avail;
			}

			saved_buffers.push_back(current_buffer);
			size_t target_size = len > BlockSize ? len : BlockSize;
			current_buffer.buffer = static_cast<char *>(malloc(target_size));
			if (!current_buffer.buffer)
				SPIRV_CROSS_THROW("Out of memory.");

			memcpy(current_buffer.buffer, s, len);
			current_buffer.offset = len;
			current_buffer.size = target_size;
		}
		else
		{
			memcpy(current_buffer.buffer + current_buffer.offset, s, len);
			current_buffer.offset += len;
		}
	}
};

} // namespace SPIRV_CROSS_NAMESPACE

#endif
