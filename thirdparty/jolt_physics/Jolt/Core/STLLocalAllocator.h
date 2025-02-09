// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2025 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/Core/STLAllocator.h>

JPH_NAMESPACE_BEGIN

#ifndef JPH_DISABLE_CUSTOM_ALLOCATOR

/// STL allocator that keeps N elements in a local buffer before falling back to regular allocations
template <typename T, size_t N>
class STLLocalAllocator : private STLAllocator<T>
{
	using Base = STLAllocator<T>;

public:
	/// General properties
	using value_type = T;
	using pointer = T *;
	using const_pointer = const T *;
	using reference = T &;
	using const_reference = const T &;
	using size_type = size_t;
	using difference_type = ptrdiff_t;

	/// The allocator is not stateless (has local buffer)
	using is_always_equal = std::false_type;

	/// We cannot copy, move or swap allocators
	using propagate_on_container_copy_assignment = std::false_type;
	using propagate_on_container_move_assignment = std::false_type;
	using propagate_on_container_swap = std::false_type;

	/// Constructor
							STLLocalAllocator() = default;
							STLLocalAllocator(const STLLocalAllocator &) = delete; // Can't copy an allocator as the buffer is local to the original
							STLLocalAllocator(STLLocalAllocator &&) = delete; // Can't move an allocator as the buffer is local to the original
	STLLocalAllocator &		operator = (const STLLocalAllocator &) = delete; // Can't copy an allocator as the buffer is local to the original

	/// Constructor used when rebinding to another type. This expects the allocator to use the original memory pool from the first allocator,
	/// but in our case we cannot use the local buffer of the original allocator as it has different size and alignment rules.
	/// To solve this we make this allocator fall back to the heap immediately.
	template <class T2>		STLLocalAllocator(const STLLocalAllocator<T2, N> &) : mNumElementsUsed(N) { }

	/// Check if inPointer is in the local buffer
	inline bool				is_local(const_pointer inPointer) const
	{
		ptrdiff_t diff = inPointer - reinterpret_cast<const_pointer>(mElements);
		return diff >= 0 && diff < ptrdiff_t(N);
	}

	/// Allocate memory
	inline pointer			allocate(size_type inN)
	{
		// If we allocate more than we have, fall back to the heap
		if (mNumElementsUsed + inN > N)
			return Base::allocate(inN);

		// Allocate from our local buffer
		pointer result = reinterpret_cast<pointer>(mElements) + mNumElementsUsed;
		mNumElementsUsed += inN;
		return result;
	}

	/// Always implements a reallocate function as we can often reallocate in place
	static constexpr bool	has_reallocate = true;

	/// Reallocate memory
	inline pointer			reallocate(pointer inOldPointer, size_type inOldSize, size_type inNewSize)
	{
		JPH_ASSERT(inNewSize > 0); // Reallocating to zero size is implementation dependent, so we don't allow it

		// If there was no previous allocation, we can go through the regular allocate function
		if (inOldPointer == nullptr)
			return allocate(inNewSize);

		// If the pointer is outside our local buffer, fall back to the heap
		if (!is_local(inOldPointer))
		{
			if constexpr (AllocatorHasReallocate<Base>::sValue)
				return Base::reallocate(inOldPointer, inOldSize, inNewSize);
			else
				return ReallocateImpl(inOldPointer, inOldSize, inNewSize);
		}

		// If we happen to have space left, we only need to update our bookkeeping
		pointer base_ptr = reinterpret_cast<pointer>(mElements) + mNumElementsUsed - inOldSize;
		if (inOldPointer == base_ptr
			&& mNumElementsUsed - inOldSize + inNewSize <= N)
		{
			mNumElementsUsed += inNewSize - inOldSize;
			return base_ptr;
		}

		// We can't reallocate in place, fall back to the heap
		return ReallocateImpl(inOldPointer, inOldSize, inNewSize);
	}

	/// Free memory
	inline void				deallocate(pointer inPointer, size_type inN)
	{
		// If the pointer is not in our local buffer, fall back to the heap
		if (!is_local(inPointer))
			return Base::deallocate(inPointer, inN);

		// Else we can only reclaim memory if it was the last allocation
		if (inPointer == reinterpret_cast<pointer>(mElements) + mNumElementsUsed - inN)
			mNumElementsUsed -= inN;
	}

	/// Allocators are not-stateless, assume if allocator address matches that the allocators are the same
	inline bool				operator == (const STLLocalAllocator<T, N> &inRHS) const
	{
		return this == &inRHS;
	}

	inline bool				operator != (const STLLocalAllocator<T, N> &inRHS) const
	{
		return this != &inRHS;
	}

	/// Converting to allocator for other type
	template <typename T2>
	struct rebind
	{
		using other = STLLocalAllocator<T2, N>;
	};

private:
	/// Implements reallocate when the base class doesn't or when we go from local buffer to heap
	inline pointer			ReallocateImpl(pointer inOldPointer, size_type inOldSize, size_type inNewSize)
	{
		pointer new_pointer = Base::allocate(inNewSize);
		size_type n = min(inOldSize, inNewSize);
		if constexpr (std::is_trivially_copyable<T>())
		{
			// Can use mem copy
			memcpy(new_pointer, inOldPointer, n * sizeof(T));
		}
		else
		{
			// Need to actually move the elements
			for (size_t i = 0; i < n; ++i)
			{
				new (new_pointer + i) T(std::move(inOldPointer[i]));
				inOldPointer[i].~T();
			}
		}
		deallocate(inOldPointer, inOldSize);
		return new_pointer;
	}

	alignas(T) uint8		mElements[N * sizeof(T)];
	size_type				mNumElementsUsed = 0;
};

/// The STLLocalAllocator always implements a reallocate function as it can often reallocate in place
template <class T, size_t N> struct AllocatorHasReallocate<STLLocalAllocator<T, N>> { static constexpr bool sValue = STLLocalAllocator<T, N>::has_reallocate; };

#else

template <typename T, size_t N> using STLLocalAllocator = std::allocator<T>;

#endif // !JPH_DISABLE_CUSTOM_ALLOCATOR

JPH_NAMESPACE_END
