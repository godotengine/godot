//
// Copyright (c) 2009-2010 Mikko Mononen memon@inside.org
//
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would be
//    appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.
//

#ifndef RECASTALLOC_H
#define RECASTALLOC_H

#include <stddef.h>

/// Provides hint values to the memory allocator on how long the
/// memory is expected to be used.
enum rcAllocHint
{
	RC_ALLOC_PERM,		///< Memory will persist after a function call.
	RC_ALLOC_TEMP		///< Memory used temporarily within a function.
};

/// A memory allocation function.
//  @param[in]		size			The size, in bytes of memory, to allocate.
//  @param[in]		rcAllocHint	A hint to the allocator on how long the memory is expected to be in use.
//  @return A pointer to the beginning of the allocated memory block, or null if the allocation failed.
///  @see rcAllocSetCustom
typedef void* (rcAllocFunc)(size_t size, rcAllocHint hint);

/// A memory deallocation function.
///  @param[in]		ptr		A pointer to a memory block previously allocated using #rcAllocFunc.
/// @see rcAllocSetCustom
typedef void (rcFreeFunc)(void* ptr);

/// Sets the base custom allocation functions to be used by Recast.
///  @param[in]		allocFunc	The memory allocation function to be used by #rcAlloc
///  @param[in]		freeFunc	The memory de-allocation function to be used by #rcFree
void rcAllocSetCustom(rcAllocFunc *allocFunc, rcFreeFunc *freeFunc);

/// Allocates a memory block.
///  @param[in]		size	The size, in bytes of memory, to allocate.
///  @param[in]		hint	A hint to the allocator on how long the memory is expected to be in use.
///  @return A pointer to the beginning of the allocated memory block, or null if the allocation failed.
/// @see rcFree
void* rcAlloc(size_t size, rcAllocHint hint);

/// Deallocates a memory block.
///  @param[in]		ptr		A pointer to a memory block previously allocated using #rcAlloc.
/// @see rcAlloc
void rcFree(void* ptr);


/// A simple dynamic array of integers.
class rcIntArray
{
	int* m_data;
	int m_size, m_cap;

	void doResize(int n);
	
	// Explicitly disabled copy constructor and copy assignment operator.
	rcIntArray(const rcIntArray&);
	rcIntArray& operator=(const rcIntArray&);

public:
	/// Constructs an instance with an initial array size of zero.
	rcIntArray() : m_data(0), m_size(0), m_cap(0) {}

	/// Constructs an instance initialized to the specified size.
	///  @param[in]		n	The initial size of the integer array.
	rcIntArray(int n) : m_data(0), m_size(0), m_cap(0) { resize(n); }
	~rcIntArray() { rcFree(m_data); }

	/// Specifies the new size of the integer array.
	///  @param[in]		n	The new size of the integer array.
	void resize(int n)
	{
		if (n > m_cap)
			doResize(n);
		
		m_size = n;
	}

	/// Push the specified integer onto the end of the array and increases the size by one.
	///  @param[in]		item	The new value.
	void push(int item) { resize(m_size+1); m_data[m_size-1] = item; }

	/// Returns the value at the end of the array and reduces the size by one.
	///  @return The value at the end of the array.
	int pop()
	{
		if (m_size > 0)
			m_size--;
		
		return m_data[m_size];
	}

	/// The value at the specified array index.
	/// @warning Does not provide overflow protection.
	///  @param[in]		i	The index of the value.
	const int& operator[](int i) const { return m_data[i]; }

	/// The value at the specified array index.
	/// @warning Does not provide overflow protection.
	///  @param[in]		i	The index of the value.
	int& operator[](int i) { return m_data[i]; }

	/// The current size of the integer array.
	int size() const { return m_size; }
};

/// A simple helper class used to delete an array when it goes out of scope.
/// @note This class is rarely if ever used by the end user.
template<class T> class rcScopedDelete
{
	T* ptr;
public:

	/// Constructs an instance with a null pointer.
	inline rcScopedDelete() : ptr(0) {}

	/// Constructs an instance with the specified pointer.
	///  @param[in]		p	An pointer to an allocated array.
	inline rcScopedDelete(T* p) : ptr(p) {}
	inline ~rcScopedDelete() { rcFree(ptr); }

	/// The root array pointer.
	///  @return The root array pointer.
	inline operator T*() { return ptr; }
	
private:
	// Explicitly disabled copy constructor and copy assignment operator.
	rcScopedDelete(const rcScopedDelete&);
	rcScopedDelete& operator=(const rcScopedDelete&);
};

#endif
