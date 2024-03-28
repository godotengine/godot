// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

/// \ingroup Interfaces
/// \ingroup Allocators
/// \brief Base class for all memory allocators.
/// \details Memory allocators are used throughout the library to ensure full control over all allocations. This allows
/// using custom allocators for better performance, a smaller memory footprint, and for adding extra debugging and/or tracking
/// features.
/// \sa MallocAllocator
class Allocator
{
public:
	/// Empty destructor.
	virtual ~Allocator(void);

	/// Allocates \a size bytes with a given \a alignment. The alignment must be a power-of-two.
	void* Allocate(size_t size, size_t alignment);

	/// Frees an allocation.
	void Free(void* ptr);

private:
	virtual void* DoAllocate(size_t size, size_t alignment) PSD_ABSTRACT;
	virtual void DoFree(void* ptr) PSD_ABSTRACT;
};

PSD_NAMESPACE_END
