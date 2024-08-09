// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once

#include "PsdAllocator.h"
#include "Psdispod.h"
#include "PsdAssert.h"


PSD_NAMESPACE_BEGIN

/// \ingroup Util
/// \namespace memoryUtil
/// \brief Provides memory allocation utilities.
namespace memoryUtil
{
	/// Allocates memory for an instance of type T, using T's default alignment.
	template <typename T>
	inline T* Allocate(Allocator* allocator);

	/// Allocates memory for \a count instances of type T, using T's default alignment.
	/// \remark Note that this does not call any constructors, and hence only works for POD types.
	template <typename T>
	inline T* AllocateArray(Allocator* allocator, size_t count);

	/// Frees memory previously allocated with \a allocator, and nullifies \a ptr.
	template <typename T>
	inline void Free(Allocator* allocator, T*& ptr);

	/// Frees an array previously allocated with \a allocator, and nullifies \a ptr.
	/// \remark Note that this does not call any destructors, and hence only works for POD types.
	template <typename T>
	inline void FreeArray(Allocator* allocator, T*& ptr);
}

#include "PsdMemoryUtil.inl"

PSD_NAMESPACE_END
