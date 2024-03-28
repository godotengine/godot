// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once

#include "PsdAllocator.h"


PSD_NAMESPACE_BEGIN

/// \ingroup Allocators
/// \brief Simple allocator implementation that uses malloc and free internally.
/// \sa Allocator
class MallocAllocator : public Allocator
{
private:
	virtual void* DoAllocate(size_t size, size_t alignment) PSD_OVERRIDE;
	virtual void DoFree(void* ptr) PSD_OVERRIDE;
};

PSD_NAMESPACE_END
