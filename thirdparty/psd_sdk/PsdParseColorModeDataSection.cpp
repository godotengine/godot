// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#include "PsdPch.h"
#include "PsdParseColorModeDataSection.h"

#include "PsdColorModeDataSection.h"
#include "PsdAllocator.h"
#include "PsdMemoryUtil.h"


PSD_NAMESPACE_BEGIN

// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
ColorModeDataSection* ParseColorModeDataSection(const Document&, File*, Allocator*)
{
	// not implemented yet.
	// only indexed color and Duotone have color mode data, but both are not supported by this library at the moment.
	return nullptr;
}


// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
void DestroyColorModeDataSection(ColorModeDataSection*& section, Allocator* allocator)
{
	PSD_ASSERT_NOT_NULL(section);
	PSD_ASSERT_NOT_NULL(allocator);

	memoryUtil::Free(allocator, section);
}

PSD_NAMESPACE_END
