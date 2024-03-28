// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

/// \ingroup Types
/// \class Section
/// \brief A struct storing data for any section in a .PSD file.
struct Section
{
	uint64_t offset;				///< The offset from the start of the file where this section is stored.
	uint32_t length;				///< The length of the section.
};

PSD_NAMESPACE_END
