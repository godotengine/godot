// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

struct Document;
class File;
class Allocator;
struct ColorModeDataSection;


/// \ingroup Parser
/// Parses the color mode data section in the document, and returns a newly created instance that needs to be freed
/// by a call to \ref DestroyColorModeDataSection.
/// \remark This function does not yet parse meaningful data.
ColorModeDataSection* ParseColorModeDataSection(const Document& document, File* file, Allocator* allocator);

/// \ingroup Parser
/// Destroys and nullifies the given \a section previously created by a call to \ref ParseColorModeDataSection.
void DestroyColorModeDataSection(ColorModeDataSection*& section, Allocator* allocator);

PSD_NAMESPACE_END
