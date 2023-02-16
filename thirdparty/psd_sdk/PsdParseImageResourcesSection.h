// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

struct Document;
class File;
class Allocator;
struct ImageResourcesSection;


/// \ingroup Parser
/// Parses the image resources section in the document, and returns a newly created instance that needs to be freed
/// by a call to \ref DestroyImageResourcesSection.
/// \remark It is valid to parse different sections of a document (e.g. using \ref ParseImageResourcesSection, \ref ParseImageDataSection,
/// or \ref ParseLayerMaskSection) in parallel from different threads.
ImageResourcesSection* ParseImageResourcesSection(const Document* document, File* file, Allocator* allocator);

/// \ingroup Parser
/// Destroys and nullifies the given \a section previously created by a call to \ref ParseImageResourcesSection.
void DestroyImageResourcesSection(ImageResourcesSection*& section, Allocator* allocator);

PSD_NAMESPACE_END
