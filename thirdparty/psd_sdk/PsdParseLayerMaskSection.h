// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

struct Document;
class File;
class Allocator;
struct Layer;
struct LayerMaskSection;


/// \ingroup Parser
/// Parses the layer mask section in the document, and returns a newly created instance that needs to be freed
/// by a call to \ref DestroyLayerMaskSection. This function does not extract layer data yet, that has to be done
/// by a call to \ref ExtractLayer for each layer.
/// \remark It is valid to parse different sections of a document (e.g. using \ref ParseImageResourcesSection, \ref ParseImageDataSection,
/// or \ref ParseLayerMaskSection) in parallel from different threads.
LayerMaskSection* ParseLayerMaskSection(const Document* document, File* file, Allocator* allocator);

/// \ingroup Parser
/// Extracts data for a given \a layer.
/// \remark It is valid and suggested to extract the data of individual layers from multiple threads in parallel.
void ExtractLayer(const Document* document, File* file, Allocator* allocator, Layer* layer);

/// \ingroup Parser
/// Destroys and nullifies the given \a section previously created by a call to \ref ParseLayerMaskSection.
void DestroyLayerMaskSection(LayerMaskSection*& section, Allocator* allocator);

PSD_NAMESPACE_END
