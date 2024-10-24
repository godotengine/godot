// Copyright 2013 Google LLC
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google LLC nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Provides an API for mapping symbols through OMAP information, if a PDB file
// is augmented with it. This allows breakpad to work with addresses in
// transformed images by transforming the symbols themselves, rather than
// transforming addresses prior to querying symbols (the way it is typically
// done by Windows-native tools, including the DIA).

#ifndef COMMON_WINDOWS_OMAP_H_
#define COMMON_WINDOWS_OMAP_H_

#include "common/windows/omap_internal.h"

namespace google_breakpad {

// If the given session contains OMAP data this extracts it, populating
// |omap_data|, and then disabling automatic translation for the session.
// OMAP data is present in the PDB if |omap_data| is not empty. This returns
// true on success, false otherwise.
bool GetOmapDataAndDisableTranslation(IDiaSession* dia_session,
                                      OmapData* omap_data);

// Given raw OMAP data builds an ImageMap. This can be used to query individual
// image ranges using MapAddressRange.
// |omap_data|| is the OMAP data extracted from the PDB.
// |image_map| will be populated with a description of the image mapping. If
//     |omap_data| is empty then this will also be empty.
void BuildImageMap(const OmapData& omap_data, ImageMap* image_map);

// Given an address range in the original image space determines how exactly it
// has been tranformed.
// |omap_data| is the OMAP data extracted from the PDB, which must not be
//     empty.
// |original_range| is the address range in the original image being queried.
// |mapped_ranges| will be populated with a full description of the mapping.
//     They may be disjoint in the transformed image so a vector is needed to
//     fully represent the mapping. This will be appended to if it is not
//     empty. If |omap_data| is empty then |mapped_ranges| will simply be
//     populated with a copy of |original_range| (the identity transform).
void MapAddressRange(const ImageMap& image_map,
                     const AddressRange& original_range,
                     AddressRangeVector* mapped_ranges);

}  // namespace google_breakpad

#endif  // COMMON_WINDOWS_OMAP_H_
