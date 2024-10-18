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

// Declares internal implementation details for functionality in omap.h and
// omap.cc.

#ifndef COMMON_WINDOWS_OMAP_INTERNAL_H_
#define COMMON_WINDOWS_OMAP_INTERNAL_H_

#include <windows.h>
#include <dia2.h>

#include <map>
#include <vector>

namespace google_breakpad {

// The OMAP struct is defined by debughlp.h, which doesn't play nicely with
// imagehlp.h. We simply redefine it.
struct OMAP {
  DWORD rva;
  DWORD rvaTo;
};
static_assert(sizeof(OMAP) == 8, "Wrong size for OMAP structure.");
typedef std::vector<OMAP> OmapTable;

// This contains the OMAP data extracted from an image.
struct OmapData {
  // The table of OMAP entries describing the transformation from the
  // original image to the transformed image.
  OmapTable omap_from;  
  // The table of OMAP entries describing the transformation from the
  // instrumented image to the original image.
  OmapTable omap_to;
  // The length of the original untransformed image.
  DWORD length_original;

  OmapData() : length_original(0) { }
};

// This represents a range of addresses in an image.
struct AddressRange {
  DWORD rva;
  DWORD length;

  AddressRange() : rva(0), length(0) { }
  AddressRange(DWORD rva, DWORD length) : rva(rva), length(length) { }

  // Returns the end address of this range.
  DWORD end() const { return rva + length; }

  // Addreses only compare as less-than or greater-than if they are not
  // overlapping. Otherwise, they compare equal.
  int Compare(const AddressRange& rhs) const;
  bool operator<(const AddressRange& rhs) const { return Compare(rhs) == -1; }
  bool operator>(const AddressRange& rhs) const { return Compare(rhs) == 1; }

  // Equality operators compare exact values.
  bool operator==(const AddressRange& rhs) const {
    return rva == rhs.rva && length == rhs.length;
  }
  bool operator!=(const  AddressRange& rhs) const { return !((*this) == rhs); }
};

typedef std::vector<AddressRange> AddressRangeVector;

// This represents an address range in an original image, and its corresponding
// range in the transformed image.
struct MappedRange {
  // An address in the original image.
  DWORD rva_original;
  // The corresponding addresses in the transformed image.
  DWORD rva_transformed;
  // The length of the address range.
  DWORD length;
  // It is possible for code to be injected into a transformed image, for which
  // there is no corresponding code in the original image. If this range of
  // transformed image is immediately followed by such injected code we maintain
  // a record of its length here.
  DWORD injected;
  // It is possible for code to be removed from the original image. This happens
  // for things like padding between blocks. There is no actual content lost,
  // but the spacing between items may be lost. This keeps track of any removed
  // content immediately following the |original| range.
  DWORD removed;
};
// A vector of mapped ranges is used as a more useful representation of
// OMAP data.
typedef std::vector<MappedRange> Mapping;

// Used as a secondary search structure accompanying a Mapping.
struct EndpointIndex {
  DWORD endpoint;
  size_t index;
};
typedef std::vector<EndpointIndex> EndpointIndexMap;

// An ImageMap is vector of mapped ranges, plus a secondary index into it for
// doing interval searches. (An interval tree would also work, but is overkill
// because we don't need insertion and deletion.)
struct ImageMap {
  // This is a description of the mapping between original and transformed
  // image, sorted by addresses in the original image.
  Mapping mapping;
  // For all interval endpoints in |mapping| this stores the minimum index of
  // an interval in |mapping| that contains the endpoint. Useful for doing
  // interval intersection queries.
  EndpointIndexMap endpoint_index_map;

  std::map<DWORD, DWORD> subsequent_rva_block;
};

}  // namespace google_breakpad

#endif  // COMMON_WINDOWS_OMAP_INTERNAL_H_
