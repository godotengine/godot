// Copyright 2013 Google Inc. All rights reserved.
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
//     * Neither the name of Google Inc. nor the names of its
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

// This contains a suite of tools for transforming symbol information when
// when that information has been extracted from a PDB containing OMAP
// information.

// OMAP information is a lightweight description of a mapping between two
// address spaces. It consists of two streams, each of them a vector 2-tuples.
// The OMAPTO stream contains tuples of the form
//
//   (RVA in transformed image, RVA in original image)
//
// while the OMAPFROM stream contains tuples of the form
//
//   (RVA in original image, RVA in transformed image)
//
// The entries in each vector are sorted by the first value of the tuple, and
// the lengths associated with a mapping are implicit as the distance between
// two successive addresses in the vector.

// Consider a trivial 10-byte function described by the following symbol:
//
//   Function: RVA 0x00001000, length 10, "foo"
//
// Now consider the same function, but with 5-bytes of instrumentation injected
// at offset 5. The OMAP streams describing this would look like:
//
//   OMAPTO  :  [ [0x00001000, 0x00001000],
//                [0x00001005, 0xFFFFFFFF],
//                [0x0000100a, 0x00001005] ]
//   OMAPFROM:  [ [0x00001000, 0x00001000],
//                [0x00001005, 0x0000100a] ]
//
// In this case the injected code has been marked as not originating in the
// source image, and thus it will have no symbol information at all. However,
// the injected code may also be associated with an original address range;
// for example, when prepending instrumentation to a basic block the
// instrumentation can be labelled as originating from the same source BB such
// that symbol resolution will still find the appropriate source code line
// number. In this case the OMAP stream would look like:
//
//   OMAPTO  :  [ [0x00001000, 0x00001000],
//                [0x00001005, 0x00001005],
//                [0x0000100a, 0x00001005] ]
//   OMAPFROM:  [ [0x00001000, 0x00001000],
//                [0x00001005, 0x0000100a] ]
//
// Suppose we asked DIA to lookup the symbol at location 0x0000100a of the
// instrumented image. It would first run this through the OMAPTO table and
// translate that address to 0x00001005. It would then lookup the symbol
// at that address and return the symbol for the function "foo". This is the
// correct result.
//
// However, if we query DIA for the length and address of the symbol it will
// tell us that it has length 10 and is at RVA 0x00001000. The location is
// correct, but the length doesn't take into account the 5-bytes of injected
// code. Symbol resolution works (starting from an instrumented address,
// mapping to an original address, and looking up a symbol), but the symbol
// metadata is incorrect.
//
// If we dump the symbols using DIA they will have their addresses
// appropriately transformed and reflect positions in the instrumented image.
// However, if we try to do a lookup using those symbols resolution can fail.
// For example, the address 0x0000100a will not map to the symbol for "foo",
// because DIA tells us it is at location 0x00001000 (correct) with length
// 10 (incorrect). The problem is one of order of operations: in this case
// we're attempting symbol resolution by looking up an instrumented address
// in the table of translated symbols.
//
// One way to handle this is to dump the OMAP information as part of the
// breakpad symbols. This requires the rest of the toolchain to be aware of
// OMAP information and to use it when present prior to performing lookup. The
// other option is to properly transform the symbols (updating length as well as
// position) so that resolution will work as expected for translated addresses.
// This is transparent to the rest of the toolchain.

#include "common/windows/omap.h"

#include <atlbase.h>

#include <algorithm>
#include <cassert>
#include <set>

#include "common/windows/dia_util.h"

namespace google_breakpad {

namespace {

static const wchar_t kOmapToDebugStreamName[] = L"OMAPTO";
static const wchar_t kOmapFromDebugStreamName[] = L"OMAPFROM";

// Dependending on where this is used in breakpad we sometimes get min/max from
// windef, and other times from algorithm. To get around this we simply
// define our own min/max functions.
template<typename T>
const T& Min(const T& t1, const T& t2) { return t1 < t2 ? t1 : t2; }
template<typename T>
const T& Max(const T& t1, const T& t2) { return t1 > t2 ? t1 : t2; }

// It makes things more readable to have two different OMAP types. We cast
// normal OMAPs into these. They must be the same size as the OMAP structure
// for this to work, hence the static asserts.
struct OmapOrigToTran {
  DWORD rva_original;
  DWORD rva_transformed;
};
struct OmapTranToOrig {
  DWORD rva_transformed;
  DWORD rva_original;
};
static_assert(sizeof(OmapOrigToTran) == sizeof(OMAP),
              "OmapOrigToTran must have same size as OMAP.");
static_assert(sizeof(OmapTranToOrig) == sizeof(OMAP),
              "OmapTranToOrig must have same size as OMAP.");
typedef std::vector<OmapOrigToTran> OmapFromTable;
typedef std::vector<OmapTranToOrig> OmapToTable;

// Used for sorting and searching through a Mapping.
bool MappedRangeOriginalLess(const MappedRange& lhs, const MappedRange& rhs) {
  if (lhs.rva_original < rhs.rva_original)
    return true;
  if (lhs.rva_original > rhs.rva_original)
    return false;
  return lhs.length < rhs.length;
}
bool MappedRangeMappedLess(const MappedRange& lhs, const MappedRange& rhs) {
  if (lhs.rva_transformed < rhs.rva_transformed)
    return true;
  if (lhs.rva_transformed > rhs.rva_transformed)
    return false;
  return lhs.length < rhs.length;
}

// Used for searching through the EndpointIndexMap.
bool EndpointIndexLess(const EndpointIndex& ei1, const EndpointIndex& ei2) {
  return ei1.endpoint < ei2.endpoint;
}

// Finds the debug stream with the given |name| in the given |session|, and
// populates |table| with its contents. Casts the data directly into OMAP
// structs.
bool FindAndLoadOmapTable(const wchar_t* name,
                          IDiaSession* session,
                          OmapTable* table) {
  assert(name != NULL);
  assert(session != NULL);
  assert(table != NULL);

  CComPtr<IDiaEnumDebugStreamData> stream;
  if (!FindDebugStream(name, session, &stream))
    return false;
  assert(stream.p != NULL);

  LONG count = 0;
  if (FAILED(stream->get_Count(&count))) {
    fprintf(stderr, "IDiaEnumDebugStreamData::get_Count failed for stream "
                    "\"%ws\"\n", name);
    return false;
  }

  // Get the length of the stream in bytes.
  DWORD bytes_read = 0;
  ULONG count_read = 0;
  if (FAILED(stream->Next(count, 0, &bytes_read, NULL, &count_read))) {
    fprintf(stderr, "IDiaEnumDebugStreamData::Next failed while reading "
                    "length of stream \"%ws\"\n", name);
    return false;
  }

  // Ensure it's consistent with the OMAP data type.
  DWORD bytes_expected = count * sizeof(OmapTable::value_type);
  if (count * sizeof(OmapTable::value_type) != bytes_read) {
    fprintf(stderr, "DIA debug stream \"%ws\" has an unexpected length", name);
    return false;
  }

  // Read the table.
  table->resize(count);
  bytes_read = 0;
  count_read = 0;
  if (FAILED(stream->Next(count, bytes_expected, &bytes_read,
                          reinterpret_cast<BYTE*>(&table->at(0)),
                          &count_read))) {
    fprintf(stderr, "IDiaEnumDebugStreamData::Next failed while reading "
                    "data from stream \"%ws\"\n", name);
    return false;
  }

  return true;
}

// This determines the original image length by looking through the segment
// table.
bool GetOriginalImageLength(IDiaSession* session, DWORD* image_length) {
  assert(session != NULL);
  assert(image_length != NULL);

  CComPtr<IDiaEnumSegments> enum_segments;
  if (!FindTable(session, &enum_segments))
    return false;
  assert(enum_segments.p != NULL);

  DWORD temp_image_length = 0;
  CComPtr<IDiaSegment> segment;
  ULONG fetched = 0;
  while (SUCCEEDED(enum_segments->Next(1, &segment, &fetched)) &&
         fetched == 1) {
    assert(segment.p != NULL);

    DWORD rva = 0;
    DWORD length = 0;
    DWORD frame = 0;
    if (FAILED(segment->get_relativeVirtualAddress(&rva)) ||
        FAILED(segment->get_length(&length)) ||
        FAILED(segment->get_frame(&frame))) {
      fprintf(stderr, "Failed to get basic properties for IDiaSegment\n");
      return false;
    }

    if (frame > 0) {
      DWORD segment_end = rva + length;
      if (segment_end > temp_image_length)
        temp_image_length = segment_end;
    }
    segment.Release();
  }

  *image_length = temp_image_length;
  return true;
}

// Detects regions of the original image that have been removed in the
// transformed image, and sets the 'removed' property on all mapped ranges
// immediately preceding a gap. The mapped ranges must be sorted by
// 'rva_original'.
void FillInRemovedLengths(Mapping* mapping) {
  assert(mapping != NULL);

  // Find and fill gaps. We do this with two sweeps. We first sweep forward
  // looking for gaps. When we identify a gap we then sweep forward with a
  // second scan and set the 'removed' property for any intervals that
  // immediately precede the gap.
  //
  // Gaps are typically between two successive intervals, but not always:
  //
  //   Range 1: ---------------
  //   Range 2:     -------
  //   Range 3:                      -------------
  //   Gap    :                ******
  //
  // In the above example the gap is between range 1 and range 3. A forward
  // sweep finds the gap, and a second forward sweep identifies that range 1
  // immediately precedes the gap and sets its 'removed' property.

  size_t fill = 0;
  DWORD rva_front = 0;
  for (size_t find = 0; find < mapping->size(); ++find) {
#ifndef NDEBUG
    // We expect the mapped ranges to be sorted by 'rva_original'.
    if (find > 0) {
      assert(mapping->at(find - 1).rva_original <=
                 mapping->at(find).rva_original);
    }
#endif

    if (rva_front < mapping->at(find).rva_original) {
      // We've found a gap. Fill it in by setting the 'removed' property for
      // any affected intervals.
      DWORD removed = mapping->at(find).rva_original - rva_front;
      for (; fill < find; ++fill) {
        if (mapping->at(fill).rva_original + mapping->at(fill).length !=
                rva_front) {
          continue;
        }

        // This interval ends right where the gap starts. It needs to have its
        // 'removed' information filled in.
        mapping->at(fill).removed = removed;
      }
    }

    // Advance the front that indicates the covered portion of the image.
    rva_front = mapping->at(find).rva_original + mapping->at(find).length;
  }
}

// Builds a unified view of the mapping between the original and transformed
// image space by merging OMAPTO and OMAPFROM data.
void BuildMapping(const OmapData& omap_data, Mapping* mapping) {
  assert(mapping != NULL);

  mapping->clear();

  if (omap_data.omap_from.empty() || omap_data.omap_to.empty())
    return;

  // The names 'omap_to' and 'omap_from' are awfully confusing, so we make
  // ourselves more explicit here. This cast is only safe because the underlying
  // types have the exact same size.
  const OmapToTable& tran2orig =
      reinterpret_cast<const OmapToTable&>(omap_data.omap_to);
  const OmapFromTable& orig2tran = reinterpret_cast<const OmapFromTable&>(
      omap_data.omap_from);

  // Handle the range of data at the beginning of the image. This is not usually
  // specified by the OMAP data.
  if (tran2orig[0].rva_transformed > 0 && orig2tran[0].rva_original > 0) {
    DWORD header_transformed = tran2orig[0].rva_transformed;
    DWORD header_original = orig2tran[0].rva_original;
    DWORD header = Min(header_transformed, header_original);

    MappedRange mr = {};
    mr.length = header;
    mr.injected = header_transformed - header;
    mr.removed = header_original - header;
    mapping->push_back(mr);
  }

  // Convert the implied lengths to explicit lengths, and infer which content
  // has been injected into the transformed image. Injected content is inferred
  // as regions of the transformed address space that does not map back to
  // known valid content in the original image.
  for (size_t i = 0; i < tran2orig.size(); ++i) {
    const OmapTranToOrig& o1 = tran2orig[i];

    // This maps to content that is outside the original image, thus it
    // describes injected content. We can skip this entry.
    if (o1.rva_original >= omap_data.length_original)
      continue;

    // Calculate the length of the current OMAP entry. This is implicit as the
    // distance between successive |rva| values, capped at the end of the
    // original image.
    DWORD length = 0;
    if (i + 1 < tran2orig.size()) {
      const OmapTranToOrig& o2 = tran2orig[i + 1];

      // We expect the table to be sorted by rva_transformed.
      assert(o1.rva_transformed <= o2.rva_transformed);

      length = o2.rva_transformed - o1.rva_transformed;
      if (o1.rva_original + length > omap_data.length_original) {
        length = omap_data.length_original - o1.rva_original;
      }
    } else {
      length = omap_data.length_original - o1.rva_original;
    }

    // Zero-length entries don't describe anything and can be ignored.
    if (length == 0)
      continue;

    // Any gaps in the transformed address-space are due to injected content.
    if (!mapping->empty()) {
      MappedRange& prev_mr = mapping->back();
      prev_mr.injected += o1.rva_transformed -
          (prev_mr.rva_transformed + prev_mr.length);
    }

    MappedRange mr = {};
    mr.rva_original = o1.rva_original;
    mr.rva_transformed = o1.rva_transformed;
    mr.length = length;
    mapping->push_back(mr);
  }

  // Sort based on the original image addresses.
  std::sort(mapping->begin(), mapping->end(), MappedRangeOriginalLess);

  // Fill in the 'removed' lengths by looking for gaps in the coverage of the
  // original address space.
  FillInRemovedLengths(mapping);

  return;
}

void BuildEndpointIndexMap(ImageMap* image_map) {
  assert(image_map != NULL);

  if (image_map->mapping.size() == 0)
    return;

  const Mapping& mapping = image_map->mapping;
  EndpointIndexMap& eim = image_map->endpoint_index_map;

  // Get the unique set of interval endpoints.
  std::set<DWORD> endpoints;
  for (size_t i = 0; i < mapping.size(); ++i) {
    endpoints.insert(mapping[i].rva_original);
    endpoints.insert(mapping[i].rva_original +
                         mapping[i].length +
                         mapping[i].removed);
  }

  // Use the endpoints to initialize the secondary search structure for the
  // mapping.
  eim.resize(endpoints.size());
  std::set<DWORD>::const_iterator it = endpoints.begin();
  for (size_t i = 0; it != endpoints.end(); ++it, ++i) {
    eim[i].endpoint = *it;
    eim[i].index = mapping.size();
  }

  // For each endpoint we want the smallest index of any interval containing
  // it. We iterate over the intervals and update the indices associated with
  // each interval endpoint contained in the current interval. In the general
  // case of an arbitrary set of intervals this is O(n^2), but the structure of
  // OMAP data makes this O(n).
  for (size_t i = 0; i < mapping.size(); ++i) {
    EndpointIndex ei1 = { mapping[i].rva_original, 0 };
    EndpointIndexMap::iterator it1 = std::lower_bound(
        eim.begin(), eim.end(), ei1, EndpointIndexLess);

    EndpointIndex ei2 = { mapping[i].rva_original + mapping[i].length +
                              mapping[i].removed, 0 };
    EndpointIndexMap::iterator it2 = std::lower_bound(
        eim.begin(), eim.end(), ei2, EndpointIndexLess);

    for (; it1 != it2; ++it1)
      it1->index = Min(i, it1->index);
  }
}

void BuildSubsequentRVAMap(const OmapData& omap_data,
                           std::map<DWORD, DWORD>* subsequent) {
  assert(subsequent->empty());
  const OmapFromTable& orig2tran =
      reinterpret_cast<const OmapFromTable&>(omap_data.omap_from);

  if (orig2tran.empty())
    return;

  for (size_t i = 0; i < orig2tran.size() - 1; ++i) {
    // Expect that orig2tran is sorted.
    if (orig2tran[i].rva_original >= orig2tran[i + 1].rva_original) {
      fprintf(stderr, "OMAP 'from' table unexpectedly unsorted\n");
      subsequent->clear();
      return;
    }
    subsequent->insert(std::make_pair(orig2tran[i].rva_original,
                                      orig2tran[i + 1].rva_original));
  }
}

// Clips the given mapped range.
void ClipMappedRangeOriginal(const AddressRange& clip_range,
                             MappedRange* mapped_range) {
  assert(mapped_range != NULL);

  // The clipping range is entirely outside of the mapped range.
  if (clip_range.end() <= mapped_range->rva_original ||
      mapped_range->rva_original + mapped_range->length +
          mapped_range->removed <= clip_range.rva) {
    mapped_range->length = 0;
    mapped_range->injected = 0;
    mapped_range->removed = 0;
    return;
  }

  // Clip the left side.
  if (mapped_range->rva_original < clip_range.rva) {
    DWORD clip_left = clip_range.rva - mapped_range->rva_original;
    mapped_range->rva_original += clip_left;
    mapped_range->rva_transformed += clip_left;

    if (clip_left > mapped_range->length) {
      // The left clipping boundary entirely erases the content section of the
      // range.
      DWORD trim = clip_left - mapped_range->length;
      mapped_range->length = 0;
      mapped_range->injected -= Min(trim, mapped_range->injected);
      // We know that trim <= mapped_range->remove.
      mapped_range->removed -= trim;
    } else {
      // The left clipping boundary removes some, but not all, of the content.
      // As such it leaves the removed/injected component intact.
      mapped_range->length -= clip_left;
    }
  }

  // Clip the right side.
  DWORD end_original = mapped_range->rva_original + mapped_range->length;
  if (clip_range.end() < end_original) {
    // The right clipping boundary lands in the 'content' section of the range,
    // entirely clearing the injected/removed portion.
    DWORD clip_right = end_original - clip_range.end();
    mapped_range->length -= clip_right;
    mapped_range->injected = 0;
    mapped_range->removed = 0;
    return;
  } else {
    // The right clipping boundary is outside of the content, but may affect
    // the removed/injected portion of the range.
    DWORD end_removed = end_original + mapped_range->removed;
    if (clip_range.end() < end_removed)
      mapped_range->removed = clip_range.end() - end_original;

    DWORD end_injected = end_original + mapped_range->injected;
    if (clip_range.end() < end_injected)
      mapped_range->injected = clip_range.end() - end_original;
  }

  return;
}

}  // namespace

int AddressRange::Compare(const AddressRange& rhs) const {
  if (end() <= rhs.rva)
    return -1;
  if (rhs.end() <= rva)
    return 1;
  return 0;
}

bool GetOmapDataAndDisableTranslation(IDiaSession* session,
                                      OmapData* omap_data) {
  assert(session != NULL);
  assert(omap_data != NULL);

  CComPtr<IDiaAddressMap> address_map;
  if (FAILED(session->QueryInterface(&address_map))) {
    fprintf(stderr, "IDiaSession::QueryInterface(IDiaAddressMap) failed\n");
    return false;
  }
  assert(address_map.p != NULL);

  BOOL omap_enabled = FALSE;
  if (FAILED(address_map->get_addressMapEnabled(&omap_enabled))) {
    fprintf(stderr, "IDiaAddressMap::get_addressMapEnabled failed\n");
    return false;
  }

  if (!omap_enabled) {
    // We indicate the non-presence of OMAP data by returning empty tables.
    omap_data->omap_from.clear();
    omap_data->omap_to.clear();
    omap_data->length_original = 0;
    return true;
  }

  // OMAP data is present. Disable translation.
  if (FAILED(address_map->put_addressMapEnabled(FALSE))) {
    fprintf(stderr, "IDiaAddressMap::put_addressMapEnabled failed\n");
    return false;
  }

  // Read the OMAP streams.
  if (!FindAndLoadOmapTable(kOmapFromDebugStreamName,
                            session,
                            &omap_data->omap_from)) {
    return false;
  }
  if (!FindAndLoadOmapTable(kOmapToDebugStreamName,
                            session,
                            &omap_data->omap_to)) {
    return false;
  }

  // Get the lengths of the address spaces.
  if (!GetOriginalImageLength(session, &omap_data->length_original))
    return false;

  return true;
}

void BuildImageMap(const OmapData& omap_data, ImageMap* image_map) {
  assert(image_map != NULL);

  BuildMapping(omap_data, &image_map->mapping);
  BuildEndpointIndexMap(image_map);
  BuildSubsequentRVAMap(omap_data, &image_map->subsequent_rva_block);
}

void MapAddressRange(const ImageMap& image_map,
                     const AddressRange& original_range,
                     AddressRangeVector* mapped_ranges) {
  assert(mapped_ranges != NULL);

  const Mapping& map = image_map.mapping;

  // Handle the trivial case of an empty image_map. This means that there is
  // no transformation to be applied, and we can simply return the original
  // range.
  if (map.empty()) {
    mapped_ranges->push_back(original_range);
    return;
  }

  // If we get a query of length 0 we need to handle it by using a non-zero
  // query length.
  AddressRange query_range(original_range);
  if (query_range.length == 0)
    query_range.length = 1;

  // Find the range of intervals that can potentially intersect our query range.
  size_t imin = 0;
  size_t imax = 0;
  {
    // The index of the earliest possible range that can affect is us done by
    // searching through the secondary indexing structure.
    const EndpointIndexMap& eim = image_map.endpoint_index_map;
    EndpointIndex q1 = { query_range.rva, 0 };
    EndpointIndexMap::const_iterator it1 = std::lower_bound(
        eim.begin(), eim.end(), q1, EndpointIndexLess);
    if (it1 == eim.end()) {
      imin  = map.size();
    } else {
      // Backup to find the interval that contains our query point.
      if (it1 != eim.begin() && query_range.rva < it1->endpoint)
        --it1;
      imin = it1->index;
    }

    // The first range that can't possibly intersect us is found by searching
    // through the image map directly as it is already sorted by interval start
    // point.
    MappedRange q2 = { query_range.end(), 0 };
    Mapping::const_iterator it2 = std::lower_bound(
        map.begin(), map.end(), q2, MappedRangeOriginalLess);
    imax = it2 - map.begin();
  }

  // Find all intervals that intersect the query range.
  Mapping temp_map;
  for (size_t i = imin; i < imax; ++i) {
    MappedRange mr = map[i];
    ClipMappedRangeOriginal(query_range, &mr);
    if (mr.length + mr.injected > 0)
      temp_map.push_back(mr);
  }

  // If there are no intersecting ranges then the query range has been removed
  // from the image in question.
  if (temp_map.empty())
    return;

  // Sort based on transformed addresses.
  std::sort(temp_map.begin(), temp_map.end(), MappedRangeMappedLess);

  // Zero-length queries can't actually be merged. We simply output the set of
  // unique RVAs that correspond to the query RVA.
  if (original_range.length == 0) {
    mapped_ranges->push_back(AddressRange(temp_map[0].rva_transformed, 0));
    for (size_t i = 1; i < temp_map.size(); ++i) {
      if (temp_map[i].rva_transformed > mapped_ranges->back().rva)
        mapped_ranges->push_back(AddressRange(temp_map[i].rva_transformed, 0));
    }
    return;
  }

  // Merge any ranges that are consecutive in the mapped image. We merge over
  // injected content if it makes ranges contiguous, but we ignore any injected
  // content at the tail end of a range. This allows us to detect symbols that
  // have been lengthened by injecting content in the middle. However, it
  // misses the case where content has been injected at the head or the tail.
  // The problem is that it doesn't know whether to attribute it to the
  // preceding or following symbol. It is up to the author of the transform to
  // output explicit OMAP info in these cases to ensure full coverage of the
  // transformed address space.
  DWORD rva_begin = temp_map[0].rva_transformed;
  DWORD rva_cur_content = rva_begin + temp_map[0].length;
  DWORD rva_cur_injected = rva_cur_content + temp_map[0].injected;
  for (size_t i = 1; i < temp_map.size(); ++i) {
    if (rva_cur_injected < temp_map[i].rva_transformed) {
      // This marks the end of a continuous range in the image. Output the
      // current range and start a new one.
      if (rva_begin < rva_cur_content) {
        mapped_ranges->push_back(
            AddressRange(rva_begin, rva_cur_content - rva_begin));
      }
      rva_begin = temp_map[i].rva_transformed;
    }

    rva_cur_content = temp_map[i].rva_transformed + temp_map[i].length;
    rva_cur_injected = rva_cur_content + temp_map[i].injected;
  }

  // Output the range in progress.
  if (rva_begin < rva_cur_content) {
    mapped_ranges->push_back(
        AddressRange(rva_begin, rva_cur_content - rva_begin));
  }

  return;
}

}  // namespace google_breakpad
