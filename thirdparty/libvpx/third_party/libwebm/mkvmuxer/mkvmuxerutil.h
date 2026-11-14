// Copyright (c) 2012 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef MKVMUXER_MKVMUXERUTIL_H_
#define MKVMUXER_MKVMUXERUTIL_H_

#include <stdint.h>

#include "mkvmuxertypes.h"

namespace mkvmuxer {
class Cluster;
class Frame;
class IMkvWriter;

// TODO(tomfinegan): mkvmuxer:: integer types continue to be used here because
// changing them causes pain for downstream projects. It would be nice if a
// solution that allows removal of the mkvmuxer:: integer types while avoiding
// pain for downstream users of libwebm. Considering that mkvmuxerutil.{cc,h}
// are really, for the great majority of cases, EBML size calculation and writer
// functions, perhaps a more EBML focused utility would be the way to go as a
// first step.

const uint64 kEbmlUnknownValue = 0x01FFFFFFFFFFFFFFULL;
const int64 kMaxBlockTimecode = 0x07FFFLL;

// Writes out |value| in Big Endian order. Returns 0 on success.
int32 SerializeInt(IMkvWriter* writer, int64 value, int32 size);

// Writes out |f| in Big Endian order. Returns 0 on success.
int32 SerializeFloat(IMkvWriter* writer, float f);

// Returns the size in bytes of the element.
int32 GetUIntSize(uint64 value);
int32 GetIntSize(int64 value);
int32 GetCodedUIntSize(uint64 value);
uint64 EbmlMasterElementSize(uint64 type, uint64 value);
uint64 EbmlElementSize(uint64 type, int64 value);
uint64 EbmlElementSize(uint64 type, uint64 value);
uint64 EbmlElementSize(uint64 type, float value);
uint64 EbmlElementSize(uint64 type, const char* value);
uint64 EbmlElementSize(uint64 type, const uint8* value, uint64 size);
uint64 EbmlDateElementSize(uint64 type);

// Returns the size in bytes of the element assuming that the element was
// written using |fixed_size| bytes. If |fixed_size| is set to zero, then it
// computes the necessary number of bytes based on |value|.
uint64 EbmlElementSize(uint64 type, uint64 value, uint64 fixed_size);

// Creates an EBML coded number from |value| and writes it out. The size of
// the coded number is determined by the value of |value|. |value| must not
// be in a coded form. Returns 0 on success.
int32 WriteUInt(IMkvWriter* writer, uint64 value);

// Creates an EBML coded number from |value| and writes it out. The size of
// the coded number is determined by the value of |size|. |value| must not
// be in a coded form. Returns 0 on success.
int32 WriteUIntSize(IMkvWriter* writer, uint64 value, int32 size);

// Output an Mkv master element. Returns true if the element was written.
bool WriteEbmlMasterElement(IMkvWriter* writer, uint64 value, uint64 size);

// Outputs an Mkv ID, calls |IMkvWriter::ElementStartNotify|, and passes the
// ID to |SerializeInt|. Returns 0 on success.
int32 WriteID(IMkvWriter* writer, uint64 type);

// Output an Mkv non-master element. Returns true if the element was written.
bool WriteEbmlElement(IMkvWriter* writer, uint64 type, uint64 value);
bool WriteEbmlElement(IMkvWriter* writer, uint64 type, int64 value);
bool WriteEbmlElement(IMkvWriter* writer, uint64 type, float value);
bool WriteEbmlElement(IMkvWriter* writer, uint64 type, const char* value);
bool WriteEbmlElement(IMkvWriter* writer, uint64 type, const uint8* value,
                      uint64 size);
bool WriteEbmlDateElement(IMkvWriter* writer, uint64 type, int64 value);

// Output an Mkv non-master element using fixed size. The element will be
// written out using exactly |fixed_size| bytes. If |fixed_size| is set to zero
// then it computes the necessary number of bytes based on |value|. Returns true
// if the element was written.
bool WriteEbmlElement(IMkvWriter* writer, uint64 type, uint64 value,
                      uint64 fixed_size);

// Output a Mkv Frame. It decides the correct element to write (Block vs
// SimpleBlock) based on the parameters of the Frame.
uint64 WriteFrame(IMkvWriter* writer, const Frame* const frame,
                  Cluster* cluster);

// Output a void element. |size| must be the entire size in bytes that will be
// void. The function will calculate the size of the void header and subtract
// it from |size|.
uint64 WriteVoidElement(IMkvWriter* writer, uint64 size);

// Returns the version number of the muxer in |major|, |minor|, |build|,
// and |revision|.
void GetVersion(int32* major, int32* minor, int32* build, int32* revision);

// Returns a random number to be used for UID, using |seed| to seed
// the random-number generator (see POSIX rand_r() for semantics).
uint64 MakeUID(unsigned int* seed);

// Colour field validation helpers. All return true when |value| is valid.
bool IsMatrixCoefficientsValueValid(uint64_t value);
bool IsChromaSitingHorzValueValid(uint64_t value);
bool IsChromaSitingVertValueValid(uint64_t value);
bool IsColourRangeValueValid(uint64_t value);
bool IsTransferCharacteristicsValueValid(uint64_t value);
bool IsPrimariesValueValid(uint64_t value);

}  // namespace mkvmuxer

#endif  // MKVMUXER_MKVMUXERUTIL_H_
