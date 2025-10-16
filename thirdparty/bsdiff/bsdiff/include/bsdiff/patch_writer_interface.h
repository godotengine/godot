// Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef _BSDIFF_PATCH_WRITER_INTERFACE_H_
#define _BSDIFF_PATCH_WRITER_INTERFACE_H_

#include <stddef.h>
#include <stdint.h>

#include "bsdiff/control_entry.h"

namespace bsdiff {

enum class BsdiffFormat {
  kLegacy,
  kBsdf2,
  kEndsley,
};

class PatchWriterInterface {
 public:
  virtual ~PatchWriterInterface() = default;

  // Initialize the patch writer for a patch where the new file will have
  // |new_size| bytes.
  virtual bool Init(size_t new_size) = 0;

  // Write the passed |data| buffer of length |size| to the Diff or Extra
  // streams respectively. Each method can be called independently from each
  // other and calls don't need to be a correlation with the AddControlEntry()
  // until Close() is called.
  virtual bool WriteDiffStream(const uint8_t* data, size_t size) = 0;
  virtual bool WriteExtraStream(const uint8_t* data, size_t size) = 0;

  // Add a new control triplet entry to the patch. These triplets may be added
  // at any point before calling Close(), regardless of whether the
  // corresponding WriteDiffStream() and WriteExtraStream() have been called
  // yet.
  virtual bool AddControlEntry(const ControlEntry& entry) = 0;

  // Finalize the patch writing process and close the file.
  virtual bool Close() = 0;

 protected:
  PatchWriterInterface() = default;
};

}  // namespace bsdiff

#endif  // _BSDIFF_PATCH_WRITER_INTERFACE_H_
