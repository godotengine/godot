// Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef _BSDIFF_DIFF_ENCODER_H_
#define _BSDIFF_DIFF_ENCODER_H_

#include <stdint.h>

#include "bsdiff/patch_writer_interface.h"

namespace bsdiff {

// Helper class to encapsulate the diff and extra stream generation logic
// derived from the old and new file buffers. Using this class is impossible to
// produce an invalid or incomplete bsdiff patch, since it has checks in place
// verifying its correct usage.

class DiffEncoder {
 public:
  // Initialize the DiffEncoder with the old and new file buffers, as well as
  // the path writer used. The |patch| will be initialized when calling Init().
  DiffEncoder(PatchWriterInterface* patch,
              const uint8_t* old_buf,
              uint64_t old_size,
              const uint8_t* new_buf,
              uint64_t new_size)
      : patch_(patch),
        old_buf_(old_buf),
        old_size_(old_size),
        new_buf_(new_buf),
        new_size_(new_size) {}

  // Initialize the diff encoder and the underlying patch.
  bool Init();

  // Add a new control triplet entry to the patch. The |entry.diff_size| bytes
  // for the diff stream and the |entry.extra_size| bytes for the extra stream
  // will be computed and added to the corresponding streams in the patch.
  // Returns whether the operation succeeded. The operation can fail if either
  // the old or new files are referenced out of bounds.
  bool AddControlEntry(const ControlEntry& entry);

  // Finalize the patch writing process and close the underlying patch writer.
  bool Close();

 private:
  // Pointer to the patch we are writing to.
  PatchWriterInterface* patch_;

  // Old and new file buffers.
  const uint8_t* old_buf_;
  uint64_t old_size_;
  const uint8_t* new_buf_;
  uint64_t new_size_;

  // Bytes of the new_buf_ already written.
  uint64_t written_output_{0};

  // The current position in the old buf.
  int64_t old_pos_{0};
};

}  // namespace bsdiff

#endif  // _BSDIFF_DIFF_ENCODER_H_
