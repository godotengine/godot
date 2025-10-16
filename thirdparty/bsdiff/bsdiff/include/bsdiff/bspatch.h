// Copyright 2015 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef _BSDIFF_BSPATCH_H_
#define _BSDIFF_BSPATCH_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>

#include "bsdiff/common.h"
#include "bsdiff/file_interface.h"
#include "bsdiff/patch_reader_interface.h"

namespace bsdiff {

BSDIFF_EXPORT
int bspatch(const std::unique_ptr<FileInterface>& old_file,
            const std::unique_ptr<FileInterface>& new_file,
            const uint8_t* patch_data,
            size_t patch_size,
			PatchReaderInterface& patch_reader);

bool WriteAll(const std::unique_ptr<FileInterface>& file,
              const uint8_t* data,
              size_t size);

}  // namespace bsdiff

#endif  // _BSDIFF_BSPATCH_H_
