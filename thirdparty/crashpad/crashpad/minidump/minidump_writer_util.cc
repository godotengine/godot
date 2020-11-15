// Copyright 2014 The Crashpad Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "minidump/minidump_writer_util.h"

#include "base/logging.h"
#include "base/numerics/safe_conversions.h"
#include "base/strings/utf_string_conversions.h"
#include "util/stdlib/strlcpy.h"

namespace crashpad {
namespace internal {

// static
void MinidumpWriterUtil::AssignTimeT(uint32_t* destination, time_t source) {
  if (!base::IsValueInRangeForNumericType<uint32_t>(source)) {
    LOG(WARNING) << "timestamp " << source << " out of range";
  }

  *destination = static_cast<uint32_t>(source);
}

// static
base::string16 MinidumpWriterUtil::ConvertUTF8ToUTF16(const std::string& utf8) {
  base::string16 utf16;
  if (!base::UTF8ToUTF16(utf8.data(), utf8.length(), &utf16)) {
    LOG(WARNING) << "string " << utf8
                 << " cannot be converted to UTF-16 losslessly";
  }
  return utf16;
}

// static
void MinidumpWriterUtil::AssignUTF8ToUTF16(base::char16* destination,
                                           size_t destination_size,
                                           const std::string& source) {
  base::string16 source_utf16 = ConvertUTF8ToUTF16(source);
  if (source_utf16.size() > destination_size - 1) {
    LOG(WARNING) << "string " << source << " UTF-16 length "
                 << source_utf16.size()
                 << " will be truncated to UTF-16 length "
                 << destination_size - 1;
  }

  source_utf16.resize(destination_size - 1);
  c16lcpy(destination, source_utf16.c_str(), destination_size);
}

}  // namespace internal
}  // namespace crashpad
