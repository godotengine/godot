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

#include "util/mac/xattr.h"

#include <errno.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/xattr.h>

#include "base/logging.h"
#include "base/macros.h"
#include "base/numerics/safe_conversions.h"
#include "base/strings/string_number_conversions.h"
#include "base/strings/stringprintf.h"

namespace crashpad {

XattrStatus ReadXattr(const base::FilePath& file,
                      const base::StringPiece& name,
                      std::string* value) {
  // First get the size of the attribute value.
  ssize_t buffer_size = getxattr(file.value().c_str(), name.data(), nullptr,
                                 0, 0, 0);
  if (buffer_size < 0) {
    if (errno == ENOATTR)
      return XattrStatus::kNoAttribute;
    PLOG(ERROR) << "getxattr size " << name << " on file " << file.value();
    return XattrStatus::kOtherError;
  }

  // Resize the buffer and read into it.
  value->resize(buffer_size);
  if (!value->empty()) {
    ssize_t bytes_read = getxattr(file.value().c_str(), name.data(),
                                  &(*value)[0], value->size(),
                                  0, 0);
    if (bytes_read < 0) {
      PLOG(ERROR) << "getxattr " << name << " on file " << file.value();
      return XattrStatus::kOtherError;
    }
    DCHECK_EQ(bytes_read, buffer_size);
  }

  return XattrStatus::kOK;
}

bool WriteXattr(const base::FilePath& file,
                const base::StringPiece& name,
                const std::string& value) {
  int rv = setxattr(file.value().c_str(), name.data(), value.c_str(),
      value.length(), 0, 0);
  PLOG_IF(ERROR, rv != 0) << "setxattr " << name << " on file "
                          << file.value();
  return rv == 0;
}

XattrStatus ReadXattrBool(const base::FilePath& file,
                          const base::StringPiece& name,
                          bool* value) {
  std::string tmp;
  XattrStatus status;
  if ((status = ReadXattr(file, name, &tmp)) != XattrStatus::kOK)
    return status;
  if (tmp == "1") {
    *value = true;
    return XattrStatus::kOK;
  } else if (tmp == "0") {
    *value = false;
    return XattrStatus::kOK;
  } else {
    LOG(ERROR) << "ReadXattrBool " << name << " on file " << file.value()
               << " could not be interpreted as boolean";
    return XattrStatus::kOtherError;
  }
}

bool WriteXattrBool(const base::FilePath& file,
                    const base::StringPiece& name,
                    bool value) {
  return WriteXattr(file, name, (value ? "1" : "0"));
}

XattrStatus ReadXattrInt(const base::FilePath& file,
                         const base::StringPiece& name,
                         int* value) {
  std::string tmp;
  XattrStatus status;
  if ((status = ReadXattr(file, name, &tmp)) != XattrStatus::kOK)
    return status;
  if (!base::StringToInt(tmp, value)) {
    LOG(ERROR) << "ReadXattrInt " << name << " on file " << file.value()
               << " could not be converted to an int";
    return XattrStatus::kOtherError;
  }
  return XattrStatus::kOK;
}

bool WriteXattrInt(const base::FilePath& file,
                   const base::StringPiece& name,
                   int value) {
  std::string tmp = base::StringPrintf("%d", value);
  return WriteXattr(file, name, tmp);
}

XattrStatus ReadXattrTimeT(const base::FilePath& file,
                           const base::StringPiece& name,
                           time_t* value) {
  // time_t on macOS is defined as a long, but it will be read into an int64_t
  // here, since there is no string conversion method for long.
  std::string tmp;
  XattrStatus status;
  if ((status = ReadXattr(file, name, &tmp)) != XattrStatus::kOK)
    return status;

  int64_t encoded_value;
  if (!base::StringToInt64(tmp, &encoded_value)) {
    LOG(ERROR) << "ReadXattrTimeT " << name << " on file " << file.value()
               << " could not be converted to an int";
    return XattrStatus::kOtherError;
  }

  *value = base::saturated_cast<time_t>(encoded_value);
  if (!base::IsValueInRangeForNumericType<time_t>(encoded_value)) {
    LOG(ERROR) << "ReadXattrTimeT " << name << " on file " << file.value()
               << " read over-sized value and will saturate";
    return XattrStatus::kOtherError;
  }

  return XattrStatus::kOK;
}

bool WriteXattrTimeT(const base::FilePath& file,
                     const base::StringPiece& name,
                     time_t value) {
  std::string tmp = base::StringPrintf("%ld", value);
  return WriteXattr(file, name, tmp);
}

XattrStatus RemoveXattr(const base::FilePath& file,
                        const base::StringPiece& name) {
  int rv = removexattr(file.value().c_str(), name.data(), 0);
  if (rv != 0) {
    if (errno == ENOATTR)
      return XattrStatus::kNoAttribute;
    PLOG(ERROR) << "removexattr " << name << " on file " << file.value();
    return XattrStatus::kOtherError;
  }
  return XattrStatus::kOK;
}

}  // namespace crashpad
