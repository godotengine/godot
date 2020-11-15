// Copyright 2017 The Crashpad Authors. All rights reserved.
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

#include "util/linux/auxiliary_vector.h"

#include <linux/auxvec.h>
#include <stdio.h>

#include "base/files/file_path.h"
#include "base/logging.h"
#include "util/file/file_reader.h"
#include "util/file/string_file.h"
#include "util/stdlib/map_insert.h"

namespace crashpad {

AuxiliaryVector::AuxiliaryVector() : values_() {}

AuxiliaryVector::~AuxiliaryVector() {}

bool AuxiliaryVector::Initialize(PtraceConnection* connection) {
  return connection->Is64Bit() ? Read<uint64_t>(connection)
                               : Read<uint32_t>(connection);
}

template <typename ULong>
bool AuxiliaryVector::Read(PtraceConnection* connection) {
  char path[32];
  snprintf(path, sizeof(path), "/proc/%d/auxv", connection->GetProcessID());

  std::string contents;
  if (!connection->ReadFileContents(base::FilePath(path), &contents)) {
    return false;
  }
  StringFile aux_file;
  aux_file.SetString(contents);

  ULong type;
  ULong value;
  while (aux_file.ReadExactly(&type, sizeof(type)) &&
         aux_file.ReadExactly(&value, sizeof(value))) {
    if (type == AT_NULL && value == 0) {
      return true;
    }
    if (type == AT_IGNORE) {
      continue;
    }
    if (!MapInsertOrReplace(&values_, type, value, nullptr)) {
      LOG(ERROR) << "duplicate auxv entry";
      return false;
    }
  }
  return false;
}

}  // namespace crashpad
