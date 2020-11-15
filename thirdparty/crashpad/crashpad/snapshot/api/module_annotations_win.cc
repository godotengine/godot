// Copyright 2016 The Crashpad Authors. All rights reserved.
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

#include "snapshot/api/module_annotations_win.h"

#include "snapshot/win/pe_image_annotations_reader.h"
#include "snapshot/win/pe_image_reader.h"
#include "snapshot/win/process_reader_win.h"
#include "util/misc/from_pointer_cast.h"
#include "util/win/get_module_information.h"

namespace crashpad {

bool ReadModuleAnnotations(HANDLE process,
                           HMODULE module,
                           std::map<std::string, std::string>* annotations) {
  ProcessReaderWin process_reader;
  if (!process_reader.Initialize(process, ProcessSuspensionState::kRunning))
    return false;

  MODULEINFO module_info;
  if (!CrashpadGetModuleInformation(
          process, module, &module_info, sizeof(module_info))) {
    PLOG(ERROR) << "CrashpadGetModuleInformation";
    return false;
  }

  PEImageReader image_reader;
  if (!image_reader.Initialize(
          &process_reader,
          FromPointerCast<WinVMAddress>(module_info.lpBaseOfDll),
          module_info.SizeOfImage,
          ""))
    return false;

  PEImageAnnotationsReader annotations_reader(
      &process_reader, &image_reader, L"");

  *annotations = annotations_reader.SimpleMap();
  return true;
}

}  // namespace crashpad
