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

#include "handler/minidump_to_upload_parameters.h"

#include "base/logging.h"
#include "client/annotation.h"
#include "snapshot/module_snapshot.h"
#include "util/stdlib/map_insert.h"

namespace crashpad {

namespace {

void InsertOrReplaceMapEntry(std::map<std::string, std::string>* map,
                             const std::string& key,
                             const std::string& value) {
  std::string old_value;
  if (!MapInsertOrReplace(map, key, value, &old_value)) {
    LOG(WARNING) << "duplicate key " << key << ", discarding value "
                 << old_value;
  }
}

}  // namespace

std::map<std::string, std::string> BreakpadHTTPFormParametersFromMinidump(
    const ProcessSnapshot* process_snapshot) {
  std::map<std::string, std::string> parameters =
      process_snapshot->AnnotationsSimpleMap();

  std::string list_annotations;
  for (const ModuleSnapshot* module : process_snapshot->Modules()) {
    for (const auto& kv : module->AnnotationsSimpleMap()) {
      if (!parameters.insert(kv).second) {
        LOG(WARNING) << "duplicate key " << kv.first << ", discarding value "
                     << kv.second;
      }
    }

    for (std::string annotation : module->AnnotationsVector()) {
      list_annotations.append(annotation);
      list_annotations.append("\n");
    }

    for (const AnnotationSnapshot& annotation : module->AnnotationObjects()) {
      if (annotation.type != static_cast<uint16_t>(Annotation::Type::kString)) {
        continue;
      }

      std::string value(reinterpret_cast<const char*>(annotation.value.data()),
                        annotation.value.size());
      std::pair<std::string, std::string> entry(annotation.name, value);
      if (!parameters.insert(entry).second) {
        LOG(WARNING) << "duplicate annotation name " << annotation.name
                     << ", discarding value " << value;
      }
    }
  }

  if (!list_annotations.empty()) {
    // Remove the final newline character.
    list_annotations.resize(list_annotations.size() - 1);

    InsertOrReplaceMapEntry(&parameters, "list_annotations", list_annotations);
  }

  UUID client_id;
  process_snapshot->ClientID(&client_id);
  InsertOrReplaceMapEntry(&parameters, "guid", client_id.ToString());

  return parameters;
}

}  // namespace crashpad
