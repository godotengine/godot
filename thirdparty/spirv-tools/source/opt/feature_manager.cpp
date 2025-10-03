// Copyright (c) 2017 Google Inc.
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

#include "source/opt/feature_manager.h"

#include <queue>
#include <stack>
#include <string>

#include "source/enum_string_mapping.h"

namespace spvtools {
namespace opt {

void FeatureManager::Analyze(Module* module) {
  AddExtensions(module);
  AddCapabilities(module);
  AddExtInstImportIds(module);
}

void FeatureManager::AddExtensions(Module* module) {
  for (auto ext : module->extensions()) {
    AddExtension(&ext);
  }
}

void FeatureManager::AddExtension(Instruction* ext) {
  assert(ext->opcode() == spv::Op::OpExtension &&
         "Expecting an extension instruction.");

  const std::string name = ext->GetInOperand(0u).AsString();
  Extension extension;
  if (GetExtensionFromString(name.c_str(), &extension)) {
    extensions_.Add(extension);
  }
}

void FeatureManager::RemoveExtension(Extension ext) {
  if (!extensions_.Contains(ext)) return;
  extensions_.Remove(ext);
}

void FeatureManager::AddCapability(spv::Capability cap) {
  if (capabilities_.Contains(cap)) return;

  capabilities_.Add(cap);

  spv_operand_desc desc = {};
  if (SPV_SUCCESS == grammar_.lookupOperand(SPV_OPERAND_TYPE_CAPABILITY,
                                            uint32_t(cap), &desc)) {
    CapabilitySet(desc->numCapabilities, desc->capabilities)
        .ForEach([this](spv::Capability c) { AddCapability(c); });
  }
}

void FeatureManager::RemoveCapability(spv::Capability cap) {
  if (!capabilities_.Contains(cap)) return;
  capabilities_.Remove(cap);
}

void FeatureManager::AddCapabilities(Module* module) {
  for (Instruction& inst : module->capabilities()) {
    AddCapability(static_cast<spv::Capability>(inst.GetSingleWordInOperand(0)));
  }
}

void FeatureManager::AddExtInstImportIds(Module* module) {
  extinst_importid_GLSLstd450_ = module->GetExtInstImportId("GLSL.std.450");
  extinst_importid_OpenCL100DebugInfo_ =
      module->GetExtInstImportId("OpenCL.DebugInfo.100");
  extinst_importid_Shader100DebugInfo_ =
      module->GetExtInstImportId("NonSemantic.Shader.DebugInfo.100");
}

bool operator==(const FeatureManager& a, const FeatureManager& b) {
  // We check that the addresses of the grammars are the same because they
  // are large objects, and this is faster.  It can be changed if needed as a
  // later time.
  if (&a.grammar_ != &b.grammar_) {
    return false;
  }

  if (a.capabilities_ != b.capabilities_) {
    return false;
  }

  if (a.extensions_ != b.extensions_) {
    return false;
  }

  if (a.extinst_importid_GLSLstd450_ != b.extinst_importid_GLSLstd450_) {
    return false;
  }

  if (a.extinst_importid_OpenCL100DebugInfo_ !=
      b.extinst_importid_OpenCL100DebugInfo_) {
    return false;
  }

  if (a.extinst_importid_Shader100DebugInfo_ !=
      b.extinst_importid_Shader100DebugInfo_) {
    return false;
  }

  return true;
}
}  // namespace opt
}  // namespace spvtools
