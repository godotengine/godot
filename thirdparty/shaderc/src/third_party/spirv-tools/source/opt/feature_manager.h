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

#ifndef SOURCE_OPT_FEATURE_MANAGER_H_
#define SOURCE_OPT_FEATURE_MANAGER_H_

#include "source/assembly_grammar.h"
#include "source/extensions.h"
#include "source/opt/module.h"

namespace spvtools {
namespace opt {

// Tracks features enabled by a module. The IRContext has a FeatureManager.
class FeatureManager {
 public:
  explicit FeatureManager(const AssemblyGrammar& grammar) : grammar_(grammar) {}

  // Returns true if |ext| is an enabled extension in the module.
  bool HasExtension(Extension ext) const { return extensions_.Contains(ext); }

  // Returns true if |cap| is an enabled capability in the module.
  bool HasCapability(SpvCapability cap) const {
    return capabilities_.Contains(cap);
  }

  // Analyzes |module| and records enabled extensions and capabilities.
  void Analyze(Module* module);

  CapabilitySet* GetCapabilities() { return &capabilities_; }
  const CapabilitySet* GetCapabilities() const { return &capabilities_; }

  uint32_t GetExtInstImportId_GLSLstd450() const {
    return extinst_importid_GLSLstd450_;
  }

 private:
  // Analyzes |module| and records enabled extensions.
  void AddExtensions(Module* module);

  // Adds the given |capability| and all implied capabilities into the current
  // FeatureManager.
  void AddCapability(SpvCapability capability);

  // Analyzes |module| and records enabled capabilities.
  void AddCapabilities(Module* module);

  // Analyzes |module| and records imported external instruction sets.
  void AddExtInstImportIds(Module* module);

  // Auxiliary object for querying SPIR-V grammar facts.
  const AssemblyGrammar& grammar_;

  // The enabled extensions.
  ExtensionSet extensions_;

  // The enabled capabilities.
  CapabilitySet capabilities_;

  // Common external instruction import ids, cached for performance.
  uint32_t extinst_importid_GLSLstd450_ = 0;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_FEATURE_MANAGER_H_
