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
  // Returns true if |ext| is an enabled extension in the module.
  bool HasExtension(Extension ext) const { return extensions_.contains(ext); }

  // Returns true if |cap| is an enabled capability in the module.
  bool HasCapability(spv::Capability cap) const {
    return capabilities_.contains(cap);
  }

  // Returns the capabilities the module declares.
  inline const CapabilitySet& GetCapabilities() const { return capabilities_; }

  // Returns the extensions the module imports.
  inline const ExtensionSet& GetExtensions() const { return extensions_; }

  uint32_t GetExtInstImportId_GLSLstd450() const {
    return extinst_importid_GLSLstd450_;
  }

  uint32_t GetExtInstImportId_OpenCL100DebugInfo() const {
    return extinst_importid_OpenCL100DebugInfo_;
  }

  uint32_t GetExtInstImportId_Shader100DebugInfo() const {
    return extinst_importid_Shader100DebugInfo_;
  }

  friend bool operator==(const FeatureManager& a, const FeatureManager& b);
  friend bool operator!=(const FeatureManager& a, const FeatureManager& b) {
    return !(a == b);
  }

 private:
  explicit FeatureManager(const AssemblyGrammar& grammar) : grammar_(grammar) {}

  // Analyzes |module| and records enabled extensions and capabilities.
  void Analyze(Module* module);

  // Add the extension |ext| to the feature manager.
  void AddExtension(Instruction* ext);

  // Analyzes |module| and records enabled extensions.
  void AddExtensions(Module* module);

  // Removes the given |extension| from the current FeatureManager.
  void RemoveExtension(Extension extension);

  // Adds the given |capability| and all implied capabilities into the current
  // FeatureManager.
  void AddCapability(spv::Capability capability);

  // Analyzes |module| and records enabled capabilities.
  void AddCapabilities(Module* module);

  // Removes the given |capability| from the current FeatureManager.
  void RemoveCapability(spv::Capability capability);

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

  // Common OpenCL100DebugInfo external instruction import ids, cached
  // for performance.
  uint32_t extinst_importid_OpenCL100DebugInfo_ = 0;

  // Common NonSemanticShader100DebugInfo external instruction import ids,
  // cached for performance.
  uint32_t extinst_importid_Shader100DebugInfo_ = 0;

  friend class IRContext;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_FEATURE_MANAGER_H_
