// Copyright (c) 2017 Pierre Moreau
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

#include "spirv-tools/linker.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "source/assembly_grammar.h"
#include "source/diagnostic.h"
#include "source/opt/build_module.h"
#include "source/opt/compact_ids_pass.h"
#include "source/opt/decoration_manager.h"
#include "source/opt/ir_loader.h"
#include "source/opt/pass_manager.h"
#include "source/opt/remove_duplicates_pass.h"
#include "source/spirv_target_env.h"
#include "source/util/make_unique.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace {

using opt::IRContext;
using opt::Instruction;
using opt::Module;
using opt::Operand;
using opt::PassManager;
using opt::RemoveDuplicatesPass;
using opt::analysis::DecorationManager;
using opt::analysis::DefUseManager;

// Stores various information about an imported or exported symbol.
struct LinkageSymbolInfo {
  SpvId id;          // ID of the symbol
  SpvId type_id;     // ID of the type of the symbol
  std::string name;  // unique name defining the symbol and used for matching
                     // imports and exports together
  std::vector<SpvId> parameter_ids;  // ID of the parameters of the symbol, if
                                     // it is a function
};
struct LinkageEntry {
  LinkageSymbolInfo imported_symbol;
  LinkageSymbolInfo exported_symbol;

  LinkageEntry(const LinkageSymbolInfo& import_info,
               const LinkageSymbolInfo& export_info)
      : imported_symbol(import_info), exported_symbol(export_info) {}
};
using LinkageTable = std::vector<LinkageEntry>;

// Shifts the IDs used in each binary of |modules| so that they occupy a
// disjoint range from the other binaries, and compute the new ID bound which
// is returned in |max_id_bound|.
//
// Both |modules| and |max_id_bound| should not be null, and |modules| should
// not be empty either. Furthermore |modules| should not contain any null
// pointers.
spv_result_t ShiftIdsInModules(const MessageConsumer& consumer,
                               std::vector<opt::Module*>* modules,
                               uint32_t* max_id_bound);

// Generates the header for the linked module and returns it in |header|.
//
// |header| should not be null, |modules| should not be empty and pointers
// should be non-null. |max_id_bound| should be strictly greater than 0.
//
// TODO(pierremoreau): What to do when binaries use different versions of
//                     SPIR-V? For now, use the max of all versions found in
//                     the input modules.
spv_result_t GenerateHeader(const MessageConsumer& consumer,
                            const std::vector<opt::Module*>& modules,
                            uint32_t max_id_bound, opt::ModuleHeader* header);

// Merge all the modules from |in_modules| into a single module owned by
// |linked_context|.
//
// |linked_context| should not be null.
spv_result_t MergeModules(const MessageConsumer& consumer,
                          const std::vector<Module*>& in_modules,
                          const AssemblyGrammar& grammar,
                          IRContext* linked_context);

// Compute all pairs of import and export and return it in |linkings_to_do|.
//
// |linkings_to_do should not be null. Built-in symbols will be ignored.
//
// TODO(pierremoreau): Linkage attributes applied by a group decoration are
//                     currently not handled. (You could have a group being
//                     applied to a single ID.)
// TODO(pierremoreau): What should be the proper behaviour with built-in
//                     symbols?
spv_result_t GetImportExportPairs(const MessageConsumer& consumer,
                                  const opt::IRContext& linked_context,
                                  const DefUseManager& def_use_manager,
                                  const DecorationManager& decoration_manager,
                                  bool allow_partial_linkage,
                                  LinkageTable* linkings_to_do);

// Checks that for each pair of import and export, the import and export have
// the same type as well as the same decorations.
//
// TODO(pierremoreau): Decorations on functions parameters are currently not
// checked.
spv_result_t CheckImportExportCompatibility(const MessageConsumer& consumer,
                                            const LinkageTable& linkings_to_do,
                                            opt::IRContext* context);

// Remove linkage specific instructions, such as prototypes of imported
// functions, declarations of imported variables, import (and export if
// necessary) linkage attribtes.
//
// |linked_context| and |decoration_manager| should not be null, and the
// 'RemoveDuplicatePass' should be run first.
//
// TODO(pierremoreau): Linkage attributes applied by a group decoration are
//                     currently not handled. (You could have a group being
//                     applied to a single ID.)
// TODO(pierremoreau): Run a pass for removing dead instructions, for example
//                     OpName for prototypes of imported funcions.
spv_result_t RemoveLinkageSpecificInstructions(
    const MessageConsumer& consumer, const LinkerOptions& options,
    const LinkageTable& linkings_to_do, DecorationManager* decoration_manager,
    opt::IRContext* linked_context);

// Verify that the unique ids of each instruction in |linked_context| (i.e. the
// merged module) are truly unique. Does not check the validity of other ids
spv_result_t VerifyIds(const MessageConsumer& consumer,
                       opt::IRContext* linked_context);

spv_result_t ShiftIdsInModules(const MessageConsumer& consumer,
                               std::vector<opt::Module*>* modules,
                               uint32_t* max_id_bound) {
  spv_position_t position = {};

  if (modules == nullptr)
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_DATA)
           << "|modules| of ShiftIdsInModules should not be null.";
  if (modules->empty())
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_DATA)
           << "|modules| of ShiftIdsInModules should not be empty.";
  if (max_id_bound == nullptr)
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_DATA)
           << "|max_id_bound| of ShiftIdsInModules should not be null.";

  uint32_t id_bound = modules->front()->IdBound() - 1u;
  for (auto module_iter = modules->begin() + 1; module_iter != modules->end();
       ++module_iter) {
    Module* module = *module_iter;
    module->ForEachInst([&id_bound](Instruction* insn) {
      insn->ForEachId([&id_bound](uint32_t* id) { *id += id_bound; });
    });
    id_bound += module->IdBound() - 1u;
    if (id_bound > 0x3FFFFF)
      return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_ID)
             << "The limit of IDs, 4194303, was exceeded:"
             << " " << id_bound << " is the current ID bound.";

    // Invalidate the DefUseManager
    module->context()->InvalidateAnalyses(opt::IRContext::kAnalysisDefUse);
  }
  ++id_bound;
  if (id_bound > 0x3FFFFF)
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_ID)
           << "The limit of IDs, 4194303, was exceeded:"
           << " " << id_bound << " is the current ID bound.";

  *max_id_bound = id_bound;

  return SPV_SUCCESS;
}

spv_result_t GenerateHeader(const MessageConsumer& consumer,
                            const std::vector<opt::Module*>& modules,
                            uint32_t max_id_bound, opt::ModuleHeader* header) {
  spv_position_t position = {};

  if (modules.empty())
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_DATA)
           << "|modules| of GenerateHeader should not be empty.";
  if (max_id_bound == 0u)
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_DATA)
           << "|max_id_bound| of GenerateHeader should not be null.";

  uint32_t version = 0u;
  for (const auto& module : modules)
    version = std::max(version, module->version());

  header->magic_number = SpvMagicNumber;
  header->version = version;
  header->generator = 17u;
  header->bound = max_id_bound;
  header->reserved = 0u;

  return SPV_SUCCESS;
}

spv_result_t MergeModules(const MessageConsumer& consumer,
                          const std::vector<Module*>& input_modules,
                          const AssemblyGrammar& grammar,
                          IRContext* linked_context) {
  spv_position_t position = {};

  if (linked_context == nullptr)
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_DATA)
           << "|linked_module| of MergeModules should not be null.";
  Module* linked_module = linked_context->module();

  if (input_modules.empty()) return SPV_SUCCESS;

  for (const auto& module : input_modules)
    for (const auto& inst : module->capabilities())
      linked_module->AddCapability(
          std::unique_ptr<Instruction>(inst.Clone(linked_context)));

  for (const auto& module : input_modules)
    for (const auto& inst : module->extensions())
      linked_module->AddExtension(
          std::unique_ptr<Instruction>(inst.Clone(linked_context)));

  for (const auto& module : input_modules)
    for (const auto& inst : module->ext_inst_imports())
      linked_module->AddExtInstImport(
          std::unique_ptr<Instruction>(inst.Clone(linked_context)));

  do {
    const Instruction* memory_model_inst = input_modules[0]->GetMemoryModel();
    if (memory_model_inst == nullptr) break;

    uint32_t addressing_model = memory_model_inst->GetSingleWordOperand(0u);
    uint32_t memory_model = memory_model_inst->GetSingleWordOperand(1u);
    for (const auto& module : input_modules) {
      memory_model_inst = module->GetMemoryModel();
      if (memory_model_inst == nullptr) continue;

      if (addressing_model != memory_model_inst->GetSingleWordOperand(0u)) {
        spv_operand_desc initial_desc = nullptr, current_desc = nullptr;
        grammar.lookupOperand(SPV_OPERAND_TYPE_ADDRESSING_MODEL,
                              addressing_model, &initial_desc);
        grammar.lookupOperand(SPV_OPERAND_TYPE_ADDRESSING_MODEL,
                              memory_model_inst->GetSingleWordOperand(0u),
                              &current_desc);
        return DiagnosticStream(position, consumer, "", SPV_ERROR_INTERNAL)
               << "Conflicting addressing models: " << initial_desc->name
               << " vs " << current_desc->name << ".";
      }
      if (memory_model != memory_model_inst->GetSingleWordOperand(1u)) {
        spv_operand_desc initial_desc = nullptr, current_desc = nullptr;
        grammar.lookupOperand(SPV_OPERAND_TYPE_MEMORY_MODEL, memory_model,
                              &initial_desc);
        grammar.lookupOperand(SPV_OPERAND_TYPE_MEMORY_MODEL,
                              memory_model_inst->GetSingleWordOperand(1u),
                              &current_desc);
        return DiagnosticStream(position, consumer, "", SPV_ERROR_INTERNAL)
               << "Conflicting memory models: " << initial_desc->name << " vs "
               << current_desc->name << ".";
      }
    }

    if (memory_model_inst != nullptr)
      linked_module->SetMemoryModel(std::unique_ptr<Instruction>(
          memory_model_inst->Clone(linked_context)));
  } while (false);

  std::vector<std::pair<uint32_t, const char*>> entry_points;
  for (const auto& module : input_modules)
    for (const auto& inst : module->entry_points()) {
      const uint32_t model = inst.GetSingleWordInOperand(0);
      const char* const name =
          reinterpret_cast<const char*>(inst.GetInOperand(2).words.data());
      const auto i = std::find_if(
          entry_points.begin(), entry_points.end(),
          [model, name](const std::pair<uint32_t, const char*>& v) {
            return v.first == model && strcmp(name, v.second) == 0;
          });
      if (i != entry_points.end()) {
        spv_operand_desc desc = nullptr;
        grammar.lookupOperand(SPV_OPERAND_TYPE_EXECUTION_MODEL, model, &desc);
        return DiagnosticStream(position, consumer, "", SPV_ERROR_INTERNAL)
               << "The entry point \"" << name << "\", with execution model "
               << desc->name << ", was already defined.";
      }
      linked_module->AddEntryPoint(
          std::unique_ptr<Instruction>(inst.Clone(linked_context)));
      entry_points.emplace_back(model, name);
    }

  for (const auto& module : input_modules)
    for (const auto& inst : module->execution_modes())
      linked_module->AddExecutionMode(
          std::unique_ptr<Instruction>(inst.Clone(linked_context)));

  for (const auto& module : input_modules)
    for (const auto& inst : module->debugs1())
      linked_module->AddDebug1Inst(
          std::unique_ptr<Instruction>(inst.Clone(linked_context)));

  for (const auto& module : input_modules)
    for (const auto& inst : module->debugs2())
      linked_module->AddDebug2Inst(
          std::unique_ptr<Instruction>(inst.Clone(linked_context)));

  for (const auto& module : input_modules)
    for (const auto& inst : module->debugs3())
      linked_module->AddDebug3Inst(
          std::unique_ptr<Instruction>(inst.Clone(linked_context)));

  // If the generated module uses SPIR-V 1.1 or higher, add an
  // OpModuleProcessed instruction about the linking step.
  if (linked_module->version() >= 0x10100) {
    const std::string processed_string("Linked by SPIR-V Tools Linker");
    const auto num_chars = processed_string.size();
    // Compute num words, accommodate the terminating null character.
    const auto num_words = (num_chars + 1 + 3) / 4;
    std::vector<uint32_t> processed_words(num_words, 0u);
    std::memcpy(processed_words.data(), processed_string.data(), num_chars);
    linked_module->AddDebug3Inst(std::unique_ptr<Instruction>(
        new Instruction(linked_context, SpvOpModuleProcessed, 0u, 0u,
                        {{SPV_OPERAND_TYPE_LITERAL_STRING, processed_words}})));
  }

  for (const auto& module : input_modules)
    for (const auto& inst : module->annotations())
      linked_module->AddAnnotationInst(
          std::unique_ptr<Instruction>(inst.Clone(linked_context)));

  // TODO(pierremoreau): Since the modules have not been validate, should we
  //                     expect SpvStorageClassFunction variables outside
  //                     functions?
  uint32_t num_global_values = 0u;
  for (const auto& module : input_modules) {
    for (const auto& inst : module->types_values()) {
      linked_module->AddType(
          std::unique_ptr<Instruction>(inst.Clone(linked_context)));
      num_global_values += inst.opcode() == SpvOpVariable;
    }
  }
  if (num_global_values > 0xFFFF)
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INTERNAL)
           << "The limit of global values, 65535, was exceeded;"
           << " " << num_global_values << " global values were found.";

  // Process functions and their basic blocks
  for (const auto& module : input_modules) {
    for (const auto& func : *module) {
      std::unique_ptr<opt::Function> cloned_func(func.Clone(linked_context));
      linked_module->AddFunction(std::move(cloned_func));
    }
  }

  return SPV_SUCCESS;
}

spv_result_t GetImportExportPairs(const MessageConsumer& consumer,
                                  const opt::IRContext& linked_context,
                                  const DefUseManager& def_use_manager,
                                  const DecorationManager& decoration_manager,
                                  bool allow_partial_linkage,
                                  LinkageTable* linkings_to_do) {
  spv_position_t position = {};

  if (linkings_to_do == nullptr)
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_DATA)
           << "|linkings_to_do| of GetImportExportPairs should not be empty.";

  std::vector<LinkageSymbolInfo> imports;
  std::unordered_map<std::string, std::vector<LinkageSymbolInfo>> exports;

  // Figure out the imports and exports
  for (const auto& decoration : linked_context.annotations()) {
    if (decoration.opcode() != SpvOpDecorate ||
        decoration.GetSingleWordInOperand(1u) != SpvDecorationLinkageAttributes)
      continue;

    const SpvId id = decoration.GetSingleWordInOperand(0u);
    // Ignore if the targeted symbol is a built-in
    bool is_built_in = false;
    for (const auto& id_decoration :
         decoration_manager.GetDecorationsFor(id, false)) {
      if (id_decoration->GetSingleWordInOperand(1u) == SpvDecorationBuiltIn) {
        is_built_in = true;
        break;
      }
    }
    if (is_built_in) {
      continue;
    }

    const uint32_t type = decoration.GetSingleWordInOperand(3u);

    LinkageSymbolInfo symbol_info;
    symbol_info.name =
        reinterpret_cast<const char*>(decoration.GetInOperand(2u).words.data());
    symbol_info.id = id;
    symbol_info.type_id = 0u;

    // Retrieve the type of the current symbol. This information will be used
    // when checking that the imported and exported symbols have the same
    // types.
    const Instruction* def_inst = def_use_manager.GetDef(id);
    if (def_inst == nullptr)
      return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_BINARY)
             << "ID " << id << " is never defined:\n";

    if (def_inst->opcode() == SpvOpVariable) {
      symbol_info.type_id = def_inst->type_id();
    } else if (def_inst->opcode() == SpvOpFunction) {
      symbol_info.type_id = def_inst->GetSingleWordInOperand(1u);

      // range-based for loop calls begin()/end(), but never cbegin()/cend(),
      // which will not work here.
      for (auto func_iter = linked_context.module()->cbegin();
           func_iter != linked_context.module()->cend(); ++func_iter) {
        if (func_iter->result_id() != id) continue;
        func_iter->ForEachParam([&symbol_info](const Instruction* inst) {
          symbol_info.parameter_ids.push_back(inst->result_id());
        });
      }
    } else {
      return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_BINARY)
             << "Only global variables and functions can be decorated using"
             << " LinkageAttributes; " << id << " is neither of them.\n";
    }

    if (type == SpvLinkageTypeImport)
      imports.push_back(symbol_info);
    else if (type == SpvLinkageTypeExport)
      exports[symbol_info.name].push_back(symbol_info);
  }

  // Find the import/export pairs
  for (const auto& import : imports) {
    std::vector<LinkageSymbolInfo> possible_exports;
    const auto& exp = exports.find(import.name);
    if (exp != exports.end()) possible_exports = exp->second;
    if (possible_exports.empty() && !allow_partial_linkage)
      return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_BINARY)
             << "Unresolved external reference to \"" << import.name << "\".";
    else if (possible_exports.size() > 1u)
      return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_BINARY)
             << "Too many external references, " << possible_exports.size()
             << ", were found for \"" << import.name << "\".";

    if (!possible_exports.empty())
      linkings_to_do->emplace_back(import, possible_exports.front());
  }

  return SPV_SUCCESS;
}

spv_result_t CheckImportExportCompatibility(const MessageConsumer& consumer,
                                            const LinkageTable& linkings_to_do,
                                            opt::IRContext* context) {
  spv_position_t position = {};

  // Ensure th import and export types are the same.
  const DefUseManager& def_use_manager = *context->get_def_use_mgr();
  const DecorationManager& decoration_manager = *context->get_decoration_mgr();
  for (const auto& linking_entry : linkings_to_do) {
    if (!RemoveDuplicatesPass::AreTypesEqual(
            *def_use_manager.GetDef(linking_entry.imported_symbol.type_id),
            *def_use_manager.GetDef(linking_entry.exported_symbol.type_id),
            context))
      return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_BINARY)
             << "Type mismatch on symbol \""
             << linking_entry.imported_symbol.name
             << "\" between imported variable/function %"
             << linking_entry.imported_symbol.id
             << " and exported variable/function %"
             << linking_entry.exported_symbol.id << ".";
  }

  // Ensure the import and export decorations are similar
  for (const auto& linking_entry : linkings_to_do) {
    if (!decoration_manager.HaveTheSameDecorations(
            linking_entry.imported_symbol.id, linking_entry.exported_symbol.id))
      return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_BINARY)
             << "Decorations mismatch on symbol \""
             << linking_entry.imported_symbol.name
             << "\" between imported variable/function %"
             << linking_entry.imported_symbol.id
             << " and exported variable/function %"
             << linking_entry.exported_symbol.id << ".";
    // TODO(pierremoreau): Decorations on function parameters should probably
    //                     match, except for FuncParamAttr if I understand the
    //                     spec correctly.
    // TODO(pierremoreau): Decorations on the function return type should
    //                     match, except for FuncParamAttr.
  }

  return SPV_SUCCESS;
}

spv_result_t RemoveLinkageSpecificInstructions(
    const MessageConsumer& consumer, const LinkerOptions& options,
    const LinkageTable& linkings_to_do, DecorationManager* decoration_manager,
    opt::IRContext* linked_context) {
  spv_position_t position = {};

  if (decoration_manager == nullptr)
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_DATA)
           << "|decoration_manager| of RemoveLinkageSpecificInstructions "
              "should not be empty.";
  if (linked_context == nullptr)
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_DATA)
           << "|linked_module| of RemoveLinkageSpecificInstructions should not "
              "be empty.";

  // TODO(pierremoreau): Remove FuncParamAttr decorations of imported
  // functions' return type.

  // Remove FuncParamAttr decorations of imported functions' parameters.
  // From the SPIR-V specification, Sec. 2.13:
  //   When resolving imported functions, the Function Control and all Function
  //   Parameter Attributes are taken from the function definition, and not
  //   from the function declaration.
  for (const auto& linking_entry : linkings_to_do) {
    for (const auto parameter_id :
         linking_entry.imported_symbol.parameter_ids) {
      decoration_manager->RemoveDecorationsFrom(
          parameter_id, [](const Instruction& inst) {
            return (inst.opcode() == SpvOpDecorate ||
                    inst.opcode() == SpvOpMemberDecorate) &&
                   inst.GetSingleWordInOperand(1u) ==
                       SpvDecorationFuncParamAttr;
          });
    }
  }

  // Remove prototypes of imported functions
  for (const auto& linking_entry : linkings_to_do) {
    for (auto func_iter = linked_context->module()->begin();
         func_iter != linked_context->module()->end();) {
      if (func_iter->result_id() == linking_entry.imported_symbol.id)
        func_iter = func_iter.Erase();
      else
        ++func_iter;
    }
  }

  // Remove declarations of imported variables
  for (const auto& linking_entry : linkings_to_do) {
    auto next = linked_context->types_values_begin();
    for (auto inst = next; inst != linked_context->types_values_end();
         inst = next) {
      ++next;
      if (inst->result_id() == linking_entry.imported_symbol.id) {
        linked_context->KillInst(&*inst);
      }
    }
  }

  // If partial linkage is allowed, we need an efficient way to check whether
  // an imported ID had a corresponding export symbol. As uses of the imported
  // symbol have already been replaced by the exported symbol, use the exported
  // symbol ID.
  // TODO(pierremoreau): This will not work if the decoration is applied
  //                     through a group, but the linker does not support that
  //                     either.
  std::unordered_set<SpvId> imports;
  if (options.GetAllowPartialLinkage()) {
    imports.reserve(linkings_to_do.size());
    for (const auto& linking_entry : linkings_to_do)
      imports.emplace(linking_entry.exported_symbol.id);
  }

  // Remove import linkage attributes
  auto next = linked_context->annotation_begin();
  for (auto inst = next; inst != linked_context->annotation_end();
       inst = next) {
    ++next;
    // If this is an import annotation:
    // * if we do not allow partial linkage, remove all import annotations;
    // * otherwise, remove the annotation only if there was a corresponding
    //   export.
    if (inst->opcode() == SpvOpDecorate &&
        inst->GetSingleWordOperand(1u) == SpvDecorationLinkageAttributes &&
        inst->GetSingleWordOperand(3u) == SpvLinkageTypeImport &&
        (!options.GetAllowPartialLinkage() ||
         imports.find(inst->GetSingleWordOperand(0u)) != imports.end())) {
      linked_context->KillInst(&*inst);
    }
  }

  // Remove export linkage attributes if making an executable
  if (!options.GetCreateLibrary()) {
    next = linked_context->annotation_begin();
    for (auto inst = next; inst != linked_context->annotation_end();
         inst = next) {
      ++next;
      if (inst->opcode() == SpvOpDecorate &&
          inst->GetSingleWordOperand(1u) == SpvDecorationLinkageAttributes &&
          inst->GetSingleWordOperand(3u) == SpvLinkageTypeExport) {
        linked_context->KillInst(&*inst);
      }
    }
  }

  // Remove Linkage capability if making an executable and partial linkage is
  // not allowed
  if (!options.GetCreateLibrary() && !options.GetAllowPartialLinkage()) {
    for (auto& inst : linked_context->capabilities())
      if (inst.GetSingleWordInOperand(0u) == SpvCapabilityLinkage) {
        linked_context->KillInst(&inst);
        // The RemoveDuplicatesPass did remove duplicated capabilities, so we
        // now there aren’t more SpvCapabilityLinkage further down.
        break;
      }
  }

  return SPV_SUCCESS;
}

spv_result_t VerifyIds(const MessageConsumer& consumer,
                       opt::IRContext* linked_context) {
  std::unordered_set<uint32_t> ids;
  bool ok = true;
  linked_context->module()->ForEachInst(
      [&ids, &ok](const opt::Instruction* inst) {
        ok &= ids.insert(inst->unique_id()).second;
      });

  if (!ok) {
    consumer(SPV_MSG_INTERNAL_ERROR, "", {}, "Non-unique id in merged module");
    return SPV_ERROR_INVALID_ID;
  }

  return SPV_SUCCESS;
}

}  // namespace

spv_result_t Link(const Context& context,
                  const std::vector<std::vector<uint32_t>>& binaries,
                  std::vector<uint32_t>* linked_binary,
                  const LinkerOptions& options) {
  std::vector<const uint32_t*> binary_ptrs;
  binary_ptrs.reserve(binaries.size());
  std::vector<size_t> binary_sizes;
  binary_sizes.reserve(binaries.size());

  for (const auto& binary : binaries) {
    binary_ptrs.push_back(binary.data());
    binary_sizes.push_back(binary.size());
  }

  return Link(context, binary_ptrs.data(), binary_sizes.data(), binaries.size(),
              linked_binary, options);
}

spv_result_t Link(const Context& context, const uint32_t* const* binaries,
                  const size_t* binary_sizes, size_t num_binaries,
                  std::vector<uint32_t>* linked_binary,
                  const LinkerOptions& options) {
  spv_position_t position = {};
  const spv_context& c_context = context.CContext();
  const MessageConsumer& consumer = c_context->consumer;

  linked_binary->clear();
  if (num_binaries == 0u)
    return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_BINARY)
           << "No modules were given.";

  std::vector<std::unique_ptr<IRContext>> ir_contexts;
  std::vector<Module*> modules;
  modules.reserve(num_binaries);
  for (size_t i = 0u; i < num_binaries; ++i) {
    const uint32_t schema = binaries[i][4u];
    if (schema != 0u) {
      position.index = 4u;
      return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_BINARY)
             << "Schema is non-zero for module " << i << ".";
    }

    std::unique_ptr<IRContext> ir_context = BuildModule(
        c_context->target_env, consumer, binaries[i], binary_sizes[i]);
    if (ir_context == nullptr)
      return DiagnosticStream(position, consumer, "", SPV_ERROR_INVALID_BINARY)
             << "Failed to build a module out of " << ir_contexts.size() << ".";
    modules.push_back(ir_context->module());
    ir_contexts.push_back(std::move(ir_context));
  }

  // Phase 1: Shift the IDs used in each binary so that they occupy a disjoint
  //          range from the other binaries, and compute the new ID bound.
  uint32_t max_id_bound = 0u;
  spv_result_t res = ShiftIdsInModules(consumer, &modules, &max_id_bound);
  if (res != SPV_SUCCESS) return res;

  // Phase 2: Generate the header
  opt::ModuleHeader header;
  res = GenerateHeader(consumer, modules, max_id_bound, &header);
  if (res != SPV_SUCCESS) return res;
  IRContext linked_context(c_context->target_env, consumer);
  linked_context.module()->SetHeader(header);

  // Phase 3: Merge all the binaries into a single one.
  AssemblyGrammar grammar(c_context);
  res = MergeModules(consumer, modules, grammar, &linked_context);
  if (res != SPV_SUCCESS) return res;

  if (options.GetVerifyIds()) {
    res = VerifyIds(consumer, &linked_context);
    if (res != SPV_SUCCESS) return res;
  }

  // Phase 4: Find the import/export pairs
  LinkageTable linkings_to_do;
  res = GetImportExportPairs(consumer, linked_context,
                             *linked_context.get_def_use_mgr(),
                             *linked_context.get_decoration_mgr(),
                             options.GetAllowPartialLinkage(), &linkings_to_do);
  if (res != SPV_SUCCESS) return res;

  // Phase 5: Ensure the import and export have the same types and decorations.
  res =
      CheckImportExportCompatibility(consumer, linkings_to_do, &linked_context);
  if (res != SPV_SUCCESS) return res;

  // Phase 6: Remove duplicates
  PassManager manager;
  manager.SetMessageConsumer(consumer);
  manager.AddPass<RemoveDuplicatesPass>();
  opt::Pass::Status pass_res = manager.Run(&linked_context);
  if (pass_res == opt::Pass::Status::Failure) return SPV_ERROR_INVALID_DATA;

  // Phase 7: Rematch import variables/functions to export variables/functions
  for (const auto& linking_entry : linkings_to_do)
    linked_context.ReplaceAllUsesWith(linking_entry.imported_symbol.id,
                                      linking_entry.exported_symbol.id);

  // Phase 8: Remove linkage specific instructions, such as import/export
  // attributes, linkage capability, etc. if applicable
  res = RemoveLinkageSpecificInstructions(consumer, options, linkings_to_do,
                                          linked_context.get_decoration_mgr(),
                                          &linked_context);
  if (res != SPV_SUCCESS) return res;

  // Phase 9: Compact the IDs used in the module
  manager.AddPass<opt::CompactIdsPass>();
  pass_res = manager.Run(&linked_context);
  if (pass_res == opt::Pass::Status::Failure) return SPV_ERROR_INVALID_DATA;

  // Phase 10: Output the module
  linked_context.module()->ToBinary(linked_binary, true);

  return SPV_SUCCESS;
}

}  // namespace spvtools
