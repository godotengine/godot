// Copyright (c) 2016 Google Inc.
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

#ifndef TEST_OPT_ASSEMBLY_BUILDER_H_
#define TEST_OPT_ASSEMBLY_BUILDER_H_

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace spvtools {
namespace opt {

// A simple SPIR-V assembly code builder for test uses. It builds an SPIR-V
// assembly module from vectors of assembly strings. It allows users to add
// instructions to the main function and the type-constants-globals section
// directly. It relies on OpName instructions and friendly-name disassembling
// to keep the ID names unchanged after assembling.
//
// An assembly module is divided into several sections, matching with the
// SPIR-V Logical Layout:
//  Global Preamble:
//    OpCapability instructions;
//    OpExtension instructions and OpExtInstImport instructions;
//    OpMemoryModel instruction;
//    OpEntryPoint and OpExecutionMode instruction;
//    OpString, OpSourceExtension, OpSource and OpSourceContinued instructions.
//  Names:
//    OpName instructions.
//  Annotations:
//    OpDecorate, OpMemberDecorate, OpGroupDecorate, OpGroupMemberDecorate and
//    OpDecorationGroup.
//  Types, Constants and Global variables:
//    Types, constants and global variables declaration instructions.
//  Main Function:
//    Main function instructions.
//  Main Function Postamble:
//    The return and function end instructions.
//
// The assembly code is built by concatenating all the strings in the above
// sections.
//
// Users define the contents in section <Type, Constants and Global Variables>
// and <Main Function>. The <Names> section is to hold the names for IDs to
// keep them unchanged before and after assembling. All defined IDs to be added
// to this code builder will be assigned with a global name through OpName
// instruction. The name is extracted from the definition instruction.
//  E.g. adding instruction: %var_a = OpConstant %int 2, will also add an
//  instruction: OpName %var_a, "var_a".
//
// Note that the name must not be used on more than one defined IDs and
// friendly-name disassembling must be enabled so that OpName instructions will
// be respected.
class AssemblyBuilder {
  // The base ID value for spec constants.
  static const uint32_t SPEC_ID_BASE = 200;

 public:
  // Initalize a minimal SPIR-V assembly code as the template. The minimal
  // module contains an empty main function and some predefined names for the
  // main function.
  AssemblyBuilder()
      : spec_id_counter_(SPEC_ID_BASE),
        global_preamble_({
            // clang-format off
                  "OpCapability Shader",
                  "OpCapability Float64",
             "%1 = OpExtInstImport \"GLSL.std.450\"",
                  "OpMemoryModel Logical GLSL450",
                  "OpEntryPoint Vertex %main \"main\"",
            // clang-format on
        }),
        names_(),
        annotations_(),
        types_consts_globals_(),
        main_func_(),
        main_func_postamble_({
            "OpReturn",
            "OpFunctionEnd",
        }) {
    AppendTypesConstantsGlobals({
        "%void = OpTypeVoid",
        "%main_func_type = OpTypeFunction %void",
    });
    AppendInMain({
        "%main = OpFunction %void None %main_func_type",
        "%main_func_entry_block = OpLabel",
    });
  }

  // Appends OpName instructions to this builder. Instrcution strings that do
  // not start with 'OpName ' will be skipped. Returns the references of this
  // assembly builder.
  AssemblyBuilder& AppendNames(const std::vector<std::string>& vec_asm_code) {
    for (auto& inst_str : vec_asm_code) {
      if (inst_str.find("OpName ") == 0) {
        names_.push_back(inst_str);
      }
    }
    return *this;
  }

  // Appends instructions to the types-constants-globals section and returns
  // the reference of this assembly builder. IDs defined in the given code will
  // be added to the Names section and then be registered with OpName
  // instruction. Corresponding decoration instruction will be added for spec
  // constants defined with opcode: 'OpSpecConstant'.
  AssemblyBuilder& AppendTypesConstantsGlobals(
      const std::vector<std::string>& vec_asm_code) {
    AddNamesForResultIDsIn(vec_asm_code);
    // Check spec constants defined with OpSpecConstant.
    for (auto& inst_str : vec_asm_code) {
      if (inst_str.find("= OpSpecConstant ") != std::string::npos ||
          inst_str.find("= OpSpecConstantTrue ") != std::string::npos ||
          inst_str.find("= OpSpecConstantFalse ") != std::string::npos) {
        AddSpecIDFor(GetResultIDName(inst_str));
      }
    }
    types_consts_globals_.insert(types_consts_globals_.end(),
                                 vec_asm_code.begin(), vec_asm_code.end());
    return *this;
  }

  // Appends instructions to the main function block, which is already labelled
  // with "main_func_entry_block". Returns the reference of this assembly
  // builder. IDs defined in the given code will be added to the Names section
  // and then be registered with OpName instruction.
  AssemblyBuilder& AppendInMain(const std::vector<std::string>& vec_asm_code) {
    AddNamesForResultIDsIn(vec_asm_code);
    main_func_.insert(main_func_.end(), vec_asm_code.begin(),
                      vec_asm_code.end());
    return *this;
  }

  // Appends annotation instructions to the annotation section, and returns the
  // reference of this assembly builder.
  AssemblyBuilder& AppendAnnotations(
      const std::vector<std::string>& vec_annotations) {
    annotations_.insert(annotations_.end(), vec_annotations.begin(),
                        vec_annotations.end());
    return *this;
  }

  // Pre-pends string to the preamble of the module. Useful for EFFCEE checks.
  AssemblyBuilder& PrependPreamble(const std::vector<std::string>& preamble) {
    preamble_.insert(preamble_.end(), preamble.begin(), preamble.end());
    return *this;
  }

  // Get the SPIR-V assembly code as string.
  std::string GetCode() const {
    std::ostringstream ss;
    for (const auto& line : preamble_) {
      ss << line << std::endl;
    }
    for (const auto& line : global_preamble_) {
      ss << line << std::endl;
    }
    for (const auto& line : names_) {
      ss << line << std::endl;
    }
    for (const auto& line : annotations_) {
      ss << line << std::endl;
    }
    for (const auto& line : types_consts_globals_) {
      ss << line << std::endl;
    }
    for (const auto& line : main_func_) {
      ss << line << std::endl;
    }
    for (const auto& line : main_func_postamble_) {
      ss << line << std::endl;
    }
    return ss.str();
  }

 private:
  // Adds a given name to the Name section with OpName. If the given name has
  // been added before, does nothing.
  void AddOpNameIfNotExist(const std::string& id_name) {
    if (!used_names_.count(id_name)) {
      std::stringstream opname_inst;
      opname_inst << "OpName "
                  << "%" << id_name << " \"" << id_name << "\"";
      names_.emplace_back(opname_inst.str());
      used_names_.insert(id_name);
    }
  }

  // Adds the names in a vector of assembly code strings to the Names section.
  // If a '=' sign is found in an instruction, this instruction will be treated
  // as an ID defining instruction. The ID name used in the instruction will be
  // extracted and added to the Names section.
  void AddNamesForResultIDsIn(const std::vector<std::string>& vec_asm_code) {
    for (const auto& line : vec_asm_code) {
      std::string name = GetResultIDName(line);
      if (!name.empty()) {
        AddOpNameIfNotExist(name);
      }
    }
  }

  // Adds an OpDecorate SpecId instruction for the given ID name.
  void AddSpecIDFor(const std::string& id_name) {
    std::stringstream decorate_inst;
    decorate_inst << "OpDecorate "
                  << "%" << id_name << " SpecId " << spec_id_counter_;
    spec_id_counter_ += 1;
    annotations_.emplace_back(decorate_inst.str());
  }

  // Extracts the ID name from a SPIR-V assembly instruction string. If the
  // instruction is an ID-defining instruction (has result ID), returns the
  // name of the result ID in string. If the instruction does not have result
  // ID, returns an empty string.
  std::string GetResultIDName(const std::string inst_str) {
    std::string name;
    if (inst_str.find('=') != std::string::npos) {
      size_t assign_sign = inst_str.find('=');
      name = inst_str.substr(0, assign_sign);
      name.erase(remove_if(name.begin(), name.end(),
                           [](char c) { return c == ' ' || c == '%'; }),
                 name.end());
    }
    return name;
  }

  uint32_t spec_id_counter_;
  // User-defined preamble.
  std::vector<std::string> preamble_;
  // The vector that contains common preambles shared across all test SPIR-V
  // code.
  std::vector<std::string> global_preamble_;
  // The vector that contains OpName instructions.
  std::vector<std::string> names_;
  // The vector that contains annotation instructions.
  std::vector<std::string> annotations_;
  // The vector that contains the code to declare types, constants and global
  // variables (aka. the Types-Constants-Globals section).
  std::vector<std::string> types_consts_globals_;
  // The vector that contains the code in main function's entry block.
  std::vector<std::string> main_func_;
  // The vector that contains the postamble of main function body.
  std::vector<std::string> main_func_postamble_;
  // All of the defined variable names.
  std::unordered_set<std::string> used_names_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // TEST_OPT_ASSEMBLY_BUILDER_H_
