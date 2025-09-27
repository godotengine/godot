// Copyright (c) 2022 Google LLC
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

#ifndef SOURCE_OPT_INTERFACE_VAR_SROA_H_
#define SOURCE_OPT_INTERFACE_VAR_SROA_H_

#include <unordered_set>

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
//
// Note that the current implementation of this pass covers only store, load,
// access chain instructions for the interface variables. Supporting other types
// of instructions is a future work.
class InterfaceVariableScalarReplacement : public Pass {
 public:
  InterfaceVariableScalarReplacement() {}

  const char* name() const override {
    return "interface-variable-scalar-replacement";
  }
  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDecorations | IRContext::kAnalysisDefUse |
           IRContext::kAnalysisConstants | IRContext::kAnalysisTypes;
  }

 private:
  // A struct containing components of a composite variable. If the composite
  // consists of multiple or recursive components, |component_variable| is
  // nullptr and |nested_composite_components| keeps the components. If it has a
  // single component, |nested_composite_components| is empty and
  // |component_variable| is the component. Note that each element of
  // |nested_composite_components| has the NestedCompositeComponents struct as
  // its type that can recursively keep the components.
  struct NestedCompositeComponents {
    NestedCompositeComponents() : component_variable(nullptr) {}

    bool HasMultipleComponents() const {
      return !nested_composite_components.empty();
    }

    const std::vector<NestedCompositeComponents>& GetComponents() const {
      return nested_composite_components;
    }

    void AddComponent(const NestedCompositeComponents& component) {
      nested_composite_components.push_back(component);
    }

    Instruction* GetComponentVariable() const { return component_variable; }

    void SetSingleComponentVariable(Instruction* var) {
      component_variable = var;
    }

   private:
    std::vector<NestedCompositeComponents> nested_composite_components;
    Instruction* component_variable;
  };

  // Collects all interface variables used by the |entry_point|.
  std::vector<Instruction*> CollectInterfaceVariables(Instruction& entry_point);

  // Returns whether |var| has the extra arrayness for the entry point
  // |entry_point| or not.
  bool HasExtraArrayness(Instruction& entry_point, Instruction* var);

  // Finds a Location BuiltIn decoration of |var| and returns it via
  // |location|. Returns true whether the location exists or not.
  bool GetVariableLocation(Instruction* var, uint32_t* location);

  // Finds a Component BuiltIn decoration of |var| and returns it via
  // |component|. Returns true whether the component exists or not.
  bool GetVariableComponent(Instruction* var, uint32_t* component);

  // Returns the interface variable instruction whose result id is
  // |interface_var_id|.
  Instruction* GetInterfaceVariable(uint32_t interface_var_id);

  // Returns the type of |var| as an instruction.
  Instruction* GetTypeOfVariable(Instruction* var);

  // Replaces an interface variable |interface_var| whose type is
  // |interface_var_type| with scalars and returns whether it succeeds or not.
  // |location| is the value of Location Decoration for |interface_var|.
  // |component| is the value of Component Decoration for |interface_var|.
  // If |extra_array_length| is 0, it means |interface_var| has a Patch
  // decoration. Otherwise, |extra_array_length| denotes the length of the extra
  // array of |interface_var|.
  bool ReplaceInterfaceVariableWithScalars(Instruction* interface_var,
                                           Instruction* interface_var_type,
                                           uint32_t location,
                                           uint32_t component,
                                           uint32_t extra_array_length);

  // Creates scalar variables with the storage classe |storage_class| to replace
  // an interface variable whose type is |interface_var_type|. If
  // |extra_array_length| is not zero, adds the extra arrayness to the created
  // scalar variables.
  NestedCompositeComponents CreateScalarInterfaceVarsForReplacement(
      Instruction* interface_var_type, spv::StorageClass storage_class,
      uint32_t extra_array_length);

  // Creates scalar variables with the storage classe |storage_class| to replace
  // the interface variable whose type is OpTypeArray |interface_var_type| with.
  // If |extra_array_length| is not zero, adds the extra arrayness to all the
  // scalar variables.
  NestedCompositeComponents CreateScalarInterfaceVarsForArray(
      Instruction* interface_var_type, spv::StorageClass storage_class,
      uint32_t extra_array_length);

  // Creates scalar variables with the storage classe |storage_class| to replace
  // the interface variable whose type is OpTypeMatrix |interface_var_type|
  // with. If |extra_array_length| is not zero, adds the extra arrayness to all
  // the scalar variables.
  NestedCompositeComponents CreateScalarInterfaceVarsForMatrix(
      Instruction* interface_var_type, spv::StorageClass storage_class,
      uint32_t extra_array_length);

  // Recursively adds Location and Component decorations to variables in
  // |vars| with |location| and |component|. Increases |location| by one after
  // it actually adds Location and Component decorations for a variable.
  void AddLocationAndComponentDecorations(const NestedCompositeComponents& vars,
                                          uint32_t* location,
                                          uint32_t component);

  // Replaces the interface variable |interface_var| with
  // |scalar_interface_vars| and returns whether it succeeds or not.
  // |extra_arrayness| is the extra arrayness of the interface variable.
  // |scalar_interface_vars| contains the nested variables to replace the
  // interface variable with.
  bool ReplaceInterfaceVarWith(
      Instruction* interface_var, uint32_t extra_arrayness,
      const NestedCompositeComponents& scalar_interface_vars);

  // Replaces |interface_var| in the operands of instructions
  // |interface_var_users| with |scalar_interface_vars|. This is a recursive
  // method and |interface_var_component_indices| is used to specify which
  // recursive component of |interface_var| is replaced. Returns composite
  // construct instructions to be replaced with load instructions of
  // |interface_var_users| via |loads_to_composites|. Returns composite
  // construct instructions to be replaced with load instructions of access
  // chain instructions in |interface_var_users| via
  // |loads_for_access_chain_to_composites|.
  bool ReplaceComponentsOfInterfaceVarWith(
      Instruction* interface_var,
      const std::vector<Instruction*>& interface_var_users,
      const NestedCompositeComponents& scalar_interface_vars,
      std::vector<uint32_t>& interface_var_component_indices,
      const uint32_t* extra_array_index,
      std::unordered_map<Instruction*, Instruction*>* loads_to_composites,
      std::unordered_map<Instruction*, Instruction*>*
          loads_for_access_chain_to_composites);

  // Replaces |interface_var| in the operands of instructions
  // |interface_var_users| with |components| that is a vector of components for
  // the interface variable |interface_var|. This is a recursive method and
  // |interface_var_component_indices| is used to specify which recursive
  // component of |interface_var| is replaced. Returns composite construct
  // instructions to be replaced with load instructions of |interface_var_users|
  // via |loads_to_composites|. Returns composite construct instructions to be
  // replaced with load instructions of access chain instructions in
  // |interface_var_users| via |loads_for_access_chain_to_composites|.
  bool ReplaceMultipleComponentsOfInterfaceVarWith(
      Instruction* interface_var,
      const std::vector<Instruction*>& interface_var_users,
      const std::vector<NestedCompositeComponents>& components,
      std::vector<uint32_t>& interface_var_component_indices,
      const uint32_t* extra_array_index,
      std::unordered_map<Instruction*, Instruction*>* loads_to_composites,
      std::unordered_map<Instruction*, Instruction*>*
          loads_for_access_chain_to_composites);

  // Replaces a component of |interface_var| that is used as an operand of
  // instruction |interface_var_user| with |scalar_var|.
  // |interface_var_component_indices| is a vector of recursive indices for
  // which recursive component of |interface_var| is replaced. If
  // |interface_var_user| is a load, returns the component value via
  // |loads_to_component_values|. If |interface_var_user| is an access chain,
  // returns the component value for loads of |interface_var_user| via
  // |loads_for_access_chain_to_component_values|.
  bool ReplaceComponentOfInterfaceVarWith(
      Instruction* interface_var, Instruction* interface_var_user,
      Instruction* scalar_var,
      const std::vector<uint32_t>& interface_var_component_indices,
      const uint32_t* extra_array_index,
      std::unordered_map<Instruction*, Instruction*>* loads_to_component_values,
      std::unordered_map<Instruction*, Instruction*>*
          loads_for_access_chain_to_component_values);

  // Creates instructions to load |scalar_var| and inserts them before
  // |insert_before|. If |extra_array_index| is not null, they load
  // |extra_array_index| th component of |scalar_var| instead of |scalar_var|
  // itself.
  Instruction* LoadScalarVar(Instruction* scalar_var,
                             const uint32_t* extra_array_index,
                             Instruction* insert_before);

  // Creates instructions to load an access chain to |var| and inserts them
  // before |insert_before|. |Indexes| will be Indexes operand of the access
  // chain.
  Instruction* LoadAccessChainToVar(Instruction* var,
                                    const std::vector<uint32_t>& indexes,
                                    Instruction* insert_before);

  // Creates instructions to store a component of an aggregate whose id is
  // |value_id| to an access chain to |scalar_var| and inserts the created
  // instructions before |insert_before|. To get the component, recursively
  // traverses the aggregate with |component_indices| as indexes.
  // Numbers in |access_chain_indices| are the Indexes operand of the access
  // chain to |scalar_var|
  void StoreComponentOfValueToAccessChainToScalarVar(
      uint32_t value_id, const std::vector<uint32_t>& component_indices,
      Instruction* scalar_var,
      const std::vector<uint32_t>& access_chain_indices,
      Instruction* insert_before);

  // Creates instructions to store a component of an aggregate whose id is
  // |value_id| to |scalar_var| and inserts the created instructions before
  // |insert_before|. To get the component, recursively traverses the aggregate
  // using |extra_array_index| and |component_indices| as indexes.
  void StoreComponentOfValueToScalarVar(
      uint32_t value_id, const std::vector<uint32_t>& component_indices,
      Instruction* scalar_var, const uint32_t* extra_array_index,
      Instruction* insert_before);

  // Creates instructions to store a component of an aggregate whose id is
  // |value_id| to |ptr| and inserts the created instructions before
  // |insert_before|. To get the component, recursively traverses the aggregate
  // using |extra_array_index| and |component_indices| as indexes.
  // |component_type_id| is the id of the type instruction of the component.
  void StoreComponentOfValueTo(uint32_t component_type_id, uint32_t value_id,
                               const std::vector<uint32_t>& component_indices,
                               Instruction* ptr,
                               const uint32_t* extra_array_index,
                               Instruction* insert_before);

  // Creates new OpCompositeExtract with |type_id| for Result Type,
  // |composite_id| for Composite operand, and |indexes| for Indexes operands.
  // If |extra_first_index| is not nullptr, uses it as the first Indexes
  // operand.
  Instruction* CreateCompositeExtract(uint32_t type_id, uint32_t composite_id,
                                      const std::vector<uint32_t>& indexes,
                                      const uint32_t* extra_first_index);

  // Creates a new OpLoad whose Result Type is |type_id| and Pointer operand is
  // |ptr|. Inserts the new instruction before |insert_before|.
  Instruction* CreateLoad(uint32_t type_id, Instruction* ptr,
                          Instruction* insert_before);

  // Clones an annotation instruction |annotation_inst| and sets the target
  // operand of the new annotation instruction as |var_id|.
  void CloneAnnotationForVariable(Instruction* annotation_inst,
                                  uint32_t var_id);

  // Replaces the interface variable |interface_var| in the operands of the
  // entry point |entry_point| with |scalar_var_id|. If it cannot find
  // |interface_var| from the operands of the entry point |entry_point|, adds
  // |scalar_var_id| as an operand of the entry point |entry_point|.
  bool ReplaceInterfaceVarInEntryPoint(Instruction* interface_var,
                                       Instruction* entry_point,
                                       uint32_t scalar_var_id);

  // Creates an access chain instruction whose Base operand is |var| and Indexes
  // operand is |index|. |component_type_id| is the id of the type instruction
  // that is the type of component. Inserts the new access chain before
  // |insert_before|.
  Instruction* CreateAccessChainWithIndex(uint32_t component_type_id,
                                          Instruction* var, uint32_t index,
                                          Instruction* insert_before);

  // Returns the pointee type of the type of variable |var|.
  uint32_t GetPointeeTypeIdOfVar(Instruction* var);

  // Replaces the access chain |access_chain| and its users with a new access
  // chain that points |scalar_var| as the Base operand having
  // |interface_var_component_indices| as Indexes operands and users of the new
  // access chain. When some of the users are load instructions, returns the
  // original load instruction to the new instruction that loads a component of
  // the original load value via |loads_to_component_values|.
  void ReplaceAccessChainWith(
      Instruction* access_chain,
      const std::vector<uint32_t>& interface_var_component_indices,
      Instruction* scalar_var,
      std::unordered_map<Instruction*, Instruction*>*
          loads_to_component_values);

  // Assuming that |access_chain| is an access chain instruction whose Base
  // operand is |base_access_chain|, replaces the operands of |access_chain|
  // with operands of |base_access_chain| and Indexes operands of
  // |access_chain|.
  void UseBaseAccessChainForAccessChain(Instruction* access_chain,
                                        Instruction* base_access_chain);

  // Creates composite construct instructions for load instructions that are the
  // keys of |loads_to_component_values| if no such composite construct
  // instructions exist. Adds a component of the composite as an operand of the
  // created composite construct instruction. Each value of
  // |loads_to_component_values| is the component. Returns the created composite
  // construct instructions using |loads_to_composites|. |depth_to_component| is
  // the number of recursive access steps to get the component from the
  // composite.
  void AddComponentsToCompositesForLoads(
      const std::unordered_map<Instruction*, Instruction*>&
          loads_to_component_values,
      std::unordered_map<Instruction*, Instruction*>* loads_to_composites,
      uint32_t depth_to_component);

  // Creates a composite construct instruction for a component of the value of
  // instruction |load| in |depth_to_component| th recursive depth and inserts
  // it after |load|.
  Instruction* CreateCompositeConstructForComponentOfLoad(
      Instruction* load, uint32_t depth_to_component);

  // Creates a new access chain instruction that points to variable |var| whose
  // type is the instruction with |var_type_id| and inserts it before
  // |insert_before|. The new access chain will have |index_ids| for Indexes
  // operands. Returns the type id of the component that is pointed by the new
  // access chain via |component_type_id|.
  Instruction* CreateAccessChainToVar(uint32_t var_type_id, Instruction* var,
                                      const std::vector<uint32_t>& index_ids,
                                      Instruction* insert_before,
                                      uint32_t* component_type_id);

  // Returns the result id of OpTypeArray instrunction whose Element Type
  // operand is |elem_type_id| and Length operand is |array_length|.
  uint32_t GetArrayType(uint32_t elem_type_id, uint32_t array_length);

  // Returns the result id of OpTypePointer instrunction whose Type
  // operand is |type_id| and Storage Class operand is |storage_class|.
  uint32_t GetPointerType(uint32_t type_id, spv::StorageClass storage_class);

  // Kills an instrunction |inst| and its users.
  void KillInstructionAndUsers(Instruction* inst);

  // Kills a vector of instrunctions |insts| and their users.
  void KillInstructionsAndUsers(const std::vector<Instruction*>& insts);

  // Kills all OpDecorate instructions for Location and Component of the
  // variable whose id is |var_id|.
  void KillLocationAndComponentDecorations(uint32_t var_id);

  // If |var| has the extra arrayness for an entry point, reports an error and
  // returns true. Otherwise, returns false.
  bool ReportErrorIfHasExtraArraynessForOtherEntry(Instruction* var);

  // If |var| does not have the extra arrayness for an entry point, reports an
  // error and returns true. Otherwise, returns false.
  bool ReportErrorIfHasNoExtraArraynessForOtherEntry(Instruction* var);

  // If |interface_var| has the extra arrayness for an entry point but it does
  // not have one for another entry point, reports an error and returns false.
  // Otherwise, returns true. |has_extra_arrayness| denotes whether it has an
  // extra arrayness for an entry point or not.
  bool CheckExtraArraynessConflictBetweenEntries(Instruction* interface_var,
                                                 bool has_extra_arrayness);

  // Conducts the scalar replacement for the interface variables used by the
  // |entry_point|.
  Pass::Status ReplaceInterfaceVarsWithScalars(Instruction& entry_point);

  // A set of interface variable ids that were already removed from operands of
  // the entry point.
  std::unordered_set<uint32_t>
      interface_vars_removed_from_entry_point_operands_;

  // A mapping from ids of new composite construct instructions that load
  // instructions are replaced with to the recursive depth of the component of
  // load that the new component construct instruction is used for.
  std::unordered_map<uint32_t, uint32_t> composite_ids_to_component_depths;

  // A set of interface variables with the extra arrayness for any of the entry
  // points.
  std::unordered_set<Instruction*> vars_with_extra_arrayness;

  // A set of interface variables without the extra arrayness for any of the
  // entry points.
  std::unordered_set<Instruction*> vars_without_extra_arrayness;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_INTERFACE_VAR_SROA_H_
