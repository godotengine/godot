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

#ifndef SOURCE_OPT_IR_CONTEXT_H_
#define SOURCE_OPT_IR_CONTEXT_H_

#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "source/assembly_grammar.h"
#include "source/enum_string_mapping.h"
#include "source/opt/cfg.h"
#include "source/opt/constants.h"
#include "source/opt/debug_info_manager.h"
#include "source/opt/decoration_manager.h"
#include "source/opt/def_use_manager.h"
#include "source/opt/dominator_analysis.h"
#include "source/opt/feature_manager.h"
#include "source/opt/fold.h"
#include "source/opt/liveness.h"
#include "source/opt/loop_descriptor.h"
#include "source/opt/module.h"
#include "source/opt/register_pressure.h"
#include "source/opt/scalar_analysis.h"
#include "source/opt/struct_cfg_analysis.h"
#include "source/opt/type_manager.h"
#include "source/opt/value_number_table.h"
#include "source/util/make_unique.h"
#include "source/util/string_utils.h"

namespace spvtools {
namespace opt {

class IRContext {
 public:
  // Available analyses.
  //
  // When adding a new analysis:
  //
  // 1. Enum values should be powers of 2. These are cast into uint32_t
  //    bitmasks, so we can have at most 31 analyses represented.
  //
  // 2. Make sure it gets invalidated or preserved by IRContext methods that add
  //    or remove IR elements (e.g., KillDef, KillInst, ReplaceAllUsesWith).
  //
  // 3. Add handling code in BuildInvalidAnalyses and InvalidateAnalyses
  enum Analysis {
    kAnalysisNone = 0 << 0,
    kAnalysisBegin = 1 << 0,
    kAnalysisDefUse = kAnalysisBegin,
    kAnalysisInstrToBlockMapping = 1 << 1,
    kAnalysisDecorations = 1 << 2,
    kAnalysisCombinators = 1 << 3,
    kAnalysisCFG = 1 << 4,
    kAnalysisDominatorAnalysis = 1 << 5,
    kAnalysisLoopAnalysis = 1 << 6,
    kAnalysisNameMap = 1 << 7,
    kAnalysisScalarEvolution = 1 << 8,
    kAnalysisRegisterPressure = 1 << 9,
    kAnalysisValueNumberTable = 1 << 10,
    kAnalysisStructuredCFG = 1 << 11,
    kAnalysisBuiltinVarId = 1 << 12,
    kAnalysisIdToFuncMapping = 1 << 13,
    kAnalysisConstants = 1 << 14,
    kAnalysisTypes = 1 << 15,
    kAnalysisDebugInfo = 1 << 16,
    kAnalysisLiveness = 1 << 17,
    kAnalysisEnd = 1 << 18
  };

  using ProcessFunction = std::function<bool(Function*)>;

  friend inline Analysis operator|(Analysis lhs, Analysis rhs);
  friend inline Analysis& operator|=(Analysis& lhs, Analysis rhs);
  friend inline Analysis operator<<(Analysis a, int shift);
  friend inline Analysis& operator<<=(Analysis& a, int shift);

  // Creates an |IRContext| that contains an owned |Module|
  IRContext(spv_target_env env, MessageConsumer c)
      : syntax_context_(spvContextCreate(env)),
        grammar_(syntax_context_),
        unique_id_(0),
        module_(new Module()),
        consumer_(std::move(c)),
        def_use_mgr_(nullptr),
        feature_mgr_(nullptr),
        valid_analyses_(kAnalysisNone),
        constant_mgr_(nullptr),
        type_mgr_(nullptr),
        id_to_name_(nullptr),
        max_id_bound_(kDefaultMaxIdBound),
        preserve_bindings_(false),
        preserve_spec_constants_(false) {
    SetContextMessageConsumer(syntax_context_, consumer_);
    module_->SetContext(this);
  }

  IRContext(spv_target_env env, std::unique_ptr<Module>&& m, MessageConsumer c)
      : syntax_context_(spvContextCreate(env)),
        grammar_(syntax_context_),
        unique_id_(0),
        module_(std::move(m)),
        consumer_(std::move(c)),
        def_use_mgr_(nullptr),
        feature_mgr_(nullptr),
        valid_analyses_(kAnalysisNone),
        type_mgr_(nullptr),
        id_to_name_(nullptr),
        max_id_bound_(kDefaultMaxIdBound),
        preserve_bindings_(false),
        preserve_spec_constants_(false) {
    SetContextMessageConsumer(syntax_context_, consumer_);
    module_->SetContext(this);
    InitializeCombinators();
  }

  ~IRContext() { spvContextDestroy(syntax_context_); }

  Module* module() const { return module_.get(); }

  // Returns a vector of pointers to constant-creation instructions in this
  // context.
  inline std::vector<Instruction*> GetConstants();
  inline std::vector<const Instruction*> GetConstants() const;

  // Iterators for annotation instructions contained in this context.
  inline Module::inst_iterator annotation_begin();
  inline Module::inst_iterator annotation_end();
  inline IteratorRange<Module::inst_iterator> annotations();
  inline IteratorRange<Module::const_inst_iterator> annotations() const;

  // Iterators for capabilities instructions contained in this module.
  inline Module::inst_iterator capability_begin();
  inline Module::inst_iterator capability_end();
  inline IteratorRange<Module::inst_iterator> capabilities();
  inline IteratorRange<Module::const_inst_iterator> capabilities() const;

  // Iterators for extensions instructions contained in this module.
  inline Module::inst_iterator extension_begin();
  inline Module::inst_iterator extension_end();
  inline IteratorRange<Module::inst_iterator> extensions();
  inline IteratorRange<Module::const_inst_iterator> extensions() const;

  // Iterators for types, constants and global variables instructions.
  inline Module::inst_iterator types_values_begin();
  inline Module::inst_iterator types_values_end();
  inline IteratorRange<Module::inst_iterator> types_values();
  inline IteratorRange<Module::const_inst_iterator> types_values() const;

  // Iterators for ext_inst import instructions contained in this module.
  inline Module::inst_iterator ext_inst_import_begin();
  inline Module::inst_iterator ext_inst_import_end();
  inline IteratorRange<Module::inst_iterator> ext_inst_imports();
  inline IteratorRange<Module::const_inst_iterator> ext_inst_imports() const;

  // There are several kinds of debug instructions, according to where they can
  // appear in the logical layout of a module:
  //  - Section 7a:  OpString, OpSourceExtension, OpSource, OpSourceContinued
  //  - Section 7b:  OpName, OpMemberName
  //  - Section 7c:  OpModuleProcessed
  //  - Mostly anywhere: OpLine and OpNoLine
  //

  // Iterators for debug 1 instructions (excluding OpLine & OpNoLine) contained
  // in this module.  These are for layout section 7a.
  inline Module::inst_iterator debug1_begin();
  inline Module::inst_iterator debug1_end();
  inline IteratorRange<Module::inst_iterator> debugs1();
  inline IteratorRange<Module::const_inst_iterator> debugs1() const;

  // Iterators for debug 2 instructions (excluding OpLine & OpNoLine) contained
  // in this module.  These are for layout section 7b.
  inline Module::inst_iterator debug2_begin();
  inline Module::inst_iterator debug2_end();
  inline IteratorRange<Module::inst_iterator> debugs2();
  inline IteratorRange<Module::const_inst_iterator> debugs2() const;

  // Iterators for debug 3 instructions (excluding OpLine & OpNoLine) contained
  // in this module.  These are for layout section 7c.
  inline Module::inst_iterator debug3_begin();
  inline Module::inst_iterator debug3_end();
  inline IteratorRange<Module::inst_iterator> debugs3();
  inline IteratorRange<Module::const_inst_iterator> debugs3() const;

  // Iterators for debug info instructions (excluding OpLine & OpNoLine)
  // contained in this module.  These are OpExtInst &
  // OpExtInstWithForwardRefsKHR for DebugInfo extension placed between section
  // 9 and 10.
  inline Module::inst_iterator ext_inst_debuginfo_begin();
  inline Module::inst_iterator ext_inst_debuginfo_end();
  inline IteratorRange<Module::inst_iterator> ext_inst_debuginfo();
  inline IteratorRange<Module::const_inst_iterator> ext_inst_debuginfo() const;

  // Add |capability| to the module, if it is not already enabled.
  inline void AddCapability(spv::Capability capability);
  // Appends a capability instruction to this module.
  inline void AddCapability(std::unique_ptr<Instruction>&& c);
  // Removes instruction declaring `capability` from this module.
  // Returns true if the capability was removed, false otherwise.
  bool RemoveCapability(spv::Capability capability);

  // Appends an extension instruction to this module.
  inline void AddExtension(const std::string& ext_name);
  inline void AddExtension(std::unique_ptr<Instruction>&& e);
  // Removes instruction declaring `extension` from this module.
  // Returns true if the extension was removed, false otherwise.
  bool RemoveExtension(Extension extension);

  // Appends an extended instruction set instruction to this module.
  inline void AddExtInstImport(const std::string& name);
  inline void AddExtInstImport(std::unique_ptr<Instruction>&& e);
  // Set the memory model for this module.
  inline void SetMemoryModel(std::unique_ptr<Instruction>&& m);
  // Get the memory model for this module.
  inline const Instruction* GetMemoryModel() const;
  // Appends an entry point instruction to this module.
  inline void AddEntryPoint(std::unique_ptr<Instruction>&& e);
  // Appends an execution mode instruction to this module.
  inline void AddExecutionMode(std::unique_ptr<Instruction>&& e);
  // Appends a debug 1 instruction (excluding OpLine & OpNoLine) to this module.
  // "debug 1" instructions are the ones in layout section 7.a), see section
  // 2.4 Logical Layout of a Module from the SPIR-V specification.
  inline void AddDebug1Inst(std::unique_ptr<Instruction>&& d);
  // Appends a debug 2 instruction (excluding OpLine & OpNoLine) to this module.
  // "debug 2" instructions are the ones in layout section 7.b), see section
  // 2.4 Logical Layout of a Module from the SPIR-V specification.
  inline void AddDebug2Inst(std::unique_ptr<Instruction>&& d);
  // Appends a debug 3 instruction (OpModuleProcessed) to this module.
  // This is due to decision by the SPIR Working Group, pending publication.
  inline void AddDebug3Inst(std::unique_ptr<Instruction>&& d);
  // Appends a OpExtInst for DebugInfo to this module.
  inline void AddExtInstDebugInfo(std::unique_ptr<Instruction>&& d);
  // Appends an annotation instruction to this module.
  inline void AddAnnotationInst(std::unique_ptr<Instruction>&& a);
  // Appends a type-declaration instruction to this module.
  inline void AddType(std::unique_ptr<Instruction>&& t);
  // Appends a constant, global variable, or OpUndef instruction to this module.
  inline void AddGlobalValue(std::unique_ptr<Instruction>&& v);
  // Prepends a function declaration to this module.
  inline void AddFunctionDeclaration(std::unique_ptr<Function>&& f);
  // Appends a function to this module.
  inline void AddFunction(std::unique_ptr<Function>&& f);

  // Returns a pointer to a def-use manager.  If the def-use manager is
  // invalid, it is rebuilt first.
  analysis::DefUseManager* get_def_use_mgr() {
    if (!AreAnalysesValid(kAnalysisDefUse)) {
      BuildDefUseManager();
    }
    return def_use_mgr_.get();
  }

  // Returns a pointer to a liveness manager.  If the liveness manager is
  // invalid, it is rebuilt first.
  analysis::LivenessManager* get_liveness_mgr() {
    if (!AreAnalysesValid(kAnalysisLiveness)) {
      BuildLivenessManager();
    }
    return liveness_mgr_.get();
  }

  // Returns a pointer to a value number table.  If the liveness analysis is
  // invalid, it is rebuilt first.
  ValueNumberTable* GetValueNumberTable() {
    if (!AreAnalysesValid(kAnalysisValueNumberTable)) {
      BuildValueNumberTable();
    }
    return vn_table_.get();
  }

  // Returns a pointer to a StructuredCFGAnalysis.  If the analysis is invalid,
  // it is rebuilt first.
  StructuredCFGAnalysis* GetStructuredCFGAnalysis() {
    if (!AreAnalysesValid(kAnalysisStructuredCFG)) {
      BuildStructuredCFGAnalysis();
    }
    return struct_cfg_analysis_.get();
  }

  // Returns a pointer to a liveness analysis.  If the liveness analysis is
  // invalid, it is rebuilt first.
  LivenessAnalysis* GetLivenessAnalysis() {
    if (!AreAnalysesValid(kAnalysisRegisterPressure)) {
      BuildRegPressureAnalysis();
    }
    return reg_pressure_.get();
  }

  // Returns the basic block for instruction |instr|. Re-builds the instruction
  // block map, if needed.
  BasicBlock* get_instr_block(Instruction* instr) {
    if (!AreAnalysesValid(kAnalysisInstrToBlockMapping)) {
      BuildInstrToBlockMapping();
    }
    auto entry = instr_to_block_.find(instr);
    return (entry != instr_to_block_.end()) ? entry->second : nullptr;
  }

  // Returns the basic block for |id|. Re-builds the instruction block map, if
  // needed.
  //
  // |id| must be a registered definition.
  BasicBlock* get_instr_block(uint32_t id) {
    Instruction* def = get_def_use_mgr()->GetDef(id);
    return get_instr_block(def);
  }

  // Sets the basic block for |inst|. Re-builds the mapping if it has become
  // invalid.
  void set_instr_block(Instruction* inst, BasicBlock* block) {
    if (AreAnalysesValid(kAnalysisInstrToBlockMapping)) {
      instr_to_block_[inst] = block;
    }
  }

  // Returns a pointer the decoration manager.  If the decoration manager is
  // invalid, it is rebuilt first.
  analysis::DecorationManager* get_decoration_mgr() {
    if (!AreAnalysesValid(kAnalysisDecorations)) {
      BuildDecorationManager();
    }
    return decoration_mgr_.get();
  }

  // Returns a pointer to the constant manager.  If no constant manager has been
  // created yet, it creates one.  NOTE: Once created, the constant manager
  // remains active and it is never re-built.
  analysis::ConstantManager* get_constant_mgr() {
    if (!AreAnalysesValid(kAnalysisConstants)) {
      BuildConstantManager();
    }
    return constant_mgr_.get();
  }

  // Returns a pointer to the type manager.  If no type manager has been created
  // yet, it creates one. NOTE: Once created, the type manager remains active it
  // is never re-built.
  analysis::TypeManager* get_type_mgr() {
    if (!AreAnalysesValid(kAnalysisTypes)) {
      BuildTypeManager();
    }
    return type_mgr_.get();
  }

  // Returns a pointer to the debug information manager.  If no debug
  // information manager has been created yet, it creates one.
  // NOTE: Once created, the debug information manager remains active
  // it is never re-built.
  analysis::DebugInfoManager* get_debug_info_mgr() {
    if (!AreAnalysesValid(kAnalysisDebugInfo)) {
      BuildDebugInfoManager();
    }
    return debug_info_mgr_.get();
  }

  // Returns a pointer to the scalar evolution analysis. If it is invalid it
  // will be rebuilt first.
  ScalarEvolutionAnalysis* GetScalarEvolutionAnalysis() {
    if (!AreAnalysesValid(kAnalysisScalarEvolution)) {
      BuildScalarEvolutionAnalysis();
    }
    return scalar_evolution_analysis_.get();
  }

  // Build the map from the ids to the OpName and OpMemberName instruction
  // associated with it.
  inline void BuildIdToNameMap();

  // Returns a range of instrucions that contain all of the OpName and
  // OpMemberNames associated with the given id.
  inline IteratorRange<std::multimap<uint32_t, Instruction*>::iterator>
  GetNames(uint32_t id);

  // Returns an OpMemberName instruction that targets |struct_type_id| at
  // index |index|. Returns nullptr if no such instruction exists.
  // While the SPIR-V spec does not prohibit having multiple OpMemberName
  // instructions for the same structure member, it is hard to imagine a member
  // having more than one name. This method returns the first one it finds.
  inline Instruction* GetMemberName(uint32_t struct_type_id, uint32_t index);

  // Copy names from |old_id| to |new_id|. Only copy member name if index is
  // less than |max_member_index|.
  inline void CloneNames(const uint32_t old_id, const uint32_t new_id,
                         const uint32_t max_member_index = UINT32_MAX);

  // Sets the message consumer to the given |consumer|. |consumer| which will be
  // invoked every time there is a message to be communicated to the outside.
  void SetMessageConsumer(MessageConsumer c) { consumer_ = std::move(c); }

  // Returns the reference to the message consumer for this pass.
  const MessageConsumer& consumer() const { return consumer_; }

  // Rebuilds the analyses in |set| that are invalid.
  void BuildInvalidAnalyses(Analysis set);

  // Invalidates all of the analyses except for those in |preserved_analyses|.
  void InvalidateAnalysesExceptFor(Analysis preserved_analyses);

  // Invalidates the analyses marked in |analyses_to_invalidate|.
  void InvalidateAnalyses(Analysis analyses_to_invalidate);

  // Deletes the instruction defining the given |id|. Returns true on
  // success, false if the given |id| is not defined at all. This method also
  // erases the name, decorations, and definition of |id|.
  //
  // Pointers and iterators pointing to the deleted instructions become invalid.
  // However other pointers and iterators are still valid.
  bool KillDef(uint32_t id);

  // Deletes the given instruction |inst|. This method erases the
  // information of the given instruction's uses of its operands. If |inst|
  // defines a result id, its name and decorations will also be deleted.
  //
  // Pointer and iterator pointing to the deleted instructions become invalid.
  // However other pointers and iterators are still valid.
  //
  // Note that if an instruction is not in an instruction list, the memory may
  // not be safe to delete, so the instruction is turned into a OpNop instead.
  // This can happen with OpLabel.
  //
  // Returns a pointer to the instruction after |inst| or |nullptr| if no such
  // instruction exists.
  Instruction* KillInst(Instruction* inst);

  // Deletes all the instruction in the range [`begin`; `end`[, for which the
  // unary predicate `condition` returned true.
  // Returns true if at least one instruction was removed, false otherwise.
  //
  // Pointer and iterator pointing to the deleted instructions become invalid.
  // However other pointers and iterators are still valid.
  bool KillInstructionIf(Module::inst_iterator begin, Module::inst_iterator end,
                         std::function<bool(Instruction*)> condition);

  // Collects the non-semantic instruction tree that uses |inst|'s result id
  // to be killed later.
  void CollectNonSemanticTree(Instruction* inst,
                              std::unordered_set<Instruction*>* to_kill);

  // Collect function reachable from |entryId|, returns |funcs|
  void CollectCallTreeFromRoots(unsigned entryId,
                                std::unordered_set<uint32_t>* funcs);

  // Returns true if all of the given analyses are valid.
  bool AreAnalysesValid(Analysis set) { return (set & valid_analyses_) == set; }

  // Replaces all uses of |before| id with |after| id. Returns true if any
  // replacement happens. This method does not kill the definition of the
  // |before| id. If |after| is the same as |before|, does nothing and returns
  // false.
  //
  // |before| and |after| must be registered definitions in the DefUseManager.
  bool ReplaceAllUsesWith(uint32_t before, uint32_t after);

  // Replace all uses of |before| id with |after| id if those uses
  // (instruction) return true for |predicate|. Returns true if
  // any replacement happens. This method does not kill the definition of the
  // |before| id. If |after| is the same as |before|, does nothing and return
  // false.
  bool ReplaceAllUsesWithPredicate(
      uint32_t before, uint32_t after,
      const std::function<bool(Instruction*)>& predicate);

  // Returns true if all of the analyses that are suppose to be valid are
  // actually valid.
  bool IsConsistent();

  // The IRContext will look at the def and uses of |inst| and update any valid
  // analyses will be updated accordingly.
  inline void AnalyzeDefUse(Instruction* inst);

  // Informs the IRContext that the uses of |inst| are going to change, and that
  // is should forget everything it know about the current uses.  Any valid
  // analyses will be updated accordingly.
  void ForgetUses(Instruction* inst);

  // The IRContext will look at the uses of |inst| and update any valid analyses
  // will be updated accordingly.
  void AnalyzeUses(Instruction* inst);

  // Kill all name and decorate ops targeting |id|.
  void KillNamesAndDecorates(uint32_t id);

  // Kill all name and decorate ops targeting the result id of |inst|.
  void KillNamesAndDecorates(Instruction* inst);

  // Change operands of debug instruction to DebugInfoNone.
  void KillOperandFromDebugInstructions(Instruction* inst);

  // Returns the next unique id for use by an instruction.
  inline uint32_t TakeNextUniqueId() {
    assert(unique_id_ != std::numeric_limits<uint32_t>::max());

    // Skip zero.
    return ++unique_id_;
  }

  // Returns true if |inst| is a combinator in the current context.
  // |combinator_ops_| is built if it has not been already.
  inline bool IsCombinatorInstruction(const Instruction* inst) {
    if (!AreAnalysesValid(kAnalysisCombinators)) {
      InitializeCombinators();
    }
    constexpr uint32_t kExtInstSetIdInIndx = 0;
    constexpr uint32_t kExtInstInstructionInIndx = 1;

    if (inst->opcode() != spv::Op::OpExtInst) {
      return combinator_ops_[0].count(uint32_t(inst->opcode())) != 0;
    } else {
      uint32_t set = inst->GetSingleWordInOperand(kExtInstSetIdInIndx);
      auto op = inst->GetSingleWordInOperand(kExtInstInstructionInIndx);
      return combinator_ops_[set].count(op) != 0;
    }
  }

  // Returns a pointer to the CFG for all the functions in |module_|.
  CFG* cfg() {
    if (!AreAnalysesValid(kAnalysisCFG)) {
      BuildCFG();
    }
    return cfg_.get();
  }

  // Gets the loop descriptor for function |f|.
  LoopDescriptor* GetLoopDescriptor(const Function* f);

  // Gets the dominator analysis for function |f|.
  DominatorAnalysis* GetDominatorAnalysis(const Function* f);

  // Gets the postdominator analysis for function |f|.
  PostDominatorAnalysis* GetPostDominatorAnalysis(const Function* f);

  // Remove the dominator tree of |f| from the cache.
  inline void RemoveDominatorAnalysis(const Function* f) {
    dominator_trees_.erase(f);
  }

  // Remove the postdominator tree of |f| from the cache.
  inline void RemovePostDominatorAnalysis(const Function* f) {
    post_dominator_trees_.erase(f);
  }

  // Return the next available SSA id and increment it.  Returns 0 if the
  // maximum SSA id has been reached.
  inline uint32_t TakeNextId() {
    uint32_t next_id = module()->TakeNextIdBound();
    if (next_id == 0) {
      if (consumer()) {
        std::string message = "ID overflow. Try running compact-ids.";
        consumer()(SPV_MSG_ERROR, "", {0, 0, 0}, message.c_str());
      }
#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
      // If TakeNextId returns 0, it is very likely that execution will
      // subsequently fail. Such failures are false alarms from a fuzzing point
      // of view: they are due to the fact that too many ids were used, rather
      // than being due to an actual bug. Thus, during a fuzzing build, it is
      // preferable to bail out when ID overflow occurs.
      //
      // A zero exit code is returned here because a non-zero code would cause
      // ClusterFuzz/OSS-Fuzz to regard the termination as a crash, and spurious
      // crash reports is what this guard aims to avoid.
      exit(0);
#endif
    }
    return next_id;
  }

  FeatureManager* get_feature_mgr() {
    if (!feature_mgr_.get()) {
      AnalyzeFeatures();
    }
    return feature_mgr_.get();
  }

  void ResetFeatureManager() { feature_mgr_.reset(nullptr); }

  // Returns the grammar for this context.
  const AssemblyGrammar& grammar() const { return grammar_; }

  // If |inst| has not yet been analysed by the def-use manager, then analyse
  // its definitions and uses.
  inline void UpdateDefUse(Instruction* inst);

  const InstructionFolder& get_instruction_folder() {
    if (!inst_folder_) {
      inst_folder_ = MakeUnique<InstructionFolder>(this);
    }
    return *inst_folder_;
  }

  uint32_t max_id_bound() const { return max_id_bound_; }
  void set_max_id_bound(uint32_t new_bound) { max_id_bound_ = new_bound; }

  bool preserve_bindings() const { return preserve_bindings_; }
  void set_preserve_bindings(bool should_preserve_bindings) {
    preserve_bindings_ = should_preserve_bindings;
  }

  bool preserve_spec_constants() const { return preserve_spec_constants_; }
  void set_preserve_spec_constants(bool should_preserve_spec_constants) {
    preserve_spec_constants_ = should_preserve_spec_constants;
  }

  // Return id of input variable only decorated with |builtin|, if in module.
  // Create variable and return its id otherwise. If builtin not currently
  // supported, return 0.
  uint32_t GetBuiltinInputVarId(uint32_t builtin);

  // Returns the function whose id is |id|, if one exists.  Returns |nullptr|
  // otherwise.
  Function* GetFunction(uint32_t id) {
    if (!AreAnalysesValid(kAnalysisIdToFuncMapping)) {
      BuildIdToFuncMapping();
    }
    auto entry = id_to_func_.find(id);
    return (entry != id_to_func_.end()) ? entry->second : nullptr;
  }

  Function* GetFunction(Instruction* inst) {
    if (inst->opcode() != spv::Op::OpFunction) {
      return nullptr;
    }
    return GetFunction(inst->result_id());
  }

  // Add to |todo| all ids of functions called directly from |func|.
  void AddCalls(const Function* func, std::queue<uint32_t>* todo);

  // Applies |pfn| to every function in the call trees that are rooted at the
  // entry points.  Returns true if any call |pfn| returns true.  By convention
  // |pfn| should return true if it modified the module.
  bool ProcessEntryPointCallTree(ProcessFunction& pfn);

  // Applies |pfn| to every function in the call trees rooted at the entry
  // points and exported functions.  Returns true if any call |pfn| returns
  // true.  By convention |pfn| should return true if it modified the module.
  bool ProcessReachableCallTree(ProcessFunction& pfn);

  // Applies |pfn| to every function in the call trees rooted at the elements of
  // |roots|.  Returns true if any call to |pfn| returns true.  By convention
  // |pfn| should return true if it modified the module.  After returning
  // |roots| will be empty.
  bool ProcessCallTreeFromRoots(ProcessFunction& pfn,
                                std::queue<uint32_t>* roots);

  // Emits a error message to the message consumer indicating the error
  // described by |message| occurred in |inst|.
  void EmitErrorMessage(std::string message, Instruction* inst);

  // Returns true if and only if there is a path to |bb| from the entry block of
  // the function that contains |bb|.
  bool IsReachable(const opt::BasicBlock& bb);

  // Return the stage of the module. Will generate error if entry points don't
  // all have the same stage.
  spv::ExecutionModel GetStage();

  // Returns true of the current target environment is at least that of the
  // given environment.
  bool IsTargetEnvAtLeast(spv_target_env env) {
    // A bit of a hack. We assume that the target environments are appended to
    // the enum, so that there is an appropriate order.
    return syntax_context_->target_env >= env;
  }

  // Return the target environment for the current context.
  spv_target_env GetTargetEnv() const { return syntax_context_->target_env; }

 private:
  // Builds the def-use manager from scratch, even if it was already valid.
  void BuildDefUseManager() {
    def_use_mgr_ = MakeUnique<analysis::DefUseManager>(module());
    valid_analyses_ = valid_analyses_ | kAnalysisDefUse;
  }

  // Builds the liveness manager from scratch, even if it was already valid.
  void BuildLivenessManager() {
    liveness_mgr_ = MakeUnique<analysis::LivenessManager>(this);
    valid_analyses_ = valid_analyses_ | kAnalysisLiveness;
  }

  // Builds the instruction-block map for the whole module.
  void BuildInstrToBlockMapping() {
    instr_to_block_.clear();
    for (auto& fn : *module_) {
      for (auto& block : fn) {
        block.ForEachInst([this, &block](Instruction* inst) {
          instr_to_block_[inst] = &block;
        });
      }
    }
    valid_analyses_ = valid_analyses_ | kAnalysisInstrToBlockMapping;
  }

  // Builds the instruction-function map for the whole module.
  void BuildIdToFuncMapping() {
    id_to_func_.clear();
    for (auto& fn : *module_) {
      id_to_func_[fn.result_id()] = &fn;
    }
    valid_analyses_ = valid_analyses_ | kAnalysisIdToFuncMapping;
  }

  void BuildDecorationManager() {
    decoration_mgr_ = MakeUnique<analysis::DecorationManager>(module());
    valid_analyses_ = valid_analyses_ | kAnalysisDecorations;
  }

  void BuildCFG() {
    cfg_ = MakeUnique<CFG>(module());
    valid_analyses_ = valid_analyses_ | kAnalysisCFG;
  }

  void BuildScalarEvolutionAnalysis() {
    scalar_evolution_analysis_ = MakeUnique<ScalarEvolutionAnalysis>(this);
    valid_analyses_ = valid_analyses_ | kAnalysisScalarEvolution;
  }

  // Builds the liveness analysis from scratch, even if it was already valid.
  void BuildRegPressureAnalysis() {
    reg_pressure_ = MakeUnique<LivenessAnalysis>(this);
    valid_analyses_ = valid_analyses_ | kAnalysisRegisterPressure;
  }

  // Builds the value number table analysis from scratch, even if it was already
  // valid.
  void BuildValueNumberTable() {
    vn_table_ = MakeUnique<ValueNumberTable>(this);
    valid_analyses_ = valid_analyses_ | kAnalysisValueNumberTable;
  }

  // Builds the structured CFG analysis from scratch, even if it was already
  // valid.
  void BuildStructuredCFGAnalysis() {
    struct_cfg_analysis_ = MakeUnique<StructuredCFGAnalysis>(this);
    valid_analyses_ = valid_analyses_ | kAnalysisStructuredCFG;
  }

  // Builds the constant manager from scratch, even if it was already
  // valid.
  void BuildConstantManager() {
    constant_mgr_ = MakeUnique<analysis::ConstantManager>(this);
    valid_analyses_ = valid_analyses_ | kAnalysisConstants;
  }

  // Builds the type manager from scratch, even if it was already
  // valid.
  void BuildTypeManager() {
    type_mgr_ = MakeUnique<analysis::TypeManager>(consumer(), this);
    valid_analyses_ = valid_analyses_ | kAnalysisTypes;
  }

  // Builds the debug information manager from scratch, even if it was
  // already valid.
  void BuildDebugInfoManager() {
    debug_info_mgr_ = MakeUnique<analysis::DebugInfoManager>(this);
    valid_analyses_ = valid_analyses_ | kAnalysisDebugInfo;
  }

  // Removes all computed dominator and post-dominator trees. This will force
  // the context to rebuild the trees on demand.
  void ResetDominatorAnalysis() {
    // Clear the cache.
    dominator_trees_.clear();
    post_dominator_trees_.clear();
    valid_analyses_ = valid_analyses_ | kAnalysisDominatorAnalysis;
  }

  // Removes all computed loop descriptors.
  void ResetLoopAnalysis() {
    // Clear the cache.
    loop_descriptors_.clear();
    valid_analyses_ = valid_analyses_ | kAnalysisLoopAnalysis;
  }

  // Removes all computed loop descriptors.
  void ResetBuiltinAnalysis() {
    // Clear the cache.
    builtin_var_id_map_.clear();
    valid_analyses_ = valid_analyses_ | kAnalysisBuiltinVarId;
  }

  // Analyzes the features in the owned module. Builds the manager if required.
  void AnalyzeFeatures() {
    feature_mgr_ =
        std::unique_ptr<FeatureManager>(new FeatureManager(grammar_));
    feature_mgr_->Analyze(module());
  }

  // Scans a module looking for it capabilities, and initializes combinator_ops_
  // accordingly.
  void InitializeCombinators();

  // Add the combinator opcode for the given capability to combinator_ops_.
  void AddCombinatorsForCapability(uint32_t capability);

  // Add the combinator opcode for the given extension to combinator_ops_.
  void AddCombinatorsForExtension(Instruction* extension);

  // Remove |inst| from |id_to_name_| if it is in map.
  void RemoveFromIdToName(const Instruction* inst);

  // Returns true if it is suppose to be valid but it is incorrect.  Returns
  // true if the cfg is invalidated.
  bool CheckCFG();

  // Return id of input variable only decorated with |builtin|, if in module.
  // Return 0 otherwise.
  uint32_t FindBuiltinInputVar(uint32_t builtin);

  // Add |var_id| to all entry points in module.
  void AddVarToEntryPoints(uint32_t var_id);

  // The SPIR-V syntax context containing grammar tables for opcodes and
  // operands.
  spv_context syntax_context_;

  // Auxiliary object for querying SPIR-V grammar facts.
  AssemblyGrammar grammar_;

  // An unique identifier for instructions in |module_|. Can be used to order
  // instructions in a container.
  //
  // This member is initialized to 0, but always issues this value plus one.
  // Therefore, 0 is not a valid unique id for an instruction.
  uint32_t unique_id_;

  // The module being processed within this IR context.
  std::unique_ptr<Module> module_;

  // A message consumer for diagnostics.
  MessageConsumer consumer_;

  // The def-use manager for |module_|.
  std::unique_ptr<analysis::DefUseManager> def_use_mgr_;

  // The instruction decoration manager for |module_|.
  std::unique_ptr<analysis::DecorationManager> decoration_mgr_;

  // The feature manager for |module_|.
  std::unique_ptr<FeatureManager> feature_mgr_;

  // A map from instructions to the basic block they belong to. This mapping is
  // built on-demand when get_instr_block() is called.
  //
  // NOTE: Do not traverse this map. Ever. Use the function and basic block
  // iterators to traverse instructions.
  std::unordered_map<Instruction*, BasicBlock*> instr_to_block_;

  // A map from ids to the function they define. This mapping is
  // built on-demand when GetFunction() is called.
  //
  // NOTE: Do not traverse this map. Ever. Use the function and basic block
  // iterators to traverse instructions.
  std::unordered_map<uint32_t, Function*> id_to_func_;

  // A bitset indicating which analyzes are currently valid.
  Analysis valid_analyses_;

  // Opcodes of shader capability core executable instructions
  // without side-effect.
  std::unordered_map<uint32_t, std::unordered_set<uint32_t>> combinator_ops_;

  // Opcodes of shader capability core executable instructions
  // without side-effect.
  std::unordered_map<uint32_t, uint32_t> builtin_var_id_map_;

  // The CFG for all the functions in |module_|.
  std::unique_ptr<CFG> cfg_;

  // Each function in the module will create its own dominator tree. We cache
  // the result so it doesn't need to be rebuilt each time.
  std::map<const Function*, DominatorAnalysis> dominator_trees_;
  std::map<const Function*, PostDominatorAnalysis> post_dominator_trees_;

  // Cache of loop descriptors for each function.
  std::unordered_map<const Function*, LoopDescriptor> loop_descriptors_;

  // Constant manager for |module_|.
  std::unique_ptr<analysis::ConstantManager> constant_mgr_;

  // Type manager for |module_|.
  std::unique_ptr<analysis::TypeManager> type_mgr_;

  // Debug information manager for |module_|.
  std::unique_ptr<analysis::DebugInfoManager> debug_info_mgr_;

  // A map from an id to its corresponding OpName and OpMemberName instructions.
  std::unique_ptr<std::multimap<uint32_t, Instruction*>> id_to_name_;

  // The cache scalar evolution analysis node.
  std::unique_ptr<ScalarEvolutionAnalysis> scalar_evolution_analysis_;

  // The liveness analysis |module_|.
  std::unique_ptr<LivenessAnalysis> reg_pressure_;

  std::unique_ptr<ValueNumberTable> vn_table_;

  std::unique_ptr<InstructionFolder> inst_folder_;

  std::unique_ptr<StructuredCFGAnalysis> struct_cfg_analysis_;

  // The liveness manager for |module_|.
  std::unique_ptr<analysis::LivenessManager> liveness_mgr_;

  // The maximum legal value for the id bound.
  uint32_t max_id_bound_;

  // Whether all bindings within |module_| should be preserved.
  bool preserve_bindings_;

  // Whether all specialization constants within |module_|
  // should be preserved.
  bool preserve_spec_constants_;
};

inline IRContext::Analysis operator|(IRContext::Analysis lhs,
                                     IRContext::Analysis rhs) {
  return static_cast<IRContext::Analysis>(static_cast<int>(lhs) |
                                          static_cast<int>(rhs));
}

inline IRContext::Analysis& operator|=(IRContext::Analysis& lhs,
                                       IRContext::Analysis rhs) {
  lhs = lhs | rhs;
  return lhs;
}

inline IRContext::Analysis operator<<(IRContext::Analysis a, int shift) {
  return static_cast<IRContext::Analysis>(static_cast<int>(a) << shift);
}

inline IRContext::Analysis& operator<<=(IRContext::Analysis& a, int shift) {
  a = static_cast<IRContext::Analysis>(static_cast<int>(a) << shift);
  return a;
}

std::vector<Instruction*> IRContext::GetConstants() {
  return module()->GetConstants();
}

std::vector<const Instruction*> IRContext::GetConstants() const {
  return ((const Module*)module())->GetConstants();
}

Module::inst_iterator IRContext::annotation_begin() {
  return module()->annotation_begin();
}

Module::inst_iterator IRContext::annotation_end() {
  return module()->annotation_end();
}

IteratorRange<Module::inst_iterator> IRContext::annotations() {
  return module_->annotations();
}

IteratorRange<Module::const_inst_iterator> IRContext::annotations() const {
  return ((const Module*)module_.get())->annotations();
}

Module::inst_iterator IRContext::capability_begin() {
  return module()->capability_begin();
}

Module::inst_iterator IRContext::capability_end() {
  return module()->capability_end();
}

IteratorRange<Module::inst_iterator> IRContext::capabilities() {
  return module()->capabilities();
}

IteratorRange<Module::const_inst_iterator> IRContext::capabilities() const {
  return ((const Module*)module())->capabilities();
}

Module::inst_iterator IRContext::extension_begin() {
  return module()->extension_begin();
}

Module::inst_iterator IRContext::extension_end() {
  return module()->extension_end();
}

IteratorRange<Module::inst_iterator> IRContext::extensions() {
  return module()->extensions();
}

IteratorRange<Module::const_inst_iterator> IRContext::extensions() const {
  return ((const Module*)module())->extensions();
}

Module::inst_iterator IRContext::types_values_begin() {
  return module()->types_values_begin();
}

Module::inst_iterator IRContext::types_values_end() {
  return module()->types_values_end();
}

IteratorRange<Module::inst_iterator> IRContext::types_values() {
  return module()->types_values();
}

IteratorRange<Module::const_inst_iterator> IRContext::types_values() const {
  return ((const Module*)module_.get())->types_values();
}

Module::inst_iterator IRContext::ext_inst_import_begin() {
  return module()->ext_inst_import_begin();
}

Module::inst_iterator IRContext::ext_inst_import_end() {
  return module()->ext_inst_import_end();
}

IteratorRange<Module::inst_iterator> IRContext::ext_inst_imports() {
  return module()->ext_inst_imports();
}

IteratorRange<Module::const_inst_iterator> IRContext::ext_inst_imports() const {
  return ((const Module*)module_.get())->ext_inst_imports();
}

Module::inst_iterator IRContext::debug1_begin() {
  return module()->debug1_begin();
}

Module::inst_iterator IRContext::debug1_end() { return module()->debug1_end(); }

IteratorRange<Module::inst_iterator> IRContext::debugs1() {
  return module()->debugs1();
}

IteratorRange<Module::const_inst_iterator> IRContext::debugs1() const {
  return ((const Module*)module_.get())->debugs1();
}

Module::inst_iterator IRContext::debug2_begin() {
  return module()->debug2_begin();
}
Module::inst_iterator IRContext::debug2_end() { return module()->debug2_end(); }

IteratorRange<Module::inst_iterator> IRContext::debugs2() {
  return module()->debugs2();
}

IteratorRange<Module::const_inst_iterator> IRContext::debugs2() const {
  return ((const Module*)module_.get())->debugs2();
}

Module::inst_iterator IRContext::debug3_begin() {
  return module()->debug3_begin();
}

Module::inst_iterator IRContext::debug3_end() { return module()->debug3_end(); }

IteratorRange<Module::inst_iterator> IRContext::debugs3() {
  return module()->debugs3();
}

IteratorRange<Module::const_inst_iterator> IRContext::debugs3() const {
  return ((const Module*)module_.get())->debugs3();
}

Module::inst_iterator IRContext::ext_inst_debuginfo_begin() {
  return module()->ext_inst_debuginfo_begin();
}

Module::inst_iterator IRContext::ext_inst_debuginfo_end() {
  return module()->ext_inst_debuginfo_end();
}

IteratorRange<Module::inst_iterator> IRContext::ext_inst_debuginfo() {
  return module()->ext_inst_debuginfo();
}

IteratorRange<Module::const_inst_iterator> IRContext::ext_inst_debuginfo()
    const {
  return ((const Module*)module_.get())->ext_inst_debuginfo();
}

void IRContext::AddCapability(spv::Capability capability) {
  if (!get_feature_mgr()->HasCapability(capability)) {
    std::unique_ptr<Instruction> capability_inst(new Instruction(
        this, spv::Op::OpCapability, 0, 0,
        {{SPV_OPERAND_TYPE_CAPABILITY, {static_cast<uint32_t>(capability)}}}));
    AddCapability(std::move(capability_inst));
  }
}

void IRContext::AddCapability(std::unique_ptr<Instruction>&& c) {
  AddCombinatorsForCapability(c->GetSingleWordInOperand(0));
  if (feature_mgr_ != nullptr) {
    feature_mgr_->AddCapability(
        static_cast<spv::Capability>(c->GetSingleWordInOperand(0)));
  }
  if (AreAnalysesValid(kAnalysisDefUse)) {
    get_def_use_mgr()->AnalyzeInstDefUse(c.get());
  }
  module()->AddCapability(std::move(c));
}

void IRContext::AddExtension(const std::string& ext_name) {
  std::vector<uint32_t> ext_words = spvtools::utils::MakeVector(ext_name);
  AddExtension(std::unique_ptr<Instruction>(
      new Instruction(this, spv::Op::OpExtension, 0u, 0u,
                      {{SPV_OPERAND_TYPE_LITERAL_STRING, ext_words}})));
}

void IRContext::AddExtension(std::unique_ptr<Instruction>&& e) {
  if (AreAnalysesValid(kAnalysisDefUse)) {
    get_def_use_mgr()->AnalyzeInstDefUse(e.get());
  }
  if (feature_mgr_ != nullptr) {
    feature_mgr_->AddExtension(&*e);
  }
  module()->AddExtension(std::move(e));
}

void IRContext::AddExtInstImport(const std::string& name) {
  std::vector<uint32_t> ext_words = spvtools::utils::MakeVector(name);
  AddExtInstImport(std::unique_ptr<Instruction>(
      new Instruction(this, spv::Op::OpExtInstImport, 0u, TakeNextId(),
                      {{SPV_OPERAND_TYPE_LITERAL_STRING, ext_words}})));
}

void IRContext::AddExtInstImport(std::unique_ptr<Instruction>&& e) {
  AddCombinatorsForExtension(e.get());
  if (AreAnalysesValid(kAnalysisDefUse)) {
    get_def_use_mgr()->AnalyzeInstDefUse(e.get());
  }
  module()->AddExtInstImport(std::move(e));
  if (feature_mgr_ != nullptr) {
    feature_mgr_->AddExtInstImportIds(module());
  }
}

void IRContext::SetMemoryModel(std::unique_ptr<Instruction>&& m) {
  module()->SetMemoryModel(std::move(m));
}

const Instruction* IRContext::GetMemoryModel() const {
  return module()->GetMemoryModel();
}

void IRContext::AddEntryPoint(std::unique_ptr<Instruction>&& e) {
  module()->AddEntryPoint(std::move(e));
}

void IRContext::AddExecutionMode(std::unique_ptr<Instruction>&& e) {
  module()->AddExecutionMode(std::move(e));
}

void IRContext::AddDebug1Inst(std::unique_ptr<Instruction>&& d) {
  module()->AddDebug1Inst(std::move(d));
}

void IRContext::AddDebug2Inst(std::unique_ptr<Instruction>&& d) {
  if (AreAnalysesValid(kAnalysisNameMap)) {
    if (d->opcode() == spv::Op::OpName ||
        d->opcode() == spv::Op::OpMemberName) {
      // OpName and OpMemberName do not have result-ids. The target of the
      // instruction is at InOperand index 0.
      id_to_name_->insert({d->GetSingleWordInOperand(0), d.get()});
    }
  }
  if (AreAnalysesValid(kAnalysisDefUse)) {
    get_def_use_mgr()->AnalyzeInstDefUse(d.get());
  }
  module()->AddDebug2Inst(std::move(d));
}

void IRContext::AddDebug3Inst(std::unique_ptr<Instruction>&& d) {
  module()->AddDebug3Inst(std::move(d));
}

void IRContext::AddExtInstDebugInfo(std::unique_ptr<Instruction>&& d) {
  module()->AddExtInstDebugInfo(std::move(d));
}

void IRContext::AddAnnotationInst(std::unique_ptr<Instruction>&& a) {
  if (AreAnalysesValid(kAnalysisDecorations)) {
    get_decoration_mgr()->AddDecoration(a.get());
  }
  if (AreAnalysesValid(kAnalysisDefUse)) {
    get_def_use_mgr()->AnalyzeInstDefUse(a.get());
  }
  module()->AddAnnotationInst(std::move(a));
}

void IRContext::AddType(std::unique_ptr<Instruction>&& t) {
  module()->AddType(std::move(t));
  if (AreAnalysesValid(kAnalysisDefUse)) {
    get_def_use_mgr()->AnalyzeInstDefUse(&*(--types_values_end()));
  }
}

void IRContext::AddGlobalValue(std::unique_ptr<Instruction>&& v) {
  if (AreAnalysesValid(kAnalysisDefUse)) {
    get_def_use_mgr()->AnalyzeInstDefUse(&*v);
  }
  module()->AddGlobalValue(std::move(v));
}

void IRContext::AddFunctionDeclaration(std::unique_ptr<Function>&& f) {
  module()->AddFunctionDeclaration(std::move(f));
}

void IRContext::AddFunction(std::unique_ptr<Function>&& f) {
  module()->AddFunction(std::move(f));
}

void IRContext::AnalyzeDefUse(Instruction* inst) {
  if (AreAnalysesValid(kAnalysisDefUse)) {
    get_def_use_mgr()->AnalyzeInstDefUse(inst);
  }
}

void IRContext::UpdateDefUse(Instruction* inst) {
  if (AreAnalysesValid(kAnalysisDefUse)) {
    get_def_use_mgr()->UpdateDefUse(inst);
  }
}

void IRContext::BuildIdToNameMap() {
  id_to_name_ = MakeUnique<std::multimap<uint32_t, Instruction*>>();
  for (Instruction& debug_inst : debugs2()) {
    if (debug_inst.opcode() == spv::Op::OpMemberName ||
        debug_inst.opcode() == spv::Op::OpName) {
      id_to_name_->insert({debug_inst.GetSingleWordInOperand(0), &debug_inst});
    }
  }
  valid_analyses_ = valid_analyses_ | kAnalysisNameMap;
}

IteratorRange<std::multimap<uint32_t, Instruction*>::iterator>
IRContext::GetNames(uint32_t id) {
  if (!AreAnalysesValid(kAnalysisNameMap)) {
    BuildIdToNameMap();
  }
  auto result = id_to_name_->equal_range(id);
  return make_range(std::move(result.first), std::move(result.second));
}

Instruction* IRContext::GetMemberName(uint32_t struct_type_id, uint32_t index) {
  if (!AreAnalysesValid(kAnalysisNameMap)) {
    BuildIdToNameMap();
  }
  auto result = id_to_name_->equal_range(struct_type_id);
  for (auto i = result.first; i != result.second; ++i) {
    auto* name_instr = i->second;
    if (name_instr->opcode() == spv::Op::OpMemberName &&
        name_instr->GetSingleWordInOperand(1) == index) {
      return name_instr;
    }
  }
  return nullptr;
}

void IRContext::CloneNames(const uint32_t old_id, const uint32_t new_id,
                           const uint32_t max_member_index) {
  std::vector<std::unique_ptr<Instruction>> names_to_add;
  auto names = GetNames(old_id);
  for (auto n : names) {
    Instruction* old_name_inst = n.second;
    if (old_name_inst->opcode() == spv::Op::OpMemberName) {
      auto midx = old_name_inst->GetSingleWordInOperand(1);
      if (midx >= max_member_index) continue;
    }
    std::unique_ptr<Instruction> new_name_inst(old_name_inst->Clone(this));
    new_name_inst->SetInOperand(0, {new_id});
    names_to_add.push_back(std::move(new_name_inst));
  }
  // We can't add the new names when we are iterating over name range above.
  // We can add all the new names now.
  for (auto& new_name : names_to_add) AddDebug2Inst(std::move(new_name));
}

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_IR_CONTEXT_H_
