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

#ifndef INCLUDE_SPIRV_TOOLS_OPTIMIZER_HPP_
#define INCLUDE_SPIRV_TOOLS_OPTIMIZER_HPP_

#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "libspirv.hpp"

namespace spvtools {

namespace opt {
class Pass;
struct DescriptorSetAndBinding;
}  // namespace opt

// C++ interface for SPIR-V optimization functionalities. It wraps the context
// (including target environment and the corresponding SPIR-V grammar) and
// provides methods for registering optimization passes and optimizing.
//
// Instances of this class provides basic thread-safety guarantee.
class Optimizer {
 public:
  // The token for an optimization pass. It is returned via one of the
  // Create*Pass() standalone functions at the end of this header file and
  // consumed by the RegisterPass() method. Tokens are one-time objects that
  // only support move; copying is not allowed.
  struct PassToken {
    struct Impl;  // Opaque struct for holding internal data.

    PassToken(std::unique_ptr<Impl>);

    // Tokens for built-in passes should be created using Create*Pass functions
    // below; for out-of-tree passes, use this constructor instead.
    // Note that this API isn't guaranteed to be stable and may change without
    // preserving source or binary compatibility in the future.
    PassToken(std::unique_ptr<opt::Pass>&& pass);

    // Tokens can only be moved. Copying is disabled.
    PassToken(const PassToken&) = delete;
    PassToken(PassToken&&);
    PassToken& operator=(const PassToken&) = delete;
    PassToken& operator=(PassToken&&);

    ~PassToken();

    std::unique_ptr<Impl> impl_;  // Unique pointer to internal data.
  };

  // Constructs an instance with the given target |env|, which is used to decode
  // the binaries to be optimized later.
  //
  // The instance will have an empty message consumer, which ignores all
  // messages from the library. Use SetMessageConsumer() to supply a consumer
  // if messages are of concern.
  explicit Optimizer(spv_target_env env);

  // Disables copy/move constructor/assignment operations.
  Optimizer(const Optimizer&) = delete;
  Optimizer(Optimizer&&) = delete;
  Optimizer& operator=(const Optimizer&) = delete;
  Optimizer& operator=(Optimizer&&) = delete;

  // Destructs this instance.
  ~Optimizer();

  // Sets the message consumer to the given |consumer|. The |consumer| will be
  // invoked once for each message communicated from the library.
  void SetMessageConsumer(MessageConsumer consumer);

  // Returns a reference to the registered message consumer.
  const MessageConsumer& consumer() const;

  // Registers the given |pass| to this optimizer. Passes will be run in the
  // exact order of registration. The token passed in will be consumed by this
  // method.
  Optimizer& RegisterPass(PassToken&& pass);

  // Registers passes that attempt to improve performance of generated code.
  // This sequence of passes is subject to constant review and will change
  // from time to time.
  Optimizer& RegisterPerformancePasses();

  // Registers passes that attempt to improve the size of generated code.
  // This sequence of passes is subject to constant review and will change
  // from time to time.
  Optimizer& RegisterSizePasses();

  // Registers passes that attempt to legalize the generated code.
  //
  // Note: this recipe is specially designed for legalizing SPIR-V. It should be
  // used by compilers after translating HLSL source code literally. It should
  // *not* be used by general workloads for performance or size improvement.
  //
  // This sequence of passes is subject to constant review and will change
  // from time to time.
  Optimizer& RegisterLegalizationPasses();

  // Register passes specified in the list of |flags|.  Each flag must be a
  // string of a form accepted by Optimizer::FlagHasValidForm().
  //
  // If the list of flags contains an invalid entry, it returns false and an
  // error message is emitted to the MessageConsumer object (use
  // Optimizer::SetMessageConsumer to define a message consumer, if needed).
  //
  // If all the passes are registered successfully, it returns true.
  bool RegisterPassesFromFlags(const std::vector<std::string>& flags);

  // Registers the optimization pass associated with |flag|.  This only accepts
  // |flag| values of the form "--pass_name[=pass_args]".  If no such pass
  // exists, it returns false.  Otherwise, the pass is registered and it returns
  // true.
  //
  // The following flags have special meaning:
  //
  // -O: Registers all performance optimization passes
  //     (Optimizer::RegisterPerformancePasses)
  //
  // -Os: Registers all size optimization passes
  //      (Optimizer::RegisterSizePasses).
  //
  // --legalize-hlsl: Registers all passes that legalize SPIR-V generated by an
  //                  HLSL front-end.
  bool RegisterPassFromFlag(const std::string& flag);

  // Validates that |flag| has a valid format.  Strings accepted:
  //
  // --pass_name[=pass_args]
  // -O
  // -Os
  //
  // If |flag| takes one of the forms above, it returns true.  Otherwise, it
  // returns false.
  bool FlagHasValidForm(const std::string& flag) const;

  // Allows changing, after creation time, the target environment to be
  // optimized for and validated.  Should be called before calling Run().
  void SetTargetEnv(const spv_target_env env);

  // Optimizes the given SPIR-V module |original_binary| and writes the
  // optimized binary into |optimized_binary|. The optimized binary uses
  // the same SPIR-V version as the original binary.
  //
  // Returns true on successful optimization, whether or not the module is
  // modified. Returns false if |original_binary| fails to validate or if errors
  // occur when processing |original_binary| using any of the registered passes.
  // In that case, no further passes are executed and the contents in
  // |optimized_binary| may be invalid.
  //
  // By default, the binary is validated before any transforms are performed,
  // and optionally after each transform.  Validation uses SPIR-V spec rules
  // for the SPIR-V version named in the binary's header (at word offset 1).
  // Additionally, if the target environment is a client API (such as
  // Vulkan 1.1), then validate for that client API version, to the extent
  // that it is verifiable from data in the binary itself.
  //
  // It's allowed to alias |original_binary| to the start of |optimized_binary|.
  bool Run(const uint32_t* original_binary, size_t original_binary_size,
           std::vector<uint32_t>* optimized_binary) const;

  // DEPRECATED: Same as above, except passes |options| to the validator when
  // trying to validate the binary.  If |skip_validation| is true, then the
  // caller is guaranteeing that |original_binary| is valid, and the validator
  // will not be run.  The |max_id_bound| is the limit on the max id in the
  // module.
  bool Run(const uint32_t* original_binary, const size_t original_binary_size,
           std::vector<uint32_t>* optimized_binary,
           const ValidatorOptions& options, bool skip_validation) const;

  // Same as above, except it takes an options object.  See the documentation
  // for |OptimizerOptions| to see which options can be set.
  //
  // By default, the binary is validated before any transforms are performed,
  // and optionally after each transform.  Validation uses SPIR-V spec rules
  // for the SPIR-V version named in the binary's header (at word offset 1).
  // Additionally, if the target environment is a client API (such as
  // Vulkan 1.1), then validate for that client API version, to the extent
  // that it is verifiable from data in the binary itself, or from the
  // validator options set on the optimizer options.
  bool Run(const uint32_t* original_binary, const size_t original_binary_size,
           std::vector<uint32_t>* optimized_binary,
           const spv_optimizer_options opt_options) const;

  // Returns a vector of strings with all the pass names added to this
  // optimizer's pass manager. These strings are valid until the associated
  // pass manager is destroyed.
  std::vector<const char*> GetPassNames() const;

  // Sets the option to print the disassembly before each pass and after the
  // last pass.  If |out| is null, then no output is generated.  Otherwise,
  // output is sent to the |out| output stream.
  Optimizer& SetPrintAll(std::ostream* out);

  // Sets the option to print the resource utilization of each pass. If |out|
  // is null, then no output is generated. Otherwise, output is sent to the
  // |out| output stream.
  Optimizer& SetTimeReport(std::ostream* out);

  // Sets the option to validate the module after each pass.
  Optimizer& SetValidateAfterAll(bool validate);

 private:
  struct Impl;                  // Opaque struct for holding internal data.
  std::unique_ptr<Impl> impl_;  // Unique pointer to internal data.
};

// Creates a null pass.
// A null pass does nothing to the SPIR-V module to be optimized.
Optimizer::PassToken CreateNullPass();

// Creates a strip-debug-info pass.
// A strip-debug-info pass removes all debug instructions (as documented in
// Section 3.42.2 of the SPIR-V spec) of the SPIR-V module to be optimized.
Optimizer::PassToken CreateStripDebugInfoPass();

// [Deprecated] This will create a strip-nonsemantic-info pass.  See below.
Optimizer::PassToken CreateStripReflectInfoPass();

// Creates a strip-nonsemantic-info pass.
// A strip-nonsemantic-info pass removes all reflections and explicitly
// non-semantic instructions.
Optimizer::PassToken CreateStripNonSemanticInfoPass();

// Creates an eliminate-dead-functions pass.
// An eliminate-dead-functions pass will remove all functions that are not in
// the call trees rooted at entry points and exported functions.  These
// functions are not needed because they will never be called.
Optimizer::PassToken CreateEliminateDeadFunctionsPass();

// Creates an eliminate-dead-members pass.
// An eliminate-dead-members pass will remove all unused members of structures.
// This will not affect the data layout of the remaining members.
Optimizer::PassToken CreateEliminateDeadMembersPass();

// Creates a set-spec-constant-default-value pass from a mapping from spec-ids
// to the default values in the form of string.
// A set-spec-constant-default-value pass sets the default values for the
// spec constants that have SpecId decorations (i.e., those defined by
// OpSpecConstant{|True|False} instructions).
Optimizer::PassToken CreateSetSpecConstantDefaultValuePass(
    const std::unordered_map<uint32_t, std::string>& id_value_map);

// Creates a set-spec-constant-default-value pass from a mapping from spec-ids
// to the default values in the form of bit pattern.
// A set-spec-constant-default-value pass sets the default values for the
// spec constants that have SpecId decorations (i.e., those defined by
// OpSpecConstant{|True|False} instructions).
Optimizer::PassToken CreateSetSpecConstantDefaultValuePass(
    const std::unordered_map<uint32_t, std::vector<uint32_t>>& id_value_map);

// Creates a flatten-decoration pass.
// A flatten-decoration pass replaces grouped decorations with equivalent
// ungrouped decorations.  That is, it replaces each OpDecorationGroup
// instruction and associated OpGroupDecorate and OpGroupMemberDecorate
// instructions with equivalent OpDecorate and OpMemberDecorate instructions.
// The pass does not attempt to preserve debug information for instructions
// it removes.
Optimizer::PassToken CreateFlattenDecorationPass();

// Creates a freeze-spec-constant-value pass.
// A freeze-spec-constant pass specializes the value of spec constants to
// their default values. This pass only processes the spec constants that have
// SpecId decorations (defined by OpSpecConstant, OpSpecConstantTrue, or
// OpSpecConstantFalse instructions) and replaces them with their normal
// counterparts (OpConstant, OpConstantTrue, or OpConstantFalse). The
// corresponding SpecId annotation instructions will also be removed. This
// pass does not fold the newly added normal constants and does not process
// other spec constants defined by OpSpecConstantComposite or
// OpSpecConstantOp.
Optimizer::PassToken CreateFreezeSpecConstantValuePass();

// Creates a fold-spec-constant-op-and-composite pass.
// A fold-spec-constant-op-and-composite pass folds spec constants defined by
// OpSpecConstantOp or OpSpecConstantComposite instruction, to normal Constants
// defined by OpConstantTrue, OpConstantFalse, OpConstant, OpConstantNull, or
// OpConstantComposite instructions. Note that spec constants defined with
// OpSpecConstant, OpSpecConstantTrue, or OpSpecConstantFalse instructions are
// not handled, as these instructions indicate their value are not determined
// and can be changed in future. A spec constant is foldable if all of its
// value(s) can be determined from the module. E.g., an integer spec constant
// defined with OpSpecConstantOp instruction can be folded if its value won't
// change later. This pass will replace the original OpSpecConstantOp
// instruction with an OpConstant instruction. When folding composite spec
// constants, new instructions may be inserted to define the components of the
// composite constant first, then the original spec constants will be replaced
// by OpConstantComposite instructions.
//
// There are some operations not supported yet:
//   OpSConvert, OpFConvert, OpQuantizeToF16 and
//   all the operations under Kernel capability.
// TODO(qining): Add support for the operations listed above.
Optimizer::PassToken CreateFoldSpecConstantOpAndCompositePass();

// Creates a unify-constant pass.
// A unify-constant pass de-duplicates the constants. Constants with the exact
// same value and identical form will be unified and only one constant will
// be kept for each unique pair of type and value.
// There are several cases not handled by this pass:
//  1) Constants defined by OpConstantNull instructions (null constants) and
//  constants defined by OpConstantFalse, OpConstant or OpConstantComposite
//  with value 0 (zero-valued normal constants) are not considered equivalent.
//  So null constants won't be used to replace zero-valued normal constants,
//  vice versa.
//  2) Whenever there are decorations to the constant's result id id, the
//  constant won't be handled, which means, it won't be used to replace any
//  other constants, neither can other constants replace it.
//  3) NaN in float point format with different bit patterns are not unified.
Optimizer::PassToken CreateUnifyConstantPass();

// Creates a eliminate-dead-constant pass.
// A eliminate-dead-constant pass removes dead constants, including normal
// constants defined by OpConstant, OpConstantComposite, OpConstantTrue, or
// OpConstantFalse and spec constants defined by OpSpecConstant,
// OpSpecConstantComposite, OpSpecConstantTrue, OpSpecConstantFalse or
// OpSpecConstantOp.
Optimizer::PassToken CreateEliminateDeadConstantPass();

// Creates a strength-reduction pass.
// A strength-reduction pass will look for opportunities to replace an
// instruction with an equivalent and less expensive one.  For example,
// multiplying by a power of 2 can be replaced by a bit shift.
Optimizer::PassToken CreateStrengthReductionPass();

// Creates a block merge pass.
// This pass searches for blocks with a single Branch to a block with no
// other predecessors and merges the blocks into a single block. Continue
// blocks and Merge blocks are not candidates for the second block.
//
// The pass is most useful after Dead Branch Elimination, which can leave
// such sequences of blocks. Merging them makes subsequent passes more
// effective, such as single block local store-load elimination.
//
// While this pass reduces the number of occurrences of this sequence, at
// this time it does not guarantee all such sequences are eliminated.
//
// Presence of phi instructions can inhibit this optimization. Handling
// these is left for future improvements.
Optimizer::PassToken CreateBlockMergePass();

// Creates an exhaustive inline pass.
// An exhaustive inline pass attempts to exhaustively inline all function
// calls in all functions in an entry point call tree. The intent is to enable,
// albeit through brute force, analysis and optimization across function
// calls by subsequent optimization passes. As the inlining is exhaustive,
// there is no attempt to optimize for size or runtime performance. Functions
// that are not in the call tree of an entry point are not changed.
Optimizer::PassToken CreateInlineExhaustivePass();

// Creates an opaque inline pass.
// An opaque inline pass inlines all function calls in all functions in all
// entry point call trees where the called function contains an opaque type
// in either its parameter types or return type. An opaque type is currently
// defined as Image, Sampler or SampledImage. The intent is to enable, albeit
// through brute force, analysis and optimization across these function calls
// by subsequent passes in order to remove the storing of opaque types which is
// not legal in Vulkan. Functions that are not in the call tree of an entry
// point are not changed.
Optimizer::PassToken CreateInlineOpaquePass();

// Creates a single-block local variable load/store elimination pass.
// For every entry point function, do single block memory optimization of
// function variables referenced only with non-access-chain loads and stores.
// For each targeted variable load, if previous store to that variable in the
// block, replace the load's result id with the value id of the store.
// If previous load within the block, replace the current load's result id
// with the previous load's result id. In either case, delete the current
// load. Finally, check if any remaining stores are useless, and delete store
// and variable if possible.
//
// The presence of access chain references and function calls can inhibit
// the above optimization.
//
// Only modules with relaxed logical addressing (see opt/instruction.h) are
// currently processed.
//
// This pass is most effective if preceded by Inlining and
// LocalAccessChainConvert. This pass will reduce the work needed to be done
// by LocalSingleStoreElim and LocalMultiStoreElim.
//
// Only functions in the call tree of an entry point are processed.
Optimizer::PassToken CreateLocalSingleBlockLoadStoreElimPass();

// Create dead branch elimination pass.
// For each entry point function, this pass will look for SelectionMerge
// BranchConditionals with constant condition and convert to a Branch to
// the indicated label. It will delete resulting dead blocks.
//
// For all phi functions in merge block, replace all uses with the id
// corresponding to the living predecessor.
//
// Note that some branches and blocks may be left to avoid creating invalid
// control flow. Improving this is left to future work.
//
// This pass is most effective when preceded by passes which eliminate
// local loads and stores, effectively propagating constant values where
// possible.
Optimizer::PassToken CreateDeadBranchElimPass();

// Creates an SSA local variable load/store elimination pass.
// For every entry point function, eliminate all loads and stores of function
// scope variables only referenced with non-access-chain loads and stores.
// Eliminate the variables as well.
//
// The presence of access chain references and function calls can inhibit
// the above optimization.
//
// Only shader modules with relaxed logical addressing (see opt/instruction.h)
// are currently processed. Currently modules with any extensions enabled are
// not processed. This is left for future work.
//
// This pass is most effective if preceded by Inlining and
// LocalAccessChainConvert. LocalSingleStoreElim and LocalSingleBlockElim
// will reduce the work that this pass has to do.
Optimizer::PassToken CreateLocalMultiStoreElimPass();

// Creates a local access chain conversion pass.
// A local access chain conversion pass identifies all function scope
// variables which are accessed only with loads, stores and access chains
// with constant indices. It then converts all loads and stores of such
// variables into equivalent sequences of loads, stores, extracts and inserts.
//
// This pass only processes entry point functions. It currently only converts
// non-nested, non-ptr access chains. It does not process modules with
// non-32-bit integer types present. Optional memory access options on loads
// and stores are ignored as we are only processing function scope variables.
//
// This pass unifies access to these variables to a single mode and simplifies
// subsequent analysis and elimination of these variables along with their
// loads and stores allowing values to propagate to their points of use where
// possible.
Optimizer::PassToken CreateLocalAccessChainConvertPass();

// Creates a local single store elimination pass.
// For each entry point function, this pass eliminates loads and stores for
// function scope variable that are stored to only once, where possible. Only
// whole variable loads and stores are eliminated; access-chain references are
// not optimized. Replace all loads of such variables with the value that is
// stored and eliminate any resulting dead code.
//
// Currently, the presence of access chains and function calls can inhibit this
// pass, however the Inlining and LocalAccessChainConvert passes can make it
// more effective. In additional, many non-load/store memory operations are
// not supported and will prohibit optimization of a function. Support of
// these operations are future work.
//
// Only shader modules with relaxed logical addressing (see opt/instruction.h)
// are currently processed.
//
// This pass will reduce the work needed to be done by LocalSingleBlockElim
// and LocalMultiStoreElim and can improve the effectiveness of other passes
// such as DeadBranchElimination which depend on values for their analysis.
Optimizer::PassToken CreateLocalSingleStoreElimPass();

// Creates an insert/extract elimination pass.
// This pass processes each entry point function in the module, searching for
// extracts on a sequence of inserts. It further searches the sequence for an
// insert with indices identical to the extract. If such an insert can be
// found before hitting a conflicting insert, the extract's result id is
// replaced with the id of the values from the insert.
//
// Besides removing extracts this pass enables subsequent dead code elimination
// passes to delete the inserts. This pass performs best after access chains are
// converted to inserts and extracts and local loads and stores are eliminated.
Optimizer::PassToken CreateInsertExtractElimPass();

// Creates a dead insert elimination pass.
// This pass processes each entry point function in the module, searching for
// unreferenced inserts into composite types. These are most often unused
// stores to vector components. They are unused because they are never
// referenced, or because there is another insert to the same component between
// the insert and the reference. After removing the inserts, dead code
// elimination is attempted on the inserted values.
//
// This pass performs best after access chains are converted to inserts and
// extracts and local loads and stores are eliminated. While executing this
// pass can be advantageous on its own, it is also advantageous to execute
// this pass after CreateInsertExtractPass() as it will remove any unused
// inserts created by that pass.
Optimizer::PassToken CreateDeadInsertElimPass();

// Create aggressive dead code elimination pass
// This pass eliminates unused code from the module. In addition,
// it detects and eliminates code which may have spurious uses but which do
// not contribute to the output of the function. The most common cause of
// such code sequences is summations in loops whose result is no longer used
// due to dead code elimination. This optimization has additional compile
// time cost over standard dead code elimination.
//
// This pass only processes entry point functions. It also only processes
// shaders with relaxed logical addressing (see opt/instruction.h). It
// currently will not process functions with function calls. Unreachable
// functions are deleted.
//
// This pass will be made more effective by first running passes that remove
// dead control flow and inlines function calls.
//
// This pass can be especially useful after running Local Access Chain
// Conversion, which tends to cause cycles of dead code to be left after
// Store/Load elimination passes are completed. These cycles cannot be
// eliminated with standard dead code elimination.
//
// If |preserve_interface| is true, all non-io variables in the entry point
// interface are considered live and are not eliminated. This mode is needed
// by GPU-Assisted validation instrumentation, where a change in the interface
// is not allowed.
//
// If |remove_outputs| is true, allow outputs to be removed from the interface.
// This is only safe if the caller knows that there is no corresponding input
// variable in the following shader. It is false by default.
Optimizer::PassToken CreateAggressiveDCEPass();
Optimizer::PassToken CreateAggressiveDCEPass(bool preserve_interface);
Optimizer::PassToken CreateAggressiveDCEPass(bool preserve_interface,
                                             bool remove_outputs);

// Creates a remove-unused-interface-variables pass.
// Removes variables referenced on the |OpEntryPoint| instruction that are not
// referenced in the entry point function or any function in its call tree. Note
// that this could cause the shader interface to no longer match other shader
// stages.
Optimizer::PassToken CreateRemoveUnusedInterfaceVariablesPass();

// Creates an empty pass.
// This is deprecated and will be removed.
// TODO(jaebaek): remove this pass after handling glslang's broken unit tests.
//                https://github.com/KhronosGroup/glslang/pull/2440
Optimizer::PassToken CreatePropagateLineInfoPass();

// Creates an empty pass.
// This is deprecated and will be removed.
// TODO(jaebaek): remove this pass after handling glslang's broken unit tests.
//                https://github.com/KhronosGroup/glslang/pull/2440
Optimizer::PassToken CreateRedundantLineInfoElimPass();

// Creates a compact ids pass.
// The pass remaps result ids to a compact and gapless range starting from %1.
Optimizer::PassToken CreateCompactIdsPass();

// Creates a remove duplicate pass.
// This pass removes various duplicates:
// * duplicate capabilities;
// * duplicate extended instruction imports;
// * duplicate types;
// * duplicate decorations.
Optimizer::PassToken CreateRemoveDuplicatesPass();

// Creates a CFG cleanup pass.
// This pass removes cruft from the control flow graph of functions that are
// reachable from entry points and exported functions. It currently includes the
// following functionality:
//
// - Removal of unreachable basic blocks.
Optimizer::PassToken CreateCFGCleanupPass();

// Create dead variable elimination pass.
// This pass will delete module scope variables, along with their decorations,
// that are not referenced.
Optimizer::PassToken CreateDeadVariableEliminationPass();

// create merge return pass.
// changes functions that have multiple return statements so they have a single
// return statement.
//
// for structured control flow it is assumed that the only unreachable blocks in
// the function are trivial merge and continue blocks.
//
// a trivial merge block contains the label and an opunreachable instructions,
// nothing else.  a trivial continue block contain a label and an opbranch to
// the header, nothing else.
//
// these conditions are guaranteed to be met after running dead-branch
// elimination.
Optimizer::PassToken CreateMergeReturnPass();

// Create value numbering pass.
// This pass will look for instructions in the same basic block that compute the
// same value, and remove the redundant ones.
Optimizer::PassToken CreateLocalRedundancyEliminationPass();

// Create LICM pass.
// This pass will look for invariant instructions inside loops and hoist them to
// the loops preheader.
Optimizer::PassToken CreateLoopInvariantCodeMotionPass();

// Creates a loop fission pass.
// This pass will split all top level loops whose register pressure exceedes the
// given |threshold|.
Optimizer::PassToken CreateLoopFissionPass(size_t threshold);

// Creates a loop fusion pass.
// This pass will look for adjacent loops that are compatible and legal to be
// fused. The fuse all such loops as long as the register usage for the fused
// loop stays under the threshold defined by |max_registers_per_loop|.
Optimizer::PassToken CreateLoopFusionPass(size_t max_registers_per_loop);

// Creates a loop peeling pass.
// This pass will look for conditions inside a loop that are true or false only
// for the N first or last iteration. For loop with such condition, those N
// iterations of the loop will be executed outside of the main loop.
// To limit code size explosion, the loop peeling can only happen if the code
// size growth for each loop is under |code_growth_threshold|.
Optimizer::PassToken CreateLoopPeelingPass();

// Creates a loop unswitch pass.
// This pass will look for loop independent branch conditions and move the
// condition out of the loop and version the loop based on the taken branch.
// Works best after LICM and local multi store elimination pass.
Optimizer::PassToken CreateLoopUnswitchPass();

// Create global value numbering pass.
// This pass will look for instructions where the same value is computed on all
// paths leading to the instruction.  Those instructions are deleted.
Optimizer::PassToken CreateRedundancyEliminationPass();

// Create scalar replacement pass.
// This pass replaces composite function scope variables with variables for each
// element if those elements are accessed individually.  The parameter is a
// limit on the number of members in the composite variable that the pass will
// consider replacing.
Optimizer::PassToken CreateScalarReplacementPass(uint32_t size_limit = 100);

// Create a private to local pass.
// This pass looks for variables declared in the private storage class that are
// used in only one function.  Those variables are moved to the function storage
// class in the function that they are used.
Optimizer::PassToken CreatePrivateToLocalPass();

// Creates a conditional constant propagation (CCP) pass.
// This pass implements the SSA-CCP algorithm in
//
//      Constant propagation with conditional branches,
//      Wegman and Zadeck, ACM TOPLAS 13(2):181-210.
//
// Constant values in expressions and conditional jumps are folded and
// simplified. This may reduce code size by removing never executed jump targets
// and computations with constant operands.
Optimizer::PassToken CreateCCPPass();

// Creates a workaround driver bugs pass.  This pass attempts to work around
// a known driver bug (issue #1209) by identifying the bad code sequences and
// rewriting them.
//
// Current workaround: Avoid OpUnreachable instructions in loops.
Optimizer::PassToken CreateWorkaround1209Pass();

// Creates a pass that converts if-then-else like assignments into OpSelect.
Optimizer::PassToken CreateIfConversionPass();

// Creates a pass that will replace instructions that are not valid for the
// current shader stage by constants.  Has no effect on non-shader modules.
Optimizer::PassToken CreateReplaceInvalidOpcodePass();

// Creates a pass that simplifies instructions using the instruction folder.
Optimizer::PassToken CreateSimplificationPass();

// Create loop unroller pass.
// Creates a pass to unroll loops which have the "Unroll" loop control
// mask set. The loops must meet a specific criteria in order to be unrolled
// safely this criteria is checked before doing the unroll by the
// LoopUtils::CanPerformUnroll method. Any loop that does not meet the criteria
// won't be unrolled. See CanPerformUnroll LoopUtils.h for more information.
Optimizer::PassToken CreateLoopUnrollPass(bool fully_unroll, int factor = 0);

// Create the SSA rewrite pass.
// This pass converts load/store operations on function local variables into
// operations on SSA IDs.  This allows SSA optimizers to act on these variables.
// Only variables that are local to the function and of supported types are
// processed (see IsSSATargetVar for details).
Optimizer::PassToken CreateSSARewritePass();

// Create pass to convert relaxed precision instructions to half precision.
// This pass converts as many relaxed float32 arithmetic operations to half as
// possible. It converts any float32 operands to half if needed. It converts
// any resulting half precision values back to float32 as needed. No variables
// are changed. No image operations are changed.
//
// Best if run after function scope store/load and composite operation
// eliminations are run. Also best if followed by instruction simplification,
// redundancy elimination and DCE.
Optimizer::PassToken CreateConvertRelaxedToHalfPass();

// Create relax float ops pass.
// This pass decorates all float32 result instructions with RelaxedPrecision
// if not already so decorated.
Optimizer::PassToken CreateRelaxFloatOpsPass();

// Create copy propagate arrays pass.
// This pass looks to copy propagate memory references for arrays.  It looks
// for specific code patterns to recognize array copies.
Optimizer::PassToken CreateCopyPropagateArraysPass();

// Create a vector dce pass.
// This pass looks for components of vectors that are unused, and removes them
// from the vector.  Note this would still leave around lots of dead code that
// a pass of ADCE will be able to remove.
Optimizer::PassToken CreateVectorDCEPass();

// Create a pass to reduce the size of loads.
// This pass looks for loads of structures where only a few of its members are
// used.  It replaces the loads feeding an OpExtract with an OpAccessChain and
// a load of the specific elements.  The parameter is a threshold to determine
// whether we have to replace the load or not.  If the ratio of the used
// components of the load is less than the threshold, we replace the load.
Optimizer::PassToken CreateReduceLoadSizePass(
    double load_replacement_threshold = 0.9);

// Create a pass to combine chained access chains.
// This pass looks for access chains fed by other access chains and combines
// them into a single instruction where possible.
Optimizer::PassToken CreateCombineAccessChainsPass();

// Create a pass to instrument bindless descriptor checking
// This pass instruments all bindless references to check that descriptor
// array indices are inbounds, and if the descriptor indexing extension is
// enabled, that the descriptor has been initialized. If the reference is
// invalid, a record is written to the debug output buffer (if space allows)
// and a null value is returned. This pass is designed to support bindless
// validation in the Vulkan validation layers.
//
// TODO(greg-lunarg): Add support for buffer references. Currently only does
// checking for image references.
//
// Dead code elimination should be run after this pass as the original,
// potentially invalid code is not removed and could cause undefined behavior,
// including crashes. It may also be beneficial to run Simplification
// (ie Constant Propagation), DeadBranchElim and BlockMerge after this pass to
// optimize instrument code involving the testing of compile-time constants.
// It is also generally recommended that this pass (and all
// instrumentation passes) be run after any legalization and optimization
// passes. This will give better analysis for the instrumentation and avoid
// potentially de-optimizing the instrument code, for example, inlining
// the debug record output function throughout the module.
//
// The instrumentation will read and write buffers in debug
// descriptor set |desc_set|. It will write |shader_id| in each output record
// to identify the shader module which generated the record.
// |desc_length_enable| controls instrumentation of runtime descriptor array
// references, |desc_init_enable| controls instrumentation of descriptor
// initialization checking, and |buff_oob_enable| controls instrumentation
// of storage and uniform buffer bounds checking, all of which require input
// buffer support. |texbuff_oob_enable| controls instrumentation of texel
// buffers, which does not require input buffer support.
Optimizer::PassToken CreateInstBindlessCheckPass(
    uint32_t desc_set, uint32_t shader_id, bool desc_length_enable = false,
    bool desc_init_enable = false, bool buff_oob_enable = false,
    bool texbuff_oob_enable = false);

// Create a pass to instrument physical buffer address checking
// This pass instruments all physical buffer address references to check that
// all referenced bytes fall in a valid buffer. If the reference is
// invalid, a record is written to the debug output buffer (if space allows)
// and a null value is returned. This pass is designed to support buffer
// address validation in the Vulkan validation layers.
//
// Dead code elimination should be run after this pass as the original,
// potentially invalid code is not removed and could cause undefined behavior,
// including crashes. Instruction simplification would likely also be
// beneficial. It is also generally recommended that this pass (and all
// instrumentation passes) be run after any legalization and optimization
// passes. This will give better analysis for the instrumentation and avoid
// potentially de-optimizing the instrument code, for example, inlining
// the debug record output function throughout the module.
//
// The instrumentation will read and write buffers in debug
// descriptor set |desc_set|. It will write |shader_id| in each output record
// to identify the shader module which generated the record.
Optimizer::PassToken CreateInstBuffAddrCheckPass(uint32_t desc_set,
                                                 uint32_t shader_id);

// Create a pass to instrument OpDebugPrintf instructions.
// This pass replaces all OpDebugPrintf instructions with instructions to write
// a record containing the string id and the all specified values into a special
// printf output buffer (if space allows). This pass is designed to support
// the printf validation in the Vulkan validation layers.
//
// The instrumentation will write buffers in debug descriptor set |desc_set|.
// It will write |shader_id| in each output record to identify the shader
// module which generated the record.
Optimizer::PassToken CreateInstDebugPrintfPass(uint32_t desc_set,
                                               uint32_t shader_id);

// Create a pass to upgrade to the VulkanKHR memory model.
// This pass upgrades the Logical GLSL450 memory model to Logical VulkanKHR.
// Additionally, it modifies memory, image, atomic and barrier operations to
// conform to that model's requirements.
Optimizer::PassToken CreateUpgradeMemoryModelPass();

// Create a pass to do code sinking.  Code sinking is a transformation
// where an instruction is moved into a more deeply nested construct.
Optimizer::PassToken CreateCodeSinkingPass();

// Create a pass to fix incorrect storage classes.  In order to make code
// generation simpler, DXC may generate code where the storage classes do not
// match up correctly.  This pass will fix the errors that it can.
Optimizer::PassToken CreateFixStorageClassPass();

// Creates a graphics robust access pass.
//
// This pass injects code to clamp indexed accesses to buffers and internal
// arrays, providing guarantees satisfying Vulkan's robustBufferAccess rules.
//
// TODO(dneto): Clamps coordinates and sample index for pointer calculations
// into storage images (OpImageTexelPointer).  For an cube array image, it
// assumes the maximum layer count times 6 is at most 0xffffffff.
//
// NOTE: This pass will fail with a message if:
// - The module is not a Shader module.
// - The module declares VariablePointers, VariablePointersStorageBuffer, or
//   RuntimeDescriptorArrayEXT capabilities.
// - The module uses an addressing model other than Logical
// - Access chain indices are wider than 64 bits.
// - Access chain index for a struct is not an OpConstant integer or is out
//   of range. (The module is already invalid if that is the case.)
// - TODO(dneto): The OpImageTexelPointer coordinate component is not 32-bits
// wide.
//
// NOTE: Access chain indices are always treated as signed integers.  So
//   if an array has a fixed size of more than 2^31 elements, then elements
//   from 2^31 and above are never accessible with a 32-bit index,
//   signed or unsigned.  For this case, this pass will clamp the index
//   between 0 and at 2^31-1, inclusive.
//   Similarly, if an array has more then 2^15 element and is accessed with
//   a 16-bit index, then elements from 2^15 and above are not accessible.
//   In this case, the pass will clamp the index between 0 and 2^15-1
//   inclusive.
Optimizer::PassToken CreateGraphicsRobustAccessPass();

// Create a pass to spread Volatile semantics to variables with SMIDNV,
// WarpIDNV, SubgroupSize, SubgroupLocalInvocationId, SubgroupEqMask,
// SubgroupGeMask, SubgroupGtMask, SubgroupLeMask, or SubgroupLtMask BuiltIn
// decorations or OpLoad for them when the shader model is the ray generation,
// closest hit, miss, intersection, or callable. This pass can be used for
// VUID-StandaloneSpirv-VulkanMemoryModel-04678 and
// VUID-StandaloneSpirv-VulkanMemoryModel-04679 (See "Standalone SPIR-V
// Validation" section of Vulkan spec "Appendix A: Vulkan Environment for
// SPIR-V"). When the SPIR-V version is 1.6 or above, the pass also spreads
// the Volatile semantics to a variable with HelperInvocation BuiltIn decoration
// in the fragement shader.
Optimizer::PassToken CreateSpreadVolatileSemanticsPass();

// Create a pass to replace a descriptor access using variable index.
// This pass replaces every access using a variable index to array variable
// |desc| that has a DescriptorSet and Binding decorations with a constant
// element of the array. In order to replace the access using a variable index
// with the constant element, it uses a switch statement.
Optimizer::PassToken CreateReplaceDescArrayAccessUsingVarIndexPass();

// Create descriptor scalar replacement pass.
// This pass replaces every array variable |desc| that has a DescriptorSet and
// Binding decorations with a new variable for each element of the array.
// Suppose |desc| was bound at binding |b|.  Then the variable corresponding to
// |desc[i]| will have binding |b+i|.  The descriptor set will be the same.  It
// is assumed that no other variable already has a binding that will used by one
// of the new variables.  If not, the pass will generate invalid Spir-V.  All
// accesses to |desc| must be OpAccessChain instructions with a literal index
// for the first index.
Optimizer::PassToken CreateDescriptorScalarReplacementPass();

// Create a pass to replace each OpKill instruction with a function call to a
// function that has a single OpKill.  Also replace each OpTerminateInvocation
// instruction  with a function call to a function that has a single
// OpTerminateInvocation.  This allows more code to be inlined.
Optimizer::PassToken CreateWrapOpKillPass();

// Replaces the extensions VK_AMD_shader_ballot,VK_AMD_gcn_shader, and
// VK_AMD_shader_trinary_minmax with equivalent code using core instructions and
// capabilities.
Optimizer::PassToken CreateAmdExtToKhrPass();

// Replaces the internal version of GLSLstd450 InterpolateAt* extended
// instructions with the externally valid version. The internal version allows
// an OpLoad of the interpolant for the first argument. This pass removes the
// OpLoad and replaces it with its pointer. glslang and possibly other
// frontends will create the internal version for HLSL. This pass will be part
// of HLSL legalization and should be called after interpolants have been
// propagated into their final positions.
Optimizer::PassToken CreateInterpolateFixupPass();

// Removes unused components from composite input variables. Current
// implementation just removes trailing unused components from input arrays
// and structs. The pass performs best after maximizing dead code removal.
// A subsequent dead code elimination pass would be beneficial in removing
// newly unused component types.
//
// WARNING: This pass can only be safely applied standalone to vertex shaders
// as it can otherwise cause interface incompatibilities with the preceding
// shader in the pipeline. If applied to non-vertex shaders, the user should
// follow by applying EliminateDeadOutputStores and
// EliminateDeadOutputComponents to the preceding shader.
Optimizer::PassToken CreateEliminateDeadInputComponentsPass();

// Removes unused components from composite output variables. Current
// implementation just removes trailing unused components from output arrays
// and structs. The pass performs best after eliminating dead output stores.
// A subsequent dead code elimination pass would be beneficial in removing
// newly unused component types. Currently only supports vertex and fragment
// shaders.
//
// WARNING: This pass cannot be safely applied standalone as it can cause
// interface incompatibility with the following shader in the pipeline. The
// user should first apply EliminateDeadInputComponents to the following
// shader, then apply EliminateDeadOutputStores to this shader.
Optimizer::PassToken CreateEliminateDeadOutputComponentsPass();

// Removes unused components from composite input variables. This safe
// version will not cause interface incompatibilities since it only changes
// vertex shaders. The current implementation just removes trailing unused
// components from input structs and input arrays. The pass performs best
// after maximizing dead code removal. A subsequent dead code elimination
// pass would be beneficial in removing newly unused component types.
Optimizer::PassToken CreateEliminateDeadInputComponentsSafePass();

// Analyzes shader and populates |live_locs| and |live_builtins|. Best results
// will be obtained if shader has all dead code eliminated first. |live_locs|
// and |live_builtins| are subsequently used when calling
// CreateEliminateDeadOutputStoresPass on the preceding shader. Currently only
// supports tesc, tese, geom, and frag shaders.
Optimizer::PassToken CreateAnalyzeLiveInputPass(
    std::unordered_set<uint32_t>* live_locs,
    std::unordered_set<uint32_t>* live_builtins);

// Removes stores to output locations not listed in |live_locs| or
// |live_builtins|. Best results are obtained if constant propagation is
// performed first. A subsequent call to ADCE will eliminate any dead code
// created by the removal of the stores. A subsequent call to
// CreateEliminateDeadOutputComponentsPass will eliminate any dead output
// components created by the elimination of the stores. Currently only supports
// vert, tesc, tese, and geom shaders.
Optimizer::PassToken CreateEliminateDeadOutputStoresPass(
    std::unordered_set<uint32_t>* live_locs,
    std::unordered_set<uint32_t>* live_builtins);

// Creates a convert-to-sampled-image pass to convert images and/or
// samplers with given pairs of descriptor set and binding to sampled image.
// If a pair of an image and a sampler have the same pair of descriptor set and
// binding that is one of the given pairs, they will be converted to a sampled
// image. In addition, if only an image has the descriptor set and binding that
// is one of the given pairs, it will be converted to a sampled image as well.
Optimizer::PassToken CreateConvertToSampledImagePass(
    const std::vector<opt::DescriptorSetAndBinding>&
        descriptor_set_binding_pairs);

// Create an interface-variable-scalar-replacement pass that replaces array or
// matrix interface variables with a series of scalar or vector interface
// variables. For example, it replaces `float3 foo[2]` with `float3 foo0, foo1`.
Optimizer::PassToken CreateInterfaceVariableScalarReplacementPass();

// Creates a remove-dont-inline pass to remove the |DontInline| function control
// from every function in the module.  This is useful if you want the inliner to
// inline these functions some reason.
Optimizer::PassToken CreateRemoveDontInlinePass();
// Create a fix-func-call-param pass to fix non memory argument for the function
// call, as spirv-validation requires function parameters to be an memory
// object, currently the pass would remove accesschain pointer argument passed
// to the function
Optimizer::PassToken CreateFixFuncCallArgumentsPass();
}  // namespace spvtools

#endif  // INCLUDE_SPIRV_TOOLS_OPTIMIZER_HPP_
