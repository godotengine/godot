// Copyright (c) 2015-2020 The Khronos Group Inc.
// Modifications Copyright (C) 2020 Advanced Micro Devices, Inc. All rights
// reserved.
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

// This file contains a disassembler:  It converts a SPIR-V binary
// to text.

#include "source/disassemble.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <memory>
#include <set>
#include <stack>
#include <unordered_map>
#include <utility>

#include "source/assembly_grammar.h"
#include "source/binary.h"
#include "source/diagnostic.h"
#include "source/ext_inst.h"
#include "source/opcode.h"
#include "source/parsed_operand.h"
#include "source/print.h"
#include "source/spirv_constant.h"
#include "source/spirv_endian.h"
#include "source/util/hex_float.h"
#include "source/util/make_unique.h"
#include "include/spirv-tools/libspirv.h"

namespace spvtools {
namespace {

// Indices to ControlFlowGraph's list of blocks from one block to its successors
struct BlockSuccessors {
  // Merge block in OpLoopMerge and OpSelectionMerge
  uint32_t merge_block_id = 0;
  // The continue block in OpLoopMerge
  uint32_t continue_block_id = 0;
  // The true and false blocks in OpBranchConditional
  uint32_t true_block_id = 0;
  uint32_t false_block_id = 0;
  // The body block of a loop, as specified by OpBranch after a merge
  // instruction
  uint32_t body_block_id = 0;
  // The same-nesting-level block that follows this one, indicated by an
  // OpBranch with no merge instruction.
  uint32_t next_block_id = 0;
  // The cases (including default) of an OpSwitch
  std::vector<uint32_t> case_block_ids;
};

class ParsedInstruction {
 public:
  ParsedInstruction(const spv_parsed_instruction_t* instruction) {
    // Make a copy of the parsed instruction, including stable memory for its
    // operands.
    instruction_ = *instruction;
    operands_ =
        std::make_unique<spv_parsed_operand_t[]>(instruction->num_operands);
    memcpy(operands_.get(), instruction->operands,
           instruction->num_operands * sizeof(*instruction->operands));
    instruction_.operands = operands_.get();
  }

  const spv_parsed_instruction_t* get() const { return &instruction_; }

 private:
  spv_parsed_instruction_t instruction_;
  std::unique_ptr<spv_parsed_operand_t[]> operands_;
};

// One block in the CFG
struct SingleBlock {
  // The byte offset in the SPIR-V where the block starts.  Used for printing in
  // a comment.
  size_t byte_offset;

  // Block instructions
  std::vector<ParsedInstruction> instructions;

  // Successors of this block
  BlockSuccessors successors;

  // The nesting level for this block.
  uint32_t nest_level = 0;
  bool nest_level_assigned = false;

  // Whether the block was reachable
  bool reachable = false;
};

// CFG for one function
struct ControlFlowGraph {
  std::vector<SingleBlock> blocks;
};

// A Disassembler instance converts a SPIR-V binary to its assembly
// representation.
class Disassembler {
 public:
  Disassembler(const AssemblyGrammar& grammar, uint32_t options,
               NameMapper name_mapper)
      : print_(spvIsInBitfield(SPV_BINARY_TO_TEXT_OPTION_PRINT, options)),
        nested_indent_(
            spvIsInBitfield(SPV_BINARY_TO_TEXT_OPTION_NESTED_INDENT, options)),
        reorder_blocks_(
            spvIsInBitfield(SPV_BINARY_TO_TEXT_OPTION_REORDER_BLOCKS, options)),
        text_(),
        out_(print_ ? out_stream() : out_stream(text_)),
        instruction_disassembler_(grammar, out_.get(), options, name_mapper),
        header_(!spvIsInBitfield(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER, options)),
        byte_offset_(0) {}

  // Emits the assembly header for the module, and sets up internal state
  // so subsequent callbacks can handle the cases where the entire module
  // is either big-endian or little-endian.
  spv_result_t HandleHeader(spv_endianness_t endian, uint32_t version,
                            uint32_t generator, uint32_t id_bound,
                            uint32_t schema);
  // Emits the assembly text for the given instruction.
  spv_result_t HandleInstruction(const spv_parsed_instruction_t& inst);

  // If not printing, populates text_result with the accumulated text.
  // Returns SPV_SUCCESS on success.
  spv_result_t SaveTextResult(spv_text* text_result) const;

 private:
  void EmitCFG();

  const bool print_;  // Should we also print to the standard output stream?
  const bool nested_indent_;  // Should the blocks be indented according to the
                              // control flow structure?
  const bool
      reorder_blocks_;       // Should the blocks be reordered for readability?
  spv_endianness_t endian_;  // The detected endianness of the binary.
  std::stringstream text_;   // Captures the text, if not printing.
  out_stream out_;  // The Output stream.  Either to text_ or standard output.
  disassemble::InstructionDisassembler instruction_disassembler_;
  const bool header_;   // Should we output header as the leading comment?
  size_t byte_offset_;  // The number of bytes processed so far.
  bool inserted_decoration_space_ = false;
  bool inserted_debug_space_ = false;
  bool inserted_type_space_ = false;

  // The CFG for the current function
  ControlFlowGraph current_function_cfg_;
};

spv_result_t Disassembler::HandleHeader(spv_endianness_t endian,
                                        uint32_t version, uint32_t generator,
                                        uint32_t id_bound, uint32_t schema) {
  endian_ = endian;

  if (header_) {
    instruction_disassembler_.EmitHeaderSpirv();
    instruction_disassembler_.EmitHeaderVersion(version);
    instruction_disassembler_.EmitHeaderGenerator(generator);
    instruction_disassembler_.EmitHeaderIdBound(id_bound);
    instruction_disassembler_.EmitHeaderSchema(schema);
  }

  byte_offset_ = SPV_INDEX_INSTRUCTION * sizeof(uint32_t);

  return SPV_SUCCESS;
}

spv_result_t Disassembler::HandleInstruction(
    const spv_parsed_instruction_t& inst) {
  instruction_disassembler_.EmitSectionComment(inst, inserted_decoration_space_,
                                               inserted_debug_space_,
                                               inserted_type_space_);

  // When nesting needs to be calculated or when the blocks are reordered, we
  // have to have the full picture of the CFG first.  Defer processing of the
  // instructions until the entire function is visited.  This is not done
  // without those options (even if simpler) to improve debuggability; for
  // example to be able to see whatever is parsed so far even if there is a
  // parse error.
  if (nested_indent_ || reorder_blocks_) {
    switch (static_cast<spv::Op>(inst.opcode)) {
      case spv::Op::OpLabel: {
        // Add a new block to the CFG
        SingleBlock new_block;
        new_block.byte_offset = byte_offset_;
        new_block.instructions.emplace_back(&inst);
        current_function_cfg_.blocks.push_back(std::move(new_block));
        break;
      }
      case spv::Op::OpFunctionEnd:
        // Process the CFG and output the instructions
        EmitCFG();
        // Output OpFunctionEnd itself too
        [[fallthrough]];
      default:
        if (!current_function_cfg_.blocks.empty()) {
          // If in a function, stash the instruction for later.
          current_function_cfg_.blocks.back().instructions.emplace_back(&inst);
        } else {
          // Otherwise emit the instruction right away.
          instruction_disassembler_.EmitInstruction(inst, byte_offset_);
        }
        break;
    }
  } else {
    instruction_disassembler_.EmitInstruction(inst, byte_offset_);
  }

  byte_offset_ += inst.num_words * sizeof(uint32_t);

  return SPV_SUCCESS;
}

// Helper to get the operand of an instruction as an id.
uint32_t GetOperand(const spv_parsed_instruction_t* instruction,
                    uint32_t operand) {
  return instruction->words[instruction->operands[operand].offset];
}

std::unordered_map<uint32_t, uint32_t> BuildControlFlowGraph(
    ControlFlowGraph& cfg) {
  std::unordered_map<uint32_t, uint32_t> id_to_index;

  for (size_t index = 0; index < cfg.blocks.size(); ++index) {
    SingleBlock& block = cfg.blocks[index];

    // For future use, build the ID->index map
    assert(static_cast<spv::Op>(block.instructions[0].get()->opcode) ==
           spv::Op::OpLabel);
    const uint32_t id = block.instructions[0].get()->result_id;

    id_to_index[id] = static_cast<uint32_t>(index);

    // Look for a merge instruction first.  The function of OpBranch depends on
    // that.
    if (block.instructions.size() >= 3) {
      const spv_parsed_instruction_t* maybe_merge =
          block.instructions[block.instructions.size() - 2].get();

      switch (static_cast<spv::Op>(maybe_merge->opcode)) {
        case spv::Op::OpLoopMerge:
          block.successors.merge_block_id = GetOperand(maybe_merge, 0);
          block.successors.continue_block_id = GetOperand(maybe_merge, 1);
          break;

        case spv::Op::OpSelectionMerge:
          block.successors.merge_block_id = GetOperand(maybe_merge, 0);
          break;

        default:
          break;
      }
    }

    // Then look at the last instruction; it must be a branch
    assert(block.instructions.size() >= 2);

    const spv_parsed_instruction_t* branch = block.instructions.back().get();
    switch (static_cast<spv::Op>(branch->opcode)) {
      case spv::Op::OpBranch:
        if (block.successors.merge_block_id != 0) {
          block.successors.body_block_id = GetOperand(branch, 0);
        } else {
          block.successors.next_block_id = GetOperand(branch, 0);
        }
        break;

      case spv::Op::OpBranchConditional:
        block.successors.true_block_id = GetOperand(branch, 1);
        block.successors.false_block_id = GetOperand(branch, 2);
        break;

      case spv::Op::OpSwitch:
        for (uint32_t case_index = 1; case_index < branch->num_operands;
             case_index += 2) {
          block.successors.case_block_ids.push_back(
              GetOperand(branch, case_index));
        }
        break;

      default:
        break;
    }
  }

  return id_to_index;
}

// Helper to deal with nesting and non-existing ids / previously-assigned
// levels.  It assigns a given nesting level `level` to the block identified by
// `id` (unless that block already has a nesting level assigned).
void Nest(ControlFlowGraph& cfg,
          const std::unordered_map<uint32_t, uint32_t>& id_to_index,
          uint32_t id, uint32_t level) {
  if (id == 0) {
    return;
  }

  const uint32_t block_index = id_to_index.at(id);
  SingleBlock& block = cfg.blocks[block_index];

  if (!block.nest_level_assigned) {
    block.nest_level = level;
    block.nest_level_assigned = true;
  }
}

// For a given block, assign nesting level to its successors.
void NestSuccessors(ControlFlowGraph& cfg, const SingleBlock& block,
                    const std::unordered_map<uint32_t, uint32_t>& id_to_index) {
  assert(block.nest_level_assigned);

  // Nest loops as such:
  //
  //     %loop = OpLabel
  //               OpLoopMerge %merge %cont ...
  //               OpBranch %body
  //     %body =     OpLabel
  //                   Op...
  //     %cont =   OpLabel
  //                 Op...
  //    %merge = OpLabel
  //               Op...
  //
  // Nest conditional branches as such:
  //
  //   %header = OpLabel
  //               OpSelectionMerge %merge ...
  //               OpBranchConditional ... %true %false
  //     %true =     OpLabel
  //                   Op...
  //    %false =     OpLabel
  //                   Op...
  //    %merge = OpLabel
  //               Op...
  //
  // Nest switch/case as such:
  //
  //   %header = OpLabel
  //               OpSelectionMerge %merge ...
  //               OpSwitch ... %default ... %case0 ... %case1 ...
  //  %default =     OpLabel
  //                   Op...
  //    %case0 =     OpLabel
  //                   Op...
  //    %case1 =     OpLabel
  //                   Op...
  //             ...
  //    %merge = OpLabel
  //               Op...
  //
  // The following can be observed:
  //
  // - In all cases, the merge block has the same nesting as this block
  // - The continue block of loops is nested 1 level deeper
  // - The body/branches/cases are nested 2 levels deeper
  //
  // Back branches to the header block, branches to the merge block, etc
  // are correctly handled by processing the header block first (that is
  // _this_ block, already processed), then following the above rules
  // (in the same order) for any block that is not already processed.
  Nest(cfg, id_to_index, block.successors.merge_block_id, block.nest_level);
  Nest(cfg, id_to_index, block.successors.continue_block_id,
       block.nest_level + 1);
  Nest(cfg, id_to_index, block.successors.true_block_id, block.nest_level + 2);
  Nest(cfg, id_to_index, block.successors.false_block_id, block.nest_level + 2);
  Nest(cfg, id_to_index, block.successors.body_block_id, block.nest_level + 2);
  Nest(cfg, id_to_index, block.successors.next_block_id, block.nest_level);
  for (uint32_t case_block_id : block.successors.case_block_ids) {
    Nest(cfg, id_to_index, case_block_id, block.nest_level + 2);
  }
}

struct StackEntry {
  // The index of the block (in ControlFlowGraph::blocks) to process.
  uint32_t block_index;
  // Whether this is the pre or post visit of the block.  Because a post-visit
  // traversal is needed, the same block is pushed back on the stack on
  // pre-visit so it can be visited again on post-visit.
  bool post_visit = false;
};

// Helper to deal with DFS traversal and non-existing ids
void VisitSuccesor(std::stack<StackEntry>* dfs_stack,
                   const std::unordered_map<uint32_t, uint32_t>& id_to_index,
                   uint32_t id) {
  if (id != 0) {
    dfs_stack->push({id_to_index.at(id), false});
  }
}

// Given the control flow graph, calculates and returns the reverse post-order
// ordering of the blocks.  The blocks are then disassembled in that order for
// readability.
std::vector<uint32_t> OrderBlocks(
    ControlFlowGraph& cfg,
    const std::unordered_map<uint32_t, uint32_t>& id_to_index) {
  std::vector<uint32_t> post_order;

  // Nest level of a function's first block is 0.
  cfg.blocks[0].nest_level = 0;
  cfg.blocks[0].nest_level_assigned = true;

  // Stack of block indices as they are visited.
  std::stack<StackEntry> dfs_stack;
  dfs_stack.push({0, false});

  std::set<uint32_t> visited;

  while (!dfs_stack.empty()) {
    const uint32_t block_index = dfs_stack.top().block_index;
    const bool post_visit = dfs_stack.top().post_visit;
    dfs_stack.pop();

    // If this is the second time the block is visited, that's the post-order
    // visit.
    if (post_visit) {
      post_order.push_back(block_index);
      continue;
    }

    // If already visited, another path got to it first (like a case
    // fallthrough), avoid reprocessing it.
    if (visited.count(block_index) > 0) {
      continue;
    }
    visited.insert(block_index);

    // Push it back in the stack for post-order visit
    dfs_stack.push({block_index, true});

    SingleBlock& block = cfg.blocks[block_index];

    // Assign nest levels of successors right away.  The successors are either
    // nested under this block, or are back or forward edges to blocks outside
    // this nesting level (no farther than the merge block), whose nesting
    // levels are already assigned before this block is visited.
    NestSuccessors(cfg, block, id_to_index);
    block.reachable = true;

    // The post-order visit yields the order in which the blocks are naturally
    // ordered _backwards_. So blocks to be ordered last should be visited
    // first.  In other words, they should be pushed to the DFS stack last.
    VisitSuccesor(&dfs_stack, id_to_index, block.successors.true_block_id);
    VisitSuccesor(&dfs_stack, id_to_index, block.successors.false_block_id);
    VisitSuccesor(&dfs_stack, id_to_index, block.successors.body_block_id);
    VisitSuccesor(&dfs_stack, id_to_index, block.successors.next_block_id);
    for (uint32_t case_block_id : block.successors.case_block_ids) {
      VisitSuccesor(&dfs_stack, id_to_index, case_block_id);
    }
    VisitSuccesor(&dfs_stack, id_to_index, block.successors.continue_block_id);
    VisitSuccesor(&dfs_stack, id_to_index, block.successors.merge_block_id);
  }

  std::vector<uint32_t> order(post_order.rbegin(), post_order.rend());

  // Finally, dump all unreachable blocks at the end
  for (size_t index = 0; index < cfg.blocks.size(); ++index) {
    SingleBlock& block = cfg.blocks[index];

    if (!block.reachable) {
      order.push_back(static_cast<uint32_t>(index));
      block.nest_level = 0;
      block.nest_level_assigned = true;
    }
  }

  return order;
}

void Disassembler::EmitCFG() {
  // Build the CFG edges.  At the same time, build an ID->block index map to
  // simplify building the CFG edges.
  const std::unordered_map<uint32_t, uint32_t> id_to_index =
      BuildControlFlowGraph(current_function_cfg_);

  // Walk the CFG in reverse post-order to find the best ordering of blocks for
  // presentation
  std::vector<uint32_t> block_order =
      OrderBlocks(current_function_cfg_, id_to_index);
  assert(block_order.size() == current_function_cfg_.blocks.size());

  // Walk the CFG either in block order or input order based on whether the
  // reorder_blocks_ option is given.
  for (uint32_t index = 0; index < current_function_cfg_.blocks.size();
       ++index) {
    const uint32_t block_index = reorder_blocks_ ? block_order[index] : index;
    const SingleBlock& block = current_function_cfg_.blocks[block_index];

    // Emit instructions for this block
    size_t byte_offset = block.byte_offset;
    assert(block.nest_level_assigned);

    for (const ParsedInstruction& inst : block.instructions) {
      instruction_disassembler_.EmitInstructionInBlock(*inst.get(), byte_offset,
                                                       block.nest_level);
      byte_offset += inst.get()->num_words * sizeof(uint32_t);
    }
  }

  current_function_cfg_.blocks.clear();
}

spv_result_t Disassembler::SaveTextResult(spv_text* text_result) const {
  if (!print_) {
    size_t length = text_.str().size();
    char* str = new char[length + 1];
    if (!str) return SPV_ERROR_OUT_OF_MEMORY;
    strncpy(str, text_.str().c_str(), length + 1);
    spv_text text = new spv_text_t();
    if (!text) {
      delete[] str;
      return SPV_ERROR_OUT_OF_MEMORY;
    }
    text->str = str;
    text->length = length;
    *text_result = text;
  }
  return SPV_SUCCESS;
}

spv_result_t DisassembleHeader(void* user_data, spv_endianness_t endian,
                               uint32_t /* magic */, uint32_t version,
                               uint32_t generator, uint32_t id_bound,
                               uint32_t schema) {
  assert(user_data);
  auto disassembler = static_cast<Disassembler*>(user_data);
  return disassembler->HandleHeader(endian, version, generator, id_bound,
                                    schema);
}

spv_result_t DisassembleInstruction(
    void* user_data, const spv_parsed_instruction_t* parsed_instruction) {
  assert(user_data);
  auto disassembler = static_cast<Disassembler*>(user_data);
  return disassembler->HandleInstruction(*parsed_instruction);
}

// Simple wrapper class to provide extra data necessary for targeted
// instruction disassembly.
class WrappedDisassembler {
 public:
  WrappedDisassembler(Disassembler* dis, const uint32_t* binary, size_t wc)
      : disassembler_(dis), inst_binary_(binary), word_count_(wc) {}

  Disassembler* disassembler() { return disassembler_; }
  const uint32_t* inst_binary() const { return inst_binary_; }
  size_t word_count() const { return word_count_; }

 private:
  Disassembler* disassembler_;
  const uint32_t* inst_binary_;
  const size_t word_count_;
};

spv_result_t DisassembleTargetHeader(void* user_data, spv_endianness_t endian,
                                     uint32_t /* magic */, uint32_t version,
                                     uint32_t generator, uint32_t id_bound,
                                     uint32_t schema) {
  assert(user_data);
  auto wrapped = static_cast<WrappedDisassembler*>(user_data);
  return wrapped->disassembler()->HandleHeader(endian, version, generator,
                                               id_bound, schema);
}

spv_result_t DisassembleTargetInstruction(
    void* user_data, const spv_parsed_instruction_t* parsed_instruction) {
  assert(user_data);
  auto wrapped = static_cast<WrappedDisassembler*>(user_data);
  // Check if this is the instruction we want to disassemble.
  if (wrapped->word_count() == parsed_instruction->num_words &&
      std::equal(wrapped->inst_binary(),
                 wrapped->inst_binary() + wrapped->word_count(),
                 parsed_instruction->words)) {
    // Found the target instruction. Disassemble it and signal that we should
    // stop searching so we don't output the same instruction again.
    if (auto error =
            wrapped->disassembler()->HandleInstruction(*parsed_instruction))
      return error;
    return SPV_REQUESTED_TERMINATION;
  }
  return SPV_SUCCESS;
}

uint32_t GetLineLengthWithoutColor(const std::string line) {
  // Currently, every added color is in the form \x1b...m, so instead of doing a
  // lot of string comparisons with spvtools::clr::* strings, we just ignore
  // those ranges.
  uint32_t length = 0;
  for (size_t i = 0; i < line.size(); ++i) {
    if (line[i] == '\x1b') {
      do {
        ++i;
      } while (i < line.size() && line[i] != 'm');
      continue;
    }

    ++length;
  }

  return length;
}

constexpr int kStandardIndent = 15;
constexpr int kBlockNestIndent = 2;
constexpr int kBlockBodyIndentOffset = 2;
constexpr uint32_t kCommentColumn = 50;
}  // namespace

namespace disassemble {
InstructionDisassembler::InstructionDisassembler(const AssemblyGrammar& grammar,
                                                 std::ostream& stream,
                                                 uint32_t options,
                                                 NameMapper name_mapper)
    : grammar_(grammar),
      stream_(stream),
      print_(spvIsInBitfield(SPV_BINARY_TO_TEXT_OPTION_PRINT, options)),
      color_(spvIsInBitfield(SPV_BINARY_TO_TEXT_OPTION_COLOR, options)),
      indent_(spvIsInBitfield(SPV_BINARY_TO_TEXT_OPTION_INDENT, options)
                  ? kStandardIndent
                  : 0),
      nested_indent_(
          spvIsInBitfield(SPV_BINARY_TO_TEXT_OPTION_NESTED_INDENT, options)),
      comment_(spvIsInBitfield(SPV_BINARY_TO_TEXT_OPTION_COMMENT, options)),
      show_byte_offset_(
          spvIsInBitfield(SPV_BINARY_TO_TEXT_OPTION_SHOW_BYTE_OFFSET, options)),
      name_mapper_(std::move(name_mapper)),
      last_instruction_comment_alignment_(0) {}

void InstructionDisassembler::EmitHeaderSpirv() { stream_ << "; SPIR-V\n"; }

void InstructionDisassembler::EmitHeaderVersion(uint32_t version) {
  stream_ << "; Version: " << SPV_SPIRV_VERSION_MAJOR_PART(version) << "."
          << SPV_SPIRV_VERSION_MINOR_PART(version) << "\n";
}

void InstructionDisassembler::EmitHeaderGenerator(uint32_t generator) {
  const char* generator_tool =
      spvGeneratorStr(SPV_GENERATOR_TOOL_PART(generator));
  stream_ << "; Generator: " << generator_tool;
  // For unknown tools, print the numeric tool value.
  if (0 == strcmp("Unknown", generator_tool)) {
    stream_ << "(" << SPV_GENERATOR_TOOL_PART(generator) << ")";
  }
  // Print the miscellaneous part of the generator word on the same
  // line as the tool name.
  stream_ << "; " << SPV_GENERATOR_MISC_PART(generator) << "\n";
}

void InstructionDisassembler::EmitHeaderIdBound(uint32_t id_bound) {
  stream_ << "; Bound: " << id_bound << "\n";
}

void InstructionDisassembler::EmitHeaderSchema(uint32_t schema) {
  stream_ << "; Schema: " << schema << "\n";
}

void InstructionDisassembler::EmitInstruction(
    const spv_parsed_instruction_t& inst, size_t inst_byte_offset) {
  EmitInstructionImpl(inst, inst_byte_offset, 0, false);
}

void InstructionDisassembler::EmitInstructionInBlock(
    const spv_parsed_instruction_t& inst, size_t inst_byte_offset,
    uint32_t block_indent) {
  EmitInstructionImpl(inst, inst_byte_offset, block_indent, true);
}

void InstructionDisassembler::EmitInstructionImpl(
    const spv_parsed_instruction_t& inst, size_t inst_byte_offset,
    uint32_t block_indent, bool is_in_block) {
  auto opcode = static_cast<spv::Op>(inst.opcode);

  // To better align the comments (if any), write the instruction to a line
  // first so its length can be readily available.
  std::ostringstream line;

  if (nested_indent_ && opcode == spv::Op::OpLabel) {
    // Separate the blocks by an empty line to make them easier to separate
    stream_ << std::endl;
  }

  if (inst.result_id) {
    SetBlue();
    const std::string id_name = name_mapper_(inst.result_id);
    if (indent_)
      line << std::setw(std::max(0, indent_ - 3 - int(id_name.size())));
    line << "%" << id_name;
    ResetColor();
    line << " = ";
  } else {
    line << std::string(indent_, ' ');
  }

  if (nested_indent_ && is_in_block) {
    // Output OpLabel at the specified nest level, and instructions inside
    // blocks nested a little more.
    uint32_t indent = block_indent;
    bool body_indent = opcode != spv::Op::OpLabel;

    line << std::string(
        indent * kBlockNestIndent + (body_indent ? kBlockBodyIndentOffset : 0),
        ' ');
  }

  line << "Op" << spvOpcodeString(opcode);

  for (uint16_t i = 0; i < inst.num_operands; i++) {
    const spv_operand_type_t type = inst.operands[i].type;
    assert(type != SPV_OPERAND_TYPE_NONE);
    if (type == SPV_OPERAND_TYPE_RESULT_ID) continue;
    line << " ";
    EmitOperand(line, inst, i);
  }

  // For the sake of comment generation, store information from some
  // instructions for the future.
  if (comment_) {
    GenerateCommentForDecoratedId(inst);
  }

  std::ostringstream comments;
  const char* comment_separator = "";

  if (show_byte_offset_) {
    SetGrey(comments);
    auto saved_flags = comments.flags();
    auto saved_fill = comments.fill();
    comments << comment_separator << "0x" << std::setw(8) << std::hex
             << std::setfill('0') << inst_byte_offset;
    comments.flags(saved_flags);
    comments.fill(saved_fill);
    ResetColor(comments);
    comment_separator = ", ";
  }

  if (comment_ && opcode == spv::Op::OpName) {
    const spv_parsed_operand_t& operand = inst.operands[0];
    const uint32_t word = inst.words[operand.offset];
    comments << comment_separator << "id %" << word;
    comment_separator = ", ";
  }

  if (comment_ && inst.result_id && id_comments_.count(inst.result_id) > 0) {
    comments << comment_separator << id_comments_[inst.result_id].str();
    comment_separator = ", ";
  }

  stream_ << line.str();

  if (!comments.str().empty()) {
    // Align the comments
    const uint32_t line_length = GetLineLengthWithoutColor(line.str());
    uint32_t align = std::max(
        {line_length + 2, last_instruction_comment_alignment_, kCommentColumn});
    // Round up the alignment to a multiple of 4 for more niceness.
    align = (align + 3) & ~0x3u;
    last_instruction_comment_alignment_ = align;

    stream_ << std::string(align - line_length, ' ') << "; " << comments.str();
  } else {
    last_instruction_comment_alignment_ = 0;
  }

  stream_ << "\n";
}

void InstructionDisassembler::GenerateCommentForDecoratedId(
    const spv_parsed_instruction_t& inst) {
  assert(comment_);
  auto opcode = static_cast<spv::Op>(inst.opcode);

  std::ostringstream partial;
  uint32_t id = 0;
  const char* separator = "";

  switch (opcode) {
    case spv::Op::OpDecorate:
      // Take everything after `OpDecorate %id` and associate it with id.
      id = inst.words[inst.operands[0].offset];
      for (uint16_t i = 1; i < inst.num_operands; i++) {
        partial << separator;
        separator = " ";
        EmitOperand(partial, inst, i);
      }
      break;
    default:
      break;
  }

  if (id == 0) {
    return;
  }

  // Add the new comment to the comments of this id
  std::ostringstream& id_comment = id_comments_[id];
  if (!id_comment.str().empty()) {
    id_comment << ", ";
  }
  id_comment << partial.str();
}

void InstructionDisassembler::EmitSectionComment(
    const spv_parsed_instruction_t& inst, bool& inserted_decoration_space,
    bool& inserted_debug_space, bool& inserted_type_space) {
  auto opcode = static_cast<spv::Op>(inst.opcode);
  if (comment_ && opcode == spv::Op::OpFunction) {
    stream_ << std::endl;
    if (nested_indent_) {
      // Double the empty lines between Function sections since nested_indent_
      // also separates blocks by a blank.
      stream_ << std::endl;
    }
    stream_ << std::string(indent_, ' ');
    stream_ << "; Function " << name_mapper_(inst.result_id) << std::endl;
  }
  if (comment_ && !inserted_decoration_space && spvOpcodeIsDecoration(opcode)) {
    inserted_decoration_space = true;
    stream_ << std::endl;
    stream_ << std::string(indent_, ' ');
    stream_ << "; Annotations" << std::endl;
  }
  if (comment_ && !inserted_debug_space && spvOpcodeIsDebug(opcode)) {
    inserted_debug_space = true;
    stream_ << std::endl;
    stream_ << std::string(indent_, ' ');
    stream_ << "; Debug Information" << std::endl;
  }
  if (comment_ && !inserted_type_space && spvOpcodeGeneratesType(opcode)) {
    inserted_type_space = true;
    stream_ << std::endl;
    stream_ << std::string(indent_, ' ');
    stream_ << "; Types, variables and constants" << std::endl;
  }
}

void InstructionDisassembler::EmitOperand(std::ostream& stream,
                                          const spv_parsed_instruction_t& inst,
                                          const uint16_t operand_index) const {
  assert(operand_index < inst.num_operands);
  const spv_parsed_operand_t& operand = inst.operands[operand_index];
  const uint32_t word = inst.words[operand.offset];
  switch (operand.type) {
    case SPV_OPERAND_TYPE_RESULT_ID:
      assert(false && "<result-id> is not supposed to be handled here");
      SetBlue(stream);
      stream << "%" << name_mapper_(word);
      break;
    case SPV_OPERAND_TYPE_ID:
    case SPV_OPERAND_TYPE_TYPE_ID:
    case SPV_OPERAND_TYPE_SCOPE_ID:
    case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID:
      SetYellow(stream);
      stream << "%" << name_mapper_(word);
      break;
    case SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER: {
      spv_ext_inst_desc ext_inst;
      SetRed(stream);
      if (grammar_.lookupExtInst(inst.ext_inst_type, word, &ext_inst) ==
          SPV_SUCCESS) {
        stream << ext_inst->name;
      } else {
        if (!spvExtInstIsNonSemantic(inst.ext_inst_type)) {
          assert(false && "should have caught this earlier");
        } else {
          // for non-semantic instruction sets we can just print the number
          stream << word;
        }
      }
    } break;
    case SPV_OPERAND_TYPE_SPEC_CONSTANT_OP_NUMBER: {
      spv_opcode_desc opcode_desc;
      if (grammar_.lookupOpcode(spv::Op(word), &opcode_desc))
        assert(false && "should have caught this earlier");
      SetRed(stream);
      stream << opcode_desc->name;
    } break;
    case SPV_OPERAND_TYPE_LITERAL_INTEGER:
    case SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER:
    case SPV_OPERAND_TYPE_LITERAL_FLOAT: {
      SetRed(stream);
      EmitNumericLiteral(&stream, inst, operand);
      ResetColor(stream);
    } break;
    case SPV_OPERAND_TYPE_LITERAL_STRING: {
      stream << "\"";
      SetGreen(stream);

      std::string str = spvDecodeLiteralStringOperand(inst, operand_index);
      for (char const& c : str) {
        if (c == '"' || c == '\\') stream << '\\';
        stream << c;
      }
      ResetColor(stream);
      stream << '"';
    } break;
    case SPV_OPERAND_TYPE_CAPABILITY:
    case SPV_OPERAND_TYPE_SOURCE_LANGUAGE:
    case SPV_OPERAND_TYPE_EXECUTION_MODEL:
    case SPV_OPERAND_TYPE_ADDRESSING_MODEL:
    case SPV_OPERAND_TYPE_MEMORY_MODEL:
    case SPV_OPERAND_TYPE_EXECUTION_MODE:
    case SPV_OPERAND_TYPE_STORAGE_CLASS:
    case SPV_OPERAND_TYPE_DIMENSIONALITY:
    case SPV_OPERAND_TYPE_SAMPLER_ADDRESSING_MODE:
    case SPV_OPERAND_TYPE_SAMPLER_FILTER_MODE:
    case SPV_OPERAND_TYPE_SAMPLER_IMAGE_FORMAT:
    case SPV_OPERAND_TYPE_FP_ROUNDING_MODE:
    case SPV_OPERAND_TYPE_LINKAGE_TYPE:
    case SPV_OPERAND_TYPE_ACCESS_QUALIFIER:
    case SPV_OPERAND_TYPE_FUNCTION_PARAMETER_ATTRIBUTE:
    case SPV_OPERAND_TYPE_DECORATION:
    case SPV_OPERAND_TYPE_BUILT_IN:
    case SPV_OPERAND_TYPE_GROUP_OPERATION:
    case SPV_OPERAND_TYPE_KERNEL_ENQ_FLAGS:
    case SPV_OPERAND_TYPE_KERNEL_PROFILING_INFO:
    case SPV_OPERAND_TYPE_RAY_FLAGS:
    case SPV_OPERAND_TYPE_RAY_QUERY_INTERSECTION:
    case SPV_OPERAND_TYPE_RAY_QUERY_COMMITTED_INTERSECTION_TYPE:
    case SPV_OPERAND_TYPE_RAY_QUERY_CANDIDATE_INTERSECTION_TYPE:
    case SPV_OPERAND_TYPE_DEBUG_BASE_TYPE_ATTRIBUTE_ENCODING:
    case SPV_OPERAND_TYPE_DEBUG_COMPOSITE_TYPE:
    case SPV_OPERAND_TYPE_DEBUG_TYPE_QUALIFIER:
    case SPV_OPERAND_TYPE_DEBUG_OPERATION:
    case SPV_OPERAND_TYPE_CLDEBUG100_DEBUG_BASE_TYPE_ATTRIBUTE_ENCODING:
    case SPV_OPERAND_TYPE_CLDEBUG100_DEBUG_COMPOSITE_TYPE:
    case SPV_OPERAND_TYPE_CLDEBUG100_DEBUG_TYPE_QUALIFIER:
    case SPV_OPERAND_TYPE_CLDEBUG100_DEBUG_OPERATION:
    case SPV_OPERAND_TYPE_CLDEBUG100_DEBUG_IMPORTED_ENTITY:
    case SPV_OPERAND_TYPE_FPDENORM_MODE:
    case SPV_OPERAND_TYPE_FPOPERATION_MODE:
    case SPV_OPERAND_TYPE_QUANTIZATION_MODES:
    case SPV_OPERAND_TYPE_FPENCODING:
    case SPV_OPERAND_TYPE_OVERFLOW_MODES: {
      spv_operand_desc entry;
      if (grammar_.lookupOperand(operand.type, word, &entry))
        assert(false && "should have caught this earlier");
      stream << entry->name;
    } break;
    case SPV_OPERAND_TYPE_FP_FAST_MATH_MODE:
    case SPV_OPERAND_TYPE_FUNCTION_CONTROL:
    case SPV_OPERAND_TYPE_LOOP_CONTROL:
    case SPV_OPERAND_TYPE_IMAGE:
    case SPV_OPERAND_TYPE_MEMORY_ACCESS:
    case SPV_OPERAND_TYPE_SELECTION_CONTROL:
    case SPV_OPERAND_TYPE_DEBUG_INFO_FLAGS:
    case SPV_OPERAND_TYPE_CLDEBUG100_DEBUG_INFO_FLAGS:
    case SPV_OPERAND_TYPE_RAW_ACCESS_CHAIN_OPERANDS:
      EmitMaskOperand(stream, operand.type, word);
      break;
    default:
      if (spvOperandIsConcreteMask(operand.type)) {
        EmitMaskOperand(stream, operand.type, word);
      } else if (spvOperandIsConcrete(operand.type)) {
        spv_operand_desc entry;
        if (grammar_.lookupOperand(operand.type, word, &entry))
          assert(false && "should have caught this earlier");
        stream << entry->name;
      } else {
        assert(false && "unhandled or invalid case");
      }
      break;
  }
  ResetColor(stream);
}

void InstructionDisassembler::EmitMaskOperand(std::ostream& stream,
                                              const spv_operand_type_t type,
                                              const uint32_t word) const {
  // Scan the mask from least significant bit to most significant bit.  For each
  // set bit, emit the name of that bit. Separate multiple names with '|'.
  uint32_t remaining_word = word;
  uint32_t mask;
  int num_emitted = 0;
  for (mask = 1; remaining_word; mask <<= 1) {
    if (remaining_word & mask) {
      remaining_word ^= mask;
      spv_operand_desc entry;
      if (grammar_.lookupOperand(type, mask, &entry))
        assert(false && "should have caught this earlier");
      if (num_emitted) stream << "|";
      stream << entry->name;
      num_emitted++;
    }
  }
  if (!num_emitted) {
    // An operand value of 0 was provided, so represent it by the name
    // of the 0 value. In many cases, that's "None".
    spv_operand_desc entry;
    if (SPV_SUCCESS == grammar_.lookupOperand(type, 0, &entry))
      stream << entry->name;
  }
}

void InstructionDisassembler::ResetColor(std::ostream& stream) const {
  if (color_) stream << spvtools::clr::reset{print_};
}
void InstructionDisassembler::SetGrey(std::ostream& stream) const {
  if (color_) stream << spvtools::clr::grey{print_};
}
void InstructionDisassembler::SetBlue(std::ostream& stream) const {
  if (color_) stream << spvtools::clr::blue{print_};
}
void InstructionDisassembler::SetYellow(std::ostream& stream) const {
  if (color_) stream << spvtools::clr::yellow{print_};
}
void InstructionDisassembler::SetRed(std::ostream& stream) const {
  if (color_) stream << spvtools::clr::red{print_};
}
void InstructionDisassembler::SetGreen(std::ostream& stream) const {
  if (color_) stream << spvtools::clr::green{print_};
}

void InstructionDisassembler::ResetColor() { ResetColor(stream_); }
void InstructionDisassembler::SetGrey() { SetGrey(stream_); }
void InstructionDisassembler::SetBlue() { SetBlue(stream_); }
void InstructionDisassembler::SetYellow() { SetYellow(stream_); }
void InstructionDisassembler::SetRed() { SetRed(stream_); }
void InstructionDisassembler::SetGreen() { SetGreen(stream_); }
}  // namespace disassemble

std::string spvInstructionBinaryToText(const spv_target_env env,
                                       const uint32_t* instCode,
                                       const size_t instWordCount,
                                       const uint32_t* code,
                                       const size_t wordCount,
                                       const uint32_t options) {
  spv_context context = spvContextCreate(env);
  const AssemblyGrammar grammar(context);
  if (!grammar.isValid()) {
    spvContextDestroy(context);
    return "";
  }

  // Generate friendly names for Ids if requested.
  std::unique_ptr<FriendlyNameMapper> friendly_mapper;
  NameMapper name_mapper = GetTrivialNameMapper();
  if (options & SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) {
    friendly_mapper = MakeUnique<FriendlyNameMapper>(context, code, wordCount);
    name_mapper = friendly_mapper->GetNameMapper();
  }

  // Now disassemble!
  Disassembler disassembler(grammar, options, name_mapper);
  WrappedDisassembler wrapped(&disassembler, instCode, instWordCount);
  spvBinaryParse(context, &wrapped, code, wordCount, DisassembleTargetHeader,
                 DisassembleTargetInstruction, nullptr);

  spv_text text = nullptr;
  std::string output;
  if (disassembler.SaveTextResult(&text) == SPV_SUCCESS) {
    output.assign(text->str, text->str + text->length);
    // Drop trailing newline characters.
    while (!output.empty() && output.back() == '\n') output.pop_back();
  }
  spvTextDestroy(text);
  spvContextDestroy(context);

  return output;
}
}  // namespace spvtools

spv_result_t spvBinaryToText(const spv_const_context context,
                             const uint32_t* code, const size_t wordCount,
                             const uint32_t options, spv_text* pText,
                             spv_diagnostic* pDiagnostic) {
  spv_context_t hijack_context = *context;
  if (pDiagnostic) {
    *pDiagnostic = nullptr;
    spvtools::UseDiagnosticAsMessageConsumer(&hijack_context, pDiagnostic);
  }

  const spvtools::AssemblyGrammar grammar(&hijack_context);
  if (!grammar.isValid()) return SPV_ERROR_INVALID_TABLE;

  // Generate friendly names for Ids if requested.
  std::unique_ptr<spvtools::FriendlyNameMapper> friendly_mapper;
  spvtools::NameMapper name_mapper = spvtools::GetTrivialNameMapper();
  if (options & SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) {
    friendly_mapper = spvtools::MakeUnique<spvtools::FriendlyNameMapper>(
        &hijack_context, code, wordCount);
    name_mapper = friendly_mapper->GetNameMapper();
  }

  // Now disassemble!
  spvtools::Disassembler disassembler(grammar, options, name_mapper);
  if (auto error =
          spvBinaryParse(&hijack_context, &disassembler, code, wordCount,
                         spvtools::DisassembleHeader,
                         spvtools::DisassembleInstruction, pDiagnostic)) {
    return error;
  }

  return disassembler.SaveTextResult(pText);
}
