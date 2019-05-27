// Copyright (c) 2018 Google LLC.
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

#include "source/opt/scalar_analysis.h"

#include <algorithm>
#include <functional>
#include <string>
#include <utility>

#include "source/opt/ir_context.h"

// Transforms a given scalar operation instruction into a DAG representation.
//
// 1. Take an instruction and traverse its operands until we reach a
// constant node or an instruction which we do not know how to compute the
// value, such as a load.
//
// 2. Create a new node for each instruction traversed and build the nodes for
// the in operands of that instruction as well.
//
// 3. Add the operand nodes as children of the first and hash the node. Use the
// hash to see if the node is already in the cache. We ensure the children are
// always in sorted order so that two nodes with the same children but inserted
// in a different order have the same hash and so that the overloaded operator==
// will return true. If the node is already in the cache return the cached
// version instead.
//
// 4. The created DAG can then be simplified by
// ScalarAnalysis::SimplifyExpression, implemented in
// scalar_analysis_simplification.cpp. See that file for further information on
// the simplification process.
//

namespace spvtools {
namespace opt {

uint32_t SENode::NumberOfNodes = 0;

ScalarEvolutionAnalysis::ScalarEvolutionAnalysis(IRContext* context)
    : context_(context), pretend_equal_{} {
  // Create and cached the CantComputeNode.
  cached_cant_compute_ =
      GetCachedOrAdd(std::unique_ptr<SECantCompute>(new SECantCompute(this)));
}

SENode* ScalarEvolutionAnalysis::CreateNegation(SENode* operand) {
  // If operand is can't compute then the whole graph is can't compute.
  if (operand->IsCantCompute()) return CreateCantComputeNode();

  if (operand->GetType() == SENode::Constant) {
    return CreateConstant(-operand->AsSEConstantNode()->FoldToSingleValue());
  }
  std::unique_ptr<SENode> negation_node{new SENegative(this)};
  negation_node->AddChild(operand);
  return GetCachedOrAdd(std::move(negation_node));
}

SENode* ScalarEvolutionAnalysis::CreateConstant(int64_t integer) {
  return GetCachedOrAdd(
      std::unique_ptr<SENode>(new SEConstantNode(this, integer)));
}

SENode* ScalarEvolutionAnalysis::CreateRecurrentExpression(
    const Loop* loop, SENode* offset, SENode* coefficient) {
  assert(loop && "Recurrent add expressions must have a valid loop.");

  // If operands are can't compute then the whole graph is can't compute.
  if (offset->IsCantCompute() || coefficient->IsCantCompute())
    return CreateCantComputeNode();

  const Loop* loop_to_use = nullptr;
  if (pretend_equal_[loop]) {
    loop_to_use = pretend_equal_[loop];
  } else {
    loop_to_use = loop;
  }

  std::unique_ptr<SERecurrentNode> phi_node{
      new SERecurrentNode(this, loop_to_use)};
  phi_node->AddOffset(offset);
  phi_node->AddCoefficient(coefficient);

  return GetCachedOrAdd(std::move(phi_node));
}

SENode* ScalarEvolutionAnalysis::AnalyzeMultiplyOp(
    const Instruction* multiply) {
  assert(multiply->opcode() == SpvOp::SpvOpIMul &&
         "Multiply node did not come from a multiply instruction");
  analysis::DefUseManager* def_use = context_->get_def_use_mgr();

  SENode* op1 =
      AnalyzeInstruction(def_use->GetDef(multiply->GetSingleWordInOperand(0)));
  SENode* op2 =
      AnalyzeInstruction(def_use->GetDef(multiply->GetSingleWordInOperand(1)));

  return CreateMultiplyNode(op1, op2);
}

SENode* ScalarEvolutionAnalysis::CreateMultiplyNode(SENode* operand_1,
                                                    SENode* operand_2) {
  // If operands are can't compute then the whole graph is can't compute.
  if (operand_1->IsCantCompute() || operand_2->IsCantCompute())
    return CreateCantComputeNode();

  if (operand_1->GetType() == SENode::Constant &&
      operand_2->GetType() == SENode::Constant) {
    return CreateConstant(operand_1->AsSEConstantNode()->FoldToSingleValue() *
                          operand_2->AsSEConstantNode()->FoldToSingleValue());
  }

  std::unique_ptr<SENode> multiply_node{new SEMultiplyNode(this)};

  multiply_node->AddChild(operand_1);
  multiply_node->AddChild(operand_2);

  return GetCachedOrAdd(std::move(multiply_node));
}

SENode* ScalarEvolutionAnalysis::CreateSubtraction(SENode* operand_1,
                                                   SENode* operand_2) {
  // Fold if both operands are constant.
  if (operand_1->GetType() == SENode::Constant &&
      operand_2->GetType() == SENode::Constant) {
    return CreateConstant(operand_1->AsSEConstantNode()->FoldToSingleValue() -
                          operand_2->AsSEConstantNode()->FoldToSingleValue());
  }

  return CreateAddNode(operand_1, CreateNegation(operand_2));
}

SENode* ScalarEvolutionAnalysis::CreateAddNode(SENode* operand_1,
                                               SENode* operand_2) {
  // Fold if both operands are constant and the |simplify| flag is true.
  if (operand_1->GetType() == SENode::Constant &&
      operand_2->GetType() == SENode::Constant) {
    return CreateConstant(operand_1->AsSEConstantNode()->FoldToSingleValue() +
                          operand_2->AsSEConstantNode()->FoldToSingleValue());
  }

  // If operands are can't compute then the whole graph is can't compute.
  if (operand_1->IsCantCompute() || operand_2->IsCantCompute())
    return CreateCantComputeNode();

  std::unique_ptr<SENode> add_node{new SEAddNode(this)};

  add_node->AddChild(operand_1);
  add_node->AddChild(operand_2);

  return GetCachedOrAdd(std::move(add_node));
}

SENode* ScalarEvolutionAnalysis::AnalyzeInstruction(const Instruction* inst) {
  auto itr = recurrent_node_map_.find(inst);
  if (itr != recurrent_node_map_.end()) return itr->second;

  SENode* output = nullptr;
  switch (inst->opcode()) {
    case SpvOp::SpvOpPhi: {
      output = AnalyzePhiInstruction(inst);
      break;
    }
    case SpvOp::SpvOpConstant:
    case SpvOp::SpvOpConstantNull: {
      output = AnalyzeConstant(inst);
      break;
    }
    case SpvOp::SpvOpISub:
    case SpvOp::SpvOpIAdd: {
      output = AnalyzeAddOp(inst);
      break;
    }
    case SpvOp::SpvOpIMul: {
      output = AnalyzeMultiplyOp(inst);
      break;
    }
    default: {
      output = CreateValueUnknownNode(inst);
      break;
    }
  }

  return output;
}

SENode* ScalarEvolutionAnalysis::AnalyzeConstant(const Instruction* inst) {
  if (inst->opcode() == SpvOp::SpvOpConstantNull) return CreateConstant(0);

  assert(inst->opcode() == SpvOp::SpvOpConstant);
  assert(inst->NumInOperands() == 1);
  int64_t value = 0;

  // Look up the instruction in the constant manager.
  const analysis::Constant* constant =
      context_->get_constant_mgr()->FindDeclaredConstant(inst->result_id());

  if (!constant) return CreateCantComputeNode();

  const analysis::IntConstant* int_constant = constant->AsIntConstant();

  // Exit out if it is a 64 bit integer.
  if (!int_constant || int_constant->words().size() != 1)
    return CreateCantComputeNode();

  if (int_constant->type()->AsInteger()->IsSigned()) {
    value = int_constant->GetS32BitValue();
  } else {
    value = int_constant->GetU32BitValue();
  }

  return CreateConstant(value);
}

// Handles both addition and subtraction. If the |sub| flag is set then the
// addition will be op1+(-op2) otherwise op1+op2.
SENode* ScalarEvolutionAnalysis::AnalyzeAddOp(const Instruction* inst) {
  assert((inst->opcode() == SpvOp::SpvOpIAdd ||
          inst->opcode() == SpvOp::SpvOpISub) &&
         "Add node must be created from a OpIAdd or OpISub instruction");

  analysis::DefUseManager* def_use = context_->get_def_use_mgr();

  SENode* op1 =
      AnalyzeInstruction(def_use->GetDef(inst->GetSingleWordInOperand(0)));

  SENode* op2 =
      AnalyzeInstruction(def_use->GetDef(inst->GetSingleWordInOperand(1)));

  // To handle subtraction we wrap the second operand in a unary negation node.
  if (inst->opcode() == SpvOp::SpvOpISub) {
    op2 = CreateNegation(op2);
  }

  return CreateAddNode(op1, op2);
}

SENode* ScalarEvolutionAnalysis::AnalyzePhiInstruction(const Instruction* phi) {
  // The phi should only have two incoming value pairs.
  if (phi->NumInOperands() != 4) {
    return CreateCantComputeNode();
  }

  analysis::DefUseManager* def_use = context_->get_def_use_mgr();

  // Get the basic block this instruction belongs to.
  BasicBlock* basic_block =
      context_->get_instr_block(const_cast<Instruction*>(phi));

  // And then the function that the basic blocks belongs to.
  Function* function = basic_block->GetParent();

  // Use the function to get the loop descriptor.
  LoopDescriptor* loop_descriptor = context_->GetLoopDescriptor(function);

  // We only handle phis in loops at the moment.
  if (!loop_descriptor) return CreateCantComputeNode();

  // Get the innermost loop which this block belongs to.
  Loop* loop = (*loop_descriptor)[basic_block->id()];

  // If the loop doesn't exist or doesn't have a preheader or latch block, exit
  // out.
  if (!loop || !loop->GetLatchBlock() || !loop->GetPreHeaderBlock() ||
      loop->GetHeaderBlock() != basic_block)
    return recurrent_node_map_[phi] = CreateCantComputeNode();

  const Loop* loop_to_use = nullptr;
  if (pretend_equal_[loop]) {
    loop_to_use = pretend_equal_[loop];
  } else {
    loop_to_use = loop;
  }
  std::unique_ptr<SERecurrentNode> phi_node{
      new SERecurrentNode(this, loop_to_use)};

  // We add the node to this map to allow it to be returned before the node is
  // fully built. This is needed as the subsequent call to AnalyzeInstruction
  // could lead back to this |phi| instruction so we return the pointer
  // immediately in AnalyzeInstruction to break the recursion.
  recurrent_node_map_[phi] = phi_node.get();

  // Traverse the operands of the instruction an create new nodes for each one.
  for (uint32_t i = 0; i < phi->NumInOperands(); i += 2) {
    uint32_t value_id = phi->GetSingleWordInOperand(i);
    uint32_t incoming_label_id = phi->GetSingleWordInOperand(i + 1);

    Instruction* value_inst = def_use->GetDef(value_id);
    SENode* value_node = AnalyzeInstruction(value_inst);

    // If any operand is CantCompute then the whole graph is CantCompute.
    if (value_node->IsCantCompute())
      return recurrent_node_map_[phi] = CreateCantComputeNode();

    // If the value is coming from the preheader block then the value is the
    // initial value of the phi.
    if (incoming_label_id == loop->GetPreHeaderBlock()->id()) {
      phi_node->AddOffset(value_node);
    } else if (incoming_label_id == loop->GetLatchBlock()->id()) {
      // Assumed to be in the form of step + phi.
      if (value_node->GetType() != SENode::Add)
        return recurrent_node_map_[phi] = CreateCantComputeNode();

      SENode* step_node = nullptr;
      SENode* phi_operand = nullptr;
      SENode* operand_1 = value_node->GetChild(0);
      SENode* operand_2 = value_node->GetChild(1);

      // Find which node is the step term.
      if (!operand_1->AsSERecurrentNode())
        step_node = operand_1;
      else if (!operand_2->AsSERecurrentNode())
        step_node = operand_2;

      // Find which node is the recurrent expression.
      if (operand_1->AsSERecurrentNode())
        phi_operand = operand_1;
      else if (operand_2->AsSERecurrentNode())
        phi_operand = operand_2;

      // If it is not in the form step + phi exit out.
      if (!(step_node && phi_operand))
        return recurrent_node_map_[phi] = CreateCantComputeNode();

      // If the phi operand is not the same phi node exit out.
      if (phi_operand != phi_node.get())
        return recurrent_node_map_[phi] = CreateCantComputeNode();

      if (!IsLoopInvariant(loop, step_node))
        return recurrent_node_map_[phi] = CreateCantComputeNode();

      phi_node->AddCoefficient(step_node);
    }
  }

  // Once the node is fully built we update the map with the version from the
  // cache (if it has already been added to the cache).
  return recurrent_node_map_[phi] = GetCachedOrAdd(std::move(phi_node));
}

SENode* ScalarEvolutionAnalysis::CreateValueUnknownNode(
    const Instruction* inst) {
  std::unique_ptr<SEValueUnknown> load_node{
      new SEValueUnknown(this, inst->result_id())};
  return GetCachedOrAdd(std::move(load_node));
}

SENode* ScalarEvolutionAnalysis::CreateCantComputeNode() {
  return cached_cant_compute_;
}

// Add the created node into the cache of nodes. If it already exists return it.
SENode* ScalarEvolutionAnalysis::GetCachedOrAdd(
    std::unique_ptr<SENode> prospective_node) {
  auto itr = node_cache_.find(prospective_node);
  if (itr != node_cache_.end()) {
    return (*itr).get();
  }

  SENode* raw_ptr_to_node = prospective_node.get();
  node_cache_.insert(std::move(prospective_node));
  return raw_ptr_to_node;
}

bool ScalarEvolutionAnalysis::IsLoopInvariant(const Loop* loop,
                                              const SENode* node) const {
  for (auto itr = node->graph_cbegin(); itr != node->graph_cend(); ++itr) {
    if (const SERecurrentNode* rec = itr->AsSERecurrentNode()) {
      const BasicBlock* header = rec->GetLoop()->GetHeaderBlock();

      // If the loop which the recurrent expression belongs to is either |loop
      // or a nested loop inside |loop| then we assume it is variant.
      if (loop->IsInsideLoop(header)) {
        return false;
      }
    } else if (const SEValueUnknown* unknown = itr->AsSEValueUnknown()) {
      // If the instruction is inside the loop we conservatively assume it is
      // loop variant.
      if (loop->IsInsideLoop(unknown->ResultId())) return false;
    }
  }

  return true;
}

SENode* ScalarEvolutionAnalysis::GetCoefficientFromRecurrentTerm(
    SENode* node, const Loop* loop) {
  // Traverse the DAG to find the recurrent expression belonging to |loop|.
  for (auto itr = node->graph_begin(); itr != node->graph_end(); ++itr) {
    SERecurrentNode* rec = itr->AsSERecurrentNode();
    if (rec && rec->GetLoop() == loop) {
      return rec->GetCoefficient();
    }
  }
  return CreateConstant(0);
}

SENode* ScalarEvolutionAnalysis::UpdateChildNode(SENode* parent,
                                                 SENode* old_child,
                                                 SENode* new_child) {
  // Only handles add.
  if (parent->GetType() != SENode::Add) return parent;

  std::vector<SENode*> new_children;
  for (SENode* child : *parent) {
    if (child == old_child) {
      new_children.push_back(new_child);
    } else {
      new_children.push_back(child);
    }
  }

  std::unique_ptr<SENode> add_node{new SEAddNode(this)};
  for (SENode* child : new_children) {
    add_node->AddChild(child);
  }

  return SimplifyExpression(GetCachedOrAdd(std::move(add_node)));
}

// Rebuild the |node| eliminating, if it exists, the recurrent term which
// belongs to the |loop|.
SENode* ScalarEvolutionAnalysis::BuildGraphWithoutRecurrentTerm(
    SENode* node, const Loop* loop) {
  // If the node is already a recurrent expression belonging to loop then just
  // return the offset.
  SERecurrentNode* recurrent = node->AsSERecurrentNode();
  if (recurrent) {
    if (recurrent->GetLoop() == loop) {
      return recurrent->GetOffset();
    } else {
      return node;
    }
  }

  std::vector<SENode*> new_children;
  // Otherwise find the recurrent node in the children of this node.
  for (auto itr : *node) {
    recurrent = itr->AsSERecurrentNode();
    if (recurrent && recurrent->GetLoop() == loop) {
      new_children.push_back(recurrent->GetOffset());
    } else {
      new_children.push_back(itr);
    }
  }

  std::unique_ptr<SENode> add_node{new SEAddNode(this)};
  for (SENode* child : new_children) {
    add_node->AddChild(child);
  }

  return SimplifyExpression(GetCachedOrAdd(std::move(add_node)));
}

// Return the recurrent term belonging to |loop| if it appears in the graph
// starting at |node| or null if it doesn't.
SERecurrentNode* ScalarEvolutionAnalysis::GetRecurrentTerm(SENode* node,
                                                           const Loop* loop) {
  for (auto itr = node->graph_begin(); itr != node->graph_end(); ++itr) {
    SERecurrentNode* rec = itr->AsSERecurrentNode();
    if (rec && rec->GetLoop() == loop) {
      return rec;
    }
  }
  return nullptr;
}
std::string SENode::AsString() const {
  switch (GetType()) {
    case Constant:
      return "Constant";
    case RecurrentAddExpr:
      return "RecurrentAddExpr";
    case Add:
      return "Add";
    case Negative:
      return "Negative";
    case Multiply:
      return "Multiply";
    case ValueUnknown:
      return "Value Unknown";
    case CanNotCompute:
      return "Can not compute";
  }
  return "NULL";
}

bool SENode::operator==(const SENode& other) const {
  if (GetType() != other.GetType()) return false;

  if (other.GetChildren().size() != children_.size()) return false;

  const SERecurrentNode* this_as_recurrent = AsSERecurrentNode();

  // Check the children are the same, for SERecurrentNodes we need to check the
  // offset and coefficient manually as the child vector is sorted by ids so the
  // offset/coefficient information is lost.
  if (!this_as_recurrent) {
    for (size_t index = 0; index < children_.size(); ++index) {
      if (other.GetChildren()[index] != children_[index]) return false;
    }
  } else {
    const SERecurrentNode* other_as_recurrent = other.AsSERecurrentNode();

    // We've already checked the types are the same, this should not fail if
    // this->AsSERecurrentNode() succeeded.
    assert(other_as_recurrent);

    if (this_as_recurrent->GetCoefficient() !=
        other_as_recurrent->GetCoefficient())
      return false;

    if (this_as_recurrent->GetOffset() != other_as_recurrent->GetOffset())
      return false;

    if (this_as_recurrent->GetLoop() != other_as_recurrent->GetLoop())
      return false;
  }

  // If we're dealing with a value unknown node check both nodes were created by
  // the same instruction.
  if (GetType() == SENode::ValueUnknown) {
    if (AsSEValueUnknown()->ResultId() !=
        other.AsSEValueUnknown()->ResultId()) {
      return false;
    }
  }

  if (AsSEConstantNode()) {
    if (AsSEConstantNode()->FoldToSingleValue() !=
        other.AsSEConstantNode()->FoldToSingleValue())
      return false;
  }

  return true;
}

bool SENode::operator!=(const SENode& other) const { return !(*this == other); }

namespace {
// Helper functions to insert 32/64 bit values into the 32 bit hash string. This
// allows us to add pointers to the string by reinterpreting the pointers as
// uintptr_t. PushToString will deduce the type, call sizeof on it and use
// that size to call into the correct PushToStringImpl functor depending on
// whether it is 32 or 64 bit.

template <typename T, size_t size_of_t>
struct PushToStringImpl;

template <typename T>
struct PushToStringImpl<T, 8> {
  void operator()(T id, std::u32string* str) {
    str->push_back(static_cast<uint32_t>(id >> 32));
    str->push_back(static_cast<uint32_t>(id));
  }
};

template <typename T>
struct PushToStringImpl<T, 4> {
  void operator()(T id, std::u32string* str) {
    str->push_back(static_cast<uint32_t>(id));
  }
};

template <typename T>
static void PushToString(T id, std::u32string* str) {
  PushToStringImpl<T, sizeof(T)>{}(id, str);
}

}  // namespace

// Implements the hashing of SENodes.
size_t SENodeHash::operator()(const SENode* node) const {
  // Concatinate the terms into a string which we can hash.
  std::u32string hash_string{};

  // Hashing the type as a string is safer than hashing the enum as the enum is
  // very likely to collide with constants.
  for (char ch : node->AsString()) {
    hash_string.push_back(static_cast<char32_t>(ch));
  }

  // We just ignore the literal value unless it is a constant.
  if (node->GetType() == SENode::Constant)
    PushToString(node->AsSEConstantNode()->FoldToSingleValue(), &hash_string);

  const SERecurrentNode* recurrent = node->AsSERecurrentNode();

  // If we're dealing with a recurrent expression hash the loop as well so that
  // nested inductions like i=0,i++ and j=0,j++ correspond to different nodes.
  if (recurrent) {
    PushToString(reinterpret_cast<uintptr_t>(recurrent->GetLoop()),
                 &hash_string);

    // Recurrent expressions can't be hashed using the normal method as the
    // order of coefficient and offset matters to the hash.
    PushToString(reinterpret_cast<uintptr_t>(recurrent->GetCoefficient()),
                 &hash_string);
    PushToString(reinterpret_cast<uintptr_t>(recurrent->GetOffset()),
                 &hash_string);

    return std::hash<std::u32string>{}(hash_string);
  }

  // Hash the result id of the original instruction which created this node if
  // it is a value unknown node.
  if (node->GetType() == SENode::ValueUnknown) {
    PushToString(node->AsSEValueUnknown()->ResultId(), &hash_string);
  }

  // Hash the pointers of the child nodes, each SENode has a unique pointer
  // associated with it.
  const std::vector<SENode*>& children = node->GetChildren();
  for (const SENode* child : children) {
    PushToString(reinterpret_cast<uintptr_t>(child), &hash_string);
  }

  return std::hash<std::u32string>{}(hash_string);
}

// This overload is the actual overload used by the node_cache_ set.
size_t SENodeHash::operator()(const std::unique_ptr<SENode>& node) const {
  return this->operator()(node.get());
}

void SENode::DumpDot(std::ostream& out, bool recurse) const {
  size_t unique_id = std::hash<const SENode*>{}(this);
  out << unique_id << " [label=\"" << AsString() << " ";
  if (GetType() == SENode::Constant) {
    out << "\nwith value: " << this->AsSEConstantNode()->FoldToSingleValue();
  }
  out << "\"]\n";
  for (const SENode* child : children_) {
    size_t child_unique_id = std::hash<const SENode*>{}(child);
    out << unique_id << " -> " << child_unique_id << " \n";
    if (recurse) child->DumpDot(out, true);
  }
}

namespace {
class IsGreaterThanZero {
 public:
  explicit IsGreaterThanZero(IRContext* context) : context_(context) {}

  // Determine if the value of |node| is always strictly greater than zero if
  // |or_equal_zero| is false or greater or equal to zero if |or_equal_zero| is
  // true. It returns true is the evaluation was able to conclude something, in
  // which case the result is stored in |result|.
  // The algorithm work by going through all the nodes and determine the
  // sign of each of them.
  bool Eval(const SENode* node, bool or_equal_zero, bool* result) {
    *result = false;
    switch (Visit(node)) {
      case Signedness::kPositiveOrNegative: {
        return false;
      }
      case Signedness::kStrictlyNegative: {
        *result = false;
        break;
      }
      case Signedness::kNegative: {
        if (!or_equal_zero) {
          return false;
        }
        *result = false;
        break;
      }
      case Signedness::kStrictlyPositive: {
        *result = true;
        break;
      }
      case Signedness::kPositive: {
        if (!or_equal_zero) {
          return false;
        }
        *result = true;
        break;
      }
    }
    return true;
  }

 private:
  enum class Signedness {
    kPositiveOrNegative,  // Yield a value positive or negative.
    kStrictlyNegative,    // Yield a value strictly less than 0.
    kNegative,            // Yield a value less or equal to 0.
    kStrictlyPositive,    // Yield a value strictly greater than 0.
    kPositive             // Yield a value greater or equal to 0.
  };

  // Combine the signedness according to arithmetic rules of a given operator.
  using Combiner = std::function<Signedness(Signedness, Signedness)>;

  // Returns a functor to interpret the signedness of 2 expressions as if they
  // were added.
  Combiner GetAddCombiner() const {
    return [](Signedness lhs, Signedness rhs) {
      switch (lhs) {
        case Signedness::kPositiveOrNegative:
          break;
        case Signedness::kStrictlyNegative:
          if (rhs == Signedness::kStrictlyNegative ||
              rhs == Signedness::kNegative)
            return lhs;
          break;
        case Signedness::kNegative: {
          if (rhs == Signedness::kStrictlyNegative)
            return Signedness::kStrictlyNegative;
          if (rhs == Signedness::kNegative) return Signedness::kNegative;
          break;
        }
        case Signedness::kStrictlyPositive: {
          if (rhs == Signedness::kStrictlyPositive ||
              rhs == Signedness::kPositive) {
            return Signedness::kStrictlyPositive;
          }
          break;
        }
        case Signedness::kPositive: {
          if (rhs == Signedness::kStrictlyPositive)
            return Signedness::kStrictlyPositive;
          if (rhs == Signedness::kPositive) return Signedness::kPositive;
          break;
        }
      }
      return Signedness::kPositiveOrNegative;
    };
  }

  // Returns a functor to interpret the signedness of 2 expressions as if they
  // were multiplied.
  Combiner GetMulCombiner() const {
    return [](Signedness lhs, Signedness rhs) {
      switch (lhs) {
        case Signedness::kPositiveOrNegative:
          break;
        case Signedness::kStrictlyNegative: {
          switch (rhs) {
            case Signedness::kPositiveOrNegative: {
              break;
            }
            case Signedness::kStrictlyNegative: {
              return Signedness::kStrictlyPositive;
            }
            case Signedness::kNegative: {
              return Signedness::kPositive;
            }
            case Signedness::kStrictlyPositive: {
              return Signedness::kStrictlyNegative;
            }
            case Signedness::kPositive: {
              return Signedness::kNegative;
            }
          }
          break;
        }
        case Signedness::kNegative: {
          switch (rhs) {
            case Signedness::kPositiveOrNegative: {
              break;
            }
            case Signedness::kStrictlyNegative:
            case Signedness::kNegative: {
              return Signedness::kPositive;
            }
            case Signedness::kStrictlyPositive:
            case Signedness::kPositive: {
              return Signedness::kNegative;
            }
          }
          break;
        }
        case Signedness::kStrictlyPositive: {
          return rhs;
        }
        case Signedness::kPositive: {
          switch (rhs) {
            case Signedness::kPositiveOrNegative: {
              break;
            }
            case Signedness::kStrictlyNegative:
            case Signedness::kNegative: {
              return Signedness::kNegative;
            }
            case Signedness::kStrictlyPositive:
            case Signedness::kPositive: {
              return Signedness::kPositive;
            }
          }
          break;
        }
      }
      return Signedness::kPositiveOrNegative;
    };
  }

  Signedness Visit(const SENode* node) {
    switch (node->GetType()) {
      case SENode::Constant:
        return Visit(node->AsSEConstantNode());
        break;
      case SENode::RecurrentAddExpr:
        return Visit(node->AsSERecurrentNode());
        break;
      case SENode::Negative:
        return Visit(node->AsSENegative());
        break;
      case SENode::CanNotCompute:
        return Visit(node->AsSECantCompute());
        break;
      case SENode::ValueUnknown:
        return Visit(node->AsSEValueUnknown());
        break;
      case SENode::Add:
        return VisitExpr(node, GetAddCombiner());
        break;
      case SENode::Multiply:
        return VisitExpr(node, GetMulCombiner());
        break;
    }
    return Signedness::kPositiveOrNegative;
  }

  // Returns the signedness of a constant |node|.
  Signedness Visit(const SEConstantNode* node) {
    if (0 == node->FoldToSingleValue()) return Signedness::kPositive;
    if (0 < node->FoldToSingleValue()) return Signedness::kStrictlyPositive;
    if (0 > node->FoldToSingleValue()) return Signedness::kStrictlyNegative;
    return Signedness::kPositiveOrNegative;
  }

  // Returns the signedness of an unknown |node| based on its type.
  Signedness Visit(const SEValueUnknown* node) {
    Instruction* insn = context_->get_def_use_mgr()->GetDef(node->ResultId());
    analysis::Type* type = context_->get_type_mgr()->GetType(insn->type_id());
    assert(type && "Can't retrieve a type for the instruction");
    analysis::Integer* int_type = type->AsInteger();
    assert(type && "Can't retrieve an integer type for the instruction");
    return int_type->IsSigned() ? Signedness::kPositiveOrNegative
                                : Signedness::kPositive;
  }

  // Returns the signedness of a recurring expression.
  Signedness Visit(const SERecurrentNode* node) {
    Signedness coeff_sign = Visit(node->GetCoefficient());
    // SERecurrentNode represent an affine expression in the range [0,
    // loop_bound], so the result cannot be strictly positive or negative.
    switch (coeff_sign) {
      default:
        break;
      case Signedness::kStrictlyNegative:
        coeff_sign = Signedness::kNegative;
        break;
      case Signedness::kStrictlyPositive:
        coeff_sign = Signedness::kPositive;
        break;
    }
    return GetAddCombiner()(coeff_sign, Visit(node->GetOffset()));
  }

  // Returns the signedness of a negation |node|.
  Signedness Visit(const SENegative* node) {
    switch (Visit(*node->begin())) {
      case Signedness::kPositiveOrNegative: {
        return Signedness::kPositiveOrNegative;
      }
      case Signedness::kStrictlyNegative: {
        return Signedness::kStrictlyPositive;
      }
      case Signedness::kNegative: {
        return Signedness::kPositive;
      }
      case Signedness::kStrictlyPositive: {
        return Signedness::kStrictlyNegative;
      }
      case Signedness::kPositive: {
        return Signedness::kNegative;
      }
    }
    return Signedness::kPositiveOrNegative;
  }

  Signedness Visit(const SECantCompute*) {
    return Signedness::kPositiveOrNegative;
  }

  // Returns the signedness of a binary expression by using the combiner
  // |reduce|.
  Signedness VisitExpr(
      const SENode* node,
      std::function<Signedness(Signedness, Signedness)> reduce) {
    Signedness result = Visit(*node->begin());
    for (const SENode* operand : make_range(++node->begin(), node->end())) {
      if (result == Signedness::kPositiveOrNegative) {
        return Signedness::kPositiveOrNegative;
      }
      result = reduce(result, Visit(operand));
    }
    return result;
  }

  IRContext* context_;
};
}  // namespace

bool ScalarEvolutionAnalysis::IsAlwaysGreaterThanZero(SENode* node,
                                                      bool* is_gt_zero) const {
  return IsGreaterThanZero(context_).Eval(node, false, is_gt_zero);
}

bool ScalarEvolutionAnalysis::IsAlwaysGreaterOrEqualToZero(
    SENode* node, bool* is_ge_zero) const {
  return IsGreaterThanZero(context_).Eval(node, true, is_ge_zero);
}

namespace {

// Remove |node| from the |mul| chain (of the form A * ... * |node| * ... * Z),
// if |node| is not in the chain, returns the original chain.
static SENode* RemoveOneNodeFromMultiplyChain(SEMultiplyNode* mul,
                                              const SENode* node) {
  SENode* lhs = mul->GetChildren()[0];
  SENode* rhs = mul->GetChildren()[1];
  if (lhs == node) {
    return rhs;
  }
  if (rhs == node) {
    return lhs;
  }
  if (lhs->AsSEMultiplyNode()) {
    SENode* res = RemoveOneNodeFromMultiplyChain(lhs->AsSEMultiplyNode(), node);
    if (res != lhs)
      return mul->GetParentAnalysis()->CreateMultiplyNode(res, rhs);
  }
  if (rhs->AsSEMultiplyNode()) {
    SENode* res = RemoveOneNodeFromMultiplyChain(rhs->AsSEMultiplyNode(), node);
    if (res != rhs)
      return mul->GetParentAnalysis()->CreateMultiplyNode(res, rhs);
  }

  return mul;
}
}  // namespace

std::pair<SExpression, int64_t> SExpression::operator/(
    SExpression rhs_wrapper) const {
  SENode* lhs = node_;
  SENode* rhs = rhs_wrapper.node_;
  // Check for division by 0.
  if (rhs->AsSEConstantNode() &&
      !rhs->AsSEConstantNode()->FoldToSingleValue()) {
    return {scev_->CreateCantComputeNode(), 0};
  }

  // Trivial case.
  if (lhs->AsSEConstantNode() && rhs->AsSEConstantNode()) {
    int64_t lhs_value = lhs->AsSEConstantNode()->FoldToSingleValue();
    int64_t rhs_value = rhs->AsSEConstantNode()->FoldToSingleValue();
    return {scev_->CreateConstant(lhs_value / rhs_value),
            lhs_value % rhs_value};
  }

  // look for a "c U / U" pattern.
  if (lhs->AsSEMultiplyNode()) {
    assert(lhs->GetChildren().size() == 2 &&
           "More than 2 operand for a multiply node.");
    SENode* res = RemoveOneNodeFromMultiplyChain(lhs->AsSEMultiplyNode(), rhs);
    if (res != lhs) {
      return {res, 0};
    }
  }

  return {scev_->CreateCantComputeNode(), 0};
}

}  // namespace opt
}  // namespace spvtools
