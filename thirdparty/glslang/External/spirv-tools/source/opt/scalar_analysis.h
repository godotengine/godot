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

#ifndef SOURCE_OPT_SCALAR_ANALYSIS_H_
#define SOURCE_OPT_SCALAR_ANALYSIS_H_

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "source/opt/basic_block.h"
#include "source/opt/instruction.h"
#include "source/opt/scalar_analysis_nodes.h"

namespace spvtools {
namespace opt {

class IRContext;
class Loop;

// Manager for the Scalar Evolution analysis. Creates and maintains a DAG of
// scalar operations generated from analysing the use def graph from incoming
// instructions. Each node is hashed as it is added so like node (for instance,
// two induction variables i=0,i++ and j=0,j++) become the same node. After
// creating a DAG with AnalyzeInstruction it can the be simplified into a more
// usable form with SimplifyExpression.
class ScalarEvolutionAnalysis {
 public:
  explicit ScalarEvolutionAnalysis(IRContext* context);

  // Create a unary negative node on |operand|.
  SENode* CreateNegation(SENode* operand);

  // Creates a subtraction between the two operands by adding |operand_1| to the
  // negation of |operand_2|.
  SENode* CreateSubtraction(SENode* operand_1, SENode* operand_2);

  // Create an addition node between two operands. The |simplify| when set will
  // allow the function to return an SEConstant instead of an addition if the
  // two input operands are also constant.
  SENode* CreateAddNode(SENode* operand_1, SENode* operand_2);

  // Create a multiply node between two operands.
  SENode* CreateMultiplyNode(SENode* operand_1, SENode* operand_2);

  // Create a node representing a constant integer.
  SENode* CreateConstant(int64_t integer);

  // Create a value unknown node, such as a load.
  SENode* CreateValueUnknownNode(const Instruction* inst);

  // Create a CantComputeNode. Used to exit out of analysis.
  SENode* CreateCantComputeNode();

  // Create a new recurrent node with |offset| and |coefficient|, with respect
  // to |loop|.
  SENode* CreateRecurrentExpression(const Loop* loop, SENode* offset,
                                    SENode* coefficient);

  // Construct the DAG by traversing use def chain of |inst|.
  SENode* AnalyzeInstruction(const Instruction* inst);

  // Simplify the |node| by grouping like terms or if contains a recurrent
  // expression, rewrite the graph so the whole DAG (from |node| down) is in
  // terms of that recurrent expression.
  //
  // For example.
  // Induction variable i=0, i++ would produce Rec(0,1) so i+1 could be
  // transformed into Rec(1,1).
  //
  // X+X*2+Y-Y+34-17 would be transformed into 3*X + 17, where X and Y are
  // ValueUnknown nodes (such as a load instruction).
  SENode* SimplifyExpression(SENode* node);

  // Add |prospective_node| into the cache and return a raw pointer to it. If
  // |prospective_node| is already in the cache just return the raw pointer.
  SENode* GetCachedOrAdd(std::unique_ptr<SENode> prospective_node);

  // Checks that the graph starting from |node| is invariant to the |loop|.
  bool IsLoopInvariant(const Loop* loop, const SENode* node) const;

  // Sets |is_gt_zero| to true if |node| represent a value always strictly
  // greater than 0. The result of |is_gt_zero| is valid only if the function
  // returns true.
  bool IsAlwaysGreaterThanZero(SENode* node, bool* is_gt_zero) const;

  // Sets |is_ge_zero| to true if |node| represent a value greater or equals to
  // 0. The result of |is_ge_zero| is valid only if the function returns true.
  bool IsAlwaysGreaterOrEqualToZero(SENode* node, bool* is_ge_zero) const;

  // Find the recurrent term belonging to |loop| in the graph starting from
  // |node| and return the coefficient of that recurrent term. Constant zero
  // will be returned if no recurrent could be found. |node| should be in
  // simplest form.
  SENode* GetCoefficientFromRecurrentTerm(SENode* node, const Loop* loop);

  // Return a rebuilt graph starting from |node| with the recurrent expression
  // belonging to |loop| being zeroed out. Returned node will be simplified.
  SENode* BuildGraphWithoutRecurrentTerm(SENode* node, const Loop* loop);

  // Return the recurrent term belonging to |loop| if it appears in the graph
  // starting at |node| or null if it doesn't.
  SERecurrentNode* GetRecurrentTerm(SENode* node, const Loop* loop);

  SENode* UpdateChildNode(SENode* parent, SENode* child, SENode* new_child);

  // The loops in |loop_pair| will be considered the same when constructing
  // SERecurrentNode objects. This enables analysing dependencies that will be
  // created during loop fusion.
  void AddLoopsToPretendAreTheSame(
      const std::pair<const Loop*, const Loop*>& loop_pair) {
    pretend_equal_[std::get<1>(loop_pair)] = std::get<0>(loop_pair);
  }

 private:
  SENode* AnalyzeConstant(const Instruction* inst);

  // Handles both addition and subtraction. If the |instruction| is OpISub
  // then the resulting node will be op1+(-op2) otherwise if it is OpIAdd then
  // the result will be op1+op2. |instruction| must be OpIAdd or OpISub.
  SENode* AnalyzeAddOp(const Instruction* instruction);

  SENode* AnalyzeMultiplyOp(const Instruction* multiply);

  SENode* AnalyzePhiInstruction(const Instruction* phi);

  IRContext* context_;

  // A map of instructions to SENodes. This is used to track recurrent
  // expressions as they are added when analyzing instructions. Recurrent
  // expressions come from phi nodes which by nature can include recursion so we
  // check if nodes have already been built when analyzing instructions.
  std::map<const Instruction*, SENode*> recurrent_node_map_;

  // On creation we create and cache the CantCompute node so we not need to
  // perform a needless create step.
  SENode* cached_cant_compute_;

  // Helper functor to allow two unique_ptr to nodes to be compare. Only
  // needed
  // for the unordered_set implementation.
  struct NodePointersEquality {
    bool operator()(const std::unique_ptr<SENode>& lhs,
                    const std::unique_ptr<SENode>& rhs) const {
      return *lhs == *rhs;
    }
  };

  // Cache of nodes. All pointers to the nodes are references to the memory
  // managed by they set.
  std::unordered_set<std::unique_ptr<SENode>, SENodeHash, NodePointersEquality>
      node_cache_;

  // Loops that should be considered the same for performing analysis for loop
  // fusion.
  std::map<const Loop*, const Loop*> pretend_equal_;
};

// Wrapping class to manipulate SENode pointer using + - * / operators.
class SExpression {
 public:
  // Implicit on purpose !
  SExpression(SENode* node)
      : node_(node->GetParentAnalysis()->SimplifyExpression(node)),
        scev_(node->GetParentAnalysis()) {}

  inline operator SENode*() const { return node_; }
  inline SENode* operator->() const { return node_; }
  const SENode& operator*() const { return *node_; }

  inline ScalarEvolutionAnalysis* GetScalarEvolutionAnalysis() const {
    return scev_;
  }

  inline SExpression operator+(SENode* rhs) const;
  template <typename T,
            typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
  inline SExpression operator+(T integer) const;
  inline SExpression operator+(SExpression rhs) const;

  inline SExpression operator-() const;
  inline SExpression operator-(SENode* rhs) const;
  template <typename T,
            typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
  inline SExpression operator-(T integer) const;
  inline SExpression operator-(SExpression rhs) const;

  inline SExpression operator*(SENode* rhs) const;
  template <typename T,
            typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
  inline SExpression operator*(T integer) const;
  inline SExpression operator*(SExpression rhs) const;

  template <typename T,
            typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
  inline std::pair<SExpression, int64_t> operator/(T integer) const;
  // Try to perform a division. Returns the pair <this.node_ / rhs, division
  // remainder>. If it fails to simplify it, the function returns a
  // CanNotCompute node.
  std::pair<SExpression, int64_t> operator/(SExpression rhs) const;

 private:
  SENode* node_;
  ScalarEvolutionAnalysis* scev_;
};

inline SExpression SExpression::operator+(SENode* rhs) const {
  return scev_->CreateAddNode(node_, rhs);
}

template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type>
inline SExpression SExpression::operator+(T integer) const {
  return *this + scev_->CreateConstant(integer);
}

inline SExpression SExpression::operator+(SExpression rhs) const {
  return *this + rhs.node_;
}

inline SExpression SExpression::operator-() const {
  return scev_->CreateNegation(node_);
}

inline SExpression SExpression::operator-(SENode* rhs) const {
  return *this + scev_->CreateNegation(rhs);
}

template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type>
inline SExpression SExpression::operator-(T integer) const {
  return *this - scev_->CreateConstant(integer);
}

inline SExpression SExpression::operator-(SExpression rhs) const {
  return *this - rhs.node_;
}

inline SExpression SExpression::operator*(SENode* rhs) const {
  return scev_->CreateMultiplyNode(node_, rhs);
}

template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type>
inline SExpression SExpression::operator*(T integer) const {
  return *this * scev_->CreateConstant(integer);
}

inline SExpression SExpression::operator*(SExpression rhs) const {
  return *this * rhs.node_;
}

template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type>
inline std::pair<SExpression, int64_t> SExpression::operator/(T integer) const {
  return *this / scev_->CreateConstant(integer);
}

template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type>
inline SExpression operator+(T lhs, SExpression rhs) {
  return rhs + lhs;
}
inline SExpression operator+(SENode* lhs, SExpression rhs) { return rhs + lhs; }

template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type>
inline SExpression operator-(T lhs, SExpression rhs) {
  // NOLINTNEXTLINE(whitespace/braces)
  return SExpression{rhs.GetScalarEvolutionAnalysis()->CreateConstant(lhs)} -
         rhs;
}
inline SExpression operator-(SENode* lhs, SExpression rhs) {
  // NOLINTNEXTLINE(whitespace/braces)
  return SExpression{lhs} - rhs;
}

template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type>
inline SExpression operator*(T lhs, SExpression rhs) {
  return rhs * lhs;
}
inline SExpression operator*(SENode* lhs, SExpression rhs) { return rhs * lhs; }

template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type>
inline std::pair<SExpression, int64_t> operator/(T lhs, SExpression rhs) {
  // NOLINTNEXTLINE(whitespace/braces)
  return SExpression{rhs.GetScalarEvolutionAnalysis()->CreateConstant(lhs)} /
         rhs;
}
inline std::pair<SExpression, int64_t> operator/(SENode* lhs, SExpression rhs) {
  // NOLINTNEXTLINE(whitespace/braces)
  return SExpression{lhs} / rhs;
}

}  // namespace opt
}  // namespace spvtools
#endif  // SOURCE_OPT_SCALAR_ANALYSIS_H_
