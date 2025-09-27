// Copyright (c) 2018 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASI,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SOURCE_OPT_SCALAR_ANALYSIS_NODES_H_
#define SOURCE_OPT_SCALAR_ANALYSIS_NODES_H_

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "source/opt/tree_iterator.h"

namespace spvtools {
namespace opt {

class Loop;
class ScalarEvolutionAnalysis;
class SEConstantNode;
class SERecurrentNode;
class SEAddNode;
class SEMultiplyNode;
class SENegative;
class SEValueUnknown;
class SECantCompute;

// Abstract class representing a node in the scalar evolution DAG. Each node
// contains a vector of pointers to its children and each subclass of SENode
// implements GetType and an As method to allow casting. SENodes can be hashed
// using the SENodeHash functor. The vector of children is sorted when a node is
// added. This is important as it allows the hash of X+Y to be the same as Y+X.
class SENode {
 public:
  enum SENodeType {
    Constant,
    RecurrentAddExpr,
    Add,
    Multiply,
    Negative,
    ValueUnknown,
    CanNotCompute
  };

  using ChildContainerType = std::vector<SENode*>;

  explicit SENode(ScalarEvolutionAnalysis* parent_analysis)
      : parent_analysis_(parent_analysis), unique_id_(++NumberOfNodes) {}

  virtual SENodeType GetType() const = 0;

  virtual ~SENode() {}

  virtual inline void AddChild(SENode* child) {
    // If this is a constant node, assert.
    if (AsSEConstantNode()) {
      assert(false && "Trying to add a child node to a constant!");
    }

    // Find the first point in the vector where |child| is greater than the node
    // currently in the vector.
    auto find_first_less_than = [child](const SENode* node) {
      return child->unique_id_ <= node->unique_id_;
    };

    auto position = std::find_if_not(children_.begin(), children_.end(),
                                     find_first_less_than);
    // Children are sorted so the hashing and equality operator will be the same
    // for a node with the same children. X+Y should be the same as Y+X.
    children_.insert(position, child);
  }

  // Get the type as an std::string. This is used to represent the node in the
  // dot output and is used to hash the type as well.
  std::string AsString() const;

  // Dump the SENode and its immediate children, if |recurse| is true then it
  // will recurse through all children to print the DAG starting from this node
  // as a root.
  void DumpDot(std::ostream& out, bool recurse = false) const;

  // Checks if two nodes are the same by hashing them.
  bool operator==(const SENode& other) const;

  // Checks if two nodes are not the same by comparing the hashes.
  bool operator!=(const SENode& other) const;

  // Return the child node at |index|.
  inline SENode* GetChild(size_t index) { return children_[index]; }
  inline const SENode* GetChild(size_t index) const { return children_[index]; }

  // Iterator to iterate over the child nodes.
  using iterator = ChildContainerType::iterator;
  using const_iterator = ChildContainerType::const_iterator;

  // Iterate over immediate child nodes.
  iterator begin() { return children_.begin(); }
  iterator end() { return children_.end(); }

  // Constant overloads for iterating over immediate child nodes.
  const_iterator begin() const { return children_.cbegin(); }
  const_iterator end() const { return children_.cend(); }
  const_iterator cbegin() { return children_.cbegin(); }
  const_iterator cend() { return children_.cend(); }

  // Collect all the recurrent nodes in this SENode
  std::vector<SERecurrentNode*> CollectRecurrentNodes() {
    std::vector<SERecurrentNode*> recurrent_nodes{};

    if (auto recurrent_node = AsSERecurrentNode()) {
      recurrent_nodes.push_back(recurrent_node);
    }

    for (auto child : GetChildren()) {
      auto child_recurrent_nodes = child->CollectRecurrentNodes();
      recurrent_nodes.insert(recurrent_nodes.end(),
                             child_recurrent_nodes.begin(),
                             child_recurrent_nodes.end());
    }

    return recurrent_nodes;
  }

  // Collect all the value unknown nodes in this SENode
  std::vector<SEValueUnknown*> CollectValueUnknownNodes() {
    std::vector<SEValueUnknown*> value_unknown_nodes{};

    if (auto value_unknown_node = AsSEValueUnknown()) {
      value_unknown_nodes.push_back(value_unknown_node);
    }

    for (auto child : GetChildren()) {
      auto child_value_unknown_nodes = child->CollectValueUnknownNodes();
      value_unknown_nodes.insert(value_unknown_nodes.end(),
                                 child_value_unknown_nodes.begin(),
                                 child_value_unknown_nodes.end());
    }

    return value_unknown_nodes;
  }

  // Iterator to iterate over the entire DAG. Even though we are using the tree
  // iterator it should still be safe to iterate over. However, nodes with
  // multiple parents will be visited multiple times, unlike in a tree.
  using dag_iterator = TreeDFIterator<SENode>;
  using const_dag_iterator = TreeDFIterator<const SENode>;

  // Iterate over all child nodes in the graph.
  dag_iterator graph_begin() { return dag_iterator(this); }
  dag_iterator graph_end() { return dag_iterator(); }
  const_dag_iterator graph_begin() const { return graph_cbegin(); }
  const_dag_iterator graph_end() const { return graph_cend(); }
  const_dag_iterator graph_cbegin() const { return const_dag_iterator(this); }
  const_dag_iterator graph_cend() const { return const_dag_iterator(); }

  // Return the vector of immediate children.
  const ChildContainerType& GetChildren() const { return children_; }
  ChildContainerType& GetChildren() { return children_; }

  // Return true if this node is a can't compute node.
  bool IsCantCompute() const { return GetType() == CanNotCompute; }

// Implements a casting method for each type.
// clang-format off
#define DeclareCastMethod(target)                  \
  virtual target* As##target() { return nullptr; } \
  virtual const target* As##target() const { return nullptr; }
  DeclareCastMethod(SEConstantNode)
  DeclareCastMethod(SERecurrentNode)
  DeclareCastMethod(SEAddNode)
  DeclareCastMethod(SEMultiplyNode)
  DeclareCastMethod(SENegative)
  DeclareCastMethod(SEValueUnknown)
  DeclareCastMethod(SECantCompute)
#undef DeclareCastMethod

  // Get the analysis which has this node in its cache.
  inline ScalarEvolutionAnalysis* GetParentAnalysis() const {
    return parent_analysis_;
  }

 protected:
  ChildContainerType children_;

  ScalarEvolutionAnalysis* parent_analysis_;

  // The unique id of this node, assigned on creation by incrementing the static
  // node count.
  uint32_t unique_id_;

  // The number of nodes created.
  static uint32_t NumberOfNodes;
};
// clang-format on

// Function object to handle the hashing of SENodes. Hashing algorithm hashes
// the type (as a string), the literal value of any constants, and the child
// pointers which are assumed to be unique.
struct SENodeHash {
  size_t operator()(const std::unique_ptr<SENode>& node) const;
  size_t operator()(const SENode* node) const;
};

// A node representing a constant integer.
class SEConstantNode : public SENode {
 public:
  SEConstantNode(ScalarEvolutionAnalysis* parent_analysis, int64_t value)
      : SENode(parent_analysis), literal_value_(value) {}

  SENodeType GetType() const final { return Constant; }

  int64_t FoldToSingleValue() const { return literal_value_; }

  SEConstantNode* AsSEConstantNode() override { return this; }
  const SEConstantNode* AsSEConstantNode() const override { return this; }

  inline void AddChild(SENode*) final {
    assert(false && "Attempting to add a child to a constant node!");
  }

 protected:
  int64_t literal_value_;
};

// A node representing a recurrent expression in the code. A recurrent
// expression is an expression whose value can be expressed as a linear
// expression of the loop iterations. Such as an induction variable. The actual
// value of a recurrent expression is coefficent_ * iteration + offset_, hence
// an induction variable i=0, i++ becomes a recurrent expression with an offset
// of zero and a coefficient of one.
class SERecurrentNode : public SENode {
 public:
  SERecurrentNode(ScalarEvolutionAnalysis* parent_analysis, const Loop* loop)
      : SENode(parent_analysis), loop_(loop) {}

  SENodeType GetType() const final { return RecurrentAddExpr; }

  inline void AddCoefficient(SENode* child) {
    coefficient_ = child;
    SENode::AddChild(child);
  }

  inline void AddOffset(SENode* child) {
    offset_ = child;
    SENode::AddChild(child);
  }

  inline const SENode* GetCoefficient() const { return coefficient_; }
  inline SENode* GetCoefficient() { return coefficient_; }

  inline const SENode* GetOffset() const { return offset_; }
  inline SENode* GetOffset() { return offset_; }

  // Return the loop which this recurrent expression is recurring within.
  const Loop* GetLoop() const { return loop_; }

  SERecurrentNode* AsSERecurrentNode() override { return this; }
  const SERecurrentNode* AsSERecurrentNode() const override { return this; }

 private:
  SENode* coefficient_;
  SENode* offset_;
  const Loop* loop_;
};

// A node representing an addition operation between child nodes.
class SEAddNode : public SENode {
 public:
  explicit SEAddNode(ScalarEvolutionAnalysis* parent_analysis)
      : SENode(parent_analysis) {}

  SENodeType GetType() const final { return Add; }

  SEAddNode* AsSEAddNode() override { return this; }
  const SEAddNode* AsSEAddNode() const override { return this; }
};

// A node representing a multiply operation between child nodes.
class SEMultiplyNode : public SENode {
 public:
  explicit SEMultiplyNode(ScalarEvolutionAnalysis* parent_analysis)
      : SENode(parent_analysis) {}

  SENodeType GetType() const final { return Multiply; }

  SEMultiplyNode* AsSEMultiplyNode() override { return this; }
  const SEMultiplyNode* AsSEMultiplyNode() const override { return this; }
};

// A node representing a unary negative operation.
class SENegative : public SENode {
 public:
  explicit SENegative(ScalarEvolutionAnalysis* parent_analysis)
      : SENode(parent_analysis) {}

  SENodeType GetType() const final { return Negative; }

  SENegative* AsSENegative() override { return this; }
  const SENegative* AsSENegative() const override { return this; }
};

// A node representing a value which we do not know the value of, such as a load
// instruction.
class SEValueUnknown : public SENode {
 public:
  // SEValueUnknowns must come from an instruction |unique_id| is the unique id
  // of that instruction. This is so we cancompare value unknowns and have a
  // unique value unknown for each instruction.
  SEValueUnknown(ScalarEvolutionAnalysis* parent_analysis, uint32_t result_id)
      : SENode(parent_analysis), result_id_(result_id) {}

  SENodeType GetType() const final { return ValueUnknown; }

  SEValueUnknown* AsSEValueUnknown() override { return this; }
  const SEValueUnknown* AsSEValueUnknown() const override { return this; }

  inline uint32_t ResultId() const { return result_id_; }

 private:
  uint32_t result_id_;
};

// A node which we cannot reason about at all.
class SECantCompute : public SENode {
 public:
  explicit SECantCompute(ScalarEvolutionAnalysis* parent_analysis)
      : SENode(parent_analysis) {}

  SENodeType GetType() const final { return CanNotCompute; }

  SECantCompute* AsSECantCompute() override { return this; }
  const SECantCompute* AsSECantCompute() const override { return this; }
};

}  // namespace opt
}  // namespace spvtools
#endif  // SOURCE_OPT_SCALAR_ANALYSIS_NODES_H_
