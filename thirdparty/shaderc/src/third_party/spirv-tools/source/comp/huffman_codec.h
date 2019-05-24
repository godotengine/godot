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

// Contains utils for reading, writing and debug printing bit streams.

#ifndef SOURCE_COMP_HUFFMAN_CODEC_H_
#define SOURCE_COMP_HUFFMAN_CODEC_H_

#include <algorithm>
#include <cassert>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <ostream>
#include <queue>
#include <sstream>
#include <stack>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace spvtools {
namespace comp {

// Used to generate and apply a Huffman coding scheme.
// |Val| is the type of variable being encoded (for example a string or a
// literal).
template <class Val>
class HuffmanCodec {
 public:
  // Huffman tree node.
  struct Node {
    Node() {}

    // Creates Node from serialization leaving weight and id undefined.
    Node(const Val& in_value, uint32_t in_left, uint32_t in_right)
        : value(in_value), left(in_left), right(in_right) {}

    Val value = Val();
    uint32_t weight = 0;
    // Ids are issued sequentially starting from 1. Ids are used as an ordering
    // tie-breaker, to make sure that the ordering (and resulting coding scheme)
    // are consistent accross multiple platforms.
    uint32_t id = 0;
    // Handles of children.
    uint32_t left = 0;
    uint32_t right = 0;
  };

  // Creates Huffman codec from a histogramm.
  // Histogramm counts must not be zero.
  explicit HuffmanCodec(const std::map<Val, uint32_t>& hist) {
    if (hist.empty()) return;

    // Heuristic estimate.
    nodes_.reserve(3 * hist.size());

    // Create NIL.
    CreateNode();

    // The queue is sorted in ascending order by weight (or by node id if
    // weights are equal).
    std::vector<uint32_t> queue_vector;
    queue_vector.reserve(hist.size());
    std::priority_queue<uint32_t, std::vector<uint32_t>,
                        std::function<bool(uint32_t, uint32_t)>>
        queue(std::bind(&HuffmanCodec::LeftIsBigger, this,
                        std::placeholders::_1, std::placeholders::_2),
              std::move(queue_vector));

    // Put all leaves in the queue.
    for (const auto& pair : hist) {
      const uint32_t node = CreateNode();
      MutableValueOf(node) = pair.first;
      MutableWeightOf(node) = pair.second;
      assert(WeightOf(node));
      queue.push(node);
    }

    // Form the tree by combining two subtrees with the least weight,
    // and pushing the root of the new tree in the queue.
    while (true) {
      // We push a node at the end of each iteration, so the queue is never
      // supposed to be empty at this point, unless there are no leaves, but
      // that case was already handled.
      assert(!queue.empty());
      const uint32_t right = queue.top();
      queue.pop();

      // If the queue is empty at this point, then the last node is
      // the root of the complete Huffman tree.
      if (queue.empty()) {
        root_ = right;
        break;
      }

      const uint32_t left = queue.top();
      queue.pop();

      // Combine left and right into a new tree and push it into the queue.
      const uint32_t parent = CreateNode();
      MutableWeightOf(parent) = WeightOf(right) + WeightOf(left);
      MutableLeftOf(parent) = left;
      MutableRightOf(parent) = right;
      queue.push(parent);
    }

    // Traverse the tree and form encoding table.
    CreateEncodingTable();
  }

  // Creates Huffman codec from saved tree structure.
  // |nodes| is the list of nodes of the tree, nodes[0] being NIL.
  // |root_handle| is the index of the root node.
  HuffmanCodec(uint32_t root_handle, std::vector<Node>&& nodes) {
    nodes_ = std::move(nodes);
    assert(!nodes_.empty());
    assert(root_handle > 0 && root_handle < nodes_.size());
    assert(!LeftOf(0) && !RightOf(0));

    root_ = root_handle;

    // Traverse the tree and form encoding table.
    CreateEncodingTable();
  }

  // Serializes the codec in the following text format:
  // (<root_handle>, {
  //   {0, 0, 0},
  //   {val1, left1, right1},
  //   {val2, left2, right2},
  //   ...
  // })
  std::string SerializeToText(int indent_num_whitespaces) const {
    const bool value_is_text = std::is_same<Val, std::string>::value;

    const std::string indent1 = std::string(indent_num_whitespaces, ' ');
    const std::string indent2 = std::string(indent_num_whitespaces + 2, ' ');

    std::stringstream code;
    code << "(" << root_ << ", {\n";

    for (const Node& node : nodes_) {
      code << indent2 << "{";
      if (value_is_text) code << "\"";
      code << node.value;
      if (value_is_text) code << "\"";
      code << ", " << node.left << ", " << node.right << "},\n";
    }

    code << indent1 << "})";

    return code.str();
  }

  // Prints the Huffman tree in the following format:
  // w------w------'x'
  //        w------'y'
  // Where w stands for the weight of the node.
  // Right tree branches appear above left branches. Taking the right path
  // adds 1 to the code, taking the left adds 0.
  void PrintTree(std::ostream& out) const { PrintTreeInternal(out, root_, 0); }

  // Traverses the tree and prints the Huffman table: value, code
  // and optionally node weight for every leaf.
  void PrintTable(std::ostream& out, bool print_weights = true) {
    std::queue<std::pair<uint32_t, std::string>> queue;
    queue.emplace(root_, "");

    while (!queue.empty()) {
      const uint32_t node = queue.front().first;
      const std::string code = queue.front().second;
      queue.pop();
      if (!RightOf(node) && !LeftOf(node)) {
        out << ValueOf(node);
        if (print_weights) out << " " << WeightOf(node);
        out << " " << code << std::endl;
      } else {
        if (LeftOf(node)) queue.emplace(LeftOf(node), code + "0");

        if (RightOf(node)) queue.emplace(RightOf(node), code + "1");
      }
    }
  }

  // Returns the Huffman table. The table was built at at construction time,
  // this function just returns a const reference.
  const std::unordered_map<Val, std::pair<uint64_t, size_t>>& GetEncodingTable()
      const {
    return encoding_table_;
  }

  // Encodes |val| and stores its Huffman code in the lower |num_bits| of
  // |bits|. Returns false of |val| is not in the Huffman table.
  bool Encode(const Val& val, uint64_t* bits, size_t* num_bits) const {
    auto it = encoding_table_.find(val);
    if (it == encoding_table_.end()) return false;
    *bits = it->second.first;
    *num_bits = it->second.second;
    return true;
  }

  // Reads bits one-by-one using callback |read_bit| until a match is found.
  // Matching value is stored in |val|. Returns false if |read_bit| terminates
  // before a code was mathced.
  // |read_bit| has type bool func(bool* bit). When called, the next bit is
  // stored in |bit|. |read_bit| returns false if the stream terminates
  // prematurely.
  bool DecodeFromStream(const std::function<bool(bool*)>& read_bit,
                        Val* val) const {
    uint32_t node = root_;
    while (true) {
      assert(node);

      if (!RightOf(node) && !LeftOf(node)) {
        *val = ValueOf(node);
        return true;
      }

      bool go_right;
      if (!read_bit(&go_right)) return false;

      if (go_right)
        node = RightOf(node);
      else
        node = LeftOf(node);
    }

    assert(0);
    return false;
  }

 private:
  // Returns value of the node referenced by |handle|.
  Val ValueOf(uint32_t node) const { return nodes_.at(node).value; }

  // Returns left child of |node|.
  uint32_t LeftOf(uint32_t node) const { return nodes_.at(node).left; }

  // Returns right child of |node|.
  uint32_t RightOf(uint32_t node) const { return nodes_.at(node).right; }

  // Returns weight of |node|.
  uint32_t WeightOf(uint32_t node) const { return nodes_.at(node).weight; }

  // Returns id of |node|.
  uint32_t IdOf(uint32_t node) const { return nodes_.at(node).id; }

  // Returns mutable reference to value of |node|.
  Val& MutableValueOf(uint32_t node) {
    assert(node);
    return nodes_.at(node).value;
  }

  // Returns mutable reference to handle of left child of |node|.
  uint32_t& MutableLeftOf(uint32_t node) {
    assert(node);
    return nodes_.at(node).left;
  }

  // Returns mutable reference to handle of right child of |node|.
  uint32_t& MutableRightOf(uint32_t node) {
    assert(node);
    return nodes_.at(node).right;
  }

  // Returns mutable reference to weight of |node|.
  uint32_t& MutableWeightOf(uint32_t node) { return nodes_.at(node).weight; }

  // Returns mutable reference to id of |node|.
  uint32_t& MutableIdOf(uint32_t node) { return nodes_.at(node).id; }

  // Returns true if |left| has bigger weight than |right|. Node ids are
  // used as tie-breaker.
  bool LeftIsBigger(uint32_t left, uint32_t right) const {
    if (WeightOf(left) == WeightOf(right)) {
      assert(IdOf(left) != IdOf(right));
      return IdOf(left) > IdOf(right);
    }
    return WeightOf(left) > WeightOf(right);
  }

  // Prints subtree (helper function used by PrintTree).
  void PrintTreeInternal(std::ostream& out, uint32_t node, size_t depth) const {
    if (!node) return;

    const size_t kTextFieldWidth = 7;

    if (!RightOf(node) && !LeftOf(node)) {
      out << ValueOf(node) << std::endl;
    } else {
      if (RightOf(node)) {
        std::stringstream label;
        label << std::setfill('-') << std::left << std::setw(kTextFieldWidth)
              << WeightOf(RightOf(node));
        out << label.str();
        PrintTreeInternal(out, RightOf(node), depth + 1);
      }

      if (LeftOf(node)) {
        out << std::string(depth * kTextFieldWidth, ' ');
        std::stringstream label;
        label << std::setfill('-') << std::left << std::setw(kTextFieldWidth)
              << WeightOf(LeftOf(node));
        out << label.str();
        PrintTreeInternal(out, LeftOf(node), depth + 1);
      }
    }
  }

  // Traverses the Huffman tree and saves paths to the leaves as bit
  // sequences to encoding_table_.
  void CreateEncodingTable() {
    struct Context {
      Context(uint32_t in_node, uint64_t in_bits, size_t in_depth)
          : node(in_node), bits(in_bits), depth(in_depth) {}
      uint32_t node;
      // Huffman tree depth cannot exceed 64 as histogramm counts are expected
      // to be positive and limited by numeric_limits<uint32_t>::max().
      // For practical applications tree depth would be much smaller than 64.
      uint64_t bits;
      size_t depth;
    };

    std::queue<Context> queue;
    queue.emplace(root_, 0, 0);

    while (!queue.empty()) {
      const Context& context = queue.front();
      const uint32_t node = context.node;
      const uint64_t bits = context.bits;
      const size_t depth = context.depth;
      queue.pop();

      if (!RightOf(node) && !LeftOf(node)) {
        auto insertion_result = encoding_table_.emplace(
            ValueOf(node), std::pair<uint64_t, size_t>(bits, depth));
        assert(insertion_result.second);
        (void)insertion_result;
      } else {
        if (LeftOf(node)) queue.emplace(LeftOf(node), bits, depth + 1);

        if (RightOf(node))
          queue.emplace(RightOf(node), bits | (1ULL << depth), depth + 1);
      }
    }
  }

  // Creates new Huffman tree node and stores it in the deleter array.
  uint32_t CreateNode() {
    const uint32_t handle = static_cast<uint32_t>(nodes_.size());
    nodes_.emplace_back(Node());
    nodes_.back().id = next_node_id_++;
    return handle;
  }

  // Huffman tree root handle.
  uint32_t root_ = 0;

  // Huffman tree deleter.
  std::vector<Node> nodes_;

  // Encoding table value -> {bits, num_bits}.
  // Huffman codes are expected to never exceed 64 bit length (this is in fact
  // impossible if frequencies are stored as uint32_t).
  std::unordered_map<Val, std::pair<uint64_t, size_t>> encoding_table_;

  // Next node id issued by CreateNode();
  uint32_t next_node_id_ = 1;
};

}  // namespace comp
}  // namespace spvtools

#endif  // SOURCE_COMP_HUFFMAN_CODEC_H_
