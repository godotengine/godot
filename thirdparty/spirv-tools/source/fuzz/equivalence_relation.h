// Copyright (c) 2019 Google LLC
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

#ifndef SOURCE_FUZZ_EQUIVALENCE_RELATION_H_
#define SOURCE_FUZZ_EQUIVALENCE_RELATION_H_

#include <algorithm>
#include <cassert>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "source/util/make_unique.h"

namespace spvtools {
namespace fuzz {

// A class for representing an equivalence relation on objects of type |T|,
// which should be a value type.  The type |T| is required to have a copy
// constructor, and |PointerHashT| and |PointerEqualsT| must be functors
// providing hashing and equality testing functionality for pointers to objects
// of type |T|.
//
// A disjoint-set (a.k.a. union-find or merge-find) data structure is used to
// represent the equivalence relation.  Path compression is used.  Union by
// rank/size is not used.
//
// Each disjoint set is represented as a tree, rooted at the representative
// of the set.
//
// Getting the representative of a value simply requires chasing parent pointers
// from the value until you reach the root.
//
// Checking equivalence of two elements requires checking that the
// representatives are equal.
//
// Traversing the tree rooted at a value's representative visits the value's
// equivalence class.
//
// |PointerHashT| and |PointerEqualsT| are used to define *equality* between
// values, and otherwise are *not* used to define the equivalence relation
// (except that equal values are equivalent).  The equivalence relation is
// constructed by repeatedly adding pairs of (typically non-equal) values that
// are deemed to be equivalent.
//
// For example in an equivalence relation on integers, 1 and 5 might be added
// as equivalent, so that IsEquivalent(1, 5) holds, because they represent
// IDs in a SPIR-V binary that are known to contain the same value at run time,
// but clearly 1 != 5.  Since 1 and 1 are equal, IsEquivalent(1, 1) will also
// hold.
//
// Each unique (up to equality) value added to the relation is copied into
// |owned_values_|, so there is one canonical memory address per unique value.
// Uniqueness is ensured by storing (and checking) a set of pointers to these
// values in |value_set_|, which uses |PointerHashT| and |PointerEqualsT|.
//
// |parent_| and |children_| encode the equivalence relation, i.e., the trees.
template <typename T, typename PointerHashT, typename PointerEqualsT>
class EquivalenceRelation {
 public:
  // Requires that |value1| and |value2| are already registered in the
  // equivalence relation.  Merges the equivalence classes associated with
  // |value1| and |value2|.
  void MakeEquivalent(const T& value1, const T& value2) {
    assert(Exists(value1) &&
           "Precondition: value1 must already be registered.");
    assert(Exists(value2) &&
           "Precondition: value2 must already be registered.");

    // Look up canonical pointers to each of the values in the value pool.
    const T* value1_ptr = *value_set_.find(&value1);
    const T* value2_ptr = *value_set_.find(&value2);

    // If the values turn out to be identical, they are already in the same
    // equivalence class so there is nothing to do.
    if (value1_ptr == value2_ptr) {
      return;
    }

    // Find the representative for each value's equivalence class, and if they
    // are not already in the same class, make one the parent of the other.
    const T* representative1 = Find(value1_ptr);
    const T* representative2 = Find(value2_ptr);
    assert(representative1 && "Representatives should never be null.");
    assert(representative2 && "Representatives should never be null.");
    if (representative1 != representative2) {
      parent_[representative1] = representative2;
      children_[representative2].push_back(representative1);
    }
  }

  // Requires that |value| is not known to the equivalence relation. Registers
  // it in its own equivalence class and returns a pointer to the equivalence
  // class representative.
  const T* Register(const T& value) {
    assert(!Exists(value));

    // This relies on T having a copy constructor.
    auto unique_pointer_to_value = MakeUnique<T>(value);
    auto pointer_to_value = unique_pointer_to_value.get();
    owned_values_.push_back(std::move(unique_pointer_to_value));
    value_set_.insert(pointer_to_value);

    // Initially say that the value is its own parent and that it has no
    // children.
    assert(pointer_to_value && "Representatives should never be null.");
    parent_[pointer_to_value] = pointer_to_value;
    children_[pointer_to_value] = std::vector<const T*>();

    return pointer_to_value;
  }

  // Returns exactly one representative per equivalence class.
  std::vector<const T*> GetEquivalenceClassRepresentatives() const {
    std::vector<const T*> result;
    for (auto& value : owned_values_) {
      if (parent_[value.get()] == value.get()) {
        result.push_back(value.get());
      }
    }
    return result;
  }

  // Returns pointers to all values in the equivalence class of |value|, which
  // must already be part of the equivalence relation.
  std::vector<const T*> GetEquivalenceClass(const T& value) const {
    assert(Exists(value));

    std::vector<const T*> result;

    // Traverse the tree of values rooted at the representative of the
    // equivalence class to which |value| belongs, and collect up all the values
    // that are encountered.  This constitutes the whole equivalence class.
    std::vector<const T*> stack;
    stack.push_back(Find(*value_set_.find(&value)));
    while (!stack.empty()) {
      const T* item = stack.back();
      result.push_back(item);
      stack.pop_back();
      for (auto child : children_[item]) {
        stack.push_back(child);
      }
    }
    return result;
  }

  // Returns true if and only if |value1| and |value2| are in the same
  // equivalence class.  Both values must already be known to the equivalence
  // relation.
  bool IsEquivalent(const T& value1, const T& value2) const {
    return Find(&value1) == Find(&value2);
  }

  // Returns all values known to be part of the equivalence relation.
  std::vector<const T*> GetAllKnownValues() const {
    std::vector<const T*> result;
    for (auto& value : owned_values_) {
      result.push_back(value.get());
    }
    return result;
  }

  // Returns true if and only if |value| is known to be part of the equivalence
  // relation.
  bool Exists(const T& value) const {
    return value_set_.find(&value) != value_set_.end();
  }

  // Returns the representative of the equivalence class of |value|, which must
  // already be known to the equivalence relation.  This is the 'Find' operation
  // in a classic union-find data structure.
  const T* Find(const T* value) const {
    assert(Exists(*value));

    // Get the canonical pointer to the value from the value pool.
    const T* known_value = *value_set_.find(value);
    assert(parent_[known_value] && "Every known value should have a parent.");

    // Compute the result by chasing parents until we find a value that is its
    // own parent.
    const T* result = known_value;
    while (parent_[result] != result) {
      result = parent_[result];
    }
    assert(result && "Representatives should never be null.");

    // At this point, |result| is the representative of the equivalence class.
    // Now perform the 'path compression' optimization by doing another pass up
    // the parent chain, setting the parent of each node to be the
    // representative, and rewriting children correspondingly.
    const T* current = known_value;
    while (parent_[current] != result) {
      const T* next = parent_[current];
      parent_[current] = result;
      children_[result].push_back(current);
      auto child_iterator =
          std::find(children_[next].begin(), children_[next].end(), current);
      assert(child_iterator != children_[next].end() &&
             "'next' is the parent of 'current', so 'current' should be a "
             "child of 'next'");
      children_[next].erase(child_iterator);
      current = next;
    }
    return result;
  }

 private:
  // Maps every value to a parent.  The representative of an equivalence class
  // is its own parent.  A value's representative can be found by walking its
  // chain of ancestors.
  //
  // Mutable because the intuitively const method, 'Find', performs path
  // compression.
  mutable std::unordered_map<const T*, const T*> parent_;

  // Stores the children of each value.  This allows the equivalence class of
  // a value to be calculated by traversing all descendents of the class's
  // representative.
  //
  // Mutable because the intuitively const method, 'Find', performs path
  // compression.
  mutable std::unordered_map<const T*, std::vector<const T*>> children_;

  // The values known to the equivalence relation are allocated in
  // |owned_values_|, and |value_pool_| provides (via |PointerHashT| and
  // |PointerEqualsT|) a means for mapping a value of interest to a pointer
  // into an equivalent value in |owned_values_|.
  std::unordered_set<const T*, PointerHashT, PointerEqualsT> value_set_;
  std::vector<std::unique_ptr<T>> owned_values_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_EQUIVALENCE_RELATION_H_
