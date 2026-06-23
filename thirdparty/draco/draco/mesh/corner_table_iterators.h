// Copyright 2016 The Draco Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#ifndef DRACO_MESH_CORNER_TABLE_ITERATORS_H_
#define DRACO_MESH_CORNER_TABLE_ITERATORS_H_

#include <iterator>

#include "draco/mesh/corner_table.h"

namespace draco {

// Class for iterating over vertices in a 1-ring around the specified vertex.
template <class CornerTableT>
class VertexRingIterator {
 public:
  // Iterator traits expected by std libraries.
  using iterator_category = std::forward_iterator_tag;
  using value_type = VertexIndex;
  using difference_type = std::ptrdiff_t;
  using pointer = VertexIndex *;
  using reference = VertexIndex &;

  // std::iterator interface requires a default constructor.
  VertexRingIterator()
      : corner_table_(nullptr),
        start_corner_(kInvalidCornerIndex),
        corner_(start_corner_),
        left_traversal_(true) {}

  // Create the iterator from the provided corner table and the central vertex.
  VertexRingIterator(const CornerTableT *table, VertexIndex vert_id)
      : corner_table_(table),
        start_corner_(table->LeftMostCorner(vert_id)),
        corner_(start_corner_),
        left_traversal_(true) {}

  // Gets the last visited ring vertex.
  VertexIndex Vertex() const {
    CornerIndex ring_corner = left_traversal_ ? corner_table_->Previous(corner_)
                                              : corner_table_->Next(corner_);
    return corner_table_->Vertex(ring_corner);
  }

  // Returns one of the corners opposite to the edge connecting the currently
  // iterated ring vertex with the central vertex.
  CornerIndex EdgeCorner() const {
    return left_traversal_ ? corner_table_->Next(corner_)
                           : corner_table_->Previous(corner_);
  }

  // Returns true when all ring vertices have been visited.
  bool End() const { return corner_ == kInvalidCornerIndex; }

  // Proceeds to the next ring vertex if possible.
  void Next() {
    if (left_traversal_) {
      corner_ = corner_table_->SwingLeft(corner_);
      if (corner_ == kInvalidCornerIndex) {
        // Open boundary reached.
        corner_ = start_corner_;
        left_traversal_ = false;
      } else if (corner_ == start_corner_) {
        // End reached.
        corner_ = kInvalidCornerIndex;
      }
    } else {
      // Go to the right until we reach a boundary there (no explicit check
      // is needed in this case).
      corner_ = corner_table_->SwingRight(corner_);
    }
  }

  // std::iterator interface.
  value_type operator*() const { return Vertex(); }
  VertexRingIterator &operator++() {
    Next();
    return *this;
  }
  VertexRingIterator operator++(int) {
    const VertexRingIterator result = *this;
    ++(*this);
    return result;
  }
  bool operator!=(const VertexRingIterator &other) const {
    return corner_ != other.corner_ || start_corner_ != other.start_corner_;
  }
  bool operator==(const VertexRingIterator &other) const {
    return !this->operator!=(other);
  }

  // Helper function for getting a valid end iterator.
  static VertexRingIterator EndIterator(VertexRingIterator other) {
    VertexRingIterator ret = other;
    ret.corner_ = kInvalidCornerIndex;
    return ret;
  }

 private:
  const CornerTableT *corner_table_;
  // The first processed corner.
  CornerIndex start_corner_;
  // The last processed corner.
  CornerIndex corner_;
  // Traversal direction.
  bool left_traversal_;
};

// Class for iterating over faces adjacent to the specified input face.
template <class CornerTableT>
class FaceAdjacencyIterator {
 public:
  // Iterator traits expected by std libraries.
  using iterator_category = std::forward_iterator_tag;
  using value_type = FaceIndex;
  using difference_type = std::ptrdiff_t;
  using pointer = FaceIndex *;
  using reference = FaceIndex &;

  // std::iterator interface requires a default constructor.
  FaceAdjacencyIterator()
      : corner_table_(nullptr),
        start_corner_(kInvalidCornerIndex),
        corner_(start_corner_) {}

  // Create the iterator from the provided corner table and the central vertex.
  FaceAdjacencyIterator(const CornerTableT *table, FaceIndex face_id)
      : corner_table_(table),
        start_corner_(table->FirstCorner(face_id)),
        corner_(start_corner_) {
    // We need to start with a corner that has a valid opposite face (if
    // there is any such corner).
    if (corner_table_->Opposite(corner_) == kInvalidCornerIndex) {
      FindNextFaceNeighbor();
    }
  }

  // Gets the last visited adjacent face.
  FaceIndex Face() const {
    return corner_table_->Face(corner_table_->Opposite(corner_));
  }

  // Returns true when all adjacent faces have been visited.
  bool End() const { return corner_ == kInvalidCornerIndex; }

  // Proceeds to the next adjacent face if possible.
  void Next() { FindNextFaceNeighbor(); }

  // std::iterator interface.
  value_type operator*() const { return Face(); }
  FaceAdjacencyIterator &operator++() {
    Next();
    return *this;
  }
  FaceAdjacencyIterator operator++(int) {
    const FaceAdjacencyIterator result = *this;
    ++(*this);
    return result;
  }
  bool operator!=(const FaceAdjacencyIterator &other) const {
    return corner_ != other.corner_ || start_corner_ != other.start_corner_;
  }
  bool operator==(const FaceAdjacencyIterator &other) const {
    return !this->operator!=(other);
  }

  // Helper function for getting a valid end iterator.
  static FaceAdjacencyIterator EndIterator(FaceAdjacencyIterator other) {
    FaceAdjacencyIterator ret = other;
    ret.corner_ = kInvalidCornerIndex;
    return ret;
  }

 private:
  // Finds the next corner with a valid opposite face.
  void FindNextFaceNeighbor() {
    while (corner_ != kInvalidCornerIndex) {
      corner_ = corner_table_->Next(corner_);
      if (corner_ == start_corner_) {
        corner_ = kInvalidCornerIndex;
        return;
      }
      if (corner_table_->Opposite(corner_) != kInvalidCornerIndex) {
        // Valid opposite face.
        return;
      }
    }
  }

  const CornerTableT *corner_table_;
  // The first processed corner.
  CornerIndex start_corner_;
  // The last processed corner.
  CornerIndex corner_;
};

// Class for iterating over corners attached to a specified vertex.
template <class CornerTableT = CornerTable>
class VertexCornersIterator {
 public:
  // Iterator traits expected by std libraries.
  using iterator_category = std::forward_iterator_tag;
  using value_type = CornerIndex;
  using difference_type = std::ptrdiff_t;
  using pointer = CornerIndex *;
  using reference = CornerIndex &;

  // std::iterator interface requires a default constructor.
  VertexCornersIterator()
      : corner_table_(nullptr),
        start_corner_(-1),
        corner_(start_corner_),
        left_traversal_(true) {}

  // Create the iterator from the provided corner table and the central vertex.
  VertexCornersIterator(const CornerTableT *table, VertexIndex vert_id)
      : corner_table_(table),
        start_corner_(table->LeftMostCorner(vert_id)),
        corner_(start_corner_),
        left_traversal_(true) {}

  // Create the iterator from the provided corner table and the first corner.
  VertexCornersIterator(const CornerTableT *table, CornerIndex corner_id)
      : corner_table_(table),
        start_corner_(corner_id),
        corner_(start_corner_),
        left_traversal_(true) {}

  // Gets the last visited corner.
  CornerIndex Corner() const { return corner_; }

  // Returns true when all ring vertices have been visited.
  bool End() const { return corner_ == kInvalidCornerIndex; }

  // Proceeds to the next corner if possible.
  void Next() {
    if (left_traversal_) {
      corner_ = corner_table_->SwingLeft(corner_);
      if (corner_ == kInvalidCornerIndex) {
        // Open boundary reached.
        corner_ = corner_table_->SwingRight(start_corner_);
        left_traversal_ = false;
      } else if (corner_ == start_corner_) {
        // End reached.
        corner_ = kInvalidCornerIndex;
      }
    } else {
      // Go to the right until we reach a boundary there (no explicit check
      // is needed in this case).
      corner_ = corner_table_->SwingRight(corner_);
    }
  }

  // std::iterator interface.
  CornerIndex operator*() const { return Corner(); }
  VertexCornersIterator &operator++() {
    Next();
    return *this;
  }
  VertexCornersIterator operator++(int) {
    const VertexCornersIterator result = *this;
    ++(*this);
    return result;
  }
  bool operator!=(const VertexCornersIterator &other) const {
    return corner_ != other.corner_ || start_corner_ != other.start_corner_;
  }
  bool operator==(const VertexCornersIterator &other) const {
    return !this->operator!=(other);
  }

  // Helper function for getting a valid end iterator.
  static VertexCornersIterator EndIterator(VertexCornersIterator other) {
    VertexCornersIterator ret = other;
    ret.corner_ = kInvalidCornerIndex;
    return ret;
  }

 protected:
  const CornerTableT *corner_table() const { return corner_table_; }
  CornerIndex start_corner() const { return start_corner_; }
  CornerIndex &corner() { return corner_; }
  bool is_left_traversal() const { return left_traversal_; }
  void swap_traversal() { left_traversal_ = !left_traversal_; }

 private:
  const CornerTableT *corner_table_;
  // The first processed corner.
  CornerIndex start_corner_;
  // The last processed corner.
  CornerIndex corner_;
  // Traversal direction.
  bool left_traversal_;
};

}  // namespace draco

#endif  // DRACO_MESH_CORNER_TABLE_ITERATORS_H_
