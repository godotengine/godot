// Copyright 2022 The Manifold Authors.
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

#pragma once
#include <mutex>

#include "manifold/manifold.h"
#include "utils.h"

namespace manifold {

enum class CsgNodeType { Union, Intersection, Difference, Leaf };

class CsgLeafNode;

class CsgNode : public std::enable_shared_from_this<CsgNode> {
 public:
  virtual std::shared_ptr<CsgLeafNode> ToLeafNode(
      ExecutionContext::Impl* ctx = nullptr) const = 0;
  virtual std::shared_ptr<CsgNode> Transform(const mat3x4& m) const = 0;
  virtual CsgNodeType GetNodeType() const = 0;
  /// Count the leaves in the subtree rooted at this node. A CsgOpNode with
  /// a cached evaluation counts as a single leaf. Used for pre-pass progress
  /// denominator: a CSG tree with N leaves reduces to 1 result in N-1 ops.
  virtual size_t NumLeaves() const = 0;

  virtual std::shared_ptr<CsgNode> Boolean(
      const std::shared_ptr<CsgNode>& second, OpType op);

  std::shared_ptr<CsgNode> Translate(const vec3& t) const;
  std::shared_ptr<CsgNode> Scale(const vec3& s) const;
  std::shared_ptr<CsgNode> Rotate(double xDegrees = 0, double yDegrees = 0,
                                  double zDegrees = 0) const;
};

class CsgLeafNode final : public CsgNode {
 public:
  CsgLeafNode();
  CsgLeafNode(std::shared_ptr<const Manifold::Impl> pImpl_);
  CsgLeafNode(std::shared_ptr<const Manifold::Impl> pImpl_, mat3x4 transform_);
  // mutex_ is not copyable; snapshot the source under its lock.
  CsgLeafNode(const CsgLeafNode& other);

  std::shared_ptr<const Manifold::Impl> GetImpl() const;

  std::shared_ptr<CsgLeafNode> ToLeafNode(
      ExecutionContext::Impl* ctx = nullptr) const override;

  std::shared_ptr<CsgNode> Transform(const mat3x4& m) const override;

  CsgNodeType GetNodeType() const override;

  size_t NumLeaves() const override { return 1; }

  static std::shared_ptr<CsgLeafNode> Compose(
      const std::vector<std::shared_ptr<CsgLeafNode>>& nodes);

  // Returns the bounding box without triggering eager transform application.
  // This is more efficient for overlap checks when we don't need the full mesh.
  Box GetBoundingBox() const;

  // Returns the number of vertices without triggering eager transform.
  size_t NumVert() const;

 private:
  mutable std::shared_ptr<const Manifold::Impl> pImpl_;
  mutable mat3x4 transform_ = la::identity;
  // Lazy realization mutates these through const accessors on a shared node,
  // so every access takes mutex_.
  mutable std::mutex mutex_;
};

class CsgOpNode final : public CsgNode {
 public:
  CsgOpNode();

  CsgOpNode(const std::vector<std::shared_ptr<CsgNode>>& children, OpType op);

  std::shared_ptr<CsgNode> Boolean(const std::shared_ptr<CsgNode>& second,
                                   OpType op) override;

  std::shared_ptr<CsgNode> Transform(const mat3x4& m) const override;

  std::shared_ptr<CsgLeafNode> ToLeafNode(
      ExecutionContext::Impl* ctx = nullptr) const override;

  CsgNodeType GetNodeType() const override;

  size_t NumLeaves() const override;

  ~CsgOpNode();

 private:
  mutable ConcurrentSharedPtr<std::vector<std::shared_ptr<CsgNode>>> impl_ =
      ConcurrentSharedPtr<std::vector<std::shared_ptr<CsgNode>>>({});
  OpType op_;
  mat3x4 transform_ = la::identity;
  // the following fields are for lazy evaluation, so they are mutable
  mutable std::shared_ptr<CsgLeafNode> cache_ = nullptr;
};

}  // namespace manifold
