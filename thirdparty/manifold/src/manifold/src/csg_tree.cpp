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

#include "csg_tree.h"

#include <algorithm>

#include "boolean3.h"
#include "impl.h"
#include "par.h"

namespace {
using namespace manifold;
struct Transform4x3 {
  const glm::mat4x3 transform;

  __host__ __device__ glm::vec3 operator()(glm::vec3 position) {
    return transform * glm::vec4(position, 1.0f);
  }
};

struct TransformNormals {
  const glm::mat3 transform;

  __host__ __device__ glm::vec3 operator()(glm::vec3 normal) {
    normal = glm::normalize(transform * normal);
    if (isnan(normal.x)) normal = glm::vec3(0.0f);
    return normal;
  }
};

struct UpdateTriBary {
  const int nextBary;

  __host__ __device__ BaryRef operator()(BaryRef ref) {
    for (int i : {0, 1, 2})
      if (ref.vertBary[i] >= 0) ref.vertBary[i] += nextBary;
    return ref;
  }
};

struct UpdateHalfedge {
  const int nextVert;
  const int nextEdge;
  const int nextFace;

  __host__ __device__ Halfedge operator()(Halfedge edge) {
    edge.startVert += nextVert;
    edge.endVert += nextVert;
    edge.pairedHalfedge += nextEdge;
    edge.face += nextFace;
    return edge;
  }
};

struct CheckOverlap {
  const Box *boxes;
  const size_t i;
  __host__ __device__ bool operator()(int j) {
    return boxes[i].DoesOverlap(boxes[j]);
  }
};
}  // namespace
namespace manifold {

std::shared_ptr<CsgNode> CsgNode::Translate(const glm::vec3 &t) const {
  glm::mat4x3 transform(1.0f);
  transform[3] += t;
  return Transform(transform);
}

std::shared_ptr<CsgNode> CsgNode::Scale(const glm::vec3 &v) const {
  glm::mat4x3 transform(1.0f);
  for (int i : {0, 1, 2}) transform[i] *= v;
  return Transform(transform);
}

std::shared_ptr<CsgNode> CsgNode::Rotate(float xDegrees, float yDegrees,
                                         float zDegrees) const {
  glm::mat3 rX(1.0f, 0.0f, 0.0f,                      //
               0.0f, cosd(xDegrees), sind(xDegrees),  //
               0.0f, -sind(xDegrees), cosd(xDegrees));
  glm::mat3 rY(cosd(yDegrees), 0.0f, -sind(yDegrees),  //
               0.0f, 1.0f, 0.0f,                       //
               sind(yDegrees), 0.0f, cosd(yDegrees));
  glm::mat3 rZ(cosd(zDegrees), sind(zDegrees), 0.0f,   //
               -sind(zDegrees), cosd(zDegrees), 0.0f,  //
               0.0f, 0.0f, 1.0f);
  glm::mat4x3 transform(rZ * rY * rX);
  return Transform(transform);
}

CsgLeafNode::CsgLeafNode() : pImpl_(std::make_shared<Manifold::Impl>()) {}

CsgLeafNode::CsgLeafNode(std::shared_ptr<const Manifold::Impl> pImpl_)
    : pImpl_(pImpl_) {}

CsgLeafNode::CsgLeafNode(std::shared_ptr<const Manifold::Impl> pImpl_,
                         glm::mat4x3 transform_)
    : pImpl_(pImpl_), transform_(transform_) {}

std::shared_ptr<const Manifold::Impl> CsgLeafNode::GetImpl() const {
  if (transform_ == glm::mat4x3(1.0f)) return pImpl_;
  pImpl_ =
      std::make_shared<const Manifold::Impl>(pImpl_->Transform(transform_));
  transform_ = glm::mat4x3(1.0f);
  return pImpl_;
}

glm::mat4x3 CsgLeafNode::GetTransform() const { return transform_; }

Box CsgLeafNode::GetBoundingBox() const {
  return pImpl_->bBox_.Transform(transform_);
}

std::shared_ptr<CsgLeafNode> CsgLeafNode::ToLeafNode() const {
  return std::make_shared<CsgLeafNode>(*this);
}

std::shared_ptr<CsgNode> CsgLeafNode::Transform(const glm::mat4x3 &m) const {
  return std::make_shared<CsgLeafNode>(pImpl_, m * glm::mat4(transform_));
}

CsgNodeType CsgLeafNode::GetNodeType() const { return CsgNodeType::LEAF; }

/**
 * Efficient union of a set of pairwise disjoint meshes.
 */
Manifold::Impl CsgLeafNode::Compose(
    const std::vector<std::shared_ptr<CsgLeafNode>> &nodes) {
  float precision = -1;
  int numVert = 0;
  int numEdge = 0;
  int numTri = 0;
  int numBary = 0;
  for (auto &node : nodes) {
    float nodeOldScale = node->pImpl_->bBox_.Scale();
    float nodeNewScale = node->GetBoundingBox().Scale();
    float nodePrecision = node->pImpl_->precision_;
    nodePrecision *= glm::max(1.0f, nodeNewScale / nodeOldScale);
    nodePrecision = glm::max(nodePrecision, kTolerance * nodeNewScale);
    if (!glm::isfinite(nodePrecision)) nodePrecision = -1;
    precision = glm::max(precision, nodePrecision);

    numVert += node->pImpl_->NumVert();
    numEdge += node->pImpl_->NumEdge();
    numTri += node->pImpl_->NumTri();
    numBary += node->pImpl_->meshRelation_.barycentric.size();
  }

  Manifold::Impl combined;
  combined.precision_ = precision;
  combined.vertPos_.resize(numVert);
  combined.halfedge_.resize(2 * numEdge);
  combined.faceNormal_.resize(numTri);
  combined.halfedgeTangent_.resize(2 * numEdge);
  combined.meshRelation_.barycentric.resize(numBary);
  combined.meshRelation_.triBary.resize(numTri);
  auto policy = autoPolicy(numTri);

  int nextVert = 0;
  int nextEdge = 0;
  int nextTri = 0;
  int nextBary = 0;
  for (auto &node : nodes) {
    if (node->transform_ == glm::mat4x3(1.0f)) {
      copy(policy, node->pImpl_->vertPos_.begin(), node->pImpl_->vertPos_.end(),
           combined.vertPos_.begin() + nextVert);
      copy(policy, node->pImpl_->faceNormal_.begin(),
           node->pImpl_->faceNormal_.end(),
           combined.faceNormal_.begin() + nextTri);
    } else {
      // no need to apply the transform to the node, just copy the vertices and
      // face normals and apply transform on the fly
      auto vertPosBegin = thrust::make_transform_iterator(
          node->pImpl_->vertPos_.begin(), Transform4x3({node->transform_}));
      glm::mat3 normalTransform =
          glm::inverse(glm::transpose(glm::mat3(node->transform_)));
      auto faceNormalBegin =
          thrust::make_transform_iterator(node->pImpl_->faceNormal_.begin(),
                                          TransformNormals({normalTransform}));
      copy_n(policy, vertPosBegin, node->pImpl_->vertPos_.size(),
             combined.vertPos_.begin() + nextVert);
      copy_n(policy, faceNormalBegin, node->pImpl_->faceNormal_.size(),
             combined.faceNormal_.begin() + nextTri);
    }
    copy(policy, node->pImpl_->halfedgeTangent_.begin(),
         node->pImpl_->halfedgeTangent_.end(),
         combined.halfedgeTangent_.begin() + nextEdge);
    copy(policy, node->pImpl_->meshRelation_.barycentric.begin(),
         node->pImpl_->meshRelation_.barycentric.end(),
         combined.meshRelation_.barycentric.begin() + nextBary);
    transform(policy, node->pImpl_->meshRelation_.triBary.begin(),
              node->pImpl_->meshRelation_.triBary.end(),
              combined.meshRelation_.triBary.begin() + nextTri,
              UpdateTriBary({nextBary}));
    transform(policy, node->pImpl_->halfedge_.begin(),
              node->pImpl_->halfedge_.end(),
              combined.halfedge_.begin() + nextEdge,
              UpdateHalfedge({nextVert, nextEdge, nextTri}));

    // Since the nodes may be copies containing the same meshIDs, it is
    // important to increment them separately so that each node instance gets
    // unique meshIDs.
    combined.IncrementMeshIDs(nextTri, node->pImpl_->NumTri());

    nextVert += node->pImpl_->NumVert();
    nextEdge += 2 * node->pImpl_->NumEdge();
    nextTri += node->pImpl_->NumTri();
    nextBary += node->pImpl_->meshRelation_.barycentric.size();
  }
  // required to remove parts that are smaller than the precision
  combined.SimplifyTopology();
  combined.Finish();
  return combined;
}

CsgOpNode::CsgOpNode() {}

CsgOpNode::CsgOpNode(const std::vector<std::shared_ptr<CsgNode>> &children,
                     Manifold::OpType op)
    : children_(children) {
  SetOp(op);
  // opportunisticly flatten the tree without costly evaluation
  GetChildren(false);
}

CsgOpNode::CsgOpNode(std::vector<std::shared_ptr<CsgNode>> &&children,
                     Manifold::OpType op)
    : children_(children) {
  SetOp(op);
  // opportunisticly flatten the tree without costly evaluation
  GetChildren(false);
}

std::shared_ptr<CsgNode> CsgOpNode::Transform(const glm::mat4x3 &m) const {
  auto node = std::make_shared<CsgOpNode>();
  node->children_ = children_;
  node->op_ = op_;
  node->transform_ = m * glm::mat4(transform_);
  node->simplified_ = simplified_;
  node->flattened_ = flattened_;
  return node;
}

std::shared_ptr<CsgLeafNode> CsgOpNode::ToLeafNode() const {
  if (cache_ != nullptr) return cache_;
  if (children_.empty()) return nullptr;
  // turn the children into leaf nodes
  GetChildren();
  switch (op_) {
    case CsgNodeType::UNION:
      BatchUnion();
      break;
    case CsgNodeType::INTERSECTION: {
      std::vector<std::shared_ptr<const Manifold::Impl>> impls;
      for (auto &child : children_) {
        impls.push_back(
            std::dynamic_pointer_cast<CsgLeafNode>(child)->GetImpl());
      }
      BatchBoolean(Manifold::OpType::INTERSECT, impls);
      children_.clear();
      children_.push_back(std::make_shared<CsgLeafNode>(impls.front()));
      break;
    };
    case CsgNodeType::DIFFERENCE: {
      // take the lhs out and treat the remaining nodes as the rhs, perform
      // union optimization for them
      auto lhs = std::dynamic_pointer_cast<CsgLeafNode>(children_.front());
      children_.erase(children_.begin());
      BatchUnion();
      auto rhs = std::dynamic_pointer_cast<CsgLeafNode>(children_.front());
      children_.clear();
      Boolean3 boolean(*lhs->GetImpl(), *rhs->GetImpl(),
                       Manifold::OpType::SUBTRACT);
      children_.push_back(
          std::make_shared<CsgLeafNode>(std::make_shared<Manifold::Impl>(
              boolean.Result(Manifold::OpType::SUBTRACT))));
    };
    case CsgNodeType::LEAF:
      // unreachable
      break;
  }
  // children_ must contain only one CsgLeafNode now, and its Transform will
  // give CsgLeafNode as well
  cache_ = std::dynamic_pointer_cast<CsgLeafNode>(
      children_.front()->Transform(transform_));
  return cache_;
}

/**
 * Efficient boolean operation on a set of nodes utilizing commutativity of the
 * operation. Only supports union and intersection.
 */
void CsgOpNode::BatchBoolean(
    Manifold::OpType operation,
    std::vector<std::shared_ptr<const Manifold::Impl>> &results) {
  ASSERT(operation != Manifold::OpType::SUBTRACT, logicErr,
         "BatchBoolean doesn't support Difference.");
  auto cmpFn = [](std::shared_ptr<const Manifold::Impl> a,
                  std::shared_ptr<const Manifold::Impl> b) {
    // invert the order because we want a min heap
    return a->NumVert() > b->NumVert();
  };

  // apply boolean operations starting from smaller meshes
  // the assumption is that boolean operations on smaller meshes is faster,
  // due to less data being copied and processed
  std::make_heap(results.begin(), results.end(), cmpFn);
  while (results.size() > 1) {
    std::pop_heap(results.begin(), results.end(), cmpFn);
    auto a = std::move(results.back());
    results.pop_back();
    std::pop_heap(results.begin(), results.end(), cmpFn);
    auto b = std::move(results.back());
    results.pop_back();
    // boolean operation
    Boolean3 boolean(*a, *b, operation);
    results.push_back(
        std::make_shared<const Manifold::Impl>(boolean.Result(operation)));
    std::push_heap(results.begin(), results.end(), cmpFn);
  }
}

/**
 * Efficient union operation on a set of nodes by doing Compose as much as
 * possible.
 * Note: Due to some unknown issues with `Compose`, we are now doing
 * `BatchBoolean` instead of using `Compose` for non-intersecting manifolds.
 */
void CsgOpNode::BatchUnion() const {
  std::vector<std::shared_ptr<const Manifold::Impl>> impls;
  for (auto &child : children_) {
    impls.push_back(std::dynamic_pointer_cast<CsgLeafNode>(child)->GetImpl());
  }
  BatchBoolean(Manifold::OpType::ADD, impls);
  children_.clear();
  children_.push_back(std::make_shared<CsgLeafNode>(impls.front()));
}

/**
 * Flatten the children to a list of leaf nodes and return them.
 * If finalize is true, the list will be guaranteed to be a list of leaf nodes
 * (i.e. no ops). Otherwise, the list may contain ops.
 * Note that this function will not apply the transform to children, as they may
 * be shared with other nodes.
 */
std::vector<std::shared_ptr<CsgNode>> &CsgOpNode::GetChildren(
    bool finalize) const {
  if (children_.empty() || (simplified_ && !finalize) || flattened_)
    return children_;
  simplified_ = true;
  flattened_ = finalize;
  std::vector<std::shared_ptr<CsgNode>> newChildren;

  CsgNodeType op = op_;
  for (auto &child : children_) {
    if (child->GetNodeType() == op) {
      auto grandchildren =
          std::dynamic_pointer_cast<CsgOpNode>(child)->GetChildren(finalize);
      int start = children_.size();
      for (auto &grandchild : grandchildren) {
        newChildren.push_back(grandchild->Transform(child->GetTransform()));
      }
    } else {
      if (!finalize || child->GetNodeType() == CsgNodeType::LEAF) {
        newChildren.push_back(child);
      } else {
        newChildren.push_back(child->ToLeafNode());
      }
    }
    // special handling for difference: we treat it as first - (second + third +
    // ...) so op = UNION after the first node
    if (op == CsgNodeType::DIFFERENCE) op = CsgNodeType::UNION;
  }
  children_ = newChildren;
  return children_;
}

void CsgOpNode::SetOp(Manifold::OpType op) {
  switch (op) {
    case Manifold::OpType::ADD:
      op_ = CsgNodeType::UNION;
      break;
    case Manifold::OpType::SUBTRACT:
      op_ = CsgNodeType::DIFFERENCE;
      break;
    case Manifold::OpType::INTERSECT:
      op_ = CsgNodeType::INTERSECTION;
      break;
  }
}

glm::mat4x3 CsgOpNode::GetTransform() const { return transform_; }

}  // namespace manifold
