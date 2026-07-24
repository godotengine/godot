// Copyright 2024 The Manifold Authors.
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

#include "execution_impl.h"
#include "impl.h"
#include "parallel.h"

namespace manifold {

/**
 * Compute the minkowski sum of two manifolds.
 *
 * @param other The other Impl to minkowski sum/diff with this one.
 * @param inset Whether it should subtract (erode) rather than add (dilate).
 */
Manifold Manifold::Impl::Minkowski(const Impl& other, bool inset,
                                   ExecutionContext::Impl* ctx) const {
  // Helper to short-circuit on cancel. Cancellation is observed both at
  // the boundaries between hull/Boolean phases and inside each
  // BatchBoolean (via the same ctx threaded into GetCsgLeafNode, which
  // routes into Boolean3's existing cancel checks). The per-hull work
  // itself is currently not interrupted -- granularity inside a batch
  // is "one Manifold::Hull(simpleHull) call."
  auto cancelled = [] {
    auto impl = std::make_shared<Impl>();
    impl->status_ = Error::Cancelled;
    return Manifold(impl);
  };
  // Build a BatchBoolean and force ctx-observed evaluation so an
  // attached cancel fires inside Boolean3 rather than after the whole
  // batch completes.
  auto evalBatch = [ctx](std::vector<Manifold>&& items, OpType op) {
    Manifold tree = Manifold::BatchBoolean(items, op);
    // Trigger lazy eval with ctx. The returned leaf may have
    // status_=Cancelled if cancel fired mid-Boolean; the caller still
    // pushes it into composedHulls and the outer cancel check catches
    // it.
    tree.GetCsgLeafNode(ctx);
    return tree;
  };
  if (IsCancelled(ctx)) return cancelled();

  const Impl* aImpl = this;
  const Impl* bImpl = &other;

  bool aConvex = aImpl->IsConvex();
  bool bConvex = bImpl->IsConvex();

  constexpr size_t BATCH_SIZE = 1000;

  // If the convex manifold was supplied first, swap them!
  if (aConvex && !bConvex) {
    std::swap(aImpl, bImpl);
    std::swap(aConvex, bConvex);
  }

  // Early-exit if either input is empty
  if (bImpl->IsEmpty()) {
    std::shared_ptr<Impl> result = std::make_shared<Impl>(*aImpl);
    return Manifold(result);
  }
  if (aImpl->IsEmpty()) {
    std::shared_ptr<Impl> result = std::make_shared<Impl>(*bImpl);
    return Manifold(result);
  }

  std::shared_ptr<Impl> aImplCopy = std::make_shared<Impl>(*aImpl);
  Manifold a(aImplCopy);
  std::vector<Manifold> composedHulls;
  // Reserve space: 1 for base + batches for non-convex cases
  size_t numBatches = (aImpl->NumTri() + BATCH_SIZE - 1) / BATCH_SIZE;
  composedHulls.reserve(1 + numBatches);
  composedHulls.push_back(a);

  // Convex-Convex Minkowski: Very Fast
  if (!inset && aConvex && bConvex) {
    const VecView<vec3> verts = bImpl->vertPos_;
    std::vector<vec3> simpleHull;
    simpleHull.reserve(verts.size() * aImpl->vertPos_.size());
    for (const vec3& vertex : aImpl->vertPos_) {
      auto t = [vertex](vec3 v) { return v + vertex; };
      simpleHull.insert(simpleHull.end(), TransformIterator(verts.begin(), t),
                        TransformIterator(verts.end(), t));
    }
    composedHulls.push_back(Manifold::Hull(simpleHull));
    // Convex - Non-Convex Minkowski: Slower
  } else if ((inset || !aConvex) && bConvex) {
    const size_t numTri = aImpl->NumTri();
    const VecView<vec3> verts = bImpl->vertPos_;

    // do it in batches of 1000 meshes
    for (size_t offset = 0; offset < numTri; offset += BATCH_SIZE) {
      if (IsCancelled(ctx)) return cancelled();
      size_t numIter = std::min(numTri - offset, BATCH_SIZE);
      std::vector<Manifold> newHulls(numIter);
      for_each_n(autoPolicy(numIter, 100), countAt(0), numIter,
                 [&newHulls, &aImpl, &verts, offset](const int iter) {
                   std::vector<vec3> simpleHull;
                   for (int i : {0, 1, 2}) {
                     const int edge = ((offset + iter) * 3) + i;
                     const auto vertex =
                         aImpl->vertPos_[aImpl->halfedge_.Start(edge)];
                     auto t = [vertex](vec3 v) { return v + vertex; };
                     simpleHull.insert(simpleHull.end(),
                                       TransformIterator(verts.begin(), t),
                                       TransformIterator(verts.end(), t));
                   }
                   newHulls[iter] = Manifold::Hull(simpleHull);
                 });
      composedHulls.push_back(evalBatch(std::move(newHulls), OpType::Add));
    }
    // Non-Convex - Non-Convex Minkowski: Very Slow
    // Process A faces sequentially with periodic batch reduction to balance
    // memory usage and performance.
  } else if (!aConvex && !bConvex) {
    const size_t numTriA = aImpl->NumTri();
    const size_t numTriB = bImpl->NumTri();

    // Reduce accumulated results after this many A faces to limit memory
    constexpr size_t REDUCE_THRESHOLD = 200;

    // Accumulated per-A-face results (periodically reduced)
    std::vector<Manifold> accumulated;
    accumulated.reserve(std::min(numTriA, REDUCE_THRESHOLD));

    // Process each A face sequentially
    for (size_t aFace = 0; aFace < numTriA; ++aFace) {
      if (IsCancelled(ctx)) return cancelled();
      vec3 a1 = aImpl->vertPos_[aImpl->halfedge_.Start((aFace * 3) + 0)];
      vec3 a2 = aImpl->vertPos_[aImpl->halfedge_.Start((aFace * 3) + 1)];
      vec3 a3 = aImpl->vertPos_[aImpl->halfedge_.Start((aFace * 3) + 2)];
      vec3 nA = aImpl->faceNormal_[aFace];

      // Create hulls for all B faces paired with this A face (parallel)
      std::vector<Manifold> faceHulls(numTriB);

      for_each_n(
          autoPolicy(numTriB, 100), countAt(0), numTriB, [&](const int bFace) {
            // Tolerance for detecting coplanar faces (skip degenerate hull
            // cases)
            constexpr double kCoplanarTol = 1e-12;
            vec3 nB = bImpl->faceNormal_[bFace];
            double dotSame = linalg::dot(nA, nB);
            double dotOpp = linalg::dot(nA, -nB);
            const bool coplanar = (std::abs(dotSame - 1.0) < kCoplanarTol) ||
                                  (std::abs(dotOpp - 1.0) < kCoplanarTol);
            if (coplanar) return;

            vec3 b1 = bImpl->vertPos_[bImpl->halfedge_.Start((bFace * 3) + 0)];
            vec3 b2 = bImpl->vertPos_[bImpl->halfedge_.Start((bFace * 3) + 1)];
            vec3 b3 = bImpl->vertPos_[bImpl->halfedge_.Start((bFace * 3) + 2)];
            faceHulls[bFace] =
                Manifold::Hull({a1 + b1, a1 + b2, a1 + b3, a2 + b1, a2 + b2,
                                a2 + b3, a3 + b1, a3 + b2, a3 + b3});
          });

      // Collect non-empty hulls for this A face
      std::vector<Manifold> validFaceHulls;
      for (auto& hull : faceHulls) {
        if (!hull.IsEmpty()) {
          validFaceHulls.push_back(std::move(hull));
        }
      }

      if (!validFaceHulls.empty()) {
        accumulated.push_back(
            evalBatch(std::move(validFaceHulls), OpType::Add));
      }

      // Periodically reduce to limit memory usage
      if (accumulated.size() >= REDUCE_THRESHOLD) {
        Manifold reduced = evalBatch(std::move(accumulated), OpType::Add);
        accumulated.clear();
        accumulated.push_back(std::move(reduced));
      }
    }

    // Final merge of remaining accumulated results
    if (!accumulated.empty()) {
      composedHulls.push_back(evalBatch(std::move(accumulated), OpType::Add));
    }
  }
  if (IsCancelled(ctx)) return cancelled();
  return evalBatch(std::move(composedHulls),
                   inset ? manifold::OpType::Subtract : manifold::OpType::Add)
      .AsOriginal();
}

}  // namespace manifold
