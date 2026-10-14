// Copyright 2026 The Manifold Authors.
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
#include <atomic>
#if defined(MANIFOLD_DEBUG) || defined(MANIFOLD_TIMING)
#include <chrono>
#endif

#include "manifold/common.h"

namespace manifold {

inline bool IsCancelled(ExecutionContext::Impl* ctx);

/** @ingroup Private
 *
 * Number of progress phases per `Boolean3::Result`. Must equal the count of
 * `phase()` sites in `boolean_result.cpp`'s `Boolean3::Result` (asserted on
 * the happy-path return). Bump in lockstep when adding/removing a site.
 */
constexpr int kPhasesPerBoolean = 11;

/** @ingroup Private
 *
 * Heavy phases in the `Manifold::Impl::Impl(MeshGLP, ctx)` ingest;
 * credit is published only after a phase completes successfully. Bump
 * in lockstep with the `donePhases.fetch_add(1, ...)` sites in the
 * ctor body.
 */
constexpr int kPhasesPerFromMesh = 7;

/** @ingroup Private
 *
 * Heavy phases in `ExecutionContext::Smooth(MeshGL[64])`: the
 * `kPhasesPerFromMesh` ingest phases followed by 7 tangent-creation
 * phases in `Manifold::Impl::CreateTangents(sharpenedEdges, ctx)`.
 * Bump in lockstep with `ADVANCE_PHASE_OR_RETURN(ctx)` sites in that
 * function.
 */
constexpr int kPhasesPerSmooth = kPhasesPerFromMesh + 7;

/** @ingroup Private
 *
 * Heavy phases in `Manifold::Impl::CreateLevelSet(...)`: the four grid
 * loops (voxel SDF sampling, NearSurface, ComputeVerts, BuildTris)
 * followed by one lumped finalize phase (CreateHalfedges through
 * SetNormalsAndCoplanar). The NearSurface hash-table resize loop may
 * re-run NearSurface, but counts as a single phase regardless. Bump in
 * lockstep with `ADVANCE_PHASE_OR_RETURN(ctx)` sites in CreateLevelSet.
 */
constexpr int kPhasesPerLevelSet = 5;

/** @ingroup Private
 *
 * Pimpl for ExecutionContext. `cancel` is private; use `IsCancelled(ctx)`
 * to read it -- this is the canonical reader, enforced by the type system.
 * `totalBooleans`, `doneBooleans`, `totalPhases`, and `donePhases` are
 * public for progress reporting and test introspection.
 *
 * `Progress() = donePhases / totalPhases`. `totalPhases` is the canonical
 * progress denominator across op types; for Boolean trees it is set at
 * `GetCsgLeafNode` to `totalBooleans * kPhasesPerBoolean`. Future
 * non-Boolean ops (Hull, LevelSet, Refine, ...) can add to `totalPhases`
 * and increment `donePhases` independently using the same mechanism.
 *
 * `totalBooleans`/`doneBooleans` are kept as introspection counters for
 * tests and tools; they are not used in `Progress()`.
 */
struct ExecutionContext::Impl {
 public:
  std::atomic<int> totalBooleans{0};
  std::atomic<int> doneBooleans{0};
  std::atomic<int> totalPhases{0};
  std::atomic<int> donePhases{0};
#if defined(MANIFOLD_DEBUG) || defined(MANIFOLD_TIMING)
  // Wall-clock of the last phase boundary, for per-phase timing via
  // ADVANCE_PHASE_OR_RETURN. Stamped at the static-factory reset; touched only
  // by the op's own thread.
  std::chrono::high_resolution_clock::time_point lastPhase{};
#endif

 private:
  std::atomic<bool> cancel{false};

  friend bool IsCancelled(Impl*);
  friend class manifold::ExecutionContext;
};

/** @ingroup Private
 *
 * Canonical reader for the cancel flag. The only path to observe cancel
 * state from internal code -- `Impl::cancel` is private and friended only
 * to this function and `ExecutionContext` (for its public `Cancel`/`Cancelled`
 * members). The ctx-aware overloads in `parallel.h` go through here.
 *
 * Returns false if `ctx` is nullptr (no-cancellation calls), otherwise
 * loads `cancel` with `memory_order_relaxed` (cancel is advisory; we
 * don't need synchronization with other operations).
 */
inline bool IsCancelled(ExecutionContext::Impl* ctx) {
  return ctx && ctx->cancel.load(std::memory_order_relaxed);
}

#if defined(MANIFOLD_DEBUG) || defined(MANIFOLD_TIMING)
// Times the span since the previous phase boundary (or the static-factory
// reset) and prints it when verbose >= 2. Called by ADVANCE_PHASE_OR_RETURN;
// defined in execution_impl.cpp so <chrono>/<iostream> stay out of this
// widely-included header.
void RecordPhase(ExecutionContext::Impl* ctx, const char* file, int line);

// Per-op local phase-timing state. Boolean3::Result keeps this on its own
// stack rather than on the shared ctx, so concurrent booleans in a CSG tree
// don't clobber each other's baseline. `uid` tags the printed lines so the
// interleaved output of parallel booleans can be split apart.
struct LocalPhaseTiming {
  std::chrono::high_resolution_clock::time_point last;
  int uid;
};
LocalPhaseTiming BeginLocalPhaseTiming();
void RecordPhase(LocalPhaseTiming& timing, int phase, const char* file,
                 int line);
#endif

}  // namespace manifold

/** @ingroup Private
 *
 * Phase-boundary checkpoint for `Manifold::Impl` methods (relies on
 * `this->MakeEmpty`). Call count must equal the method's `kPhasesPer*`.
 */
#if defined(MANIFOLD_DEBUG) || defined(MANIFOLD_TIMING)
#define MANIFOLD_RECORD_PHASE(ctx) \
  manifold::RecordPhase((ctx), __FILE__, __LINE__)
#else
#define MANIFOLD_RECORD_PHASE(ctx) ((void)0)
#endif

#define ADVANCE_PHASE_OR_RETURN(ctx)                             \
  do {                                                           \
    if (manifold::IsCancelled(ctx)) {                            \
      MakeEmpty(manifold::Manifold::Error::Cancelled);           \
      return;                                                    \
    }                                                            \
    if (ctx) {                                                   \
      (ctx)->donePhases.fetch_add(1, std::memory_order_relaxed); \
      MANIFOLD_RECORD_PHASE(ctx);                                \
    }                                                            \
  } while (0)
