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
#ifndef DRACO_MESH_VALENCE_CACHE_H_
#define DRACO_MESH_VALENCE_CACHE_H_

#include "draco/attributes/geometry_indices.h"
#include "draco/core/draco_index_type_vector.h"
#include "draco/core/macros.h"

namespace draco {

// ValenceCache provides support for the caching of valences off of some kind of
// CornerTable 'type' of class.
// No valences should be queried before Caching is
// performed and values should be removed/recached when changes to the
// underlying mesh are taking place.

template <class CornerTableT>
class ValenceCache {
  const CornerTableT &table_;

 public:
  explicit ValenceCache(const CornerTableT &table) : table_(table) {}

  // Do not call before CacheValences() / CacheValencesInaccurate().
  inline int8_t ValenceFromCacheInaccurate(CornerIndex c) const {
    if (c == kInvalidCornerIndex) {
      return -1;
    }
    return ValenceFromCacheInaccurate(table_.Vertex(c));
  }
  inline int32_t ValenceFromCache(CornerIndex c) const {
    if (c == kInvalidCornerIndex) {
      return -1;
    }
    return ValenceFromCache(table_.Vertex(c));
  }

  inline int32_t ConfidentValenceFromCache(VertexIndex v) const {
    DRACO_DCHECK_LT(v.value(), table_.num_vertices());
    DRACO_DCHECK_EQ(vertex_valence_cache_32_bit_.size(), table_.num_vertices());
    return vertex_valence_cache_32_bit_[v];
  }

  // Collect the valence for all vertices so they can be reused later.  The
  // 'inaccurate' versions of this family of functions clips the true valence
  // of the vertices to 8 signed bits as a space optimization.  This clipping
  // will lead to occasionally wrong results.  If accurate results are required
  // under all circumstances, do not use the 'inaccurate' version or else
  // use it and fetch the correct result in the event the value appears clipped.
  // The topology of the mesh should be a constant when Valence Cache functions
  // are being used.  Modification of the mesh while cache(s) are filled will
  // not guarantee proper results on subsequent calls unless they are rebuilt.
  void CacheValencesInaccurate() const {
    if (vertex_valence_cache_8_bit_.size() == 0) {
      const VertexIndex vertex_count = VertexIndex(table_.num_vertices());
      vertex_valence_cache_8_bit_.resize(vertex_count.value());
      for (VertexIndex v = VertexIndex(0); v < vertex_count; v += 1) {
        vertex_valence_cache_8_bit_[v] = static_cast<int8_t>(
            (std::min)(static_cast<int32_t>(std::numeric_limits<int8_t>::max()),
                       table_.Valence(v)));
      }
    }
  }
  void CacheValences() const {
    if (vertex_valence_cache_32_bit_.size() == 0) {
      const VertexIndex vertex_count = VertexIndex(table_.num_vertices());
      vertex_valence_cache_32_bit_.resize(vertex_count.value());
      for (VertexIndex v = VertexIndex(0); v < vertex_count; v += 1) {
        vertex_valence_cache_32_bit_[v] = table_.Valence(v);
      }
    }
  }

  inline int8_t ConfidentValenceFromCacheInaccurate(CornerIndex c) const {
    DRACO_DCHECK_GE(c.value(), 0);
    return ConfidentValenceFromCacheInaccurate(table_.ConfidentVertex(c));
  }
  inline int32_t ConfidentValenceFromCache(CornerIndex c) const {
    DRACO_DCHECK_GE(c.value(), 0);
    return ConfidentValenceFromCache(table_.ConfidentVertex(c));
  }
  inline int8_t ValenceFromCacheInaccurate(VertexIndex v) const {
    DRACO_DCHECK_EQ(vertex_valence_cache_8_bit_.size(), table_.num_vertices());
    if (v == kInvalidVertexIndex || v.value() >= table_.num_vertices()) {
      return -1;
    }
    return ConfidentValenceFromCacheInaccurate(v);
  }
  inline int8_t ConfidentValenceFromCacheInaccurate(VertexIndex v) const {
    DRACO_DCHECK_LT(v.value(), table_.num_vertices());
    DRACO_DCHECK_EQ(vertex_valence_cache_8_bit_.size(), table_.num_vertices());
    return vertex_valence_cache_8_bit_[v];
  }

  // TODO(draco-eng) Add unit tests for ValenceCache functions.
  inline int32_t ValenceFromCache(VertexIndex v) const {
    DRACO_DCHECK_EQ(vertex_valence_cache_32_bit_.size(), table_.num_vertices());
    if (v == kInvalidVertexIndex || v.value() >= table_.num_vertices()) {
      return -1;
    }
    return ConfidentValenceFromCache(v);
  }

  // Clear the cache of valences and deallocate the memory.
  void ClearValenceCacheInaccurate() const {
    vertex_valence_cache_8_bit_.clear();
    // Force erasure.
    IndexTypeVector<VertexIndex, int8_t>().swap(vertex_valence_cache_8_bit_);
  }
  void ClearValenceCache() const {
    vertex_valence_cache_32_bit_.clear();
    // Force erasure.
    IndexTypeVector<VertexIndex, int32_t>().swap(vertex_valence_cache_32_bit_);
  }

  bool IsCacheEmpty() const {
    return vertex_valence_cache_8_bit_.size() == 0 &&
           vertex_valence_cache_32_bit_.size() == 0;
  }

 private:
  // Retain valences and clip them to char size.
  mutable IndexTypeVector<VertexIndex, int8_t> vertex_valence_cache_8_bit_;
  mutable IndexTypeVector<VertexIndex, int32_t> vertex_valence_cache_32_bit_;
};

}  // namespace draco

#endif  // DRACO_MESH_VALENCE_CACHE_H_
