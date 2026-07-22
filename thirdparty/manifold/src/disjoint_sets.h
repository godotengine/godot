// from https://github.com/wjakob/dset, changed to add connected component
// computation
//
// Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>
//
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
// claim that you wrote the original software. If you use this software
// in a product, an acknowledgment in the product documentation would be
// appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
// misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.
//
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

class DisjointSets {
 public:
  DisjointSets(uint32_t size) : mData(size) {
    for (uint32_t i = 0; i < size; ++i) mData[i] = (uint32_t)i;
  }

  uint32_t find(uint32_t id) const {
    while (id != parent(id)) {
      uint64_t value = mData[id];
      uint32_t new_parent = parent((uint32_t)value);
      uint64_t new_value = (value & 0xFFFFFFFF00000000ULL) | new_parent;
      /* Try to update parent (may fail, that's ok) */
      if (value != new_value) mData[id].compare_exchange_weak(value, new_value);
      id = new_parent;
    }
    return id;
  }

  bool same(uint32_t id1, uint32_t id2) const {
    for (;;) {
      id1 = find(id1);
      id2 = find(id2);
      if (id1 == id2) return true;
      if (parent(id1) == id1) return false;
    }
  }

  uint32_t unite(uint32_t id1, uint32_t id2) {
    for (;;) {
      id1 = find(id1);
      id2 = find(id2);

      if (id1 == id2) return id1;

      uint32_t r1 = rank(id1), r2 = rank(id2);

      if (r1 > r2 || (r1 == r2 && id1 < id2)) {
        std::swap(r1, r2);
        std::swap(id1, id2);
      }

      uint64_t oldEntry = ((uint64_t)r1 << 32) | id1;
      uint64_t newEntry = ((uint64_t)r1 << 32) | id2;

      if (!mData[id1].compare_exchange_strong(oldEntry, newEntry)) continue;

      if (r1 == r2) {
        oldEntry = ((uint64_t)r2 << 32) | id2;
        newEntry = ((uint64_t)(r2 + 1) << 32) | id2;
        /* Try to update the rank (may fail, retry if rank = 0) */
        if (!mData[id2].compare_exchange_strong(oldEntry, newEntry) && r2 == 0)
          continue;
      }

      break;
    }
    return id2;
  }

  uint32_t size() const { return (uint32_t)mData.size(); }

  uint32_t rank(uint32_t id) const {
    return ((uint32_t)(mData[id] >> 32)) & 0x7FFFFFFFu;
  }

  uint32_t parent(uint32_t id) const { return (uint32_t)mData[id]; }

  int connectedComponents(std::vector<int>& components) {
    components.resize(mData.size());
    int lonelyNodes = 0;
    std::unordered_map<uint32_t, int> toLabel;
    for (size_t i = 0; i < mData.size(); ++i) {
      // we optimize for connected component of size 1
      // no need to put them into the hashmap
      auto iParent = find(i);
      if (rank(iParent) == 0) {
        components[i] = static_cast<int>(toLabel.size()) + lonelyNodes++;
        continue;
      }
      auto iter = toLabel.find(iParent);
      if (iter == toLabel.end()) {
        auto s = static_cast<uint32_t>(toLabel.size()) + lonelyNodes;
        toLabel.insert(std::make_pair(iParent, s));
        components[i] = s;
      } else {
        components[i] = iter->second;
      }
    }
    return toLabel.size() + lonelyNodes;
  }

  mutable std::vector<std::atomic<uint64_t>> mData;
};
