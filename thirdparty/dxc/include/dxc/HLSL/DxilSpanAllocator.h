///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilSpanAllocator.h                                                       //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Allocator for resources or other things that span a range of space        //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "dxc/Support/Global.h"
#include <set>
#include <map>

namespace hlsl {

template<typename T_index, typename T_element>
class SpanAllocator {
public:
  struct Span {
    Span(const T_element *element, T_index start, T_index end)
      : element(element), start(start), end(end) {
      DXASSERT_NOMSG(!(end < start));
    }
    const T_element *element;
    T_index start, end;  // inclusive
    bool operator<(const Span &other) const { return end < other.start; }
  };
  typedef std::set<Span> SpanSet;

public:
  SpanAllocator(T_index Min, T_index Max)
    : m_Min(Min), m_Max(Max), m_FirstFree(Min),
      m_Unbounded(nullptr), m_AllocationFull(false) {
    DXASSERT_NOMSG(Min <= Max);
  }
  T_index GetMin() { return m_Min; }
  T_index GetMax() { return m_Max; }
  T_index GetFirstFree() { return m_FirstFree; }
  bool IsFull() { return m_AllocationFull; }
  void SetUnbounded(const T_element *element) { m_Unbounded = element; }
  const T_element *GetUnbounded() const { return m_Unbounded; }
  const SpanSet &GetSpans() const { return m_Spans; }

  // Find size gap starting at pos, updating pos, and returning true if successful
  bool Find(T_index size, T_index &pos, T_index align = 1) {
    DXASSERT_NOMSG(size);
    if (size - 1 > m_Max - m_Min)
      return false;
    if (pos < m_FirstFree)
      pos = m_FirstFree;
    if (!UpdatePos(pos, size, align))
      return false;
    T_index end = pos + (size - 1);
    auto next = m_Spans.lower_bound(Span(nullptr, pos, end));
    if (next == m_Spans.end() || end < next->start)
      return true;  // it fits here
    return Find(size, next, pos, align);
  }

  // Finds the farthest position at which an element could be allocated.
  bool FindForUnbounded(T_index &pos, T_index align = 1) {
    if (m_Spans.empty()) {
      pos = m_Min;
      return UpdatePos(pos, /*size*/1, align);
    }

    pos = m_Spans.crbegin()->end;
    return IncPos(pos, /*inc*/ 1, /*size*/1, align);
  }

  // allocate element size in first available space, returns false on failure
  bool Allocate(const T_element *element, T_index size, T_index &pos, T_index align = 1) {
    DXASSERT_NOMSG(size);
    if (size - 1 > m_Max - m_Min)
      return false;
    if (m_AllocationFull)
      return false;
    pos = m_FirstFree;
    if (!UpdatePos(pos, size, align))
      return false;
    auto result = m_Spans.emplace(element, pos, pos + (size - 1));
    if (result.second) {
      AdvanceFirstFree(result.first);
      return true;
    }
    // Collision, find a gap from iterator
    if (!Find(size, result.first, pos, align))
      return false;
    result = m_Spans.emplace(element, pos, pos + (size - 1));
    return result.second;
  }

  bool AllocateUnbounded(const T_element *element, T_index &pos, T_index align = 1) {
    if (m_AllocationFull)
      return false;
    if (m_Spans.empty()) {
      pos = m_Min;
      if (!UpdatePos(pos, /*size*/1, align))
        return false;
    } else {
      // This will allocate after the last span
      auto it = m_Spans.end();  --it;   // find last span
      DXASSERT_NOMSG(it != m_Spans.end());
      pos = it->end;
      if (!IncPos(pos, /*inc*/1, /*size*/1, align))
        return false;
    }
    const T_element *conflict = Insert(element, pos, m_Max);
    DXASSERT_NOMSG(!conflict);
    if (!conflict)
      SetUnbounded(element);
    return !conflict;
  }

  // Insert at specific location, returning conflicting element on collision
  const T_element *Insert(const T_element *element, T_index start, T_index end) {
    DXASSERT_NOMSG(m_Min <= start && start <= end && end <= m_Max);
    auto result = m_Spans.emplace(element, start, end);
    if (!result.second)
      return result.first->element;
    AdvanceFirstFree(result.first);
    return nullptr;
  }

  // Insert at specific location, overwriting anything previously there,
  // losing their element pointer, but conserving the spans they represented.
  void ForceInsertAndClobber(const T_element *element, T_index start, T_index end) {
    DXASSERT_NOMSG(m_Min <= start && start <= end && end <= m_Max);
    for (;;) {
      auto result = m_Spans.emplace(element, start, end);
      if (result.second)
        break;
      // Delete the spans we overlap with, but make sure our new span covers what they covered.
      start = std::min(result.first->start, start);
      end = std::max(result.first->end, end);
      m_Spans.erase(result.first);
    }
  }

private:
  // Find size gap starting at iterator, updating pos, and returning true if successful
  bool Find(T_index size, typename SpanSet::const_iterator it, T_index &pos, T_index align = 1) {
    pos = it->end;
    if (!IncPos(pos, /*inc*/1, size, align))
      return false;
    for (++it; it != m_Spans.end() && (it->start < pos || it->start - pos < size); ++it) {
      pos = it->end;
      if (!IncPos(pos, /*inc*/1, size, align))
        return false;
    }
    return true;
  }

  // Advance m_FirstFree if it's in span
  void AdvanceFirstFree(typename SpanSet::const_iterator it) {
    if (it->start <= m_FirstFree && m_FirstFree <= it->end) {
      for (; it != m_Spans.end(); ) {
        if (it->end >= m_Max) {
          m_AllocationFull = true;
          break;
        }
        m_FirstFree = it->end + 1;
        ++it;
        if (it != m_Spans.end() && m_FirstFree < it->start)
          break;
      }
    }
  }

  T_index Align(T_index pos, T_index align) {
    T_index rem = (1 < align) ? pos % align : 0;
    return rem ? pos + (align - rem) : pos;
  }

  bool IncPos(T_index &pos, T_index inc = 1, T_index size = 1, T_index align = 1) {
    DXASSERT_NOMSG(inc > 0);
    if (pos + inc < pos)
      return false; // overflow
    pos += inc;
    return UpdatePos(pos, size, align);
  }
  bool UpdatePos(T_index &pos, T_index size = 1, T_index align = 1) {
    if ((size - 1) > m_Max - m_Min || pos < m_Min || pos > m_Max - (size - 1))
      return false;
    T_index aligned = Align(pos, align);
    if (aligned < pos || aligned > m_Max - (size - 1))
      return false; // overflow on alignment, or won't fit
    pos = aligned;
    return true;
  }

private:
  SpanSet m_Spans;
  T_index m_Min, m_Max, m_FirstFree;
  const T_element *m_Unbounded;
  bool m_AllocationFull;
};

template<typename T_index, typename T_element>
class SpacesAllocator {
public:
  typedef SpanAllocator<T_index, T_element> Allocator;
  typedef std::map<T_index, Allocator > AllocatorMap;

  Allocator &Get(T_index SpaceID) {
    auto it = m_Allocators.find(SpaceID);
    if (it != m_Allocators.end())
      return it->second;
    auto result = m_Allocators.emplace(SpaceID, Allocator(0, UINT_MAX));
    DXASSERT(result.second, "Failed to allocate new Allocator");
    return result.first->second;
  }

private:
  AllocatorMap m_Allocators;
};

}
