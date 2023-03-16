// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2011, SIL International, All rights reserved.


#pragma once

#include "inc/Code.h"
#include "inc/Slot.h"

namespace graphite2 {

struct Rule {
  const vm::Machine::Code * constraint,
                 * action;
  unsigned short   sort;
  byte             preContext;
#ifndef NDEBUG
  uint16           rule_idx;
#endif

  Rule();
  ~Rule() {}

  CLASS_NEW_DELETE;

private:
  Rule(const Rule &);
  Rule & operator = (const Rule &);
};

inline
Rule::Rule()
: constraint(0),
  action(0),
  sort(0),
  preContext(0)
{
#ifndef NDEBUG
  rule_idx = 0;
#endif
}


struct RuleEntry
{
  const Rule   * rule;

  inline
  bool operator < (const RuleEntry &r) const
  {
    const unsigned short lsort = rule->sort, rsort = r.rule->sort;
    return lsort > rsort || (lsort == rsort && rule < r.rule);
  }

  inline
  bool operator == (const RuleEntry &r) const
  {
    return rule == r.rule;
  }
};


struct State
{
  const RuleEntry     * rules,
                      * rules_end;

  bool   empty() const;
};

inline
bool State::empty() const
{
    return rules_end == rules;
}


class SlotMap
{
public:
  enum {MAX_SLOTS=64};
  SlotMap(Segment & seg, uint8 direction, size_t maxSize);

  Slot       * * begin();
  Slot       * * end();
  size_t         size() const;
  unsigned short context() const;
  void           reset(Slot &, unsigned short);

  Slot * const & operator[](int n) const;
  Slot       * & operator [] (int);
  void           pushSlot(Slot * const slot);
  void           collectGarbage(Slot *& aSlot);

  Slot         * highwater() { return m_highwater; }
  void           highwater(Slot *s) { m_highwater = s; m_highpassed = false; }
  bool           highpassed() const { return m_highpassed; }
  void           highpassed(bool v) { m_highpassed = v; }

  uint8          dir() const { return m_dir; }
  int            decMax() { return --m_maxSize; }

  Segment &    segment;
private:
  Slot         * m_slot_map[MAX_SLOTS+1];
  unsigned short m_size;
  unsigned short m_precontext;
  Slot         * m_highwater;
  int            m_maxSize;
  uint8          m_dir;
  bool           m_highpassed;
};


class FiniteStateMachine
{
public:
  enum {MAX_RULES=128};

private:
  class Rules
  {
  public:
      Rules();
      void              clear();
      const RuleEntry * begin() const;
      const RuleEntry * end() const;
      size_t            size() const;

      void accumulate_rules(const State &state);

  private:
      RuleEntry * m_begin,
                * m_end,
                  m_rules[MAX_RULES*2];
  };

public:
  FiniteStateMachine(SlotMap & map, json * logger);
  void      reset(Slot * & slot, const short unsigned int max_pre_ctxt);

  Rules     rules;
  SlotMap   & slots;
  json    * const dbgout;
};


inline
FiniteStateMachine::FiniteStateMachine(SlotMap& map, json * logger)
: slots(map),
  dbgout(logger)
{
}

inline
void FiniteStateMachine::reset(Slot * & slot, const short unsigned int max_pre_ctxt)
{
  rules.clear();
  int ctxt = 0;
  for (; ctxt != max_pre_ctxt && slot->prev(); ++ctxt, slot = slot->prev());
  slots.reset(*slot, ctxt);
}

inline
FiniteStateMachine::Rules::Rules()
  : m_begin(m_rules), m_end(m_rules)
{
}

inline
void FiniteStateMachine::Rules::clear()
{
  m_end = m_begin;
}

inline
const RuleEntry * FiniteStateMachine::Rules::begin() const
{
  return m_begin;
}

inline
const RuleEntry * FiniteStateMachine::Rules::end() const
{
  return m_end;
}

inline
size_t FiniteStateMachine::Rules::size() const
{
  return m_end - m_begin;
}

inline
void FiniteStateMachine::Rules::accumulate_rules(const State &state)
{
  // Only bother if there are rules in the State object.
  if (state.empty()) return;

  // Merge the new sorted rules list into the current sorted result set.
  const RuleEntry * lre = begin(), * rre = state.rules;
  RuleEntry * out = m_rules + (m_begin == m_rules)*MAX_RULES;
  const RuleEntry * const lrend = out + MAX_RULES,
                  * const rrend = state.rules_end;
  m_begin = out;
  while (lre != end() && out != lrend)
  {
    if (*lre < *rre)      *out++ = *lre++;
    else if (*rre < *lre) { *out++ = *rre++; }
    else                { *out++ = *lre++; ++rre; }

    if (rre == rrend)
    {
      while (lre != end() && out != lrend) { *out++ = *lre++; }
      m_end = out;
      return;
    }
  }
  while (rre != rrend && out != lrend) { *out++ = *rre++; }
  m_end = out;
}

inline
SlotMap::SlotMap(Segment & seg, uint8 direction, size_t maxSize)
: segment(seg), m_size(0), m_precontext(0), m_highwater(0),
    m_maxSize(int(maxSize)), m_dir(direction), m_highpassed(false)
{
    m_slot_map[0] = 0;
}

inline
Slot * * SlotMap::begin()
{
  return &m_slot_map[1]; // allow map to go 1 before slot_map when inserting
                         // at start of segment.
}

inline
Slot * * SlotMap::end()
{
  return m_slot_map + m_size + 1;
}

inline
size_t SlotMap::size() const
{
  return m_size;
}

inline
short unsigned int SlotMap::context() const
{
  return m_precontext;
}

inline
void SlotMap::reset(Slot & slot, short unsigned int ctxt)
{
  m_size = 0;
  m_precontext = ctxt;
  *m_slot_map = slot.prev();
}

inline
void SlotMap::pushSlot(Slot*const slot)
{
  m_slot_map[++m_size] = slot;
}

inline
Slot * const & SlotMap::operator[](int n) const
{
  return m_slot_map[n + 1];
}

inline
Slot * & SlotMap::operator[](int n)
{
  return m_slot_map[n + 1];
}

} // namespace graphite2
