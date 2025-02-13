// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2010, SIL International, All rights reserved.

#pragma once

#include <cstdlib>
#include "inc/Code.h"

namespace graphite2 {

class Segment;
class Face;
class Silf;
struct Rule;
struct RuleEntry;
struct State;
class FiniteStateMachine;
class Error;
class ShiftCollider;
class KernCollider;
class json;

enum passtype;

class Pass
{
public:
    Pass();
    ~Pass();

    bool readPass(const byte * pPass, size_t pass_length, size_t subtable_base, Face & face,
        enum passtype pt, uint32 version, Error &e);
    bool runGraphite(vm::Machine & m, FiniteStateMachine & fsm, bool reverse) const;
    void init(Silf *silf) { m_silf = silf; }
    byte collisionLoops() const { return m_numCollRuns; }
    bool reverseDir() const { return m_isReverseDir; }

    CLASS_NEW_DELETE
private:
    void    findNDoRule(Slot* & iSlot, vm::Machine &, FiniteStateMachine& fsm) const;
    int     doAction(const vm::Machine::Code* codeptr, Slot * & slot_out, vm::Machine &) const;
    bool    testPassConstraint(vm::Machine & m) const;
    bool    testConstraint(const Rule & r, vm::Machine &) const;
    bool    readRules(const byte * rule_map, const size_t num_entries,
                     const byte *precontext, const uint16 * sort_key,
                     const uint16 * o_constraint, const byte *constraint_data,
                     const uint16 * o_action, const byte * action_data,
                     Face &, enum passtype pt, Error &e);
    bool    readStates(const byte * starts, const byte * states, const byte * o_rule_map, Face &, Error &e);
    bool    readRanges(const byte * ranges, size_t num_ranges, Error &e);
    uint16  glyphToCol(const uint16 gid) const;
    bool    runFSM(FiniteStateMachine & fsm, Slot * slot) const;
    void    dumpRuleEventConsidered(const FiniteStateMachine & fsm, const RuleEntry & re) const;
    void    dumpRuleEventOutput(const FiniteStateMachine & fsm, const Rule & r, Slot * os) const;
    void    adjustSlot(int delta, Slot * & slot_out, SlotMap &) const;
    bool    collisionShift(Segment *seg, int dir, json * const dbgout) const;
    bool    collisionKern(Segment *seg, int dir, json * const dbgout) const;
    bool    collisionFinish(Segment *seg, GR_MAYBE_UNUSED json * const dbgout) const;
    bool    resolveCollisions(Segment *seg, Slot *slot, Slot *start, ShiftCollider &coll, bool isRev,
                     int dir, bool &moved, bool &hasCol, json * const dbgout) const;
    float   resolveKern(Segment *seg, Slot *slot, Slot *start, int dir,
                     float &ymin, float &ymax, json *const dbgout) const;

    const Silf        * m_silf;
    uint16            * m_cols;
    Rule              * m_rules; // rules
    RuleEntry         * m_ruleMap;
    uint16            * m_startStates; // prectxt length
    uint16            * m_transitions;
    State             * m_states;
    vm::Machine::Code * m_codes;
    byte              * m_progs;

    byte   m_numCollRuns;
    byte   m_kernColls;
    byte   m_iMaxLoop;
    uint16 m_numGlyphs;
    uint16 m_numRules;
    uint16 m_numStates;
    uint16 m_numTransition;
    uint16 m_numSuccess;
    uint16 m_successStart;
    uint16 m_numColumns;
    byte m_minPreCtxt;
    byte m_maxPreCtxt;
    byte m_colThreshold;
    bool m_isReverseDir;
    vm::Machine::Code m_cPConstraint;

private:        //defensive
    Pass(const Pass&);
    Pass& operator=(const Pass&);
};

} // namespace graphite2
