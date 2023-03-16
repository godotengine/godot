// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2010, SIL International, All rights reserved.

#include "inc/Main.h"
#include "inc/debug.h"
#include "inc/Endian.h"
#include "inc/Pass.h"
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include "inc/Segment.h"
#include "inc/Code.h"
#include "inc/Rule.h"
#include "inc/Error.h"
#include "inc/Collider.h"

using namespace graphite2;
using vm::Machine;
typedef Machine::Code  Code;

enum KernCollison
{
    None       = 0,
    CrossSpace = 1,
    InWord     = 2,
    reserved   = 3
};

Pass::Pass()
: m_silf(0),
  m_cols(0),
  m_rules(0),
  m_ruleMap(0),
  m_startStates(0),
  m_transitions(0),
  m_states(0),
  m_codes(0),
  m_progs(0),
  m_numCollRuns(0),
  m_kernColls(0),
  m_iMaxLoop(0),
  m_numGlyphs(0),
  m_numRules(0),
  m_numStates(0),
  m_numTransition(0),
  m_numSuccess(0),
  m_successStart(0),
  m_numColumns(0),
  m_minPreCtxt(0),
  m_maxPreCtxt(0),
  m_colThreshold(0),
  m_isReverseDir(false)
{
}

Pass::~Pass()
{
    free(m_cols);
    free(m_startStates);
    free(m_transitions);
    free(m_states);
    free(m_ruleMap);

    if (m_rules) delete [] m_rules;
    if (m_codes) delete [] m_codes;
    free(m_progs);
}

bool Pass::readPass(const byte * const pass_start, size_t pass_length, size_t subtable_base,
        GR_MAYBE_UNUSED Face & face, passtype pt, GR_MAYBE_UNUSED uint32 version, Error &e)
{
    const byte * p              = pass_start,
               * const pass_end = p + pass_length;
    size_t numRanges;

    if (e.test(pass_length < 40, E_BADPASSLENGTH)) return face.error(e);
    // Read in basic values
    const byte flags = be::read<byte>(p);
    if (e.test((flags & 0x1f) &&
            (pt < PASS_TYPE_POSITIONING || !m_silf->aCollision() || !face.glyphs().hasBoxes() || !(m_silf->flags() & 0x20)),
            E_BADCOLLISIONPASS))
        return face.error(e);
    m_numCollRuns = flags & 0x7;
    m_kernColls   = (flags >> 3) & 0x3;
    m_isReverseDir = (flags >> 5) & 0x1;
    m_iMaxLoop = be::read<byte>(p);
    if (m_iMaxLoop < 1) m_iMaxLoop = 1;
    be::skip<byte>(p,2); // skip maxContext & maxBackup
    m_numRules = be::read<uint16>(p);
    if (e.test(!m_numRules && m_numCollRuns == 0, E_BADEMPTYPASS)) return face.error(e);
    be::skip<uint16>(p);   // fsmOffset - not sure why we would want this
    const byte * const pcCode = pass_start + be::read<uint32>(p) - subtable_base,
               * const rcCode = pass_start + be::read<uint32>(p) - subtable_base,
               * const aCode  = pass_start + be::read<uint32>(p) - subtable_base;
    be::skip<uint32>(p);
    m_numStates = be::read<uint16>(p);
    m_numTransition = be::read<uint16>(p);
    m_numSuccess = be::read<uint16>(p);
    m_numColumns = be::read<uint16>(p);
    numRanges = be::read<uint16>(p);
    be::skip<uint16>(p, 3); // skip searchRange, entrySelector & rangeShift.
    assert(p - pass_start == 40);
    // Perform some sanity checks.
    if ( e.test(m_numTransition > m_numStates, E_BADNUMTRANS)
            || e.test(m_numSuccess > m_numStates, E_BADNUMSUCCESS)
            || e.test(m_numSuccess + m_numTransition < m_numStates, E_BADNUMSTATES)
            || e.test(m_numRules && numRanges == 0, E_NORANGES)
            || e.test(m_numColumns > 0x7FFF, E_BADNUMCOLUMNS))
        return face.error(e);

    m_successStart = m_numStates - m_numSuccess;
    // test for beyond end - 1 to account for reading uint16
    if (e.test(p + numRanges * 6 - 2 > pass_end, E_BADPASSLENGTH)) return face.error(e);
    m_numGlyphs = be::peek<uint16>(p + numRanges * 6 - 4) + 1;
    // Calculate the start of various arrays.
    const byte * const ranges = p;
    be::skip<uint16>(p, numRanges*3);
    const byte * const o_rule_map = p;
    be::skip<uint16>(p, m_numSuccess + 1);

    // More sanity checks
    if (e.test(reinterpret_cast<const byte *>(o_rule_map + m_numSuccess*sizeof(uint16)) > pass_end
            || p > pass_end, E_BADRULEMAPLEN))
        return face.error(e);
    const size_t numEntries = be::peek<uint16>(o_rule_map + m_numSuccess*sizeof(uint16));
    const byte * const   rule_map = p;
    be::skip<uint16>(p, numEntries);

    if (e.test(p + 2*sizeof(uint8) > pass_end, E_BADPASSLENGTH)) return face.error(e);
    m_minPreCtxt = be::read<uint8>(p);
    m_maxPreCtxt = be::read<uint8>(p);
    if (e.test(m_minPreCtxt > m_maxPreCtxt, E_BADCTXTLENBOUNDS)) return face.error(e);
    const byte * const start_states = p;
    be::skip<int16>(p, m_maxPreCtxt - m_minPreCtxt + 1);
    const uint16 * const sort_keys = reinterpret_cast<const uint16 *>(p);
    be::skip<uint16>(p, m_numRules);
    const byte * const precontext = p;
    be::skip<byte>(p, m_numRules);

    if (e.test(p + sizeof(uint16) + sizeof(uint8) > pass_end, E_BADCTXTLENS)) return face.error(e);
    m_colThreshold = be::read<uint8>(p);
    if (m_colThreshold == 0) m_colThreshold = 10;       // A default
    const size_t pass_constraint_len = be::read<uint16>(p);

    const uint16 * const o_constraint = reinterpret_cast<const uint16 *>(p);
    be::skip<uint16>(p, m_numRules + 1);
    const uint16 * const o_actions = reinterpret_cast<const uint16 *>(p);
    be::skip<uint16>(p, m_numRules + 1);
    const byte * const states = p;
    if (e.test(2u*m_numTransition*m_numColumns >= (unsigned)(pass_end - p), E_BADPASSLENGTH)
            || e.test(p >= pass_end, E_BADPASSLENGTH))
        return face.error(e);
    be::skip<int16>(p, m_numTransition*m_numColumns);
    be::skip<uint8>(p);
    if (e.test(p != pcCode, E_BADPASSCCODEPTR)) return face.error(e);
    be::skip<byte>(p, pass_constraint_len);
    if (e.test(p != rcCode, E_BADRULECCODEPTR)
        || e.test(size_t(rcCode - pcCode) != pass_constraint_len, E_BADCCODELEN)) return face.error(e);
    be::skip<byte>(p, be::peek<uint16>(o_constraint + m_numRules));
    if (e.test(p != aCode, E_BADACTIONCODEPTR)) return face.error(e);
    be::skip<byte>(p, be::peek<uint16>(o_actions + m_numRules));

    // We should be at the end or within the pass
    if (e.test(p > pass_end, E_BADPASSLENGTH)) return face.error(e);

    // Load the pass constraint if there is one.
    if (pass_constraint_len)
    {
        face.error_context(face.error_context() + 1);
        m_cPConstraint = vm::Machine::Code(true, pcCode, pcCode + pass_constraint_len,
                                  precontext[0], be::peek<uint16>(sort_keys), *m_silf, face, PASS_TYPE_UNKNOWN);
        if (e.test(!m_cPConstraint, E_OUTOFMEM)
                || e.test(m_cPConstraint.status() != Code::loaded, int(m_cPConstraint.status()) + E_CODEFAILURE))
            return face.error(e);
        face.error_context(face.error_context() - 1);
    }
    if (m_numRules)
    {
        if (!readRanges(ranges, numRanges, e)) return face.error(e);
        if (!readRules(rule_map, numEntries,  precontext, sort_keys,
                   o_constraint, rcCode, o_actions, aCode, face, pt, e)) return false;
    }
#ifdef GRAPHITE2_TELEMETRY
    telemetry::category _states_cat(face.tele.states);
#endif
    return m_numRules ? readStates(start_states, states, o_rule_map, face, e) : true;
}


bool Pass::readRules(const byte * rule_map, const size_t num_entries,
                     const byte *precontext, const uint16 * sort_key,
                     const uint16 * o_constraint, const byte *rc_data,
                     const uint16 * o_action,     const byte * ac_data,
                     Face & face, passtype pt, Error &e)
{
    const byte * const ac_data_end = ac_data + be::peek<uint16>(o_action + m_numRules);
    const byte * const rc_data_end = rc_data + be::peek<uint16>(o_constraint + m_numRules);

    precontext += m_numRules;
    sort_key   += m_numRules;
    o_constraint += m_numRules;
    o_action += m_numRules;

    // Load rules.
    const byte * ac_begin = 0, * rc_begin = 0,
               * ac_end = ac_data + be::peek<uint16>(o_action),
               * rc_end = rc_data + be::peek<uint16>(o_constraint);

    // Allocate pools
    m_rules = new Rule [m_numRules];
    m_codes = new Code [m_numRules*2];
    int totalSlots = 0;
    const uint16 *tsort = sort_key;
    for (int i = 0; i < m_numRules; ++i)
        totalSlots += be::peek<uint16>(--tsort);
    const size_t prog_pool_sz = vm::Machine::Code::estimateCodeDataOut(ac_end - ac_data + rc_end - rc_data, 2 * m_numRules, totalSlots);
    m_progs = gralloc<byte>(prog_pool_sz);
    byte * prog_pool_free = m_progs,
         * prog_pool_end  = m_progs + prog_pool_sz;
    if (e.test(!(m_rules && m_codes && m_progs), E_OUTOFMEM)) return face.error(e);

    Rule * r = m_rules + m_numRules - 1;
    for (size_t n = m_numRules; r >= m_rules; --n, --r, ac_end = ac_begin, rc_end = rc_begin)
    {
        face.error_context((face.error_context() & 0xFFFF00) + EC_ARULE + int((n - 1) << 24));
        r->preContext = *--precontext;
        r->sort       = be::peek<uint16>(--sort_key);
#ifndef NDEBUG
        r->rule_idx   = uint16(n - 1);
#endif
        if (r->sort > 63 || r->preContext >= r->sort || r->preContext > m_maxPreCtxt || r->preContext < m_minPreCtxt)
            return false;
        ac_begin      = ac_data + be::peek<uint16>(--o_action);
        --o_constraint;
        rc_begin      = be::peek<uint16>(o_constraint) ? rc_data + be::peek<uint16>(o_constraint) : rc_end;

        if (ac_begin > ac_end || ac_begin > ac_data_end || ac_end > ac_data_end
                || rc_begin > rc_end || rc_begin > rc_data_end || rc_end > rc_data_end
                || vm::Machine::Code::estimateCodeDataOut(ac_end - ac_begin + rc_end - rc_begin, 2, r->sort) > size_t(prog_pool_end - prog_pool_free))
            return false;
        r->action     = new (m_codes+n*2-2) vm::Machine::Code(false, ac_begin, ac_end, r->preContext, r->sort, *m_silf, face, pt, &prog_pool_free);
        r->constraint = new (m_codes+n*2-1) vm::Machine::Code(true,  rc_begin, rc_end, r->preContext, r->sort, *m_silf, face, pt, &prog_pool_free);

        if (e.test(!r->action || !r->constraint, E_OUTOFMEM)
                || e.test(r->action->status() != Code::loaded, int(r->action->status()) + E_CODEFAILURE)
                || e.test(r->constraint->status() != Code::loaded, int(r->constraint->status()) + E_CODEFAILURE)
                || e.test(!r->constraint->immutable(), E_MUTABLECCODE))
            return face.error(e);
    }

    byte * const moved_progs = prog_pool_free > m_progs ? static_cast<byte *>(realloc(m_progs, prog_pool_free - m_progs)) : 0;
    if (e.test(!moved_progs, E_OUTOFMEM))
    {
        free(m_progs);
        m_progs = 0;
        return face.error(e);
    }

    if (moved_progs != m_progs)
    {
        for (Code * c = m_codes, * const ce = c + m_numRules*2; c != ce; ++c)
        {
            c->externalProgramMoved(moved_progs - m_progs);
        }
        m_progs = moved_progs;
    }

    // Load the rule entries map
    face.error_context((face.error_context() & 0xFFFF00) + EC_APASS);
    //TODO: Coverity: 1315804: FORWARD_NULL
    RuleEntry * re = m_ruleMap = gralloc<RuleEntry>(num_entries);
    if (e.test(!re, E_OUTOFMEM)) return face.error(e);
    for (size_t n = num_entries; n; --n, ++re)
    {
        const ptrdiff_t rn = be::read<uint16>(rule_map);
        if (e.test(rn >= m_numRules, E_BADRULENUM))  return face.error(e);
        re->rule = m_rules + rn;
    }

    return true;
}

static int cmpRuleEntry(const void *a, const void *b) { return (*(RuleEntry *)a < *(RuleEntry *)b ? -1 :
                                                                (*(RuleEntry *)b < *(RuleEntry *)a ? 1 : 0)); }

bool Pass::readStates(const byte * starts, const byte *states, const byte * o_rule_map, GR_MAYBE_UNUSED Face & face, Error &e)
{
#ifdef GRAPHITE2_TELEMETRY
    telemetry::category _states_cat(face.tele.starts);
#endif
    m_startStates = gralloc<uint16>(m_maxPreCtxt - m_minPreCtxt + 1);
#ifdef GRAPHITE2_TELEMETRY
    telemetry::set_category(face.tele.states);
#endif
    m_states      = gralloc<State>(m_numStates);
#ifdef GRAPHITE2_TELEMETRY
    telemetry::set_category(face.tele.transitions);
#endif
    m_transitions      = gralloc<uint16>(m_numTransition * m_numColumns);

    if (e.test(!m_startStates || !m_states || !m_transitions, E_OUTOFMEM)) return face.error(e);
    // load start states
    for (uint16 * s = m_startStates,
                * const s_end = s + m_maxPreCtxt - m_minPreCtxt + 1; s != s_end; ++s)
    {
        *s = be::read<uint16>(starts);
        if (e.test(*s >= m_numStates, E_BADSTATE))
        {
            face.error_context((face.error_context() & 0xFFFF00) + EC_ASTARTS + int((s - m_startStates) << 24));
            return face.error(e); // true;
        }
    }

    // load state transition table.
    for (uint16 * t = m_transitions,
                * const t_end = t + m_numTransition*m_numColumns; t != t_end; ++t)
    {
        *t = be::read<uint16>(states);
        if (e.test(*t >= m_numStates, E_BADSTATE))
        {
            face.error_context((face.error_context() & 0xFFFF00) + EC_ATRANS + int(((t - m_transitions) / m_numColumns) << 8));
            return face.error(e);
        }
    }

    State * s = m_states,
          * const success_begin = m_states + m_numStates - m_numSuccess;
    const RuleEntry * rule_map_end = m_ruleMap + be::peek<uint16>(o_rule_map + m_numSuccess*sizeof(uint16));
    for (size_t n = m_numStates; n; --n, ++s)
    {
        RuleEntry * const begin = s < success_begin ? 0 : m_ruleMap + be::read<uint16>(o_rule_map),
                  * const end   = s < success_begin ? 0 : m_ruleMap + be::peek<uint16>(o_rule_map);

        if (e.test(begin >= rule_map_end || end > rule_map_end || begin > end, E_BADRULEMAPPING))
        {
            face.error_context((face.error_context() & 0xFFFF00) + EC_ARULEMAP + int(n << 24));
            return face.error(e);
        }
        s->rules = begin;
        s->rules_end = (end - begin <= FiniteStateMachine::MAX_RULES)? end :
            begin + FiniteStateMachine::MAX_RULES;
        if (begin)      // keep UBSan happy can't call qsort with null begin
            qsort(begin, end - begin, sizeof(RuleEntry), &cmpRuleEntry);
    }

    return true;
}

bool Pass::readRanges(const byte * ranges, size_t num_ranges, Error &e)
{
    m_cols = gralloc<uint16>(m_numGlyphs);
    if (e.test(!m_cols, E_OUTOFMEM)) return false;
    memset(m_cols, 0xFF, m_numGlyphs * sizeof(uint16));
    for (size_t n = num_ranges; n; --n)
    {
        uint16     * ci     = m_cols + be::read<uint16>(ranges),
                   * ci_end = m_cols + be::read<uint16>(ranges) + 1,
                     col    = be::read<uint16>(ranges);

        if (e.test(ci >= ci_end || ci_end > m_cols+m_numGlyphs || col >= m_numColumns, E_BADRANGE))
            return false;

        // A glyph must only belong to one column at a time
        while (ci != ci_end && *ci == 0xffff)
            *ci++ = col;

        if (e.test(ci != ci_end, E_BADRANGE))
            return false;
    }
    return true;
}


bool Pass::runGraphite(vm::Machine & m, FiniteStateMachine & fsm, bool reverse) const
{
    Slot *s = m.slotMap().segment.first();
    if (!s || !testPassConstraint(m)) return true;
    if (reverse)
    {
        m.slotMap().segment.reverseSlots();
        s = m.slotMap().segment.first();
    }
    if (m_numRules)
    {
        Slot *currHigh = s->next();

#if !defined GRAPHITE2_NTRACING
        if (fsm.dbgout)  *fsm.dbgout << "rules" << json::array;
        json::closer rules_array_closer(fsm.dbgout);
#endif

        m.slotMap().highwater(currHigh);
        int lc = m_iMaxLoop;
        do
        {
            findNDoRule(s, m, fsm);
            if (m.status() != Machine::finished) return false;
            if (s && (s == m.slotMap().highwater() || m.slotMap().highpassed() || --lc == 0)) {
                if (!lc)
                    s = m.slotMap().highwater();
                lc = m_iMaxLoop;
                if (s)
                    m.slotMap().highwater(s->next());
            }
        } while (s);
    }
    //TODO: Use enums for flags
    const bool collisions = m_numCollRuns || m_kernColls;

    if (!collisions || !m.slotMap().segment.hasCollisionInfo())
        return true;

    if (m_numCollRuns)
    {
        if (!(m.slotMap().segment.flags() & Segment::SEG_INITCOLLISIONS))
        {
            m.slotMap().segment.positionSlots(0, 0, 0, m.slotMap().dir(), true);
//            m.slotMap().segment.flags(m.slotMap().segment.flags() | Segment::SEG_INITCOLLISIONS);
        }
        if (!collisionShift(&m.slotMap().segment, m.slotMap().dir(), fsm.dbgout))
            return false;
    }
    if ((m_kernColls) && !collisionKern(&m.slotMap().segment, m.slotMap().dir(), fsm.dbgout))
        return false;
    if (collisions && !collisionFinish(&m.slotMap().segment, fsm.dbgout))
        return false;
    return true;
}

bool Pass::runFSM(FiniteStateMachine& fsm, Slot * slot) const
{
    fsm.reset(slot, m_maxPreCtxt);
    if (fsm.slots.context() < m_minPreCtxt)
        return false;

    uint16 state = m_startStates[m_maxPreCtxt - fsm.slots.context()];
    uint8  free_slots = SlotMap::MAX_SLOTS;
    do
    {
        fsm.slots.pushSlot(slot);
        if (slot->gid() >= m_numGlyphs
         || m_cols[slot->gid()] == 0xffffU
         || --free_slots == 0
         || state >= m_numTransition)
            return free_slots != 0;

        const uint16 * transitions = m_transitions + state*m_numColumns;
        state = transitions[m_cols[slot->gid()]];
        if (state >= m_successStart)
            fsm.rules.accumulate_rules(m_states[state]);

        slot = slot->next();
    } while (state != 0 && slot);

    fsm.slots.pushSlot(slot);
    return true;
}

#if !defined GRAPHITE2_NTRACING

inline
Slot * input_slot(const SlotMap &  slots, const int n)
{
    Slot * s = slots[slots.context() + n];
    if (!s->isCopied())     return s;

    return s->prev() ? s->prev()->next() : (s->next() ? s->next()->prev() : slots.segment.last());
}

inline
Slot * output_slot(const SlotMap &  slots, const int n)
{
    Slot * s = slots[slots.context() + n - 1];
    return s ? s->next() : slots.segment.first();
}

#endif //!defined GRAPHITE2_NTRACING

void Pass::findNDoRule(Slot * & slot, Machine &m, FiniteStateMachine & fsm) const
{
    assert(slot);

    if (runFSM(fsm, slot))
    {
        // Search for the first rule which passes the constraint
        const RuleEntry *        r = fsm.rules.begin(),
                        * const re = fsm.rules.end();
        while (r != re && !testConstraint(*r->rule, m))
        {
            ++r;
            if (m.status() != Machine::finished)
                return;
        }

#if !defined GRAPHITE2_NTRACING
        if (fsm.dbgout)
        {
            if (fsm.rules.size() != 0)
            {
                *fsm.dbgout << json::item << json::object;
                dumpRuleEventConsidered(fsm, *r);
                if (r != re)
                {
                    const int adv = doAction(r->rule->action, slot, m);
                    dumpRuleEventOutput(fsm, *r->rule, slot);
                    if (r->rule->action->deletes()) fsm.slots.collectGarbage(slot);
                    adjustSlot(adv, slot, fsm.slots);
                    *fsm.dbgout << "cursor" << objectid(dslot(&fsm.slots.segment, slot))
                            << json::close; // Close RuelEvent object

                    return;
                }
                else
                {
                    *fsm.dbgout << json::close  // close "considered" array
                            << "output" << json::null
                            << "cursor" << objectid(dslot(&fsm.slots.segment, slot->next()))
                            << json::close;
                }
            }
        }
        else
#endif
        {
            if (r != re)
            {
                const int adv = doAction(r->rule->action, slot, m);
                if (m.status() != Machine::finished) return;
                if (r->rule->action->deletes()) fsm.slots.collectGarbage(slot);
                adjustSlot(adv, slot, fsm.slots);
                return;
            }
        }
    }

    slot = slot->next();
    return;
}

#if !defined GRAPHITE2_NTRACING

void Pass::dumpRuleEventConsidered(const FiniteStateMachine & fsm, const RuleEntry & re) const
{
    *fsm.dbgout << "considered" << json::array;
    for (const RuleEntry *r = fsm.rules.begin(); r != &re; ++r)
    {
        if (r->rule->preContext > fsm.slots.context())
            continue;
        *fsm.dbgout << json::flat << json::object
                    << "id" << r->rule - m_rules
                    << "failed" << true
                    << "input" << json::flat << json::object
                        << "start" << objectid(dslot(&fsm.slots.segment, input_slot(fsm.slots, -r->rule->preContext)))
                        << "length" << r->rule->sort
                        << json::close  // close "input"
                    << json::close; // close Rule object
    }
}


void Pass::dumpRuleEventOutput(const FiniteStateMachine & fsm, const Rule & r, Slot * const last_slot) const
{
    *fsm.dbgout     << json::item << json::flat << json::object
                        << "id"     << &r - m_rules
                        << "failed" << false
                        << "input" << json::flat << json::object
                            << "start" << objectid(dslot(&fsm.slots.segment, input_slot(fsm.slots, 0)))
                            << "length" << r.sort - r.preContext
                            << json::close // close "input"
                        << json::close  // close Rule object
                << json::close // close considered array
                << "output" << json::object
                    << "range" << json::flat << json::object
                        << "start"  << objectid(dslot(&fsm.slots.segment, input_slot(fsm.slots, 0)))
                        << "end"    << objectid(dslot(&fsm.slots.segment, last_slot))
                    << json::close // close "input"
                    << "slots"  << json::array;
    const Position rsb_prepos = last_slot ? last_slot->origin() : fsm.slots.segment.advance();
    fsm.slots.segment.positionSlots(0, 0, 0, fsm.slots.segment.currdir());

    for(Slot * slot = output_slot(fsm.slots, 0); slot != last_slot; slot = slot->next())
        *fsm.dbgout     << dslot(&fsm.slots.segment, slot);
    *fsm.dbgout         << json::close  // close "slots"
                    << "postshift"  << (last_slot ? last_slot->origin() : fsm.slots.segment.advance()) - rsb_prepos
                << json::close;         // close "output" object

}

#endif


inline
bool Pass::testPassConstraint(Machine & m) const
{
    if (!m_cPConstraint) return true;

    assert(m_cPConstraint.constraint());

    m.slotMap().reset(*m.slotMap().segment.first(), 0);
    m.slotMap().pushSlot(m.slotMap().segment.first());
    vm::slotref * map = m.slotMap().begin();
    const uint32 ret = m_cPConstraint.run(m, map);

#if !defined GRAPHITE2_NTRACING
    json * const dbgout = m.slotMap().segment.getFace()->logger();
    if (dbgout)
        *dbgout << "constraint" << (ret && m.status() == Machine::finished);
#endif

    return ret && m.status() == Machine::finished;
}


bool Pass::testConstraint(const Rule & r, Machine & m) const
{
    const uint16 curr_context = m.slotMap().context();
    if (unsigned(r.sort + curr_context - r.preContext) > m.slotMap().size()
        || curr_context - r.preContext < 0) return false;

    vm::slotref * map = m.slotMap().begin() + curr_context - r.preContext;
    if (map[r.sort - 1] == 0)
        return false;

    if (!*r.constraint) return true;
    assert(r.constraint->constraint());
    for (int n = r.sort; n && map; --n, ++map)
    {
        if (!*map) continue;
        const int32 ret = r.constraint->run(m, map);
        if (!ret || m.status() != Machine::finished)
            return false;
    }

    return true;
}


void SlotMap::collectGarbage(Slot * &aSlot)
{
    for(Slot **s = begin(), *const *const se = end() - 1; s != se; ++s) {
        Slot *& slot = *s;
        if(slot && (slot->isDeleted() || slot->isCopied()))
        {
            if (slot == aSlot)
                aSlot = slot->prev() ? slot->prev() : slot->next();
            segment.freeSlot(slot);
        }
    }
}



int Pass::doAction(const Code *codeptr, Slot * & slot_out, vm::Machine & m) const
{
    assert(codeptr);
    if (!*codeptr) return 0;
    SlotMap   & smap = m.slotMap();
    vm::slotref * map = &smap[smap.context()];
    smap.highpassed(false);

    int32 ret = codeptr->run(m, map);

    if (m.status() != Machine::finished)
    {
        slot_out = NULL;
        smap.highwater(0);
        return 0;
    }

    slot_out = *map;
    return ret;
}


void Pass::adjustSlot(int delta, Slot * & slot_out, SlotMap & smap) const
{
    if (!slot_out)
    {
        if (smap.highpassed() || slot_out == smap.highwater())
        {
            slot_out = smap.segment.last();
            ++delta;
            if (!smap.highwater() || smap.highwater() == slot_out)
                smap.highpassed(false);
        }
        else
        {
            slot_out = smap.segment.first();
            --delta;
        }
    }
    if (delta < 0)
    {
        while (++delta <= 0 && slot_out)
        {
            slot_out = slot_out->prev();
            if (smap.highpassed() && smap.highwater() == slot_out)
                smap.highpassed(false);
        }
    }
    else if (delta > 0)
    {
        while (--delta >= 0 && slot_out)
        {
            if (slot_out == smap.highwater() && slot_out)
                smap.highpassed(true);
            slot_out = slot_out->next();
        }
    }
}

bool Pass::collisionShift(Segment *seg, int dir, json * const dbgout) const
{
    ShiftCollider shiftcoll(dbgout);
    // bool isfirst = true;
    bool hasCollisions = false;
    Slot *start = seg->first();      // turn on collision fixing for the first slot
    Slot *end = NULL;
    bool moved = false;

#if !defined GRAPHITE2_NTRACING
    if (dbgout)
        *dbgout << "collisions" << json::array
            << json::flat << json::object << "num-loops" << m_numCollRuns << json::close;
#endif

    while (start)
    {
#if !defined GRAPHITE2_NTRACING
        if (dbgout)  *dbgout << json::object << "phase" << "1" << "moves" << json::array;
#endif
        hasCollisions = false;
        end = NULL;
        // phase 1 : position shiftable glyphs, ignoring kernable glyphs
        for (Slot *s = start; s; s = s->next())
        {
            const SlotCollision * c = seg->collisionInfo(s);
            if (start && (c->flags() & (SlotCollision::COLL_FIX | SlotCollision::COLL_KERN)) == SlotCollision::COLL_FIX
                      && !resolveCollisions(seg, s, start, shiftcoll, false, dir, moved, hasCollisions, dbgout))
                return false;
            if (s != start && (c->flags() & SlotCollision::COLL_END))
            {
                end = s->next();
                break;
            }
        }

#if !defined GRAPHITE2_NTRACING
        if (dbgout)
            *dbgout << json::close << json::close; // phase-1
#endif

        // phase 2 : loop until happy.
        for (int i = 0; i < m_numCollRuns - 1; ++i)
        {
            if (hasCollisions || moved)
            {

#if !defined GRAPHITE2_NTRACING
                if (dbgout)
                    *dbgout << json::object << "phase" << "2a" << "loop" << i << "moves" << json::array;
#endif
                // phase 2a : if any shiftable glyphs are in collision, iterate backwards,
                // fixing them and ignoring other non-collided glyphs. Note that this handles ONLY
                // glyphs that are actually in collision from phases 1 or 2b, and working backwards
                // has the intended effect of breaking logjams.
                if (hasCollisions)
                {
                    hasCollisions = false;
                    #if 0
                    moved = true;
                    for (Slot *s = start; s != end; s = s->next())
                    {
                        SlotCollision * c = seg->collisionInfo(s);
                        c->setShift(Position(0, 0));
                    }
                    #endif
                    Slot *lend = end ? end->prev() : seg->last();
                    Slot *lstart = start->prev();
                    for (Slot *s = lend; s != lstart; s = s->prev())
                    {
                        SlotCollision * c = seg->collisionInfo(s);
                        if (start && (c->flags() & (SlotCollision::COLL_FIX | SlotCollision::COLL_KERN | SlotCollision::COLL_ISCOL))
                                        == (SlotCollision::COLL_FIX | SlotCollision::COLL_ISCOL)) // ONLY if this glyph is still colliding
                        {
                            if (!resolveCollisions(seg, s, lend, shiftcoll, true, dir, moved, hasCollisions, dbgout))
                                return false;
                            c->setFlags(c->flags() | SlotCollision::COLL_TEMPLOCK);
                        }
                    }
                }

#if !defined GRAPHITE2_NTRACING
                if (dbgout)
                    *dbgout << json::close << json::close // phase 2a
                        << json::object << "phase" << "2b" << "loop" << i << "moves" << json::array;
#endif

                // phase 2b : redo basic diacritic positioning pass for ALL glyphs. Each successive loop adjusts
                // glyphs from their current adjusted position, which has the effect of gradually minimizing the
                // resulting adjustment; ie, the final result will be gradually closer to the original location.
                // Also it allows more flexibility in the final adjustment, since it is moving along the
                // possible 8 vectors from successively different starting locations.
                if (moved)
                {
                    moved = false;
                    for (Slot *s = start; s != end; s = s->next())
                    {
                        SlotCollision * c = seg->collisionInfo(s);
                        if (start && (c->flags() & (SlotCollision::COLL_FIX | SlotCollision::COLL_TEMPLOCK
                                                        | SlotCollision::COLL_KERN)) == SlotCollision::COLL_FIX
                                  && !resolveCollisions(seg, s, start, shiftcoll, false, dir, moved, hasCollisions, dbgout))
                            return false;
                        else if (c->flags() & SlotCollision::COLL_TEMPLOCK)
                            c->setFlags(c->flags() & ~SlotCollision::COLL_TEMPLOCK);
                    }
                }
        //      if (!hasCollisions) // no, don't leave yet because phase 2b will continue to improve things
        //          break;
#if !defined GRAPHITE2_NTRACING
                if (dbgout)
                    *dbgout << json::close << json::close; // phase 2
#endif
            }
        }
        if (!end)
            break;
        start = NULL;
        for (Slot *s = end->prev(); s; s = s->next())
        {
            if (seg->collisionInfo(s)->flags() & SlotCollision::COLL_START)
            {
                start = s;
                break;
            }
        }
    }
    return true;
}

bool Pass::collisionKern(Segment *seg, int dir, json * const dbgout) const
{
    Slot *start = seg->first();
    float ymin = 1e38f;
    float ymax = -1e38f;
    const GlyphCache &gc = seg->getFace()->glyphs();

    // phase 3 : handle kerning of clusters
#if !defined GRAPHITE2_NTRACING
    if (dbgout)
        *dbgout << json::object << "phase" << "3" << "moves" << json::array;
#endif

    for (Slot *s = seg->first(); s; s = s->next())
    {
        if (!gc.check(s->gid()))
            return false;
        const SlotCollision * c = seg->collisionInfo(s);
        const Rect &bbox = seg->theGlyphBBoxTemporary(s->gid());
        float y = s->origin().y + c->shift().y;
        if (!(c->flags() & SlotCollision::COLL_ISSPACE))
        {
            ymax = max(y + bbox.tr.y, ymax);
            ymin = min(y + bbox.bl.y, ymin);
        }
        if (start && (c->flags() & (SlotCollision::COLL_KERN | SlotCollision::COLL_FIX))
                        == (SlotCollision::COLL_KERN | SlotCollision::COLL_FIX))
            resolveKern(seg, s, start, dir, ymin, ymax, dbgout);
        if (c->flags() & SlotCollision::COLL_END)
            start = NULL;
        if (c->flags() & SlotCollision::COLL_START)
            start = s;
    }

#if !defined GRAPHITE2_NTRACING
    if (dbgout)
        *dbgout << json::close << json::close; // phase 3
#endif
    return true;
}

bool Pass::collisionFinish(Segment *seg, GR_MAYBE_UNUSED json * const dbgout) const
{
    for (Slot *s = seg->first(); s; s = s->next())
    {
        SlotCollision *c = seg->collisionInfo(s);
        if (c->shift().x != 0 || c->shift().y != 0)
        {
            const Position newOffset = c->shift();
            const Position nullPosition(0, 0);
            c->setOffset(newOffset + c->offset());
            c->setShift(nullPosition);
        }
    }
//    seg->positionSlots();

#if !defined GRAPHITE2_NTRACING
        if (dbgout)
            *dbgout << json::close;
#endif
    return true;
}

// Can slot s be kerned, or is it attached to something that can be kerned?
static bool inKernCluster(Segment *seg, Slot *s)
{
    SlotCollision *c = seg->collisionInfo(s);
    if (c->flags() & SlotCollision::COLL_KERN /** && c->flags() & SlotCollision::COLL_FIX **/ )
        return true;
    while (s->attachedTo())
    {
        s = s->attachedTo();
        c = seg->collisionInfo(s);
        if (c->flags() & SlotCollision::COLL_KERN /** && c->flags() & SlotCollision::COLL_FIX **/ )
            return true;
    }
    return false;
}

// Fix collisions for the given slot.
// Return true if everything was fixed, false if there are still collisions remaining.
// isRev means be we are processing backwards.
bool Pass::resolveCollisions(Segment *seg, Slot *slotFix, Slot *start,
        ShiftCollider &coll, GR_MAYBE_UNUSED bool isRev, int dir, bool &moved, bool &hasCol,
        json * const dbgout) const
{
    Slot * nbor;  // neighboring slot
    SlotCollision *cFix = seg->collisionInfo(slotFix);
    if (!coll.initSlot(seg, slotFix, cFix->limit(), cFix->margin(), cFix->marginWt(),
            cFix->shift(), cFix->offset(), dir, dbgout))
        return false;
    bool collides = false;
    // When we're processing forward, ignore kernable glyphs that preceed the target glyph.
    // When processing backward, don't ignore these until we pass slotFix.
    bool ignoreForKern = !isRev;
    bool rtl = dir & 1;
    Slot *base = slotFix;
    while (base->attachedTo())
        base = base->attachedTo();
    Position zero(0., 0.);

    // Look for collisions with the neighboring glyphs.
    for (nbor = start; nbor; nbor = isRev ? nbor->prev() : nbor->next())
    {
        SlotCollision *cNbor = seg->collisionInfo(nbor);
        bool sameCluster = nbor->isChildOf(base);
        if (nbor != slotFix         						// don't process if this is the slot of interest
                      && !(cNbor->ignore())    				// don't process if ignoring
                      && (nbor == base || sameCluster       // process if in the same cluster as slotFix
                            || !inKernCluster(seg, nbor))   // or this cluster is not to be kerned
//                            || (rtl ^ ignoreForKern))       // or it comes before(ltr) or after(rtl)
                      && (!isRev    // if processing forwards then good to merge otherwise only:
                            || !(cNbor->flags() & SlotCollision::COLL_FIX)     // merge in immovable stuff
                            || ((cNbor->flags() & SlotCollision::COLL_KERN) && !sameCluster)     // ignore other kernable clusters
                            || (cNbor->flags() & SlotCollision::COLL_ISCOL))   // test against other collided glyphs
                      && !coll.mergeSlot(seg, nbor, cNbor, cNbor->shift(), !ignoreForKern, sameCluster, collides, false, dbgout))
            return false;
        else if (nbor == slotFix)
            // Switching sides of this glyph - if we were ignoring kernable stuff before, don't anymore.
            ignoreForKern = !ignoreForKern;

        if (nbor != start && (cNbor->flags() & (isRev ? SlotCollision::COLL_START : SlotCollision::COLL_END)))
            break;
    }
    bool isCol = false;
    if (collides || cFix->shift().x != 0.f || cFix->shift().y != 0.f)
    {
        Position shift = coll.resolve(seg, isCol, dbgout);
        // isCol has been set to true if a collision remains.
        if (std::fabs(shift.x) < 1e38f && std::fabs(shift.y) < 1e38f)
        {
            if (sqr(shift.x-cFix->shift().x) + sqr(shift.y-cFix->shift().y) >= m_colThreshold * m_colThreshold)
                moved = true;
            cFix->setShift(shift);
            if (slotFix->firstChild())
            {
                Rect bbox;
                Position here = slotFix->origin() + shift;
                float clusterMin = here.x;
                slotFix->firstChild()->finalise(seg, NULL, here, bbox, 0, clusterMin, rtl, false);
            }
        }
    }
    else
    {
        // This glyph is not colliding with anything.
#if !defined GRAPHITE2_NTRACING
        if (dbgout)
        {
            *dbgout << json::object
                            << "missed" << objectid(dslot(seg, slotFix));
            coll.outputJsonDbg(dbgout, seg, -1);
            *dbgout << json::close;
        }
#endif
    }

    // Set the is-collision flag bit.
    if (isCol)
    { cFix->setFlags(cFix->flags() | SlotCollision::COLL_ISCOL | SlotCollision::COLL_KNOWN); }
    else
    { cFix->setFlags((cFix->flags() & ~SlotCollision::COLL_ISCOL) | SlotCollision::COLL_KNOWN); }
    hasCol |= isCol;
    return true;
}

float Pass::resolveKern(Segment *seg, Slot *slotFix, GR_MAYBE_UNUSED Slot *start, int dir,
    float &ymin, float &ymax, json *const dbgout) const
{
    Slot *nbor; // neighboring slot
    float currSpace = 0.;
    bool collides = false;
    unsigned int space_count = 0;
    Slot *base = slotFix;
    while (base->attachedTo())
        base = base->attachedTo();
    SlotCollision *cFix = seg->collisionInfo(base);
    const GlyphCache &gc = seg->getFace()->glyphs();
    const Rect &bbb = seg->theGlyphBBoxTemporary(slotFix->gid());
    const float by = slotFix->origin().y + cFix->shift().y;

    if (base != slotFix)
    {
        cFix->setFlags(cFix->flags() | SlotCollision::COLL_KERN | SlotCollision::COLL_FIX);
        return 0;
    }
    bool seenEnd = (cFix->flags() & SlotCollision::COLL_END) != 0;
    bool isInit = false;
    KernCollider coll(dbgout);

    ymax = max(by + bbb.tr.y, ymax);
    ymin = min(by + bbb.bl.y, ymin);
    for (nbor = slotFix->next(); nbor; nbor = nbor->next())
    {
        if (!gc.check(nbor->gid()))
            return 0.;
        const Rect &bb = seg->theGlyphBBoxTemporary(nbor->gid());
        SlotCollision *cNbor = seg->collisionInfo(nbor);
        const float nby = nbor->origin().y + cNbor->shift().y;
        if (nbor->isChildOf(base))
        {
            ymax = max(nby + bb.tr.y, ymax);
            ymin = min(nby + bb.bl.y, ymin);
            continue;
        }
        if ((bb.bl.y == 0.f && bb.tr.y == 0.f) || (cNbor->flags() & SlotCollision::COLL_ISSPACE))
        {
            if (m_kernColls == InWord)
                break;
            // Add space for a space glyph.
            currSpace += nbor->advance();
            ++space_count;
        }
        else
        {
            space_count = 0;
            if (nbor != slotFix && !cNbor->ignore())
            {
                seenEnd = true;
                if (!isInit)
                {
                    if (!coll.initSlot(seg, slotFix, cFix->limit(), cFix->margin(),
                                    cFix->shift(), cFix->offset(), dir, ymin, ymax, dbgout))
                        return 0.;
                    isInit = true;
                }
                collides |= coll.mergeSlot(seg, nbor, cNbor->shift(), currSpace, dir, dbgout);
            }
        }
        if (cNbor->flags() & SlotCollision::COLL_END)
        {
            if (seenEnd && space_count < 2)
                break;
            else
                seenEnd = true;
        }
    }
    if (collides)
    {
        Position mv = coll.resolve(seg, slotFix, dir, dbgout);
        coll.shift(mv, dir);
        Position delta = slotFix->advancePos() + mv - cFix->shift();
        slotFix->advance(delta);
        cFix->setShift(mv);
        return mv.x;
    }
    return 0.;
}
