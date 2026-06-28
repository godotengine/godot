// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2010, SIL International, All rights reserved.

// This call threaded interpreter implmentation for machine.h
// Author: Tim Eves

// Build either this interpreter or the direct_machine implementation.
// The call threaded interpreter is portable across compilers and
// architectures as well as being useful to debug (you can set breakpoints on
// opcodes) but is slower that the direct threaded interpreter by a factor of 2

#include <cassert>
#include <cstring>
#include <graphite2/Segment.h>
#include "inc/Machine.h"
#include "inc/Segment.h"
#include "inc/Slot.h"
#include "inc/Rule.h"

// Disable the unused parameter warning as th compiler is mistaken since dp
// is always updated (even if by 0) on every opcode.
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#define registers           const byte * & dp, vm::Machine::stack_t * & sp, \
                            vm::Machine::stack_t * const sb, regbank & reg

// These are required by opcodes.h and should not be changed
#define STARTOP(name)       bool name(registers) REGPARM(4);\
                            bool name(registers) {
#define ENDOP                   return (sp - sb)/Machine::STACK_MAX==0; \
                            }

#define EXIT(status)        { push(status); return false; }

// This is required by opcode_table.h
#define do_(name)           instr(name)


using namespace graphite2;
using namespace vm;

struct regbank  {
    slotref         is;
    slotref *       map;
    SlotMap       & smap;
    slotref * const map_base;
    const instr * & ip;
    uint8           direction;
    int8            flags;
    Machine::status_t & status;
};

typedef bool        (* ip_t)(registers);

// Pull in the opcode definitions
// We pull these into a private namespace so these otherwise common names dont
// pollute the toplevel namespace.
namespace {
#define smap    reg.smap
#define seg     smap.segment
#define is      reg.is
#define ip      reg.ip
#define map     reg.map
#define mapb    reg.map_base
#define flags   reg.flags
#define dir     reg.direction
#define status  reg.status

#include "inc/opcodes.h"

#undef smap
#undef seg
#undef is
#undef ip
#undef map
#undef mapb
#undef flags
#undef dir
}

Machine::stack_t  Machine::run(const instr   * program,
                               const byte    * data,
                               slotref     * & map)

{
    assert(program != 0);

    // Declare virtual machine registers
    const instr   * ip = program-1;
    const byte    * dp = data;
    stack_t       * sp = _stack + Machine::STACK_GUARD,
            * const sb = sp;
    regbank         reg = {*map, map, _map, _map.begin()+_map.context(), ip, _map.dir(), 0, _status};

    // Run the program
    while ((reinterpret_cast<ip_t>(*++ip))(dp, sp, sb, reg)) {}
    const stack_t ret = sp == _stack+STACK_GUARD+1 ? *sp-- : 0;

    check_final_stack(sp);
    map = reg.map;
    *map = reg.is;
    return ret;
}

// Pull in the opcode table
namespace {
#include "inc/opcode_table.h"
}

const opcode_t * Machine::getOpcodeTable() throw()
{
    return opcode_table;
}
