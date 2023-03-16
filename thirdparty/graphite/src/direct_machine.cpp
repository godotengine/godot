// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2010, SIL International, All rights reserved.

// This direct threaded interpreter implmentation for machine.h
// Author: Tim Eves

// Build either this interpreter or the call_machine implementation.
// The direct threaded interpreter is relies upon a gcc feature called
// labels-as-values so is only portable to compilers that support the
// extension (gcc only as far as I know) however it should build on any
// architecture gcc supports.
// This is twice as fast as the call threaded model and is likely faster on
// inorder processors with short pipelines and little branch prediction such
// as the ARM and possibly Atom chips.


#include <cassert>
#include <cstring>
#include "inc/Machine.h"
#include "inc/Segment.h"
#include "inc/Slot.h"
#include "inc/Rule.h"

#define STARTOP(name)           name: {
#define ENDOP                   }; goto *((sp - sb)/Machine::STACK_MAX ? &&end : *++ip);
#define EXIT(status)            { push(status); goto end; }

#define do_(name)               &&name


using namespace graphite2;
using namespace vm;

namespace {

// The GCC manual has this to say about labels as values:
//   The &&foo expressions for the same label might have different values
//   if the containing function is inlined or cloned. If a program relies
//   on them being always the same, __attribute__((__noinline__,__noclone__))
//   should be used to prevent inlining and cloning.
//
// is_return in Code.cpp relies on being able to do comparisons, so it needs
// them to be always the same.
//
// The GCC manual further adds:
//   If &&foo is used in a static variable initializer, inlining and
//   cloning is forbidden.
//
// In this file, &&foo *is* used in a static variable initializer, and it's not
// entirely clear whether this should prevent inlining of the function or not.
// In practice, though, clang 7 can end up inlining the function with ThinLTO,
// which breaks at least is_return. https://bugs.llvm.org/show_bug.cgi?id=39241
// So all in all, we need at least the __noinline__ attribute. __noclone__
// is not supported by clang.
__attribute__((__noinline__))
const void * direct_run(const bool          get_table_mode,
                        const instr       * program,
                        const byte        * data,
                        Machine::stack_t  * stack,
                        slotref         * & __map,
                        uint8                _dir,
                        Machine::status_t & status,
                        SlotMap           * __smap=0)
{
    // We need to define and return to opcode table from within this function
    // other inorder to take the addresses of the instruction bodies.
    #include "inc/opcode_table.h"
    if (get_table_mode)
        return opcode_table;

    // Declare virtual machine registers
    const instr           * ip = program;
    const byte            * dp = data;
    Machine::stack_t      * sp = stack + Machine::STACK_GUARD,
                    * const sb = sp;
    SlotMap             & smap = *__smap;
    Segment              & seg = smap.segment;
    slotref                 is = *__map,
                         * map = __map,
                  * const mapb = smap.begin()+smap.context();
    uint8                  dir = _dir;
    int8                 flags = 0;

    // start the program
    goto **ip;

    // Pull in the opcode definitions
    #include "inc/opcodes.h"

    end:
    __map  = map;
    *__map = is;
    return sp;
}

}

const opcode_t * Machine::getOpcodeTable() throw()
{
    slotref * dummy;
    Machine::status_t dumstat = Machine::finished;
    return static_cast<const opcode_t *>(direct_run(true, 0, 0, 0, dummy, 0, dumstat));
}


Machine::stack_t  Machine::run(const instr   * program,
                               const byte    * data,
                               slotref     * & is)
{
    assert(program != 0);

    const stack_t *sp = static_cast<const stack_t *>(
                direct_run(false, program, data, _stack, is, _map.dir(), _status, &_map));
    const stack_t ret = sp == _stack+STACK_GUARD+1 ? *sp-- : 0;
    check_final_stack(sp);
    return ret;
}
