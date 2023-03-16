// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2010, SIL International, All rights reserved.

// This class represents loaded graphite stack machine code.  It performs
// basic sanity checks, on the incoming code to prevent more obvious problems
// from crashing graphite.
// Author: Tim Eves

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include "graphite2/Segment.h"
#include "inc/Code.h"
#include "inc/Face.h"
#include "inc/GlyphFace.h"
#include "inc/GlyphCache.h"
#include "inc/Machine.h"
#include "inc/Rule.h"
#include "inc/Silf.h"

#include <cstdio>

#ifdef NDEBUG
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#endif


using namespace graphite2;
using namespace vm;

namespace {

inline bool is_return(const instr i) {
    const opcode_t * opmap = Machine::getOpcodeTable();
    const instr pop_ret  = *opmap[POP_RET].impl,
                ret_zero = *opmap[RET_ZERO].impl,
                ret_true = *opmap[RET_TRUE].impl;
    return i == pop_ret || i == ret_zero || i == ret_true;
}

struct context
{
    context(uint8 ref=0) : codeRef(ref) {flags.changed=false; flags.referenced=false;}
    struct {
        uint8   changed:1,
                referenced:1;
    } flags;
    uint8       codeRef;
};

} // end namespace


class Machine::Code::decoder
{
public:
    struct limits;
    static const int NUMCONTEXTS = 256;

    decoder(limits & lims, Code &code, enum passtype pt) throw();

    bool        load(const byte * bc_begin, const byte * bc_end);
    void        apply_analysis(instr * const code, instr * code_end);
    byte        max_ref() { return _max_ref; }
    int         out_index() const { return _out_index; }

private:
    void        set_ref(int index) throw();
    void        set_noref(int index) throw();
    void        set_changed(int index) throw();
    opcode      fetch_opcode(const byte * bc);
    void        analyse_opcode(const opcode, const int8 * const dp) throw();
    bool        emit_opcode(opcode opc, const byte * & bc);
    bool        validate_opcode(const byte opc, const byte * const bc);
    bool        valid_upto(const uint16 limit, const uint16 x) const throw();
    bool        test_context() const throw();
    bool        test_ref(int8 index) const throw();
    bool        test_attr(attrCode attr) const throw();
    void        failure(const status_t s) const throw() { _code.failure(s); }

    Code              & _code;
    int                 _out_index;
    uint16              _out_length;
    instr             * _instr;
    byte              * _data;
    limits            & _max;
    enum passtype       _passtype;
    int                 _stack_depth;
    bool                _in_ctxt_item;
    int16               _slotref;
    context             _contexts[NUMCONTEXTS];
    byte                _max_ref;
};


struct Machine::Code::decoder::limits
{
  const byte       * bytecode;
  const uint8        pre_context;
  const uint16       rule_length,
                     classes,
                     glyf_attrs,
                     features;
  const byte         attrid[gr_slatMax];
};

inline Machine::Code::decoder::decoder(limits & lims, Code &code, enum passtype pt) throw()
: _code(code),
  _out_index(code._constraint ? 0 : lims.pre_context),
  _out_length(code._constraint ? 1 : lims.rule_length),
  _instr(code._code), _data(code._data), _max(lims), _passtype(pt),
  _stack_depth(0),
  _in_ctxt_item(false),
  _slotref(0),
  _max_ref(0)
{ }



Machine::Code::Code(bool is_constraint, const byte * bytecode_begin, const byte * const bytecode_end,
           uint8 pre_context, uint16 rule_length, const Silf & silf, const Face & face,
           enum passtype pt, byte * * const _out)
 :  _code(0), _data(0), _data_size(0), _instr_count(0), _max_ref(0), _status(loaded),
    _constraint(is_constraint), _modify(false), _delete(false), _own(_out==0)
{
#ifdef GRAPHITE2_TELEMETRY
    telemetry::category _code_cat(face.tele.code);
#endif
    assert(bytecode_begin != 0);
    if (bytecode_begin == bytecode_end)
    {
      // ::new (this) Code();
      return;
    }
    assert(bytecode_end > bytecode_begin);
    const opcode_t *    op_to_fn = Machine::getOpcodeTable();

    // Allocate code and data target buffers, these sizes are a worst case
    // estimate.  Once we know their real sizes the we'll shrink them.
    if (_out)   _code = reinterpret_cast<instr *>(*_out);
    else        _code = static_cast<instr *>(malloc(estimateCodeDataOut(bytecode_end-bytecode_begin, 1, is_constraint ? 0 : rule_length)));
    _data = reinterpret_cast<byte *>(_code + (bytecode_end - bytecode_begin));

    if (!_code || !_data) {
        failure(alloc_failed);
        return;
    }

    decoder::limits lims = {
        bytecode_end,
        pre_context,
        rule_length,
        silf.numClasses(),
        face.glyphs().numAttrs(),
        face.numFeatures(),
        {1,1,1,1,1,1,1,1,
         1,1,1,1,1,1,1,255,
         1,1,1,1,1,1,1,1,
         1,1,1,1,1,1,0,0,
         0,0,0,0,0,0,0,0,
         0,0,0,0,0,0,0,0,
         0,0,0,0,0,0,0, silf.numUser()}
    };

    decoder dec(lims, *this, pt);
    if(!dec.load(bytecode_begin, bytecode_end))
       return;

    // Is this an empty program?
    if (_instr_count == 0)
    {
      release_buffers();
      ::new (this) Code();
      return;
    }

    // When we reach the end check we've terminated it correctly
    if (!is_return(_code[_instr_count-1])) {
        failure(missing_return);
        return;
    }

    assert((_constraint && immutable()) || !_constraint);
    dec.apply_analysis(_code, _code + _instr_count);
    _max_ref = dec.max_ref();

    // Now we know exactly how much code and data the program really needs
    // realloc the buffers to exactly the right size so we don't waste any
    // memory.
    assert((bytecode_end - bytecode_begin) >= ptrdiff_t(_instr_count));
    assert((bytecode_end - bytecode_begin) >= ptrdiff_t(_data_size));
    memmove(_code + (_instr_count+1), _data, _data_size*sizeof(byte));
    size_t const total_sz = ((_instr_count+1) + (_data_size + sizeof(instr)-1)/sizeof(instr))*sizeof(instr);
    if (_out)
        *_out += total_sz;
    else
    {
      instr * const old_code = _code;
      _code = static_cast<instr *>(realloc(_code, total_sz));
      if (!_code) free(old_code);
    }
   _data = reinterpret_cast<byte *>(_code + (_instr_count+1));

    if (!_code)
    {
        failure(alloc_failed);
        return;
    }

    // Make this RET_ZERO, we should never reach this but just in case ...
    _code[_instr_count] = op_to_fn[RET_ZERO].impl[_constraint];

#ifdef GRAPHITE2_TELEMETRY
    telemetry::count_bytes(_data_size + (_instr_count+1)*sizeof(instr));
#endif
}

Machine::Code::~Code() throw ()
{
    if (_own)
        release_buffers();
}


bool Machine::Code::decoder::load(const byte * bc, const byte * bc_end)
{
    _max.bytecode = bc_end;
    while (bc < bc_end)
    {
        const opcode opc = fetch_opcode(bc++);
        if (opc == vm::MAX_OPCODE)
            return false;

        analyse_opcode(opc, reinterpret_cast<const int8 *>(bc));

        if (!emit_opcode(opc, bc))
            return false;
    }

    return bool(_code);
}

// Validation check and fixups.
//

opcode Machine::Code::decoder::fetch_opcode(const byte * bc)
{
    const byte opc = *bc++;

    // Do some basic sanity checks based on what we know about the opcode
    if (!validate_opcode(opc, bc))  return MAX_OPCODE;

    // And check its arguments as far as possible
    switch (opcode(opc))
    {
        case NOP :
            break;
        case PUSH_BYTE :
        case PUSH_BYTEU :
        case PUSH_SHORT :
        case PUSH_SHORTU :
        case PUSH_LONG :
            ++_stack_depth;
            break;
        case ADD :
        case SUB :
        case MUL :
        case DIV :
        case MIN_ :
        case MAX_ :
        case AND :
        case OR :
        case EQUAL :
        case NOT_EQ :
        case LESS :
        case GTR :
        case LESS_EQ :
        case GTR_EQ :
        case BITOR :
        case BITAND :
            if (--_stack_depth <= 0)
                failure(underfull_stack);
            break;
        case NEG :
        case TRUNC8 :
        case TRUNC16 :
        case NOT :
        case BITNOT :
        case BITSET :
            if (_stack_depth <= 0)
                failure(underfull_stack);
            break;
        case COND :
            _stack_depth -= 2;
            if (_stack_depth <= 0)
                failure(underfull_stack);
            break;
        case NEXT_N :           // runtime checked
            break;
        case NEXT :
        case COPY_NEXT :
            ++_out_index;
            if (_out_index < -1 || _out_index > _out_length || _slotref > _max.rule_length)
                failure(out_of_range_data);
            break;
        case PUT_GLYPH_8BIT_OBS :
            valid_upto(_max.classes, bc[0]);
            test_context();
            break;
        case PUT_SUBS_8BIT_OBS :
            test_ref(int8(bc[0]));
            valid_upto(_max.classes, bc[1]);
            valid_upto(_max.classes, bc[2]);
            test_context();
            break;
        case PUT_COPY :
            test_ref(int8(bc[0]));
            test_context();
            break;
        case INSERT :
            if (_passtype >= PASS_TYPE_POSITIONING)
                failure(invalid_opcode);
            ++_out_length;
            if (_out_index < 0) ++_out_index;
            if (_out_index < -1 || _out_index >= _out_length)
                failure(out_of_range_data);
            break;
        case DELETE :
            if (_passtype >= PASS_TYPE_POSITIONING)
                failure(invalid_opcode);
            if (_out_index < _max.pre_context)
                failure(out_of_range_data);
            --_out_index;
            --_out_length;
            if (_out_index < -1 || _out_index > _out_length)
                failure(out_of_range_data);
            break;
        case ASSOC :
            if (bc[0] == 0)
                failure(out_of_range_data);
            for (uint8 num = bc[0]; num; --num)
                test_ref(int8(bc[num]));
            test_context();
            break;
        case CNTXT_ITEM :
            valid_upto(_max.rule_length, _max.pre_context + int8(bc[0]));
            if (bc + 2 + bc[1] >= _max.bytecode)    failure(jump_past_end);
            if (_in_ctxt_item)                      failure(nested_context_item);
            break;
        case ATTR_SET :
        case ATTR_ADD :
        case ATTR_SUB :
        case ATTR_SET_SLOT :
            if (--_stack_depth < 0)
                failure(underfull_stack);
            valid_upto(gr_slatMax, bc[0]);
            if (attrCode(bc[0]) == gr_slatUserDefn)     // use IATTR for user attributes
                failure(out_of_range_data);
            test_attr(attrCode(bc[0]));
            test_context();
            break;
        case IATTR_SET_SLOT :
            if (--_stack_depth < 0)
                failure(underfull_stack);
            if (valid_upto(gr_slatMax, bc[0]))
                valid_upto(_max.attrid[bc[0]], bc[1]);
            test_attr(attrCode(bc[0]));
            test_context();
            break;
        case PUSH_SLOT_ATTR :
            ++_stack_depth;
            valid_upto(gr_slatMax, bc[0]);
            test_ref(int8(bc[1]));
            if (attrCode(bc[0]) == gr_slatUserDefn)     // use IATTR for user attributes
                failure(out_of_range_data);
            test_attr(attrCode(bc[0]));
            break;
        case PUSH_GLYPH_ATTR_OBS :
        case PUSH_ATT_TO_GATTR_OBS :
            ++_stack_depth;
            valid_upto(_max.glyf_attrs, bc[0]);
            test_ref(int8(bc[1]));
            break;
        case PUSH_ATT_TO_GLYPH_METRIC :
        case PUSH_GLYPH_METRIC :
            ++_stack_depth;
            valid_upto(kgmetDescent, bc[0]);
            test_ref(int8(bc[1]));
            // level: dp[2] no check necessary
            break;
        case PUSH_FEAT :
            ++_stack_depth;
            valid_upto(_max.features, bc[0]);
            test_ref(int8(bc[1]));
            break;
        case PUSH_ISLOT_ATTR :
            ++_stack_depth;
            if (valid_upto(gr_slatMax, bc[0]))
            {
                test_ref(int8(bc[1]));
                valid_upto(_max.attrid[bc[0]], bc[2]);
            }
            test_attr(attrCode(bc[0]));
            break;
        case PUSH_IGLYPH_ATTR :// not implemented
            ++_stack_depth;
            break;
        case POP_RET :
            if (--_stack_depth < 0)
                failure(underfull_stack);
            GR_FALLTHROUGH;
            // no break
        case RET_ZERO :
        case RET_TRUE :
            break;
        case IATTR_SET :
        case IATTR_ADD :
        case IATTR_SUB :
            if (--_stack_depth < 0)
                failure(underfull_stack);
            if (valid_upto(gr_slatMax, bc[0]))
                valid_upto(_max.attrid[bc[0]], bc[1]);
            test_attr(attrCode(bc[0]));
            test_context();
            break;
        case PUSH_PROC_STATE :  // dummy: dp[0] no check necessary
        case PUSH_VERSION :
            ++_stack_depth;
            break;
        case PUT_SUBS :
            test_ref(int8(bc[0]));
            valid_upto(_max.classes, uint16(bc[1]<< 8) | bc[2]);
            valid_upto(_max.classes, uint16(bc[3]<< 8) | bc[4]);
            test_context();
            break;
        case PUT_SUBS2 :        // not implemented
        case PUT_SUBS3 :        // not implemented
            break;
        case PUT_GLYPH :
            valid_upto(_max.classes, uint16(bc[0]<< 8) | bc[1]);
            test_context();
            break;
        case PUSH_GLYPH_ATTR :
        case PUSH_ATT_TO_GLYPH_ATTR :
            ++_stack_depth;
            valid_upto(_max.glyf_attrs, uint16(bc[0]<< 8) | bc[1]);
            test_ref(int8(bc[2]));
            break;
        case SET_FEAT :
            valid_upto(_max.features, bc[0]);
            test_ref(int8(bc[1]));
            break;
        default:
            failure(invalid_opcode);
            break;
    }

    return bool(_code) ? opcode(opc) : MAX_OPCODE;
}


void Machine::Code::decoder::analyse_opcode(const opcode opc, const int8  * arg) throw()
{
  switch (opc)
  {
    case DELETE :
      _code._delete = true;
      break;
    case ASSOC :
      set_changed(0);
//      for (uint8 num = arg[0]; num; --num)
//        _analysis.set_noref(num);
      break;
    case PUT_GLYPH_8BIT_OBS :
    case PUT_GLYPH :
      _code._modify = true;
      set_changed(0);
      break;
    case ATTR_SET :
    case ATTR_ADD :
    case ATTR_SUB :
    case ATTR_SET_SLOT :
    case IATTR_SET_SLOT :
    case IATTR_SET :
    case IATTR_ADD :
    case IATTR_SUB :
      set_noref(0);
      break;
    case NEXT :
    case COPY_NEXT :
      ++_slotref;
      _contexts[_slotref] = context(uint8(_code._instr_count+1));
      // if (_analysis.slotref > _analysis.max_ref) _analysis.max_ref = _analysis.slotref;
      break;
    case INSERT :
      if (_slotref >= 0) --_slotref;
      _code._modify = true;
      break;
    case PUT_SUBS_8BIT_OBS :    // slotref on 1st parameter
    case PUT_SUBS :
      _code._modify = true;
      set_changed(0);
      GR_FALLTHROUGH;
      // no break
    case PUT_COPY :
      if (arg[0] != 0) { set_changed(0); _code._modify = true; }
      set_ref(arg[0]);
      break;
    case PUSH_GLYPH_ATTR_OBS :
    case PUSH_SLOT_ATTR :
    case PUSH_GLYPH_METRIC :
    case PUSH_ATT_TO_GATTR_OBS :
    case PUSH_ATT_TO_GLYPH_METRIC :
    case PUSH_ISLOT_ATTR :
    case PUSH_FEAT :
    case SET_FEAT :
      set_ref(arg[1]);
      break;
    case PUSH_ATT_TO_GLYPH_ATTR :
    case PUSH_GLYPH_ATTR :
      set_ref(arg[2]);
      break;
    default:
        break;
  }
}


bool Machine::Code::decoder::emit_opcode(opcode opc, const byte * & bc)
{
    const opcode_t * op_to_fn = Machine::getOpcodeTable();
    const opcode_t & op       = op_to_fn[opc];
    if (op.impl[_code._constraint] == 0)
    {
        failure(unimplemented_opcode_used);
        return false;
    }

    const size_t     param_sz = op.param_sz == VARARGS ? bc[0] + 1 : op.param_sz;

    // Add this instruction
    *_instr++ = op.impl[_code._constraint];
    ++_code._instr_count;

    // Grab the parameters
    if (param_sz) {
        memcpy(_data, bc, param_sz * sizeof(byte));
        bc               += param_sz;
        _data            += param_sz;
        _code._data_size += param_sz;
    }

    // recursively decode a context item so we can split the skip into
    // instruction and data portions.
    if (opc == CNTXT_ITEM)
    {
        assert(_out_index == 0);
        _in_ctxt_item = true;
        _out_index = _max.pre_context + int8(_data[-2]);
        _slotref = int8(_data[-2]);
        _out_length = _max.rule_length;

        const size_t ctxt_start = _code._instr_count;
        byte & instr_skip = _data[-1];
        byte & data_skip  = *_data++;
        ++_code._data_size;
        const byte *curr_end = _max.bytecode;

        if (load(bc, bc + instr_skip))
        {
            bc += instr_skip;
            data_skip  = instr_skip - byte(_code._instr_count - ctxt_start);
            instr_skip =  byte(_code._instr_count - ctxt_start);
            _max.bytecode = curr_end;

            _out_length = 1;
            _out_index = 0;
            _slotref = 0;
            _in_ctxt_item = false;
        }
        else
        {
            _out_index = 0;
            _slotref = 0;
            return false;
        }
    }

    return bool(_code);
}


void Machine::Code::decoder::apply_analysis(instr * const code, instr * code_end)
{
    // insert TEMP_COPY commands for slots that need them (that change and are referenced later)
    int tempcount = 0;
    if (_code._constraint) return;

    const instr temp_copy = Machine::getOpcodeTable()[TEMP_COPY].impl[0];
    for (const context * c = _contexts, * const ce = c + _slotref; c < ce; ++c)
    {
        if (!c->flags.referenced || !c->flags.changed) continue;

        instr * const tip = code + c->codeRef + tempcount;
        memmove(tip+1, tip, (code_end - tip) * sizeof(instr));
        *tip = temp_copy;
        ++code_end;
        ++tempcount;
        _code._delete = true;
    }

    _code._instr_count = code_end - code;
}


inline
bool Machine::Code::decoder::validate_opcode(const byte opc, const byte * const bc)
{
    if (opc >= MAX_OPCODE)
    {
        failure(invalid_opcode);
        return false;
    }
    const opcode_t & op = Machine::getOpcodeTable()[opc];
    if (op.impl[_code._constraint] == 0)
    {
        failure(unimplemented_opcode_used);
        return false;
    }
    if (op.param_sz == VARARGS && bc >= _max.bytecode)
    {
        failure(arguments_exhausted);
        return false;
    }
    const size_t param_sz = op.param_sz == VARARGS ? bc[0] + 1 : op.param_sz;
    if (bc - 1 + param_sz >= _max.bytecode)
    {
        failure(arguments_exhausted);
        return false;
    }
    return true;
}


bool Machine::Code::decoder::valid_upto(const uint16 limit, const uint16 x) const throw()
{
    const bool t = (limit != 0) && (x < limit);
    if (!t) failure(out_of_range_data);
    return t;
}

inline
bool Machine::Code::decoder::test_ref(int8 index) const throw()
{
    if (_code._constraint && !_in_ctxt_item)
    {
        if (index > 0 || -index > _max.pre_context)
        {
            failure(out_of_range_data);
            return false;
        }
    }
    else
    {
      if (_max.rule_length == 0
          || (_slotref + _max.pre_context + index >= _max.rule_length)
          || (_slotref + _max.pre_context + index < 0))
      {
        failure(out_of_range_data);
        return false;
      }
    }
    return true;
}

bool Machine::Code::decoder::test_context() const throw()
{
    if (_out_index >= _out_length || _out_index < 0 || _slotref >= NUMCONTEXTS - 1)
    {
        failure(out_of_range_data);
        return false;
    }
    return true;
}

bool Machine::Code::decoder::test_attr(attrCode) const throw()
{
#if 0   // This code is coming but causes backward compatibility problems.
    if (_passtype < PASS_TYPE_POSITIONING)
    {
        if (attr != gr_slatBreak && attr != gr_slatDir && attr != gr_slatUserDefn
                                 && attr != gr_slatCompRef)
        {
            failure(out_of_range_data);
            return false;
        }
    }
#endif
    return true;
}

inline
void Machine::Code::failure(const status_t s) throw() {
    release_buffers();
    _status = s;
}


inline
void Machine::Code::decoder::set_ref(int index) throw() {
    if (index + _slotref < 0 || index + _slotref >= NUMCONTEXTS) return;
    _contexts[index + _slotref].flags.referenced = true;
    if (index + _slotref > _max_ref) _max_ref = index + _slotref;
}


inline
void Machine::Code::decoder::set_noref(int index) throw() {
    if (index + _slotref < 0 || index + _slotref >= NUMCONTEXTS) return;
    if (index + _slotref > _max_ref) _max_ref = index + _slotref;
}


inline
void Machine::Code::decoder::set_changed(int index) throw() {
    if (index + _slotref < 0 || index + _slotref >= NUMCONTEXTS) return;
    _contexts[index + _slotref].flags.changed= true;
    if (index + _slotref > _max_ref) _max_ref = index + _slotref;
}


void Machine::Code::release_buffers() throw()
{
    if (_own)
        free(_code);
    _code = 0;
    _data = 0;
    _own  = false;
}


int32 Machine::Code::run(Machine & m, slotref * & map) const
{
//    assert(_own);
    assert(*this);          // Check we are actually runnable

    if (m.slotMap().size() <= size_t(_max_ref + m.slotMap().context())
        || m.slotMap()[_max_ref + m.slotMap().context()] == 0)
    {
        m._status = Machine::slot_offset_out_bounds;
        return 1;
//        return m.run(_code, _data, map);
    }

    return  m.run(_code, _data, map);
}
