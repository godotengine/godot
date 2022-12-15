// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2010, SIL International, All rights reserved.

#pragma once
// This file will be pulled into and integrated into a machine implmentation
// DO NOT build directly and under no circumstances ever #include headers in
// here or you will break the direct_machine.
//
// Implementers' notes
// ==================
// You have access to a few primitives and the full C++ code:
//    declare_params(n) Tells the interpreter how many bytes of parameter
//                      space to claim for this instruction uses and
//                      initialises the param pointer.  You *must* before the
//                      first use of param.
//    use_params(n)     Claim n extra bytes of param space beyond what was
//                      claimed using delcare_param.
//    param             A const byte pointer for the parameter space claimed by
//                      this instruction.
//    binop(op)         Implement a binary operation on the stack using the
//                      specified C++ operator.
//    NOT_IMPLEMENTED   Any instruction body containing this will exit the
//                      program with an assertion error.  Instructions that are
//                      not implemented should also be marked NILOP in the
//                      opcodes tables this will cause the code class to spot
//                      them in a live code stream and throw a runtime_error
//                      instead.
//    push(n)           Push the value n onto the stack.
//    pop()             Pop the top most value and return it.
//
//    You have access to the following named fast 'registers':
//        sp        = The pointer to the current top of stack, the last value
//                    pushed.
//        seg       = A reference to the Segment this code is running over.
//        is        = The current slot index
//        isb       = The original base slot index at the start of this rule
//        isf       = The first positioned slot
//        isl       = The last positioned slot
//        ip        = The current instruction pointer
//        endPos    = Position of advance of last cluster
//        dir       = writing system directionality of the font


// #define NOT_IMPLEMENTED     assert(false)
// #define NOT_IMPLEMENTED

#define binop(op)           const uint32 a = pop(); *sp = uint32(*sp) op a
#define sbinop(op)          const int32 a = pop(); *sp = int32(*sp) op a
#define use_params(n)       dp += n

#define declare_params(n)   const byte * param = dp; \
                            use_params(n);

#define push(n)             { *++sp = n; }
#define pop()               (*sp--)
#define slotat(x)           (map[(x)])
#define DIE                 { is=seg.last(); status = Machine::died_early; EXIT(1); }
#define POSITIONED          1

STARTOP(nop)
    do {} while (0);
ENDOP

STARTOP(push_byte)
    declare_params(1);
    push(int8(*param));
ENDOP

STARTOP(push_byte_u)
    declare_params(1);
    push(uint8(*param));
ENDOP

STARTOP(push_short)
    declare_params(2);
    const int16 r   = int16(param[0]) << 8
                    | uint8(param[1]);
    push(r);
ENDOP

STARTOP(push_short_u)
    declare_params(2);
    const uint16 r  = uint16(param[0]) << 8
                    | uint8(param[1]);
    push(r);
ENDOP

STARTOP(push_long)
    declare_params(4);
    const  int32 r  = int32(param[0]) << 24
                    | uint32(param[1]) << 16
                    | uint32(param[2]) << 8
                    | uint8(param[3]);
    push(r);
ENDOP

STARTOP(add)
    binop(+);
ENDOP

STARTOP(sub)
    binop(-);
ENDOP

STARTOP(mul)
    binop(*);
ENDOP

STARTOP(div_)
    const int32 b = pop();
    const int32 a = int32(*sp);
    if (b == 0 || (a == std::numeric_limits<int32>::min() && b == -1)) DIE;
    *sp = int32(*sp) / b;
ENDOP

STARTOP(min_)
    const int32 a = pop(), b = *sp;
    if (a < b) *sp = a;
ENDOP

STARTOP(max_)
    const int32 a = pop(), b = *sp;
    if (a > b) *sp = a;
ENDOP

STARTOP(neg)
    *sp = uint32(-int32(*sp));
ENDOP

STARTOP(trunc8)
    *sp = uint8(*sp);
ENDOP

STARTOP(trunc16)
    *sp = uint16(*sp);
ENDOP

STARTOP(cond)
    const uint32 f = pop(), t = pop(), c = pop();
    push(c ? t : f);
ENDOP

STARTOP(and_)
    binop(&&);
ENDOP

STARTOP(or_)
    binop(||);
ENDOP

STARTOP(not_)
    *sp = !*sp;
ENDOP

STARTOP(equal)
    binop(==);
ENDOP

STARTOP(not_eq_)
    binop(!=);
ENDOP

STARTOP(less)
    sbinop(<);
ENDOP

STARTOP(gtr)
    sbinop(>);
ENDOP

STARTOP(less_eq)
    sbinop(<=);
ENDOP

STARTOP(gtr_eq)
    sbinop(>=);
ENDOP

STARTOP(next)
    if (map - &smap[0] >= int(smap.size())) DIE
    if (is)
    {
        if (is == smap.highwater())
            smap.highpassed(true);
        is = is->next();
    }
    ++map;
ENDOP

//STARTOP(next_n)
//    use_params(1);
//    NOT_IMPLEMENTED;
    //declare_params(1);
    //const size_t num = uint8(*param);
//ENDOP

//STARTOP(copy_next)
//     if (is) is = is->next();
//     ++map;
// ENDOP

STARTOP(put_glyph_8bit_obs)
    declare_params(1);
    const unsigned int output_class = uint8(*param);
    is->setGlyph(&seg, seg.getClassGlyph(output_class, 0));
ENDOP

STARTOP(put_subs_8bit_obs)
    declare_params(3);
    const int           slot_ref     = int8(param[0]);
    const unsigned int  input_class  = uint8(param[1]),
                        output_class = uint8(param[2]);
    uint16 index;
    slotref slot = slotat(slot_ref);
    if (slot)
    {
        index = seg.findClassIndex(input_class, slot->gid());
        is->setGlyph(&seg, seg.getClassGlyph(output_class, index));
    }
ENDOP

STARTOP(put_copy)
    declare_params(1);
    const int  slot_ref = int8(*param);
    if (is && !is->isDeleted())
    {
        slotref ref = slotat(slot_ref);
        if (ref && ref != is)
        {
            int16 *tempUserAttrs = is->userAttrs();
            if (is->attachedTo() || is->firstChild()) DIE
            Slot *prev = is->prev();
            Slot *next = is->next();
            memcpy(tempUserAttrs, ref->userAttrs(), seg.numAttrs() * sizeof(uint16));
            memcpy(is, ref, sizeof(Slot));
            is->firstChild(NULL);
            is->nextSibling(NULL);
            is->userAttrs(tempUserAttrs);
            is->next(next);
            is->prev(prev);
            if (is->attachedTo())
                is->attachedTo()->child(is);
        }
        is->markCopied(false);
        is->markDeleted(false);
    }
ENDOP

STARTOP(insert)
    if (smap.decMax() <= 0) DIE;
    Slot *newSlot = seg.newSlot();
    if (!newSlot) DIE;
    Slot *iss = is;
    while (iss && iss->isDeleted()) iss = iss->next();
    if (!iss)
    {
        if (seg.last())
        {
            seg.last()->next(newSlot);
            newSlot->prev(seg.last());
            newSlot->before(seg.last()->before());
            seg.last(newSlot);
        }
        else
        {
            seg.first(newSlot);
            seg.last(newSlot);
        }
    }
    else if (iss->prev())
    {
        iss->prev()->next(newSlot);
        newSlot->prev(iss->prev());
        newSlot->before(iss->prev()->after());
    }
    else
    {
        newSlot->prev(NULL);
        newSlot->before(iss->before());
        seg.first(newSlot);
    }
    newSlot->next(iss);
    if (iss)
    {
        iss->prev(newSlot);
        newSlot->originate(iss->original());
        newSlot->after(iss->before());
    }
    else if (newSlot->prev())
    {
        newSlot->originate(newSlot->prev()->original());
        newSlot->after(newSlot->prev()->after());
    }
    else
    {
        newSlot->originate(seg.defaultOriginal());
    }
    if (is == smap.highwater())
        smap.highpassed(false);
    is = newSlot;
    seg.extendLength(1);
    if (map != &smap[-1])
        --map;
ENDOP

STARTOP(delete_)
    if (!is || is->isDeleted()) DIE
    is->markDeleted(true);
    if (is->prev())
        is->prev()->next(is->next());
    else
        seg.first(is->next());

    if (is->next())
        is->next()->prev(is->prev());
    else
        seg.last(is->prev());


    if (is == smap.highwater())
            smap.highwater(is->next());
    if (is->prev())
        is = is->prev();
    seg.extendLength(-1);
ENDOP

STARTOP(assoc)
    declare_params(1);
    unsigned int  num = uint8(*param);
    const int8 *  assocs = reinterpret_cast<const int8 *>(param+1);
    use_params(num);
    int max = -1;
    int min = -1;

    while (num-- > 0)
    {
        int sr = *assocs++;
        slotref ts = slotat(sr);
        if (ts && (min == -1 || ts->before() < min)) min = ts->before();
        if (ts && ts->after() > max) max = ts->after();
    }
    if (min > -1)   // implies max > -1
    {
        is->before(min);
        is->after(max);
    }
ENDOP

STARTOP(cntxt_item)
    // It turns out this is a cunningly disguised condition forward jump.
    declare_params(3);
    const int       is_arg = int8(param[0]);
    const size_t    iskip  = uint8(param[1]),
                    dskip  = uint8(param[2]);

    if (mapb + is_arg != map)
    {
        ip += iskip;
        dp += dskip;
        push(true);
    }
ENDOP

STARTOP(attr_set)
    declare_params(1);
    const attrCode      slat = attrCode(uint8(*param));
    const          int  val  = pop();
    is->setAttr(&seg, slat, 0, val, smap);
ENDOP

STARTOP(attr_add)
    declare_params(1);
    const attrCode      slat = attrCode(uint8(*param));
    const     uint32_t  val  = pop();
    if ((slat == gr_slatPosX || slat == gr_slatPosY) && (flags & POSITIONED) == 0)
    {
        seg.positionSlots(0, *smap.begin(), *(smap.end()-1), seg.currdir());
        flags |= POSITIONED;
    }
    uint32_t res = uint32_t(is->getAttr(&seg, slat, 0));
    is->setAttr(&seg, slat, 0, int32_t(val + res), smap);
ENDOP

STARTOP(attr_sub)
    declare_params(1);
    const attrCode      slat = attrCode(uint8(*param));
    const     uint32_t  val  = pop();
    if ((slat == gr_slatPosX || slat == gr_slatPosY) && (flags & POSITIONED) == 0)
    {
        seg.positionSlots(0, *smap.begin(), *(smap.end()-1), seg.currdir());
        flags |= POSITIONED;
    }
    uint32_t res = uint32_t(is->getAttr(&seg, slat, 0));
    is->setAttr(&seg, slat, 0, int32_t(res - val), smap);
ENDOP

STARTOP(attr_set_slot)
    declare_params(1);
    const attrCode  slat   = attrCode(uint8(*param));
    const int       offset = int(map - smap.begin())*int(slat == gr_slatAttTo);
    const int       val    = pop()  + offset;
    is->setAttr(&seg, slat, offset, val, smap);
ENDOP

STARTOP(iattr_set_slot)
    declare_params(2);
    const attrCode  slat = attrCode(uint8(param[0]));
    const uint8     idx  = uint8(param[1]);
    const int       val  = int(pop()  + (map - smap.begin())*int(slat == gr_slatAttTo));
    is->setAttr(&seg, slat, idx, val, smap);
ENDOP

STARTOP(push_slot_attr)
    declare_params(2);
    const attrCode      slat     = attrCode(uint8(param[0]));
    const int           slot_ref = int8(param[1]);
    if ((slat == gr_slatPosX || slat == gr_slatPosY) && (flags & POSITIONED) == 0)
    {
        seg.positionSlots(0, *smap.begin(), *(smap.end()-1), seg.currdir());
        flags |= POSITIONED;
    }
    slotref slot = slotat(slot_ref);
    if (slot)
    {
        int res = slot->getAttr(&seg, slat, 0);
        push(res);
    }
ENDOP

STARTOP(push_glyph_attr_obs)
    declare_params(2);
    const unsigned int  glyph_attr = uint8(param[0]);
    const int           slot_ref   = int8(param[1]);
    slotref slot = slotat(slot_ref);
    if (slot)
        push(int32(seg.glyphAttr(slot->gid(), glyph_attr)));
ENDOP

STARTOP(push_glyph_metric)
    declare_params(3);
    const unsigned int  glyph_attr  = uint8(param[0]);
    const int           slot_ref    = int8(param[1]);
    const signed int    attr_level  = uint8(param[2]);
    slotref slot = slotat(slot_ref);
    if (slot)
        push(seg.getGlyphMetric(slot, glyph_attr, attr_level, dir));
ENDOP

STARTOP(push_feat)
    declare_params(2);
    const unsigned int  feat        = uint8(param[0]);
    const int           slot_ref    = int8(param[1]);
    slotref slot = slotat(slot_ref);
    if (slot)
    {
        uint8 fid = seg.charinfo(slot->original())->fid();
        push(seg.getFeature(fid, feat));
    }
ENDOP

STARTOP(push_att_to_gattr_obs)
    declare_params(2);
    const unsigned int  glyph_attr  = uint8(param[0]);
    const int           slot_ref    = int8(param[1]);
    slotref slot = slotat(slot_ref);
    if (slot)
    {
        slotref att = slot->attachedTo();
        if (att) slot = att;
        push(int32(seg.glyphAttr(slot->gid(), glyph_attr)));
    }
ENDOP

STARTOP(push_att_to_glyph_metric)
    declare_params(3);
    const unsigned int  glyph_attr  = uint8(param[0]);
    const int           slot_ref    = int8(param[1]);
    const signed int    attr_level  = uint8(param[2]);
    slotref slot = slotat(slot_ref);
    if (slot)
    {
        slotref att = slot->attachedTo();
        if (att) slot = att;
        push(int32(seg.getGlyphMetric(slot, glyph_attr, attr_level, dir)));
    }
ENDOP

STARTOP(push_islot_attr)
    declare_params(3);
    const attrCode  slat     = attrCode(uint8(param[0]));
    const int           slot_ref = int8(param[1]),
                        idx      = uint8(param[2]);
    if ((slat == gr_slatPosX || slat == gr_slatPosY) && (flags & POSITIONED) == 0)
    {
        seg.positionSlots(0, *smap.begin(), *(smap.end()-1), seg.currdir());
        flags |= POSITIONED;
    }
    slotref slot = slotat(slot_ref);
    if (slot)
    {
        int res = slot->getAttr(&seg, slat, idx);
        push(res);
    }
ENDOP

#if 0
STARTOP(push_iglyph_attr) // not implemented
    NOT_IMPLEMENTED;
ENDOP
#endif

STARTOP(pop_ret)
    const uint32 ret = pop();
    EXIT(ret);
ENDOP

STARTOP(ret_zero)
    EXIT(0);
ENDOP

STARTOP(ret_true)
    EXIT(1);
ENDOP

STARTOP(iattr_set)
    declare_params(2);
    const attrCode      slat = attrCode(uint8(param[0]));
    const uint8         idx  = uint8(param[1]);
    const          int  val  = pop();
    is->setAttr(&seg, slat, idx, val, smap);
ENDOP

STARTOP(iattr_add)
    declare_params(2);
    const attrCode      slat = attrCode(uint8(param[0]));
    const uint8         idx  = uint8(param[1]);
    const     uint32_t  val  = pop();
    if ((slat == gr_slatPosX || slat == gr_slatPosY) && (flags & POSITIONED) == 0)
    {
        seg.positionSlots(0, *smap.begin(), *(smap.end()-1), seg.currdir());
        flags |= POSITIONED;
    }
    uint32_t res = uint32_t(is->getAttr(&seg, slat, idx));
    is->setAttr(&seg, slat, idx, int32_t(val + res), smap);
ENDOP

STARTOP(iattr_sub)
    declare_params(2);
    const attrCode      slat = attrCode(uint8(param[0]));
    const uint8         idx  = uint8(param[1]);
    const     uint32_t  val  = pop();
    if ((slat == gr_slatPosX || slat == gr_slatPosY) && (flags & POSITIONED) == 0)
    {
        seg.positionSlots(0, *smap.begin(), *(smap.end()-1), seg.currdir());
        flags |= POSITIONED;
    }
    uint32_t res = uint32_t(is->getAttr(&seg, slat, idx));
    is->setAttr(&seg, slat, idx, int32_t(res - val), smap);
ENDOP

STARTOP(push_proc_state)
    use_params(1);
    push(1);
ENDOP

STARTOP(push_version)
    push(0x00030000);
ENDOP

STARTOP(put_subs)
    declare_params(5);
    const int        slot_ref     = int8(param[0]);
    const unsigned int  input_class  = uint8(param[1]) << 8
                                     | uint8(param[2]);
    const unsigned int  output_class = uint8(param[3]) << 8
                                     | uint8(param[4]);
    slotref slot = slotat(slot_ref);
    if (slot)
    {
        int index = seg.findClassIndex(input_class, slot->gid());
        is->setGlyph(&seg, seg.getClassGlyph(output_class, index));
    }
ENDOP

#if 0
STARTOP(put_subs2) // not implemented
    NOT_IMPLEMENTED;
ENDOP

STARTOP(put_subs3) // not implemented
    NOT_IMPLEMENTED;
ENDOP
#endif

STARTOP(put_glyph)
    declare_params(2);
    const unsigned int output_class  = uint8(param[0]) << 8
                                     | uint8(param[1]);
    is->setGlyph(&seg, seg.getClassGlyph(output_class, 0));
ENDOP

STARTOP(push_glyph_attr)
    declare_params(3);
    const unsigned int  glyph_attr  = uint8(param[0]) << 8
                                    | uint8(param[1]);
    const int           slot_ref    = int8(param[2]);
    slotref slot = slotat(slot_ref);
    if (slot)
        push(int32(seg.glyphAttr(slot->gid(), glyph_attr)));
ENDOP

STARTOP(push_att_to_glyph_attr)
    declare_params(3);
    const unsigned int  glyph_attr  = uint8(param[0]) << 8
                                    | uint8(param[1]);
    const int           slot_ref    = int8(param[2]);
    slotref slot = slotat(slot_ref);
    if (slot)
    {
        slotref att = slot->attachedTo();
        if (att) slot = att;
        push(int32(seg.glyphAttr(slot->gid(), glyph_attr)));
    }
ENDOP

STARTOP(temp_copy)
    slotref newSlot = seg.newSlot();
    if (!newSlot || !is) DIE;
    int16 *tempUserAttrs = newSlot->userAttrs();
    memcpy(newSlot, is, sizeof(Slot));
    memcpy(tempUserAttrs, is->userAttrs(), seg.numAttrs() * sizeof(uint16));
    newSlot->userAttrs(tempUserAttrs);
    newSlot->markCopied(true);
    *map = newSlot;
ENDOP

STARTOP(band)
    binop(&);
ENDOP

STARTOP(bor)
    binop(|);
ENDOP

STARTOP(bnot)
    *sp = ~*sp;
ENDOP

STARTOP(setbits)
    declare_params(4);
    const uint16 m  = uint16(param[0]) << 8
                    | uint8(param[1]);
    const uint16 v  = uint16(param[2]) << 8
                    | uint8(param[3]);
    *sp = ((*sp) & ~m) | v;
ENDOP

STARTOP(set_feat)
    declare_params(2);
    const unsigned int  feat        = uint8(param[0]);
    const int           slot_ref    = int8(param[1]);
    slotref slot = slotat(slot_ref);
    if (slot)
    {
        uint8 fid = seg.charinfo(slot->original())->fid();
        seg.setFeature(fid, feat, pop());
    }
ENDOP
