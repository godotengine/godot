/**************************************************************************/
/*  pga2.cpp                                                              */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "pga2.h"
#include "pga2.inc"

PGAVector2 PGAVector2::gp(const PGAVector2 &p_mv) const {
    PGAVector2 r;
    GP_FULL(r, (*this), p_mv)
    return r;
}

PGAVector2 PGAVector2::op(const PGAVector2 &p_mv) const {
    PGAVector2 r;
    OP_FULL(r, (*this), p_mv)
    return r;
}

PGAVector2 PGAVector2::ip(const PGAVector2 &p_mv) const {
    PGAVector2 r;
    IP_FULL(r, (*this), p_mv)
    return r;
}

PGAVector2 PGAVector2::rp(const PGAVector2 &p_mv) const {
    PGAVector2 r;
    RP_FULL(r, (*this), p_mv)
    return r;
}

PGAVector2 PGAVector2::cp(const PGAVector2 &p_mv) const {
    PGAVector2 r;
    CP_FULL(r, (*this), p_mv)
    return r;
}

real_t PGAVector2::sp(const PGAVector2 &p_mv) const {
    real_t r;
    SP_FULL(r, (*this), p_mv)
    return r;
}

PGAVector2 PGAVector2::reverse() const {
    PGAVector2 r;
    REVERSE_0(r, (*this))
    REVERSE_1(r, (*this))
    REVERSE_2(r, (*this))
    REVERSE_3(r, (*this))
    return r;
}

PGAVector2 PGAVector2::dual() const {
    PGAVector2 r;
    DUAL_0(r, (*this))
    DUAL_1(r, (*this))
    DUAL_2(r, (*this))
    DUAL_3(r, (*this))
    return r;
}

PGAVector2 PGAVector2::grade(GradeMask mask) const {
}

void PGAVector2::operator+=(const PGAVector2 &p_mv) {
    ADD_0((*this), (*this), p_mv, =)
    ADD_1((*this), (*this), p_mv, =)
    ADD_2((*this), (*this), p_mv, =)
    ADD_3((*this), (*this), p_mv, =)
}

void PGAVector2::operator-=(const PGAVector2 &p_mv) {
    SUB_0((*this), (*this), p_mv, =)
    SUB_1((*this), (*this), p_mv, =)
    SUB_2((*this), (*this), p_mv, =)
    SUB_3((*this), (*this), p_mv, =)
}

void PGAVector2::operator*=(const real_t p_s) {
    MUL_0((*this), (*this), p_s, =)
    MUL_1((*this), (*this), p_s, =)
    MUL_2((*this), (*this), p_s, =)
    MUL_3((*this), (*this), p_s, =)
}

void PGAVector2::operator/=(const real_t p_s) {
    DIV_0((*this), (*this), p_s, =)
    DIV_1((*this), (*this), p_s, =)
    DIV_2((*this), (*this), p_s, =)
    DIV_3((*this), (*this), p_s, =)
}