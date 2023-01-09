/*
 * Copyright © 2010 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */


/**
 * \file ir_expression_flattening.h
 *
 * Takes the leaves of expression trees and makes them dereferences of
 * assignments of the leaves to temporaries, according to a predicate.
 *
 * This is used for automatic function inlining, where we want to take
 * an expression containing a call and move the call out to its own
 * assignment so that we can inline it at the appropriate place in the
 * instruction stream.
 */

#ifndef GLSL_IR_EXPRESSION_FLATTENING_H
#define GLSL_IR_EXPRESSION_FLATTENING_H

void do_expression_flattening(exec_list *instructions,
			      bool (*predicate)(ir_instruction *ir));

#endif /* GLSL_IR_EXPRESSION_FLATTENING_H */
