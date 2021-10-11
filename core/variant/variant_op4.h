/*************************************************************************/
/*  variant_op.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "variant_op.h"

typedef void (*VariantEvaluatorFunction)(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid);

static Variant::Type operator_return_type_table[Variant::OP_MAX][Variant::VARIANT_MAX][Variant::VARIANT_MAX];
static VariantEvaluatorFunction operator_evaluator_table[Variant::OP_MAX][Variant::VARIANT_MAX][Variant::VARIANT_MAX];
static Variant::ValidatedOperatorEvaluator validated_operator_evaluator_table[Variant::OP_MAX][Variant::VARIANT_MAX][Variant::VARIANT_MAX];
static Variant::PTROperatorEvaluator ptr_operator_evaluator_table[Variant::OP_MAX][Variant::VARIANT_MAX][Variant::VARIANT_MAX];

template <class T>
void register_op(Variant::Operator p_op, Variant::Type p_type_a, Variant::Type p_type_b) {
	operator_return_type_table[p_op][p_type_a][p_type_b] = T::get_return_type();
	operator_evaluator_table[p_op][p_type_a][p_type_b] = T::evaluate;
	validated_operator_evaluator_table[p_op][p_type_a][p_type_b] = T::validated_evaluate;
	ptr_operator_evaluator_table[p_op][p_type_a][p_type_b] = T::ptr_evaluate;
}
