# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#
# Authors:
#    Jason Ekstrand (jason@jlekstrand.net)

from collections import OrderedDict
import nir_algebraic
from nir_opcodes import type_sizes
import itertools
import struct
from math import pi
import math

# Convenience variables
a = 'a'
b = 'b'
c = 'c'
d = 'd'
e = 'e'

signed_zero_inf_nan_preserve_16 = 'nir_is_float_control_signed_zero_inf_nan_preserve(info->float_controls_execution_mode, 16)'
signed_zero_inf_nan_preserve_32 = 'nir_is_float_control_signed_zero_inf_nan_preserve(info->float_controls_execution_mode, 32)'

ignore_exact = nir_algebraic.ignore_exact

# Written in the form (<search>, <replace>) where <search> is an expression
# and <replace> is either an expression or a value.  An expression is
# defined as a tuple of the form ([~]<op>, <src0>, <src1>, <src2>, <src3>)
# where each source is either an expression or a value.  A value can be
# either a numeric constant or a string representing a variable name.
#
# If the opcode in a search expression is prefixed by a '~' character, this
# indicates that the operation is inexact.  Such operations will only get
# applied to SSA values that do not have the exact bit set.  This should be
# used by by any optimizations that are not bit-for-bit exact.  It should not,
# however, be used for backend-requested lowering operations as those need to
# happen regardless of precision.
#
# Variable names are specified as "[#]name[@type][(cond)][.swiz]" where:
# "#" indicates that the given variable will only match constants,
# type indicates that the given variable will only match values from ALU
#    instructions with the given output type,
# (cond) specifies an additional condition function (see nir_search_helpers.h),
# swiz is a swizzle applied to the variable (only in the <replace> expression)
#
# For constants, you have to be careful to make sure that it is the right
# type because python is unaware of the source and destination types of the
# opcodes.
#
# All expression types can have a bit-size specified.  For opcodes, this
# looks like "op@32", for variables it is "a@32" or "a@uint32" to specify a
# type and size.  In the search half of the expression this indicates that it
# should only match that particular bit-size.  In the replace half of the
# expression this indicates that the constructed value should have that
# bit-size.
#
# If the opcode in a replacement expression is prefixed by a '!' character,
# this indicated that the new expression will be marked exact.
#
# A special condition "many-comm-expr" can be used with expressions to note
# that the expression and its subexpressions have more commutative expressions
# than nir_replace_instr can handle.  If this special condition is needed with
# another condition, the two can be separated by a comma (e.g.,
# "(many-comm-expr,is_used_once)").

# based on https://web.archive.org/web/20180105155939/http://forum.devmaster.net/t/fast-and-accurate-sine-cosine/9648
def lowered_sincos(c):
    x = ('fsub', ('fmul', 2.0, ('ffract', ('fadd', ('fmul', 0.5 / pi, a), c))), 1.0)
    x = ('fmul', ('fsub', x, ('fmul', x, ('fabs', x))), 4.0)
    return ('ffma', ('ffma', x, ('fabs', x), ('fneg', x)), 0.225, x)

def intBitsToFloat(i):
    return struct.unpack('!f', struct.pack('!I', i))[0]

optimizations = [

   (('imul', a, '#b(is_pos_power_of_two)'), ('ishl', a, ('find_lsb', b)), '!options->lower_bitops'),
   (('imul', 'a@8', 0x80), ('ishl', a, 7), '!options->lower_bitops'),
   (('imul', 'a@16', 0x8000), ('ishl', a, 15), '!options->lower_bitops'),
   (('imul', 'a@32', 0x80000000), ('ishl', a, 31), '!options->lower_bitops'),
   (('imul', 'a@64', 0x8000000000000000), ('ishl', a, 63), '!options->lower_bitops'),
   (('imul', a, '#b(is_neg_power_of_two)'), ('ineg', ('ishl', a, ('find_lsb', ('iabs', b)))), '!options->lower_bitops'),
   (('ishl', a, '#b'), ('imul', a, ('ishl', 1, b)), 'options->lower_bitops'),

   (('imul@64', a, '#b(is_bitcount2)'), ('iadd', ('ishl', a, ('ufind_msb', b)), ('ishl', a, ('find_lsb', b))),
    '!options->lower_bitops && (options->lower_int64_options & (nir_lower_imul64 | nir_lower_shift64)) == nir_lower_imul64'),

   (('unpack_64_2x32_split_x', ('imul_2x32_64(is_used_once)', a, b)), ('imul', a, b)),
   (('unpack_64_2x32_split_x', ('umul_2x32_64(is_used_once)', a, b)), ('imul', a, b)),
   (('imul_2x32_64', a, b), ('pack_64_2x32_split', ('imul', a, b), ('imul_high', a, b)), 'options->lower_mul_2x32_64'),
   (('umul_2x32_64', a, b), ('pack_64_2x32_split', ('imul', a, b), ('umul_high', a, b)), 'options->lower_mul_2x32_64'),
   (('udiv', a, 1), a),
   (('idiv', a, 1), a),
   (('umod', a, 1), 0),
   (('imod', a, 1), 0),
   (('imod', a, -1), 0),
   (('irem', a, 1), 0),
   (('irem', a, -1), 0),
   (('udiv', a, '#b(is_pos_power_of_two)'), ('ushr', a, ('find_lsb', b)), '!options->lower_bitops'),
   (('idiv', a, '#b(is_pos_power_of_two)'), ('imul', ('isign', a), ('ushr', ('iabs', a), ('find_lsb', b))), '!options->lower_bitops'),
   (('idiv', a, '#b(is_neg_power_of_two)'), ('ineg', ('imul', ('isign', a), ('ushr', ('iabs', a), ('find_lsb', ('iabs', b))))), '!options->lower_bitops'),
   (('umod', a, '#b(is_pos_power_of_two)'), ('iand', a, ('isub', b, 1)), '!options->lower_bitops'),
   (('imod', a, '#b(is_pos_power_of_two)'), ('iand', a, ('isub', b, 1)), '!options->lower_bitops'),
   (('imod', a, '#b(is_neg_power_of_two)'), ('bcsel', ('ieq', ('ior', a, b), b), 0, ('ior', a, b)), '!options->lower_bitops'),
   # 'irem(a, b)' -> 'a - ((a < 0 ? (a + b - 1) : a) & -b)'
   (('irem', a, '#b(is_pos_power_of_two)'),
    ('isub', a, ('iand', ('bcsel', ('ilt', a, 0), ('iadd', a, ('isub', b, 1)), a), ('ineg', b))),
    '!options->lower_bitops'),
   (('irem', a, '#b(is_neg_power_of_two)'), ('irem', a, ('iabs', b)), '!options->lower_bitops'),

   (('~fneg', ('fneg', a)), a),
   (('ineg', ('ineg', a)), a),
   (('fabs', ('fneg', a)), ('fabs', a)),
   (('fabs', ('u2f', a)), ('u2f', a)),
   (('iabs', ('iabs', a)), ('iabs', a)),
   (('iabs', ('ineg', a)), ('iabs', a)),
   (('f2b', ('fneg', a)), ('f2b', a)),
   (('~fadd', a, 0.0), a),
   # a+0.0 is 'a' unless 'a' is denormal or -0.0. If it's only used by a
   # floating point instruction, they should flush any input denormals and we
   # can replace -0.0 with 0.0 if the float execution mode allows it.
   (('fadd(is_only_used_as_float)', 'a@16', 0.0), a, '!'+signed_zero_inf_nan_preserve_16),
   (('fadd(is_only_used_as_float)', 'a@32', 0.0), a, '!'+signed_zero_inf_nan_preserve_32),
   (('iadd', a, 0), a),
   (('iadd_sat', a, 0), a),
   (('isub_sat', a, 0), a),
   (('uadd_sat', a, 0), a),
   (('usub_sat', a, 0), a),
   (('usadd_4x8_vc4', a, 0), a),
   (('usadd_4x8_vc4', a, ~0), ~0),
   (('~fadd', ('fmul', a, b), ('fmul', a, c)), ('fmul', a, ('fadd', b, c))),
   (('~fadd', ('fmulz', a, b), ('fmulz', a, c)), ('fmulz', a, ('fadd', b, c))),
   (('~ffma', a, b, ('ffma(is_used_once)', a, c, d)), ('ffma', a, ('fadd', b, c), d)),
   (('~ffma', a, b, ('fmul(is_used_once)', a, c)), ('fmul', a, ('fadd', b, c))),
   (('~fadd', ('fmul(is_used_once)', a, b), ('ffma(is_used_once)', a, c, d)), ('ffma', a, ('fadd', b, c), d)),
   (('~ffma', a, ('fmul(is_used_once)', b, c), ('fmul(is_used_once)', b, d)), ('fmul', b, ('ffma', a, c, d))),
   (('~ffmaz', a, b, ('ffmaz(is_used_once)', a, c, d)), ('ffmaz', a, ('fadd', b, c), d)),
   (('~ffmaz', a, b, ('fmulz(is_used_once)', a, c)), ('fmulz', a, ('fadd', b, c))),
   (('~fadd', ('fmulz(is_used_once)', a, b), ('ffmaz(is_used_once)', a, c, d)), ('ffmaz', a, ('fadd', b, c), d)),
   (('~ffmaz', a, ('fmulz(is_used_once)', b, c), ('fmulz(is_used_once)', b, d)), ('fmulz', b, ('ffmaz', a, c, d))),
   (('iadd', ('imul', a, b), ('imul', a, c)), ('imul', a, ('iadd', b, c))),
   (('iadd', ('ishl', b, a), ('ishl', c, a)), ('ishl', ('iadd', b, c), a)),
   (('iand', ('ior', a, b), ('ior', a, c)), ('ior', a, ('iand', b, c))),
   (('ior', ('iand', a, b), ('iand', a, c)), ('iand', a, ('ior', b, c))),
   (('ieq', ('iand', a, '#b(is_pos_power_of_two)'), b), ('ine', ('iand', a, b), 0)),
   (('ine', ('iand', a, '#b(is_pos_power_of_two)'), b), ('ieq', ('iand', a, b), 0)),
   (('~fadd', ('fneg', a), a), 0.0),
   (('iadd', ('ineg', a), a), 0),
   (('iadd', ('ineg', a), ('iadd', a, b)), b),
   (('iadd', a, ('iadd', ('ineg', a), b)), b),
   (('~fadd', ('fneg', a), ('fadd', a, b)), b),
   (('~fadd', a, ('fadd', ('fneg', a), b)), b),
   (('fadd', ('fsat', a), ('fsat', ('fneg', a))), ('fsat', ('fabs', a))),
   (('~fmul', a, 0.0), 0.0),
   # The only effect a*0.0 should have is when 'a' is infinity, -0.0 or NaN
   (('fmul', 'a@16', 0.0), 0.0, '!'+signed_zero_inf_nan_preserve_16),
   (('fmul', 'a@32', 0.0), 0.0, '!'+signed_zero_inf_nan_preserve_32),
   (('fmulz', a, 0.0), 0.0),
   (('fmulz', a, 'b(is_finite_not_zero)'), ('fmul', a, b), '!'+signed_zero_inf_nan_preserve_32),
   (('fmulz', 'a(is_finite)', 'b(is_finite)'), ('fmul', a, b)),
   (('fmulz', a, a), ('fmul', a, a)),
   (('ffmaz', a, 'b(is_finite_not_zero)', c), ('ffma', a, b, c), '!'+signed_zero_inf_nan_preserve_32),
   (('ffmaz', 'a(is_finite)', 'b(is_finite)', c), ('ffma', a, b, c)),
   (('ffmaz', a, a, b), ('ffma', a, a, b)),
   (('imul', a, 0), 0),
   (('umul_unorm_4x8_vc4', a, 0), 0),
   (('umul_unorm_4x8_vc4', a, ~0), a),
   (('~fmul', a, 1.0), a),
   (('~fmulz', a, 1.0), a),
   # The only effect a*1.0 can have is flushing denormals. If it's only used by
   # a floating point instruction, they should flush any input denormals and
   # this multiplication isn't needed.
   (('fmul(is_only_used_as_float)', a, 1.0), a),
   (('imul', a, 1), a),
   (('fmul', a, -1.0), ('fneg', a)),
   (('imul', a, -1), ('ineg', a)),
   # If a < 0: fsign(a)*a*a => -1*a*a => -a*a => abs(a)*a
   # If a > 0: fsign(a)*a*a => 1*a*a => a*a => abs(a)*a
   # If a == 0: fsign(a)*a*a => 0*0*0 => abs(0)*0
   # If a != a: fsign(a)*a*a => 0*NaN*NaN => abs(NaN)*NaN
   (('fmul', ('fsign', a), ('fmul', a, a)), ('fmul', ('fabs', a), a)),
   (('fmul', ('fmul', ('fsign', a), a), a), ('fmul', ('fabs', a), a)),
   (('~ffma', 0.0, a, b), b),
   (('ffma@16(is_only_used_as_float)', 0.0, a, b), b, '!'+signed_zero_inf_nan_preserve_16),
   (('ffma@32(is_only_used_as_float)', 0.0, a, b), b, '!'+signed_zero_inf_nan_preserve_32),
   (('ffmaz', 0.0, a, b), ('fadd', 0.0, b)),
   (('~ffma', a, b, 0.0), ('fmul', a, b)),
   (('ffma@16', a, b, 0.0), ('fmul', a, b), '!'+signed_zero_inf_nan_preserve_16),
   (('ffma@32', a, b, 0.0), ('fmul', a, b), '!'+signed_zero_inf_nan_preserve_32),
   (('ffmaz', a, b, 0.0), ('fmulz', a, b), '!'+signed_zero_inf_nan_preserve_32),
   (('ffma', 1.0, a, b), ('fadd', a, b)),
   (('ffmaz', 1.0, a, b), ('fadd', a, b), '!'+signed_zero_inf_nan_preserve_32),
   (('ffma', -1.0, a, b), ('fadd', ('fneg', a), b)),
   (('ffmaz', -1.0, a, b), ('fadd', ('fneg', a), b), '!'+signed_zero_inf_nan_preserve_32),
   (('~ffma', '#a', '#b', c), ('fadd', ('fmul', a, b), c)),
   (('~ffmaz', '#a', '#b', c), ('fadd', ('fmulz', a, b), c)),
   (('~flrp', a, b, 0.0), a),
   (('~flrp', a, b, 1.0), b),
   (('~flrp', a, a, b), a),
   (('~flrp', 0.0, a, b), ('fmul', a, b)),

   # flrp(a, a + b, c) => a + flrp(0, b, c) => a + (b * c)
   (('~flrp', a, ('fadd(is_used_once)', a, b), c), ('fadd', ('fmul', b, c), a)),

   (('sdot_4x8_iadd', a, 0, b), b),
   (('udot_4x8_uadd', a, 0, b), b),
   (('sdot_4x8_iadd_sat', a, 0, b), b),
   (('udot_4x8_uadd_sat', a, 0, b), b),
   (('sdot_2x16_iadd', a, 0, b), b),
   (('udot_2x16_uadd', a, 0, b), b),
   (('sdot_2x16_iadd_sat', a, 0, b), b),
   (('udot_2x16_uadd_sat', a, 0, b), b),

   # sudot_4x8_iadd is not commutative at all, so the patterns must be
   # duplicated with zeros on each of the first positions.
   (('sudot_4x8_iadd', a, 0, b), b),
   (('sudot_4x8_iadd', 0, a, b), b),
   (('sudot_4x8_iadd_sat', a, 0, b), b),
   (('sudot_4x8_iadd_sat', 0, a, b), b),

   (('iadd', ('sdot_4x8_iadd(is_used_once)', a, b, '#c'), '#d'), ('sdot_4x8_iadd', a, b, ('iadd', c, d))),
   (('iadd', ('udot_4x8_uadd(is_used_once)', a, b, '#c'), '#d'), ('udot_4x8_uadd', a, b, ('iadd', c, d))),
   (('iadd', ('sudot_4x8_iadd(is_used_once)', a, b, '#c'), '#d'), ('sudot_4x8_iadd', a, b, ('iadd', c, d))),
   (('iadd', ('sdot_2x16_iadd(is_used_once)', a, b, '#c'), '#d'), ('sdot_2x16_iadd', a, b, ('iadd', c, d))),
   (('iadd', ('udot_2x16_uadd(is_used_once)', a, b, '#c'), '#d'), ('udot_2x16_uadd', a, b, ('iadd', c, d))),

   # Try to let constant folding eliminate the dot-product part.  These are
   # safe because the dot product cannot overflow 32 bits.
   (('iadd', ('sdot_4x8_iadd', 'a(is_not_const)', b, 0), c), ('sdot_4x8_iadd', a, b, c)),
   (('iadd', ('udot_4x8_uadd', 'a(is_not_const)', b, 0), c), ('udot_4x8_uadd', a, b, c)),
   (('iadd', ('sudot_4x8_iadd', 'a(is_not_const)', b, 0), c), ('sudot_4x8_iadd', a, b, c)),
   (('iadd', ('sudot_4x8_iadd', a, 'b(is_not_const)', 0), c), ('sudot_4x8_iadd', a, b, c)),
   (('iadd', ('sdot_2x16_iadd', 'a(is_not_const)', b, 0), c), ('sdot_2x16_iadd', a, b, c)),
   (('iadd', ('udot_2x16_uadd', 'a(is_not_const)', b, 0), c), ('udot_2x16_uadd', a, b, c)),
   (('sdot_4x8_iadd', '#a', '#b', 'c(is_not_const)'), ('iadd', ('sdot_4x8_iadd', a, b, 0), c)),
   (('udot_4x8_uadd', '#a', '#b', 'c(is_not_const)'), ('iadd', ('udot_4x8_uadd', a, b, 0), c)),
   (('sudot_4x8_iadd', '#a', '#b', 'c(is_not_const)'), ('iadd', ('sudot_4x8_iadd', a, b, 0), c)),
   (('sdot_2x16_iadd', '#a', '#b', 'c(is_not_const)'), ('iadd', ('sdot_2x16_iadd', a, b, 0), c)),
   (('udot_2x16_uadd', '#a', '#b', 'c(is_not_const)'), ('iadd', ('udot_2x16_uadd', a, b, 0), c)),
   (('sdot_4x8_iadd_sat', '#a', '#b', 'c(is_not_const)'), ('iadd_sat', ('sdot_4x8_iadd', a, b, 0), c), '!options->lower_iadd_sat'),
   (('udot_4x8_uadd_sat', '#a', '#b', 'c(is_not_const)'), ('uadd_sat', ('udot_4x8_uadd', a, b, 0), c), '!options->lower_uadd_sat'),
   (('sudot_4x8_iadd_sat', '#a', '#b', 'c(is_not_const)'), ('iadd_sat', ('sudot_4x8_iadd', a, b, 0), c), '!options->lower_iadd_sat'),
   (('sdot_2x16_iadd_sat', '#a', '#b', 'c(is_not_const)'), ('iadd_sat', ('sdot_2x16_iadd', a, b, 0), c), '!options->lower_iadd_sat'),
   (('udot_2x16_uadd_sat', '#a', '#b', 'c(is_not_const)'), ('uadd_sat', ('udot_2x16_uadd', a, b, 0), c), '!options->lower_uadd_sat'),

   # Optimize open-coded fmulz.
   # (b==0.0 ? 0.0 : a) * (a==0.0 ? 0.0 : b) -> fmulz(a, b)
   (('fmul@32', ('bcsel', ignore_exact('feq', b, 0.0), 0.0, a), ('bcsel', ignore_exact('feq', a, 0.0), 0.0, b)),
    ('fmulz', a, b), 'options->has_fmulz && !'+signed_zero_inf_nan_preserve_32),
   (('fmul@32', a, ('bcsel', ignore_exact('feq', a, 0.0), 0.0, '#b(is_not_const_zero)')),
    ('fmulz', a, b), 'options->has_fmulz && !'+signed_zero_inf_nan_preserve_32),

   # ffma(b==0.0 ? 0.0 : a, a==0.0 ? 0.0 : b, c) -> ffmaz(a, b, c)
   (('ffma@32', ('bcsel', ignore_exact('feq', b, 0.0), 0.0, a), ('bcsel', ignore_exact('feq', a, 0.0), 0.0, b), c),
    ('ffmaz', a, b, c), 'options->has_fmulz && !'+signed_zero_inf_nan_preserve_32),
   (('ffma@32', a, ('bcsel', ignore_exact('feq', a, 0.0), 0.0, '#b(is_not_const_zero)'), c),
    ('ffmaz', a, b, c), 'options->has_fmulz && !'+signed_zero_inf_nan_preserve_32),

   # b == 0.0 ? 1.0 : fexp2(fmul(a, b)) -> fexp2(fmulz(a, b))
   (('bcsel', ignore_exact('feq', b, 0.0), 1.0, ('fexp2', ('fmul@32', a, b))),
    ('fexp2', ('fmulz', a, b)),
    'options->has_fmulz && !'+signed_zero_inf_nan_preserve_32),
]

# Shorthand for the expansion of just the dot product part of the [iu]dp4a
# instructions.
sdot_4x8_a_b = ('iadd', ('iadd', ('imul', ('extract_i8', a, 0), ('extract_i8', b, 0)),
                                 ('imul', ('extract_i8', a, 1), ('extract_i8', b, 1))),
                        ('iadd', ('imul', ('extract_i8', a, 2), ('extract_i8', b, 2)),
                                 ('imul', ('extract_i8', a, 3), ('extract_i8', b, 3))))
udot_4x8_a_b = ('iadd', ('iadd', ('imul', ('extract_u8', a, 0), ('extract_u8', b, 0)),
                                 ('imul', ('extract_u8', a, 1), ('extract_u8', b, 1))),
                        ('iadd', ('imul', ('extract_u8', a, 2), ('extract_u8', b, 2)),
                                 ('imul', ('extract_u8', a, 3), ('extract_u8', b, 3))))
sudot_4x8_a_b = ('iadd', ('iadd', ('imul', ('extract_i8', a, 0), ('extract_u8', b, 0)),
                                  ('imul', ('extract_i8', a, 1), ('extract_u8', b, 1))),
                         ('iadd', ('imul', ('extract_i8', a, 2), ('extract_u8', b, 2)),
                                  ('imul', ('extract_i8', a, 3), ('extract_u8', b, 3))))
sdot_2x16_a_b = ('iadd', ('imul', ('extract_i16', a, 0), ('extract_i16', b, 0)),
                         ('imul', ('extract_i16', a, 1), ('extract_i16', b, 1)))
udot_2x16_a_b = ('iadd', ('imul', ('extract_u16', a, 0), ('extract_u16', b, 0)),
                         ('imul', ('extract_u16', a, 1), ('extract_u16', b, 1)))

optimizations.extend([
   (('sdot_4x8_iadd', a, b, c), ('iadd', sdot_4x8_a_b, c), '!options->has_sdot_4x8'),
   (('udot_4x8_uadd', a, b, c), ('iadd', udot_4x8_a_b, c), '!options->has_udot_4x8'),
   (('sudot_4x8_iadd', a, b, c), ('iadd', sudot_4x8_a_b, c), '!options->has_sudot_4x8'),
   (('sdot_2x16_iadd', a, b, c), ('iadd', sdot_2x16_a_b, c), '!options->has_dot_2x16'),
   (('udot_2x16_uadd', a, b, c), ('iadd', udot_2x16_a_b, c), '!options->has_dot_2x16'),

   # For the unsigned dot-product, the largest possible value 4*(255*255) =
   # 0x3f804, so we don't have to worry about that intermediate result
   # overflowing.  0x100000000 - 0x3f804 = 0xfffc07fc.  If c is a constant
   # that is less than 0xfffc07fc, then the result cannot overflow ever.
   (('udot_4x8_uadd_sat', a, b, '#c(is_ult_0xfffc07fc)'), ('udot_4x8_uadd', a, b, c)),
   (('udot_4x8_uadd_sat', a, b, c), ('uadd_sat', udot_4x8_a_b, c), '!options->has_udot_4x8'),

   # For the signed dot-product, the largest positive value is 4*(-128*-128) =
   # 0x10000, and the largest negative value is 4*(-128*127) = -0xfe00.  We
   # don't have to worry about that intermediate result overflowing or
   # underflowing.
   (('sdot_4x8_iadd_sat', a, b, c), ('iadd_sat', sdot_4x8_a_b, c), '!options->has_sdot_4x8'),

   (('sudot_4x8_iadd_sat', a, b, c), ('iadd_sat', sudot_4x8_a_b, c), '!options->has_sudot_4x8'),

   (('udot_2x16_uadd_sat', a, b, c), ('uadd_sat', udot_2x16_a_b, c), '!options->has_dot_2x16'),
   (('sdot_2x16_iadd_sat', a, b, c), ('iadd_sat', sdot_2x16_a_b, c), '!options->has_dot_2x16'),
])

# Float sizes
for s in [16, 32, 64]:
    optimizations.extend([
       (('~flrp@{}'.format(s), a, b, ('b2f', 'c@1')), ('bcsel', c, b, a), 'options->lower_flrp{}'.format(s)),

       (('~flrp@{}'.format(s), a, ('fadd', a, b), c), ('fadd', ('fmul', b, c), a), 'options->lower_flrp{}'.format(s)),
       (('~flrp@{}'.format(s), ('fadd(is_used_once)', a, b), ('fadd(is_used_once)', a, c), d), ('fadd', ('flrp', b, c, d), a), 'options->lower_flrp{}'.format(s)),
       (('~flrp@{}'.format(s), a, ('fmul(is_used_once)', a, b), c), ('fmul', ('flrp', 1.0, b, c), a), 'options->lower_flrp{}'.format(s)),

       (('~fadd@{}'.format(s), ('fmul', a, ('fadd', 1.0, ('fneg', c))), ('fmul', b, c)), ('flrp', a, b, c), '!options->lower_flrp{}'.format(s)),
       # These are the same as the previous three rules, but it depends on
       # 1-fsat(x) <=> fsat(1-x).  See below.
       (('~fadd@{}'.format(s), ('fmul', a, ('fsat', ('fadd', 1.0, ('fneg', c)))), ('fmul', b, ('fsat', c))), ('flrp', a, b, ('fsat', c)), '!options->lower_flrp{}'.format(s)),
       (('~fadd@{}'.format(s), a, ('fmul', c, ('fadd', b, ('fneg', a)))), ('flrp', a, b, c), '!options->lower_flrp{}'.format(s)),

       (('~fadd@{}'.format(s),    ('fmul', a, ('fadd', 1.0, ('fneg', ('b2f', 'c@1')))), ('fmul', b, ('b2f',  c))), ('bcsel', c, b, a), 'options->lower_flrp{}'.format(s)),
       (('~fadd@{}'.format(s), a, ('fmul', ('b2f', 'c@1'), ('fadd', b, ('fneg', a)))), ('bcsel', c, b, a), 'options->lower_flrp{}'.format(s)),

       (('~ffma@{}'.format(s), a, ('fadd', 1.0, ('fneg', ('b2f', 'c@1'))), ('fmul', b, ('b2f', 'c@1'))), ('bcsel', c, b, a)),
       (('~ffma@{}'.format(s), b, ('b2f', 'c@1'), ('ffma', ('fneg', a), ('b2f', 'c@1'), a)), ('bcsel', c, b, a)),

       # These two aren't flrp lowerings, but do appear in some shaders.
       (('~ffma@{}'.format(s), ('b2f', 'c@1'), ('fadd', b, ('fneg', a)), a), ('bcsel', c, b, a)),
       (('~ffma@{}'.format(s), ('b2f', 'c@1'), ('ffma', ('fneg', a), b, d), ('fmul', a, b)), ('bcsel', c, d, ('fmul', a, b))),

       # 1 - ((1 - a) * (1 - b))
       # 1 - (1 - a - b + a*b)
       # 1 - 1 + a + b - a*b
       # a + b - a*b
       # a + b*(1 - a)
       # b*(1 - a) + 1*a
       # flrp(b, 1, a)
       (('~fadd@{}'.format(s), 1.0, ('fneg', ('fmul', ('fadd', 1.0, ('fneg', a)), ('fadd', 1.0, ('fneg', b))))), ('flrp', b, 1.0, a), '!options->lower_flrp{}'.format(s)),
    ])

optimizations.extend([
   (('~flrp', ('fmul(is_used_once)', a, b), ('fmul(is_used_once)', a, c), d), ('fmul', ('flrp', b, c, d), a)),

   (('~flrp', a, 0.0, c), ('fadd', ('fmul', ('fneg', a), c), a)),

   (('ftrunc@16', a), ('bcsel', ('flt', a, 0.0), ('fneg', ('ffloor', ('fabs', a))), ('ffloor', ('fabs', a))), 'options->lower_ftrunc'),
   (('ftrunc@32', a), ('bcsel', ('flt', a, 0.0), ('fneg', ('ffloor', ('fabs', a))), ('ffloor', ('fabs', a))), 'options->lower_ftrunc'),
   (('ftrunc@64', a), ('bcsel', ('flt', a, 0.0), ('fneg', ('ffloor', ('fabs', a))), ('ffloor', ('fabs', a))), 'options->lower_ftrunc || (options->lower_doubles_options & nir_lower_dtrunc)'),

   (('ffloor@16', a), ('fsub', a, ('ffract', a)), 'options->lower_ffloor'),
   (('ffloor@32', a), ('fsub', a, ('ffract', a)), 'options->lower_ffloor'),
   (('ffloor@64', a), ('fsub', a, ('ffract', a)), '(options->lower_ffloor || (options->lower_doubles_options & nir_lower_dfloor)) && !(options->lower_doubles_options & nir_lower_dfract)'),
   (('fadd@16', a, ('fadd@16', b, ('fneg', ('ffract', a)))), ('fadd@16', b, ('ffloor', a)), '!options->lower_ffloor'),
   (('fadd@32', a, ('fadd@32', b, ('fneg', ('ffract', a)))), ('fadd@32', b, ('ffloor', a)), '!options->lower_ffloor'),
   (('fadd@64', a, ('fadd@64', b, ('fneg', ('ffract', a)))), ('fadd@64', b, ('ffloor', a)), '!options->lower_ffloor && !(options->lower_doubles_options & nir_lower_dfloor)'),
   (('fadd@16', a, ('fneg', ('ffract', a))), ('ffloor', a), '!options->lower_ffloor'),
   (('fadd@32', a, ('fneg', ('ffract', a))), ('ffloor', a), '!options->lower_ffloor'),
   (('fadd@64', a, ('fneg', ('ffract', a))), ('ffloor', a), '!options->lower_ffloor && !(options->lower_doubles_options & nir_lower_dfloor)'),
   (('ffract@16', a), ('fsub', a, ('ffloor', a)), 'options->lower_ffract'),
   (('ffract@32', a), ('fsub', a, ('ffloor', a)), 'options->lower_ffract'),
   (('ffract@64', a), ('fsub', a, ('ffloor', a)), 'options->lower_ffract || (options->lower_doubles_options & nir_lower_dfract)'),
   (('fceil', a), ('fneg', ('ffloor', ('fneg', a))), 'options->lower_fceil'),
   (('ffma@16', a, b, c), ('fadd', ('fmul', a, b), c), 'options->lower_ffma16'),
   (('ffma@32', a, b, c), ('fadd', ('fmul', a, b), c), 'options->lower_ffma32'),
   (('ffma@64', a, b, c), ('fadd', ('fmul', a, b), c), 'options->lower_ffma64'),
   (('ffmaz', a, b, c), ('fadd', ('fmulz', a, b), c), 'options->lower_ffma32'),
   # Always lower inexact ffma, because it will be fused back by late optimizations (nir_opt_algebraic_late).
   (('~ffma@16', a, b, c), ('fadd', ('fmul', a, b), c), 'options->fuse_ffma16'),
   (('~ffma@32', a, b, c), ('fadd', ('fmul', a, b), c), 'options->fuse_ffma32'),
   (('~ffma@64', a, b, c), ('fadd', ('fmul', a, b), c), 'options->fuse_ffma64'),
   (('~ffmaz', a, b, c), ('fadd', ('fmulz', a, b), c), 'options->fuse_ffma32'),

   (('~fmul', ('fadd', ('iand', ('ineg', ('b2i', 'a@bool')), ('fmul', b, c)), '#d'), '#e'),
    ('bcsel', a, ('fmul', ('fadd', ('fmul', b, c), d), e), ('fmul', d, e))),

   (('fdph', a, b), ('fdot4', ('vec4', 'a.x', 'a.y', 'a.z', 1.0), b), 'options->lower_fdph'),

   (('fdot4', ('vec4', a, b,   c,   1.0), d), ('fdph',  ('vec3', a, b, c), d), '!options->lower_fdph'),
   (('fdot4', ('vec4', a, 0.0, 0.0, 0.0), b), ('fmul', a, b)),
   (('fdot4', ('vec4', a, b,   0.0, 0.0), c), ('fdot2', ('vec2', a, b), c)),
   (('fdot4', ('vec4', a, b,   c,   0.0), d), ('fdot3', ('vec3', a, b, c), d)),

   (('fdot3', ('vec3', a, 0.0, 0.0), b), ('fmul', a, b)),
   (('fdot3', ('vec3', a, b,   0.0), c), ('fdot2', ('vec2', a, b), c)),

   (('fdot2', ('vec2', a, 0.0), b), ('fmul', a, b)),
   (('fdot2', a, 1.0), ('fadd', 'a.x', 'a.y')),

   # Lower fdot to fsum when it is available
   (('fdot2', a, b), ('fsum2', ('fmul', a, b)), 'options->lower_fdot'),
   (('fdot3', a, b), ('fsum3', ('fmul', a, b)), 'options->lower_fdot'),
   (('fdot4', a, b), ('fsum4', ('fmul', a, b)), 'options->lower_fdot'),
   (('fsum2', a), ('fadd', 'a.x', 'a.y'), 'options->lower_fdot'),

   # If x >= 0 and x <= 1: fsat(1 - x) == 1 - fsat(x) trivially
   # If x < 0: 1 - fsat(x) => 1 - 0 => 1 and fsat(1 - x) => fsat(> 1) => 1
   # If x > 1: 1 - fsat(x) => 1 - 1 => 0 and fsat(1 - x) => fsat(< 0) => 0
   (('~fadd', ('fneg(is_used_once)', ('fsat(is_used_once)', 'a(is_not_fmul)')), 1.0), ('fsat', ('fadd', 1.0, ('fneg', a)))),

   # (a * #b + #c) << #d
   # ((a * #b) << #d) + (#c << #d)
   # (a * (#b << #d)) + (#c << #d)
   (('ishl', ('iadd', ('imul', a, '#b'), '#c'), '#d'),
    ('iadd', ('imul', a, ('ishl', b, d)), ('ishl', c, d))),

   # (a * #b) << #c
   # a * (#b << #c)
   (('ishl', ('imul', a, '#b'), '#c'), ('imul', a, ('ishl', b, c))),
])

# Care must be taken here.  Shifts in NIR uses only the lower log2(bitsize)
# bits of the second source.  These replacements must correctly handle the
# case where (b % bitsize) + (c % bitsize) >= bitsize.
for s in [8, 16, 32, 64]:
   mask = s - 1

   ishl = "ishl@{}".format(s)
   ishr = "ishr@{}".format(s)
   ushr = "ushr@{}".format(s)

   in_bounds = ('ult', ('iadd', ('iand', b, mask), ('iand', c, mask)), s)

   optimizations.extend([
       ((ishl, (ishl, a, '#b'), '#c'), ('bcsel', in_bounds, (ishl, a, ('iadd', b, c)), 0)),
       ((ushr, (ushr, a, '#b'), '#c'), ('bcsel', in_bounds, (ushr, a, ('iadd', b, c)), 0)),

       # To get get -1 for large shifts of negative values, ishr must instead
       # clamp the shift count to the maximum value.
       ((ishr, (ishr, a, '#b'), '#c'),
        (ishr, a, ('imin', ('iadd', ('iand', b, mask), ('iand', c, mask)), s - 1))),
   ])

# Optimize a pattern of address calculation created by DXVK where the offset is
# divided by 4 and then multipled by 4. This can be turned into an iand and the
# additions before can be reassociated to CSE the iand instruction.

for size, mask in ((8, 0xff), (16, 0xffff), (32, 0xffffffff), (64, 0xffffffffffffffff)):
    a_sz = 'a@{}'.format(size)

    optimizations.extend([
       # 'a >> #b << #b' -> 'a & ~((1 << #b) - 1)'
       (('ishl', ('ushr', a_sz, '#b'), b), ('iand', a, ('ishl', mask, b))),
       (('ishl', ('ishr', a_sz, '#b'), b), ('iand', a, ('ishl', mask, b))),

       # This does not trivially work with ishr.
       (('ushr', ('ishl', a_sz, '#b'), b), ('iand', a, ('ushr', mask, b))),
    ])

optimizations.extend([
    (('iand', ('ishl', 'a@32', '#b(is_first_5_bits_uge_2)'), -4), ('ishl', a, b)),
    (('iand', ('imul', a, '#b(is_unsigned_multiple_of_4)'), -4), ('imul', a, b)),
])

for log2 in range(1, 7): # powers of two from 2 to 64
   v = 1 << log2
   mask = 0xffffffff & ~(v - 1)
   b_is_multiple = '#b(is_unsigned_multiple_of_{})'.format(v)

   optimizations.extend([
       # Reassociate for improved CSE
       (('iand@32', ('iadd@32', a, b_is_multiple), mask), ('iadd', ('iand', a, mask), b)),
   ])

# To save space in the state tables, reduce to the set that is known to help.
# Previously, this was range(1, 32).  In addition, a couple rules inside the
# loop are commented out.  Revisit someday, probably after mesa/#2635 has some
# resolution.
for i in [1, 2, 16, 24]:
    lo_mask = 0xffffffff >> i
    hi_mask = (0xffffffff << i) & 0xffffffff

    optimizations.extend([
        # This pattern seems to only help in the soft-fp64 code.
        (('ishl@32', ('iand', 'a@32', lo_mask), i), ('ishl', a, i)),
#        (('ushr@32', ('iand', 'a@32', hi_mask), i), ('ushr', a, i)),
#        (('ishr@32', ('iand', 'a@32', hi_mask), i), ('ishr', a, i)),

        (('iand', ('ishl', 'a@32', i), hi_mask), ('ishl', a, i)),
        (('iand', ('ushr', 'a@32', i), lo_mask), ('ushr', a, i)),
#        (('iand', ('ishr', 'a@32', i), lo_mask), ('ushr', a, i)), # Yes, ushr is correct
    ])

optimizations.extend([
   # This is common for address calculations.  Reassociating may enable the
   # 'a<<c' to be CSE'd.  It also helps architectures that have an ISHLADD
   # instruction or a constant offset field for in load / store instructions.
   (('ishl', ('iadd', a, '#b'), '#c'), ('iadd', ('ishl', a, c), ('ishl', b, c))),

   # (a + #b) * #c => (a * #c) + (#b * #c)
   (('imul', ('iadd(is_used_once)', a, '#b'), '#c'), ('iadd', ('imul', a, c), ('imul', b, c))),

   # ((a + #b) + c) * #d => ((a + c) * #d) + (#b * #d)
   (('imul', ('iadd(is_used_once)', ('iadd(is_used_once)', a, '#b'), c), '#d'),
    ('iadd', ('imul', ('iadd', a, c), d), ('imul', b, d))),
   (('ishl', ('iadd(is_used_once)', ('iadd(is_used_once)', a, '#b'), c), '#d'),
    ('iadd', ('ishl', ('iadd', a, c), d), ('ishl', b, d))),

   # Comparison simplifications
   (('inot', ('flt(is_used_once)', 'a(is_a_number)', 'b(is_a_number)')), ('fge', a, b)),
   (('inot', ('fge(is_used_once)', 'a(is_a_number)', 'b(is_a_number)')), ('flt', a, b)),
   (('inot', ('feq(is_used_once)', a, b)), ('fneu', a, b)),
   (('inot', ('fneu(is_used_once)', a, b)), ('feq', a, b)),
   (('inot', ('ilt(is_used_once)', a, b)), ('ige', a, b)),
   (('inot', ('ult(is_used_once)', a, b)), ('uge', a, b)),
   (('inot', ('ige(is_used_once)', a, b)), ('ilt', a, b)),
   (('inot', ('uge(is_used_once)', a, b)), ('ult', a, b)),
   (('inot', ('ieq(is_used_once)', a, b)), ('ine', a, b)),
   (('inot', ('ine(is_used_once)', a, b)), ('ieq', a, b)),

   (('iand', ('feq', a, b), ('fneu', a, b)), False),
   (('iand', ('flt', a, b), ('flt', b, a)), False),
   (('iand', ('ieq', a, b), ('ine', a, b)), False),
   (('iand', ('ilt', a, b), ('ilt', b, a)), False),
   (('iand', ('ult', a, b), ('ult', b, a)), False),

   # This helps some shaders because, after some optimizations, they end up
   # with patterns like (-a < -b) || (b < a).  In an ideal world, this sort of
   # matching would be handled by CSE.
   (('flt', ('fneg', a), ('fneg', b)), ('flt', b, a)),
   (('fge', ('fneg', a), ('fneg', b)), ('fge', b, a)),
   (('feq', ('fneg', a), ('fneg', b)), ('feq', b, a)),
   (('fneu', ('fneg', a), ('fneg', b)), ('fneu', b, a)),
   (('flt', ('fneg', a), -1.0), ('flt', 1.0, a)),
   (('flt', -1.0, ('fneg', a)), ('flt', a, 1.0)),
   (('fge', ('fneg', a), -1.0), ('fge', 1.0, a)),
   (('fge', -1.0, ('fneg', a)), ('fge', a, 1.0)),
   (('fneu', ('fneg', a), -1.0), ('fneu', 1.0, a)),
   (('feq', -1.0, ('fneg', a)), ('feq', a, 1.0)),

   (('ieq', ('ineg', a), 0),  ('ieq', a, 0)),
   (('ine', ('ineg', a), 0),  ('ine', a, 0)),
   (('ieq', ('iabs', a), 0),  ('ieq', a, 0)),
   (('ine', ('iabs', a), 0),  ('ine', a, 0)),

   # b < fsat(NaN) -> b < 0 -> false, and b < Nan -> false.
   (('flt', '#b(is_gt_0_and_lt_1)', ('fsat(is_used_once)', a)), ('flt', b, a)),

   # fsat(NaN) >= b -> 0 >= b -> false, and NaN >= b -> false.
   (('fge', ('fsat(is_used_once)', a), '#b(is_gt_0_and_lt_1)'), ('fge', a, b)),

   # b == fsat(NaN) -> b == 0 -> false, and b == NaN -> false.
   (('feq', ('fsat(is_used_once)', a), '#b(is_gt_0_and_lt_1)'), ('feq', a, b)),

   # b != fsat(NaN) -> b != 0 -> true, and b != NaN -> true.
   (('fneu', ('fsat(is_used_once)', a), '#b(is_gt_0_and_lt_1)'), ('fneu', a, b)),

   # fsat(NaN) >= 1 -> 0 >= 1 -> false, and NaN >= 1 -> false.
   (('fge', ('fsat(is_used_once)', a), 1.0), ('fge', a, 1.0)),

   # 0 < fsat(NaN) -> 0 < 0 -> false, and 0 < NaN -> false.
   (('flt', 0.0, ('fsat(is_used_once)', a)), ('flt', 0.0, a)),

   # 0.0 >= b2f(a)
   # b2f(a) <= 0.0
   # b2f(a) == 0.0 because b2f(a) can only be 0 or 1
   # inot(a)
   (('fge', 0.0, ('b2f', 'a@1')), ('inot', a)),

   (('fge', ('fneg', ('b2f', 'a@1')), 0.0), ('inot', a)),

   (('fneu', ('fadd', ('b2f', 'a@1'), ('b2f', 'b@1')), 0.0), ('ior', a, b)),
   (('fneu', ('bcsel', a, 1.0, ('b2f', 'b@1'))   , 0.0), ('ior', a, b)),
   (('fneu', ('b2f', 'a@1'), ('fneg', ('b2f', 'b@1'))),      ('ior', a, b)),
   (('fneu', ('fmul', ('b2f', 'a@1'), ('b2f', 'b@1')), 0.0), ('iand', a, b)),
   (('fneu', ('bcsel', a, ('b2f', 'b@1'), 0.0)   , 0.0), ('iand', a, b)),
   (('fneu', ('fadd', ('b2f', 'a@1'), ('fneg', ('b2f', 'b@1'))), 0.0), ('ixor', a, b)),
   (('fneu',          ('b2f', 'a@1') ,          ('b2f', 'b@1') ),      ('ixor', a, b)),
   (('fneu', ('fneg', ('b2f', 'a@1')), ('fneg', ('b2f', 'b@1'))),      ('ixor', a, b)),
   (('feq', ('fadd', ('b2f', 'a@1'), ('b2f', 'b@1')), 0.0), ('inot', ('ior', a, b))),
   (('feq', ('bcsel', a, 1.0, ('b2f', 'b@1'))   , 0.0), ('inot', ('ior', a, b))),
   (('feq', ('b2f', 'a@1'), ('fneg', ('b2f', 'b@1'))),      ('inot', ('ior', a, b))),
   (('feq', ('fmul', ('b2f', 'a@1'), ('b2f', 'b@1')), 0.0), ('inot', ('iand', a, b))),
   (('feq', ('bcsel', a, ('b2f', 'b@1'), 0.0)   , 0.0), ('inot', ('iand', a, b))),
   (('feq', ('fadd', ('b2f', 'a@1'), ('fneg', ('b2f', 'b@1'))), 0.0), ('ieq', a, b)),
   (('feq',          ('b2f', 'a@1') ,          ('b2f', 'b@1') ),      ('ieq', a, b)),
   (('feq', ('fneg', ('b2f', 'a@1')), ('fneg', ('b2f', 'b@1'))),      ('ieq', a, b)),

   # -(b2f(a) + b2f(b)) < 0
   # 0 < b2f(a) + b2f(b)
   # 0 != b2f(a) + b2f(b)       b2f must be 0 or 1, so the sum is non-negative
   # a || b
   (('flt', ('fneg', ('fadd', ('b2f', 'a@1'), ('b2f', 'b@1'))), 0.0), ('ior', a, b)),
   (('flt', 0.0, ('fadd', ('b2f', 'a@1'), ('b2f', 'b@1'))), ('ior', a, b)),

   # -(b2f(a) + b2f(b)) >= 0
   # 0 >= b2f(a) + b2f(b)
   # 0 == b2f(a) + b2f(b)       b2f must be 0 or 1, so the sum is non-negative
   # !(a || b)
   (('fge', ('fneg', ('fadd', ('b2f', 'a@1'), ('b2f', 'b@1'))), 0.0), ('inot', ('ior', a, b))),
   (('fge', 0.0, ('fadd', ('b2f', 'a@1'), ('b2f', 'b@1'))), ('inot', ('ior', a, b))),

   (('flt', a, ('fneg', a)), ('flt', a, 0.0)),
   (('fge', a, ('fneg', a)), ('fge', a, 0.0)),

   # Some optimizations (below) convert things like (a < b || c < b) into
   # (min(a, c) < b).  However, this interfers with the previous optimizations
   # that try to remove comparisons with negated sums of b2f.  This just
   # breaks that apart.
   (('flt', ('fmin', c, ('fneg', ('fadd', ('b2f', 'a@1'), ('b2f', 'b@1')))), 0.0),
    ('ior', ('flt', c, 0.0), ('ior', a, b))),

   (('~flt', ('fadd', a, b), a), ('flt', b, 0.0)),
   (('~fge', ('fadd', a, b), a), ('fge', b, 0.0)),
   (('~feq', ('fadd', a, b), a), ('feq', b, 0.0)),
   (('~fneu', ('fadd', a, b), a), ('fneu', b, 0.0)),
   (('~flt',                        ('fadd(is_used_once)', a, '#b'),  '#c'), ('flt', a, ('fadd', c, ('fneg', b)))),
   (('~flt', ('fneg(is_used_once)', ('fadd(is_used_once)', a, '#b')), '#c'), ('flt', ('fneg', ('fadd', c, b)), a)),
   (('~fge',                        ('fadd(is_used_once)', a, '#b'),  '#c'), ('fge', a, ('fadd', c, ('fneg', b)))),
   (('~fge', ('fneg(is_used_once)', ('fadd(is_used_once)', a, '#b')), '#c'), ('fge', ('fneg', ('fadd', c, b)), a)),
   (('~feq',                        ('fadd(is_used_once)', a, '#b'),  '#c'), ('feq', a, ('fadd', c, ('fneg', b)))),
   (('~feq', ('fneg(is_used_once)', ('fadd(is_used_once)', a, '#b')), '#c'), ('feq', ('fneg', ('fadd', c, b)), a)),
   (('~fneu',                        ('fadd(is_used_once)', a, '#b'),  '#c'), ('fneu', a, ('fadd', c, ('fneg', b)))),
   (('~fneu', ('fneg(is_used_once)', ('fadd(is_used_once)', a, '#b')), '#c'), ('fneu', ('fneg', ('fadd', c, b)), a)),

   # Cannot remove the addition from ilt or ige due to overflow.
   (('ieq', ('iadd', a, b), a), ('ieq', b, 0)),
   (('ine', ('iadd', a, b), a), ('ine', b, 0)),

   (('feq', ('b2f', 'a@1'), 0.0), ('inot', a)),
   (('fneu', ('b2f', 'a@1'), 0.0), a),
   (('ieq', ('b2i', 'a@1'), 0),   ('inot', a)),
   (('ine', ('b2i', 'a@1'), 0),   a),

   (('fneu', ('u2f', a), 0.0), ('ine', a, 0)),
   (('feq', ('u2f', a), 0.0), ('ieq', a, 0)),
   (('fge', ('u2f', a), 0.0), True),
   (('fge', 0.0, ('u2f', a)), ('uge', 0, a)),    # ieq instead?
   (('flt', ('u2f', a), 0.0), False),
   (('flt', 0.0, ('u2f', a)), ('ult', 0, a)),    # ine instead?
   (('fneu', ('i2f', a), 0.0), ('ine', a, 0)),
   (('feq', ('i2f', a), 0.0), ('ieq', a, 0)),
   (('fge', ('i2f', a), 0.0), ('ige', a, 0)),
   (('fge', 0.0, ('i2f', a)), ('ige', 0, a)),
   (('flt', ('i2f', a), 0.0), ('ilt', a, 0)),
   (('flt', 0.0, ('i2f', a)), ('ilt', 0, a)),

   # 0.0 < fabs(a)
   # fabs(a) > 0.0
   # fabs(a) != 0.0 because fabs(a) must be >= 0
   # a != 0.0
   (('~flt', 0.0, ('fabs', a)), ('fneu', a, 0.0)),

   # -fabs(a) < 0.0
   # fabs(a) > 0.0
   (('~flt', ('fneg', ('fabs', a)), 0.0), ('fneu', a, 0.0)),

   # 0.0 >= fabs(a)
   # 0.0 == fabs(a)   because fabs(a) must be >= 0
   # 0.0 == a
   (('fge', 0.0, ('fabs', a)), ('feq', a, 0.0)),

   # -fabs(a) >= 0.0
   # 0.0 >= fabs(a)
   (('fge', ('fneg', ('fabs', a)), 0.0), ('feq', a, 0.0)),

   # (a >= 0.0) && (a <= 1.0) -> fsat(a) == a
   #
   # This should be NaN safe.
   #
   # NaN >= 0 && 1 >= NaN -> false && false -> false
   #
   # vs.
   #
   # NaN == fsat(NaN) -> NaN == 0 -> false
   (('iand', ('fge', a, 0.0), ('fge', 1.0, a)), ('feq', a, ('fsat', a)), '!options->lower_fsat'),

   # Note: fmin(-a, -b) == -fmax(a, b)
   (('fmax',                        ('b2f(is_used_once)', 'a@1'),           ('b2f', 'b@1')),           ('b2f', ('ior', a, b))),
   (('fmax', ('fneg(is_used_once)', ('b2f(is_used_once)', 'a@1')), ('fneg', ('b2f', 'b@1'))), ('fneg', ('b2f', ('iand', a, b)))),
   (('fmin',                        ('b2f(is_used_once)', 'a@1'),           ('b2f', 'b@1')),           ('b2f', ('iand', a, b))),
   (('fmin', ('fneg(is_used_once)', ('b2f(is_used_once)', 'a@1')), ('fneg', ('b2f', 'b@1'))), ('fneg', ('b2f', ('ior', a, b)))),

   # fmin(b2f(a), b)
   # bcsel(a, fmin(b2f(a), b), fmin(b2f(a), b))
   # bcsel(a, fmin(b2f(True), b), fmin(b2f(False), b))
   # bcsel(a, fmin(1.0, b), fmin(0.0, b))
   #
   # Since b is a constant, constant folding will eliminate the fmin and the
   # fmax.  If b is > 1.0, the bcsel will be replaced with a b2f.
   (('fmin', ('b2f', 'a@1'), '#b'), ('bcsel', a, ('fmin', b, 1.0), ('fmin', b, 0.0))),

   (('flt', ('fadd(is_used_once)', a, ('fneg', b)), 0.0), ('flt', a, b)),

   (('fge', ('fneg', ('fabs', a)), 0.0), ('feq', a, 0.0)),
   (('~bcsel', ('flt', b, a), b, a), ('fmin', a, b)),
   (('~bcsel', ('flt', a, b), b, a), ('fmax', a, b)),
   (('~bcsel', ('fge', a, b), b, a), ('fmin', a, b)),
   (('~bcsel', ('fge', b, a), b, a), ('fmax', a, b)),
   (('bcsel', ('ult', b, a), b, a), ('umin', a, b)),
   (('bcsel', ('ult', a, b), b, a), ('umax', a, b)),
   (('bcsel', ('uge', a, b), b, a), ('umin', a, b)),
   (('bcsel', ('uge', b, a), b, a), ('umax', a, b)),
   (('bcsel', ('ilt', b, a), b, a), ('imin', a, b)),
   (('bcsel', ('ilt', a, b), b, a), ('imax', a, b)),
   (('bcsel', ('ige', a, b), b, a), ('imin', a, b)),
   (('bcsel', ('ige', b, a), b, a), ('imax', a, b)),
   (('bcsel', ('inot', a), b, c), ('bcsel', a, c, b)),
   (('bcsel', a, ('bcsel', a, b, c), d), ('bcsel', a, b, d)),
   (('bcsel', a, b, ('bcsel', a, c, d)), ('bcsel', a, b, d)),
   (('bcsel', a, ('bcsel', b, c, d), ('bcsel(is_used_once)', b, c, 'e')), ('bcsel', b, c, ('bcsel', a, d, 'e'))),
   (('bcsel', a, ('bcsel(is_used_once)', b, c, d), ('bcsel', b, c, 'e')), ('bcsel', b, c, ('bcsel', a, d, 'e'))),
   (('bcsel', a, ('bcsel', b, c, d), ('bcsel(is_used_once)', b, 'e', d)), ('bcsel', b, ('bcsel', a, c, 'e'), d)),
   (('bcsel', a, ('bcsel(is_used_once)', b, c, d), ('bcsel', b, 'e', d)), ('bcsel', b, ('bcsel', a, c, 'e'), d)),
   (('bcsel', a, True, b), ('ior', a, b)),
   (('bcsel', a, a, b), ('ior', a, b)),
   (('bcsel', a, b, False), ('iand', a, b)),
   (('bcsel', a, b, a), ('iand', a, b)),
   (('~fmin', a, a), a),
   (('~fmax', a, a), a),
   (('imin', a, a), a),
   (('imax', a, a), a),
   (('umin', a, a), a),
   (('umin', a, 0), 0),
   (('umin', a, -1), a),
   (('umax', a, a), a),
   (('umax', a, 0), a),
   (('umax', a, -1), -1),
   (('fmax', ('fmax', a, b), b), ('fmax', a, b)),
   (('umax', ('umax', a, b), b), ('umax', a, b)),
   (('imax', ('imax', a, b), b), ('imax', a, b)),
   (('fmin', ('fmin', a, b), b), ('fmin', a, b)),
   (('umin', ('umin', a, b), b), ('umin', a, b)),
   (('imin', ('imin', a, b), b), ('imin', a, b)),
   (('fmax', ('fmax', ('fmax', a, b), c), a), ('fmax', ('fmax', a, b), c)),
   (('umax', ('umax', ('umax', a, b), c), a), ('umax', ('umax', a, b), c)),
   (('imax', ('imax', ('imax', a, b), c), a), ('imax', ('imax', a, b), c)),
   (('fmin', ('fmin', ('fmin', a, b), c), a), ('fmin', ('fmin', a, b), c)),
   (('umin', ('umin', ('umin', a, b), c), a), ('umin', ('umin', a, b), c)),
   (('imin', ('imin', ('imin', a, b), c), a), ('imin', ('imin', a, b), c)),
])

for N in [8, 16, 32, 64]:
    b2iN = 'b2i{0}'.format(N)
    optimizations.extend([
        (('ieq', (b2iN, 'a@1'), (b2iN, 'b@1')), ('ieq', a, b)),
        (('ine', (b2iN, 'a@1'), (b2iN, 'b@1')), ('ine', a, b)),
    ])

for N in [16, 32, 64]:
    b2fN = 'b2f{0}'.format(N)
    optimizations.extend([
        (('feq', (b2fN, 'a@1'), (b2fN, 'b@1')), ('ieq', a, b)),
        (('fneu', (b2fN, 'a@1'), (b2fN, 'b@1')), ('ine', a, b)),
    ])

# Integer sizes
for s in [8, 16, 32, 64]:
    optimizations.extend([
       (('iand@{}'.format(s), a, ('inot', ('ishr', a, s - 1))), ('imax', a, 0)),

       # Simplify logic to detect sign of an integer.
       (('ieq', ('iand', 'a@{}'.format(s), 1 << (s - 1)), 0),            ('ige', a, 0)),
       (('ine', ('iand', 'a@{}'.format(s), 1 << (s - 1)), 1 << (s - 1)), ('ige', a, 0)),
       (('ine', ('iand', 'a@{}'.format(s), 1 << (s - 1)), 0),            ('ilt', a, 0)),
       (('ieq', ('iand', 'a@{}'.format(s), 1 << (s - 1)), 1 << (s - 1)), ('ilt', a, 0)),
       (('ine', ('ushr', 'a@{}'.format(s), s - 1), 0), ('ilt', a, 0)),
       (('ieq', ('ushr', 'a@{}'.format(s), s - 1), 0), ('ige', a, 0)),
       (('ieq', ('ushr', 'a@{}'.format(s), s - 1), 1), ('ilt', a, 0)),
       (('ine', ('ushr', 'a@{}'.format(s), s - 1), 1), ('ige', a, 0)),
       (('ine', ('ishr', 'a@{}'.format(s), s - 1), 0), ('ilt', a, 0)),
       (('ieq', ('ishr', 'a@{}'.format(s), s - 1), 0), ('ige', a, 0)),
       (('ieq', ('ishr', 'a@{}'.format(s), s - 1), -1), ('ilt', a, 0)),
       (('ine', ('ishr', 'a@{}'.format(s), s - 1), -1), ('ige', a, 0)),
    ])

optimizations.extend([
   (('fmin', a, ('fneg', a)), ('fneg', ('fabs', a))),
   (('imin', a, ('ineg', a)), ('ineg', ('iabs', a))),
   (('fmin', a, ('fneg', ('fabs', a))), ('fneg', ('fabs', a))),
   (('imin', a, ('ineg', ('iabs', a))), ('ineg', ('iabs', a))),
   (('~fmin', a, ('fabs', a)), a),
   (('imin', a, ('iabs', a)), a),
   (('~fmax', a, ('fneg', ('fabs', a))), a),
   (('imax', a, ('ineg', ('iabs', a))), a),
   (('fmax', a, ('fabs', a)), ('fabs', a)),
   (('imax', a, ('iabs', a)), ('iabs', a)),
   (('fmax', a, ('fneg', a)), ('fabs', a)),
   (('imax', a, ('ineg', a)), ('iabs', a), '!options->lower_iabs'),
   (('~fmax', ('fabs', a), 0.0), ('fabs', a)),
   (('fmin', ('fmax', a, 0.0), 1.0), ('fsat', a), '!options->lower_fsat'),
   # fmax(fmin(a, 1.0), 0.0) is inexact because it returns 1.0 on NaN, while
   # fsat(a) returns 0.0.
   (('~fmax', ('fmin', a, 1.0), 0.0), ('fsat', a), '!options->lower_fsat'),
   # fmin(fmax(a, -1.0), 0.0) is inexact because it returns -1.0 on NaN, while
   # fneg(fsat(fneg(a))) returns -0.0 on NaN.
   (('~fmin', ('fmax', a, -1.0),  0.0), ('fneg', ('fsat', ('fneg', a))), '!options->lower_fsat'),
   # fmax(fmin(a, 0.0), -1.0) is inexact because it returns 0.0 on NaN, while
   # fneg(fsat(fneg(a))) returns -0.0 on NaN. This only matters if
   # SignedZeroInfNanPreserve is set, but we don't currently have any way of
   # representing this in the optimizations other than the usual ~.
   (('~fmax', ('fmin', a,  0.0), -1.0), ('fneg', ('fsat', ('fneg', a))), '!options->lower_fsat'),
   # fsat(fsign(NaN)) = fsat(0) = 0, and b2f(0 < NaN) = b2f(False) = 0. Mark
   # the new comparison precise to prevent it being changed to 'a != 0'.
   (('fsat', ('fsign', a)), ('b2f', ('!flt', 0.0, a))),
   (('fsat', ('b2f', a)), ('b2f', a)),
   (('fsat', a), ('fmin', ('fmax', a, 0.0), 1.0), 'options->lower_fsat'),
   (('fsat', ('fsat', a)), ('fsat', a)),
   (('fsat', ('fneg(is_used_once)', ('fadd(is_used_once)', a, b))), ('fsat', ('fadd', ('fneg', a), ('fneg', b))), '!options->lower_fsat'),
   (('fsat', ('fneg(is_used_once)', ('fmul(is_used_once)', a, b))), ('fsat', ('fmul', ('fneg', a), b)), '!options->lower_fsat'),
   (('fsat', ('fneg(is_used_once)', ('fmulz(is_used_once)', a, b))), ('fsat', ('fmulz', ('fneg', a), b)), '!options->lower_fsat && !'+signed_zero_inf_nan_preserve_32),
   (('fsat', ('fabs(is_used_once)', ('fmul(is_used_once)', a, b))), ('fsat', ('fmul', ('fabs', a), ('fabs', b))), '!options->lower_fsat'),
   (('fmin', ('fmax', ('fmin', ('fmax', a, b), c), b), c), ('fmin', ('fmax', a, b), c)),
   (('imin', ('imax', ('imin', ('imax', a, b), c), b), c), ('imin', ('imax', a, b), c)),
   (('umin', ('umax', ('umin', ('umax', a, b), c), b), c), ('umin', ('umax', a, b), c)),
   # Both the left and right patterns are "b" when isnan(a), so this is exact.
   (('fmax', ('fsat', a), '#b(is_zero_to_one)'), ('fsat', ('fmax', a, b))),
   # The left pattern is 0.0 when isnan(a) (because fmin(fsat(NaN), b) ->
   # fmin(0.0, b)) while the right one is "b", so this optimization is inexact.
   (('~fmin', ('fsat', a), '#b(is_zero_to_one)'), ('fsat', ('fmin', a, b))),

   # max(-min(b, a), b) -> max(abs(b), -a)
   # min(-max(b, a), b) -> min(-abs(b), -a)
   (('fmax', ('fneg', ('fmin', b, a)), b), ('fmax', ('fabs', b), ('fneg', a))),
   (('fmin', ('fneg', ('fmax', b, a)), b), ('fmin', ('fneg', ('fabs', b)), ('fneg', a))),

   # If a in [0,b] then b-a is also in [0,b].  Since b in [0,1], max(b-a, 0) =
   # fsat(b-a).
   #
   # If a > b, then b-a < 0 and max(b-a, 0) = fsat(b-a) = 0
   #
   # This should be NaN safe since max(NaN, 0) = fsat(NaN) = 0.
   (('fmax', ('fadd(is_used_once)', ('fneg', 'a(is_not_negative)'), '#b(is_zero_to_one)'), 0.0),
    ('fsat', ('fadd', ('fneg',  a), b)), '!options->lower_fsat'),

   (('extract_u8', ('imin', ('imax', a, 0), 0xff), 0), ('imin', ('imax', a, 0), 0xff)),

   # The ior versions are exact because fmin and fmax will always pick a
   # non-NaN value, if one exists.  Therefore (a < NaN) || (a < c) == a <
   # fmax(NaN, c) == a < c.  Mark the fmin or fmax in the replacement as exact
   # to prevent other optimizations from ruining the "NaN clensing" property
   # of the fmin or fmax.
   (('ior', ('flt(is_used_once)', a, b), ('flt', a, c)), ('flt', a, ('!fmax', b, c))),
   (('ior', ('flt(is_used_once)', a, c), ('flt', b, c)), ('flt', ('!fmin', a, b), c)),
   (('ior', ('fge(is_used_once)', a, b), ('fge', a, c)), ('fge', a, ('!fmin', b, c))),
   (('ior', ('fge(is_used_once)', a, c), ('fge', b, c)), ('fge', ('!fmax', a, b), c)),
   (('ior', ('flt', a, '#b'), ('flt', a, '#c')), ('flt', a, ('!fmax', b, c))),
   (('ior', ('flt', '#a', c), ('flt', '#b', c)), ('flt', ('!fmin', a, b), c)),
   (('ior', ('fge', a, '#b'), ('fge', a, '#c')), ('fge', a, ('!fmin', b, c))),
   (('ior', ('fge', '#a', c), ('fge', '#b', c)), ('fge', ('!fmax', a, b), c)),
   (('~iand', ('flt(is_used_once)', a, b), ('flt', a, c)), ('flt', a, ('fmin', b, c))),
   (('~iand', ('flt(is_used_once)', a, c), ('flt', b, c)), ('flt', ('fmax', a, b), c)),
   (('~iand', ('fge(is_used_once)', a, b), ('fge', a, c)), ('fge', a, ('fmax', b, c))),
   (('~iand', ('fge(is_used_once)', a, c), ('fge', b, c)), ('fge', ('fmin', a, b), c)),
   (('iand', ('flt', a, '#b(is_a_number)'), ('flt', a, '#c(is_a_number)')), ('flt', a, ('fmin', b, c))),
   (('iand', ('flt', '#a(is_a_number)', c), ('flt', '#b(is_a_number)', c)), ('flt', ('fmax', a, b), c)),
   (('iand', ('fge', a, '#b(is_a_number)'), ('fge', a, '#c(is_a_number)')), ('fge', a, ('fmax', b, c))),
   (('iand', ('fge', '#a(is_a_number)', c), ('fge', '#b(is_a_number)', c)), ('fge', ('fmin', a, b), c)),

   (('ior', ('ilt(is_used_once)', a, b), ('ilt', a, c)), ('ilt', a, ('imax', b, c))),
   (('ior', ('ilt(is_used_once)', a, c), ('ilt', b, c)), ('ilt', ('imin', a, b), c)),
   (('ior', ('ige(is_used_once)', a, b), ('ige', a, c)), ('ige', a, ('imin', b, c))),
   (('ior', ('ige(is_used_once)', a, c), ('ige', b, c)), ('ige', ('imax', a, b), c)),
   (('ior', ('ult(is_used_once)', a, b), ('ult', a, c)), ('ult', a, ('umax', b, c))),
   (('ior', ('ult(is_used_once)', a, c), ('ult', b, c)), ('ult', ('umin', a, b), c)),
   (('ior', ('uge(is_used_once)', a, b), ('uge', a, c)), ('uge', a, ('umin', b, c))),
   (('ior', ('uge(is_used_once)', a, c), ('uge', b, c)), ('uge', ('umax', a, b), c)),
   (('iand', ('ilt(is_used_once)', a, b), ('ilt', a, c)), ('ilt', a, ('imin', b, c))),
   (('iand', ('ilt(is_used_once)', a, c), ('ilt', b, c)), ('ilt', ('imax', a, b), c)),
   (('iand', ('ige(is_used_once)', a, b), ('ige', a, c)), ('ige', a, ('imax', b, c))),
   (('iand', ('ige(is_used_once)', a, c), ('ige', b, c)), ('ige', ('imin', a, b), c)),
   (('iand', ('ult(is_used_once)', a, b), ('ult', a, c)), ('ult', a, ('umin', b, c))),
   (('iand', ('ult(is_used_once)', a, c), ('ult', b, c)), ('ult', ('umax', a, b), c)),
   (('iand', ('uge(is_used_once)', a, b), ('uge', a, c)), ('uge', a, ('umax', b, c))),
   (('iand', ('uge(is_used_once)', a, c), ('uge', b, c)), ('uge', ('umin', a, b), c)),

   # A number of shaders contain a pattern like a.x < 0.0 || a.x > 1.0 || a.y
   # < 0.0, || a.y > 1.0 || ...  These patterns rearrange and replace in a
   # single step.  Doing just the replacement can lead to an infinite loop as
   # the pattern is repeatedly applied to the result of the previous
   # application of the pattern.
   (('ior', ('ior(is_used_once)', ('flt(is_used_once)', a, c), d), ('flt', b, c)), ('ior', ('flt', ('!fmin', a, b), c), d)),
   (('ior', ('ior(is_used_once)', ('flt', a, c), d), ('flt(is_used_once)', b, c)), ('ior', ('flt', ('!fmin', a, b), c), d)),
   (('ior', ('ior(is_used_once)', ('flt(is_used_once)', a, b), d), ('flt', a, c)), ('ior', ('flt', a, ('!fmax', b, c)), d)),
   (('ior', ('ior(is_used_once)', ('flt', a, b), d), ('flt(is_used_once)', a, c)), ('ior', ('flt', a, ('!fmax', b, c)), d)),

   # This is how SpvOpFOrdNotEqual might be implemented.  If both values are
   # numbers, then it can be replaced with fneu.
   (('ior', ('flt', 'a(is_a_number)', 'b(is_a_number)'), ('flt', b, a)), ('fneu', a, b)),

   # Other patterns may optimize the resulting iand tree further.
   (('umin', ('iand', a, '#b(is_pos_power_of_two)'), ('iand', c, b)),
    ('iand', ('iand', a, b), ('iand', c, b))),
])

# Float sizes
for s in [16, 32, 64]:
    optimizations.extend([
       # These derive from the previous patterns with the application of b < 0 <=>
       # 0 < -b.  The transformation should be applied if either comparison is
       # used once as this ensures that the number of comparisons will not
       # increase.  The sources to the ior and iand are not symmetric, so the
       # rules have to be duplicated to get this behavior.
       (('ior', ('flt(is_used_once)', 0.0, 'a@{}'.format(s)), ('flt', 'b@{}'.format(s), 0.0)), ('flt', 0.0, ('fmax', a, ('fneg', b)))),
       (('ior', ('flt', 0.0, 'a@{}'.format(s)), ('flt(is_used_once)', 'b@{}'.format(s), 0.0)), ('flt', 0.0, ('fmax', a, ('fneg', b)))),
       (('ior', ('fge(is_used_once)', 0.0, 'a@{}'.format(s)), ('fge', 'b@{}'.format(s), 0.0)), ('fge', 0.0, ('fmin', a, ('fneg', b)))),
       (('ior', ('fge', 0.0, 'a@{}'.format(s)), ('fge(is_used_once)', 'b@{}'.format(s), 0.0)), ('fge', 0.0, ('fmin', a, ('fneg', b)))),
       (('~iand', ('flt(is_used_once)', 0.0, 'a@{}'.format(s)), ('flt', 'b@{}'.format(s), 0.0)), ('flt', 0.0, ('fmin', a, ('fneg', b)))),
       (('~iand', ('flt', 0.0, 'a@{}'.format(s)), ('flt(is_used_once)', 'b@{}'.format(s), 0.0)), ('flt', 0.0, ('fmin', a, ('fneg', b)))),
       (('~iand', ('fge(is_used_once)', 0.0, 'a@{}'.format(s)), ('fge', 'b@{}'.format(s), 0.0)), ('fge', 0.0, ('fmax', a, ('fneg', b)))),
       (('~iand', ('fge', 0.0, 'a@{}'.format(s)), ('fge(is_used_once)', 'b@{}'.format(s), 0.0)), ('fge', 0.0, ('fmax', a, ('fneg', b)))),

       # The (i2f32, ...) part is an open-coded fsign.  When that is combined
       # with the bcsel, it's basically copysign(1.0, a).  There are some
       # behavior differences between this pattern and copysign w.r.t. ±0 and
       # NaN.  copysign(x, y) blindly takes the sign bit from y and applies it
       # to x, regardless of whether either or both values are NaN.
       #
       # If a != a: bcsel(False, 1.0, i2f(b2i(False) - b2i(False))) = 0,
       #            int(NaN >= 0.0) - int(NaN < 0.0) = 0 - 0 = 0
       # If a == ±0: bcsel(True, 1.0, ...) = 1.0,
       #            int(±0.0 >= 0.0) - int(±0.0 < 0.0) = 1 - 0 = 1
       #
       # For all other values of 'a', the original and replacement behave as
       # copysign.
       #
       # Marking the replacement comparisons as precise prevents any future
       # optimizations from replacing either of the comparisons with the
       # logical-not of the other.
       #
       # Note: Use b2i32 in the replacement because some platforms that
       # support fp16 don't support int16.
       (('bcsel@{}'.format(s), ('feq', a, 0.0), 1.0, ('i2f{}'.format(s), ('iadd', ('b2i{}'.format(s), ('flt', 0.0, 'a@{}'.format(s))), ('ineg', ('b2i{}'.format(s), ('flt', 'a@{}'.format(s), 0.0)))))),
        ('i2f{}'.format(s), ('iadd', ('b2i32', ('!fge', a, 0.0)), ('ineg', ('b2i32', ('!flt', a, 0.0)))))),

       (('bcsel', a, ('b2f(is_used_once)', 'b@{}'.format(s)), ('b2f', 'c@{}'.format(s))), ('b2f', ('bcsel', a, b, c))),

       # The C spec says, "If the value of the integral part cannot be represented
       # by the integer type, the behavior is undefined."  "Undefined" can mean
       # "the conversion doesn't happen at all."
       (('~i2f{}'.format(s), ('f2i', 'a@{}'.format(s))), ('ftrunc', a)),

       # Ironically, mark these as imprecise because removing the conversions may
       # preserve more precision than doing the conversions (e.g.,
       # uint(float(0x81818181u)) == 0x81818200).
       (('~f2i{}'.format(s), ('i2f', 'a@{}'.format(s))), a),
       (('~f2i{}'.format(s), ('u2f', 'a@{}'.format(s))), a),
       (('~f2u{}'.format(s), ('i2f', 'a@{}'.format(s))), a),
       (('~f2u{}'.format(s), ('u2f', 'a@{}'.format(s))), a),

       (('fadd', ('b2f{}'.format(s), ('flt', 0.0, 'a@{}'.format(s))), ('fneg', ('b2f{}'.format(s), ('flt', 'a@{}'.format(s), 0.0)))), ('fsign', a), '!options->lower_fsign'),
       (('iadd', ('b2i{}'.format(s), ('flt', 0, 'a@{}'.format(s))), ('ineg', ('b2i{}'.format(s), ('flt', 'a@{}'.format(s), 0)))), ('f2i{}'.format(s), ('fsign', a)), '!options->lower_fsign'),
    ])

    # float? -> float? -> floatS ==> float? -> floatS
    (('~f2f{}'.format(s), ('f2f', a)), ('f2f{}'.format(s), a)),

    # int? -> float? -> floatS ==> int? -> floatS
    (('~f2f{}'.format(s), ('u2f', a)), ('u2f{}'.format(s), a)),
    (('~f2f{}'.format(s), ('i2f', a)), ('i2f{}'.format(s), a)),

    # float? -> float? -> intS ==> float? -> intS
    (('~f2u{}'.format(s), ('f2f', a)), ('f2u{}'.format(s), a)),
    (('~f2i{}'.format(s), ('f2f', a)), ('f2i{}'.format(s), a)),

    for B in [32, 64]:
        if s < B:
            optimizations.extend([
               # S = smaller, B = bigger
               # typeS -> typeB -> typeS ==> identity
               (('f2f{}'.format(s), ('f2f{}'.format(B), 'a@{}'.format(s))), a),
               (('i2i{}'.format(s), ('i2i{}'.format(B), 'a@{}'.format(s))), a),
               (('u2u{}'.format(s), ('u2u{}'.format(B), 'a@{}'.format(s))), a),

               # bool1 -> typeB -> typeS ==> bool1 -> typeS
               (('f2f{}'.format(s), ('b2f{}'.format(B), 'a@1')), ('b2f{}'.format(s), a)),
               (('i2i{}'.format(s), ('b2i{}'.format(B), 'a@1')), ('b2i{}'.format(s), a)),
               (('u2u{}'.format(s), ('b2i{}'.format(B), 'a@1')), ('b2i{}'.format(s), a)),

               # floatS -> floatB -> intB ==> floatS -> intB
               (('f2u{}'.format(B), ('f2f{}'.format(B), 'a@{}'.format(s))), ('f2u{}'.format(B), a)),
               (('f2i{}'.format(B), ('f2f{}'.format(B), 'a@{}'.format(s))), ('f2i{}'.format(B), a)),

               # int? -> floatB -> floatS ==> int? -> floatS
               (('f2f{}'.format(s), ('u2f{}'.format(B), a)), ('u2f{}'.format(s), a)),
               (('f2f{}'.format(s), ('i2f{}'.format(B), a)), ('i2f{}'.format(s), a)),

               # intS -> intB -> floatB ==> intS -> floatB
               (('u2f{}'.format(B), ('u2u{}'.format(B), 'a@{}'.format(s))), ('u2f{}'.format(B), a)),
               (('i2f{}'.format(B), ('i2i{}'.format(B), 'a@{}'.format(s))), ('i2f{}'.format(B), a)),
            ])

# mediump variants of the above
optimizations.extend([
    # int32 -> float32 -> float16 ==> int32 -> float16
    (('f2fmp', ('u2f32', 'a@32')), ('u2fmp', a)),
    (('f2fmp', ('i2f32', 'a@32')), ('i2fmp', a)),

    # float32 -> float16 -> int16 ==> float32 -> int16
    (('f2u16', ('f2fmp', 'a@32')), ('f2u16', a)),
    (('f2i16', ('f2fmp', 'a@32')), ('f2i16', a)),

    # float32 -> int32 -> int16 ==> float32 -> int16
    (('i2imp', ('f2u32', 'a@32')), ('f2ump', a)),
    (('i2imp', ('f2i32', 'a@32')), ('f2imp', a)),

    # int32 -> int16 -> float16 ==> int32 -> float16
    (('u2f16', ('i2imp', 'a@32')), ('u2f16', a)),
    (('i2f16', ('i2imp', 'a@32')), ('i2f16', a)),
])

# Clean up junk left from 8-bit integer to 16-bit integer lowering.
optimizations.extend([
    # The u2u16(u2u8(X)) just masks off the upper 8-bits of X.  This can be
    # accomplished by mask the upper 8-bit of the immediate operand to the
    # iand instruction.  Often times, both patterns will end up being applied
    # to the same original expression tree.
    (('iand', ('u2u16', ('u2u8', 'a@16')), '#b'),               ('iand', a, ('iand', b, 0xff))),
    (('u2u16', ('u2u8(is_used_once)', ('iand', 'a@16', '#b'))), ('iand', a, ('iand', b, 0xff))),
])

for op in ['iand', 'ior', 'ixor']:
    optimizations.extend([
        (('u2u8', (op, ('u2u16', ('u2u8', 'a@16')), ('u2u16', ('u2u8', 'b@16')))), ('u2u8', (op, a, b))),
        (('u2u8', (op, ('u2u16', ('u2u8', 'a@32')), ('u2u16', ('u2u8', 'b@32')))), ('u2u8', (op, a, b))),

        # Undistribute extract from a logic op
        ((op, ('extract_i8', a, '#b'), ('extract_i8', c, b)), ('extract_i8', (op, a, c), b)),
        ((op, ('extract_u8', a, '#b'), ('extract_u8', c, b)), ('extract_u8', (op, a, c), b)),
        ((op, ('extract_i16', a, '#b'), ('extract_i16', c, b)), ('extract_i16', (op, a, c), b)),
        ((op, ('extract_u16', a, '#b'), ('extract_u16', c, b)), ('extract_u16', (op, a, c), b)),

        # Undistribute shifts from a logic op
        ((op, ('ushr(is_used_once)', a, '#b'), ('ushr', c, b)), ('ushr', (op, a, c), b)),
        ((op, ('ishr(is_used_once)', a, '#b'), ('ishr', c, b)), ('ishr', (op, a, c), b)),
        ((op, ('ishl(is_used_once)', a, '#b'), ('ishl', c, b)), ('ishl', (op, a, c), b)),
    ])

# Integer sizes
for s in [8, 16, 32, 64]:
    last_shift_bit = int(math.log2(s)) - 1

    optimizations.extend([
       (('iand', ('ieq', 'a@{}'.format(s), 0), ('ieq', 'b@{}'.format(s), 0)), ('ieq', ('ior', a, b), 0), 'options->lower_umax'),
       (('ior',  ('ine', 'a@{}'.format(s), 0), ('ine', 'b@{}'.format(s), 0)), ('ine', ('ior', a, b), 0), 'options->lower_umin'),
       (('iand', ('ieq', 'a@{}'.format(s), 0), ('ieq', 'b@{}'.format(s), 0)), ('ieq', ('umax', a, b), 0), '!options->lower_umax'),
       (('ior',  ('ieq', 'a@{}'.format(s), 0), ('ieq', 'b@{}'.format(s), 0)), ('ieq', ('umin', a, b), 0), '!options->lower_umin'),
       (('iand', ('ine', 'a@{}'.format(s), 0), ('ine', 'b@{}'.format(s), 0)), ('ine', ('umin', a, b), 0), '!options->lower_umin'),
       (('ior',  ('ine', 'a@{}'.format(s), 0), ('ine', 'b@{}'.format(s), 0)), ('ine', ('umax', a, b), 0), '!options->lower_umax'),

       # True/False are ~0 and 0 in NIR.  b2i of True is 1, and -1 is ~0 (True).
       (('ineg', ('b2i{}'.format(s), 'a@{}'.format(s))), a),

       # SM5 32-bit shifts are defined to use the 5 least significant bits (or 4 bits for 16 bits)
       (('ishl', 'a@{}'.format(s), ('iand', s - 1, b)), ('ishl', a, b)),
       (('ishr', 'a@{}'.format(s), ('iand', s - 1, b)), ('ishr', a, b)),
       (('ushr', 'a@{}'.format(s), ('iand', s - 1, b)), ('ushr', a, b)),
       (('ushr', 'a@{}'.format(s), ('ishl(is_used_once)', ('iand', b, 1), last_shift_bit)), ('ushr', a, ('ishl', b, last_shift_bit))),
    ])

optimizations.extend([
   # Common pattern like 'if (i == 0 || i == 1 || ...)'
   (('ior', ('ieq', a, 0), ('ieq', a, 1)), ('uge', 1, a)),
   (('ior', ('uge', 1, a), ('ieq', a, 2)), ('uge', 2, a)),
   (('ior', ('uge', 2, a), ('ieq', a, 3)), ('uge', 3, a)),
   (('ior', a, ('ieq', a, False)), True),

   (('ine', ('ineg', ('b2i', 'a@1')), ('ineg', ('b2i', 'b@1'))), ('ine', a, b)),
   (('b2i', ('ine', 'a@1', 'b@1')), ('b2i', ('ixor', a, b))),

   (('ishl', ('b2i32', ('ine', ('iand', 'a@32', '#b(is_pos_power_of_two)'), 0)), '#c'),
    ('bcsel', ('ige', ('iand', c, 31), ('find_lsb', b)),
              ('ishl', ('iand', a, b), ('iadd', ('iand', c, 31), ('ineg', ('find_lsb', b)))),
              ('ushr', ('iand', a, b), ('iadd', ('ineg', ('iand', c, 31)), ('find_lsb', b)))
    )
   ),

   (('b2i32', ('ine', ('iand', 'a@32', '#b(is_pos_power_of_two)'), 0)),
    ('ushr', ('iand', a, b), ('find_lsb', b)), '!options->lower_bitops'),

   (('ior',  ('b2i', a), ('iand', b, 1)), ('iand', ('ior', ('b2i', a), b), 1)),
   (('iand', ('b2i', a), ('iand', b, 1)), ('iand', ('b2i', a), b)),

   # This pattern occurs coutresy of __flt64_nonnan in the soft-fp64 code.
   # The first part of the iand comes from the !__feq64_nonnan.
   #
   # The second pattern is a reformulation of the first based on the relation
   # (a == 0 || y == 0) <=> umin(a, y) == 0, where b in the first equation
   # happens to be y == 0.
   (('iand', ('inot', ('iand', ('ior', ('ieq', a, 0),  b), c)), ('ilt', a, 0)),
    ('iand', ('inot', ('iand',                         b , c)), ('ilt', a, 0))),
   (('iand', ('inot', ('iand', ('ieq', ('umin', a, b), 0), c)), ('ilt', a, 0)),
    ('iand', ('inot', ('iand', ('ieq',             b , 0), c)), ('ilt', a, 0))),

   # These patterns can result when (a < b || a < c) => (a < min(b, c))
   # transformations occur before constant propagation and loop-unrolling.
   #
   # The flt versions are exact.  If isnan(a), the original pattern is
   # trivially false, and the replacements are false too.  If isnan(b):
   #
   #    a < fmax(NaN, a) => a < a => false vs a < NaN => false
   (('flt', a, ('fmax', b, a)), ('flt', a, b)),
   (('flt', ('fmin', a, b), a), ('flt', b, a)),
   (('~fge', a, ('fmin', b, a)), True),
   (('~fge', ('fmax', a, b), a), True),
   (('flt', a, ('fmin', b, a)), False),
   (('flt', ('fmax', a, b), a), False),
   (('~fge', a, ('fmax', b, a)), ('fge', a, b)),
   (('~fge', ('fmin', a, b), a), ('fge', b, a)),

   (('ilt', a, ('imax', b, a)), ('ilt', a, b)),
   (('ilt', ('imin', a, b), a), ('ilt', b, a)),
   (('ige', a, ('imin', b, a)), True),
   (('ige', ('imax', a, b), a), True),
   (('ult', a, ('umax', b, a)), ('ult', a, b)),
   (('ult', ('umin', a, b), a), ('ult', b, a)),
   (('uge', a, ('umin', b, a)), True),
   (('uge', ('umax', a, b), a), True),
   (('ilt', a, ('imin', b, a)), False),
   (('ilt', ('imax', a, b), a), False),
   (('ige', a, ('imax', b, a)), ('ige', a, b)),
   (('ige', ('imin', a, b), a), ('ige', b, a)),
   (('ult', a, ('umin', b, a)), False),
   (('ult', ('umax', a, b), a), False),
   (('uge', a, ('umax', b, a)), ('uge', a, b)),
   (('uge', ('umin', a, b), a), ('uge', b, a)),
   (('ult', a, ('iand', b, a)), False),
   (('ult', ('ior', a, b), a), False),
   (('uge', a, ('iand', b, a)), True),
   (('uge', ('ior', a, b), a), True),

   (('ilt', '#a', ('imax', '#b', c)), ('ior', ('ilt', a, b), ('ilt', a, c))),
   (('ilt', ('imin', '#a', b), '#c'), ('ior', ('ilt', a, c), ('ilt', b, c))),
   (('ige', '#a', ('imin', '#b', c)), ('ior', ('ige', a, b), ('ige', a, c))),
   (('ige', ('imax', '#a', b), '#c'), ('ior', ('ige', a, c), ('ige', b, c))),
   (('ult', '#a', ('umax', '#b', c)), ('ior', ('ult', a, b), ('ult', a, c))),
   (('ult', ('umin', '#a', b), '#c'), ('ior', ('ult', a, c), ('ult', b, c))),
   (('uge', '#a', ('umin', '#b', c)), ('ior', ('uge', a, b), ('uge', a, c))),
   (('uge', ('umax', '#a', b), '#c'), ('ior', ('uge', a, c), ('uge', b, c))),
   (('ilt', '#a', ('imin', '#b', c)), ('iand', ('ilt', a, b), ('ilt', a, c))),
   (('ilt', ('imax', '#a', b), '#c'), ('iand', ('ilt', a, c), ('ilt', b, c))),
   (('ige', '#a', ('imax', '#b', c)), ('iand', ('ige', a, b), ('ige', a, c))),
   (('ige', ('imin', '#a', b), '#c'), ('iand', ('ige', a, c), ('ige', b, c))),
   (('ult', '#a', ('umin', '#b', c)), ('iand', ('ult', a, b), ('ult', a, c))),
   (('ult', ('umax', '#a', b), '#c'), ('iand', ('ult', a, c), ('ult', b, c))),
   (('uge', '#a', ('umax', '#b', c)), ('iand', ('uge', a, b), ('uge', a, c))),
   (('uge', ('umin', '#a', b), '#c'), ('iand', ('uge', a, c), ('uge', b, c))),

   # Thanks to sign extension, the ishr(a, b) is negative if and only if a is
   # negative.
   (('bcsel', ('ilt', a, 0), ('ineg', ('ishr', a, b)), ('ishr', a, b)),
    ('iabs', ('ishr', a, b))),
   (('iabs', ('ishr', ('iabs', a), b)), ('ishr', ('iabs', a), b)),

   (('fabs', ('slt', a, b)), ('slt', a, b)),
   (('fabs', ('sge', a, b)), ('sge', a, b)),
   (('fabs', ('seq', a, b)), ('seq', a, b)),
   (('fabs', ('sne', a, b)), ('sne', a, b)),
   (('slt', a, b), ('b2f', ('flt', a, b)), 'options->lower_scmp'),
   (('sge', a, b), ('b2f', ('fge', a, b)), 'options->lower_scmp'),
   (('seq', a, b), ('b2f', ('feq', a, b)), 'options->lower_scmp'),
   (('sne', a, b), ('b2f', ('fneu', a, b)), 'options->lower_scmp'),
   (('seq', ('seq', a, b), 1.0), ('seq', a, b)),
   (('seq', ('sne', a, b), 1.0), ('sne', a, b)),
   (('seq', ('slt', a, b), 1.0), ('slt', a, b)),
   (('seq', ('sge', a, b), 1.0), ('sge', a, b)),
   (('sne', ('seq', a, b), 0.0), ('seq', a, b)),
   (('sne', ('sne', a, b), 0.0), ('sne', a, b)),
   (('sne', ('slt', a, b), 0.0), ('slt', a, b)),
   (('sne', ('sge', a, b), 0.0), ('sge', a, b)),
   (('seq', ('seq', a, b), 0.0), ('sne', a, b)),
   (('seq', ('sne', a, b), 0.0), ('seq', a, b)),
   (('seq', ('slt', a, b), 0.0), ('sge', a, b)),
   (('seq', ('sge', a, b), 0.0), ('slt', a, b)),
   (('sne', ('seq', a, b), 1.0), ('sne', a, b)),
   (('sne', ('sne', a, b), 1.0), ('seq', a, b)),
   (('sne', ('slt', a, b), 1.0), ('sge', a, b)),
   (('sne', ('sge', a, b), 1.0), ('slt', a, b)),
   (('fall_equal2', a, b), ('fmin', ('seq', 'a.x', 'b.x'), ('seq', 'a.y', 'b.y')), 'options->lower_vector_cmp'),
   (('fall_equal3', a, b), ('seq', ('fany_nequal3', a, b), 0.0), 'options->lower_vector_cmp'),
   (('fall_equal4', a, b), ('seq', ('fany_nequal4', a, b), 0.0), 'options->lower_vector_cmp'),
   (('fall_equal8', a, b), ('seq', ('fany_nequal8', a, b), 0.0), 'options->lower_vector_cmp'),
   (('fall_equal16', a, b), ('seq', ('fany_nequal16', a, b), 0.0), 'options->lower_vector_cmp'),
   (('fany_nequal2', a, b), ('fmax', ('sne', 'a.x', 'b.x'), ('sne', 'a.y', 'b.y')), 'options->lower_vector_cmp'),
   (('fany_nequal3', a, b), ('fsat', ('fdot3', ('sne', a, b), ('sne', a, b))), 'options->lower_vector_cmp'),
   (('fany_nequal4', a, b), ('fsat', ('fdot4', ('sne', a, b), ('sne', a, b))), 'options->lower_vector_cmp'),
   (('fany_nequal8', a, b), ('fsat', ('fdot8', ('sne', a, b), ('sne', a, b))), 'options->lower_vector_cmp'),
   (('fany_nequal16', a, b), ('fsat', ('fdot16', ('sne', a, b), ('sne', a, b))), 'options->lower_vector_cmp'),
])

def vector_cmp(reduce_op, cmp_op, comps):
   if len(comps) == 1:
      return (cmp_op, 'a.' + comps[0], 'b.' + comps[0])
   else:
      mid = len(comps) // 2
      return (reduce_op, vector_cmp(reduce_op, cmp_op, comps[:mid]),
                         vector_cmp(reduce_op, cmp_op, comps[mid:]))

for op in [
   ('ball_iequal', 'ieq', 'iand'),
   ('ball_fequal', 'feq', 'iand'),
   ('bany_inequal', 'ine', 'ior'),
   ('bany_fnequal', 'fneu', 'ior'),
]:
   optimizations.extend([
      ((op[0] + '2', a, b), vector_cmp(op[2], op[1], 'xy'), 'options->lower_vector_cmp'),
      ((op[0] + '3', a, b), vector_cmp(op[2], op[1], 'xyz'), 'options->lower_vector_cmp'),
      ((op[0] + '4', a, b), vector_cmp(op[2], op[1], 'xyzw'), 'options->lower_vector_cmp'),
      ((op[0] + '8', a, b), vector_cmp(op[2], op[1], 'abcdefgh'), 'options->lower_vector_cmp'),
      ((op[0] + '16', a, b), vector_cmp(op[2], op[1], 'abcdefghijklmnop'), 'options->lower_vector_cmp'),
   ])

optimizations.extend([
   (('feq', ('seq', a, b), 1.0), ('feq', a, b)),
   (('feq', ('sne', a, b), 1.0), ('fneu', a, b)),
   (('feq', ('slt', a, b), 1.0), ('flt', a, b)),
   (('feq', ('sge', a, b), 1.0), ('fge', a, b)),
   (('fneu', ('seq', a, b), 0.0), ('feq', a, b)),
   (('fneu', ('sne', a, b), 0.0), ('fneu', a, b)),
   (('fneu', ('slt', a, b), 0.0), ('flt', a, b)),
   (('fneu', ('sge', a, b), 0.0), ('fge', a, b)),
   (('feq', ('seq', a, b), 0.0), ('fneu', a, b)),
   (('feq', ('sne', a, b), 0.0), ('feq', a, b)),
   (('feq', ('slt', a, b), 0.0), ('fge', a, b)),
   (('feq', ('sge', a, b), 0.0), ('flt', a, b)),
   (('fneu', ('seq', a, b), 1.0), ('fneu', a, b)),
   (('fneu', ('sne', a, b), 1.0), ('feq', a, b)),
   (('fneu', ('slt', a, b), 1.0), ('fge', a, b)),
   (('fneu', ('sge', a, b), 1.0), ('flt', a, b)),

   (('fneu', ('fneg', a), a), ('fneu', a, 0.0)),
   (('feq', ('fneg', a), a), ('feq', a, 0.0)),
   # Emulating booleans
   (('imul', ('b2i', 'a@1'), ('b2i', 'b@1')), ('b2i', ('iand', a, b))),
   (('iand', ('b2i', 'a@1'), ('b2i', 'b@1')), ('b2i', ('iand', a, b))),
   (('ior', ('b2i', 'a@1'), ('b2i', 'b@1')), ('b2i', ('ior', a, b))),
   (('fmul', ('b2f', 'a@1'), ('b2f', 'b@1')), ('b2f', ('iand', a, b))),
   (('fsat', ('fadd', ('b2f', 'a@1'), ('b2f', 'b@1'))), ('b2f', ('ior', a, b))),
   (('iand', 'a@bool16', 1.0), ('b2f', a)),
   (('iand', 'a@bool32', 1.0), ('b2f', a)),
   (('flt', ('fneg', ('b2f', 'a@1')), 0), a), # Generated by TGSI KILL_IF.
   # Comparison with the same args.  Note that these are only done for the
   # float versions when the source must be a number.  Generally, NaN cmp NaN
   # produces the opposite result of X cmp X.  flt is the outlier.  NaN < NaN
   # is false, and, for any number X, X < X is also false.
   (('ilt', a, a), False),
   (('ige', a, a), True),
   (('ieq', a, a), True),
   (('ine', a, a), False),
   (('ult', a, a), False),
   (('uge', a, a), True),
   (('flt', a, a), False),
   (('fge', 'a(is_a_number)', a), True),
   (('feq', 'a(is_a_number)', a), True),
   (('fneu', 'a(is_a_number)', a), False),
   # Logical and bit operations
   (('iand', a, a), a),
   (('iand', a, 0), 0),
   (('iand', a, -1), a),
   (('iand', a, ('inot', a)), 0),
   (('ior', a, a), a),
   (('ior', a, 0), a),
   (('ior', a, -1), -1),
   (('ior', a, ('inot', a)), -1),
   (('ixor', a, a), 0),
   (('ixor', a, 0), a),
   (('ixor', a, ('ixor', a, b)), b),
   (('ixor', a, -1), ('inot', a)),
   (('inot', ('inot', a)), a),
   (('ior', ('iand', a, b), b), b),
   (('ior', ('ior', a, b), b), ('ior', a, b)),
   (('iand', ('ior', a, b), b), b),
   (('iand', ('iand', a, b), b), ('iand', a, b)),

   # It is common for sequences of (x & 1) to occur in large trees.  Replacing
   # an expression like ((a & 1) & (b & 1)) with ((a & b) & 1) allows the "&
   # 1" to eventually bubble up to the top of the tree.
   (('iand', ('iand(is_used_once)', a, b), ('iand(is_used_once)', a, c)),
    ('iand', a, ('iand', b, c))),

   (('iand@64', a, '#b(is_lower_half_zero)'),
    ('pack_64_2x32_split', 0,
                           ('iand', ('unpack_64_2x32_split_y', a), ('unpack_64_2x32_split_y', b)))),
   (('iand@64', a, '#b(is_upper_half_zero)'),
    ('pack_64_2x32_split', ('iand', ('unpack_64_2x32_split_x', a), ('unpack_64_2x32_split_x', b)),
                           0)),
   (('iand@64', a, '#b(is_lower_half_negative_one)'),
    ('pack_64_2x32_split', ('unpack_64_2x32_split_x', a),
                           ('iand', ('unpack_64_2x32_split_y', a), ('unpack_64_2x32_split_y', b)))),
   (('iand@64', a, '#b(is_upper_half_negative_one)'),
    ('pack_64_2x32_split', ('iand', ('unpack_64_2x32_split_x', a), ('unpack_64_2x32_split_x', b)),
                           ('unpack_64_2x32_split_y', a))),

   (('ior@64', a, '#b(is_lower_half_zero)'),
    ('pack_64_2x32_split', ('unpack_64_2x32_split_x', a),
                           ('ior', ('unpack_64_2x32_split_y', a), ('unpack_64_2x32_split_y', b)))),
   (('ior@64', a, '#b(is_upper_half_zero)'),
    ('pack_64_2x32_split', ('ior', ('unpack_64_2x32_split_x', a), ('unpack_64_2x32_split_x', b)),
                           ('unpack_64_2x32_split_y', a))),
   (('ior@64', a, '#b(is_lower_half_negative_one)'),
    ('pack_64_2x32_split', -1,
                           ('ior', ('unpack_64_2x32_split_y', a), ('unpack_64_2x32_split_y', b)))),
   (('ior@64', a, '#b(is_upper_half_negative_one)'),
    ('pack_64_2x32_split', ('ior', ('unpack_64_2x32_split_x', a), ('unpack_64_2x32_split_x', b)),
                           -1)),

   (('ixor@64', a, '#b(is_lower_half_zero)'),
    ('pack_64_2x32_split', ('unpack_64_2x32_split_x', a),
                           ('ixor', ('unpack_64_2x32_split_y', a), ('unpack_64_2x32_split_y', b)))),
   (('ixor@64', a, '#b(is_upper_half_zero)'),
    ('pack_64_2x32_split', ('ixor', ('unpack_64_2x32_split_x', a), ('unpack_64_2x32_split_x', b)),
                           ('unpack_64_2x32_split_y', a))),

   # DeMorgan's Laws
   (('iand', ('inot', a), ('inot', b)), ('inot', ('ior',  a, b))),
   (('ior',  ('inot', a), ('inot', b)), ('inot', ('iand', a, b))),
   # Shift optimizations
   (('ishl', 0, a), 0),
   (('ishl', a, 0), a),
   (('ishr', 0, a), 0),
   (('ishr', -1, a), -1),
   (('ishr', a, 0), a),
   (('ushr', 0, a), 0),
   (('ushr', a, 0), a),
   (('ior', ('ishl@16', a, b), ('ushr@16', a, ('iadd', 16, ('ineg', b)))), ('urol', a, b), '!options->lower_rotate'),
   (('ior', ('ishl@16', a, b), ('ushr@16', a, ('isub', 16, b))), ('urol', a, b), '!options->lower_rotate'),
   (('ior', ('ishl@32', a, b), ('ushr@32', a, ('iadd', 32, ('ineg', b)))), ('urol', a, b), '!options->lower_rotate'),
   (('ior', ('ishl@32', a, b), ('ushr@32', a, ('isub', 32, b))), ('urol', a, b), '!options->lower_rotate'),
   (('ior', ('ushr@16', a, b), ('ishl@16', a, ('iadd', 16, ('ineg', b)))), ('uror', a, b), '!options->lower_rotate'),
   (('ior', ('ushr@16', a, b), ('ishl@16', a, ('isub', 16, b))), ('uror', a, b), '!options->lower_rotate'),
   (('ior', ('ushr@32', a, b), ('ishl@32', a, ('iadd', 32, ('ineg', b)))), ('uror', a, b), '!options->lower_rotate'),
   (('ior', ('ushr@32', a, b), ('ishl@32', a, ('isub', 32, b))), ('uror', a, b), '!options->lower_rotate'),
   (('urol@8',  a, b), ('ior', ('ishl', a, b), ('ushr', a, ('isub',  8, b))), 'options->lower_rotate'),
   (('urol@16', a, b), ('ior', ('ishl', a, b), ('ushr', a, ('isub', 16, b))), 'options->lower_rotate'),
   (('urol@32', a, b), ('ior', ('ishl', a, b), ('ushr', a, ('isub', 32, b))), 'options->lower_rotate'),
   (('urol@64', a, b), ('ior', ('ishl', a, b), ('ushr', a, ('isub', 64, b))), 'options->lower_rotate'),
   (('uror@8',  a, b), ('ior', ('ushr', a, b), ('ishl', a, ('isub',  8, b))), 'options->lower_rotate'),
   (('uror@16', a, b), ('ior', ('ushr', a, b), ('ishl', a, ('isub', 16, b))), 'options->lower_rotate'),
   (('uror@32', a, b), ('ior', ('ushr', a, b), ('ishl', a, ('isub', 32, b))), 'options->lower_rotate'),
   (('uror@64', a, b), ('ior', ('ushr', a, b), ('ishl', a, ('isub', 64, b))), 'options->lower_rotate'),
   # Exponential/logarithmic identities
   (('~fexp2', ('flog2', a)), a), # 2^lg2(a) = a
   (('~flog2', ('fexp2', a)), a), # lg2(2^a) = a
   (('fpow', a, b), ('fexp2', ('fmul', ('flog2', a), b)), 'options->lower_fpow'), # a^b = 2^(lg2(a)*b)
   (('~fexp2', ('fmul', ('flog2', a), b)), ('fpow', a, b), '!options->lower_fpow'), # 2^(lg2(a)*b) = a^b
   (('~fexp2', ('fadd', ('fmul', ('flog2', a), b), ('fmul', ('flog2', c), d))),
    ('~fmul', ('fpow', a, b), ('fpow', c, d)), '!options->lower_fpow'), # 2^(lg2(a) * b + lg2(c) + d) = a^b * c^d
   (('~fexp2', ('fmul', ('flog2', a), 0.5)), ('fsqrt', a)),
   (('~fexp2', ('fmul', ('flog2', a), 2.0)), ('fmul', a, a)),
   (('~fexp2', ('fmul', ('flog2', a), 4.0)), ('fmul', ('fmul', a, a), ('fmul', a, a))),
   (('~fpow', a, 1.0), a),
   (('~fpow', a, 2.0), ('fmul', a, a)),
   (('~fpow', a, 4.0), ('fmul', ('fmul', a, a), ('fmul', a, a))),
   (('~fpow', 2.0, a), ('fexp2', a)),
   (('~fpow', ('fpow', a, 2.2), 0.454545), a),
   (('~fpow', ('fabs', ('fpow', a, 2.2)), 0.454545), ('fabs', a)),
   (('~fsqrt', ('fexp2', a)), ('fexp2', ('fmul', 0.5, a))),
   (('~frcp', ('fexp2', a)), ('fexp2', ('fneg', a))),
   (('~frsq', ('fexp2', a)), ('fexp2', ('fmul', -0.5, a))),
   (('~flog2', ('fsqrt', a)), ('fmul', 0.5, ('flog2', a))),
   (('~flog2', ('frcp', a)), ('fneg', ('flog2', a))),
   (('~flog2', ('frsq', a)), ('fmul', -0.5, ('flog2', a))),
   (('~flog2', ('fpow', a, b)), ('fmul', b, ('flog2', a))),
   (('~fmul', ('fexp2(is_used_once)', a), ('fexp2(is_used_once)', b)), ('fexp2', ('fadd', a, b))),
   (('bcsel', ('flt', a, 0.0), 0.0, ('fsqrt', a)), ('fsqrt', ('fmax', a, 0.0))),
   (('~fmul', ('fsqrt', a), ('fsqrt', a)), ('fabs',a)),
   (('~fmulz', ('fsqrt', a), ('fsqrt', a)), ('fabs', a)),
   # Division and reciprocal
   (('~fdiv', 1.0, a), ('frcp', a)),
   (('fdiv', a, b), ('fmul', a, ('frcp', b)), 'options->lower_fdiv'),
   (('~frcp', ('frcp', a)), a),
   (('~frcp', ('fsqrt', a)), ('frsq', a)),
   (('fsqrt', a), ('frcp', ('frsq', a)), 'options->lower_fsqrt'),
   (('~frcp', ('frsq', a)), ('fsqrt', a), '!options->lower_fsqrt'),
   # Trig
   (('fsin', a), lowered_sincos(0.5), 'options->lower_sincos'),
   (('fcos', a), lowered_sincos(0.75), 'options->lower_sincos'),
   # Boolean simplifications
   (('ieq', a, True), a),
   (('ine(is_not_used_by_if)', a, True), ('inot', a)),
   (('ine', a, False), a),
   (('ieq(is_not_used_by_if)', a, False), ('inot', 'a')),
   (('bcsel', a, True, False), a),
   (('bcsel', a, False, True), ('inot', a)),
   (('bcsel', True, b, c), b),
   (('bcsel', False, b, c), c),

   (('bcsel@16', a, 1.0, 0.0), ('b2f', a)),
   (('bcsel@16', a, 0.0, 1.0), ('b2f', ('inot', a))),
   (('bcsel@16', a, -1.0, -0.0), ('fneg', ('b2f', a))),
   (('bcsel@16', a, -0.0, -1.0), ('fneg', ('b2f', ('inot', a)))),
   (('bcsel@32', a, 1.0, 0.0), ('b2f', a)),
   (('bcsel@32', a, 0.0, 1.0), ('b2f', ('inot', a))),
   (('bcsel@32', a, -1.0, -0.0), ('fneg', ('b2f', a))),
   (('bcsel@32', a, -0.0, -1.0), ('fneg', ('b2f', ('inot', a)))),
   (('bcsel@64', a, 1.0, 0.0), ('b2f', a), '!(options->lower_doubles_options & nir_lower_fp64_full_software)'),
   (('bcsel@64', a, 0.0, 1.0), ('b2f', ('inot', a)), '!(options->lower_doubles_options & nir_lower_fp64_full_software)'),
   (('bcsel@64', a, -1.0, -0.0), ('fneg', ('b2f', a)), '!(options->lower_doubles_options & nir_lower_fp64_full_software)'),
   (('bcsel@64', a, -0.0, -1.0), ('fneg', ('b2f', ('inot', a))), '!(options->lower_doubles_options & nir_lower_fp64_full_software)'),

   (('bcsel', a, b, b), b),
   (('~fcsel', a, b, b), b),

   # D3D Boolean emulation
   (('bcsel', a, -1, 0), ('ineg', ('b2i', 'a@1'))),
   (('bcsel', a, 0, -1), ('ineg', ('b2i', ('inot', a)))),
   (('bcsel', a, 1, 0), ('b2i', 'a@1')),
   (('bcsel', a, 0, 1), ('b2i', ('inot', a))),
   (('iand', ('ineg', ('b2i', 'a@1')), ('ineg', ('b2i', 'b@1'))),
    ('ineg', ('b2i', ('iand', a, b)))),
   (('ior', ('ineg', ('b2i','a@1')), ('ineg', ('b2i', 'b@1'))),
    ('ineg', ('b2i', ('ior', a, b)))),
   (('ieq', ('ineg', ('b2i', 'a@1')), -1), a),
   (('ine', ('ineg', ('b2i', 'a@1')), -1), ('inot', a)),
   (('ige', ('ineg', ('b2i', 'a@1')), 0), ('inot', a)),
   (('ilt', ('ineg', ('b2i', 'a@1')), 0), a),
   (('ult', 0, ('ineg', ('b2i', 'a@1'))), a),
   (('iand', ('ineg', ('b2i', a)), 1.0), ('b2f', a)),
   (('iand', ('ineg', ('b2i', a)), 1),   ('b2i', a)),

   # With D3D booleans, imax is AND and umax is OR
   (('imax', ('ineg', ('b2i', 'a@1')), ('ineg', ('b2i', 'b@1'))),
    ('ineg', ('b2i', ('iand', a, b)))),
   (('imin', ('ineg', ('b2i', 'a@1')), ('ineg', ('b2i', 'b@1'))),
    ('ineg', ('b2i', ('ior', a, b)))),
   (('umax', ('ineg', ('b2i', 'a@1')), ('ineg', ('b2i', 'b@1'))),
    ('ineg', ('b2i', ('ior', a, b)))),
   (('umin', ('ineg', ('b2i', 'a@1')), ('ineg', ('b2i', 'b@1'))),
    ('ineg', ('b2i', ('iand', a, b)))),
   (('umax', ('b2i', 'a@1'), ('b2i', 'b@1')), ('b2i', ('ior',  a, b))),
   (('umin', ('b2i', 'a@1'), ('b2i', 'b@1')), ('b2i', ('iand', a, b))),

   (('ine', ('umin', ('ineg', ('b2i', 'a@1')), b), 0), ('iand', a, ('ine', b, 0))),
   (('ine', ('umax', ('ineg', ('b2i', 'a@1')), b), 0), ('ior' , a, ('ine', b, 0))),

   # Conversions
   (('f2i', ('ftrunc', a)), ('f2i', a)),
   (('f2u', ('ftrunc', a)), ('f2u', a)),
   (('inot', ('f2b1', a)), ('feq', a, 0.0)),

   # Conversions from 16 bits to 32 bits and back can always be removed
   (('f2fmp', ('f2f32', 'a@16')), a),
   (('i2imp', ('i2i32', 'a@16')), a),
   (('i2imp', ('u2u32', 'a@16')), a),

   (('f2imp', ('f2f32', 'a@16')), ('f2i16', a)),
   (('f2ump', ('f2f32', 'a@16')), ('f2u16', a)),
   (('i2fmp', ('i2i32', 'a@16')), ('i2f16', a)),
   (('u2fmp', ('u2u32', 'a@16')), ('u2f16', a)),

   (('f2fmp', ('b2f32', 'a@1')), ('b2f16', a)),
   (('i2imp', ('b2i32', 'a@1')), ('b2i16', a)),
   (('i2imp', ('b2i32', 'a@1')), ('b2i16', a)),

   (('f2imp', ('b2f32', 'a@1')), ('b2i16', a)),
   (('f2ump', ('b2f32', 'a@1')), ('b2i16', a)),
   (('i2fmp', ('b2i32', 'a@1')), ('b2f16', a)),
   (('u2fmp', ('b2i32', 'a@1')), ('b2f16', a)),

   # Conversions to 16 bits would be lossy so they should only be removed if
   # the instruction was generated by the precision lowering pass.
   (('f2f32', ('f2fmp', 'a@32')), a),
   (('i2i32', ('i2imp', 'a@32')), a),
   (('u2u32', ('i2imp', 'a@32')), a),

   (('i2i32', ('f2imp', 'a@32')), ('f2i32', a)),
   (('u2u32', ('f2ump', 'a@32')), ('f2u32', a)),
   (('f2f32', ('i2fmp', 'a@32')), ('i2f32', a)),
   (('f2f32', ('u2fmp', 'a@32')), ('u2f32', a)),

   # Conversions from float32 to float64 and back can be removed as long as
   # it doesn't need to be precise, since the conversion may e.g. flush denorms
   (('~f2f32', ('f2f64', 'a@32')), a),

   (('ffloor', 'a(is_integral)'), a),
   (('fceil', 'a(is_integral)'), a),
   (('ftrunc', 'a(is_integral)'), a),
   (('fround_even', 'a(is_integral)'), a),

   # fract(x) = x - floor(x), so fract(NaN) = NaN
   (('~ffract', 'a(is_integral)'), 0.0),
   (('fabs', 'a(is_not_negative)'), a),
   (('iabs', 'a(is_not_negative)'), a),
   (('fsat', 'a(is_not_positive)'), 0.0),

   (('~fmin', 'a(is_not_negative)', 1.0), ('fsat', a), '!options->lower_fsat'),

   # The result of the multiply must be in [-1, 0], so the result of the ffma
   # must be in [0, 1].
   (('flt', ('fadd', ('fmul', ('fsat', a), ('fneg', ('fsat', a))), 1.0), 0.0), False),
   (('flt', ('fadd', ('fneg', ('fmul', ('fsat', a), ('fsat', a))), 1.0), 0.0), False),
   (('fmax', ('fadd', ('fmul', ('fsat', a), ('fneg', ('fsat', a))), 1.0), 0.0), ('fadd', ('fmul', ('fsat', a), ('fneg', ('fsat', a))), 1.0)),
   (('fmax', ('fadd', ('fneg', ('fmul', ('fsat', a), ('fsat', a))), 1.0), 0.0), ('fadd', ('fneg', ('fmul', ('fsat', a), ('fsat', a))), 1.0)),

   (('fneu', 'a(is_not_zero)', 0.0), True),
   (('feq', 'a(is_not_zero)', 0.0), False),

   # In this chart, + means value > 0 and - means value < 0.
   #
   # + >= + -> unknown  0 >= + -> false    - >= + -> false
   # + >= 0 -> true     0 >= 0 -> true     - >= 0 -> false
   # + >= - -> true     0 >= - -> true     - >= - -> unknown
   #
   # Using grouping conceptually similar to a Karnaugh map...
   #
   # (+ >= 0, + >= -, 0 >= 0, 0 >= -) == (is_not_negative >= is_not_positive) -> true
   # (0 >= +, - >= +) == (is_not_positive >= gt_zero) -> false
   # (- >= +, - >= 0) == (lt_zero >= is_not_negative) -> false
   #
   # The flt / ilt cases just invert the expected result.
   #
   # The results expecting true, must be marked imprecise.  The results
   # expecting false are fine because NaN compared >= or < anything is false.

   (('fge', 'a(is_a_number_not_negative)', 'b(is_a_number_not_positive)'), True),
   (('fge', 'a(is_not_positive)',          'b(is_gt_zero)'),               False),
   (('fge', 'a(is_lt_zero)',               'b(is_not_negative)'),          False),

   (('flt', 'a(is_not_negative)',          'b(is_not_positive)'),          False),
   (('flt', 'a(is_a_number_not_positive)', 'b(is_a_number_gt_zero)'),      True),
   (('flt', 'a(is_a_number_lt_zero)',      'b(is_a_number_not_negative)'), True),

   (('ine', 'a(is_not_zero)', 0), True),
   (('ieq', 'a(is_not_zero)', 0), False),

   (('ige', 'a(is_not_negative)', 'b(is_not_positive)'), True),
   (('ige', 'a(is_not_positive)', 'b(is_gt_zero)'),      False),
   (('ige', 'a(is_lt_zero)',      'b(is_not_negative)'), False),

   (('ilt', 'a(is_not_negative)', 'b(is_not_positive)'), False),
   (('ilt', 'a(is_not_positive)', 'b(is_gt_zero)'),      True),
   (('ilt', 'a(is_lt_zero)',      'b(is_not_negative)'), True),

   (('ult', 0, 'a(is_gt_zero)'), True),
   (('ult', a, 0), False),

   # Packing and then unpacking does nothing
   (('unpack_64_2x32_split_x', ('pack_64_2x32_split', a, b)), a),
   (('unpack_64_2x32_split_y', ('pack_64_2x32_split', a, b)), b),
   (('unpack_64_2x32_split_x', ('pack_64_2x32', a)), 'a.x'),
   (('unpack_64_2x32_split_y', ('pack_64_2x32', a)), 'a.y'),
   (('unpack_64_2x32_split_x', ('u2u64', 'a@32')), a),
   (('unpack_64_2x32_split_y', ('u2u64', a)), 0),
   (('unpack_64_2x32_split_x', ('i2i64', 'a@32')), a),
   (('unpack_64_2x32_split_y', ('i2i64(is_used_once)', 'a@32')), ('ishr', a, 31)),
   (('unpack_64_2x32', ('pack_64_2x32_split', a, b)), ('vec2', a, b)),
   (('unpack_64_2x32', ('pack_64_2x32', a)), a),
   (('unpack_double_2x32_dxil', ('pack_double_2x32_dxil', a)), a),
   (('pack_64_2x32_split', ('unpack_64_2x32_split_x', a),
                           ('unpack_64_2x32_split_y', a)), a),
   (('pack_64_2x32', ('vec2', ('unpack_64_2x32_split_x', a),
                              ('unpack_64_2x32_split_y', a))), a),
   (('pack_64_2x32', ('unpack_64_2x32', a)), a),
   (('pack_double_2x32_dxil', ('unpack_double_2x32_dxil', a)), a),

   # Comparing two halves of an unpack separately.  While this optimization
   # should be correct for non-constant values, it's less obvious that it's
   # useful in that case.  For constant values, the pack will fold and we're
   # guaranteed to reduce the whole tree to one instruction.
   (('iand', ('ieq', ('unpack_32_2x16_split_x', a), '#b'),
             ('ieq', ('unpack_32_2x16_split_y', a), '#c')),
    ('ieq', a, ('pack_32_2x16_split', b, c))),

   # Byte extraction
   (('ushr', 'a@16',  8), ('extract_u8', a, 1), '!options->lower_extract_byte'),
   (('ushr', 'a@32', 24), ('extract_u8', a, 3), '!options->lower_extract_byte'),
   (('ushr', 'a@64', 56), ('extract_u8', a, 7), '!options->lower_extract_byte'),
   (('ishr', 'a@16',  8), ('extract_i8', a, 1), '!options->lower_extract_byte'),
   (('ishr', 'a@32', 24), ('extract_i8', a, 3), '!options->lower_extract_byte'),
   (('ishr', 'a@64', 56), ('extract_i8', a, 7), '!options->lower_extract_byte'),
   (('iand', 0xff, a), ('extract_u8', a, 0), '!options->lower_extract_byte'),

   # Common pattern in many Vulkan CTS tests that read 8-bit integers from a
   # storage buffer.
   (('u2u8', ('extract_u16', a, 1)), ('u2u8', ('extract_u8', a, 2)), '!options->lower_extract_byte'),
   (('u2u8', ('ushr', a, 8)), ('u2u8', ('extract_u8', a, 1)), '!options->lower_extract_byte'),

   # Common pattern after lowering 8-bit integers to 16-bit.
   (('i2i16', ('u2u8', ('extract_u8', a, b))), ('i2i16', ('extract_i8', a, b))),
   (('u2u16', ('u2u8', ('extract_u8', a, b))), ('u2u16', ('extract_u8', a, b))),

   (('ubfe', a,  0, 8), ('extract_u8', a, 0), '!options->lower_extract_byte'),
   (('ubfe', a,  8, 8), ('extract_u8', a, 1), '!options->lower_extract_byte'),
   (('ubfe', a, 16, 8), ('extract_u8', a, 2), '!options->lower_extract_byte'),
   (('ubfe', a, 24, 8), ('extract_u8', a, 3), '!options->lower_extract_byte'),
   (('ibfe', a,  0, 8), ('extract_i8', a, 0), '!options->lower_extract_byte'),
   (('ibfe', a,  8, 8), ('extract_i8', a, 1), '!options->lower_extract_byte'),
   (('ibfe', a, 16, 8), ('extract_i8', a, 2), '!options->lower_extract_byte'),
   (('ibfe', a, 24, 8), ('extract_i8', a, 3), '!options->lower_extract_byte'),

   (('extract_u8', ('extract_i8', a, b), 0), ('extract_u8', a, b)),
   (('extract_u8', ('extract_u8', a, b), 0), ('extract_u8', a, b)),

    # Word extraction
   (('ushr', ('ishl', 'a@32', 16), 16), ('extract_u16', a, 0), '!options->lower_extract_word'),
   (('ushr', 'a@32', 16), ('extract_u16', a, 1), '!options->lower_extract_word'),
   (('ishr', ('ishl', 'a@32', 16), 16), ('extract_i16', a, 0), '!options->lower_extract_word'),
   (('ishr', 'a@32', 16), ('extract_i16', a, 1), '!options->lower_extract_word'),
   (('iand', 0xffff, a), ('extract_u16', a, 0), '!options->lower_extract_word'),

   (('ubfe', a,  0, 16), ('extract_u16', a, 0), '!options->lower_extract_word'),
   (('ubfe', a, 16, 16), ('extract_u16', a, 1), '!options->lower_extract_word'),
   (('ibfe', a,  0, 16), ('extract_i16', a, 0), '!options->lower_extract_word'),
   (('ibfe', a, 16, 16), ('extract_i16', a, 1), '!options->lower_extract_word'),

   # Packing a u8vec4 to write to an SSBO.
   (('ior', ('ishl', ('u2u32', 'a@8'), 24), ('ior', ('ishl', ('u2u32', 'b@8'), 16), ('ior', ('ishl', ('u2u32', 'c@8'), 8), ('u2u32', 'd@8')))),
    ('pack_32_4x8', ('vec4', d, c, b, a)), 'options->has_pack_32_4x8'),

   (('extract_u16', ('extract_i16', a, b), 0), ('extract_u16', a, b)),
   (('extract_u16', ('extract_u16', a, b), 0), ('extract_u16', a, b)),

   # Lower pack/unpack
   (('pack_64_2x32_split', a, b), ('ior', ('u2u64', a), ('ishl', ('u2u64', b), 32)), 'options->lower_pack_64_2x32_split'),
   (('pack_32_2x16_split', a, b), ('ior', ('u2u32', a), ('ishl', ('u2u32', b), 16)), 'options->lower_pack_32_2x16_split'),
   (('unpack_64_2x32_split_x', a), ('u2u32', a), 'options->lower_unpack_64_2x32_split'),
   (('unpack_64_2x32_split_y', a), ('u2u32', ('ushr', a, 32)), 'options->lower_unpack_64_2x32_split'),
   (('unpack_32_2x16_split_x', a), ('u2u16', a), 'options->lower_unpack_32_2x16_split'),
   (('unpack_32_2x16_split_y', a), ('u2u16', ('ushr', a, 16)), 'options->lower_unpack_32_2x16_split'),

   # Useless masking before unpacking
   (('unpack_half_2x16_split_x', ('iand', a, 0xffff)), ('unpack_half_2x16_split_x', a)),
   (('unpack_32_2x16_split_x', ('iand', a, 0xffff)), ('unpack_32_2x16_split_x', a)),
   (('unpack_64_2x32_split_x', ('iand', a, 0xffffffff)), ('unpack_64_2x32_split_x', a)),
   (('unpack_half_2x16_split_y', ('iand', a, 0xffff0000)), ('unpack_half_2x16_split_y', a)),
   (('unpack_32_2x16_split_y', ('iand', a, 0xffff0000)), ('unpack_32_2x16_split_y', a)),
   (('unpack_64_2x32_split_y', ('iand', a, 0xffffffff00000000)), ('unpack_64_2x32_split_y', a)),

   (('unpack_half_2x16_split_x', ('extract_u16', a, 0)), ('unpack_half_2x16_split_x', a)),
   (('unpack_half_2x16_split_x', ('extract_u16', a, 1)), ('unpack_half_2x16_split_y', a)),
   (('unpack_half_2x16_split_x', ('ushr', a, 16)), ('unpack_half_2x16_split_y', a)),
   (('unpack_32_2x16_split_x', ('extract_u16', a, 0)), ('unpack_32_2x16_split_x', a)),
   (('unpack_32_2x16_split_x', ('extract_u16', a, 1)), ('unpack_32_2x16_split_y', a)),

   # Optimize half packing
   (('ishl', ('pack_half_2x16', ('vec2', a, 0)), 16), ('pack_half_2x16', ('vec2', 0, a))),
   (('ushr', ('pack_half_2x16', ('vec2', 0, a)), 16), ('pack_half_2x16', ('vec2', a, 0))),

   (('iadd', ('pack_half_2x16', ('vec2', a, 0)), ('pack_half_2x16', ('vec2', 0, b))),
    ('pack_half_2x16', ('vec2', a, b))),
   (('ior', ('pack_half_2x16', ('vec2', a, 0)), ('pack_half_2x16', ('vec2', 0, b))),
    ('pack_half_2x16', ('vec2', a, b))),

   (('ishl', ('pack_half_2x16_split', a, 0), 16), ('pack_half_2x16_split', 0, a)),
   (('ushr', ('pack_half_2x16_split', 0, a), 16), ('pack_half_2x16_split', a, 0)),
   (('extract_u16', ('pack_half_2x16_split', 0, a), 1), ('pack_half_2x16_split', a, 0)),

   (('iadd', ('pack_half_2x16_split', a, 0), ('pack_half_2x16_split', 0, b)), ('pack_half_2x16_split', a, b)),
   (('ior',  ('pack_half_2x16_split', a, 0), ('pack_half_2x16_split', 0, b)), ('pack_half_2x16_split', a, b)),

   (('extract_i8', ('pack_32_4x8_split', a, b, c, d), 0), ('i2i', a)),
   (('extract_i8', ('pack_32_4x8_split', a, b, c, d), 1), ('i2i', b)),
   (('extract_i8', ('pack_32_4x8_split', a, b, c, d), 2), ('i2i', c)),
   (('extract_i8', ('pack_32_4x8_split', a, b, c, d), 3), ('i2i', d)),
   (('extract_u8', ('pack_32_4x8_split', a, b, c, d), 0), ('u2u', a)),
   (('extract_u8', ('pack_32_4x8_split', a, b, c, d), 1), ('u2u', b)),
   (('extract_u8', ('pack_32_4x8_split', a, b, c, d), 2), ('u2u', c)),
   (('extract_u8', ('pack_32_4x8_split', a, b, c, d), 3), ('u2u', d)),
])

# After the ('extract_u8', a, 0) pattern, above, triggers, there will be
# patterns like those below.
for op in ('ushr', 'ishr'):
   optimizations.extend([(('extract_u8', (op, 'a@16',  8),     0), ('extract_u8', a, 1))])
   optimizations.extend([(('extract_u8', (op, 'a@32',  8 * i), 0), ('extract_u8', a, i)) for i in range(1, 4)])
   optimizations.extend([(('extract_u8', (op, 'a@64',  8 * i), 0), ('extract_u8', a, i)) for i in range(1, 8)])

optimizations.extend([(('extract_u8', ('extract_u16', a, 1), 0), ('extract_u8', a, 2))])

# After the ('extract_[iu]8', a, 3) patterns, above, trigger, there will be
# patterns like those below.
for op in ('extract_u8', 'extract_i8'):
   optimizations.extend([((op, ('ishl', 'a@16',      8),     1), (op, a, 0))])
   optimizations.extend([((op, ('ishl', 'a@32', 24 - 8 * i), 3), (op, a, i)) for i in range(2, -1, -1)])
   optimizations.extend([((op, ('ishl', 'a@64', 56 - 8 * i), 7), (op, a, i)) for i in range(6, -1, -1)])

optimizations.extend([
   # Subtracts
   (('ussub_4x8_vc4', a, 0), a),
   (('ussub_4x8_vc4', a, ~0), 0),
   # Lower all Subtractions first - they can get recombined later
   (('fsub', a, b), ('fadd', a, ('fneg', b))),
   (('isub', a, b), ('iadd', a, ('ineg', b))),
   (('uabs_usub', a, b), ('bcsel', ('ult', a, b), ('ineg', ('isub', a, b)), ('isub', a, b))),
   # This is correct.  We don't need isub_sat because the result type is unsigned, so it cannot overflow.
   (('uabs_isub', a, b), ('bcsel', ('ilt', a, b), ('ineg', ('isub', a, b)), ('isub', a, b))),

   # Propagate negation up multiplication chains
   (('fmul(is_used_by_non_fsat)', ('fneg', a), b), ('fneg', ('fmul', a, b))),
   (('fmulz(is_used_by_non_fsat)', ('fneg', a), b), ('fneg', ('fmulz', a, b)), '!'+signed_zero_inf_nan_preserve_32),
   (('ffma', ('fneg', a), ('fneg', b), c), ('ffma', a, b, c)),
   (('ffmaz', ('fneg', a), ('fneg', b), c), ('ffmaz', a, b, c)),
   (('imul', ('ineg', a), b), ('ineg', ('imul', a, b))),

   # Propagate constants up multiplication chains
   (('~fmul(is_used_once)', ('fmul(is_used_once)', 'a(is_not_const)', 'b(is_not_const)'), '#c'), ('fmul', ('fmul', a, c), b)),
   (('~fmulz(is_used_once)', ('fmulz(is_used_once)', 'a(is_not_const)', 'b(is_not_const)'), '#c'), ('fmulz', ('fmulz', a, c), b)),
   (('~fmul(is_used_once)', ('fmulz(is_used_once)', 'a(is_not_const)', 'b(is_not_const)'), '#c(is_finite_not_zero)'), ('fmulz', ('fmul', a, c), b)),
   (('imul(is_used_once)', ('imul(is_used_once)', 'a(is_not_const)', 'b(is_not_const)'), '#c'), ('imul', ('imul', a, c), b)),
   (('~ffma', ('fmul(is_used_once)', 'a(is_not_const)', 'b(is_not_const)'), '#c', d), ('ffma', ('fmul', a, c), b, d)),
   (('~ffmaz', ('fmulz(is_used_once)', 'a(is_not_const)', 'b(is_not_const)'), '#c', d), ('ffmaz', ('fmulz', a, c), b, d)),
   (('~ffma', ('fmulz(is_used_once)', 'a(is_not_const)', 'b(is_not_const)'), '#c(is_finite_not_zero)', d), ('ffmaz', ('fmul', a, c), b, d)),
   # Prefer moving out a multiplication for more MAD/FMA-friendly code
   (('~fadd(is_used_once)', ('fadd(is_used_once)', 'a(is_not_const)', 'b(is_fmul)'), '#c'), ('fadd', ('fadd', a, c), b)),
   (('~fadd(is_used_once)', ('fadd(is_used_once)', 'a(is_not_const)', 'b(is_not_const)'), '#c'), ('fadd', ('fadd', a, c), b)),
   (('~fadd(is_used_once)', ('ffma(is_used_once)', 'a(is_not_const)', b, 'c(is_not_const)'), '#d'), ('fadd', ('ffma', a, b, d), c)),
   (('~fadd(is_used_once)', ('ffmaz(is_used_once)', 'a(is_not_const)', b, 'c(is_not_const)'), '#d'), ('fadd', ('ffmaz', a, b, d), c)),
   (('iadd(is_used_once)', ('iadd(is_used_once)', 'a(is_not_const)', 'b(is_not_const)'), '#c'), ('iadd', ('iadd', a, c), b)),

   # Reassociate constants in add/mul chains so they can be folded together.
   # For now, we mostly only handle cases where the constants are separated by
   # a single non-constant.  We could do better eventually.
   (('~fmul', '#a', ('fmul', 'b(is_not_const)', '#c')), ('fmul', ('fmul', a, c), b)),
   (('~fmulz', '#a', ('fmulz', 'b(is_not_const)', '#c')), ('fmulz', ('fmulz', a, c), b)),
   (('~fmul', '#a(is_finite_not_zero)', ('fmulz', 'b(is_not_const)', '#c')), ('fmulz', ('fmul', a, c), b)),
   (('~ffma', '#a', ('fmul', 'b(is_not_const)', '#c'), d), ('ffma', ('fmul', a, c), b, d)),
   (('~ffmaz', '#a', ('fmulz', 'b(is_not_const)', '#c'), d), ('ffmaz', ('fmulz', a, c), b, d)),
   (('~ffmaz', '#a(is_finite_not_zero)', ('fmulz', 'b(is_not_const)', '#c'), d), ('ffmaz', ('fmul', a, c), b, d)),
   (('imul', '#a', ('imul', 'b(is_not_const)', '#c')), ('imul', ('imul', a, c), b)),
   (('~fadd', '#a',          ('fadd', 'b(is_not_const)', '#c')),  ('fadd', ('fadd', a,          c),           b)),
   (('~fadd', '#a', ('fneg', ('fadd', 'b(is_not_const)', '#c'))), ('fadd', ('fadd', a, ('fneg', c)), ('fneg', b))),
   (('~fadd', '#a',          ('ffma', 'b(is_not_const)', 'c(is_not_const)', '#d')),  ('ffma',          b,  c, ('fadd', a,          d))),
   (('~fadd', '#a', ('fneg', ('ffma', 'b(is_not_const)', 'c(is_not_const)', '#d'))), ('ffma', ('fneg', b), c, ('fadd', a, ('fneg', d)))),
   (('~fadd', '#a',          ('ffmaz', 'b(is_not_const)', 'c(is_not_const)', '#d')),  ('ffmaz',          b,  c, ('fadd', a,          d))),
   (('~fadd', '#a', ('fneg', ('ffmaz', 'b(is_not_const)', 'c(is_not_const)', '#d'))), ('ffmaz', ('fneg', b), c, ('fadd', a, ('fneg', d)))),
   (('iadd', '#a', ('iadd', 'b(is_not_const)', '#c')), ('iadd', ('iadd', a, c), b)),
   (('iand', '#a', ('iand', 'b(is_not_const)', '#c')), ('iand', ('iand', a, c), b)),
   (('ior',  '#a', ('ior',  'b(is_not_const)', '#c')), ('ior',  ('ior',  a, c), b)),
   (('ixor', '#a', ('ixor', 'b(is_not_const)', '#c')), ('ixor', ('ixor', a, c), b)),

   # Reassociate add chains for more MAD/FMA-friendly code
   (('~fadd', ('fadd(is_used_once)', 'a(is_fmul)', 'b(is_fmul)'), 'c(is_not_fmul)'), ('fadd', ('fadd', a, c), b)),

   # Drop mul-div by the same value when there's no wrapping.
   (('idiv', ('imul(no_signed_wrap)', a, b), b), a),

   # By definition...
   (('bcsel', ('ige', ('find_lsb', a), 0), ('find_lsb', a), -1), ('find_lsb', a)),
   (('bcsel', ('ige', ('ifind_msb', a), 0), ('ifind_msb', a), -1), ('ifind_msb', a)),
   (('bcsel', ('ige', ('ufind_msb', a), 0), ('ufind_msb', a), -1), ('ufind_msb', a)),
   (('bcsel', ('ige', ('ifind_msb_rev', a), 0), ('ifind_msb_rev', a), -1), ('ifind_msb_rev', a)),
   (('bcsel', ('ige', ('ufind_msb_rev', a), 0), ('ufind_msb_rev', a), -1), ('ufind_msb_rev', a)),

   (('bcsel', ('ine', a, 0), ('find_lsb', a), -1), ('find_lsb', a)),
   (('bcsel', ('ine', a, 0), ('ifind_msb', a), -1), ('ifind_msb', a)),
   (('bcsel', ('ine', a, 0), ('ufind_msb', a), -1), ('ufind_msb', a)),
   (('bcsel', ('ine', a, 0), ('ifind_msb_rev', a), -1), ('ifind_msb_rev', a)),
   (('bcsel', ('ine', a, 0), ('ufind_msb_rev', a), -1), ('ufind_msb_rev', a)),

   (('bcsel', ('ine', a, -1), ('ifind_msb', a), -1), ('ifind_msb', a)),
   (('bcsel', ('ine', a, -1), ('ifind_msb_rev', a), -1), ('ifind_msb_rev', a)),

   (('bcsel', ('ine', ('ifind_msb', 'a@32'), -1), ('iadd', 31, ('ineg', ('ifind_msb', a))), -1), ('ifind_msb_rev', a), 'options->has_find_msb_rev'),
   (('bcsel', ('ine', ('ufind_msb', 'a@32'), -1), ('iadd', 31, ('ineg', ('ufind_msb', a))), -1), ('ufind_msb_rev', a), 'options->has_find_msb_rev'),
   (('bcsel', ('ieq', ('ifind_msb', 'a@32'), -1), -1, ('iadd', 31, ('ineg', ('ifind_msb', a)))), ('ifind_msb_rev', a), 'options->has_find_msb_rev'),
   (('bcsel', ('ieq', ('ufind_msb', 'a@32'), -1), -1, ('iadd', 31, ('ineg', ('ufind_msb', a)))), ('ufind_msb_rev', a), 'options->has_find_msb_rev'),
   (('bcsel', ('ine', ('ifind_msb', 'a@32'), -1), ('iadd', 31, ('ineg', ('ifind_msb', a))), ('ifind_msb', a)), ('ifind_msb_rev', a), 'options->has_find_msb_rev'),
   (('bcsel', ('ine', ('ufind_msb', 'a@32'), -1), ('iadd', 31, ('ineg', ('ufind_msb', a))), ('ufind_msb', a)), ('ufind_msb_rev', a), 'options->has_find_msb_rev'),
   (('bcsel', ('ieq', ('ifind_msb', 'a@32'), -1), ('ifind_msb', a), ('iadd', 31, ('ineg', ('ifind_msb', a)))), ('ifind_msb_rev', a), 'options->has_find_msb_rev'),
   (('bcsel', ('ieq', ('ufind_msb', 'a@32'), -1), ('ufind_msb', a), ('iadd', 31, ('ineg', ('ufind_msb', a)))), ('ufind_msb_rev', a), 'options->has_find_msb_rev'),
   (('bcsel', ('ine', 'a@32', 0), ('iadd', 31, ('ineg', ('ufind_msb', a))), -1), ('ufind_msb_rev', a), 'options->has_find_msb_rev'),
   (('bcsel', ('ieq', 'a@32', 0), -1, ('iadd', 31, ('ineg', ('ufind_msb', a)))), ('ufind_msb_rev', a), 'options->has_find_msb_rev'),
   (('bcsel', ('ine', 'a@32', 0), ('iadd', 31, ('ineg', ('ufind_msb', a))), ('ufind_msb', a)), ('ufind_msb_rev', a), 'options->has_find_msb_rev'),
   (('bcsel', ('ieq', 'a@32', 0), ('ufind_msb', a), ('iadd', 31, ('ineg', ('ufind_msb', a)))), ('ufind_msb_rev', a), 'options->has_find_msb_rev'),

   (('bcsel', ('ine', ('ifind_msb_rev', 'a@32'), -1), ('iadd', 31, ('ineg', ('ifind_msb_rev', a))), -1), ('ifind_msb', a), '!options->lower_find_msb_to_reverse'),
   (('bcsel', ('ine', ('ufind_msb_rev', 'a@32'), -1), ('iadd', 31, ('ineg', ('ufind_msb_rev', a))), -1), ('ufind_msb', a), '!options->lower_find_msb_to_reverse'),
   (('bcsel', ('ieq', ('ifind_msb_rev', 'a@32'), -1), -1, ('iadd', 31, ('ineg', ('ifind_msb_rev', a)))), ('ifind_msb', a), '!options->lower_find_msb_to_reverse'),
   (('bcsel', ('ieq', ('ufind_msb_rev', 'a@32'), -1), -1, ('iadd', 31, ('ineg', ('ufind_msb_rev', a)))), ('ufind_msb', a), '!options->lower_find_msb_to_reverse'),
   (('bcsel', ('ine', ('ifind_msb_rev', 'a@32'), -1), ('iadd', 31, ('ineg', ('ifind_msb_rev', a))), ('ifind_msb_rev', a)), ('ifind_msb', a), '!options->lower_find_msb_to_reverse'),
   (('bcsel', ('ine', ('ufind_msb_rev', 'a@32'), -1), ('iadd', 31, ('ineg', ('ufind_msb_rev', a))), ('ufind_msb_rev', a)), ('ufind_msb', a), '!options->lower_find_msb_to_reverse'),
   (('bcsel', ('ieq', ('ifind_msb_rev', 'a@32'), -1), ('ifind_msb_rev', a), ('iadd', 31, ('ineg', ('ifind_msb_rev', a)))), ('ifind_msb', a), '!options->lower_find_msb_to_reverse'),
   (('bcsel', ('ieq', ('ufind_msb_rev', 'a@32'), -1), ('ufind_msb_rev', a), ('iadd', 31, ('ineg', ('ufind_msb_rev', a)))), ('ufind_msb', a), '!options->lower_find_msb_to_reverse'),
   (('bcsel', ('ine', 'a@32', 0), ('iadd', 31, ('ineg', ('ufind_msb_rev', a))), -1), ('ufind_msb', a), '!options->lower_find_msb_to_reverse'),
   (('bcsel', ('ieq', 'a@32', 0), -1, ('iadd', 31, ('ineg', ('ufind_msb_rev', a)))), ('ufind_msb', a), '!options->lower_find_msb_to_reverse'),
   (('bcsel', ('ine', 'a@32', 0), ('iadd', 31, ('ineg', ('ufind_msb_rev', a))), ('ufind_msb_rev', a)), ('ufind_msb', a), '!options->lower_find_msb_to_reverse'),
   (('bcsel', ('ieq', 'a@32', 0), ('ufind_msb_rev', a), ('iadd', 31, ('ineg', ('ufind_msb_rev', a)))), ('ufind_msb', a), '!options->lower_find_msb_to_reverse'),

   (('find_lsb', ('bitfield_reverse', a)), ('ufind_msb_rev', a), 'options->has_find_msb_rev'),
   (('ufind_msb_rev', ('bitfield_reverse', a)), ('find_lsb', a), '!options->lower_find_lsb'),

   (('~fmul', ('bcsel(is_used_once)', c, -1.0, 1.0), b), ('bcsel', c, ('fneg', b), b)),
   (('~fmul', ('bcsel(is_used_once)', c, 1.0, -1.0), b), ('bcsel', c, b, ('fneg', b))),
   (('~fmulz', ('bcsel(is_used_once)', c, -1.0, 1.0), b), ('bcsel', c, ('fneg', b), b)),
   (('~fmulz', ('bcsel(is_used_once)', c, 1.0, -1.0), b), ('bcsel', c, b, ('fneg', b))),
   (('fabs', ('bcsel(is_used_once)', b, ('fneg', a), a)), ('fabs', a)),
   (('fabs', ('bcsel(is_used_once)', b, a, ('fneg', a))), ('fabs', a)),
   (('~bcsel', ('flt', a, 0.0), ('fneg', a), a), ('fabs', a)),

   (('bcsel', a, ('bcsel', b, c, d), d), ('bcsel', ('iand', a, b), c, d)),
   (('bcsel', a, b, ('bcsel', c, b, d)), ('bcsel', ('ior', a, c), b, d)),

   # Misc. lowering
   (('fmod', a, b), ('fsub', a, ('fmul', b, ('ffloor', ('fdiv', a, b)))), 'options->lower_fmod'),
   (('frem', a, b), ('fsub', a, ('fmul', b, ('ftrunc', ('fdiv', a, b)))), 'options->lower_fmod'),
   (('uadd_carry', a, b), ('b2i', ('ult', ('iadd', a, b), a)), 'options->lower_uadd_carry'),
   (('usub_borrow', a, b), ('b2i', ('ult', a, b)), 'options->lower_usub_borrow'),

   (('bitfield_insert', 'base', 'insert', 'offset', 'bits'),
    ('bcsel', ('ult', 31, 'bits'), 'insert',
              ('bfi', ('bfm', 'bits', 'offset'), 'insert', 'base')),
    'options->lower_bitfield_insert'),
   (('ihadd', a, b), ('iadd', ('iand', a, b), ('ishr', ('ixor', a, b), 1)), 'options->lower_hadd'),
   (('uhadd', a, b), ('iadd', ('iand', a, b), ('ushr', ('ixor', a, b), 1)), 'options->lower_hadd'),
   (('irhadd', a, b), ('isub', ('ior', a, b), ('ishr', ('ixor', a, b), 1)), 'options->lower_hadd'),
   (('urhadd', a, b), ('isub', ('ior', a, b), ('ushr', ('ixor', a, b), 1)), 'options->lower_hadd'),
   (('ihadd@64', a, b), ('iadd', ('iand', a, b), ('ishr', ('ixor', a, b), 1)), 'options->lower_hadd64 || (options->lower_int64_options & nir_lower_iadd64) != 0'),
   (('uhadd@64', a, b), ('iadd', ('iand', a, b), ('ushr', ('ixor', a, b), 1)), 'options->lower_hadd64 || (options->lower_int64_options & nir_lower_iadd64) != 0'),
   (('irhadd@64', a, b), ('isub', ('ior', a, b), ('ishr', ('ixor', a, b), 1)), 'options->lower_hadd64 || (options->lower_int64_options & nir_lower_iadd64) != 0'),
   (('urhadd@64', a, b), ('isub', ('ior', a, b), ('ushr', ('ixor', a, b), 1)), 'options->lower_hadd64 || (options->lower_int64_options & nir_lower_iadd64) != 0'),

   (('imul_32x16', a, b), ('imul', a, ('extract_i16', b, 0)), 'options->lower_mul_32x16'),
   (('umul_32x16', a, b), ('imul', a, ('extract_u16', b, 0)), 'options->lower_mul_32x16'),

   (('uadd_sat@64', a, b), ('bcsel', ('ult', ('iadd', a, b), a), -1, ('iadd', a, b)), 'options->lower_uadd_sat || (options->lower_int64_options & nir_lower_iadd64) != 0'),
   (('uadd_sat', a, b), ('bcsel', ('ult', ('iadd', a, b), a), -1, ('iadd', a, b)), 'options->lower_uadd_sat'),
   (('usub_sat', a, b), ('bcsel', ('ult', a, b), 0, ('isub', a, b)), 'options->lower_usub_sat'),
   (('usub_sat@64', a, b), ('bcsel', ('ult', a, b), 0, ('isub', a, b)), '(options->lower_int64_options & nir_lower_usub_sat64) != 0'),

   # int64_t sum = a + b;
   #
   # if (a < 0 && b < 0 && a < sum)
   #    sum = INT64_MIN;
   # } else if (a >= 0 && b >= 0 && sum < a)
   #    sum = INT64_MAX;
   # }
   #
   # A couple optimizations are applied.
   #
   # 1. a < sum => sum >= 0.  This replacement works because it is known that
   #    a < 0 and b < 0, so sum should also be < 0 unless there was
   #    underflow.
   #
   # 2. sum < a => sum < 0.  This replacement works because it is known that
   #    a >= 0 and b >= 0, so sum should also be >= 0 unless there was
   #    overflow.
   #
   # 3. Invert the second if-condition and swap the order of parameters for
   #    the bcsel. !(a >= 0 && b >= 0 && sum < 0) becomes !(a >= 0) || !(b >=
   #    0) || !(sum < 0), and that becomes (a < 0) || (b < 0) || (sum >= 0)
   #
   # On Intel Gen11, this saves ~11 instructions.
   (('iadd_sat@64', a, b), ('bcsel',
                            ('iand', ('iand', ('ilt', a, 0), ('ilt', b, 0)), ('ige', ('iadd', a, b), 0)),
                            0x8000000000000000,
                            ('bcsel',
                             ('ior', ('ior', ('ilt', a, 0), ('ilt', b, 0)), ('ige', ('iadd', a, b), 0)),
                             ('iadd', a, b),
                             0x7fffffffffffffff)),
    '(options->lower_int64_options & nir_lower_iadd_sat64) != 0'),

   # int64_t sum = a - b;
   #
   # if (a < 0 && b >= 0 && a < sum)
   #    sum = INT64_MIN;
   # } else if (a >= 0 && b < 0 && a >= sum)
   #    sum = INT64_MAX;
   # }
   #
   # Optimizations similar to the iadd_sat case are applied here.
   (('isub_sat@64', a, b), ('bcsel',
                            ('iand', ('iand', ('ilt', a, 0), ('ige', b, 0)), ('ige', ('isub', a, b), 0)),
                            0x8000000000000000,
                            ('bcsel',
                             ('ior', ('ior', ('ilt', a, 0), ('ige', b, 0)), ('ige', ('isub', a, b), 0)),
                             ('isub', a, b),
                             0x7fffffffffffffff)),
    '(options->lower_int64_options & nir_lower_iadd_sat64) != 0'),

   # These are done here instead of in the backend because the int64 lowering
   # pass will make a mess of the patterns.  The first patterns are
   # conditioned on nir_lower_minmax64 because it was not clear that it was
   # always an improvement on platforms that have real int64 support.  No
   # shaders in shader-db hit this, so it was hard to say one way or the
   # other.
   (('ilt', ('imax(is_used_once)', 'a@64', 'b@64'), 0), ('ilt', ('imax', ('unpack_64_2x32_split_y', a), ('unpack_64_2x32_split_y', b)), 0), '(options->lower_int64_options & nir_lower_minmax64) != 0'),
   (('ilt', ('imin(is_used_once)', 'a@64', 'b@64'), 0), ('ilt', ('imin', ('unpack_64_2x32_split_y', a), ('unpack_64_2x32_split_y', b)), 0), '(options->lower_int64_options & nir_lower_minmax64) != 0'),
   (('ige', ('imax(is_used_once)', 'a@64', 'b@64'), 0), ('ige', ('imax', ('unpack_64_2x32_split_y', a), ('unpack_64_2x32_split_y', b)), 0), '(options->lower_int64_options & nir_lower_minmax64) != 0'),
   (('ige', ('imin(is_used_once)', 'a@64', 'b@64'), 0), ('ige', ('imin', ('unpack_64_2x32_split_y', a), ('unpack_64_2x32_split_y', b)), 0), '(options->lower_int64_options & nir_lower_minmax64) != 0'),
   (('ilt', 'a@64', 0), ('ilt', ('unpack_64_2x32_split_y', a), 0), '(options->lower_int64_options & nir_lower_icmp64) != 0'),
   (('ige', 'a@64', 0), ('ige', ('unpack_64_2x32_split_y', a), 0), '(options->lower_int64_options & nir_lower_icmp64) != 0'),

   (('ine', 'a@64', 0), ('ine', ('ior', ('unpack_64_2x32_split_x', a), ('unpack_64_2x32_split_y', a)), 0), '(options->lower_int64_options & nir_lower_icmp64) != 0'),
   (('ieq', 'a@64', 0), ('ieq', ('ior', ('unpack_64_2x32_split_x', a), ('unpack_64_2x32_split_y', a)), 0), '(options->lower_int64_options & nir_lower_icmp64) != 0'),
   # 0u < uint(a) <=> uint(a) != 0u
   (('ult', 0, 'a@64'), ('ine', ('ior', ('unpack_64_2x32_split_x', a), ('unpack_64_2x32_split_y', a)), 0), '(options->lower_int64_options & nir_lower_icmp64) != 0'),

   # Alternative lowering that doesn't rely on bfi.
   (('bitfield_insert', 'base', 'insert', 'offset', 'bits'),
    ('bcsel', ('ult', 31, 'bits'),
     'insert',
    (('ior',
     ('iand', 'base', ('inot', ('ishl', ('isub', ('ishl', 1, 'bits'), 1), 'offset'))),
     ('iand', ('ishl', 'insert', 'offset'), ('ishl', ('isub', ('ishl', 1, 'bits'), 1), 'offset'))))),
    'options->lower_bitfield_insert_to_shifts'),

   # Alternative lowering that uses bitfield_select.
   (('bitfield_insert', 'base', 'insert', 'offset', 'bits'),
    ('bcsel', ('ult', 31, 'bits'), 'insert',
              ('bitfield_select', ('bfm', 'bits', 'offset'), ('ishl', 'insert', 'offset'), 'base')),
    'options->lower_bitfield_insert_to_bitfield_select'),

   (('ibitfield_extract', 'value', 'offset', 'bits'),
    ('bcsel', ('ult', 31, 'bits'), 'value',
              ('ibfe', 'value', 'offset', 'bits')),
    'options->lower_bitfield_extract'),

   (('ubitfield_extract', 'value', 'offset', 'bits'),
    ('bcsel', ('ult', 31, 'bits'), 'value',
              ('ubfe', 'value', 'offset', 'bits')),
    'options->lower_bitfield_extract'),

   # (src0 & src1) | (~src0 & src2). Constant fold if src2 is 0.
   (('bitfield_select', a, b, 0), ('iand', a, b)),
   (('bitfield_select', a, ('iand', a, b), c), ('bitfield_select', a, b, c)),

   # Note that these opcodes are defined to only use the five least significant bits of 'offset' and 'bits'
   (('ubfe', 'value', 'offset', ('iand', 31, 'bits')), ('ubfe', 'value', 'offset', 'bits')),
   (('ubfe', 'value', ('iand', 31, 'offset'), 'bits'), ('ubfe', 'value', 'offset', 'bits')),
   (('ibfe', 'value', 'offset', ('iand', 31, 'bits')), ('ibfe', 'value', 'offset', 'bits')),
   (('ibfe', 'value', ('iand', 31, 'offset'), 'bits'), ('ibfe', 'value', 'offset', 'bits')),
   (('bfm', 'bits', ('iand', 31, 'offset')), ('bfm', 'bits', 'offset')),
   (('bfm', ('iand', 31, 'bits'), 'offset'), ('bfm', 'bits', 'offset')),

   # Optimizations for ubitfield_extract(value, offset, umin(bits, 32-(offset&0x1f))) and such
   (('ult', a, ('umin', ('iand', a, b), c)), False),
   (('ult', 31, ('umin', '#bits(is_ult_32)', a)), False),
   (('ubfe', 'value', 'offset', ('umin', 'width', ('iadd', 32, ('ineg', ('iand', 31, 'offset'))))),
    ('ubfe', 'value', 'offset', 'width')),
   (('ibfe', 'value', 'offset', ('umin', 'width', ('iadd', 32, ('ineg', ('iand', 31, 'offset'))))),
    ('ibfe', 'value', 'offset', 'width')),
   (('bfm', ('umin', 'width', ('iadd', 32, ('ineg', ('iand', 31, 'offset')))), 'offset'),
    ('bfm', 'width', 'offset')),

   # open-coded BFM
   (('iadd@32', ('ishl', 1, a), -1), ('bfm', a, 0), 'options->lower_bitfield_insert_to_bitfield_select || options->lower_bitfield_insert'),
   (('ishl', ('bfm', a, 0), b), ('bfm', a, b)),

   # Section 8.8 (Integer Functions) of the GLSL 4.60 spec says:
   #
   #    If bits is zero, the result will be zero.
   #
   # These patterns prevent other patterns from generating invalid results
   # when count is zero.
   (('ubfe', a, b, 0), 0),
   (('ibfe', a, b, 0), 0),

   (('ubfe', a, 0, '#b'), ('iand', a, ('ushr', 0xffffffff, ('ineg', b)))),

   (('b2i32', ('ine', ('ubfe', a, b, 1), 0)), ('ubfe', a, b, 1)),
   (('b2i32', ('ine', ('ibfe', a, b, 1), 0)), ('ubfe', a, b, 1)), # ubfe in the replacement is correct
   (('ine', ('ibfe(is_used_once)', a, '#b', '#c'), 0), ('ine', ('iand', a, ('ishl', ('ushr', 0xffffffff, ('ineg', c)), b)), 0)),
   (('ieq', ('ibfe(is_used_once)', a, '#b', '#c'), 0), ('ieq', ('iand', a, ('ishl', ('ushr', 0xffffffff, ('ineg', c)), b)), 0)),
   (('ine', ('ubfe(is_used_once)', a, '#b', '#c'), 0), ('ine', ('iand', a, ('ishl', ('ushr', 0xffffffff, ('ineg', c)), b)), 0)),
   (('ieq', ('ubfe(is_used_once)', a, '#b', '#c'), 0), ('ieq', ('iand', a, ('ishl', ('ushr', 0xffffffff, ('ineg', c)), b)), 0)),

   (('ibitfield_extract', 'value', 'offset', 'bits'),
    ('bcsel', ('ieq', 0, 'bits'),
     0,
     ('ishr',
       ('ishl', 'value', ('isub', ('isub', 32, 'bits'), 'offset')),
       ('isub', 32, 'bits'))),
    'options->lower_bitfield_extract_to_shifts'),

   (('ubitfield_extract', 'value', 'offset', 'bits'),
    ('iand',
     ('ushr', 'value', 'offset'),
     ('bcsel', ('ieq', 'bits', 32),
      0xffffffff,
      ('isub', ('ishl', 1, 'bits'), 1))),
    'options->lower_bitfield_extract_to_shifts'),

   (('ifind_msb', 'value'),
    ('ufind_msb', ('bcsel', ('ilt', 'value', 0), ('inot', 'value'), 'value')),
    'options->lower_ifind_msb'),

   (('ifind_msb', 'value'),
    ('bcsel', ('ige', ('ifind_msb_rev', 'value'), 0),
     ('isub', 31, ('ifind_msb_rev', 'value')),
     ('ifind_msb_rev', 'value')),
    'options->lower_find_msb_to_reverse'),

    (('ufind_msb', 'value'),
     ('bcsel', ('ige', ('ufind_msb_rev', 'value'), 0),
      ('isub', 31, ('ufind_msb_rev', 'value')),
      ('ufind_msb_rev', 'value')),
     'options->lower_find_msb_to_reverse'),

   (('uclz', a), ('umin', 32, ('ufind_msb_rev', a)), 'options->lower_uclz'),

   (('find_lsb', 'value'),
    ('ufind_msb', ('iand', 'value', ('ineg', 'value'))),
    'options->lower_find_lsb'),

   (('extract_i8', a, 'b@32'),
    ('ishr', ('ishl', a, ('imul', ('isub', 3, b), 8)), 24),
    'options->lower_extract_byte'),

   (('extract_u8', a, 'b@32'),
    ('iand', ('ushr', a, ('imul', b, 8)), 0xff),
    'options->lower_extract_byte'),

   (('extract_i16', a, 'b@32'),
    ('ishr', ('ishl', a, ('imul', ('isub', 1, b), 16)), 16),
    'options->lower_extract_word'),

   (('extract_u16', a, 'b@32'),
    ('iand', ('ushr', a, ('imul', b, 16)), 0xffff),
    'options->lower_extract_word'),

    (('pack_unorm_2x16', 'v'),
     ('pack_uvec2_to_uint',
        ('f2u32', ('fround_even', ('fmul', ('fsat', 'v'), 65535.0)))),
     'options->lower_pack_unorm_2x16'),

    (('pack_unorm_4x8', 'v'),
     ('pack_uvec4_to_uint',
        ('f2u32', ('fround_even', ('fmul', ('fsat', 'v'), 255.0)))),
     'options->lower_pack_unorm_4x8'),

    (('pack_snorm_2x16', 'v'),
     ('pack_uvec2_to_uint',
        ('f2i32', ('fround_even', ('fmul', ('fmin', 1.0, ('fmax', -1.0, 'v')), 32767.0)))),
     'options->lower_pack_snorm_2x16'),

    (('pack_snorm_4x8', 'v'),
     ('pack_uvec4_to_uint',
        ('f2i32', ('fround_even', ('fmul', ('fmin', 1.0, ('fmax', -1.0, 'v')), 127.0)))),
     'options->lower_pack_snorm_4x8'),

    (('unpack_unorm_2x16', 'v'),
     ('fdiv', ('u2f32', ('vec2', ('extract_u16', 'v', 0),
                                  ('extract_u16', 'v', 1))),
              65535.0),
     'options->lower_unpack_unorm_2x16'),

    (('unpack_unorm_4x8', 'v'),
     ('fdiv', ('u2f32', ('vec4', ('extract_u8', 'v', 0),
                                  ('extract_u8', 'v', 1),
                                  ('extract_u8', 'v', 2),
                                  ('extract_u8', 'v', 3))),
              255.0),
     'options->lower_unpack_unorm_4x8'),

    (('unpack_snorm_2x16', 'v'),
     ('fmin', 1.0, ('fmax', -1.0, ('fdiv', ('i2f', ('vec2', ('extract_i16', 'v', 0),
                                                            ('extract_i16', 'v', 1))),
                                           32767.0))),
     'options->lower_unpack_snorm_2x16'),

    (('unpack_snorm_4x8', 'v'),
     ('fmin', 1.0, ('fmax', -1.0, ('fdiv', ('i2f', ('vec4', ('extract_i8', 'v', 0),
                                                            ('extract_i8', 'v', 1),
                                                            ('extract_i8', 'v', 2),
                                                            ('extract_i8', 'v', 3))),
                                           127.0))),
     'options->lower_unpack_snorm_4x8'),

   (('pack_half_2x16_split', 'a@32', 'b@32'),
    ('ior', ('ishl', ('u2u32', ('f2f16', b)), 16), ('u2u32', ('f2f16', a))),
    'options->lower_pack_split'),

   (('unpack_half_2x16_split_x', 'a@32'),
    ('f2f32', ('u2u16', a)),
    'options->lower_pack_split'),

   (('unpack_half_2x16_split_y', 'a@32'),
    ('f2f32', ('u2u16', ('ushr', a, 16))),
    'options->lower_pack_split'),

   (('pack_32_2x16_split', 'a@16', 'b@16'),
    ('ior', ('ishl', ('u2u32', b), 16), ('u2u32', a)),
    'options->lower_pack_split'),

   (('unpack_32_2x16_split_x', 'a@32'),
    ('u2u16', a),
    'options->lower_pack_split'),

   (('unpack_32_2x16_split_y', 'a@32'),
    ('u2u16', ('ushr', 'a', 16)),
    'options->lower_pack_split'),

   (('isign', a), ('imin', ('imax', a, -1), 1), 'options->lower_isign'),
   (('imin', ('imax', a, -1), 1), ('isign', a), '!options->lower_isign'),
   (('imax', ('imin', a, 1), -1), ('isign', a), '!options->lower_isign'),
   # float(0 < NaN) - float(NaN < 0) = float(False) - float(False) = 0 - 0 = 0
   # Mark the new comparisons precise to prevent them being changed to 'a !=
   # 0' or 'a == 0'.
   (('fsign', a), ('fsub', ('b2f', ('!flt', 0.0, a)), ('b2f', ('!flt', a, 0.0))), 'options->lower_fsign'),

   # Address/offset calculations:
   # Drivers supporting imul24 should use the nir_lower_amul() pass, this
   # rule converts everyone else to imul:
   (('amul', a, b), ('imul', a, b), '!options->has_imul24'),

   (('umul24', a, b),
    ('imul', ('iand', a, 0xffffff), ('iand', b, 0xffffff)),
    '!options->has_umul24'),
   (('umad24', a, b, c),
    ('iadd', ('imul', ('iand', a, 0xffffff), ('iand', b, 0xffffff)), c),
    '!options->has_umad24'),

   # Relaxed 24bit ops
   (('imul24_relaxed', a, b), ('imul24', a, b), 'options->has_imul24'),
   (('imul24_relaxed', a, b), ('imul', a, b), '!options->has_imul24'),
   (('umad24_relaxed', a, b, c), ('umad24', a, b, c), 'options->has_umad24'),
   (('umad24_relaxed', a, b, c), ('iadd', ('umul24_relaxed', a, b), c), '!options->has_umad24'),
   (('umul24_relaxed', a, b), ('umul24', a, b), 'options->has_umul24'),
   (('umul24_relaxed', a, b), ('imul', a, b), '!options->has_umul24'),

   (('imad24_ir3', a, b, 0), ('imul24', a, b)),
   (('imad24_ir3', a, 0, c), (c)),
   (('imad24_ir3', a, 1, c), ('iadd', a, c)),

   # if first two srcs are const, crack apart the imad so constant folding
   # can clean up the imul:
   # TODO ffma should probably get a similar rule:
   (('imad24_ir3', '#a', '#b', c), ('iadd', ('imul', a, b), c)),

   # These will turn 24b address/offset calc back into 32b shifts, but
   # it should be safe to get back some of the bits of precision that we
   # already decided were no necessary:
   (('imul24', a, '#b@32(is_pos_power_of_two)'), ('ishl', a, ('find_lsb', b)), '!options->lower_bitops'),
   (('imul24', a, '#b@32(is_neg_power_of_two)'), ('ineg', ('ishl', a, ('find_lsb', ('iabs', b)))), '!options->lower_bitops'),
   (('imul24', a, 0), (0)),
])

for bit_size in [8, 16, 32, 64]:
   cond = '!options->lower_uadd_sat'
   if bit_size == 64:
      cond += ' && !(options->lower_int64_options & nir_lower_iadd64)'
   add = 'iadd@' + str(bit_size)

   optimizations += [
      (('bcsel', ('ult', ('iadd', a, b), a), -1, (add, a, b)), ('uadd_sat', a, b), cond),
      (('bcsel', ('uge', ('iadd', a, b), a), (add, a, b), -1), ('uadd_sat', a, b), cond),
      (('bcsel', ('ieq', ('uadd_carry', a, b), 0), (add, a, b), -1), ('uadd_sat', a, b), cond),
      (('bcsel', ('ine', ('uadd_carry', a, b), 0), -1, (add, a, b)), ('uadd_sat', a, b), cond),
   ]

for bit_size in [8, 16, 32, 64]:
   cond = '!options->lower_usub_sat'
   if bit_size == 64:
      cond += ' && !(options->lower_int64_options & nir_lower_usub_sat64)'
   add = 'iadd@' + str(bit_size)

   optimizations += [
      (('bcsel', ('ult', a, b), 0, (add, a, ('ineg', b))), ('usub_sat', a, b), cond),
      (('bcsel', ('uge', a, b), (add, a, ('ineg', b)), 0), ('usub_sat', a, b), cond),
      (('bcsel', ('ieq', ('usub_borrow', a, b), 0), (add, a, ('ineg', b)), 0), ('usub_sat', a, b), cond),
      (('bcsel', ('ine', ('usub_borrow', a, b), 0), 0, (add, a, ('ineg', b))), ('usub_sat', a, b), cond),
   ]

# bit_size dependent lowerings
for bit_size in [8, 16, 32, 64]:
   # convenience constants
   intmax = (1 << (bit_size - 1)) - 1
   intmin = 1 << (bit_size - 1)

   optimizations += [
      (('iadd_sat@' + str(bit_size), a, b),
       ('bcsel', ('ige', b, 1), ('bcsel', ('ilt', ('iadd', a, b), a), intmax, ('iadd', a, b)),
                                ('bcsel', ('ilt', a, ('iadd', a, b)), intmin, ('iadd', a, b))), 'options->lower_iadd_sat'),
      (('isub_sat@' + str(bit_size), a, b),
       ('bcsel', ('ilt', b, 0), ('bcsel', ('ilt', ('isub', a, b), a), intmax, ('isub', a, b)),
                                ('bcsel', ('ilt', a, ('isub', a, b)), intmin, ('isub', a, b))), 'options->lower_iadd_sat'),
   ]

invert = OrderedDict([('feq', 'fneu'), ('fneu', 'feq')])

for left, right in itertools.combinations_with_replacement(invert.keys(), 2):
   optimizations.append((('inot', ('ior(is_used_once)', (left, a, b), (right, c, d))),
                         ('iand', (invert[left], a, b), (invert[right], c, d))))
   optimizations.append((('inot', ('iand(is_used_once)', (left, a, b), (right, c, d))),
                         ('ior', (invert[left], a, b), (invert[right], c, d))))

# Optimize f2bN(b2f(x)) -> x
for size in type_sizes('bool'):
    aN = 'a@' + str(size)
    f2bN = 'f2b' + str(size)
    optimizations.append(((f2bN, ('b2f', aN)), a))

# Optimize x2yN(b2x(x)) -> b2y
for x, y in itertools.product(['f', 'u', 'i'], ['f', 'u', 'i']):
   if x != 'f' and y != 'f' and x != y:
      continue

   b2x = 'b2f' if x == 'f' else 'b2i'
   b2y = 'b2f' if y == 'f' else 'b2i'
   x2yN = '{}2{}'.format(x, y)
   optimizations.append(((x2yN, (b2x, a)), (b2y, a)))

# Optimize away x2xN(a@N)
for t in ['int', 'uint', 'float', 'bool']:
   for N in type_sizes(t):
      x2xN = '{0}2{0}{1}'.format(t[0], N)
      aN = 'a@{0}'.format(N)
      optimizations.append(((x2xN, aN), a))

# Optimize x2xN(y2yM(a@P)) -> y2yN(a) for integers
# In particular, we can optimize away everything except upcast of downcast and
# upcasts where the type differs from the other cast
for N, M in itertools.product(type_sizes('uint'), type_sizes('uint')):
   if N < M:
      # The outer cast is a down-cast.  It doesn't matter what the size of the
      # argument of the inner cast is because we'll never been in the upcast
      # of downcast case.  Regardless of types, we'll always end up with y2yN
      # in the end.
      for x, y in itertools.product(['i', 'u'], ['i', 'u']):
         x2xN = '{0}2{0}{1}'.format(x, N)
         y2yM = '{0}2{0}{1}'.format(y, M)
         y2yN = '{0}2{0}{1}'.format(y, N)
         optimizations.append(((x2xN, (y2yM, a)), (y2yN, a)))
   elif N > M:
      # If the outer cast is an up-cast, we have to be more careful about the
      # size of the argument of the inner cast and with types.  In this case,
      # the type is always the type of type up-cast which is given by the
      # outer cast.
      for P in type_sizes('uint'):
         # We can't optimize away up-cast of down-cast.
         if M < P:
            continue

         # Because we're doing down-cast of down-cast, the types always have
         # to match between the two casts
         for x in ['i', 'u']:
            x2xN = '{0}2{0}{1}'.format(x, N)
            x2xM = '{0}2{0}{1}'.format(x, M)
            aP = 'a@{0}'.format(P)
            optimizations.append(((x2xN, (x2xM, aP)), (x2xN, a)))
   else:
      # The N == M case is handled by other optimizations
      pass

# Downcast operations should be able to see through pack
for t in ['i', 'u']:
    for N in [8, 16, 32]:
        x2xN = '{0}2{0}{1}'.format(t, N)
        optimizations += [
            ((x2xN, ('pack_64_2x32_split', a, b)), (x2xN, a)),
            ((x2xN, ('pack_64_2x32_split', a, b)), (x2xN, a)),
        ]

# Optimize comparisons with up-casts
for t in ['int', 'uint', 'float']:
    for N, M in itertools.product(type_sizes(t), repeat=2):
        if N == 1 or N >= M:
            continue

        cond = 'true'
        if N == 8:
            cond = 'options->support_8bit_alu'
        elif N == 16:
            cond = 'options->support_16bit_alu'
        x2xM = '{0}2{0}{1}'.format(t[0], M)
        x2xN = '{0}2{0}{1}'.format(t[0], N)
        aN = 'a@' + str(N)
        bN = 'b@' + str(N)
        xeq = 'feq' if t == 'float' else 'ieq'
        xne = 'fneu' if t == 'float' else 'ine'
        xge = '{0}ge'.format(t[0])
        xlt = '{0}lt'.format(t[0])

        # Up-casts are lossless so for correctly signed comparisons of
        # up-casted values we can do the comparison at the largest of the two
        # original sizes and drop one or both of the casts.  (We have
        # optimizations to drop the no-op casts which this may generate.)
        for P in type_sizes(t):
            if P == 1 or P > N:
                continue

            bP = 'b@' + str(P)
            optimizations += [
                ((xeq, (x2xM, aN), (x2xM, bP)), (xeq, a, (x2xN, b)), cond),
                ((xne, (x2xM, aN), (x2xM, bP)), (xne, a, (x2xN, b)), cond),
                ((xge, (x2xM, aN), (x2xM, bP)), (xge, a, (x2xN, b)), cond),
                ((xlt, (x2xM, aN), (x2xM, bP)), (xlt, a, (x2xN, b)), cond),
                ((xge, (x2xM, bP), (x2xM, aN)), (xge, (x2xN, b), a), cond),
                ((xlt, (x2xM, bP), (x2xM, aN)), (xlt, (x2xN, b), a), cond),
            ]

        # The next bit doesn't work on floats because the range checks would
        # get way too complicated.
        if t in ['int', 'uint']:
            if t == 'int':
                xN_min = -(1 << (N - 1))
                xN_max = (1 << (N - 1)) - 1
            elif t == 'uint':
                xN_min = 0
                xN_max = (1 << N) - 1
            else:
                assert False

            # If we're up-casting and comparing to a constant, we can unfold
            # the comparison into a comparison with the shrunk down constant
            # and a check that the constant fits in the smaller bit size.
            optimizations += [
                ((xeq, (x2xM, aN), '#b'),
                 ('iand', (xeq, a, (x2xN, b)), (xeq, (x2xM, (x2xN, b)), b)), cond),
                ((xne, (x2xM, aN), '#b'),
                 ('ior', (xne, a, (x2xN, b)), (xne, (x2xM, (x2xN, b)), b)), cond),
                ((xlt, (x2xM, aN), '#b'),
                 ('iand', (xlt, xN_min, b),
                          ('ior', (xlt, xN_max, b), (xlt, a, (x2xN, b)))), cond),
                ((xlt, '#a', (x2xM, bN)),
                 ('iand', (xlt, a, xN_max),
                          ('ior', (xlt, a, xN_min), (xlt, (x2xN, a), b))), cond),
                ((xge, (x2xM, aN), '#b'),
                 ('iand', (xge, xN_max, b),
                          ('ior', (xge, xN_min, b), (xge, a, (x2xN, b)))), cond),
                ((xge, '#a', (x2xM, bN)),
                 ('iand', (xge, a, xN_min),
                          ('ior', (xge, a, xN_max), (xge, (x2xN, a), b))), cond),
            ]

# Convert masking followed by signed downcast to just unsigned downcast
optimizations += [
    (('i2i32', ('iand', 'a@64', 0xffffffff)), ('u2u32', a)),
    (('i2i16', ('iand', 'a@32', 0xffff)), ('u2u16', a)),
    (('i2i16', ('iand', 'a@64', 0xffff)), ('u2u16', a)),
    (('i2i8', ('iand', 'a@16', 0xff)), ('u2u8', a)),
    (('i2i8', ('iand', 'a@32', 0xff)), ('u2u8', a)),
    (('i2i8', ('iand', 'a@64', 0xff)), ('u2u8', a)),
]

# Some operations such as iadd have the property that the bottom N bits of the
# output only depends on the bottom N bits of each of the inputs so we can
# remove casts
for N in [16, 32]:
    for M in [8, 16]:
        if M >= N:
            continue

        aN = 'a@' + str(N)
        u2uM = 'u2u{0}'.format(M)
        i2iM = 'i2i{0}'.format(M)

        for x in ['u', 'i']:
            x2xN = '{0}2{0}{1}'.format(x, N)
            extract_xM = 'extract_{0}{1}'.format(x, M)

            x2xN_M_bits = '{0}(only_lower_{1}_bits_used)'.format(x2xN, M)
            extract_xM_M_bits = \
                '{0}(only_lower_{1}_bits_used)'.format(extract_xM, M)
            optimizations += [
                ((x2xN_M_bits, (u2uM, aN)), a),
                ((extract_xM_M_bits, aN, 0), a),
            ]

            bcsel_M_bits = 'bcsel(only_lower_{0}_bits_used)'.format(M)
            optimizations += [
                ((bcsel_M_bits, c, (x2xN, (u2uM, aN)), b), ('bcsel', c, a, b)),
                ((bcsel_M_bits, c, (x2xN, (i2iM, aN)), b), ('bcsel', c, a, b)),
                ((bcsel_M_bits, c, (extract_xM, aN, 0), b), ('bcsel', c, a, b)),
            ]

            for op in ['iadd', 'imul', 'iand', 'ior', 'ixor']:
                op_M_bits = '{0}(only_lower_{1}_bits_used)'.format(op, M)
                optimizations += [
                    ((op_M_bits, (x2xN, (u2uM, aN)), b), (op, a, b)),
                    ((op_M_bits, (x2xN, (i2iM, aN)), b), (op, a, b)),
                    ((op_M_bits, (extract_xM, aN, 0), b), (op, a, b)),
                ]

def fexp2i(exp, bits):
   # Generate an expression which constructs value 2.0^exp or 0.0.
   #
   # We assume that exp is already in a valid range:
   #
   #   * [-15, 15] for 16-bit float
   #   * [-127, 127] for 32-bit float
   #   * [-1023, 1023] for 16-bit float
   #
   # If exp is the lowest value in the valid range, a value of 0.0 is
   # constructed.  Otherwise, the value 2.0^exp is constructed.
   if bits == 16:
      return ('i2i16', ('ishl', ('iadd', exp, 15), 10))
   elif bits == 32:
      return ('ishl', ('iadd', exp, 127), 23)
   elif bits == 64:
      return ('pack_64_2x32_split', 0, ('ishl', ('iadd', exp, 1023), 20))
   else:
      assert False

def ldexp(f, exp, bits):
   # The maximum possible range for a normal exponent is [-126, 127] and,
   # throwing in denormals, you get a maximum range of [-149, 127].  This
   # means that we can potentially have a swing of +-276.  If you start with
   # FLT_MAX, you actually have to do ldexp(FLT_MAX, -278) to get it to flush
   # all the way to zero.  The GLSL spec only requires that we handle a subset
   # of this range.  From version 4.60 of the spec:
   #
   #    "If exp is greater than +128 (single-precision) or +1024
   #    (double-precision), the value returned is undefined. If exp is less
   #    than -126 (single-precision) or -1022 (double-precision), the value
   #    returned may be flushed to zero. Additionally, splitting the value
   #    into a significand and exponent using frexp() and then reconstructing
   #    a floating-point value using ldexp() should yield the original input
   #    for zero and all finite non-denormalized values."
   #
   # The SPIR-V spec has similar language.
   #
   # In order to handle the maximum value +128 using the fexp2i() helper
   # above, we have to split the exponent in half and do two multiply
   # operations.
   #
   # First, we clamp exp to a reasonable range.  Specifically, we clamp to
   # twice the full range that is valid for the fexp2i() function above.  If
   # exp/2 is the bottom value of that range, the fexp2i() expression will
   # yield 0.0f which, when multiplied by f, will flush it to zero which is
   # allowed by the GLSL and SPIR-V specs for low exponent values.  If the
   # value is clamped from above, then it must have been above the supported
   # range of the GLSL built-in and therefore any return value is acceptable.
   if bits == 16:
      exp = ('imin', ('imax', exp, -30), 30)
   elif bits == 32:
      exp = ('imin', ('imax', exp, -254), 254)
   elif bits == 64:
      exp = ('imin', ('imax', exp, -2046), 2046)
   else:
      assert False

   # Now we compute two powers of 2, one for exp/2 and one for exp-exp/2.
   # (We use ishr which isn't the same for -1, but the -1 case still works
   # since we use exp-exp/2 as the second exponent.)  While the spec
   # technically defines ldexp as f * 2.0^exp, simply multiplying once doesn't
   # work with denormals and doesn't allow for the full swing in exponents
   # that you can get with normalized values.  Instead, we create two powers
   # of two and multiply by them each in turn.  That way the effective range
   # of our exponent is doubled.
   pow2_1 = fexp2i(('ishr', exp, 1), bits)
   pow2_2 = fexp2i(('isub', exp, ('ishr', exp, 1)), bits)
   return ('fmul', ('fmul', f, pow2_1), pow2_2)

optimizations += [
   (('ldexp@16', 'x', 'exp'), ldexp('x', 'exp', 16), 'options->lower_ldexp'),
   (('ldexp@32', 'x', 'exp'), ldexp('x', 'exp', 32), 'options->lower_ldexp'),
   (('ldexp@64', 'x', 'exp'), ldexp('x', 'exp', 64), 'options->lower_ldexp'),
]

# Unreal Engine 4 demo applications open-codes bitfieldReverse()
def bitfield_reverse_ue4(u):
    step1 = ('ior', ('ishl', u, 16), ('ushr', u, 16))
    step2 = ('ior', ('ishl', ('iand', step1, 0x00ff00ff), 8), ('ushr', ('iand', step1, 0xff00ff00), 8))
    step3 = ('ior', ('ishl', ('iand', step2, 0x0f0f0f0f), 4), ('ushr', ('iand', step2, 0xf0f0f0f0), 4))
    step4 = ('ior', ('ishl', ('iand', step3, 0x33333333), 2), ('ushr', ('iand', step3, 0xcccccccc), 2))
    step5 = ('ior(many-comm-expr)', ('ishl', ('iand', step4, 0x55555555), 1), ('ushr', ('iand', step4, 0xaaaaaaaa), 1))

    return step5

# Cyberpunk 2077 open-codes bitfieldReverse()
def bitfield_reverse_cp2077(u):
    step1 = ('ior', ('ishl', u, 16), ('ushr', u, 16))
    step2 = ('ior', ('iand', ('ishl', step1, 1), 0xaaaaaaaa), ('iand', ('ushr', step1, 1), 0x55555555))
    step3 = ('ior', ('iand', ('ishl', step2, 2), 0xcccccccc), ('iand', ('ushr', step2, 2), 0x33333333))
    step4 = ('ior', ('iand', ('ishl', step3, 4), 0xf0f0f0f0), ('iand', ('ushr', step3, 4), 0x0f0f0f0f))
    step5 = ('ior(many-comm-expr)', ('iand', ('ishl', step4, 8), 0xff00ff00), ('iand', ('ushr', step4, 8), 0x00ff00ff))

    return step5

optimizations += [(bitfield_reverse_ue4('x@32'), ('bitfield_reverse', 'x'), '!options->lower_bitfield_reverse')]
optimizations += [(bitfield_reverse_cp2077('x@32'), ('bitfield_reverse', 'x'), '!options->lower_bitfield_reverse')]

# "all_equal(eq(a, b), vec(~0))" is the same as "all_equal(a, b)"
# "any_nequal(neq(a, b), vec(0))" is the same as "any_nequal(a, b)"
for ncomp in [2, 3, 4, 8, 16]:
   optimizations += [
      (('ball_iequal' + str(ncomp), ('ieq', a, b), ~0), ('ball_iequal' + str(ncomp), a, b)),
      (('ball_iequal' + str(ncomp), ('feq', a, b), ~0), ('ball_fequal' + str(ncomp), a, b)),
      (('bany_inequal' + str(ncomp), ('ine', a, b), 0), ('bany_inequal' + str(ncomp), a, b)),
      (('bany_inequal' + str(ncomp), ('fneu', a, b), 0), ('bany_fnequal' + str(ncomp), a, b)),
   ]

# For any float comparison operation, "cmp", if you have "a == a && a cmp b"
# then the "a == a" is redundant because it's equivalent to "a is not NaN"
# and, if a is a NaN then the second comparison will fail anyway.
for op in ['flt', 'fge', 'feq']:
   optimizations += [
      (('iand', ('feq', a, a), (op, a, b)), ('!' + op, a, b)),
      (('iand', ('feq', a, a), (op, b, a)), ('!' + op, b, a)),
   ]

# Add optimizations to handle the case where the result of a ternary is
# compared to a constant.  This way we can take things like
#
# (a ? 0 : 1) > 0
#
# and turn it into
#
# a ? (0 > 0) : (1 > 0)
#
# which constant folding will eat for lunch.  The resulting ternary will
# further get cleaned up by the boolean reductions above and we will be
# left with just the original variable "a".
for op in ['feq', 'fneu', 'ieq', 'ine']:
   optimizations += [
      ((op, ('bcsel', 'a', '#b', '#c'), '#d'),
       ('bcsel', 'a', (op, 'b', 'd'), (op, 'c', 'd'))),
   ]

for op in ['flt', 'fge', 'ilt', 'ige', 'ult', 'uge']:
   optimizations += [
      ((op, ('bcsel', 'a', '#b', '#c'), '#d'),
       ('bcsel', 'a', (op, 'b', 'd'), (op, 'c', 'd'))),
      ((op, '#d', ('bcsel', a, '#b', '#c')),
       ('bcsel', 'a', (op, 'd', 'b'), (op, 'd', 'c'))),
   ]


# For example, this converts things like
#
#    1 + mix(0, a - 1, condition)
#
# into
#
#    mix(1, (a-1)+1, condition)
#
# Other optimizations will rearrange the constants.
for op in ['fadd', 'fmul', 'fmulz', 'iadd', 'imul']:
   optimizations += [
      ((op, ('bcsel(is_used_once)', a, '#b', c), '#d'), ('bcsel', a, (op, b, d), (op, c, d)))
   ]

# For derivatives in compute shaders, GLSL_NV_compute_shader_derivatives
# states:
#
#     If neither layout qualifier is specified, derivatives in compute shaders
#     return zero, which is consistent with the handling of built-in texture
#     functions like texture() in GLSL 4.50 compute shaders.
for op in ['fddx', 'fddx_fine', 'fddx_coarse',
           'fddy', 'fddy_fine', 'fddy_coarse']:
   optimizations += [
      ((op, 'a'), 0.0, 'info->stage == MESA_SHADER_COMPUTE && info->cs.derivative_group == DERIVATIVE_GROUP_NONE')
]

# Some optimizations for ir3-specific instructions.
optimizations += [
   # 'al * bl': If either 'al' or 'bl' is zero, return zero.
   (('umul_low', '#a(is_lower_half_zero)', 'b'), (0)),
   # '(ah * bl) << 16 + c': If either 'ah' or 'bl' is zero, return 'c'.
   (('imadsh_mix16', '#a@32(is_lower_half_zero)', 'b@32', 'c@32'), ('c')),
   (('imadsh_mix16', 'a@32', '#b@32(is_upper_half_zero)', 'c@32'), ('c')),
]

# These kinds of sequences can occur after nir_opt_peephole_select.
#
# NOTE: fadd is not handled here because that gets in the way of ffma
# generation in the i965 driver.  Instead, fadd and ffma are handled in
# late_optimizations.

for op in ['flrp']:
    optimizations += [
        (('bcsel', a, (op + '(is_used_once)', b, c, d), (op, b, c, e)), (op, b, c, ('bcsel', a, d, e))),
        (('bcsel', a, (op, b, c, d), (op + '(is_used_once)', b, c, e)), (op, b, c, ('bcsel', a, d, e))),
        (('bcsel', a, (op + '(is_used_once)', b, c, d), (op, b, e, d)), (op, b, ('bcsel', a, c, e), d)),
        (('bcsel', a, (op, b, c, d), (op + '(is_used_once)', b, e, d)), (op, b, ('bcsel', a, c, e), d)),
        (('bcsel', a, (op + '(is_used_once)', b, c, d), (op, e, c, d)), (op, ('bcsel', a, b, e), c, d)),
        (('bcsel', a, (op, b, c, d), (op + '(is_used_once)', e, c, d)), (op, ('bcsel', a, b, e), c, d)),
    ]

for op in ['fmulz', 'fmul', 'iadd', 'imul', 'iand', 'ior', 'ixor', 'fmin', 'fmax', 'imin', 'imax', 'umin', 'umax']:
    optimizations += [
        (('bcsel', a, (op + '(is_used_once)', b, c), (op, b, 'd(is_not_const)')), (op, b, ('bcsel', a, c, d))),
        (('bcsel', a, (op + '(is_used_once)', b, 'c(is_not_const)'), (op, b, d)), (op, b, ('bcsel', a, c, d))),
        (('bcsel', a, (op, b, 'c(is_not_const)'), (op + '(is_used_once)', b, d)), (op, b, ('bcsel', a, c, d))),
        (('bcsel', a, (op, b, c), (op + '(is_used_once)', b, 'd(is_not_const)')), (op, b, ('bcsel', a, c, d))),
    ]

for op in ['fpow']:
    optimizations += [
        (('bcsel', a, (op + '(is_used_once)', b, c), (op, b, d)), (op, b, ('bcsel', a, c, d))),
        (('bcsel', a, (op, b, c), (op + '(is_used_once)', b, d)), (op, b, ('bcsel', a, c, d))),
        (('bcsel', a, (op + '(is_used_once)', b, c), (op, d, c)), (op, ('bcsel', a, b, d), c)),
        (('bcsel', a, (op, b, c), (op + '(is_used_once)', d, c)), (op, ('bcsel', a, b, d), c)),
    ]

for op in ['frcp', 'frsq', 'fsqrt', 'fexp2', 'flog2', 'fsign', 'fsin', 'fcos', 'fsin_amd', 'fcos_amd', 'fsin_mdg', 'fcos_mdg', 'fsin_agx', 'fneg', 'fabs', 'fsign']:
    optimizations += [
        (('bcsel', c, (op + '(is_used_once)', a), (op + '(is_used_once)', b)), (op, ('bcsel', c, a, b))),
    ]

for op in ['ineg', 'iabs', 'inot', 'isign']:
    optimizations += [
        ((op, ('bcsel', c, '#a', '#b')), ('bcsel', c, (op, a), (op, b))),
    ]

optimizations.extend([
    (('fisnormal', 'a@16'), ('ult', 0xfff, ('iadd', ('ishl', a, 1), 0x800)), 'options->lower_fisnormal'),
    (('fisnormal', 'a@32'), ('ult', 0x1ffffff, ('iadd', ('ishl', a, 1), 0x1000000)), 'options->lower_fisnormal'),
    (('fisnormal', 'a@64'), ('ult', 0x3fffffffffffff, ('iadd', ('ishl', a, 1), 0x20000000000000)), 'options->lower_fisnormal')
    ])

# This section contains optimizations to propagate downsizing conversions of
# constructed vectors into vectors of downsized components. Whether this is
# useful depends on the SIMD semantics of the backend. On a true SIMD machine,
# this reduces the register pressure of the vector itself and often enables the
# conversions to be eliminated via other algebraic rules or constant folding.
# In the worst case on a SIMD architecture, the propagated conversions may be
# revectorized via nir_opt_vectorize so instruction count is minimally
# impacted.
#
# On a machine with SIMD-within-a-register only, this actually
# counterintuitively hurts instruction count. These machines are the same that
# require vectorize_vec2_16bit, so we predicate the optimizations on that flag
# not being set.
#
# Finally for scalar architectures, there should be no difference in generated
# code since it all ends up scalarized at the end, but it might minimally help
# compile-times.

for i in range(2, 4 + 1):
   for T in ('f', 'u', 'i'):
      vec_inst = ('vec' + str(i),)

      indices = ['a', 'b', 'c', 'd']
      suffix_in = tuple((indices[j] + '@32') for j in range(i))

      to_16 = '{}2{}16'.format(T, T)
      to_mp = '{}2{}mp'.format(T, T)

      out_16 = tuple((to_16, indices[j]) for j in range(i))
      out_mp = tuple((to_mp, indices[j]) for j in range(i))

      optimizations  += [
         ((to_16, vec_inst + suffix_in), vec_inst + out_16, '!options->vectorize_vec2_16bit'),
      ]
      # u2ump doesn't exist, because it's equal to i2imp
      if T in ['f', 'i']:
          optimizations  += [
             ((to_mp, vec_inst + suffix_in), vec_inst + out_mp, '!options->vectorize_vec2_16bit')
          ]

# This section contains "late" optimizations that should be run before
# creating ffmas and calling regular optimizations for the final time.
# Optimizations should go here if they help code generation and conflict
# with the regular optimizations.
before_ffma_optimizations = [
   # Propagate constants down multiplication chains
   (('~fmul(is_used_once)', ('fmul(is_used_once)', 'a(is_not_const)', '#b'), 'c(is_not_const)'), ('fmul', ('fmul', a, c), b)),
   (('imul(is_used_once)', ('imul(is_used_once)', 'a(is_not_const)', '#b'), 'c(is_not_const)'), ('imul', ('imul', a, c), b)),
   (('~fadd(is_used_once)', ('fadd(is_used_once)', 'a(is_not_const)', '#b'), 'c(is_not_const)'), ('fadd', ('fadd', a, c), b)),
   (('iadd(is_used_once)', ('iadd(is_used_once)', 'a(is_not_const)', '#b'), 'c(is_not_const)'), ('iadd', ('iadd', a, c), b)),

   (('~fadd', ('fmul', a, b), ('fmul', a, c)), ('fmul', a, ('fadd', b, c))),
   (('iadd', ('imul', a, b), ('imul', a, c)), ('imul', a, ('iadd', b, c))),
   (('~fadd', ('fneg', a), a), 0.0),
   (('iadd', ('ineg', a), a), 0),
   (('iadd', ('ineg', a), ('iadd', a, b)), b),
   (('iadd', a, ('iadd', ('ineg', a), b)), b),
   (('~fadd', ('fneg', a), ('fadd', a, b)), b),
   (('~fadd', a, ('fadd', ('fneg', a), b)), b),

   (('~flrp', ('fadd(is_used_once)', a, -1.0), ('fadd(is_used_once)', a,  1.0), d), ('fadd', ('flrp', -1.0,  1.0, d), a)),
   (('~flrp', ('fadd(is_used_once)', a,  1.0), ('fadd(is_used_once)', a, -1.0), d), ('fadd', ('flrp',  1.0, -1.0, d), a)),
   (('~flrp', ('fadd(is_used_once)', a, '#b'), ('fadd(is_used_once)', a, '#c'), d), ('fadd', ('fmul', d, ('fadd', c, ('fneg', b))), ('fadd', a, b))),
]

# This section contains "late" optimizations that should be run after the
# regular optimizations have finished.  Optimizations should go here if
# they help code generation but do not necessarily produce code that is
# more easily optimizable.
late_optimizations = [
   # The rearrangements are fine w.r.t. NaN.  However, they produce incorrect
   # results if one operand is +Inf and the other is -Inf.
   #
   # 1. Inf + -Inf = NaN
   # 2. ∀x: x + NaN = NaN and x - NaN = NaN
   # 3. ∀x: x != NaN = true
   # 4. ∀x, ∀ cmp ∈ {<, >, ≤, ≥, =}: x cmp NaN = false
   #
   #               a=Inf, b=-Inf   a=-Inf, b=Inf    a=NaN    b=NaN
   #  (a+b) < 0        false            false       false    false
   #      a < -b       false            false       false    false
   # -(a+b) < 0        false            false       false    false
   #     -a < b        false            false       false    false
   #  (a+b) >= 0       false            false       false    false
   #      a >= -b      true             true        false    false
   # -(a+b) >= 0       false            false       false    false
   #     -a >= b       true             true        false    false
   #  (a+b) == 0       false            false       false    false
   #      a == -b      true             true        false    false
   #  (a+b) != 0       true             true        true     true
   #      a != -b      false            false       true     true
   (('flt',                        ('fadd(is_used_once)', a, b),  0.0), ('flt',          a, ('fneg', b))),
   (('flt', ('fneg(is_used_once)', ('fadd(is_used_once)', a, b)), 0.0), ('flt', ('fneg', a),         b)),
   (('flt', 0.0,                        ('fadd(is_used_once)', a, b) ), ('flt', ('fneg', a),         b)),
   (('flt', 0.0, ('fneg(is_used_once)', ('fadd(is_used_once)', a, b))), ('flt',          a, ('fneg', b))),
   (('~fge',                        ('fadd(is_used_once)', a, b),  0.0), ('fge',          a, ('fneg', b))),
   (('~fge', ('fneg(is_used_once)', ('fadd(is_used_once)', a, b)), 0.0), ('fge', ('fneg', a),         b)),
   (('~fge', 0.0,                        ('fadd(is_used_once)', a, b) ), ('fge', ('fneg', a),         b)),
   (('~fge', 0.0, ('fneg(is_used_once)', ('fadd(is_used_once)', a, b))), ('fge',          a, ('fneg', b))),
   (('~feq', ('fadd(is_used_once)', a, b), 0.0), ('feq', a, ('fneg', b))),
   (('~fneu', ('fadd(is_used_once)', a, b), 0.0), ('fneu', a, ('fneg', b))),

   # If either source must be finite, then the original (a+b) cannot produce
   # NaN due to Inf-Inf.  The patterns and the replacements produce the same
   # result if b is NaN. Therefore, the replacements are exact.
   (('fge',                        ('fadd(is_used_once)', 'a(is_finite)', b),  0.0), ('fge',          a, ('fneg', b))),
   (('fge', ('fneg(is_used_once)', ('fadd(is_used_once)', 'a(is_finite)', b)), 0.0), ('fge', ('fneg', a),         b)),
   (('fge', 0.0,                        ('fadd(is_used_once)', 'a(is_finite)', b) ), ('fge', ('fneg', a),         b)),
   (('fge', 0.0, ('fneg(is_used_once)', ('fadd(is_used_once)', 'a(is_finite)', b))), ('fge',          a, ('fneg', b))),
   (('feq',  ('fadd(is_used_once)', 'a(is_finite)', b), 0.0), ('feq',  a, ('fneg', b))),
   (('fneu', ('fadd(is_used_once)', 'a(is_finite)', b), 0.0), ('fneu', a, ('fneg', b))),

   # This is how SpvOpFOrdNotEqual might be implemented.  Replace it with
   # SpvOpLessOrGreater.
   (('iand', ('fneu', a, b),   ('iand', ('feq', a, a), ('feq', b, b))), ('ior', ('!flt', a, b), ('!flt', b, a))),
   (('iand', ('fneu', a, 0.0),          ('feq', a, a)                ), ('!flt', 0.0, ('fabs', a))),

   # This is how SpvOpFUnordEqual might be implemented.  Replace it with
   # !SpvOpLessOrGreater.
   (('ior', ('feq', a, b),   ('ior', ('fneu', a, a), ('fneu', b, b))), ('inot', ('ior', ('!flt', a, b), ('!flt', b, a)))),
   (('ior', ('feq', a, 0.0),         ('fneu', a, a),                ), ('inot', ('!flt', 0.0, ('fabs', a)))),

   # nir_lower_to_source_mods will collapse this, but its existence during the
   # optimization loop can prevent other optimizations.
   (('fneg', ('fneg', a)), a)
]

# re-combine inexact mul+add to ffma. Do this before fsub so that a * b - c
# gets combined to fma(a, b, -c).
for sz, mulz in itertools.product([16, 32, 64], [False, True]):
    # fmulz/ffmaz only for fp32
    if mulz and sz != 32:
        continue

    # Fuse the correct fmul. Only consider fmuls where the only users are fadd
    # (or fneg/fabs which are assumed to be propagated away), as a heuristic to
    # avoid fusing in cases where it's harmful.
    fmul = ('fmulz' if mulz else 'fmul') + '(is_only_used_by_fadd)'
    ffma = 'ffmaz' if mulz else 'ffma'

    fadd = '~fadd@{}'.format(sz)
    option = 'options->fuse_ffma{}'.format(sz)

    late_optimizations.extend([
        ((fadd, (fmul, a, b), c), (ffma, a, b, c), option),

        ((fadd, ('fneg(is_only_used_by_fadd)', (fmul, a, b)), c),
         (ffma, ('fneg', a), b, c), option),

        ((fadd, ('fabs(is_only_used_by_fadd)', (fmul, a, b)), c),
         (ffma, ('fabs', a), ('fabs', b), c), option),

        ((fadd, ('fneg(is_only_used_by_fadd)', ('fabs', (fmul, a, b))), c),
         (ffma, ('fneg', ('fabs', a)), ('fabs', b), c), option),
    ])

late_optimizations.extend([
   # Subtractions get lowered during optimization, so we need to recombine them
   (('fadd@8', a, ('fneg', 'b')), ('fsub', 'a', 'b'), 'options->has_fsub'),
   (('fadd@16', a, ('fneg', 'b')), ('fsub', 'a', 'b'), 'options->has_fsub'),
   (('fadd@32', a, ('fneg', 'b')), ('fsub', 'a', 'b'), 'options->has_fsub'),
   (('fadd@64', a, ('fneg', 'b')), ('fsub', 'a', 'b'), 'options->has_fsub && !(options->lower_doubles_options & nir_lower_dsub)'),

   (('fneg', a), ('fmul', a, -1.0), 'options->lower_fneg'),
   (('iadd', a, ('ineg', 'b')), ('isub', 'a', 'b'), 'options->has_isub || options->lower_ineg'),
   (('ineg', a), ('isub', 0, a), 'options->lower_ineg'),
   (('iabs', a), ('imax', a, ('ineg', a)), 'options->lower_iabs'),

   (('iadd', ('iadd(is_used_once)', 'a(is_not_const)', 'b(is_not_const)'), 'c(is_not_const)'), ('iadd3', a, b, c), 'options->has_iadd3'),
   (('iadd', ('isub(is_used_once)', 'a(is_not_const)', 'b(is_not_const)'), 'c(is_not_const)'), ('iadd3', a, ('ineg', b), c), 'options->has_iadd3'),
   (('isub', ('isub(is_used_once)', 'a(is_not_const)', 'b(is_not_const)'), 'c(is_not_const)'), ('iadd3', a, ('ineg', b), ('ineg', c)), 'options->has_iadd3'),

    # fneg_lo / fneg_hi
   (('vec2(is_only_used_as_float)', ('fneg@16', a), b), ('fmul', ('vec2', a, b), ('vec2', -1.0, 1.0)), 'options->vectorize_vec2_16bit'),
   (('vec2(is_only_used_as_float)', a, ('fneg@16', b)), ('fmul', ('vec2', a, b), ('vec2', 1.0, -1.0)), 'options->vectorize_vec2_16bit'),

   # These are duplicated from the main optimizations table.  The late
   # patterns that rearrange expressions like x - .5 < 0 to x < .5 can create
   # new patterns like these.  The patterns that compare with zero are removed
   # because they are unlikely to be created in by anything in
   # late_optimizations.
   (('flt', '#b(is_gt_0_and_lt_1)', ('fsat(is_used_once)', a)), ('flt', b, a)),
   (('fge', ('fsat(is_used_once)', a), '#b(is_gt_0_and_lt_1)'), ('fge', a, b)),
   (('feq', ('fsat(is_used_once)', a), '#b(is_gt_0_and_lt_1)'), ('feq', a, b)),
   (('fneu', ('fsat(is_used_once)', a), '#b(is_gt_0_and_lt_1)'), ('fneu', a, b)),

   (('fge', ('fsat(is_used_once)', a), 1.0), ('fge', a, 1.0)),

   (('~fge', ('fmin(is_used_once)', ('fadd(is_used_once)', a, b), ('fadd', c, d)), 0.0), ('iand', ('fge', a, ('fneg', b)), ('fge', c, ('fneg', d)))),

   (('flt', ('fneg', a), ('fneg', b)), ('flt', b, a)),
   (('fge', ('fneg', a), ('fneg', b)), ('fge', b, a)),
   (('feq', ('fneg', a), ('fneg', b)), ('feq', b, a)),
   (('fneu', ('fneg', a), ('fneg', b)), ('fneu', b, a)),
   (('flt', ('fneg', a), -1.0), ('flt', 1.0, a)),
   (('flt', -1.0, ('fneg', a)), ('flt', a, 1.0)),
   (('fge', ('fneg', a), -1.0), ('fge', 1.0, a)),
   (('fge', -1.0, ('fneg', a)), ('fge', a, 1.0)),
   (('fneu', ('fneg', a), -1.0), ('fneu', 1.0, a)),
   (('feq', -1.0, ('fneg', a)), ('feq', a, 1.0)),

   (('ior', a, a), a),
   (('iand', a, a), a),

   (('~fadd', ('fneg(is_used_once)', ('fsat(is_used_once)', 'a(is_not_fmul)')), 1.0), ('fsat', ('fadd', 1.0, ('fneg', a)))),

   (('fdot2', a, b), ('fdot2_replicated', a, b), 'options->fdot_replicates'),
   (('fdot3', a, b), ('fdot3_replicated', a, b), 'options->fdot_replicates'),
   (('fdot4', a, b), ('fdot4_replicated', a, b), 'options->fdot_replicates'),
   (('fdph', a, b), ('fdph_replicated', a, b), 'options->fdot_replicates'),

   (('~flrp', ('fadd(is_used_once)', a, b), ('fadd(is_used_once)', a, c), d), ('fadd', ('flrp', b, c, d), a)),

   # Approximate handling of fround_even for DX9 addressing from gallium nine on
   # DX9-class hardware with no proper fround support.  This is in
   # late_optimizations so that the is_integral() opts in the main pass get a
   # chance to eliminate the fround_even first.
   (('fround_even', a), ('bcsel',
                         ('feq', ('ffract', a), 0.5),
                         ('fadd', ('ffloor', ('fadd', a, 0.5)), 1.0),
                         ('ffloor', ('fadd', a, 0.5))), 'options->lower_fround_even'),

   # A similar operation could apply to any ffma(#a, b, #(-a/2)), but this
   # particular operation is common for expanding values stored in a texture
   # from [0,1] to [-1,1].
   (('~ffma@32', a,  2.0, -1.0), ('flrp', -1.0,  1.0,          a ), '!options->lower_flrp32'),
   (('~ffma@32', a, -2.0, -1.0), ('flrp', -1.0,  1.0, ('fneg', a)), '!options->lower_flrp32'),
   (('~ffma@32', a, -2.0,  1.0), ('flrp',  1.0, -1.0,          a ), '!options->lower_flrp32'),
   (('~ffma@32', a,  2.0,  1.0), ('flrp',  1.0, -1.0, ('fneg', a)), '!options->lower_flrp32'),
   (('~fadd@32', ('fmul(is_used_once)',  2.0, a), -1.0), ('flrp', -1.0,  1.0,          a ), '!options->lower_flrp32'),
   (('~fadd@32', ('fmul(is_used_once)', -2.0, a), -1.0), ('flrp', -1.0,  1.0, ('fneg', a)), '!options->lower_flrp32'),
   (('~fadd@32', ('fmul(is_used_once)', -2.0, a),  1.0), ('flrp',  1.0, -1.0,          a ), '!options->lower_flrp32'),
   (('~fadd@32', ('fmul(is_used_once)',  2.0, a),  1.0), ('flrp',  1.0, -1.0, ('fneg', a)), '!options->lower_flrp32'),

    # flrp(a, b, a)
    # a*(1-a) + b*a
    # a + -a*a + a*b    (1)
    # a + a*(b - a)
    # Option 1: ffma(a, (b-a), a)
    #
    # Alternately, after (1):
    # a*(1+b) + -a*a
    # a*((1+b) + -a)
    #
    # Let b=1
    #
    # Option 2: ffma(a, 2, -(a*a))
    # Option 3: ffma(a, 2, (-a)*a)
    # Option 4: ffma(a, -a, (2*a)
    # Option 5: a * (2 - a)
    #
    # There are a lot of other possible combinations.
   (('~ffma@32', ('fadd', b, ('fneg', a)), a, a), ('flrp', a, b, a), '!options->lower_flrp32'),
   (('~ffma@32', a, 2.0, ('fneg', ('fmul', a, a))), ('flrp', a, 1.0, a), '!options->lower_flrp32'),
   (('~ffma@32', a, 2.0, ('fmul', ('fneg', a), a)), ('flrp', a, 1.0, a), '!options->lower_flrp32'),
   (('~ffma@32', a, ('fneg', a), ('fmul', 2.0, a)), ('flrp', a, 1.0, a), '!options->lower_flrp32'),
   (('~fmul@32', a, ('fadd', 2.0, ('fneg', a))),    ('flrp', a, 1.0, a), '!options->lower_flrp32'),

   # we do these late so that we don't get in the way of creating ffmas
   (('fmin', ('fadd(is_used_once)', '#c', a), ('fadd(is_used_once)', '#c', b)), ('fadd', c, ('fmin', a, b))),
   (('fmax', ('fadd(is_used_once)', '#c', a), ('fadd(is_used_once)', '#c', b)), ('fadd', c, ('fmax', a, b))),

   # Putting this in 'optimizations' interferes with the bcsel(a, op(b, c),
   # op(b, d)) => op(b, bcsel(a, c, d)) transformations.  I do not know why.
   (('bcsel', ('feq', ('fsqrt', 'a(is_not_negative)'), 0.0), intBitsToFloat(0x7f7fffff), ('frsq', a)),
    ('fmin', ('frsq', a), intBitsToFloat(0x7f7fffff))),

   # Things that look like DPH in the source shader may get expanded to
   # something that looks like dot(v1.xyz, v2.xyz) + v1.w by the time it gets
   # to NIR.  After FFMA is generated, this can look like:
   #
   #    fadd(ffma(v1.z, v2.z, ffma(v1.y, v2.y, fmul(v1.x, v2.x))), v1.w)
   #
   # Reassociate the last addition into the first multiplication.
   #
   # Some shaders do not use 'invariant' in vertex and (possibly) geometry
   # shader stages on some outputs that are intended to be invariant.  For
   # various reasons, this optimization may not be fully applied in all
   # shaders used for different rendering passes of the same geometry.  This
   # can result in Z-fighting artifacts (at best).  For now, disable this
   # optimization in these stages.  See bugzilla #111490.  In tessellation
   # stages applications seem to use 'precise' when necessary, so allow the
   # optimization in those stages.
   (('~fadd', ('ffma(is_used_once)', a, b, ('ffma', c, d, ('fmul(is_used_once)', 'e(is_not_const_and_not_fsign)', 'f(is_not_const_and_not_fsign)'))), 'g(is_not_const)'),
    ('ffma', a, b, ('ffma', c, d, ('ffma', e, 'f', 'g'))), '(info->stage != MESA_SHADER_VERTEX && info->stage != MESA_SHADER_GEOMETRY) && !options->intel_vec4'),
   (('~fadd', ('ffma(is_used_once)', a, b, ('fmul(is_used_once)', 'c(is_not_const_and_not_fsign)', 'd(is_not_const_and_not_fsign)') ), 'e(is_not_const)'),
    ('ffma', a, b, ('ffma', c, d, e)), '(info->stage != MESA_SHADER_VERTEX && info->stage != MESA_SHADER_GEOMETRY) && !options->intel_vec4'),
   (('~fadd', ('fneg', ('ffma(is_used_once)', a, b, ('ffma', c, d, ('fmul(is_used_once)', 'e(is_not_const_and_not_fsign)', 'f(is_not_const_and_not_fsign)')))), 'g(is_not_const)'),
    ('ffma', ('fneg', a), b, ('ffma', ('fneg', c), d, ('ffma', ('fneg', e), 'f', 'g'))), '(info->stage != MESA_SHADER_VERTEX && info->stage != MESA_SHADER_GEOMETRY) && !options->intel_vec4'),

   (('~fadd', ('ffmaz(is_used_once)', a, b, ('ffmaz', c, d, ('fmulz(is_used_once)', 'e(is_not_const_and_not_fsign)', 'f(is_not_const_and_not_fsign)'))), 'g(is_not_const)'),
    ('ffmaz', a, b, ('ffmaz', c, d, ('ffmaz', e, 'f', 'g'))), '(info->stage != MESA_SHADER_VERTEX && info->stage != MESA_SHADER_GEOMETRY) && !options->intel_vec4'),
   (('~fadd', ('ffmaz(is_used_once)', a, b, ('fmulz(is_used_once)', 'c(is_not_const_and_not_fsign)', 'd(is_not_const_and_not_fsign)') ), 'e(is_not_const)'),
    ('ffmaz', a, b, ('ffmaz', c, d, e)), '(info->stage != MESA_SHADER_VERTEX && info->stage != MESA_SHADER_GEOMETRY) && !options->intel_vec4'),
   (('~fadd', ('fneg', ('ffmaz(is_used_once)', a, b, ('ffmaz', c, d, ('fmulz(is_used_once)', 'e(is_not_const_and_not_fsign)', 'f(is_not_const_and_not_fsign)')))), 'g(is_not_const)'),
    ('ffmaz', ('fneg', a), b, ('ffmaz', ('fneg', c), d, ('ffmaz', ('fneg', e), 'f', 'g'))), '(info->stage != MESA_SHADER_VERTEX && info->stage != MESA_SHADER_GEOMETRY) && !options->intel_vec4'),

   # Section 8.8 (Integer Functions) of the GLSL 4.60 spec says:
   #
   #    If bits is zero, the result will be zero.
   #
   # These prevent the next two lowerings generating incorrect results when
   # count is zero.
   (('ubfe', a, b, 0), 0),
   (('ibfe', a, b, 0), 0),

   # On Intel GPUs, BFE is a 3-source instruction.  Like all 3-source
   # instructions on Intel GPUs, it cannot have an immediate values as
   # sources.  There are also limitations on source register strides.  As a
   # result, it is very easy for 3-source instruction combined with either
   # loads of immediate values or copies from weird register strides to be
   # more expensive than the primitive instructions it represents.
   (('ubfe', a, '#b', '#c'), ('iand', ('ushr', 0xffffffff, ('ineg', c)), ('ushr', a, b)), 'options->avoid_ternary_with_two_constants'),

   # b is the lowest order bit to be extracted and c is the number of bits to
   # extract.  The inner shift removes the bits above b + c by shifting left
   # 32 - (b + c).  ishl only sees the low 5 bits of the shift count, which is
   # -(b + c).  The outer shift moves the bit that was at b to bit zero.
   # After the first shift, that bit is now at b + (32 - (b + c)) or 32 - c.
   # This means that it must be shifted right by 32 - c or -c bits.
   (('ibfe', a, '#b', '#c'), ('ishr', ('ishl', a, ('ineg', ('iadd', b, c))), ('ineg', c)), 'options->avoid_ternary_with_two_constants'),

   # Clean up no-op shifts that may result from the bfe lowerings.
   (('ishl', a, 0), a),
   (('ishl', a, -32), a),
   (('ishr', a, 0), a),
   (('ishr', a, -32), a),
   (('ushr', a, 0), a),

   (('extract_i8', ('extract_i8', a, b), 0), ('extract_i8', a, b)),
   (('extract_i8', ('extract_u8', a, b), 0), ('extract_i8', a, b)),
   (('extract_u8', ('extract_i8', a, b), 0), ('extract_u8', a, b)),
   (('extract_u8', ('extract_u8', a, b), 0), ('extract_u8', a, b)),
])

# A few more extract cases we'd rather leave late
for N in [16, 32]:
    aN = 'a@{0}'.format(N)
    u2uM = 'u2u{0}'.format(M)
    i2iM = 'i2i{0}'.format(M)

    for x in ['u', 'i']:
        x2xN = '{0}2{0}{1}'.format(x, N)
        extract_x8 = 'extract_{0}8'.format(x)
        extract_x16 = 'extract_{0}16'.format(x)

        late_optimizations.extend([
            ((x2xN, ('u2u8', aN)), (extract_x8, a, 0), '!options->lower_extract_byte'),
            ((x2xN, ('i2i8', aN)), (extract_x8, a, 0), '!options->lower_extract_byte'),
        ])

        if N > 16:
            late_optimizations.extend([
                ((x2xN, ('u2u16', aN)), (extract_x16, a, 0), '!options->lower_extract_word'),
                ((x2xN, ('i2i16', aN)), (extract_x16, a, 0), '!options->lower_extract_word'),
            ])

# Byte insertion
late_optimizations.extend([(('ishl', ('extract_u8', 'a@32', 0), 8 * i), ('insert_u8', a, i), '!options->lower_insert_byte') for i in range(1, 4)])
late_optimizations.extend([(('iand', ('ishl', 'a@32', 8 * i), 0xff << (8 * i)), ('insert_u8', a, i), '!options->lower_insert_byte') for i in range(1, 4)])
late_optimizations.append((('ishl', 'a@32', 24), ('insert_u8', a, 3), '!options->lower_insert_byte'))

late_optimizations += [
   # Word insertion
   (('ishl', 'a@32', 16), ('insert_u16', a, 1), '!options->lower_insert_word'),

   # Extract and then insert
   (('insert_u8', ('extract_u8', 'a', 0), b), ('insert_u8', a, b)),
   (('insert_u16', ('extract_u16', 'a', 0), b), ('insert_u16', a, b)),
]

# Integer sizes
for s in [8, 16, 32, 64]:
    late_optimizations.extend([
        (('iand', ('ine(is_used_once)', 'a@{}'.format(s), 0), ('ine', 'b@{}'.format(s), 0)), ('ine', ('umin', a, b), 0)),
        (('ior',  ('ieq(is_used_once)', 'a@{}'.format(s), 0), ('ieq', 'b@{}'.format(s), 0)), ('ieq', ('umin', a, b), 0)),
    ])

# Float sizes
for s in [16, 32, 64]:
    late_optimizations.extend([
       (('~fadd@{}'.format(s), 1.0, ('fmul(is_used_once)', c , ('fadd', b, -1.0 ))), ('fadd', ('fadd', 1.0, ('fneg', c)), ('fmul', b, c)), 'options->lower_flrp{}'.format(s)),
       (('bcsel', a, 0, ('b2f{}'.format(s), ('inot', 'b@bool'))), ('b2f{}'.format(s), ('inot', ('ior', a, b)))),
    ])

for op in ['fadd']:
    late_optimizations += [
        (('bcsel', a, (op + '(is_used_once)', b, c), (op, b, d)), (op, b, ('bcsel', a, c, d))),
        (('bcsel', a, (op, b, c), (op + '(is_used_once)', b, d)), (op, b, ('bcsel', a, c, d))),
    ]

for op in ['ffma', 'ffmaz']:
    late_optimizations += [
        (('bcsel', a, (op + '(is_used_once)', b, c, d), (op, b, c, e)), (op, b, c, ('bcsel', a, d, e))),
        (('bcsel', a, (op, b, c, d), (op + '(is_used_once)', b, c, e)), (op, b, c, ('bcsel', a, d, e))),

        (('bcsel', a, (op + '(is_used_once)', b, c, d), (op, b, e, d)), (op, b, ('bcsel', a, c, e), d)),
        (('bcsel', a, (op, b, c, d), (op + '(is_used_once)', b, e, d)), (op, b, ('bcsel', a, c, e), d)),
    ]

# mediump: If an opcode is surrounded by conversions, remove the conversions.
# The rationale is that type conversions + the low precision opcode are more
# expensive that the same arithmetic opcode at higher precision.
#
# This must be done in late optimizations, because we need normal optimizations to
# first eliminate temporary up-conversions such as in op1(f2fmp(f2f32(op2()))).
#
# Unary opcodes
for op in ['fabs', 'fceil', 'fcos', 'fddx', 'fddx_coarse', 'fddx_fine', 'fddy',
           'fddy_coarse', 'fddy_fine', 'fexp2', 'ffloor', 'ffract', 'flog2', 'fneg',
           'frcp', 'fround_even', 'frsq', 'fsat', 'fsign', 'fsin', 'fsqrt']:
    late_optimizations += [(('~f2f32', (op, ('f2fmp', a))), (op, a))]

# Binary opcodes
for op in ['fadd', 'fdiv', 'fmax', 'fmin', 'fmod', 'fmul', 'fpow', 'frem']:
    late_optimizations += [(('~f2f32', (op, ('f2fmp', a), ('f2fmp', b))), (op, a, b))]

# Ternary opcodes
for op in ['ffma', 'flrp']:
    late_optimizations += [(('~f2f32', (op, ('f2fmp', a), ('f2fmp', b), ('f2fmp', c))), (op, a, b, c))]

# Comparison opcodes
for op in ['feq', 'fge', 'flt', 'fneu']:
    late_optimizations += [(('~' + op, ('f2fmp', a), ('f2fmp', b)), (op, a, b))]

# Do this last, so that the f2fmp patterns above have effect.
late_optimizations += [
  # Convert *2*mp instructions to concrete *2*16 instructions. At this point
  # any conversions that could have been removed will have been removed in
  # nir_opt_algebraic so any remaining ones are required.
  (('f2fmp', a), ('f2f16', a)),
  (('f2imp', a), ('f2i16', a)),
  (('f2ump', a), ('f2u16', a)),
  (('i2imp', a), ('i2i16', a)),
  (('i2fmp', a), ('i2f16', a)),
  (('i2imp', a), ('u2u16', a)),
  (('u2fmp', a), ('u2f16', a)),
  (('fisfinite', a), ('flt', ('fabs', a), float("inf"))),

  (('fcsel', ('slt', 0, a), b, c), ('fcsel_gt', a, b, c), "options->has_fused_comp_and_csel"),
  (('fcsel', ('slt', a, 0), b, c), ('fcsel_gt', ('fneg', a), b, c), "options->has_fused_comp_and_csel"),
  (('fcsel', ('sge', a, 0), b, c), ('fcsel_ge', a, b, c), "options->has_fused_comp_and_csel"),
  (('fcsel', ('sge', 0, a), b, c), ('fcsel_ge', ('fneg', a), b, c), "options->has_fused_comp_and_csel"),

  (('bcsel', ('ilt', 0, 'a@32'), 'b@32', 'c@32'), ('i32csel_gt', a, b, c), "options->has_fused_comp_and_csel"),
  (('bcsel', ('ilt', 'a@32', 0), 'b@32', 'c@32'), ('i32csel_ge', a, c, b), "options->has_fused_comp_and_csel"),
  (('bcsel', ('ige', 'a@32', 0), 'b@32', 'c@32'), ('i32csel_ge', a, b, c), "options->has_fused_comp_and_csel"),
  (('bcsel', ('ige', 0, 'a@32'), 'b@32', 'c@32'), ('i32csel_gt', a, c, b), "options->has_fused_comp_and_csel"),

  (('bcsel', ('flt', 0, 'a@32'), 'b@32', 'c@32'), ('fcsel_gt', a, b, c), "options->has_fused_comp_and_csel"),
  (('bcsel', ('flt', 'a@32', 0), 'b@32', 'c@32'), ('fcsel_gt', ('fneg', a), b, c), "options->has_fused_comp_and_csel"),
  (('bcsel', ('fge', 'a@32', 0), 'b@32', 'c@32'), ('fcsel_ge', a, b, c), "options->has_fused_comp_and_csel"),
  (('bcsel', ('fge', 0, 'a@32'), 'b@32', 'c@32'), ('fcsel_ge', ('fneg', a), b, c), "options->has_fused_comp_and_csel"),
]

distribute_src_mods = [
   # Try to remove some spurious negations rather than pushing them down.
   (('fmul', ('fneg', a), ('fneg', b)), ('fmul', a, b)),
   (('ffma', ('fneg', a), ('fneg', b), c), ('ffma', a, b, c)),
   (('fdot2_replicated', ('fneg', a), ('fneg', b)), ('fdot2_replicated', a, b)),
   (('fdot3_replicated', ('fneg', a), ('fneg', b)), ('fdot3_replicated', a, b)),
   (('fdot4_replicated', ('fneg', a), ('fneg', b)), ('fdot4_replicated', a, b)),
   (('fneg', ('fneg', a)), a),

   (('fneg', ('fmul(is_used_once)', a, b)), ('fmul', ('fneg', a), b)),
   (('fabs', ('fmul(is_used_once)', a, b)), ('fmul', ('fabs', a), ('fabs', b))),

   (('fneg', ('ffma(is_used_once)', a, b, c)), ('ffma', ('fneg', a), b, ('fneg', c))),
   (('fneg', ('flrp(is_used_once)', a, b, c)), ('flrp', ('fneg', a), ('fneg', b), c)),
   (('fneg', ('~fadd(is_used_once)', a, b)), ('fadd', ('fneg', a), ('fneg', b))),

   # Note that fmin <-> fmax.  I don't think there is a way to distribute
   # fabs() into fmin or fmax.
   (('fneg', ('fmin(is_used_once)', a, b)), ('fmax', ('fneg', a), ('fneg', b))),
   (('fneg', ('fmax(is_used_once)', a, b)), ('fmin', ('fneg', a), ('fneg', b))),

   (('fneg', ('fdot2_replicated(is_used_once)', a, b)), ('fdot2_replicated', ('fneg', a), b)),
   (('fneg', ('fdot3_replicated(is_used_once)', a, b)), ('fdot3_replicated', ('fneg', a), b)),
   (('fneg', ('fdot4_replicated(is_used_once)', a, b)), ('fdot4_replicated', ('fneg', a), b)),

   # fdph works mostly like fdot, but to get the correct result, the negation
   # must be applied to the second source.
   (('fneg', ('fdph_replicated(is_used_once)', a, b)), ('fdph_replicated', a, ('fneg', b))),

   (('fneg', ('fsign(is_used_once)', a)), ('fsign', ('fneg', a))),
   (('fabs', ('fsign(is_used_once)', a)), ('fsign', ('fabs', a))),
]

print(nir_algebraic.AlgebraicPass("nir_opt_algebraic", optimizations).render())
print(nir_algebraic.AlgebraicPass("nir_opt_algebraic_before_ffma",
                                  before_ffma_optimizations).render())
print(nir_algebraic.AlgebraicPass("nir_opt_algebraic_late",
                                  late_optimizations).render())
print(nir_algebraic.AlgebraicPass("nir_opt_algebraic_distribute_src_mods",
                                  distribute_src_mods).render())
