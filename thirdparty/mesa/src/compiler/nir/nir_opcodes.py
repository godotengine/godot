#
# Copyright (C) 2014 Connor Abbott
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
#    Connor Abbott (cwabbott0@gmail.com)

import re

# Class that represents all the information we have about the opcode
# NOTE: this must be kept in sync with nir_op_info

class Opcode(object):
   """Class that represents all the information we have about the opcode
   NOTE: this must be kept in sync with nir_op_info
   """
   def __init__(self, name, output_size, output_type, input_sizes,
                input_types, is_conversion, algebraic_properties, const_expr):
      """Parameters:

      - name is the name of the opcode (prepend nir_op_ for the enum name)
      - all types are strings that get nir_type_ prepended to them
      - input_types is a list of types
      - is_conversion is true if this opcode represents a type conversion
      - algebraic_properties is a space-seperated string, where nir_op_is_ is
        prepended before each entry
      - const_expr is an expression or series of statements that computes the
        constant value of the opcode given the constant values of its inputs.

      Constant expressions are formed from the variables src0, src1, ...,
      src(N-1), where N is the number of arguments.  The output of the
      expression should be stored in the dst variable.  Per-component input
      and output variables will be scalars and non-per-component input and
      output variables will be a struct with fields named x, y, z, and w
      all of the correct type.  Input and output variables can be assumed
      to already be of the correct type and need no conversion.  In
      particular, the conversion from the C bool type to/from  NIR_TRUE and
      NIR_FALSE happens automatically.

      For per-component instructions, the entire expression will be
      executed once for each component.  For non-per-component
      instructions, the expression is expected to store the correct values
      in dst.x, dst.y, etc.  If "dst" does not exist anywhere in the
      constant expression, an assignment to dst will happen automatically
      and the result will be equivalent to "dst = <expression>" for
      per-component instructions and "dst.x = dst.y = ... = <expression>"
      for non-per-component instructions.
      """
      assert isinstance(name, str)
      assert isinstance(output_size, int)
      assert isinstance(output_type, str)
      assert isinstance(input_sizes, list)
      assert isinstance(input_sizes[0], int)
      assert isinstance(input_types, list)
      assert isinstance(input_types[0], str)
      assert isinstance(is_conversion, bool)
      assert isinstance(algebraic_properties, str)
      assert isinstance(const_expr, str)
      assert len(input_sizes) == len(input_types)
      assert 0 <= output_size <= 5 or (output_size == 8) or (output_size == 16)
      for size in input_sizes:
         assert 0 <= size <= 5 or (size == 8) or (size == 16)
         if output_size != 0:
            assert size != 0
      self.name = name
      self.num_inputs = len(input_sizes)
      self.output_size = output_size
      self.output_type = output_type
      self.input_sizes = input_sizes
      self.input_types = input_types
      self.is_conversion = is_conversion
      self.algebraic_properties = algebraic_properties
      self.const_expr = const_expr

# helper variables for strings
tfloat = "float"
tint = "int"
tbool = "bool"
tbool1 = "bool1"
tbool8 = "bool8"
tbool16 = "bool16"
tbool32 = "bool32"
tuint = "uint"
tuint8 = "uint8"
tint16 = "int16"
tuint16 = "uint16"
tfloat16 = "float16"
tfloat32 = "float32"
tint32 = "int32"
tuint32 = "uint32"
tint64 = "int64"
tuint64 = "uint64"
tfloat64 = "float64"

_TYPE_SPLIT_RE = re.compile(r'(?P<type>int|uint|float|bool)(?P<bits>\d+)?')

def type_has_size(type_):
    m = _TYPE_SPLIT_RE.match(type_)
    assert m is not None, 'Invalid NIR type string: "{}"'.format(type_)
    return m.group('bits') is not None

def type_size(type_):
    m = _TYPE_SPLIT_RE.match(type_)
    assert m is not None, 'Invalid NIR type string: "{}"'.format(type_)
    assert m.group('bits') is not None, \
           'NIR type string has no bit size: "{}"'.format(type_)
    return int(m.group('bits'))

def type_sizes(type_):
    if type_has_size(type_):
        return [type_size(type_)]
    elif type_ == 'bool':
        return [1, 8, 16, 32]
    elif type_ == 'float':
        return [16, 32, 64]
    else:
        return [1, 8, 16, 32, 64]

def type_base_type(type_):
    m = _TYPE_SPLIT_RE.match(type_)
    assert m is not None, 'Invalid NIR type string: "{}"'.format(type_)
    return m.group('type')

# Operation where the first two sources are commutative.
#
# For 2-source operations, this just mathematical commutativity.  Some
# 3-source operations, like ffma, are only commutative in the first two
# sources.
_2src_commutative = "2src_commutative "
associative = "associative "
selection = "selection "

# global dictionary of opcodes
opcodes = {}

def opcode(name, output_size, output_type, input_sizes, input_types,
           is_conversion, algebraic_properties, const_expr):
   assert name not in opcodes
   opcodes[name] = Opcode(name, output_size, output_type, input_sizes,
                          input_types, is_conversion, algebraic_properties,
                          const_expr)

def unop_convert(name, out_type, in_type, const_expr):
   opcode(name, 0, out_type, [0], [in_type], False, "", const_expr)

def unop(name, ty, const_expr):
   opcode(name, 0, ty, [0], [ty], False, "", const_expr)

def unop_horiz(name, output_size, output_type, input_size, input_type,
               const_expr):
   opcode(name, output_size, output_type, [input_size], [input_type],
          False, "", const_expr)

def unop_reduce(name, output_size, output_type, input_type, prereduce_expr,
                reduce_expr, final_expr):
   def prereduce(src):
      return "(" + prereduce_expr.format(src=src) + ")"
   def final(src):
      return final_expr.format(src="(" + src + ")")
   def reduce_(src0, src1):
      return reduce_expr.format(src0=src0, src1=src1)
   src0 = prereduce("src0.x")
   src1 = prereduce("src0.y")
   src2 = prereduce("src0.z")
   src3 = prereduce("src0.w")
   unop_horiz(name + "2", output_size, output_type, 2, input_type,
              final(reduce_(src0, src1)))
   unop_horiz(name + "3", output_size, output_type, 3, input_type,
              final(reduce_(reduce_(src0, src1), src2)))
   unop_horiz(name + "4", output_size, output_type, 4, input_type,
              final(reduce_(reduce_(src0, src1), reduce_(src2, src3))))

def unop_numeric_convert(name, out_type, in_type, const_expr):
   opcode(name, 0, out_type, [0], [in_type], True, "", const_expr)

unop("mov", tuint, "src0")

unop("ineg", tint, "-src0")
unop("fneg", tfloat, "-src0")
unop("inot", tint, "~src0") # invert every bit of the integer

# nir_op_fsign roughly implements the OpenGL / Vulkan rules for sign(float).
# The GLSL.std.450 FSign instruction is defined as:
#
#    Result is 1.0 if x > 0, 0.0 if x = 0, or -1.0 if x < 0.
#
# If the source is equal to zero, there is a preference for the result to have
# the same sign, but this is not required (it is required by OpenCL).  If the
# source is not a number, there is a preference for the result to be +0.0, but
# this is not required (it is required by OpenCL).  If the source is not a
# number, and the result is not +0.0, the result should definitely **not** be
# NaN.
#
# The values returned for constant folding match the behavior required by
# OpenCL.
unop("fsign", tfloat, ("bit_size == 64 ? " +
                       "(isnan(src0) ? 0.0  : ((src0 == 0.0 ) ? src0 : (src0 > 0.0 ) ? 1.0  : -1.0 )) : " +
                       "(isnan(src0) ? 0.0f : ((src0 == 0.0f) ? src0 : (src0 > 0.0f) ? 1.0f : -1.0f))"))
unop("isign", tint, "(src0 == 0) ? 0 : ((src0 > 0) ? 1 : -1)")
unop("iabs", tint, "(src0 < 0) ? -src0 : src0")
unop("fabs", tfloat, "fabs(src0)")
unop("fsat", tfloat, ("fmin(fmax(src0, 0.0), 1.0)"))
unop("frcp", tfloat, "bit_size == 64 ? 1.0 / src0 : 1.0f / src0")
unop("frsq", tfloat, "bit_size == 64 ? 1.0 / sqrt(src0) : 1.0f / sqrtf(src0)")
unop("fsqrt", tfloat, "bit_size == 64 ? sqrt(src0) : sqrtf(src0)")
unop("fexp2", tfloat, "exp2f(src0)")
unop("flog2", tfloat, "log2f(src0)")

# Generate all of the numeric conversion opcodes
for src_t in [tint, tuint, tfloat, tbool]:
   if src_t == tbool:
      dst_types = [tfloat, tint, tbool]
   elif src_t == tint:
      dst_types = [tfloat, tint]
   elif src_t == tuint:
      dst_types = [tfloat, tuint]
   elif src_t == tfloat:
      dst_types = [tint, tuint, tfloat, tbool]

   for dst_t in dst_types:
      for dst_bit_size in type_sizes(dst_t):
          if dst_bit_size == 16 and dst_t == tfloat and src_t == tfloat:
              rnd_modes = ['_rtne', '_rtz', '']
              for rnd_mode in rnd_modes:
                  if rnd_mode == '_rtne':
                      conv_expr = """
                      if (bit_size > 16) {
                         dst = _mesa_half_to_float(_mesa_float_to_float16_rtne(src0));
                      } else {
                         dst = src0;
                      }
                      """
                  elif rnd_mode == '_rtz':
                      conv_expr = """
                      if (bit_size > 16) {
                         dst = _mesa_half_to_float(_mesa_float_to_float16_rtz(src0));
                      } else {
                         dst = src0;
                      }
                      """
                  else:
                      conv_expr = "src0"

                  unop_numeric_convert("{0}2{1}{2}{3}".format(src_t[0],
                                                              dst_t[0],
                                                              dst_bit_size,
                                                              rnd_mode),
                                       dst_t + str(dst_bit_size),
                                       src_t, conv_expr)
          elif dst_bit_size == 32 and dst_t == tfloat and src_t == tfloat:
              conv_expr = """
              if (bit_size > 32 && nir_is_rounding_mode_rtz(execution_mode, 32)) {
                 dst = _mesa_double_to_float_rtz(src0);
              } else {
                 dst = src0;
              }
              """
              unop_numeric_convert("{0}2{1}{2}".format(src_t[0], dst_t[0],
                                                       dst_bit_size),
                                   dst_t + str(dst_bit_size), src_t, conv_expr)
          else:
              conv_expr = "src0 != 0" if dst_t == tbool else "src0"
              unop_numeric_convert("{0}2{1}{2}".format(src_t[0], dst_t[0],
                                                       dst_bit_size),
                                   dst_t + str(dst_bit_size), src_t, conv_expr)

# Special opcode that is the same as f2f16, i2i16, u2u16 except that it is safe
# to remove it if the result is immediately converted back to 32 bits again.
# This is generated as part of the precision lowering pass. mp stands for medium
# precision.
unop_numeric_convert("f2fmp", tfloat16, tfloat32, opcodes["f2f16"].const_expr)
unop_numeric_convert("i2imp", tint16, tint32, opcodes["i2i16"].const_expr)
# u2ump isn't defined, because the behavior is equal to i2imp
unop_numeric_convert("f2imp", tint16, tfloat32, opcodes["f2i16"].const_expr)
unop_numeric_convert("f2ump", tuint16, tfloat32, opcodes["f2u16"].const_expr)
unop_numeric_convert("i2fmp", tfloat16, tint32, opcodes["i2f16"].const_expr)
unop_numeric_convert("u2fmp", tfloat16, tuint32, opcodes["u2f16"].const_expr)

# Unary floating-point rounding operations.


unop("ftrunc", tfloat, "bit_size == 64 ? trunc(src0) : truncf(src0)")
unop("fceil", tfloat, "bit_size == 64 ? ceil(src0) : ceilf(src0)")
unop("ffloor", tfloat, "bit_size == 64 ? floor(src0) : floorf(src0)")
unop("ffract", tfloat, "src0 - (bit_size == 64 ? floor(src0) : floorf(src0))")
unop("fround_even", tfloat, "bit_size == 64 ? _mesa_roundeven(src0) : _mesa_roundevenf(src0)")

unop("fquantize2f16", tfloat, "(fabs(src0) < ldexpf(1.0, -14)) ? copysignf(0.0f, src0) : _mesa_half_to_float(_mesa_float_to_half(src0))")

# Trigonometric operations.


unop("fsin", tfloat, "bit_size == 64 ? sin(src0) : sinf(src0)")
unop("fcos", tfloat, "bit_size == 64 ? cos(src0) : cosf(src0)")

# dfrexp
unop_convert("frexp_exp", tint32, tfloat, "frexp(src0, &dst);")
unop_convert("frexp_sig", tfloat, tfloat, "int n; dst = frexp(src0, &n);")

# Partial derivatives.


unop("fddx", tfloat, "0.0") # the derivative of a constant is 0.
unop("fddy", tfloat, "0.0")
unop("fddx_fine", tfloat, "0.0")
unop("fddy_fine", tfloat, "0.0")
unop("fddx_coarse", tfloat, "0.0")
unop("fddy_coarse", tfloat, "0.0")


# Floating point pack and unpack operations.

def pack_2x16(fmt):
   unop_horiz("pack_" + fmt + "_2x16", 1, tuint32, 2, tfloat32, """
dst.x = (uint32_t) pack_fmt_1x16(src0.x);
dst.x |= ((uint32_t) pack_fmt_1x16(src0.y)) << 16;
""".replace("fmt", fmt))

def pack_4x8(fmt):
   unop_horiz("pack_" + fmt + "_4x8", 1, tuint32, 4, tfloat32, """
dst.x = (uint32_t) pack_fmt_1x8(src0.x);
dst.x |= ((uint32_t) pack_fmt_1x8(src0.y)) << 8;
dst.x |= ((uint32_t) pack_fmt_1x8(src0.z)) << 16;
dst.x |= ((uint32_t) pack_fmt_1x8(src0.w)) << 24;
""".replace("fmt", fmt))

def unpack_2x16(fmt):
   unop_horiz("unpack_" + fmt + "_2x16", 2, tfloat32, 1, tuint32, """
dst.x = unpack_fmt_1x16((uint16_t)(src0.x & 0xffff));
dst.y = unpack_fmt_1x16((uint16_t)(src0.x << 16));
""".replace("fmt", fmt))

def unpack_4x8(fmt):
   unop_horiz("unpack_" + fmt + "_4x8", 4, tfloat32, 1, tuint32, """
dst.x = unpack_fmt_1x8((uint8_t)(src0.x & 0xff));
dst.y = unpack_fmt_1x8((uint8_t)((src0.x >> 8) & 0xff));
dst.z = unpack_fmt_1x8((uint8_t)((src0.x >> 16) & 0xff));
dst.w = unpack_fmt_1x8((uint8_t)(src0.x >> 24));
""".replace("fmt", fmt))


pack_2x16("snorm")
pack_4x8("snorm")
pack_2x16("unorm")
pack_4x8("unorm")
pack_2x16("half")
unpack_2x16("snorm")
unpack_4x8("snorm")
unpack_2x16("unorm")
unpack_4x8("unorm")
unpack_2x16("half")

# Convert two unsigned integers into a packed unsigned short (clamp is applied).
unop_horiz("pack_uint_2x16", 1, tuint32, 2, tuint32, """
dst.x = _mesa_unsigned_to_unsigned(src0.x, 16);
dst.x |= _mesa_unsigned_to_unsigned(src0.y, 16) << 16;
""")

# Convert two signed integers into a packed signed short (clamp is applied).
unop_horiz("pack_sint_2x16", 1, tint32, 2, tint32, """
dst.x = _mesa_signed_to_signed(src0.x, 16) & 0xffff;
dst.x |= _mesa_signed_to_signed(src0.y, 16) << 16;
""")

unop_horiz("pack_uvec2_to_uint", 1, tuint32, 2, tuint32, """
dst.x = (src0.x & 0xffff) | (src0.y << 16);
""")

unop_horiz("pack_uvec4_to_uint", 1, tuint32, 4, tuint32, """
dst.x = (src0.x <<  0) |
        (src0.y <<  8) |
        (src0.z << 16) |
        (src0.w << 24);
""")

unop_horiz("pack_32_4x8", 1, tuint32, 4, tuint8,
           "dst.x = src0.x | ((uint32_t)src0.y << 8) | ((uint32_t)src0.z << 16) | ((uint32_t)src0.w << 24);")

unop_horiz("pack_32_2x16", 1, tuint32, 2, tuint16,
           "dst.x = src0.x | ((uint32_t)src0.y << 16);")

unop_horiz("pack_64_2x32", 1, tuint64, 2, tuint32,
           "dst.x = src0.x | ((uint64_t)src0.y << 32);")

unop_horiz("pack_64_4x16", 1, tuint64, 4, tuint16,
           "dst.x = src0.x | ((uint64_t)src0.y << 16) | ((uint64_t)src0.z << 32) | ((uint64_t)src0.w << 48);")

unop_horiz("unpack_64_2x32", 2, tuint32, 1, tuint64,
           "dst.x = src0.x; dst.y = src0.x >> 32;")

unop_horiz("unpack_64_4x16", 4, tuint16, 1, tuint64,
           "dst.x = src0.x; dst.y = src0.x >> 16; dst.z = src0.x >> 32; dst.w = src0.w >> 48;")

unop_horiz("unpack_32_2x16", 2, tuint16, 1, tuint32,
           "dst.x = src0.x; dst.y = src0.x >> 16;")

unop_horiz("unpack_32_4x8", 4, tuint8, 1, tuint32,
           "dst.x = src0.x; dst.y = src0.x >> 8; dst.z = src0.x >> 16; dst.w = src0.x >> 24;")

unop_horiz("unpack_half_2x16_flush_to_zero", 2, tfloat32, 1, tuint32, """
dst.x = unpack_half_1x16_flush_to_zero((uint16_t)(src0.x & 0xffff));
dst.y = unpack_half_1x16_flush_to_zero((uint16_t)(src0.x << 16));
""")

# Lowered floating point unpacking operations.

unop_convert("unpack_half_2x16_split_x", tfloat32, tuint32,
             "unpack_half_1x16((uint16_t)(src0 & 0xffff))")
unop_convert("unpack_half_2x16_split_y", tfloat32, tuint32,
             "unpack_half_1x16((uint16_t)(src0 >> 16))")

unop_convert("unpack_half_2x16_split_x_flush_to_zero", tfloat32, tuint32,
             "unpack_half_1x16_flush_to_zero((uint16_t)(src0 & 0xffff))")
unop_convert("unpack_half_2x16_split_y_flush_to_zero", tfloat32, tuint32,
             "unpack_half_1x16_flush_to_zero((uint16_t)(src0 >> 16))")

unop_convert("unpack_32_2x16_split_x", tuint16, tuint32, "src0")
unop_convert("unpack_32_2x16_split_y", tuint16, tuint32, "src0 >> 16")

unop_convert("unpack_64_2x32_split_x", tuint32, tuint64, "src0")
unop_convert("unpack_64_2x32_split_y", tuint32, tuint64, "src0 >> 32")

# Bit operations, part of ARB_gpu_shader5.


unop("bitfield_reverse", tuint32, """
/* we're not winning any awards for speed here, but that's ok */
dst = 0;
for (unsigned bit = 0; bit < 32; bit++)
   dst |= ((src0 >> bit) & 1) << (31 - bit);
""")
unop_convert("bit_count", tuint32, tuint, """
dst = 0;
for (unsigned bit = 0; bit < bit_size; bit++) {
   if ((src0 >> bit) & 1)
      dst++;
}
""")

unop_convert("ufind_msb", tint32, tuint, """
dst = -1;
for (int bit = bit_size - 1; bit >= 0; bit--) {
   if ((src0 >> bit) & 1) {
      dst = bit;
      break;
   }
}
""")

unop_convert("ufind_msb_rev", tint32, tuint, """
dst = -1;
for (int bit = 0; bit < bit_size; bit++) {
   if ((src0 << bit) & 0x80000000) {
      dst = bit;
      break;
   }
}
""")

unop("uclz", tuint32, """
int bit;
for (bit = bit_size - 1; bit >= 0; bit--) {
   if ((src0 & (1u << bit)) != 0)
      break;
}
dst = (unsigned)(bit_size - bit - 1);
""")

unop("ifind_msb", tint32, """
dst = -1;
for (int bit = bit_size - 1; bit >= 0; bit--) {
   /* If src0 < 0, we're looking for the first 0 bit.
    * if src0 >= 0, we're looking for the first 1 bit.
    */
   if ((((src0 >> bit) & 1) && (src0 >= 0)) ||
      (!((src0 >> bit) & 1) && (src0 < 0))) {
      dst = bit;
      break;
   }
}
""")

unop_convert("ifind_msb_rev", tint32, tint, """
dst = -1;
/* We are looking for the highest bit that's not the same as the sign bit. */
uint32_t sign = src0 & 0x80000000u;
for (int bit = 0; bit < 32; bit++) {
   if (((src0 << bit) & 0x80000000u) != sign) {
      dst = bit;
      break;
   }
}
""")

unop_convert("find_lsb", tint32, tint, """
dst = -1;
for (unsigned bit = 0; bit < bit_size; bit++) {
   if ((src0 >> bit) & 1) {
      dst = bit;
      break;
   }
}
""")

# AMD_gcn_shader extended instructions
unop_horiz("cube_face_coord_amd", 2, tfloat32, 3, tfloat32, """
dst.x = dst.y = 0.0;
float absX = fabsf(src0.x);
float absY = fabsf(src0.y);
float absZ = fabsf(src0.z);

float ma = 0.0;
if (absX >= absY && absX >= absZ) { ma = 2 * src0.x; }
if (absY >= absX && absY >= absZ) { ma = 2 * src0.y; }
if (absZ >= absX && absZ >= absY) { ma = 2 * src0.z; }

if (src0.x >= 0 && absX >= absY && absX >= absZ) { dst.x = -src0.z; dst.y = -src0.y; }
if (src0.x < 0 && absX >= absY && absX >= absZ) { dst.x = src0.z; dst.y = -src0.y; }
if (src0.y >= 0 && absY >= absX && absY >= absZ) { dst.x = src0.x; dst.y = src0.z; }
if (src0.y < 0 && absY >= absX && absY >= absZ) { dst.x = src0.x; dst.y = -src0.z; }
if (src0.z >= 0 && absZ >= absX && absZ >= absY) { dst.x = src0.x; dst.y = -src0.y; }
if (src0.z < 0 && absZ >= absX && absZ >= absY) { dst.x = -src0.x; dst.y = -src0.y; }

dst.x = dst.x * (1.0f / ma) + 0.5f;
dst.y = dst.y * (1.0f / ma) + 0.5f;
""")

unop_horiz("cube_face_index_amd", 1, tfloat32, 3, tfloat32, """
dst.x = 0.0;
float absX = fabsf(src0.x);
float absY = fabsf(src0.y);
float absZ = fabsf(src0.z);
if (src0.x >= 0 && absX >= absY && absX >= absZ) dst.x = 0;
if (src0.x < 0 && absX >= absY && absX >= absZ) dst.x = 1;
if (src0.y >= 0 && absY >= absX && absY >= absZ) dst.x = 2;
if (src0.y < 0 && absY >= absX && absY >= absZ) dst.x = 3;
if (src0.z >= 0 && absZ >= absX && absZ >= absY) dst.x = 4;
if (src0.z < 0 && absZ >= absX && absZ >= absY) dst.x = 5;
""")

# Sum of vector components
unop_reduce("fsum", 1, tfloat, tfloat, "{src}", "{src0} + {src1}", "{src}")

def binop_convert(name, out_type, in_type, alg_props, const_expr):
   opcode(name, 0, out_type, [0, 0], [in_type, in_type],
          False, alg_props, const_expr)

def binop(name, ty, alg_props, const_expr):
   binop_convert(name, ty, ty, alg_props, const_expr)

def binop_compare(name, ty, alg_props, const_expr):
   binop_convert(name, tbool1, ty, alg_props, const_expr)

def binop_compare8(name, ty, alg_props, const_expr):
   binop_convert(name, tbool8, ty, alg_props, const_expr)

def binop_compare16(name, ty, alg_props, const_expr):
   binop_convert(name, tbool16, ty, alg_props, const_expr)

def binop_compare32(name, ty, alg_props, const_expr):
   binop_convert(name, tbool32, ty, alg_props, const_expr)

def binop_compare_all_sizes(name, ty, alg_props, const_expr):
   binop_compare(name, ty, alg_props, const_expr)
   binop_compare8(name + "8", ty, alg_props, const_expr)
   binop_compare16(name + "16", ty, alg_props, const_expr)
   binop_compare32(name + "32", ty, alg_props, const_expr)

def binop_horiz(name, out_size, out_type, src1_size, src1_type, src2_size,
                src2_type, const_expr):
   opcode(name, out_size, out_type, [src1_size, src2_size], [src1_type, src2_type],
          False, "", const_expr)

def binop_reduce(name, output_size, output_type, src_type, prereduce_expr,
                 reduce_expr, final_expr, suffix=""):
   def final(src):
      return final_expr.format(src= "(" + src + ")")
   def reduce_(src0, src1):
      return reduce_expr.format(src0=src0, src1=src1)
   def prereduce(src0, src1):
      return "(" + prereduce_expr.format(src0=src0, src1=src1) + ")"
   srcs = [prereduce("src0." + letter, "src1." + letter) for letter in "xyzwefghijklmnop"]
   def pairwise_reduce(start, size):
      if (size == 1):
         return srcs[start]
      return reduce_(pairwise_reduce(start + size // 2, size // 2), pairwise_reduce(start, size // 2))
   for size in [2, 4, 8, 16]:
      opcode(name + str(size) + suffix, output_size, output_type,
             [size, size], [src_type, src_type], False, _2src_commutative,
             final(pairwise_reduce(0, size)))
   opcode(name + "3" + suffix, output_size, output_type,
          [3, 3], [src_type, src_type], False, _2src_commutative,
          final(reduce_(reduce_(srcs[2], srcs[1]), srcs[0])))
   opcode(name + "5" + suffix, output_size, output_type,
          [5, 5], [src_type, src_type], False, _2src_commutative,
          final(reduce_(srcs[4], reduce_(reduce_(srcs[3], srcs[2]), reduce_(srcs[1], srcs[0])))))

def binop_reduce_all_sizes(name, output_size, src_type, prereduce_expr,
                           reduce_expr, final_expr):
   binop_reduce(name, output_size, tbool1, src_type,
                prereduce_expr, reduce_expr, final_expr)
   binop_reduce("b8" + name[1:], output_size, tbool8, src_type,
                prereduce_expr, reduce_expr, final_expr)
   binop_reduce("b16" + name[1:], output_size, tbool16, src_type,
                prereduce_expr, reduce_expr, final_expr)
   binop_reduce("b32" + name[1:], output_size, tbool32, src_type,
                prereduce_expr, reduce_expr, final_expr)

binop("fadd", tfloat, _2src_commutative + associative,"""
if (nir_is_rounding_mode_rtz(execution_mode, bit_size)) {
   if (bit_size == 64)
      dst = _mesa_double_add_rtz(src0, src1);
   else
      dst = _mesa_double_to_float_rtz((double)src0 + (double)src1);
} else {
   dst = src0 + src1;
}
""")
binop("iadd", tint, _2src_commutative + associative, "(uint64_t)src0 + (uint64_t)src1")
binop("iadd_sat", tint, _2src_commutative, """
      src1 > 0 ?
         (src0 + src1 < src0 ? u_intN_max(bit_size) : src0 + src1) :
         (src0 < src0 + src1 ? u_intN_min(bit_size) : src0 + src1)
""")
binop("uadd_sat", tuint, _2src_commutative,
      "(src0 + src1) < src0 ? u_uintN_max(sizeof(src0) * 8) : (src0 + src1)")
binop("isub_sat", tint, "", """
      src1 < 0 ?
         (src0 - src1 < src0 ? u_intN_max(bit_size) : src0 - src1) :
         (src0 < src0 - src1 ? u_intN_min(bit_size) : src0 - src1)
""")
binop("usub_sat", tuint, "", "src0 < src1 ? 0 : src0 - src1")

binop("fsub", tfloat, "", """
if (nir_is_rounding_mode_rtz(execution_mode, bit_size)) {
   if (bit_size == 64)
      dst = _mesa_double_sub_rtz(src0, src1);
   else
      dst = _mesa_double_to_float_rtz((double)src0 - (double)src1);
} else {
   dst = src0 - src1;
}
""")
binop("isub", tint, "", "src0 - src1")
binop_convert("uabs_isub", tuint, tint, "", """
              src1 > src0 ? (uint64_t) src1 - (uint64_t) src0
                          : (uint64_t) src0 - (uint64_t) src1
""")
binop("uabs_usub", tuint, "", "(src1 > src0) ? (src1 - src0) : (src0 - src1)")

binop("fmul", tfloat, _2src_commutative + associative, """
if (nir_is_rounding_mode_rtz(execution_mode, bit_size)) {
   if (bit_size == 64)
      dst = _mesa_double_mul_rtz(src0, src1);
   else
      dst = _mesa_double_to_float_rtz((double)src0 * (double)src1);
} else {
   dst = src0 * src1;
}
""")

# Unlike fmul, anything (even infinity or NaN) multiplied by zero is always zero.
# fmulz(0.0, inf) and fmulz(0.0, nan) must be +/-0.0, even if
# SIGNED_ZERO_INF_NAN_PRESERVE is not used. If SIGNED_ZERO_INF_NAN_PRESERVE is used, then
# the result must be a positive zero if either operand is zero.
binop("fmulz", tfloat32, _2src_commutative + associative, """
if (src0 == 0.0 || src1 == 0.0)
   dst = 0.0;
else if (nir_is_rounding_mode_rtz(execution_mode, 32))
   dst = _mesa_double_to_float_rtz((double)src0 * (double)src1);
else
   dst = src0 * src1;
""")

# low 32-bits of signed/unsigned integer multiply
binop("imul", tint, _2src_commutative + associative, """
   /* Use 64-bit multiplies to prevent overflow of signed arithmetic */
   dst = (uint64_t)src0 * (uint64_t)src1;
""")

# Generate 64 bit result from 2 32 bits quantity
binop_convert("imul_2x32_64", tint64, tint32, _2src_commutative,
              "(int64_t)src0 * (int64_t)src1")
binop_convert("umul_2x32_64", tuint64, tuint32, _2src_commutative,
              "(uint64_t)src0 * (uint64_t)src1")

# high 32-bits of signed integer multiply
binop("imul_high", tint, _2src_commutative, """
if (bit_size == 64) {
   /* We need to do a full 128-bit x 128-bit multiply in order for the sign
    * extension to work properly.  The casts are kind-of annoying but needed
    * to prevent compiler warnings.
    */
   uint32_t src0_u32[4] = {
      src0,
      (int64_t)src0 >> 32,
      (int64_t)src0 >> 63,
      (int64_t)src0 >> 63,
   };
   uint32_t src1_u32[4] = {
      src1,
      (int64_t)src1 >> 32,
      (int64_t)src1 >> 63,
      (int64_t)src1 >> 63,
   };
   uint32_t prod_u32[4];
   ubm_mul_u32arr(prod_u32, src0_u32, src1_u32);
   dst = (uint64_t)prod_u32[2] | ((uint64_t)prod_u32[3] << 32);
} else {
   /* First, sign-extend to 64-bit, then convert to unsigned to prevent
    * potential overflow of signed multiply */
   dst = ((uint64_t)(int64_t)src0 * (uint64_t)(int64_t)src1) >> bit_size;
}
""")

# high 32-bits of unsigned integer multiply
binop("umul_high", tuint, _2src_commutative, """
if (bit_size == 64) {
   /* The casts are kind-of annoying but needed to prevent compiler warnings. */
   uint32_t src0_u32[2] = { src0, (uint64_t)src0 >> 32 };
   uint32_t src1_u32[2] = { src1, (uint64_t)src1 >> 32 };
   uint32_t prod_u32[4];
   ubm_mul_u32arr(prod_u32, src0_u32, src1_u32);
   dst = (uint64_t)prod_u32[2] | ((uint64_t)prod_u32[3] << 32);
} else {
   dst = ((uint64_t)src0 * (uint64_t)src1) >> bit_size;
}
""")

# low 32-bits of unsigned integer multiply
binop("umul_low", tuint32, _2src_commutative, """
uint64_t mask = (1 << (bit_size / 2)) - 1;
dst = ((uint64_t)src0 & mask) * ((uint64_t)src1 & mask);
""")

# Multiply 32-bits with low 16-bits.
binop("imul_32x16", tint32, "", "src0 * (int16_t) src1")
binop("umul_32x16", tuint32, "", "src0 * (uint16_t) src1")

binop("fdiv", tfloat, "", "src0 / src1")
binop("idiv", tint, "", "src1 == 0 ? 0 : (src0 / src1)")
binop("udiv", tuint, "", "src1 == 0 ? 0 : (src0 / src1)")

# returns an integer (1 or 0) representing the carry resulting from the
# addition of the two unsigned arguments.

binop_convert("uadd_carry", tuint, tuint, _2src_commutative, "src0 + src1 < src0")

# returns an integer (1 or 0) representing the borrow resulting from the
# subtraction of the two unsigned arguments.

binop_convert("usub_borrow", tuint, tuint, "", "src0 < src1")

# hadd: (a + b) >> 1 (without overflow)
# x + y = x - (x & ~y) + (x & ~y) + y - (~x & y) + (~x & y)
#       =      (x & y) + (x & ~y) +      (x & y) + (~x & y)
#       = 2 *  (x & y) + (x & ~y) +                (~x & y)
#       =     ((x & y) << 1) + (x ^ y)
#
# Since we know that the bottom bit of (x & y) << 1 is zero,
#
# (x + y) >> 1 = (((x & y) << 1) + (x ^ y)) >> 1
#              =   (x & y) +      ((x ^ y)  >> 1)
binop("ihadd", tint, _2src_commutative, "(src0 & src1) + ((src0 ^ src1) >> 1)")
binop("uhadd", tuint, _2src_commutative, "(src0 & src1) + ((src0 ^ src1) >> 1)")

# rhadd: (a + b + 1) >> 1 (without overflow)
# x + y + 1 = x + (~x & y) - (~x & y) + y + (x & ~y) - (x & ~y) + 1
#           =      (x | y) - (~x & y) +      (x | y) - (x & ~y) + 1
#           = 2 *  (x | y) - ((~x & y) +               (x & ~y)) + 1
#           =     ((x | y) << 1) - (x ^ y) + 1
#
# Since we know that the bottom bit of (x & y) << 1 is zero,
#
# (x + y + 1) >> 1 = (x | y) + (-(x ^ y) + 1) >> 1)
#                  = (x | y) -  ((x ^ y)      >> 1)
binop("irhadd", tint, _2src_commutative, "(src0 | src1) - ((src0 ^ src1) >> 1)")
binop("urhadd", tuint, _2src_commutative, "(src0 | src1) - ((src0 ^ src1) >> 1)")

binop("umod", tuint, "", "src1 == 0 ? 0 : src0 % src1")

# For signed integers, there are several different possible definitions of
# "modulus" or "remainder".  We follow the conventions used by LLVM and
# SPIR-V.  The irem opcode implements the standard C/C++ signed "%"
# operation while the imod opcode implements the more mathematical
# "modulus" operation.  For details on the difference, see
#
# http://mathforum.org/library/drmath/view/52343.html

binop("irem", tint, "", "src1 == 0 ? 0 : src0 % src1")
binop("imod", tint, "",
      "src1 == 0 ? 0 : ((src0 % src1 == 0 || (src0 >= 0) == (src1 >= 0)) ?"
      "                 src0 % src1 : src0 % src1 + src1)")
binop("fmod", tfloat, "", "src0 - src1 * floorf(src0 / src1)")
binop("frem", tfloat, "", "src0 - src1 * truncf(src0 / src1)")

#
# Comparisons
#


# these integer-aware comparisons return a boolean (0 or ~0)

binop_compare_all_sizes("flt", tfloat, "", "src0 < src1")
binop_compare_all_sizes("fge", tfloat, "", "src0 >= src1")
binop_compare_all_sizes("feq", tfloat, _2src_commutative, "src0 == src1")
binop_compare_all_sizes("fneu", tfloat, _2src_commutative, "src0 != src1")
binop_compare_all_sizes("ilt", tint, "", "src0 < src1")
binop_compare_all_sizes("ige", tint, "", "src0 >= src1")
binop_compare_all_sizes("ieq", tint, _2src_commutative, "src0 == src1")
binop_compare_all_sizes("ine", tint, _2src_commutative, "src0 != src1")
binop_compare_all_sizes("ult", tuint, "", "src0 < src1")
binop_compare_all_sizes("uge", tuint, "", "src0 >= src1")

# integer-aware GLSL-style comparisons that compare floats and ints

binop_reduce_all_sizes("ball_fequal",  1, tfloat, "{src0} == {src1}",
                       "{src0} && {src1}", "{src}")
binop_reduce_all_sizes("bany_fnequal", 1, tfloat, "{src0} != {src1}",
                       "{src0} || {src1}", "{src}")
binop_reduce_all_sizes("ball_iequal",  1, tint, "{src0} == {src1}",
                       "{src0} && {src1}", "{src}")
binop_reduce_all_sizes("bany_inequal", 1, tint, "{src0} != {src1}",
                       "{src0} || {src1}", "{src}")

# non-integer-aware GLSL-style comparisons that return 0.0 or 1.0

binop_reduce("fall_equal",  1, tfloat32, tfloat32, "{src0} == {src1}",
             "{src0} && {src1}", "{src} ? 1.0f : 0.0f")
binop_reduce("fany_nequal", 1, tfloat32, tfloat32, "{src0} != {src1}",
             "{src0} || {src1}", "{src} ? 1.0f : 0.0f")

# These comparisons for integer-less hardware return 1.0 and 0.0 for true
# and false respectively

binop("slt", tfloat, "", "(src0 < src1) ? 1.0f : 0.0f") # Set on Less Than
binop("sge", tfloat, "", "(src0 >= src1) ? 1.0f : 0.0f") # Set on Greater or Equal
binop("seq", tfloat, _2src_commutative, "(src0 == src1) ? 1.0f : 0.0f") # Set on Equal
binop("sne", tfloat, _2src_commutative, "(src0 != src1) ? 1.0f : 0.0f") # Set on Not Equal

# SPIRV shifts are undefined for shift-operands >= bitsize,
# but SM5 shifts are defined to use only the least significant bits.
# The NIR definition is according to the SM5 specification.
opcode("ishl", 0, tint, [0, 0], [tint, tuint32], False, "",
       "(uint64_t)src0 << (src1 & (sizeof(src0) * 8 - 1))")
opcode("ishr", 0, tint, [0, 0], [tint, tuint32], False, "",
       "src0 >> (src1 & (sizeof(src0) * 8 - 1))")
opcode("ushr", 0, tuint, [0, 0], [tuint, tuint32], False, "",
       "src0 >> (src1 & (sizeof(src0) * 8 - 1))")

opcode("urol", 0, tuint, [0, 0], [tuint, tuint32], False, "", """
   uint32_t rotate_mask = sizeof(src0) * 8 - 1;
   dst = (src0 << (src1 & rotate_mask)) |
         (src0 >> (-src1 & rotate_mask));
""")
opcode("uror", 0, tuint, [0, 0], [tuint, tuint32], False, "", """
   uint32_t rotate_mask = sizeof(src0) * 8 - 1;
   dst = (src0 >> (src1 & rotate_mask)) |
         (src0 << (-src1 & rotate_mask));
""")

# bitwise logic operators
#
# These are also used as boolean and, or, xor for hardware supporting
# integers.


binop("iand", tuint, _2src_commutative + associative, "src0 & src1")
binop("ior", tuint, _2src_commutative + associative, "src0 | src1")
binop("ixor", tuint, _2src_commutative + associative, "src0 ^ src1")


binop_reduce("fdot", 1, tfloat, tfloat, "{src0} * {src1}", "{src0} + {src1}",
             "{src}")

binop_reduce("fdot", 0, tfloat, tfloat,
             "{src0} * {src1}", "{src0} + {src1}", "{src}",
             suffix="_replicated")

opcode("fdph", 1, tfloat, [3, 4], [tfloat, tfloat], False, "",
       "src0.x * src1.x + src0.y * src1.y + src0.z * src1.z + src1.w")
opcode("fdph_replicated", 0, tfloat, [3, 4], [tfloat, tfloat], False, "",
       "src0.x * src1.x + src0.y * src1.y + src0.z * src1.z + src1.w")

binop("fmin", tfloat, _2src_commutative + associative, "fmin(src0, src1)")
binop("imin", tint, _2src_commutative + associative, "src1 > src0 ? src0 : src1")
binop("umin", tuint, _2src_commutative + associative, "src1 > src0 ? src0 : src1")
binop("fmax", tfloat, _2src_commutative + associative, "fmax(src0, src1)")
binop("imax", tint, _2src_commutative + associative, "src1 > src0 ? src1 : src0")
binop("umax", tuint, _2src_commutative + associative, "src1 > src0 ? src1 : src0")

binop("fpow", tfloat, "", "bit_size == 64 ? powf(src0, src1) : pow(src0, src1)")

binop_horiz("pack_half_2x16_split", 1, tuint32, 1, tfloat32, 1, tfloat32,
            "pack_half_1x16(src0.x) | (pack_half_1x16(src1.x) << 16)")

binop_convert("pack_64_2x32_split", tuint64, tuint32, "",
              "src0 | ((uint64_t)src1 << 32)")

binop_convert("pack_32_2x16_split", tuint32, tuint16, "",
              "src0 | ((uint32_t)src1 << 16)")

opcode("pack_32_4x8_split", 0, tuint32, [0, 0, 0, 0], [tuint8, tuint8, tuint8, tuint8],
       False, "",
       "src0 | ((uint32_t)src1 << 8) | ((uint32_t)src2 << 16) | ((uint32_t)src3 << 24)")

# bfm implements the behavior of the first operation of the SM5 "bfi" assembly
# and that of the "bfi1" i965 instruction. That is, the bits and offset values
# are from the low five bits of src0 and src1, respectively.
binop_convert("bfm", tuint32, tint32, "", """
int bits = src0 & 0x1F;
int offset = src1 & 0x1F;
dst = ((1u << bits) - 1) << offset;
""")

opcode("ldexp", 0, tfloat, [0, 0], [tfloat, tint32], False, "", """
dst = (bit_size == 64) ? ldexp(src0, src1) : ldexpf(src0, src1);
/* flush denormals to zero. */
if (!isnormal(dst))
   dst = copysignf(0.0f, src0);
""")

# Combines the first component of each input to make a 2-component vector.

binop_horiz("vec2", 2, tuint, 1, tuint, 1, tuint, """
dst.x = src0.x;
dst.y = src1.x;
""")

# Byte extraction
binop("extract_u8", tuint, "", "(uint8_t)(src0 >> (src1 * 8))")
binop("extract_i8", tint, "", "(int8_t)(src0 >> (src1 * 8))")

# Word extraction
binop("extract_u16", tuint, "", "(uint16_t)(src0 >> (src1 * 16))")
binop("extract_i16", tint, "", "(int16_t)(src0 >> (src1 * 16))")

# Byte/word insertion
binop("insert_u8", tuint, "", "(src0 & 0xff) << (src1 * 8)")
binop("insert_u16", tuint, "", "(src0 & 0xffff) << (src1 * 16)")


def triop(name, ty, alg_props, const_expr):
   opcode(name, 0, ty, [0, 0, 0], [ty, ty, ty], False, alg_props, const_expr)
def triop_horiz(name, output_size, src1_size, src2_size, src3_size, const_expr):
   opcode(name, output_size, tuint,
   [src1_size, src2_size, src3_size],
   [tuint, tuint, tuint], False, "", const_expr)

triop("ffma", tfloat, _2src_commutative, """
if (nir_is_rounding_mode_rtz(execution_mode, bit_size)) {
   if (bit_size == 64)
      dst = _mesa_double_fma_rtz(src0, src1, src2);
   else if (bit_size == 32)
      dst = _mesa_float_fma_rtz(src0, src1, src2);
   else
      dst = _mesa_double_to_float_rtz(_mesa_double_fma_rtz(src0, src1, src2));
} else {
   if (bit_size == 32)
      dst = fmaf(src0, src1, src2);
   else
      dst = fma(src0, src1, src2);
}
""")

# Unlike ffma, anything (even infinity or NaN) multiplied by zero is always zero.
# ffmaz(0.0, inf, src2) and ffmaz(0.0, nan, src2) must be +/-0.0 + src2, even if
# SIGNED_ZERO_INF_NAN_PRESERVE is not used. If SIGNED_ZERO_INF_NAN_PRESERVE is used, then
# the result must be a positive zero plus src2 if either src0 or src1 is zero.
triop("ffmaz", tfloat32, _2src_commutative, """
if (src0 == 0.0 || src1 == 0.0)
   dst = 0.0 + src2;
else if (nir_is_rounding_mode_rtz(execution_mode, 32))
   dst = _mesa_float_fma_rtz(src0, src1, src2);
else
   dst = fmaf(src0, src1, src2);
""")

triop("flrp", tfloat, "", "src0 * (1 - src2) + src1 * src2")

# Ternary addition
triop("iadd3", tint, _2src_commutative + associative, "src0 + src1 + src2")

# Conditional Select
#
# A vector conditional select instruction (like ?:, but operating per-
# component on vectors). There are two versions, one for floating point
# bools (0.0 vs 1.0) and one for integer bools (0 vs ~0).

triop("fcsel", tfloat32, selection, "(src0 != 0.0f) ? src1 : src2")

opcode("bcsel", 0, tuint, [0, 0, 0],
       [tbool1, tuint, tuint], False, selection, "src0 ? src1 : src2")
opcode("b8csel", 0, tuint, [0, 0, 0],
       [tbool8, tuint, tuint], False, selection, "src0 ? src1 : src2")
opcode("b16csel", 0, tuint, [0, 0, 0],
       [tbool16, tuint, tuint], False, selection, "src0 ? src1 : src2")
opcode("b32csel", 0, tuint, [0, 0, 0],
       [tbool32, tuint, tuint], False, selection, "src0 ? src1 : src2")

triop("i32csel_gt", tint32, selection, "(src0 > 0) ? src1 : src2")
triop("i32csel_ge", tint32, selection, "(src0 >= 0) ? src1 : src2")

triop("fcsel_gt", tfloat32, selection, "(src0 > 0.0f) ? src1 : src2")
triop("fcsel_ge", tfloat32, selection, "(src0 >= 0.0f) ? src1 : src2")

# SM5 bfi assembly
triop("bfi", tuint32, "", """
unsigned mask = src0, insert = src1, base = src2;
if (mask == 0) {
   dst = base;
} else {
   unsigned tmp = mask;
   while (!(tmp & 1)) {
      tmp >>= 1;
      insert <<= 1;
   }
   dst = (base & ~mask) | (insert & mask);
}
""")


triop("bitfield_select", tuint, "", "(src0 & src1) | (~src0 & src2)")

# SM5 ubfe/ibfe assembly: only the 5 least significant bits of offset and bits are used.
opcode("ubfe", 0, tuint32,
       [0, 0, 0], [tuint32, tuint32, tuint32], False, "", """
unsigned base = src0;
unsigned offset = src1 & 0x1F;
unsigned bits = src2 & 0x1F;
if (bits == 0) {
   dst = 0;
} else if (offset + bits < 32) {
   dst = (base << (32 - bits - offset)) >> (32 - bits);
} else {
   dst = base >> offset;
}
""")
opcode("ibfe", 0, tint32,
       [0, 0, 0], [tint32, tuint32, tuint32], False, "", """
int base = src0;
unsigned offset = src1 & 0x1F;
unsigned bits = src2 & 0x1F;
if (bits == 0) {
   dst = 0;
} else if (offset + bits < 32) {
   dst = (base << (32 - bits - offset)) >> (32 - bits);
} else {
   dst = base >> offset;
}
""")

# GLSL bitfieldExtract()
opcode("ubitfield_extract", 0, tuint32,
       [0, 0, 0], [tuint32, tint32, tint32], False, "", """
unsigned base = src0;
int offset = src1, bits = src2;
if (bits == 0) {
   dst = 0;
} else if (bits < 0 || offset < 0 || offset + bits > 32) {
   dst = 0; /* undefined per the spec */
} else {
   dst = (base >> offset) & ((1ull << bits) - 1);
}
""")
opcode("ibitfield_extract", 0, tint32,
       [0, 0, 0], [tint32, tint32, tint32], False, "", """
int base = src0;
int offset = src1, bits = src2;
if (bits == 0) {
   dst = 0;
} else if (offset < 0 || bits < 0 || offset + bits > 32) {
   dst = 0;
} else {
   dst = (base << (32 - offset - bits)) >> (32 - bits); /* use sign-extending shift */
}
""")

# Sum of absolute differences with accumulation.
# (Equivalent to AMD's v_sad_u8 instruction.)
# The first two sources contain packed 8-bit unsigned integers, the instruction
# will calculate the absolute difference of these, and then add them together.
# There is also a third source which is a 32-bit unsigned integer and added to the result.
triop_horiz("sad_u8x4", 1, 1, 1, 1, """
uint8_t s0_b0 = (src0.x & 0x000000ff) >> 0;
uint8_t s0_b1 = (src0.x & 0x0000ff00) >> 8;
uint8_t s0_b2 = (src0.x & 0x00ff0000) >> 16;
uint8_t s0_b3 = (src0.x & 0xff000000) >> 24;

uint8_t s1_b0 = (src1.x & 0x000000ff) >> 0;
uint8_t s1_b1 = (src1.x & 0x0000ff00) >> 8;
uint8_t s1_b2 = (src1.x & 0x00ff0000) >> 16;
uint8_t s1_b3 = (src1.x & 0xff000000) >> 24;

dst.x = src2.x +
        (s0_b0 > s1_b0 ? (s0_b0 - s1_b0) : (s1_b0 - s0_b0)) +
        (s0_b1 > s1_b1 ? (s0_b1 - s1_b1) : (s1_b1 - s0_b1)) +
        (s0_b2 > s1_b2 ? (s0_b2 - s1_b2) : (s1_b2 - s0_b2)) +
        (s0_b3 > s1_b3 ? (s0_b3 - s1_b3) : (s1_b3 - s0_b3));
""")

# Combines the first component of each input to make a 3-component vector.

triop_horiz("vec3", 3, 1, 1, 1, """
dst.x = src0.x;
dst.y = src1.x;
dst.z = src2.x;
""")

def quadop_horiz(name, output_size, src1_size, src2_size, src3_size,
                 src4_size, const_expr):
   opcode(name, output_size, tuint,
          [src1_size, src2_size, src3_size, src4_size],
          [tuint, tuint, tuint, tuint],
          False, "", const_expr)

opcode("bitfield_insert", 0, tuint32, [0, 0, 0, 0],
       [tuint32, tuint32, tint32, tint32], False, "", """
unsigned base = src0, insert = src1;
int offset = src2, bits = src3;
if (bits == 0) {
   dst = base;
} else if (offset < 0 || bits < 0 || bits + offset > 32) {
   dst = 0;
} else {
   unsigned mask = ((1ull << bits) - 1) << offset;
   dst = (base & ~mask) | ((insert << offset) & mask);
}
""")

quadop_horiz("vec4", 4, 1, 1, 1, 1, """
dst.x = src0.x;
dst.y = src1.x;
dst.z = src2.x;
dst.w = src3.x;
""")

opcode("vec5", 5, tuint,
       [1] * 5, [tuint] * 5,
       False, "", """
dst.x = src0.x;
dst.y = src1.x;
dst.z = src2.x;
dst.w = src3.x;
dst.e = src4.x;
""")

opcode("vec8", 8, tuint,
       [1] * 8, [tuint] * 8,
       False, "", """
dst.x = src0.x;
dst.y = src1.x;
dst.z = src2.x;
dst.w = src3.x;
dst.e = src4.x;
dst.f = src5.x;
dst.g = src6.x;
dst.h = src7.x;
""")

opcode("vec16", 16, tuint,
       [1] * 16, [tuint] * 16,
       False, "", """
dst.x = src0.x;
dst.y = src1.x;
dst.z = src2.x;
dst.w = src3.x;
dst.e = src4.x;
dst.f = src5.x;
dst.g = src6.x;
dst.h = src7.x;
dst.i = src8.x;
dst.j = src9.x;
dst.k = src10.x;
dst.l = src11.x;
dst.m = src12.x;
dst.n = src13.x;
dst.o = src14.x;
dst.p = src15.x;
""")

# An integer multiply instruction for address calculation.  This is
# similar to imul, except that the results are undefined in case of
# overflow.  Overflow is defined according to the size of the variable
# being dereferenced.
#
# This relaxed definition, compared to imul, allows an optimization
# pass to propagate bounds (ie, from an load/store intrinsic) to the
# sources, such that lower precision integer multiplies can be used.
# This is useful on hw that has 24b or perhaps 16b integer multiply
# instructions.
binop("amul", tint, _2src_commutative + associative, "src0 * src1")

# ir3-specific instruction that maps directly to mul-add shift high mix,
# (IMADSH_MIX16 i.e. ah * bl << 16 + c). It is used for lowering integer
# multiplication (imul) on Freedreno backend..
opcode("imadsh_mix16", 0, tint32,
       [0, 0, 0], [tint32, tint32, tint32], False, "", """
dst = ((((src0 & 0xffff0000) >> 16) * (src1 & 0x0000ffff)) << 16) + src2;
""")

# ir3-specific instruction that maps directly to ir3 mad.s24.
#
# 24b multiply into 32b result (with sign extension) plus 32b int
triop("imad24_ir3", tint32, _2src_commutative,
      "(((int32_t)src0 << 8) >> 8) * (((int32_t)src1 << 8) >> 8) + src2")

# r600-specific instruction that evaluates unnormalized cube texture coordinates
# and face index
# The actual texture coordinates are evaluated from this according to
#    dst.yx / abs(dst.z) + 1.5
unop_horiz("cube_r600", 4, tfloat32, 3, tfloat32, """
   dst.x = dst.y = dst.z = 0.0;
   float absX = fabsf(src0.x);
   float absY = fabsf(src0.y);
   float absZ = fabsf(src0.z);

   if (absX >= absY && absX >= absZ) { dst.z = 2 * src0.x; }
   if (absY >= absX && absY >= absZ) { dst.z = 2 * src0.y; }
   if (absZ >= absX && absZ >= absY) { dst.z = 2 * src0.z; }

   if (src0.x >= 0 && absX >= absY && absX >= absZ) {
      dst.y = -src0.z; dst.x = -src0.y; dst.w = 0;
   }
   if (src0.x < 0 && absX >= absY && absX >= absZ) {
      dst.y = src0.z; dst.x = -src0.y; dst.w = 1;
   }
   if (src0.y >= 0 && absY >= absX && absY >= absZ) {
      dst.y = src0.x; dst.x = src0.z; dst.w = 2;
   }
   if (src0.y < 0 && absY >= absX && absY >= absZ) {
      dst.y = src0.x; dst.x = -src0.z; dst.w = 3;
   }
   if (src0.z >= 0 && absZ >= absX && absZ >= absY) {
      dst.y = src0.x; dst.x = -src0.y; dst.w = 4;
   }
   if (src0.z < 0 && absZ >= absX && absZ >= absY) {
      dst.y = -src0.x; dst.x = -src0.y; dst.w = 5;
   }
""")

# r600/gcn specific sin and cos
# these trigeometric functions need some lowering because the supported
# input values are expected to be normalized by dividing by (2 * pi)
unop("fsin_amd", tfloat, "sinf(6.2831853 * src0)")
unop("fcos_amd", tfloat, "cosf(6.2831853 * src0)")

# Midgard specific sin and cos
# These expect their inputs to be divided by pi.
unop("fsin_mdg", tfloat, "sinf(3.141592653589793 * src0)")
unop("fcos_mdg", tfloat, "cosf(3.141592653589793 * src0)")

# AGX specific sin with input expressed in quadrants. Used in the lowering for
# fsin/fcos. This corresponds to a sequence of 3 ALU ops in the backend (where
# the angle is further decomposed by quadrant, sinc is computed, and the angle
# is multiplied back for sin). Lowering fsin/fcos to fsin_agx requires some
# additional ALU that NIR may be able to optimize.
unop("fsin_agx", tfloat, "sinf(src0 * (6.2831853/4.0))")

# 24b multiply into 32b result (with sign extension)
binop("imul24", tint32, _2src_commutative + associative,
      "(((int32_t)src0 << 8) >> 8) * (((int32_t)src1 << 8) >> 8)")

# unsigned 24b multiply into 32b result plus 32b int
triop("umad24", tuint32, _2src_commutative,
      "(((uint32_t)src0 << 8) >> 8) * (((uint32_t)src1 << 8) >> 8) + src2")

# unsigned 24b multiply into 32b result uint
binop("umul24", tint32, _2src_commutative + associative,
      "(((uint32_t)src0 << 8) >> 8) * (((uint32_t)src1 << 8) >> 8)")

# relaxed versions of the above, which assume input is in the 24bit range (no clamping)
binop("imul24_relaxed", tint32, _2src_commutative + associative, "src0 * src1")
triop("umad24_relaxed", tuint32, _2src_commutative, "src0 * src1 + src2")
binop("umul24_relaxed", tuint32, _2src_commutative + associative, "src0 * src1")

unop_convert("fisnormal", tbool1, tfloat, "isnormal(src0)")
unop_convert("fisfinite", tbool1, tfloat, "isfinite(src0)")
unop_convert("fisfinite32", tbool32, tfloat, "isfinite(src0)")

# vc4-specific opcodes

# Saturated vector add for 4 8bit ints.
binop("usadd_4x8_vc4", tint32, _2src_commutative + associative, """
dst = 0;
for (int i = 0; i < 32; i += 8) {
   dst |= MIN2(((src0 >> i) & 0xff) + ((src1 >> i) & 0xff), 0xff) << i;
}
""")

# Saturated vector subtract for 4 8bit ints.
binop("ussub_4x8_vc4", tint32, "", """
dst = 0;
for (int i = 0; i < 32; i += 8) {
   int src0_chan = (src0 >> i) & 0xff;
   int src1_chan = (src1 >> i) & 0xff;
   if (src0_chan > src1_chan)
      dst |= (src0_chan - src1_chan) << i;
}
""")

# vector min for 4 8bit ints.
binop("umin_4x8_vc4", tint32, _2src_commutative + associative, """
dst = 0;
for (int i = 0; i < 32; i += 8) {
   dst |= MIN2((src0 >> i) & 0xff, (src1 >> i) & 0xff) << i;
}
""")

# vector max for 4 8bit ints.
binop("umax_4x8_vc4", tint32, _2src_commutative + associative, """
dst = 0;
for (int i = 0; i < 32; i += 8) {
   dst |= MAX2((src0 >> i) & 0xff, (src1 >> i) & 0xff) << i;
}
""")

# unorm multiply: (a * b) / 255.
binop("umul_unorm_4x8_vc4", tint32, _2src_commutative + associative, """
dst = 0;
for (int i = 0; i < 32; i += 8) {
   int src0_chan = (src0 >> i) & 0xff;
   int src1_chan = (src1 >> i) & 0xff;
   dst |= ((src0_chan * src1_chan) / 255) << i;
}
""")

# Mali-specific opcodes
unop("fsat_signed_mali", tfloat, ("fmin(fmax(src0, -1.0), 1.0)"))
unop("fclamp_pos_mali", tfloat, ("fmax(src0, 0.0)"))

# Magnitude equal to fddx/y, sign undefined. Derivative of a constant is zero.
unop("fddx_must_abs_mali", tfloat, "0.0")
unop("fddy_must_abs_mali", tfloat, "0.0")

# DXIL specific double [un]pack
# DXIL doesn't support generic [un]pack instructions, so we want those
# lowered to bit ops. HLSL doesn't support 64bit bitcasts to/from
# double, only [un]pack. Technically DXIL does, but considering they
# can't be generated from HLSL, we want to match what would be coming from DXC.
# This is essentially just the standard [un]pack, except that it doesn't get
# lowered so we can handle it in the backend and turn it into MakeDouble/SplitDouble
unop_horiz("pack_double_2x32_dxil", 1, tuint64, 2, tuint32,
           "dst.x = src0.x | ((uint64_t)src0.y << 32);")
unop_horiz("unpack_double_2x32_dxil", 2, tuint32, 1, tuint64,
           "dst.x = src0.x; dst.y = src0.x >> 32;")

# src0 and src1 are i8vec4 packed in an int32, and src2 is an int32.  The int8
# components are sign-extended to 32-bits, and a dot-product is performed on
# the resulting vectors.  src2 is added to the result of the dot-product.
opcode("sdot_4x8_iadd", 0, tint32, [0, 0, 0], [tuint32, tuint32, tint32],
       False, _2src_commutative, """
   const int32_t v0x = (int8_t)(src0      );
   const int32_t v0y = (int8_t)(src0 >>  8);
   const int32_t v0z = (int8_t)(src0 >> 16);
   const int32_t v0w = (int8_t)(src0 >> 24);
   const int32_t v1x = (int8_t)(src1      );
   const int32_t v1y = (int8_t)(src1 >>  8);
   const int32_t v1z = (int8_t)(src1 >> 16);
   const int32_t v1w = (int8_t)(src1 >> 24);

   dst = (v0x * v1x) + (v0y * v1y) + (v0z * v1z) + (v0w * v1w) + src2;
""")

# Like sdot_4x8_iadd, but unsigned.
opcode("udot_4x8_uadd", 0, tuint32, [0, 0, 0], [tuint32, tuint32, tuint32],
       False, _2src_commutative, """
   const uint32_t v0x = (uint8_t)(src0      );
   const uint32_t v0y = (uint8_t)(src0 >>  8);
   const uint32_t v0z = (uint8_t)(src0 >> 16);
   const uint32_t v0w = (uint8_t)(src0 >> 24);
   const uint32_t v1x = (uint8_t)(src1      );
   const uint32_t v1y = (uint8_t)(src1 >>  8);
   const uint32_t v1z = (uint8_t)(src1 >> 16);
   const uint32_t v1w = (uint8_t)(src1 >> 24);

   dst = (v0x * v1x) + (v0y * v1y) + (v0z * v1z) + (v0w * v1w) + src2;
""")

# src0 is i8vec4 packed in an int32, src1 is u8vec4 packed in an int32, and
# src2 is an int32.  The 8-bit components are extended to 32-bits, and a
# dot-product is performed on the resulting vectors.  src2 is added to the
# result of the dot-product.
#
# NOTE: Unlike many of the other dp4a opcodes, this mixed signs of source 0
# and source 1 mean that this opcode is not 2-source commutative
opcode("sudot_4x8_iadd", 0, tint32, [0, 0, 0], [tuint32, tuint32, tint32],
       False, "", """
   const int32_t v0x = (int8_t)(src0      );
   const int32_t v0y = (int8_t)(src0 >>  8);
   const int32_t v0z = (int8_t)(src0 >> 16);
   const int32_t v0w = (int8_t)(src0 >> 24);
   const uint32_t v1x = (uint8_t)(src1      );
   const uint32_t v1y = (uint8_t)(src1 >>  8);
   const uint32_t v1z = (uint8_t)(src1 >> 16);
   const uint32_t v1w = (uint8_t)(src1 >> 24);

   dst = (v0x * v1x) + (v0y * v1y) + (v0z * v1z) + (v0w * v1w) + src2;
""")

# Like sdot_4x8_iadd, but the result is clampled to the range [-0x80000000, 0x7ffffffff].
opcode("sdot_4x8_iadd_sat", 0, tint32, [0, 0, 0], [tuint32, tuint32, tint32],
       False, _2src_commutative, """
   const int64_t v0x = (int8_t)(src0      );
   const int64_t v0y = (int8_t)(src0 >>  8);
   const int64_t v0z = (int8_t)(src0 >> 16);
   const int64_t v0w = (int8_t)(src0 >> 24);
   const int64_t v1x = (int8_t)(src1      );
   const int64_t v1y = (int8_t)(src1 >>  8);
   const int64_t v1z = (int8_t)(src1 >> 16);
   const int64_t v1w = (int8_t)(src1 >> 24);

   const int64_t tmp = (v0x * v1x) + (v0y * v1y) + (v0z * v1z) + (v0w * v1w) + src2;

   dst = tmp >= INT32_MAX ? INT32_MAX : (tmp <= INT32_MIN ? INT32_MIN : tmp);
""")

# Like udot_4x8_uadd, but the result is clampled to the range [0, 0xfffffffff].
opcode("udot_4x8_uadd_sat", 0, tint32, [0, 0, 0], [tuint32, tuint32, tint32],
       False, _2src_commutative, """
   const uint64_t v0x = (uint8_t)(src0      );
   const uint64_t v0y = (uint8_t)(src0 >>  8);
   const uint64_t v0z = (uint8_t)(src0 >> 16);
   const uint64_t v0w = (uint8_t)(src0 >> 24);
   const uint64_t v1x = (uint8_t)(src1      );
   const uint64_t v1y = (uint8_t)(src1 >>  8);
   const uint64_t v1z = (uint8_t)(src1 >> 16);
   const uint64_t v1w = (uint8_t)(src1 >> 24);

   const uint64_t tmp = (v0x * v1x) + (v0y * v1y) + (v0z * v1z) + (v0w * v1w) + src2;

   dst = tmp >= UINT32_MAX ? UINT32_MAX : tmp;
""")

# Like sudot_4x8_iadd, but the result is clampled to the range [-0x80000000, 0x7ffffffff].
#
# NOTE: Unlike many of the other dp4a opcodes, this mixed signs of source 0
# and source 1 mean that this opcode is not 2-source commutative
opcode("sudot_4x8_iadd_sat", 0, tint32, [0, 0, 0], [tuint32, tuint32, tint32],
       False, "", """
   const int64_t v0x = (int8_t)(src0      );
   const int64_t v0y = (int8_t)(src0 >>  8);
   const int64_t v0z = (int8_t)(src0 >> 16);
   const int64_t v0w = (int8_t)(src0 >> 24);
   const uint64_t v1x = (uint8_t)(src1      );
   const uint64_t v1y = (uint8_t)(src1 >>  8);
   const uint64_t v1z = (uint8_t)(src1 >> 16);
   const uint64_t v1w = (uint8_t)(src1 >> 24);

   const int64_t tmp = (v0x * v1x) + (v0y * v1y) + (v0z * v1z) + (v0w * v1w) + src2;

   dst = tmp >= INT32_MAX ? INT32_MAX : (tmp <= INT32_MIN ? INT32_MIN : tmp);
""")

# src0 and src1 are i16vec2 packed in an int32, and src2 is an int32.  The int16
# components are sign-extended to 32-bits, and a dot-product is performed on
# the resulting vectors.  src2 is added to the result of the dot-product.
opcode("sdot_2x16_iadd", 0, tint32, [0, 0, 0], [tuint32, tuint32, tint32],
       False, _2src_commutative, """
   const int32_t v0x = (int16_t)(src0      );
   const int32_t v0y = (int16_t)(src0 >> 16);
   const int32_t v1x = (int16_t)(src1      );
   const int32_t v1y = (int16_t)(src1 >> 16);

   dst = (v0x * v1x) + (v0y * v1y) + src2;
""")

# Like sdot_2x16_iadd, but unsigned.
opcode("udot_2x16_uadd", 0, tuint32, [0, 0, 0], [tuint32, tuint32, tuint32],
       False, _2src_commutative, """
   const uint32_t v0x = (uint16_t)(src0      );
   const uint32_t v0y = (uint16_t)(src0 >> 16);
   const uint32_t v1x = (uint16_t)(src1      );
   const uint32_t v1y = (uint16_t)(src1 >> 16);

   dst = (v0x * v1x) + (v0y * v1y) + src2;
""")

# Like sdot_2x16_iadd, but the result is clampled to the range [-0x80000000, 0x7ffffffff].
opcode("sdot_2x16_iadd_sat", 0, tint32, [0, 0, 0], [tuint32, tuint32, tint32],
       False, _2src_commutative, """
   const int64_t v0x = (int16_t)(src0      );
   const int64_t v0y = (int16_t)(src0 >> 16);
   const int64_t v1x = (int16_t)(src1      );
   const int64_t v1y = (int16_t)(src1 >> 16);

   const int64_t tmp = (v0x * v1x) + (v0y * v1y) + src2;

   dst = tmp >= INT32_MAX ? INT32_MAX : (tmp <= INT32_MIN ? INT32_MIN : tmp);
""")

# Like udot_2x16_uadd, but the result is clampled to the range [0, 0xfffffffff].
opcode("udot_2x16_uadd_sat", 0, tint32, [0, 0, 0], [tuint32, tuint32, tint32],
       False, _2src_commutative, """
   const uint64_t v0x = (uint16_t)(src0      );
   const uint64_t v0y = (uint16_t)(src0 >> 16);
   const uint64_t v1x = (uint16_t)(src1      );
   const uint64_t v1y = (uint16_t)(src1 >> 16);

   const uint64_t tmp = (v0x * v1x) + (v0y * v1y) + src2;

   dst = tmp >= UINT32_MAX ? UINT32_MAX : tmp;
""")
