import re
from nir_opcodes import opcodes
from nir_opcodes import type_has_size, type_size, type_sizes, type_base_type

def type_add_size(type_, size):
    if type_has_size(type_):
        return type_
    return type_ + str(size)

def op_bit_sizes(op):
    sizes = None
    if not type_has_size(op.output_type):
        sizes = set(type_sizes(op.output_type))

    for input_type in op.input_types:
        if not type_has_size(input_type):
            if sizes is None:
                sizes = set(type_sizes(input_type))
            else:
                sizes = sizes.intersection(set(type_sizes(input_type)))

    return sorted(list(sizes)) if sizes is not None else None

def get_const_field(type_):
    if type_size(type_) == 1:
        return 'b'
    elif type_base_type(type_) == 'bool':
        return 'i' + str(type_size(type_))
    elif type_ == "float16":
        return "u16"
    else:
        return type_base_type(type_)[0] + str(type_size(type_))

template = """\
/*
 * Copyright (C) 2014 Intel Corporation
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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * Authors:
 *    Jason Ekstrand (jason@jlekstrand.net)
 */

#include <math.h>
#include "util/rounding.h" /* for _mesa_roundeven */
#include "util/half_float.h"
#include "util/double.h"
#include "util/softfloat.h"
#include "util/bigmath.h"
#include "util/format/format_utils.h"
#include "nir_constant_expressions.h"

/**
 * \brief Checks if the provided value is a denorm and flushes it to zero.
 */
static void
constant_denorm_flush_to_zero(nir_const_value *value, unsigned bit_size)
{
    switch(bit_size) {
    case 64:
        if (0 == (value->u64 & 0x7ff0000000000000))
            value->u64 &= 0x8000000000000000;
        break;
    case 32:
        if (0 == (value->u32 & 0x7f800000))
            value->u32 &= 0x80000000;
        break;
    case 16:
        if (0 == (value->u16 & 0x7c00))
            value->u16 &= 0x8000;
    }
}

/**
 * Evaluate one component of packSnorm4x8.
 */
static uint8_t
pack_snorm_1x8(float x)
{
    /* From section 8.4 of the GLSL 4.30 spec:
     *
     *    packSnorm4x8
     *    ------------
     *    The conversion for component c of v to fixed point is done as
     *    follows:
     *
     *      packSnorm4x8: round(clamp(c, -1, +1) * 127.0)
     *
     * We must first cast the float to an int, because casting a negative
     * float to a uint is undefined.
     */
   return (uint8_t) (int)
          _mesa_roundevenf(CLAMP(x, -1.0f, +1.0f) * 127.0f);
}

/**
 * Evaluate one component of packSnorm2x16.
 */
static uint16_t
pack_snorm_1x16(float x)
{
    /* From section 8.4 of the GLSL ES 3.00 spec:
     *
     *    packSnorm2x16
     *    -------------
     *    The conversion for component c of v to fixed point is done as
     *    follows:
     *
     *      packSnorm2x16: round(clamp(c, -1, +1) * 32767.0)
     *
     * We must first cast the float to an int, because casting a negative
     * float to a uint is undefined.
     */
   return (uint16_t) (int)
          _mesa_roundevenf(CLAMP(x, -1.0f, +1.0f) * 32767.0f);
}

/**
 * Evaluate one component of unpackSnorm4x8.
 */
static float
unpack_snorm_1x8(uint8_t u)
{
    /* From section 8.4 of the GLSL 4.30 spec:
     *
     *    unpackSnorm4x8
     *    --------------
     *    The conversion for unpacked fixed-point value f to floating point is
     *    done as follows:
     *
     *       unpackSnorm4x8: clamp(f / 127.0, -1, +1)
     */
   return CLAMP((int8_t) u / 127.0f, -1.0f, +1.0f);
}

/**
 * Evaluate one component of unpackSnorm2x16.
 */
static float
unpack_snorm_1x16(uint16_t u)
{
    /* From section 8.4 of the GLSL ES 3.00 spec:
     *
     *    unpackSnorm2x16
     *    ---------------
     *    The conversion for unpacked fixed-point value f to floating point is
     *    done as follows:
     *
     *       unpackSnorm2x16: clamp(f / 32767.0, -1, +1)
     */
   return CLAMP((int16_t) u / 32767.0f, -1.0f, +1.0f);
}

/**
 * Evaluate one component packUnorm4x8.
 */
static uint8_t
pack_unorm_1x8(float x)
{
    /* From section 8.4 of the GLSL 4.30 spec:
     *
     *    packUnorm4x8
     *    ------------
     *    The conversion for component c of v to fixed point is done as
     *    follows:
     *
     *       packUnorm4x8: round(clamp(c, 0, +1) * 255.0)
     */
   return (uint8_t) (int)
          _mesa_roundevenf(CLAMP(x, 0.0f, 1.0f) * 255.0f);
}

/**
 * Evaluate one component packUnorm2x16.
 */
static uint16_t
pack_unorm_1x16(float x)
{
    /* From section 8.4 of the GLSL ES 3.00 spec:
     *
     *    packUnorm2x16
     *    -------------
     *    The conversion for component c of v to fixed point is done as
     *    follows:
     *
     *       packUnorm2x16: round(clamp(c, 0, +1) * 65535.0)
     */
   return (uint16_t) (int)
          _mesa_roundevenf(CLAMP(x, 0.0f, 1.0f) * 65535.0f);
}

/**
 * Evaluate one component of unpackUnorm4x8.
 */
static float
unpack_unorm_1x8(uint8_t u)
{
    /* From section 8.4 of the GLSL 4.30 spec:
     *
     *    unpackUnorm4x8
     *    --------------
     *    The conversion for unpacked fixed-point value f to floating point is
     *    done as follows:
     *
     *       unpackUnorm4x8: f / 255.0
     */
   return (float) u / 255.0f;
}

/**
 * Evaluate one component of unpackUnorm2x16.
 */
static float
unpack_unorm_1x16(uint16_t u)
{
    /* From section 8.4 of the GLSL ES 3.00 spec:
     *
     *    unpackUnorm2x16
     *    ---------------
     *    The conversion for unpacked fixed-point value f to floating point is
     *    done as follows:
     *
     *       unpackUnorm2x16: f / 65535.0
     */
   return (float) u / 65535.0f;
}

/**
 * Evaluate one component of packHalf2x16.
 */
static uint16_t
pack_half_1x16(float x)
{
   return _mesa_float_to_half(x);
}

/**
 * Evaluate one component of unpackHalf2x16.
 */
static float
unpack_half_1x16_flush_to_zero(uint16_t u)
{
   if (0 == (u & 0x7c00))
      u &= 0x8000;
   return _mesa_half_to_float(u);
}

/**
 * Evaluate one component of unpackHalf2x16.
 */
static float
unpack_half_1x16(uint16_t u)
{
   return _mesa_half_to_float(u);
}

/* Some typed vector structures to make things like src0.y work */
typedef int8_t int1_t;
typedef uint8_t uint1_t;
typedef float float16_t;
typedef float float32_t;
typedef double float64_t;
typedef bool bool1_t;
typedef bool bool8_t;
typedef bool bool16_t;
typedef bool bool32_t;
typedef bool bool64_t;
% for type in ["float", "int", "uint", "bool"]:
% for width in type_sizes(type):
struct ${type}${width}_vec {
   ${type}${width}_t x;
   ${type}${width}_t y;
   ${type}${width}_t z;
   ${type}${width}_t w;
   ${type}${width}_t e;
   ${type}${width}_t f;
   ${type}${width}_t g;
   ${type}${width}_t h;
   ${type}${width}_t i;
   ${type}${width}_t j;
   ${type}${width}_t k;
   ${type}${width}_t l;
   ${type}${width}_t m;
   ${type}${width}_t n;
   ${type}${width}_t o;
   ${type}${width}_t p;
};
% endfor
% endfor

<%def name="evaluate_op(op, bit_size, execution_mode)">
   <%
   output_type = type_add_size(op.output_type, bit_size)
   input_types = [type_add_size(type_, bit_size) for type_ in op.input_types]
   %>

   ## For each non-per-component input, create a variable srcN that
   ## contains x, y, z, and w elements which are filled in with the
   ## appropriately-typed values.
   % for j in range(op.num_inputs):
      % if op.input_sizes[j] == 0:
         <% continue %>
      % elif "src" + str(j) not in op.const_expr:
         ## Avoid unused variable warnings
         <% continue %>
      %endif

      const struct ${input_types[j]}_vec src${j} = {
      % for k in range(op.input_sizes[j]):
         % if input_types[j] == "int1":
             /* 1-bit integers use a 0/-1 convention */
             -(int1_t)_src[${j}][${k}].b,
         % elif input_types[j] == "float16":
            _mesa_half_to_float(_src[${j}][${k}].u16),
         % else:
            _src[${j}][${k}].${get_const_field(input_types[j])},
         % endif
      % endfor
      % for k in range(op.input_sizes[j], 16):
         0,
      % endfor
      };
   % endfor

   % if op.output_size == 0:
      ## For per-component instructions, we need to iterate over the
      ## components and apply the constant expression one component
      ## at a time.
      for (unsigned _i = 0; _i < num_components; _i++) {
         ## For each per-component input, create a variable srcN that
         ## contains the value of the current (_i'th) component.
         % for j in range(op.num_inputs):
            % if op.input_sizes[j] != 0:
               <% continue %>
            % elif "src" + str(j) not in op.const_expr:
               ## Avoid unused variable warnings
               <% continue %>
            % elif input_types[j] == "int1":
               /* 1-bit integers use a 0/-1 convention */
               const int1_t src${j} = -(int1_t)_src[${j}][_i].b;
            % elif input_types[j] == "float16":
               const float src${j} =
                  _mesa_half_to_float(_src[${j}][_i].u16);
            % else:
               const ${input_types[j]}_t src${j} =
                  _src[${j}][_i].${get_const_field(input_types[j])};
            % endif
         % endfor

         ## Create an appropriately-typed variable dst and assign the
         ## result of the const_expr to it.  If const_expr already contains
         ## writes to dst, just include const_expr directly.
         % if "dst" in op.const_expr:
            ${output_type}_t dst;

            ${op.const_expr}
         % else:
            ${output_type}_t dst = ${op.const_expr};
         % endif

         ## Store the current component of the actual destination to the
         ## value of dst.
         % if output_type == "int1" or output_type == "uint1":
            /* 1-bit integers get truncated */
            _dst_val[_i].b = dst & 1;
         % elif output_type.startswith("bool"):
            ## Sanitize the C value to a proper NIR 0/-1 bool
            _dst_val[_i].${get_const_field(output_type)} = -(int)dst;
         % elif output_type == "float16":
            if (nir_is_rounding_mode_rtz(execution_mode, 16)) {
               _dst_val[_i].u16 = _mesa_float_to_float16_rtz(dst);
            } else {
               _dst_val[_i].u16 = _mesa_float_to_float16_rtne(dst);
            }
         % else:
            _dst_val[_i].${get_const_field(output_type)} = dst;
         % endif

         % if op.name != "fquantize2f16" and type_base_type(output_type) == "float":
            % if type_has_size(output_type):
               if (nir_is_denorm_flush_to_zero(execution_mode, ${type_size(output_type)})) {
                  constant_denorm_flush_to_zero(&_dst_val[_i], ${type_size(output_type)});
               }
            % else:
               if (nir_is_denorm_flush_to_zero(execution_mode, ${bit_size})) {
                  constant_denorm_flush_to_zero(&_dst_val[i], bit_size);
               }
            %endif
         % endif
      }
   % else:
      ## In the non-per-component case, create a struct dst with
      ## appropriately-typed elements x, y, z, and w and assign the result
      ## of the const_expr to all components of dst, or include the
      ## const_expr directly if it writes to dst already.
      struct ${output_type}_vec dst;

      % if "dst" in op.const_expr:
         ${op.const_expr}
      % else:
         ## Splat the value to all components.  This way expressions which
         ## write the same value to all components don't need to explicitly
         ## write to dest.
         dst.x = dst.y = dst.z = dst.w = ${op.const_expr};
      % endif

      ## For each component in the destination, copy the value of dst to
      ## the actual destination.
      % for k in range(op.output_size):
         % if output_type == "int1" or output_type == "uint1":
            /* 1-bit integers get truncated */
            _dst_val[${k}].b = dst.${"xyzwefghijklmnop"[k]} & 1;
         % elif output_type.startswith("bool"):
            ## Sanitize the C value to a proper NIR 0/-1 bool
            _dst_val[${k}].${get_const_field(output_type)} = -(int)dst.${"xyzwefghijklmnop"[k]};
         % elif output_type == "float16":
            if (nir_is_rounding_mode_rtz(execution_mode, 16)) {
               _dst_val[${k}].u16 = _mesa_float_to_float16_rtz(dst.${"xyzwefghijklmnop"[k]});
            } else {
               _dst_val[${k}].u16 = _mesa_float_to_float16_rtne(dst.${"xyzwefghijklmnop"[k]});
            }
         % else:
            _dst_val[${k}].${get_const_field(output_type)} = dst.${"xyzwefghijklmnop"[k]};
         % endif

         % if op.name != "fquantize2f16" and type_base_type(output_type) == "float":
            % if type_has_size(output_type):
               if (nir_is_denorm_flush_to_zero(execution_mode, ${type_size(output_type)})) {
                  constant_denorm_flush_to_zero(&_dst_val[${k}], ${type_size(output_type)});
               }
            % else:
               if (nir_is_denorm_flush_to_zero(execution_mode, ${bit_size})) {
                  constant_denorm_flush_to_zero(&_dst_val[${k}], bit_size);
               }
            % endif
         % endif
      % endfor
   % endif
</%def>

% for name, op in sorted(opcodes.items()):
% if op.name == "fsat":
#if defined(_MSC_VER) && (defined(_M_ARM64) || defined(_M_ARM64EC))
#pragma optimize("", off) /* Temporary work-around for MSVC compiler bug, present in VS2019 16.9.2 */
#endif
% endif
static void
evaluate_${name}(nir_const_value *_dst_val,
                 UNUSED unsigned num_components,
                 ${"UNUSED" if op_bit_sizes(op) is None else ""} unsigned bit_size,
                 UNUSED nir_const_value **_src,
                 UNUSED unsigned execution_mode)
{
   % if op_bit_sizes(op) is not None:
      switch (bit_size) {
      % for bit_size in op_bit_sizes(op):
      case ${bit_size}: {
         ${evaluate_op(op, bit_size, execution_mode)}
         break;
      }
      % endfor

      default:
         unreachable("unknown bit width");
      }
   % else:
      ${evaluate_op(op, 0, execution_mode)}
   % endif
}
% if op.name == "fsat":
#if defined(_MSC_VER) && (defined(_M_ARM64) || defined(_M_ARM64EC))
#pragma optimize("", on) /* Temporary work-around for MSVC compiler bug, present in VS2019 16.9.2 */
#endif
% endif
% endfor

void
nir_eval_const_opcode(nir_op op, nir_const_value *dest,
                      unsigned num_components, unsigned bit_width,
                      nir_const_value **src,
                      unsigned float_controls_execution_mode)
{
   switch (op) {
% for name in sorted(opcodes.keys()):
   case nir_op_${name}:
      evaluate_${name}(dest, num_components, bit_width, src, float_controls_execution_mode);
      return;
% endfor
   default:
      unreachable("shouldn't get here");
   }
}"""

from mako.template import Template

print(Template(template).render(opcodes=opcodes, type_sizes=type_sizes,
                                type_base_type=type_base_type,
                                type_size=type_size,
                                type_has_size=type_has_size,
                                type_add_size=type_add_size,
                                op_bit_sizes=op_bit_sizes,
                                get_const_field=get_const_field))
