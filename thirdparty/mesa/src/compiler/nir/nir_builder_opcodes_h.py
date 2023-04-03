template = """\
/* Copyright (C) 2015 Broadcom
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
 */

#ifndef _NIR_BUILDER_OPCODES_
#define _NIR_BUILDER_OPCODES_

<%
def src_decl_list(num_srcs):
   return ', '.join('nir_ssa_def *src' + str(i) for i in range(num_srcs))

def src_list(num_srcs):
   return ', '.join('src' + str(i) for i in range(num_srcs))

def needs_num_components(opcode):
   return "replicated" in opcode.name
%>

% for name, opcode in sorted(opcodes.items()):
% if not needs_num_components(opcode):
static inline nir_ssa_def *
nir_${name}(nir_builder *build, ${src_decl_list(opcode.num_inputs)})
{
% if opcode.is_conversion and \
     type_base_type(opcode.output_type) == opcode.input_types[0]:
   if (src0->bit_size == ${type_size(opcode.output_type)})
      return src0;
%endif
% if opcode.num_inputs <= 4:
   return nir_build_alu${opcode.num_inputs}(build, nir_op_${name}, ${src_list(opcode.num_inputs)});
% else:
   nir_ssa_def *srcs[${opcode.num_inputs}] = {${src_list(opcode.num_inputs)}};
   return nir_build_alu_src_arr(build, nir_op_${name}, srcs);
% endif
}
% endif
% endfor

% for name, opcode in sorted(INTR_OPCODES.items()):
struct _nir_${name}_indices {
   int _; /* exists to avoid empty initializers */
% for index in opcode.indices:
   ${index.c_data_type} ${index.name};
% endfor
};
% endfor

<%
def intrinsic_decl_list(opcode):
    need_components = opcode.dest_components == 0 and \
                      0 not in opcode.src_components

    res = ''
    if (opcode.has_dest or opcode.num_srcs) and need_components:
        res += ', unsigned num_components'
    if opcode.has_dest and len(opcode.bit_sizes) != 1 and opcode.bit_size_src == -1:
        res += ', unsigned bit_size'
    for i in range(opcode.num_srcs):
        res += ', nir_ssa_def *src' + str(i)
    if opcode.indices:
        res += ', struct _nir_' + opcode.name + '_indices indices'
    return res

def intrinsic_macro_list(opcode):
    need_components = opcode.dest_components == 0 and \
                      0 not in opcode.src_components

    res = ''
    if (opcode.has_dest or opcode.num_srcs) and need_components:
        res += ', num_components'
    if opcode.has_dest and len(opcode.bit_sizes) != 1 and opcode.bit_size_src == -1:
        res += ', bit_size'
    for i in range(opcode.num_srcs):
        res += ', src' + str(i)
    return res

def get_intrinsic_bitsize(opcode):
    if len(opcode.bit_sizes) == 1:
        return str(opcode.bit_sizes[0])
    elif opcode.bit_size_src != -1:
        return 'src' + str(opcode.bit_size_src) + '->bit_size'
    else:
        return 'bit_size'
%>

% for name, opcode in sorted(INTR_OPCODES.items()):
% if opcode.has_dest:
static inline nir_ssa_def *
% else:
static inline nir_intrinsic_instr *
% endif
_nir_build_${name}(nir_builder *build${intrinsic_decl_list(opcode)})
{
   nir_intrinsic_instr *intrin = nir_intrinsic_instr_create(
      build->shader, nir_intrinsic_${name});

   % if 0 in opcode.src_components:
   intrin->num_components = src${opcode.src_components.index(0)}->num_components;
   % elif opcode.dest_components == 0:
   intrin->num_components = num_components;
   % endif
   % if opcode.has_dest:
      % if opcode.dest_components == 0:
      nir_ssa_dest_init(&intrin->instr, &intrin->dest, intrin->num_components, ${get_intrinsic_bitsize(opcode)}, NULL);
      % else:
      nir_ssa_dest_init(&intrin->instr, &intrin->dest, ${opcode.dest_components}, ${get_intrinsic_bitsize(opcode)}, NULL);
      % endif
   % endif
   % for i in range(opcode.num_srcs):
   intrin->src[${i}] = nir_src_for_ssa(src${i});
   % endfor
   % if WRITE_MASK in opcode.indices and 0 in opcode.src_components:
   if (!indices.write_mask)
      indices.write_mask = BITFIELD_MASK(intrin->num_components);
   % endif
   % if ALIGN_MUL in opcode.indices and 0 in opcode.src_components:
   if (!indices.align_mul)
      indices.align_mul = src${opcode.src_components.index(0)}->bit_size / 8u;
   % elif ALIGN_MUL in opcode.indices and opcode.dest_components == 0:
   if (!indices.align_mul)
      indices.align_mul = intrin->dest.ssa.bit_size / 8u;
   % endif
   % for index in opcode.indices:
   nir_intrinsic_set_${index.name}(intrin, indices.${index.name});
   % endfor

   nir_builder_instr_insert(build, &intrin->instr);
   % if opcode.has_dest:
   return &intrin->dest.ssa;
   % else:
   return intrin;
   % endif
}
% endfor

% for name, opcode in sorted(INTR_OPCODES.items()):
% if opcode.indices:
#ifdef __cplusplus
#define nir_build_${name}(build${intrinsic_macro_list(opcode)}, ...) ${'\\\\'}
_nir_build_${name}(build${intrinsic_macro_list(opcode)}, _nir_${name}_indices{0, __VA_ARGS__})
#else
#define nir_build_${name}(build${intrinsic_macro_list(opcode)}, ...) ${'\\\\'}
_nir_build_${name}(build${intrinsic_macro_list(opcode)}, (struct _nir_${name}_indices){0, __VA_ARGS__})
#endif
% else:
#define nir_build_${name} _nir_build_${name}
% endif
#define nir_${name} nir_build_${name}
% endfor

#endif /* _NIR_BUILDER_OPCODES_ */"""

from nir_opcodes import opcodes, type_size, type_base_type
from nir_intrinsics import INTR_OPCODES, WRITE_MASK, ALIGN_MUL
from mako.template import Template

print(Template(template).render(opcodes=opcodes,
                                type_size=type_size,
                                type_base_type=type_base_type,
                                INTR_OPCODES=INTR_OPCODES,
                                WRITE_MASK=WRITE_MASK,
                                ALIGN_MUL=ALIGN_MUL))
