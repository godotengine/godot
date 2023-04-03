#
# Copyright (C) 2015 Intel Corporation
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

import mako.template
import sys

class type(object):
   def __init__(self, c_type, union_field, glsl_type):
      self.c_type = c_type
      self.union_field = union_field
      self.glsl_type = glsl_type


class type_signature_iter(object):
   """Basic iterator for a set of type signatures.  Various kinds of sequences of
   types come in, and an iteration of type_signature objects come out.

   """

   def __init__(self, source_types, num_operands):
      """Initialize an iterator from a sequence of input types and a number
      operands.  This is for signatures where all the operands have the same
      type and the result type of the operation is the same as the input type.

      """
      self.dest_type = None
      self.source_types = source_types
      self.num_operands = num_operands
      self.i = 0

   def __init__(self, dest_type, source_types, num_operands):
      """Initialize an iterator from a result tpye, a sequence of input types and a
      number operands.  This is for signatures where all the operands have the
      same type but the result type of the operation is different from the
      input type.

      """
      self.dest_type = dest_type
      self.source_types = source_types
      self.num_operands = num_operands
      self.i = 0

   def __iter__(self):
      return self

   def __next__(self):
      if self.i < len(self.source_types):
         i = self.i
         self.i += 1

         if self.dest_type is None:
            dest_type = self.source_types[i]
         else:
            dest_type = self.dest_type

         return (dest_type, self.num_operands * (self.source_types[i],))
      else:
         raise StopIteration()

   next = __next__


uint_type = type("unsigned", "u", "GLSL_TYPE_UINT")
int_type = type("int", "i", "GLSL_TYPE_INT")
uint64_type = type("uint64_t", "u64", "GLSL_TYPE_UINT64")
int64_type = type("int64_t", "i64", "GLSL_TYPE_INT64")
float_type = type("float", "f", "GLSL_TYPE_FLOAT")
double_type = type("double", "d", "GLSL_TYPE_DOUBLE")
bool_type = type("bool", "b", "GLSL_TYPE_BOOL")

all_types = (uint_type, int_type, float_type, double_type, uint64_type, int64_type, bool_type)
numeric_types = (uint_type, int_type, float_type, double_type, uint64_type, int64_type)
signed_numeric_types = (int_type, float_type, double_type, int64_type)
integer_types = (uint_type, int_type, uint64_type, int64_type)
real_types = (float_type, double_type)

# This template is for operations that can have operands of a several
# different types, and each type may or may not has a different C expression.
# This is used by most operations.
constant_template_common = mako.template.Template("""\
   case ${op.get_enum_name()}:
      for (unsigned c = 0; c < op[0]->type->components(); c++) {
         switch (op[0]->type->base_type) {
    % for dst_type, src_types in op.signatures():
         case ${src_types[0].glsl_type}:
            data.${dst_type.union_field}[c] = ${op.get_c_expression(src_types)};
            break;
    % endfor
         default:
            unreachable("invalid type");
         }
      }
      break;""")

# This template is for binary operations that can operate on some combination
# of scalar and vector operands.
constant_template_vector_scalar = mako.template.Template("""\
   case ${op.get_enum_name()}:
    % if "mixed" in op.flags:
        % for i in range(op.num_operands):
      assert(op[${i}]->type->base_type == ${op.source_types[0].glsl_type} ||
            % for src_type in op.source_types[1:-1]:
             op[${i}]->type->base_type == ${src_type.glsl_type} ||
            % endfor
             op[${i}]->type->base_type == ${op.source_types[-1].glsl_type});
        % endfor
    % else:
      assert(op[0]->type == op[1]->type || op0_scalar || op1_scalar);
    % endif
      for (unsigned c = 0, c0 = 0, c1 = 0;
           c < components;
           c0 += c0_inc, c1 += c1_inc, c++) {

         switch (op[0]->type->base_type) {
    % for dst_type, src_types in op.signatures():
         case ${src_types[0].glsl_type}:
            data.${dst_type.union_field}[c] = ${op.get_c_expression(src_types, ("c0", "c1", "c2"))};
            break;
    % endfor
         default:
            unreachable("invalid type");
         }
      }
      break;""")

# This template is for multiplication.  It is unique because it has to support
# matrix * vector and matrix * matrix operations, and those are just different.
constant_template_mul = mako.template.Template("""\
   case ${op.get_enum_name()}:
      /* Check for equal types, or unequal types involving scalars */
      if ((op[0]->type == op[1]->type && !op[0]->type->is_matrix())
          || op0_scalar || op1_scalar) {
         for (unsigned c = 0, c0 = 0, c1 = 0;
              c < components;
              c0 += c0_inc, c1 += c1_inc, c++) {

            switch (op[0]->type->base_type) {
    % for dst_type, src_types in op.signatures():
            case ${src_types[0].glsl_type}:
               data.${dst_type.union_field}[c] = ${op.get_c_expression(src_types, ("c0", "c1", "c2"))};
               break;
    % endfor
            default:
               unreachable("invalid type");
            }
         }
      } else {
         assert(op[0]->type->is_matrix() || op[1]->type->is_matrix());

         /* Multiply an N-by-M matrix with an M-by-P matrix.  Since either
          * matrix can be a GLSL vector, either N or P can be 1.
          *
          * For vec*mat, the vector is treated as a row vector.  This
          * means the vector is a 1-row x M-column matrix.
          *
          * For mat*vec, the vector is treated as a column vector.  Since
          * matrix_columns is 1 for vectors, this just works.
          */
         const unsigned n = op[0]->type->is_vector()
            ? 1 : op[0]->type->vector_elements;
         const unsigned m = op[1]->type->vector_elements;
         const unsigned p = op[1]->type->matrix_columns;
         for (unsigned j = 0; j < p; j++) {
            for (unsigned i = 0; i < n; i++) {
               for (unsigned k = 0; k < m; k++) {
                  if (op[0]->type->is_double())
                     data.d[i+n*j] += op[0]->value.d[i+n*k]*op[1]->value.d[k+m*j];
                  else
                     data.f[i+n*j] += op[0]->value.f[i+n*k]*op[1]->value.f[k+m*j];
               }
            }
         }
      }
      break;""")

# This template is for operations that are horizontal and either have only a
# single type or the implementation for all types is identical.  That is, the
# operation consumes a vector and produces a scalar.
constant_template_horizontal_single_implementation = mako.template.Template("""\
   case ${op.get_enum_name()}:
      data.${op.dest_type.union_field}[0] = ${op.c_expression['default']};
      break;""")

# This template is for operations that are horizontal and do not assign the
# result.  The various unpack operations are examples.
constant_template_horizontal_nonassignment = mako.template.Template("""\
   case ${op.get_enum_name()}:
      ${op.c_expression['default']};
      break;""")

# This template is for binary operations that are horizontal.  That is, the
# operation consumes a vector and produces a scalar.
constant_template_horizontal = mako.template.Template("""\
   case ${op.get_enum_name()}:
      switch (op[0]->type->base_type) {
    % for dst_type, src_types in op.signatures():
      case ${src_types[0].glsl_type}:
         data.${dst_type.union_field}[0] = ${op.get_c_expression(src_types)};
         break;
    % endfor
      default:
         unreachable("invalid type");
      }
      break;""")

# This template is for ir_binop_vector_extract.
constant_template_vector_extract = mako.template.Template("""\
   case ${op.get_enum_name()}: {
      const int c = CLAMP(op[1]->value.i[0], 0,
                          (int) op[0]->type->vector_elements - 1);

      switch (op[0]->type->base_type) {
    % for dst_type, src_types in op.signatures():
      case ${src_types[0].glsl_type}:
         data.${dst_type.union_field}[0] = op[0]->value.${src_types[0].union_field}[c];
         break;
    % endfor
      default:
         unreachable("invalid type");
      }
      break;
   }""")

# This template is for ir_triop_vector_insert.
constant_template_vector_insert = mako.template.Template("""\
   case ${op.get_enum_name()}: {
      const unsigned idx = op[2]->value.u[0];

      memcpy(&data, &op[0]->value, sizeof(data));

      switch (return_type->base_type) {
    % for dst_type, src_types in op.signatures():
      case ${src_types[0].glsl_type}:
         data.${dst_type.union_field}[idx] = op[1]->value.${src_types[0].union_field}[0];
         break;
    % endfor
      default:
         unreachable("invalid type");
      }
      break;
   }""")

# This template is for ir_quadop_vector.
constant_template_vector = mako.template.Template("""\
   case ${op.get_enum_name()}:
      for (unsigned c = 0; c < return_type->vector_elements; c++) {
         switch (return_type->base_type) {
    % for dst_type, src_types in op.signatures():
         case ${src_types[0].glsl_type}:
            data.${dst_type.union_field}[c] = op[c]->value.${src_types[0].union_field}[0];
            break;
    % endfor
         default:
            unreachable("invalid type");
         }
      }
      break;""")

# This template is for ir_triop_lrp.
constant_template_lrp = mako.template.Template("""\
   case ${op.get_enum_name()}: {
      assert(op[0]->type->is_float() || op[0]->type->is_double());
      assert(op[1]->type->is_float() || op[1]->type->is_double());
      assert(op[2]->type->is_float() || op[2]->type->is_double());

      unsigned c2_inc = op[2]->type->is_scalar() ? 0 : 1;
      for (unsigned c = 0, c2 = 0; c < components; c2 += c2_inc, c++) {
         switch (return_type->base_type) {
    % for dst_type, src_types in op.signatures():
         case ${src_types[0].glsl_type}:
            data.${dst_type.union_field}[c] = ${op.get_c_expression(src_types, ("c", "c", "c2"))};
            break;
    % endfor
         default:
            unreachable("invalid type");
         }
      }
      break;
   }""")

# This template is for ir_triop_csel.  This expression is really unique
# because not all of the operands are the same type, and the second operand
# determines the type of the expression (instead of the first).
constant_template_csel = mako.template.Template("""\
   case ${op.get_enum_name()}:
      for (unsigned c = 0; c < components; c++) {
         switch (return_type->base_type) {
    % for dst_type, src_types in op.signatures():
         case ${src_types[1].glsl_type}:
            data.${dst_type.union_field}[c] = ${op.get_c_expression(src_types)};
            break;
    % endfor
         default:
            unreachable("invalid type");
         }
      }
      break;""")


vector_scalar_operation = "vector-scalar"
horizontal_operation = "horizontal"
types_identical_operation = "identical"
non_assign_operation = "nonassign"
mixed_type_operation = "mixed"

class operation(object):
   def __init__(self, name, num_operands, printable_name = None, source_types = None, dest_type = None, c_expression = None, flags = None, all_signatures = None):
      self.name = name
      self.num_operands = num_operands

      if printable_name is None:
         self.printable_name = name
      else:
         self.printable_name = printable_name

      self.all_signatures = all_signatures

      if source_types is None:
         self.source_types = tuple()
      else:
         self.source_types = source_types

      self.dest_type = dest_type

      if c_expression is None:
         self.c_expression = None
      elif isinstance(c_expression, str):
         self.c_expression = {'default': c_expression}
      else:
         self.c_expression = c_expression

      if flags is None:
         self.flags = frozenset()
      elif isinstance(flags, str):
         self.flags = frozenset([flags])
      else:
         self.flags = frozenset(flags)


   def get_enum_name(self):
      return "ir_{0}op_{1}".format(("un", "bin", "tri", "quad")[self.num_operands-1], self.name)


   def get_template(self):
      if self.c_expression is None:
         return None

      if horizontal_operation in self.flags:
         if non_assign_operation in self.flags:
            return constant_template_horizontal_nonassignment.render(op=self)
         elif types_identical_operation in self.flags:
            return constant_template_horizontal_single_implementation.render(op=self)
         else:
            return constant_template_horizontal.render(op=self)

      if self.num_operands == 2:
         if self.name == "mul":
            return constant_template_mul.render(op=self)
         elif self.name == "vector_extract":
            return constant_template_vector_extract.render(op=self)
         elif vector_scalar_operation in self.flags:
            return constant_template_vector_scalar.render(op=self)
      elif self.num_operands == 3:
         if self.name == "vector_insert":
            return constant_template_vector_insert.render(op=self)
         elif self.name == "lrp":
            return constant_template_lrp.render(op=self)
         elif self.name == "csel":
            return constant_template_csel.render(op=self)
      elif self.num_operands == 4:
         if self.name == "vector":
            return constant_template_vector.render(op=self)

      return constant_template_common.render(op=self)


   def get_c_expression(self, types, indices=("c", "c", "c")):
      src0 = "op[0]->value.{0}[{1}]".format(types[0].union_field, indices[0])
      src1 = "op[1]->value.{0}[{1}]".format(types[1].union_field, indices[1]) if len(types) >= 2 else "ERROR"
      src2 = "op[2]->value.{0}[{1}]".format(types[2].union_field, indices[2]) if len(types) >= 3 else "ERROR"
      src3 = "op[3]->value.{0}[c]".format(types[3].union_field) if len(types) >= 4 else "ERROR"

      expr = self.c_expression[types[0].union_field] if types[0].union_field in self.c_expression else self.c_expression['default']

      return expr.format(src0=src0,
                         src1=src1,
                         src2=src2,
                         src3=src3)


   def signatures(self):
      if self.all_signatures is not None:
         return self.all_signatures
      else:
         return type_signature_iter(self.dest_type, self.source_types, self.num_operands)


ir_expression_operation = [
   operation("bit_not", 1, printable_name="~", source_types=integer_types, c_expression="~ {src0}"),
   operation("logic_not", 1, printable_name="!", source_types=(bool_type,), c_expression="!{src0}"),
   operation("neg", 1, source_types=numeric_types, c_expression={'u': "-((int) {src0})", 'u64': "-((int64_t) {src0})", 'default': "-{src0}"}),
   operation("abs", 1, source_types=signed_numeric_types, c_expression={'i': "{src0} < 0 ? -{src0} : {src0}", 'f': "fabsf({src0})", 'd': "fabs({src0})", 'i64': "{src0} < 0 ? -{src0} : {src0}"}),
   operation("sign", 1, source_types=signed_numeric_types, c_expression={'i': "({src0} > 0) - ({src0} < 0)", 'f': "float(({src0} > 0.0F) - ({src0} < 0.0F))", 'd': "double(({src0} > 0.0) - ({src0} < 0.0))", 'i64': "({src0} > 0) - ({src0} < 0)"}),
   operation("rcp", 1, source_types=real_types, c_expression={'f': "1.0F / {src0}", 'd': "1.0 / {src0}"}),
   operation("rsq", 1, source_types=real_types, c_expression={'f': "1.0F / sqrtf({src0})", 'd': "1.0 / sqrt({src0})"}),
   operation("sqrt", 1, source_types=real_types, c_expression={'f': "sqrtf({src0})", 'd': "sqrt({src0})"}),
   operation("exp", 1, source_types=(float_type,), c_expression="expf({src0})"),         # Log base e on gentype
   operation("log", 1, source_types=(float_type,), c_expression="logf({src0})"),         # Natural log on gentype
   operation("exp2", 1, source_types=(float_type,), c_expression="exp2f({src0})"),
   operation("log2", 1, source_types=(float_type,), c_expression="log2f({src0})"),

   # Float-to-integer conversion.
   operation("f2i", 1, source_types=(float_type,), dest_type=int_type, c_expression="(int) {src0}"),
   # Float-to-unsigned conversion.
   operation("f2u", 1, source_types=(float_type,), dest_type=uint_type, c_expression="(unsigned) {src0}"),
   # Integer-to-float conversion.
   operation("i2f", 1, source_types=(int_type,), dest_type=float_type, c_expression="(float) {src0}"),
   # Float-to-boolean conversion
   operation("f2b", 1, source_types=(float_type,), dest_type=bool_type, c_expression="{src0} != 0.0F ? true : false"),
   # Boolean-to-float conversion
   operation("b2f", 1, source_types=(bool_type,), dest_type=float_type, c_expression="{src0} ? 1.0F : 0.0F"),
   # Boolean-to-float16 conversion
   operation("b2f16", 1, source_types=(bool_type,), dest_type=float_type, c_expression="{src0} ? 1.0F : 0.0F"),
   # int-to-boolean conversion
   operation("i2b", 1, source_types=(uint_type, int_type), dest_type=bool_type, c_expression="{src0} ? true : false"),
   # Boolean-to-int conversion
   operation("b2i", 1, source_types=(bool_type,), dest_type=int_type, c_expression="{src0} ? 1 : 0"),
   # Unsigned-to-float conversion.
   operation("u2f", 1, source_types=(uint_type,), dest_type=float_type, c_expression="(float) {src0}"),
   # Integer-to-unsigned conversion.
   operation("i2u", 1, source_types=(int_type,), dest_type=uint_type, c_expression="{src0}"),
   # Unsigned-to-integer conversion.
   operation("u2i", 1, source_types=(uint_type,), dest_type=int_type, c_expression="{src0}"),
   # Double-to-float conversion.
   operation("d2f", 1, source_types=(double_type,), dest_type=float_type, c_expression="{src0}"),
   # Float-to-double conversion.
   operation("f2d", 1, source_types=(float_type,), dest_type=double_type, c_expression="{src0}"),
   # Half-float conversions. These all operate on and return float types,
   # since the framework expands half to full float before calling in.  We
   # still have to handle them here so that we can constant propagate through
   # them, but they are no-ops.
   operation("f2f16", 1, source_types=(float_type,), dest_type=float_type, c_expression="{src0}"),
   operation("f2fmp", 1, source_types=(float_type,), dest_type=float_type, c_expression="{src0}"),
   operation("f162f", 1, source_types=(float_type,), dest_type=float_type, c_expression="{src0}"),
   # int16<->int32 conversion.
   operation("i2i", 1, source_types=(int_type,), dest_type=int_type, c_expression="{src0}"),
   operation("i2imp", 1, source_types=(int_type,), dest_type=int_type, c_expression="{src0}"),
   operation("u2u", 1, source_types=(uint_type,), dest_type=uint_type, c_expression="{src0}"),
   operation("u2ump", 1, source_types=(uint_type,), dest_type=uint_type, c_expression="{src0}"),
   # Double-to-integer conversion.
   operation("d2i", 1, source_types=(double_type,), dest_type=int_type, c_expression="{src0}"),
   # Integer-to-double conversion.
   operation("i2d", 1, source_types=(int_type,), dest_type=double_type, c_expression="{src0}"),
   # Double-to-unsigned conversion.
   operation("d2u", 1, source_types=(double_type,), dest_type=uint_type, c_expression="{src0}"),
   # Unsigned-to-double conversion.
   operation("u2d", 1, source_types=(uint_type,), dest_type=double_type, c_expression="{src0}"),
   # Double-to-boolean conversion.
   operation("d2b", 1, source_types=(double_type,), dest_type=bool_type, c_expression="{src0} != 0.0"),
   # Float16-to-boolean conversion.
   operation("f162b", 1, source_types=(float_type,), dest_type=bool_type, c_expression="{src0} != 0.0"),
   # 'Bit-identical int-to-float "conversion"
   operation("bitcast_i2f", 1, source_types=(int_type,), dest_type=float_type, c_expression="bitcast_u2f({src0})"),
   # 'Bit-identical float-to-int "conversion"
   operation("bitcast_f2i", 1, source_types=(float_type,), dest_type=int_type, c_expression="bitcast_f2u({src0})"),
   # 'Bit-identical uint-to-float "conversion"
   operation("bitcast_u2f", 1, source_types=(uint_type,), dest_type=float_type, c_expression="bitcast_u2f({src0})"),
   # 'Bit-identical float-to-uint "conversion"
   operation("bitcast_f2u", 1, source_types=(float_type,), dest_type=uint_type, c_expression="bitcast_f2u({src0})"),
   # Bit-identical u64-to-double "conversion"
   operation("bitcast_u642d", 1, source_types=(uint64_type,), dest_type=double_type, c_expression="bitcast_u642d({src0})"),
   # Bit-identical i64-to-double "conversion"
   operation("bitcast_i642d", 1, source_types=(int64_type,), dest_type=double_type, c_expression="bitcast_i642d({src0})"),
   # Bit-identical double-to_u64 "conversion"
   operation("bitcast_d2u64", 1, source_types=(double_type,), dest_type=uint64_type, c_expression="bitcast_d2u64({src0})"),
   # Bit-identical double-to-i64 "conversion"
   operation("bitcast_d2i64", 1, source_types=(double_type,), dest_type=int64_type, c_expression="bitcast_d2i64({src0})"),
   # i64-to-i32 conversion
   operation("i642i", 1, source_types=(int64_type,), dest_type=int_type, c_expression="{src0}"),
   # ui64-to-i32 conversion
   operation("u642i", 1, source_types=(uint64_type,), dest_type=int_type, c_expression="{src0}"),
   operation("i642u", 1, source_types=(int64_type,), dest_type=uint_type, c_expression="{src0}"),
   operation("u642u", 1, source_types=(uint64_type,), dest_type=uint_type, c_expression="{src0}"),
   operation("i642b", 1, source_types=(int64_type,), dest_type=bool_type, c_expression="{src0} != 0"),
   operation("i642f", 1, source_types=(int64_type,), dest_type=float_type, c_expression="{src0}"),
   operation("u642f", 1, source_types=(uint64_type,), dest_type=float_type, c_expression="{src0}"),
   operation("i642d", 1, source_types=(int64_type,), dest_type=double_type, c_expression="{src0}"),
   operation("u642d", 1, source_types=(uint64_type,), dest_type=double_type, c_expression="{src0}"),
   operation("i2i64", 1, source_types=(int_type,), dest_type=int64_type, c_expression="{src0}"),
   operation("u2i64", 1, source_types=(uint_type,), dest_type=int64_type, c_expression="{src0}"),
   operation("b2i64", 1, source_types=(bool_type,), dest_type=int64_type, c_expression="{src0}"),
   operation("f2i64", 1, source_types=(float_type,), dest_type=int64_type, c_expression="{src0}"),
   operation("d2i64", 1, source_types=(double_type,), dest_type=int64_type, c_expression="{src0}"),
   operation("i2u64", 1, source_types=(int_type,), dest_type=uint64_type, c_expression="{src0}"),
   operation("u2u64", 1, source_types=(uint_type,), dest_type=uint64_type, c_expression="{src0}"),
   operation("f2u64", 1, source_types=(float_type,), dest_type=uint64_type, c_expression="{src0}"),
   operation("d2u64", 1, source_types=(double_type,), dest_type=uint64_type, c_expression="{src0}"),
   operation("u642i64", 1, source_types=(uint64_type,), dest_type=int64_type, c_expression="{src0}"),
   operation("i642u64", 1, source_types=(int64_type,), dest_type=uint64_type, c_expression="{src0}"),


   # Unary floating-point rounding operations.
   operation("trunc", 1, source_types=real_types, c_expression={'f': "truncf({src0})", 'd': "trunc({src0})"}),
   operation("ceil", 1, source_types=real_types, c_expression={'f': "ceilf({src0})", 'd': "ceil({src0})"}),
   operation("floor", 1, source_types=real_types, c_expression={'f': "floorf({src0})", 'd': "floor({src0})"}),
   operation("fract", 1, source_types=real_types, c_expression={'f': "{src0} - floorf({src0})", 'd': "{src0} - floor({src0})"}),
   operation("round_even", 1, source_types=real_types, c_expression={'f': "_mesa_roundevenf({src0})", 'd': "_mesa_roundeven({src0})"}),

   # Trigonometric operations.
   operation("sin", 1, source_types=(float_type,), c_expression="sinf({src0})"),
   operation("cos", 1, source_types=(float_type,), c_expression="cosf({src0})"),
   operation("atan", 1, source_types=(float_type,), c_expression="atan({src0})"),

   # Partial derivatives.
   operation("dFdx", 1, source_types=(float_type,), c_expression="0.0f"),
   operation("dFdx_coarse", 1, printable_name="dFdxCoarse", source_types=(float_type,), c_expression="0.0f"),
   operation("dFdx_fine", 1, printable_name="dFdxFine", source_types=(float_type,), c_expression="0.0f"),
   operation("dFdy", 1, source_types=(float_type,), c_expression="0.0f"),
   operation("dFdy_coarse", 1, printable_name="dFdyCoarse", source_types=(float_type,), c_expression="0.0f"),
   operation("dFdy_fine", 1, printable_name="dFdyFine", source_types=(float_type,), c_expression="0.0f"),

   # Floating point pack and unpack operations.
   operation("pack_snorm_2x16", 1, printable_name="packSnorm2x16", source_types=(float_type,), dest_type=uint_type, c_expression="pack_2x16(pack_snorm_1x16, op[0]->value.f[0], op[0]->value.f[1])", flags=horizontal_operation),
   operation("pack_snorm_4x8", 1, printable_name="packSnorm4x8", source_types=(float_type,), dest_type=uint_type, c_expression="pack_4x8(pack_snorm_1x8, op[0]->value.f[0], op[0]->value.f[1], op[0]->value.f[2], op[0]->value.f[3])", flags=horizontal_operation),
   operation("pack_unorm_2x16", 1, printable_name="packUnorm2x16", source_types=(float_type,), dest_type=uint_type, c_expression="pack_2x16(pack_unorm_1x16, op[0]->value.f[0], op[0]->value.f[1])", flags=horizontal_operation),
   operation("pack_unorm_4x8", 1, printable_name="packUnorm4x8", source_types=(float_type,), dest_type=uint_type, c_expression="pack_4x8(pack_unorm_1x8, op[0]->value.f[0], op[0]->value.f[1], op[0]->value.f[2], op[0]->value.f[3])", flags=horizontal_operation),
   operation("pack_half_2x16", 1, printable_name="packHalf2x16", source_types=(float_type,), dest_type=uint_type, c_expression="pack_2x16(pack_half_1x16, op[0]->value.f[0], op[0]->value.f[1])", flags=horizontal_operation),
   operation("unpack_snorm_2x16", 1, printable_name="unpackSnorm2x16", source_types=(uint_type,), dest_type=float_type, c_expression="unpack_2x16(unpack_snorm_1x16, op[0]->value.u[0], &data.f[0], &data.f[1])", flags=frozenset((horizontal_operation, non_assign_operation))),
   operation("unpack_snorm_4x8", 1, printable_name="unpackSnorm4x8", source_types=(uint_type,), dest_type=float_type, c_expression="unpack_4x8(unpack_snorm_1x8, op[0]->value.u[0], &data.f[0], &data.f[1], &data.f[2], &data.f[3])", flags=frozenset((horizontal_operation, non_assign_operation))),
   operation("unpack_unorm_2x16", 1, printable_name="unpackUnorm2x16", source_types=(uint_type,), dest_type=float_type, c_expression="unpack_2x16(unpack_unorm_1x16, op[0]->value.u[0], &data.f[0], &data.f[1])", flags=frozenset((horizontal_operation, non_assign_operation))),
   operation("unpack_unorm_4x8", 1, printable_name="unpackUnorm4x8", source_types=(uint_type,), dest_type=float_type, c_expression="unpack_4x8(unpack_unorm_1x8, op[0]->value.u[0], &data.f[0], &data.f[1], &data.f[2], &data.f[3])", flags=frozenset((horizontal_operation, non_assign_operation))),
   operation("unpack_half_2x16", 1, printable_name="unpackHalf2x16", source_types=(uint_type,), dest_type=float_type, c_expression="unpack_2x16(unpack_half_1x16, op[0]->value.u[0], &data.f[0], &data.f[1])", flags=frozenset((horizontal_operation, non_assign_operation))),

   # Bit operations, part of ARB_gpu_shader5.
   operation("bitfield_reverse", 1, source_types=(uint_type, int_type), c_expression="bitfield_reverse({src0})"),
   operation("bit_count", 1, source_types=(uint_type, int_type), dest_type=int_type, c_expression="util_bitcount({src0})"),
   operation("find_msb", 1, source_types=(uint_type, int_type), dest_type=int_type, c_expression={'u': "find_msb_uint({src0})", 'i': "find_msb_int({src0})"}),
   operation("find_lsb", 1, source_types=(uint_type, int_type), dest_type=int_type, c_expression="find_msb_uint({src0} & -{src0})"),
   operation("clz", 1, source_types=(uint_type,), dest_type=uint_type, c_expression="(unsigned)(31 - find_msb_uint({src0}))"),

   operation("saturate", 1, printable_name="sat", source_types=(float_type,), c_expression="CLAMP({src0}, 0.0f, 1.0f)"),

   # Double packing, part of ARB_gpu_shader_fp64.
   operation("pack_double_2x32", 1, printable_name="packDouble2x32", source_types=(uint_type,), dest_type=double_type, c_expression="data.u64[0] = pack_2x32(op[0]->value.u[0], op[0]->value.u[1])", flags=frozenset((horizontal_operation, non_assign_operation))),
   operation("unpack_double_2x32", 1, printable_name="unpackDouble2x32", source_types=(double_type,), dest_type=uint_type, c_expression="unpack_2x32(op[0]->value.u64[0], &data.u[0], &data.u[1])", flags=frozenset((horizontal_operation, non_assign_operation))),

   # Sampler/Image packing, part of ARB_bindless_texture.
   operation("pack_sampler_2x32", 1, printable_name="packSampler2x32", source_types=(uint_type,), dest_type=uint64_type, c_expression="data.u64[0] = pack_2x32(op[0]->value.u[0], op[0]->value.u[1])", flags=frozenset((horizontal_operation, non_assign_operation))),
   operation("pack_image_2x32", 1, printable_name="packImage2x32", source_types=(uint_type,), dest_type=uint64_type, c_expression="data.u64[0] = pack_2x32(op[0]->value.u[0], op[0]->value.u[1])", flags=frozenset((horizontal_operation, non_assign_operation))),
   operation("unpack_sampler_2x32", 1, printable_name="unpackSampler2x32", source_types=(uint64_type,), dest_type=uint_type, c_expression="unpack_2x32(op[0]->value.u64[0], &data.u[0], &data.u[1])", flags=frozenset((horizontal_operation, non_assign_operation))),
   operation("unpack_image_2x32", 1, printable_name="unpackImage2x32", source_types=(uint64_type,), dest_type=uint_type, c_expression="unpack_2x32(op[0]->value.u64[0], &data.u[0], &data.u[1])", flags=frozenset((horizontal_operation, non_assign_operation))),

   operation("frexp_sig", 1),
   operation("frexp_exp", 1),

   operation("subroutine_to_int", 1),

   # Interpolate fs input at centroid
   #
   # operand0 is the fs input.
   operation("interpolate_at_centroid", 1),

   # Ask the driver for the total size of a buffer block.
   # operand0 is the ir_constant buffer block index in the linked shader.
   operation("get_buffer_size", 1),

   # Calculate length of an unsized array inside a buffer block.
   # This opcode is going to be replaced in a lowering pass inside
   # the linker.
   #
   # operand0 is the unsized array's ir_value for the calculation
   # of its length.
   operation("ssbo_unsized_array_length", 1),

   # Calculate length of an implicitly sized array.
   # This opcode is going to be replaced with a constant expression at link
   # time.
   operation("implicitly_sized_array_length", 1),

   # 64-bit integer packing ops.
   operation("pack_int_2x32", 1, printable_name="packInt2x32", source_types=(int_type,), dest_type=int64_type, c_expression="data.u64[0] = pack_2x32(op[0]->value.u[0], op[0]->value.u[1])", flags=frozenset((horizontal_operation, non_assign_operation))),
   operation("pack_uint_2x32", 1, printable_name="packUint2x32", source_types=(uint_type,), dest_type=uint64_type, c_expression="data.u64[0] = pack_2x32(op[0]->value.u[0], op[0]->value.u[1])", flags=frozenset((horizontal_operation, non_assign_operation))),
   operation("unpack_int_2x32", 1, printable_name="unpackInt2x32", source_types=(int64_type,), dest_type=int_type, c_expression="unpack_2x32(op[0]->value.u64[0], &data.u[0], &data.u[1])", flags=frozenset((horizontal_operation, non_assign_operation))),
   operation("unpack_uint_2x32", 1, printable_name="unpackUint2x32", source_types=(uint64_type,), dest_type=uint_type, c_expression="unpack_2x32(op[0]->value.u64[0], &data.u[0], &data.u[1])", flags=frozenset((horizontal_operation, non_assign_operation))),

   operation("add", 2, printable_name="+", source_types=numeric_types, c_expression="{src0} + {src1}", flags=vector_scalar_operation),
   operation("sub", 2, printable_name="-", source_types=numeric_types, c_expression="{src0} - {src1}", flags=vector_scalar_operation),
   operation("add_sat", 2, printable_name="add_sat", source_types=integer_types, c_expression={
      'u': "({src0} + {src1}) < {src0} ? UINT32_MAX : ({src0} + {src1})",
      'i': "iadd_saturate({src0}, {src1})",
      'u64': "({src0} + {src1}) < {src0} ? UINT64_MAX : ({src0} + {src1})",
      'i64': "iadd64_saturate({src0}, {src1})"
   }),
   operation("sub_sat", 2, printable_name="sub_sat", source_types=integer_types, c_expression={
      'u': "({src1} > {src0}) ? 0 : {src0} - {src1}",
      'i': "isub_saturate({src0}, {src1})",
      'u64': "({src1} > {src0}) ? 0 : {src0} - {src1}",
      'i64': "isub64_saturate({src0}, {src1})"
   }),
   operation("abs_sub", 2, printable_name="abs_sub", source_types=integer_types, c_expression={
      'u': "({src1} > {src0}) ? {src1} - {src0} : {src0} - {src1}",
      'i': "({src1} > {src0}) ? (unsigned){src1} - (unsigned){src0} : (unsigned){src0} - (unsigned){src1}",
      'u64': "({src1} > {src0}) ? {src1} - {src0} : {src0} - {src1}",
      'i64': "({src1} > {src0}) ? (uint64_t){src1} - (uint64_t){src0} : (uint64_t){src0} - (uint64_t){src1}",
   }),
   operation("avg", 2, printable_name="average", source_types=integer_types, c_expression="({src0} >> 1) + ({src1} >> 1) + (({src0} & {src1}) & 1)"),
   operation("avg_round", 2, printable_name="average_rounded", source_types=integer_types, c_expression="({src0} >> 1) + ({src1} >> 1) + (({src0} | {src1}) & 1)"),

   # "Floating-point or low 32-bit integer multiply."
   operation("mul", 2, printable_name="*", source_types=numeric_types, c_expression="{src0} * {src1}"),
   operation("mul_32x16", 2, printable_name="*", source_types=(uint_type, int_type), c_expression={
      'u': "{src0} * (uint16_t){src1}",
      'i': "{src0} * (int16_t){src0}"
   }),
   operation("imul_high", 2),       # Calculates the high 32-bits of a 64-bit multiply.
   operation("div", 2, printable_name="/", source_types=numeric_types, c_expression={'u': "{src1} == 0 ? 0 : {src0} / {src1}", 'i': "{src1} == 0 ? 0 : {src0} / {src1}", 'u64': "{src1} == 0 ? 0 : {src0} / {src1}", 'i64': "{src1} == 0 ? 0 : {src0} / {src1}", 'default': "{src0} / {src1}"}, flags=vector_scalar_operation),

   # Returns the carry resulting from the addition of the two arguments.
   operation("carry", 2),

   # Returns the borrow resulting from the subtraction of the second argument
   # from the first argument.
   operation("borrow", 2),

   # Either (vector % vector) or (vector % scalar)
   #
   # We don't use fmod because it rounds toward zero; GLSL specifies the use
   # of floor.
   operation("mod", 2, printable_name="%", source_types=numeric_types, c_expression={'u': "{src1} == 0 ? 0 : {src0} % {src1}", 'i': "{src1} == 0 ? 0 : {src0} % {src1}", 'f': "{src0} - {src1} * floorf({src0} / {src1})", 'd': "{src0} - {src1} * floor({src0} / {src1})", 'u64': "{src1} == 0 ? 0 : {src0} % {src1}", 'i64': "{src1} == 0 ? 0 : {src0} % {src1}"}, flags=vector_scalar_operation),

   # Binary comparison operators which return a boolean vector.
   # The type of both operands must be equal.
   operation("less", 2, printable_name="<", source_types=numeric_types, dest_type=bool_type, c_expression="{src0} < {src1}"),
   operation("gequal", 2, printable_name=">=", source_types=numeric_types, dest_type=bool_type, c_expression="{src0} >= {src1}"),
   operation("equal", 2, printable_name="==", source_types=all_types, dest_type=bool_type, c_expression="{src0} == {src1}"),
   operation("nequal", 2, printable_name="!=", source_types=all_types, dest_type=bool_type, c_expression="{src0} != {src1}"),

   # Returns single boolean for whether all components of operands[0]
   # equal the components of operands[1].
   operation("all_equal", 2, source_types=all_types, dest_type=bool_type, c_expression="op[0]->has_value(op[1])", flags=frozenset((horizontal_operation, types_identical_operation))),

   # Returns single boolean for whether any component of operands[0]
   # is not equal to the corresponding component of operands[1].
   operation("any_nequal", 2, source_types=all_types, dest_type=bool_type, c_expression="!op[0]->has_value(op[1])", flags=frozenset((horizontal_operation, types_identical_operation))),

   # Bit-wise binary operations.
   operation("lshift", 2, printable_name="<<", source_types=integer_types, c_expression="{src0} << {src1}", flags=frozenset((vector_scalar_operation, mixed_type_operation))),
   operation("rshift", 2, printable_name=">>", source_types=integer_types, c_expression="{src0} >> {src1}", flags=frozenset((vector_scalar_operation, mixed_type_operation))),
   operation("bit_and", 2, printable_name="&", source_types=integer_types, c_expression="{src0} & {src1}", flags=vector_scalar_operation),
   operation("bit_xor", 2, printable_name="^", source_types=integer_types, c_expression="{src0} ^ {src1}", flags=vector_scalar_operation),
   operation("bit_or", 2, printable_name="|", source_types=integer_types, c_expression="{src0} | {src1}", flags=vector_scalar_operation),

   operation("logic_and", 2, printable_name="&&", source_types=(bool_type,), c_expression="{src0} && {src1}"),
   operation("logic_xor", 2, printable_name="^^", source_types=(bool_type,), c_expression="{src0} != {src1}"),
   operation("logic_or", 2, printable_name="||", source_types=(bool_type,), c_expression="{src0} || {src1}"),

   operation("dot", 2, source_types=real_types, c_expression={'f': "dot_f(op[0], op[1])", 'd': "dot_d(op[0], op[1])"}, flags=horizontal_operation),
   operation("min", 2, source_types=numeric_types, c_expression="MIN2({src0}, {src1})", flags=vector_scalar_operation),
   operation("max", 2, source_types=numeric_types, c_expression="MAX2({src0}, {src1})", flags=vector_scalar_operation),

   operation("pow", 2, source_types=(float_type,), c_expression="powf({src0}, {src1})"),

   # Load a value the size of a given GLSL type from a uniform block.
   #
   # operand0 is the ir_constant uniform block index in the linked shader.
   # operand1 is a byte offset within the uniform block.
   operation("ubo_load", 2),

   # Multiplies a number by two to a power, part of ARB_gpu_shader5.
   operation("ldexp", 2,
             all_signatures=((float_type, (float_type, int_type)),
                             (double_type, (double_type, int_type))),
             c_expression={'f': "ldexpf_flush_subnormal({src0}, {src1})",
                           'd': "ldexp_flush_subnormal({src0}, {src1})"}),

   # Extract a scalar from a vector
   #
   # operand0 is the vector
   # operand1 is the index of the field to read from operand0
   operation("vector_extract", 2, source_types=all_types, c_expression="anything-except-None"),

   # Interpolate fs input at offset
   #
   # operand0 is the fs input
   # operand1 is the offset from the pixel center
   operation("interpolate_at_offset", 2),

   # Interpolate fs input at sample position
   #
   # operand0 is the fs input
   # operand1 is the sample ID
   operation("interpolate_at_sample", 2),

   operation("atan2", 2, source_types=(float_type,), c_expression="atan2({src0}, {src1})"),

   # Fused floating-point multiply-add, part of ARB_gpu_shader5.
   operation("fma", 3, source_types=real_types, c_expression="{src0} * {src1} + {src2}"),

   operation("lrp", 3, source_types=real_types, c_expression={'f': "{src0} * (1.0f - {src2}) + ({src1} * {src2})", 'd': "{src0} * (1.0 - {src2}) + ({src1} * {src2})"}),

   # Conditional Select
   #
   # A vector conditional select instruction (like ?:, but operating per-
   # component on vectors).
   #
   # See also lower_instructions_visitor::ldexp_to_arith
   operation("csel", 3,
             all_signatures=zip(all_types, zip(len(all_types) * (bool_type,), all_types, all_types)),
             c_expression="{src0} ? {src1} : {src2}"),

   operation("bitfield_extract", 3,
             all_signatures=((int_type, (uint_type, int_type, int_type)),
                             (int_type, (int_type, int_type, int_type))),
             c_expression={'u': "bitfield_extract_uint({src0}, {src1}, {src2})",
                           'i': "bitfield_extract_int({src0}, {src1}, {src2})"}),

   # Generate a value with one field of a vector changed
   #
   # operand0 is the vector
   # operand1 is the value to write into the vector result
   # operand2 is the index in operand0 to be modified
   operation("vector_insert", 3, source_types=all_types, c_expression="anything-except-None"),

   operation("bitfield_insert", 4,
             all_signatures=((uint_type, (uint_type, uint_type, int_type, int_type)),
                             (int_type, (int_type, int_type, int_type, int_type))),
             c_expression="bitfield_insert({src0}, {src1}, {src2}, {src3})"),

   operation("vector", 4, source_types=all_types, c_expression="anything-except-None"),
]


if __name__ == "__main__":
   copyright = """/*
 * Copyright (C) 2010 Intel Corporation
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
"""
   enum_template = mako.template.Template(copyright + """
enum ir_expression_operation {
% for item in values:
   ${item.get_enum_name()},
% endfor

   /* Sentinels marking the last of each kind of operation. */
% for item in lasts:
   ir_last_${("un", "bin", "tri", "quad")[item.num_operands - 1]}op = ${item.get_enum_name()},
% endfor
   ir_last_opcode = ir_quadop_${lasts[3].name}
};""")

   strings_template = mako.template.Template(copyright + """
const char *const ir_expression_operation_strings[] = {
% for item in values:
   "${item.printable_name}",
% endfor
};

const char *const ir_expression_operation_enum_strings[] = {
% for item in values:
   "${item.name}",
% endfor
};""")

   constant_template = mako.template.Template("""\
   switch (this->operation) {
% for op in values:
    % if op.c_expression is not None:
${op.get_template()}

    % endif
% endfor
   default:
      /* FINISHME: Should handle all expression types. */
      return NULL;
   }
""")

   if sys.argv[1] == "enum":
      lasts = [None, None, None, None]
      for item in reversed(ir_expression_operation):
         i = item.num_operands - 1
         if lasts[i] is None:
            lasts[i] = item

      print(enum_template.render(values=ir_expression_operation,
                                 lasts=lasts))
   elif sys.argv[1] == "strings":
      print(strings_template.render(values=ir_expression_operation))
   elif sys.argv[1] == "constant":
      print(constant_template.render(values=ir_expression_operation))
