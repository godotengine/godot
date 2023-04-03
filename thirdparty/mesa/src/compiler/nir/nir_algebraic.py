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

import ast
from collections import defaultdict
import itertools
import struct
import sys
import mako.template
import re
import traceback

from nir_opcodes import opcodes, type_sizes

# This should be the same as NIR_SEARCH_MAX_COMM_OPS in nir_search.c
nir_search_max_comm_ops = 8

# These opcodes are only employed by nir_search.  This provides a mapping from
# opcode to destination type.
conv_opcode_types = {
    'i2f' : 'float',
    'u2f' : 'float',
    'f2f' : 'float',
    'f2u' : 'uint',
    'f2i' : 'int',
    'u2u' : 'uint',
    'i2i' : 'int',
    'b2f' : 'float',
    'b2i' : 'int',
    'i2b' : 'bool',
    'f2b' : 'bool',
}

def get_cond_index(conds, cond):
    if cond:
        if cond in conds:
            return conds[cond]
        else:
            cond_index = len(conds)
            conds[cond] = cond_index
            return cond_index
    else:
        return -1

def get_c_opcode(op):
      if op in conv_opcode_types:
         return 'nir_search_op_' + op
      else:
         return 'nir_op_' + op

_type_re = re.compile(r"(?P<type>int|uint|bool|float)?(?P<bits>\d+)?")

def type_bits(type_str):
   m = _type_re.match(type_str)
   assert m.group('type')

   if m.group('bits') is None:
      return 0
   else:
      return int(m.group('bits'))

# Represents a set of variables, each with a unique id
class VarSet(object):
   def __init__(self):
      self.names = {}
      self.ids = itertools.count()
      self.immutable = False;

   def __getitem__(self, name):
      if name not in self.names:
         assert not self.immutable, "Unknown replacement variable: " + name
         self.names[name] = next(self.ids)

      return self.names[name]

   def lock(self):
      self.immutable = True

class SearchExpression(object):
   def __init__(self, expr):
      self.opcode = expr[0]
      self.sources = expr[1:]
      self.ignore_exact = False

   @staticmethod
   def create(val):
      if isinstance(val, tuple):
         return SearchExpression(val)
      else:
         assert(isinstance(val, SearchExpression))
         return val

   def __repr__(self):
      l = [self.opcode, *self.sources]
      if self.ignore_exact:
         l.append('ignore_exact')
      return repr((*l,))

class Value(object):
   @staticmethod
   def create(val, name_base, varset, algebraic_pass):
      if isinstance(val, bytes):
         val = val.decode('utf-8')

      if isinstance(val, tuple) or isinstance(val, SearchExpression):
         return Expression(val, name_base, varset, algebraic_pass)
      elif isinstance(val, Expression):
         return val
      elif isinstance(val, str):
         return Variable(val, name_base, varset, algebraic_pass)
      elif isinstance(val, (bool, float, int)):
         return Constant(val, name_base)

   def __init__(self, val, name, type_str):
      self.in_val = str(val)
      self.name = name
      self.type_str = type_str

   def __str__(self):
      return self.in_val

   def get_bit_size(self):
      """Get the physical bit-size that has been chosen for this value, or if
      there is none, the canonical value which currently represents this
      bit-size class. Variables will be preferred, i.e. if there are any
      variables in the equivalence class, the canonical value will be a
      variable. We do this since we'll need to know which variable each value
      is equivalent to when constructing the replacement expression. This is
      the "find" part of the union-find algorithm.
      """
      bit_size = self

      while isinstance(bit_size, Value):
         if bit_size._bit_size is None:
            break
         bit_size = bit_size._bit_size

      if bit_size is not self:
         self._bit_size = bit_size
      return bit_size

   def set_bit_size(self, other):
      """Make self.get_bit_size() return what other.get_bit_size() return
      before calling this, or just "other" if it's a concrete bit-size. This is
      the "union" part of the union-find algorithm.
      """

      self_bit_size = self.get_bit_size()
      other_bit_size = other if isinstance(other, int) else other.get_bit_size()

      if self_bit_size == other_bit_size:
         return

      self_bit_size._bit_size = other_bit_size

   @property
   def type_enum(self):
      return "nir_search_value_" + self.type_str

   @property
   def c_bit_size(self):
      bit_size = self.get_bit_size()
      if isinstance(bit_size, int):
         return bit_size
      elif isinstance(bit_size, Variable):
         return -bit_size.index - 1
      else:
         # If the bit-size class is neither a variable, nor an actual bit-size, then
         # - If it's in the search expression, we don't need to check anything
         # - If it's in the replace expression, either it's ambiguous (in which
         # case we'd reject it), or it equals the bit-size of the search value
         # We represent these cases with a 0 bit-size.
         return 0

   __template = mako.template.Template("""   { .${val.type_str} = {
      { ${val.type_enum}, ${val.c_bit_size} },
% if isinstance(val, Constant):
      ${val.type()}, { ${val.hex()} /* ${val.value} */ },
% elif isinstance(val, Variable):
      ${val.index}, /* ${val.var_name} */
      ${'true' if val.is_constant else 'false'},
      ${val.type() or 'nir_type_invalid' },
      ${val.cond_index},
      ${val.swizzle()},
% elif isinstance(val, Expression):
      ${'true' if val.inexact else 'false'},
      ${'true' if val.exact else 'false'},
      ${'true' if val.ignore_exact else 'false'},
      ${val.c_opcode()},
      ${val.comm_expr_idx}, ${val.comm_exprs},
      { ${', '.join(src.array_index for src in val.sources)} },
      ${val.cond_index},
% endif
   } },
""")

   def render(self, cache):
      struct_init = self.__template.render(val=self,
                                           Constant=Constant,
                                           Variable=Variable,
                                           Expression=Expression)
      if struct_init in cache:
         # If it's in the cache, register a name remap in the cache and render
         # only a comment saying it's been remapped
         self.array_index = cache[struct_init]
         return "   /* {} -> {} in the cache */\n".format(self.name,
                                                       cache[struct_init])
      else:
         self.array_index = str(cache["next_index"])
         cache[struct_init] = self.array_index
         cache["next_index"] += 1
         return struct_init

_constant_re = re.compile(r"(?P<value>[^@\(]+)(?:@(?P<bits>\d+))?")

class Constant(Value):
   def __init__(self, val, name):
      Value.__init__(self, val, name, "constant")

      if isinstance(val, (str)):
         m = _constant_re.match(val)
         self.value = ast.literal_eval(m.group('value'))
         self._bit_size = int(m.group('bits')) if m.group('bits') else None
      else:
         self.value = val
         self._bit_size = None

      if isinstance(self.value, bool):
         assert self._bit_size is None or self._bit_size == 1
         self._bit_size = 1

   def hex(self):
      if isinstance(self.value, (bool)):
         return 'NIR_TRUE' if self.value else 'NIR_FALSE'
      if isinstance(self.value, int):
         return hex(self.value)
      elif isinstance(self.value, float):
         return hex(struct.unpack('Q', struct.pack('d', self.value))[0])
      else:
         assert False

   def type(self):
      if isinstance(self.value, (bool)):
         return "nir_type_bool"
      elif isinstance(self.value, int):
         return "nir_type_int"
      elif isinstance(self.value, float):
         return "nir_type_float"

   def equivalent(self, other):
      """Check that two constants are equivalent.

      This is check is much weaker than equality.  One generally cannot be
      used in place of the other.  Using this implementation for the __eq__
      will break BitSizeValidator.

      """
      if not isinstance(other, type(self)):
         return False

      return self.value == other.value

# The $ at the end forces there to be an error if any part of the string
# doesn't match one of the field patterns.
_var_name_re = re.compile(r"(?P<const>#)?(?P<name>\w+)"
                          r"(?:@(?P<type>int|uint|bool|float)?(?P<bits>\d+)?)?"
                          r"(?P<cond>\([^\)]+\))?"
                          r"(?P<swiz>\.[xyzwabcdefghijklmnop]+)?"
                          r"$")

class Variable(Value):
   def __init__(self, val, name, varset, algebraic_pass):
      Value.__init__(self, val, name, "variable")

      m = _var_name_re.match(val)
      assert m and m.group('name') is not None, \
            "Malformed variable name \"{}\".".format(val)

      self.var_name = m.group('name')

      # Prevent common cases where someone puts quotes around a literal
      # constant.  If we want to support names that have numeric or
      # punctuation characters, we can me the first assertion more flexible.
      assert self.var_name.isalpha()
      assert self.var_name != 'True'
      assert self.var_name != 'False'

      self.is_constant = m.group('const') is not None
      self.cond_index = get_cond_index(algebraic_pass.variable_cond, m.group('cond'))
      self.required_type = m.group('type')
      self._bit_size = int(m.group('bits')) if m.group('bits') else None
      self.swiz = m.group('swiz')

      if self.required_type == 'bool':
         if self._bit_size is not None:
            assert self._bit_size in type_sizes(self.required_type)
         else:
            self._bit_size = 1

      if self.required_type is not None:
         assert self.required_type in ('float', 'bool', 'int', 'uint')

      self.index = varset[self.var_name]

   def type(self):
      if self.required_type == 'bool':
         return "nir_type_bool"
      elif self.required_type in ('int', 'uint'):
         return "nir_type_int"
      elif self.required_type == 'float':
         return "nir_type_float"

   def equivalent(self, other):
      """Check that two variables are equivalent.

      This is check is much weaker than equality.  One generally cannot be
      used in place of the other.  Using this implementation for the __eq__
      will break BitSizeValidator.

      """
      if not isinstance(other, type(self)):
         return False

      return self.index == other.index

   def swizzle(self):
      if self.swiz is not None:
         swizzles = {'x' : 0, 'y' : 1, 'z' : 2, 'w' : 3,
                     'a' : 0, 'b' : 1, 'c' : 2, 'd' : 3,
                     'e' : 4, 'f' : 5, 'g' : 6, 'h' : 7,
                     'i' : 8, 'j' : 9, 'k' : 10, 'l' : 11,
                     'm' : 12, 'n' : 13, 'o' : 14, 'p' : 15 }
         return '{' + ', '.join([str(swizzles[c]) for c in self.swiz[1:]]) + '}'
      return '{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}'

_opcode_re = re.compile(r"(?P<inexact>~)?(?P<exact>!)?(?P<opcode>\w+)(?:@(?P<bits>\d+))?"
                        r"(?P<cond>\([^\)]+\))?")

class Expression(Value):
   def __init__(self, expr, name_base, varset, algebraic_pass):
      Value.__init__(self, expr, name_base, "expression")

      expr = SearchExpression.create(expr)

      m = _opcode_re.match(expr.opcode)
      assert m and m.group('opcode') is not None

      self.opcode = m.group('opcode')
      self._bit_size = int(m.group('bits')) if m.group('bits') else None
      self.inexact = m.group('inexact') is not None
      self.exact = m.group('exact') is not None
      self.ignore_exact = expr.ignore_exact
      self.cond = m.group('cond')

      assert not self.inexact or not self.exact, \
            'Expression cannot be both exact and inexact.'

      # "many-comm-expr" isn't really a condition.  It's notification to the
      # generator that this pattern is known to have too many commutative
      # expressions, and an error should not be generated for this case.
      self.many_commutative_expressions = False
      if self.cond and self.cond.find("many-comm-expr") >= 0:
         # Split the condition into a comma-separated list.  Remove
         # "many-comm-expr".  If there is anything left, put it back together.
         c = self.cond[1:-1].split(",")
         c.remove("many-comm-expr")
         assert(len(c) <= 1)

         self.cond = c[0] if c else None
         self.many_commutative_expressions = True

      # Deduplicate references to the condition functions for the expressions
      # and save the index for the order they were added.
      self.cond_index = get_cond_index(algebraic_pass.expression_cond, self.cond)

      self.sources = [ Value.create(src, "{0}_{1}".format(name_base, i), varset, algebraic_pass)
                       for (i, src) in enumerate(expr.sources) ]

      # nir_search_expression::srcs is hard-coded to 4
      assert len(self.sources) <= 4

      if self.opcode in conv_opcode_types:
         assert self._bit_size is None, \
                'Expression cannot use an unsized conversion opcode with ' \
                'an explicit size; that\'s silly.'

      self.__index_comm_exprs(0)

   def equivalent(self, other):
      """Check that two variables are equivalent.

      This is check is much weaker than equality.  One generally cannot be
      used in place of the other.  Using this implementation for the __eq__
      will break BitSizeValidator.

      This implementation does not check for equivalence due to commutativity,
      but it could.

      """
      if not isinstance(other, type(self)):
         return False

      if len(self.sources) != len(other.sources):
         return False

      if self.opcode != other.opcode:
         return False

      return all(s.equivalent(o) for s, o in zip(self.sources, other.sources))

   def __index_comm_exprs(self, base_idx):
      """Recursively count and index commutative expressions
      """
      self.comm_exprs = 0

      # A note about the explicit "len(self.sources)" check. The list of
      # sources comes from user input, and that input might be bad.  Check
      # that the expected second source exists before accessing it. Without
      # this check, a unit test that does "('iadd', 'a')" will crash.
      if self.opcode not in conv_opcode_types and \
         "2src_commutative" in opcodes[self.opcode].algebraic_properties and \
         len(self.sources) >= 2 and \
         not self.sources[0].equivalent(self.sources[1]):
         self.comm_expr_idx = base_idx
         self.comm_exprs += 1
      else:
         self.comm_expr_idx = -1

      for s in self.sources:
         if isinstance(s, Expression):
            s.__index_comm_exprs(base_idx + self.comm_exprs)
            self.comm_exprs += s.comm_exprs

      return self.comm_exprs

   def c_opcode(self):
      return get_c_opcode(self.opcode)

   def render(self, cache):
      srcs = "".join(src.render(cache) for src in self.sources)
      return srcs + super(Expression, self).render(cache)

class BitSizeValidator(object):
   """A class for validating bit sizes of expressions.

   NIR supports multiple bit-sizes on expressions in order to handle things
   such as fp64.  The source and destination of every ALU operation is
   assigned a type and that type may or may not specify a bit size.  Sources
   and destinations whose type does not specify a bit size are considered
   "unsized" and automatically take on the bit size of the corresponding
   register or SSA value.  NIR has two simple rules for bit sizes that are
   validated by nir_validator:

    1) A given SSA def or register has a single bit size that is respected by
       everything that reads from it or writes to it.

    2) The bit sizes of all unsized inputs/outputs on any given ALU
       instruction must match.  They need not match the sized inputs or
       outputs but they must match each other.

   In order to keep nir_algebraic relatively simple and easy-to-use,
   nir_search supports a type of bit-size inference based on the two rules
   above.  This is similar to type inference in many common programming
   languages.  If, for instance, you are constructing an add operation and you
   know the second source is 16-bit, then you know that the other source and
   the destination must also be 16-bit.  There are, however, cases where this
   inference can be ambiguous or contradictory.  Consider, for instance, the
   following transformation:

   (('usub_borrow', a, b), ('b2i@32', ('ult', a, b)))

   This transformation can potentially cause a problem because usub_borrow is
   well-defined for any bit-size of integer.  However, b2i always generates a
   32-bit result so it could end up replacing a 64-bit expression with one
   that takes two 64-bit values and produces a 32-bit value.  As another
   example, consider this expression:

   (('bcsel', a, b, 0), ('iand', a, b))

   In this case, in the search expression a must be 32-bit but b can
   potentially have any bit size.  If we had a 64-bit b value, we would end up
   trying to and a 32-bit value with a 64-bit value which would be invalid

   This class solves that problem by providing a validation layer that proves
   that a given search-and-replace operation is 100% well-defined before we
   generate any code.  This ensures that bugs are caught at compile time
   rather than at run time.

   Each value maintains a "bit-size class", which is either an actual bit size
   or an equivalence class with other values that must have the same bit size.
   The validator works by combining bit-size classes with each other according
   to the NIR rules outlined above, checking that there are no inconsistencies.
   When doing this for the replacement expression, we make sure to never change
   the equivalence class of any of the search values. We could make the example
   transforms above work by doing some extra run-time checking of the search
   expression, but we make the user specify those constraints themselves, to
   avoid any surprises. Since the replacement bitsizes can only be connected to
   the source bitsize via variables (variables must have the same bitsize in
   the source and replacment expressions) or the roots of the expression (the
   replacement expression must produce the same bit size as the search
   expression), we prevent merging a variable with anything when processing the
   replacement expression, or specializing the search bitsize
   with anything. The former prevents

   (('bcsel', a, b, 0), ('iand', a, b))

   from being allowed, since we'd have to merge the bitsizes for a and b due to
   the 'iand', while the latter prevents

   (('usub_borrow', a, b), ('b2i@32', ('ult', a, b)))

   from being allowed, since the search expression has the bit size of a and b,
   which can't be specialized to 32 which is the bitsize of the replace
   expression. It also prevents something like:

   (('b2i', ('i2b', a)), ('ineq', a, 0))

   since the bitsize of 'b2i', which can be anything, can't be specialized to
   the bitsize of a.

   After doing all this, we check that every subexpression of the replacement
   was assigned a constant bitsize, the bitsize of a variable, or the bitsize
   of the search expresssion, since those are the things that are known when
   constructing the replacement expresssion. Finally, we record the bitsize
   needed in nir_search_value so that we know what to do when building the
   replacement expression.
   """

   def __init__(self, varset):
      self._var_classes = [None] * len(varset.names)

   def compare_bitsizes(self, a, b):
      """Determines which bitsize class is a specialization of the other, or
      whether neither is. When we merge two different bitsizes, the
      less-specialized bitsize always points to the more-specialized one, so
      that calling get_bit_size() always gets you the most specialized bitsize.
      The specialization partial order is given by:
      - Physical bitsizes are always the most specialized, and a different
        bitsize can never specialize another.
      - In the search expression, variables can always be specialized to each
        other and to physical bitsizes. In the replace expression, we disallow
        this to avoid adding extra constraints to the search expression that
        the user didn't specify.
      - Expressions and constants without a bitsize can always be specialized to
        each other and variables, but not the other way around.

        We return -1 if a <= b (b can be specialized to a), 0 if a = b, 1 if a >= b,
        and None if they are not comparable (neither a <= b nor b <= a).
      """
      if isinstance(a, int):
         if isinstance(b, int):
            return 0 if a == b else None
         elif isinstance(b, Variable):
            return -1 if self.is_search else None
         else:
            return -1
      elif isinstance(a, Variable):
         if isinstance(b, int):
            return 1 if self.is_search else None
         elif isinstance(b, Variable):
            return 0 if self.is_search or a.index == b.index else None
         else:
            return -1
      else:
         if isinstance(b, int):
            return 1
         elif isinstance(b, Variable):
            return 1
         else:
            return 0

   def unify_bit_size(self, a, b, error_msg):
      """Record that a must have the same bit-size as b. If both
      have been assigned conflicting physical bit-sizes, call "error_msg" with
      the bit-sizes of self and other to get a message and raise an error.
      In the replace expression, disallow merging variables with other
      variables and physical bit-sizes as well.
      """
      a_bit_size = a.get_bit_size()
      b_bit_size = b if isinstance(b, int) else b.get_bit_size()

      cmp_result = self.compare_bitsizes(a_bit_size, b_bit_size)

      assert cmp_result is not None, \
         error_msg(a_bit_size, b_bit_size)

      if cmp_result < 0:
         b_bit_size.set_bit_size(a)
      elif not isinstance(a_bit_size, int):
         a_bit_size.set_bit_size(b)

   def merge_variables(self, val):
      """Perform the first part of type inference by merging all the different
      uses of the same variable. We always do this as if we're in the search
      expression, even if we're actually not, since otherwise we'd get errors
      if the search expression specified some constraint but the replace
      expression didn't, because we'd be merging a variable and a constant.
      """
      if isinstance(val, Variable):
         if self._var_classes[val.index] is None:
            self._var_classes[val.index] = val
         else:
            other = self._var_classes[val.index]
            self.unify_bit_size(other, val,
                  lambda other_bit_size, bit_size:
                     'Variable {} has conflicting bit size requirements: ' \
                     'it must have bit size {} and {}'.format(
                        val.var_name, other_bit_size, bit_size))
      elif isinstance(val, Expression):
         for src in val.sources:
            self.merge_variables(src)

   def validate_value(self, val):
      """Validate the an expression by performing classic Hindley-Milner
      type inference on bitsizes. This will detect if there are any conflicting
      requirements, and unify variables so that we know which variables must
      have the same bitsize. If we're operating on the replace expression, we
      will refuse to merge different variables together or merge a variable
      with a constant, in order to prevent surprises due to rules unexpectedly
      not matching at runtime.
      """
      if not isinstance(val, Expression):
         return

      # Generic conversion ops are special in that they have a single unsized
      # source and an unsized destination and the two don't have to match.
      # This means there's no validation or unioning to do here besides the
      # len(val.sources) check.
      if val.opcode in conv_opcode_types:
         assert len(val.sources) == 1, \
            "Expression {} has {} sources, expected 1".format(
               val, len(val.sources))
         self.validate_value(val.sources[0])
         return

      nir_op = opcodes[val.opcode]
      assert len(val.sources) == nir_op.num_inputs, \
         "Expression {} has {} sources, expected {}".format(
            val, len(val.sources), nir_op.num_inputs)

      for src in val.sources:
         self.validate_value(src)

      dst_type_bits = type_bits(nir_op.output_type)

      # First, unify all the sources. That way, an error coming up because two
      # sources have an incompatible bit-size won't produce an error message
      # involving the destination.
      first_unsized_src = None
      for src_type, src in zip(nir_op.input_types, val.sources):
         src_type_bits = type_bits(src_type)
         if src_type_bits == 0:
            if first_unsized_src is None:
               first_unsized_src = src
               continue

            if self.is_search:
               self.unify_bit_size(first_unsized_src, src,
                  lambda first_unsized_src_bit_size, src_bit_size:
                     'Source {} of {} must have bit size {}, while source {} ' \
                     'must have incompatible bit size {}'.format(
                        first_unsized_src, val, first_unsized_src_bit_size,
                        src, src_bit_size))
            else:
               self.unify_bit_size(first_unsized_src, src,
                  lambda first_unsized_src_bit_size, src_bit_size:
                     'Sources {} (bit size of {}) and {} (bit size of {}) ' \
                     'of {} may not have the same bit size when building the ' \
                     'replacement expression.'.format(
                        first_unsized_src, first_unsized_src_bit_size, src,
                        src_bit_size, val))
         else:
            if self.is_search:
               self.unify_bit_size(src, src_type_bits,
                  lambda src_bit_size, unused:
                     '{} must have {} bits, but as a source of nir_op_{} '\
                     'it must have {} bits'.format(
                        src, src_bit_size, nir_op.name, src_type_bits))
            else:
               self.unify_bit_size(src, src_type_bits,
                  lambda src_bit_size, unused:
                     '{} has the bit size of {}, but as a source of ' \
                     'nir_op_{} it must have {} bits, which may not be the ' \
                     'same'.format(
                        src, src_bit_size, nir_op.name, src_type_bits))

      if dst_type_bits == 0:
         if first_unsized_src is not None:
            if self.is_search:
               self.unify_bit_size(val, first_unsized_src,
                  lambda val_bit_size, src_bit_size:
                     '{} must have the bit size of {}, while its source {} ' \
                     'must have incompatible bit size {}'.format(
                        val, val_bit_size, first_unsized_src, src_bit_size))
            else:
               self.unify_bit_size(val, first_unsized_src,
                  lambda val_bit_size, src_bit_size:
                     '{} must have {} bits, but its source {} ' \
                     '(bit size of {}) may not have that bit size ' \
                     'when building the replacement.'.format(
                        val, val_bit_size, first_unsized_src, src_bit_size))
      else:
         self.unify_bit_size(val, dst_type_bits,
            lambda dst_bit_size, unused:
               '{} must have {} bits, but as a destination of nir_op_{} ' \
               'it must have {} bits'.format(
                  val, dst_bit_size, nir_op.name, dst_type_bits))

   def validate_replace(self, val, search):
      bit_size = val.get_bit_size()
      assert isinstance(bit_size, int) or isinstance(bit_size, Variable) or \
            bit_size == search.get_bit_size(), \
            'Ambiguous bit size for replacement value {}: ' \
            'it cannot be deduced from a variable, a fixed bit size ' \
            'somewhere, or the search expression.'.format(val)

      if isinstance(val, Expression):
         for src in val.sources:
            self.validate_replace(src, search)
      elif isinstance(val, Variable):
          # These catch problems when someone copies and pastes the search
          # into the replacement.
          assert not val.is_constant, \
              'Replacement variables must not be marked constant.'

          assert val.cond_index == -1, \
              'Replacement variables must not have a condition.'

          assert not val.required_type, \
              'Replacement variables must not have a required type.'

   def validate(self, search, replace):
      self.is_search = True
      self.merge_variables(search)
      self.merge_variables(replace)
      self.validate_value(search)

      self.is_search = False
      self.validate_value(replace)

      # Check that search is always more specialized than replace. Note that
      # we're doing this in replace mode, disallowing merging variables.
      search_bit_size = search.get_bit_size()
      replace_bit_size = replace.get_bit_size()
      cmp_result = self.compare_bitsizes(search_bit_size, replace_bit_size)

      assert cmp_result is not None and cmp_result <= 0, \
         'The search expression bit size {} and replace expression ' \
         'bit size {} may not be the same'.format(
               search_bit_size, replace_bit_size)

      replace.set_bit_size(search)

      self.validate_replace(replace, search)

_optimization_ids = itertools.count()

condition_list = ['true']

class SearchAndReplace(object):
   def __init__(self, transform, algebraic_pass):
      self.id = next(_optimization_ids)

      search = transform[0]
      replace = transform[1]
      if len(transform) > 2:
         self.condition = transform[2]
      else:
         self.condition = 'true'

      if self.condition not in condition_list:
         condition_list.append(self.condition)
      self.condition_index = condition_list.index(self.condition)

      varset = VarSet()
      if isinstance(search, Expression):
         self.search = search
      else:
         self.search = Expression(search, "search{0}".format(self.id), varset, algebraic_pass)

      varset.lock()

      if isinstance(replace, Value):
         self.replace = replace
      else:
         self.replace = Value.create(replace, "replace{0}".format(self.id), varset, algebraic_pass)

      BitSizeValidator(varset).validate(self.search, self.replace)

class TreeAutomaton(object):
   """This class calculates a bottom-up tree automaton to quickly search for
   the left-hand sides of tranforms. Tree automatons are a generalization of
   classical NFA's and DFA's, where the transition function determines the
   state of the parent node based on the state of its children. We construct a
   deterministic automaton to match patterns, using a similar algorithm to the
   classical NFA to DFA construction. At the moment, it only matches opcodes
   and constants (without checking the actual value), leaving more detailed
   checking to the search function which actually checks the leaves. The
   automaton acts as a quick filter for the search function, requiring only n
   + 1 table lookups for each n-source operation. The implementation is based
   on the theory described in "Tree Automatons: Two Taxonomies and a Toolkit."
   In the language of that reference, this is a frontier-to-root deterministic
   automaton using only symbol filtering. The filtering is crucial to reduce
   both the time taken to generate the tables and the size of the tables.
   """
   def __init__(self, transforms):
      self.patterns = [t.search for t in transforms]
      self._compute_items()
      self._build_table()
      #print('num items: {}'.format(len(set(self.items.values()))))
      #print('num states: {}'.format(len(self.states)))
      #for state, patterns in zip(self.states, self.patterns):
      #   print('{}: num patterns: {}'.format(state, len(patterns)))

   class IndexMap(object):
      """An indexed list of objects, where one can either lookup an object by
      index or find the index associated to an object quickly using a hash
      table. Compared to a list, it has a constant time index(). Compared to a
      set, it provides a stable iteration order.
      """
      def __init__(self, iterable=()):
         self.objects = []
         self.map = {}
         for obj in iterable:
            self.add(obj)

      def __getitem__(self, i):
         return self.objects[i]

      def __contains__(self, obj):
         return obj in self.map

      def __len__(self):
         return len(self.objects)

      def __iter__(self):
         return iter(self.objects)

      def clear(self):
         self.objects = []
         self.map.clear()

      def index(self, obj):
         return self.map[obj]

      def add(self, obj):
         if obj in self.map:
            return self.map[obj]
         else:
            index = len(self.objects)
            self.objects.append(obj)
            self.map[obj] = index
            return index

      def __repr__(self):
         return 'IndexMap([' + ', '.join(repr(e) for e in self.objects) + '])'

   class Item(object):
      """This represents an "item" in the language of "Tree Automatons." This
      is just a subtree of some pattern, which represents a potential partial
      match at runtime. We deduplicate them, so that identical subtrees of
      different patterns share the same object, and store some extra
      information needed for the main algorithm as well.
      """
      def __init__(self, opcode, children):
         self.opcode = opcode
         self.children = children
         # These are the indices of patterns for which this item is the root node.
         self.patterns = []
         # This the set of opcodes for parents of this item. Used to speed up
         # filtering.
         self.parent_ops = set()

      def __str__(self):
         return '(' + ', '.join([self.opcode] + [str(c) for c in self.children]) + ')'

      def __repr__(self):
         return str(self)

   def _compute_items(self):
      """Build a set of all possible items, deduplicating them."""
      # This is a map from (opcode, sources) to item.
      self.items = {}

      # The set of all opcodes used by the patterns. Used later to avoid
      # building and emitting all the tables for opcodes that aren't used.
      self.opcodes = self.IndexMap()

      def get_item(opcode, children, pattern=None):
         commutative = len(children) >= 2 \
               and "2src_commutative" in opcodes[opcode].algebraic_properties
         item = self.items.setdefault((opcode, children),
                                      self.Item(opcode, children))
         if commutative:
            self.items[opcode, (children[1], children[0]) + children[2:]] = item
         if pattern is not None:
            item.patterns.append(pattern)
         return item

      self.wildcard = get_item("__wildcard", ())
      self.const = get_item("__const", ())

      def process_subpattern(src, pattern=None):
         if isinstance(src, Constant):
            # Note: we throw away the actual constant value!
            return self.const
         elif isinstance(src, Variable):
            if src.is_constant:
               return self.const
            else:
               # Note: we throw away which variable it is here! This special
               # item is equivalent to nu in "Tree Automatons."
               return self.wildcard
         else:
            assert isinstance(src, Expression)
            opcode = src.opcode
            stripped = opcode.rstrip('0123456789')
            if stripped in conv_opcode_types:
               # Matches that use conversion opcodes with a specific type,
               # like f2b1, are tricky.  Either we construct the automaton to
               # match specific NIR opcodes like nir_op_f2b1, in which case we
               # need to create separate items for each possible NIR opcode
               # for patterns that have a generic opcode like f2b, or we
               # construct it to match the search opcode, in which case we
               # need to map f2b1 to f2b when constructing the automaton. Here
               # we do the latter.
               opcode = stripped
            self.opcodes.add(opcode)
            children = tuple(process_subpattern(c) for c in src.sources)
            item = get_item(opcode, children, pattern)
            for i, child in enumerate(children):
               child.parent_ops.add(opcode)
            return item

      for i, pattern in enumerate(self.patterns):
         process_subpattern(pattern, i)

   def _build_table(self):
      """This is the core algorithm which builds up the transition table. It
      is based off of Algorithm 5.7.38 "Reachability-based tabulation of Cl .
      Comp_a and Filt_{a,i} using integers to identify match sets." It
      simultaneously builds up a list of all possible "match sets" or
      "states", where each match set represents the set of Item's that match a
      given instruction, and builds up the transition table between states.
      """
      # Map from opcode + filtered state indices to transitioned state.
      self.table = defaultdict(dict)
      # Bijection from state to index. q in the original algorithm is
      # len(self.states)
      self.states = self.IndexMap()
      # Lists of pattern matches separated by None
      self.state_patterns = [None]
      # Offset in the ->transforms table for each state index
      self.state_pattern_offsets = []
      # Map from state index to filtered state index for each opcode.
      self.filter = defaultdict(list)
      # Bijections from filtered state to filtered state index for each
      # opcode, called the "representor sets" in the original algorithm.
      # q_{a,j} in the original algorithm is len(self.rep[op]).
      self.rep = defaultdict(self.IndexMap)

      # Everything in self.states with a index at least worklist_index is part
      # of the worklist of newly created states. There is also a worklist of
      # newly fitered states for each opcode, for which worklist_indices
      # serves a similar purpose. worklist_index corresponds to p in the
      # original algorithm, while worklist_indices is p_{a,j} (although since
      # we only filter by opcode/symbol, it's really just p_a).
      self.worklist_index = 0
      worklist_indices = defaultdict(lambda: 0)

      # This is the set of opcodes for which the filtered worklist is non-empty.
      # It's used to avoid scanning opcodes for which there is nothing to
      # process when building the transition table. It corresponds to new_a in
      # the original algorithm.
      new_opcodes = self.IndexMap()

      # Process states on the global worklist, filtering them for each opcode,
      # updating the filter tables, and updating the filtered worklists if any
      # new filtered states are found. Similar to ComputeRepresenterSets() in
      # the original algorithm, although that only processes a single state.
      def process_new_states():
         while self.worklist_index < len(self.states):
            state = self.states[self.worklist_index]
            # Calculate pattern matches for this state. Each pattern is
            # assigned to a unique item, so we don't have to worry about
            # deduplicating them here. However, we do have to sort them so
            # that they're visited at runtime in the order they're specified
            # in the source.
            patterns = list(sorted(p for item in state for p in item.patterns))

            if patterns:
                # Add our patterns to the global table.
                self.state_pattern_offsets.append(len(self.state_patterns))
                self.state_patterns.extend(patterns)
                self.state_patterns.append(None)
            else:
                # Point to the initial sentinel in the global table.
                self.state_pattern_offsets.append(0)

            # calculate filter table for this state, and update filtered
            # worklists.
            for op in self.opcodes:
               filt = self.filter[op]
               rep = self.rep[op]
               filtered = frozenset(item for item in state if \
                  op in item.parent_ops)
               if filtered in rep:
                  rep_index = rep.index(filtered)
               else:
                  rep_index = rep.add(filtered)
                  new_opcodes.add(op)
               assert len(filt) == self.worklist_index
               filt.append(rep_index)
            self.worklist_index += 1

      # There are two start states: one which can only match as a wildcard,
      # and one which can match as a wildcard or constant. These will be the
      # states of intrinsics/other instructions and load_const instructions,
      # respectively. The indices of these must match the definitions of
      # WILDCARD_STATE and CONST_STATE below, so that the runtime C code can
      # initialize things correctly.
      self.states.add(frozenset((self.wildcard,)))
      self.states.add(frozenset((self.const,self.wildcard)))
      process_new_states()

      while len(new_opcodes) > 0:
         for op in new_opcodes:
            rep = self.rep[op]
            table = self.table[op]
            op_worklist_index = worklist_indices[op]
            if op in conv_opcode_types:
               num_srcs = 1
            else:
               num_srcs = opcodes[op].num_inputs

            # Iterate over all possible source combinations where at least one
            # is on the worklist.
            for src_indices in itertools.product(range(len(rep)), repeat=num_srcs):
               if all(src_idx < op_worklist_index for src_idx in src_indices):
                  continue

               srcs = tuple(rep[src_idx] for src_idx in src_indices)

               # Try all possible pairings of source items and add the
               # corresponding parent items. This is Comp_a from the paper.
               parent = set(self.items[op, item_srcs] for item_srcs in
                  itertools.product(*srcs) if (op, item_srcs) in self.items)

               # We could always start matching something else with a
               # wildcard. This is Cl from the paper.
               parent.add(self.wildcard)

               table[src_indices] = self.states.add(frozenset(parent))
            worklist_indices[op] = len(rep)
         new_opcodes.clear()
         process_new_states()

_algebraic_pass_template = mako.template.Template("""
#include "nir.h"
#include "nir_builder.h"
#include "nir_search.h"
#include "nir_search_helpers.h"

/* What follows is NIR algebraic transform code for the following ${len(xforms)}
 * transforms:
% for xform in xforms:
 *    ${xform.search} => ${xform.replace}
% endfor
 */

<% cache = {"next_index": 0} %>
static const nir_search_value_union ${pass_name}_values[] = {
% for xform in xforms:
   /* ${xform.search} => ${xform.replace} */
${xform.search.render(cache)}
${xform.replace.render(cache)}
% endfor
};

% if expression_cond:
static const nir_search_expression_cond ${pass_name}_expression_cond[] = {
% for cond in expression_cond:
   ${cond[0]},
% endfor
};
% endif

% if variable_cond:
static const nir_search_variable_cond ${pass_name}_variable_cond[] = {
% for cond in variable_cond:
   ${cond[0]},
% endfor
};
% endif

static const struct transform ${pass_name}_transforms[] = {
% for i in automaton.state_patterns:
% if i is not None:
   { ${xforms[i].search.array_index}, ${xforms[i].replace.array_index}, ${xforms[i].condition_index} },
% else:
   { ~0, ~0, ~0 }, /* Sentinel */

% endif
% endfor
};

static const struct per_op_table ${pass_name}_pass_op_table[nir_num_search_ops] = {
% for op in automaton.opcodes:
   [${get_c_opcode(op)}] = {
% if all(e == 0 for e in automaton.filter[op]):
      .filter = NULL,
% else:
      .filter = (const uint16_t []) {
      % for e in automaton.filter[op]:
         ${e},
      % endfor
      },
% endif
      <%
        num_filtered = len(automaton.rep[op])
      %>
      .num_filtered_states = ${num_filtered},
      .table = (const uint16_t []) {
      <%
        num_srcs = len(next(iter(automaton.table[op])))
      %>
      % for indices in itertools.product(range(num_filtered), repeat=num_srcs):
         ${automaton.table[op][indices]},
      % endfor
      },
   },
% endfor
};

/* Mapping from state index to offset in transforms (0 being no transforms) */
static const uint16_t ${pass_name}_transform_offsets[] = {
% for offset in automaton.state_pattern_offsets:
   ${offset},
% endfor
};

static const nir_algebraic_table ${pass_name}_table = {
   .transforms = ${pass_name}_transforms,
   .transform_offsets = ${pass_name}_transform_offsets,
   .pass_op_table = ${pass_name}_pass_op_table,
   .values = ${pass_name}_values,
   .expression_cond = ${ pass_name + "_expression_cond" if expression_cond else "NULL" },
   .variable_cond = ${ pass_name + "_variable_cond" if variable_cond else "NULL" },
};

bool
${pass_name}(nir_shader *shader)
{
   bool progress = false;
   bool condition_flags[${len(condition_list)}];
   const nir_shader_compiler_options *options = shader->options;
   const shader_info *info = &shader->info;
   (void) options;
   (void) info;

   /* This is not a great place for this, but it seems to be the best place
    * for it. Check that at most one kind of lowering is requested for
    * bitfield extract and bitfield insert. Otherwise the lowering can fight
    * with each other and optimizations.
    */
   assert((int)options->lower_bitfield_extract +
          (int)options->lower_bitfield_extract_to_shifts <= 1);
   assert((int)options->lower_bitfield_insert +
          (int)options->lower_bitfield_insert_to_shifts +
          (int)options->lower_bitfield_insert_to_bitfield_select <= 1);


   STATIC_ASSERT(${str(cache["next_index"])} == ARRAY_SIZE(${pass_name}_values));
   % for index, condition in enumerate(condition_list):
   condition_flags[${index}] = ${condition};
   % endfor

   nir_foreach_function(function, shader) {
      if (function->impl) {
         progress |= nir_algebraic_impl(function->impl, condition_flags,
                                        &${pass_name}_table);
      }
   }

   return progress;
}
""")


class AlgebraicPass(object):
   def __init__(self, pass_name, transforms):
      self.xforms = []
      self.opcode_xforms = defaultdict(lambda : [])
      self.pass_name = pass_name
      self.expression_cond = {}
      self.variable_cond = {}

      error = False

      for xform in transforms:
         if not isinstance(xform, SearchAndReplace):
            try:
               xform = SearchAndReplace(xform, self)
            except:
               print("Failed to parse transformation:", file=sys.stderr)
               print("  " + str(xform), file=sys.stderr)
               traceback.print_exc(file=sys.stderr)
               print('', file=sys.stderr)
               error = True
               continue

         self.xforms.append(xform)
         if xform.search.opcode in conv_opcode_types:
            dst_type = conv_opcode_types[xform.search.opcode]
            for size in type_sizes(dst_type):
               sized_opcode = xform.search.opcode + str(size)
               self.opcode_xforms[sized_opcode].append(xform)
         else:
            self.opcode_xforms[xform.search.opcode].append(xform)

         # Check to make sure the search pattern does not unexpectedly contain
         # more commutative expressions than match_expression (nir_search.c)
         # can handle.
         comm_exprs = xform.search.comm_exprs

         if xform.search.many_commutative_expressions:
            if comm_exprs <= nir_search_max_comm_ops:
               print("Transform expected to have too many commutative " \
                     "expression but did not " \
                     "({} <= {}).".format(comm_exprs, nir_search_max_comm_op),
                     file=sys.stderr)
               print("  " + str(xform), file=sys.stderr)
               traceback.print_exc(file=sys.stderr)
               print('', file=sys.stderr)
               error = True
         else:
            if comm_exprs > nir_search_max_comm_ops:
               print("Transformation with too many commutative expressions " \
                     "({} > {}).  Modify pattern or annotate with " \
                     "\"many-comm-expr\".".format(comm_exprs,
                                                  nir_search_max_comm_ops),
                     file=sys.stderr)
               print("  " + str(xform.search), file=sys.stderr)
               print("{}".format(xform.search.cond), file=sys.stderr)
               error = True

      self.automaton = TreeAutomaton(self.xforms)

      if error:
         sys.exit(1)


   def render(self):
      return _algebraic_pass_template.render(pass_name=self.pass_name,
                                             xforms=self.xforms,
                                             opcode_xforms=self.opcode_xforms,
                                             condition_list=condition_list,
                                             automaton=self.automaton,
                                             expression_cond = sorted(self.expression_cond.items(), key=lambda kv: kv[1]),
                                             variable_cond = sorted(self.variable_cond.items(), key=lambda kv: kv[1]),
                                             get_c_opcode=get_c_opcode,
                                             itertools=itertools)

# The replacement expression isn't necessarily exact if the search expression is exact.
def ignore_exact(*expr):
   expr = SearchExpression.create(expr)
   expr.ignore_exact = True
   return expr
