/*
 * Copyright © Microsoft Corporation
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef DXIL_INTERNAL_H
#define DXIL_INTERNAL_H

#include "dxil_module.h"

#include "util/list.h"

#include <stdint.h>

// Malloc.h defines a macro for alloca. Let's at least make sure that all includers
// of this header have the same definition of alloca.
#include <malloc.h>

struct dxil_type_list {
   struct dxil_type **types;
   size_t num_types;
};

struct dxil_type {
   enum type_type {
      TYPE_VOID,
      TYPE_INTEGER,
      TYPE_FLOAT,
      TYPE_POINTER,
      TYPE_STRUCT,
      TYPE_ARRAY,
      TYPE_VECTOR,
      TYPE_FUNCTION
   } type;

   union {
      unsigned int_bits;
      unsigned float_bits;
      const struct dxil_type *ptr_target_type;
      struct {
         const char *name;
         struct dxil_type_list elem;
      } struct_def;
      struct {
         const struct dxil_type *ret_type;
         struct dxil_type_list args;
      } function_def;
      struct {
         const struct dxil_type *elem_type;
         size_t num_elems;
      } array_or_vector_def;
   };

   struct list_head head;
   unsigned id;
};

struct dxil_value {
   int id;
   const struct dxil_type *type;
};

struct dxil_gvar {
   const char *name;
   const struct dxil_type *type;
   bool constant;
   enum dxil_address_space as;
   int align;

   const struct dxil_value *initializer;
   struct dxil_value value;
   struct list_head head;
};

struct dxil_func {
   char *name;
   const struct dxil_type *type;
   bool decl;
   unsigned attr_set;

   struct dxil_value value;
   struct list_head head;
};

struct dxil_attrib {
   enum {
      DXIL_ATTR_ENUM
   } type;

   union {
      enum dxil_attr_kind kind;
   };
};

struct attrib_set {
   struct dxil_attrib attrs[2];
   unsigned num_attrs;
   struct list_head head;
};

struct dxil_instr_binop {
   enum dxil_bin_opcode opcode;
   const struct dxil_value *operands[2];
   enum dxil_opt_flags flags;
};

struct dxil_instr_cmp {
   enum dxil_cmp_pred pred;
   const struct dxil_value *operands[2];
};

struct dxil_instr_select {
   const struct dxil_value *operands[3];
};

struct dxil_instr_cast {
   enum dxil_cast_opcode opcode;
   const struct dxil_type *type;
   const struct dxil_value *value;
};

struct dxil_instr_call {
   const struct dxil_func *func;
   struct dxil_value **args;
   size_t num_args;
};

struct dxil_instr_ret {
   struct dxil_value *value;
};

struct dxil_instr_extractval {
   const struct dxil_value *src;
   const struct dxil_type *type;
   unsigned int idx;
};

struct dxil_instr_br {
   const struct dxil_value *cond;
   unsigned succ[2];
};

struct dxil_instr_phi {
   const struct dxil_type *type;
   struct dxil_phi_src {
      const struct dxil_value *value;
      unsigned block;
   } *incoming;
   size_t num_incoming;
};

struct dxil_instr_alloca {
   const struct dxil_type *alloc_type;
   const struct dxil_type *size_type;
   const struct dxil_value *size;
   unsigned align;
};

struct dxil_instr_gep {
   bool inbounds;
   const struct dxil_type *source_elem_type;
   struct dxil_value **operands;
   size_t num_operands;
};

struct dxil_instr_load {
   const struct dxil_value *ptr;
   const struct dxil_type *type;
   unsigned align;
   bool is_volatile;
};

struct dxil_instr_store {
   const struct dxil_value *value, *ptr;
   unsigned align;
   bool is_volatile;
};

struct dxil_instr_atomicrmw {
   const struct dxil_value *value, *ptr;
   enum dxil_rmw_op op;
   bool is_volatile;
   enum dxil_atomic_ordering ordering;
   enum dxil_sync_scope syncscope;
};

struct dxil_instr_cmpxchg {
   const struct dxil_value *cmpval, *newval, *ptr;
   bool is_volatile;
   enum dxil_atomic_ordering ordering;
   enum dxil_sync_scope syncscope;
};

struct dxil_instr {
   enum instr_type {
      INSTR_BINOP,
      INSTR_CMP,
      INSTR_SELECT,
      INSTR_CAST,
      INSTR_BR,
      INSTR_PHI,
      INSTR_CALL,
      INSTR_RET,
      INSTR_EXTRACTVAL,
      INSTR_ALLOCA,
      INSTR_GEP,
      INSTR_LOAD,
      INSTR_STORE,
      INSTR_ATOMICRMW,
      INSTR_CMPXCHG,
   } type;

   union {
      struct dxil_instr_binop binop;
      struct dxil_instr_cmp cmp;
      struct dxil_instr_select select;
      struct dxil_instr_cast cast;
      struct dxil_instr_call call;
      struct dxil_instr_ret ret;
      struct dxil_instr_extractval extractval;
      struct dxil_instr_phi phi;
      struct dxil_instr_br br;
      struct dxil_instr_alloca alloca;
      struct dxil_instr_gep gep;
      struct dxil_instr_load load;
      struct dxil_instr_store store;
      struct dxil_instr_atomicrmw atomicrmw;
      struct dxil_instr_cmpxchg cmpxchg;
   };

   bool has_value;
   struct dxil_value value;

   struct list_head head;
};

struct dxil_const {
   struct dxil_value value;

   bool undef;
   union {
      intmax_t int_value;
      double float_value;
      const struct dxil_value **array_values;
      const struct dxil_value **struct_values;
   };

   struct list_head head;
};

struct dxil_mdnode {
   enum mdnode_type {
      MD_STRING,
      MD_VALUE,
      MD_NODE
   } type;

   union {
      char *string;

      struct {
         const struct dxil_type *type;
         const struct dxil_value *value;
      } value;

      struct {
         const struct dxil_mdnode **subnodes;
         size_t num_subnodes;
      } node;
   };

   struct list_head head;
   unsigned id;
};

struct dxil_named_node {
   char *name;
   const struct dxil_mdnode **subnodes;
   size_t num_subnodes;
   struct list_head head;
};

#endif // DXIL_INTERNAL_H
