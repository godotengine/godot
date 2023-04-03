/*
 * Copyright © 2017 Connor Abbott
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

#include "nir_serialize.h"
#include "nir_control_flow.h"
#include "nir_xfb_info.h"
#include "util/u_dynarray.h"
#include "util/u_math.h"

#define NIR_SERIALIZE_FUNC_HAS_IMPL ((void *)(intptr_t)1)
#define MAX_OBJECT_IDS (1 << 20)

typedef struct {
   size_t blob_offset;
   nir_ssa_def *src;
   nir_block *block;
} write_phi_fixup;

typedef struct {
   const nir_shader *nir;

   struct blob *blob;

   /* maps pointer to index */
   struct hash_table *remap_table;

   /* the next index to assign to a NIR in-memory object */
   uint32_t next_idx;

   /* Array of write_phi_fixup structs representing phi sources that need to
    * be resolved in the second pass.
    */
   struct util_dynarray phi_fixups;

   /* The last serialized type. */
   const struct glsl_type *last_type;
   const struct glsl_type *last_interface_type;
   struct nir_variable_data last_var_data;

   /* For skipping equal ALU headers (typical after scalarization). */
   nir_instr_type last_instr_type;
   uintptr_t last_alu_header_offset;
   uint32_t last_alu_header;

   /* Don't write optional data such as variable names. */
   bool strip;
} write_ctx;

typedef struct {
   nir_shader *nir;

   struct blob_reader *blob;

   /* the next index to assign to a NIR in-memory object */
   uint32_t next_idx;

   /* The length of the index -> object table */
   uint32_t idx_table_len;

   /* map from index to deserialized pointer */
   void **idx_table;

   /* List of phi sources. */
   struct list_head phi_srcs;

   /* The last deserialized type. */
   const struct glsl_type *last_type;
   const struct glsl_type *last_interface_type;
   struct nir_variable_data last_var_data;
} read_ctx;

static void
write_add_object(write_ctx *ctx, const void *obj)
{
   uint32_t index = ctx->next_idx++;
   assert(index != MAX_OBJECT_IDS);
   _mesa_hash_table_insert(ctx->remap_table, obj, (void *)(uintptr_t) index);
}

static uint32_t
write_lookup_object(write_ctx *ctx, const void *obj)
{
   struct hash_entry *entry = _mesa_hash_table_search(ctx->remap_table, obj);
   assert(entry);
   return (uint32_t)(uintptr_t) entry->data;
}

static void
read_add_object(read_ctx *ctx, void *obj)
{
   assert(ctx->next_idx < ctx->idx_table_len);
   ctx->idx_table[ctx->next_idx++] = obj;
}

static void *
read_lookup_object(read_ctx *ctx, uint32_t idx)
{
   assert(idx < ctx->idx_table_len);
   return ctx->idx_table[idx];
}

static void *
read_object(read_ctx *ctx)
{
   return read_lookup_object(ctx, blob_read_uint32(ctx->blob));
}

static uint32_t
encode_bit_size_3bits(uint8_t bit_size)
{
   /* Encode values of 0, 1, 2, 4, 8, 16, 32, 64 in 3 bits. */
   assert(bit_size <= 64 && util_is_power_of_two_or_zero(bit_size));
   if (bit_size)
      return util_logbase2(bit_size) + 1;
   return 0;
}

static uint8_t
decode_bit_size_3bits(uint8_t bit_size)
{
   if (bit_size)
      return 1 << (bit_size - 1);
   return 0;
}

#define NUM_COMPONENTS_IS_SEPARATE_7   7

static uint8_t
encode_num_components_in_3bits(uint8_t num_components)
{
   if (num_components <= 4)
      return num_components;
   if (num_components == 8)
      return 5;
   if (num_components == 16)
      return 6;

   /* special value indicating that num_components is in the next uint32 */
   return NUM_COMPONENTS_IS_SEPARATE_7;
}

static uint8_t
decode_num_components_in_3bits(uint8_t value)
{
   if (value <= 4)
      return value;
   if (value == 5)
      return 8;
   if (value == 6)
      return 16;

   unreachable("invalid num_components encoding");
   return 0;
}

static void
write_constant(write_ctx *ctx, const nir_constant *c)
{
   blob_write_bytes(ctx->blob, c->values, sizeof(c->values));
   blob_write_uint32(ctx->blob, c->num_elements);
   for (unsigned i = 0; i < c->num_elements; i++)
      write_constant(ctx, c->elements[i]);
}

static nir_constant *
read_constant(read_ctx *ctx, nir_variable *nvar)
{
   nir_constant *c = ralloc(nvar, nir_constant);

   blob_copy_bytes(ctx->blob, (uint8_t *)c->values, sizeof(c->values));
   c->num_elements = blob_read_uint32(ctx->blob);
   c->elements = ralloc_array(nvar, nir_constant *, c->num_elements);
   for (unsigned i = 0; i < c->num_elements; i++)
      c->elements[i] = read_constant(ctx, nvar);

   return c;
}

enum var_data_encoding {
   var_encode_full,
   var_encode_shader_temp,
   var_encode_function_temp,
   var_encode_location_diff,
};

union packed_var {
   uint32_t u32;
   struct {
      unsigned has_name:1;
      unsigned has_constant_initializer:1;
      unsigned has_pointer_initializer:1;
      unsigned has_interface_type:1;
      unsigned num_state_slots:7;
      unsigned data_encoding:2;
      unsigned type_same_as_last:1;
      unsigned interface_type_same_as_last:1;
      unsigned ray_query:1;
      unsigned num_members:16;
   } u;
};

union packed_var_data_diff {
   uint32_t u32;
   struct {
      int location:13;
      int location_frac:3;
      int driver_location:16;
   } u;
};

static void
write_variable(write_ctx *ctx, const nir_variable *var)
{
   write_add_object(ctx, var);

   assert(var->num_state_slots < (1 << 7));

   STATIC_ASSERT(sizeof(union packed_var) == 4);
   union packed_var flags;
   flags.u32 = 0;

   flags.u.has_name = !ctx->strip && var->name;
   flags.u.has_constant_initializer = !!(var->constant_initializer);
   flags.u.has_pointer_initializer = !!(var->pointer_initializer);
   flags.u.has_interface_type = !!(var->interface_type);
   flags.u.type_same_as_last = var->type == ctx->last_type;
   flags.u.interface_type_same_as_last =
      var->interface_type && var->interface_type == ctx->last_interface_type;
   flags.u.num_state_slots = var->num_state_slots;
   flags.u.num_members = var->num_members;

   struct nir_variable_data data = var->data;

   /* When stripping, we expect that the location is no longer needed,
    * which is typically after shaders are linked.
    */
   if (ctx->strip &&
       data.mode != nir_var_system_value &&
       data.mode != nir_var_shader_in &&
       data.mode != nir_var_shader_out)
      data.location = 0;

   /* Temporary variables don't serialize var->data. */
   if (data.mode == nir_var_shader_temp)
      flags.u.data_encoding = var_encode_shader_temp;
   else if (data.mode == nir_var_function_temp)
      flags.u.data_encoding = var_encode_function_temp;
   else {
      struct nir_variable_data tmp = data;

      tmp.location = ctx->last_var_data.location;
      tmp.location_frac = ctx->last_var_data.location_frac;
      tmp.driver_location = ctx->last_var_data.driver_location;

      /* See if we can encode only the difference in locations from the last
       * variable.
       */
      if (memcmp(&ctx->last_var_data, &tmp, sizeof(tmp)) == 0 &&
          abs((int)data.location -
              (int)ctx->last_var_data.location) < (1 << 12) &&
          abs((int)data.driver_location -
              (int)ctx->last_var_data.driver_location) < (1 << 15))
         flags.u.data_encoding = var_encode_location_diff;
      else
         flags.u.data_encoding = var_encode_full;
   }

   flags.u.ray_query = var->data.ray_query;

   blob_write_uint32(ctx->blob, flags.u32);

   if (!flags.u.type_same_as_last) {
      encode_type_to_blob(ctx->blob, var->type);
      ctx->last_type = var->type;
   }

   if (var->interface_type && !flags.u.interface_type_same_as_last) {
      encode_type_to_blob(ctx->blob, var->interface_type);
      ctx->last_interface_type = var->interface_type;
   }

   if (flags.u.has_name)
      blob_write_string(ctx->blob, var->name);

   if (flags.u.data_encoding == var_encode_full ||
       flags.u.data_encoding == var_encode_location_diff) {
      if (flags.u.data_encoding == var_encode_full) {
         blob_write_bytes(ctx->blob, &data, sizeof(data));
      } else {
         /* Serialize only the difference in locations from the last variable.
          */
         union packed_var_data_diff diff;

         diff.u.location = data.location - ctx->last_var_data.location;
         diff.u.location_frac = data.location_frac -
                                ctx->last_var_data.location_frac;
         diff.u.driver_location = data.driver_location -
                                  ctx->last_var_data.driver_location;

         blob_write_uint32(ctx->blob, diff.u32);
      }

      ctx->last_var_data = data;
   }

   for (unsigned i = 0; i < var->num_state_slots; i++) {
      blob_write_bytes(ctx->blob, &var->state_slots[i],
                       sizeof(var->state_slots[i]));
   }
   if (var->constant_initializer)
      write_constant(ctx, var->constant_initializer);
   if (var->pointer_initializer)
      write_lookup_object(ctx, var->pointer_initializer);
   if (var->num_members > 0) {
      blob_write_bytes(ctx->blob, (uint8_t *) var->members,
                       var->num_members * sizeof(*var->members));
   }
}

static nir_variable *
read_variable(read_ctx *ctx)
{
   nir_variable *var = rzalloc(ctx->nir, nir_variable);
   read_add_object(ctx, var);

   union packed_var flags;
   flags.u32 = blob_read_uint32(ctx->blob);

   if (flags.u.type_same_as_last) {
      var->type = ctx->last_type;
   } else {
      var->type = decode_type_from_blob(ctx->blob);
      ctx->last_type = var->type;
   }

   if (flags.u.has_interface_type) {
      if (flags.u.interface_type_same_as_last) {
         var->interface_type = ctx->last_interface_type;
      } else {
         var->interface_type = decode_type_from_blob(ctx->blob);
         ctx->last_interface_type = var->interface_type;
      }
   }

   if (flags.u.has_name) {
      const char *name = blob_read_string(ctx->blob);
      var->name = ralloc_strdup(var, name);
   } else {
      var->name = NULL;
   }

   if (flags.u.data_encoding == var_encode_shader_temp)
      var->data.mode = nir_var_shader_temp;
   else if (flags.u.data_encoding == var_encode_function_temp)
      var->data.mode = nir_var_function_temp;
   else if (flags.u.data_encoding == var_encode_full) {
      blob_copy_bytes(ctx->blob, (uint8_t *) &var->data, sizeof(var->data));
      ctx->last_var_data = var->data;
   } else { /* var_encode_location_diff */
      union packed_var_data_diff diff;
      diff.u32 = blob_read_uint32(ctx->blob);

      var->data = ctx->last_var_data;
      var->data.location += diff.u.location;
      var->data.location_frac += diff.u.location_frac;
      var->data.driver_location += diff.u.driver_location;

      ctx->last_var_data = var->data;
   }

   var->data.ray_query = flags.u.ray_query;

   var->num_state_slots = flags.u.num_state_slots;
   if (var->num_state_slots != 0) {
      var->state_slots = ralloc_array(var, nir_state_slot,
                                      var->num_state_slots);
      for (unsigned i = 0; i < var->num_state_slots; i++) {
         blob_copy_bytes(ctx->blob, &var->state_slots[i],
                         sizeof(var->state_slots[i]));
      }
   }
   if (flags.u.has_constant_initializer)
      var->constant_initializer = read_constant(ctx, var);
   else
      var->constant_initializer = NULL;

   if (flags.u.has_pointer_initializer)
      var->pointer_initializer = read_object(ctx);
   else
      var->pointer_initializer = NULL;

   var->num_members = flags.u.num_members;
   if (var->num_members > 0) {
      var->members = ralloc_array(var, struct nir_variable_data,
                                  var->num_members);
      blob_copy_bytes(ctx->blob, (uint8_t *) var->members,
                      var->num_members * sizeof(*var->members));
   }

   return var;
}

static void
write_var_list(write_ctx *ctx, const struct exec_list *src)
{
   blob_write_uint32(ctx->blob, exec_list_length(src));
   foreach_list_typed(nir_variable, var, node, src) {
      write_variable(ctx, var);
   }
}

static void
read_var_list(read_ctx *ctx, struct exec_list *dst)
{
   exec_list_make_empty(dst);
   unsigned num_vars = blob_read_uint32(ctx->blob);
   for (unsigned i = 0; i < num_vars; i++) {
      nir_variable *var = read_variable(ctx);
      exec_list_push_tail(dst, &var->node);
   }
}

static void
write_register(write_ctx *ctx, const nir_register *reg)
{
   write_add_object(ctx, reg);
   blob_write_uint32(ctx->blob, reg->num_components);
   blob_write_uint32(ctx->blob, reg->bit_size);
   blob_write_uint32(ctx->blob, reg->num_array_elems);
   blob_write_uint32(ctx->blob, reg->index);
   blob_write_uint8(ctx->blob, reg->divergent);
}

static nir_register *
read_register(read_ctx *ctx)
{
   nir_register *reg = ralloc(ctx->nir, nir_register);
   read_add_object(ctx, reg);
   reg->num_components = blob_read_uint32(ctx->blob);
   reg->bit_size = blob_read_uint32(ctx->blob);
   reg->num_array_elems = blob_read_uint32(ctx->blob);
   reg->index = blob_read_uint32(ctx->blob);
   reg->divergent = blob_read_uint8(ctx->blob);

   list_inithead(&reg->uses);
   list_inithead(&reg->defs);
   list_inithead(&reg->if_uses);

   return reg;
}

static void
write_reg_list(write_ctx *ctx, const struct exec_list *src)
{
   blob_write_uint32(ctx->blob, exec_list_length(src));
   foreach_list_typed(nir_register, reg, node, src)
      write_register(ctx, reg);
}

static void
read_reg_list(read_ctx *ctx, struct exec_list *dst)
{
   exec_list_make_empty(dst);
   unsigned num_regs = blob_read_uint32(ctx->blob);
   for (unsigned i = 0; i < num_regs; i++) {
      nir_register *reg = read_register(ctx);
      exec_list_push_tail(dst, &reg->node);
   }
}

union packed_src {
   uint32_t u32;
   struct {
      unsigned is_ssa:1;   /* <-- Header */
      unsigned is_indirect:1;
      unsigned object_idx:20;
      unsigned _footer:10; /* <-- Footer */
   } any;
   struct {
      unsigned _header:22; /* <-- Header */
      unsigned negate:1;   /* <-- Footer */
      unsigned abs:1;
      unsigned swizzle_x:2;
      unsigned swizzle_y:2;
      unsigned swizzle_z:2;
      unsigned swizzle_w:2;
   } alu;
   struct {
      unsigned _header:22; /* <-- Header */
      unsigned src_type:5; /* <-- Footer */
      unsigned _pad:5;
   } tex;
};

static void
write_src_full(write_ctx *ctx, const nir_src *src, union packed_src header)
{
   /* Since sources are very frequent, we try to save some space when storing
    * them. In particular, we store whether the source is a register and
    * whether the register has an indirect index in the low two bits. We can
    * assume that the high two bits of the index are zero, since otherwise our
    * address space would've been exhausted allocating the remap table!
    */
   header.any.is_ssa = src->is_ssa;
   if (src->is_ssa) {
      header.any.object_idx = write_lookup_object(ctx, src->ssa);
      blob_write_uint32(ctx->blob, header.u32);
   } else {
      header.any.object_idx = write_lookup_object(ctx, src->reg.reg);
      header.any.is_indirect = !!src->reg.indirect;
      blob_write_uint32(ctx->blob, header.u32);
      blob_write_uint32(ctx->blob, src->reg.base_offset);
      if (src->reg.indirect) {
         union packed_src header = {0};
         write_src_full(ctx, src->reg.indirect, header);
      }
   }
}

static void
write_src(write_ctx *ctx, const nir_src *src)
{
   union packed_src header = {0};
   write_src_full(ctx, src, header);
}

static union packed_src
read_src(read_ctx *ctx, nir_src *src)
{
   STATIC_ASSERT(sizeof(union packed_src) == 4);
   union packed_src header;
   header.u32 = blob_read_uint32(ctx->blob);

   src->is_ssa = header.any.is_ssa;
   if (src->is_ssa) {
      src->ssa = read_lookup_object(ctx, header.any.object_idx);
   } else {
      src->reg.reg = read_lookup_object(ctx, header.any.object_idx);
      src->reg.base_offset = blob_read_uint32(ctx->blob);
      if (header.any.is_indirect) {
         src->reg.indirect = gc_alloc(ctx->nir->gctx, nir_src, 1);
         read_src(ctx, src->reg.indirect);
      } else {
         src->reg.indirect = NULL;
      }
   }
   return header;
}

union packed_dest {
   uint8_t u8;
   struct {
      uint8_t is_ssa:1;
      uint8_t num_components:3;
      uint8_t bit_size:3;
      uint8_t divergent:1;
   } ssa;
   struct {
      uint8_t is_ssa:1;
      uint8_t is_indirect:1;
      uint8_t _pad:6;
   } reg;
};

enum intrinsic_const_indices_encoding {
   /* Use packed_const_indices to store tightly packed indices.
    *
    * The common case for load_ubo is 0, 0, 0, which is trivially represented.
    * The common cases for load_interpolated_input also fit here, e.g.: 7, 3
    */
   const_indices_all_combined,

   const_indices_8bit,  /* 8 bits per element */
   const_indices_16bit, /* 16 bits per element */
   const_indices_32bit, /* 32 bits per element */
};

enum load_const_packing {
   /* Constants are not packed and are stored in following dwords. */
   load_const_full,

   /* packed_value contains high 19 bits, low bits are 0,
    * good for floating-point decimals
    */
   load_const_scalar_hi_19bits,

   /* packed_value contains low 19 bits, high bits are sign-extended */
   load_const_scalar_lo_19bits_sext,
};

union packed_instr {
   uint32_t u32;
   struct {
      unsigned instr_type:4; /* always present */
      unsigned _pad:20;
      unsigned dest:8;       /* always last */
   } any;
   struct {
      unsigned instr_type:4;
      unsigned exact:1;
      unsigned no_signed_wrap:1;
      unsigned no_unsigned_wrap:1;
      unsigned saturate:1;
      /* Reg: writemask; SSA: swizzles for 2 srcs */
      unsigned writemask_or_two_swizzles:4;
      unsigned op:9;
      unsigned packed_src_ssa_16bit:1;
      /* Scalarized ALUs always have the same header. */
      unsigned num_followup_alu_sharing_header:2;
      unsigned dest:8;
   } alu;
   struct {
      unsigned instr_type:4;
      unsigned deref_type:3;
      unsigned cast_type_same_as_last:1;
      unsigned modes:5; /* See (de|en)code_deref_modes() */
      unsigned _pad:9;
      unsigned in_bounds:1;
      unsigned packed_src_ssa_16bit:1; /* deref_var redefines this */
      unsigned dest:8;
   } deref;
   struct {
      unsigned instr_type:4;
      unsigned deref_type:3;
      unsigned _pad:1;
      unsigned object_idx:16; /* if 0, the object ID is a separate uint32 */
      unsigned dest:8;
   } deref_var;
   struct {
      unsigned instr_type:4;
      unsigned intrinsic:10;
      unsigned const_indices_encoding:2;
      unsigned packed_const_indices:8;
      unsigned dest:8;
   } intrinsic;
   struct {
      unsigned instr_type:4;
      unsigned last_component:4;
      unsigned bit_size:3;
      unsigned packing:2; /* enum load_const_packing */
      unsigned packed_value:19; /* meaning determined by packing */
   } load_const;
   struct {
      unsigned instr_type:4;
      unsigned last_component:4;
      unsigned bit_size:3;
      unsigned _pad:21;
   } undef;
   struct {
      unsigned instr_type:4;
      unsigned num_srcs:4;
      unsigned op:5;
      unsigned _pad:11;
      unsigned dest:8;
   } tex;
   struct {
      unsigned instr_type:4;
      unsigned num_srcs:20;
      unsigned dest:8;
   } phi;
   struct {
      unsigned instr_type:4;
      unsigned type:2;
      unsigned _pad:26;
   } jump;
};

/* Write "lo24" as low 24 bits in the first uint32. */
static void
write_dest(write_ctx *ctx, const nir_dest *dst, union packed_instr header,
           nir_instr_type instr_type)
{
   STATIC_ASSERT(sizeof(union packed_dest) == 1);
   union packed_dest dest;
   dest.u8 = 0;

   dest.ssa.is_ssa = dst->is_ssa;
   if (dst->is_ssa) {
      dest.ssa.num_components =
         encode_num_components_in_3bits(dst->ssa.num_components);
      dest.ssa.bit_size = encode_bit_size_3bits(dst->ssa.bit_size);
      dest.ssa.divergent = dst->ssa.divergent;
   } else {
      dest.reg.is_indirect = !!(dst->reg.indirect);
   }
   header.any.dest = dest.u8;

   /* Check if the current ALU instruction has the same header as the previous
    * instruction that is also ALU. If it is, we don't have to write
    * the current header. This is a typical occurence after scalarization.
    */
   if (instr_type == nir_instr_type_alu) {
      bool equal_header = false;

      if (ctx->last_instr_type == nir_instr_type_alu) {
         assert(ctx->last_alu_header_offset);
         union packed_instr last_header;
         last_header.u32 = ctx->last_alu_header;

         /* Clear the field that counts ALUs with equal headers. */
         union packed_instr clean_header;
         clean_header.u32 = last_header.u32;
         clean_header.alu.num_followup_alu_sharing_header = 0;

         /* There can be at most 4 consecutive ALU instructions
          * sharing the same header.
          */
         if (last_header.alu.num_followup_alu_sharing_header < 3 &&
             header.u32 == clean_header.u32) {
            last_header.alu.num_followup_alu_sharing_header++;
            blob_overwrite_uint32(ctx->blob, ctx->last_alu_header_offset,
                                  last_header.u32);
            ctx->last_alu_header = last_header.u32;
            equal_header = true;
         }
      }

      if (!equal_header) {
         ctx->last_alu_header_offset = blob_reserve_uint32(ctx->blob);
         blob_overwrite_uint32(ctx->blob, ctx->last_alu_header_offset, header.u32);
         ctx->last_alu_header = header.u32;
      }
   } else {
      blob_write_uint32(ctx->blob, header.u32);
   }

   if (dest.ssa.is_ssa &&
       dest.ssa.num_components == NUM_COMPONENTS_IS_SEPARATE_7)
      blob_write_uint32(ctx->blob, dst->ssa.num_components);

   if (dst->is_ssa) {
      write_add_object(ctx, &dst->ssa);
   } else {
      blob_write_uint32(ctx->blob, write_lookup_object(ctx, dst->reg.reg));
      blob_write_uint32(ctx->blob, dst->reg.base_offset);
      if (dst->reg.indirect)
         write_src(ctx, dst->reg.indirect);
   }
}

static void
read_dest(read_ctx *ctx, nir_dest *dst, nir_instr *instr,
          union packed_instr header)
{
   union packed_dest dest;
   dest.u8 = header.any.dest;

   if (dest.ssa.is_ssa) {
      unsigned bit_size = decode_bit_size_3bits(dest.ssa.bit_size);
      unsigned num_components;
      if (dest.ssa.num_components == NUM_COMPONENTS_IS_SEPARATE_7)
         num_components = blob_read_uint32(ctx->blob);
      else
         num_components = decode_num_components_in_3bits(dest.ssa.num_components);
      nir_ssa_dest_init(instr, dst, num_components, bit_size, NULL);
      dst->ssa.divergent = dest.ssa.divergent;
      read_add_object(ctx, &dst->ssa);
   } else {
      dst->reg.reg = read_object(ctx);
      dst->reg.base_offset = blob_read_uint32(ctx->blob);
      if (dest.reg.is_indirect) {
         dst->reg.indirect = gc_alloc(ctx->nir->gctx, nir_src, 1);
         read_src(ctx, dst->reg.indirect);
      }
   }
}

static bool
are_object_ids_16bit(write_ctx *ctx)
{
   /* Check the highest object ID, because they are monotonic. */
   return ctx->next_idx < (1 << 16);
}

static bool
is_alu_src_ssa_16bit(write_ctx *ctx, const nir_alu_instr *alu)
{
   unsigned num_srcs = nir_op_infos[alu->op].num_inputs;

   for (unsigned i = 0; i < num_srcs; i++) {
      if (!alu->src[i].src.is_ssa || alu->src[i].abs || alu->src[i].negate)
         return false;

      unsigned src_components = nir_ssa_alu_instr_src_components(alu, i);

      for (unsigned chan = 0; chan < src_components; chan++) {
         /* The swizzles for src0.x and src1.x are stored
          * in writemask_or_two_swizzles for SSA ALUs.
          */
         if (alu->dest.dest.is_ssa && i < 2 && chan == 0 &&
             alu->src[i].swizzle[chan] < 4)
            continue;

         if (alu->src[i].swizzle[chan] != chan)
            return false;
      }
   }

   return are_object_ids_16bit(ctx);
}

static void
write_alu(write_ctx *ctx, const nir_alu_instr *alu)
{
   unsigned num_srcs = nir_op_infos[alu->op].num_inputs;
   unsigned dst_components = nir_dest_num_components(alu->dest.dest);

   /* 9 bits for nir_op */
   STATIC_ASSERT(nir_num_opcodes <= 512);
   union packed_instr header;
   header.u32 = 0;

   header.alu.instr_type = alu->instr.type;
   header.alu.exact = alu->exact;
   header.alu.no_signed_wrap = alu->no_signed_wrap;
   header.alu.no_unsigned_wrap = alu->no_unsigned_wrap;
   header.alu.saturate = alu->dest.saturate;
   header.alu.op = alu->op;
   header.alu.packed_src_ssa_16bit = is_alu_src_ssa_16bit(ctx, alu);

   if (header.alu.packed_src_ssa_16bit &&
       alu->dest.dest.is_ssa) {
      /* For packed srcs of SSA ALUs, this field stores the swizzles. */
      header.alu.writemask_or_two_swizzles = alu->src[0].swizzle[0];
      if (num_srcs > 1)
         header.alu.writemask_or_two_swizzles |= alu->src[1].swizzle[0] << 2;
   } else if (!alu->dest.dest.is_ssa && dst_components <= 4) {
      /* For vec4 registers, this field is a writemask. */
      header.alu.writemask_or_two_swizzles = alu->dest.write_mask;
   }

   write_dest(ctx, &alu->dest.dest, header, alu->instr.type);

   if (!alu->dest.dest.is_ssa && dst_components > 4)
      blob_write_uint32(ctx->blob, alu->dest.write_mask);

   if (header.alu.packed_src_ssa_16bit) {
      for (unsigned i = 0; i < num_srcs; i++) {
         assert(alu->src[i].src.is_ssa);
         unsigned idx = write_lookup_object(ctx, alu->src[i].src.ssa);
         assert(idx < (1 << 16));
         blob_write_uint16(ctx->blob, idx);
      }
   } else {
      for (unsigned i = 0; i < num_srcs; i++) {
         unsigned src_channels = nir_ssa_alu_instr_src_components(alu, i);
         unsigned src_components = nir_src_num_components(alu->src[i].src);
         union packed_src src;
         bool packed = src_components <= 4 && src_channels <= 4;
         src.u32 = 0;

         src.alu.negate = alu->src[i].negate;
         src.alu.abs = alu->src[i].abs;

         if (packed) {
            src.alu.swizzle_x = alu->src[i].swizzle[0];
            src.alu.swizzle_y = alu->src[i].swizzle[1];
            src.alu.swizzle_z = alu->src[i].swizzle[2];
            src.alu.swizzle_w = alu->src[i].swizzle[3];
         }

         write_src_full(ctx, &alu->src[i].src, src);

         /* Store swizzles for vec8 and vec16. */
         if (!packed) {
            for (unsigned o = 0; o < src_channels; o += 8) {
               unsigned value = 0;

               for (unsigned j = 0; j < 8 && o + j < src_channels; j++) {
                  value |= (uint32_t)alu->src[i].swizzle[o + j] <<
                           (4 * j); /* 4 bits per swizzle */
               }

               blob_write_uint32(ctx->blob, value);
            }
         }
      }
   }
}

static nir_alu_instr *
read_alu(read_ctx *ctx, union packed_instr header)
{
   unsigned num_srcs = nir_op_infos[header.alu.op].num_inputs;
   nir_alu_instr *alu = nir_alu_instr_create(ctx->nir, header.alu.op);

   alu->exact = header.alu.exact;
   alu->no_signed_wrap = header.alu.no_signed_wrap;
   alu->no_unsigned_wrap = header.alu.no_unsigned_wrap;
   alu->dest.saturate = header.alu.saturate;

   read_dest(ctx, &alu->dest.dest, &alu->instr, header);

   unsigned dst_components = nir_dest_num_components(alu->dest.dest);

   if (alu->dest.dest.is_ssa) {
      alu->dest.write_mask = u_bit_consecutive(0, dst_components);
   } else if (dst_components <= 4) {
      alu->dest.write_mask = header.alu.writemask_or_two_swizzles;
   } else {
      alu->dest.write_mask = blob_read_uint32(ctx->blob);
   }

   if (header.alu.packed_src_ssa_16bit) {
      for (unsigned i = 0; i < num_srcs; i++) {
         nir_alu_src *src = &alu->src[i];
         src->src.is_ssa = true;
         src->src.ssa = read_lookup_object(ctx, blob_read_uint16(ctx->blob));

         memset(&src->swizzle, 0, sizeof(src->swizzle));

         unsigned src_components = nir_ssa_alu_instr_src_components(alu, i);

         for (unsigned chan = 0; chan < src_components; chan++)
            src->swizzle[chan] = chan;
      }
   } else {
      for (unsigned i = 0; i < num_srcs; i++) {
         union packed_src src = read_src(ctx, &alu->src[i].src);
         unsigned src_channels = nir_ssa_alu_instr_src_components(alu, i);
         unsigned src_components = nir_src_num_components(alu->src[i].src);
         bool packed = src_components <= 4 && src_channels <= 4;

         alu->src[i].negate = src.alu.negate;
         alu->src[i].abs = src.alu.abs;

         memset(&alu->src[i].swizzle, 0, sizeof(alu->src[i].swizzle));

         if (packed) {
            alu->src[i].swizzle[0] = src.alu.swizzle_x;
            alu->src[i].swizzle[1] = src.alu.swizzle_y;
            alu->src[i].swizzle[2] = src.alu.swizzle_z;
            alu->src[i].swizzle[3] = src.alu.swizzle_w;
         } else {
            /* Load swizzles for vec8 and vec16. */
            for (unsigned o = 0; o < src_channels; o += 8) {
               unsigned value = blob_read_uint32(ctx->blob);

               for (unsigned j = 0; j < 8 && o + j < src_channels; j++) {
                  alu->src[i].swizzle[o + j] =
                     (value >> (4 * j)) & 0xf; /* 4 bits per swizzle */
               }
            }
         }
      }
   }

   if (header.alu.packed_src_ssa_16bit &&
       alu->dest.dest.is_ssa) {
      alu->src[0].swizzle[0] = header.alu.writemask_or_two_swizzles & 0x3;
      if (num_srcs > 1)
         alu->src[1].swizzle[0] = header.alu.writemask_or_two_swizzles >> 2;
   }

   return alu;
}

#define MODE_ENC_GENERIC_BIT (1 << 4)

static nir_variable_mode
decode_deref_modes(unsigned modes)
{
   if (modes & MODE_ENC_GENERIC_BIT) {
      modes &= ~MODE_ENC_GENERIC_BIT;
      return modes << (ffs(nir_var_mem_generic) - 1);
   } else {
      return 1 << modes;
   }
}

static unsigned
encode_deref_modes(nir_variable_mode modes)
{
   /* Mode sets on derefs generally come in two forms.  For certain OpenCL
    * cases, we can have more than one of the generic modes set.  In this
    * case, we need the full bitfield.  Fortunately, there are only 4 of
    * these.  For all other modes, we can only have one mode at a time so we
    * can compress them by only storing the bit position.  This, plus one bit
    * to select encoding, lets us pack the entire bitfield in 5 bits.
    */
   STATIC_ASSERT((nir_var_all & ~nir_var_mem_generic) <
                 (1 << MODE_ENC_GENERIC_BIT));

   unsigned enc;
   if (modes == 0 || (modes & nir_var_mem_generic)) {
      assert(!(modes & ~nir_var_mem_generic));
      enc = modes >> (ffs(nir_var_mem_generic) - 1);
      assert(enc < MODE_ENC_GENERIC_BIT);
      enc |= MODE_ENC_GENERIC_BIT;
   } else {
      assert(util_is_power_of_two_nonzero(modes));
      enc = ffs(modes) - 1;
      assert(enc < MODE_ENC_GENERIC_BIT);
   }
   assert(modes == decode_deref_modes(enc));
   return enc;
}

static void
write_deref(write_ctx *ctx, const nir_deref_instr *deref)
{
   assert(deref->deref_type < 8);

   union packed_instr header;
   header.u32 = 0;

   header.deref.instr_type = deref->instr.type;
   header.deref.deref_type = deref->deref_type;

   if (deref->deref_type == nir_deref_type_cast) {
      header.deref.modes = encode_deref_modes(deref->modes);
      header.deref.cast_type_same_as_last = deref->type == ctx->last_type;
   }

   unsigned var_idx = 0;
   if (deref->deref_type == nir_deref_type_var) {
      var_idx = write_lookup_object(ctx, deref->var);
      if (var_idx && var_idx < (1 << 16))
         header.deref_var.object_idx = var_idx;
   }

   if (deref->deref_type == nir_deref_type_array ||
       deref->deref_type == nir_deref_type_ptr_as_array) {
      header.deref.packed_src_ssa_16bit =
         deref->parent.is_ssa && deref->arr.index.is_ssa &&
         are_object_ids_16bit(ctx);

      header.deref.in_bounds = deref->arr.in_bounds;
   }

   write_dest(ctx, &deref->dest, header, deref->instr.type);

   switch (deref->deref_type) {
   case nir_deref_type_var:
      if (!header.deref_var.object_idx)
         blob_write_uint32(ctx->blob, var_idx);
      break;

   case nir_deref_type_struct:
      write_src(ctx, &deref->parent);
      blob_write_uint32(ctx->blob, deref->strct.index);
      break;

   case nir_deref_type_array:
   case nir_deref_type_ptr_as_array:
      if (header.deref.packed_src_ssa_16bit) {
         blob_write_uint16(ctx->blob,
                           write_lookup_object(ctx, deref->parent.ssa));
         blob_write_uint16(ctx->blob,
                           write_lookup_object(ctx, deref->arr.index.ssa));
      } else {
         write_src(ctx, &deref->parent);
         write_src(ctx, &deref->arr.index);
      }
      break;

   case nir_deref_type_cast:
      write_src(ctx, &deref->parent);
      blob_write_uint32(ctx->blob, deref->cast.ptr_stride);
      blob_write_uint32(ctx->blob, deref->cast.align_mul);
      blob_write_uint32(ctx->blob, deref->cast.align_offset);
      if (!header.deref.cast_type_same_as_last) {
         encode_type_to_blob(ctx->blob, deref->type);
         ctx->last_type = deref->type;
      }
      break;

   case nir_deref_type_array_wildcard:
      write_src(ctx, &deref->parent);
      break;

   default:
      unreachable("Invalid deref type");
   }
}

static nir_deref_instr *
read_deref(read_ctx *ctx, union packed_instr header)
{
   nir_deref_type deref_type = header.deref.deref_type;
   nir_deref_instr *deref = nir_deref_instr_create(ctx->nir, deref_type);

   read_dest(ctx, &deref->dest, &deref->instr, header);

   nir_deref_instr *parent;

   switch (deref->deref_type) {
   case nir_deref_type_var:
      if (header.deref_var.object_idx)
         deref->var = read_lookup_object(ctx, header.deref_var.object_idx);
      else
         deref->var = read_object(ctx);

      deref->type = deref->var->type;
      break;

   case nir_deref_type_struct:
      read_src(ctx, &deref->parent);
      parent = nir_src_as_deref(deref->parent);
      deref->strct.index = blob_read_uint32(ctx->blob);
      deref->type = glsl_get_struct_field(parent->type, deref->strct.index);
      break;

   case nir_deref_type_array:
   case nir_deref_type_ptr_as_array:
      if (header.deref.packed_src_ssa_16bit) {
         deref->parent.is_ssa = true;
         deref->parent.ssa = read_lookup_object(ctx, blob_read_uint16(ctx->blob));
         deref->arr.index.is_ssa = true;
         deref->arr.index.ssa = read_lookup_object(ctx, blob_read_uint16(ctx->blob));
      } else {
         read_src(ctx, &deref->parent);
         read_src(ctx, &deref->arr.index);
      }

      deref->arr.in_bounds = header.deref.in_bounds;

      parent = nir_src_as_deref(deref->parent);
      if (deref->deref_type == nir_deref_type_array)
         deref->type = glsl_get_array_element(parent->type);
      else
         deref->type = parent->type;
      break;

   case nir_deref_type_cast:
      read_src(ctx, &deref->parent);
      deref->cast.ptr_stride = blob_read_uint32(ctx->blob);
      deref->cast.align_mul = blob_read_uint32(ctx->blob);
      deref->cast.align_offset = blob_read_uint32(ctx->blob);
      if (header.deref.cast_type_same_as_last) {
         deref->type = ctx->last_type;
      } else {
         deref->type = decode_type_from_blob(ctx->blob);
         ctx->last_type = deref->type;
      }
      break;

   case nir_deref_type_array_wildcard:
      read_src(ctx, &deref->parent);
      parent = nir_src_as_deref(deref->parent);
      deref->type = glsl_get_array_element(parent->type);
      break;

   default:
      unreachable("Invalid deref type");
   }

   if (deref_type == nir_deref_type_var) {
      deref->modes = deref->var->data.mode;
   } else if (deref->deref_type == nir_deref_type_cast) {
      deref->modes = decode_deref_modes(header.deref.modes);
   } else {
      assert(deref->parent.is_ssa);
      deref->modes = nir_instr_as_deref(deref->parent.ssa->parent_instr)->modes;
   }

   return deref;
}

static void
write_intrinsic(write_ctx *ctx, const nir_intrinsic_instr *intrin)
{
   /* 10 bits for nir_intrinsic_op */
   STATIC_ASSERT(nir_num_intrinsics <= 1024);
   unsigned num_srcs = nir_intrinsic_infos[intrin->intrinsic].num_srcs;
   unsigned num_indices = nir_intrinsic_infos[intrin->intrinsic].num_indices;
   assert(intrin->intrinsic < 1024);

   union packed_instr header;
   header.u32 = 0;

   header.intrinsic.instr_type = intrin->instr.type;
   header.intrinsic.intrinsic = intrin->intrinsic;

   /* Analyze constant indices to decide how to encode them. */
   if (num_indices) {
      unsigned max_bits = 0;
      for (unsigned i = 0; i < num_indices; i++) {
         unsigned max = util_last_bit(intrin->const_index[i]);
         max_bits = MAX2(max_bits, max);
      }

      if (max_bits * num_indices <= 8) {
         header.intrinsic.const_indices_encoding = const_indices_all_combined;

         /* Pack all const indices into 8 bits. */
         unsigned bit_size = 8 / num_indices;
         for (unsigned i = 0; i < num_indices; i++) {
            header.intrinsic.packed_const_indices |=
               intrin->const_index[i] << (i * bit_size);
         }
      } else if (max_bits <= 8)
         header.intrinsic.const_indices_encoding = const_indices_8bit;
      else if (max_bits <= 16)
         header.intrinsic.const_indices_encoding = const_indices_16bit;
      else
         header.intrinsic.const_indices_encoding = const_indices_32bit;
   }

   if (nir_intrinsic_infos[intrin->intrinsic].has_dest)
      write_dest(ctx, &intrin->dest, header, intrin->instr.type);
   else
      blob_write_uint32(ctx->blob, header.u32);

   for (unsigned i = 0; i < num_srcs; i++)
      write_src(ctx, &intrin->src[i]);

   if (num_indices) {
      switch (header.intrinsic.const_indices_encoding) {
      case const_indices_8bit:
         for (unsigned i = 0; i < num_indices; i++)
            blob_write_uint8(ctx->blob, intrin->const_index[i]);
         break;
      case const_indices_16bit:
         for (unsigned i = 0; i < num_indices; i++)
            blob_write_uint16(ctx->blob, intrin->const_index[i]);
         break;
      case const_indices_32bit:
         for (unsigned i = 0; i < num_indices; i++)
            blob_write_uint32(ctx->blob, intrin->const_index[i]);
         break;
      }
   }
}

static nir_intrinsic_instr *
read_intrinsic(read_ctx *ctx, union packed_instr header)
{
   nir_intrinsic_op op = header.intrinsic.intrinsic;
   nir_intrinsic_instr *intrin = nir_intrinsic_instr_create(ctx->nir, op);

   unsigned num_srcs = nir_intrinsic_infos[op].num_srcs;
   unsigned num_indices = nir_intrinsic_infos[op].num_indices;

   if (nir_intrinsic_infos[op].has_dest)
      read_dest(ctx, &intrin->dest, &intrin->instr, header);

   for (unsigned i = 0; i < num_srcs; i++)
      read_src(ctx, &intrin->src[i]);

   /* Vectorized instrinsics have num_components same as dst or src that has
    * 0 components in the info. Find it.
    */
   if (nir_intrinsic_infos[op].has_dest &&
       nir_intrinsic_infos[op].dest_components == 0) {
      intrin->num_components = nir_dest_num_components(intrin->dest);
   } else {
      for (unsigned i = 0; i < num_srcs; i++) {
         if (nir_intrinsic_infos[op].src_components[i] == 0) {
            intrin->num_components = nir_src_num_components(intrin->src[i]);
            break;
         }
      }
   }

   if (num_indices) {
      switch (header.intrinsic.const_indices_encoding) {
      case const_indices_all_combined: {
         unsigned bit_size = 8 / num_indices;
         unsigned bit_mask = u_bit_consecutive(0, bit_size);
         for (unsigned i = 0; i < num_indices; i++) {
            intrin->const_index[i] =
               (header.intrinsic.packed_const_indices >> (i * bit_size)) &
               bit_mask;
         }
         break;
      }
      case const_indices_8bit:
         for (unsigned i = 0; i < num_indices; i++)
            intrin->const_index[i] = blob_read_uint8(ctx->blob);
         break;
      case const_indices_16bit:
         for (unsigned i = 0; i < num_indices; i++)
            intrin->const_index[i] = blob_read_uint16(ctx->blob);
         break;
      case const_indices_32bit:
         for (unsigned i = 0; i < num_indices; i++)
            intrin->const_index[i] = blob_read_uint32(ctx->blob);
         break;
      }
   }

   return intrin;
}

static void
write_load_const(write_ctx *ctx, const nir_load_const_instr *lc)
{
   assert(lc->def.num_components >= 1 && lc->def.num_components <= 16);
   union packed_instr header;
   header.u32 = 0;

   header.load_const.instr_type = lc->instr.type;
   header.load_const.last_component = lc->def.num_components - 1;
   header.load_const.bit_size = encode_bit_size_3bits(lc->def.bit_size);
   header.load_const.packing = load_const_full;

   /* Try to pack 1-component constants into the 19 free bits in the header. */
   if (lc->def.num_components == 1) {
      switch (lc->def.bit_size) {
      case 64:
         if ((lc->value[0].u64 & 0x1fffffffffffull) == 0) {
            /* packed_value contains high 19 bits, low bits are 0 */
            header.load_const.packing = load_const_scalar_hi_19bits;
            header.load_const.packed_value = lc->value[0].u64 >> 45;
         } else if (util_mask_sign_extend(lc->value[0].i64, 19) == lc->value[0].i64) {
            /* packed_value contains low 19 bits, high bits are sign-extended */
            header.load_const.packing = load_const_scalar_lo_19bits_sext;
            header.load_const.packed_value = lc->value[0].u64;
         }
         break;

      case 32:
         if ((lc->value[0].u32 & 0x1fff) == 0) {
            header.load_const.packing = load_const_scalar_hi_19bits;
            header.load_const.packed_value = lc->value[0].u32 >> 13;
         } else if (util_mask_sign_extend(lc->value[0].i32, 19) == lc->value[0].i32) {
            header.load_const.packing = load_const_scalar_lo_19bits_sext;
            header.load_const.packed_value = lc->value[0].u32;
         }
         break;

      case 16:
         header.load_const.packing = load_const_scalar_lo_19bits_sext;
         header.load_const.packed_value = lc->value[0].u16;
         break;
      case 8:
         header.load_const.packing = load_const_scalar_lo_19bits_sext;
         header.load_const.packed_value = lc->value[0].u8;
         break;
      case 1:
         header.load_const.packing = load_const_scalar_lo_19bits_sext;
         header.load_const.packed_value = lc->value[0].b;
         break;
      default:
         unreachable("invalid bit_size");
      }
   }

   blob_write_uint32(ctx->blob, header.u32);

   if (header.load_const.packing == load_const_full) {
      switch (lc->def.bit_size) {
      case 64:
         blob_write_bytes(ctx->blob, lc->value,
                          sizeof(*lc->value) * lc->def.num_components);
         break;

      case 32:
         for (unsigned i = 0; i < lc->def.num_components; i++)
            blob_write_uint32(ctx->blob, lc->value[i].u32);
         break;

      case 16:
         for (unsigned i = 0; i < lc->def.num_components; i++)
            blob_write_uint16(ctx->blob, lc->value[i].u16);
         break;

      default:
         assert(lc->def.bit_size <= 8);
         for (unsigned i = 0; i < lc->def.num_components; i++)
            blob_write_uint8(ctx->blob, lc->value[i].u8);
         break;
      }
   }

   write_add_object(ctx, &lc->def);
}

static nir_load_const_instr *
read_load_const(read_ctx *ctx, union packed_instr header)
{
   nir_load_const_instr *lc =
      nir_load_const_instr_create(ctx->nir, header.load_const.last_component + 1,
                                  decode_bit_size_3bits(header.load_const.bit_size));
   lc->def.divergent = false;

   switch (header.load_const.packing) {
   case load_const_scalar_hi_19bits:
      switch (lc->def.bit_size) {
      case 64:
         lc->value[0].u64 = (uint64_t)header.load_const.packed_value << 45;
         break;
      case 32:
         lc->value[0].u32 = (uint64_t)header.load_const.packed_value << 13;
         break;
      default:
         unreachable("invalid bit_size");
      }
      break;

   case load_const_scalar_lo_19bits_sext:
      switch (lc->def.bit_size) {
      case 64:
         lc->value[0].i64 = ((int64_t)header.load_const.packed_value << 45) >> 45;
         break;
      case 32:
         lc->value[0].i32 = ((int32_t)header.load_const.packed_value << 13) >> 13;
         break;
      case 16:
         lc->value[0].u16 = header.load_const.packed_value;
         break;
      case 8:
         lc->value[0].u8 = header.load_const.packed_value;
         break;
      case 1:
         lc->value[0].b = header.load_const.packed_value;
         break;
      default:
         unreachable("invalid bit_size");
      }
      break;

   case load_const_full:
      switch (lc->def.bit_size) {
      case 64:
         blob_copy_bytes(ctx->blob, lc->value, sizeof(*lc->value) * lc->def.num_components);
         break;

      case 32:
         for (unsigned i = 0; i < lc->def.num_components; i++)
            lc->value[i].u32 = blob_read_uint32(ctx->blob);
         break;

      case 16:
         for (unsigned i = 0; i < lc->def.num_components; i++)
            lc->value[i].u16 = blob_read_uint16(ctx->blob);
         break;

      default:
         assert(lc->def.bit_size <= 8);
         for (unsigned i = 0; i < lc->def.num_components; i++)
            lc->value[i].u8 = blob_read_uint8(ctx->blob);
         break;
      }
      break;
   }

   read_add_object(ctx, &lc->def);
   return lc;
}

static void
write_ssa_undef(write_ctx *ctx, const nir_ssa_undef_instr *undef)
{
   assert(undef->def.num_components >= 1 && undef->def.num_components <= 16);

   union packed_instr header;
   header.u32 = 0;

   header.undef.instr_type = undef->instr.type;
   header.undef.last_component = undef->def.num_components - 1;
   header.undef.bit_size = encode_bit_size_3bits(undef->def.bit_size);

   blob_write_uint32(ctx->blob, header.u32);
   write_add_object(ctx, &undef->def);
}

static nir_ssa_undef_instr *
read_ssa_undef(read_ctx *ctx, union packed_instr header)
{
   nir_ssa_undef_instr *undef =
      nir_ssa_undef_instr_create(ctx->nir, header.undef.last_component + 1,
                                 decode_bit_size_3bits(header.undef.bit_size));

   undef->def.divergent = false;

   read_add_object(ctx, &undef->def);
   return undef;
}

union packed_tex_data {
   uint32_t u32;
   struct {
      unsigned sampler_dim:4;
      unsigned dest_type:8;
      unsigned coord_components:3;
      unsigned is_array:1;
      unsigned is_shadow:1;
      unsigned is_new_style_shadow:1;
      unsigned is_sparse:1;
      unsigned component:2;
      unsigned texture_non_uniform:1;
      unsigned sampler_non_uniform:1;
      unsigned array_is_lowered_cube:1;
      unsigned unused:6; /* Mark unused for valgrind. */
   } u;
};

static void
write_tex(write_ctx *ctx, const nir_tex_instr *tex)
{
   assert(tex->num_srcs < 16);
   assert(tex->op < 32);

   union packed_instr header;
   header.u32 = 0;

   header.tex.instr_type = tex->instr.type;
   header.tex.num_srcs = tex->num_srcs;
   header.tex.op = tex->op;

   write_dest(ctx, &tex->dest, header, tex->instr.type);

   blob_write_uint32(ctx->blob, tex->texture_index);
   blob_write_uint32(ctx->blob, tex->sampler_index);
   if (tex->op == nir_texop_tg4)
      blob_write_bytes(ctx->blob, tex->tg4_offsets, sizeof(tex->tg4_offsets));

   STATIC_ASSERT(sizeof(union packed_tex_data) == sizeof(uint32_t));
   union packed_tex_data packed = {
      .u.sampler_dim = tex->sampler_dim,
      .u.dest_type = tex->dest_type,
      .u.coord_components = tex->coord_components,
      .u.is_array = tex->is_array,
      .u.is_shadow = tex->is_shadow,
      .u.is_new_style_shadow = tex->is_new_style_shadow,
      .u.is_sparse = tex->is_sparse,
      .u.component = tex->component,
      .u.texture_non_uniform = tex->texture_non_uniform,
      .u.sampler_non_uniform = tex->sampler_non_uniform,
      .u.array_is_lowered_cube = tex->array_is_lowered_cube,
   };
   blob_write_uint32(ctx->blob, packed.u32);

   for (unsigned i = 0; i < tex->num_srcs; i++) {
      union packed_src src;
      src.u32 = 0;
      src.tex.src_type = tex->src[i].src_type;
      write_src_full(ctx, &tex->src[i].src, src);
   }
}

static nir_tex_instr *
read_tex(read_ctx *ctx, union packed_instr header)
{
   nir_tex_instr *tex = nir_tex_instr_create(ctx->nir, header.tex.num_srcs);

   read_dest(ctx, &tex->dest, &tex->instr, header);

   tex->op = header.tex.op;
   tex->texture_index = blob_read_uint32(ctx->blob);
   tex->sampler_index = blob_read_uint32(ctx->blob);
   if (tex->op == nir_texop_tg4)
      blob_copy_bytes(ctx->blob, tex->tg4_offsets, sizeof(tex->tg4_offsets));

   union packed_tex_data packed;
   packed.u32 = blob_read_uint32(ctx->blob);
   tex->sampler_dim = packed.u.sampler_dim;
   tex->dest_type = packed.u.dest_type;
   tex->coord_components = packed.u.coord_components;
   tex->is_array = packed.u.is_array;
   tex->is_shadow = packed.u.is_shadow;
   tex->is_new_style_shadow = packed.u.is_new_style_shadow;
   tex->is_sparse = packed.u.is_sparse;
   tex->component = packed.u.component;
   tex->texture_non_uniform = packed.u.texture_non_uniform;
   tex->sampler_non_uniform = packed.u.sampler_non_uniform;
   tex->array_is_lowered_cube = packed.u.array_is_lowered_cube;

   for (unsigned i = 0; i < tex->num_srcs; i++) {
      union packed_src src = read_src(ctx, &tex->src[i].src);
      tex->src[i].src_type = src.tex.src_type;
   }

   return tex;
}

static void
write_phi(write_ctx *ctx, const nir_phi_instr *phi)
{
   union packed_instr header;
   header.u32 = 0;

   header.phi.instr_type = phi->instr.type;
   header.phi.num_srcs = exec_list_length(&phi->srcs);

   /* Phi nodes are special, since they may reference SSA definitions and
    * basic blocks that don't exist yet. We leave two empty uint32_t's here,
    * and then store enough information so that a later fixup pass can fill
    * them in correctly.
    */
   write_dest(ctx, &phi->dest, header, phi->instr.type);

   nir_foreach_phi_src(src, phi) {
      assert(src->src.is_ssa);
      size_t blob_offset = blob_reserve_uint32(ctx->blob);
      ASSERTED size_t blob_offset2 = blob_reserve_uint32(ctx->blob);
      assert(blob_offset + sizeof(uint32_t) == blob_offset2);
      write_phi_fixup fixup = {
         .blob_offset = blob_offset,
         .src = src->src.ssa,
         .block = src->pred,
      };
      util_dynarray_append(&ctx->phi_fixups, write_phi_fixup, fixup);
   }
}

static void
write_fixup_phis(write_ctx *ctx)
{
   util_dynarray_foreach(&ctx->phi_fixups, write_phi_fixup, fixup) {
      blob_overwrite_uint32(ctx->blob, fixup->blob_offset,
                            write_lookup_object(ctx, fixup->src));
      blob_overwrite_uint32(ctx->blob, fixup->blob_offset + sizeof(uint32_t),
                            write_lookup_object(ctx, fixup->block));
   }

   util_dynarray_clear(&ctx->phi_fixups);
}

static nir_phi_instr *
read_phi(read_ctx *ctx, nir_block *blk, union packed_instr header)
{
   nir_phi_instr *phi = nir_phi_instr_create(ctx->nir);

   read_dest(ctx, &phi->dest, &phi->instr, header);

   /* For similar reasons as before, we just store the index directly into the
    * pointer, and let a later pass resolve the phi sources.
    *
    * In order to ensure that the copied sources (which are just the indices
    * from the blob for now) don't get inserted into the old shader's use-def
    * lists, we have to add the phi instruction *before* we set up its
    * sources.
    */
   nir_instr_insert_after_block(blk, &phi->instr);

   for (unsigned i = 0; i < header.phi.num_srcs; i++) {
      nir_ssa_def *def = (nir_ssa_def *)(uintptr_t) blob_read_uint32(ctx->blob);
      nir_block *pred = (nir_block *)(uintptr_t) blob_read_uint32(ctx->blob);
      nir_phi_src *src = nir_phi_instr_add_src(phi, pred, nir_src_for_ssa(def));

      /* Since we're not letting nir_insert_instr handle use/def stuff for us,
       * we have to set the parent_instr manually.  It doesn't really matter
       * when we do it, so we might as well do it here.
       */
      src->src.parent_instr = &phi->instr;

      /* Stash it in the list of phi sources.  We'll walk this list and fix up
       * sources at the very end of read_function_impl.
       */
      list_add(&src->src.use_link, &ctx->phi_srcs);
   }

   return phi;
}

static void
read_fixup_phis(read_ctx *ctx)
{
   list_for_each_entry_safe(nir_phi_src, src, &ctx->phi_srcs, src.use_link) {
      src->pred = read_lookup_object(ctx, (uintptr_t)src->pred);
      src->src.ssa = read_lookup_object(ctx, (uintptr_t)src->src.ssa);

      /* Remove from this list */
      list_del(&src->src.use_link);

      list_addtail(&src->src.use_link, &src->src.ssa->uses);
   }
   assert(list_is_empty(&ctx->phi_srcs));
}

static void
write_jump(write_ctx *ctx, const nir_jump_instr *jmp)
{
   /* These aren't handled because they require special block linking */
   assert(jmp->type != nir_jump_goto && jmp->type != nir_jump_goto_if);

   assert(jmp->type < 4);

   union packed_instr header;
   header.u32 = 0;

   header.jump.instr_type = jmp->instr.type;
   header.jump.type = jmp->type;

   blob_write_uint32(ctx->blob, header.u32);
}

static nir_jump_instr *
read_jump(read_ctx *ctx, union packed_instr header)
{
   /* These aren't handled because they require special block linking */
   assert(header.jump.type != nir_jump_goto &&
          header.jump.type != nir_jump_goto_if);

   nir_jump_instr *jmp = nir_jump_instr_create(ctx->nir, header.jump.type);
   return jmp;
}

static void
write_call(write_ctx *ctx, const nir_call_instr *call)
{
   blob_write_uint32(ctx->blob, write_lookup_object(ctx, call->callee));

   for (unsigned i = 0; i < call->num_params; i++)
      write_src(ctx, &call->params[i]);
}

static nir_call_instr *
read_call(read_ctx *ctx)
{
   nir_function *callee = read_object(ctx);
   nir_call_instr *call = nir_call_instr_create(ctx->nir, callee);

   for (unsigned i = 0; i < call->num_params; i++)
      read_src(ctx, &call->params[i]);

   return call;
}

static void
write_instr(write_ctx *ctx, const nir_instr *instr)
{
   /* We have only 4 bits for the instruction type. */
   assert(instr->type < 16);

   switch (instr->type) {
   case nir_instr_type_alu:
      write_alu(ctx, nir_instr_as_alu(instr));
      break;
   case nir_instr_type_deref:
      write_deref(ctx, nir_instr_as_deref(instr));
      break;
   case nir_instr_type_intrinsic:
      write_intrinsic(ctx, nir_instr_as_intrinsic(instr));
      break;
   case nir_instr_type_load_const:
      write_load_const(ctx, nir_instr_as_load_const(instr));
      break;
   case nir_instr_type_ssa_undef:
      write_ssa_undef(ctx, nir_instr_as_ssa_undef(instr));
      break;
   case nir_instr_type_tex:
      write_tex(ctx, nir_instr_as_tex(instr));
      break;
   case nir_instr_type_phi:
      write_phi(ctx, nir_instr_as_phi(instr));
      break;
   case nir_instr_type_jump:
      write_jump(ctx, nir_instr_as_jump(instr));
      break;
   case nir_instr_type_call:
      blob_write_uint32(ctx->blob, instr->type);
      write_call(ctx, nir_instr_as_call(instr));
      break;
   case nir_instr_type_parallel_copy:
      unreachable("Cannot write parallel copies");
   default:
      unreachable("bad instr type");
   }
}

/* Return the number of instructions read. */
static unsigned
read_instr(read_ctx *ctx, nir_block *block)
{
   STATIC_ASSERT(sizeof(union packed_instr) == 4);
   union packed_instr header;
   header.u32 = blob_read_uint32(ctx->blob);
   nir_instr *instr;

   switch (header.any.instr_type) {
   case nir_instr_type_alu:
      for (unsigned i = 0; i <= header.alu.num_followup_alu_sharing_header; i++)
         nir_instr_insert_after_block(block, &read_alu(ctx, header)->instr);
      return header.alu.num_followup_alu_sharing_header + 1;
   case nir_instr_type_deref:
      instr = &read_deref(ctx, header)->instr;
      break;
   case nir_instr_type_intrinsic:
      instr = &read_intrinsic(ctx, header)->instr;
      break;
   case nir_instr_type_load_const:
      instr = &read_load_const(ctx, header)->instr;
      break;
   case nir_instr_type_ssa_undef:
      instr = &read_ssa_undef(ctx, header)->instr;
      break;
   case nir_instr_type_tex:
      instr = &read_tex(ctx, header)->instr;
      break;
   case nir_instr_type_phi:
      /* Phi instructions are a bit of a special case when reading because we
       * don't want inserting the instruction to automatically handle use/defs
       * for us.  Instead, we need to wait until all the blocks/instructions
       * are read so that we can set their sources up.
       */
      read_phi(ctx, block, header);
      return 1;
   case nir_instr_type_jump:
      instr = &read_jump(ctx, header)->instr;
      break;
   case nir_instr_type_call:
      instr = &read_call(ctx)->instr;
      break;
   case nir_instr_type_parallel_copy:
      unreachable("Cannot read parallel copies");
   default:
      unreachable("bad instr type");
   }

   nir_instr_insert_after_block(block, instr);
   return 1;
}

static void
write_block(write_ctx *ctx, const nir_block *block)
{
   write_add_object(ctx, block);
   blob_write_uint32(ctx->blob, exec_list_length(&block->instr_list));

   ctx->last_instr_type = ~0;
   ctx->last_alu_header_offset = 0;

   nir_foreach_instr(instr, block) {
      write_instr(ctx, instr);
      ctx->last_instr_type = instr->type;
   }
}

static void
read_block(read_ctx *ctx, struct exec_list *cf_list)
{
   /* Don't actually create a new block.  Just use the one from the tail of
    * the list.  NIR guarantees that the tail of the list is a block and that
    * no two blocks are side-by-side in the IR;  It should be empty.
    */
   nir_block *block =
      exec_node_data(nir_block, exec_list_get_tail(cf_list), cf_node.node);

   read_add_object(ctx, block);
   unsigned num_instrs = blob_read_uint32(ctx->blob);
   for (unsigned i = 0; i < num_instrs;) {
      i += read_instr(ctx, block);
   }
}

static void
write_cf_list(write_ctx *ctx, const struct exec_list *cf_list);

static void
read_cf_list(read_ctx *ctx, struct exec_list *cf_list);

static void
write_if(write_ctx *ctx, nir_if *nif)
{
   write_src(ctx, &nif->condition);
   blob_write_uint8(ctx->blob, nif->control);

   write_cf_list(ctx, &nif->then_list);
   write_cf_list(ctx, &nif->else_list);
}

static void
read_if(read_ctx *ctx, struct exec_list *cf_list)
{
   nir_if *nif = nir_if_create(ctx->nir);

   read_src(ctx, &nif->condition);
   nif->control = blob_read_uint8(ctx->blob);

   nir_cf_node_insert_end(cf_list, &nif->cf_node);

   read_cf_list(ctx, &nif->then_list);
   read_cf_list(ctx, &nif->else_list);
}

static void
write_loop(write_ctx *ctx, nir_loop *loop)
{
   blob_write_uint8(ctx->blob, loop->control);
   blob_write_uint8(ctx->blob, loop->divergent);
   write_cf_list(ctx, &loop->body);
}

static void
read_loop(read_ctx *ctx, struct exec_list *cf_list)
{
   nir_loop *loop = nir_loop_create(ctx->nir);

   nir_cf_node_insert_end(cf_list, &loop->cf_node);

   loop->control = blob_read_uint8(ctx->blob);
   loop->divergent = blob_read_uint8(ctx->blob);
   read_cf_list(ctx, &loop->body);
}

static void
write_cf_node(write_ctx *ctx, nir_cf_node *cf)
{
   blob_write_uint32(ctx->blob, cf->type);

   switch (cf->type) {
   case nir_cf_node_block:
      write_block(ctx, nir_cf_node_as_block(cf));
      break;
   case nir_cf_node_if:
      write_if(ctx, nir_cf_node_as_if(cf));
      break;
   case nir_cf_node_loop:
      write_loop(ctx, nir_cf_node_as_loop(cf));
      break;
   default:
      unreachable("bad cf type");
   }
}

static void
read_cf_node(read_ctx *ctx, struct exec_list *list)
{
   nir_cf_node_type type = blob_read_uint32(ctx->blob);

   switch (type) {
   case nir_cf_node_block:
      read_block(ctx, list);
      break;
   case nir_cf_node_if:
      read_if(ctx, list);
      break;
   case nir_cf_node_loop:
      read_loop(ctx, list);
      break;
   default:
      unreachable("bad cf type");
   }
}

static void
write_cf_list(write_ctx *ctx, const struct exec_list *cf_list)
{
   blob_write_uint32(ctx->blob, exec_list_length(cf_list));
   foreach_list_typed(nir_cf_node, cf, node, cf_list) {
      write_cf_node(ctx, cf);
   }
}

static void
read_cf_list(read_ctx *ctx, struct exec_list *cf_list)
{
   uint32_t num_cf_nodes = blob_read_uint32(ctx->blob);
   for (unsigned i = 0; i < num_cf_nodes; i++)
      read_cf_node(ctx, cf_list);
}

static void
write_function_impl(write_ctx *ctx, const nir_function_impl *fi)
{
   blob_write_uint8(ctx->blob, fi->structured);
   blob_write_uint8(ctx->blob, !!fi->preamble);

   if (fi->preamble)
      blob_write_uint32(ctx->blob, write_lookup_object(ctx, fi->preamble));

   write_var_list(ctx, &fi->locals);
   write_reg_list(ctx, &fi->registers);
   blob_write_uint32(ctx->blob, fi->reg_alloc);

   write_cf_list(ctx, &fi->body);
   write_fixup_phis(ctx);
}

static nir_function_impl *
read_function_impl(read_ctx *ctx, nir_function *fxn)
{
   nir_function_impl *fi = nir_function_impl_create_bare(ctx->nir);
   fi->function = fxn;

   fi->structured = blob_read_uint8(ctx->blob);
   bool preamble = blob_read_uint8(ctx->blob);

   if (preamble)
      fi->preamble = read_object(ctx);

   read_var_list(ctx, &fi->locals);
   read_reg_list(ctx, &fi->registers);
   fi->reg_alloc = blob_read_uint32(ctx->blob);

   read_cf_list(ctx, &fi->body);
   read_fixup_phis(ctx);

   fi->valid_metadata = 0;

   return fi;
}

static void
write_function(write_ctx *ctx, const nir_function *fxn)
{
   uint32_t flags = 0;
   if (fxn->is_entrypoint)
      flags |= 0x1;
   if (fxn->is_preamble)
      flags |= 0x2;
   if (fxn->name)
      flags |= 0x4;
   if (fxn->impl)
      flags |= 0x8;
   blob_write_uint32(ctx->blob, flags);
   if (fxn->name)
      blob_write_string(ctx->blob, fxn->name);

   write_add_object(ctx, fxn);

   blob_write_uint32(ctx->blob, fxn->num_params);
   for (unsigned i = 0; i < fxn->num_params; i++) {
      uint32_t val =
         ((uint32_t)fxn->params[i].num_components) |
         ((uint32_t)fxn->params[i].bit_size) << 8;
      blob_write_uint32(ctx->blob, val);
   }

   /* At first glance, it looks like we should write the function_impl here.
    * However, call instructions need to be able to reference at least the
    * function and those will get processed as we write the function_impls.
    * We stop here and write function_impls as a second pass.
    */
}

static void
read_function(read_ctx *ctx)
{
   uint32_t flags = blob_read_uint32(ctx->blob);
   bool has_name = flags & 0x4;
   char *name = has_name ? blob_read_string(ctx->blob) : NULL;

   nir_function *fxn = nir_function_create(ctx->nir, name);

   read_add_object(ctx, fxn);

   fxn->num_params = blob_read_uint32(ctx->blob);
   fxn->params = ralloc_array(fxn, nir_parameter, fxn->num_params);
   for (unsigned i = 0; i < fxn->num_params; i++) {
      uint32_t val = blob_read_uint32(ctx->blob);
      fxn->params[i].num_components = val & 0xff;
      fxn->params[i].bit_size = (val >> 8) & 0xff;
   }

   fxn->is_entrypoint = flags & 0x1;
   fxn->is_preamble = flags & 0x2;
   if (flags & 0x8)
      fxn->impl = NIR_SERIALIZE_FUNC_HAS_IMPL;
}

static void
write_xfb_info(write_ctx *ctx, const nir_xfb_info *xfb)
{
   if (xfb == NULL) {
      blob_write_uint32(ctx->blob, 0);
   } else {
      size_t size = nir_xfb_info_size(xfb->output_count);
      assert(size <= UINT32_MAX);
      blob_write_uint32(ctx->blob, size);
      blob_write_bytes(ctx->blob, xfb, size);
   }
}

static nir_xfb_info *
read_xfb_info(read_ctx *ctx)
{
   uint32_t size = blob_read_uint32(ctx->blob);
   if (size == 0)
      return NULL;

   struct nir_xfb_info *xfb = ralloc_size(ctx->nir, size);
   blob_copy_bytes(ctx->blob, (void *)xfb, size);

   return xfb;
}

/**
 * Serialize NIR into a binary blob.
 *
 * \param strip  Don't serialize information only useful for debugging,
 *               such as variable names, making cache hits from similar
 *               shaders more likely.
 */
void
nir_serialize(struct blob *blob, const nir_shader *nir, bool strip)
{
   write_ctx ctx = {0};
   ctx.remap_table = _mesa_pointer_hash_table_create(NULL);
   ctx.blob = blob;
   ctx.nir = nir;
   ctx.strip = strip;
   util_dynarray_init(&ctx.phi_fixups, NULL);

   size_t idx_size_offset = blob_reserve_uint32(blob);

   struct shader_info info = nir->info;
   uint32_t strings = 0;
   if (!strip && info.name)
      strings |= 0x1;
   if (!strip && info.label)
      strings |= 0x2;
   blob_write_uint32(blob, strings);
   if (!strip && info.name)
      blob_write_string(blob, info.name);
   if (!strip && info.label)
      blob_write_string(blob, info.label);
   info.name = info.label = NULL;
   blob_write_bytes(blob, (uint8_t *) &info, sizeof(info));

   write_var_list(&ctx, &nir->variables);

   blob_write_uint32(blob, nir->num_inputs);
   blob_write_uint32(blob, nir->num_uniforms);
   blob_write_uint32(blob, nir->num_outputs);
   blob_write_uint32(blob, nir->scratch_size);

   blob_write_uint32(blob, exec_list_length(&nir->functions));
   nir_foreach_function(fxn, nir) {
      write_function(&ctx, fxn);
   }

   nir_foreach_function(fxn, nir) {
      if (fxn->impl)
         write_function_impl(&ctx, fxn->impl);
   }

   blob_write_uint32(blob, nir->constant_data_size);
   if (nir->constant_data_size > 0)
      blob_write_bytes(blob, nir->constant_data, nir->constant_data_size);

   write_xfb_info(&ctx, nir->xfb_info);

   if (nir->info.stage == MESA_SHADER_KERNEL) {
      blob_write_uint32(blob, nir->printf_info_count);
      for (int i = 0; i < nir->printf_info_count; i++) {
         u_printf_info *info = &nir->printf_info[i];
         blob_write_uint32(blob, info->num_args);
         blob_write_uint32(blob, info->string_size);
         blob_write_bytes(blob, info->arg_sizes,
                          info->num_args * sizeof(*info->arg_sizes));
         /* we can't use blob_write_string, because it contains multiple NULL
          * terminated strings */
         blob_write_bytes(blob, info->strings,
                          info->string_size * sizeof(*info->strings));
      }
   }

   blob_overwrite_uint32(blob, idx_size_offset, ctx.next_idx);

   _mesa_hash_table_destroy(ctx.remap_table, NULL);
   util_dynarray_fini(&ctx.phi_fixups);
}

nir_shader *
nir_deserialize(void *mem_ctx,
                const struct nir_shader_compiler_options *options,
                struct blob_reader *blob)
{
   read_ctx ctx = {0};
   ctx.blob = blob;
   list_inithead(&ctx.phi_srcs);
   ctx.idx_table_len = blob_read_uint32(blob);
   ctx.idx_table = calloc(ctx.idx_table_len, sizeof(uintptr_t));

   uint32_t strings = blob_read_uint32(blob);
   char *name = (strings & 0x1) ? blob_read_string(blob) : NULL;
   char *label = (strings & 0x2) ? blob_read_string(blob) : NULL;

   struct shader_info info;
   blob_copy_bytes(blob, (uint8_t *) &info, sizeof(info));

   ctx.nir = nir_shader_create(mem_ctx, info.stage, options, NULL);

   info.name = name ? ralloc_strdup(ctx.nir, name) : NULL;
   info.label = label ? ralloc_strdup(ctx.nir, label) : NULL;

   ctx.nir->info = info;

   read_var_list(&ctx, &ctx.nir->variables);

   ctx.nir->num_inputs = blob_read_uint32(blob);
   ctx.nir->num_uniforms = blob_read_uint32(blob);
   ctx.nir->num_outputs = blob_read_uint32(blob);
   ctx.nir->scratch_size = blob_read_uint32(blob);

   unsigned num_functions = blob_read_uint32(blob);
   for (unsigned i = 0; i < num_functions; i++)
      read_function(&ctx);

   nir_foreach_function(fxn, ctx.nir) {
      if (fxn->impl == NIR_SERIALIZE_FUNC_HAS_IMPL)
         fxn->impl = read_function_impl(&ctx, fxn);
   }

   ctx.nir->constant_data_size = blob_read_uint32(blob);
   if (ctx.nir->constant_data_size > 0) {
      ctx.nir->constant_data =
         ralloc_size(ctx.nir, ctx.nir->constant_data_size);
      blob_copy_bytes(blob, ctx.nir->constant_data,
                      ctx.nir->constant_data_size);
   }

   ctx.nir->xfb_info = read_xfb_info(&ctx);

   if (ctx.nir->info.stage == MESA_SHADER_KERNEL) {
      ctx.nir->printf_info_count = blob_read_uint32(blob);
      ctx.nir->printf_info =
         ralloc_array(ctx.nir, u_printf_info, ctx.nir->printf_info_count);

      for (int i = 0; i < ctx.nir->printf_info_count; i++) {
         u_printf_info *info = &ctx.nir->printf_info[i];
         info->num_args = blob_read_uint32(blob);
         info->string_size = blob_read_uint32(blob);
         info->arg_sizes = ralloc_array(ctx.nir, unsigned, info->num_args);
         blob_copy_bytes(blob, info->arg_sizes,
                         info->num_args * sizeof(*info->arg_sizes));
         info->strings = ralloc_array(ctx.nir, char, info->string_size);
         blob_copy_bytes(blob, info->strings,
                         info->string_size * sizeof(*info->strings));
      }
   }

   free(ctx.idx_table);

   nir_validate_shader(ctx.nir, "after deserialize");

   return ctx.nir;
}

void
nir_shader_serialize_deserialize(nir_shader *shader)
{
   const struct nir_shader_compiler_options *options = shader->options;

   struct blob writer;
   blob_init(&writer);
   nir_serialize(&writer, shader, false);

   /* Delete all of dest's ralloc children but leave dest alone */
   void *dead_ctx = ralloc_context(NULL);
   ralloc_adopt(dead_ctx, shader);
   ralloc_free(dead_ctx);

   dead_ctx = ralloc_context(NULL);

   struct blob_reader reader;
   blob_reader_init(&reader, writer.data, writer.size);
   nir_shader *copy = nir_deserialize(dead_ctx, options, &reader);

   blob_finish(&writer);

   nir_shader_replace(shader, copy);
   ralloc_free(dead_ctx);
}
