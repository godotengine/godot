/*
 * Copyright © 2015 Intel Corporation
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
 *
 */

#include "vtn_private.h"
#include "spirv_info.h"
#include "nir_deref.h"
#include <vulkan/vulkan_core.h>

#include "drivers/d3d12/d3d12_godot_nir_bridge.h"

static struct vtn_pointer*
vtn_align_pointer(struct vtn_builder *b, struct vtn_pointer *ptr,
                  unsigned alignment)
{
   if (alignment == 0)
      return ptr;

   if (!util_is_power_of_two_nonzero(alignment)) {
      vtn_warn("Provided alignment is not a power of two");
      alignment = 1 << (ffs(alignment) - 1);
   }

   /* If this pointer doesn't have a deref, bail.  This either means we're
    * using the old offset+alignment pointers which don't support carrying
    * alignment information or we're a pointer that is below the block
    * boundary in our access chain in which case alignment is meaningless.
    */
   if (ptr->deref == NULL)
      return ptr;

   /* Ignore alignment information on logical pointers.  This way, we don't
    * trip up drivers with unnecessary casts.
    */
   nir_address_format addr_format = vtn_mode_to_address_format(b, ptr->mode);
   if (addr_format == nir_address_format_logical)
      return ptr;

   struct vtn_pointer *copy = ralloc(b, struct vtn_pointer);
   *copy = *ptr;
   copy->deref = nir_alignment_deref_cast(&b->nb, ptr->deref, alignment, 0);

   return copy;
}

static void
ptr_decoration_cb(struct vtn_builder *b, struct vtn_value *val, int member,
                  const struct vtn_decoration *dec, void *void_ptr)
{
   struct vtn_pointer *ptr = void_ptr;

   switch (dec->decoration) {
   case SpvDecorationNonUniformEXT:
      ptr->access |= ACCESS_NON_UNIFORM;
      break;

   default:
      break;
   }
}

struct access_align {
   enum gl_access_qualifier access;
   uint32_t alignment;
};

static void
access_align_cb(struct vtn_builder *b, struct vtn_value *val, int member,
                const struct vtn_decoration *dec, void *void_ptr)
{
   struct access_align *aa = void_ptr;

   switch (dec->decoration) {
   case SpvDecorationAlignment:
      aa->alignment = dec->operands[0];
      break;

   case SpvDecorationNonUniformEXT:
      aa->access |= ACCESS_NON_UNIFORM;
      break;

   default:
      break;
   }
}

static struct vtn_pointer*
vtn_decorate_pointer(struct vtn_builder *b, struct vtn_value *val,
                     struct vtn_pointer *ptr)
{
   struct access_align aa = { 0, };
   vtn_foreach_decoration(b, val, access_align_cb, &aa);

   ptr = vtn_align_pointer(b, ptr, aa.alignment);

   /* If we're adding access flags, make a copy of the pointer.  We could
    * probably just OR them in without doing so but this prevents us from
    * leaking them any further than actually specified in the SPIR-V.
    */
   if (aa.access & ~ptr->access) {
      struct vtn_pointer *copy = ralloc(b, struct vtn_pointer);
      *copy = *ptr;
      copy->access |= aa.access;
      return copy;
   }

   return ptr;
}

struct vtn_value *
vtn_push_pointer(struct vtn_builder *b, uint32_t value_id,
                 struct vtn_pointer *ptr)
{
   struct vtn_value *val = vtn_push_value(b, value_id, vtn_value_type_pointer);
   val->pointer = vtn_decorate_pointer(b, val, ptr);
   return val;
}

void
vtn_copy_value(struct vtn_builder *b, uint32_t src_value_id,
               uint32_t dst_value_id)
{
   struct vtn_value *src = vtn_untyped_value(b, src_value_id);
   struct vtn_value *dst = vtn_untyped_value(b, dst_value_id);
   struct vtn_value src_copy = *src;

   vtn_fail_if(dst->value_type != vtn_value_type_invalid,
               "SPIR-V id %u has already been written by another instruction",
               dst_value_id);

   vtn_fail_if(dst->type->id != src->type->id,
               "Result Type must equal Operand type");

   src_copy.name = dst->name;
   src_copy.decoration = dst->decoration;
   src_copy.type = dst->type;
   *dst = src_copy;

   if (dst->value_type == vtn_value_type_pointer)
      dst->pointer = vtn_decorate_pointer(b, dst, dst->pointer);
}

static struct vtn_access_chain *
vtn_access_chain_create(struct vtn_builder *b, unsigned length)
{
   struct vtn_access_chain *chain;

   /* Subtract 1 from the length since there's already one built in */
   size_t size = sizeof(*chain) +
                 (MAX2(length, 1) - 1) * sizeof(chain->link[0]);
   chain = rzalloc_size(b, size);
   chain->length = length;

   return chain;
}

static bool
vtn_mode_is_cross_invocation(struct vtn_builder *b,
                             enum vtn_variable_mode mode)
{
   /* TODO: add TCS here once nir_remove_unused_io_vars() can handle vector indexing. */
   bool cross_invocation_outputs = b->shader->info.stage == MESA_SHADER_MESH;
   return mode == vtn_variable_mode_ssbo ||
          mode == vtn_variable_mode_ubo ||
          mode == vtn_variable_mode_phys_ssbo ||
          mode == vtn_variable_mode_push_constant ||
          mode == vtn_variable_mode_workgroup ||
          mode == vtn_variable_mode_cross_workgroup ||
          (cross_invocation_outputs && mode == vtn_variable_mode_output) ||
          (b->shader->info.stage == MESA_SHADER_TASK && mode == vtn_variable_mode_task_payload);
}

static bool
vtn_pointer_is_external_block(struct vtn_builder *b,
                              struct vtn_pointer *ptr)
{
   return ptr->mode == vtn_variable_mode_ssbo ||
          ptr->mode == vtn_variable_mode_ubo ||
          ptr->mode == vtn_variable_mode_phys_ssbo;
}

static nir_ssa_def *
vtn_access_link_as_ssa(struct vtn_builder *b, struct vtn_access_link link,
                       unsigned stride, unsigned bit_size)
{
   vtn_assert(stride > 0);
   if (link.mode == vtn_access_mode_literal) {
      return nir_imm_intN_t(&b->nb, link.id * stride, bit_size);
   } else {
      nir_ssa_def *ssa = vtn_ssa_value(b, link.id)->def;
      if (ssa->bit_size != bit_size)
         ssa = nir_i2iN(&b->nb, ssa, bit_size);
      return nir_imul_imm(&b->nb, ssa, stride);
   }
}

static VkDescriptorType
vk_desc_type_for_mode(struct vtn_builder *b, enum vtn_variable_mode mode)
{
   switch (mode) {
   case vtn_variable_mode_ubo:
      return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
   case vtn_variable_mode_ssbo:
      return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
   case vtn_variable_mode_accel_struct:
      return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
   default:
      vtn_fail("Invalid mode for vulkan_resource_index");
   }
}

static nir_ssa_def *
vtn_variable_resource_index(struct vtn_builder *b, struct vtn_variable *var,
                            nir_ssa_def *desc_array_index)
{
   vtn_assert(b->options->environment == NIR_SPIRV_VULKAN);

   if (!desc_array_index)
      desc_array_index = nir_imm_int(&b->nb, 0);

   if (b->vars_used_indirectly) {
      vtn_assert(var->var);
      _mesa_set_add(b->vars_used_indirectly, var->var);
   }

   nir_intrinsic_instr *instr =
      nir_intrinsic_instr_create(b->nb.shader,
                                 nir_intrinsic_vulkan_resource_index);
   instr->src[0] = nir_src_for_ssa(desc_array_index);
   nir_intrinsic_set_desc_set(instr, var->descriptor_set);
   nir_intrinsic_set_binding(instr, var->binding);
   nir_intrinsic_set_desc_type(instr, vk_desc_type_for_mode(b, var->mode));

   nir_address_format addr_format = vtn_mode_to_address_format(b, var->mode);
   nir_ssa_dest_init(&instr->instr, &instr->dest,
                     nir_address_format_num_components(addr_format),
                     nir_address_format_bit_size(addr_format), NULL);
   instr->num_components = instr->dest.ssa.num_components;
   nir_builder_instr_insert(&b->nb, &instr->instr);

   return &instr->dest.ssa;
}

static nir_ssa_def *
vtn_resource_reindex(struct vtn_builder *b, enum vtn_variable_mode mode,
                     nir_ssa_def *base_index, nir_ssa_def *offset_index)
{
   vtn_assert(b->options->environment == NIR_SPIRV_VULKAN);

   nir_intrinsic_instr *instr =
      nir_intrinsic_instr_create(b->nb.shader,
                                 nir_intrinsic_vulkan_resource_reindex);
   instr->src[0] = nir_src_for_ssa(base_index);
   instr->src[1] = nir_src_for_ssa(offset_index);
   nir_intrinsic_set_desc_type(instr, vk_desc_type_for_mode(b, mode));

   nir_address_format addr_format = vtn_mode_to_address_format(b, mode);
   nir_ssa_dest_init(&instr->instr, &instr->dest,
                     nir_address_format_num_components(addr_format),
                     nir_address_format_bit_size(addr_format), NULL);
   instr->num_components = instr->dest.ssa.num_components;
   nir_builder_instr_insert(&b->nb, &instr->instr);

   return &instr->dest.ssa;
}

static nir_ssa_def *
vtn_descriptor_load(struct vtn_builder *b, enum vtn_variable_mode mode,
                    nir_ssa_def *desc_index)
{
   vtn_assert(b->options->environment == NIR_SPIRV_VULKAN);

   nir_intrinsic_instr *desc_load =
      nir_intrinsic_instr_create(b->nb.shader,
                                 nir_intrinsic_load_vulkan_descriptor);
   desc_load->src[0] = nir_src_for_ssa(desc_index);
   nir_intrinsic_set_desc_type(desc_load, vk_desc_type_for_mode(b, mode));

   nir_address_format addr_format = vtn_mode_to_address_format(b, mode);
   nir_ssa_dest_init(&desc_load->instr, &desc_load->dest,
                     nir_address_format_num_components(addr_format),
                     nir_address_format_bit_size(addr_format), NULL);
   desc_load->num_components = desc_load->dest.ssa.num_components;
   nir_builder_instr_insert(&b->nb, &desc_load->instr);

   return &desc_load->dest.ssa;
}

static struct vtn_pointer *
vtn_pointer_dereference(struct vtn_builder *b,
                        struct vtn_pointer *base,
                        struct vtn_access_chain *deref_chain)
{
   struct vtn_type *type = base->type;
   enum gl_access_qualifier access = base->access | deref_chain->access;
   unsigned idx = 0;

   nir_deref_instr *tail;
   if (base->deref) {
      tail = base->deref;
   } else if (b->options->environment == NIR_SPIRV_VULKAN &&
              (vtn_pointer_is_external_block(b, base) ||
               base->mode == vtn_variable_mode_accel_struct)) {
      nir_ssa_def *block_index = base->block_index;

      /* We dereferencing an external block pointer.  Correctness of this
       * operation relies on one particular line in the SPIR-V spec, section
       * entitled "Validation Rules for Shader Capabilities":
       *
       *    "Block and BufferBlock decorations cannot decorate a structure
       *    type that is nested at any level inside another structure type
       *    decorated with Block or BufferBlock."
       *
       * This means that we can detect the point where we cross over from
       * descriptor indexing to buffer indexing by looking for the block
       * decorated struct type.  Anything before the block decorated struct
       * type is a descriptor indexing operation and anything after the block
       * decorated struct is a buffer offset operation.
       */

      /* Figure out the descriptor array index if any
       *
       * Some of the Vulkan CTS tests with hand-rolled SPIR-V have been known
       * to forget the Block or BufferBlock decoration from time to time.
       * It's more robust if we check for both !block_index and for the type
       * to contain a block.  This way there's a decent chance that arrays of
       * UBOs/SSBOs will work correctly even if variable pointers are
       * completley toast.
       */
      nir_ssa_def *desc_arr_idx = NULL;
      if (!block_index || vtn_type_contains_block(b, type) ||
          base->mode == vtn_variable_mode_accel_struct) {
         /* If our type contains a block, then we're still outside the block
          * and we need to process enough levels of dereferences to get inside
          * of it.  Same applies to acceleration structures.
          */
         if (deref_chain->ptr_as_array) {
            unsigned aoa_size = glsl_get_aoa_size(type->type);
            desc_arr_idx = vtn_access_link_as_ssa(b, deref_chain->link[idx],
                                                  MAX2(aoa_size, 1), 32);
            idx++;
         }

         for (; idx < deref_chain->length; idx++) {
            if (type->base_type != vtn_base_type_array) {
               vtn_assert(type->base_type == vtn_base_type_struct);
               break;
            }

            unsigned aoa_size = glsl_get_aoa_size(type->array_element->type);
            nir_ssa_def *arr_offset =
               vtn_access_link_as_ssa(b, deref_chain->link[idx],
                                      MAX2(aoa_size, 1), 32);
            if (desc_arr_idx)
               desc_arr_idx = nir_iadd(&b->nb, desc_arr_idx, arr_offset);
            else
               desc_arr_idx = arr_offset;

            type = type->array_element;
            access |= type->access;
         }
      }

      if (!block_index) {
         vtn_assert(base->var && base->type);
         block_index = vtn_variable_resource_index(b, base->var, desc_arr_idx);
      } else if (desc_arr_idx) {
         block_index = vtn_resource_reindex(b, base->mode,
                                            block_index, desc_arr_idx);
      }

      if (idx == deref_chain->length) {
         /* The entire deref was consumed in finding the block index.  Return
          * a pointer which just has a block index and a later access chain
          * will dereference deeper.
          */
         struct vtn_pointer *ptr = rzalloc(b, struct vtn_pointer);
         ptr->mode = base->mode;
         ptr->type = type;
         ptr->block_index = block_index;
         ptr->access = access;
         return ptr;
      }

      /* If we got here, there's more access chain to handle and we have the
       * final block index.  Insert a descriptor load and cast to a deref to
       * start the deref chain.
       */
      nir_ssa_def *desc = vtn_descriptor_load(b, base->mode, block_index);

      assert(base->mode == vtn_variable_mode_ssbo ||
             base->mode == vtn_variable_mode_ubo);
      nir_variable_mode nir_mode =
         base->mode == vtn_variable_mode_ssbo ? nir_var_mem_ssbo : nir_var_mem_ubo;

      tail = nir_build_deref_cast(&b->nb, desc, nir_mode,
                                  vtn_type_get_nir_type(b, type, base->mode),
                                  base->ptr_type->stride);
   } else if (base->mode == vtn_variable_mode_shader_record) {
      /* For ShaderRecordBufferKHR variables, we don't have a nir_variable.
       * It's just a fancy handle around a pointer to the shader record for
       * the current shader.
       */
      tail = nir_build_deref_cast(&b->nb, nir_load_shader_record_ptr(&b->nb),
                                  nir_var_mem_constant,
                                  vtn_type_get_nir_type(b, base->type,
                                                           base->mode),
                                  0 /* ptr_as_array stride */);
   } else {
      assert(base->var && base->var->var);
      tail = nir_build_deref_var(&b->nb, base->var->var);
      if (base->ptr_type && base->ptr_type->type) {
         tail->dest.ssa.num_components =
            glsl_get_vector_elements(base->ptr_type->type);
         tail->dest.ssa.bit_size = glsl_get_bit_size(base->ptr_type->type);
      }
   }

   if (idx == 0 && deref_chain->ptr_as_array) {
      /* We start with a deref cast to get the stride.  Hopefully, we'll be
       * able to delete that cast eventually.
       */
      tail = nir_build_deref_cast(&b->nb, &tail->dest.ssa, tail->modes,
                                  tail->type, base->ptr_type->stride);

      nir_ssa_def *index = vtn_access_link_as_ssa(b, deref_chain->link[0], 1,
                                                  tail->dest.ssa.bit_size);
      tail = nir_build_deref_ptr_as_array(&b->nb, tail, index);
      idx++;
   }

   for (; idx < deref_chain->length; idx++) {
      if (glsl_type_is_struct_or_ifc(type->type)) {
         vtn_assert(deref_chain->link[idx].mode == vtn_access_mode_literal);
         unsigned field = deref_chain->link[idx].id;
         tail = nir_build_deref_struct(&b->nb, tail, field);
         type = type->members[field];
      } else {
         nir_ssa_def *arr_index =
            vtn_access_link_as_ssa(b, deref_chain->link[idx], 1,
                                   tail->dest.ssa.bit_size);
         tail = nir_build_deref_array(&b->nb, tail, arr_index);
         type = type->array_element;
      }
      tail->arr.in_bounds = deref_chain->in_bounds;

      access |= type->access;
   }

   struct vtn_pointer *ptr = rzalloc(b, struct vtn_pointer);
   ptr->mode = base->mode;
   ptr->type = type;
   ptr->var = base->var;
   ptr->deref = tail;
   ptr->access = access;

   return ptr;
}

nir_deref_instr *
vtn_pointer_to_deref(struct vtn_builder *b, struct vtn_pointer *ptr)
{
   if (!ptr->deref) {
      struct vtn_access_chain chain = {
         .length = 0,
      };
      ptr = vtn_pointer_dereference(b, ptr, &chain);
   }

   return ptr->deref;
}

static void
_vtn_local_load_store(struct vtn_builder *b, bool load, nir_deref_instr *deref,
                      struct vtn_ssa_value *inout,
                      enum gl_access_qualifier access)
{
   if (glsl_type_is_vector_or_scalar(deref->type)) {
      if (load) {
         inout->def = nir_load_deref_with_access(&b->nb, deref, access);
      } else {
         nir_store_deref_with_access(&b->nb, deref, inout->def, ~0, access);
      }
   } else if (glsl_type_is_array(deref->type) ||
              glsl_type_is_matrix(deref->type)) {
      unsigned elems = glsl_get_length(deref->type);
      for (unsigned i = 0; i < elems; i++) {
         nir_deref_instr *child =
            nir_build_deref_array_imm(&b->nb, deref, i);
         _vtn_local_load_store(b, load, child, inout->elems[i], access);
      }
   } else {
      vtn_assert(glsl_type_is_struct_or_ifc(deref->type));
      unsigned elems = glsl_get_length(deref->type);
      for (unsigned i = 0; i < elems; i++) {
         nir_deref_instr *child = nir_build_deref_struct(&b->nb, deref, i);
         _vtn_local_load_store(b, load, child, inout->elems[i], access);
      }
   }
}

nir_deref_instr *
vtn_nir_deref(struct vtn_builder *b, uint32_t id)
{
   struct vtn_pointer *ptr = vtn_pointer(b, id);
   return vtn_pointer_to_deref(b, ptr);
}

/*
 * Gets the NIR-level deref tail, which may have as a child an array deref
 * selecting which component due to OpAccessChain supporting per-component
 * indexing in SPIR-V.
 */
static nir_deref_instr *
get_deref_tail(nir_deref_instr *deref)
{
   if (deref->deref_type != nir_deref_type_array)
      return deref;

   nir_deref_instr *parent =
      nir_instr_as_deref(deref->parent.ssa->parent_instr);

   if (glsl_type_is_vector(parent->type))
      return parent;
   else
      return deref;
}

struct vtn_ssa_value *
vtn_local_load(struct vtn_builder *b, nir_deref_instr *src,
               enum gl_access_qualifier access)
{
   nir_deref_instr *src_tail = get_deref_tail(src);
   struct vtn_ssa_value *val = vtn_create_ssa_value(b, src_tail->type);
   _vtn_local_load_store(b, true, src_tail, val, access);

   if (src_tail != src) {
      val->type = src->type;
      val->def = nir_vector_extract(&b->nb, val->def, src->arr.index.ssa);
   }

   return val;
}

void
vtn_local_store(struct vtn_builder *b, struct vtn_ssa_value *src,
                nir_deref_instr *dest, enum gl_access_qualifier access)
{
   nir_deref_instr *dest_tail = get_deref_tail(dest);

   if (dest_tail != dest) {
      struct vtn_ssa_value *val = vtn_create_ssa_value(b, dest_tail->type);
      _vtn_local_load_store(b, true, dest_tail, val, access);

      val->def = nir_vector_insert(&b->nb, val->def, src->def,
                                   dest->arr.index.ssa);
      _vtn_local_load_store(b, false, dest_tail, val, access);
   } else {
      _vtn_local_load_store(b, false, dest_tail, src, access);
   }
}

static nir_ssa_def *
vtn_pointer_to_descriptor(struct vtn_builder *b, struct vtn_pointer *ptr)
{
   assert(ptr->mode == vtn_variable_mode_accel_struct);
   if (!ptr->block_index) {
      struct vtn_access_chain chain = {
         .length = 0,
      };
      ptr = vtn_pointer_dereference(b, ptr, &chain);
   }

   vtn_assert(ptr->deref == NULL && ptr->block_index != NULL);
   return vtn_descriptor_load(b, ptr->mode, ptr->block_index);
}

static void
_vtn_variable_load_store(struct vtn_builder *b, bool load,
                         struct vtn_pointer *ptr,
                         enum gl_access_qualifier access,
                         struct vtn_ssa_value **inout)
{
   if (ptr->mode == vtn_variable_mode_uniform ||
       ptr->mode == vtn_variable_mode_image) {
      if (ptr->type->base_type == vtn_base_type_image ||
          ptr->type->base_type == vtn_base_type_sampler) {
         /* See also our handling of OpTypeSampler and OpTypeImage */
         vtn_assert(load);
         (*inout)->def = vtn_pointer_to_ssa(b, ptr);
         return;
      } else if (ptr->type->base_type == vtn_base_type_sampled_image) {
         /* See also our handling of OpTypeSampledImage */
         vtn_assert(load);
         struct vtn_sampled_image si = {
            .image = vtn_pointer_to_deref(b, ptr),
            .sampler = vtn_pointer_to_deref(b, ptr),
         };
         (*inout)->def = vtn_sampled_image_to_nir_ssa(b, si);
         return;
      }
   } else if (ptr->mode == vtn_variable_mode_accel_struct) {
      vtn_assert(load);
      (*inout)->def = vtn_pointer_to_descriptor(b, ptr);
      return;
   }

   enum glsl_base_type base_type = glsl_get_base_type(ptr->type->type);
   switch (base_type) {
   case GLSL_TYPE_UINT:
   case GLSL_TYPE_INT:
   case GLSL_TYPE_UINT16:
   case GLSL_TYPE_INT16:
   case GLSL_TYPE_UINT8:
   case GLSL_TYPE_INT8:
   case GLSL_TYPE_UINT64:
   case GLSL_TYPE_INT64:
   case GLSL_TYPE_FLOAT:
   case GLSL_TYPE_FLOAT16:
   case GLSL_TYPE_BOOL:
   case GLSL_TYPE_DOUBLE:
      if (glsl_type_is_vector_or_scalar(ptr->type->type)) {
         /* We hit a vector or scalar; go ahead and emit the load[s] */
         nir_deref_instr *deref = vtn_pointer_to_deref(b, ptr);
         if (vtn_mode_is_cross_invocation(b, ptr->mode)) {
            /* If it's cross-invocation, we call nir_load/store_deref
             * directly.  The vtn_local_load/store helpers are too clever and
             * do magic to avoid array derefs of vectors.  That magic is both
             * less efficient than the direct load/store and, in the case of
             * stores, is broken because it creates a race condition if two
             * threads are writing to different components of the same vector
             * due to the load+insert+store it uses to emulate the array
             * deref.
             */
            if (load) {
               (*inout)->def = nir_load_deref_with_access(&b->nb, deref,
                                                          ptr->type->access | access);
            } else {
               nir_store_deref_with_access(&b->nb, deref, (*inout)->def, ~0,
                                           ptr->type->access | access);
            }
         } else {
            if (load) {
               *inout = vtn_local_load(b, deref, ptr->type->access | access);
            } else {
               vtn_local_store(b, *inout, deref, ptr->type->access | access);
            }
         }
         return;
      }
      FALLTHROUGH;

   case GLSL_TYPE_INTERFACE:
   case GLSL_TYPE_ARRAY:
   case GLSL_TYPE_STRUCT: {
      unsigned elems = glsl_get_length(ptr->type->type);
      struct vtn_access_chain chain = {
         .length = 1,
         .link = {
            { .mode = vtn_access_mode_literal, },
         }
      };
      for (unsigned i = 0; i < elems; i++) {
         chain.link[0].id = i;
         struct vtn_pointer *elem = vtn_pointer_dereference(b, ptr, &chain);
         _vtn_variable_load_store(b, load, elem, ptr->type->access | access,
                                  &(*inout)->elems[i]);
      }
      return;
   }

   default:
      vtn_fail("Invalid access chain type");
   }
}

struct vtn_ssa_value *
vtn_variable_load(struct vtn_builder *b, struct vtn_pointer *src,
                  enum gl_access_qualifier access)
{
   struct vtn_ssa_value *val = vtn_create_ssa_value(b, src->type->type);
   _vtn_variable_load_store(b, true, src, src->access | access, &val);
   return val;
}

void
vtn_variable_store(struct vtn_builder *b, struct vtn_ssa_value *src,
                   struct vtn_pointer *dest, enum gl_access_qualifier access)
{
   _vtn_variable_load_store(b, false, dest, dest->access | access, &src);
}

static void
_vtn_variable_copy(struct vtn_builder *b, struct vtn_pointer *dest,
                   struct vtn_pointer *src, enum gl_access_qualifier dest_access,
                   enum gl_access_qualifier src_access)
{
   vtn_assert(glsl_get_bare_type(src->type->type) ==
              glsl_get_bare_type(dest->type->type));
   enum glsl_base_type base_type = glsl_get_base_type(src->type->type);
   switch (base_type) {
   case GLSL_TYPE_UINT:
   case GLSL_TYPE_INT:
   case GLSL_TYPE_UINT16:
   case GLSL_TYPE_INT16:
   case GLSL_TYPE_UINT8:
   case GLSL_TYPE_INT8:
   case GLSL_TYPE_UINT64:
   case GLSL_TYPE_INT64:
   case GLSL_TYPE_FLOAT:
   case GLSL_TYPE_FLOAT16:
   case GLSL_TYPE_DOUBLE:
   case GLSL_TYPE_BOOL:
      /* At this point, we have a scalar, vector, or matrix so we know that
       * there cannot be any structure splitting still in the way.  By
       * stopping at the matrix level rather than the vector level, we
       * ensure that matrices get loaded in the optimal way even if they
       * are storred row-major in a UBO.
       */
      vtn_variable_store(b, vtn_variable_load(b, src, src_access), dest, dest_access);
      return;

   case GLSL_TYPE_INTERFACE:
   case GLSL_TYPE_ARRAY:
   case GLSL_TYPE_STRUCT: {
      struct vtn_access_chain chain = {
         .length = 1,
         .link = {
            { .mode = vtn_access_mode_literal, },
         }
      };
      unsigned elems = glsl_get_length(src->type->type);
      for (unsigned i = 0; i < elems; i++) {
         chain.link[0].id = i;
         struct vtn_pointer *src_elem =
            vtn_pointer_dereference(b, src, &chain);
         struct vtn_pointer *dest_elem =
            vtn_pointer_dereference(b, dest, &chain);

         _vtn_variable_copy(b, dest_elem, src_elem, dest_access, src_access);
      }
      return;
   }

   default:
      vtn_fail("Invalid access chain type");
   }
}

static void
vtn_variable_copy(struct vtn_builder *b, struct vtn_pointer *dest,
                  struct vtn_pointer *src, enum gl_access_qualifier dest_access,
                  enum gl_access_qualifier src_access)
{
   /* TODO: At some point, we should add a special-case for when we can
    * just emit a copy_var intrinsic.
    */
   _vtn_variable_copy(b, dest, src, dest_access, src_access);
}

static void
set_mode_system_value(struct vtn_builder *b, nir_variable_mode *mode)
{
   vtn_assert(*mode == nir_var_system_value || *mode == nir_var_shader_in ||
              /* Hack for NV_mesh_shader due to lack of dedicated storage class. */
              *mode == nir_var_mem_task_payload);
   *mode = nir_var_system_value;
}

static void
vtn_get_builtin_location(struct vtn_builder *b,
                         SpvBuiltIn builtin, int *location,
                         nir_variable_mode *mode)
{
   switch (builtin) {
   case SpvBuiltInPosition:
   case SpvBuiltInPositionPerViewNV:
      *location = VARYING_SLOT_POS;
      break;
   case SpvBuiltInPointSize:
      *location = VARYING_SLOT_PSIZ;
      break;
   case SpvBuiltInClipDistance:
   case SpvBuiltInClipDistancePerViewNV:
      *location = VARYING_SLOT_CLIP_DIST0;
      break;
   case SpvBuiltInCullDistance:
   case SpvBuiltInCullDistancePerViewNV:
      *location = VARYING_SLOT_CULL_DIST0;
      break;
   case SpvBuiltInVertexId:
   case SpvBuiltInVertexIndex:
      /* The Vulkan spec defines VertexIndex to be non-zero-based and doesn't
       * allow VertexId.  The ARB_gl_spirv spec defines VertexId to be the
       * same as gl_VertexID, which is non-zero-based, and removes
       * VertexIndex.  Since they're both defined to be non-zero-based, we use
       * SYSTEM_VALUE_VERTEX_ID for both.
       */
      *location = SYSTEM_VALUE_VERTEX_ID;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInInstanceIndex:
      *location = SYSTEM_VALUE_INSTANCE_INDEX;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInInstanceId:
      *location = SYSTEM_VALUE_INSTANCE_ID;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInPrimitiveId:
      if (b->shader->info.stage == MESA_SHADER_FRAGMENT) {
         vtn_assert(*mode == nir_var_shader_in);
         *location = VARYING_SLOT_PRIMITIVE_ID;
      } else if (*mode == nir_var_shader_out) {
         *location = VARYING_SLOT_PRIMITIVE_ID;
      } else {
         *location = SYSTEM_VALUE_PRIMITIVE_ID;
         set_mode_system_value(b, mode);
      }
      break;
   case SpvBuiltInInvocationId:
      *location = SYSTEM_VALUE_INVOCATION_ID;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInLayer:
   case SpvBuiltInLayerPerViewNV:
      *location = VARYING_SLOT_LAYER;
      if (b->shader->info.stage == MESA_SHADER_FRAGMENT)
         *mode = nir_var_shader_in;
      else if (b->shader->info.stage == MESA_SHADER_GEOMETRY)
         *mode = nir_var_shader_out;
      else if (b->options && b->options->caps.shader_viewport_index_layer &&
               (b->shader->info.stage == MESA_SHADER_VERTEX ||
                b->shader->info.stage == MESA_SHADER_TESS_EVAL ||
                b->shader->info.stage == MESA_SHADER_MESH))
         *mode = nir_var_shader_out;
      else
         vtn_fail("invalid stage for SpvBuiltInLayer");
      break;
   case SpvBuiltInViewportIndex:
      *location = VARYING_SLOT_VIEWPORT;
      if (b->shader->info.stage == MESA_SHADER_GEOMETRY)
         *mode = nir_var_shader_out;
      else if (b->options && b->options->caps.shader_viewport_index_layer &&
               (b->shader->info.stage == MESA_SHADER_VERTEX ||
                b->shader->info.stage == MESA_SHADER_TESS_EVAL ||
                b->shader->info.stage == MESA_SHADER_MESH))
         *mode = nir_var_shader_out;
      else if (b->shader->info.stage == MESA_SHADER_FRAGMENT)
         *mode = nir_var_shader_in;
      else
         vtn_fail("invalid stage for SpvBuiltInViewportIndex");
      break;
   case SpvBuiltInViewportMaskNV:
   case SpvBuiltInViewportMaskPerViewNV:
      *location = VARYING_SLOT_VIEWPORT_MASK;
      *mode = nir_var_shader_out;
      break;
   case SpvBuiltInTessLevelOuter:
      *location = VARYING_SLOT_TESS_LEVEL_OUTER;
      break;
   case SpvBuiltInTessLevelInner:
      *location = VARYING_SLOT_TESS_LEVEL_INNER;
      break;
   case SpvBuiltInTessCoord:
      *location = SYSTEM_VALUE_TESS_COORD;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInPatchVertices:
      *location = SYSTEM_VALUE_VERTICES_IN;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInFragCoord:
      vtn_assert(*mode == nir_var_shader_in);
      *mode = nir_var_system_value;
      *location = SYSTEM_VALUE_FRAG_COORD;
      break;
   case SpvBuiltInPointCoord:
      vtn_assert(*mode == nir_var_shader_in);
      set_mode_system_value(b, mode);
      *location = SYSTEM_VALUE_POINT_COORD;
      break;
   case SpvBuiltInFrontFacing:
      *location = SYSTEM_VALUE_FRONT_FACE;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInSampleId:
      *location = SYSTEM_VALUE_SAMPLE_ID;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInSamplePosition:
      *location = SYSTEM_VALUE_SAMPLE_POS;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInSampleMask:
      if (*mode == nir_var_shader_out) {
         *location = FRAG_RESULT_SAMPLE_MASK;
      } else {
         *location = SYSTEM_VALUE_SAMPLE_MASK_IN;
         set_mode_system_value(b, mode);
      }
      break;
   case SpvBuiltInFragDepth:
      *location = FRAG_RESULT_DEPTH;
      vtn_assert(*mode == nir_var_shader_out);
      break;
   case SpvBuiltInHelperInvocation:
      *location = SYSTEM_VALUE_HELPER_INVOCATION;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInNumWorkgroups:
      *location = SYSTEM_VALUE_NUM_WORKGROUPS;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInWorkgroupSize:
   case SpvBuiltInEnqueuedWorkgroupSize:
      *location = SYSTEM_VALUE_WORKGROUP_SIZE;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInWorkgroupId:
      *location = SYSTEM_VALUE_WORKGROUP_ID;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInLocalInvocationId:
      *location = SYSTEM_VALUE_LOCAL_INVOCATION_ID;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInLocalInvocationIndex:
      *location = SYSTEM_VALUE_LOCAL_INVOCATION_INDEX;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInGlobalInvocationId:
      *location = SYSTEM_VALUE_GLOBAL_INVOCATION_ID;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInGlobalLinearId:
      *location = SYSTEM_VALUE_GLOBAL_INVOCATION_INDEX;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInGlobalOffset:
      *location = SYSTEM_VALUE_BASE_GLOBAL_INVOCATION_ID;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInBaseVertex:
      /* OpenGL gl_BaseVertex (SYSTEM_VALUE_BASE_VERTEX) is not the same
       * semantic as Vulkan BaseVertex (SYSTEM_VALUE_FIRST_VERTEX).
       */
      if (b->options->environment == NIR_SPIRV_OPENGL)
         *location = SYSTEM_VALUE_BASE_VERTEX;
      else
         *location = SYSTEM_VALUE_FIRST_VERTEX;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInBaseInstance:
      *location = SYSTEM_VALUE_BASE_INSTANCE;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInDrawIndex:
      *location = SYSTEM_VALUE_DRAW_ID;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInSubgroupSize:
      *location = SYSTEM_VALUE_SUBGROUP_SIZE;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInSubgroupId:
      *location = SYSTEM_VALUE_SUBGROUP_ID;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInSubgroupLocalInvocationId:
      *location = SYSTEM_VALUE_SUBGROUP_INVOCATION;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInNumSubgroups:
      *location = SYSTEM_VALUE_NUM_SUBGROUPS;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInDeviceIndex:
      *location = SYSTEM_VALUE_DEVICE_INDEX;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInViewIndex:
      if (b->options && b->options->view_index_is_input) {
         *location = VARYING_SLOT_VIEW_INDEX;
         vtn_assert(*mode == nir_var_shader_in);
      } else {
         *location = SYSTEM_VALUE_VIEW_INDEX;
         set_mode_system_value(b, mode);
      }
      break;
   case SpvBuiltInSubgroupEqMask:
      *location = SYSTEM_VALUE_SUBGROUP_EQ_MASK,
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInSubgroupGeMask:
      *location = SYSTEM_VALUE_SUBGROUP_GE_MASK,
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInSubgroupGtMask:
      *location = SYSTEM_VALUE_SUBGROUP_GT_MASK,
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInSubgroupLeMask:
      *location = SYSTEM_VALUE_SUBGROUP_LE_MASK,
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInSubgroupLtMask:
      *location = SYSTEM_VALUE_SUBGROUP_LT_MASK,
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInFragStencilRefEXT:
      *location = FRAG_RESULT_STENCIL;
      vtn_assert(*mode == nir_var_shader_out);
      break;
   case SpvBuiltInWorkDim:
      *location = SYSTEM_VALUE_WORK_DIM;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInGlobalSize:
      *location = SYSTEM_VALUE_GLOBAL_GROUP_SIZE;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInBaryCoordNoPerspAMD:
      *location = SYSTEM_VALUE_BARYCENTRIC_LINEAR_PIXEL;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInBaryCoordNoPerspCentroidAMD:
      *location = SYSTEM_VALUE_BARYCENTRIC_LINEAR_CENTROID;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInBaryCoordNoPerspSampleAMD:
      *location = SYSTEM_VALUE_BARYCENTRIC_LINEAR_SAMPLE;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInBaryCoordSmoothAMD:
      *location = SYSTEM_VALUE_BARYCENTRIC_PERSP_PIXEL;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInBaryCoordSmoothCentroidAMD:
      *location = SYSTEM_VALUE_BARYCENTRIC_PERSP_CENTROID;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInBaryCoordSmoothSampleAMD:
      *location = SYSTEM_VALUE_BARYCENTRIC_PERSP_SAMPLE;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInBaryCoordPullModelAMD:
      *location = SYSTEM_VALUE_BARYCENTRIC_PULL_MODEL;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInLaunchIdKHR:
      *location = SYSTEM_VALUE_RAY_LAUNCH_ID;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInLaunchSizeKHR:
      *location = SYSTEM_VALUE_RAY_LAUNCH_SIZE;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInWorldRayOriginKHR:
      *location = SYSTEM_VALUE_RAY_WORLD_ORIGIN;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInWorldRayDirectionKHR:
      *location = SYSTEM_VALUE_RAY_WORLD_DIRECTION;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInObjectRayOriginKHR:
      *location = SYSTEM_VALUE_RAY_OBJECT_ORIGIN;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInObjectRayDirectionKHR:
      *location = SYSTEM_VALUE_RAY_OBJECT_DIRECTION;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInObjectToWorldKHR:
      *location = SYSTEM_VALUE_RAY_OBJECT_TO_WORLD;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInWorldToObjectKHR:
      *location = SYSTEM_VALUE_RAY_WORLD_TO_OBJECT;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInRayTminKHR:
      *location = SYSTEM_VALUE_RAY_T_MIN;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInRayTmaxKHR:
   case SpvBuiltInHitTNV:
      *location = SYSTEM_VALUE_RAY_T_MAX;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInInstanceCustomIndexKHR:
      *location = SYSTEM_VALUE_RAY_INSTANCE_CUSTOM_INDEX;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInHitKindKHR:
      *location = SYSTEM_VALUE_RAY_HIT_KIND;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInIncomingRayFlagsKHR:
      *location = SYSTEM_VALUE_RAY_FLAGS;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInRayGeometryIndexKHR:
      *location = SYSTEM_VALUE_RAY_GEOMETRY_INDEX;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInCullMaskKHR:
      *location = SYSTEM_VALUE_CULL_MASK;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInShadingRateKHR:
      *location = SYSTEM_VALUE_FRAG_SHADING_RATE;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInPrimitiveShadingRateKHR:
      if (b->shader->info.stage == MESA_SHADER_VERTEX ||
          b->shader->info.stage == MESA_SHADER_GEOMETRY ||
          b->shader->info.stage == MESA_SHADER_MESH) {
         *location = VARYING_SLOT_PRIMITIVE_SHADING_RATE;
         *mode = nir_var_shader_out;
      } else {
         vtn_fail("invalid stage for SpvBuiltInPrimitiveShadingRateKHR");
      }
      break;
   case SpvBuiltInPrimitiveCountNV:
      *location = VARYING_SLOT_PRIMITIVE_COUNT;
      break;
   case SpvBuiltInPrimitivePointIndicesEXT:
   case SpvBuiltInPrimitiveLineIndicesEXT:
   case SpvBuiltInPrimitiveTriangleIndicesEXT:
   case SpvBuiltInPrimitiveIndicesNV:
      *location = VARYING_SLOT_PRIMITIVE_INDICES;
      break;
   case SpvBuiltInTaskCountNV:
      /* NV_mesh_shader only. */
      *location = VARYING_SLOT_TASK_COUNT;
      *mode = nir_var_shader_out;
      break;
   case SpvBuiltInMeshViewCountNV:
      *location = SYSTEM_VALUE_MESH_VIEW_COUNT;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInMeshViewIndicesNV:
      *location = SYSTEM_VALUE_MESH_VIEW_INDICES;
      set_mode_system_value(b, mode);
      break;
   case SpvBuiltInCullPrimitiveEXT:
      *location = VARYING_SLOT_CULL_PRIMITIVE;
      break;
   default:
      vtn_fail("Unsupported builtin: %s (%u)",
               spirv_builtin_to_string(builtin), builtin);
   }
}

static void
apply_var_decoration(struct vtn_builder *b,
                     struct nir_variable_data *var_data,
                     const struct vtn_decoration *dec)
{
   switch (dec->decoration) {
   case SpvDecorationRelaxedPrecision:
      var_data->precision = GLSL_PRECISION_MEDIUM;
      break;
   case SpvDecorationNoPerspective:
      var_data->interpolation = INTERP_MODE_NOPERSPECTIVE;
      break;
   case SpvDecorationFlat:
      var_data->interpolation = INTERP_MODE_FLAT;
      break;
   case SpvDecorationExplicitInterpAMD:
      var_data->interpolation = INTERP_MODE_EXPLICIT;
      break;
   case SpvDecorationCentroid:
      var_data->centroid = true;
      break;
   case SpvDecorationSample:
      var_data->sample = true;
      break;
   case SpvDecorationInvariant:
      var_data->invariant = true;
      break;
   case SpvDecorationConstant:
      var_data->read_only = true;
      break;
   case SpvDecorationNonReadable:
      var_data->access |= ACCESS_NON_READABLE;
      break;
   case SpvDecorationNonWritable:
      var_data->read_only = true;
      var_data->access |= ACCESS_NON_WRITEABLE;
      break;
   case SpvDecorationRestrict:
      var_data->access |= ACCESS_RESTRICT;
      break;
   case SpvDecorationAliased:
      var_data->access &= ~ACCESS_RESTRICT;
      break;
   case SpvDecorationVolatile:
      var_data->access |= ACCESS_VOLATILE;
      break;
   case SpvDecorationCoherent:
      var_data->access |= ACCESS_COHERENT;
      break;
   case SpvDecorationComponent:
      var_data->location_frac = dec->operands[0];
      break;
   case SpvDecorationIndex:
      var_data->index = dec->operands[0];
      break;
   case SpvDecorationBuiltIn: {
      SpvBuiltIn builtin = dec->operands[0];

      nir_variable_mode mode = var_data->mode;
      vtn_get_builtin_location(b, builtin, &var_data->location, &mode);
      var_data->mode = mode;

      switch (builtin) {
      case SpvBuiltInTessLevelOuter:
      case SpvBuiltInTessLevelInner:
      case SpvBuiltInClipDistance:
      case SpvBuiltInClipDistancePerViewNV:
      case SpvBuiltInCullDistance:
      case SpvBuiltInCullDistancePerViewNV:
         var_data->compact = true;
         break;
      case SpvBuiltInPrimitivePointIndicesEXT:
      case SpvBuiltInPrimitiveLineIndicesEXT:
      case SpvBuiltInPrimitiveTriangleIndicesEXT:
         /* Not defined as per-primitive in the EXT, but they behave
          * like per-primitive outputs so it's easier to treat them like that.
          * They may still require special treatment in the backend in order to
          * control where and how they are stored.
          *
          * EXT_mesh_shader: write-only array of vectors indexed by the primitive index
          * NV_mesh_shader: read/write flat array
          */
         var_data->per_primitive = true;
         break;
      default:
         break;
      }

      break;
   }

   case SpvDecorationSpecId:
   case SpvDecorationRowMajor:
   case SpvDecorationColMajor:
   case SpvDecorationMatrixStride:
   case SpvDecorationUniform:
   case SpvDecorationUniformId:
   case SpvDecorationLinkageAttributes:
      break; /* Do nothing with these here */

   case SpvDecorationPatch:
      var_data->patch = true;
      break;

   case SpvDecorationLocation:
      vtn_fail("Should be handled earlier by var_decoration_cb()");

   case SpvDecorationBlock:
   case SpvDecorationBufferBlock:
   case SpvDecorationArrayStride:
   case SpvDecorationGLSLShared:
   case SpvDecorationGLSLPacked:
      break; /* These can apply to a type but we don't care about them */

   case SpvDecorationBinding:
   case SpvDecorationDescriptorSet:
   case SpvDecorationNoContraction:
   case SpvDecorationInputAttachmentIndex:
      vtn_warn("Decoration not allowed for variable or structure member: %s",
               spirv_decoration_to_string(dec->decoration));
      break;

   case SpvDecorationXfbBuffer:
      var_data->explicit_xfb_buffer = true;
      var_data->xfb.buffer = dec->operands[0];
      var_data->always_active_io = true;
      break;
   case SpvDecorationXfbStride:
      var_data->explicit_xfb_stride = true;
      var_data->xfb.stride = dec->operands[0];
      break;
   case SpvDecorationOffset:
      var_data->explicit_offset = true;
      var_data->offset = dec->operands[0];
      break;

   case SpvDecorationStream:
      var_data->stream = dec->operands[0];
      break;

   case SpvDecorationCPacked:
   case SpvDecorationSaturatedConversion:
   case SpvDecorationFuncParamAttr:
   case SpvDecorationFPRoundingMode:
   case SpvDecorationFPFastMathMode:
   case SpvDecorationAlignment:
      if (b->shader->info.stage != MESA_SHADER_KERNEL) {
         vtn_warn("Decoration only allowed for CL-style kernels: %s",
                  spirv_decoration_to_string(dec->decoration));
      }
      break;

   case SpvDecorationUserSemantic:
   case SpvDecorationUserTypeGOOGLE:
      /* User semantic decorations can safely be ignored by the driver. */
      break;

   case SpvDecorationRestrictPointerEXT:
   case SpvDecorationAliasedPointerEXT:
      /* TODO: We should actually plumb alias information through NIR. */
      break;

   case SpvDecorationPerPrimitiveNV:
      vtn_fail_if(
         !(b->shader->info.stage == MESA_SHADER_MESH && var_data->mode == nir_var_shader_out) &&
         !(b->shader->info.stage == MESA_SHADER_FRAGMENT && var_data->mode == nir_var_shader_in),
         "PerPrimitiveNV decoration only allowed for Mesh shader outputs or Fragment shader inputs");
      var_data->per_primitive = true;
      break;

   case SpvDecorationPerTaskNV:
      vtn_fail_if(
         (b->shader->info.stage != MESA_SHADER_MESH &&
          b->shader->info.stage != MESA_SHADER_TASK) ||
         var_data->mode != nir_var_mem_task_payload,
         "PerTaskNV decoration only allowed on Task/Mesh payload variables.");
      break;

   case SpvDecorationPerViewNV:
      vtn_fail_if(b->shader->info.stage != MESA_SHADER_MESH,
                  "PerViewNV decoration only allowed in Mesh shaders");
      var_data->per_view = true;
      break;

   default:
      vtn_fail_with_decoration("Unhandled decoration", dec->decoration);
   }
}

static void
gather_var_kind_cb(struct vtn_builder *b, struct vtn_value *val, int member,
                   const struct vtn_decoration *dec, void *void_var)
{
   struct vtn_variable *vtn_var = void_var;
   switch (dec->decoration) {
   case SpvDecorationPatch:
      vtn_var->var->data.patch = true;
      break;
   case SpvDecorationPerPrimitiveNV:
      vtn_var->var->data.per_primitive = true;
      break;
   case SpvDecorationPerViewNV:
      vtn_var->var->data.per_view = true;
      break;
   default:
      /* Nothing to do. */
      break;
   }
}

static void
var_decoration_cb(struct vtn_builder *b, struct vtn_value *val, int member,
                  const struct vtn_decoration *dec, void *void_var)
{
   struct vtn_variable *vtn_var = void_var;

   /* Handle decorations that apply to a vtn_variable as a whole */
   switch (dec->decoration) {
   case SpvDecorationDescriptorSet:
   case SpvDecorationBinding:
      if (dec->decoration == SpvDecorationDescriptorSet) {
         vtn_var->orig_descriptor_set = dec->operands[0];
      } else {
         vtn_var->orig_binding = dec->operands[0];
      }
      vtn_var->descriptor_set = 0;
	   vtn_var->binding = vtn_var->orig_descriptor_set * GODOT_NIR_DESCRIPTOR_SET_MULTIPLIER + vtn_var->orig_binding * GODOT_NIR_BINDING_MULTIPLIER;
      vtn_var->explicit_binding = true;
      return;
   case SpvDecorationInputAttachmentIndex:
      vtn_var->input_attachment_index = dec->operands[0];
      return;
   case SpvDecorationPatch:
      vtn_var->var->data.patch = true;
      break;
   case SpvDecorationOffset:
      vtn_var->offset = dec->operands[0];
      break;
   case SpvDecorationNonWritable:
      vtn_var->access |= ACCESS_NON_WRITEABLE;
      break;
   case SpvDecorationNonReadable:
      vtn_var->access |= ACCESS_NON_READABLE;
      break;
   case SpvDecorationVolatile:
      vtn_var->access |= ACCESS_VOLATILE;
      break;
   case SpvDecorationCoherent:
      vtn_var->access |= ACCESS_COHERENT;
      break;
   case SpvDecorationCounterBuffer:
      /* Counter buffer decorations can safely be ignored by the driver. */
      return;
   default:
      break;
   }

   if (val->value_type == vtn_value_type_pointer) {
      assert(val->pointer->var == void_var);
      assert(member == -1);
   } else {
      assert(val->value_type == vtn_value_type_type);
   }

   /* Location is odd.  If applied to a split structure, we have to walk the
    * whole thing and accumulate the location.  It's easier to handle as a
    * special case.
    */
   if (dec->decoration == SpvDecorationLocation) {
      unsigned location = dec->operands[0];
      if (b->shader->info.stage == MESA_SHADER_FRAGMENT &&
          vtn_var->mode == vtn_variable_mode_output) {
         location += FRAG_RESULT_DATA0;
      } else if (b->shader->info.stage == MESA_SHADER_VERTEX &&
                 vtn_var->mode == vtn_variable_mode_input) {
         location += VERT_ATTRIB_GENERIC0;
      } else if (vtn_var->mode == vtn_variable_mode_input ||
                 vtn_var->mode == vtn_variable_mode_output) {
         location += vtn_var->var->data.patch ? VARYING_SLOT_PATCH0 : VARYING_SLOT_VAR0;
      } else if (vtn_var->mode == vtn_variable_mode_call_data ||
                 vtn_var->mode == vtn_variable_mode_ray_payload) {
         /* This location is fine as-is */
      } else if (vtn_var->mode != vtn_variable_mode_uniform &&
                 vtn_var->mode != vtn_variable_mode_image) {
         vtn_warn("Location must be on input, output, uniform, sampler or "
                  "image variable");
         return;
      }

      if (vtn_var->var->num_members == 0) {
         /* This handles the member and lone variable cases */
         vtn_var->var->data.location = location;
      } else {
         /* This handles the structure member case */
         assert(vtn_var->var->members);

         if (member == -1)
            vtn_var->base_location = location;
         else
            vtn_var->var->members[member].location = location;
      }

      return;
   } else {
      if (vtn_var->var) {
         if (vtn_var->var->num_members == 0) {
            /* We call this function on types as well as variables and not all
             * struct types get split so we can end up having stray member
             * decorations; just ignore them.
             */
            if (member == -1)
               apply_var_decoration(b, &vtn_var->var->data, dec);
         } else if (member >= 0) {
            /* Member decorations must come from a type */
            assert(val->value_type == vtn_value_type_type);
            apply_var_decoration(b, &vtn_var->var->members[member], dec);
         } else {
            unsigned length =
               glsl_get_length(glsl_without_array(vtn_var->type->type));
            for (unsigned i = 0; i < length; i++)
               apply_var_decoration(b, &vtn_var->var->members[i], dec);
         }
      } else {
         /* A few variables, those with external storage, have no actual
          * nir_variables associated with them.  Fortunately, all decorations
          * we care about for those variables are on the type only.
          */
         vtn_assert(vtn_var->mode == vtn_variable_mode_ubo ||
                    vtn_var->mode == vtn_variable_mode_ssbo ||
                    vtn_var->mode == vtn_variable_mode_push_constant);
      }
   }
}

enum vtn_variable_mode
vtn_storage_class_to_mode(struct vtn_builder *b,
                          SpvStorageClass class,
                          struct vtn_type *interface_type,
                          nir_variable_mode *nir_mode_out)
{
   enum vtn_variable_mode mode;
   nir_variable_mode nir_mode;
   switch (class) {
   case SpvStorageClassUniform:
      /* Assume it's an UBO if we lack the interface_type. */
      if (!interface_type || interface_type->block) {
         mode = vtn_variable_mode_ubo;
         nir_mode = nir_var_mem_ubo;
      } else if (interface_type->buffer_block) {
         mode = vtn_variable_mode_ssbo;
         nir_mode = nir_var_mem_ssbo;
      } else {
         /* Default-block uniforms, coming from gl_spirv */
         mode = vtn_variable_mode_uniform;
         nir_mode = nir_var_uniform;
      }
      break;
   case SpvStorageClassStorageBuffer:
      mode = vtn_variable_mode_ssbo;
      nir_mode = nir_var_mem_ssbo;
      break;
   case SpvStorageClassPhysicalStorageBuffer:
      mode = vtn_variable_mode_phys_ssbo;
      nir_mode = nir_var_mem_global;
      break;
   case SpvStorageClassUniformConstant:
      /* interface_type is only NULL when OpTypeForwardPointer is used and
       * OpTypeForwardPointer can only be used for struct types, not images or
       * acceleration structures.
       */
      if (interface_type)
         interface_type = vtn_type_without_array(interface_type);

      if (interface_type &&
          interface_type->base_type == vtn_base_type_image &&
          glsl_type_is_image(interface_type->glsl_image)) {
         mode = vtn_variable_mode_image;
         nir_mode = nir_var_image;
      } else if (b->shader->info.stage == MESA_SHADER_KERNEL) {
         mode = vtn_variable_mode_constant;
         nir_mode = nir_var_mem_constant;
      } else {
         /* interface_type is only NULL when OpTypeForwardPointer is used and
          * OpTypeForwardPointer cannot be used with the UniformConstant
          * storage class.
          */
         assert(interface_type != NULL);
         if (interface_type->base_type == vtn_base_type_accel_struct) {
            mode = vtn_variable_mode_accel_struct;
            nir_mode = nir_var_uniform;
         } else {
            mode = vtn_variable_mode_uniform;
            nir_mode = nir_var_uniform;
         }
      }
      break;
   case SpvStorageClassPushConstant:
      mode = vtn_variable_mode_push_constant;
      nir_mode = nir_var_mem_push_const;
      break;
   case SpvStorageClassInput:
      mode = vtn_variable_mode_input;
      nir_mode = nir_var_shader_in;

      /* NV_mesh_shader: fixup due to lack of dedicated storage class */
      if (b->shader->info.stage == MESA_SHADER_MESH) {
         mode = vtn_variable_mode_task_payload;
         nir_mode = nir_var_mem_task_payload;
      }
      break;
   case SpvStorageClassOutput:
      mode = vtn_variable_mode_output;
      nir_mode = nir_var_shader_out;

      /* NV_mesh_shader: fixup due to lack of dedicated storage class */
      if (b->shader->info.stage == MESA_SHADER_TASK) {
         mode = vtn_variable_mode_task_payload;
         nir_mode = nir_var_mem_task_payload;
      }
      break;
   case SpvStorageClassPrivate:
      mode = vtn_variable_mode_private;
      nir_mode = nir_var_shader_temp;
      break;
   case SpvStorageClassFunction:
      mode = vtn_variable_mode_function;
      nir_mode = nir_var_function_temp;
      break;
   case SpvStorageClassWorkgroup:
      mode = vtn_variable_mode_workgroup;
      nir_mode = nir_var_mem_shared;
      break;
   case SpvStorageClassTaskPayloadWorkgroupEXT:
      mode = vtn_variable_mode_task_payload;
      nir_mode = nir_var_mem_task_payload;
      break;
   case SpvStorageClassAtomicCounter:
      mode = vtn_variable_mode_atomic_counter;
      nir_mode = nir_var_uniform;
      break;
   case SpvStorageClassCrossWorkgroup:
      mode = vtn_variable_mode_cross_workgroup;
      nir_mode = nir_var_mem_global;
      break;
   case SpvStorageClassImage:
      mode = vtn_variable_mode_image;
      nir_mode = nir_var_image;
      break;
   case SpvStorageClassCallableDataKHR:
      mode = vtn_variable_mode_call_data;
      nir_mode = nir_var_shader_temp;
      break;
   case SpvStorageClassIncomingCallableDataKHR:
      mode = vtn_variable_mode_call_data_in;
      nir_mode = nir_var_shader_call_data;
      break;
   case SpvStorageClassRayPayloadKHR:
      mode = vtn_variable_mode_ray_payload;
      nir_mode = nir_var_shader_temp;
      break;
   case SpvStorageClassIncomingRayPayloadKHR:
      mode = vtn_variable_mode_ray_payload_in;
      nir_mode = nir_var_shader_call_data;
      break;
   case SpvStorageClassHitAttributeKHR:
      mode = vtn_variable_mode_hit_attrib;
      nir_mode = nir_var_ray_hit_attrib;
      break;
   case SpvStorageClassShaderRecordBufferKHR:
      mode = vtn_variable_mode_shader_record;
      nir_mode = nir_var_mem_constant;
      break;

   case SpvStorageClassGeneric:
      mode = vtn_variable_mode_generic;
      nir_mode = nir_var_mem_generic;
      break;
   default:
      vtn_fail("Unhandled variable storage class: %s (%u)",
               spirv_storageclass_to_string(class), class);
   }

   if (nir_mode_out)
      *nir_mode_out = nir_mode;

   return mode;
}

nir_address_format
vtn_mode_to_address_format(struct vtn_builder *b, enum vtn_variable_mode mode)
{
   switch (mode) {
   case vtn_variable_mode_ubo:
      return b->options->ubo_addr_format;

   case vtn_variable_mode_ssbo:
      return b->options->ssbo_addr_format;

   case vtn_variable_mode_phys_ssbo:
      return b->options->phys_ssbo_addr_format;

   case vtn_variable_mode_push_constant:
      return b->options->push_const_addr_format;

   case vtn_variable_mode_workgroup:
      return b->options->shared_addr_format;

   case vtn_variable_mode_generic:
   case vtn_variable_mode_cross_workgroup:
      return b->options->global_addr_format;

   case vtn_variable_mode_shader_record:
   case vtn_variable_mode_constant:
      return b->options->constant_addr_format;

   case vtn_variable_mode_accel_struct:
      return nir_address_format_64bit_global;

   case vtn_variable_mode_task_payload:
      return b->options->task_payload_addr_format;

   case vtn_variable_mode_function:
      if (b->physical_ptrs)
         return b->options->temp_addr_format;
      FALLTHROUGH;

   case vtn_variable_mode_private:
   case vtn_variable_mode_uniform:
   case vtn_variable_mode_atomic_counter:
   case vtn_variable_mode_input:
   case vtn_variable_mode_output:
   case vtn_variable_mode_image:
   case vtn_variable_mode_call_data:
   case vtn_variable_mode_call_data_in:
   case vtn_variable_mode_ray_payload:
   case vtn_variable_mode_ray_payload_in:
   case vtn_variable_mode_hit_attrib:
      return nir_address_format_logical;
   }

   unreachable("Invalid variable mode");
}

nir_ssa_def *
vtn_pointer_to_ssa(struct vtn_builder *b, struct vtn_pointer *ptr)
{
   if ((vtn_pointer_is_external_block(b, ptr) &&
        vtn_type_contains_block(b, ptr->type) &&
        ptr->mode != vtn_variable_mode_phys_ssbo) ||
       ptr->mode == vtn_variable_mode_accel_struct) {
      /* In this case, we're looking for a block index and not an actual
       * deref.
       *
       * For PhysicalStorageBuffer pointers, we don't have a block index
       * at all because we get the pointer directly from the client.  This
       * assumes that there will never be a SSBO binding variable using the
       * PhysicalStorageBuffer storage class.  This assumption appears
       * to be correct according to the Vulkan spec because the table,
       * "Shader Resource and Storage Class Correspondence," the only the
       * Uniform storage class with BufferBlock or the StorageBuffer
       * storage class with Block can be used.
       */
      if (!ptr->block_index) {
         /* If we don't have a block_index then we must be a pointer to the
          * variable itself.
          */
         vtn_assert(!ptr->deref);

         struct vtn_access_chain chain = {
            .length = 0,
         };
         ptr = vtn_pointer_dereference(b, ptr, &chain);
      }

      return ptr->block_index;
   } else {
      return &vtn_pointer_to_deref(b, ptr)->dest.ssa;
   }
}

struct vtn_pointer *
vtn_pointer_from_ssa(struct vtn_builder *b, nir_ssa_def *ssa,
                     struct vtn_type *ptr_type)
{
   vtn_assert(ptr_type->base_type == vtn_base_type_pointer);

   struct vtn_pointer *ptr = rzalloc(b, struct vtn_pointer);
   struct vtn_type *without_array =
      vtn_type_without_array(ptr_type->deref);

   nir_variable_mode nir_mode;
   ptr->mode = vtn_storage_class_to_mode(b, ptr_type->storage_class,
                                         without_array, &nir_mode);
   ptr->type = ptr_type->deref;
   ptr->ptr_type = ptr_type;

   const struct glsl_type *deref_type =
      vtn_type_get_nir_type(b, ptr_type->deref, ptr->mode);
   if (!vtn_pointer_is_external_block(b, ptr) &&
       ptr->mode != vtn_variable_mode_accel_struct) {
      ptr->deref = nir_build_deref_cast(&b->nb, ssa, nir_mode,
                                        deref_type, ptr_type->stride);
   } else if ((vtn_type_contains_block(b, ptr->type) &&
               ptr->mode != vtn_variable_mode_phys_ssbo) ||
              ptr->mode == vtn_variable_mode_accel_struct) {
      /* This is a pointer to somewhere in an array of blocks, not a
       * pointer to somewhere inside the block.  Set the block index
       * instead of making a cast.
       */
      ptr->block_index = ssa;
   } else {
      /* This is a pointer to something internal or a pointer inside a
       * block.  It's just a regular cast.
       *
       * For PhysicalStorageBuffer pointers, we don't have a block index
       * at all because we get the pointer directly from the client.  This
       * assumes that there will never be a SSBO binding variable using the
       * PhysicalStorageBuffer storage class.  This assumption appears
       * to be correct according to the Vulkan spec because the table,
       * "Shader Resource and Storage Class Correspondence," the only the
       * Uniform storage class with BufferBlock or the StorageBuffer
       * storage class with Block can be used.
       */
      ptr->deref = nir_build_deref_cast(&b->nb, ssa, nir_mode,
                                        deref_type, ptr_type->stride);
      ptr->deref->dest.ssa.num_components =
         glsl_get_vector_elements(ptr_type->type);
      ptr->deref->dest.ssa.bit_size = glsl_get_bit_size(ptr_type->type);
   }

   return ptr;
}

static void
assign_missing_member_locations(struct vtn_variable *var)
{
   unsigned length =
      glsl_get_length(glsl_without_array(var->type->type));
   int location = var->base_location;

   for (unsigned i = 0; i < length; i++) {
      /* From the Vulkan spec:
       *
       * “If the structure type is a Block but without a Location, then each
       *  of its members must have a Location decoration.”
       *
       */
      if (var->type->block) {
         assert(var->base_location != -1 ||
                var->var->members[i].location != -1);
      }

      /* From the Vulkan spec:
       *
       * “Any member with its own Location decoration is assigned that
       *  location. Each remaining member is assigned the location after the
       *  immediately preceding member in declaration order.”
       */
      if (var->var->members[i].location != -1)
         location = var->var->members[i].location;
      else
         var->var->members[i].location = location;

      /* Below we use type instead of interface_type, because interface_type
       * is only available when it is a Block. This code also supports
       * input/outputs that are just structs
       */
      const struct glsl_type *member_type =
         glsl_get_struct_field(glsl_without_array(var->type->type), i);

      location +=
         glsl_count_attribute_slots(member_type,
                                    false /* is_gl_vertex_input */);
   }
}

nir_deref_instr *
vtn_get_call_payload_for_location(struct vtn_builder *b, uint32_t location_id)
{
   uint32_t location = vtn_constant_uint(b, location_id);
   nir_foreach_variable_with_modes(var, b->nb.shader, nir_var_shader_temp) {
      if (var->data.explicit_location &&
          var->data.location == location)
         return nir_build_deref_var(&b->nb, var);
   }
   vtn_fail("Couldn't find variable with a storage class of CallableDataKHR "
            "or RayPayloadKHR and location %d", location);
}

static bool
vtn_type_is_ray_query(struct vtn_type *type)
{
   return vtn_type_without_array(type)->base_type == vtn_base_type_ray_query;
}

static void
vtn_create_variable(struct vtn_builder *b, struct vtn_value *val,
                    struct vtn_type *ptr_type, SpvStorageClass storage_class,
                    struct vtn_value *initializer)
{
   vtn_assert(ptr_type->base_type == vtn_base_type_pointer);
   struct vtn_type *type = ptr_type->deref;

   struct vtn_type *without_array = vtn_type_without_array(ptr_type->deref);

   enum vtn_variable_mode mode;
   nir_variable_mode nir_mode;
   mode = vtn_storage_class_to_mode(b, storage_class, without_array, &nir_mode);

   switch (mode) {
   case vtn_variable_mode_ubo:
      /* There's no other way to get vtn_variable_mode_ubo */
      vtn_assert(without_array->block);
      break;
   case vtn_variable_mode_ssbo:
      if (storage_class == SpvStorageClassStorageBuffer &&
          !without_array->block) {
         if (b->variable_pointers) {
            vtn_fail("Variables in the StorageBuffer storage class must "
                     "have a struct type with the Block decoration");
         } else {
            /* If variable pointers are not present, it's still malformed
             * SPIR-V but we can parse it and do the right thing anyway.
             * Since some of the 8-bit storage tests have bugs in this are,
             * just make it a warning for now.
             */
            vtn_warn("Variables in the StorageBuffer storage class must "
                     "have a struct type with the Block decoration");
         }
      }
      break;

   case vtn_variable_mode_generic:
      vtn_fail("Cannot create a variable with the Generic storage class");
      break;

   case vtn_variable_mode_image:
      if (storage_class == SpvStorageClassImage)
         vtn_fail("Cannot create a variable with the Image storage class");
      else
         vtn_assert(storage_class == SpvStorageClassUniformConstant);
      break;

   case vtn_variable_mode_phys_ssbo:
      vtn_fail("Cannot create a variable with the "
               "PhysicalStorageBuffer storage class");
      break;

   default:
      /* No tallying is needed */
      break;
   }

   struct vtn_variable *var = rzalloc(b, struct vtn_variable);
   var->type = type;
   var->mode = mode;
   var->base_location = -1;

   val->pointer = rzalloc(b, struct vtn_pointer);
   val->pointer->mode = var->mode;
   val->pointer->type = var->type;
   val->pointer->ptr_type = ptr_type;
   val->pointer->var = var;
   val->pointer->access = var->type->access;

   switch (var->mode) {
   case vtn_variable_mode_function:
   case vtn_variable_mode_private:
   case vtn_variable_mode_uniform:
   case vtn_variable_mode_atomic_counter:
   case vtn_variable_mode_constant:
   case vtn_variable_mode_call_data:
   case vtn_variable_mode_call_data_in:
   case vtn_variable_mode_image:
   case vtn_variable_mode_ray_payload:
   case vtn_variable_mode_ray_payload_in:
   case vtn_variable_mode_hit_attrib:
      /* For these, we create the variable normally */
      var->var = rzalloc(b->shader, nir_variable);
      var->var->name = ralloc_strdup(var->var, val->name);
      var->var->type = vtn_type_get_nir_type(b, var->type, var->mode);

      /* This is a total hack but we need some way to flag variables which are
       * going to be call payloads.  See get_call_payload_deref.
       */
      if (storage_class == SpvStorageClassCallableDataKHR ||
          storage_class == SpvStorageClassRayPayloadKHR)
         var->var->data.explicit_location = true;

      var->var->data.mode = nir_mode;
      var->var->data.location = -1;
      var->var->data.ray_query = vtn_type_is_ray_query(var->type);
      var->var->interface_type = NULL;
      break;

   case vtn_variable_mode_ubo:
   case vtn_variable_mode_ssbo:
   case vtn_variable_mode_push_constant:
   case vtn_variable_mode_accel_struct:
   case vtn_variable_mode_shader_record:
      var->var = rzalloc(b->shader, nir_variable);
      var->var->name = ralloc_strdup(var->var, val->name);

      var->var->type = vtn_type_get_nir_type(b, var->type, var->mode);
      var->var->interface_type = var->var->type;

      var->var->data.mode = nir_mode;
      var->var->data.location = -1;
      var->var->data.driver_location = 0;
      var->var->data.access = var->type->access;
      break;

   case vtn_variable_mode_workgroup:
   case vtn_variable_mode_cross_workgroup:
   case vtn_variable_mode_task_payload:
      /* Create the variable normally */
      var->var = rzalloc(b->shader, nir_variable);
      var->var->name = ralloc_strdup(var->var, val->name);
      var->var->type = vtn_type_get_nir_type(b, var->type, var->mode);
      var->var->data.mode = nir_mode;
      break;

   case vtn_variable_mode_input:
   case vtn_variable_mode_output: {
      var->var = rzalloc(b->shader, nir_variable);
      var->var->name = ralloc_strdup(var->var, val->name);
      var->var->type = vtn_type_get_nir_type(b, var->type, var->mode);
      var->var->data.mode = nir_mode;

      /* In order to know whether or not we're a per-vertex inout, we need
       * the patch qualifier.  This means walking the variable decorations
       * early before we actually create any variables.  Not a big deal.
       *
       * GLSLang really likes to place decorations in the most interior
       * thing it possibly can.  In particular, if you have a struct, it
       * will place the patch decorations on the struct members.  This
       * should be handled by the variable splitting below just fine.
       *
       * If you have an array-of-struct, things get even more weird as it
       * will place the patch decorations on the struct even though it's
       * inside an array and some of the members being patch and others not
       * makes no sense whatsoever.  Since the only sensible thing is for
       * it to be all or nothing, we'll call it patch if any of the members
       * are declared patch.
       */
      vtn_foreach_decoration(b, val, gather_var_kind_cb, var);
      if (glsl_type_is_array(var->type->type) &&
          glsl_type_is_struct_or_ifc(without_array->type)) {
         vtn_foreach_decoration(b, vtn_value(b, without_array->id,
                                             vtn_value_type_type),
                                gather_var_kind_cb, var);
      }

      struct vtn_type *per_vertex_type = var->type;
      if (nir_is_arrayed_io(var->var, b->shader->info.stage))
         per_vertex_type = var->type->array_element;

      /* Figure out the interface block type. */
      struct vtn_type *iface_type = per_vertex_type;
      if (var->mode == vtn_variable_mode_output &&
          (b->shader->info.stage == MESA_SHADER_VERTEX ||
           b->shader->info.stage == MESA_SHADER_TESS_EVAL ||
           b->shader->info.stage == MESA_SHADER_GEOMETRY)) {
         /* For vertex data outputs, we can end up with arrays of blocks for
          * transform feedback where each array element corresponds to a
          * different XFB output buffer.
          */
         while (iface_type->base_type == vtn_base_type_array)
            iface_type = iface_type->array_element;
      }
      if (iface_type->base_type == vtn_base_type_struct && iface_type->block)
         var->var->interface_type = vtn_type_get_nir_type(b, iface_type,
                                                          var->mode);

      /* If it's a block, set it up as per-member so can be splitted later by
       * nir_split_per_member_structs.
       *
       * This is for a couple of reasons.  For one, builtins may all come in a
       * block and we really want those split out into separate variables.
       * For another, interpolation qualifiers can be applied to members of
       * the top-level struct and we need to be able to preserve that
       * information.
       */
      if (per_vertex_type->base_type == vtn_base_type_struct &&
          per_vertex_type->block) {
         var->var->num_members = glsl_get_length(per_vertex_type->type);
         var->var->members = rzalloc_array(var->var, struct nir_variable_data,
                                           var->var->num_members);

         for (unsigned i = 0; i < var->var->num_members; i++) {
            var->var->members[i].mode = nir_mode;
            var->var->members[i].patch = var->var->data.patch;
            var->var->members[i].location = -1;
         }
      }

      /* For inputs and outputs, we need to grab locations and builtin
       * information from the per-vertex type.
       */
      vtn_foreach_decoration(b, vtn_value(b, per_vertex_type->id,
                                          vtn_value_type_type),
                             var_decoration_cb, var);

      break;
   }

   case vtn_variable_mode_phys_ssbo:
   case vtn_variable_mode_generic:
      unreachable("Should have been caught before");
   }

   /* Ignore incorrectly generated Undef initializers. */
   if (b->wa_llvm_spirv_ignore_workgroup_initializer &&
       initializer &&
       storage_class == SpvStorageClassWorkgroup)
      initializer = NULL;

   /* Only initialize variable when there is an initializer and it's not
    * undef.
    */
   if (initializer && !initializer->is_undef_constant) {
      switch (storage_class) {
      case SpvStorageClassWorkgroup:
         /* VK_KHR_zero_initialize_workgroup_memory. */
         vtn_fail_if(b->options->environment != NIR_SPIRV_VULKAN,
                     "Only Vulkan supports variable initializer "
                     "for Workgroup variable %u",
                     vtn_id_for_value(b, val));
         vtn_fail_if(initializer->value_type != vtn_value_type_constant ||
                     !initializer->is_null_constant,
                     "Workgroup variable %u can only have OpConstantNull "
                     "as initializer, but have %u instead",
                     vtn_id_for_value(b, val),
                     vtn_id_for_value(b, initializer));
         b->shader->info.zero_initialize_shared_memory = true;
         break;

      case SpvStorageClassUniformConstant:
         vtn_fail_if(b->options->environment != NIR_SPIRV_OPENGL &&
                     b->options->environment != NIR_SPIRV_OPENCL,
                     "Only OpenGL and OpenCL support variable initializer "
                     "for UniformConstant variable %u\n",
                     vtn_id_for_value(b, val));
         vtn_fail_if(initializer->value_type != vtn_value_type_constant,
                     "UniformConstant variable %u can only have a constant "
                     "initializer, but have %u instead",
                     vtn_id_for_value(b, val),
                     vtn_id_for_value(b, initializer));
         break;

      case SpvStorageClassOutput:
      case SpvStorageClassPrivate:
         vtn_assert(b->options->environment != NIR_SPIRV_OPENCL);
         /* These can have any initializer. */
         break;

      case SpvStorageClassFunction:
         /* These can have any initializer. */
         break;

      case SpvStorageClassCrossWorkgroup:
         vtn_assert(b->options->environment == NIR_SPIRV_OPENCL);
         vtn_fail("Initializer for CrossWorkgroup variable %u "
                  "not yet supported in Mesa.",
                  vtn_id_for_value(b, val));
         break;

      default: {
         const enum nir_spirv_execution_environment env =
            b->options->environment;
         const char *env_name =
            env == NIR_SPIRV_VULKAN ? "Vulkan" :
            env == NIR_SPIRV_OPENCL ? "OpenCL" :
            env == NIR_SPIRV_OPENGL ? "OpenGL" :
            NULL;
         vtn_assert(env_name);
         vtn_fail("In %s, any OpVariable with an Initializer operand "
                  "must have %s%s%s, or Function as "
                  "its Storage Class operand.  Variable %u has an "
                  "Initializer but its Storage Class is %s.",
                  env_name,
                  env == NIR_SPIRV_VULKAN ? "Private, Output, Workgroup" : "",
                  env == NIR_SPIRV_OPENCL ? "CrossWorkgroup, UniformConstant" : "",
                  env == NIR_SPIRV_OPENGL ? "Private, Output, UniformConstant" : "",
                  vtn_id_for_value(b, val),
                  spirv_storageclass_to_string(storage_class));
         }
      }

      switch (initializer->value_type) {
      case vtn_value_type_constant:
         var->var->constant_initializer =
            nir_constant_clone(initializer->constant, var->var);
         break;
      case vtn_value_type_pointer:
         var->var->pointer_initializer = initializer->pointer->var->var;
         break;
      default:
         vtn_fail("SPIR-V variable initializer %u must be constant or pointer",
                  vtn_id_for_value(b, initializer));
      }
   }

   if (var->mode == vtn_variable_mode_uniform ||
       var->mode == vtn_variable_mode_image ||
       var->mode == vtn_variable_mode_ssbo) {
      /* SSBOs and images are assumed to not alias in the Simple, GLSL and Vulkan memory models */
      var->var->data.access |= b->mem_model != SpvMemoryModelOpenCL ? ACCESS_RESTRICT : 0;
   }

   vtn_foreach_decoration(b, val, var_decoration_cb, var);
   vtn_foreach_decoration(b, val, ptr_decoration_cb, val->pointer);

   /* Propagate access flags from the OpVariable decorations. */
   val->pointer->access |= var->access;

   if ((var->mode == vtn_variable_mode_input ||
        var->mode == vtn_variable_mode_output) &&
       var->var->members) {
      assign_missing_member_locations(var);
   }

   if (var->mode == vtn_variable_mode_uniform ||
       var->mode == vtn_variable_mode_image ||
       var->mode == vtn_variable_mode_ubo ||
       var->mode == vtn_variable_mode_ssbo ||
       var->mode == vtn_variable_mode_atomic_counter) {
      /* XXX: We still need the binding information in the nir_variable
       * for these. We should fix that.
       */
      var->var->data.binding = var->binding;
      var->var->data.explicit_binding = var->explicit_binding;
      var->var->data.descriptor_set = var->descriptor_set;
      var->var->data.index = var->input_attachment_index;
      var->var->data.offset = var->offset;

      if (glsl_type_is_image(glsl_without_array(var->var->type)))
         var->var->data.image.format = without_array->image_format;
   }

   if (var->mode == vtn_variable_mode_function) {
      vtn_assert(var->var != NULL && var->var->members == NULL);
      nir_function_impl_add_variable(b->nb.impl, var->var);
   } else if (var->var) {
      nir_shader_add_variable(b->shader, var->var);
   } else {
      vtn_assert(vtn_pointer_is_external_block(b, val->pointer) ||
                 var->mode == vtn_variable_mode_accel_struct ||
                 var->mode == vtn_variable_mode_shader_record);
   }
}

static void
vtn_assert_types_equal(struct vtn_builder *b, SpvOp opcode,
                       struct vtn_type *dst_type,
                       struct vtn_type *src_type)
{
   if (dst_type->id == src_type->id)
      return;

   if (vtn_types_compatible(b, dst_type, src_type)) {
      /* Early versions of GLSLang would re-emit types unnecessarily and you
       * would end up with OpLoad, OpStore, or OpCopyMemory opcodes which have
       * mismatched source and destination types.
       *
       * https://github.com/KhronosGroup/glslang/issues/304
       * https://github.com/KhronosGroup/glslang/issues/307
       * https://bugs.freedesktop.org/show_bug.cgi?id=104338
       * https://bugs.freedesktop.org/show_bug.cgi?id=104424
       */
      vtn_warn("Source and destination types of %s do not have the same "
               "ID (but are compatible): %u vs %u",
                spirv_op_to_string(opcode), dst_type->id, src_type->id);
      return;
   }

   vtn_fail("Source and destination types of %s do not match: %s vs. %s",
            spirv_op_to_string(opcode),
            glsl_get_type_name(dst_type->type),
            glsl_get_type_name(src_type->type));
}

static nir_ssa_def *
nir_shrink_zero_pad_vec(nir_builder *b, nir_ssa_def *val,
                        unsigned num_components)
{
   if (val->num_components == num_components)
      return val;

   nir_ssa_def *comps[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < num_components; i++) {
      if (i < val->num_components)
         comps[i] = nir_channel(b, val, i);
      else
         comps[i] = nir_imm_intN_t(b, 0, val->bit_size);
   }
   return nir_vec(b, comps, num_components);
}

static nir_ssa_def *
nir_sloppy_bitcast(nir_builder *b, nir_ssa_def *val,
                   const struct glsl_type *type)
{
   const unsigned num_components = glsl_get_vector_elements(type);
   const unsigned bit_size = glsl_get_bit_size(type);

   /* First, zero-pad to ensure that the value is big enough that when we
    * bit-cast it, we don't loose anything.
    */
   if (val->bit_size < bit_size) {
      const unsigned src_num_components_needed =
         vtn_align_u32(val->num_components, bit_size / val->bit_size);
      val = nir_shrink_zero_pad_vec(b, val, src_num_components_needed);
   }

   val = nir_bitcast_vector(b, val, bit_size);

   return nir_shrink_zero_pad_vec(b, val, num_components);
}

static bool
vtn_get_mem_operands(struct vtn_builder *b, const uint32_t *w, unsigned count,
                     unsigned *idx, SpvMemoryAccessMask *access, unsigned *alignment,
                     SpvScope *dest_scope, SpvScope *src_scope)
{
   *access = 0;
   *alignment = 0;
   if (*idx >= count)
      return false;

   *access = w[(*idx)++];
   if (*access & SpvMemoryAccessAlignedMask) {
      vtn_assert(*idx < count);
      *alignment = w[(*idx)++];
   }

   if (*access & SpvMemoryAccessMakePointerAvailableMask) {
      vtn_assert(*idx < count);
      vtn_assert(dest_scope);
      *dest_scope = vtn_constant_uint(b, w[(*idx)++]);
   }

   if (*access & SpvMemoryAccessMakePointerVisibleMask) {
      vtn_assert(*idx < count);
      vtn_assert(src_scope);
      *src_scope = vtn_constant_uint(b, w[(*idx)++]);
   }

   return true;
}

static enum gl_access_qualifier
spv_access_to_gl_access(SpvMemoryAccessMask access)
{
   unsigned result = 0;

   if (access & SpvMemoryAccessVolatileMask)
      result |= ACCESS_VOLATILE;
   if (access & SpvMemoryAccessNontemporalMask)
      result |= ACCESS_STREAM_CACHE_POLICY;

   return result;
}


SpvMemorySemanticsMask
vtn_mode_to_memory_semantics(enum vtn_variable_mode mode)
{
   switch (mode) {
   case vtn_variable_mode_ssbo:
   case vtn_variable_mode_phys_ssbo:
      return SpvMemorySemanticsUniformMemoryMask;
   case vtn_variable_mode_workgroup:
      return SpvMemorySemanticsWorkgroupMemoryMask;
   case vtn_variable_mode_cross_workgroup:
      return SpvMemorySemanticsCrossWorkgroupMemoryMask;
   case vtn_variable_mode_atomic_counter:
      return SpvMemorySemanticsAtomicCounterMemoryMask;
   case vtn_variable_mode_image:
      return SpvMemorySemanticsImageMemoryMask;
   case vtn_variable_mode_output:
      return SpvMemorySemanticsOutputMemoryMask;
   default:
      return SpvMemorySemanticsMaskNone;
   }
}

static void
vtn_emit_make_visible_barrier(struct vtn_builder *b, SpvMemoryAccessMask access,
                              SpvScope scope, enum vtn_variable_mode mode)
{
   if (!(access & SpvMemoryAccessMakePointerVisibleMask))
      return;

   vtn_emit_memory_barrier(b, scope, SpvMemorySemanticsMakeVisibleMask |
                                     SpvMemorySemanticsAcquireMask |
                                     vtn_mode_to_memory_semantics(mode));
}

static void
vtn_emit_make_available_barrier(struct vtn_builder *b, SpvMemoryAccessMask access,
                                SpvScope scope, enum vtn_variable_mode mode)
{
   if (!(access & SpvMemoryAccessMakePointerAvailableMask))
      return;

   vtn_emit_memory_barrier(b, scope, SpvMemorySemanticsMakeAvailableMask |
                                     SpvMemorySemanticsReleaseMask |
                                     vtn_mode_to_memory_semantics(mode));
}

static void
ptr_nonuniform_workaround_cb(struct vtn_builder *b, struct vtn_value *val,
                  int member, const struct vtn_decoration *dec, void *void_ptr)
{
   enum gl_access_qualifier *access = void_ptr;

   switch (dec->decoration) {
   case SpvDecorationNonUniformEXT:
      *access |= ACCESS_NON_UNIFORM;
      break;

   default:
      break;
   }
}

void
vtn_handle_variables(struct vtn_builder *b, SpvOp opcode,
                     const uint32_t *w, unsigned count)
{
   switch (opcode) {
   case SpvOpUndef: {
      struct vtn_value *val = vtn_push_value(b, w[2], vtn_value_type_undef);
      val->type = vtn_get_type(b, w[1]);
      val->is_undef_constant = true;
      break;
   }

   case SpvOpVariable: {
      struct vtn_type *ptr_type = vtn_get_type(b, w[1]);

      SpvStorageClass storage_class = w[3];

      const bool is_global = storage_class != SpvStorageClassFunction;
      const bool is_io = storage_class == SpvStorageClassInput ||
                         storage_class == SpvStorageClassOutput;

      /* Skip global variables that are not used by the entrypoint.  Before
       * SPIR-V 1.4 the interface is only used for I/O variables, so extra
       * variables will still need to be removed later.
       */
      if (!b->options->create_library &&
          (is_io || (b->version >= 0x10400 && is_global))) {
         if (!bsearch(&w[2], b->interface_ids, b->interface_ids_count, 4, cmp_uint32_t))
            break;
      }

      struct vtn_value *val = vtn_push_value(b, w[2], vtn_value_type_pointer);
      struct vtn_value *initializer = count > 4 ? vtn_untyped_value(b, w[4]) : NULL;

      vtn_create_variable(b, val, ptr_type, storage_class, initializer);

      break;
   }

   case SpvOpConstantSampler: {
      /* Synthesize a pointer-to-sampler type, create a variable of that type,
       * and give the variable a constant initializer with the sampler params */
      struct vtn_type *sampler_type = vtn_value(b, w[1], vtn_value_type_type)->type;
      struct vtn_value *val = vtn_push_value(b, w[2], vtn_value_type_pointer);

      struct vtn_type *ptr_type = rzalloc(b, struct vtn_type);
      ptr_type->base_type = vtn_base_type_pointer;
      ptr_type->deref = sampler_type;
      ptr_type->storage_class = SpvStorageClassUniform;

      ptr_type->type = nir_address_format_to_glsl_type(
         vtn_mode_to_address_format(b, vtn_variable_mode_function));

      vtn_create_variable(b, val, ptr_type, ptr_type->storage_class, NULL);

      nir_variable *nir_var = val->pointer->var->var;
      nir_var->data.sampler.is_inline_sampler = true;
      nir_var->data.sampler.addressing_mode = w[3];
      nir_var->data.sampler.normalized_coordinates = w[4];
      nir_var->data.sampler.filter_mode = w[5];

      break;
   }

   case SpvOpAccessChain:
   case SpvOpPtrAccessChain:
   case SpvOpInBoundsAccessChain:
   case SpvOpInBoundsPtrAccessChain: {
      struct vtn_access_chain *chain = vtn_access_chain_create(b, count - 4);
      enum gl_access_qualifier access = 0;
      chain->ptr_as_array = (opcode == SpvOpPtrAccessChain || opcode == SpvOpInBoundsPtrAccessChain);

      unsigned idx = 0;
      for (int i = 4; i < count; i++) {
         struct vtn_value *link_val = vtn_untyped_value(b, w[i]);
         if (link_val->value_type == vtn_value_type_constant) {
            chain->link[idx].mode = vtn_access_mode_literal;
            chain->link[idx].id = vtn_constant_int(b, w[i]);
         } else {
            chain->link[idx].mode = vtn_access_mode_id;
            chain->link[idx].id = w[i];
         }

         /* Workaround for https://gitlab.freedesktop.org/mesa/mesa/-/issues/3406 */
         vtn_foreach_decoration(b, link_val, ptr_nonuniform_workaround_cb, &access);

         idx++;
      }

      struct vtn_type *ptr_type = vtn_get_type(b, w[1]);

      struct vtn_pointer *base = vtn_pointer(b, w[3]);

      chain->in_bounds = (opcode == SpvOpInBoundsAccessChain || opcode == SpvOpInBoundsPtrAccessChain);

      /* Workaround for https://gitlab.freedesktop.org/mesa/mesa/-/issues/3406 */
      access |= base->access & ACCESS_NON_UNIFORM;

      struct vtn_pointer *ptr = vtn_pointer_dereference(b, base, chain);
      ptr->ptr_type = ptr_type;
      ptr->access |= access;
      vtn_push_pointer(b, w[2], ptr);
      break;
   }

   case SpvOpCopyMemory: {
      struct vtn_value *dest_val = vtn_pointer_value(b, w[1]);
      struct vtn_value *src_val = vtn_pointer_value(b, w[2]);
      struct vtn_pointer *dest = vtn_value_to_pointer(b, dest_val);
      struct vtn_pointer *src = vtn_value_to_pointer(b, src_val);

      vtn_assert_types_equal(b, opcode, dest_val->type->deref,
                                        src_val->type->deref);

      unsigned idx = 3, dest_alignment, src_alignment;
      SpvMemoryAccessMask dest_access, src_access;
      SpvScope dest_scope, src_scope;
      vtn_get_mem_operands(b, w, count, &idx, &dest_access, &dest_alignment,
                           &dest_scope, &src_scope);
      if (!vtn_get_mem_operands(b, w, count, &idx, &src_access, &src_alignment,
                                NULL, &src_scope)) {
         src_alignment = dest_alignment;
         src_access = dest_access;
      }
      src = vtn_align_pointer(b, src, src_alignment);
      dest = vtn_align_pointer(b, dest, dest_alignment);

      vtn_emit_make_visible_barrier(b, src_access, src_scope, src->mode);

      vtn_variable_copy(b, dest, src,
                        spv_access_to_gl_access(dest_access),
                        spv_access_to_gl_access(src_access));

      vtn_emit_make_available_barrier(b, dest_access, dest_scope, dest->mode);
      break;
   }

   case SpvOpCopyMemorySized: {
      struct vtn_value *dest_val = vtn_pointer_value(b, w[1]);
      struct vtn_value *src_val = vtn_pointer_value(b, w[2]);
      nir_ssa_def *size = vtn_get_nir_ssa(b, w[3]);
      struct vtn_pointer *dest = vtn_value_to_pointer(b, dest_val);
      struct vtn_pointer *src = vtn_value_to_pointer(b, src_val);

      unsigned idx = 4, dest_alignment, src_alignment;
      SpvMemoryAccessMask dest_access, src_access;
      SpvScope dest_scope, src_scope;
      vtn_get_mem_operands(b, w, count, &idx, &dest_access, &dest_alignment,
                           &dest_scope, &src_scope);
      if (!vtn_get_mem_operands(b, w, count, &idx, &src_access, &src_alignment,
                                NULL, &src_scope)) {
         src_alignment = dest_alignment;
         src_access = dest_access;
      }
      src = vtn_align_pointer(b, src, src_alignment);
      dest = vtn_align_pointer(b, dest, dest_alignment);

      vtn_emit_make_visible_barrier(b, src_access, src_scope, src->mode);

      nir_memcpy_deref_with_access(&b->nb,
                                   vtn_pointer_to_deref(b, dest),
                                   vtn_pointer_to_deref(b, src),
                                   size,
                                   spv_access_to_gl_access(dest_access),
                                   spv_access_to_gl_access(src_access));

      vtn_emit_make_available_barrier(b, dest_access, dest_scope, dest->mode);
      break;
   }

   case SpvOpLoad: {
      struct vtn_type *res_type = vtn_get_type(b, w[1]);
      struct vtn_value *src_val = vtn_value(b, w[3], vtn_value_type_pointer);
      struct vtn_pointer *src = vtn_value_to_pointer(b, src_val);

      vtn_assert_types_equal(b, opcode, res_type, src_val->type->deref);

      unsigned idx = 4, alignment;
      SpvMemoryAccessMask access;
      SpvScope scope;
      vtn_get_mem_operands(b, w, count, &idx, &access, &alignment, NULL, &scope);
      src = vtn_align_pointer(b, src, alignment);

      vtn_emit_make_visible_barrier(b, access, scope, src->mode);

      vtn_push_ssa_value(b, w[2], vtn_variable_load(b, src, spv_access_to_gl_access(access)));
      break;
   }

   case SpvOpStore: {
      struct vtn_value *dest_val = vtn_pointer_value(b, w[1]);
      struct vtn_pointer *dest = vtn_value_to_pointer(b, dest_val);
      struct vtn_value *src_val = vtn_untyped_value(b, w[2]);

      /* OpStore requires us to actually have a storage type */
      vtn_fail_if(dest->type->type == NULL,
                  "Invalid destination type for OpStore");

      if (glsl_get_base_type(dest->type->type) == GLSL_TYPE_BOOL &&
          glsl_get_base_type(src_val->type->type) == GLSL_TYPE_UINT) {
         /* Early versions of GLSLang would use uint types for UBOs/SSBOs but
          * would then store them to a local variable as bool.  Work around
          * the issue by doing an implicit conversion.
          *
          * https://github.com/KhronosGroup/glslang/issues/170
          * https://bugs.freedesktop.org/show_bug.cgi?id=104424
          */
         vtn_warn("OpStore of value of type OpTypeInt to a pointer to type "
                  "OpTypeBool.  Doing an implicit conversion to work around "
                  "the problem.");
         struct vtn_ssa_value *bool_ssa =
            vtn_create_ssa_value(b, dest->type->type);
         bool_ssa->def = nir_i2b(&b->nb, vtn_ssa_value(b, w[2])->def);
         vtn_variable_store(b, bool_ssa, dest, 0);
         break;
      }

      vtn_assert_types_equal(b, opcode, dest_val->type->deref, src_val->type);

      unsigned idx = 3, alignment;
      SpvMemoryAccessMask access;
      SpvScope scope;
      vtn_get_mem_operands(b, w, count, &idx, &access, &alignment, &scope, NULL);
      dest = vtn_align_pointer(b, dest, alignment);

      struct vtn_ssa_value *src = vtn_ssa_value(b, w[2]);
      vtn_variable_store(b, src, dest, spv_access_to_gl_access(access));

      vtn_emit_make_available_barrier(b, access, scope, dest->mode);
      break;
   }

   case SpvOpArrayLength: {
      struct vtn_pointer *ptr = vtn_pointer(b, w[3]);
      const uint32_t field = w[4];

      vtn_fail_if(ptr->type->base_type != vtn_base_type_struct,
                  "OpArrayLength must take a pointer to a structure type");
      vtn_fail_if(field != ptr->type->length - 1 ||
                  ptr->type->members[field]->base_type != vtn_base_type_array,
                  "OpArrayLength must reference the last memeber of the "
                  "structure and that must be an array");

      if (b->options->use_deref_buffer_array_length) {
         struct vtn_access_chain chain = {
            .length = 1,
            .link = {
               { .mode = vtn_access_mode_literal, .id = field },
            }
         };
         struct vtn_pointer *array = vtn_pointer_dereference(b, ptr, &chain);

         nir_ssa_def *array_length =
            nir_build_deref_buffer_array_length(&b->nb, 32,
                                                vtn_pointer_to_ssa(b, array),
                                                .access=ptr->access | ptr->type->access);

         vtn_push_nir_ssa(b, w[2], array_length);
      } else {
         const uint32_t offset = ptr->type->offsets[field];
         const uint32_t stride = ptr->type->members[field]->stride;

         if (!ptr->block_index) {
            struct vtn_access_chain chain = {
               .length = 0,
            };
            ptr = vtn_pointer_dereference(b, ptr, &chain);
            vtn_assert(ptr->block_index);
         }

         nir_ssa_def *buf_size = nir_get_ssbo_size(&b->nb, ptr->block_index,
                                                   .access=ptr->access | ptr->type->access);

         /* array_length = max(buffer_size - offset, 0) / stride */
         nir_ssa_def *array_length =
            nir_udiv_imm(&b->nb,
                         nir_usub_sat(&b->nb,
                                      buf_size,
                                      nir_imm_int(&b->nb, offset)),
                         stride);

         vtn_push_nir_ssa(b, w[2], array_length);
      }
      break;
   }

   case SpvOpConvertPtrToU: {
      struct vtn_type *u_type = vtn_get_type(b, w[1]);
      struct vtn_type *ptr_type = vtn_get_value_type(b, w[3]);

      vtn_fail_if(ptr_type->base_type != vtn_base_type_pointer ||
                  ptr_type->type == NULL,
                  "OpConvertPtrToU can only be used on physical pointers");

      vtn_fail_if(u_type->base_type != vtn_base_type_vector &&
                  u_type->base_type != vtn_base_type_scalar,
                  "OpConvertPtrToU can only be used to cast to a vector or "
                  "scalar type");

      /* The pointer will be converted to an SSA value automatically */
      nir_ssa_def *ptr = vtn_get_nir_ssa(b, w[3]);
      nir_ssa_def *u = nir_sloppy_bitcast(&b->nb, ptr, u_type->type);
      vtn_push_nir_ssa(b, w[2], u);
      break;
   }

   case SpvOpConvertUToPtr: {
      struct vtn_type *ptr_type = vtn_get_type(b, w[1]);
      struct vtn_type *u_type = vtn_get_value_type(b, w[3]);

      vtn_fail_if(ptr_type->base_type != vtn_base_type_pointer ||
                  ptr_type->type == NULL,
                  "OpConvertUToPtr can only be used on physical pointers");

      vtn_fail_if(u_type->base_type != vtn_base_type_vector &&
                  u_type->base_type != vtn_base_type_scalar,
                  "OpConvertUToPtr can only be used to cast from a vector or "
                  "scalar type");

      nir_ssa_def *u = vtn_get_nir_ssa(b, w[3]);
      nir_ssa_def *ptr = nir_sloppy_bitcast(&b->nb, u, ptr_type->type);
      vtn_push_pointer(b, w[2], vtn_pointer_from_ssa(b, ptr, ptr_type));
      break;
   }

   case SpvOpGenericCastToPtrExplicit: {
      struct vtn_type *dst_type = vtn_get_type(b, w[1]);
      struct vtn_type *src_type = vtn_get_value_type(b, w[3]);
      SpvStorageClass storage_class = w[4];

      vtn_fail_if(dst_type->base_type != vtn_base_type_pointer ||
                  dst_type->storage_class != storage_class,
                  "Result type of an SpvOpGenericCastToPtrExplicit must be "
                  "an OpTypePointer. Its Storage Class must match the "
                  "storage class specified in the instruction");

      vtn_fail_if(src_type->base_type != vtn_base_type_pointer ||
                  src_type->deref->id != dst_type->deref->id,
                  "Source pointer of an SpvOpGenericCastToPtrExplicit must "
                  "have a type of OpTypePointer whose Type is the same as "
                  "the Type of Result Type");

      vtn_fail_if(src_type->storage_class != SpvStorageClassGeneric,
                  "Source pointer of an SpvOpGenericCastToPtrExplicit must "
                  "point to the Generic Storage Class.");

      vtn_fail_if(storage_class != SpvStorageClassWorkgroup &&
                  storage_class != SpvStorageClassCrossWorkgroup &&
                  storage_class != SpvStorageClassFunction,
                  "Storage must be one of the following literal values from "
                  "Storage Class: Workgroup, CrossWorkgroup, or Function.");

      nir_deref_instr *src_deref = vtn_nir_deref(b, w[3]);

      nir_variable_mode nir_mode;
      enum vtn_variable_mode mode =
         vtn_storage_class_to_mode(b, storage_class, dst_type->deref, &nir_mode);
      nir_address_format addr_format = vtn_mode_to_address_format(b, mode);

      nir_ssa_def *null_value =
         nir_build_imm(&b->nb, nir_address_format_num_components(addr_format),
                               nir_address_format_bit_size(addr_format),
                               nir_address_format_null_value(addr_format));

      nir_ssa_def *valid = nir_build_deref_mode_is(&b->nb, 1, &src_deref->dest.ssa, nir_mode);
      vtn_push_nir_ssa(b, w[2], nir_bcsel(&b->nb, valid,
                                                  &src_deref->dest.ssa,
                                                  null_value));
      break;
   }

   case SpvOpGenericPtrMemSemantics: {
      struct vtn_type *dst_type = vtn_get_type(b, w[1]);
      struct vtn_type *src_type = vtn_get_value_type(b, w[3]);

      vtn_fail_if(dst_type->base_type != vtn_base_type_scalar ||
                  dst_type->type != glsl_uint_type(),
                  "Result type of an SpvOpGenericPtrMemSemantics must be "
                  "an OpTypeInt with 32-bit Width and 0 Signedness.");

      vtn_fail_if(src_type->base_type != vtn_base_type_pointer ||
                  src_type->storage_class != SpvStorageClassGeneric,
                  "Source pointer of an SpvOpGenericPtrMemSemantics must "
                  "point to the Generic Storage Class");

      nir_deref_instr *src_deref = vtn_nir_deref(b, w[3]);

      nir_ssa_def *global_bit =
         nir_bcsel(&b->nb, nir_build_deref_mode_is(&b->nb, 1, &src_deref->dest.ssa,
                                                   nir_var_mem_global),
                   nir_imm_int(&b->nb, SpvMemorySemanticsCrossWorkgroupMemoryMask),
                   nir_imm_int(&b->nb, 0));

      nir_ssa_def *shared_bit =
         nir_bcsel(&b->nb, nir_build_deref_mode_is(&b->nb, 1, &src_deref->dest.ssa,
                                                   nir_var_mem_shared),
                   nir_imm_int(&b->nb, SpvMemorySemanticsWorkgroupMemoryMask),
                   nir_imm_int(&b->nb, 0));

      vtn_push_nir_ssa(b, w[2], nir_iand(&b->nb, global_bit, shared_bit));
      break;
   }

   case SpvOpSubgroupBlockReadINTEL: {
      struct vtn_type *res_type = vtn_get_type(b, w[1]);
      nir_deref_instr *src = vtn_nir_deref(b, w[3]);

      nir_intrinsic_instr *load =
         nir_intrinsic_instr_create(b->nb.shader,
                                    nir_intrinsic_load_deref_block_intel);
      load->src[0] = nir_src_for_ssa(&src->dest.ssa);
      nir_ssa_dest_init_for_type(&load->instr, &load->dest,
                                 res_type->type, NULL);
      load->num_components = load->dest.ssa.num_components;
      nir_builder_instr_insert(&b->nb, &load->instr);

      vtn_push_nir_ssa(b, w[2], &load->dest.ssa);
      break;
   }

   case SpvOpSubgroupBlockWriteINTEL: {
      nir_deref_instr *dest = vtn_nir_deref(b, w[1]);
      nir_ssa_def *data = vtn_ssa_value(b, w[2])->def;

      nir_intrinsic_instr *store =
         nir_intrinsic_instr_create(b->nb.shader,
                                    nir_intrinsic_store_deref_block_intel);
      store->src[0] = nir_src_for_ssa(&dest->dest.ssa);
      store->src[1] = nir_src_for_ssa(data);
      store->num_components = data->num_components;
      nir_builder_instr_insert(&b->nb, &store->instr);
      break;
   }

   case SpvOpConvertUToAccelerationStructureKHR: {
      struct vtn_type *as_type = vtn_get_type(b, w[1]);
      struct vtn_type *u_type = vtn_get_value_type(b, w[3]);
      vtn_fail_if(!((u_type->base_type == vtn_base_type_vector &&
                     u_type->type == glsl_vector_type(GLSL_TYPE_UINT, 2)) ||
                    (u_type->base_type == vtn_base_type_scalar &&
                     u_type->type == glsl_uint64_t_type())),
                  "OpConvertUToAccelerationStructure may only be used to "
                  "cast from a 64-bit scalar integer or a 2-component vector "
                  "of 32-bit integers");
      vtn_fail_if(as_type->base_type != vtn_base_type_accel_struct,
                  "The result type of an OpConvertUToAccelerationStructure "
                  "must be OpTypeAccelerationStructure");

      nir_ssa_def *u = vtn_get_nir_ssa(b, w[3]);
      vtn_push_nir_ssa(b, w[2], nir_sloppy_bitcast(&b->nb, u, as_type->type));
      break;
   }

   default:
      vtn_fail_with_opcode("Unhandled opcode", opcode);
   }
}
