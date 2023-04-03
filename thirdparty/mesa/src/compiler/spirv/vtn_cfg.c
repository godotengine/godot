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
 */

#include "vtn_private.h"
#include "spirv_info.h"
#include "nir/nir_vla.h"
#include "util/u_debug.h"

static struct vtn_block *
vtn_block(struct vtn_builder *b, uint32_t value_id)
{
   return vtn_value(b, value_id, vtn_value_type_block)->block;
}

static unsigned
glsl_type_count_function_params(const struct glsl_type *type)
{
   if (glsl_type_is_vector_or_scalar(type)) {
      return 1;
   } else if (glsl_type_is_array_or_matrix(type)) {
      return glsl_get_length(type) *
             glsl_type_count_function_params(glsl_get_array_element(type));
   } else {
      assert(glsl_type_is_struct_or_ifc(type));
      unsigned count = 0;
      unsigned elems = glsl_get_length(type);
      for (unsigned i = 0; i < elems; i++) {
         const struct glsl_type *elem_type = glsl_get_struct_field(type, i);
         count += glsl_type_count_function_params(elem_type);
      }
      return count;
   }
}

static void
glsl_type_add_to_function_params(const struct glsl_type *type,
                                 nir_function *func,
                                 unsigned *param_idx)
{
   if (glsl_type_is_vector_or_scalar(type)) {
      func->params[(*param_idx)++] = (nir_parameter) {
         .num_components = glsl_get_vector_elements(type),
         .bit_size = glsl_get_bit_size(type),
      };
   } else if (glsl_type_is_array_or_matrix(type)) {
      unsigned elems = glsl_get_length(type);
      const struct glsl_type *elem_type = glsl_get_array_element(type);
      for (unsigned i = 0; i < elems; i++)
         glsl_type_add_to_function_params(elem_type,func, param_idx);
   } else {
      assert(glsl_type_is_struct_or_ifc(type));
      unsigned elems = glsl_get_length(type);
      for (unsigned i = 0; i < elems; i++) {
         const struct glsl_type *elem_type = glsl_get_struct_field(type, i);
         glsl_type_add_to_function_params(elem_type, func, param_idx);
      }
   }
}

static void
vtn_ssa_value_add_to_call_params(struct vtn_builder *b,
                                 struct vtn_ssa_value *value,
                                 nir_call_instr *call,
                                 unsigned *param_idx)
{
   if (glsl_type_is_vector_or_scalar(value->type)) {
      call->params[(*param_idx)++] = nir_src_for_ssa(value->def);
   } else {
      unsigned elems = glsl_get_length(value->type);
      for (unsigned i = 0; i < elems; i++) {
         vtn_ssa_value_add_to_call_params(b, value->elems[i],
                                          call, param_idx);
      }
   }
}

static void
vtn_ssa_value_load_function_param(struct vtn_builder *b,
                                  struct vtn_ssa_value *value,
                                  unsigned *param_idx)
{
   if (glsl_type_is_vector_or_scalar(value->type)) {
      value->def = nir_load_param(&b->nb, (*param_idx)++);
   } else {
      unsigned elems = glsl_get_length(value->type);
      for (unsigned i = 0; i < elems; i++)
         vtn_ssa_value_load_function_param(b, value->elems[i], param_idx);
   }
}

void
vtn_handle_function_call(struct vtn_builder *b, SpvOp opcode,
                         const uint32_t *w, unsigned count)
{
   struct vtn_function *vtn_callee =
      vtn_value(b, w[3], vtn_value_type_function)->func;

   vtn_callee->referenced = true;

   nir_call_instr *call = nir_call_instr_create(b->nb.shader,
                                                vtn_callee->nir_func);

   unsigned param_idx = 0;

   nir_deref_instr *ret_deref = NULL;
   struct vtn_type *ret_type = vtn_callee->type->return_type;
   if (ret_type->base_type != vtn_base_type_void) {
      nir_variable *ret_tmp =
         nir_local_variable_create(b->nb.impl,
                                   glsl_get_bare_type(ret_type->type),
                                   "return_tmp");
      ret_deref = nir_build_deref_var(&b->nb, ret_tmp);
      call->params[param_idx++] = nir_src_for_ssa(&ret_deref->dest.ssa);
   }

   for (unsigned i = 0; i < vtn_callee->type->length; i++) {
      vtn_ssa_value_add_to_call_params(b, vtn_ssa_value(b, w[4 + i]),
                                       call, &param_idx);
   }
   assert(param_idx == call->num_params);

   nir_builder_instr_insert(&b->nb, &call->instr);

   if (ret_type->base_type == vtn_base_type_void) {
      vtn_push_value(b, w[2], vtn_value_type_undef);
   } else {
      vtn_push_ssa_value(b, w[2], vtn_local_load(b, ret_deref, 0));
   }
}

static void
function_decoration_cb(struct vtn_builder *b, struct vtn_value *val, int member,
                       const struct vtn_decoration *dec, void *void_func)
{
   struct vtn_function *func = void_func;

   switch (dec->decoration) {
   case SpvDecorationLinkageAttributes: {
      unsigned name_words;
      const char *name =
         vtn_string_literal(b, dec->operands, dec->num_operands, &name_words);
      vtn_fail_if(name_words >= dec->num_operands,
                  "Malformed LinkageAttributes decoration");
      (void)name; /* TODO: What is this? */
      func->linkage = dec->operands[name_words];
      break;
   }

   default:
      break;
   }
}

static bool
vtn_cfg_handle_prepass_instruction(struct vtn_builder *b, SpvOp opcode,
                                   const uint32_t *w, unsigned count)
{
   switch (opcode) {
   case SpvOpFunction: {
      vtn_assert(b->func == NULL);
      b->func = rzalloc(b, struct vtn_function);

      b->func->node.type = vtn_cf_node_type_function;
      b->func->node.parent = NULL;
      list_inithead(&b->func->body);
      b->func->linkage = SpvLinkageTypeMax;
      b->func->control = w[3];

      UNUSED const struct glsl_type *result_type = vtn_get_type(b, w[1])->type;
      struct vtn_value *val = vtn_push_value(b, w[2], vtn_value_type_function);
      val->func = b->func;

      vtn_foreach_decoration(b, val, function_decoration_cb, b->func);

      b->func->type = vtn_get_type(b, w[4]);
      const struct vtn_type *func_type = b->func->type;

      vtn_assert(func_type->return_type->type == result_type);

      nir_function *func =
         nir_function_create(b->shader, ralloc_strdup(b->shader, val->name));

      unsigned num_params = 0;
      for (unsigned i = 0; i < func_type->length; i++)
         num_params += glsl_type_count_function_params(func_type->params[i]->type);

      /* Add one parameter for the function return value */
      if (func_type->return_type->base_type != vtn_base_type_void)
         num_params++;

      func->num_params = num_params;
      func->params = ralloc_array(b->shader, nir_parameter, num_params);

      unsigned idx = 0;
      if (func_type->return_type->base_type != vtn_base_type_void) {
         nir_address_format addr_format =
            vtn_mode_to_address_format(b, vtn_variable_mode_function);
         /* The return value is a regular pointer */
         func->params[idx++] = (nir_parameter) {
            .num_components = nir_address_format_num_components(addr_format),
            .bit_size = nir_address_format_bit_size(addr_format),
         };
      }

      for (unsigned i = 0; i < func_type->length; i++)
         glsl_type_add_to_function_params(func_type->params[i]->type, func, &idx);
      assert(idx == num_params);

      b->func->nir_func = func;

      /* Set up a nir_function_impl and the builder so we can load arguments
       * directly in our OpFunctionParameter handler.
       */
      nir_function_impl *impl = nir_function_impl_create(func);
      nir_builder_init(&b->nb, impl);
      b->nb.cursor = nir_before_cf_list(&impl->body);
      b->nb.exact = b->exact;

      b->func_param_idx = 0;

      /* The return value is the first parameter */
      if (func_type->return_type->base_type != vtn_base_type_void)
         b->func_param_idx++;
      break;
   }

   case SpvOpFunctionEnd:
      b->func->end = w;
      if (b->func->start_block == NULL) {
         vtn_fail_if(b->func->linkage != SpvLinkageTypeImport,
                     "A function declaration (an OpFunction with no basic "
                     "blocks), must have a Linkage Attributes Decoration "
                     "with the Import Linkage Type.");

         /* In this case, the function didn't have any actual blocks.  It's
          * just a prototype so delete the function_impl.
          */
         b->func->nir_func->impl = NULL;
      } else {
         vtn_fail_if(b->func->linkage == SpvLinkageTypeImport,
                     "A function definition (an OpFunction with basic blocks) "
                     "cannot be decorated with the Import Linkage Type.");
      }
      b->func = NULL;
      break;

   case SpvOpFunctionParameter: {
      vtn_assert(b->func_param_idx < b->func->nir_func->num_params);
      struct vtn_type *type = vtn_get_type(b, w[1]);
      struct vtn_ssa_value *value = vtn_create_ssa_value(b, type->type);
      vtn_ssa_value_load_function_param(b, value, &b->func_param_idx);
      vtn_push_ssa_value(b, w[2], value);
      break;
   }

   case SpvOpLabel: {
      vtn_assert(b->block == NULL);
      b->block = rzalloc(b, struct vtn_block);
      b->block->node.type = vtn_cf_node_type_block;
      b->block->label = w;
      vtn_push_value(b, w[1], vtn_value_type_block)->block = b->block;

      if (b->func->start_block == NULL) {
         /* This is the first block encountered for this function.  In this
          * case, we set the start block and add it to the list of
          * implemented functions that we'll walk later.
          */
         b->func->start_block = b->block;
         list_addtail(&b->func->node.link, &b->functions);
      }
      break;
   }

   case SpvOpSelectionMerge:
   case SpvOpLoopMerge:
      vtn_assert(b->block && b->block->merge == NULL);
      b->block->merge = w;
      break;

   case SpvOpBranch:
   case SpvOpBranchConditional:
   case SpvOpSwitch:
   case SpvOpKill:
   case SpvOpTerminateInvocation:
   case SpvOpIgnoreIntersectionKHR:
   case SpvOpTerminateRayKHR:
   case SpvOpEmitMeshTasksEXT:
   case SpvOpReturn:
   case SpvOpReturnValue:
   case SpvOpUnreachable:
      if (b->wa_ignore_return_after_emit_mesh_tasks &&
          opcode == SpvOpReturn && !b->block) {
            /* At this point block was already reset by
             * SpvOpEmitMeshTasksEXT. */
            break;
      }
      vtn_assert(b->block && b->block->branch == NULL);
      b->block->branch = w;
      b->block = NULL;
      break;

   default:
      /* Continue on as per normal */
      return true;
   }

   return true;
}

/* This function performs a depth-first search of the cases and puts them
 * in fall-through order.
 */
static void
vtn_order_case(struct vtn_switch *swtch, struct vtn_case *cse)
{
   if (cse->visited)
      return;

   cse->visited = true;

   list_del(&cse->node.link);

   if (cse->fallthrough) {
      vtn_order_case(swtch, cse->fallthrough);

      /* If we have a fall-through, place this case right before the case it
       * falls through to.  This ensures that fallthroughs come one after
       * the other.  These two can never get separated because that would
       * imply something else falling through to the same case.  Also, this
       * can't break ordering because the DFS ensures that this case is
       * visited before anything that falls through to it.
       */
      list_addtail(&cse->node.link, &cse->fallthrough->node.link);
   } else {
      list_add(&cse->node.link, &swtch->cases);
   }
}

static void
vtn_switch_order_cases(struct vtn_switch *swtch)
{
   struct list_head cases;
   list_replace(&swtch->cases, &cases);
   list_inithead(&swtch->cases);
   while (!list_is_empty(&cases)) {
      struct vtn_case *cse =
         list_first_entry(&cases, struct vtn_case, node.link);
      vtn_order_case(swtch, cse);
   }
}

static void
vtn_block_set_merge_cf_node(struct vtn_builder *b, struct vtn_block *block,
                            struct vtn_cf_node *cf_node)
{
   vtn_fail_if(block->merge_cf_node != NULL,
               "The merge block declared by a header block cannot be a "
               "merge block declared by any other header block.");

   block->merge_cf_node = cf_node;
}

#define VTN_DECL_CF_NODE_FIND(_type)                        \
static inline struct vtn_##_type *                          \
vtn_cf_node_find_##_type(struct vtn_cf_node *node)          \
{                                                           \
   while (node && node->type != vtn_cf_node_type_##_type)   \
      node = node->parent;                                  \
   return (struct vtn_##_type *)node;                       \
}

UNUSED VTN_DECL_CF_NODE_FIND(if)
VTN_DECL_CF_NODE_FIND(loop)
VTN_DECL_CF_NODE_FIND(case)
VTN_DECL_CF_NODE_FIND(switch)
VTN_DECL_CF_NODE_FIND(function)

static enum vtn_branch_type
vtn_handle_branch(struct vtn_builder *b,
                  struct vtn_cf_node *cf_parent,
                  struct vtn_block *target_block)
{
   struct vtn_loop *loop = vtn_cf_node_find_loop(cf_parent);

   /* Detect a loop back-edge first.  That way none of the code below
    * accidentally operates on a loop back-edge.
    */
   if (loop && target_block == loop->header_block)
      return vtn_branch_type_loop_back_edge;

   /* Try to detect fall-through */
   if (target_block->switch_case) {
      /* When it comes to handling switch cases, we can break calls to
       * vtn_handle_branch into two cases: calls from within a case construct
       * and calls for the jump to each case construct.  In the second case,
       * cf_parent is the vtn_switch itself and vtn_cf_node_find_case() will
       * return the outer switch case in which this switch is contained.  It's
       * fine if the target block is a switch case from an outer switch as
       * long as it is also the switch break for this switch.
       */
      struct vtn_case *switch_case = vtn_cf_node_find_case(cf_parent);

      /* This doesn't get called for the OpSwitch */
      vtn_fail_if(switch_case == NULL,
                  "A switch case can only be entered through an OpSwitch or "
                  "falling through from another switch case.");

      /* Because block->switch_case is only set on the entry block for a given
       * switch case, we only ever get here if we're jumping to the start of a
       * switch case.  It's possible, however, that a switch case could jump
       * to itself via a back-edge.  That *should* get caught by the loop
       * handling case above but if we have a back edge without a loop merge,
       * we could en up here.
       */
      vtn_fail_if(target_block->switch_case == switch_case,
                  "A switch cannot fall-through to itself.  Likely, there is "
                  "a back-edge which is not to a loop header.");

      vtn_fail_if(target_block->switch_case->node.parent !=
                     switch_case->node.parent,
                  "A switch case fall-through must come from the same "
                  "OpSwitch construct");

      vtn_fail_if(switch_case->fallthrough != NULL &&
                  switch_case->fallthrough != target_block->switch_case,
                  "Each case construct can have at most one branch to "
                  "another case construct");

      switch_case->fallthrough = target_block->switch_case;

      /* We don't immediately return vtn_branch_type_switch_fallthrough
       * because it may also be a loop or switch break for an inner loop or
       * switch and that takes precedence.
       */
   }

   if (loop && target_block == loop->cont_block)
      return vtn_branch_type_loop_continue;

   /* We walk blocks as a breadth-first search on the control-flow construct
    * tree where, when we find a construct, we add the vtn_cf_node for that
    * construct and continue iterating at the merge target block (if any).
    * Therefore, we want merges whose with parent == cf_parent to be treated
    * as regular branches.  We only want to consider merges if they break out
    * of the current CF construct.
    */
   if (target_block->merge_cf_node != NULL &&
       target_block->merge_cf_node->parent != cf_parent) {
      switch (target_block->merge_cf_node->type) {
      case vtn_cf_node_type_if:
         for (struct vtn_cf_node *node = cf_parent;
              node != target_block->merge_cf_node; node = node->parent) {
            vtn_fail_if(node == NULL || node->type != vtn_cf_node_type_if,
                        "Branching to the merge block of a selection "
                        "construct can only be used to break out of a "
                        "selection construct");

            struct vtn_if *if_stmt = vtn_cf_node_as_if(node);

            /* This should be guaranteed by our iteration */
            assert(if_stmt->merge_block != target_block);

            vtn_fail_if(if_stmt->merge_block != NULL,
                        "Branching to the merge block of a selection "
                        "construct can only be used to break out of the "
                        "inner most nested selection level");
         }
         return vtn_branch_type_if_merge;

      case vtn_cf_node_type_loop:
         vtn_fail_if(target_block->merge_cf_node != &loop->node,
                     "Loop breaks can only break out of the inner most "
                     "nested loop level");
         return vtn_branch_type_loop_break;

      case vtn_cf_node_type_switch: {
         struct vtn_switch *swtch = vtn_cf_node_find_switch(cf_parent);
         vtn_fail_if(target_block->merge_cf_node != &swtch->node,
                     "Switch breaks can only break out of the inner most "
                     "nested switch level");
         return vtn_branch_type_switch_break;
      }

      default:
         unreachable("Invalid CF node type for a merge");
      }
   }

   if (target_block->switch_case)
      return vtn_branch_type_switch_fallthrough;

   return vtn_branch_type_none;
}

struct vtn_cfg_work_item {
   struct list_head link;

   struct vtn_cf_node *cf_parent;
   struct list_head *cf_list;
   struct vtn_block *start_block;
};

static void
vtn_add_cfg_work_item(struct vtn_builder *b,
                      struct list_head *work_list,
                      struct vtn_cf_node *cf_parent,
                      struct list_head *cf_list,
                      struct vtn_block *start_block)
{
   struct vtn_cfg_work_item *work = ralloc(b, struct vtn_cfg_work_item);
   work->cf_parent = cf_parent;
   work->cf_list = cf_list;
   work->start_block = start_block;
   list_addtail(&work->link, work_list);
}

/* returns the default block */
static void
vtn_parse_switch(struct vtn_builder *b,
                 struct vtn_switch *swtch,
                 const uint32_t *branch,
                 struct list_head *case_list)
{
   const uint32_t *branch_end = branch + (branch[0] >> SpvWordCountShift);

   struct vtn_value *sel_val = vtn_untyped_value(b, branch[1]);
   vtn_fail_if(!sel_val->type ||
               sel_val->type->base_type != vtn_base_type_scalar,
               "Selector of OpSwitch must have a type of OpTypeInt");

   nir_alu_type sel_type =
      nir_get_nir_type_for_glsl_type(sel_val->type->type);
   vtn_fail_if(nir_alu_type_get_base_type(sel_type) != nir_type_int &&
               nir_alu_type_get_base_type(sel_type) != nir_type_uint,
               "Selector of OpSwitch must have a type of OpTypeInt");

   struct hash_table *block_to_case = _mesa_pointer_hash_table_create(b);

   bool is_default = true;
   const unsigned bitsize = nir_alu_type_get_type_size(sel_type);
   for (const uint32_t *w = branch + 2; w < branch_end;) {
      uint64_t literal = 0;
      if (!is_default) {
         if (bitsize <= 32) {
            literal = *(w++);
         } else {
            assert(bitsize == 64);
            literal = vtn_u64_literal(w);
            w += 2;
         }
      }
      struct vtn_block *case_block = vtn_block(b, *(w++));

      struct hash_entry *case_entry =
         _mesa_hash_table_search(block_to_case, case_block);

      struct vtn_case *cse;
      if (case_entry) {
         cse = case_entry->data;
      } else {
         cse = rzalloc(b, struct vtn_case);

         cse->node.type = vtn_cf_node_type_case;
         cse->node.parent = swtch ? &swtch->node : NULL;
         cse->block = case_block;
         list_inithead(&cse->body);
         util_dynarray_init(&cse->values, b);

         list_addtail(&cse->node.link, case_list);
         _mesa_hash_table_insert(block_to_case, case_block, cse);
      }

      if (is_default) {
         cse->is_default = true;
      } else {
         util_dynarray_append(&cse->values, uint64_t, literal);
      }

      is_default = false;
   }

   _mesa_hash_table_destroy(block_to_case, NULL);
}

/* Processes a block and returns the next block to process or NULL if we've
 * reached the end of the construct.
 */
static struct vtn_block *
vtn_process_block(struct vtn_builder *b,
                  struct list_head *work_list,
                  struct vtn_cf_node *cf_parent,
                  struct list_head *cf_list,
                  struct vtn_block *block)
{
   if (!list_is_empty(cf_list)) {
      /* vtn_process_block() acts like an iterator: it processes the given
       * block and then returns the next block to process.  For a given
       * control-flow construct, vtn_build_cfg() calls vtn_process_block()
       * repeatedly until it finally returns NULL.  Therefore, we know that
       * the only blocks on which vtn_process_block() can be called are either
       * the first block in a construct or a block that vtn_process_block()
       * returned for the current construct.  If cf_list is empty then we know
       * that we're processing the first block in the construct and we have to
       * add it to the list.
       *
       * If cf_list is not empty, then it must be the block returned by the
       * previous call to vtn_process_block().  We know a priori that
       * vtn_process_block only returns either normal branches
       * (vtn_branch_type_none) or merge target blocks.
       */
      switch (vtn_handle_branch(b, cf_parent, block)) {
      case vtn_branch_type_none:
         /* For normal branches, we want to process them and add them to the
          * current construct.  Merge target blocks also look like normal
          * branches from the perspective of this construct.  See also
          * vtn_handle_branch().
          */
         break;

      case vtn_branch_type_loop_continue:
      case vtn_branch_type_switch_fallthrough:
         /* The two cases where we can get early exits from a construct that
          * are not to that construct's merge target are loop continues and
          * switch fall-throughs.  In these cases, we need to break out of the
          * current construct by returning NULL.
          */
         return NULL;

      default:
         /* The only way we can get here is if something was used as two kinds
          * of merges at the same time and that's illegal.
          */
         vtn_fail("A block was used as a merge target from two or more "
                  "structured control-flow constructs");
      }
   }

   /* Once a block has been processed, it is placed into and the list link
    * will point to something non-null.  If we see a node we've already
    * processed here, it either exists in multiple functions or it's an
    * invalid back-edge.
    */
   if (block->node.parent != NULL) {
      vtn_fail_if(vtn_cf_node_find_function(&block->node) !=
                  vtn_cf_node_find_function(cf_parent),
                  "A block cannot exist in two functions at the "
                  "same time");

      vtn_fail("Invalid back or cross-edge in the CFG");
   }

   if (block->merge && (*block->merge & SpvOpCodeMask) == SpvOpLoopMerge &&
       block->loop == NULL) {
      vtn_fail_if((*block->branch & SpvOpCodeMask) != SpvOpBranch &&
                  (*block->branch & SpvOpCodeMask) != SpvOpBranchConditional,
                  "An OpLoopMerge instruction must immediately precede "
                  "either an OpBranch or OpBranchConditional instruction.");

      struct vtn_loop *loop = rzalloc(b, struct vtn_loop);

      loop->node.type = vtn_cf_node_type_loop;
      loop->node.parent = cf_parent;
      list_inithead(&loop->body);
      list_inithead(&loop->cont_body);
      loop->header_block = block;
      loop->break_block = vtn_block(b, block->merge[1]);
      loop->cont_block = vtn_block(b, block->merge[2]);
      loop->control = block->merge[3];

      list_addtail(&loop->node.link, cf_list);
      block->loop = loop;

      /* Note: The work item for the main loop body will start with the
       * current block as its start block.  If we weren't careful, we would
       * get here again and end up in an infinite loop.  This is why we set
       * block->loop above and check for it before creating one.  This way,
       * we only create the loop once and the second iteration that tries to
       * handle this loop goes to the cases below and gets handled as a
       * regular block.
       */
      vtn_add_cfg_work_item(b, work_list, &loop->node,
                            &loop->body, loop->header_block);

      /* For continue targets, SPIR-V guarantees the following:
       *
       *  - the Continue Target must dominate the back-edge block
       *  - the back-edge block must post dominate the Continue Target
       *
       * If the header block is the same as the continue target, this
       * condition is trivially satisfied and there is no real continue
       * section.
       */
      if (loop->cont_block != loop->header_block) {
         vtn_add_cfg_work_item(b, work_list, &loop->node,
                               &loop->cont_body, loop->cont_block);
      }

      vtn_block_set_merge_cf_node(b, loop->break_block, &loop->node);

      return loop->break_block;
   }

   /* Add the block to the CF list */
   block->node.parent = cf_parent;
   list_addtail(&block->node.link, cf_list);

   switch (*block->branch & SpvOpCodeMask) {
   case SpvOpBranch: {
      struct vtn_block *branch_block = vtn_block(b, block->branch[1]);

      block->branch_type = vtn_handle_branch(b, cf_parent, branch_block);

      if (block->branch_type == vtn_branch_type_none)
         return branch_block;
      else
         return NULL;
   }

   case SpvOpReturn:
   case SpvOpReturnValue:
      block->branch_type = vtn_branch_type_return;
      return NULL;

   case SpvOpKill:
      block->branch_type = vtn_branch_type_discard;
      return NULL;

   case SpvOpTerminateInvocation:
      block->branch_type = vtn_branch_type_terminate_invocation;
      return NULL;

   case SpvOpIgnoreIntersectionKHR:
      block->branch_type = vtn_branch_type_ignore_intersection;
      return NULL;

   case SpvOpTerminateRayKHR:
      block->branch_type = vtn_branch_type_terminate_ray;
      return NULL;

   case SpvOpEmitMeshTasksEXT:
      block->branch_type = vtn_branch_type_emit_mesh_tasks;
      return NULL;

   case SpvOpBranchConditional: {
      struct vtn_value *cond_val = vtn_untyped_value(b, block->branch[1]);
      vtn_fail_if(!cond_val->type ||
                  cond_val->type->base_type != vtn_base_type_scalar ||
                  cond_val->type->type != glsl_bool_type(),
                  "Condition must be a Boolean type scalar");

      struct vtn_if *if_stmt = rzalloc(b, struct vtn_if);

      if_stmt->node.type = vtn_cf_node_type_if;
      if_stmt->node.parent = cf_parent;
      if_stmt->header_block = block;
      list_inithead(&if_stmt->then_body);
      list_inithead(&if_stmt->else_body);

      list_addtail(&if_stmt->node.link, cf_list);

      if (block->merge &&
          (*block->merge & SpvOpCodeMask) == SpvOpSelectionMerge) {
         /* We may not always have a merge block and that merge doesn't
          * technically have to be an OpSelectionMerge.  We could have a block
          * with an OpLoopMerge which ends in an OpBranchConditional.
          */
         if_stmt->merge_block = vtn_block(b, block->merge[1]);
         vtn_block_set_merge_cf_node(b, if_stmt->merge_block, &if_stmt->node);

         if_stmt->control = block->merge[2];
      }

      struct vtn_block *then_block = vtn_block(b, block->branch[2]);
      if_stmt->then_type = vtn_handle_branch(b, &if_stmt->node, then_block);
      if (if_stmt->then_type == vtn_branch_type_none) {
         vtn_add_cfg_work_item(b, work_list, &if_stmt->node,
                               &if_stmt->then_body, then_block);
      }

      struct vtn_block *else_block = vtn_block(b, block->branch[3]);
      if (then_block != else_block) {
         if_stmt->else_type = vtn_handle_branch(b, &if_stmt->node, else_block);
         if (if_stmt->else_type == vtn_branch_type_none) {
            vtn_add_cfg_work_item(b, work_list, &if_stmt->node,
                                  &if_stmt->else_body, else_block);
         }
      }

      return if_stmt->merge_block;
   }

   case SpvOpSwitch: {
      struct vtn_switch *swtch = rzalloc(b, struct vtn_switch);

      swtch->node.type = vtn_cf_node_type_switch;
      swtch->node.parent = cf_parent;
      swtch->selector = block->branch[1];
      list_inithead(&swtch->cases);

      list_addtail(&swtch->node.link, cf_list);

      /* We may not always have a merge block */
      if (block->merge) {
         vtn_fail_if((*block->merge & SpvOpCodeMask) != SpvOpSelectionMerge,
                     "An OpLoopMerge instruction must immediately precede "
                     "either an OpBranch or OpBranchConditional "
                     "instruction.");
         swtch->break_block = vtn_block(b, block->merge[1]);
         vtn_block_set_merge_cf_node(b, swtch->break_block, &swtch->node);
      }

      /* First, we go through and record all of the cases. */
      vtn_parse_switch(b, swtch, block->branch, &swtch->cases);

      /* Gather the branch types for the switch */
      vtn_foreach_cf_node(case_node, &swtch->cases) {
         struct vtn_case *cse = vtn_cf_node_as_case(case_node);

         cse->type = vtn_handle_branch(b, &swtch->node, cse->block);
         switch (cse->type) {
         case vtn_branch_type_none:
            /* This is a "real" cases which has stuff in it */
            vtn_fail_if(cse->block->switch_case != NULL,
                        "OpSwitch has a case which is also in another "
                        "OpSwitch construct");
            cse->block->switch_case = cse;
            vtn_add_cfg_work_item(b, work_list, &cse->node,
                                  &cse->body, cse->block);
            break;

         case vtn_branch_type_switch_break:
         case vtn_branch_type_loop_break:
         case vtn_branch_type_loop_continue:
            /* Switch breaks as well as loop breaks and continues can be
             * used to break out of a switch construct or as direct targets
             * of the OpSwitch.
             */
            break;

         default:
            vtn_fail("Target of OpSwitch is not a valid structured exit "
                     "from the switch construct.");
         }
      }

      return swtch->break_block;
   }

   case SpvOpUnreachable:
      return NULL;

   default:
      vtn_fail("Block did not end with a valid branch instruction");
   }
}

void
vtn_build_cfg(struct vtn_builder *b, const uint32_t *words, const uint32_t *end)
{
   vtn_foreach_instruction(b, words, end,
                           vtn_cfg_handle_prepass_instruction);

   if (b->shader->info.stage == MESA_SHADER_KERNEL)
      return;

   vtn_foreach_cf_node(func_node, &b->functions) {
      struct vtn_function *func = vtn_cf_node_as_function(func_node);

      /* We build the CFG for each function by doing a breadth-first search on
       * the control-flow graph.  We keep track of our state using a worklist.
       * Doing a BFS ensures that we visit each structured control-flow
       * construct and its merge node before we visit the stuff inside the
       * construct.
       */
      struct list_head work_list;
      list_inithead(&work_list);
      vtn_add_cfg_work_item(b, &work_list, &func->node, &func->body,
                            func->start_block);

      while (!list_is_empty(&work_list)) {
         struct vtn_cfg_work_item *work =
            list_first_entry(&work_list, struct vtn_cfg_work_item, link);
         list_del(&work->link);

         for (struct vtn_block *block = work->start_block; block; ) {
            block = vtn_process_block(b, &work_list, work->cf_parent,
                                      work->cf_list, block);
         }
      }
   }
}

static bool
vtn_handle_phis_first_pass(struct vtn_builder *b, SpvOp opcode,
                           const uint32_t *w, unsigned count)
{
   if (opcode == SpvOpLabel)
      return true; /* Nothing to do */

   /* If this isn't a phi node, stop. */
   if (opcode != SpvOpPhi)
      return false;

   /* For handling phi nodes, we do a poor-man's out-of-ssa on the spot.
    * For each phi, we create a variable with the appropreate type and
    * do a load from that variable.  Then, in a second pass, we add
    * stores to that variable to each of the predecessor blocks.
    *
    * We could do something more intelligent here.  However, in order to
    * handle loops and things properly, we really need dominance
    * information.  It would end up basically being the into-SSA
    * algorithm all over again.  It's easier if we just let
    * lower_vars_to_ssa do that for us instead of repeating it here.
    */
   struct vtn_type *type = vtn_get_type(b, w[1]);
   nir_variable *phi_var =
      nir_local_variable_create(b->nb.impl, type->type, "phi");

   struct vtn_value *phi_val = vtn_untyped_value(b, w[2]);
   if (vtn_value_is_relaxed_precision(b, phi_val))
      phi_var->data.precision = GLSL_PRECISION_MEDIUM;

   _mesa_hash_table_insert(b->phi_table, w, phi_var);

   vtn_push_ssa_value(b, w[2],
      vtn_local_load(b, nir_build_deref_var(&b->nb, phi_var), 0));

   return true;
}

static bool
vtn_handle_phi_second_pass(struct vtn_builder *b, SpvOp opcode,
                           const uint32_t *w, unsigned count)
{
   if (opcode != SpvOpPhi)
      return true;

   struct hash_entry *phi_entry = _mesa_hash_table_search(b->phi_table, w);

   /* It's possible that this phi is in an unreachable block in which case it
    * may never have been emitted and therefore may not be in the hash table.
    * In this case, there's no var for it and it's safe to just bail.
    */
   if (phi_entry == NULL)
      return true;

   nir_variable *phi_var = phi_entry->data;

   for (unsigned i = 3; i < count; i += 2) {
      struct vtn_block *pred = vtn_block(b, w[i + 1]);

      /* If block does not have end_nop, that is because it is an unreacheable
       * block, and hence it is not worth to handle it */
      if (!pred->end_nop)
         continue;

      b->nb.cursor = nir_after_instr(&pred->end_nop->instr);

      struct vtn_ssa_value *src = vtn_ssa_value(b, w[i]);

      vtn_local_store(b, src, nir_build_deref_var(&b->nb, phi_var), 0);
   }

   return true;
}

static void
vtn_emit_ret_store(struct vtn_builder *b, const struct vtn_block *block)
{
   if ((*block->branch & SpvOpCodeMask) != SpvOpReturnValue)
      return;

   vtn_fail_if(b->func->type->return_type->base_type == vtn_base_type_void,
               "Return with a value from a function returning void");
   struct vtn_ssa_value *src = vtn_ssa_value(b, block->branch[1]);
   const struct glsl_type *ret_type =
      glsl_get_bare_type(b->func->type->return_type->type);
   nir_deref_instr *ret_deref =
      nir_build_deref_cast(&b->nb, nir_load_param(&b->nb, 0),
                           nir_var_function_temp, ret_type, 0);
   vtn_local_store(b, src, ret_deref, 0);
}

static void
vtn_emit_branch(struct vtn_builder *b, enum vtn_branch_type branch_type,
                const struct vtn_block *block,
                nir_variable *switch_fall_var, bool *has_switch_break)
{
   switch (branch_type) {
   case vtn_branch_type_if_merge:
      break; /* Nothing to do */
   case vtn_branch_type_switch_break:
      nir_store_var(&b->nb, switch_fall_var, nir_imm_false(&b->nb), 1);
      *has_switch_break = true;
      break;
   case vtn_branch_type_switch_fallthrough:
      break; /* Nothing to do */
   case vtn_branch_type_loop_break:
      nir_jump(&b->nb, nir_jump_break);
      break;
   case vtn_branch_type_loop_continue:
      nir_jump(&b->nb, nir_jump_continue);
      break;
   case vtn_branch_type_loop_back_edge:
      break;
   case vtn_branch_type_return:
      vtn_assert(block);
      vtn_emit_ret_store(b, block);
      nir_jump(&b->nb, nir_jump_return);
      break;
   case vtn_branch_type_discard:
      if (b->convert_discard_to_demote)
         nir_demote(&b->nb);
      else
         nir_discard(&b->nb);
      break;
   case vtn_branch_type_terminate_invocation:
      nir_terminate(&b->nb);
      break;
   case vtn_branch_type_ignore_intersection:
      nir_ignore_ray_intersection(&b->nb);
      nir_jump(&b->nb, nir_jump_halt);
      break;
   case vtn_branch_type_terminate_ray:
      nir_terminate_ray(&b->nb);
      nir_jump(&b->nb, nir_jump_halt);
      break;
   case vtn_branch_type_emit_mesh_tasks: {
      assert(block);
      assert(block->branch);

      const uint32_t *w = block->branch;
      vtn_assert((w[0] & SpvOpCodeMask) == SpvOpEmitMeshTasksEXT);

      /* Launches mesh shader workgroups from the task shader.
       * Arguments are: vec(x, y, z), payload pointer
       */
      nir_ssa_def *dimensions =
         nir_vec3(&b->nb, vtn_get_nir_ssa(b, w[1]),
                          vtn_get_nir_ssa(b, w[2]),
                          vtn_get_nir_ssa(b, w[3]));

      /* The payload variable is optional.
       * We don't have a NULL deref in NIR, so just emit the explicit
       * intrinsic when there is no payload.
       */
      const unsigned count = w[0] >> SpvWordCountShift;
      if (count == 4)
         nir_launch_mesh_workgroups(&b->nb, dimensions);
      else if (count == 5)
         nir_launch_mesh_workgroups_with_payload_deref(&b->nb, dimensions,
                                                       vtn_get_nir_ssa(b, w[4]));
      else
         vtn_fail("Invalid EmitMeshTasksEXT.");

      nir_jump(&b->nb, nir_jump_halt);
      break;
   }
   default:
      vtn_fail("Invalid branch type");
   }
}

static nir_ssa_def *
vtn_switch_case_condition(struct vtn_builder *b, struct vtn_switch *swtch,
                          nir_ssa_def *sel, struct vtn_case *cse)
{
   if (cse->is_default) {
      nir_ssa_def *any = nir_imm_false(&b->nb);
      vtn_foreach_cf_node(other_node, &swtch->cases) {
         struct vtn_case *other = vtn_cf_node_as_case(other_node);
         if (other->is_default)
            continue;

         any = nir_ior(&b->nb, any,
                       vtn_switch_case_condition(b, swtch, sel, other));
      }
      return nir_inot(&b->nb, any);
   } else {
      nir_ssa_def *cond = nir_imm_false(&b->nb);
      util_dynarray_foreach(&cse->values, uint64_t, val)
         cond = nir_ior(&b->nb, cond, nir_ieq_imm(&b->nb, sel, *val));
      return cond;
   }
}

static nir_loop_control
vtn_loop_control(struct vtn_builder *b, struct vtn_loop *vtn_loop)
{
   if (vtn_loop->control == SpvLoopControlMaskNone)
      return nir_loop_control_none;
   else if (vtn_loop->control & SpvLoopControlDontUnrollMask)
      return nir_loop_control_dont_unroll;
   else if (vtn_loop->control & SpvLoopControlUnrollMask)
      return nir_loop_control_unroll;
   else if (vtn_loop->control & SpvLoopControlDependencyInfiniteMask ||
            vtn_loop->control & SpvLoopControlDependencyLengthMask ||
            vtn_loop->control & SpvLoopControlMinIterationsMask ||
            vtn_loop->control & SpvLoopControlMaxIterationsMask ||
            vtn_loop->control & SpvLoopControlIterationMultipleMask ||
            vtn_loop->control & SpvLoopControlPeelCountMask ||
            vtn_loop->control & SpvLoopControlPartialCountMask) {
      /* We do not do anything special with these yet. */
      return nir_loop_control_none;
   } else {
      vtn_fail("Invalid loop control");
   }
}

static nir_selection_control
vtn_selection_control(struct vtn_builder *b, struct vtn_if *vtn_if)
{
   if (vtn_if->control == SpvSelectionControlMaskNone)
      return nir_selection_control_none;
   else if (vtn_if->control & SpvSelectionControlDontFlattenMask)
      return nir_selection_control_dont_flatten;
   else if (vtn_if->control & SpvSelectionControlFlattenMask)
      return nir_selection_control_flatten;
   else
      vtn_fail("Invalid selection control");
}

static void
vtn_emit_cf_list_structured(struct vtn_builder *b, struct list_head *cf_list,
                            nir_variable *switch_fall_var,
                            bool *has_switch_break,
                            vtn_instruction_handler handler)
{
   vtn_foreach_cf_node(node, cf_list) {
      switch (node->type) {
      case vtn_cf_node_type_block: {
         struct vtn_block *block = vtn_cf_node_as_block(node);

         const uint32_t *block_start = block->label;
         const uint32_t *block_end = block->merge ? block->merge :
                                                    block->branch;

         block_start = vtn_foreach_instruction(b, block_start, block_end,
                                               vtn_handle_phis_first_pass);

         vtn_foreach_instruction(b, block_start, block_end, handler);

         block->end_nop = nir_nop(&b->nb);

         if (block->branch_type != vtn_branch_type_none) {
            vtn_emit_branch(b, block->branch_type, block,
                            switch_fall_var, has_switch_break);
            return;
         }

         break;
      }

      case vtn_cf_node_type_if: {
         struct vtn_if *vtn_if = vtn_cf_node_as_if(node);
         const uint32_t *branch = vtn_if->header_block->branch;
         vtn_assert((branch[0] & SpvOpCodeMask) == SpvOpBranchConditional);

         bool sw_break = false;
         /* If both branches are the same, just emit the first block, which is
          * the only one we filled when building the CFG.
          */
         if (branch[2] == branch[3]) {
            if (vtn_if->then_type == vtn_branch_type_none) {
               vtn_emit_cf_list_structured(b, &vtn_if->then_body,
                                           switch_fall_var, &sw_break, handler);
            } else {
               vtn_emit_branch(b, vtn_if->then_type, NULL, switch_fall_var, &sw_break);
            }
            break;
         }

         nir_if *nif =
            nir_push_if(&b->nb, vtn_get_nir_ssa(b, branch[1]));

         nif->control = vtn_selection_control(b, vtn_if);

         if (vtn_if->then_type == vtn_branch_type_none) {
            vtn_emit_cf_list_structured(b, &vtn_if->then_body,
                                        switch_fall_var, &sw_break, handler);
         } else {
            vtn_emit_branch(b, vtn_if->then_type, NULL, switch_fall_var, &sw_break);
         }

         nir_push_else(&b->nb, nif);
         if (vtn_if->else_type == vtn_branch_type_none) {
            vtn_emit_cf_list_structured(b, &vtn_if->else_body,
                                        switch_fall_var, &sw_break, handler);
         } else {
            vtn_emit_branch(b, vtn_if->else_type, NULL, switch_fall_var, &sw_break);
         }

         nir_pop_if(&b->nb, nif);

         /* If we encountered a switch break somewhere inside of the if,
          * then it would have been handled correctly by calling
          * emit_cf_list or emit_branch for the interrior.  However, we
          * need to predicate everything following on wether or not we're
          * still going.
          */
         if (sw_break) {
            *has_switch_break = true;
            nir_push_if(&b->nb, nir_load_var(&b->nb, switch_fall_var));
         }
         break;
      }

      case vtn_cf_node_type_loop: {
         struct vtn_loop *vtn_loop = vtn_cf_node_as_loop(node);

         nir_loop *loop = nir_push_loop(&b->nb);
         loop->control = vtn_loop_control(b, vtn_loop);

         vtn_emit_cf_list_structured(b, &vtn_loop->body, NULL, NULL, handler);

         if (!list_is_empty(&vtn_loop->cont_body)) {
            /* If we have a non-trivial continue body then we need to put
             * it at the beginning of the loop with a flag to ensure that
             * it doesn't get executed in the first iteration.
             */
            nir_variable *do_cont =
               nir_local_variable_create(b->nb.impl, glsl_bool_type(), "cont");

            b->nb.cursor = nir_before_cf_node(&loop->cf_node);
            nir_store_var(&b->nb, do_cont, nir_imm_false(&b->nb), 1);

            b->nb.cursor = nir_before_cf_list(&loop->body);

            nir_if *cont_if =
               nir_push_if(&b->nb, nir_load_var(&b->nb, do_cont));

            vtn_emit_cf_list_structured(b, &vtn_loop->cont_body, NULL, NULL,
                                        handler);

            nir_pop_if(&b->nb, cont_if);

            nir_store_var(&b->nb, do_cont, nir_imm_true(&b->nb), 1);
         }

         nir_pop_loop(&b->nb, loop);
         break;
      }

      case vtn_cf_node_type_switch: {
         struct vtn_switch *vtn_switch = vtn_cf_node_as_switch(node);

         /* Before we can emit anything, we need to sort the list of cases in
          * fall-through order.
          */
         vtn_switch_order_cases(vtn_switch);

         /* First, we create a variable to keep track of whether or not the
          * switch is still going at any given point.  Any switch breaks
          * will set this variable to false.
          */
         nir_variable *fall_var =
            nir_local_variable_create(b->nb.impl, glsl_bool_type(), "fall");
         nir_store_var(&b->nb, fall_var, nir_imm_false(&b->nb), 1);

         nir_ssa_def *sel = vtn_get_nir_ssa(b, vtn_switch->selector);

         /* Now we can walk the list of cases and actually emit code */
         vtn_foreach_cf_node(case_node, &vtn_switch->cases) {
            struct vtn_case *cse = vtn_cf_node_as_case(case_node);

            /* If this case jumps directly to the break block, we don't have
             * to handle the case as the body is empty and doesn't fall
             * through.
             */
            if (cse->block == vtn_switch->break_block)
               continue;

            /* Figure out the condition */
            nir_ssa_def *cond =
               vtn_switch_case_condition(b, vtn_switch, sel, cse);
            /* Take fallthrough into account */
            cond = nir_ior(&b->nb, cond, nir_load_var(&b->nb, fall_var));

            nir_if *case_if = nir_push_if(&b->nb, cond);

            bool has_break = false;
            nir_store_var(&b->nb, fall_var, nir_imm_true(&b->nb), 1);
            vtn_emit_cf_list_structured(b, &cse->body, fall_var, &has_break,
                                        handler);
            (void)has_break; /* We don't care */

            nir_pop_if(&b->nb, case_if);
         }

         break;
      }

      default:
         vtn_fail("Invalid CF node type");
      }
   }
}

static struct nir_block *
vtn_new_unstructured_block(struct vtn_builder *b, struct vtn_function *func)
{
   struct nir_block *n = nir_block_create(b->shader);
   exec_list_push_tail(&func->nir_func->impl->body, &n->cf_node.node);
   n->cf_node.parent = &func->nir_func->impl->cf_node;
   return n;
}

static void
vtn_add_unstructured_block(struct vtn_builder *b,
                           struct vtn_function *func,
                           struct list_head *work_list,
                           struct vtn_block *block)
{
   if (!block->block) {
      block->block = vtn_new_unstructured_block(b, func);
      list_addtail(&block->node.link, work_list);
   }
}

static void
vtn_emit_cf_func_unstructured(struct vtn_builder *b, struct vtn_function *func,
                              vtn_instruction_handler handler)
{
   struct list_head work_list;
   list_inithead(&work_list);

   func->start_block->block = nir_start_block(func->nir_func->impl);
   list_addtail(&func->start_block->node.link, &work_list);
   while (!list_is_empty(&work_list)) {
      struct vtn_block *block =
         list_first_entry(&work_list, struct vtn_block, node.link);
      list_del(&block->node.link);

      vtn_assert(block->block);

      const uint32_t *block_start = block->label;
      const uint32_t *block_end = block->branch;

      b->nb.cursor = nir_after_block(block->block);
      block_start = vtn_foreach_instruction(b, block_start, block_end,
                                            vtn_handle_phis_first_pass);
      vtn_foreach_instruction(b, block_start, block_end, handler);
      block->end_nop = nir_nop(&b->nb);

      SpvOp op = *block_end & SpvOpCodeMask;
      switch (op) {
      case SpvOpBranch: {
         struct vtn_block *branch_block = vtn_block(b, block->branch[1]);
         vtn_add_unstructured_block(b, func, &work_list, branch_block);
         nir_goto(&b->nb, branch_block->block);
         break;
      }

      case SpvOpBranchConditional: {
         nir_ssa_def *cond = vtn_ssa_value(b, block->branch[1])->def;
         struct vtn_block *then_block = vtn_block(b, block->branch[2]);
         struct vtn_block *else_block = vtn_block(b, block->branch[3]);

         vtn_add_unstructured_block(b, func, &work_list, then_block);
         if (then_block == else_block) {
            nir_goto(&b->nb, then_block->block);
         } else {
            vtn_add_unstructured_block(b, func, &work_list, else_block);
            nir_goto_if(&b->nb, then_block->block, nir_src_for_ssa(cond),
                                else_block->block);
         }

         break;
      }

      case SpvOpSwitch: {
         struct list_head cases;
         list_inithead(&cases);
         vtn_parse_switch(b, NULL, block->branch, &cases);

         nir_ssa_def *sel = vtn_get_nir_ssa(b, block->branch[1]);

         struct vtn_case *def = NULL;
         vtn_foreach_cf_node(case_node, &cases) {
            struct vtn_case *cse = vtn_cf_node_as_case(case_node);
            if (cse->is_default) {
               assert(def == NULL);
               def = cse;
               continue;
            }

            nir_ssa_def *cond = nir_imm_false(&b->nb);
            util_dynarray_foreach(&cse->values, uint64_t, val)
               cond = nir_ior(&b->nb, cond, nir_ieq_imm(&b->nb, sel, *val));

            /* block for the next check */
            nir_block *e = vtn_new_unstructured_block(b, func);
            vtn_add_unstructured_block(b, func, &work_list, cse->block);

            /* add branching */
            nir_goto_if(&b->nb, cse->block->block, nir_src_for_ssa(cond), e);
            b->nb.cursor = nir_after_block(e);
         }

         vtn_assert(def != NULL);
         vtn_add_unstructured_block(b, func, &work_list, def->block);

         /* now that all cases are handled, branch into the default block */
         nir_goto(&b->nb, def->block->block);
         break;
      }

      case SpvOpKill: {
         nir_discard(&b->nb);
         nir_goto(&b->nb, b->func->nir_func->impl->end_block);
         break;
      }

      case SpvOpUnreachable:
      case SpvOpReturn:
      case SpvOpReturnValue: {
         vtn_emit_ret_store(b, block);
         nir_goto(&b->nb, b->func->nir_func->impl->end_block);
         break;
      }

      default:
         vtn_fail("Unhandled opcode %s", spirv_op_to_string(op));
      }
   }
}

void
vtn_function_emit(struct vtn_builder *b, struct vtn_function *func,
                  vtn_instruction_handler preamble_instruction_handler,
                  const uint32_t *preamble_words,
                  vtn_instruction_handler instruction_handler)
{
   static int force_unstructured = -1;
   if (force_unstructured < 0) {
      force_unstructured =
         debug_get_bool_option("MESA_SPIRV_FORCE_UNSTRUCTURED", false);
   }

   nir_function_impl *impl = func->nir_func->impl;
   nir_builder_init(&b->nb, impl);
   b->func = func;
   b->nb.cursor = nir_after_cf_list(&impl->body);
   b->nb.exact = b->exact;
   b->phi_table = _mesa_pointer_hash_table_create(b);

   const uint32_t *word_end = b->spirv + b->spirv_word_count;
   vtn_foreach_instruction(b, preamble_words, word_end, preamble_instruction_handler);

   if (b->shader->info.stage == MESA_SHADER_KERNEL || force_unstructured) {
      impl->structured = false;
      vtn_emit_cf_func_unstructured(b, func, instruction_handler);
   } else {
      vtn_emit_cf_list_structured(b, &func->body, NULL, NULL,
                                  instruction_handler);
   }

   vtn_foreach_instruction(b, func->start_block->label, func->end,
                           vtn_handle_phi_second_pass);

   if (func->nir_func->impl->structured)
      nir_copy_prop_impl(impl);
   nir_rematerialize_derefs_in_use_blocks_impl(impl);

   /*
    * There are some cases where we need to repair SSA to insert
    * the needed phi nodes:
    *
    * - Continue blocks for loops get inserted before the body of the loop
    *   but instructions in the continue may use SSA defs in the loop body.
    *
    * - Early termination instructions `OpKill` and `OpTerminateInvocation`,
    *   in NIR. They're represented by regular intrinsics with no control-flow
    *   semantics. This means that the SSA form from the SPIR-V may not
    *   100% match NIR.
    *
    * - Switches with only default case may also define SSA which may
    *   subsequently be used out of the switch.
    */
   if (func->nir_func->impl->structured)
      nir_repair_ssa_impl(impl);

   func->emitted = true;
}
