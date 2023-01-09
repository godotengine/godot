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

#include "dxil_dump.h"
#include "dxil_internal.h"

#define DIXL_DUMP_DECL
#include "dxil_dump_decls.h"

#include "dxil_module.h"


#include "util/string_buffer.h"
#include "util/list.h"

#include <stdio.h>

struct dxil_dumper {
   struct _mesa_string_buffer *buf;
   int current_indent;
};

struct dxil_dumper *dxil_dump_create(void)
{
   struct dxil_dumper *d = calloc(1, sizeof(struct dxil_dumper));
   d->buf = _mesa_string_buffer_create(NULL, 1024);
   d->current_indent = 0;
   return d;
}

void dxil_dump_free(struct dxil_dumper *d)
{
   _mesa_string_buffer_destroy(d->buf);
   d->buf = 0;
   free(d);
}

void dxil_dump_buf_to_file(struct dxil_dumper *d, FILE *f)
{
   assert(f);
   assert(d);
   assert(d->buf);
   fprintf(f, "%s", d->buf->buf);
}

static
void dxil_dump_indention_inc(struct dxil_dumper *d)
{
   ++d->current_indent;
}

static
void dxil_dump_indention_dec(struct dxil_dumper *d)
{
   --d->current_indent;
   assert(d->current_indent >= 0);
}

static
void dxil_dump_indent(struct dxil_dumper *d)
{
   for (int i = 0; i  < 2 * d->current_indent; ++i)
      _mesa_string_buffer_append_char(d->buf, ' ');
}

void
dxil_dump_module(struct dxil_dumper *d, struct dxil_module *m)
{
   assert(m);
   assert(d);

   _mesa_string_buffer_printf(d->buf, "DXIL MODULE:\n");
   dump_metadata(d, m);
   dump_shader_info(d, &m->info);
   dump_types(d, &m->type_list);
   dump_gvars(d, &m->gvar_list);
   dump_funcs(d, &m->func_list);
   dump_attr_set_list(d, &m->attr_set_list);
   dump_constants(d, &m->const_list);

   struct dxil_func_def *func_def;
   LIST_FOR_EACH_ENTRY(func_def, &m->func_def_list, head) {
      dump_instrs(d, &func_def->instr_list);
   }

   dump_mdnodes(d, &m->mdnode_list);
   dump_named_nodes(d, &m->md_named_node_list);
   dump_io_signatures(d->buf, m);
   dump_psv(d->buf, m);
   _mesa_string_buffer_printf(d->buf, "END DXIL MODULE\n");
}

static void
dump_metadata(struct dxil_dumper *d, struct dxil_module *m)
{
   _mesa_string_buffer_printf(d->buf, "Shader: %s\n",
                              dump_shader_string(m->shader_kind));

   _mesa_string_buffer_printf(d->buf, "Version: %d.%d\n",
                              m->major_version, m->minor_version);

   dump_features(d->buf, &m->feats);
}

static void
dump_shader_info(struct dxil_dumper *d, struct dxil_shader_info *info)
{
   _mesa_string_buffer_append(d->buf, "Shader Info:\n");
   if (info->has_out_position)
      _mesa_string_buffer_append(d->buf, "  has_out_position\n");
}

static const char *
dump_shader_string(enum dxil_shader_kind kind)
{
#define SHADER_STR(X) case DXIL_ ## X ## _SHADER: return #X

   switch (kind) {
   SHADER_STR(VERTEX);
   SHADER_STR(PIXEL);
   SHADER_STR(GEOMETRY);
   SHADER_STR(COMPUTE);
   default:
      return "UNSUPPORTED";
   }
#undef SHADER_STR
}

static void
dump_features(struct _mesa_string_buffer *buf, struct dxil_features *feat)
{
   _mesa_string_buffer_printf(buf, "Features:\n");
#define PRINT_FEAT(F) if (feat->F) _mesa_string_buffer_printf(buf, "  %s\n", #F)
   PRINT_FEAT(doubles);
   PRINT_FEAT(cs_4x_raw_sb);
   PRINT_FEAT(uavs_at_every_stage);
   PRINT_FEAT(use_64uavs);
   PRINT_FEAT(min_precision);
   PRINT_FEAT(dx11_1_double_extensions);
   PRINT_FEAT(dx11_1_shader_extensions);
   PRINT_FEAT(dx9_comparison_filtering);
   PRINT_FEAT(tiled_resources);
   PRINT_FEAT(stencil_ref);
   PRINT_FEAT(inner_coverage);
   PRINT_FEAT(typed_uav_load_additional_formats);
   PRINT_FEAT(rovs);
   PRINT_FEAT(array_layer_from_vs_or_ds);
   PRINT_FEAT(wave_ops);
   PRINT_FEAT(int64_ops);
   PRINT_FEAT(view_id);
   PRINT_FEAT(barycentrics);
   PRINT_FEAT(native_low_precision);
   PRINT_FEAT(shading_rate);
   PRINT_FEAT(raytracing_tier_1_1);
   PRINT_FEAT(sampler_feedback);
#undef PRINT_FEAT
}

static void
dump_types(struct dxil_dumper *d, struct list_head *list)
{
   if (!list_length(list))
      return;

   _mesa_string_buffer_append(d->buf, "Types:\n");
   dxil_dump_indention_inc(d);
   list_for_each_entry(struct dxil_type, type, list, head) {
      dxil_dump_indent(d);
      dump_type(d, type);
      _mesa_string_buffer_append(d->buf, "\n");
   }
   dxil_dump_indention_dec(d);
}

static void dump_type_name(struct dxil_dumper *d, const struct dxil_type *type)
{
   if (!type) {
      _mesa_string_buffer_append(d->buf, "(type error)");
      return;
   }

   switch (type->type) {
   case TYPE_VOID:
      _mesa_string_buffer_append(d->buf, "void");
      break;
   case TYPE_INTEGER:
      _mesa_string_buffer_printf(d->buf, "int%d", type->int_bits);
      break;
   case TYPE_FLOAT:
      _mesa_string_buffer_printf(d->buf, "float%d", type->float_bits);
      break;
   case TYPE_POINTER:
      dump_type_name(d, type->ptr_target_type);
      _mesa_string_buffer_append(d->buf, "*");
      break;
   case TYPE_STRUCT:
      _mesa_string_buffer_printf(d->buf, "struct %s", type->struct_def.name);
      break;
   case TYPE_ARRAY:
      dump_type_name(d, type->array_or_vector_def.elem_type);
      _mesa_string_buffer_printf(d->buf, "[%d]", type->array_or_vector_def.num_elems);
      break;
   case TYPE_FUNCTION:
      _mesa_string_buffer_append(d->buf, "(");
      dump_type_name(d, type->function_def.ret_type);
      _mesa_string_buffer_append(d->buf, ")(");
      for (size_t i = 0; i < type->function_def.args.num_types; ++i) {
         if (i > 0)
            _mesa_string_buffer_append(d->buf, ", ");
         dump_type_name(d, type->function_def.args.types[i]);
      }
      _mesa_string_buffer_append(d->buf, ")");
      break;
   case TYPE_VECTOR:
      _mesa_string_buffer_append(d->buf, "vector<");
      dump_type_name(d, type->array_or_vector_def.elem_type);
      _mesa_string_buffer_printf(d->buf, ", %d>", type->array_or_vector_def.num_elems);
      break;
   default:
      _mesa_string_buffer_printf(d->buf, "unknown type %d", type->type);
   }
}

static void
dump_type(struct dxil_dumper *d, const struct dxil_type *type)
{
   switch (type->type) {
   case TYPE_STRUCT:
      _mesa_string_buffer_printf(d->buf, "struct %s {\n", type->struct_def.name);
      dxil_dump_indention_inc(d);

      for (size_t i = 0; i < type->struct_def.elem.num_types; ++i) {
         dxil_dump_indent(d);
         dump_type(d, type->struct_def.elem.types[i]);
         _mesa_string_buffer_append(d->buf, "\n");
      }
      dxil_dump_indention_dec(d);
      dxil_dump_indent(d);
      _mesa_string_buffer_append(d->buf, "}\n");
      break;
   default:
      dump_type_name(d, type);
      break;
   }
}

static void
dump_gvars(struct dxil_dumper *d, struct list_head *list)
{
   if (!list_length(list))
      return;

   _mesa_string_buffer_append(d->buf, "Global variables:\n");
   dxil_dump_indention_inc(d);
   list_for_each_entry(struct dxil_gvar, gvar, list, head) {
      dxil_dump_indent(d);
      _mesa_string_buffer_printf(d->buf, "address_space(%d) ", gvar->as);
      if (gvar->constant)
         _mesa_string_buffer_append(d->buf, "const ");
      if (gvar->align)
         _mesa_string_buffer_append(d->buf, "align ");
      if (gvar->initializer)
         _mesa_string_buffer_printf(d->buf, "init_id:%d\n", gvar->initializer->id);
      dump_type_name(d, gvar->type);
      _mesa_string_buffer_printf(d->buf, " val_id:%d\n", gvar->value.id);
   }
   dxil_dump_indention_dec(d);
}

static void
dump_funcs(struct dxil_dumper *d, struct list_head *list)
{
   if (!list_length(list))
      return;

   _mesa_string_buffer_append(d->buf, "Functions:\n");
   dxil_dump_indention_inc(d);
   list_for_each_entry(struct dxil_func, func, list, head) {
      dxil_dump_indent(d);
      if (func->decl)
         _mesa_string_buffer_append(d->buf, "declare ");
      _mesa_string_buffer_append(d->buf, func->name);
      _mesa_string_buffer_append_char(d->buf, ' ');
      dump_type_name(d, func->type);
      if (func->attr_set)
         _mesa_string_buffer_printf(d->buf, " #%d", func->attr_set);
      _mesa_string_buffer_append_char(d->buf, '\n');
   }
   dxil_dump_indention_dec(d);
}

static void
dump_attr_set_list(struct dxil_dumper *d, struct list_head *list)
{
   if (!list_length(list))
      return;

   _mesa_string_buffer_append(d->buf, "Attribute set:\n");
   dxil_dump_indention_inc(d);
   int attr_id = 1;
   list_for_each_entry(struct attrib_set, attr, list, head) {
      _mesa_string_buffer_printf(d->buf, "  #%d: {", attr_id++);
      for (unsigned i = 0; i < attr->num_attrs; ++i) {
         if (i > 0)
            _mesa_string_buffer_append_char(d->buf, ' ');

         assert(attr->attrs[i].type == DXIL_ATTR_ENUM);
         const char *value = "";
         switch (attr->attrs[i].kind) {
         case DXIL_ATTR_KIND_NONE: value = "none"; break;
         case DXIL_ATTR_KIND_NO_UNWIND: value = "nounwind"; break;
         case DXIL_ATTR_KIND_READ_NONE: value = "readnone"; break;
         case DXIL_ATTR_KIND_READ_ONLY: value = "readonly"; break;
         case DXIL_ATTR_KIND_NO_DUPLICATE: value = "noduplicate"; break;
         }
         _mesa_string_buffer_append(d->buf, value);
      }
      _mesa_string_buffer_append(d->buf, "}\n");
   }
   dxil_dump_indention_dec(d);
}

static void
dump_constants(struct dxil_dumper *d, struct list_head *list)
{
   if (!list_length(list))
      return;

   _mesa_string_buffer_append(d->buf, "Constants:\n");
   dxil_dump_indention_inc(d);
   list_for_each_entry(struct dxil_const, cnst, list, head) {
      _mesa_string_buffer_append_char(d->buf, ' ');
      dump_value(d, &cnst->value);
      _mesa_string_buffer_append(d->buf, " = ");
      dump_type_name(d, cnst->value.type);
      if (!cnst->undef) {
         switch (cnst->value.type->type) {
         case TYPE_FLOAT:
            _mesa_string_buffer_printf(d->buf, " %10.5f\n", cnst->float_value);
            break;
         case TYPE_INTEGER:
            _mesa_string_buffer_printf(d->buf, " %d\n", cnst->int_value);
            break;
         case TYPE_ARRAY:
            _mesa_string_buffer_append(d->buf, "{");
            for (unsigned i = 0;
                 i < cnst->value.type->array_or_vector_def.num_elems; i++) {
               _mesa_string_buffer_printf(d->buf, " %%%d",
                                          cnst->array_values[i]->id);
               dump_type_name(d, cnst->value.type);
               if (i != cnst->value.type->array_or_vector_def.num_elems - 1)
                  _mesa_string_buffer_append(d->buf, ",");
               _mesa_string_buffer_append(d->buf, " ");
            }
            _mesa_string_buffer_append(d->buf, "}\n");
            break;
         default:
            unreachable("Unsupported const type");
         }
      } else
         _mesa_string_buffer_append(d->buf, " undef\n");
   }
   dxil_dump_indention_dec(d);
}

static void
dump_instrs(struct dxil_dumper *d, struct list_head *list)
{
   _mesa_string_buffer_append(d->buf, "Shader body:\n");
   dxil_dump_indention_inc(d);

   list_for_each_entry(struct dxil_instr, instr, list, head) {

      dxil_dump_indent(d);
      if (instr->has_value) {
         dump_value(d, &instr->value);
         _mesa_string_buffer_append(d->buf, " = ");
      } else {
         _mesa_string_buffer_append_char(d->buf, ' ');
      }

      switch (instr->type) {
      case INSTR_BINOP: dump_instr_binop(d, &instr->binop); break;
      case INSTR_CMP:   dump_instr_cmp(d, &instr->cmp);break;
      case INSTR_SELECT:dump_instr_select(d, &instr->select); break;
      case INSTR_CAST:  dump_instr_cast(d, &instr->cast); break;
      case INSTR_CALL:  dump_instr_call(d, &instr->call); break;
      case INSTR_RET:   dump_instr_ret(d, &instr->ret); break;
      case INSTR_EXTRACTVAL: dump_instr_extractval(d, &instr->extractval); break;
      case INSTR_BR:  dump_instr_branch(d, &instr->br); break;
      case INSTR_PHI:  dump_instr_phi(d, &instr->phi); break;
      case INSTR_ALLOCA: dump_instr_alloca(d, &instr->alloca); break;
      case INSTR_GEP: dump_instr_gep(d, &instr->gep); break;
      case INSTR_LOAD: dump_instr_load(d, &instr->load); break;
      case INSTR_STORE: dump_instr_store(d, &instr->store); break;
      case INSTR_ATOMICRMW: dump_instr_atomicrmw(d, &instr->atomicrmw); break;
      default:
         _mesa_string_buffer_printf(d->buf, "unknown instruction type %d", instr->type);
      }

      _mesa_string_buffer_append(d->buf, "\n");
   }
   dxil_dump_indention_dec(d);
}

static void
dump_instr_binop(struct dxil_dumper *d, struct dxil_instr_binop *binop)
{
   const char *str = binop->opcode < DXIL_BINOP_INSTR_COUNT ?
                        binop_strings[binop->opcode] : "INVALID";

   _mesa_string_buffer_printf(d->buf, "%s ", str);
   dump_instr_print_operands(d, 2, binop->operands);
}

static void
dump_instr_cmp(struct dxil_dumper *d, struct dxil_instr_cmp *cmp)
{
   const char *str = cmp->pred < DXIL_CMP_INSTR_COUNT ?
                        pred_strings[cmp->pred] : "INVALID";

   _mesa_string_buffer_printf(d->buf, "%s ", str);
   dump_instr_print_operands(d, 2, cmp->operands);
}

static void
dump_instr_select(struct dxil_dumper *d, struct dxil_instr_select *select)
{
   _mesa_string_buffer_append(d->buf, "sel ");
   dump_instr_print_operands(d, 3, select->operands);
}

static void
dump_instr_cast(struct dxil_dumper *d, struct dxil_instr_cast *cast)
{
   const char *str = cast->opcode < DXIL_CAST_INSTR_COUNT ?
                        cast_opcode_strings[cast->opcode] : "INVALID";

   _mesa_string_buffer_printf(d->buf, "%s.", str);
   dump_type_name(d, cast->type);
   _mesa_string_buffer_append_char(d->buf, ' ');
   dump_value(d, cast->value);
}

static void
dump_instr_call(struct dxil_dumper *d, struct dxil_instr_call *call)
{
   assert(call->num_args == call->func->type->function_def.args.num_types);
   struct dxil_type **func_arg_types = call->func->type->function_def.args.types;

   _mesa_string_buffer_printf(d->buf, "%s(", call->func->name);
   for (unsigned i = 0; i < call->num_args; ++i) {
      if (i > 0)
         _mesa_string_buffer_append(d->buf, ", ");
      dump_type_name(d, func_arg_types[i]);
      _mesa_string_buffer_append_char(d->buf, ' ');
      dump_value(d, call->args[i]);
   }
   _mesa_string_buffer_append_char(d->buf, ')');
}

static void
dump_instr_ret(struct dxil_dumper *d, struct dxil_instr_ret *ret)
{
   _mesa_string_buffer_append(d->buf, "ret ");
   if (ret->value)
      dump_value(d, ret->value);
}

static void
dump_instr_extractval(struct dxil_dumper *d, struct dxil_instr_extractval *extr)
{
   _mesa_string_buffer_append(d->buf, "extractvalue ");
   dump_type_name(d, extr->type);
   dump_value(d, extr->src);
   _mesa_string_buffer_printf(d->buf, ", %d", extr->idx);
}

static void
dump_instr_branch(struct dxil_dumper *d, struct dxil_instr_br *br)
{
   _mesa_string_buffer_append(d->buf, "branch ");
   if (br->cond)
      dump_value(d, br->cond);
   else
      _mesa_string_buffer_append(d->buf, " (uncond)");
   _mesa_string_buffer_printf(d->buf, " %d %d", br->succ[0], br->succ[1]);
}

static void
dump_instr_phi(struct dxil_dumper *d, struct dxil_instr_phi *phi)
{
   _mesa_string_buffer_append(d->buf, "phi ");
   dump_type_name(d, phi->type);
   struct dxil_phi_src *src = phi->incoming;
   for (unsigned i = 0; i < phi->num_incoming; ++i, ++src) {
      if (i > 0)
         _mesa_string_buffer_append(d->buf, ", ");
      dump_value(d, src->value);
      _mesa_string_buffer_printf(d->buf, "(%d)", src->block);
   }
}

static void
dump_instr_alloca(struct dxil_dumper *d, struct dxil_instr_alloca *alloca)
{
   _mesa_string_buffer_append(d->buf, "alloca ");
   dump_type_name(d, alloca->alloc_type);
   _mesa_string_buffer_append(d->buf, ", ");
   dump_type_name(d, alloca->size_type);
   _mesa_string_buffer_append(d->buf, ", ");
   dump_value(d, alloca->size);
   unsigned align_mask = (1 << 6 ) - 1;
   unsigned align = alloca->align & align_mask;
   _mesa_string_buffer_printf(d->buf, ", %d", 1 << (align - 1));
}

static void
dump_instr_gep(struct dxil_dumper *d, struct dxil_instr_gep *gep)
{
   _mesa_string_buffer_append(d->buf, "getelementptr ");
   if (gep->inbounds)
      _mesa_string_buffer_append(d->buf, "inbounds ");
   dump_type_name(d, gep->source_elem_type);
   _mesa_string_buffer_append(d->buf, ", ");
   for (unsigned i = 0; i < gep->num_operands; ++i) {
      if (i > 0)
         _mesa_string_buffer_append(d->buf, ", ");
      dump_value(d, gep->operands[i]);
   }
}

static void
dump_instr_load(struct dxil_dumper *d, struct dxil_instr_load *load)
{
   _mesa_string_buffer_append(d->buf, "load ");
   if (load->is_volatile)
      _mesa_string_buffer_append(d->buf, " volatile");
   dump_type_name(d, load->type);
   _mesa_string_buffer_append(d->buf, ", ");
   dump_value(d, load->ptr);
   _mesa_string_buffer_printf(d->buf, ", %d", load->align);
}

static void
dump_instr_store(struct dxil_dumper *d, struct dxil_instr_store *store)
{
   _mesa_string_buffer_append(d->buf, "store ");
   if (store->is_volatile)
      _mesa_string_buffer_append(d->buf, " volatile");
   dump_value(d, store->value);
   _mesa_string_buffer_append(d->buf, ", ");
   dump_value(d, store->ptr);
   _mesa_string_buffer_printf(d->buf, ", %d", store->align);
}

static const char *rmworder_str[] = {
   [DXIL_ATOMIC_ORDERING_NOTATOMIC] = "not-atomic",
   [DXIL_ATOMIC_ORDERING_UNORDERED] = "unordered",
   [DXIL_ATOMIC_ORDERING_MONOTONIC] = "monotonic",
   [DXIL_ATOMIC_ORDERING_ACQUIRE] = "acquire",
   [DXIL_ATOMIC_ORDERING_RELEASE] = "release",
   [DXIL_ATOMIC_ORDERING_ACQREL] = "acqrel",
   [DXIL_ATOMIC_ORDERING_SEQCST] = "seqcst",
};

static const char *rmwsync_str[] = {
   [DXIL_SYNC_SCOPE_SINGLETHREAD] = "single-thread",
   [DXIL_SYNC_SCOPE_CROSSTHREAD] = "cross-thread",
};

static const char *rmwop_str[] = {
   [DXIL_RMWOP_XCHG] = "xchg",
   [DXIL_RMWOP_ADD] = "add",
   [DXIL_RMWOP_SUB] = "sub",
   [DXIL_RMWOP_AND] = "and",
   [DXIL_RMWOP_NAND] = "nand",
   [DXIL_RMWOP_OR] = "or",
   [DXIL_RMWOP_XOR] = "xor",
   [DXIL_RMWOP_MAX] = "max",
   [DXIL_RMWOP_MIN] = "min",
   [DXIL_RMWOP_UMAX] = "umax",
   [DXIL_RMWOP_UMIN] = "umin",
};

static void
dump_instr_atomicrmw(struct dxil_dumper *d, struct dxil_instr_atomicrmw *rmw)
{
   _mesa_string_buffer_printf(d->buf, "atomicrmw.%s ", rmwop_str[rmw->op]);

   if (rmw->is_volatile)
      _mesa_string_buffer_append(d->buf, " volatile");
   dump_value(d, rmw->value);
   _mesa_string_buffer_append(d->buf, ", ");
   dump_value(d, rmw->ptr);
   _mesa_string_buffer_printf(d->buf, ", ordering(%s)", rmworder_str[rmw->ordering]);
   _mesa_string_buffer_printf(d->buf, ", sync_scope(%s)", rmwsync_str[rmw->syncscope]);
}

static void
dump_instr_print_operands(struct dxil_dumper *d, int num,
                          const struct dxil_value *val[])
{
   for (int i = 0; i < num; ++i) {
      if (i > 0)
         _mesa_string_buffer_append(d->buf, ", ");
      dump_value(d, val[i]);
   }
}

static void
dump_value(struct dxil_dumper *d, const struct dxil_value *val)
{
   if (val->id < 10)
      _mesa_string_buffer_append(d->buf, " ");
   if (val->id < 100)
      _mesa_string_buffer_append(d->buf, " ");
   _mesa_string_buffer_printf(d->buf, "%%%d", val->id);
  dump_type_name(d, val->type);
}

static void
dump_mdnodes(struct dxil_dumper *d, struct list_head *list)
{
   if (!list_length(list))
      return;

   _mesa_string_buffer_append(d->buf, "MD-Nodes:\n");
   dxil_dump_indention_inc(d);
   list_for_each_entry(struct dxil_mdnode, node, list, head) {
      dump_mdnode(d, node);
   }
   dxil_dump_indention_dec(d);
}

static void
dump_mdnode(struct dxil_dumper *d, const struct dxil_mdnode *node)
{
   dxil_dump_indent(d);
   switch (node->type) {
   case MD_STRING:
      _mesa_string_buffer_printf(d->buf, "S:%s\n", node->string);
      break;
   case MD_VALUE:
      _mesa_string_buffer_append(d->buf, "V:");
      dump_type_name(d, node->value.type);
      _mesa_string_buffer_append_char(d->buf, ' ');
      dump_value(d, node->value.value);
      _mesa_string_buffer_append_char(d->buf, '\n');
      break;
   case MD_NODE:
      _mesa_string_buffer_append(d->buf, " \\\n");
      dxil_dump_indention_inc(d);
      for (size_t i = 0; i < node->node.num_subnodes; ++i) {
         if (node->node.subnodes[i])
            dump_mdnode(d, node->node.subnodes[i]);
         else {
            dxil_dump_indent(d);
            _mesa_string_buffer_append(d->buf, "(nullptr)\n");
         }
      }
      dxil_dump_indention_dec(d);
      break;
   }
}

static void
dump_named_nodes(struct dxil_dumper *d, struct list_head *list)
{
   if (!list_length(list))
      return;

   _mesa_string_buffer_append(d->buf, "Named Nodes:\n");
   dxil_dump_indention_inc(d);
   list_for_each_entry(struct dxil_named_node, node, list, head) {
      dxil_dump_indent(d);
      _mesa_string_buffer_printf(d->buf, "%s:\n", node->name);
      dxil_dump_indention_inc(d);
      for (size_t i = 0; i < node->num_subnodes; ++i) {
         if (node->subnodes[i])
            dump_mdnode(d, node->subnodes[i]);
         else {
            dxil_dump_indent(d);
            _mesa_string_buffer_append(d->buf, "(nullptr)\n");
         }
      }
      dxil_dump_indention_dec(d);
   }
   dxil_dump_indention_dec(d);
}

static void
mask_to_string(uint32_t mask, char str[5])
{
   const char *mc = "xyzw";
   for (int i = 0; i < 4 && mask; ++i) {
      str[i] = (mask & (1 << i)) ? mc[i] : '_';
   }
   str[4] = 0;
}

static void dump_io_signatures(struct _mesa_string_buffer *buf, struct dxil_module *m)
{
   _mesa_string_buffer_append(buf, "\nInput signature:\n");
   dump_io_signature(buf, m->num_sig_inputs, m->inputs);
   _mesa_string_buffer_append(buf, "\nOutput signature:\n");
   dump_io_signature(buf, m->num_sig_outputs, m->outputs);
}

static void dump_io_signature(struct _mesa_string_buffer *buf, unsigned num,
                              struct dxil_signature_record *io)
{
   _mesa_string_buffer_append(buf, " SEMANTIC-NAME Index Mask Reg SysValue Format\n");
   _mesa_string_buffer_append(buf, "----------------------------------------------\n");
   for (unsigned i = 0; i < num; ++i, ++io)  {
      for (unsigned j = 0; j < io->num_elements; ++j) {
         char mask[5] = "";
         mask_to_string(io->elements[j].mask, mask);
         _mesa_string_buffer_printf(buf, "%-15s %3d %4s %3d %-8s %-7s\n",
                                    io->name, io->elements[j].semantic_index,
                                    mask, io->elements[j].reg, io->sysvalue,
                                    component_type_as_string(io->elements[j].comp_type));
      }
   }
}

static const char *component_type_as_string(uint32_t type)
{
   return  (type < DXIL_PROG_SIG_COMP_TYPE_COUNT) ?
            dxil_type_strings[type] : "invalid";
}

static void dump_psv(struct _mesa_string_buffer *buf,
                     struct dxil_module *m)
{
   _mesa_string_buffer_append(buf, "\nPipeline State Validation\nInputs:\n");
   dump_psv_io(buf, m, m->num_sig_inputs, m->psv_inputs);
   _mesa_string_buffer_append(buf, "\nOutputs:\n");
   dump_psv_io(buf, m, m->num_sig_outputs, m->psv_outputs);
}

static void dump_psv_io(struct _mesa_string_buffer *buf, struct dxil_module *m,
                        unsigned num, struct dxil_psv_signature_element *io)
{
   _mesa_string_buffer_append(buf, " SEMANTIC-NAME Rows Cols Kind Comp-Type Interp dynmask+stream Indices\n");
   _mesa_string_buffer_append(buf, "----------------------------------------------\n");
   for (unsigned  i = 0; i < num; ++i, ++io)  {
      _mesa_string_buffer_printf(buf, "%-14s %d+%d  %d+%d %4d   %-7s    %-4d        %-9d [",
              m->sem_string_table->buf + io->semantic_name_offset,
              (int)io->start_row, (int)io->rows,
              (int)((io->cols_and_start & 0xf) >> 4),
              (int)(io->cols_and_start & 0xf),
              (int)io->semantic_kind,
              component_type_as_string(io->component_type),
              (int)io->interpolation_mode,
              (int)io->dynamic_mask_and_stream);
      for (int k = 0; k < io->rows; ++k) {
         if (k > 0)
            _mesa_string_buffer_append(buf, ", ");
         _mesa_string_buffer_printf(buf,"%d ", m->sem_index_table.data[io->start_row  + k]);
      }
      _mesa_string_buffer_append(buf, "]\n");
   }
}
