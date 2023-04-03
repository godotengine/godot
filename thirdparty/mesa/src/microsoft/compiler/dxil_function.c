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

#include "dxil_function.h"
#include "dxil_module.h"

#define MAX_FUNC_PARAMS 17

struct predefined_func_descr {
   const char *base_name;
   const char *retval_descr;
   const char *param_descr;
   enum dxil_attr_kind attr;
};

static struct  predefined_func_descr predefined_funcs[] = {
{"dx.op.atomicBinOp", "O", "i@iiiii", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.cbufferLoad", "O", "i@ii", DXIL_ATTR_KIND_READ_ONLY},
{"dx.op.cbufferLoadLegacy", "B", "i@i", DXIL_ATTR_KIND_READ_ONLY},
{"dx.op.createHandle", "@", "iciib", DXIL_ATTR_KIND_READ_ONLY},
{"dx.op.storeOutput", "v", "iiicO", DXIL_ATTR_KIND_NO_UNWIND},
{"dx.op.loadInput", "O", "iiici", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.tertiary", "O", "iOOO", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.quaternary", "O", "iOOOO", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.threadId", "i", "ii", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.threadIdInGroup", "i", "ii", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.flattenedThreadIdInGroup", "i", "i", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.groupId", "i", "ii", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.unary", "O", "iO", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.unaryBits", "i", "iO", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.isSpecialFloat", "b", "iO", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.binary", "O", "iOO", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.bufferStore", "v", "i@iiOOOOc", DXIL_ATTR_KIND_NONE},
{"dx.op.bufferLoad", "R", "i@ii", DXIL_ATTR_KIND_READ_ONLY},
{"dx.op.attributeAtVertex", "O", "iiicc", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.sample", "R", "i@@ffffiiif", DXIL_ATTR_KIND_READ_ONLY},
{"dx.op.sampleBias", "R", "i@@ffffiiiff", DXIL_ATTR_KIND_READ_ONLY},
{"dx.op.sampleLevel", "R", "i@@ffffiiif", DXIL_ATTR_KIND_READ_ONLY},
{"dx.op.sampleGrad", "R", "i@@ffffiiifffffff", DXIL_ATTR_KIND_READ_ONLY},
{"dx.op.sampleCmp", "R", "i@@ffffiiiff", DXIL_ATTR_KIND_READ_ONLY},
{"dx.op.sampleCmpLevel", "R", "i@@ffffiiiff", DXIL_ATTR_KIND_READ_ONLY},
{"dx.op.sampleCmpLevelZero", "R", "i@@ffffiiif", DXIL_ATTR_KIND_READ_ONLY},
{"dx.op.textureLoad", "R", "i@iiiiiii", DXIL_ATTR_KIND_READ_ONLY},
{"dx.op.textureGather", "R", "i@@ffffiii", DXIL_ATTR_KIND_READ_ONLY},
{"dx.op.textureGatherCmp", "R", "i@@ffffiiif", DXIL_ATTR_KIND_READ_ONLY},
{"dx.op.discard", "v", "ib", DXIL_ATTR_KIND_NO_UNWIND},
{"dx.op.sampleIndex", "i", "i", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.emitStream", "v", "ic", DXIL_ATTR_KIND_NONE},
{"dx.op.cutStream", "v", "ic", DXIL_ATTR_KIND_NONE},
{"dx.op.getDimensions", "D", "i@i", DXIL_ATTR_KIND_READ_ONLY},
{"dx.op.calculateLOD", "f", "i@@fffb", DXIL_ATTR_KIND_READ_ONLY},
{"dx.op.barrier", "v", "ii", DXIL_ATTR_KIND_NO_DUPLICATE},
{"dx.op.atomicCompareExchange", "O", "i@iiiii", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.textureStore", "v", "i@iiiOOOOc", DXIL_ATTR_KIND_NONE},
{"dx.op.primitiveID", "i", "i", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.outputControlPointID", "i", "i", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.gsInstanceID", "i", "i", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.viewID", "i", "i", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.domainLocation", "f", "ii", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.legacyF16ToF32", "f", "ii", DXIL_ATTR_KIND_READ_ONLY},
{"dx.op.legacyF32ToF16", "i", "if", DXIL_ATTR_KIND_READ_ONLY},
{"dx.op.makeDouble", "g", "iii", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.splitDouble", "G", "ig", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.texture2DMSGetSamplePosition", "S", "i@i", DXIL_ATTR_KIND_READ_ONLY},
{"dx.op.renderTargetGetSamplePosition", "S", "ii", DXIL_ATTR_KIND_READ_ONLY},
{"dx.op.evalSnapped", "O", "iiicii", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.evalCentroid", "O", "iiic", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.evalSampleIndex", "O", "iiici", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.coverage", "i", "i", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.storePatchConstant", "v", "iiicO", DXIL_ATTR_KIND_NO_UNWIND},
{"dx.op.loadPatchConstant", "O", "iiic", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.loadOutputControlPoint", "O", "iiici", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.createHandleFromBinding", "@", "i#ib", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.annotateHandle", "@", "i@P", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.isHelperLane", "b", "i", DXIL_ATTR_KIND_READ_ONLY},
{"dx.op.waveIsFirstLane", "b", "i", DXIL_ATTR_KIND_NO_UNWIND},
{"dx.op.waveGetLaneIndex", "i", "i", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.waveGetLaneCount", "i", "i", DXIL_ATTR_KIND_READ_NONE},
{"dx.op.waveReadLaneFirst", "O", "iO", DXIL_ATTR_KIND_NO_UNWIND},
};

struct func_descr {
   const char *name;
   enum overload_type overload;
};

struct func_rb_node {
   struct rb_node node;
   const struct dxil_func *func;
   struct func_descr descr;
};

static inline
const struct func_rb_node *
func_rb_node(const struct rb_node *n)
{
   return (const struct func_rb_node *)n;
}

static int
func_compare_to_name_and_overload(const struct rb_node *node, const void *data)
{
   const struct func_descr *descr = (const struct func_descr *)data;
   const struct func_rb_node *f = func_rb_node(node);
   if (f->descr.overload < descr->overload)
      return -1;
   if (f->descr.overload > descr->overload)
      return 1;

   return strcmp(f->descr.name, descr->name);
}

static const struct dxil_func *
allocate_function_from_predefined(struct dxil_module *mod,
                                       const char *name,
                                       enum overload_type overload)
{
   for (unsigned i = 0; i < ARRAY_SIZE(predefined_funcs); ++i) {
      if (!strcmp(predefined_funcs[i].base_name, name)) {
         return dxil_alloc_func(mod, name, overload,
                                predefined_funcs[i].retval_descr,
                                predefined_funcs[i].param_descr,
                                predefined_funcs[i].attr);
      }
   }
   unreachable("Invalid function name");
}

const struct dxil_func *
dxil_get_function(struct dxil_module *mod,
                  const char *name, enum overload_type overload)
{
   struct func_descr descr = { name, overload };
   const struct rb_node *node = rb_tree_search(mod->functions, &descr,
                                               func_compare_to_name_and_overload);
   if (node)
      return func_rb_node(node)->func;

   return allocate_function_from_predefined(mod, name, overload);
}

static int func_compare_name(const struct rb_node *lhs, const struct rb_node *rhs)
{
   const struct func_rb_node *node = func_rb_node(rhs);
   return func_compare_to_name_and_overload(lhs, &node->descr);
}

static void
dxil_add_function(struct rb_tree *functions, const struct dxil_func *func,
                  const char *name, enum overload_type overload)
{
   struct func_rb_node *f = rzalloc(functions, struct func_rb_node);
   f->func = func;
   f->descr.name = name;
   f->descr.overload = overload;
   rb_tree_insert(functions, &f->node, func_compare_name);
}

static const struct dxil_type *
get_type_from_string(struct dxil_module *mod, const char *param_descr,
                     enum overload_type overload,  int *idx)
{
   assert(param_descr);
   char type_id = param_descr[(*idx)++];
   assert(*idx <= (int)strlen(param_descr));

   switch (type_id) {
   case DXIL_FUNC_PARAM_INT64: return dxil_module_get_int_type(mod, 64);
   case DXIL_FUNC_PARAM_INT32: return dxil_module_get_int_type(mod, 32);
   case DXIL_FUNC_PARAM_INT16: return dxil_module_get_int_type(mod, 16);
   case DXIL_FUNC_PARAM_INT8: return dxil_module_get_int_type(mod, 8);
   case DXIL_FUNC_PARAM_BOOL: return dxil_module_get_int_type(mod, 1);
   case DXIL_FUNC_PARAM_FLOAT64: return dxil_module_get_float_type(mod, 64);
   case DXIL_FUNC_PARAM_FLOAT32: return dxil_module_get_float_type(mod, 32);
   case DXIL_FUNC_PARAM_FLOAT16: return dxil_module_get_float_type(mod, 16);
   case DXIL_FUNC_PARAM_HANDLE: return dxil_module_get_handle_type(mod);
   case DXIL_FUNC_PARAM_VOID: return dxil_module_get_void_type(mod);
   case DXIL_FUNC_PARAM_FROM_OVERLOAD:  return dxil_get_overload_type(mod, overload);
   case DXIL_FUNC_PARAM_RESRET: return dxil_module_get_resret_type(mod, overload);
   case DXIL_FUNC_PARAM_DIM: return dxil_module_get_dimret_type(mod);
   case DXIL_FUNC_PARAM_SAMPLE_POS: return dxil_module_get_samplepos_type(mod);
   case DXIL_FUNC_PARAM_CBUF_RET: return dxil_module_get_cbuf_ret_type(mod, overload);
   case DXIL_FUNC_PARAM_SPLIT_DOUBLE: return dxil_module_get_split_double_ret_type(mod);
   case DXIL_FUNC_PARAM_RES_BIND: return dxil_module_get_res_bind_type(mod);
   case DXIL_FUNC_PARAM_RES_PROPS: return dxil_module_get_res_props_type(mod);
   case DXIL_FUNC_PARAM_POINTER: {
         const struct dxil_type *target = get_type_from_string(mod, param_descr, overload, idx);
         return dxil_module_get_pointer_type(mod, target);
      }
   default:
      assert(0 && "unknown type identifier");
   }
   return NULL;
}

const struct dxil_func *
dxil_alloc_func_with_rettype(struct dxil_module *mod, const char *name,
                             enum overload_type overload,
                             const struct dxil_type *retval_type,
                             const char *param_descr,
                             enum dxil_attr_kind attr)
{
   assert(param_descr);
   const struct dxil_type *arg_types[MAX_FUNC_PARAMS];

   int index = 0;
   unsigned num_params = 0;

   while (param_descr[num_params]) {
      const struct dxil_type *t = get_type_from_string(mod, param_descr, overload, &index);
      if (!t)
         return false;
      assert(num_params < MAX_FUNC_PARAMS);
      arg_types[num_params++] = t;
   }

   const struct dxil_type *func_type =
      dxil_module_add_function_type(mod, retval_type,
                                    arg_types, num_params);
   if (!func_type) {
      fprintf(stderr, "%s: Func type allocation failed\n", __func__);
      return false;
   }

   char full_name[100];
   snprintf(full_name, sizeof (full_name), "%s%s%s", name,
            overload == DXIL_NONE ? "" : ".", dxil_overload_suffix(overload));
   const struct dxil_func *func = dxil_add_function_decl(mod, full_name, func_type, attr);

   if (func)
      dxil_add_function(mod->functions, func, name, overload);

   return func;
}

const struct dxil_func *
dxil_alloc_func(struct dxil_module *mod, const char *name, enum overload_type overload,
                const char *retval_type_descr,
                const char *param_descr, enum dxil_attr_kind attr)
{

   int index = 0;
   const struct dxil_type *retval_type = get_type_from_string(mod, retval_type_descr, overload, &index);
   assert(retval_type_descr[index] == 0);

   return dxil_alloc_func_with_rettype(mod, name, overload, retval_type,
                                       param_descr, attr);
}
