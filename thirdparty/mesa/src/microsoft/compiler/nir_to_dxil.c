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

#include "nir_to_dxil.h"

#include "dxil_container.h"
#include "dxil_dump.h"
#include "dxil_enums.h"
#include "dxil_function.h"
#include "dxil_module.h"
#include "dxil_nir.h"
#include "dxil_signature.h"

#include "nir/nir_builder.h"
#include "util/ralloc.h"
#include "util/u_debug.h"
#include "util/u_dynarray.h"
#include "util/u_math.h"

#include "git_sha1.h"

#include "vulkan/vulkan_core.h"

#include <stdint.h>

#include "drivers/d3d12/d3d12_godot_nir_bridge.h"

int debug_dxil = 0;

static const struct debug_named_value
dxil_debug_options[] = {
   { "verbose", DXIL_DEBUG_VERBOSE, NULL },
   { "dump_blob",  DXIL_DEBUG_DUMP_BLOB , "Write shader blobs" },
   { "trace",  DXIL_DEBUG_TRACE , "Trace instruction conversion" },
   { "dump_module", DXIL_DEBUG_DUMP_MODULE, "dump module tree to stderr"},
   DEBUG_NAMED_VALUE_END
};

DEBUG_GET_ONCE_FLAGS_OPTION(debug_dxil, "DXIL_DEBUG", dxil_debug_options, 0)

static void
log_nir_instr_unsupported(const struct dxil_logger *logger,
                          const char *message_prefix, const nir_instr *instr)
{
   char *msg = NULL;
   char *instr_str = nir_instr_as_str(instr, NULL);
   asprintf(&msg, "%s: %s\n", message_prefix, instr_str);
   ralloc_free(instr_str);
   assert(msg);
   logger->log(logger->priv, msg);
   free(msg);
}

static void
default_logger_func(void *priv, const char *msg)
{
   fprintf(stderr, "%s", msg);
   unreachable("Unhandled error");
}

static const struct dxil_logger default_logger = { .priv = NULL, .log = default_logger_func };

#define TRACE_CONVERSION(instr) \
   if (debug_dxil & DXIL_DEBUG_TRACE) \
      do { \
         fprintf(stderr, "Convert '"); \
         nir_print_instr(instr, stderr); \
         fprintf(stderr, "'\n"); \
      } while (0)

static const nir_shader_compiler_options
nir_options = {
   .lower_ineg = true,
   .lower_fneg = true,
   .lower_ffma16 = true,
   .lower_ffma32 = true,
   .lower_isign = true,
   .lower_fsign = true,
   .lower_iabs = true,
   .lower_fmod = true,
   .lower_fpow = true,
   .lower_scmp = true,
   .lower_ldexp = true,
   .lower_flrp16 = true,
   .lower_flrp32 = true,
   .lower_flrp64 = true,
   .lower_bitfield_extract = true,
   .lower_find_msb_to_reverse = true,
   .lower_extract_word = true,
   .lower_extract_byte = true,
   .lower_insert_word = true,
   .lower_insert_byte = true,
   .lower_all_io_to_elements = true,
   .lower_all_io_to_temps = true,
   .lower_hadd = true,
   .lower_uadd_sat = true,
   .lower_usub_sat = true,
   .lower_iadd_sat = true,
   .lower_uadd_carry = true,
   .lower_usub_borrow = true,
   .lower_mul_high = true,
   .lower_rotate = true,
   .lower_pack_half_2x16 = true,
   .lower_pack_unorm_4x8 = true,
   .lower_pack_snorm_4x8 = true,
   .lower_pack_snorm_2x16 = true,
   .lower_pack_unorm_2x16 = true,
   .lower_pack_64_2x32_split = true,
   .lower_pack_32_2x16_split = true,
   .lower_unpack_64_2x32_split = true,
   .lower_unpack_32_2x16_split = true,
   .lower_unpack_half_2x16 = true,
   .lower_unpack_snorm_2x16 = true,
   .lower_unpack_snorm_4x8 = true,
   .lower_unpack_unorm_2x16 = true,
   .lower_unpack_unorm_4x8 = true,
   .lower_interpolate_at = true,
   .has_fsub = true,
   .has_isub = true,
   .use_scoped_barrier = true,
   .vertex_id_zero_based = true,
   .lower_base_vertex = true,
   .lower_helper_invocation = true,
   .has_cs_global_id = true,
   .has_txs = true,
   .lower_mul_2x32_64 = true,
   .lower_doubles_options =
      nir_lower_drcp |
      nir_lower_dsqrt |
      nir_lower_drsq |
      nir_lower_dfract |
      nir_lower_dtrunc |
      nir_lower_dfloor |
      nir_lower_dceil |
      nir_lower_dround_even,
   .max_unroll_iterations = 32, /* arbitrary */
   .force_indirect_unrolling = (nir_var_shader_in | nir_var_shader_out | nir_var_function_temp),
};

const nir_shader_compiler_options*
dxil_get_nir_compiler_options(void)
{
   return &nir_options;
}

static bool
emit_llvm_ident(struct dxil_module *m)
{
   const struct dxil_mdnode *compiler = dxil_get_metadata_string(m, "Mesa version " PACKAGE_VERSION MESA_GIT_SHA1);
   if (!compiler)
      return false;

   const struct dxil_mdnode *llvm_ident = dxil_get_metadata_node(m, &compiler, 1);
   return llvm_ident &&
          dxil_add_metadata_named_node(m, "llvm.ident", &llvm_ident, 1);
}

static bool
emit_named_version(struct dxil_module *m, const char *name,
                   int major, int minor)
{
   const struct dxil_mdnode *major_node = dxil_get_metadata_int32(m, major);
   const struct dxil_mdnode *minor_node = dxil_get_metadata_int32(m, minor);
   const struct dxil_mdnode *version_nodes[] = { major_node, minor_node };
   const struct dxil_mdnode *version = dxil_get_metadata_node(m, version_nodes,
                                                     ARRAY_SIZE(version_nodes));
   return dxil_add_metadata_named_node(m, name, &version, 1);
}

static const char *
get_shader_kind_str(enum dxil_shader_kind kind)
{
   switch (kind) {
   case DXIL_PIXEL_SHADER:
      return "ps";
   case DXIL_VERTEX_SHADER:
      return "vs";
   case DXIL_GEOMETRY_SHADER:
      return "gs";
   case DXIL_HULL_SHADER:
      return "hs";
   case DXIL_DOMAIN_SHADER:
      return "ds";
   case DXIL_COMPUTE_SHADER:
      return "cs";
   default:
      unreachable("invalid shader kind");
   }
}

static bool
emit_dx_shader_model(struct dxil_module *m)
{
   const struct dxil_mdnode *type_node = dxil_get_metadata_string(m, get_shader_kind_str(m->shader_kind));
   const struct dxil_mdnode *major_node = dxil_get_metadata_int32(m, m->major_version);
   const struct dxil_mdnode *minor_node = dxil_get_metadata_int32(m, m->minor_version);
   const struct dxil_mdnode *shader_model[] = { type_node, major_node,
                                                minor_node };
   const struct dxil_mdnode *dx_shader_model = dxil_get_metadata_node(m, shader_model, ARRAY_SIZE(shader_model));

   return dxil_add_metadata_named_node(m, "dx.shaderModel",
                                       &dx_shader_model, 1);
}

enum {
   DXIL_TYPED_BUFFER_ELEMENT_TYPE_TAG = 0,
   DXIL_STRUCTURED_BUFFER_ELEMENT_STRIDE_TAG = 1
};

enum dxil_intr {
   DXIL_INTR_LOAD_INPUT = 4,
   DXIL_INTR_STORE_OUTPUT = 5,
   DXIL_INTR_FABS = 6,
   DXIL_INTR_SATURATE = 7,

   DXIL_INTR_ISFINITE = 10,
   DXIL_INTR_ISNORMAL = 11,

   DXIL_INTR_FCOS = 12,
   DXIL_INTR_FSIN = 13,

   DXIL_INTR_FEXP2 = 21,
   DXIL_INTR_FRC = 22,
   DXIL_INTR_FLOG2 = 23,

   DXIL_INTR_SQRT = 24,
   DXIL_INTR_RSQRT = 25,
   DXIL_INTR_ROUND_NE = 26,
   DXIL_INTR_ROUND_NI = 27,
   DXIL_INTR_ROUND_PI = 28,
   DXIL_INTR_ROUND_Z = 29,

   DXIL_INTR_BFREV = 30,
   DXIL_INTR_COUNTBITS = 31,
   DXIL_INTR_FIRSTBIT_LO = 32,
   DXIL_INTR_FIRSTBIT_HI = 33,
   DXIL_INTR_FIRSTBIT_SHI = 34,

   DXIL_INTR_FMAX = 35,
   DXIL_INTR_FMIN = 36,
   DXIL_INTR_IMAX = 37,
   DXIL_INTR_IMIN = 38,
   DXIL_INTR_UMAX = 39,
   DXIL_INTR_UMIN = 40,

   DXIL_INTR_FMA = 47,

   DXIL_INTR_IBFE = 51,
   DXIL_INTR_UBFE = 52,
   DXIL_INTR_BFI = 53,

   DXIL_INTR_CREATE_HANDLE = 57,
   DXIL_INTR_CBUFFER_LOAD_LEGACY = 59,

   DXIL_INTR_SAMPLE = 60,
   DXIL_INTR_SAMPLE_BIAS = 61,
   DXIL_INTR_SAMPLE_LEVEL = 62,
   DXIL_INTR_SAMPLE_GRAD = 63,
   DXIL_INTR_SAMPLE_CMP = 64,
   DXIL_INTR_SAMPLE_CMP_LVL_ZERO = 65,

   DXIL_INTR_TEXTURE_LOAD = 66,
   DXIL_INTR_TEXTURE_STORE = 67,

   DXIL_INTR_BUFFER_LOAD = 68,
   DXIL_INTR_BUFFER_STORE = 69,

   DXIL_INTR_TEXTURE_SIZE = 72,
   DXIL_INTR_TEXTURE_GATHER = 73,
   DXIL_INTR_TEXTURE_GATHER_CMP = 74,

   DXIL_INTR_TEXTURE2DMS_GET_SAMPLE_POSITION = 75,
   DXIL_INTR_RENDER_TARGET_GET_SAMPLE_POSITION = 76,
   DXIL_INTR_RENDER_TARGET_GET_SAMPLE_COUNT = 77,

   DXIL_INTR_ATOMIC_BINOP = 78,
   DXIL_INTR_ATOMIC_CMPXCHG = 79,
   DXIL_INTR_BARRIER = 80,
   DXIL_INTR_TEXTURE_LOD = 81,

   DXIL_INTR_DISCARD = 82,
   DXIL_INTR_DDX_COARSE = 83,
   DXIL_INTR_DDY_COARSE = 84,
   DXIL_INTR_DDX_FINE = 85,
   DXIL_INTR_DDY_FINE = 86,

   DXIL_INTR_EVAL_SNAPPED = 87,
   DXIL_INTR_EVAL_SAMPLE_INDEX = 88,
   DXIL_INTR_EVAL_CENTROID = 89,

   DXIL_INTR_SAMPLE_INDEX = 90,
   DXIL_INTR_COVERAGE = 91,

   DXIL_INTR_THREAD_ID = 93,
   DXIL_INTR_GROUP_ID = 94,
   DXIL_INTR_THREAD_ID_IN_GROUP = 95,
   DXIL_INTR_FLATTENED_THREAD_ID_IN_GROUP = 96,

   DXIL_INTR_EMIT_STREAM = 97,
   DXIL_INTR_CUT_STREAM = 98,

   DXIL_INTR_GS_INSTANCE_ID = 100,

   DXIL_INTR_MAKE_DOUBLE = 101,
   DXIL_INTR_SPLIT_DOUBLE = 102,

   DXIL_INTR_LOAD_OUTPUT_CONTROL_POINT = 103,
   DXIL_INTR_LOAD_PATCH_CONSTANT = 104,
   DXIL_INTR_DOMAIN_LOCATION = 105,
   DXIL_INTR_STORE_PATCH_CONSTANT = 106,
   DXIL_INTR_OUTPUT_CONTROL_POINT_ID = 107,
   DXIL_INTR_PRIMITIVE_ID = 108,

   DXIL_INTR_WAVE_IS_FIRST_LANE = 110,
   DXIL_INTR_WAVE_GET_LANE_INDEX = 111,
   DXIL_INTR_WAVE_GET_LANE_COUNT = 112,
   DXIL_INTR_WAVE_READ_LANE_FIRST = 118,

   DXIL_INTR_LEGACY_F32TOF16 = 130,
   DXIL_INTR_LEGACY_F16TOF32 = 131,

   DXIL_INTR_ATTRIBUTE_AT_VERTEX = 137,
   DXIL_INTR_VIEW_ID = 138,

   DXIL_INTR_ANNOTATE_HANDLE = 216,
   DXIL_INTR_CREATE_HANDLE_FROM_BINDING = 217,

   DXIL_INTR_IS_HELPER_LANE = 221,
   DXIL_INTR_SAMPLE_CMP_LEVEL = 224,
};

enum dxil_atomic_op {
   DXIL_ATOMIC_ADD = 0,
   DXIL_ATOMIC_AND = 1,
   DXIL_ATOMIC_OR = 2,
   DXIL_ATOMIC_XOR = 3,
   DXIL_ATOMIC_IMIN = 4,
   DXIL_ATOMIC_IMAX = 5,
   DXIL_ATOMIC_UMIN = 6,
   DXIL_ATOMIC_UMAX = 7,
   DXIL_ATOMIC_EXCHANGE = 8,
};

typedef struct {
   unsigned id;
   unsigned binding;
   unsigned size;
   unsigned space;
} resource_array_layout;

static void
fill_resource_metadata(struct dxil_module *m, const struct dxil_mdnode **fields,
                       const struct dxil_type *struct_type,
                       const char *name, const resource_array_layout *layout)
{
   const struct dxil_type *pointer_type = dxil_module_get_pointer_type(m, struct_type);
   const struct dxil_value *pointer_undef = dxil_module_get_undef(m, pointer_type);

   fields[0] = dxil_get_metadata_int32(m, layout->id); // resource ID
   fields[1] = dxil_get_metadata_value(m, pointer_type, pointer_undef); // global constant symbol
   fields[2] = dxil_get_metadata_string(m, name ? name : ""); // name
   fields[3] = dxil_get_metadata_int32(m, layout->space); // space ID
   fields[4] = dxil_get_metadata_int32(m, layout->binding); // lower bound
   fields[5] = dxil_get_metadata_int32(m, layout->size); // range size
}

static const struct dxil_mdnode *
emit_srv_metadata(struct dxil_module *m, const struct dxil_type *elem_type,
                  const char *name, const resource_array_layout *layout,
                  enum dxil_component_type comp_type,
                  enum dxil_resource_kind res_kind)
{
   const struct dxil_mdnode *fields[9];

   const struct dxil_mdnode *metadata_tag_nodes[2];

   fill_resource_metadata(m, fields, elem_type, name, layout);
   fields[6] = dxil_get_metadata_int32(m, res_kind); // resource shape
   fields[7] = dxil_get_metadata_int1(m, 0); // sample count
   if (res_kind != DXIL_RESOURCE_KIND_RAW_BUFFER &&
       res_kind != DXIL_RESOURCE_KIND_STRUCTURED_BUFFER) {
      metadata_tag_nodes[0] = dxil_get_metadata_int32(m, DXIL_TYPED_BUFFER_ELEMENT_TYPE_TAG);
      metadata_tag_nodes[1] = dxil_get_metadata_int32(m, comp_type);
      fields[8] = dxil_get_metadata_node(m, metadata_tag_nodes, ARRAY_SIZE(metadata_tag_nodes)); // metadata
   } else if (res_kind == DXIL_RESOURCE_KIND_RAW_BUFFER)
      fields[8] = NULL;
   else
      unreachable("Structured buffers not supported yet");

   return dxil_get_metadata_node(m, fields, ARRAY_SIZE(fields));
}

static const struct dxil_mdnode *
emit_uav_metadata(struct dxil_module *m, const struct dxil_type *struct_type,
                  const char *name, const resource_array_layout *layout,
                  enum dxil_component_type comp_type,
                  enum dxil_resource_kind res_kind)
{
   const struct dxil_mdnode *fields[11];

   const struct dxil_mdnode *metadata_tag_nodes[2];

   fill_resource_metadata(m, fields, struct_type, name, layout);
   fields[6] = dxil_get_metadata_int32(m, res_kind); // resource shape
   fields[7] = dxil_get_metadata_int1(m, false); // globally-coherent
   fields[8] = dxil_get_metadata_int1(m, false); // has counter
   fields[9] = dxil_get_metadata_int1(m, false); // is ROV
   if (res_kind != DXIL_RESOURCE_KIND_RAW_BUFFER &&
       res_kind != DXIL_RESOURCE_KIND_STRUCTURED_BUFFER) {
      metadata_tag_nodes[0] = dxil_get_metadata_int32(m, DXIL_TYPED_BUFFER_ELEMENT_TYPE_TAG);
      metadata_tag_nodes[1] = dxil_get_metadata_int32(m, comp_type);
      fields[10] = dxil_get_metadata_node(m, metadata_tag_nodes, ARRAY_SIZE(metadata_tag_nodes)); // metadata
   } else if (res_kind == DXIL_RESOURCE_KIND_RAW_BUFFER)
      fields[10] = NULL;
   else
      unreachable("Structured buffers not supported yet");

   return dxil_get_metadata_node(m, fields, ARRAY_SIZE(fields));
}

static const struct dxil_mdnode *
emit_cbv_metadata(struct dxil_module *m, const struct dxil_type *struct_type,
                  const char *name, const resource_array_layout *layout,
                  unsigned size)
{
   const struct dxil_mdnode *fields[8];

   fill_resource_metadata(m, fields, struct_type, name, layout);
   fields[6] = dxil_get_metadata_int32(m, size); // constant buffer size
   fields[7] = NULL; // metadata

   return dxil_get_metadata_node(m, fields, ARRAY_SIZE(fields));
}

static const struct dxil_mdnode *
emit_sampler_metadata(struct dxil_module *m, const struct dxil_type *struct_type,
                      nir_variable *var, const resource_array_layout *layout)
{
   const struct dxil_mdnode *fields[8];
   const struct glsl_type *type = glsl_without_array(var->type);

   fill_resource_metadata(m, fields, struct_type, var->name, layout);
   enum dxil_sampler_kind sampler_kind = glsl_sampler_type_is_shadow(type) ?
          DXIL_SAMPLER_KIND_COMPARISON : DXIL_SAMPLER_KIND_DEFAULT;
   fields[6] = dxil_get_metadata_int32(m, sampler_kind); // sampler kind
   fields[7] = NULL; // metadata

   return dxil_get_metadata_node(m, fields, ARRAY_SIZE(fields));
}


#define MAX_SRVS 128
#define MAX_UAVS 64
#define MAX_CBVS 64 // ??
#define MAX_SAMPLERS 64 // ??

struct dxil_def {
   const struct dxil_value *chans[NIR_MAX_VEC_COMPONENTS];
};

struct ntd_context {
   void *ralloc_ctx;
   const struct nir_to_dxil_options *opts;
   struct nir_shader *shader;

   struct dxil_module mod;

   struct util_dynarray srv_metadata_nodes;
   const struct dxil_value *srv_handles[MAX_SRVS];

   struct util_dynarray uav_metadata_nodes;
   const struct dxil_value *ssbo_handles[MAX_UAVS];
   const struct dxil_value *image_handles[MAX_UAVS];
   uint32_t num_uavs;

   struct util_dynarray cbv_metadata_nodes;
   const struct dxil_value *cbv_handles[MAX_CBVS];

   struct util_dynarray sampler_metadata_nodes;
   const struct dxil_value *sampler_handles[MAX_SAMPLERS];

   struct util_dynarray resources;

   const struct dxil_mdnode *shader_property_nodes[6];
   size_t num_shader_property_nodes;

   struct dxil_def *defs;
   unsigned num_defs;
   struct hash_table *phis;

   const struct dxil_value *sharedvars;
   const struct dxil_value *scratchvars;
   struct hash_table *consts;

   nir_variable *ps_front_face;
   nir_variable *system_value[SYSTEM_VALUE_MAX];

   nir_function *tess_ctrl_patch_constant_func;
   unsigned tess_input_control_point_count;

   struct dxil_func_def *main_func_def;
   struct dxil_func_def *tess_ctrl_patch_constant_func_def;
   unsigned unnamed_ubo_count;

   const struct dxil_logger *logger;
};

static const char*
unary_func_name(enum dxil_intr intr)
{
   switch (intr) {
   case DXIL_INTR_COUNTBITS:
   case DXIL_INTR_FIRSTBIT_HI:
   case DXIL_INTR_FIRSTBIT_SHI:
   case DXIL_INTR_FIRSTBIT_LO:
      return "dx.op.unaryBits";
   case DXIL_INTR_ISFINITE:
   case DXIL_INTR_ISNORMAL:
      return "dx.op.isSpecialFloat";
   default:
      return "dx.op.unary";
   }
}

static const struct dxil_value *
emit_unary_call(struct ntd_context *ctx, enum overload_type overload,
                enum dxil_intr intr,
                const struct dxil_value *op0)
{
   const struct dxil_func *func = dxil_get_function(&ctx->mod,
                                                    unary_func_name(intr),
                                                    overload);
   if (!func)
      return NULL;

   const struct dxil_value *opcode = dxil_module_get_int32_const(&ctx->mod, intr);
   if (!opcode)
      return NULL;

   const struct dxil_value *args[] = {
     opcode,
     op0
   };

   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static const struct dxil_value *
emit_binary_call(struct ntd_context *ctx, enum overload_type overload,
                 enum dxil_intr intr,
                 const struct dxil_value *op0, const struct dxil_value *op1)
{
   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.binary", overload);
   if (!func)
      return NULL;

   const struct dxil_value *opcode = dxil_module_get_int32_const(&ctx->mod, intr);
   if (!opcode)
      return NULL;

   const struct dxil_value *args[] = {
     opcode,
     op0,
     op1
   };

   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static const struct dxil_value *
emit_tertiary_call(struct ntd_context *ctx, enum overload_type overload,
                   enum dxil_intr intr,
                   const struct dxil_value *op0,
                   const struct dxil_value *op1,
                   const struct dxil_value *op2)
{
   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.tertiary", overload);
   if (!func)
      return NULL;

   const struct dxil_value *opcode = dxil_module_get_int32_const(&ctx->mod, intr);
   if (!opcode)
      return NULL;

   const struct dxil_value *args[] = {
     opcode,
     op0,
     op1,
     op2
   };

   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static const struct dxil_value *
emit_quaternary_call(struct ntd_context *ctx, enum overload_type overload,
                     enum dxil_intr intr,
                     const struct dxil_value *op0,
                     const struct dxil_value *op1,
                     const struct dxil_value *op2,
                     const struct dxil_value *op3)
{
   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.quaternary", overload);
   if (!func)
      return NULL;

   const struct dxil_value *opcode = dxil_module_get_int32_const(&ctx->mod, intr);
   if (!opcode)
      return NULL;

   const struct dxil_value *args[] = {
     opcode,
     op0,
     op1,
     op2,
     op3
   };

   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static const struct dxil_value *
emit_threadid_call(struct ntd_context *ctx, const struct dxil_value *comp)
{
   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.threadId", DXIL_I32);
   if (!func)
      return NULL;

   const struct dxil_value *opcode = dxil_module_get_int32_const(&ctx->mod,
       DXIL_INTR_THREAD_ID);
   if (!opcode)
      return NULL;

   const struct dxil_value *args[] = {
     opcode,
     comp
   };

   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static const struct dxil_value *
emit_threadidingroup_call(struct ntd_context *ctx,
                          const struct dxil_value *comp)
{
   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.threadIdInGroup", DXIL_I32);

   if (!func)
      return NULL;

   const struct dxil_value *opcode = dxil_module_get_int32_const(&ctx->mod,
       DXIL_INTR_THREAD_ID_IN_GROUP);
   if (!opcode)
      return NULL;

   const struct dxil_value *args[] = {
     opcode,
     comp
   };

   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static const struct dxil_value *
emit_flattenedthreadidingroup_call(struct ntd_context *ctx)
{
   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.flattenedThreadIdInGroup", DXIL_I32);

   if (!func)
      return NULL;

   const struct dxil_value *opcode = dxil_module_get_int32_const(&ctx->mod,
      DXIL_INTR_FLATTENED_THREAD_ID_IN_GROUP);
   if (!opcode)
      return NULL;

   const struct dxil_value *args[] = {
     opcode
   };

   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static const struct dxil_value *
emit_groupid_call(struct ntd_context *ctx, const struct dxil_value *comp)
{
   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.groupId", DXIL_I32);

   if (!func)
      return NULL;

   const struct dxil_value *opcode = dxil_module_get_int32_const(&ctx->mod,
       DXIL_INTR_GROUP_ID);
   if (!opcode)
      return NULL;

   const struct dxil_value *args[] = {
     opcode,
     comp
   };

   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static const struct dxil_value *
emit_bufferload_call(struct ntd_context *ctx,
                     const struct dxil_value *handle,
                     const struct dxil_value *coord[2],
                     enum overload_type overload)
{
   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.bufferLoad", overload);
   if (!func)
      return NULL;

   const struct dxil_value *opcode = dxil_module_get_int32_const(&ctx->mod,
      DXIL_INTR_BUFFER_LOAD);
   const struct dxil_value *args[] = { opcode, handle, coord[0], coord[1] };

   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static bool
emit_bufferstore_call(struct ntd_context *ctx,
                      const struct dxil_value *handle,
                      const struct dxil_value *coord[2],
                      const struct dxil_value *value[4],
                      const struct dxil_value *write_mask,
                      enum overload_type overload)
{
   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.bufferStore", overload);

   if (!func)
      return false;

   const struct dxil_value *opcode = dxil_module_get_int32_const(&ctx->mod,
      DXIL_INTR_BUFFER_STORE);
   const struct dxil_value *args[] = {
      opcode, handle, coord[0], coord[1],
      value[0], value[1], value[2], value[3],
      write_mask
   };

   return dxil_emit_call_void(&ctx->mod, func,
                              args, ARRAY_SIZE(args));
}

static const struct dxil_value *
emit_textureload_call(struct ntd_context *ctx,
                      const struct dxil_value *handle,
                      const struct dxil_value *coord[3],
                      enum overload_type overload)
{
   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.textureLoad", overload);
   if (!func)
      return NULL;
   const struct dxil_type *int_type = dxil_module_get_int_type(&ctx->mod, 32);
   const struct dxil_value *int_undef = dxil_module_get_undef(&ctx->mod, int_type);

   const struct dxil_value *opcode = dxil_module_get_int32_const(&ctx->mod,
      DXIL_INTR_TEXTURE_LOAD);
   const struct dxil_value *args[] = { opcode, handle,
      /*lod_or_sample*/ int_undef,
      coord[0], coord[1], coord[2],
      /* offsets */ int_undef, int_undef, int_undef};

   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static bool
emit_texturestore_call(struct ntd_context *ctx,
                       const struct dxil_value *handle,
                       const struct dxil_value *coord[3],
                       const struct dxil_value *value[4],
                       const struct dxil_value *write_mask,
                       enum overload_type overload)
{
   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.textureStore", overload);

   if (!func)
      return false;

   const struct dxil_value *opcode = dxil_module_get_int32_const(&ctx->mod,
      DXIL_INTR_TEXTURE_STORE);
   const struct dxil_value *args[] = {
      opcode, handle, coord[0], coord[1], coord[2],
      value[0], value[1], value[2], value[3],
      write_mask
   };

   return dxil_emit_call_void(&ctx->mod, func,
                              args, ARRAY_SIZE(args));
}

static const struct dxil_value *
emit_atomic_binop(struct ntd_context *ctx,
                  const struct dxil_value *handle,
                  enum dxil_atomic_op atomic_op,
                  const struct dxil_value *coord[3],
                  const struct dxil_value *value)
{
   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.atomicBinOp", DXIL_I32);

   if (!func)
      return false;

   const struct dxil_value *opcode =
      dxil_module_get_int32_const(&ctx->mod, DXIL_INTR_ATOMIC_BINOP);
   const struct dxil_value *atomic_op_value =
      dxil_module_get_int32_const(&ctx->mod, atomic_op);
   const struct dxil_value *args[] = {
      opcode, handle, atomic_op_value,
      coord[0], coord[1], coord[2], value
   };

   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static const struct dxil_value *
emit_atomic_cmpxchg(struct ntd_context *ctx,
                    const struct dxil_value *handle,
                    const struct dxil_value *coord[3],
                    const struct dxil_value *cmpval,
                    const struct dxil_value *newval)
{
   const struct dxil_func *func =
      dxil_get_function(&ctx->mod, "dx.op.atomicCompareExchange", DXIL_I32);

   if (!func)
      return false;

   const struct dxil_value *opcode =
      dxil_module_get_int32_const(&ctx->mod, DXIL_INTR_ATOMIC_CMPXCHG);
   const struct dxil_value *args[] = {
      opcode, handle, coord[0], coord[1], coord[2], cmpval, newval
   };

   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static const struct dxil_value *
emit_createhandle_call_pre_6_6(struct ntd_context *ctx,
                               enum dxil_resource_class resource_class,
                               unsigned lower_bound,
                               unsigned upper_bound,
                               unsigned space,
                               unsigned resource_range_id,
                               const struct dxil_value *resource_range_index,
                               bool non_uniform_resource_index)
{
   const struct dxil_value *opcode = dxil_module_get_int32_const(&ctx->mod, DXIL_INTR_CREATE_HANDLE);
   const struct dxil_value *resource_class_value = dxil_module_get_int8_const(&ctx->mod, resource_class);
   const struct dxil_value *resource_range_id_value = dxil_module_get_int32_const(&ctx->mod, resource_range_id);
   const struct dxil_value *non_uniform_resource_index_value = dxil_module_get_int1_const(&ctx->mod, non_uniform_resource_index);
   if (!opcode || !resource_class_value || !resource_range_id_value ||
       !non_uniform_resource_index_value)
      return NULL;

   const struct dxil_value *args[] = {
      opcode,
      resource_class_value,
      resource_range_id_value,
      resource_range_index,
      non_uniform_resource_index_value
   };

   const struct dxil_func *func =
         dxil_get_function(&ctx->mod, "dx.op.createHandle", DXIL_NONE);

   if (!func)
         return NULL;

   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static const struct dxil_value *
emit_annotate_handle(struct ntd_context *ctx,
                     enum dxil_resource_class resource_class,
                     unsigned resource_range_id,
                     const struct dxil_value *unannotated_handle)
{
   const struct dxil_value *opcode = dxil_module_get_int32_const(&ctx->mod, DXIL_INTR_ANNOTATE_HANDLE);
   if (!opcode)
      return NULL;

   const struct util_dynarray *mdnodes;
   switch (resource_class) {
   case DXIL_RESOURCE_CLASS_SRV:
      mdnodes = &ctx->srv_metadata_nodes;
      break;
   case DXIL_RESOURCE_CLASS_UAV:
      mdnodes = &ctx->uav_metadata_nodes;
      break;
   case DXIL_RESOURCE_CLASS_CBV:
      mdnodes = &ctx->cbv_metadata_nodes;
      break;
   case DXIL_RESOURCE_CLASS_SAMPLER:
      mdnodes = &ctx->sampler_metadata_nodes;
      break;
   default:
      unreachable("Invalid resource class");
   }

   const struct dxil_mdnode *mdnode = *util_dynarray_element(mdnodes, const struct dxil_mdnode *, resource_range_id);
   const struct dxil_value *res_props = dxil_module_get_res_props_const(&ctx->mod, resource_class, mdnode);
   if (!res_props)
      return NULL;

   const struct dxil_value *args[] = {
      opcode,
      unannotated_handle,
      res_props
   };

   const struct dxil_func *func =
      dxil_get_function(&ctx->mod, "dx.op.annotateHandle", DXIL_NONE);

   if (!func)
      return NULL;

   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static const struct dxil_value *
emit_createhandle_and_annotate(struct ntd_context *ctx,
                               enum dxil_resource_class resource_class,
                               unsigned lower_bound,
                               unsigned upper_bound,
                               unsigned space,
                               unsigned resource_range_id,
                               const struct dxil_value *resource_range_index,
                               bool non_uniform_resource_index)
{
   const struct dxil_value *opcode = dxil_module_get_int32_const(&ctx->mod, DXIL_INTR_CREATE_HANDLE_FROM_BINDING);
   const struct dxil_value *res_bind = dxil_module_get_res_bind_const(&ctx->mod, lower_bound, upper_bound, space, resource_class);
   const struct dxil_value *non_uniform_resource_index_value = dxil_module_get_int1_const(&ctx->mod, non_uniform_resource_index);
   if (!opcode || !res_bind || !non_uniform_resource_index_value)
      return NULL;

   const struct dxil_value *args[] = {
      opcode,
      res_bind,
      resource_range_index,
      non_uniform_resource_index_value
   };

   const struct dxil_func *func =
      dxil_get_function(&ctx->mod, "dx.op.createHandleFromBinding", DXIL_NONE);

   if (!func)
      return NULL;

   const struct dxil_value *unannotated_handle = dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
   if (!unannotated_handle)
      return NULL;

   return emit_annotate_handle(ctx, resource_class, resource_range_id, unannotated_handle);
}

static const struct dxil_value *
emit_createhandle_call(struct ntd_context *ctx,
                       enum dxil_resource_class resource_class,
                       unsigned lower_bound,
                       unsigned upper_bound,
                       unsigned space,
                       unsigned resource_range_id,
                       const struct dxil_value *resource_range_index,
                       bool non_uniform_resource_index)
{
   if (ctx->mod.minor_version < 6)
      return emit_createhandle_call_pre_6_6(ctx, resource_class, lower_bound, upper_bound, space, resource_range_id, resource_range_index, non_uniform_resource_index);
   else
      return emit_createhandle_and_annotate(ctx, resource_class, lower_bound, upper_bound, space, resource_range_id, resource_range_index, non_uniform_resource_index);
}

static const struct dxil_value *
emit_createhandle_call_const_index(struct ntd_context *ctx,
                                   enum dxil_resource_class resource_class,
                                   unsigned lower_bound,
                                   unsigned upper_bound,
                                   unsigned space,
                                   unsigned resource_range_id,
                                   unsigned resource_range_index,
                                   bool non_uniform_resource_index)
{

   const struct dxil_value *resource_range_index_value = dxil_module_get_int32_const(&ctx->mod, resource_range_index);
   if (!resource_range_index_value)
      return NULL;

   return emit_createhandle_call(ctx, resource_class, lower_bound, upper_bound, space,
                                 resource_range_id, resource_range_index_value,
                                 non_uniform_resource_index);
}

static void
add_resource(struct ntd_context *ctx, enum dxil_resource_type type,
             enum dxil_resource_kind kind,
             const resource_array_layout *layout)
{
   struct dxil_resource_v0 *resource_v0 = NULL;
   struct dxil_resource_v1 *resource_v1 = NULL;
   if (ctx->mod.minor_validator >= 6) {
      resource_v1 = util_dynarray_grow(&ctx->resources, struct dxil_resource_v1, 1);
      resource_v0 = &resource_v1->v0;
   } else {
      resource_v0 = util_dynarray_grow(&ctx->resources, struct dxil_resource_v0, 1);
   }
   resource_v0->resource_type = type;
   resource_v0->space = layout->space;
   resource_v0->lower_bound = layout->binding;
   if (layout->size == 0 || (uint64_t)layout->size + layout->binding >= UINT_MAX)
      resource_v0->upper_bound = UINT_MAX;
   else
      resource_v0->upper_bound = layout->binding + layout->size - 1;
   if (type == DXIL_RES_UAV_TYPED ||
       type == DXIL_RES_UAV_RAW ||
       type == DXIL_RES_UAV_STRUCTURED) {
      uint32_t new_uav_count = ctx->num_uavs + layout->size;
      if (layout->size == 0 || new_uav_count < ctx->num_uavs)
         ctx->num_uavs = UINT_MAX;
      else
         ctx->num_uavs = new_uav_count;
      if (ctx->mod.minor_validator >= 6 && ctx->num_uavs > 8)
         ctx->mod.feats.use_64uavs = 1;
   }

   if (resource_v1) {
      resource_v1->resource_kind = kind;
      /* No flags supported yet */
      resource_v1->resource_flags = 0;
   }

   ctx->opts->godot_nir_callbacks->report_resource(layout->binding, layout->space, (uint32_t)type, ctx->opts->godot_nir_callbacks->data);
}

static const struct dxil_value *
emit_createhandle_call_dynamic(struct ntd_context *ctx,
                               enum dxil_resource_class resource_class,
                               unsigned space,
                               unsigned binding,
                               const struct dxil_value *resource_range_index,
                               bool non_uniform_resource_index)
{
   unsigned offset = 0;
   unsigned count = 0;

   unsigned num_srvs = util_dynarray_num_elements(&ctx->srv_metadata_nodes, const struct dxil_mdnode *);
   unsigned num_uavs = util_dynarray_num_elements(&ctx->uav_metadata_nodes, const struct dxil_mdnode *);
   unsigned num_cbvs = util_dynarray_num_elements(&ctx->cbv_metadata_nodes, const struct dxil_mdnode *);
   unsigned num_samplers = util_dynarray_num_elements(&ctx->sampler_metadata_nodes, const struct dxil_mdnode *);

   switch (resource_class) {
   case DXIL_RESOURCE_CLASS_UAV:
      offset = num_srvs + num_samplers + num_cbvs;
      count = num_uavs;
      break;
   case DXIL_RESOURCE_CLASS_SRV:
      offset = num_samplers + num_cbvs;
      count = num_srvs;
      break;
   case DXIL_RESOURCE_CLASS_SAMPLER:
      offset = num_cbvs;
      count = num_samplers;
      break;
   case DXIL_RESOURCE_CLASS_CBV:
      offset = 0;
      count = num_cbvs;
      break;
   }

   unsigned resource_element_size = ctx->mod.minor_validator >= 6 ?
      sizeof(struct dxil_resource_v1) : sizeof(struct dxil_resource_v0);
   assert(offset + count <= ctx->resources.size / resource_element_size);
   for (unsigned i = offset; i < offset + count; ++i) {
      const struct dxil_resource_v0 *resource = (const struct dxil_resource_v0 *)((const char *)ctx->resources.data + resource_element_size * i);
      if (resource->space == space &&
          resource->lower_bound <= binding &&
          resource->upper_bound >= binding) {
         return emit_createhandle_call(ctx, resource_class, resource->lower_bound,
                                       resource->upper_bound, space,
                                       i - offset,
                                       resource_range_index,
                                       non_uniform_resource_index);
      }
   }

   unreachable("Resource access for undeclared range");
}

static bool
emit_srv(struct ntd_context *ctx, nir_variable *var, unsigned count)
{
   unsigned id = util_dynarray_num_elements(&ctx->srv_metadata_nodes, const struct dxil_mdnode *);
   unsigned binding = var->data.binding;
   resource_array_layout layout = {id, binding, count, var->data.descriptor_set};

   enum dxil_component_type comp_type;
   enum dxil_resource_kind res_kind;
   enum dxil_resource_type res_type;
   if (var->data.mode == nir_var_mem_ssbo) {
      comp_type = DXIL_COMP_TYPE_INVALID;
      res_kind = DXIL_RESOURCE_KIND_RAW_BUFFER;
      res_type = DXIL_RES_SRV_RAW;
   } else {
      comp_type = dxil_get_comp_type(var->type);
      res_kind = dxil_get_resource_kind(var->type);
      res_type = DXIL_RES_SRV_TYPED;
   }
   const struct dxil_type *res_type_as_type = dxil_module_get_res_type(&ctx->mod, res_kind, comp_type, false /* readwrite */);

   if (glsl_type_is_array(var->type))
      res_type_as_type = dxil_module_get_array_type(&ctx->mod, res_type_as_type, count);

   const struct dxil_mdnode *srv_meta = emit_srv_metadata(&ctx->mod, res_type_as_type, var->name,
                                                          &layout, comp_type, res_kind);

   if (!srv_meta)
      return false;

   util_dynarray_append(&ctx->srv_metadata_nodes, const struct dxil_mdnode *, srv_meta);
   add_resource(ctx, res_type, res_kind, &layout);
   if (res_type == DXIL_RES_SRV_RAW)
      ctx->mod.raw_and_structured_buffers = true;

   return true;
}

static bool
emit_globals(struct ntd_context *ctx, unsigned size)
{
   nir_foreach_variable_with_modes(var, ctx->shader, nir_var_mem_ssbo)
      size++;

   if (!size)
      return true;

   const struct dxil_type *struct_type = dxil_module_get_res_type(&ctx->mod,
      DXIL_RESOURCE_KIND_RAW_BUFFER, DXIL_COMP_TYPE_INVALID, true /* readwrite */);
   if (!struct_type)
      return false;

   const struct dxil_type *array_type =
      dxil_module_get_array_type(&ctx->mod, struct_type, size);
   if (!array_type)
      return false;

   resource_array_layout layout = {0, 0, size, 0};
   const struct dxil_mdnode *uav_meta =
      emit_uav_metadata(&ctx->mod, array_type,
                                   "globals", &layout,
                                   DXIL_COMP_TYPE_INVALID,
                                   DXIL_RESOURCE_KIND_RAW_BUFFER);
   if (!uav_meta)
      return false;

   util_dynarray_append(&ctx->uav_metadata_nodes, const struct dxil_mdnode *, uav_meta);
   if (ctx->mod.minor_validator < 6 &&
       util_dynarray_num_elements(&ctx->uav_metadata_nodes, const struct dxil_mdnode *) > 8)
      ctx->mod.feats.use_64uavs = 1;
   /* Handles to UAVs used for kernel globals are created on-demand */
   add_resource(ctx, DXIL_RES_UAV_RAW, DXIL_RESOURCE_KIND_RAW_BUFFER, &layout);
   ctx->mod.raw_and_structured_buffers = true;
   return true;
}

static bool
emit_uav(struct ntd_context *ctx, unsigned binding, unsigned space, unsigned count,
         enum dxil_component_type comp_type, enum dxil_resource_kind res_kind, const char *name)
{
   unsigned id = util_dynarray_num_elements(&ctx->uav_metadata_nodes, const struct dxil_mdnode *);
   resource_array_layout layout = { id, binding, count, space };

   const struct dxil_type *res_type = dxil_module_get_res_type(&ctx->mod, res_kind, comp_type, true /* readwrite */);
   res_type = dxil_module_get_array_type(&ctx->mod, res_type, count);
   const struct dxil_mdnode *uav_meta = emit_uav_metadata(&ctx->mod, res_type, name,
                                                          &layout, comp_type, res_kind);

   if (!uav_meta)
      return false;

   util_dynarray_append(&ctx->uav_metadata_nodes, const struct dxil_mdnode *, uav_meta);
   if (ctx->mod.minor_validator < 6 &&
       util_dynarray_num_elements(&ctx->uav_metadata_nodes, const struct dxil_mdnode *) > 8)
      ctx->mod.feats.use_64uavs = 1;

   add_resource(ctx, res_kind == DXIL_RESOURCE_KIND_RAW_BUFFER ? DXIL_RES_UAV_RAW : DXIL_RES_UAV_TYPED, res_kind, &layout);
   if (res_kind == DXIL_RESOURCE_KIND_RAW_BUFFER)
      ctx->mod.raw_and_structured_buffers = true;
   if (ctx->mod.shader_kind != DXIL_PIXEL_SHADER &&
       ctx->mod.shader_kind != DXIL_COMPUTE_SHADER)
      ctx->mod.feats.uavs_at_every_stage = true;

   return true;
}

static bool
emit_uav_var(struct ntd_context *ctx, nir_variable *var, unsigned count)
{
   unsigned binding, space;
   if (ctx->opts->environment == DXIL_ENVIRONMENT_GL) {
      /* For GL, the image intrinsics are already lowered, using driver_location
       * as the 0-based image index. Use space 1 so that we can keep using these
       * NIR constants without having to remap them, and so they don't overlap
       * SSBOs, which are also 0-based UAV bindings.
       */
      binding = var->data.driver_location;
      space = 1;
   } else {
      binding = var->data.binding;
      space = var->data.descriptor_set;
   }
   enum dxil_component_type comp_type = dxil_get_comp_type(var->type);
   enum dxil_resource_kind res_kind = dxil_get_resource_kind(var->type);
   const char *name = var->name;

   return emit_uav(ctx, binding, space, count, comp_type, res_kind, name);
}

static void
var_fill_const_array_with_vector_or_scalar(struct ntd_context *ctx,
                                           const struct nir_constant *c,
                                           const struct glsl_type *type,
                                           void *const_vals,
                                           unsigned int offset)
{
   assert(glsl_type_is_vector_or_scalar(type));
   unsigned int components = glsl_get_vector_elements(type);
   unsigned bit_size = glsl_get_bit_size(type);
   unsigned int increment = bit_size / 8;

   for (unsigned int comp = 0; comp < components; comp++) {
      uint8_t *dst = (uint8_t *)const_vals + offset;

      switch (bit_size) {
      case 64:
         memcpy(dst, &c->values[comp].u64, sizeof(c->values[0].u64));
         break;
      case 32:
         memcpy(dst, &c->values[comp].u32, sizeof(c->values[0].u32));
         break;
      case 16:
         memcpy(dst, &c->values[comp].u16, sizeof(c->values[0].u16));
         break;
      case 8:
         assert(glsl_base_type_is_integer(glsl_get_base_type(type)));
         memcpy(dst, &c->values[comp].u8, sizeof(c->values[0].u8));
         break;
      default:
         unreachable("unexpeted bit-size");
      }

      offset += increment;
   }
}

static void
var_fill_const_array(struct ntd_context *ctx, const struct nir_constant *c,
                     const struct glsl_type *type, void *const_vals,
                     unsigned int offset)
{
   assert(!glsl_type_is_interface(type));

   if (glsl_type_is_vector_or_scalar(type)) {
      var_fill_const_array_with_vector_or_scalar(ctx, c, type,
                                                 const_vals,
                                                 offset);
   } else if (glsl_type_is_array(type)) {
      assert(!glsl_type_is_unsized_array(type));
      const struct glsl_type *without = glsl_get_array_element(type);
      unsigned stride = glsl_get_explicit_stride(type);

      for (unsigned elt = 0; elt < glsl_get_length(type); elt++) {
         var_fill_const_array(ctx, c->elements[elt], without,
                              const_vals, offset);
         offset += stride;
      }
   } else if (glsl_type_is_struct(type)) {
      for (unsigned int elt = 0; elt < glsl_get_length(type); elt++) {
         const struct glsl_type *elt_type = glsl_get_struct_field(type, elt);
         unsigned field_offset = glsl_get_struct_field_offset(type, elt);

         var_fill_const_array(ctx, c->elements[elt],
                              elt_type, const_vals,
                              offset + field_offset);
      }
   } else
      unreachable("unknown GLSL type in var_fill_const_array");
}

static bool
emit_global_consts(struct ntd_context *ctx)
{
   nir_foreach_variable_with_modes(var, ctx->shader, nir_var_shader_temp) {
      assert(var->constant_initializer);

      unsigned int num_members = DIV_ROUND_UP(glsl_get_cl_size(var->type), 4);
      uint32_t *const_ints = ralloc_array(ctx->ralloc_ctx, uint32_t, num_members);
      var_fill_const_array(ctx, var->constant_initializer, var->type,
                                 const_ints, 0);
      const struct dxil_value **const_vals =
         ralloc_array(ctx->ralloc_ctx, const struct dxil_value *, num_members);
      if (!const_vals)
         return false;
      for (int i = 0; i < num_members; i++)
         const_vals[i] = dxil_module_get_int32_const(&ctx->mod, const_ints[i]);

      const struct dxil_type *elt_type = dxil_module_get_int_type(&ctx->mod, 32);
      if (!elt_type)
         return false;
      const struct dxil_type *type =
         dxil_module_get_array_type(&ctx->mod, elt_type, num_members);
      if (!type)
         return false;
      const struct dxil_value *agg_vals =
         dxil_module_get_array_const(&ctx->mod, type, const_vals);
      if (!agg_vals)
         return false;

      const struct dxil_value *gvar = dxil_add_global_ptr_var(&ctx->mod, var->name, type,
                                                              DXIL_AS_DEFAULT, 4,
                                                              agg_vals);
      if (!gvar)
         return false;

      if (!_mesa_hash_table_insert(ctx->consts, var, (void *)gvar))
         return false;
   }

   return true;
}

static bool
emit_cbv(struct ntd_context *ctx, unsigned binding, unsigned space,
         unsigned size, unsigned count, char *name)
{
   assert(count != 0);

   unsigned idx = util_dynarray_num_elements(&ctx->cbv_metadata_nodes, const struct dxil_mdnode *);

   const struct dxil_type *float32 = dxil_module_get_float_type(&ctx->mod, 32);
   const struct dxil_type *array_type = dxil_module_get_array_type(&ctx->mod, float32, size);
   const struct dxil_type *buffer_type = dxil_module_get_struct_type(&ctx->mod, name,
                                                                     &array_type, 1);
   // All ubo[1]s should have been lowered to ubo with static indexing
   const struct dxil_type *final_type = count != 1 ? dxil_module_get_array_type(&ctx->mod, buffer_type, count) : buffer_type;
   resource_array_layout layout = {idx, binding, count, space};
   const struct dxil_mdnode *cbv_meta = emit_cbv_metadata(&ctx->mod, final_type,
                                                          name, &layout, 4 * size);

   if (!cbv_meta)
      return false;

   util_dynarray_append(&ctx->cbv_metadata_nodes, const struct dxil_mdnode *, cbv_meta);
   add_resource(ctx, DXIL_RES_CBV, DXIL_RESOURCE_KIND_CBUFFER, &layout);

   return true;
}

static bool
emit_ubo_var(struct ntd_context *ctx, nir_variable *var)
{
   unsigned count = 1;
   if (glsl_type_is_array(var->type))
      count = glsl_get_length(var->type);

   char *name = var->name;
   char temp_name[30];
   if (name && strlen(name) == 0) {
      snprintf(temp_name, sizeof(temp_name), "__unnamed_ubo_%d",
               ctx->unnamed_ubo_count++);
      name = temp_name;
   }

   const struct glsl_type *type = glsl_without_array(var->type);
   assert(glsl_type_is_struct(type) || glsl_type_is_interface(type));
   unsigned dwords = ALIGN_POT(glsl_get_explicit_size(type, false), 16) / 4;

   return emit_cbv(ctx, var->data.binding, var->data.descriptor_set,
                   dwords, count, name);
}

static bool
emit_sampler(struct ntd_context *ctx, nir_variable *var, unsigned count)
{
   unsigned id = util_dynarray_num_elements(&ctx->sampler_metadata_nodes, const struct dxil_mdnode *);
   unsigned binding = var->data.binding;
   resource_array_layout layout = {id, binding, count, var->data.descriptor_set};
   const struct dxil_type *int32_type = dxil_module_get_int_type(&ctx->mod, 32);
   const struct dxil_type *sampler_type = dxil_module_get_struct_type(&ctx->mod, "struct.SamplerState", &int32_type, 1);

   if (glsl_type_is_array(var->type))
      sampler_type = dxil_module_get_array_type(&ctx->mod, sampler_type, count);

   const struct dxil_mdnode *sampler_meta = emit_sampler_metadata(&ctx->mod, sampler_type, var, &layout);

   if (!sampler_meta)
      return false;

   util_dynarray_append(&ctx->sampler_metadata_nodes, const struct dxil_mdnode *, sampler_meta);
   add_resource(ctx, DXIL_RES_SAMPLER, DXIL_RESOURCE_KIND_SAMPLER, &layout);

   return true;
}

static bool
emit_static_indexing_handles(struct ntd_context *ctx)
{
   /* Vulkan always uses dynamic handles, from instructions in the NIR */
   if (ctx->opts->environment == DXIL_ENVIRONMENT_VULKAN)
      return true;

   unsigned last_res_class = -1;
   unsigned id = 0;

   unsigned resource_element_size = ctx->mod.minor_validator >= 6 ?
      sizeof(struct dxil_resource_v1) : sizeof(struct dxil_resource_v0);
   for (struct dxil_resource_v0 *res = (struct dxil_resource_v0 *)ctx->resources.data;
        res < (struct dxil_resource_v0 *)((char *)ctx->resources.data + ctx->resources.size);
        res = (struct dxil_resource_v0 *)((char *)res + resource_element_size)) {
      enum dxil_resource_class res_class;
      const struct dxil_value **handle_array;
      switch (res->resource_type) {
      case DXIL_RES_SRV_TYPED:
      case DXIL_RES_SRV_RAW:
      case DXIL_RES_SRV_STRUCTURED:
         res_class = DXIL_RESOURCE_CLASS_SRV;
         handle_array = ctx->srv_handles;
         break;
      case DXIL_RES_CBV:
         res_class = DXIL_RESOURCE_CLASS_CBV;
         handle_array = ctx->cbv_handles;
         break;
      case DXIL_RES_SAMPLER:
         res_class = DXIL_RESOURCE_CLASS_SAMPLER;
         handle_array = ctx->sampler_handles;
         break;
      case DXIL_RES_UAV_RAW:
         res_class = DXIL_RESOURCE_CLASS_UAV;
         handle_array = ctx->ssbo_handles;
         break;
      case DXIL_RES_UAV_TYPED:
      case DXIL_RES_UAV_STRUCTURED:
      case DXIL_RES_UAV_STRUCTURED_WITH_COUNTER:
         res_class = DXIL_RESOURCE_CLASS_UAV;
         handle_array = ctx->image_handles;
         break;
      default:
         unreachable("Unexpected resource type");
      }

      if (last_res_class != res_class)
         id = 0;
      else
         id++;
      last_res_class = res_class;

      if (res->space > 1)
         continue;
      assert(res->space == 0 ||
         (res->space == 1 &&
            res->resource_type != DXIL_RES_UAV_RAW &&
            ctx->opts->environment == DXIL_ENVIRONMENT_GL));

      /* CL uses dynamic handles for the "globals" UAV array, but uses static
       * handles for UBOs, textures, and samplers.
       */
      if (ctx->opts->environment == DXIL_ENVIRONMENT_CL &&
          res->resource_type == DXIL_RES_UAV_RAW)
         continue;

      for (unsigned i = res->lower_bound; i <= res->upper_bound; ++i) {
         handle_array[i] = emit_createhandle_call_const_index(ctx,
                                                              res_class,
                                                              res->lower_bound,
                                                              res->upper_bound,
                                                              res->space,
                                                              id,
                                                              i,
                                                              false);
         if (!handle_array[i])
            return false;
      }
   }
   return true;
}

static const struct dxil_mdnode *
emit_gs_state(struct ntd_context *ctx)
{
   const struct dxil_mdnode *gs_state_nodes[5];
   const nir_shader *s = ctx->shader;

   gs_state_nodes[0] = dxil_get_metadata_int32(&ctx->mod, dxil_get_input_primitive(s->info.gs.input_primitive));
   gs_state_nodes[1] = dxil_get_metadata_int32(&ctx->mod, s->info.gs.vertices_out);
   gs_state_nodes[2] = dxil_get_metadata_int32(&ctx->mod, MAX2(s->info.gs.active_stream_mask, 1));
   gs_state_nodes[3] = dxil_get_metadata_int32(&ctx->mod, dxil_get_primitive_topology(s->info.gs.output_primitive));
   gs_state_nodes[4] = dxil_get_metadata_int32(&ctx->mod, s->info.gs.invocations);

   for (unsigned i = 0; i < ARRAY_SIZE(gs_state_nodes); ++i) {
      if (!gs_state_nodes[i])
         return NULL;
   }

   return dxil_get_metadata_node(&ctx->mod, gs_state_nodes, ARRAY_SIZE(gs_state_nodes));
}

static enum dxil_tessellator_domain
get_tessellator_domain(enum tess_primitive_mode primitive_mode)
{
   switch (primitive_mode) {
   case TESS_PRIMITIVE_QUADS: return DXIL_TESSELLATOR_DOMAIN_QUAD;
   case TESS_PRIMITIVE_TRIANGLES: return DXIL_TESSELLATOR_DOMAIN_TRI;
   case TESS_PRIMITIVE_ISOLINES: return DXIL_TESSELLATOR_DOMAIN_ISOLINE;
   default:
      unreachable("Invalid tessellator primitive mode");
   }
}

static enum dxil_tessellator_partitioning
get_tessellator_partitioning(enum gl_tess_spacing spacing)
{
   switch (spacing) {
   default:
   case TESS_SPACING_EQUAL:
      return DXIL_TESSELLATOR_PARTITIONING_INTEGER;
   case TESS_SPACING_FRACTIONAL_EVEN:
      return DXIL_TESSELLATOR_PARTITIONING_FRACTIONAL_EVEN;
   case TESS_SPACING_FRACTIONAL_ODD:
      return DXIL_TESSELLATOR_PARTITIONING_FRACTIONAL_ODD;
   }
}

static enum dxil_tessellator_output_primitive
get_tessellator_output_primitive(const struct shader_info *info)
{
   if (info->tess.point_mode)
      return DXIL_TESSELLATOR_OUTPUT_PRIMITIVE_POINT;
   if (info->tess._primitive_mode == TESS_PRIMITIVE_ISOLINES)
      return DXIL_TESSELLATOR_OUTPUT_PRIMITIVE_LINE;
   /* Note: GL tessellation domain is inverted from D3D, which means triangle
    * winding needs to be inverted.
    */
   if (info->tess.ccw)
      return DXIL_TESSELLATOR_OUTPUT_PRIMITIVE_TRIANGLE_CW;
   return DXIL_TESSELLATOR_OUTPUT_PRIMITIVE_TRIANGLE_CCW;
}

static const struct dxil_mdnode *
emit_hs_state(struct ntd_context *ctx)
{
   const struct dxil_mdnode *hs_state_nodes[7];

   hs_state_nodes[0] = dxil_get_metadata_func(&ctx->mod, ctx->tess_ctrl_patch_constant_func_def->func);
   hs_state_nodes[1] = dxil_get_metadata_int32(&ctx->mod, ctx->tess_input_control_point_count);
   hs_state_nodes[2] = dxil_get_metadata_int32(&ctx->mod, ctx->shader->info.tess.tcs_vertices_out);
   hs_state_nodes[3] = dxil_get_metadata_int32(&ctx->mod, get_tessellator_domain(ctx->shader->info.tess._primitive_mode));
   hs_state_nodes[4] = dxil_get_metadata_int32(&ctx->mod, get_tessellator_partitioning(ctx->shader->info.tess.spacing));
   hs_state_nodes[5] = dxil_get_metadata_int32(&ctx->mod, get_tessellator_output_primitive(&ctx->shader->info));
   hs_state_nodes[6] = dxil_get_metadata_float32(&ctx->mod, 64.0f);

   return dxil_get_metadata_node(&ctx->mod, hs_state_nodes, ARRAY_SIZE(hs_state_nodes));
}

static const struct dxil_mdnode *
emit_ds_state(struct ntd_context *ctx)
{
   const struct dxil_mdnode *ds_state_nodes[2];

   ds_state_nodes[0] = dxil_get_metadata_int32(&ctx->mod, get_tessellator_domain(ctx->shader->info.tess._primitive_mode));
   ds_state_nodes[1] = dxil_get_metadata_int32(&ctx->mod, ctx->shader->info.tess.tcs_vertices_out);

   return dxil_get_metadata_node(&ctx->mod, ds_state_nodes, ARRAY_SIZE(ds_state_nodes));
}

static const struct dxil_mdnode *
emit_threads(struct ntd_context *ctx)
{
   const nir_shader *s = ctx->shader;
   const struct dxil_mdnode *threads_x = dxil_get_metadata_int32(&ctx->mod, MAX2(s->info.workgroup_size[0], 1));
   const struct dxil_mdnode *threads_y = dxil_get_metadata_int32(&ctx->mod, MAX2(s->info.workgroup_size[1], 1));
   const struct dxil_mdnode *threads_z = dxil_get_metadata_int32(&ctx->mod, MAX2(s->info.workgroup_size[2], 1));
   if (!threads_x || !threads_y || !threads_z)
      return false;

   const struct dxil_mdnode *threads_nodes[] = { threads_x, threads_y, threads_z };
   return dxil_get_metadata_node(&ctx->mod, threads_nodes, ARRAY_SIZE(threads_nodes));
}

static int64_t
get_module_flags(struct ntd_context *ctx)
{
   /* See the DXIL documentation for the definition of these flags:
    *
    * https://github.com/Microsoft/DirectXShaderCompiler/blob/master/docs/DXIL.rst#shader-flags
    */

   uint64_t flags = 0;
   if (ctx->mod.feats.doubles)
      flags |= (1 << 2);
   if (ctx->shader->info.stage == MESA_SHADER_FRAGMENT &&
       ctx->shader->info.fs.early_fragment_tests)
      flags |= (1 << 3);
   if (ctx->mod.raw_and_structured_buffers)
      flags |= (1 << 4);
   if (ctx->mod.feats.min_precision)
      flags |= (1 << 5);
   if (ctx->mod.feats.dx11_1_double_extensions)
      flags |= (1 << 6);
   if (ctx->mod.feats.array_layer_from_vs_or_ds)
      flags |= (1 << 9);
   if (ctx->mod.feats.inner_coverage)
      flags |= (1 << 10);
   if (ctx->mod.feats.stencil_ref)
      flags |= (1 << 11);
   if (ctx->mod.feats.tiled_resources)
      flags |= (1 << 12);
   if (ctx->mod.feats.typed_uav_load_additional_formats)
      flags |= (1 << 13);
   if (ctx->mod.feats.use_64uavs)
      flags |= (1 << 15);
   if (ctx->mod.feats.uavs_at_every_stage)
      flags |= (1 << 16);
   if (ctx->mod.feats.cs_4x_raw_sb)
      flags |= (1 << 17);
   if (ctx->mod.feats.rovs)
      flags |= (1 << 18);
   if (ctx->mod.feats.wave_ops)
      flags |= (1 << 19);
   if (ctx->mod.feats.int64_ops)
      flags |= (1 << 20);
   if (ctx->mod.feats.view_id)
      flags |= (1 << 21);
   if (ctx->mod.feats.barycentrics)
      flags |= (1 << 22);
   if (ctx->mod.feats.native_low_precision)
      flags |= (1 << 23) | (1 << 5);
   if (ctx->mod.feats.shading_rate)
      flags |= (1 << 24);
   if (ctx->mod.feats.raytracing_tier_1_1)
      flags |= (1 << 25);
   if (ctx->mod.feats.sampler_feedback)
      flags |= (1 << 26);
   if (ctx->mod.feats.atomic_int64_typed)
      flags |= (1 << 27);
   if (ctx->mod.feats.atomic_int64_tgsm)
      flags |= (1 << 28);
   if (ctx->mod.feats.derivatives_in_mesh_or_amp)
      flags |= (1 << 29);
   if (ctx->mod.feats.resource_descriptor_heap_indexing)
      flags |= (1 << 30);
   if (ctx->mod.feats.sampler_descriptor_heap_indexing)
      flags |= (1 << 31);
   if (ctx->mod.feats.atomic_int64_heap_resource)
      flags |= (1ull << 32);
   if (ctx->mod.feats.advanced_texture_ops)
      flags |= (1ull << 34);
   if (ctx->mod.feats.writable_msaa)
      flags |= (1ull << 35);

   if (ctx->opts->disable_math_refactoring)
      flags |= (1 << 1);

   /* Work around https://github.com/microsoft/DirectXShaderCompiler/issues/4616
    * When targeting SM6.7 and with at least one UAV, if no other flags are present,
    * set the resources-may-not-alias flag, or else the DXIL validator may end up
    * with uninitialized memory which will fail validation, due to missing that flag.
    */
   if (flags == 0 && ctx->mod.minor_version >= 7 && ctx->num_uavs > 0)
      flags |= (1ull << 33);

   return flags;
}

static const struct dxil_mdnode *
emit_entrypoint(struct ntd_context *ctx,
                const struct dxil_func *func, const char *name,
                const struct dxil_mdnode *signatures,
                const struct dxil_mdnode *resources,
                const struct dxil_mdnode *shader_props)
{
   char truncated_name[254] = { 0 };
   strncpy(truncated_name, name, ARRAY_SIZE(truncated_name) - 1);

   const struct dxil_mdnode *func_md = dxil_get_metadata_func(&ctx->mod, func);
   const struct dxil_mdnode *name_md = dxil_get_metadata_string(&ctx->mod, truncated_name);
   const struct dxil_mdnode *nodes[] = {
      func_md,
      name_md,
      signatures,
      resources,
      shader_props
   };
   return dxil_get_metadata_node(&ctx->mod, nodes,
                                 ARRAY_SIZE(nodes));
}

static const struct dxil_mdnode *
emit_resources(struct ntd_context *ctx)
{
   bool emit_resources = false;
   const struct dxil_mdnode *resources_nodes[] = {
      NULL, NULL, NULL, NULL
   };

#define ARRAY_AND_SIZE(arr) arr.data, util_dynarray_num_elements(&arr, const struct dxil_mdnode *)

   if (ctx->srv_metadata_nodes.size) {
      resources_nodes[0] = dxil_get_metadata_node(&ctx->mod, ARRAY_AND_SIZE(ctx->srv_metadata_nodes));
      emit_resources = true;
   }

   if (ctx->uav_metadata_nodes.size) {
      resources_nodes[1] = dxil_get_metadata_node(&ctx->mod, ARRAY_AND_SIZE(ctx->uav_metadata_nodes));
      emit_resources = true;
   }

   if (ctx->cbv_metadata_nodes.size) {
      resources_nodes[2] = dxil_get_metadata_node(&ctx->mod, ARRAY_AND_SIZE(ctx->cbv_metadata_nodes));
      emit_resources = true;
   }

   if (ctx->sampler_metadata_nodes.size) {
      resources_nodes[3] = dxil_get_metadata_node(&ctx->mod, ARRAY_AND_SIZE(ctx->sampler_metadata_nodes));
      emit_resources = true;
   }

#undef ARRAY_AND_SIZE

   return emit_resources ?
      dxil_get_metadata_node(&ctx->mod, resources_nodes, ARRAY_SIZE(resources_nodes)): NULL;
}

static bool
emit_tag(struct ntd_context *ctx, enum dxil_shader_tag tag,
         const struct dxil_mdnode *value_node)
{
   const struct dxil_mdnode *tag_node = dxil_get_metadata_int32(&ctx->mod, tag);
   if (!tag_node || !value_node)
      return false;
   assert(ctx->num_shader_property_nodes <= ARRAY_SIZE(ctx->shader_property_nodes) - 2);
   ctx->shader_property_nodes[ctx->num_shader_property_nodes++] = tag_node;
   ctx->shader_property_nodes[ctx->num_shader_property_nodes++] = value_node;

   return true;
}

static bool
emit_metadata(struct ntd_context *ctx)
{
   /* DXIL versions are 1.x for shader model 6.x */
   assert(ctx->mod.major_version == 6);
   unsigned dxilMajor = 1;
   unsigned dxilMinor = ctx->mod.minor_version;
   unsigned valMajor = ctx->mod.major_validator;
   unsigned valMinor = ctx->mod.minor_validator;
   if (!emit_llvm_ident(&ctx->mod) ||
       !emit_named_version(&ctx->mod, "dx.version", dxilMajor, dxilMinor) ||
       !emit_named_version(&ctx->mod, "dx.valver", valMajor, valMinor) ||
       !emit_dx_shader_model(&ctx->mod))
      return false;

   const struct dxil_func_def *main_func_def = ctx->main_func_def;
   if (!main_func_def)
      return false;
   const struct dxil_func *main_func = main_func_def->func;

   const struct dxil_mdnode *resources_node = emit_resources(ctx);

   const struct dxil_mdnode *main_entrypoint = dxil_get_metadata_func(&ctx->mod, main_func);
   const struct dxil_mdnode *node27 = dxil_get_metadata_node(&ctx->mod, NULL, 0);

   const struct dxil_mdnode *node4 = dxil_get_metadata_int32(&ctx->mod, 0);
   const struct dxil_mdnode *nodes_4_27_27[] = {
      node4, node27, node27
   };
   const struct dxil_mdnode *node28 = dxil_get_metadata_node(&ctx->mod, nodes_4_27_27,
                                                      ARRAY_SIZE(nodes_4_27_27));

   const struct dxil_mdnode *node29 = dxil_get_metadata_node(&ctx->mod, &node28, 1);

   const struct dxil_mdnode *node3 = dxil_get_metadata_int32(&ctx->mod, 1);
   const struct dxil_mdnode *main_type_annotation_nodes[] = {
      node3, main_entrypoint, node29
   };
   const struct dxil_mdnode *main_type_annotation = dxil_get_metadata_node(&ctx->mod, main_type_annotation_nodes,
                                                                           ARRAY_SIZE(main_type_annotation_nodes));

   if (ctx->mod.shader_kind == DXIL_GEOMETRY_SHADER) {
      if (!emit_tag(ctx, DXIL_SHADER_TAG_GS_STATE, emit_gs_state(ctx)))
         return false;
   } else if (ctx->mod.shader_kind == DXIL_HULL_SHADER) {
      ctx->tess_input_control_point_count = 32;
      nir_foreach_variable_with_modes(var, ctx->shader, nir_var_shader_in) {
         if (nir_is_arrayed_io(var, MESA_SHADER_TESS_CTRL)) {
            ctx->tess_input_control_point_count = glsl_array_size(var->type);
            break;
         }
      }

      if (!emit_tag(ctx, DXIL_SHADER_TAG_HS_STATE, emit_hs_state(ctx)))
         return false;
   } else if (ctx->mod.shader_kind == DXIL_DOMAIN_SHADER) {
      if (!emit_tag(ctx, DXIL_SHADER_TAG_DS_STATE, emit_ds_state(ctx)))
         return false;
   } else if (ctx->mod.shader_kind == DXIL_COMPUTE_SHADER) {
      if (!emit_tag(ctx, DXIL_SHADER_TAG_NUM_THREADS, emit_threads(ctx)))
         return false;
   }

   uint64_t flags = get_module_flags(ctx);
   if (flags != 0) {
      if (!emit_tag(ctx, DXIL_SHADER_TAG_FLAGS, dxil_get_metadata_int64(&ctx->mod, flags)))
         return false;
   }
   const struct dxil_mdnode *shader_properties = NULL;
   if (ctx->num_shader_property_nodes > 0) {
      shader_properties = dxil_get_metadata_node(&ctx->mod, ctx->shader_property_nodes,
                                                 ctx->num_shader_property_nodes);
      if (!shader_properties)
         return false;
   }

   nir_function_impl *entry_func_impl = nir_shader_get_entrypoint(ctx->shader);
   const struct dxil_mdnode *dx_entry_point = emit_entrypoint(ctx, main_func,
       entry_func_impl->function->name, get_signatures(&ctx->mod), resources_node, shader_properties);
   if (!dx_entry_point)
      return false;

   if (resources_node) {
      const struct dxil_mdnode *dx_resources = resources_node;
      dxil_add_metadata_named_node(&ctx->mod, "dx.resources",
                                       &dx_resources, 1);
   }

   const struct dxil_mdnode *dx_type_annotations[] = { main_type_annotation };
   return dxil_add_metadata_named_node(&ctx->mod, "dx.typeAnnotations",
                                       dx_type_annotations,
                                       ARRAY_SIZE(dx_type_annotations)) &&
          dxil_add_metadata_named_node(&ctx->mod, "dx.entryPoints",
                                       &dx_entry_point, 1);
}

static const struct dxil_value *
bitcast_to_int(struct ntd_context *ctx, unsigned bit_size,
               const struct dxil_value *value)
{
   const struct dxil_type *type = dxil_module_get_int_type(&ctx->mod, bit_size);
   if (!type)
      return NULL;

   return dxil_emit_cast(&ctx->mod, DXIL_CAST_BITCAST, type, value);
}

static const struct dxil_value *
bitcast_to_float(struct ntd_context *ctx, unsigned bit_size,
                 const struct dxil_value *value)
{
   const struct dxil_type *type = dxil_module_get_float_type(&ctx->mod, bit_size);
   if (!type)
      return NULL;

   return dxil_emit_cast(&ctx->mod, DXIL_CAST_BITCAST, type, value);
}

static void
store_ssa_def(struct ntd_context *ctx, nir_ssa_def *ssa, unsigned chan,
              const struct dxil_value *value)
{
   assert(ssa->index < ctx->num_defs);
   assert(chan < ssa->num_components);
   /* We pre-defined the dest value because of a phi node, so bitcast while storing if the
    * base type differs */
   if (ctx->defs[ssa->index].chans[chan]) {
      const struct dxil_type *expect_type = dxil_value_get_type(ctx->defs[ssa->index].chans[chan]);
      const struct dxil_type *value_type = dxil_value_get_type(value);
      if (dxil_type_to_nir_type(expect_type) != dxil_type_to_nir_type(value_type))
         value = dxil_emit_cast(&ctx->mod, DXIL_CAST_BITCAST, expect_type, value);
   }
   ctx->defs[ssa->index].chans[chan] = value;
}

static void
store_dest_value(struct ntd_context *ctx, nir_dest *dest, unsigned chan,
                 const struct dxil_value *value)
{
   assert(dest->is_ssa);
   assert(value);
   store_ssa_def(ctx, &dest->ssa, chan, value);
}

static void
store_dest(struct ntd_context *ctx, nir_dest *dest, unsigned chan,
           const struct dxil_value *value, nir_alu_type type)
{
   switch (nir_alu_type_get_base_type(type)) {
   case nir_type_float:
      if (nir_dest_bit_size(*dest) == 64)
         ctx->mod.feats.doubles = true;
      store_dest_value(ctx, dest, chan, value);
      break;
   case nir_type_uint:
   case nir_type_int:
      if (nir_dest_bit_size(*dest) == 16)
         ctx->mod.feats.native_low_precision = true;
      if (nir_dest_bit_size(*dest) == 64)
         ctx->mod.feats.int64_ops = true;
      FALLTHROUGH;
   case nir_type_bool:
      store_dest_value(ctx, dest, chan, value);
      break;
   default:
      unreachable("unexpected nir_alu_type");
   }
}

static void
store_alu_dest(struct ntd_context *ctx, nir_alu_instr *alu, unsigned chan,
               const struct dxil_value *value)
{
   assert(!alu->dest.saturate);
   store_dest(ctx, &alu->dest.dest, chan, value,
              nir_op_infos[alu->op].output_type);
}

static const struct dxil_value *
get_src_ssa(struct ntd_context *ctx, const nir_ssa_def *ssa, unsigned chan)
{
   assert(ssa->index < ctx->num_defs);
   assert(chan < ssa->num_components);
   assert(ctx->defs[ssa->index].chans[chan]);
   return ctx->defs[ssa->index].chans[chan];
}

static const struct dxil_value *
get_src(struct ntd_context *ctx, nir_src *src, unsigned chan,
        nir_alu_type type)
{
   assert(src->is_ssa);
   const struct dxil_value *value = get_src_ssa(ctx, src->ssa, chan);

   const int bit_size = nir_src_bit_size(*src);

   switch (nir_alu_type_get_base_type(type)) {
   case nir_type_int:
   case nir_type_uint: {
      const struct dxil_type *expect_type =  dxil_module_get_int_type(&ctx->mod, bit_size);
      /* nohing to do */
      if (dxil_value_type_equal_to(value, expect_type)) {
         assert(bit_size != 64 || ctx->mod.feats.int64_ops);
         return value;
      }
      if (bit_size == 64) {
         assert(ctx->mod.feats.doubles);
         ctx->mod.feats.int64_ops = true;
      }
      assert(dxil_value_type_bitsize_equal_to(value, bit_size));
      return bitcast_to_int(ctx,  bit_size, value);
      }

   case nir_type_float:
      assert(nir_src_bit_size(*src) >= 16);
      if (dxil_value_type_equal_to(value, dxil_module_get_float_type(&ctx->mod, bit_size))) {
         assert(nir_src_bit_size(*src) != 64 || ctx->mod.feats.doubles);
         return value;
      }
      if (bit_size == 64) {
         assert(ctx->mod.feats.int64_ops);
         ctx->mod.feats.doubles = true;
      }
      assert(dxil_value_type_bitsize_equal_to(value, bit_size));
      return bitcast_to_float(ctx, bit_size, value);

   case nir_type_bool:
      if (!dxil_value_type_bitsize_equal_to(value, 1)) {
         return dxil_emit_cast(&ctx->mod, DXIL_CAST_TRUNC,
                               dxil_module_get_int_type(&ctx->mod, 1), value);
      }
      return value;

   default:
      unreachable("unexpected nir_alu_type");
   }
}

static const struct dxil_type *
get_alu_src_type(struct ntd_context *ctx, nir_alu_instr *alu, unsigned src)
{
   assert(!alu->src[src].abs);
   assert(!alu->src[src].negate);
   nir_ssa_def *ssa_src = alu->src[src].src.ssa;
   unsigned chan = alu->src[src].swizzle[0];
   const struct dxil_value *value = get_src_ssa(ctx, ssa_src, chan);
   return dxil_value_get_type(value);
}

static const struct dxil_value *
get_alu_src(struct ntd_context *ctx, nir_alu_instr *alu, unsigned src)
{
   assert(!alu->src[src].abs);
   assert(!alu->src[src].negate);

   unsigned chan = alu->src[src].swizzle[0];
   return get_src(ctx, &alu->src[src].src, chan,
                  nir_op_infos[alu->op].input_types[src]);
}

static bool
emit_binop(struct ntd_context *ctx, nir_alu_instr *alu,
           enum dxil_bin_opcode opcode,
           const struct dxil_value *op0, const struct dxil_value *op1)
{
   bool is_float_op = nir_alu_type_get_base_type(nir_op_infos[alu->op].output_type) == nir_type_float;

   enum dxil_opt_flags flags = 0;
   if (is_float_op && !alu->exact)
      flags |= DXIL_UNSAFE_ALGEBRA;

   const struct dxil_value *v = dxil_emit_binop(&ctx->mod, opcode, op0, op1, flags);
   if (!v)
      return false;
   store_alu_dest(ctx, alu, 0, v);
   return true;
}

static bool
emit_shift(struct ntd_context *ctx, nir_alu_instr *alu,
           enum dxil_bin_opcode opcode,
           const struct dxil_value *op0, const struct dxil_value *op1)
{
   unsigned op0_bit_size = nir_src_bit_size(alu->src[0].src);
   unsigned op1_bit_size = nir_src_bit_size(alu->src[1].src);
   if (op0_bit_size != op1_bit_size) {
      const struct dxil_type *type =
         dxil_module_get_int_type(&ctx->mod, op0_bit_size);
      enum dxil_cast_opcode cast_op =
         op1_bit_size < op0_bit_size ? DXIL_CAST_ZEXT : DXIL_CAST_TRUNC;
      op1 = dxil_emit_cast(&ctx->mod, cast_op, type, op1);
   }

   const struct dxil_value *v =
      dxil_emit_binop(&ctx->mod, opcode, op0, op1, 0);
   if (!v)
      return false;
   store_alu_dest(ctx, alu, 0, v);
   return true;
}

static bool
emit_cmp(struct ntd_context *ctx, nir_alu_instr *alu,
         enum dxil_cmp_pred pred,
         const struct dxil_value *op0, const struct dxil_value *op1)
{
   const struct dxil_value *v = dxil_emit_cmp(&ctx->mod, pred, op0, op1);
   if (!v)
      return false;
   store_alu_dest(ctx, alu, 0, v);
   return true;
}

static enum dxil_cast_opcode
get_cast_op(nir_alu_instr *alu)
{
   unsigned dst_bits = nir_dest_bit_size(alu->dest.dest);
   unsigned src_bits = nir_src_bit_size(alu->src[0].src);

   switch (alu->op) {
   /* bool -> int */
   case nir_op_b2i16:
   case nir_op_b2i32:
   case nir_op_b2i64:
      return DXIL_CAST_ZEXT;

   /* float -> float */
   case nir_op_f2f16_rtz:
   case nir_op_f2f32:
   case nir_op_f2f64:
      assert(dst_bits != src_bits);
      if (dst_bits < src_bits)
         return DXIL_CAST_FPTRUNC;
      else
         return DXIL_CAST_FPEXT;

   /* int -> int */
   case nir_op_i2i16:
   case nir_op_i2i32:
   case nir_op_i2i64:
      assert(dst_bits != src_bits);
      if (dst_bits < src_bits)
         return DXIL_CAST_TRUNC;
      else
         return DXIL_CAST_SEXT;

   /* uint -> uint */
   case nir_op_u2u16:
   case nir_op_u2u32:
   case nir_op_u2u64:
      assert(dst_bits != src_bits);
      if (dst_bits < src_bits)
         return DXIL_CAST_TRUNC;
      else
         return DXIL_CAST_ZEXT;

   /* float -> int */
   case nir_op_f2i16:
   case nir_op_f2i32:
   case nir_op_f2i64:
      return DXIL_CAST_FPTOSI;

   /* float -> uint */
   case nir_op_f2u16:
   case nir_op_f2u32:
   case nir_op_f2u64:
      return DXIL_CAST_FPTOUI;

   /* int -> float */
   case nir_op_i2f16:
   case nir_op_i2f32:
   case nir_op_i2f64:
      return DXIL_CAST_SITOFP;

   /* uint -> float */
   case nir_op_u2f16:
   case nir_op_u2f32:
   case nir_op_u2f64:
      return DXIL_CAST_UITOFP;

   default:
      unreachable("unexpected cast op");
   }
}

static const struct dxil_type *
get_cast_dest_type(struct ntd_context *ctx, nir_alu_instr *alu)
{
   unsigned dst_bits = nir_dest_bit_size(alu->dest.dest);
   switch (nir_alu_type_get_base_type(nir_op_infos[alu->op].output_type)) {
   case nir_type_bool:
      assert(dst_bits == 1);
      FALLTHROUGH;
   case nir_type_int:
   case nir_type_uint:
      return dxil_module_get_int_type(&ctx->mod, dst_bits);

   case nir_type_float:
      return dxil_module_get_float_type(&ctx->mod, dst_bits);

   default:
      unreachable("unknown nir_alu_type");
   }
}

static bool
is_double(nir_alu_type alu_type, unsigned bit_size)
{
   return nir_alu_type_get_base_type(alu_type) == nir_type_float &&
          bit_size == 64;
}

static bool
emit_cast(struct ntd_context *ctx, nir_alu_instr *alu,
          const struct dxil_value *value)
{
   enum dxil_cast_opcode opcode = get_cast_op(alu);
   const struct dxil_type *type = get_cast_dest_type(ctx, alu);
   if (!type)
      return false;

   const nir_op_info *info = &nir_op_infos[alu->op];
   switch (opcode) {
   case DXIL_CAST_UITOFP:
   case DXIL_CAST_SITOFP:
      if (is_double(info->output_type, nir_dest_bit_size(alu->dest.dest)))
         ctx->mod.feats.dx11_1_double_extensions = true;
      break;
   case DXIL_CAST_FPTOUI:
   case DXIL_CAST_FPTOSI:
      if (is_double(info->input_types[0], nir_src_bit_size(alu->src[0].src)))
         ctx->mod.feats.dx11_1_double_extensions = true;
      break;
   default:
      break;
   }

   const struct dxil_value *v = dxil_emit_cast(&ctx->mod, opcode, type,
                                               value);
   if (!v)
      return false;
   store_alu_dest(ctx, alu, 0, v);
   return true;
}

static enum overload_type
get_overload(nir_alu_type alu_type, unsigned bit_size)
{
   switch (nir_alu_type_get_base_type(alu_type)) {
   case nir_type_int:
   case nir_type_uint:
      switch (bit_size) {
      case 16: return DXIL_I16;
      case 32: return DXIL_I32;
      case 64: return DXIL_I64;
      default:
         unreachable("unexpected bit_size");
      }
   case nir_type_float:
      switch (bit_size) {
      case 16: return DXIL_F16;
      case 32: return DXIL_F32;
      case 64: return DXIL_F64;
      default:
         unreachable("unexpected bit_size");
      }
   default:
      unreachable("unexpected output type");
   }
}

static bool
emit_unary_intin(struct ntd_context *ctx, nir_alu_instr *alu,
                 enum dxil_intr intr, const struct dxil_value *op)
{
   const nir_op_info *info = &nir_op_infos[alu->op];
   unsigned src_bits = nir_src_bit_size(alu->src[0].src);
   enum overload_type overload = get_overload(info->input_types[0], src_bits);

   const struct dxil_value *v = emit_unary_call(ctx, overload, intr, op);
   if (!v)
      return false;
   store_alu_dest(ctx, alu, 0, v);
   return true;
}

static bool
emit_binary_intin(struct ntd_context *ctx, nir_alu_instr *alu,
                  enum dxil_intr intr,
                  const struct dxil_value *op0, const struct dxil_value *op1)
{
   const nir_op_info *info = &nir_op_infos[alu->op];
   assert(info->output_type == info->input_types[0]);
   assert(info->output_type == info->input_types[1]);
   unsigned dst_bits = nir_dest_bit_size(alu->dest.dest);
   assert(nir_src_bit_size(alu->src[0].src) == dst_bits);
   assert(nir_src_bit_size(alu->src[1].src) == dst_bits);
   enum overload_type overload = get_overload(info->output_type, dst_bits);

   const struct dxil_value *v = emit_binary_call(ctx, overload, intr,
                                                 op0, op1);
   if (!v)
      return false;
   store_alu_dest(ctx, alu, 0, v);
   return true;
}

static bool
emit_tertiary_intin(struct ntd_context *ctx, nir_alu_instr *alu,
                    enum dxil_intr intr,
                    const struct dxil_value *op0,
                    const struct dxil_value *op1,
                    const struct dxil_value *op2)
{
   const nir_op_info *info = &nir_op_infos[alu->op];
   unsigned dst_bits = nir_dest_bit_size(alu->dest.dest);
   assert(nir_src_bit_size(alu->src[0].src) == dst_bits);
   assert(nir_src_bit_size(alu->src[1].src) == dst_bits);
   assert(nir_src_bit_size(alu->src[2].src) == dst_bits);

   assert(get_overload(info->output_type, dst_bits) == get_overload(info->input_types[0], dst_bits));
   assert(get_overload(info->output_type, dst_bits) == get_overload(info->input_types[1], dst_bits));
   assert(get_overload(info->output_type, dst_bits) == get_overload(info->input_types[2], dst_bits));

   enum overload_type overload = get_overload(info->output_type, dst_bits);

   const struct dxil_value *v = emit_tertiary_call(ctx, overload, intr,
                                                   op0, op1, op2);
   if (!v)
      return false;
   store_alu_dest(ctx, alu, 0, v);
   return true;
}

static bool
emit_bitfield_insert(struct ntd_context *ctx, nir_alu_instr *alu,
                     const struct dxil_value *base,
                     const struct dxil_value *insert,
                     const struct dxil_value *offset,
                     const struct dxil_value *width)
{
   /* DXIL is width, offset, insert, base, NIR is base, insert, offset, width */
   const struct dxil_value *v = emit_quaternary_call(ctx, DXIL_I32, DXIL_INTR_BFI,
                                                     width, offset, insert, base);
   if (!v)
      return false;

   /* DXIL uses the 5 LSB from width/offset. Special-case width >= 32 == copy insert. */
   const struct dxil_value *compare_width = dxil_emit_cmp(&ctx->mod, DXIL_ICMP_SGE,
      width, dxil_module_get_int32_const(&ctx->mod, 32));
   v = dxil_emit_select(&ctx->mod, compare_width, insert, v);
   store_alu_dest(ctx, alu, 0, v);
   return true;
}

static bool emit_select(struct ntd_context *ctx, nir_alu_instr *alu,
                        const struct dxil_value *sel,
                        const struct dxil_value *val_true,
                        const struct dxil_value *val_false)
{
   assert(sel);
   assert(val_true);
   assert(val_false);

   const struct dxil_value *v = dxil_emit_select(&ctx->mod, sel, val_true, val_false);
   if (!v)
      return false;

   store_alu_dest(ctx, alu, 0, v);
   return true;
}

static bool
emit_b2f16(struct ntd_context *ctx, nir_alu_instr *alu, const struct dxil_value *val)
{
   assert(val);

   struct dxil_module *m = &ctx->mod;

   const struct dxil_value *c1 = dxil_module_get_float16_const(m, 0x3C00);
   const struct dxil_value *c0 = dxil_module_get_float16_const(m, 0);

   if (!c0 || !c1)
      return false;

   return emit_select(ctx, alu, val, c1, c0);
}

static bool
emit_b2f32(struct ntd_context *ctx, nir_alu_instr *alu, const struct dxil_value *val)
{
   assert(val);

   struct dxil_module *m = &ctx->mod;

   const struct dxil_value *c1 = dxil_module_get_float_const(m, 1.0f);
   const struct dxil_value *c0 = dxil_module_get_float_const(m, 0.0f);

   if (!c0 || !c1)
      return false;

   return emit_select(ctx, alu, val, c1, c0);
}

static bool
emit_b2f64(struct ntd_context *ctx, nir_alu_instr *alu, const struct dxil_value *val)
{
   assert(val);

   struct dxil_module *m = &ctx->mod;

   const struct dxil_value *c1 = dxil_module_get_double_const(m, 1.0);
   const struct dxil_value *c0 = dxil_module_get_double_const(m, 0.0);

   if (!c0 || !c1)
      return false;

   ctx->mod.feats.doubles = 1;
   return emit_select(ctx, alu, val, c1, c0);
}

static bool
emit_f2b32(struct ntd_context *ctx, nir_alu_instr *alu, const struct dxil_value *val)
{
   assert(val);

   const struct dxil_value *zero = dxil_module_get_float_const(&ctx->mod, 0.0f);
   return emit_cmp(ctx, alu, DXIL_FCMP_UNE, val, zero);
}

static bool
emit_f16tof32(struct ntd_context *ctx, nir_alu_instr *alu, const struct dxil_value *val, bool shift)
{
   if (shift) {
      val = dxil_emit_binop(&ctx->mod, DXIL_BINOP_LSHR, val,
         dxil_module_get_int32_const(&ctx->mod, 16), 0);
      if (!val)
         return false;
   }

   const struct dxil_func *func = dxil_get_function(&ctx->mod,
                                                    "dx.op.legacyF16ToF32",
                                                    DXIL_NONE);
   if (!func)
      return false;

   const struct dxil_value *opcode = dxil_module_get_int32_const(&ctx->mod, DXIL_INTR_LEGACY_F16TOF32);
   if (!opcode)
      return false;

   const struct dxil_value *args[] = {
     opcode,
     val
   };

   const struct dxil_value *v = dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
   if (!v)
      return false;
   store_alu_dest(ctx, alu, 0, v);
   return true;
}

static bool
emit_f32tof16(struct ntd_context *ctx, nir_alu_instr *alu, const struct dxil_value *val0, const struct dxil_value *val1)
{
   const struct dxil_func *func = dxil_get_function(&ctx->mod,
                                                    "dx.op.legacyF32ToF16",
                                                    DXIL_NONE);
   if (!func)
      return false;

   const struct dxil_value *opcode = dxil_module_get_int32_const(&ctx->mod, DXIL_INTR_LEGACY_F32TOF16);
   if (!opcode)
      return false;

   const struct dxil_value *args[] = {
     opcode,
     val0
   };

   const struct dxil_value *v = dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
   if (!v)
      return false;

   if (!nir_src_is_const(alu->src[1].src) || nir_src_as_int(alu->src[1].src) != 0) {
      args[1] = val1;
      const struct dxil_value *v_high = dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
      if (!v_high)
         return false;

      v_high = dxil_emit_binop(&ctx->mod, DXIL_BINOP_SHL, v_high,
         dxil_module_get_int32_const(&ctx->mod, 16), 0);
      if (!v_high)
         return false;

      v = dxil_emit_binop(&ctx->mod, DXIL_BINOP_OR, v, v_high, 0);
      if (!v)
         return false;
   }

   store_alu_dest(ctx, alu, 0, v);
   return true;
}

static bool
emit_vec(struct ntd_context *ctx, nir_alu_instr *alu, unsigned num_inputs)
{
   const struct dxil_type *type = get_alu_src_type(ctx, alu, 0);
   nir_alu_type t = dxil_type_to_nir_type(type);

   for (unsigned i = 0; i < num_inputs; i++) {
      const struct dxil_value *src =
         get_src(ctx, &alu->src[i].src, alu->src[i].swizzle[0], t);
      if (!src)
         return false;

      store_alu_dest(ctx, alu, i, src);
   }
   return true;
}

static bool
emit_make_double(struct ntd_context *ctx, nir_alu_instr *alu)
{
   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.makeDouble", DXIL_F64);
   if (!func)
      return false;

   const struct dxil_value *opcode = dxil_module_get_int32_const(&ctx->mod, DXIL_INTR_MAKE_DOUBLE);
   if (!opcode)
      return false;

   const struct dxil_value *args[3] = {
      opcode,
      get_src(ctx, &alu->src[0].src, alu->src[0].swizzle[0], nir_type_uint32),
      get_src(ctx, &alu->src[0].src, alu->src[0].swizzle[1], nir_type_uint32),
   };
   if (!args[1] || !args[2])
      return false;

   const struct dxil_value *v = dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
   if (!v)
      return false;
   store_dest(ctx, &alu->dest.dest, 0, v, nir_type_float64);
   return true;
}

static bool
emit_split_double(struct ntd_context *ctx, nir_alu_instr *alu)
{
   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.splitDouble", DXIL_F64);
   if (!func)
      return false;

   const struct dxil_value *opcode = dxil_module_get_int32_const(&ctx->mod, DXIL_INTR_SPLIT_DOUBLE);
   if (!opcode)
      return false;

   const struct dxil_value *args[] = {
      opcode,
      get_src(ctx, &alu->src[0].src, alu->src[0].swizzle[0], nir_type_float64)
   };
   if (!args[1])
      return false;

   const struct dxil_value *v = dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
   if (!v)
      return false;

   const struct dxil_value *hi = dxil_emit_extractval(&ctx->mod, v, 0);
   const struct dxil_value *lo = dxil_emit_extractval(&ctx->mod, v, 1);
   if (!hi || !lo)
      return false;

   store_dest_value(ctx, &alu->dest.dest, 0, hi);
   store_dest_value(ctx, &alu->dest.dest, 1, lo);
   return true;
}

static bool
emit_alu(struct ntd_context *ctx, nir_alu_instr *alu)
{
   /* handle vec-instructions first; they are the only ones that produce
    * vector results.
    */
   switch (alu->op) {
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4:
   case nir_op_vec8:
   case nir_op_vec16:
      return emit_vec(ctx, alu, nir_op_infos[alu->op].num_inputs);
   case nir_op_mov: {
         assert(nir_dest_num_components(alu->dest.dest) == 1);
         store_ssa_def(ctx, &alu->dest.dest.ssa, 0, get_src_ssa(ctx,
                        alu->src->src.ssa, alu->src->swizzle[0]));
         return true;
      }
   case nir_op_pack_double_2x32_dxil:
      return emit_make_double(ctx, alu);
   case nir_op_unpack_double_2x32_dxil:
      return emit_split_double(ctx, alu);
   default:
      /* silence warnings */
      ;
   }

   /* other ops should be scalar */
   assert(alu->dest.write_mask == 1);
   const struct dxil_value *src[4];
   assert(nir_op_infos[alu->op].num_inputs <= 4);
   for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
      src[i] = get_alu_src(ctx, alu, i);
      if (!src[i])
         return false;
   }

   switch (alu->op) {
   case nir_op_iadd:
   case nir_op_fadd: return emit_binop(ctx, alu, DXIL_BINOP_ADD, src[0], src[1]);

   case nir_op_isub:
   case nir_op_fsub: return emit_binop(ctx, alu, DXIL_BINOP_SUB, src[0], src[1]);

   case nir_op_imul:
   case nir_op_fmul: return emit_binop(ctx, alu, DXIL_BINOP_MUL, src[0], src[1]);

   case nir_op_fdiv:
      if (alu->dest.dest.ssa.bit_size == 64)
         ctx->mod.feats.dx11_1_double_extensions = 1;
      FALLTHROUGH;
   case nir_op_idiv:
      return emit_binop(ctx, alu, DXIL_BINOP_SDIV, src[0], src[1]);

   case nir_op_udiv: return emit_binop(ctx, alu, DXIL_BINOP_UDIV, src[0], src[1]);
   case nir_op_irem: return emit_binop(ctx, alu, DXIL_BINOP_SREM, src[0], src[1]);
   case nir_op_imod: return emit_binop(ctx, alu, DXIL_BINOP_UREM, src[0], src[1]);
   case nir_op_umod: return emit_binop(ctx, alu, DXIL_BINOP_UREM, src[0], src[1]);
   case nir_op_ishl: return emit_shift(ctx, alu, DXIL_BINOP_SHL, src[0], src[1]);
   case nir_op_ishr: return emit_shift(ctx, alu, DXIL_BINOP_ASHR, src[0], src[1]);
   case nir_op_ushr: return emit_shift(ctx, alu, DXIL_BINOP_LSHR, src[0], src[1]);
   case nir_op_iand: return emit_binop(ctx, alu, DXIL_BINOP_AND, src[0], src[1]);
   case nir_op_ior:  return emit_binop(ctx, alu, DXIL_BINOP_OR, src[0], src[1]);
   case nir_op_ixor: return emit_binop(ctx, alu, DXIL_BINOP_XOR, src[0], src[1]);
   case nir_op_inot: {
      unsigned bit_size = alu->dest.dest.ssa.bit_size;
      intmax_t val = bit_size == 1 ? 1 : -1;
      const struct dxil_value *negative_one = dxil_module_get_int_const(&ctx->mod, val, bit_size);
      return emit_binop(ctx, alu, DXIL_BINOP_XOR, src[0], negative_one);
   }
   case nir_op_ieq:  return emit_cmp(ctx, alu, DXIL_ICMP_EQ, src[0], src[1]);
   case nir_op_ine:  return emit_cmp(ctx, alu, DXIL_ICMP_NE, src[0], src[1]);
   case nir_op_ige:  return emit_cmp(ctx, alu, DXIL_ICMP_SGE, src[0], src[1]);
   case nir_op_uge:  return emit_cmp(ctx, alu, DXIL_ICMP_UGE, src[0], src[1]);
   case nir_op_ilt:  return emit_cmp(ctx, alu, DXIL_ICMP_SLT, src[0], src[1]);
   case nir_op_ult:  return emit_cmp(ctx, alu, DXIL_ICMP_ULT, src[0], src[1]);
   case nir_op_feq:  return emit_cmp(ctx, alu, DXIL_FCMP_OEQ, src[0], src[1]);
   case nir_op_fneu: return emit_cmp(ctx, alu, DXIL_FCMP_UNE, src[0], src[1]);
   case nir_op_flt:  return emit_cmp(ctx, alu, DXIL_FCMP_OLT, src[0], src[1]);
   case nir_op_fge:  return emit_cmp(ctx, alu, DXIL_FCMP_OGE, src[0], src[1]);
   case nir_op_bcsel: return emit_select(ctx, alu, src[0], src[1], src[2]);
   case nir_op_ftrunc: return emit_unary_intin(ctx, alu, DXIL_INTR_ROUND_Z, src[0]);
   case nir_op_fabs: return emit_unary_intin(ctx, alu, DXIL_INTR_FABS, src[0]);
   case nir_op_fcos: return emit_unary_intin(ctx, alu, DXIL_INTR_FCOS, src[0]);
   case nir_op_fsin: return emit_unary_intin(ctx, alu, DXIL_INTR_FSIN, src[0]);
   case nir_op_fceil: return emit_unary_intin(ctx, alu, DXIL_INTR_ROUND_PI, src[0]);
   case nir_op_fexp2: return emit_unary_intin(ctx, alu, DXIL_INTR_FEXP2, src[0]);
   case nir_op_flog2: return emit_unary_intin(ctx, alu, DXIL_INTR_FLOG2, src[0]);
   case nir_op_ffloor: return emit_unary_intin(ctx, alu, DXIL_INTR_ROUND_NI, src[0]);
   case nir_op_ffract: return emit_unary_intin(ctx, alu, DXIL_INTR_FRC, src[0]);
   case nir_op_fisnormal: return emit_unary_intin(ctx, alu, DXIL_INTR_ISNORMAL, src[0]);
   case nir_op_fisfinite: return emit_unary_intin(ctx, alu, DXIL_INTR_ISFINITE, src[0]);

   case nir_op_fddx:
   case nir_op_fddx_coarse: return emit_unary_intin(ctx, alu, DXIL_INTR_DDX_COARSE, src[0]);
   case nir_op_fddx_fine: return emit_unary_intin(ctx, alu, DXIL_INTR_DDX_FINE, src[0]);
   case nir_op_fddy:
   case nir_op_fddy_coarse: return emit_unary_intin(ctx, alu, DXIL_INTR_DDY_COARSE, src[0]);
   case nir_op_fddy_fine: return emit_unary_intin(ctx, alu, DXIL_INTR_DDY_FINE, src[0]);

   case nir_op_fround_even: return emit_unary_intin(ctx, alu, DXIL_INTR_ROUND_NE, src[0]);
   case nir_op_frcp: {
         const struct dxil_value *one = dxil_module_get_float_const(&ctx->mod, 1.0f);
         return emit_binop(ctx, alu, DXIL_BINOP_SDIV, one, src[0]);
      }
   case nir_op_fsat: return emit_unary_intin(ctx, alu, DXIL_INTR_SATURATE, src[0]);
   case nir_op_bit_count: return emit_unary_intin(ctx, alu, DXIL_INTR_COUNTBITS, src[0]);
   case nir_op_bitfield_reverse: return emit_unary_intin(ctx, alu, DXIL_INTR_BFREV, src[0]);
   case nir_op_ufind_msb_rev: return emit_unary_intin(ctx, alu, DXIL_INTR_FIRSTBIT_HI, src[0]);
   case nir_op_ifind_msb_rev: return emit_unary_intin(ctx, alu, DXIL_INTR_FIRSTBIT_SHI, src[0]);
   case nir_op_find_lsb: return emit_unary_intin(ctx, alu, DXIL_INTR_FIRSTBIT_LO, src[0]);
   case nir_op_imax: return emit_binary_intin(ctx, alu, DXIL_INTR_IMAX, src[0], src[1]);
   case nir_op_imin: return emit_binary_intin(ctx, alu, DXIL_INTR_IMIN, src[0], src[1]);
   case nir_op_umax: return emit_binary_intin(ctx, alu, DXIL_INTR_UMAX, src[0], src[1]);
   case nir_op_umin: return emit_binary_intin(ctx, alu, DXIL_INTR_UMIN, src[0], src[1]);
   case nir_op_frsq: return emit_unary_intin(ctx, alu, DXIL_INTR_RSQRT, src[0]);
   case nir_op_fsqrt: return emit_unary_intin(ctx, alu, DXIL_INTR_SQRT, src[0]);
   case nir_op_fmax: return emit_binary_intin(ctx, alu, DXIL_INTR_FMAX, src[0], src[1]);
   case nir_op_fmin: return emit_binary_intin(ctx, alu, DXIL_INTR_FMIN, src[0], src[1]);
   case nir_op_ffma:
      if (alu->dest.dest.ssa.bit_size == 64)
         ctx->mod.feats.dx11_1_double_extensions = 1;
      return emit_tertiary_intin(ctx, alu, DXIL_INTR_FMA, src[0], src[1], src[2]);

   case nir_op_ibfe: return emit_tertiary_intin(ctx, alu, DXIL_INTR_IBFE, src[2], src[1], src[0]);
   case nir_op_ubfe: return emit_tertiary_intin(ctx, alu, DXIL_INTR_UBFE, src[2], src[1], src[0]);
   case nir_op_bitfield_insert: return emit_bitfield_insert(ctx, alu, src[0], src[1], src[2], src[3]);

   case nir_op_unpack_half_2x16_split_x: return emit_f16tof32(ctx, alu, src[0], false);
   case nir_op_unpack_half_2x16_split_y: return emit_f16tof32(ctx, alu, src[0], true);
   case nir_op_pack_half_2x16_split: return emit_f32tof16(ctx, alu, src[0], src[1]);

   case nir_op_b2i16:
   case nir_op_i2i16:
   case nir_op_f2i16:
   case nir_op_f2u16:
   case nir_op_u2u16:
   case nir_op_u2f16:
   case nir_op_i2f16:
   case nir_op_f2f16_rtz:
   case nir_op_b2i32:
   case nir_op_f2f32:
   case nir_op_f2i32:
   case nir_op_f2u32:
   case nir_op_i2f32:
   case nir_op_i2i32:
   case nir_op_u2f32:
   case nir_op_u2u32:
   case nir_op_b2i64:
   case nir_op_f2f64:
   case nir_op_f2i64:
   case nir_op_f2u64:
   case nir_op_i2f64:
   case nir_op_i2i64:
   case nir_op_u2f64:
   case nir_op_u2u64:
      return emit_cast(ctx, alu, src[0]);

   case nir_op_f2b32: return emit_f2b32(ctx, alu, src[0]);
   case nir_op_b2f16: return emit_b2f16(ctx, alu, src[0]);
   case nir_op_b2f32: return emit_b2f32(ctx, alu, src[0]);
   case nir_op_b2f64: return emit_b2f64(ctx, alu, src[0]);
   default:
      log_nir_instr_unsupported(ctx->logger, "Unimplemented ALU instruction",
                                &alu->instr);
      return false;
   }
}

static const struct dxil_value *
load_ubo(struct ntd_context *ctx, const struct dxil_value *handle,
         const struct dxil_value *offset, enum overload_type overload)
{
   assert(handle && offset);

   const struct dxil_value *opcode = dxil_module_get_int32_const(&ctx->mod, DXIL_INTR_CBUFFER_LOAD_LEGACY);
   if (!opcode)
      return NULL;

   const struct dxil_value *args[] = {
      opcode, handle, offset
   };

   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.cbufferLoadLegacy", overload);
   if (!func)
      return NULL;
   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static bool
emit_barrier_impl(struct ntd_context *ctx, nir_variable_mode modes, nir_scope execution_scope, nir_scope mem_scope)
{
   const struct dxil_value *opcode, *mode;
   const struct dxil_func *func;
   uint32_t flags = 0;

   if (execution_scope == NIR_SCOPE_WORKGROUP)
      flags |= DXIL_BARRIER_MODE_SYNC_THREAD_GROUP;

   if (modes & (nir_var_mem_ssbo | nir_var_mem_global | nir_var_image)) {
      if (mem_scope > NIR_SCOPE_WORKGROUP)
         flags |= DXIL_BARRIER_MODE_UAV_FENCE_GLOBAL;
      else
         flags |= DXIL_BARRIER_MODE_UAV_FENCE_THREAD_GROUP;
   }

   if (modes & nir_var_mem_shared)
      flags |= DXIL_BARRIER_MODE_GROUPSHARED_MEM_FENCE;

   func = dxil_get_function(&ctx->mod, "dx.op.barrier", DXIL_NONE);
   if (!func)
      return false;

   opcode = dxil_module_get_int32_const(&ctx->mod, DXIL_INTR_BARRIER);
   if (!opcode)
      return false;

   mode = dxil_module_get_int32_const(&ctx->mod, flags);
   if (!mode)
      return false;

   const struct dxil_value *args[] = { opcode, mode };

   return dxil_emit_call_void(&ctx->mod, func,
                              args, ARRAY_SIZE(args));
}

static bool
emit_barrier(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   return emit_barrier_impl(ctx,
      nir_intrinsic_memory_modes(intr),
      nir_intrinsic_execution_scope(intr),
      nir_intrinsic_memory_scope(intr));
}

/* Memory barrier for UAVs (buffers/images) at cross-workgroup scope */
static bool
emit_memory_barrier(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   return emit_barrier_impl(ctx,
      nir_var_mem_global,
      NIR_SCOPE_NONE,
      NIR_SCOPE_DEVICE);
}

/* Memory barrier for TGSM */
static bool
emit_memory_barrier_shared(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   return emit_barrier_impl(ctx,
      nir_var_mem_shared,
      NIR_SCOPE_NONE,
      NIR_SCOPE_WORKGROUP);
}

/* Memory barrier for all intra-workgroup memory accesses (UAVs and TGSM) */
static bool
emit_group_memory_barrier(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   return emit_barrier_impl(ctx,
      nir_var_mem_shared | nir_var_mem_global,
      NIR_SCOPE_NONE,
      NIR_SCOPE_WORKGROUP);
}

static bool
emit_control_barrier(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   return emit_barrier_impl(ctx,
      nir_var_mem_shared,
      NIR_SCOPE_WORKGROUP,
      NIR_SCOPE_NONE);
}

static bool
emit_load_global_invocation_id(struct ntd_context *ctx,
                                    nir_intrinsic_instr *intr)
{
   assert(intr->dest.is_ssa);
   nir_component_mask_t comps = nir_ssa_def_components_read(&intr->dest.ssa);

   for (int i = 0; i < nir_intrinsic_dest_components(intr); i++) {
      if (comps & (1 << i)) {
         const struct dxil_value *idx = dxil_module_get_int32_const(&ctx->mod, i);
         if (!idx)
            return false;
         const struct dxil_value *globalid = emit_threadid_call(ctx, idx);

         if (!globalid)
            return false;

         store_dest_value(ctx, &intr->dest, i, globalid);
      }
   }
   return true;
}

static bool
emit_load_local_invocation_id(struct ntd_context *ctx,
                              nir_intrinsic_instr *intr)
{
   assert(intr->dest.is_ssa);
   nir_component_mask_t comps = nir_ssa_def_components_read(&intr->dest.ssa);

   for (int i = 0; i < nir_intrinsic_dest_components(intr); i++) {
      if (comps & (1 << i)) {
         const struct dxil_value
            *idx = dxil_module_get_int32_const(&ctx->mod, i);
         if (!idx)
            return false;
         const struct dxil_value
            *threadidingroup = emit_threadidingroup_call(ctx, idx);
         if (!threadidingroup)
            return false;
         store_dest_value(ctx, &intr->dest, i, threadidingroup);
      }
   }
   return true;
}

static bool
emit_load_local_invocation_index(struct ntd_context *ctx,
                                 nir_intrinsic_instr *intr)
{
   assert(intr->dest.is_ssa);

   const struct dxil_value
      *flattenedthreadidingroup = emit_flattenedthreadidingroup_call(ctx);
   if (!flattenedthreadidingroup)
      return false;
   store_dest_value(ctx, &intr->dest, 0, flattenedthreadidingroup);
   
   return true;
}

static bool
emit_load_local_workgroup_id(struct ntd_context *ctx,
                              nir_intrinsic_instr *intr)
{
   assert(intr->dest.is_ssa);
   nir_component_mask_t comps = nir_ssa_def_components_read(&intr->dest.ssa);

   for (int i = 0; i < nir_intrinsic_dest_components(intr); i++) {
      if (comps & (1 << i)) {
         const struct dxil_value *idx = dxil_module_get_int32_const(&ctx->mod, i);
         if (!idx)
            return false;
         const struct dxil_value *groupid = emit_groupid_call(ctx, idx);
         if (!groupid)
            return false;
         store_dest_value(ctx, &intr->dest, i, groupid);
      }
   }
   return true;
}

static const struct dxil_value *
call_unary_external_function(struct ntd_context *ctx,
                             const char *name,
                             int32_t dxil_intr,
                             enum overload_type overload)
{
   const struct dxil_func *func =
      dxil_get_function(&ctx->mod, name, overload);
   if (!func)
      return false;

   const struct dxil_value *opcode =
      dxil_module_get_int32_const(&ctx->mod, dxil_intr);
   if (!opcode)
      return false;

   const struct dxil_value *args[] = {opcode};

   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static bool
emit_load_unary_external_function(struct ntd_context *ctx,
                                  nir_intrinsic_instr *intr, const char *name,
                                  int32_t dxil_intr,
                                  enum overload_type overload)
{
   const struct dxil_value *value = call_unary_external_function(ctx, name, dxil_intr, overload);
   store_dest_value(ctx, &intr->dest, 0, value);

   return true;
}

static bool
emit_load_sample_mask_in(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   const struct dxil_value *value = call_unary_external_function(ctx,
      "dx.op.coverage", DXIL_INTR_COVERAGE, DXIL_I32);

   /* Mask coverage with (1 << sample index). Note, done as an AND to handle extrapolation cases. */
   if (ctx->mod.info.has_per_sample_input) {
      value = dxil_emit_binop(&ctx->mod, DXIL_BINOP_AND, value,
         dxil_emit_binop(&ctx->mod, DXIL_BINOP_SHL,
            dxil_module_get_int32_const(&ctx->mod, 1),
            call_unary_external_function(ctx, "dx.op.sampleIndex", DXIL_INTR_SAMPLE_INDEX, DXIL_I32), 0), 0);
   }

   store_dest_value(ctx, &intr->dest, 0, value);
   return true;
}

static bool
emit_load_tess_coord(struct ntd_context *ctx,
                     nir_intrinsic_instr *intr)
{
   const struct dxil_func *func =
      dxil_get_function(&ctx->mod, "dx.op.domainLocation", DXIL_F32);
   if (!func)
      return false;

   const struct dxil_value *opcode =
      dxil_module_get_int32_const(&ctx->mod, DXIL_INTR_DOMAIN_LOCATION);
   if (!opcode)
      return false;

   unsigned num_coords = ctx->shader->info.tess._primitive_mode == TESS_PRIMITIVE_TRIANGLES ? 3 : 2;
   for (unsigned i = 0; i < num_coords; ++i) {
      unsigned component_idx = i;

      const struct dxil_value *component = dxil_module_get_int32_const(&ctx->mod, component_idx);
      if (!component)
         return false;

      const struct dxil_value *args[] = { opcode, component };

      const struct dxil_value *value =
         dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
      store_dest_value(ctx, &intr->dest, i, value);
   }

   for (unsigned i = num_coords; i < intr->dest.ssa.num_components; ++i) {
      const struct dxil_value *value = dxil_module_get_float_const(&ctx->mod, 0.0f);
      store_dest_value(ctx, &intr->dest, i, value);
   }

   return true;
}

static const struct dxil_value *
get_int32_undef(struct dxil_module *m)
{
   const struct dxil_type *int32_type =
      dxil_module_get_int_type(m, 32);
   if (!int32_type)
      return NULL;

   return dxil_module_get_undef(m, int32_type);
}

static const struct dxil_value *
emit_gep_for_index(struct ntd_context *ctx, const nir_variable *var,
                   const struct dxil_value *index)
{
   assert(var->data.mode == nir_var_shader_temp);

   struct hash_entry *he = _mesa_hash_table_search(ctx->consts, var);
   assert(he != NULL);
   const struct dxil_value *ptr = he->data;

   const struct dxil_value *zero = dxil_module_get_int32_const(&ctx->mod, 0);
   if (!zero)
      return NULL;

   const struct dxil_value *ops[] = { ptr, zero, index };
   return dxil_emit_gep_inbounds(&ctx->mod, ops, ARRAY_SIZE(ops));
}

static const struct dxil_value *
get_resource_handle(struct ntd_context *ctx, nir_src *src, enum dxil_resource_class class,
                    enum dxil_resource_kind kind)
{
   /* This source might be one of:
    * 1. Constant resource index - just look it up in precomputed handle arrays
    *    If it's null in that array, create a handle, and store the result
    * 2. A handle from load_vulkan_descriptor - just get the stored SSA value
    * 3. Dynamic resource index - create a handle for it here
    */
   assert(src->ssa->num_components == 1 && src->ssa->bit_size == 32);
   nir_const_value *const_block_index = nir_src_as_const_value(*src);
   const struct dxil_value **handle_entry = NULL;
   if (const_block_index) {
      assert(ctx->opts->environment != DXIL_ENVIRONMENT_VULKAN);
      switch (kind) {
      case DXIL_RESOURCE_KIND_CBUFFER:
         handle_entry = &ctx->cbv_handles[const_block_index->u32];
         break;
      case DXIL_RESOURCE_KIND_RAW_BUFFER:
         if (class == DXIL_RESOURCE_CLASS_UAV)
            handle_entry = &ctx->ssbo_handles[const_block_index->u32];
         else
            handle_entry = &ctx->srv_handles[const_block_index->u32];
         break;
      case DXIL_RESOURCE_KIND_SAMPLER:
         handle_entry = &ctx->sampler_handles[const_block_index->u32];
         break;
      default:
         if (class == DXIL_RESOURCE_CLASS_UAV)
            handle_entry = &ctx->image_handles[const_block_index->u32];
         else
            handle_entry = &ctx->srv_handles[const_block_index->u32];
         break;
      }
   }

   if (handle_entry && *handle_entry)
      return *handle_entry;

   const struct dxil_value *value = get_src_ssa(ctx, src->ssa, 0);
   if (nir_src_as_deref(*src) ||
       ctx->opts->environment == DXIL_ENVIRONMENT_VULKAN) {
      return value;
   }

   unsigned space = 0;
   if (ctx->opts->environment == DXIL_ENVIRONMENT_GL &&
       class == DXIL_RESOURCE_CLASS_UAV) {
      if (kind == DXIL_RESOURCE_KIND_RAW_BUFFER)
         space = 2;
      else
         space = 1;
   }

   /* The base binding here will almost always be zero. The only cases where we end
    * up in this type of dynamic indexing are:
    * 1. GL UBOs
    * 2. GL SSBOs
    * 2. CL SSBOs
    * In all cases except GL UBOs, the resources are a single zero-based array.
    * In that case, the base is 1, because uniforms use 0 and cannot by dynamically
    * indexed. All other cases should either fall into static indexing (first early return),
    * deref-based dynamic handle creation (images, or Vulkan textures/samplers), or
    * load_vulkan_descriptor handle creation.
    */
   unsigned base_binding = 0;
   if (ctx->opts->environment == DXIL_ENVIRONMENT_GL &&
       class == DXIL_RESOURCE_CLASS_CBV)
      base_binding = 1;

   const struct dxil_value *handle = emit_createhandle_call_dynamic(ctx, class,
      space, base_binding, value, !const_block_index);
   if (handle_entry)
      *handle_entry = handle;

   return handle;
}

static bool
emit_load_ssbo(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   const struct dxil_value *int32_undef = get_int32_undef(&ctx->mod);

   enum dxil_resource_class class = DXIL_RESOURCE_CLASS_UAV;
   if (ctx->opts->environment == DXIL_ENVIRONMENT_VULKAN) {
      nir_variable *var = nir_get_binding_variable(ctx->shader, nir_chase_binding(intr->src[0]));
      if (var && var->data.access & ACCESS_NON_WRITEABLE)
         class = DXIL_RESOURCE_CLASS_SRV;
   }

   const struct dxil_value *handle = get_resource_handle(ctx, &intr->src[0], class, DXIL_RESOURCE_KIND_RAW_BUFFER);
   const struct dxil_value *offset =
      get_src(ctx, &intr->src[1], 0, nir_type_uint);
   if (!int32_undef || !handle || !offset)
      return false;

   assert(nir_src_bit_size(intr->src[0]) == 32);
   assert(nir_intrinsic_dest_components(intr) <= 4);

   const struct dxil_value *coord[2] = {
      offset,
      int32_undef
   };

   const struct dxil_value *load = emit_bufferload_call(ctx, handle, coord, DXIL_I32);
   if (!load)
      return false;

   for (int i = 0; i < nir_intrinsic_dest_components(intr); i++) {
      const struct dxil_value *val =
         dxil_emit_extractval(&ctx->mod, load, i);
      if (!val)
         return false;
      store_dest_value(ctx, &intr->dest, i, val);
   }
   return true;
}

static bool
emit_store_ssbo(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   const struct dxil_value* handle = get_resource_handle(ctx, &intr->src[1], DXIL_RESOURCE_CLASS_UAV, DXIL_RESOURCE_KIND_RAW_BUFFER);
   const struct dxil_value *offset =
      get_src(ctx, &intr->src[2], 0, nir_type_uint);
   if (!handle || !offset)
      return false;

   assert(nir_src_bit_size(intr->src[0]) == 32);
   unsigned num_components = nir_src_num_components(intr->src[0]);
   assert(num_components <= 4);
   const struct dxil_value *value[4];
   for (unsigned i = 0; i < num_components; ++i) {
      value[i] = get_src(ctx, &intr->src[0], i, nir_type_uint);
      if (!value[i])
         return false;
   }

   const struct dxil_value *int32_undef = get_int32_undef(&ctx->mod);
   if (!int32_undef)
      return false;

   const struct dxil_value *coord[2] = {
      offset,
      int32_undef
   };

   for (int i = num_components; i < 4; ++i)
      value[i] = int32_undef;

   const struct dxil_value *write_mask =
      dxil_module_get_int8_const(&ctx->mod, (1u << num_components) - 1);
   if (!write_mask)
      return false;

   return emit_bufferstore_call(ctx, handle, coord, value, write_mask, DXIL_I32);
}

static bool
emit_store_ssbo_masked(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   const struct dxil_value *value =
      get_src(ctx, &intr->src[0], 0, nir_type_uint);
   const struct dxil_value *mask =
      get_src(ctx, &intr->src[1], 0, nir_type_uint);
   const struct dxil_value* handle = get_resource_handle(ctx, &intr->src[2], DXIL_RESOURCE_CLASS_UAV, DXIL_RESOURCE_KIND_RAW_BUFFER);
   const struct dxil_value *offset =
      get_src(ctx, &intr->src[3], 0, nir_type_uint);
   if (!value || !mask || !handle || !offset)
      return false;

   const struct dxil_value *int32_undef = get_int32_undef(&ctx->mod);
   if (!int32_undef)
      return false;

   const struct dxil_value *coord[3] = {
      offset, int32_undef, int32_undef
   };

   return
      emit_atomic_binop(ctx, handle, DXIL_ATOMIC_AND, coord, mask) != NULL &&
      emit_atomic_binop(ctx, handle, DXIL_ATOMIC_OR, coord, value) != NULL;
}

static bool
emit_store_shared(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   const struct dxil_value *zero, *index;

   /* All shared mem accesses should have been lowered to scalar 32bit
    * accesses.
    */
   assert(nir_src_bit_size(intr->src[0]) == 32);
   assert(nir_src_num_components(intr->src[0]) == 1);

   zero = dxil_module_get_int32_const(&ctx->mod, 0);
   if (!zero)
      return false;

   if (intr->intrinsic == nir_intrinsic_store_shared_dxil)
      index = get_src(ctx, &intr->src[1], 0, nir_type_uint);
   else
      index = get_src(ctx, &intr->src[2], 0, nir_type_uint);
   if (!index)
      return false;

   const struct dxil_value *ops[] = { ctx->sharedvars, zero, index };
   const struct dxil_value *ptr, *value;

   ptr = dxil_emit_gep_inbounds(&ctx->mod, ops, ARRAY_SIZE(ops));
   if (!ptr)
      return false;

   value = get_src(ctx, &intr->src[0], 0, nir_type_uint);
   if (!value)
      return false;

   if (intr->intrinsic == nir_intrinsic_store_shared_dxil)
      return dxil_emit_store(&ctx->mod, value, ptr, 4, false);

   const struct dxil_value *mask = get_src(ctx, &intr->src[1], 0, nir_type_uint);
   if (!mask)
      return false;

   if (!dxil_emit_atomicrmw(&ctx->mod, mask, ptr, DXIL_RMWOP_AND, false,
                            DXIL_ATOMIC_ORDERING_ACQREL,
                            DXIL_SYNC_SCOPE_CROSSTHREAD))
      return false;

   if (!dxil_emit_atomicrmw(&ctx->mod, value, ptr, DXIL_RMWOP_OR, false,
                            DXIL_ATOMIC_ORDERING_ACQREL,
                            DXIL_SYNC_SCOPE_CROSSTHREAD))
      return false;

   return true;
}

static bool
emit_store_scratch(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   const struct dxil_value *zero, *index;

   /* All scratch mem accesses should have been lowered to scalar 32bit
    * accesses.
    */
   assert(nir_src_bit_size(intr->src[0]) == 32);
   assert(nir_src_num_components(intr->src[0]) == 1);

   zero = dxil_module_get_int32_const(&ctx->mod, 0);
   if (!zero)
      return false;

   index = get_src(ctx, &intr->src[1], 0, nir_type_uint);
   if (!index)
      return false;

   const struct dxil_value *ops[] = { ctx->scratchvars, zero, index };
   const struct dxil_value *ptr, *value;

   ptr = dxil_emit_gep_inbounds(&ctx->mod, ops, ARRAY_SIZE(ops));
   if (!ptr)
      return false;

   value = get_src(ctx, &intr->src[0], 0, nir_type_uint);
   if (!value)
      return false;

   return dxil_emit_store(&ctx->mod, value, ptr, 4, false);
}

static bool
emit_load_ubo(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   const struct dxil_value* handle = get_resource_handle(ctx, &intr->src[0], DXIL_RESOURCE_CLASS_CBV, DXIL_RESOURCE_KIND_CBUFFER);
   if (!handle)
      return false;

   const struct dxil_value *offset;
   nir_const_value *const_offset = nir_src_as_const_value(intr->src[1]);
   if (const_offset) {
      offset = dxil_module_get_int32_const(&ctx->mod, const_offset->i32 >> 4);
   } else {
      const struct dxil_value *offset_src = get_src(ctx, &intr->src[1], 0, nir_type_uint);
      const struct dxil_value *c4 = dxil_module_get_int32_const(&ctx->mod, 4);
      if (!offset_src || !c4)
         return false;

      offset = dxil_emit_binop(&ctx->mod, DXIL_BINOP_ASHR, offset_src, c4, 0);
   }

   const struct dxil_value *agg = load_ubo(ctx, handle, offset, DXIL_F32);

   if (!agg)
      return false;

   for (unsigned i = 0; i < nir_dest_num_components(intr->dest); ++i) {
      const struct dxil_value *retval = dxil_emit_extractval(&ctx->mod, agg, i);
      store_dest(ctx, &intr->dest, i, retval,
                 nir_dest_bit_size(intr->dest) > 1 ? nir_type_float : nir_type_bool);
   }
   return true;
}

static bool
emit_load_ubo_dxil(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   assert(nir_dest_num_components(intr->dest) <= 4);
   assert(nir_dest_bit_size(intr->dest) == 32);

   const struct dxil_value* handle = get_resource_handle(ctx, &intr->src[0], DXIL_RESOURCE_CLASS_CBV, DXIL_RESOURCE_KIND_CBUFFER);
   const struct dxil_value *offset =
      get_src(ctx, &intr->src[1], 0, nir_type_uint);

   if (!handle || !offset)
      return false;

   const struct dxil_value *agg = load_ubo(ctx, handle, offset, DXIL_I32);
   if (!agg)
      return false;

   for (unsigned i = 0; i < nir_dest_num_components(intr->dest); i++)
      store_dest_value(ctx, &intr->dest, i,
                       dxil_emit_extractval(&ctx->mod, agg, i));

   return true;
}

/* Need to add patch-ness as a matching parameter, since driver_location is *not* unique
 * between control points and patch variables in HS/DS
 */
static nir_variable *
find_patch_matching_variable_by_driver_location(nir_shader *s, nir_variable_mode mode, unsigned driver_location, bool patch)
{
   nir_foreach_variable_with_modes(var, s, mode) {
      if (var->data.driver_location == driver_location &&
          var->data.patch == patch)
         return var;
   }
   return NULL;
}

static bool
emit_store_output_via_intrinsic(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   assert(intr->intrinsic == nir_intrinsic_store_output ||
          ctx->mod.shader_kind == DXIL_HULL_SHADER);
   bool is_patch_constant = intr->intrinsic == nir_intrinsic_store_output &&
      ctx->mod.shader_kind == DXIL_HULL_SHADER;
   nir_alu_type out_type = nir_intrinsic_src_type(intr);
   enum overload_type overload = get_overload(out_type, intr->src[0].ssa->bit_size);
   const struct dxil_func *func = dxil_get_function(&ctx->mod, is_patch_constant ?
      "dx.op.storePatchConstant" : "dx.op.storeOutput",
      overload);

   if (!func)
      return false;

   const struct dxil_value *opcode = dxil_module_get_int32_const(&ctx->mod, is_patch_constant ?
      DXIL_INTR_STORE_PATCH_CONSTANT : DXIL_INTR_STORE_OUTPUT);
   const struct dxil_value *output_id = dxil_module_get_int32_const(&ctx->mod, nir_intrinsic_base(intr));
   unsigned row_index = intr->intrinsic == nir_intrinsic_store_output ? 1 : 2;

   /* NIR has these as 1 row, N cols, but DXIL wants them as N rows, 1 col. We muck with these in the signature
    * generation, so muck with them here too.
    */
   nir_io_semantics semantics = nir_intrinsic_io_semantics(intr);
   bool is_tess_level = is_patch_constant &&
                        (semantics.location == VARYING_SLOT_TESS_LEVEL_INNER ||
                         semantics.location == VARYING_SLOT_TESS_LEVEL_OUTER);

   const struct dxil_value *row = NULL;
   const struct dxil_value *col = NULL;
   if (is_tess_level)
      col = dxil_module_get_int8_const(&ctx->mod, 0);
   else
      row = get_src(ctx, &intr->src[row_index], 0, nir_type_int);

   bool success = true;
   uint32_t writemask = nir_intrinsic_write_mask(intr);

   nir_variable *var = find_patch_matching_variable_by_driver_location(ctx->shader, nir_var_shader_out, nir_intrinsic_base(intr), is_patch_constant);
   unsigned var_base_component = var->data.location_frac;
   unsigned base_component = nir_intrinsic_component(intr) - var_base_component;

   if (ctx->mod.minor_validator >= 5) {
      struct dxil_signature_record *sig_rec = is_patch_constant ?
         &ctx->mod.patch_consts[nir_intrinsic_base(intr)] :
         &ctx->mod.outputs[nir_intrinsic_base(intr)];
      unsigned comp_size = intr->src[0].ssa->bit_size == 64 ? 2 : 1;
      unsigned comp_mask = 0;
      if (is_tess_level)
         comp_mask = 1;
      else if (comp_size == 1)
         comp_mask = writemask << var_base_component;
      else {
         for (unsigned i = 0; i < intr->num_components; ++i)
            if ((writemask & (1 << i)))
               comp_mask |= 3 << ((i + var_base_component) * comp_size);
      }
      for (unsigned r = 0; r < sig_rec->num_elements; ++r)
         sig_rec->elements[r].never_writes_mask &= ~comp_mask;

      if (!nir_src_is_const(intr->src[row_index])) {
         struct dxil_psv_signature_element *psv_rec = is_patch_constant ?
            &ctx->mod.psv_patch_consts[nir_intrinsic_base(intr)] :
            &ctx->mod.psv_outputs[nir_intrinsic_base(intr)];
         psv_rec->dynamic_mask_and_stream |= comp_mask;
      }
   }

   for (unsigned i = 0; i < intr->num_components && success; ++i) {
      if (writemask & (1 << i)) {
         if (is_tess_level)
            row = dxil_module_get_int32_const(&ctx->mod, i + base_component);
         else
            col = dxil_module_get_int8_const(&ctx->mod, i + base_component);
         const struct dxil_value *value = get_src(ctx, &intr->src[0], i, out_type);
         if (!col || !row || !value)
            return false;

         const struct dxil_value *args[] = {
            opcode, output_id, row, col, value
         };
         success &= dxil_emit_call_void(&ctx->mod, func, args, ARRAY_SIZE(args));
      }
   }

   return success;
}

static bool
emit_load_input_via_intrinsic(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   bool attr_at_vertex = false;
   if (ctx->mod.shader_kind == DXIL_PIXEL_SHADER &&
      ctx->opts->interpolate_at_vertex &&
      ctx->opts->provoking_vertex != 0 &&
      (nir_intrinsic_dest_type(intr) & nir_type_float)) {
      nir_variable *var = nir_find_variable_with_driver_location(ctx->shader, nir_var_shader_in, nir_intrinsic_base(intr));

      attr_at_vertex = var && var->data.interpolation == INTERP_MODE_FLAT;
   }

   bool is_patch_constant = (ctx->mod.shader_kind == DXIL_DOMAIN_SHADER &&
                             intr->intrinsic == nir_intrinsic_load_input) ||
                            (ctx->mod.shader_kind == DXIL_HULL_SHADER &&
                             intr->intrinsic == nir_intrinsic_load_output);
   bool is_output_control_point = intr->intrinsic == nir_intrinsic_load_per_vertex_output;

   unsigned opcode_val;
   const char *func_name;
   if (attr_at_vertex) {
      opcode_val = DXIL_INTR_ATTRIBUTE_AT_VERTEX;
      func_name = "dx.op.attributeAtVertex";
      if (ctx->mod.minor_validator >= 6)
         ctx->mod.feats.barycentrics = 1;
   } else if (is_patch_constant) {
      opcode_val = DXIL_INTR_LOAD_PATCH_CONSTANT;
      func_name = "dx.op.loadPatchConstant";
   } else if (is_output_control_point) {
      opcode_val = DXIL_INTR_LOAD_OUTPUT_CONTROL_POINT;
      func_name = "dx.op.loadOutputControlPoint";
   } else {
      opcode_val = DXIL_INTR_LOAD_INPUT;
      func_name = "dx.op.loadInput";
   }

   const struct dxil_value *opcode = dxil_module_get_int32_const(&ctx->mod, opcode_val);
   if (!opcode)
      return false;

   const struct dxil_value *input_id = dxil_module_get_int32_const(&ctx->mod,
      is_patch_constant || is_output_control_point ?
         nir_intrinsic_base(intr) :
         ctx->mod.input_mappings[nir_intrinsic_base(intr)]);
   if (!input_id)
      return false;

   bool is_per_vertex =
      intr->intrinsic == nir_intrinsic_load_per_vertex_input ||
      intr->intrinsic == nir_intrinsic_load_per_vertex_output;
   int row_index = is_per_vertex ? 1 : 0;
   const struct dxil_value *vertex_id = NULL;
   if (!is_patch_constant) {
      if (is_per_vertex) {
         vertex_id = get_src(ctx, &intr->src[0], 0, nir_type_int);
      } else if (attr_at_vertex) {
         vertex_id = dxil_module_get_int8_const(&ctx->mod, ctx->opts->provoking_vertex);
      } else {
         const struct dxil_type *int32_type = dxil_module_get_int_type(&ctx->mod, 32);
         if (!int32_type)
            return false;

         vertex_id = dxil_module_get_undef(&ctx->mod, int32_type);
      }
      if (!vertex_id)
         return false;
   }

   /* NIR has these as 1 row, N cols, but DXIL wants them as N rows, 1 col. We muck with these in the signature
    * generation, so muck with them here too.
    */
   nir_io_semantics semantics = nir_intrinsic_io_semantics(intr);
   bool is_tess_level = is_patch_constant &&
                        (semantics.location == VARYING_SLOT_TESS_LEVEL_INNER ||
                         semantics.location == VARYING_SLOT_TESS_LEVEL_OUTER);

   const struct dxil_value *row = NULL;
   const struct dxil_value *comp = NULL;
   if (is_tess_level)
      comp = dxil_module_get_int8_const(&ctx->mod, 0);
   else
      row = get_src(ctx, &intr->src[row_index], 0, nir_type_int);

   nir_alu_type out_type = nir_intrinsic_dest_type(intr);
   enum overload_type overload = get_overload(out_type, intr->dest.ssa.bit_size);

   const struct dxil_func *func = dxil_get_function(&ctx->mod, func_name, overload);

   if (!func)
      return false;

   nir_variable *var = find_patch_matching_variable_by_driver_location(ctx->shader, nir_var_shader_in, nir_intrinsic_base(intr), is_patch_constant);
   unsigned var_base_component = var ? var->data.location_frac : 0;
   unsigned base_component = nir_intrinsic_component(intr) - var_base_component;

   if (ctx->mod.minor_validator >= 5 &&
       !is_output_control_point &&
       intr->intrinsic != nir_intrinsic_load_output) {
      struct dxil_signature_record *sig_rec = is_patch_constant ?
         &ctx->mod.patch_consts[nir_intrinsic_base(intr)] :
         &ctx->mod.inputs[ctx->mod.input_mappings[nir_intrinsic_base(intr)]];
      unsigned comp_size = intr->dest.ssa.bit_size == 64 ? 2 : 1;
      unsigned comp_mask = (1 << (intr->num_components * comp_size)) - 1;
      comp_mask <<= (var_base_component * comp_size);
      if (is_tess_level)
         comp_mask = 1;
      for (unsigned r = 0; r < sig_rec->num_elements; ++r)
         sig_rec->elements[r].always_reads_mask |= (comp_mask & sig_rec->elements[r].mask);

      if (!nir_src_is_const(intr->src[row_index])) {
         struct dxil_psv_signature_element *psv_rec = is_patch_constant ?
            &ctx->mod.psv_patch_consts[nir_intrinsic_base(intr)] :
            &ctx->mod.psv_inputs[ctx->mod.input_mappings[nir_intrinsic_base(intr)]];
         psv_rec->dynamic_mask_and_stream |= comp_mask;
      }
   }

   for (unsigned i = 0; i < intr->num_components; ++i) {
      if (is_tess_level)
         row = dxil_module_get_int32_const(&ctx->mod, i + base_component);
      else
         comp = dxil_module_get_int8_const(&ctx->mod, i + base_component);

      if (!row || !comp)
         return false;

      const struct dxil_value *args[] = {
         opcode, input_id, row, comp, vertex_id
      };

      unsigned num_args = ARRAY_SIZE(args) - (is_patch_constant ? 1 : 0);
      const struct dxil_value *retval = dxil_emit_call(&ctx->mod, func, args, num_args);
      if (!retval)
         return false;
      store_dest(ctx, &intr->dest, i, retval, out_type);
   }
   return true;
}

static bool
emit_load_interpolated_input(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   nir_intrinsic_instr *barycentric = nir_src_as_intrinsic(intr->src[0]);

   const struct dxil_value *args[6] = { 0 };

   unsigned opcode_val;
   const char *func_name;
   unsigned num_args;
   switch (barycentric->intrinsic) {
   case nir_intrinsic_load_barycentric_at_offset:
      opcode_val = DXIL_INTR_EVAL_SNAPPED;
      func_name = "dx.op.evalSnapped";
      num_args = 6;
      for (unsigned i = 0; i < 2; ++i) {
         const struct dxil_value *float_offset = get_src(ctx, &barycentric->src[0], i, nir_type_float);
         /* GLSL uses [-0.5f, 0.5f), DXIL uses (-8, 7) */
         const struct dxil_value *offset_16 = dxil_emit_binop(&ctx->mod,
            DXIL_BINOP_MUL, float_offset, dxil_module_get_float_const(&ctx->mod, 16.0f), 0);
         args[i + 4] = dxil_emit_cast(&ctx->mod, DXIL_CAST_FPTOSI,
            dxil_module_get_int_type(&ctx->mod, 32), offset_16);
      }
      break;
   case nir_intrinsic_load_barycentric_pixel:
      opcode_val = DXIL_INTR_EVAL_SNAPPED;
      func_name = "dx.op.evalSnapped";
      num_args = 6;
      args[4] = args[5] = dxil_module_get_int32_const(&ctx->mod, 0);
      break;
   case nir_intrinsic_load_barycentric_at_sample:
      opcode_val = DXIL_INTR_EVAL_SAMPLE_INDEX;
      func_name = "dx.op.evalSampleIndex";
      num_args = 5;
      args[4] = get_src(ctx, &barycentric->src[0], 0, nir_type_int);
      break;
   case nir_intrinsic_load_barycentric_centroid:
      opcode_val = DXIL_INTR_EVAL_CENTROID;
      func_name = "dx.op.evalCentroid";
      num_args = 4;
      break;
   default:
      unreachable("Unsupported interpolation barycentric intrinsic");
   }
   args[0] = dxil_module_get_int32_const(&ctx->mod, opcode_val);
   args[1] = dxil_module_get_int32_const(&ctx->mod, nir_intrinsic_base(intr));
   args[2] = get_src(ctx, &intr->src[1], 0, nir_type_int);

   const struct dxil_func *func = dxil_get_function(&ctx->mod, func_name, DXIL_F32);

   if (!func)
      return false;

   nir_variable *var = find_patch_matching_variable_by_driver_location(ctx->shader, nir_var_shader_in, nir_intrinsic_base(intr), false);
   unsigned var_base_component = var ? var->data.location_frac : 0;
   unsigned base_component = nir_intrinsic_component(intr) - var_base_component;

   if (ctx->mod.minor_validator >= 5) {
      struct dxil_signature_record *sig_rec =
         &ctx->mod.inputs[ctx->mod.input_mappings[nir_intrinsic_base(intr)]];
      unsigned comp_size = intr->dest.ssa.bit_size == 64 ? 2 : 1;
      unsigned comp_mask = (1 << (intr->num_components * comp_size)) - 1;
      comp_mask <<= (var_base_component * comp_size);
      for (unsigned r = 0; r < sig_rec->num_elements; ++r)
         sig_rec->elements[r].always_reads_mask |= (comp_mask & sig_rec->elements[r].mask);

      if (!nir_src_is_const(intr->src[1])) {
         struct dxil_psv_signature_element *psv_rec =
            &ctx->mod.psv_inputs[ctx->mod.input_mappings[nir_intrinsic_base(intr)]];
         psv_rec->dynamic_mask_and_stream |= comp_mask;
      }
   }

   for (unsigned i = 0; i < intr->num_components; ++i) {
      args[3] = dxil_module_get_int8_const(&ctx->mod, i + base_component);

      const struct dxil_value *retval = dxil_emit_call(&ctx->mod, func, args, num_args);
      if (!retval)
         return false;
      store_dest(ctx, &intr->dest, i, retval, nir_type_float);
   }
   return true;
}

static bool
emit_load_ptr(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   struct nir_variable *var =
      nir_deref_instr_get_variable(nir_src_as_deref(intr->src[0]));

   const struct dxil_value *index =
      get_src(ctx, &intr->src[1], 0, nir_type_uint);
   if (!index)
      return false;

   const struct dxil_value *ptr = emit_gep_for_index(ctx, var, index);
   if (!ptr)
      return false;

   const struct dxil_value *retval =
      dxil_emit_load(&ctx->mod, ptr, 4, false);
   if (!retval)
      return false;

   store_dest(ctx, &intr->dest, 0, retval, nir_type_uint);
   return true;
}

static bool
emit_load_shared(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   const struct dxil_value *zero, *index;
   unsigned bit_size = nir_dest_bit_size(intr->dest);
   unsigned align = bit_size / 8;

   /* All shared mem accesses should have been lowered to scalar 32bit
    * accesses.
    */
   assert(bit_size == 32);
   assert(nir_dest_num_components(intr->dest) == 1);

   zero = dxil_module_get_int32_const(&ctx->mod, 0);
   if (!zero)
      return false;

   index = get_src(ctx, &intr->src[0], 0, nir_type_uint);
   if (!index)
      return false;

   const struct dxil_value *ops[] = { ctx->sharedvars, zero, index };
   const struct dxil_value *ptr, *retval;

   ptr = dxil_emit_gep_inbounds(&ctx->mod, ops, ARRAY_SIZE(ops));
   if (!ptr)
      return false;

   retval = dxil_emit_load(&ctx->mod, ptr, align, false);
   if (!retval)
      return false;

   store_dest(ctx, &intr->dest, 0, retval, nir_type_uint);
   return true;
}

static bool
emit_load_scratch(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   const struct dxil_value *zero, *index;
   unsigned bit_size = nir_dest_bit_size(intr->dest);
   unsigned align = bit_size / 8;

   /* All scratch mem accesses should have been lowered to scalar 32bit
    * accesses.
    */
   assert(bit_size == 32);
   assert(nir_dest_num_components(intr->dest) == 1);

   zero = dxil_module_get_int32_const(&ctx->mod, 0);
   if (!zero)
      return false;

   index = get_src(ctx, &intr->src[0], 0, nir_type_uint);
   if (!index)
      return false;

   const struct dxil_value *ops[] = { ctx->scratchvars, zero, index };
   const struct dxil_value *ptr, *retval;

   ptr = dxil_emit_gep_inbounds(&ctx->mod, ops, ARRAY_SIZE(ops));
   if (!ptr)
      return false;

   retval = dxil_emit_load(&ctx->mod, ptr, align, false);
   if (!retval)
      return false;

   store_dest(ctx, &intr->dest, 0, retval, nir_type_uint);
   return true;
}

static bool
emit_discard_if_with_value(struct ntd_context *ctx, const struct dxil_value *value)
{
   const struct dxil_value *opcode = dxil_module_get_int32_const(&ctx->mod, DXIL_INTR_DISCARD);
   if (!opcode)
      return false;

   const struct dxil_value *args[] = {
     opcode,
     value
   };

   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.discard", DXIL_NONE);
   if (!func)
      return false;

   return dxil_emit_call_void(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static bool
emit_discard_if(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   const struct dxil_value *value = get_src(ctx, &intr->src[0], 0, nir_type_bool);
   if (!value)
      return false;

   return emit_discard_if_with_value(ctx, value);
}

static bool
emit_discard(struct ntd_context *ctx)
{
   const struct dxil_value *value = dxil_module_get_int1_const(&ctx->mod, true);
   return emit_discard_if_with_value(ctx, value);
}

static bool
emit_emit_vertex(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   const struct dxil_value *opcode = dxil_module_get_int32_const(&ctx->mod, DXIL_INTR_EMIT_STREAM);
   const struct dxil_value *stream_id = dxil_module_get_int8_const(&ctx->mod, nir_intrinsic_stream_id(intr));
   if (!opcode || !stream_id)
      return false;

   const struct dxil_value *args[] = {
     opcode,
     stream_id
   };

   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.emitStream", DXIL_NONE);
   if (!func)
      return false;

   return dxil_emit_call_void(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static bool
emit_end_primitive(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   const struct dxil_value *opcode = dxil_module_get_int32_const(&ctx->mod, DXIL_INTR_CUT_STREAM);
   const struct dxil_value *stream_id = dxil_module_get_int8_const(&ctx->mod, nir_intrinsic_stream_id(intr));
   if (!opcode || !stream_id)
      return false;

   const struct dxil_value *args[] = {
     opcode,
     stream_id
   };

   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.cutStream", DXIL_NONE);
   if (!func)
      return false;

   return dxil_emit_call_void(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static bool
emit_image_store(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   const struct dxil_value *handle = get_resource_handle(ctx, &intr->src[0], DXIL_RESOURCE_CLASS_UAV, DXIL_RESOURCE_KIND_TEXTURE2D);
   if (!handle)
      return false;

   bool is_array = false;
   if (intr->intrinsic == nir_intrinsic_image_deref_store)
      is_array = glsl_sampler_type_is_array(nir_src_as_deref(intr->src[0])->type);
   else
      is_array = nir_intrinsic_image_array(intr);

   const struct dxil_value *int32_undef = get_int32_undef(&ctx->mod);
   if (!int32_undef)
      return false;

   const struct dxil_value *coord[3] = { int32_undef, int32_undef, int32_undef };
   enum glsl_sampler_dim image_dim = intr->intrinsic == nir_intrinsic_image_store ?
      nir_intrinsic_image_dim(intr) :
      glsl_get_sampler_dim(nir_src_as_deref(intr->src[0])->type);
   unsigned num_coords = glsl_get_sampler_dim_coordinate_components(image_dim);
   if (is_array)
      ++num_coords;

   assert(num_coords <= nir_src_num_components(intr->src[1]));
   for (unsigned i = 0; i < num_coords; ++i) {
      coord[i] = get_src(ctx, &intr->src[1], i, nir_type_uint);
      if (!coord[i])
         return false;
   }

   nir_alu_type in_type = nir_intrinsic_src_type(intr);
   enum overload_type overload = get_overload(in_type, 32);

   assert(nir_src_bit_size(intr->src[3]) == 32);
   unsigned num_components = nir_src_num_components(intr->src[3]);
   assert(num_components <= 4);
   const struct dxil_value *value[4];
   for (unsigned i = 0; i < num_components; ++i) {
      value[i] = get_src(ctx, &intr->src[3], i, in_type);
      if (!value[i])
         return false;
   }

   for (int i = num_components; i < 4; ++i)
      value[i] = int32_undef;

   const struct dxil_value *write_mask =
      dxil_module_get_int8_const(&ctx->mod, (1u << num_components) - 1);
   if (!write_mask)
      return false;

   if (image_dim == GLSL_SAMPLER_DIM_BUF) {
      coord[1] = int32_undef;
      return emit_bufferstore_call(ctx, handle, coord, value, write_mask, overload);
   } else
      return emit_texturestore_call(ctx, handle, coord, value, write_mask, overload);
}

static bool
emit_image_load(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   const struct dxil_value *handle = get_resource_handle(ctx, &intr->src[0], DXIL_RESOURCE_CLASS_UAV, DXIL_RESOURCE_KIND_TEXTURE2D);
   if (!handle)
      return false;

   bool is_array = false;
   if (intr->intrinsic == nir_intrinsic_image_deref_load)
      is_array = glsl_sampler_type_is_array(nir_src_as_deref(intr->src[0])->type);
   else
      is_array = nir_intrinsic_image_array(intr);

   const struct dxil_value *int32_undef = get_int32_undef(&ctx->mod);
   if (!int32_undef)
      return false;

   const struct dxil_value *coord[3] = { int32_undef, int32_undef, int32_undef };
   enum glsl_sampler_dim image_dim = intr->intrinsic == nir_intrinsic_image_load ?
      nir_intrinsic_image_dim(intr) :
      glsl_get_sampler_dim(nir_src_as_deref(intr->src[0])->type);
   unsigned num_coords = glsl_get_sampler_dim_coordinate_components(image_dim);
   if (is_array)
      ++num_coords;

   assert(num_coords <= nir_src_num_components(intr->src[1]));
   for (unsigned i = 0; i < num_coords; ++i) {
      coord[i] = get_src(ctx, &intr->src[1], i, nir_type_uint);
      if (!coord[i])
         return false;
   }

   nir_alu_type out_type = nir_intrinsic_dest_type(intr);
   enum overload_type overload = get_overload(out_type, 32);

   const struct dxil_value *load_result;
   if (image_dim == GLSL_SAMPLER_DIM_BUF) {
      coord[1] = int32_undef;
      load_result = emit_bufferload_call(ctx, handle, coord, overload);
   } else
      load_result = emit_textureload_call(ctx, handle, coord, overload);

   if (!load_result)
      return false;

   assert(nir_dest_bit_size(intr->dest) == 32);
   unsigned num_components = nir_dest_num_components(intr->dest);
   assert(num_components <= 4);
   for (unsigned i = 0; i < num_components; ++i) {
      const struct dxil_value *component = dxil_emit_extractval(&ctx->mod, load_result, i);
      if (!component)
         return false;
      store_dest(ctx, &intr->dest, i, component, out_type);
   }

   /* FIXME: This flag should be set to true when the RWTexture is attached
    * a vector, and we always declare a vec4 right now, so it should always be
    * true. Might be worth reworking the dxil_module_get_res_type() to use a
    * scalar when the image only has one component.
    */
   ctx->mod.feats.typed_uav_load_additional_formats = true;

   return true;
}

static bool
emit_image_atomic(struct ntd_context *ctx, nir_intrinsic_instr *intr,
                  enum dxil_atomic_op op, nir_alu_type type)
{
   const struct dxil_value *handle = get_resource_handle(ctx, &intr->src[0], DXIL_RESOURCE_CLASS_UAV, DXIL_RESOURCE_KIND_TEXTURE2D);
   if (!handle)
      return false;

   bool is_array = false;
   nir_deref_instr *src_as_deref = nir_src_as_deref(intr->src[0]);
   if (src_as_deref)
      is_array = glsl_sampler_type_is_array(src_as_deref->type);
   else
      is_array = nir_intrinsic_image_array(intr);

   const struct dxil_value *int32_undef = get_int32_undef(&ctx->mod);
   if (!int32_undef)
      return false;

   const struct dxil_value *coord[3] = { int32_undef, int32_undef, int32_undef };
   enum glsl_sampler_dim image_dim = src_as_deref ?
      glsl_get_sampler_dim(src_as_deref->type) :
      nir_intrinsic_image_dim(intr);
   unsigned num_coords = glsl_get_sampler_dim_coordinate_components(image_dim);
   if (is_array)
      ++num_coords;

   assert(num_coords <= nir_src_num_components(intr->src[1]));
   for (unsigned i = 0; i < num_coords; ++i) {
      coord[i] = get_src(ctx, &intr->src[1], i, nir_type_uint);
      if (!coord[i])
         return false;
   }

   const struct dxil_value *value = get_src(ctx, &intr->src[3], 0, type);
   if (!value)
      return false;

   const struct dxil_value *retval =
      emit_atomic_binop(ctx, handle, op, coord, value);

   if (!retval)
      return false;

   store_dest(ctx, &intr->dest, 0, retval, type);
   return true;
}

static bool
emit_image_atomic_comp_swap(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   const struct dxil_value *handle = get_resource_handle(ctx, &intr->src[0], DXIL_RESOURCE_CLASS_UAV, DXIL_RESOURCE_KIND_TEXTURE2D);
   if (!handle)
      return false;

   bool is_array = false;
   if (intr->intrinsic == nir_intrinsic_image_deref_atomic_comp_swap)
      is_array = glsl_sampler_type_is_array(nir_src_as_deref(intr->src[0])->type);
   else
      is_array = nir_intrinsic_image_array(intr);

   const struct dxil_value *int32_undef = get_int32_undef(&ctx->mod);
   if (!int32_undef)
      return false;

   const struct dxil_value *coord[3] = { int32_undef, int32_undef, int32_undef };
   enum glsl_sampler_dim image_dim = intr->intrinsic == nir_intrinsic_image_atomic_comp_swap ?
      nir_intrinsic_image_dim(intr) :
      glsl_get_sampler_dim(nir_src_as_deref(intr->src[0])->type);
   unsigned num_coords = glsl_get_sampler_dim_coordinate_components(image_dim);
   if (is_array)
      ++num_coords;

   assert(num_coords <= nir_src_num_components(intr->src[1]));
   for (unsigned i = 0; i < num_coords; ++i) {
      coord[i] = get_src(ctx, &intr->src[1], i, nir_type_uint);
      if (!coord[i])
         return false;
   }

   const struct dxil_value *cmpval = get_src(ctx, &intr->src[3], 0, nir_type_uint);
   const struct dxil_value *newval = get_src(ctx, &intr->src[4], 0, nir_type_uint);
   if (!cmpval || !newval)
      return false;

   const struct dxil_value *retval =
      emit_atomic_cmpxchg(ctx, handle, coord, cmpval, newval);

   if (!retval)
      return false;

   store_dest(ctx, &intr->dest, 0, retval, nir_type_uint);
   return true;
}

struct texop_parameters {
   const struct dxil_value *tex;
   const struct dxil_value *sampler;
   const struct dxil_value *bias, *lod_or_sample, *min_lod;
   const struct dxil_value *coord[4], *offset[3], *dx[3], *dy[3];
   const struct dxil_value *cmp;
   enum overload_type overload;
};

static const struct dxil_value *
emit_texture_size(struct ntd_context *ctx, struct texop_parameters *params)
{
   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.getDimensions", DXIL_NONE);
   if (!func)
      return false;

   const struct dxil_value *args[] = {
      dxil_module_get_int32_const(&ctx->mod, DXIL_INTR_TEXTURE_SIZE),
      params->tex,
      params->lod_or_sample
   };

   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static bool
emit_image_size(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   const struct dxil_value *handle = get_resource_handle(ctx, &intr->src[0], DXIL_RESOURCE_CLASS_UAV, DXIL_RESOURCE_KIND_TEXTURE2D);
   if (!handle)
      return false;

   const struct dxil_value *lod = get_src(ctx, &intr->src[1], 0, nir_type_uint);
   if (!lod)
      return false;

   struct texop_parameters params = {
      .tex = handle,
      .lod_or_sample = lod
   };
   const struct dxil_value *dimensions = emit_texture_size(ctx, &params);
   if (!dimensions)
      return false;

   for (unsigned i = 0; i < nir_dest_num_components(intr->dest); ++i) {
      const struct dxil_value *retval = dxil_emit_extractval(&ctx->mod, dimensions, i);
      store_dest(ctx, &intr->dest, i, retval, nir_type_uint);
   }

   return true;
}

static bool
emit_get_ssbo_size(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   enum dxil_resource_class class = DXIL_RESOURCE_CLASS_UAV;
   if (ctx->opts->environment == DXIL_ENVIRONMENT_VULKAN) {
      nir_variable *var = nir_get_binding_variable(ctx->shader, nir_chase_binding(intr->src[0]));
      if (var && var->data.access & ACCESS_NON_WRITEABLE)
         class = DXIL_RESOURCE_CLASS_SRV;
   }

   const struct dxil_value *handle = get_resource_handle(ctx, &intr->src[0], class, DXIL_RESOURCE_KIND_RAW_BUFFER);
   if (!handle)
      return false;

   struct texop_parameters params = {
      .tex = handle,
      .lod_or_sample = dxil_module_get_undef(
                        &ctx->mod, dxil_module_get_int_type(&ctx->mod, 32))
   };

   const struct dxil_value *dimensions = emit_texture_size(ctx, &params);
   if (!dimensions)
      return false;

   const struct dxil_value *retval = dxil_emit_extractval(&ctx->mod, dimensions, 0);
   store_dest(ctx, &intr->dest, 0, retval, nir_type_uint);

   return true;
}

static bool
emit_ssbo_atomic(struct ntd_context *ctx, nir_intrinsic_instr *intr,
                   enum dxil_atomic_op op, nir_alu_type type)
{
   const struct dxil_value* handle = get_resource_handle(ctx, &intr->src[0], DXIL_RESOURCE_CLASS_UAV, DXIL_RESOURCE_KIND_RAW_BUFFER);
   const struct dxil_value *offset =
      get_src(ctx, &intr->src[1], 0, nir_type_uint);
   const struct dxil_value *value =
      get_src(ctx, &intr->src[2], 0, type);

   if (!value || !handle || !offset)
      return false;

   const struct dxil_value *int32_undef = get_int32_undef(&ctx->mod);
   if (!int32_undef)
      return false;

   const struct dxil_value *coord[3] = {
      offset, int32_undef, int32_undef
   };

   const struct dxil_value *retval =
      emit_atomic_binop(ctx, handle, op, coord, value);

   if (!retval)
      return false;

   store_dest(ctx, &intr->dest, 0, retval, type);
   return true;
}

static bool
emit_ssbo_atomic_comp_swap(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   const struct dxil_value* handle = get_resource_handle(ctx, &intr->src[0], DXIL_RESOURCE_CLASS_UAV, DXIL_RESOURCE_KIND_RAW_BUFFER);
   const struct dxil_value *offset =
      get_src(ctx, &intr->src[1], 0, nir_type_uint);
   const struct dxil_value *cmpval =
      get_src(ctx, &intr->src[2], 0, nir_type_int);
   const struct dxil_value *newval =
      get_src(ctx, &intr->src[3], 0, nir_type_int);

   if (!cmpval || !newval || !handle || !offset)
      return false;

   const struct dxil_value *int32_undef = get_int32_undef(&ctx->mod);
   if (!int32_undef)
      return false;

   const struct dxil_value *coord[3] = {
      offset, int32_undef, int32_undef
   };

   const struct dxil_value *retval =
      emit_atomic_cmpxchg(ctx, handle, coord, cmpval, newval);

   if (!retval)
      return false;

   store_dest(ctx, &intr->dest, 0, retval, nir_type_int);
   return true;
}

static bool
emit_shared_atomic(struct ntd_context *ctx, nir_intrinsic_instr *intr,
                   enum dxil_rmw_op op, nir_alu_type type)
{
   const struct dxil_value *zero, *index;

   assert(nir_src_bit_size(intr->src[1]) == 32);

   zero = dxil_module_get_int32_const(&ctx->mod, 0);
   if (!zero)
      return false;

   index = get_src(ctx, &intr->src[0], 0, nir_type_uint);
   if (!index)
      return false;

   const struct dxil_value *ops[] = { ctx->sharedvars, zero, index };
   const struct dxil_value *ptr, *value, *retval;

   ptr = dxil_emit_gep_inbounds(&ctx->mod, ops, ARRAY_SIZE(ops));
   if (!ptr)
      return false;

   value = get_src(ctx, &intr->src[1], 0, type);
   if (!value)
      return false;

   retval = dxil_emit_atomicrmw(&ctx->mod, value, ptr, op, false,
                                DXIL_ATOMIC_ORDERING_ACQREL,
                                DXIL_SYNC_SCOPE_CROSSTHREAD);
   if (!retval)
      return false;

   store_dest(ctx, &intr->dest, 0, retval, type);
   return true;
}

static bool
emit_shared_atomic_comp_swap(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   const struct dxil_value *zero, *index;

   assert(nir_src_bit_size(intr->src[1]) == 32);

   zero = dxil_module_get_int32_const(&ctx->mod, 0);
   if (!zero)
      return false;

   index = get_src(ctx, &intr->src[0], 0, nir_type_uint);
   if (!index)
      return false;

   const struct dxil_value *ops[] = { ctx->sharedvars, zero, index };
   const struct dxil_value *ptr, *cmpval, *newval, *retval;

   ptr = dxil_emit_gep_inbounds(&ctx->mod, ops, ARRAY_SIZE(ops));
   if (!ptr)
      return false;

   cmpval = get_src(ctx, &intr->src[1], 0, nir_type_uint);
   newval = get_src(ctx, &intr->src[2], 0, nir_type_uint);
   if (!cmpval || !newval)
      return false;

   retval = dxil_emit_cmpxchg(&ctx->mod, cmpval, newval, ptr, false,
                              DXIL_ATOMIC_ORDERING_ACQREL,
                              DXIL_SYNC_SCOPE_CROSSTHREAD);
   if (!retval)
      return false;

   store_dest(ctx, &intr->dest, 0, retval, nir_type_uint);
   return true;
}

static bool
emit_vulkan_resource_index(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   unsigned int binding = nir_intrinsic_binding(intr);

   bool const_index = nir_src_is_const(intr->src[0]);
   if (const_index) {
      binding += nir_src_as_const_value(intr->src[0])->u32;
   }

   const struct dxil_value *index_value = dxil_module_get_int32_const(&ctx->mod, binding);
   if (!index_value)
      return false;

   if (!const_index) {
      const struct dxil_value *offset = get_src(ctx, &intr->src[0], 0, nir_type_uint32);
      if (!offset)
         return false;

      index_value = dxil_emit_binop(&ctx->mod, DXIL_BINOP_ADD, index_value, offset, 0);
      if (!index_value)
         return false;
   }

   store_dest(ctx, &intr->dest, 0, index_value, nir_type_uint32);
   store_dest(ctx, &intr->dest, 1, dxil_module_get_int32_const(&ctx->mod, 0), nir_type_uint32);
   return true;
}

static bool
emit_load_vulkan_descriptor(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   nir_intrinsic_instr* index = nir_src_as_intrinsic(intr->src[0]);
   /* We currently do not support reindex */
   assert(index && index->intrinsic == nir_intrinsic_vulkan_resource_index);

   unsigned binding = nir_intrinsic_binding(index);
   unsigned space = nir_intrinsic_desc_set(index);

   /* The descriptor_set field for variables is only 5 bits. We shouldn't have intrinsics trying to go beyond that. */
   assert(space < 32);

   nir_variable *var = nir_get_binding_variable(ctx->shader, nir_chase_binding(intr->src[0]));

   const struct dxil_value *handle = NULL;
   enum dxil_resource_class resource_class;

   switch (nir_intrinsic_desc_type(intr)) {
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      resource_class = DXIL_RESOURCE_CLASS_CBV;
      break;
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      if (var->data.access & ACCESS_NON_WRITEABLE)
         resource_class = DXIL_RESOURCE_CLASS_SRV;
      else
         resource_class = DXIL_RESOURCE_CLASS_UAV;
      break;
   default:
      unreachable("unknown descriptor type");
      return false;
   }

   const struct dxil_value *index_value = get_src(ctx, &intr->src[0], 0, nir_type_uint32);
   if (!index_value)
      return false;

   handle = emit_createhandle_call_dynamic(ctx, resource_class, space, binding, index_value, false);

   store_dest_value(ctx, &intr->dest, 0, handle);
   store_dest(ctx, &intr->dest, 1, get_src(ctx, &intr->src[0], 1, nir_type_uint32), nir_type_uint32);

   return true;
}

static bool
emit_load_sample_pos_from_id(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.renderTargetGetSamplePosition", DXIL_NONE);
   if (!func)
      return false;

   const struct dxil_value *opcode = dxil_module_get_int32_const(&ctx->mod, DXIL_INTR_RENDER_TARGET_GET_SAMPLE_POSITION);
   if (!opcode)
      return false;

   const struct dxil_value *args[] = {
      opcode,
      get_src(ctx, &intr->src[0], 0, nir_type_uint32),
   };
   if (!args[1])
      return false;

   const struct dxil_value *v = dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
   if (!v)
      return false;

   for (unsigned i = 0; i < 2; ++i) {
      /* GL coords go from 0 -> 1, D3D from -0.5 -> 0.5 */
      const struct dxil_value *coord = dxil_emit_binop(&ctx->mod, DXIL_BINOP_ADD,
         dxil_emit_extractval(&ctx->mod, v, i),
         dxil_module_get_float_const(&ctx->mod, 0.5f), 0);
      store_dest(ctx, &intr->dest, i, coord, nir_type_float32);
   }
   return true;
}

static bool
emit_load_sample_id(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   assert(ctx->mod.info.has_per_sample_input ||
          intr->intrinsic == nir_intrinsic_load_sample_id_no_per_sample);

   if (ctx->mod.info.has_per_sample_input)
      return emit_load_unary_external_function(ctx, intr, "dx.op.sampleIndex",
                                               DXIL_INTR_SAMPLE_INDEX, DXIL_I32);

   store_dest_value(ctx, &intr->dest, 0, dxil_module_get_int32_const(&ctx->mod, 0));
   return true;
}

static bool
emit_read_first_invocation(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   ctx->mod.feats.wave_ops = 1;
   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.waveReadLaneFirst",
                                                    get_overload(nir_type_uint, intr->dest.ssa.bit_size));
   const struct dxil_value *args[] = {
      dxil_module_get_int32_const(&ctx->mod, DXIL_INTR_WAVE_READ_LANE_FIRST),
      get_src(ctx, intr->src, 0, nir_type_uint),
   };
   if (!func || !args[0] || !args[1])
      return false;

   const struct dxil_value *ret = dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
   if (!ret)
      return false;
   store_dest_value(ctx, &intr->dest, 0, ret);
   return true;
}

static bool
emit_intrinsic(struct ntd_context *ctx, nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_global_invocation_id:
   case nir_intrinsic_load_global_invocation_id_zero_base:
      return emit_load_global_invocation_id(ctx, intr);
   case nir_intrinsic_load_local_invocation_id:
      return emit_load_local_invocation_id(ctx, intr);
   case nir_intrinsic_load_local_invocation_index:
      return emit_load_local_invocation_index(ctx, intr);
   case nir_intrinsic_load_workgroup_id:
   case nir_intrinsic_load_workgroup_id_zero_base:
      return emit_load_local_workgroup_id(ctx, intr);
   case nir_intrinsic_load_ssbo:
      return emit_load_ssbo(ctx, intr);
   case nir_intrinsic_store_ssbo:
      return emit_store_ssbo(ctx, intr);
   case nir_intrinsic_store_ssbo_masked_dxil:
      return emit_store_ssbo_masked(ctx, intr);
   case nir_intrinsic_store_shared_dxil:
   case nir_intrinsic_store_shared_masked_dxil:
      return emit_store_shared(ctx, intr);
   case nir_intrinsic_store_scratch_dxil:
      return emit_store_scratch(ctx, intr);
   case nir_intrinsic_load_ptr_dxil:
      return emit_load_ptr(ctx, intr);
   case nir_intrinsic_load_ubo:
      return emit_load_ubo(ctx, intr);
   case nir_intrinsic_load_ubo_dxil:
      return emit_load_ubo_dxil(ctx, intr);
   case nir_intrinsic_load_primitive_id:
      return emit_load_unary_external_function(ctx, intr, "dx.op.primitiveID",
                                               DXIL_INTR_PRIMITIVE_ID, DXIL_I32);
   case nir_intrinsic_load_sample_id:
   case nir_intrinsic_load_sample_id_no_per_sample:
      return emit_load_sample_id(ctx, intr);
   case nir_intrinsic_load_invocation_id:
      switch (ctx->mod.shader_kind) {
      case DXIL_HULL_SHADER:
         return emit_load_unary_external_function(ctx, intr, "dx.op.outputControlPointID",
                                                  DXIL_INTR_OUTPUT_CONTROL_POINT_ID, DXIL_I32);
      case DXIL_GEOMETRY_SHADER:
         return emit_load_unary_external_function(ctx, intr, "dx.op.gsInstanceID",
                                                  DXIL_INTR_GS_INSTANCE_ID, DXIL_I32);
      default:
         unreachable("Unexpected shader kind for invocation ID");
      }
   case nir_intrinsic_load_view_index:
      ctx->mod.feats.view_id = true;
      return emit_load_unary_external_function(ctx, intr, "dx.op.viewID",
                                               DXIL_INTR_VIEW_ID, DXIL_I32);
   case nir_intrinsic_load_sample_mask_in:
      return emit_load_sample_mask_in(ctx, intr);
   case nir_intrinsic_load_tess_coord:
      return emit_load_tess_coord(ctx, intr);
   case nir_intrinsic_load_shared_dxil:
      return emit_load_shared(ctx, intr);
   case nir_intrinsic_load_scratch_dxil:
      return emit_load_scratch(ctx, intr);
   case nir_intrinsic_discard_if:
   case nir_intrinsic_demote_if:
      return emit_discard_if(ctx, intr);
   case nir_intrinsic_discard:
   case nir_intrinsic_demote:
      return emit_discard(ctx);
   case nir_intrinsic_emit_vertex:
      return emit_emit_vertex(ctx, intr);
   case nir_intrinsic_end_primitive:
      return emit_end_primitive(ctx, intr);
   case nir_intrinsic_scoped_barrier:
      return emit_barrier(ctx, intr);
   case nir_intrinsic_memory_barrier:
   case nir_intrinsic_memory_barrier_buffer:
   case nir_intrinsic_memory_barrier_image:
   case nir_intrinsic_memory_barrier_atomic_counter:
      return emit_memory_barrier(ctx, intr);
   case nir_intrinsic_memory_barrier_shared:
      return emit_memory_barrier_shared(ctx, intr);
   case nir_intrinsic_group_memory_barrier:
      return emit_group_memory_barrier(ctx, intr);
   case nir_intrinsic_control_barrier:
      return emit_control_barrier(ctx, intr);
   case nir_intrinsic_ssbo_atomic_add:
      return emit_ssbo_atomic(ctx, intr, DXIL_ATOMIC_ADD, nir_type_int);
   case nir_intrinsic_ssbo_atomic_imin:
      return emit_ssbo_atomic(ctx, intr, DXIL_ATOMIC_IMIN, nir_type_int);
   case nir_intrinsic_ssbo_atomic_umin:
      return emit_ssbo_atomic(ctx, intr, DXIL_ATOMIC_UMIN, nir_type_uint);
   case nir_intrinsic_ssbo_atomic_imax:
      return emit_ssbo_atomic(ctx, intr, DXIL_ATOMIC_IMAX, nir_type_int);
   case nir_intrinsic_ssbo_atomic_umax:
      return emit_ssbo_atomic(ctx, intr, DXIL_ATOMIC_UMAX, nir_type_uint);
   case nir_intrinsic_ssbo_atomic_and:
      return emit_ssbo_atomic(ctx, intr, DXIL_ATOMIC_AND, nir_type_uint);
   case nir_intrinsic_ssbo_atomic_or:
      return emit_ssbo_atomic(ctx, intr, DXIL_ATOMIC_OR, nir_type_uint);
   case nir_intrinsic_ssbo_atomic_xor:
      return emit_ssbo_atomic(ctx, intr, DXIL_ATOMIC_XOR, nir_type_uint);
   case nir_intrinsic_ssbo_atomic_exchange:
      return emit_ssbo_atomic(ctx, intr, DXIL_ATOMIC_EXCHANGE, nir_type_int);
   case nir_intrinsic_ssbo_atomic_comp_swap:
      return emit_ssbo_atomic_comp_swap(ctx, intr);
   case nir_intrinsic_shared_atomic_add_dxil:
      return emit_shared_atomic(ctx, intr, DXIL_RMWOP_ADD, nir_type_int);
   case nir_intrinsic_shared_atomic_imin_dxil:
      return emit_shared_atomic(ctx, intr, DXIL_RMWOP_MIN, nir_type_int);
   case nir_intrinsic_shared_atomic_umin_dxil:
      return emit_shared_atomic(ctx, intr, DXIL_RMWOP_UMIN, nir_type_uint);
   case nir_intrinsic_shared_atomic_imax_dxil:
      return emit_shared_atomic(ctx, intr, DXIL_RMWOP_MAX, nir_type_int);
   case nir_intrinsic_shared_atomic_umax_dxil:
      return emit_shared_atomic(ctx, intr, DXIL_RMWOP_UMAX, nir_type_uint);
   case nir_intrinsic_shared_atomic_and_dxil:
      return emit_shared_atomic(ctx, intr, DXIL_RMWOP_AND, nir_type_uint);
   case nir_intrinsic_shared_atomic_or_dxil:
      return emit_shared_atomic(ctx, intr, DXIL_RMWOP_OR, nir_type_uint);
   case nir_intrinsic_shared_atomic_xor_dxil:
      return emit_shared_atomic(ctx, intr, DXIL_RMWOP_XOR, nir_type_uint);
   case nir_intrinsic_shared_atomic_exchange_dxil:
      return emit_shared_atomic(ctx, intr, DXIL_RMWOP_XCHG, nir_type_int);
   case nir_intrinsic_shared_atomic_comp_swap_dxil:
      return emit_shared_atomic_comp_swap(ctx, intr);
   case nir_intrinsic_image_deref_atomic_add:
   case nir_intrinsic_image_atomic_add:
      return emit_image_atomic(ctx, intr, DXIL_ATOMIC_ADD, nir_type_int);
   case nir_intrinsic_image_deref_atomic_imin:
   case nir_intrinsic_image_atomic_imin:
      return emit_image_atomic(ctx, intr, DXIL_ATOMIC_IMIN, nir_type_int);
   case nir_intrinsic_image_deref_atomic_umin:
   case nir_intrinsic_image_atomic_umin:
      return emit_image_atomic(ctx, intr, DXIL_ATOMIC_UMIN, nir_type_uint);
   case nir_intrinsic_image_deref_atomic_imax:
   case nir_intrinsic_image_atomic_imax:
      return emit_image_atomic(ctx, intr, DXIL_ATOMIC_IMAX, nir_type_int);
   case nir_intrinsic_image_deref_atomic_umax:
   case nir_intrinsic_image_atomic_umax:
      return emit_image_atomic(ctx, intr, DXIL_ATOMIC_IMAX, nir_type_uint);
   case nir_intrinsic_image_deref_atomic_and:
   case nir_intrinsic_image_atomic_and:
      return emit_image_atomic(ctx, intr, DXIL_ATOMIC_AND, nir_type_uint);
   case nir_intrinsic_image_deref_atomic_or:
   case nir_intrinsic_image_atomic_or:
      return emit_image_atomic(ctx, intr, DXIL_ATOMIC_OR, nir_type_uint);
   case nir_intrinsic_image_deref_atomic_xor:
   case nir_intrinsic_image_atomic_xor:
      return emit_image_atomic(ctx, intr, DXIL_ATOMIC_XOR, nir_type_uint);
   case nir_intrinsic_image_deref_atomic_exchange:
   case nir_intrinsic_image_atomic_exchange:
      return emit_image_atomic(ctx, intr, DXIL_ATOMIC_EXCHANGE, nir_type_uint);
   case nir_intrinsic_image_deref_atomic_comp_swap:
   case nir_intrinsic_image_atomic_comp_swap:
      return emit_image_atomic_comp_swap(ctx, intr);
   case nir_intrinsic_image_store:
   case nir_intrinsic_image_deref_store:
      return emit_image_store(ctx, intr);
   case nir_intrinsic_image_load:
   case nir_intrinsic_image_deref_load:
      return emit_image_load(ctx, intr);
   case nir_intrinsic_image_size:
   case nir_intrinsic_image_deref_size:
      return emit_image_size(ctx, intr);
   case nir_intrinsic_get_ssbo_size:
      return emit_get_ssbo_size(ctx, intr);
   case nir_intrinsic_load_input:
   case nir_intrinsic_load_per_vertex_input:
   case nir_intrinsic_load_output:
   case nir_intrinsic_load_per_vertex_output:
      return emit_load_input_via_intrinsic(ctx, intr);
   case nir_intrinsic_store_output:
   case nir_intrinsic_store_per_vertex_output:
      return emit_store_output_via_intrinsic(ctx, intr);

   case nir_intrinsic_load_barycentric_at_offset:
   case nir_intrinsic_load_barycentric_at_sample:
   case nir_intrinsic_load_barycentric_centroid:
   case nir_intrinsic_load_barycentric_pixel:
      /* Emit nothing, we only support these as inputs to load_interpolated_input */
      return true;
   case nir_intrinsic_load_interpolated_input:
      return emit_load_interpolated_input(ctx, intr);
      break;

   case nir_intrinsic_vulkan_resource_index:
      return emit_vulkan_resource_index(ctx, intr);
   case nir_intrinsic_load_vulkan_descriptor:
      return emit_load_vulkan_descriptor(ctx, intr);

   case nir_intrinsic_load_sample_pos_from_id:
      return emit_load_sample_pos_from_id(ctx, intr);

   case nir_intrinsic_is_helper_invocation:
      return emit_load_unary_external_function(
         ctx, intr, "dx.op.isHelperLane", DXIL_INTR_IS_HELPER_LANE, DXIL_I32);
   case nir_intrinsic_elect:
      ctx->mod.feats.wave_ops = 1;
      return emit_load_unary_external_function(
         ctx, intr, "dx.op.waveIsFirstLane", DXIL_INTR_WAVE_IS_FIRST_LANE, DXIL_NONE);
   case nir_intrinsic_load_subgroup_size:
      ctx->mod.feats.wave_ops = 1;
      return emit_load_unary_external_function(
         ctx, intr, "dx.op.waveGetLaneCount", DXIL_INTR_WAVE_GET_LANE_COUNT, DXIL_NONE);
   case nir_intrinsic_load_subgroup_invocation:
      ctx->mod.feats.wave_ops = 1;
      return emit_load_unary_external_function(
         ctx, intr, "dx.op.waveGetLaneIndex", DXIL_INTR_WAVE_GET_LANE_INDEX, DXIL_NONE);

   case nir_intrinsic_read_first_invocation:
      return emit_read_first_invocation(ctx, intr);

   case nir_intrinsic_load_constant_non_opt:
      const struct dxil_value* value = get_src(ctx, &intr->src[0], 0, nir_type_uint);
      store_dest_value(ctx, &intr->dest, 0, value);
      return true;

   case nir_intrinsic_load_num_workgroups:
   case nir_intrinsic_load_workgroup_size:
   default:
      log_nir_instr_unsupported(
         ctx->logger, "Unimplemented intrinsic instruction", &intr->instr);
      return false;
   }
}

static bool
emit_load_const(struct ntd_context *ctx, nir_load_const_instr *load_const)
{
   for (int i = 0; i < load_const->def.num_components; ++i) {
      const struct dxil_value *value;
      switch (load_const->def.bit_size) {
      case 1:
         value = dxil_module_get_int1_const(&ctx->mod,
                                            load_const->value[i].b);
         break;
      case 16:
         ctx->mod.feats.native_low_precision = true;
         value = dxil_module_get_int16_const(&ctx->mod,
                                             load_const->value[i].u16);
         break;
      case 32:
         value = dxil_module_get_int32_const(&ctx->mod,
                                             load_const->value[i].u32);
         break;
      case 64:
         ctx->mod.feats.int64_ops = true;
         value = dxil_module_get_int64_const(&ctx->mod,
                                             load_const->value[i].u64);
         break;
      default:
         unreachable("unexpected bit_size");
      }
      if (!value)
         return false;

      store_ssa_def(ctx, &load_const->def, i, value);
   }
   return true;
}

static bool
emit_deref(struct ntd_context* ctx, nir_deref_instr* instr)
{
   assert(instr->deref_type == nir_deref_type_var ||
          instr->deref_type == nir_deref_type_array);

   /* In the CL environment, there's nothing to emit. Any references to
    * derefs will emit the necessary logic to handle scratch/shared GEP addressing
    */
   if (ctx->opts->environment == DXIL_ENVIRONMENT_CL)
      return true;

   /* In the Vulkan environment, we don't have cached handles for textures or
    * samplers, so let's use the opportunity of walking through the derefs to
    * emit those.
    */
   nir_variable *var = nir_deref_instr_get_variable(instr);
   assert(var);

   if (!glsl_type_is_sampler(glsl_without_array(var->type)) &&
       !glsl_type_is_image(glsl_without_array(var->type)) &&
       !glsl_type_is_texture(glsl_without_array(var->type)))
      return true;

   const struct glsl_type *type = instr->type;
   const struct dxil_value *binding;
   unsigned binding_val = ctx->opts->environment == DXIL_ENVIRONMENT_GL ?
      var->data.driver_location : var->data.binding;

   if (instr->deref_type == nir_deref_type_var) {
      binding = dxil_module_get_int32_const(&ctx->mod, binding_val);
   } else {
      const struct dxil_value *base = get_src(ctx, &instr->parent, 0, nir_type_uint32);
      const struct dxil_value *offset = get_src(ctx, &instr->arr.index, 0, nir_type_uint32);
      if (!base || !offset)
         return false;

      if (glsl_type_is_array(instr->type)) {
         offset = dxil_emit_binop(&ctx->mod, DXIL_BINOP_MUL, offset,
            dxil_module_get_int32_const(&ctx->mod, glsl_get_aoa_size(instr->type)), 0);
         if (!offset)
            return false;
      }
      binding = dxil_emit_binop(&ctx->mod, DXIL_BINOP_ADD, base, offset, 0);
   }

   if (!binding)
      return false;

   /* Haven't finished chasing the deref chain yet, just store the value */
   if (glsl_type_is_array(type)) {
      store_dest(ctx, &instr->dest, 0, binding, nir_type_uint32);
      return true;
   }

   assert(glsl_type_is_sampler(type) || glsl_type_is_image(type) || glsl_type_is_texture(type));
   enum dxil_resource_class res_class;
   if (glsl_type_is_image(type)) {
      if (ctx->opts->environment == DXIL_ENVIRONMENT_VULKAN &&
          (var->data.access & ACCESS_NON_WRITEABLE))
         res_class = DXIL_RESOURCE_CLASS_SRV;
      else
         res_class = DXIL_RESOURCE_CLASS_UAV;
   } else if (glsl_type_is_sampler(type)) {
      res_class = DXIL_RESOURCE_CLASS_SAMPLER;
   } else {
      res_class = DXIL_RESOURCE_CLASS_SRV;
   }
   
   unsigned descriptor_set = ctx->opts->environment == DXIL_ENVIRONMENT_VULKAN ?
      var->data.descriptor_set : (glsl_type_is_image(type) ? 1 : 0);
   const struct dxil_value *handle = emit_createhandle_call_dynamic(ctx, res_class,
      descriptor_set, binding_val, binding, false);
   if (!handle)
      return false;

   store_dest_value(ctx, &instr->dest, 0, handle);
   return true;
}

static bool
emit_cond_branch(struct ntd_context *ctx, const struct dxil_value *cond,
                 int true_block, int false_block)
{
   assert(cond);
   assert(true_block >= 0);
   assert(false_block >= 0);
   return dxil_emit_branch(&ctx->mod, cond, true_block, false_block);
}

static bool
emit_branch(struct ntd_context *ctx, int block)
{
   assert(block >= 0);
   return dxil_emit_branch(&ctx->mod, NULL, block, -1);
}

static bool
emit_jump(struct ntd_context *ctx, nir_jump_instr *instr)
{
   switch (instr->type) {
   case nir_jump_break:
   case nir_jump_continue:
      assert(instr->instr.block->successors[0]);
      assert(!instr->instr.block->successors[1]);
      return emit_branch(ctx, instr->instr.block->successors[0]->index);

   default:
      unreachable("Unsupported jump type\n");
   }
}

struct phi_block {
   unsigned num_components;
   struct dxil_instr *comp[NIR_MAX_VEC_COMPONENTS];
};

static bool
emit_phi(struct ntd_context *ctx, nir_phi_instr *instr)
{
   unsigned bit_size = nir_dest_bit_size(instr->dest);
   const struct dxil_type *type = dxil_module_get_int_type(&ctx->mod,
                                                           bit_size);

   struct phi_block *vphi = ralloc(ctx->phis, struct phi_block);
   vphi->num_components = nir_dest_num_components(instr->dest);

   for (unsigned i = 0; i < vphi->num_components; ++i) {
      struct dxil_instr *phi = vphi->comp[i] = dxil_emit_phi(&ctx->mod, type);
      if (!phi)
         return false;
      store_dest_value(ctx, &instr->dest, i, dxil_instr_get_return_value(phi));
   }
   _mesa_hash_table_insert(ctx->phis, instr, vphi);
   return true;
}

static bool
fixup_phi(struct ntd_context *ctx, nir_phi_instr *instr,
          struct phi_block *vphi)
{
   const struct dxil_value *values[16];
   unsigned blocks[16];
   for (unsigned i = 0; i < vphi->num_components; ++i) {
      size_t num_incoming = 0;
      nir_foreach_phi_src(src, instr) {
         assert(src->src.is_ssa);
         const struct dxil_value *val = get_src_ssa(ctx, src->src.ssa, i);
         values[num_incoming] = val;
         blocks[num_incoming] = src->pred->index;
         ++num_incoming;
         if (num_incoming == ARRAY_SIZE(values)) {
            if (!dxil_phi_add_incoming(vphi->comp[i], values, blocks,
                                       num_incoming))
               return false;
            num_incoming = 0;
         }
      }
      if (num_incoming > 0 && !dxil_phi_add_incoming(vphi->comp[i], values,
                                                     blocks, num_incoming))
         return false;
   }
   return true;
}

static unsigned
get_n_src(struct ntd_context *ctx, const struct dxil_value **values,
          unsigned max_components, nir_tex_src *src, nir_alu_type type)
{
   unsigned num_components = nir_src_num_components(src->src);
   unsigned i = 0;

   assert(num_components <= max_components);

   for (i = 0; i < num_components; ++i) {
      values[i] = get_src(ctx, &src->src, i, type);
      if (!values[i])
         return 0;
   }

   return num_components;
}

#define PAD_SRC(ctx, array, components, undef) \
   for (unsigned i = components; i < ARRAY_SIZE(array); ++i) { \
      array[i] = undef; \
   }

static const struct dxil_value *
emit_sample(struct ntd_context *ctx, struct texop_parameters *params)
{
   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.sample", params->overload);
   if (!func)
      return NULL;

   const struct dxil_value *args[11] = {
      dxil_module_get_int32_const(&ctx->mod, DXIL_INTR_SAMPLE),
      params->tex, params->sampler,
      params->coord[0], params->coord[1], params->coord[2], params->coord[3],
      params->offset[0], params->offset[1], params->offset[2],
      params->min_lod
   };

   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static const struct dxil_value *
emit_sample_bias(struct ntd_context *ctx, struct texop_parameters *params)
{
   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.sampleBias", params->overload);
   if (!func)
      return NULL;

   assert(params->bias != NULL);

   const struct dxil_value *args[12] = {
      dxil_module_get_int32_const(&ctx->mod, DXIL_INTR_SAMPLE_BIAS),
      params->tex, params->sampler,
      params->coord[0], params->coord[1], params->coord[2], params->coord[3],
      params->offset[0], params->offset[1], params->offset[2],
      params->bias, params->min_lod
   };

   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static const struct dxil_value *
emit_sample_level(struct ntd_context *ctx, struct texop_parameters *params)
{
   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.sampleLevel", params->overload);
   if (!func)
      return NULL;

   assert(params->lod_or_sample != NULL);

   const struct dxil_value *args[11] = {
      dxil_module_get_int32_const(&ctx->mod, DXIL_INTR_SAMPLE_LEVEL),
      params->tex, params->sampler,
      params->coord[0], params->coord[1], params->coord[2], params->coord[3],
      params->offset[0], params->offset[1], params->offset[2],
      params->lod_or_sample
   };

   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static const struct dxil_value *
emit_sample_cmp(struct ntd_context *ctx, struct texop_parameters *params)
{
   const struct dxil_func *func;
   enum dxil_intr opcode;

   func = dxil_get_function(&ctx->mod, "dx.op.sampleCmp", DXIL_F32);
   opcode = DXIL_INTR_SAMPLE_CMP;

   if (!func)
      return NULL;

   const struct dxil_value *args[12] = {
      dxil_module_get_int32_const(&ctx->mod, opcode),
      params->tex, params->sampler,
      params->coord[0], params->coord[1], params->coord[2], params->coord[3],
      params->offset[0], params->offset[1], params->offset[2],
      params->cmp, params->min_lod
   };

   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static const struct dxil_value *
emit_sample_cmp_level_zero(struct ntd_context *ctx, struct texop_parameters *params)
{
   const struct dxil_func *func;
   enum dxil_intr opcode;

   func = dxil_get_function(&ctx->mod, "dx.op.sampleCmpLevelZero", DXIL_F32);
   opcode = DXIL_INTR_SAMPLE_CMP_LVL_ZERO;

   if (!func)
      return NULL;

   const struct dxil_value *args[11] = {
      dxil_module_get_int32_const(&ctx->mod, opcode),
      params->tex, params->sampler,
      params->coord[0], params->coord[1], params->coord[2], params->coord[3],
      params->offset[0], params->offset[1], params->offset[2],
      params->cmp
   };

   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static const struct dxil_value *
emit_sample_cmp_level(struct ntd_context *ctx, struct texop_parameters *params)
{
   ctx->mod.feats.advanced_texture_ops = true;
   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.sampleCmpLevel", params->overload);
   if (!func)
      return NULL;

   assert(params->lod_or_sample != NULL);

   const struct dxil_value *args[12] = {
      dxil_module_get_int32_const(&ctx->mod, DXIL_INTR_SAMPLE_CMP_LEVEL),
      params->tex, params->sampler,
      params->coord[0], params->coord[1], params->coord[2], params->coord[3],
      params->offset[0], params->offset[1], params->offset[2],
      params->cmp, params->lod_or_sample
   };

   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static const struct dxil_value *
emit_sample_grad(struct ntd_context *ctx, struct texop_parameters *params)
{
   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.sampleGrad", params->overload);
   if (!func)
      return false;

   const struct dxil_value *args[17] = {
      dxil_module_get_int32_const(&ctx->mod, DXIL_INTR_SAMPLE_GRAD),
      params->tex, params->sampler,
      params->coord[0], params->coord[1], params->coord[2], params->coord[3],
      params->offset[0], params->offset[1], params->offset[2],
      params->dx[0], params->dx[1], params->dx[2],
      params->dy[0], params->dy[1], params->dy[2],
      params->min_lod
   };

   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static const struct dxil_value *
emit_texel_fetch(struct ntd_context *ctx, struct texop_parameters *params)
{
   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.textureLoad", params->overload);
   if (!func)
      return false;

   if (!params->lod_or_sample)
      params->lod_or_sample = dxil_module_get_undef(&ctx->mod, dxil_module_get_int_type(&ctx->mod, 32));

   const struct dxil_value *args[] = {
      dxil_module_get_int32_const(&ctx->mod, DXIL_INTR_TEXTURE_LOAD),
      params->tex,
      params->lod_or_sample, params->coord[0], params->coord[1], params->coord[2],
      params->offset[0], params->offset[1], params->offset[2]
   };

   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static const struct dxil_value *
emit_texture_lod(struct ntd_context *ctx, struct texop_parameters *params, bool clamped)
{
   const struct dxil_func *func = dxil_get_function(&ctx->mod, "dx.op.calculateLOD", DXIL_F32);
   if (!func)
      return false;

   const struct dxil_value *args[] = {
      dxil_module_get_int32_const(&ctx->mod, DXIL_INTR_TEXTURE_LOD),
      params->tex,
      params->sampler,
      params->coord[0],
      params->coord[1],
      params->coord[2],
      dxil_module_get_int1_const(&ctx->mod, clamped ? 1 : 0)
   };

   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args));
}

static const struct dxil_value *
emit_texture_gather(struct ntd_context *ctx, struct texop_parameters *params, unsigned component)
{
   const struct dxil_func *func = dxil_get_function(&ctx->mod,
      params->cmp ? "dx.op.textureGatherCmp" : "dx.op.textureGather", params->overload);
   if (!func)
      return false;

   const struct dxil_value *args[] = {
      dxil_module_get_int32_const(&ctx->mod, params->cmp ? 
         DXIL_INTR_TEXTURE_GATHER_CMP : DXIL_INTR_TEXTURE_GATHER),
      params->tex,
      params->sampler,
      params->coord[0],
      params->coord[1],
      params->coord[2],
      params->coord[3],
      params->offset[0],
      params->offset[1],
      dxil_module_get_int32_const(&ctx->mod, component),
      params->cmp
   };

   return dxil_emit_call(&ctx->mod, func, args, ARRAY_SIZE(args) - (params->cmp ? 0 : 1));
}

static bool
emit_tex(struct ntd_context *ctx, nir_tex_instr *instr)
{
   struct texop_parameters params;
   memset(&params, 0, sizeof(struct texop_parameters));
   if (ctx->opts->environment != DXIL_ENVIRONMENT_VULKAN) {
      params.tex = ctx->srv_handles[instr->texture_index];
      params.sampler = ctx->sampler_handles[instr->sampler_index];
   }

   const struct dxil_type *int_type = dxil_module_get_int_type(&ctx->mod, 32);
   const struct dxil_type *float_type = dxil_module_get_float_type(&ctx->mod, 32);
   const struct dxil_value *int_undef = dxil_module_get_undef(&ctx->mod, int_type);
   const struct dxil_value *float_undef = dxil_module_get_undef(&ctx->mod, float_type);

   unsigned coord_components = 0, offset_components = 0, dx_components = 0, dy_components = 0;
   params.overload = get_overload(instr->dest_type, 32);

   bool lod_is_zero = false;
   for (unsigned i = 0; i < instr->num_srcs; i++) {
      nir_alu_type type = nir_tex_instr_src_type(instr, i);

      switch (instr->src[i].src_type) {
      case nir_tex_src_coord:
         coord_components = get_n_src(ctx, params.coord, ARRAY_SIZE(params.coord),
                                      &instr->src[i], type);
         if (!coord_components)
            return false;
         break;

      case nir_tex_src_offset:
         offset_components = get_n_src(ctx, params.offset, ARRAY_SIZE(params.offset),
                                       &instr->src[i],  nir_type_int);
         if (!offset_components)
            return false;

         /* Dynamic offsets were only allowed with gather, until "advanced texture ops" in SM7 */
         if (!nir_src_is_const(instr->src[i].src) && instr->op != nir_texop_tg4)
            ctx->mod.feats.advanced_texture_ops = true;
         break;

      case nir_tex_src_bias:
         assert(instr->op == nir_texop_txb);
         assert(nir_src_num_components(instr->src[i].src) == 1);
         params.bias = get_src(ctx, &instr->src[i].src, 0, nir_type_float);
         if (!params.bias)
            return false;
         break;

      case nir_tex_src_lod:
         assert(nir_src_num_components(instr->src[i].src) == 1);
         if (instr->op == nir_texop_txf_ms) {
            assert(nir_src_as_int(instr->src[i].src) == 0);
            break;
         }

         /* Buffers don't have a LOD */
         if (instr->sampler_dim != GLSL_SAMPLER_DIM_BUF)
            params.lod_or_sample = get_src(ctx, &instr->src[i].src, 0, type);
         else
            params.lod_or_sample = int_undef;
         if (!params.lod_or_sample)
            return false;

         if (nir_src_is_const(instr->src[i].src) && nir_src_as_float(instr->src[i].src) == 0.0f)
            lod_is_zero = true;
         break;

      case nir_tex_src_min_lod:
         assert(nir_src_num_components(instr->src[i].src) == 1);
         params.min_lod = get_src(ctx, &instr->src[i].src, 0, type);
         if (!params.min_lod)
            return false;
         break;

      case nir_tex_src_comparator:
         assert(nir_src_num_components(instr->src[i].src) == 1);
         params.cmp = get_src(ctx, &instr->src[i].src, 0, nir_type_float);
         if (!params.cmp)
            return false;
         break;

      case nir_tex_src_ddx:
         dx_components = get_n_src(ctx, params.dx, ARRAY_SIZE(params.dx),
                                   &instr->src[i], nir_type_float);
         if (!dx_components)
            return false;
         break;

      case nir_tex_src_ddy:
         dy_components = get_n_src(ctx, params.dy, ARRAY_SIZE(params.dy),
                                   &instr->src[i], nir_type_float);
         if (!dy_components)
            return false;
         break;

      case nir_tex_src_ms_index:
         params.lod_or_sample = get_src(ctx, &instr->src[i].src, 0, nir_type_int);
         if (!params.lod_or_sample)
            return false;
         break;

      case nir_tex_src_texture_deref:
         assert(ctx->opts->environment == DXIL_ENVIRONMENT_VULKAN);
         params.tex = get_src_ssa(ctx, instr->src[i].src.ssa, 0);
         break;

      case nir_tex_src_sampler_deref:
         assert(ctx->opts->environment == DXIL_ENVIRONMENT_VULKAN);
         params.sampler = get_src_ssa(ctx, instr->src[i].src.ssa, 0);
         break;

      case nir_tex_src_texture_offset:
         params.tex = emit_createhandle_call_dynamic(ctx, DXIL_RESOURCE_CLASS_SRV,
            0, instr->texture_index,
            dxil_emit_binop(&ctx->mod, DXIL_BINOP_ADD,
               get_src_ssa(ctx, instr->src[i].src.ssa, 0),
               dxil_module_get_int32_const(&ctx->mod, instr->texture_index), 0),
            instr->texture_non_uniform);
         break;

      case nir_tex_src_sampler_offset:
         if (nir_tex_instr_need_sampler(instr)) {
            params.sampler = emit_createhandle_call_dynamic(ctx, DXIL_RESOURCE_CLASS_SAMPLER,
               0, instr->sampler_index,
               dxil_emit_binop(&ctx->mod, DXIL_BINOP_ADD,
                  get_src_ssa(ctx, instr->src[i].src.ssa, 0),
                  dxil_module_get_int32_const(&ctx->mod, instr->sampler_index), 0),
               instr->sampler_non_uniform);
         }
         break;

      case nir_tex_src_projector:
         unreachable("Texture projector should have been lowered");

      default:
         fprintf(stderr, "texture source: %d\n", instr->src[i].src_type);
         unreachable("unknown texture source");
      }
   }

   assert(params.tex != NULL);
   assert(instr->op == nir_texop_txf ||
          instr->op == nir_texop_txf_ms ||
          nir_tex_instr_is_query(instr) ||
          params.sampler != NULL);

   PAD_SRC(ctx, params.coord, coord_components, float_undef);
   PAD_SRC(ctx, params.offset, offset_components, int_undef);
   if (!params.min_lod) params.min_lod = float_undef;

   const struct dxil_value *sample = NULL;
   switch (instr->op) {
   case nir_texop_txb:
      sample = emit_sample_bias(ctx, &params);
      break;

   case nir_texop_tex:
      if (params.cmp != NULL) {
         sample = emit_sample_cmp(ctx, &params);
         break;
      } else if (ctx->mod.shader_kind == DXIL_PIXEL_SHADER) {
         sample = emit_sample(ctx, &params);
         break;
      }
      params.lod_or_sample = dxil_module_get_float_const(&ctx->mod, 0);
      lod_is_zero = true;
      FALLTHROUGH;
   case nir_texop_txl:
      if (lod_is_zero && params.cmp != NULL && ctx->opts->shader_model_max < SHADER_MODEL_6_7) {
         /* Prior to SM 6.7, if the level is constant 0.0, ignore the LOD argument,
          * so level-less DXIL instructions are used. This is needed to avoid emitting
          * dx.op.sampleCmpLevel, which would not be available.
          */
         sample = emit_sample_cmp_level_zero(ctx, &params);
      } else {
         if (params.cmp != NULL)
            sample = emit_sample_cmp_level(ctx, &params);
         else
            sample = emit_sample_level(ctx, &params);
      }
      break;

   case nir_texop_txd:
      PAD_SRC(ctx, params.dx, dx_components, float_undef);
      PAD_SRC(ctx, params.dy, dy_components,float_undef);
      sample = emit_sample_grad(ctx, &params);
      break;

   case nir_texop_txf:
   case nir_texop_txf_ms:
      if (instr->sampler_dim == GLSL_SAMPLER_DIM_BUF) {
         params.coord[1] = int_undef;
         sample = emit_bufferload_call(ctx, params.tex, params.coord, params.overload);
      } else {
         PAD_SRC(ctx, params.coord, coord_components, int_undef);
         sample = emit_texel_fetch(ctx, &params);
      }
      break;

   case nir_texop_txs:
      sample = emit_texture_size(ctx, &params);
      break;

   case nir_texop_tg4:
      sample = emit_texture_gather(ctx, &params, instr->component);
      break;

   case nir_texop_lod:
      sample = emit_texture_lod(ctx, &params, true);
      store_dest(ctx, &instr->dest, 0, sample, nir_alu_type_get_base_type(instr->dest_type));
      sample = emit_texture_lod(ctx, &params, false);
      store_dest(ctx, &instr->dest, 1, sample, nir_alu_type_get_base_type(instr->dest_type));
      return true;

   case nir_texop_query_levels: {
      params.lod_or_sample = dxil_module_get_int_const(&ctx->mod, 0, 32);
      sample = emit_texture_size(ctx, &params);
      const struct dxil_value *retval = dxil_emit_extractval(&ctx->mod, sample, 3);
      store_dest(ctx, &instr->dest, 0, retval, nir_alu_type_get_base_type(instr->dest_type));
      return true;
   }

   case nir_texop_texture_samples: {
      params.lod_or_sample = int_undef;
      sample = emit_texture_size(ctx, &params);
      const struct dxil_value *retval = dxil_emit_extractval(&ctx->mod, sample, 3);
      store_dest(ctx, &instr->dest, 0, retval, nir_alu_type_get_base_type(instr->dest_type));
      return true;
   }

   default:
      fprintf(stderr, "texture op: %d\n", instr->op);
      unreachable("unknown texture op");
   }

   if (!sample)
      return false;

   for (unsigned i = 0; i < nir_dest_num_components(instr->dest); ++i) {
      const struct dxil_value *retval = dxil_emit_extractval(&ctx->mod, sample, i);
      store_dest(ctx, &instr->dest, i, retval, nir_alu_type_get_base_type(instr->dest_type));
   }

   return true;
}

static bool
emit_undefined(struct ntd_context *ctx, nir_ssa_undef_instr *undef)
{
   for (unsigned i = 0; i < undef->def.num_components; ++i)
      store_ssa_def(ctx, &undef->def, i, dxil_module_get_int32_const(&ctx->mod, 0));
   return true;
}

static bool emit_instr(struct ntd_context *ctx, struct nir_instr* instr)
{
   switch (instr->type) {
   case nir_instr_type_alu:
      return emit_alu(ctx, nir_instr_as_alu(instr));
   case nir_instr_type_intrinsic:
      return emit_intrinsic(ctx, nir_instr_as_intrinsic(instr));
   case nir_instr_type_load_const:
      return emit_load_const(ctx, nir_instr_as_load_const(instr));
   case nir_instr_type_deref:
      return emit_deref(ctx, nir_instr_as_deref(instr));
   case nir_instr_type_jump:
      return emit_jump(ctx, nir_instr_as_jump(instr));
   case nir_instr_type_phi:
      return emit_phi(ctx, nir_instr_as_phi(instr));
   case nir_instr_type_tex:
      return emit_tex(ctx, nir_instr_as_tex(instr));
   case nir_instr_type_ssa_undef:
      return emit_undefined(ctx, nir_instr_as_ssa_undef(instr));
   default:
      log_nir_instr_unsupported(ctx->logger, "Unimplemented instruction type",
                                instr);
      return false;
   }
}


static bool
emit_block(struct ntd_context *ctx, struct nir_block *block)
{
   assert(block->index < ctx->mod.cur_emitting_func->num_basic_block_ids);
   ctx->mod.cur_emitting_func->basic_block_ids[block->index] = ctx->mod.cur_emitting_func->curr_block;

   nir_foreach_instr(instr, block) {
      TRACE_CONVERSION(instr);

      if (!emit_instr(ctx, instr))  {
         return false;
      }
   }
   return true;
}

static bool
emit_cf_list(struct ntd_context *ctx, struct exec_list *list);

static bool
emit_if(struct ntd_context *ctx, struct nir_if *if_stmt)
{
   assert(nir_src_num_components(if_stmt->condition) == 1);
   const struct dxil_value *cond = get_src(ctx, &if_stmt->condition, 0,
                                           nir_type_bool);
   if (!cond)
      return false;

   /* prepare blocks */
   nir_block *then_block = nir_if_first_then_block(if_stmt);
   assert(nir_if_last_then_block(if_stmt)->successors[0]);
   assert(!nir_if_last_then_block(if_stmt)->successors[1]);
   int then_succ = nir_if_last_then_block(if_stmt)->successors[0]->index;

   nir_block *else_block = NULL;
   int else_succ = -1;
   if (!exec_list_is_empty(&if_stmt->else_list)) {
      else_block = nir_if_first_else_block(if_stmt);
      assert(nir_if_last_else_block(if_stmt)->successors[0]);
      assert(!nir_if_last_else_block(if_stmt)->successors[1]);
      else_succ = nir_if_last_else_block(if_stmt)->successors[0]->index;
   }

   if (!emit_cond_branch(ctx, cond, then_block->index,
                         else_block ? else_block->index : then_succ))
      return false;

   /* handle then-block */
   if (!emit_cf_list(ctx, &if_stmt->then_list) ||
       (!nir_block_ends_in_jump(nir_if_last_then_block(if_stmt)) &&
        !emit_branch(ctx, then_succ)))
      return false;

   if (else_block) {
      /* handle else-block */
      if (!emit_cf_list(ctx, &if_stmt->else_list) ||
          (!nir_block_ends_in_jump(nir_if_last_else_block(if_stmt)) &&
           !emit_branch(ctx, else_succ)))
         return false;
   }

   return true;
}

static bool
emit_loop(struct ntd_context *ctx, nir_loop *loop)
{
   nir_block *first_block = nir_loop_first_block(loop);
   nir_block *last_block = nir_loop_last_block(loop);

   assert(last_block->successors[0]);
   assert(!last_block->successors[1]);

   if (!emit_branch(ctx, first_block->index))
      return false;

   if (!emit_cf_list(ctx, &loop->body))
      return false;

   /* If the loop's last block doesn't explicitly jump somewhere, then there's
    * an implicit continue that should take it back to the first loop block
    */
   nir_instr *last_instr = nir_block_last_instr(last_block);
   if ((!last_instr || last_instr->type != nir_instr_type_jump) &&
       !emit_branch(ctx, first_block->index))
      return false;

   return true;
}

static bool
emit_cf_list(struct ntd_context *ctx, struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block:
         if (!emit_block(ctx, nir_cf_node_as_block(node)))
            return false;
         break;

      case nir_cf_node_if:
         if (!emit_if(ctx, nir_cf_node_as_if(node)))
            return false;
         break;

      case nir_cf_node_loop:
         if (!emit_loop(ctx, nir_cf_node_as_loop(node)))
            return false;
         break;

      default:
         unreachable("unsupported cf-list node");
         break;
      }
   }
   return true;
}

static void
insert_sorted_by_binding(struct exec_list *var_list, nir_variable *new_var)
{
   nir_foreach_variable_in_list(var, var_list) {
      if (var->data.binding > new_var->data.binding) {
         exec_node_insert_node_before(&var->node, &new_var->node);
         return;
      }
   }
   exec_list_push_tail(var_list, &new_var->node);
}


static void
sort_uniforms_by_binding_and_remove_structs(nir_shader *s)
{
   struct exec_list new_list;
   exec_list_make_empty(&new_list);

   nir_foreach_variable_with_modes_safe(var, s, nir_var_uniform) {
      exec_node_remove(&var->node);
      const struct glsl_type *type = glsl_without_array(var->type);
      if (!glsl_type_is_struct(type))
         insert_sorted_by_binding(&new_list, var);
   }
   exec_list_append(&s->variables, &new_list);
}

static void
prepare_phi_values(struct ntd_context *ctx, nir_function_impl *impl)
{
   /* PHI nodes are difficult to get right when tracking the types:
    * Since the incoming sources are linked to blocks, we can't bitcast
    * on the fly while loading. So scan the shader and insert a typed dummy
    * value for each phi source, and when storing we convert if the incoming
    * value has a different type then the one expected by the phi node.
    * We choose int as default, because it supports more bit sizes.
    */
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_phi) {
            nir_phi_instr *ir = nir_instr_as_phi(instr);
            unsigned bitsize = nir_dest_bit_size(ir->dest);
            const struct dxil_value *dummy = dxil_module_get_int_const(&ctx->mod, 0, bitsize);
            nir_foreach_phi_src(src, ir) {
               for(unsigned int i = 0; i < ir->dest.ssa.num_components; ++i)
                  store_ssa_def(ctx, src->src.ssa, i, dummy);
            }
         }
      }
   }
}

static bool
emit_cbvs(struct ntd_context *ctx)
{
   if (ctx->opts->environment != DXIL_ENVIRONMENT_GL) {
      nir_foreach_variable_with_modes(var, ctx->shader, nir_var_mem_ubo) {
         if (!emit_ubo_var(ctx, var))
            return false;
      }
   } else {
      if (ctx->shader->info.num_ubos) {
         const unsigned ubo_size = 16384 /*4096 vec4's*/;
         bool has_ubo0 = !ctx->opts->no_ubo0;
         bool has_state_vars = ctx->opts->last_ubo_is_not_arrayed;
         unsigned ubo1_array_size = ctx->shader->info.num_ubos -
            (has_state_vars ? 2 : 1);

         if (has_ubo0 &&
             !emit_cbv(ctx, 0, 0, ubo_size, 1, "__ubo_uniforms"))
            return false;
         if (ubo1_array_size &&
             !emit_cbv(ctx, 1, 0, ubo_size, ubo1_array_size, "__ubos"))
            return false;
         if (has_state_vars &&
             !emit_cbv(ctx, ctx->shader->info.num_ubos - 1, 0, ubo_size, 1, "__ubo_state_vars"))
            return false;
      }
   }

   return true;
}

static bool
emit_scratch(struct ntd_context *ctx)
{
   if (ctx->shader->scratch_size) {
      /*
       * We always allocate an u32 array, no matter the actual variable types.
       * According to the DXIL spec, the minimum load/store granularity is
       * 32-bit, anything smaller requires using a read-extract/read-write-modify
       * approach.
       */
      unsigned size = ALIGN_POT(ctx->shader->scratch_size, sizeof(uint32_t));
      const struct dxil_type *int32 = dxil_module_get_int_type(&ctx->mod, 32);
      const struct dxil_value *array_length = dxil_module_get_int32_const(&ctx->mod, size / sizeof(uint32_t));
      if (!int32 || !array_length)
         return false;

      const struct dxil_type *type = dxil_module_get_array_type(
         &ctx->mod, int32, size / sizeof(uint32_t));
      if (!type)
         return false;

      ctx->scratchvars = dxil_emit_alloca(&ctx->mod, type, int32, array_length, 4);
      if (!ctx->scratchvars)
         return false;
   }

   return true;
}

/* The validator complains if we don't have ops that reference a global variable. */
static bool
shader_has_shared_ops(struct nir_shader *s)
{
   nir_foreach_function(func, s) {
      if (!func->impl)
         continue;
      nir_foreach_block(block, func->impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            switch (intrin->intrinsic) {
            case nir_intrinsic_load_shared_dxil:
            case nir_intrinsic_store_shared_dxil:
            case nir_intrinsic_shared_atomic_add_dxil:
            case nir_intrinsic_shared_atomic_and_dxil:
            case nir_intrinsic_shared_atomic_comp_swap_dxil:
            case nir_intrinsic_shared_atomic_exchange_dxil:
            case nir_intrinsic_shared_atomic_imax_dxil:
            case nir_intrinsic_shared_atomic_imin_dxil:
            case nir_intrinsic_shared_atomic_or_dxil:
            case nir_intrinsic_shared_atomic_umax_dxil:
            case nir_intrinsic_shared_atomic_umin_dxil:
            case nir_intrinsic_shared_atomic_xor_dxil:
               return true;
            default: break;
            }
         }
      }
   }
   return false;
}

static bool
emit_function(struct ntd_context *ctx, nir_function *func)
{
   assert(func->num_params == 0);
   nir_function_impl *impl = func->impl;
   if (!impl)
      return true;

   nir_metadata_require(impl, nir_metadata_block_index);

   const struct dxil_type *void_type = dxil_module_get_void_type(&ctx->mod);
   const struct dxil_type *func_type = dxil_module_add_function_type(&ctx->mod, void_type, NULL, 0);
   struct dxil_func_def *func_def = dxil_add_function_def(&ctx->mod, func->name, func_type, impl->num_blocks);
   if (!func_def)
      return false;

   if (func->is_entrypoint)
      ctx->main_func_def = func_def;
   else if (func == ctx->tess_ctrl_patch_constant_func)
      ctx->tess_ctrl_patch_constant_func_def = func_def;

   ctx->defs = rzalloc_array(ctx->ralloc_ctx, struct dxil_def, impl->ssa_alloc);
   if (!ctx->defs)
      return false;
   ctx->num_defs = impl->ssa_alloc;

   ctx->phis = _mesa_pointer_hash_table_create(ctx->ralloc_ctx);
   if (!ctx->phis)
      return false;

   prepare_phi_values(ctx, impl);

   if (!emit_scratch(ctx))
      return false;

   if (!emit_static_indexing_handles(ctx))
      return false;

   if (!emit_cf_list(ctx, &impl->body))
      return false;

   hash_table_foreach(ctx->phis, entry) {
      if (!fixup_phi(ctx, (nir_phi_instr *)entry->key,
                     (struct phi_block *)entry->data))
         return false;
   }

   if (!dxil_emit_ret_void(&ctx->mod))
      return false;

   ralloc_free(ctx->defs);
   ctx->defs = NULL;
   _mesa_hash_table_destroy(ctx->phis, NULL);
   return true;
}

static bool
emit_module(struct ntd_context *ctx, const struct nir_to_dxil_options *opts)
{
   /* The validator forces us to emit resources in a specific order:
    * CBVs, Samplers, SRVs, UAVs. While we are at it also remove
    * stale struct uniforms, they are lowered but might not have been removed */
   sort_uniforms_by_binding_and_remove_structs(ctx->shader);

   /* CBVs */
   if (!emit_cbvs(ctx))
      return false;

   /* Samplers */
   nir_foreach_variable_with_modes(var, ctx->shader, nir_var_uniform) {
      unsigned count = glsl_type_get_sampler_count(var->type);
      assert(count == 0 || glsl_type_is_bare_sampler(glsl_without_array(var->type)));
      if (count > 0 && !emit_sampler(ctx, var, count))
         return false;
   }

   /* SRVs */
   nir_foreach_variable_with_modes(var, ctx->shader, nir_var_uniform) {
      if (glsl_type_is_texture(glsl_without_array(var->type)) &&
          !emit_srv(ctx, var, glsl_type_get_texture_count(var->type)))
         return false;
   }

   if (ctx->opts->environment == DXIL_ENVIRONMENT_VULKAN) {
      nir_foreach_image_variable(var, ctx->shader) {
         if ((var->data.access & ACCESS_NON_WRITEABLE) &&
             !emit_srv(ctx, var, glsl_type_get_image_count(var->type)))
            return false;
      }
   }

   /* Handle read-only SSBOs as SRVs */
   if (ctx->opts->environment == DXIL_ENVIRONMENT_VULKAN) {
      nir_foreach_variable_with_modes(var, ctx->shader, nir_var_mem_ssbo) {
         if ((var->data.access & ACCESS_NON_WRITEABLE) != 0) {
            unsigned count = 1;
            if (glsl_type_is_array(var->type))
               count = glsl_get_length(var->type);
            if (!emit_srv(ctx, var, count))
               return false;
         }
      }
   }

   if (ctx->shader->info.shared_size && shader_has_shared_ops(ctx->shader)) {
      const struct dxil_type *type;
      unsigned size;

     /*
      * We always allocate an u32 array, no matter the actual variable types.
      * According to the DXIL spec, the minimum load/store granularity is
      * 32-bit, anything smaller requires using a read-extract/read-write-modify
      * approach. Non-atomic 64-bit accesses are allowed, but the
      * GEP(cast(gvar, u64[] *), offset) and cast(GEP(gvar, offset), u64 *))
      * sequences don't seem to be accepted by the DXIL validator when the
      * pointer is in the groupshared address space, making the 32-bit -> 64-bit
      * pointer cast impossible.
      */
      size = ALIGN_POT(ctx->shader->info.shared_size, sizeof(uint32_t));
      type = dxil_module_get_array_type(&ctx->mod,
                                        dxil_module_get_int_type(&ctx->mod, 32),
                                        size / sizeof(uint32_t));
      ctx->sharedvars = dxil_add_global_ptr_var(&ctx->mod, "shared", type,
                                                DXIL_AS_GROUPSHARED,
                                                ffs(sizeof(uint64_t)),
                                                NULL);
   }

   /* UAVs */
   if (ctx->shader->info.stage == MESA_SHADER_KERNEL) {
      if (!emit_globals(ctx, opts->num_kernel_globals))
         return false;

      ctx->consts = _mesa_pointer_hash_table_create(ctx->ralloc_ctx);
      if (!ctx->consts)
         return false;
      if (!emit_global_consts(ctx))
         return false;
   } else if (ctx->opts->environment == DXIL_ENVIRONMENT_VULKAN) {
      /* Handle read/write SSBOs as UAVs */
      nir_foreach_variable_with_modes(var, ctx->shader, nir_var_mem_ssbo) {
         if ((var->data.access & ACCESS_NON_WRITEABLE) == 0) {
            unsigned count = 1;
            if (glsl_type_is_array(var->type))
               count = glsl_get_length(var->type);
            if (!emit_uav(ctx, var->data.binding, var->data.descriptor_set,
                        count, DXIL_COMP_TYPE_INVALID,
                        DXIL_RESOURCE_KIND_RAW_BUFFER, var->name))
               return false;
            
         }
      }
   } else {
      for (unsigned i = 0; i < ctx->shader->info.num_ssbos; ++i) {
         char name[64];
         snprintf(name, sizeof(name), "__ssbo%d", i);
         if (!emit_uav(ctx, i, 0, 1, DXIL_COMP_TYPE_INVALID,
                       DXIL_RESOURCE_KIND_RAW_BUFFER, name))
            return false;
      }
      /* To work around a WARP bug, bind these descriptors a second time in descriptor
       * space 2. Space 0 will be used for static indexing, while space 2 will be used
       * for dynamic indexing. Space 0 will be individual SSBOs in the DXIL shader, while
       * space 2 will be a single array.
       */
      if (ctx->shader->info.num_ssbos &&
          !emit_uav(ctx, 0, 2, ctx->shader->info.num_ssbos, DXIL_COMP_TYPE_INVALID,
                    DXIL_RESOURCE_KIND_RAW_BUFFER, "__ssbo_dynamic"))
         return false;
   }

   nir_foreach_image_variable(var, ctx->shader) {
      if (ctx->opts->environment == DXIL_ENVIRONMENT_VULKAN &&
          var && (var->data.access & ACCESS_NON_WRITEABLE))
         continue; // already handled in SRV

      if (!emit_uav_var(ctx, var, glsl_type_get_image_count(var->type)))
         return false;
   }

   ctx->mod.info.has_per_sample_input =
      BITSET_TEST(ctx->shader->info.system_values_read, SYSTEM_VALUE_SAMPLE_ID) ||
      ctx->shader->info.fs.uses_sample_shading ||
      ctx->shader->info.fs.uses_sample_qualifier;
   if (!ctx->mod.info.has_per_sample_input && ctx->shader->info.stage == MESA_SHADER_FRAGMENT) {
      nir_foreach_variable_with_modes(var, ctx->shader, nir_var_shader_in | nir_var_system_value) {
         if (var->data.sample) {
            ctx->mod.info.has_per_sample_input = true;
            break;
         }
      }
   }

   /* From the Vulkan spec 1.3.238, section 15.8:
    * When Sample Shading is enabled, the x and y components of FragCoord reflect the location 
    * of one of the samples corresponding to the shader invocation.
    * 
    * In other words, if the fragment shader is executing per-sample, then the position variable
    * should always be per-sample, 
    * 
    * Also:
    * The Centroid interpolation decoration is ignored, but allowed, on FragCoord.
    */
   if (ctx->opts->environment == DXIL_ENVIRONMENT_VULKAN) {
      nir_variable *pos_var = nir_find_variable_with_location(ctx->shader, nir_var_shader_in, VARYING_SLOT_POS);
      if (pos_var) {
         if (ctx->mod.info.has_per_sample_input)
            pos_var->data.sample = true;
         pos_var->data.centroid = false;
      }
   }

   unsigned input_clip_size = ctx->mod.shader_kind == DXIL_PIXEL_SHADER ?
      ctx->shader->info.clip_distance_array_size : ctx->opts->input_clip_size;
   preprocess_signatures(&ctx->mod, ctx->shader, input_clip_size);

   nir_foreach_function(func, ctx->shader) {
      if (!emit_function(ctx, func))
         return false;
   }

   if (ctx->shader->info.stage == MESA_SHADER_FRAGMENT) {
      nir_foreach_variable_with_modes(var, ctx->shader, nir_var_shader_out) {
         if (var->data.location == FRAG_RESULT_STENCIL) {
            ctx->mod.feats.stencil_ref = true;
         }
      }
   } else if (ctx->shader->info.stage == MESA_SHADER_VERTEX ||
              ctx->shader->info.stage == MESA_SHADER_TESS_EVAL) {
      if (ctx->shader->info.outputs_written &
          (VARYING_BIT_VIEWPORT | VARYING_BIT_LAYER))
         ctx->mod.feats.array_layer_from_vs_or_ds = true;
   }

   if (ctx->mod.feats.native_low_precision && ctx->mod.minor_version < 2) {
      ctx->logger->log(ctx->logger->priv,
                       "Shader uses 16bit, which requires shader model 6.2, but 6.2 is unsupported\n");
      return false;
   }

   return emit_metadata(ctx) &&
          dxil_emit_module(&ctx->mod);
}

static unsigned int
get_dxil_shader_kind(struct nir_shader *s)
{
   switch (s->info.stage) {
   case MESA_SHADER_VERTEX:
      return DXIL_VERTEX_SHADER;
   case MESA_SHADER_TESS_CTRL:
      return DXIL_HULL_SHADER;
   case MESA_SHADER_TESS_EVAL:
      return DXIL_DOMAIN_SHADER;
   case MESA_SHADER_GEOMETRY:
      return DXIL_GEOMETRY_SHADER;
   case MESA_SHADER_FRAGMENT:
      return DXIL_PIXEL_SHADER;
   case MESA_SHADER_KERNEL:
   case MESA_SHADER_COMPUTE:
      return DXIL_COMPUTE_SHADER;
   default:
      unreachable("unknown shader stage in nir_to_dxil");
      return DXIL_COMPUTE_SHADER;
   }
}

static unsigned
lower_bit_size_callback(const nir_instr* instr, void *data)
{
   if (instr->type != nir_instr_type_alu)
      return 0;
   const nir_alu_instr *alu = nir_instr_as_alu(instr);

   if (nir_op_infos[alu->op].is_conversion)
      return 0;

   unsigned num_inputs = nir_op_infos[alu->op].num_inputs;
   const struct nir_to_dxil_options *opts = (const struct nir_to_dxil_options*)data;
   unsigned min_bit_size = opts->lower_int16 ? 32 : 16;

   unsigned ret = 0;
   for (unsigned i = 0; i < num_inputs; i++) {
      unsigned bit_size = nir_src_bit_size(alu->src[i].src);
      if (bit_size != 1 && bit_size < min_bit_size)
         ret = min_bit_size;
   }

   return ret;
}

static void
optimize_nir(struct nir_shader *s, const struct nir_to_dxil_options *opts)
{
   bool progress;
   do {
      progress = false;
      NIR_PASS_V(s, nir_lower_vars_to_ssa);
      NIR_PASS(progress, s, nir_lower_indirect_derefs, nir_var_function_temp, UINT32_MAX);
      NIR_PASS(progress, s, nir_lower_alu_to_scalar, NULL, NULL);
      NIR_PASS(progress, s, nir_copy_prop);
      NIR_PASS(progress, s, nir_opt_copy_prop_vars);
      NIR_PASS(progress, s, nir_lower_bit_size, lower_bit_size_callback, (void*)opts);
      NIR_PASS(progress, s, dxil_nir_lower_8bit_conv);
      if (opts->lower_int16)
         NIR_PASS(progress, s, dxil_nir_lower_16bit_conv);
      NIR_PASS(progress, s, nir_opt_remove_phis);
      NIR_PASS(progress, s, nir_opt_dce);
      NIR_PASS(progress, s, nir_opt_if, nir_opt_if_aggressive_last_continue | nir_opt_if_optimize_phi_true_false);
      NIR_PASS(progress, s, nir_opt_dead_cf);
      NIR_PASS(progress, s, nir_opt_cse);
      NIR_PASS(progress, s, nir_opt_peephole_select, 8, true, true);
      NIR_PASS(progress, s, nir_opt_algebraic);
      NIR_PASS(progress, s, dxil_nir_lower_x2b);
      if (s->options->lower_int64_options)
         NIR_PASS(progress, s, nir_lower_int64);
      NIR_PASS(progress, s, nir_lower_alu);
      NIR_PASS(progress, s, nir_opt_constant_folding);
      NIR_PASS(progress, s, nir_opt_undef);
      NIR_PASS(progress, s, nir_lower_undef_to_zero);
      NIR_PASS(progress, s, nir_opt_deref);
      NIR_PASS(progress, s, dxil_nir_lower_upcast_phis, opts->lower_int16 ? 32 : 16);
      NIR_PASS(progress, s, nir_lower_64bit_phis);
      NIR_PASS_V(s, nir_lower_system_values);
   } while (progress);

   do {
      progress = false;
      NIR_PASS(progress, s, nir_opt_algebraic_late);
   } while (progress);
}

static
void dxil_fill_validation_state(struct ntd_context *ctx,
                                struct dxil_validation_state *state)
{
   unsigned resource_element_size = ctx->mod.minor_validator >= 6 ?
      sizeof(struct dxil_resource_v1) : sizeof(struct dxil_resource_v0);
   state->num_resources = ctx->resources.size / resource_element_size;
   state->resources.v0 = (struct dxil_resource_v0*)ctx->resources.data;
   state->state.psv1.psv0.max_expected_wave_lane_count = UINT_MAX;
   state->state.psv1.shader_stage = (uint8_t)ctx->mod.shader_kind;
   state->state.psv1.uses_view_id = (uint8_t)ctx->mod.feats.view_id;
   state->state.psv1.sig_input_elements = (uint8_t)ctx->mod.num_sig_inputs;
   state->state.psv1.sig_output_elements = (uint8_t)ctx->mod.num_sig_outputs;
   state->state.psv1.sig_patch_const_or_prim_elements = (uint8_t)ctx->mod.num_sig_patch_consts;

   switch (ctx->mod.shader_kind) {
   case DXIL_VERTEX_SHADER:
      state->state.psv1.psv0.vs.output_position_present = ctx->mod.info.has_out_position;
      break;
   case DXIL_PIXEL_SHADER:
      /* TODO: handle depth outputs */
      state->state.psv1.psv0.ps.depth_output = ctx->mod.info.has_out_depth;
      state->state.psv1.psv0.ps.sample_frequency =
         ctx->mod.info.has_per_sample_input;
      break;
   case DXIL_COMPUTE_SHADER:
      state->state.num_threads_x = MAX2(ctx->shader->info.workgroup_size[0], 1);
      state->state.num_threads_y = MAX2(ctx->shader->info.workgroup_size[1], 1);
      state->state.num_threads_z = MAX2(ctx->shader->info.workgroup_size[2], 1);
      break;
   case DXIL_GEOMETRY_SHADER:
      state->state.psv1.max_vertex_count = ctx->shader->info.gs.vertices_out;
      state->state.psv1.psv0.gs.input_primitive = dxil_get_input_primitive(ctx->shader->info.gs.input_primitive);
      state->state.psv1.psv0.gs.output_toplology = dxil_get_primitive_topology(ctx->shader->info.gs.output_primitive);
      state->state.psv1.psv0.gs.output_stream_mask = MAX2(ctx->shader->info.gs.active_stream_mask, 1);
      state->state.psv1.psv0.gs.output_position_present = ctx->mod.info.has_out_position;
      break;
   case DXIL_HULL_SHADER:
      state->state.psv1.psv0.hs.input_control_point_count = ctx->tess_input_control_point_count;
      state->state.psv1.psv0.hs.output_control_point_count = ctx->shader->info.tess.tcs_vertices_out;
      state->state.psv1.psv0.hs.tessellator_domain = get_tessellator_domain(ctx->shader->info.tess._primitive_mode);
      state->state.psv1.psv0.hs.tessellator_output_primitive = get_tessellator_output_primitive(&ctx->shader->info);
      state->state.psv1.sig_patch_const_or_prim_vectors = ctx->mod.num_psv_patch_consts;
      break;
   case DXIL_DOMAIN_SHADER:
      state->state.psv1.psv0.ds.input_control_point_count = ctx->shader->info.tess.tcs_vertices_out;
      state->state.psv1.psv0.ds.tessellator_domain = get_tessellator_domain(ctx->shader->info.tess._primitive_mode);
      state->state.psv1.psv0.ds.output_position_present = ctx->mod.info.has_out_position;
      state->state.psv1.sig_patch_const_or_prim_vectors = ctx->mod.num_psv_patch_consts;
      break;
   default:
      assert(0 && "Shader type not (yet) supported");
   }
}

static nir_variable *
add_sysvalue(struct ntd_context *ctx,
              uint8_t value, char *name,
              int driver_location)
{

   nir_variable *var = rzalloc(ctx->shader, nir_variable);
   if (!var)
      return NULL;
   var->data.driver_location = driver_location;
   var->data.location = value;
   var->type = glsl_uint_type();
   var->name = name;
   var->data.mode = nir_var_system_value;
   var->data.interpolation = INTERP_MODE_FLAT;
   return var;
}

static bool
append_input_or_sysvalue(struct ntd_context *ctx,
                         int input_loc,  int sv_slot,
                         char *name, int driver_location)
{
   if (input_loc >= 0) {
      /* Check inputs whether a variable is available the corresponds
       * to the sysvalue */
      nir_foreach_variable_with_modes(var, ctx->shader, nir_var_shader_in) {
         if (var->data.location == input_loc) {
            ctx->system_value[sv_slot] = var;
            return true;
         }
      }
   }

   ctx->system_value[sv_slot] = add_sysvalue(ctx, sv_slot, name, driver_location);
   if (!ctx->system_value[sv_slot])
      return false;

   nir_shader_add_variable(ctx->shader, ctx->system_value[sv_slot]);
   return true;
}

struct sysvalue_name {
   gl_system_value value;
   int slot;
   char *name;
   gl_shader_stage only_in_shader;
} possible_sysvalues[] = {
   {SYSTEM_VALUE_VERTEX_ID_ZERO_BASE, -1, "SV_VertexID", MESA_SHADER_NONE},
   {SYSTEM_VALUE_INSTANCE_ID, -1, "SV_InstanceID", MESA_SHADER_NONE},
   {SYSTEM_VALUE_FRONT_FACE, VARYING_SLOT_FACE, "SV_IsFrontFace", MESA_SHADER_NONE},
   {SYSTEM_VALUE_PRIMITIVE_ID, VARYING_SLOT_PRIMITIVE_ID, "SV_PrimitiveID", MESA_SHADER_GEOMETRY},
   {SYSTEM_VALUE_SAMPLE_ID, -1, "SV_SampleIndex", MESA_SHADER_NONE},
};

static bool
allocate_sysvalues(struct ntd_context *ctx)
{
   unsigned driver_location = 0;
   nir_foreach_variable_with_modes(var, ctx->shader, nir_var_shader_in)
      driver_location = MAX2(driver_location, var->data.driver_location + 1);
   nir_foreach_variable_with_modes(var, ctx->shader, nir_var_system_value)
      driver_location = MAX2(driver_location, var->data.driver_location + 1);

   if (ctx->shader->info.stage == MESA_SHADER_FRAGMENT &&
       !BITSET_TEST(ctx->shader->info.system_values_read, SYSTEM_VALUE_SAMPLE_ID)) {
      bool need_sample_id = ctx->shader->info.fs.uses_sample_shading;

      /* "var->data.sample = true" sometimes just mean, "I want per-sample
       * shading", which explains why we can end up with vars having flat
       * interpolation with the per-sample bit set. If there's only such
       * type of variables, we need to tell DXIL that we read SV_SampleIndex
       * to make DXIL validation happy.
       */
      nir_foreach_variable_with_modes(var, ctx->shader, nir_var_shader_in) {
         bool var_can_be_sample_rate = !var->data.centroid && var->data.interpolation != INTERP_MODE_FLAT;
         /* If there's an input that will actually force sample-rate shading, then we don't
          * need SV_SampleIndex. */
         if (var->data.sample && var_can_be_sample_rate) {
            need_sample_id = false;
            break;
         }
         /* If there's an input that wants to be sample-rate, but can't be, then we might
          * need SV_SampleIndex. */
         if (var->data.sample && !var_can_be_sample_rate)
            need_sample_id = true;
      }

      if (need_sample_id)
         BITSET_SET(ctx->shader->info.system_values_read, SYSTEM_VALUE_SAMPLE_ID);
   }

   for (unsigned i = 0; i < ARRAY_SIZE(possible_sysvalues); ++i) {
      struct sysvalue_name *info = &possible_sysvalues[i];
      if (info->only_in_shader != MESA_SHADER_NONE &&
          info->only_in_shader != ctx->shader->info.stage)
         continue;
      if (BITSET_TEST(ctx->shader->info.system_values_read, info->value)) {
         if (!append_input_or_sysvalue(ctx, info->slot,
                                       info->value, info->name,
                                       driver_location++))
            return false;
      }
   }
   return true;
}

static int
type_size_vec4(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

static const unsigned dxil_validator_min_capable_version = DXIL_VALIDATOR_1_4;
static const unsigned dxil_validator_max_capable_version = DXIL_VALIDATOR_1_7;
static const unsigned dxil_min_shader_model = SHADER_MODEL_6_0;
static const unsigned dxil_max_shader_model = SHADER_MODEL_6_7;

bool
nir_to_dxil(struct nir_shader *s, const struct nir_to_dxil_options *opts,
            const struct dxil_logger *logger, struct blob *blob)
{
   assert(opts);
   bool retval = true;
   debug_dxil = (int)debug_get_option_debug_dxil();
   blob_init(blob);

   if (opts->shader_model_max < dxil_min_shader_model) {
      debug_printf("D3D12: cannot support emitting shader models lower than %d.%d\n",
                   dxil_min_shader_model >> 16,
                   dxil_min_shader_model & 0xffff);
      return false;
   }

   if (opts->shader_model_max > dxil_max_shader_model) {
      debug_printf("D3D12: cannot support emitting higher than shader model %d.%d\n",
                   dxil_max_shader_model >> 16,
                   dxil_max_shader_model & 0xffff);
      return false;
   }

   if (opts->validator_version_max != NO_DXIL_VALIDATION &&
       opts->validator_version_max < dxil_validator_min_capable_version) {
      debug_printf("D3D12: Invalid validator version %d.%d, must be 1.4 or greater\n",
         opts->validator_version_max >> 16,
         opts->validator_version_max & 0xffff);
      return false;
   }

   /* If no validation, write a blob as if it was going to be validated by the newest understood validator.
    * Same if the validator is newer than we know how to write for.
    */
   uint32_t validator_version =
      opts->validator_version_max == NO_DXIL_VALIDATION ||
      opts->validator_version_max > dxil_validator_max_capable_version ?
      dxil_validator_max_capable_version : opts->validator_version_max;

   struct ntd_context *ctx = calloc(1, sizeof(*ctx));
   if (!ctx)
      return false;

   ctx->opts = opts;
   ctx->shader = s;
   ctx->logger = logger ? logger : &default_logger;

   ctx->ralloc_ctx = ralloc_context(NULL);
   if (!ctx->ralloc_ctx) {
      retval = false;
      goto out;
   }

   util_dynarray_init(&ctx->srv_metadata_nodes, ctx->ralloc_ctx);
   util_dynarray_init(&ctx->uav_metadata_nodes, ctx->ralloc_ctx);
   util_dynarray_init(&ctx->cbv_metadata_nodes, ctx->ralloc_ctx);
   util_dynarray_init(&ctx->sampler_metadata_nodes, ctx->ralloc_ctx);
   util_dynarray_init(&ctx->resources, ctx->ralloc_ctx);
   dxil_module_init(&ctx->mod, ctx->ralloc_ctx);
   ctx->mod.shader_kind = get_dxil_shader_kind(s);
   ctx->mod.major_version = 6;
   /* Use the highest shader model that's supported and can be validated */
   ctx->mod.minor_version =
      MIN2(opts->shader_model_max & 0xffff, validator_version & 0xffff);
   ctx->mod.major_validator = validator_version >> 16;
   ctx->mod.minor_validator = validator_version & 0xffff;
   ctx->mod.godot_nir_callbacks = opts->godot_nir_callbacks;

   if (s->info.stage <= MESA_SHADER_FRAGMENT) {
      uint64_t in_mask =
         s->info.stage == MESA_SHADER_VERTEX ?
         0 : (VARYING_BIT_PRIMITIVE_ID | VARYING_BIT_VIEWPORT | VARYING_BIT_LAYER);
      uint64_t out_mask =
         s->info.stage == MESA_SHADER_FRAGMENT ?
         ((1ull << FRAG_RESULT_STENCIL) | (1ull << FRAG_RESULT_SAMPLE_MASK)) :
         (VARYING_BIT_PRIMITIVE_ID | VARYING_BIT_VIEWPORT | VARYING_BIT_LAYER);

      NIR_PASS_V(s, dxil_nir_fix_io_uint_type, in_mask, out_mask);
   }

   NIR_PASS_V(s, dxil_nir_lower_fquantize2f16);
   NIR_PASS_V(s, nir_lower_frexp);
   NIR_PASS_V(s, nir_lower_flrp, 16 | 32 | 64, true);
   NIR_PASS_V(s, nir_lower_io, nir_var_shader_in | nir_var_shader_out, type_size_vec4, nir_lower_io_lower_64bit_to_32);
   NIR_PASS_V(s, dxil_nir_ensure_position_writes);
   NIR_PASS_V(s, nir_lower_pack);
   NIR_PASS_V(s, dxil_nir_lower_system_values);
   NIR_PASS_V(s, nir_lower_io_to_scalar, nir_var_shader_in | nir_var_system_value | nir_var_shader_out);
   if (opts->shader_model_max < SHADER_MODEL_6_6) {
      /* In a later pass, load_helper_invocation will be lowered to sample mask based fallback,
       * so both load- and is- will be emulated eventually.
       */
      NIR_PASS_V(s, nir_lower_is_helper_invocation);
   }

   if (ctx->mod.shader_kind == DXIL_HULL_SHADER)
      NIR_PASS_V(s, dxil_nir_split_tess_ctrl, &ctx->tess_ctrl_patch_constant_func);

   if (ctx->mod.shader_kind == DXIL_HULL_SHADER ||
       ctx->mod.shader_kind == DXIL_DOMAIN_SHADER) {
      /* Make sure any derefs are gone after lower_io before updating tess level vars */
      NIR_PASS_V(s, nir_opt_dce);
      NIR_PASS_V(s, dxil_nir_fixup_tess_level_for_domain);
   }

   optimize_nir(s, opts);

   NIR_PASS_V(s, nir_remove_dead_variables,
              nir_var_function_temp | nir_var_shader_temp, NULL);

   if (!allocate_sysvalues(ctx))
      return false;

   NIR_PASS_V(s, dxil_nir_lower_sysval_to_load_input, ctx->system_value);
   NIR_PASS_V(s, nir_opt_dce);

   if (debug_dxil & DXIL_DEBUG_VERBOSE)
      nir_print_shader(s, stderr);

   if (!emit_module(ctx, opts)) {
      debug_printf("D3D12: dxil_container_add_module failed\n");
      retval = false;
      goto out;
   }

   if (debug_dxil & DXIL_DEBUG_DUMP_MODULE) {
      struct dxil_dumper *dumper = dxil_dump_create();
      dxil_dump_module(dumper, &ctx->mod);
      fprintf(stderr, "\n");
      dxil_dump_buf_to_file(dumper, stderr);
      fprintf(stderr, "\n\n");
      dxil_dump_free(dumper);
   }

   struct dxil_container container;
   dxil_container_init(&container);
   if (!dxil_container_add_features(&container, &ctx->mod.feats)) {
      debug_printf("D3D12: dxil_container_add_features failed\n");
      retval = false;
      goto out;
   }

   if (!dxil_container_add_io_signature(&container,
                                        DXIL_ISG1,
                                        ctx->mod.num_sig_inputs,
                                        ctx->mod.inputs,
                                        ctx->mod.minor_validator >= 7)) {
      debug_printf("D3D12: failed to write input signature\n");
      retval = false;
      goto out;
   }

   if (!dxil_container_add_io_signature(&container,
                                        DXIL_OSG1,
                                        ctx->mod.num_sig_outputs,
                                        ctx->mod.outputs,
                                        ctx->mod.minor_validator >= 7)) {
      debug_printf("D3D12: failed to write output signature\n");
      retval = false;
      goto out;
   }

   if ((ctx->mod.shader_kind == DXIL_HULL_SHADER ||
        ctx->mod.shader_kind == DXIL_DOMAIN_SHADER) &&
       !dxil_container_add_io_signature(&container,
                                        DXIL_PSG1,
                                        ctx->mod.num_sig_patch_consts,
                                        ctx->mod.patch_consts,
                                        ctx->mod.minor_validator >= 7)) {
      debug_printf("D3D12: failed to write patch constant signature\n");
      retval = false;
      goto out;
   }

   struct dxil_validation_state validation_state;
   memset(&validation_state, 0, sizeof(validation_state));
   dxil_fill_validation_state(ctx, &validation_state);

   if (!dxil_container_add_state_validation(&container,&ctx->mod,
                                            &validation_state)) {
      debug_printf("D3D12: failed to write state-validation\n");
      retval = false;
      goto out;
   }

   uint64_t extra_bit_offset = 0;

   if (!dxil_container_add_module(&container, &ctx->mod, &extra_bit_offset)) {
      debug_printf("D3D12: failed to write module\n");
      retval = false;
      goto out;
   }

   if (!dxil_container_write(&container, blob, &extra_bit_offset)) {
      debug_printf("D3D12: dxil_container_write failed\n");
      retval = false;
      goto out;
   }
   dxil_container_finish(&container);

   opts->godot_nir_callbacks->report_sc_extra_bit_offset_fn(extra_bit_offset, opts->godot_nir_callbacks->data);

   if (debug_dxil & DXIL_DEBUG_DUMP_BLOB) {
      static int shader_id = 0;
      char buffer[64];
      snprintf(buffer, sizeof(buffer), "shader_%s_%d.blob",
               get_shader_kind_str(ctx->mod.shader_kind), shader_id++);
      debug_printf("Try to write blob to %s\n", buffer);
      FILE *f = fopen(buffer, "wb");
      if (f) {
         fwrite(blob->data, 1, blob->size, f);
         fclose(f);
      }
   }

out:
   dxil_module_release(&ctx->mod);
   ralloc_free(ctx->ralloc_ctx);
   free(ctx);
   return retval;
}
