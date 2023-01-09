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

 /*
  * See the DirectX Shader Compiler for documentation for DXIL details:
  * https://github.com/Microsoft/DirectXShaderCompiler/blob/master/docs/DXIL.rst
  */

#ifndef DXIL_MODULE_H
#define DXIL_MODULE_H

typedef struct GodotNirCallbacks GodotNirCallbacks;

#ifdef __cplusplus
extern "C" {
#endif

#include "dxil_buffer.h"
#include "dxil_signature.h"

#include "util/list.h"


#define DXIL_SHADER_MAX_IO_ROWS 80

enum dxil_shader_kind {
   DXIL_PIXEL_SHADER = 0,
   DXIL_VERTEX_SHADER = 1,
   DXIL_GEOMETRY_SHADER = 2,
   DXIL_HULL_SHADER = 3,
   DXIL_DOMAIN_SHADER = 4,
   DXIL_COMPUTE_SHADER = 5,
};

extern int debug_dxil;

enum dxil_debug_flags {
   DXIL_DEBUG_VERBOSE    = 1 << 0,
   DXIL_DEBUG_DUMP_BLOB  = 1 << 1,
   DXIL_DEBUG_TRACE      = 1 << 2,
   DXIL_DEBUG_DUMP_MODULE = 1 << 3,
};

enum dxil_bin_opcode {
   DXIL_BINOP_ADD = 0,
   DXIL_BINOP_SUB = 1,
   DXIL_BINOP_MUL = 2,
   DXIL_BINOP_UDIV = 3,
   DXIL_BINOP_SDIV = 4,
   DXIL_BINOP_UREM = 5,
   DXIL_BINOP_SREM = 6,
   DXIL_BINOP_SHL = 7,
   DXIL_BINOP_LSHR = 8,
   DXIL_BINOP_ASHR = 9,
   DXIL_BINOP_AND = 10,
   DXIL_BINOP_OR = 11,
   DXIL_BINOP_XOR = 12,
   DXIL_BINOP_INSTR_COUNT
};

enum dxil_cast_opcode {
   DXIL_CAST_TRUNC = 0,
   DXIL_CAST_ZEXT = 1,
   DXIL_CAST_SEXT = 2,
   DXIL_CAST_FPTOUI = 3,
   DXIL_CAST_FPTOSI = 4,
   DXIL_CAST_UITOFP = 5,
   DXIL_CAST_SITOFP = 6,
   DXIL_CAST_FPTRUNC = 7,
   DXIL_CAST_FPEXT = 8,
   DXIL_CAST_PTRTOINT = 9,
   DXIL_CAST_INTTOPTR = 10,
   DXIL_CAST_BITCAST = 11,
   DXIL_CAST_ADDRSPACECAST = 12,
   DXIL_CAST_INSTR_COUNT
};

enum dxil_cmp_pred {
   DXIL_FCMP_FALSE = 0,
   DXIL_FCMP_OEQ = 1,
   DXIL_FCMP_OGT = 2,
   DXIL_FCMP_OGE = 3,
   DXIL_FCMP_OLT = 4,
   DXIL_FCMP_OLE = 5,
   DXIL_FCMP_ONE = 6,
   DXIL_FCMP_ORD = 7,
   DXIL_FCMP_UNO = 8,
   DXIL_FCMP_UEQ = 9,
   DXIL_FCMP_UGT = 10,
   DXIL_FCMP_UGE = 11,
   DXIL_FCMP_ULT = 12,
   DXIL_FCMP_ULE = 13,
   DXIL_FCMP_UNE = 14,
   DXIL_FCMP_TRUE = 15,
   DXIL_ICMP_EQ = 32,
   DXIL_ICMP_NE = 33,
   DXIL_ICMP_UGT = 34,
   DXIL_ICMP_UGE = 35,
   DXIL_ICMP_ULT = 36,
   DXIL_ICMP_ULE = 37,
   DXIL_ICMP_SGT = 38,
   DXIL_ICMP_SGE = 39,
   DXIL_ICMP_SLT = 40,
   DXIL_ICMP_SLE = 41,
   DXIL_CMP_INSTR_COUNT
};

enum dxil_opt_flags {
  DXIL_UNSAFE_ALGEBRA = (1 << 0),
  DXIL_NO_NANS = (1 << 1),
  DXIL_NO_INFS = (1 << 2),
  DXIL_NO_SIGNED_ZEROS = (1 << 3),
  DXIL_ALLOW_RECIPROCAL = (1 << 4)
};

struct dxil_features {
   unsigned doubles : 1,
            cs_4x_raw_sb : 1,
            uavs_at_every_stage : 1,
            use_64uavs : 1,
            min_precision : 1,
            dx11_1_double_extensions : 1,
            dx11_1_shader_extensions : 1,
            dx9_comparison_filtering : 1,
            tiled_resources : 1,
            stencil_ref : 1,
            inner_coverage : 1,
            typed_uav_load_additional_formats : 1,
            rovs : 1,
            array_layer_from_vs_or_ds : 1,
            wave_ops : 1,
            int64_ops : 1,
            view_id : 1,
            barycentrics : 1,
            native_low_precision : 1,
            shading_rate : 1,
            raytracing_tier_1_1 : 1,
            sampler_feedback : 1,
            atomic_int64_typed : 1,
            atomic_int64_tgsm : 1,
            derivatives_in_mesh_or_amp : 1,
            resource_descriptor_heap_indexing : 1,
            sampler_descriptor_heap_indexing : 1,
            unnamed : 1,
            atomic_int64_heap_resource : 1,
            advanced_texture_ops : 1,
            writable_msaa : 1;
};

struct dxil_shader_info {
   unsigned has_out_position:1;
   unsigned has_out_depth:1;
   unsigned has_per_sample_input:1;
};

struct dxil_func_def {
   struct list_head head;
   const struct dxil_func *func;

   struct list_head instr_list;
   int *basic_block_ids; /* maps from "user" ids to LLVM ids */
   size_t num_basic_block_ids;
   unsigned curr_block;
};

struct dxil_module {
   void *ralloc_ctx;
   enum dxil_shader_kind shader_kind;
   unsigned major_version, minor_version;
   unsigned major_validator, minor_validator;
   struct dxil_features feats;
   unsigned raw_and_structured_buffers : 1;
   struct dxil_shader_info info;

   struct dxil_buffer buf;

   /* The number of entries in the arrays below */
   unsigned num_sig_inputs;
   unsigned num_sig_outputs;
   unsigned num_sig_patch_consts;

   /* The number of "vectors" of elements. This is used to determine the sizes
    * of the dependency tables.
    */
   unsigned num_psv_inputs;
   unsigned num_psv_outputs[4];
   unsigned num_psv_patch_consts;

   struct dxil_signature_record inputs[DXIL_SHADER_MAX_IO_ROWS];
   struct dxil_signature_record outputs[DXIL_SHADER_MAX_IO_ROWS];
   struct dxil_signature_record patch_consts[DXIL_SHADER_MAX_IO_ROWS];

   /* This array is indexed using var->data.driver_location, which
    * is not a direct match to IO rows, since a row is a vec4, and
    * variables can occupy less than that, and several vars can
    * be packed in a row. Hence the x4, but I doubt we can end up
    * with more than 80x4 variables in practice. Maybe this array
    * should be allocated dynamically based on on the maximum
    * driver_location across all input vars.
    */
   unsigned input_mappings[DXIL_SHADER_MAX_IO_ROWS * 4];

   struct dxil_psv_signature_element psv_inputs[DXIL_SHADER_MAX_IO_ROWS];
   struct dxil_psv_signature_element psv_outputs[DXIL_SHADER_MAX_IO_ROWS];
   struct dxil_psv_signature_element psv_patch_consts[DXIL_SHADER_MAX_IO_ROWS];

   struct _mesa_string_buffer *sem_string_table;
   struct dxil_psv_sem_index_table sem_index_table;

   struct {
      unsigned abbrev_width;
      intptr_t offset;
   } blocks[16];
   size_t num_blocks;

   struct list_head type_list;
   struct list_head gvar_list;
   struct list_head func_list;
   struct list_head func_def_list;
   struct list_head attr_set_list;
   struct list_head const_list;
   struct list_head mdnode_list;
   struct list_head md_named_node_list;
   const struct dxil_type *void_type;
   const struct dxil_type *int1_type, *int8_type, *int16_type,
                          *int32_type, *int64_type;
   const struct dxil_type *float16_type, *float32_type, *float64_type;

   struct rb_tree *functions;

   struct dxil_func_def *cur_emitting_func;

   const GodotNirCallbacks *godot_nir_callbacks;
};

struct dxil_instr;
struct dxil_value;

void
dxil_module_init(struct dxil_module *m, void *ralloc_ctx);

void
dxil_module_release(struct dxil_module *m);

const struct dxil_value *
dxil_add_global_var(struct dxil_module *m, const char *name,
                    const struct dxil_type *type,
                    enum dxil_address_space as, int align,
                    const struct dxil_value *value);

const struct dxil_value *
dxil_add_global_ptr_var(struct dxil_module *m, const char *name,
                        const struct dxil_type *type,
                        enum dxil_address_space as, int align,
                        const struct dxil_value *value);

struct dxil_func_def *
dxil_add_function_def(struct dxil_module *m, const char *name,
                      const struct dxil_type *type, unsigned num_blocks);

const struct dxil_func *
dxil_add_function_decl(struct dxil_module *m, const char *name,
                       const struct dxil_type *type,
                       enum dxil_attr_kind attr);

const struct dxil_type *
dxil_module_get_void_type(struct dxil_module *m);

const struct dxil_type *
dxil_module_get_int_type(struct dxil_module *m, unsigned bit_size);

const struct dxil_type *
dxil_module_get_float_type(struct dxil_module *m, unsigned bit_size);

const struct dxil_type *
dxil_module_get_pointer_type(struct dxil_module *m,
                             const struct dxil_type *target);

const struct dxil_type *
dxil_get_overload_type(struct dxil_module *mod, enum overload_type overload);

const struct dxil_type *
dxil_module_get_handle_type(struct dxil_module *m);

const struct dxil_type *
dxil_module_get_cbuf_ret_type(struct dxil_module *mod, enum overload_type overload);

const struct dxil_type *
dxil_module_get_split_double_ret_type(struct dxil_module *mod);

const struct dxil_type *
dxil_module_get_res_type(struct dxil_module *m, enum dxil_resource_kind kind,
                         enum dxil_component_type comp_type, bool readwrite);

const struct dxil_type *
dxil_module_get_resret_type(struct dxil_module *m, enum overload_type overload);

const struct dxil_type *
dxil_module_get_dimret_type(struct dxil_module *m);

const struct dxil_type *
dxil_module_get_samplepos_type(struct dxil_module *m);

const struct dxil_type *
dxil_module_get_res_bind_type(struct dxil_module *m);

const struct dxil_type *
dxil_module_get_res_props_type(struct dxil_module *m);

const struct dxil_type *
dxil_module_get_struct_type(struct dxil_module *m,
                            const char *name,
                            const struct dxil_type **elem_types,
                            size_t num_elem_types);

const struct dxil_type *
dxil_module_get_array_type(struct dxil_module *m,
                           const struct dxil_type *elem_type,
                           size_t num_elems);

const struct dxil_type *
dxil_module_get_vector_type(struct dxil_module *m,
                            const struct dxil_type *elem_type,
                            size_t num_elems);

const struct dxil_type *
dxil_module_add_function_type(struct dxil_module *m,
                              const struct dxil_type *ret_type,
                              const struct dxil_type **arg_types,
                              size_t num_arg_types);

nir_alu_type
dxil_type_to_nir_type(const struct dxil_type *type);

bool
dxil_value_type_equal_to(const struct dxil_value *value,
                         const struct dxil_type *lhs);

bool
dxil_value_type_bitsize_equal_to(const struct dxil_value *value, unsigned bitsize);

const struct dxil_type *
dxil_value_get_type(const struct dxil_value *value);

const struct dxil_value *
dxil_module_get_int1_const(struct dxil_module *m, bool value);

const struct dxil_value *
dxil_module_get_int8_const(struct dxil_module *m, int8_t value);

const struct dxil_value *
dxil_module_get_int16_const(struct dxil_module *m, int16_t value);

const struct dxil_value *
dxil_module_get_int32_const(struct dxil_module *m, int32_t value);

const struct dxil_value *
dxil_module_get_int64_const(struct dxil_module *m, int64_t value);

const struct dxil_value *
dxil_module_get_int_const(struct dxil_module *m, intmax_t value,
                          unsigned bit_size);

const struct dxil_value *
dxil_module_get_float16_const(struct dxil_module *m, uint16_t);

const struct dxil_value *
dxil_module_get_float_const(struct dxil_module *m, float value);

const struct dxil_value *
dxil_module_get_double_const(struct dxil_module *m, double value);

const struct dxil_value *
dxil_module_get_array_const(struct dxil_module *m, const struct dxil_type *type,
                            const struct dxil_value **values);

const struct dxil_value *
dxil_module_get_undef(struct dxil_module *m, const struct dxil_type *type);

const struct dxil_value *
dxil_module_get_res_bind_const(struct dxil_module *m,
                               uint32_t lower_bound,
                               uint32_t upper_bound,
                               uint32_t space,
                               uint8_t class);

const struct dxil_value *
dxil_module_get_res_props_const(struct dxil_module *m,
                                enum dxil_resource_class class,
                                const struct dxil_mdnode *mdnode);

const struct dxil_mdnode *
dxil_get_metadata_string(struct dxil_module *m, const char *str);

const struct dxil_mdnode *
dxil_get_metadata_value(struct dxil_module *m, const struct dxil_type *type,
                        const struct dxil_value *value);

const struct dxil_mdnode *
dxil_get_metadata_func(struct dxil_module *m, const struct dxil_func *func);

const struct dxil_mdnode *
dxil_get_metadata_int1(struct dxil_module *m, bool value);

const struct dxil_mdnode *
dxil_get_metadata_int8(struct dxil_module *m, int8_t value);

const struct dxil_mdnode *
dxil_get_metadata_int32(struct dxil_module *m, int32_t value);

const struct dxil_mdnode *
dxil_get_metadata_int64(struct dxil_module *m, int64_t value);

const struct dxil_mdnode *
dxil_get_metadata_float32(struct dxil_module *m, float value);

const struct dxil_mdnode *
dxil_get_metadata_node(struct dxil_module *m,
                       const struct dxil_mdnode *subnodes[],
                       size_t num_subnodes);

bool
dxil_add_metadata_named_node(struct dxil_module *m, const char *name,
                             const struct dxil_mdnode *subnodes[],
                             size_t num_subnodes);

const struct dxil_value *
dxil_emit_binop(struct dxil_module *m, enum dxil_bin_opcode opcode,
                const struct dxil_value *op0, const struct dxil_value *op1,
                enum dxil_opt_flags flags);

const struct dxil_value *
dxil_emit_cmp(struct dxil_module *m, enum dxil_cmp_pred pred,
              const struct dxil_value *op0, const struct dxil_value *op1);

const struct dxil_value *
dxil_emit_select(struct dxil_module *m,
                const struct dxil_value *op0,
                const struct dxil_value *op1,
                const struct dxil_value *op2);

const struct dxil_value *
dxil_emit_extractval(struct dxil_module *m, const struct dxil_value *src,
                     const unsigned int index);

const struct dxil_value *
dxil_emit_cast(struct dxil_module *m, enum dxil_cast_opcode opcode,
               const struct dxil_type *type,
               const struct dxil_value *value);

bool
dxil_emit_branch(struct dxil_module *m, const struct dxil_value *cond,
                 unsigned true_block, unsigned false_block);

const struct dxil_value *
dxil_instr_get_return_value(struct dxil_instr *instr);

struct dxil_instr *
dxil_emit_phi(struct dxil_module *m, const struct dxil_type *type);

bool
dxil_phi_add_incoming(struct dxil_instr *instr,
                      const struct dxil_value *incoming_values[],
                      const unsigned incoming_blocks[],
                      size_t num_incoming);

const struct dxil_value *
dxil_emit_call(struct dxil_module *m,
               const struct dxil_func *func,
               const struct dxil_value **args, size_t num_args);

bool
dxil_emit_call_void(struct dxil_module *m,
                    const struct dxil_func *func,
                    const struct dxil_value **args, size_t num_args);

bool
dxil_emit_ret_void(struct dxil_module *m);

const struct dxil_value *
dxil_emit_alloca(struct dxil_module *m, const struct dxil_type *alloc_type,
                 const struct dxil_type *size_type,
                 const struct dxil_value *size,
                 unsigned int align);

const struct dxil_value *
dxil_emit_gep_inbounds(struct dxil_module *m,
                       const struct dxil_value **operands,
                       size_t num_operands);

const struct dxil_value *
dxil_emit_load(struct dxil_module *m, const struct dxil_value *ptr,
               unsigned align,
               bool is_volatile);

bool
dxil_emit_store(struct dxil_module *m, const struct dxil_value *value,
                const struct dxil_value *ptr, unsigned align,
                bool is_volatile);

const struct dxil_value *
dxil_emit_cmpxchg(struct dxil_module *m, const struct dxil_value *cmpval,
                  const struct dxil_value *newval,
                  const struct dxil_value *ptr, bool is_volatile,
                  enum dxil_atomic_ordering ordering,
                  enum dxil_sync_scope syncscope);

const struct dxil_value *
dxil_emit_atomicrmw(struct dxil_module *m, const struct dxil_value *value,
                    const struct dxil_value *ptr, enum dxil_rmw_op op,
                    bool is_volatile, enum dxil_atomic_ordering ordering,
                    enum dxil_sync_scope syncscope);

bool
dxil_emit_module(struct dxil_module *m);

#ifdef __cplusplus
}
#endif

#endif
