/*
 * Copyright © 2014 Connor Abbott
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
 *    Connor Abbott (cwabbott0@gmail.com)
 *
 */

#ifndef NIR_H
#define NIR_H

#include "util/hash_table.h"
#include "compiler/glsl/list.h"
#include "util/list.h"
#include "util/log.h"
#include "util/ralloc.h"
#include "util/set.h"
#include "util/bitscan.h"
#include "util/bitset.h"
#include "util/compiler.h"
#include "util/enum_operators.h"
#include "util/macros.h"
#include "util/u_printf.h"
#include "util/format/u_format.h"
#include "compiler/nir_types.h"
#include "compiler/shader_enums.h"
#include "compiler/shader_info.h"
#define XXH_INLINE_ALL
#include "util/xxhash.h"
#include <stdio.h>

#ifndef NDEBUG
#include "util/u_debug.h"
#endif /* NDEBUG */

#include "nir_opcodes.h"

#ifdef __cplusplus
extern "C" {
#endif

extern uint32_t nir_debug;
extern bool nir_debug_print_shader[MESA_SHADER_KERNEL + 1];

#ifndef NDEBUG
#define NIR_DEBUG(flag) unlikely(nir_debug & (NIR_DEBUG_ ## flag))
#else
#define NIR_DEBUG(flag) false
#endif

#define NIR_DEBUG_CLONE                  (1u << 0)
#define NIR_DEBUG_SERIALIZE              (1u << 1)
#define NIR_DEBUG_NOVALIDATE             (1u << 2)
#define NIR_DEBUG_VALIDATE_SSA_DOMINANCE (1u << 3)
#define NIR_DEBUG_TGSI                   (1u << 4)
#define NIR_DEBUG_PRINT_VS               (1u << 5)
#define NIR_DEBUG_PRINT_TCS              (1u << 6)
#define NIR_DEBUG_PRINT_TES              (1u << 7)
#define NIR_DEBUG_PRINT_GS               (1u << 8)
#define NIR_DEBUG_PRINT_FS               (1u << 9)
#define NIR_DEBUG_PRINT_CS               (1u << 10)
#define NIR_DEBUG_PRINT_TS               (1u << 11)
#define NIR_DEBUG_PRINT_MS               (1u << 12)
#define NIR_DEBUG_PRINT_RGS              (1u << 13)
#define NIR_DEBUG_PRINT_AHS              (1u << 14)
#define NIR_DEBUG_PRINT_CHS              (1u << 15)
#define NIR_DEBUG_PRINT_MHS              (1u << 16)
#define NIR_DEBUG_PRINT_IS               (1u << 17)
#define NIR_DEBUG_PRINT_CBS              (1u << 18)
#define NIR_DEBUG_PRINT_KS               (1u << 19)
#define NIR_DEBUG_PRINT_CONSTS           (1u << 20)

#define NIR_DEBUG_PRINT (NIR_DEBUG_PRINT_VS  | \
                         NIR_DEBUG_PRINT_TCS | \
                         NIR_DEBUG_PRINT_TES | \
                         NIR_DEBUG_PRINT_GS  | \
                         NIR_DEBUG_PRINT_FS  | \
                         NIR_DEBUG_PRINT_CS  | \
                         NIR_DEBUG_PRINT_TS  | \
                         NIR_DEBUG_PRINT_MS  | \
                         NIR_DEBUG_PRINT_RGS | \
                         NIR_DEBUG_PRINT_AHS | \
                         NIR_DEBUG_PRINT_CHS | \
                         NIR_DEBUG_PRINT_MHS | \
                         NIR_DEBUG_PRINT_IS  | \
                         NIR_DEBUG_PRINT_CBS | \
                         NIR_DEBUG_PRINT_KS)

#define NIR_FALSE 0u
#define NIR_TRUE (~0u)
#define NIR_MAX_VEC_COMPONENTS 16
#define NIR_MAX_MATRIX_COLUMNS 4
#define NIR_STREAM_PACKED (1 << 8)
typedef uint16_t nir_component_mask_t;

static inline bool
nir_num_components_valid(unsigned num_components)
{
   return (num_components >= 1  &&
           num_components <= 5) ||
           num_components == 8  ||
           num_components == 16;
}

static inline nir_component_mask_t
nir_component_mask(unsigned num_components)
{
   assert(nir_num_components_valid(num_components));
   return (1u << num_components) - 1;
}

void
nir_process_debug_variable(void);

bool nir_component_mask_can_reinterpret(nir_component_mask_t mask,
                                        unsigned old_bit_size,
                                        unsigned new_bit_size);
nir_component_mask_t
nir_component_mask_reinterpret(nir_component_mask_t mask,
                               unsigned old_bit_size,
                               unsigned new_bit_size);

/** Defines a cast function
 *
 * This macro defines a cast function from in_type to out_type where
 * out_type is some structure type that contains a field of type out_type.
 *
 * Note that you have to be a bit careful as the generated cast function
 * destroys constness.
 */
#define NIR_DEFINE_CAST(name, in_type, out_type, field, \
                        type_field, type_value)         \
static inline out_type *                                \
name(const in_type *parent)                             \
{                                                       \
   assert(parent && parent->type_field == type_value);  \
   return exec_node_data(out_type, parent, field);      \
}

struct nir_function;
struct nir_shader;
struct nir_instr;
struct nir_builder;
struct nir_xfb_info;


/**
 * Description of built-in state associated with a uniform
 *
 * \sa nir_variable::state_slots
 */
typedef struct {
   gl_state_index16 tokens[STATE_LENGTH];
   uint16_t swizzle;
} nir_state_slot;

typedef enum {
   nir_var_system_value    = (1 << 0),
   nir_var_uniform         = (1 << 1),
   nir_var_shader_in       = (1 << 2),
   nir_var_shader_out      = (1 << 3),
   nir_var_image           = (1 << 4),
   /** Incoming call or ray payload data for ray-tracing shaders */
   nir_var_shader_call_data = (1 << 5),
   /** Ray hit attributes */
   nir_var_ray_hit_attrib  = (1 << 6),

   /* Modes named nir_var_mem_* have explicit data layout */
   nir_var_mem_ubo          = (1 << 7),
   nir_var_mem_push_const   = (1 << 8),
   nir_var_mem_ssbo         = (1 << 9),
   nir_var_mem_constant     = (1 << 10),
   nir_var_mem_task_payload = (1 << 11),

   /* Generic modes intentionally come last. See encode_dref_modes() in
    * nir_serialize.c for more details.
    */
   nir_var_shader_temp     = (1 << 12),
   nir_var_function_temp   = (1 << 13),
   nir_var_mem_shared      = (1 << 14),
   nir_var_mem_global      = (1 << 15),

   nir_var_mem_generic     = (nir_var_shader_temp |
                              nir_var_function_temp |
                              nir_var_mem_shared |
                              nir_var_mem_global),

   nir_var_read_only_modes = nir_var_shader_in | nir_var_uniform |
                             nir_var_system_value | nir_var_mem_constant |
                             nir_var_mem_ubo,
   /* Modes where vector derefs can be indexed as arrays. nir_var_shader_out is only for mesh
    * stages.
    */
   nir_var_vec_indexable_modes = nir_var_mem_ubo | nir_var_mem_ssbo |
                                 nir_var_mem_shared | nir_var_mem_global |
                                 nir_var_mem_push_const | nir_var_mem_task_payload |
                                 nir_var_shader_out,
   nir_num_variable_modes  = 16,
   nir_var_all             = (1 << nir_num_variable_modes) - 1,
} nir_variable_mode;
MESA_DEFINE_CPP_ENUM_BITFIELD_OPERATORS(nir_variable_mode)

/**
 * Rounding modes.
 */
typedef enum {
   nir_rounding_mode_undef = 0,
   nir_rounding_mode_rtne  = 1, /* round to nearest even */
   nir_rounding_mode_ru    = 2, /* round up */
   nir_rounding_mode_rd    = 3, /* round down */
   nir_rounding_mode_rtz   = 4, /* round towards zero */
} nir_rounding_mode;

/**
 * Ray query values that can read from a RayQueryKHR object.
 */
typedef enum {
   nir_ray_query_value_intersection_type,
   nir_ray_query_value_intersection_t,
   nir_ray_query_value_intersection_instance_custom_index,
   nir_ray_query_value_intersection_instance_id,
   nir_ray_query_value_intersection_instance_sbt_index,
   nir_ray_query_value_intersection_geometry_index,
   nir_ray_query_value_intersection_primitive_index,
   nir_ray_query_value_intersection_barycentrics,
   nir_ray_query_value_intersection_front_face,
   nir_ray_query_value_intersection_object_ray_direction,
   nir_ray_query_value_intersection_object_ray_origin,
   nir_ray_query_value_intersection_object_to_world,
   nir_ray_query_value_intersection_world_to_object,
   nir_ray_query_value_intersection_candidate_aabb_opaque,
   nir_ray_query_value_tmin,
   nir_ray_query_value_flags,
   nir_ray_query_value_world_ray_direction,
   nir_ray_query_value_world_ray_origin,
} nir_ray_query_value;

typedef union {
   bool b;
   float f32;
   double f64;
   int8_t i8;
   uint8_t u8;
   int16_t i16;
   uint16_t u16;
   int32_t i32;
   uint32_t u32;
   int64_t i64;
   uint64_t u64;
} nir_const_value;

#define nir_const_value_to_array(arr, c, components, m) \
{ \
   for (unsigned i = 0; i < components; ++i) \
      arr[i] = c[i].m; \
} while (false)

static inline nir_const_value
nir_const_value_for_raw_uint(uint64_t x, unsigned bit_size)
{
   nir_const_value v;
   memset(&v, 0, sizeof(v));

   switch (bit_size) {
   case 1:  v.b   = x;  break;
   case 8:  v.u8  = x;  break;
   case 16: v.u16 = x;  break;
   case 32: v.u32 = x;  break;
   case 64: v.u64 = x;  break;
   default:
      unreachable("Invalid bit size");
   }

   return v;
}

static inline nir_const_value
nir_const_value_for_int(int64_t i, unsigned bit_size)
{
   nir_const_value v;
   memset(&v, 0, sizeof(v));

   assert(bit_size <= 64);
   if (bit_size < 64) {
      assert(i >= (-(1ll << (bit_size - 1))));
      assert(i < (1ll << (bit_size - 1)));
   }

   return nir_const_value_for_raw_uint(i, bit_size);
}

static inline nir_const_value
nir_const_value_for_uint(uint64_t u, unsigned bit_size)
{
   nir_const_value v;
   memset(&v, 0, sizeof(v));

   assert(bit_size <= 64);
   if (bit_size < 64)
      assert(u < (1ull << bit_size));

   return nir_const_value_for_raw_uint(u, bit_size);
}

static inline nir_const_value
nir_const_value_for_bool(bool b, unsigned bit_size)
{
   /* Booleans use a 0/-1 convention */
   return nir_const_value_for_int(-(int)b, bit_size);
}

/* This one isn't inline because it requires half-float conversion */
nir_const_value nir_const_value_for_float(double b, unsigned bit_size);

static inline int64_t
nir_const_value_as_int(nir_const_value value, unsigned bit_size)
{
   switch (bit_size) {
   /* int1_t uses 0/-1 convention */
   case 1:  return -(int)value.b;
   case 8:  return value.i8;
   case 16: return value.i16;
   case 32: return value.i32;
   case 64: return value.i64;
   default:
      unreachable("Invalid bit size");
   }
}

static inline uint64_t
nir_const_value_as_uint(nir_const_value value, unsigned bit_size)
{
   switch (bit_size) {
   case 1:  return value.b;
   case 8:  return value.u8;
   case 16: return value.u16;
   case 32: return value.u32;
   case 64: return value.u64;
   default:
      unreachable("Invalid bit size");
   }
}

static inline bool
nir_const_value_as_bool(nir_const_value value, unsigned bit_size)
{
   int64_t i = nir_const_value_as_int(value, bit_size);

   /* Booleans of any size use 0/-1 convention */
   assert(i == 0 || i == -1);

   return i;
}

/* This one isn't inline because it requires half-float conversion */
double nir_const_value_as_float(nir_const_value value, unsigned bit_size);

typedef struct nir_constant {
   /**
    * Value of the constant.
    *
    * The field used to back the values supplied by the constant is determined
    * by the type associated with the \c nir_variable.  Constants may be
    * scalars, vectors, or matrices.
    */
   nir_const_value values[NIR_MAX_VEC_COMPONENTS];

   /* we could get this from the var->type but makes clone *much* easier to
    * not have to care about the type.
    */
   unsigned num_elements;

   /* Array elements / Structure Fields */
   struct nir_constant **elements;
} nir_constant;

/**
 * \brief Layout qualifiers for gl_FragDepth.
 *
 * The AMD/ARB_conservative_depth extensions allow gl_FragDepth to be redeclared
 * with a layout qualifier.
 */
typedef enum {
    nir_depth_layout_none, /**< No depth layout is specified. */
    nir_depth_layout_any,
    nir_depth_layout_greater,
    nir_depth_layout_less,
    nir_depth_layout_unchanged
} nir_depth_layout;

/**
 * Enum keeping track of how a variable was declared.
 */
typedef enum {
   /**
    * Normal declaration.
    */
   nir_var_declared_normally = 0,

   /**
    * Variable is implicitly generated by the compiler and should not be
    * visible via the API.
    */
   nir_var_hidden,
} nir_var_declaration_type;

/**
 * Either a uniform, global variable, shader input, or shader output. Based on
 * ir_variable - it should be easy to translate between the two.
 */

typedef struct nir_variable {
   struct exec_node node;

   /**
    * Declared type of the variable
    */
   const struct glsl_type *type;

   /**
    * Declared name of the variable
    */
   char *name;

   struct nir_variable_data {
      /**
       * Storage class of the variable.
       *
       * \sa nir_variable_mode
       */
      unsigned mode:16;

      /**
       * Is the variable read-only?
       *
       * This is set for variables declared as \c const, shader inputs,
       * and uniforms.
       */
      unsigned read_only:1;
      unsigned centroid:1;
      unsigned sample:1;
      unsigned patch:1;
      unsigned invariant:1;

      /**
       * Is the variable a ray query?
       */
      unsigned ray_query:1;

     /**
       * Precision qualifier.
       *
       * In desktop GLSL we do not care about precision qualifiers at all, in
       * fact, the spec says that precision qualifiers are ignored.
       *
       * To make things easy, we make it so that this field is always
       * GLSL_PRECISION_NONE on desktop shaders. This way all the variables
       * have the same precision value and the checks we add in the compiler
       * for this field will never break a desktop shader compile.
       */
      unsigned precision:2;

      /**
       * Has this variable been statically assigned?
       *
       * This answers whether the variable was assigned in any path of
       * the shader during ast_to_hir.  This doesn't answer whether it is
       * still written after dead code removal, nor is it maintained in
       * non-ast_to_hir.cpp (GLSL parsing) paths.
       */
      unsigned assigned:1;

      /**
       * Can this variable be coalesced with another?
       *
       * This is set by nir_lower_io_to_temporaries to say that any
       * copies involving this variable should stay put. Propagating it can
       * duplicate the resulting load/store, which is not wanted, and may
       * result in a load/store of the variable with an indirect offset which
       * the backend may not be able to handle.
       */
      unsigned cannot_coalesce:1;

      /**
       * When separate shader programs are enabled, only input/outputs between
       * the stages of a multi-stage separate program can be safely removed
       * from the shader interface. Other input/outputs must remains active.
       *
       * This is also used to make sure xfb varyings that are unused by the
       * fragment shader are not removed.
       */
      unsigned always_active_io:1;

      /**
       * Interpolation mode for shader inputs / outputs
       *
       * \sa glsl_interp_mode
       */
      unsigned interpolation:3;

      /**
       * If non-zero, then this variable may be packed along with other variables
       * into a single varying slot, so this offset should be applied when
       * accessing components.  For example, an offset of 1 means that the x
       * component of this variable is actually stored in component y of the
       * location specified by \c location.
       */
      unsigned location_frac:2;

      /**
       * If true, this variable represents an array of scalars that should
       * be tightly packed.  In other words, consecutive array elements
       * should be stored one component apart, rather than one slot apart.
       */
      unsigned compact:1;

      /**
       * Whether this is a fragment shader output implicitly initialized with
       * the previous contents of the specified render target at the
       * framebuffer location corresponding to this shader invocation.
       */
      unsigned fb_fetch_output:1;

      /**
       * Non-zero if this variable is considered bindless as defined by
       * ARB_bindless_texture.
       */
      unsigned bindless:1;

      /**
       * Was an explicit binding set in the shader?
       */
      unsigned explicit_binding:1;

      /**
       * Was the location explicitly set in the shader?
       *
       * If the location is explicitly set in the shader, it \b cannot be changed
       * by the linker or by the API (e.g., calls to \c glBindAttribLocation have
       * no effect).
       */
      unsigned explicit_location:1;

      /**
       * Is this varying used by transform feedback?
       *
       * This is used by the linker to decide if it's safe to pack the varying.
       */
      unsigned is_xfb:1;

      /**
       * Is this varying used only by transform feedback?
       *
       * This is used by the linker to decide if its safe to pack the varying.
       */
      unsigned is_xfb_only:1;

      /**
       * Was a transfer feedback buffer set in the shader?
       */
      unsigned explicit_xfb_buffer:1;

      /**
       * Was a transfer feedback stride set in the shader?
       */
      unsigned explicit_xfb_stride:1;

      /**
       * Was an explicit offset set in the shader?
       */
      unsigned explicit_offset:1;

      /**
       * Layout of the matrix.  Uses glsl_matrix_layout values.
       */
      unsigned matrix_layout:2;

      /**
       * Non-zero if this variable was created by lowering a named interface
       * block.
       */
      unsigned from_named_ifc_block:1;

      /**
       * Non-zero if the variable must be a shader input. This is useful for
       * constraints on function parameters.
       */
      unsigned must_be_shader_input:1;

      /**
       * How the variable was declared.  See nir_var_declaration_type.
       *
       * This is used to detect variables generated by the compiler, so should
       * not be visible via the API.
       */
      unsigned how_declared:2;

      /**
       * Is this variable per-view?  If so, we know it must be an array with
       * size corresponding to the number of views.
       */
      unsigned per_view:1;

      /**
       * Whether the variable is per-primitive.
       * Can be use by Mesh Shader outputs and corresponding Fragment Shader inputs.
       */
      unsigned per_primitive:1;

      /**
       * \brief Layout qualifier for gl_FragDepth. See nir_depth_layout.
       *
       * This is not equal to \c ir_depth_layout_none if and only if this
       * variable is \c gl_FragDepth and a layout qualifier is specified.
       */
      unsigned depth_layout:3;

      /**
       * Vertex stream output identifier.
       *
       * For packed outputs, NIR_STREAM_PACKED is set and bits [2*i+1,2*i]
       * indicate the stream of the i-th component.
       */
      unsigned stream:9;

      /**
       * See gl_access_qualifier.
       *
       * Access flags for memory variables (SSBO/global), image uniforms, and
       * bindless images in uniforms/inputs/outputs.
       */
      unsigned access:9;

      /**
       * Descriptor set binding for sampler or UBO.
       */
      unsigned descriptor_set:5;

      /**
       * output index for dual source blending.
       */
      unsigned index;

      /**
       * Initial binding point for a sampler or UBO.
       *
       * For array types, this represents the binding point for the first element.
       */
      unsigned binding;

      /**
       * Storage location of the base of this variable
       *
       * The precise meaning of this field depends on the nature of the variable.
       *
       *   - Vertex shader input: one of the values from \c gl_vert_attrib.
       *   - Vertex shader output: one of the values from \c gl_varying_slot.
       *   - Geometry shader input: one of the values from \c gl_varying_slot.
       *   - Geometry shader output: one of the values from \c gl_varying_slot.
       *   - Fragment shader input: one of the values from \c gl_varying_slot.
       *   - Fragment shader output: one of the values from \c gl_frag_result.
       *   - Task shader output: one of the values from \c gl_varying_slot.
       *   - Mesh shader input: one of the values from \c gl_varying_slot.
       *   - Mesh shader output: one of the values from \c gl_varying_slot.
       *   - Uniforms: Per-stage uniform slot number for default uniform block.
       *   - Uniforms: Index within the uniform block definition for UBO members.
       *   - Non-UBO Uniforms: uniform slot number.
       *   - Other: This field is not currently used.
       *
       * If the variable is a uniform, shader input, or shader output, and the
       * slot has not been assigned, the value will be -1.
       */
      int location;

      /**
       * The actual location of the variable in the IR. Only valid for inputs,
       * outputs, uniforms (including samplers and images), and for UBO and SSBO
       * variables in GLSL.
       */
      unsigned driver_location;

      /**
       * Location an atomic counter or transform feedback is stored at.
       */
      unsigned offset;

      union {
         struct {
            /** Image internal format if specified explicitly, otherwise PIPE_FORMAT_NONE. */
            enum pipe_format format;
         } image;

         struct {
            /**
             * For OpenCL inline samplers. See cl_sampler_addressing_mode and cl_sampler_filter_mode
             */
            unsigned is_inline_sampler : 1;
            unsigned addressing_mode : 3;
            unsigned normalized_coordinates : 1;
            unsigned filter_mode : 1;
         } sampler;

         struct {
            /**
             * Transform feedback buffer.
             */
            uint16_t buffer:2;

            /**
             * Transform feedback stride.
             */
            uint16_t stride;
         } xfb;
      };
   } data;

   /**
    * Identifier for this variable generated by nir_index_vars() that is unique
    * among other variables in the same exec_list.
    */
   unsigned index;

   /* Number of nir_variable_data members */
   uint16_t num_members;

   /**
    * Built-in state that backs this uniform
    *
    * Once set at variable creation, \c state_slots must remain invariant.
    * This is because, ideally, this array would be shared by all clones of
    * this variable in the IR tree.  In other words, we'd really like for it
    * to be a fly-weight.
    *
    * If the variable is not a uniform, \c num_state_slots will be zero and
    * \c state_slots will be \c NULL.
    */
   /*@{*/
   uint16_t num_state_slots;    /**< Number of state slots used */
   nir_state_slot *state_slots;  /**< State descriptors. */
   /*@}*/

   /**
    * Constant expression assigned in the initializer of the variable
    *
    * This field should only be used temporarily by creators of NIR shaders
    * and then nir_lower_variable_initializers can be used to get rid of them.
    * Most of the rest of NIR ignores this field or asserts that it's NULL.
    */
   nir_constant *constant_initializer;

   /**
    * Global variable assigned in the initializer of the variable
    * This field should only be used temporarily by creators of NIR shaders
    * and then nir_lower_variable_initializers can be used to get rid of them.
    * Most of the rest of NIR ignores this field or asserts that it's NULL.
    */
   struct nir_variable *pointer_initializer;

   /**
    * For variables that are in an interface block or are an instance of an
    * interface block, this is the \c GLSL_TYPE_INTERFACE type for that block.
    *
    * \sa ir_variable::location
    */
   const struct glsl_type *interface_type;

   /**
    * Description of per-member data for per-member struct variables
    *
    * This is used for variables which are actually an amalgamation of
    * multiple entities such as a struct of built-in values or a struct of
    * inputs each with their own layout specifier.  This is only allowed on
    * variables with a struct or array of array of struct type.
    */
   struct nir_variable_data *members;
} nir_variable;

static inline bool
_nir_shader_variable_has_mode(nir_variable *var, unsigned modes)
{
   /* This isn't a shader variable */
   assert(!(modes & nir_var_function_temp));
   return var->data.mode & modes;
}

#define nir_foreach_variable_in_list(var, var_list) \
   foreach_list_typed(nir_variable, var, node, var_list)

#define nir_foreach_variable_in_list_safe(var, var_list) \
   foreach_list_typed_safe(nir_variable, var, node, var_list)

#define nir_foreach_variable_in_shader(var, shader) \
   nir_foreach_variable_in_list(var, &(shader)->variables)

#define nir_foreach_variable_in_shader_safe(var, shader) \
   nir_foreach_variable_in_list_safe(var, &(shader)->variables)

#define nir_foreach_variable_with_modes(var, shader, modes) \
   nir_foreach_variable_in_shader(var, shader) \
      if (_nir_shader_variable_has_mode(var, modes))

#define nir_foreach_variable_with_modes_safe(var, shader, modes) \
   nir_foreach_variable_in_shader_safe(var, shader) \
      if (_nir_shader_variable_has_mode(var, modes))

#define nir_foreach_shader_in_variable(var, shader) \
   nir_foreach_variable_with_modes(var, shader, nir_var_shader_in)

#define nir_foreach_shader_in_variable_safe(var, shader) \
   nir_foreach_variable_with_modes_safe(var, shader, nir_var_shader_in)

#define nir_foreach_shader_out_variable(var, shader) \
   nir_foreach_variable_with_modes(var, shader, nir_var_shader_out)

#define nir_foreach_shader_out_variable_safe(var, shader) \
   nir_foreach_variable_with_modes_safe(var, shader, nir_var_shader_out)

#define nir_foreach_uniform_variable(var, shader) \
   nir_foreach_variable_with_modes(var, shader, nir_var_uniform)

#define nir_foreach_uniform_variable_safe(var, shader) \
   nir_foreach_variable_with_modes_safe(var, shader, nir_var_uniform)

#define nir_foreach_image_variable(var, shader) \
   nir_foreach_variable_with_modes(var, shader, nir_var_image)

#define nir_foreach_image_variable_safe(var, shader) \
   nir_foreach_variable_with_modes_safe(var, shader, nir_var_image)

static inline bool
nir_variable_is_global(const nir_variable *var)
{
   return var->data.mode != nir_var_function_temp;
}

typedef struct nir_register {
   struct exec_node node;

   unsigned num_components; /** < number of vector components */
   unsigned num_array_elems; /** < size of array (0 for no array) */

   /* The bit-size of each channel; must be one of 8, 16, 32, or 64 */
   uint8_t bit_size;

   /**
    * True if this register may have different values in different SIMD
    * invocations of the shader.
    */
   bool divergent;

   /** generic register index. */
   unsigned index;

   /** set of nir_srcs where this register is used (read from) */
   struct list_head uses;

   /** set of nir_dests where this register is defined (written to) */
   struct list_head defs;

   /** set of nir_ifs where this register is used as a condition */
   struct list_head if_uses;
} nir_register;

#define nir_foreach_register(reg, reg_list) \
   foreach_list_typed(nir_register, reg, node, reg_list)
#define nir_foreach_register_safe(reg, reg_list) \
   foreach_list_typed_safe(nir_register, reg, node, reg_list)

typedef enum PACKED {
   nir_instr_type_alu,
   nir_instr_type_deref,
   nir_instr_type_call,
   nir_instr_type_tex,
   nir_instr_type_intrinsic,
   nir_instr_type_load_const,
   nir_instr_type_jump,
   nir_instr_type_ssa_undef,
   nir_instr_type_phi,
   nir_instr_type_parallel_copy,
} nir_instr_type;

typedef struct nir_instr {
   struct exec_node node;
   struct nir_block *block;
   nir_instr_type type;

   /* A temporary for optimization and analysis passes to use for storing
    * flags.  For instance, DCE uses this to store the "dead/live" info.
    */
   uint8_t pass_flags;

   /** generic instruction index. */
   uint32_t index;
} nir_instr;

static inline nir_instr *
nir_instr_next(nir_instr *instr)
{
   struct exec_node *next = exec_node_get_next(&instr->node);
   if (exec_node_is_tail_sentinel(next))
      return NULL;
   else
      return exec_node_data(nir_instr, next, node);
}

static inline nir_instr *
nir_instr_prev(nir_instr *instr)
{
   struct exec_node *prev = exec_node_get_prev(&instr->node);
   if (exec_node_is_head_sentinel(prev))
      return NULL;
   else
      return exec_node_data(nir_instr, prev, node);
}

static inline bool
nir_instr_is_first(const nir_instr *instr)
{
   return exec_node_is_head_sentinel(exec_node_get_prev_const(&instr->node));
}

static inline bool
nir_instr_is_last(const nir_instr *instr)
{
   return exec_node_is_tail_sentinel(exec_node_get_next_const(&instr->node));
}

typedef struct nir_ssa_def {
   /** Instruction which produces this SSA value. */
   nir_instr *parent_instr;

   /** set of nir_instrs where this register is used (read from) */
   struct list_head uses;

   /** set of nir_ifs where this register is used as a condition */
   struct list_head if_uses;

   /** generic SSA definition index. */
   unsigned index;

   uint8_t num_components;

   /* The bit-size of each channel; must be one of 8, 16, 32, or 64 */
   uint8_t bit_size;

   /**
    * True if this SSA value may have different values in different SIMD
    * invocations of the shader.  This is set by nir_divergence_analysis.
    */
   bool divergent;
} nir_ssa_def;

struct nir_src;

typedef struct {
   nir_register *reg;
   struct nir_src *indirect; /** < NULL for no indirect offset */
   unsigned base_offset;

   /* TODO use-def chain goes here */
} nir_reg_src;

typedef struct {
   nir_instr *parent_instr;
   struct list_head def_link;

   nir_register *reg;
   struct nir_src *indirect; /** < NULL for no indirect offset */
   unsigned base_offset;

   /* TODO def-use chain goes here */
} nir_reg_dest;

struct nir_if;

typedef struct nir_src {
   union {
      /** Instruction that consumes this value as a source. */
      nir_instr *parent_instr;
      struct nir_if *parent_if;
   };

   struct list_head use_link;

   union {
      nir_reg_src reg;
      nir_ssa_def *ssa;
   };

   bool is_ssa;
} nir_src;

static inline nir_src
nir_src_init(void)
{
   nir_src src = { { NULL } };
   return src;
}

#define NIR_SRC_INIT nir_src_init()

#define nir_foreach_use(src, reg_or_ssa_def) \
   list_for_each_entry(nir_src, src, &(reg_or_ssa_def)->uses, use_link)

#define nir_foreach_use_safe(src, reg_or_ssa_def) \
   list_for_each_entry_safe(nir_src, src, &(reg_or_ssa_def)->uses, use_link)

#define nir_foreach_if_use(src, reg_or_ssa_def) \
   list_for_each_entry(nir_src, src, &(reg_or_ssa_def)->if_uses, use_link)

#define nir_foreach_if_use_safe(src, reg_or_ssa_def) \
   list_for_each_entry_safe(nir_src, src, &(reg_or_ssa_def)->if_uses, use_link)

typedef struct {
   union {
      nir_reg_dest reg;
      nir_ssa_def ssa;
   };

   bool is_ssa;
} nir_dest;

static inline nir_dest
nir_dest_init(void)
{
   nir_dest dest = { { { NULL } } };
   return dest;
}

#define NIR_DEST_INIT nir_dest_init()

#define nir_foreach_def(dest, reg) \
   list_for_each_entry(nir_dest, dest, &(reg)->defs, reg.def_link)

#define nir_foreach_def_safe(dest, reg) \
   list_for_each_entry_safe(nir_dest, dest, &(reg)->defs, reg.def_link)

static inline nir_src
nir_src_for_ssa(nir_ssa_def *def)
{
   nir_src src = NIR_SRC_INIT;

   src.is_ssa = true;
   src.ssa = def;

   return src;
}

static inline nir_src
nir_src_for_reg(nir_register *reg)
{
   nir_src src = NIR_SRC_INIT;

   src.is_ssa = false;
   src.reg.reg = reg;
   src.reg.indirect = NULL;
   src.reg.base_offset = 0;

   return src;
}

static inline nir_dest
nir_dest_for_reg(nir_register *reg)
{
   nir_dest dest = NIR_DEST_INIT;

   dest.reg.reg = reg;

   return dest;
}

static inline unsigned
nir_src_bit_size(nir_src src)
{
   return src.is_ssa ? src.ssa->bit_size : src.reg.reg->bit_size;
}

static inline unsigned
nir_src_num_components(nir_src src)
{
   return src.is_ssa ? src.ssa->num_components : src.reg.reg->num_components;
}

static inline bool
nir_src_is_const(nir_src src)
{
   return src.is_ssa &&
          src.ssa->parent_instr->type == nir_instr_type_load_const;
}

static inline bool
nir_src_is_undef(nir_src src)
{
   return src.is_ssa &&
          src.ssa->parent_instr->type == nir_instr_type_ssa_undef;
}

static inline bool
nir_src_is_divergent(nir_src src)
{
   return src.is_ssa ? src.ssa->divergent : src.reg.reg->divergent;
}

static inline unsigned
nir_dest_bit_size(nir_dest dest)
{
   return dest.is_ssa ? dest.ssa.bit_size : dest.reg.reg->bit_size;
}

static inline unsigned
nir_dest_num_components(nir_dest dest)
{
   return dest.is_ssa ? dest.ssa.num_components : dest.reg.reg->num_components;
}

static inline bool
nir_dest_is_divergent(nir_dest dest)
{
   return dest.is_ssa ? dest.ssa.divergent : dest.reg.reg->divergent;
}

/* Are all components the same, ie. .xxxx */
static inline bool
nir_is_same_comp_swizzle(uint8_t *swiz, unsigned nr_comp)
{
   for (unsigned i = 1; i < nr_comp; i++)
      if (swiz[i] != swiz[0])
         return false;
   return true;
}

/* Are all components sequential, ie. .yzw */
static inline bool
nir_is_sequential_comp_swizzle(uint8_t *swiz, unsigned nr_comp)
{
   for (unsigned i = 1; i < nr_comp; i++)
      if (swiz[i] != (swiz[0] + i))
         return false;
   return true;
}

void nir_src_copy(nir_src *dest, const nir_src *src, nir_instr *instr);
void nir_dest_copy(nir_dest *dest, const nir_dest *src, nir_instr *instr);

typedef struct {
   /** Base source */
   nir_src src;

   /**
    * \name input modifiers
    */
   /*@{*/
   /**
    * For inputs interpreted as floating point, flips the sign bit. For
    * inputs interpreted as integers, performs the two's complement negation.
    */
   bool negate;

   /**
    * Clears the sign bit for floating point values, and computes the integer
    * absolute value for integers. Note that the negate modifier acts after
    * the absolute value modifier, therefore if both are set then all inputs
    * will become negative.
    */
   bool abs;
   /*@}*/

   /**
    * For each input component, says which component of the register it is
    * chosen from.
    *
    * Note that which elements of the swizzle are used and which are ignored
    * are based on the write mask for most opcodes - for example, a statement
    * like "foo.xzw = bar.zyx" would have a writemask of 1101b and a swizzle
    * of {2, 1, x, 0} where x means "don't care."
    */
   uint8_t swizzle[NIR_MAX_VEC_COMPONENTS];
} nir_alu_src;

typedef struct {
   /** Base destination */
   nir_dest dest;

   /**
    * Saturate output modifier
    *
    * Only valid for opcodes that output floating-point numbers. Clamps the
    * output to between 0.0 and 1.0 inclusive.
    */
   bool saturate;

   /**
    * Write-mask
    *
    * Ignored if dest.is_ssa is true
    */
   nir_component_mask_t write_mask;
} nir_alu_dest;

/** NIR sized and unsized types
 *
 * The values in this enum are carefully chosen so that the sized type is
 * just the unsized type OR the number of bits.
 */
typedef enum PACKED {
   nir_type_invalid = 0, /* Not a valid type */
   nir_type_int =       2,
   nir_type_uint =      4,
   nir_type_bool =      6,
   nir_type_float =     128,
   nir_type_bool1 =     1  | nir_type_bool,
   nir_type_bool8 =     8  | nir_type_bool,
   nir_type_bool16 =    16 | nir_type_bool,
   nir_type_bool32 =    32 | nir_type_bool,
   nir_type_int1 =      1  | nir_type_int,
   nir_type_int8 =      8  | nir_type_int,
   nir_type_int16 =     16 | nir_type_int,
   nir_type_int32 =     32 | nir_type_int,
   nir_type_int64 =     64 | nir_type_int,
   nir_type_uint1 =     1  | nir_type_uint,
   nir_type_uint8 =     8  | nir_type_uint,
   nir_type_uint16 =    16 | nir_type_uint,
   nir_type_uint32 =    32 | nir_type_uint,
   nir_type_uint64 =    64 | nir_type_uint,
   nir_type_float16 =   16 | nir_type_float,
   nir_type_float32 =   32 | nir_type_float,
   nir_type_float64 =   64 | nir_type_float,
} nir_alu_type;

#define NIR_ALU_TYPE_SIZE_MASK 0x79
#define NIR_ALU_TYPE_BASE_TYPE_MASK 0x86

static inline unsigned
nir_alu_type_get_type_size(nir_alu_type type)
{
   return type & NIR_ALU_TYPE_SIZE_MASK;
}

static inline nir_alu_type
nir_alu_type_get_base_type(nir_alu_type type)
{
   return (nir_alu_type)(type & NIR_ALU_TYPE_BASE_TYPE_MASK);
}

nir_alu_type
nir_get_nir_type_for_glsl_base_type(enum glsl_base_type base_type);

static inline nir_alu_type
nir_get_nir_type_for_glsl_type(const struct glsl_type *type)
{
   return nir_get_nir_type_for_glsl_base_type(glsl_get_base_type(type));
}

enum glsl_base_type
nir_get_glsl_base_type_for_nir_type(nir_alu_type base_type);

nir_op nir_type_conversion_op(nir_alu_type src, nir_alu_type dst,
                              nir_rounding_mode rnd);

nir_op
nir_op_vec(unsigned components);

bool
nir_op_is_vec(nir_op op);

static inline bool
nir_is_float_control_signed_zero_inf_nan_preserve(unsigned execution_mode, unsigned bit_size)
{
    return (16 == bit_size && execution_mode & FLOAT_CONTROLS_SIGNED_ZERO_INF_NAN_PRESERVE_FP16) ||
        (32 == bit_size && execution_mode & FLOAT_CONTROLS_SIGNED_ZERO_INF_NAN_PRESERVE_FP32) ||
        (64 == bit_size && execution_mode & FLOAT_CONTROLS_SIGNED_ZERO_INF_NAN_PRESERVE_FP64);
}

static inline bool
nir_is_denorm_flush_to_zero(unsigned execution_mode, unsigned bit_size)
{
    return (16 == bit_size && execution_mode & FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP16) ||
        (32 == bit_size && execution_mode & FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32) ||
        (64 == bit_size && execution_mode & FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP64);
}

static inline bool
nir_is_denorm_preserve(unsigned execution_mode, unsigned bit_size)
{
    return (16 == bit_size && execution_mode & FLOAT_CONTROLS_DENORM_PRESERVE_FP16) ||
        (32 == bit_size && execution_mode & FLOAT_CONTROLS_DENORM_PRESERVE_FP32) ||
        (64 == bit_size && execution_mode & FLOAT_CONTROLS_DENORM_PRESERVE_FP64);
}

static inline bool
nir_is_rounding_mode_rtne(unsigned execution_mode, unsigned bit_size)
{
    return (16 == bit_size && execution_mode & FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP16) ||
        (32 == bit_size && execution_mode & FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP32) ||
        (64 == bit_size && execution_mode & FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP64);
}

static inline bool
nir_is_rounding_mode_rtz(unsigned execution_mode, unsigned bit_size)
{
    return (16 == bit_size && execution_mode & FLOAT_CONTROLS_ROUNDING_MODE_RTZ_FP16) ||
        (32 == bit_size && execution_mode & FLOAT_CONTROLS_ROUNDING_MODE_RTZ_FP32) ||
        (64 == bit_size && execution_mode & FLOAT_CONTROLS_ROUNDING_MODE_RTZ_FP64);
}

static inline bool
nir_has_any_rounding_mode_rtz(unsigned execution_mode)
{
    return (execution_mode & FLOAT_CONTROLS_ROUNDING_MODE_RTZ_FP16) ||
        (execution_mode & FLOAT_CONTROLS_ROUNDING_MODE_RTZ_FP32) ||
        (execution_mode & FLOAT_CONTROLS_ROUNDING_MODE_RTZ_FP64);
}

static inline bool
nir_has_any_rounding_mode_rtne(unsigned execution_mode)
{
    return (execution_mode & FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP16) ||
        (execution_mode & FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP32) ||
        (execution_mode & FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP64);
}

static inline nir_rounding_mode
nir_get_rounding_mode_from_float_controls(unsigned execution_mode,
                                          nir_alu_type type)
{
   if (nir_alu_type_get_base_type(type) != nir_type_float)
      return nir_rounding_mode_undef;

   unsigned bit_size = nir_alu_type_get_type_size(type);

   if (nir_is_rounding_mode_rtz(execution_mode, bit_size))
      return nir_rounding_mode_rtz;
   if (nir_is_rounding_mode_rtne(execution_mode, bit_size))
      return nir_rounding_mode_rtne;
   return nir_rounding_mode_undef;
}

static inline bool
nir_has_any_rounding_mode_enabled(unsigned execution_mode)
{
   bool result =
      nir_has_any_rounding_mode_rtne(execution_mode) ||
      nir_has_any_rounding_mode_rtz(execution_mode);
   return result;
}

typedef enum {
   /**
    * Operation where the first two sources are commutative.
    *
    * For 2-source operations, this just mathematical commutativity.  Some
    * 3-source operations, like ffma, are only commutative in the first two
    * sources.
    */
   NIR_OP_IS_2SRC_COMMUTATIVE = (1 << 0),

   /**
    * Operation is associative
    */
   NIR_OP_IS_ASSOCIATIVE = (1 << 1),

   /**
    * Operation where src[0] is used to select src[1] on true or src[2] false.
    * src[0] may be Boolean, or it may be another type used in an implicit
    * comparison.
    */
   NIR_OP_IS_SELECTION = (1 << 2),
} nir_op_algebraic_property;

/* vec16 is the widest ALU op in NIR, making the max number of input of ALU
 * instructions to be the same as NIR_MAX_VEC_COMPONENTS.
 */
#define NIR_ALU_MAX_INPUTS NIR_MAX_VEC_COMPONENTS

typedef struct nir_op_info {
   /** Name of the NIR ALU opcode */
   const char *name;

   /** Number of inputs (sources) */
   uint8_t num_inputs;

   /**
    * The number of components in the output
    *
    * If non-zero, this is the size of the output and input sizes are
    * explicitly given; swizzle and writemask are still in effect, but if
    * the output component is masked out, then the input component may
    * still be in use.
    *
    * If zero, the opcode acts in the standard, per-component manner; the
    * operation is performed on each component (except the ones that are
    * masked out) with the input being taken from the input swizzle for
    * that component.
    *
    * The size of some of the inputs may be given (i.e. non-zero) even
    * though output_size is zero; in that case, the inputs with a zero
    * size act per-component, while the inputs with non-zero size don't.
    */
   uint8_t output_size;

   /**
    * The type of vector that the instruction outputs. Note that the
    * staurate modifier is only allowed on outputs with the float type.
    */
   nir_alu_type output_type;

   /**
    * The number of components in each input
    *
    * See nir_op_infos::output_size for more detail about the relationship
    * between input and output sizes.
    */
   uint8_t input_sizes[NIR_ALU_MAX_INPUTS];

   /**
    * The type of vector that each input takes. Note that negate and
    * absolute value are only allowed on inputs with int or float type and
    * behave differently on the two.
    */
   nir_alu_type input_types[NIR_ALU_MAX_INPUTS];

   /** Algebraic properties of this opcode */
   nir_op_algebraic_property algebraic_properties;

   /** Whether this represents a numeric conversion opcode */
   bool is_conversion;
} nir_op_info;

/** Metadata for each nir_op, indexed by opcode */
extern const nir_op_info nir_op_infos[nir_num_opcodes];

static inline bool
nir_op_is_selection(nir_op op)
{
   return (nir_op_infos[op].algebraic_properties & NIR_OP_IS_SELECTION) != 0;
}

typedef struct nir_alu_instr {
   /** Base instruction */
   nir_instr instr;

   /** Opcode */
   nir_op op;

   /** Indicates that this ALU instruction generates an exact value
    *
    * This is kind of a mixture of GLSL "precise" and "invariant" and not
    * really equivalent to either.  This indicates that the value generated by
    * this operation is high-precision and any code transformations that touch
    * it must ensure that the resulting value is bit-for-bit identical to the
    * original.
    */
   bool exact:1;

   /**
    * Indicates that this instruction doese not cause signed integer wrapping
    * to occur, in the form of overflow or underflow.
    */
   bool no_signed_wrap:1;

   /**
    * Indicates that this instruction does not cause unsigned integer wrapping
    * to occur, in the form of overflow or underflow.
    */
   bool no_unsigned_wrap:1;

   /** Destination */
   nir_alu_dest dest;

   /** Sources
    *
    * The size of the array is given by nir_op_info::num_inputs.
    */
   nir_alu_src src[];
} nir_alu_instr;

void nir_alu_src_copy(nir_alu_src *dest, const nir_alu_src *src,
                      nir_alu_instr *instr);
void nir_alu_dest_copy(nir_alu_dest *dest, const nir_alu_dest *src,
                       nir_alu_instr *instr);

bool nir_alu_instr_is_copy(nir_alu_instr *instr);

/* is this source channel used? */
bool
nir_alu_instr_channel_used(const nir_alu_instr *instr, unsigned src,
                           unsigned channel);
nir_component_mask_t
nir_alu_instr_src_read_mask(const nir_alu_instr *instr, unsigned src);
/**
 * Get the number of channels used for a source
 */
unsigned
nir_ssa_alu_instr_src_components(const nir_alu_instr *instr, unsigned src);

bool
nir_alu_instr_is_comparison(const nir_alu_instr *instr);

bool nir_const_value_negative_equal(nir_const_value c1, nir_const_value c2,
                                    nir_alu_type full_type);

bool nir_alu_srcs_equal(const nir_alu_instr *alu1, const nir_alu_instr *alu2,
                        unsigned src1, unsigned src2);

bool nir_alu_srcs_negative_equal(const nir_alu_instr *alu1,
                                 const nir_alu_instr *alu2,
                                 unsigned src1, unsigned src2);

bool nir_alu_src_is_trivial_ssa(const nir_alu_instr *alu, unsigned srcn);

typedef enum {
   nir_deref_type_var,
   nir_deref_type_array,
   nir_deref_type_array_wildcard,
   nir_deref_type_ptr_as_array,
   nir_deref_type_struct,
   nir_deref_type_cast,
} nir_deref_type;

typedef struct {
   nir_instr instr;

   /** The type of this deref instruction */
   nir_deref_type deref_type;

   /** Bitmask what modes the underlying variable might be
    *
    * For OpenCL-style generic pointers, we may not know exactly what mode it
    * is at any given point in time in the compile process.  This bitfield
    * contains the set of modes which it MAY be.
    *
    * Generally, this field should not be accessed directly.  Use one of the
    * nir_deref_mode_ helpers instead.
    */
   nir_variable_mode modes;

   /** The dereferenced type of the resulting pointer value */
   const struct glsl_type *type;

   union {
      /** Variable being dereferenced if deref_type is a deref_var */
      nir_variable *var;

      /** Parent deref if deref_type is not deref_var */
      nir_src parent;
   };

   /** Additional deref parameters */
   union {
      struct {
         nir_src index;
         bool in_bounds;
      } arr;

      struct {
         unsigned index;
      } strct;

      struct {
         unsigned ptr_stride;
         unsigned align_mul;
         unsigned align_offset;
      } cast;
   };

   /** Destination to store the resulting "pointer" */
   nir_dest dest;
} nir_deref_instr;

/** Returns true if deref might have one of the given modes
 *
 * For multi-mode derefs, this returns true if any of the possible modes for
 * the deref to have any of the specified modes.  This function returning true
 * does NOT mean that the deref definitely has one of those modes.  It simply
 * means that, with the best information we have at the time, it might.
 */
static inline bool
nir_deref_mode_may_be(const nir_deref_instr *deref, nir_variable_mode modes)
{
   assert(!(modes & ~nir_var_all));
   assert(deref->modes != 0);
   return deref->modes & modes;
}

/** Returns true if deref must have one of the given modes
 *
 * For multi-mode derefs, this returns true if NIR can prove that the given
 * deref has one of the specified modes.  This function returning false does
 * NOT mean that deref doesn't have one of the given mode.  It very well may
 * have one of those modes, we just don't have enough information to prove
 * that it does for sure.
 */
static inline bool
nir_deref_mode_must_be(const nir_deref_instr *deref, nir_variable_mode modes)
{
   assert(!(modes & ~nir_var_all));
   assert(deref->modes != 0);
   return !(deref->modes & ~modes);
}

/** Returns true if deref has the given mode
 *
 * This returns true if the deref has exactly the mode specified.  If the
 * deref may have that mode but may also have a different mode (i.e. modes has
 * multiple bits set), this will assert-fail.
 *
 * If you're confused about which nir_deref_mode_ helper to use, use this one
 * or nir_deref_mode_is_one_of below.
 */
static inline bool
nir_deref_mode_is(const nir_deref_instr *deref, nir_variable_mode mode)
{
   assert(util_bitcount(mode) == 1 && (mode & nir_var_all));
   assert(deref->modes != 0);

   /* This is only for "simple" cases so, if modes might interact with this
    * deref then the deref has to have a single mode.
    */
   if (nir_deref_mode_may_be(deref, mode)) {
      assert(util_bitcount(deref->modes) == 1);
      assert(deref->modes == mode);
   }

   return deref->modes == mode;
}

/** Returns true if deref has one of the given modes
 *
 * This returns true if the deref has exactly one possible mode and that mode
 * is one of the modes specified.  If the deref may have one of those modes
 * but may also have a different mode (i.e. modes has multiple bits set), this
 * will assert-fail.
 */
static inline bool
nir_deref_mode_is_one_of(const nir_deref_instr *deref, nir_variable_mode modes)
{
   /* This is only for "simple" cases so, if modes might interact with this
    * deref then the deref has to have a single mode.
    */
   if (nir_deref_mode_may_be(deref, modes)) {
      assert(util_bitcount(deref->modes) == 1);
      assert(nir_deref_mode_must_be(deref, modes));
   }

   return nir_deref_mode_may_be(deref, modes);
}

/** Returns true if deref's possible modes lie in the given set of modes
 *
 * This returns true if the deref's modes lie in the given set of modes.  If
 * the deref's modes overlap with the specified modes but aren't entirely
 * contained in the specified set of modes, this will assert-fail.  In
 * particular, if this is used in a generic pointers scenario, the specified
 * modes has to contain all or none of the possible generic pointer modes.
 *
 * This is intended mostly for mass-lowering of derefs which might have
 * generic pointers.
 */
static inline bool
nir_deref_mode_is_in_set(const nir_deref_instr *deref, nir_variable_mode modes)
{
   if (nir_deref_mode_may_be(deref, modes))
      assert(nir_deref_mode_must_be(deref, modes));

   return nir_deref_mode_may_be(deref, modes);
}

static inline nir_deref_instr *nir_src_as_deref(nir_src src);

static inline nir_deref_instr *
nir_deref_instr_parent(const nir_deref_instr *instr)
{
   if (instr->deref_type == nir_deref_type_var)
      return NULL;
   else
      return nir_src_as_deref(instr->parent);
}

static inline nir_variable *
nir_deref_instr_get_variable(const nir_deref_instr *instr)
{
   while (instr->deref_type != nir_deref_type_var) {
      if (instr->deref_type == nir_deref_type_cast)
         return NULL;

      instr = nir_deref_instr_parent(instr);
   }

   return instr->var;
}

bool nir_deref_instr_has_indirect(nir_deref_instr *instr);
bool nir_deref_instr_is_known_out_of_bounds(nir_deref_instr *instr);

typedef enum {
   nir_deref_instr_has_complex_use_allow_memcpy_src = (1 << 0),
   nir_deref_instr_has_complex_use_allow_memcpy_dst = (1 << 1),
} nir_deref_instr_has_complex_use_options;

bool nir_deref_instr_has_complex_use(nir_deref_instr *instr,
                                     nir_deref_instr_has_complex_use_options opts);

bool nir_deref_instr_remove_if_unused(nir_deref_instr *instr);

unsigned nir_deref_instr_array_stride(nir_deref_instr *instr);

typedef struct {
   nir_instr instr;

   struct nir_function *callee;

   unsigned num_params;
   nir_src params[];
} nir_call_instr;

#include "nir_intrinsics.h"

#define NIR_INTRINSIC_MAX_CONST_INDEX 7

/** Represents an intrinsic
 *
 * An intrinsic is an instruction type for handling things that are
 * more-or-less regular operations but don't just consume and produce SSA
 * values like ALU operations do.  Intrinsics are not for things that have
 * special semantic meaning such as phi nodes and parallel copies.
 * Examples of intrinsics include variable load/store operations, system
 * value loads, and the like.  Even though texturing more-or-less falls
 * under this category, texturing is its own instruction type because
 * trying to represent texturing with intrinsics would lead to a
 * combinatorial explosion of intrinsic opcodes.
 *
 * By having a single instruction type for handling a lot of different
 * cases, optimization passes can look for intrinsics and, for the most
 * part, completely ignore them.  Each intrinsic type also has a few
 * possible flags that govern whether or not they can be reordered or
 * eliminated.  That way passes like dead code elimination can still work
 * on intrisics without understanding the meaning of each.
 *
 * Each intrinsic has some number of constant indices, some number of
 * variables, and some number of sources.  What these sources, variables,
 * and indices mean depends on the intrinsic and is documented with the
 * intrinsic declaration in nir_intrinsics.h.  Intrinsics and texture
 * instructions are the only types of instruction that can operate on
 * variables.
 */
typedef struct {
   nir_instr instr;

   nir_intrinsic_op intrinsic;

   nir_dest dest;

   /** number of components if this is a vectorized intrinsic
    *
    * Similarly to ALU operations, some intrinsics are vectorized.
    * An intrinsic is vectorized if nir_intrinsic_infos.dest_components == 0.
    * For vectorized intrinsics, the num_components field specifies the
    * number of destination components and the number of source components
    * for all sources with nir_intrinsic_infos.src_components[i] == 0.
    */
   uint8_t num_components;

   int const_index[NIR_INTRINSIC_MAX_CONST_INDEX];

   nir_src src[];
} nir_intrinsic_instr;

static inline nir_variable *
nir_intrinsic_get_var(nir_intrinsic_instr *intrin, unsigned i)
{
   return nir_deref_instr_get_variable(nir_src_as_deref(intrin->src[i]));
}

typedef enum {
   /* Memory ordering. */
   NIR_MEMORY_ACQUIRE        = 1 << 0,
   NIR_MEMORY_RELEASE        = 1 << 1,
   NIR_MEMORY_ACQ_REL        = NIR_MEMORY_ACQUIRE | NIR_MEMORY_RELEASE,

   /* Memory visibility operations. */
   NIR_MEMORY_MAKE_AVAILABLE = 1 << 2,
   NIR_MEMORY_MAKE_VISIBLE   = 1 << 3,
} nir_memory_semantics;

typedef enum {
   NIR_SCOPE_NONE,
   NIR_SCOPE_INVOCATION,
   NIR_SCOPE_SUBGROUP,
   NIR_SCOPE_SHADER_CALL,
   NIR_SCOPE_WORKGROUP,
   NIR_SCOPE_QUEUE_FAMILY,
   NIR_SCOPE_DEVICE,
} nir_scope;

/**
 * \name NIR intrinsics semantic flags
 *
 * information about what the compiler can do with the intrinsics.
 *
 * \sa nir_intrinsic_info::flags
 */
typedef enum {
   /**
    * whether the intrinsic can be safely eliminated if none of its output
    * value is not being used.
    */
   NIR_INTRINSIC_CAN_ELIMINATE = (1 << 0),

   /**
    * Whether the intrinsic can be reordered with respect to any other
    * intrinsic, i.e. whether the only reordering dependencies of the
    * intrinsic are due to the register reads/writes.
    */
   NIR_INTRINSIC_CAN_REORDER = (1 << 1),
} nir_intrinsic_semantic_flag;

/**
 * Maximum valid value for a nir align_mul value (in intrinsics or derefs).
 *
 * Offsets can be signed, so this is the largest power of two in int32_t.
 */
#define NIR_ALIGN_MUL_MAX 0x40000000

typedef struct nir_io_semantics {
   unsigned location:7; /* gl_vert_attrib, gl_varying_slot, or gl_frag_result */
   unsigned num_slots:6;  /* max 32, may be pessimistic with const indexing */
   unsigned dual_source_blend_index:1;
   unsigned fb_fetch_output:1; /* for GL_KHR_blend_equation_advanced */
   unsigned gs_streams:8; /* xxyyzzww: 2-bit stream index for each component */
   unsigned medium_precision:1; /* GLSL mediump qualifier */
   unsigned per_view:1;
   unsigned high_16bits:1; /* whether accessing low or high half of the slot */
   unsigned invariant:1; /* The variable has the invariant flag set */
   /* CLIP_DISTn, LAYER, VIEWPORT, and TESS_LEVEL_* have up to 3 uses:
    * - an output consumed by the next stage
    * - a system value output affecting fixed-func hardware, e.g. the clipper
    * - a transform feedback output written to memory
    * The following fields disable the first two. Transform feedback is disabled
    * by transform feedback info.
    */
   unsigned no_varying:1; /* whether this output isn't consumed by the next stage */
   unsigned no_sysval_output:1; /* whether this system value output has no
                                   effect due to current pipeline states */
   unsigned _pad:3;
} nir_io_semantics;

/* Transform feedback info for 2 outputs. nir_intrinsic_store_output contains
 * this structure twice to support up to 4 outputs. The structure is limited
 * to 32 bits because it's stored in nir_intrinsic_instr::const_index[].
 */
typedef struct nir_io_xfb {
   struct {
      /* start_component is equal to the index of out[]; add 2 for io_xfb2 */
      /* start_component is not relative to nir_intrinsic_component */
      /* get the stream index from nir_io_semantics */
      uint8_t num_components:4;  /* max 4; if this is 0, xfb is disabled */
      uint8_t buffer:4;  /* buffer index, max 3 */
      uint8_t offset;    /* transform feedback buffer offset in dwords,
                            max (1K - 4) bytes */
   } out[2];
} nir_io_xfb;

unsigned
nir_instr_xfb_write_mask(nir_intrinsic_instr *instr);

#define NIR_INTRINSIC_MAX_INPUTS 11

typedef struct {
   const char *name;

   uint8_t num_srcs; /** < number of register/SSA inputs */

   /** number of components of each input register
    *
    * If this value is 0, the number of components is given by the
    * num_components field of nir_intrinsic_instr.  If this value is -1, the
    * intrinsic consumes however many components are provided and it is not
    * validated at all.
    */
   int8_t src_components[NIR_INTRINSIC_MAX_INPUTS];

   bool has_dest;

   /** number of components of the output register
    *
    * If this value is 0, the number of components is given by the
    * num_components field of nir_intrinsic_instr.
    */
   uint8_t dest_components;

   /** bitfield of legal bit sizes */
   uint8_t dest_bit_sizes;

   /** source which the destination bit size must match
    *
    * Some intrinsics, such as subgroup intrinsics, are data manipulation
    * intrinsics and they have similar bit-size rules to ALU ops. This enables
    * validation to validate a bit more and enables auto-generated builder code
    * to properly determine destination bit sizes automatically.
    */
   int8_t bit_size_src;

   /** the number of constant indices used by the intrinsic */
   uint8_t num_indices;

   /** list of indices */
   uint8_t indices[NIR_INTRINSIC_MAX_CONST_INDEX];

   /** indicates the usage of intr->const_index[n] */
   uint8_t index_map[NIR_INTRINSIC_NUM_INDEX_FLAGS];

   /** semantic flags for calls to this intrinsic */
   nir_intrinsic_semantic_flag flags;
} nir_intrinsic_info;

extern const nir_intrinsic_info nir_intrinsic_infos[nir_num_intrinsics];

unsigned
nir_intrinsic_src_components(const nir_intrinsic_instr *intr, unsigned srcn);

unsigned
nir_intrinsic_dest_components(nir_intrinsic_instr *intr);

/**
 * Helper to copy const_index[] from src to dst, without assuming they
 * match in order.
 */
void nir_intrinsic_copy_const_indices(nir_intrinsic_instr *dst, nir_intrinsic_instr *src);

#include "nir_intrinsics_indices.h"

static inline void
nir_intrinsic_set_align(nir_intrinsic_instr *intrin,
                        unsigned align_mul, unsigned align_offset)
{
   assert(util_is_power_of_two_nonzero(align_mul));
   assert(align_offset < align_mul);
   nir_intrinsic_set_align_mul(intrin, align_mul);
   nir_intrinsic_set_align_offset(intrin, align_offset);
}

/** Returns a simple alignment for a load/store intrinsic offset
 *
 * Instead of the full mul+offset alignment scheme provided by the ALIGN_MUL
 * and ALIGN_OFFSET parameters, this helper takes both into account and
 * provides a single simple alignment parameter.  The offset X is guaranteed
 * to satisfy X % align == 0.
 */
static inline unsigned
nir_intrinsic_align(const nir_intrinsic_instr *intrin)
{
   const unsigned align_mul = nir_intrinsic_align_mul(intrin);
   const unsigned align_offset = nir_intrinsic_align_offset(intrin);
   assert(align_offset < align_mul);
   return align_offset ? 1 << (ffs(align_offset) - 1) : align_mul;
}

static inline bool
nir_intrinsic_has_align(const nir_intrinsic_instr *intrin)
{
   return nir_intrinsic_has_align_mul(intrin) &&
          nir_intrinsic_has_align_offset(intrin);
}

unsigned
nir_image_intrinsic_coord_components(const nir_intrinsic_instr *instr);

/* Converts a image_deref_* intrinsic into a image_* one */
void nir_rewrite_image_intrinsic(nir_intrinsic_instr *instr,
                                 nir_ssa_def *handle, bool bindless);

/* Determine if an intrinsic can be arbitrarily reordered and eliminated. */
static inline bool
nir_intrinsic_can_reorder(nir_intrinsic_instr *instr)
{
   if (nir_intrinsic_has_access(instr) &&
       nir_intrinsic_access(instr) & ACCESS_VOLATILE)
      return false;

   if (instr->intrinsic == nir_intrinsic_load_deref) {
      nir_deref_instr *deref = nir_src_as_deref(instr->src[0]);
      return nir_deref_mode_is_in_set(deref, nir_var_read_only_modes) ||
             (nir_intrinsic_access(instr) & ACCESS_CAN_REORDER);
   } else if (instr->intrinsic == nir_intrinsic_load_ssbo ||
              instr->intrinsic == nir_intrinsic_bindless_image_load ||
              instr->intrinsic == nir_intrinsic_image_deref_load ||
              instr->intrinsic == nir_intrinsic_image_load) {
      return nir_intrinsic_access(instr) & ACCESS_CAN_REORDER;
   } else {
      const nir_intrinsic_info *info =
         &nir_intrinsic_infos[instr->intrinsic];
      return (info->flags & NIR_INTRINSIC_CAN_ELIMINATE) &&
             (info->flags & NIR_INTRINSIC_CAN_REORDER);
   }
}

bool nir_intrinsic_writes_external_memory(const nir_intrinsic_instr *instr);

static inline bool
nir_intrinsic_is_ray_query(nir_intrinsic_op intrinsic)
{
   switch (intrinsic) {
   case nir_intrinsic_rq_confirm_intersection:
   case nir_intrinsic_rq_generate_intersection:
   case nir_intrinsic_rq_initialize:
   case nir_intrinsic_rq_load:
   case nir_intrinsic_rq_proceed:
   case nir_intrinsic_rq_terminate:
      return true;
   default:
      return false;
   }
}

/** Texture instruction source type */
typedef enum {
   /** Texture coordinate
    *
    * Must have nir_tex_instr::coord_components components.
    */
   nir_tex_src_coord,

   /** Projector
    *
    * The texture coordinate (except for the array component, if any) is
    * divided by this value before LOD computation and sampling.
    *
    * Must be a float scalar.
    */
   nir_tex_src_projector,

   /** Shadow comparator
    *
    * For shadow sampling, the fetched texel values are compared against the
    * shadow comparator using the compare op specified by the sampler object
    * and converted to 1.0 if the comparison succeeds and 0.0 if it fails.
    * Interpolation happens after this conversion so the actual result may be
    * anywhere in the range [0.0, 1.0].
    *
    * Only valid if nir_tex_instr::is_shadow and must be a float scalar.
    */
   nir_tex_src_comparator,

   /** Coordinate offset
    *
    * An integer value that is added to the texel address before sampling.
    * This is only allowed with operations that take an explicit LOD as it is
    * applied in integer texel space after LOD selection and not normalized
    * coordinate space.
    */
   nir_tex_src_offset,

   /** LOD bias
    *
    * This value is added to the computed LOD before mip-mapping.
    */
   nir_tex_src_bias,

   /** Explicit LOD */
   nir_tex_src_lod,

   /** Min LOD
    *
    * The computed LOD is clamped to be at least as large as min_lod before
    * mip-mapping.
    */
   nir_tex_src_min_lod,

   /** MSAA sample index */
   nir_tex_src_ms_index,

   /** Intel-specific MSAA compression data */
   nir_tex_src_ms_mcs_intel,

   /** Explicit horizontal (X-major) coordinate derivative */
   nir_tex_src_ddx,

   /** Explicit vertical (Y-major) coordinate derivative */
   nir_tex_src_ddy,

   /** Texture variable dereference */
   nir_tex_src_texture_deref,

   /** Sampler variable dereference */
   nir_tex_src_sampler_deref,

   /** Texture index offset
    *
    * This is added to nir_tex_instr::texture_index.  Unless
    * nir_tex_instr::texture_non_uniform is set, this is guaranteed to be
    * dynamically uniform.
    */
   nir_tex_src_texture_offset,

   /** Dynamically uniform sampler index offset
    *
    * This is added to nir_tex_instr::sampler_index.  Unless
    * nir_tex_instr::sampler_non_uniform is set, this is guaranteed to be
    * dynamically uniform.  This should not be present until GLSL ES 3.20, GLSL
    * 4.00, or ARB_gpu_shader5, because in ES 3.10 and GL 3.30 samplers said
    * "When aggregated into arrays within a shader, samplers can only be indexed
    * with a constant integral expression."
    */
   nir_tex_src_sampler_offset,

   /** Bindless texture handle
    *
    * This is, unfortunately, a bit overloaded at the moment.  There are
    * generally two types of bindless handles:
    *
    *  1. For GL_ARB_bindless bindless handles. These are part of the
    *     GL/Gallium-level API and are always a 64-bit integer.
    *
    *  2. HW-specific handles.  GL_ARB_bindless handles may be lowered to
    *     these.  Also, these are used by many Vulkan drivers to implement
    *     descriptor sets, especially for UPDATE_AFTER_BIND descriptors.
    *     The details of hardware handles (bit size, format, etc.) is
    *     HW-specific.
    *
    * Because of this overloading and the resulting ambiguity, we currently
    * don't validate anything for these.
    */
   nir_tex_src_texture_handle,

   /** Bindless sampler handle
    *
    * See nir_tex_src_texture_handle,
    */
   nir_tex_src_sampler_handle,

   /** Plane index for multi-plane YCbCr textures */
   nir_tex_src_plane,

   /**
    * Backend-specific vec4 tex src argument.
    *
    * Can be used to have NIR optimization (copy propagation, lower_vec_to_movs)
    * apply to the packing of the tex srcs.  This lowering must only happen
    * after nir_lower_tex().
    *
    * The nir_tex_instr_src_type() of this argument is float, so no lowering
    * will happen if nir_lower_int_to_float is used.
    */
   nir_tex_src_backend1,

   /** Second backend-specific vec4 tex src argument, see nir_tex_src_backend1. */
   nir_tex_src_backend2,

   nir_num_tex_src_types
} nir_tex_src_type;

/** A texture instruction source */
typedef struct {
   /** Base source */
   nir_src src;

   /** Type of this source */
   nir_tex_src_type src_type;
} nir_tex_src;

/** Texture instruction opcode */
typedef enum {
   nir_texop_tex,                /**< Regular texture look-up */
   nir_texop_txb,                /**< Texture look-up with LOD bias */
   nir_texop_txl,                /**< Texture look-up with explicit LOD */
   nir_texop_txd,                /**< Texture look-up with partial derivatives */
   nir_texop_txf,                /**< Texel fetch with explicit LOD */
   nir_texop_txf_ms,             /**< Multisample texture fetch */
   nir_texop_txf_ms_fb,          /**< Multisample texture fetch from framebuffer */
   nir_texop_txf_ms_mcs_intel,   /**< Multisample compression value fetch */
   nir_texop_txs,                /**< Texture size */
   nir_texop_lod,                /**< Texture lod query */
   nir_texop_tg4,                /**< Texture gather */
   nir_texop_query_levels,       /**< Texture levels query */
   nir_texop_texture_samples,    /**< Texture samples query */
   nir_texop_samples_identical,  /**< Query whether all samples are definitely
                                  * identical.
                                  */
   nir_texop_tex_prefetch,       /**< Regular texture look-up, eligible for pre-dispatch */
   nir_texop_fragment_fetch_amd,      /**< Multisample fragment color texture fetch */
   nir_texop_fragment_mask_fetch_amd, /**< Multisample fragment mask texture fetch */
   nir_texop_descriptor_amd,     /**< Returns a buffer or image descriptor. */
   nir_texop_sampler_descriptor_amd, /**< Returns a sampler descriptor. */
} nir_texop;

/** Represents a texture instruction */
typedef struct {
   /** Base instruction */
   nir_instr instr;

   /** Dimensionality of the texture operation
    *
    * This will typically match the dimensionality of the texture deref type
    * if a nir_tex_src_texture_deref is present.  However, it may not if
    * texture lowering has occurred.
    */
   enum glsl_sampler_dim sampler_dim;

   /** ALU type of the destination
    *
    * This is the canonical sampled type for this texture operation and may
    * not exactly match the sampled type of the deref type when a
    * nir_tex_src_texture_deref is present.  For OpenCL, the sampled type of
    * the texture deref will be GLSL_TYPE_VOID and this is allowed to be
    * anything.  With SPIR-V, the signedness of integer types is allowed to
    * differ.  For all APIs, the bit size may differ if the driver has done
    * any sort of mediump or similar lowering since texture types always have
    * 32-bit sampled types.
    */
   nir_alu_type dest_type;

   /** Texture opcode */
   nir_texop op;

   /** Destination */
   nir_dest dest;

   /** Array of sources
    *
    * This array has nir_tex_instr::num_srcs elements
    */
   nir_tex_src *src;

   /** Number of sources */
   unsigned num_srcs;

   /** Number of components in the coordinate, if any */
   unsigned coord_components;

   /** True if the texture instruction acts on an array texture */
   bool is_array;

   /** True if the texture instruction performs a shadow comparison
    *
    * If this is true, the texture instruction must have a
    * nir_tex_src_comparator.
    */
   bool is_shadow;

   /**
    * If is_shadow is true, whether this is the old-style shadow that outputs
    * 4 components or the new-style shadow that outputs 1 component.
    */
   bool is_new_style_shadow;

   /**
    * True if this texture instruction should return a sparse residency code.
    * The code is in the last component of the result.
    */
   bool is_sparse;

   /** nir_texop_tg4 component selector
    *
    * This determines which RGBA component is gathered.
    */
   unsigned component : 2;

   /** Validation needs to know this for gradient component count */
   unsigned array_is_lowered_cube : 1;

   /** Gather offsets */
   int8_t tg4_offsets[4][2];

   /** True if the texture index or handle is not dynamically uniform */
   bool texture_non_uniform;

   /** True if the sampler index or handle is not dynamically uniform.
    *
    * This may be set when VK_EXT_descriptor_indexing is supported and the
    * appropriate capability is enabled.
    *
    * This should always be false in GLSL (GLSL ES 3.20 says "When aggregated
    * into arrays within a shader, opaque types can only be indexed with a
    * dynamically uniform integral expression", and GLSL 4.60 says "When
    * aggregated into arrays within a shader, [texture, sampler, and
    * samplerShadow] types can only be indexed with a dynamically uniform
    * expression, or texture lookup will result in undefined values.").
    */
   bool sampler_non_uniform;

   /** The texture index
    *
    * If this texture instruction has a nir_tex_src_texture_offset source,
    * then the texture index is given by texture_index + texture_offset.
    */
   unsigned texture_index;

   /** The sampler index
    *
    * The following operations do not require a sampler and, as such, this
    * field should be ignored:
    *    - nir_texop_txf
    *    - nir_texop_txf_ms
    *    - nir_texop_txs
    *    - nir_texop_query_levels
    *    - nir_texop_texture_samples
    *    - nir_texop_samples_identical
    *
    * If this texture instruction has a nir_tex_src_sampler_offset source,
    * then the sampler index is given by sampler_index + sampler_offset.
    */
   unsigned sampler_index;
} nir_tex_instr;

/**
 * Returns true if the texture operation requires a sampler as a general rule
 *
 * Note that the specific hw/driver backend could require to a sampler
 * object/configuration packet in any case, for some other reason.
 *
 * @see nir_tex_instr::sampler_index.
 */
bool nir_tex_instr_need_sampler(const nir_tex_instr *instr);

/** Returns the number of components returned by this nir_tex_instr
 *
 * Useful for code building texture instructions when you don't want to think
 * about how many components a particular texture op returns.  This does not
 * include the sparse residency code.
 */
unsigned
nir_tex_instr_result_size(const nir_tex_instr *instr);

/**
 * Returns the destination size of this nir_tex_instr including the sparse
 * residency code, if any.
 */
static inline unsigned
nir_tex_instr_dest_size(const nir_tex_instr *instr)
{
   /* One more component is needed for the residency code. */
   return nir_tex_instr_result_size(instr) + instr->is_sparse;
}

/**
 * Returns true if this texture operation queries something about the texture
 * rather than actually sampling it.
 */
bool
nir_tex_instr_is_query(const nir_tex_instr *instr);

/** Returns true if this texture instruction does implicit derivatives
 *
 * This is important as there are extra control-flow rules around derivatives
 * and texture instructions which perform them implicitly.
 */
bool
nir_tex_instr_has_implicit_derivative(const nir_tex_instr *instr);

/** Returns the ALU type of the given texture instruction source */
nir_alu_type
nir_tex_instr_src_type(const nir_tex_instr *instr, unsigned src);

/**
 * Returns the number of components required by the given texture instruction
 * source
 */
unsigned
nir_tex_instr_src_size(const nir_tex_instr *instr, unsigned src);

/**
 * Returns the index of the texture instruction source with the given
 * nir_tex_src_type or -1 if no such source exists.
 */
static inline int
nir_tex_instr_src_index(const nir_tex_instr *instr, nir_tex_src_type type)
{
   for (unsigned i = 0; i < instr->num_srcs; i++)
      if (instr->src[i].src_type == type)
         return (int) i;

   return -1;
}

/** Adds a source to a texture instruction */
void nir_tex_instr_add_src(nir_tex_instr *tex,
                           nir_tex_src_type src_type,
                           nir_src src);

/** Removes a source from a texture instruction */
void nir_tex_instr_remove_src(nir_tex_instr *tex, unsigned src_idx);

bool nir_tex_instr_has_explicit_tg4_offsets(nir_tex_instr *tex);

typedef struct {
   nir_instr instr;

   nir_ssa_def def;

   nir_const_value value[];
} nir_load_const_instr;

typedef enum {
   /** Return from a function
    *
    * This instruction is a classic function return.  It jumps to
    * nir_function_impl::end_block.  No return value is provided in this
    * instruction.  Instead, the function is expected to write any return
    * data to a deref passed in from the caller.
    */
   nir_jump_return,

   /** Immediately exit the current shader
    *
    * This instruction is roughly the equivalent of C's "exit()" in that it
    * immediately terminates the current shader invocation.  From a CFG
    * perspective, it looks like a jump to nir_function_impl::end_block but
    * it actually jumps to the end block of the shader entrypoint.  A halt
    * instruction in the shader entrypoint itself is semantically identical
    * to a return.
    *
    * For shaders with built-in I/O, any outputs written prior to a halt
    * instruction remain written and any outputs not written prior to the
    * halt have undefined values.  It does NOT cause an implicit discard of
    * written results.  If one wants discard results in a fragment shader,
    * for instance, a discard or demote intrinsic is required.
    */
   nir_jump_halt,

   /** Break out of the inner-most loop
    *
    * This has the same semantics as C's "break" statement.
    */
   nir_jump_break,

   /** Jump back to the top of the inner-most loop
    *
    * This has the same semantics as C's "continue" statement assuming that a
    * NIR loop is implemented as "while (1) { body }".
    */
   nir_jump_continue,

   /** Jumps for unstructured CFG.
    *
    * As within an unstructured CFG we can't rely on block ordering we need to
    * place explicit jumps at the end of every block.
    */
   nir_jump_goto,
   nir_jump_goto_if,
} nir_jump_type;

typedef struct {
   nir_instr instr;
   nir_jump_type type;
   nir_src condition;
   struct nir_block *target;
   struct nir_block *else_target;
} nir_jump_instr;

/* creates a new SSA variable in an undefined state */

typedef struct {
   nir_instr instr;
   nir_ssa_def def;
} nir_ssa_undef_instr;

typedef struct {
   struct exec_node node;

   /* The predecessor block corresponding to this source */
   struct nir_block *pred;

   nir_src src;
} nir_phi_src;

#define nir_foreach_phi_src(phi_src, phi) \
   foreach_list_typed(nir_phi_src, phi_src, node, &(phi)->srcs)
#define nir_foreach_phi_src_safe(phi_src, phi) \
   foreach_list_typed_safe(nir_phi_src, phi_src, node, &(phi)->srcs)

typedef struct {
   nir_instr instr;

   struct exec_list srcs; /** < list of nir_phi_src */

   nir_dest dest;
} nir_phi_instr;

static inline nir_phi_src *
nir_phi_get_src_from_block(nir_phi_instr *phi, struct nir_block *block)
{
   nir_foreach_phi_src(src, phi) {
      if (src->pred == block)
         return src;
   }

   assert(!"Block is not a predecessor of phi.");
   return NULL;
}

typedef struct {
   struct exec_node node;
   nir_src src;
   nir_dest dest;
} nir_parallel_copy_entry;

#define nir_foreach_parallel_copy_entry(entry, pcopy) \
   foreach_list_typed(nir_parallel_copy_entry, entry, node, &(pcopy)->entries)

typedef struct {
   nir_instr instr;

   /* A list of nir_parallel_copy_entrys.  The sources of all of the
    * entries are copied to the corresponding destinations "in parallel".
    * In other words, if we have two entries: a -> b and b -> a, the values
    * get swapped.
    */
   struct exec_list entries;
} nir_parallel_copy_instr;

NIR_DEFINE_CAST(nir_instr_as_alu, nir_instr, nir_alu_instr, instr,
                type, nir_instr_type_alu)
NIR_DEFINE_CAST(nir_instr_as_deref, nir_instr, nir_deref_instr, instr,
                type, nir_instr_type_deref)
NIR_DEFINE_CAST(nir_instr_as_call, nir_instr, nir_call_instr, instr,
                type, nir_instr_type_call)
NIR_DEFINE_CAST(nir_instr_as_jump, nir_instr, nir_jump_instr, instr,
                type, nir_instr_type_jump)
NIR_DEFINE_CAST(nir_instr_as_tex, nir_instr, nir_tex_instr, instr,
                type, nir_instr_type_tex)
NIR_DEFINE_CAST(nir_instr_as_intrinsic, nir_instr, nir_intrinsic_instr, instr,
                type, nir_instr_type_intrinsic)
NIR_DEFINE_CAST(nir_instr_as_load_const, nir_instr, nir_load_const_instr, instr,
                type, nir_instr_type_load_const)
NIR_DEFINE_CAST(nir_instr_as_ssa_undef, nir_instr, nir_ssa_undef_instr, instr,
                type, nir_instr_type_ssa_undef)
NIR_DEFINE_CAST(nir_instr_as_phi, nir_instr, nir_phi_instr, instr,
                type, nir_instr_type_phi)
NIR_DEFINE_CAST(nir_instr_as_parallel_copy, nir_instr,
                nir_parallel_copy_instr, instr,
                type, nir_instr_type_parallel_copy)


#define NIR_DEFINE_SRC_AS_CONST(type, suffix)               \
static inline type                                          \
nir_src_comp_as_##suffix(nir_src src, unsigned comp)        \
{                                                           \
   assert(nir_src_is_const(src));                           \
   nir_load_const_instr *load =                             \
      nir_instr_as_load_const(src.ssa->parent_instr);       \
   assert(comp < load->def.num_components);                 \
   return nir_const_value_as_##suffix(load->value[comp],    \
                                      load->def.bit_size);  \
}                                                           \
                                                            \
static inline type                                          \
nir_src_as_##suffix(nir_src src)                            \
{                                                           \
   assert(nir_src_num_components(src) == 1);                \
   return nir_src_comp_as_##suffix(src, 0);                 \
}

NIR_DEFINE_SRC_AS_CONST(int64_t,    int)
NIR_DEFINE_SRC_AS_CONST(uint64_t,   uint)
NIR_DEFINE_SRC_AS_CONST(bool,       bool)
NIR_DEFINE_SRC_AS_CONST(double,     float)

#undef NIR_DEFINE_SRC_AS_CONST


typedef struct {
   nir_ssa_def *def;
   unsigned comp;
} nir_ssa_scalar;

static inline bool
nir_ssa_scalar_is_const(nir_ssa_scalar s)
{
   return s.def->parent_instr->type == nir_instr_type_load_const;
}

static inline bool
nir_ssa_scalar_is_undef(nir_ssa_scalar s)
{
   return s.def->parent_instr->type == nir_instr_type_ssa_undef;
}

static inline nir_const_value
nir_ssa_scalar_as_const_value(nir_ssa_scalar s)
{
   assert(s.comp < s.def->num_components);
   nir_load_const_instr *load = nir_instr_as_load_const(s.def->parent_instr);
   return load->value[s.comp];
}

#define NIR_DEFINE_SCALAR_AS_CONST(type, suffix)                     \
static inline type                                                   \
nir_ssa_scalar_as_##suffix(nir_ssa_scalar s)                         \
{                                                                    \
   return nir_const_value_as_##suffix(                               \
      nir_ssa_scalar_as_const_value(s), s.def->bit_size);            \
}

NIR_DEFINE_SCALAR_AS_CONST(int64_t,    int)
NIR_DEFINE_SCALAR_AS_CONST(uint64_t,   uint)
NIR_DEFINE_SCALAR_AS_CONST(bool,       bool)
NIR_DEFINE_SCALAR_AS_CONST(double,     float)

#undef NIR_DEFINE_SCALAR_AS_CONST

static inline bool
nir_ssa_scalar_is_alu(nir_ssa_scalar s)
{
   return s.def->parent_instr->type == nir_instr_type_alu;
}

static inline nir_op
nir_ssa_scalar_alu_op(nir_ssa_scalar s)
{
   return nir_instr_as_alu(s.def->parent_instr)->op;
}

static inline nir_ssa_scalar
nir_ssa_scalar_chase_alu_src(nir_ssa_scalar s, unsigned alu_src_idx)
{
   nir_ssa_scalar out = { NULL, 0 };

   nir_alu_instr *alu = nir_instr_as_alu(s.def->parent_instr);
   assert(alu_src_idx < nir_op_infos[alu->op].num_inputs);

   /* Our component must be written */
   assert(s.comp < s.def->num_components);
   assert(alu->dest.write_mask & (1u << s.comp));

   assert(alu->src[alu_src_idx].src.is_ssa);
   out.def = alu->src[alu_src_idx].src.ssa;

   if (nir_op_infos[alu->op].input_sizes[alu_src_idx] == 0) {
      /* The ALU src is unsized so the source component follows the
       * destination component.
       */
      out.comp = alu->src[alu_src_idx].swizzle[s.comp];
   } else {
      /* This is a sized source so all source components work together to
       * produce all the destination components.  Since we need to return a
       * scalar, this only works if the source is a scalar.
       */
      assert(nir_op_infos[alu->op].input_sizes[alu_src_idx] == 1);
      out.comp = alu->src[alu_src_idx].swizzle[0];
   }
   assert(out.comp < out.def->num_components);

   return out;
}

nir_ssa_scalar nir_ssa_scalar_chase_movs(nir_ssa_scalar s);

static inline nir_ssa_scalar
nir_get_ssa_scalar(nir_ssa_def *def, unsigned channel)
{
   nir_ssa_scalar s = { def, channel };
   return s;
}

/** Returns a nir_ssa_scalar where we've followed the bit-exact mov/vec use chain to the original definition */
static inline nir_ssa_scalar
nir_ssa_scalar_resolved(nir_ssa_def *def, unsigned channel)
{
   return nir_ssa_scalar_chase_movs(nir_get_ssa_scalar(def, channel));
}


typedef struct {
   bool success;

   nir_variable *var;
   unsigned desc_set;
   unsigned binding;
   unsigned num_indices;
   nir_src indices[4];
   bool read_first_invocation;
} nir_binding;

nir_binding nir_chase_binding(nir_src rsrc);
nir_variable *nir_get_binding_variable(struct nir_shader *shader, nir_binding binding);


/*
 * Control flow
 *
 * Control flow consists of a tree of control flow nodes, which include
 * if-statements and loops. The leaves of the tree are basic blocks, lists of
 * instructions that always run start-to-finish. Each basic block also keeps
 * track of its successors (blocks which may run immediately after the current
 * block) and predecessors (blocks which could have run immediately before the
 * current block). Each function also has a start block and an end block which
 * all return statements point to (which is always empty). Together, all the
 * blocks with their predecessors and successors make up the control flow
 * graph (CFG) of the function. There are helpers that modify the tree of
 * control flow nodes while modifying the CFG appropriately; these should be
 * used instead of modifying the tree directly.
 */

typedef enum {
   nir_cf_node_block,
   nir_cf_node_if,
   nir_cf_node_loop,
   nir_cf_node_function
} nir_cf_node_type;

typedef struct nir_cf_node {
   struct exec_node node;
   nir_cf_node_type type;
   struct nir_cf_node *parent;
} nir_cf_node;

typedef struct nir_block {
   nir_cf_node cf_node;

   struct exec_list instr_list; /** < list of nir_instr */

   /** generic block index; generated by nir_index_blocks */
   unsigned index;

   /*
    * Each block can only have up to 2 successors, so we put them in a simple
    * array - no need for anything more complicated.
    */
   struct nir_block *successors[2];

   /* Set of nir_block predecessors in the CFG */
   struct set *predecessors;

   /*
    * this node's immediate dominator in the dominance tree - set to NULL for
    * the start block.
    */
   struct nir_block *imm_dom;

   /* This node's children in the dominance tree */
   unsigned num_dom_children;
   struct nir_block **dom_children;

   /* Set of nir_blocks on the dominance frontier of this block */
   struct set *dom_frontier;

   /*
    * These two indices have the property that dom_{pre,post}_index for each
    * child of this block in the dominance tree will always be between
    * dom_pre_index and dom_post_index for this block, which makes testing if
    * a given block is dominated by another block an O(1) operation.
    */
   uint32_t dom_pre_index, dom_post_index;

   /**
    * Value just before the first nir_instr->index in the block, but after
    * end_ip that of any predecessor block.
    */
   uint32_t start_ip;
   /**
    * Value just after the last nir_instr->index in the block, but before the
    * start_ip of any successor block.
    */
   uint32_t end_ip;

   /* SSA def live in and out for this block; used for liveness analysis.
    * Indexed by ssa_def->index
    */
   BITSET_WORD *live_in;
   BITSET_WORD *live_out;
} nir_block;

static inline bool
nir_block_is_reachable(nir_block *b)
{
   /* See also nir_block_dominates */
   return b->dom_post_index != 0;
}

static inline nir_instr *
nir_block_first_instr(nir_block *block)
{
   struct exec_node *head = exec_list_get_head(&block->instr_list);
   return exec_node_data(nir_instr, head, node);
}

static inline nir_instr *
nir_block_last_instr(nir_block *block)
{
   struct exec_node *tail = exec_list_get_tail(&block->instr_list);
   return exec_node_data(nir_instr, tail, node);
}

static inline bool
nir_block_ends_in_jump(nir_block *block)
{
   return !exec_list_is_empty(&block->instr_list) &&
          nir_block_last_instr(block)->type == nir_instr_type_jump;
}

static inline bool
nir_block_ends_in_return_or_halt(nir_block *block)
{
   if (exec_list_is_empty(&block->instr_list))
      return false;

   nir_instr *instr = nir_block_last_instr(block);
   if (instr->type != nir_instr_type_jump)
      return false;

   nir_jump_instr *jump_instr = nir_instr_as_jump(instr);
   return jump_instr->type == nir_jump_return ||
          jump_instr->type == nir_jump_halt;
}

static inline bool
nir_block_ends_in_break(nir_block *block)
{
   if (exec_list_is_empty(&block->instr_list))
      return false;

   nir_instr *instr = nir_block_last_instr(block);
   return instr->type == nir_instr_type_jump &&
      nir_instr_as_jump(instr)->type == nir_jump_break;
}

#define nir_foreach_instr(instr, block) \
   foreach_list_typed(nir_instr, instr, node, &(block)->instr_list)
#define nir_foreach_instr_reverse(instr, block) \
   foreach_list_typed_reverse(nir_instr, instr, node, &(block)->instr_list)
#define nir_foreach_instr_safe(instr, block) \
   foreach_list_typed_safe(nir_instr, instr, node, &(block)->instr_list)
#define nir_foreach_instr_reverse_safe(instr, block) \
   foreach_list_typed_reverse_safe(nir_instr, instr, node, &(block)->instr_list)

static inline nir_phi_instr *
nir_block_last_phi_instr(nir_block *block)
{
   nir_phi_instr *last_phi = NULL;
   nir_foreach_instr(instr, block) {
      if (instr->type == nir_instr_type_phi)
         last_phi = nir_instr_as_phi(instr);
      else
         return last_phi;
   }
   return last_phi;
}

typedef enum {
   nir_selection_control_none = 0x0,

   /**
    * Defined by SPIR-V spec 3.22 "Selection Control".
    * The application prefers to remove control flow.
    */
   nir_selection_control_flatten = 0x1,

   /**
    * Defined by SPIR-V spec 3.22 "Selection Control".
    * The application prefers to keep control flow.
    */
   nir_selection_control_dont_flatten = 0x2,

   /**
    * May be applied by the compiler stack when it knows
    * that a branch is divergent, and:
    * - either both the if and else are always taken
    * - the if or else is empty and the other is always taken
    */
   nir_selection_control_divergent_always_taken = 0x3,
} nir_selection_control;

typedef struct nir_if {
   nir_cf_node cf_node;
   nir_src condition;
   nir_selection_control control;

   struct exec_list then_list; /** < list of nir_cf_node */
   struct exec_list else_list; /** < list of nir_cf_node */
} nir_if;

typedef struct {
   nir_if *nif;

   /** Instruction that generates nif::condition. */
   nir_instr *conditional_instr;

   /** Block within ::nif that has the break instruction. */
   nir_block *break_block;

   /** Last block for the then- or else-path that does not contain the break. */
   nir_block *continue_from_block;

   /** True when ::break_block is in the else-path of ::nif. */
   bool continue_from_then;
   bool induction_rhs;

   /* This is true if the terminators exact trip count is unknown. For
    * example:
    *
    *    for (int i = 0; i < imin(x, 4); i++)
    *       ...
    *
    * Here loop analysis would have set a max_trip_count of 4 however we dont
    * know for sure that this is the exact trip count.
    */
   bool exact_trip_count_unknown;

   struct list_head loop_terminator_link;
} nir_loop_terminator;

typedef struct {
   /* Induction variable. */
   nir_ssa_def *def;

   /* Init statement with only uniform. */
   nir_src *init_src;

   /* Update statement with only uniform. */
   nir_alu_src *update_src;
} nir_loop_induction_variable;

typedef struct {
   /* Estimated cost (in number of instructions) of the loop */
   unsigned instr_cost;

   /* Contains fp64 ops that will be lowered */
   bool has_soft_fp64;

   /* Guessed trip count based on array indexing */
   unsigned guessed_trip_count;

   /* Maximum number of times the loop is run (if known) */
   unsigned max_trip_count;

   /* Do we know the exact number of times the loop will be run */
   bool exact_trip_count_known;

   /* Unroll the loop regardless of its size */
   bool force_unroll;

   /* Does the loop contain complex loop terminators, continues or other
    * complex behaviours? If this is true we can't rely on
    * loop_terminator_list to be complete or accurate.
    */
   bool complex_loop;

   nir_loop_terminator *limiting_terminator;

   /* A list of loop_terminators terminating this loop. */
   struct list_head loop_terminator_list;

   /* array of induction variables for this loop */
   nir_loop_induction_variable *induction_vars;
   unsigned num_induction_vars;
} nir_loop_info;

typedef enum {
   nir_loop_control_none = 0x0,
   nir_loop_control_unroll = 0x1,
   nir_loop_control_dont_unroll = 0x2,
} nir_loop_control;

typedef struct {
   nir_cf_node cf_node;

   struct exec_list body; /** < list of nir_cf_node */

   nir_loop_info *info;
   nir_loop_control control;
   bool partially_unrolled;
   bool divergent;
} nir_loop;

/**
 * Various bits of metadata that can may be created or required by
 * optimization and analysis passes
 */
typedef enum {
   nir_metadata_none = 0x0,

   /** Indicates that nir_block::index values are valid.
    *
    * The start block has index 0 and they increase through a natural walk of
    * the CFG.  nir_function_impl::num_blocks is the number of blocks and
    * every block index is in the range [0, nir_function_impl::num_blocks].
    *
    * A pass can preserve this metadata type if it doesn't touch the CFG.
    */
   nir_metadata_block_index = 0x1,

   /** Indicates that block dominance information is valid
    *
    * This includes:
    *
    *   - nir_block::num_dom_children
    *   - nir_block::dom_children
    *   - nir_block::dom_frontier
    *   - nir_block::dom_pre_index
    *   - nir_block::dom_post_index
    *
    * A pass can preserve this metadata type if it doesn't touch the CFG.
    */
   nir_metadata_dominance = 0x2,

   /** Indicates that SSA def data-flow liveness information is valid
    *
    * This includes:
    *
    *   - nir_block::live_in
    *   - nir_block::live_out
    *
    * A pass can preserve this metadata type if it never adds or removes any
    * SSA defs or uses of SSA defs (most passes shouldn't preserve this
    * metadata type).
    */
   nir_metadata_live_ssa_defs = 0x4,

   /** A dummy metadata value to track when a pass forgot to call
    * nir_metadata_preserve.
    *
    * A pass should always clear this value even if it doesn't make any
    * progress to indicate that it thought about preserving metadata.
    */
   nir_metadata_not_properly_reset = 0x8,

   /** Indicates that loop analysis information is valid.
    *
    * This includes everything pointed to by nir_loop::info.
    *
    * A pass can preserve this metadata type if it is guaranteed to not affect
    * any loop metadata.  However, since loop metadata includes things like
    * loop counts which depend on arithmetic in the loop, this is very hard to
    * determine.  Most passes shouldn't preserve this metadata type.
    */
   nir_metadata_loop_analysis = 0x10,

   /** Indicates that nir_instr::index values are valid.
    *
    * The start instruction has index 0 and they increase through a natural
    * walk of instructions in blocks in the CFG.  The indices my have holes
    * after passes such as DCE.
    *
    * A pass can preserve this metadata type if it never adds or moves any
    * instructions (most passes shouldn't preserve this metadata type), but
    * can preserve it if it only removes instructions.
    */
   nir_metadata_instr_index = 0x20,

   /** All metadata
    *
    * This includes all nir_metadata flags except not_properly_reset.  Passes
    * which do not change the shader in any way should call
    *
    *    nir_metadata_preserve(impl, nir_metadata_all);
    */
   nir_metadata_all = ~nir_metadata_not_properly_reset,
} nir_metadata;
MESA_DEFINE_CPP_ENUM_BITFIELD_OPERATORS(nir_metadata)

typedef struct {
   nir_cf_node cf_node;

   /** pointer to the function of which this is an implementation */
   struct nir_function *function;

   /**
    * For entrypoints, a pointer to a nir_function_impl which runs before
    * it, once per draw or dispatch, communicating via store_preamble and
    * load_preamble intrinsics. If NULL then there is no preamble.
    */
   struct nir_function *preamble;

   struct exec_list body; /** < list of nir_cf_node */

   nir_block *end_block;

   /** list for all local variables in the function */
   struct exec_list locals;

   /** list of local registers in the function */
   struct exec_list registers;

   /** next available local register index */
   unsigned reg_alloc;

   /** next available SSA value index */
   unsigned ssa_alloc;

   /* total number of basic blocks, only valid when block_index_dirty = false */
   unsigned num_blocks;

   /** True if this nir_function_impl uses structured control-flow
    *
    * Structured nir_function_impls have different validation rules.
    */
   bool structured;

   nir_metadata valid_metadata;
} nir_function_impl;

#define nir_foreach_function_temp_variable(var, impl) \
   foreach_list_typed(nir_variable, var, node, &(impl)->locals)

#define nir_foreach_function_temp_variable_safe(var, impl) \
   foreach_list_typed_safe(nir_variable, var, node, &(impl)->locals)

ATTRIBUTE_RETURNS_NONNULL static inline nir_block *
nir_start_block(nir_function_impl *impl)
{
   return (nir_block *) impl->body.head_sentinel.next;
}

ATTRIBUTE_RETURNS_NONNULL static inline nir_block *
nir_impl_last_block(nir_function_impl *impl)
{
   return (nir_block *) impl->body.tail_sentinel.prev;
}

static inline nir_cf_node *
nir_cf_node_next(nir_cf_node *node)
{
   struct exec_node *next = exec_node_get_next(&node->node);
   if (exec_node_is_tail_sentinel(next))
      return NULL;
   else
      return exec_node_data(nir_cf_node, next, node);
}

static inline nir_cf_node *
nir_cf_node_prev(nir_cf_node *node)
{
   struct exec_node *prev = exec_node_get_prev(&node->node);
   if (exec_node_is_head_sentinel(prev))
      return NULL;
   else
      return exec_node_data(nir_cf_node, prev, node);
}

static inline bool
nir_cf_node_is_first(const nir_cf_node *node)
{
   return exec_node_is_head_sentinel(node->node.prev);
}

static inline bool
nir_cf_node_is_last(const nir_cf_node *node)
{
   return exec_node_is_tail_sentinel(node->node.next);
}

NIR_DEFINE_CAST(nir_cf_node_as_block, nir_cf_node, nir_block, cf_node,
                type, nir_cf_node_block)
NIR_DEFINE_CAST(nir_cf_node_as_if, nir_cf_node, nir_if, cf_node,
                type, nir_cf_node_if)
NIR_DEFINE_CAST(nir_cf_node_as_loop, nir_cf_node, nir_loop, cf_node,
                type, nir_cf_node_loop)
NIR_DEFINE_CAST(nir_cf_node_as_function, nir_cf_node,
                nir_function_impl, cf_node, type, nir_cf_node_function)

static inline nir_block *
nir_if_first_then_block(nir_if *if_stmt)
{
   struct exec_node *head = exec_list_get_head(&if_stmt->then_list);
   return nir_cf_node_as_block(exec_node_data(nir_cf_node, head, node));
}

static inline nir_block *
nir_if_last_then_block(nir_if *if_stmt)
{
   struct exec_node *tail = exec_list_get_tail(&if_stmt->then_list);
   return nir_cf_node_as_block(exec_node_data(nir_cf_node, tail, node));
}

static inline nir_block *
nir_if_first_else_block(nir_if *if_stmt)
{
   struct exec_node *head = exec_list_get_head(&if_stmt->else_list);
   return nir_cf_node_as_block(exec_node_data(nir_cf_node, head, node));
}

static inline nir_block *
nir_if_last_else_block(nir_if *if_stmt)
{
   struct exec_node *tail = exec_list_get_tail(&if_stmt->else_list);
   return nir_cf_node_as_block(exec_node_data(nir_cf_node, tail, node));
}

static inline nir_block *
nir_loop_first_block(nir_loop *loop)
{
   struct exec_node *head = exec_list_get_head(&loop->body);
   return nir_cf_node_as_block(exec_node_data(nir_cf_node, head, node));
}

static inline nir_block *
nir_loop_last_block(nir_loop *loop)
{
   struct exec_node *tail = exec_list_get_tail(&loop->body);
   return nir_cf_node_as_block(exec_node_data(nir_cf_node, tail, node));
}

/**
 * Return true if this list of cf_nodes contains a single empty block.
 */
static inline bool
nir_cf_list_is_empty_block(struct exec_list *cf_list)
{
   if (exec_list_is_singular(cf_list)) {
      struct exec_node *head = exec_list_get_head(cf_list);
      nir_block *block =
         nir_cf_node_as_block(exec_node_data(nir_cf_node, head, node));
      return exec_list_is_empty(&block->instr_list);
   }
   return false;
}

typedef struct {
   uint8_t num_components;
   uint8_t bit_size;
} nir_parameter;

typedef struct nir_function {
   struct exec_node node;

   const char *name;
   struct nir_shader *shader;

   unsigned num_params;
   nir_parameter *params;

   /** The implementation of this function.
    *
    * If the function is only declared and not implemented, this is NULL.
    */
   nir_function_impl *impl;

   bool is_entrypoint;
   bool is_preamble;
} nir_function;

typedef enum {
   nir_lower_imul64 = (1 << 0),
   nir_lower_isign64 = (1 << 1),
   /** Lower all int64 modulus and division opcodes */
   nir_lower_divmod64 = (1 << 2),
   /** Lower all 64-bit umul_high and imul_high opcodes */
   nir_lower_imul_high64 = (1 << 3),
   nir_lower_mov64 = (1 << 4),
   nir_lower_icmp64 = (1 << 5),
   nir_lower_iadd64 = (1 << 6),
   nir_lower_iabs64 = (1 << 7),
   nir_lower_ineg64 = (1 << 8),
   nir_lower_logic64 = (1 << 9),
   nir_lower_minmax64 = (1 << 10),
   nir_lower_shift64 = (1 << 11),
   nir_lower_imul_2x32_64 = (1 << 12),
   nir_lower_extract64 = (1 << 13),
   nir_lower_ufind_msb64 = (1 << 14),
   nir_lower_bit_count64 = (1 << 15),
   nir_lower_subgroup_shuffle64 = (1 << 16),
   nir_lower_scan_reduce_bitwise64 = (1 << 17),
   nir_lower_scan_reduce_iadd64 = (1 << 18),
   nir_lower_vote_ieq64 = (1 << 19),
   nir_lower_usub_sat64 = (1 << 20),
   nir_lower_iadd_sat64 = (1 << 21),
} nir_lower_int64_options;

typedef enum {
   nir_lower_drcp = (1 << 0),
   nir_lower_dsqrt = (1 << 1),
   nir_lower_drsq = (1 << 2),
   nir_lower_dtrunc = (1 << 3),
   nir_lower_dfloor = (1 << 4),
   nir_lower_dceil = (1 << 5),
   nir_lower_dfract = (1 << 6),
   nir_lower_dround_even = (1 << 7),
   nir_lower_dmod = (1 << 8),
   nir_lower_dsub = (1 << 9),
   nir_lower_ddiv = (1 << 10),
   nir_lower_fp64_full_software = (1 << 11),
} nir_lower_doubles_options;

typedef enum {
   nir_divergence_single_prim_per_subgroup = (1 << 0),
   nir_divergence_single_patch_per_tcs_subgroup = (1 << 1),
   nir_divergence_single_patch_per_tes_subgroup = (1 << 2),
   nir_divergence_view_index_uniform = (1 << 3),
   nir_divergence_single_frag_shading_rate_per_subgroup = (1 << 4),
   nir_divergence_multiple_workgroup_per_compute_subgroup = (1 << 5),
   nir_divergence_shader_record_ptr_uniform = (1 << 6),
} nir_divergence_options;

typedef enum {
   nir_pack_varying_interp_mode_none          = (1 << 0),
   nir_pack_varying_interp_mode_smooth        = (1 << 1),
   nir_pack_varying_interp_mode_flat          = (1 << 2),
   nir_pack_varying_interp_mode_noperspective = (1 << 3),
   nir_pack_varying_interp_loc_sample         = (1 << 16),
   nir_pack_varying_interp_loc_centroid       = (1 << 17),
   nir_pack_varying_interp_loc_center         = (1 << 18),
} nir_pack_varying_options;

/** An instruction filtering callback
 *
 * Returns true if the instruction should be processed and false otherwise.
 */
typedef bool (*nir_instr_filter_cb)(const nir_instr *, const void *);

/** A vectorization width callback
 *
 * Returns the maximum vectorization width per instruction.
 * 0, if the instruction must not be modified.
 *
 * The vectorization width must be a power of 2.
 */
typedef uint8_t (*nir_vectorize_cb)(const nir_instr *, const void *);

typedef struct nir_shader_compiler_options {
   bool lower_fdiv;
   bool lower_ffma16;
   bool lower_ffma32;
   bool lower_ffma64;
   bool fuse_ffma16;
   bool fuse_ffma32;
   bool fuse_ffma64;
   bool lower_flrp16;
   bool lower_flrp32;
   /** Lowers flrp when it does not support doubles */
   bool lower_flrp64;
   bool lower_fpow;
   bool lower_fsat;
   bool lower_fsqrt;
   bool lower_sincos;
   bool lower_fmod;
   /** Lowers ibitfield_extract/ubitfield_extract to ibfe/ubfe. */
   bool lower_bitfield_extract;
   /** Lowers ibitfield_extract/ubitfield_extract to compares, shifts. */
   bool lower_bitfield_extract_to_shifts;
   /** Lowers bitfield_insert to bfi/bfm */
   bool lower_bitfield_insert;
   /** Lowers bitfield_insert to compares, and shifts. */
   bool lower_bitfield_insert_to_shifts;
   /** Lowers bitfield_insert to bfm/bitfield_select. */
   bool lower_bitfield_insert_to_bitfield_select;
   /** Lowers bitfield_reverse to shifts. */
   bool lower_bitfield_reverse;
   /** Lowers bit_count to shifts. */
   bool lower_bit_count;
   /** Lowers ifind_msb to compare and ufind_msb */
   bool lower_ifind_msb;
   /** Lowers ifind_msb and ufind_msb to reverse variants */
   bool lower_find_msb_to_reverse;
   /** Lowers find_lsb to ufind_msb and logic ops */
   bool lower_find_lsb;
   bool lower_uadd_carry;
   bool lower_usub_borrow;
   /** Lowers imul_high/umul_high to 16-bit multiplies and carry operations. */
   bool lower_mul_high;
   /** lowers fneg to fmul(x, -1.0). Driver must call nir_opt_algebraic_late() */
   bool lower_fneg;
   /** lowers ineg to isub. Driver must call nir_opt_algebraic_late(). */
   bool lower_ineg;
   /** lowers fisnormal to alu ops. */
   bool lower_fisnormal;

   /* lower {slt,sge,seq,sne} to {flt,fge,feq,fneu} + b2f: */
   bool lower_scmp;

   /* lower b/fall_equalN/b/fany_nequalN (ex:fany_nequal4 to sne+fdot4+fsat) */
   bool lower_vector_cmp;

   /** enable rules to avoid bit ops */
   bool lower_bitops;

   /** enables rules to lower isign to imin+imax */
   bool lower_isign;

   /** enables rules to lower fsign to fsub and flt */
   bool lower_fsign;

   /** enables rules to lower iabs to ineg+imax */
   bool lower_iabs;

   /** enable rules that avoid generating umax from signed integer ops */
   bool lower_umax;

   /** enable rules that avoid generating umin from signed integer ops */
   bool lower_umin;

   /* lower fdph to fdot4 */
   bool lower_fdph;

   /** lower fdot to fmul and fsum/fadd. */
   bool lower_fdot;

   /* Does the native fdot instruction replicate its result for four
    * components?  If so, then opt_algebraic_late will turn all fdotN
    * instructions into fdotN_replicated instructions.
    */
   bool fdot_replicates;

   /** lowers ffloor to fsub+ffract: */
   bool lower_ffloor;

   /** lowers ffract to fsub+ffloor: */
   bool lower_ffract;

   /** lowers fceil to fneg+ffloor+fneg: */
   bool lower_fceil;

   bool lower_ftrunc;

   /** Lowers fround_even to ffract+feq+csel.
    *
    * Not correct in that it doesn't correctly handle the "_even" part of the
    * rounding, but good enough for DX9 array indexing handling on DX9-class
    * hardware.
    */
   bool lower_fround_even;

   bool lower_ldexp;

   bool lower_pack_half_2x16;
   bool lower_pack_unorm_2x16;
   bool lower_pack_snorm_2x16;
   bool lower_pack_unorm_4x8;
   bool lower_pack_snorm_4x8;
   bool lower_pack_64_2x32;
   bool lower_pack_64_4x16;
   bool lower_pack_32_2x16;
   bool lower_pack_64_2x32_split;
   bool lower_pack_32_2x16_split;
   bool lower_unpack_half_2x16;
   bool lower_unpack_unorm_2x16;
   bool lower_unpack_snorm_2x16;
   bool lower_unpack_unorm_4x8;
   bool lower_unpack_snorm_4x8;
   bool lower_unpack_64_2x32_split;
   bool lower_unpack_32_2x16_split;

   bool lower_pack_split;

   bool lower_extract_byte;
   bool lower_extract_word;
   bool lower_insert_byte;
   bool lower_insert_word;

   bool lower_all_io_to_temps;
   bool lower_all_io_to_elements;

   /* Indicates that the driver only has zero-based vertex id */
   bool vertex_id_zero_based;

   /**
    * If enabled, gl_BaseVertex will be lowered as:
    * is_indexed_draw (~0/0) & firstvertex
    */
   bool lower_base_vertex;

   /**
    * If enabled, gl_HelperInvocation will be lowered as:
    *
    *   !((1 << sample_id) & sample_mask_in))
    *
    * This depends on some possibly hw implementation details, which may
    * not be true for all hw.  In particular that the FS is only executed
    * for covered samples or for helper invocations.  So, do not blindly
    * enable this option.
    *
    * Note: See also issue #22 in ARB_shader_image_load_store
    */
   bool lower_helper_invocation;

   /**
    * Convert gl_SampleMaskIn to gl_HelperInvocation as follows:
    *
    *   gl_SampleMaskIn == 0 ---> gl_HelperInvocation
    *   gl_SampleMaskIn != 0 ---> !gl_HelperInvocation
    */
   bool optimize_sample_mask_in;

   bool lower_cs_local_index_to_id;
   bool lower_cs_local_id_to_index;

   /* Prevents lowering global_invocation_id to be in terms of workgroup_id */
   bool has_cs_global_id;

   bool lower_device_index_to_zero;

   /* Set if nir_lower_pntc_ytransform() should invert gl_PointCoord.
    * Either when frame buffer is flipped or GL_POINT_SPRITE_COORD_ORIGIN
    * is GL_LOWER_LEFT.
    */
   bool lower_wpos_pntc;

   /**
    * Set if nir_op_[iu]hadd and nir_op_[iu]rhadd instructions should be
    * lowered to simple arithmetic.
    *
    * If this flag is set, the lowering will be applied to all bit-sizes of
    * these instructions.
    *
    * \sa ::lower_hadd64
    */
   bool lower_hadd;

   /**
    * Set if only 64-bit nir_op_[iu]hadd and nir_op_[iu]rhadd instructions
    * should be lowered to simple arithmetic.
    *
    * If this flag is set, the lowering will be applied to only 64-bit
    * versions of these instructions.
    *
    * \sa ::lower_hadd
    */
   bool lower_hadd64;

   /**
    * Set if nir_op_uadd_sat should be lowered to simple arithmetic.
    *
    * If this flag is set, the lowering will be applied to all bit-sizes of
    * these instructions.
    */
   bool lower_uadd_sat;

   /**
    * Set if nir_op_usub_sat should be lowered to simple arithmetic.
    *
    * If this flag is set, the lowering will be applied to all bit-sizes of
    * these instructions.
    */
   bool lower_usub_sat;

   /**
    * Set if nir_op_iadd_sat and nir_op_isub_sat should be lowered to simple
    * arithmetic.
    *
    * If this flag is set, the lowering will be applied to all bit-sizes of
    * these instructions.
    */
   bool lower_iadd_sat;

   /**
    * Set if imul_32x16 and umul_32x16 should be lowered to simple
    * arithmetic.
    */
   bool lower_mul_32x16;

   /**
    * Set if uclz should be lowered to find_msb_rev.
    */
   bool lower_uclz;

   /**
    * Should IO be re-vectorized?  Some scalar ISAs still operate on vec4's
    * for IO purposes and would prefer loads/stores be vectorized.
    */
   bool vectorize_io;
   bool vectorize_tess_levels;
   bool lower_to_scalar;
   nir_instr_filter_cb lower_to_scalar_filter;

   /**
    * Disables potentially harmful algebraic transformations for architectures
    * with SIMD-within-a-register semantics.
    *
    * Note, to actually vectorize 16bit instructions, use nir_opt_vectorize()
    * with a suitable callback function.
    */
   bool vectorize_vec2_16bit;

   /**
    * Should the linker unify inputs_read/outputs_written between adjacent
    * shader stages which are linked into a single program?
    */
   bool unify_interfaces;

   /**
    * Should nir_lower_io() create load_interpolated_input intrinsics?
    *
    * If not, it generates regular load_input intrinsics and interpolation
    * information must be inferred from the list of input nir_variables.
    */
   bool use_interpolated_input_intrinsics;


   /**
    * Whether nir_lower_io() will lower interpolateAt functions to
    * load_interpolated_input intrinsics.
    *
    * Unlike use_interpolated_input_intrinsics this will only lower these
    * functions and leave input load intrinsics untouched.
    */
   bool lower_interpolate_at;

   /* Lowers when 32x32->64 bit multiplication is not supported */
   bool lower_mul_2x32_64;

   /* Lowers when rotate instruction is not supported */
   bool lower_rotate;

   /** Backend supports ternary addition */
   bool has_iadd3;

   /**
    * Backend supports imul24, and would like to use it (when possible)
    * for address/offset calculation.  If true, driver should call
    * nir_lower_amul().  (If not set, amul will automatically be lowered
    * to imul.)
    */
   bool has_imul24;

   /** Backend supports umul24, if not set  umul24 will automatically be lowered
    * to imul with masked inputs */
   bool has_umul24;

   /** Backend supports umad24, if not set  umad24 will automatically be lowered
    * to imul with masked inputs and iadd */
   bool has_umad24;

   /* Backend supports fused comapre against zero and csel */
   bool has_fused_comp_and_csel;

   /** Backend supports fsub, if not set fsub will automatically be lowered to
    * fadd(x, fneg(y)). If true, driver should call nir_opt_algebraic_late(). */
   bool has_fsub;

   /** Backend supports isub, if not set isub will automatically be lowered to
    * iadd(x, ineg(y)). If true, driver should call nir_opt_algebraic_late(). */
   bool has_isub;

   /** Backend supports pack_32_4x8 or pack_32_4x8_split. */
   bool has_pack_32_4x8;

   /** Backend supports txs, if not nir_lower_tex(..) uses txs-free variants
    * for rect texture lowering. */
   bool has_txs;

   /** Backend supports sdot_4x8 opcodes. */
   bool has_sdot_4x8;

   /** Backend supports udot_4x8 opcodes. */
   bool has_udot_4x8;

   /** Backend supports sudot_4x8 opcodes. */
   bool has_sudot_4x8;

   /** Backend supports sdot_2x16 and udot_2x16 opcodes. */
   bool has_dot_2x16;

   /* Whether to generate only scoped_barrier intrinsics instead of the set of
    * memory and control barrier intrinsics based on GLSL.
    */
   bool use_scoped_barrier;

   /** Backend supports fmulz (and ffmaz if lower_ffma32=false) */
   bool has_fmulz;

   /** Backend supports 32bit ufind_msb_rev and ifind_msb_rev. */
   bool has_find_msb_rev;

   /**
    * Is this the Intel vec4 backend?
    *
    * Used to inhibit algebraic optimizations that are known to be harmful on
    * the Intel vec4 backend.  This is generally applicable to any
    * optimization that might cause more immediate values to be used in
    * 3-source (e.g., ffma and flrp) instructions.
    */
   bool intel_vec4;

   /**
    * For most Intel GPUs, all ternary operations such as FMA and BFE cannot
    * have immediates, so two to three instructions may eventually be needed.
    */
   bool avoid_ternary_with_two_constants;

   /** Whether 8-bit ALU is supported. */
   bool support_8bit_alu;

   /** Whether 16-bit ALU is supported. */
   bool support_16bit_alu;

   unsigned max_unroll_iterations;
   unsigned max_unroll_iterations_aggressive;
   unsigned max_unroll_iterations_fp64;

   bool lower_uniforms_to_ubo;

   /* If the precision is ignored, backends that don't handle
    * different precisions when passing data between stages and use
    * vectorized IO can pack more varyings when linking. */
   bool linker_ignore_precision;

   /* Specifies if indirect sampler array access will trigger forced loop
    * unrolling.
    */
   bool force_indirect_unrolling_sampler;

   /* Some older drivers don't support GLSL versions with the concept of flat
    * varyings and also don't support integers. This setting helps us avoid
    * marking varyings as flat and potentially having them changed to ints via
    * varying packing.
    */
   bool no_integers;

   /**
    * Specifies which type of indirectly accessed variables should force
    * loop unrolling.
    */
   nir_variable_mode force_indirect_unrolling;

   nir_lower_int64_options lower_int64_options;
   nir_lower_doubles_options lower_doubles_options;
   nir_divergence_options divergence_analysis_options;

   /**
    * Support pack varyings with different interpolation location
    * (center, centroid, sample) and mode (flat, noperspective, smooth)
    * into same slot.
    */
   nir_pack_varying_options pack_varying_options;

   /**
    * Lower load_deref/store_deref of inputs and outputs into
    * load_input/store_input intrinsics. This is used by nir_lower_io_passes.
    */
   bool lower_io_variables;

   /**
    * Lower color inputs to load_colorN that are kind of like system values
    * if lower_io_variables is also set. shader_info will contain
    * the interpolation settings. This is used by nir_lower_io_passes.
    */
   bool lower_fs_color_inputs;

   /**
    * The masks of shader stages that support indirect indexing with
    * load_input and store_output intrinsics. It's used when
    * lower_io_variables is true. This is used by nir_lower_io_passes.
    */
   uint8_t support_indirect_inputs;
   uint8_t support_indirect_outputs;

   /**
    * Remove varying loaded from uniform, let fragment shader load the
    * uniform directly. GPU passing varying by memory can benifit from it
    * for sure; but GPU passing varying by on chip resource may not.
    * Because it saves on chip resource but may increase memory pressure when
    * fragment task is far more than vertex one, so better left it disabled.
    */
   bool lower_varying_from_uniform;

   /** store the variable offset into the instrinsic range_base instead
    *  of adding it to the image index.
    */
   bool lower_image_offset_to_range_base;

   /** store the variable offset into the instrinsic range_base instead
    *  of adding it to the atomic source
    */
   bool lower_atomic_offset_to_range_base;
} nir_shader_compiler_options;

typedef struct nir_shader {
   gc_ctx *gctx;

   /** list of uniforms (nir_variable) */
   struct exec_list variables;

   /** Set of driver-specific options for the shader.
    *
    * The memory for the options is expected to be kept in a single static
    * copy by the driver.
    */
   const struct nir_shader_compiler_options *options;

   /** Various bits of compile-time information about a given shader */
   struct shader_info info;

   struct exec_list functions; /** < list of nir_function */

   /**
    * The size of the variable space for load_input_*, load_uniform_*, etc.
    * intrinsics.  This is in back-end specific units which is likely one of
    * bytes, dwords, or vec4s depending on context and back-end.
    */
   unsigned num_inputs, num_uniforms, num_outputs;

   /** Size in bytes of required implicitly bound global memory */
   unsigned global_mem_size;

   /** Size in bytes of required scratch space */
   unsigned scratch_size;

   /** Constant data associated with this shader.
    *
    * Constant data is loaded through load_constant intrinsics (as compared to
    * the NIR load_const instructions which have the constant value inlined
    * into them).  This is usually generated by nir_opt_large_constants (so
    * shaders don't have to load_const into a temporary array when they want
    * to indirect on a const array).
    */
   void *constant_data;
   /** Size of the constant data associated with the shader, in bytes */
   unsigned constant_data_size;

   struct nir_xfb_info *xfb_info;

   unsigned printf_info_count;
   u_printf_info *printf_info;
} nir_shader;

#define nir_foreach_function(func, shader) \
   foreach_list_typed(nir_function, func, node, &(shader)->functions)

static inline nir_function_impl *
nir_shader_get_entrypoint(const nir_shader *shader)
{
   nir_function *func = NULL;

   nir_foreach_function(function, shader) {
      assert(func == NULL);
      if (function->is_entrypoint) {
         func = function;
#ifndef NDEBUG
         break;
#endif
      }
   }

   if (!func)
      return NULL;

   assert(func->num_params == 0);
   assert(func->impl);
   return func->impl;
}

static inline nir_function *
nir_shader_get_function_for_name(const nir_shader *shader, const char *name)
{
   nir_foreach_function(func, shader) {
      if (strcmp(func->name, name) == 0)
         return func;
   }

   return NULL;
}

void nir_remove_non_entrypoints(nir_shader *shader);

nir_shader *nir_shader_create(void *mem_ctx,
                              gl_shader_stage stage,
                              const nir_shader_compiler_options *options,
                              shader_info *si);

nir_register *nir_local_reg_create(nir_function_impl *impl);

void nir_reg_remove(nir_register *reg);

/** Adds a variable to the appropriate list in nir_shader */
void nir_shader_add_variable(nir_shader *shader, nir_variable *var);

static inline void
nir_function_impl_add_variable(nir_function_impl *impl, nir_variable *var)
{
   assert(var->data.mode == nir_var_function_temp);
   exec_list_push_tail(&impl->locals, &var->node);
}

/** creates a variable, sets a few defaults, and adds it to the list */
nir_variable *nir_variable_create(nir_shader *shader,
                                  nir_variable_mode mode,
                                  const struct glsl_type *type,
                                  const char *name);
/** creates a local variable and adds it to the list */
nir_variable *nir_local_variable_create(nir_function_impl *impl,
                                        const struct glsl_type *type,
                                        const char *name);

nir_variable *nir_find_variable_with_location(nir_shader *shader,
                                              nir_variable_mode mode,
                                              unsigned location);

nir_variable *nir_find_variable_with_driver_location(nir_shader *shader,
                                                     nir_variable_mode mode,
                                                     unsigned location);

void nir_sort_variables_with_modes(nir_shader *shader,
                                   int (*compar)(const nir_variable *,
                                                 const nir_variable *),
                                   nir_variable_mode modes);

/** creates a function and adds it to the shader's list of functions */
nir_function *nir_function_create(nir_shader *shader, const char *name);

nir_function_impl *nir_function_impl_create(nir_function *func);
/** creates a function_impl that isn't tied to any particular function */
nir_function_impl *nir_function_impl_create_bare(nir_shader *shader);

nir_block *nir_block_create(nir_shader *shader);
nir_if *nir_if_create(nir_shader *shader);
nir_loop *nir_loop_create(nir_shader *shader);

nir_function_impl *nir_cf_node_get_function(nir_cf_node *node);

/** requests that the given pieces of metadata be generated */
void nir_metadata_require(nir_function_impl *impl, nir_metadata required, ...);
/** dirties all but the preserved metadata */
void nir_metadata_preserve(nir_function_impl *impl, nir_metadata preserved);
/** Preserves all metadata for the given shader */
void nir_shader_preserve_all_metadata(nir_shader *shader);

/** creates an instruction with default swizzle/writemask/etc. with NULL registers */
nir_alu_instr *nir_alu_instr_create(nir_shader *shader, nir_op op);

nir_deref_instr *nir_deref_instr_create(nir_shader *shader,
                                        nir_deref_type deref_type);

nir_jump_instr *nir_jump_instr_create(nir_shader *shader, nir_jump_type type);

nir_load_const_instr *nir_load_const_instr_create(nir_shader *shader,
                                                  unsigned num_components,
                                                  unsigned bit_size);

nir_intrinsic_instr *nir_intrinsic_instr_create(nir_shader *shader,
                                                nir_intrinsic_op op);

nir_call_instr *nir_call_instr_create(nir_shader *shader,
                                      nir_function *callee);

/** Creates a NIR texture instruction */
nir_tex_instr *nir_tex_instr_create(nir_shader *shader, unsigned num_srcs);

nir_phi_instr *nir_phi_instr_create(nir_shader *shader);
nir_phi_src *nir_phi_instr_add_src(nir_phi_instr *instr, nir_block *pred, nir_src src);

nir_parallel_copy_instr *nir_parallel_copy_instr_create(nir_shader *shader);

nir_ssa_undef_instr *nir_ssa_undef_instr_create(nir_shader *shader,
                                                unsigned num_components,
                                                unsigned bit_size);

nir_const_value nir_alu_binop_identity(nir_op binop, unsigned bit_size);

/**
 * NIR Cursors and Instruction Insertion API
 * @{
 *
 * A tiny struct representing a point to insert/extract instructions or
 * control flow nodes.  Helps reduce the combinatorial explosion of possible
 * points to insert/extract.
 *
 * \sa nir_control_flow.h
 */
typedef enum {
   nir_cursor_before_block,
   nir_cursor_after_block,
   nir_cursor_before_instr,
   nir_cursor_after_instr,
} nir_cursor_option;

typedef struct {
   nir_cursor_option option;
   union {
      nir_block *block;
      nir_instr *instr;
   };
} nir_cursor;

static inline nir_block *
nir_cursor_current_block(nir_cursor cursor)
{
   if (cursor.option == nir_cursor_before_instr ||
       cursor.option == nir_cursor_after_instr) {
      return cursor.instr->block;
   } else {
      return cursor.block;
   }
}

bool nir_cursors_equal(nir_cursor a, nir_cursor b);

static inline nir_cursor
nir_before_block(nir_block *block)
{
   nir_cursor cursor;
   cursor.option = nir_cursor_before_block;
   cursor.block = block;
   return cursor;
}

static inline nir_cursor
nir_after_block(nir_block *block)
{
   nir_cursor cursor;
   cursor.option = nir_cursor_after_block;
   cursor.block = block;
   return cursor;
}

static inline nir_cursor
nir_before_instr(nir_instr *instr)
{
   nir_cursor cursor;
   cursor.option = nir_cursor_before_instr;
   cursor.instr = instr;
   return cursor;
}

static inline nir_cursor
nir_after_instr(nir_instr *instr)
{
   nir_cursor cursor;
   cursor.option = nir_cursor_after_instr;
   cursor.instr = instr;
   return cursor;
}

static inline nir_cursor
nir_before_block_after_phis(nir_block *block)
{
   nir_phi_instr *last_phi = nir_block_last_phi_instr(block);
   if (last_phi)
      return nir_after_instr(&last_phi->instr);
   else
      return nir_before_block(block);
}

static inline nir_cursor
nir_after_block_before_jump(nir_block *block)
{
   nir_instr *last_instr = nir_block_last_instr(block);
   if (last_instr && last_instr->type == nir_instr_type_jump) {
      return nir_before_instr(last_instr);
   } else {
      return nir_after_block(block);
   }
}

static inline nir_cursor
nir_before_src(nir_src *src, bool is_if_condition)
{
   if (is_if_condition) {
      nir_block *prev_block =
         nir_cf_node_as_block(nir_cf_node_prev(&src->parent_if->cf_node));
      assert(!nir_block_ends_in_jump(prev_block));
      return nir_after_block(prev_block);
   } else if (src->parent_instr->type == nir_instr_type_phi) {
#ifndef NDEBUG
      nir_phi_instr *cond_phi = nir_instr_as_phi(src->parent_instr);
      bool found = false;
      nir_foreach_phi_src(phi_src, cond_phi) {
         if (phi_src->src.ssa == src->ssa) {
            found = true;
            break;
         }
      }
      assert(found);
#endif
      /* The list_entry() macro is a generic container-of macro, it just happens
       * to have a more specific name.
       */
      nir_phi_src *phi_src = list_entry(src, nir_phi_src, src);
      return nir_after_block_before_jump(phi_src->pred);
   } else {
      return nir_before_instr(src->parent_instr);
   }
}

static inline nir_cursor
nir_before_cf_node(nir_cf_node *node)
{
   if (node->type == nir_cf_node_block)
      return nir_before_block(nir_cf_node_as_block(node));

   return nir_after_block(nir_cf_node_as_block(nir_cf_node_prev(node)));
}

static inline nir_cursor
nir_after_cf_node(nir_cf_node *node)
{
   if (node->type == nir_cf_node_block)
      return nir_after_block(nir_cf_node_as_block(node));

   return nir_before_block(nir_cf_node_as_block(nir_cf_node_next(node)));
}

static inline nir_cursor
nir_after_phis(nir_block *block)
{
   nir_foreach_instr(instr, block) {
      if (instr->type != nir_instr_type_phi)
         return nir_before_instr(instr);
   }
   return nir_after_block(block);
}

static inline nir_cursor
nir_after_instr_and_phis(nir_instr *instr)
{
   if (instr->type == nir_instr_type_phi)
      return nir_after_phis(instr->block);
   else
      return nir_after_instr(instr);
}

static inline nir_cursor
nir_after_cf_node_and_phis(nir_cf_node *node)
{
   if (node->type == nir_cf_node_block)
      return nir_after_block(nir_cf_node_as_block(node));

   nir_block *block = nir_cf_node_as_block(nir_cf_node_next(node));

   return nir_after_phis(block);
}

static inline nir_cursor
nir_before_cf_list(struct exec_list *cf_list)
{
   nir_cf_node *first_node = exec_node_data(nir_cf_node,
                                            exec_list_get_head(cf_list), node);
   return nir_before_cf_node(first_node);
}

static inline nir_cursor
nir_after_cf_list(struct exec_list *cf_list)
{
   nir_cf_node *last_node = exec_node_data(nir_cf_node,
                                           exec_list_get_tail(cf_list), node);
   return nir_after_cf_node(last_node);
}

/**
 * Insert a NIR instruction at the given cursor.
 *
 * Note: This does not update the cursor.
 */
void nir_instr_insert(nir_cursor cursor, nir_instr *instr);

bool nir_instr_move(nir_cursor cursor, nir_instr *instr);

static inline void
nir_instr_insert_before(nir_instr *instr, nir_instr *before)
{
   nir_instr_insert(nir_before_instr(instr), before);
}

static inline void
nir_instr_insert_after(nir_instr *instr, nir_instr *after)
{
   nir_instr_insert(nir_after_instr(instr), after);
}

static inline void
nir_instr_insert_before_block(nir_block *block, nir_instr *before)
{
   nir_instr_insert(nir_before_block(block), before);
}

static inline void
nir_instr_insert_after_block(nir_block *block, nir_instr *after)
{
   nir_instr_insert(nir_after_block(block), after);
}

static inline void
nir_instr_insert_before_cf(nir_cf_node *node, nir_instr *before)
{
   nir_instr_insert(nir_before_cf_node(node), before);
}

static inline void
nir_instr_insert_after_cf(nir_cf_node *node, nir_instr *after)
{
   nir_instr_insert(nir_after_cf_node(node), after);
}

static inline void
nir_instr_insert_before_cf_list(struct exec_list *list, nir_instr *before)
{
   nir_instr_insert(nir_before_cf_list(list), before);
}

static inline void
nir_instr_insert_after_cf_list(struct exec_list *list, nir_instr *after)
{
   nir_instr_insert(nir_after_cf_list(list), after);
}

void nir_instr_remove_v(nir_instr *instr);
void nir_instr_free(nir_instr *instr);
void nir_instr_free_list(struct exec_list *list);

static inline nir_cursor
nir_instr_remove(nir_instr *instr)
{
   nir_cursor cursor;
   nir_instr *prev = nir_instr_prev(instr);
   if (prev) {
      cursor = nir_after_instr(prev);
   } else {
      cursor = nir_before_block(instr->block);
   }
   nir_instr_remove_v(instr);
   return cursor;
}

nir_cursor nir_instr_free_and_dce(nir_instr *instr);

/** @} */

nir_ssa_def *nir_instr_ssa_def(nir_instr *instr);
bool nir_instr_def_is_register(nir_instr *instr);

typedef bool (*nir_foreach_ssa_def_cb)(nir_ssa_def *def, void *state);
typedef bool (*nir_foreach_dest_cb)(nir_dest *dest, void *state);
typedef bool (*nir_foreach_src_cb)(nir_src *src, void *state);
bool nir_foreach_ssa_def(nir_instr *instr, nir_foreach_ssa_def_cb cb,
                         void *state);
static inline bool nir_foreach_dest(nir_instr *instr, nir_foreach_dest_cb cb, void *state);
static inline bool nir_foreach_src(nir_instr *instr, nir_foreach_src_cb cb, void *state);
bool nir_foreach_phi_src_leaving_block(nir_block *instr,
                                       nir_foreach_src_cb cb,
                                       void *state);

nir_const_value *nir_src_as_const_value(nir_src src);

#define NIR_SRC_AS_(name, c_type, type_enum, cast_macro)                \
static inline c_type *                                                  \
nir_src_as_ ## name (nir_src src)                                       \
{                                                                       \
    return src.is_ssa && src.ssa->parent_instr->type == type_enum       \
           ? cast_macro(src.ssa->parent_instr) : NULL;                  \
}

NIR_SRC_AS_(alu_instr, nir_alu_instr, nir_instr_type_alu, nir_instr_as_alu)
NIR_SRC_AS_(intrinsic, nir_intrinsic_instr,
            nir_instr_type_intrinsic, nir_instr_as_intrinsic)
NIR_SRC_AS_(deref, nir_deref_instr, nir_instr_type_deref, nir_instr_as_deref)

bool nir_src_is_always_uniform(nir_src src);
bool nir_srcs_equal(nir_src src1, nir_src src2);
bool nir_instrs_equal(const nir_instr *instr1, const nir_instr *instr2);

static inline void
nir_instr_rewrite_src_ssa(ASSERTED nir_instr *instr,
                          nir_src *src, nir_ssa_def *new_ssa)
{
   assert(src->parent_instr == instr);
   assert(src->is_ssa && src->ssa);
   list_del(&src->use_link);
   src->ssa = new_ssa;
   list_addtail(&src->use_link, &new_ssa->uses);
}

void nir_instr_rewrite_src(nir_instr *instr, nir_src *src, nir_src new_src);
void nir_instr_move_src(nir_instr *dest_instr, nir_src *dest, nir_src *src);

static inline void
nir_if_rewrite_condition_ssa(ASSERTED nir_if *if_stmt,
                             nir_src *src, nir_ssa_def *new_ssa)
{
   assert(src->parent_if == if_stmt);
   assert(src->is_ssa && src->ssa);
   list_del(&src->use_link);
   src->ssa = new_ssa;
   list_addtail(&src->use_link, &new_ssa->if_uses);
}

void nir_if_rewrite_condition(nir_if *if_stmt, nir_src new_src);
void nir_instr_rewrite_dest(nir_instr *instr, nir_dest *dest,
                            nir_dest new_dest);

void nir_ssa_dest_init(nir_instr *instr, nir_dest *dest,
                       unsigned num_components, unsigned bit_size,
                       const char *name);
void nir_ssa_def_init(nir_instr *instr, nir_ssa_def *def,
                      unsigned num_components, unsigned bit_size);
static inline void
nir_ssa_dest_init_for_type(nir_instr *instr, nir_dest *dest,
                           const struct glsl_type *type,
                           const char *name)
{
   assert(glsl_type_is_vector_or_scalar(type));
   nir_ssa_dest_init(instr, dest, glsl_get_components(type),
                     glsl_get_bit_size(type), name);
}
void nir_ssa_def_rewrite_uses(nir_ssa_def *def, nir_ssa_def *new_ssa);
void nir_ssa_def_rewrite_uses_src(nir_ssa_def *def, nir_src new_src);
void nir_ssa_def_rewrite_uses_after(nir_ssa_def *def, nir_ssa_def *new_ssa,
                                    nir_instr *after_me);

nir_component_mask_t nir_src_components_read(const nir_src *src);
nir_component_mask_t nir_ssa_def_components_read(const nir_ssa_def *def);

static inline bool
nir_ssa_def_is_unused(nir_ssa_def *ssa)
{
   return list_is_empty(&ssa->uses) && list_is_empty(&ssa->if_uses);
}


/** Returns the next block, disregarding structure
 *
 * The ordering is deterministic but has no guarantees beyond that.  In
 * particular, it is not guaranteed to be dominance-preserving.
 */
nir_block *nir_block_unstructured_next(nir_block *block);
nir_block *nir_unstructured_start_block(nir_function_impl *impl);

#define nir_foreach_block_unstructured(block, impl) \
   for (nir_block *block = nir_unstructured_start_block(impl); block != NULL; \
        block = nir_block_unstructured_next(block))

#define nir_foreach_block_unstructured_safe(block, impl) \
   for (nir_block *block = nir_unstructured_start_block(impl), \
        *next = nir_block_unstructured_next(block); \
        block != NULL; \
        block = next, next = nir_block_unstructured_next(block))

/*
 * finds the next basic block in source-code order, returns NULL if there is
 * none
 */

nir_block *nir_block_cf_tree_next(nir_block *block);

/* Performs the opposite of nir_block_cf_tree_next() */

nir_block *nir_block_cf_tree_prev(nir_block *block);

/* Gets the first block in a CF node in source-code order */

nir_block *nir_cf_node_cf_tree_first(nir_cf_node *node);

/* Gets the last block in a CF node in source-code order */

nir_block *nir_cf_node_cf_tree_last(nir_cf_node *node);

/* Gets the next block after a CF node in source-code order */

nir_block *nir_cf_node_cf_tree_next(nir_cf_node *node);

/* Macros for loops that visit blocks in source-code order */

#define nir_foreach_block(block, impl) \
   for (nir_block *block = nir_start_block(impl); block != NULL; \
        block = nir_block_cf_tree_next(block))

#define nir_foreach_block_safe(block, impl) \
   for (nir_block *block = nir_start_block(impl), \
        *next = nir_block_cf_tree_next(block); \
        block != NULL; \
        block = next, next = nir_block_cf_tree_next(block))

#define nir_foreach_block_reverse(block, impl) \
   for (nir_block *block = nir_impl_last_block(impl); block != NULL; \
        block = nir_block_cf_tree_prev(block))

#define nir_foreach_block_reverse_safe(block, impl) \
   for (nir_block *block = nir_impl_last_block(impl), \
        *prev = nir_block_cf_tree_prev(block); \
        block != NULL; \
        block = prev, prev = nir_block_cf_tree_prev(block))

#define nir_foreach_block_in_cf_node(block, node) \
   for (nir_block *block = nir_cf_node_cf_tree_first(node); \
        block != nir_cf_node_cf_tree_next(node); \
        block = nir_block_cf_tree_next(block))

/* If the following CF node is an if, this function returns that if.
 * Otherwise, it returns NULL.
 */
nir_if *nir_block_get_following_if(nir_block *block);

nir_loop *nir_block_get_following_loop(nir_block *block);

nir_block **nir_block_get_predecessors_sorted(const nir_block *block, void *mem_ctx);

void nir_index_local_regs(nir_function_impl *impl);
void nir_index_ssa_defs(nir_function_impl *impl);
unsigned nir_index_instrs(nir_function_impl *impl);

void nir_index_blocks(nir_function_impl *impl);

unsigned nir_shader_index_vars(nir_shader *shader, nir_variable_mode modes);
unsigned nir_function_impl_index_vars(nir_function_impl *impl);

void nir_print_shader(nir_shader *shader, FILE *fp);
void nir_print_shader_annotated(nir_shader *shader, FILE *fp, struct hash_table *errors);
void nir_print_instr(const nir_instr *instr, FILE *fp);
void nir_print_deref(const nir_deref_instr *deref, FILE *fp);
void nir_log_shader_annotated_tagged(enum mesa_log_level level, const char *tag, nir_shader *shader, struct hash_table *annotations);
#define nir_log_shadere(s) nir_log_shader_annotated_tagged(MESA_LOG_ERROR, (MESA_LOG_TAG), (s), NULL)
#define nir_log_shaderw(s) nir_log_shader_annotated_tagged(MESA_LOG_WARN, (MESA_LOG_TAG), (s), NULL)
#define nir_log_shaderi(s) nir_log_shader_annotated_tagged(MESA_LOG_INFO, (MESA_LOG_TAG), (s), NULL)
#define nir_log_shader_annotated(s, annotations) nir_log_shader_annotated_tagged(MESA_LOG_ERROR, (MESA_LOG_TAG), (s), annotations)

char *nir_shader_as_str(nir_shader *nir, void *mem_ctx);
char *nir_shader_as_str_annotated(nir_shader *nir, struct hash_table *annotations, void *mem_ctx);
char *nir_instr_as_str(const nir_instr *instr, void *mem_ctx);

/** Shallow clone of a single instruction. */
nir_instr *nir_instr_clone(nir_shader *s, const nir_instr *orig);

/** Clone a single instruction, including a remap table to rewrite sources. */
nir_instr *nir_instr_clone_deep(nir_shader *s, const nir_instr *orig,
                                struct hash_table *remap_table);

/** Shallow clone of a single ALU instruction. */
nir_alu_instr *nir_alu_instr_clone(nir_shader *s, const nir_alu_instr *orig);

nir_shader *nir_shader_clone(void *mem_ctx, const nir_shader *s);
nir_function_impl *nir_function_impl_clone(nir_shader *shader,
                                           const nir_function_impl *fi);
nir_constant *nir_constant_clone(const nir_constant *c, nir_variable *var);
nir_variable *nir_variable_clone(const nir_variable *c, nir_shader *shader);

void nir_shader_replace(nir_shader *dest, nir_shader *src);

void nir_shader_serialize_deserialize(nir_shader *s);

#ifndef NDEBUG
void nir_validate_shader(nir_shader *shader, const char *when);
void nir_validate_ssa_dominance(nir_shader *shader, const char *when);
void nir_metadata_set_validation_flag(nir_shader *shader);
void nir_metadata_check_validation_flag(nir_shader *shader);

static inline bool
should_skip_nir(const char *name)
{
   static const char *list = NULL;
   if (!list) {
      /* Comma separated list of names to skip. */
      list = getenv("NIR_SKIP");
      if (!list)
         list = "";
   }

   if (!list[0])
      return false;

   return comma_separated_list_contains(list, name);
}

static inline bool
should_print_nir(nir_shader *shader)
{
   if (shader->info.internal ||
       shader->info.stage < 0 ||
       shader->info.stage > MESA_SHADER_KERNEL)
      return false;

   return unlikely(nir_debug_print_shader[shader->info.stage]);
}
#else
static inline void nir_validate_shader(nir_shader *shader, const char *when) { (void) shader; (void)when; }
static inline void nir_validate_ssa_dominance(nir_shader *shader, const char *when) { (void) shader; (void)when; }
static inline void nir_metadata_set_validation_flag(nir_shader *shader) { (void) shader; }
static inline void nir_metadata_check_validation_flag(nir_shader *shader) { (void) shader; }
static inline bool should_skip_nir(UNUSED const char *pass_name) { return false; }
static inline bool should_print_nir(UNUSED nir_shader *shader) { return false; }
#endif /* NDEBUG */

#define _PASS(pass, nir, do_pass) do {                               \
   if (should_skip_nir(#pass)) {                                     \
      printf("skipping %s\n", #pass);                                \
      break;                                                         \
   }                                                                 \
   do_pass                                                           \
   if (NIR_DEBUG(CLONE)) {                                           \
      nir_shader *clone = nir_shader_clone(ralloc_parent(nir), nir); \
      nir_shader_replace(nir, clone);                                \
   }                                                                 \
   if (NIR_DEBUG(SERIALIZE)) {                                       \
      nir_shader_serialize_deserialize(nir);                         \
   }                                                                 \
} while (0)

#define NIR_PASS(progress, nir, pass, ...) _PASS(pass, nir,          \
   nir_metadata_set_validation_flag(nir);                            \
   if (should_print_nir(nir))                                        \
      printf("%s\n", #pass);                                         \
   if (pass(nir, ##__VA_ARGS__)) {                                   \
      nir_validate_shader(nir, "after " #pass " in " __FILE__);      \
      UNUSED bool _;                                                 \
      progress = true;                                               \
      if (should_print_nir(nir))                                     \
         nir_print_shader(nir, stdout);                              \
      nir_metadata_check_validation_flag(nir);                       \
   }                                                                 \
)

#define NIR_PASS_V(nir, pass, ...) _PASS(pass, nir,                  \
   if (should_print_nir(nir))                                        \
      printf("%s\n", #pass);                                         \
   pass(nir, ##__VA_ARGS__);                                         \
   nir_validate_shader(nir, "after " #pass " in " __FILE__);         \
   if (should_print_nir(nir))                                        \
      nir_print_shader(nir, stdout);                                 \
)

#define NIR_SKIP(name) should_skip_nir(#name)

/** An instruction filtering callback with writemask
 *
 * Returns true if the instruction should be processed with the associated
 * writemask and false otherwise.
 */
typedef bool (*nir_instr_writemask_filter_cb)(const nir_instr *,
                                              unsigned writemask, const void *);

/** A simple instruction lowering callback
 *
 * Many instruction lowering passes can be written as a simple function which
 * takes an instruction as its input and returns a sequence of instructions
 * that implement the consumed instruction.  This function type represents
 * such a lowering function.  When called, a function with this prototype
 * should either return NULL indicating that no lowering needs to be done or
 * emit a sequence of instructions using the provided builder (whose cursor
 * will already be placed after the instruction to be lowered) and return the
 * resulting nir_ssa_def.
 */
typedef nir_ssa_def *(*nir_lower_instr_cb)(struct nir_builder *,
                                           nir_instr *, void *);

/**
 * Special return value for nir_lower_instr_cb when some progress occurred
 * (like changing an input to the instr) that didn't result in a replacement
 * SSA def being generated.
 */
#define NIR_LOWER_INSTR_PROGRESS ((nir_ssa_def *)(uintptr_t)1)

/**
 * Special return value for nir_lower_instr_cb when some progress occurred
 * that should remove the current instruction that doesn't create an output
 * (like a store)
 */

#define NIR_LOWER_INSTR_PROGRESS_REPLACE ((nir_ssa_def *)(uintptr_t)2)

/** Iterate over all the instructions in a nir_function_impl and lower them
 *  using the provided callbacks
 *
 * This function implements the guts of a standard lowering pass for you.  It
 * iterates over all of the instructions in a nir_function_impl and calls the
 * filter callback on each one.  If the filter callback returns true, it then
 * calls the lowering call back on the instruction.  (Splitting it this way
 * allows us to avoid some save/restore work for instructions we know won't be
 * lowered.)  If the instruction is dead after the lowering is complete, it
 * will be removed.  If new instructions are added, the lowering callback will
 * also be called on them in case multiple lowerings are required.
 *
 * If the callback indicates that the original instruction is replaced (either
 * through a new SSA def or NIR_LOWER_INSTR_PROGRESS_REPLACE), then the
 * instruction is removed along with any now-dead SSA defs it used.
 *
 * The metadata for the nir_function_impl will also be updated.  If any blocks
 * are added (they cannot be removed), dominance and block indices will be
 * invalidated.
 */
bool nir_function_impl_lower_instructions(nir_function_impl *impl,
                                          nir_instr_filter_cb filter,
                                          nir_lower_instr_cb lower,
                                          void *cb_data);
bool nir_shader_lower_instructions(nir_shader *shader,
                                   nir_instr_filter_cb filter,
                                   nir_lower_instr_cb lower,
                                   void *cb_data);

void nir_calc_dominance_impl(nir_function_impl *impl);
void nir_calc_dominance(nir_shader *shader);

nir_block *nir_dominance_lca(nir_block *b1, nir_block *b2);
bool nir_block_dominates(nir_block *parent, nir_block *child);
bool nir_block_is_unreachable(nir_block *block);

void nir_dump_dom_tree_impl(nir_function_impl *impl, FILE *fp);
void nir_dump_dom_tree(nir_shader *shader, FILE *fp);

void nir_dump_dom_frontier_impl(nir_function_impl *impl, FILE *fp);
void nir_dump_dom_frontier(nir_shader *shader, FILE *fp);

void nir_dump_cfg_impl(nir_function_impl *impl, FILE *fp);
void nir_dump_cfg(nir_shader *shader, FILE *fp);

void nir_gs_count_vertices_and_primitives(const nir_shader *shader,
                                          int *out_vtxcnt,
                                          int *out_prmcnt,
                                          unsigned num_streams);

typedef enum {
   nir_group_all,
   nir_group_same_resource_only,
} nir_load_grouping;

void nir_group_loads(nir_shader *shader, nir_load_grouping grouping,
                     unsigned max_distance);

bool nir_shrink_vec_array_vars(nir_shader *shader, nir_variable_mode modes);
bool nir_split_array_vars(nir_shader *shader, nir_variable_mode modes);
bool nir_split_var_copies(nir_shader *shader);
bool nir_split_per_member_structs(nir_shader *shader);
bool nir_split_struct_vars(nir_shader *shader, nir_variable_mode modes);

bool nir_lower_returns_impl(nir_function_impl *impl);
bool nir_lower_returns(nir_shader *shader);

void nir_inline_function_impl(struct nir_builder *b,
                              const nir_function_impl *impl,
                              nir_ssa_def **params,
                              struct hash_table *shader_var_remap);
bool nir_inline_functions(nir_shader *shader);

void nir_find_inlinable_uniforms(nir_shader *shader);
void nir_inline_uniforms(nir_shader *shader, unsigned num_uniforms,
                         const uint32_t *uniform_values,
                         const uint16_t *uniform_dw_offsets);

bool nir_propagate_invariant(nir_shader *shader, bool invariant_prim);

void nir_lower_var_copy_instr(nir_intrinsic_instr *copy, nir_shader *shader);
void nir_lower_deref_copy_instr(struct nir_builder *b,
                                nir_intrinsic_instr *copy);
bool nir_lower_var_copies(nir_shader *shader);

bool nir_opt_memcpy(nir_shader *shader);
bool nir_lower_memcpy(nir_shader *shader);

void nir_fixup_deref_modes(nir_shader *shader);

bool nir_lower_global_vars_to_local(nir_shader *shader);

typedef enum {
   nir_lower_direct_array_deref_of_vec_load     = (1 << 0),
   nir_lower_indirect_array_deref_of_vec_load   = (1 << 1),
   nir_lower_direct_array_deref_of_vec_store    = (1 << 2),
   nir_lower_indirect_array_deref_of_vec_store  = (1 << 3),
} nir_lower_array_deref_of_vec_options;

bool nir_lower_array_deref_of_vec(nir_shader *shader, nir_variable_mode modes,
                                  nir_lower_array_deref_of_vec_options options);

bool nir_lower_indirect_derefs(nir_shader *shader, nir_variable_mode modes,
                               uint32_t max_lower_array_len);

bool nir_lower_indirect_var_derefs(nir_shader *shader,
                                   const struct set *vars);

bool nir_lower_locals_to_regs(nir_shader *shader);

void nir_lower_io_to_temporaries(nir_shader *shader,
                                 nir_function_impl *entrypoint,
                                 bool outputs, bool inputs);

bool nir_lower_vars_to_scratch(nir_shader *shader,
                               nir_variable_mode modes,
                               int size_threshold,
                               glsl_type_size_align_func size_align);

void nir_lower_clip_halfz(nir_shader *shader);

void nir_shader_gather_info(nir_shader *shader, nir_function_impl *entrypoint);

void nir_gather_ssa_types(nir_function_impl *impl,
                          BITSET_WORD *float_types,
                          BITSET_WORD *int_types);

void nir_assign_var_locations(nir_shader *shader, nir_variable_mode mode,
                              unsigned *size,
                              int (*type_size)(const struct glsl_type *, bool));

/* Some helpers to do very simple linking */
bool nir_remove_unused_varyings(nir_shader *producer, nir_shader *consumer);
bool nir_remove_unused_io_vars(nir_shader *shader, nir_variable_mode mode,
                               uint64_t *used_by_other_stage,
                               uint64_t *used_by_other_stage_patches);
void nir_compact_varyings(nir_shader *producer, nir_shader *consumer,
                          bool default_to_smooth_interp);
void nir_link_xfb_varyings(nir_shader *producer, nir_shader *consumer);
bool nir_link_opt_varyings(nir_shader *producer, nir_shader *consumer);
void nir_link_varying_precision(nir_shader *producer, nir_shader *consumer);

bool nir_slot_is_sysval_output(gl_varying_slot slot);
bool nir_slot_is_varying(gl_varying_slot slot);
bool nir_slot_is_sysval_output_and_varying(gl_varying_slot slot);
void nir_remove_varying(nir_intrinsic_instr *intr);
void nir_remove_sysval_output(nir_intrinsic_instr *intr);

bool nir_lower_amul(nir_shader *shader,
                    int (*type_size)(const struct glsl_type *, bool));

bool nir_lower_ubo_vec4(nir_shader *shader);

void nir_assign_io_var_locations(nir_shader *shader,
                                 nir_variable_mode mode,
                                 unsigned *size,
                                 gl_shader_stage stage);

typedef struct {
   uint8_t num_linked_io_vars;
   uint8_t num_linked_patch_io_vars;
} nir_linked_io_var_info;

nir_linked_io_var_info
nir_assign_linked_io_var_locations(nir_shader *producer,
                                   nir_shader *consumer);

typedef enum {
   /* If set, this causes all 64-bit IO operations to be lowered on-the-fly
    * to 32-bit operations.  This is only valid for nir_var_shader_in/out
    * modes.
    */
   nir_lower_io_lower_64bit_to_32 = (1 << 0),

   /* If set, this forces all non-flat fragment shader inputs to be
    * interpolated as if with the "sample" qualifier.  This requires
    * nir_shader_compiler_options::use_interpolated_input_intrinsics.
    */
   nir_lower_io_force_sample_interpolation = (1 << 1),
} nir_lower_io_options;
bool nir_lower_io(nir_shader *shader,
                  nir_variable_mode modes,
                  int (*type_size)(const struct glsl_type *, bool),
                  nir_lower_io_options);

bool nir_io_add_const_offset_to_base(nir_shader *nir, nir_variable_mode modes);

void
nir_lower_io_passes(nir_shader *nir);

bool nir_io_add_intrinsic_xfb_info(nir_shader *nir);

bool
nir_lower_vars_to_explicit_types(nir_shader *shader,
                                 nir_variable_mode modes,
                                 glsl_type_size_align_func type_info);
void
nir_gather_explicit_io_initializers(nir_shader *shader,
                                    void *dst, size_t dst_size,
                                    nir_variable_mode mode);

bool nir_lower_vec3_to_vec4(nir_shader *shader, nir_variable_mode modes);

typedef enum {
   /**
    * An address format which is a simple 32-bit global GPU address.
    */
   nir_address_format_32bit_global,

   /**
    * An address format which is a simple 64-bit global GPU address.
    */
   nir_address_format_64bit_global,

   /**
    * An address format which is a 64-bit global GPU address encoded as a
    * 2x32-bit vector.
    */
   nir_address_format_2x32bit_global,

   /**
    * An address format which is a 64-bit global base address and a 32-bit
    * offset.
    *
    * The address is comprised as a 32-bit vec4 where .xy are a uint64_t base
    * address stored with the low bits in .x and high bits in .y, .z is
    * undefined, and .w is an offset.  This is intended to match
    * 64bit_bounded_global but without the bounds checking.
    */
   nir_address_format_64bit_global_32bit_offset,

   /**
    * An address format which is a bounds-checked 64-bit global GPU address.
    *
    * The address is comprised as a 32-bit vec4 where .xy are a uint64_t base
    * address stored with the low bits in .x and high bits in .y, .z is a
    * size, and .w is an offset.  When the final I/O operation is lowered, .w
    * is checked against .z and the operation is predicated on the result.
    */
   nir_address_format_64bit_bounded_global,

   /**
    * An address format which is comprised of a vec2 where the first
    * component is a buffer index and the second is an offset.
    */
   nir_address_format_32bit_index_offset,

   /**
    * An address format which is a 64-bit value, where the high 32 bits
    * are a buffer index, and the low 32 bits are an offset.
    */
    nir_address_format_32bit_index_offset_pack64,

   /**
    * An address format which is comprised of a vec3 where the first two
    * components specify the buffer and the third is an offset.
    */
   nir_address_format_vec2_index_32bit_offset,

   /**
    * An address format which represents generic pointers with a 62-bit
    * pointer and a 2-bit enum in the top two bits.  The top two bits have
    * the following meanings:
    *
    *  - 0x0: Global memory
    *  - 0x1: Shared memory
    *  - 0x2: Scratch memory
    *  - 0x3: Global memory
    *
    * The redundancy between 0x0 and 0x3 is because of Intel sign-extension of
    * addresses.  Valid global memory addresses may naturally have either 0 or
    * ~0 as their high bits.
    *
    * Shared and scratch pointers are represented as 32-bit offsets with the
    * top 32 bits only being used for the enum.  This allows us to avoid
    * 64-bit address calculations in a bunch of cases.
    */
   nir_address_format_62bit_generic,

   /**
    * An address format which is a simple 32-bit offset.
    */
   nir_address_format_32bit_offset,

   /**
    * An address format which is a simple 32-bit offset cast to 64-bit.
    */
    nir_address_format_32bit_offset_as_64bit,

   /**
    * An address format representing a purely logical addressing model.  In
    * this model, all deref chains must be complete from the dereference
    * operation to the variable.  Cast derefs are not allowed.  These
    * addresses will be 32-bit scalars but the format is immaterial because
    * you can always chase the chain.
    */
   nir_address_format_logical,
} nir_address_format;

unsigned
nir_address_format_bit_size(nir_address_format addr_format);

unsigned
nir_address_format_num_components(nir_address_format addr_format);

static inline const struct glsl_type *
nir_address_format_to_glsl_type(nir_address_format addr_format)
{
   unsigned bit_size = nir_address_format_bit_size(addr_format);
   assert(bit_size == 32 || bit_size == 64);
   return glsl_vector_type(bit_size == 32 ? GLSL_TYPE_UINT : GLSL_TYPE_UINT64,
                           nir_address_format_num_components(addr_format));
}

const nir_const_value *nir_address_format_null_value(nir_address_format addr_format);

nir_ssa_def *nir_build_addr_ieq(struct nir_builder *b, nir_ssa_def *addr0, nir_ssa_def *addr1,
                                nir_address_format addr_format);

nir_ssa_def *nir_build_addr_isub(struct nir_builder *b, nir_ssa_def *addr0, nir_ssa_def *addr1,
                                 nir_address_format addr_format);

nir_ssa_def * nir_explicit_io_address_from_deref(struct nir_builder *b,
                                                 nir_deref_instr *deref,
                                                 nir_ssa_def *base_addr,
                                                 nir_address_format addr_format);

bool nir_get_explicit_deref_align(nir_deref_instr *deref,
                                  bool default_to_type_align,
                                  uint32_t *align_mul,
                                  uint32_t *align_offset);

void nir_lower_explicit_io_instr(struct nir_builder *b,
                                 nir_intrinsic_instr *io_instr,
                                 nir_ssa_def *addr,
                                 nir_address_format addr_format);

bool nir_lower_explicit_io(nir_shader *shader,
                           nir_variable_mode modes,
                           nir_address_format);

typedef bool (*nir_should_vectorize_mem_func)(unsigned align_mul,
                                              unsigned align_offset,
                                              unsigned bit_size,
                                              unsigned num_components,
                                              nir_intrinsic_instr *low, nir_intrinsic_instr *high,
                                              void *data);

typedef struct {
   nir_should_vectorize_mem_func callback;
   nir_variable_mode modes;
   nir_variable_mode robust_modes;
   void *cb_data;
   bool has_shared2_amd;
} nir_load_store_vectorize_options;

bool nir_opt_load_store_vectorize(nir_shader *shader, const nir_load_store_vectorize_options *options);

typedef struct nir_lower_shader_calls_options {
   /* Address format used for load/store operations on the call stack. */
   nir_address_format address_format;

   /* Stack alignment */
   unsigned stack_alignment;

   /* Put loads from the stack as close as possible from where they're needed.
    * You might want to disable combined_loads for best effects.
    */
   bool localized_loads;

   /* If this function pointer is not NULL, lower_shader_calls will run
    * nir_opt_load_store_vectorize for stack load/store operations. Otherwise
    * the optimizaion is not run.
    */
   nir_should_vectorize_mem_func vectorizer_callback;

   /* Data passed to vectorizer_callback */
   void *vectorizer_data;
} nir_lower_shader_calls_options;

bool
nir_lower_shader_calls(nir_shader *shader,
                       const nir_lower_shader_calls_options *options,
                       nir_shader ***resume_shaders_out,
                       uint32_t *num_resume_shaders_out,
                       void *mem_ctx);

nir_src *nir_get_io_offset_src(nir_intrinsic_instr *instr);
nir_src *nir_get_io_arrayed_index_src(nir_intrinsic_instr *instr);
nir_src *nir_get_shader_call_payload_src(nir_intrinsic_instr *call);

bool nir_is_arrayed_io(const nir_variable *var, gl_shader_stage stage);

bool nir_lower_regs_to_ssa_impl(nir_function_impl *impl);
bool nir_lower_regs_to_ssa(nir_shader *shader);
bool nir_lower_vars_to_ssa(nir_shader *shader);

bool nir_remove_dead_derefs(nir_shader *shader);
bool nir_remove_dead_derefs_impl(nir_function_impl *impl);

typedef struct nir_remove_dead_variables_options {
   bool (*can_remove_var)(nir_variable *var, void *data);
   void *can_remove_var_data;
} nir_remove_dead_variables_options;

bool nir_remove_dead_variables(nir_shader *shader, nir_variable_mode modes,
                               const nir_remove_dead_variables_options *options);

bool nir_lower_variable_initializers(nir_shader *shader,
                                     nir_variable_mode modes);
bool nir_zero_initialize_shared_memory(nir_shader *shader,
                                       const unsigned shared_size,
                                       const unsigned chunk_size);

bool nir_move_vec_src_uses_to_dest(nir_shader *shader);
bool nir_lower_vec_to_movs(nir_shader *shader, nir_instr_writemask_filter_cb cb,
                           const void *_data);
void nir_lower_alpha_test(nir_shader *shader, enum compare_func func,
                          bool alpha_to_one,
                          const gl_state_index16 *alpha_ref_state_tokens);
bool nir_lower_alu(nir_shader *shader);

bool nir_lower_flrp(nir_shader *shader, unsigned lowering_mask,
                    bool always_precise);

bool nir_scale_fdiv(nir_shader *shader);

bool nir_lower_alu_to_scalar(nir_shader *shader, nir_instr_filter_cb cb, const void *data);
bool nir_lower_alu_width(nir_shader *shader, nir_vectorize_cb cb, const void *data);
bool nir_lower_bool_to_bitsize(nir_shader *shader);
bool nir_lower_bool_to_float(nir_shader *shader, bool has_fcsel_ne);
bool nir_lower_bool_to_int32(nir_shader *shader);
bool nir_opt_simplify_convert_alu_types(nir_shader *shader);
bool nir_lower_const_arrays_to_uniforms(nir_shader *shader,
                                        unsigned max_uniform_components);
bool nir_lower_convert_alu_types(nir_shader *shader,
                                 bool (*should_lower)(nir_intrinsic_instr *));
bool nir_lower_constant_convert_alu_types(nir_shader *shader);
bool nir_lower_alu_conversion_to_intrinsic(nir_shader *shader);
bool nir_lower_int_to_float(nir_shader *shader);
bool nir_lower_load_const_to_scalar(nir_shader *shader);
bool nir_lower_read_invocation_to_scalar(nir_shader *shader);
bool nir_lower_phis_to_scalar(nir_shader *shader, bool lower_all);
void nir_lower_io_arrays_to_elements(nir_shader *producer, nir_shader *consumer);
void nir_lower_io_arrays_to_elements_no_indirects(nir_shader *shader,
                                                  bool outputs_only);
void nir_lower_io_to_scalar(nir_shader *shader, nir_variable_mode mask);
bool nir_lower_io_to_scalar_early(nir_shader *shader, nir_variable_mode mask);
bool nir_lower_io_to_vector(nir_shader *shader, nir_variable_mode mask);
bool nir_vectorize_tess_levels(nir_shader *shader);
nir_shader * nir_create_passthrough_tcs_impl(const nir_shader_compiler_options *options,
                                             unsigned *locations, unsigned num_locations,
                                             uint8_t patch_vertices);
nir_shader * nir_create_passthrough_tcs(const nir_shader_compiler_options *options,
                                        const nir_shader *vs, uint8_t patch_vertices);
nir_shader * nir_create_passthrough_gs(const nir_shader_compiler_options *options,
                                       const nir_shader *prev_stage,
                                       enum shader_prim primitive_type,
                                       unsigned vertices);

bool nir_lower_fragcolor(nir_shader *shader, unsigned max_cbufs);
bool nir_lower_fragcoord_wtrans(nir_shader *shader);
void nir_lower_viewport_transform(nir_shader *shader);
bool nir_lower_uniforms_to_ubo(nir_shader *shader, bool dword_packed, bool load_vec4);

bool nir_lower_is_helper_invocation(nir_shader *shader);

bool nir_lower_single_sampled(nir_shader *shader);

typedef struct nir_lower_subgroups_options {
   uint8_t subgroup_size;
   uint8_t ballot_bit_size;
   uint8_t ballot_components;
   bool lower_to_scalar:1;
   bool lower_vote_trivial:1;
   bool lower_vote_eq:1;
   bool lower_subgroup_masks:1;
   bool lower_relative_shuffle:1;
   bool lower_shuffle_to_32bit:1;
   bool lower_shuffle_to_swizzle_amd:1;
   bool lower_shuffle:1;
   bool lower_quad:1;
   bool lower_quad_broadcast_dynamic:1;
   bool lower_quad_broadcast_dynamic_to_const:1;
   bool lower_elect:1;
   bool lower_read_invocation_to_cond:1;
} nir_lower_subgroups_options;

bool nir_lower_subgroups(nir_shader *shader,
                         const nir_lower_subgroups_options *options);

bool nir_lower_system_values(nir_shader *shader);

nir_ssa_def *
nir_build_lowered_load_helper_invocation(struct nir_builder *b);

typedef struct nir_lower_compute_system_values_options {
   bool has_base_global_invocation_id:1;
   bool has_base_workgroup_id:1;
   bool shuffle_local_ids_for_quad_derivatives:1;
   bool lower_local_invocation_index:1;
   bool lower_cs_local_id_to_index:1;
   bool lower_workgroup_id_to_index:1;
} nir_lower_compute_system_values_options;

bool nir_lower_compute_system_values(nir_shader *shader,
                                     const nir_lower_compute_system_values_options *options);

struct nir_lower_sysvals_to_varyings_options {
   bool frag_coord:1;
   bool front_face:1;
   bool point_coord:1;
};

bool
nir_lower_sysvals_to_varyings(nir_shader *shader,
                              const struct nir_lower_sysvals_to_varyings_options *options);

enum PACKED nir_lower_tex_packing {
   /** No packing */
   nir_lower_tex_packing_none = 0,
   /**
    * The sampler returns up to 2 32-bit words of half floats or 16-bit signed
    * or unsigned ints based on the sampler type
    */
   nir_lower_tex_packing_16,
   /** The sampler returns 1 32-bit word of 4x8 unorm */
   nir_lower_tex_packing_8,
};

typedef struct nir_lower_tex_options {
   /**
    * bitmask of (1 << GLSL_SAMPLER_DIM_x) to control for which
    * sampler types a texture projector is lowered.
    */
   unsigned lower_txp;

   /**
    * If true, lower texture projector for any array sampler dims
    */
   bool lower_txp_array;

   /**
    * If true, lower away nir_tex_src_offset for all texelfetch instructions.
    */
   bool lower_txf_offset;

   /**
    * If true, lower away nir_tex_src_offset for all rect textures.
    */
   bool lower_rect_offset;

   /**
    * If not NULL, this filter will return true for tex instructions that
    * should lower away nir_tex_src_offset.
    */
   nir_instr_filter_cb lower_offset_filter;

   /**
    * If true, lower rect textures to 2D, using txs to fetch the
    * texture dimensions and dividing the texture coords by the
    * texture dims to normalize.
    */
   bool lower_rect;

   /**
    * If true, convert yuv to rgb.
    */
   unsigned lower_y_uv_external;
   unsigned lower_y_u_v_external;
   unsigned lower_yx_xuxv_external;
   unsigned lower_xy_uxvx_external;
   unsigned lower_ayuv_external;
   unsigned lower_xyuv_external;
   unsigned lower_yuv_external;
   unsigned lower_yu_yv_external;
   unsigned lower_y41x_external;
   unsigned bt709_external;
   unsigned bt2020_external;
   unsigned yuv_full_range_external;

   /**
    * To emulate certain texture wrap modes, this can be used
    * to saturate the specified tex coord to [0.0, 1.0].  The
    * bits are according to sampler #, ie. if, for example:
    *
    *   (conf->saturate_s & (1 << n))
    *
    * is true, then the s coord for sampler n is saturated.
    *
    * Note that clamping must happen *after* projector lowering
    * so any projected texture sample instruction with a clamped
    * coordinate gets automatically lowered, regardless of the
    * 'lower_txp' setting.
    */
   unsigned saturate_s;
   unsigned saturate_t;
   unsigned saturate_r;

   /* Bitmask of textures that need swizzling.
    *
    * If (swizzle_result & (1 << texture_index)), then the swizzle in
    * swizzles[texture_index] is applied to the result of the texturing
    * operation.
    */
   unsigned swizzle_result;

   /* A swizzle for each texture.  Values 0-3 represent x, y, z, or w swizzles
    * while 4 and 5 represent 0 and 1 respectively.
    *
    * Indexed by texture-id.
    */
   uint8_t swizzles[32][4];

   /* Can be used to scale sampled values in range required by the
    * format.
    *
    * Indexed by texture-id.
    */
   float scale_factors[32];

   /**
    * Bitmap of textures that need srgb to linear conversion.  If
    * (lower_srgb & (1 << texture_index)) then the rgb (xyz) components
    * of the texture are lowered to linear.
    */
   unsigned lower_srgb;

   /**
    * If true, lower nir_texop_txd on cube maps with nir_texop_txl.
    */
   bool lower_txd_cube_map;

   /**
    * If true, lower nir_texop_txd on 3D surfaces with nir_texop_txl.
    */
   bool lower_txd_3d;

   /**
    * If true, lower nir_texop_txd any array surfaces with nir_texop_txl.
    */
   bool lower_txd_array;

   /**
    * If true, lower nir_texop_txd on shadow samplers (except cube maps)
    * with nir_texop_txl. Notice that cube map shadow samplers are lowered
    * with lower_txd_cube_map.
    */
   bool lower_txd_shadow;

   /**
    * If true, lower nir_texop_txd on all samplers to a nir_texop_txl.
    * Implies lower_txd_cube_map and lower_txd_shadow.
    */
   bool lower_txd;

   /**
    * If true, lower nir_texop_txb that try to use shadow compare and min_lod
    * at the same time to a nir_texop_lod, some math, and nir_texop_tex.
    */
   bool lower_txb_shadow_clamp;

   /**
    * If true, lower nir_texop_txd on shadow samplers when it uses min_lod
    * with nir_texop_txl.  This includes cube maps.
    */
   bool lower_txd_shadow_clamp;

   /**
    * If true, lower nir_texop_txd on when it uses both offset and min_lod
    * with nir_texop_txl.  This includes cube maps.
    */
   bool lower_txd_offset_clamp;

   /**
    * If true, lower nir_texop_txd with min_lod to a nir_texop_txl if the
    * sampler is bindless.
    */
   bool lower_txd_clamp_bindless_sampler;

   /**
    * If true, lower nir_texop_txd with min_lod to a nir_texop_txl if the
    * sampler index is not statically determinable to be less than 16.
    */
   bool lower_txd_clamp_if_sampler_index_not_lt_16;

   /**
    * If true, lower nir_texop_txs with a non-0-lod into nir_texop_txs with
    * 0-lod followed by a nir_ishr.
    */
   bool lower_txs_lod;

   /**
    * If true, lower nir_texop_txs for cube arrays to a nir_texop_txs with a
    * 2D array type followed by a nir_idiv by 6.
    */
   bool lower_txs_cube_array;

   /**
    * If true, apply a .bagr swizzle on tg4 results to handle Broadcom's
    * mixed-up tg4 locations.
    */
   bool lower_tg4_broadcom_swizzle;

   /**
    * If true, lowers tg4 with 4 constant offsets to 4 tg4 calls
    */
   bool lower_tg4_offsets;

   /**
    * Lower txf_ms to fragment_mask_fetch and fragment_fetch and samples_identical to
    * fragment_mask_fetch.
    */
   bool lower_to_fragment_fetch_amd;

   /**
    * To lower packed sampler return formats.
    *
    * Indexed by sampler-id.
    */
   enum nir_lower_tex_packing lower_tex_packing[32];

   /**
    * If true, lower nir_texop_lod to return -FLT_MAX if the sum of the
    * absolute values of derivatives is 0 for all coordinates.
    */
   bool lower_lod_zero_width;

   /* Turns nir_op_tex and other ops with an implicit derivative, in stages
    * without implicit derivatives (like the vertex shader) to have an explicit
    * LOD with a value of 0.
    */
   bool lower_invalid_implicit_lod;

   /* If true, round the layer component of the coordinates source to the nearest
    * integer for all array ops.
    */
   bool lower_array_layer_round_even;

   /**
    * Payload data to be sent to callback / filter functions.
    */
   void *callback_data;
} nir_lower_tex_options;

/** Lowers complex texture instructions to simpler ones */
bool nir_lower_tex(nir_shader *shader,
                   const nir_lower_tex_options *options);


typedef struct nir_lower_tex_shadow_swizzle {
   unsigned swizzle_r:3;
   unsigned swizzle_g:3;
   unsigned swizzle_b:3;
   unsigned swizzle_a:3;
} nir_lower_tex_shadow_swizzle;

bool
nir_lower_tex_shadow(nir_shader *s,
                     unsigned n_states,
                     enum compare_func *compare_func,
                     nir_lower_tex_shadow_swizzle *tex_swizzles);

typedef struct nir_lower_image_options {
   /**
    * If true, lower cube size operations.
    */
   bool lower_cube_size;

   /**
    * Lower multi sample image load and samples_identical to use fragment_mask_load.
    */
   bool lower_to_fragment_mask_load_amd;
} nir_lower_image_options;

bool nir_lower_image(nir_shader *nir,
                     const nir_lower_image_options *options);

bool nir_lower_readonly_images_to_tex(nir_shader *shader, bool per_variable);

enum nir_lower_non_uniform_access_type {
   nir_lower_non_uniform_ubo_access     = (1 << 0),
   nir_lower_non_uniform_ssbo_access    = (1 << 1),
   nir_lower_non_uniform_texture_access = (1 << 2),
   nir_lower_non_uniform_image_access   = (1 << 3),
};

/* Given the nir_src used for the resource, return the channels which might be non-uniform. */
typedef nir_component_mask_t (*nir_lower_non_uniform_access_callback)(const nir_src *, void *);

typedef struct nir_lower_non_uniform_access_options {
   enum nir_lower_non_uniform_access_type types;
   nir_lower_non_uniform_access_callback callback;
   void *callback_data;
} nir_lower_non_uniform_access_options;

bool nir_has_non_uniform_access(nir_shader *shader, enum nir_lower_non_uniform_access_type types);
bool nir_opt_non_uniform_access(nir_shader *shader);
bool nir_lower_non_uniform_access(nir_shader *shader,
                                  const nir_lower_non_uniform_access_options *options);

typedef struct {
   /* Whether 16-bit floating point arithmetic should be allowed in 8-bit
    * division lowering
    */
   bool allow_fp16;
} nir_lower_idiv_options;

bool nir_lower_idiv(nir_shader *shader, const nir_lower_idiv_options *options);

typedef struct nir_input_attachment_options {
   bool use_fragcoord_sysval;
   bool use_layer_id_sysval;
   bool use_view_id_for_layer;
} nir_input_attachment_options;

bool nir_lower_input_attachments(nir_shader *shader,
                                 const nir_input_attachment_options *options);

bool nir_lower_clip_vs(nir_shader *shader, unsigned ucp_enables,
                       bool use_vars,
                       bool use_clipdist_array,
                       const gl_state_index16 clipplane_state_tokens[][STATE_LENGTH]);
bool nir_lower_clip_gs(nir_shader *shader, unsigned ucp_enables,
                       bool use_clipdist_array,
                       const gl_state_index16 clipplane_state_tokens[][STATE_LENGTH]);
bool nir_lower_clip_fs(nir_shader *shader, unsigned ucp_enables,
                       bool use_clipdist_array);
bool nir_lower_clip_cull_distance_arrays(nir_shader *nir);
bool nir_lower_clip_disable(nir_shader *shader, unsigned clip_plane_enable);

void nir_lower_point_size_mov(nir_shader *shader,
                              const gl_state_index16 *pointsize_state_tokens);

bool nir_lower_frexp(nir_shader *nir);

void nir_lower_two_sided_color(nir_shader *shader, bool face_sysval);

bool nir_lower_clamp_color_outputs(nir_shader *shader);

bool nir_lower_flatshade(nir_shader *shader);

void nir_lower_passthrough_edgeflags(nir_shader *shader);
bool nir_lower_patch_vertices(nir_shader *nir, unsigned static_count,
                              const gl_state_index16 *uniform_state_tokens);

typedef struct nir_lower_wpos_ytransform_options {
   gl_state_index16 state_tokens[STATE_LENGTH];
   bool fs_coord_origin_upper_left :1;
   bool fs_coord_origin_lower_left :1;
   bool fs_coord_pixel_center_integer :1;
   bool fs_coord_pixel_center_half_integer :1;
} nir_lower_wpos_ytransform_options;

bool nir_lower_wpos_ytransform(nir_shader *shader,
                               const nir_lower_wpos_ytransform_options *options);
bool nir_lower_wpos_center(nir_shader *shader);

bool nir_lower_pntc_ytransform(nir_shader *shader,
                               const gl_state_index16 clipplane_state_tokens[][STATE_LENGTH]);

bool nir_lower_wrmasks(nir_shader *shader, nir_instr_filter_cb cb, const void *data);

bool nir_lower_fb_read(nir_shader *shader);

typedef struct nir_lower_drawpixels_options {
   gl_state_index16 texcoord_state_tokens[STATE_LENGTH];
   gl_state_index16 scale_state_tokens[STATE_LENGTH];
   gl_state_index16 bias_state_tokens[STATE_LENGTH];
   unsigned drawpix_sampler;
   unsigned pixelmap_sampler;
   bool pixel_maps :1;
   bool scale_and_bias :1;
} nir_lower_drawpixels_options;

void nir_lower_drawpixels(nir_shader *shader,
                          const nir_lower_drawpixels_options *options);

typedef struct nir_lower_bitmap_options {
   unsigned sampler;
   bool swizzle_xxxx;
} nir_lower_bitmap_options;

void nir_lower_bitmap(nir_shader *shader, const nir_lower_bitmap_options *options);

bool nir_lower_atomics_to_ssbo(nir_shader *shader, unsigned offset_align_state);

typedef enum  {
   nir_lower_int_source_mods = 1 << 0,
   nir_lower_fabs_source_mods = 1 << 1,
   nir_lower_fneg_source_mods = 1 << 2,
   nir_lower_64bit_source_mods = 1 << 3,
   nir_lower_triop_abs = 1 << 4,
   nir_lower_all_source_mods = (1 << 5) - 1
} nir_lower_to_source_mods_flags;

#define nir_lower_float_source_mods (nir_lower_fabs_source_mods | nir_lower_fneg_source_mods)

bool nir_lower_to_source_mods(nir_shader *shader, nir_lower_to_source_mods_flags options);

typedef enum {
   nir_lower_gs_intrinsics_per_stream = 1 << 0,
   nir_lower_gs_intrinsics_count_primitives = 1 << 1,
   nir_lower_gs_intrinsics_count_vertices_per_primitive = 1 << 2,
   nir_lower_gs_intrinsics_overwrite_incomplete = 1 << 3,
} nir_lower_gs_intrinsics_flags;

bool nir_lower_gs_intrinsics(nir_shader *shader, nir_lower_gs_intrinsics_flags options);

typedef struct {
   bool payload_to_shared_for_atomics : 1;
   bool payload_to_shared_for_small_types : 1;
   uint32_t payload_offset_in_bytes;
} nir_lower_task_shader_options;

bool nir_lower_task_shader(nir_shader *shader, nir_lower_task_shader_options options);

typedef unsigned (*nir_lower_bit_size_callback)(const nir_instr *, void *);

bool nir_lower_bit_size(nir_shader *shader,
                        nir_lower_bit_size_callback callback,
                        void *callback_data);
bool nir_lower_64bit_phis(nir_shader *shader);

bool nir_split_64bit_vec3_and_vec4(nir_shader *shader);

nir_lower_int64_options nir_lower_int64_op_to_options_mask(nir_op opcode);
bool nir_lower_int64(nir_shader *shader);

nir_lower_doubles_options nir_lower_doubles_op_to_options_mask(nir_op opcode);
bool nir_lower_doubles(nir_shader *shader, const nir_shader *softfp64,
                       nir_lower_doubles_options options);
bool nir_lower_pack(nir_shader *shader);

bool nir_recompute_io_bases(nir_shader *nir, nir_variable_mode modes);
bool nir_lower_mediump_vars(nir_shader *nir, nir_variable_mode modes);
bool nir_lower_mediump_io(nir_shader *nir, nir_variable_mode modes,
                          uint64_t varying_mask, bool use_16bit_slots);
bool nir_force_mediump_io(nir_shader *nir, nir_variable_mode modes,
                          nir_alu_type types);
bool nir_unpack_16bit_varying_slots(nir_shader *nir, nir_variable_mode modes);

struct nir_fold_tex_srcs_options {
   unsigned sampler_dims;
   unsigned src_types;
};

struct nir_fold_16bit_tex_image_options {
   nir_rounding_mode rounding_mode;
   nir_alu_type fold_tex_dest_types;
   bool fold_image_load_store_data;
   bool fold_image_srcs;
   unsigned fold_srcs_options_count;
   struct nir_fold_tex_srcs_options *fold_srcs_options;
};

bool nir_fold_16bit_tex_image(nir_shader *nir,
                              struct nir_fold_16bit_tex_image_options *options);

typedef struct {
   bool legalize_type;         /* whether this src should be legalized */
   uint8_t bit_size;           /* bit_size to enforce */
   nir_tex_src_type match_src; /* if bit_size is 0, match bit size of this */
} nir_tex_src_type_constraint, nir_tex_src_type_constraints[nir_num_tex_src_types];

bool nir_legalize_16bit_sampler_srcs(nir_shader *nir,
                                     nir_tex_src_type_constraints constraints);

bool nir_lower_point_size(nir_shader *shader, float min, float max);

void nir_lower_texcoord_replace(nir_shader *s, unsigned coord_replace,
                                bool point_coord_is_sysval, bool yinvert);

typedef enum {
   nir_lower_interpolation_at_sample = (1 << 1),
   nir_lower_interpolation_at_offset = (1 << 2),
   nir_lower_interpolation_centroid  = (1 << 3),
   nir_lower_interpolation_pixel     = (1 << 4),
   nir_lower_interpolation_sample    = (1 << 5),
} nir_lower_interpolation_options;

bool nir_lower_interpolation(nir_shader *shader,
                             nir_lower_interpolation_options options);

typedef enum {
   nir_lower_discard_if_to_cf = (1 << 0),
   nir_lower_demote_if_to_cf = (1 << 1),
   nir_lower_terminate_if_to_cf = (1 << 2),
} nir_lower_discard_if_options;

bool nir_lower_discard_if(nir_shader *shader, nir_lower_discard_if_options options);

bool nir_lower_discard_or_demote(nir_shader *shader,
                                 bool force_correct_quad_ops_after_discard);

bool nir_lower_memory_model(nir_shader *shader);

bool nir_lower_goto_ifs(nir_shader *shader);

bool nir_shader_uses_view_index(nir_shader *shader);
bool nir_can_lower_multiview(nir_shader *shader);
bool nir_lower_multiview(nir_shader *shader, uint32_t view_mask);


bool nir_lower_fp16_casts(nir_shader *shader);
bool nir_normalize_cubemap_coords(nir_shader *shader);

bool nir_shader_supports_implicit_lod(nir_shader *shader);

void nir_live_ssa_defs_impl(nir_function_impl *impl);

const BITSET_WORD *nir_get_live_ssa_defs(nir_cursor cursor, void *mem_ctx);

void nir_loop_analyze_impl(nir_function_impl *impl,
                           nir_variable_mode indirect_mask,
                           bool force_unroll_sampler_indirect);

bool nir_ssa_defs_interfere(nir_ssa_def *a, nir_ssa_def *b);

bool nir_repair_ssa_impl(nir_function_impl *impl);
bool nir_repair_ssa(nir_shader *shader);

void nir_convert_loop_to_lcssa(nir_loop *loop);
bool nir_convert_to_lcssa(nir_shader *shader, bool skip_invariants, bool skip_bool_invariants);
void nir_divergence_analysis(nir_shader *shader);
bool nir_update_instr_divergence(nir_shader *shader, nir_instr *instr);
bool nir_has_divergent_loop(nir_shader *shader);

/* If phi_webs_only is true, only convert SSA values involved in phi nodes to
 * registers.  If false, convert all values (even those not involved in a phi
 * node) to registers.
 */
bool nir_convert_from_ssa(nir_shader *shader, bool phi_webs_only);

bool nir_lower_phis_to_regs_block(nir_block *block);
bool nir_lower_ssa_defs_to_regs_block(nir_block *block);
bool nir_rematerialize_derefs_in_use_blocks_impl(nir_function_impl *impl);

bool nir_lower_samplers(nir_shader *shader);
bool nir_lower_cl_images(nir_shader *shader, bool lower_image_derefs, bool lower_sampler_derefs);
bool nir_dedup_inline_samplers(nir_shader *shader);
bool nir_lower_ssbo(nir_shader *shader);

typedef struct nir_lower_printf_options {
   bool treat_doubles_as_floats : 1;
   unsigned max_buffer_size;
} nir_lower_printf_options;

bool nir_lower_printf(nir_shader *nir, const nir_lower_printf_options *options);

/* This is here for unit tests. */
bool nir_opt_comparison_pre_impl(nir_function_impl *impl);

bool nir_opt_comparison_pre(nir_shader *shader);

typedef struct nir_opt_access_options {
   bool is_vulkan;
} nir_opt_access_options;

bool nir_opt_access(nir_shader *shader, const nir_opt_access_options *options);
bool nir_opt_algebraic(nir_shader *shader);
bool nir_opt_algebraic_before_ffma(nir_shader *shader);
bool nir_opt_algebraic_late(nir_shader *shader);
bool nir_opt_algebraic_distribute_src_mods(nir_shader *shader);
bool nir_opt_constant_folding(nir_shader *shader);

/* Try to combine a and b into a.  Return true if combination was possible,
 * which will result in b being removed by the pass.  Return false if
 * combination wasn't possible.
 */
typedef bool (*nir_combine_memory_barrier_cb)(
   nir_intrinsic_instr *a, nir_intrinsic_instr *b, void *data);

bool nir_opt_combine_memory_barriers(nir_shader *shader,
                                     nir_combine_memory_barrier_cb combine_cb,
                                     void *data);

bool nir_opt_combine_stores(nir_shader *shader, nir_variable_mode modes);

bool nir_copy_prop_impl(nir_function_impl *impl);
bool nir_copy_prop(nir_shader *shader);

bool nir_opt_copy_prop_vars(nir_shader *shader);

bool nir_opt_cse(nir_shader *shader);

bool nir_opt_dce(nir_shader *shader);

bool nir_opt_dead_cf(nir_shader *shader);

bool nir_opt_dead_write_vars(nir_shader *shader);

bool nir_opt_deref_impl(nir_function_impl *impl);
bool nir_opt_deref(nir_shader *shader);

bool nir_opt_find_array_copies(nir_shader *shader);

bool nir_opt_fragdepth(nir_shader *shader);

bool nir_opt_gcm(nir_shader *shader, bool value_number);

bool nir_opt_idiv_const(nir_shader *shader, unsigned min_bit_size);

typedef enum {
   nir_opt_if_aggressive_last_continue = (1 << 0),
   nir_opt_if_optimize_phi_true_false = (1 << 1),
} nir_opt_if_options;

bool nir_opt_if(nir_shader *shader, nir_opt_if_options options);

bool nir_opt_intrinsics(nir_shader *shader);

bool nir_opt_large_constants(nir_shader *shader,
                             glsl_type_size_align_func size_align,
                             unsigned threshold);

bool nir_opt_loop_unroll(nir_shader *shader);

typedef enum {
    nir_move_const_undef  = (1 << 0),
    nir_move_load_ubo     = (1 << 1),
    nir_move_load_input   = (1 << 2),
    nir_move_comparisons  = (1 << 3),
    nir_move_copies       = (1 << 4),
    nir_move_load_ssbo    = (1 << 5),
    nir_move_load_uniform = (1 << 6),
} nir_move_options;

bool nir_can_move_instr(nir_instr *instr, nir_move_options options);

bool nir_opt_sink(nir_shader *shader, nir_move_options options);

bool nir_opt_move(nir_shader *shader, nir_move_options options);

typedef struct {
   /** nir_load_uniform max base offset */
   uint32_t uniform_max;

   /** nir_load_ubo_vec4 max base offset */
   uint32_t ubo_vec4_max;

   /** nir_var_mem_shared max base offset */
   uint32_t shared_max;

   /** nir_load/store_buffer_amd max base offset */
   uint32_t buffer_max;
} nir_opt_offsets_options;

bool nir_opt_offsets(nir_shader *shader, const nir_opt_offsets_options *options);

bool nir_opt_peephole_select(nir_shader *shader, unsigned limit,
                             bool indirect_load_ok, bool expensive_alu_ok);

bool nir_opt_rematerialize_compares(nir_shader *shader);

bool nir_opt_remove_phis(nir_shader *shader);
bool nir_opt_remove_phis_block(nir_block *block);

bool nir_opt_phi_precision(nir_shader *shader);

bool nir_opt_shrink_stores(nir_shader *shader, bool shrink_image_store);

bool nir_opt_shrink_vectors(nir_shader *shader);

bool nir_opt_trivial_continues(nir_shader *shader);

bool nir_opt_undef(nir_shader *shader);

bool nir_lower_undef_to_zero(nir_shader *shader);

bool nir_opt_uniform_atomics(nir_shader *shader);

bool nir_opt_vectorize(nir_shader *shader, nir_vectorize_cb filter,
                       void *data);

bool nir_opt_conditional_discard(nir_shader *shader);
bool nir_opt_move_discards_to_top(nir_shader *shader);

bool nir_opt_ray_queries(nir_shader *shader);

bool nir_opt_ray_query_ranges(nir_shader *shader);

void nir_sweep(nir_shader *shader);

void nir_remap_dual_slot_attributes(nir_shader *shader,
                                    uint64_t *dual_slot_inputs);
uint64_t nir_get_single_slot_attribs_mask(uint64_t attribs, uint64_t dual_slot);

nir_intrinsic_op nir_intrinsic_from_system_value(gl_system_value val);
gl_system_value nir_system_value_from_intrinsic(nir_intrinsic_op intrin);

static inline bool
nir_variable_is_in_ubo(const nir_variable *var)
{
   return (var->data.mode == nir_var_mem_ubo &&
           var->interface_type != NULL);
}

static inline bool
nir_variable_is_in_ssbo(const nir_variable *var)
{
   return (var->data.mode == nir_var_mem_ssbo &&
           var->interface_type != NULL);
}

static inline bool
nir_variable_is_in_block(const nir_variable *var)
{
   return nir_variable_is_in_ubo(var) || nir_variable_is_in_ssbo(var);
}

/* See default_ub_config in nir_range_analysis.c for documentation. */
typedef struct nir_unsigned_upper_bound_config {
   unsigned min_subgroup_size;
   unsigned max_subgroup_size;
   unsigned max_workgroup_invocations;
   unsigned max_workgroup_count[3];
   unsigned max_workgroup_size[3];

   uint32_t vertex_attrib_max[32];
} nir_unsigned_upper_bound_config;

uint32_t
nir_unsigned_upper_bound(nir_shader *shader, struct hash_table *range_ht,
                         nir_ssa_scalar scalar,
                         const nir_unsigned_upper_bound_config *config);

bool
nir_addition_might_overflow(nir_shader *shader, struct hash_table *range_ht,
                            nir_ssa_scalar ssa, unsigned const_val,
                            const nir_unsigned_upper_bound_config *config);

typedef struct {
   /* True if gl_DrawID is considered uniform, i.e. if the preamble is run
    * at least once per "internal" draw rather than per user-visible draw.
    */
   bool drawid_uniform;

   /* True if the subgroup size is uniform. */
   bool subgroup_size_uniform;

   /* size/align for load/store_preamble. */
   void (*def_size)(nir_ssa_def *def, unsigned *size, unsigned *align);

   /* Total available size for load/store_preamble storage, in units
    * determined by def_size.
    */
   unsigned preamble_storage_size;

   /* Give the cost for an instruction. nir_opt_preamble will prioritize
    * instructions with higher costs. Instructions with cost 0 may still be
    * lifted, but only when required to lift other instructions with non-0
    * cost (e.g. a load_const source of an expression).
    */
   float (*instr_cost_cb)(nir_instr *instr, const void *data);

   /* Give the cost of rewriting the instruction to use load_preamble. This
    * may happen from inserting move instructions, etc. If the benefit doesn't
    * exceed the cost here then we won't rewrite it.
    */
   float (*rewrite_cost_cb)(nir_ssa_def *def, const void *data);

   /* Instructions whose definitions should not be rewritten. These could
    * still be moved to the preamble, but they shouldn't be the root of a
    * replacement expression. Instructions with cost 0 and derefs are
    * automatically included by the pass.
    */
   nir_instr_filter_cb avoid_instr_cb;

   const void *cb_data;
} nir_opt_preamble_options;

bool
nir_opt_preamble(nir_shader *shader,
                 const nir_opt_preamble_options *options,
                 unsigned *size);

nir_function_impl *nir_shader_get_preamble(nir_shader *shader);

bool nir_lower_point_smooth(nir_shader *shader);
bool nir_lower_poly_line_smooth(nir_shader *shader, unsigned num_smooth_aa_sample);

#include "nir_inline_helpers.h"

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NIR_H */
