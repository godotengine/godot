/* -*- c++ -*- */
/*
 * Copyright © 2010 Intel Corporation
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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef IR_H
#define IR_H

#include <stdio.h>
#include <stdlib.h>

#include "util/ralloc.h"
#include "util/format/u_format.h"
#include "util/half_float.h"
#include "compiler/glsl_types.h"
#include "list.h"
#include "ir_visitor.h"
#include "ir_hierarchical_visitor.h"

#ifdef __cplusplus

/**
 * \defgroup IR Intermediate representation nodes
 *
 * @{
 */

/**
 * Class tags
 *
 * Each concrete class derived from \c ir_instruction has a value in this
 * enumerant.  The value for the type is stored in \c ir_instruction::ir_type
 * by the constructor.  While using type tags is not very C++, it is extremely
 * convenient.  For example, during debugging you can simply inspect
 * \c ir_instruction::ir_type to find out the actual type of the object.
 *
 * In addition, it is possible to use a switch-statement based on \c
 * \c ir_instruction::ir_type to select different behavior for different object
 * types.  For functions that have only slight differences for several object
 * types, this allows writing very straightforward, readable code.
 */
enum ir_node_type {
   ir_type_dereference_array,
   ir_type_dereference_record,
   ir_type_dereference_variable,
   ir_type_constant,
   ir_type_expression,
   ir_type_swizzle,
   ir_type_texture,
   ir_type_variable,
   ir_type_assignment,
   ir_type_call,
   ir_type_function,
   ir_type_function_signature,
   ir_type_if,
   ir_type_loop,
   ir_type_loop_jump,
   ir_type_return,
   ir_type_discard,
   ir_type_demote,
   ir_type_emit_vertex,
   ir_type_end_primitive,
   ir_type_barrier,
   ir_type_max, /**< maximum ir_type enum number, for validation */
   ir_type_unset = ir_type_max
};


/**
 * Base class of all IR instructions
 */
class ir_instruction : public exec_node {
public:
   enum ir_node_type ir_type;

   /**
    * GCC 4.7+ and clang warn when deleting an ir_instruction unless
    * there's a virtual destructor present.  Because we almost
    * universally use ralloc for our memory management of
    * ir_instructions, the destructor doesn't need to do any work.
    */
   virtual ~ir_instruction()
   {
   }

   /** ir_print_visitor helper for debugging. */
   void print(void) const;
   void fprint(FILE *f) const;

   virtual void accept(ir_visitor *) = 0;
   virtual ir_visitor_status accept(ir_hierarchical_visitor *) = 0;
   virtual ir_instruction *clone(void *mem_ctx,
				 struct hash_table *ht) const = 0;

   bool is_rvalue() const
   {
      return ir_type == ir_type_dereference_array ||
             ir_type == ir_type_dereference_record ||
             ir_type == ir_type_dereference_variable ||
             ir_type == ir_type_constant ||
             ir_type == ir_type_expression ||
             ir_type == ir_type_swizzle ||
             ir_type == ir_type_texture;
   }

   bool is_dereference() const
   {
      return ir_type == ir_type_dereference_array ||
             ir_type == ir_type_dereference_record ||
             ir_type == ir_type_dereference_variable;
   }

   bool is_jump() const
   {
      return ir_type == ir_type_loop_jump ||
             ir_type == ir_type_return ||
             ir_type == ir_type_discard;
   }

   /**
    * \name IR instruction downcast functions
    *
    * These functions either cast the object to a derived class or return
    * \c NULL if the object's type does not match the specified derived class.
    * Additional downcast functions will be added as needed.
    */
   /*@{*/
   #define AS_BASE(TYPE)                                \
   class ir_##TYPE *as_##TYPE()                         \
   {                                                    \
      return is_##TYPE() ? (ir_##TYPE *) this : NULL;   \
   }                                                    \
   const class ir_##TYPE *as_##TYPE() const             \
   {                                                    \
      return is_##TYPE() ? (ir_##TYPE *) this : NULL;   \
   }

   AS_BASE(rvalue)
   AS_BASE(dereference)
   AS_BASE(jump)
   #undef AS_BASE

   #define AS_CHILD(TYPE) \
   class ir_##TYPE * as_##TYPE() \
   { \
      return ir_type == ir_type_##TYPE ? (ir_##TYPE *) this : NULL; \
   }                                                                      \
   const class ir_##TYPE * as_##TYPE() const                              \
   {                                                                      \
      return ir_type == ir_type_##TYPE ? (const ir_##TYPE *) this : NULL; \
   }
   AS_CHILD(variable)
   AS_CHILD(function)
   AS_CHILD(dereference_array)
   AS_CHILD(dereference_variable)
   AS_CHILD(dereference_record)
   AS_CHILD(expression)
   AS_CHILD(loop)
   AS_CHILD(assignment)
   AS_CHILD(call)
   AS_CHILD(return)
   AS_CHILD(if)
   AS_CHILD(swizzle)
   AS_CHILD(texture)
   AS_CHILD(constant)
   AS_CHILD(discard)
   #undef AS_CHILD
   /*@}*/

   /**
    * IR equality method: Return true if the referenced instruction would
    * return the same value as this one.
    *
    * This intended to be used for CSE and algebraic optimizations, on rvalues
    * in particular.  No support for other instruction types (assignments,
    * jumps, calls, etc.) is planned.
    */
   virtual bool equals(const ir_instruction *ir,
                       enum ir_node_type ignore = ir_type_unset) const;

protected:
   ir_instruction(enum ir_node_type t)
      : ir_type(t)
   {
   }

private:
   ir_instruction()
   {
      assert(!"Should not get here.");
   }
};


/**
 * The base class for all "values"/expression trees.
 */
class ir_rvalue : public ir_instruction {
public:
   const struct glsl_type *type;

   virtual ir_rvalue *clone(void *mem_ctx, struct hash_table *) const;

   virtual void accept(ir_visitor *v)
   {
      v->visit(this);
   }

   virtual ir_visitor_status accept(ir_hierarchical_visitor *);

   virtual ir_constant *constant_expression_value(void *mem_ctx,
                                                  struct hash_table *variable_context = NULL);

   ir_rvalue *as_rvalue_to_saturate();

   virtual bool is_lvalue(const struct _mesa_glsl_parse_state * = NULL) const
   {
      return false;
   }

   /**
    * Get the variable that is ultimately referenced by an r-value
    */
   virtual ir_variable *variable_referenced() const
   {
      return NULL;
   }


   /**
    * If an r-value is a reference to a whole variable, get that variable
    *
    * \return
    * Pointer to a variable that is completely dereferenced by the r-value.  If
    * the r-value is not a dereference or the dereference does not access the
    * entire variable (i.e., it's just one array element, struct field), \c NULL
    * is returned.
    */
   virtual ir_variable *whole_variable_referenced()
   {
      return NULL;
   }

   /**
    * Determine if an r-value has the value zero
    *
    * The base implementation of this function always returns \c false.  The
    * \c ir_constant class over-rides this function to return \c true \b only
    * for vector and scalar types that have all elements set to the value
    * zero (or \c false for booleans).
    *
    * \sa ir_constant::has_value, ir_rvalue::is_one, ir_rvalue::is_negative_one
    */
   virtual bool is_zero() const;

   /**
    * Determine if an r-value has the value one
    *
    * The base implementation of this function always returns \c false.  The
    * \c ir_constant class over-rides this function to return \c true \b only
    * for vector and scalar types that have all elements set to the value
    * one (or \c true for booleans).
    *
    * \sa ir_constant::has_value, ir_rvalue::is_zero, ir_rvalue::is_negative_one
    */
   virtual bool is_one() const;

   /**
    * Determine if an r-value has the value negative one
    *
    * The base implementation of this function always returns \c false.  The
    * \c ir_constant class over-rides this function to return \c true \b only
    * for vector and scalar types that have all elements set to the value
    * negative one.  For boolean types, the result is always \c false.
    *
    * \sa ir_constant::has_value, ir_rvalue::is_zero, ir_rvalue::is_one
    */
   virtual bool is_negative_one() const;

   /**
    * Determine if an r-value is an unsigned integer constant which can be
    * stored in 16 bits.
    *
    * \sa ir_constant::is_uint16_constant.
    */
   virtual bool is_uint16_constant() const { return false; }

   /**
    * Return a generic value of error_type.
    *
    * Allocation will be performed with 'mem_ctx' as ralloc owner.
    */
   static ir_rvalue *error_value(void *mem_ctx);

protected:
   ir_rvalue(enum ir_node_type t);
};


/**
 * Variable storage classes
 */
enum ir_variable_mode {
   ir_var_auto = 0,             /**< Function local variables and globals. */
   ir_var_uniform,              /**< Variable declared as a uniform. */
   ir_var_shader_storage,       /**< Variable declared as an ssbo. */
   ir_var_shader_shared,        /**< Variable declared as shared. */
   ir_var_shader_in,
   ir_var_shader_out,
   ir_var_function_in,
   ir_var_function_out,
   ir_var_function_inout,
   ir_var_const_in,             /**< "in" param that must be a constant expression */
   ir_var_system_value,         /**< Ex: front-face, instance-id, etc. */
   ir_var_temporary,            /**< Temporary variable generated during compilation. */
   ir_var_mode_count            /**< Number of variable modes */
};

/**
 * Enum keeping track of how a variable was declared.  For error checking of
 * the gl_PerVertex redeclaration rules.
 */
enum ir_var_declaration_type {
   /**
    * Normal declaration (for most variables, this means an explicit
    * declaration.  Exception: temporaries are always implicitly declared, but
    * they still use ir_var_declared_normally).
    *
    * Note: an ir_variable that represents a named interface block uses
    * ir_var_declared_normally.
    */
   ir_var_declared_normally = 0,

   /**
    * Variable was explicitly declared (or re-declared) in an unnamed
    * interface block.
    */
   ir_var_declared_in_block,

   /**
    * Variable is an implicitly declared built-in that has not been explicitly
    * re-declared by the shader.
    */
   ir_var_declared_implicitly,

   /**
    * Variable is implicitly generated by the compiler and should not be
    * visible via the API.
    */
   ir_var_hidden,
};

/**
 * \brief Layout qualifiers for gl_FragDepth.
 *
 * The AMD/ARB_conservative_depth extensions allow gl_FragDepth to be redeclared
 * with a layout qualifier.
 */
enum ir_depth_layout {
    ir_depth_layout_none, /**< No depth layout is specified. */
    ir_depth_layout_any,
    ir_depth_layout_greater,
    ir_depth_layout_less,
    ir_depth_layout_unchanged
};

/**
 * \brief Convert depth layout qualifier to string.
 */
const char*
depth_layout_string(ir_depth_layout layout);

/**
 * Description of built-in state associated with a uniform
 *
 * \sa ir_variable::state_slots
 */
struct ir_state_slot {
   gl_state_index16 tokens[STATE_LENGTH];
   int swizzle;
};


/**
 * Get the string value for an interpolation qualifier
 *
 * \return The string that would be used in a shader to specify \c
 * mode will be returned.
 *
 * This function is used to generate error messages of the form "shader
 * uses %s interpolation qualifier", so in the case where there is no
 * interpolation qualifier, it returns "no".
 *
 * This function should only be used on a shader input or output variable.
 */
const char *interpolation_string(unsigned interpolation);


class ir_variable : public ir_instruction {
public:
   ir_variable(const struct glsl_type *, const char *, ir_variable_mode);

   virtual ir_variable *clone(void *mem_ctx, struct hash_table *ht) const;

   virtual void accept(ir_visitor *v)
   {
      v->visit(this);
   }

   virtual ir_visitor_status accept(ir_hierarchical_visitor *);


   /**
    * Determine whether or not a variable is part of a uniform or
    * shader storage block.
    */
   inline bool is_in_buffer_block() const
   {
      return (this->data.mode == ir_var_uniform ||
              this->data.mode == ir_var_shader_storage) &&
             this->interface_type != NULL;
   }

   /**
    * Determine whether or not a variable is part of a shader storage block.
    */
   inline bool is_in_shader_storage_block() const
   {
      return this->data.mode == ir_var_shader_storage &&
             this->interface_type != NULL;
   }

   /**
    * Determine whether or not a variable is the declaration of an interface
    * block
    *
    * For the first declaration below, there will be an \c ir_variable named
    * "instance" whose type and whose instance_type will be the same
    * \c glsl_type.  For the second declaration, there will be an \c ir_variable
    * named "f" whose type is float and whose instance_type is B2.
    *
    * "instance" is an interface instance variable, but "f" is not.
    *
    * uniform B1 {
    *     float f;
    * } instance;
    *
    * uniform B2 {
    *     float f;
    * };
    */
   inline bool is_interface_instance() const
   {
      return this->type->without_array() == this->interface_type;
   }

   /**
    * Return whether this variable contains a bindless sampler/image.
    */
   inline bool contains_bindless() const
   {
      if (!this->type->contains_sampler() && !this->type->contains_image())
         return false;

      return this->data.bindless || this->data.mode != ir_var_uniform;
   }

   /**
    * Set this->interface_type on a newly created variable.
    */
   void init_interface_type(const struct glsl_type *type)
   {
      assert(this->interface_type == NULL);
      this->interface_type = type;
      if (this->is_interface_instance()) {
         this->u.max_ifc_array_access =
            ralloc_array(this, int, type->length);
         for (unsigned i = 0; i < type->length; i++) {
            this->u.max_ifc_array_access[i] = -1;
         }
      }
   }

   /**
    * Change this->interface_type on a variable that previously had a
    * different, but compatible, interface_type.  This is used during linking
    * to set the size of arrays in interface blocks.
    */
   void change_interface_type(const struct glsl_type *type)
   {
      if (this->u.max_ifc_array_access != NULL) {
         /* max_ifc_array_access has already been allocated, so make sure the
          * new interface has the same number of fields as the old one.
          */
         assert(this->interface_type->length == type->length);
      }
      this->interface_type = type;
   }

   /**
    * Change this->interface_type on a variable that previously had a
    * different, and incompatible, interface_type. This is used during
    * compilation to handle redeclaration of the built-in gl_PerVertex
    * interface block.
    */
   void reinit_interface_type(const struct glsl_type *type)
   {
      if (this->u.max_ifc_array_access != NULL) {
#ifndef NDEBUG
         /* Redeclaring gl_PerVertex is only allowed if none of the built-ins
          * it defines have been accessed yet; so it's safe to throw away the
          * old max_ifc_array_access pointer, since all of its values are
          * zero.
          */
         for (unsigned i = 0; i < this->interface_type->length; i++)
            assert(this->u.max_ifc_array_access[i] == -1);
#endif
         ralloc_free(this->u.max_ifc_array_access);
         this->u.max_ifc_array_access = NULL;
      }
      this->interface_type = NULL;
      init_interface_type(type);
   }

   const glsl_type *get_interface_type() const
   {
      return this->interface_type;
   }

   enum glsl_interface_packing get_interface_type_packing() const
   {
     return this->interface_type->get_interface_packing();
   }
   /**
    * Get the max_ifc_array_access pointer
    *
    * A "set" function is not needed because the array is dynamically allocated
    * as necessary.
    */
   inline int *get_max_ifc_array_access()
   {
      assert(this->data._num_state_slots == 0);
      return this->u.max_ifc_array_access;
   }

   inline unsigned get_num_state_slots() const
   {
      assert(!this->is_interface_instance()
             || this->data._num_state_slots == 0);
      return this->data._num_state_slots;
   }

   inline void set_num_state_slots(unsigned n)
   {
      assert(!this->is_interface_instance()
             || n == 0);
      this->data._num_state_slots = n;
   }

   inline ir_state_slot *get_state_slots()
   {
      return this->is_interface_instance() ? NULL : this->u.state_slots;
   }

   inline const ir_state_slot *get_state_slots() const
   {
      return this->is_interface_instance() ? NULL : this->u.state_slots;
   }

   inline ir_state_slot *allocate_state_slots(unsigned n)
   {
      assert(!this->is_interface_instance());

      this->u.state_slots = ralloc_array(this, ir_state_slot, n);
      this->data._num_state_slots = 0;

      if (this->u.state_slots != NULL)
         this->data._num_state_slots = n;

      return this->u.state_slots;
   }

   inline bool is_interpolation_flat() const
   {
      return this->data.interpolation == INTERP_MODE_FLAT ||
             this->type->contains_integer() ||
             this->type->contains_double();
   }

   inline bool is_name_ralloced() const
   {
      return this->name != ir_variable::tmp_name &&
             this->name != this->name_storage;
   }

   inline bool is_fb_fetch_color_output() const
   {
      return this->data.fb_fetch_output &&
             this->data.location != FRAG_RESULT_DEPTH &&
             this->data.location != FRAG_RESULT_STENCIL;
   }

   /**
    * Enable emitting extension warnings for this variable
    */
   void enable_extension_warning(const char *extension);

   /**
    * Get the extension warning string for this variable
    *
    * If warnings are not enabled, \c NULL is returned.
    */
   const char *get_extension_warning() const;

   /**
    * Declared type of the variable
    */
   const struct glsl_type *type;

   /**
    * Declared name of the variable
    */
   const char *name;

private:
   /**
    * If the name length fits into name_storage, it's used, otherwise
    * the name is ralloc'd. shader-db mining showed that 70% of variables
    * fit here. This is a win over ralloc where only ralloc_header has
    * 20 bytes on 64-bit (28 bytes with DEBUG), and we can also skip malloc.
    */
   char name_storage[16];

public:
   struct ir_variable_data {

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
      /**
       * Was an 'invariant' qualifier explicitly set in the shader?
       *
       * This is used to cross validate qualifiers.
       */
      unsigned explicit_invariant:1;
      /**
       * Is the variable invariant?
       *
       * It can happen either by having the 'invariant' qualifier
       * explicitly set in the shader or by being used in calculations
       * of other invariant variables.
       */
      unsigned invariant:1;
      unsigned precise:1;

      /**
       * Has this variable been used for reading or writing?
       *
       * Several GLSL semantic checks require knowledge of whether or not a
       * variable has been used.  For example, it is an error to redeclare a
       * variable as invariant after it has been used.
       *
       * This is maintained in the ast_to_hir.cpp path and during linking,
       * but not in Mesa's fixed function or ARB program paths.
       */
      unsigned used:1;

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
       * When separate shader programs are enabled, only input/outputs between
       * the stages of a multi-stage separate program can be safely removed
       * from the shader interface. Other input/outputs must remains active.
       */
      unsigned always_active_io:1;

      /**
       * Enum indicating how the variable was declared.  See
       * ir_var_declaration_type.
       *
       * This is used to detect certain kinds of illegal variable redeclarations.
       */
      unsigned how_declared:2;

      /**
       * Storage class of the variable.
       *
       * \sa ir_variable_mode
       */
      unsigned mode:4;

      /**
       * Interpolation mode for shader inputs / outputs
       *
       * \sa glsl_interp_mode
       */
      unsigned interpolation:2;

      /**
       * Was the location explicitly set in the shader?
       *
       * If the location is explicitly set in the shader, it \b cannot be changed
       * by the linker or by the API (e.g., calls to \c glBindAttribLocation have
       * no effect).
       */
      unsigned explicit_location:1;
      unsigned explicit_index:1;

      /**
       * Was an initial binding explicitly set in the shader?
       *
       * If so, constant_value contains an integer ir_constant representing the
       * initial binding point.
       */
      unsigned explicit_binding:1;

      /**
       * Was an initial component explicitly set in the shader?
       */
      unsigned explicit_component:1;

      /**
       * Does this variable have an initializer?
       *
       * This is used by the linker to cross-validiate initializers of global
       * variables.
       */
      unsigned has_initializer:1;

      /**
       * Is the initializer created by the compiler (glsl_zero_init)
       */
      unsigned is_implicit_initializer:1;

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
       * Was a transform feedback buffer set in the shader?
       */
      unsigned explicit_xfb_buffer:1;

      /**
       * Was a transform feedback offset set in the shader?
       */
      unsigned explicit_xfb_offset:1;

      /**
       * Was a transform feedback stride set in the shader?
       */
      unsigned explicit_xfb_stride:1;

      /**
       * If non-zero, then this variable may be packed along with other variables
       * into a single varying slot, so this offset should be applied when
       * accessing components.  For example, an offset of 1 means that the x
       * component of this variable is actually stored in component y of the
       * location specified by \c location.
       */
      unsigned location_frac:2;

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
       * Output index for dual source blending.
       *
       * \note
       * The GLSL spec only allows the values 0 or 1 for the index in \b dual
       * source blending.
       */
      unsigned index:1;

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
       * \brief Layout qualifier for gl_FragDepth.
       *
       * This is not equal to \c ir_depth_layout_none if and only if this
       * variable is \c gl_FragDepth and a layout qualifier is specified.
       */
      ir_depth_layout depth_layout:3;

      /**
       * Memory qualifiers.
       */
      unsigned memory_read_only:1; /**< "readonly" qualifier. */
      unsigned memory_write_only:1; /**< "writeonly" qualifier. */
      unsigned memory_coherent:1;
      unsigned memory_volatile:1;
      unsigned memory_restrict:1;

      /**
       * ARB_shader_storage_buffer_object
       */
      unsigned from_ssbo_unsized_array:1; /**< unsized array buffer variable. */

      unsigned implicit_sized_array:1;

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
       * Non-zero if this variable is considered bound as defined by
       * ARB_bindless_texture.
       */
      unsigned bound:1;

      /**
       * Non-zero if the variable shall not be implicitly converted during
       * functions matching.
       */
      unsigned implicit_conversion_prohibited:1;

      /**
       * Emit a warning if this variable is accessed.
       */
   private:
      uint8_t warn_extension_index;

   public:
      /**
       * Image internal format if specified explicitly, otherwise
       * PIPE_FORMAT_NONE.
       */
      enum pipe_format image_format;

   private:
      /**
       * Number of state slots used
       *
       * \note
       * This could be stored in as few as 7-bits, if necessary.  If it is made
       * smaller, add an assertion to \c ir_variable::allocate_state_slots to
       * be safe.
       */
      uint16_t _num_state_slots;

   public:
      /**
       * Initial binding point for a sampler, atomic, or UBO.
       *
       * For array types, this represents the binding point for the first element.
       */
      uint16_t binding;

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
       *   - Uniforms: Per-stage uniform slot number for default uniform block.
       *   - Uniforms: Index within the uniform block definition for UBO members.
       *   - Non-UBO Uniforms: explicit location until linking then reused to
       *     store uniform slot number.
       *   - Other: This field is not currently used.
       *
       * If the variable is a uniform, shader input, or shader output, and the
       * slot has not been assigned, the value will be -1.
       */
      int location;

      /**
       * for glsl->tgsi/mesa IR we need to store the index into the
       * parameters for uniforms, initially the code overloaded location
       * but this causes problems with indirect samplers and AoA.
       * This is assigned in _mesa_generate_parameters_list_for_uniforms.
       */
      int param_index;

      /**
       * Vertex stream output identifier.
       *
       * For packed outputs, bit 31 is set and bits [2*i+1,2*i] indicate the
       * stream of the i-th component.
       */
      unsigned stream;

      /**
       * Atomic, transform feedback or block member offset.
       */
      unsigned offset;

      /**
       * Highest element accessed with a constant expression array index
       *
       * Not used for non-array variables. -1 is never accessed.
       */
      int max_array_access;

      /**
       * Transform feedback buffer.
       */
      unsigned xfb_buffer;

      /**
       * Transform feedback stride.
       */
      unsigned xfb_stride;

      /**
       * Allow (only) ir_variable direct access private members.
       */
      friend class ir_variable;
   } data;

   /**
    * Value assigned in the initializer of a variable declared "const"
    */
   ir_constant *constant_value;

   /**
    * Constant expression assigned in the initializer of the variable
    *
    * \warning
    * This field and \c ::constant_value are distinct.  Even if the two fields
    * refer to constants with the same value, they must point to separate
    * objects.
    */
   ir_constant *constant_initializer;

private:
   static const char *const warn_extension_table[];

   union {
      /**
       * For variables which satisfy the is_interface_instance() predicate,
       * this points to an array of integers such that if the ith member of
       * the interface block is an array, max_ifc_array_access[i] is the
       * maximum array element of that member that has been accessed.  If the
       * ith member of the interface block is not an array,
       * max_ifc_array_access[i] is unused.
       *
       * For variables whose type is not an interface block, this pointer is
       * NULL.
       */
      int *max_ifc_array_access;

      /**
       * Built-in state that backs this uniform
       *
       * Once set at variable creation, \c state_slots must remain invariant.
       *
       * If the variable is not a uniform, \c _num_state_slots will be zero
       * and \c state_slots will be \c NULL.
       */
      ir_state_slot *state_slots;
   } u;

   /**
    * For variables that are in an interface block or are an instance of an
    * interface block, this is the \c GLSL_TYPE_INTERFACE type for that block.
    *
    * \sa ir_variable::location
    */
   const glsl_type *interface_type;

   /**
    * Name used for anonymous compiler temporaries
    */
   static const char tmp_name[];

public:
   /**
    * Should the construct keep names for ir_var_temporary variables?
    *
    * When this global is false, names passed to the constructor for
    * \c ir_var_temporary variables will be dropped.  Instead, the variable will
    * be named "compiler_temp".  This name will be in static storage.
    *
    * \warning
    * \b NEVER change the mode of an \c ir_var_temporary.
    *
    * \warning
    * This variable is \b not thread-safe.  It is global, \b not
    * per-context. It begins life false.  A context can, at some point, make
    * it true.  From that point on, it will be true forever.  This should be
    * okay since it will only be set true while debugging.
    */
   static bool temporaries_allocate_names;
};

/**
 * A function that returns whether a built-in function is available in the
 * current shading language (based on version, ES or desktop, and extensions).
 */
typedef bool (*builtin_available_predicate)(const _mesa_glsl_parse_state *);

#define MAKE_INTRINSIC_FOR_TYPE(op, t) \
   ir_intrinsic_generic_ ## op - ir_intrinsic_generic_load + ir_intrinsic_ ## t ## _ ## load

#define MAP_INTRINSIC_TO_TYPE(i, t) \
   ir_intrinsic_id(int(i) - int(ir_intrinsic_generic_load) + int(ir_intrinsic_ ## t ## _ ## load))

enum ir_intrinsic_id {
   ir_intrinsic_invalid = 0,

   /**
    * \name Generic intrinsics
    *
    * Each of these intrinsics has a specific version for shared variables and
    * SSBOs.
    */
   /*@{*/
   ir_intrinsic_generic_load,
   ir_intrinsic_generic_store,
   ir_intrinsic_generic_atomic_add,
   ir_intrinsic_generic_atomic_and,
   ir_intrinsic_generic_atomic_or,
   ir_intrinsic_generic_atomic_xor,
   ir_intrinsic_generic_atomic_min,
   ir_intrinsic_generic_atomic_max,
   ir_intrinsic_generic_atomic_exchange,
   ir_intrinsic_generic_atomic_comp_swap,
   /*@}*/

   ir_intrinsic_atomic_counter_read,
   ir_intrinsic_atomic_counter_increment,
   ir_intrinsic_atomic_counter_predecrement,
   ir_intrinsic_atomic_counter_add,
   ir_intrinsic_atomic_counter_and,
   ir_intrinsic_atomic_counter_or,
   ir_intrinsic_atomic_counter_xor,
   ir_intrinsic_atomic_counter_min,
   ir_intrinsic_atomic_counter_max,
   ir_intrinsic_atomic_counter_exchange,
   ir_intrinsic_atomic_counter_comp_swap,

   ir_intrinsic_image_load,
   ir_intrinsic_image_store,
   ir_intrinsic_image_atomic_add,
   ir_intrinsic_image_atomic_and,
   ir_intrinsic_image_atomic_or,
   ir_intrinsic_image_atomic_xor,
   ir_intrinsic_image_atomic_min,
   ir_intrinsic_image_atomic_max,
   ir_intrinsic_image_atomic_exchange,
   ir_intrinsic_image_atomic_comp_swap,
   ir_intrinsic_image_size,
   ir_intrinsic_image_samples,
   ir_intrinsic_image_atomic_inc_wrap,
   ir_intrinsic_image_atomic_dec_wrap,
   ir_intrinsic_image_sparse_load,

   ir_intrinsic_memory_barrier,
   ir_intrinsic_shader_clock,
   ir_intrinsic_group_memory_barrier,
   ir_intrinsic_memory_barrier_atomic_counter,
   ir_intrinsic_memory_barrier_buffer,
   ir_intrinsic_memory_barrier_image,
   ir_intrinsic_memory_barrier_shared,
   ir_intrinsic_begin_invocation_interlock,
   ir_intrinsic_end_invocation_interlock,

   ir_intrinsic_vote_all,
   ir_intrinsic_vote_any,
   ir_intrinsic_vote_eq,
   ir_intrinsic_ballot,
   ir_intrinsic_read_invocation,
   ir_intrinsic_read_first_invocation,

   ir_intrinsic_helper_invocation,

   ir_intrinsic_shared_load,
   ir_intrinsic_shared_store = MAKE_INTRINSIC_FOR_TYPE(store, shared),
   ir_intrinsic_shared_atomic_add = MAKE_INTRINSIC_FOR_TYPE(atomic_add, shared),
   ir_intrinsic_shared_atomic_and = MAKE_INTRINSIC_FOR_TYPE(atomic_and, shared),
   ir_intrinsic_shared_atomic_or = MAKE_INTRINSIC_FOR_TYPE(atomic_or, shared),
   ir_intrinsic_shared_atomic_xor = MAKE_INTRINSIC_FOR_TYPE(atomic_xor, shared),
   ir_intrinsic_shared_atomic_min = MAKE_INTRINSIC_FOR_TYPE(atomic_min, shared),
   ir_intrinsic_shared_atomic_max = MAKE_INTRINSIC_FOR_TYPE(atomic_max, shared),
   ir_intrinsic_shared_atomic_exchange = MAKE_INTRINSIC_FOR_TYPE(atomic_exchange, shared),
   ir_intrinsic_shared_atomic_comp_swap = MAKE_INTRINSIC_FOR_TYPE(atomic_comp_swap, shared),

   ir_intrinsic_is_sparse_texels_resident,
};

/*@{*/
/**
 * The representation of a function instance; may be the full definition or
 * simply a prototype.
 */
class ir_function_signature : public ir_instruction {
   /* An ir_function_signature will be part of the list of signatures in
    * an ir_function.
    */
public:
   ir_function_signature(const glsl_type *return_type,
                         builtin_available_predicate builtin_avail = NULL);

   virtual ir_function_signature *clone(void *mem_ctx,
					struct hash_table *ht) const;
   ir_function_signature *clone_prototype(void *mem_ctx,
					  struct hash_table *ht) const;

   virtual void accept(ir_visitor *v)
   {
      v->visit(this);
   }

   virtual ir_visitor_status accept(ir_hierarchical_visitor *);

   /**
    * Attempt to evaluate this function as a constant expression,
    * given a list of the actual parameters and the variable context.
    * Returns NULL for non-built-ins.
    */
   ir_constant *constant_expression_value(void *mem_ctx,
                                          exec_list *actual_parameters,
                                          struct hash_table *variable_context);

   /**
    * Get the name of the function for which this is a signature
    */
   const char *function_name() const;

   /**
    * Get a handle to the function for which this is a signature
    *
    * There is no setter function, this function returns a \c const pointer,
    * and \c ir_function_signature::_function is private for a reason.  The
    * only way to make a connection between a function and function signature
    * is via \c ir_function::add_signature.  This helps ensure that certain
    * invariants (i.e., a function signature is in the list of signatures for
    * its \c _function) are met.
    *
    * \sa ir_function::add_signature
    */
   inline const class ir_function *function() const
   {
      return this->_function;
   }

   /**
    * Check whether the qualifiers match between this signature's parameters
    * and the supplied parameter list.  If not, returns the name of the first
    * parameter with mismatched qualifiers (for use in error messages).
    */
   const char *qualifiers_match(exec_list *params);

   /**
    * Replace the current parameter list with the given one.  This is useful
    * if the current information came from a prototype, and either has invalid
    * or missing parameter names.
    */
   void replace_parameters(exec_list *new_params);

   /**
    * Function return type.
    *
    * \note The precision qualifier is stored separately in return_precision.
    */
   const struct glsl_type *return_type;

   /**
    * List of ir_variable of function parameters.
    *
    * This represents the storage.  The paramaters passed in a particular
    * call will be in ir_call::actual_paramaters.
    */
   struct exec_list parameters;

   /** Whether or not this function has a body (which may be empty). */
   unsigned is_defined:1;

   /*
    * Precision qualifier for the return type.
    *
    * See the comment for ir_variable_data::precision for more details.
    */
   unsigned return_precision:2;

   /** Whether or not this function signature is a built-in. */
   bool is_builtin() const;

   /**
    * Whether or not this function is an intrinsic to be implemented
    * by the driver.
    */
   inline bool is_intrinsic() const
   {
      return intrinsic_id != ir_intrinsic_invalid;
   }

   /** Identifier for this intrinsic. */
   enum ir_intrinsic_id intrinsic_id;

   /** Whether or not a built-in is available for this shader. */
   bool is_builtin_available(const _mesa_glsl_parse_state *state) const;

   /** Body of instructions in the function. */
   struct exec_list body;

private:
   /**
    * A function pointer to a predicate that answers whether a built-in
    * function is available in the current shader.  NULL if not a built-in.
    */
   builtin_available_predicate builtin_avail;

   /** Function of which this signature is one overload. */
   class ir_function *_function;

   /** Function signature of which this one is a prototype clone */
   const ir_function_signature *origin;

   friend class ir_function;

   /**
    * Helper function to run a list of instructions for constant
    * expression evaluation.
    *
    * The hash table represents the values of the visible variables.
    * There are no scoping issues because the table is indexed on
    * ir_variable pointers, not variable names.
    *
    * Returns false if the expression is not constant, true otherwise,
    * and the value in *result if result is non-NULL.
    */
   bool constant_expression_evaluate_expression_list(void *mem_ctx,
                                                     const struct exec_list &body,
						     struct hash_table *variable_context,
						     ir_constant **result);
};


/**
 * Header for tracking multiple overloaded functions with the same name.
 * Contains a list of ir_function_signatures representing each of the
 * actual functions.
 */
class ir_function : public ir_instruction {
public:
   ir_function(const char *name);

   virtual ir_function *clone(void *mem_ctx, struct hash_table *ht) const;

   virtual void accept(ir_visitor *v)
   {
      v->visit(this);
   }

   virtual ir_visitor_status accept(ir_hierarchical_visitor *);

   void add_signature(ir_function_signature *sig)
   {
      sig->_function = this;
      this->signatures.push_tail(sig);
   }

   /**
    * Find a signature that matches a set of actual parameters, taking implicit
    * conversions into account.  Also flags whether the match was exact.
    */
   ir_function_signature *matching_signature(_mesa_glsl_parse_state *state,
                                             const exec_list *actual_param,
                                             bool allow_builtins,
					     bool *match_is_exact);

   /**
    * Find a signature that matches a set of actual parameters, taking implicit
    * conversions into account.
    */
   ir_function_signature *matching_signature(_mesa_glsl_parse_state *state,
                                             const exec_list *actual_param,
                                             bool allow_builtins);

   /**
    * Find a signature that exactly matches a set of actual parameters without
    * any implicit type conversions.
    */
   ir_function_signature *exact_matching_signature(_mesa_glsl_parse_state *state,
                                                   const exec_list *actual_ps);

   /**
    * Name of the function.
    */
   const char *name;

   /** Whether or not this function has a signature that isn't a built-in. */
   bool has_user_signature();

   /**
    * List of ir_function_signature for each overloaded function with this name.
    */
   struct exec_list signatures;

   /**
    * is this function a subroutine type declaration
    * e.g. subroutine void type1(float arg1);
    */
   bool is_subroutine;

   /**
    * is this function associated to a subroutine type
    * e.g. subroutine (type1, type2) function_name { function_body };
    * would have num_subroutine_types 2,
    * and pointers to the type1 and type2 types.
    */
   int num_subroutine_types;
   const struct glsl_type **subroutine_types;

   int subroutine_index;
};

inline const char *ir_function_signature::function_name() const
{
   return this->_function->name;
}
/*@}*/


/**
 * IR instruction representing high-level if-statements
 */
class ir_if : public ir_instruction {
public:
   ir_if(ir_rvalue *condition)
      : ir_instruction(ir_type_if), condition(condition)
   {
   }

   virtual ir_if *clone(void *mem_ctx, struct hash_table *ht) const;

   virtual void accept(ir_visitor *v)
   {
      v->visit(this);
   }

   virtual ir_visitor_status accept(ir_hierarchical_visitor *);

   ir_rvalue *condition;
   /** List of ir_instruction for the body of the then branch */
   exec_list  then_instructions;
   /** List of ir_instruction for the body of the else branch */
   exec_list  else_instructions;
};


/**
 * IR instruction representing a high-level loop structure.
 */
class ir_loop : public ir_instruction {
public:
   ir_loop();

   virtual ir_loop *clone(void *mem_ctx, struct hash_table *ht) const;

   virtual void accept(ir_visitor *v)
   {
      v->visit(this);
   }

   virtual ir_visitor_status accept(ir_hierarchical_visitor *);

   /** List of ir_instruction that make up the body of the loop. */
   exec_list body_instructions;
};


class ir_assignment : public ir_instruction {
public:
   ir_assignment(ir_rvalue *lhs, ir_rvalue *rhs);

   /**
    * Construct an assignment with an explicit write mask
    *
    * \note
    * Since a write mask is supplied, the LHS must already be a bare
    * \c ir_dereference.  The cannot be any swizzles in the LHS.
    */
   ir_assignment(ir_dereference *lhs, ir_rvalue *rhs, unsigned write_mask);

   virtual ir_assignment *clone(void *mem_ctx, struct hash_table *ht) const;

   virtual ir_constant *constant_expression_value(void *mem_ctx,
                                                  struct hash_table *variable_context = NULL);

   virtual void accept(ir_visitor *v)
   {
      v->visit(this);
   }

   virtual ir_visitor_status accept(ir_hierarchical_visitor *);

   /**
    * Get a whole variable written by an assignment
    *
    * If the LHS of the assignment writes a whole variable, the variable is
    * returned.  Otherwise \c NULL is returned.  Examples of whole-variable
    * assignment are:
    *
    *  - Assigning to a scalar
    *  - Assigning to all components of a vector
    *  - Whole array (or matrix) assignment
    *  - Whole structure assignment
    */
   ir_variable *whole_variable_written();

   /**
    * Set the LHS of an assignment
    */
   void set_lhs(ir_rvalue *lhs);

   /**
    * Left-hand side of the assignment.
    *
    * This should be treated as read only.  If you need to set the LHS of an
    * assignment, use \c ir_assignment::set_lhs.
    */
   ir_dereference *lhs;

   /**
    * Value being assigned
    */
   ir_rvalue *rhs;

   /**
    * Component mask written
    *
    * For non-vector types in the LHS, this field will be zero.  For vector
    * types, a bit will be set for each component that is written.  Note that
    * for \c vec2 and \c vec3 types only the lower bits will ever be set.
    *
    * A partially-set write mask means that each enabled channel gets
    * the value from a consecutive channel of the rhs.  For example,
    * to write just .xyw of gl_FrontColor with color:
    *
    * (assign (constant bool (1)) (xyw)
    *     (var_ref gl_FragColor)
    *     (swiz xyw (var_ref color)))
    */
   unsigned write_mask:4;
};

#include "ir_expression_operation.h"

extern const char *const ir_expression_operation_strings[ir_last_opcode + 1];
extern const char *const ir_expression_operation_enum_strings[ir_last_opcode + 1];

class ir_expression : public ir_rvalue {
public:
   ir_expression(int op, const struct glsl_type *type,
                 ir_rvalue *op0, ir_rvalue *op1 = NULL,
                 ir_rvalue *op2 = NULL, ir_rvalue *op3 = NULL);

   /**
    * Constructor for unary operation expressions
    */
   ir_expression(int op, ir_rvalue *);

   /**
    * Constructor for binary operation expressions
    */
   ir_expression(int op, ir_rvalue *op0, ir_rvalue *op1);

   /**
    * Constructor for ternary operation expressions
    */
   ir_expression(int op, ir_rvalue *op0, ir_rvalue *op1, ir_rvalue *op2);

   virtual bool equals(const ir_instruction *ir,
                       enum ir_node_type ignore = ir_type_unset) const;

   virtual ir_expression *clone(void *mem_ctx, struct hash_table *ht) const;

   /**
    * Attempt to constant-fold the expression
    *
    * The "variable_context" hash table links ir_variable * to ir_constant *
    * that represent the variables' values.  \c NULL represents an empty
    * context.
    *
    * If the expression cannot be constant folded, this method will return
    * \c NULL.
    */
   virtual ir_constant *constant_expression_value(void *mem_ctx,
                                                  struct hash_table *variable_context = NULL);

   /**
    * This is only here for ir_reader to used for testing purposes please use
    * the precomputed num_operands field if you need the number of operands.
    */
   static unsigned get_num_operands(ir_expression_operation);

   /**
    * Return whether the expression operates on vectors horizontally.
    */
   bool is_horizontal() const
   {
      return operation == ir_binop_all_equal ||
             operation == ir_binop_any_nequal ||
             operation == ir_binop_dot ||
             operation == ir_binop_vector_extract ||
             operation == ir_triop_vector_insert ||
             operation == ir_binop_ubo_load ||
             operation == ir_quadop_vector;
   }

   /**
    * Do a reverse-lookup to translate the given string into an operator.
    */
   static ir_expression_operation get_operator(const char *);

   virtual void accept(ir_visitor *v)
   {
      v->visit(this);
   }

   virtual ir_visitor_status accept(ir_hierarchical_visitor *);

   virtual ir_variable *variable_referenced() const;

   /**
    * Determine the number of operands used by an expression
    */
   void init_num_operands()
   {
      if (operation == ir_quadop_vector) {
         num_operands = this->type->vector_elements;
      } else {
         num_operands = get_num_operands(operation);
      }
   }

   ir_expression_operation operation;
   ir_rvalue *operands[4];
   uint8_t num_operands;
};


/**
 * HIR instruction representing a high-level function call, containing a list
 * of parameters and returning a value in the supplied temporary.
 */
class ir_call : public ir_instruction {
public:
   ir_call(ir_function_signature *callee,
	   ir_dereference_variable *return_deref,
	   exec_list *actual_parameters)
      : ir_instruction(ir_type_call), return_deref(return_deref), callee(callee), sub_var(NULL), array_idx(NULL)
   {
      assert(callee->return_type != NULL);
      actual_parameters->move_nodes_to(& this->actual_parameters);
   }

   ir_call(ir_function_signature *callee,
	   ir_dereference_variable *return_deref,
	   exec_list *actual_parameters,
	   ir_variable *var, ir_rvalue *array_idx)
      : ir_instruction(ir_type_call), return_deref(return_deref), callee(callee), sub_var(var), array_idx(array_idx)
   {
      assert(callee->return_type != NULL);
      actual_parameters->move_nodes_to(& this->actual_parameters);
   }

   virtual ir_call *clone(void *mem_ctx, struct hash_table *ht) const;

   virtual ir_constant *constant_expression_value(void *mem_ctx,
                                                  struct hash_table *variable_context = NULL);

   virtual void accept(ir_visitor *v)
   {
      v->visit(this);
   }

   virtual ir_visitor_status accept(ir_hierarchical_visitor *);

   /**
    * Get the name of the function being called.
    */
   const char *callee_name() const
   {
      return callee->function_name();
   }

   /**
    * Generates an inline version of the function before @ir,
    * storing the return value in return_deref.
    */
   void generate_inline(ir_instruction *ir);

   /**
    * Storage for the function's return value.
    * This must be NULL if the return type is void.
    */
   ir_dereference_variable *return_deref;

   /**
    * The specific function signature being called.
    */
   ir_function_signature *callee;

   /* List of ir_rvalue of paramaters passed in this call. */
   exec_list actual_parameters;

   /*
    * ARB_shader_subroutine support -
    * the subroutine uniform variable and array index
    * rvalue to be used in the lowering pass later.
    */
   ir_variable *sub_var;
   ir_rvalue *array_idx;
};


/**
 * \name Jump-like IR instructions.
 *
 * These include \c break, \c continue, \c return, and \c discard.
 */
/*@{*/
class ir_jump : public ir_instruction {
protected:
   ir_jump(enum ir_node_type t)
      : ir_instruction(t)
   {
   }
};

class ir_return : public ir_jump {
public:
   ir_return()
      : ir_jump(ir_type_return), value(NULL)
   {
   }

   ir_return(ir_rvalue *value)
      : ir_jump(ir_type_return), value(value)
   {
   }

   virtual ir_return *clone(void *mem_ctx, struct hash_table *) const;

   ir_rvalue *get_value() const
   {
      return value;
   }

   virtual void accept(ir_visitor *v)
   {
      v->visit(this);
   }

   virtual ir_visitor_status accept(ir_hierarchical_visitor *);

   ir_rvalue *value;
};


/**
 * Jump instructions used inside loops
 *
 * These include \c break and \c continue.  The \c break within a loop is
 * different from the \c break within a switch-statement.
 *
 * \sa ir_switch_jump
 */
class ir_loop_jump : public ir_jump {
public:
   enum jump_mode {
      jump_break,
      jump_continue
   };

   ir_loop_jump(jump_mode mode)
      : ir_jump(ir_type_loop_jump)
   {
      this->mode = mode;
   }

   virtual ir_loop_jump *clone(void *mem_ctx, struct hash_table *) const;

   virtual void accept(ir_visitor *v)
   {
      v->visit(this);
   }

   virtual ir_visitor_status accept(ir_hierarchical_visitor *);

   bool is_break() const
   {
      return mode == jump_break;
   }

   bool is_continue() const
   {
      return mode == jump_continue;
   }

   /** Mode selector for the jump instruction. */
   enum jump_mode mode;
};

/**
 * IR instruction representing discard statements.
 */
class ir_discard : public ir_jump {
public:
   ir_discard()
      : ir_jump(ir_type_discard)
   {
      this->condition = NULL;
   }

   ir_discard(ir_rvalue *cond)
      : ir_jump(ir_type_discard)
   {
      this->condition = cond;
   }

   virtual ir_discard *clone(void *mem_ctx, struct hash_table *ht) const;

   virtual void accept(ir_visitor *v)
   {
      v->visit(this);
   }

   virtual ir_visitor_status accept(ir_hierarchical_visitor *);

   ir_rvalue *condition;
};
/*@}*/


/**
 * IR instruction representing demote statements from
 * GL_EXT_demote_to_helper_invocation.
 */
class ir_demote : public ir_instruction {
public:
   ir_demote()
      : ir_instruction(ir_type_demote)
   {
   }

   virtual ir_demote *clone(void *mem_ctx, struct hash_table *ht) const;

   virtual void accept(ir_visitor *v)
   {
      v->visit(this);
   }

   virtual ir_visitor_status accept(ir_hierarchical_visitor *);
};


/**
 * Texture sampling opcodes used in ir_texture
 */
enum ir_texture_opcode {
   ir_tex,		/**< Regular texture look-up */
   ir_txb,		/**< Texture look-up with LOD bias */
   ir_txl,		/**< Texture look-up with explicit LOD */
   ir_txd,		/**< Texture look-up with partial derivatives */
   ir_txf,		/**< Texel fetch with explicit LOD */
   ir_txf_ms,           /**< Multisample texture fetch */
   ir_txs,		/**< Texture size */
   ir_lod,		/**< Texture lod query */
   ir_tg4,		/**< Texture gather */
   ir_query_levels,     /**< Texture levels query */
   ir_texture_samples,  /**< Texture samples query */
   ir_samples_identical, /**< Query whether all samples are definitely identical. */
};


/**
 * IR instruction to sample a texture
 *
 * The specific form of the IR instruction depends on the \c mode value
 * selected from \c ir_texture_opcodes.  In the printed IR, these will
 * appear as:
 *
 *                                             Texel offset (0 or an expression)
 *                                             | Projection divisor
 *                                             | |  Shadow comparator
 *                                             | |  |   Lod clamp
 *                                             | |  |   |
 *                                             v v  v   v
 * (tex <type> <sampler> <coordinate> <sparse> 0 1 ( ) ( ))
 * (txb <type> <sampler> <coordinate> <sparse> 0 1 ( ) ( ) <bias>)
 * (txl <type> <sampler> <coordinate> <sparse> 0 1 ( )     <lod>)
 * (txd <type> <sampler> <coordinate> <sparse> 0 1 ( ) ( ) (dPdx dPdy))
 * (txf <type> <sampler> <coordinate> <sparse> 0	         <lod>)
 * (txf_ms
 *      <type> <sampler> <coordinate> <sparse>             <sample_index>)
 * (txs <type> <sampler> <lod>)
 * (lod <type> <sampler> <coordinate>)
 * (tg4 <type> <sampler> <coordinate> <sparse>             <offset> <component>)
 * (query_levels <type> <sampler>)
 * (samples_identical <sampler> <coordinate>)
 */
class ir_texture : public ir_rvalue {
public:
   ir_texture(enum ir_texture_opcode op, bool sparse = false)
      : ir_rvalue(ir_type_texture),
        op(op), sampler(NULL), coordinate(NULL), projector(NULL),
        shadow_comparator(NULL), offset(NULL), clamp(NULL),
        is_sparse(sparse)
   {
      memset(&lod_info, 0, sizeof(lod_info));
   }

   virtual ir_texture *clone(void *mem_ctx, struct hash_table *) const;

   virtual ir_constant *constant_expression_value(void *mem_ctx,
                                                  struct hash_table *variable_context = NULL);

   virtual void accept(ir_visitor *v)
   {
      v->visit(this);
   }

   virtual ir_visitor_status accept(ir_hierarchical_visitor *);

   virtual bool equals(const ir_instruction *ir,
                       enum ir_node_type ignore = ir_type_unset) const;

   /**
    * Return a string representing the ir_texture_opcode.
    */
   const char *opcode_string();

   /** Set the sampler and type. */
   void set_sampler(ir_dereference *sampler, const glsl_type *type);

   /**
    * Do a reverse-lookup to translate a string into an ir_texture_opcode.
    */
   static ir_texture_opcode get_opcode(const char *);

   enum ir_texture_opcode op;

   /** Sampler to use for the texture access. */
   ir_dereference *sampler;

   /** Texture coordinate to sample */
   ir_rvalue *coordinate;

   /**
    * Value used for projective divide.
    *
    * If there is no projective divide (the common case), this will be
    * \c NULL.  Optimization passes should check for this to point to a constant
    * of 1.0 and replace that with \c NULL.
    */
   ir_rvalue *projector;

   /**
    * Coordinate used for comparison on shadow look-ups.
    *
    * If there is no shadow comparison, this will be \c NULL.  For the
    * \c ir_txf opcode, this *must* be \c NULL.
    */
   ir_rvalue *shadow_comparator;

   /** Texel offset. */
   ir_rvalue *offset;

   /** Lod clamp. */
   ir_rvalue *clamp;

   union {
      ir_rvalue *lod;		/**< Floating point LOD */
      ir_rvalue *bias;		/**< Floating point LOD bias */
      ir_rvalue *sample_index;  /**< MSAA sample index */
      ir_rvalue *component;     /**< Gather component selector */
      struct {
	 ir_rvalue *dPdx;	/**< Partial derivative of coordinate wrt X */
	 ir_rvalue *dPdy;	/**< Partial derivative of coordinate wrt Y */
      } grad;
   } lod_info;

   /* Whether a sparse texture */
   bool is_sparse;
};


struct ir_swizzle_mask {
   unsigned x:2;
   unsigned y:2;
   unsigned z:2;
   unsigned w:2;

   /**
    * Number of components in the swizzle.
    */
   unsigned num_components:3;

   /**
    * Does the swizzle contain duplicate components?
    *
    * L-value swizzles cannot contain duplicate components.
    */
   unsigned has_duplicates:1;
};


class ir_swizzle : public ir_rvalue {
public:
   ir_swizzle(ir_rvalue *, unsigned x, unsigned y, unsigned z, unsigned w,
              unsigned count);

   ir_swizzle(ir_rvalue *val, const unsigned *components, unsigned count);

   ir_swizzle(ir_rvalue *val, ir_swizzle_mask mask);

   virtual ir_swizzle *clone(void *mem_ctx, struct hash_table *) const;

   virtual ir_constant *constant_expression_value(void *mem_ctx,
                                                  struct hash_table *variable_context = NULL);

   /**
    * Construct an ir_swizzle from the textual representation.  Can fail.
    */
   static ir_swizzle *create(ir_rvalue *, const char *, unsigned vector_length);

   virtual void accept(ir_visitor *v)
   {
      v->visit(this);
   }

   virtual ir_visitor_status accept(ir_hierarchical_visitor *);

   virtual bool equals(const ir_instruction *ir,
                       enum ir_node_type ignore = ir_type_unset) const;

   bool is_lvalue(const struct _mesa_glsl_parse_state *state) const
   {
      return val->is_lvalue(state) && !mask.has_duplicates;
   }

   /**
    * Get the variable that is ultimately referenced by an r-value
    */
   virtual ir_variable *variable_referenced() const;

   ir_rvalue *val;
   ir_swizzle_mask mask;

private:
   /**
    * Initialize the mask component of a swizzle
    *
    * This is used by the \c ir_swizzle constructors.
    */
   void init_mask(const unsigned *components, unsigned count);
};


class ir_dereference : public ir_rvalue {
public:
   virtual ir_dereference *clone(void *mem_ctx, struct hash_table *) const = 0;

   bool is_lvalue(const struct _mesa_glsl_parse_state *state) const;

   /**
    * Get the variable that is ultimately referenced by an r-value
    */
   virtual ir_variable *variable_referenced() const = 0;

   /**
    * Get the precision. This can either come from the eventual variable that
    * is dereferenced, or from a record member.
    */
   virtual int precision() const = 0;

protected:
   ir_dereference(enum ir_node_type t)
      : ir_rvalue(t)
   {
   }
};


class ir_dereference_variable : public ir_dereference {
public:
   ir_dereference_variable(ir_variable *var);

   virtual ir_dereference_variable *clone(void *mem_ctx,
					  struct hash_table *) const;

   virtual ir_constant *constant_expression_value(void *mem_ctx,
                                                  struct hash_table *variable_context = NULL);

   virtual bool equals(const ir_instruction *ir,
                       enum ir_node_type ignore = ir_type_unset) const;

   /**
    * Get the variable that is ultimately referenced by an r-value
    */
   virtual ir_variable *variable_referenced() const
   {
      return this->var;
   }

   virtual int precision() const
   {
      return this->var->data.precision;
   }

   virtual ir_variable *whole_variable_referenced()
   {
      /* ir_dereference_variable objects always dereference the entire
       * variable.  However, if this dereference is dereferenced by anything
       * else, the complete dereference chain is not a whole-variable
       * dereference.  This method should only be called on the top most
       * ir_rvalue in a dereference chain.
       */
      return this->var;
   }

   virtual void accept(ir_visitor *v)
   {
      v->visit(this);
   }

   virtual ir_visitor_status accept(ir_hierarchical_visitor *);

   /**
    * Object being dereferenced.
    */
   ir_variable *var;
};


class ir_dereference_array : public ir_dereference {
public:
   ir_dereference_array(ir_rvalue *value, ir_rvalue *array_index);

   ir_dereference_array(ir_variable *var, ir_rvalue *array_index);

   virtual ir_dereference_array *clone(void *mem_ctx,
				       struct hash_table *) const;

   virtual ir_constant *constant_expression_value(void *mem_ctx,
                                                  struct hash_table *variable_context = NULL);

   virtual bool equals(const ir_instruction *ir,
                       enum ir_node_type ignore = ir_type_unset) const;

   /**
    * Get the variable that is ultimately referenced by an r-value
    */
   virtual ir_variable *variable_referenced() const
   {
      return this->array->variable_referenced();
   }

   virtual int precision() const
   {
      ir_dereference *deref = this->array->as_dereference();

      if (deref == NULL)
         return GLSL_PRECISION_NONE;
      else
         return deref->precision();
   }

   virtual void accept(ir_visitor *v)
   {
      v->visit(this);
   }

   virtual ir_visitor_status accept(ir_hierarchical_visitor *);

   ir_rvalue *array;
   ir_rvalue *array_index;

private:
   void set_array(ir_rvalue *value);
};


class ir_dereference_record : public ir_dereference {
public:
   ir_dereference_record(ir_rvalue *value, const char *field);

   ir_dereference_record(ir_variable *var, const char *field);

   virtual ir_dereference_record *clone(void *mem_ctx,
					struct hash_table *) const;

   virtual ir_constant *constant_expression_value(void *mem_ctx,
                                                  struct hash_table *variable_context = NULL);

   /**
    * Get the variable that is ultimately referenced by an r-value
    */
   virtual ir_variable *variable_referenced() const
   {
      return this->record->variable_referenced();
   }

   virtual int precision() const
   {
      glsl_struct_field *field = record->type->fields.structure + field_idx;

      return field->precision;
   }

   virtual void accept(ir_visitor *v)
   {
      v->visit(this);
   }

   virtual ir_visitor_status accept(ir_hierarchical_visitor *);

   ir_rvalue *record;
   int field_idx;
};


/**
 * Data stored in an ir_constant
 */
union ir_constant_data {
      unsigned u[16];
      int i[16];
      float f[16];
      bool b[16];
      double d[16];
      uint16_t f16[16];
      uint16_t u16[16];
      int16_t i16[16];
      uint64_t u64[16];
      int64_t i64[16];
};


class ir_constant : public ir_rvalue {
public:
   ir_constant(const struct glsl_type *type, const ir_constant_data *data);
   ir_constant(bool b, unsigned vector_elements=1);
   ir_constant(int16_t i16, unsigned vector_elements=1);
   ir_constant(uint16_t u16, unsigned vector_elements=1);
   ir_constant(unsigned int u, unsigned vector_elements=1);
   ir_constant(int i, unsigned vector_elements=1);
   ir_constant(float16_t f16, unsigned vector_elements=1);
   ir_constant(float f, unsigned vector_elements=1);
   ir_constant(double d, unsigned vector_elements=1);
   ir_constant(uint64_t u64, unsigned vector_elements=1);
   ir_constant(int64_t i64, unsigned vector_elements=1);

   /**
    * Construct an ir_constant from a list of ir_constant values
    */
   ir_constant(const struct glsl_type *type, exec_list *values);

   /**
    * Construct an ir_constant from a scalar component of another ir_constant
    *
    * The new \c ir_constant inherits the type of the component from the
    * source constant.
    *
    * \note
    * In the case of a matrix constant, the new constant is a scalar, \b not
    * a vector.
    */
   ir_constant(const ir_constant *c, unsigned i);

   /**
    * Return a new ir_constant of the specified type containing all zeros.
    */
   static ir_constant *zero(void *mem_ctx, const glsl_type *type);

   virtual ir_constant *clone(void *mem_ctx, struct hash_table *) const;

   virtual ir_constant *constant_expression_value(void *mem_ctx,
                                                  struct hash_table *variable_context = NULL);

   virtual void accept(ir_visitor *v)
   {
      v->visit(this);
   }

   virtual ir_visitor_status accept(ir_hierarchical_visitor *);

   virtual bool equals(const ir_instruction *ir,
                       enum ir_node_type ignore = ir_type_unset) const;

   /**
    * Get a particular component of a constant as a specific type
    *
    * This is useful, for example, to get a value from an integer constant
    * as a float or bool.  This appears frequently when constructors are
    * called with all constant parameters.
    */
   /*@{*/
   bool get_bool_component(unsigned i) const;
   float get_float_component(unsigned i) const;
   uint16_t get_float16_component(unsigned i) const;
   double get_double_component(unsigned i) const;
   int16_t get_int16_component(unsigned i) const;
   uint16_t get_uint16_component(unsigned i) const;
   int get_int_component(unsigned i) const;
   unsigned get_uint_component(unsigned i) const;
   int64_t get_int64_component(unsigned i) const;
   uint64_t get_uint64_component(unsigned i) const;
   /*@}*/

   ir_constant *get_array_element(unsigned i) const;

   ir_constant *get_record_field(int idx);

   /**
    * Copy the values on another constant at a given offset.
    *
    * The offset is ignored for array or struct copies, it's only for
    * scalars or vectors into vectors or matrices.
    *
    * With identical types on both sides and zero offset it's clone()
    * without creating a new object.
    */

   void copy_offset(ir_constant *src, int offset);

   /**
    * Copy the values on another constant at a given offset and
    * following an assign-like mask.
    *
    * The mask is ignored for scalars.
    *
    * Note that this function only handles what assign can handle,
    * i.e. at most a vector as source and a column of a matrix as
    * destination.
    */

   void copy_masked_offset(ir_constant *src, int offset, unsigned int mask);

   /**
    * Determine whether a constant has the same value as another constant
    *
    * \sa ir_constant::is_zero, ir_constant::is_one,
    * ir_constant::is_negative_one
    */
   bool has_value(const ir_constant *) const;

   /**
    * Return true if this ir_constant represents the given value.
    *
    * For vectors, this checks that each component is the given value.
    */
   virtual bool is_value(float f, int i) const;
   virtual bool is_zero() const;
   virtual bool is_one() const;
   virtual bool is_negative_one() const;

   /**
    * Return true for constants that could be stored as 16-bit unsigned values.
    *
    * Note that this will return true even for signed integer ir_constants, as
    * long as the value is non-negative and fits in 16-bits.
    */
   virtual bool is_uint16_constant() const;

   /**
    * Value of the constant.
    *
    * The field used to back the values supplied by the constant is determined
    * by the type associated with the \c ir_instruction.  Constants may be
    * scalars, vectors, or matrices.
    */
   union ir_constant_data value;

   /* Array elements and structure fields */
   ir_constant **const_elements;

private:
   /**
    * Parameterless constructor only used by the clone method
    */
   ir_constant(void);
};

/**
 * IR instruction to emit a vertex in a geometry shader.
 */
class ir_emit_vertex : public ir_instruction {
public:
   ir_emit_vertex(ir_rvalue *stream)
      : ir_instruction(ir_type_emit_vertex),
        stream(stream)
   {
      assert(stream);
   }

   virtual void accept(ir_visitor *v)
   {
      v->visit(this);
   }

   virtual ir_emit_vertex *clone(void *mem_ctx, struct hash_table *ht) const
   {
      return new(mem_ctx) ir_emit_vertex(this->stream->clone(mem_ctx, ht));
   }

   virtual ir_visitor_status accept(ir_hierarchical_visitor *);

   int stream_id() const
   {
      return stream->as_constant()->value.i[0];
   }

   ir_rvalue *stream;
};

/**
 * IR instruction to complete the current primitive and start a new one in a
 * geometry shader.
 */
class ir_end_primitive : public ir_instruction {
public:
   ir_end_primitive(ir_rvalue *stream)
      : ir_instruction(ir_type_end_primitive),
        stream(stream)
   {
      assert(stream);
   }

   virtual void accept(ir_visitor *v)
   {
      v->visit(this);
   }

   virtual ir_end_primitive *clone(void *mem_ctx, struct hash_table *ht) const
   {
      return new(mem_ctx) ir_end_primitive(this->stream->clone(mem_ctx, ht));
   }

   virtual ir_visitor_status accept(ir_hierarchical_visitor *);

   int stream_id() const
   {
      return stream->as_constant()->value.i[0];
   }

   ir_rvalue *stream;
};

/**
 * IR instruction for tessellation control and compute shader barrier.
 */
class ir_barrier : public ir_instruction {
public:
   ir_barrier()
      : ir_instruction(ir_type_barrier)
   {
   }

   virtual void accept(ir_visitor *v)
   {
      v->visit(this);
   }

   virtual ir_barrier *clone(void *mem_ctx, struct hash_table *) const
   {
      return new(mem_ctx) ir_barrier();
   }

   virtual ir_visitor_status accept(ir_hierarchical_visitor *);
};

/*@}*/

/**
 * Apply a visitor to each IR node in a list
 */
void
visit_exec_list(exec_list *list, ir_visitor *visitor);

/**
 * Validate invariants on each IR node in a list
 */
void validate_ir_tree(exec_list *instructions);

struct _mesa_glsl_parse_state;
struct gl_shader_program;

/**
 * Detect whether an unlinked shader contains static recursion
 *
 * If the list of instructions is determined to contain static recursion,
 * \c _mesa_glsl_error will be called to emit error messages for each function
 * that is in the recursion cycle.
 */
void
detect_recursion_unlinked(struct _mesa_glsl_parse_state *state,
			  exec_list *instructions);

/**
 * Detect whether a linked shader contains static recursion
 *
 * If the list of instructions is determined to contain static recursion,
 * \c link_error_printf will be called to emit error messages for each function
 * that is in the recursion cycle.  In addition,
 * \c gl_shader_program::LinkStatus will be set to false.
 */
void
detect_recursion_linked(struct gl_shader_program *prog,
			exec_list *instructions);

/**
 * Make a clone of each IR instruction in a list
 *
 * \param in   List of IR instructions that are to be cloned
 * \param out  List to hold the cloned instructions
 */
void
clone_ir_list(void *mem_ctx, exec_list *out, const exec_list *in);

extern void
_mesa_glsl_initialize_variables(exec_list *instructions,
				struct _mesa_glsl_parse_state *state);

extern void
reparent_ir(exec_list *list, void *mem_ctx);

extern char *
prototype_string(const glsl_type *return_type, const char *name,
		 exec_list *parameters);

const char *
mode_string(const ir_variable *var);

/**
 * Built-in / reserved GL variables names start with "gl_"
 */
static inline bool
is_gl_identifier(const char *s)
{
   return s && s[0] == 'g' && s[1] == 'l' && s[2] == '_';
}

extern "C" {
#endif /* __cplusplus */

extern void _mesa_print_ir(FILE *f, struct exec_list *instructions,
                           struct _mesa_glsl_parse_state *state);

extern void
fprint_ir(FILE *f, const void *instruction);

extern const struct gl_builtin_uniform_desc *
_mesa_glsl_get_builtin_uniform_desc(const char *name);

#ifdef __cplusplus
} /* extern "C" */
#endif

unsigned
vertices_per_prim(GLenum prim);

#endif /* IR_H */
