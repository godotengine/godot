/*
 * Copyright © 2009 Intel Corporation
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

#ifndef PROGRAM_PARSER_H
#define PROGRAM_PARSER_H

#include "main/config.h"
#include "program/prog_instruction.h"
#include "program/prog_parameter.h"

struct gl_context;

enum asm_type {
   at_none,
   at_address,
   at_attrib,
   at_param,
   at_temp,
   at_output
};

struct asm_symbol {
   struct asm_symbol *next;    /**< List linkage for freeing. */
   const char *name;
   enum asm_type type;
   unsigned attrib_binding;
   unsigned output_binding;   /**< Output / result register number. */

   /**
    * One of PROGRAM_STATE_VAR or PROGRAM_CONSTANT.
    */
   unsigned param_binding_type;

   /** 
    * Offset into the program_parameter_list where the tokens representing our
    * bound state (or constants) start.
    */
   unsigned param_binding_begin;

   /**
    * Constants put into the parameter list may be swizzled.  This
    * field contain's the symbol's swizzle. (SWIZZLE_X/Y/Z/W)
    */
   unsigned param_binding_swizzle;

   /* This is how many entries in the program_parameter_list we take up
    * with our state tokens or constants. Note that this is _not_ the same as
    * the number of param registers we eventually use.
    */
   unsigned param_binding_length;

   /**
    * Index of the temp register assigned to this variable.
    */
   unsigned temp_binding;

   /**
    * Flag whether or not a PARAM is an array
    */
   unsigned param_is_array:1;


   /**
    * Flag whether or not a PARAM array is accessed indirectly
    */
   unsigned param_accessed_indirectly:1;


   /**
    * \brief Is first pass of parameter layout done with this variable?
    *
    * The parameter layout routine operates in two passes.  This flag tracks
    * whether or not the first pass has handled this variable.
    *
    * \sa _mesa_layout_parameters
    */
   unsigned pass1_done:1;
};


struct asm_vector {
   unsigned count;
   gl_constant_value data[4];
};


struct asm_swizzle_mask {
   unsigned swizzle:12;
   unsigned mask:4;
};


struct asm_src_register {
   struct prog_src_register Base;

   /**
    * Symbol associated with indirect access to parameter arrays.
    *
    * If \c Base::RelAddr is 1, this will point to the symbol for the parameter
    * that is being dereferenced.  Further, \c Base::Index will be the offset
    * from the address register being used.
    */
   struct asm_symbol *Symbol;
};


struct asm_instruction {
   struct prog_instruction Base;
   struct asm_instruction *next;
   struct asm_src_register SrcReg[3];
};


struct asm_parser_state {
   struct gl_context *ctx;
   struct gl_program *prog;

   /** Memory context to attach instructions to. */
   void *mem_ctx;

   /**
    * Per-program target limits
    */
   struct gl_program_constants *limits;

   struct _mesa_symbol_table *st;

   /**
    * Linked list of symbols
    *
    * This list is \b only used when cleaning up compiler state and freeing
    * memory.
    */
   struct asm_symbol *sym;

   /**
    * State for the lexer.
    */
   void *scanner;

   /**
    * Linked list of instructions generated during parsing.
    */
   /*@{*/
   struct asm_instruction *inst_head;
   struct asm_instruction *inst_tail;
   /*@}*/


   /**
    * Selected limits copied from gl_constants
    *
    * These are limits from the GL context, but various bits in the program
    * must be validated against these values.
    */
   /*@{*/
   unsigned MaxTextureCoordUnits;
   unsigned MaxTextureImageUnits;
   unsigned MaxTextureUnits;
   unsigned MaxClipPlanes;
   unsigned MaxLights;
   unsigned MaxProgramMatrices;
   unsigned MaxDrawBuffers;
   /*@}*/

   /**
    * Value to use in state vector accessors for environment and local
    * parameters
    */
   unsigned state_param_enum_env;
   unsigned state_param_enum_local;


   /**
    * Input attributes bound to specific names
    *
    * This is only needed so that errors can be properly produced when
    * multiple ATTRIB statements bind illegal combinations of vertex
    * attributes.
    */
   GLbitfield64 InputsBound;

   enum {
      invalid_mode = 0,
      ARB_vertex,
      ARB_fragment
   } mode;

   struct {
      unsigned PositionInvariant:1;
      unsigned Fog:2;
      unsigned PrecisionHint:2;
      unsigned DrawBuffers:1;
      unsigned Shadow:1;
      unsigned TexRect:1;
      unsigned TexArray:1;
      unsigned OriginUpperLeft:1;
      unsigned PixelCenterInteger:1;
   } option;

   struct {
      unsigned UsesKill:1;
   } fragment;
};

#define OPTION_NONE        0
#define OPTION_FOG_EXP     1
#define OPTION_FOG_EXP2    2
#define OPTION_FOG_LINEAR  3
#define OPTION_NICEST      1
#define OPTION_FASTEST     2

typedef struct YYLTYPE {
   int first_line;
   int first_column;
   int last_line;
   int last_column;
   int position;
} YYLTYPE;

#define YYLTYPE_IS_DECLARED 1
#define YYLTYPE_IS_TRIVIAL 1


extern GLboolean _mesa_parse_arb_program(struct gl_context *ctx, GLenum target,
    const GLubyte *str, GLsizei len, struct asm_parser_state *state);



/* From program_lexer.l. */
extern void _mesa_program_lexer_dtor(void *scanner);

extern void _mesa_program_lexer_ctor(void **scanner,
    struct asm_parser_state *state, const char *string, size_t len);


/**
 *\name From program_parse_extra.c
 */
/*@{*/

/**
 * Parses and processes an option string to an ARB vertex program
 *
 * \return
 * Non-zero on success, zero on failure.
 */
extern int _mesa_ARBvp_parse_option(struct asm_parser_state *state,
    const char *option);

/**
 * Parses and processes an option string to an ARB fragment program
 *
 * \return
 * Non-zero on success, zero on failure.
 */
extern int _mesa_ARBfp_parse_option(struct asm_parser_state *state,
    const char *option);

/**
 * Parses and processes instruction suffixes
 *
 * Instruction suffixes, such as \c _SAT, are processed.  The relevant bits
 * are set in \c inst.  If suffixes are encountered that are either not known
 * or not supported by the modes and options set in \c state, zero will be
 * returned.
 *
 * \return
 * Non-zero on success, zero on failure.
 */
extern int _mesa_parse_instruction_suffix(const struct asm_parser_state *state,
    const char *suffix, struct prog_instruction *inst);

/*@}*/

#endif /* PROGRAM_PARSER_H */
