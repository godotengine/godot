/*
 * Mesa 3-D graphics library ATI Fragment Shader
 *
 * Copyright (C) 2004  David Airlie   All Rights Reserved.
 *
 */

#ifndef ATIFRAGSHADER_H
#define ATIFRAGSHADER_H

#include "util/glheader.h"


struct gl_context;

#define MAX_NUM_INSTRUCTIONS_PER_PASS_ATI 8
#define MAX_NUM_PASSES_ATI                2
#define MAX_NUM_FRAGMENT_REGISTERS_ATI    6
#define MAX_NUM_FRAGMENT_CONSTANTS_ATI    8

struct ati_fs_opcode_st
{
   GLenum opcode;
   GLint num_src_args;
};

struct atifragshader_src_register
{
   GLuint Index;
   GLuint argRep;
   GLuint argMod;
};

struct atifragshader_dst_register
{
   GLuint Index;
   GLuint dstMod;
   GLuint dstMask;
};

#define ATI_FRAGMENT_SHADER_COLOR_OP 0
#define ATI_FRAGMENT_SHADER_ALPHA_OP 1
#define ATI_FRAGMENT_SHADER_PASS_OP  2
#define ATI_FRAGMENT_SHADER_SAMPLE_OP 3

/* two opcodes - one for color/one for alpha */
/* up to three source registers for most ops */
struct atifs_instruction
{
   GLenum Opcode[2];
   GLuint ArgCount[2];
   struct atifragshader_src_register SrcReg[2][3];
   struct atifragshader_dst_register DstReg[2];
};

/* different from arithmetic shader instruction */
struct atifs_setupinst
{
   GLenum Opcode;
   GLuint src;
   GLenum swizzle;
};


extern struct ati_fragment_shader *
_mesa_new_ati_fragment_shader(struct gl_context *ctx, GLuint id);

extern void
_mesa_delete_ati_fragment_shader(struct gl_context *ctx,
                                 struct ati_fragment_shader *s);

#endif /* ATIFRAGSHADER_H */
