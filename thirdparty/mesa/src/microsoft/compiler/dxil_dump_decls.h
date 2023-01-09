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

#ifndef DIXL_DUMP_DECL
#error This header can only be included from dxil_dump.c
#endif

#include "dxil_module.h"

static void
dump_metadata(struct dxil_dumper *buf, struct dxil_module *m);
static void
dump_shader_info(struct dxil_dumper *buf, struct dxil_shader_info *info);
static const char *
dump_shader_string(enum dxil_shader_kind kind);
static void
dump_features(struct _mesa_string_buffer *buf, struct dxil_features *feat);
static void
dump_types(struct dxil_dumper *buf, struct list_head *list);
static void
dump_gvars(struct dxil_dumper *buf, struct list_head *list);
static void
dump_constants(struct dxil_dumper *buf, struct list_head *list);
static void
dump_funcs(struct dxil_dumper *buf, struct list_head *list);
static void
dump_attr_set_list(struct dxil_dumper *buf, struct list_head *list);
static void
dump_instrs(struct dxil_dumper *buf, struct list_head *list);
static void
dump_mdnodes(struct dxil_dumper *buf, struct list_head *list);
static void
dump_mdnode(struct dxil_dumper *d, const struct dxil_mdnode *node);
static void
dump_named_nodes(struct dxil_dumper *d, struct list_head *list);
static void
dump_type(struct dxil_dumper *buf, const struct dxil_type *type);
static void
dump_instr_binop(struct dxil_dumper *d, struct dxil_instr_binop *binop);
static void
dump_instr_cmp(struct dxil_dumper *d, struct dxil_instr_cmp *cmp);
static void
dump_instr_select(struct dxil_dumper *d, struct dxil_instr_select *select);
static void
dump_instr_cast(struct dxil_dumper *d, struct dxil_instr_cast *cast);
static void
dump_instr_call(struct dxil_dumper *d, struct dxil_instr_call *call);
static void
dump_instr_ret(struct dxil_dumper *d, struct dxil_instr_ret *ret);
static void
dump_instr_extractval(struct dxil_dumper *d, struct dxil_instr_extractval *ret);
static void
dump_instr_branch(struct dxil_dumper *d, struct dxil_instr_br *br);
static void
dump_instr_phi(struct dxil_dumper *d, struct dxil_instr_phi *phi);
static void
dump_instr_alloca(struct dxil_dumper *d, struct dxil_instr_alloca *alloca);
static void
dump_instr_gep(struct dxil_dumper *d, struct dxil_instr_gep *gep);
static void
dump_instr_load(struct dxil_dumper *d, struct dxil_instr_load *store);
static void
dump_instr_store(struct dxil_dumper *d, struct dxil_instr_store *store);
static void
dump_instr_atomicrmw(struct dxil_dumper *d, struct dxil_instr_atomicrmw *rmw);

static void
dump_instr_print_operands(struct dxil_dumper *d, int num,
                          const struct dxil_value *val[]);

static void dump_io_signatures(struct _mesa_string_buffer *buf,
                               struct dxil_module *m);
static void
dump_io_signature(struct _mesa_string_buffer *buf, unsigned num,
                  struct dxil_signature_record *io);

static const char *component_type_as_string(uint32_t type);

static void dump_psv(struct _mesa_string_buffer *buf,
                     struct dxil_module *m);
static void dump_psv_io(struct _mesa_string_buffer *buf, struct dxil_module *m,
                        unsigned num, struct dxil_psv_signature_element *io);

static void
dump_value(struct dxil_dumper *d, const struct dxil_value *val);

static const char *binop_strings[DXIL_BINOP_INSTR_COUNT] = {
   [DXIL_BINOP_ADD] = "add",
   [DXIL_BINOP_SUB] = "sub",
   [DXIL_BINOP_MUL] = "mul",
   [DXIL_BINOP_UDIV] = "udiv",
   [DXIL_BINOP_SDIV] = "sdiv",
   [DXIL_BINOP_UREM] = "urem",
   [DXIL_BINOP_SREM] = "srem",
   [DXIL_BINOP_SHL] = "shl",
   [DXIL_BINOP_LSHR] = "lshr",
   [DXIL_BINOP_ASHR] = "ashr",
   [DXIL_BINOP_AND] = "and",
   [DXIL_BINOP_OR] = "or",
   [DXIL_BINOP_XOR]= "xor"
};

static const char *pred_strings[DXIL_CMP_INSTR_COUNT] = {
   [DXIL_FCMP_FALSE] = "FALSE",
   [DXIL_FCMP_OEQ] = "ord-fEQ",
   [DXIL_FCMP_OGT] = "ord-fGT",
   [DXIL_FCMP_OGE] = "ord-fGE",
   [DXIL_FCMP_OLT] = "ord-fLT",
   [DXIL_FCMP_OLE] = "ord-fLE",
   [DXIL_FCMP_ONE] = "ord-fNE",
   [DXIL_FCMP_ORD] = "ord-fRD",
   [DXIL_FCMP_UNO] = "unord-fNO",
   [DXIL_FCMP_UEQ] = "unord-fEQ",
   [DXIL_FCMP_UGT] = "unord-fGT",
   [DXIL_FCMP_UGE] = "unord-fGE",
   [DXIL_FCMP_ULT] = "unord-fLT",
   [DXIL_FCMP_ULE] = "unord-fLE",
   [DXIL_FCMP_UNE] = "unord-fNE",
   [DXIL_FCMP_TRUE] = "TRUE",
   [DXIL_ICMP_EQ] = "iEQ",
   [DXIL_ICMP_NE] = "iNE",
   [DXIL_ICMP_UGT] = "uiGT",
   [DXIL_ICMP_UGE] = "uiGE",
   [DXIL_ICMP_ULT] = "uiLT",
   [DXIL_ICMP_ULE] = "uiLE",
   [DXIL_ICMP_SGT] = "iGT",
   [DXIL_ICMP_SGE] = "iGE",
   [DXIL_ICMP_SLT] = "iLT",
   [DXIL_ICMP_SLE] = "iLE"
};

static const char *cast_opcode_strings[DXIL_CAST_INSTR_COUNT] = {
   [DXIL_CAST_TRUNC] = "trunc",
   [DXIL_CAST_ZEXT] = "zext",
   [DXIL_CAST_SEXT] = "sext",
   [DXIL_CAST_FPTOUI] = "ftoui",
   [DXIL_CAST_FPTOSI] = "ftoi",
   [DXIL_CAST_UITOFP] = "uitof",
   [DXIL_CAST_SITOFP] = "itof",
   [DXIL_CAST_FPTRUNC] = "ftrunc",
   [DXIL_CAST_FPEXT] = "fext",
   [DXIL_CAST_PTRTOINT] = "ptrtoint",
   [DXIL_CAST_INTTOPTR] = "inttoptr",
   [DXIL_CAST_BITCAST] = "bitcast",
   [DXIL_CAST_ADDRSPACECAST] = "addrspacecast",
};

static const char *dxil_type_strings[DXIL_PROG_SIG_COMP_TYPE_COUNT] = {
   [DXIL_PROG_SIG_COMP_TYPE_UNKNOWN] = "unknown",
   [DXIL_PROG_SIG_COMP_TYPE_UINT32] = "uint32",
   [DXIL_PROG_SIG_COMP_TYPE_SINT32] = "int32",
   [DXIL_PROG_SIG_COMP_TYPE_FLOAT32] = "float32",
   [DXIL_PROG_SIG_COMP_TYPE_UINT16] = "uint16",
   [DXIL_PROG_SIG_COMP_TYPE_SINT16] = "int16",
   [DXIL_PROG_SIG_COMP_TYPE_FLOAT16] = "float16",
   [DXIL_PROG_SIG_COMP_TYPE_UINT64] = "uint64",
   [DXIL_PROG_SIG_COMP_TYPE_SINT64] = "int64",
   [DXIL_PROG_SIG_COMP_TYPE_FLOAT64] = "float64"
};
