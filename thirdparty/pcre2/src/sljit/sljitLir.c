/*
 *    Stack-less Just-In-Time compiler
 *
 *    Copyright Zoltan Herczeg (hzmester@freemail.hu). All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are
 * permitted provided that the following conditions are met:
 *
 *   1. Redistributions of source code must retain the above copyright notice, this list of
 *      conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above copyright notice, this list
 *      of conditions and the following disclaimer in the documentation and/or other materials
 *      provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER(S) AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDER(S) OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "sljitLir.h"

#ifdef _WIN32

#include <windows.h>

#endif /* _WIN32 */

#if !(defined SLJIT_STD_MACROS_DEFINED && SLJIT_STD_MACROS_DEFINED)

/* These libraries are needed for the macros below. */
#include <stdlib.h>
#include <string.h>

#endif /* SLJIT_STD_MACROS_DEFINED */

#define CHECK_ERROR() \
	do { \
		if (SLJIT_UNLIKELY(compiler->error)) \
			return compiler->error; \
	} while (0)

#define CHECK_ERROR_PTR() \
	do { \
		if (SLJIT_UNLIKELY(compiler->error)) \
			return NULL; \
	} while (0)

#define FAIL_IF(expr) \
	do { \
		if (SLJIT_UNLIKELY(expr)) \
			return compiler->error; \
	} while (0)

#define PTR_FAIL_IF(expr) \
	do { \
		if (SLJIT_UNLIKELY(expr)) \
			return NULL; \
	} while (0)

#define FAIL_IF_NULL(ptr) \
	do { \
		if (SLJIT_UNLIKELY(!(ptr))) { \
			compiler->error = SLJIT_ERR_ALLOC_FAILED; \
			return SLJIT_ERR_ALLOC_FAILED; \
		} \
	} while (0)

#define PTR_FAIL_IF_NULL(ptr) \
	do { \
		if (SLJIT_UNLIKELY(!(ptr))) { \
			compiler->error = SLJIT_ERR_ALLOC_FAILED; \
			return NULL; \
		} \
	} while (0)

#define PTR_FAIL_WITH_EXEC_IF(ptr) \
	do { \
		if (SLJIT_UNLIKELY(!(ptr))) { \
			compiler->error = SLJIT_ERR_EX_ALLOC_FAILED; \
			return NULL; \
		} \
	} while (0)

#if !(defined SLJIT_CONFIG_UNSUPPORTED && SLJIT_CONFIG_UNSUPPORTED)

#define SSIZE_OF(type) ((sljit_s32)sizeof(sljit_ ## type))

#define VARIABLE_FLAG_SHIFT (10)
#define VARIABLE_FLAG_MASK (0x3f << VARIABLE_FLAG_SHIFT)
#define GET_FLAG_TYPE(op) ((op) >> VARIABLE_FLAG_SHIFT)

#define GET_OPCODE(op) \
	((op) & ~(SLJIT_32 | SLJIT_SET_Z | VARIABLE_FLAG_MASK))

#define HAS_FLAGS(op) \
	((op) & (SLJIT_SET_Z | VARIABLE_FLAG_MASK))

#define GET_ALL_FLAGS(op) \
	((op) & (SLJIT_32 | SLJIT_SET_Z | VARIABLE_FLAG_MASK))

#if (defined SLJIT_64BIT_ARCHITECTURE && SLJIT_64BIT_ARCHITECTURE)
#define TYPE_CAST_NEEDED(op) \
	((op) >= SLJIT_MOV_U8 && (op) <= SLJIT_MOV_S32)
#else /* !SLJIT_64BIT_ARCHITECTURE */
#define TYPE_CAST_NEEDED(op) \
	((op) >= SLJIT_MOV_U8 && (op) <= SLJIT_MOV_S16)
#endif /* SLJIT_64BIT_ARCHITECTURE */

#define BUF_SIZE	4096

#if (defined SLJIT_32BIT_ARCHITECTURE && SLJIT_32BIT_ARCHITECTURE)
#define ABUF_SIZE	2048
#else
#define ABUF_SIZE	4096
#endif

/* Parameter parsing. */
#define REG_MASK		0x3f
#define OFFS_REG(reg)		(((reg) >> 8) & REG_MASK)
#define OFFS_REG_MASK		(REG_MASK << 8)
#define TO_OFFS_REG(reg)	((reg) << 8)
/* When reg cannot be unused. */
#define FAST_IS_REG(reg)	((reg) <= REG_MASK)

/* Mask for argument types. */
#define SLJIT_ARG_MASK		0x7
#define SLJIT_ARG_FULL_MASK	(SLJIT_ARG_MASK | SLJIT_ARG_TYPE_SCRATCH_REG)

/* Mask for sljit_emit_mem. */
#define REG_PAIR_MASK		0xff00
#define REG_PAIR_FIRST(reg)	((reg) & 0xff)
#define REG_PAIR_SECOND(reg)	((reg) >> 8)

/* Mask for sljit_emit_enter. */
#define SLJIT_KEPT_SAVEDS_COUNT(options) ((options) & 0x3)

/* Jump flags. */
#define JUMP_LABEL	0x1
#define JUMP_ADDR	0x2
/* SLJIT_REWRITABLE_JUMP is 0x1000. */

#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86)
#	define PATCH_MB		0x4
#	define PATCH_MW		0x8
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
#	define PATCH_MD		0x10
#endif
#	define TYPE_SHIFT	13
#endif /* SLJIT_CONFIG_X86 */

#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5) || (defined SLJIT_CONFIG_ARM_V7 && SLJIT_CONFIG_ARM_V7)
#	define IS_BL		0x4
#	define PATCH_B		0x8
#endif /* SLJIT_CONFIG_ARM_V5 || SLJIT_CONFIG_ARM_V7 */

#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
#	define CPOOL_SIZE	512
#endif /* SLJIT_CONFIG_ARM_V5 */

#if (defined SLJIT_CONFIG_ARM_THUMB2 && SLJIT_CONFIG_ARM_THUMB2)
#	define IS_COND		0x04
#	define IS_BL		0x08
	/* conditional + imm8 */
#	define PATCH_TYPE1	0x10
	/* conditional + imm20 */
#	define PATCH_TYPE2	0x20
	/* IT + imm24 */
#	define PATCH_TYPE3	0x30
	/* imm11 */
#	define PATCH_TYPE4	0x40
	/* imm24 */
#	define PATCH_TYPE5	0x50
	/* BL + imm24 */
#	define PATCH_BL		0x60
	/* 0xf00 cc code for branches */
#endif /* SLJIT_CONFIG_ARM_THUMB2 */

#if (defined SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64)
#	define IS_COND		0x004
#	define IS_CBZ		0x008
#	define IS_BL		0x010
#	define PATCH_B		0x020
#	define PATCH_COND	0x040
#	define PATCH_ABS48	0x080
#	define PATCH_ABS64	0x100
#endif /* SLJIT_CONFIG_ARM_64 */

#if (defined SLJIT_CONFIG_PPC && SLJIT_CONFIG_PPC)
#	define IS_COND		0x004
#	define IS_CALL		0x008
#	define PATCH_B		0x010
#	define PATCH_ABS_B	0x020
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
#	define PATCH_ABS32	0x040
#	define PATCH_ABS48	0x080
#endif /* SLJIT_CONFIG_PPC_64 */
#	define REMOVE_COND	0x100
#endif /* SLJIT_CONFIG_PPC */

#if (defined SLJIT_CONFIG_MIPS && SLJIT_CONFIG_MIPS)
#	define IS_MOVABLE	0x004
#	define IS_JAL		0x008
#	define IS_CALL		0x010
#	define IS_BIT26_COND	0x020
#	define IS_BIT16_COND	0x040
#	define IS_BIT23_COND	0x080

#	define IS_COND		(IS_BIT26_COND | IS_BIT16_COND | IS_BIT23_COND)

#	define PATCH_B		0x100
#	define PATCH_J		0x200

#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
#	define PATCH_ABS32	0x400
#	define PATCH_ABS48	0x800
#endif /* SLJIT_CONFIG_MIPS_64 */

	/* instruction types */
#	define MOVABLE_INS	0
	/* 1 - 31 last destination register */
	/* no destination (i.e: store) */
#	define UNMOVABLE_INS	32
	/* FPU status register */
#	define FCSR_FCC		33
#endif /* SLJIT_CONFIG_MIPS */

#if (defined SLJIT_CONFIG_RISCV && SLJIT_CONFIG_RISCV)
#	define IS_COND		0x004
#	define IS_CALL		0x008

#	define PATCH_B		0x010
#	define PATCH_J		0x020

#if (defined SLJIT_CONFIG_RISCV_64 && SLJIT_CONFIG_RISCV_64)
#	define PATCH_REL32	0x040
#	define PATCH_ABS32	0x080
#	define PATCH_ABS44	0x100
#	define PATCH_ABS52	0x200
#else /* !SLJIT_CONFIG_RISCV_64 */
#	define PATCH_REL32	0x0
#endif /* SLJIT_CONFIG_RISCV_64 */
#endif /* SLJIT_CONFIG_RISCV */

/* Stack management. */

#define GET_SAVED_REGISTERS_SIZE(scratches, saveds, extra) \
	(((scratches < SLJIT_NUMBER_OF_SCRATCH_REGISTERS ? 0 : (scratches - SLJIT_NUMBER_OF_SCRATCH_REGISTERS)) + \
		(saveds) + (sljit_s32)(extra)) * (sljit_s32)sizeof(sljit_sw))

#define GET_SAVED_FLOAT_REGISTERS_SIZE(fscratches, fsaveds, size) \
	(((fscratches < SLJIT_NUMBER_OF_SCRATCH_FLOAT_REGISTERS ? 0 : (fscratches - SLJIT_NUMBER_OF_SCRATCH_FLOAT_REGISTERS)) + \
		(fsaveds)) * (sljit_s32)(size))

#define ADJUST_LOCAL_OFFSET(p, i) \
	if ((p) == (SLJIT_MEM1(SLJIT_SP))) \
		(i) += SLJIT_LOCALS_OFFSET;

#endif /* !(defined SLJIT_CONFIG_UNSUPPORTED && SLJIT_CONFIG_UNSUPPORTED) */

/* Utils can still be used even if SLJIT_CONFIG_UNSUPPORTED is set. */
#include "sljitUtils.c"

#if !(defined SLJIT_CONFIG_UNSUPPORTED && SLJIT_CONFIG_UNSUPPORTED)

#if (defined SLJIT_EXECUTABLE_ALLOCATOR && SLJIT_EXECUTABLE_ALLOCATOR)

#if (defined SLJIT_PROT_EXECUTABLE_ALLOCATOR && SLJIT_PROT_EXECUTABLE_ALLOCATOR)
#include "sljitProtExecAllocator.c"
#elif (defined SLJIT_WX_EXECUTABLE_ALLOCATOR && SLJIT_WX_EXECUTABLE_ALLOCATOR)
#include "sljitWXExecAllocator.c"
#else
#include "sljitExecAllocator.c"
#endif

#endif

#if (defined SLJIT_PROT_EXECUTABLE_ALLOCATOR && SLJIT_PROT_EXECUTABLE_ALLOCATOR)
#define SLJIT_ADD_EXEC_OFFSET(ptr, exec_offset) ((sljit_u8 *)(ptr) + (exec_offset))
#else
#define SLJIT_ADD_EXEC_OFFSET(ptr, exec_offset) ((sljit_u8 *)(ptr))
#endif

#ifndef SLJIT_UPDATE_WX_FLAGS
#define SLJIT_UPDATE_WX_FLAGS(from, to, enable_exec)
#endif

/* Argument checking features. */

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)

/* Returns with error when an invalid argument is passed. */

#define CHECK_ARGUMENT(x) \
	do { \
		if (SLJIT_UNLIKELY(!(x))) \
			return 1; \
	} while (0)

#define CHECK_RETURN_TYPE sljit_s32
#define CHECK_RETURN_OK return 0

#define CHECK(x) \
	do { \
		if (SLJIT_UNLIKELY(x)) { \
			compiler->error = SLJIT_ERR_BAD_ARGUMENT; \
			return SLJIT_ERR_BAD_ARGUMENT; \
		} \
	} while (0)

#define CHECK_PTR(x) \
	do { \
		if (SLJIT_UNLIKELY(x)) { \
			compiler->error = SLJIT_ERR_BAD_ARGUMENT; \
			return NULL; \
		} \
	} while (0)

#define CHECK_REG_INDEX(x) \
	do { \
		if (SLJIT_UNLIKELY(x)) { \
			return -2; \
		} \
	} while (0)

#elif (defined SLJIT_DEBUG && SLJIT_DEBUG)

/* Assertion failure occures if an invalid argument is passed. */
#undef SLJIT_ARGUMENT_CHECKS
#define SLJIT_ARGUMENT_CHECKS 1

#define CHECK_ARGUMENT(x) SLJIT_ASSERT(x)
#define CHECK_RETURN_TYPE void
#define CHECK_RETURN_OK return
#define CHECK(x) x
#define CHECK_PTR(x) x
#define CHECK_REG_INDEX(x) x

#elif (defined SLJIT_VERBOSE && SLJIT_VERBOSE)

/* Arguments are not checked. */
#define CHECK_RETURN_TYPE void
#define CHECK_RETURN_OK return
#define CHECK(x) x
#define CHECK_PTR(x) x
#define CHECK_REG_INDEX(x) x

#else

/* Arguments are not checked. */
#define CHECK(x)
#define CHECK_PTR(x)
#define CHECK_REG_INDEX(x)

#endif /* SLJIT_ARGUMENT_CHECKS */

/* --------------------------------------------------------------------- */
/*  Public functions                                                     */
/* --------------------------------------------------------------------- */

#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86)
#define SLJIT_NEEDS_COMPILER_INIT 1
static sljit_s32 compiler_initialized = 0;
/* A thread safe initialization. */
static void init_compiler(void);
#endif

SLJIT_API_FUNC_ATTRIBUTE struct sljit_compiler* sljit_create_compiler(void *allocator_data, void *exec_allocator_data)
{
	struct sljit_compiler *compiler = (struct sljit_compiler*)SLJIT_MALLOC(sizeof(struct sljit_compiler), allocator_data);
	if (!compiler)
		return NULL;
	SLJIT_ZEROMEM(compiler, sizeof(struct sljit_compiler));

	SLJIT_COMPILE_ASSERT(
		sizeof(sljit_s8) == 1 && sizeof(sljit_u8) == 1
		&& sizeof(sljit_s16) == 2 && sizeof(sljit_u16) == 2
		&& sizeof(sljit_s32) == 4 && sizeof(sljit_u32) == 4
		&& (sizeof(sljit_p) == 4 || sizeof(sljit_p) == 8)
		&& sizeof(sljit_p) <= sizeof(sljit_sw)
		&& (sizeof(sljit_sw) == 4 || sizeof(sljit_sw) == 8)
		&& (sizeof(sljit_uw) == 4 || sizeof(sljit_uw) == 8),
		invalid_integer_types);
	SLJIT_COMPILE_ASSERT(SLJIT_REWRITABLE_JUMP != SLJIT_32,
		rewritable_jump_and_single_op_must_not_be_the_same);
	SLJIT_COMPILE_ASSERT(!(SLJIT_EQUAL & 0x1) && !(SLJIT_LESS & 0x1) && !(SLJIT_F_EQUAL & 0x1) && !(SLJIT_JUMP & 0x1),
		conditional_flags_must_be_even_numbers);

	/* Only the non-zero members must be set. */
	compiler->error = SLJIT_SUCCESS;

	compiler->allocator_data = allocator_data;
	compiler->exec_allocator_data = exec_allocator_data;
	compiler->buf = (struct sljit_memory_fragment*)SLJIT_MALLOC(BUF_SIZE, allocator_data);
	compiler->abuf = (struct sljit_memory_fragment*)SLJIT_MALLOC(ABUF_SIZE, allocator_data);

	if (!compiler->buf || !compiler->abuf) {
		if (compiler->buf)
			SLJIT_FREE(compiler->buf, allocator_data);
		if (compiler->abuf)
			SLJIT_FREE(compiler->abuf, allocator_data);
		SLJIT_FREE(compiler, allocator_data);
		return NULL;
	}

	compiler->buf->next = NULL;
	compiler->buf->used_size = 0;
	compiler->abuf->next = NULL;
	compiler->abuf->used_size = 0;

	compiler->scratches = -1;
	compiler->saveds = -1;
	compiler->fscratches = -1;
	compiler->fsaveds = -1;
	compiler->local_size = -1;

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	compiler->args_size = -1;
#endif

#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
	compiler->cpool = (sljit_uw*)SLJIT_MALLOC(CPOOL_SIZE * sizeof(sljit_uw)
		+ CPOOL_SIZE * sizeof(sljit_u8), allocator_data);
	if (!compiler->cpool) {
		SLJIT_FREE(compiler->buf, allocator_data);
		SLJIT_FREE(compiler->abuf, allocator_data);
		SLJIT_FREE(compiler, allocator_data);
		return NULL;
	}
	compiler->cpool_unique = (sljit_u8*)(compiler->cpool + CPOOL_SIZE);
	compiler->cpool_diff = 0xffffffff;
#endif

#if (defined SLJIT_CONFIG_MIPS && SLJIT_CONFIG_MIPS)
	compiler->delay_slot = UNMOVABLE_INS;
#endif

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS) \
		|| (defined SLJIT_DEBUG && SLJIT_DEBUG)
	compiler->last_flags = 0;
	compiler->last_return = -1;
	compiler->logical_local_size = 0;
#endif

#if (defined SLJIT_NEEDS_COMPILER_INIT && SLJIT_NEEDS_COMPILER_INIT)
	if (!compiler_initialized) {
		init_compiler();
		compiler_initialized = 1;
	}
#endif

	return compiler;
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_free_compiler(struct sljit_compiler *compiler)
{
	struct sljit_memory_fragment *buf;
	struct sljit_memory_fragment *curr;
	void *allocator_data = compiler->allocator_data;
	SLJIT_UNUSED_ARG(allocator_data);

	buf = compiler->buf;
	while (buf) {
		curr = buf;
		buf = buf->next;
		SLJIT_FREE(curr, allocator_data);
	}

	buf = compiler->abuf;
	while (buf) {
		curr = buf;
		buf = buf->next;
		SLJIT_FREE(curr, allocator_data);
	}

#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
	SLJIT_FREE(compiler->cpool, allocator_data);
#endif
	SLJIT_FREE(compiler, allocator_data);
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_compiler_memory_error(struct sljit_compiler *compiler)
{
	if (compiler->error == SLJIT_SUCCESS)
		compiler->error = SLJIT_ERR_ALLOC_FAILED;
}

#if (defined SLJIT_CONFIG_ARM_THUMB2 && SLJIT_CONFIG_ARM_THUMB2)
SLJIT_API_FUNC_ATTRIBUTE void sljit_free_code(void* code, void *exec_allocator_data)
{
	SLJIT_UNUSED_ARG(exec_allocator_data);

	/* Remove thumb mode flag. */
	SLJIT_FREE_EXEC((void*)((sljit_uw)code & ~(sljit_uw)0x1), exec_allocator_data);
}
#elif (defined SLJIT_INDIRECT_CALL && SLJIT_INDIRECT_CALL)
SLJIT_API_FUNC_ATTRIBUTE void sljit_free_code(void* code, void *exec_allocator_data)
{
	SLJIT_UNUSED_ARG(exec_allocator_data);

	/* Resolve indirection. */
	code = (void*)(*(sljit_uw*)code);
	SLJIT_FREE_EXEC(code, exec_allocator_data);
}
#else
SLJIT_API_FUNC_ATTRIBUTE void sljit_free_code(void* code, void *exec_allocator_data)
{
	SLJIT_UNUSED_ARG(exec_allocator_data);

	SLJIT_FREE_EXEC(code, exec_allocator_data);
}
#endif

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_label(struct sljit_jump *jump, struct sljit_label* label)
{
	if (SLJIT_LIKELY(!!jump) && SLJIT_LIKELY(!!label)) {
		jump->flags &= (sljit_uw)~JUMP_ADDR;
		jump->flags |= JUMP_LABEL;
		jump->u.label = label;
	}
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_target(struct sljit_jump *jump, sljit_uw target)
{
	if (SLJIT_LIKELY(!!jump)) {
		jump->flags &= (sljit_uw)~JUMP_LABEL;
		jump->flags |= JUMP_ADDR;
		jump->u.target = target;
	}
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_put_label(struct sljit_put_label *put_label, struct sljit_label *label)
{
	if (SLJIT_LIKELY(!!put_label))
		put_label->label = label;
}

#define SLJIT_CURRENT_FLAGS_ALL \
	(SLJIT_CURRENT_FLAGS_32 | SLJIT_CURRENT_FLAGS_ADD | SLJIT_CURRENT_FLAGS_SUB | SLJIT_CURRENT_FLAGS_COMPARE)

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_current_flags(struct sljit_compiler *compiler, sljit_s32 current_flags)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(current_flags);

#if (defined SLJIT_HAS_STATUS_FLAGS_STATE && SLJIT_HAS_STATUS_FLAGS_STATE)
	compiler->status_flags_state = current_flags;
#endif

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	compiler->last_flags = 0;
	if ((current_flags & ~(VARIABLE_FLAG_MASK | SLJIT_SET_Z | SLJIT_CURRENT_FLAGS_ALL)) == 0) {
		compiler->last_flags = GET_FLAG_TYPE(current_flags) | (current_flags & (SLJIT_32 | SLJIT_SET_Z));
	}
#endif
}

/* --------------------------------------------------------------------- */
/*  Private functions                                                    */
/* --------------------------------------------------------------------- */

static void* ensure_buf(struct sljit_compiler *compiler, sljit_uw size)
{
	sljit_u8 *ret;
	struct sljit_memory_fragment *new_frag;

	SLJIT_ASSERT(size <= 256);
	if (compiler->buf->used_size + size <= (BUF_SIZE - (sljit_uw)SLJIT_OFFSETOF(struct sljit_memory_fragment, memory))) {
		ret = compiler->buf->memory + compiler->buf->used_size;
		compiler->buf->used_size += size;
		return ret;
	}
	new_frag = (struct sljit_memory_fragment*)SLJIT_MALLOC(BUF_SIZE, compiler->allocator_data);
	PTR_FAIL_IF_NULL(new_frag);
	new_frag->next = compiler->buf;
	compiler->buf = new_frag;
	new_frag->used_size = size;
	return new_frag->memory;
}

static void* ensure_abuf(struct sljit_compiler *compiler, sljit_uw size)
{
	sljit_u8 *ret;
	struct sljit_memory_fragment *new_frag;

	SLJIT_ASSERT(size <= 256);
	if (compiler->abuf->used_size + size <= (ABUF_SIZE - (sljit_uw)SLJIT_OFFSETOF(struct sljit_memory_fragment, memory))) {
		ret = compiler->abuf->memory + compiler->abuf->used_size;
		compiler->abuf->used_size += size;
		return ret;
	}
	new_frag = (struct sljit_memory_fragment*)SLJIT_MALLOC(ABUF_SIZE, compiler->allocator_data);
	PTR_FAIL_IF_NULL(new_frag);
	new_frag->next = compiler->abuf;
	compiler->abuf = new_frag;
	new_frag->used_size = size;
	return new_frag->memory;
}

SLJIT_API_FUNC_ATTRIBUTE void* sljit_alloc_memory(struct sljit_compiler *compiler, sljit_s32 size)
{
	CHECK_ERROR_PTR();

#if (defined SLJIT_64BIT_ARCHITECTURE && SLJIT_64BIT_ARCHITECTURE)
	if (size <= 0 || size > 128)
		return NULL;
	size = (size + 7) & ~7;
#else
	if (size <= 0 || size > 64)
		return NULL;
	size = (size + 3) & ~3;
#endif
	return ensure_abuf(compiler, (sljit_uw)size);
}

static SLJIT_INLINE void reverse_buf(struct sljit_compiler *compiler)
{
	struct sljit_memory_fragment *buf = compiler->buf;
	struct sljit_memory_fragment *prev = NULL;
	struct sljit_memory_fragment *tmp;

	do {
		tmp = buf->next;
		buf->next = prev;
		prev = buf;
		buf = tmp;
	} while (buf != NULL);

	compiler->buf = prev;
}

/* Only used in RISC architectures where the instruction size is constant */
#if !(defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86) \
	&& !(defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X)

static SLJIT_INLINE sljit_uw compute_next_addr(struct sljit_label *label, struct sljit_jump *jump,
	struct sljit_const *const_, struct sljit_put_label *put_label)
{
	sljit_uw result = ~(sljit_uw)0;

	if (label)
		result = label->size;

	if (jump && jump->addr < result)
		result = jump->addr;

	if (const_ && const_->addr < result)
		result = const_->addr;

	if (put_label && put_label->addr < result)
		result = put_label->addr;

	return result;
}

#endif /* !SLJIT_CONFIG_X86 && !SLJIT_CONFIG_S390X */

static SLJIT_INLINE void set_emit_enter(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 args, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	SLJIT_UNUSED_ARG(args);
	SLJIT_UNUSED_ARG(local_size);

	compiler->options = options;
	compiler->scratches = scratches;
	compiler->saveds = saveds;
	compiler->fscratches = fscratches;
	compiler->fsaveds = fsaveds;
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	compiler->last_return = args & SLJIT_ARG_MASK;
	compiler->logical_local_size = local_size;
#endif
}

static SLJIT_INLINE void set_set_context(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 args, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	SLJIT_UNUSED_ARG(args);
	SLJIT_UNUSED_ARG(local_size);

	compiler->options = options;
	compiler->scratches = scratches;
	compiler->saveds = saveds;
	compiler->fscratches = fscratches;
	compiler->fsaveds = fsaveds;
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	compiler->last_return = args & SLJIT_ARG_MASK;
	compiler->logical_local_size = local_size;
#endif
}

static SLJIT_INLINE void set_label(struct sljit_label *label, struct sljit_compiler *compiler)
{
	label->next = NULL;
	label->size = compiler->size;
	if (compiler->last_label)
		compiler->last_label->next = label;
	else
		compiler->labels = label;
	compiler->last_label = label;
}

static SLJIT_INLINE void set_jump(struct sljit_jump *jump, struct sljit_compiler *compiler, sljit_u32 flags)
{
	jump->next = NULL;
	jump->flags = flags;
	if (compiler->last_jump)
		compiler->last_jump->next = jump;
	else
		compiler->jumps = jump;
	compiler->last_jump = jump;
}

static SLJIT_INLINE void set_const(struct sljit_const *const_, struct sljit_compiler *compiler)
{
	const_->next = NULL;
	const_->addr = compiler->size;
	if (compiler->last_const)
		compiler->last_const->next = const_;
	else
		compiler->consts = const_;
	compiler->last_const = const_;
}

static SLJIT_INLINE void set_put_label(struct sljit_put_label *put_label, struct sljit_compiler *compiler, sljit_uw offset)
{
	put_label->next = NULL;
	put_label->label = NULL;
	put_label->addr = compiler->size - offset;
	put_label->flags = 0;
	if (compiler->last_put_label)
		compiler->last_put_label->next = put_label;
	else
		compiler->put_labels = put_label;
	compiler->last_put_label = put_label;
}

#define ADDRESSING_DEPENDS_ON(exp, reg) \
	(((exp) & SLJIT_MEM) && (((exp) & REG_MASK) == reg || OFFS_REG(exp) == reg))

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)

static sljit_s32 function_check_arguments(sljit_s32 arg_types, sljit_s32 scratches, sljit_s32 saveds, sljit_s32 fscratches)
{
	sljit_s32 word_arg_count, scratch_arg_end, saved_arg_count, float_arg_count, curr_type;

	curr_type = (arg_types & SLJIT_ARG_FULL_MASK);

	if (curr_type >= SLJIT_ARG_TYPE_F64) {
		if (curr_type > SLJIT_ARG_TYPE_F32 || fscratches == 0)
			return 0;
	} else if (curr_type >= SLJIT_ARG_TYPE_W) {
		if (scratches == 0)
			return 0;
	}

	arg_types >>= SLJIT_ARG_SHIFT;

	word_arg_count = 0;
	scratch_arg_end = 0;
	saved_arg_count = 0;
	float_arg_count = 0;
	while (arg_types != 0) {
		if (word_arg_count + float_arg_count >= 4)
			return 0;

		curr_type = (arg_types & SLJIT_ARG_MASK);

		if (arg_types & SLJIT_ARG_TYPE_SCRATCH_REG) {
			if (saveds == -1 || curr_type < SLJIT_ARG_TYPE_W || curr_type > SLJIT_ARG_TYPE_P)
				return 0;

			word_arg_count++;
			scratch_arg_end = word_arg_count;
		} else {
			if (curr_type < SLJIT_ARG_TYPE_W || curr_type > SLJIT_ARG_TYPE_F32)
				return 0;

			if (curr_type < SLJIT_ARG_TYPE_F64) {
				word_arg_count++;
				saved_arg_count++;
			} else
				float_arg_count++;
		}

		arg_types >>= SLJIT_ARG_SHIFT;
	}

	if (saveds == -1)
		return (word_arg_count <= scratches && float_arg_count <= fscratches);

	return (saved_arg_count <= saveds && scratch_arg_end <= scratches && float_arg_count <= fscratches);
}

#define FUNCTION_CHECK_IS_REG(r) \
	(((r) >= SLJIT_R0 && (r) < (SLJIT_R0 + compiler->scratches)) \
	|| ((r) > (SLJIT_S0 - compiler->saveds) && (r) <= SLJIT_S0))

#define FUNCTION_CHECK_IS_FREG(fr) \
	(((fr) >= SLJIT_FR0 && (fr) < (SLJIT_FR0 + compiler->fscratches)) \
	|| ((fr) > (SLJIT_FS0 - compiler->fsaveds) && (fr) <= SLJIT_FS0))

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
#define CHECK_IF_VIRTUAL_REGISTER(p) ((p) <= SLJIT_S3 && (p) >= SLJIT_S8)
#else
#define CHECK_IF_VIRTUAL_REGISTER(p) 0
#endif

static sljit_s32 function_check_src_mem(struct sljit_compiler *compiler, sljit_s32 p, sljit_sw i)
{
	if (compiler->scratches == -1 || compiler->saveds == -1)
		return 0;

	if (!(p & SLJIT_MEM))
		return 0;

	if (p == SLJIT_MEM1(SLJIT_SP))
		return (i >= 0 && i < compiler->logical_local_size);

	if (!(!(p & REG_MASK) || FUNCTION_CHECK_IS_REG(p & REG_MASK)))
		return 0;

	if (CHECK_IF_VIRTUAL_REGISTER(p & REG_MASK))
		return 0;

	if (p & OFFS_REG_MASK) {
		if (!(p & REG_MASK))
			return 0;

		if (!(FUNCTION_CHECK_IS_REG(OFFS_REG(p))))
			return 0;

		if (CHECK_IF_VIRTUAL_REGISTER(OFFS_REG(p)))
			return 0;

		if ((i & ~0x3) != 0)
			return 0;
	}

	return (p & ~(SLJIT_MEM | REG_MASK | OFFS_REG_MASK)) == 0;
}

#define FUNCTION_CHECK_SRC_MEM(p, i) \
	CHECK_ARGUMENT(function_check_src_mem(compiler, p, i));

static sljit_s32 function_check_src(struct sljit_compiler *compiler, sljit_s32 p, sljit_sw i)
{
	if (compiler->scratches == -1 || compiler->saveds == -1)
		return 0;

	if (FUNCTION_CHECK_IS_REG(p))
		return (i == 0);

	if (p == SLJIT_IMM)
		return 1;

	return function_check_src_mem(compiler, p, i);
}

#define FUNCTION_CHECK_SRC(p, i) \
	CHECK_ARGUMENT(function_check_src(compiler, p, i));

static sljit_s32 function_check_dst(struct sljit_compiler *compiler, sljit_s32 p, sljit_sw i)
{
	if (compiler->scratches == -1 || compiler->saveds == -1)
		return 0;

	if (FUNCTION_CHECK_IS_REG(p))
		return (i == 0);

	return function_check_src_mem(compiler, p, i);
}

#define FUNCTION_CHECK_DST(p, i) \
	CHECK_ARGUMENT(function_check_dst(compiler, p, i));

static sljit_s32 function_fcheck(struct sljit_compiler *compiler, sljit_s32 p, sljit_sw i)
{
	if (compiler->scratches == -1 || compiler->saveds == -1)
		return 0;

	if (FUNCTION_CHECK_IS_FREG(p))
		return (i == 0);

	return function_check_src_mem(compiler, p, i);
}

#define FUNCTION_FCHECK(p, i) \
	CHECK_ARGUMENT(function_fcheck(compiler, p, i));

#endif /* SLJIT_ARGUMENT_CHECKS */

#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)

SLJIT_API_FUNC_ATTRIBUTE void sljit_compiler_verbose(struct sljit_compiler *compiler, FILE* verbose)
{
	compiler->verbose = verbose;
}

#if (defined SLJIT_64BIT_ARCHITECTURE && SLJIT_64BIT_ARCHITECTURE)
#ifdef _WIN64
#ifdef __GNUC__
#	define SLJIT_PRINT_D	"ll"
#else
#	define SLJIT_PRINT_D	"I64"
#endif
#else
#	define SLJIT_PRINT_D	"l"
#endif
#else
#	define SLJIT_PRINT_D	""
#endif

static void sljit_verbose_reg(struct sljit_compiler *compiler, sljit_s32 r)
{
	if (r < (SLJIT_R0 + compiler->scratches))
		fprintf(compiler->verbose, "r%d", r - SLJIT_R0);
	else if (r != SLJIT_SP)
		fprintf(compiler->verbose, "s%d", SLJIT_NUMBER_OF_REGISTERS - r);
	else
		fprintf(compiler->verbose, "sp");
}

static void sljit_verbose_freg(struct sljit_compiler *compiler, sljit_s32 r)
{
	if (r < (SLJIT_FR0 + compiler->fscratches))
		fprintf(compiler->verbose, "fr%d", r - SLJIT_FR0);
	else
		fprintf(compiler->verbose, "fs%d", SLJIT_NUMBER_OF_FLOAT_REGISTERS - r);
}

static void sljit_verbose_param(struct sljit_compiler *compiler, sljit_s32 p, sljit_sw i)
{
	if ((p) & SLJIT_IMM)
		fprintf(compiler->verbose, "#%" SLJIT_PRINT_D "d", (i));
	else if ((p) & SLJIT_MEM) {
		if ((p) & REG_MASK) {
			fputc('[', compiler->verbose);
			sljit_verbose_reg(compiler, (p) & REG_MASK);
			if ((p) & OFFS_REG_MASK) {
				fprintf(compiler->verbose, " + ");
				sljit_verbose_reg(compiler, OFFS_REG(p));
				if (i)
					fprintf(compiler->verbose, " * %d", 1 << (i));
			}
			else if (i)
				fprintf(compiler->verbose, " + %" SLJIT_PRINT_D "d", (i));
			fputc(']', compiler->verbose);
		}
		else
			fprintf(compiler->verbose, "[#%" SLJIT_PRINT_D "d]", (i));
	} else
		sljit_verbose_reg(compiler, p);
}

static void sljit_verbose_fparam(struct sljit_compiler *compiler, sljit_s32 p, sljit_sw i)
{
	if ((p) & SLJIT_MEM) {
		if ((p) & REG_MASK) {
			fputc('[', compiler->verbose);
			sljit_verbose_reg(compiler, (p) & REG_MASK);
			if ((p) & OFFS_REG_MASK) {
				fprintf(compiler->verbose, " + ");
				sljit_verbose_reg(compiler, OFFS_REG(p));
				if (i)
					fprintf(compiler->verbose, "%d", 1 << (i));
			}
			else if (i)
				fprintf(compiler->verbose, " + %" SLJIT_PRINT_D "d", (i));
			fputc(']', compiler->verbose);
		}
		else
			fprintf(compiler->verbose, "[#%" SLJIT_PRINT_D "d]", (i));
	}
	else
		sljit_verbose_freg(compiler, p);
}

static const char* op0_names[] = {
	"breakpoint", "nop", "lmul.uw", "lmul.sw",
	"divmod.u", "divmod.s", "div.u", "div.s",
	"endbr", "skip_frames_before_return"
};

static const char* op1_names[] = {
	"", ".u8", ".s8", ".u16",
	".s16", ".u32", ".s32", "32",
	".p", "not", "clz", "ctz"
};

static const char* op2_names[] = {
	"add", "addc", "sub", "subc",
	"mul", "and", "or", "xor",
	"shl", "mshl", "lshr", "mlshr",
	"ashr", "mashr", "rotl", "rotr"
};

static const char* op_src_names[] = {
	"fast_return", "skip_frames_before_fast_return",
	"prefetch_l1", "prefetch_l2",
	"prefetch_l3", "prefetch_once",
};

static const char* fop1_names[] = {
	"mov", "conv", "conv", "conv",
	"conv", "conv", "cmp", "neg",
	"abs",
};

static const char* fop2_names[] = {
	"add", "sub", "mul", "div"
};

static const char* jump_names[] = {
	"equal", "not_equal",
	"less", "greater_equal",
	"greater", "less_equal",
	"sig_less", "sig_greater_equal",
	"sig_greater", "sig_less_equal",
	"overflow", "not_overflow",
	"carry", "",
	"f_equal", "f_not_equal",
	"f_less", "f_greater_equal",
	"f_greater", "f_less_equal",
	"unordered", "ordered",
	"ordered_equal", "unordered_or_not_equal",
	"ordered_less", "unordered_or_greater_equal",
	"ordered_greater", "unordered_or_less_equal",
	"unordered_or_equal", "ordered_not_equal",
	"unordered_or_less", "ordered_greater_equal",
	"unordered_or_greater", "ordered_less_equal",
	"jump", "fast_call",
	"call", "call_reg_arg"
};

static const char* call_arg_names[] = {
	"void", "w", "32", "p", "f64", "f32"
};

#endif /* SLJIT_VERBOSE */

/* --------------------------------------------------------------------- */
/*  Arch dependent                                                       */
/* --------------------------------------------------------------------- */

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS) \
	|| (defined SLJIT_VERBOSE && SLJIT_VERBOSE)

#define SLJIT_SKIP_CHECKS(compiler) (compiler)->skip_checks = 1

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_generate_code(struct sljit_compiler *compiler)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	struct sljit_jump *jump;
#endif

	SLJIT_UNUSED_ARG(compiler);

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(compiler->size > 0);
	jump = compiler->jumps;
	while (jump) {
		/* All jumps have target. */
		CHECK_ARGUMENT(jump->flags & (JUMP_LABEL | JUMP_ADDR));
		jump = jump->next;
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_enter(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	SLJIT_UNUSED_ARG(compiler);

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	if (options & SLJIT_ENTER_REG_ARG) {
		CHECK_ARGUMENT(!(options & ~(0x3 | SLJIT_ENTER_REG_ARG)));
	} else {
		CHECK_ARGUMENT(options == 0);
	}
	CHECK_ARGUMENT(SLJIT_KEPT_SAVEDS_COUNT(options) <= 3 && SLJIT_KEPT_SAVEDS_COUNT(options) <= saveds);
	CHECK_ARGUMENT(scratches >= 0 && scratches <= SLJIT_NUMBER_OF_REGISTERS);
	CHECK_ARGUMENT(saveds >= 0 && saveds <= SLJIT_NUMBER_OF_SAVED_REGISTERS);
	CHECK_ARGUMENT(scratches + saveds <= SLJIT_NUMBER_OF_REGISTERS);
	CHECK_ARGUMENT(fscratches >= 0 && fscratches <= SLJIT_NUMBER_OF_FLOAT_REGISTERS);
	CHECK_ARGUMENT(fsaveds >= 0 && fsaveds <= SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS);
	CHECK_ARGUMENT(fscratches + fsaveds <= SLJIT_NUMBER_OF_FLOAT_REGISTERS);
	CHECK_ARGUMENT(local_size >= 0 && local_size <= SLJIT_MAX_LOCAL_SIZE);
	CHECK_ARGUMENT((arg_types & SLJIT_ARG_FULL_MASK) <= SLJIT_ARG_TYPE_F32);
	CHECK_ARGUMENT(function_check_arguments(arg_types, scratches, (options & SLJIT_ENTER_REG_ARG) ? 0 : saveds, fscratches));

	compiler->last_flags = 0;
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  enter ret[%s", call_arg_names[arg_types & SLJIT_ARG_MASK]);

		arg_types >>= SLJIT_ARG_SHIFT;
		if (arg_types) {
			fprintf(compiler->verbose, "], args[");
			do {
				fprintf(compiler->verbose, "%s%s", call_arg_names[arg_types & SLJIT_ARG_MASK],
					(arg_types & SLJIT_ARG_TYPE_SCRATCH_REG) ? "_r" : "");
				arg_types >>= SLJIT_ARG_SHIFT;
				if (arg_types)
					fprintf(compiler->verbose, ",");
			} while (arg_types);
		}

		fprintf(compiler->verbose, "],");

		if (options & SLJIT_ENTER_REG_ARG) {
			fprintf(compiler->verbose, " enter:reg_arg,");

			if (SLJIT_KEPT_SAVEDS_COUNT(options) > 0)
				fprintf(compiler->verbose, " keep:%d,", SLJIT_KEPT_SAVEDS_COUNT(options));
		}

		fprintf(compiler->verbose, "scratches:%d, saveds:%d, fscratches:%d, fsaveds:%d, local_size:%d\n",
			scratches, saveds, fscratches, fsaveds, local_size);
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_set_context(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	SLJIT_UNUSED_ARG(compiler);

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	if (options & SLJIT_ENTER_REG_ARG) {
		CHECK_ARGUMENT(!(options & ~(0x3 | SLJIT_ENTER_REG_ARG)));
	} else {
		CHECK_ARGUMENT(options == 0);
	}
	CHECK_ARGUMENT(SLJIT_KEPT_SAVEDS_COUNT(options) <= 3 && SLJIT_KEPT_SAVEDS_COUNT(options) <= saveds);
	CHECK_ARGUMENT(scratches >= 0 && scratches <= SLJIT_NUMBER_OF_REGISTERS);
	CHECK_ARGUMENT(saveds >= 0 && saveds <= SLJIT_NUMBER_OF_SAVED_REGISTERS);
	CHECK_ARGUMENT(scratches + saveds <= SLJIT_NUMBER_OF_REGISTERS);
	CHECK_ARGUMENT(fscratches >= 0 && fscratches <= SLJIT_NUMBER_OF_FLOAT_REGISTERS);
	CHECK_ARGUMENT(fsaveds >= 0 && fsaveds <= SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS);
	CHECK_ARGUMENT(fscratches + fsaveds <= SLJIT_NUMBER_OF_FLOAT_REGISTERS);
	CHECK_ARGUMENT(local_size >= 0 && local_size <= SLJIT_MAX_LOCAL_SIZE);
	CHECK_ARGUMENT((arg_types & SLJIT_ARG_FULL_MASK) < SLJIT_ARG_TYPE_F64);
	CHECK_ARGUMENT(function_check_arguments(arg_types, scratches, (options & SLJIT_ENTER_REG_ARG) ? 0 : saveds, fscratches));

	compiler->last_flags = 0;
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  set_context ret[%s", call_arg_names[arg_types & SLJIT_ARG_MASK]);

		arg_types >>= SLJIT_ARG_SHIFT;
		if (arg_types) {
			fprintf(compiler->verbose, "], args[");
			do {
				fprintf(compiler->verbose, "%s%s", call_arg_names[arg_types & SLJIT_ARG_MASK],
					(arg_types & SLJIT_ARG_TYPE_SCRATCH_REG) ? "_r" : "");
				arg_types >>= SLJIT_ARG_SHIFT;
				if (arg_types)
					fprintf(compiler->verbose, ",");
			} while (arg_types);
		}

		fprintf(compiler->verbose, "],");

		if (options & SLJIT_ENTER_REG_ARG) {
			fprintf(compiler->verbose, " enter:reg_arg,");

			if (SLJIT_KEPT_SAVEDS_COUNT(options) > 0)
				fprintf(compiler->verbose, " keep:%d,", SLJIT_KEPT_SAVEDS_COUNT(options));
		}

		fprintf(compiler->verbose, " scratches:%d, saveds:%d, fscratches:%d, fsaveds:%d, local_size:%d\n",
			scratches, saveds, fscratches, fsaveds, local_size);
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_return_void(struct sljit_compiler *compiler)
{
	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(compiler->last_return == SLJIT_ARG_TYPE_VOID);
#endif

#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  return_void\n");
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_return(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 src, sljit_sw srcw)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(compiler->scratches >= 0);

	switch (compiler->last_return) {
	case SLJIT_ARG_TYPE_W:
		CHECK_ARGUMENT(op >= SLJIT_MOV && op <= SLJIT_MOV_S32);
		break;
	case SLJIT_ARG_TYPE_32:
		CHECK_ARGUMENT(op == SLJIT_MOV32 || (op >= SLJIT_MOV32_U8 && op <= SLJIT_MOV32_S16));
		break;
	case SLJIT_ARG_TYPE_P:
		CHECK_ARGUMENT(op == SLJIT_MOV_P);
		break;
	case SLJIT_ARG_TYPE_F64:
		CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_FPU));
		CHECK_ARGUMENT(op == SLJIT_MOV_F64);
		break;
	case SLJIT_ARG_TYPE_F32:
		CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_FPU));
		CHECK_ARGUMENT(op == SLJIT_MOV_F32);
		break;
	default:
		/* Context not initialized, void, etc. */
		CHECK_ARGUMENT(0);
		break;
	}

	if (GET_OPCODE(op) < SLJIT_MOV_F64) {
		FUNCTION_CHECK_SRC(src, srcw);
	} else {
		FUNCTION_FCHECK(src, srcw);
	}
	compiler->last_flags = 0;
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		if (GET_OPCODE(op) < SLJIT_MOV_F64) {
			fprintf(compiler->verbose, "  return%s%s ", !(op & SLJIT_32) ? "" : "32",
				op1_names[GET_OPCODE(op) - SLJIT_OP1_BASE]);
			sljit_verbose_param(compiler, src, srcw);
		} else {
			fprintf(compiler->verbose, "  return%s ", !(op & SLJIT_32) ? ".f64" : ".f32");
			sljit_verbose_fparam(compiler, src, srcw);
		}
		fprintf(compiler->verbose, "\n");
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_return_to(struct sljit_compiler *compiler,
	sljit_s32 src, sljit_sw srcw)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	FUNCTION_CHECK_SRC(src, srcw);
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  return_to ");
		sljit_verbose_param(compiler, src, srcw);
		fprintf(compiler->verbose, "\n");
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_fast_enter(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	FUNCTION_CHECK_DST(dst, dstw);
	compiler->last_flags = 0;
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  fast_enter ");
		sljit_verbose_param(compiler, dst, dstw);
		fprintf(compiler->verbose, "\n");
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_op0(struct sljit_compiler *compiler, sljit_s32 op)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT((op >= SLJIT_BREAKPOINT && op <= SLJIT_LMUL_SW)
		|| ((op & ~SLJIT_32) >= SLJIT_DIVMOD_UW && (op & ~SLJIT_32) <= SLJIT_DIV_SW)
		|| (op >= SLJIT_ENDBR && op <= SLJIT_SKIP_FRAMES_BEFORE_RETURN));
	CHECK_ARGUMENT(GET_OPCODE(op) < SLJIT_LMUL_UW || GET_OPCODE(op) >= SLJIT_ENDBR || compiler->scratches >= 2);
	if ((GET_OPCODE(op) >= SLJIT_LMUL_UW && GET_OPCODE(op) <= SLJIT_DIV_SW) || op == SLJIT_SKIP_FRAMES_BEFORE_RETURN)
		compiler->last_flags = 0;
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose))
	{
		fprintf(compiler->verbose, "  %s", op0_names[GET_OPCODE(op) - SLJIT_OP0_BASE]);
		if (GET_OPCODE(op) >= SLJIT_DIVMOD_UW && GET_OPCODE(op) <= SLJIT_DIV_SW) {
			fprintf(compiler->verbose, (op & SLJIT_32) ? "32" : "w");
		}
		fprintf(compiler->verbose, "\n");
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_op1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(GET_OPCODE(op) >= SLJIT_MOV && GET_OPCODE(op) <= SLJIT_CTZ);

	switch (GET_OPCODE(op)) {
	case SLJIT_NOT:
		/* Only SLJIT_32 and SLJIT_SET_Z are allowed. */
		CHECK_ARGUMENT(!(op & VARIABLE_FLAG_MASK));
		break;
	case SLJIT_MOV:
	case SLJIT_MOV_U32:
	case SLJIT_MOV_P:
		/* Nothing allowed */
		CHECK_ARGUMENT(!(op & (SLJIT_32 | SLJIT_SET_Z | VARIABLE_FLAG_MASK)));
		break;
	default:
		/* Only SLJIT_32 is allowed. */
		CHECK_ARGUMENT(!(op & (SLJIT_SET_Z | VARIABLE_FLAG_MASK)));
		break;
	}

	FUNCTION_CHECK_DST(dst, dstw);
	FUNCTION_CHECK_SRC(src, srcw);

	if (GET_OPCODE(op) >= SLJIT_NOT) {
		CHECK_ARGUMENT(src != SLJIT_IMM);
		compiler->last_flags = GET_FLAG_TYPE(op) | (op & (SLJIT_32 | SLJIT_SET_Z));
	}
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		if (GET_OPCODE(op) <= SLJIT_MOV_P)
		{
			fprintf(compiler->verbose, "  mov%s%s ", !(op & SLJIT_32) ? "" : "32",
				op1_names[GET_OPCODE(op) - SLJIT_OP1_BASE]);
		}
		else
		{
			fprintf(compiler->verbose, "  %s%s%s%s%s ", op1_names[GET_OPCODE(op) - SLJIT_OP1_BASE], !(op & SLJIT_32) ? "" : "32",
				!(op & SLJIT_SET_Z) ? "" : ".z", !(op & VARIABLE_FLAG_MASK) ? "" : ".",
				!(op & VARIABLE_FLAG_MASK) ? "" : jump_names[GET_FLAG_TYPE(op)]);
		}

		sljit_verbose_param(compiler, dst, dstw);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_param(compiler, src, srcw);
		fprintf(compiler->verbose, "\n");
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_op2(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 unset,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(GET_OPCODE(op) >= SLJIT_ADD && GET_OPCODE(op) <= SLJIT_ROTR);

	switch (GET_OPCODE(op)) {
	case SLJIT_AND:
	case SLJIT_OR:
	case SLJIT_XOR:
	case SLJIT_SHL:
	case SLJIT_MSHL:
	case SLJIT_LSHR:
	case SLJIT_MLSHR:
	case SLJIT_ASHR:
	case SLJIT_MASHR:
		CHECK_ARGUMENT(!(op & VARIABLE_FLAG_MASK));
		break;
	case SLJIT_MUL:
		CHECK_ARGUMENT(!(op & SLJIT_SET_Z));
		CHECK_ARGUMENT(!(op & VARIABLE_FLAG_MASK)
			|| GET_FLAG_TYPE(op) == SLJIT_OVERFLOW);
		break;
	case SLJIT_ADD:
		CHECK_ARGUMENT(!(op & VARIABLE_FLAG_MASK)
			|| GET_FLAG_TYPE(op) == GET_FLAG_TYPE(SLJIT_SET_CARRY)
			|| GET_FLAG_TYPE(op) == SLJIT_OVERFLOW);
		break;
	case SLJIT_SUB:
		CHECK_ARGUMENT(!(op & VARIABLE_FLAG_MASK)
			|| (GET_FLAG_TYPE(op) >= SLJIT_LESS && GET_FLAG_TYPE(op) <= SLJIT_OVERFLOW)
			|| GET_FLAG_TYPE(op) == GET_FLAG_TYPE(SLJIT_SET_CARRY));
		break;
	case SLJIT_ADDC:
	case SLJIT_SUBC:
		CHECK_ARGUMENT(!(op & VARIABLE_FLAG_MASK)
			|| GET_FLAG_TYPE(op) == GET_FLAG_TYPE(SLJIT_SET_CARRY));
		CHECK_ARGUMENT((compiler->last_flags & 0xff) == GET_FLAG_TYPE(SLJIT_SET_CARRY));
		CHECK_ARGUMENT((op & SLJIT_32) == (compiler->last_flags & SLJIT_32));
		break;
	case SLJIT_ROTL:
	case SLJIT_ROTR:
		CHECK_ARGUMENT(!(op & (SLJIT_SET_Z | VARIABLE_FLAG_MASK)));
		break;
	default:
		SLJIT_UNREACHABLE();
		break;
	}

	if (unset) {
		CHECK_ARGUMENT(HAS_FLAGS(op));
	} else {
		FUNCTION_CHECK_DST(dst, dstw);
	}
	FUNCTION_CHECK_SRC(src1, src1w);
	FUNCTION_CHECK_SRC(src2, src2w);
	compiler->last_flags = GET_FLAG_TYPE(op) | (op & (SLJIT_32 | SLJIT_SET_Z));
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  %s%s%s%s%s ", op2_names[GET_OPCODE(op) - SLJIT_OP2_BASE], !(op & SLJIT_32) ? "" : "32",
			!(op & SLJIT_SET_Z) ? "" : ".z", !(op & VARIABLE_FLAG_MASK) ? "" : ".",
			!(op & VARIABLE_FLAG_MASK) ? "" : jump_names[GET_FLAG_TYPE(op)]);
		if (unset)
			fprintf(compiler->verbose, "unset");
		else
			sljit_verbose_param(compiler, dst, dstw);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_param(compiler, src1, src1w);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_param(compiler, src2, src2w);
		fprintf(compiler->verbose, "\n");
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_shift_into(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src_dst,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(GET_OPCODE(op) == SLJIT_SHL || GET_OPCODE(op) == SLJIT_LSHR
		|| GET_OPCODE(op) == SLJIT_MSHL || GET_OPCODE(op) == SLJIT_MLSHR);
	CHECK_ARGUMENT((op & ~(0xff | SLJIT_32 | SLJIT_SHIFT_INTO_NON_ZERO)) == 0);
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_REG(src_dst));
	FUNCTION_CHECK_SRC(src1, src1w);
	FUNCTION_CHECK_SRC(src2, src2w);
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  %s%s.into%s ", op2_names[GET_OPCODE(op) - SLJIT_OP2_BASE], !(op & SLJIT_32) ? "" : "32",
			(op & SLJIT_SHIFT_INTO_NON_ZERO) ? ".nz" : "");

		sljit_verbose_reg(compiler, src_dst);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_param(compiler, src1, src1w);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_param(compiler, src2, src2w);
		fprintf(compiler->verbose, "\n");
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_op_src(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src, sljit_sw srcw)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(op >= SLJIT_FAST_RETURN && op <= SLJIT_PREFETCH_ONCE);
	FUNCTION_CHECK_SRC(src, srcw);

	if (op == SLJIT_FAST_RETURN || op == SLJIT_SKIP_FRAMES_BEFORE_FAST_RETURN)
	{
		CHECK_ARGUMENT(src != SLJIT_IMM);
		compiler->last_flags = 0;
	}
	else if (op >= SLJIT_PREFETCH_L1 && op <= SLJIT_PREFETCH_ONCE)
	{
		CHECK_ARGUMENT(src & SLJIT_MEM);
	}
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  %s ", op_src_names[op - SLJIT_OP_SRC_BASE]);
		sljit_verbose_param(compiler, src, srcw);
		fprintf(compiler->verbose, "\n");
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_get_register_index(sljit_s32 reg)
{
	SLJIT_UNUSED_ARG(reg);
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(reg > 0 && reg <= SLJIT_NUMBER_OF_REGISTERS);
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_get_float_register_index(sljit_s32 reg)
{
	SLJIT_UNUSED_ARG(reg);
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(reg > 0 && reg <= SLJIT_NUMBER_OF_FLOAT_REGISTERS);
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_op_custom(struct sljit_compiler *compiler,
	void *instruction, sljit_u32 size)
{
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	sljit_u32 i;
#endif

	SLJIT_UNUSED_ARG(compiler);

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(instruction);

#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86)
	CHECK_ARGUMENT(size > 0 && size < 16);
#elif (defined SLJIT_CONFIG_ARM_THUMB2 && SLJIT_CONFIG_ARM_THUMB2)
	CHECK_ARGUMENT((size == 2 && (((sljit_sw)instruction) & 0x1) == 0)
		|| (size == 4 && (((sljit_sw)instruction) & 0x3) == 0));
#elif (defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X)
	CHECK_ARGUMENT(size == 2 || size == 4 || size == 6);
#else
	CHECK_ARGUMENT(size == 4 && (((sljit_sw)instruction) & 0x3) == 0);
#endif

	compiler->last_flags = 0;
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  op_custom");
		for (i = 0; i < size; i++)
			fprintf(compiler->verbose, " 0x%x", ((sljit_u8*)instruction)[i]);
		fprintf(compiler->verbose, "\n");
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_fop1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_FPU));
	CHECK_ARGUMENT(GET_OPCODE(op) >= SLJIT_MOV_F64 && GET_OPCODE(op) <= SLJIT_ABS_F64);
	CHECK_ARGUMENT(!(op & (SLJIT_SET_Z | VARIABLE_FLAG_MASK)));
	FUNCTION_FCHECK(src, srcw);
	FUNCTION_FCHECK(dst, dstw);
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_F32)
			fprintf(compiler->verbose, "  %s%s ", fop1_names[SLJIT_CONV_F64_FROM_F32 - SLJIT_FOP1_BASE],
				(op & SLJIT_32) ? ".f32.from.f64" : ".f64.from.f32");
		else
			fprintf(compiler->verbose, "  %s%s ", fop1_names[GET_OPCODE(op) - SLJIT_FOP1_BASE],
				(op & SLJIT_32) ? ".f32" : ".f64");

		sljit_verbose_fparam(compiler, dst, dstw);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_fparam(compiler, src, srcw);
		fprintf(compiler->verbose, "\n");
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_fop1_cmp(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	compiler->last_flags = GET_FLAG_TYPE(op) | (op & SLJIT_32);
#endif

	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_FPU));
	CHECK_ARGUMENT(GET_OPCODE(op) == SLJIT_CMP_F64);
	CHECK_ARGUMENT(!(op & SLJIT_SET_Z));
	CHECK_ARGUMENT((op & VARIABLE_FLAG_MASK)
		|| (GET_FLAG_TYPE(op) >= SLJIT_F_EQUAL && GET_FLAG_TYPE(op) <= SLJIT_ORDERED_LESS_EQUAL));
	FUNCTION_FCHECK(src1, src1w);
	FUNCTION_FCHECK(src2, src2w);
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  %s%s", fop1_names[SLJIT_CMP_F64 - SLJIT_FOP1_BASE], (op & SLJIT_32) ? ".f32" : ".f64");
		if (op & VARIABLE_FLAG_MASK) {
			fprintf(compiler->verbose, ".%s", jump_names[GET_FLAG_TYPE(op)]);
		}
		fprintf(compiler->verbose, " ");
		sljit_verbose_fparam(compiler, src1, src1w);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_fparam(compiler, src2, src2w);
		fprintf(compiler->verbose, "\n");
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_fop1_conv_sw_from_f64(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_FPU));
	CHECK_ARGUMENT(GET_OPCODE(op) >= SLJIT_CONV_SW_FROM_F64 && GET_OPCODE(op) <= SLJIT_CONV_S32_FROM_F64);
	CHECK_ARGUMENT(!(op & (SLJIT_SET_Z | VARIABLE_FLAG_MASK)));
	FUNCTION_FCHECK(src, srcw);
	FUNCTION_CHECK_DST(dst, dstw);
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  %s%s.from%s ", fop1_names[GET_OPCODE(op) - SLJIT_FOP1_BASE],
			(GET_OPCODE(op) == SLJIT_CONV_S32_FROM_F64) ? ".s32" : ".sw",
			(op & SLJIT_32) ? ".f32" : ".f64");
		sljit_verbose_param(compiler, dst, dstw);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_fparam(compiler, src, srcw);
		fprintf(compiler->verbose, "\n");
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_fop1_conv_f64_from_sw(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_FPU));
	CHECK_ARGUMENT(GET_OPCODE(op) >= SLJIT_CONV_F64_FROM_SW && GET_OPCODE(op) <= SLJIT_CONV_F64_FROM_S32);
	CHECK_ARGUMENT(!(op & (SLJIT_SET_Z | VARIABLE_FLAG_MASK)));
	FUNCTION_CHECK_SRC(src, srcw);
	FUNCTION_FCHECK(dst, dstw);
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  %s%s.from%s ", fop1_names[GET_OPCODE(op) - SLJIT_FOP1_BASE],
			(op & SLJIT_32) ? ".f32" : ".f64",
			(GET_OPCODE(op) == SLJIT_CONV_F64_FROM_S32) ? ".s32" : ".sw");
		sljit_verbose_fparam(compiler, dst, dstw);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_param(compiler, src, srcw);
		fprintf(compiler->verbose, "\n");
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_fop2(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_FPU));
	CHECK_ARGUMENT(GET_OPCODE(op) >= SLJIT_ADD_F64 && GET_OPCODE(op) <= SLJIT_DIV_F64);
	CHECK_ARGUMENT(!(op & (SLJIT_SET_Z | VARIABLE_FLAG_MASK)));
	FUNCTION_FCHECK(src1, src1w);
	FUNCTION_FCHECK(src2, src2w);
	FUNCTION_FCHECK(dst, dstw);
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  %s%s ", fop2_names[GET_OPCODE(op) - SLJIT_FOP2_BASE], (op & SLJIT_32) ? ".f32" : ".f64");
		sljit_verbose_fparam(compiler, dst, dstw);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_fparam(compiler, src1, src1w);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_fparam(compiler, src2, src2w);
		fprintf(compiler->verbose, "\n");
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_label(struct sljit_compiler *compiler)
{
	SLJIT_UNUSED_ARG(compiler);

	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	compiler->last_flags = 0;
#endif

#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose))
		fprintf(compiler->verbose, "label:\n");
#endif
	CHECK_RETURN_OK;
}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86) \
	|| (defined SLJIT_CONFIG_ARM && SLJIT_CONFIG_ARM)
#define CHECK_UNORDERED(type, last_flags) \
	((((type) & 0xff) == SLJIT_UNORDERED || ((type) & 0xff) == SLJIT_ORDERED) && \
		((last_flags) & 0xff) >= SLJIT_UNORDERED && ((last_flags) & 0xff) <= SLJIT_ORDERED_LESS_EQUAL)
#else
#define CHECK_UNORDERED(type, last_flags) 0
#endif
#endif /* SLJIT_ARGUMENT_CHECKS */

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_jump(struct sljit_compiler *compiler, sljit_s32 type)
{
	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(!(type & ~(0xff | SLJIT_REWRITABLE_JUMP)));
	CHECK_ARGUMENT((type & 0xff) >= SLJIT_EQUAL && (type & 0xff) <= SLJIT_FAST_CALL);

	if ((type & 0xff) < SLJIT_JUMP) {
		if ((type & 0xff) <= SLJIT_NOT_ZERO)
			CHECK_ARGUMENT(compiler->last_flags & SLJIT_SET_Z);
		else if ((compiler->last_flags & 0xff) == SLJIT_CARRY) {
			CHECK_ARGUMENT((type & 0xff) == SLJIT_CARRY || (type & 0xff) == SLJIT_NOT_CARRY);
			compiler->last_flags = 0;
		} else
			CHECK_ARGUMENT((type & 0xff) == (compiler->last_flags & 0xff)
				|| ((type & 0xff) == SLJIT_NOT_OVERFLOW && (compiler->last_flags & 0xff) == SLJIT_OVERFLOW)
				|| CHECK_UNORDERED(type, compiler->last_flags));
	}
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose))
		fprintf(compiler->verbose, "  jump%s %s\n", !(type & SLJIT_REWRITABLE_JUMP) ? "" : ".r",
			jump_names[type & 0xff]);
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_call(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 arg_types)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(!(type & ~(0xff | SLJIT_REWRITABLE_JUMP | SLJIT_CALL_RETURN)));
	CHECK_ARGUMENT((type & 0xff) >= SLJIT_CALL && (type & 0xff) <= SLJIT_CALL_REG_ARG);
	CHECK_ARGUMENT(function_check_arguments(arg_types, compiler->scratches, -1, compiler->fscratches));

	if (type & SLJIT_CALL_RETURN) {
		CHECK_ARGUMENT((arg_types & SLJIT_ARG_MASK) == compiler->last_return);

		if (compiler->options & SLJIT_ENTER_REG_ARG) {
			CHECK_ARGUMENT((type & 0xff) == SLJIT_CALL_REG_ARG);
		} else {
			CHECK_ARGUMENT((type & 0xff) != SLJIT_CALL_REG_ARG);
		}
	}
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  %s%s%s ret[%s", jump_names[type & 0xff],
			!(type & SLJIT_REWRITABLE_JUMP) ? "" : ".r",
			!(type & SLJIT_CALL_RETURN) ? "" : ".ret",
			call_arg_names[arg_types & SLJIT_ARG_MASK]);

		arg_types >>= SLJIT_ARG_SHIFT;
		if (arg_types) {
			fprintf(compiler->verbose, "], args[");
			do {
				fprintf(compiler->verbose, "%s", call_arg_names[arg_types & SLJIT_ARG_MASK]);
				arg_types >>= SLJIT_ARG_SHIFT;
				if (arg_types)
					fprintf(compiler->verbose, ",");
			} while (arg_types);
		}
		fprintf(compiler->verbose, "]\n");
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_cmp(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(!(type & ~(0xff | SLJIT_REWRITABLE_JUMP | SLJIT_32)));
	CHECK_ARGUMENT((type & 0xff) >= SLJIT_EQUAL && (type & 0xff) <= SLJIT_SIG_LESS_EQUAL);
	FUNCTION_CHECK_SRC(src1, src1w);
	FUNCTION_CHECK_SRC(src2, src2w);
	compiler->last_flags = 0;
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  cmp%s%s %s, ", (type & SLJIT_32) ? "32" : "",
			!(type & SLJIT_REWRITABLE_JUMP) ? "" : ".r", jump_names[type & 0xff]);
		sljit_verbose_param(compiler, src1, src1w);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_param(compiler, src2, src2w);
		fprintf(compiler->verbose, "\n");
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_fcmp(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_FPU));
	CHECK_ARGUMENT(!(type & ~(0xff | SLJIT_REWRITABLE_JUMP | SLJIT_32)));
	CHECK_ARGUMENT((type & 0xff) >= SLJIT_F_EQUAL && (type & 0xff) <= SLJIT_ORDERED_LESS_EQUAL
				&& ((type & 0xff) <= SLJIT_ORDERED || sljit_cmp_info(type & 0xff)));
	FUNCTION_FCHECK(src1, src1w);
	FUNCTION_FCHECK(src2, src2w);
	compiler->last_flags = 0;
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  fcmp%s%s %s, ", (type & SLJIT_32) ? ".f32" : ".f64",
			!(type & SLJIT_REWRITABLE_JUMP) ? "" : ".r", jump_names[type & 0xff]);
		sljit_verbose_fparam(compiler, src1, src1w);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_fparam(compiler, src2, src2w);
		fprintf(compiler->verbose, "\n");
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_ijump(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 src, sljit_sw srcw)
{
	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(type >= SLJIT_JUMP && type <= SLJIT_FAST_CALL);
	FUNCTION_CHECK_SRC(src, srcw);
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  ijump.%s ", jump_names[type]);
		sljit_verbose_param(compiler, src, srcw);
		fprintf(compiler->verbose, "\n");
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_icall(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 arg_types,
	sljit_s32 src, sljit_sw srcw)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(!(type & ~(0xff | SLJIT_CALL_RETURN)));
	CHECK_ARGUMENT((type & 0xff) >= SLJIT_CALL && (type & 0xff) <= SLJIT_CALL_REG_ARG);
	CHECK_ARGUMENT(function_check_arguments(arg_types, compiler->scratches, -1, compiler->fscratches));
	FUNCTION_CHECK_SRC(src, srcw);

	if (type & SLJIT_CALL_RETURN) {
		CHECK_ARGUMENT((arg_types & SLJIT_ARG_MASK) == compiler->last_return);

		if (compiler->options & SLJIT_ENTER_REG_ARG) {
			CHECK_ARGUMENT((type & 0xff) == SLJIT_CALL_REG_ARG);
		} else {
			CHECK_ARGUMENT((type & 0xff) != SLJIT_CALL_REG_ARG);
		}
	}
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  i%s%s ret[%s", jump_names[type & 0xff],
			!(type & SLJIT_CALL_RETURN) ? "" : ".ret",
			call_arg_names[arg_types & SLJIT_ARG_MASK]);

		arg_types >>= SLJIT_ARG_SHIFT;
		if (arg_types) {
			fprintf(compiler->verbose, "], args[");
			do {
				fprintf(compiler->verbose, "%s", call_arg_names[arg_types & SLJIT_ARG_MASK]);
				arg_types >>= SLJIT_ARG_SHIFT;
				if (arg_types)
					fprintf(compiler->verbose, ",");
			} while (arg_types);
		}
		fprintf(compiler->verbose, "], ");
		sljit_verbose_param(compiler, src, srcw);
		fprintf(compiler->verbose, "\n");
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_op_flags(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 type)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(type >= SLJIT_EQUAL && type <= SLJIT_ORDERED_LESS_EQUAL);
	CHECK_ARGUMENT(op == SLJIT_MOV || op == SLJIT_MOV32
		|| (GET_OPCODE(op) >= SLJIT_AND && GET_OPCODE(op) <= SLJIT_XOR));
	CHECK_ARGUMENT(!(op & VARIABLE_FLAG_MASK));

	if (type <= SLJIT_NOT_ZERO)
		CHECK_ARGUMENT(compiler->last_flags & SLJIT_SET_Z);
	else
		CHECK_ARGUMENT(type == (compiler->last_flags & 0xff)
			|| (type == SLJIT_NOT_CARRY && (compiler->last_flags & 0xff) == SLJIT_CARRY)
			|| (type == SLJIT_NOT_OVERFLOW && (compiler->last_flags & 0xff) == SLJIT_OVERFLOW)
			|| CHECK_UNORDERED(type, compiler->last_flags));

	FUNCTION_CHECK_DST(dst, dstw);

	if (GET_OPCODE(op) >= SLJIT_ADD)
		compiler->last_flags = GET_FLAG_TYPE(op) | (op & (SLJIT_32 | SLJIT_SET_Z));
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  flags.%s%s%s ",
			GET_OPCODE(op) < SLJIT_OP2_BASE ? "mov" : op2_names[GET_OPCODE(op) - SLJIT_OP2_BASE],
			GET_OPCODE(op) < SLJIT_OP2_BASE ? op1_names[GET_OPCODE(op) - SLJIT_OP1_BASE] : ((op & SLJIT_32) ? "32" : ""),
			!(op & SLJIT_SET_Z) ? "" : ".z");
		sljit_verbose_param(compiler, dst, dstw);
		fprintf(compiler->verbose, ", %s\n", jump_names[type]);
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_cmov(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_reg,
	sljit_s32 src, sljit_sw srcw)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	sljit_s32 cond = type & ~SLJIT_32;

	CHECK_ARGUMENT(cond >= SLJIT_EQUAL && cond <= SLJIT_ORDERED_LESS_EQUAL);

	CHECK_ARGUMENT(compiler->scratches != -1 && compiler->saveds != -1);
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_REG(dst_reg));
	if (src != SLJIT_IMM) {
		CHECK_ARGUMENT(FUNCTION_CHECK_IS_REG(src));
		CHECK_ARGUMENT(srcw == 0);
	}

	if (cond <= SLJIT_NOT_ZERO)
		CHECK_ARGUMENT(compiler->last_flags & SLJIT_SET_Z);
	else
		CHECK_ARGUMENT(cond == (compiler->last_flags & 0xff)
			|| (cond == SLJIT_NOT_CARRY && (compiler->last_flags & 0xff) == SLJIT_CARRY)
			|| (cond == SLJIT_NOT_OVERFLOW && (compiler->last_flags & 0xff) == SLJIT_OVERFLOW)
			|| CHECK_UNORDERED(cond, compiler->last_flags));
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  cmov%s %s, ",
			!(type & SLJIT_32) ? "" : "32",
			jump_names[type & ~SLJIT_32]);
		sljit_verbose_reg(compiler, dst_reg);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_param(compiler, src, srcw);
		fprintf(compiler->verbose, "\n");
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_mem(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 reg,
	sljit_s32 mem, sljit_sw memw)
{
	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	sljit_s32 allowed_flags;

	if (type & SLJIT_MEM_UNALIGNED) {
		CHECK_ARGUMENT(!(type & (SLJIT_MEM_UNALIGNED_16 | SLJIT_MEM_UNALIGNED_32)));
	} else if (type & SLJIT_MEM_UNALIGNED_16) {
		CHECK_ARGUMENT(!(type & SLJIT_MEM_UNALIGNED_32));
	} else {
		CHECK_ARGUMENT((reg & REG_PAIR_MASK) || (type & SLJIT_MEM_UNALIGNED_32));
	}

	allowed_flags = SLJIT_MEM_UNALIGNED;

	switch (type & 0xff) {
	case SLJIT_MOV_U32:
	case SLJIT_MOV_S32:
	case SLJIT_MOV32:
		allowed_flags = SLJIT_MEM_UNALIGNED | SLJIT_MEM_UNALIGNED_16;
		break;
	case SLJIT_MOV:
	case SLJIT_MOV_P:
		allowed_flags = SLJIT_MEM_UNALIGNED | SLJIT_MEM_UNALIGNED_16 | SLJIT_MEM_UNALIGNED_32;
		break;
	}

	CHECK_ARGUMENT((type & ~(0xff | SLJIT_32 | SLJIT_MEM_STORE | allowed_flags)) == 0);

	if (reg & REG_PAIR_MASK) {
		CHECK_ARGUMENT((type & 0xff) == SLJIT_MOV);
		CHECK_ARGUMENT(FUNCTION_CHECK_IS_REG(REG_PAIR_FIRST(reg)));
		CHECK_ARGUMENT(FUNCTION_CHECK_IS_REG(REG_PAIR_SECOND(reg)));
		CHECK_ARGUMENT(REG_PAIR_FIRST(reg) != REG_PAIR_SECOND(reg));
	} else {
		CHECK_ARGUMENT((type & 0xff) >= SLJIT_MOV && (type & 0xff) <= SLJIT_MOV_P);
		CHECK_ARGUMENT(!(type & SLJIT_32) || ((type & 0xff) >= SLJIT_MOV_U8 && (type & 0xff) <= SLJIT_MOV_S16));
		CHECK_ARGUMENT(FUNCTION_CHECK_IS_REG(reg));
	}

	FUNCTION_CHECK_SRC_MEM(mem, memw);
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		if ((type & 0xff) == SLJIT_MOV32)
			fprintf(compiler->verbose, "  %s32",
				(type & SLJIT_MEM_STORE) ? "store" : "load");
		else
			fprintf(compiler->verbose, "  %s%s%s",
				(type & SLJIT_MEM_STORE) ? "store" : "load",
				!(type & SLJIT_32) ? "" : "32",
				op1_names[(type & 0xff) - SLJIT_OP1_BASE]);

		if (type & SLJIT_MEM_UNALIGNED)
			printf(".un");
		else if (type & SLJIT_MEM_UNALIGNED_16)
			printf(".un16");
		else if (type & SLJIT_MEM_UNALIGNED_32)
			printf(".un32");

		if (reg & REG_PAIR_MASK) {
			fprintf(compiler->verbose, " {");
			sljit_verbose_reg(compiler, REG_PAIR_FIRST(reg));
			fprintf(compiler->verbose, ", ");
			sljit_verbose_reg(compiler, REG_PAIR_SECOND(reg));
			fprintf(compiler->verbose, "}, ");
		} else {
			fprintf(compiler->verbose, " ");
			sljit_verbose_reg(compiler, reg);
			fprintf(compiler->verbose, ", ");
		}
		sljit_verbose_param(compiler, mem, memw);
		fprintf(compiler->verbose, "\n");
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_mem_update(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 reg,
	sljit_s32 mem, sljit_sw memw)
{
	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT((type & 0xff) >= SLJIT_MOV && (type & 0xff) <= SLJIT_MOV_P);
	CHECK_ARGUMENT((type & ~(0xff | SLJIT_32 | SLJIT_MEM_STORE | SLJIT_MEM_SUPP | SLJIT_MEM_POST)) == 0);
	CHECK_ARGUMENT((mem & REG_MASK) != 0 && (mem & REG_MASK) != reg);

	FUNCTION_CHECK_SRC_MEM(mem, memw);
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		if (type & SLJIT_MEM_SUPP)
			CHECK_RETURN_OK;
		if (sljit_emit_mem_update(compiler, type | SLJIT_MEM_SUPP, reg, mem, memw) == SLJIT_ERR_UNSUPPORTED) {
			fprintf(compiler->verbose, "    # mem: unsupported form, no instructions are emitted\n");
			CHECK_RETURN_OK;
		}

		if ((type & 0xff) == SLJIT_MOV32)
			fprintf(compiler->verbose, "  %s32.%s ",
				(type & SLJIT_MEM_STORE) ? "store" : "load",
				(type & SLJIT_MEM_POST) ? "post" : "pre");
		else
			fprintf(compiler->verbose, "  %s%s%s.%s ",
				(type & SLJIT_MEM_STORE) ? "store" : "load",
				!(type & SLJIT_32) ? "" : "32",
				op1_names[(type & 0xff) - SLJIT_OP1_BASE],
				(type & SLJIT_MEM_POST) ? "post" : "pre");

		sljit_verbose_reg(compiler, reg);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_param(compiler, mem, memw);
		fprintf(compiler->verbose, "\n");
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_fmem(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 mem, sljit_sw memw)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT((type & 0xff) == SLJIT_MOV_F64);

	if (type & SLJIT_MEM_UNALIGNED) {
		CHECK_ARGUMENT(!(type & (SLJIT_MEM_UNALIGNED_16 | SLJIT_MEM_UNALIGNED_32)));
	} else if (type & SLJIT_MEM_UNALIGNED_16) {
		CHECK_ARGUMENT(!(type & SLJIT_MEM_UNALIGNED_32));
	} else {
		CHECK_ARGUMENT(type & SLJIT_MEM_UNALIGNED_32);
		CHECK_ARGUMENT(!(type & SLJIT_32));
	}

	CHECK_ARGUMENT(!(type & ~(0xff | SLJIT_32 | SLJIT_MEM_STORE | SLJIT_MEM_UNALIGNED | SLJIT_MEM_UNALIGNED_16 | SLJIT_MEM_UNALIGNED_32)));
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_FREG(freg));
	FUNCTION_CHECK_SRC_MEM(mem, memw);
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  %s.%s",
			(type & SLJIT_MEM_STORE) ? "store" : "load",
			!(type & SLJIT_32) ? "f64" : "f32");

		if (type & SLJIT_MEM_UNALIGNED)
			printf(".un");
		else if (type & SLJIT_MEM_UNALIGNED_16)
			printf(".un16");
		else if (type & SLJIT_MEM_UNALIGNED_32)
			printf(".un32");

		fprintf(compiler->verbose, " ");
		sljit_verbose_freg(compiler, freg);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_param(compiler, mem, memw);
		fprintf(compiler->verbose, "\n");
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_fmem_update(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 mem, sljit_sw memw)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT((type & 0xff) == SLJIT_MOV_F64);
	CHECK_ARGUMENT((type & ~(0xff | SLJIT_32 | SLJIT_MEM_STORE | SLJIT_MEM_SUPP | SLJIT_MEM_POST)) == 0);
	FUNCTION_CHECK_SRC_MEM(mem, memw);
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_FREG(freg));
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		if (type & SLJIT_MEM_SUPP)
			CHECK_RETURN_OK;
		if (sljit_emit_fmem_update(compiler, type | SLJIT_MEM_SUPP, freg, mem, memw) == SLJIT_ERR_UNSUPPORTED) {
			fprintf(compiler->verbose, "    # fmem: unsupported form, no instructions are emitted\n");
			CHECK_RETURN_OK;
		}

		fprintf(compiler->verbose, "  %s.%s.%s ",
			(type & SLJIT_MEM_STORE) ? "store" : "load",
			!(type & SLJIT_32) ? "f64" : "f32",
			(type & SLJIT_MEM_POST) ? "post" : "pre");

		sljit_verbose_freg(compiler, freg);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_param(compiler, mem, memw);
		fprintf(compiler->verbose, "\n");
	}
#endif
	CHECK_RETURN_OK;

}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_get_local_base(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw, sljit_sw offset)
{
	/* Any offset is allowed. */
	SLJIT_UNUSED_ARG(offset);

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	FUNCTION_CHECK_DST(dst, dstw);
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  local_base ");
		sljit_verbose_param(compiler, dst, dstw);
		fprintf(compiler->verbose, ", #%" SLJIT_PRINT_D "d\n", offset);
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_const(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw, sljit_sw init_value)
{
	SLJIT_UNUSED_ARG(init_value);

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	FUNCTION_CHECK_DST(dst, dstw);
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  const ");
		sljit_verbose_param(compiler, dst, dstw);
		fprintf(compiler->verbose, ", #%" SLJIT_PRINT_D "d\n", init_value);
	}
#endif
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_put_label(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	FUNCTION_CHECK_DST(dst, dstw);
#endif
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  put_label ");
		sljit_verbose_param(compiler, dst, dstw);
		fprintf(compiler->verbose, "\n");
	}
#endif
	CHECK_RETURN_OK;
}

#else /* !SLJIT_ARGUMENT_CHECKS && !SLJIT_VERBOSE */

#define SLJIT_SKIP_CHECKS(compiler)

#endif /* SLJIT_ARGUMENT_CHECKS || SLJIT_VERBOSE */

#define SELECT_FOP1_OPERATION_WITH_CHECKS(compiler, op, dst, dstw, src, srcw) \
	SLJIT_COMPILE_ASSERT(!(SLJIT_CONV_SW_FROM_F64 & 0x1) && !(SLJIT_CONV_F64_FROM_SW & 0x1), \
		invalid_float_opcodes); \
	if (GET_OPCODE(op) >= SLJIT_CONV_SW_FROM_F64 && GET_OPCODE(op) <= SLJIT_CMP_F64) { \
		if (GET_OPCODE(op) == SLJIT_CMP_F64) { \
			CHECK(check_sljit_emit_fop1_cmp(compiler, op, dst, dstw, src, srcw)); \
			ADJUST_LOCAL_OFFSET(dst, dstw); \
			ADJUST_LOCAL_OFFSET(src, srcw); \
			return sljit_emit_fop1_cmp(compiler, op, dst, dstw, src, srcw); \
		} \
		if ((GET_OPCODE(op) | 0x1) == SLJIT_CONV_S32_FROM_F64) { \
			CHECK(check_sljit_emit_fop1_conv_sw_from_f64(compiler, op, dst, dstw, src, srcw)); \
			ADJUST_LOCAL_OFFSET(dst, dstw); \
			ADJUST_LOCAL_OFFSET(src, srcw); \
			return sljit_emit_fop1_conv_sw_from_f64(compiler, op, dst, dstw, src, srcw); \
		} \
		CHECK(check_sljit_emit_fop1_conv_f64_from_sw(compiler, op, dst, dstw, src, srcw)); \
		ADJUST_LOCAL_OFFSET(dst, dstw); \
		ADJUST_LOCAL_OFFSET(src, srcw); \
		return sljit_emit_fop1_conv_f64_from_sw(compiler, op, dst, dstw, src, srcw); \
	} \
	CHECK(check_sljit_emit_fop1(compiler, op, dst, dstw, src, srcw)); \
	ADJUST_LOCAL_OFFSET(dst, dstw); \
	ADJUST_LOCAL_OFFSET(src, srcw);

#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86) \
		|| (defined SLJIT_CONFIG_PPC && SLJIT_CONFIG_PPC) \
		|| ((defined SLJIT_CONFIG_MIPS && SLJIT_CONFIG_MIPS) && !(defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 1 && SLJIT_MIPS_REV < 6)) \
		|| (defined SLJIT_CONFIG_RISCV && SLJIT_CONFIG_RISCV) \
		|| (defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X)

static SLJIT_INLINE sljit_s32 sljit_emit_cmov_generic(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_reg,
	sljit_s32 src, sljit_sw srcw)
{
	struct sljit_label *label;
	struct sljit_jump *jump;
	sljit_s32 op = (type & SLJIT_32) ? SLJIT_MOV32 : SLJIT_MOV;

	SLJIT_SKIP_CHECKS(compiler);
	jump = sljit_emit_jump(compiler, (type & ~SLJIT_32) ^ 0x1);
	FAIL_IF(!jump);

	SLJIT_SKIP_CHECKS(compiler);
	FAIL_IF(sljit_emit_op1(compiler, op, dst_reg, 0, src, srcw));

	SLJIT_SKIP_CHECKS(compiler);
	label = sljit_emit_label(compiler);
	FAIL_IF(!label);

	sljit_set_label(jump, label);
	return SLJIT_SUCCESS;
}

#endif

#if (!(defined SLJIT_CONFIG_MIPS && SLJIT_CONFIG_MIPS) || (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6)) \
	&& !(defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)

static sljit_s32 sljit_emit_mem_unaligned(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 reg,
	sljit_s32 mem, sljit_sw memw)
{
	SLJIT_SKIP_CHECKS(compiler);

	if (type & SLJIT_MEM_STORE)
		return sljit_emit_op1(compiler, type & (0xff | SLJIT_32), mem, memw, reg, 0);
	return sljit_emit_op1(compiler, type & (0xff | SLJIT_32), reg, 0, mem, memw);
}

#endif /* (!SLJIT_CONFIG_MIPS || SLJIT_MIPS_REV >= 6) && !SLJIT_CONFIG_ARM_V5 */

#if (!(defined SLJIT_CONFIG_MIPS && SLJIT_CONFIG_MIPS) || (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6)) \
	&& !(defined SLJIT_CONFIG_ARM_32 && SLJIT_CONFIG_ARM_32)

static sljit_s32 sljit_emit_fmem_unaligned(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 mem, sljit_sw memw)
{
	SLJIT_SKIP_CHECKS(compiler);

	if (type & SLJIT_MEM_STORE)
		return sljit_emit_fop1(compiler, type & (0xff | SLJIT_32), mem, memw, freg, 0);
	return sljit_emit_fop1(compiler, type & (0xff | SLJIT_32), freg, 0, mem, memw);
}

#endif /* (!SLJIT_CONFIG_MIPS || SLJIT_MIPS_REV >= 6) && !SLJIT_CONFIG_ARM */

/* CPU description section */

#if (defined SLJIT_32BIT_ARCHITECTURE && SLJIT_32BIT_ARCHITECTURE)
#define SLJIT_CPUINFO_PART1 " 32bit ("
#elif (defined SLJIT_64BIT_ARCHITECTURE && SLJIT_64BIT_ARCHITECTURE)
#define SLJIT_CPUINFO_PART1 " 64bit ("
#else
#error "Internal error: CPU type info missing"
#endif

#if (defined SLJIT_LITTLE_ENDIAN && SLJIT_LITTLE_ENDIAN)
#define SLJIT_CPUINFO_PART2 "little endian + "
#elif (defined SLJIT_BIG_ENDIAN && SLJIT_BIG_ENDIAN)
#define SLJIT_CPUINFO_PART2 "big endian + "
#else
#error "Internal error: CPU type info missing"
#endif

#if (defined SLJIT_UNALIGNED && SLJIT_UNALIGNED)
#define SLJIT_CPUINFO_PART3 "unaligned)"
#else
#define SLJIT_CPUINFO_PART3 "aligned)"
#endif

#define SLJIT_CPUINFO SLJIT_CPUINFO_PART1 SLJIT_CPUINFO_PART2 SLJIT_CPUINFO_PART3

#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86)
#	include "sljitNativeX86_common.c"
#elif (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
#	include "sljitNativeARM_32.c"
#elif (defined SLJIT_CONFIG_ARM_V7 && SLJIT_CONFIG_ARM_V7)
#	include "sljitNativeARM_32.c"
#elif (defined SLJIT_CONFIG_ARM_THUMB2 && SLJIT_CONFIG_ARM_THUMB2)
#	include "sljitNativeARM_T2_32.c"
#elif (defined SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64)
#	include "sljitNativeARM_64.c"
#elif (defined SLJIT_CONFIG_PPC && SLJIT_CONFIG_PPC)
#	include "sljitNativePPC_common.c"
#elif (defined SLJIT_CONFIG_MIPS && SLJIT_CONFIG_MIPS)
#	include "sljitNativeMIPS_common.c"
#elif (defined SLJIT_CONFIG_RISCV && SLJIT_CONFIG_RISCV)
#	include "sljitNativeRISCV_common.c"
#elif (defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X)
#	include "sljitNativeS390X.c"
#endif

static SLJIT_INLINE sljit_s32 emit_mov_before_return(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 src, sljit_sw srcw)
{
#if (defined SLJIT_64BIT_ARCHITECTURE && SLJIT_64BIT_ARCHITECTURE)
	/* At the moment the pointer size is always equal to sljit_sw. May be changed in the future. */
	if (src == SLJIT_RETURN_REG && (op == SLJIT_MOV || op == SLJIT_MOV_P))
		return SLJIT_SUCCESS;
#else
	if (src == SLJIT_RETURN_REG && (op == SLJIT_MOV || op == SLJIT_MOV_U32 || op == SLJIT_MOV_S32 || op == SLJIT_MOV_P))
		return SLJIT_SUCCESS;
#endif

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_op1(compiler, op, SLJIT_RETURN_REG, 0, src, srcw);
}

#if !(defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32) \
	&& !((defined SLJIT_CONFIG_ARM_32 && SLJIT_CONFIG_ARM_32) && defined __SOFTFP__)

static SLJIT_INLINE sljit_s32 emit_fmov_before_return(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 src, sljit_sw srcw)
{
	if (src == SLJIT_FR0)
		return SLJIT_SUCCESS;

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_fop1(compiler, op, SLJIT_RETURN_FREG, 0, src, srcw);
}

#endif /* !SLJIT_CONFIG_X86_32 && !(SLJIT_CONFIG_ARM_32 && __SOFTFP__) */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 src, sljit_sw srcw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_return(compiler, op, src, srcw));

	if (GET_OPCODE(op) < SLJIT_MOV_F64) {
		FAIL_IF(emit_mov_before_return(compiler, op, src, srcw));
	} else {
		FAIL_IF(emit_fmov_before_return(compiler, op, src, srcw));
	}

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_return_void(compiler);
}

#if !(defined SLJIT_CONFIG_MIPS && SLJIT_CONFIG_MIPS) \
	&& !(defined SLJIT_CONFIG_RISCV && SLJIT_CONFIG_RISCV)

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_cmp(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	/* Default compare for most architectures. */
	sljit_s32 flags, tmp_src, condition;
	sljit_sw tmp_srcw;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_cmp(compiler, type, src1, src1w, src2, src2w));

	condition = type & 0xff;
#if (defined SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64)
	if ((condition == SLJIT_EQUAL || condition == SLJIT_NOT_EQUAL)) {
		if ((src1 & SLJIT_IMM) && !src1w) {
			src1 = src2;
			src1w = src2w;
			src2 = SLJIT_IMM;
			src2w = 0;
		}
		if ((src2 & SLJIT_IMM) && !src2w)
			return emit_cmp_to0(compiler, type, src1, src1w);
	}
#endif

	if (SLJIT_UNLIKELY((src1 & SLJIT_IMM) && !(src2 & SLJIT_IMM))) {
		/* Immediate is preferred as second argument by most architectures. */
		switch (condition) {
		case SLJIT_LESS:
			condition = SLJIT_GREATER;
			break;
		case SLJIT_GREATER_EQUAL:
			condition = SLJIT_LESS_EQUAL;
			break;
		case SLJIT_GREATER:
			condition = SLJIT_LESS;
			break;
		case SLJIT_LESS_EQUAL:
			condition = SLJIT_GREATER_EQUAL;
			break;
		case SLJIT_SIG_LESS:
			condition = SLJIT_SIG_GREATER;
			break;
		case SLJIT_SIG_GREATER_EQUAL:
			condition = SLJIT_SIG_LESS_EQUAL;
			break;
		case SLJIT_SIG_GREATER:
			condition = SLJIT_SIG_LESS;
			break;
		case SLJIT_SIG_LESS_EQUAL:
			condition = SLJIT_SIG_GREATER_EQUAL;
			break;
		}

		type = condition | (type & (SLJIT_32 | SLJIT_REWRITABLE_JUMP));
		tmp_src = src1;
		src1 = src2;
		src2 = tmp_src;
		tmp_srcw = src1w;
		src1w = src2w;
		src2w = tmp_srcw;
	}

	if (condition <= SLJIT_NOT_ZERO)
		flags = SLJIT_SET_Z;
	else
		flags = condition << VARIABLE_FLAG_SHIFT;

	SLJIT_SKIP_CHECKS(compiler);
	PTR_FAIL_IF(sljit_emit_op2u(compiler,
		SLJIT_SUB | flags | (type & SLJIT_32), src1, src1w, src2, src2w));

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_jump(compiler, condition | (type & (SLJIT_REWRITABLE_JUMP | SLJIT_32)));
}

#endif /* !SLJIT_CONFIG_MIPS */

#if (defined SLJIT_CONFIG_ARM && SLJIT_CONFIG_ARM)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_cmp_info(sljit_s32 type)
{
	if (type < SLJIT_UNORDERED || type > SLJIT_ORDERED_LESS_EQUAL)
		return 0;

	switch (type) {
	case SLJIT_UNORDERED_OR_EQUAL:
	case SLJIT_ORDERED_NOT_EQUAL:
		return 0;
	}

	return 1;
}

#endif /* SLJIT_CONFIG_ARM */

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_fcmp(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_fcmp(compiler, type, src1, src1w, src2, src2w));

	SLJIT_SKIP_CHECKS(compiler);
	sljit_emit_fop1(compiler, SLJIT_CMP_F64 | ((type & 0xff) << VARIABLE_FLAG_SHIFT) | (type & SLJIT_32), src1, src1w, src2, src2w);

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_jump(compiler, type);
}

#if !(defined SLJIT_CONFIG_ARM && SLJIT_CONFIG_ARM) \
	&& !(defined SLJIT_CONFIG_PPC && SLJIT_CONFIG_PPC)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_mem_update(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 reg,
	sljit_s32 mem, sljit_sw memw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_mem_update(compiler, type, reg, mem, memw));
	SLJIT_UNUSED_ARG(type);
	SLJIT_UNUSED_ARG(reg);
	SLJIT_UNUSED_ARG(mem);
	SLJIT_UNUSED_ARG(memw);

	return SLJIT_ERR_UNSUPPORTED;
}

#endif /* !SLJIT_CONFIG_ARM && !SLJIT_CONFIG_PPC */

#if !(defined SLJIT_CONFIG_ARM_32 && SLJIT_CONFIG_ARM_32) \
	&& !(defined SLJIT_CONFIG_MIPS && SLJIT_CONFIG_MIPS)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fmem(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 mem, sljit_sw memw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_fmem(compiler, type, freg, mem, memw));

	return sljit_emit_fmem_unaligned(compiler, type, freg, mem, memw);
}

#endif /* !SLJIT_CONFIG_ARM_32 && !SLJIT_CONFIG_MIPS */

#if !(defined SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64) \
	&& !(defined SLJIT_CONFIG_PPC && SLJIT_CONFIG_PPC)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fmem_update(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 mem, sljit_sw memw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_fmem_update(compiler, type, freg, mem, memw));
	SLJIT_UNUSED_ARG(type);
	SLJIT_UNUSED_ARG(freg);
	SLJIT_UNUSED_ARG(mem);
	SLJIT_UNUSED_ARG(memw);

	return SLJIT_ERR_UNSUPPORTED;
}

#endif /* !SLJIT_CONFIG_ARM_64 && !SLJIT_CONFIG_PPC */

#if !(defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86) \
	&& !(defined SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_local_base(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw, sljit_sw offset)
{
	CHECK_ERROR();
	CHECK(check_sljit_get_local_base(compiler, dst, dstw, offset));

	ADJUST_LOCAL_OFFSET(SLJIT_MEM1(SLJIT_SP), offset);

	SLJIT_SKIP_CHECKS(compiler);

	if (offset != 0)
		return sljit_emit_op2(compiler, SLJIT_ADD, dst, dstw, SLJIT_SP, 0, SLJIT_IMM, offset);
	return sljit_emit_op1(compiler, SLJIT_MOV, dst, dstw, SLJIT_SP, 0);
}

#endif

#else /* SLJIT_CONFIG_UNSUPPORTED */

/* Empty function bodies for those machines, which are not (yet) supported. */

SLJIT_API_FUNC_ATTRIBUTE const char* sljit_get_platform_name(void)
{
	return "unsupported";
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_compiler* sljit_create_compiler(void *allocator_data, void *exec_allocator_data)
{
	SLJIT_UNUSED_ARG(allocator_data);
	SLJIT_UNUSED_ARG(exec_allocator_data);
	SLJIT_UNREACHABLE();
	return NULL;
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_free_compiler(struct sljit_compiler *compiler)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNREACHABLE();
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_compiler_memory_error(struct sljit_compiler *compiler)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNREACHABLE();
}

SLJIT_API_FUNC_ATTRIBUTE void* sljit_alloc_memory(struct sljit_compiler *compiler, sljit_s32 size)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(size);
	SLJIT_UNREACHABLE();
	return NULL;
}

#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
SLJIT_API_FUNC_ATTRIBUTE void sljit_compiler_verbose(struct sljit_compiler *compiler, FILE* verbose)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(verbose);
	SLJIT_UNREACHABLE();
}
#endif

SLJIT_API_FUNC_ATTRIBUTE void* sljit_generate_code(struct sljit_compiler *compiler)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNREACHABLE();
	return NULL;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_has_cpu_feature(sljit_s32 feature_type)
{
	SLJIT_UNUSED_ARG(feature_type);
	SLJIT_UNREACHABLE();
	return 0;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_cmp_info(sljit_s32 type)
{
	SLJIT_UNUSED_ARG(type);
	SLJIT_UNREACHABLE();
	return 0;
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_free_code(void* code, void *exec_allocator_data)
{
	SLJIT_UNUSED_ARG(code);
	SLJIT_UNUSED_ARG(exec_allocator_data);
	SLJIT_UNREACHABLE();
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_enter(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(options);
	SLJIT_UNUSED_ARG(arg_types);
	SLJIT_UNUSED_ARG(scratches);
	SLJIT_UNUSED_ARG(saveds);
	SLJIT_UNUSED_ARG(fscratches);
	SLJIT_UNUSED_ARG(fsaveds);
	SLJIT_UNUSED_ARG(local_size);
	SLJIT_UNREACHABLE();
	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_set_context(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(options);
	SLJIT_UNUSED_ARG(arg_types);
	SLJIT_UNUSED_ARG(scratches);
	SLJIT_UNUSED_ARG(saveds);
	SLJIT_UNUSED_ARG(fscratches);
	SLJIT_UNUSED_ARG(fsaveds);
	SLJIT_UNUSED_ARG(local_size);
	SLJIT_UNREACHABLE();
	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return_void(struct sljit_compiler *compiler)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNREACHABLE();
	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 src, sljit_sw srcw)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(op);
	SLJIT_UNUSED_ARG(src);
	SLJIT_UNUSED_ARG(srcw);
	SLJIT_UNREACHABLE();
	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return_to(struct sljit_compiler *compiler, sljit_s32 src, sljit_sw srcw)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(src);
	SLJIT_UNUSED_ARG(srcw);
	SLJIT_UNREACHABLE();
	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fast_enter(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(dst);
	SLJIT_UNUSED_ARG(dstw);
	SLJIT_UNREACHABLE();
	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op0(struct sljit_compiler *compiler, sljit_s32 op)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(op);
	SLJIT_UNREACHABLE();
	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(op);
	SLJIT_UNUSED_ARG(dst);
	SLJIT_UNUSED_ARG(dstw);
	SLJIT_UNUSED_ARG(src);
	SLJIT_UNUSED_ARG(srcw);
	SLJIT_UNREACHABLE();
	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(op);
	SLJIT_UNUSED_ARG(dst);
	SLJIT_UNUSED_ARG(dstw);
	SLJIT_UNUSED_ARG(src1);
	SLJIT_UNUSED_ARG(src1w);
	SLJIT_UNUSED_ARG(src2);
	SLJIT_UNUSED_ARG(src2w);
	SLJIT_UNREACHABLE();
	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2u(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(op);
	SLJIT_UNUSED_ARG(src1);
	SLJIT_UNUSED_ARG(src1w);
	SLJIT_UNUSED_ARG(src2);
	SLJIT_UNUSED_ARG(src2w);
	SLJIT_UNREACHABLE();
	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_shift_into(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src_dst,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(op);
	SLJIT_UNUSED_ARG(src_dst);
	SLJIT_UNUSED_ARG(src1);
	SLJIT_UNUSED_ARG(src1w);
	SLJIT_UNUSED_ARG(src2);
	SLJIT_UNUSED_ARG(src2w);
	SLJIT_UNREACHABLE();
	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_src(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src, sljit_sw srcw)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(op);
	SLJIT_UNUSED_ARG(src);
	SLJIT_UNUSED_ARG(srcw);
	SLJIT_UNREACHABLE();
	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_register_index(sljit_s32 reg)
{
	SLJIT_UNREACHABLE();
	return reg;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_custom(struct sljit_compiler *compiler,
	void *instruction, sljit_u32 size)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(instruction);
	SLJIT_UNUSED_ARG(size);
	SLJIT_UNREACHABLE();
	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_current_flags(struct sljit_compiler *compiler, sljit_s32 current_flags)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(current_flags);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(op);
	SLJIT_UNUSED_ARG(dst);
	SLJIT_UNUSED_ARG(dstw);
	SLJIT_UNUSED_ARG(src);
	SLJIT_UNUSED_ARG(srcw);
	SLJIT_UNREACHABLE();
	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop2(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(op);
	SLJIT_UNUSED_ARG(dst);
	SLJIT_UNUSED_ARG(dstw);
	SLJIT_UNUSED_ARG(src1);
	SLJIT_UNUSED_ARG(src1w);
	SLJIT_UNUSED_ARG(src2);
	SLJIT_UNUSED_ARG(src2w);
	SLJIT_UNREACHABLE();
	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_label* sljit_emit_label(struct sljit_compiler *compiler)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNREACHABLE();
	return NULL;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_jump(struct sljit_compiler *compiler, sljit_s32 type)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(type);
	SLJIT_UNREACHABLE();
	return NULL;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_call(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 arg_types)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(type);
	SLJIT_UNUSED_ARG(arg_types);
	SLJIT_UNREACHABLE();
	return NULL;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_cmp(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(type);
	SLJIT_UNUSED_ARG(src1);
	SLJIT_UNUSED_ARG(src1w);
	SLJIT_UNUSED_ARG(src2);
	SLJIT_UNUSED_ARG(src2w);
	SLJIT_UNREACHABLE();
	return NULL;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_fcmp(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(type);
	SLJIT_UNUSED_ARG(src1);
	SLJIT_UNUSED_ARG(src1w);
	SLJIT_UNUSED_ARG(src2);
	SLJIT_UNUSED_ARG(src2w);
	SLJIT_UNREACHABLE();
	return NULL;
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_label(struct sljit_jump *jump, struct sljit_label* label)
{
	SLJIT_UNUSED_ARG(jump);
	SLJIT_UNUSED_ARG(label);
	SLJIT_UNREACHABLE();
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_target(struct sljit_jump *jump, sljit_uw target)
{
	SLJIT_UNUSED_ARG(jump);
	SLJIT_UNUSED_ARG(target);
	SLJIT_UNREACHABLE();
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_put_label(struct sljit_put_label *put_label, struct sljit_label *label)
{
	SLJIT_UNUSED_ARG(put_label);
	SLJIT_UNUSED_ARG(label);
	SLJIT_UNREACHABLE();
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_ijump(struct sljit_compiler *compiler, sljit_s32 type, sljit_s32 src, sljit_sw srcw)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(type);
	SLJIT_UNUSED_ARG(src);
	SLJIT_UNUSED_ARG(srcw);
	SLJIT_UNREACHABLE();
	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_icall(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 arg_types,
	sljit_s32 src, sljit_sw srcw)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(type);
	SLJIT_UNUSED_ARG(arg_types);
	SLJIT_UNUSED_ARG(src);
	SLJIT_UNUSED_ARG(srcw);
	SLJIT_UNREACHABLE();
	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_flags(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 type)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(op);
	SLJIT_UNUSED_ARG(dst);
	SLJIT_UNUSED_ARG(dstw);
	SLJIT_UNUSED_ARG(type);
	SLJIT_UNREACHABLE();
	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_cmov(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_reg,
	sljit_s32 src, sljit_sw srcw)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(type);
	SLJIT_UNUSED_ARG(dst_reg);
	SLJIT_UNUSED_ARG(src);
	SLJIT_UNUSED_ARG(srcw);
	SLJIT_UNREACHABLE();
	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_mem(struct sljit_compiler *compiler, sljit_s32 type, sljit_s32 reg, sljit_s32 mem, sljit_sw memw)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(type);
	SLJIT_UNUSED_ARG(reg);
	SLJIT_UNUSED_ARG(mem);
	SLJIT_UNUSED_ARG(memw);
	SLJIT_UNREACHABLE();
	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_mem_update(struct sljit_compiler *compiler, sljit_s32 type, sljit_s32 reg, sljit_s32 mem, sljit_sw memw)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(type);
	SLJIT_UNUSED_ARG(reg);
	SLJIT_UNUSED_ARG(mem);
	SLJIT_UNUSED_ARG(memw);
	SLJIT_UNREACHABLE();
	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fmem(struct sljit_compiler *compiler, sljit_s32 type, sljit_s32 freg, sljit_s32 mem, sljit_sw memw)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(type);
	SLJIT_UNUSED_ARG(freg);
	SLJIT_UNUSED_ARG(mem);
	SLJIT_UNUSED_ARG(memw);
	SLJIT_UNREACHABLE();
	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fmem_update(struct sljit_compiler *compiler, sljit_s32 type, sljit_s32 freg, sljit_s32 mem, sljit_sw memw)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(type);
	SLJIT_UNUSED_ARG(freg);
	SLJIT_UNUSED_ARG(mem);
	SLJIT_UNUSED_ARG(memw);
	SLJIT_UNREACHABLE();
	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_local_base(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw, sljit_sw offset)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(dst);
	SLJIT_UNUSED_ARG(dstw);
	SLJIT_UNUSED_ARG(offset);
	SLJIT_UNREACHABLE();
	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_const* sljit_emit_const(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw, sljit_sw initval)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(dst);
	SLJIT_UNUSED_ARG(dstw);
	SLJIT_UNUSED_ARG(initval);
	SLJIT_UNREACHABLE();
	return NULL;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_put_label* sljit_emit_put_label(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(dst);
	SLJIT_UNUSED_ARG(dstw);
	return NULL;
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_jump_addr(sljit_uw addr, sljit_uw new_target, sljit_sw executable_offset)
{
	SLJIT_UNUSED_ARG(addr);
	SLJIT_UNUSED_ARG(new_target);
	SLJIT_UNUSED_ARG(executable_offset);
	SLJIT_UNREACHABLE();
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_const(sljit_uw addr, sljit_sw new_constant, sljit_sw executable_offset)
{
	SLJIT_UNUSED_ARG(addr);
	SLJIT_UNUSED_ARG(new_constant);
	SLJIT_UNUSED_ARG(executable_offset);
	SLJIT_UNREACHABLE();
}

#endif /* !SLJIT_CONFIG_UNSUPPORTED */
