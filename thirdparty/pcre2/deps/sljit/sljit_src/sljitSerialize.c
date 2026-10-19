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

#define SLJIT_GET_LABEL_INDEX(label) \
	((label)->u.index < SLJIT_LABEL_ALIGNED ? (label)->u.index : ((struct sljit_extended_label*)(label))->index)

SLJIT_API_FUNC_ATTRIBUTE sljit_uw sljit_get_label_index(struct sljit_label *label)
{
	return SLJIT_GET_LABEL_INDEX(label);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_jump_has_label(struct sljit_jump *jump)
{
	return !(jump->flags & JUMP_ADDR) && (jump->u.label != NULL);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_jump_has_target(struct sljit_jump *jump)
{
	return (jump->flags & JUMP_ADDR) != 0;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_jump_is_mov_addr(struct sljit_jump *jump)
{
	return (jump->flags & JUMP_MOV_ADDR) != 0;
}

#define SLJIT_SERIALIZE_DEBUG ((sljit_u16)0x1)

struct sljit_serialized_compiler {
	sljit_u32 signature;
	sljit_u16 version;
	sljit_u16 cpu_type;

	sljit_uw buf_segment_count;
	sljit_uw label_count;
	sljit_uw aligned_label_count;
	sljit_uw jump_count;
	sljit_uw const_count;

	sljit_s32 options;
	sljit_s32 scratches;
	sljit_s32 saveds;
	sljit_s32 fscratches;
	sljit_s32 fsaveds;
	sljit_s32 local_size;
	sljit_uw size;

#if (defined SLJIT_HAS_STATUS_FLAGS_STATE && SLJIT_HAS_STATUS_FLAGS_STATE)
	sljit_s32 status_flags_state;
#endif /* SLJIT_HAS_STATUS_FLAGS_STATE */

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	sljit_s32 args_size;
#endif /* SLJIT_CONFIG_X86_32 */

#if ((defined SLJIT_CONFIG_ARM_32 && SLJIT_CONFIG_ARM_32) && (defined __SOFTFP__)) \
		|| (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	sljit_uw args_size;
#endif /* (SLJIT_CONFIG_ARM_32 && __SOFTFP__) || SLJIT_CONFIG_MIPS_32 */

#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
	sljit_uw cpool_diff;
	sljit_uw cpool_fill;
	sljit_uw patches;
#endif /* SLJIT_CONFIG_ARM_V6 */

#if (defined SLJIT_CONFIG_MIPS && SLJIT_CONFIG_MIPS)
	sljit_s32 delay_slot;
#endif /* SLJIT_CONFIG_MIPS */

};

struct sljit_serialized_debug_info {
	sljit_sw last_flags;
	sljit_s32 last_return;
	sljit_s32 logical_local_size;
};

struct sljit_serialized_label {
	sljit_uw size;
};

struct sljit_serialized_aligned_label {
	sljit_uw size;
	sljit_uw data;
};

struct sljit_serialized_jump {
	sljit_uw addr;
	sljit_uw flags;
	sljit_uw value;
};

struct sljit_serialized_const {
	sljit_uw addr;
};

#define SLJIT_SERIALIZE_ALIGN(v) (((v) + sizeof(sljit_uw) - 1) & ~(sljit_uw)(sizeof(sljit_uw) - 1))
#if (defined SLJIT_LITTLE_ENDIAN && SLJIT_LITTLE_ENDIAN)
#define SLJIT_SERIALIZE_SIGNATURE 0x534c4a54
#else /* !SLJIT_LITTLE_ENDIAN */
#define SLJIT_SERIALIZE_SIGNATURE 0x544a4c53
#endif /* SLJIT_LITTLE_ENDIAN */
#define SLJIT_SERIALIZE_VERSION 1

SLJIT_API_FUNC_ATTRIBUTE sljit_uw* sljit_serialize_compiler(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_uw *size)
{
	sljit_uw serialized_size = sizeof(struct sljit_serialized_compiler);
	struct sljit_memory_fragment *buf;
	struct sljit_label *label;
	struct sljit_jump *jump;
	struct sljit_const *const_;
	struct sljit_serialized_compiler *serialized_compiler;
	struct sljit_serialized_label *serialized_label;
	struct sljit_serialized_aligned_label *serialized_aligned_label;
	struct sljit_serialized_jump *serialized_jump;
	struct sljit_serialized_const *serialized_const;
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS) \
		|| (defined SLJIT_DEBUG && SLJIT_DEBUG)
	struct sljit_serialized_debug_info *serialized_debug_info;
#endif /* SLJIT_ARGUMENT_CHECKS || SLJIT_DEBUG */
	sljit_uw counter, used_size;
	sljit_u8 *result;
	sljit_u8 *ptr;
	SLJIT_UNUSED_ARG(options);

	if (size != NULL)
		*size = 0;

	PTR_FAIL_IF(compiler->error);

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS) \
		|| (defined SLJIT_DEBUG && SLJIT_DEBUG)
	if (!(options & SLJIT_SERIALIZE_IGNORE_DEBUG))
		serialized_size += sizeof(struct sljit_serialized_debug_info);
#endif /* SLJIT_ARGUMENT_CHECKS || SLJIT_DEBUG */

#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
	serialized_size += SLJIT_SERIALIZE_ALIGN(compiler->cpool_fill * (sizeof(sljit_uw) + 1));
#endif /* SLJIT_CONFIG_ARM_V6 */

	/* Compute the size of the data. */
	buf = compiler->buf;
	while (buf != NULL) {
		serialized_size += sizeof(sljit_uw) + SLJIT_SERIALIZE_ALIGN(buf->used_size);
		buf = buf->next;
	}

	label = compiler->labels;
	while (label != NULL) {
		used_size = sizeof(struct sljit_serialized_label);

		if (label->u.index >= SLJIT_LABEL_ALIGNED)
			used_size += sizeof(struct sljit_serialized_aligned_label);

		serialized_size += used_size;
		label = label->next;
	}

	jump = compiler->jumps;
	while (jump != NULL) {
		serialized_size += sizeof(struct sljit_serialized_jump);
		jump = jump->next;
	}

	const_ = compiler->consts;
	while (const_ != NULL) {
		serialized_size += sizeof(struct sljit_serialized_const);
		const_ = const_->next;
	}

	result = (sljit_u8*)SLJIT_MALLOC(serialized_size, compiler->allocator_data);
	PTR_FAIL_IF_NULL(result);

	if (size != NULL)
		*size = serialized_size;

	ptr = result;
	serialized_compiler = (struct sljit_serialized_compiler*)ptr;
	ptr += sizeof(struct sljit_serialized_compiler);

	serialized_compiler->signature = SLJIT_SERIALIZE_SIGNATURE;
	serialized_compiler->version = SLJIT_SERIALIZE_VERSION;
	serialized_compiler->cpu_type = 0;
	serialized_compiler->label_count = compiler->label_count;
	serialized_compiler->options = compiler->options;
	serialized_compiler->scratches = compiler->scratches;
	serialized_compiler->saveds = compiler->saveds;
	serialized_compiler->fscratches = compiler->fscratches;
	serialized_compiler->fsaveds = compiler->fsaveds;
	serialized_compiler->local_size = compiler->local_size;
	serialized_compiler->size = compiler->size;

#if (defined SLJIT_HAS_STATUS_FLAGS_STATE && SLJIT_HAS_STATUS_FLAGS_STATE)
	serialized_compiler->status_flags_state = compiler->status_flags_state;
#endif /* SLJIT_HAS_STATUS_FLAGS_STATE */

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32) \
		|| ((defined SLJIT_CONFIG_ARM_32 && SLJIT_CONFIG_ARM_32) && (defined __SOFTFP__)) \
		|| (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	serialized_compiler->args_size = compiler->args_size;
#endif /* SLJIT_CONFIG_X86_32 || (SLJIT_CONFIG_ARM_32 && __SOFTFP__) || SLJIT_CONFIG_MIPS_32 */

#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
	serialized_compiler->cpool_diff = compiler->cpool_diff;
	serialized_compiler->cpool_fill = compiler->cpool_fill;
	serialized_compiler->patches = compiler->patches;

	SLJIT_MEMCPY(ptr, compiler->cpool, compiler->cpool_fill * sizeof(sljit_uw));
	SLJIT_MEMCPY(ptr + compiler->cpool_fill * sizeof(sljit_uw), compiler->cpool_unique, compiler->cpool_fill);
	ptr += SLJIT_SERIALIZE_ALIGN(compiler->cpool_fill * (sizeof(sljit_uw) + 1));
#endif /* SLJIT_CONFIG_ARM_V6 */

#if (defined SLJIT_CONFIG_MIPS && SLJIT_CONFIG_MIPS)
	serialized_compiler->delay_slot = compiler->delay_slot;
#endif /* SLJIT_CONFIG_MIPS */

	buf = compiler->buf;
	counter = 0;
	while (buf != NULL) {
		used_size = buf->used_size;
		*(sljit_uw*)ptr = used_size;
		ptr += sizeof(sljit_uw);
		SLJIT_MEMCPY(ptr, buf->memory, used_size);
		ptr += SLJIT_SERIALIZE_ALIGN(used_size);
		buf = buf->next;
		counter++;
	}
	serialized_compiler->buf_segment_count = counter;

	label = compiler->labels;
	counter = 0;
	while (label != NULL) {
		serialized_label = (struct sljit_serialized_label*)ptr;
		serialized_label->size = (label->u.index < SLJIT_LABEL_ALIGNED) ? label->size : label->u.index;
		ptr += sizeof(struct sljit_serialized_label);

		if (label->u.index >= SLJIT_LABEL_ALIGNED) {
			serialized_aligned_label = (struct sljit_serialized_aligned_label*)ptr;
			serialized_aligned_label->size = label->size;
			serialized_aligned_label->data = ((struct sljit_extended_label*)label)->data;
			ptr += sizeof(struct sljit_serialized_aligned_label);
			counter++;
		}

		label = label->next;
	}
	serialized_compiler->aligned_label_count = counter;

	jump = compiler->jumps;
	counter = 0;
	while (jump != NULL) {
		serialized_jump = (struct sljit_serialized_jump*)ptr;
		serialized_jump->addr = jump->addr;
		serialized_jump->flags = jump->flags;

		if (jump->flags & JUMP_ADDR)
			serialized_jump->value = jump->u.target;
		else if (jump->u.label != NULL)
			serialized_jump->value = SLJIT_GET_LABEL_INDEX(jump->u.label);
		else
			serialized_jump->value = SLJIT_MAX_ADDRESS;

		ptr += sizeof(struct sljit_serialized_jump);
		jump = jump->next;
		counter++;
	}
	serialized_compiler->jump_count = counter;

	const_ = compiler->consts;
	counter = 0;
	while (const_ != NULL) {
		serialized_const = (struct sljit_serialized_const*)ptr;
		serialized_const->addr = const_->addr;
		ptr += sizeof(struct sljit_serialized_const);
		const_ = const_->next;
		counter++;
	}
	serialized_compiler->const_count = counter;

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS) \
		|| (defined SLJIT_DEBUG && SLJIT_DEBUG)
	if (!(options & SLJIT_SERIALIZE_IGNORE_DEBUG)) {
		serialized_debug_info = (struct sljit_serialized_debug_info*)ptr;
		serialized_debug_info->last_flags = compiler->last_flags;
		serialized_debug_info->last_return = compiler->last_return;
		serialized_debug_info->logical_local_size = compiler->logical_local_size;
		serialized_compiler->cpu_type |= SLJIT_SERIALIZE_DEBUG;
#if (defined SLJIT_DEBUG && SLJIT_DEBUG)
		ptr += sizeof(struct sljit_serialized_debug_info);
#endif /* SLJIT_DEBUG */
	}
#endif /* SLJIT_ARGUMENT_CHECKS || SLJIT_DEBUG */

	SLJIT_ASSERT((sljit_uw)(ptr - result) == serialized_size);
	return (sljit_uw*)result;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_compiler *sljit_deserialize_compiler(sljit_uw* buffer, sljit_uw size,
	sljit_s32 options, void *allocator_data)
{
	struct sljit_compiler *compiler;
	struct sljit_serialized_compiler *serialized_compiler;
	struct sljit_serialized_label *serialized_label;
	struct sljit_serialized_aligned_label *serialized_aligned_label;
	struct sljit_serialized_jump *serialized_jump;
	struct sljit_serialized_const *serialized_const;
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS) \
		|| (defined SLJIT_DEBUG && SLJIT_DEBUG)
	struct sljit_serialized_debug_info *serialized_debug_info;
#endif /* SLJIT_ARGUMENT_CHECKS || SLJIT_DEBUG */
	struct sljit_memory_fragment *buf;
	struct sljit_memory_fragment *last_buf;
	struct sljit_label *label;
	struct sljit_label *last_label;
	struct sljit_label **label_list = NULL;
	struct sljit_label **label_list_ptr = NULL;
	struct sljit_jump *jump;
	struct sljit_jump *last_jump;
	struct sljit_const *const_;
	struct sljit_const *last_const;
	sljit_u8 *ptr = (sljit_u8*)buffer;
	sljit_u8 *end = ptr + size;
	sljit_uw i, type, used_size, aligned_size;
	sljit_uw label_count, aligned_label_count;
	SLJIT_UNUSED_ARG(options);

	if (size < sizeof(struct sljit_serialized_compiler) || (size & (sizeof(sljit_uw) - 1)) != 0)
		return NULL;

	serialized_compiler = (struct sljit_serialized_compiler*)ptr;

	if (serialized_compiler->signature != SLJIT_SERIALIZE_SIGNATURE || serialized_compiler->version != SLJIT_SERIALIZE_VERSION)
		return NULL;

	compiler = sljit_create_compiler(allocator_data);
	PTR_FAIL_IF(compiler == NULL);

	compiler->label_count = serialized_compiler->label_count;
	compiler->options = serialized_compiler->options;
	compiler->scratches = serialized_compiler->scratches;
	compiler->saveds = serialized_compiler->saveds;
	compiler->fscratches = serialized_compiler->fscratches;
	compiler->fsaveds = serialized_compiler->fsaveds;
	compiler->local_size = serialized_compiler->local_size;
	compiler->size = serialized_compiler->size;

#if (defined SLJIT_HAS_STATUS_FLAGS_STATE && SLJIT_HAS_STATUS_FLAGS_STATE)
	compiler->status_flags_state = serialized_compiler->status_flags_state;
#endif /* SLJIT_HAS_STATUS_FLAGS_STATE */

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32) \
		|| ((defined SLJIT_CONFIG_ARM_32 && SLJIT_CONFIG_ARM_32) && (defined __SOFTFP__)) \
		|| (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	compiler->args_size = serialized_compiler->args_size;
#endif /* SLJIT_CONFIG_X86_32 || (SLJIT_CONFIG_ARM_32 && __SOFTFP__) || SLJIT_CONFIG_MIPS_32 */

#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
	used_size = serialized_compiler->cpool_fill;
	aligned_size = SLJIT_SERIALIZE_ALIGN(used_size * (sizeof(sljit_uw) + 1));
	compiler->cpool_diff = serialized_compiler->cpool_diff;
	compiler->cpool_fill = used_size;
	compiler->patches = serialized_compiler->patches;

	if ((sljit_uw)(end - ptr) < aligned_size)
		goto error;

	SLJIT_MEMCPY(compiler->cpool, ptr, used_size * sizeof(sljit_uw));
	SLJIT_MEMCPY(compiler->cpool_unique, ptr + used_size * sizeof(sljit_uw), used_size);
	ptr += aligned_size;
#endif /* SLJIT_CONFIG_ARM_V6 */

#if (defined SLJIT_CONFIG_MIPS && SLJIT_CONFIG_MIPS)
	compiler->delay_slot = serialized_compiler->delay_slot;
#endif /* SLJIT_CONFIG_MIPS */

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS) \
		|| (defined SLJIT_DEBUG && SLJIT_DEBUG)
	if (!(serialized_compiler->cpu_type & SLJIT_SERIALIZE_DEBUG))
		goto error;
#endif /* SLJIT_ARGUMENT_CHECKS || SLJIT_DEBUG */

	ptr += sizeof(struct sljit_serialized_compiler);
	i = serialized_compiler->buf_segment_count;
	last_buf = NULL;
	while (i > 0) {
		if ((sljit_uw)(end - ptr) < sizeof(sljit_uw))
			goto error;

		used_size = *(sljit_uw*)ptr;
		aligned_size = SLJIT_SERIALIZE_ALIGN(used_size);
		ptr += sizeof(sljit_uw);

		if ((sljit_uw)(end - ptr) < aligned_size)
			goto error;

		if (last_buf == NULL) {
			SLJIT_ASSERT(compiler->buf != NULL && compiler->buf->next == NULL);
			buf = compiler->buf;
		} else {
			buf = (struct sljit_memory_fragment*)SLJIT_MALLOC(BUF_SIZE, allocator_data);
			if (!buf)
				goto error;
			buf->next = NULL;
		}

		buf->used_size = used_size;
		SLJIT_MEMCPY(buf->memory, ptr, used_size);

		if (last_buf != NULL)
			last_buf->next = buf;
		last_buf = buf;

		ptr += aligned_size;
		i--;
	}

	last_label = NULL;
	label_count = serialized_compiler->label_count;
	aligned_label_count = serialized_compiler->aligned_label_count;
	i = (label_count * sizeof(struct sljit_serialized_label)) + (aligned_label_count * sizeof(struct sljit_serialized_aligned_label));

	if ((sljit_uw)(end - ptr) < i)
		goto error;

	label_list = (struct sljit_label **)SLJIT_MALLOC(label_count * sizeof(struct sljit_label*), allocator_data);
	if (label_list == NULL)
		goto error;

	label_list_ptr = label_list;
	for (i = 0; i < label_count; i++) {
		serialized_label = (struct sljit_serialized_label*)ptr;
		type = serialized_label->size;

		if (type < SLJIT_LABEL_ALIGNED) {
			label = (struct sljit_label*)ensure_abuf(compiler, sizeof(struct sljit_label));
		} else {
			label = (struct sljit_label*)ensure_abuf(compiler, sizeof(struct sljit_extended_label));
		}

		if (label == NULL)
			goto error;

		label->next = NULL;

		if (last_label != NULL)
			last_label->next = label;
		else
			compiler->labels = label;
		last_label = label;

		*label_list_ptr++ = label;

		ptr += sizeof(struct sljit_serialized_label);

		if (type < SLJIT_LABEL_ALIGNED) {
			label->u.index = i;
			label->size = type;
		} else {
			if (aligned_label_count == 0)
				goto error;

			aligned_label_count--;

			serialized_aligned_label = (struct sljit_serialized_aligned_label*)ptr;
			label->u.index = type;
			label->size = serialized_aligned_label->size;

			((struct sljit_extended_label*)label)->index = i;
			((struct sljit_extended_label*)label)->data = serialized_aligned_label->data;
			ptr += sizeof(struct sljit_serialized_aligned_label);
		}
	}
	compiler->last_label = last_label;

	if (aligned_label_count != 0)
		goto error;

	last_jump = NULL;
	i = serialized_compiler->jump_count;
	if ((sljit_uw)(end - ptr) < i * sizeof(struct sljit_serialized_jump))
		goto error;

	while (i > 0) {
		jump = (struct sljit_jump*)ensure_abuf(compiler, sizeof(struct sljit_jump));
		if (jump == NULL)
			goto error;

		serialized_jump = (struct sljit_serialized_jump*)ptr;
		jump->next = NULL;
		jump->addr = serialized_jump->addr;
		jump->flags = serialized_jump->flags;

		if (!(serialized_jump->flags & JUMP_ADDR)) {
			if (serialized_jump->value != SLJIT_MAX_ADDRESS) {
				if (serialized_jump->value >= label_count)
					goto error;
				jump->u.label = label_list[serialized_jump->value];
			} else
				jump->u.label = NULL;
		} else
			jump->u.target = serialized_jump->value;

		if (last_jump != NULL)
			last_jump->next = jump;
		else
			compiler->jumps = jump;
		last_jump = jump;

		ptr += sizeof(struct sljit_serialized_jump);
		i--;
	}
	compiler->last_jump = last_jump;

	SLJIT_FREE(label_list, allocator_data);
	label_list = NULL;

	last_const = NULL;
	i = serialized_compiler->const_count;
	if ((sljit_uw)(end - ptr) < i * sizeof(struct sljit_serialized_const))
		goto error;

	while (i > 0) {
		const_ = (struct sljit_const*)ensure_abuf(compiler, sizeof(struct sljit_const));
		if (const_ == NULL)
			goto error;

		serialized_const = (struct sljit_serialized_const*)ptr;
		const_->next = NULL;
		const_->addr = serialized_const->addr;

		if (last_const != NULL)
			last_const->next = const_;
		else
			compiler->consts = const_;
		last_const = const_;

		ptr += sizeof(struct sljit_serialized_const);
		i--;
	}
	compiler->last_const = last_const;

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS) \
		|| (defined SLJIT_DEBUG && SLJIT_DEBUG)
	if ((sljit_uw)(end - ptr) < sizeof(struct sljit_serialized_debug_info))
		goto error;

	serialized_debug_info = (struct sljit_serialized_debug_info*)ptr;
	compiler->last_flags = (sljit_s32)serialized_debug_info->last_flags;
	compiler->last_return = serialized_debug_info->last_return;
	compiler->logical_local_size = serialized_debug_info->logical_local_size;
#endif /* SLJIT_ARGUMENT_CHECKS || SLJIT_DEBUG */

	return compiler;

error:
	sljit_free_compiler(compiler);
	if (label_list != NULL)
		SLJIT_FREE(label_list, allocator_data);
	return NULL;
}
