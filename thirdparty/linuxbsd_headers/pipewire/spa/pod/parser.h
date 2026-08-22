/* Spa */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_POD_PARSER_H
#define SPA_POD_PARSER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <errno.h>
#include <stdarg.h>

#include <spa/pod/iter.h>
#include <spa/pod/vararg.h>

/**
 * \addtogroup spa_pod
 * \{
 */

struct spa_pod_parser_state {
	uint32_t offset;
	uint32_t flags;
	struct spa_pod_frame *frame;
};

struct spa_pod_parser {
	const void *data;
	uint32_t size;
	uint32_t _padding;
	struct spa_pod_parser_state state;
};

#define SPA_POD_PARSER_INIT(buffer,size)  ((struct spa_pod_parser){ (buffer), (size), 0, {} })

static inline void spa_pod_parser_init(struct spa_pod_parser *parser,
				       const void *data, uint32_t size)
{
	*parser = SPA_POD_PARSER_INIT(data, size);
}

static inline void spa_pod_parser_pod(struct spa_pod_parser *parser,
				      const struct spa_pod *pod)
{
	spa_pod_parser_init(parser, pod, SPA_POD_SIZE(pod));
}

static inline void
spa_pod_parser_get_state(struct spa_pod_parser *parser, struct spa_pod_parser_state *state)
{
	*state = parser->state;
}

static inline void
spa_pod_parser_reset(struct spa_pod_parser *parser, struct spa_pod_parser_state *state)
{
	parser->state = *state;
}

static inline struct spa_pod *
spa_pod_parser_deref(struct spa_pod_parser *parser, uint32_t offset, uint32_t size)
{
	/* Cast to uint64_t to avoid wraparound.  Add 8 for the pod itself. */
	const uint64_t long_offset = (uint64_t)offset + 8;
	if (long_offset <= size && (offset & 7) == 0) {
		/* Use void* because creating a misaligned pointer is undefined. */
		void *pod = SPA_PTROFF(parser->data, offset, void);
		/*
		 * Check that the pointer is aligned and that the size (rounded
		 * to the next multiple of 8) is in bounds.
		 */
		if (SPA_IS_ALIGNED(pod, __alignof__(struct spa_pod)) &&
		    long_offset + SPA_ROUND_UP_N((uint64_t)SPA_POD_BODY_SIZE(pod), 8) <= size)
			return (struct spa_pod *)pod;
	}
	return NULL;
}

static inline struct spa_pod *spa_pod_parser_frame(struct spa_pod_parser *parser, struct spa_pod_frame *frame)
{
	return SPA_PTROFF(parser->data, frame->offset, struct spa_pod);
}

static inline void spa_pod_parser_push(struct spa_pod_parser *parser,
		      struct spa_pod_frame *frame, const struct spa_pod *pod, uint32_t offset)
{
	frame->pod = *pod;
	frame->offset = offset;
	frame->parent = parser->state.frame;
	frame->flags = parser->state.flags;
	parser->state.frame = frame;
}

static inline struct spa_pod *spa_pod_parser_current(struct spa_pod_parser *parser)
{
	struct spa_pod_frame *f = parser->state.frame;
	uint32_t size = f ? f->offset + SPA_POD_SIZE(&f->pod) : parser->size;
	return spa_pod_parser_deref(parser, parser->state.offset, size);
}

static inline void spa_pod_parser_advance(struct spa_pod_parser *parser, const struct spa_pod *pod)
{
	parser->state.offset += SPA_ROUND_UP_N(SPA_POD_SIZE(pod), 8);
}

static inline struct spa_pod *spa_pod_parser_next(struct spa_pod_parser *parser)
{
	struct spa_pod *pod = spa_pod_parser_current(parser);
	if (pod)
		spa_pod_parser_advance(parser, pod);
	return pod;
}

static inline int spa_pod_parser_pop(struct spa_pod_parser *parser,
		      struct spa_pod_frame *frame)
{
	parser->state.frame = frame->parent;
	parser->state.offset = frame->offset + SPA_ROUND_UP_N(SPA_POD_SIZE(&frame->pod), 8);
	return 0;
}

static inline int spa_pod_parser_get_bool(struct spa_pod_parser *parser, bool *value)
{
	int res = -EPIPE;
	const struct spa_pod *pod = spa_pod_parser_current(parser);
	if (pod != NULL && (res = spa_pod_get_bool(pod, value)) >= 0)
		spa_pod_parser_advance(parser, pod);
	return res;
}

static inline int spa_pod_parser_get_id(struct spa_pod_parser *parser, uint32_t *value)
{
	int res = -EPIPE;
	const struct spa_pod *pod = spa_pod_parser_current(parser);
	if (pod != NULL && (res = spa_pod_get_id(pod, value)) >= 0)
		spa_pod_parser_advance(parser, pod);
	return res;
}

static inline int spa_pod_parser_get_int(struct spa_pod_parser *parser, int32_t *value)
{
	int res = -EPIPE;
	const struct spa_pod *pod = spa_pod_parser_current(parser);
	if (pod != NULL && (res = spa_pod_get_int(pod, value)) >= 0)
		spa_pod_parser_advance(parser, pod);
	return res;
}

static inline int spa_pod_parser_get_long(struct spa_pod_parser *parser, int64_t *value)
{
	int res = -EPIPE;
	const struct spa_pod *pod = spa_pod_parser_current(parser);
	if (pod != NULL && (res = spa_pod_get_long(pod, value)) >= 0)
		spa_pod_parser_advance(parser, pod);
	return res;
}

static inline int spa_pod_parser_get_float(struct spa_pod_parser *parser, float *value)
{
	int res = -EPIPE;
	const struct spa_pod *pod = spa_pod_parser_current(parser);
	if (pod != NULL && (res = spa_pod_get_float(pod, value)) >= 0)
		spa_pod_parser_advance(parser, pod);
	return res;
}

static inline int spa_pod_parser_get_double(struct spa_pod_parser *parser, double *value)
{
	int res = -EPIPE;
	const struct spa_pod *pod = spa_pod_parser_current(parser);
	if (pod != NULL && (res = spa_pod_get_double(pod, value)) >= 0)
		spa_pod_parser_advance(parser, pod);
	return res;
}

static inline int spa_pod_parser_get_string(struct spa_pod_parser *parser, const char **value)
{
	int res = -EPIPE;
	const struct spa_pod *pod = spa_pod_parser_current(parser);
	if (pod != NULL && (res = spa_pod_get_string(pod, value)) >= 0)
		spa_pod_parser_advance(parser, pod);
	return res;
}

static inline int spa_pod_parser_get_bytes(struct spa_pod_parser *parser, const void **value, uint32_t *len)
{
	int res = -EPIPE;
	const struct spa_pod *pod = spa_pod_parser_current(parser);
	if (pod != NULL && (res = spa_pod_get_bytes(pod, value, len)) >= 0)
		spa_pod_parser_advance(parser, pod);
	return res;
}

static inline int spa_pod_parser_get_pointer(struct spa_pod_parser *parser, uint32_t *type, const void **value)
{
	int res = -EPIPE;
	const struct spa_pod *pod = spa_pod_parser_current(parser);
	if (pod != NULL && (res = spa_pod_get_pointer(pod, type, value)) >= 0)
		spa_pod_parser_advance(parser, pod);
	return res;
}

static inline int spa_pod_parser_get_fd(struct spa_pod_parser *parser, int64_t *value)
{
	int res = -EPIPE;
	const struct spa_pod *pod = spa_pod_parser_current(parser);
	if (pod != NULL && (res = spa_pod_get_fd(pod, value)) >= 0)
		spa_pod_parser_advance(parser, pod);
	return res;
}

static inline int spa_pod_parser_get_rectangle(struct spa_pod_parser *parser, struct spa_rectangle *value)
{
	int res = -EPIPE;
	const struct spa_pod *pod = spa_pod_parser_current(parser);
	if (pod != NULL && (res = spa_pod_get_rectangle(pod, value)) >= 0)
		spa_pod_parser_advance(parser, pod);
	return res;
}

static inline int spa_pod_parser_get_fraction(struct spa_pod_parser *parser, struct spa_fraction *value)
{
	int res = -EPIPE;
	const struct spa_pod *pod = spa_pod_parser_current(parser);
	if (pod != NULL && (res = spa_pod_get_fraction(pod, value)) >= 0)
		spa_pod_parser_advance(parser, pod);
	return res;
}

static inline int spa_pod_parser_get_pod(struct spa_pod_parser *parser, struct spa_pod **value)
{
	struct spa_pod *pod = spa_pod_parser_current(parser);
	if (pod == NULL)
		return -EPIPE;
	*value = pod;
	spa_pod_parser_advance(parser, pod);
	return 0;
}
static inline int spa_pod_parser_push_struct(struct spa_pod_parser *parser,
		struct spa_pod_frame *frame)
{
	const struct spa_pod *pod = spa_pod_parser_current(parser);
	if (pod == NULL)
		return -EPIPE;
	if (!spa_pod_is_struct(pod))
		return -EINVAL;
	spa_pod_parser_push(parser, frame, pod, parser->state.offset);
	parser->state.offset += sizeof(struct spa_pod_struct);
	return 0;
}

static inline int spa_pod_parser_push_object(struct spa_pod_parser *parser,
		struct spa_pod_frame *frame, uint32_t type, uint32_t *id)
{
	const struct spa_pod *pod = spa_pod_parser_current(parser);
	if (pod == NULL)
		return -EPIPE;
	if (!spa_pod_is_object(pod))
		return -EINVAL;
	if (type != SPA_POD_OBJECT_TYPE(pod))
		return -EPROTO;
	if (id != NULL)
		*id = SPA_POD_OBJECT_ID(pod);
	spa_pod_parser_push(parser, frame, pod, parser->state.offset);
	parser->state.offset = parser->size;
	return 0;
}

static inline bool spa_pod_parser_can_collect(const struct spa_pod *pod, char type)
{
	if (pod == NULL)
		return false;

	if (SPA_POD_TYPE(pod) == SPA_TYPE_Choice) {
		if (!spa_pod_is_choice(pod))
			return false;
		if (type == 'V')
			return true;
		if (SPA_POD_CHOICE_TYPE(pod) != SPA_CHOICE_None)
			return false;
		pod = SPA_POD_CHOICE_CHILD(pod);
	}

	switch (type) {
	case 'P':
		return true;
	case 'b':
		return spa_pod_is_bool(pod);
	case 'I':
		return spa_pod_is_id(pod);
	case 'i':
		return spa_pod_is_int(pod);
	case 'l':
		return spa_pod_is_long(pod);
	case 'f':
		return spa_pod_is_float(pod);
	case 'd':
		return spa_pod_is_double(pod);
	case 's':
		return spa_pod_is_string(pod) || spa_pod_is_none(pod);
	case 'S':
		return spa_pod_is_string(pod);
	case 'y':
		return spa_pod_is_bytes(pod);
	case 'R':
		return spa_pod_is_rectangle(pod);
	case 'F':
		return spa_pod_is_fraction(pod);
	case 'B':
		return spa_pod_is_bitmap(pod);
	case 'a':
		return spa_pod_is_array(pod);
	case 'p':
		return spa_pod_is_pointer(pod);
	case 'h':
		return spa_pod_is_fd(pod);
	case 'T':
		return spa_pod_is_struct(pod) || spa_pod_is_none(pod);
	case 'O':
		return spa_pod_is_object(pod) || spa_pod_is_none(pod);
	case 'V':
	default:
		return false;
	}
}

#define SPA_POD_PARSER_COLLECT(pod,_type,args)						\
do {											\
	switch (_type) {								\
	case 'b':									\
		*va_arg(args, bool*) = SPA_POD_VALUE(struct spa_pod_bool, pod);		\
		break;									\
	case 'I':									\
	case 'i':									\
		*va_arg(args, int32_t*) = SPA_POD_VALUE(struct spa_pod_int, pod);	\
		break;									\
	case 'l':									\
		*va_arg(args, int64_t*) = SPA_POD_VALUE(struct spa_pod_long, pod);	\
		break;									\
	case 'f':									\
		*va_arg(args, float*) = SPA_POD_VALUE(struct spa_pod_float, pod);	\
		break;									\
	case 'd':									\
		*va_arg(args, double*) = SPA_POD_VALUE(struct spa_pod_double, pod);	\
		break;									\
	case 's':									\
		*va_arg(args, char**) =							\
			((pod) == NULL || (SPA_POD_TYPE(pod) == SPA_TYPE_None)		\
				? NULL							\
				: (char *)SPA_POD_CONTENTS(struct spa_pod_string, pod));	\
		break;									\
	case 'S':									\
	{										\
		char *dest = va_arg(args, char*);					\
		uint32_t maxlen = va_arg(args, uint32_t);				\
		strncpy(dest, (char *)SPA_POD_CONTENTS(struct spa_pod_string, pod), maxlen-1);	\
		dest[maxlen-1] = '\0';							\
		break;									\
	}										\
	case 'y':									\
		*(va_arg(args, void **)) = SPA_POD_CONTENTS(struct spa_pod_bytes, pod);	\
		*(va_arg(args, uint32_t *)) = SPA_POD_BODY_SIZE(pod);			\
		break;									\
	case 'R':									\
		*va_arg(args, struct spa_rectangle*) =					\
				SPA_POD_VALUE(struct spa_pod_rectangle, pod);		\
		break;									\
	case 'F':									\
		*va_arg(args, struct spa_fraction*) =					\
				SPA_POD_VALUE(struct spa_pod_fraction, pod);		\
		break;									\
	case 'B':									\
		*va_arg(args, uint32_t **) =						\
			(uint32_t *) SPA_POD_CONTENTS(struct spa_pod_bitmap, pod);	\
		break;									\
	case 'a':									\
		*va_arg(args, uint32_t*) = SPA_POD_ARRAY_VALUE_SIZE(pod);		\
		*va_arg(args, uint32_t*) = SPA_POD_ARRAY_VALUE_TYPE(pod);		\
		*va_arg(args, uint32_t*) = SPA_POD_ARRAY_N_VALUES(pod);			\
		*va_arg(args, void**) = SPA_POD_ARRAY_VALUES(pod);			\
		break;									\
	case 'p':									\
	{										\
		struct spa_pod_pointer_body *b =					\
				(struct spa_pod_pointer_body *) SPA_POD_BODY(pod);	\
		*(va_arg(args, uint32_t *)) = b->type;					\
		*(va_arg(args, const void **)) = b->value;				\
		break;									\
	}										\
	case 'h':									\
		*va_arg(args, int64_t*) = SPA_POD_VALUE(struct spa_pod_fd, pod);	\
		break;									\
	case 'P':									\
	case 'T':									\
	case 'O':									\
	case 'V':									\
	{										\
		const struct spa_pod **d = va_arg(args, const struct spa_pod**);	\
		if (d)									\
			*d = ((pod) == NULL || (SPA_POD_TYPE(pod) == SPA_TYPE_None)	\
				? NULL : (pod));						\
		break;									\
	}										\
	default:									\
		break;									\
	}										\
} while(false)

#define SPA_POD_PARSER_SKIP(_type,args)							\
do {											\
	switch (_type) {								\
	case 'S':									\
		va_arg(args, char*);							\
		va_arg(args, uint32_t);							\
		break;									\
	case 'a':									\
		va_arg(args, void*);							\
		va_arg(args, void*);							\
		SPA_FALLTHROUGH 							\
	case 'p':									\
	case 'y':									\
		va_arg(args, void*);							\
		SPA_FALLTHROUGH 							\
	case 'b':									\
	case 'I':									\
	case 'i':									\
	case 'l':									\
	case 'f':									\
	case 'd':									\
	case 's':									\
	case 'R':									\
	case 'F':									\
	case 'B':									\
	case 'h':									\
	case 'V':									\
	case 'P':									\
	case 'T':									\
	case 'O':									\
		va_arg(args, void*);							\
		break;									\
	}										\
} while(false)

static inline int spa_pod_parser_getv(struct spa_pod_parser *parser, va_list args)
{
	struct spa_pod_frame *f = parser->state.frame;
        uint32_t ftype = f ? f->pod.type : (uint32_t)SPA_TYPE_Struct;
	const struct spa_pod_prop *prop = NULL;
	int count = 0;

	do {
		bool optional;
		const struct spa_pod *pod = NULL;
		const char *format;

		if (f && ftype == SPA_TYPE_Object) {
			uint32_t key = va_arg(args, uint32_t);
			const struct spa_pod_object *object;

			if (key == 0)
				break;

			object = (const struct spa_pod_object *)spa_pod_parser_frame(parser, f);
			prop = spa_pod_object_find_prop(object, prop, key);
			pod = prop ? &prop->value : NULL;
		}

		if ((format = va_arg(args, char *)) == NULL)
			break;

		if (ftype == SPA_TYPE_Struct)
			pod = spa_pod_parser_next(parser);

		if ((optional = (*format == '?')))
			format++;

		if (!spa_pod_parser_can_collect(pod, *format)) {
			if (!optional) {
				if (pod == NULL)
					return -ESRCH;
				else
					return -EPROTO;
			}
			SPA_POD_PARSER_SKIP(*format, args);
		} else {
			if (pod->type == SPA_TYPE_Choice && *format != 'V')
				pod = SPA_POD_CHOICE_CHILD(pod);

			SPA_POD_PARSER_COLLECT(pod, *format, args);
			count++;
		}
	} while (true);

	return count;
}

static inline int spa_pod_parser_get(struct spa_pod_parser *parser, ...)
{
	int res;
	va_list args;

	va_start(args, parser);
	res = spa_pod_parser_getv(parser, args);
	va_end(args);

	return res;
}

#define SPA_POD_OPT_Bool(val)				"?" SPA_POD_Bool(val)
#define SPA_POD_OPT_Id(val)				"?" SPA_POD_Id(val)
#define SPA_POD_OPT_Int(val)				"?" SPA_POD_Int(val)
#define SPA_POD_OPT_Long(val)				"?" SPA_POD_Long(val)
#define SPA_POD_OPT_Float(val)				"?" SPA_POD_Float(val)
#define SPA_POD_OPT_Double(val)				"?" SPA_POD_Double(val)
#define SPA_POD_OPT_String(val)				"?" SPA_POD_String(val)
#define SPA_POD_OPT_Stringn(val,len)			"?" SPA_POD_Stringn(val,len)
#define SPA_POD_OPT_Bytes(val,len)			"?" SPA_POD_Bytes(val,len)
#define SPA_POD_OPT_Rectangle(val)			"?" SPA_POD_Rectangle(val)
#define SPA_POD_OPT_Fraction(val)			"?" SPA_POD_Fraction(val)
#define SPA_POD_OPT_Array(csize,ctype,n_vals,vals)	"?" SPA_POD_Array(csize,ctype,n_vals,vals)
#define SPA_POD_OPT_Pointer(type,val)			"?" SPA_POD_Pointer(type,val)
#define SPA_POD_OPT_Fd(val)				"?" SPA_POD_Fd(val)
#define SPA_POD_OPT_Pod(val)				"?" SPA_POD_Pod(val)
#define SPA_POD_OPT_PodObject(val)			"?" SPA_POD_PodObject(val)
#define SPA_POD_OPT_PodStruct(val)			"?" SPA_POD_PodStruct(val)
#define SPA_POD_OPT_PodChoice(val)			"?" SPA_POD_PodChoice(val)

#define spa_pod_parser_get_object(p,type,id,...)				\
({										\
	struct spa_pod_frame _f;						\
	int _res;								\
	if ((_res = spa_pod_parser_push_object(p, &_f, type, id)) == 0) {	\
		_res = spa_pod_parser_get(p,##__VA_ARGS__, 0);			\
		spa_pod_parser_pop(p, &_f);					\
	}									\
	_res;									\
})

#define spa_pod_parser_get_struct(p,...)				\
({									\
	struct spa_pod_frame _f;					\
	int _res;							\
	if ((_res = spa_pod_parser_push_struct(p, &_f)) == 0) {		\
		_res = spa_pod_parser_get(p,##__VA_ARGS__, NULL);	\
		spa_pod_parser_pop(p, &_f);				\
	}								\
	_res;							\
})

#define spa_pod_parse_object(pod,type,id,...)			\
({								\
	struct spa_pod_parser _p;				\
	spa_pod_parser_pod(&_p, pod);				\
	spa_pod_parser_get_object(&_p,type,id,##__VA_ARGS__);	\
})

#define spa_pod_parse_struct(pod,...)				\
({								\
	struct spa_pod_parser _p;				\
	spa_pod_parser_pod(&_p, pod);				\
	spa_pod_parser_get_struct(&_p,##__VA_ARGS__);		\
})

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_POD_PARSER_H */
