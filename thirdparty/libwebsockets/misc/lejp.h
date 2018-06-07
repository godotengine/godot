#include "libwebsockets.h"
struct lejp_ctx;

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(_x) (sizeof(_x) / sizeof(_x[0]))
#endif
#define LEJP_FLAG_WS_KEEP 64
#define LEJP_FLAG_WS_COMMENTLINE 32

enum lejp_states {
	LEJP_IDLE = 0,
	LEJP_MEMBERS = 1,
	LEJP_M_P = 2,
	LEJP_MP_STRING = LEJP_FLAG_WS_KEEP | 3,
	LEJP_MP_STRING_ESC = LEJP_FLAG_WS_KEEP | 4,
	LEJP_MP_STRING_ESC_U1 = LEJP_FLAG_WS_KEEP | 5,
	LEJP_MP_STRING_ESC_U2 = LEJP_FLAG_WS_KEEP | 6,
	LEJP_MP_STRING_ESC_U3 = LEJP_FLAG_WS_KEEP | 7,
	LEJP_MP_STRING_ESC_U4 = LEJP_FLAG_WS_KEEP | 8,
	LEJP_MP_DELIM = 9,
	LEJP_MP_VALUE = 10,
	LEJP_MP_VALUE_NUM_INT = LEJP_FLAG_WS_KEEP | 11,
	LEJP_MP_VALUE_NUM_EXP = LEJP_FLAG_WS_KEEP | 12,
	LEJP_MP_VALUE_TOK = LEJP_FLAG_WS_KEEP | 13,
	LEJP_MP_COMMA_OR_END = 14,
	LEJP_MP_ARRAY_END = 15,
};

enum lejp_reasons {
	LEJP_CONTINUE = -1,
	LEJP_REJECT_IDLE_NO_BRACE = -2,
	LEJP_REJECT_MEMBERS_NO_CLOSE = -3,
	LEJP_REJECT_MP_NO_OPEN_QUOTE = -4,
	LEJP_REJECT_MP_STRING_UNDERRUN = -5,
	LEJP_REJECT_MP_ILLEGAL_CTRL = -6,
	LEJP_REJECT_MP_STRING_ESC_ILLEGAL_ESC = -7,
	LEJP_REJECT_ILLEGAL_HEX = -8,
	LEJP_REJECT_MP_DELIM_MISSING_COLON = -9,
	LEJP_REJECT_MP_DELIM_BAD_VALUE_START = -10,
	LEJP_REJECT_MP_VAL_NUM_INT_NO_FRAC = -11,
	LEJP_REJECT_MP_VAL_NUM_FORMAT = -12,
	LEJP_REJECT_MP_VAL_NUM_EXP_BAD_EXP = -13,
	LEJP_REJECT_MP_VAL_TOK_UNKNOWN = -14,
	LEJP_REJECT_MP_C_OR_E_UNDERF = -15,
	LEJP_REJECT_MP_C_OR_E_NOTARRAY = -16,
	LEJP_REJECT_MP_ARRAY_END_MISSING = -17,
	LEJP_REJECT_STACK_OVERFLOW = -18,
	LEJP_REJECT_MP_DELIM_ISTACK = -19,
	LEJP_REJECT_NUM_TOO_LONG = -20,
	LEJP_REJECT_MP_C_OR_E_NEITHER = -21,
	LEJP_REJECT_UNKNOWN = -22,
	LEJP_REJECT_CALLBACK = -23
};

#define LEJP_FLAG_CB_IS_VALUE 64

enum lejp_callbacks {
	LEJPCB_CONSTRUCTED	= 0,
	LEJPCB_DESTRUCTED	= 1,

	LEJPCB_START		= 2,
	LEJPCB_COMPLETE		= 3,
	LEJPCB_FAILED		= 4,

	LEJPCB_PAIR_NAME	= 5,

	LEJPCB_VAL_TRUE		= LEJP_FLAG_CB_IS_VALUE | 6,
	LEJPCB_VAL_FALSE	= LEJP_FLAG_CB_IS_VALUE | 7,
	LEJPCB_VAL_NULL		= LEJP_FLAG_CB_IS_VALUE | 8,
	LEJPCB_VAL_NUM_INT	= LEJP_FLAG_CB_IS_VALUE | 9,
	LEJPCB_VAL_NUM_FLOAT	= LEJP_FLAG_CB_IS_VALUE | 10,
	LEJPCB_VAL_STR_START	= 11, /* notice handle separately */
	LEJPCB_VAL_STR_CHUNK	= LEJP_FLAG_CB_IS_VALUE | 12,
	LEJPCB_VAL_STR_END	= LEJP_FLAG_CB_IS_VALUE | 13,

	LEJPCB_ARRAY_START	= 14,
	LEJPCB_ARRAY_END	= 15,

	LEJPCB_OBJECT_START	= 16,
	LEJPCB_OBJECT_END	= 17
};

/**
 * _lejp_callback() - User parser actions
 * \param ctx:	LEJP context
 * \param reason:	Callback reason
 *
 *	Your user callback is associated with the context at construction time,
 *	and receives calls as the parsing progresses.
 *
 *	All of the callbacks may be ignored and just return 0.
 *
 *	The reasons it might get called, found in @reason, are:
 *
 *  LEJPCB_CONSTRUCTED:  The context was just constructed... you might want to
 *		perform one-time allocation for the life of the context.
 *
 *  LEJPCB_DESTRUCTED:	The context is being destructed... if you made any
 *		allocations at construction-time, you can free them now
 *
 *  LEJPCB_START:	Parsing is beginning at the first byte of input
 *
 *  LEJPCB_COMPLETE:	Parsing has completed successfully.  You'll get a 0 or
 *			positive return code from lejp_parse indicating the
 *			amount of unused bytes left in the input buffer
 *
 *  LEJPCB_FAILED:	Parsing failed.  You'll get a negative error code
 *  			returned from lejp_parse
 *
 *  LEJPCB_PAIR_NAME:	When a "name":"value" pair has had the name parsed,
 *			this callback occurs.  You can find the new name at
 *			the end of ctx->path[]
 *
 *  LEJPCB_VAL_TRUE:	The "true" value appeared
 *
 *  LEJPCB_VAL_FALSE:	The "false" value appeared
 *
 *  LEJPCB_VAL_NULL:	The "null" value appeared
 *
 *  LEJPCB_VAL_NUM_INT:	A string representing an integer is in ctx->buf
 *
 *  LEJPCB_VAL_NUM_FLOAT: A string representing a float is in ctx->buf
 *
 *  LEJPCB_VAL_STR_START: We are starting to parse a string, no data yet
 *
 *  LEJPCB_VAL_STR_CHUNK: We parsed LEJP_STRING_CHUNK -1 bytes of string data in
 *			ctx->buf, which is as much as we can buffer, so we are
 *			spilling it.  If all your strings are less than
 *			LEJP_STRING_CHUNK - 1 bytes, you will never see this
 *			callback.
 *
 *  LEJPCB_VAL_STR_END:	String parsing has completed, the last chunk of the
 *			string is in ctx->buf.
 *
 *  LEJPCB_ARRAY_START:	An array started
 *
 *  LEJPCB_ARRAY_END:	An array ended
 *
 *  LEJPCB_OBJECT_START: An object started
 *
 *  LEJPCB_OBJECT_END:	An object ended
 */
LWS_EXTERN signed char _lejp_callback(struct lejp_ctx *ctx, char reason);

typedef signed char (*lejp_callback)(struct lejp_ctx *ctx, char reason);

#ifndef LEJP_MAX_DEPTH
#define LEJP_MAX_DEPTH 12
#endif
#ifndef LEJP_MAX_INDEX_DEPTH
#define LEJP_MAX_INDEX_DEPTH 5
#endif
#ifndef LEJP_MAX_PATH
#define LEJP_MAX_PATH 128
#endif
#ifndef LEJP_STRING_CHUNK
/* must be >= 30 to assemble floats */
#define LEJP_STRING_CHUNK 255
#endif

enum num_flags {
	LEJP_SEEN_MINUS = (1 << 0),
	LEJP_SEEN_POINT = (1 << 1),
	LEJP_SEEN_POST_POINT = (1 << 2),
	LEJP_SEEN_EXP = (1 << 3)
};

struct _lejp_stack {
	char s; /* lejp_state stack*/
	char p;	/* path length */
	char i; /* index array length */
	char b; /* user bitfield */
};

struct lejp_ctx {

	/* sorted by type for most compact alignment
	 *
	 * pointers
	 */

	signed char (*callback)(struct lejp_ctx *ctx, char reason);
	void *user;
	const char * const *paths;

	/* arrays */

	struct _lejp_stack st[LEJP_MAX_DEPTH];
	unsigned short i[LEJP_MAX_INDEX_DEPTH]; /* index array */
	unsigned short wild[LEJP_MAX_INDEX_DEPTH]; /* index array */
	char path[LEJP_MAX_PATH];
	char buf[LEJP_STRING_CHUNK];

	/* int */

	unsigned int line;

	/* short */

	unsigned short uni;

	/* char */

	unsigned char npos;
	unsigned char dcount;
	unsigned char f;
	unsigned char sp; /* stack head */
	unsigned char ipos; /* index stack depth */
	unsigned char ppos;
	unsigned char count_paths;
	unsigned char path_match;
	unsigned char path_match_len;
	unsigned char wildcount;
};

LWS_VISIBLE LWS_EXTERN void
lejp_construct(struct lejp_ctx *ctx,
	       signed char (*callback)(struct lejp_ctx *ctx, char reason),
	       void *user, const char * const *paths, unsigned char paths_count);

LWS_VISIBLE LWS_EXTERN void
lejp_destruct(struct lejp_ctx *ctx);

LWS_VISIBLE LWS_EXTERN int
lejp_parse(struct lejp_ctx *ctx, const unsigned char *json, int len);

LWS_VISIBLE LWS_EXTERN void
lejp_change_callback(struct lejp_ctx *ctx,
		     signed char (*callback)(struct lejp_ctx *ctx, char reason));

LWS_VISIBLE LWS_EXTERN int
lejp_get_wildcard(struct lejp_ctx *ctx, int wildcard, char *dest, int len);
