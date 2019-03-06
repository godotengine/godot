/*
 * libwebsockets - small server side websockets and web server implementation
 *
 * Copyright (C) 2010-2018 Andy Green <andy@warmcat.com>
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation:
 *  version 2.1 of the License.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 *  MA  02110-1301  USA
 *
 * included from libwebsockets.h
 */

/* Do not treat - as a terminal character, so "my-token" is one token */
#define LWS_TOKENIZE_F_MINUS_NONTERM	(1 << 0)
/* Separately report aggregate colon-delimited tokens */
#define LWS_TOKENIZE_F_AGG_COLON	(1 << 1)
/* Enforce sequencing for a simple token , token , token ... list */
#define LWS_TOKENIZE_F_COMMA_SEP_LIST	(1 << 2)
/* Allow more characters in the tokens and less delimiters... default is
 * only alphanumeric + underscore in tokens */
#define LWS_TOKENIZE_F_RFC7230_DELIMS	(1 << 3)
/* Do not treat . as a terminal character, so "warmcat.com" is one token */
#define LWS_TOKENIZE_F_DOT_NONTERM	(1 << 4)
/* If something starts looking like a float, like 1.2, force to be string token.
 * This lets you receive dotted-quads like 192.168.0.1 as string tokens, and
 * avoids illegal float format detection like 1.myserver.com */
#define LWS_TOKENIZE_F_NO_FLOATS	(1 << 5)

typedef enum {

	LWS_TOKZE_ERRS			=  5, /* the number of errors defined */

	LWS_TOKZE_ERR_BROKEN_UTF8	= -5,	/* malformed or partial utf8 */
	LWS_TOKZE_ERR_UNTERM_STRING	= -4,	/* ended while we were in "" */
	LWS_TOKZE_ERR_MALFORMED_FLOAT	= -3,	/* like 0..1 or 0.1.1 */
	LWS_TOKZE_ERR_NUM_ON_LHS	= -2,	/* like 123= or 0.1= */
	LWS_TOKZE_ERR_COMMA_LIST	= -1,	/* like ",tok", or, "tok,," */

	LWS_TOKZE_ENDED = 0,		/* no more content */

	/* Note: results have ordinal 1+, EOT is 0 and errors are < 0 */

	LWS_TOKZE_DELIMITER,		/* a delimiter appeared */
	LWS_TOKZE_TOKEN,		/* a token appeared */
	LWS_TOKZE_INTEGER,		/* an integer appeared */
	LWS_TOKZE_FLOAT,		/* a float appeared */
	LWS_TOKZE_TOKEN_NAME_EQUALS,	/* token [whitespace] = */
	LWS_TOKZE_TOKEN_NAME_COLON,	/* token [whitespace] : (only with
					   LWS_TOKENIZE_F_AGG_COLON flag) */
	LWS_TOKZE_QUOTED_STRING,	/* "*", where * may have any char */

} lws_tokenize_elem;

/*
 * helper enums to allow caller to enforce legal delimiter sequencing, eg
 * disallow "token,,token", "token,", and ",token"
 */

enum lws_tokenize_delimiter_tracking {
	LWSTZ_DT_NEED_FIRST_CONTENT,
	LWSTZ_DT_NEED_DELIM,
	LWSTZ_DT_NEED_NEXT_CONTENT,
};

struct lws_tokenize {
	const char *start; /**< set to the start of the string to tokenize */
	const char *token; /**< the start of an identified token or delimiter */
	int len;	/**< set to the length of the string to tokenize */
	int token_len;	/**< the length of the identied token or delimiter */

	int flags;	/**< optional LWS_TOKENIZE_F_ flags, or 0 */
	int delim;
};

/**
 * lws_tokenize() - breaks down a string into tokens and delimiters in-place
 *
 * \param ts: the lws_tokenize struct to init
 * \param start: the string to tokenize
 * \param flags: LWS_TOKENIZE_F_ option flags
 *
 * This initializes the tokenize struct to point to the given string, and
 * sets the length to 2GiB - 1 (so there must be a terminating NUL)... you can
 * override this requirement by setting ts.len yourself before using it.
 *
 * .delim is also initialized to LWSTZ_DT_NEED_FIRST_CONTENT.
 */

LWS_VISIBLE LWS_EXTERN void
lws_tokenize_init(struct lws_tokenize *ts, const char *start, int flags);

/**
 * lws_tokenize() - breaks down a string into tokens and delimiters in-place
 *
 * \param ts: the lws_tokenize struct with information and state on what to do
 *
 * The \p ts struct should have its start, len and flags members initialized to
 * reflect the string to be tokenized and any options.
 *
 * Then `lws_tokenize()` may be called repeatedly on the struct, returning one
 * of `lws_tokenize_elem` each time, and with the struct's `token` and
 * `token_len` members set to describe the content of the delimiter or token
 * payload each time.
 *
 * There are no allocations during the process.
 *
 * returns lws_tokenize_elem that was identified (LWS_TOKZE_ENDED means reached
 * the end of the string).
 */

LWS_VISIBLE LWS_EXTERN lws_tokenize_elem
lws_tokenize(struct lws_tokenize *ts);

/**
 * lws_tokenize_cstr() - copy token string to NUL-terminated buffer
 *
 * \param ts: pointer to lws_tokenize struct to operate on
 * \param str: destination buffer
 * \pparam max: bytes in destination buffer
 *
 * returns 0 if OK or nonzero if the string + NUL won't fit.
 */

LWS_VISIBLE LWS_EXTERN int
lws_tokenize_cstr(struct lws_tokenize *ts, char *str, int max);
