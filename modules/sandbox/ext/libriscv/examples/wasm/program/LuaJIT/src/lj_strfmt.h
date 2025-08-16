/*
** String formatting.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

#ifndef _LJ_STRFMT_H
#define _LJ_STRFMT_H

#include "lj_obj.h"

typedef uint32_t SFormat;  /* Format indicator. */

/* Format parser state. */
typedef struct FormatState {
  const uint8_t *p;	/* Current format string pointer. */
  const uint8_t *e;	/* End of format string. */
  const char *str;	/* Returned literal string. */
  MSize len;		/* Size of literal string. */
} FormatState;

/* Format types (max. 16). */
typedef enum FormatType {
  STRFMT_EOF, STRFMT_ERR, STRFMT_LIT,
  STRFMT_INT, STRFMT_UINT, STRFMT_NUM, STRFMT_STR, STRFMT_CHAR, STRFMT_PTR
} FormatType;

/* Format subtypes (bits are reused). */
#define STRFMT_T_HEX	0x0010	/* STRFMT_UINT */
#define STRFMT_T_OCT	0x0020	/* STRFMT_UINT */
#define STRFMT_T_FP_A	0x0000	/* STRFMT_NUM */
#define STRFMT_T_FP_E	0x0010	/* STRFMT_NUM */
#define STRFMT_T_FP_F	0x0020	/* STRFMT_NUM */
#define STRFMT_T_FP_G	0x0030	/* STRFMT_NUM */
#define STRFMT_T_QUOTED	0x0010	/* STRFMT_STR */

/* Format flags. */
#define STRFMT_F_LEFT	0x0100
#define STRFMT_F_PLUS	0x0200
#define STRFMT_F_ZERO	0x0400
#define STRFMT_F_SPACE	0x0800
#define STRFMT_F_ALT	0x1000
#define STRFMT_F_UPPER	0x2000

/* Format indicator fields. */
#define STRFMT_SH_WIDTH	16
#define STRFMT_SH_PREC	24

#define STRFMT_TYPE(sf)		((FormatType)((sf) & 15))
#define STRFMT_WIDTH(sf)	(((sf) >> STRFMT_SH_WIDTH) & 255u)
#define STRFMT_PREC(sf)		((((sf) >> STRFMT_SH_PREC) & 255u) - 1u)
#define STRFMT_FP(sf)		(((sf) >> 4) & 3)

/* Formats for conversion characters. */
#define STRFMT_A	(STRFMT_NUM|STRFMT_T_FP_A)
#define STRFMT_C	(STRFMT_CHAR)
#define STRFMT_D	(STRFMT_INT)
#define STRFMT_E	(STRFMT_NUM|STRFMT_T_FP_E)
#define STRFMT_F	(STRFMT_NUM|STRFMT_T_FP_F)
#define STRFMT_G	(STRFMT_NUM|STRFMT_T_FP_G)
#define STRFMT_I	STRFMT_D
#define STRFMT_O	(STRFMT_UINT|STRFMT_T_OCT)
#define STRFMT_P	(STRFMT_PTR)
#define STRFMT_Q	(STRFMT_STR|STRFMT_T_QUOTED)
#define STRFMT_S	(STRFMT_STR)
#define STRFMT_U	(STRFMT_UINT)
#define STRFMT_X	(STRFMT_UINT|STRFMT_T_HEX)
#define STRFMT_G14	(STRFMT_G | ((14+1) << STRFMT_SH_PREC))

/* Maximum buffer sizes for conversions. */
#define STRFMT_MAXBUF_XINT	(1+22)  /* '0' prefix + uint64_t in octal. */
#define STRFMT_MAXBUF_INT	(1+10)  /* Sign + int32_t in decimal. */
#define STRFMT_MAXBUF_NUM	32  /* Must correspond with STRFMT_G14. */
#define STRFMT_MAXBUF_PTR	(2+2*sizeof(ptrdiff_t))  /* "0x" + hex ptr. */

/* Format parser. */
LJ_FUNC SFormat LJ_FASTCALL lj_strfmt_parse(FormatState *fs);

static LJ_AINLINE void lj_strfmt_init(FormatState *fs, const char *p, MSize len)
{
  fs->p = (const uint8_t *)p;
  fs->e = (const uint8_t *)p + len;
  /* Must be NUL-terminated. May have NULs inside, too. */
  lj_assertX(*fs->e == 0, "format not NUL-terminated");
}

/* Raw conversions. */
LJ_FUNC char * LJ_FASTCALL lj_strfmt_wint(char *p, int32_t k);
LJ_FUNC char * LJ_FASTCALL lj_strfmt_wptr(char *p, const void *v);
LJ_FUNC char * LJ_FASTCALL lj_strfmt_wuleb128(char *p, uint32_t v);
LJ_FUNC const char *lj_strfmt_wstrnum(lua_State *L, cTValue *o, MSize *lenp);

/* Unformatted conversions to buffer. */
LJ_FUNC SBuf * LJ_FASTCALL lj_strfmt_putint(SBuf *sb, int32_t k);
#if LJ_HASJIT
LJ_FUNC SBuf * LJ_FASTCALL lj_strfmt_putnum(SBuf *sb, cTValue *o);
#endif
LJ_FUNC SBuf * LJ_FASTCALL lj_strfmt_putptr(SBuf *sb, const void *v);
#if LJ_HASJIT
LJ_FUNC SBuf * LJ_FASTCALL lj_strfmt_putquoted(SBuf *sb, GCstr *str);
#endif

/* Formatted conversions to buffer. */
LJ_FUNC SBuf *lj_strfmt_putfxint(SBuf *sb, SFormat sf, uint64_t k);
LJ_FUNC SBuf *lj_strfmt_putfnum_int(SBuf *sb, SFormat sf, lua_Number n);
LJ_FUNC SBuf *lj_strfmt_putfnum_uint(SBuf *sb, SFormat sf, lua_Number n);
LJ_FUNC SBuf *lj_strfmt_putfnum(SBuf *sb, SFormat, lua_Number n);
LJ_FUNC SBuf *lj_strfmt_putfchar(SBuf *sb, SFormat, int32_t c);
#if LJ_HASJIT
LJ_FUNC SBuf *lj_strfmt_putfstr(SBuf *sb, SFormat, GCstr *str);
#endif
LJ_FUNC int lj_strfmt_putarg(lua_State *L, SBuf *sb, int arg, int retry);

/* Conversions to strings. */
LJ_FUNC GCstr * LJ_FASTCALL lj_strfmt_int(lua_State *L, int32_t k);
LJ_FUNCA GCstr * LJ_FASTCALL lj_strfmt_num(lua_State *L, cTValue *o);
LJ_FUNCA GCstr * LJ_FASTCALL lj_strfmt_number(lua_State *L, cTValue *o);
#if LJ_HASJIT
LJ_FUNC GCstr * LJ_FASTCALL lj_strfmt_char(lua_State *L, int c);
#endif
LJ_FUNC GCstr * LJ_FASTCALL lj_strfmt_obj(lua_State *L, cTValue *o);

/* Internal string formatting. */
LJ_FUNC const char *lj_strfmt_pushvf(lua_State *L, const char *fmt,
				     va_list argp);
LJ_FUNC const char *lj_strfmt_pushf(lua_State *L, const char *fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
  __attribute__ ((format (printf, 2, 3)))
#endif
  ;

#endif
