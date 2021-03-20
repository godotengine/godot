/* A Bison parser, made by GNU Bison 3.7.4.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2020 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output, and Bison version.  */
#define YYBISON 30704

/* Bison version string.  */
#define YYBISON_VERSION "3.7.4"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 1

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* First part of user prologue.  */
#line 69 "MachineIndependent/glslang.y"


/* Based on:
ANSI C Yacc grammar

In 1985, Jeff Lee published his Yacc grammar (which is accompanied by a
matching Lex specification) for the April 30, 1985 draft version of the
ANSI C standard.  Tom Stockfisch reposted it to net.sources in 1987; that
original, as mentioned in the answer to question 17.25 of the comp.lang.c
FAQ, can be ftp'ed from ftp.uu.net, file usenet/net.sources/ansi.c.grammar.Z.

I intend to keep this version as close to the current C Standard grammar as
possible; please let me know if you discover discrepancies.

Jutta Degener, 1995
*/

#include "SymbolTable.h"
#include "ParseHelper.h"
#include "../Public/ShaderLang.h"
#include "attribute.h"

using namespace glslang;


#line 97 "MachineIndependent/glslang_tab.cpp"

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

#include "glslang_tab.cpp.h"
/* Symbol kind.  */
enum yysymbol_kind_t
{
  YYSYMBOL_YYEMPTY = -2,
  YYSYMBOL_YYEOF = 0,                      /* "end of file"  */
  YYSYMBOL_YYerror = 1,                    /* error  */
  YYSYMBOL_YYUNDEF = 2,                    /* "invalid token"  */
  YYSYMBOL_CONST = 3,                      /* CONST  */
  YYSYMBOL_BOOL = 4,                       /* BOOL  */
  YYSYMBOL_INT = 5,                        /* INT  */
  YYSYMBOL_UINT = 6,                       /* UINT  */
  YYSYMBOL_FLOAT = 7,                      /* FLOAT  */
  YYSYMBOL_BVEC2 = 8,                      /* BVEC2  */
  YYSYMBOL_BVEC3 = 9,                      /* BVEC3  */
  YYSYMBOL_BVEC4 = 10,                     /* BVEC4  */
  YYSYMBOL_IVEC2 = 11,                     /* IVEC2  */
  YYSYMBOL_IVEC3 = 12,                     /* IVEC3  */
  YYSYMBOL_IVEC4 = 13,                     /* IVEC4  */
  YYSYMBOL_UVEC2 = 14,                     /* UVEC2  */
  YYSYMBOL_UVEC3 = 15,                     /* UVEC3  */
  YYSYMBOL_UVEC4 = 16,                     /* UVEC4  */
  YYSYMBOL_VEC2 = 17,                      /* VEC2  */
  YYSYMBOL_VEC3 = 18,                      /* VEC3  */
  YYSYMBOL_VEC4 = 19,                      /* VEC4  */
  YYSYMBOL_MAT2 = 20,                      /* MAT2  */
  YYSYMBOL_MAT3 = 21,                      /* MAT3  */
  YYSYMBOL_MAT4 = 22,                      /* MAT4  */
  YYSYMBOL_MAT2X2 = 23,                    /* MAT2X2  */
  YYSYMBOL_MAT2X3 = 24,                    /* MAT2X3  */
  YYSYMBOL_MAT2X4 = 25,                    /* MAT2X4  */
  YYSYMBOL_MAT3X2 = 26,                    /* MAT3X2  */
  YYSYMBOL_MAT3X3 = 27,                    /* MAT3X3  */
  YYSYMBOL_MAT3X4 = 28,                    /* MAT3X4  */
  YYSYMBOL_MAT4X2 = 29,                    /* MAT4X2  */
  YYSYMBOL_MAT4X3 = 30,                    /* MAT4X3  */
  YYSYMBOL_MAT4X4 = 31,                    /* MAT4X4  */
  YYSYMBOL_SAMPLER2D = 32,                 /* SAMPLER2D  */
  YYSYMBOL_SAMPLER3D = 33,                 /* SAMPLER3D  */
  YYSYMBOL_SAMPLERCUBE = 34,               /* SAMPLERCUBE  */
  YYSYMBOL_SAMPLER2DSHADOW = 35,           /* SAMPLER2DSHADOW  */
  YYSYMBOL_SAMPLERCUBESHADOW = 36,         /* SAMPLERCUBESHADOW  */
  YYSYMBOL_SAMPLER2DARRAY = 37,            /* SAMPLER2DARRAY  */
  YYSYMBOL_SAMPLER2DARRAYSHADOW = 38,      /* SAMPLER2DARRAYSHADOW  */
  YYSYMBOL_ISAMPLER2D = 39,                /* ISAMPLER2D  */
  YYSYMBOL_ISAMPLER3D = 40,                /* ISAMPLER3D  */
  YYSYMBOL_ISAMPLERCUBE = 41,              /* ISAMPLERCUBE  */
  YYSYMBOL_ISAMPLER2DARRAY = 42,           /* ISAMPLER2DARRAY  */
  YYSYMBOL_USAMPLER2D = 43,                /* USAMPLER2D  */
  YYSYMBOL_USAMPLER3D = 44,                /* USAMPLER3D  */
  YYSYMBOL_USAMPLERCUBE = 45,              /* USAMPLERCUBE  */
  YYSYMBOL_USAMPLER2DARRAY = 46,           /* USAMPLER2DARRAY  */
  YYSYMBOL_SAMPLER = 47,                   /* SAMPLER  */
  YYSYMBOL_SAMPLERSHADOW = 48,             /* SAMPLERSHADOW  */
  YYSYMBOL_TEXTURE2D = 49,                 /* TEXTURE2D  */
  YYSYMBOL_TEXTURE3D = 50,                 /* TEXTURE3D  */
  YYSYMBOL_TEXTURECUBE = 51,               /* TEXTURECUBE  */
  YYSYMBOL_TEXTURE2DARRAY = 52,            /* TEXTURE2DARRAY  */
  YYSYMBOL_ITEXTURE2D = 53,                /* ITEXTURE2D  */
  YYSYMBOL_ITEXTURE3D = 54,                /* ITEXTURE3D  */
  YYSYMBOL_ITEXTURECUBE = 55,              /* ITEXTURECUBE  */
  YYSYMBOL_ITEXTURE2DARRAY = 56,           /* ITEXTURE2DARRAY  */
  YYSYMBOL_UTEXTURE2D = 57,                /* UTEXTURE2D  */
  YYSYMBOL_UTEXTURE3D = 58,                /* UTEXTURE3D  */
  YYSYMBOL_UTEXTURECUBE = 59,              /* UTEXTURECUBE  */
  YYSYMBOL_UTEXTURE2DARRAY = 60,           /* UTEXTURE2DARRAY  */
  YYSYMBOL_ATTRIBUTE = 61,                 /* ATTRIBUTE  */
  YYSYMBOL_VARYING = 62,                   /* VARYING  */
  YYSYMBOL_FLOAT16_T = 63,                 /* FLOAT16_T  */
  YYSYMBOL_FLOAT32_T = 64,                 /* FLOAT32_T  */
  YYSYMBOL_DOUBLE = 65,                    /* DOUBLE  */
  YYSYMBOL_FLOAT64_T = 66,                 /* FLOAT64_T  */
  YYSYMBOL_INT64_T = 67,                   /* INT64_T  */
  YYSYMBOL_UINT64_T = 68,                  /* UINT64_T  */
  YYSYMBOL_INT32_T = 69,                   /* INT32_T  */
  YYSYMBOL_UINT32_T = 70,                  /* UINT32_T  */
  YYSYMBOL_INT16_T = 71,                   /* INT16_T  */
  YYSYMBOL_UINT16_T = 72,                  /* UINT16_T  */
  YYSYMBOL_INT8_T = 73,                    /* INT8_T  */
  YYSYMBOL_UINT8_T = 74,                   /* UINT8_T  */
  YYSYMBOL_I64VEC2 = 75,                   /* I64VEC2  */
  YYSYMBOL_I64VEC3 = 76,                   /* I64VEC3  */
  YYSYMBOL_I64VEC4 = 77,                   /* I64VEC4  */
  YYSYMBOL_U64VEC2 = 78,                   /* U64VEC2  */
  YYSYMBOL_U64VEC3 = 79,                   /* U64VEC3  */
  YYSYMBOL_U64VEC4 = 80,                   /* U64VEC4  */
  YYSYMBOL_I32VEC2 = 81,                   /* I32VEC2  */
  YYSYMBOL_I32VEC3 = 82,                   /* I32VEC3  */
  YYSYMBOL_I32VEC4 = 83,                   /* I32VEC4  */
  YYSYMBOL_U32VEC2 = 84,                   /* U32VEC2  */
  YYSYMBOL_U32VEC3 = 85,                   /* U32VEC3  */
  YYSYMBOL_U32VEC4 = 86,                   /* U32VEC4  */
  YYSYMBOL_I16VEC2 = 87,                   /* I16VEC2  */
  YYSYMBOL_I16VEC3 = 88,                   /* I16VEC3  */
  YYSYMBOL_I16VEC4 = 89,                   /* I16VEC4  */
  YYSYMBOL_U16VEC2 = 90,                   /* U16VEC2  */
  YYSYMBOL_U16VEC3 = 91,                   /* U16VEC3  */
  YYSYMBOL_U16VEC4 = 92,                   /* U16VEC4  */
  YYSYMBOL_I8VEC2 = 93,                    /* I8VEC2  */
  YYSYMBOL_I8VEC3 = 94,                    /* I8VEC3  */
  YYSYMBOL_I8VEC4 = 95,                    /* I8VEC4  */
  YYSYMBOL_U8VEC2 = 96,                    /* U8VEC2  */
  YYSYMBOL_U8VEC3 = 97,                    /* U8VEC3  */
  YYSYMBOL_U8VEC4 = 98,                    /* U8VEC4  */
  YYSYMBOL_DVEC2 = 99,                     /* DVEC2  */
  YYSYMBOL_DVEC3 = 100,                    /* DVEC3  */
  YYSYMBOL_DVEC4 = 101,                    /* DVEC4  */
  YYSYMBOL_DMAT2 = 102,                    /* DMAT2  */
  YYSYMBOL_DMAT3 = 103,                    /* DMAT3  */
  YYSYMBOL_DMAT4 = 104,                    /* DMAT4  */
  YYSYMBOL_F16VEC2 = 105,                  /* F16VEC2  */
  YYSYMBOL_F16VEC3 = 106,                  /* F16VEC3  */
  YYSYMBOL_F16VEC4 = 107,                  /* F16VEC4  */
  YYSYMBOL_F16MAT2 = 108,                  /* F16MAT2  */
  YYSYMBOL_F16MAT3 = 109,                  /* F16MAT3  */
  YYSYMBOL_F16MAT4 = 110,                  /* F16MAT4  */
  YYSYMBOL_F32VEC2 = 111,                  /* F32VEC2  */
  YYSYMBOL_F32VEC3 = 112,                  /* F32VEC3  */
  YYSYMBOL_F32VEC4 = 113,                  /* F32VEC4  */
  YYSYMBOL_F32MAT2 = 114,                  /* F32MAT2  */
  YYSYMBOL_F32MAT3 = 115,                  /* F32MAT3  */
  YYSYMBOL_F32MAT4 = 116,                  /* F32MAT4  */
  YYSYMBOL_F64VEC2 = 117,                  /* F64VEC2  */
  YYSYMBOL_F64VEC3 = 118,                  /* F64VEC3  */
  YYSYMBOL_F64VEC4 = 119,                  /* F64VEC4  */
  YYSYMBOL_F64MAT2 = 120,                  /* F64MAT2  */
  YYSYMBOL_F64MAT3 = 121,                  /* F64MAT3  */
  YYSYMBOL_F64MAT4 = 122,                  /* F64MAT4  */
  YYSYMBOL_DMAT2X2 = 123,                  /* DMAT2X2  */
  YYSYMBOL_DMAT2X3 = 124,                  /* DMAT2X3  */
  YYSYMBOL_DMAT2X4 = 125,                  /* DMAT2X4  */
  YYSYMBOL_DMAT3X2 = 126,                  /* DMAT3X2  */
  YYSYMBOL_DMAT3X3 = 127,                  /* DMAT3X3  */
  YYSYMBOL_DMAT3X4 = 128,                  /* DMAT3X4  */
  YYSYMBOL_DMAT4X2 = 129,                  /* DMAT4X2  */
  YYSYMBOL_DMAT4X3 = 130,                  /* DMAT4X3  */
  YYSYMBOL_DMAT4X4 = 131,                  /* DMAT4X4  */
  YYSYMBOL_F16MAT2X2 = 132,                /* F16MAT2X2  */
  YYSYMBOL_F16MAT2X3 = 133,                /* F16MAT2X3  */
  YYSYMBOL_F16MAT2X4 = 134,                /* F16MAT2X4  */
  YYSYMBOL_F16MAT3X2 = 135,                /* F16MAT3X2  */
  YYSYMBOL_F16MAT3X3 = 136,                /* F16MAT3X3  */
  YYSYMBOL_F16MAT3X4 = 137,                /* F16MAT3X4  */
  YYSYMBOL_F16MAT4X2 = 138,                /* F16MAT4X2  */
  YYSYMBOL_F16MAT4X3 = 139,                /* F16MAT4X3  */
  YYSYMBOL_F16MAT4X4 = 140,                /* F16MAT4X4  */
  YYSYMBOL_F32MAT2X2 = 141,                /* F32MAT2X2  */
  YYSYMBOL_F32MAT2X3 = 142,                /* F32MAT2X3  */
  YYSYMBOL_F32MAT2X4 = 143,                /* F32MAT2X4  */
  YYSYMBOL_F32MAT3X2 = 144,                /* F32MAT3X2  */
  YYSYMBOL_F32MAT3X3 = 145,                /* F32MAT3X3  */
  YYSYMBOL_F32MAT3X4 = 146,                /* F32MAT3X4  */
  YYSYMBOL_F32MAT4X2 = 147,                /* F32MAT4X2  */
  YYSYMBOL_F32MAT4X3 = 148,                /* F32MAT4X3  */
  YYSYMBOL_F32MAT4X4 = 149,                /* F32MAT4X4  */
  YYSYMBOL_F64MAT2X2 = 150,                /* F64MAT2X2  */
  YYSYMBOL_F64MAT2X3 = 151,                /* F64MAT2X3  */
  YYSYMBOL_F64MAT2X4 = 152,                /* F64MAT2X4  */
  YYSYMBOL_F64MAT3X2 = 153,                /* F64MAT3X2  */
  YYSYMBOL_F64MAT3X3 = 154,                /* F64MAT3X3  */
  YYSYMBOL_F64MAT3X4 = 155,                /* F64MAT3X4  */
  YYSYMBOL_F64MAT4X2 = 156,                /* F64MAT4X2  */
  YYSYMBOL_F64MAT4X3 = 157,                /* F64MAT4X3  */
  YYSYMBOL_F64MAT4X4 = 158,                /* F64MAT4X4  */
  YYSYMBOL_ATOMIC_UINT = 159,              /* ATOMIC_UINT  */
  YYSYMBOL_ACCSTRUCTNV = 160,              /* ACCSTRUCTNV  */
  YYSYMBOL_ACCSTRUCTEXT = 161,             /* ACCSTRUCTEXT  */
  YYSYMBOL_RAYQUERYEXT = 162,              /* RAYQUERYEXT  */
  YYSYMBOL_FCOOPMATNV = 163,               /* FCOOPMATNV  */
  YYSYMBOL_ICOOPMATNV = 164,               /* ICOOPMATNV  */
  YYSYMBOL_UCOOPMATNV = 165,               /* UCOOPMATNV  */
  YYSYMBOL_SAMPLERCUBEARRAY = 166,         /* SAMPLERCUBEARRAY  */
  YYSYMBOL_SAMPLERCUBEARRAYSHADOW = 167,   /* SAMPLERCUBEARRAYSHADOW  */
  YYSYMBOL_ISAMPLERCUBEARRAY = 168,        /* ISAMPLERCUBEARRAY  */
  YYSYMBOL_USAMPLERCUBEARRAY = 169,        /* USAMPLERCUBEARRAY  */
  YYSYMBOL_SAMPLER1D = 170,                /* SAMPLER1D  */
  YYSYMBOL_SAMPLER1DARRAY = 171,           /* SAMPLER1DARRAY  */
  YYSYMBOL_SAMPLER1DARRAYSHADOW = 172,     /* SAMPLER1DARRAYSHADOW  */
  YYSYMBOL_ISAMPLER1D = 173,               /* ISAMPLER1D  */
  YYSYMBOL_SAMPLER1DSHADOW = 174,          /* SAMPLER1DSHADOW  */
  YYSYMBOL_SAMPLER2DRECT = 175,            /* SAMPLER2DRECT  */
  YYSYMBOL_SAMPLER2DRECTSHADOW = 176,      /* SAMPLER2DRECTSHADOW  */
  YYSYMBOL_ISAMPLER2DRECT = 177,           /* ISAMPLER2DRECT  */
  YYSYMBOL_USAMPLER2DRECT = 178,           /* USAMPLER2DRECT  */
  YYSYMBOL_SAMPLERBUFFER = 179,            /* SAMPLERBUFFER  */
  YYSYMBOL_ISAMPLERBUFFER = 180,           /* ISAMPLERBUFFER  */
  YYSYMBOL_USAMPLERBUFFER = 181,           /* USAMPLERBUFFER  */
  YYSYMBOL_SAMPLER2DMS = 182,              /* SAMPLER2DMS  */
  YYSYMBOL_ISAMPLER2DMS = 183,             /* ISAMPLER2DMS  */
  YYSYMBOL_USAMPLER2DMS = 184,             /* USAMPLER2DMS  */
  YYSYMBOL_SAMPLER2DMSARRAY = 185,         /* SAMPLER2DMSARRAY  */
  YYSYMBOL_ISAMPLER2DMSARRAY = 186,        /* ISAMPLER2DMSARRAY  */
  YYSYMBOL_USAMPLER2DMSARRAY = 187,        /* USAMPLER2DMSARRAY  */
  YYSYMBOL_SAMPLEREXTERNALOES = 188,       /* SAMPLEREXTERNALOES  */
  YYSYMBOL_SAMPLEREXTERNAL2DY2YEXT = 189,  /* SAMPLEREXTERNAL2DY2YEXT  */
  YYSYMBOL_ISAMPLER1DARRAY = 190,          /* ISAMPLER1DARRAY  */
  YYSYMBOL_USAMPLER1D = 191,               /* USAMPLER1D  */
  YYSYMBOL_USAMPLER1DARRAY = 192,          /* USAMPLER1DARRAY  */
  YYSYMBOL_F16SAMPLER1D = 193,             /* F16SAMPLER1D  */
  YYSYMBOL_F16SAMPLER2D = 194,             /* F16SAMPLER2D  */
  YYSYMBOL_F16SAMPLER3D = 195,             /* F16SAMPLER3D  */
  YYSYMBOL_F16SAMPLER2DRECT = 196,         /* F16SAMPLER2DRECT  */
  YYSYMBOL_F16SAMPLERCUBE = 197,           /* F16SAMPLERCUBE  */
  YYSYMBOL_F16SAMPLER1DARRAY = 198,        /* F16SAMPLER1DARRAY  */
  YYSYMBOL_F16SAMPLER2DARRAY = 199,        /* F16SAMPLER2DARRAY  */
  YYSYMBOL_F16SAMPLERCUBEARRAY = 200,      /* F16SAMPLERCUBEARRAY  */
  YYSYMBOL_F16SAMPLERBUFFER = 201,         /* F16SAMPLERBUFFER  */
  YYSYMBOL_F16SAMPLER2DMS = 202,           /* F16SAMPLER2DMS  */
  YYSYMBOL_F16SAMPLER2DMSARRAY = 203,      /* F16SAMPLER2DMSARRAY  */
  YYSYMBOL_F16SAMPLER1DSHADOW = 204,       /* F16SAMPLER1DSHADOW  */
  YYSYMBOL_F16SAMPLER2DSHADOW = 205,       /* F16SAMPLER2DSHADOW  */
  YYSYMBOL_F16SAMPLER1DARRAYSHADOW = 206,  /* F16SAMPLER1DARRAYSHADOW  */
  YYSYMBOL_F16SAMPLER2DARRAYSHADOW = 207,  /* F16SAMPLER2DARRAYSHADOW  */
  YYSYMBOL_F16SAMPLER2DRECTSHADOW = 208,   /* F16SAMPLER2DRECTSHADOW  */
  YYSYMBOL_F16SAMPLERCUBESHADOW = 209,     /* F16SAMPLERCUBESHADOW  */
  YYSYMBOL_F16SAMPLERCUBEARRAYSHADOW = 210, /* F16SAMPLERCUBEARRAYSHADOW  */
  YYSYMBOL_IMAGE1D = 211,                  /* IMAGE1D  */
  YYSYMBOL_IIMAGE1D = 212,                 /* IIMAGE1D  */
  YYSYMBOL_UIMAGE1D = 213,                 /* UIMAGE1D  */
  YYSYMBOL_IMAGE2D = 214,                  /* IMAGE2D  */
  YYSYMBOL_IIMAGE2D = 215,                 /* IIMAGE2D  */
  YYSYMBOL_UIMAGE2D = 216,                 /* UIMAGE2D  */
  YYSYMBOL_IMAGE3D = 217,                  /* IMAGE3D  */
  YYSYMBOL_IIMAGE3D = 218,                 /* IIMAGE3D  */
  YYSYMBOL_UIMAGE3D = 219,                 /* UIMAGE3D  */
  YYSYMBOL_IMAGE2DRECT = 220,              /* IMAGE2DRECT  */
  YYSYMBOL_IIMAGE2DRECT = 221,             /* IIMAGE2DRECT  */
  YYSYMBOL_UIMAGE2DRECT = 222,             /* UIMAGE2DRECT  */
  YYSYMBOL_IMAGECUBE = 223,                /* IMAGECUBE  */
  YYSYMBOL_IIMAGECUBE = 224,               /* IIMAGECUBE  */
  YYSYMBOL_UIMAGECUBE = 225,               /* UIMAGECUBE  */
  YYSYMBOL_IMAGEBUFFER = 226,              /* IMAGEBUFFER  */
  YYSYMBOL_IIMAGEBUFFER = 227,             /* IIMAGEBUFFER  */
  YYSYMBOL_UIMAGEBUFFER = 228,             /* UIMAGEBUFFER  */
  YYSYMBOL_IMAGE1DARRAY = 229,             /* IMAGE1DARRAY  */
  YYSYMBOL_IIMAGE1DARRAY = 230,            /* IIMAGE1DARRAY  */
  YYSYMBOL_UIMAGE1DARRAY = 231,            /* UIMAGE1DARRAY  */
  YYSYMBOL_IMAGE2DARRAY = 232,             /* IMAGE2DARRAY  */
  YYSYMBOL_IIMAGE2DARRAY = 233,            /* IIMAGE2DARRAY  */
  YYSYMBOL_UIMAGE2DARRAY = 234,            /* UIMAGE2DARRAY  */
  YYSYMBOL_IMAGECUBEARRAY = 235,           /* IMAGECUBEARRAY  */
  YYSYMBOL_IIMAGECUBEARRAY = 236,          /* IIMAGECUBEARRAY  */
  YYSYMBOL_UIMAGECUBEARRAY = 237,          /* UIMAGECUBEARRAY  */
  YYSYMBOL_IMAGE2DMS = 238,                /* IMAGE2DMS  */
  YYSYMBOL_IIMAGE2DMS = 239,               /* IIMAGE2DMS  */
  YYSYMBOL_UIMAGE2DMS = 240,               /* UIMAGE2DMS  */
  YYSYMBOL_IMAGE2DMSARRAY = 241,           /* IMAGE2DMSARRAY  */
  YYSYMBOL_IIMAGE2DMSARRAY = 242,          /* IIMAGE2DMSARRAY  */
  YYSYMBOL_UIMAGE2DMSARRAY = 243,          /* UIMAGE2DMSARRAY  */
  YYSYMBOL_F16IMAGE1D = 244,               /* F16IMAGE1D  */
  YYSYMBOL_F16IMAGE2D = 245,               /* F16IMAGE2D  */
  YYSYMBOL_F16IMAGE3D = 246,               /* F16IMAGE3D  */
  YYSYMBOL_F16IMAGE2DRECT = 247,           /* F16IMAGE2DRECT  */
  YYSYMBOL_F16IMAGECUBE = 248,             /* F16IMAGECUBE  */
  YYSYMBOL_F16IMAGE1DARRAY = 249,          /* F16IMAGE1DARRAY  */
  YYSYMBOL_F16IMAGE2DARRAY = 250,          /* F16IMAGE2DARRAY  */
  YYSYMBOL_F16IMAGECUBEARRAY = 251,        /* F16IMAGECUBEARRAY  */
  YYSYMBOL_F16IMAGEBUFFER = 252,           /* F16IMAGEBUFFER  */
  YYSYMBOL_F16IMAGE2DMS = 253,             /* F16IMAGE2DMS  */
  YYSYMBOL_F16IMAGE2DMSARRAY = 254,        /* F16IMAGE2DMSARRAY  */
  YYSYMBOL_I64IMAGE1D = 255,               /* I64IMAGE1D  */
  YYSYMBOL_U64IMAGE1D = 256,               /* U64IMAGE1D  */
  YYSYMBOL_I64IMAGE2D = 257,               /* I64IMAGE2D  */
  YYSYMBOL_U64IMAGE2D = 258,               /* U64IMAGE2D  */
  YYSYMBOL_I64IMAGE3D = 259,               /* I64IMAGE3D  */
  YYSYMBOL_U64IMAGE3D = 260,               /* U64IMAGE3D  */
  YYSYMBOL_I64IMAGE2DRECT = 261,           /* I64IMAGE2DRECT  */
  YYSYMBOL_U64IMAGE2DRECT = 262,           /* U64IMAGE2DRECT  */
  YYSYMBOL_I64IMAGECUBE = 263,             /* I64IMAGECUBE  */
  YYSYMBOL_U64IMAGECUBE = 264,             /* U64IMAGECUBE  */
  YYSYMBOL_I64IMAGEBUFFER = 265,           /* I64IMAGEBUFFER  */
  YYSYMBOL_U64IMAGEBUFFER = 266,           /* U64IMAGEBUFFER  */
  YYSYMBOL_I64IMAGE1DARRAY = 267,          /* I64IMAGE1DARRAY  */
  YYSYMBOL_U64IMAGE1DARRAY = 268,          /* U64IMAGE1DARRAY  */
  YYSYMBOL_I64IMAGE2DARRAY = 269,          /* I64IMAGE2DARRAY  */
  YYSYMBOL_U64IMAGE2DARRAY = 270,          /* U64IMAGE2DARRAY  */
  YYSYMBOL_I64IMAGECUBEARRAY = 271,        /* I64IMAGECUBEARRAY  */
  YYSYMBOL_U64IMAGECUBEARRAY = 272,        /* U64IMAGECUBEARRAY  */
  YYSYMBOL_I64IMAGE2DMS = 273,             /* I64IMAGE2DMS  */
  YYSYMBOL_U64IMAGE2DMS = 274,             /* U64IMAGE2DMS  */
  YYSYMBOL_I64IMAGE2DMSARRAY = 275,        /* I64IMAGE2DMSARRAY  */
  YYSYMBOL_U64IMAGE2DMSARRAY = 276,        /* U64IMAGE2DMSARRAY  */
  YYSYMBOL_TEXTURECUBEARRAY = 277,         /* TEXTURECUBEARRAY  */
  YYSYMBOL_ITEXTURECUBEARRAY = 278,        /* ITEXTURECUBEARRAY  */
  YYSYMBOL_UTEXTURECUBEARRAY = 279,        /* UTEXTURECUBEARRAY  */
  YYSYMBOL_TEXTURE1D = 280,                /* TEXTURE1D  */
  YYSYMBOL_ITEXTURE1D = 281,               /* ITEXTURE1D  */
  YYSYMBOL_UTEXTURE1D = 282,               /* UTEXTURE1D  */
  YYSYMBOL_TEXTURE1DARRAY = 283,           /* TEXTURE1DARRAY  */
  YYSYMBOL_ITEXTURE1DARRAY = 284,          /* ITEXTURE1DARRAY  */
  YYSYMBOL_UTEXTURE1DARRAY = 285,          /* UTEXTURE1DARRAY  */
  YYSYMBOL_TEXTURE2DRECT = 286,            /* TEXTURE2DRECT  */
  YYSYMBOL_ITEXTURE2DRECT = 287,           /* ITEXTURE2DRECT  */
  YYSYMBOL_UTEXTURE2DRECT = 288,           /* UTEXTURE2DRECT  */
  YYSYMBOL_TEXTUREBUFFER = 289,            /* TEXTUREBUFFER  */
  YYSYMBOL_ITEXTUREBUFFER = 290,           /* ITEXTUREBUFFER  */
  YYSYMBOL_UTEXTUREBUFFER = 291,           /* UTEXTUREBUFFER  */
  YYSYMBOL_TEXTURE2DMS = 292,              /* TEXTURE2DMS  */
  YYSYMBOL_ITEXTURE2DMS = 293,             /* ITEXTURE2DMS  */
  YYSYMBOL_UTEXTURE2DMS = 294,             /* UTEXTURE2DMS  */
  YYSYMBOL_TEXTURE2DMSARRAY = 295,         /* TEXTURE2DMSARRAY  */
  YYSYMBOL_ITEXTURE2DMSARRAY = 296,        /* ITEXTURE2DMSARRAY  */
  YYSYMBOL_UTEXTURE2DMSARRAY = 297,        /* UTEXTURE2DMSARRAY  */
  YYSYMBOL_F16TEXTURE1D = 298,             /* F16TEXTURE1D  */
  YYSYMBOL_F16TEXTURE2D = 299,             /* F16TEXTURE2D  */
  YYSYMBOL_F16TEXTURE3D = 300,             /* F16TEXTURE3D  */
  YYSYMBOL_F16TEXTURE2DRECT = 301,         /* F16TEXTURE2DRECT  */
  YYSYMBOL_F16TEXTURECUBE = 302,           /* F16TEXTURECUBE  */
  YYSYMBOL_F16TEXTURE1DARRAY = 303,        /* F16TEXTURE1DARRAY  */
  YYSYMBOL_F16TEXTURE2DARRAY = 304,        /* F16TEXTURE2DARRAY  */
  YYSYMBOL_F16TEXTURECUBEARRAY = 305,      /* F16TEXTURECUBEARRAY  */
  YYSYMBOL_F16TEXTUREBUFFER = 306,         /* F16TEXTUREBUFFER  */
  YYSYMBOL_F16TEXTURE2DMS = 307,           /* F16TEXTURE2DMS  */
  YYSYMBOL_F16TEXTURE2DMSARRAY = 308,      /* F16TEXTURE2DMSARRAY  */
  YYSYMBOL_SUBPASSINPUT = 309,             /* SUBPASSINPUT  */
  YYSYMBOL_SUBPASSINPUTMS = 310,           /* SUBPASSINPUTMS  */
  YYSYMBOL_ISUBPASSINPUT = 311,            /* ISUBPASSINPUT  */
  YYSYMBOL_ISUBPASSINPUTMS = 312,          /* ISUBPASSINPUTMS  */
  YYSYMBOL_USUBPASSINPUT = 313,            /* USUBPASSINPUT  */
  YYSYMBOL_USUBPASSINPUTMS = 314,          /* USUBPASSINPUTMS  */
  YYSYMBOL_F16SUBPASSINPUT = 315,          /* F16SUBPASSINPUT  */
  YYSYMBOL_F16SUBPASSINPUTMS = 316,        /* F16SUBPASSINPUTMS  */
  YYSYMBOL_LEFT_OP = 317,                  /* LEFT_OP  */
  YYSYMBOL_RIGHT_OP = 318,                 /* RIGHT_OP  */
  YYSYMBOL_INC_OP = 319,                   /* INC_OP  */
  YYSYMBOL_DEC_OP = 320,                   /* DEC_OP  */
  YYSYMBOL_LE_OP = 321,                    /* LE_OP  */
  YYSYMBOL_GE_OP = 322,                    /* GE_OP  */
  YYSYMBOL_EQ_OP = 323,                    /* EQ_OP  */
  YYSYMBOL_NE_OP = 324,                    /* NE_OP  */
  YYSYMBOL_AND_OP = 325,                   /* AND_OP  */
  YYSYMBOL_OR_OP = 326,                    /* OR_OP  */
  YYSYMBOL_XOR_OP = 327,                   /* XOR_OP  */
  YYSYMBOL_MUL_ASSIGN = 328,               /* MUL_ASSIGN  */
  YYSYMBOL_DIV_ASSIGN = 329,               /* DIV_ASSIGN  */
  YYSYMBOL_ADD_ASSIGN = 330,               /* ADD_ASSIGN  */
  YYSYMBOL_MOD_ASSIGN = 331,               /* MOD_ASSIGN  */
  YYSYMBOL_LEFT_ASSIGN = 332,              /* LEFT_ASSIGN  */
  YYSYMBOL_RIGHT_ASSIGN = 333,             /* RIGHT_ASSIGN  */
  YYSYMBOL_AND_ASSIGN = 334,               /* AND_ASSIGN  */
  YYSYMBOL_XOR_ASSIGN = 335,               /* XOR_ASSIGN  */
  YYSYMBOL_OR_ASSIGN = 336,                /* OR_ASSIGN  */
  YYSYMBOL_SUB_ASSIGN = 337,               /* SUB_ASSIGN  */
  YYSYMBOL_STRING_LITERAL = 338,           /* STRING_LITERAL  */
  YYSYMBOL_LEFT_PAREN = 339,               /* LEFT_PAREN  */
  YYSYMBOL_RIGHT_PAREN = 340,              /* RIGHT_PAREN  */
  YYSYMBOL_LEFT_BRACKET = 341,             /* LEFT_BRACKET  */
  YYSYMBOL_RIGHT_BRACKET = 342,            /* RIGHT_BRACKET  */
  YYSYMBOL_LEFT_BRACE = 343,               /* LEFT_BRACE  */
  YYSYMBOL_RIGHT_BRACE = 344,              /* RIGHT_BRACE  */
  YYSYMBOL_DOT = 345,                      /* DOT  */
  YYSYMBOL_COMMA = 346,                    /* COMMA  */
  YYSYMBOL_COLON = 347,                    /* COLON  */
  YYSYMBOL_EQUAL = 348,                    /* EQUAL  */
  YYSYMBOL_SEMICOLON = 349,                /* SEMICOLON  */
  YYSYMBOL_BANG = 350,                     /* BANG  */
  YYSYMBOL_DASH = 351,                     /* DASH  */
  YYSYMBOL_TILDE = 352,                    /* TILDE  */
  YYSYMBOL_PLUS = 353,                     /* PLUS  */
  YYSYMBOL_STAR = 354,                     /* STAR  */
  YYSYMBOL_SLASH = 355,                    /* SLASH  */
  YYSYMBOL_PERCENT = 356,                  /* PERCENT  */
  YYSYMBOL_LEFT_ANGLE = 357,               /* LEFT_ANGLE  */
  YYSYMBOL_RIGHT_ANGLE = 358,              /* RIGHT_ANGLE  */
  YYSYMBOL_VERTICAL_BAR = 359,             /* VERTICAL_BAR  */
  YYSYMBOL_CARET = 360,                    /* CARET  */
  YYSYMBOL_AMPERSAND = 361,                /* AMPERSAND  */
  YYSYMBOL_QUESTION = 362,                 /* QUESTION  */
  YYSYMBOL_INVARIANT = 363,                /* INVARIANT  */
  YYSYMBOL_HIGH_PRECISION = 364,           /* HIGH_PRECISION  */
  YYSYMBOL_MEDIUM_PRECISION = 365,         /* MEDIUM_PRECISION  */
  YYSYMBOL_LOW_PRECISION = 366,            /* LOW_PRECISION  */
  YYSYMBOL_PRECISION = 367,                /* PRECISION  */
  YYSYMBOL_PACKED = 368,                   /* PACKED  */
  YYSYMBOL_RESOURCE = 369,                 /* RESOURCE  */
  YYSYMBOL_SUPERP = 370,                   /* SUPERP  */
  YYSYMBOL_FLOATCONSTANT = 371,            /* FLOATCONSTANT  */
  YYSYMBOL_INTCONSTANT = 372,              /* INTCONSTANT  */
  YYSYMBOL_UINTCONSTANT = 373,             /* UINTCONSTANT  */
  YYSYMBOL_BOOLCONSTANT = 374,             /* BOOLCONSTANT  */
  YYSYMBOL_IDENTIFIER = 375,               /* IDENTIFIER  */
  YYSYMBOL_TYPE_NAME = 376,                /* TYPE_NAME  */
  YYSYMBOL_CENTROID = 377,                 /* CENTROID  */
  YYSYMBOL_IN = 378,                       /* IN  */
  YYSYMBOL_OUT = 379,                      /* OUT  */
  YYSYMBOL_INOUT = 380,                    /* INOUT  */
  YYSYMBOL_STRUCT = 381,                   /* STRUCT  */
  YYSYMBOL_VOID = 382,                     /* VOID  */
  YYSYMBOL_WHILE = 383,                    /* WHILE  */
  YYSYMBOL_BREAK = 384,                    /* BREAK  */
  YYSYMBOL_CONTINUE = 385,                 /* CONTINUE  */
  YYSYMBOL_DO = 386,                       /* DO  */
  YYSYMBOL_ELSE = 387,                     /* ELSE  */
  YYSYMBOL_FOR = 388,                      /* FOR  */
  YYSYMBOL_IF = 389,                       /* IF  */
  YYSYMBOL_DISCARD = 390,                  /* DISCARD  */
  YYSYMBOL_RETURN = 391,                   /* RETURN  */
  YYSYMBOL_SWITCH = 392,                   /* SWITCH  */
  YYSYMBOL_CASE = 393,                     /* CASE  */
  YYSYMBOL_DEFAULT = 394,                  /* DEFAULT  */
  YYSYMBOL_TERMINATE_INVOCATION = 395,     /* TERMINATE_INVOCATION  */
  YYSYMBOL_TERMINATE_RAY = 396,            /* TERMINATE_RAY  */
  YYSYMBOL_IGNORE_INTERSECTION = 397,      /* IGNORE_INTERSECTION  */
  YYSYMBOL_UNIFORM = 398,                  /* UNIFORM  */
  YYSYMBOL_SHARED = 399,                   /* SHARED  */
  YYSYMBOL_BUFFER = 400,                   /* BUFFER  */
  YYSYMBOL_FLAT = 401,                     /* FLAT  */
  YYSYMBOL_SMOOTH = 402,                   /* SMOOTH  */
  YYSYMBOL_LAYOUT = 403,                   /* LAYOUT  */
  YYSYMBOL_DOUBLECONSTANT = 404,           /* DOUBLECONSTANT  */
  YYSYMBOL_INT16CONSTANT = 405,            /* INT16CONSTANT  */
  YYSYMBOL_UINT16CONSTANT = 406,           /* UINT16CONSTANT  */
  YYSYMBOL_FLOAT16CONSTANT = 407,          /* FLOAT16CONSTANT  */
  YYSYMBOL_INT32CONSTANT = 408,            /* INT32CONSTANT  */
  YYSYMBOL_UINT32CONSTANT = 409,           /* UINT32CONSTANT  */
  YYSYMBOL_INT64CONSTANT = 410,            /* INT64CONSTANT  */
  YYSYMBOL_UINT64CONSTANT = 411,           /* UINT64CONSTANT  */
  YYSYMBOL_SUBROUTINE = 412,               /* SUBROUTINE  */
  YYSYMBOL_DEMOTE = 413,                   /* DEMOTE  */
  YYSYMBOL_PAYLOADNV = 414,                /* PAYLOADNV  */
  YYSYMBOL_PAYLOADINNV = 415,              /* PAYLOADINNV  */
  YYSYMBOL_HITATTRNV = 416,                /* HITATTRNV  */
  YYSYMBOL_CALLDATANV = 417,               /* CALLDATANV  */
  YYSYMBOL_CALLDATAINNV = 418,             /* CALLDATAINNV  */
  YYSYMBOL_PAYLOADEXT = 419,               /* PAYLOADEXT  */
  YYSYMBOL_PAYLOADINEXT = 420,             /* PAYLOADINEXT  */
  YYSYMBOL_HITATTREXT = 421,               /* HITATTREXT  */
  YYSYMBOL_CALLDATAEXT = 422,              /* CALLDATAEXT  */
  YYSYMBOL_CALLDATAINEXT = 423,            /* CALLDATAINEXT  */
  YYSYMBOL_PATCH = 424,                    /* PATCH  */
  YYSYMBOL_SAMPLE = 425,                   /* SAMPLE  */
  YYSYMBOL_NONUNIFORM = 426,               /* NONUNIFORM  */
  YYSYMBOL_COHERENT = 427,                 /* COHERENT  */
  YYSYMBOL_VOLATILE = 428,                 /* VOLATILE  */
  YYSYMBOL_RESTRICT = 429,                 /* RESTRICT  */
  YYSYMBOL_READONLY = 430,                 /* READONLY  */
  YYSYMBOL_WRITEONLY = 431,                /* WRITEONLY  */
  YYSYMBOL_DEVICECOHERENT = 432,           /* DEVICECOHERENT  */
  YYSYMBOL_QUEUEFAMILYCOHERENT = 433,      /* QUEUEFAMILYCOHERENT  */
  YYSYMBOL_WORKGROUPCOHERENT = 434,        /* WORKGROUPCOHERENT  */
  YYSYMBOL_SUBGROUPCOHERENT = 435,         /* SUBGROUPCOHERENT  */
  YYSYMBOL_NONPRIVATE = 436,               /* NONPRIVATE  */
  YYSYMBOL_SHADERCALLCOHERENT = 437,       /* SHADERCALLCOHERENT  */
  YYSYMBOL_NOPERSPECTIVE = 438,            /* NOPERSPECTIVE  */
  YYSYMBOL_EXPLICITINTERPAMD = 439,        /* EXPLICITINTERPAMD  */
  YYSYMBOL_PERVERTEXNV = 440,              /* PERVERTEXNV  */
  YYSYMBOL_PERPRIMITIVENV = 441,           /* PERPRIMITIVENV  */
  YYSYMBOL_PERVIEWNV = 442,                /* PERVIEWNV  */
  YYSYMBOL_PERTASKNV = 443,                /* PERTASKNV  */
  YYSYMBOL_PRECISE = 444,                  /* PRECISE  */
  YYSYMBOL_YYACCEPT = 445,                 /* $accept  */
  YYSYMBOL_variable_identifier = 446,      /* variable_identifier  */
  YYSYMBOL_primary_expression = 447,       /* primary_expression  */
  YYSYMBOL_postfix_expression = 448,       /* postfix_expression  */
  YYSYMBOL_integer_expression = 449,       /* integer_expression  */
  YYSYMBOL_function_call = 450,            /* function_call  */
  YYSYMBOL_function_call_or_method = 451,  /* function_call_or_method  */
  YYSYMBOL_function_call_generic = 452,    /* function_call_generic  */
  YYSYMBOL_function_call_header_no_parameters = 453, /* function_call_header_no_parameters  */
  YYSYMBOL_function_call_header_with_parameters = 454, /* function_call_header_with_parameters  */
  YYSYMBOL_function_call_header = 455,     /* function_call_header  */
  YYSYMBOL_function_identifier = 456,      /* function_identifier  */
  YYSYMBOL_unary_expression = 457,         /* unary_expression  */
  YYSYMBOL_unary_operator = 458,           /* unary_operator  */
  YYSYMBOL_multiplicative_expression = 459, /* multiplicative_expression  */
  YYSYMBOL_additive_expression = 460,      /* additive_expression  */
  YYSYMBOL_shift_expression = 461,         /* shift_expression  */
  YYSYMBOL_relational_expression = 462,    /* relational_expression  */
  YYSYMBOL_equality_expression = 463,      /* equality_expression  */
  YYSYMBOL_and_expression = 464,           /* and_expression  */
  YYSYMBOL_exclusive_or_expression = 465,  /* exclusive_or_expression  */
  YYSYMBOL_inclusive_or_expression = 466,  /* inclusive_or_expression  */
  YYSYMBOL_logical_and_expression = 467,   /* logical_and_expression  */
  YYSYMBOL_logical_xor_expression = 468,   /* logical_xor_expression  */
  YYSYMBOL_logical_or_expression = 469,    /* logical_or_expression  */
  YYSYMBOL_conditional_expression = 470,   /* conditional_expression  */
  YYSYMBOL_471_1 = 471,                    /* $@1  */
  YYSYMBOL_assignment_expression = 472,    /* assignment_expression  */
  YYSYMBOL_assignment_operator = 473,      /* assignment_operator  */
  YYSYMBOL_expression = 474,               /* expression  */
  YYSYMBOL_constant_expression = 475,      /* constant_expression  */
  YYSYMBOL_declaration = 476,              /* declaration  */
  YYSYMBOL_block_structure = 477,          /* block_structure  */
  YYSYMBOL_478_2 = 478,                    /* $@2  */
  YYSYMBOL_identifier_list = 479,          /* identifier_list  */
  YYSYMBOL_function_prototype = 480,       /* function_prototype  */
  YYSYMBOL_function_declarator = 481,      /* function_declarator  */
  YYSYMBOL_function_header_with_parameters = 482, /* function_header_with_parameters  */
  YYSYMBOL_function_header = 483,          /* function_header  */
  YYSYMBOL_parameter_declarator = 484,     /* parameter_declarator  */
  YYSYMBOL_parameter_declaration = 485,    /* parameter_declaration  */
  YYSYMBOL_parameter_type_specifier = 486, /* parameter_type_specifier  */
  YYSYMBOL_init_declarator_list = 487,     /* init_declarator_list  */
  YYSYMBOL_single_declaration = 488,       /* single_declaration  */
  YYSYMBOL_fully_specified_type = 489,     /* fully_specified_type  */
  YYSYMBOL_invariant_qualifier = 490,      /* invariant_qualifier  */
  YYSYMBOL_interpolation_qualifier = 491,  /* interpolation_qualifier  */
  YYSYMBOL_layout_qualifier = 492,         /* layout_qualifier  */
  YYSYMBOL_layout_qualifier_id_list = 493, /* layout_qualifier_id_list  */
  YYSYMBOL_layout_qualifier_id = 494,      /* layout_qualifier_id  */
  YYSYMBOL_precise_qualifier = 495,        /* precise_qualifier  */
  YYSYMBOL_type_qualifier = 496,           /* type_qualifier  */
  YYSYMBOL_single_type_qualifier = 497,    /* single_type_qualifier  */
  YYSYMBOL_storage_qualifier = 498,        /* storage_qualifier  */
  YYSYMBOL_non_uniform_qualifier = 499,    /* non_uniform_qualifier  */
  YYSYMBOL_type_name_list = 500,           /* type_name_list  */
  YYSYMBOL_type_specifier = 501,           /* type_specifier  */
  YYSYMBOL_array_specifier = 502,          /* array_specifier  */
  YYSYMBOL_type_parameter_specifier_opt = 503, /* type_parameter_specifier_opt  */
  YYSYMBOL_type_parameter_specifier = 504, /* type_parameter_specifier  */
  YYSYMBOL_type_parameter_specifier_list = 505, /* type_parameter_specifier_list  */
  YYSYMBOL_type_specifier_nonarray = 506,  /* type_specifier_nonarray  */
  YYSYMBOL_precision_qualifier = 507,      /* precision_qualifier  */
  YYSYMBOL_struct_specifier = 508,         /* struct_specifier  */
  YYSYMBOL_509_3 = 509,                    /* $@3  */
  YYSYMBOL_510_4 = 510,                    /* $@4  */
  YYSYMBOL_struct_declaration_list = 511,  /* struct_declaration_list  */
  YYSYMBOL_struct_declaration = 512,       /* struct_declaration  */
  YYSYMBOL_struct_declarator_list = 513,   /* struct_declarator_list  */
  YYSYMBOL_struct_declarator = 514,        /* struct_declarator  */
  YYSYMBOL_initializer = 515,              /* initializer  */
  YYSYMBOL_initializer_list = 516,         /* initializer_list  */
  YYSYMBOL_declaration_statement = 517,    /* declaration_statement  */
  YYSYMBOL_statement = 518,                /* statement  */
  YYSYMBOL_simple_statement = 519,         /* simple_statement  */
  YYSYMBOL_demote_statement = 520,         /* demote_statement  */
  YYSYMBOL_compound_statement = 521,       /* compound_statement  */
  YYSYMBOL_522_5 = 522,                    /* $@5  */
  YYSYMBOL_523_6 = 523,                    /* $@6  */
  YYSYMBOL_statement_no_new_scope = 524,   /* statement_no_new_scope  */
  YYSYMBOL_statement_scoped = 525,         /* statement_scoped  */
  YYSYMBOL_526_7 = 526,                    /* $@7  */
  YYSYMBOL_527_8 = 527,                    /* $@8  */
  YYSYMBOL_compound_statement_no_new_scope = 528, /* compound_statement_no_new_scope  */
  YYSYMBOL_statement_list = 529,           /* statement_list  */
  YYSYMBOL_expression_statement = 530,     /* expression_statement  */
  YYSYMBOL_selection_statement = 531,      /* selection_statement  */
  YYSYMBOL_selection_statement_nonattributed = 532, /* selection_statement_nonattributed  */
  YYSYMBOL_selection_rest_statement = 533, /* selection_rest_statement  */
  YYSYMBOL_condition = 534,                /* condition  */
  YYSYMBOL_switch_statement = 535,         /* switch_statement  */
  YYSYMBOL_switch_statement_nonattributed = 536, /* switch_statement_nonattributed  */
  YYSYMBOL_537_9 = 537,                    /* $@9  */
  YYSYMBOL_switch_statement_list = 538,    /* switch_statement_list  */
  YYSYMBOL_case_label = 539,               /* case_label  */
  YYSYMBOL_iteration_statement = 540,      /* iteration_statement  */
  YYSYMBOL_iteration_statement_nonattributed = 541, /* iteration_statement_nonattributed  */
  YYSYMBOL_542_10 = 542,                   /* $@10  */
  YYSYMBOL_543_11 = 543,                   /* $@11  */
  YYSYMBOL_544_12 = 544,                   /* $@12  */
  YYSYMBOL_for_init_statement = 545,       /* for_init_statement  */
  YYSYMBOL_conditionopt = 546,             /* conditionopt  */
  YYSYMBOL_for_rest_statement = 547,       /* for_rest_statement  */
  YYSYMBOL_jump_statement = 548,           /* jump_statement  */
  YYSYMBOL_translation_unit = 549,         /* translation_unit  */
  YYSYMBOL_external_declaration = 550,     /* external_declaration  */
  YYSYMBOL_function_definition = 551,      /* function_definition  */
  YYSYMBOL_552_13 = 552,                   /* $@13  */
  YYSYMBOL_attribute = 553,                /* attribute  */
  YYSYMBOL_attribute_list = 554,           /* attribute_list  */
  YYSYMBOL_single_attribute = 555          /* single_attribute  */
};
typedef enum yysymbol_kind_t yysymbol_kind_t;


/* Second part of user prologue.  */
#line 133 "MachineIndependent/glslang.y"


/* windows only pragma */
#ifdef _MSC_VER
    #pragma warning(disable : 4065)
    #pragma warning(disable : 4127)
    #pragma warning(disable : 4244)
#endif

#define parseContext (*pParseContext)
#define yyerror(context, msg) context->parserError(msg)

extern int yylex(YYSTYPE*, TParseContext&);


#line 702 "MachineIndependent/glslang_tab.cpp"


#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))


/* Stored state numbers (used for stacks). */
typedef yytype_int16 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && ! defined __ICC && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                            \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

#if 1

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* 1 */

#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  416
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   10112

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  445
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  111
/* YYNRULES -- Number of rules.  */
#define YYNRULES  616
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  764

/* YYMAXUTOK -- Last valid token kind.  */
#define YYMAXUTOK   699


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK                     \
   ? YY_CAST (yysymbol_kind_t, yytranslate[YYX])        \
   : YYSYMBOL_YYUNDEF)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int16 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   139,   140,   141,   142,   143,   144,
     145,   146,   147,   148,   149,   150,   151,   152,   153,   154,
     155,   156,   157,   158,   159,   160,   161,   162,   163,   164,
     165,   166,   167,   168,   169,   170,   171,   172,   173,   174,
     175,   176,   177,   178,   179,   180,   181,   182,   183,   184,
     185,   186,   187,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   197,   198,   199,   200,   201,   202,   203,   204,
     205,   206,   207,   208,   209,   210,   211,   212,   213,   214,
     215,   216,   217,   218,   219,   220,   221,   222,   223,   224,
     225,   226,   227,   228,   229,   230,   231,   232,   233,   234,
     235,   236,   237,   238,   239,   240,   241,   242,   243,   244,
     245,   246,   247,   248,   249,   250,   251,   252,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,   350,   351,   352,   353,   354,
     355,   356,   357,   358,   359,   360,   361,   362,   363,   364,
     365,   366,   367,   368,   369,   370,   371,   372,   373,   374,
     375,   376,   377,   378,   379,   380,   381,   382,   383,   384,
     385,   386,   387,   388,   389,   390,   391,   392,   393,   394,
     395,   396,   397,   398,   399,   400,   401,   402,   403,   404,
     405,   406,   407,   408,   409,   410,   411,   412,   413,   414,
     415,   416,   417,   418,   419,   420,   421,   422,   423,   424,
     425,   426,   427,   428,   429,   430,   431,   432,   433,   434,
     435,   436,   437,   438,   439,   440,   441,   442,   443,   444
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,   371,   371,   377,   380,   385,   388,   391,   395,   399,
     402,   406,   410,   414,   418,   422,   426,   432,   440,   443,
     446,   449,   452,   457,   465,   472,   479,   485,   489,   496,
     499,   505,   512,   522,   530,   535,   563,   572,   578,   582,
     586,   606,   607,   608,   609,   615,   616,   621,   626,   635,
     636,   641,   649,   650,   656,   665,   666,   671,   676,   681,
     689,   690,   699,   711,   712,   721,   722,   731,   732,   741,
     742,   750,   751,   759,   760,   768,   769,   769,   787,   788,
     804,   808,   812,   816,   821,   825,   829,   833,   837,   841,
     845,   852,   855,   866,   873,   878,   883,   890,   894,   898,
     902,   907,   912,   921,   921,   932,   936,   943,   950,   953,
     960,   968,   988,  1011,  1026,  1051,  1062,  1072,  1082,  1092,
    1101,  1104,  1108,  1112,  1117,  1125,  1132,  1137,  1142,  1147,
    1156,  1166,  1193,  1202,  1209,  1217,  1224,  1231,  1239,  1249,
    1256,  1267,  1273,  1276,  1283,  1287,  1291,  1300,  1310,  1313,
    1324,  1327,  1330,  1334,  1338,  1343,  1347,  1354,  1358,  1363,
    1369,  1375,  1382,  1387,  1395,  1401,  1413,  1427,  1433,  1438,
    1446,  1454,  1462,  1470,  1478,  1486,  1494,  1502,  1509,  1516,
    1520,  1525,  1530,  1535,  1540,  1545,  1550,  1554,  1558,  1562,
    1566,  1572,  1583,  1590,  1593,  1602,  1607,  1617,  1622,  1630,
    1634,  1644,  1647,  1653,  1659,  1666,  1676,  1680,  1684,  1688,
    1693,  1697,  1702,  1707,  1712,  1717,  1722,  1727,  1732,  1737,
    1742,  1748,  1754,  1760,  1765,  1770,  1775,  1780,  1785,  1790,
    1795,  1800,  1805,  1810,  1815,  1821,  1828,  1833,  1838,  1843,
    1848,  1853,  1858,  1863,  1868,  1873,  1878,  1883,  1891,  1899,
    1907,  1913,  1919,  1925,  1931,  1937,  1943,  1949,  1955,  1961,
    1967,  1973,  1979,  1985,  1991,  1997,  2003,  2009,  2015,  2021,
    2027,  2033,  2039,  2045,  2051,  2057,  2063,  2069,  2075,  2081,
    2087,  2093,  2099,  2105,  2113,  2121,  2129,  2137,  2145,  2153,
    2161,  2169,  2177,  2185,  2193,  2201,  2207,  2213,  2219,  2225,
    2231,  2237,  2243,  2249,  2255,  2261,  2267,  2273,  2279,  2285,
    2291,  2297,  2303,  2309,  2315,  2321,  2327,  2333,  2339,  2345,
    2351,  2357,  2363,  2369,  2375,  2381,  2387,  2393,  2399,  2405,
    2411,  2417,  2421,  2425,  2429,  2434,  2440,  2445,  2450,  2455,
    2460,  2465,  2470,  2476,  2481,  2486,  2491,  2496,  2501,  2507,
    2513,  2519,  2525,  2531,  2537,  2543,  2549,  2555,  2561,  2567,
    2573,  2579,  2585,  2590,  2595,  2600,  2605,  2610,  2615,  2621,
    2626,  2631,  2636,  2641,  2646,  2651,  2656,  2662,  2667,  2672,
    2677,  2682,  2687,  2692,  2697,  2702,  2707,  2712,  2717,  2722,
    2727,  2732,  2738,  2743,  2748,  2754,  2760,  2765,  2770,  2775,
    2781,  2786,  2791,  2796,  2802,  2807,  2812,  2817,  2823,  2828,
    2833,  2838,  2844,  2850,  2856,  2862,  2867,  2873,  2879,  2885,
    2890,  2895,  2900,  2905,  2910,  2916,  2921,  2926,  2931,  2937,
    2942,  2947,  2952,  2958,  2963,  2968,  2973,  2979,  2984,  2989,
    2994,  3000,  3005,  3010,  3015,  3021,  3026,  3031,  3036,  3042,
    3047,  3052,  3057,  3063,  3068,  3073,  3078,  3084,  3089,  3094,
    3099,  3105,  3110,  3115,  3120,  3126,  3131,  3136,  3141,  3147,
    3152,  3157,  3162,  3168,  3173,  3178,  3183,  3189,  3194,  3199,
    3204,  3210,  3215,  3220,  3225,  3230,  3235,  3240,  3245,  3250,
    3255,  3260,  3265,  3270,  3275,  3280,  3285,  3290,  3295,  3300,
    3305,  3310,  3315,  3320,  3325,  3330,  3336,  3342,  3348,  3354,
    3361,  3368,  3374,  3380,  3386,  3392,  3398,  3404,  3411,  3416,
    3432,  3437,  3442,  3450,  3450,  3461,  3461,  3471,  3474,  3487,
    3509,  3536,  3540,  3546,  3551,  3562,  3566,  3572,  3583,  3586,
    3593,  3597,  3598,  3604,  3605,  3606,  3607,  3608,  3609,  3610,
    3612,  3618,  3627,  3628,  3632,  3628,  3644,  3645,  3649,  3649,
    3656,  3656,  3670,  3673,  3681,  3689,  3700,  3701,  3705,  3709,
    3716,  3723,  3727,  3735,  3739,  3752,  3756,  3763,  3763,  3783,
    3786,  3792,  3804,  3816,  3820,  3827,  3827,  3842,  3842,  3858,
    3858,  3879,  3882,  3888,  3891,  3897,  3901,  3908,  3913,  3918,
    3925,  3928,  3932,  3937,  3941,  3951,  3955,  3964,  3967,  3971,
    3980,  3980,  4022,  4028,  4031,  4036,  4039
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if 1
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name (yysymbol_kind_t yysymbol) YY_ATTRIBUTE_UNUSED;

/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "\"end of file\"", "error", "\"invalid token\"", "CONST", "BOOL", "INT",
  "UINT", "FLOAT", "BVEC2", "BVEC3", "BVEC4", "IVEC2", "IVEC3", "IVEC4",
  "UVEC2", "UVEC3", "UVEC4", "VEC2", "VEC3", "VEC4", "MAT2", "MAT3",
  "MAT4", "MAT2X2", "MAT2X3", "MAT2X4", "MAT3X2", "MAT3X3", "MAT3X4",
  "MAT4X2", "MAT4X3", "MAT4X4", "SAMPLER2D", "SAMPLER3D", "SAMPLERCUBE",
  "SAMPLER2DSHADOW", "SAMPLERCUBESHADOW", "SAMPLER2DARRAY",
  "SAMPLER2DARRAYSHADOW", "ISAMPLER2D", "ISAMPLER3D", "ISAMPLERCUBE",
  "ISAMPLER2DARRAY", "USAMPLER2D", "USAMPLER3D", "USAMPLERCUBE",
  "USAMPLER2DARRAY", "SAMPLER", "SAMPLERSHADOW", "TEXTURE2D", "TEXTURE3D",
  "TEXTURECUBE", "TEXTURE2DARRAY", "ITEXTURE2D", "ITEXTURE3D",
  "ITEXTURECUBE", "ITEXTURE2DARRAY", "UTEXTURE2D", "UTEXTURE3D",
  "UTEXTURECUBE", "UTEXTURE2DARRAY", "ATTRIBUTE", "VARYING", "FLOAT16_T",
  "FLOAT32_T", "DOUBLE", "FLOAT64_T", "INT64_T", "UINT64_T", "INT32_T",
  "UINT32_T", "INT16_T", "UINT16_T", "INT8_T", "UINT8_T", "I64VEC2",
  "I64VEC3", "I64VEC4", "U64VEC2", "U64VEC3", "U64VEC4", "I32VEC2",
  "I32VEC3", "I32VEC4", "U32VEC2", "U32VEC3", "U32VEC4", "I16VEC2",
  "I16VEC3", "I16VEC4", "U16VEC2", "U16VEC3", "U16VEC4", "I8VEC2",
  "I8VEC3", "I8VEC4", "U8VEC2", "U8VEC3", "U8VEC4", "DVEC2", "DVEC3",
  "DVEC4", "DMAT2", "DMAT3", "DMAT4", "F16VEC2", "F16VEC3", "F16VEC4",
  "F16MAT2", "F16MAT3", "F16MAT4", "F32VEC2", "F32VEC3", "F32VEC4",
  "F32MAT2", "F32MAT3", "F32MAT4", "F64VEC2", "F64VEC3", "F64VEC4",
  "F64MAT2", "F64MAT3", "F64MAT4", "DMAT2X2", "DMAT2X3", "DMAT2X4",
  "DMAT3X2", "DMAT3X3", "DMAT3X4", "DMAT4X2", "DMAT4X3", "DMAT4X4",
  "F16MAT2X2", "F16MAT2X3", "F16MAT2X4", "F16MAT3X2", "F16MAT3X3",
  "F16MAT3X4", "F16MAT4X2", "F16MAT4X3", "F16MAT4X4", "F32MAT2X2",
  "F32MAT2X3", "F32MAT2X4", "F32MAT3X2", "F32MAT3X3", "F32MAT3X4",
  "F32MAT4X2", "F32MAT4X3", "F32MAT4X4", "F64MAT2X2", "F64MAT2X3",
  "F64MAT2X4", "F64MAT3X2", "F64MAT3X3", "F64MAT3X4", "F64MAT4X2",
  "F64MAT4X3", "F64MAT4X4", "ATOMIC_UINT", "ACCSTRUCTNV", "ACCSTRUCTEXT",
  "RAYQUERYEXT", "FCOOPMATNV", "ICOOPMATNV", "UCOOPMATNV",
  "SAMPLERCUBEARRAY", "SAMPLERCUBEARRAYSHADOW", "ISAMPLERCUBEARRAY",
  "USAMPLERCUBEARRAY", "SAMPLER1D", "SAMPLER1DARRAY",
  "SAMPLER1DARRAYSHADOW", "ISAMPLER1D", "SAMPLER1DSHADOW", "SAMPLER2DRECT",
  "SAMPLER2DRECTSHADOW", "ISAMPLER2DRECT", "USAMPLER2DRECT",
  "SAMPLERBUFFER", "ISAMPLERBUFFER", "USAMPLERBUFFER", "SAMPLER2DMS",
  "ISAMPLER2DMS", "USAMPLER2DMS", "SAMPLER2DMSARRAY", "ISAMPLER2DMSARRAY",
  "USAMPLER2DMSARRAY", "SAMPLEREXTERNALOES", "SAMPLEREXTERNAL2DY2YEXT",
  "ISAMPLER1DARRAY", "USAMPLER1D", "USAMPLER1DARRAY", "F16SAMPLER1D",
  "F16SAMPLER2D", "F16SAMPLER3D", "F16SAMPLER2DRECT", "F16SAMPLERCUBE",
  "F16SAMPLER1DARRAY", "F16SAMPLER2DARRAY", "F16SAMPLERCUBEARRAY",
  "F16SAMPLERBUFFER", "F16SAMPLER2DMS", "F16SAMPLER2DMSARRAY",
  "F16SAMPLER1DSHADOW", "F16SAMPLER2DSHADOW", "F16SAMPLER1DARRAYSHADOW",
  "F16SAMPLER2DARRAYSHADOW", "F16SAMPLER2DRECTSHADOW",
  "F16SAMPLERCUBESHADOW", "F16SAMPLERCUBEARRAYSHADOW", "IMAGE1D",
  "IIMAGE1D", "UIMAGE1D", "IMAGE2D", "IIMAGE2D", "UIMAGE2D", "IMAGE3D",
  "IIMAGE3D", "UIMAGE3D", "IMAGE2DRECT", "IIMAGE2DRECT", "UIMAGE2DRECT",
  "IMAGECUBE", "IIMAGECUBE", "UIMAGECUBE", "IMAGEBUFFER", "IIMAGEBUFFER",
  "UIMAGEBUFFER", "IMAGE1DARRAY", "IIMAGE1DARRAY", "UIMAGE1DARRAY",
  "IMAGE2DARRAY", "IIMAGE2DARRAY", "UIMAGE2DARRAY", "IMAGECUBEARRAY",
  "IIMAGECUBEARRAY", "UIMAGECUBEARRAY", "IMAGE2DMS", "IIMAGE2DMS",
  "UIMAGE2DMS", "IMAGE2DMSARRAY", "IIMAGE2DMSARRAY", "UIMAGE2DMSARRAY",
  "F16IMAGE1D", "F16IMAGE2D", "F16IMAGE3D", "F16IMAGE2DRECT",
  "F16IMAGECUBE", "F16IMAGE1DARRAY", "F16IMAGE2DARRAY",
  "F16IMAGECUBEARRAY", "F16IMAGEBUFFER", "F16IMAGE2DMS",
  "F16IMAGE2DMSARRAY", "I64IMAGE1D", "U64IMAGE1D", "I64IMAGE2D",
  "U64IMAGE2D", "I64IMAGE3D", "U64IMAGE3D", "I64IMAGE2DRECT",
  "U64IMAGE2DRECT", "I64IMAGECUBE", "U64IMAGECUBE", "I64IMAGEBUFFER",
  "U64IMAGEBUFFER", "I64IMAGE1DARRAY", "U64IMAGE1DARRAY",
  "I64IMAGE2DARRAY", "U64IMAGE2DARRAY", "I64IMAGECUBEARRAY",
  "U64IMAGECUBEARRAY", "I64IMAGE2DMS", "U64IMAGE2DMS", "I64IMAGE2DMSARRAY",
  "U64IMAGE2DMSARRAY", "TEXTURECUBEARRAY", "ITEXTURECUBEARRAY",
  "UTEXTURECUBEARRAY", "TEXTURE1D", "ITEXTURE1D", "UTEXTURE1D",
  "TEXTURE1DARRAY", "ITEXTURE1DARRAY", "UTEXTURE1DARRAY", "TEXTURE2DRECT",
  "ITEXTURE2DRECT", "UTEXTURE2DRECT", "TEXTUREBUFFER", "ITEXTUREBUFFER",
  "UTEXTUREBUFFER", "TEXTURE2DMS", "ITEXTURE2DMS", "UTEXTURE2DMS",
  "TEXTURE2DMSARRAY", "ITEXTURE2DMSARRAY", "UTEXTURE2DMSARRAY",
  "F16TEXTURE1D", "F16TEXTURE2D", "F16TEXTURE3D", "F16TEXTURE2DRECT",
  "F16TEXTURECUBE", "F16TEXTURE1DARRAY", "F16TEXTURE2DARRAY",
  "F16TEXTURECUBEARRAY", "F16TEXTUREBUFFER", "F16TEXTURE2DMS",
  "F16TEXTURE2DMSARRAY", "SUBPASSINPUT", "SUBPASSINPUTMS", "ISUBPASSINPUT",
  "ISUBPASSINPUTMS", "USUBPASSINPUT", "USUBPASSINPUTMS", "F16SUBPASSINPUT",
  "F16SUBPASSINPUTMS", "LEFT_OP", "RIGHT_OP", "INC_OP", "DEC_OP", "LE_OP",
  "GE_OP", "EQ_OP", "NE_OP", "AND_OP", "OR_OP", "XOR_OP", "MUL_ASSIGN",
  "DIV_ASSIGN", "ADD_ASSIGN", "MOD_ASSIGN", "LEFT_ASSIGN", "RIGHT_ASSIGN",
  "AND_ASSIGN", "XOR_ASSIGN", "OR_ASSIGN", "SUB_ASSIGN", "STRING_LITERAL",
  "LEFT_PAREN", "RIGHT_PAREN", "LEFT_BRACKET", "RIGHT_BRACKET",
  "LEFT_BRACE", "RIGHT_BRACE", "DOT", "COMMA", "COLON", "EQUAL",
  "SEMICOLON", "BANG", "DASH", "TILDE", "PLUS", "STAR", "SLASH", "PERCENT",
  "LEFT_ANGLE", "RIGHT_ANGLE", "VERTICAL_BAR", "CARET", "AMPERSAND",
  "QUESTION", "INVARIANT", "HIGH_PRECISION", "MEDIUM_PRECISION",
  "LOW_PRECISION", "PRECISION", "PACKED", "RESOURCE", "SUPERP",
  "FLOATCONSTANT", "INTCONSTANT", "UINTCONSTANT", "BOOLCONSTANT",
  "IDENTIFIER", "TYPE_NAME", "CENTROID", "IN", "OUT", "INOUT", "STRUCT",
  "VOID", "WHILE", "BREAK", "CONTINUE", "DO", "ELSE", "FOR", "IF",
  "DISCARD", "RETURN", "SWITCH", "CASE", "DEFAULT", "TERMINATE_INVOCATION",
  "TERMINATE_RAY", "IGNORE_INTERSECTION", "UNIFORM", "SHARED", "BUFFER",
  "FLAT", "SMOOTH", "LAYOUT", "DOUBLECONSTANT", "INT16CONSTANT",
  "UINT16CONSTANT", "FLOAT16CONSTANT", "INT32CONSTANT", "UINT32CONSTANT",
  "INT64CONSTANT", "UINT64CONSTANT", "SUBROUTINE", "DEMOTE", "PAYLOADNV",
  "PAYLOADINNV", "HITATTRNV", "CALLDATANV", "CALLDATAINNV", "PAYLOADEXT",
  "PAYLOADINEXT", "HITATTREXT", "CALLDATAEXT", "CALLDATAINEXT", "PATCH",
  "SAMPLE", "NONUNIFORM", "COHERENT", "VOLATILE", "RESTRICT", "READONLY",
  "WRITEONLY", "DEVICECOHERENT", "QUEUEFAMILYCOHERENT",
  "WORKGROUPCOHERENT", "SUBGROUPCOHERENT", "NONPRIVATE",
  "SHADERCALLCOHERENT", "NOPERSPECTIVE", "EXPLICITINTERPAMD",
  "PERVERTEXNV", "PERPRIMITIVENV", "PERVIEWNV", "PERTASKNV", "PRECISE",
  "$accept", "variable_identifier", "primary_expression",
  "postfix_expression", "integer_expression", "function_call",
  "function_call_or_method", "function_call_generic",
  "function_call_header_no_parameters",
  "function_call_header_with_parameters", "function_call_header",
  "function_identifier", "unary_expression", "unary_operator",
  "multiplicative_expression", "additive_expression", "shift_expression",
  "relational_expression", "equality_expression", "and_expression",
  "exclusive_or_expression", "inclusive_or_expression",
  "logical_and_expression", "logical_xor_expression",
  "logical_or_expression", "conditional_expression", "$@1",
  "assignment_expression", "assignment_operator", "expression",
  "constant_expression", "declaration", "block_structure", "$@2",
  "identifier_list", "function_prototype", "function_declarator",
  "function_header_with_parameters", "function_header",
  "parameter_declarator", "parameter_declaration",
  "parameter_type_specifier", "init_declarator_list", "single_declaration",
  "fully_specified_type", "invariant_qualifier", "interpolation_qualifier",
  "layout_qualifier", "layout_qualifier_id_list", "layout_qualifier_id",
  "precise_qualifier", "type_qualifier", "single_type_qualifier",
  "storage_qualifier", "non_uniform_qualifier", "type_name_list",
  "type_specifier", "array_specifier", "type_parameter_specifier_opt",
  "type_parameter_specifier", "type_parameter_specifier_list",
  "type_specifier_nonarray", "precision_qualifier", "struct_specifier",
  "$@3", "$@4", "struct_declaration_list", "struct_declaration",
  "struct_declarator_list", "struct_declarator", "initializer",
  "initializer_list", "declaration_statement", "statement",
  "simple_statement", "demote_statement", "compound_statement", "$@5",
  "$@6", "statement_no_new_scope", "statement_scoped", "$@7", "$@8",
  "compound_statement_no_new_scope", "statement_list",
  "expression_statement", "selection_statement",
  "selection_statement_nonattributed", "selection_rest_statement",
  "condition", "switch_statement", "switch_statement_nonattributed", "$@9",
  "switch_statement_list", "case_label", "iteration_statement",
  "iteration_statement_nonattributed", "$@10", "$@11", "$@12",
  "for_init_statement", "conditionopt", "for_rest_statement",
  "jump_statement", "translation_unit", "external_declaration",
  "function_definition", "$@13", "attribute", "attribute_list",
  "single_attribute", YY_NULLPTR
};

static const char *
yysymbol_name (yysymbol_kind_t yysymbol)
{
  return yytname[yysymbol];
}
#endif

#ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_int16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,   350,   351,   352,   353,   354,
     355,   356,   357,   358,   359,   360,   361,   362,   363,   364,
     365,   366,   367,   368,   369,   370,   371,   372,   373,   374,
     375,   376,   377,   378,   379,   380,   381,   382,   383,   384,
     385,   386,   387,   388,   389,   390,   391,   392,   393,   394,
     395,   396,   397,   398,   399,   400,   401,   402,   403,   404,
     405,   406,   407,   408,   409,   410,   411,   412,   413,   414,
     415,   416,   417,   418,   419,   420,   421,   422,   423,   424,
     425,   426,   427,   428,   429,   430,   431,   432,   433,   434,
     435,   436,   437,   438,   439,   440,   441,   442,   443,   444,
     445,   446,   447,   448,   449,   450,   451,   452,   453,   454,
     455,   456,   457,   458,   459,   460,   461,   462,   463,   464,
     465,   466,   467,   468,   469,   470,   471,   472,   473,   474,
     475,   476,   477,   478,   479,   480,   481,   482,   483,   484,
     485,   486,   487,   488,   489,   490,   491,   492,   493,   494,
     495,   496,   497,   498,   499,   500,   501,   502,   503,   504,
     505,   506,   507,   508,   509,   510,   511,   512,   513,   514,
     515,   516,   517,   518,   519,   520,   521,   522,   523,   524,
     525,   526,   527,   528,   529,   530,   531,   532,   533,   534,
     535,   536,   537,   538,   539,   540,   541,   542,   543,   544,
     545,   546,   547,   548,   549,   550,   551,   552,   553,   554,
     555,   556,   557,   558,   559,   560,   561,   562,   563,   564,
     565,   566,   567,   568,   569,   570,   571,   572,   573,   574,
     575,   576,   577,   578,   579,   580,   581,   582,   583,   584,
     585,   586,   587,   588,   589,   590,   591,   592,   593,   594,
     595,   596,   597,   598,   599,   600,   601,   602,   603,   604,
     605,   606,   607,   608,   609,   610,   611,   612,   613,   614,
     615,   616,   617,   618,   619,   620,   621,   622,   623,   624,
     625,   626,   627,   628,   629,   630,   631,   632,   633,   634,
     635,   636,   637,   638,   639,   640,   641,   642,   643,   644,
     645,   646,   647,   648,   649,   650,   651,   652,   653,   654,
     655,   656,   657,   658,   659,   660,   661,   662,   663,   664,
     665,   666,   667,   668,   669,   670,   671,   672,   673,   674,
     675,   676,   677,   678,   679,   680,   681,   682,   683,   684,
     685,   686,   687,   688,   689,   690,   691,   692,   693,   694,
     695,   696,   697,   698,   699
};
#endif

#define YYPACT_NINF (-732)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-559)

#define yytable_value_is_error(Yyn) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
    4303,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
     109,  -732,  -732,  -732,  -732,  -732,     1,  -732,  -732,  -732,
    -732,  -732,  -732,  -324,  -261,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,    11,    44,    22,
       7,  6513,  -332,  -732,   -10,  -732,  -732,  -732,  -732,  4745,
    -732,  -732,  -732,  -732,    46,  -732,  -732,   767,  -732,  -732,
      16,  -732,    69,    -5,    47,  -732,  -338,  -732,    91,  -732,
    6513,  -732,  -732,  -732,  6513,    72,    80,  -732,    13,  -732,
      74,  -732,  -732,  9069,   126,  -732,  -732,  -732,   127,  6513,
    -732,   144,  -732,    17,  -732,  -732,    61,  7377,  -732,    10,
    1209,  -732,  -732,  -732,  -732,   126,    25,  -732,  7800,    26,
    -732,   119,  -732,    78,  9069,  9069,  -732,  9069,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,    36,  -732,  -732,
    -732,   164,    65,  9492,   171,  -732,  9069,  -732,  -732,  -340,
     173,  -732,  6513,   140,  5187,  -732,  6513,  9069,  -732,    -5,
    -732,   141,  -732,  -732,   124,   130,   179,    27,   117,   156,
     158,   160,   195,   194,    20,   181,  8223,  -732,   183,   182,
    -732,  -732,   186,   178,   180,  -732,   189,   192,   184,  8646,
     193,  9069,   187,   188,   190,   196,   197,   129,  -732,  -732,
      89,  -732,    44,   199,   204,  -732,  -732,  -732,  -732,  -732,
    1651,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -376,   173,  7800,    54,  7800,  -732,  -732,  7800,  6513,  -732,
     161,  -732,  -732,  -732,    70,  -732,  -732,  9069,   169,  -732,
    -732,  9069,   207,  -732,  -732,  -732,  9069,  -732,   140,   126,
     103,  -732,  -732,  -732,  5629,  -732,  -732,  -732,  -732,  9069,
    9069,  9069,  9069,  9069,  9069,  9069,  9069,  9069,  9069,  9069,
    9069,  9069,  9069,  9069,  9069,  9069,  9069,  9069,  -732,  -732,
    -732,   209,   177,  -732,  2093,  -732,  -732,  -732,  2093,  -732,
    9069,  -732,  -732,   122,  9069,   152,  -732,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  9069,  9069,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  7800,  -732,   143,  -732,  6071,  -732,  -732,   211,   208,
    -732,  -732,  -732,   123,   173,   140,  -732,  -732,  -732,  -732,
    -732,   124,   124,   130,   130,   179,   179,   179,   179,    27,
      27,   117,   156,   158,   160,   195,   194,  9069,  -732,   216,
      87,  -732,  2093,  3861,   174,  3419,    75,  -732,    85,  -732,
    -732,  -732,  -732,  -732,  6954,  -732,  -732,  -732,  -732,   154,
    9069,   217,   177,   191,   208,   185,  6513,   221,   223,  -732,
    -732,  3861,   220,  -732,  -732,  -732,  9069,   224,  -732,  -732,
    -732,   218,  2535,  9069,  -732,   219,   225,   198,   226,  2977,
    -732,   227,  -732,  -732,  7800,  -732,  -732,  -732,    86,  9069,
    2535,   220,  -732,  -732,  2093,  -732,   222,   208,  -732,  -732,
    2093,   228,  -732,  -732
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_int16 yydefact[] =
{
       0,   157,   210,   208,   209,   207,   214,   215,   216,   217,
     218,   219,   220,   221,   222,   211,   212,   213,   223,   224,
     225,   226,   227,   228,   229,   230,   231,   232,   233,   234,
     336,   337,   338,   339,   340,   341,   342,   362,   363,   364,
     365,   366,   367,   368,   377,   390,   391,   378,   379,   381,
     380,   382,   383,   384,   385,   386,   387,   388,   389,   165,
     166,   236,   237,   235,   238,   245,   246,   243,   244,   241,
     242,   239,   240,   268,   269,   270,   280,   281,   282,   265,
     266,   267,   277,   278,   279,   262,   263,   264,   274,   275,
     276,   259,   260,   261,   271,   272,   273,   247,   248,   249,
     283,   284,   285,   250,   251,   252,   295,   296,   297,   253,
     254,   255,   307,   308,   309,   256,   257,   258,   319,   320,
     321,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     298,   299,   300,   301,   302,   303,   304,   305,   306,   310,
     311,   312,   313,   314,   315,   316,   317,   318,   322,   323,
     324,   325,   326,   327,   328,   329,   330,   334,   331,   332,
     333,   515,   516,   517,   346,   347,   370,   373,   335,   344,
     345,   361,   343,   392,   393,   396,   397,   398,   400,   401,
     402,   404,   405,   406,   408,   409,   505,   506,   369,   371,
     372,   348,   349,   350,   394,   351,   355,   356,   359,   399,
     403,   407,   352,   353,   357,   358,   395,   354,   360,   439,
     441,   442,   443,   445,   446,   447,   449,   450,   451,   453,
     454,   455,   457,   458,   459,   461,   462,   463,   465,   466,
     467,   469,   470,   471,   473,   474,   475,   477,   478,   479,
     481,   482,   440,   444,   448,   452,   456,   464,   468,   472,
     460,   476,   480,   483,   484,   485,   486,   487,   488,   489,
     490,   491,   492,   493,   494,   495,   496,   497,   498,   499,
     500,   501,   502,   503,   504,   374,   375,   376,   410,   419,
     421,   415,   420,   422,   423,   425,   426,   427,   429,   430,
     431,   433,   434,   435,   437,   438,   411,   412,   413,   424,
     414,   416,   417,   418,   428,   432,   436,   507,   508,   511,
     512,   513,   514,   509,   510,   609,   132,   520,   521,   522,
       0,   519,   161,   159,   160,   158,     0,   206,   162,   163,
     164,   134,   133,     0,   190,   171,   173,   169,   175,   177,
     172,   174,   170,   176,   178,   167,   168,   192,   179,   186,
     187,   188,   189,   180,   181,   182,   183,   184,   185,   135,
     136,   137,   138,   139,   140,   147,   608,     0,   610,     0,
     109,   108,     0,   120,   125,   154,   153,   151,   155,     0,
     148,   150,   156,   130,   202,   152,   518,     0,   605,   607,
       0,   525,     0,     0,     0,    97,     0,    94,     0,   107,
       0,   116,   110,   118,     0,   119,     0,    95,   126,   100,
       0,   149,   131,     0,   195,   201,     1,   606,     0,     0,
     523,   144,   146,     0,   142,   193,     0,     0,    98,     0,
       0,   611,   111,   115,   117,   113,   121,   112,     0,   127,
     103,     0,   101,     0,     0,     0,     9,     0,    43,    42,
      44,    41,     5,     6,     7,     8,     2,    16,    14,    15,
      17,    10,    11,    12,    13,     3,    18,    37,    20,    25,
      26,     0,     0,    30,     0,   204,     0,    36,    34,     0,
     196,    96,     0,     0,     0,   527,     0,     0,   141,     0,
     191,     0,   197,    45,    49,    52,    55,    60,    63,    65,
      67,    69,    71,    73,    75,     0,     0,    99,     0,   553,
     562,   566,     0,     0,     0,   587,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    45,    78,    91,
       0,   540,     0,   156,   130,   543,   564,   542,   550,   541,
       0,   544,   545,   568,   546,   575,   547,   548,   583,   549,
       0,   114,     0,   122,     0,   535,   129,     0,     0,   105,
       0,   102,    38,    39,     0,    22,    23,     0,     0,    28,
      27,     0,   206,    31,    33,    40,     0,   203,     0,   533,
       0,   531,   526,   528,     0,    93,   145,   143,   194,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    76,   198,
     199,     0,     0,   552,     0,   585,   598,   597,     0,   589,
       0,   601,   599,     0,     0,     0,   582,   602,   603,   604,
     551,    81,    82,    84,    83,    86,    87,    88,    89,    90,
      85,    80,     0,     0,   567,   563,   565,   569,   576,   584,
     124,     0,   538,     0,   128,     0,   106,     4,     0,    24,
      21,    32,   205,     0,   534,     0,   529,   524,    46,    47,
      48,    51,    50,    53,    54,    58,    59,    56,    57,    61,
      62,    64,    66,    68,    70,    72,    74,     0,   200,   615,
       0,   613,   554,     0,     0,     0,     0,   600,     0,   581,
      79,    92,   123,   536,     0,   104,    19,   530,   532,     0,
       0,     0,     0,     0,   573,     0,     0,     0,     0,   592,
     591,   594,   560,   577,   537,   539,     0,     0,   612,   614,
     555,     0,     0,     0,   593,     0,     0,   572,     0,     0,
     570,     0,    77,   616,     0,   557,   586,   556,     0,   595,
       0,   560,   559,   561,   579,   574,     0,   596,   590,   571,
     580,     0,   588,   578
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,  -732,
    -732,  -732,  9402,  -732,   -90,   -89,  -153,   -92,   -29,   -28,
     -27,   -26,   -30,   -25,  -732,   -88,  -732,  -101,  -732,  -113,
    -132,     2,  -732,  -732,  -732,     4,  -732,  -732,  -732,   200,
     201,   202,  -732,  -732,  -343,  -732,  -732,  -732,  -732,    92,
    -732,   -36,   -46,  -732,     9,  -732,     0,   -67,  -732,  -732,
    -732,  -732,   263,  -732,  -732,  -732,  -481,  -142,     8,   -78,
    -214,  -732,  -107,  -204,  -731,  -732,  -149,  -732,  -732,  -160,
    -159,  -732,  -732,   212,  -269,  -104,  -732,    45,  -732,  -127,
    -732,    48,  -732,  -732,  -732,  -732,    49,  -732,  -732,  -732,
    -732,  -732,  -732,  -732,  -732,   210,  -732,  -732,  -732,  -732,
    -116
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,   465,   466,   467,   658,   468,   469,   470,   471,   472,
     473,   474,   527,   476,   494,   495,   496,   497,   498,   499,
     500,   501,   502,   503,   504,   528,   687,   529,   642,   530,
     586,   531,   367,   558,   443,   532,   369,   370,   371,   401,
     402,   403,   372,   373,   374,   375,   376,   377,   423,   424,
     378,   379,   380,   381,   477,   426,   478,   429,   414,   415,
     479,   384,   385,   386,   486,   419,   484,   485,   580,   581,
     556,   653,   535,   536,   537,   538,   539,   614,   713,   746,
     737,   738,   739,   747,   540,   541,   542,   543,   740,   717,
     544,   545,   741,   761,   546,   547,   548,   693,   618,   695,
     721,   735,   736,   549,   387,   388,   389,   398,   550,   690,
     691
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
     383,   745,   366,   427,   368,   584,   576,   512,   753,   382,
     515,   428,   516,   517,   406,   393,   520,   407,   577,   745,
       2,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,   655,   394,    61,
      62,    63,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   106,   107,   108,   109,   110,   111,
     112,   113,   114,   115,   116,   117,   118,   119,   120,   121,
     122,   123,   124,   125,   126,   127,   128,   129,   130,   131,
     132,   133,   134,   135,   136,   137,   138,   139,   140,   141,
     142,   143,   144,   145,   146,   147,   148,   149,   150,   151,
     152,   153,   154,   155,   156,   157,   158,   159,   160,   161,
     162,   163,   164,   165,   166,   167,   168,   169,   170,   171,
     172,   173,   174,   175,   176,   177,   178,   179,   180,   181,
     182,   183,   184,   185,   186,   187,   188,   189,   190,   191,
     192,   193,   194,   195,   196,   197,   198,   199,   200,   201,
     202,   203,   204,   205,   206,   207,   208,   209,   210,   211,
     212,   213,   214,   215,   216,   217,   218,   219,   220,   221,
     222,   223,   224,   225,   226,   227,   228,   229,   230,   231,
     232,   233,   234,   235,   236,   237,   238,   239,   240,   241,
     242,   243,   244,   245,   246,   247,   248,   249,   250,   251,
     252,   253,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,   280,   281,
     282,   283,   284,   285,   286,   287,   288,   289,   290,   291,
     292,   293,   294,   295,   296,   297,   298,   299,   300,   301,
     302,   303,   304,   305,   306,   307,   308,   309,   310,   311,
     312,   313,   314,   411,   564,   404,   646,   555,   650,   505,
     652,   439,   583,   654,   391,   692,   607,   480,   596,   597,
     715,   506,   437,   400,   427,   565,   566,   488,   411,   507,
     395,   438,   399,   489,   404,   408,   427,   506,   551,   553,
     421,   405,   573,   552,   557,   -35,   392,   567,   715,   412,
     382,   568,   608,   482,   598,   599,   396,   383,   382,   366,
     418,   368,   321,   397,   422,   506,   382,   326,   327,   585,
     405,   490,   651,   413,   405,   570,   623,   491,   625,   382,
     657,   571,   420,   382,   694,   722,   643,   440,   611,   483,
     441,   643,   425,   442,   560,   723,   756,   561,   382,   711,
     534,   643,   643,   712,   430,   643,   411,   702,   644,   533,
     600,   601,   583,   675,   676,   677,   678,   435,   482,   665,
     482,   555,   666,   555,   659,   436,   555,   631,   632,   633,
     634,   635,   636,   637,   638,   639,   640,   427,   643,   665,
     661,   697,   707,   317,   318,   319,   481,   641,   589,   590,
     591,   592,   578,   593,   483,   760,   483,   703,   646,   704,
     725,   382,   487,   382,   559,   382,   594,   595,   643,   699,
     643,   726,   671,   672,   569,   673,   674,   696,   679,   680,
     574,   698,   664,   583,   506,   579,   588,   602,   603,   604,
     605,   606,   482,   609,   612,   615,   613,   616,   619,   617,
     755,   620,   624,   621,   626,   730,   656,   627,   -36,   628,
     534,   700,   701,   -34,   660,   629,   630,   -29,   482,   533,
     555,   688,   689,   706,   643,   710,   646,   718,   483,   728,
     731,   732,   733,  -558,   743,   750,   744,   382,   749,   509,
     754,   762,   763,   681,   709,   682,   685,   683,   727,   684,
     714,   587,   686,   390,   483,   751,   663,   708,   719,   752,
     758,   720,   759,   382,   734,   647,   729,   417,   648,   649,
       0,   432,     0,   555,   433,     0,   434,     0,   714,     0,
     431,     0,     0,     0,   534,     0,     0,     0,   534,   482,
     748,     0,   585,   533,     0,   742,     0,   533,     0,     0,
       0,     0,     0,     0,     0,     0,   757,     0,     0,     0,
       0,     0,     0,   555,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   483,     0,   716,     0,     0,
       0,     0,     0,     0,   382,     0,     0,     0,     0,     0,
     411,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   716,     0,     0,     0,     0,
       0,     0,   534,   534,     0,   534,     0,     0,     0,     0,
       0,   533,   533,     0,   533,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   412,     0,     0,     0,
       0,   534,     0,     0,     0,   382,     0,     0,     0,     0,
     533,     0,   534,     0,     0,     0,     0,     0,     0,   534,
       0,   533,     0,     0,     0,     0,     0,     0,   533,     0,
     534,     0,     0,     0,   534,     0,     0,     0,     0,   533,
     534,     0,     0,   533,     0,     0,     0,   416,     0,   533,
       1,     2,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,    57,    58,    59,    60,
      61,    62,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,   106,   107,   108,   109,   110,
     111,   112,   113,   114,   115,   116,   117,   118,   119,   120,
     121,   122,   123,   124,   125,   126,   127,   128,   129,   130,
     131,   132,   133,   134,   135,   136,   137,   138,   139,   140,
     141,   142,   143,   144,   145,   146,   147,   148,   149,   150,
     151,   152,   153,   154,   155,   156,   157,   158,   159,   160,
     161,   162,   163,   164,   165,   166,   167,   168,   169,   170,
     171,   172,   173,   174,   175,   176,   177,   178,   179,   180,
     181,   182,   183,   184,   185,   186,   187,   188,   189,   190,
     191,   192,   193,   194,   195,   196,   197,   198,   199,   200,
     201,   202,   203,   204,   205,   206,   207,   208,   209,   210,
     211,   212,   213,   214,   215,   216,   217,   218,   219,   220,
     221,   222,   223,   224,   225,   226,   227,   228,   229,   230,
     231,   232,   233,   234,   235,   236,   237,   238,   239,   240,
     241,   242,   243,   244,   245,   246,   247,   248,   249,   250,
     251,   252,   253,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   284,   285,   286,   287,   288,   289,   290,
     291,   292,   293,   294,   295,   296,   297,   298,   299,   300,
     301,   302,   303,   304,   305,   306,   307,   308,   309,   310,
     311,   312,   313,   314,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   315,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     316,   317,   318,   319,   320,     0,     0,     0,     0,     0,
       0,     0,     0,   321,   322,   323,   324,   325,   326,   327,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   328,   329,   330,   331,   332,
     333,     0,     0,     0,     0,     0,     0,     0,     0,   334,
       0,   335,   336,   337,   338,   339,   340,   341,   342,   343,
     344,   345,   346,   347,   348,   349,   350,   351,   352,   353,
     354,   355,   356,   357,   358,   359,   360,   361,   362,   363,
     364,   365,     1,     2,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    57,    58,
      59,    60,    61,    62,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,   107,   108,
     109,   110,   111,   112,   113,   114,   115,   116,   117,   118,
     119,   120,   121,   122,   123,   124,   125,   126,   127,   128,
     129,   130,   131,   132,   133,   134,   135,   136,   137,   138,
     139,   140,   141,   142,   143,   144,   145,   146,   147,   148,
     149,   150,   151,   152,   153,   154,   155,   156,   157,   158,
     159,   160,   161,   162,   163,   164,   165,   166,   167,   168,
     169,   170,   171,   172,   173,   174,   175,   176,   177,   178,
     179,   180,   181,   182,   183,   184,   185,   186,   187,   188,
     189,   190,   191,   192,   193,   194,   195,   196,   197,   198,
     199,   200,   201,   202,   203,   204,   205,   206,   207,   208,
     209,   210,   211,   212,   213,   214,   215,   216,   217,   218,
     219,   220,   221,   222,   223,   224,   225,   226,   227,   228,
     229,   230,   231,   232,   233,   234,   235,   236,   237,   238,
     239,   240,   241,   242,   243,   244,   245,   246,   247,   248,
     249,   250,   251,   252,   253,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   283,   284,   285,   286,   287,   288,
     289,   290,   291,   292,   293,   294,   295,   296,   297,   298,
     299,   300,   301,   302,   303,   304,   305,   306,   307,   308,
     309,   310,   311,   312,   313,   314,     0,     0,   444,   445,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   446,   447,     0,
     508,     0,   509,   510,     0,     0,     0,     0,   511,   448,
     449,   450,   451,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   316,   317,   318,   319,   320,     0,     0,     0,
     452,   453,   454,   455,   456,   321,   322,   323,   324,   325,
     326,   327,   512,   513,   514,   515,     0,   516,   517,   518,
     519,   520,   521,   522,   523,   524,   525,   328,   329,   330,
     331,   332,   333,   457,   458,   459,   460,   461,   462,   463,
     464,   334,   526,   335,   336,   337,   338,   339,   340,   341,
     342,   343,   344,   345,   346,   347,   348,   349,   350,   351,
     352,   353,   354,   355,   356,   357,   358,   359,   360,   361,
     362,   363,   364,   365,     1,     2,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      37,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
      57,    58,    59,    60,    61,    62,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
     107,   108,   109,   110,   111,   112,   113,   114,   115,   116,
     117,   118,   119,   120,   121,   122,   123,   124,   125,   126,
     127,   128,   129,   130,   131,   132,   133,   134,   135,   136,
     137,   138,   139,   140,   141,   142,   143,   144,   145,   146,
     147,   148,   149,   150,   151,   152,   153,   154,   155,   156,
     157,   158,   159,   160,   161,   162,   163,   164,   165,   166,
     167,   168,   169,   170,   171,   172,   173,   174,   175,   176,
     177,   178,   179,   180,   181,   182,   183,   184,   185,   186,
     187,   188,   189,   190,   191,   192,   193,   194,   195,   196,
     197,   198,   199,   200,   201,   202,   203,   204,   205,   206,
     207,   208,   209,   210,   211,   212,   213,   214,   215,   216,
     217,   218,   219,   220,   221,   222,   223,   224,   225,   226,
     227,   228,   229,   230,   231,   232,   233,   234,   235,   236,
     237,   238,   239,   240,   241,   242,   243,   244,   245,   246,
     247,   248,   249,   250,   251,   252,   253,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,   283,   284,   285,   286,
     287,   288,   289,   290,   291,   292,   293,   294,   295,   296,
     297,   298,   299,   300,   301,   302,   303,   304,   305,   306,
     307,   308,   309,   310,   311,   312,   313,   314,     0,     0,
     444,   445,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   446,
     447,     0,   508,     0,   509,   645,     0,     0,     0,     0,
     511,   448,   449,   450,   451,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   316,   317,   318,   319,   320,     0,
       0,     0,   452,   453,   454,   455,   456,   321,   322,   323,
     324,   325,   326,   327,   512,   513,   514,   515,     0,   516,
     517,   518,   519,   520,   521,   522,   523,   524,   525,   328,
     329,   330,   331,   332,   333,   457,   458,   459,   460,   461,
     462,   463,   464,   334,   526,   335,   336,   337,   338,   339,
     340,   341,   342,   343,   344,   345,   346,   347,   348,   349,
     350,   351,   352,   353,   354,   355,   356,   357,   358,   359,
     360,   361,   362,   363,   364,   365,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   139,   140,   141,   142,   143,   144,
     145,   146,   147,   148,   149,   150,   151,   152,   153,   154,
     155,   156,   157,   158,   159,   160,   161,   162,   163,   164,
     165,   166,   167,   168,   169,   170,   171,   172,   173,   174,
     175,   176,   177,   178,   179,   180,   181,   182,   183,   184,
     185,   186,   187,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   197,   198,   199,   200,   201,   202,   203,   204,
     205,   206,   207,   208,   209,   210,   211,   212,   213,   214,
     215,   216,   217,   218,   219,   220,   221,   222,   223,   224,
     225,   226,   227,   228,   229,   230,   231,   232,   233,   234,
     235,   236,   237,   238,   239,   240,   241,   242,   243,   244,
     245,   246,   247,   248,   249,   250,   251,   252,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
       0,     0,   444,   445,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   446,   447,     0,   508,     0,   509,     0,     0,     0,
       0,     0,   511,   448,   449,   450,   451,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   316,   317,   318,   319,
     320,     0,     0,     0,   452,   453,   454,   455,   456,   321,
     322,   323,   324,   325,   326,   327,   512,   513,   514,   515,
       0,   516,   517,   518,   519,   520,   521,   522,   523,   524,
     525,   328,   329,   330,   331,   332,   333,   457,   458,   459,
     460,   461,   462,   463,   464,   334,   526,   335,   336,   337,
     338,   339,   340,   341,   342,   343,   344,   345,   346,   347,
     348,   349,   350,   351,   352,   353,   354,   355,   356,   357,
     358,   359,   360,   361,   362,   363,   364,   365,     1,     2,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    59,    60,    61,    62,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,   107,   108,   109,   110,   111,   112,
     113,   114,   115,   116,   117,   118,   119,   120,   121,   122,
     123,   124,   125,   126,   127,   128,   129,   130,   131,   132,
     133,   134,   135,   136,   137,   138,   139,   140,   141,   142,
     143,   144,   145,   146,   147,   148,   149,   150,   151,   152,
     153,   154,   155,   156,   157,   158,   159,   160,   161,   162,
     163,   164,   165,   166,   167,   168,   169,   170,   171,   172,
     173,   174,   175,   176,   177,   178,   179,   180,   181,   182,
     183,   184,   185,   186,   187,   188,   189,   190,   191,   192,
     193,   194,   195,   196,   197,   198,   199,   200,   201,   202,
     203,   204,   205,   206,   207,   208,   209,   210,   211,   212,
     213,   214,   215,   216,   217,   218,   219,   220,   221,   222,
     223,   224,   225,   226,   227,   228,   229,   230,   231,   232,
     233,   234,   235,   236,   237,   238,   239,   240,   241,   242,
     243,   244,   245,   246,   247,   248,   249,   250,   251,   252,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   284,   285,   286,   287,   288,   289,   290,   291,   292,
     293,   294,   295,   296,   297,   298,   299,   300,   301,   302,
     303,   304,   305,   306,   307,   308,   309,   310,   311,   312,
     313,   314,     0,     0,   444,   445,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   446,   447,     0,   508,     0,   430,     0,
       0,     0,     0,     0,   511,   448,   449,   450,   451,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   316,   317,
     318,   319,   320,     0,     0,     0,   452,   453,   454,   455,
     456,   321,   322,   323,   324,   325,   326,   327,   512,   513,
     514,   515,     0,   516,   517,   518,   519,   520,   521,   522,
     523,   524,   525,   328,   329,   330,   331,   332,   333,   457,
     458,   459,   460,   461,   462,   463,   464,   334,   526,   335,
     336,   337,   338,   339,   340,   341,   342,   343,   344,   345,
     346,   347,   348,   349,   350,   351,   352,   353,   354,   355,
     356,   357,   358,   359,   360,   361,   362,   363,   364,   365,
       1,     2,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,    57,    58,    59,    60,
      61,    62,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,   106,   107,   108,   109,   110,
     111,   112,   113,   114,   115,   116,   117,   118,   119,   120,
     121,   122,   123,   124,   125,   126,   127,   128,   129,   130,
     131,   132,   133,   134,   135,   136,   137,   138,   139,   140,
     141,   142,   143,   144,   145,   146,   147,   148,   149,   150,
     151,   152,   153,   154,   155,   156,   157,   158,   159,   160,
     161,   162,   163,   164,   165,   166,   167,   168,   169,   170,
     171,   172,   173,   174,   175,   176,   177,   178,   179,   180,
     181,   182,   183,   184,   185,   186,   187,   188,   189,   190,
     191,   192,   193,   194,   195,   196,   197,   198,   199,   200,
     201,   202,   203,   204,   205,   206,   207,   208,   209,   210,
     211,   212,   213,   214,   215,   216,   217,   218,   219,   220,
     221,   222,   223,   224,   225,   226,   227,   228,   229,   230,
     231,   232,   233,   234,   235,   236,   237,   238,   239,   240,
     241,   242,   243,   244,   245,   246,   247,   248,   249,   250,
     251,   252,   253,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   284,   285,   286,   287,   288,   289,   290,
     291,   292,   293,   294,   295,   296,   297,   298,   299,   300,
     301,   302,   303,   304,   305,   306,   307,   308,   309,   310,
     311,   312,   313,   314,     0,     0,   444,   445,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   446,   447,     0,   508,     0,
       0,     0,     0,     0,     0,     0,   511,   448,   449,   450,
     451,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     316,   317,   318,   319,   320,     0,     0,     0,   452,   453,
     454,   455,   456,   321,   322,   323,   324,   325,   326,   327,
     512,   513,   514,   515,     0,   516,   517,   518,   519,   520,
     521,   522,   523,   524,   525,   328,   329,   330,   331,   332,
     333,   457,   458,   459,   460,   461,   462,   463,   464,   334,
     526,   335,   336,   337,   338,   339,   340,   341,   342,   343,
     344,   345,   346,   347,   348,   349,   350,   351,   352,   353,
     354,   355,   356,   357,   358,   359,   360,   361,   362,   363,
     364,   365,     1,     2,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    57,    58,
      59,    60,    61,    62,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,   107,   108,
     109,   110,   111,   112,   113,   114,   115,   116,   117,   118,
     119,   120,   121,   122,   123,   124,   125,   126,   127,   128,
     129,   130,   131,   132,   133,   134,   135,   136,   137,   138,
     139,   140,   141,   142,   143,   144,   145,   146,   147,   148,
     149,   150,   151,   152,   153,   154,   155,   156,   157,   158,
     159,   160,   161,   162,   163,   164,   165,   166,   167,   168,
     169,   170,   171,   172,   173,   174,   175,   176,   177,   178,
     179,   180,   181,   182,   183,   184,   185,   186,   187,   188,
     189,   190,   191,   192,   193,   194,   195,   196,   197,   198,
     199,   200,   201,   202,   203,   204,   205,   206,   207,   208,
     209,   210,   211,   212,   213,   214,   215,   216,   217,   218,
     219,   220,   221,   222,   223,   224,   225,   226,   227,   228,
     229,   230,   231,   232,   233,   234,   235,   236,   237,   238,
     239,   240,   241,   242,   243,   244,   245,   246,   247,   248,
     249,   250,   251,   252,   253,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   283,   284,   285,   286,   287,   288,
     289,   290,   291,   292,   293,   294,   295,   296,   297,   298,
     299,   300,   301,   302,   303,   304,   305,   306,   307,   308,
     309,   310,   311,   312,   313,   314,     0,     0,   444,   445,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   446,   447,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   511,   448,
     449,   450,   451,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   316,   317,   318,   319,   320,     0,     0,     0,
     452,   453,   454,   455,   456,   321,   322,   323,   324,   325,
     326,   327,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   328,   329,   330,
     331,   332,   333,   457,   458,   459,   460,   461,   462,   463,
     464,   334,     0,   335,   336,   337,   338,   339,   340,   341,
     342,   343,   344,   345,   346,   347,   348,   349,   350,   351,
     352,   353,   354,   355,   356,   357,   358,   359,   360,   361,
     362,   363,   364,   365,     1,     2,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      37,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
      57,    58,    59,    60,    61,    62,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
     107,   108,   109,   110,   111,   112,   113,   114,   115,   116,
     117,   118,   119,   120,   121,   122,   123,   124,   125,   126,
     127,   128,   129,   130,   131,   132,   133,   134,   135,   136,
     137,   138,   139,   140,   141,   142,   143,   144,   145,   146,
     147,   148,   149,   150,   151,   152,   153,   154,   155,   156,
     157,   158,   159,   160,   161,   162,   163,   164,   165,   166,
     167,   168,   169,   170,   171,   172,   173,   174,   175,   176,
     177,   178,   179,   180,   181,   182,   183,   184,   185,   186,
     187,   188,   189,   190,   191,   192,   193,   194,   195,   196,
     197,   198,   199,   200,   201,   202,   203,   204,   205,   206,
     207,   208,   209,   210,   211,   212,   213,   214,   215,   216,
     217,   218,   219,   220,   221,   222,   223,   224,   225,   226,
     227,   228,   229,   230,   231,   232,   233,   234,   235,   236,
     237,   238,   239,   240,   241,   242,   243,   244,   245,   246,
     247,   248,   249,   250,   251,   252,   253,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,   283,   284,   285,   286,
     287,   288,   289,   290,   291,   292,   293,   294,   295,   296,
     297,   298,   299,   300,   301,   302,   303,   304,   305,   306,
     307,   308,   309,   310,   311,   312,   313,   314,     0,     0,
     444,   445,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   446,
     447,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   448,   449,   450,   451,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   316,   317,   318,   319,     0,     0,
       0,     0,   452,   453,   454,   455,   456,   321,   322,   323,
     324,   325,   326,   327,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   328,
     329,   330,   331,   332,   333,   457,   458,   459,   460,   461,
     462,   463,   464,   334,     0,   335,   336,   337,   338,   339,
     340,   341,   342,   343,   344,   345,   346,   347,   348,   349,
     350,   351,   352,   353,   354,   355,   356,   357,   358,   359,
     360,   361,   362,   363,   364,   365,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   139,   140,   141,   142,   143,   144,
     145,   146,   147,   148,   149,   150,   151,   152,   153,   154,
     155,   156,   157,   158,   159,   160,   161,   162,   163,   164,
     165,   166,   167,   168,   169,   170,   171,   172,   173,   174,
     175,   176,   177,   178,   179,   180,   181,   182,   183,   184,
     185,   186,   187,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   197,   198,   199,   200,   201,   202,   203,   204,
     205,   206,   207,   208,   209,   210,   211,   212,   213,   214,
     215,   216,   217,   218,   219,   220,   221,   222,   223,   224,
     225,   226,   227,   228,   229,   230,   231,   232,   233,   234,
     235,   236,   237,   238,   239,   240,   241,   242,   243,   244,
     245,   246,   247,   248,   249,   250,   251,   252,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   315,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   316,   317,   318,   319,
     320,     0,     0,     0,     0,     0,     0,     0,     0,   321,
     322,   323,   324,   325,   326,   327,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   328,   329,   330,   331,   332,   333,     0,     0,     0,
       0,     0,     0,     0,     0,   334,     0,   335,   336,   337,
     338,   339,   340,   341,   342,   343,   344,   345,   346,   347,
     348,   349,   350,   351,   352,   353,   354,   355,   356,   357,
     358,   359,   360,   361,   362,   363,   364,   365,     1,     2,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    59,    60,    61,    62,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,   107,   108,   109,   110,   111,   112,
     113,   114,   115,   116,   117,   118,   119,   120,   121,   122,
     123,   124,   125,   126,   127,   128,   129,   130,   131,   132,
     133,   134,   135,   136,   137,   138,   139,   140,   141,   142,
     143,   144,   145,   146,   147,   148,   149,   150,   151,   152,
     153,   154,   155,   156,   157,   158,   159,   160,   161,   162,
     163,   164,   165,   166,   167,   168,   169,   170,   171,   172,
     173,   174,   175,   176,   177,   178,   179,   180,   181,   182,
     183,   184,   185,   186,   187,   188,   189,   190,   191,   192,
     193,   194,   195,   196,   197,   198,   199,   200,   201,   202,
     203,   204,   205,   206,   207,   208,   209,   210,   211,   212,
     213,   214,   215,   216,   217,   218,   219,   220,   221,   222,
     223,   224,   225,   226,   227,   228,   229,   230,   231,   232,
     233,   234,   235,   236,   237,   238,   239,   240,   241,   242,
     243,   244,   245,   246,   247,   248,   249,   250,   251,   252,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   284,   285,   286,   287,   288,   289,   290,   291,   292,
     293,   294,   295,   296,   297,   298,   299,   300,   301,   302,
     303,   304,   305,   306,   307,   308,   309,   310,   311,   312,
     313,   314,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   409,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   316,   317,
     318,   319,     0,     0,     0,     0,     0,     0,     0,     0,
     410,   321,   322,   323,   324,   325,   326,   327,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   328,   329,   330,   331,   332,   333,     0,
       0,     0,     0,     0,     0,     0,     0,   334,     0,   335,
     336,   337,   338,   339,   340,   341,   342,   343,   344,   345,
     346,   347,   348,   349,   350,   351,   352,   353,   354,   355,
     356,   357,   358,   359,   360,   361,   362,   363,   364,   365,
       1,     2,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,    57,    58,    59,    60,
      61,    62,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,   106,   107,   108,   109,   110,
     111,   112,   113,   114,   115,   116,   117,   118,   119,   120,
     121,   122,   123,   124,   125,   126,   127,   128,   129,   130,
     131,   132,   133,   134,   135,   136,   137,   138,   139,   140,
     141,   142,   143,   144,   145,   146,   147,   148,   149,   150,
     151,   152,   153,   154,   155,   156,   157,   158,   159,   160,
     161,   162,   163,   164,   165,   166,   167,   168,   169,   170,
     171,   172,   173,   174,   175,   176,   177,   178,   179,   180,
     181,   182,   183,   184,   185,   186,   187,   188,   189,   190,
     191,   192,   193,   194,   195,   196,   197,   198,   199,   200,
     201,   202,   203,   204,   205,   206,   207,   208,   209,   210,
     211,   212,   213,   214,   215,   216,   217,   218,   219,   220,
     221,   222,   223,   224,   225,   226,   227,   228,   229,   230,
     231,   232,   233,   234,   235,   236,   237,   238,   239,   240,
     241,   242,   243,   244,   245,   246,   247,   248,   249,   250,
     251,   252,   253,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   284,   285,   286,   287,   288,   289,   290,
     291,   292,   293,   294,   295,   296,   297,   298,   299,   300,
     301,   302,   303,   304,   305,   306,   307,   308,   309,   310,
     311,   312,   313,   314,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   582,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     316,   317,   318,   319,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   321,   322,   323,   324,   325,   326,   327,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   328,   329,   330,   331,   332,
     333,     0,     0,     0,     0,     0,     0,     0,     0,   334,
       0,   335,   336,   337,   338,   339,   340,   341,   342,   343,
     344,   345,   346,   347,   348,   349,   350,   351,   352,   353,
     354,   355,   356,   357,   358,   359,   360,   361,   362,   363,
     364,   365,     1,     2,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    57,    58,
      59,    60,    61,    62,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,   107,   108,
     109,   110,   111,   112,   113,   114,   115,   116,   117,   118,
     119,   120,   121,   122,   123,   124,   125,   126,   127,   128,
     129,   130,   131,   132,   133,   134,   135,   136,   137,   138,
     139,   140,   141,   142,   143,   144,   145,   146,   147,   148,
     149,   150,   151,   152,   153,   154,   155,   156,   157,   158,
     159,   160,   161,   162,   163,   164,   165,   166,   167,   168,
     169,   170,   171,   172,   173,   174,   175,   176,   177,   178,
     179,   180,   181,   182,   183,   184,   185,   186,   187,   188,
     189,   190,   191,   192,   193,   194,   195,   196,   197,   198,
     199,   200,   201,   202,   203,   204,   205,   206,   207,   208,
     209,   210,   211,   212,   213,   214,   215,   216,   217,   218,
     219,   220,   221,   222,   223,   224,   225,   226,   227,   228,
     229,   230,   231,   232,   233,   234,   235,   236,   237,   238,
     239,   240,   241,   242,   243,   244,   245,   246,   247,   248,
     249,   250,   251,   252,   253,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   283,   284,   285,   286,   287,   288,
     289,   290,   291,   292,   293,   294,   295,   296,   297,   298,
     299,   300,   301,   302,   303,   304,   305,   306,   307,   308,
     309,   310,   311,   312,   313,   314,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   667,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   316,   317,   318,   319,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   321,   322,   323,   324,   325,
     326,   327,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   328,   329,   330,
     331,   332,   333,     0,     0,     0,     0,     0,     0,     0,
       0,   334,     0,   335,   336,   337,   338,   339,   340,   341,
     342,   343,   344,   345,   346,   347,   348,   349,   350,   351,
     352,   353,   354,   355,   356,   357,   358,   359,   360,   361,
     362,   363,   364,   365,     1,     2,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      37,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
      57,    58,    59,    60,    61,    62,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
     107,   108,   109,   110,   111,   112,   113,   114,   115,   116,
     117,   118,   119,   120,   121,   122,   123,   124,   125,   126,
     127,   128,   129,   130,   131,   132,   133,   134,   135,   136,
     137,   138,   139,   140,   141,   142,   143,   144,   145,   146,
     147,   148,   149,   150,   151,   152,   153,   154,   155,   156,
     157,   158,   159,   160,   161,   162,   163,   164,   165,   166,
     167,   168,   169,   170,   171,   172,   173,   174,   175,   176,
     177,   178,   179,   180,   181,   182,   183,   184,   185,   186,
     187,   188,   189,   190,   191,   192,   193,   194,   195,   196,
     197,   198,   199,   200,   201,   202,   203,   204,   205,   206,
     207,   208,   209,   210,   211,   212,   213,   214,   215,   216,
     217,   218,   219,   220,   221,   222,   223,   224,   225,   226,
     227,   228,   229,   230,   231,   232,   233,   234,   235,   236,
     237,   238,   239,   240,   241,   242,   243,   244,   245,   246,
     247,   248,   249,   250,   251,   252,   253,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,   283,   284,   285,   286,
     287,   288,   289,   290,   291,   292,   293,   294,   295,   296,
     297,   298,   299,   300,   301,   302,   303,   304,   305,   306,
     307,   308,   309,   310,   311,   312,   313,   314,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   705,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   316,   317,   318,   319,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   321,   322,   323,
     324,   325,   326,   327,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   328,
     329,   330,   331,   332,   333,     0,     0,     0,     0,     0,
       0,     0,     0,   334,     0,   335,   336,   337,   338,   339,
     340,   341,   342,   343,   344,   345,   346,   347,   348,   349,
     350,   351,   352,   353,   354,   355,   356,   357,   358,   359,
     360,   361,   362,   363,   364,   365,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   139,   140,   141,   142,   143,   144,
     145,   146,   147,   148,   149,   150,   151,   152,   153,   154,
     155,   156,   157,   158,   159,   160,   161,   162,   163,   164,
     165,   166,   167,   168,   169,   170,   171,   172,   173,   174,
     175,   176,   177,   178,   179,   180,   181,   182,   183,   184,
     185,   186,   187,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   197,   198,   199,   200,   201,   202,   203,   204,
     205,   206,   207,   208,   209,   210,   211,   212,   213,   214,
     215,   216,   217,   218,   219,   220,   221,   222,   223,   224,
     225,   226,   227,   228,   229,   230,   231,   232,   233,   234,
     235,   236,   237,   238,   239,   240,   241,   242,   243,   244,
     245,   246,   247,   248,   249,   250,   251,   252,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   316,   317,   318,   319,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   321,
     322,   323,   324,   325,   326,   327,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   328,   329,   330,   331,   332,   333,     0,     0,     0,
       0,     0,     0,     0,     0,   334,     0,   335,   336,   337,
     338,   339,   340,   341,   342,   343,   344,   345,   346,   347,
     348,   349,   350,   351,   352,   353,   354,   355,   356,   357,
     358,   359,   360,   361,   362,   363,   364,   365,     2,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,    57,    58,     0,     0,    61,    62,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,   107,   108,   109,   110,   111,   112,   113,
     114,   115,   116,   117,   118,   119,   120,   121,   122,   123,
     124,   125,   126,   127,   128,   129,   130,   131,   132,   133,
     134,   135,   136,   137,   138,   139,   140,   141,   142,   143,
     144,   145,   146,   147,   148,   149,   150,   151,   152,   153,
     154,   155,   156,   157,   158,   159,   160,   161,   162,   163,
     164,   165,   166,   167,   168,   169,   170,   171,   172,   173,
     174,   175,   176,   177,   178,   179,   180,   181,   182,   183,
     184,   185,   186,   187,   188,   189,   190,   191,   192,   193,
     194,   195,   196,   197,   198,   199,   200,   201,   202,   203,
     204,   205,   206,   207,   208,   209,   210,   211,   212,   213,
     214,   215,   216,   217,   218,   219,   220,   221,   222,   223,
     224,   225,   226,   227,   228,   229,   230,   231,   232,   233,
     234,   235,   236,   237,   238,   239,   240,   241,   242,   243,
     244,   245,   246,   247,   248,   249,   250,   251,   252,   253,
     254,   255,   256,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     284,   285,   286,   287,   288,   289,   290,   291,   292,   293,
     294,   295,   296,   297,   298,   299,   300,   301,   302,   303,
     304,   305,   306,   307,   308,   309,   310,   311,   312,   313,
     314,     0,     0,   444,   445,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   446,   447,     0,     0,     0,   554,   724,     0,
       0,     0,     0,     0,   448,   449,   450,   451,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   452,   453,   454,   455,   456,
     321,     0,     0,     0,     0,   326,   327,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   457,   458,
     459,   460,   461,   462,   463,   464,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     347,     2,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,    57,    58,     0,     0,
      61,    62,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,   106,   107,   108,   109,   110,
     111,   112,   113,   114,   115,   116,   117,   118,   119,   120,
     121,   122,   123,   124,   125,   126,   127,   128,   129,   130,
     131,   132,   133,   134,   135,   136,   137,   138,   139,   140,
     141,   142,   143,   144,   145,   146,   147,   148,   149,   150,
     151,   152,   153,   154,   155,   156,   157,   158,   159,   160,
     161,   162,   163,   164,   165,   166,   167,   168,   169,   170,
     171,   172,   173,   174,   175,   176,   177,   178,   179,   180,
     181,   182,   183,   184,   185,   186,   187,   188,   189,   190,
     191,   192,   193,   194,   195,   196,   197,   198,   199,   200,
     201,   202,   203,   204,   205,   206,   207,   208,   209,   210,
     211,   212,   213,   214,   215,   216,   217,   218,   219,   220,
     221,   222,   223,   224,   225,   226,   227,   228,   229,   230,
     231,   232,   233,   234,   235,   236,   237,   238,   239,   240,
     241,   242,   243,   244,   245,   246,   247,   248,   249,   250,
     251,   252,   253,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   284,   285,   286,   287,   288,   289,   290,
     291,   292,   293,   294,   295,   296,   297,   298,   299,   300,
     301,   302,   303,   304,   305,   306,   307,   308,   309,   310,
     311,   312,   313,   314,     0,     0,   444,   445,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   446,   447,     0,     0,   492,
       0,     0,     0,     0,     0,     0,     0,   448,   449,   450,
     451,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   452,   453,
     454,   455,   456,   321,     0,     0,     0,     0,   326,   327,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   457,   458,   459,   460,   461,   462,   463,   464,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   347,     2,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    33,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    57,
      58,     0,     0,    61,    62,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,   107,
     108,   109,   110,   111,   112,   113,   114,   115,   116,   117,
     118,   119,   120,   121,   122,   123,   124,   125,   126,   127,
     128,   129,   130,   131,   132,   133,   134,   135,   136,   137,
     138,   139,   140,   141,   142,   143,   144,   145,   146,   147,
     148,   149,   150,   151,   152,   153,   154,   155,   156,   157,
     158,   159,   160,   161,   162,   163,   164,   165,   166,   167,
     168,   169,   170,   171,   172,   173,   174,   175,   176,   177,
     178,   179,   180,   181,   182,   183,   184,   185,   186,   187,
     188,   189,   190,   191,   192,   193,   194,   195,   196,   197,
     198,   199,   200,   201,   202,   203,   204,   205,   206,   207,
     208,   209,   210,   211,   212,   213,   214,   215,   216,   217,
     218,   219,   220,   221,   222,   223,   224,   225,   226,   227,
     228,   229,   230,   231,   232,   233,   234,   235,   236,   237,
     238,   239,   240,   241,   242,   243,   244,   245,   246,   247,
     248,   249,   250,   251,   252,   253,   254,   255,   256,   257,
     258,   259,   260,   261,   262,   263,   264,   265,   266,   267,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,   279,   280,   281,   282,   283,   284,   285,   286,   287,
     288,   289,   290,   291,   292,   293,   294,   295,   296,   297,
     298,   299,   300,   301,   302,   303,   304,   305,   306,   307,
     308,   309,   310,   311,   312,   313,   314,     0,     0,   444,
     445,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   446,   447,
       0,     0,     0,   554,     0,     0,     0,     0,     0,     0,
     448,   449,   450,   451,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   452,   453,   454,   455,   456,   321,     0,     0,     0,
       0,   326,   327,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   457,   458,   459,   460,   461,   462,
     463,   464,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   347,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,     0,     0,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   139,   140,   141,   142,   143,   144,
     145,   146,   147,   148,   149,   150,   151,   152,   153,   154,
     155,   156,   157,   158,   159,   160,   161,   162,   163,   164,
     165,   166,   167,   168,   169,   170,   171,   172,   173,   174,
     175,   176,   177,   178,   179,   180,   181,   182,   183,   184,
     185,   186,   187,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   197,   198,   199,   200,   201,   202,   203,   204,
     205,   206,   207,   208,   209,   210,   211,   212,   213,   214,
     215,   216,   217,   218,   219,   220,   221,   222,   223,   224,
     225,   226,   227,   228,   229,   230,   231,   232,   233,   234,
     235,   236,   237,   238,   239,   240,   241,   242,   243,   244,
     245,   246,   247,   248,   249,   250,   251,   252,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
       0,     0,   444,   445,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   446,   447,     0,     0,   610,     0,     0,     0,     0,
       0,     0,     0,   448,   449,   450,   451,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   452,   453,   454,   455,   456,   321,
       0,     0,     0,     0,   326,   327,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   457,   458,   459,
     460,   461,   462,   463,   464,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   347,
       2,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,     0,     0,    61,
      62,    63,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   106,   107,   108,   109,   110,   111,
     112,   113,   114,   115,   116,   117,   118,   119,   120,   121,
     122,   123,   124,   125,   126,   127,   128,   129,   130,   131,
     132,   133,   134,   135,   136,   137,   138,   139,   140,   141,
     142,   143,   144,   145,   146,   147,   148,   149,   150,   151,
     152,   153,   154,   155,   156,   157,   158,   159,   160,   161,
     162,   163,   164,   165,   166,   167,   168,   169,   170,   171,
     172,   173,   174,   175,   176,   177,   178,   179,   180,   181,
     182,   183,   184,   185,   186,   187,   188,   189,   190,   191,
     192,   193,   194,   195,   196,   197,   198,   199,   200,   201,
     202,   203,   204,   205,   206,   207,   208,   209,   210,   211,
     212,   213,   214,   215,   216,   217,   218,   219,   220,   221,
     222,   223,   224,   225,   226,   227,   228,   229,   230,   231,
     232,   233,   234,   235,   236,   237,   238,   239,   240,   241,
     242,   243,   244,   245,   246,   247,   248,   249,   250,   251,
     252,   253,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,   280,   281,
     282,   283,   284,   285,   286,   287,   288,   289,   290,   291,
     292,   293,   294,   295,   296,   297,   298,   299,   300,   301,
     302,   303,   304,   305,   306,   307,   308,   309,   310,   311,
     312,   313,   314,     0,     0,   444,   445,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   446,   447,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   622,   448,   449,   450,   451,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   452,   453,   454,
     455,   456,   321,     0,     0,     0,     0,   326,   327,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     457,   458,   459,   460,   461,   462,   463,   464,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   347,     2,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    57,    58,
       0,     0,    61,    62,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,   107,   108,
     109,   110,   111,   112,   113,   114,   115,   116,   117,   118,
     119,   120,   121,   122,   123,   124,   125,   126,   127,   128,
     129,   130,   131,   132,   133,   134,   135,   136,   137,   138,
     139,   140,   141,   142,   143,   144,   145,   146,   147,   148,
     149,   150,   151,   152,   153,   154,   155,   156,   157,   158,
     159,   160,   161,   162,   163,   164,   165,   166,   167,   168,
     169,   170,   171,   172,   173,   174,   175,   176,   177,   178,
     179,   180,   181,   182,   183,   184,   185,   186,   187,   188,
     189,   190,   191,   192,   193,   194,   195,   196,   197,   198,
     199,   200,   201,   202,   203,   204,   205,   206,   207,   208,
     209,   210,   211,   212,   213,   214,   215,   216,   217,   218,
     219,   220,   221,   222,   223,   224,   225,   226,   227,   228,
     229,   230,   231,   232,   233,   234,   235,   236,   237,   238,
     239,   240,   241,   242,   243,   244,   245,   246,   247,   248,
     249,   250,   251,   252,   253,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   283,   284,   285,   286,   287,   288,
     289,   290,   291,   292,   293,   294,   295,   296,   297,   298,
     299,   300,   301,   302,   303,   304,   305,   306,   307,   308,
     309,   310,   311,   312,   313,   314,     0,     0,   444,   445,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   446,   447,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   448,
     449,   450,   451,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     452,   453,   454,   455,   456,   321,     0,     0,     0,     0,
     326,   327,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   457,   458,   459,   460,   461,   462,   463,
     464,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   347,     2,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
      36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    57,    58,     0,     0,    61,    62,    63,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
     106,   107,   108,   109,   110,   111,   112,   113,   114,   115,
     116,   117,   118,   119,   120,   121,   122,   123,   124,   125,
     126,   127,   128,   129,   130,   131,   132,   133,   134,   135,
     136,   137,   138,   139,   140,   141,   142,   143,   144,   145,
     146,   147,   148,   149,   150,   151,   152,   153,   154,   155,
     156,   157,   158,   159,   160,   161,   162,   163,   164,   165,
     166,   167,   168,   169,   170,   171,   172,   173,   174,   175,
     176,   177,   178,   179,   180,   181,   182,   183,   184,   185,
     186,   187,   188,   189,   190,   191,   192,   193,   194,   195,
     196,   197,   198,   199,   200,   201,   202,   203,   204,   205,
     206,   207,   208,   209,   210,   211,   212,   213,   214,   215,
     216,   217,   218,   219,   220,   221,   222,   223,   224,   225,
     226,   227,   228,   229,   230,   231,   232,   233,   234,   235,
     236,   237,   238,   239,   240,   241,   242,   243,   244,   245,
     246,   247,   248,   249,   250,   251,   252,   253,   254,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   284,   285,
     286,   287,   288,   289,   290,   291,   292,   293,   294,   295,
     296,   297,   298,   299,   300,   301,   302,   303,   304,   305,
     306,   307,   308,   309,   310,   311,   312,   313,   314,     0,
       0,   444,   445,     0,     0,   475,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   493,
     446,   447,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   448,   449,   450,   451,   562,   563,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   452,   453,   454,   455,   456,   321,     0,
       0,     0,     0,   326,   572,     0,     0,     0,   575,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   493,
       0,     0,     0,     0,     0,     0,   457,   458,   459,   460,
     461,   462,   463,   464,     0,     0,     0,     0,   493,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   347,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   662,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   668,   669,   670,   493,   493,   493,   493,   493,   493,
     493,   493,   493,   493,   493,   493,   493,   493,   493,   493,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   493
};

static const yytype_int16 yycheck[] =
{
       0,   732,     0,   341,     0,   486,   346,   383,   739,     0,
     386,   349,   388,   389,   346,   339,   392,   349,   358,   750,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,    57,    58,    59,    60,   558,   339,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,   107,   108,   109,   110,   111,   112,   113,
     114,   115,   116,   117,   118,   119,   120,   121,   122,   123,
     124,   125,   126,   127,   128,   129,   130,   131,   132,   133,
     134,   135,   136,   137,   138,   139,   140,   141,   142,   143,
     144,   145,   146,   147,   148,   149,   150,   151,   152,   153,
     154,   155,   156,   157,   158,   159,   160,   161,   162,   163,
     164,   165,   166,   167,   168,   169,   170,   171,   172,   173,
     174,   175,   176,   177,   178,   179,   180,   181,   182,   183,
     184,   185,   186,   187,   188,   189,   190,   191,   192,   193,
     194,   195,   196,   197,   198,   199,   200,   201,   202,   203,
     204,   205,   206,   207,   208,   209,   210,   211,   212,   213,
     214,   215,   216,   217,   218,   219,   220,   221,   222,   223,
     224,   225,   226,   227,   228,   229,   230,   231,   232,   233,
     234,   235,   236,   237,   238,   239,   240,   241,   242,   243,
     244,   245,   246,   247,   248,   249,   250,   251,   252,   253,
     254,   255,   256,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     284,   285,   286,   287,   288,   289,   290,   291,   292,   293,
     294,   295,   296,   297,   298,   299,   300,   301,   302,   303,
     304,   305,   306,   307,   308,   309,   310,   311,   312,   313,
     314,   315,   316,   379,   447,   371,   540,   438,   552,   427,
     554,   408,   484,   557,   343,   614,   326,   414,   321,   322,
     693,   341,   339,   346,   341,   319,   320,   340,   404,   349,
     349,   348,   340,   346,   400,   375,   341,   341,   435,   436,
     375,   371,   473,   348,   348,   339,   375,   341,   721,   379,
     371,   345,   362,   419,   357,   358,   375,   387,   379,   387,
     390,   387,   376,   349,   399,   341,   387,   381,   382,   487,
     400,   340,   348,   357,   404,   340,   519,   346,   521,   400,
     340,   346,   343,   404,   618,   340,   346,   343,   506,   419,
     346,   346,   375,   349,   346,   340,   340,   349,   419,   342,
     430,   346,   346,   346,   343,   346,   482,   651,   349,   430,
     323,   324,   584,   596,   597,   598,   599,   375,   484,   346,
     486,   552,   349,   554,   567,   375,   557,   328,   329,   330,
     331,   332,   333,   334,   335,   336,   337,   341,   346,   346,
     571,   349,   349,   364,   365,   366,   349,   348,   354,   355,
     356,   351,   482,   353,   484,   754,   486,   344,   692,   346,
     704,   482,   348,   484,   375,   486,   317,   318,   346,   347,
     346,   347,   592,   593,   340,   594,   595,   620,   600,   601,
     339,   624,   579,   655,   341,   375,   375,   361,   360,   359,
     325,   327,   558,   342,   341,   339,   344,   349,   339,   349,
     744,   339,   339,   349,   347,   344,   375,   349,   339,   349,
     540,   642,   643,   339,   375,   349,   349,   340,   584,   540,
     651,   342,   375,   342,   346,   339,   760,   383,   558,   342,
     375,   340,   339,   343,   340,   340,   348,   558,   349,   343,
     343,   349,   344,   602,   687,   603,   606,   604,   710,   605,
     693,   489,   607,   320,   584,   387,   578,   665,   695,   738,
     750,   695,   751,   584,   721,   550,   712,   387,   550,   550,
      -1,   400,    -1,   704,   404,    -1,   404,    -1,   721,    -1,
     398,    -1,    -1,    -1,   614,    -1,    -1,    -1,   618,   655,
     733,    -1,   710,   614,    -1,   726,    -1,   618,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   749,    -1,    -1,    -1,
      -1,    -1,    -1,   744,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   655,    -1,   693,    -1,    -1,
      -1,    -1,    -1,    -1,   655,    -1,    -1,    -1,    -1,    -1,
     716,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   721,    -1,    -1,    -1,    -1,
      -1,    -1,   692,   693,    -1,   695,    -1,    -1,    -1,    -1,
      -1,   692,   693,    -1,   695,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   716,    -1,    -1,    -1,
      -1,   721,    -1,    -1,    -1,   716,    -1,    -1,    -1,    -1,
     721,    -1,   732,    -1,    -1,    -1,    -1,    -1,    -1,   739,
      -1,   732,    -1,    -1,    -1,    -1,    -1,    -1,   739,    -1,
     750,    -1,    -1,    -1,   754,    -1,    -1,    -1,    -1,   750,
     760,    -1,    -1,   754,    -1,    -1,    -1,     0,    -1,   760,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    59,    60,    61,    62,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,   107,   108,   109,   110,   111,   112,
     113,   114,   115,   116,   117,   118,   119,   120,   121,   122,
     123,   124,   125,   126,   127,   128,   129,   130,   131,   132,
     133,   134,   135,   136,   137,   138,   139,   140,   141,   142,
     143,   144,   145,   146,   147,   148,   149,   150,   151,   152,
     153,   154,   155,   156,   157,   158,   159,   160,   161,   162,
     163,   164,   165,   166,   167,   168,   169,   170,   171,   172,
     173,   174,   175,   176,   177,   178,   179,   180,   181,   182,
     183,   184,   185,   186,   187,   188,   189,   190,   191,   192,
     193,   194,   195,   196,   197,   198,   199,   200,   201,   202,
     203,   204,   205,   206,   207,   208,   209,   210,   211,   212,
     213,   214,   215,   216,   217,   218,   219,   220,   221,   222,
     223,   224,   225,   226,   227,   228,   229,   230,   231,   232,
     233,   234,   235,   236,   237,   238,   239,   240,   241,   242,
     243,   244,   245,   246,   247,   248,   249,   250,   251,   252,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   284,   285,   286,   287,   288,   289,   290,   291,   292,
     293,   294,   295,   296,   297,   298,   299,   300,   301,   302,
     303,   304,   305,   306,   307,   308,   309,   310,   311,   312,
     313,   314,   315,   316,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   349,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     363,   364,   365,   366,   367,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   376,   377,   378,   379,   380,   381,   382,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   398,   399,   400,   401,   402,
     403,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   412,
      -1,   414,   415,   416,   417,   418,   419,   420,   421,   422,
     423,   424,   425,   426,   427,   428,   429,   430,   431,   432,
     433,   434,   435,   436,   437,   438,   439,   440,   441,   442,
     443,   444,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,    57,    58,    59,    60,
      61,    62,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,   106,   107,   108,   109,   110,
     111,   112,   113,   114,   115,   116,   117,   118,   119,   120,
     121,   122,   123,   124,   125,   126,   127,   128,   129,   130,
     131,   132,   133,   134,   135,   136,   137,   138,   139,   140,
     141,   142,   143,   144,   145,   146,   147,   148,   149,   150,
     151,   152,   153,   154,   155,   156,   157,   158,   159,   160,
     161,   162,   163,   164,   165,   166,   167,   168,   169,   170,
     171,   172,   173,   174,   175,   176,   177,   178,   179,   180,
     181,   182,   183,   184,   185,   186,   187,   188,   189,   190,
     191,   192,   193,   194,   195,   196,   197,   198,   199,   200,
     201,   202,   203,   204,   205,   206,   207,   208,   209,   210,
     211,   212,   213,   214,   215,   216,   217,   218,   219,   220,
     221,   222,   223,   224,   225,   226,   227,   228,   229,   230,
     231,   232,   233,   234,   235,   236,   237,   238,   239,   240,
     241,   242,   243,   244,   245,   246,   247,   248,   249,   250,
     251,   252,   253,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   284,   285,   286,   287,   288,   289,   290,
     291,   292,   293,   294,   295,   296,   297,   298,   299,   300,
     301,   302,   303,   304,   305,   306,   307,   308,   309,   310,
     311,   312,   313,   314,   315,   316,    -1,    -1,   319,   320,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   338,   339,    -1,
     341,    -1,   343,   344,    -1,    -1,    -1,    -1,   349,   350,
     351,   352,   353,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   363,   364,   365,   366,   367,    -1,    -1,    -1,
     371,   372,   373,   374,   375,   376,   377,   378,   379,   380,
     381,   382,   383,   384,   385,   386,    -1,   388,   389,   390,
     391,   392,   393,   394,   395,   396,   397,   398,   399,   400,
     401,   402,   403,   404,   405,   406,   407,   408,   409,   410,
     411,   412,   413,   414,   415,   416,   417,   418,   419,   420,
     421,   422,   423,   424,   425,   426,   427,   428,   429,   430,
     431,   432,   433,   434,   435,   436,   437,   438,   439,   440,
     441,   442,   443,   444,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    57,    58,
      59,    60,    61,    62,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,   107,   108,
     109,   110,   111,   112,   113,   114,   115,   116,   117,   118,
     119,   120,   121,   122,   123,   124,   125,   126,   127,   128,
     129,   130,   131,   132,   133,   134,   135,   136,   137,   138,
     139,   140,   141,   142,   143,   144,   145,   146,   147,   148,
     149,   150,   151,   152,   153,   154,   155,   156,   157,   158,
     159,   160,   161,   162,   163,   164,   165,   166,   167,   168,
     169,   170,   171,   172,   173,   174,   175,   176,   177,   178,
     179,   180,   181,   182,   183,   184,   185,   186,   187,   188,
     189,   190,   191,   192,   193,   194,   195,   196,   197,   198,
     199,   200,   201,   202,   203,   204,   205,   206,   207,   208,
     209,   210,   211,   212,   213,   214,   215,   216,   217,   218,
     219,   220,   221,   222,   223,   224,   225,   226,   227,   228,
     229,   230,   231,   232,   233,   234,   235,   236,   237,   238,
     239,   240,   241,   242,   243,   244,   245,   246,   247,   248,
     249,   250,   251,   252,   253,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   283,   284,   285,   286,   287,   288,
     289,   290,   291,   292,   293,   294,   295,   296,   297,   298,
     299,   300,   301,   302,   303,   304,   305,   306,   307,   308,
     309,   310,   311,   312,   313,   314,   315,   316,    -1,    -1,
     319,   320,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   338,
     339,    -1,   341,    -1,   343,   344,    -1,    -1,    -1,    -1,
     349,   350,   351,   352,   353,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   363,   364,   365,   366,   367,    -1,
      -1,    -1,   371,   372,   373,   374,   375,   376,   377,   378,
     379,   380,   381,   382,   383,   384,   385,   386,    -1,   388,
     389,   390,   391,   392,   393,   394,   395,   396,   397,   398,
     399,   400,   401,   402,   403,   404,   405,   406,   407,   408,
     409,   410,   411,   412,   413,   414,   415,   416,   417,   418,
     419,   420,   421,   422,   423,   424,   425,   426,   427,   428,
     429,   430,   431,   432,   433,   434,   435,   436,   437,   438,
     439,   440,   441,   442,   443,   444,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      37,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
      57,    58,    59,    60,    61,    62,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
     107,   108,   109,   110,   111,   112,   113,   114,   115,   116,
     117,   118,   119,   120,   121,   122,   123,   124,   125,   126,
     127,   128,   129,   130,   131,   132,   133,   134,   135,   136,
     137,   138,   139,   140,   141,   142,   143,   144,   145,   146,
     147,   148,   149,   150,   151,   152,   153,   154,   155,   156,
     157,   158,   159,   160,   161,   162,   163,   164,   165,   166,
     167,   168,   169,   170,   171,   172,   173,   174,   175,   176,
     177,   178,   179,   180,   181,   182,   183,   184,   185,   186,
     187,   188,   189,   190,   191,   192,   193,   194,   195,   196,
     197,   198,   199,   200,   201,   202,   203,   204,   205,   206,
     207,   208,   209,   210,   211,   212,   213,   214,   215,   216,
     217,   218,   219,   220,   221,   222,   223,   224,   225,   226,
     227,   228,   229,   230,   231,   232,   233,   234,   235,   236,
     237,   238,   239,   240,   241,   242,   243,   244,   245,   246,
     247,   248,   249,   250,   251,   252,   253,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,   283,   284,   285,   286,
     287,   288,   289,   290,   291,   292,   293,   294,   295,   296,
     297,   298,   299,   300,   301,   302,   303,   304,   305,   306,
     307,   308,   309,   310,   311,   312,   313,   314,   315,   316,
      -1,    -1,   319,   320,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   338,   339,    -1,   341,    -1,   343,    -1,    -1,    -1,
      -1,    -1,   349,   350,   351,   352,   353,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   363,   364,   365,   366,
     367,    -1,    -1,    -1,   371,   372,   373,   374,   375,   376,
     377,   378,   379,   380,   381,   382,   383,   384,   385,   386,
      -1,   388,   389,   390,   391,   392,   393,   394,   395,   396,
     397,   398,   399,   400,   401,   402,   403,   404,   405,   406,
     407,   408,   409,   410,   411,   412,   413,   414,   415,   416,
     417,   418,   419,   420,   421,   422,   423,   424,   425,   426,
     427,   428,   429,   430,   431,   432,   433,   434,   435,   436,
     437,   438,   439,   440,   441,   442,   443,   444,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   139,   140,   141,   142,   143,   144,
     145,   146,   147,   148,   149,   150,   151,   152,   153,   154,
     155,   156,   157,   158,   159,   160,   161,   162,   163,   164,
     165,   166,   167,   168,   169,   170,   171,   172,   173,   174,
     175,   176,   177,   178,   179,   180,   181,   182,   183,   184,
     185,   186,   187,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   197,   198,   199,   200,   201,   202,   203,   204,
     205,   206,   207,   208,   209,   210,   211,   212,   213,   214,
     215,   216,   217,   218,   219,   220,   221,   222,   223,   224,
     225,   226,   227,   228,   229,   230,   231,   232,   233,   234,
     235,   236,   237,   238,   239,   240,   241,   242,   243,   244,
     245,   246,   247,   248,   249,   250,   251,   252,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,    -1,    -1,   319,   320,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   338,   339,    -1,   341,    -1,   343,    -1,
      -1,    -1,    -1,    -1,   349,   350,   351,   352,   353,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   363,   364,
     365,   366,   367,    -1,    -1,    -1,   371,   372,   373,   374,
     375,   376,   377,   378,   379,   380,   381,   382,   383,   384,
     385,   386,    -1,   388,   389,   390,   391,   392,   393,   394,
     395,   396,   397,   398,   399,   400,   401,   402,   403,   404,
     405,   406,   407,   408,   409,   410,   411,   412,   413,   414,
     415,   416,   417,   418,   419,   420,   421,   422,   423,   424,
     425,   426,   427,   428,   429,   430,   431,   432,   433,   434,
     435,   436,   437,   438,   439,   440,   441,   442,   443,   444,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    59,    60,    61,    62,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,   107,   108,   109,   110,   111,   112,
     113,   114,   115,   116,   117,   118,   119,   120,   121,   122,
     123,   124,   125,   126,   127,   128,   129,   130,   131,   132,
     133,   134,   135,   136,   137,   138,   139,   140,   141,   142,
     143,   144,   145,   146,   147,   148,   149,   150,   151,   152,
     153,   154,   155,   156,   157,   158,   159,   160,   161,   162,
     163,   164,   165,   166,   167,   168,   169,   170,   171,   172,
     173,   174,   175,   176,   177,   178,   179,   180,   181,   182,
     183,   184,   185,   186,   187,   188,   189,   190,   191,   192,
     193,   194,   195,   196,   197,   198,   199,   200,   201,   202,
     203,   204,   205,   206,   207,   208,   209,   210,   211,   212,
     213,   214,   215,   216,   217,   218,   219,   220,   221,   222,
     223,   224,   225,   226,   227,   228,   229,   230,   231,   232,
     233,   234,   235,   236,   237,   238,   239,   240,   241,   242,
     243,   244,   245,   246,   247,   248,   249,   250,   251,   252,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   284,   285,   286,   287,   288,   289,   290,   291,   292,
     293,   294,   295,   296,   297,   298,   299,   300,   301,   302,
     303,   304,   305,   306,   307,   308,   309,   310,   311,   312,
     313,   314,   315,   316,    -1,    -1,   319,   320,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   338,   339,    -1,   341,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   349,   350,   351,   352,
     353,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     363,   364,   365,   366,   367,    -1,    -1,    -1,   371,   372,
     373,   374,   375,   376,   377,   378,   379,   380,   381,   382,
     383,   384,   385,   386,    -1,   388,   389,   390,   391,   392,
     393,   394,   395,   396,   397,   398,   399,   400,   401,   402,
     403,   404,   405,   406,   407,   408,   409,   410,   411,   412,
     413,   414,   415,   416,   417,   418,   419,   420,   421,   422,
     423,   424,   425,   426,   427,   428,   429,   430,   431,   432,
     433,   434,   435,   436,   437,   438,   439,   440,   441,   442,
     443,   444,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,    57,    58,    59,    60,
      61,    62,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,   106,   107,   108,   109,   110,
     111,   112,   113,   114,   115,   116,   117,   118,   119,   120,
     121,   122,   123,   124,   125,   126,   127,   128,   129,   130,
     131,   132,   133,   134,   135,   136,   137,   138,   139,   140,
     141,   142,   143,   144,   145,   146,   147,   148,   149,   150,
     151,   152,   153,   154,   155,   156,   157,   158,   159,   160,
     161,   162,   163,   164,   165,   166,   167,   168,   169,   170,
     171,   172,   173,   174,   175,   176,   177,   178,   179,   180,
     181,   182,   183,   184,   185,   186,   187,   188,   189,   190,
     191,   192,   193,   194,   195,   196,   197,   198,   199,   200,
     201,   202,   203,   204,   205,   206,   207,   208,   209,   210,
     211,   212,   213,   214,   215,   216,   217,   218,   219,   220,
     221,   222,   223,   224,   225,   226,   227,   228,   229,   230,
     231,   232,   233,   234,   235,   236,   237,   238,   239,   240,
     241,   242,   243,   244,   245,   246,   247,   248,   249,   250,
     251,   252,   253,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   284,   285,   286,   287,   288,   289,   290,
     291,   292,   293,   294,   295,   296,   297,   298,   299,   300,
     301,   302,   303,   304,   305,   306,   307,   308,   309,   310,
     311,   312,   313,   314,   315,   316,    -1,    -1,   319,   320,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   338,   339,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   349,   350,
     351,   352,   353,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   363,   364,   365,   366,   367,    -1,    -1,    -1,
     371,   372,   373,   374,   375,   376,   377,   378,   379,   380,
     381,   382,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   398,   399,   400,
     401,   402,   403,   404,   405,   406,   407,   408,   409,   410,
     411,   412,    -1,   414,   415,   416,   417,   418,   419,   420,
     421,   422,   423,   424,   425,   426,   427,   428,   429,   430,
     431,   432,   433,   434,   435,   436,   437,   438,   439,   440,
     441,   442,   443,   444,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    57,    58,
      59,    60,    61,    62,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,   107,   108,
     109,   110,   111,   112,   113,   114,   115,   116,   117,   118,
     119,   120,   121,   122,   123,   124,   125,   126,   127,   128,
     129,   130,   131,   132,   133,   134,   135,   136,   137,   138,
     139,   140,   141,   142,   143,   144,   145,   146,   147,   148,
     149,   150,   151,   152,   153,   154,   155,   156,   157,   158,
     159,   160,   161,   162,   163,   164,   165,   166,   167,   168,
     169,   170,   171,   172,   173,   174,   175,   176,   177,   178,
     179,   180,   181,   182,   183,   184,   185,   186,   187,   188,
     189,   190,   191,   192,   193,   194,   195,   196,   197,   198,
     199,   200,   201,   202,   203,   204,   205,   206,   207,   208,
     209,   210,   211,   212,   213,   214,   215,   216,   217,   218,
     219,   220,   221,   222,   223,   224,   225,   226,   227,   228,
     229,   230,   231,   232,   233,   234,   235,   236,   237,   238,
     239,   240,   241,   242,   243,   244,   245,   246,   247,   248,
     249,   250,   251,   252,   253,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   283,   284,   285,   286,   287,   288,
     289,   290,   291,   292,   293,   294,   295,   296,   297,   298,
     299,   300,   301,   302,   303,   304,   305,   306,   307,   308,
     309,   310,   311,   312,   313,   314,   315,   316,    -1,    -1,
     319,   320,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   338,
     339,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   350,   351,   352,   353,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   363,   364,   365,   366,    -1,    -1,
      -1,    -1,   371,   372,   373,   374,   375,   376,   377,   378,
     379,   380,   381,   382,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   398,
     399,   400,   401,   402,   403,   404,   405,   406,   407,   408,
     409,   410,   411,   412,    -1,   414,   415,   416,   417,   418,
     419,   420,   421,   422,   423,   424,   425,   426,   427,   428,
     429,   430,   431,   432,   433,   434,   435,   436,   437,   438,
     439,   440,   441,   442,   443,   444,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      37,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
      57,    58,    59,    60,    61,    62,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
     107,   108,   109,   110,   111,   112,   113,   114,   115,   116,
     117,   118,   119,   120,   121,   122,   123,   124,   125,   126,
     127,   128,   129,   130,   131,   132,   133,   134,   135,   136,
     137,   138,   139,   140,   141,   142,   143,   144,   145,   146,
     147,   148,   149,   150,   151,   152,   153,   154,   155,   156,
     157,   158,   159,   160,   161,   162,   163,   164,   165,   166,
     167,   168,   169,   170,   171,   172,   173,   174,   175,   176,
     177,   178,   179,   180,   181,   182,   183,   184,   185,   186,
     187,   188,   189,   190,   191,   192,   193,   194,   195,   196,
     197,   198,   199,   200,   201,   202,   203,   204,   205,   206,
     207,   208,   209,   210,   211,   212,   213,   214,   215,   216,
     217,   218,   219,   220,   221,   222,   223,   224,   225,   226,
     227,   228,   229,   230,   231,   232,   233,   234,   235,   236,
     237,   238,   239,   240,   241,   242,   243,   244,   245,   246,
     247,   248,   249,   250,   251,   252,   253,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,   283,   284,   285,   286,
     287,   288,   289,   290,   291,   292,   293,   294,   295,   296,
     297,   298,   299,   300,   301,   302,   303,   304,   305,   306,
     307,   308,   309,   310,   311,   312,   313,   314,   315,   316,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   349,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   363,   364,   365,   366,
     367,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   376,
     377,   378,   379,   380,   381,   382,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   398,   399,   400,   401,   402,   403,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   412,    -1,   414,   415,   416,
     417,   418,   419,   420,   421,   422,   423,   424,   425,   426,
     427,   428,   429,   430,   431,   432,   433,   434,   435,   436,
     437,   438,   439,   440,   441,   442,   443,   444,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   139,   140,   141,   142,   143,   144,
     145,   146,   147,   148,   149,   150,   151,   152,   153,   154,
     155,   156,   157,   158,   159,   160,   161,   162,   163,   164,
     165,   166,   167,   168,   169,   170,   171,   172,   173,   174,
     175,   176,   177,   178,   179,   180,   181,   182,   183,   184,
     185,   186,   187,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   197,   198,   199,   200,   201,   202,   203,   204,
     205,   206,   207,   208,   209,   210,   211,   212,   213,   214,
     215,   216,   217,   218,   219,   220,   221,   222,   223,   224,
     225,   226,   227,   228,   229,   230,   231,   232,   233,   234,
     235,   236,   237,   238,   239,   240,   241,   242,   243,   244,
     245,   246,   247,   248,   249,   250,   251,   252,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   349,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   363,   364,
     365,   366,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     375,   376,   377,   378,   379,   380,   381,   382,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   398,   399,   400,   401,   402,   403,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   412,    -1,   414,
     415,   416,   417,   418,   419,   420,   421,   422,   423,   424,
     425,   426,   427,   428,   429,   430,   431,   432,   433,   434,
     435,   436,   437,   438,   439,   440,   441,   442,   443,   444,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    59,    60,    61,    62,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,   107,   108,   109,   110,   111,   112,
     113,   114,   115,   116,   117,   118,   119,   120,   121,   122,
     123,   124,   125,   126,   127,   128,   129,   130,   131,   132,
     133,   134,   135,   136,   137,   138,   139,   140,   141,   142,
     143,   144,   145,   146,   147,   148,   149,   150,   151,   152,
     153,   154,   155,   156,   157,   158,   159,   160,   161,   162,
     163,   164,   165,   166,   167,   168,   169,   170,   171,   172,
     173,   174,   175,   176,   177,   178,   179,   180,   181,   182,
     183,   184,   185,   186,   187,   188,   189,   190,   191,   192,
     193,   194,   195,   196,   197,   198,   199,   200,   201,   202,
     203,   204,   205,   206,   207,   208,   209,   210,   211,   212,
     213,   214,   215,   216,   217,   218,   219,   220,   221,   222,
     223,   224,   225,   226,   227,   228,   229,   230,   231,   232,
     233,   234,   235,   236,   237,   238,   239,   240,   241,   242,
     243,   244,   245,   246,   247,   248,   249,   250,   251,   252,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   284,   285,   286,   287,   288,   289,   290,   291,   292,
     293,   294,   295,   296,   297,   298,   299,   300,   301,   302,
     303,   304,   305,   306,   307,   308,   309,   310,   311,   312,
     313,   314,   315,   316,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   344,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     363,   364,   365,   366,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   376,   377,   378,   379,   380,   381,   382,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   398,   399,   400,   401,   402,
     403,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   412,
      -1,   414,   415,   416,   417,   418,   419,   420,   421,   422,
     423,   424,   425,   426,   427,   428,   429,   430,   431,   432,
     433,   434,   435,   436,   437,   438,   439,   440,   441,   442,
     443,   444,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,    57,    58,    59,    60,
      61,    62,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,   106,   107,   108,   109,   110,
     111,   112,   113,   114,   115,   116,   117,   118,   119,   120,
     121,   122,   123,   124,   125,   126,   127,   128,   129,   130,
     131,   132,   133,   134,   135,   136,   137,   138,   139,   140,
     141,   142,   143,   144,   145,   146,   147,   148,   149,   150,
     151,   152,   153,   154,   155,   156,   157,   158,   159,   160,
     161,   162,   163,   164,   165,   166,   167,   168,   169,   170,
     171,   172,   173,   174,   175,   176,   177,   178,   179,   180,
     181,   182,   183,   184,   185,   186,   187,   188,   189,   190,
     191,   192,   193,   194,   195,   196,   197,   198,   199,   200,
     201,   202,   203,   204,   205,   206,   207,   208,   209,   210,
     211,   212,   213,   214,   215,   216,   217,   218,   219,   220,
     221,   222,   223,   224,   225,   226,   227,   228,   229,   230,
     231,   232,   233,   234,   235,   236,   237,   238,   239,   240,
     241,   242,   243,   244,   245,   246,   247,   248,   249,   250,
     251,   252,   253,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   284,   285,   286,   287,   288,   289,   290,
     291,   292,   293,   294,   295,   296,   297,   298,   299,   300,
     301,   302,   303,   304,   305,   306,   307,   308,   309,   310,
     311,   312,   313,   314,   315,   316,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   344,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   363,   364,   365,   366,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   376,   377,   378,   379,   380,
     381,   382,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   398,   399,   400,
     401,   402,   403,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   412,    -1,   414,   415,   416,   417,   418,   419,   420,
     421,   422,   423,   424,   425,   426,   427,   428,   429,   430,
     431,   432,   433,   434,   435,   436,   437,   438,   439,   440,
     441,   442,   443,   444,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    57,    58,
      59,    60,    61,    62,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,   107,   108,
     109,   110,   111,   112,   113,   114,   115,   116,   117,   118,
     119,   120,   121,   122,   123,   124,   125,   126,   127,   128,
     129,   130,   131,   132,   133,   134,   135,   136,   137,   138,
     139,   140,   141,   142,   143,   144,   145,   146,   147,   148,
     149,   150,   151,   152,   153,   154,   155,   156,   157,   158,
     159,   160,   161,   162,   163,   164,   165,   166,   167,   168,
     169,   170,   171,   172,   173,   174,   175,   176,   177,   178,
     179,   180,   181,   182,   183,   184,   185,   186,   187,   188,
     189,   190,   191,   192,   193,   194,   195,   196,   197,   198,
     199,   200,   201,   202,   203,   204,   205,   206,   207,   208,
     209,   210,   211,   212,   213,   214,   215,   216,   217,   218,
     219,   220,   221,   222,   223,   224,   225,   226,   227,   228,
     229,   230,   231,   232,   233,   234,   235,   236,   237,   238,
     239,   240,   241,   242,   243,   244,   245,   246,   247,   248,
     249,   250,   251,   252,   253,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   283,   284,   285,   286,   287,   288,
     289,   290,   291,   292,   293,   294,   295,   296,   297,   298,
     299,   300,   301,   302,   303,   304,   305,   306,   307,   308,
     309,   310,   311,   312,   313,   314,   315,   316,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   344,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   363,   364,   365,   366,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   376,   377,   378,
     379,   380,   381,   382,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   398,
     399,   400,   401,   402,   403,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   412,    -1,   414,   415,   416,   417,   418,
     419,   420,   421,   422,   423,   424,   425,   426,   427,   428,
     429,   430,   431,   432,   433,   434,   435,   436,   437,   438,
     439,   440,   441,   442,   443,   444,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      37,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
      57,    58,    59,    60,    61,    62,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
     107,   108,   109,   110,   111,   112,   113,   114,   115,   116,
     117,   118,   119,   120,   121,   122,   123,   124,   125,   126,
     127,   128,   129,   130,   131,   132,   133,   134,   135,   136,
     137,   138,   139,   140,   141,   142,   143,   144,   145,   146,
     147,   148,   149,   150,   151,   152,   153,   154,   155,   156,
     157,   158,   159,   160,   161,   162,   163,   164,   165,   166,
     167,   168,   169,   170,   171,   172,   173,   174,   175,   176,
     177,   178,   179,   180,   181,   182,   183,   184,   185,   186,
     187,   188,   189,   190,   191,   192,   193,   194,   195,   196,
     197,   198,   199,   200,   201,   202,   203,   204,   205,   206,
     207,   208,   209,   210,   211,   212,   213,   214,   215,   216,
     217,   218,   219,   220,   221,   222,   223,   224,   225,   226,
     227,   228,   229,   230,   231,   232,   233,   234,   235,   236,
     237,   238,   239,   240,   241,   242,   243,   244,   245,   246,
     247,   248,   249,   250,   251,   252,   253,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,   283,   284,   285,   286,
     287,   288,   289,   290,   291,   292,   293,   294,   295,   296,
     297,   298,   299,   300,   301,   302,   303,   304,   305,   306,
     307,   308,   309,   310,   311,   312,   313,   314,   315,   316,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   363,   364,   365,   366,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   376,
     377,   378,   379,   380,   381,   382,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   398,   399,   400,   401,   402,   403,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   412,    -1,   414,   415,   416,
     417,   418,   419,   420,   421,   422,   423,   424,   425,   426,
     427,   428,   429,   430,   431,   432,   433,   434,   435,   436,
     437,   438,   439,   440,   441,   442,   443,   444,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
      36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    57,    58,    59,    60,    -1,    -1,    63,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
     106,   107,   108,   109,   110,   111,   112,   113,   114,   115,
     116,   117,   118,   119,   120,   121,   122,   123,   124,   125,
     126,   127,   128,   129,   130,   131,   132,   133,   134,   135,
     136,   137,   138,   139,   140,   141,   142,   143,   144,   145,
     146,   147,   148,   149,   150,   151,   152,   153,   154,   155,
     156,   157,   158,   159,   160,   161,   162,   163,   164,   165,
     166,   167,   168,   169,   170,   171,   172,   173,   174,   175,
     176,   177,   178,   179,   180,   181,   182,   183,   184,   185,
     186,   187,   188,   189,   190,   191,   192,   193,   194,   195,
     196,   197,   198,   199,   200,   201,   202,   203,   204,   205,
     206,   207,   208,   209,   210,   211,   212,   213,   214,   215,
     216,   217,   218,   219,   220,   221,   222,   223,   224,   225,
     226,   227,   228,   229,   230,   231,   232,   233,   234,   235,
     236,   237,   238,   239,   240,   241,   242,   243,   244,   245,
     246,   247,   248,   249,   250,   251,   252,   253,   254,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   284,   285,
     286,   287,   288,   289,   290,   291,   292,   293,   294,   295,
     296,   297,   298,   299,   300,   301,   302,   303,   304,   305,
     306,   307,   308,   309,   310,   311,   312,   313,   314,   315,
     316,    -1,    -1,   319,   320,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   338,   339,    -1,    -1,    -1,   343,   344,    -1,
      -1,    -1,    -1,    -1,   350,   351,   352,   353,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   371,   372,   373,   374,   375,
     376,    -1,    -1,    -1,    -1,   381,   382,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   404,   405,
     406,   407,   408,   409,   410,   411,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     426,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    59,    60,    -1,    -1,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,   107,   108,   109,   110,   111,   112,
     113,   114,   115,   116,   117,   118,   119,   120,   121,   122,
     123,   124,   125,   126,   127,   128,   129,   130,   131,   132,
     133,   134,   135,   136,   137,   138,   139,   140,   141,   142,
     143,   144,   145,   146,   147,   148,   149,   150,   151,   152,
     153,   154,   155,   156,   157,   158,   159,   160,   161,   162,
     163,   164,   165,   166,   167,   168,   169,   170,   171,   172,
     173,   174,   175,   176,   177,   178,   179,   180,   181,   182,
     183,   184,   185,   186,   187,   188,   189,   190,   191,   192,
     193,   194,   195,   196,   197,   198,   199,   200,   201,   202,
     203,   204,   205,   206,   207,   208,   209,   210,   211,   212,
     213,   214,   215,   216,   217,   218,   219,   220,   221,   222,
     223,   224,   225,   226,   227,   228,   229,   230,   231,   232,
     233,   234,   235,   236,   237,   238,   239,   240,   241,   242,
     243,   244,   245,   246,   247,   248,   249,   250,   251,   252,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   284,   285,   286,   287,   288,   289,   290,   291,   292,
     293,   294,   295,   296,   297,   298,   299,   300,   301,   302,
     303,   304,   305,   306,   307,   308,   309,   310,   311,   312,
     313,   314,   315,   316,    -1,    -1,   319,   320,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   338,   339,    -1,    -1,   342,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   350,   351,   352,
     353,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   371,   372,
     373,   374,   375,   376,    -1,    -1,    -1,    -1,   381,   382,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   404,   405,   406,   407,   408,   409,   410,   411,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   426,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    36,    37,    38,    39,
      40,    41,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    56,    57,    58,    59,
      60,    -1,    -1,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    93,    94,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,   106,   107,   108,   109,
     110,   111,   112,   113,   114,   115,   116,   117,   118,   119,
     120,   121,   122,   123,   124,   125,   126,   127,   128,   129,
     130,   131,   132,   133,   134,   135,   136,   137,   138,   139,
     140,   141,   142,   143,   144,   145,   146,   147,   148,   149,
     150,   151,   152,   153,   154,   155,   156,   157,   158,   159,
     160,   161,   162,   163,   164,   165,   166,   167,   168,   169,
     170,   171,   172,   173,   174,   175,   176,   177,   178,   179,
     180,   181,   182,   183,   184,   185,   186,   187,   188,   189,
     190,   191,   192,   193,   194,   195,   196,   197,   198,   199,
     200,   201,   202,   203,   204,   205,   206,   207,   208,   209,
     210,   211,   212,   213,   214,   215,   216,   217,   218,   219,
     220,   221,   222,   223,   224,   225,   226,   227,   228,   229,
     230,   231,   232,   233,   234,   235,   236,   237,   238,   239,
     240,   241,   242,   243,   244,   245,   246,   247,   248,   249,
     250,   251,   252,   253,   254,   255,   256,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   278,   279,
     280,   281,   282,   283,   284,   285,   286,   287,   288,   289,
     290,   291,   292,   293,   294,   295,   296,   297,   298,   299,
     300,   301,   302,   303,   304,   305,   306,   307,   308,   309,
     310,   311,   312,   313,   314,   315,   316,    -1,    -1,   319,
     320,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   338,   339,
      -1,    -1,    -1,   343,    -1,    -1,    -1,    -1,    -1,    -1,
     350,   351,   352,   353,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   371,   372,   373,   374,   375,   376,    -1,    -1,    -1,
      -1,   381,   382,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   404,   405,   406,   407,   408,   409,
     410,   411,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   426,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      37,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
      57,    58,    59,    60,    -1,    -1,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
     107,   108,   109,   110,   111,   112,   113,   114,   115,   116,
     117,   118,   119,   120,   121,   122,   123,   124,   125,   126,
     127,   128,   129,   130,   131,   132,   133,   134,   135,   136,
     137,   138,   139,   140,   141,   142,   143,   144,   145,   146,
     147,   148,   149,   150,   151,   152,   153,   154,   155,   156,
     157,   158,   159,   160,   161,   162,   163,   164,   165,   166,
     167,   168,   169,   170,   171,   172,   173,   174,   175,   176,
     177,   178,   179,   180,   181,   182,   183,   184,   185,   186,
     187,   188,   189,   190,   191,   192,   193,   194,   195,   196,
     197,   198,   199,   200,   201,   202,   203,   204,   205,   206,
     207,   208,   209,   210,   211,   212,   213,   214,   215,   216,
     217,   218,   219,   220,   221,   222,   223,   224,   225,   226,
     227,   228,   229,   230,   231,   232,   233,   234,   235,   236,
     237,   238,   239,   240,   241,   242,   243,   244,   245,   246,
     247,   248,   249,   250,   251,   252,   253,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,   283,   284,   285,   286,
     287,   288,   289,   290,   291,   292,   293,   294,   295,   296,
     297,   298,   299,   300,   301,   302,   303,   304,   305,   306,
     307,   308,   309,   310,   311,   312,   313,   314,   315,   316,
      -1,    -1,   319,   320,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   338,   339,    -1,    -1,   342,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   350,   351,   352,   353,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   371,   372,   373,   374,   375,   376,
      -1,    -1,    -1,    -1,   381,   382,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   404,   405,   406,
     407,   408,   409,   410,   411,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   426,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,    57,    58,    59,    60,    -1,    -1,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,   107,   108,   109,   110,   111,   112,   113,
     114,   115,   116,   117,   118,   119,   120,   121,   122,   123,
     124,   125,   126,   127,   128,   129,   130,   131,   132,   133,
     134,   135,   136,   137,   138,   139,   140,   141,   142,   143,
     144,   145,   146,   147,   148,   149,   150,   151,   152,   153,
     154,   155,   156,   157,   158,   159,   160,   161,   162,   163,
     164,   165,   166,   167,   168,   169,   170,   171,   172,   173,
     174,   175,   176,   177,   178,   179,   180,   181,   182,   183,
     184,   185,   186,   187,   188,   189,   190,   191,   192,   193,
     194,   195,   196,   197,   198,   199,   200,   201,   202,   203,
     204,   205,   206,   207,   208,   209,   210,   211,   212,   213,
     214,   215,   216,   217,   218,   219,   220,   221,   222,   223,
     224,   225,   226,   227,   228,   229,   230,   231,   232,   233,
     234,   235,   236,   237,   238,   239,   240,   241,   242,   243,
     244,   245,   246,   247,   248,   249,   250,   251,   252,   253,
     254,   255,   256,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     284,   285,   286,   287,   288,   289,   290,   291,   292,   293,
     294,   295,   296,   297,   298,   299,   300,   301,   302,   303,
     304,   305,   306,   307,   308,   309,   310,   311,   312,   313,
     314,   315,   316,    -1,    -1,   319,   320,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   338,   339,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   349,   350,   351,   352,   353,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   371,   372,   373,
     374,   375,   376,    -1,    -1,    -1,    -1,   381,   382,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     404,   405,   406,   407,   408,   409,   410,   411,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   426,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,    57,    58,    59,    60,
      -1,    -1,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,   106,   107,   108,   109,   110,
     111,   112,   113,   114,   115,   116,   117,   118,   119,   120,
     121,   122,   123,   124,   125,   126,   127,   128,   129,   130,
     131,   132,   133,   134,   135,   136,   137,   138,   139,   140,
     141,   142,   143,   144,   145,   146,   147,   148,   149,   150,
     151,   152,   153,   154,   155,   156,   157,   158,   159,   160,
     161,   162,   163,   164,   165,   166,   167,   168,   169,   170,
     171,   172,   173,   174,   175,   176,   177,   178,   179,   180,
     181,   182,   183,   184,   185,   186,   187,   188,   189,   190,
     191,   192,   193,   194,   195,   196,   197,   198,   199,   200,
     201,   202,   203,   204,   205,   206,   207,   208,   209,   210,
     211,   212,   213,   214,   215,   216,   217,   218,   219,   220,
     221,   222,   223,   224,   225,   226,   227,   228,   229,   230,
     231,   232,   233,   234,   235,   236,   237,   238,   239,   240,
     241,   242,   243,   244,   245,   246,   247,   248,   249,   250,
     251,   252,   253,   254,   255,   256,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   284,   285,   286,   287,   288,   289,   290,
     291,   292,   293,   294,   295,   296,   297,   298,   299,   300,
     301,   302,   303,   304,   305,   306,   307,   308,   309,   310,
     311,   312,   313,   314,   315,   316,    -1,    -1,   319,   320,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   338,   339,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   350,
     351,   352,   353,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     371,   372,   373,   374,   375,   376,    -1,    -1,    -1,    -1,
     381,   382,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   404,   405,   406,   407,   408,   409,   410,
     411,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   426,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    33,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    57,
      58,    59,    60,    -1,    -1,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,   107,
     108,   109,   110,   111,   112,   113,   114,   115,   116,   117,
     118,   119,   120,   121,   122,   123,   124,   125,   126,   127,
     128,   129,   130,   131,   132,   133,   134,   135,   136,   137,
     138,   139,   140,   141,   142,   143,   144,   145,   146,   147,
     148,   149,   150,   151,   152,   153,   154,   155,   156,   157,
     158,   159,   160,   161,   162,   163,   164,   165,   166,   167,
     168,   169,   170,   171,   172,   173,   174,   175,   176,   177,
     178,   179,   180,   181,   182,   183,   184,   185,   186,   187,
     188,   189,   190,   191,   192,   193,   194,   195,   196,   197,
     198,   199,   200,   201,   202,   203,   204,   205,   206,   207,
     208,   209,   210,   211,   212,   213,   214,   215,   216,   217,
     218,   219,   220,   221,   222,   223,   224,   225,   226,   227,
     228,   229,   230,   231,   232,   233,   234,   235,   236,   237,
     238,   239,   240,   241,   242,   243,   244,   245,   246,   247,
     248,   249,   250,   251,   252,   253,   254,   255,   256,   257,
     258,   259,   260,   261,   262,   263,   264,   265,   266,   267,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,   279,   280,   281,   282,   283,   284,   285,   286,   287,
     288,   289,   290,   291,   292,   293,   294,   295,   296,   297,
     298,   299,   300,   301,   302,   303,   304,   305,   306,   307,
     308,   309,   310,   311,   312,   313,   314,   315,   316,    -1,
      -1,   319,   320,    -1,    -1,   413,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   427,
     338,   339,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   350,   351,   352,   353,   444,   445,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   371,   372,   373,   374,   375,   376,    -1,
      -1,    -1,    -1,   381,   382,    -1,    -1,    -1,   476,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   487,
      -1,    -1,    -1,    -1,    -1,    -1,   404,   405,   406,   407,
     408,   409,   410,   411,    -1,    -1,    -1,    -1,   506,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   426,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   576,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   589,   590,   591,   592,   593,   594,   595,   596,   597,
     598,   599,   600,   601,   602,   603,   604,   605,   606,   607,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   710
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_int16 yystos[] =
{
       0,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,    59,    60,    61,
      62,    63,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   106,   107,   108,   109,   110,   111,
     112,   113,   114,   115,   116,   117,   118,   119,   120,   121,
     122,   123,   124,   125,   126,   127,   128,   129,   130,   131,
     132,   133,   134,   135,   136,   137,   138,   139,   140,   141,
     142,   143,   144,   145,   146,   147,   148,   149,   150,   151,
     152,   153,   154,   155,   156,   157,   158,   159,   160,   161,
     162,   163,   164,   165,   166,   167,   168,   169,   170,   171,
     172,   173,   174,   175,   176,   177,   178,   179,   180,   181,
     182,   183,   184,   185,   186,   187,   188,   189,   190,   191,
     192,   193,   194,   195,   196,   197,   198,   199,   200,   201,
     202,   203,   204,   205,   206,   207,   208,   209,   210,   211,
     212,   213,   214,   215,   216,   217,   218,   219,   220,   221,
     222,   223,   224,   225,   226,   227,   228,   229,   230,   231,
     232,   233,   234,   235,   236,   237,   238,   239,   240,   241,
     242,   243,   244,   245,   246,   247,   248,   249,   250,   251,
     252,   253,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,   280,   281,
     282,   283,   284,   285,   286,   287,   288,   289,   290,   291,
     292,   293,   294,   295,   296,   297,   298,   299,   300,   301,
     302,   303,   304,   305,   306,   307,   308,   309,   310,   311,
     312,   313,   314,   315,   316,   349,   363,   364,   365,   366,
     367,   376,   377,   378,   379,   380,   381,   382,   398,   399,
     400,   401,   402,   403,   412,   414,   415,   416,   417,   418,
     419,   420,   421,   422,   423,   424,   425,   426,   427,   428,
     429,   430,   431,   432,   433,   434,   435,   436,   437,   438,
     439,   440,   441,   442,   443,   444,   476,   477,   480,   481,
     482,   483,   487,   488,   489,   490,   491,   492,   495,   496,
     497,   498,   499,   501,   506,   507,   508,   549,   550,   551,
     507,   343,   375,   339,   339,   349,   375,   349,   552,   340,
     346,   484,   485,   486,   496,   501,   346,   349,   375,   349,
     375,   497,   501,   357,   503,   504,     0,   550,   501,   510,
     343,   375,   399,   493,   494,   375,   500,   341,   349,   502,
     343,   528,   485,   484,   486,   375,   375,   339,   348,   502,
     343,   346,   349,   479,   319,   320,   338,   339,   350,   351,
     352,   353,   371,   372,   373,   374,   375,   404,   405,   406,
     407,   408,   409,   410,   411,   446,   447,   448,   450,   451,
     452,   453,   454,   455,   456,   457,   458,   499,   501,   505,
     502,   349,   496,   501,   511,   512,   509,   348,   340,   346,
     340,   346,   342,   457,   459,   460,   461,   462,   463,   464,
     465,   466,   467,   468,   469,   470,   341,   349,   341,   343,
     344,   349,   383,   384,   385,   386,   388,   389,   390,   391,
     392,   393,   394,   395,   396,   397,   413,   457,   470,   472,
     474,   476,   480,   499,   501,   517,   518,   519,   520,   521,
     529,   530,   531,   532,   535,   536,   539,   540,   541,   548,
     553,   502,   348,   502,   343,   472,   515,   348,   478,   375,
     346,   349,   457,   457,   474,   319,   320,   341,   345,   340,
     340,   346,   382,   472,   339,   457,   346,   358,   501,   375,
     513,   514,   344,   512,   511,   470,   475,   494,   375,   354,
     355,   356,   351,   353,   317,   318,   321,   322,   357,   358,
     323,   324,   361,   360,   359,   325,   327,   326,   362,   342,
     342,   470,   341,   344,   522,   339,   349,   349,   543,   339,
     339,   349,   349,   474,   339,   474,   347,   349,   349,   349,
     349,   328,   329,   330,   331,   332,   333,   334,   335,   336,
     337,   348,   473,   346,   349,   344,   518,   532,   536,   541,
     515,   348,   515,   516,   515,   511,   375,   340,   449,   474,
     375,   472,   457,   513,   502,   346,   349,   344,   457,   457,
     457,   459,   459,   460,   460,   461,   461,   461,   461,   462,
     462,   463,   464,   465,   466,   467,   468,   471,   342,   375,
     554,   555,   529,   542,   518,   544,   474,   349,   474,   347,
     472,   472,   515,   344,   346,   344,   342,   349,   514,   474,
     339,   342,   346,   523,   474,   489,   496,   534,   383,   517,
     530,   545,   340,   340,   344,   515,   347,   475,   342,   555,
     344,   375,   340,   339,   534,   546,   547,   525,   526,   527,
     533,   537,   472,   340,   348,   519,   524,   528,   474,   349,
     340,   387,   521,   519,   343,   515,   340,   474,   524,   525,
     529,   538,   349,   344
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_int16 yyr1[] =
{
       0,   445,   446,   447,   447,   447,   447,   447,   447,   447,
     447,   447,   447,   447,   447,   447,   447,   447,   448,   448,
     448,   448,   448,   448,   449,   450,   451,   452,   452,   453,
     453,   454,   454,   455,   456,   456,   456,   457,   457,   457,
     457,   458,   458,   458,   458,   459,   459,   459,   459,   460,
     460,   460,   461,   461,   461,   462,   462,   462,   462,   462,
     463,   463,   463,   464,   464,   465,   465,   466,   466,   467,
     467,   468,   468,   469,   469,   470,   471,   470,   472,   472,
     473,   473,   473,   473,   473,   473,   473,   473,   473,   473,
     473,   474,   474,   475,   476,   476,   476,   476,   476,   476,
     476,   476,   476,   478,   477,   479,   479,   480,   481,   481,
     482,   482,   483,   484,   484,   485,   485,   485,   485,   486,
     487,   487,   487,   487,   487,   488,   488,   488,   488,   488,
     489,   489,   490,   491,   491,   491,   491,   491,   491,   491,
     491,   492,   493,   493,   494,   494,   494,   495,   496,   496,
     497,   497,   497,   497,   497,   497,   497,   498,   498,   498,
     498,   498,   498,   498,   498,   498,   498,   498,   498,   498,
     498,   498,   498,   498,   498,   498,   498,   498,   498,   498,
     498,   498,   498,   498,   498,   498,   498,   498,   498,   498,
     498,   498,   499,   500,   500,   501,   501,   502,   502,   502,
     502,   503,   503,   504,   505,   505,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     506,   506,   506,   506,   506,   506,   506,   506,   506,   506,
     507,   507,   507,   509,   508,   510,   508,   511,   511,   512,
     512,   513,   513,   514,   514,   515,   515,   515,   516,   516,
     517,   518,   518,   519,   519,   519,   519,   519,   519,   519,
     519,   520,   521,   522,   523,   521,   524,   524,   526,   525,
     527,   525,   528,   528,   529,   529,   530,   530,   531,   531,
     532,   533,   533,   534,   534,   535,   535,   537,   536,   538,
     538,   539,   539,   540,   540,   542,   541,   543,   541,   544,
     541,   545,   545,   546,   546,   547,   547,   548,   548,   548,
     548,   548,   548,   548,   548,   549,   549,   550,   550,   550,
     552,   551,   553,   554,   554,   555,   555
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     1,     1,     3,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     4,
       1,     3,     2,     2,     1,     1,     1,     2,     2,     2,
       1,     2,     3,     2,     1,     1,     1,     1,     2,     2,
       2,     1,     1,     1,     1,     1,     3,     3,     3,     1,
       3,     3,     1,     3,     3,     1,     3,     3,     3,     3,
       1,     3,     3,     1,     3,     1,     3,     1,     3,     1,
       3,     1,     3,     1,     3,     1,     0,     6,     1,     3,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     3,     1,     2,     2,     4,     2,     3,     4,
       2,     3,     4,     0,     6,     2,     3,     2,     1,     1,
       2,     3,     3,     2,     3,     2,     1,     2,     1,     1,
       1,     3,     4,     6,     5,     1,     2,     3,     5,     4,
       1,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     4,     1,     3,     1,     3,     1,     1,     1,     2,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     4,     1,     1,     3,     2,     3,     2,     3,     3,
       4,     1,     0,     3,     1,     3,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     0,     6,     0,     5,     1,     2,     3,
       4,     1,     3,     1,     2,     1,     3,     4,     1,     3,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     2,     2,     0,     0,     5,     1,     1,     0,     2,
       0,     2,     2,     3,     1,     2,     1,     2,     1,     2,
       5,     3,     1,     1,     4,     1,     2,     0,     8,     0,
       1,     3,     2,     1,     2,     0,     6,     0,     8,     0,
       7,     1,     1,     1,     0,     2,     3,     2,     2,     2,
       3,     2,     2,     2,     2,     1,     2,     1,     1,     1,
       0,     3,     5,     1,     3,     1,     4
};


enum { YYENOMEM = -2 };

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == YYEMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (pParseContext, YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Backward compatibility with an undocumented macro.
   Use YYerror or YYUNDEF. */
#define YYERRCODE YYUNDEF


/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)

/* This macro is provided for backward compatibility. */
# ifndef YY_LOCATION_PRINT
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif


# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Kind, Value, pParseContext); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo,
                       yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep, glslang::TParseContext* pParseContext)
{
  FILE *yyoutput = yyo;
  YYUSE (yyoutput);
  YYUSE (pParseContext);
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yykind < YYNTOKENS)
    YYPRINT (yyo, yytoknum[yykind], *yyvaluep);
# endif
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo,
                 yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep, glslang::TParseContext* pParseContext)
{
  YYFPRINTF (yyo, "%s %s (",
             yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name (yykind));

  yy_symbol_value_print (yyo, yykind, yyvaluep, pParseContext);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp,
                 int yyrule, glslang::TParseContext* pParseContext)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       YY_ACCESSING_SYMBOL (+yyssp[yyi + 1 - yynrhs]),
                       &yyvsp[(yyi + 1) - (yynrhs)], pParseContext);
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule, pParseContext); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args) ((void) 0)
# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


/* Context of a parse error.  */
typedef struct
{
  yy_state_t *yyssp;
  yysymbol_kind_t yytoken;
} yypcontext_t;

/* Put in YYARG at most YYARGN of the expected tokens given the
   current YYCTX, and return the number of tokens stored in YYARG.  If
   YYARG is null, return the number of expected tokens (guaranteed to
   be less than YYNTOKENS).  Return YYENOMEM on memory exhaustion.
   Return 0 if there are more than YYARGN expected tokens, yet fill
   YYARG up to YYARGN. */
static int
yypcontext_expected_tokens (const yypcontext_t *yyctx,
                            yysymbol_kind_t yyarg[], int yyargn)
{
  /* Actual size of YYARG. */
  int yycount = 0;
  int yyn = yypact[+*yyctx->yyssp];
  if (!yypact_value_is_default (yyn))
    {
      /* Start YYX at -YYN if negative to avoid negative indexes in
         YYCHECK.  In other words, skip the first -YYN actions for
         this state because they are default actions.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;
      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yyx;
      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
        if (yycheck[yyx + yyn] == yyx && yyx != YYSYMBOL_YYerror
            && !yytable_value_is_error (yytable[yyx + yyn]))
          {
            if (!yyarg)
              ++yycount;
            else if (yycount == yyargn)
              return 0;
            else
              yyarg[yycount++] = YY_CAST (yysymbol_kind_t, yyx);
          }
    }
  if (yyarg && yycount == 0 && 0 < yyargn)
    yyarg[0] = YYSYMBOL_YYEMPTY;
  return yycount;
}




#ifndef yystrlen
# if defined __GLIBC__ && defined _STRING_H
#  define yystrlen(S) (YY_CAST (YYPTRDIFF_T, strlen (S)))
# else
/* Return the length of YYSTR.  */
static YYPTRDIFF_T
yystrlen (const char *yystr)
{
  YYPTRDIFF_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
# endif
#endif

#ifndef yystpcpy
# if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#  define yystpcpy stpcpy
# else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
yystpcpy (char *yydest, const char *yysrc)
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
# endif
#endif

#ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYPTRDIFF_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYPTRDIFF_T yyn = 0;
      char const *yyp = yystr;
      for (;;)
        switch (*++yyp)
          {
          case '\'':
          case ',':
            goto do_not_strip_quotes;

          case '\\':
            if (*++yyp != '\\')
              goto do_not_strip_quotes;
            else
              goto append;

          append:
          default:
            if (yyres)
              yyres[yyn] = *yyp;
            yyn++;
            break;

          case '"':
            if (yyres)
              yyres[yyn] = '\0';
            return yyn;
          }
    do_not_strip_quotes: ;
    }

  if (yyres)
    return yystpcpy (yyres, yystr) - yyres;
  else
    return yystrlen (yystr);
}
#endif


static int
yy_syntax_error_arguments (const yypcontext_t *yyctx,
                           yysymbol_kind_t yyarg[], int yyargn)
{
  /* Actual size of YYARG. */
  int yycount = 0;
  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yyctx->yytoken != YYSYMBOL_YYEMPTY)
    {
      int yyn;
      if (yyarg)
        yyarg[yycount] = yyctx->yytoken;
      ++yycount;
      yyn = yypcontext_expected_tokens (yyctx,
                                        yyarg ? yyarg + 1 : yyarg, yyargn - 1);
      if (yyn == YYENOMEM)
        return YYENOMEM;
      else
        yycount += yyn;
    }
  return yycount;
}

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return -1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return YYENOMEM if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYPTRDIFF_T *yymsg_alloc, char **yymsg,
                const yypcontext_t *yyctx)
{
  enum { YYARGS_MAX = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat: reported tokens (one for the "unexpected",
     one per "expected"). */
  yysymbol_kind_t yyarg[YYARGS_MAX];
  /* Cumulated lengths of YYARG.  */
  YYPTRDIFF_T yysize = 0;

  /* Actual size of YYARG. */
  int yycount = yy_syntax_error_arguments (yyctx, yyarg, YYARGS_MAX);
  if (yycount == YYENOMEM)
    return YYENOMEM;

  switch (yycount)
    {
#define YYCASE_(N, S)                       \
      case N:                               \
        yyformat = S;                       \
        break
    default: /* Avoid compiler warnings. */
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
#undef YYCASE_
    }

  /* Compute error message size.  Don't count the "%s"s, but reserve
     room for the terminator.  */
  yysize = yystrlen (yyformat) - 2 * yycount + 1;
  {
    int yyi;
    for (yyi = 0; yyi < yycount; ++yyi)
      {
        YYPTRDIFF_T yysize1
          = yysize + yytnamerr (YY_NULLPTR, yytname[yyarg[yyi]]);
        if (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM)
          yysize = yysize1;
        else
          return YYENOMEM;
      }
  }

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return -1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yytname[yyarg[yyi++]]);
          yyformat += 2;
        }
      else
        {
          ++yyp;
          ++yyformat;
        }
  }
  return 0;
}


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg,
            yysymbol_kind_t yykind, YYSTYPE *yyvaluep, glslang::TParseContext* pParseContext)
{
  YYUSE (yyvaluep);
  YYUSE (pParseContext);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yykind, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}






/*----------.
| yyparse.  |
`----------*/

int
yyparse (glslang::TParseContext* pParseContext)
{
/* Lookahead token kind.  */
int yychar;


/* The semantic value of the lookahead symbol.  */
/* Default value used for initialization, for pacifying older GCCs
   or non-GCC compilers.  */
YY_INITIAL_VALUE (static YYSTYPE yyval_default;)
YYSTYPE yylval YY_INITIAL_VALUE (= yyval_default);

    /* Number of syntax errors so far.  */
    int yynerrs = 0;

    yy_state_fast_t yystate = 0;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus = 0;

    /* Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* Their size.  */
    YYPTRDIFF_T yystacksize = YYINITDEPTH;

    /* The state stack: array, bottom, top.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss = yyssa;
    yy_state_t *yyssp = yyss;

    /* The semantic value stack: array, bottom, top.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs = yyvsa;
    YYSTYPE *yyvsp = yyvs;

  int yyn;
  /* The return value of yyparse.  */
  int yyresult;
  /* Lookahead symbol kind.  */
  yysymbol_kind_t yytoken = YYSYMBOL_YYEMPTY;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYPTRDIFF_T yymsg_alloc = sizeof yymsgbuf;

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yychar = YYEMPTY; /* Cause a token to be read.  */
  goto yysetstate;


/*------------------------------------------------------------.
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END
  YY_STACK_PRINT (yyss, yyssp);

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    goto yyexhaustedlab;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          goto yyexhaustedlab;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;


/*-----------.
| yybackup.  |
`-----------*/
yybackup:
  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either empty, or end-of-input, or a valid lookahead.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token\n"));
      yychar = yylex (&yylval, parseContext);
    }

  if (yychar <= YYEOF)
    {
      yychar = YYEOF;
      yytoken = YYSYMBOL_YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else if (yychar == YYerror)
    {
      /* The scanner already issued an error message, process directly
         to error recovery.  But do not keep the error token as
         lookahead, it is too special and may lead us to an endless
         loop in error recovery. */
      yychar = YYUNDEF;
      yytoken = YYSYMBOL_YYerror;
      goto yyerrlab1;
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  /* Discard the shifted token.  */
  yychar = YYEMPTY;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
  case 2: /* variable_identifier: IDENTIFIER  */
#line 371 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.intermTypedNode) = parseContext.handleVariable((yyvsp[0].lex).loc, (yyvsp[0].lex).symbol, (yyvsp[0].lex).string);
    }
#line 4594 "MachineIndependent/glslang_tab.cpp"
    break;

  case 3: /* primary_expression: variable_identifier  */
#line 377 "MachineIndependent/glslang.y"
                          {
        (yyval.interm.intermTypedNode) = (yyvsp[0].interm.intermTypedNode);
    }
#line 4602 "MachineIndependent/glslang_tab.cpp"
    break;

  case 4: /* primary_expression: LEFT_PAREN expression RIGHT_PAREN  */
#line 380 "MachineIndependent/glslang.y"
                                        {
        (yyval.interm.intermTypedNode) = (yyvsp[-1].interm.intermTypedNode);
        if ((yyval.interm.intermTypedNode)->getAsConstantUnion())
            (yyval.interm.intermTypedNode)->getAsConstantUnion()->setExpression();
    }
#line 4612 "MachineIndependent/glslang_tab.cpp"
    break;

  case 5: /* primary_expression: FLOATCONSTANT  */
#line 385 "MachineIndependent/glslang.y"
                    {
        (yyval.interm.intermTypedNode) = parseContext.intermediate.addConstantUnion((yyvsp[0].lex).d, EbtFloat, (yyvsp[0].lex).loc, true);
    }
#line 4620 "MachineIndependent/glslang_tab.cpp"
    break;

  case 6: /* primary_expression: INTCONSTANT  */
#line 388 "MachineIndependent/glslang.y"
                  {
        (yyval.interm.intermTypedNode) = parseContext.intermediate.addConstantUnion((yyvsp[0].lex).i, (yyvsp[0].lex).loc, true);
    }
#line 4628 "MachineIndependent/glslang_tab.cpp"
    break;

  case 7: /* primary_expression: UINTCONSTANT  */
#line 391 "MachineIndependent/glslang.y"
                   {
        parseContext.fullIntegerCheck((yyvsp[0].lex).loc, "unsigned literal");
        (yyval.interm.intermTypedNode) = parseContext.intermediate.addConstantUnion((yyvsp[0].lex).u, (yyvsp[0].lex).loc, true);
    }
#line 4637 "MachineIndependent/glslang_tab.cpp"
    break;

  case 8: /* primary_expression: BOOLCONSTANT  */
#line 395 "MachineIndependent/glslang.y"
                   {
        (yyval.interm.intermTypedNode) = parseContext.intermediate.addConstantUnion((yyvsp[0].lex).b, (yyvsp[0].lex).loc, true);
    }
#line 4645 "MachineIndependent/glslang_tab.cpp"
    break;

  case 9: /* primary_expression: STRING_LITERAL  */
#line 399 "MachineIndependent/glslang.y"
                     {
        (yyval.interm.intermTypedNode) = parseContext.intermediate.addConstantUnion((yyvsp[0].lex).string, (yyvsp[0].lex).loc, true);
    }
#line 4653 "MachineIndependent/glslang_tab.cpp"
    break;

  case 10: /* primary_expression: INT32CONSTANT  */
#line 402 "MachineIndependent/glslang.y"
                    {
        parseContext.explicitInt32Check((yyvsp[0].lex).loc, "32-bit signed literal");
        (yyval.interm.intermTypedNode) = parseContext.intermediate.addConstantUnion((yyvsp[0].lex).i, (yyvsp[0].lex).loc, true);
    }
#line 4662 "MachineIndependent/glslang_tab.cpp"
    break;

  case 11: /* primary_expression: UINT32CONSTANT  */
#line 406 "MachineIndependent/glslang.y"
                     {
        parseContext.explicitInt32Check((yyvsp[0].lex).loc, "32-bit signed literal");
        (yyval.interm.intermTypedNode) = parseContext.intermediate.addConstantUnion((yyvsp[0].lex).u, (yyvsp[0].lex).loc, true);
    }
#line 4671 "MachineIndependent/glslang_tab.cpp"
    break;

  case 12: /* primary_expression: INT64CONSTANT  */
#line 410 "MachineIndependent/glslang.y"
                    {
        parseContext.int64Check((yyvsp[0].lex).loc, "64-bit integer literal");
        (yyval.interm.intermTypedNode) = parseContext.intermediate.addConstantUnion((yyvsp[0].lex).i64, (yyvsp[0].lex).loc, true);
    }
#line 4680 "MachineIndependent/glslang_tab.cpp"
    break;

  case 13: /* primary_expression: UINT64CONSTANT  */
#line 414 "MachineIndependent/glslang.y"
                     {
        parseContext.int64Check((yyvsp[0].lex).loc, "64-bit unsigned integer literal");
        (yyval.interm.intermTypedNode) = parseContext.intermediate.addConstantUnion((yyvsp[0].lex).u64, (yyvsp[0].lex).loc, true);
    }
#line 4689 "MachineIndependent/glslang_tab.cpp"
    break;

  case 14: /* primary_expression: INT16CONSTANT  */
#line 418 "MachineIndependent/glslang.y"
                    {
        parseContext.explicitInt16Check((yyvsp[0].lex).loc, "16-bit integer literal");
        (yyval.interm.intermTypedNode) = parseContext.intermediate.addConstantUnion((short)(yyvsp[0].lex).i, (yyvsp[0].lex).loc, true);
    }
#line 4698 "MachineIndependent/glslang_tab.cpp"
    break;

  case 15: /* primary_expression: UINT16CONSTANT  */
#line 422 "MachineIndependent/glslang.y"
                     {
        parseContext.explicitInt16Check((yyvsp[0].lex).loc, "16-bit unsigned integer literal");
        (yyval.interm.intermTypedNode) = parseContext.intermediate.addConstantUnion((unsigned short)(yyvsp[0].lex).u, (yyvsp[0].lex).loc, true);
    }
#line 4707 "MachineIndependent/glslang_tab.cpp"
    break;

  case 16: /* primary_expression: DOUBLECONSTANT  */
#line 426 "MachineIndependent/glslang.y"
                     {
        parseContext.requireProfile((yyvsp[0].lex).loc, ECoreProfile | ECompatibilityProfile, "double literal");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck((yyvsp[0].lex).loc, "double literal");
        (yyval.interm.intermTypedNode) = parseContext.intermediate.addConstantUnion((yyvsp[0].lex).d, EbtDouble, (yyvsp[0].lex).loc, true);
    }
#line 4718 "MachineIndependent/glslang_tab.cpp"
    break;

  case 17: /* primary_expression: FLOAT16CONSTANT  */
#line 432 "MachineIndependent/glslang.y"
                      {
        parseContext.float16Check((yyvsp[0].lex).loc, "half float literal");
        (yyval.interm.intermTypedNode) = parseContext.intermediate.addConstantUnion((yyvsp[0].lex).d, EbtFloat16, (yyvsp[0].lex).loc, true);
    }
#line 4727 "MachineIndependent/glslang_tab.cpp"
    break;

  case 18: /* postfix_expression: primary_expression  */
#line 440 "MachineIndependent/glslang.y"
                         {
        (yyval.interm.intermTypedNode) = (yyvsp[0].interm.intermTypedNode);
    }
#line 4735 "MachineIndependent/glslang_tab.cpp"
    break;

  case 19: /* postfix_expression: postfix_expression LEFT_BRACKET integer_expression RIGHT_BRACKET  */
#line 443 "MachineIndependent/glslang.y"
                                                                       {
        (yyval.interm.intermTypedNode) = parseContext.handleBracketDereference((yyvsp[-2].lex).loc, (yyvsp[-3].interm.intermTypedNode), (yyvsp[-1].interm.intermTypedNode));
    }
#line 4743 "MachineIndependent/glslang_tab.cpp"
    break;

  case 20: /* postfix_expression: function_call  */
#line 446 "MachineIndependent/glslang.y"
                    {
        (yyval.interm.intermTypedNode) = (yyvsp[0].interm.intermTypedNode);
    }
#line 4751 "MachineIndependent/glslang_tab.cpp"
    break;

  case 21: /* postfix_expression: postfix_expression DOT IDENTIFIER  */
#line 449 "MachineIndependent/glslang.y"
                                        {
        (yyval.interm.intermTypedNode) = parseContext.handleDotDereference((yyvsp[0].lex).loc, (yyvsp[-2].interm.intermTypedNode), *(yyvsp[0].lex).string);
    }
#line 4759 "MachineIndependent/glslang_tab.cpp"
    break;

  case 22: /* postfix_expression: postfix_expression INC_OP  */
#line 452 "MachineIndependent/glslang.y"
                                {
        parseContext.variableCheck((yyvsp[-1].interm.intermTypedNode));
        parseContext.lValueErrorCheck((yyvsp[0].lex).loc, "++", (yyvsp[-1].interm.intermTypedNode));
        (yyval.interm.intermTypedNode) = parseContext.handleUnaryMath((yyvsp[0].lex).loc, "++", EOpPostIncrement, (yyvsp[-1].interm.intermTypedNode));
    }
#line 4769 "MachineIndependent/glslang_tab.cpp"
    break;

  case 23: /* postfix_expression: postfix_expression DEC_OP  */
#line 457 "MachineIndependent/glslang.y"
                                {
        parseContext.variableCheck((yyvsp[-1].interm.intermTypedNode));
        parseContext.lValueErrorCheck((yyvsp[0].lex).loc, "--", (yyvsp[-1].interm.intermTypedNode));
        (yyval.interm.intermTypedNode) = parseContext.handleUnaryMath((yyvsp[0].lex).loc, "--", EOpPostDecrement, (yyvsp[-1].interm.intermTypedNode));
    }
#line 4779 "MachineIndependent/glslang_tab.cpp"
    break;

  case 24: /* integer_expression: expression  */
#line 465 "MachineIndependent/glslang.y"
                 {
        parseContext.integerCheck((yyvsp[0].interm.intermTypedNode), "[]");
        (yyval.interm.intermTypedNode) = (yyvsp[0].interm.intermTypedNode);
    }
#line 4788 "MachineIndependent/glslang_tab.cpp"
    break;

  case 25: /* function_call: function_call_or_method  */
#line 472 "MachineIndependent/glslang.y"
                              {
        (yyval.interm.intermTypedNode) = parseContext.handleFunctionCall((yyvsp[0].interm).loc, (yyvsp[0].interm).function, (yyvsp[0].interm).intermNode);
        delete (yyvsp[0].interm).function;
    }
#line 4797 "MachineIndependent/glslang_tab.cpp"
    break;

  case 26: /* function_call_or_method: function_call_generic  */
#line 479 "MachineIndependent/glslang.y"
                            {
        (yyval.interm) = (yyvsp[0].interm);
    }
#line 4805 "MachineIndependent/glslang_tab.cpp"
    break;

  case 27: /* function_call_generic: function_call_header_with_parameters RIGHT_PAREN  */
#line 485 "MachineIndependent/glslang.y"
                                                       {
        (yyval.interm) = (yyvsp[-1].interm);
        (yyval.interm).loc = (yyvsp[0].lex).loc;
    }
#line 4814 "MachineIndependent/glslang_tab.cpp"
    break;

  case 28: /* function_call_generic: function_call_header_no_parameters RIGHT_PAREN  */
#line 489 "MachineIndependent/glslang.y"
                                                     {
        (yyval.interm) = (yyvsp[-1].interm);
        (yyval.interm).loc = (yyvsp[0].lex).loc;
    }
#line 4823 "MachineIndependent/glslang_tab.cpp"
    break;

  case 29: /* function_call_header_no_parameters: function_call_header VOID  */
#line 496 "MachineIndependent/glslang.y"
                                {
        (yyval.interm) = (yyvsp[-1].interm);
    }
#line 4831 "MachineIndependent/glslang_tab.cpp"
    break;

  case 30: /* function_call_header_no_parameters: function_call_header  */
#line 499 "MachineIndependent/glslang.y"
                           {
        (yyval.interm) = (yyvsp[0].interm);
    }
#line 4839 "MachineIndependent/glslang_tab.cpp"
    break;

  case 31: /* function_call_header_with_parameters: function_call_header assignment_expression  */
#line 505 "MachineIndependent/glslang.y"
                                                 {
        TParameter param = { 0, new TType };
        param.type->shallowCopy((yyvsp[0].interm.intermTypedNode)->getType());
        (yyvsp[-1].interm).function->addParameter(param);
        (yyval.interm).function = (yyvsp[-1].interm).function;
        (yyval.interm).intermNode = (yyvsp[0].interm.intermTypedNode);
    }
#line 4851 "MachineIndependent/glslang_tab.cpp"
    break;

  case 32: /* function_call_header_with_parameters: function_call_header_with_parameters COMMA assignment_expression  */
#line 512 "MachineIndependent/glslang.y"
                                                                       {
        TParameter param = { 0, new TType };
        param.type->shallowCopy((yyvsp[0].interm.intermTypedNode)->getType());
        (yyvsp[-2].interm).function->addParameter(param);
        (yyval.interm).function = (yyvsp[-2].interm).function;
        (yyval.interm).intermNode = parseContext.intermediate.growAggregate((yyvsp[-2].interm).intermNode, (yyvsp[0].interm.intermTypedNode), (yyvsp[-1].lex).loc);
    }
#line 4863 "MachineIndependent/glslang_tab.cpp"
    break;

  case 33: /* function_call_header: function_identifier LEFT_PAREN  */
#line 522 "MachineIndependent/glslang.y"
                                     {
        (yyval.interm) = (yyvsp[-1].interm);
    }
#line 4871 "MachineIndependent/glslang_tab.cpp"
    break;

  case 34: /* function_identifier: type_specifier  */
#line 530 "MachineIndependent/glslang.y"
                     {
        // Constructor
        (yyval.interm).intermNode = 0;
        (yyval.interm).function = parseContext.handleConstructorCall((yyvsp[0].interm.type).loc, (yyvsp[0].interm.type));
    }
#line 4881 "MachineIndependent/glslang_tab.cpp"
    break;

  case 35: /* function_identifier: postfix_expression  */
#line 535 "MachineIndependent/glslang.y"
                         {
        //
        // Should be a method or subroutine call, but we haven't recognized the arguments yet.
        //
        (yyval.interm).function = 0;
        (yyval.interm).intermNode = 0;

        TIntermMethod* method = (yyvsp[0].interm.intermTypedNode)->getAsMethodNode();
        if (method) {
            (yyval.interm).function = new TFunction(&method->getMethodName(), TType(EbtInt), EOpArrayLength);
            (yyval.interm).intermNode = method->getObject();
        } else {
            TIntermSymbol* symbol = (yyvsp[0].interm.intermTypedNode)->getAsSymbolNode();
            if (symbol) {
                parseContext.reservedErrorCheck(symbol->getLoc(), symbol->getName());
                TFunction *function = new TFunction(&symbol->getName(), TType(EbtVoid));
                (yyval.interm).function = function;
            } else
                parseContext.error((yyvsp[0].interm.intermTypedNode)->getLoc(), "function call, method, or subroutine call expected", "", "");
        }

        if ((yyval.interm).function == 0) {
            // error recover
            TString* empty = NewPoolTString("");
            (yyval.interm).function = new TFunction(empty, TType(EbtVoid), EOpNull);
        }
    }
#line 4913 "MachineIndependent/glslang_tab.cpp"
    break;

  case 36: /* function_identifier: non_uniform_qualifier  */
#line 563 "MachineIndependent/glslang.y"
                            {
        // Constructor
        (yyval.interm).intermNode = 0;
        (yyval.interm).function = parseContext.handleConstructorCall((yyvsp[0].interm.type).loc, (yyvsp[0].interm.type));
    }
#line 4923 "MachineIndependent/glslang_tab.cpp"
    break;

  case 37: /* unary_expression: postfix_expression  */
#line 572 "MachineIndependent/glslang.y"
                         {
        parseContext.variableCheck((yyvsp[0].interm.intermTypedNode));
        (yyval.interm.intermTypedNode) = (yyvsp[0].interm.intermTypedNode);
        if (TIntermMethod* method = (yyvsp[0].interm.intermTypedNode)->getAsMethodNode())
            parseContext.error((yyvsp[0].interm.intermTypedNode)->getLoc(), "incomplete method syntax", method->getMethodName().c_str(), "");
    }
#line 4934 "MachineIndependent/glslang_tab.cpp"
    break;

  case 38: /* unary_expression: INC_OP unary_expression  */
#line 578 "MachineIndependent/glslang.y"
                              {
        parseContext.lValueErrorCheck((yyvsp[-1].lex).loc, "++", (yyvsp[0].interm.intermTypedNode));
        (yyval.interm.intermTypedNode) = parseContext.handleUnaryMath((yyvsp[-1].lex).loc, "++", EOpPreIncrement, (yyvsp[0].interm.intermTypedNode));
    }
#line 4943 "MachineIndependent/glslang_tab.cpp"
    break;

  case 39: /* unary_expression: DEC_OP unary_expression  */
#line 582 "MachineIndependent/glslang.y"
                              {
        parseContext.lValueErrorCheck((yyvsp[-1].lex).loc, "--", (yyvsp[0].interm.intermTypedNode));
        (yyval.interm.intermTypedNode) = parseContext.handleUnaryMath((yyvsp[-1].lex).loc, "--", EOpPreDecrement, (yyvsp[0].interm.intermTypedNode));
    }
#line 4952 "MachineIndependent/glslang_tab.cpp"
    break;

  case 40: /* unary_expression: unary_operator unary_expression  */
#line 586 "MachineIndependent/glslang.y"
                                      {
        if ((yyvsp[-1].interm).op != EOpNull) {
            char errorOp[2] = {0, 0};
            switch((yyvsp[-1].interm).op) {
            case EOpNegative:   errorOp[0] = '-'; break;
            case EOpLogicalNot: errorOp[0] = '!'; break;
            case EOpBitwiseNot: errorOp[0] = '~'; break;
            default: break; // some compilers want this
            }
            (yyval.interm.intermTypedNode) = parseContext.handleUnaryMath((yyvsp[-1].interm).loc, errorOp, (yyvsp[-1].interm).op, (yyvsp[0].interm.intermTypedNode));
        } else {
            (yyval.interm.intermTypedNode) = (yyvsp[0].interm.intermTypedNode);
            if ((yyval.interm.intermTypedNode)->getAsConstantUnion())
                (yyval.interm.intermTypedNode)->getAsConstantUnion()->setExpression();
        }
    }
#line 4973 "MachineIndependent/glslang_tab.cpp"
    break;

  case 41: /* unary_operator: PLUS  */
#line 606 "MachineIndependent/glslang.y"
            { (yyval.interm).loc = (yyvsp[0].lex).loc; (yyval.interm).op = EOpNull; }
#line 4979 "MachineIndependent/glslang_tab.cpp"
    break;

  case 42: /* unary_operator: DASH  */
#line 607 "MachineIndependent/glslang.y"
            { (yyval.interm).loc = (yyvsp[0].lex).loc; (yyval.interm).op = EOpNegative; }
#line 4985 "MachineIndependent/glslang_tab.cpp"
    break;

  case 43: /* unary_operator: BANG  */
#line 608 "MachineIndependent/glslang.y"
            { (yyval.interm).loc = (yyvsp[0].lex).loc; (yyval.interm).op = EOpLogicalNot; }
#line 4991 "MachineIndependent/glslang_tab.cpp"
    break;

  case 44: /* unary_operator: TILDE  */
#line 609 "MachineIndependent/glslang.y"
            { (yyval.interm).loc = (yyvsp[0].lex).loc; (yyval.interm).op = EOpBitwiseNot;
              parseContext.fullIntegerCheck((yyvsp[0].lex).loc, "bitwise not"); }
#line 4998 "MachineIndependent/glslang_tab.cpp"
    break;

  case 45: /* multiplicative_expression: unary_expression  */
#line 615 "MachineIndependent/glslang.y"
                       { (yyval.interm.intermTypedNode) = (yyvsp[0].interm.intermTypedNode); }
#line 5004 "MachineIndependent/glslang_tab.cpp"
    break;

  case 46: /* multiplicative_expression: multiplicative_expression STAR unary_expression  */
#line 616 "MachineIndependent/glslang.y"
                                                      {
        (yyval.interm.intermTypedNode) = parseContext.handleBinaryMath((yyvsp[-1].lex).loc, "*", EOpMul, (yyvsp[-2].interm.intermTypedNode), (yyvsp[0].interm.intermTypedNode));
        if ((yyval.interm.intermTypedNode) == 0)
            (yyval.interm.intermTypedNode) = (yyvsp[-2].interm.intermTypedNode);
    }
#line 5014 "MachineIndependent/glslang_tab.cpp"
    break;

  case 47: /* multiplicative_expression: multiplicative_expression SLASH unary_expression  */
#line 621 "MachineIndependent/glslang.y"
                                                       {
        (yyval.interm.intermTypedNode) = parseContext.handleBinaryMath((yyvsp[-1].lex).loc, "/", EOpDiv, (yyvsp[-2].interm.intermTypedNode), (yyvsp[0].interm.intermTypedNode));
        if ((yyval.interm.intermTypedNode) == 0)
            (yyval.interm.intermTypedNode) = (yyvsp[-2].interm.intermTypedNode);
    }
#line 5024 "MachineIndependent/glslang_tab.cpp"
    break;

  case 48: /* multiplicative_expression: multiplicative_expression PERCENT unary_expression  */
#line 626 "MachineIndependent/glslang.y"
                                                         {
        parseContext.fullIntegerCheck((yyvsp[-1].lex).loc, "%");
        (yyval.interm.intermTypedNode) = parseContext.handleBinaryMath((yyvsp[-1].lex).loc, "%", EOpMod, (yyvsp[-2].interm.intermTypedNode), (yyvsp[0].interm.intermTypedNode));
        if ((yyval.interm.intermTypedNode) == 0)
            (yyval.interm.intermTypedNode) = (yyvsp[-2].interm.intermTypedNode);
    }
#line 5035 "MachineIndependent/glslang_tab.cpp"
    break;

  case 49: /* additive_expression: multiplicative_expression  */
#line 635 "MachineIndependent/glslang.y"
                                { (yyval.interm.intermTypedNode) = (yyvsp[0].interm.intermTypedNode); }
#line 5041 "MachineIndependent/glslang_tab.cpp"
    break;

  case 50: /* additive_expression: additive_expression PLUS multiplicative_expression  */
#line 636 "MachineIndependent/glslang.y"
                                                         {
        (yyval.interm.intermTypedNode) = parseContext.handleBinaryMath((yyvsp[-1].lex).loc, "+", EOpAdd, (yyvsp[-2].interm.intermTypedNode), (yyvsp[0].interm.intermTypedNode));
        if ((yyval.interm.intermTypedNode) == 0)
            (yyval.interm.intermTypedNode) = (yyvsp[-2].interm.intermTypedNode);
    }
#line 5051 "MachineIndependent/glslang_tab.cpp"
    break;

  case 51: /* additive_expression: additive_expression DASH multiplicative_expression  */
#line 641 "MachineIndependent/glslang.y"
                                                         {
        (yyval.interm.intermTypedNode) = parseContext.handleBinaryMath((yyvsp[-1].lex).loc, "-", EOpSub, (yyvsp[-2].interm.intermTypedNode), (yyvsp[0].interm.intermTypedNode));
        if ((yyval.interm.intermTypedNode) == 0)
            (yyval.interm.intermTypedNode) = (yyvsp[-2].interm.intermTypedNode);
    }
#line 5061 "MachineIndependent/glslang_tab.cpp"
    break;

  case 52: /* shift_expression: additive_expression  */
#line 649 "MachineIndependent/glslang.y"
                          { (yyval.interm.intermTypedNode) = (yyvsp[0].interm.intermTypedNode); }
#line 5067 "MachineIndependent/glslang_tab.cpp"
    break;

  case 53: /* shift_expression: shift_expression LEFT_OP additive_expression  */
#line 650 "MachineIndependent/glslang.y"
                                                   {
        parseContext.fullIntegerCheck((yyvsp[-1].lex).loc, "bit shift left");
        (yyval.interm.intermTypedNode) = parseContext.handleBinaryMath((yyvsp[-1].lex).loc, "<<", EOpLeftShift, (yyvsp[-2].interm.intermTypedNode), (yyvsp[0].interm.intermTypedNode));
        if ((yyval.interm.intermTypedNode) == 0)
            (yyval.interm.intermTypedNode) = (yyvsp[-2].interm.intermTypedNode);
    }
#line 5078 "MachineIndependent/glslang_tab.cpp"
    break;

  case 54: /* shift_expression: shift_expression RIGHT_OP additive_expression  */
#line 656 "MachineIndependent/glslang.y"
                                                    {
        parseContext.fullIntegerCheck((yyvsp[-1].lex).loc, "bit shift right");
        (yyval.interm.intermTypedNode) = parseContext.handleBinaryMath((yyvsp[-1].lex).loc, ">>", EOpRightShift, (yyvsp[-2].interm.intermTypedNode), (yyvsp[0].interm.intermTypedNode));
        if ((yyval.interm.intermTypedNode) == 0)
            (yyval.interm.intermTypedNode) = (yyvsp[-2].interm.intermTypedNode);
    }
#line 5089 "MachineIndependent/glslang_tab.cpp"
    break;

  case 55: /* relational_expression: shift_expression  */
#line 665 "MachineIndependent/glslang.y"
                       { (yyval.interm.intermTypedNode) = (yyvsp[0].interm.intermTypedNode); }
#line 5095 "MachineIndependent/glslang_tab.cpp"
    break;

  case 56: /* relational_expression: relational_expression LEFT_ANGLE shift_expression  */
#line 666 "MachineIndependent/glslang.y"
                                                        {
        (yyval.interm.intermTypedNode) = parseContext.handleBinaryMath((yyvsp[-1].lex).loc, "<", EOpLessThan, (yyvsp[-2].interm.intermTypedNode), (yyvsp[0].interm.intermTypedNode));
        if ((yyval.interm.intermTypedNode) == 0)
            (yyval.interm.intermTypedNode) = parseContext.intermediate.addConstantUnion(false, (yyvsp[-1].lex).loc);
    }
#line 5105 "MachineIndependent/glslang_tab.cpp"
    break;

  case 57: /* relational_expression: relational_expression RIGHT_ANGLE shift_expression  */
#line 671 "MachineIndependent/glslang.y"
                                                          {
        (yyval.interm.intermTypedNode) = parseContext.handleBinaryMath((yyvsp[-1].lex).loc, ">", EOpGreaterThan, (yyvsp[-2].interm.intermTypedNode), (yyvsp[0].interm.intermTypedNode));
        if ((yyval.interm.intermTypedNode) == 0)
            (yyval.interm.intermTypedNode) = parseContext.intermediate.addConstantUnion(false, (yyvsp[-1].lex).loc);
    }
#line 5115 "MachineIndependent/glslang_tab.cpp"
    break;

  case 58: /* relational_expression: relational_expression LE_OP shift_expression  */
#line 676 "MachineIndependent/glslang.y"
                                                    {
        (yyval.interm.intermTypedNode) = parseContext.handleBinaryMath((yyvsp[-1].lex).loc, "<=", EOpLessThanEqual, (yyvsp[-2].interm.intermTypedNode), (yyvsp[0].interm.intermTypedNode));
        if ((yyval.interm.intermTypedNode) == 0)
            (yyval.interm.intermTypedNode) = parseContext.intermediate.addConstantUnion(false, (yyvsp[-1].lex).loc);
    }
#line 5125 "MachineIndependent/glslang_tab.cpp"
    break;

  case 59: /* relational_expression: relational_expression GE_OP shift_expression  */
#line 681 "MachineIndependent/glslang.y"
                                                    {
        (yyval.interm.intermTypedNode) = parseContext.handleBinaryMath((yyvsp[-1].lex).loc, ">=", EOpGreaterThanEqual, (yyvsp[-2].interm.intermTypedNode), (yyvsp[0].interm.intermTypedNode));
        if ((yyval.interm.intermTypedNode) == 0)
            (yyval.interm.intermTypedNode) = parseContext.intermediate.addConstantUnion(false, (yyvsp[-1].lex).loc);
    }
#line 5135 "MachineIndependent/glslang_tab.cpp"
    break;

  case 60: /* equality_expression: relational_expression  */
#line 689 "MachineIndependent/glslang.y"
                            { (yyval.interm.intermTypedNode) = (yyvsp[0].interm.intermTypedNode); }
#line 5141 "MachineIndependent/glslang_tab.cpp"
    break;

  case 61: /* equality_expression: equality_expression EQ_OP relational_expression  */
#line 690 "MachineIndependent/glslang.y"
                                                       {
        parseContext.arrayObjectCheck((yyvsp[-1].lex).loc, (yyvsp[-2].interm.intermTypedNode)->getType(), "array comparison");
        parseContext.opaqueCheck((yyvsp[-1].lex).loc, (yyvsp[-2].interm.intermTypedNode)->getType(), "==");
        parseContext.specializationCheck((yyvsp[-1].lex).loc, (yyvsp[-2].interm.intermTypedNode)->getType(), "==");
        parseContext.referenceCheck((yyvsp[-1].lex).loc, (yyvsp[-2].interm.intermTypedNode)->getType(), "==");
        (yyval.interm.intermTypedNode) = parseContext.handleBinaryMath((yyvsp[-1].lex).loc, "==", EOpEqual, (yyvsp[-2].interm.intermTypedNode), (yyvsp[0].interm.intermTypedNode));
        if ((yyval.interm.intermTypedNode) == 0)
            (yyval.interm.intermTypedNode) = parseContext.intermediate.addConstantUnion(false, (yyvsp[-1].lex).loc);
    }
#line 5155 "MachineIndependent/glslang_tab.cpp"
    break;

  case 62: /* equality_expression: equality_expression NE_OP relational_expression  */
#line 699 "MachineIndependent/glslang.y"
                                                      {
        parseContext.arrayObjectCheck((yyvsp[-1].lex).loc, (yyvsp[-2].interm.intermTypedNode)->getType(), "array comparison");
        parseContext.opaqueCheck((yyvsp[-1].lex).loc, (yyvsp[-2].interm.intermTypedNode)->getType(), "!=");
        parseContext.specializationCheck((yyvsp[-1].lex).loc, (yyvsp[-2].interm.intermTypedNode)->getType(), "!=");
        parseContext.referenceCheck((yyvsp[-1].lex).loc, (yyvsp[-2].interm.intermTypedNode)->getType(), "!=");
        (yyval.interm.intermTypedNode) = parseContext.handleBinaryMath((yyvsp[-1].lex).loc, "!=", EOpNotEqual, (yyvsp[-2].interm.intermTypedNode), (yyvsp[0].interm.intermTypedNode));
        if ((yyval.interm.intermTypedNode) == 0)
            (yyval.interm.intermTypedNode) = parseContext.intermediate.addConstantUnion(false, (yyvsp[-1].lex).loc);
    }
#line 5169 "MachineIndependent/glslang_tab.cpp"
    break;

  case 63: /* and_expression: equality_expression  */
#line 711 "MachineIndependent/glslang.y"
                          { (yyval.interm.intermTypedNode) = (yyvsp[0].interm.intermTypedNode); }
#line 5175 "MachineIndependent/glslang_tab.cpp"
    break;

  case 64: /* and_expression: and_expression AMPERSAND equality_expression  */
#line 712 "MachineIndependent/glslang.y"
                                                   {
        parseContext.fullIntegerCheck((yyvsp[-1].lex).loc, "bitwise and");
        (yyval.interm.intermTypedNode) = parseContext.handleBinaryMath((yyvsp[-1].lex).loc, "&", EOpAnd, (yyvsp[-2].interm.intermTypedNode), (yyvsp[0].interm.intermTypedNode));
        if ((yyval.interm.intermTypedNode) == 0)
            (yyval.interm.intermTypedNode) = (yyvsp[-2].interm.intermTypedNode);
    }
#line 5186 "MachineIndependent/glslang_tab.cpp"
    break;

  case 65: /* exclusive_or_expression: and_expression  */
#line 721 "MachineIndependent/glslang.y"
                     { (yyval.interm.intermTypedNode) = (yyvsp[0].interm.intermTypedNode); }
#line 5192 "MachineIndependent/glslang_tab.cpp"
    break;

  case 66: /* exclusive_or_expression: exclusive_or_expression CARET and_expression  */
#line 722 "MachineIndependent/glslang.y"
                                                   {
        parseContext.fullIntegerCheck((yyvsp[-1].lex).loc, "bitwise exclusive or");
        (yyval.interm.intermTypedNode) = parseContext.handleBinaryMath((yyvsp[-1].lex).loc, "^", EOpExclusiveOr, (yyvsp[-2].interm.intermTypedNode), (yyvsp[0].interm.intermTypedNode));
        if ((yyval.interm.intermTypedNode) == 0)
            (yyval.interm.intermTypedNode) = (yyvsp[-2].interm.intermTypedNode);
    }
#line 5203 "MachineIndependent/glslang_tab.cpp"
    break;

  case 67: /* inclusive_or_expression: exclusive_or_expression  */
#line 731 "MachineIndependent/glslang.y"
                              { (yyval.interm.intermTypedNode) = (yyvsp[0].interm.intermTypedNode); }
#line 5209 "MachineIndependent/glslang_tab.cpp"
    break;

  case 68: /* inclusive_or_expression: inclusive_or_expression VERTICAL_BAR exclusive_or_expression  */
#line 732 "MachineIndependent/glslang.y"
                                                                   {
        parseContext.fullIntegerCheck((yyvsp[-1].lex).loc, "bitwise inclusive or");
        (yyval.interm.intermTypedNode) = parseContext.handleBinaryMath((yyvsp[-1].lex).loc, "|", EOpInclusiveOr, (yyvsp[-2].interm.intermTypedNode), (yyvsp[0].interm.intermTypedNode));
        if ((yyval.interm.intermTypedNode) == 0)
            (yyval.interm.intermTypedNode) = (yyvsp[-2].interm.intermTypedNode);
    }
#line 5220 "MachineIndependent/glslang_tab.cpp"
    break;

  case 69: /* logical_and_expression: inclusive_or_expression  */
#line 741 "MachineIndependent/glslang.y"
                              { (yyval.interm.intermTypedNode) = (yyvsp[0].interm.intermTypedNode); }
#line 5226 "MachineIndependent/glslang_tab.cpp"
    break;

  case 70: /* logical_and_expression: logical_and_expression AND_OP inclusive_or_expression  */
#line 742 "MachineIndependent/glslang.y"
                                                            {
        (yyval.interm.intermTypedNode) = parseContext.handleBinaryMath((yyvsp[-1].lex).loc, "&&", EOpLogicalAnd, (yyvsp[-2].interm.intermTypedNode), (yyvsp[0].interm.intermTypedNode));
        if ((yyval.interm.intermTypedNode) == 0)
            (yyval.interm.intermTypedNode) = parseContext.intermediate.addConstantUnion(false, (yyvsp[-1].lex).loc);
    }
#line 5236 "MachineIndependent/glslang_tab.cpp"
    break;

  case 71: /* logical_xor_expression: logical_and_expression  */
#line 750 "MachineIndependent/glslang.y"
                             { (yyval.interm.intermTypedNode) = (yyvsp[0].interm.intermTypedNode); }
#line 5242 "MachineIndependent/glslang_tab.cpp"
    break;

  case 72: /* logical_xor_expression: logical_xor_expression XOR_OP logical_and_expression  */
#line 751 "MachineIndependent/glslang.y"
                                                            {
        (yyval.interm.intermTypedNode) = parseContext.handleBinaryMath((yyvsp[-1].lex).loc, "^^", EOpLogicalXor, (yyvsp[-2].interm.intermTypedNode), (yyvsp[0].interm.intermTypedNode));
        if ((yyval.interm.intermTypedNode) == 0)
            (yyval.interm.intermTypedNode) = parseContext.intermediate.addConstantUnion(false, (yyvsp[-1].lex).loc);
    }
#line 5252 "MachineIndependent/glslang_tab.cpp"
    break;

  case 73: /* logical_or_expression: logical_xor_expression  */
#line 759 "MachineIndependent/glslang.y"
                             { (yyval.interm.intermTypedNode) = (yyvsp[0].interm.intermTypedNode); }
#line 5258 "MachineIndependent/glslang_tab.cpp"
    break;

  case 74: /* logical_or_expression: logical_or_expression OR_OP logical_xor_expression  */
#line 760 "MachineIndependent/glslang.y"
                                                          {
        (yyval.interm.intermTypedNode) = parseContext.handleBinaryMath((yyvsp[-1].lex).loc, "||", EOpLogicalOr, (yyvsp[-2].interm.intermTypedNode), (yyvsp[0].interm.intermTypedNode));
        if ((yyval.interm.intermTypedNode) == 0)
            (yyval.interm.intermTypedNode) = parseContext.intermediate.addConstantUnion(false, (yyvsp[-1].lex).loc);
    }
#line 5268 "MachineIndependent/glslang_tab.cpp"
    break;

  case 75: /* conditional_expression: logical_or_expression  */
#line 768 "MachineIndependent/glslang.y"
                            { (yyval.interm.intermTypedNode) = (yyvsp[0].interm.intermTypedNode); }
#line 5274 "MachineIndependent/glslang_tab.cpp"
    break;

  case 76: /* $@1: %empty  */
#line 769 "MachineIndependent/glslang.y"
                                     {
        ++parseContext.controlFlowNestingLevel;
    }
#line 5282 "MachineIndependent/glslang_tab.cpp"
    break;

  case 77: /* conditional_expression: logical_or_expression QUESTION $@1 expression COLON assignment_expression  */
#line 772 "MachineIndependent/glslang.y"
                                             {
        --parseContext.controlFlowNestingLevel;
        parseContext.boolCheck((yyvsp[-4].lex).loc, (yyvsp[-5].interm.intermTypedNode));
        parseContext.rValueErrorCheck((yyvsp[-4].lex).loc, "?", (yyvsp[-5].interm.intermTypedNode));
        parseContext.rValueErrorCheck((yyvsp[-1].lex).loc, ":", (yyvsp[-2].interm.intermTypedNode));
        parseContext.rValueErrorCheck((yyvsp[-1].lex).loc, ":", (yyvsp[0].interm.intermTypedNode));
        (yyval.interm.intermTypedNode) = parseContext.intermediate.addSelection((yyvsp[-5].interm.intermTypedNode), (yyvsp[-2].interm.intermTypedNode), (yyvsp[0].interm.intermTypedNode), (yyvsp[-4].lex).loc);
        if ((yyval.interm.intermTypedNode) == 0) {
            parseContext.binaryOpError((yyvsp[-4].lex).loc, ":", (yyvsp[-2].interm.intermTypedNode)->getCompleteString(), (yyvsp[0].interm.intermTypedNode)->getCompleteString());
            (yyval.interm.intermTypedNode) = (yyvsp[0].interm.intermTypedNode);
        }
    }
#line 5299 "MachineIndependent/glslang_tab.cpp"
    break;

  case 78: /* assignment_expression: conditional_expression  */
#line 787 "MachineIndependent/glslang.y"
                             { (yyval.interm.intermTypedNode) = (yyvsp[0].interm.intermTypedNode); }
#line 5305 "MachineIndependent/glslang_tab.cpp"
    break;

  case 79: /* assignment_expression: unary_expression assignment_operator assignment_expression  */
#line 788 "MachineIndependent/glslang.y"
                                                                 {
        parseContext.arrayObjectCheck((yyvsp[-1].interm).loc, (yyvsp[-2].interm.intermTypedNode)->getType(), "array assignment");
        parseContext.opaqueCheck((yyvsp[-1].interm).loc, (yyvsp[-2].interm.intermTypedNode)->getType(), "=");
        parseContext.storage16BitAssignmentCheck((yyvsp[-1].interm).loc, (yyvsp[-2].interm.intermTypedNode)->getType(), "=");
        parseContext.specializationCheck((yyvsp[-1].interm).loc, (yyvsp[-2].interm.intermTypedNode)->getType(), "=");
        parseContext.lValueErrorCheck((yyvsp[-1].interm).loc, "assign", (yyvsp[-2].interm.intermTypedNode));
        parseContext.rValueErrorCheck((yyvsp[-1].interm).loc, "assign", (yyvsp[0].interm.intermTypedNode));
        (yyval.interm.intermTypedNode) = parseContext.addAssign((yyvsp[-1].interm).loc, (yyvsp[-1].interm).op, (yyvsp[-2].interm.intermTypedNode), (yyvsp[0].interm.intermTypedNode));
        if ((yyval.interm.intermTypedNode) == 0) {
            parseContext.assignError((yyvsp[-1].interm).loc, "assign", (yyvsp[-2].interm.intermTypedNode)->getCompleteString(), (yyvsp[0].interm.intermTypedNode)->getCompleteString());
            (yyval.interm.intermTypedNode) = (yyvsp[-2].interm.intermTypedNode);
        }
    }
#line 5323 "MachineIndependent/glslang_tab.cpp"
    break;

  case 80: /* assignment_operator: EQUAL  */
#line 804 "MachineIndependent/glslang.y"
            {
        (yyval.interm).loc = (yyvsp[0].lex).loc;
        (yyval.interm).op = EOpAssign;
    }
#line 5332 "MachineIndependent/glslang_tab.cpp"
    break;

  case 81: /* assignment_operator: MUL_ASSIGN  */
#line 808 "MachineIndependent/glslang.y"
                 {
        (yyval.interm).loc = (yyvsp[0].lex).loc;
        (yyval.interm).op = EOpMulAssign;
    }
#line 5341 "MachineIndependent/glslang_tab.cpp"
    break;

  case 82: /* assignment_operator: DIV_ASSIGN  */
#line 812 "MachineIndependent/glslang.y"
                 {
        (yyval.interm).loc = (yyvsp[0].lex).loc;
        (yyval.interm).op = EOpDivAssign;
    }
#line 5350 "MachineIndependent/glslang_tab.cpp"
    break;

  case 83: /* assignment_operator: MOD_ASSIGN  */
#line 816 "MachineIndependent/glslang.y"
                 {
        parseContext.fullIntegerCheck((yyvsp[0].lex).loc, "%=");
        (yyval.interm).loc = (yyvsp[0].lex).loc;
        (yyval.interm).op = EOpModAssign;
    }
#line 5360 "MachineIndependent/glslang_tab.cpp"
    break;

  case 84: /* assignment_operator: ADD_ASSIGN  */
#line 821 "MachineIndependent/glslang.y"
                 {
        (yyval.interm).loc = (yyvsp[0].lex).loc;
        (yyval.interm).op = EOpAddAssign;
    }
#line 5369 "MachineIndependent/glslang_tab.cpp"
    break;

  case 85: /* assignment_operator: SUB_ASSIGN  */
#line 825 "MachineIndependent/glslang.y"
                 {
        (yyval.interm).loc = (yyvsp[0].lex).loc;
        (yyval.interm).op = EOpSubAssign;
    }
#line 5378 "MachineIndependent/glslang_tab.cpp"
    break;

  case 86: /* assignment_operator: LEFT_ASSIGN  */
#line 829 "MachineIndependent/glslang.y"
                  {
        parseContext.fullIntegerCheck((yyvsp[0].lex).loc, "bit-shift left assign");
        (yyval.interm).loc = (yyvsp[0].lex).loc; (yyval.interm).op = EOpLeftShiftAssign;
    }
#line 5387 "MachineIndependent/glslang_tab.cpp"
    break;

  case 87: /* assignment_operator: RIGHT_ASSIGN  */
#line 833 "MachineIndependent/glslang.y"
                   {
        parseContext.fullIntegerCheck((yyvsp[0].lex).loc, "bit-shift right assign");
        (yyval.interm).loc = (yyvsp[0].lex).loc; (yyval.interm).op = EOpRightShiftAssign;
    }
#line 5396 "MachineIndependent/glslang_tab.cpp"
    break;

  case 88: /* assignment_operator: AND_ASSIGN  */
#line 837 "MachineIndependent/glslang.y"
                 {
        parseContext.fullIntegerCheck((yyvsp[0].lex).loc, "bitwise-and assign");
        (yyval.interm).loc = (yyvsp[0].lex).loc; (yyval.interm).op = EOpAndAssign;
    }
#line 5405 "MachineIndependent/glslang_tab.cpp"
    break;

  case 89: /* assignment_operator: XOR_ASSIGN  */
#line 841 "MachineIndependent/glslang.y"
                 {
        parseContext.fullIntegerCheck((yyvsp[0].lex).loc, "bitwise-xor assign");
        (yyval.interm).loc = (yyvsp[0].lex).loc; (yyval.interm).op = EOpExclusiveOrAssign;
    }
#line 5414 "MachineIndependent/glslang_tab.cpp"
    break;

  case 90: /* assignment_operator: OR_ASSIGN  */
#line 845 "MachineIndependent/glslang.y"
                {
        parseContext.fullIntegerCheck((yyvsp[0].lex).loc, "bitwise-or assign");
        (yyval.interm).loc = (yyvsp[0].lex).loc; (yyval.interm).op = EOpInclusiveOrAssign;
    }
#line 5423 "MachineIndependent/glslang_tab.cpp"
    break;

  case 91: /* expression: assignment_expression  */
#line 852 "MachineIndependent/glslang.y"
                            {
        (yyval.interm.intermTypedNode) = (yyvsp[0].interm.intermTypedNode);
    }
#line 5431 "MachineIndependent/glslang_tab.cpp"
    break;

  case 92: /* expression: expression COMMA assignment_expression  */
#line 855 "MachineIndependent/glslang.y"
                                             {
        parseContext.samplerConstructorLocationCheck((yyvsp[-1].lex).loc, ",", (yyvsp[0].interm.intermTypedNode));
        (yyval.interm.intermTypedNode) = parseContext.intermediate.addComma((yyvsp[-2].interm.intermTypedNode), (yyvsp[0].interm.intermTypedNode), (yyvsp[-1].lex).loc);
        if ((yyval.interm.intermTypedNode) == 0) {
            parseContext.binaryOpError((yyvsp[-1].lex).loc, ",", (yyvsp[-2].interm.intermTypedNode)->getCompleteString(), (yyvsp[0].interm.intermTypedNode)->getCompleteString());
            (yyval.interm.intermTypedNode) = (yyvsp[0].interm.intermTypedNode);
        }
    }
#line 5444 "MachineIndependent/glslang_tab.cpp"
    break;

  case 93: /* constant_expression: conditional_expression  */
#line 866 "MachineIndependent/glslang.y"
                             {
        parseContext.constantValueCheck((yyvsp[0].interm.intermTypedNode), "");
        (yyval.interm.intermTypedNode) = (yyvsp[0].interm.intermTypedNode);
    }
#line 5453 "MachineIndependent/glslang_tab.cpp"
    break;

  case 94: /* declaration: function_prototype SEMICOLON  */
#line 873 "MachineIndependent/glslang.y"
                                   {
        parseContext.handleFunctionDeclarator((yyvsp[-1].interm).loc, *(yyvsp[-1].interm).function, true /* prototype */);
        (yyval.interm.intermNode) = 0;
        // TODO: 4.0 functionality: subroutines: make the identifier a user type for this signature
    }
#line 5463 "MachineIndependent/glslang_tab.cpp"
    break;

  case 95: /* declaration: init_declarator_list SEMICOLON  */
#line 878 "MachineIndependent/glslang.y"
                                     {
        if ((yyvsp[-1].interm).intermNode && (yyvsp[-1].interm).intermNode->getAsAggregate())
            (yyvsp[-1].interm).intermNode->getAsAggregate()->setOperator(EOpSequence);
        (yyval.interm.intermNode) = (yyvsp[-1].interm).intermNode;
    }
#line 5473 "MachineIndependent/glslang_tab.cpp"
    break;

  case 96: /* declaration: PRECISION precision_qualifier type_specifier SEMICOLON  */
#line 883 "MachineIndependent/glslang.y"
                                                             {
        parseContext.profileRequires((yyvsp[-3].lex).loc, ENoProfile, 130, 0, "precision statement");
        // lazy setting of the previous scope's defaults, has effect only the first time it is called in a particular scope
        parseContext.symbolTable.setPreviousDefaultPrecisions(&parseContext.defaultPrecision[0]);
        parseContext.setDefaultPrecision((yyvsp[-3].lex).loc, (yyvsp[-1].interm.type), (yyvsp[-2].interm.type).qualifier.precision);
        (yyval.interm.intermNode) = 0;
    }
#line 5485 "MachineIndependent/glslang_tab.cpp"
    break;

  case 97: /* declaration: block_structure SEMICOLON  */
#line 890 "MachineIndependent/glslang.y"
                                {
        parseContext.declareBlock((yyvsp[-1].interm).loc, *(yyvsp[-1].interm).typeList);
        (yyval.interm.intermNode) = 0;
    }
#line 5494 "MachineIndependent/glslang_tab.cpp"
    break;

  case 98: /* declaration: block_structure IDENTIFIER SEMICOLON  */
#line 894 "MachineIndependent/glslang.y"
                                           {
        parseContext.declareBlock((yyvsp[-2].interm).loc, *(yyvsp[-2].interm).typeList, (yyvsp[-1].lex).string);
        (yyval.interm.intermNode) = 0;
    }
#line 5503 "MachineIndependent/glslang_tab.cpp"
    break;

  case 99: /* declaration: block_structure IDENTIFIER array_specifier SEMICOLON  */
#line 898 "MachineIndependent/glslang.y"
                                                           {
        parseContext.declareBlock((yyvsp[-3].interm).loc, *(yyvsp[-3].interm).typeList, (yyvsp[-2].lex).string, (yyvsp[-1].interm).arraySizes);
        (yyval.interm.intermNode) = 0;
    }
#line 5512 "MachineIndependent/glslang_tab.cpp"
    break;

  case 100: /* declaration: type_qualifier SEMICOLON  */
#line 902 "MachineIndependent/glslang.y"
                               {
        parseContext.globalQualifierFixCheck((yyvsp[-1].interm.type).loc, (yyvsp[-1].interm.type).qualifier);
        parseContext.updateStandaloneQualifierDefaults((yyvsp[-1].interm.type).loc, (yyvsp[-1].interm.type));
        (yyval.interm.intermNode) = 0;
    }
#line 5522 "MachineIndependent/glslang_tab.cpp"
    break;

  case 101: /* declaration: type_qualifier IDENTIFIER SEMICOLON  */
#line 907 "MachineIndependent/glslang.y"
                                          {
        parseContext.checkNoShaderLayouts((yyvsp[-2].interm.type).loc, (yyvsp[-2].interm.type).shaderQualifiers);
        parseContext.addQualifierToExisting((yyvsp[-2].interm.type).loc, (yyvsp[-2].interm.type).qualifier, *(yyvsp[-1].lex).string);
        (yyval.interm.intermNode) = 0;
    }
#line 5532 "MachineIndependent/glslang_tab.cpp"
    break;

  case 102: /* declaration: type_qualifier IDENTIFIER identifier_list SEMICOLON  */
#line 912 "MachineIndependent/glslang.y"
                                                          {
        parseContext.checkNoShaderLayouts((yyvsp[-3].interm.type).loc, (yyvsp[-3].interm.type).shaderQualifiers);
        (yyvsp[-1].interm.identifierList)->push_back((yyvsp[-2].lex).string);
        parseContext.addQualifierToExisting((yyvsp[-3].interm.type).loc, (yyvsp[-3].interm.type).qualifier, *(yyvsp[-1].interm.identifierList));
        (yyval.interm.intermNode) = 0;
    }
#line 5543 "MachineIndependent/glslang_tab.cpp"
    break;

  case 103: /* $@2: %empty  */
#line 921 "MachineIndependent/glslang.y"
                                           { parseContext.nestedBlockCheck((yyvsp[-2].interm.type).loc); }
#line 5549 "MachineIndependent/glslang_tab.cpp"
    break;

  case 104: /* block_structure: type_qualifier IDENTIFIER LEFT_BRACE $@2 struct_declaration_list RIGHT_BRACE  */
#line 921 "MachineIndependent/glslang.y"
                                                                                                                          {
        --parseContext.blockNestingLevel;
        parseContext.blockName = (yyvsp[-4].lex).string;
        parseContext.globalQualifierFixCheck((yyvsp[-5].interm.type).loc, (yyvsp[-5].interm.type).qualifier);
        parseContext.checkNoShaderLayouts((yyvsp[-5].interm.type).loc, (yyvsp[-5].interm.type).shaderQualifiers);
        parseContext.currentBlockQualifier = (yyvsp[-5].interm.type).qualifier;
        (yyval.interm).loc = (yyvsp[-5].interm.type).loc;
        (yyval.interm).typeList = (yyvsp[-1].interm.typeList);
    }
#line 5563 "MachineIndependent/glslang_tab.cpp"
    break;

  case 105: /* identifier_list: COMMA IDENTIFIER  */
#line 932 "MachineIndependent/glslang.y"
                       {
        (yyval.interm.identifierList) = new TIdentifierList;
        (yyval.interm.identifierList)->push_back((yyvsp[0].lex).string);
    }
#line 5572 "MachineIndependent/glslang_tab.cpp"
    break;

  case 106: /* identifier_list: identifier_list COMMA IDENTIFIER  */
#line 936 "MachineIndependent/glslang.y"
                                       {
        (yyval.interm.identifierList) = (yyvsp[-2].interm.identifierList);
        (yyval.interm.identifierList)->push_back((yyvsp[0].lex).string);
    }
#line 5581 "MachineIndependent/glslang_tab.cpp"
    break;

  case 107: /* function_prototype: function_declarator RIGHT_PAREN  */
#line 943 "MachineIndependent/glslang.y"
                                       {
        (yyval.interm).function = (yyvsp[-1].interm.function);
        (yyval.interm).loc = (yyvsp[0].lex).loc;
    }
#line 5590 "MachineIndependent/glslang_tab.cpp"
    break;

  case 108: /* function_declarator: function_header  */
#line 950 "MachineIndependent/glslang.y"
                      {
        (yyval.interm.function) = (yyvsp[0].interm.function);
    }
#line 5598 "MachineIndependent/glslang_tab.cpp"
    break;

  case 109: /* function_declarator: function_header_with_parameters  */
#line 953 "MachineIndependent/glslang.y"
                                      {
        (yyval.interm.function) = (yyvsp[0].interm.function);
    }
#line 5606 "MachineIndependent/glslang_tab.cpp"
    break;

  case 110: /* function_header_with_parameters: function_header parameter_declaration  */
#line 960 "MachineIndependent/glslang.y"
                                            {
        // Add the parameter
        (yyval.interm.function) = (yyvsp[-1].interm.function);
        if ((yyvsp[0].interm).param.type->getBasicType() != EbtVoid)
            (yyvsp[-1].interm.function)->addParameter((yyvsp[0].interm).param);
        else
            delete (yyvsp[0].interm).param.type;
    }
#line 5619 "MachineIndependent/glslang_tab.cpp"
    break;

  case 111: /* function_header_with_parameters: function_header_with_parameters COMMA parameter_declaration  */
#line 968 "MachineIndependent/glslang.y"
                                                                  {
        //
        // Only first parameter of one-parameter functions can be void
        // The check for named parameters not being void is done in parameter_declarator
        //
        if ((yyvsp[0].interm).param.type->getBasicType() == EbtVoid) {
            //
            // This parameter > first is void
            //
            parseContext.error((yyvsp[-1].lex).loc, "cannot be an argument type except for '(void)'", "void", "");
            delete (yyvsp[0].interm).param.type;
        } else {
            // Add the parameter
            (yyval.interm.function) = (yyvsp[-2].interm.function);
            (yyvsp[-2].interm.function)->addParameter((yyvsp[0].interm).param);
        }
    }
#line 5641 "MachineIndependent/glslang_tab.cpp"
    break;

  case 112: /* function_header: fully_specified_type IDENTIFIER LEFT_PAREN  */
#line 988 "MachineIndependent/glslang.y"
                                                 {
        if ((yyvsp[-2].interm.type).qualifier.storage != EvqGlobal && (yyvsp[-2].interm.type).qualifier.storage != EvqTemporary) {
            parseContext.error((yyvsp[-1].lex).loc, "no qualifiers allowed for function return",
                               GetStorageQualifierString((yyvsp[-2].interm.type).qualifier.storage), "");
        }
        if ((yyvsp[-2].interm.type).arraySizes)
            parseContext.arraySizeRequiredCheck((yyvsp[-2].interm.type).loc, *(yyvsp[-2].interm.type).arraySizes);

        // Add the function as a prototype after parsing it (we do not support recursion)
        TFunction *function;
        TType type((yyvsp[-2].interm.type));

        // Potentially rename shader entry point function.  No-op most of the time.
        parseContext.renameShaderFunction((yyvsp[-1].lex).string);

        // Make the function
        function = new TFunction((yyvsp[-1].lex).string, type);
        (yyval.interm.function) = function;
    }
#line 5665 "MachineIndependent/glslang_tab.cpp"
    break;

  case 113: /* parameter_declarator: type_specifier IDENTIFIER  */
#line 1011 "MachineIndependent/glslang.y"
                                {
        if ((yyvsp[-1].interm.type).arraySizes) {
            parseContext.profileRequires((yyvsp[-1].interm.type).loc, ENoProfile, 120, E_GL_3DL_array_objects, "arrayed type");
            parseContext.profileRequires((yyvsp[-1].interm.type).loc, EEsProfile, 300, 0, "arrayed type");
            parseContext.arraySizeRequiredCheck((yyvsp[-1].interm.type).loc, *(yyvsp[-1].interm.type).arraySizes);
        }
        if ((yyvsp[-1].interm.type).basicType == EbtVoid) {
            parseContext.error((yyvsp[0].lex).loc, "illegal use of type 'void'", (yyvsp[0].lex).string->c_str(), "");
        }
        parseContext.reservedErrorCheck((yyvsp[0].lex).loc, *(yyvsp[0].lex).string);

        TParameter param = {(yyvsp[0].lex).string, new TType((yyvsp[-1].interm.type))};
        (yyval.interm).loc = (yyvsp[0].lex).loc;
        (yyval.interm).param = param;
    }
#line 5685 "MachineIndependent/glslang_tab.cpp"
    break;

  case 114: /* parameter_declarator: type_specifier IDENTIFIER array_specifier  */
#line 1026 "MachineIndependent/glslang.y"
                                                {
        if ((yyvsp[-2].interm.type).arraySizes) {
            parseContext.profileRequires((yyvsp[-2].interm.type).loc, ENoProfile, 120, E_GL_3DL_array_objects, "arrayed type");
            parseContext.profileRequires((yyvsp[-2].interm.type).loc, EEsProfile, 300, 0, "arrayed type");
            parseContext.arraySizeRequiredCheck((yyvsp[-2].interm.type).loc, *(yyvsp[-2].interm.type).arraySizes);
        }
        TType* type = new TType((yyvsp[-2].interm.type));
        type->transferArraySizes((yyvsp[0].interm).arraySizes);
        type->copyArrayInnerSizes((yyvsp[-2].interm.type).arraySizes);

        parseContext.arrayOfArrayVersionCheck((yyvsp[-1].lex).loc, type->getArraySizes());
        parseContext.arraySizeRequiredCheck((yyvsp[0].interm).loc, *(yyvsp[0].interm).arraySizes);
        parseContext.reservedErrorCheck((yyvsp[-1].lex).loc, *(yyvsp[-1].lex).string);

        TParameter param = { (yyvsp[-1].lex).string, type };

        (yyval.interm).loc = (yyvsp[-1].lex).loc;
        (yyval.interm).param = param;
    }
#line 5709 "MachineIndependent/glslang_tab.cpp"
    break;

  case 115: /* parameter_declaration: type_qualifier parameter_declarator  */
#line 1051 "MachineIndependent/glslang.y"
                                          {
        (yyval.interm) = (yyvsp[0].interm);
        if ((yyvsp[-1].interm.type).qualifier.precision != EpqNone)
            (yyval.interm).param.type->getQualifier().precision = (yyvsp[-1].interm.type).qualifier.precision;
        parseContext.precisionQualifierCheck((yyval.interm).loc, (yyval.interm).param.type->getBasicType(), (yyval.interm).param.type->getQualifier());

        parseContext.checkNoShaderLayouts((yyvsp[-1].interm.type).loc, (yyvsp[-1].interm.type).shaderQualifiers);
        parseContext.parameterTypeCheck((yyvsp[0].interm).loc, (yyvsp[-1].interm.type).qualifier.storage, *(yyval.interm).param.type);
        parseContext.paramCheckFix((yyvsp[-1].interm.type).loc, (yyvsp[-1].interm.type).qualifier, *(yyval.interm).param.type);

    }
#line 5725 "MachineIndependent/glslang_tab.cpp"
    break;

  case 116: /* parameter_declaration: parameter_declarator  */
#line 1062 "MachineIndependent/glslang.y"
                           {
        (yyval.interm) = (yyvsp[0].interm);

        parseContext.parameterTypeCheck((yyvsp[0].interm).loc, EvqIn, *(yyvsp[0].interm).param.type);
        parseContext.paramCheckFixStorage((yyvsp[0].interm).loc, EvqTemporary, *(yyval.interm).param.type);
        parseContext.precisionQualifierCheck((yyval.interm).loc, (yyval.interm).param.type->getBasicType(), (yyval.interm).param.type->getQualifier());
    }
#line 5737 "MachineIndependent/glslang_tab.cpp"
    break;

  case 117: /* parameter_declaration: type_qualifier parameter_type_specifier  */
#line 1072 "MachineIndependent/glslang.y"
                                              {
        (yyval.interm) = (yyvsp[0].interm);
        if ((yyvsp[-1].interm.type).qualifier.precision != EpqNone)
            (yyval.interm).param.type->getQualifier().precision = (yyvsp[-1].interm.type).qualifier.precision;
        parseContext.precisionQualifierCheck((yyvsp[-1].interm.type).loc, (yyval.interm).param.type->getBasicType(), (yyval.interm).param.type->getQualifier());

        parseContext.checkNoShaderLayouts((yyvsp[-1].interm.type).loc, (yyvsp[-1].interm.type).shaderQualifiers);
        parseContext.parameterTypeCheck((yyvsp[0].interm).loc, (yyvsp[-1].interm.type).qualifier.storage, *(yyval.interm).param.type);
        parseContext.paramCheckFix((yyvsp[-1].interm.type).loc, (yyvsp[-1].interm.type).qualifier, *(yyval.interm).param.type);
    }
#line 5752 "MachineIndependent/glslang_tab.cpp"
    break;

  case 118: /* parameter_declaration: parameter_type_specifier  */
#line 1082 "MachineIndependent/glslang.y"
                               {
        (yyval.interm) = (yyvsp[0].interm);

        parseContext.parameterTypeCheck((yyvsp[0].interm).loc, EvqIn, *(yyvsp[0].interm).param.type);
        parseContext.paramCheckFixStorage((yyvsp[0].interm).loc, EvqTemporary, *(yyval.interm).param.type);
        parseContext.precisionQualifierCheck((yyval.interm).loc, (yyval.interm).param.type->getBasicType(), (yyval.interm).param.type->getQualifier());
    }
#line 5764 "MachineIndependent/glslang_tab.cpp"
    break;

  case 119: /* parameter_type_specifier: type_specifier  */
#line 1092 "MachineIndependent/glslang.y"
                     {
        TParameter param = { 0, new TType((yyvsp[0].interm.type)) };
        (yyval.interm).param = param;
        if ((yyvsp[0].interm.type).arraySizes)
            parseContext.arraySizeRequiredCheck((yyvsp[0].interm.type).loc, *(yyvsp[0].interm.type).arraySizes);
    }
#line 5775 "MachineIndependent/glslang_tab.cpp"
    break;

  case 120: /* init_declarator_list: single_declaration  */
#line 1101 "MachineIndependent/glslang.y"
                         {
        (yyval.interm) = (yyvsp[0].interm);
    }
#line 5783 "MachineIndependent/glslang_tab.cpp"
    break;

  case 121: /* init_declarator_list: init_declarator_list COMMA IDENTIFIER  */
#line 1104 "MachineIndependent/glslang.y"
                                            {
        (yyval.interm) = (yyvsp[-2].interm);
        parseContext.declareVariable((yyvsp[0].lex).loc, *(yyvsp[0].lex).string, (yyvsp[-2].interm).type);
    }
#line 5792 "MachineIndependent/glslang_tab.cpp"
    break;

  case 122: /* init_declarator_list: init_declarator_list COMMA IDENTIFIER array_specifier  */
#line 1108 "MachineIndependent/glslang.y"
                                                            {
        (yyval.interm) = (yyvsp[-3].interm);
        parseContext.declareVariable((yyvsp[-1].lex).loc, *(yyvsp[-1].lex).string, (yyvsp[-3].interm).type, (yyvsp[0].interm).arraySizes);
    }
#line 5801 "MachineIndependent/glslang_tab.cpp"
    break;

  case 123: /* init_declarator_list: init_declarator_list COMMA IDENTIFIER array_specifier EQUAL initializer  */
#line 1112 "MachineIndependent/glslang.y"
                                                                              {
        (yyval.interm).type = (yyvsp[-5].interm).type;
        TIntermNode* initNode = parseContext.declareVariable((yyvsp[-3].lex).loc, *(yyvsp[-3].lex).string, (yyvsp[-5].interm).type, (yyvsp[-2].interm).arraySizes, (yyvsp[0].interm.intermTypedNode));
        (yyval.interm).intermNode = parseContext.intermediate.growAggregate((yyvsp[-5].interm).intermNode, initNode, (yyvsp[-1].lex).loc);
    }
#line 5811 "MachineIndependent/glslang_tab.cpp"
    break;

  case 124: /* init_declarator_list: init_declarator_list COMMA IDENTIFIER EQUAL initializer  */
#line 1117 "MachineIndependent/glslang.y"
                                                              {
        (yyval.interm).type = (yyvsp[-4].interm).type;
        TIntermNode* initNode = parseContext.declareVariable((yyvsp[-2].lex).loc, *(yyvsp[-2].lex).string, (yyvsp[-4].interm).type, 0, (yyvsp[0].interm.intermTypedNode));
        (yyval.interm).intermNode = parseContext.intermediate.growAggregate((yyvsp[-4].interm).intermNode, initNode, (yyvsp[-1].lex).loc);
    }
#line 5821 "MachineIndependent/glslang_tab.cpp"
    break;

  case 125: /* single_declaration: fully_specified_type  */
#line 1125 "MachineIndependent/glslang.y"
                           {
        (yyval.interm).type = (yyvsp[0].interm.type);
        (yyval.interm).intermNode = 0;

        parseContext.declareTypeDefaults((yyval.interm).loc, (yyval.interm).type);

    }
#line 5833 "MachineIndependent/glslang_tab.cpp"
    break;

  case 126: /* single_declaration: fully_specified_type IDENTIFIER  */
#line 1132 "MachineIndependent/glslang.y"
                                      {
        (yyval.interm).type = (yyvsp[-1].interm.type);
        (yyval.interm).intermNode = 0;
        parseContext.declareVariable((yyvsp[0].lex).loc, *(yyvsp[0].lex).string, (yyvsp[-1].interm.type));
    }
#line 5843 "MachineIndependent/glslang_tab.cpp"
    break;

  case 127: /* single_declaration: fully_specified_type IDENTIFIER array_specifier  */
#line 1137 "MachineIndependent/glslang.y"
                                                      {
        (yyval.interm).type = (yyvsp[-2].interm.type);
        (yyval.interm).intermNode = 0;
        parseContext.declareVariable((yyvsp[-1].lex).loc, *(yyvsp[-1].lex).string, (yyvsp[-2].interm.type), (yyvsp[0].interm).arraySizes);
    }
#line 5853 "MachineIndependent/glslang_tab.cpp"
    break;

  case 128: /* single_declaration: fully_specified_type IDENTIFIER array_specifier EQUAL initializer  */
#line 1142 "MachineIndependent/glslang.y"
                                                                        {
        (yyval.interm).type = (yyvsp[-4].interm.type);
        TIntermNode* initNode = parseContext.declareVariable((yyvsp[-3].lex).loc, *(yyvsp[-3].lex).string, (yyvsp[-4].interm.type), (yyvsp[-2].interm).arraySizes, (yyvsp[0].interm.intermTypedNode));
        (yyval.interm).intermNode = parseContext.intermediate.growAggregate(0, initNode, (yyvsp[-1].lex).loc);
    }
#line 5863 "MachineIndependent/glslang_tab.cpp"
    break;

  case 129: /* single_declaration: fully_specified_type IDENTIFIER EQUAL initializer  */
#line 1147 "MachineIndependent/glslang.y"
                                                        {
        (yyval.interm).type = (yyvsp[-3].interm.type);
        TIntermNode* initNode = parseContext.declareVariable((yyvsp[-2].lex).loc, *(yyvsp[-2].lex).string, (yyvsp[-3].interm.type), 0, (yyvsp[0].interm.intermTypedNode));
        (yyval.interm).intermNode = parseContext.intermediate.growAggregate(0, initNode, (yyvsp[-1].lex).loc);
    }
#line 5873 "MachineIndependent/glslang_tab.cpp"
    break;

  case 130: /* fully_specified_type: type_specifier  */
#line 1156 "MachineIndependent/glslang.y"
                     {
        (yyval.interm.type) = (yyvsp[0].interm.type);

        parseContext.globalQualifierTypeCheck((yyvsp[0].interm.type).loc, (yyvsp[0].interm.type).qualifier, (yyval.interm.type));
        if ((yyvsp[0].interm.type).arraySizes) {
            parseContext.profileRequires((yyvsp[0].interm.type).loc, ENoProfile, 120, E_GL_3DL_array_objects, "arrayed type");
            parseContext.profileRequires((yyvsp[0].interm.type).loc, EEsProfile, 300, 0, "arrayed type");
        }
        parseContext.precisionQualifierCheck((yyval.interm.type).loc, (yyval.interm.type).basicType, (yyval.interm.type).qualifier);
    }
#line 5888 "MachineIndependent/glslang_tab.cpp"
    break;

  case 131: /* fully_specified_type: type_qualifier type_specifier  */
#line 1166 "MachineIndependent/glslang.y"
                                     {
        parseContext.globalQualifierFixCheck((yyvsp[-1].interm.type).loc, (yyvsp[-1].interm.type).qualifier);
        parseContext.globalQualifierTypeCheck((yyvsp[-1].interm.type).loc, (yyvsp[-1].interm.type).qualifier, (yyvsp[0].interm.type));

        if ((yyvsp[0].interm.type).arraySizes) {
            parseContext.profileRequires((yyvsp[0].interm.type).loc, ENoProfile, 120, E_GL_3DL_array_objects, "arrayed type");
            parseContext.profileRequires((yyvsp[0].interm.type).loc, EEsProfile, 300, 0, "arrayed type");
        }

        if ((yyvsp[0].interm.type).arraySizes && parseContext.arrayQualifierError((yyvsp[0].interm.type).loc, (yyvsp[-1].interm.type).qualifier))
            (yyvsp[0].interm.type).arraySizes = nullptr;

        parseContext.checkNoShaderLayouts((yyvsp[0].interm.type).loc, (yyvsp[-1].interm.type).shaderQualifiers);
        (yyvsp[0].interm.type).shaderQualifiers.merge((yyvsp[-1].interm.type).shaderQualifiers);
        parseContext.mergeQualifiers((yyvsp[0].interm.type).loc, (yyvsp[0].interm.type).qualifier, (yyvsp[-1].interm.type).qualifier, true);
        parseContext.precisionQualifierCheck((yyvsp[0].interm.type).loc, (yyvsp[0].interm.type).basicType, (yyvsp[0].interm.type).qualifier);

        (yyval.interm.type) = (yyvsp[0].interm.type);

        if (! (yyval.interm.type).qualifier.isInterpolation() &&
            ((parseContext.language == EShLangVertex   && (yyval.interm.type).qualifier.storage == EvqVaryingOut) ||
             (parseContext.language == EShLangFragment && (yyval.interm.type).qualifier.storage == EvqVaryingIn)))
            (yyval.interm.type).qualifier.smooth = true;
    }
#line 5917 "MachineIndependent/glslang_tab.cpp"
    break;

  case 132: /* invariant_qualifier: INVARIANT  */
#line 1193 "MachineIndependent/glslang.y"
                {
        parseContext.globalCheck((yyvsp[0].lex).loc, "invariant");
        parseContext.profileRequires((yyval.interm.type).loc, ENoProfile, 120, 0, "invariant");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.invariant = true;
    }
#line 5928 "MachineIndependent/glslang_tab.cpp"
    break;

  case 133: /* interpolation_qualifier: SMOOTH  */
#line 1202 "MachineIndependent/glslang.y"
             {
        parseContext.globalCheck((yyvsp[0].lex).loc, "smooth");
        parseContext.profileRequires((yyvsp[0].lex).loc, ENoProfile, 130, 0, "smooth");
        parseContext.profileRequires((yyvsp[0].lex).loc, EEsProfile, 300, 0, "smooth");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.smooth = true;
    }
#line 5940 "MachineIndependent/glslang_tab.cpp"
    break;

  case 134: /* interpolation_qualifier: FLAT  */
#line 1209 "MachineIndependent/glslang.y"
           {
        parseContext.globalCheck((yyvsp[0].lex).loc, "flat");
        parseContext.profileRequires((yyvsp[0].lex).loc, ENoProfile, 130, 0, "flat");
        parseContext.profileRequires((yyvsp[0].lex).loc, EEsProfile, 300, 0, "flat");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.flat = true;
    }
#line 5952 "MachineIndependent/glslang_tab.cpp"
    break;

  case 135: /* interpolation_qualifier: NOPERSPECTIVE  */
#line 1217 "MachineIndependent/glslang.y"
                    {
        parseContext.globalCheck((yyvsp[0].lex).loc, "noperspective");
        parseContext.profileRequires((yyvsp[0].lex).loc, EEsProfile, 0, E_GL_NV_shader_noperspective_interpolation, "noperspective");
        parseContext.profileRequires((yyvsp[0].lex).loc, ENoProfile, 130, 0, "noperspective");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.nopersp = true;
    }
#line 5964 "MachineIndependent/glslang_tab.cpp"
    break;

  case 136: /* interpolation_qualifier: EXPLICITINTERPAMD  */
#line 1224 "MachineIndependent/glslang.y"
                        {
        parseContext.globalCheck((yyvsp[0].lex).loc, "__explicitInterpAMD");
        parseContext.profileRequires((yyvsp[0].lex).loc, ECoreProfile, 450, E_GL_AMD_shader_explicit_vertex_parameter, "explicit interpolation");
        parseContext.profileRequires((yyvsp[0].lex).loc, ECompatibilityProfile, 450, E_GL_AMD_shader_explicit_vertex_parameter, "explicit interpolation");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.explicitInterp = true;
    }
#line 5976 "MachineIndependent/glslang_tab.cpp"
    break;

  case 137: /* interpolation_qualifier: PERVERTEXNV  */
#line 1231 "MachineIndependent/glslang.y"
                  {
        parseContext.globalCheck((yyvsp[0].lex).loc, "pervertexNV");
        parseContext.profileRequires((yyvsp[0].lex).loc, ECoreProfile, 0, E_GL_NV_fragment_shader_barycentric, "fragment shader barycentric");
        parseContext.profileRequires((yyvsp[0].lex).loc, ECompatibilityProfile, 0, E_GL_NV_fragment_shader_barycentric, "fragment shader barycentric");
        parseContext.profileRequires((yyvsp[0].lex).loc, EEsProfile, 0, E_GL_NV_fragment_shader_barycentric, "fragment shader barycentric");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.pervertexNV = true;
    }
#line 5989 "MachineIndependent/glslang_tab.cpp"
    break;

  case 138: /* interpolation_qualifier: PERPRIMITIVENV  */
#line 1239 "MachineIndependent/glslang.y"
                     {
        // No need for profile version or extension check. Shader stage already checks both.
        parseContext.globalCheck((yyvsp[0].lex).loc, "perprimitiveNV");
        parseContext.requireStage((yyvsp[0].lex).loc, (EShLanguageMask)(EShLangFragmentMask | EShLangMeshNVMask), "perprimitiveNV");
        // Fragment shader stage doesn't check for extension. So we explicitly add below extension check.
        if (parseContext.language == EShLangFragment)
            parseContext.requireExtensions((yyvsp[0].lex).loc, 1, &E_GL_NV_mesh_shader, "perprimitiveNV");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.perPrimitiveNV = true;
    }
#line 6004 "MachineIndependent/glslang_tab.cpp"
    break;

  case 139: /* interpolation_qualifier: PERVIEWNV  */
#line 1249 "MachineIndependent/glslang.y"
                {
        // No need for profile version or extension check. Shader stage already checks both.
        parseContext.globalCheck((yyvsp[0].lex).loc, "perviewNV");
        parseContext.requireStage((yyvsp[0].lex).loc, EShLangMeshNV, "perviewNV");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.perViewNV = true;
    }
#line 6016 "MachineIndependent/glslang_tab.cpp"
    break;

  case 140: /* interpolation_qualifier: PERTASKNV  */
#line 1256 "MachineIndependent/glslang.y"
                {
        // No need for profile version or extension check. Shader stage already checks both.
        parseContext.globalCheck((yyvsp[0].lex).loc, "taskNV");
        parseContext.requireStage((yyvsp[0].lex).loc, (EShLanguageMask)(EShLangTaskNVMask | EShLangMeshNVMask), "taskNV");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.perTaskNV = true;
    }
#line 6028 "MachineIndependent/glslang_tab.cpp"
    break;

  case 141: /* layout_qualifier: LAYOUT LEFT_PAREN layout_qualifier_id_list RIGHT_PAREN  */
#line 1267 "MachineIndependent/glslang.y"
                                                             {
        (yyval.interm.type) = (yyvsp[-1].interm.type);
    }
#line 6036 "MachineIndependent/glslang_tab.cpp"
    break;

  case 142: /* layout_qualifier_id_list: layout_qualifier_id  */
#line 1273 "MachineIndependent/glslang.y"
                          {
        (yyval.interm.type) = (yyvsp[0].interm.type);
    }
#line 6044 "MachineIndependent/glslang_tab.cpp"
    break;

  case 143: /* layout_qualifier_id_list: layout_qualifier_id_list COMMA layout_qualifier_id  */
#line 1276 "MachineIndependent/glslang.y"
                                                         {
        (yyval.interm.type) = (yyvsp[-2].interm.type);
        (yyval.interm.type).shaderQualifiers.merge((yyvsp[0].interm.type).shaderQualifiers);
        parseContext.mergeObjectLayoutQualifiers((yyval.interm.type).qualifier, (yyvsp[0].interm.type).qualifier, false);
    }
#line 6054 "MachineIndependent/glslang_tab.cpp"
    break;

  case 144: /* layout_qualifier_id: IDENTIFIER  */
#line 1283 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        parseContext.setLayoutQualifier((yyvsp[0].lex).loc, (yyval.interm.type), *(yyvsp[0].lex).string);
    }
#line 6063 "MachineIndependent/glslang_tab.cpp"
    break;

  case 145: /* layout_qualifier_id: IDENTIFIER EQUAL constant_expression  */
#line 1287 "MachineIndependent/glslang.y"
                                           {
        (yyval.interm.type).init((yyvsp[-2].lex).loc);
        parseContext.setLayoutQualifier((yyvsp[-2].lex).loc, (yyval.interm.type), *(yyvsp[-2].lex).string, (yyvsp[0].interm.intermTypedNode));
    }
#line 6072 "MachineIndependent/glslang_tab.cpp"
    break;

  case 146: /* layout_qualifier_id: SHARED  */
#line 1291 "MachineIndependent/glslang.y"
             { // because "shared" is both an identifier and a keyword
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        TString strShared("shared");
        parseContext.setLayoutQualifier((yyvsp[0].lex).loc, (yyval.interm.type), strShared);
    }
#line 6082 "MachineIndependent/glslang_tab.cpp"
    break;

  case 147: /* precise_qualifier: PRECISE  */
#line 1300 "MachineIndependent/glslang.y"
              {
        parseContext.profileRequires((yyval.interm.type).loc, ECoreProfile | ECompatibilityProfile, 400, E_GL_ARB_gpu_shader5, "precise");
        parseContext.profileRequires((yyvsp[0].lex).loc, EEsProfile, 320, Num_AEP_gpu_shader5, AEP_gpu_shader5, "precise");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.noContraction = true;
    }
#line 6093 "MachineIndependent/glslang_tab.cpp"
    break;

  case 148: /* type_qualifier: single_type_qualifier  */
#line 1310 "MachineIndependent/glslang.y"
                            {
        (yyval.interm.type) = (yyvsp[0].interm.type);
    }
#line 6101 "MachineIndependent/glslang_tab.cpp"
    break;

  case 149: /* type_qualifier: type_qualifier single_type_qualifier  */
#line 1313 "MachineIndependent/glslang.y"
                                           {
        (yyval.interm.type) = (yyvsp[-1].interm.type);
        if ((yyval.interm.type).basicType == EbtVoid)
            (yyval.interm.type).basicType = (yyvsp[0].interm.type).basicType;

        (yyval.interm.type).shaderQualifiers.merge((yyvsp[0].interm.type).shaderQualifiers);
        parseContext.mergeQualifiers((yyval.interm.type).loc, (yyval.interm.type).qualifier, (yyvsp[0].interm.type).qualifier, false);
    }
#line 6114 "MachineIndependent/glslang_tab.cpp"
    break;

  case 150: /* single_type_qualifier: storage_qualifier  */
#line 1324 "MachineIndependent/glslang.y"
                        {
        (yyval.interm.type) = (yyvsp[0].interm.type);
    }
#line 6122 "MachineIndependent/glslang_tab.cpp"
    break;

  case 151: /* single_type_qualifier: layout_qualifier  */
#line 1327 "MachineIndependent/glslang.y"
                       {
        (yyval.interm.type) = (yyvsp[0].interm.type);
    }
#line 6130 "MachineIndependent/glslang_tab.cpp"
    break;

  case 152: /* single_type_qualifier: precision_qualifier  */
#line 1330 "MachineIndependent/glslang.y"
                          {
        parseContext.checkPrecisionQualifier((yyvsp[0].interm.type).loc, (yyvsp[0].interm.type).qualifier.precision);
        (yyval.interm.type) = (yyvsp[0].interm.type);
    }
#line 6139 "MachineIndependent/glslang_tab.cpp"
    break;

  case 153: /* single_type_qualifier: interpolation_qualifier  */
#line 1334 "MachineIndependent/glslang.y"
                              {
        // allow inheritance of storage qualifier from block declaration
        (yyval.interm.type) = (yyvsp[0].interm.type);
    }
#line 6148 "MachineIndependent/glslang_tab.cpp"
    break;

  case 154: /* single_type_qualifier: invariant_qualifier  */
#line 1338 "MachineIndependent/glslang.y"
                          {
        // allow inheritance of storage qualifier from block declaration
        (yyval.interm.type) = (yyvsp[0].interm.type);
    }
#line 6157 "MachineIndependent/glslang_tab.cpp"
    break;

  case 155: /* single_type_qualifier: precise_qualifier  */
#line 1343 "MachineIndependent/glslang.y"
                        {
        // allow inheritance of storage qualifier from block declaration
        (yyval.interm.type) = (yyvsp[0].interm.type);
    }
#line 6166 "MachineIndependent/glslang_tab.cpp"
    break;

  case 156: /* single_type_qualifier: non_uniform_qualifier  */
#line 1347 "MachineIndependent/glslang.y"
                            {
        (yyval.interm.type) = (yyvsp[0].interm.type);
    }
#line 6174 "MachineIndependent/glslang_tab.cpp"
    break;

  case 157: /* storage_qualifier: CONST  */
#line 1354 "MachineIndependent/glslang.y"
            {
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.storage = EvqConst;  // will later turn into EvqConstReadOnly, if the initializer is not constant
    }
#line 6183 "MachineIndependent/glslang_tab.cpp"
    break;

  case 158: /* storage_qualifier: INOUT  */
#line 1358 "MachineIndependent/glslang.y"
            {
        parseContext.globalCheck((yyvsp[0].lex).loc, "inout");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.storage = EvqInOut;
    }
#line 6193 "MachineIndependent/glslang_tab.cpp"
    break;

  case 159: /* storage_qualifier: IN  */
#line 1363 "MachineIndependent/glslang.y"
         {
        parseContext.globalCheck((yyvsp[0].lex).loc, "in");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        // whether this is a parameter "in" or a pipeline "in" will get sorted out a bit later
        (yyval.interm.type).qualifier.storage = EvqIn;
    }
#line 6204 "MachineIndependent/glslang_tab.cpp"
    break;

  case 160: /* storage_qualifier: OUT  */
#line 1369 "MachineIndependent/glslang.y"
          {
        parseContext.globalCheck((yyvsp[0].lex).loc, "out");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        // whether this is a parameter "out" or a pipeline "out" will get sorted out a bit later
        (yyval.interm.type).qualifier.storage = EvqOut;
    }
#line 6215 "MachineIndependent/glslang_tab.cpp"
    break;

  case 161: /* storage_qualifier: CENTROID  */
#line 1375 "MachineIndependent/glslang.y"
               {
        parseContext.profileRequires((yyvsp[0].lex).loc, ENoProfile, 120, 0, "centroid");
        parseContext.profileRequires((yyvsp[0].lex).loc, EEsProfile, 300, 0, "centroid");
        parseContext.globalCheck((yyvsp[0].lex).loc, "centroid");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.centroid = true;
    }
#line 6227 "MachineIndependent/glslang_tab.cpp"
    break;

  case 162: /* storage_qualifier: UNIFORM  */
#line 1382 "MachineIndependent/glslang.y"
              {
        parseContext.globalCheck((yyvsp[0].lex).loc, "uniform");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.storage = EvqUniform;
    }
#line 6237 "MachineIndependent/glslang_tab.cpp"
    break;

  case 163: /* storage_qualifier: SHARED  */
#line 1387 "MachineIndependent/glslang.y"
             {
        parseContext.globalCheck((yyvsp[0].lex).loc, "shared");
        parseContext.profileRequires((yyvsp[0].lex).loc, ECoreProfile | ECompatibilityProfile, 430, E_GL_ARB_compute_shader, "shared");
        parseContext.profileRequires((yyvsp[0].lex).loc, EEsProfile, 310, 0, "shared");
        parseContext.requireStage((yyvsp[0].lex).loc, (EShLanguageMask)(EShLangComputeMask | EShLangMeshNVMask | EShLangTaskNVMask), "shared");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.storage = EvqShared;
    }
#line 6250 "MachineIndependent/glslang_tab.cpp"
    break;

  case 164: /* storage_qualifier: BUFFER  */
#line 1395 "MachineIndependent/glslang.y"
             {
        parseContext.globalCheck((yyvsp[0].lex).loc, "buffer");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.storage = EvqBuffer;
    }
#line 6260 "MachineIndependent/glslang_tab.cpp"
    break;

  case 165: /* storage_qualifier: ATTRIBUTE  */
#line 1401 "MachineIndependent/glslang.y"
                {
        parseContext.requireStage((yyvsp[0].lex).loc, EShLangVertex, "attribute");
        parseContext.checkDeprecated((yyvsp[0].lex).loc, ECoreProfile, 130, "attribute");
        parseContext.checkDeprecated((yyvsp[0].lex).loc, ENoProfile, 130, "attribute");
        parseContext.requireNotRemoved((yyvsp[0].lex).loc, ECoreProfile, 420, "attribute");
        parseContext.requireNotRemoved((yyvsp[0].lex).loc, EEsProfile, 300, "attribute");

        parseContext.globalCheck((yyvsp[0].lex).loc, "attribute");

        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.storage = EvqVaryingIn;
    }
#line 6277 "MachineIndependent/glslang_tab.cpp"
    break;

  case 166: /* storage_qualifier: VARYING  */
#line 1413 "MachineIndependent/glslang.y"
              {
        parseContext.checkDeprecated((yyvsp[0].lex).loc, ENoProfile, 130, "varying");
        parseContext.checkDeprecated((yyvsp[0].lex).loc, ECoreProfile, 130, "varying");
        parseContext.requireNotRemoved((yyvsp[0].lex).loc, ECoreProfile, 420, "varying");
        parseContext.requireNotRemoved((yyvsp[0].lex).loc, EEsProfile, 300, "varying");

        parseContext.globalCheck((yyvsp[0].lex).loc, "varying");

        (yyval.interm.type).init((yyvsp[0].lex).loc);
        if (parseContext.language == EShLangVertex)
            (yyval.interm.type).qualifier.storage = EvqVaryingOut;
        else
            (yyval.interm.type).qualifier.storage = EvqVaryingIn;
    }
#line 6296 "MachineIndependent/glslang_tab.cpp"
    break;

  case 167: /* storage_qualifier: PATCH  */
#line 1427 "MachineIndependent/glslang.y"
            {
        parseContext.globalCheck((yyvsp[0].lex).loc, "patch");
        parseContext.requireStage((yyvsp[0].lex).loc, (EShLanguageMask)(EShLangTessControlMask | EShLangTessEvaluationMask), "patch");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.patch = true;
    }
#line 6307 "MachineIndependent/glslang_tab.cpp"
    break;

  case 168: /* storage_qualifier: SAMPLE  */
#line 1433 "MachineIndependent/glslang.y"
             {
        parseContext.globalCheck((yyvsp[0].lex).loc, "sample");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.sample = true;
    }
#line 6317 "MachineIndependent/glslang_tab.cpp"
    break;

  case 169: /* storage_qualifier: HITATTRNV  */
#line 1438 "MachineIndependent/glslang.y"
                {
        parseContext.globalCheck((yyvsp[0].lex).loc, "hitAttributeNV");
        parseContext.requireStage((yyvsp[0].lex).loc, (EShLanguageMask)(EShLangIntersectMask | EShLangClosestHitMask
            | EShLangAnyHitMask), "hitAttributeNV");
        parseContext.profileRequires((yyvsp[0].lex).loc, ECoreProfile, 460, E_GL_NV_ray_tracing, "hitAttributeNV");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.storage = EvqHitAttr;
    }
#line 6330 "MachineIndependent/glslang_tab.cpp"
    break;

  case 170: /* storage_qualifier: HITATTREXT  */
#line 1446 "MachineIndependent/glslang.y"
                 {
        parseContext.globalCheck((yyvsp[0].lex).loc, "hitAttributeEXT");
        parseContext.requireStage((yyvsp[0].lex).loc, (EShLanguageMask)(EShLangIntersectMask | EShLangClosestHitMask
            | EShLangAnyHitMask), "hitAttributeEXT");
        parseContext.profileRequires((yyvsp[0].lex).loc, ECoreProfile, 460, E_GL_EXT_ray_tracing, "hitAttributeNV");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.storage = EvqHitAttr;
    }
#line 6343 "MachineIndependent/glslang_tab.cpp"
    break;

  case 171: /* storage_qualifier: PAYLOADNV  */
#line 1454 "MachineIndependent/glslang.y"
                {
        parseContext.globalCheck((yyvsp[0].lex).loc, "rayPayloadNV");
        parseContext.requireStage((yyvsp[0].lex).loc, (EShLanguageMask)(EShLangRayGenMask | EShLangClosestHitMask |
            EShLangAnyHitMask | EShLangMissMask), "rayPayloadNV");
        parseContext.profileRequires((yyvsp[0].lex).loc, ECoreProfile, 460, E_GL_NV_ray_tracing, "rayPayloadNV");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.storage = EvqPayload;
    }
#line 6356 "MachineIndependent/glslang_tab.cpp"
    break;

  case 172: /* storage_qualifier: PAYLOADEXT  */
#line 1462 "MachineIndependent/glslang.y"
                 {
        parseContext.globalCheck((yyvsp[0].lex).loc, "rayPayloadEXT");
        parseContext.requireStage((yyvsp[0].lex).loc, (EShLanguageMask)(EShLangRayGenMask | EShLangClosestHitMask |
            EShLangAnyHitMask | EShLangMissMask), "rayPayloadEXT");
        parseContext.profileRequires((yyvsp[0].lex).loc, ECoreProfile, 460, E_GL_EXT_ray_tracing, "rayPayloadEXT");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.storage = EvqPayload;
    }
#line 6369 "MachineIndependent/glslang_tab.cpp"
    break;

  case 173: /* storage_qualifier: PAYLOADINNV  */
#line 1470 "MachineIndependent/glslang.y"
                  {
        parseContext.globalCheck((yyvsp[0].lex).loc, "rayPayloadInNV");
        parseContext.requireStage((yyvsp[0].lex).loc, (EShLanguageMask)(EShLangClosestHitMask |
            EShLangAnyHitMask | EShLangMissMask), "rayPayloadInNV");
        parseContext.profileRequires((yyvsp[0].lex).loc, ECoreProfile, 460, E_GL_NV_ray_tracing, "rayPayloadInNV");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.storage = EvqPayloadIn;
    }
#line 6382 "MachineIndependent/glslang_tab.cpp"
    break;

  case 174: /* storage_qualifier: PAYLOADINEXT  */
#line 1478 "MachineIndependent/glslang.y"
                   {
        parseContext.globalCheck((yyvsp[0].lex).loc, "rayPayloadInEXT");
        parseContext.requireStage((yyvsp[0].lex).loc, (EShLanguageMask)(EShLangClosestHitMask |
            EShLangAnyHitMask | EShLangMissMask), "rayPayloadInEXT");
        parseContext.profileRequires((yyvsp[0].lex).loc, ECoreProfile, 460, E_GL_EXT_ray_tracing, "rayPayloadInEXT");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.storage = EvqPayloadIn;
    }
#line 6395 "MachineIndependent/glslang_tab.cpp"
    break;

  case 175: /* storage_qualifier: CALLDATANV  */
#line 1486 "MachineIndependent/glslang.y"
                 {
        parseContext.globalCheck((yyvsp[0].lex).loc, "callableDataNV");
        parseContext.requireStage((yyvsp[0].lex).loc, (EShLanguageMask)(EShLangRayGenMask |
            EShLangClosestHitMask | EShLangMissMask | EShLangCallableMask), "callableDataNV");
        parseContext.profileRequires((yyvsp[0].lex).loc, ECoreProfile, 460, E_GL_NV_ray_tracing, "callableDataNV");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.storage = EvqCallableData;
    }
#line 6408 "MachineIndependent/glslang_tab.cpp"
    break;

  case 176: /* storage_qualifier: CALLDATAEXT  */
#line 1494 "MachineIndependent/glslang.y"
                  {
        parseContext.globalCheck((yyvsp[0].lex).loc, "callableDataEXT");
        parseContext.requireStage((yyvsp[0].lex).loc, (EShLanguageMask)(EShLangRayGenMask |
            EShLangClosestHitMask | EShLangMissMask | EShLangCallableMask), "callableDataEXT");
        parseContext.profileRequires((yyvsp[0].lex).loc, ECoreProfile, 460, E_GL_EXT_ray_tracing, "callableDataEXT");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.storage = EvqCallableData;
    }
#line 6421 "MachineIndependent/glslang_tab.cpp"
    break;

  case 177: /* storage_qualifier: CALLDATAINNV  */
#line 1502 "MachineIndependent/glslang.y"
                   {
        parseContext.globalCheck((yyvsp[0].lex).loc, "callableDataInNV");
        parseContext.requireStage((yyvsp[0].lex).loc, (EShLanguageMask)(EShLangCallableMask), "callableDataInNV");
        parseContext.profileRequires((yyvsp[0].lex).loc, ECoreProfile, 460, E_GL_NV_ray_tracing, "callableDataInNV");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.storage = EvqCallableDataIn;
    }
#line 6433 "MachineIndependent/glslang_tab.cpp"
    break;

  case 178: /* storage_qualifier: CALLDATAINEXT  */
#line 1509 "MachineIndependent/glslang.y"
                    {
        parseContext.globalCheck((yyvsp[0].lex).loc, "callableDataInEXT");
        parseContext.requireStage((yyvsp[0].lex).loc, (EShLanguageMask)(EShLangCallableMask), "callableDataInEXT");
        parseContext.profileRequires((yyvsp[0].lex).loc, ECoreProfile, 460, E_GL_EXT_ray_tracing, "callableDataInEXT");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.storage = EvqCallableDataIn;
    }
#line 6445 "MachineIndependent/glslang_tab.cpp"
    break;

  case 179: /* storage_qualifier: COHERENT  */
#line 1516 "MachineIndependent/glslang.y"
               {
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.coherent = true;
    }
#line 6454 "MachineIndependent/glslang_tab.cpp"
    break;

  case 180: /* storage_qualifier: DEVICECOHERENT  */
#line 1520 "MachineIndependent/glslang.y"
                     {
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        parseContext.requireExtensions((yyvsp[0].lex).loc, 1, &E_GL_KHR_memory_scope_semantics, "devicecoherent");
        (yyval.interm.type).qualifier.devicecoherent = true;
    }
#line 6464 "MachineIndependent/glslang_tab.cpp"
    break;

  case 181: /* storage_qualifier: QUEUEFAMILYCOHERENT  */
#line 1525 "MachineIndependent/glslang.y"
                          {
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        parseContext.requireExtensions((yyvsp[0].lex).loc, 1, &E_GL_KHR_memory_scope_semantics, "queuefamilycoherent");
        (yyval.interm.type).qualifier.queuefamilycoherent = true;
    }
#line 6474 "MachineIndependent/glslang_tab.cpp"
    break;

  case 182: /* storage_qualifier: WORKGROUPCOHERENT  */
#line 1530 "MachineIndependent/glslang.y"
                        {
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        parseContext.requireExtensions((yyvsp[0].lex).loc, 1, &E_GL_KHR_memory_scope_semantics, "workgroupcoherent");
        (yyval.interm.type).qualifier.workgroupcoherent = true;
    }
#line 6484 "MachineIndependent/glslang_tab.cpp"
    break;

  case 183: /* storage_qualifier: SUBGROUPCOHERENT  */
#line 1535 "MachineIndependent/glslang.y"
                       {
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        parseContext.requireExtensions((yyvsp[0].lex).loc, 1, &E_GL_KHR_memory_scope_semantics, "subgroupcoherent");
        (yyval.interm.type).qualifier.subgroupcoherent = true;
    }
#line 6494 "MachineIndependent/glslang_tab.cpp"
    break;

  case 184: /* storage_qualifier: NONPRIVATE  */
#line 1540 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        parseContext.requireExtensions((yyvsp[0].lex).loc, 1, &E_GL_KHR_memory_scope_semantics, "nonprivate");
        (yyval.interm.type).qualifier.nonprivate = true;
    }
#line 6504 "MachineIndependent/glslang_tab.cpp"
    break;

  case 185: /* storage_qualifier: SHADERCALLCOHERENT  */
#line 1545 "MachineIndependent/glslang.y"
                         {
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        parseContext.requireExtensions((yyvsp[0].lex).loc, 1, &E_GL_EXT_ray_tracing, "shadercallcoherent");
        (yyval.interm.type).qualifier.shadercallcoherent = true;
    }
#line 6514 "MachineIndependent/glslang_tab.cpp"
    break;

  case 186: /* storage_qualifier: VOLATILE  */
#line 1550 "MachineIndependent/glslang.y"
               {
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.volatil = true;
    }
#line 6523 "MachineIndependent/glslang_tab.cpp"
    break;

  case 187: /* storage_qualifier: RESTRICT  */
#line 1554 "MachineIndependent/glslang.y"
               {
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.restrict = true;
    }
#line 6532 "MachineIndependent/glslang_tab.cpp"
    break;

  case 188: /* storage_qualifier: READONLY  */
#line 1558 "MachineIndependent/glslang.y"
               {
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.readonly = true;
    }
#line 6541 "MachineIndependent/glslang_tab.cpp"
    break;

  case 189: /* storage_qualifier: WRITEONLY  */
#line 1562 "MachineIndependent/glslang.y"
                {
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.writeonly = true;
    }
#line 6550 "MachineIndependent/glslang_tab.cpp"
    break;

  case 190: /* storage_qualifier: SUBROUTINE  */
#line 1566 "MachineIndependent/glslang.y"
                 {
        parseContext.spvRemoved((yyvsp[0].lex).loc, "subroutine");
        parseContext.globalCheck((yyvsp[0].lex).loc, "subroutine");
        parseContext.unimplemented((yyvsp[0].lex).loc, "subroutine");
        (yyval.interm.type).init((yyvsp[0].lex).loc);
    }
#line 6561 "MachineIndependent/glslang_tab.cpp"
    break;

  case 191: /* storage_qualifier: SUBROUTINE LEFT_PAREN type_name_list RIGHT_PAREN  */
#line 1572 "MachineIndependent/glslang.y"
                                                       {
        parseContext.spvRemoved((yyvsp[-3].lex).loc, "subroutine");
        parseContext.globalCheck((yyvsp[-3].lex).loc, "subroutine");
        parseContext.unimplemented((yyvsp[-3].lex).loc, "subroutine");
        (yyval.interm.type).init((yyvsp[-3].lex).loc);
    }
#line 6572 "MachineIndependent/glslang_tab.cpp"
    break;

  case 192: /* non_uniform_qualifier: NONUNIFORM  */
#line 1583 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.type).init((yyvsp[0].lex).loc);
        (yyval.interm.type).qualifier.nonUniform = true;
    }
#line 6581 "MachineIndependent/glslang_tab.cpp"
    break;

  case 193: /* type_name_list: IDENTIFIER  */
#line 1590 "MachineIndependent/glslang.y"
                 {
        // TODO
    }
#line 6589 "MachineIndependent/glslang_tab.cpp"
    break;

  case 194: /* type_name_list: type_name_list COMMA IDENTIFIER  */
#line 1593 "MachineIndependent/glslang.y"
                                      {
        // TODO: 4.0 semantics: subroutines
        // 1) make sure each identifier is a type declared earlier with SUBROUTINE
        // 2) save all of the identifiers for future comparison with the declared function
    }
#line 6599 "MachineIndependent/glslang_tab.cpp"
    break;

  case 195: /* type_specifier: type_specifier_nonarray type_parameter_specifier_opt  */
#line 1602 "MachineIndependent/glslang.y"
                                                           {
        (yyval.interm.type) = (yyvsp[-1].interm.type);
        (yyval.interm.type).qualifier.precision = parseContext.getDefaultPrecision((yyval.interm.type));
        (yyval.interm.type).typeParameters = (yyvsp[0].interm.typeParameters);
    }
#line 6609 "MachineIndependent/glslang_tab.cpp"
    break;

  case 196: /* type_specifier: type_specifier_nonarray type_parameter_specifier_opt array_specifier  */
#line 1607 "MachineIndependent/glslang.y"
                                                                           {
        parseContext.arrayOfArrayVersionCheck((yyvsp[0].interm).loc, (yyvsp[0].interm).arraySizes);
        (yyval.interm.type) = (yyvsp[-2].interm.type);
        (yyval.interm.type).qualifier.precision = parseContext.getDefaultPrecision((yyval.interm.type));
        (yyval.interm.type).typeParameters = (yyvsp[-1].interm.typeParameters);
        (yyval.interm.type).arraySizes = (yyvsp[0].interm).arraySizes;
    }
#line 6621 "MachineIndependent/glslang_tab.cpp"
    break;

  case 197: /* array_specifier: LEFT_BRACKET RIGHT_BRACKET  */
#line 1617 "MachineIndependent/glslang.y"
                                 {
        (yyval.interm).loc = (yyvsp[-1].lex).loc;
        (yyval.interm).arraySizes = new TArraySizes;
        (yyval.interm).arraySizes->addInnerSize();
    }
#line 6631 "MachineIndependent/glslang_tab.cpp"
    break;

  case 198: /* array_specifier: LEFT_BRACKET conditional_expression RIGHT_BRACKET  */
#line 1622 "MachineIndependent/glslang.y"
                                                        {
        (yyval.interm).loc = (yyvsp[-2].lex).loc;
        (yyval.interm).arraySizes = new TArraySizes;

        TArraySize size;
        parseContext.arraySizeCheck((yyvsp[-1].interm.intermTypedNode)->getLoc(), (yyvsp[-1].interm.intermTypedNode), size, "array size");
        (yyval.interm).arraySizes->addInnerSize(size);
    }
#line 6644 "MachineIndependent/glslang_tab.cpp"
    break;

  case 199: /* array_specifier: array_specifier LEFT_BRACKET RIGHT_BRACKET  */
#line 1630 "MachineIndependent/glslang.y"
                                                 {
        (yyval.interm) = (yyvsp[-2].interm);
        (yyval.interm).arraySizes->addInnerSize();
    }
#line 6653 "MachineIndependent/glslang_tab.cpp"
    break;

  case 200: /* array_specifier: array_specifier LEFT_BRACKET conditional_expression RIGHT_BRACKET  */
#line 1634 "MachineIndependent/glslang.y"
                                                                        {
        (yyval.interm) = (yyvsp[-3].interm);

        TArraySize size;
        parseContext.arraySizeCheck((yyvsp[-1].interm.intermTypedNode)->getLoc(), (yyvsp[-1].interm.intermTypedNode), size, "array size");
        (yyval.interm).arraySizes->addInnerSize(size);
    }
#line 6665 "MachineIndependent/glslang_tab.cpp"
    break;

  case 201: /* type_parameter_specifier_opt: type_parameter_specifier  */
#line 1644 "MachineIndependent/glslang.y"
                               {
        (yyval.interm.typeParameters) = (yyvsp[0].interm.typeParameters);
    }
#line 6673 "MachineIndependent/glslang_tab.cpp"
    break;

  case 202: /* type_parameter_specifier_opt: %empty  */
#line 1647 "MachineIndependent/glslang.y"
                        {
        (yyval.interm.typeParameters) = 0;
    }
#line 6681 "MachineIndependent/glslang_tab.cpp"
    break;

  case 203: /* type_parameter_specifier: LEFT_ANGLE type_parameter_specifier_list RIGHT_ANGLE  */
#line 1653 "MachineIndependent/glslang.y"
                                                           {
        (yyval.interm.typeParameters) = (yyvsp[-1].interm.typeParameters);
    }
#line 6689 "MachineIndependent/glslang_tab.cpp"
    break;

  case 204: /* type_parameter_specifier_list: unary_expression  */
#line 1659 "MachineIndependent/glslang.y"
                       {
        (yyval.interm.typeParameters) = new TArraySizes;

        TArraySize size;
        parseContext.arraySizeCheck((yyvsp[0].interm.intermTypedNode)->getLoc(), (yyvsp[0].interm.intermTypedNode), size, "type parameter");
        (yyval.interm.typeParameters)->addInnerSize(size);
    }
#line 6701 "MachineIndependent/glslang_tab.cpp"
    break;

  case 205: /* type_parameter_specifier_list: type_parameter_specifier_list COMMA unary_expression  */
#line 1666 "MachineIndependent/glslang.y"
                                                           {
        (yyval.interm.typeParameters) = (yyvsp[-2].interm.typeParameters);

        TArraySize size;
        parseContext.arraySizeCheck((yyvsp[0].interm.intermTypedNode)->getLoc(), (yyvsp[0].interm.intermTypedNode), size, "type parameter");
        (yyval.interm.typeParameters)->addInnerSize(size);
    }
#line 6713 "MachineIndependent/glslang_tab.cpp"
    break;

  case 206: /* type_specifier_nonarray: VOID  */
#line 1676 "MachineIndependent/glslang.y"
           {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtVoid;
    }
#line 6722 "MachineIndependent/glslang_tab.cpp"
    break;

  case 207: /* type_specifier_nonarray: FLOAT  */
#line 1680 "MachineIndependent/glslang.y"
            {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
    }
#line 6731 "MachineIndependent/glslang_tab.cpp"
    break;

  case 208: /* type_specifier_nonarray: INT  */
#line 1684 "MachineIndependent/glslang.y"
          {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtInt;
    }
#line 6740 "MachineIndependent/glslang_tab.cpp"
    break;

  case 209: /* type_specifier_nonarray: UINT  */
#line 1688 "MachineIndependent/glslang.y"
           {
        parseContext.fullIntegerCheck((yyvsp[0].lex).loc, "unsigned integer");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtUint;
    }
#line 6750 "MachineIndependent/glslang_tab.cpp"
    break;

  case 210: /* type_specifier_nonarray: BOOL  */
#line 1693 "MachineIndependent/glslang.y"
           {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtBool;
    }
#line 6759 "MachineIndependent/glslang_tab.cpp"
    break;

  case 211: /* type_specifier_nonarray: VEC2  */
#line 1697 "MachineIndependent/glslang.y"
           {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setVector(2);
    }
#line 6769 "MachineIndependent/glslang_tab.cpp"
    break;

  case 212: /* type_specifier_nonarray: VEC3  */
#line 1702 "MachineIndependent/glslang.y"
           {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setVector(3);
    }
#line 6779 "MachineIndependent/glslang_tab.cpp"
    break;

  case 213: /* type_specifier_nonarray: VEC4  */
#line 1707 "MachineIndependent/glslang.y"
           {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setVector(4);
    }
#line 6789 "MachineIndependent/glslang_tab.cpp"
    break;

  case 214: /* type_specifier_nonarray: BVEC2  */
#line 1712 "MachineIndependent/glslang.y"
            {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtBool;
        (yyval.interm.type).setVector(2);
    }
#line 6799 "MachineIndependent/glslang_tab.cpp"
    break;

  case 215: /* type_specifier_nonarray: BVEC3  */
#line 1717 "MachineIndependent/glslang.y"
            {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtBool;
        (yyval.interm.type).setVector(3);
    }
#line 6809 "MachineIndependent/glslang_tab.cpp"
    break;

  case 216: /* type_specifier_nonarray: BVEC4  */
#line 1722 "MachineIndependent/glslang.y"
            {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtBool;
        (yyval.interm.type).setVector(4);
    }
#line 6819 "MachineIndependent/glslang_tab.cpp"
    break;

  case 217: /* type_specifier_nonarray: IVEC2  */
#line 1727 "MachineIndependent/glslang.y"
            {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtInt;
        (yyval.interm.type).setVector(2);
    }
#line 6829 "MachineIndependent/glslang_tab.cpp"
    break;

  case 218: /* type_specifier_nonarray: IVEC3  */
#line 1732 "MachineIndependent/glslang.y"
            {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtInt;
        (yyval.interm.type).setVector(3);
    }
#line 6839 "MachineIndependent/glslang_tab.cpp"
    break;

  case 219: /* type_specifier_nonarray: IVEC4  */
#line 1737 "MachineIndependent/glslang.y"
            {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtInt;
        (yyval.interm.type).setVector(4);
    }
#line 6849 "MachineIndependent/glslang_tab.cpp"
    break;

  case 220: /* type_specifier_nonarray: UVEC2  */
#line 1742 "MachineIndependent/glslang.y"
            {
        parseContext.fullIntegerCheck((yyvsp[0].lex).loc, "unsigned integer vector");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtUint;
        (yyval.interm.type).setVector(2);
    }
#line 6860 "MachineIndependent/glslang_tab.cpp"
    break;

  case 221: /* type_specifier_nonarray: UVEC3  */
#line 1748 "MachineIndependent/glslang.y"
            {
        parseContext.fullIntegerCheck((yyvsp[0].lex).loc, "unsigned integer vector");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtUint;
        (yyval.interm.type).setVector(3);
    }
#line 6871 "MachineIndependent/glslang_tab.cpp"
    break;

  case 222: /* type_specifier_nonarray: UVEC4  */
#line 1754 "MachineIndependent/glslang.y"
            {
        parseContext.fullIntegerCheck((yyvsp[0].lex).loc, "unsigned integer vector");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtUint;
        (yyval.interm.type).setVector(4);
    }
#line 6882 "MachineIndependent/glslang_tab.cpp"
    break;

  case 223: /* type_specifier_nonarray: MAT2  */
#line 1760 "MachineIndependent/glslang.y"
           {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setMatrix(2, 2);
    }
#line 6892 "MachineIndependent/glslang_tab.cpp"
    break;

  case 224: /* type_specifier_nonarray: MAT3  */
#line 1765 "MachineIndependent/glslang.y"
           {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setMatrix(3, 3);
    }
#line 6902 "MachineIndependent/glslang_tab.cpp"
    break;

  case 225: /* type_specifier_nonarray: MAT4  */
#line 1770 "MachineIndependent/glslang.y"
           {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setMatrix(4, 4);
    }
#line 6912 "MachineIndependent/glslang_tab.cpp"
    break;

  case 226: /* type_specifier_nonarray: MAT2X2  */
#line 1775 "MachineIndependent/glslang.y"
             {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setMatrix(2, 2);
    }
#line 6922 "MachineIndependent/glslang_tab.cpp"
    break;

  case 227: /* type_specifier_nonarray: MAT2X3  */
#line 1780 "MachineIndependent/glslang.y"
             {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setMatrix(2, 3);
    }
#line 6932 "MachineIndependent/glslang_tab.cpp"
    break;

  case 228: /* type_specifier_nonarray: MAT2X4  */
#line 1785 "MachineIndependent/glslang.y"
             {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setMatrix(2, 4);
    }
#line 6942 "MachineIndependent/glslang_tab.cpp"
    break;

  case 229: /* type_specifier_nonarray: MAT3X2  */
#line 1790 "MachineIndependent/glslang.y"
             {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setMatrix(3, 2);
    }
#line 6952 "MachineIndependent/glslang_tab.cpp"
    break;

  case 230: /* type_specifier_nonarray: MAT3X3  */
#line 1795 "MachineIndependent/glslang.y"
             {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setMatrix(3, 3);
    }
#line 6962 "MachineIndependent/glslang_tab.cpp"
    break;

  case 231: /* type_specifier_nonarray: MAT3X4  */
#line 1800 "MachineIndependent/glslang.y"
             {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setMatrix(3, 4);
    }
#line 6972 "MachineIndependent/glslang_tab.cpp"
    break;

  case 232: /* type_specifier_nonarray: MAT4X2  */
#line 1805 "MachineIndependent/glslang.y"
             {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setMatrix(4, 2);
    }
#line 6982 "MachineIndependent/glslang_tab.cpp"
    break;

  case 233: /* type_specifier_nonarray: MAT4X3  */
#line 1810 "MachineIndependent/glslang.y"
             {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setMatrix(4, 3);
    }
#line 6992 "MachineIndependent/glslang_tab.cpp"
    break;

  case 234: /* type_specifier_nonarray: MAT4X4  */
#line 1815 "MachineIndependent/glslang.y"
             {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setMatrix(4, 4);
    }
#line 7002 "MachineIndependent/glslang_tab.cpp"
    break;

  case 235: /* type_specifier_nonarray: DOUBLE  */
#line 1821 "MachineIndependent/glslang.y"
             {
        parseContext.requireProfile((yyvsp[0].lex).loc, ECoreProfile | ECompatibilityProfile, "double");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck((yyvsp[0].lex).loc, "double");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
    }
#line 7014 "MachineIndependent/glslang_tab.cpp"
    break;

  case 236: /* type_specifier_nonarray: FLOAT16_T  */
#line 1828 "MachineIndependent/glslang.y"
                {
        parseContext.float16ScalarVectorCheck((yyvsp[0].lex).loc, "float16_t", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat16;
    }
#line 7024 "MachineIndependent/glslang_tab.cpp"
    break;

  case 237: /* type_specifier_nonarray: FLOAT32_T  */
#line 1833 "MachineIndependent/glslang.y"
                {
        parseContext.explicitFloat32Check((yyvsp[0].lex).loc, "float32_t", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
    }
#line 7034 "MachineIndependent/glslang_tab.cpp"
    break;

  case 238: /* type_specifier_nonarray: FLOAT64_T  */
#line 1838 "MachineIndependent/glslang.y"
                {
        parseContext.explicitFloat64Check((yyvsp[0].lex).loc, "float64_t", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
    }
#line 7044 "MachineIndependent/glslang_tab.cpp"
    break;

  case 239: /* type_specifier_nonarray: INT8_T  */
#line 1843 "MachineIndependent/glslang.y"
             {
        parseContext.int8ScalarVectorCheck((yyvsp[0].lex).loc, "8-bit signed integer", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtInt8;
    }
#line 7054 "MachineIndependent/glslang_tab.cpp"
    break;

  case 240: /* type_specifier_nonarray: UINT8_T  */
#line 1848 "MachineIndependent/glslang.y"
              {
        parseContext.int8ScalarVectorCheck((yyvsp[0].lex).loc, "8-bit unsigned integer", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtUint8;
    }
#line 7064 "MachineIndependent/glslang_tab.cpp"
    break;

  case 241: /* type_specifier_nonarray: INT16_T  */
#line 1853 "MachineIndependent/glslang.y"
              {
        parseContext.int16ScalarVectorCheck((yyvsp[0].lex).loc, "16-bit signed integer", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtInt16;
    }
#line 7074 "MachineIndependent/glslang_tab.cpp"
    break;

  case 242: /* type_specifier_nonarray: UINT16_T  */
#line 1858 "MachineIndependent/glslang.y"
               {
        parseContext.int16ScalarVectorCheck((yyvsp[0].lex).loc, "16-bit unsigned integer", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtUint16;
    }
#line 7084 "MachineIndependent/glslang_tab.cpp"
    break;

  case 243: /* type_specifier_nonarray: INT32_T  */
#line 1863 "MachineIndependent/glslang.y"
              {
        parseContext.explicitInt32Check((yyvsp[0].lex).loc, "32-bit signed integer", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtInt;
    }
#line 7094 "MachineIndependent/glslang_tab.cpp"
    break;

  case 244: /* type_specifier_nonarray: UINT32_T  */
#line 1868 "MachineIndependent/glslang.y"
               {
        parseContext.explicitInt32Check((yyvsp[0].lex).loc, "32-bit unsigned integer", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtUint;
    }
#line 7104 "MachineIndependent/glslang_tab.cpp"
    break;

  case 245: /* type_specifier_nonarray: INT64_T  */
#line 1873 "MachineIndependent/glslang.y"
              {
        parseContext.int64Check((yyvsp[0].lex).loc, "64-bit integer", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtInt64;
    }
#line 7114 "MachineIndependent/glslang_tab.cpp"
    break;

  case 246: /* type_specifier_nonarray: UINT64_T  */
#line 1878 "MachineIndependent/glslang.y"
               {
        parseContext.int64Check((yyvsp[0].lex).loc, "64-bit unsigned integer", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtUint64;
    }
#line 7124 "MachineIndependent/glslang_tab.cpp"
    break;

  case 247: /* type_specifier_nonarray: DVEC2  */
#line 1883 "MachineIndependent/glslang.y"
            {
        parseContext.requireProfile((yyvsp[0].lex).loc, ECoreProfile | ECompatibilityProfile, "double vector");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck((yyvsp[0].lex).loc, "double vector");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setVector(2);
    }
#line 7137 "MachineIndependent/glslang_tab.cpp"
    break;

  case 248: /* type_specifier_nonarray: DVEC3  */
#line 1891 "MachineIndependent/glslang.y"
            {
        parseContext.requireProfile((yyvsp[0].lex).loc, ECoreProfile | ECompatibilityProfile, "double vector");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck((yyvsp[0].lex).loc, "double vector");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setVector(3);
    }
#line 7150 "MachineIndependent/glslang_tab.cpp"
    break;

  case 249: /* type_specifier_nonarray: DVEC4  */
#line 1899 "MachineIndependent/glslang.y"
            {
        parseContext.requireProfile((yyvsp[0].lex).loc, ECoreProfile | ECompatibilityProfile, "double vector");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck((yyvsp[0].lex).loc, "double vector");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setVector(4);
    }
#line 7163 "MachineIndependent/glslang_tab.cpp"
    break;

  case 250: /* type_specifier_nonarray: F16VEC2  */
#line 1907 "MachineIndependent/glslang.y"
              {
        parseContext.float16ScalarVectorCheck((yyvsp[0].lex).loc, "half float vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat16;
        (yyval.interm.type).setVector(2);
    }
#line 7174 "MachineIndependent/glslang_tab.cpp"
    break;

  case 251: /* type_specifier_nonarray: F16VEC3  */
#line 1913 "MachineIndependent/glslang.y"
              {
        parseContext.float16ScalarVectorCheck((yyvsp[0].lex).loc, "half float vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat16;
        (yyval.interm.type).setVector(3);
    }
#line 7185 "MachineIndependent/glslang_tab.cpp"
    break;

  case 252: /* type_specifier_nonarray: F16VEC4  */
#line 1919 "MachineIndependent/glslang.y"
              {
        parseContext.float16ScalarVectorCheck((yyvsp[0].lex).loc, "half float vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat16;
        (yyval.interm.type).setVector(4);
    }
#line 7196 "MachineIndependent/glslang_tab.cpp"
    break;

  case 253: /* type_specifier_nonarray: F32VEC2  */
#line 1925 "MachineIndependent/glslang.y"
              {
        parseContext.explicitFloat32Check((yyvsp[0].lex).loc, "float32_t vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setVector(2);
    }
#line 7207 "MachineIndependent/glslang_tab.cpp"
    break;

  case 254: /* type_specifier_nonarray: F32VEC3  */
#line 1931 "MachineIndependent/glslang.y"
              {
        parseContext.explicitFloat32Check((yyvsp[0].lex).loc, "float32_t vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setVector(3);
    }
#line 7218 "MachineIndependent/glslang_tab.cpp"
    break;

  case 255: /* type_specifier_nonarray: F32VEC4  */
#line 1937 "MachineIndependent/glslang.y"
              {
        parseContext.explicitFloat32Check((yyvsp[0].lex).loc, "float32_t vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setVector(4);
    }
#line 7229 "MachineIndependent/glslang_tab.cpp"
    break;

  case 256: /* type_specifier_nonarray: F64VEC2  */
#line 1943 "MachineIndependent/glslang.y"
              {
        parseContext.explicitFloat64Check((yyvsp[0].lex).loc, "float64_t vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setVector(2);
    }
#line 7240 "MachineIndependent/glslang_tab.cpp"
    break;

  case 257: /* type_specifier_nonarray: F64VEC3  */
#line 1949 "MachineIndependent/glslang.y"
              {
        parseContext.explicitFloat64Check((yyvsp[0].lex).loc, "float64_t vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setVector(3);
    }
#line 7251 "MachineIndependent/glslang_tab.cpp"
    break;

  case 258: /* type_specifier_nonarray: F64VEC4  */
#line 1955 "MachineIndependent/glslang.y"
              {
        parseContext.explicitFloat64Check((yyvsp[0].lex).loc, "float64_t vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setVector(4);
    }
#line 7262 "MachineIndependent/glslang_tab.cpp"
    break;

  case 259: /* type_specifier_nonarray: I8VEC2  */
#line 1961 "MachineIndependent/glslang.y"
             {
        parseContext.int8ScalarVectorCheck((yyvsp[0].lex).loc, "8-bit signed integer vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtInt8;
        (yyval.interm.type).setVector(2);
    }
#line 7273 "MachineIndependent/glslang_tab.cpp"
    break;

  case 260: /* type_specifier_nonarray: I8VEC3  */
#line 1967 "MachineIndependent/glslang.y"
             {
        parseContext.int8ScalarVectorCheck((yyvsp[0].lex).loc, "8-bit signed integer vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtInt8;
        (yyval.interm.type).setVector(3);
    }
#line 7284 "MachineIndependent/glslang_tab.cpp"
    break;

  case 261: /* type_specifier_nonarray: I8VEC4  */
#line 1973 "MachineIndependent/glslang.y"
             {
        parseContext.int8ScalarVectorCheck((yyvsp[0].lex).loc, "8-bit signed integer vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtInt8;
        (yyval.interm.type).setVector(4);
    }
#line 7295 "MachineIndependent/glslang_tab.cpp"
    break;

  case 262: /* type_specifier_nonarray: I16VEC2  */
#line 1979 "MachineIndependent/glslang.y"
              {
        parseContext.int16ScalarVectorCheck((yyvsp[0].lex).loc, "16-bit signed integer vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtInt16;
        (yyval.interm.type).setVector(2);
    }
#line 7306 "MachineIndependent/glslang_tab.cpp"
    break;

  case 263: /* type_specifier_nonarray: I16VEC3  */
#line 1985 "MachineIndependent/glslang.y"
              {
        parseContext.int16ScalarVectorCheck((yyvsp[0].lex).loc, "16-bit signed integer vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtInt16;
        (yyval.interm.type).setVector(3);
    }
#line 7317 "MachineIndependent/glslang_tab.cpp"
    break;

  case 264: /* type_specifier_nonarray: I16VEC4  */
#line 1991 "MachineIndependent/glslang.y"
              {
        parseContext.int16ScalarVectorCheck((yyvsp[0].lex).loc, "16-bit signed integer vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtInt16;
        (yyval.interm.type).setVector(4);
    }
#line 7328 "MachineIndependent/glslang_tab.cpp"
    break;

  case 265: /* type_specifier_nonarray: I32VEC2  */
#line 1997 "MachineIndependent/glslang.y"
              {
        parseContext.explicitInt32Check((yyvsp[0].lex).loc, "32-bit signed integer vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtInt;
        (yyval.interm.type).setVector(2);
    }
#line 7339 "MachineIndependent/glslang_tab.cpp"
    break;

  case 266: /* type_specifier_nonarray: I32VEC3  */
#line 2003 "MachineIndependent/glslang.y"
              {
        parseContext.explicitInt32Check((yyvsp[0].lex).loc, "32-bit signed integer vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtInt;
        (yyval.interm.type).setVector(3);
    }
#line 7350 "MachineIndependent/glslang_tab.cpp"
    break;

  case 267: /* type_specifier_nonarray: I32VEC4  */
#line 2009 "MachineIndependent/glslang.y"
              {
        parseContext.explicitInt32Check((yyvsp[0].lex).loc, "32-bit signed integer vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtInt;
        (yyval.interm.type).setVector(4);
    }
#line 7361 "MachineIndependent/glslang_tab.cpp"
    break;

  case 268: /* type_specifier_nonarray: I64VEC2  */
#line 2015 "MachineIndependent/glslang.y"
              {
        parseContext.int64Check((yyvsp[0].lex).loc, "64-bit integer vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtInt64;
        (yyval.interm.type).setVector(2);
    }
#line 7372 "MachineIndependent/glslang_tab.cpp"
    break;

  case 269: /* type_specifier_nonarray: I64VEC3  */
#line 2021 "MachineIndependent/glslang.y"
              {
        parseContext.int64Check((yyvsp[0].lex).loc, "64-bit integer vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtInt64;
        (yyval.interm.type).setVector(3);
    }
#line 7383 "MachineIndependent/glslang_tab.cpp"
    break;

  case 270: /* type_specifier_nonarray: I64VEC4  */
#line 2027 "MachineIndependent/glslang.y"
              {
        parseContext.int64Check((yyvsp[0].lex).loc, "64-bit integer vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtInt64;
        (yyval.interm.type).setVector(4);
    }
#line 7394 "MachineIndependent/glslang_tab.cpp"
    break;

  case 271: /* type_specifier_nonarray: U8VEC2  */
#line 2033 "MachineIndependent/glslang.y"
             {
        parseContext.int8ScalarVectorCheck((yyvsp[0].lex).loc, "8-bit unsigned integer vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtUint8;
        (yyval.interm.type).setVector(2);
    }
#line 7405 "MachineIndependent/glslang_tab.cpp"
    break;

  case 272: /* type_specifier_nonarray: U8VEC3  */
#line 2039 "MachineIndependent/glslang.y"
             {
        parseContext.int8ScalarVectorCheck((yyvsp[0].lex).loc, "8-bit unsigned integer vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtUint8;
        (yyval.interm.type).setVector(3);
    }
#line 7416 "MachineIndependent/glslang_tab.cpp"
    break;

  case 273: /* type_specifier_nonarray: U8VEC4  */
#line 2045 "MachineIndependent/glslang.y"
             {
        parseContext.int8ScalarVectorCheck((yyvsp[0].lex).loc, "8-bit unsigned integer vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtUint8;
        (yyval.interm.type).setVector(4);
    }
#line 7427 "MachineIndependent/glslang_tab.cpp"
    break;

  case 274: /* type_specifier_nonarray: U16VEC2  */
#line 2051 "MachineIndependent/glslang.y"
              {
        parseContext.int16ScalarVectorCheck((yyvsp[0].lex).loc, "16-bit unsigned integer vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtUint16;
        (yyval.interm.type).setVector(2);
    }
#line 7438 "MachineIndependent/glslang_tab.cpp"
    break;

  case 275: /* type_specifier_nonarray: U16VEC3  */
#line 2057 "MachineIndependent/glslang.y"
              {
        parseContext.int16ScalarVectorCheck((yyvsp[0].lex).loc, "16-bit unsigned integer vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtUint16;
        (yyval.interm.type).setVector(3);
    }
#line 7449 "MachineIndependent/glslang_tab.cpp"
    break;

  case 276: /* type_specifier_nonarray: U16VEC4  */
#line 2063 "MachineIndependent/glslang.y"
              {
        parseContext.int16ScalarVectorCheck((yyvsp[0].lex).loc, "16-bit unsigned integer vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtUint16;
        (yyval.interm.type).setVector(4);
    }
#line 7460 "MachineIndependent/glslang_tab.cpp"
    break;

  case 277: /* type_specifier_nonarray: U32VEC2  */
#line 2069 "MachineIndependent/glslang.y"
              {
        parseContext.explicitInt32Check((yyvsp[0].lex).loc, "32-bit unsigned integer vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtUint;
        (yyval.interm.type).setVector(2);
    }
#line 7471 "MachineIndependent/glslang_tab.cpp"
    break;

  case 278: /* type_specifier_nonarray: U32VEC3  */
#line 2075 "MachineIndependent/glslang.y"
              {
        parseContext.explicitInt32Check((yyvsp[0].lex).loc, "32-bit unsigned integer vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtUint;
        (yyval.interm.type).setVector(3);
    }
#line 7482 "MachineIndependent/glslang_tab.cpp"
    break;

  case 279: /* type_specifier_nonarray: U32VEC4  */
#line 2081 "MachineIndependent/glslang.y"
              {
        parseContext.explicitInt32Check((yyvsp[0].lex).loc, "32-bit unsigned integer vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtUint;
        (yyval.interm.type).setVector(4);
    }
#line 7493 "MachineIndependent/glslang_tab.cpp"
    break;

  case 280: /* type_specifier_nonarray: U64VEC2  */
#line 2087 "MachineIndependent/glslang.y"
              {
        parseContext.int64Check((yyvsp[0].lex).loc, "64-bit unsigned integer vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtUint64;
        (yyval.interm.type).setVector(2);
    }
#line 7504 "MachineIndependent/glslang_tab.cpp"
    break;

  case 281: /* type_specifier_nonarray: U64VEC3  */
#line 2093 "MachineIndependent/glslang.y"
              {
        parseContext.int64Check((yyvsp[0].lex).loc, "64-bit unsigned integer vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtUint64;
        (yyval.interm.type).setVector(3);
    }
#line 7515 "MachineIndependent/glslang_tab.cpp"
    break;

  case 282: /* type_specifier_nonarray: U64VEC4  */
#line 2099 "MachineIndependent/glslang.y"
              {
        parseContext.int64Check((yyvsp[0].lex).loc, "64-bit unsigned integer vector", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtUint64;
        (yyval.interm.type).setVector(4);
    }
#line 7526 "MachineIndependent/glslang_tab.cpp"
    break;

  case 283: /* type_specifier_nonarray: DMAT2  */
#line 2105 "MachineIndependent/glslang.y"
            {
        parseContext.requireProfile((yyvsp[0].lex).loc, ECoreProfile | ECompatibilityProfile, "double matrix");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck((yyvsp[0].lex).loc, "double matrix");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setMatrix(2, 2);
    }
#line 7539 "MachineIndependent/glslang_tab.cpp"
    break;

  case 284: /* type_specifier_nonarray: DMAT3  */
#line 2113 "MachineIndependent/glslang.y"
            {
        parseContext.requireProfile((yyvsp[0].lex).loc, ECoreProfile | ECompatibilityProfile, "double matrix");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck((yyvsp[0].lex).loc, "double matrix");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setMatrix(3, 3);
    }
#line 7552 "MachineIndependent/glslang_tab.cpp"
    break;

  case 285: /* type_specifier_nonarray: DMAT4  */
#line 2121 "MachineIndependent/glslang.y"
            {
        parseContext.requireProfile((yyvsp[0].lex).loc, ECoreProfile | ECompatibilityProfile, "double matrix");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck((yyvsp[0].lex).loc, "double matrix");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setMatrix(4, 4);
    }
#line 7565 "MachineIndependent/glslang_tab.cpp"
    break;

  case 286: /* type_specifier_nonarray: DMAT2X2  */
#line 2129 "MachineIndependent/glslang.y"
              {
        parseContext.requireProfile((yyvsp[0].lex).loc, ECoreProfile | ECompatibilityProfile, "double matrix");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck((yyvsp[0].lex).loc, "double matrix");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setMatrix(2, 2);
    }
#line 7578 "MachineIndependent/glslang_tab.cpp"
    break;

  case 287: /* type_specifier_nonarray: DMAT2X3  */
#line 2137 "MachineIndependent/glslang.y"
              {
        parseContext.requireProfile((yyvsp[0].lex).loc, ECoreProfile | ECompatibilityProfile, "double matrix");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck((yyvsp[0].lex).loc, "double matrix");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setMatrix(2, 3);
    }
#line 7591 "MachineIndependent/glslang_tab.cpp"
    break;

  case 288: /* type_specifier_nonarray: DMAT2X4  */
#line 2145 "MachineIndependent/glslang.y"
              {
        parseContext.requireProfile((yyvsp[0].lex).loc, ECoreProfile | ECompatibilityProfile, "double matrix");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck((yyvsp[0].lex).loc, "double matrix");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setMatrix(2, 4);
    }
#line 7604 "MachineIndependent/glslang_tab.cpp"
    break;

  case 289: /* type_specifier_nonarray: DMAT3X2  */
#line 2153 "MachineIndependent/glslang.y"
              {
        parseContext.requireProfile((yyvsp[0].lex).loc, ECoreProfile | ECompatibilityProfile, "double matrix");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck((yyvsp[0].lex).loc, "double matrix");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setMatrix(3, 2);
    }
#line 7617 "MachineIndependent/glslang_tab.cpp"
    break;

  case 290: /* type_specifier_nonarray: DMAT3X3  */
#line 2161 "MachineIndependent/glslang.y"
              {
        parseContext.requireProfile((yyvsp[0].lex).loc, ECoreProfile | ECompatibilityProfile, "double matrix");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck((yyvsp[0].lex).loc, "double matrix");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setMatrix(3, 3);
    }
#line 7630 "MachineIndependent/glslang_tab.cpp"
    break;

  case 291: /* type_specifier_nonarray: DMAT3X4  */
#line 2169 "MachineIndependent/glslang.y"
              {
        parseContext.requireProfile((yyvsp[0].lex).loc, ECoreProfile | ECompatibilityProfile, "double matrix");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck((yyvsp[0].lex).loc, "double matrix");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setMatrix(3, 4);
    }
#line 7643 "MachineIndependent/glslang_tab.cpp"
    break;

  case 292: /* type_specifier_nonarray: DMAT4X2  */
#line 2177 "MachineIndependent/glslang.y"
              {
        parseContext.requireProfile((yyvsp[0].lex).loc, ECoreProfile | ECompatibilityProfile, "double matrix");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck((yyvsp[0].lex).loc, "double matrix");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setMatrix(4, 2);
    }
#line 7656 "MachineIndependent/glslang_tab.cpp"
    break;

  case 293: /* type_specifier_nonarray: DMAT4X3  */
#line 2185 "MachineIndependent/glslang.y"
              {
        parseContext.requireProfile((yyvsp[0].lex).loc, ECoreProfile | ECompatibilityProfile, "double matrix");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck((yyvsp[0].lex).loc, "double matrix");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setMatrix(4, 3);
    }
#line 7669 "MachineIndependent/glslang_tab.cpp"
    break;

  case 294: /* type_specifier_nonarray: DMAT4X4  */
#line 2193 "MachineIndependent/glslang.y"
              {
        parseContext.requireProfile((yyvsp[0].lex).loc, ECoreProfile | ECompatibilityProfile, "double matrix");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck((yyvsp[0].lex).loc, "double matrix");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setMatrix(4, 4);
    }
#line 7682 "MachineIndependent/glslang_tab.cpp"
    break;

  case 295: /* type_specifier_nonarray: F16MAT2  */
#line 2201 "MachineIndependent/glslang.y"
              {
        parseContext.float16Check((yyvsp[0].lex).loc, "half float matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat16;
        (yyval.interm.type).setMatrix(2, 2);
    }
#line 7693 "MachineIndependent/glslang_tab.cpp"
    break;

  case 296: /* type_specifier_nonarray: F16MAT3  */
#line 2207 "MachineIndependent/glslang.y"
              {
        parseContext.float16Check((yyvsp[0].lex).loc, "half float matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat16;
        (yyval.interm.type).setMatrix(3, 3);
    }
#line 7704 "MachineIndependent/glslang_tab.cpp"
    break;

  case 297: /* type_specifier_nonarray: F16MAT4  */
#line 2213 "MachineIndependent/glslang.y"
              {
        parseContext.float16Check((yyvsp[0].lex).loc, "half float matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat16;
        (yyval.interm.type).setMatrix(4, 4);
    }
#line 7715 "MachineIndependent/glslang_tab.cpp"
    break;

  case 298: /* type_specifier_nonarray: F16MAT2X2  */
#line 2219 "MachineIndependent/glslang.y"
                {
        parseContext.float16Check((yyvsp[0].lex).loc, "half float matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat16;
        (yyval.interm.type).setMatrix(2, 2);
    }
#line 7726 "MachineIndependent/glslang_tab.cpp"
    break;

  case 299: /* type_specifier_nonarray: F16MAT2X3  */
#line 2225 "MachineIndependent/glslang.y"
                {
        parseContext.float16Check((yyvsp[0].lex).loc, "half float matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat16;
        (yyval.interm.type).setMatrix(2, 3);
    }
#line 7737 "MachineIndependent/glslang_tab.cpp"
    break;

  case 300: /* type_specifier_nonarray: F16MAT2X4  */
#line 2231 "MachineIndependent/glslang.y"
                {
        parseContext.float16Check((yyvsp[0].lex).loc, "half float matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat16;
        (yyval.interm.type).setMatrix(2, 4);
    }
#line 7748 "MachineIndependent/glslang_tab.cpp"
    break;

  case 301: /* type_specifier_nonarray: F16MAT3X2  */
#line 2237 "MachineIndependent/glslang.y"
                {
        parseContext.float16Check((yyvsp[0].lex).loc, "half float matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat16;
        (yyval.interm.type).setMatrix(3, 2);
    }
#line 7759 "MachineIndependent/glslang_tab.cpp"
    break;

  case 302: /* type_specifier_nonarray: F16MAT3X3  */
#line 2243 "MachineIndependent/glslang.y"
                {
        parseContext.float16Check((yyvsp[0].lex).loc, "half float matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat16;
        (yyval.interm.type).setMatrix(3, 3);
    }
#line 7770 "MachineIndependent/glslang_tab.cpp"
    break;

  case 303: /* type_specifier_nonarray: F16MAT3X4  */
#line 2249 "MachineIndependent/glslang.y"
                {
        parseContext.float16Check((yyvsp[0].lex).loc, "half float matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat16;
        (yyval.interm.type).setMatrix(3, 4);
    }
#line 7781 "MachineIndependent/glslang_tab.cpp"
    break;

  case 304: /* type_specifier_nonarray: F16MAT4X2  */
#line 2255 "MachineIndependent/glslang.y"
                {
        parseContext.float16Check((yyvsp[0].lex).loc, "half float matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat16;
        (yyval.interm.type).setMatrix(4, 2);
    }
#line 7792 "MachineIndependent/glslang_tab.cpp"
    break;

  case 305: /* type_specifier_nonarray: F16MAT4X3  */
#line 2261 "MachineIndependent/glslang.y"
                {
        parseContext.float16Check((yyvsp[0].lex).loc, "half float matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat16;
        (yyval.interm.type).setMatrix(4, 3);
    }
#line 7803 "MachineIndependent/glslang_tab.cpp"
    break;

  case 306: /* type_specifier_nonarray: F16MAT4X4  */
#line 2267 "MachineIndependent/glslang.y"
                {
        parseContext.float16Check((yyvsp[0].lex).loc, "half float matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat16;
        (yyval.interm.type).setMatrix(4, 4);
    }
#line 7814 "MachineIndependent/glslang_tab.cpp"
    break;

  case 307: /* type_specifier_nonarray: F32MAT2  */
#line 2273 "MachineIndependent/glslang.y"
              {
        parseContext.explicitFloat32Check((yyvsp[0].lex).loc, "float32_t matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setMatrix(2, 2);
    }
#line 7825 "MachineIndependent/glslang_tab.cpp"
    break;

  case 308: /* type_specifier_nonarray: F32MAT3  */
#line 2279 "MachineIndependent/glslang.y"
              {
        parseContext.explicitFloat32Check((yyvsp[0].lex).loc, "float32_t matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setMatrix(3, 3);
    }
#line 7836 "MachineIndependent/glslang_tab.cpp"
    break;

  case 309: /* type_specifier_nonarray: F32MAT4  */
#line 2285 "MachineIndependent/glslang.y"
              {
        parseContext.explicitFloat32Check((yyvsp[0].lex).loc, "float32_t matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setMatrix(4, 4);
    }
#line 7847 "MachineIndependent/glslang_tab.cpp"
    break;

  case 310: /* type_specifier_nonarray: F32MAT2X2  */
#line 2291 "MachineIndependent/glslang.y"
                {
        parseContext.explicitFloat32Check((yyvsp[0].lex).loc, "float32_t matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setMatrix(2, 2);
    }
#line 7858 "MachineIndependent/glslang_tab.cpp"
    break;

  case 311: /* type_specifier_nonarray: F32MAT2X3  */
#line 2297 "MachineIndependent/glslang.y"
                {
        parseContext.explicitFloat32Check((yyvsp[0].lex).loc, "float32_t matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setMatrix(2, 3);
    }
#line 7869 "MachineIndependent/glslang_tab.cpp"
    break;

  case 312: /* type_specifier_nonarray: F32MAT2X4  */
#line 2303 "MachineIndependent/glslang.y"
                {
        parseContext.explicitFloat32Check((yyvsp[0].lex).loc, "float32_t matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setMatrix(2, 4);
    }
#line 7880 "MachineIndependent/glslang_tab.cpp"
    break;

  case 313: /* type_specifier_nonarray: F32MAT3X2  */
#line 2309 "MachineIndependent/glslang.y"
                {
        parseContext.explicitFloat32Check((yyvsp[0].lex).loc, "float32_t matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setMatrix(3, 2);
    }
#line 7891 "MachineIndependent/glslang_tab.cpp"
    break;

  case 314: /* type_specifier_nonarray: F32MAT3X3  */
#line 2315 "MachineIndependent/glslang.y"
                {
        parseContext.explicitFloat32Check((yyvsp[0].lex).loc, "float32_t matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setMatrix(3, 3);
    }
#line 7902 "MachineIndependent/glslang_tab.cpp"
    break;

  case 315: /* type_specifier_nonarray: F32MAT3X4  */
#line 2321 "MachineIndependent/glslang.y"
                {
        parseContext.explicitFloat32Check((yyvsp[0].lex).loc, "float32_t matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setMatrix(3, 4);
    }
#line 7913 "MachineIndependent/glslang_tab.cpp"
    break;

  case 316: /* type_specifier_nonarray: F32MAT4X2  */
#line 2327 "MachineIndependent/glslang.y"
                {
        parseContext.explicitFloat32Check((yyvsp[0].lex).loc, "float32_t matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setMatrix(4, 2);
    }
#line 7924 "MachineIndependent/glslang_tab.cpp"
    break;

  case 317: /* type_specifier_nonarray: F32MAT4X3  */
#line 2333 "MachineIndependent/glslang.y"
                {
        parseContext.explicitFloat32Check((yyvsp[0].lex).loc, "float32_t matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setMatrix(4, 3);
    }
#line 7935 "MachineIndependent/glslang_tab.cpp"
    break;

  case 318: /* type_specifier_nonarray: F32MAT4X4  */
#line 2339 "MachineIndependent/glslang.y"
                {
        parseContext.explicitFloat32Check((yyvsp[0].lex).loc, "float32_t matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).setMatrix(4, 4);
    }
#line 7946 "MachineIndependent/glslang_tab.cpp"
    break;

  case 319: /* type_specifier_nonarray: F64MAT2  */
#line 2345 "MachineIndependent/glslang.y"
              {
        parseContext.explicitFloat64Check((yyvsp[0].lex).loc, "float64_t matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setMatrix(2, 2);
    }
#line 7957 "MachineIndependent/glslang_tab.cpp"
    break;

  case 320: /* type_specifier_nonarray: F64MAT3  */
#line 2351 "MachineIndependent/glslang.y"
              {
        parseContext.explicitFloat64Check((yyvsp[0].lex).loc, "float64_t matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setMatrix(3, 3);
    }
#line 7968 "MachineIndependent/glslang_tab.cpp"
    break;

  case 321: /* type_specifier_nonarray: F64MAT4  */
#line 2357 "MachineIndependent/glslang.y"
              {
        parseContext.explicitFloat64Check((yyvsp[0].lex).loc, "float64_t matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setMatrix(4, 4);
    }
#line 7979 "MachineIndependent/glslang_tab.cpp"
    break;

  case 322: /* type_specifier_nonarray: F64MAT2X2  */
#line 2363 "MachineIndependent/glslang.y"
                {
        parseContext.explicitFloat64Check((yyvsp[0].lex).loc, "float64_t matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setMatrix(2, 2);
    }
#line 7990 "MachineIndependent/glslang_tab.cpp"
    break;

  case 323: /* type_specifier_nonarray: F64MAT2X3  */
#line 2369 "MachineIndependent/glslang.y"
                {
        parseContext.explicitFloat64Check((yyvsp[0].lex).loc, "float64_t matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setMatrix(2, 3);
    }
#line 8001 "MachineIndependent/glslang_tab.cpp"
    break;

  case 324: /* type_specifier_nonarray: F64MAT2X4  */
#line 2375 "MachineIndependent/glslang.y"
                {
        parseContext.explicitFloat64Check((yyvsp[0].lex).loc, "float64_t matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setMatrix(2, 4);
    }
#line 8012 "MachineIndependent/glslang_tab.cpp"
    break;

  case 325: /* type_specifier_nonarray: F64MAT3X2  */
#line 2381 "MachineIndependent/glslang.y"
                {
        parseContext.explicitFloat64Check((yyvsp[0].lex).loc, "float64_t matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setMatrix(3, 2);
    }
#line 8023 "MachineIndependent/glslang_tab.cpp"
    break;

  case 326: /* type_specifier_nonarray: F64MAT3X3  */
#line 2387 "MachineIndependent/glslang.y"
                {
        parseContext.explicitFloat64Check((yyvsp[0].lex).loc, "float64_t matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setMatrix(3, 3);
    }
#line 8034 "MachineIndependent/glslang_tab.cpp"
    break;

  case 327: /* type_specifier_nonarray: F64MAT3X4  */
#line 2393 "MachineIndependent/glslang.y"
                {
        parseContext.explicitFloat64Check((yyvsp[0].lex).loc, "float64_t matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setMatrix(3, 4);
    }
#line 8045 "MachineIndependent/glslang_tab.cpp"
    break;

  case 328: /* type_specifier_nonarray: F64MAT4X2  */
#line 2399 "MachineIndependent/glslang.y"
                {
        parseContext.explicitFloat64Check((yyvsp[0].lex).loc, "float64_t matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setMatrix(4, 2);
    }
#line 8056 "MachineIndependent/glslang_tab.cpp"
    break;

  case 329: /* type_specifier_nonarray: F64MAT4X3  */
#line 2405 "MachineIndependent/glslang.y"
                {
        parseContext.explicitFloat64Check((yyvsp[0].lex).loc, "float64_t matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setMatrix(4, 3);
    }
#line 8067 "MachineIndependent/glslang_tab.cpp"
    break;

  case 330: /* type_specifier_nonarray: F64MAT4X4  */
#line 2411 "MachineIndependent/glslang.y"
                {
        parseContext.explicitFloat64Check((yyvsp[0].lex).loc, "float64_t matrix", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtDouble;
        (yyval.interm.type).setMatrix(4, 4);
    }
#line 8078 "MachineIndependent/glslang_tab.cpp"
    break;

  case 331: /* type_specifier_nonarray: ACCSTRUCTNV  */
#line 2417 "MachineIndependent/glslang.y"
                  {
       (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
       (yyval.interm.type).basicType = EbtAccStruct;
    }
#line 8087 "MachineIndependent/glslang_tab.cpp"
    break;

  case 332: /* type_specifier_nonarray: ACCSTRUCTEXT  */
#line 2421 "MachineIndependent/glslang.y"
                   {
       (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
       (yyval.interm.type).basicType = EbtAccStruct;
    }
#line 8096 "MachineIndependent/glslang_tab.cpp"
    break;

  case 333: /* type_specifier_nonarray: RAYQUERYEXT  */
#line 2425 "MachineIndependent/glslang.y"
                  {
       (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
       (yyval.interm.type).basicType = EbtRayQuery;
    }
#line 8105 "MachineIndependent/glslang_tab.cpp"
    break;

  case 334: /* type_specifier_nonarray: ATOMIC_UINT  */
#line 2429 "MachineIndependent/glslang.y"
                  {
        parseContext.vulkanRemoved((yyvsp[0].lex).loc, "atomic counter types");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtAtomicUint;
    }
#line 8115 "MachineIndependent/glslang_tab.cpp"
    break;

  case 335: /* type_specifier_nonarray: SAMPLER1D  */
#line 2434 "MachineIndependent/glslang.y"
                {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat, Esd1D);
    }
#line 8125 "MachineIndependent/glslang_tab.cpp"
    break;

  case 336: /* type_specifier_nonarray: SAMPLER2D  */
#line 2440 "MachineIndependent/glslang.y"
                {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat, Esd2D);
    }
#line 8135 "MachineIndependent/glslang_tab.cpp"
    break;

  case 337: /* type_specifier_nonarray: SAMPLER3D  */
#line 2445 "MachineIndependent/glslang.y"
                {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat, Esd3D);
    }
#line 8145 "MachineIndependent/glslang_tab.cpp"
    break;

  case 338: /* type_specifier_nonarray: SAMPLERCUBE  */
#line 2450 "MachineIndependent/glslang.y"
                  {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat, EsdCube);
    }
#line 8155 "MachineIndependent/glslang_tab.cpp"
    break;

  case 339: /* type_specifier_nonarray: SAMPLER2DSHADOW  */
#line 2455 "MachineIndependent/glslang.y"
                      {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat, Esd2D, false, true);
    }
#line 8165 "MachineIndependent/glslang_tab.cpp"
    break;

  case 340: /* type_specifier_nonarray: SAMPLERCUBESHADOW  */
#line 2460 "MachineIndependent/glslang.y"
                        {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat, EsdCube, false, true);
    }
#line 8175 "MachineIndependent/glslang_tab.cpp"
    break;

  case 341: /* type_specifier_nonarray: SAMPLER2DARRAY  */
#line 2465 "MachineIndependent/glslang.y"
                     {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat, Esd2D, true);
    }
#line 8185 "MachineIndependent/glslang_tab.cpp"
    break;

  case 342: /* type_specifier_nonarray: SAMPLER2DARRAYSHADOW  */
#line 2470 "MachineIndependent/glslang.y"
                           {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat, Esd2D, true, true);
    }
#line 8195 "MachineIndependent/glslang_tab.cpp"
    break;

  case 343: /* type_specifier_nonarray: SAMPLER1DSHADOW  */
#line 2476 "MachineIndependent/glslang.y"
                      {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat, Esd1D, false, true);
    }
#line 8205 "MachineIndependent/glslang_tab.cpp"
    break;

  case 344: /* type_specifier_nonarray: SAMPLER1DARRAY  */
#line 2481 "MachineIndependent/glslang.y"
                     {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat, Esd1D, true);
    }
#line 8215 "MachineIndependent/glslang_tab.cpp"
    break;

  case 345: /* type_specifier_nonarray: SAMPLER1DARRAYSHADOW  */
#line 2486 "MachineIndependent/glslang.y"
                           {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat, Esd1D, true, true);
    }
#line 8225 "MachineIndependent/glslang_tab.cpp"
    break;

  case 346: /* type_specifier_nonarray: SAMPLERCUBEARRAY  */
#line 2491 "MachineIndependent/glslang.y"
                       {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat, EsdCube, true);
    }
#line 8235 "MachineIndependent/glslang_tab.cpp"
    break;

  case 347: /* type_specifier_nonarray: SAMPLERCUBEARRAYSHADOW  */
#line 2496 "MachineIndependent/glslang.y"
                             {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat, EsdCube, true, true);
    }
#line 8245 "MachineIndependent/glslang_tab.cpp"
    break;

  case 348: /* type_specifier_nonarray: F16SAMPLER1D  */
#line 2501 "MachineIndependent/glslang.y"
                   {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat16, Esd1D);
    }
#line 8256 "MachineIndependent/glslang_tab.cpp"
    break;

  case 349: /* type_specifier_nonarray: F16SAMPLER2D  */
#line 2507 "MachineIndependent/glslang.y"
                   {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat16, Esd2D);
    }
#line 8267 "MachineIndependent/glslang_tab.cpp"
    break;

  case 350: /* type_specifier_nonarray: F16SAMPLER3D  */
#line 2513 "MachineIndependent/glslang.y"
                   {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat16, Esd3D);
    }
#line 8278 "MachineIndependent/glslang_tab.cpp"
    break;

  case 351: /* type_specifier_nonarray: F16SAMPLERCUBE  */
#line 2519 "MachineIndependent/glslang.y"
                     {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat16, EsdCube);
    }
#line 8289 "MachineIndependent/glslang_tab.cpp"
    break;

  case 352: /* type_specifier_nonarray: F16SAMPLER1DSHADOW  */
#line 2525 "MachineIndependent/glslang.y"
                         {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat16, Esd1D, false, true);
    }
#line 8300 "MachineIndependent/glslang_tab.cpp"
    break;

  case 353: /* type_specifier_nonarray: F16SAMPLER2DSHADOW  */
#line 2531 "MachineIndependent/glslang.y"
                         {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat16, Esd2D, false, true);
    }
#line 8311 "MachineIndependent/glslang_tab.cpp"
    break;

  case 354: /* type_specifier_nonarray: F16SAMPLERCUBESHADOW  */
#line 2537 "MachineIndependent/glslang.y"
                           {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat16, EsdCube, false, true);
    }
#line 8322 "MachineIndependent/glslang_tab.cpp"
    break;

  case 355: /* type_specifier_nonarray: F16SAMPLER1DARRAY  */
#line 2543 "MachineIndependent/glslang.y"
                        {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat16, Esd1D, true);
    }
#line 8333 "MachineIndependent/glslang_tab.cpp"
    break;

  case 356: /* type_specifier_nonarray: F16SAMPLER2DARRAY  */
#line 2549 "MachineIndependent/glslang.y"
                        {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat16, Esd2D, true);
    }
#line 8344 "MachineIndependent/glslang_tab.cpp"
    break;

  case 357: /* type_specifier_nonarray: F16SAMPLER1DARRAYSHADOW  */
#line 2555 "MachineIndependent/glslang.y"
                              {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat16, Esd1D, true, true);
    }
#line 8355 "MachineIndependent/glslang_tab.cpp"
    break;

  case 358: /* type_specifier_nonarray: F16SAMPLER2DARRAYSHADOW  */
#line 2561 "MachineIndependent/glslang.y"
                              {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat16, Esd2D, true, true);
    }
#line 8366 "MachineIndependent/glslang_tab.cpp"
    break;

  case 359: /* type_specifier_nonarray: F16SAMPLERCUBEARRAY  */
#line 2567 "MachineIndependent/glslang.y"
                          {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat16, EsdCube, true);
    }
#line 8377 "MachineIndependent/glslang_tab.cpp"
    break;

  case 360: /* type_specifier_nonarray: F16SAMPLERCUBEARRAYSHADOW  */
#line 2573 "MachineIndependent/glslang.y"
                                {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat16, EsdCube, true, true);
    }
#line 8388 "MachineIndependent/glslang_tab.cpp"
    break;

  case 361: /* type_specifier_nonarray: ISAMPLER1D  */
#line 2579 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtInt, Esd1D);
    }
#line 8398 "MachineIndependent/glslang_tab.cpp"
    break;

  case 362: /* type_specifier_nonarray: ISAMPLER2D  */
#line 2585 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtInt, Esd2D);
    }
#line 8408 "MachineIndependent/glslang_tab.cpp"
    break;

  case 363: /* type_specifier_nonarray: ISAMPLER3D  */
#line 2590 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtInt, Esd3D);
    }
#line 8418 "MachineIndependent/glslang_tab.cpp"
    break;

  case 364: /* type_specifier_nonarray: ISAMPLERCUBE  */
#line 2595 "MachineIndependent/glslang.y"
                   {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtInt, EsdCube);
    }
#line 8428 "MachineIndependent/glslang_tab.cpp"
    break;

  case 365: /* type_specifier_nonarray: ISAMPLER2DARRAY  */
#line 2600 "MachineIndependent/glslang.y"
                      {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtInt, Esd2D, true);
    }
#line 8438 "MachineIndependent/glslang_tab.cpp"
    break;

  case 366: /* type_specifier_nonarray: USAMPLER2D  */
#line 2605 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtUint, Esd2D);
    }
#line 8448 "MachineIndependent/glslang_tab.cpp"
    break;

  case 367: /* type_specifier_nonarray: USAMPLER3D  */
#line 2610 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtUint, Esd3D);
    }
#line 8458 "MachineIndependent/glslang_tab.cpp"
    break;

  case 368: /* type_specifier_nonarray: USAMPLERCUBE  */
#line 2615 "MachineIndependent/glslang.y"
                   {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtUint, EsdCube);
    }
#line 8468 "MachineIndependent/glslang_tab.cpp"
    break;

  case 369: /* type_specifier_nonarray: ISAMPLER1DARRAY  */
#line 2621 "MachineIndependent/glslang.y"
                      {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtInt, Esd1D, true);
    }
#line 8478 "MachineIndependent/glslang_tab.cpp"
    break;

  case 370: /* type_specifier_nonarray: ISAMPLERCUBEARRAY  */
#line 2626 "MachineIndependent/glslang.y"
                        {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtInt, EsdCube, true);
    }
#line 8488 "MachineIndependent/glslang_tab.cpp"
    break;

  case 371: /* type_specifier_nonarray: USAMPLER1D  */
#line 2631 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtUint, Esd1D);
    }
#line 8498 "MachineIndependent/glslang_tab.cpp"
    break;

  case 372: /* type_specifier_nonarray: USAMPLER1DARRAY  */
#line 2636 "MachineIndependent/glslang.y"
                      {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtUint, Esd1D, true);
    }
#line 8508 "MachineIndependent/glslang_tab.cpp"
    break;

  case 373: /* type_specifier_nonarray: USAMPLERCUBEARRAY  */
#line 2641 "MachineIndependent/glslang.y"
                        {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtUint, EsdCube, true);
    }
#line 8518 "MachineIndependent/glslang_tab.cpp"
    break;

  case 374: /* type_specifier_nonarray: TEXTURECUBEARRAY  */
#line 2646 "MachineIndependent/glslang.y"
                       {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtFloat, EsdCube, true);
    }
#line 8528 "MachineIndependent/glslang_tab.cpp"
    break;

  case 375: /* type_specifier_nonarray: ITEXTURECUBEARRAY  */
#line 2651 "MachineIndependent/glslang.y"
                        {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtInt, EsdCube, true);
    }
#line 8538 "MachineIndependent/glslang_tab.cpp"
    break;

  case 376: /* type_specifier_nonarray: UTEXTURECUBEARRAY  */
#line 2656 "MachineIndependent/glslang.y"
                        {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtUint, EsdCube, true);
    }
#line 8548 "MachineIndependent/glslang_tab.cpp"
    break;

  case 377: /* type_specifier_nonarray: USAMPLER2DARRAY  */
#line 2662 "MachineIndependent/glslang.y"
                      {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtUint, Esd2D, true);
    }
#line 8558 "MachineIndependent/glslang_tab.cpp"
    break;

  case 378: /* type_specifier_nonarray: TEXTURE2D  */
#line 2667 "MachineIndependent/glslang.y"
                {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtFloat, Esd2D);
    }
#line 8568 "MachineIndependent/glslang_tab.cpp"
    break;

  case 379: /* type_specifier_nonarray: TEXTURE3D  */
#line 2672 "MachineIndependent/glslang.y"
                {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtFloat, Esd3D);
    }
#line 8578 "MachineIndependent/glslang_tab.cpp"
    break;

  case 380: /* type_specifier_nonarray: TEXTURE2DARRAY  */
#line 2677 "MachineIndependent/glslang.y"
                     {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtFloat, Esd2D, true);
    }
#line 8588 "MachineIndependent/glslang_tab.cpp"
    break;

  case 381: /* type_specifier_nonarray: TEXTURECUBE  */
#line 2682 "MachineIndependent/glslang.y"
                  {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtFloat, EsdCube);
    }
#line 8598 "MachineIndependent/glslang_tab.cpp"
    break;

  case 382: /* type_specifier_nonarray: ITEXTURE2D  */
#line 2687 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtInt, Esd2D);
    }
#line 8608 "MachineIndependent/glslang_tab.cpp"
    break;

  case 383: /* type_specifier_nonarray: ITEXTURE3D  */
#line 2692 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtInt, Esd3D);
    }
#line 8618 "MachineIndependent/glslang_tab.cpp"
    break;

  case 384: /* type_specifier_nonarray: ITEXTURECUBE  */
#line 2697 "MachineIndependent/glslang.y"
                   {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtInt, EsdCube);
    }
#line 8628 "MachineIndependent/glslang_tab.cpp"
    break;

  case 385: /* type_specifier_nonarray: ITEXTURE2DARRAY  */
#line 2702 "MachineIndependent/glslang.y"
                      {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtInt, Esd2D, true);
    }
#line 8638 "MachineIndependent/glslang_tab.cpp"
    break;

  case 386: /* type_specifier_nonarray: UTEXTURE2D  */
#line 2707 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtUint, Esd2D);
    }
#line 8648 "MachineIndependent/glslang_tab.cpp"
    break;

  case 387: /* type_specifier_nonarray: UTEXTURE3D  */
#line 2712 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtUint, Esd3D);
    }
#line 8658 "MachineIndependent/glslang_tab.cpp"
    break;

  case 388: /* type_specifier_nonarray: UTEXTURECUBE  */
#line 2717 "MachineIndependent/glslang.y"
                   {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtUint, EsdCube);
    }
#line 8668 "MachineIndependent/glslang_tab.cpp"
    break;

  case 389: /* type_specifier_nonarray: UTEXTURE2DARRAY  */
#line 2722 "MachineIndependent/glslang.y"
                      {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtUint, Esd2D, true);
    }
#line 8678 "MachineIndependent/glslang_tab.cpp"
    break;

  case 390: /* type_specifier_nonarray: SAMPLER  */
#line 2727 "MachineIndependent/glslang.y"
              {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setPureSampler(false);
    }
#line 8688 "MachineIndependent/glslang_tab.cpp"
    break;

  case 391: /* type_specifier_nonarray: SAMPLERSHADOW  */
#line 2732 "MachineIndependent/glslang.y"
                    {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setPureSampler(true);
    }
#line 8698 "MachineIndependent/glslang_tab.cpp"
    break;

  case 392: /* type_specifier_nonarray: SAMPLER2DRECT  */
#line 2738 "MachineIndependent/glslang.y"
                    {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat, EsdRect);
    }
#line 8708 "MachineIndependent/glslang_tab.cpp"
    break;

  case 393: /* type_specifier_nonarray: SAMPLER2DRECTSHADOW  */
#line 2743 "MachineIndependent/glslang.y"
                          {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat, EsdRect, false, true);
    }
#line 8718 "MachineIndependent/glslang_tab.cpp"
    break;

  case 394: /* type_specifier_nonarray: F16SAMPLER2DRECT  */
#line 2748 "MachineIndependent/glslang.y"
                       {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat16, EsdRect);
    }
#line 8729 "MachineIndependent/glslang_tab.cpp"
    break;

  case 395: /* type_specifier_nonarray: F16SAMPLER2DRECTSHADOW  */
#line 2754 "MachineIndependent/glslang.y"
                             {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat16, EsdRect, false, true);
    }
#line 8740 "MachineIndependent/glslang_tab.cpp"
    break;

  case 396: /* type_specifier_nonarray: ISAMPLER2DRECT  */
#line 2760 "MachineIndependent/glslang.y"
                     {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtInt, EsdRect);
    }
#line 8750 "MachineIndependent/glslang_tab.cpp"
    break;

  case 397: /* type_specifier_nonarray: USAMPLER2DRECT  */
#line 2765 "MachineIndependent/glslang.y"
                     {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtUint, EsdRect);
    }
#line 8760 "MachineIndependent/glslang_tab.cpp"
    break;

  case 398: /* type_specifier_nonarray: SAMPLERBUFFER  */
#line 2770 "MachineIndependent/glslang.y"
                    {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat, EsdBuffer);
    }
#line 8770 "MachineIndependent/glslang_tab.cpp"
    break;

  case 399: /* type_specifier_nonarray: F16SAMPLERBUFFER  */
#line 2775 "MachineIndependent/glslang.y"
                       {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat16, EsdBuffer);
    }
#line 8781 "MachineIndependent/glslang_tab.cpp"
    break;

  case 400: /* type_specifier_nonarray: ISAMPLERBUFFER  */
#line 2781 "MachineIndependent/glslang.y"
                     {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtInt, EsdBuffer);
    }
#line 8791 "MachineIndependent/glslang_tab.cpp"
    break;

  case 401: /* type_specifier_nonarray: USAMPLERBUFFER  */
#line 2786 "MachineIndependent/glslang.y"
                     {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtUint, EsdBuffer);
    }
#line 8801 "MachineIndependent/glslang_tab.cpp"
    break;

  case 402: /* type_specifier_nonarray: SAMPLER2DMS  */
#line 2791 "MachineIndependent/glslang.y"
                  {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat, Esd2D, false, false, true);
    }
#line 8811 "MachineIndependent/glslang_tab.cpp"
    break;

  case 403: /* type_specifier_nonarray: F16SAMPLER2DMS  */
#line 2796 "MachineIndependent/glslang.y"
                     {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat16, Esd2D, false, false, true);
    }
#line 8822 "MachineIndependent/glslang_tab.cpp"
    break;

  case 404: /* type_specifier_nonarray: ISAMPLER2DMS  */
#line 2802 "MachineIndependent/glslang.y"
                   {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtInt, Esd2D, false, false, true);
    }
#line 8832 "MachineIndependent/glslang_tab.cpp"
    break;

  case 405: /* type_specifier_nonarray: USAMPLER2DMS  */
#line 2807 "MachineIndependent/glslang.y"
                   {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtUint, Esd2D, false, false, true);
    }
#line 8842 "MachineIndependent/glslang_tab.cpp"
    break;

  case 406: /* type_specifier_nonarray: SAMPLER2DMSARRAY  */
#line 2812 "MachineIndependent/glslang.y"
                       {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat, Esd2D, true, false, true);
    }
#line 8852 "MachineIndependent/glslang_tab.cpp"
    break;

  case 407: /* type_specifier_nonarray: F16SAMPLER2DMSARRAY  */
#line 2817 "MachineIndependent/glslang.y"
                          {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat16, Esd2D, true, false, true);
    }
#line 8863 "MachineIndependent/glslang_tab.cpp"
    break;

  case 408: /* type_specifier_nonarray: ISAMPLER2DMSARRAY  */
#line 2823 "MachineIndependent/glslang.y"
                        {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtInt, Esd2D, true, false, true);
    }
#line 8873 "MachineIndependent/glslang_tab.cpp"
    break;

  case 409: /* type_specifier_nonarray: USAMPLER2DMSARRAY  */
#line 2828 "MachineIndependent/glslang.y"
                        {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtUint, Esd2D, true, false, true);
    }
#line 8883 "MachineIndependent/glslang_tab.cpp"
    break;

  case 410: /* type_specifier_nonarray: TEXTURE1D  */
#line 2833 "MachineIndependent/glslang.y"
                {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtFloat, Esd1D);
    }
#line 8893 "MachineIndependent/glslang_tab.cpp"
    break;

  case 411: /* type_specifier_nonarray: F16TEXTURE1D  */
#line 2838 "MachineIndependent/glslang.y"
                   {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float texture", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtFloat16, Esd1D);
    }
#line 8904 "MachineIndependent/glslang_tab.cpp"
    break;

  case 412: /* type_specifier_nonarray: F16TEXTURE2D  */
#line 2844 "MachineIndependent/glslang.y"
                   {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float texture", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtFloat16, Esd2D);
    }
#line 8915 "MachineIndependent/glslang_tab.cpp"
    break;

  case 413: /* type_specifier_nonarray: F16TEXTURE3D  */
#line 2850 "MachineIndependent/glslang.y"
                   {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float texture", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtFloat16, Esd3D);
    }
#line 8926 "MachineIndependent/glslang_tab.cpp"
    break;

  case 414: /* type_specifier_nonarray: F16TEXTURECUBE  */
#line 2856 "MachineIndependent/glslang.y"
                     {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float texture", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtFloat16, EsdCube);
    }
#line 8937 "MachineIndependent/glslang_tab.cpp"
    break;

  case 415: /* type_specifier_nonarray: TEXTURE1DARRAY  */
#line 2862 "MachineIndependent/glslang.y"
                     {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtFloat, Esd1D, true);
    }
#line 8947 "MachineIndependent/glslang_tab.cpp"
    break;

  case 416: /* type_specifier_nonarray: F16TEXTURE1DARRAY  */
#line 2867 "MachineIndependent/glslang.y"
                        {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float texture", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtFloat16, Esd1D, true);
    }
#line 8958 "MachineIndependent/glslang_tab.cpp"
    break;

  case 417: /* type_specifier_nonarray: F16TEXTURE2DARRAY  */
#line 2873 "MachineIndependent/glslang.y"
                        {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float texture", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtFloat16, Esd2D, true);
    }
#line 8969 "MachineIndependent/glslang_tab.cpp"
    break;

  case 418: /* type_specifier_nonarray: F16TEXTURECUBEARRAY  */
#line 2879 "MachineIndependent/glslang.y"
                          {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float texture", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtFloat16, EsdCube, true);
    }
#line 8980 "MachineIndependent/glslang_tab.cpp"
    break;

  case 419: /* type_specifier_nonarray: ITEXTURE1D  */
#line 2885 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtInt, Esd1D);
    }
#line 8990 "MachineIndependent/glslang_tab.cpp"
    break;

  case 420: /* type_specifier_nonarray: ITEXTURE1DARRAY  */
#line 2890 "MachineIndependent/glslang.y"
                      {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtInt, Esd1D, true);
    }
#line 9000 "MachineIndependent/glslang_tab.cpp"
    break;

  case 421: /* type_specifier_nonarray: UTEXTURE1D  */
#line 2895 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtUint, Esd1D);
    }
#line 9010 "MachineIndependent/glslang_tab.cpp"
    break;

  case 422: /* type_specifier_nonarray: UTEXTURE1DARRAY  */
#line 2900 "MachineIndependent/glslang.y"
                      {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtUint, Esd1D, true);
    }
#line 9020 "MachineIndependent/glslang_tab.cpp"
    break;

  case 423: /* type_specifier_nonarray: TEXTURE2DRECT  */
#line 2905 "MachineIndependent/glslang.y"
                    {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtFloat, EsdRect);
    }
#line 9030 "MachineIndependent/glslang_tab.cpp"
    break;

  case 424: /* type_specifier_nonarray: F16TEXTURE2DRECT  */
#line 2910 "MachineIndependent/glslang.y"
                       {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float texture", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtFloat16, EsdRect);
    }
#line 9041 "MachineIndependent/glslang_tab.cpp"
    break;

  case 425: /* type_specifier_nonarray: ITEXTURE2DRECT  */
#line 2916 "MachineIndependent/glslang.y"
                     {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtInt, EsdRect);
    }
#line 9051 "MachineIndependent/glslang_tab.cpp"
    break;

  case 426: /* type_specifier_nonarray: UTEXTURE2DRECT  */
#line 2921 "MachineIndependent/glslang.y"
                     {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtUint, EsdRect);
    }
#line 9061 "MachineIndependent/glslang_tab.cpp"
    break;

  case 427: /* type_specifier_nonarray: TEXTUREBUFFER  */
#line 2926 "MachineIndependent/glslang.y"
                    {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtFloat, EsdBuffer);
    }
#line 9071 "MachineIndependent/glslang_tab.cpp"
    break;

  case 428: /* type_specifier_nonarray: F16TEXTUREBUFFER  */
#line 2931 "MachineIndependent/glslang.y"
                       {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float texture", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtFloat16, EsdBuffer);
    }
#line 9082 "MachineIndependent/glslang_tab.cpp"
    break;

  case 429: /* type_specifier_nonarray: ITEXTUREBUFFER  */
#line 2937 "MachineIndependent/glslang.y"
                     {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtInt, EsdBuffer);
    }
#line 9092 "MachineIndependent/glslang_tab.cpp"
    break;

  case 430: /* type_specifier_nonarray: UTEXTUREBUFFER  */
#line 2942 "MachineIndependent/glslang.y"
                     {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtUint, EsdBuffer);
    }
#line 9102 "MachineIndependent/glslang_tab.cpp"
    break;

  case 431: /* type_specifier_nonarray: TEXTURE2DMS  */
#line 2947 "MachineIndependent/glslang.y"
                  {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtFloat, Esd2D, false, false, true);
    }
#line 9112 "MachineIndependent/glslang_tab.cpp"
    break;

  case 432: /* type_specifier_nonarray: F16TEXTURE2DMS  */
#line 2952 "MachineIndependent/glslang.y"
                     {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float texture", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtFloat16, Esd2D, false, false, true);
    }
#line 9123 "MachineIndependent/glslang_tab.cpp"
    break;

  case 433: /* type_specifier_nonarray: ITEXTURE2DMS  */
#line 2958 "MachineIndependent/glslang.y"
                   {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtInt, Esd2D, false, false, true);
    }
#line 9133 "MachineIndependent/glslang_tab.cpp"
    break;

  case 434: /* type_specifier_nonarray: UTEXTURE2DMS  */
#line 2963 "MachineIndependent/glslang.y"
                   {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtUint, Esd2D, false, false, true);
    }
#line 9143 "MachineIndependent/glslang_tab.cpp"
    break;

  case 435: /* type_specifier_nonarray: TEXTURE2DMSARRAY  */
#line 2968 "MachineIndependent/glslang.y"
                       {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtFloat, Esd2D, true, false, true);
    }
#line 9153 "MachineIndependent/glslang_tab.cpp"
    break;

  case 436: /* type_specifier_nonarray: F16TEXTURE2DMSARRAY  */
#line 2973 "MachineIndependent/glslang.y"
                          {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float texture", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtFloat16, Esd2D, true, false, true);
    }
#line 9164 "MachineIndependent/glslang_tab.cpp"
    break;

  case 437: /* type_specifier_nonarray: ITEXTURE2DMSARRAY  */
#line 2979 "MachineIndependent/glslang.y"
                        {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtInt, Esd2D, true, false, true);
    }
#line 9174 "MachineIndependent/glslang_tab.cpp"
    break;

  case 438: /* type_specifier_nonarray: UTEXTURE2DMSARRAY  */
#line 2984 "MachineIndependent/glslang.y"
                        {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setTexture(EbtUint, Esd2D, true, false, true);
    }
#line 9184 "MachineIndependent/glslang_tab.cpp"
    break;

  case 439: /* type_specifier_nonarray: IMAGE1D  */
#line 2989 "MachineIndependent/glslang.y"
              {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtFloat, Esd1D);
    }
#line 9194 "MachineIndependent/glslang_tab.cpp"
    break;

  case 440: /* type_specifier_nonarray: F16IMAGE1D  */
#line 2994 "MachineIndependent/glslang.y"
                 {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float image", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtFloat16, Esd1D);
    }
#line 9205 "MachineIndependent/glslang_tab.cpp"
    break;

  case 441: /* type_specifier_nonarray: IIMAGE1D  */
#line 3000 "MachineIndependent/glslang.y"
               {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtInt, Esd1D);
    }
#line 9215 "MachineIndependent/glslang_tab.cpp"
    break;

  case 442: /* type_specifier_nonarray: UIMAGE1D  */
#line 3005 "MachineIndependent/glslang.y"
               {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtUint, Esd1D);
    }
#line 9225 "MachineIndependent/glslang_tab.cpp"
    break;

  case 443: /* type_specifier_nonarray: IMAGE2D  */
#line 3010 "MachineIndependent/glslang.y"
              {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtFloat, Esd2D);
    }
#line 9235 "MachineIndependent/glslang_tab.cpp"
    break;

  case 444: /* type_specifier_nonarray: F16IMAGE2D  */
#line 3015 "MachineIndependent/glslang.y"
                 {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float image", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtFloat16, Esd2D);
    }
#line 9246 "MachineIndependent/glslang_tab.cpp"
    break;

  case 445: /* type_specifier_nonarray: IIMAGE2D  */
#line 3021 "MachineIndependent/glslang.y"
               {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtInt, Esd2D);
    }
#line 9256 "MachineIndependent/glslang_tab.cpp"
    break;

  case 446: /* type_specifier_nonarray: UIMAGE2D  */
#line 3026 "MachineIndependent/glslang.y"
               {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtUint, Esd2D);
    }
#line 9266 "MachineIndependent/glslang_tab.cpp"
    break;

  case 447: /* type_specifier_nonarray: IMAGE3D  */
#line 3031 "MachineIndependent/glslang.y"
              {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtFloat, Esd3D);
    }
#line 9276 "MachineIndependent/glslang_tab.cpp"
    break;

  case 448: /* type_specifier_nonarray: F16IMAGE3D  */
#line 3036 "MachineIndependent/glslang.y"
                 {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float image", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtFloat16, Esd3D);
    }
#line 9287 "MachineIndependent/glslang_tab.cpp"
    break;

  case 449: /* type_specifier_nonarray: IIMAGE3D  */
#line 3042 "MachineIndependent/glslang.y"
               {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtInt, Esd3D);
    }
#line 9297 "MachineIndependent/glslang_tab.cpp"
    break;

  case 450: /* type_specifier_nonarray: UIMAGE3D  */
#line 3047 "MachineIndependent/glslang.y"
               {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtUint, Esd3D);
    }
#line 9307 "MachineIndependent/glslang_tab.cpp"
    break;

  case 451: /* type_specifier_nonarray: IMAGE2DRECT  */
#line 3052 "MachineIndependent/glslang.y"
                  {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtFloat, EsdRect);
    }
#line 9317 "MachineIndependent/glslang_tab.cpp"
    break;

  case 452: /* type_specifier_nonarray: F16IMAGE2DRECT  */
#line 3057 "MachineIndependent/glslang.y"
                     {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float image", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtFloat16, EsdRect);
    }
#line 9328 "MachineIndependent/glslang_tab.cpp"
    break;

  case 453: /* type_specifier_nonarray: IIMAGE2DRECT  */
#line 3063 "MachineIndependent/glslang.y"
                   {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtInt, EsdRect);
    }
#line 9338 "MachineIndependent/glslang_tab.cpp"
    break;

  case 454: /* type_specifier_nonarray: UIMAGE2DRECT  */
#line 3068 "MachineIndependent/glslang.y"
                   {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtUint, EsdRect);
    }
#line 9348 "MachineIndependent/glslang_tab.cpp"
    break;

  case 455: /* type_specifier_nonarray: IMAGECUBE  */
#line 3073 "MachineIndependent/glslang.y"
                {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtFloat, EsdCube);
    }
#line 9358 "MachineIndependent/glslang_tab.cpp"
    break;

  case 456: /* type_specifier_nonarray: F16IMAGECUBE  */
#line 3078 "MachineIndependent/glslang.y"
                   {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float image", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtFloat16, EsdCube);
    }
#line 9369 "MachineIndependent/glslang_tab.cpp"
    break;

  case 457: /* type_specifier_nonarray: IIMAGECUBE  */
#line 3084 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtInt, EsdCube);
    }
#line 9379 "MachineIndependent/glslang_tab.cpp"
    break;

  case 458: /* type_specifier_nonarray: UIMAGECUBE  */
#line 3089 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtUint, EsdCube);
    }
#line 9389 "MachineIndependent/glslang_tab.cpp"
    break;

  case 459: /* type_specifier_nonarray: IMAGEBUFFER  */
#line 3094 "MachineIndependent/glslang.y"
                  {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtFloat, EsdBuffer);
    }
#line 9399 "MachineIndependent/glslang_tab.cpp"
    break;

  case 460: /* type_specifier_nonarray: F16IMAGEBUFFER  */
#line 3099 "MachineIndependent/glslang.y"
                     {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float image", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtFloat16, EsdBuffer);
    }
#line 9410 "MachineIndependent/glslang_tab.cpp"
    break;

  case 461: /* type_specifier_nonarray: IIMAGEBUFFER  */
#line 3105 "MachineIndependent/glslang.y"
                   {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtInt, EsdBuffer);
    }
#line 9420 "MachineIndependent/glslang_tab.cpp"
    break;

  case 462: /* type_specifier_nonarray: UIMAGEBUFFER  */
#line 3110 "MachineIndependent/glslang.y"
                   {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtUint, EsdBuffer);
    }
#line 9430 "MachineIndependent/glslang_tab.cpp"
    break;

  case 463: /* type_specifier_nonarray: IMAGE1DARRAY  */
#line 3115 "MachineIndependent/glslang.y"
                   {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtFloat, Esd1D, true);
    }
#line 9440 "MachineIndependent/glslang_tab.cpp"
    break;

  case 464: /* type_specifier_nonarray: F16IMAGE1DARRAY  */
#line 3120 "MachineIndependent/glslang.y"
                      {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float image", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtFloat16, Esd1D, true);
    }
#line 9451 "MachineIndependent/glslang_tab.cpp"
    break;

  case 465: /* type_specifier_nonarray: IIMAGE1DARRAY  */
#line 3126 "MachineIndependent/glslang.y"
                    {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtInt, Esd1D, true);
    }
#line 9461 "MachineIndependent/glslang_tab.cpp"
    break;

  case 466: /* type_specifier_nonarray: UIMAGE1DARRAY  */
#line 3131 "MachineIndependent/glslang.y"
                    {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtUint, Esd1D, true);
    }
#line 9471 "MachineIndependent/glslang_tab.cpp"
    break;

  case 467: /* type_specifier_nonarray: IMAGE2DARRAY  */
#line 3136 "MachineIndependent/glslang.y"
                   {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtFloat, Esd2D, true);
    }
#line 9481 "MachineIndependent/glslang_tab.cpp"
    break;

  case 468: /* type_specifier_nonarray: F16IMAGE2DARRAY  */
#line 3141 "MachineIndependent/glslang.y"
                      {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float image", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtFloat16, Esd2D, true);
    }
#line 9492 "MachineIndependent/glslang_tab.cpp"
    break;

  case 469: /* type_specifier_nonarray: IIMAGE2DARRAY  */
#line 3147 "MachineIndependent/glslang.y"
                    {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtInt, Esd2D, true);
    }
#line 9502 "MachineIndependent/glslang_tab.cpp"
    break;

  case 470: /* type_specifier_nonarray: UIMAGE2DARRAY  */
#line 3152 "MachineIndependent/glslang.y"
                    {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtUint, Esd2D, true);
    }
#line 9512 "MachineIndependent/glslang_tab.cpp"
    break;

  case 471: /* type_specifier_nonarray: IMAGECUBEARRAY  */
#line 3157 "MachineIndependent/glslang.y"
                     {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtFloat, EsdCube, true);
    }
#line 9522 "MachineIndependent/glslang_tab.cpp"
    break;

  case 472: /* type_specifier_nonarray: F16IMAGECUBEARRAY  */
#line 3162 "MachineIndependent/glslang.y"
                        {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float image", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtFloat16, EsdCube, true);
    }
#line 9533 "MachineIndependent/glslang_tab.cpp"
    break;

  case 473: /* type_specifier_nonarray: IIMAGECUBEARRAY  */
#line 3168 "MachineIndependent/glslang.y"
                      {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtInt, EsdCube, true);
    }
#line 9543 "MachineIndependent/glslang_tab.cpp"
    break;

  case 474: /* type_specifier_nonarray: UIMAGECUBEARRAY  */
#line 3173 "MachineIndependent/glslang.y"
                      {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtUint, EsdCube, true);
    }
#line 9553 "MachineIndependent/glslang_tab.cpp"
    break;

  case 475: /* type_specifier_nonarray: IMAGE2DMS  */
#line 3178 "MachineIndependent/glslang.y"
                {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtFloat, Esd2D, false, false, true);
    }
#line 9563 "MachineIndependent/glslang_tab.cpp"
    break;

  case 476: /* type_specifier_nonarray: F16IMAGE2DMS  */
#line 3183 "MachineIndependent/glslang.y"
                   {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float image", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtFloat16, Esd2D, false, false, true);
    }
#line 9574 "MachineIndependent/glslang_tab.cpp"
    break;

  case 477: /* type_specifier_nonarray: IIMAGE2DMS  */
#line 3189 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtInt, Esd2D, false, false, true);
    }
#line 9584 "MachineIndependent/glslang_tab.cpp"
    break;

  case 478: /* type_specifier_nonarray: UIMAGE2DMS  */
#line 3194 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtUint, Esd2D, false, false, true);
    }
#line 9594 "MachineIndependent/glslang_tab.cpp"
    break;

  case 479: /* type_specifier_nonarray: IMAGE2DMSARRAY  */
#line 3199 "MachineIndependent/glslang.y"
                     {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtFloat, Esd2D, true, false, true);
    }
#line 9604 "MachineIndependent/glslang_tab.cpp"
    break;

  case 480: /* type_specifier_nonarray: F16IMAGE2DMSARRAY  */
#line 3204 "MachineIndependent/glslang.y"
                        {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float image", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtFloat16, Esd2D, true, false, true);
    }
#line 9615 "MachineIndependent/glslang_tab.cpp"
    break;

  case 481: /* type_specifier_nonarray: IIMAGE2DMSARRAY  */
#line 3210 "MachineIndependent/glslang.y"
                      {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtInt, Esd2D, true, false, true);
    }
#line 9625 "MachineIndependent/glslang_tab.cpp"
    break;

  case 482: /* type_specifier_nonarray: UIMAGE2DMSARRAY  */
#line 3215 "MachineIndependent/glslang.y"
                      {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtUint, Esd2D, true, false, true);
    }
#line 9635 "MachineIndependent/glslang_tab.cpp"
    break;

  case 483: /* type_specifier_nonarray: I64IMAGE1D  */
#line 3220 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtInt64, Esd1D);
    }
#line 9645 "MachineIndependent/glslang_tab.cpp"
    break;

  case 484: /* type_specifier_nonarray: U64IMAGE1D  */
#line 3225 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtUint64, Esd1D);
    }
#line 9655 "MachineIndependent/glslang_tab.cpp"
    break;

  case 485: /* type_specifier_nonarray: I64IMAGE2D  */
#line 3230 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtInt64, Esd2D);
    }
#line 9665 "MachineIndependent/glslang_tab.cpp"
    break;

  case 486: /* type_specifier_nonarray: U64IMAGE2D  */
#line 3235 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtUint64, Esd2D);
    }
#line 9675 "MachineIndependent/glslang_tab.cpp"
    break;

  case 487: /* type_specifier_nonarray: I64IMAGE3D  */
#line 3240 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtInt64, Esd3D);
    }
#line 9685 "MachineIndependent/glslang_tab.cpp"
    break;

  case 488: /* type_specifier_nonarray: U64IMAGE3D  */
#line 3245 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtUint64, Esd3D);
    }
#line 9695 "MachineIndependent/glslang_tab.cpp"
    break;

  case 489: /* type_specifier_nonarray: I64IMAGE2DRECT  */
#line 3250 "MachineIndependent/glslang.y"
                     {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtInt64, EsdRect);
    }
#line 9705 "MachineIndependent/glslang_tab.cpp"
    break;

  case 490: /* type_specifier_nonarray: U64IMAGE2DRECT  */
#line 3255 "MachineIndependent/glslang.y"
                     {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtUint64, EsdRect);
    }
#line 9715 "MachineIndependent/glslang_tab.cpp"
    break;

  case 491: /* type_specifier_nonarray: I64IMAGECUBE  */
#line 3260 "MachineIndependent/glslang.y"
                   {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtInt64, EsdCube);
    }
#line 9725 "MachineIndependent/glslang_tab.cpp"
    break;

  case 492: /* type_specifier_nonarray: U64IMAGECUBE  */
#line 3265 "MachineIndependent/glslang.y"
                   {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtUint64, EsdCube);
    }
#line 9735 "MachineIndependent/glslang_tab.cpp"
    break;

  case 493: /* type_specifier_nonarray: I64IMAGEBUFFER  */
#line 3270 "MachineIndependent/glslang.y"
                     {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtInt64, EsdBuffer);
    }
#line 9745 "MachineIndependent/glslang_tab.cpp"
    break;

  case 494: /* type_specifier_nonarray: U64IMAGEBUFFER  */
#line 3275 "MachineIndependent/glslang.y"
                     {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtUint64, EsdBuffer);
    }
#line 9755 "MachineIndependent/glslang_tab.cpp"
    break;

  case 495: /* type_specifier_nonarray: I64IMAGE1DARRAY  */
#line 3280 "MachineIndependent/glslang.y"
                      {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtInt64, Esd1D, true);
    }
#line 9765 "MachineIndependent/glslang_tab.cpp"
    break;

  case 496: /* type_specifier_nonarray: U64IMAGE1DARRAY  */
#line 3285 "MachineIndependent/glslang.y"
                      {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtUint64, Esd1D, true);
    }
#line 9775 "MachineIndependent/glslang_tab.cpp"
    break;

  case 497: /* type_specifier_nonarray: I64IMAGE2DARRAY  */
#line 3290 "MachineIndependent/glslang.y"
                      {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtInt64, Esd2D, true);
    }
#line 9785 "MachineIndependent/glslang_tab.cpp"
    break;

  case 498: /* type_specifier_nonarray: U64IMAGE2DARRAY  */
#line 3295 "MachineIndependent/glslang.y"
                      {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtUint64, Esd2D, true);
    }
#line 9795 "MachineIndependent/glslang_tab.cpp"
    break;

  case 499: /* type_specifier_nonarray: I64IMAGECUBEARRAY  */
#line 3300 "MachineIndependent/glslang.y"
                        {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtInt64, EsdCube, true);
    }
#line 9805 "MachineIndependent/glslang_tab.cpp"
    break;

  case 500: /* type_specifier_nonarray: U64IMAGECUBEARRAY  */
#line 3305 "MachineIndependent/glslang.y"
                        {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtUint64, EsdCube, true);
    }
#line 9815 "MachineIndependent/glslang_tab.cpp"
    break;

  case 501: /* type_specifier_nonarray: I64IMAGE2DMS  */
#line 3310 "MachineIndependent/glslang.y"
                   {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtInt64, Esd2D, false, false, true);
    }
#line 9825 "MachineIndependent/glslang_tab.cpp"
    break;

  case 502: /* type_specifier_nonarray: U64IMAGE2DMS  */
#line 3315 "MachineIndependent/glslang.y"
                   {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtUint64, Esd2D, false, false, true);
    }
#line 9835 "MachineIndependent/glslang_tab.cpp"
    break;

  case 503: /* type_specifier_nonarray: I64IMAGE2DMSARRAY  */
#line 3320 "MachineIndependent/glslang.y"
                        {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtInt64, Esd2D, true, false, true);
    }
#line 9845 "MachineIndependent/glslang_tab.cpp"
    break;

  case 504: /* type_specifier_nonarray: U64IMAGE2DMSARRAY  */
#line 3325 "MachineIndependent/glslang.y"
                        {
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setImage(EbtUint64, Esd2D, true, false, true);
    }
#line 9855 "MachineIndependent/glslang_tab.cpp"
    break;

  case 505: /* type_specifier_nonarray: SAMPLEREXTERNALOES  */
#line 3330 "MachineIndependent/glslang.y"
                         {  // GL_OES_EGL_image_external
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat, Esd2D);
        (yyval.interm.type).sampler.external = true;
    }
#line 9866 "MachineIndependent/glslang_tab.cpp"
    break;

  case 506: /* type_specifier_nonarray: SAMPLEREXTERNAL2DY2YEXT  */
#line 3336 "MachineIndependent/glslang.y"
                              { // GL_EXT_YUV_target
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.set(EbtFloat, Esd2D);
        (yyval.interm.type).sampler.yuv = true;
    }
#line 9877 "MachineIndependent/glslang_tab.cpp"
    break;

  case 507: /* type_specifier_nonarray: SUBPASSINPUT  */
#line 3342 "MachineIndependent/glslang.y"
                   {
        parseContext.requireStage((yyvsp[0].lex).loc, EShLangFragment, "subpass input");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setSubpass(EbtFloat);
    }
#line 9888 "MachineIndependent/glslang_tab.cpp"
    break;

  case 508: /* type_specifier_nonarray: SUBPASSINPUTMS  */
#line 3348 "MachineIndependent/glslang.y"
                     {
        parseContext.requireStage((yyvsp[0].lex).loc, EShLangFragment, "subpass input");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setSubpass(EbtFloat, true);
    }
#line 9899 "MachineIndependent/glslang_tab.cpp"
    break;

  case 509: /* type_specifier_nonarray: F16SUBPASSINPUT  */
#line 3354 "MachineIndependent/glslang.y"
                      {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float subpass input", parseContext.symbolTable.atBuiltInLevel());
        parseContext.requireStage((yyvsp[0].lex).loc, EShLangFragment, "subpass input");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setSubpass(EbtFloat16);
    }
#line 9911 "MachineIndependent/glslang_tab.cpp"
    break;

  case 510: /* type_specifier_nonarray: F16SUBPASSINPUTMS  */
#line 3361 "MachineIndependent/glslang.y"
                        {
        parseContext.float16OpaqueCheck((yyvsp[0].lex).loc, "half float subpass input", parseContext.symbolTable.atBuiltInLevel());
        parseContext.requireStage((yyvsp[0].lex).loc, EShLangFragment, "subpass input");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setSubpass(EbtFloat16, true);
    }
#line 9923 "MachineIndependent/glslang_tab.cpp"
    break;

  case 511: /* type_specifier_nonarray: ISUBPASSINPUT  */
#line 3368 "MachineIndependent/glslang.y"
                    {
        parseContext.requireStage((yyvsp[0].lex).loc, EShLangFragment, "subpass input");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setSubpass(EbtInt);
    }
#line 9934 "MachineIndependent/glslang_tab.cpp"
    break;

  case 512: /* type_specifier_nonarray: ISUBPASSINPUTMS  */
#line 3374 "MachineIndependent/glslang.y"
                      {
        parseContext.requireStage((yyvsp[0].lex).loc, EShLangFragment, "subpass input");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setSubpass(EbtInt, true);
    }
#line 9945 "MachineIndependent/glslang_tab.cpp"
    break;

  case 513: /* type_specifier_nonarray: USUBPASSINPUT  */
#line 3380 "MachineIndependent/glslang.y"
                    {
        parseContext.requireStage((yyvsp[0].lex).loc, EShLangFragment, "subpass input");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setSubpass(EbtUint);
    }
#line 9956 "MachineIndependent/glslang_tab.cpp"
    break;

  case 514: /* type_specifier_nonarray: USUBPASSINPUTMS  */
#line 3386 "MachineIndependent/glslang.y"
                      {
        parseContext.requireStage((yyvsp[0].lex).loc, EShLangFragment, "subpass input");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtSampler;
        (yyval.interm.type).sampler.setSubpass(EbtUint, true);
    }
#line 9967 "MachineIndependent/glslang_tab.cpp"
    break;

  case 515: /* type_specifier_nonarray: FCOOPMATNV  */
#line 3392 "MachineIndependent/glslang.y"
                 {
        parseContext.fcoopmatCheck((yyvsp[0].lex).loc, "fcoopmatNV", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtFloat;
        (yyval.interm.type).coopmat = true;
    }
#line 9978 "MachineIndependent/glslang_tab.cpp"
    break;

  case 516: /* type_specifier_nonarray: ICOOPMATNV  */
#line 3398 "MachineIndependent/glslang.y"
                 {
        parseContext.intcoopmatCheck((yyvsp[0].lex).loc, "icoopmatNV", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtInt;
        (yyval.interm.type).coopmat = true;
    }
#line 9989 "MachineIndependent/glslang_tab.cpp"
    break;

  case 517: /* type_specifier_nonarray: UCOOPMATNV  */
#line 3404 "MachineIndependent/glslang.y"
                 {
        parseContext.intcoopmatCheck((yyvsp[0].lex).loc, "ucoopmatNV", parseContext.symbolTable.atBuiltInLevel());
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        (yyval.interm.type).basicType = EbtUint;
        (yyval.interm.type).coopmat = true;
    }
#line 10000 "MachineIndependent/glslang_tab.cpp"
    break;

  case 518: /* type_specifier_nonarray: struct_specifier  */
#line 3411 "MachineIndependent/glslang.y"
                       {
        (yyval.interm.type) = (yyvsp[0].interm.type);
        (yyval.interm.type).qualifier.storage = parseContext.symbolTable.atGlobalLevel() ? EvqGlobal : EvqTemporary;
        parseContext.structTypeCheck((yyval.interm.type).loc, (yyval.interm.type));
    }
#line 10010 "MachineIndependent/glslang_tab.cpp"
    break;

  case 519: /* type_specifier_nonarray: TYPE_NAME  */
#line 3416 "MachineIndependent/glslang.y"
                {
        //
        // This is for user defined type names.  The lexical phase looked up the
        // type.
        //
        if (const TVariable* variable = ((yyvsp[0].lex).symbol)->getAsVariable()) {
            const TType& structure = variable->getType();
            (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
            (yyval.interm.type).basicType = EbtStruct;
            (yyval.interm.type).userDef = &structure;
        } else
            parseContext.error((yyvsp[0].lex).loc, "expected type name", (yyvsp[0].lex).string->c_str(), "");
    }
#line 10028 "MachineIndependent/glslang_tab.cpp"
    break;

  case 520: /* precision_qualifier: HIGH_PRECISION  */
#line 3432 "MachineIndependent/glslang.y"
                     {
        parseContext.profileRequires((yyvsp[0].lex).loc, ENoProfile, 130, 0, "highp precision qualifier");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        parseContext.handlePrecisionQualifier((yyvsp[0].lex).loc, (yyval.interm.type).qualifier, EpqHigh);
    }
#line 10038 "MachineIndependent/glslang_tab.cpp"
    break;

  case 521: /* precision_qualifier: MEDIUM_PRECISION  */
#line 3437 "MachineIndependent/glslang.y"
                       {
        parseContext.profileRequires((yyvsp[0].lex).loc, ENoProfile, 130, 0, "mediump precision qualifier");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        parseContext.handlePrecisionQualifier((yyvsp[0].lex).loc, (yyval.interm.type).qualifier, EpqMedium);
    }
#line 10048 "MachineIndependent/glslang_tab.cpp"
    break;

  case 522: /* precision_qualifier: LOW_PRECISION  */
#line 3442 "MachineIndependent/glslang.y"
                    {
        parseContext.profileRequires((yyvsp[0].lex).loc, ENoProfile, 130, 0, "lowp precision qualifier");
        (yyval.interm.type).init((yyvsp[0].lex).loc, parseContext.symbolTable.atGlobalLevel());
        parseContext.handlePrecisionQualifier((yyvsp[0].lex).loc, (yyval.interm.type).qualifier, EpqLow);
    }
#line 10058 "MachineIndependent/glslang_tab.cpp"
    break;

  case 523: /* $@3: %empty  */
#line 3450 "MachineIndependent/glslang.y"
                                   { parseContext.nestedStructCheck((yyvsp[-2].lex).loc); }
#line 10064 "MachineIndependent/glslang_tab.cpp"
    break;

  case 524: /* struct_specifier: STRUCT IDENTIFIER LEFT_BRACE $@3 struct_declaration_list RIGHT_BRACE  */
#line 3450 "MachineIndependent/glslang.y"
                                                                                                                   {
        TType* structure = new TType((yyvsp[-1].interm.typeList), *(yyvsp[-4].lex).string);
        parseContext.structArrayCheck((yyvsp[-4].lex).loc, *structure);
        TVariable* userTypeDef = new TVariable((yyvsp[-4].lex).string, *structure, true);
        if (! parseContext.symbolTable.insert(*userTypeDef))
            parseContext.error((yyvsp[-4].lex).loc, "redefinition", (yyvsp[-4].lex).string->c_str(), "struct");
        (yyval.interm.type).init((yyvsp[-5].lex).loc);
        (yyval.interm.type).basicType = EbtStruct;
        (yyval.interm.type).userDef = structure;
        --parseContext.structNestingLevel;
    }
#line 10080 "MachineIndependent/glslang_tab.cpp"
    break;

  case 525: /* $@4: %empty  */
#line 3461 "MachineIndependent/glslang.y"
                        { parseContext.nestedStructCheck((yyvsp[-1].lex).loc); }
#line 10086 "MachineIndependent/glslang_tab.cpp"
    break;

  case 526: /* struct_specifier: STRUCT LEFT_BRACE $@4 struct_declaration_list RIGHT_BRACE  */
#line 3461 "MachineIndependent/glslang.y"
                                                                                                        {
        TType* structure = new TType((yyvsp[-1].interm.typeList), TString(""));
        (yyval.interm.type).init((yyvsp[-4].lex).loc);
        (yyval.interm.type).basicType = EbtStruct;
        (yyval.interm.type).userDef = structure;
        --parseContext.structNestingLevel;
    }
#line 10098 "MachineIndependent/glslang_tab.cpp"
    break;

  case 527: /* struct_declaration_list: struct_declaration  */
#line 3471 "MachineIndependent/glslang.y"
                         {
        (yyval.interm.typeList) = (yyvsp[0].interm.typeList);
    }
#line 10106 "MachineIndependent/glslang_tab.cpp"
    break;

  case 528: /* struct_declaration_list: struct_declaration_list struct_declaration  */
#line 3474 "MachineIndependent/glslang.y"
                                                 {
        (yyval.interm.typeList) = (yyvsp[-1].interm.typeList);
        for (unsigned int i = 0; i < (yyvsp[0].interm.typeList)->size(); ++i) {
            for (unsigned int j = 0; j < (yyval.interm.typeList)->size(); ++j) {
                if ((*(yyval.interm.typeList))[j].type->getFieldName() == (*(yyvsp[0].interm.typeList))[i].type->getFieldName())
                    parseContext.error((*(yyvsp[0].interm.typeList))[i].loc, "duplicate member name:", "", (*(yyvsp[0].interm.typeList))[i].type->getFieldName().c_str());
            }
            (yyval.interm.typeList)->push_back((*(yyvsp[0].interm.typeList))[i]);
        }
    }
#line 10121 "MachineIndependent/glslang_tab.cpp"
    break;

  case 529: /* struct_declaration: type_specifier struct_declarator_list SEMICOLON  */
#line 3487 "MachineIndependent/glslang.y"
                                                      {
        if ((yyvsp[-2].interm.type).arraySizes) {
            parseContext.profileRequires((yyvsp[-2].interm.type).loc, ENoProfile, 120, E_GL_3DL_array_objects, "arrayed type");
            parseContext.profileRequires((yyvsp[-2].interm.type).loc, EEsProfile, 300, 0, "arrayed type");
            if (parseContext.isEsProfile())
                parseContext.arraySizeRequiredCheck((yyvsp[-2].interm.type).loc, *(yyvsp[-2].interm.type).arraySizes);
        }

        (yyval.interm.typeList) = (yyvsp[-1].interm.typeList);

        parseContext.voidErrorCheck((yyvsp[-2].interm.type).loc, (*(yyvsp[-1].interm.typeList))[0].type->getFieldName(), (yyvsp[-2].interm.type).basicType);
        parseContext.precisionQualifierCheck((yyvsp[-2].interm.type).loc, (yyvsp[-2].interm.type).basicType, (yyvsp[-2].interm.type).qualifier);

        for (unsigned int i = 0; i < (yyval.interm.typeList)->size(); ++i) {
            TType type((yyvsp[-2].interm.type));
            type.setFieldName((*(yyval.interm.typeList))[i].type->getFieldName());
            type.transferArraySizes((*(yyval.interm.typeList))[i].type->getArraySizes());
            type.copyArrayInnerSizes((yyvsp[-2].interm.type).arraySizes);
            parseContext.arrayOfArrayVersionCheck((*(yyval.interm.typeList))[i].loc, type.getArraySizes());
            (*(yyval.interm.typeList))[i].type->shallowCopy(type);
        }
    }
#line 10148 "MachineIndependent/glslang_tab.cpp"
    break;

  case 530: /* struct_declaration: type_qualifier type_specifier struct_declarator_list SEMICOLON  */
#line 3509 "MachineIndependent/glslang.y"
                                                                     {
        if ((yyvsp[-2].interm.type).arraySizes) {
            parseContext.profileRequires((yyvsp[-2].interm.type).loc, ENoProfile, 120, E_GL_3DL_array_objects, "arrayed type");
            parseContext.profileRequires((yyvsp[-2].interm.type).loc, EEsProfile, 300, 0, "arrayed type");
            if (parseContext.isEsProfile())
                parseContext.arraySizeRequiredCheck((yyvsp[-2].interm.type).loc, *(yyvsp[-2].interm.type).arraySizes);
        }

        (yyval.interm.typeList) = (yyvsp[-1].interm.typeList);

        parseContext.memberQualifierCheck((yyvsp[-3].interm.type));
        parseContext.voidErrorCheck((yyvsp[-2].interm.type).loc, (*(yyvsp[-1].interm.typeList))[0].type->getFieldName(), (yyvsp[-2].interm.type).basicType);
        parseContext.mergeQualifiers((yyvsp[-2].interm.type).loc, (yyvsp[-2].interm.type).qualifier, (yyvsp[-3].interm.type).qualifier, true);
        parseContext.precisionQualifierCheck((yyvsp[-2].interm.type).loc, (yyvsp[-2].interm.type).basicType, (yyvsp[-2].interm.type).qualifier);

        for (unsigned int i = 0; i < (yyval.interm.typeList)->size(); ++i) {
            TType type((yyvsp[-2].interm.type));
            type.setFieldName((*(yyval.interm.typeList))[i].type->getFieldName());
            type.transferArraySizes((*(yyval.interm.typeList))[i].type->getArraySizes());
            type.copyArrayInnerSizes((yyvsp[-2].interm.type).arraySizes);
            parseContext.arrayOfArrayVersionCheck((*(yyval.interm.typeList))[i].loc, type.getArraySizes());
            (*(yyval.interm.typeList))[i].type->shallowCopy(type);
        }
    }
#line 10177 "MachineIndependent/glslang_tab.cpp"
    break;

  case 531: /* struct_declarator_list: struct_declarator  */
#line 3536 "MachineIndependent/glslang.y"
                        {
        (yyval.interm.typeList) = new TTypeList;
        (yyval.interm.typeList)->push_back((yyvsp[0].interm.typeLine));
    }
#line 10186 "MachineIndependent/glslang_tab.cpp"
    break;

  case 532: /* struct_declarator_list: struct_declarator_list COMMA struct_declarator  */
#line 3540 "MachineIndependent/glslang.y"
                                                     {
        (yyval.interm.typeList)->push_back((yyvsp[0].interm.typeLine));
    }
#line 10194 "MachineIndependent/glslang_tab.cpp"
    break;

  case 533: /* struct_declarator: IDENTIFIER  */
#line 3546 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.typeLine).type = new TType(EbtVoid);
        (yyval.interm.typeLine).loc = (yyvsp[0].lex).loc;
        (yyval.interm.typeLine).type->setFieldName(*(yyvsp[0].lex).string);
    }
#line 10204 "MachineIndependent/glslang_tab.cpp"
    break;

  case 534: /* struct_declarator: IDENTIFIER array_specifier  */
#line 3551 "MachineIndependent/glslang.y"
                                 {
        parseContext.arrayOfArrayVersionCheck((yyvsp[-1].lex).loc, (yyvsp[0].interm).arraySizes);

        (yyval.interm.typeLine).type = new TType(EbtVoid);
        (yyval.interm.typeLine).loc = (yyvsp[-1].lex).loc;
        (yyval.interm.typeLine).type->setFieldName(*(yyvsp[-1].lex).string);
        (yyval.interm.typeLine).type->transferArraySizes((yyvsp[0].interm).arraySizes);
    }
#line 10217 "MachineIndependent/glslang_tab.cpp"
    break;

  case 535: /* initializer: assignment_expression  */
#line 3562 "MachineIndependent/glslang.y"
                            {
        (yyval.interm.intermTypedNode) = (yyvsp[0].interm.intermTypedNode);
    }
#line 10225 "MachineIndependent/glslang_tab.cpp"
    break;

  case 536: /* initializer: LEFT_BRACE initializer_list RIGHT_BRACE  */
#line 3566 "MachineIndependent/glslang.y"
                                              {
        const char* initFeature = "{ } style initializers";
        parseContext.requireProfile((yyvsp[-2].lex).loc, ~EEsProfile, initFeature);
        parseContext.profileRequires((yyvsp[-2].lex).loc, ~EEsProfile, 420, E_GL_ARB_shading_language_420pack, initFeature);
        (yyval.interm.intermTypedNode) = (yyvsp[-1].interm.intermTypedNode);
    }
#line 10236 "MachineIndependent/glslang_tab.cpp"
    break;

  case 537: /* initializer: LEFT_BRACE initializer_list COMMA RIGHT_BRACE  */
#line 3572 "MachineIndependent/glslang.y"
                                                    {
        const char* initFeature = "{ } style initializers";
        parseContext.requireProfile((yyvsp[-3].lex).loc, ~EEsProfile, initFeature);
        parseContext.profileRequires((yyvsp[-3].lex).loc, ~EEsProfile, 420, E_GL_ARB_shading_language_420pack, initFeature);
        (yyval.interm.intermTypedNode) = (yyvsp[-2].interm.intermTypedNode);
    }
#line 10247 "MachineIndependent/glslang_tab.cpp"
    break;

  case 538: /* initializer_list: initializer  */
#line 3583 "MachineIndependent/glslang.y"
                  {
        (yyval.interm.intermTypedNode) = parseContext.intermediate.growAggregate(0, (yyvsp[0].interm.intermTypedNode), (yyvsp[0].interm.intermTypedNode)->getLoc());
    }
#line 10255 "MachineIndependent/glslang_tab.cpp"
    break;

  case 539: /* initializer_list: initializer_list COMMA initializer  */
#line 3586 "MachineIndependent/glslang.y"
                                         {
        (yyval.interm.intermTypedNode) = parseContext.intermediate.growAggregate((yyvsp[-2].interm.intermTypedNode), (yyvsp[0].interm.intermTypedNode));
    }
#line 10263 "MachineIndependent/glslang_tab.cpp"
    break;

  case 540: /* declaration_statement: declaration  */
#line 3593 "MachineIndependent/glslang.y"
                  { (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode); }
#line 10269 "MachineIndependent/glslang_tab.cpp"
    break;

  case 541: /* statement: compound_statement  */
#line 3597 "MachineIndependent/glslang.y"
                          { (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode); }
#line 10275 "MachineIndependent/glslang_tab.cpp"
    break;

  case 542: /* statement: simple_statement  */
#line 3598 "MachineIndependent/glslang.y"
                          { (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode); }
#line 10281 "MachineIndependent/glslang_tab.cpp"
    break;

  case 543: /* simple_statement: declaration_statement  */
#line 3604 "MachineIndependent/glslang.y"
                            { (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode); }
#line 10287 "MachineIndependent/glslang_tab.cpp"
    break;

  case 544: /* simple_statement: expression_statement  */
#line 3605 "MachineIndependent/glslang.y"
                            { (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode); }
#line 10293 "MachineIndependent/glslang_tab.cpp"
    break;

  case 545: /* simple_statement: selection_statement  */
#line 3606 "MachineIndependent/glslang.y"
                            { (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode); }
#line 10299 "MachineIndependent/glslang_tab.cpp"
    break;

  case 546: /* simple_statement: switch_statement  */
#line 3607 "MachineIndependent/glslang.y"
                            { (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode); }
#line 10305 "MachineIndependent/glslang_tab.cpp"
    break;

  case 547: /* simple_statement: case_label  */
#line 3608 "MachineIndependent/glslang.y"
                            { (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode); }
#line 10311 "MachineIndependent/glslang_tab.cpp"
    break;

  case 548: /* simple_statement: iteration_statement  */
#line 3609 "MachineIndependent/glslang.y"
                            { (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode); }
#line 10317 "MachineIndependent/glslang_tab.cpp"
    break;

  case 549: /* simple_statement: jump_statement  */
#line 3610 "MachineIndependent/glslang.y"
                            { (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode); }
#line 10323 "MachineIndependent/glslang_tab.cpp"
    break;

  case 550: /* simple_statement: demote_statement  */
#line 3612 "MachineIndependent/glslang.y"
                            { (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode); }
#line 10329 "MachineIndependent/glslang_tab.cpp"
    break;

  case 551: /* demote_statement: DEMOTE SEMICOLON  */
#line 3618 "MachineIndependent/glslang.y"
                       {
        parseContext.requireStage((yyvsp[-1].lex).loc, EShLangFragment, "demote");
        parseContext.requireExtensions((yyvsp[-1].lex).loc, 1, &E_GL_EXT_demote_to_helper_invocation, "demote");
        (yyval.interm.intermNode) = parseContext.intermediate.addBranch(EOpDemote, (yyvsp[-1].lex).loc);
    }
#line 10339 "MachineIndependent/glslang_tab.cpp"
    break;

  case 552: /* compound_statement: LEFT_BRACE RIGHT_BRACE  */
#line 3627 "MachineIndependent/glslang.y"
                             { (yyval.interm.intermNode) = 0; }
#line 10345 "MachineIndependent/glslang_tab.cpp"
    break;

  case 553: /* $@5: %empty  */
#line 3628 "MachineIndependent/glslang.y"
                 {
        parseContext.symbolTable.push();
        ++parseContext.statementNestingLevel;
    }
#line 10354 "MachineIndependent/glslang_tab.cpp"
    break;

  case 554: /* $@6: %empty  */
#line 3632 "MachineIndependent/glslang.y"
                     {
        parseContext.symbolTable.pop(&parseContext.defaultPrecision[0]);
        --parseContext.statementNestingLevel;
    }
#line 10363 "MachineIndependent/glslang_tab.cpp"
    break;

  case 555: /* compound_statement: LEFT_BRACE $@5 statement_list $@6 RIGHT_BRACE  */
#line 3636 "MachineIndependent/glslang.y"
                  {
        if ((yyvsp[-2].interm.intermNode) && (yyvsp[-2].interm.intermNode)->getAsAggregate())
            (yyvsp[-2].interm.intermNode)->getAsAggregate()->setOperator(EOpSequence);
        (yyval.interm.intermNode) = (yyvsp[-2].interm.intermNode);
    }
#line 10373 "MachineIndependent/glslang_tab.cpp"
    break;

  case 556: /* statement_no_new_scope: compound_statement_no_new_scope  */
#line 3644 "MachineIndependent/glslang.y"
                                      { (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode); }
#line 10379 "MachineIndependent/glslang_tab.cpp"
    break;

  case 557: /* statement_no_new_scope: simple_statement  */
#line 3645 "MachineIndependent/glslang.y"
                                      { (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode); }
#line 10385 "MachineIndependent/glslang_tab.cpp"
    break;

  case 558: /* $@7: %empty  */
#line 3649 "MachineIndependent/glslang.y"
      {
        ++parseContext.controlFlowNestingLevel;
    }
#line 10393 "MachineIndependent/glslang_tab.cpp"
    break;

  case 559: /* statement_scoped: $@7 compound_statement  */
#line 3652 "MachineIndependent/glslang.y"
                          {
        --parseContext.controlFlowNestingLevel;
        (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode);
    }
#line 10402 "MachineIndependent/glslang_tab.cpp"
    break;

  case 560: /* $@8: %empty  */
#line 3656 "MachineIndependent/glslang.y"
      {
        parseContext.symbolTable.push();
        ++parseContext.statementNestingLevel;
        ++parseContext.controlFlowNestingLevel;
    }
#line 10412 "MachineIndependent/glslang_tab.cpp"
    break;

  case 561: /* statement_scoped: $@8 simple_statement  */
#line 3661 "MachineIndependent/glslang.y"
                       {
        parseContext.symbolTable.pop(&parseContext.defaultPrecision[0]);
        --parseContext.statementNestingLevel;
        --parseContext.controlFlowNestingLevel;
        (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode);
    }
#line 10423 "MachineIndependent/glslang_tab.cpp"
    break;

  case 562: /* compound_statement_no_new_scope: LEFT_BRACE RIGHT_BRACE  */
#line 3670 "MachineIndependent/glslang.y"
                             {
        (yyval.interm.intermNode) = 0;
    }
#line 10431 "MachineIndependent/glslang_tab.cpp"
    break;

  case 563: /* compound_statement_no_new_scope: LEFT_BRACE statement_list RIGHT_BRACE  */
#line 3673 "MachineIndependent/glslang.y"
                                            {
        if ((yyvsp[-1].interm.intermNode) && (yyvsp[-1].interm.intermNode)->getAsAggregate())
            (yyvsp[-1].interm.intermNode)->getAsAggregate()->setOperator(EOpSequence);
        (yyval.interm.intermNode) = (yyvsp[-1].interm.intermNode);
    }
#line 10441 "MachineIndependent/glslang_tab.cpp"
    break;

  case 564: /* statement_list: statement  */
#line 3681 "MachineIndependent/glslang.y"
                {
        (yyval.interm.intermNode) = parseContext.intermediate.makeAggregate((yyvsp[0].interm.intermNode));
        if ((yyvsp[0].interm.intermNode) && (yyvsp[0].interm.intermNode)->getAsBranchNode() && ((yyvsp[0].interm.intermNode)->getAsBranchNode()->getFlowOp() == EOpCase ||
                                            (yyvsp[0].interm.intermNode)->getAsBranchNode()->getFlowOp() == EOpDefault)) {
            parseContext.wrapupSwitchSubsequence(0, (yyvsp[0].interm.intermNode));
            (yyval.interm.intermNode) = 0;  // start a fresh subsequence for what's after this case
        }
    }
#line 10454 "MachineIndependent/glslang_tab.cpp"
    break;

  case 565: /* statement_list: statement_list statement  */
#line 3689 "MachineIndependent/glslang.y"
                               {
        if ((yyvsp[0].interm.intermNode) && (yyvsp[0].interm.intermNode)->getAsBranchNode() && ((yyvsp[0].interm.intermNode)->getAsBranchNode()->getFlowOp() == EOpCase ||
                                            (yyvsp[0].interm.intermNode)->getAsBranchNode()->getFlowOp() == EOpDefault)) {
            parseContext.wrapupSwitchSubsequence((yyvsp[-1].interm.intermNode) ? (yyvsp[-1].interm.intermNode)->getAsAggregate() : 0, (yyvsp[0].interm.intermNode));
            (yyval.interm.intermNode) = 0;  // start a fresh subsequence for what's after this case
        } else
            (yyval.interm.intermNode) = parseContext.intermediate.growAggregate((yyvsp[-1].interm.intermNode), (yyvsp[0].interm.intermNode));
    }
#line 10467 "MachineIndependent/glslang_tab.cpp"
    break;

  case 566: /* expression_statement: SEMICOLON  */
#line 3700 "MachineIndependent/glslang.y"
                 { (yyval.interm.intermNode) = 0; }
#line 10473 "MachineIndependent/glslang_tab.cpp"
    break;

  case 567: /* expression_statement: expression SEMICOLON  */
#line 3701 "MachineIndependent/glslang.y"
                            { (yyval.interm.intermNode) = static_cast<TIntermNode*>((yyvsp[-1].interm.intermTypedNode)); }
#line 10479 "MachineIndependent/glslang_tab.cpp"
    break;

  case 568: /* selection_statement: selection_statement_nonattributed  */
#line 3705 "MachineIndependent/glslang.y"
                                        {
        (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode);
    }
#line 10487 "MachineIndependent/glslang_tab.cpp"
    break;

  case 569: /* selection_statement: attribute selection_statement_nonattributed  */
#line 3709 "MachineIndependent/glslang.y"
                                                  {
        parseContext.handleSelectionAttributes(*(yyvsp[-1].interm.attributes), (yyvsp[0].interm.intermNode));
        (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode);
    }
#line 10496 "MachineIndependent/glslang_tab.cpp"
    break;

  case 570: /* selection_statement_nonattributed: IF LEFT_PAREN expression RIGHT_PAREN selection_rest_statement  */
#line 3716 "MachineIndependent/glslang.y"
                                                                    {
        parseContext.boolCheck((yyvsp[-4].lex).loc, (yyvsp[-2].interm.intermTypedNode));
        (yyval.interm.intermNode) = parseContext.intermediate.addSelection((yyvsp[-2].interm.intermTypedNode), (yyvsp[0].interm.nodePair), (yyvsp[-4].lex).loc);
    }
#line 10505 "MachineIndependent/glslang_tab.cpp"
    break;

  case 571: /* selection_rest_statement: statement_scoped ELSE statement_scoped  */
#line 3723 "MachineIndependent/glslang.y"
                                             {
        (yyval.interm.nodePair).node1 = (yyvsp[-2].interm.intermNode);
        (yyval.interm.nodePair).node2 = (yyvsp[0].interm.intermNode);
    }
#line 10514 "MachineIndependent/glslang_tab.cpp"
    break;

  case 572: /* selection_rest_statement: statement_scoped  */
#line 3727 "MachineIndependent/glslang.y"
                       {
        (yyval.interm.nodePair).node1 = (yyvsp[0].interm.intermNode);
        (yyval.interm.nodePair).node2 = 0;
    }
#line 10523 "MachineIndependent/glslang_tab.cpp"
    break;

  case 573: /* condition: expression  */
#line 3735 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.intermTypedNode) = (yyvsp[0].interm.intermTypedNode);
        parseContext.boolCheck((yyvsp[0].interm.intermTypedNode)->getLoc(), (yyvsp[0].interm.intermTypedNode));
    }
#line 10532 "MachineIndependent/glslang_tab.cpp"
    break;

  case 574: /* condition: fully_specified_type IDENTIFIER EQUAL initializer  */
#line 3739 "MachineIndependent/glslang.y"
                                                        {
        parseContext.boolCheck((yyvsp[-2].lex).loc, (yyvsp[-3].interm.type));

        TType type((yyvsp[-3].interm.type));
        TIntermNode* initNode = parseContext.declareVariable((yyvsp[-2].lex).loc, *(yyvsp[-2].lex).string, (yyvsp[-3].interm.type), 0, (yyvsp[0].interm.intermTypedNode));
        if (initNode)
            (yyval.interm.intermTypedNode) = initNode->getAsTyped();
        else
            (yyval.interm.intermTypedNode) = 0;
    }
#line 10547 "MachineIndependent/glslang_tab.cpp"
    break;

  case 575: /* switch_statement: switch_statement_nonattributed  */
#line 3752 "MachineIndependent/glslang.y"
                                     {
        (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode);
    }
#line 10555 "MachineIndependent/glslang_tab.cpp"
    break;

  case 576: /* switch_statement: attribute switch_statement_nonattributed  */
#line 3756 "MachineIndependent/glslang.y"
                                               {
        parseContext.handleSwitchAttributes(*(yyvsp[-1].interm.attributes), (yyvsp[0].interm.intermNode));
        (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode);
    }
#line 10564 "MachineIndependent/glslang_tab.cpp"
    break;

  case 577: /* $@9: %empty  */
#line 3763 "MachineIndependent/glslang.y"
                                               {
        // start new switch sequence on the switch stack
        ++parseContext.controlFlowNestingLevel;
        ++parseContext.statementNestingLevel;
        parseContext.switchSequenceStack.push_back(new TIntermSequence);
        parseContext.switchLevel.push_back(parseContext.statementNestingLevel);
        parseContext.symbolTable.push();
    }
#line 10577 "MachineIndependent/glslang_tab.cpp"
    break;

  case 578: /* switch_statement_nonattributed: SWITCH LEFT_PAREN expression RIGHT_PAREN $@9 LEFT_BRACE switch_statement_list RIGHT_BRACE  */
#line 3771 "MachineIndependent/glslang.y"
                                                 {
        (yyval.interm.intermNode) = parseContext.addSwitch((yyvsp[-7].lex).loc, (yyvsp[-5].interm.intermTypedNode), (yyvsp[-1].interm.intermNode) ? (yyvsp[-1].interm.intermNode)->getAsAggregate() : 0);
        delete parseContext.switchSequenceStack.back();
        parseContext.switchSequenceStack.pop_back();
        parseContext.switchLevel.pop_back();
        parseContext.symbolTable.pop(&parseContext.defaultPrecision[0]);
        --parseContext.statementNestingLevel;
        --parseContext.controlFlowNestingLevel;
    }
#line 10591 "MachineIndependent/glslang_tab.cpp"
    break;

  case 579: /* switch_statement_list: %empty  */
#line 3783 "MachineIndependent/glslang.y"
                    {
        (yyval.interm.intermNode) = 0;
    }
#line 10599 "MachineIndependent/glslang_tab.cpp"
    break;

  case 580: /* switch_statement_list: statement_list  */
#line 3786 "MachineIndependent/glslang.y"
                     {
        (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode);
    }
#line 10607 "MachineIndependent/glslang_tab.cpp"
    break;

  case 581: /* case_label: CASE expression COLON  */
#line 3792 "MachineIndependent/glslang.y"
                            {
        (yyval.interm.intermNode) = 0;
        if (parseContext.switchLevel.size() == 0)
            parseContext.error((yyvsp[-2].lex).loc, "cannot appear outside switch statement", "case", "");
        else if (parseContext.switchLevel.back() != parseContext.statementNestingLevel)
            parseContext.error((yyvsp[-2].lex).loc, "cannot be nested inside control flow", "case", "");
        else {
            parseContext.constantValueCheck((yyvsp[-1].interm.intermTypedNode), "case");
            parseContext.integerCheck((yyvsp[-1].interm.intermTypedNode), "case");
            (yyval.interm.intermNode) = parseContext.intermediate.addBranch(EOpCase, (yyvsp[-1].interm.intermTypedNode), (yyvsp[-2].lex).loc);
        }
    }
#line 10624 "MachineIndependent/glslang_tab.cpp"
    break;

  case 582: /* case_label: DEFAULT COLON  */
#line 3804 "MachineIndependent/glslang.y"
                    {
        (yyval.interm.intermNode) = 0;
        if (parseContext.switchLevel.size() == 0)
            parseContext.error((yyvsp[-1].lex).loc, "cannot appear outside switch statement", "default", "");
        else if (parseContext.switchLevel.back() != parseContext.statementNestingLevel)
            parseContext.error((yyvsp[-1].lex).loc, "cannot be nested inside control flow", "default", "");
        else
            (yyval.interm.intermNode) = parseContext.intermediate.addBranch(EOpDefault, (yyvsp[-1].lex).loc);
    }
#line 10638 "MachineIndependent/glslang_tab.cpp"
    break;

  case 583: /* iteration_statement: iteration_statement_nonattributed  */
#line 3816 "MachineIndependent/glslang.y"
                                        {
        (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode);
    }
#line 10646 "MachineIndependent/glslang_tab.cpp"
    break;

  case 584: /* iteration_statement: attribute iteration_statement_nonattributed  */
#line 3820 "MachineIndependent/glslang.y"
                                                  {
        parseContext.handleLoopAttributes(*(yyvsp[-1].interm.attributes), (yyvsp[0].interm.intermNode));
        (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode);
    }
#line 10655 "MachineIndependent/glslang_tab.cpp"
    break;

  case 585: /* $@10: %empty  */
#line 3827 "MachineIndependent/glslang.y"
                       {
        if (! parseContext.limits.whileLoops)
            parseContext.error((yyvsp[-1].lex).loc, "while loops not available", "limitation", "");
        parseContext.symbolTable.push();
        ++parseContext.loopNestingLevel;
        ++parseContext.statementNestingLevel;
        ++parseContext.controlFlowNestingLevel;
    }
#line 10668 "MachineIndependent/glslang_tab.cpp"
    break;

  case 586: /* iteration_statement_nonattributed: WHILE LEFT_PAREN $@10 condition RIGHT_PAREN statement_no_new_scope  */
#line 3835 "MachineIndependent/glslang.y"
                                                   {
        parseContext.symbolTable.pop(&parseContext.defaultPrecision[0]);
        (yyval.interm.intermNode) = parseContext.intermediate.addLoop((yyvsp[0].interm.intermNode), (yyvsp[-2].interm.intermTypedNode), 0, true, (yyvsp[-5].lex).loc);
        --parseContext.loopNestingLevel;
        --parseContext.statementNestingLevel;
        --parseContext.controlFlowNestingLevel;
    }
#line 10680 "MachineIndependent/glslang_tab.cpp"
    break;

  case 587: /* $@11: %empty  */
#line 3842 "MachineIndependent/glslang.y"
         {
        ++parseContext.loopNestingLevel;
        ++parseContext.statementNestingLevel;
        ++parseContext.controlFlowNestingLevel;
    }
#line 10690 "MachineIndependent/glslang_tab.cpp"
    break;

  case 588: /* iteration_statement_nonattributed: DO $@11 statement WHILE LEFT_PAREN expression RIGHT_PAREN SEMICOLON  */
#line 3847 "MachineIndependent/glslang.y"
                                                                  {
        if (! parseContext.limits.whileLoops)
            parseContext.error((yyvsp[-7].lex).loc, "do-while loops not available", "limitation", "");

        parseContext.boolCheck((yyvsp[0].lex).loc, (yyvsp[-2].interm.intermTypedNode));

        (yyval.interm.intermNode) = parseContext.intermediate.addLoop((yyvsp[-5].interm.intermNode), (yyvsp[-2].interm.intermTypedNode), 0, false, (yyvsp[-4].lex).loc);
        --parseContext.loopNestingLevel;
        --parseContext.statementNestingLevel;
        --parseContext.controlFlowNestingLevel;
    }
#line 10706 "MachineIndependent/glslang_tab.cpp"
    break;

  case 589: /* $@12: %empty  */
#line 3858 "MachineIndependent/glslang.y"
                     {
        parseContext.symbolTable.push();
        ++parseContext.loopNestingLevel;
        ++parseContext.statementNestingLevel;
        ++parseContext.controlFlowNestingLevel;
    }
#line 10717 "MachineIndependent/glslang_tab.cpp"
    break;

  case 590: /* iteration_statement_nonattributed: FOR LEFT_PAREN $@12 for_init_statement for_rest_statement RIGHT_PAREN statement_no_new_scope  */
#line 3864 "MachineIndependent/glslang.y"
                                                                               {
        parseContext.symbolTable.pop(&parseContext.defaultPrecision[0]);
        (yyval.interm.intermNode) = parseContext.intermediate.makeAggregate((yyvsp[-3].interm.intermNode), (yyvsp[-5].lex).loc);
        TIntermLoop* forLoop = parseContext.intermediate.addLoop((yyvsp[0].interm.intermNode), reinterpret_cast<TIntermTyped*>((yyvsp[-2].interm.nodePair).node1), reinterpret_cast<TIntermTyped*>((yyvsp[-2].interm.nodePair).node2), true, (yyvsp[-6].lex).loc);
        if (! parseContext.limits.nonInductiveForLoops)
            parseContext.inductiveLoopCheck((yyvsp[-6].lex).loc, (yyvsp[-3].interm.intermNode), forLoop);
        (yyval.interm.intermNode) = parseContext.intermediate.growAggregate((yyval.interm.intermNode), forLoop, (yyvsp[-6].lex).loc);
        (yyval.interm.intermNode)->getAsAggregate()->setOperator(EOpSequence);
        --parseContext.loopNestingLevel;
        --parseContext.statementNestingLevel;
        --parseContext.controlFlowNestingLevel;
    }
#line 10734 "MachineIndependent/glslang_tab.cpp"
    break;

  case 591: /* for_init_statement: expression_statement  */
#line 3879 "MachineIndependent/glslang.y"
                           {
        (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode);
    }
#line 10742 "MachineIndependent/glslang_tab.cpp"
    break;

  case 592: /* for_init_statement: declaration_statement  */
#line 3882 "MachineIndependent/glslang.y"
                            {
        (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode);
    }
#line 10750 "MachineIndependent/glslang_tab.cpp"
    break;

  case 593: /* conditionopt: condition  */
#line 3888 "MachineIndependent/glslang.y"
                {
        (yyval.interm.intermTypedNode) = (yyvsp[0].interm.intermTypedNode);
    }
#line 10758 "MachineIndependent/glslang_tab.cpp"
    break;

  case 594: /* conditionopt: %empty  */
#line 3891 "MachineIndependent/glslang.y"
                        {
        (yyval.interm.intermTypedNode) = 0;
    }
#line 10766 "MachineIndependent/glslang_tab.cpp"
    break;

  case 595: /* for_rest_statement: conditionopt SEMICOLON  */
#line 3897 "MachineIndependent/glslang.y"
                             {
        (yyval.interm.nodePair).node1 = (yyvsp[-1].interm.intermTypedNode);
        (yyval.interm.nodePair).node2 = 0;
    }
#line 10775 "MachineIndependent/glslang_tab.cpp"
    break;

  case 596: /* for_rest_statement: conditionopt SEMICOLON expression  */
#line 3901 "MachineIndependent/glslang.y"
                                         {
        (yyval.interm.nodePair).node1 = (yyvsp[-2].interm.intermTypedNode);
        (yyval.interm.nodePair).node2 = (yyvsp[0].interm.intermTypedNode);
    }
#line 10784 "MachineIndependent/glslang_tab.cpp"
    break;

  case 597: /* jump_statement: CONTINUE SEMICOLON  */
#line 3908 "MachineIndependent/glslang.y"
                         {
        if (parseContext.loopNestingLevel <= 0)
            parseContext.error((yyvsp[-1].lex).loc, "continue statement only allowed in loops", "", "");
        (yyval.interm.intermNode) = parseContext.intermediate.addBranch(EOpContinue, (yyvsp[-1].lex).loc);
    }
#line 10794 "MachineIndependent/glslang_tab.cpp"
    break;

  case 598: /* jump_statement: BREAK SEMICOLON  */
#line 3913 "MachineIndependent/glslang.y"
                      {
        if (parseContext.loopNestingLevel + parseContext.switchSequenceStack.size() <= 0)
            parseContext.error((yyvsp[-1].lex).loc, "break statement only allowed in switch and loops", "", "");
        (yyval.interm.intermNode) = parseContext.intermediate.addBranch(EOpBreak, (yyvsp[-1].lex).loc);
    }
#line 10804 "MachineIndependent/glslang_tab.cpp"
    break;

  case 599: /* jump_statement: RETURN SEMICOLON  */
#line 3918 "MachineIndependent/glslang.y"
                       {
        (yyval.interm.intermNode) = parseContext.intermediate.addBranch(EOpReturn, (yyvsp[-1].lex).loc);
        if (parseContext.currentFunctionType->getBasicType() != EbtVoid)
            parseContext.error((yyvsp[-1].lex).loc, "non-void function must return a value", "return", "");
        if (parseContext.inMain)
            parseContext.postEntryPointReturn = true;
    }
#line 10816 "MachineIndependent/glslang_tab.cpp"
    break;

  case 600: /* jump_statement: RETURN expression SEMICOLON  */
#line 3925 "MachineIndependent/glslang.y"
                                  {
        (yyval.interm.intermNode) = parseContext.handleReturnValue((yyvsp[-2].lex).loc, (yyvsp[-1].interm.intermTypedNode));
    }
#line 10824 "MachineIndependent/glslang_tab.cpp"
    break;

  case 601: /* jump_statement: DISCARD SEMICOLON  */
#line 3928 "MachineIndependent/glslang.y"
                        {
        parseContext.requireStage((yyvsp[-1].lex).loc, EShLangFragment, "discard");
        (yyval.interm.intermNode) = parseContext.intermediate.addBranch(EOpKill, (yyvsp[-1].lex).loc);
    }
#line 10833 "MachineIndependent/glslang_tab.cpp"
    break;

  case 602: /* jump_statement: TERMINATE_INVOCATION SEMICOLON  */
#line 3932 "MachineIndependent/glslang.y"
                                     {
        parseContext.requireStage((yyvsp[-1].lex).loc, EShLangFragment, "terminateInvocation");
        (yyval.interm.intermNode) = parseContext.intermediate.addBranch(EOpTerminateInvocation, (yyvsp[-1].lex).loc);
    }
#line 10842 "MachineIndependent/glslang_tab.cpp"
    break;

  case 603: /* jump_statement: TERMINATE_RAY SEMICOLON  */
#line 3937 "MachineIndependent/glslang.y"
                              {
        parseContext.requireStage((yyvsp[-1].lex).loc, EShLangAnyHit, "terminateRayEXT");
        (yyval.interm.intermNode) = parseContext.intermediate.addBranch(EOpTerminateRayKHR, (yyvsp[-1].lex).loc);
    }
#line 10851 "MachineIndependent/glslang_tab.cpp"
    break;

  case 604: /* jump_statement: IGNORE_INTERSECTION SEMICOLON  */
#line 3941 "MachineIndependent/glslang.y"
                                    {
        parseContext.requireStage((yyvsp[-1].lex).loc, EShLangAnyHit, "ignoreIntersectionEXT");
        (yyval.interm.intermNode) = parseContext.intermediate.addBranch(EOpIgnoreIntersectionKHR, (yyvsp[-1].lex).loc);
    }
#line 10860 "MachineIndependent/glslang_tab.cpp"
    break;

  case 605: /* translation_unit: external_declaration  */
#line 3951 "MachineIndependent/glslang.y"
                           {
        (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode);
        parseContext.intermediate.setTreeRoot((yyval.interm.intermNode));
    }
#line 10869 "MachineIndependent/glslang_tab.cpp"
    break;

  case 606: /* translation_unit: translation_unit external_declaration  */
#line 3955 "MachineIndependent/glslang.y"
                                            {
        if ((yyvsp[0].interm.intermNode) != nullptr) {
            (yyval.interm.intermNode) = parseContext.intermediate.growAggregate((yyvsp[-1].interm.intermNode), (yyvsp[0].interm.intermNode));
            parseContext.intermediate.setTreeRoot((yyval.interm.intermNode));
        }
    }
#line 10880 "MachineIndependent/glslang_tab.cpp"
    break;

  case 607: /* external_declaration: function_definition  */
#line 3964 "MachineIndependent/glslang.y"
                          {
        (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode);
    }
#line 10888 "MachineIndependent/glslang_tab.cpp"
    break;

  case 608: /* external_declaration: declaration  */
#line 3967 "MachineIndependent/glslang.y"
                  {
        (yyval.interm.intermNode) = (yyvsp[0].interm.intermNode);
    }
#line 10896 "MachineIndependent/glslang_tab.cpp"
    break;

  case 609: /* external_declaration: SEMICOLON  */
#line 3971 "MachineIndependent/glslang.y"
                {
        parseContext.requireProfile((yyvsp[0].lex).loc, ~EEsProfile, "extraneous semicolon");
        parseContext.profileRequires((yyvsp[0].lex).loc, ~EEsProfile, 460, nullptr, "extraneous semicolon");
        (yyval.interm.intermNode) = nullptr;
    }
#line 10906 "MachineIndependent/glslang_tab.cpp"
    break;

  case 610: /* $@13: %empty  */
#line 3980 "MachineIndependent/glslang.y"
                         {
        (yyvsp[0].interm).function = parseContext.handleFunctionDeclarator((yyvsp[0].interm).loc, *(yyvsp[0].interm).function, false /* not prototype */);
        (yyvsp[0].interm).intermNode = parseContext.handleFunctionDefinition((yyvsp[0].interm).loc, *(yyvsp[0].interm).function);

        // For ES 100 only, according to ES shading language 100 spec: A function
        // body has a scope nested inside the function's definition.
        if (parseContext.profile == EEsProfile && parseContext.version == 100)
        {
            parseContext.symbolTable.push();
            ++parseContext.statementNestingLevel;
        }
    }
#line 10923 "MachineIndependent/glslang_tab.cpp"
    break;

  case 611: /* function_definition: function_prototype $@13 compound_statement_no_new_scope  */
#line 3992 "MachineIndependent/glslang.y"
                                    {
        //   May be best done as post process phase on intermediate code
        if (parseContext.currentFunctionType->getBasicType() != EbtVoid && ! parseContext.functionReturnsValue)
            parseContext.error((yyvsp[-2].interm).loc, "function does not return a value:", "", (yyvsp[-2].interm).function->getName().c_str());
        parseContext.symbolTable.pop(&parseContext.defaultPrecision[0]);
        (yyval.interm.intermNode) = parseContext.intermediate.growAggregate((yyvsp[-2].interm).intermNode, (yyvsp[0].interm.intermNode));
        parseContext.intermediate.setAggregateOperator((yyval.interm.intermNode), EOpFunction, (yyvsp[-2].interm).function->getType(), (yyvsp[-2].interm).loc);
        (yyval.interm.intermNode)->getAsAggregate()->setName((yyvsp[-2].interm).function->getMangledName().c_str());

        // store the pragma information for debug and optimize and other vendor specific
        // information. This information can be queried from the parse tree
        (yyval.interm.intermNode)->getAsAggregate()->setOptimize(parseContext.contextPragma.optimize);
        (yyval.interm.intermNode)->getAsAggregate()->setDebug(parseContext.contextPragma.debug);
        (yyval.interm.intermNode)->getAsAggregate()->setPragmaTable(parseContext.contextPragma.pragmaTable);

        // Set currentFunctionType to empty pointer when goes outside of the function
        parseContext.currentFunctionType = nullptr;

        // For ES 100 only, according to ES shading language 100 spec: A function
        // body has a scope nested inside the function's definition.
        if (parseContext.profile == EEsProfile && parseContext.version == 100)
        {
            parseContext.symbolTable.pop(&parseContext.defaultPrecision[0]);
            --parseContext.statementNestingLevel;
        }
    }
#line 10954 "MachineIndependent/glslang_tab.cpp"
    break;

  case 612: /* attribute: LEFT_BRACKET LEFT_BRACKET attribute_list RIGHT_BRACKET RIGHT_BRACKET  */
#line 4022 "MachineIndependent/glslang.y"
                                                                           {
        (yyval.interm.attributes) = (yyvsp[-2].interm.attributes);
        parseContext.requireExtensions((yyvsp[-4].lex).loc, 1, &E_GL_EXT_control_flow_attributes, "attribute");
    }
#line 10963 "MachineIndependent/glslang_tab.cpp"
    break;

  case 613: /* attribute_list: single_attribute  */
#line 4028 "MachineIndependent/glslang.y"
                       {
        (yyval.interm.attributes) = (yyvsp[0].interm.attributes);
    }
#line 10971 "MachineIndependent/glslang_tab.cpp"
    break;

  case 614: /* attribute_list: attribute_list COMMA single_attribute  */
#line 4031 "MachineIndependent/glslang.y"
                                            {
        (yyval.interm.attributes) = parseContext.mergeAttributes((yyvsp[-2].interm.attributes), (yyvsp[0].interm.attributes));
    }
#line 10979 "MachineIndependent/glslang_tab.cpp"
    break;

  case 615: /* single_attribute: IDENTIFIER  */
#line 4036 "MachineIndependent/glslang.y"
                 {
        (yyval.interm.attributes) = parseContext.makeAttributes(*(yyvsp[0].lex).string);
    }
#line 10987 "MachineIndependent/glslang_tab.cpp"
    break;

  case 616: /* single_attribute: IDENTIFIER LEFT_PAREN constant_expression RIGHT_PAREN  */
#line 4039 "MachineIndependent/glslang.y"
                                                            {
        (yyval.interm.attributes) = parseContext.makeAttributes(*(yyvsp[-3].lex).string, (yyvsp[-1].interm.intermTypedNode));
    }
#line 10995 "MachineIndependent/glslang_tab.cpp"
    break;


#line 10999 "MachineIndependent/glslang_tab.cpp"

      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", YY_CAST (yysymbol_kind_t, yyr1[yyn]), &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYSYMBOL_YYEMPTY : YYTRANSLATE (yychar);
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
      {
        yypcontext_t yyctx
          = {yyssp, yytoken};
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = yysyntax_error (&yymsg_alloc, &yymsg, &yyctx);
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == -1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = YY_CAST (char *,
                             YYSTACK_ALLOC (YY_CAST (YYSIZE_T, yymsg_alloc)));
            if (yymsg)
              {
                yysyntax_error_status
                  = yysyntax_error (&yymsg_alloc, &yymsg, &yyctx);
                yymsgp = yymsg;
              }
            else
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = YYENOMEM;
              }
          }
        yyerror (pParseContext, yymsgp);
        if (yysyntax_error_status == YYENOMEM)
          goto yyexhaustedlab;
      }
    }

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval, pParseContext);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  /* Pop stack until we find a state that shifts the error token.  */
  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYSYMBOL_YYerror;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYSYMBOL_YYerror)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  YY_ACCESSING_SYMBOL (yystate), yyvsp, pParseContext);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", YY_ACCESSING_SYMBOL (yyn), yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;


/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;


#if 1
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (pParseContext, YY_("memory exhausted"));
  yyresult = 2;
  goto yyreturn;
#endif


/*-------------------------------------------------------.
| yyreturn -- parsing is finished, clean up and return.  |
`-------------------------------------------------------*/
yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval, pParseContext);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  YY_ACCESSING_SYMBOL (+*yyssp), yyvsp, pParseContext);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
  return yyresult;
}

#line 4044 "MachineIndependent/glslang.y"

