/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
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
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

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

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

#ifndef YY_YY_MACHINEINDEPENDENT_GLSLANG_TAB_CPP_H_INCLUDED
# define YY_YY_MACHINEINDEPENDENT_GLSLANG_TAB_CPP_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 1
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token kinds.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    YYEMPTY = -2,
    YYEOF = 0,                     /* "end of file"  */
    YYerror = 256,                 /* error  */
    YYUNDEF = 257,                 /* "invalid token"  */
    CONST = 258,                   /* CONST  */
    BOOL = 259,                    /* BOOL  */
    INT = 260,                     /* INT  */
    UINT = 261,                    /* UINT  */
    FLOAT = 262,                   /* FLOAT  */
    BVEC2 = 263,                   /* BVEC2  */
    BVEC3 = 264,                   /* BVEC3  */
    BVEC4 = 265,                   /* BVEC4  */
    IVEC2 = 266,                   /* IVEC2  */
    IVEC3 = 267,                   /* IVEC3  */
    IVEC4 = 268,                   /* IVEC4  */
    UVEC2 = 269,                   /* UVEC2  */
    UVEC3 = 270,                   /* UVEC3  */
    UVEC4 = 271,                   /* UVEC4  */
    VEC2 = 272,                    /* VEC2  */
    VEC3 = 273,                    /* VEC3  */
    VEC4 = 274,                    /* VEC4  */
    MAT2 = 275,                    /* MAT2  */
    MAT3 = 276,                    /* MAT3  */
    MAT4 = 277,                    /* MAT4  */
    MAT2X2 = 278,                  /* MAT2X2  */
    MAT2X3 = 279,                  /* MAT2X3  */
    MAT2X4 = 280,                  /* MAT2X4  */
    MAT3X2 = 281,                  /* MAT3X2  */
    MAT3X3 = 282,                  /* MAT3X3  */
    MAT3X4 = 283,                  /* MAT3X4  */
    MAT4X2 = 284,                  /* MAT4X2  */
    MAT4X3 = 285,                  /* MAT4X3  */
    MAT4X4 = 286,                  /* MAT4X4  */
    SAMPLER2D = 287,               /* SAMPLER2D  */
    SAMPLER3D = 288,               /* SAMPLER3D  */
    SAMPLERCUBE = 289,             /* SAMPLERCUBE  */
    SAMPLER2DSHADOW = 290,         /* SAMPLER2DSHADOW  */
    SAMPLERCUBESHADOW = 291,       /* SAMPLERCUBESHADOW  */
    SAMPLER2DARRAY = 292,          /* SAMPLER2DARRAY  */
    SAMPLER2DARRAYSHADOW = 293,    /* SAMPLER2DARRAYSHADOW  */
    ISAMPLER2D = 294,              /* ISAMPLER2D  */
    ISAMPLER3D = 295,              /* ISAMPLER3D  */
    ISAMPLERCUBE = 296,            /* ISAMPLERCUBE  */
    ISAMPLER2DARRAY = 297,         /* ISAMPLER2DARRAY  */
    USAMPLER2D = 298,              /* USAMPLER2D  */
    USAMPLER3D = 299,              /* USAMPLER3D  */
    USAMPLERCUBE = 300,            /* USAMPLERCUBE  */
    USAMPLER2DARRAY = 301,         /* USAMPLER2DARRAY  */
    SAMPLER = 302,                 /* SAMPLER  */
    SAMPLERSHADOW = 303,           /* SAMPLERSHADOW  */
    TEXTURE2D = 304,               /* TEXTURE2D  */
    TEXTURE3D = 305,               /* TEXTURE3D  */
    TEXTURECUBE = 306,             /* TEXTURECUBE  */
    TEXTURE2DARRAY = 307,          /* TEXTURE2DARRAY  */
    ITEXTURE2D = 308,              /* ITEXTURE2D  */
    ITEXTURE3D = 309,              /* ITEXTURE3D  */
    ITEXTURECUBE = 310,            /* ITEXTURECUBE  */
    ITEXTURE2DARRAY = 311,         /* ITEXTURE2DARRAY  */
    UTEXTURE2D = 312,              /* UTEXTURE2D  */
    UTEXTURE3D = 313,              /* UTEXTURE3D  */
    UTEXTURECUBE = 314,            /* UTEXTURECUBE  */
    UTEXTURE2DARRAY = 315,         /* UTEXTURE2DARRAY  */
    ATTRIBUTE = 316,               /* ATTRIBUTE  */
    VARYING = 317,                 /* VARYING  */
    BFLOAT16_T = 318,              /* BFLOAT16_T  */
    FLOAT16_T = 319,               /* FLOAT16_T  */
    FLOAT32_T = 320,               /* FLOAT32_T  */
    DOUBLE = 321,                  /* DOUBLE  */
    FLOAT64_T = 322,               /* FLOAT64_T  */
    INT64_T = 323,                 /* INT64_T  */
    UINT64_T = 324,                /* UINT64_T  */
    INT32_T = 325,                 /* INT32_T  */
    UINT32_T = 326,                /* UINT32_T  */
    INT16_T = 327,                 /* INT16_T  */
    UINT16_T = 328,                /* UINT16_T  */
    INT8_T = 329,                  /* INT8_T  */
    UINT8_T = 330,                 /* UINT8_T  */
    I64VEC2 = 331,                 /* I64VEC2  */
    I64VEC3 = 332,                 /* I64VEC3  */
    I64VEC4 = 333,                 /* I64VEC4  */
    U64VEC2 = 334,                 /* U64VEC2  */
    U64VEC3 = 335,                 /* U64VEC3  */
    U64VEC4 = 336,                 /* U64VEC4  */
    I32VEC2 = 337,                 /* I32VEC2  */
    I32VEC3 = 338,                 /* I32VEC3  */
    I32VEC4 = 339,                 /* I32VEC4  */
    U32VEC2 = 340,                 /* U32VEC2  */
    U32VEC3 = 341,                 /* U32VEC3  */
    U32VEC4 = 342,                 /* U32VEC4  */
    I16VEC2 = 343,                 /* I16VEC2  */
    I16VEC3 = 344,                 /* I16VEC3  */
    I16VEC4 = 345,                 /* I16VEC4  */
    U16VEC2 = 346,                 /* U16VEC2  */
    U16VEC3 = 347,                 /* U16VEC3  */
    U16VEC4 = 348,                 /* U16VEC4  */
    I8VEC2 = 349,                  /* I8VEC2  */
    I8VEC3 = 350,                  /* I8VEC3  */
    I8VEC4 = 351,                  /* I8VEC4  */
    U8VEC2 = 352,                  /* U8VEC2  */
    U8VEC3 = 353,                  /* U8VEC3  */
    U8VEC4 = 354,                  /* U8VEC4  */
    DVEC2 = 355,                   /* DVEC2  */
    DVEC3 = 356,                   /* DVEC3  */
    DVEC4 = 357,                   /* DVEC4  */
    DMAT2 = 358,                   /* DMAT2  */
    DMAT3 = 359,                   /* DMAT3  */
    DMAT4 = 360,                   /* DMAT4  */
    BF16VEC2 = 361,                /* BF16VEC2  */
    BF16VEC3 = 362,                /* BF16VEC3  */
    BF16VEC4 = 363,                /* BF16VEC4  */
    F16VEC2 = 364,                 /* F16VEC2  */
    F16VEC3 = 365,                 /* F16VEC3  */
    F16VEC4 = 366,                 /* F16VEC4  */
    F16MAT2 = 367,                 /* F16MAT2  */
    F16MAT3 = 368,                 /* F16MAT3  */
    F16MAT4 = 369,                 /* F16MAT4  */
    F32VEC2 = 370,                 /* F32VEC2  */
    F32VEC3 = 371,                 /* F32VEC3  */
    F32VEC4 = 372,                 /* F32VEC4  */
    F32MAT2 = 373,                 /* F32MAT2  */
    F32MAT3 = 374,                 /* F32MAT3  */
    F32MAT4 = 375,                 /* F32MAT4  */
    F64VEC2 = 376,                 /* F64VEC2  */
    F64VEC3 = 377,                 /* F64VEC3  */
    F64VEC4 = 378,                 /* F64VEC4  */
    F64MAT2 = 379,                 /* F64MAT2  */
    F64MAT3 = 380,                 /* F64MAT3  */
    F64MAT4 = 381,                 /* F64MAT4  */
    DMAT2X2 = 382,                 /* DMAT2X2  */
    DMAT2X3 = 383,                 /* DMAT2X3  */
    DMAT2X4 = 384,                 /* DMAT2X4  */
    DMAT3X2 = 385,                 /* DMAT3X2  */
    DMAT3X3 = 386,                 /* DMAT3X3  */
    DMAT3X4 = 387,                 /* DMAT3X4  */
    DMAT4X2 = 388,                 /* DMAT4X2  */
    DMAT4X3 = 389,                 /* DMAT4X3  */
    DMAT4X4 = 390,                 /* DMAT4X4  */
    F16MAT2X2 = 391,               /* F16MAT2X2  */
    F16MAT2X3 = 392,               /* F16MAT2X3  */
    F16MAT2X4 = 393,               /* F16MAT2X4  */
    F16MAT3X2 = 394,               /* F16MAT3X2  */
    F16MAT3X3 = 395,               /* F16MAT3X3  */
    F16MAT3X4 = 396,               /* F16MAT3X4  */
    F16MAT4X2 = 397,               /* F16MAT4X2  */
    F16MAT4X3 = 398,               /* F16MAT4X3  */
    F16MAT4X4 = 399,               /* F16MAT4X4  */
    F32MAT2X2 = 400,               /* F32MAT2X2  */
    F32MAT2X3 = 401,               /* F32MAT2X3  */
    F32MAT2X4 = 402,               /* F32MAT2X4  */
    F32MAT3X2 = 403,               /* F32MAT3X2  */
    F32MAT3X3 = 404,               /* F32MAT3X3  */
    F32MAT3X4 = 405,               /* F32MAT3X4  */
    F32MAT4X2 = 406,               /* F32MAT4X2  */
    F32MAT4X3 = 407,               /* F32MAT4X3  */
    F32MAT4X4 = 408,               /* F32MAT4X4  */
    F64MAT2X2 = 409,               /* F64MAT2X2  */
    F64MAT2X3 = 410,               /* F64MAT2X3  */
    F64MAT2X4 = 411,               /* F64MAT2X4  */
    F64MAT3X2 = 412,               /* F64MAT3X2  */
    F64MAT3X3 = 413,               /* F64MAT3X3  */
    F64MAT3X4 = 414,               /* F64MAT3X4  */
    F64MAT4X2 = 415,               /* F64MAT4X2  */
    F64MAT4X3 = 416,               /* F64MAT4X3  */
    F64MAT4X4 = 417,               /* F64MAT4X4  */
    ATOMIC_UINT = 418,             /* ATOMIC_UINT  */
    ACCSTRUCTNV = 419,             /* ACCSTRUCTNV  */
    ACCSTRUCTEXT = 420,            /* ACCSTRUCTEXT  */
    RAYQUERYEXT = 421,             /* RAYQUERYEXT  */
    FCOOPMATNV = 422,              /* FCOOPMATNV  */
    ICOOPMATNV = 423,              /* ICOOPMATNV  */
    UCOOPMATNV = 424,              /* UCOOPMATNV  */
    COOPMAT = 425,                 /* COOPMAT  */
    COOPVECNV = 426,               /* COOPVECNV  */
    HITOBJECTNV = 427,             /* HITOBJECTNV  */
    HITOBJECTATTRNV = 428,         /* HITOBJECTATTRNV  */
    TENSORLAYOUTNV = 429,          /* TENSORLAYOUTNV  */
    TENSORVIEWNV = 430,            /* TENSORVIEWNV  */
    SAMPLERCUBEARRAY = 431,        /* SAMPLERCUBEARRAY  */
    SAMPLERCUBEARRAYSHADOW = 432,  /* SAMPLERCUBEARRAYSHADOW  */
    ISAMPLERCUBEARRAY = 433,       /* ISAMPLERCUBEARRAY  */
    USAMPLERCUBEARRAY = 434,       /* USAMPLERCUBEARRAY  */
    SAMPLER1D = 435,               /* SAMPLER1D  */
    SAMPLER1DARRAY = 436,          /* SAMPLER1DARRAY  */
    SAMPLER1DARRAYSHADOW = 437,    /* SAMPLER1DARRAYSHADOW  */
    ISAMPLER1D = 438,              /* ISAMPLER1D  */
    SAMPLER1DSHADOW = 439,         /* SAMPLER1DSHADOW  */
    SAMPLER2DRECT = 440,           /* SAMPLER2DRECT  */
    SAMPLER2DRECTSHADOW = 441,     /* SAMPLER2DRECTSHADOW  */
    ISAMPLER2DRECT = 442,          /* ISAMPLER2DRECT  */
    USAMPLER2DRECT = 443,          /* USAMPLER2DRECT  */
    SAMPLERBUFFER = 444,           /* SAMPLERBUFFER  */
    ISAMPLERBUFFER = 445,          /* ISAMPLERBUFFER  */
    USAMPLERBUFFER = 446,          /* USAMPLERBUFFER  */
    SAMPLER2DMS = 447,             /* SAMPLER2DMS  */
    ISAMPLER2DMS = 448,            /* ISAMPLER2DMS  */
    USAMPLER2DMS = 449,            /* USAMPLER2DMS  */
    SAMPLER2DMSARRAY = 450,        /* SAMPLER2DMSARRAY  */
    ISAMPLER2DMSARRAY = 451,       /* ISAMPLER2DMSARRAY  */
    USAMPLER2DMSARRAY = 452,       /* USAMPLER2DMSARRAY  */
    SAMPLEREXTERNALOES = 453,      /* SAMPLEREXTERNALOES  */
    SAMPLEREXTERNAL2DY2YEXT = 454, /* SAMPLEREXTERNAL2DY2YEXT  */
    ISAMPLER1DARRAY = 455,         /* ISAMPLER1DARRAY  */
    USAMPLER1D = 456,              /* USAMPLER1D  */
    USAMPLER1DARRAY = 457,         /* USAMPLER1DARRAY  */
    F16SAMPLER1D = 458,            /* F16SAMPLER1D  */
    F16SAMPLER2D = 459,            /* F16SAMPLER2D  */
    F16SAMPLER3D = 460,            /* F16SAMPLER3D  */
    F16SAMPLER2DRECT = 461,        /* F16SAMPLER2DRECT  */
    F16SAMPLERCUBE = 462,          /* F16SAMPLERCUBE  */
    F16SAMPLER1DARRAY = 463,       /* F16SAMPLER1DARRAY  */
    F16SAMPLER2DARRAY = 464,       /* F16SAMPLER2DARRAY  */
    F16SAMPLERCUBEARRAY = 465,     /* F16SAMPLERCUBEARRAY  */
    F16SAMPLERBUFFER = 466,        /* F16SAMPLERBUFFER  */
    F16SAMPLER2DMS = 467,          /* F16SAMPLER2DMS  */
    F16SAMPLER2DMSARRAY = 468,     /* F16SAMPLER2DMSARRAY  */
    F16SAMPLER1DSHADOW = 469,      /* F16SAMPLER1DSHADOW  */
    F16SAMPLER2DSHADOW = 470,      /* F16SAMPLER2DSHADOW  */
    F16SAMPLER1DARRAYSHADOW = 471, /* F16SAMPLER1DARRAYSHADOW  */
    F16SAMPLER2DARRAYSHADOW = 472, /* F16SAMPLER2DARRAYSHADOW  */
    F16SAMPLER2DRECTSHADOW = 473,  /* F16SAMPLER2DRECTSHADOW  */
    F16SAMPLERCUBESHADOW = 474,    /* F16SAMPLERCUBESHADOW  */
    F16SAMPLERCUBEARRAYSHADOW = 475, /* F16SAMPLERCUBEARRAYSHADOW  */
    IMAGE1D = 476,                 /* IMAGE1D  */
    IIMAGE1D = 477,                /* IIMAGE1D  */
    UIMAGE1D = 478,                /* UIMAGE1D  */
    IMAGE2D = 479,                 /* IMAGE2D  */
    IIMAGE2D = 480,                /* IIMAGE2D  */
    UIMAGE2D = 481,                /* UIMAGE2D  */
    IMAGE3D = 482,                 /* IMAGE3D  */
    IIMAGE3D = 483,                /* IIMAGE3D  */
    UIMAGE3D = 484,                /* UIMAGE3D  */
    IMAGE2DRECT = 485,             /* IMAGE2DRECT  */
    IIMAGE2DRECT = 486,            /* IIMAGE2DRECT  */
    UIMAGE2DRECT = 487,            /* UIMAGE2DRECT  */
    IMAGECUBE = 488,               /* IMAGECUBE  */
    IIMAGECUBE = 489,              /* IIMAGECUBE  */
    UIMAGECUBE = 490,              /* UIMAGECUBE  */
    IMAGEBUFFER = 491,             /* IMAGEBUFFER  */
    IIMAGEBUFFER = 492,            /* IIMAGEBUFFER  */
    UIMAGEBUFFER = 493,            /* UIMAGEBUFFER  */
    IMAGE1DARRAY = 494,            /* IMAGE1DARRAY  */
    IIMAGE1DARRAY = 495,           /* IIMAGE1DARRAY  */
    UIMAGE1DARRAY = 496,           /* UIMAGE1DARRAY  */
    IMAGE2DARRAY = 497,            /* IMAGE2DARRAY  */
    IIMAGE2DARRAY = 498,           /* IIMAGE2DARRAY  */
    UIMAGE2DARRAY = 499,           /* UIMAGE2DARRAY  */
    IMAGECUBEARRAY = 500,          /* IMAGECUBEARRAY  */
    IIMAGECUBEARRAY = 501,         /* IIMAGECUBEARRAY  */
    UIMAGECUBEARRAY = 502,         /* UIMAGECUBEARRAY  */
    IMAGE2DMS = 503,               /* IMAGE2DMS  */
    IIMAGE2DMS = 504,              /* IIMAGE2DMS  */
    UIMAGE2DMS = 505,              /* UIMAGE2DMS  */
    IMAGE2DMSARRAY = 506,          /* IMAGE2DMSARRAY  */
    IIMAGE2DMSARRAY = 507,         /* IIMAGE2DMSARRAY  */
    UIMAGE2DMSARRAY = 508,         /* UIMAGE2DMSARRAY  */
    F16IMAGE1D = 509,              /* F16IMAGE1D  */
    F16IMAGE2D = 510,              /* F16IMAGE2D  */
    F16IMAGE3D = 511,              /* F16IMAGE3D  */
    F16IMAGE2DRECT = 512,          /* F16IMAGE2DRECT  */
    F16IMAGECUBE = 513,            /* F16IMAGECUBE  */
    F16IMAGE1DARRAY = 514,         /* F16IMAGE1DARRAY  */
    F16IMAGE2DARRAY = 515,         /* F16IMAGE2DARRAY  */
    F16IMAGECUBEARRAY = 516,       /* F16IMAGECUBEARRAY  */
    F16IMAGEBUFFER = 517,          /* F16IMAGEBUFFER  */
    F16IMAGE2DMS = 518,            /* F16IMAGE2DMS  */
    F16IMAGE2DMSARRAY = 519,       /* F16IMAGE2DMSARRAY  */
    I64IMAGE1D = 520,              /* I64IMAGE1D  */
    U64IMAGE1D = 521,              /* U64IMAGE1D  */
    I64IMAGE2D = 522,              /* I64IMAGE2D  */
    U64IMAGE2D = 523,              /* U64IMAGE2D  */
    I64IMAGE3D = 524,              /* I64IMAGE3D  */
    U64IMAGE3D = 525,              /* U64IMAGE3D  */
    I64IMAGE2DRECT = 526,          /* I64IMAGE2DRECT  */
    U64IMAGE2DRECT = 527,          /* U64IMAGE2DRECT  */
    I64IMAGECUBE = 528,            /* I64IMAGECUBE  */
    U64IMAGECUBE = 529,            /* U64IMAGECUBE  */
    I64IMAGEBUFFER = 530,          /* I64IMAGEBUFFER  */
    U64IMAGEBUFFER = 531,          /* U64IMAGEBUFFER  */
    I64IMAGE1DARRAY = 532,         /* I64IMAGE1DARRAY  */
    U64IMAGE1DARRAY = 533,         /* U64IMAGE1DARRAY  */
    I64IMAGE2DARRAY = 534,         /* I64IMAGE2DARRAY  */
    U64IMAGE2DARRAY = 535,         /* U64IMAGE2DARRAY  */
    I64IMAGECUBEARRAY = 536,       /* I64IMAGECUBEARRAY  */
    U64IMAGECUBEARRAY = 537,       /* U64IMAGECUBEARRAY  */
    I64IMAGE2DMS = 538,            /* I64IMAGE2DMS  */
    U64IMAGE2DMS = 539,            /* U64IMAGE2DMS  */
    I64IMAGE2DMSARRAY = 540,       /* I64IMAGE2DMSARRAY  */
    U64IMAGE2DMSARRAY = 541,       /* U64IMAGE2DMSARRAY  */
    TEXTURECUBEARRAY = 542,        /* TEXTURECUBEARRAY  */
    ITEXTURECUBEARRAY = 543,       /* ITEXTURECUBEARRAY  */
    UTEXTURECUBEARRAY = 544,       /* UTEXTURECUBEARRAY  */
    TEXTURE1D = 545,               /* TEXTURE1D  */
    ITEXTURE1D = 546,              /* ITEXTURE1D  */
    UTEXTURE1D = 547,              /* UTEXTURE1D  */
    TEXTURE1DARRAY = 548,          /* TEXTURE1DARRAY  */
    ITEXTURE1DARRAY = 549,         /* ITEXTURE1DARRAY  */
    UTEXTURE1DARRAY = 550,         /* UTEXTURE1DARRAY  */
    TEXTURE2DRECT = 551,           /* TEXTURE2DRECT  */
    ITEXTURE2DRECT = 552,          /* ITEXTURE2DRECT  */
    UTEXTURE2DRECT = 553,          /* UTEXTURE2DRECT  */
    TEXTUREBUFFER = 554,           /* TEXTUREBUFFER  */
    ITEXTUREBUFFER = 555,          /* ITEXTUREBUFFER  */
    UTEXTUREBUFFER = 556,          /* UTEXTUREBUFFER  */
    TEXTURE2DMS = 557,             /* TEXTURE2DMS  */
    ITEXTURE2DMS = 558,            /* ITEXTURE2DMS  */
    UTEXTURE2DMS = 559,            /* UTEXTURE2DMS  */
    TEXTURE2DMSARRAY = 560,        /* TEXTURE2DMSARRAY  */
    ITEXTURE2DMSARRAY = 561,       /* ITEXTURE2DMSARRAY  */
    UTEXTURE2DMSARRAY = 562,       /* UTEXTURE2DMSARRAY  */
    F16TEXTURE1D = 563,            /* F16TEXTURE1D  */
    F16TEXTURE2D = 564,            /* F16TEXTURE2D  */
    F16TEXTURE3D = 565,            /* F16TEXTURE3D  */
    F16TEXTURE2DRECT = 566,        /* F16TEXTURE2DRECT  */
    F16TEXTURECUBE = 567,          /* F16TEXTURECUBE  */
    F16TEXTURE1DARRAY = 568,       /* F16TEXTURE1DARRAY  */
    F16TEXTURE2DARRAY = 569,       /* F16TEXTURE2DARRAY  */
    F16TEXTURECUBEARRAY = 570,     /* F16TEXTURECUBEARRAY  */
    F16TEXTUREBUFFER = 571,        /* F16TEXTUREBUFFER  */
    F16TEXTURE2DMS = 572,          /* F16TEXTURE2DMS  */
    F16TEXTURE2DMSARRAY = 573,     /* F16TEXTURE2DMSARRAY  */
    SUBPASSINPUT = 574,            /* SUBPASSINPUT  */
    SUBPASSINPUTMS = 575,          /* SUBPASSINPUTMS  */
    ISUBPASSINPUT = 576,           /* ISUBPASSINPUT  */
    ISUBPASSINPUTMS = 577,         /* ISUBPASSINPUTMS  */
    USUBPASSINPUT = 578,           /* USUBPASSINPUT  */
    USUBPASSINPUTMS = 579,         /* USUBPASSINPUTMS  */
    F16SUBPASSINPUT = 580,         /* F16SUBPASSINPUT  */
    F16SUBPASSINPUTMS = 581,       /* F16SUBPASSINPUTMS  */
    SPIRV_INSTRUCTION = 582,       /* SPIRV_INSTRUCTION  */
    SPIRV_EXECUTION_MODE = 583,    /* SPIRV_EXECUTION_MODE  */
    SPIRV_EXECUTION_MODE_ID = 584, /* SPIRV_EXECUTION_MODE_ID  */
    SPIRV_DECORATE = 585,          /* SPIRV_DECORATE  */
    SPIRV_DECORATE_ID = 586,       /* SPIRV_DECORATE_ID  */
    SPIRV_DECORATE_STRING = 587,   /* SPIRV_DECORATE_STRING  */
    SPIRV_TYPE = 588,              /* SPIRV_TYPE  */
    SPIRV_STORAGE_CLASS = 589,     /* SPIRV_STORAGE_CLASS  */
    SPIRV_BY_REFERENCE = 590,      /* SPIRV_BY_REFERENCE  */
    SPIRV_LITERAL = 591,           /* SPIRV_LITERAL  */
    ATTACHMENTEXT = 592,           /* ATTACHMENTEXT  */
    IATTACHMENTEXT = 593,          /* IATTACHMENTEXT  */
    UATTACHMENTEXT = 594,          /* UATTACHMENTEXT  */
    LEFT_OP = 595,                 /* LEFT_OP  */
    RIGHT_OP = 596,                /* RIGHT_OP  */
    INC_OP = 597,                  /* INC_OP  */
    DEC_OP = 598,                  /* DEC_OP  */
    LE_OP = 599,                   /* LE_OP  */
    GE_OP = 600,                   /* GE_OP  */
    EQ_OP = 601,                   /* EQ_OP  */
    NE_OP = 602,                   /* NE_OP  */
    AND_OP = 603,                  /* AND_OP  */
    OR_OP = 604,                   /* OR_OP  */
    XOR_OP = 605,                  /* XOR_OP  */
    MUL_ASSIGN = 606,              /* MUL_ASSIGN  */
    DIV_ASSIGN = 607,              /* DIV_ASSIGN  */
    ADD_ASSIGN = 608,              /* ADD_ASSIGN  */
    MOD_ASSIGN = 609,              /* MOD_ASSIGN  */
    LEFT_ASSIGN = 610,             /* LEFT_ASSIGN  */
    RIGHT_ASSIGN = 611,            /* RIGHT_ASSIGN  */
    AND_ASSIGN = 612,              /* AND_ASSIGN  */
    XOR_ASSIGN = 613,              /* XOR_ASSIGN  */
    OR_ASSIGN = 614,               /* OR_ASSIGN  */
    SUB_ASSIGN = 615,              /* SUB_ASSIGN  */
    STRING_LITERAL = 616,          /* STRING_LITERAL  */
    LEFT_PAREN = 617,              /* LEFT_PAREN  */
    RIGHT_PAREN = 618,             /* RIGHT_PAREN  */
    LEFT_BRACKET = 619,            /* LEFT_BRACKET  */
    RIGHT_BRACKET = 620,           /* RIGHT_BRACKET  */
    LEFT_BRACE = 621,              /* LEFT_BRACE  */
    RIGHT_BRACE = 622,             /* RIGHT_BRACE  */
    DOT = 623,                     /* DOT  */
    COMMA = 624,                   /* COMMA  */
    COLON = 625,                   /* COLON  */
    EQUAL = 626,                   /* EQUAL  */
    SEMICOLON = 627,               /* SEMICOLON  */
    BANG = 628,                    /* BANG  */
    DASH = 629,                    /* DASH  */
    TILDE = 630,                   /* TILDE  */
    PLUS = 631,                    /* PLUS  */
    STAR = 632,                    /* STAR  */
    SLASH = 633,                   /* SLASH  */
    PERCENT = 634,                 /* PERCENT  */
    LEFT_ANGLE = 635,              /* LEFT_ANGLE  */
    RIGHT_ANGLE = 636,             /* RIGHT_ANGLE  */
    VERTICAL_BAR = 637,            /* VERTICAL_BAR  */
    CARET = 638,                   /* CARET  */
    AMPERSAND = 639,               /* AMPERSAND  */
    QUESTION = 640,                /* QUESTION  */
    INVARIANT = 641,               /* INVARIANT  */
    HIGH_PRECISION = 642,          /* HIGH_PRECISION  */
    MEDIUM_PRECISION = 643,        /* MEDIUM_PRECISION  */
    LOW_PRECISION = 644,           /* LOW_PRECISION  */
    PRECISION = 645,               /* PRECISION  */
    PACKED = 646,                  /* PACKED  */
    RESOURCE = 647,                /* RESOURCE  */
    SUPERP = 648,                  /* SUPERP  */
    FLOATCONSTANT = 649,           /* FLOATCONSTANT  */
    INTCONSTANT = 650,             /* INTCONSTANT  */
    UINTCONSTANT = 651,            /* UINTCONSTANT  */
    BOOLCONSTANT = 652,            /* BOOLCONSTANT  */
    IDENTIFIER = 653,              /* IDENTIFIER  */
    TYPE_NAME = 654,               /* TYPE_NAME  */
    CENTROID = 655,                /* CENTROID  */
    IN = 656,                      /* IN  */
    OUT = 657,                     /* OUT  */
    INOUT = 658,                   /* INOUT  */
    STRUCT = 659,                  /* STRUCT  */
    VOID = 660,                    /* VOID  */
    WHILE = 661,                   /* WHILE  */
    BREAK = 662,                   /* BREAK  */
    CONTINUE = 663,                /* CONTINUE  */
    DO = 664,                      /* DO  */
    ELSE = 665,                    /* ELSE  */
    FOR = 666,                     /* FOR  */
    IF = 667,                      /* IF  */
    DISCARD = 668,                 /* DISCARD  */
    RETURN = 669,                  /* RETURN  */
    SWITCH = 670,                  /* SWITCH  */
    CASE = 671,                    /* CASE  */
    DEFAULT = 672,                 /* DEFAULT  */
    TERMINATE_INVOCATION = 673,    /* TERMINATE_INVOCATION  */
    TERMINATE_RAY = 674,           /* TERMINATE_RAY  */
    IGNORE_INTERSECTION = 675,     /* IGNORE_INTERSECTION  */
    UNIFORM = 676,                 /* UNIFORM  */
    SHARED = 677,                  /* SHARED  */
    BUFFER = 678,                  /* BUFFER  */
    TILEIMAGEEXT = 679,            /* TILEIMAGEEXT  */
    FLAT = 680,                    /* FLAT  */
    SMOOTH = 681,                  /* SMOOTH  */
    LAYOUT = 682,                  /* LAYOUT  */
    DOUBLECONSTANT = 683,          /* DOUBLECONSTANT  */
    INT16CONSTANT = 684,           /* INT16CONSTANT  */
    UINT16CONSTANT = 685,          /* UINT16CONSTANT  */
    FLOAT16CONSTANT = 686,         /* FLOAT16CONSTANT  */
    INT32CONSTANT = 687,           /* INT32CONSTANT  */
    UINT32CONSTANT = 688,          /* UINT32CONSTANT  */
    INT64CONSTANT = 689,           /* INT64CONSTANT  */
    UINT64CONSTANT = 690,          /* UINT64CONSTANT  */
    SUBROUTINE = 691,              /* SUBROUTINE  */
    DEMOTE = 692,                  /* DEMOTE  */
    FUNCTION = 693,                /* FUNCTION  */
    PAYLOADNV = 694,               /* PAYLOADNV  */
    PAYLOADINNV = 695,             /* PAYLOADINNV  */
    HITATTRNV = 696,               /* HITATTRNV  */
    CALLDATANV = 697,              /* CALLDATANV  */
    CALLDATAINNV = 698,            /* CALLDATAINNV  */
    PAYLOADEXT = 699,              /* PAYLOADEXT  */
    PAYLOADINEXT = 700,            /* PAYLOADINEXT  */
    HITATTREXT = 701,              /* HITATTREXT  */
    CALLDATAEXT = 702,             /* CALLDATAEXT  */
    CALLDATAINEXT = 703,           /* CALLDATAINEXT  */
    PATCH = 704,                   /* PATCH  */
    SAMPLE = 705,                  /* SAMPLE  */
    NONUNIFORM = 706,              /* NONUNIFORM  */
    COHERENT = 707,                /* COHERENT  */
    VOLATILE = 708,                /* VOLATILE  */
    RESTRICT = 709,                /* RESTRICT  */
    READONLY = 710,                /* READONLY  */
    WRITEONLY = 711,               /* WRITEONLY  */
    NONTEMPORAL = 712,             /* NONTEMPORAL  */
    DEVICECOHERENT = 713,          /* DEVICECOHERENT  */
    QUEUEFAMILYCOHERENT = 714,     /* QUEUEFAMILYCOHERENT  */
    WORKGROUPCOHERENT = 715,       /* WORKGROUPCOHERENT  */
    SUBGROUPCOHERENT = 716,        /* SUBGROUPCOHERENT  */
    NONPRIVATE = 717,              /* NONPRIVATE  */
    SHADERCALLCOHERENT = 718,      /* SHADERCALLCOHERENT  */
    NOPERSPECTIVE = 719,           /* NOPERSPECTIVE  */
    EXPLICITINTERPAMD = 720,       /* EXPLICITINTERPAMD  */
    PERVERTEXEXT = 721,            /* PERVERTEXEXT  */
    PERVERTEXNV = 722,             /* PERVERTEXNV  */
    PERPRIMITIVENV = 723,          /* PERPRIMITIVENV  */
    PERVIEWNV = 724,               /* PERVIEWNV  */
    PERTASKNV = 725,               /* PERTASKNV  */
    PERPRIMITIVEEXT = 726,         /* PERPRIMITIVEEXT  */
    TASKPAYLOADWORKGROUPEXT = 727, /* TASKPAYLOADWORKGROUPEXT  */
    PRECISE = 728                  /* PRECISE  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 72 "MachineIndependent/glslang.y"

    struct {
        glslang::TSourceLoc loc;
        union {
            glslang::TString *string;
            int i;
            unsigned int u;
            long long i64;
            unsigned long long u64;
            bool b;
            double d;
        };
        glslang::TSymbol* symbol;
    } lex;
    struct {
        glslang::TSourceLoc loc;
        glslang::TOperator op;
        union {
            TIntermNode* intermNode;
            glslang::TIntermNodePair nodePair;
            glslang::TIntermTyped* intermTypedNode;
            glslang::TAttributes* attributes;
            glslang::TSpirvRequirement* spirvReq;
            glslang::TSpirvInstruction* spirvInst;
            glslang::TSpirvTypeParameters* spirvTypeParams;
        };
        union {
            glslang::TPublicType type;
            glslang::TFunction* function;
            glslang::TParameter param;
            glslang::TTypeLoc typeLine;
            glslang::TTypeList* typeList;
            glslang::TArraySizes* arraySizes;
            glslang::TIdentifierList* identifierList;
        };
        glslang::TTypeParameters* typeParameters;
    } interm;

#line 576 "MachineIndependent/glslang_tab.cpp.h"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif




int yyparse (glslang::TParseContext* pParseContext);


#endif /* !YY_YY_MACHINEINDEPENDENT_GLSLANG_TAB_CPP_H_INCLUDED  */
