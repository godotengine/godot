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
    FLOAT16_T = 318,               /* FLOAT16_T  */
    FLOAT32_T = 319,               /* FLOAT32_T  */
    DOUBLE = 320,                  /* DOUBLE  */
    FLOAT64_T = 321,               /* FLOAT64_T  */
    INT64_T = 322,                 /* INT64_T  */
    UINT64_T = 323,                /* UINT64_T  */
    INT32_T = 324,                 /* INT32_T  */
    UINT32_T = 325,                /* UINT32_T  */
    INT16_T = 326,                 /* INT16_T  */
    UINT16_T = 327,                /* UINT16_T  */
    INT8_T = 328,                  /* INT8_T  */
    UINT8_T = 329,                 /* UINT8_T  */
    I64VEC2 = 330,                 /* I64VEC2  */
    I64VEC3 = 331,                 /* I64VEC3  */
    I64VEC4 = 332,                 /* I64VEC4  */
    U64VEC2 = 333,                 /* U64VEC2  */
    U64VEC3 = 334,                 /* U64VEC3  */
    U64VEC4 = 335,                 /* U64VEC4  */
    I32VEC2 = 336,                 /* I32VEC2  */
    I32VEC3 = 337,                 /* I32VEC3  */
    I32VEC4 = 338,                 /* I32VEC4  */
    U32VEC2 = 339,                 /* U32VEC2  */
    U32VEC3 = 340,                 /* U32VEC3  */
    U32VEC4 = 341,                 /* U32VEC4  */
    I16VEC2 = 342,                 /* I16VEC2  */
    I16VEC3 = 343,                 /* I16VEC3  */
    I16VEC4 = 344,                 /* I16VEC4  */
    U16VEC2 = 345,                 /* U16VEC2  */
    U16VEC3 = 346,                 /* U16VEC3  */
    U16VEC4 = 347,                 /* U16VEC4  */
    I8VEC2 = 348,                  /* I8VEC2  */
    I8VEC3 = 349,                  /* I8VEC3  */
    I8VEC4 = 350,                  /* I8VEC4  */
    U8VEC2 = 351,                  /* U8VEC2  */
    U8VEC3 = 352,                  /* U8VEC3  */
    U8VEC4 = 353,                  /* U8VEC4  */
    DVEC2 = 354,                   /* DVEC2  */
    DVEC3 = 355,                   /* DVEC3  */
    DVEC4 = 356,                   /* DVEC4  */
    DMAT2 = 357,                   /* DMAT2  */
    DMAT3 = 358,                   /* DMAT3  */
    DMAT4 = 359,                   /* DMAT4  */
    F16VEC2 = 360,                 /* F16VEC2  */
    F16VEC3 = 361,                 /* F16VEC3  */
    F16VEC4 = 362,                 /* F16VEC4  */
    F16MAT2 = 363,                 /* F16MAT2  */
    F16MAT3 = 364,                 /* F16MAT3  */
    F16MAT4 = 365,                 /* F16MAT4  */
    F32VEC2 = 366,                 /* F32VEC2  */
    F32VEC3 = 367,                 /* F32VEC3  */
    F32VEC4 = 368,                 /* F32VEC4  */
    F32MAT2 = 369,                 /* F32MAT2  */
    F32MAT3 = 370,                 /* F32MAT3  */
    F32MAT4 = 371,                 /* F32MAT4  */
    F64VEC2 = 372,                 /* F64VEC2  */
    F64VEC3 = 373,                 /* F64VEC3  */
    F64VEC4 = 374,                 /* F64VEC4  */
    F64MAT2 = 375,                 /* F64MAT2  */
    F64MAT3 = 376,                 /* F64MAT3  */
    F64MAT4 = 377,                 /* F64MAT4  */
    DMAT2X2 = 378,                 /* DMAT2X2  */
    DMAT2X3 = 379,                 /* DMAT2X3  */
    DMAT2X4 = 380,                 /* DMAT2X4  */
    DMAT3X2 = 381,                 /* DMAT3X2  */
    DMAT3X3 = 382,                 /* DMAT3X3  */
    DMAT3X4 = 383,                 /* DMAT3X4  */
    DMAT4X2 = 384,                 /* DMAT4X2  */
    DMAT4X3 = 385,                 /* DMAT4X3  */
    DMAT4X4 = 386,                 /* DMAT4X4  */
    F16MAT2X2 = 387,               /* F16MAT2X2  */
    F16MAT2X3 = 388,               /* F16MAT2X3  */
    F16MAT2X4 = 389,               /* F16MAT2X4  */
    F16MAT3X2 = 390,               /* F16MAT3X2  */
    F16MAT3X3 = 391,               /* F16MAT3X3  */
    F16MAT3X4 = 392,               /* F16MAT3X4  */
    F16MAT4X2 = 393,               /* F16MAT4X2  */
    F16MAT4X3 = 394,               /* F16MAT4X3  */
    F16MAT4X4 = 395,               /* F16MAT4X4  */
    F32MAT2X2 = 396,               /* F32MAT2X2  */
    F32MAT2X3 = 397,               /* F32MAT2X3  */
    F32MAT2X4 = 398,               /* F32MAT2X4  */
    F32MAT3X2 = 399,               /* F32MAT3X2  */
    F32MAT3X3 = 400,               /* F32MAT3X3  */
    F32MAT3X4 = 401,               /* F32MAT3X4  */
    F32MAT4X2 = 402,               /* F32MAT4X2  */
    F32MAT4X3 = 403,               /* F32MAT4X3  */
    F32MAT4X4 = 404,               /* F32MAT4X4  */
    F64MAT2X2 = 405,               /* F64MAT2X2  */
    F64MAT2X3 = 406,               /* F64MAT2X3  */
    F64MAT2X4 = 407,               /* F64MAT2X4  */
    F64MAT3X2 = 408,               /* F64MAT3X2  */
    F64MAT3X3 = 409,               /* F64MAT3X3  */
    F64MAT3X4 = 410,               /* F64MAT3X4  */
    F64MAT4X2 = 411,               /* F64MAT4X2  */
    F64MAT4X3 = 412,               /* F64MAT4X3  */
    F64MAT4X4 = 413,               /* F64MAT4X4  */
    ATOMIC_UINT = 414,             /* ATOMIC_UINT  */
    ACCSTRUCTNV = 415,             /* ACCSTRUCTNV  */
    ACCSTRUCTEXT = 416,            /* ACCSTRUCTEXT  */
    RAYQUERYEXT = 417,             /* RAYQUERYEXT  */
    FCOOPMATNV = 418,              /* FCOOPMATNV  */
    ICOOPMATNV = 419,              /* ICOOPMATNV  */
    UCOOPMATNV = 420,              /* UCOOPMATNV  */
    COOPMAT = 421,                 /* COOPMAT  */
    HITOBJECTNV = 422,             /* HITOBJECTNV  */
    HITOBJECTATTRNV = 423,         /* HITOBJECTATTRNV  */
    SAMPLERCUBEARRAY = 424,        /* SAMPLERCUBEARRAY  */
    SAMPLERCUBEARRAYSHADOW = 425,  /* SAMPLERCUBEARRAYSHADOW  */
    ISAMPLERCUBEARRAY = 426,       /* ISAMPLERCUBEARRAY  */
    USAMPLERCUBEARRAY = 427,       /* USAMPLERCUBEARRAY  */
    SAMPLER1D = 428,               /* SAMPLER1D  */
    SAMPLER1DARRAY = 429,          /* SAMPLER1DARRAY  */
    SAMPLER1DARRAYSHADOW = 430,    /* SAMPLER1DARRAYSHADOW  */
    ISAMPLER1D = 431,              /* ISAMPLER1D  */
    SAMPLER1DSHADOW = 432,         /* SAMPLER1DSHADOW  */
    SAMPLER2DRECT = 433,           /* SAMPLER2DRECT  */
    SAMPLER2DRECTSHADOW = 434,     /* SAMPLER2DRECTSHADOW  */
    ISAMPLER2DRECT = 435,          /* ISAMPLER2DRECT  */
    USAMPLER2DRECT = 436,          /* USAMPLER2DRECT  */
    SAMPLERBUFFER = 437,           /* SAMPLERBUFFER  */
    ISAMPLERBUFFER = 438,          /* ISAMPLERBUFFER  */
    USAMPLERBUFFER = 439,          /* USAMPLERBUFFER  */
    SAMPLER2DMS = 440,             /* SAMPLER2DMS  */
    ISAMPLER2DMS = 441,            /* ISAMPLER2DMS  */
    USAMPLER2DMS = 442,            /* USAMPLER2DMS  */
    SAMPLER2DMSARRAY = 443,        /* SAMPLER2DMSARRAY  */
    ISAMPLER2DMSARRAY = 444,       /* ISAMPLER2DMSARRAY  */
    USAMPLER2DMSARRAY = 445,       /* USAMPLER2DMSARRAY  */
    SAMPLEREXTERNALOES = 446,      /* SAMPLEREXTERNALOES  */
    SAMPLEREXTERNAL2DY2YEXT = 447, /* SAMPLEREXTERNAL2DY2YEXT  */
    ISAMPLER1DARRAY = 448,         /* ISAMPLER1DARRAY  */
    USAMPLER1D = 449,              /* USAMPLER1D  */
    USAMPLER1DARRAY = 450,         /* USAMPLER1DARRAY  */
    F16SAMPLER1D = 451,            /* F16SAMPLER1D  */
    F16SAMPLER2D = 452,            /* F16SAMPLER2D  */
    F16SAMPLER3D = 453,            /* F16SAMPLER3D  */
    F16SAMPLER2DRECT = 454,        /* F16SAMPLER2DRECT  */
    F16SAMPLERCUBE = 455,          /* F16SAMPLERCUBE  */
    F16SAMPLER1DARRAY = 456,       /* F16SAMPLER1DARRAY  */
    F16SAMPLER2DARRAY = 457,       /* F16SAMPLER2DARRAY  */
    F16SAMPLERCUBEARRAY = 458,     /* F16SAMPLERCUBEARRAY  */
    F16SAMPLERBUFFER = 459,        /* F16SAMPLERBUFFER  */
    F16SAMPLER2DMS = 460,          /* F16SAMPLER2DMS  */
    F16SAMPLER2DMSARRAY = 461,     /* F16SAMPLER2DMSARRAY  */
    F16SAMPLER1DSHADOW = 462,      /* F16SAMPLER1DSHADOW  */
    F16SAMPLER2DSHADOW = 463,      /* F16SAMPLER2DSHADOW  */
    F16SAMPLER1DARRAYSHADOW = 464, /* F16SAMPLER1DARRAYSHADOW  */
    F16SAMPLER2DARRAYSHADOW = 465, /* F16SAMPLER2DARRAYSHADOW  */
    F16SAMPLER2DRECTSHADOW = 466,  /* F16SAMPLER2DRECTSHADOW  */
    F16SAMPLERCUBESHADOW = 467,    /* F16SAMPLERCUBESHADOW  */
    F16SAMPLERCUBEARRAYSHADOW = 468, /* F16SAMPLERCUBEARRAYSHADOW  */
    IMAGE1D = 469,                 /* IMAGE1D  */
    IIMAGE1D = 470,                /* IIMAGE1D  */
    UIMAGE1D = 471,                /* UIMAGE1D  */
    IMAGE2D = 472,                 /* IMAGE2D  */
    IIMAGE2D = 473,                /* IIMAGE2D  */
    UIMAGE2D = 474,                /* UIMAGE2D  */
    IMAGE3D = 475,                 /* IMAGE3D  */
    IIMAGE3D = 476,                /* IIMAGE3D  */
    UIMAGE3D = 477,                /* UIMAGE3D  */
    IMAGE2DRECT = 478,             /* IMAGE2DRECT  */
    IIMAGE2DRECT = 479,            /* IIMAGE2DRECT  */
    UIMAGE2DRECT = 480,            /* UIMAGE2DRECT  */
    IMAGECUBE = 481,               /* IMAGECUBE  */
    IIMAGECUBE = 482,              /* IIMAGECUBE  */
    UIMAGECUBE = 483,              /* UIMAGECUBE  */
    IMAGEBUFFER = 484,             /* IMAGEBUFFER  */
    IIMAGEBUFFER = 485,            /* IIMAGEBUFFER  */
    UIMAGEBUFFER = 486,            /* UIMAGEBUFFER  */
    IMAGE1DARRAY = 487,            /* IMAGE1DARRAY  */
    IIMAGE1DARRAY = 488,           /* IIMAGE1DARRAY  */
    UIMAGE1DARRAY = 489,           /* UIMAGE1DARRAY  */
    IMAGE2DARRAY = 490,            /* IMAGE2DARRAY  */
    IIMAGE2DARRAY = 491,           /* IIMAGE2DARRAY  */
    UIMAGE2DARRAY = 492,           /* UIMAGE2DARRAY  */
    IMAGECUBEARRAY = 493,          /* IMAGECUBEARRAY  */
    IIMAGECUBEARRAY = 494,         /* IIMAGECUBEARRAY  */
    UIMAGECUBEARRAY = 495,         /* UIMAGECUBEARRAY  */
    IMAGE2DMS = 496,               /* IMAGE2DMS  */
    IIMAGE2DMS = 497,              /* IIMAGE2DMS  */
    UIMAGE2DMS = 498,              /* UIMAGE2DMS  */
    IMAGE2DMSARRAY = 499,          /* IMAGE2DMSARRAY  */
    IIMAGE2DMSARRAY = 500,         /* IIMAGE2DMSARRAY  */
    UIMAGE2DMSARRAY = 501,         /* UIMAGE2DMSARRAY  */
    F16IMAGE1D = 502,              /* F16IMAGE1D  */
    F16IMAGE2D = 503,              /* F16IMAGE2D  */
    F16IMAGE3D = 504,              /* F16IMAGE3D  */
    F16IMAGE2DRECT = 505,          /* F16IMAGE2DRECT  */
    F16IMAGECUBE = 506,            /* F16IMAGECUBE  */
    F16IMAGE1DARRAY = 507,         /* F16IMAGE1DARRAY  */
    F16IMAGE2DARRAY = 508,         /* F16IMAGE2DARRAY  */
    F16IMAGECUBEARRAY = 509,       /* F16IMAGECUBEARRAY  */
    F16IMAGEBUFFER = 510,          /* F16IMAGEBUFFER  */
    F16IMAGE2DMS = 511,            /* F16IMAGE2DMS  */
    F16IMAGE2DMSARRAY = 512,       /* F16IMAGE2DMSARRAY  */
    I64IMAGE1D = 513,              /* I64IMAGE1D  */
    U64IMAGE1D = 514,              /* U64IMAGE1D  */
    I64IMAGE2D = 515,              /* I64IMAGE2D  */
    U64IMAGE2D = 516,              /* U64IMAGE2D  */
    I64IMAGE3D = 517,              /* I64IMAGE3D  */
    U64IMAGE3D = 518,              /* U64IMAGE3D  */
    I64IMAGE2DRECT = 519,          /* I64IMAGE2DRECT  */
    U64IMAGE2DRECT = 520,          /* U64IMAGE2DRECT  */
    I64IMAGECUBE = 521,            /* I64IMAGECUBE  */
    U64IMAGECUBE = 522,            /* U64IMAGECUBE  */
    I64IMAGEBUFFER = 523,          /* I64IMAGEBUFFER  */
    U64IMAGEBUFFER = 524,          /* U64IMAGEBUFFER  */
    I64IMAGE1DARRAY = 525,         /* I64IMAGE1DARRAY  */
    U64IMAGE1DARRAY = 526,         /* U64IMAGE1DARRAY  */
    I64IMAGE2DARRAY = 527,         /* I64IMAGE2DARRAY  */
    U64IMAGE2DARRAY = 528,         /* U64IMAGE2DARRAY  */
    I64IMAGECUBEARRAY = 529,       /* I64IMAGECUBEARRAY  */
    U64IMAGECUBEARRAY = 530,       /* U64IMAGECUBEARRAY  */
    I64IMAGE2DMS = 531,            /* I64IMAGE2DMS  */
    U64IMAGE2DMS = 532,            /* U64IMAGE2DMS  */
    I64IMAGE2DMSARRAY = 533,       /* I64IMAGE2DMSARRAY  */
    U64IMAGE2DMSARRAY = 534,       /* U64IMAGE2DMSARRAY  */
    TEXTURECUBEARRAY = 535,        /* TEXTURECUBEARRAY  */
    ITEXTURECUBEARRAY = 536,       /* ITEXTURECUBEARRAY  */
    UTEXTURECUBEARRAY = 537,       /* UTEXTURECUBEARRAY  */
    TEXTURE1D = 538,               /* TEXTURE1D  */
    ITEXTURE1D = 539,              /* ITEXTURE1D  */
    UTEXTURE1D = 540,              /* UTEXTURE1D  */
    TEXTURE1DARRAY = 541,          /* TEXTURE1DARRAY  */
    ITEXTURE1DARRAY = 542,         /* ITEXTURE1DARRAY  */
    UTEXTURE1DARRAY = 543,         /* UTEXTURE1DARRAY  */
    TEXTURE2DRECT = 544,           /* TEXTURE2DRECT  */
    ITEXTURE2DRECT = 545,          /* ITEXTURE2DRECT  */
    UTEXTURE2DRECT = 546,          /* UTEXTURE2DRECT  */
    TEXTUREBUFFER = 547,           /* TEXTUREBUFFER  */
    ITEXTUREBUFFER = 548,          /* ITEXTUREBUFFER  */
    UTEXTUREBUFFER = 549,          /* UTEXTUREBUFFER  */
    TEXTURE2DMS = 550,             /* TEXTURE2DMS  */
    ITEXTURE2DMS = 551,            /* ITEXTURE2DMS  */
    UTEXTURE2DMS = 552,            /* UTEXTURE2DMS  */
    TEXTURE2DMSARRAY = 553,        /* TEXTURE2DMSARRAY  */
    ITEXTURE2DMSARRAY = 554,       /* ITEXTURE2DMSARRAY  */
    UTEXTURE2DMSARRAY = 555,       /* UTEXTURE2DMSARRAY  */
    F16TEXTURE1D = 556,            /* F16TEXTURE1D  */
    F16TEXTURE2D = 557,            /* F16TEXTURE2D  */
    F16TEXTURE3D = 558,            /* F16TEXTURE3D  */
    F16TEXTURE2DRECT = 559,        /* F16TEXTURE2DRECT  */
    F16TEXTURECUBE = 560,          /* F16TEXTURECUBE  */
    F16TEXTURE1DARRAY = 561,       /* F16TEXTURE1DARRAY  */
    F16TEXTURE2DARRAY = 562,       /* F16TEXTURE2DARRAY  */
    F16TEXTURECUBEARRAY = 563,     /* F16TEXTURECUBEARRAY  */
    F16TEXTUREBUFFER = 564,        /* F16TEXTUREBUFFER  */
    F16TEXTURE2DMS = 565,          /* F16TEXTURE2DMS  */
    F16TEXTURE2DMSARRAY = 566,     /* F16TEXTURE2DMSARRAY  */
    SUBPASSINPUT = 567,            /* SUBPASSINPUT  */
    SUBPASSINPUTMS = 568,          /* SUBPASSINPUTMS  */
    ISUBPASSINPUT = 569,           /* ISUBPASSINPUT  */
    ISUBPASSINPUTMS = 570,         /* ISUBPASSINPUTMS  */
    USUBPASSINPUT = 571,           /* USUBPASSINPUT  */
    USUBPASSINPUTMS = 572,         /* USUBPASSINPUTMS  */
    F16SUBPASSINPUT = 573,         /* F16SUBPASSINPUT  */
    F16SUBPASSINPUTMS = 574,       /* F16SUBPASSINPUTMS  */
    SPIRV_INSTRUCTION = 575,       /* SPIRV_INSTRUCTION  */
    SPIRV_EXECUTION_MODE = 576,    /* SPIRV_EXECUTION_MODE  */
    SPIRV_EXECUTION_MODE_ID = 577, /* SPIRV_EXECUTION_MODE_ID  */
    SPIRV_DECORATE = 578,          /* SPIRV_DECORATE  */
    SPIRV_DECORATE_ID = 579,       /* SPIRV_DECORATE_ID  */
    SPIRV_DECORATE_STRING = 580,   /* SPIRV_DECORATE_STRING  */
    SPIRV_TYPE = 581,              /* SPIRV_TYPE  */
    SPIRV_STORAGE_CLASS = 582,     /* SPIRV_STORAGE_CLASS  */
    SPIRV_BY_REFERENCE = 583,      /* SPIRV_BY_REFERENCE  */
    SPIRV_LITERAL = 584,           /* SPIRV_LITERAL  */
    ATTACHMENTEXT = 585,           /* ATTACHMENTEXT  */
    IATTACHMENTEXT = 586,          /* IATTACHMENTEXT  */
    UATTACHMENTEXT = 587,          /* UATTACHMENTEXT  */
    LEFT_OP = 588,                 /* LEFT_OP  */
    RIGHT_OP = 589,                /* RIGHT_OP  */
    INC_OP = 590,                  /* INC_OP  */
    DEC_OP = 591,                  /* DEC_OP  */
    LE_OP = 592,                   /* LE_OP  */
    GE_OP = 593,                   /* GE_OP  */
    EQ_OP = 594,                   /* EQ_OP  */
    NE_OP = 595,                   /* NE_OP  */
    AND_OP = 596,                  /* AND_OP  */
    OR_OP = 597,                   /* OR_OP  */
    XOR_OP = 598,                  /* XOR_OP  */
    MUL_ASSIGN = 599,              /* MUL_ASSIGN  */
    DIV_ASSIGN = 600,              /* DIV_ASSIGN  */
    ADD_ASSIGN = 601,              /* ADD_ASSIGN  */
    MOD_ASSIGN = 602,              /* MOD_ASSIGN  */
    LEFT_ASSIGN = 603,             /* LEFT_ASSIGN  */
    RIGHT_ASSIGN = 604,            /* RIGHT_ASSIGN  */
    AND_ASSIGN = 605,              /* AND_ASSIGN  */
    XOR_ASSIGN = 606,              /* XOR_ASSIGN  */
    OR_ASSIGN = 607,               /* OR_ASSIGN  */
    SUB_ASSIGN = 608,              /* SUB_ASSIGN  */
    STRING_LITERAL = 609,          /* STRING_LITERAL  */
    LEFT_PAREN = 610,              /* LEFT_PAREN  */
    RIGHT_PAREN = 611,             /* RIGHT_PAREN  */
    LEFT_BRACKET = 612,            /* LEFT_BRACKET  */
    RIGHT_BRACKET = 613,           /* RIGHT_BRACKET  */
    LEFT_BRACE = 614,              /* LEFT_BRACE  */
    RIGHT_BRACE = 615,             /* RIGHT_BRACE  */
    DOT = 616,                     /* DOT  */
    COMMA = 617,                   /* COMMA  */
    COLON = 618,                   /* COLON  */
    EQUAL = 619,                   /* EQUAL  */
    SEMICOLON = 620,               /* SEMICOLON  */
    BANG = 621,                    /* BANG  */
    DASH = 622,                    /* DASH  */
    TILDE = 623,                   /* TILDE  */
    PLUS = 624,                    /* PLUS  */
    STAR = 625,                    /* STAR  */
    SLASH = 626,                   /* SLASH  */
    PERCENT = 627,                 /* PERCENT  */
    LEFT_ANGLE = 628,              /* LEFT_ANGLE  */
    RIGHT_ANGLE = 629,             /* RIGHT_ANGLE  */
    VERTICAL_BAR = 630,            /* VERTICAL_BAR  */
    CARET = 631,                   /* CARET  */
    AMPERSAND = 632,               /* AMPERSAND  */
    QUESTION = 633,                /* QUESTION  */
    INVARIANT = 634,               /* INVARIANT  */
    HIGH_PRECISION = 635,          /* HIGH_PRECISION  */
    MEDIUM_PRECISION = 636,        /* MEDIUM_PRECISION  */
    LOW_PRECISION = 637,           /* LOW_PRECISION  */
    PRECISION = 638,               /* PRECISION  */
    PACKED = 639,                  /* PACKED  */
    RESOURCE = 640,                /* RESOURCE  */
    SUPERP = 641,                  /* SUPERP  */
    FLOATCONSTANT = 642,           /* FLOATCONSTANT  */
    INTCONSTANT = 643,             /* INTCONSTANT  */
    UINTCONSTANT = 644,            /* UINTCONSTANT  */
    BOOLCONSTANT = 645,            /* BOOLCONSTANT  */
    IDENTIFIER = 646,              /* IDENTIFIER  */
    TYPE_NAME = 647,               /* TYPE_NAME  */
    CENTROID = 648,                /* CENTROID  */
    IN = 649,                      /* IN  */
    OUT = 650,                     /* OUT  */
    INOUT = 651,                   /* INOUT  */
    STRUCT = 652,                  /* STRUCT  */
    VOID = 653,                    /* VOID  */
    WHILE = 654,                   /* WHILE  */
    BREAK = 655,                   /* BREAK  */
    CONTINUE = 656,                /* CONTINUE  */
    DO = 657,                      /* DO  */
    ELSE = 658,                    /* ELSE  */
    FOR = 659,                     /* FOR  */
    IF = 660,                      /* IF  */
    DISCARD = 661,                 /* DISCARD  */
    RETURN = 662,                  /* RETURN  */
    SWITCH = 663,                  /* SWITCH  */
    CASE = 664,                    /* CASE  */
    DEFAULT = 665,                 /* DEFAULT  */
    TERMINATE_INVOCATION = 666,    /* TERMINATE_INVOCATION  */
    TERMINATE_RAY = 667,           /* TERMINATE_RAY  */
    IGNORE_INTERSECTION = 668,     /* IGNORE_INTERSECTION  */
    UNIFORM = 669,                 /* UNIFORM  */
    SHARED = 670,                  /* SHARED  */
    BUFFER = 671,                  /* BUFFER  */
    TILEIMAGEEXT = 672,            /* TILEIMAGEEXT  */
    FLAT = 673,                    /* FLAT  */
    SMOOTH = 674,                  /* SMOOTH  */
    LAYOUT = 675,                  /* LAYOUT  */
    DOUBLECONSTANT = 676,          /* DOUBLECONSTANT  */
    INT16CONSTANT = 677,           /* INT16CONSTANT  */
    UINT16CONSTANT = 678,          /* UINT16CONSTANT  */
    FLOAT16CONSTANT = 679,         /* FLOAT16CONSTANT  */
    INT32CONSTANT = 680,           /* INT32CONSTANT  */
    UINT32CONSTANT = 681,          /* UINT32CONSTANT  */
    INT64CONSTANT = 682,           /* INT64CONSTANT  */
    UINT64CONSTANT = 683,          /* UINT64CONSTANT  */
    SUBROUTINE = 684,              /* SUBROUTINE  */
    DEMOTE = 685,                  /* DEMOTE  */
    PAYLOADNV = 686,               /* PAYLOADNV  */
    PAYLOADINNV = 687,             /* PAYLOADINNV  */
    HITATTRNV = 688,               /* HITATTRNV  */
    CALLDATANV = 689,              /* CALLDATANV  */
    CALLDATAINNV = 690,            /* CALLDATAINNV  */
    PAYLOADEXT = 691,              /* PAYLOADEXT  */
    PAYLOADINEXT = 692,            /* PAYLOADINEXT  */
    HITATTREXT = 693,              /* HITATTREXT  */
    CALLDATAEXT = 694,             /* CALLDATAEXT  */
    CALLDATAINEXT = 695,           /* CALLDATAINEXT  */
    PATCH = 696,                   /* PATCH  */
    SAMPLE = 697,                  /* SAMPLE  */
    NONUNIFORM = 698,              /* NONUNIFORM  */
    COHERENT = 699,                /* COHERENT  */
    VOLATILE = 700,                /* VOLATILE  */
    RESTRICT = 701,                /* RESTRICT  */
    READONLY = 702,                /* READONLY  */
    WRITEONLY = 703,               /* WRITEONLY  */
    DEVICECOHERENT = 704,          /* DEVICECOHERENT  */
    QUEUEFAMILYCOHERENT = 705,     /* QUEUEFAMILYCOHERENT  */
    WORKGROUPCOHERENT = 706,       /* WORKGROUPCOHERENT  */
    SUBGROUPCOHERENT = 707,        /* SUBGROUPCOHERENT  */
    NONPRIVATE = 708,              /* NONPRIVATE  */
    SHADERCALLCOHERENT = 709,      /* SHADERCALLCOHERENT  */
    NOPERSPECTIVE = 710,           /* NOPERSPECTIVE  */
    EXPLICITINTERPAMD = 711,       /* EXPLICITINTERPAMD  */
    PERVERTEXEXT = 712,            /* PERVERTEXEXT  */
    PERVERTEXNV = 713,             /* PERVERTEXNV  */
    PERPRIMITIVENV = 714,          /* PERPRIMITIVENV  */
    PERVIEWNV = 715,               /* PERVIEWNV  */
    PERTASKNV = 716,               /* PERTASKNV  */
    PERPRIMITIVEEXT = 717,         /* PERPRIMITIVEEXT  */
    TASKPAYLOADWORKGROUPEXT = 718, /* TASKPAYLOADWORKGROUPEXT  */
    PRECISE = 719                  /* PRECISE  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 97 "MachineIndependent/glslang.y"

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

#line 567 "MachineIndependent/glslang_tab.cpp.h"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif




int yyparse (glslang::TParseContext* pParseContext);


#endif /* !YY_YY_MACHINEINDEPENDENT_GLSLANG_TAB_CPP_H_INCLUDED  */
