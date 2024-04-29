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
    HITOBJECTNV = 421,             /* HITOBJECTNV  */
    HITOBJECTATTRNV = 422,         /* HITOBJECTATTRNV  */
    SAMPLERCUBEARRAY = 423,        /* SAMPLERCUBEARRAY  */
    SAMPLERCUBEARRAYSHADOW = 424,  /* SAMPLERCUBEARRAYSHADOW  */
    ISAMPLERCUBEARRAY = 425,       /* ISAMPLERCUBEARRAY  */
    USAMPLERCUBEARRAY = 426,       /* USAMPLERCUBEARRAY  */
    SAMPLER1D = 427,               /* SAMPLER1D  */
    SAMPLER1DARRAY = 428,          /* SAMPLER1DARRAY  */
    SAMPLER1DARRAYSHADOW = 429,    /* SAMPLER1DARRAYSHADOW  */
    ISAMPLER1D = 430,              /* ISAMPLER1D  */
    SAMPLER1DSHADOW = 431,         /* SAMPLER1DSHADOW  */
    SAMPLER2DRECT = 432,           /* SAMPLER2DRECT  */
    SAMPLER2DRECTSHADOW = 433,     /* SAMPLER2DRECTSHADOW  */
    ISAMPLER2DRECT = 434,          /* ISAMPLER2DRECT  */
    USAMPLER2DRECT = 435,          /* USAMPLER2DRECT  */
    SAMPLERBUFFER = 436,           /* SAMPLERBUFFER  */
    ISAMPLERBUFFER = 437,          /* ISAMPLERBUFFER  */
    USAMPLERBUFFER = 438,          /* USAMPLERBUFFER  */
    SAMPLER2DMS = 439,             /* SAMPLER2DMS  */
    ISAMPLER2DMS = 440,            /* ISAMPLER2DMS  */
    USAMPLER2DMS = 441,            /* USAMPLER2DMS  */
    SAMPLER2DMSARRAY = 442,        /* SAMPLER2DMSARRAY  */
    ISAMPLER2DMSARRAY = 443,       /* ISAMPLER2DMSARRAY  */
    USAMPLER2DMSARRAY = 444,       /* USAMPLER2DMSARRAY  */
    SAMPLEREXTERNALOES = 445,      /* SAMPLEREXTERNALOES  */
    SAMPLEREXTERNAL2DY2YEXT = 446, /* SAMPLEREXTERNAL2DY2YEXT  */
    ISAMPLER1DARRAY = 447,         /* ISAMPLER1DARRAY  */
    USAMPLER1D = 448,              /* USAMPLER1D  */
    USAMPLER1DARRAY = 449,         /* USAMPLER1DARRAY  */
    F16SAMPLER1D = 450,            /* F16SAMPLER1D  */
    F16SAMPLER2D = 451,            /* F16SAMPLER2D  */
    F16SAMPLER3D = 452,            /* F16SAMPLER3D  */
    F16SAMPLER2DRECT = 453,        /* F16SAMPLER2DRECT  */
    F16SAMPLERCUBE = 454,          /* F16SAMPLERCUBE  */
    F16SAMPLER1DARRAY = 455,       /* F16SAMPLER1DARRAY  */
    F16SAMPLER2DARRAY = 456,       /* F16SAMPLER2DARRAY  */
    F16SAMPLERCUBEARRAY = 457,     /* F16SAMPLERCUBEARRAY  */
    F16SAMPLERBUFFER = 458,        /* F16SAMPLERBUFFER  */
    F16SAMPLER2DMS = 459,          /* F16SAMPLER2DMS  */
    F16SAMPLER2DMSARRAY = 460,     /* F16SAMPLER2DMSARRAY  */
    F16SAMPLER1DSHADOW = 461,      /* F16SAMPLER1DSHADOW  */
    F16SAMPLER2DSHADOW = 462,      /* F16SAMPLER2DSHADOW  */
    F16SAMPLER1DARRAYSHADOW = 463, /* F16SAMPLER1DARRAYSHADOW  */
    F16SAMPLER2DARRAYSHADOW = 464, /* F16SAMPLER2DARRAYSHADOW  */
    F16SAMPLER2DRECTSHADOW = 465,  /* F16SAMPLER2DRECTSHADOW  */
    F16SAMPLERCUBESHADOW = 466,    /* F16SAMPLERCUBESHADOW  */
    F16SAMPLERCUBEARRAYSHADOW = 467, /* F16SAMPLERCUBEARRAYSHADOW  */
    IMAGE1D = 468,                 /* IMAGE1D  */
    IIMAGE1D = 469,                /* IIMAGE1D  */
    UIMAGE1D = 470,                /* UIMAGE1D  */
    IMAGE2D = 471,                 /* IMAGE2D  */
    IIMAGE2D = 472,                /* IIMAGE2D  */
    UIMAGE2D = 473,                /* UIMAGE2D  */
    IMAGE3D = 474,                 /* IMAGE3D  */
    IIMAGE3D = 475,                /* IIMAGE3D  */
    UIMAGE3D = 476,                /* UIMAGE3D  */
    IMAGE2DRECT = 477,             /* IMAGE2DRECT  */
    IIMAGE2DRECT = 478,            /* IIMAGE2DRECT  */
    UIMAGE2DRECT = 479,            /* UIMAGE2DRECT  */
    IMAGECUBE = 480,               /* IMAGECUBE  */
    IIMAGECUBE = 481,              /* IIMAGECUBE  */
    UIMAGECUBE = 482,              /* UIMAGECUBE  */
    IMAGEBUFFER = 483,             /* IMAGEBUFFER  */
    IIMAGEBUFFER = 484,            /* IIMAGEBUFFER  */
    UIMAGEBUFFER = 485,            /* UIMAGEBUFFER  */
    IMAGE1DARRAY = 486,            /* IMAGE1DARRAY  */
    IIMAGE1DARRAY = 487,           /* IIMAGE1DARRAY  */
    UIMAGE1DARRAY = 488,           /* UIMAGE1DARRAY  */
    IMAGE2DARRAY = 489,            /* IMAGE2DARRAY  */
    IIMAGE2DARRAY = 490,           /* IIMAGE2DARRAY  */
    UIMAGE2DARRAY = 491,           /* UIMAGE2DARRAY  */
    IMAGECUBEARRAY = 492,          /* IMAGECUBEARRAY  */
    IIMAGECUBEARRAY = 493,         /* IIMAGECUBEARRAY  */
    UIMAGECUBEARRAY = 494,         /* UIMAGECUBEARRAY  */
    IMAGE2DMS = 495,               /* IMAGE2DMS  */
    IIMAGE2DMS = 496,              /* IIMAGE2DMS  */
    UIMAGE2DMS = 497,              /* UIMAGE2DMS  */
    IMAGE2DMSARRAY = 498,          /* IMAGE2DMSARRAY  */
    IIMAGE2DMSARRAY = 499,         /* IIMAGE2DMSARRAY  */
    UIMAGE2DMSARRAY = 500,         /* UIMAGE2DMSARRAY  */
    F16IMAGE1D = 501,              /* F16IMAGE1D  */
    F16IMAGE2D = 502,              /* F16IMAGE2D  */
    F16IMAGE3D = 503,              /* F16IMAGE3D  */
    F16IMAGE2DRECT = 504,          /* F16IMAGE2DRECT  */
    F16IMAGECUBE = 505,            /* F16IMAGECUBE  */
    F16IMAGE1DARRAY = 506,         /* F16IMAGE1DARRAY  */
    F16IMAGE2DARRAY = 507,         /* F16IMAGE2DARRAY  */
    F16IMAGECUBEARRAY = 508,       /* F16IMAGECUBEARRAY  */
    F16IMAGEBUFFER = 509,          /* F16IMAGEBUFFER  */
    F16IMAGE2DMS = 510,            /* F16IMAGE2DMS  */
    F16IMAGE2DMSARRAY = 511,       /* F16IMAGE2DMSARRAY  */
    I64IMAGE1D = 512,              /* I64IMAGE1D  */
    U64IMAGE1D = 513,              /* U64IMAGE1D  */
    I64IMAGE2D = 514,              /* I64IMAGE2D  */
    U64IMAGE2D = 515,              /* U64IMAGE2D  */
    I64IMAGE3D = 516,              /* I64IMAGE3D  */
    U64IMAGE3D = 517,              /* U64IMAGE3D  */
    I64IMAGE2DRECT = 518,          /* I64IMAGE2DRECT  */
    U64IMAGE2DRECT = 519,          /* U64IMAGE2DRECT  */
    I64IMAGECUBE = 520,            /* I64IMAGECUBE  */
    U64IMAGECUBE = 521,            /* U64IMAGECUBE  */
    I64IMAGEBUFFER = 522,          /* I64IMAGEBUFFER  */
    U64IMAGEBUFFER = 523,          /* U64IMAGEBUFFER  */
    I64IMAGE1DARRAY = 524,         /* I64IMAGE1DARRAY  */
    U64IMAGE1DARRAY = 525,         /* U64IMAGE1DARRAY  */
    I64IMAGE2DARRAY = 526,         /* I64IMAGE2DARRAY  */
    U64IMAGE2DARRAY = 527,         /* U64IMAGE2DARRAY  */
    I64IMAGECUBEARRAY = 528,       /* I64IMAGECUBEARRAY  */
    U64IMAGECUBEARRAY = 529,       /* U64IMAGECUBEARRAY  */
    I64IMAGE2DMS = 530,            /* I64IMAGE2DMS  */
    U64IMAGE2DMS = 531,            /* U64IMAGE2DMS  */
    I64IMAGE2DMSARRAY = 532,       /* I64IMAGE2DMSARRAY  */
    U64IMAGE2DMSARRAY = 533,       /* U64IMAGE2DMSARRAY  */
    TEXTURECUBEARRAY = 534,        /* TEXTURECUBEARRAY  */
    ITEXTURECUBEARRAY = 535,       /* ITEXTURECUBEARRAY  */
    UTEXTURECUBEARRAY = 536,       /* UTEXTURECUBEARRAY  */
    TEXTURE1D = 537,               /* TEXTURE1D  */
    ITEXTURE1D = 538,              /* ITEXTURE1D  */
    UTEXTURE1D = 539,              /* UTEXTURE1D  */
    TEXTURE1DARRAY = 540,          /* TEXTURE1DARRAY  */
    ITEXTURE1DARRAY = 541,         /* ITEXTURE1DARRAY  */
    UTEXTURE1DARRAY = 542,         /* UTEXTURE1DARRAY  */
    TEXTURE2DRECT = 543,           /* TEXTURE2DRECT  */
    ITEXTURE2DRECT = 544,          /* ITEXTURE2DRECT  */
    UTEXTURE2DRECT = 545,          /* UTEXTURE2DRECT  */
    TEXTUREBUFFER = 546,           /* TEXTUREBUFFER  */
    ITEXTUREBUFFER = 547,          /* ITEXTUREBUFFER  */
    UTEXTUREBUFFER = 548,          /* UTEXTUREBUFFER  */
    TEXTURE2DMS = 549,             /* TEXTURE2DMS  */
    ITEXTURE2DMS = 550,            /* ITEXTURE2DMS  */
    UTEXTURE2DMS = 551,            /* UTEXTURE2DMS  */
    TEXTURE2DMSARRAY = 552,        /* TEXTURE2DMSARRAY  */
    ITEXTURE2DMSARRAY = 553,       /* ITEXTURE2DMSARRAY  */
    UTEXTURE2DMSARRAY = 554,       /* UTEXTURE2DMSARRAY  */
    F16TEXTURE1D = 555,            /* F16TEXTURE1D  */
    F16TEXTURE2D = 556,            /* F16TEXTURE2D  */
    F16TEXTURE3D = 557,            /* F16TEXTURE3D  */
    F16TEXTURE2DRECT = 558,        /* F16TEXTURE2DRECT  */
    F16TEXTURECUBE = 559,          /* F16TEXTURECUBE  */
    F16TEXTURE1DARRAY = 560,       /* F16TEXTURE1DARRAY  */
    F16TEXTURE2DARRAY = 561,       /* F16TEXTURE2DARRAY  */
    F16TEXTURECUBEARRAY = 562,     /* F16TEXTURECUBEARRAY  */
    F16TEXTUREBUFFER = 563,        /* F16TEXTUREBUFFER  */
    F16TEXTURE2DMS = 564,          /* F16TEXTURE2DMS  */
    F16TEXTURE2DMSARRAY = 565,     /* F16TEXTURE2DMSARRAY  */
    SUBPASSINPUT = 566,            /* SUBPASSINPUT  */
    SUBPASSINPUTMS = 567,          /* SUBPASSINPUTMS  */
    ISUBPASSINPUT = 568,           /* ISUBPASSINPUT  */
    ISUBPASSINPUTMS = 569,         /* ISUBPASSINPUTMS  */
    USUBPASSINPUT = 570,           /* USUBPASSINPUT  */
    USUBPASSINPUTMS = 571,         /* USUBPASSINPUTMS  */
    F16SUBPASSINPUT = 572,         /* F16SUBPASSINPUT  */
    F16SUBPASSINPUTMS = 573,       /* F16SUBPASSINPUTMS  */
    SPIRV_INSTRUCTION = 574,       /* SPIRV_INSTRUCTION  */
    SPIRV_EXECUTION_MODE = 575,    /* SPIRV_EXECUTION_MODE  */
    SPIRV_EXECUTION_MODE_ID = 576, /* SPIRV_EXECUTION_MODE_ID  */
    SPIRV_DECORATE = 577,          /* SPIRV_DECORATE  */
    SPIRV_DECORATE_ID = 578,       /* SPIRV_DECORATE_ID  */
    SPIRV_DECORATE_STRING = 579,   /* SPIRV_DECORATE_STRING  */
    SPIRV_TYPE = 580,              /* SPIRV_TYPE  */
    SPIRV_STORAGE_CLASS = 581,     /* SPIRV_STORAGE_CLASS  */
    SPIRV_BY_REFERENCE = 582,      /* SPIRV_BY_REFERENCE  */
    SPIRV_LITERAL = 583,           /* SPIRV_LITERAL  */
    ATTACHMENTEXT = 584,           /* ATTACHMENTEXT  */
    IATTACHMENTEXT = 585,          /* IATTACHMENTEXT  */
    UATTACHMENTEXT = 586,          /* UATTACHMENTEXT  */
    LEFT_OP = 587,                 /* LEFT_OP  */
    RIGHT_OP = 588,                /* RIGHT_OP  */
    INC_OP = 589,                  /* INC_OP  */
    DEC_OP = 590,                  /* DEC_OP  */
    LE_OP = 591,                   /* LE_OP  */
    GE_OP = 592,                   /* GE_OP  */
    EQ_OP = 593,                   /* EQ_OP  */
    NE_OP = 594,                   /* NE_OP  */
    AND_OP = 595,                  /* AND_OP  */
    OR_OP = 596,                   /* OR_OP  */
    XOR_OP = 597,                  /* XOR_OP  */
    MUL_ASSIGN = 598,              /* MUL_ASSIGN  */
    DIV_ASSIGN = 599,              /* DIV_ASSIGN  */
    ADD_ASSIGN = 600,              /* ADD_ASSIGN  */
    MOD_ASSIGN = 601,              /* MOD_ASSIGN  */
    LEFT_ASSIGN = 602,             /* LEFT_ASSIGN  */
    RIGHT_ASSIGN = 603,            /* RIGHT_ASSIGN  */
    AND_ASSIGN = 604,              /* AND_ASSIGN  */
    XOR_ASSIGN = 605,              /* XOR_ASSIGN  */
    OR_ASSIGN = 606,               /* OR_ASSIGN  */
    SUB_ASSIGN = 607,              /* SUB_ASSIGN  */
    STRING_LITERAL = 608,          /* STRING_LITERAL  */
    LEFT_PAREN = 609,              /* LEFT_PAREN  */
    RIGHT_PAREN = 610,             /* RIGHT_PAREN  */
    LEFT_BRACKET = 611,            /* LEFT_BRACKET  */
    RIGHT_BRACKET = 612,           /* RIGHT_BRACKET  */
    LEFT_BRACE = 613,              /* LEFT_BRACE  */
    RIGHT_BRACE = 614,             /* RIGHT_BRACE  */
    DOT = 615,                     /* DOT  */
    COMMA = 616,                   /* COMMA  */
    COLON = 617,                   /* COLON  */
    EQUAL = 618,                   /* EQUAL  */
    SEMICOLON = 619,               /* SEMICOLON  */
    BANG = 620,                    /* BANG  */
    DASH = 621,                    /* DASH  */
    TILDE = 622,                   /* TILDE  */
    PLUS = 623,                    /* PLUS  */
    STAR = 624,                    /* STAR  */
    SLASH = 625,                   /* SLASH  */
    PERCENT = 626,                 /* PERCENT  */
    LEFT_ANGLE = 627,              /* LEFT_ANGLE  */
    RIGHT_ANGLE = 628,             /* RIGHT_ANGLE  */
    VERTICAL_BAR = 629,            /* VERTICAL_BAR  */
    CARET = 630,                   /* CARET  */
    AMPERSAND = 631,               /* AMPERSAND  */
    QUESTION = 632,                /* QUESTION  */
    INVARIANT = 633,               /* INVARIANT  */
    HIGH_PRECISION = 634,          /* HIGH_PRECISION  */
    MEDIUM_PRECISION = 635,        /* MEDIUM_PRECISION  */
    LOW_PRECISION = 636,           /* LOW_PRECISION  */
    PRECISION = 637,               /* PRECISION  */
    PACKED = 638,                  /* PACKED  */
    RESOURCE = 639,                /* RESOURCE  */
    SUPERP = 640,                  /* SUPERP  */
    FLOATCONSTANT = 641,           /* FLOATCONSTANT  */
    INTCONSTANT = 642,             /* INTCONSTANT  */
    UINTCONSTANT = 643,            /* UINTCONSTANT  */
    BOOLCONSTANT = 644,            /* BOOLCONSTANT  */
    IDENTIFIER = 645,              /* IDENTIFIER  */
    TYPE_NAME = 646,               /* TYPE_NAME  */
    CENTROID = 647,                /* CENTROID  */
    IN = 648,                      /* IN  */
    OUT = 649,                     /* OUT  */
    INOUT = 650,                   /* INOUT  */
    STRUCT = 651,                  /* STRUCT  */
    VOID = 652,                    /* VOID  */
    WHILE = 653,                   /* WHILE  */
    BREAK = 654,                   /* BREAK  */
    CONTINUE = 655,                /* CONTINUE  */
    DO = 656,                      /* DO  */
    ELSE = 657,                    /* ELSE  */
    FOR = 658,                     /* FOR  */
    IF = 659,                      /* IF  */
    DISCARD = 660,                 /* DISCARD  */
    RETURN = 661,                  /* RETURN  */
    SWITCH = 662,                  /* SWITCH  */
    CASE = 663,                    /* CASE  */
    DEFAULT = 664,                 /* DEFAULT  */
    TERMINATE_INVOCATION = 665,    /* TERMINATE_INVOCATION  */
    TERMINATE_RAY = 666,           /* TERMINATE_RAY  */
    IGNORE_INTERSECTION = 667,     /* IGNORE_INTERSECTION  */
    UNIFORM = 668,                 /* UNIFORM  */
    SHARED = 669,                  /* SHARED  */
    BUFFER = 670,                  /* BUFFER  */
    TILEIMAGEEXT = 671,            /* TILEIMAGEEXT  */
    FLAT = 672,                    /* FLAT  */
    SMOOTH = 673,                  /* SMOOTH  */
    LAYOUT = 674,                  /* LAYOUT  */
    DOUBLECONSTANT = 675,          /* DOUBLECONSTANT  */
    INT16CONSTANT = 676,           /* INT16CONSTANT  */
    UINT16CONSTANT = 677,          /* UINT16CONSTANT  */
    FLOAT16CONSTANT = 678,         /* FLOAT16CONSTANT  */
    INT32CONSTANT = 679,           /* INT32CONSTANT  */
    UINT32CONSTANT = 680,          /* UINT32CONSTANT  */
    INT64CONSTANT = 681,           /* INT64CONSTANT  */
    UINT64CONSTANT = 682,          /* UINT64CONSTANT  */
    SUBROUTINE = 683,              /* SUBROUTINE  */
    DEMOTE = 684,                  /* DEMOTE  */
    PAYLOADNV = 685,               /* PAYLOADNV  */
    PAYLOADINNV = 686,             /* PAYLOADINNV  */
    HITATTRNV = 687,               /* HITATTRNV  */
    CALLDATANV = 688,              /* CALLDATANV  */
    CALLDATAINNV = 689,            /* CALLDATAINNV  */
    PAYLOADEXT = 690,              /* PAYLOADEXT  */
    PAYLOADINEXT = 691,            /* PAYLOADINEXT  */
    HITATTREXT = 692,              /* HITATTREXT  */
    CALLDATAEXT = 693,             /* CALLDATAEXT  */
    CALLDATAINEXT = 694,           /* CALLDATAINEXT  */
    PATCH = 695,                   /* PATCH  */
    SAMPLE = 696,                  /* SAMPLE  */
    NONUNIFORM = 697,              /* NONUNIFORM  */
    COHERENT = 698,                /* COHERENT  */
    VOLATILE = 699,                /* VOLATILE  */
    RESTRICT = 700,                /* RESTRICT  */
    READONLY = 701,                /* READONLY  */
    WRITEONLY = 702,               /* WRITEONLY  */
    DEVICECOHERENT = 703,          /* DEVICECOHERENT  */
    QUEUEFAMILYCOHERENT = 704,     /* QUEUEFAMILYCOHERENT  */
    WORKGROUPCOHERENT = 705,       /* WORKGROUPCOHERENT  */
    SUBGROUPCOHERENT = 706,        /* SUBGROUPCOHERENT  */
    NONPRIVATE = 707,              /* NONPRIVATE  */
    SHADERCALLCOHERENT = 708,      /* SHADERCALLCOHERENT  */
    NOPERSPECTIVE = 709,           /* NOPERSPECTIVE  */
    EXPLICITINTERPAMD = 710,       /* EXPLICITINTERPAMD  */
    PERVERTEXEXT = 711,            /* PERVERTEXEXT  */
    PERVERTEXNV = 712,             /* PERVERTEXNV  */
    PERPRIMITIVENV = 713,          /* PERPRIMITIVENV  */
    PERVIEWNV = 714,               /* PERVIEWNV  */
    PERTASKNV = 715,               /* PERTASKNV  */
    PERPRIMITIVEEXT = 716,         /* PERPRIMITIVEEXT  */
    TASKPAYLOADWORKGROUPEXT = 717, /* TASKPAYLOADWORKGROUPEXT  */
    PRECISE = 718                  /* PRECISE  */
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
        glslang::TArraySizes* typeParameters;
    } interm;

#line 566 "MachineIndependent/glslang_tab.cpp.h"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif




int yyparse (glslang::TParseContext* pParseContext);


#endif /* !YY_YY_MACHINEINDEPENDENT_GLSLANG_TAB_CPP_H_INCLUDED  */
