/* A Bison parser, made by GNU Bison 3.7.4.  */

/* Bison interface for Yacc-like parsers in C

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
    SAMPLERCUBEARRAY = 421,        /* SAMPLERCUBEARRAY  */
    SAMPLERCUBEARRAYSHADOW = 422,  /* SAMPLERCUBEARRAYSHADOW  */
    ISAMPLERCUBEARRAY = 423,       /* ISAMPLERCUBEARRAY  */
    USAMPLERCUBEARRAY = 424,       /* USAMPLERCUBEARRAY  */
    SAMPLER1D = 425,               /* SAMPLER1D  */
    SAMPLER1DARRAY = 426,          /* SAMPLER1DARRAY  */
    SAMPLER1DARRAYSHADOW = 427,    /* SAMPLER1DARRAYSHADOW  */
    ISAMPLER1D = 428,              /* ISAMPLER1D  */
    SAMPLER1DSHADOW = 429,         /* SAMPLER1DSHADOW  */
    SAMPLER2DRECT = 430,           /* SAMPLER2DRECT  */
    SAMPLER2DRECTSHADOW = 431,     /* SAMPLER2DRECTSHADOW  */
    ISAMPLER2DRECT = 432,          /* ISAMPLER2DRECT  */
    USAMPLER2DRECT = 433,          /* USAMPLER2DRECT  */
    SAMPLERBUFFER = 434,           /* SAMPLERBUFFER  */
    ISAMPLERBUFFER = 435,          /* ISAMPLERBUFFER  */
    USAMPLERBUFFER = 436,          /* USAMPLERBUFFER  */
    SAMPLER2DMS = 437,             /* SAMPLER2DMS  */
    ISAMPLER2DMS = 438,            /* ISAMPLER2DMS  */
    USAMPLER2DMS = 439,            /* USAMPLER2DMS  */
    SAMPLER2DMSARRAY = 440,        /* SAMPLER2DMSARRAY  */
    ISAMPLER2DMSARRAY = 441,       /* ISAMPLER2DMSARRAY  */
    USAMPLER2DMSARRAY = 442,       /* USAMPLER2DMSARRAY  */
    SAMPLEREXTERNALOES = 443,      /* SAMPLEREXTERNALOES  */
    SAMPLEREXTERNAL2DY2YEXT = 444, /* SAMPLEREXTERNAL2DY2YEXT  */
    ISAMPLER1DARRAY = 445,         /* ISAMPLER1DARRAY  */
    USAMPLER1D = 446,              /* USAMPLER1D  */
    USAMPLER1DARRAY = 447,         /* USAMPLER1DARRAY  */
    F16SAMPLER1D = 448,            /* F16SAMPLER1D  */
    F16SAMPLER2D = 449,            /* F16SAMPLER2D  */
    F16SAMPLER3D = 450,            /* F16SAMPLER3D  */
    F16SAMPLER2DRECT = 451,        /* F16SAMPLER2DRECT  */
    F16SAMPLERCUBE = 452,          /* F16SAMPLERCUBE  */
    F16SAMPLER1DARRAY = 453,       /* F16SAMPLER1DARRAY  */
    F16SAMPLER2DARRAY = 454,       /* F16SAMPLER2DARRAY  */
    F16SAMPLERCUBEARRAY = 455,     /* F16SAMPLERCUBEARRAY  */
    F16SAMPLERBUFFER = 456,        /* F16SAMPLERBUFFER  */
    F16SAMPLER2DMS = 457,          /* F16SAMPLER2DMS  */
    F16SAMPLER2DMSARRAY = 458,     /* F16SAMPLER2DMSARRAY  */
    F16SAMPLER1DSHADOW = 459,      /* F16SAMPLER1DSHADOW  */
    F16SAMPLER2DSHADOW = 460,      /* F16SAMPLER2DSHADOW  */
    F16SAMPLER1DARRAYSHADOW = 461, /* F16SAMPLER1DARRAYSHADOW  */
    F16SAMPLER2DARRAYSHADOW = 462, /* F16SAMPLER2DARRAYSHADOW  */
    F16SAMPLER2DRECTSHADOW = 463,  /* F16SAMPLER2DRECTSHADOW  */
    F16SAMPLERCUBESHADOW = 464,    /* F16SAMPLERCUBESHADOW  */
    F16SAMPLERCUBEARRAYSHADOW = 465, /* F16SAMPLERCUBEARRAYSHADOW  */
    IMAGE1D = 466,                 /* IMAGE1D  */
    IIMAGE1D = 467,                /* IIMAGE1D  */
    UIMAGE1D = 468,                /* UIMAGE1D  */
    IMAGE2D = 469,                 /* IMAGE2D  */
    IIMAGE2D = 470,                /* IIMAGE2D  */
    UIMAGE2D = 471,                /* UIMAGE2D  */
    IMAGE3D = 472,                 /* IMAGE3D  */
    IIMAGE3D = 473,                /* IIMAGE3D  */
    UIMAGE3D = 474,                /* UIMAGE3D  */
    IMAGE2DRECT = 475,             /* IMAGE2DRECT  */
    IIMAGE2DRECT = 476,            /* IIMAGE2DRECT  */
    UIMAGE2DRECT = 477,            /* UIMAGE2DRECT  */
    IMAGECUBE = 478,               /* IMAGECUBE  */
    IIMAGECUBE = 479,              /* IIMAGECUBE  */
    UIMAGECUBE = 480,              /* UIMAGECUBE  */
    IMAGEBUFFER = 481,             /* IMAGEBUFFER  */
    IIMAGEBUFFER = 482,            /* IIMAGEBUFFER  */
    UIMAGEBUFFER = 483,            /* UIMAGEBUFFER  */
    IMAGE1DARRAY = 484,            /* IMAGE1DARRAY  */
    IIMAGE1DARRAY = 485,           /* IIMAGE1DARRAY  */
    UIMAGE1DARRAY = 486,           /* UIMAGE1DARRAY  */
    IMAGE2DARRAY = 487,            /* IMAGE2DARRAY  */
    IIMAGE2DARRAY = 488,           /* IIMAGE2DARRAY  */
    UIMAGE2DARRAY = 489,           /* UIMAGE2DARRAY  */
    IMAGECUBEARRAY = 490,          /* IMAGECUBEARRAY  */
    IIMAGECUBEARRAY = 491,         /* IIMAGECUBEARRAY  */
    UIMAGECUBEARRAY = 492,         /* UIMAGECUBEARRAY  */
    IMAGE2DMS = 493,               /* IMAGE2DMS  */
    IIMAGE2DMS = 494,              /* IIMAGE2DMS  */
    UIMAGE2DMS = 495,              /* UIMAGE2DMS  */
    IMAGE2DMSARRAY = 496,          /* IMAGE2DMSARRAY  */
    IIMAGE2DMSARRAY = 497,         /* IIMAGE2DMSARRAY  */
    UIMAGE2DMSARRAY = 498,         /* UIMAGE2DMSARRAY  */
    F16IMAGE1D = 499,              /* F16IMAGE1D  */
    F16IMAGE2D = 500,              /* F16IMAGE2D  */
    F16IMAGE3D = 501,              /* F16IMAGE3D  */
    F16IMAGE2DRECT = 502,          /* F16IMAGE2DRECT  */
    F16IMAGECUBE = 503,            /* F16IMAGECUBE  */
    F16IMAGE1DARRAY = 504,         /* F16IMAGE1DARRAY  */
    F16IMAGE2DARRAY = 505,         /* F16IMAGE2DARRAY  */
    F16IMAGECUBEARRAY = 506,       /* F16IMAGECUBEARRAY  */
    F16IMAGEBUFFER = 507,          /* F16IMAGEBUFFER  */
    F16IMAGE2DMS = 508,            /* F16IMAGE2DMS  */
    F16IMAGE2DMSARRAY = 509,       /* F16IMAGE2DMSARRAY  */
    I64IMAGE1D = 510,              /* I64IMAGE1D  */
    U64IMAGE1D = 511,              /* U64IMAGE1D  */
    I64IMAGE2D = 512,              /* I64IMAGE2D  */
    U64IMAGE2D = 513,              /* U64IMAGE2D  */
    I64IMAGE3D = 514,              /* I64IMAGE3D  */
    U64IMAGE3D = 515,              /* U64IMAGE3D  */
    I64IMAGE2DRECT = 516,          /* I64IMAGE2DRECT  */
    U64IMAGE2DRECT = 517,          /* U64IMAGE2DRECT  */
    I64IMAGECUBE = 518,            /* I64IMAGECUBE  */
    U64IMAGECUBE = 519,            /* U64IMAGECUBE  */
    I64IMAGEBUFFER = 520,          /* I64IMAGEBUFFER  */
    U64IMAGEBUFFER = 521,          /* U64IMAGEBUFFER  */
    I64IMAGE1DARRAY = 522,         /* I64IMAGE1DARRAY  */
    U64IMAGE1DARRAY = 523,         /* U64IMAGE1DARRAY  */
    I64IMAGE2DARRAY = 524,         /* I64IMAGE2DARRAY  */
    U64IMAGE2DARRAY = 525,         /* U64IMAGE2DARRAY  */
    I64IMAGECUBEARRAY = 526,       /* I64IMAGECUBEARRAY  */
    U64IMAGECUBEARRAY = 527,       /* U64IMAGECUBEARRAY  */
    I64IMAGE2DMS = 528,            /* I64IMAGE2DMS  */
    U64IMAGE2DMS = 529,            /* U64IMAGE2DMS  */
    I64IMAGE2DMSARRAY = 530,       /* I64IMAGE2DMSARRAY  */
    U64IMAGE2DMSARRAY = 531,       /* U64IMAGE2DMSARRAY  */
    TEXTURECUBEARRAY = 532,        /* TEXTURECUBEARRAY  */
    ITEXTURECUBEARRAY = 533,       /* ITEXTURECUBEARRAY  */
    UTEXTURECUBEARRAY = 534,       /* UTEXTURECUBEARRAY  */
    TEXTURE1D = 535,               /* TEXTURE1D  */
    ITEXTURE1D = 536,              /* ITEXTURE1D  */
    UTEXTURE1D = 537,              /* UTEXTURE1D  */
    TEXTURE1DARRAY = 538,          /* TEXTURE1DARRAY  */
    ITEXTURE1DARRAY = 539,         /* ITEXTURE1DARRAY  */
    UTEXTURE1DARRAY = 540,         /* UTEXTURE1DARRAY  */
    TEXTURE2DRECT = 541,           /* TEXTURE2DRECT  */
    ITEXTURE2DRECT = 542,          /* ITEXTURE2DRECT  */
    UTEXTURE2DRECT = 543,          /* UTEXTURE2DRECT  */
    TEXTUREBUFFER = 544,           /* TEXTUREBUFFER  */
    ITEXTUREBUFFER = 545,          /* ITEXTUREBUFFER  */
    UTEXTUREBUFFER = 546,          /* UTEXTUREBUFFER  */
    TEXTURE2DMS = 547,             /* TEXTURE2DMS  */
    ITEXTURE2DMS = 548,            /* ITEXTURE2DMS  */
    UTEXTURE2DMS = 549,            /* UTEXTURE2DMS  */
    TEXTURE2DMSARRAY = 550,        /* TEXTURE2DMSARRAY  */
    ITEXTURE2DMSARRAY = 551,       /* ITEXTURE2DMSARRAY  */
    UTEXTURE2DMSARRAY = 552,       /* UTEXTURE2DMSARRAY  */
    F16TEXTURE1D = 553,            /* F16TEXTURE1D  */
    F16TEXTURE2D = 554,            /* F16TEXTURE2D  */
    F16TEXTURE3D = 555,            /* F16TEXTURE3D  */
    F16TEXTURE2DRECT = 556,        /* F16TEXTURE2DRECT  */
    F16TEXTURECUBE = 557,          /* F16TEXTURECUBE  */
    F16TEXTURE1DARRAY = 558,       /* F16TEXTURE1DARRAY  */
    F16TEXTURE2DARRAY = 559,       /* F16TEXTURE2DARRAY  */
    F16TEXTURECUBEARRAY = 560,     /* F16TEXTURECUBEARRAY  */
    F16TEXTUREBUFFER = 561,        /* F16TEXTUREBUFFER  */
    F16TEXTURE2DMS = 562,          /* F16TEXTURE2DMS  */
    F16TEXTURE2DMSARRAY = 563,     /* F16TEXTURE2DMSARRAY  */
    SUBPASSINPUT = 564,            /* SUBPASSINPUT  */
    SUBPASSINPUTMS = 565,          /* SUBPASSINPUTMS  */
    ISUBPASSINPUT = 566,           /* ISUBPASSINPUT  */
    ISUBPASSINPUTMS = 567,         /* ISUBPASSINPUTMS  */
    USUBPASSINPUT = 568,           /* USUBPASSINPUT  */
    USUBPASSINPUTMS = 569,         /* USUBPASSINPUTMS  */
    F16SUBPASSINPUT = 570,         /* F16SUBPASSINPUT  */
    F16SUBPASSINPUTMS = 571,       /* F16SUBPASSINPUTMS  */
    LEFT_OP = 572,                 /* LEFT_OP  */
    RIGHT_OP = 573,                /* RIGHT_OP  */
    INC_OP = 574,                  /* INC_OP  */
    DEC_OP = 575,                  /* DEC_OP  */
    LE_OP = 576,                   /* LE_OP  */
    GE_OP = 577,                   /* GE_OP  */
    EQ_OP = 578,                   /* EQ_OP  */
    NE_OP = 579,                   /* NE_OP  */
    AND_OP = 580,                  /* AND_OP  */
    OR_OP = 581,                   /* OR_OP  */
    XOR_OP = 582,                  /* XOR_OP  */
    MUL_ASSIGN = 583,              /* MUL_ASSIGN  */
    DIV_ASSIGN = 584,              /* DIV_ASSIGN  */
    ADD_ASSIGN = 585,              /* ADD_ASSIGN  */
    MOD_ASSIGN = 586,              /* MOD_ASSIGN  */
    LEFT_ASSIGN = 587,             /* LEFT_ASSIGN  */
    RIGHT_ASSIGN = 588,            /* RIGHT_ASSIGN  */
    AND_ASSIGN = 589,              /* AND_ASSIGN  */
    XOR_ASSIGN = 590,              /* XOR_ASSIGN  */
    OR_ASSIGN = 591,               /* OR_ASSIGN  */
    SUB_ASSIGN = 592,              /* SUB_ASSIGN  */
    STRING_LITERAL = 593,          /* STRING_LITERAL  */
    LEFT_PAREN = 594,              /* LEFT_PAREN  */
    RIGHT_PAREN = 595,             /* RIGHT_PAREN  */
    LEFT_BRACKET = 596,            /* LEFT_BRACKET  */
    RIGHT_BRACKET = 597,           /* RIGHT_BRACKET  */
    LEFT_BRACE = 598,              /* LEFT_BRACE  */
    RIGHT_BRACE = 599,             /* RIGHT_BRACE  */
    DOT = 600,                     /* DOT  */
    COMMA = 601,                   /* COMMA  */
    COLON = 602,                   /* COLON  */
    EQUAL = 603,                   /* EQUAL  */
    SEMICOLON = 604,               /* SEMICOLON  */
    BANG = 605,                    /* BANG  */
    DASH = 606,                    /* DASH  */
    TILDE = 607,                   /* TILDE  */
    PLUS = 608,                    /* PLUS  */
    STAR = 609,                    /* STAR  */
    SLASH = 610,                   /* SLASH  */
    PERCENT = 611,                 /* PERCENT  */
    LEFT_ANGLE = 612,              /* LEFT_ANGLE  */
    RIGHT_ANGLE = 613,             /* RIGHT_ANGLE  */
    VERTICAL_BAR = 614,            /* VERTICAL_BAR  */
    CARET = 615,                   /* CARET  */
    AMPERSAND = 616,               /* AMPERSAND  */
    QUESTION = 617,                /* QUESTION  */
    INVARIANT = 618,               /* INVARIANT  */
    HIGH_PRECISION = 619,          /* HIGH_PRECISION  */
    MEDIUM_PRECISION = 620,        /* MEDIUM_PRECISION  */
    LOW_PRECISION = 621,           /* LOW_PRECISION  */
    PRECISION = 622,               /* PRECISION  */
    PACKED = 623,                  /* PACKED  */
    RESOURCE = 624,                /* RESOURCE  */
    SUPERP = 625,                  /* SUPERP  */
    FLOATCONSTANT = 626,           /* FLOATCONSTANT  */
    INTCONSTANT = 627,             /* INTCONSTANT  */
    UINTCONSTANT = 628,            /* UINTCONSTANT  */
    BOOLCONSTANT = 629,            /* BOOLCONSTANT  */
    IDENTIFIER = 630,              /* IDENTIFIER  */
    TYPE_NAME = 631,               /* TYPE_NAME  */
    CENTROID = 632,                /* CENTROID  */
    IN = 633,                      /* IN  */
    OUT = 634,                     /* OUT  */
    INOUT = 635,                   /* INOUT  */
    STRUCT = 636,                  /* STRUCT  */
    VOID = 637,                    /* VOID  */
    WHILE = 638,                   /* WHILE  */
    BREAK = 639,                   /* BREAK  */
    CONTINUE = 640,                /* CONTINUE  */
    DO = 641,                      /* DO  */
    ELSE = 642,                    /* ELSE  */
    FOR = 643,                     /* FOR  */
    IF = 644,                      /* IF  */
    DISCARD = 645,                 /* DISCARD  */
    RETURN = 646,                  /* RETURN  */
    SWITCH = 647,                  /* SWITCH  */
    CASE = 648,                    /* CASE  */
    DEFAULT = 649,                 /* DEFAULT  */
    TERMINATE_INVOCATION = 650,    /* TERMINATE_INVOCATION  */
    TERMINATE_RAY = 651,           /* TERMINATE_RAY  */
    IGNORE_INTERSECTION = 652,     /* IGNORE_INTERSECTION  */
    UNIFORM = 653,                 /* UNIFORM  */
    SHARED = 654,                  /* SHARED  */
    BUFFER = 655,                  /* BUFFER  */
    FLAT = 656,                    /* FLAT  */
    SMOOTH = 657,                  /* SMOOTH  */
    LAYOUT = 658,                  /* LAYOUT  */
    DOUBLECONSTANT = 659,          /* DOUBLECONSTANT  */
    INT16CONSTANT = 660,           /* INT16CONSTANT  */
    UINT16CONSTANT = 661,          /* UINT16CONSTANT  */
    FLOAT16CONSTANT = 662,         /* FLOAT16CONSTANT  */
    INT32CONSTANT = 663,           /* INT32CONSTANT  */
    UINT32CONSTANT = 664,          /* UINT32CONSTANT  */
    INT64CONSTANT = 665,           /* INT64CONSTANT  */
    UINT64CONSTANT = 666,          /* UINT64CONSTANT  */
    SUBROUTINE = 667,              /* SUBROUTINE  */
    DEMOTE = 668,                  /* DEMOTE  */
    PAYLOADNV = 669,               /* PAYLOADNV  */
    PAYLOADINNV = 670,             /* PAYLOADINNV  */
    HITATTRNV = 671,               /* HITATTRNV  */
    CALLDATANV = 672,              /* CALLDATANV  */
    CALLDATAINNV = 673,            /* CALLDATAINNV  */
    PAYLOADEXT = 674,              /* PAYLOADEXT  */
    PAYLOADINEXT = 675,            /* PAYLOADINEXT  */
    HITATTREXT = 676,              /* HITATTREXT  */
    CALLDATAEXT = 677,             /* CALLDATAEXT  */
    CALLDATAINEXT = 678,           /* CALLDATAINEXT  */
    PATCH = 679,                   /* PATCH  */
    SAMPLE = 680,                  /* SAMPLE  */
    NONUNIFORM = 681,              /* NONUNIFORM  */
    COHERENT = 682,                /* COHERENT  */
    VOLATILE = 683,                /* VOLATILE  */
    RESTRICT = 684,                /* RESTRICT  */
    READONLY = 685,                /* READONLY  */
    WRITEONLY = 686,               /* WRITEONLY  */
    DEVICECOHERENT = 687,          /* DEVICECOHERENT  */
    QUEUEFAMILYCOHERENT = 688,     /* QUEUEFAMILYCOHERENT  */
    WORKGROUPCOHERENT = 689,       /* WORKGROUPCOHERENT  */
    SUBGROUPCOHERENT = 690,        /* SUBGROUPCOHERENT  */
    NONPRIVATE = 691,              /* NONPRIVATE  */
    SHADERCALLCOHERENT = 692,      /* SHADERCALLCOHERENT  */
    NOPERSPECTIVE = 693,           /* NOPERSPECTIVE  */
    EXPLICITINTERPAMD = 694,       /* EXPLICITINTERPAMD  */
    PERVERTEXNV = 695,             /* PERVERTEXNV  */
    PERPRIMITIVENV = 696,          /* PERPRIMITIVENV  */
    PERVIEWNV = 697,               /* PERVIEWNV  */
    PERTASKNV = 698,               /* PERTASKNV  */
    PRECISE = 699                  /* PRECISE  */
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

#line 544 "MachineIndependent/glslang_tab.cpp.h"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif



int yyparse (glslang::TParseContext* pParseContext);

#endif /* !YY_YY_MACHINEINDEPENDENT_GLSLANG_TAB_CPP_H_INCLUDED  */
