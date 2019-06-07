/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015 Free Software Foundation, Inc.

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

#ifndef YY_YY_MACHINEINDEPENDENT_GLSLANG_TAB_CPP_H_INCLUDED
# define YY_YY_MACHINEINDEPENDENT_GLSLANG_TAB_CPP_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 1
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    ATTRIBUTE = 258,
    VARYING = 259,
    FLOAT16_T = 260,
    FLOAT = 261,
    FLOAT32_T = 262,
    DOUBLE = 263,
    FLOAT64_T = 264,
    CONST = 265,
    BOOL = 266,
    INT = 267,
    UINT = 268,
    INT64_T = 269,
    UINT64_T = 270,
    INT32_T = 271,
    UINT32_T = 272,
    INT16_T = 273,
    UINT16_T = 274,
    INT8_T = 275,
    UINT8_T = 276,
    BREAK = 277,
    CONTINUE = 278,
    DO = 279,
    ELSE = 280,
    FOR = 281,
    IF = 282,
    DISCARD = 283,
    RETURN = 284,
    SWITCH = 285,
    CASE = 286,
    DEFAULT = 287,
    SUBROUTINE = 288,
    BVEC2 = 289,
    BVEC3 = 290,
    BVEC4 = 291,
    IVEC2 = 292,
    IVEC3 = 293,
    IVEC4 = 294,
    UVEC2 = 295,
    UVEC3 = 296,
    UVEC4 = 297,
    I64VEC2 = 298,
    I64VEC3 = 299,
    I64VEC4 = 300,
    U64VEC2 = 301,
    U64VEC3 = 302,
    U64VEC4 = 303,
    I32VEC2 = 304,
    I32VEC3 = 305,
    I32VEC4 = 306,
    U32VEC2 = 307,
    U32VEC3 = 308,
    U32VEC4 = 309,
    I16VEC2 = 310,
    I16VEC3 = 311,
    I16VEC4 = 312,
    U16VEC2 = 313,
    U16VEC3 = 314,
    U16VEC4 = 315,
    I8VEC2 = 316,
    I8VEC3 = 317,
    I8VEC4 = 318,
    U8VEC2 = 319,
    U8VEC3 = 320,
    U8VEC4 = 321,
    VEC2 = 322,
    VEC3 = 323,
    VEC4 = 324,
    MAT2 = 325,
    MAT3 = 326,
    MAT4 = 327,
    CENTROID = 328,
    IN = 329,
    OUT = 330,
    INOUT = 331,
    UNIFORM = 332,
    PATCH = 333,
    SAMPLE = 334,
    BUFFER = 335,
    SHARED = 336,
    NONUNIFORM = 337,
    PAYLOADNV = 338,
    PAYLOADINNV = 339,
    HITATTRNV = 340,
    CALLDATANV = 341,
    CALLDATAINNV = 342,
    COHERENT = 343,
    VOLATILE = 344,
    RESTRICT = 345,
    READONLY = 346,
    WRITEONLY = 347,
    DEVICECOHERENT = 348,
    QUEUEFAMILYCOHERENT = 349,
    WORKGROUPCOHERENT = 350,
    SUBGROUPCOHERENT = 351,
    NONPRIVATE = 352,
    DVEC2 = 353,
    DVEC3 = 354,
    DVEC4 = 355,
    DMAT2 = 356,
    DMAT3 = 357,
    DMAT4 = 358,
    F16VEC2 = 359,
    F16VEC3 = 360,
    F16VEC4 = 361,
    F16MAT2 = 362,
    F16MAT3 = 363,
    F16MAT4 = 364,
    F32VEC2 = 365,
    F32VEC3 = 366,
    F32VEC4 = 367,
    F32MAT2 = 368,
    F32MAT3 = 369,
    F32MAT4 = 370,
    F64VEC2 = 371,
    F64VEC3 = 372,
    F64VEC4 = 373,
    F64MAT2 = 374,
    F64MAT3 = 375,
    F64MAT4 = 376,
    NOPERSPECTIVE = 377,
    FLAT = 378,
    SMOOTH = 379,
    LAYOUT = 380,
    EXPLICITINTERPAMD = 381,
    PERVERTEXNV = 382,
    PERPRIMITIVENV = 383,
    PERVIEWNV = 384,
    PERTASKNV = 385,
    MAT2X2 = 386,
    MAT2X3 = 387,
    MAT2X4 = 388,
    MAT3X2 = 389,
    MAT3X3 = 390,
    MAT3X4 = 391,
    MAT4X2 = 392,
    MAT4X3 = 393,
    MAT4X4 = 394,
    DMAT2X2 = 395,
    DMAT2X3 = 396,
    DMAT2X4 = 397,
    DMAT3X2 = 398,
    DMAT3X3 = 399,
    DMAT3X4 = 400,
    DMAT4X2 = 401,
    DMAT4X3 = 402,
    DMAT4X4 = 403,
    F16MAT2X2 = 404,
    F16MAT2X3 = 405,
    F16MAT2X4 = 406,
    F16MAT3X2 = 407,
    F16MAT3X3 = 408,
    F16MAT3X4 = 409,
    F16MAT4X2 = 410,
    F16MAT4X3 = 411,
    F16MAT4X4 = 412,
    F32MAT2X2 = 413,
    F32MAT2X3 = 414,
    F32MAT2X4 = 415,
    F32MAT3X2 = 416,
    F32MAT3X3 = 417,
    F32MAT3X4 = 418,
    F32MAT4X2 = 419,
    F32MAT4X3 = 420,
    F32MAT4X4 = 421,
    F64MAT2X2 = 422,
    F64MAT2X3 = 423,
    F64MAT2X4 = 424,
    F64MAT3X2 = 425,
    F64MAT3X3 = 426,
    F64MAT3X4 = 427,
    F64MAT4X2 = 428,
    F64MAT4X3 = 429,
    F64MAT4X4 = 430,
    ATOMIC_UINT = 431,
    ACCSTRUCTNV = 432,
    FCOOPMATNV = 433,
    SAMPLER1D = 434,
    SAMPLER2D = 435,
    SAMPLER3D = 436,
    SAMPLERCUBE = 437,
    SAMPLER1DSHADOW = 438,
    SAMPLER2DSHADOW = 439,
    SAMPLERCUBESHADOW = 440,
    SAMPLER1DARRAY = 441,
    SAMPLER2DARRAY = 442,
    SAMPLER1DARRAYSHADOW = 443,
    SAMPLER2DARRAYSHADOW = 444,
    ISAMPLER1D = 445,
    ISAMPLER2D = 446,
    ISAMPLER3D = 447,
    ISAMPLERCUBE = 448,
    ISAMPLER1DARRAY = 449,
    ISAMPLER2DARRAY = 450,
    USAMPLER1D = 451,
    USAMPLER2D = 452,
    USAMPLER3D = 453,
    USAMPLERCUBE = 454,
    USAMPLER1DARRAY = 455,
    USAMPLER2DARRAY = 456,
    SAMPLER2DRECT = 457,
    SAMPLER2DRECTSHADOW = 458,
    ISAMPLER2DRECT = 459,
    USAMPLER2DRECT = 460,
    SAMPLERBUFFER = 461,
    ISAMPLERBUFFER = 462,
    USAMPLERBUFFER = 463,
    SAMPLERCUBEARRAY = 464,
    SAMPLERCUBEARRAYSHADOW = 465,
    ISAMPLERCUBEARRAY = 466,
    USAMPLERCUBEARRAY = 467,
    SAMPLER2DMS = 468,
    ISAMPLER2DMS = 469,
    USAMPLER2DMS = 470,
    SAMPLER2DMSARRAY = 471,
    ISAMPLER2DMSARRAY = 472,
    USAMPLER2DMSARRAY = 473,
    SAMPLEREXTERNALOES = 474,
    SAMPLEREXTERNAL2DY2YEXT = 475,
    F16SAMPLER1D = 476,
    F16SAMPLER2D = 477,
    F16SAMPLER3D = 478,
    F16SAMPLER2DRECT = 479,
    F16SAMPLERCUBE = 480,
    F16SAMPLER1DARRAY = 481,
    F16SAMPLER2DARRAY = 482,
    F16SAMPLERCUBEARRAY = 483,
    F16SAMPLERBUFFER = 484,
    F16SAMPLER2DMS = 485,
    F16SAMPLER2DMSARRAY = 486,
    F16SAMPLER1DSHADOW = 487,
    F16SAMPLER2DSHADOW = 488,
    F16SAMPLER1DARRAYSHADOW = 489,
    F16SAMPLER2DARRAYSHADOW = 490,
    F16SAMPLER2DRECTSHADOW = 491,
    F16SAMPLERCUBESHADOW = 492,
    F16SAMPLERCUBEARRAYSHADOW = 493,
    SAMPLER = 494,
    SAMPLERSHADOW = 495,
    TEXTURE1D = 496,
    TEXTURE2D = 497,
    TEXTURE3D = 498,
    TEXTURECUBE = 499,
    TEXTURE1DARRAY = 500,
    TEXTURE2DARRAY = 501,
    ITEXTURE1D = 502,
    ITEXTURE2D = 503,
    ITEXTURE3D = 504,
    ITEXTURECUBE = 505,
    ITEXTURE1DARRAY = 506,
    ITEXTURE2DARRAY = 507,
    UTEXTURE1D = 508,
    UTEXTURE2D = 509,
    UTEXTURE3D = 510,
    UTEXTURECUBE = 511,
    UTEXTURE1DARRAY = 512,
    UTEXTURE2DARRAY = 513,
    TEXTURE2DRECT = 514,
    ITEXTURE2DRECT = 515,
    UTEXTURE2DRECT = 516,
    TEXTUREBUFFER = 517,
    ITEXTUREBUFFER = 518,
    UTEXTUREBUFFER = 519,
    TEXTURECUBEARRAY = 520,
    ITEXTURECUBEARRAY = 521,
    UTEXTURECUBEARRAY = 522,
    TEXTURE2DMS = 523,
    ITEXTURE2DMS = 524,
    UTEXTURE2DMS = 525,
    TEXTURE2DMSARRAY = 526,
    ITEXTURE2DMSARRAY = 527,
    UTEXTURE2DMSARRAY = 528,
    F16TEXTURE1D = 529,
    F16TEXTURE2D = 530,
    F16TEXTURE3D = 531,
    F16TEXTURE2DRECT = 532,
    F16TEXTURECUBE = 533,
    F16TEXTURE1DARRAY = 534,
    F16TEXTURE2DARRAY = 535,
    F16TEXTURECUBEARRAY = 536,
    F16TEXTUREBUFFER = 537,
    F16TEXTURE2DMS = 538,
    F16TEXTURE2DMSARRAY = 539,
    SUBPASSINPUT = 540,
    SUBPASSINPUTMS = 541,
    ISUBPASSINPUT = 542,
    ISUBPASSINPUTMS = 543,
    USUBPASSINPUT = 544,
    USUBPASSINPUTMS = 545,
    F16SUBPASSINPUT = 546,
    F16SUBPASSINPUTMS = 547,
    IMAGE1D = 548,
    IIMAGE1D = 549,
    UIMAGE1D = 550,
    IMAGE2D = 551,
    IIMAGE2D = 552,
    UIMAGE2D = 553,
    IMAGE3D = 554,
    IIMAGE3D = 555,
    UIMAGE3D = 556,
    IMAGE2DRECT = 557,
    IIMAGE2DRECT = 558,
    UIMAGE2DRECT = 559,
    IMAGECUBE = 560,
    IIMAGECUBE = 561,
    UIMAGECUBE = 562,
    IMAGEBUFFER = 563,
    IIMAGEBUFFER = 564,
    UIMAGEBUFFER = 565,
    IMAGE1DARRAY = 566,
    IIMAGE1DARRAY = 567,
    UIMAGE1DARRAY = 568,
    IMAGE2DARRAY = 569,
    IIMAGE2DARRAY = 570,
    UIMAGE2DARRAY = 571,
    IMAGECUBEARRAY = 572,
    IIMAGECUBEARRAY = 573,
    UIMAGECUBEARRAY = 574,
    IMAGE2DMS = 575,
    IIMAGE2DMS = 576,
    UIMAGE2DMS = 577,
    IMAGE2DMSARRAY = 578,
    IIMAGE2DMSARRAY = 579,
    UIMAGE2DMSARRAY = 580,
    F16IMAGE1D = 581,
    F16IMAGE2D = 582,
    F16IMAGE3D = 583,
    F16IMAGE2DRECT = 584,
    F16IMAGECUBE = 585,
    F16IMAGE1DARRAY = 586,
    F16IMAGE2DARRAY = 587,
    F16IMAGECUBEARRAY = 588,
    F16IMAGEBUFFER = 589,
    F16IMAGE2DMS = 590,
    F16IMAGE2DMSARRAY = 591,
    STRUCT = 592,
    VOID = 593,
    WHILE = 594,
    IDENTIFIER = 595,
    TYPE_NAME = 596,
    FLOATCONSTANT = 597,
    DOUBLECONSTANT = 598,
    INT16CONSTANT = 599,
    UINT16CONSTANT = 600,
    INT32CONSTANT = 601,
    UINT32CONSTANT = 602,
    INTCONSTANT = 603,
    UINTCONSTANT = 604,
    INT64CONSTANT = 605,
    UINT64CONSTANT = 606,
    BOOLCONSTANT = 607,
    FLOAT16CONSTANT = 608,
    LEFT_OP = 609,
    RIGHT_OP = 610,
    INC_OP = 611,
    DEC_OP = 612,
    LE_OP = 613,
    GE_OP = 614,
    EQ_OP = 615,
    NE_OP = 616,
    AND_OP = 617,
    OR_OP = 618,
    XOR_OP = 619,
    MUL_ASSIGN = 620,
    DIV_ASSIGN = 621,
    ADD_ASSIGN = 622,
    MOD_ASSIGN = 623,
    LEFT_ASSIGN = 624,
    RIGHT_ASSIGN = 625,
    AND_ASSIGN = 626,
    XOR_ASSIGN = 627,
    OR_ASSIGN = 628,
    SUB_ASSIGN = 629,
    LEFT_PAREN = 630,
    RIGHT_PAREN = 631,
    LEFT_BRACKET = 632,
    RIGHT_BRACKET = 633,
    LEFT_BRACE = 634,
    RIGHT_BRACE = 635,
    DOT = 636,
    COMMA = 637,
    COLON = 638,
    EQUAL = 639,
    SEMICOLON = 640,
    BANG = 641,
    DASH = 642,
    TILDE = 643,
    PLUS = 644,
    STAR = 645,
    SLASH = 646,
    PERCENT = 647,
    LEFT_ANGLE = 648,
    RIGHT_ANGLE = 649,
    VERTICAL_BAR = 650,
    CARET = 651,
    AMPERSAND = 652,
    QUESTION = 653,
    INVARIANT = 654,
    PRECISE = 655,
    HIGH_PRECISION = 656,
    MEDIUM_PRECISION = 657,
    LOW_PRECISION = 658,
    PRECISION = 659,
    PACKED = 660,
    RESOURCE = 661,
    SUPERP = 662
  };
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED

union YYSTYPE
{
#line 71 "MachineIndependent/glslang.y" /* yacc.c:1909  */

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

#line 498 "MachineIndependent/glslang_tab.cpp.h" /* yacc.c:1909  */
};

typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif



int yyparse (glslang::TParseContext* pParseContext);

#endif /* !YY_YY_MACHINEINDEPENDENT_GLSLANG_TAB_CPP_H_INCLUDED  */
