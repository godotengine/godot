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
    CONST = 258,
    BOOL = 259,
    INT = 260,
    UINT = 261,
    FLOAT = 262,
    BVEC2 = 263,
    BVEC3 = 264,
    BVEC4 = 265,
    IVEC2 = 266,
    IVEC3 = 267,
    IVEC4 = 268,
    UVEC2 = 269,
    UVEC3 = 270,
    UVEC4 = 271,
    VEC2 = 272,
    VEC3 = 273,
    VEC4 = 274,
    MAT2 = 275,
    MAT3 = 276,
    MAT4 = 277,
    MAT2X2 = 278,
    MAT2X3 = 279,
    MAT2X4 = 280,
    MAT3X2 = 281,
    MAT3X3 = 282,
    MAT3X4 = 283,
    MAT4X2 = 284,
    MAT4X3 = 285,
    MAT4X4 = 286,
    SAMPLER2D = 287,
    SAMPLER3D = 288,
    SAMPLERCUBE = 289,
    SAMPLER2DSHADOW = 290,
    SAMPLERCUBESHADOW = 291,
    SAMPLER2DARRAY = 292,
    SAMPLER2DARRAYSHADOW = 293,
    ISAMPLER2D = 294,
    ISAMPLER3D = 295,
    ISAMPLERCUBE = 296,
    ISAMPLER2DARRAY = 297,
    USAMPLER2D = 298,
    USAMPLER3D = 299,
    USAMPLERCUBE = 300,
    USAMPLER2DARRAY = 301,
    SAMPLER = 302,
    SAMPLERSHADOW = 303,
    TEXTURE2D = 304,
    TEXTURE3D = 305,
    TEXTURECUBE = 306,
    TEXTURE2DARRAY = 307,
    ITEXTURE2D = 308,
    ITEXTURE3D = 309,
    ITEXTURECUBE = 310,
    ITEXTURE2DARRAY = 311,
    UTEXTURE2D = 312,
    UTEXTURE3D = 313,
    UTEXTURECUBE = 314,
    UTEXTURE2DARRAY = 315,
    ATTRIBUTE = 316,
    VARYING = 317,
    FLOAT16_T = 318,
    FLOAT32_T = 319,
    DOUBLE = 320,
    FLOAT64_T = 321,
    INT64_T = 322,
    UINT64_T = 323,
    INT32_T = 324,
    UINT32_T = 325,
    INT16_T = 326,
    UINT16_T = 327,
    INT8_T = 328,
    UINT8_T = 329,
    I64VEC2 = 330,
    I64VEC3 = 331,
    I64VEC4 = 332,
    U64VEC2 = 333,
    U64VEC3 = 334,
    U64VEC4 = 335,
    I32VEC2 = 336,
    I32VEC3 = 337,
    I32VEC4 = 338,
    U32VEC2 = 339,
    U32VEC3 = 340,
    U32VEC4 = 341,
    I16VEC2 = 342,
    I16VEC3 = 343,
    I16VEC4 = 344,
    U16VEC2 = 345,
    U16VEC3 = 346,
    U16VEC4 = 347,
    I8VEC2 = 348,
    I8VEC3 = 349,
    I8VEC4 = 350,
    U8VEC2 = 351,
    U8VEC3 = 352,
    U8VEC4 = 353,
    DVEC2 = 354,
    DVEC3 = 355,
    DVEC4 = 356,
    DMAT2 = 357,
    DMAT3 = 358,
    DMAT4 = 359,
    F16VEC2 = 360,
    F16VEC3 = 361,
    F16VEC4 = 362,
    F16MAT2 = 363,
    F16MAT3 = 364,
    F16MAT4 = 365,
    F32VEC2 = 366,
    F32VEC3 = 367,
    F32VEC4 = 368,
    F32MAT2 = 369,
    F32MAT3 = 370,
    F32MAT4 = 371,
    F64VEC2 = 372,
    F64VEC3 = 373,
    F64VEC4 = 374,
    F64MAT2 = 375,
    F64MAT3 = 376,
    F64MAT4 = 377,
    DMAT2X2 = 378,
    DMAT2X3 = 379,
    DMAT2X4 = 380,
    DMAT3X2 = 381,
    DMAT3X3 = 382,
    DMAT3X4 = 383,
    DMAT4X2 = 384,
    DMAT4X3 = 385,
    DMAT4X4 = 386,
    F16MAT2X2 = 387,
    F16MAT2X3 = 388,
    F16MAT2X4 = 389,
    F16MAT3X2 = 390,
    F16MAT3X3 = 391,
    F16MAT3X4 = 392,
    F16MAT4X2 = 393,
    F16MAT4X3 = 394,
    F16MAT4X4 = 395,
    F32MAT2X2 = 396,
    F32MAT2X3 = 397,
    F32MAT2X4 = 398,
    F32MAT3X2 = 399,
    F32MAT3X3 = 400,
    F32MAT3X4 = 401,
    F32MAT4X2 = 402,
    F32MAT4X3 = 403,
    F32MAT4X4 = 404,
    F64MAT2X2 = 405,
    F64MAT2X3 = 406,
    F64MAT2X4 = 407,
    F64MAT3X2 = 408,
    F64MAT3X3 = 409,
    F64MAT3X4 = 410,
    F64MAT4X2 = 411,
    F64MAT4X3 = 412,
    F64MAT4X4 = 413,
    ATOMIC_UINT = 414,
    ACCSTRUCTNV = 415,
    ACCSTRUCTEXT = 416,
    RAYQUERYEXT = 417,
    FCOOPMATNV = 418,
    ICOOPMATNV = 419,
    UCOOPMATNV = 420,
    SAMPLERCUBEARRAY = 421,
    SAMPLERCUBEARRAYSHADOW = 422,
    ISAMPLERCUBEARRAY = 423,
    USAMPLERCUBEARRAY = 424,
    SAMPLER1D = 425,
    SAMPLER1DARRAY = 426,
    SAMPLER1DARRAYSHADOW = 427,
    ISAMPLER1D = 428,
    SAMPLER1DSHADOW = 429,
    SAMPLER2DRECT = 430,
    SAMPLER2DRECTSHADOW = 431,
    ISAMPLER2DRECT = 432,
    USAMPLER2DRECT = 433,
    SAMPLERBUFFER = 434,
    ISAMPLERBUFFER = 435,
    USAMPLERBUFFER = 436,
    SAMPLER2DMS = 437,
    ISAMPLER2DMS = 438,
    USAMPLER2DMS = 439,
    SAMPLER2DMSARRAY = 440,
    ISAMPLER2DMSARRAY = 441,
    USAMPLER2DMSARRAY = 442,
    SAMPLEREXTERNALOES = 443,
    SAMPLEREXTERNAL2DY2YEXT = 444,
    ISAMPLER1DARRAY = 445,
    USAMPLER1D = 446,
    USAMPLER1DARRAY = 447,
    F16SAMPLER1D = 448,
    F16SAMPLER2D = 449,
    F16SAMPLER3D = 450,
    F16SAMPLER2DRECT = 451,
    F16SAMPLERCUBE = 452,
    F16SAMPLER1DARRAY = 453,
    F16SAMPLER2DARRAY = 454,
    F16SAMPLERCUBEARRAY = 455,
    F16SAMPLERBUFFER = 456,
    F16SAMPLER2DMS = 457,
    F16SAMPLER2DMSARRAY = 458,
    F16SAMPLER1DSHADOW = 459,
    F16SAMPLER2DSHADOW = 460,
    F16SAMPLER1DARRAYSHADOW = 461,
    F16SAMPLER2DARRAYSHADOW = 462,
    F16SAMPLER2DRECTSHADOW = 463,
    F16SAMPLERCUBESHADOW = 464,
    F16SAMPLERCUBEARRAYSHADOW = 465,
    IMAGE1D = 466,
    IIMAGE1D = 467,
    UIMAGE1D = 468,
    IMAGE2D = 469,
    IIMAGE2D = 470,
    UIMAGE2D = 471,
    IMAGE3D = 472,
    IIMAGE3D = 473,
    UIMAGE3D = 474,
    IMAGE2DRECT = 475,
    IIMAGE2DRECT = 476,
    UIMAGE2DRECT = 477,
    IMAGECUBE = 478,
    IIMAGECUBE = 479,
    UIMAGECUBE = 480,
    IMAGEBUFFER = 481,
    IIMAGEBUFFER = 482,
    UIMAGEBUFFER = 483,
    IMAGE1DARRAY = 484,
    IIMAGE1DARRAY = 485,
    UIMAGE1DARRAY = 486,
    IMAGE2DARRAY = 487,
    IIMAGE2DARRAY = 488,
    UIMAGE2DARRAY = 489,
    IMAGECUBEARRAY = 490,
    IIMAGECUBEARRAY = 491,
    UIMAGECUBEARRAY = 492,
    IMAGE2DMS = 493,
    IIMAGE2DMS = 494,
    UIMAGE2DMS = 495,
    IMAGE2DMSARRAY = 496,
    IIMAGE2DMSARRAY = 497,
    UIMAGE2DMSARRAY = 498,
    F16IMAGE1D = 499,
    F16IMAGE2D = 500,
    F16IMAGE3D = 501,
    F16IMAGE2DRECT = 502,
    F16IMAGECUBE = 503,
    F16IMAGE1DARRAY = 504,
    F16IMAGE2DARRAY = 505,
    F16IMAGECUBEARRAY = 506,
    F16IMAGEBUFFER = 507,
    F16IMAGE2DMS = 508,
    F16IMAGE2DMSARRAY = 509,
    TEXTURECUBEARRAY = 510,
    ITEXTURECUBEARRAY = 511,
    UTEXTURECUBEARRAY = 512,
    TEXTURE1D = 513,
    ITEXTURE1D = 514,
    UTEXTURE1D = 515,
    TEXTURE1DARRAY = 516,
    ITEXTURE1DARRAY = 517,
    UTEXTURE1DARRAY = 518,
    TEXTURE2DRECT = 519,
    ITEXTURE2DRECT = 520,
    UTEXTURE2DRECT = 521,
    TEXTUREBUFFER = 522,
    ITEXTUREBUFFER = 523,
    UTEXTUREBUFFER = 524,
    TEXTURE2DMS = 525,
    ITEXTURE2DMS = 526,
    UTEXTURE2DMS = 527,
    TEXTURE2DMSARRAY = 528,
    ITEXTURE2DMSARRAY = 529,
    UTEXTURE2DMSARRAY = 530,
    F16TEXTURE1D = 531,
    F16TEXTURE2D = 532,
    F16TEXTURE3D = 533,
    F16TEXTURE2DRECT = 534,
    F16TEXTURECUBE = 535,
    F16TEXTURE1DARRAY = 536,
    F16TEXTURE2DARRAY = 537,
    F16TEXTURECUBEARRAY = 538,
    F16TEXTUREBUFFER = 539,
    F16TEXTURE2DMS = 540,
    F16TEXTURE2DMSARRAY = 541,
    SUBPASSINPUT = 542,
    SUBPASSINPUTMS = 543,
    ISUBPASSINPUT = 544,
    ISUBPASSINPUTMS = 545,
    USUBPASSINPUT = 546,
    USUBPASSINPUTMS = 547,
    F16SUBPASSINPUT = 548,
    F16SUBPASSINPUTMS = 549,
    LEFT_OP = 550,
    RIGHT_OP = 551,
    INC_OP = 552,
    DEC_OP = 553,
    LE_OP = 554,
    GE_OP = 555,
    EQ_OP = 556,
    NE_OP = 557,
    AND_OP = 558,
    OR_OP = 559,
    XOR_OP = 560,
    MUL_ASSIGN = 561,
    DIV_ASSIGN = 562,
    ADD_ASSIGN = 563,
    MOD_ASSIGN = 564,
    LEFT_ASSIGN = 565,
    RIGHT_ASSIGN = 566,
    AND_ASSIGN = 567,
    XOR_ASSIGN = 568,
    OR_ASSIGN = 569,
    SUB_ASSIGN = 570,
    STRING_LITERAL = 571,
    LEFT_PAREN = 572,
    RIGHT_PAREN = 573,
    LEFT_BRACKET = 574,
    RIGHT_BRACKET = 575,
    LEFT_BRACE = 576,
    RIGHT_BRACE = 577,
    DOT = 578,
    COMMA = 579,
    COLON = 580,
    EQUAL = 581,
    SEMICOLON = 582,
    BANG = 583,
    DASH = 584,
    TILDE = 585,
    PLUS = 586,
    STAR = 587,
    SLASH = 588,
    PERCENT = 589,
    LEFT_ANGLE = 590,
    RIGHT_ANGLE = 591,
    VERTICAL_BAR = 592,
    CARET = 593,
    AMPERSAND = 594,
    QUESTION = 595,
    INVARIANT = 596,
    HIGH_PRECISION = 597,
    MEDIUM_PRECISION = 598,
    LOW_PRECISION = 599,
    PRECISION = 600,
    PACKED = 601,
    RESOURCE = 602,
    SUPERP = 603,
    FLOATCONSTANT = 604,
    INTCONSTANT = 605,
    UINTCONSTANT = 606,
    BOOLCONSTANT = 607,
    IDENTIFIER = 608,
    TYPE_NAME = 609,
    CENTROID = 610,
    IN = 611,
    OUT = 612,
    INOUT = 613,
    STRUCT = 614,
    VOID = 615,
    WHILE = 616,
    BREAK = 617,
    CONTINUE = 618,
    DO = 619,
    ELSE = 620,
    FOR = 621,
    IF = 622,
    DISCARD = 623,
    RETURN = 624,
    SWITCH = 625,
    CASE = 626,
    DEFAULT = 627,
    UNIFORM = 628,
    SHARED = 629,
    BUFFER = 630,
    FLAT = 631,
    SMOOTH = 632,
    LAYOUT = 633,
    DOUBLECONSTANT = 634,
    INT16CONSTANT = 635,
    UINT16CONSTANT = 636,
    FLOAT16CONSTANT = 637,
    INT32CONSTANT = 638,
    UINT32CONSTANT = 639,
    INT64CONSTANT = 640,
    UINT64CONSTANT = 641,
    SUBROUTINE = 642,
    DEMOTE = 643,
    PAYLOADNV = 644,
    PAYLOADINNV = 645,
    HITATTRNV = 646,
    CALLDATANV = 647,
    CALLDATAINNV = 648,
    PAYLOADEXT = 649,
    PAYLOADINEXT = 650,
    HITATTREXT = 651,
    CALLDATAEXT = 652,
    CALLDATAINEXT = 653,
    PATCH = 654,
    SAMPLE = 655,
    NONUNIFORM = 656,
    COHERENT = 657,
    VOLATILE = 658,
    RESTRICT = 659,
    READONLY = 660,
    WRITEONLY = 661,
    DEVICECOHERENT = 662,
    QUEUEFAMILYCOHERENT = 663,
    WORKGROUPCOHERENT = 664,
    SUBGROUPCOHERENT = 665,
    NONPRIVATE = 666,
    SHADERCALLCOHERENT = 667,
    NOPERSPECTIVE = 668,
    EXPLICITINTERPAMD = 669,
    PERVERTEXNV = 670,
    PERPRIMITIVENV = 671,
    PERVIEWNV = 672,
    PERTASKNV = 673,
    PRECISE = 674
  };
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED

union YYSTYPE
{
#line 97 "MachineIndependent/glslang.y" /* yacc.c:1909  */

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

#line 510 "MachineIndependent/glslang_tab.cpp.h" /* yacc.c:1909  */
};

typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif



int yyparse (glslang::TParseContext* pParseContext);

#endif /* !YY_YY_MACHINEINDEPENDENT_GLSLANG_TAB_CPP_H_INCLUDED  */
