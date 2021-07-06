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

#ifndef YY_YY_GLSLANG_TAB_H_INCLUDED
#define YY_YY_GLSLANG_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
#    define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif
/* "%code requires" blocks.  */

#define YYLTYPE TSourceLoc
#define YYLTYPE_IS_DECLARED 1
#define YYLTYPE_IS_TRIVIAL 1

/* Token type.  */
#ifndef YYTOKENTYPE
#    define YYTOKENTYPE
enum yytokentype
{
    INVARIANT                 = 258,
    HIGH_PRECISION            = 259,
    MEDIUM_PRECISION          = 260,
    LOW_PRECISION             = 261,
    PRECISION                 = 262,
    ATTRIBUTE                 = 263,
    CONST_QUAL                = 264,
    BOOL_TYPE                 = 265,
    FLOAT_TYPE                = 266,
    INT_TYPE                  = 267,
    UINT_TYPE                 = 268,
    BREAK                     = 269,
    CONTINUE                  = 270,
    DO                        = 271,
    ELSE                      = 272,
    FOR                       = 273,
    IF                        = 274,
    DISCARD                   = 275,
    RETURN                    = 276,
    SWITCH                    = 277,
    CASE                      = 278,
    DEFAULT                   = 279,
    BVEC2                     = 280,
    BVEC3                     = 281,
    BVEC4                     = 282,
    IVEC2                     = 283,
    IVEC3                     = 284,
    IVEC4                     = 285,
    VEC2                      = 286,
    VEC3                      = 287,
    VEC4                      = 288,
    UVEC2                     = 289,
    UVEC3                     = 290,
    UVEC4                     = 291,
    MATRIX2                   = 292,
    MATRIX3                   = 293,
    MATRIX4                   = 294,
    IN_QUAL                   = 295,
    OUT_QUAL                  = 296,
    INOUT_QUAL                = 297,
    UNIFORM                   = 298,
    BUFFER                    = 299,
    VARYING                   = 300,
    MATRIX2x3                 = 301,
    MATRIX3x2                 = 302,
    MATRIX2x4                 = 303,
    MATRIX4x2                 = 304,
    MATRIX3x4                 = 305,
    MATRIX4x3                 = 306,
    CENTROID                  = 307,
    FLAT                      = 308,
    SMOOTH                    = 309,
    READONLY                  = 310,
    WRITEONLY                 = 311,
    COHERENT                  = 312,
    RESTRICT                  = 313,
    VOLATILE                  = 314,
    SHARED                    = 315,
    STRUCT                    = 316,
    VOID_TYPE                 = 317,
    WHILE                     = 318,
    SAMPLER2D                 = 319,
    SAMPLERCUBE               = 320,
    SAMPLER_EXTERNAL_OES      = 321,
    SAMPLER2DRECT             = 322,
    SAMPLER2DARRAY            = 323,
    ISAMPLER2D                = 324,
    ISAMPLER3D                = 325,
    ISAMPLERCUBE              = 326,
    ISAMPLER2DARRAY           = 327,
    USAMPLER2D                = 328,
    USAMPLER3D                = 329,
    USAMPLERCUBE              = 330,
    USAMPLER2DARRAY           = 331,
    SAMPLER2DMS               = 332,
    ISAMPLER2DMS              = 333,
    USAMPLER2DMS              = 334,
    SAMPLER2DMSARRAY          = 335,
    ISAMPLER2DMSARRAY         = 336,
    USAMPLER2DMSARRAY         = 337,
    SAMPLER3D                 = 338,
    SAMPLER3DRECT             = 339,
    SAMPLER2DSHADOW           = 340,
    SAMPLERCUBESHADOW         = 341,
    SAMPLER2DARRAYSHADOW      = 342,
    SAMPLEREXTERNAL2DY2YEXT   = 343,
    IMAGE2D                   = 344,
    IIMAGE2D                  = 345,
    UIMAGE2D                  = 346,
    IMAGE3D                   = 347,
    IIMAGE3D                  = 348,
    UIMAGE3D                  = 349,
    IMAGE2DARRAY              = 350,
    IIMAGE2DARRAY             = 351,
    UIMAGE2DARRAY             = 352,
    IMAGECUBE                 = 353,
    IIMAGECUBE                = 354,
    UIMAGECUBE                = 355,
    ATOMICUINT                = 356,
    LAYOUT                    = 357,
    YUVCSCSTANDARDEXT         = 358,
    YUVCSCSTANDARDEXTCONSTANT = 359,
    IDENTIFIER                = 360,
    TYPE_NAME                 = 361,
    FLOATCONSTANT             = 362,
    INTCONSTANT               = 363,
    UINTCONSTANT              = 364,
    BOOLCONSTANT              = 365,
    FIELD_SELECTION           = 366,
    LEFT_OP                   = 367,
    RIGHT_OP                  = 368,
    INC_OP                    = 369,
    DEC_OP                    = 370,
    LE_OP                     = 371,
    GE_OP                     = 372,
    EQ_OP                     = 373,
    NE_OP                     = 374,
    AND_OP                    = 375,
    OR_OP                     = 376,
    XOR_OP                    = 377,
    MUL_ASSIGN                = 378,
    DIV_ASSIGN                = 379,
    ADD_ASSIGN                = 380,
    MOD_ASSIGN                = 381,
    LEFT_ASSIGN               = 382,
    RIGHT_ASSIGN              = 383,
    AND_ASSIGN                = 384,
    XOR_ASSIGN                = 385,
    OR_ASSIGN                 = 386,
    SUB_ASSIGN                = 387,
    LEFT_PAREN                = 388,
    RIGHT_PAREN               = 389,
    LEFT_BRACKET              = 390,
    RIGHT_BRACKET             = 391,
    LEFT_BRACE                = 392,
    RIGHT_BRACE               = 393,
    DOT                       = 394,
    COMMA                     = 395,
    COLON                     = 396,
    EQUAL                     = 397,
    SEMICOLON                 = 398,
    BANG                      = 399,
    DASH                      = 400,
    TILDE                     = 401,
    PLUS                      = 402,
    STAR                      = 403,
    SLASH                     = 404,
    PERCENT                   = 405,
    LEFT_ANGLE                = 406,
    RIGHT_ANGLE               = 407,
    VERTICAL_BAR              = 408,
    CARET                     = 409,
    AMPERSAND                 = 410,
    QUESTION                  = 411
};
#endif

/* Value type.  */
#if !defined YYSTYPE && !defined YYSTYPE_IS_DECLARED

union YYSTYPE
{

    struct
    {
        union
        {
            const char *string;  // pool allocated.
            float f;
            int i;
            unsigned int u;
            bool b;
        };
        const TSymbol *symbol;
    } lex;
    struct
    {
        TOperator op;
        union
        {
            TIntermNode *intermNode;
            TIntermNodePair nodePair;
            TIntermTyped *intermTypedNode;
            TIntermAggregate *intermAggregate;
            TIntermBlock *intermBlock;
            TIntermDeclaration *intermDeclaration;
            TIntermFunctionPrototype *intermFunctionPrototype;
            TIntermSwitch *intermSwitch;
            TIntermCase *intermCase;
        };
        union
        {
            TVector<unsigned int> *arraySizes;
            TTypeSpecifierNonArray typeSpecifierNonArray;
            TPublicType type;
            TPrecision precision;
            TLayoutQualifier layoutQualifier;
            TQualifier qualifier;
            TFunction *function;
            TFunctionLookup *functionLookup;
            TParameter param;
            TDeclarator *declarator;
            TDeclaratorList *declaratorList;
            TFieldList *fieldList;
            TQualifierWrapperBase *qualifierWrapper;
            TTypeQualifierBuilder *typeQualifierBuilder;
        };
    } interm;
};

typedef union YYSTYPE YYSTYPE;
#    define YYSTYPE_IS_TRIVIAL 1
#    define YYSTYPE_IS_DECLARED 1
#endif

/* Location type.  */
#if !defined YYLTYPE && !defined YYLTYPE_IS_DECLARED
typedef struct YYLTYPE YYLTYPE;
struct YYLTYPE
{
    int first_line;
    int first_column;
    int last_line;
    int last_column;
};
#    define YYLTYPE_IS_DECLARED 1
#    define YYLTYPE_IS_TRIVIAL 1
#endif

int yyparse(TParseContext *context, void *scanner);

#endif /* !YY_YY_GLSLANG_TAB_H_INCLUDED  */
