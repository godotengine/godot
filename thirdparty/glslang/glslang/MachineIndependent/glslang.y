//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2012-2013 LunarG, Inc.
// Copyright (C) 2017 ARM Limited.
// Copyright (C) 2015-2019 Google, Inc.
// Modifications Copyright (C) 2020 Advanced Micro Devices, Inc. All rights reserved.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

//
// Do not edit the .y file, only edit the .m4 file.
// The .y bison file is not a source file, it is a derivative of the .m4 file.
// The m4 file needs to be processed by m4 to generate the .y bison file.
//
// Code sandwiched between a pair:
//
//    GLSLANG_WEB_EXCLUDE_ON
//      ...
//      ...
//      ...
//    GLSLANG_WEB_EXCLUDE_OFF
//
// Will be excluded from the grammar when m4 is executed as:
//
//    m4 -P -DGLSLANG_WEB
//
// It will be included when m4 is executed as:
//
//    m4 -P
//




/**
 * This is bison grammar and productions for parsing all versions of the
 * GLSL shading languages.
 */
%{

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

%}

%define parse.error verbose

%union {
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
}

%{

/* windows only pragma */
#ifdef _MSC_VER
    #pragma warning(disable : 4065)
    #pragma warning(disable : 4127)
    #pragma warning(disable : 4244)
#endif

#define parseContext (*pParseContext)
#define yyerror(context, msg) context->parserError(msg)

extern int yylex(YYSTYPE*, TParseContext&);

%}

%parse-param {glslang::TParseContext* pParseContext}
%lex-param {parseContext}
%pure-parser  // enable thread safety
%expect 1     // One shift reduce conflict because of if | else

%token <lex> CONST BOOL INT UINT FLOAT
%token <lex> BVEC2 BVEC3 BVEC4
%token <lex> IVEC2 IVEC3 IVEC4
%token <lex> UVEC2 UVEC3 UVEC4
%token <lex> VEC2 VEC3 VEC4
%token <lex> MAT2 MAT3 MAT4
%token <lex> MAT2X2 MAT2X3 MAT2X4
%token <lex> MAT3X2 MAT3X3 MAT3X4
%token <lex> MAT4X2 MAT4X3 MAT4X4

// combined image/sampler
%token <lex> SAMPLER2D SAMPLER3D SAMPLERCUBE SAMPLER2DSHADOW
%token <lex> SAMPLERCUBESHADOW SAMPLER2DARRAY
%token <lex> SAMPLER2DARRAYSHADOW ISAMPLER2D ISAMPLER3D ISAMPLERCUBE
%token <lex> ISAMPLER2DARRAY USAMPLER2D USAMPLER3D
%token <lex> USAMPLERCUBE USAMPLER2DARRAY

// separate image/sampler
%token <lex> SAMPLER SAMPLERSHADOW
%token <lex>  TEXTURE2D  TEXTURE3D  TEXTURECUBE  TEXTURE2DARRAY
%token <lex> ITEXTURE2D ITEXTURE3D ITEXTURECUBE ITEXTURE2DARRAY
%token <lex> UTEXTURE2D UTEXTURE3D UTEXTURECUBE UTEXTURE2DARRAY



%token <lex> ATTRIBUTE VARYING
%token <lex> FLOAT16_T FLOAT32_T DOUBLE FLOAT64_T
%token <lex> INT64_T UINT64_T INT32_T UINT32_T INT16_T UINT16_T INT8_T UINT8_T
%token <lex> I64VEC2 I64VEC3 I64VEC4
%token <lex> U64VEC2 U64VEC3 U64VEC4
%token <lex> I32VEC2 I32VEC3 I32VEC4
%token <lex> U32VEC2 U32VEC3 U32VEC4
%token <lex> I16VEC2 I16VEC3 I16VEC4
%token <lex> U16VEC2 U16VEC3 U16VEC4
%token <lex> I8VEC2  I8VEC3  I8VEC4
%token <lex> U8VEC2  U8VEC3  U8VEC4
%token <lex> DVEC2 DVEC3 DVEC4 DMAT2 DMAT3 DMAT4
%token <lex> F16VEC2 F16VEC3 F16VEC4 F16MAT2 F16MAT3 F16MAT4
%token <lex> F32VEC2 F32VEC3 F32VEC4 F32MAT2 F32MAT3 F32MAT4
%token <lex> F64VEC2 F64VEC3 F64VEC4 F64MAT2 F64MAT3 F64MAT4
%token <lex> DMAT2X2 DMAT2X3 DMAT2X4
%token <lex> DMAT3X2 DMAT3X3 DMAT3X4
%token <lex> DMAT4X2 DMAT4X3 DMAT4X4
%token <lex> F16MAT2X2 F16MAT2X3 F16MAT2X4
%token <lex> F16MAT3X2 F16MAT3X3 F16MAT3X4
%token <lex> F16MAT4X2 F16MAT4X3 F16MAT4X4
%token <lex> F32MAT2X2 F32MAT2X3 F32MAT2X4
%token <lex> F32MAT3X2 F32MAT3X3 F32MAT3X4
%token <lex> F32MAT4X2 F32MAT4X3 F32MAT4X4
%token <lex> F64MAT2X2 F64MAT2X3 F64MAT2X4
%token <lex> F64MAT3X2 F64MAT3X3 F64MAT3X4
%token <lex> F64MAT4X2 F64MAT4X3 F64MAT4X4
%token <lex> ATOMIC_UINT
%token <lex> ACCSTRUCTNV
%token <lex> ACCSTRUCTEXT
%token <lex> RAYQUERYEXT
%token <lex> FCOOPMATNV ICOOPMATNV UCOOPMATNV

// combined image/sampler
%token <lex> SAMPLERCUBEARRAY SAMPLERCUBEARRAYSHADOW
%token <lex> ISAMPLERCUBEARRAY USAMPLERCUBEARRAY
%token <lex> SAMPLER1D SAMPLER1DARRAY SAMPLER1DARRAYSHADOW ISAMPLER1D SAMPLER1DSHADOW
%token <lex> SAMPLER2DRECT SAMPLER2DRECTSHADOW ISAMPLER2DRECT USAMPLER2DRECT
%token <lex> SAMPLERBUFFER ISAMPLERBUFFER USAMPLERBUFFER
%token <lex> SAMPLER2DMS ISAMPLER2DMS USAMPLER2DMS
%token <lex> SAMPLER2DMSARRAY ISAMPLER2DMSARRAY USAMPLER2DMSARRAY
%token <lex> SAMPLEREXTERNALOES
%token <lex> SAMPLEREXTERNAL2DY2YEXT
%token <lex> ISAMPLER1DARRAY USAMPLER1D USAMPLER1DARRAY 
%token <lex> F16SAMPLER1D F16SAMPLER2D F16SAMPLER3D F16SAMPLER2DRECT F16SAMPLERCUBE
%token <lex> F16SAMPLER1DARRAY F16SAMPLER2DARRAY F16SAMPLERCUBEARRAY
%token <lex> F16SAMPLERBUFFER F16SAMPLER2DMS F16SAMPLER2DMSARRAY
%token <lex> F16SAMPLER1DSHADOW F16SAMPLER2DSHADOW F16SAMPLER1DARRAYSHADOW F16SAMPLER2DARRAYSHADOW
%token <lex> F16SAMPLER2DRECTSHADOW F16SAMPLERCUBESHADOW F16SAMPLERCUBEARRAYSHADOW

// images
%token <lex> IMAGE1D IIMAGE1D UIMAGE1D IMAGE2D IIMAGE2D
%token <lex> UIMAGE2D IMAGE3D IIMAGE3D UIMAGE3D
%token <lex> IMAGE2DRECT IIMAGE2DRECT UIMAGE2DRECT
%token <lex> IMAGECUBE IIMAGECUBE UIMAGECUBE
%token <lex> IMAGEBUFFER IIMAGEBUFFER UIMAGEBUFFER
%token <lex> IMAGE1DARRAY IIMAGE1DARRAY UIMAGE1DARRAY
%token <lex> IMAGE2DARRAY IIMAGE2DARRAY UIMAGE2DARRAY
%token <lex> IMAGECUBEARRAY IIMAGECUBEARRAY UIMAGECUBEARRAY
%token <lex> IMAGE2DMS IIMAGE2DMS UIMAGE2DMS
%token <lex> IMAGE2DMSARRAY IIMAGE2DMSARRAY UIMAGE2DMSARRAY

%token <lex> F16IMAGE1D F16IMAGE2D F16IMAGE3D F16IMAGE2DRECT
%token <lex> F16IMAGECUBE F16IMAGE1DARRAY F16IMAGE2DARRAY F16IMAGECUBEARRAY
%token <lex> F16IMAGEBUFFER F16IMAGE2DMS F16IMAGE2DMSARRAY

%token <lex> I64IMAGE1D U64IMAGE1D
%token <lex> I64IMAGE2D U64IMAGE2D
%token <lex> I64IMAGE3D U64IMAGE3D
%token <lex> I64IMAGE2DRECT U64IMAGE2DRECT
%token <lex> I64IMAGECUBE U64IMAGECUBE
%token <lex> I64IMAGEBUFFER U64IMAGEBUFFER
%token <lex> I64IMAGE1DARRAY U64IMAGE1DARRAY
%token <lex> I64IMAGE2DARRAY U64IMAGE2DARRAY
%token <lex> I64IMAGECUBEARRAY U64IMAGECUBEARRAY
%token <lex> I64IMAGE2DMS U64IMAGE2DMS
%token <lex> I64IMAGE2DMSARRAY U64IMAGE2DMSARRAY

// texture without sampler
%token <lex> TEXTURECUBEARRAY ITEXTURECUBEARRAY UTEXTURECUBEARRAY
%token <lex> TEXTURE1D ITEXTURE1D UTEXTURE1D
%token <lex> TEXTURE1DARRAY ITEXTURE1DARRAY UTEXTURE1DARRAY
%token <lex> TEXTURE2DRECT ITEXTURE2DRECT UTEXTURE2DRECT
%token <lex> TEXTUREBUFFER ITEXTUREBUFFER UTEXTUREBUFFER
%token <lex> TEXTURE2DMS ITEXTURE2DMS UTEXTURE2DMS
%token <lex> TEXTURE2DMSARRAY ITEXTURE2DMSARRAY UTEXTURE2DMSARRAY

%token <lex> F16TEXTURE1D F16TEXTURE2D F16TEXTURE3D F16TEXTURE2DRECT F16TEXTURECUBE
%token <lex> F16TEXTURE1DARRAY F16TEXTURE2DARRAY F16TEXTURECUBEARRAY
%token <lex> F16TEXTUREBUFFER F16TEXTURE2DMS F16TEXTURE2DMSARRAY

// input attachments
%token <lex> SUBPASSINPUT SUBPASSINPUTMS ISUBPASSINPUT ISUBPASSINPUTMS USUBPASSINPUT USUBPASSINPUTMS
%token <lex> F16SUBPASSINPUT F16SUBPASSINPUTMS

// spirv intrinsics
%token <lex> SPIRV_INSTRUCTION SPIRV_EXECUTION_MODE SPIRV_EXECUTION_MODE_ID
%token <lex> SPIRV_DECORATE SPIRV_DECORATE_ID SPIRV_DECORATE_STRING
%token <lex> SPIRV_TYPE SPIRV_STORAGE_CLASS SPIRV_BY_REFERENCE SPIRV_LITERAL



%token <lex> LEFT_OP RIGHT_OP
%token <lex> INC_OP DEC_OP LE_OP GE_OP EQ_OP NE_OP
%token <lex> AND_OP OR_OP XOR_OP MUL_ASSIGN DIV_ASSIGN ADD_ASSIGN
%token <lex> MOD_ASSIGN LEFT_ASSIGN RIGHT_ASSIGN AND_ASSIGN XOR_ASSIGN OR_ASSIGN
%token <lex> SUB_ASSIGN
%token <lex> STRING_LITERAL

%token <lex> LEFT_PAREN RIGHT_PAREN LEFT_BRACKET RIGHT_BRACKET LEFT_BRACE RIGHT_BRACE DOT
%token <lex> COMMA COLON EQUAL SEMICOLON BANG DASH TILDE PLUS STAR SLASH PERCENT
%token <lex> LEFT_ANGLE RIGHT_ANGLE VERTICAL_BAR CARET AMPERSAND QUESTION

%token <lex> INVARIANT
%token <lex> HIGH_PRECISION MEDIUM_PRECISION LOW_PRECISION PRECISION
%token <lex> PACKED RESOURCE SUPERP

%token <lex> FLOATCONSTANT INTCONSTANT UINTCONSTANT BOOLCONSTANT
%token <lex> IDENTIFIER TYPE_NAME
%token <lex> CENTROID IN OUT INOUT
%token <lex> STRUCT VOID WHILE
%token <lex> BREAK CONTINUE DO ELSE FOR IF DISCARD RETURN SWITCH CASE DEFAULT
%token <lex> TERMINATE_INVOCATION
%token <lex> TERMINATE_RAY IGNORE_INTERSECTION
%token <lex> UNIFORM SHARED BUFFER
%token <lex> FLAT SMOOTH LAYOUT


%token <lex> DOUBLECONSTANT INT16CONSTANT UINT16CONSTANT FLOAT16CONSTANT INT32CONSTANT UINT32CONSTANT
%token <lex> INT64CONSTANT UINT64CONSTANT
%token <lex> SUBROUTINE DEMOTE
%token <lex> PAYLOADNV PAYLOADINNV HITATTRNV CALLDATANV CALLDATAINNV
%token <lex> PAYLOADEXT PAYLOADINEXT HITATTREXT CALLDATAEXT CALLDATAINEXT
%token <lex> PATCH SAMPLE NONUNIFORM
%token <lex> COHERENT VOLATILE RESTRICT READONLY WRITEONLY DEVICECOHERENT QUEUEFAMILYCOHERENT WORKGROUPCOHERENT
%token <lex> SUBGROUPCOHERENT NONPRIVATE SHADERCALLCOHERENT
%token <lex> NOPERSPECTIVE EXPLICITINTERPAMD PERVERTEXNV PERPRIMITIVENV PERVIEWNV PERTASKNV
%token <lex> PRECISE


%type <interm> assignment_operator unary_operator
%type <interm.intermTypedNode> variable_identifier primary_expression postfix_expression
%type <interm.intermTypedNode> expression integer_expression assignment_expression
%type <interm.intermTypedNode> unary_expression multiplicative_expression additive_expression
%type <interm.intermTypedNode> relational_expression equality_expression
%type <interm.intermTypedNode> conditional_expression constant_expression
%type <interm.intermTypedNode> logical_or_expression logical_xor_expression logical_and_expression
%type <interm.intermTypedNode> shift_expression and_expression exclusive_or_expression inclusive_or_expression
%type <interm.intermTypedNode> function_call initializer condition conditionopt

%type <interm.intermNode> translation_unit function_definition
%type <interm.intermNode> statement simple_statement
%type <interm.intermNode> statement_list switch_statement_list compound_statement
%type <interm.intermNode> declaration_statement selection_statement selection_statement_nonattributed expression_statement
%type <interm.intermNode> switch_statement switch_statement_nonattributed case_label
%type <interm.intermNode> declaration external_declaration
%type <interm.intermNode> for_init_statement compound_statement_no_new_scope
%type <interm.nodePair> selection_rest_statement for_rest_statement
%type <interm.intermNode> iteration_statement iteration_statement_nonattributed jump_statement statement_no_new_scope statement_scoped
%type <interm> single_declaration init_declarator_list

%type <interm> parameter_declaration parameter_declarator parameter_type_specifier

%type <interm> array_specifier
%type <interm.type> invariant_qualifier interpolation_qualifier storage_qualifier precision_qualifier
%type <interm.type> layout_qualifier layout_qualifier_id_list layout_qualifier_id

%type <interm.typeParameters> type_parameter_specifier
%type <interm.typeParameters> type_parameter_specifier_opt
%type <interm.typeParameters> type_parameter_specifier_list

%type <interm.type> type_qualifier fully_specified_type type_specifier
%type <interm.type> single_type_qualifier
%type <interm.type> type_specifier_nonarray
%type <interm.type> struct_specifier
%type <interm.typeLine> struct_declarator
%type <interm.typeList> struct_declarator_list struct_declaration struct_declaration_list
%type <interm> block_structure
%type <interm.function> function_header function_declarator
%type <interm.function> function_header_with_parameters
%type <interm> function_call_header_with_parameters function_call_header_no_parameters function_call_generic function_prototype
%type <interm> function_call_or_method function_identifier function_call_header

%type <interm.identifierList> identifier_list


%type <interm.type> precise_qualifier non_uniform_qualifier
%type <interm.typeList> type_name_list
%type <interm.attributes> attribute attribute_list single_attribute
%type <interm.intermNode> demote_statement
%type <interm.intermTypedNode> initializer_list
%type <interm.spirvReq> spirv_requirements_list spirv_requirements_parameter
%type <interm.intermNode> spirv_extension_list spirv_capability_list
%type <interm.intermNode> spirv_execution_mode_qualifier
%type <interm.intermNode> spirv_execution_mode_parameter_list spirv_execution_mode_parameter spirv_execution_mode_id_parameter_list
%type <interm.type> spirv_storage_class_qualifier
%type <interm.type> spirv_decorate_qualifier
%type <interm.intermNode> spirv_decorate_parameter_list spirv_decorate_parameter
%type <interm.intermNode> spirv_decorate_id_parameter_list
%type <interm.intermNode> spirv_decorate_string_parameter_list
%type <interm.type> spirv_type_specifier
%type <interm.spirvTypeParams> spirv_type_parameter_list spirv_type_parameter
%type <interm.spirvInst> spirv_instruction_qualifier
%type <interm.spirvInst> spirv_instruction_qualifier_list spirv_instruction_qualifier_id


%start translation_unit
%%

variable_identifier
    : IDENTIFIER {
        $$ = parseContext.handleVariable($1.loc, $1.symbol, $1.string);
    }
    ;

primary_expression
    : variable_identifier {
        $$ = $1;
    }
    | LEFT_PAREN expression RIGHT_PAREN {
        $$ = $2;
        if ($$->getAsConstantUnion())
            $$->getAsConstantUnion()->setExpression();
    }
    | FLOATCONSTANT {
        $$ = parseContext.intermediate.addConstantUnion($1.d, EbtFloat, $1.loc, true);
    }
    | INTCONSTANT {
        $$ = parseContext.intermediate.addConstantUnion($1.i, $1.loc, true);
    }
    | UINTCONSTANT {
        parseContext.fullIntegerCheck($1.loc, "unsigned literal");
        $$ = parseContext.intermediate.addConstantUnion($1.u, $1.loc, true);
    }
    | BOOLCONSTANT {
        $$ = parseContext.intermediate.addConstantUnion($1.b, $1.loc, true);
    }

    | STRING_LITERAL {
        $$ = parseContext.intermediate.addConstantUnion($1.string, $1.loc, true);
    }
    | INT32CONSTANT {
        parseContext.explicitInt32Check($1.loc, "32-bit signed literal");
        $$ = parseContext.intermediate.addConstantUnion($1.i, $1.loc, true);
    }
    | UINT32CONSTANT {
        parseContext.explicitInt32Check($1.loc, "32-bit signed literal");
        $$ = parseContext.intermediate.addConstantUnion($1.u, $1.loc, true);
    }
    | INT64CONSTANT {
        parseContext.int64Check($1.loc, "64-bit integer literal");
        $$ = parseContext.intermediate.addConstantUnion($1.i64, $1.loc, true);
    }
    | UINT64CONSTANT {
        parseContext.int64Check($1.loc, "64-bit unsigned integer literal");
        $$ = parseContext.intermediate.addConstantUnion($1.u64, $1.loc, true);
    }
    | INT16CONSTANT {
        parseContext.explicitInt16Check($1.loc, "16-bit integer literal");
        $$ = parseContext.intermediate.addConstantUnion((short)$1.i, $1.loc, true);
    }
    | UINT16CONSTANT {
        parseContext.explicitInt16Check($1.loc, "16-bit unsigned integer literal");
        $$ = parseContext.intermediate.addConstantUnion((unsigned short)$1.u, $1.loc, true);
    }
    | DOUBLECONSTANT {
        parseContext.requireProfile($1.loc, ECoreProfile | ECompatibilityProfile, "double literal");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck($1.loc, "double literal");
        $$ = parseContext.intermediate.addConstantUnion($1.d, EbtDouble, $1.loc, true);
    }
    | FLOAT16CONSTANT {
        parseContext.float16Check($1.loc, "half float literal");
        $$ = parseContext.intermediate.addConstantUnion($1.d, EbtFloat16, $1.loc, true);
    }

    ;

postfix_expression
    : primary_expression {
        $$ = $1;
    }
    | postfix_expression LEFT_BRACKET integer_expression RIGHT_BRACKET {
        $$ = parseContext.handleBracketDereference($2.loc, $1, $3);
    }
    | function_call {
        $$ = $1;
    }
    | postfix_expression DOT IDENTIFIER {
        $$ = parseContext.handleDotDereference($3.loc, $1, *$3.string);
    }
    | postfix_expression INC_OP {
        parseContext.variableCheck($1);
        parseContext.lValueErrorCheck($2.loc, "++", $1);
        $$ = parseContext.handleUnaryMath($2.loc, "++", EOpPostIncrement, $1);
    }
    | postfix_expression DEC_OP {
        parseContext.variableCheck($1);
        parseContext.lValueErrorCheck($2.loc, "--", $1);
        $$ = parseContext.handleUnaryMath($2.loc, "--", EOpPostDecrement, $1);
    }
    ;

integer_expression
    : expression {
        parseContext.integerCheck($1, "[]");
        $$ = $1;
    }
    ;

function_call
    : function_call_or_method {
        $$ = parseContext.handleFunctionCall($1.loc, $1.function, $1.intermNode);
        delete $1.function;
    }
    ;

function_call_or_method
    : function_call_generic {
        $$ = $1;
    }
    ;

function_call_generic
    : function_call_header_with_parameters RIGHT_PAREN {
        $$ = $1;
        $$.loc = $2.loc;
    }
    | function_call_header_no_parameters RIGHT_PAREN {
        $$ = $1;
        $$.loc = $2.loc;
    }
    ;

function_call_header_no_parameters
    : function_call_header VOID {
        $$ = $1;
    }
    | function_call_header {
        $$ = $1;
    }
    ;

function_call_header_with_parameters
    : function_call_header assignment_expression {
        TParameter param = { 0, new TType };
        param.type->shallowCopy($2->getType());
        $1.function->addParameter(param);
        $$.function = $1.function;
        $$.intermNode = $2;
    }
    | function_call_header_with_parameters COMMA assignment_expression {
        TParameter param = { 0, new TType };
        param.type->shallowCopy($3->getType());
        $1.function->addParameter(param);
        $$.function = $1.function;
        $$.intermNode = parseContext.intermediate.growAggregate($1.intermNode, $3, $2.loc);
    }
    ;

function_call_header
    : function_identifier LEFT_PAREN {
        $$ = $1;
    }
    ;

// Grammar Note:  Constructors look like functions, but are recognized as types.

function_identifier
    : type_specifier {
        // Constructor
        $$.intermNode = 0;
        $$.function = parseContext.handleConstructorCall($1.loc, $1);
    }
    | postfix_expression {
        //
        // Should be a method or subroutine call, but we haven't recognized the arguments yet.
        //
        $$.function = 0;
        $$.intermNode = 0;

        TIntermMethod* method = $1->getAsMethodNode();
        if (method) {
            $$.function = new TFunction(&method->getMethodName(), TType(EbtInt), EOpArrayLength);
            $$.intermNode = method->getObject();
        } else {
            TIntermSymbol* symbol = $1->getAsSymbolNode();
            if (symbol) {
                parseContext.reservedErrorCheck(symbol->getLoc(), symbol->getName());
                TFunction *function = new TFunction(&symbol->getName(), TType(EbtVoid));
                $$.function = function;
            } else
                parseContext.error($1->getLoc(), "function call, method, or subroutine call expected", "", "");
        }

        if ($$.function == 0) {
            // error recover
            TString* empty = NewPoolTString("");
            $$.function = new TFunction(empty, TType(EbtVoid), EOpNull);
        }
    }

    | non_uniform_qualifier {
        // Constructor
        $$.intermNode = 0;
        $$.function = parseContext.handleConstructorCall($1.loc, $1);
    }

    ;

unary_expression
    : postfix_expression {
        parseContext.variableCheck($1);
        $$ = $1;
        if (TIntermMethod* method = $1->getAsMethodNode())
            parseContext.error($1->getLoc(), "incomplete method syntax", method->getMethodName().c_str(), "");
    }
    | INC_OP unary_expression {
        parseContext.lValueErrorCheck($1.loc, "++", $2);
        $$ = parseContext.handleUnaryMath($1.loc, "++", EOpPreIncrement, $2);
    }
    | DEC_OP unary_expression {
        parseContext.lValueErrorCheck($1.loc, "--", $2);
        $$ = parseContext.handleUnaryMath($1.loc, "--", EOpPreDecrement, $2);
    }
    | unary_operator unary_expression {
        if ($1.op != EOpNull) {
            char errorOp[2] = {0, 0};
            switch($1.op) {
            case EOpNegative:   errorOp[0] = '-'; break;
            case EOpLogicalNot: errorOp[0] = '!'; break;
            case EOpBitwiseNot: errorOp[0] = '~'; break;
            default: break; // some compilers want this
            }
            $$ = parseContext.handleUnaryMath($1.loc, errorOp, $1.op, $2);
        } else {
            $$ = $2;
            if ($$->getAsConstantUnion())
                $$->getAsConstantUnion()->setExpression();
        }
    }
    ;
// Grammar Note:  No traditional style type casts.

unary_operator
    : PLUS  { $$.loc = $1.loc; $$.op = EOpNull; }
    | DASH  { $$.loc = $1.loc; $$.op = EOpNegative; }
    | BANG  { $$.loc = $1.loc; $$.op = EOpLogicalNot; }
    | TILDE { $$.loc = $1.loc; $$.op = EOpBitwiseNot;
              parseContext.fullIntegerCheck($1.loc, "bitwise not"); }
    ;
// Grammar Note:  No '*' or '&' unary ops.  Pointers are not supported.

multiplicative_expression
    : unary_expression { $$ = $1; }
    | multiplicative_expression STAR unary_expression {
        $$ = parseContext.handleBinaryMath($2.loc, "*", EOpMul, $1, $3);
        if ($$ == 0)
            $$ = $1;
    }
    | multiplicative_expression SLASH unary_expression {
        $$ = parseContext.handleBinaryMath($2.loc, "/", EOpDiv, $1, $3);
        if ($$ == 0)
            $$ = $1;
    }
    | multiplicative_expression PERCENT unary_expression {
        parseContext.fullIntegerCheck($2.loc, "%");
        $$ = parseContext.handleBinaryMath($2.loc, "%", EOpMod, $1, $3);
        if ($$ == 0)
            $$ = $1;
    }
    ;

additive_expression
    : multiplicative_expression { $$ = $1; }
    | additive_expression PLUS multiplicative_expression {
        $$ = parseContext.handleBinaryMath($2.loc, "+", EOpAdd, $1, $3);
        if ($$ == 0)
            $$ = $1;
    }
    | additive_expression DASH multiplicative_expression {
        $$ = parseContext.handleBinaryMath($2.loc, "-", EOpSub, $1, $3);
        if ($$ == 0)
            $$ = $1;
    }
    ;

shift_expression
    : additive_expression { $$ = $1; }
    | shift_expression LEFT_OP additive_expression {
        parseContext.fullIntegerCheck($2.loc, "bit shift left");
        $$ = parseContext.handleBinaryMath($2.loc, "<<", EOpLeftShift, $1, $3);
        if ($$ == 0)
            $$ = $1;
    }
    | shift_expression RIGHT_OP additive_expression {
        parseContext.fullIntegerCheck($2.loc, "bit shift right");
        $$ = parseContext.handleBinaryMath($2.loc, ">>", EOpRightShift, $1, $3);
        if ($$ == 0)
            $$ = $1;
    }
    ;

relational_expression
    : shift_expression { $$ = $1; }
    | relational_expression LEFT_ANGLE shift_expression {
        $$ = parseContext.handleBinaryMath($2.loc, "<", EOpLessThan, $1, $3);
        if ($$ == 0)
            $$ = parseContext.intermediate.addConstantUnion(false, $2.loc);
    }
    | relational_expression RIGHT_ANGLE shift_expression  {
        $$ = parseContext.handleBinaryMath($2.loc, ">", EOpGreaterThan, $1, $3);
        if ($$ == 0)
            $$ = parseContext.intermediate.addConstantUnion(false, $2.loc);
    }
    | relational_expression LE_OP shift_expression  {
        $$ = parseContext.handleBinaryMath($2.loc, "<=", EOpLessThanEqual, $1, $3);
        if ($$ == 0)
            $$ = parseContext.intermediate.addConstantUnion(false, $2.loc);
    }
    | relational_expression GE_OP shift_expression  {
        $$ = parseContext.handleBinaryMath($2.loc, ">=", EOpGreaterThanEqual, $1, $3);
        if ($$ == 0)
            $$ = parseContext.intermediate.addConstantUnion(false, $2.loc);
    }
    ;

equality_expression
    : relational_expression { $$ = $1; }
    | equality_expression EQ_OP relational_expression  {
        parseContext.arrayObjectCheck($2.loc, $1->getType(), "array comparison");
        parseContext.opaqueCheck($2.loc, $1->getType(), "==");
        parseContext.specializationCheck($2.loc, $1->getType(), "==");
        parseContext.referenceCheck($2.loc, $1->getType(), "==");
        $$ = parseContext.handleBinaryMath($2.loc, "==", EOpEqual, $1, $3);
        if ($$ == 0)
            $$ = parseContext.intermediate.addConstantUnion(false, $2.loc);
    }
    | equality_expression NE_OP relational_expression {
        parseContext.arrayObjectCheck($2.loc, $1->getType(), "array comparison");
        parseContext.opaqueCheck($2.loc, $1->getType(), "!=");
        parseContext.specializationCheck($2.loc, $1->getType(), "!=");
        parseContext.referenceCheck($2.loc, $1->getType(), "!=");
        $$ = parseContext.handleBinaryMath($2.loc, "!=", EOpNotEqual, $1, $3);
        if ($$ == 0)
            $$ = parseContext.intermediate.addConstantUnion(false, $2.loc);
    }
    ;

and_expression
    : equality_expression { $$ = $1; }
    | and_expression AMPERSAND equality_expression {
        parseContext.fullIntegerCheck($2.loc, "bitwise and");
        $$ = parseContext.handleBinaryMath($2.loc, "&", EOpAnd, $1, $3);
        if ($$ == 0)
            $$ = $1;
    }
    ;

exclusive_or_expression
    : and_expression { $$ = $1; }
    | exclusive_or_expression CARET and_expression {
        parseContext.fullIntegerCheck($2.loc, "bitwise exclusive or");
        $$ = parseContext.handleBinaryMath($2.loc, "^", EOpExclusiveOr, $1, $3);
        if ($$ == 0)
            $$ = $1;
    }
    ;

inclusive_or_expression
    : exclusive_or_expression { $$ = $1; }
    | inclusive_or_expression VERTICAL_BAR exclusive_or_expression {
        parseContext.fullIntegerCheck($2.loc, "bitwise inclusive or");
        $$ = parseContext.handleBinaryMath($2.loc, "|", EOpInclusiveOr, $1, $3);
        if ($$ == 0)
            $$ = $1;
    }
    ;

logical_and_expression
    : inclusive_or_expression { $$ = $1; }
    | logical_and_expression AND_OP inclusive_or_expression {
        $$ = parseContext.handleBinaryMath($2.loc, "&&", EOpLogicalAnd, $1, $3);
        if ($$ == 0)
            $$ = parseContext.intermediate.addConstantUnion(false, $2.loc);
    }
    ;

logical_xor_expression
    : logical_and_expression { $$ = $1; }
    | logical_xor_expression XOR_OP logical_and_expression  {
        $$ = parseContext.handleBinaryMath($2.loc, "^^", EOpLogicalXor, $1, $3);
        if ($$ == 0)
            $$ = parseContext.intermediate.addConstantUnion(false, $2.loc);
    }
    ;

logical_or_expression
    : logical_xor_expression { $$ = $1; }
    | logical_or_expression OR_OP logical_xor_expression  {
        $$ = parseContext.handleBinaryMath($2.loc, "||", EOpLogicalOr, $1, $3);
        if ($$ == 0)
            $$ = parseContext.intermediate.addConstantUnion(false, $2.loc);
    }
    ;

conditional_expression
    : logical_or_expression { $$ = $1; }
    | logical_or_expression QUESTION {
        ++parseContext.controlFlowNestingLevel;
    }
      expression COLON assignment_expression {
        --parseContext.controlFlowNestingLevel;
        parseContext.boolCheck($2.loc, $1);
        parseContext.rValueErrorCheck($2.loc, "?", $1);
        parseContext.rValueErrorCheck($5.loc, ":", $4);
        parseContext.rValueErrorCheck($5.loc, ":", $6);
        $$ = parseContext.intermediate.addSelection($1, $4, $6, $2.loc);
        if ($$ == 0) {
            parseContext.binaryOpError($2.loc, ":", $4->getCompleteString(), $6->getCompleteString());
            $$ = $6;
        }
    }
    ;

assignment_expression
    : conditional_expression { $$ = $1; }
    | unary_expression assignment_operator assignment_expression {
        parseContext.arrayObjectCheck($2.loc, $1->getType(), "array assignment");
        parseContext.opaqueCheck($2.loc, $1->getType(), "=");
        parseContext.storage16BitAssignmentCheck($2.loc, $1->getType(), "=");
        parseContext.specializationCheck($2.loc, $1->getType(), "=");
        parseContext.lValueErrorCheck($2.loc, "assign", $1);
        parseContext.rValueErrorCheck($2.loc, "assign", $3);
        $$ = parseContext.addAssign($2.loc, $2.op, $1, $3);
        if ($$ == 0) {
            parseContext.assignError($2.loc, "assign", $1->getCompleteString(), $3->getCompleteString());
            $$ = $1;
        }
    }
    ;

assignment_operator
    : EQUAL {
        $$.loc = $1.loc;
        $$.op = EOpAssign;
    }
    | MUL_ASSIGN {
        $$.loc = $1.loc;
        $$.op = EOpMulAssign;
    }
    | DIV_ASSIGN {
        $$.loc = $1.loc;
        $$.op = EOpDivAssign;
    }
    | MOD_ASSIGN {
        parseContext.fullIntegerCheck($1.loc, "%=");
        $$.loc = $1.loc;
        $$.op = EOpModAssign;
    }
    | ADD_ASSIGN {
        $$.loc = $1.loc;
        $$.op = EOpAddAssign;
    }
    | SUB_ASSIGN {
        $$.loc = $1.loc;
        $$.op = EOpSubAssign;
    }
    | LEFT_ASSIGN {
        parseContext.fullIntegerCheck($1.loc, "bit-shift left assign");
        $$.loc = $1.loc; $$.op = EOpLeftShiftAssign;
    }
    | RIGHT_ASSIGN {
        parseContext.fullIntegerCheck($1.loc, "bit-shift right assign");
        $$.loc = $1.loc; $$.op = EOpRightShiftAssign;
    }
    | AND_ASSIGN {
        parseContext.fullIntegerCheck($1.loc, "bitwise-and assign");
        $$.loc = $1.loc; $$.op = EOpAndAssign;
    }
    | XOR_ASSIGN {
        parseContext.fullIntegerCheck($1.loc, "bitwise-xor assign");
        $$.loc = $1.loc; $$.op = EOpExclusiveOrAssign;
    }
    | OR_ASSIGN {
        parseContext.fullIntegerCheck($1.loc, "bitwise-or assign");
        $$.loc = $1.loc; $$.op = EOpInclusiveOrAssign;
    }
    ;

expression
    : assignment_expression {
        $$ = $1;
    }
    | expression COMMA assignment_expression {
        parseContext.samplerConstructorLocationCheck($2.loc, ",", $3);
        $$ = parseContext.intermediate.addComma($1, $3, $2.loc);
        if ($$ == 0) {
            parseContext.binaryOpError($2.loc, ",", $1->getCompleteString(), $3->getCompleteString());
            $$ = $3;
        }
    }
    ;

constant_expression
    : conditional_expression {
        parseContext.constantValueCheck($1, "");
        $$ = $1;
    }
    ;

declaration
    : function_prototype SEMICOLON {
        parseContext.handleFunctionDeclarator($1.loc, *$1.function, true /* prototype */);
        $$ = 0;
        // TODO: 4.0 functionality: subroutines: make the identifier a user type for this signature
    }

    | spirv_instruction_qualifier function_prototype SEMICOLON {
        parseContext.requireExtensions($2.loc, 1, &E_GL_EXT_spirv_intrinsics, "SPIR-V instruction qualifier");
        $2.function->setSpirvInstruction(*$1); // Attach SPIR-V intruction qualifier
        parseContext.handleFunctionDeclarator($2.loc, *$2.function, true /* prototype */);
        $$ = 0;
        // TODO: 4.0 functionality: subroutines: make the identifier a user type for this signature
    }
    | spirv_execution_mode_qualifier SEMICOLON {
        parseContext.globalCheck($2.loc, "SPIR-V execution mode qualifier");
        parseContext.requireExtensions($2.loc, 1, &E_GL_EXT_spirv_intrinsics, "SPIR-V execution mode qualifier");
        $$ = 0;
    }

    | init_declarator_list SEMICOLON {
        if ($1.intermNode && $1.intermNode->getAsAggregate())
            $1.intermNode->getAsAggregate()->setOperator(EOpSequence);
        $$ = $1.intermNode;
    }
    | PRECISION precision_qualifier type_specifier SEMICOLON {
        parseContext.profileRequires($1.loc, ENoProfile, 130, 0, "precision statement");
        // lazy setting of the previous scope's defaults, has effect only the first time it is called in a particular scope
        parseContext.symbolTable.setPreviousDefaultPrecisions(&parseContext.defaultPrecision[0]);
        parseContext.setDefaultPrecision($1.loc, $3, $2.qualifier.precision);
        $$ = 0;
    }
    | block_structure SEMICOLON {
        parseContext.declareBlock($1.loc, *$1.typeList);
        $$ = 0;
    }
    | block_structure IDENTIFIER SEMICOLON {
        parseContext.declareBlock($1.loc, *$1.typeList, $2.string);
        $$ = 0;
    }
    | block_structure IDENTIFIER array_specifier SEMICOLON {
        parseContext.declareBlock($1.loc, *$1.typeList, $2.string, $3.arraySizes);
        $$ = 0;
    }
    | type_qualifier SEMICOLON {
        parseContext.globalQualifierFixCheck($1.loc, $1.qualifier);
        parseContext.updateStandaloneQualifierDefaults($1.loc, $1);
        $$ = 0;
    }
    | type_qualifier IDENTIFIER SEMICOLON {
        parseContext.checkNoShaderLayouts($1.loc, $1.shaderQualifiers);
        parseContext.addQualifierToExisting($1.loc, $1.qualifier, *$2.string);
        $$ = 0;
    }
    | type_qualifier IDENTIFIER identifier_list SEMICOLON {
        parseContext.checkNoShaderLayouts($1.loc, $1.shaderQualifiers);
        $3->push_back($2.string);
        parseContext.addQualifierToExisting($1.loc, $1.qualifier, *$3);
        $$ = 0;
    }
    ;

block_structure
    : type_qualifier IDENTIFIER LEFT_BRACE { parseContext.nestedBlockCheck($1.loc); } struct_declaration_list RIGHT_BRACE {
        --parseContext.blockNestingLevel;
        parseContext.blockName = $2.string;
        parseContext.globalQualifierFixCheck($1.loc, $1.qualifier);
        parseContext.checkNoShaderLayouts($1.loc, $1.shaderQualifiers);
        parseContext.currentBlockQualifier = $1.qualifier;
        $$.loc = $1.loc;
        $$.typeList = $5;
    }

identifier_list
    : COMMA IDENTIFIER {
        $$ = new TIdentifierList;
        $$->push_back($2.string);
    }
    | identifier_list COMMA IDENTIFIER {
        $$ = $1;
        $$->push_back($3.string);
    }
    ;

function_prototype
    : function_declarator RIGHT_PAREN  {
        $$.function = $1;
        $$.loc = $2.loc;
    }
    | function_declarator RIGHT_PAREN attribute {
        $$.function = $1;
        $$.loc = $2.loc;
        parseContext.requireExtensions($2.loc, 1, &E_GL_EXT_subgroup_uniform_control_flow, "attribute");
        parseContext.handleFunctionAttributes($2.loc, *$3, $$.function);
    }
    | attribute function_declarator RIGHT_PAREN {
        $$.function = $2;
        $$.loc = $3.loc;
        parseContext.requireExtensions($3.loc, 1, &E_GL_EXT_subgroup_uniform_control_flow, "attribute");
        parseContext.handleFunctionAttributes($3.loc, *$1, $$.function);
    }
    | attribute function_declarator RIGHT_PAREN attribute {
        $$.function = $2;
        $$.loc = $3.loc;
        parseContext.requireExtensions($3.loc, 1, &E_GL_EXT_subgroup_uniform_control_flow, "attribute");
        parseContext.handleFunctionAttributes($3.loc, *$1, $$.function);
        parseContext.handleFunctionAttributes($3.loc, *$4, $$.function);
    }
    ;

function_declarator
    : function_header {
        $$ = $1;
    }
    | function_header_with_parameters {
        $$ = $1;
    }
    ;


function_header_with_parameters
    : function_header parameter_declaration {
        // Add the parameter
        $$ = $1;
        if ($2.param.type->getBasicType() != EbtVoid)
            $1->addParameter($2.param);
        else
            delete $2.param.type;
    }
    | function_header_with_parameters COMMA parameter_declaration {
        //
        // Only first parameter of one-parameter functions can be void
        // The check for named parameters not being void is done in parameter_declarator
        //
        if ($3.param.type->getBasicType() == EbtVoid) {
            //
            // This parameter > first is void
            //
            parseContext.error($2.loc, "cannot be an argument type except for '(void)'", "void", "");
            delete $3.param.type;
        } else {
            // Add the parameter
            $$ = $1;
            $1->addParameter($3.param);
        }
    }
    ;

function_header
    : fully_specified_type IDENTIFIER LEFT_PAREN {
        if ($1.qualifier.storage != EvqGlobal && $1.qualifier.storage != EvqTemporary) {
            parseContext.error($2.loc, "no qualifiers allowed for function return",
                               GetStorageQualifierString($1.qualifier.storage), "");
        }
        if ($1.arraySizes)
            parseContext.arraySizeRequiredCheck($1.loc, *$1.arraySizes);

        // Add the function as a prototype after parsing it (we do not support recursion)
        TFunction *function;
        TType type($1);

        // Potentially rename shader entry point function.  No-op most of the time.
        parseContext.renameShaderFunction($2.string);

        // Make the function
        function = new TFunction($2.string, type);
        $$ = function;
    }
    ;

parameter_declarator
    // Type + name
    : type_specifier IDENTIFIER {
        if ($1.arraySizes) {
            parseContext.profileRequires($1.loc, ENoProfile, 120, E_GL_3DL_array_objects, "arrayed type");
            parseContext.profileRequires($1.loc, EEsProfile, 300, 0, "arrayed type");
            parseContext.arraySizeRequiredCheck($1.loc, *$1.arraySizes);
        }
        if ($1.basicType == EbtVoid) {
            parseContext.error($2.loc, "illegal use of type 'void'", $2.string->c_str(), "");
        }
        parseContext.reservedErrorCheck($2.loc, *$2.string);

        TParameter param = {$2.string, new TType($1)};
        $$.loc = $2.loc;
        $$.param = param;
    }
    | type_specifier IDENTIFIER array_specifier {
        if ($1.arraySizes) {
            parseContext.profileRequires($1.loc, ENoProfile, 120, E_GL_3DL_array_objects, "arrayed type");
            parseContext.profileRequires($1.loc, EEsProfile, 300, 0, "arrayed type");
            parseContext.arraySizeRequiredCheck($1.loc, *$1.arraySizes);
        }
        TType* type = new TType($1);
        type->transferArraySizes($3.arraySizes);
        type->copyArrayInnerSizes($1.arraySizes);

        parseContext.arrayOfArrayVersionCheck($2.loc, type->getArraySizes());
        parseContext.arraySizeRequiredCheck($3.loc, *$3.arraySizes);
        parseContext.reservedErrorCheck($2.loc, *$2.string);

        TParameter param = { $2.string, type };

        $$.loc = $2.loc;
        $$.param = param;
    }
    ;

parameter_declaration
    //
    // With name
    //
    : type_qualifier parameter_declarator {
        $$ = $2;
        if ($1.qualifier.precision != EpqNone)
            $$.param.type->getQualifier().precision = $1.qualifier.precision;
        parseContext.precisionQualifierCheck($$.loc, $$.param.type->getBasicType(), $$.param.type->getQualifier());

        parseContext.checkNoShaderLayouts($1.loc, $1.shaderQualifiers);
        parseContext.parameterTypeCheck($2.loc, $1.qualifier.storage, *$$.param.type);
        parseContext.paramCheckFix($1.loc, $1.qualifier, *$$.param.type);

    }
    | parameter_declarator {
        $$ = $1;

        parseContext.parameterTypeCheck($1.loc, EvqIn, *$1.param.type);
        parseContext.paramCheckFixStorage($1.loc, EvqTemporary, *$$.param.type);
        parseContext.precisionQualifierCheck($$.loc, $$.param.type->getBasicType(), $$.param.type->getQualifier());
    }
    //
    // Without name
    //
    | type_qualifier parameter_type_specifier {
        $$ = $2;
        if ($1.qualifier.precision != EpqNone)
            $$.param.type->getQualifier().precision = $1.qualifier.precision;
        parseContext.precisionQualifierCheck($1.loc, $$.param.type->getBasicType(), $$.param.type->getQualifier());

        parseContext.checkNoShaderLayouts($1.loc, $1.shaderQualifiers);
        parseContext.parameterTypeCheck($2.loc, $1.qualifier.storage, *$$.param.type);
        parseContext.paramCheckFix($1.loc, $1.qualifier, *$$.param.type);
    }
    | parameter_type_specifier {
        $$ = $1;

        parseContext.parameterTypeCheck($1.loc, EvqIn, *$1.param.type);
        parseContext.paramCheckFixStorage($1.loc, EvqTemporary, *$$.param.type);
        parseContext.precisionQualifierCheck($$.loc, $$.param.type->getBasicType(), $$.param.type->getQualifier());
    }
    ;

parameter_type_specifier
    : type_specifier {
        TParameter param = { 0, new TType($1) };
        $$.param = param;
        if ($1.arraySizes)
            parseContext.arraySizeRequiredCheck($1.loc, *$1.arraySizes);
    }
    ;

init_declarator_list
    : single_declaration {
        $$ = $1;
    }
    | init_declarator_list COMMA IDENTIFIER {
        $$ = $1;
        parseContext.declareVariable($3.loc, *$3.string, $1.type);
    }
    | init_declarator_list COMMA IDENTIFIER array_specifier {
        $$ = $1;
        parseContext.declareVariable($3.loc, *$3.string, $1.type, $4.arraySizes);
    }
    | init_declarator_list COMMA IDENTIFIER array_specifier EQUAL initializer {
        $$.type = $1.type;
        TIntermNode* initNode = parseContext.declareVariable($3.loc, *$3.string, $1.type, $4.arraySizes, $6);
        $$.intermNode = parseContext.intermediate.growAggregate($1.intermNode, initNode, $5.loc);
    }
    | init_declarator_list COMMA IDENTIFIER EQUAL initializer {
        $$.type = $1.type;
        TIntermNode* initNode = parseContext.declareVariable($3.loc, *$3.string, $1.type, 0, $5);
        $$.intermNode = parseContext.intermediate.growAggregate($1.intermNode, initNode, $4.loc);
    }
    ;

single_declaration
    : fully_specified_type {
        $$.type = $1;
        $$.intermNode = 0;

        parseContext.declareTypeDefaults($$.loc, $$.type);

    }
    | fully_specified_type IDENTIFIER {
        $$.type = $1;
        $$.intermNode = 0;
        parseContext.declareVariable($2.loc, *$2.string, $1);
    }
    | fully_specified_type IDENTIFIER array_specifier {
        $$.type = $1;
        $$.intermNode = 0;
        parseContext.declareVariable($2.loc, *$2.string, $1, $3.arraySizes);
    }
    | fully_specified_type IDENTIFIER array_specifier EQUAL initializer {
        $$.type = $1;
        TIntermNode* initNode = parseContext.declareVariable($2.loc, *$2.string, $1, $3.arraySizes, $5);
        $$.intermNode = parseContext.intermediate.growAggregate(0, initNode, $4.loc);
    }
    | fully_specified_type IDENTIFIER EQUAL initializer {
        $$.type = $1;
        TIntermNode* initNode = parseContext.declareVariable($2.loc, *$2.string, $1, 0, $4);
        $$.intermNode = parseContext.intermediate.growAggregate(0, initNode, $3.loc);
    }

// Grammar Note:  No 'enum', or 'typedef'.

fully_specified_type
    : type_specifier {
        $$ = $1;

        parseContext.globalQualifierTypeCheck($1.loc, $1.qualifier, $$);
        if ($1.arraySizes) {
            parseContext.profileRequires($1.loc, ENoProfile, 120, E_GL_3DL_array_objects, "arrayed type");
            parseContext.profileRequires($1.loc, EEsProfile, 300, 0, "arrayed type");
        }
        parseContext.precisionQualifierCheck($$.loc, $$.basicType, $$.qualifier);
    }
    | type_qualifier type_specifier  {
        parseContext.globalQualifierFixCheck($1.loc, $1.qualifier);
        parseContext.globalQualifierTypeCheck($1.loc, $1.qualifier, $2);

        if ($2.arraySizes) {
            parseContext.profileRequires($2.loc, ENoProfile, 120, E_GL_3DL_array_objects, "arrayed type");
            parseContext.profileRequires($2.loc, EEsProfile, 300, 0, "arrayed type");
        }

        if ($2.arraySizes && parseContext.arrayQualifierError($2.loc, $1.qualifier))
            $2.arraySizes = nullptr;

        parseContext.checkNoShaderLayouts($2.loc, $1.shaderQualifiers);
        $2.shaderQualifiers.merge($1.shaderQualifiers);
        parseContext.mergeQualifiers($2.loc, $2.qualifier, $1.qualifier, true);
        parseContext.precisionQualifierCheck($2.loc, $2.basicType, $2.qualifier);

        $$ = $2;

        if (! $$.qualifier.isInterpolation() &&
            ((parseContext.language == EShLangVertex   && $$.qualifier.storage == EvqVaryingOut) ||
             (parseContext.language == EShLangFragment && $$.qualifier.storage == EvqVaryingIn)))
            $$.qualifier.smooth = true;
    }
    ;

invariant_qualifier
    : INVARIANT {
        parseContext.globalCheck($1.loc, "invariant");
        parseContext.profileRequires($$.loc, ENoProfile, 120, 0, "invariant");
        $$.init($1.loc);
        $$.qualifier.invariant = true;
    }
    ;

interpolation_qualifier
    : SMOOTH {
        parseContext.globalCheck($1.loc, "smooth");
        parseContext.profileRequires($1.loc, ENoProfile, 130, 0, "smooth");
        parseContext.profileRequires($1.loc, EEsProfile, 300, 0, "smooth");
        $$.init($1.loc);
        $$.qualifier.smooth = true;
    }
    | FLAT {
        parseContext.globalCheck($1.loc, "flat");
        parseContext.profileRequires($1.loc, ENoProfile, 130, 0, "flat");
        parseContext.profileRequires($1.loc, EEsProfile, 300, 0, "flat");
        $$.init($1.loc);
        $$.qualifier.flat = true;
    }

    | NOPERSPECTIVE {
        parseContext.globalCheck($1.loc, "noperspective");
        parseContext.profileRequires($1.loc, EEsProfile, 0, E_GL_NV_shader_noperspective_interpolation, "noperspective");
        parseContext.profileRequires($1.loc, ENoProfile, 130, 0, "noperspective");
        $$.init($1.loc);
        $$.qualifier.nopersp = true;
    }
    | EXPLICITINTERPAMD {
        parseContext.globalCheck($1.loc, "__explicitInterpAMD");
        parseContext.profileRequires($1.loc, ECoreProfile, 450, E_GL_AMD_shader_explicit_vertex_parameter, "explicit interpolation");
        parseContext.profileRequires($1.loc, ECompatibilityProfile, 450, E_GL_AMD_shader_explicit_vertex_parameter, "explicit interpolation");
        $$.init($1.loc);
        $$.qualifier.explicitInterp = true;
    }
    | PERVERTEXNV {
        parseContext.globalCheck($1.loc, "pervertexNV");
        parseContext.profileRequires($1.loc, ECoreProfile, 0, E_GL_NV_fragment_shader_barycentric, "fragment shader barycentric");
        parseContext.profileRequires($1.loc, ECompatibilityProfile, 0, E_GL_NV_fragment_shader_barycentric, "fragment shader barycentric");
        parseContext.profileRequires($1.loc, EEsProfile, 0, E_GL_NV_fragment_shader_barycentric, "fragment shader barycentric");
        $$.init($1.loc);
        $$.qualifier.pervertexNV = true;
    }
    | PERPRIMITIVENV {
        // No need for profile version or extension check. Shader stage already checks both.
        parseContext.globalCheck($1.loc, "perprimitiveNV");
        parseContext.requireStage($1.loc, (EShLanguageMask)(EShLangFragmentMask | EShLangMeshNVMask), "perprimitiveNV");
        // Fragment shader stage doesn't check for extension. So we explicitly add below extension check.
        if (parseContext.language == EShLangFragment)
            parseContext.requireExtensions($1.loc, 1, &E_GL_NV_mesh_shader, "perprimitiveNV");
        $$.init($1.loc);
        $$.qualifier.perPrimitiveNV = true;
    }
    | PERVIEWNV {
        // No need for profile version or extension check. Shader stage already checks both.
        parseContext.globalCheck($1.loc, "perviewNV");
        parseContext.requireStage($1.loc, EShLangMeshNV, "perviewNV");
        $$.init($1.loc);
        $$.qualifier.perViewNV = true;
    }
    | PERTASKNV {
        // No need for profile version or extension check. Shader stage already checks both.
        parseContext.globalCheck($1.loc, "taskNV");
        parseContext.requireStage($1.loc, (EShLanguageMask)(EShLangTaskNVMask | EShLangMeshNVMask), "taskNV");
        $$.init($1.loc);
        $$.qualifier.perTaskNV = true;
    }

    ;

layout_qualifier
    : LAYOUT LEFT_PAREN layout_qualifier_id_list RIGHT_PAREN {
        $$ = $3;
    }
    ;

layout_qualifier_id_list
    : layout_qualifier_id {
        $$ = $1;
    }
    | layout_qualifier_id_list COMMA layout_qualifier_id {
        $$ = $1;
        $$.shaderQualifiers.merge($3.shaderQualifiers);
        parseContext.mergeObjectLayoutQualifiers($$.qualifier, $3.qualifier, false);
    }

layout_qualifier_id
    : IDENTIFIER {
        $$.init($1.loc);
        parseContext.setLayoutQualifier($1.loc, $$, *$1.string);
    }
    | IDENTIFIER EQUAL constant_expression {
        $$.init($1.loc);
        parseContext.setLayoutQualifier($1.loc, $$, *$1.string, $3);
    }
    | SHARED { // because "shared" is both an identifier and a keyword
        $$.init($1.loc);
        TString strShared("shared");
        parseContext.setLayoutQualifier($1.loc, $$, strShared);
    }
    ;


precise_qualifier
    : PRECISE {
        parseContext.profileRequires($$.loc, ECoreProfile | ECompatibilityProfile, 400, E_GL_ARB_gpu_shader5, "precise");
        parseContext.profileRequires($1.loc, EEsProfile, 320, Num_AEP_gpu_shader5, AEP_gpu_shader5, "precise");
        $$.init($1.loc);
        $$.qualifier.noContraction = true;
    }
    ;


type_qualifier
    : single_type_qualifier {
        $$ = $1;
    }
    | type_qualifier single_type_qualifier {
        $$ = $1;
        if ($$.basicType == EbtVoid)
            $$.basicType = $2.basicType;

        $$.shaderQualifiers.merge($2.shaderQualifiers);
        parseContext.mergeQualifiers($$.loc, $$.qualifier, $2.qualifier, false);
    }
    ;

single_type_qualifier
    : storage_qualifier {
        $$ = $1;
    }
    | layout_qualifier {
        $$ = $1;
    }
    | precision_qualifier {
        parseContext.checkPrecisionQualifier($1.loc, $1.qualifier.precision);
        $$ = $1;
    }
    | interpolation_qualifier {
        // allow inheritance of storage qualifier from block declaration
        $$ = $1;
    }
    | invariant_qualifier {
        // allow inheritance of storage qualifier from block declaration
        $$ = $1;
    }

    | precise_qualifier {
        // allow inheritance of storage qualifier from block declaration
        $$ = $1;
    }
    | non_uniform_qualifier {
        $$ = $1;
    }
    | spirv_storage_class_qualifier {
        parseContext.globalCheck($1.loc, "spirv_storage_class");
        parseContext.requireExtensions($1.loc, 1, &E_GL_EXT_spirv_intrinsics, "SPIR-V storage class qualifier");
        $$ = $1;
    }
    | spirv_decorate_qualifier {
        parseContext.requireExtensions($1.loc, 1, &E_GL_EXT_spirv_intrinsics, "SPIR-V decorate qualifier");
        $$ = $1;
    }
    | SPIRV_BY_REFERENCE {
        parseContext.requireExtensions($1.loc, 1, &E_GL_EXT_spirv_intrinsics, "spirv_by_reference");
        $$.init($1.loc);
        $$.qualifier.setSpirvByReference();
    }
    | SPIRV_LITERAL {
        parseContext.requireExtensions($1.loc, 1, &E_GL_EXT_spirv_intrinsics, "spirv_by_literal");
        $$.init($1.loc);
        $$.qualifier.setSpirvLiteral();
    }

    ;

storage_qualifier
    : CONST {
        $$.init($1.loc);
        $$.qualifier.storage = EvqConst;  // will later turn into EvqConstReadOnly, if the initializer is not constant
    }
    | INOUT {
        parseContext.globalCheck($1.loc, "inout");
        $$.init($1.loc);
        $$.qualifier.storage = EvqInOut;
    }
    | IN {
        parseContext.globalCheck($1.loc, "in");
        $$.init($1.loc);
        // whether this is a parameter "in" or a pipeline "in" will get sorted out a bit later
        $$.qualifier.storage = EvqIn;
    }
    | OUT {
        parseContext.globalCheck($1.loc, "out");
        $$.init($1.loc);
        // whether this is a parameter "out" or a pipeline "out" will get sorted out a bit later
        $$.qualifier.storage = EvqOut;
    }
    | CENTROID {
        parseContext.profileRequires($1.loc, ENoProfile, 120, 0, "centroid");
        parseContext.profileRequires($1.loc, EEsProfile, 300, 0, "centroid");
        parseContext.globalCheck($1.loc, "centroid");
        $$.init($1.loc);
        $$.qualifier.centroid = true;
    }
    | UNIFORM {
        parseContext.globalCheck($1.loc, "uniform");
        $$.init($1.loc);
        $$.qualifier.storage = EvqUniform;
    }
    | SHARED {
        parseContext.globalCheck($1.loc, "shared");
        parseContext.profileRequires($1.loc, ECoreProfile | ECompatibilityProfile, 430, E_GL_ARB_compute_shader, "shared");
        parseContext.profileRequires($1.loc, EEsProfile, 310, 0, "shared");
        parseContext.requireStage($1.loc, (EShLanguageMask)(EShLangComputeMask | EShLangMeshNVMask | EShLangTaskNVMask), "shared");
        $$.init($1.loc);
        $$.qualifier.storage = EvqShared;
    }
    | BUFFER {
        parseContext.globalCheck($1.loc, "buffer");
        $$.init($1.loc);
        $$.qualifier.storage = EvqBuffer;
    }

    | ATTRIBUTE {
        parseContext.requireStage($1.loc, EShLangVertex, "attribute");
        parseContext.checkDeprecated($1.loc, ECoreProfile, 130, "attribute");
        parseContext.checkDeprecated($1.loc, ENoProfile, 130, "attribute");
        parseContext.requireNotRemoved($1.loc, ECoreProfile, 420, "attribute");
        parseContext.requireNotRemoved($1.loc, EEsProfile, 300, "attribute");

        parseContext.globalCheck($1.loc, "attribute");

        $$.init($1.loc);
        $$.qualifier.storage = EvqVaryingIn;
    }
    | VARYING {
        parseContext.checkDeprecated($1.loc, ENoProfile, 130, "varying");
        parseContext.checkDeprecated($1.loc, ECoreProfile, 130, "varying");
        parseContext.requireNotRemoved($1.loc, ECoreProfile, 420, "varying");
        parseContext.requireNotRemoved($1.loc, EEsProfile, 300, "varying");

        parseContext.globalCheck($1.loc, "varying");

        $$.init($1.loc);
        if (parseContext.language == EShLangVertex)
            $$.qualifier.storage = EvqVaryingOut;
        else
            $$.qualifier.storage = EvqVaryingIn;
    }
    | PATCH {
        parseContext.globalCheck($1.loc, "patch");
        parseContext.requireStage($1.loc, (EShLanguageMask)(EShLangTessControlMask | EShLangTessEvaluationMask), "patch");
        $$.init($1.loc);
        $$.qualifier.patch = true;
    }
    | SAMPLE {
        parseContext.globalCheck($1.loc, "sample");
        $$.init($1.loc);
        $$.qualifier.sample = true;
    }
    | HITATTRNV {
        parseContext.globalCheck($1.loc, "hitAttributeNV");
        parseContext.requireStage($1.loc, (EShLanguageMask)(EShLangIntersectMask | EShLangClosestHitMask
            | EShLangAnyHitMask), "hitAttributeNV");
        parseContext.profileRequires($1.loc, ECoreProfile, 460, E_GL_NV_ray_tracing, "hitAttributeNV");
        $$.init($1.loc);
        $$.qualifier.storage = EvqHitAttr;
    }
    | HITATTREXT {
        parseContext.globalCheck($1.loc, "hitAttributeEXT");
        parseContext.requireStage($1.loc, (EShLanguageMask)(EShLangIntersectMask | EShLangClosestHitMask
            | EShLangAnyHitMask), "hitAttributeEXT");
        parseContext.profileRequires($1.loc, ECoreProfile, 460, E_GL_EXT_ray_tracing, "hitAttributeNV");
        $$.init($1.loc);
        $$.qualifier.storage = EvqHitAttr;
    }
    | PAYLOADNV {
        parseContext.globalCheck($1.loc, "rayPayloadNV");
        parseContext.requireStage($1.loc, (EShLanguageMask)(EShLangRayGenMask | EShLangClosestHitMask |
            EShLangAnyHitMask | EShLangMissMask), "rayPayloadNV");
        parseContext.profileRequires($1.loc, ECoreProfile, 460, E_GL_NV_ray_tracing, "rayPayloadNV");
        $$.init($1.loc);
        $$.qualifier.storage = EvqPayload;
    }
    | PAYLOADEXT {
        parseContext.globalCheck($1.loc, "rayPayloadEXT");
        parseContext.requireStage($1.loc, (EShLanguageMask)(EShLangRayGenMask | EShLangClosestHitMask |
            EShLangAnyHitMask | EShLangMissMask), "rayPayloadEXT");
        parseContext.profileRequires($1.loc, ECoreProfile, 460, E_GL_EXT_ray_tracing, "rayPayloadEXT");
        $$.init($1.loc);
        $$.qualifier.storage = EvqPayload;
    }
    | PAYLOADINNV {
        parseContext.globalCheck($1.loc, "rayPayloadInNV");
        parseContext.requireStage($1.loc, (EShLanguageMask)(EShLangClosestHitMask |
            EShLangAnyHitMask | EShLangMissMask), "rayPayloadInNV");
        parseContext.profileRequires($1.loc, ECoreProfile, 460, E_GL_NV_ray_tracing, "rayPayloadInNV");
        $$.init($1.loc);
        $$.qualifier.storage = EvqPayloadIn;
    }
    | PAYLOADINEXT {
        parseContext.globalCheck($1.loc, "rayPayloadInEXT");
        parseContext.requireStage($1.loc, (EShLanguageMask)(EShLangClosestHitMask |
            EShLangAnyHitMask | EShLangMissMask), "rayPayloadInEXT");
        parseContext.profileRequires($1.loc, ECoreProfile, 460, E_GL_EXT_ray_tracing, "rayPayloadInEXT");
        $$.init($1.loc);
        $$.qualifier.storage = EvqPayloadIn;
    }
    | CALLDATANV {
        parseContext.globalCheck($1.loc, "callableDataNV");
        parseContext.requireStage($1.loc, (EShLanguageMask)(EShLangRayGenMask |
            EShLangClosestHitMask | EShLangMissMask | EShLangCallableMask), "callableDataNV");
        parseContext.profileRequires($1.loc, ECoreProfile, 460, E_GL_NV_ray_tracing, "callableDataNV");
        $$.init($1.loc);
        $$.qualifier.storage = EvqCallableData;
    }
    | CALLDATAEXT {
        parseContext.globalCheck($1.loc, "callableDataEXT");
        parseContext.requireStage($1.loc, (EShLanguageMask)(EShLangRayGenMask |
            EShLangClosestHitMask | EShLangMissMask | EShLangCallableMask), "callableDataEXT");
        parseContext.profileRequires($1.loc, ECoreProfile, 460, E_GL_EXT_ray_tracing, "callableDataEXT");
        $$.init($1.loc);
        $$.qualifier.storage = EvqCallableData;
    }
    | CALLDATAINNV {
        parseContext.globalCheck($1.loc, "callableDataInNV");
        parseContext.requireStage($1.loc, (EShLanguageMask)(EShLangCallableMask), "callableDataInNV");
        parseContext.profileRequires($1.loc, ECoreProfile, 460, E_GL_NV_ray_tracing, "callableDataInNV");
        $$.init($1.loc);
        $$.qualifier.storage = EvqCallableDataIn;
    }
    | CALLDATAINEXT {
        parseContext.globalCheck($1.loc, "callableDataInEXT");
        parseContext.requireStage($1.loc, (EShLanguageMask)(EShLangCallableMask), "callableDataInEXT");
        parseContext.profileRequires($1.loc, ECoreProfile, 460, E_GL_EXT_ray_tracing, "callableDataInEXT");
        $$.init($1.loc);
        $$.qualifier.storage = EvqCallableDataIn;
    }
    | COHERENT {
        $$.init($1.loc);
        $$.qualifier.coherent = true;
    }
    | DEVICECOHERENT {
        $$.init($1.loc);
        parseContext.requireExtensions($1.loc, 1, &E_GL_KHR_memory_scope_semantics, "devicecoherent");
        $$.qualifier.devicecoherent = true;
    }
    | QUEUEFAMILYCOHERENT {
        $$.init($1.loc);
        parseContext.requireExtensions($1.loc, 1, &E_GL_KHR_memory_scope_semantics, "queuefamilycoherent");
        $$.qualifier.queuefamilycoherent = true;
    }
    | WORKGROUPCOHERENT {
        $$.init($1.loc);
        parseContext.requireExtensions($1.loc, 1, &E_GL_KHR_memory_scope_semantics, "workgroupcoherent");
        $$.qualifier.workgroupcoherent = true;
    }
    | SUBGROUPCOHERENT {
        $$.init($1.loc);
        parseContext.requireExtensions($1.loc, 1, &E_GL_KHR_memory_scope_semantics, "subgroupcoherent");
        $$.qualifier.subgroupcoherent = true;
    }
    | NONPRIVATE {
        $$.init($1.loc);
        parseContext.requireExtensions($1.loc, 1, &E_GL_KHR_memory_scope_semantics, "nonprivate");
        $$.qualifier.nonprivate = true;
    }
    | SHADERCALLCOHERENT {
        $$.init($1.loc);
        parseContext.requireExtensions($1.loc, 1, &E_GL_EXT_ray_tracing, "shadercallcoherent");
        $$.qualifier.shadercallcoherent = true;
    }
    | VOLATILE {
        $$.init($1.loc);
        $$.qualifier.volatil = true;
    }
    | RESTRICT {
        $$.init($1.loc);
        $$.qualifier.restrict = true;
    }
    | READONLY {
        $$.init($1.loc);
        $$.qualifier.readonly = true;
    }
    | WRITEONLY {
        $$.init($1.loc);
        $$.qualifier.writeonly = true;
    }
    | SUBROUTINE {
        parseContext.spvRemoved($1.loc, "subroutine");
        parseContext.globalCheck($1.loc, "subroutine");
        parseContext.unimplemented($1.loc, "subroutine");
        $$.init($1.loc);
    }
    | SUBROUTINE LEFT_PAREN type_name_list RIGHT_PAREN {
        parseContext.spvRemoved($1.loc, "subroutine");
        parseContext.globalCheck($1.loc, "subroutine");
        parseContext.unimplemented($1.loc, "subroutine");
        $$.init($1.loc);
    }

    ;


non_uniform_qualifier
    : NONUNIFORM {
        $$.init($1.loc);
        $$.qualifier.nonUniform = true;
    }
    ;

type_name_list
    : IDENTIFIER {
        // TODO
    }
    | type_name_list COMMA IDENTIFIER {
        // TODO: 4.0 semantics: subroutines
        // 1) make sure each identifier is a type declared earlier with SUBROUTINE
        // 2) save all of the identifiers for future comparison with the declared function
    }
    ;


type_specifier
    : type_specifier_nonarray type_parameter_specifier_opt {
        $$ = $1;
        $$.qualifier.precision = parseContext.getDefaultPrecision($$);
        $$.typeParameters = $2;
    }
    | type_specifier_nonarray type_parameter_specifier_opt array_specifier {
        parseContext.arrayOfArrayVersionCheck($3.loc, $3.arraySizes);
        $$ = $1;
        $$.qualifier.precision = parseContext.getDefaultPrecision($$);
        $$.typeParameters = $2;
        $$.arraySizes = $3.arraySizes;
    }
    ;

array_specifier
    : LEFT_BRACKET RIGHT_BRACKET {
        $$.loc = $1.loc;
        $$.arraySizes = new TArraySizes;
        $$.arraySizes->addInnerSize();
    }
    | LEFT_BRACKET conditional_expression RIGHT_BRACKET {
        $$.loc = $1.loc;
        $$.arraySizes = new TArraySizes;

        TArraySize size;
        parseContext.arraySizeCheck($2->getLoc(), $2, size, "array size");
        $$.arraySizes->addInnerSize(size);
    }
    | array_specifier LEFT_BRACKET RIGHT_BRACKET {
        $$ = $1;
        $$.arraySizes->addInnerSize();
    }
    | array_specifier LEFT_BRACKET conditional_expression RIGHT_BRACKET {
        $$ = $1;

        TArraySize size;
        parseContext.arraySizeCheck($3->getLoc(), $3, size, "array size");
        $$.arraySizes->addInnerSize(size);
    }
    ;

type_parameter_specifier_opt
    : type_parameter_specifier {
        $$ = $1;
    }
    | /* May be null */ {
        $$ = 0;
    }
    ;

type_parameter_specifier
    : LEFT_ANGLE type_parameter_specifier_list RIGHT_ANGLE {
        $$ = $2;
    }
    ;

type_parameter_specifier_list
    : unary_expression {
        $$ = new TArraySizes;

        TArraySize size;
        parseContext.arraySizeCheck($1->getLoc(), $1, size, "type parameter");
        $$->addInnerSize(size);
    }
    | type_parameter_specifier_list COMMA unary_expression {
        $$ = $1;

        TArraySize size;
        parseContext.arraySizeCheck($3->getLoc(), $3, size, "type parameter");
        $$->addInnerSize(size);
    }
    ;

type_specifier_nonarray
    : VOID {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtVoid;
    }
    | FLOAT {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
    }
    | INT {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtInt;
    }
    | UINT {
        parseContext.fullIntegerCheck($1.loc, "unsigned integer");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtUint;
    }
    | BOOL {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtBool;
    }
    | VEC2 {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setVector(2);
    }
    | VEC3 {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setVector(3);
    }
    | VEC4 {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setVector(4);
    }
    | BVEC2 {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtBool;
        $$.setVector(2);
    }
    | BVEC3 {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtBool;
        $$.setVector(3);
    }
    | BVEC4 {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtBool;
        $$.setVector(4);
    }
    | IVEC2 {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtInt;
        $$.setVector(2);
    }
    | IVEC3 {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtInt;
        $$.setVector(3);
    }
    | IVEC4 {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtInt;
        $$.setVector(4);
    }
    | UVEC2 {
        parseContext.fullIntegerCheck($1.loc, "unsigned integer vector");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtUint;
        $$.setVector(2);
    }
    | UVEC3 {
        parseContext.fullIntegerCheck($1.loc, "unsigned integer vector");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtUint;
        $$.setVector(3);
    }
    | UVEC4 {
        parseContext.fullIntegerCheck($1.loc, "unsigned integer vector");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtUint;
        $$.setVector(4);
    }
    | MAT2 {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setMatrix(2, 2);
    }
    | MAT3 {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setMatrix(3, 3);
    }
    | MAT4 {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setMatrix(4, 4);
    }
    | MAT2X2 {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setMatrix(2, 2);
    }
    | MAT2X3 {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setMatrix(2, 3);
    }
    | MAT2X4 {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setMatrix(2, 4);
    }
    | MAT3X2 {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setMatrix(3, 2);
    }
    | MAT3X3 {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setMatrix(3, 3);
    }
    | MAT3X4 {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setMatrix(3, 4);
    }
    | MAT4X2 {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setMatrix(4, 2);
    }
    | MAT4X3 {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setMatrix(4, 3);
    }
    | MAT4X4 {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setMatrix(4, 4);
    }

    | DOUBLE {
        parseContext.requireProfile($1.loc, ECoreProfile | ECompatibilityProfile, "double");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck($1.loc, "double");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
    }
    | FLOAT16_T {
        parseContext.float16ScalarVectorCheck($1.loc, "float16_t", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat16;
    }
    | FLOAT32_T {
        parseContext.explicitFloat32Check($1.loc, "float32_t", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
    }
    | FLOAT64_T {
        parseContext.explicitFloat64Check($1.loc, "float64_t", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
    }
    | INT8_T {
        parseContext.int8ScalarVectorCheck($1.loc, "8-bit signed integer", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtInt8;
    }
    | UINT8_T {
        parseContext.int8ScalarVectorCheck($1.loc, "8-bit unsigned integer", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtUint8;
    }
    | INT16_T {
        parseContext.int16ScalarVectorCheck($1.loc, "16-bit signed integer", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtInt16;
    }
    | UINT16_T {
        parseContext.int16ScalarVectorCheck($1.loc, "16-bit unsigned integer", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtUint16;
    }
    | INT32_T {
        parseContext.explicitInt32Check($1.loc, "32-bit signed integer", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtInt;
    }
    | UINT32_T {
        parseContext.explicitInt32Check($1.loc, "32-bit unsigned integer", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtUint;
    }
    | INT64_T {
        parseContext.int64Check($1.loc, "64-bit integer", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtInt64;
    }
    | UINT64_T {
        parseContext.int64Check($1.loc, "64-bit unsigned integer", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtUint64;
    }
    | DVEC2 {
        parseContext.requireProfile($1.loc, ECoreProfile | ECompatibilityProfile, "double vector");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck($1.loc, "double vector");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setVector(2);
    }
    | DVEC3 {
        parseContext.requireProfile($1.loc, ECoreProfile | ECompatibilityProfile, "double vector");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck($1.loc, "double vector");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setVector(3);
    }
    | DVEC4 {
        parseContext.requireProfile($1.loc, ECoreProfile | ECompatibilityProfile, "double vector");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck($1.loc, "double vector");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setVector(4);
    }
    | F16VEC2 {
        parseContext.float16ScalarVectorCheck($1.loc, "half float vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat16;
        $$.setVector(2);
    }
    | F16VEC3 {
        parseContext.float16ScalarVectorCheck($1.loc, "half float vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat16;
        $$.setVector(3);
    }
    | F16VEC4 {
        parseContext.float16ScalarVectorCheck($1.loc, "half float vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat16;
        $$.setVector(4);
    }
    | F32VEC2 {
        parseContext.explicitFloat32Check($1.loc, "float32_t vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setVector(2);
    }
    | F32VEC3 {
        parseContext.explicitFloat32Check($1.loc, "float32_t vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setVector(3);
    }
    | F32VEC4 {
        parseContext.explicitFloat32Check($1.loc, "float32_t vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setVector(4);
    }
    | F64VEC2 {
        parseContext.explicitFloat64Check($1.loc, "float64_t vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setVector(2);
    }
    | F64VEC3 {
        parseContext.explicitFloat64Check($1.loc, "float64_t vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setVector(3);
    }
    | F64VEC4 {
        parseContext.explicitFloat64Check($1.loc, "float64_t vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setVector(4);
    }
    | I8VEC2 {
        parseContext.int8ScalarVectorCheck($1.loc, "8-bit signed integer vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtInt8;
        $$.setVector(2);
    }
    | I8VEC3 {
        parseContext.int8ScalarVectorCheck($1.loc, "8-bit signed integer vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtInt8;
        $$.setVector(3);
    }
    | I8VEC4 {
        parseContext.int8ScalarVectorCheck($1.loc, "8-bit signed integer vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtInt8;
        $$.setVector(4);
    }
    | I16VEC2 {
        parseContext.int16ScalarVectorCheck($1.loc, "16-bit signed integer vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtInt16;
        $$.setVector(2);
    }
    | I16VEC3 {
        parseContext.int16ScalarVectorCheck($1.loc, "16-bit signed integer vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtInt16;
        $$.setVector(3);
    }
    | I16VEC4 {
        parseContext.int16ScalarVectorCheck($1.loc, "16-bit signed integer vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtInt16;
        $$.setVector(4);
    }
    | I32VEC2 {
        parseContext.explicitInt32Check($1.loc, "32-bit signed integer vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtInt;
        $$.setVector(2);
    }
    | I32VEC3 {
        parseContext.explicitInt32Check($1.loc, "32-bit signed integer vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtInt;
        $$.setVector(3);
    }
    | I32VEC4 {
        parseContext.explicitInt32Check($1.loc, "32-bit signed integer vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtInt;
        $$.setVector(4);
    }
    | I64VEC2 {
        parseContext.int64Check($1.loc, "64-bit integer vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtInt64;
        $$.setVector(2);
    }
    | I64VEC3 {
        parseContext.int64Check($1.loc, "64-bit integer vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtInt64;
        $$.setVector(3);
    }
    | I64VEC4 {
        parseContext.int64Check($1.loc, "64-bit integer vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtInt64;
        $$.setVector(4);
    }
    | U8VEC2 {
        parseContext.int8ScalarVectorCheck($1.loc, "8-bit unsigned integer vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtUint8;
        $$.setVector(2);
    }
    | U8VEC3 {
        parseContext.int8ScalarVectorCheck($1.loc, "8-bit unsigned integer vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtUint8;
        $$.setVector(3);
    }
    | U8VEC4 {
        parseContext.int8ScalarVectorCheck($1.loc, "8-bit unsigned integer vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtUint8;
        $$.setVector(4);
    }
    | U16VEC2 {
        parseContext.int16ScalarVectorCheck($1.loc, "16-bit unsigned integer vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtUint16;
        $$.setVector(2);
    }
    | U16VEC3 {
        parseContext.int16ScalarVectorCheck($1.loc, "16-bit unsigned integer vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtUint16;
        $$.setVector(3);
    }
    | U16VEC4 {
        parseContext.int16ScalarVectorCheck($1.loc, "16-bit unsigned integer vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtUint16;
        $$.setVector(4);
    }
    | U32VEC2 {
        parseContext.explicitInt32Check($1.loc, "32-bit unsigned integer vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtUint;
        $$.setVector(2);
    }
    | U32VEC3 {
        parseContext.explicitInt32Check($1.loc, "32-bit unsigned integer vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtUint;
        $$.setVector(3);
    }
    | U32VEC4 {
        parseContext.explicitInt32Check($1.loc, "32-bit unsigned integer vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtUint;
        $$.setVector(4);
    }
    | U64VEC2 {
        parseContext.int64Check($1.loc, "64-bit unsigned integer vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtUint64;
        $$.setVector(2);
    }
    | U64VEC3 {
        parseContext.int64Check($1.loc, "64-bit unsigned integer vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtUint64;
        $$.setVector(3);
    }
    | U64VEC4 {
        parseContext.int64Check($1.loc, "64-bit unsigned integer vector", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtUint64;
        $$.setVector(4);
    }
    | DMAT2 {
        parseContext.requireProfile($1.loc, ECoreProfile | ECompatibilityProfile, "double matrix");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck($1.loc, "double matrix");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setMatrix(2, 2);
    }
    | DMAT3 {
        parseContext.requireProfile($1.loc, ECoreProfile | ECompatibilityProfile, "double matrix");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck($1.loc, "double matrix");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setMatrix(3, 3);
    }
    | DMAT4 {
        parseContext.requireProfile($1.loc, ECoreProfile | ECompatibilityProfile, "double matrix");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck($1.loc, "double matrix");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setMatrix(4, 4);
    }
    | DMAT2X2 {
        parseContext.requireProfile($1.loc, ECoreProfile | ECompatibilityProfile, "double matrix");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck($1.loc, "double matrix");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setMatrix(2, 2);
    }
    | DMAT2X3 {
        parseContext.requireProfile($1.loc, ECoreProfile | ECompatibilityProfile, "double matrix");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck($1.loc, "double matrix");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setMatrix(2, 3);
    }
    | DMAT2X4 {
        parseContext.requireProfile($1.loc, ECoreProfile | ECompatibilityProfile, "double matrix");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck($1.loc, "double matrix");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setMatrix(2, 4);
    }
    | DMAT3X2 {
        parseContext.requireProfile($1.loc, ECoreProfile | ECompatibilityProfile, "double matrix");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck($1.loc, "double matrix");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setMatrix(3, 2);
    }
    | DMAT3X3 {
        parseContext.requireProfile($1.loc, ECoreProfile | ECompatibilityProfile, "double matrix");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck($1.loc, "double matrix");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setMatrix(3, 3);
    }
    | DMAT3X4 {
        parseContext.requireProfile($1.loc, ECoreProfile | ECompatibilityProfile, "double matrix");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck($1.loc, "double matrix");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setMatrix(3, 4);
    }
    | DMAT4X2 {
        parseContext.requireProfile($1.loc, ECoreProfile | ECompatibilityProfile, "double matrix");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck($1.loc, "double matrix");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setMatrix(4, 2);
    }
    | DMAT4X3 {
        parseContext.requireProfile($1.loc, ECoreProfile | ECompatibilityProfile, "double matrix");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck($1.loc, "double matrix");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setMatrix(4, 3);
    }
    | DMAT4X4 {
        parseContext.requireProfile($1.loc, ECoreProfile | ECompatibilityProfile, "double matrix");
        if (! parseContext.symbolTable.atBuiltInLevel())
            parseContext.doubleCheck($1.loc, "double matrix");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setMatrix(4, 4);
    }
    | F16MAT2 {
        parseContext.float16Check($1.loc, "half float matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat16;
        $$.setMatrix(2, 2);
    }
    | F16MAT3 {
        parseContext.float16Check($1.loc, "half float matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat16;
        $$.setMatrix(3, 3);
    }
    | F16MAT4 {
        parseContext.float16Check($1.loc, "half float matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat16;
        $$.setMatrix(4, 4);
    }
    | F16MAT2X2 {
        parseContext.float16Check($1.loc, "half float matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat16;
        $$.setMatrix(2, 2);
    }
    | F16MAT2X3 {
        parseContext.float16Check($1.loc, "half float matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat16;
        $$.setMatrix(2, 3);
    }
    | F16MAT2X4 {
        parseContext.float16Check($1.loc, "half float matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat16;
        $$.setMatrix(2, 4);
    }
    | F16MAT3X2 {
        parseContext.float16Check($1.loc, "half float matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat16;
        $$.setMatrix(3, 2);
    }
    | F16MAT3X3 {
        parseContext.float16Check($1.loc, "half float matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat16;
        $$.setMatrix(3, 3);
    }
    | F16MAT3X4 {
        parseContext.float16Check($1.loc, "half float matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat16;
        $$.setMatrix(3, 4);
    }
    | F16MAT4X2 {
        parseContext.float16Check($1.loc, "half float matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat16;
        $$.setMatrix(4, 2);
    }
    | F16MAT4X3 {
        parseContext.float16Check($1.loc, "half float matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat16;
        $$.setMatrix(4, 3);
    }
    | F16MAT4X4 {
        parseContext.float16Check($1.loc, "half float matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat16;
        $$.setMatrix(4, 4);
    }
    | F32MAT2 {
        parseContext.explicitFloat32Check($1.loc, "float32_t matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setMatrix(2, 2);
    }
    | F32MAT3 {
        parseContext.explicitFloat32Check($1.loc, "float32_t matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setMatrix(3, 3);
    }
    | F32MAT4 {
        parseContext.explicitFloat32Check($1.loc, "float32_t matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setMatrix(4, 4);
    }
    | F32MAT2X2 {
        parseContext.explicitFloat32Check($1.loc, "float32_t matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setMatrix(2, 2);
    }
    | F32MAT2X3 {
        parseContext.explicitFloat32Check($1.loc, "float32_t matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setMatrix(2, 3);
    }
    | F32MAT2X4 {
        parseContext.explicitFloat32Check($1.loc, "float32_t matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setMatrix(2, 4);
    }
    | F32MAT3X2 {
        parseContext.explicitFloat32Check($1.loc, "float32_t matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setMatrix(3, 2);
    }
    | F32MAT3X3 {
        parseContext.explicitFloat32Check($1.loc, "float32_t matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setMatrix(3, 3);
    }
    | F32MAT3X4 {
        parseContext.explicitFloat32Check($1.loc, "float32_t matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setMatrix(3, 4);
    }
    | F32MAT4X2 {
        parseContext.explicitFloat32Check($1.loc, "float32_t matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setMatrix(4, 2);
    }
    | F32MAT4X3 {
        parseContext.explicitFloat32Check($1.loc, "float32_t matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setMatrix(4, 3);
    }
    | F32MAT4X4 {
        parseContext.explicitFloat32Check($1.loc, "float32_t matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.setMatrix(4, 4);
    }
    | F64MAT2 {
        parseContext.explicitFloat64Check($1.loc, "float64_t matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setMatrix(2, 2);
    }
    | F64MAT3 {
        parseContext.explicitFloat64Check($1.loc, "float64_t matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setMatrix(3, 3);
    }
    | F64MAT4 {
        parseContext.explicitFloat64Check($1.loc, "float64_t matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setMatrix(4, 4);
    }
    | F64MAT2X2 {
        parseContext.explicitFloat64Check($1.loc, "float64_t matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setMatrix(2, 2);
    }
    | F64MAT2X3 {
        parseContext.explicitFloat64Check($1.loc, "float64_t matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setMatrix(2, 3);
    }
    | F64MAT2X4 {
        parseContext.explicitFloat64Check($1.loc, "float64_t matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setMatrix(2, 4);
    }
    | F64MAT3X2 {
        parseContext.explicitFloat64Check($1.loc, "float64_t matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setMatrix(3, 2);
    }
    | F64MAT3X3 {
        parseContext.explicitFloat64Check($1.loc, "float64_t matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setMatrix(3, 3);
    }
    | F64MAT3X4 {
        parseContext.explicitFloat64Check($1.loc, "float64_t matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setMatrix(3, 4);
    }
    | F64MAT4X2 {
        parseContext.explicitFloat64Check($1.loc, "float64_t matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setMatrix(4, 2);
    }
    | F64MAT4X3 {
        parseContext.explicitFloat64Check($1.loc, "float64_t matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setMatrix(4, 3);
    }
    | F64MAT4X4 {
        parseContext.explicitFloat64Check($1.loc, "float64_t matrix", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtDouble;
        $$.setMatrix(4, 4);
    }
    | ACCSTRUCTNV {
       $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
       $$.basicType = EbtAccStruct;
    }
    | ACCSTRUCTEXT {
       $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
       $$.basicType = EbtAccStruct;
    }
    | RAYQUERYEXT {
       $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
       $$.basicType = EbtRayQuery;
    }
    | ATOMIC_UINT {
        parseContext.vulkanRemoved($1.loc, "atomic counter types");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtAtomicUint;
    }
    | SAMPLER1D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat, Esd1D);
    }

    | SAMPLER2D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat, Esd2D);
    }
    | SAMPLER3D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat, Esd3D);
    }
    | SAMPLERCUBE {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat, EsdCube);
    }
    | SAMPLER2DSHADOW {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat, Esd2D, false, true);
    }
    | SAMPLERCUBESHADOW {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat, EsdCube, false, true);
    }
    | SAMPLER2DARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat, Esd2D, true);
    }
    | SAMPLER2DARRAYSHADOW {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat, Esd2D, true, true);
    }

    | SAMPLER1DSHADOW {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat, Esd1D, false, true);
    }
    | SAMPLER1DARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat, Esd1D, true);
    }
    | SAMPLER1DARRAYSHADOW {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat, Esd1D, true, true);
    }
    | SAMPLERCUBEARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat, EsdCube, true);
    }
    | SAMPLERCUBEARRAYSHADOW {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat, EsdCube, true, true);
    }
    | F16SAMPLER1D {
        parseContext.float16OpaqueCheck($1.loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat16, Esd1D);
    }
    | F16SAMPLER2D {
        parseContext.float16OpaqueCheck($1.loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat16, Esd2D);
    }
    | F16SAMPLER3D {
        parseContext.float16OpaqueCheck($1.loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat16, Esd3D);
    }
    | F16SAMPLERCUBE {
        parseContext.float16OpaqueCheck($1.loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat16, EsdCube);
    }
    | F16SAMPLER1DSHADOW {
        parseContext.float16OpaqueCheck($1.loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat16, Esd1D, false, true);
    }
    | F16SAMPLER2DSHADOW {
        parseContext.float16OpaqueCheck($1.loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat16, Esd2D, false, true);
    }
    | F16SAMPLERCUBESHADOW {
        parseContext.float16OpaqueCheck($1.loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat16, EsdCube, false, true);
    }
    | F16SAMPLER1DARRAY {
        parseContext.float16OpaqueCheck($1.loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat16, Esd1D, true);
    }
    | F16SAMPLER2DARRAY {
        parseContext.float16OpaqueCheck($1.loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat16, Esd2D, true);
    }
    | F16SAMPLER1DARRAYSHADOW {
        parseContext.float16OpaqueCheck($1.loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat16, Esd1D, true, true);
    }
    | F16SAMPLER2DARRAYSHADOW {
        parseContext.float16OpaqueCheck($1.loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat16, Esd2D, true, true);
    }
    | F16SAMPLERCUBEARRAY {
        parseContext.float16OpaqueCheck($1.loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat16, EsdCube, true);
    }
    | F16SAMPLERCUBEARRAYSHADOW {
        parseContext.float16OpaqueCheck($1.loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat16, EsdCube, true, true);
    }
    | ISAMPLER1D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtInt, Esd1D);
    }

    | ISAMPLER2D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtInt, Esd2D);
    }
    | ISAMPLER3D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtInt, Esd3D);
    }
    | ISAMPLERCUBE {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtInt, EsdCube);
    }
    | ISAMPLER2DARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtInt, Esd2D, true);
    }
    | USAMPLER2D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtUint, Esd2D);
    }
    | USAMPLER3D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtUint, Esd3D);
    }
    | USAMPLERCUBE {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtUint, EsdCube);
    }

    | ISAMPLER1DARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtInt, Esd1D, true);
    }
    | ISAMPLERCUBEARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtInt, EsdCube, true);
    }
    | USAMPLER1D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtUint, Esd1D);
    }
    | USAMPLER1DARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtUint, Esd1D, true);
    }
    | USAMPLERCUBEARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtUint, EsdCube, true);
    }
    | TEXTURECUBEARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtFloat, EsdCube, true);
    }
    | ITEXTURECUBEARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtInt, EsdCube, true);
    }
    | UTEXTURECUBEARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtUint, EsdCube, true);
    }

    | USAMPLER2DARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtUint, Esd2D, true);
    }
    | TEXTURE2D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtFloat, Esd2D);
    }
    | TEXTURE3D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtFloat, Esd3D);
    }
    | TEXTURE2DARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtFloat, Esd2D, true);
    }
    | TEXTURECUBE {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtFloat, EsdCube);
    }
    | ITEXTURE2D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtInt, Esd2D);
    }
    | ITEXTURE3D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtInt, Esd3D);
    }
    | ITEXTURECUBE {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtInt, EsdCube);
    }
    | ITEXTURE2DARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtInt, Esd2D, true);
    }
    | UTEXTURE2D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtUint, Esd2D);
    }
    | UTEXTURE3D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtUint, Esd3D);
    }
    | UTEXTURECUBE {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtUint, EsdCube);
    }
    | UTEXTURE2DARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtUint, Esd2D, true);
    }
    | SAMPLER {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setPureSampler(false);
    }
    | SAMPLERSHADOW {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setPureSampler(true);
    }

    | SAMPLER2DRECT {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat, EsdRect);
    }
    | SAMPLER2DRECTSHADOW {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat, EsdRect, false, true);
    }
    | F16SAMPLER2DRECT {
        parseContext.float16OpaqueCheck($1.loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat16, EsdRect);
    }
    | F16SAMPLER2DRECTSHADOW {
        parseContext.float16OpaqueCheck($1.loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat16, EsdRect, false, true);
    }
    | ISAMPLER2DRECT {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtInt, EsdRect);
    }
    | USAMPLER2DRECT {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtUint, EsdRect);
    }
    | SAMPLERBUFFER {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat, EsdBuffer);
    }
    | F16SAMPLERBUFFER {
        parseContext.float16OpaqueCheck($1.loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat16, EsdBuffer);
    }
    | ISAMPLERBUFFER {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtInt, EsdBuffer);
    }
    | USAMPLERBUFFER {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtUint, EsdBuffer);
    }
    | SAMPLER2DMS {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat, Esd2D, false, false, true);
    }
    | F16SAMPLER2DMS {
        parseContext.float16OpaqueCheck($1.loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat16, Esd2D, false, false, true);
    }
    | ISAMPLER2DMS {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtInt, Esd2D, false, false, true);
    }
    | USAMPLER2DMS {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtUint, Esd2D, false, false, true);
    }
    | SAMPLER2DMSARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat, Esd2D, true, false, true);
    }
    | F16SAMPLER2DMSARRAY {
        parseContext.float16OpaqueCheck($1.loc, "half float sampler", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat16, Esd2D, true, false, true);
    }
    | ISAMPLER2DMSARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtInt, Esd2D, true, false, true);
    }
    | USAMPLER2DMSARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtUint, Esd2D, true, false, true);
    }
    | TEXTURE1D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtFloat, Esd1D);
    }
    | F16TEXTURE1D {
        parseContext.float16OpaqueCheck($1.loc, "half float texture", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtFloat16, Esd1D);
    }
    | F16TEXTURE2D {
        parseContext.float16OpaqueCheck($1.loc, "half float texture", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtFloat16, Esd2D);
    }
    | F16TEXTURE3D {
        parseContext.float16OpaqueCheck($1.loc, "half float texture", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtFloat16, Esd3D);
    }
    | F16TEXTURECUBE {
        parseContext.float16OpaqueCheck($1.loc, "half float texture", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtFloat16, EsdCube);
    }
    | TEXTURE1DARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtFloat, Esd1D, true);
    }
    | F16TEXTURE1DARRAY {
        parseContext.float16OpaqueCheck($1.loc, "half float texture", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtFloat16, Esd1D, true);
    }
    | F16TEXTURE2DARRAY {
        parseContext.float16OpaqueCheck($1.loc, "half float texture", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtFloat16, Esd2D, true);
    }
    | F16TEXTURECUBEARRAY {
        parseContext.float16OpaqueCheck($1.loc, "half float texture", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtFloat16, EsdCube, true);
    }
    | ITEXTURE1D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtInt, Esd1D);
    }
    | ITEXTURE1DARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtInt, Esd1D, true);
    }
    | UTEXTURE1D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtUint, Esd1D);
    }
    | UTEXTURE1DARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtUint, Esd1D, true);
    }
    | TEXTURE2DRECT {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtFloat, EsdRect);
    }
    | F16TEXTURE2DRECT {
        parseContext.float16OpaqueCheck($1.loc, "half float texture", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtFloat16, EsdRect);
    }
    | ITEXTURE2DRECT {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtInt, EsdRect);
    }
    | UTEXTURE2DRECT {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtUint, EsdRect);
    }
    | TEXTUREBUFFER {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtFloat, EsdBuffer);
    }
    | F16TEXTUREBUFFER {
        parseContext.float16OpaqueCheck($1.loc, "half float texture", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtFloat16, EsdBuffer);
    }
    | ITEXTUREBUFFER {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtInt, EsdBuffer);
    }
    | UTEXTUREBUFFER {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtUint, EsdBuffer);
    }
    | TEXTURE2DMS {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtFloat, Esd2D, false, false, true);
    }
    | F16TEXTURE2DMS {
        parseContext.float16OpaqueCheck($1.loc, "half float texture", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtFloat16, Esd2D, false, false, true);
    }
    | ITEXTURE2DMS {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtInt, Esd2D, false, false, true);
    }
    | UTEXTURE2DMS {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtUint, Esd2D, false, false, true);
    }
    | TEXTURE2DMSARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtFloat, Esd2D, true, false, true);
    }
    | F16TEXTURE2DMSARRAY {
        parseContext.float16OpaqueCheck($1.loc, "half float texture", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtFloat16, Esd2D, true, false, true);
    }
    | ITEXTURE2DMSARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtInt, Esd2D, true, false, true);
    }
    | UTEXTURE2DMSARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setTexture(EbtUint, Esd2D, true, false, true);
    }
    | IMAGE1D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtFloat, Esd1D);
    }
    | F16IMAGE1D {
        parseContext.float16OpaqueCheck($1.loc, "half float image", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtFloat16, Esd1D);
    }
    | IIMAGE1D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtInt, Esd1D);
    }
    | UIMAGE1D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtUint, Esd1D);
    }
    | IMAGE2D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtFloat, Esd2D);
    }
    | F16IMAGE2D {
        parseContext.float16OpaqueCheck($1.loc, "half float image", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtFloat16, Esd2D);
    }
    | IIMAGE2D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtInt, Esd2D);
    }
    | UIMAGE2D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtUint, Esd2D);
    }
    | IMAGE3D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtFloat, Esd3D);
    }
    | F16IMAGE3D {
        parseContext.float16OpaqueCheck($1.loc, "half float image", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtFloat16, Esd3D);
    }
    | IIMAGE3D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtInt, Esd3D);
    }
    | UIMAGE3D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtUint, Esd3D);
    }
    | IMAGE2DRECT {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtFloat, EsdRect);
    }
    | F16IMAGE2DRECT {
        parseContext.float16OpaqueCheck($1.loc, "half float image", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtFloat16, EsdRect);
    }
    | IIMAGE2DRECT {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtInt, EsdRect);
    }
    | UIMAGE2DRECT {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtUint, EsdRect);
    }
    | IMAGECUBE {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtFloat, EsdCube);
    }
    | F16IMAGECUBE {
        parseContext.float16OpaqueCheck($1.loc, "half float image", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtFloat16, EsdCube);
    }
    | IIMAGECUBE {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtInt, EsdCube);
    }
    | UIMAGECUBE {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtUint, EsdCube);
    }
    | IMAGEBUFFER {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtFloat, EsdBuffer);
    }
    | F16IMAGEBUFFER {
        parseContext.float16OpaqueCheck($1.loc, "half float image", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtFloat16, EsdBuffer);
    }
    | IIMAGEBUFFER {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtInt, EsdBuffer);
    }
    | UIMAGEBUFFER {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtUint, EsdBuffer);
    }
    | IMAGE1DARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtFloat, Esd1D, true);
    }
    | F16IMAGE1DARRAY {
        parseContext.float16OpaqueCheck($1.loc, "half float image", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtFloat16, Esd1D, true);
    }
    | IIMAGE1DARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtInt, Esd1D, true);
    }
    | UIMAGE1DARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtUint, Esd1D, true);
    }
    | IMAGE2DARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtFloat, Esd2D, true);
    }
    | F16IMAGE2DARRAY {
        parseContext.float16OpaqueCheck($1.loc, "half float image", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtFloat16, Esd2D, true);
    }
    | IIMAGE2DARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtInt, Esd2D, true);
    }
    | UIMAGE2DARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtUint, Esd2D, true);
    }
    | IMAGECUBEARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtFloat, EsdCube, true);
    }
    | F16IMAGECUBEARRAY {
        parseContext.float16OpaqueCheck($1.loc, "half float image", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtFloat16, EsdCube, true);
    }
    | IIMAGECUBEARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtInt, EsdCube, true);
    }
    | UIMAGECUBEARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtUint, EsdCube, true);
    }
    | IMAGE2DMS {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtFloat, Esd2D, false, false, true);
    }
    | F16IMAGE2DMS {
        parseContext.float16OpaqueCheck($1.loc, "half float image", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtFloat16, Esd2D, false, false, true);
    }
    | IIMAGE2DMS {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtInt, Esd2D, false, false, true);
    }
    | UIMAGE2DMS {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtUint, Esd2D, false, false, true);
    }
    | IMAGE2DMSARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtFloat, Esd2D, true, false, true);
    }
    | F16IMAGE2DMSARRAY {
        parseContext.float16OpaqueCheck($1.loc, "half float image", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtFloat16, Esd2D, true, false, true);
    }
    | IIMAGE2DMSARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtInt, Esd2D, true, false, true);
    }
    | UIMAGE2DMSARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtUint, Esd2D, true, false, true);
    }
    | I64IMAGE1D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtInt64, Esd1D);
    }
    | U64IMAGE1D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtUint64, Esd1D);
    }
    | I64IMAGE2D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtInt64, Esd2D);
    }
    | U64IMAGE2D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtUint64, Esd2D);
    }
    | I64IMAGE3D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtInt64, Esd3D);
    }
    | U64IMAGE3D {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtUint64, Esd3D);
    }
    | I64IMAGE2DRECT {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtInt64, EsdRect);
    }
    | U64IMAGE2DRECT {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtUint64, EsdRect);
    }
    | I64IMAGECUBE {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtInt64, EsdCube);
    }
    | U64IMAGECUBE {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtUint64, EsdCube);
    }
    | I64IMAGEBUFFER {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtInt64, EsdBuffer);
    }
    | U64IMAGEBUFFER {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtUint64, EsdBuffer);
    }
    | I64IMAGE1DARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtInt64, Esd1D, true);
    }
    | U64IMAGE1DARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtUint64, Esd1D, true);
    }
    | I64IMAGE2DARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtInt64, Esd2D, true);
    }
    | U64IMAGE2DARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtUint64, Esd2D, true);
    }
    | I64IMAGECUBEARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtInt64, EsdCube, true);
    }
    | U64IMAGECUBEARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtUint64, EsdCube, true);
    }
    | I64IMAGE2DMS {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtInt64, Esd2D, false, false, true);
    }
    | U64IMAGE2DMS {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtUint64, Esd2D, false, false, true);
    }
    | I64IMAGE2DMSARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtInt64, Esd2D, true, false, true);
    }
    | U64IMAGE2DMSARRAY {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setImage(EbtUint64, Esd2D, true, false, true);
    }
    | SAMPLEREXTERNALOES {  // GL_OES_EGL_image_external
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat, Esd2D);
        $$.sampler.external = true;
    }
    | SAMPLEREXTERNAL2DY2YEXT { // GL_EXT_YUV_target
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.set(EbtFloat, Esd2D);
        $$.sampler.yuv = true;
    }
    | SUBPASSINPUT {
        parseContext.requireStage($1.loc, EShLangFragment, "subpass input");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setSubpass(EbtFloat);
    }
    | SUBPASSINPUTMS {
        parseContext.requireStage($1.loc, EShLangFragment, "subpass input");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setSubpass(EbtFloat, true);
    }
    | F16SUBPASSINPUT {
        parseContext.float16OpaqueCheck($1.loc, "half float subpass input", parseContext.symbolTable.atBuiltInLevel());
        parseContext.requireStage($1.loc, EShLangFragment, "subpass input");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setSubpass(EbtFloat16);
    }
    | F16SUBPASSINPUTMS {
        parseContext.float16OpaqueCheck($1.loc, "half float subpass input", parseContext.symbolTable.atBuiltInLevel());
        parseContext.requireStage($1.loc, EShLangFragment, "subpass input");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setSubpass(EbtFloat16, true);
    }
    | ISUBPASSINPUT {
        parseContext.requireStage($1.loc, EShLangFragment, "subpass input");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setSubpass(EbtInt);
    }
    | ISUBPASSINPUTMS {
        parseContext.requireStage($1.loc, EShLangFragment, "subpass input");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setSubpass(EbtInt, true);
    }
    | USUBPASSINPUT {
        parseContext.requireStage($1.loc, EShLangFragment, "subpass input");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setSubpass(EbtUint);
    }
    | USUBPASSINPUTMS {
        parseContext.requireStage($1.loc, EShLangFragment, "subpass input");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtSampler;
        $$.sampler.setSubpass(EbtUint, true);
    }
    | FCOOPMATNV {
        parseContext.fcoopmatCheck($1.loc, "fcoopmatNV", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtFloat;
        $$.coopmat = true;
    }
    | ICOOPMATNV {
        parseContext.intcoopmatCheck($1.loc, "icoopmatNV", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtInt;
        $$.coopmat = true;
    }
    | UCOOPMATNV {
        parseContext.intcoopmatCheck($1.loc, "ucoopmatNV", parseContext.symbolTable.atBuiltInLevel());
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.basicType = EbtUint;
        $$.coopmat = true;
    }
    | spirv_type_specifier {
        parseContext.requireExtensions($1.loc, 1, &E_GL_EXT_spirv_intrinsics, "SPIR-V type specifier");
        $$ = $1;
    }

    | struct_specifier {
        $$ = $1;
        $$.qualifier.storage = parseContext.symbolTable.atGlobalLevel() ? EvqGlobal : EvqTemporary;
        parseContext.structTypeCheck($$.loc, $$);
    }
    | TYPE_NAME {
        //
        // This is for user defined type names.  The lexical phase looked up the
        // type.
        //
        if (const TVariable* variable = ($1.symbol)->getAsVariable()) {
            const TType& structure = variable->getType();
            $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
            $$.basicType = EbtStruct;
            $$.userDef = &structure;
        } else
            parseContext.error($1.loc, "expected type name", $1.string->c_str(), "");
    }
    ;

precision_qualifier
    : HIGH_PRECISION {
        parseContext.profileRequires($1.loc, ENoProfile, 130, 0, "highp precision qualifier");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        parseContext.handlePrecisionQualifier($1.loc, $$.qualifier, EpqHigh);
    }
    | MEDIUM_PRECISION {
        parseContext.profileRequires($1.loc, ENoProfile, 130, 0, "mediump precision qualifier");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        parseContext.handlePrecisionQualifier($1.loc, $$.qualifier, EpqMedium);
    }
    | LOW_PRECISION {
        parseContext.profileRequires($1.loc, ENoProfile, 130, 0, "lowp precision qualifier");
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        parseContext.handlePrecisionQualifier($1.loc, $$.qualifier, EpqLow);
    }
    ;

struct_specifier
    : STRUCT IDENTIFIER LEFT_BRACE { parseContext.nestedStructCheck($1.loc); } struct_declaration_list RIGHT_BRACE {
        TType* structure = new TType($5, *$2.string);
        parseContext.structArrayCheck($2.loc, *structure);
        TVariable* userTypeDef = new TVariable($2.string, *structure, true);
        if (! parseContext.symbolTable.insert(*userTypeDef))
            parseContext.error($2.loc, "redefinition", $2.string->c_str(), "struct");
        $$.init($1.loc);
        $$.basicType = EbtStruct;
        $$.userDef = structure;
        --parseContext.structNestingLevel;
    }
    | STRUCT LEFT_BRACE { parseContext.nestedStructCheck($1.loc); } struct_declaration_list RIGHT_BRACE {
        TType* structure = new TType($4, TString(""));
        $$.init($1.loc);
        $$.basicType = EbtStruct;
        $$.userDef = structure;
        --parseContext.structNestingLevel;
    }
    ;

struct_declaration_list
    : struct_declaration {
        $$ = $1;
    }
    | struct_declaration_list struct_declaration {
        $$ = $1;
        for (unsigned int i = 0; i < $2->size(); ++i) {
            for (unsigned int j = 0; j < $$->size(); ++j) {
                if ((*$$)[j].type->getFieldName() == (*$2)[i].type->getFieldName())
                    parseContext.error((*$2)[i].loc, "duplicate member name:", "", (*$2)[i].type->getFieldName().c_str());
            }
            $$->push_back((*$2)[i]);
        }
    }
    ;

struct_declaration
    : type_specifier struct_declarator_list SEMICOLON {
        if ($1.arraySizes) {
            parseContext.profileRequires($1.loc, ENoProfile, 120, E_GL_3DL_array_objects, "arrayed type");
            parseContext.profileRequires($1.loc, EEsProfile, 300, 0, "arrayed type");
            if (parseContext.isEsProfile())
                parseContext.arraySizeRequiredCheck($1.loc, *$1.arraySizes);
        }

        $$ = $2;

        parseContext.voidErrorCheck($1.loc, (*$2)[0].type->getFieldName(), $1.basicType);
        parseContext.precisionQualifierCheck($1.loc, $1.basicType, $1.qualifier);

        for (unsigned int i = 0; i < $$->size(); ++i) {
            TType type($1);
            type.setFieldName((*$$)[i].type->getFieldName());
            type.transferArraySizes((*$$)[i].type->getArraySizes());
            type.copyArrayInnerSizes($1.arraySizes);
            parseContext.arrayOfArrayVersionCheck((*$$)[i].loc, type.getArraySizes());
            (*$$)[i].type->shallowCopy(type);
        }
    }
    | type_qualifier type_specifier struct_declarator_list SEMICOLON {
        if ($2.arraySizes) {
            parseContext.profileRequires($2.loc, ENoProfile, 120, E_GL_3DL_array_objects, "arrayed type");
            parseContext.profileRequires($2.loc, EEsProfile, 300, 0, "arrayed type");
            if (parseContext.isEsProfile())
                parseContext.arraySizeRequiredCheck($2.loc, *$2.arraySizes);
        }

        $$ = $3;

        parseContext.memberQualifierCheck($1);
        parseContext.voidErrorCheck($2.loc, (*$3)[0].type->getFieldName(), $2.basicType);
        parseContext.mergeQualifiers($2.loc, $2.qualifier, $1.qualifier, true);
        parseContext.precisionQualifierCheck($2.loc, $2.basicType, $2.qualifier);

        for (unsigned int i = 0; i < $$->size(); ++i) {
            TType type($2);
            type.setFieldName((*$$)[i].type->getFieldName());
            type.transferArraySizes((*$$)[i].type->getArraySizes());
            type.copyArrayInnerSizes($2.arraySizes);
            parseContext.arrayOfArrayVersionCheck((*$$)[i].loc, type.getArraySizes());
            (*$$)[i].type->shallowCopy(type);
        }
    }
    ;

struct_declarator_list
    : struct_declarator {
        $$ = new TTypeList;
        $$->push_back($1);
    }
    | struct_declarator_list COMMA struct_declarator {
        $$->push_back($3);
    }
    ;

struct_declarator
    : IDENTIFIER {
        $$.type = new TType(EbtVoid);
        $$.loc = $1.loc;
        $$.type->setFieldName(*$1.string);
    }
    | IDENTIFIER array_specifier {
        parseContext.arrayOfArrayVersionCheck($1.loc, $2.arraySizes);

        $$.type = new TType(EbtVoid);
        $$.loc = $1.loc;
        $$.type->setFieldName(*$1.string);
        $$.type->transferArraySizes($2.arraySizes);
    }
    ;

initializer
    : assignment_expression {
        $$ = $1;
    }

    | LEFT_BRACE initializer_list RIGHT_BRACE {
        const char* initFeature = "{ } style initializers";
        parseContext.requireProfile($1.loc, ~EEsProfile, initFeature);
        parseContext.profileRequires($1.loc, ~EEsProfile, 420, E_GL_ARB_shading_language_420pack, initFeature);
        $$ = $2;
    }
    | LEFT_BRACE initializer_list COMMA RIGHT_BRACE {
        const char* initFeature = "{ } style initializers";
        parseContext.requireProfile($1.loc, ~EEsProfile, initFeature);
        parseContext.profileRequires($1.loc, ~EEsProfile, 420, E_GL_ARB_shading_language_420pack, initFeature);
        $$ = $2;
    }
    | LEFT_BRACE RIGHT_BRACE {
        const char* initFeature = "empty { } initializer";
        parseContext.profileRequires($1.loc, EEsProfile, 0, E_GL_EXT_null_initializer, initFeature);
        parseContext.profileRequires($1.loc, ~EEsProfile, 0, E_GL_EXT_null_initializer, initFeature);
        $$ = parseContext.intermediate.makeAggregate($1.loc);
    }

    ;


initializer_list
    : initializer {
        $$ = parseContext.intermediate.growAggregate(0, $1, $1->getLoc());
    }
    | initializer_list COMMA initializer {
        $$ = parseContext.intermediate.growAggregate($1, $3);
    }
    ;


declaration_statement
    : declaration { $$ = $1; }
    ;

statement
    : compound_statement  { $$ = $1; }
    | simple_statement    { $$ = $1; }
    ;

// Grammar Note:  labeled statements for switch statements only; 'goto' is not supported.

simple_statement
    : declaration_statement { $$ = $1; }
    | expression_statement  { $$ = $1; }
    | selection_statement   { $$ = $1; }
    | switch_statement      { $$ = $1; }
    | case_label            { $$ = $1; }
    | iteration_statement   { $$ = $1; }
    | jump_statement        { $$ = $1; }

    | demote_statement      { $$ = $1; }

    ;


demote_statement
    : DEMOTE SEMICOLON {
        parseContext.requireStage($1.loc, EShLangFragment, "demote");
        parseContext.requireExtensions($1.loc, 1, &E_GL_EXT_demote_to_helper_invocation, "demote");
        $$ = parseContext.intermediate.addBranch(EOpDemote, $1.loc);
    }
    ;


compound_statement
    : LEFT_BRACE RIGHT_BRACE { $$ = 0; }
    | LEFT_BRACE {
        parseContext.symbolTable.push();
        ++parseContext.statementNestingLevel;
    }
      statement_list {
        parseContext.symbolTable.pop(&parseContext.defaultPrecision[0]);
        --parseContext.statementNestingLevel;
    }
      RIGHT_BRACE {
        if ($3 && $3->getAsAggregate())
            $3->getAsAggregate()->setOperator(EOpSequence);
        $$ = $3;
    }
    ;

statement_no_new_scope
    : compound_statement_no_new_scope { $$ = $1; }
    | simple_statement                { $$ = $1; }
    ;

statement_scoped
    : {
        ++parseContext.controlFlowNestingLevel;
    }
      compound_statement  {
        --parseContext.controlFlowNestingLevel;
        $$ = $2;
    }
    | {
        parseContext.symbolTable.push();
        ++parseContext.statementNestingLevel;
        ++parseContext.controlFlowNestingLevel;
    }
      simple_statement {
        parseContext.symbolTable.pop(&parseContext.defaultPrecision[0]);
        --parseContext.statementNestingLevel;
        --parseContext.controlFlowNestingLevel;
        $$ = $2;
    }

compound_statement_no_new_scope
    // Statement that doesn't create a new scope, for selection_statement, iteration_statement
    : LEFT_BRACE RIGHT_BRACE {
        $$ = 0;
    }
    | LEFT_BRACE statement_list RIGHT_BRACE {
        if ($2 && $2->getAsAggregate())
            $2->getAsAggregate()->setOperator(EOpSequence);
        $$ = $2;
    }
    ;

statement_list
    : statement {
        $$ = parseContext.intermediate.makeAggregate($1);
        if ($1 && $1->getAsBranchNode() && ($1->getAsBranchNode()->getFlowOp() == EOpCase ||
                                            $1->getAsBranchNode()->getFlowOp() == EOpDefault)) {
            parseContext.wrapupSwitchSubsequence(0, $1);
            $$ = 0;  // start a fresh subsequence for what's after this case
        }
    }
    | statement_list statement {
        if ($2 && $2->getAsBranchNode() && ($2->getAsBranchNode()->getFlowOp() == EOpCase ||
                                            $2->getAsBranchNode()->getFlowOp() == EOpDefault)) {
            parseContext.wrapupSwitchSubsequence($1 ? $1->getAsAggregate() : 0, $2);
            $$ = 0;  // start a fresh subsequence for what's after this case
        } else
            $$ = parseContext.intermediate.growAggregate($1, $2);
    }
    ;

expression_statement
    : SEMICOLON  { $$ = 0; }
    | expression SEMICOLON  { $$ = static_cast<TIntermNode*>($1); }
    ;

selection_statement
    : selection_statement_nonattributed {
        $$ = $1;
    }

    | attribute selection_statement_nonattributed {
        parseContext.requireExtensions($2->getLoc(), 1, &E_GL_EXT_control_flow_attributes, "attribute");
        parseContext.handleSelectionAttributes(*$1, $2);
        $$ = $2;
    }


selection_statement_nonattributed
    : IF LEFT_PAREN expression RIGHT_PAREN selection_rest_statement {
        parseContext.boolCheck($1.loc, $3);
        $$ = parseContext.intermediate.addSelection($3, $5, $1.loc);
    }
    ;

selection_rest_statement
    : statement_scoped ELSE statement_scoped {
        $$.node1 = $1;
        $$.node2 = $3;
    }
    | statement_scoped {
        $$.node1 = $1;
        $$.node2 = 0;
    }
    ;

condition
    // In 1996 c++ draft, conditions can include single declarations
    : expression {
        $$ = $1;
        parseContext.boolCheck($1->getLoc(), $1);
    }
    | fully_specified_type IDENTIFIER EQUAL initializer {
        parseContext.boolCheck($2.loc, $1);

        TType type($1);
        TIntermNode* initNode = parseContext.declareVariable($2.loc, *$2.string, $1, 0, $4);
        if (initNode)
            $$ = initNode->getAsTyped();
        else
            $$ = 0;
    }
    ;

switch_statement
    : switch_statement_nonattributed {
        $$ = $1;
    }

    | attribute switch_statement_nonattributed {
        parseContext.requireExtensions($2->getLoc(), 1, &E_GL_EXT_control_flow_attributes, "attribute");
        parseContext.handleSwitchAttributes(*$1, $2);
        $$ = $2;
    }


switch_statement_nonattributed
    : SWITCH LEFT_PAREN expression RIGHT_PAREN {
        // start new switch sequence on the switch stack
        ++parseContext.controlFlowNestingLevel;
        ++parseContext.statementNestingLevel;
        parseContext.switchSequenceStack.push_back(new TIntermSequence);
        parseContext.switchLevel.push_back(parseContext.statementNestingLevel);
        parseContext.symbolTable.push();
    }
    LEFT_BRACE switch_statement_list RIGHT_BRACE {
        $$ = parseContext.addSwitch($1.loc, $3, $7 ? $7->getAsAggregate() : 0);
        delete parseContext.switchSequenceStack.back();
        parseContext.switchSequenceStack.pop_back();
        parseContext.switchLevel.pop_back();
        parseContext.symbolTable.pop(&parseContext.defaultPrecision[0]);
        --parseContext.statementNestingLevel;
        --parseContext.controlFlowNestingLevel;
    }
    ;

switch_statement_list
    : /* nothing */ {
        $$ = 0;
    }
    | statement_list {
        $$ = $1;
    }
    ;

case_label
    : CASE expression COLON {
        $$ = 0;
        if (parseContext.switchLevel.size() == 0)
            parseContext.error($1.loc, "cannot appear outside switch statement", "case", "");
        else if (parseContext.switchLevel.back() != parseContext.statementNestingLevel)
            parseContext.error($1.loc, "cannot be nested inside control flow", "case", "");
        else {
            parseContext.constantValueCheck($2, "case");
            parseContext.integerCheck($2, "case");
            $$ = parseContext.intermediate.addBranch(EOpCase, $2, $1.loc);
        }
    }
    | DEFAULT COLON {
        $$ = 0;
        if (parseContext.switchLevel.size() == 0)
            parseContext.error($1.loc, "cannot appear outside switch statement", "default", "");
        else if (parseContext.switchLevel.back() != parseContext.statementNestingLevel)
            parseContext.error($1.loc, "cannot be nested inside control flow", "default", "");
        else
            $$ = parseContext.intermediate.addBranch(EOpDefault, $1.loc);
    }
    ;

iteration_statement
    : iteration_statement_nonattributed {
        $$ = $1;
    }

    | attribute iteration_statement_nonattributed {
        parseContext.requireExtensions($2->getLoc(), 1, &E_GL_EXT_control_flow_attributes, "attribute");
        parseContext.handleLoopAttributes(*$1, $2);
        $$ = $2;
    }


iteration_statement_nonattributed
    : WHILE LEFT_PAREN {
        if (! parseContext.limits.whileLoops)
            parseContext.error($1.loc, "while loops not available", "limitation", "");
        parseContext.symbolTable.push();
        ++parseContext.loopNestingLevel;
        ++parseContext.statementNestingLevel;
        ++parseContext.controlFlowNestingLevel;
    }
      condition RIGHT_PAREN statement_no_new_scope {
        parseContext.symbolTable.pop(&parseContext.defaultPrecision[0]);
        $$ = parseContext.intermediate.addLoop($6, $4, 0, true, $1.loc);
        --parseContext.loopNestingLevel;
        --parseContext.statementNestingLevel;
        --parseContext.controlFlowNestingLevel;
    }
    | DO {
        ++parseContext.loopNestingLevel;
        ++parseContext.statementNestingLevel;
        ++parseContext.controlFlowNestingLevel;
    }
      statement WHILE LEFT_PAREN expression RIGHT_PAREN SEMICOLON {
        if (! parseContext.limits.whileLoops)
            parseContext.error($1.loc, "do-while loops not available", "limitation", "");

        parseContext.boolCheck($8.loc, $6);

        $$ = parseContext.intermediate.addLoop($3, $6, 0, false, $4.loc);
        --parseContext.loopNestingLevel;
        --parseContext.statementNestingLevel;
        --parseContext.controlFlowNestingLevel;
    }
    | FOR LEFT_PAREN {
        parseContext.symbolTable.push();
        ++parseContext.loopNestingLevel;
        ++parseContext.statementNestingLevel;
        ++parseContext.controlFlowNestingLevel;
    }
      for_init_statement for_rest_statement RIGHT_PAREN statement_no_new_scope {
        parseContext.symbolTable.pop(&parseContext.defaultPrecision[0]);
        $$ = parseContext.intermediate.makeAggregate($4, $2.loc);
        TIntermLoop* forLoop = parseContext.intermediate.addLoop($7, reinterpret_cast<TIntermTyped*>($5.node1), reinterpret_cast<TIntermTyped*>($5.node2), true, $1.loc);
        if (! parseContext.limits.nonInductiveForLoops)
            parseContext.inductiveLoopCheck($1.loc, $4, forLoop);
        $$ = parseContext.intermediate.growAggregate($$, forLoop, $1.loc);
        $$->getAsAggregate()->setOperator(EOpSequence);
        --parseContext.loopNestingLevel;
        --parseContext.statementNestingLevel;
        --parseContext.controlFlowNestingLevel;
    }
    ;

for_init_statement
    : expression_statement {
        $$ = $1;
    }
    | declaration_statement {
        $$ = $1;
    }
    ;

conditionopt
    : condition {
        $$ = $1;
    }
    | /* May be null */ {
        $$ = 0;
    }
    ;

for_rest_statement
    : conditionopt SEMICOLON {
        $$.node1 = $1;
        $$.node2 = 0;
    }
    | conditionopt SEMICOLON expression  {
        $$.node1 = $1;
        $$.node2 = $3;
    }
    ;

jump_statement
    : CONTINUE SEMICOLON {
        if (parseContext.loopNestingLevel <= 0)
            parseContext.error($1.loc, "continue statement only allowed in loops", "", "");
        $$ = parseContext.intermediate.addBranch(EOpContinue, $1.loc);
    }
    | BREAK SEMICOLON {
        if (parseContext.loopNestingLevel + parseContext.switchSequenceStack.size() <= 0)
            parseContext.error($1.loc, "break statement only allowed in switch and loops", "", "");
        $$ = parseContext.intermediate.addBranch(EOpBreak, $1.loc);
    }
    | RETURN SEMICOLON {
        $$ = parseContext.intermediate.addBranch(EOpReturn, $1.loc);
        if (parseContext.currentFunctionType->getBasicType() != EbtVoid)
            parseContext.error($1.loc, "non-void function must return a value", "return", "");
        if (parseContext.inMain)
            parseContext.postEntryPointReturn = true;
    }
    | RETURN expression SEMICOLON {
        $$ = parseContext.handleReturnValue($1.loc, $2);
    }
    | DISCARD SEMICOLON {
        parseContext.requireStage($1.loc, EShLangFragment, "discard");
        $$ = parseContext.intermediate.addBranch(EOpKill, $1.loc);
    }
    | TERMINATE_INVOCATION SEMICOLON {
        parseContext.requireStage($1.loc, EShLangFragment, "terminateInvocation");
        $$ = parseContext.intermediate.addBranch(EOpTerminateInvocation, $1.loc);
    }

    | TERMINATE_RAY SEMICOLON {
        parseContext.requireStage($1.loc, EShLangAnyHit, "terminateRayEXT");
        $$ = parseContext.intermediate.addBranch(EOpTerminateRayKHR, $1.loc);
    }
    | IGNORE_INTERSECTION SEMICOLON {
        parseContext.requireStage($1.loc, EShLangAnyHit, "ignoreIntersectionEXT");
        $$ = parseContext.intermediate.addBranch(EOpIgnoreIntersectionKHR, $1.loc);
    }

    ;

// Grammar Note:  No 'goto'.  Gotos are not supported.

translation_unit
    : external_declaration {
        $$ = $1;
        parseContext.intermediate.setTreeRoot($$);
    }
    | translation_unit external_declaration {
        if ($2 != nullptr) {
            $$ = parseContext.intermediate.growAggregate($1, $2);
            parseContext.intermediate.setTreeRoot($$);
        }
    }
    ;

external_declaration
    : function_definition {
        $$ = $1;
    }
    | declaration {
        $$ = $1;
    }

    | SEMICOLON {
        parseContext.requireProfile($1.loc, ~EEsProfile, "extraneous semicolon");
        parseContext.profileRequires($1.loc, ~EEsProfile, 460, nullptr, "extraneous semicolon");
        $$ = nullptr;
    }

    ;

function_definition
    : function_prototype {
        $1.function = parseContext.handleFunctionDeclarator($1.loc, *$1.function, false /* not prototype */);
        $1.intermNode = parseContext.handleFunctionDefinition($1.loc, *$1.function);

        // For ES 100 only, according to ES shading language 100 spec: A function
        // body has a scope nested inside the function's definition.
        if (parseContext.profile == EEsProfile && parseContext.version == 100)
        {
            parseContext.symbolTable.push();
            ++parseContext.statementNestingLevel;
        }
    }
    compound_statement_no_new_scope {
        //   May be best done as post process phase on intermediate code
        if (parseContext.currentFunctionType->getBasicType() != EbtVoid && ! parseContext.functionReturnsValue)
            parseContext.error($1.loc, "function does not return a value:", "", $1.function->getName().c_str());
        parseContext.symbolTable.pop(&parseContext.defaultPrecision[0]);
        $$ = parseContext.intermediate.growAggregate($1.intermNode, $3);
        parseContext.intermediate.setAggregateOperator($$, EOpFunction, $1.function->getType(), $1.loc);
        $$->getAsAggregate()->setName($1.function->getMangledName().c_str());

        // store the pragma information for debug and optimize and other vendor specific
        // information. This information can be queried from the parse tree
        $$->getAsAggregate()->setOptimize(parseContext.contextPragma.optimize);
        $$->getAsAggregate()->setDebug(parseContext.contextPragma.debug);
        $$->getAsAggregate()->setPragmaTable(parseContext.contextPragma.pragmaTable);

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
    ;


attribute
    : LEFT_BRACKET LEFT_BRACKET attribute_list RIGHT_BRACKET RIGHT_BRACKET {
        $$ = $3;
    }

attribute_list
    : single_attribute {
        $$ = $1;
    }
    | attribute_list COMMA single_attribute {
        $$ = parseContext.mergeAttributes($1, $3);
    }

single_attribute
    : IDENTIFIER {
        $$ = parseContext.makeAttributes(*$1.string);
    }
    | IDENTIFIER LEFT_PAREN constant_expression RIGHT_PAREN {
        $$ = parseContext.makeAttributes(*$1.string, $3);
    }



spirv_requirements_list
    : spirv_requirements_parameter {
        $$ = $1;
    }
    | spirv_requirements_list COMMA spirv_requirements_parameter {
        $$ = parseContext.mergeSpirvRequirements($2.loc, $1, $3);
    }

spirv_requirements_parameter
    : IDENTIFIER EQUAL LEFT_BRACKET spirv_extension_list RIGHT_BRACKET {
        $$ = parseContext.makeSpirvRequirement($2.loc, *$1.string, $4->getAsAggregate(), nullptr);
    }
    | IDENTIFIER EQUAL LEFT_BRACKET spirv_capability_list RIGHT_BRACKET {
        $$ = parseContext.makeSpirvRequirement($2.loc, *$1.string, nullptr, $4->getAsAggregate());
    }

spirv_extension_list
    : STRING_LITERAL {
        $$ = parseContext.intermediate.makeAggregate(parseContext.intermediate.addConstantUnion($1.string, $1.loc, true));
    }
    | spirv_extension_list COMMA STRING_LITERAL {
        $$ = parseContext.intermediate.growAggregate($1, parseContext.intermediate.addConstantUnion($3.string, $3.loc, true));
    }

spirv_capability_list
    : INTCONSTANT {
        $$ = parseContext.intermediate.makeAggregate(parseContext.intermediate.addConstantUnion($1.i, $1.loc, true));
    }
    | spirv_capability_list COMMA INTCONSTANT {
        $$ = parseContext.intermediate.growAggregate($1, parseContext.intermediate.addConstantUnion($3.i, $3.loc, true));
    }

spirv_execution_mode_qualifier
    : SPIRV_EXECUTION_MODE LEFT_PAREN INTCONSTANT RIGHT_PAREN {
        parseContext.intermediate.insertSpirvExecutionMode($3.i);
        $$ = 0;
    }
    | SPIRV_EXECUTION_MODE LEFT_PAREN spirv_requirements_list COMMA INTCONSTANT RIGHT_PAREN {
        parseContext.intermediate.insertSpirvRequirement($3);
        parseContext.intermediate.insertSpirvExecutionMode($5.i);
        $$ = 0;
    }
    | SPIRV_EXECUTION_MODE LEFT_PAREN INTCONSTANT COMMA spirv_execution_mode_parameter_list RIGHT_PAREN {
        parseContext.intermediate.insertSpirvExecutionMode($3.i, $5->getAsAggregate());
        $$ = 0;
    }
    | SPIRV_EXECUTION_MODE LEFT_PAREN spirv_requirements_list COMMA INTCONSTANT COMMA spirv_execution_mode_parameter_list RIGHT_PAREN {
        parseContext.intermediate.insertSpirvRequirement($3);
        parseContext.intermediate.insertSpirvExecutionMode($5.i, $7->getAsAggregate());
        $$ = 0;
    }
    | SPIRV_EXECUTION_MODE_ID LEFT_PAREN INTCONSTANT COMMA spirv_execution_mode_id_parameter_list RIGHT_PAREN {
        parseContext.intermediate.insertSpirvExecutionModeId($3.i, $5->getAsAggregate());
        $$ = 0;
    }
    | SPIRV_EXECUTION_MODE_ID LEFT_PAREN spirv_requirements_list COMMA INTCONSTANT COMMA spirv_execution_mode_id_parameter_list RIGHT_PAREN {
        parseContext.intermediate.insertSpirvRequirement($3);
        parseContext.intermediate.insertSpirvExecutionModeId($5.i, $7->getAsAggregate());
        $$ = 0;
    }

spirv_execution_mode_parameter_list
    : spirv_execution_mode_parameter {
        $$ = parseContext.intermediate.makeAggregate($1);
    }
    | spirv_execution_mode_parameter_list COMMA spirv_execution_mode_parameter {
        $$ = parseContext.intermediate.growAggregate($1, $3);
    }

spirv_execution_mode_parameter
    : FLOATCONSTANT {
        $$ = parseContext.intermediate.addConstantUnion($1.d, EbtFloat, $1.loc, true);
    }
    | INTCONSTANT {
        $$ = parseContext.intermediate.addConstantUnion($1.i, $1.loc, true);
    }
    | UINTCONSTANT {
        $$ = parseContext.intermediate.addConstantUnion($1.u, $1.loc, true);
    }
    | BOOLCONSTANT {
        $$ = parseContext.intermediate.addConstantUnion($1.b, $1.loc, true);
    }
    | STRING_LITERAL {
        $$ = parseContext.intermediate.addConstantUnion($1.string, $1.loc, true);
    }

spirv_execution_mode_id_parameter_list
    : constant_expression {
        if ($1->getBasicType() != EbtFloat &&
            $1->getBasicType() != EbtInt &&
            $1->getBasicType() != EbtUint &&
            $1->getBasicType() != EbtBool &&
            $1->getBasicType() != EbtString)
            parseContext.error($1->getLoc(), "this type not allowed", $1->getType().getBasicString(), "");
        $$ = parseContext.intermediate.makeAggregate($1);
    }
    | spirv_execution_mode_id_parameter_list COMMA constant_expression {
        if ($3->getBasicType() != EbtFloat &&
            $3->getBasicType() != EbtInt &&
            $3->getBasicType() != EbtUint &&
            $3->getBasicType() != EbtBool &&
            $3->getBasicType() != EbtString)
            parseContext.error($3->getLoc(), "this type not allowed", $3->getType().getBasicString(), "");
        $$ = parseContext.intermediate.growAggregate($1, $3);
    }

spirv_storage_class_qualifier
    : SPIRV_STORAGE_CLASS LEFT_PAREN INTCONSTANT RIGHT_PAREN {
        $$.init($1.loc);
        $$.qualifier.storage = EvqSpirvStorageClass;
        $$.qualifier.spirvStorageClass = $3.i;
    }
    | SPIRV_STORAGE_CLASS LEFT_PAREN spirv_requirements_list COMMA INTCONSTANT RIGHT_PAREN {
        $$.init($1.loc);
        parseContext.intermediate.insertSpirvRequirement($3);
        $$.qualifier.storage = EvqSpirvStorageClass;
        $$.qualifier.spirvStorageClass = $5.i;
    }

spirv_decorate_qualifier
    : SPIRV_DECORATE LEFT_PAREN INTCONSTANT RIGHT_PAREN{
        $$.init($1.loc);
        $$.qualifier.setSpirvDecorate($3.i);
    }
    | SPIRV_DECORATE LEFT_PAREN spirv_requirements_list COMMA INTCONSTANT RIGHT_PAREN{
        $$.init($1.loc);
        parseContext.intermediate.insertSpirvRequirement($3);
        $$.qualifier.setSpirvDecorate($5.i);
    }
    | SPIRV_DECORATE LEFT_PAREN INTCONSTANT COMMA spirv_decorate_parameter_list RIGHT_PAREN {
        $$.init($1.loc);
        $$.qualifier.setSpirvDecorate($3.i, $5->getAsAggregate());
    }
    | SPIRV_DECORATE LEFT_PAREN spirv_requirements_list COMMA INTCONSTANT COMMA spirv_decorate_parameter_list RIGHT_PAREN {
        $$.init($1.loc);
        parseContext.intermediate.insertSpirvRequirement($3);
        $$.qualifier.setSpirvDecorate($5.i, $7->getAsAggregate());
    }
    | SPIRV_DECORATE_ID LEFT_PAREN INTCONSTANT COMMA spirv_decorate_id_parameter_list RIGHT_PAREN {
        $$.init($1.loc);
        $$.qualifier.setSpirvDecorateId($3.i, $5->getAsAggregate());
    }
    | SPIRV_DECORATE_ID LEFT_PAREN spirv_requirements_list COMMA INTCONSTANT COMMA spirv_decorate_id_parameter_list RIGHT_PAREN {
        $$.init($1.loc);
        parseContext.intermediate.insertSpirvRequirement($3);
        $$.qualifier.setSpirvDecorateId($5.i, $7->getAsAggregate());
    }
    | SPIRV_DECORATE_STRING LEFT_PAREN INTCONSTANT COMMA spirv_decorate_string_parameter_list RIGHT_PAREN {
        $$.init($1.loc);
        $$.qualifier.setSpirvDecorateString($3.i, $5->getAsAggregate());
    }
    | SPIRV_DECORATE_STRING LEFT_PAREN spirv_requirements_list COMMA INTCONSTANT COMMA spirv_decorate_string_parameter_list RIGHT_PAREN {
        $$.init($1.loc);
        parseContext.intermediate.insertSpirvRequirement($3);
        $$.qualifier.setSpirvDecorateString($5.i, $7->getAsAggregate());
    }

spirv_decorate_parameter_list
    : spirv_decorate_parameter {
        $$ = parseContext.intermediate.makeAggregate($1);
    }
    | spirv_decorate_parameter_list COMMA spirv_decorate_parameter {
        $$ = parseContext.intermediate.growAggregate($1, $3);
    }

spirv_decorate_parameter
    : FLOATCONSTANT {
        $$ = parseContext.intermediate.addConstantUnion($1.d, EbtFloat, $1.loc, true);
    }
    | INTCONSTANT {
        $$ = parseContext.intermediate.addConstantUnion($1.i, $1.loc, true);
    }
    | UINTCONSTANT {
        $$ = parseContext.intermediate.addConstantUnion($1.u, $1.loc, true);
    }
    | BOOLCONSTANT {
        $$ = parseContext.intermediate.addConstantUnion($1.b, $1.loc, true);
    }

spirv_decorate_id_parameter_list
    : constant_expression {
        if ($1->getBasicType() != EbtFloat &&
            $1->getBasicType() != EbtInt &&
            $1->getBasicType() != EbtUint &&
            $1->getBasicType() != EbtBool)
            parseContext.error($1->getLoc(), "this type not allowed", $1->getType().getBasicString(), "");
        $$ = parseContext.intermediate.makeAggregate($1);
    }
    | spirv_decorate_id_parameter_list COMMA constant_expression {
        if ($3->getBasicType() != EbtFloat &&
            $3->getBasicType() != EbtInt &&
            $3->getBasicType() != EbtUint &&
            $3->getBasicType() != EbtBool)
            parseContext.error($3->getLoc(), "this type not allowed", $3->getType().getBasicString(), "");
        $$ = parseContext.intermediate.growAggregate($1, $3);
    }

spirv_decorate_string_parameter_list
    : STRING_LITERAL {
        $$ = parseContext.intermediate.makeAggregate(
            parseContext.intermediate.addConstantUnion($1.string, $1.loc, true));
    }
    | spirv_decorate_string_parameter_list COMMA STRING_LITERAL {
        $$ = parseContext.intermediate.growAggregate($1, parseContext.intermediate.addConstantUnion($3.string, $3.loc, true));
    }

spirv_type_specifier
    : SPIRV_TYPE LEFT_PAREN spirv_instruction_qualifier_list COMMA spirv_type_parameter_list RIGHT_PAREN {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.setSpirvType(*$3, $5);
    }
    | SPIRV_TYPE LEFT_PAREN spirv_requirements_list COMMA spirv_instruction_qualifier_list COMMA spirv_type_parameter_list RIGHT_PAREN {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        parseContext.intermediate.insertSpirvRequirement($3);
        $$.setSpirvType(*$5, $7);
    }
    | SPIRV_TYPE LEFT_PAREN spirv_instruction_qualifier_list RIGHT_PAREN {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        $$.setSpirvType(*$3);
    }
    | SPIRV_TYPE LEFT_PAREN spirv_requirements_list COMMA spirv_instruction_qualifier_list RIGHT_PAREN {
        $$.init($1.loc, parseContext.symbolTable.atGlobalLevel());
        parseContext.intermediate.insertSpirvRequirement($3);
        $$.setSpirvType(*$5);
    }

spirv_type_parameter_list
    : spirv_type_parameter {
        $$ = $1;
    }
    | spirv_type_parameter_list COMMA spirv_type_parameter {
        $$ = parseContext.mergeSpirvTypeParameters($1, $3);
    }

spirv_type_parameter
    : constant_expression {
        $$ = parseContext.makeSpirvTypeParameters($1->getLoc(), $1->getAsConstantUnion());
    }
    | type_specifier {
        $$ = parseContext.makeSpirvTypeParameters($1);
    }

spirv_instruction_qualifier
    : SPIRV_INSTRUCTION LEFT_PAREN spirv_instruction_qualifier_list RIGHT_PAREN {
        $$ = $3;
    }
    | SPIRV_INSTRUCTION LEFT_PAREN spirv_requirements_list COMMA spirv_instruction_qualifier_list RIGHT_PAREN {
        parseContext.intermediate.insertSpirvRequirement($3);
        $$ = $5;
    }

spirv_instruction_qualifier_list
    : spirv_instruction_qualifier_id {
        $$ = $1;
    }
    | spirv_instruction_qualifier_list COMMA spirv_instruction_qualifier_id {
        $$ = parseContext.mergeSpirvInstruction($2.loc, $1, $3);
    }

spirv_instruction_qualifier_id
    : IDENTIFIER EQUAL STRING_LITERAL {
        $$ = parseContext.makeSpirvInstruction($2.loc, *$1.string, *$3.string);
    }
    | IDENTIFIER EQUAL INTCONSTANT {
        $$ = parseContext.makeSpirvInstruction($2.loc, *$1.string, $3.i);
    }


%%
