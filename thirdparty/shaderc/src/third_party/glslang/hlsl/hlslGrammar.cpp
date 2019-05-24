//
// Copyright (C) 2016-2018 Google, Inc.
// Copyright (C) 2016 LunarG, Inc.
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
//    Neither the name of Google, Inc., nor the names of its
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
// This is a set of mutually recursive methods implementing the HLSL grammar.
// Generally, each returns
//  - through an argument: a type specifically appropriate to which rule it
//    recognized
//  - through the return value: true/false to indicate whether or not it
//    recognized its rule
//
// As much as possible, only grammar recognition should happen in this file,
// with all other work being farmed out to hlslParseHelper.cpp, which in turn
// will build the AST.
//
// The next token, yet to be "accepted" is always sitting in 'token'.
// When a method says it accepts a rule, that means all tokens involved
// in the rule will have been consumed, and none left in 'token'.
//

#include "hlslTokens.h"
#include "hlslGrammar.h"
#include "hlslAttributes.h"

namespace glslang {

// Root entry point to this recursive decent parser.
// Return true if compilation unit was successfully accepted.
bool HlslGrammar::parse()
{
    advanceToken();
    return acceptCompilationUnit();
}

void HlslGrammar::expected(const char* syntax)
{
    parseContext.error(token.loc, "Expected", syntax, "");
}

void HlslGrammar::unimplemented(const char* error)
{
    parseContext.error(token.loc, "Unimplemented", error, "");
}

// IDENTIFIER
// THIS
// type that can be used as IDENTIFIER
//
// Only process the next token if it is an identifier.
// Return true if it was an identifier.
bool HlslGrammar::acceptIdentifier(HlslToken& idToken)
{
    // IDENTIFIER
    if (peekTokenClass(EHTokIdentifier)) {
        idToken = token;
        advanceToken();
        return true;
    }

    // THIS
    // -> maps to the IDENTIFIER spelled with the internal special name for 'this'
    if (peekTokenClass(EHTokThis)) {
        idToken = token;
        advanceToken();
        idToken.tokenClass = EHTokIdentifier;
        idToken.string = NewPoolTString(intermediate.implicitThisName);
        return true;
    }

    // type that can be used as IDENTIFIER

    // Even though "sample", "bool", "float", etc keywords (for types, interpolation modifiers),
    // they ARE still accepted as identifiers.  This is not a dense space: e.g, "void" is not a
    // valid identifier, nor is "linear".  This code special cases the known instances of this, so
    // e.g, "int sample;" or "float float;" is accepted.  Other cases can be added here if needed.

    const char* idString = getTypeString(peek());
    if (idString == nullptr)
        return false;

    token.string     = NewPoolTString(idString);
    token.tokenClass = EHTokIdentifier;
    idToken = token;
    typeIdentifiers = true;

    advanceToken();

    return true;
}

// compilationUnit
//      : declaration_list EOF
//
bool HlslGrammar::acceptCompilationUnit()
{
    if (! acceptDeclarationList(unitNode))
        return false;

    if (! peekTokenClass(EHTokNone))
        return false;

    // set root of AST
    if (unitNode && !unitNode->getAsAggregate())
        unitNode = intermediate.growAggregate(nullptr, unitNode);
    intermediate.setTreeRoot(unitNode);

    return true;
}

// Recognize the following, but with the extra condition that it can be
// successfully terminated by EOF or '}'.
//
// declaration_list
//      : list of declaration_or_semicolon followed by EOF or RIGHT_BRACE
//
// declaration_or_semicolon
//      : declaration
//      : SEMICOLON
//
bool HlslGrammar::acceptDeclarationList(TIntermNode*& nodeList)
{
    do {
        // HLSL allows extra semicolons between global declarations
        do { } while (acceptTokenClass(EHTokSemicolon));

        // EOF or RIGHT_BRACE
        if (peekTokenClass(EHTokNone) || peekTokenClass(EHTokRightBrace))
            return true;

        // declaration
        if (! acceptDeclaration(nodeList))
            return false;
    } while (true);

    return true;
}

// sampler_state
//      : LEFT_BRACE [sampler_state_assignment ... ] RIGHT_BRACE
//
// sampler_state_assignment
//     : sampler_state_identifier EQUAL value SEMICOLON
//
// sampler_state_identifier
//     : ADDRESSU
//     | ADDRESSV
//     | ADDRESSW
//     | BORDERCOLOR
//     | FILTER
//     | MAXANISOTROPY
//     | MAXLOD
//     | MINLOD
//     | MIPLODBIAS
//
bool HlslGrammar::acceptSamplerState()
{
    // TODO: this should be genericized to accept a list of valid tokens and
    // return token/value pairs.  Presently it is specific to texture values.

    if (! acceptTokenClass(EHTokLeftBrace))
        return true;

    parseContext.warn(token.loc, "unimplemented", "immediate sampler state", "");

    do {
        // read state name
        HlslToken state;
        if (! acceptIdentifier(state))
            break;  // end of list

        // FXC accepts any case
        TString stateName = *state.string;
        std::transform(stateName.begin(), stateName.end(), stateName.begin(), ::tolower);

        if (! acceptTokenClass(EHTokAssign)) {
            expected("assign");
            return false;
        }

        if (stateName == "minlod" || stateName == "maxlod") {
            if (! peekTokenClass(EHTokIntConstant)) {
                expected("integer");
                return false;
            }

            TIntermTyped* lod = nullptr;
            if (! acceptLiteral(lod))  // should never fail, since we just looked for an integer
                return false;
        } else if (stateName == "maxanisotropy") {
            if (! peekTokenClass(EHTokIntConstant)) {
                expected("integer");
                return false;
            }

            TIntermTyped* maxAnisotropy = nullptr;
            if (! acceptLiteral(maxAnisotropy))  // should never fail, since we just looked for an integer
                return false;
        } else if (stateName == "filter") {
            HlslToken filterMode;
            if (! acceptIdentifier(filterMode)) {
                expected("filter mode");
                return false;
            }
        } else if (stateName == "addressu" || stateName == "addressv" || stateName == "addressw") {
            HlslToken addrMode;
            if (! acceptIdentifier(addrMode)) {
                expected("texture address mode");
                return false;
            }
        } else if (stateName == "miplodbias") {
            TIntermTyped* lodBias = nullptr;
            if (! acceptLiteral(lodBias)) {
                expected("lod bias");
                return false;
            }
        } else if (stateName == "bordercolor") {
            return false;
        } else {
            expected("texture state");
            return false;
        }

        // SEMICOLON
        if (! acceptTokenClass(EHTokSemicolon)) {
            expected("semicolon");
            return false;
        }
    } while (true);

    if (! acceptTokenClass(EHTokRightBrace))
        return false;

    return true;
}

// sampler_declaration_dx9
//    : SAMPLER identifier EQUAL sampler_type sampler_state
//
bool HlslGrammar::acceptSamplerDeclarationDX9(TType& /*type*/)
{
    if (! acceptTokenClass(EHTokSampler))
        return false;

    // TODO: remove this when DX9 style declarations are implemented.
    unimplemented("Direct3D 9 sampler declaration");

    // read sampler name
    HlslToken name;
    if (! acceptIdentifier(name)) {
        expected("sampler name");
        return false;
    }

    if (! acceptTokenClass(EHTokAssign)) {
        expected("=");
        return false;
    }

    return false;
}

// declaration
//      : attributes attributed_declaration
//      | NAMESPACE IDENTIFIER LEFT_BRACE declaration_list RIGHT_BRACE
//
// attributed_declaration
//      : sampler_declaration_dx9 post_decls SEMICOLON
//      | fully_specified_type                           // for cbuffer/tbuffer
//      | fully_specified_type declarator_list SEMICOLON // for non cbuffer/tbuffer
//      | fully_specified_type identifier function_parameters post_decls compound_statement  // function definition
//      | fully_specified_type identifier sampler_state post_decls compound_statement        // sampler definition
//      | typedef declaration
//
// declarator_list
//      : declarator COMMA declarator COMMA declarator...  // zero or more declarators
//
// declarator
//      : identifier array_specifier post_decls
//      | identifier array_specifier post_decls EQUAL assignment_expression
//      | identifier function_parameters post_decls                                          // function prototype
//
// Parsing has to go pretty far in to know whether it's a variable, prototype, or
// function definition, so the implementation below doesn't perfectly divide up the grammar
// as above.  (The 'identifier' in the first item in init_declarator list is the
// same as 'identifier' for function declarations.)
//
// This can generate more than one subtree, one per initializer or a function body.
// All initializer subtrees are put in their own aggregate node, making one top-level
// node for all the initializers. Each function created is a top-level node to grow
// into the passed-in nodeList.
//
// If 'nodeList' is passed in as non-null, it must be an aggregate to extend for
// each top-level node the declaration creates. Otherwise, if only one top-level
// node in generated here, that is want is returned in nodeList.
//
bool HlslGrammar::acceptDeclaration(TIntermNode*& nodeList)
{
    // NAMESPACE IDENTIFIER LEFT_BRACE declaration_list RIGHT_BRACE
    if (acceptTokenClass(EHTokNamespace)) {
        HlslToken namespaceToken;
        if (!acceptIdentifier(namespaceToken)) {
            expected("namespace name");
            return false;
        }
        parseContext.pushNamespace(*namespaceToken.string);
        if (!acceptTokenClass(EHTokLeftBrace)) {
            expected("{");
            return false;
        }
        if (!acceptDeclarationList(nodeList)) {
            expected("declaration list");
            return false;
        }
        if (!acceptTokenClass(EHTokRightBrace)) {
            expected("}");
            return false;
        }
        parseContext.popNamespace();
        return true;
    }

    bool declarator_list = false; // true when processing comma separation

    // attributes
    TFunctionDeclarator declarator;
    acceptAttributes(declarator.attributes);

    // typedef
    bool typedefDecl = acceptTokenClass(EHTokTypedef);

    TType declaredType;

    // DX9 sampler declaration use a different syntax
    // DX9 shaders need to run through HLSL compiler (fxc) via a back compat mode, it isn't going to
    // be possible to simultaneously compile D3D10+ style shaders and DX9 shaders. If we want to compile DX9
    // HLSL shaders, this will have to be a master level switch
    // As such, the sampler keyword in D3D10+ turns into an automatic sampler type, and is commonly used
    // For that reason, this line is commented out
    // if (acceptSamplerDeclarationDX9(declaredType))
    //     return true;

    bool forbidDeclarators = (peekTokenClass(EHTokCBuffer) || peekTokenClass(EHTokTBuffer));
    // fully_specified_type
    if (! acceptFullySpecifiedType(declaredType, nodeList, declarator.attributes, forbidDeclarators))
        return false;

    // cbuffer and tbuffer end with the closing '}'.
    // No semicolon is included.
    if (forbidDeclarators)
        return true;

    // declarator_list
    //    : declarator
    //         : identifier
    HlslToken idToken;
    TIntermAggregate* initializers = nullptr;
    while (acceptIdentifier(idToken)) {
        TString *fullName = idToken.string;
        if (parseContext.symbolTable.atGlobalLevel())
            parseContext.getFullNamespaceName(fullName);
        if (peekTokenClass(EHTokLeftParen)) {
            // looks like function parameters

            // merge in the attributes into the return type
            parseContext.transferTypeAttributes(token.loc, declarator.attributes, declaredType, true);

            // Potentially rename shader entry point function.  No-op most of the time.
            parseContext.renameShaderFunction(fullName);

            // function_parameters
            declarator.function = new TFunction(fullName, declaredType);
            if (!acceptFunctionParameters(*declarator.function)) {
                expected("function parameter list");
                return false;
            }

            // post_decls
            acceptPostDecls(declarator.function->getWritableType().getQualifier());

            // compound_statement (function body definition) or just a prototype?
            declarator.loc = token.loc;
            if (peekTokenClass(EHTokLeftBrace)) {
                if (declarator_list)
                    parseContext.error(idToken.loc, "function body can't be in a declarator list", "{", "");
                if (typedefDecl)
                    parseContext.error(idToken.loc, "function body can't be in a typedef", "{", "");
                return acceptFunctionDefinition(declarator, nodeList, nullptr);
            } else {
                if (typedefDecl)
                    parseContext.error(idToken.loc, "function typedefs not implemented", "{", "");
                parseContext.handleFunctionDeclarator(declarator.loc, *declarator.function, true);
            }
        } else {
            // A variable declaration.

            // merge in the attributes, the first time around, into the shared type
            if (! declarator_list)
                parseContext.transferTypeAttributes(token.loc, declarator.attributes, declaredType);

            // Fix the storage qualifier if it's a global.
            if (declaredType.getQualifier().storage == EvqTemporary && parseContext.symbolTable.atGlobalLevel())
                declaredType.getQualifier().storage = EvqUniform;

            // recognize array_specifier
            TArraySizes* arraySizes = nullptr;
            acceptArraySpecifier(arraySizes);

            // We can handle multiple variables per type declaration, so
            // the number of types can expand when arrayness is different.
            TType variableType;
            variableType.shallowCopy(declaredType);

            // In the most general case, arrayness is potentially coming both from the
            // declared type and from the variable: "int[] a[];" or just one or the other.
            // Merge it all to the variableType, so all arrayness is part of the variableType.
            variableType.transferArraySizes(arraySizes);
            variableType.copyArrayInnerSizes(declaredType.getArraySizes());

            // samplers accept immediate sampler state
            if (variableType.getBasicType() == EbtSampler) {
                if (! acceptSamplerState())
                    return false;
            }

            // post_decls
            acceptPostDecls(variableType.getQualifier());

            // EQUAL assignment_expression
            TIntermTyped* expressionNode = nullptr;
            if (acceptTokenClass(EHTokAssign)) {
                if (typedefDecl)
                    parseContext.error(idToken.loc, "can't have an initializer", "typedef", "");
                if (! acceptAssignmentExpression(expressionNode)) {
                    expected("initializer");
                    return false;
                }
            }

            // TODO: things scoped within an annotation need their own name space;
            // TODO: strings are not yet handled.
            if (variableType.getBasicType() != EbtString && parseContext.getAnnotationNestingLevel() == 0) {
                if (typedefDecl)
                    parseContext.declareTypedef(idToken.loc, *fullName, variableType);
                else if (variableType.getBasicType() == EbtBlock) {
                    if (expressionNode)
                        parseContext.error(idToken.loc, "buffer aliasing not yet supported", "block initializer", "");
                    parseContext.declareBlock(idToken.loc, variableType, fullName);
                    parseContext.declareStructBufferCounter(idToken.loc, variableType, *fullName);
                } else {
                    if (variableType.getQualifier().storage == EvqUniform && ! variableType.containsOpaque()) {
                        // this isn't really an individual variable, but a member of the $Global buffer
                        parseContext.growGlobalUniformBlock(idToken.loc, variableType, *fullName);
                    } else {
                        // Declare the variable and add any initializer code to the AST.
                        // The top-level node is always made into an aggregate, as that's
                        // historically how the AST has been.
                        initializers = intermediate.growAggregate(initializers, 
                            parseContext.declareVariable(idToken.loc, *fullName, variableType, expressionNode),
                            idToken.loc);
                    }
                }
            }
        }

        // COMMA
        if (acceptTokenClass(EHTokComma))
            declarator_list = true;
    }

    // The top-level initializer node is a sequence.
    if (initializers != nullptr)
        initializers->setOperator(EOpSequence);

    // if we have a locally scoped static, it needs a globally scoped initializer
    if (declaredType.getQualifier().storage == EvqGlobal && !parseContext.symbolTable.atGlobalLevel()) {
        unitNode = intermediate.growAggregate(unitNode, initializers, idToken.loc);
    } else {
        // Add the initializers' aggregate to the nodeList we were handed.
        if (nodeList)
            nodeList = intermediate.growAggregate(nodeList, initializers);
        else
            nodeList = initializers;
    }

    // SEMICOLON
    if (! acceptTokenClass(EHTokSemicolon)) {
        // This may have been a false detection of what appeared to be a declaration, but
        // was actually an assignment such as "float = 4", where "float" is an identifier.
        // We put the token back to let further parsing happen for cases where that may
        // happen.  This errors on the side of caution, and mostly triggers the error.
        if (peek() == EHTokAssign || peek() == EHTokLeftBracket || peek() == EHTokDot || peek() == EHTokComma)
            recedeToken();
        else
            expected(";");
        return false;
    }

    return true;
}

// control_declaration
//      : fully_specified_type identifier EQUAL expression
//
bool HlslGrammar::acceptControlDeclaration(TIntermNode*& node)
{
    node = nullptr;
    TAttributes attributes;

    // fully_specified_type
    TType type;
    if (! acceptFullySpecifiedType(type, attributes))
        return false;

    if (attributes.size() > 0)
        parseContext.warn(token.loc, "attributes don't apply to control declaration", "", "");

    // filter out type casts
    if (peekTokenClass(EHTokLeftParen)) {
        recedeToken();
        return false;
    }

    // identifier
    HlslToken idToken;
    if (! acceptIdentifier(idToken)) {
        expected("identifier");
        return false;
    }

    // EQUAL
    TIntermTyped* expressionNode = nullptr;
    if (! acceptTokenClass(EHTokAssign)) {
        expected("=");
        return false;
    }

    // expression
    if (! acceptExpression(expressionNode)) {
        expected("initializer");
        return false;
    }

    node = parseContext.declareVariable(idToken.loc, *idToken.string, type, expressionNode);

    return true;
}

// fully_specified_type
//      : type_specifier
//      | type_qualifier type_specifier
//
bool HlslGrammar::acceptFullySpecifiedType(TType& type, const TAttributes& attributes)
{
    TIntermNode* nodeList = nullptr;
    return acceptFullySpecifiedType(type, nodeList, attributes);
}
bool HlslGrammar::acceptFullySpecifiedType(TType& type, TIntermNode*& nodeList, const TAttributes& attributes, bool forbidDeclarators)
{
    // type_qualifier
    TQualifier qualifier;
    qualifier.clear();
    if (! acceptQualifier(qualifier))
        return false;
    TSourceLoc loc = token.loc;

    // type_specifier
    if (! acceptType(type, nodeList)) {
        // If this is not a type, we may have inadvertently gone down a wrong path
        // by parsing "sample", which can be treated like either an identifier or a
        // qualifier.  Back it out, if we did.
        if (qualifier.sample)
            recedeToken();

        return false;
    }

    if (type.getBasicType() == EbtBlock) {
        // the type was a block, which set some parts of the qualifier
        parseContext.mergeQualifiers(type.getQualifier(), qualifier);
    
        // merge in the attributes
        parseContext.transferTypeAttributes(token.loc, attributes, type);

        // further, it can create an anonymous instance of the block
        // (cbuffer and tbuffer don't consume the next identifier, and
        // should set forbidDeclarators)
        if (forbidDeclarators || peek() != EHTokIdentifier)
            parseContext.declareBlock(loc, type);
    } else {
        // Some qualifiers are set when parsing the type.  Merge those with
        // whatever comes from acceptQualifier.
        assert(qualifier.layoutFormat == ElfNone);

        qualifier.layoutFormat = type.getQualifier().layoutFormat;
        qualifier.precision    = type.getQualifier().precision;

        if (type.getQualifier().storage == EvqOut ||
            type.getQualifier().storage == EvqBuffer) {
            qualifier.storage      = type.getQualifier().storage;
            qualifier.readonly     = type.getQualifier().readonly;
        }

        if (type.isBuiltIn())
            qualifier.builtIn = type.getQualifier().builtIn;

        type.getQualifier() = qualifier;
    }

    return true;
}

// type_qualifier
//      : qualifier qualifier ...
//
// Zero or more of these, so this can't return false.
//
bool HlslGrammar::acceptQualifier(TQualifier& qualifier)
{
    do {
        switch (peek()) {
        case EHTokStatic:
            qualifier.storage = EvqGlobal;
            break;
        case EHTokExtern:
            // TODO: no meaning in glslang?
            break;
        case EHTokShared:
            // TODO: hint
            break;
        case EHTokGroupShared:
            qualifier.storage = EvqShared;
            break;
        case EHTokUniform:
            qualifier.storage = EvqUniform;
            break;
        case EHTokConst:
            qualifier.storage = EvqConst;
            break;
        case EHTokVolatile:
            qualifier.volatil = true;
            break;
        case EHTokLinear:
            qualifier.smooth = true;
            break;
        case EHTokCentroid:
            qualifier.centroid = true;
            break;
        case EHTokNointerpolation:
            qualifier.flat = true;
            break;
        case EHTokNoperspective:
            qualifier.nopersp = true;
            break;
        case EHTokSample:
            qualifier.sample = true;
            break;
        case EHTokRowMajor:
            qualifier.layoutMatrix = ElmColumnMajor;
            break;
        case EHTokColumnMajor:
            qualifier.layoutMatrix = ElmRowMajor;
            break;
        case EHTokPrecise:
            qualifier.noContraction = true;
            break;
        case EHTokIn:
            qualifier.storage = (qualifier.storage == EvqOut) ? EvqInOut : EvqIn;
            break;
        case EHTokOut:
            qualifier.storage = (qualifier.storage == EvqIn) ? EvqInOut : EvqOut;
            break;
        case EHTokInOut:
            qualifier.storage = EvqInOut;
            break;
        case EHTokLayout:
            if (! acceptLayoutQualifierList(qualifier))
                return false;
            continue;
        case EHTokGloballyCoherent:
            qualifier.coherent = true;
            break;
        case EHTokInline:
            // TODO: map this to SPIR-V function control
            break;

        // GS geometries: these are specified on stage input variables, and are an error (not verified here)
        // for output variables.
        case EHTokPoint:
            qualifier.storage = EvqIn;
            if (!parseContext.handleInputGeometry(token.loc, ElgPoints))
                return false;
            break;
        case EHTokLine:
            qualifier.storage = EvqIn;
            if (!parseContext.handleInputGeometry(token.loc, ElgLines))
                return false;
            break;
        case EHTokTriangle:
            qualifier.storage = EvqIn;
            if (!parseContext.handleInputGeometry(token.loc, ElgTriangles))
                return false;
            break;
        case EHTokLineAdj:
            qualifier.storage = EvqIn;
            if (!parseContext.handleInputGeometry(token.loc, ElgLinesAdjacency))
                return false;
            break;
        case EHTokTriangleAdj:
            qualifier.storage = EvqIn;
            if (!parseContext.handleInputGeometry(token.loc, ElgTrianglesAdjacency))
                return false;
            break;

        default:
            return true;
        }
        advanceToken();
    } while (true);
}

// layout_qualifier_list
//      : LAYOUT LEFT_PAREN layout_qualifier COMMA layout_qualifier ... RIGHT_PAREN
//
// layout_qualifier
//      : identifier
//      | identifier EQUAL expression
//
// Zero or more of these, so this can't return false.
//
bool HlslGrammar::acceptLayoutQualifierList(TQualifier& qualifier)
{
    if (! acceptTokenClass(EHTokLayout))
        return false;

    // LEFT_PAREN
    if (! acceptTokenClass(EHTokLeftParen))
        return false;

    do {
        // identifier
        HlslToken idToken;
        if (! acceptIdentifier(idToken))
            break;

        // EQUAL expression
        if (acceptTokenClass(EHTokAssign)) {
            TIntermTyped* expr;
            if (! acceptConditionalExpression(expr)) {
                expected("expression");
                return false;
            }
            parseContext.setLayoutQualifier(idToken.loc, qualifier, *idToken.string, expr);
        } else
            parseContext.setLayoutQualifier(idToken.loc, qualifier, *idToken.string);

        // COMMA
        if (! acceptTokenClass(EHTokComma))
            break;
    } while (true);

    // RIGHT_PAREN
    if (! acceptTokenClass(EHTokRightParen)) {
        expected(")");
        return false;
    }

    return true;
}

// template_type
//      : FLOAT
//      | DOUBLE
//      | INT
//      | DWORD
//      | UINT
//      | BOOL
//
bool HlslGrammar::acceptTemplateVecMatBasicType(TBasicType& basicType)
{
    switch (peek()) {
    case EHTokFloat:
        basicType = EbtFloat;
        break;
    case EHTokDouble:
        basicType = EbtDouble;
        break;
    case EHTokInt:
    case EHTokDword:
        basicType = EbtInt;
        break;
    case EHTokUint:
        basicType = EbtUint;
        break;
    case EHTokBool:
        basicType = EbtBool;
        break;
    default:
        return false;
    }

    advanceToken();

    return true;
}

// vector_template_type
//      : VECTOR
//      | VECTOR LEFT_ANGLE template_type COMMA integer_literal RIGHT_ANGLE
//
bool HlslGrammar::acceptVectorTemplateType(TType& type)
{
    if (! acceptTokenClass(EHTokVector))
        return false;

    if (! acceptTokenClass(EHTokLeftAngle)) {
        // in HLSL, 'vector' alone means float4.
        new(&type) TType(EbtFloat, EvqTemporary, 4);
        return true;
    }

    TBasicType basicType;
    if (! acceptTemplateVecMatBasicType(basicType)) {
        expected("scalar type");
        return false;
    }

    // COMMA
    if (! acceptTokenClass(EHTokComma)) {
        expected(",");
        return false;
    }

    // integer
    if (! peekTokenClass(EHTokIntConstant)) {
        expected("literal integer");
        return false;
    }

    TIntermTyped* vecSize;
    if (! acceptLiteral(vecSize))
        return false;

    const int vecSizeI = vecSize->getAsConstantUnion()->getConstArray()[0].getIConst();

    new(&type) TType(basicType, EvqTemporary, vecSizeI);

    if (vecSizeI == 1)
        type.makeVector();

    if (!acceptTokenClass(EHTokRightAngle)) {
        expected("right angle bracket");
        return false;
    }

    return true;
}

// matrix_template_type
//      : MATRIX
//      | MATRIX LEFT_ANGLE template_type COMMA integer_literal COMMA integer_literal RIGHT_ANGLE
//
bool HlslGrammar::acceptMatrixTemplateType(TType& type)
{
    if (! acceptTokenClass(EHTokMatrix))
        return false;

    if (! acceptTokenClass(EHTokLeftAngle)) {
        // in HLSL, 'matrix' alone means float4x4.
        new(&type) TType(EbtFloat, EvqTemporary, 0, 4, 4);
        return true;
    }

    TBasicType basicType;
    if (! acceptTemplateVecMatBasicType(basicType)) {
        expected("scalar type");
        return false;
    }

    // COMMA
    if (! acceptTokenClass(EHTokComma)) {
        expected(",");
        return false;
    }

    // integer rows
    if (! peekTokenClass(EHTokIntConstant)) {
        expected("literal integer");
        return false;
    }

    TIntermTyped* rows;
    if (! acceptLiteral(rows))
        return false;

    // COMMA
    if (! acceptTokenClass(EHTokComma)) {
        expected(",");
        return false;
    }

    // integer cols
    if (! peekTokenClass(EHTokIntConstant)) {
        expected("literal integer");
        return false;
    }

    TIntermTyped* cols;
    if (! acceptLiteral(cols))
        return false;

    new(&type) TType(basicType, EvqTemporary, 0,
                     rows->getAsConstantUnion()->getConstArray()[0].getIConst(),
                     cols->getAsConstantUnion()->getConstArray()[0].getIConst());

    if (!acceptTokenClass(EHTokRightAngle)) {
        expected("right angle bracket");
        return false;
    }

    return true;
}

// layout_geometry
//      : LINESTREAM
//      | POINTSTREAM
//      | TRIANGLESTREAM
//
bool HlslGrammar::acceptOutputPrimitiveGeometry(TLayoutGeometry& geometry)
{
    // read geometry type
    const EHlslTokenClass geometryType = peek();

    switch (geometryType) {
    case EHTokPointStream:    geometry = ElgPoints;        break;
    case EHTokLineStream:     geometry = ElgLineStrip;     break;
    case EHTokTriangleStream: geometry = ElgTriangleStrip; break;
    default:
        return false;  // not a layout geometry
    }

    advanceToken();  // consume the layout keyword
    return true;
}

// tessellation_decl_type
//      : INPUTPATCH
//      | OUTPUTPATCH
//
bool HlslGrammar::acceptTessellationDeclType(TBuiltInVariable& patchType)
{
    // read geometry type
    const EHlslTokenClass tessType = peek();

    switch (tessType) {
    case EHTokInputPatch:    patchType = EbvInputPatch;  break;
    case EHTokOutputPatch:   patchType = EbvOutputPatch; break;
    default:
        return false;  // not a tessellation decl
    }

    advanceToken();  // consume the keyword
    return true;
}

// tessellation_patch_template_type
//      : tessellation_decl_type LEFT_ANGLE type comma integer_literal RIGHT_ANGLE
//
bool HlslGrammar::acceptTessellationPatchTemplateType(TType& type)
{
    TBuiltInVariable patchType;

    if (! acceptTessellationDeclType(patchType))
        return false;
    
    if (! acceptTokenClass(EHTokLeftAngle))
        return false;

    if (! acceptType(type)) {
        expected("tessellation patch type");
        return false;
    }

    if (! acceptTokenClass(EHTokComma))
        return false;

    // integer size
    if (! peekTokenClass(EHTokIntConstant)) {
        expected("literal integer");
        return false;
    }

    TIntermTyped* size;
    if (! acceptLiteral(size))
        return false;

    TArraySizes* arraySizes = new TArraySizes;
    arraySizes->addInnerSize(size->getAsConstantUnion()->getConstArray()[0].getIConst());
    type.transferArraySizes(arraySizes);
    type.getQualifier().builtIn = patchType;

    if (! acceptTokenClass(EHTokRightAngle)) {
        expected("right angle bracket");
        return false;
    }

    return true;
}
    
// stream_out_template_type
//      : output_primitive_geometry_type LEFT_ANGLE type RIGHT_ANGLE
//
bool HlslGrammar::acceptStreamOutTemplateType(TType& type, TLayoutGeometry& geometry)
{
    geometry = ElgNone;

    if (! acceptOutputPrimitiveGeometry(geometry))
        return false;

    if (! acceptTokenClass(EHTokLeftAngle))
        return false;

    if (! acceptType(type)) {
        expected("stream output type");
        return false;
    }

    type.getQualifier().storage = EvqOut;
    type.getQualifier().builtIn = EbvGsOutputStream;

    if (! acceptTokenClass(EHTokRightAngle)) {
        expected("right angle bracket");
        return false;
    }

    return true;
}

// annotations
//      : LEFT_ANGLE declaration SEMI_COLON ... declaration SEMICOLON RIGHT_ANGLE
//
bool HlslGrammar::acceptAnnotations(TQualifier&)
{
    if (! acceptTokenClass(EHTokLeftAngle))
        return false;

    // note that we are nesting a name space
    parseContext.nestAnnotations();

    // declaration SEMI_COLON ... declaration SEMICOLON RIGHT_ANGLE
    do {
        // eat any extra SEMI_COLON; don't know if the grammar calls for this or not
        while (acceptTokenClass(EHTokSemicolon))
            ;

        if (acceptTokenClass(EHTokRightAngle))
            break;

        // declaration
        TIntermNode* node = nullptr;
        if (! acceptDeclaration(node)) {
            expected("declaration in annotation");
            return false;
        }
    } while (true);

    parseContext.unnestAnnotations();
    return true;
}

// subpass input type
//      : SUBPASSINPUT
//      | SUBPASSINPUT VECTOR LEFT_ANGLE template_type RIGHT_ANGLE
//      | SUBPASSINPUTMS
//      | SUBPASSINPUTMS VECTOR LEFT_ANGLE template_type RIGHT_ANGLE
bool HlslGrammar::acceptSubpassInputType(TType& type)
{
    // read subpass type
    const EHlslTokenClass subpassInputType = peek();

    bool multisample;

    switch (subpassInputType) {
    case EHTokSubpassInput:   multisample = false; break;
    case EHTokSubpassInputMS: multisample = true;  break;
    default:
        return false;  // not a subpass input declaration
    }

    advanceToken();  // consume the sampler type keyword

    TType subpassType(EbtFloat, EvqUniform, 4); // default type is float4

    if (acceptTokenClass(EHTokLeftAngle)) {
        if (! acceptType(subpassType)) {
            expected("scalar or vector type");
            return false;
        }

        const TBasicType basicRetType = subpassType.getBasicType() ;

        switch (basicRetType) {
        case EbtFloat:
        case EbtUint:
        case EbtInt:
        case EbtStruct:
            break;
        default:
            unimplemented("basic type in subpass input");
            return false;
        }

        if (! acceptTokenClass(EHTokRightAngle)) {
            expected("right angle bracket");
            return false;
        }
    }

    const TBasicType subpassBasicType = subpassType.isStruct() ? (*subpassType.getStruct())[0].type->getBasicType()
        : subpassType.getBasicType();

    TSampler sampler;
    sampler.setSubpass(subpassBasicType, multisample);

    // Remember the declared return type.  Function returns false on error.
    if (!parseContext.setTextureReturnType(sampler, subpassType, token.loc))
        return false;

    type.shallowCopy(TType(sampler, EvqUniform));

    return true;
}

// sampler_type for DX9 compatibility 
//      : SAMPLER
//      | SAMPLER1D
//      | SAMPLER2D
//      | SAMPLER3D
//      | SAMPLERCUBE
bool HlslGrammar::acceptSamplerTypeDX9(TType &type)
{
    // read sampler type
    const EHlslTokenClass samplerType = peek();

    TSamplerDim dim = EsdNone;
    TType txType(EbtFloat, EvqUniform, 4); // default type is float4

    bool isShadow = false;

    switch (samplerType)
    {
    case EHTokSampler:		dim = Esd2D;	break;
    case EHTokSampler1d:	dim = Esd1D;	break;
    case EHTokSampler2d:	dim = Esd2D;	break;
    case EHTokSampler3d:	dim = Esd3D;	break;
    case EHTokSamplerCube:	dim = EsdCube;	break;
    default:
        return false; // not a dx9 sampler declaration
    }

    advanceToken(); // consume the sampler type keyword

    TArraySizes *arraySizes = nullptr; // TODO: array

    TSampler sampler;
    sampler.set(txType.getBasicType(), dim, false, isShadow, false);

    if (!parseContext.setTextureReturnType(sampler, txType, token.loc))
        return false;

    type.shallowCopy(TType(sampler, EvqUniform, arraySizes));
    type.getQualifier().layoutFormat = ElfNone;

    return true;
}

// sampler_type
//      : SAMPLER
//      | SAMPLER1D
//      | SAMPLER2D
//      | SAMPLER3D
//      | SAMPLERCUBE
//      | SAMPLERSTATE
//      | SAMPLERCOMPARISONSTATE
bool HlslGrammar::acceptSamplerType(TType& type)
{
    // read sampler type
    const EHlslTokenClass samplerType = peek();

    // TODO: for DX9
    // TSamplerDim dim = EsdNone;

    bool isShadow = false;

    switch (samplerType) {
    case EHTokSampler:      break;
    case EHTokSampler1d:    /*dim = Esd1D*/; break;
    case EHTokSampler2d:    /*dim = Esd2D*/; break;
    case EHTokSampler3d:    /*dim = Esd3D*/; break;
    case EHTokSamplerCube:  /*dim = EsdCube*/; break;
    case EHTokSamplerState: break;
    case EHTokSamplerComparisonState: isShadow = true; break;
    default:
        return false;  // not a sampler declaration
    }

    advanceToken();  // consume the sampler type keyword

    TArraySizes* arraySizes = nullptr; // TODO: array

    TSampler sampler;
    sampler.setPureSampler(isShadow);

    type.shallowCopy(TType(sampler, EvqUniform, arraySizes));

    return true;
}

// texture_type
//      | BUFFER
//      | TEXTURE1D
//      | TEXTURE1DARRAY
//      | TEXTURE2D
//      | TEXTURE2DARRAY
//      | TEXTURE3D
//      | TEXTURECUBE
//      | TEXTURECUBEARRAY
//      | TEXTURE2DMS
//      | TEXTURE2DMSARRAY
//      | RWBUFFER
//      | RWTEXTURE1D
//      | RWTEXTURE1DARRAY
//      | RWTEXTURE2D
//      | RWTEXTURE2DARRAY
//      | RWTEXTURE3D

bool HlslGrammar::acceptTextureType(TType& type)
{
    const EHlslTokenClass textureType = peek();

    TSamplerDim dim = EsdNone;
    bool array = false;
    bool ms    = false;
    bool image = false;
    bool combined = true;

    switch (textureType) {
    case EHTokBuffer:            dim = EsdBuffer; combined = false;    break;
    case EHTokTexture1d:         dim = Esd1D;                          break;
    case EHTokTexture1darray:    dim = Esd1D; array = true;            break;
    case EHTokTexture2d:         dim = Esd2D;                          break;
    case EHTokTexture2darray:    dim = Esd2D; array = true;            break;
    case EHTokTexture3d:         dim = Esd3D;                          break;
    case EHTokTextureCube:       dim = EsdCube;                        break;
    case EHTokTextureCubearray:  dim = EsdCube; array = true;          break;
    case EHTokTexture2DMS:       dim = Esd2D; ms = true;               break;
    case EHTokTexture2DMSarray:  dim = Esd2D; array = true; ms = true; break;
    case EHTokRWBuffer:          dim = EsdBuffer; image=true;          break;
    case EHTokRWTexture1d:       dim = Esd1D; array=false; image=true; break;
    case EHTokRWTexture1darray:  dim = Esd1D; array=true;  image=true; break;
    case EHTokRWTexture2d:       dim = Esd2D; array=false; image=true; break;
    case EHTokRWTexture2darray:  dim = Esd2D; array=true;  image=true; break;
    case EHTokRWTexture3d:       dim = Esd3D; array=false; image=true; break;
    default:
        return false;  // not a texture declaration
    }

    advanceToken();  // consume the texture object keyword

    TType txType(EbtFloat, EvqUniform, 4); // default type is float4

    TIntermTyped* msCount = nullptr;

    // texture type: required for multisample types and RWBuffer/RWTextures!
    if (acceptTokenClass(EHTokLeftAngle)) {
        if (! acceptType(txType)) {
            expected("scalar or vector type");
            return false;
        }

        const TBasicType basicRetType = txType.getBasicType() ;

        switch (basicRetType) {
        case EbtFloat:
        case EbtUint:
        case EbtInt:
        case EbtStruct:
            break;
        default:
            unimplemented("basic type in texture");
            return false;
        }

        // Buffers can handle small mats if they fit in 4 components
        if (dim == EsdBuffer && txType.isMatrix()) {
            if ((txType.getMatrixCols() * txType.getMatrixRows()) > 4) {
                expected("components < 4 in matrix buffer type");
                return false;
            }

            // TODO: except we don't handle it yet...
            unimplemented("matrix type in buffer");
            return false;
        }

        if (!txType.isScalar() && !txType.isVector() && !txType.isStruct()) {
            expected("scalar, vector, or struct type");
            return false;
        }

        if (ms && acceptTokenClass(EHTokComma)) {
            // read sample count for multisample types, if given
            if (! peekTokenClass(EHTokIntConstant)) {
                expected("multisample count");
                return false;
            }

            if (! acceptLiteral(msCount))  // should never fail, since we just found an integer
                return false;
        }

        if (! acceptTokenClass(EHTokRightAngle)) {
            expected("right angle bracket");
            return false;
        }
    } else if (ms) {
        expected("texture type for multisample");
        return false;
    } else if (image) {
        expected("type for RWTexture/RWBuffer");
        return false;
    }

    TArraySizes* arraySizes = nullptr;
    const bool shadow = false; // declared on the sampler

    TSampler sampler;
    TLayoutFormat format = ElfNone;

    // Buffer, RWBuffer and RWTexture (images) require a TLayoutFormat.  We handle only a limit set.
    if (image || dim == EsdBuffer)
        format = parseContext.getLayoutFromTxType(token.loc, txType);

    const TBasicType txBasicType = txType.isStruct() ? (*txType.getStruct())[0].type->getBasicType()
        : txType.getBasicType();

    // Non-image Buffers are combined
    if (dim == EsdBuffer && !image) {
        sampler.set(txType.getBasicType(), dim, array);
    } else {
        // DX10 textures are separated.  TODO: DX9.
        if (image) {
            sampler.setImage(txBasicType, dim, array, shadow, ms);
        } else {
            sampler.setTexture(txBasicType, dim, array, shadow, ms);
        }
    }

    // Remember the declared return type.  Function returns false on error.
    if (!parseContext.setTextureReturnType(sampler, txType, token.loc))
        return false;

    // Force uncombined, if necessary
    if (!combined)
        sampler.combined = false;

    type.shallowCopy(TType(sampler, EvqUniform, arraySizes));
    type.getQualifier().layoutFormat = format;

    return true;
}

// If token is for a type, update 'type' with the type information,
// and return true and advance.
// Otherwise, return false, and don't advance
bool HlslGrammar::acceptType(TType& type)
{
    TIntermNode* nodeList = nullptr;
    return acceptType(type, nodeList);
}
bool HlslGrammar::acceptType(TType& type, TIntermNode*& nodeList)
{
    // Basic types for min* types, use native halfs if the option allows them.
    bool enable16BitTypes = parseContext.hlslEnable16BitTypes();

    const TBasicType min16float_bt = enable16BitTypes ? EbtFloat16 : EbtFloat;
    const TBasicType min10float_bt = enable16BitTypes ? EbtFloat16 : EbtFloat;
    const TBasicType half_bt       = enable16BitTypes ? EbtFloat16 : EbtFloat;
    const TBasicType min16int_bt   = enable16BitTypes ? EbtInt16   : EbtInt;
    const TBasicType min12int_bt   = enable16BitTypes ? EbtInt16   : EbtInt;
    const TBasicType min16uint_bt  = enable16BitTypes ? EbtUint16  : EbtUint;

    // Some types might have turned into identifiers. Take the hit for checking
    // when this has happened.
    if (typeIdentifiers) {
        const char* identifierString = getTypeString(peek());
        if (identifierString != nullptr) {
            TString name = identifierString;
            // if it's an identifier, it's not a type
            if (parseContext.symbolTable.find(name) != nullptr)
                return false;
        }
    }

    bool isUnorm = false;
    bool isSnorm = false;

    // Accept snorm and unorm.  Presently, this is ignored, save for an error check below.
    switch (peek()) {
    case EHTokUnorm:
        isUnorm = true;
        advanceToken();  // eat the token
        break;
    case EHTokSNorm:
        isSnorm = true;
        advanceToken();  // eat the token
        break;
    default:
        break;
    }

    switch (peek()) {
    case EHTokVector:
        return acceptVectorTemplateType(type);
        break;

    case EHTokMatrix:
        return acceptMatrixTemplateType(type);
        break;

    case EHTokPointStream:            // fall through
    case EHTokLineStream:             // ...
    case EHTokTriangleStream:         // ...
        {
            TLayoutGeometry geometry;
            if (! acceptStreamOutTemplateType(type, geometry))
                return false;

            if (! parseContext.handleOutputGeometry(token.loc, geometry))
                return false;

            return true;
        }

    case EHTokInputPatch:             // fall through
    case EHTokOutputPatch:            // ...
        {
            if (! acceptTessellationPatchTemplateType(type))
                return false;

            return true;
        }

    case EHTokSampler:                // fall through
    case EHTokSampler1d:              // ...
    case EHTokSampler2d:              // ...
    case EHTokSampler3d:              // ...
    case EHTokSamplerCube:            // ...
        if (parseContext.hlslDX9Compatible())
            return acceptSamplerTypeDX9(type);
        else
            return acceptSamplerType(type);
        break;

    case EHTokSamplerState:           // fall through
    case EHTokSamplerComparisonState: // ...
        return acceptSamplerType(type);
        break;

    case EHTokSubpassInput:           // fall through
    case EHTokSubpassInputMS:         // ...
        return acceptSubpassInputType(type);
        break;

    case EHTokBuffer:                 // fall through
    case EHTokTexture1d:              // ...
    case EHTokTexture1darray:         // ...
    case EHTokTexture2d:              // ...
    case EHTokTexture2darray:         // ...
    case EHTokTexture3d:              // ...
    case EHTokTextureCube:            // ...
    case EHTokTextureCubearray:       // ...
    case EHTokTexture2DMS:            // ...
    case EHTokTexture2DMSarray:       // ...
    case EHTokRWTexture1d:            // ...
    case EHTokRWTexture1darray:       // ...
    case EHTokRWTexture2d:            // ...
    case EHTokRWTexture2darray:       // ...
    case EHTokRWTexture3d:            // ...
    case EHTokRWBuffer:               // ...
        return acceptTextureType(type);
        break;

    case EHTokAppendStructuredBuffer:
    case EHTokByteAddressBuffer:
    case EHTokConsumeStructuredBuffer:
    case EHTokRWByteAddressBuffer:
    case EHTokRWStructuredBuffer:
    case EHTokStructuredBuffer:
        return acceptStructBufferType(type);
        break;

    case EHTokTextureBuffer:
        return acceptTextureBufferType(type);
        break;

    case EHTokConstantBuffer:
        return acceptConstantBufferType(type);

    case EHTokClass:
    case EHTokStruct:
    case EHTokCBuffer:
    case EHTokTBuffer:
        return acceptStruct(type, nodeList);

    case EHTokIdentifier:
        // An identifier could be for a user-defined type.
        // Note we cache the symbol table lookup, to save for a later rule
        // when this is not a type.
        if (parseContext.lookupUserType(*token.string, type) != nullptr) {
            advanceToken();
            return true;
        } else
            return false;

    case EHTokVoid:
        new(&type) TType(EbtVoid);
        break;

    case EHTokString:
        new(&type) TType(EbtString);
        break;

    case EHTokFloat:
        new(&type) TType(EbtFloat);
        break;
    case EHTokFloat1:
        new(&type) TType(EbtFloat);
        type.makeVector();
        break;
    case EHTokFloat2:
        new(&type) TType(EbtFloat, EvqTemporary, 2);
        break;
    case EHTokFloat3:
        new(&type) TType(EbtFloat, EvqTemporary, 3);
        break;
    case EHTokFloat4:
        new(&type) TType(EbtFloat, EvqTemporary, 4);
        break;

    case EHTokDouble:
        new(&type) TType(EbtDouble);
        break;
    case EHTokDouble1:
        new(&type) TType(EbtDouble);
        type.makeVector();
        break;
    case EHTokDouble2:
        new(&type) TType(EbtDouble, EvqTemporary, 2);
        break;
    case EHTokDouble3:
        new(&type) TType(EbtDouble, EvqTemporary, 3);
        break;
    case EHTokDouble4:
        new(&type) TType(EbtDouble, EvqTemporary, 4);
        break;

    case EHTokInt:
    case EHTokDword:
        new(&type) TType(EbtInt);
        break;
    case EHTokInt1:
        new(&type) TType(EbtInt);
        type.makeVector();
        break;
    case EHTokInt2:
        new(&type) TType(EbtInt, EvqTemporary, 2);
        break;
    case EHTokInt3:
        new(&type) TType(EbtInt, EvqTemporary, 3);
        break;
    case EHTokInt4:
        new(&type) TType(EbtInt, EvqTemporary, 4);
        break;

    case EHTokUint:
        new(&type) TType(EbtUint);
        break;
    case EHTokUint1:
        new(&type) TType(EbtUint);
        type.makeVector();
        break;
    case EHTokUint2:
        new(&type) TType(EbtUint, EvqTemporary, 2);
        break;
    case EHTokUint3:
        new(&type) TType(EbtUint, EvqTemporary, 3);
        break;
    case EHTokUint4:
        new(&type) TType(EbtUint, EvqTemporary, 4);
        break;

    case EHTokUint64:
        new(&type) TType(EbtUint64);
        break;

    case EHTokBool:
        new(&type) TType(EbtBool);
        break;
    case EHTokBool1:
        new(&type) TType(EbtBool);
        type.makeVector();
        break;
    case EHTokBool2:
        new(&type) TType(EbtBool, EvqTemporary, 2);
        break;
    case EHTokBool3:
        new(&type) TType(EbtBool, EvqTemporary, 3);
        break;
    case EHTokBool4:
        new(&type) TType(EbtBool, EvqTemporary, 4);
        break;

    case EHTokHalf:
        new(&type) TType(half_bt, EvqTemporary);
        break;
    case EHTokHalf1:
        new(&type) TType(half_bt, EvqTemporary);
        type.makeVector();
        break;
    case EHTokHalf2:
        new(&type) TType(half_bt, EvqTemporary, 2);
        break;
    case EHTokHalf3:
        new(&type) TType(half_bt, EvqTemporary, 3);
        break;
    case EHTokHalf4:
        new(&type) TType(half_bt, EvqTemporary, 4);
        break;

    case EHTokMin16float:
        new(&type) TType(min16float_bt, EvqTemporary, EpqMedium);
        break;
    case EHTokMin16float1:
        new(&type) TType(min16float_bt, EvqTemporary, EpqMedium);
        type.makeVector();
        break;
    case EHTokMin16float2:
        new(&type) TType(min16float_bt, EvqTemporary, EpqMedium, 2);
        break;
    case EHTokMin16float3:
        new(&type) TType(min16float_bt, EvqTemporary, EpqMedium, 3);
        break;
    case EHTokMin16float4:
        new(&type) TType(min16float_bt, EvqTemporary, EpqMedium, 4);
        break;

    case EHTokMin10float:
        new(&type) TType(min10float_bt, EvqTemporary, EpqMedium);
        break;
    case EHTokMin10float1:
        new(&type) TType(min10float_bt, EvqTemporary, EpqMedium);
        type.makeVector();
        break;
    case EHTokMin10float2:
        new(&type) TType(min10float_bt, EvqTemporary, EpqMedium, 2);
        break;
    case EHTokMin10float3:
        new(&type) TType(min10float_bt, EvqTemporary, EpqMedium, 3);
        break;
    case EHTokMin10float4:
        new(&type) TType(min10float_bt, EvqTemporary, EpqMedium, 4);
        break;

    case EHTokMin16int:
        new(&type) TType(min16int_bt, EvqTemporary, EpqMedium);
        break;
    case EHTokMin16int1:
        new(&type) TType(min16int_bt, EvqTemporary, EpqMedium);
        type.makeVector();
        break;
    case EHTokMin16int2:
        new(&type) TType(min16int_bt, EvqTemporary, EpqMedium, 2);
        break;
    case EHTokMin16int3:
        new(&type) TType(min16int_bt, EvqTemporary, EpqMedium, 3);
        break;
    case EHTokMin16int4:
        new(&type) TType(min16int_bt, EvqTemporary, EpqMedium, 4);
        break;

    case EHTokMin12int:
        new(&type) TType(min12int_bt, EvqTemporary, EpqMedium);
        break;
    case EHTokMin12int1:
        new(&type) TType(min12int_bt, EvqTemporary, EpqMedium);
        type.makeVector();
        break;
    case EHTokMin12int2:
        new(&type) TType(min12int_bt, EvqTemporary, EpqMedium, 2);
        break;
    case EHTokMin12int3:
        new(&type) TType(min12int_bt, EvqTemporary, EpqMedium, 3);
        break;
    case EHTokMin12int4:
        new(&type) TType(min12int_bt, EvqTemporary, EpqMedium, 4);
        break;

    case EHTokMin16uint:
        new(&type) TType(min16uint_bt, EvqTemporary, EpqMedium);
        break;
    case EHTokMin16uint1:
        new(&type) TType(min16uint_bt, EvqTemporary, EpqMedium);
        type.makeVector();
        break;
    case EHTokMin16uint2:
        new(&type) TType(min16uint_bt, EvqTemporary, EpqMedium, 2);
        break;
    case EHTokMin16uint3:
        new(&type) TType(min16uint_bt, EvqTemporary, EpqMedium, 3);
        break;
    case EHTokMin16uint4:
        new(&type) TType(min16uint_bt, EvqTemporary, EpqMedium, 4);
        break;

    case EHTokInt1x1:
        new(&type) TType(EbtInt, EvqTemporary, 0, 1, 1);
        break;
    case EHTokInt1x2:
        new(&type) TType(EbtInt, EvqTemporary, 0, 1, 2);
        break;
    case EHTokInt1x3:
        new(&type) TType(EbtInt, EvqTemporary, 0, 1, 3);
        break;
    case EHTokInt1x4:
        new(&type) TType(EbtInt, EvqTemporary, 0, 1, 4);
        break;
    case EHTokInt2x1:
        new(&type) TType(EbtInt, EvqTemporary, 0, 2, 1);
        break;
    case EHTokInt2x2:
        new(&type) TType(EbtInt, EvqTemporary, 0, 2, 2);
        break;
    case EHTokInt2x3:
        new(&type) TType(EbtInt, EvqTemporary, 0, 2, 3);
        break;
    case EHTokInt2x4:
        new(&type) TType(EbtInt, EvqTemporary, 0, 2, 4);
        break;
    case EHTokInt3x1:
        new(&type) TType(EbtInt, EvqTemporary, 0, 3, 1);
        break;
    case EHTokInt3x2:
        new(&type) TType(EbtInt, EvqTemporary, 0, 3, 2);
        break;
    case EHTokInt3x3:
        new(&type) TType(EbtInt, EvqTemporary, 0, 3, 3);
        break;
    case EHTokInt3x4:
        new(&type) TType(EbtInt, EvqTemporary, 0, 3, 4);
        break;
    case EHTokInt4x1:
        new(&type) TType(EbtInt, EvqTemporary, 0, 4, 1);
        break;
    case EHTokInt4x2:
        new(&type) TType(EbtInt, EvqTemporary, 0, 4, 2);
        break;
    case EHTokInt4x3:
        new(&type) TType(EbtInt, EvqTemporary, 0, 4, 3);
        break;
    case EHTokInt4x4:
        new(&type) TType(EbtInt, EvqTemporary, 0, 4, 4);
        break;

    case EHTokUint1x1:
        new(&type) TType(EbtUint, EvqTemporary, 0, 1, 1);
        break;
    case EHTokUint1x2:
        new(&type) TType(EbtUint, EvqTemporary, 0, 1, 2);
        break;
    case EHTokUint1x3:
        new(&type) TType(EbtUint, EvqTemporary, 0, 1, 3);
        break;
    case EHTokUint1x4:
        new(&type) TType(EbtUint, EvqTemporary, 0, 1, 4);
        break;
    case EHTokUint2x1:
        new(&type) TType(EbtUint, EvqTemporary, 0, 2, 1);
        break;
    case EHTokUint2x2:
        new(&type) TType(EbtUint, EvqTemporary, 0, 2, 2);
        break;
    case EHTokUint2x3:
        new(&type) TType(EbtUint, EvqTemporary, 0, 2, 3);
        break;
    case EHTokUint2x4:
        new(&type) TType(EbtUint, EvqTemporary, 0, 2, 4);
        break;
    case EHTokUint3x1:
        new(&type) TType(EbtUint, EvqTemporary, 0, 3, 1);
        break;
    case EHTokUint3x2:
        new(&type) TType(EbtUint, EvqTemporary, 0, 3, 2);
        break;
    case EHTokUint3x3:
        new(&type) TType(EbtUint, EvqTemporary, 0, 3, 3);
        break;
    case EHTokUint3x4:
        new(&type) TType(EbtUint, EvqTemporary, 0, 3, 4);
        break;
    case EHTokUint4x1:
        new(&type) TType(EbtUint, EvqTemporary, 0, 4, 1);
        break;
    case EHTokUint4x2:
        new(&type) TType(EbtUint, EvqTemporary, 0, 4, 2);
        break;
    case EHTokUint4x3:
        new(&type) TType(EbtUint, EvqTemporary, 0, 4, 3);
        break;
    case EHTokUint4x4:
        new(&type) TType(EbtUint, EvqTemporary, 0, 4, 4);
        break;

    case EHTokBool1x1:
        new(&type) TType(EbtBool, EvqTemporary, 0, 1, 1);
        break;
    case EHTokBool1x2:
        new(&type) TType(EbtBool, EvqTemporary, 0, 1, 2);
        break;
    case EHTokBool1x3:
        new(&type) TType(EbtBool, EvqTemporary, 0, 1, 3);
        break;
    case EHTokBool1x4:
        new(&type) TType(EbtBool, EvqTemporary, 0, 1, 4);
        break;
    case EHTokBool2x1:
        new(&type) TType(EbtBool, EvqTemporary, 0, 2, 1);
        break;
    case EHTokBool2x2:
        new(&type) TType(EbtBool, EvqTemporary, 0, 2, 2);
        break;
    case EHTokBool2x3:
        new(&type) TType(EbtBool, EvqTemporary, 0, 2, 3);
        break;
    case EHTokBool2x4:
        new(&type) TType(EbtBool, EvqTemporary, 0, 2, 4);
        break;
    case EHTokBool3x1:
        new(&type) TType(EbtBool, EvqTemporary, 0, 3, 1);
        break;
    case EHTokBool3x2:
        new(&type) TType(EbtBool, EvqTemporary, 0, 3, 2);
        break;
    case EHTokBool3x3:
        new(&type) TType(EbtBool, EvqTemporary, 0, 3, 3);
        break;
    case EHTokBool3x4:
        new(&type) TType(EbtBool, EvqTemporary, 0, 3, 4);
        break;
    case EHTokBool4x1:
        new(&type) TType(EbtBool, EvqTemporary, 0, 4, 1);
        break;
    case EHTokBool4x2:
        new(&type) TType(EbtBool, EvqTemporary, 0, 4, 2);
        break;
    case EHTokBool4x3:
        new(&type) TType(EbtBool, EvqTemporary, 0, 4, 3);
        break;
    case EHTokBool4x4:
        new(&type) TType(EbtBool, EvqTemporary, 0, 4, 4);
        break;

    case EHTokFloat1x1:
        new(&type) TType(EbtFloat, EvqTemporary, 0, 1, 1);
        break;
    case EHTokFloat1x2:
        new(&type) TType(EbtFloat, EvqTemporary, 0, 1, 2);
        break;
    case EHTokFloat1x3:
        new(&type) TType(EbtFloat, EvqTemporary, 0, 1, 3);
        break;
    case EHTokFloat1x4:
        new(&type) TType(EbtFloat, EvqTemporary, 0, 1, 4);
        break;
    case EHTokFloat2x1:
        new(&type) TType(EbtFloat, EvqTemporary, 0, 2, 1);
        break;
    case EHTokFloat2x2:
        new(&type) TType(EbtFloat, EvqTemporary, 0, 2, 2);
        break;
    case EHTokFloat2x3:
        new(&type) TType(EbtFloat, EvqTemporary, 0, 2, 3);
        break;
    case EHTokFloat2x4:
        new(&type) TType(EbtFloat, EvqTemporary, 0, 2, 4);
        break;
    case EHTokFloat3x1:
        new(&type) TType(EbtFloat, EvqTemporary, 0, 3, 1);
        break;
    case EHTokFloat3x2:
        new(&type) TType(EbtFloat, EvqTemporary, 0, 3, 2);
        break;
    case EHTokFloat3x3:
        new(&type) TType(EbtFloat, EvqTemporary, 0, 3, 3);
        break;
    case EHTokFloat3x4:
        new(&type) TType(EbtFloat, EvqTemporary, 0, 3, 4);
        break;
    case EHTokFloat4x1:
        new(&type) TType(EbtFloat, EvqTemporary, 0, 4, 1);
        break;
    case EHTokFloat4x2:
        new(&type) TType(EbtFloat, EvqTemporary, 0, 4, 2);
        break;
    case EHTokFloat4x3:
        new(&type) TType(EbtFloat, EvqTemporary, 0, 4, 3);
        break;
    case EHTokFloat4x4:
        new(&type) TType(EbtFloat, EvqTemporary, 0, 4, 4);
        break;

    case EHTokHalf1x1:
        new(&type) TType(half_bt, EvqTemporary, 0, 1, 1);
        break;
    case EHTokHalf1x2:
        new(&type) TType(half_bt, EvqTemporary, 0, 1, 2);
        break;
    case EHTokHalf1x3:
        new(&type) TType(half_bt, EvqTemporary, 0, 1, 3);
        break;
    case EHTokHalf1x4:
        new(&type) TType(half_bt, EvqTemporary, 0, 1, 4);
        break;
    case EHTokHalf2x1:
        new(&type) TType(half_bt, EvqTemporary, 0, 2, 1);
        break;
    case EHTokHalf2x2:
        new(&type) TType(half_bt, EvqTemporary, 0, 2, 2);
        break;
    case EHTokHalf2x3:
        new(&type) TType(half_bt, EvqTemporary, 0, 2, 3);
        break;
    case EHTokHalf2x4:
        new(&type) TType(half_bt, EvqTemporary, 0, 2, 4);
        break;
    case EHTokHalf3x1:
        new(&type) TType(half_bt, EvqTemporary, 0, 3, 1);
        break;
    case EHTokHalf3x2:
        new(&type) TType(half_bt, EvqTemporary, 0, 3, 2);
        break;
    case EHTokHalf3x3:
        new(&type) TType(half_bt, EvqTemporary, 0, 3, 3);
        break;
    case EHTokHalf3x4:
        new(&type) TType(half_bt, EvqTemporary, 0, 3, 4);
        break;
    case EHTokHalf4x1:
        new(&type) TType(half_bt, EvqTemporary, 0, 4, 1);
        break;
    case EHTokHalf4x2:
        new(&type) TType(half_bt, EvqTemporary, 0, 4, 2);
        break;
    case EHTokHalf4x3:
        new(&type) TType(half_bt, EvqTemporary, 0, 4, 3);
        break;
    case EHTokHalf4x4:
        new(&type) TType(half_bt, EvqTemporary, 0, 4, 4);
        break;

    case EHTokDouble1x1:
        new(&type) TType(EbtDouble, EvqTemporary, 0, 1, 1);
        break;
    case EHTokDouble1x2:
        new(&type) TType(EbtDouble, EvqTemporary, 0, 1, 2);
        break;
    case EHTokDouble1x3:
        new(&type) TType(EbtDouble, EvqTemporary, 0, 1, 3);
        break;
    case EHTokDouble1x4:
        new(&type) TType(EbtDouble, EvqTemporary, 0, 1, 4);
        break;
    case EHTokDouble2x1:
        new(&type) TType(EbtDouble, EvqTemporary, 0, 2, 1);
        break;
    case EHTokDouble2x2:
        new(&type) TType(EbtDouble, EvqTemporary, 0, 2, 2);
        break;
    case EHTokDouble2x3:
        new(&type) TType(EbtDouble, EvqTemporary, 0, 2, 3);
        break;
    case EHTokDouble2x4:
        new(&type) TType(EbtDouble, EvqTemporary, 0, 2, 4);
        break;
    case EHTokDouble3x1:
        new(&type) TType(EbtDouble, EvqTemporary, 0, 3, 1);
        break;
    case EHTokDouble3x2:
        new(&type) TType(EbtDouble, EvqTemporary, 0, 3, 2);
        break;
    case EHTokDouble3x3:
        new(&type) TType(EbtDouble, EvqTemporary, 0, 3, 3);
        break;
    case EHTokDouble3x4:
        new(&type) TType(EbtDouble, EvqTemporary, 0, 3, 4);
        break;
    case EHTokDouble4x1:
        new(&type) TType(EbtDouble, EvqTemporary, 0, 4, 1);
        break;
    case EHTokDouble4x2:
        new(&type) TType(EbtDouble, EvqTemporary, 0, 4, 2);
        break;
    case EHTokDouble4x3:
        new(&type) TType(EbtDouble, EvqTemporary, 0, 4, 3);
        break;
    case EHTokDouble4x4:
        new(&type) TType(EbtDouble, EvqTemporary, 0, 4, 4);
        break;

    default:
        return false;
    }

    advanceToken();

    if ((isUnorm || isSnorm) && !type.isFloatingDomain()) {
        parseContext.error(token.loc, "unorm and snorm only valid in floating point domain", "", "");
        return false;
    }

    return true;
}

// struct
//      : struct_type IDENTIFIER post_decls LEFT_BRACE struct_declaration_list RIGHT_BRACE
//      | struct_type            post_decls LEFT_BRACE struct_declaration_list RIGHT_BRACE
//      | struct_type IDENTIFIER // use of previously declared struct type
//
// struct_type
//      : STRUCT
//      | CLASS
//      | CBUFFER
//      | TBUFFER
//
bool HlslGrammar::acceptStruct(TType& type, TIntermNode*& nodeList)
{
    // This storage qualifier will tell us whether it's an AST
    // block type or just a generic structure type.
    TStorageQualifier storageQualifier = EvqTemporary;
    bool readonly = false;

    if (acceptTokenClass(EHTokCBuffer)) {
        // CBUFFER
        storageQualifier = EvqUniform;
    } else if (acceptTokenClass(EHTokTBuffer)) {
        // TBUFFER
        storageQualifier = EvqBuffer;
        readonly = true;
    } else if (! acceptTokenClass(EHTokClass) && ! acceptTokenClass(EHTokStruct)) {
        // Neither CLASS nor STRUCT
        return false;
    }

    // Now known to be one of CBUFFER, TBUFFER, CLASS, or STRUCT


    // IDENTIFIER.  It might also be a keyword which can double as an identifier.
    // For example:  'cbuffer ConstantBuffer' or 'struct ConstantBuffer' is legal.
    // 'cbuffer int' is also legal, and 'struct int' appears rejected only because
    // it attempts to redefine the 'int' type.
    const char* idString = getTypeString(peek());
    TString structName = "";
    if (peekTokenClass(EHTokIdentifier) || idString != nullptr) {
        if (idString != nullptr)
            structName = *idString;
        else
            structName = *token.string;
        advanceToken();
    }

    // post_decls
    TQualifier postDeclQualifier;
    postDeclQualifier.clear();
    bool postDeclsFound = acceptPostDecls(postDeclQualifier);

    // LEFT_BRACE, or
    // struct_type IDENTIFIER
    if (! acceptTokenClass(EHTokLeftBrace)) {
        if (structName.size() > 0 && !postDeclsFound && parseContext.lookupUserType(structName, type) != nullptr) {
            // struct_type IDENTIFIER
            return true;
        } else {
            expected("{");
            return false;
        }
    }


    // struct_declaration_list
    TTypeList* typeList;
    // Save each member function so they can be processed after we have a fully formed 'this'.
    TVector<TFunctionDeclarator> functionDeclarators;

    parseContext.pushNamespace(structName);
    bool acceptedList = acceptStructDeclarationList(typeList, nodeList, functionDeclarators);
    parseContext.popNamespace();

    if (! acceptedList) {
        expected("struct member declarations");
        return false;
    }

    // RIGHT_BRACE
    if (! acceptTokenClass(EHTokRightBrace)) {
        expected("}");
        return false;
    }

    // create the user-defined type
    if (storageQualifier == EvqTemporary)
        new(&type) TType(typeList, structName);
    else {
        postDeclQualifier.storage = storageQualifier;
        postDeclQualifier.readonly = readonly;
        new(&type) TType(typeList, structName, postDeclQualifier); // sets EbtBlock
    }

    parseContext.declareStruct(token.loc, structName, type);

    // For member functions: now that we know the type of 'this', go back and
    // - add their implicit argument with 'this' (not to the mangling, just the argument list)
    // - parse the functions, their tokens were saved for deferred parsing (now)
    for (int b = 0; b < (int)functionDeclarators.size(); ++b) {
        // update signature
        if (functionDeclarators[b].function->hasImplicitThis())
            functionDeclarators[b].function->addThisParameter(type, intermediate.implicitThisName);
    }

    // All member functions get parsed inside the class/struct namespace and with the
    // class/struct members in a symbol-table level.
    parseContext.pushNamespace(structName);
    parseContext.pushThisScope(type, functionDeclarators);
    bool deferredSuccess = true;
    for (int b = 0; b < (int)functionDeclarators.size() && deferredSuccess; ++b) {
        // parse body
        pushTokenStream(functionDeclarators[b].body);
        if (! acceptFunctionBody(functionDeclarators[b], nodeList))
            deferredSuccess = false;
        popTokenStream();
    }
    parseContext.popThisScope();
    parseContext.popNamespace();

    return deferredSuccess;
}

// constantbuffer
//    : CONSTANTBUFFER LEFT_ANGLE type RIGHT_ANGLE
bool HlslGrammar::acceptConstantBufferType(TType& type)
{
    if (! acceptTokenClass(EHTokConstantBuffer))
        return false;

    if (! acceptTokenClass(EHTokLeftAngle)) {
        expected("left angle bracket");
        return false;
    }
    
    TType templateType;
    if (! acceptType(templateType)) {
        expected("type");
        return false;
    }

    if (! acceptTokenClass(EHTokRightAngle)) {
        expected("right angle bracket");
        return false;
    }

    TQualifier postDeclQualifier;
    postDeclQualifier.clear();
    postDeclQualifier.storage = EvqUniform;

    if (templateType.isStruct()) {
        // Make a block from the type parsed as the template argument
        TTypeList* typeList = templateType.getWritableStruct();
        new(&type) TType(typeList, "", postDeclQualifier); // sets EbtBlock

        type.getQualifier().storage = EvqUniform;

        return true;
    } else {
        parseContext.error(token.loc, "non-structure type in ConstantBuffer", "", "");
        return false;
    }
}

// texture_buffer
//    : TEXTUREBUFFER LEFT_ANGLE type RIGHT_ANGLE
bool HlslGrammar::acceptTextureBufferType(TType& type)
{
    if (! acceptTokenClass(EHTokTextureBuffer))
        return false;

    if (! acceptTokenClass(EHTokLeftAngle)) {
        expected("left angle bracket");
        return false;
    }
    
    TType templateType;
    if (! acceptType(templateType)) {
        expected("type");
        return false;
    }

    if (! acceptTokenClass(EHTokRightAngle)) {
        expected("right angle bracket");
        return false;
    }

    templateType.getQualifier().storage = EvqBuffer;
    templateType.getQualifier().readonly = true;

    TType blockType(templateType.getWritableStruct(), "", templateType.getQualifier());

    blockType.getQualifier().storage = EvqBuffer;
    blockType.getQualifier().readonly = true;

    type.shallowCopy(blockType);

    return true;
}


// struct_buffer
//    : APPENDSTRUCTUREDBUFFER
//    | BYTEADDRESSBUFFER
//    | CONSUMESTRUCTUREDBUFFER
//    | RWBYTEADDRESSBUFFER
//    | RWSTRUCTUREDBUFFER
//    | STRUCTUREDBUFFER
bool HlslGrammar::acceptStructBufferType(TType& type)
{
    const EHlslTokenClass structBuffType = peek();

    // TODO: globallycoherent
    bool hasTemplateType = true;
    bool readonly = false;

    TStorageQualifier storage = EvqBuffer;
    TBuiltInVariable  builtinType = EbvNone;

    switch (structBuffType) {
    case EHTokAppendStructuredBuffer:
        builtinType = EbvAppendConsume;
        break;
    case EHTokByteAddressBuffer:
        hasTemplateType = false;
        readonly = true;
        builtinType = EbvByteAddressBuffer;
        break;
    case EHTokConsumeStructuredBuffer:
        builtinType = EbvAppendConsume;
        break;
    case EHTokRWByteAddressBuffer:
        hasTemplateType = false;
        builtinType = EbvRWByteAddressBuffer;
        break;
    case EHTokRWStructuredBuffer:
        builtinType = EbvRWStructuredBuffer;
        break;
    case EHTokStructuredBuffer:
        builtinType = EbvStructuredBuffer;
        readonly = true;
        break;
    default:
        return false;  // not a structure buffer type
    }

    advanceToken();  // consume the structure keyword

    // type on which this StructedBuffer is templatized.  E.g, StructedBuffer<MyStruct> ==> MyStruct
    TType* templateType = new TType;

    if (hasTemplateType) {
        if (! acceptTokenClass(EHTokLeftAngle)) {
            expected("left angle bracket");
            return false;
        }
    
        if (! acceptType(*templateType)) {
            expected("type");
            return false;
        }
        if (! acceptTokenClass(EHTokRightAngle)) {
            expected("right angle bracket");
            return false;
        }
    } else {
        // byte address buffers have no explicit type.
        TType uintType(EbtUint, storage);
        templateType->shallowCopy(uintType);
    }

    // Create an unsized array out of that type.
    // TODO: does this work if it's already an array type?
    TArraySizes* unsizedArray = new TArraySizes;
    unsizedArray->addInnerSize(UnsizedArraySize);
    templateType->transferArraySizes(unsizedArray);
    templateType->getQualifier().storage = storage;

    // field name is canonical for all structbuffers
    templateType->setFieldName("@data");

    TTypeList* blockStruct = new TTypeList;
    TTypeLoc  member = { templateType, token.loc };
    blockStruct->push_back(member);

    // This is the type of the buffer block (SSBO)
    TType blockType(blockStruct, "", templateType->getQualifier());

    blockType.getQualifier().storage = storage;
    blockType.getQualifier().readonly = readonly;
    blockType.getQualifier().builtIn = builtinType;

    // We may have created an equivalent type before, in which case we should use its
    // deep structure.
    parseContext.shareStructBufferType(blockType);

    type.shallowCopy(blockType);

    return true;
}

// struct_declaration_list
//      : struct_declaration SEMI_COLON struct_declaration SEMI_COLON ...
//
// struct_declaration
//      : attributes fully_specified_type struct_declarator COMMA struct_declarator ...
//      | attributes fully_specified_type IDENTIFIER function_parameters post_decls compound_statement // member-function definition
//
// struct_declarator
//      : IDENTIFIER post_decls
//      | IDENTIFIER array_specifier post_decls
//      | IDENTIFIER function_parameters post_decls                                         // member-function prototype
//
bool HlslGrammar::acceptStructDeclarationList(TTypeList*& typeList, TIntermNode*& nodeList,
                                              TVector<TFunctionDeclarator>& declarators)
{
    typeList = new TTypeList();
    HlslToken idToken;

    do {
        // success on seeing the RIGHT_BRACE coming up
        if (peekTokenClass(EHTokRightBrace))
            break;

        // struct_declaration

        // attributes
        TAttributes attributes;
        acceptAttributes(attributes);

        bool declarator_list = false;

        // fully_specified_type
        TType memberType;
        if (! acceptFullySpecifiedType(memberType, nodeList, attributes)) {
            expected("member type");
            return false;
        }
        
        // merge in the attributes
        parseContext.transferTypeAttributes(token.loc, attributes, memberType);

        // struct_declarator COMMA struct_declarator ...
        bool functionDefinitionAccepted = false;
        do {
            if (! acceptIdentifier(idToken)) {
                expected("member name");
                return false;
            }

            if (peekTokenClass(EHTokLeftParen)) {
                // function_parameters
                if (!declarator_list) {
                    declarators.resize(declarators.size() + 1);
                    // request a token stream for deferred processing
                    functionDefinitionAccepted = acceptMemberFunctionDefinition(nodeList, memberType, *idToken.string,
                                                                                declarators.back());
                    if (functionDefinitionAccepted)
                        break;
                }
                expected("member-function definition");
                return false;
            } else {
                // add it to the list of members
                TTypeLoc member = { new TType(EbtVoid), token.loc };
                member.type->shallowCopy(memberType);
                member.type->setFieldName(*idToken.string);
                typeList->push_back(member);

                // array_specifier
                TArraySizes* arraySizes = nullptr;
                acceptArraySpecifier(arraySizes);
                if (arraySizes)
                    typeList->back().type->transferArraySizes(arraySizes);

                acceptPostDecls(member.type->getQualifier());

                // EQUAL assignment_expression
                if (acceptTokenClass(EHTokAssign)) {
                    parseContext.warn(idToken.loc, "struct-member initializers ignored", "typedef", "");
                    TIntermTyped* expressionNode = nullptr;
                    if (! acceptAssignmentExpression(expressionNode)) {
                        expected("initializer");
                        return false;
                    }
                }
            }
            // success on seeing the SEMICOLON coming up
            if (peekTokenClass(EHTokSemicolon))
                break;

            // COMMA
            if (acceptTokenClass(EHTokComma))
                declarator_list = true;
            else {
                expected(",");
                return false;
            }

        } while (true);

        // SEMI_COLON
        if (! functionDefinitionAccepted && ! acceptTokenClass(EHTokSemicolon)) {
            expected(";");
            return false;
        }

    } while (true);

    return true;
}

// member_function_definition
//    | function_parameters post_decls compound_statement
//
// Expects type to have EvqGlobal for a static member and
// EvqTemporary for non-static member.
bool HlslGrammar::acceptMemberFunctionDefinition(TIntermNode*& nodeList, const TType& type, TString& memberName,
                                                 TFunctionDeclarator& declarator)
{
    bool accepted = false;

    TString* functionName = &memberName;
    parseContext.getFullNamespaceName(functionName);
    declarator.function = new TFunction(functionName, type);
    if (type.getQualifier().storage == EvqTemporary)
        declarator.function->setImplicitThis();
    else
        declarator.function->setIllegalImplicitThis();

    // function_parameters
    if (acceptFunctionParameters(*declarator.function)) {
        // post_decls
        acceptPostDecls(declarator.function->getWritableType().getQualifier());

        // compound_statement (function body definition)
        if (peekTokenClass(EHTokLeftBrace)) {
            declarator.loc = token.loc;
            declarator.body = new TVector<HlslToken>;
            accepted = acceptFunctionDefinition(declarator, nodeList, declarator.body);
        }
    } else
        expected("function parameter list");

    return accepted;
}

// function_parameters
//      : LEFT_PAREN parameter_declaration COMMA parameter_declaration ... RIGHT_PAREN
//      | LEFT_PAREN VOID RIGHT_PAREN
//
bool HlslGrammar::acceptFunctionParameters(TFunction& function)
{
    // LEFT_PAREN
    if (! acceptTokenClass(EHTokLeftParen))
        return false;

    // VOID RIGHT_PAREN
    if (! acceptTokenClass(EHTokVoid)) {
        do {
            // parameter_declaration
            if (! acceptParameterDeclaration(function))
                break;

            // COMMA
            if (! acceptTokenClass(EHTokComma))
                break;
        } while (true);
    }

    // RIGHT_PAREN
    if (! acceptTokenClass(EHTokRightParen)) {
        expected(")");
        return false;
    }

    return true;
}

// default_parameter_declaration
//      : EQUAL conditional_expression
//      : EQUAL initializer
bool HlslGrammar::acceptDefaultParameterDeclaration(const TType& type, TIntermTyped*& node)
{
    node = nullptr;

    // Valid not to have a default_parameter_declaration
    if (!acceptTokenClass(EHTokAssign))
        return true;

    if (!acceptConditionalExpression(node)) {
        if (!acceptInitializer(node))
            return false;

        // For initializer lists, we have to const-fold into a constructor for the type, so build
        // that.
        TFunction* constructor = parseContext.makeConstructorCall(token.loc, type);
        if (constructor == nullptr)  // cannot construct
            return false;

        TIntermTyped* arguments = nullptr;
        for (int i = 0; i < int(node->getAsAggregate()->getSequence().size()); i++)
            parseContext.handleFunctionArgument(constructor, arguments, node->getAsAggregate()->getSequence()[i]->getAsTyped());

        node = parseContext.handleFunctionCall(token.loc, constructor, node);
    }

    if (node == nullptr)
        return false;

    // If this is simply a constant, we can use it directly.
    if (node->getAsConstantUnion())
        return true;

    // Otherwise, it has to be const-foldable.
    TIntermTyped* origNode = node;

    node = intermediate.fold(node->getAsAggregate());

    if (node != nullptr && origNode != node)
        return true;

    parseContext.error(token.loc, "invalid default parameter value", "", "");

    return false;
}

// parameter_declaration
//      : attributes attributed_declaration
//
// attributed_declaration
//      : fully_specified_type post_decls [ = default_parameter_declaration ]
//      | fully_specified_type identifier array_specifier post_decls [ = default_parameter_declaration ]
//
bool HlslGrammar::acceptParameterDeclaration(TFunction& function)
{
    // attributes
    TAttributes attributes;
    acceptAttributes(attributes);

    // fully_specified_type
    TType* type = new TType;
    if (! acceptFullySpecifiedType(*type, attributes))
        return false;

    // merge in the attributes
    parseContext.transferTypeAttributes(token.loc, attributes, *type);

    // identifier
    HlslToken idToken;
    acceptIdentifier(idToken);

    // array_specifier
    TArraySizes* arraySizes = nullptr;
    acceptArraySpecifier(arraySizes);
    if (arraySizes) {
        if (arraySizes->hasUnsized()) {
            parseContext.error(token.loc, "function parameter requires array size", "[]", "");
            return false;
        }

        type->transferArraySizes(arraySizes);
    }

    // post_decls
    acceptPostDecls(type->getQualifier());

    TIntermTyped* defaultValue;
    if (!acceptDefaultParameterDeclaration(*type, defaultValue))
        return false;

    parseContext.paramFix(*type);

    // If any prior parameters have default values, all the parameters after that must as well.
    if (defaultValue == nullptr && function.getDefaultParamCount() > 0) {
        parseContext.error(idToken.loc, "invalid parameter after default value parameters", idToken.string->c_str(), "");
        return false;
    }

    TParameter param = { idToken.string, type, defaultValue };
    function.addParameter(param);

    return true;
}

// Do the work to create the function definition in addition to
// parsing the body (compound_statement).
//
// If 'deferredTokens' are passed in, just get the token stream,
// don't process.
//
bool HlslGrammar::acceptFunctionDefinition(TFunctionDeclarator& declarator, TIntermNode*& nodeList,
                                           TVector<HlslToken>* deferredTokens)
{
    parseContext.handleFunctionDeclarator(declarator.loc, *declarator.function, false /* not prototype */);

    if (deferredTokens)
        return captureBlockTokens(*deferredTokens);
    else
        return acceptFunctionBody(declarator, nodeList);
}

bool HlslGrammar::acceptFunctionBody(TFunctionDeclarator& declarator, TIntermNode*& nodeList)
{
    // we might get back an entry-point
    TIntermNode* entryPointNode = nullptr;

    // This does a pushScope()
    TIntermNode* functionNode = parseContext.handleFunctionDefinition(declarator.loc, *declarator.function,
                                                                      declarator.attributes, entryPointNode);

    // compound_statement
    TIntermNode* functionBody = nullptr;
    if (! acceptCompoundStatement(functionBody))
        return false;

    // this does a popScope()
    parseContext.handleFunctionBody(declarator.loc, *declarator.function, functionBody, functionNode);

    // Hook up the 1 or 2 function definitions.
    nodeList = intermediate.growAggregate(nodeList, functionNode);
    nodeList = intermediate.growAggregate(nodeList, entryPointNode);

    return true;
}

// Accept an expression with parenthesis around it, where
// the parenthesis ARE NOT expression parenthesis, but the
// syntactically required ones like in "if ( expression )".
//
// Also accepts a declaration expression; "if (int a = expression)".
//
// Note this one is not set up to be speculative; as it gives
// errors if not found.
//
bool HlslGrammar::acceptParenExpression(TIntermTyped*& expression)
{
    expression = nullptr;

    // LEFT_PAREN
    if (! acceptTokenClass(EHTokLeftParen))
        expected("(");

    bool decl = false;
    TIntermNode* declNode = nullptr;
    decl = acceptControlDeclaration(declNode);
    if (decl) {
        if (declNode == nullptr || declNode->getAsTyped() == nullptr) {
            expected("initialized declaration");
            return false;
        } else
            expression = declNode->getAsTyped();
    } else {
        // no declaration
        if (! acceptExpression(expression)) {
            expected("expression");
            return false;
        }
    }

    // RIGHT_PAREN
    if (! acceptTokenClass(EHTokRightParen))
        expected(")");

    return true;
}

// The top-level full expression recognizer.
//
// expression
//      : assignment_expression COMMA assignment_expression COMMA assignment_expression ...
//
bool HlslGrammar::acceptExpression(TIntermTyped*& node)
{
    node = nullptr;

    // assignment_expression
    if (! acceptAssignmentExpression(node))
        return false;

    if (! peekTokenClass(EHTokComma))
        return true;

    do {
        // ... COMMA
        TSourceLoc loc = token.loc;
        advanceToken();

        // ... assignment_expression
        TIntermTyped* rightNode = nullptr;
        if (! acceptAssignmentExpression(rightNode)) {
            expected("assignment expression");
            return false;
        }

        node = intermediate.addComma(node, rightNode, loc);

        if (! peekTokenClass(EHTokComma))
            return true;
    } while (true);
}

// initializer
//      : LEFT_BRACE RIGHT_BRACE
//      | LEFT_BRACE initializer_list RIGHT_BRACE
//
// initializer_list
//      : assignment_expression COMMA assignment_expression COMMA ...
//
bool HlslGrammar::acceptInitializer(TIntermTyped*& node)
{
    // LEFT_BRACE
    if (! acceptTokenClass(EHTokLeftBrace))
        return false;

    // RIGHT_BRACE
    TSourceLoc loc = token.loc;
    if (acceptTokenClass(EHTokRightBrace)) {
        // a zero-length initializer list
        node = intermediate.makeAggregate(loc);
        return true;
    }

    // initializer_list
    node = nullptr;
    do {
        // assignment_expression
        TIntermTyped* expr;
        if (! acceptAssignmentExpression(expr)) {
            expected("assignment expression in initializer list");
            return false;
        }

        const bool firstNode = (node == nullptr);

        node = intermediate.growAggregate(node, expr, loc);

        // If every sub-node in the list has qualifier EvqConst, the returned node becomes
        // EvqConst.  Otherwise, it becomes EvqTemporary. That doesn't happen with e.g.
        // EvqIn or EvqPosition, since the collection isn't EvqPosition if all the members are.
        if (firstNode && expr->getQualifier().storage == EvqConst)
            node->getQualifier().storage = EvqConst;
        else if (expr->getQualifier().storage != EvqConst)
            node->getQualifier().storage = EvqTemporary;

        // COMMA
        if (acceptTokenClass(EHTokComma)) {
            if (acceptTokenClass(EHTokRightBrace))  // allow trailing comma
                return true;
            continue;
        }

        // RIGHT_BRACE
        if (acceptTokenClass(EHTokRightBrace))
            return true;

        expected(", or }");
        return false;
    } while (true);
}

// Accept an assignment expression, where assignment operations
// associate right-to-left.  That is, it is implicit, for example
//
//    a op (b op (c op d))
//
// assigment_expression
//      : initializer
//      | conditional_expression
//      | conditional_expression assign_op conditional_expression assign_op conditional_expression ...
//
bool HlslGrammar::acceptAssignmentExpression(TIntermTyped*& node)
{
    // initializer
    if (peekTokenClass(EHTokLeftBrace)) {
        if (acceptInitializer(node))
            return true;

        expected("initializer");
        return false;
    }

    // conditional_expression
    if (! acceptConditionalExpression(node))
        return false;

    // assignment operation?
    TOperator assignOp = HlslOpMap::assignment(peek());
    if (assignOp == EOpNull)
        return true;

    // assign_op
    TSourceLoc loc = token.loc;
    advanceToken();

    // conditional_expression assign_op conditional_expression ...
    // Done by recursing this function, which automatically
    // gets the right-to-left associativity.
    TIntermTyped* rightNode = nullptr;
    if (! acceptAssignmentExpression(rightNode)) {
        expected("assignment expression");
        return false;
    }

    node = parseContext.handleAssign(loc, assignOp, node, rightNode);
    node = parseContext.handleLvalue(loc, "assign", node);

    if (node == nullptr) {
        parseContext.error(loc, "could not create assignment", "", "");
        return false;
    }

    if (! peekTokenClass(EHTokComma))
        return true;

    return true;
}

// Accept a conditional expression, which associates right-to-left,
// accomplished by the "true" expression calling down to lower
// precedence levels than this level.
//
// conditional_expression
//      : binary_expression
//      | binary_expression QUESTION expression COLON assignment_expression
//
bool HlslGrammar::acceptConditionalExpression(TIntermTyped*& node)
{
    // binary_expression
    if (! acceptBinaryExpression(node, PlLogicalOr))
        return false;

    if (! acceptTokenClass(EHTokQuestion))
        return true;

    node = parseContext.convertConditionalExpression(token.loc, node, false);
    if (node == nullptr)
        return false;

    ++parseContext.controlFlowNestingLevel;  // this only needs to work right if no errors

    TIntermTyped* trueNode = nullptr;
    if (! acceptExpression(trueNode)) {
        expected("expression after ?");
        return false;
    }
    TSourceLoc loc = token.loc;

    if (! acceptTokenClass(EHTokColon)) {
        expected(":");
        return false;
    }

    TIntermTyped* falseNode = nullptr;
    if (! acceptAssignmentExpression(falseNode)) {
        expected("expression after :");
        return false;
    }

    --parseContext.controlFlowNestingLevel;

    node = intermediate.addSelection(node, trueNode, falseNode, loc);

    return true;
}

// Accept a binary expression, for binary operations that
// associate left-to-right.  This is, it is implicit, for example
//
//    ((a op b) op c) op d
//
// binary_expression
//      : expression op expression op expression ...
//
// where 'expression' is the next higher level in precedence.
//
bool HlslGrammar::acceptBinaryExpression(TIntermTyped*& node, PrecedenceLevel precedenceLevel)
{
    if (precedenceLevel > PlMul)
        return acceptUnaryExpression(node);

    // assignment_expression
    if (! acceptBinaryExpression(node, (PrecedenceLevel)(precedenceLevel + 1)))
        return false;

    do {
        TOperator op = HlslOpMap::binary(peek());
        PrecedenceLevel tokenLevel = HlslOpMap::precedenceLevel(op);
        if (tokenLevel < precedenceLevel)
            return true;

        // ... op
        TSourceLoc loc = token.loc;
        advanceToken();

        // ... expression
        TIntermTyped* rightNode = nullptr;
        if (! acceptBinaryExpression(rightNode, (PrecedenceLevel)(precedenceLevel + 1))) {
            expected("expression");
            return false;
        }

        node = intermediate.addBinaryMath(op, node, rightNode, loc);
        if (node == nullptr) {
            parseContext.error(loc, "Could not perform requested binary operation", "", "");
            return false;
        }
    } while (true);
}

// unary_expression
//      : (type) unary_expression
//      | + unary_expression
//      | - unary_expression
//      | ! unary_expression
//      | ~ unary_expression
//      | ++ unary_expression
//      | -- unary_expression
//      | postfix_expression
//
bool HlslGrammar::acceptUnaryExpression(TIntermTyped*& node)
{
    // (type) unary_expression
    // Have to look two steps ahead, because this could be, e.g., a
    // postfix_expression instead, since that also starts with at "(".
    if (acceptTokenClass(EHTokLeftParen)) {
        TType castType;
        if (acceptType(castType)) {
            // recognize any array_specifier as part of the type
            TArraySizes* arraySizes = nullptr;
            acceptArraySpecifier(arraySizes);
            if (arraySizes != nullptr)
                castType.transferArraySizes(arraySizes);
            TSourceLoc loc = token.loc;
            if (acceptTokenClass(EHTokRightParen)) {
                // We've matched "(type)" now, get the expression to cast
                if (! acceptUnaryExpression(node))
                    return false;

                // Hook it up like a constructor
                TFunction* constructorFunction = parseContext.makeConstructorCall(loc, castType);
                if (constructorFunction == nullptr) {
                    expected("type that can be constructed");
                    return false;
                }
                TIntermTyped* arguments = nullptr;
                parseContext.handleFunctionArgument(constructorFunction, arguments, node);
                node = parseContext.handleFunctionCall(loc, constructorFunction, arguments);

                return node != nullptr;
            } else {
                // This could be a parenthesized constructor, ala (int(3)), and we just accepted
                // the '(int' part.  We must back up twice.
                recedeToken();
                recedeToken();

                // Note, there are no array constructors like
                //   (float[2](...))
                if (arraySizes != nullptr)
                    parseContext.error(loc, "parenthesized array constructor not allowed", "([]())", "", "");
            }
        } else {
            // This isn't a type cast, but it still started "(", so if it is a
            // unary expression, it can only be a postfix_expression, so try that.
            // Back it up first.
            recedeToken();
            return acceptPostfixExpression(node);
        }
    }

    // peek for "op unary_expression"
    TOperator unaryOp = HlslOpMap::preUnary(peek());

    // postfix_expression (if no unary operator)
    if (unaryOp == EOpNull)
        return acceptPostfixExpression(node);

    // op unary_expression
    TSourceLoc loc = token.loc;
    advanceToken();
    if (! acceptUnaryExpression(node))
        return false;

    // + is a no-op
    if (unaryOp == EOpAdd)
        return true;

    node = intermediate.addUnaryMath(unaryOp, node, loc);

    // These unary ops require lvalues
    if (unaryOp == EOpPreIncrement || unaryOp == EOpPreDecrement)
        node = parseContext.handleLvalue(loc, "unary operator", node);

    return node != nullptr;
}

// postfix_expression
//      : LEFT_PAREN expression RIGHT_PAREN
//      | literal
//      | constructor
//      | IDENTIFIER [ COLONCOLON IDENTIFIER [ COLONCOLON IDENTIFIER ... ] ]
//      | function_call
//      | postfix_expression LEFT_BRACKET integer_expression RIGHT_BRACKET
//      | postfix_expression DOT IDENTIFIER
//      | postfix_expression DOT IDENTIFIER arguments
//      | postfix_expression arguments
//      | postfix_expression INC_OP
//      | postfix_expression DEC_OP
//
bool HlslGrammar::acceptPostfixExpression(TIntermTyped*& node)
{
    // Not implemented as self-recursive:
    // The logical "right recursion" is done with a loop at the end

    // idToken will pick up either a variable or a function name in a function call
    HlslToken idToken;

    // Find something before the postfix operations, as they can't operate
    // on nothing.  So, no "return true", they fall through, only "return false".
    if (acceptTokenClass(EHTokLeftParen)) {
        // LEFT_PAREN expression RIGHT_PAREN
        if (! acceptExpression(node)) {
            expected("expression");
            return false;
        }
        if (! acceptTokenClass(EHTokRightParen)) {
            expected(")");
            return false;
        }
    } else if (acceptLiteral(node)) {
        // literal (nothing else to do yet)
    } else if (acceptConstructor(node)) {
        // constructor (nothing else to do yet)
    } else if (acceptIdentifier(idToken)) {
        // user-type, namespace name, variable, or function name
        TString* fullName = idToken.string;
        while (acceptTokenClass(EHTokColonColon)) {
            // user-type or namespace name
            fullName = NewPoolTString(fullName->c_str());
            fullName->append(parseContext.scopeMangler);
            if (acceptIdentifier(idToken))
                fullName->append(*idToken.string);
            else {
                expected("identifier after ::");
                return false;
            }
        }
        if (! peekTokenClass(EHTokLeftParen)) {
            node = parseContext.handleVariable(idToken.loc, fullName);
            if (node == nullptr)
                return false;
        } else if (acceptFunctionCall(idToken.loc, *fullName, node, nullptr)) {
            // function_call (nothing else to do yet)
        } else {
            expected("function call arguments");
            return false;
        }
    } else {
        // nothing found, can't post operate
        return false;
    }

    // Something was found, chain as many postfix operations as exist.
    do {
        TSourceLoc loc = token.loc;
        TOperator postOp = HlslOpMap::postUnary(peek());

        // Consume only a valid post-unary operator, otherwise we are done.
        switch (postOp) {
        case EOpIndexDirectStruct:
        case EOpIndexIndirect:
        case EOpPostIncrement:
        case EOpPostDecrement:
        case EOpScoping:
            advanceToken();
            break;
        default:
            return true;
        }

        // We have a valid post-unary operator, process it.
        switch (postOp) {
        case EOpScoping:
        case EOpIndexDirectStruct:
        {
            // DOT IDENTIFIER
            // includes swizzles, member variables, and member functions
            HlslToken field;
            if (! acceptIdentifier(field)) {
                expected("swizzle or member");
                return false;
            }

            if (peekTokenClass(EHTokLeftParen)) {
                // member function
                TIntermTyped* thisNode = node;

                // arguments
                if (! acceptFunctionCall(field.loc, *field.string, node, thisNode)) {
                    expected("function parameters");
                    return false;
                }
            } else
                node = parseContext.handleDotDereference(field.loc, node, *field.string);

            break;
        }
        case EOpIndexIndirect:
        {
            // LEFT_BRACKET integer_expression RIGHT_BRACKET
            TIntermTyped* indexNode = nullptr;
            if (! acceptExpression(indexNode) ||
                ! peekTokenClass(EHTokRightBracket)) {
                expected("expression followed by ']'");
                return false;
            }
            advanceToken();
            node = parseContext.handleBracketDereference(indexNode->getLoc(), node, indexNode);
            if (node == nullptr)
                return false;
            break;
        }
        case EOpPostIncrement:
            // INC_OP
            // fall through
        case EOpPostDecrement:
            // DEC_OP
            node = intermediate.addUnaryMath(postOp, node, loc);
            node = parseContext.handleLvalue(loc, "unary operator", node);
            break;
        default:
            assert(0);
            break;
        }
    } while (true);
}

// constructor
//      : type argument_list
//
bool HlslGrammar::acceptConstructor(TIntermTyped*& node)
{
    // type
    TType type;
    if (acceptType(type)) {
        TFunction* constructorFunction = parseContext.makeConstructorCall(token.loc, type);
        if (constructorFunction == nullptr)
            return false;

        // arguments
        TIntermTyped* arguments = nullptr;
        if (! acceptArguments(constructorFunction, arguments)) {
            // It's possible this is a type keyword used as an identifier.  Put the token back
            // for later use.
            recedeToken();
            return false;
        }

        // hook it up
        node = parseContext.handleFunctionCall(arguments->getLoc(), constructorFunction, arguments);

        return node != nullptr;
    }

    return false;
}

// The function_call identifier was already recognized, and passed in as idToken.
//
// function_call
//      : [idToken] arguments
//
bool HlslGrammar::acceptFunctionCall(const TSourceLoc& loc, TString& name, TIntermTyped*& node, TIntermTyped* baseObject)
{
    // name
    TString* functionName = nullptr;
    if (baseObject == nullptr) {
        functionName = &name;
    } else if (parseContext.isBuiltInMethod(loc, baseObject, name)) {
        // Built-in methods are not in the symbol table as methods, but as global functions
        // taking an explicit 'this' as the first argument.
        functionName = NewPoolTString(BUILTIN_PREFIX);
        functionName->append(name);
    } else {
        if (! baseObject->getType().isStruct()) {
            expected("structure");
            return false;
        }
        functionName = NewPoolTString("");
        functionName->append(baseObject->getType().getTypeName());
        parseContext.addScopeMangler(*functionName);
        functionName->append(name);
    }

    // function
    TFunction* function = new TFunction(functionName, TType(EbtVoid));

    // arguments
    TIntermTyped* arguments = nullptr;
    if (baseObject != nullptr) {
        // Non-static member functions have an implicit first argument of the base object.
        parseContext.handleFunctionArgument(function, arguments, baseObject);
    }
    if (! acceptArguments(function, arguments))
        return false;

    // call
    node = parseContext.handleFunctionCall(loc, function, arguments);

    return node != nullptr;
}

// arguments
//      : LEFT_PAREN expression COMMA expression COMMA ... RIGHT_PAREN
//
// The arguments are pushed onto the 'function' argument list and
// onto the 'arguments' aggregate.
//
bool HlslGrammar::acceptArguments(TFunction* function, TIntermTyped*& arguments)
{
    // LEFT_PAREN
    if (! acceptTokenClass(EHTokLeftParen))
        return false;

    // RIGHT_PAREN
    if (acceptTokenClass(EHTokRightParen))
        return true;

    // must now be at least one expression...
    do {
        // expression
        TIntermTyped* arg;
        if (! acceptAssignmentExpression(arg))
            return false;

        // hook it up
        parseContext.handleFunctionArgument(function, arguments, arg);

        // COMMA
        if (! acceptTokenClass(EHTokComma))
            break;
    } while (true);

    // RIGHT_PAREN
    if (! acceptTokenClass(EHTokRightParen)) {
        expected(")");
        return false;
    }

    return true;
}

bool HlslGrammar::acceptLiteral(TIntermTyped*& node)
{
    switch (token.tokenClass) {
    case EHTokIntConstant:
        node = intermediate.addConstantUnion(token.i, token.loc, true);
        break;
    case EHTokUintConstant:
        node = intermediate.addConstantUnion(token.u, token.loc, true);
        break;
    case EHTokFloat16Constant:
        node = intermediate.addConstantUnion(token.d, EbtFloat16, token.loc, true);
        break;
    case EHTokFloatConstant:
        node = intermediate.addConstantUnion(token.d, EbtFloat, token.loc, true);
        break;
    case EHTokDoubleConstant:
        node = intermediate.addConstantUnion(token.d, EbtDouble, token.loc, true);
        break;
    case EHTokBoolConstant:
        node = intermediate.addConstantUnion(token.b, token.loc, true);
        break;
    case EHTokStringConstant:
        node = intermediate.addConstantUnion(token.string, token.loc, true);
        break;

    default:
        return false;
    }

    advanceToken();

    return true;
}

// simple_statement
//      : SEMICOLON
//      | declaration_statement
//      | expression SEMICOLON
//
bool HlslGrammar::acceptSimpleStatement(TIntermNode*& statement)
{
    // SEMICOLON
    if (acceptTokenClass(EHTokSemicolon))
        return true;

    // declaration
    if (acceptDeclaration(statement))
        return true;

    // expression
    TIntermTyped* node;
    if (acceptExpression(node))
        statement = node;
    else
        return false;

    // SEMICOLON (following an expression)
    if (acceptTokenClass(EHTokSemicolon))
        return true;
    else {
        expected(";");
        return false;
    }
}

// compound_statement
//      : LEFT_CURLY statement statement ... RIGHT_CURLY
//
bool HlslGrammar::acceptCompoundStatement(TIntermNode*& retStatement)
{
    TIntermAggregate* compoundStatement = nullptr;

    // LEFT_CURLY
    if (! acceptTokenClass(EHTokLeftBrace))
        return false;

    // statement statement ...
    TIntermNode* statement = nullptr;
    while (acceptStatement(statement)) {
        TIntermBranch* branch = statement ? statement->getAsBranchNode() : nullptr;
        if (branch != nullptr && (branch->getFlowOp() == EOpCase ||
                                  branch->getFlowOp() == EOpDefault)) {
            // hook up individual subsequences within a switch statement
            parseContext.wrapupSwitchSubsequence(compoundStatement, statement);
            compoundStatement = nullptr;
        } else {
            // hook it up to the growing compound statement
            compoundStatement = intermediate.growAggregate(compoundStatement, statement);
        }
    }
    if (compoundStatement)
        compoundStatement->setOperator(EOpSequence);

    retStatement = compoundStatement;

    // RIGHT_CURLY
    return acceptTokenClass(EHTokRightBrace);
}

bool HlslGrammar::acceptScopedStatement(TIntermNode*& statement)
{
    parseContext.pushScope();
    bool result = acceptStatement(statement);
    parseContext.popScope();

    return result;
}

bool HlslGrammar::acceptScopedCompoundStatement(TIntermNode*& statement)
{
    parseContext.pushScope();
    bool result = acceptCompoundStatement(statement);
    parseContext.popScope();

    return result;
}

// statement
//      : attributes attributed_statement
//
// attributed_statement
//      : compound_statement
//      | simple_statement
//      | selection_statement
//      | switch_statement
//      | case_label
//      | default_label
//      | iteration_statement
//      | jump_statement
//
bool HlslGrammar::acceptStatement(TIntermNode*& statement)
{
    statement = nullptr;

    // attributes
    TAttributes attributes;
    acceptAttributes(attributes);

    // attributed_statement
    switch (peek()) {
    case EHTokLeftBrace:
        return acceptScopedCompoundStatement(statement);

    case EHTokIf:
        return acceptSelectionStatement(statement, attributes);

    case EHTokSwitch:
        return acceptSwitchStatement(statement, attributes);

    case EHTokFor:
    case EHTokDo:
    case EHTokWhile:
        return acceptIterationStatement(statement, attributes);

    case EHTokContinue:
    case EHTokBreak:
    case EHTokDiscard:
    case EHTokReturn:
        return acceptJumpStatement(statement);

    case EHTokCase:
        return acceptCaseLabel(statement);
    case EHTokDefault:
        return acceptDefaultLabel(statement);

    case EHTokRightBrace:
        // Performance: not strictly necessary, but stops a bunch of hunting early,
        // and is how sequences of statements end.
        return false;

    default:
        return acceptSimpleStatement(statement);
    }

    return true;
}

// attributes
//      : [zero or more:] bracketed-attribute
//
// bracketed-attribute:
//      : LEFT_BRACKET scoped-attribute RIGHT_BRACKET
//      : LEFT_BRACKET LEFT_BRACKET scoped-attribute RIGHT_BRACKET RIGHT_BRACKET
//
// scoped-attribute:
//      : attribute
//      | namespace COLON COLON attribute
//
// attribute:
//      : UNROLL
//      | UNROLL LEFT_PAREN literal RIGHT_PAREN
//      | FASTOPT
//      | ALLOW_UAV_CONDITION
//      | BRANCH
//      | FLATTEN
//      | FORCECASE
//      | CALL
//      | DOMAIN
//      | EARLYDEPTHSTENCIL
//      | INSTANCE
//      | MAXTESSFACTOR
//      | OUTPUTCONTROLPOINTS
//      | OUTPUTTOPOLOGY
//      | PARTITIONING
//      | PATCHCONSTANTFUNC
//      | NUMTHREADS LEFT_PAREN x_size, y_size,z z_size RIGHT_PAREN
//
void HlslGrammar::acceptAttributes(TAttributes& attributes)
{
    // For now, accept the [ XXX(X) ] syntax, but drop all but
    // numthreads, which is used to set the CS local size.
    // TODO: subset to correct set?  Pass on?
    do {
        HlslToken attributeToken;

        // LEFT_BRACKET?
        if (! acceptTokenClass(EHTokLeftBracket))
            return;
        // another LEFT_BRACKET?
        bool doubleBrackets = false;
        if (acceptTokenClass(EHTokLeftBracket))
            doubleBrackets = true;

        // attribute? (could be namespace; will adjust later)
        if (!acceptIdentifier(attributeToken)) {
            if (!peekTokenClass(EHTokRightBracket)) {
                expected("namespace or attribute identifier");
                advanceToken();
            }
        }

        TString nameSpace;
        if (acceptTokenClass(EHTokColonColon)) {
            // namespace COLON COLON
            nameSpace = *attributeToken.string;
            // attribute
            if (!acceptIdentifier(attributeToken)) {
                expected("attribute identifier");
                return;
            }
        }

        TIntermAggregate* expressions = nullptr;

        // (x, ...)
        if (acceptTokenClass(EHTokLeftParen)) {
            expressions = new TIntermAggregate;

            TIntermTyped* node;
            bool expectingExpression = false;

            while (acceptAssignmentExpression(node)) {
                expectingExpression = false;
                expressions->getSequence().push_back(node);
                if (acceptTokenClass(EHTokComma))
                    expectingExpression = true;
            }

            // 'expressions' is an aggregate with the expressions in it
            if (! acceptTokenClass(EHTokRightParen))
                expected(")");

            // Error for partial or missing expression
            if (expectingExpression || expressions->getSequence().empty())
                expected("expression");
        }

        // RIGHT_BRACKET
        if (!acceptTokenClass(EHTokRightBracket)) {
            expected("]");
            return;
        }
        // another RIGHT_BRACKET?
        if (doubleBrackets && !acceptTokenClass(EHTokRightBracket)) {
            expected("]]");
            return;
        }

        // Add any values we found into the attribute map.
        if (attributeToken.string != nullptr) {
            TAttributeType attributeType = parseContext.attributeFromName(nameSpace, *attributeToken.string);
            if (attributeType == EatNone)
                parseContext.warn(attributeToken.loc, "unrecognized attribute", attributeToken.string->c_str(), "");
            else {
                TAttributeArgs attributeArgs = { attributeType, expressions };
                attributes.push_back(attributeArgs);
            }
        }
    } while (true);
}

// selection_statement
//      : IF LEFT_PAREN expression RIGHT_PAREN statement
//      : IF LEFT_PAREN expression RIGHT_PAREN statement ELSE statement
//
bool HlslGrammar::acceptSelectionStatement(TIntermNode*& statement, const TAttributes& attributes)
{
    TSourceLoc loc = token.loc;

    // IF
    if (! acceptTokenClass(EHTokIf))
        return false;

    // so that something declared in the condition is scoped to the lifetimes
    // of the then-else statements
    parseContext.pushScope();

    // LEFT_PAREN expression RIGHT_PAREN
    TIntermTyped* condition;
    if (! acceptParenExpression(condition))
        return false;
    condition = parseContext.convertConditionalExpression(loc, condition);
    if (condition == nullptr)
        return false;

    // create the child statements
    TIntermNodePair thenElse = { nullptr, nullptr };

    ++parseContext.controlFlowNestingLevel;  // this only needs to work right if no errors

    // then statement
    if (! acceptScopedStatement(thenElse.node1)) {
        expected("then statement");
        return false;
    }

    // ELSE
    if (acceptTokenClass(EHTokElse)) {
        // else statement
        if (! acceptScopedStatement(thenElse.node2)) {
            expected("else statement");
            return false;
        }
    }

    // Put the pieces together
    statement = intermediate.addSelection(condition, thenElse, loc);
    parseContext.handleSelectionAttributes(loc, statement->getAsSelectionNode(), attributes);

    parseContext.popScope();
    --parseContext.controlFlowNestingLevel;

    return true;
}

// switch_statement
//      : SWITCH LEFT_PAREN expression RIGHT_PAREN compound_statement
//
bool HlslGrammar::acceptSwitchStatement(TIntermNode*& statement, const TAttributes& attributes)
{
    // SWITCH
    TSourceLoc loc = token.loc;

    if (! acceptTokenClass(EHTokSwitch))
        return false;

    // LEFT_PAREN expression RIGHT_PAREN
    parseContext.pushScope();
    TIntermTyped* switchExpression;
    if (! acceptParenExpression(switchExpression)) {
        parseContext.popScope();
        return false;
    }

    // compound_statement
    parseContext.pushSwitchSequence(new TIntermSequence);

    ++parseContext.controlFlowNestingLevel;
    bool statementOkay = acceptCompoundStatement(statement);
    --parseContext.controlFlowNestingLevel;

    if (statementOkay)
        statement = parseContext.addSwitch(loc, switchExpression, statement ? statement->getAsAggregate() : nullptr,
                                           attributes);

    parseContext.popSwitchSequence();
    parseContext.popScope();

    return statementOkay;
}

// iteration_statement
//      : WHILE LEFT_PAREN condition RIGHT_PAREN statement
//      | DO LEFT_BRACE statement RIGHT_BRACE WHILE LEFT_PAREN expression RIGHT_PAREN SEMICOLON
//      | FOR LEFT_PAREN for_init_statement for_rest_statement RIGHT_PAREN statement
//
// Non-speculative, only call if it needs to be found; WHILE or DO or FOR already seen.
bool HlslGrammar::acceptIterationStatement(TIntermNode*& statement, const TAttributes& attributes)
{
    TSourceLoc loc = token.loc;
    TIntermTyped* condition = nullptr;

    EHlslTokenClass loop = peek();
    assert(loop == EHTokDo || loop == EHTokFor || loop == EHTokWhile);

    //  WHILE or DO or FOR
    advanceToken();

    TIntermLoop* loopNode = nullptr;
    switch (loop) {
    case EHTokWhile:
        // so that something declared in the condition is scoped to the lifetime
        // of the while sub-statement
        parseContext.pushScope();  // this only needs to work right if no errors
        parseContext.nestLooping();
        ++parseContext.controlFlowNestingLevel;

        // LEFT_PAREN condition RIGHT_PAREN
        if (! acceptParenExpression(condition))
            return false;
        condition = parseContext.convertConditionalExpression(loc, condition);
        if (condition == nullptr)
            return false;

        // statement
        if (! acceptScopedStatement(statement)) {
            expected("while sub-statement");
            return false;
        }

        parseContext.unnestLooping();
        parseContext.popScope();
        --parseContext.controlFlowNestingLevel;

        loopNode = intermediate.addLoop(statement, condition, nullptr, true, loc);
        statement = loopNode;
        break;

    case EHTokDo:
        parseContext.nestLooping();  // this only needs to work right if no errors
        ++parseContext.controlFlowNestingLevel;

        // statement
        if (! acceptScopedStatement(statement)) {
            expected("do sub-statement");
            return false;
        }

        // WHILE
        if (! acceptTokenClass(EHTokWhile)) {
            expected("while");
            return false;
        }

        // LEFT_PAREN condition RIGHT_PAREN
        if (! acceptParenExpression(condition))
            return false;
        condition = parseContext.convertConditionalExpression(loc, condition);
        if (condition == nullptr)
            return false;

        if (! acceptTokenClass(EHTokSemicolon))
            expected(";");

        parseContext.unnestLooping();
        --parseContext.controlFlowNestingLevel;

        loopNode = intermediate.addLoop(statement, condition, 0, false, loc);
        statement = loopNode;
        break;

    case EHTokFor:
    {
        // LEFT_PAREN
        if (! acceptTokenClass(EHTokLeftParen))
            expected("(");

        // so that something declared in the condition is scoped to the lifetime
        // of the for sub-statement
        parseContext.pushScope();

        // initializer
        TIntermNode* initNode = nullptr;
        if (! acceptSimpleStatement(initNode))
            expected("for-loop initializer statement");

        parseContext.nestLooping();  // this only needs to work right if no errors
        ++parseContext.controlFlowNestingLevel;

        // condition SEMI_COLON
        acceptExpression(condition);
        if (! acceptTokenClass(EHTokSemicolon))
            expected(";");
        if (condition != nullptr) {
            condition = parseContext.convertConditionalExpression(loc, condition);
            if (condition == nullptr)
                return false;
        }

        // iterator SEMI_COLON
        TIntermTyped* iterator = nullptr;
        acceptExpression(iterator);
        if (! acceptTokenClass(EHTokRightParen))
            expected(")");

        // statement
        if (! acceptScopedStatement(statement)) {
            expected("for sub-statement");
            return false;
        }

        statement = intermediate.addForLoop(statement, initNode, condition, iterator, true, loc, loopNode);

        parseContext.popScope();
        parseContext.unnestLooping();
        --parseContext.controlFlowNestingLevel;

        break;
    }

    default:
        return false;
    }

    parseContext.handleLoopAttributes(loc, loopNode, attributes);
    return true;
}

// jump_statement
//      : CONTINUE SEMICOLON
//      | BREAK SEMICOLON
//      | DISCARD SEMICOLON
//      | RETURN SEMICOLON
//      | RETURN expression SEMICOLON
//
bool HlslGrammar::acceptJumpStatement(TIntermNode*& statement)
{
    EHlslTokenClass jump = peek();
    switch (jump) {
    case EHTokContinue:
    case EHTokBreak:
    case EHTokDiscard:
    case EHTokReturn:
        advanceToken();
        break;
    default:
        // not something we handle in this function
        return false;
    }

    switch (jump) {
    case EHTokContinue:
        statement = intermediate.addBranch(EOpContinue, token.loc);
        if (parseContext.loopNestingLevel == 0) {
            expected("loop");
            return false;
        }
        break;
    case EHTokBreak:
        statement = intermediate.addBranch(EOpBreak, token.loc);
        if (parseContext.loopNestingLevel == 0 && parseContext.switchSequenceStack.size() == 0) {
            expected("loop or switch");
            return false;
        }
        break;
    case EHTokDiscard:
        statement = intermediate.addBranch(EOpKill, token.loc);
        break;

    case EHTokReturn:
    {
        // expression
        TIntermTyped* node;
        if (acceptExpression(node)) {
            // hook it up
            statement = parseContext.handleReturnValue(token.loc, node);
        } else
            statement = intermediate.addBranch(EOpReturn, token.loc);
        break;
    }

    default:
        assert(0);
        return false;
    }

    // SEMICOLON
    if (! acceptTokenClass(EHTokSemicolon))
        expected(";");

    return true;
}

// case_label
//      : CASE expression COLON
//
bool HlslGrammar::acceptCaseLabel(TIntermNode*& statement)
{
    TSourceLoc loc = token.loc;
    if (! acceptTokenClass(EHTokCase))
        return false;

    TIntermTyped* expression;
    if (! acceptExpression(expression)) {
        expected("case expression");
        return false;
    }

    if (! acceptTokenClass(EHTokColon)) {
        expected(":");
        return false;
    }

    statement = parseContext.intermediate.addBranch(EOpCase, expression, loc);

    return true;
}

// default_label
//      : DEFAULT COLON
//
bool HlslGrammar::acceptDefaultLabel(TIntermNode*& statement)
{
    TSourceLoc loc = token.loc;
    if (! acceptTokenClass(EHTokDefault))
        return false;

    if (! acceptTokenClass(EHTokColon)) {
        expected(":");
        return false;
    }

    statement = parseContext.intermediate.addBranch(EOpDefault, loc);

    return true;
}

// array_specifier
//      : LEFT_BRACKET integer_expression RGHT_BRACKET ... // optional
//      : LEFT_BRACKET RGHT_BRACKET // optional
//
void HlslGrammar::acceptArraySpecifier(TArraySizes*& arraySizes)
{
    arraySizes = nullptr;

    // Early-out if there aren't any array dimensions
    if (!peekTokenClass(EHTokLeftBracket))
        return;

    // If we get here, we have at least one array dimension.  This will track the sizes we find.
    arraySizes = new TArraySizes;

    // Collect each array dimension.
    while (acceptTokenClass(EHTokLeftBracket)) {
        TSourceLoc loc = token.loc;
        TIntermTyped* sizeExpr = nullptr;

        // Array sizing expression is optional.  If omitted, array will be later sized by initializer list.
        const bool hasArraySize = acceptAssignmentExpression(sizeExpr);

        if (! acceptTokenClass(EHTokRightBracket)) {
            expected("]");
            return;
        }

        if (hasArraySize) {
            TArraySize arraySize;
            parseContext.arraySizeCheck(loc, sizeExpr, arraySize);
            arraySizes->addInnerSize(arraySize);
        } else {
            arraySizes->addInnerSize(0);  // sized by initializers.
        }
    }
}

// post_decls
//      : COLON semantic // optional
//        COLON PACKOFFSET LEFT_PAREN c[Subcomponent][.component] RIGHT_PAREN // optional
//        COLON REGISTER LEFT_PAREN [shader_profile,] Type#[subcomp]opt (COMMA SPACEN)opt RIGHT_PAREN // optional
//        COLON LAYOUT layout_qualifier_list
//        annotations // optional
//
// Return true if any tokens were accepted. That is,
// false can be returned on successfully recognizing nothing,
// not necessarily meaning bad syntax.
//
bool HlslGrammar::acceptPostDecls(TQualifier& qualifier)
{
    bool found = false;

    do {
        // COLON
        if (acceptTokenClass(EHTokColon)) {
            found = true;
            HlslToken idToken;
            if (peekTokenClass(EHTokLayout))
                acceptLayoutQualifierList(qualifier);
            else if (acceptTokenClass(EHTokPackOffset)) {
                // PACKOFFSET LEFT_PAREN c[Subcomponent][.component] RIGHT_PAREN
                if (! acceptTokenClass(EHTokLeftParen)) {
                    expected("(");
                    return false;
                }
                HlslToken locationToken;
                if (! acceptIdentifier(locationToken)) {
                    expected("c[subcomponent][.component]");
                    return false;
                }
                HlslToken componentToken;
                if (acceptTokenClass(EHTokDot)) {
                    if (! acceptIdentifier(componentToken)) {
                        expected("component");
                        return false;
                    }
                }
                if (! acceptTokenClass(EHTokRightParen)) {
                    expected(")");
                    break;
                }
                parseContext.handlePackOffset(locationToken.loc, qualifier, *locationToken.string, componentToken.string);
            } else if (! acceptIdentifier(idToken)) {
                expected("layout, semantic, packoffset, or register");
                return false;
            } else if (*idToken.string == "register") {
                // REGISTER LEFT_PAREN [shader_profile,] Type#[subcomp]opt (COMMA SPACEN)opt RIGHT_PAREN
                // LEFT_PAREN
                if (! acceptTokenClass(EHTokLeftParen)) {
                    expected("(");
                    return false;
                }
                HlslToken registerDesc;  // for Type#
                HlslToken profile;
                if (! acceptIdentifier(registerDesc)) {
                    expected("register number description");
                    return false;
                }
                if (registerDesc.string->size() > 1 && !isdigit((*registerDesc.string)[1]) &&
                                                       acceptTokenClass(EHTokComma)) {
                    // Then we didn't really see the registerDesc yet, it was
                    // actually the profile.  Adjust...
                    profile = registerDesc;
                    if (! acceptIdentifier(registerDesc)) {
                        expected("register number description");
                        return false;
                    }
                }
                int subComponent = 0;
                if (acceptTokenClass(EHTokLeftBracket)) {
                    // LEFT_BRACKET subcomponent RIGHT_BRACKET
                    if (! peekTokenClass(EHTokIntConstant)) {
                        expected("literal integer");
                        return false;
                    }
                    subComponent = token.i;
                    advanceToken();
                    if (! acceptTokenClass(EHTokRightBracket)) {
                        expected("]");
                        break;
                    }
                }
                // (COMMA SPACEN)opt
                HlslToken spaceDesc;
                if (acceptTokenClass(EHTokComma)) {
                    if (! acceptIdentifier(spaceDesc)) {
                        expected ("space identifier");
                        return false;
                    }
                }
                // RIGHT_PAREN
                if (! acceptTokenClass(EHTokRightParen)) {
                    expected(")");
                    break;
                }
                parseContext.handleRegister(registerDesc.loc, qualifier, profile.string, *registerDesc.string, subComponent, spaceDesc.string);
            } else {
                // semantic, in idToken.string
                TString semanticUpperCase = *idToken.string;
                std::transform(semanticUpperCase.begin(), semanticUpperCase.end(), semanticUpperCase.begin(), ::toupper);
                parseContext.handleSemantic(idToken.loc, qualifier, mapSemantic(semanticUpperCase.c_str()), semanticUpperCase);
            }
        } else if (peekTokenClass(EHTokLeftAngle)) {
            found = true;
            acceptAnnotations(qualifier);
        } else
            break;

    } while (true);

    return found;
}

//
// Get the stream of tokens from the scanner, but skip all syntactic/semantic
// processing.
//
bool HlslGrammar::captureBlockTokens(TVector<HlslToken>& tokens)
{
    if (! peekTokenClass(EHTokLeftBrace))
        return false;

    int braceCount = 0;

    do {
        switch (peek()) {
        case EHTokLeftBrace:
            ++braceCount;
            break;
        case EHTokRightBrace:
            --braceCount;
            break;
        case EHTokNone:
            // End of input before balance { } is bad...
            return false;
        default:
            break;
        }

        tokens.push_back(token);
        advanceToken();
    } while (braceCount > 0);

    return true;
}

// Return a string for just the types that can also be declared as an identifier.
const char* HlslGrammar::getTypeString(EHlslTokenClass tokenClass) const
{
    switch (tokenClass) {
    case EHTokSample:     return "sample";
    case EHTokHalf:       return "half";
    case EHTokHalf1x1:    return "half1x1";
    case EHTokHalf1x2:    return "half1x2";
    case EHTokHalf1x3:    return "half1x3";
    case EHTokHalf1x4:    return "half1x4";
    case EHTokHalf2x1:    return "half2x1";
    case EHTokHalf2x2:    return "half2x2";
    case EHTokHalf2x3:    return "half2x3";
    case EHTokHalf2x4:    return "half2x4";
    case EHTokHalf3x1:    return "half3x1";
    case EHTokHalf3x2:    return "half3x2";
    case EHTokHalf3x3:    return "half3x3";
    case EHTokHalf3x4:    return "half3x4";
    case EHTokHalf4x1:    return "half4x1";
    case EHTokHalf4x2:    return "half4x2";
    case EHTokHalf4x3:    return "half4x3";
    case EHTokHalf4x4:    return "half4x4";
    case EHTokBool:       return "bool";
    case EHTokFloat:      return "float";
    case EHTokDouble:     return "double";
    case EHTokInt:        return "int";
    case EHTokUint:       return "uint";
    case EHTokMin16float: return "min16float";
    case EHTokMin10float: return "min10float";
    case EHTokMin16int:   return "min16int";
    case EHTokMin12int:   return "min12int";
    case EHTokConstantBuffer: return "ConstantBuffer";
    case EHTokLayout:     return "layout";
    default:
        return nullptr;
    }
}

} // end namespace glslang
