//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2012-2013 LunarG, Inc.
// Copyright (C) 2017 ARM Limited.
// Copyright (C) 2015-2018 Google, Inc.
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
// Symbol table for parsing.  Most functionality and main ideas
// are documented in the header file.
//

#include "SymbolTable.h"

namespace glslang {

//
// TType helper function needs a place to live.
//

//
// Recursively generate mangled names.
//
void TType::buildMangledName(TString& mangledName) const
{
    if (isMatrix())
        mangledName += 'm';
    else if (isVector())
        mangledName += 'v';

    switch (basicType) {
    case EbtFloat:              mangledName += 'f';      break;
    case EbtInt:                mangledName += 'i';      break;
    case EbtUint:               mangledName += 'u';      break;
    case EbtBool:               mangledName += 'b';      break;
#ifndef GLSLANG_WEB
    case EbtDouble:             mangledName += 'd';      break;
    case EbtFloat16:            mangledName += "f16";    break;
    case EbtInt8:               mangledName += "i8";     break;
    case EbtUint8:              mangledName += "u8";     break;
    case EbtInt16:              mangledName += "i16";    break;
    case EbtUint16:             mangledName += "u16";    break;
    case EbtInt64:              mangledName += "i64";    break;
    case EbtUint64:             mangledName += "u64";    break;
    case EbtAtomicUint:         mangledName += "au";     break;
    case EbtAccStruct:          mangledName += "as";     break;
    case EbtRayQuery:           mangledName += "rq";     break;
    case EbtSpirvType:          mangledName += "spv-t";  break;
#endif
    case EbtSampler:
        switch (sampler.type) {
#ifndef GLSLANG_WEB
        case EbtFloat16: mangledName += "f16"; break;
#endif
        case EbtInt:   mangledName += "i"; break;
        case EbtUint:  mangledName += "u"; break;
        case EbtInt64:   mangledName += "i64"; break;
        case EbtUint64:  mangledName += "u64"; break;
        default: break; // some compilers want this
        }
        if (sampler.isImageClass())
            mangledName += "I";  // a normal image or subpass
        else if (sampler.isPureSampler())
            mangledName += "p";  // a "pure" sampler
        else if (!sampler.isCombined())
            mangledName += "t";  // a "pure" texture
        else
            mangledName += "s";  // traditional combined sampler
        if (sampler.isArrayed())
            mangledName += "A";
        if (sampler.isShadow())
            mangledName += "S";
        if (sampler.isExternal())
            mangledName += "E";
        if (sampler.isYuv())
            mangledName += "Y";
        switch (sampler.dim) {
        case Esd2D:       mangledName += "2";  break;
        case Esd3D:       mangledName += "3";  break;
        case EsdCube:     mangledName += "C";  break;
#ifndef GLSLANG_WEB
        case Esd1D:       mangledName += "1";  break;
        case EsdRect:     mangledName += "R2"; break;
        case EsdBuffer:   mangledName += "B";  break;
        case EsdSubpass:  mangledName += "P";  break;
#endif
        default: break; // some compilers want this
        }

#ifdef ENABLE_HLSL
        if (sampler.hasReturnStruct()) {
            // Name mangle for sampler return struct uses struct table index.
            mangledName += "-tx-struct";

            char text[16]; // plenty enough space for the small integers.
            snprintf(text, sizeof(text), "%u-", sampler.getStructReturnIndex());
            mangledName += text;
        } else {
            switch (sampler.getVectorSize()) {
            case 1: mangledName += "1"; break;
            case 2: mangledName += "2"; break;
            case 3: mangledName += "3"; break;
            case 4: break; // default to prior name mangle behavior
            }
        }
#endif

        if (sampler.isMultiSample())
            mangledName += "M";
        break;
    case EbtStruct:
    case EbtBlock:
        if (basicType == EbtStruct)
            mangledName += "struct-";
        else
            mangledName += "block-";
        if (typeName)
            mangledName += *typeName;
        for (unsigned int i = 0; i < structure->size(); ++i) {
            if ((*structure)[i].type->getBasicType() == EbtVoid)
                continue;
            mangledName += '-';
            (*structure)[i].type->buildMangledName(mangledName);
        }
    default:
        break;
    }

    if (getVectorSize() > 0)
        mangledName += static_cast<char>('0' + getVectorSize());
    else {
        mangledName += static_cast<char>('0' + getMatrixCols());
        mangledName += static_cast<char>('0' + getMatrixRows());
    }

    if (arraySizes) {
        const int maxSize = 11;
        char buf[maxSize];
        for (int i = 0; i < arraySizes->getNumDims(); ++i) {
            if (arraySizes->getDimNode(i)) {
                if (arraySizes->getDimNode(i)->getAsSymbolNode())
                    snprintf(buf, maxSize, "s%lld", arraySizes->getDimNode(i)->getAsSymbolNode()->getId());
                else
                    snprintf(buf, maxSize, "s%p", arraySizes->getDimNode(i));
            } else
                snprintf(buf, maxSize, "%d", arraySizes->getDimSize(i));
            mangledName += '[';
            mangledName += buf;
            mangledName += ']';
        }
    }
}

#if !defined(GLSLANG_WEB) && !defined(GLSLANG_ANGLE)

//
// Dump functions.
//

void TSymbol::dumpExtensions(TInfoSink& infoSink) const
{
    int numExtensions = getNumExtensions();
    if (numExtensions) {
        infoSink.debug << " <";

        for (int i = 0; i < numExtensions; i++)
            infoSink.debug << getExtensions()[i] << ",";

        infoSink.debug << ">";
    }
}

void TVariable::dump(TInfoSink& infoSink, bool complete) const
{
    if (complete) {
        infoSink.debug << getName().c_str() << ": " << type.getCompleteString();
        dumpExtensions(infoSink);
    } else {
        infoSink.debug << getName().c_str() << ": " << type.getStorageQualifierString() << " "
                       << type.getBasicTypeString();

        if (type.isArray())
            infoSink.debug << "[0]";
    }

    infoSink.debug << "\n";
}

void TFunction::dump(TInfoSink& infoSink, bool complete) const
{
    if (complete) {
        infoSink.debug << getName().c_str() << ": " << returnType.getCompleteString() << " " << getName().c_str()
                       << "(";

        int numParams = getParamCount();
        for (int i = 0; i < numParams; i++) {
            const TParameter &param = parameters[i];
            infoSink.debug << param.type->getCompleteString() << " "
                           << (param.type->isStruct() ? "of " + param.type->getTypeName() + " " : "")
                           << (param.name ? *param.name : "") << (i < numParams - 1 ? "," : "");
        }

        infoSink.debug << ")";
        dumpExtensions(infoSink);
    } else {
        infoSink.debug << getName().c_str() << ": " << returnType.getBasicTypeString() << " "
                       << getMangledName().c_str() << "n";
    }

    infoSink.debug << "\n";
}

void TAnonMember::dump(TInfoSink& TInfoSink, bool) const
{
    TInfoSink.debug << "anonymous member " << getMemberNumber() << " of " << getAnonContainer().getName().c_str()
                    << "\n";
}

void TSymbolTableLevel::dump(TInfoSink& infoSink, bool complete) const
{
    tLevel::const_iterator it;
    for (it = level.begin(); it != level.end(); ++it)
        (*it).second->dump(infoSink, complete);
}

void TSymbolTable::dump(TInfoSink& infoSink, bool complete) const
{
    for (int level = currentLevel(); level >= 0; --level) {
        infoSink.debug << "LEVEL " << level << "\n";
        table[level]->dump(infoSink, complete);
    }
}

#endif

//
// Functions have buried pointers to delete.
//
TFunction::~TFunction()
{
    for (TParamList::iterator i = parameters.begin(); i != parameters.end(); ++i)
        delete (*i).type;
}

//
// Symbol table levels are a map of pointers to symbols that have to be deleted.
//
TSymbolTableLevel::~TSymbolTableLevel()
{
    for (tLevel::iterator it = level.begin(); it != level.end(); ++it) {
        const TString& name = it->first;
        auto retargetIter = std::find_if(retargetedSymbols.begin(), retargetedSymbols.end(),
                                      [&name](const std::pair<TString, TString>& i) { return i.first == name; });
        if (retargetIter == retargetedSymbols.end())
            delete (*it).second;
    }


    delete [] defaultPrecision;
}

//
// Change all function entries in the table with the non-mangled name
// to be related to the provided built-in operation.
//
void TSymbolTableLevel::relateToOperator(const char* name, TOperator op)
{
    tLevel::const_iterator candidate = level.lower_bound(name);
    while (candidate != level.end()) {
        const TString& candidateName = (*candidate).first;
        TString::size_type parenAt = candidateName.find_first_of('(');
        if (parenAt != candidateName.npos && candidateName.compare(0, parenAt, name) == 0) {
            TFunction* function = (*candidate).second->getAsFunction();
            function->relateToOperator(op);
        } else
            break;
        ++candidate;
    }
}

// Make all function overloads of the given name require an extension(s).
// Should only be used for a version/profile that actually needs the extension(s).
void TSymbolTableLevel::setFunctionExtensions(const char* name, int num, const char* const extensions[])
{
    tLevel::const_iterator candidate = level.lower_bound(name);
    while (candidate != level.end()) {
        const TString& candidateName = (*candidate).first;
        TString::size_type parenAt = candidateName.find_first_of('(');
        if (parenAt != candidateName.npos && candidateName.compare(0, parenAt, name) == 0) {
            TSymbol* symbol = candidate->second;
            symbol->setExtensions(num, extensions);
        } else
            break;
        ++candidate;
    }
}

//
// Make all symbols in this table level read only.
//
void TSymbolTableLevel::readOnly()
{
    for (tLevel::iterator it = level.begin(); it != level.end(); ++it)
        (*it).second->makeReadOnly();
}

//
// Copy a symbol, but the copy is writable; call readOnly() afterward if that's not desired.
//
TSymbol::TSymbol(const TSymbol& copyOf)
{
    name = NewPoolTString(copyOf.name->c_str());
    uniqueId = copyOf.uniqueId;
    writable = true;
}

TVariable::TVariable(const TVariable& copyOf) : TSymbol(copyOf)
{
    type.deepCopy(copyOf.type);
    userType = copyOf.userType;

    // we don't support specialization-constant subtrees in cloned tables, only extensions
    constSubtree = nullptr;
    extensions = nullptr;
    memberExtensions = nullptr;
    if (copyOf.getNumExtensions() > 0)
        setExtensions(copyOf.getNumExtensions(), copyOf.getExtensions());
    if (copyOf.hasMemberExtensions()) {
        for (int m = 0; m < (int)copyOf.type.getStruct()->size(); ++m) {
            if (copyOf.getNumMemberExtensions(m) > 0)
                setMemberExtensions(m, copyOf.getNumMemberExtensions(m), copyOf.getMemberExtensions(m));
        }
    }

    if (! copyOf.constArray.empty()) {
        assert(! copyOf.type.isStruct());
        TConstUnionArray newArray(copyOf.constArray, 0, copyOf.constArray.size());
        constArray = newArray;
    }
}

TVariable* TVariable::clone() const
{
    TVariable *variable = new TVariable(*this);

    return variable;
}

TFunction::TFunction(const TFunction& copyOf) : TSymbol(copyOf)
{
    for (unsigned int i = 0; i < copyOf.parameters.size(); ++i) {
        TParameter param;
        parameters.push_back(param);
        (void)parameters.back().copyParam(copyOf.parameters[i]);
    }

    extensions = nullptr;
    if (copyOf.getNumExtensions() > 0)
        setExtensions(copyOf.getNumExtensions(), copyOf.getExtensions());
    returnType.deepCopy(copyOf.returnType);
    mangledName = copyOf.mangledName;
    op = copyOf.op;
    defined = copyOf.defined;
    prototyped = copyOf.prototyped;
    implicitThis = copyOf.implicitThis;
    illegalImplicitThis = copyOf.illegalImplicitThis;
    defaultParamCount = copyOf.defaultParamCount;
#ifndef GLSLANG_WEB
    spirvInst = copyOf.spirvInst;
#endif
}

TFunction* TFunction::clone() const
{
    TFunction *function = new TFunction(*this);

    return function;
}

TAnonMember* TAnonMember::clone() const
{
    // Anonymous members of a given block should be cloned at a higher level,
    // where they can all be assured to still end up pointing to a single
    // copy of the original container.
    assert(0);

    return 0;
}

TSymbolTableLevel* TSymbolTableLevel::clone() const
{
    TSymbolTableLevel *symTableLevel = new TSymbolTableLevel();
    symTableLevel->anonId = anonId;
    symTableLevel->thisLevel = thisLevel;
    symTableLevel->retargetedSymbols.clear();
    for (auto &s : retargetedSymbols) {
        symTableLevel->retargetedSymbols.push_back({s.first, s.second});
    }
    std::vector<bool> containerCopied(anonId, false);
    tLevel::const_iterator iter;
    for (iter = level.begin(); iter != level.end(); ++iter) {
        const TAnonMember* anon = iter->second->getAsAnonMember();
        if (anon) {
            // Insert all the anonymous members of this same container at once,
            // avoid inserting the remaining members in the future, once this has been done,
            // allowing them to all be part of the same new container.
            if (! containerCopied[anon->getAnonId()]) {
                TVariable* container = anon->getAnonContainer().clone();
                container->changeName(NewPoolTString(""));
                // insert the container and all its members
                symTableLevel->insert(*container, false);
                containerCopied[anon->getAnonId()] = true;
            }
        } else {
            const TString& name = iter->first;
            auto retargetIter = std::find_if(retargetedSymbols.begin(), retargetedSymbols.end(),
                                          [&name](const std::pair<TString, TString>& i) { return i.first == name; });
            if (retargetIter != retargetedSymbols.end())
                continue;
            symTableLevel->insert(*iter->second->clone(), false);
        }
    }
    // Now point retargeted symbols to the newly created versions of them
    for (auto &s : retargetedSymbols) {
        TSymbol* sym = symTableLevel->find(s.second);
        if (!sym)
            continue;
        symTableLevel->insert(s.first, sym);
    }

    return symTableLevel;
}

void TSymbolTable::copyTable(const TSymbolTable& copyOf)
{
    assert(adoptedLevels == copyOf.adoptedLevels);

    uniqueId = copyOf.uniqueId;
    noBuiltInRedeclarations = copyOf.noBuiltInRedeclarations;
    separateNameSpaces = copyOf.separateNameSpaces;
    for (unsigned int i = copyOf.adoptedLevels; i < copyOf.table.size(); ++i)
        table.push_back(copyOf.table[i]->clone());
}

} // end namespace glslang
