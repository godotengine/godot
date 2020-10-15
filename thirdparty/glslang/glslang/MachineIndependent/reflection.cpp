//
// Copyright (C) 2013-2016 LunarG, Inc.
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

#if !defined(GLSLANG_WEB) && !defined(GLSLANG_ANGLE)

#include "../Include/Common.h"
#include "reflection.h"
#include "LiveTraverser.h"
#include "localintermediate.h"

#include "gl_types.h"

//
// Grow the reflection database through a friend traverser class of TReflection and a
// collection of functions to do a liveness traversal that note what uniforms are used
// in semantically non-dead code.
//
// Can be used multiple times, once per stage, to grow a program reflection.
//
// High-level algorithm for one stage:
//
// 1. Put the entry point on the list of live functions.
//
// 2. Traverse any live function, while skipping if-tests with a compile-time constant
//    condition of false, and while adding any encountered function calls to the live
//    function list.
//
//    Repeat until the live function list is empty.
//
// 3. Add any encountered uniform variables and blocks to the reflection database.
//
// Can be attempted with a failed link, but will return false if recursion had been detected, or
// there wasn't exactly one entry point.
//

namespace glslang {

//
// The traverser: mostly pass through, except
//  - processing binary nodes to see if they are dereferences of an aggregates to track
//  - processing symbol nodes to see if they are non-aggregate objects to track
//
// This ignores semantically dead code by using TLiveTraverser.
//
// This is in the glslang namespace directly so it can be a friend of TReflection.
//

class TReflectionTraverser : public TIntermTraverser {
public:
    TReflectionTraverser(const TIntermediate& i, TReflection& r) :
	                     TIntermTraverser(), intermediate(i), reflection(r), updateStageMasks(true) { }

    virtual bool visitBinary(TVisit, TIntermBinary* node);
    virtual void visitSymbol(TIntermSymbol* base);

    // Add a simple reference to a uniform variable to the uniform database, no dereference involved.
    // However, no dereference doesn't mean simple... it could be a complex aggregate.
    void addUniform(const TIntermSymbol& base)
    {
        if (processedDerefs.find(&base) == processedDerefs.end()) {
            processedDerefs.insert(&base);

            int blockIndex = -1;
            int offset     = -1;
            TList<TIntermBinary*> derefs;
            TString baseName = base.getName();

            if (base.getType().getBasicType() == EbtBlock) {
                offset = 0;
                bool anonymous = IsAnonymous(baseName);
                const TString& blockName = base.getType().getTypeName();

                if (!anonymous)
                    baseName = blockName;
                else
                    baseName = "";

                blockIndex = addBlockName(blockName, base.getType(), intermediate.getBlockSize(base.getType()));
            }

            // Use a degenerate (empty) set of dereferences to immediately put as at the end of
            // the dereference change expected by blowUpActiveAggregate.
            blowUpActiveAggregate(base.getType(), baseName, derefs, derefs.end(), offset, blockIndex, 0, -1, 0,
                                    base.getQualifier().storage, updateStageMasks);
        }
    }

    void addPipeIOVariable(const TIntermSymbol& base)
    {
        if (processedDerefs.find(&base) == processedDerefs.end()) {
            processedDerefs.insert(&base);

            const TString &name = base.getName();
            const TType &type = base.getType();
            const bool input = base.getQualifier().isPipeInput();

            TReflection::TMapIndexToReflection &ioItems =
                input ? reflection.indexToPipeInput : reflection.indexToPipeOutput;


            TReflection::TNameToIndex &ioMapper =
                input ? reflection.pipeInNameToIndex : reflection.pipeOutNameToIndex;

            if (reflection.options & EShReflectionUnwrapIOBlocks) {
                bool anonymous = IsAnonymous(name);

                TString baseName;
                if (type.getBasicType() == EbtBlock) {
                    baseName = anonymous ? TString() : type.getTypeName();
                } else {
                    baseName = anonymous ? TString() : name;
                }

                // by convention if this is an arrayed block we ignore the array in the reflection
                if (type.isArray() && type.getBasicType() == EbtBlock) {
                    blowUpIOAggregate(input, baseName, TType(type, 0));
                } else {               
                    blowUpIOAggregate(input, baseName, type);
                }
            } else {
                TReflection::TNameToIndex::const_iterator it = ioMapper.find(name.c_str());
                if (it == ioMapper.end()) {
                    // seperate pipe i/o params from uniforms and blocks
                    // in is only for input in first stage as out is only for last stage. check traverse in call stack.
                    ioMapper[name.c_str()] = static_cast<int>(ioItems.size());
                    ioItems.push_back(
                        TObjectReflection(name.c_str(), type, 0, mapToGlType(type), mapToGlArraySize(type), 0));
                    EShLanguageMask& stages = ioItems.back().stages;
                    stages = static_cast<EShLanguageMask>(stages | 1 << intermediate.getStage());
                } else {
                    EShLanguageMask& stages = ioItems[it->second].stages;
                    stages = static_cast<EShLanguageMask>(stages | 1 << intermediate.getStage());
                }
            }
        }
    }

    // Lookup or calculate the offset of all block members at once, using the recursively
    // defined block offset rules.
    void getOffsets(const TType& type, TVector<int>& offsets)
    {
        const TTypeList& memberList = *type.getStruct();
        int memberSize = 0;
        int offset = 0;

        for (size_t m = 0; m < offsets.size(); ++m) {
            // if the user supplied an offset, snap to it now
            if (memberList[m].type->getQualifier().hasOffset())
                offset = memberList[m].type->getQualifier().layoutOffset;

            // calculate the offset of the next member and align the current offset to this member
            intermediate.updateOffset(type, *memberList[m].type, offset, memberSize);

            // save the offset of this member
            offsets[m] = offset;

            // update for the next member
            offset += memberSize;
        }
    }

    // Calculate the stride of an array type
    int getArrayStride(const TType& baseType, const TType& type)
    {
        int dummySize;
        int stride;

        // consider blocks to have 0 stride, so that all offsets are relative to the start of their block
        if (type.getBasicType() == EbtBlock)
            return 0;

        TLayoutMatrix subMatrixLayout = type.getQualifier().layoutMatrix;
        intermediate.getMemberAlignment(type, dummySize, stride,
                                        baseType.getQualifier().layoutPacking,
                                        subMatrixLayout != ElmNone
                                            ? subMatrixLayout == ElmRowMajor
                                            : baseType.getQualifier().layoutMatrix == ElmRowMajor);

        return stride;
    }

    // count the total number of leaf members from iterating out of a block type
    int countAggregateMembers(const TType& parentType)
    {
        if (! parentType.isStruct())
            return 1;

        const bool strictArraySuffix = (reflection.options & EShReflectionStrictArraySuffix);

        bool blockParent = (parentType.getBasicType() == EbtBlock && parentType.getQualifier().storage == EvqBuffer);

        const TTypeList &memberList = *parentType.getStruct();

        int ret = 0;

        for (size_t i = 0; i < memberList.size(); i++)
        {
            const TType &memberType = *memberList[i].type;
            int numMembers = countAggregateMembers(memberType);
            // for sized arrays of structs, apply logic to expand out the same as we would below in
            // blowUpActiveAggregate
            if (memberType.isArray() && ! memberType.getArraySizes()->hasUnsized() && memberType.isStruct()) {
                if (! strictArraySuffix || ! blockParent)
                    numMembers *= memberType.getArraySizes()->getCumulativeSize();
            }
            ret += numMembers;
        }

        return ret;
    }

    // Traverse the provided deref chain, including the base, and
    // - build a full reflection-granularity name, array size, etc. entry out of it, if it goes down to that granularity
    // - recursively expand any variable array index in the middle of that traversal
    // - recursively expand what's left at the end if the deref chain did not reach down to reflection granularity
    //
    // arraySize tracks, just for the final dereference in the chain, if there was a specific known size.
    // A value of 0 for arraySize will mean to use the full array's size.
    void blowUpActiveAggregate(const TType& baseType, const TString& baseName, const TList<TIntermBinary*>& derefs,
                               TList<TIntermBinary*>::const_iterator deref, int offset, int blockIndex, int arraySize,
                               int topLevelArraySize, int topLevelArrayStride, TStorageQualifier baseStorage, bool active)
    {
        // when strictArraySuffix is enabled, we closely follow the rules from ARB_program_interface_query.
        // Broadly:
        // * arrays-of-structs always have a [x] suffix.
        // * with array-of-struct variables in the root of a buffer block, only ever return [0].
        // * otherwise, array suffixes are added whenever we iterate, even if that means expanding out an array.
        const bool strictArraySuffix = (reflection.options & EShReflectionStrictArraySuffix);

        // is this variable inside a buffer block. This flag is set back to false after we iterate inside the first array element.
        bool blockParent = (baseType.getBasicType() == EbtBlock && baseType.getQualifier().storage == EvqBuffer);

        // process the part of the dereference chain that was explicit in the shader
        TString name = baseName;
        const TType* terminalType = &baseType;
        for (; deref != derefs.end(); ++deref) {
            TIntermBinary* visitNode = *deref;
            terminalType = &visitNode->getType();
            int index;
            switch (visitNode->getOp()) {
            case EOpIndexIndirect: {
                int stride = getArrayStride(baseType, visitNode->getLeft()->getType());

                if (topLevelArrayStride == 0)
                    topLevelArrayStride = stride;

                // Visit all the indices of this array, and for each one add on the remaining dereferencing
                for (int i = 0; i < std::max(visitNode->getLeft()->getType().getOuterArraySize(), 1); ++i) {
                    TString newBaseName = name;
                    if (terminalType->getBasicType() == EbtBlock) {}
                    else if (strictArraySuffix && blockParent)
                        newBaseName.append(TString("[0]"));
                    else if (strictArraySuffix || baseType.getBasicType() != EbtBlock)
                        newBaseName.append(TString("[") + String(i) + "]");
                    TList<TIntermBinary*>::const_iterator nextDeref = deref;
                    ++nextDeref;
                    blowUpActiveAggregate(*terminalType, newBaseName, derefs, nextDeref, offset, blockIndex, arraySize,
                                          topLevelArraySize, topLevelArrayStride, baseStorage, active);

                    if (offset >= 0)
                        offset += stride;
                }

                // it was all completed in the recursive calls above
                return;
            }
            case EOpIndexDirect: {
                int stride = getArrayStride(baseType, visitNode->getLeft()->getType());

                index = visitNode->getRight()->getAsConstantUnion()->getConstArray()[0].getIConst();
                if (terminalType->getBasicType() == EbtBlock) {}
                else if (strictArraySuffix && blockParent)
                    name.append(TString("[0]"));
                else if (strictArraySuffix || baseType.getBasicType() != EbtBlock) {
                    name.append(TString("[") + String(index) + "]");

                    if (offset >= 0)
                        offset += stride * index;
                }

                if (topLevelArrayStride == 0)
                    topLevelArrayStride = stride;

                // expand top-level arrays in blocks with [0] suffix
                if (topLevelArrayStride != 0 && visitNode->getLeft()->getType().isArray()) {
                    blockParent = false;
                }
                break;
            }
            case EOpIndexDirectStruct:
                index = visitNode->getRight()->getAsConstantUnion()->getConstArray()[0].getIConst();
                if (offset >= 0)
                    offset += intermediate.getOffset(visitNode->getLeft()->getType(), index);
                if (name.size() > 0)
                    name.append(".");
                name.append((*visitNode->getLeft()->getType().getStruct())[index].type->getFieldName());

                // expand non top-level arrays with [x] suffix
                if (visitNode->getLeft()->getType().getBasicType() != EbtBlock && terminalType->isArray())
                {
                    blockParent = false;
                }
                break;
            default:
                break;
            }
        }

        // if the terminalType is still too coarse a granularity, this is still an aggregate to expand, expand it...
        if (! isReflectionGranularity(*terminalType)) {
            // the base offset of this node, that children are relative to
            int baseOffset = offset;

            if (terminalType->isArray()) {
                // Visit all the indices of this array, and for each one,
                // fully explode the remaining aggregate to dereference

                int stride = 0;
                if (offset >= 0)
                    stride = getArrayStride(baseType, *terminalType);

                int arrayIterateSize = std::max(terminalType->getOuterArraySize(), 1);

                // for top-level arrays in blocks, only expand [0] to avoid explosion of items
                if ((strictArraySuffix && blockParent) ||
                    ((topLevelArraySize == arrayIterateSize) && (topLevelArrayStride == 0))) {
                    arrayIterateSize = 1;
                }

                if (topLevelArrayStride == 0)
                    topLevelArrayStride = stride;

                for (int i = 0; i < arrayIterateSize; ++i) {
                    TString newBaseName = name;
                    if (terminalType->getBasicType() != EbtBlock)
                        newBaseName.append(TString("[") + String(i) + "]");
                    TType derefType(*terminalType, 0);
                    if (offset >= 0)
                        offset = baseOffset + stride * i;

                    blowUpActiveAggregate(derefType, newBaseName, derefs, derefs.end(), offset, blockIndex, 0,
                                          topLevelArraySize, topLevelArrayStride, baseStorage, active);
                }
            } else {
                // Visit all members of this aggregate, and for each one,
                // fully explode the remaining aggregate to dereference
                const TTypeList& typeList = *terminalType->getStruct();

                TVector<int> memberOffsets;

                if (baseOffset >= 0) {
                    memberOffsets.resize(typeList.size());
                    getOffsets(*terminalType, memberOffsets);
                }

                for (int i = 0; i < (int)typeList.size(); ++i) {
                    TString newBaseName = name;
                    if (newBaseName.size() > 0)
                        newBaseName.append(".");
                    newBaseName.append(typeList[i].type->getFieldName());
                    TType derefType(*terminalType, i);
                    if (offset >= 0)
                        offset = baseOffset + memberOffsets[i];

                    int arrayStride = topLevelArrayStride;
                    if (terminalType->getBasicType() == EbtBlock && terminalType->getQualifier().storage == EvqBuffer &&
                        derefType.isArray()) {
                        arrayStride = getArrayStride(baseType, derefType);
                    }

                    if (topLevelArraySize == -1 && arrayStride == 0 && blockParent)
                        topLevelArraySize = 1;

                    if (strictArraySuffix && blockParent) {
                        // if this member is an array, store the top-level array stride but start the explosion from
                        // the inner struct type.
                        if (derefType.isArray() && derefType.isStruct()) {
                            newBaseName.append("[0]");
                            auto dimSize = derefType.isUnsizedArray() ? 0 : derefType.getArraySizes()->getDimSize(0);
                            blowUpActiveAggregate(TType(derefType, 0), newBaseName, derefs, derefs.end(), memberOffsets[i],
                                blockIndex, 0, dimSize, arrayStride, terminalType->getQualifier().storage, false);
                        }
                        else if (derefType.isArray()) {
                            auto dimSize = derefType.isUnsizedArray() ? 0 : derefType.getArraySizes()->getDimSize(0);
                            blowUpActiveAggregate(derefType, newBaseName, derefs, derefs.end(), memberOffsets[i], blockIndex,
                                0, dimSize, 0, terminalType->getQualifier().storage, false);
                        }
                        else {
                            blowUpActiveAggregate(derefType, newBaseName, derefs, derefs.end(), memberOffsets[i], blockIndex,
                                0, 1, 0, terminalType->getQualifier().storage, false);
                        }
                    } else {
                        blowUpActiveAggregate(derefType, newBaseName, derefs, derefs.end(), offset, blockIndex, 0,
                                              topLevelArraySize, arrayStride, baseStorage, active);
                    }
                }
            }

            // it was all completed in the recursive calls above
            return;
        }

        if ((reflection.options & EShReflectionBasicArraySuffix) && terminalType->isArray()) {
            name.append(TString("[0]"));
        }

        // Finally, add a full string to the reflection database, and update the array size if necessary.
        // If the dereferenced entity to record is an array, compute the size and update the maximum size.

        // there might not be a final array dereference, it could have been copied as an array object
        if (arraySize == 0)
            arraySize = mapToGlArraySize(*terminalType);

        TReflection::TMapIndexToReflection& variables = reflection.GetVariableMapForStorage(baseStorage);

        TReflection::TNameToIndex::const_iterator it = reflection.nameToIndex.find(name.c_str());
        if (it == reflection.nameToIndex.end()) {
            int uniformIndex = (int)variables.size();
            reflection.nameToIndex[name.c_str()] = uniformIndex;
            variables.push_back(TObjectReflection(name.c_str(), *terminalType, offset, mapToGlType(*terminalType),
                                                  arraySize, blockIndex));
            if (terminalType->isArray()) {
                variables.back().arrayStride = getArrayStride(baseType, *terminalType);
                if (topLevelArrayStride == 0)
                    topLevelArrayStride = variables.back().arrayStride;
            }

            if ((reflection.options & EShReflectionSeparateBuffers) && terminalType->isAtomic())
                reflection.atomicCounterUniformIndices.push_back(uniformIndex);

            variables.back().topLevelArraySize = topLevelArraySize;
            variables.back().topLevelArrayStride = topLevelArrayStride;
            
            if ((reflection.options & EShReflectionAllBlockVariables) && active) {
                EShLanguageMask& stages = variables.back().stages;
                stages = static_cast<EShLanguageMask>(stages | 1 << intermediate.getStage());
            }
        } else {
            if (arraySize > 1) {
                int& reflectedArraySize = variables[it->second].size;
                reflectedArraySize = std::max(arraySize, reflectedArraySize);
            }

            if ((reflection.options & EShReflectionAllBlockVariables) && active) {
              EShLanguageMask& stages = variables[it->second].stages;
              stages = static_cast<EShLanguageMask>(stages | 1 << intermediate.getStage());
            }
        }
    }
    
    // similar to blowUpActiveAggregate, but with simpler rules and no dereferences to follow.
    void blowUpIOAggregate(bool input, const TString &baseName, const TType &type)
    {
        TString name = baseName;

        // if the type is still too coarse a granularity, this is still an aggregate to expand, expand it...
        if (! isReflectionGranularity(type)) {
            if (type.isArray()) {
                // Visit all the indices of this array, and for each one,
                // fully explode the remaining aggregate to dereference
                for (int i = 0; i < std::max(type.getOuterArraySize(), 1); ++i) {
                    TString newBaseName = name;
                    newBaseName.append(TString("[") + String(i) + "]");
                    TType derefType(type, 0);

                    blowUpIOAggregate(input, newBaseName, derefType);
                }
            } else {
                // Visit all members of this aggregate, and for each one,
                // fully explode the remaining aggregate to dereference
                const TTypeList& typeList = *type.getStruct();

                for (int i = 0; i < (int)typeList.size(); ++i) {
                    TString newBaseName = name;
                    if (newBaseName.size() > 0)
                        newBaseName.append(".");
                    newBaseName.append(typeList[i].type->getFieldName());
                    TType derefType(type, i);

                    blowUpIOAggregate(input, newBaseName, derefType);
                }
            }

            // it was all completed in the recursive calls above
            return;
        }

        if ((reflection.options & EShReflectionBasicArraySuffix) && type.isArray()) {
            name.append(TString("[0]"));
        }

        TReflection::TMapIndexToReflection &ioItems =
            input ? reflection.indexToPipeInput : reflection.indexToPipeOutput;

        std::string namespacedName = input ? "in " : "out ";
        namespacedName += name.c_str();

        TReflection::TNameToIndex::const_iterator it = reflection.nameToIndex.find(namespacedName);
        if (it == reflection.nameToIndex.end()) {
            reflection.nameToIndex[namespacedName] = (int)ioItems.size();
            ioItems.push_back(
                TObjectReflection(name.c_str(), type, 0, mapToGlType(type), mapToGlArraySize(type), 0));

            EShLanguageMask& stages = ioItems.back().stages;
            stages = static_cast<EShLanguageMask>(stages | 1 << intermediate.getStage());
        } else {
            EShLanguageMask& stages = ioItems[it->second].stages;
            stages = static_cast<EShLanguageMask>(stages | 1 << intermediate.getStage());
        }
    }

    // Add a uniform dereference where blocks/struct/arrays are involved in the access.
    // Handles the situation where the left node is at the correct or too coarse a
    // granularity for reflection.  (That is, further dereferences up the tree will be
    // skipped.) Earlier dereferences, down the tree, will be handled
    // at the same time, and logged to prevent reprocessing as the tree is traversed.
    //
    // Note: Other things like the following must be caught elsewhere:
    //  - a simple non-array, non-struct variable (no dereference even conceivable)
    //  - an aggregrate consumed en masse, without a dereference
    //
    // So, this code is for cases like
    //   - a struct/block dereferencing a member (whether the member is array or not)
    //   - an array of struct
    //   - structs/arrays containing the above
    //
    void addDereferencedUniform(TIntermBinary* topNode)
    {
        // See if too fine-grained to process (wait to get further down the tree)
        const TType& leftType = topNode->getLeft()->getType();
        if ((leftType.isVector() || leftType.isMatrix()) && ! leftType.isArray())
            return;

        // We have an array or structure or block dereference, see if it's a uniform
        // based dereference (if not, skip it).
        TIntermSymbol* base = findBase(topNode);
        if (! base || ! base->getQualifier().isUniformOrBuffer())
            return;

        // See if we've already processed this (e.g., in the middle of something
        // we did earlier), and if so skip it
        if (processedDerefs.find(topNode) != processedDerefs.end())
            return;

        // Process this uniform dereference

        int offset = -1;
        int blockIndex = -1;
        bool anonymous = false;

        // See if we need to record the block itself
        bool block = base->getBasicType() == EbtBlock;
        if (block) {
            offset = 0;
            anonymous = IsAnonymous(base->getName());

            const TString& blockName = base->getType().getTypeName();
            TString baseName;
            
            if (! anonymous)
                baseName = blockName;

            blockIndex = addBlockName(blockName, base->getType(), intermediate.getBlockSize(base->getType()));

            if (reflection.options & EShReflectionAllBlockVariables) {
                // Use a degenerate (empty) set of dereferences to immediately put as at the end of
                // the dereference change expected by blowUpActiveAggregate.
                TList<TIntermBinary*> derefs;

                // otherwise - if we're not using strict array suffix rules, or this isn't a block so we are
                // expanding root arrays anyway, just start the iteration from the base block type.
                blowUpActiveAggregate(base->getType(), baseName, derefs, derefs.end(), 0, blockIndex, 0, -1, 0,
                                          base->getQualifier().storage, false);
            }
        }

        // Process the dereference chain, backward, accumulating the pieces for later forward traversal.
        // If the topNode is a reflection-granularity-array dereference, don't include that last dereference.
        TList<TIntermBinary*> derefs;
        for (TIntermBinary* visitNode = topNode; visitNode; visitNode = visitNode->getLeft()->getAsBinaryNode()) {
            if (isReflectionGranularity(visitNode->getLeft()->getType()))
                continue;

            derefs.push_front(visitNode);
            processedDerefs.insert(visitNode);
        }
        processedDerefs.insert(base);

        // See if we have a specific array size to stick to while enumerating the explosion of the aggregate
        int arraySize = 0;
        if (isReflectionGranularity(topNode->getLeft()->getType()) && topNode->getLeft()->isArray()) {
            if (topNode->getOp() == EOpIndexDirect)
                arraySize = topNode->getRight()->getAsConstantUnion()->getConstArray()[0].getIConst() + 1;
        }

        // Put the dereference chain together, forward
        TString baseName;
        if (! anonymous) {
            if (block)
                baseName = base->getType().getTypeName();
            else
                baseName = base->getName();
        }
        blowUpActiveAggregate(base->getType(), baseName, derefs, derefs.begin(), offset, blockIndex, arraySize, -1, 0,
                              base->getQualifier().storage, true);
    }

    int addBlockName(const TString& name, const TType& type, int size)
    {
        int blockIndex = 0;
        if (type.isArray()) {
            TType derefType(type, 0);
            for (int e = 0; e < type.getOuterArraySize(); ++e) {
                int memberBlockIndex = addBlockName(name + "[" + String(e) + "]", derefType, size);
                if (e == 0)
                    blockIndex = memberBlockIndex;
            }
        } else {
            TReflection::TMapIndexToReflection& blocks = reflection.GetBlockMapForStorage(type.getQualifier().storage);

            TReflection::TNameToIndex::const_iterator it = reflection.nameToIndex.find(name.c_str());
            if (reflection.nameToIndex.find(name.c_str()) == reflection.nameToIndex.end()) {
                blockIndex = (int)blocks.size();
                reflection.nameToIndex[name.c_str()] = blockIndex;
                blocks.push_back(TObjectReflection(name.c_str(), type, -1, -1, size, blockIndex));

                blocks.back().numMembers = countAggregateMembers(type);

                EShLanguageMask& stages = blocks.back().stages;
                stages = static_cast<EShLanguageMask>(stages | 1 << intermediate.getStage());
            }
            else {
                blockIndex = it->second;

                EShLanguageMask& stages = blocks[blockIndex].stages;
                stages = static_cast<EShLanguageMask>(stages | 1 << intermediate.getStage());
            }
        }

        return blockIndex;
    }

    // Are we at a level in a dereference chain at which individual active uniform queries are made?
    bool isReflectionGranularity(const TType& type)
    {
        return type.getBasicType() != EbtBlock && type.getBasicType() != EbtStruct && !type.isArrayOfArrays();
    }

    // For a binary operation indexing into an aggregate, chase down the base of the aggregate.
    // Return 0 if the topology does not fit this situation.
    TIntermSymbol* findBase(const TIntermBinary* node)
    {
        TIntermSymbol *base = node->getLeft()->getAsSymbolNode();
        if (base)
            return base;
        TIntermBinary* left = node->getLeft()->getAsBinaryNode();
        if (! left)
            return nullptr;

        return findBase(left);
    }

    //
    // Translate a glslang sampler type into the GL API #define number.
    //
    int mapSamplerToGlType(TSampler sampler)
    {
        if (! sampler.image) {
            // a sampler...
            switch (sampler.type) {
            case EbtFloat:
                switch ((int)sampler.dim) {
                case Esd1D:
                    switch ((int)sampler.shadow) {
                    case false: return sampler.arrayed ? GL_SAMPLER_1D_ARRAY : GL_SAMPLER_1D;
                    case true:  return sampler.arrayed ? GL_SAMPLER_1D_ARRAY_SHADOW : GL_SAMPLER_1D_SHADOW;
                    }
                case Esd2D:
                    switch ((int)sampler.ms) {
                    case false:
                        switch ((int)sampler.shadow) {
                        case false: return sampler.arrayed ? GL_SAMPLER_2D_ARRAY : GL_SAMPLER_2D;
                        case true:  return sampler.arrayed ? GL_SAMPLER_2D_ARRAY_SHADOW : GL_SAMPLER_2D_SHADOW;
                        }
                    case true:      return sampler.arrayed ? GL_SAMPLER_2D_MULTISAMPLE_ARRAY : GL_SAMPLER_2D_MULTISAMPLE;
                    }
                case Esd3D:
                    return GL_SAMPLER_3D;
                case EsdCube:
                    switch ((int)sampler.shadow) {
                    case false: return sampler.arrayed ? GL_SAMPLER_CUBE_MAP_ARRAY : GL_SAMPLER_CUBE;
                    case true:  return sampler.arrayed ? GL_SAMPLER_CUBE_MAP_ARRAY_SHADOW : GL_SAMPLER_CUBE_SHADOW;
                    }
                case EsdRect:
                    return sampler.shadow ? GL_SAMPLER_2D_RECT_SHADOW : GL_SAMPLER_2D_RECT;
                case EsdBuffer:
                    return GL_SAMPLER_BUFFER;
                }
            case EbtFloat16:
                switch ((int)sampler.dim) {
                case Esd1D:
                    switch ((int)sampler.shadow) {
                    case false: return sampler.arrayed ? GL_FLOAT16_SAMPLER_1D_ARRAY_AMD : GL_FLOAT16_SAMPLER_1D_AMD;
                    case true:  return sampler.arrayed ? GL_FLOAT16_SAMPLER_1D_ARRAY_SHADOW_AMD : GL_FLOAT16_SAMPLER_1D_SHADOW_AMD;
                    }
                case Esd2D:
                    switch ((int)sampler.ms) {
                    case false:
                        switch ((int)sampler.shadow) {
                        case false: return sampler.arrayed ? GL_FLOAT16_SAMPLER_2D_ARRAY_AMD : GL_FLOAT16_SAMPLER_2D_AMD;
                        case true:  return sampler.arrayed ? GL_FLOAT16_SAMPLER_2D_ARRAY_SHADOW_AMD : GL_FLOAT16_SAMPLER_2D_SHADOW_AMD;
                        }
                    case true:      return sampler.arrayed ? GL_FLOAT16_SAMPLER_2D_MULTISAMPLE_ARRAY_AMD : GL_FLOAT16_SAMPLER_2D_MULTISAMPLE_AMD;
                    }
                case Esd3D:
                    return GL_FLOAT16_SAMPLER_3D_AMD;
                case EsdCube:
                    switch ((int)sampler.shadow) {
                    case false: return sampler.arrayed ? GL_FLOAT16_SAMPLER_CUBE_MAP_ARRAY_AMD : GL_FLOAT16_SAMPLER_CUBE_AMD;
                    case true:  return sampler.arrayed ? GL_FLOAT16_SAMPLER_CUBE_MAP_ARRAY_SHADOW_AMD : GL_FLOAT16_SAMPLER_CUBE_SHADOW_AMD;
                    }
                case EsdRect:
                    return sampler.shadow ? GL_FLOAT16_SAMPLER_2D_RECT_SHADOW_AMD : GL_FLOAT16_SAMPLER_2D_RECT_AMD;
                case EsdBuffer:
                    return GL_FLOAT16_SAMPLER_BUFFER_AMD;
                }
            case EbtInt:
                switch ((int)sampler.dim) {
                case Esd1D:
                    return sampler.arrayed ? GL_INT_SAMPLER_1D_ARRAY : GL_INT_SAMPLER_1D;
                case Esd2D:
                    switch ((int)sampler.ms) {
                    case false:  return sampler.arrayed ? GL_INT_SAMPLER_2D_ARRAY : GL_INT_SAMPLER_2D;
                    case true:   return sampler.arrayed ? GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY
                                                        : GL_INT_SAMPLER_2D_MULTISAMPLE;
                    }
                case Esd3D:
                    return GL_INT_SAMPLER_3D;
                case EsdCube:
                    return sampler.arrayed ? GL_INT_SAMPLER_CUBE_MAP_ARRAY : GL_INT_SAMPLER_CUBE;
                case EsdRect:
                    return GL_INT_SAMPLER_2D_RECT;
                case EsdBuffer:
                    return GL_INT_SAMPLER_BUFFER;
                }
            case EbtUint:
                switch ((int)sampler.dim) {
                case Esd1D:
                    return sampler.arrayed ? GL_UNSIGNED_INT_SAMPLER_1D_ARRAY : GL_UNSIGNED_INT_SAMPLER_1D;
                case Esd2D:
                    switch ((int)sampler.ms) {
                    case false:  return sampler.arrayed ? GL_UNSIGNED_INT_SAMPLER_2D_ARRAY : GL_UNSIGNED_INT_SAMPLER_2D;
                    case true:   return sampler.arrayed ? GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY
                                                        : GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE;
                    }
                case Esd3D:
                    return GL_UNSIGNED_INT_SAMPLER_3D;
                case EsdCube:
                    return sampler.arrayed ? GL_UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY : GL_UNSIGNED_INT_SAMPLER_CUBE;
                case EsdRect:
                    return GL_UNSIGNED_INT_SAMPLER_2D_RECT;
                case EsdBuffer:
                    return GL_UNSIGNED_INT_SAMPLER_BUFFER;
                }
            default:
                return 0;
            }
        } else {
            // an image...
            switch (sampler.type) {
            case EbtFloat:
                switch ((int)sampler.dim) {
                case Esd1D:
                    return sampler.arrayed ? GL_IMAGE_1D_ARRAY : GL_IMAGE_1D;
                case Esd2D:
                    switch ((int)sampler.ms) {
                    case false:     return sampler.arrayed ? GL_IMAGE_2D_ARRAY : GL_IMAGE_2D;
                    case true:      return sampler.arrayed ? GL_IMAGE_2D_MULTISAMPLE_ARRAY : GL_IMAGE_2D_MULTISAMPLE;
                    }
                case Esd3D:
                    return GL_IMAGE_3D;
                case EsdCube:
                    return sampler.arrayed ? GL_IMAGE_CUBE_MAP_ARRAY : GL_IMAGE_CUBE;
                case EsdRect:
                    return GL_IMAGE_2D_RECT;
                case EsdBuffer:
                    return GL_IMAGE_BUFFER;
                }
            case EbtFloat16:
                switch ((int)sampler.dim) {
                case Esd1D:
                    return sampler.arrayed ? GL_FLOAT16_IMAGE_1D_ARRAY_AMD : GL_FLOAT16_IMAGE_1D_AMD;
                case Esd2D:
                    switch ((int)sampler.ms) {
                    case false:     return sampler.arrayed ? GL_FLOAT16_IMAGE_2D_ARRAY_AMD : GL_FLOAT16_IMAGE_2D_AMD;
                    case true:      return sampler.arrayed ? GL_FLOAT16_IMAGE_2D_MULTISAMPLE_ARRAY_AMD : GL_FLOAT16_IMAGE_2D_MULTISAMPLE_AMD;
                    }
                case Esd3D:
                    return GL_FLOAT16_IMAGE_3D_AMD;
                case EsdCube:
                    return sampler.arrayed ? GL_FLOAT16_IMAGE_CUBE_MAP_ARRAY_AMD : GL_FLOAT16_IMAGE_CUBE_AMD;
                case EsdRect:
                    return GL_FLOAT16_IMAGE_2D_RECT_AMD;
                case EsdBuffer:
                    return GL_FLOAT16_IMAGE_BUFFER_AMD;
                }
            case EbtInt:
                switch ((int)sampler.dim) {
                case Esd1D:
                    return sampler.arrayed ? GL_INT_IMAGE_1D_ARRAY : GL_INT_IMAGE_1D;
                case Esd2D:
                    switch ((int)sampler.ms) {
                    case false:  return sampler.arrayed ? GL_INT_IMAGE_2D_ARRAY : GL_INT_IMAGE_2D;
                    case true:   return sampler.arrayed ? GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY : GL_INT_IMAGE_2D_MULTISAMPLE;
                    }
                case Esd3D:
                    return GL_INT_IMAGE_3D;
                case EsdCube:
                    return sampler.arrayed ? GL_INT_IMAGE_CUBE_MAP_ARRAY : GL_INT_IMAGE_CUBE;
                case EsdRect:
                    return GL_INT_IMAGE_2D_RECT;
                case EsdBuffer:
                    return GL_INT_IMAGE_BUFFER;
                }
            case EbtUint:
                switch ((int)sampler.dim) {
                case Esd1D:
                    return sampler.arrayed ? GL_UNSIGNED_INT_IMAGE_1D_ARRAY : GL_UNSIGNED_INT_IMAGE_1D;
                case Esd2D:
                    switch ((int)sampler.ms) {
                    case false:  return sampler.arrayed ? GL_UNSIGNED_INT_IMAGE_2D_ARRAY : GL_UNSIGNED_INT_IMAGE_2D;
                    case true:   return sampler.arrayed ? GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY
                                                        : GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE;
                    }
                case Esd3D:
                    return GL_UNSIGNED_INT_IMAGE_3D;
                case EsdCube:
                    return sampler.arrayed ? GL_UNSIGNED_INT_IMAGE_CUBE_MAP_ARRAY : GL_UNSIGNED_INT_IMAGE_CUBE;
                case EsdRect:
                    return GL_UNSIGNED_INT_IMAGE_2D_RECT;
                case EsdBuffer:
                    return GL_UNSIGNED_INT_IMAGE_BUFFER;
                }
            default:
                return 0;
            }
        }
    }

    //
    // Translate a glslang type into the GL API #define number.
    // Ignores arrayness.
    //
    int mapToGlType(const TType& type)
    {
        switch (type.getBasicType()) {
        case EbtSampler:
            return mapSamplerToGlType(type.getSampler());
        case EbtStruct:
        case EbtBlock:
        case EbtVoid:
            return 0;
        default:
            break;
        }

        if (type.isVector()) {
            int offset = type.getVectorSize() - 2;
            switch (type.getBasicType()) {
            case EbtFloat:      return GL_FLOAT_VEC2                  + offset;
            case EbtDouble:     return GL_DOUBLE_VEC2                 + offset;
            case EbtFloat16:    return GL_FLOAT16_VEC2_NV             + offset;
            case EbtInt:        return GL_INT_VEC2                    + offset;
            case EbtUint:       return GL_UNSIGNED_INT_VEC2           + offset;
            case EbtInt64:      return GL_INT64_ARB                   + offset;
            case EbtUint64:     return GL_UNSIGNED_INT64_ARB          + offset;
            case EbtBool:       return GL_BOOL_VEC2                   + offset;
            case EbtAtomicUint: return GL_UNSIGNED_INT_ATOMIC_COUNTER + offset;
            default:            return 0;
            }
        }
        if (type.isMatrix()) {
            switch (type.getBasicType()) {
            case EbtFloat:
                switch (type.getMatrixCols()) {
                case 2:
                    switch (type.getMatrixRows()) {
                    case 2:    return GL_FLOAT_MAT2;
                    case 3:    return GL_FLOAT_MAT2x3;
                    case 4:    return GL_FLOAT_MAT2x4;
                    default:   return 0;
                    }
                case 3:
                    switch (type.getMatrixRows()) {
                    case 2:    return GL_FLOAT_MAT3x2;
                    case 3:    return GL_FLOAT_MAT3;
                    case 4:    return GL_FLOAT_MAT3x4;
                    default:   return 0;
                    }
                case 4:
                    switch (type.getMatrixRows()) {
                    case 2:    return GL_FLOAT_MAT4x2;
                    case 3:    return GL_FLOAT_MAT4x3;
                    case 4:    return GL_FLOAT_MAT4;
                    default:   return 0;
                    }
                }
            case EbtDouble:
                switch (type.getMatrixCols()) {
                case 2:
                    switch (type.getMatrixRows()) {
                    case 2:    return GL_DOUBLE_MAT2;
                    case 3:    return GL_DOUBLE_MAT2x3;
                    case 4:    return GL_DOUBLE_MAT2x4;
                    default:   return 0;
                    }
                case 3:
                    switch (type.getMatrixRows()) {
                    case 2:    return GL_DOUBLE_MAT3x2;
                    case 3:    return GL_DOUBLE_MAT3;
                    case 4:    return GL_DOUBLE_MAT3x4;
                    default:   return 0;
                    }
                case 4:
                    switch (type.getMatrixRows()) {
                    case 2:    return GL_DOUBLE_MAT4x2;
                    case 3:    return GL_DOUBLE_MAT4x3;
                    case 4:    return GL_DOUBLE_MAT4;
                    default:   return 0;
                    }
                }
            case EbtFloat16:
                switch (type.getMatrixCols()) {
                case 2:
                    switch (type.getMatrixRows()) {
                    case 2:    return GL_FLOAT16_MAT2_AMD;
                    case 3:    return GL_FLOAT16_MAT2x3_AMD;
                    case 4:    return GL_FLOAT16_MAT2x4_AMD;
                    default:   return 0;
                    }
                case 3:
                    switch (type.getMatrixRows()) {
                    case 2:    return GL_FLOAT16_MAT3x2_AMD;
                    case 3:    return GL_FLOAT16_MAT3_AMD;
                    case 4:    return GL_FLOAT16_MAT3x4_AMD;
                    default:   return 0;
                    }
                case 4:
                    switch (type.getMatrixRows()) {
                    case 2:    return GL_FLOAT16_MAT4x2_AMD;
                    case 3:    return GL_FLOAT16_MAT4x3_AMD;
                    case 4:    return GL_FLOAT16_MAT4_AMD;
                    default:   return 0;
                    }
                }
            default:
                return 0;
            }
        }
        if (type.getVectorSize() == 1) {
            switch (type.getBasicType()) {
            case EbtFloat:      return GL_FLOAT;
            case EbtDouble:     return GL_DOUBLE;
            case EbtFloat16:    return GL_FLOAT16_NV;
            case EbtInt:        return GL_INT;
            case EbtUint:       return GL_UNSIGNED_INT;
            case EbtInt64:      return GL_INT64_ARB;
            case EbtUint64:     return GL_UNSIGNED_INT64_ARB;
            case EbtBool:       return GL_BOOL;
            case EbtAtomicUint: return GL_UNSIGNED_INT_ATOMIC_COUNTER;
            default:            return 0;
            }
        }

        return 0;
    }

    int mapToGlArraySize(const TType& type)
    {
        return type.isArray() ? type.getOuterArraySize() : 1;
    }

    const TIntermediate& intermediate;
    TReflection& reflection;
    std::set<const TIntermNode*> processedDerefs;
    bool updateStageMasks;

protected:
    TReflectionTraverser(TReflectionTraverser&);
    TReflectionTraverser& operator=(TReflectionTraverser&);
};

//
// Implement the traversal functions of interest.
//

// To catch dereferenced aggregates that must be reflected.
// This catches them at the highest level possible in the tree.
bool TReflectionTraverser::visitBinary(TVisit /* visit */, TIntermBinary* node)
{
    switch (node->getOp()) {
    case EOpIndexDirect:
    case EOpIndexIndirect:
    case EOpIndexDirectStruct:
        addDereferencedUniform(node);
        break;
    default:
        break;
    }

    // still need to visit everything below, which could contain sub-expressions
    // containing different uniforms
    return true;
}

// To reflect non-dereferenced objects.
void TReflectionTraverser::visitSymbol(TIntermSymbol* base)
{
    if (base->getQualifier().storage == EvqUniform) {
        if (base->getBasicType() == EbtBlock) {
            if (reflection.options & EShReflectionSharedStd140UBO) {
                addUniform(*base);
            }
        } else {
            addUniform(*base);
        }
    }

    // #TODO add std140/layout active rules for ssbo, same with ubo.
    // Storage buffer blocks will be collected and expanding in this part.
    if((reflection.options & EShReflectionSharedStd140SSBO) &&
       (base->getQualifier().storage == EvqBuffer && base->getBasicType() == EbtBlock &&
        (base->getQualifier().layoutPacking == ElpStd140 || base->getQualifier().layoutPacking == ElpShared)))
        addUniform(*base);

    if ((intermediate.getStage() == reflection.firstStage && base->getQualifier().isPipeInput()) ||
        (intermediate.getStage() == reflection.lastStage && base->getQualifier().isPipeOutput()))
        addPipeIOVariable(*base);
}

//
// Implement TObjectReflection methods.
//

TObjectReflection::TObjectReflection(const std::string &pName, const TType &pType, int pOffset, int pGLDefineType,
                                     int pSize, int pIndex)
    : name(pName), offset(pOffset), glDefineType(pGLDefineType), size(pSize), index(pIndex), counterIndex(-1),
      numMembers(-1), arrayStride(0), topLevelArrayStride(0), stages(EShLanguageMask(0)), type(pType.clone())
{
}

int TObjectReflection::getBinding() const
{
    if (type == nullptr || !type->getQualifier().hasBinding())
        return -1;
    return type->getQualifier().layoutBinding;
}

void TObjectReflection::dump() const
{
    printf("%s: offset %d, type %x, size %d, index %d, binding %d, stages %d", name.c_str(), offset, glDefineType, size,
           index, getBinding(), stages);

    if (counterIndex != -1)
        printf(", counter %d", counterIndex);

    if (numMembers != -1)
        printf(", numMembers %d", numMembers);

    if (arrayStride != 0)
        printf(", arrayStride %d", arrayStride);

    if (topLevelArrayStride != 0)
        printf(", topLevelArrayStride %d", topLevelArrayStride);

    printf("\n");
}

//
// Implement TReflection methods.
//

// Track any required attribute reflection, such as compute shader numthreads.
//
void TReflection::buildAttributeReflection(EShLanguage stage, const TIntermediate& intermediate)
{
    if (stage == EShLangCompute) {
        // Remember thread dimensions
        for (int dim=0; dim<3; ++dim)
            localSize[dim] = intermediate.getLocalSize(dim);
    }
}

// build counter block index associations for buffers
void TReflection::buildCounterIndices(const TIntermediate& intermediate)
{
#ifdef ENABLE_HLSL
    // search for ones that have counters
    for (int i = 0; i < int(indexToUniformBlock.size()); ++i) {
        const TString counterName(intermediate.addCounterBufferName(indexToUniformBlock[i].name).c_str());
        const int index = getIndex(counterName);

        if (index >= 0)
            indexToUniformBlock[i].counterIndex = index;
    }
#endif
}

// build Shader Stages mask for all uniforms
void TReflection::buildUniformStageMask(const TIntermediate& intermediate)
{
    if (options & EShReflectionAllBlockVariables)
        return;

    for (int i = 0; i < int(indexToUniform.size()); ++i) {
        indexToUniform[i].stages = static_cast<EShLanguageMask>(indexToUniform[i].stages | 1 << intermediate.getStage());
    }

    for (int i = 0; i < int(indexToBufferVariable.size()); ++i) {
        indexToBufferVariable[i].stages =
            static_cast<EShLanguageMask>(indexToBufferVariable[i].stages | 1 << intermediate.getStage());
    }
}

// Merge live symbols from 'intermediate' into the existing reflection database.
//
// Returns false if the input is too malformed to do this.
bool TReflection::addStage(EShLanguage stage, const TIntermediate& intermediate)
{
    if (intermediate.getTreeRoot() == nullptr ||
        intermediate.getNumEntryPoints() != 1 ||
        intermediate.isRecursive())
        return false;

    buildAttributeReflection(stage, intermediate);

    TReflectionTraverser it(intermediate, *this);

    for (auto& sequnence : intermediate.getTreeRoot()->getAsAggregate()->getSequence()) {
        if (sequnence->getAsAggregate() != nullptr) {
            if (sequnence->getAsAggregate()->getOp() == glslang::EOpLinkerObjects) {
                it.updateStageMasks = false;
                TIntermAggregate* linkerObjects = sequnence->getAsAggregate();
                for (auto& sequnence : linkerObjects->getSequence()) {
                    auto pNode = sequnence->getAsSymbolNode();
                    if (pNode != nullptr) {
                        if ((pNode->getQualifier().storage == EvqUniform &&
                            (options & EShReflectionSharedStd140UBO)) ||
                           (pNode->getQualifier().storage == EvqBuffer &&
                            (options & EShReflectionSharedStd140SSBO))) {
                            // collect std140 and shared uniform block form AST
                            if ((pNode->getBasicType() == EbtBlock) &&
                                ((pNode->getQualifier().layoutPacking == ElpStd140) ||
                                 (pNode->getQualifier().layoutPacking == ElpShared))) {
                                   pNode->traverse(&it);
                            }
                        }
                        else if ((options & EShReflectionAllIOVariables) &&
                            (pNode->getQualifier().isPipeInput() || pNode->getQualifier().isPipeOutput()))
                        {
                            pNode->traverse(&it);
                        }
                    }
                }
            } else {
                // This traverser will travers all function in AST.
                // If we want reflect uncalled function, we need set linke message EShMsgKeepUncalled.
                // When EShMsgKeepUncalled been set to true, all function will be keep in AST, even it is a uncalled function.
                // This will keep some uniform variables in reflection, if those uniform variables is used in these uncalled function.
                //
                // If we just want reflect only live node, we can use a default link message or set EShMsgKeepUncalled false.
                // When linke message not been set EShMsgKeepUncalled, linker won't keep uncalled function in AST.
                // So, travers all function node can equivalent to travers live function.
                it.updateStageMasks = true;
                sequnence->getAsAggregate()->traverse(&it);
            }
        }
    }
    it.updateStageMasks = true;

    buildCounterIndices(intermediate);
    buildUniformStageMask(intermediate);

    return true;
}

void TReflection::dump()
{
    printf("Uniform reflection:\n");
    for (size_t i = 0; i < indexToUniform.size(); ++i)
        indexToUniform[i].dump();
    printf("\n");

    printf("Uniform block reflection:\n");
    for (size_t i = 0; i < indexToUniformBlock.size(); ++i)
        indexToUniformBlock[i].dump();
    printf("\n");

    printf("Buffer variable reflection:\n");
    for (size_t i = 0; i < indexToBufferVariable.size(); ++i)
      indexToBufferVariable[i].dump();
    printf("\n");

    printf("Buffer block reflection:\n");
    for (size_t i = 0; i < indexToBufferBlock.size(); ++i)
      indexToBufferBlock[i].dump();
    printf("\n");

    printf("Pipeline input reflection:\n");
    for (size_t i = 0; i < indexToPipeInput.size(); ++i)
        indexToPipeInput[i].dump();
    printf("\n");

    printf("Pipeline output reflection:\n");
    for (size_t i = 0; i < indexToPipeOutput.size(); ++i)
        indexToPipeOutput[i].dump();
    printf("\n");

    if (getLocalSize(0) > 1) {
        static const char* axis[] = { "X", "Y", "Z" };

        for (int dim=0; dim<3; ++dim)
            if (getLocalSize(dim) > 1)
                printf("Local size %s: %u\n", axis[dim], getLocalSize(dim));

        printf("\n");
    }

    // printf("Live names\n");
    // for (TNameToIndex::const_iterator it = nameToIndex.begin(); it != nameToIndex.end(); ++it)
    //    printf("%s: %d\n", it->first.c_str(), it->second);
    // printf("\n");
}

} // end namespace glslang

#endif // !GLSLANG_WEB && !GLSLANG_ANGLE
