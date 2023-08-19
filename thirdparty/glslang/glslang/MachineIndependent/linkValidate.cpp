//
// Copyright (C) 2013 LunarG, Inc.
// Copyright (C) 2017 ARM Limited.
// Copyright (C) 2015-2018 Google, Inc.
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
// Do link-time merging and validation of intermediate representations.
//
// Basic model is that during compilation, each compilation unit (shader) is
// compiled into one TIntermediate instance.  Then, at link time, multiple
// units for the same stage can be merged together, which can generate errors.
// Then, after all merging, a single instance of TIntermediate represents
// the whole stage.  A final error check can be done on the resulting stage,
// even if no merging was done (i.e., the stage was only one compilation unit).
//

#include "localintermediate.h"
#include "../Include/InfoSink.h"
#include "SymbolTable.h"

namespace glslang {

//
// Link-time error emitter.
//
void TIntermediate::error(TInfoSink& infoSink, const char* message, EShLanguage unitStage)
{
#ifndef GLSLANG_WEB
    infoSink.info.prefix(EPrefixError);
    if (unitStage < EShLangCount)
        infoSink.info << "Linking " << StageName(getStage()) << " and " << StageName(unitStage) << " stages: " << message << "\n";
    else
        infoSink.info << "Linking " << StageName(language) << " stage: " << message << "\n";
#endif

    ++numErrors;
}

// Link-time warning.
void TIntermediate::warn(TInfoSink& infoSink, const char* message, EShLanguage unitStage)
{
#ifndef GLSLANG_WEB
    infoSink.info.prefix(EPrefixWarning);
    if (unitStage < EShLangCount)
        infoSink.info << "Linking " << StageName(language) << " and " << StageName(unitStage) << " stages: " << message << "\n";
    else
        infoSink.info << "Linking " << StageName(language) << " stage: " << message << "\n";
#endif
}

// TODO: 4.4 offset/align:  "Two blocks linked together in the same program with the same block
// name must have the exact same set of members qualified with offset and their integral-constant
// expression values must be the same, or a link-time error results."

//
// Merge the information from 'unit' into 'this'
//
void TIntermediate::merge(TInfoSink& infoSink, TIntermediate& unit)
{
#if !defined(GLSLANG_WEB)
    mergeCallGraphs(infoSink, unit);
    mergeModes(infoSink, unit);
    mergeTrees(infoSink, unit);
#endif
}

//
// check that link objects between stages
//
void TIntermediate::mergeUniformObjects(TInfoSink& infoSink, TIntermediate& unit) {
    if (unit.treeRoot == nullptr || treeRoot == nullptr)
        return;

    // Get the linker-object lists
    TIntermSequence& linkerObjects = findLinkerObjects()->getSequence();
    TIntermSequence unitLinkerObjects = unit.findLinkerObjects()->getSequence();

    // filter unitLinkerObjects to only contain uniforms
    auto end = std::remove_if(unitLinkerObjects.begin(), unitLinkerObjects.end(),
        [](TIntermNode* node) {return node->getAsSymbolNode()->getQualifier().storage != EvqUniform &&
                                      node->getAsSymbolNode()->getQualifier().storage != EvqBuffer; });
    unitLinkerObjects.resize(end - unitLinkerObjects.begin());

    // merge uniforms and do error checking
    bool mergeExistingOnly = false;
    mergeGlobalUniformBlocks(infoSink, unit, mergeExistingOnly);
    mergeLinkerObjects(infoSink, linkerObjects, unitLinkerObjects, unit.getStage());
}

//
// do error checking on the shader boundary in / out vars
//
void TIntermediate::checkStageIO(TInfoSink& infoSink, TIntermediate& unit) {
    if (unit.treeRoot == nullptr || treeRoot == nullptr)
        return;

    // Get copies of the linker-object lists
    TIntermSequence linkerObjects = findLinkerObjects()->getSequence();
    TIntermSequence unitLinkerObjects = unit.findLinkerObjects()->getSequence();

    // filter linkerObjects to only contain out variables
    auto end = std::remove_if(linkerObjects.begin(), linkerObjects.end(),
        [](TIntermNode* node) {return node->getAsSymbolNode()->getQualifier().storage != EvqVaryingOut; });
    linkerObjects.resize(end - linkerObjects.begin());

    // filter unitLinkerObjects to only contain in variables
    auto unitEnd = std::remove_if(unitLinkerObjects.begin(), unitLinkerObjects.end(),
        [](TIntermNode* node) {return node->getAsSymbolNode()->getQualifier().storage != EvqVaryingIn; });
    unitLinkerObjects.resize(unitEnd - unitLinkerObjects.begin());

    // do matching and error checking
    mergeLinkerObjects(infoSink, linkerObjects, unitLinkerObjects, unit.getStage());

    // TODO: final check; make sure that any statically used `in` have matching `out` written to
}

void TIntermediate::mergeCallGraphs(TInfoSink& infoSink, TIntermediate& unit)
{
    if (unit.getNumEntryPoints() > 0) {
        if (getNumEntryPoints() > 0)
            error(infoSink, "can't handle multiple entry points per stage");
        else {
            entryPointName = unit.getEntryPointName();
            entryPointMangledName = unit.getEntryPointMangledName();
        }
    }
    numEntryPoints += unit.getNumEntryPoints();

    callGraph.insert(callGraph.end(), unit.callGraph.begin(), unit.callGraph.end());
}

#if !defined(GLSLANG_WEB)

#define MERGE_MAX(member) member = std::max(member, unit.member)
#define MERGE_TRUE(member) if (unit.member) member = unit.member;

void TIntermediate::mergeModes(TInfoSink& infoSink, TIntermediate& unit)
{
    if (language != unit.language)
        error(infoSink, "stages must match when linking into a single stage");

    if (getSource() == EShSourceNone)
        setSource(unit.getSource());
    if (getSource() != unit.getSource())
        error(infoSink, "can't link compilation units from different source languages");

    if (treeRoot == nullptr) {
        profile = unit.profile;
        version = unit.version;
        requestedExtensions = unit.requestedExtensions;
    } else {
        if ((isEsProfile()) != (unit.isEsProfile()))
            error(infoSink, "Cannot cross link ES and desktop profiles");
        else if (unit.profile == ECompatibilityProfile)
            profile = ECompatibilityProfile;
        version = std::max(version, unit.version);
        requestedExtensions.insert(unit.requestedExtensions.begin(), unit.requestedExtensions.end());
    }

    MERGE_MAX(spvVersion.spv);
    MERGE_MAX(spvVersion.vulkanGlsl);
    MERGE_MAX(spvVersion.vulkan);
    MERGE_MAX(spvVersion.openGl);
    MERGE_TRUE(spvVersion.vulkanRelaxed);

    numErrors += unit.getNumErrors();
    // Only one push_constant is allowed, mergeLinkerObjects() will ensure the push_constant
    // is the same for all units.
    if (numPushConstants > 1 || unit.numPushConstants > 1)
        error(infoSink, "Only one push_constant block is allowed per stage");
    numPushConstants = std::min(numPushConstants + unit.numPushConstants, 1);

    if (unit.invocations != TQualifier::layoutNotSet) {
        if (invocations == TQualifier::layoutNotSet)
            invocations = unit.invocations;
        else if (invocations != unit.invocations)
            error(infoSink, "number of invocations must match between compilation units");
    }

    if (vertices == TQualifier::layoutNotSet)
        vertices = unit.vertices;
    else if (unit.vertices != TQualifier::layoutNotSet && vertices != unit.vertices) {
        if (language == EShLangGeometry || language == EShLangMesh)
            error(infoSink, "Contradictory layout max_vertices values");
        else if (language == EShLangTessControl)
            error(infoSink, "Contradictory layout vertices values");
        else
            assert(0);
    }
    if (primitives == TQualifier::layoutNotSet)
        primitives = unit.primitives;
    else if (primitives != unit.primitives) {
        if (language == EShLangMesh)
            error(infoSink, "Contradictory layout max_primitives values");
        else
            assert(0);
    }

    if (inputPrimitive == ElgNone)
        inputPrimitive = unit.inputPrimitive;
    else if (unit.inputPrimitive != ElgNone && inputPrimitive != unit.inputPrimitive)
        error(infoSink, "Contradictory input layout primitives");

    if (outputPrimitive == ElgNone)
        outputPrimitive = unit.outputPrimitive;
    else if (unit.outputPrimitive != ElgNone && outputPrimitive != unit.outputPrimitive)
        error(infoSink, "Contradictory output layout primitives");

    if (originUpperLeft != unit.originUpperLeft || pixelCenterInteger != unit.pixelCenterInteger)
        error(infoSink, "gl_FragCoord redeclarations must match across shaders");

    if (vertexSpacing == EvsNone)
        vertexSpacing = unit.vertexSpacing;
    else if (vertexSpacing != unit.vertexSpacing)
        error(infoSink, "Contradictory input vertex spacing");

    if (vertexOrder == EvoNone)
        vertexOrder = unit.vertexOrder;
    else if (vertexOrder != unit.vertexOrder)
        error(infoSink, "Contradictory triangle ordering");

    MERGE_TRUE(pointMode);

    for (int i = 0; i < 3; ++i) {
        if (unit.localSizeNotDefault[i]) {
            if (!localSizeNotDefault[i]) {
                localSize[i] = unit.localSize[i];
                localSizeNotDefault[i] = true;
            }
            else if (localSize[i] != unit.localSize[i])
                error(infoSink, "Contradictory local size");
        }

        if (localSizeSpecId[i] == TQualifier::layoutNotSet)
            localSizeSpecId[i] = unit.localSizeSpecId[i];
        else if (localSizeSpecId[i] != unit.localSizeSpecId[i])
            error(infoSink, "Contradictory local size specialization ids");
    }

    MERGE_TRUE(earlyFragmentTests);
    MERGE_TRUE(postDepthCoverage);
    MERGE_TRUE(nonCoherentColorAttachmentReadEXT);
    MERGE_TRUE(nonCoherentDepthAttachmentReadEXT);
    MERGE_TRUE(nonCoherentStencilAttachmentReadEXT);

    if (depthLayout == EldNone)
        depthLayout = unit.depthLayout;
    else if (depthLayout != unit.depthLayout)
        error(infoSink, "Contradictory depth layouts");

    MERGE_TRUE(depthReplacing);
    MERGE_TRUE(hlslFunctionality1);

    blendEquations |= unit.blendEquations;

    MERGE_TRUE(xfbMode);

    for (size_t b = 0; b < xfbBuffers.size(); ++b) {
        if (xfbBuffers[b].stride == TQualifier::layoutXfbStrideEnd)
            xfbBuffers[b].stride = unit.xfbBuffers[b].stride;
        else if (xfbBuffers[b].stride != unit.xfbBuffers[b].stride)
            error(infoSink, "Contradictory xfb_stride");
        xfbBuffers[b].implicitStride = std::max(xfbBuffers[b].implicitStride, unit.xfbBuffers[b].implicitStride);
        if (unit.xfbBuffers[b].contains64BitType)
            xfbBuffers[b].contains64BitType = true;
        if (unit.xfbBuffers[b].contains32BitType)
            xfbBuffers[b].contains32BitType = true;
        if (unit.xfbBuffers[b].contains16BitType)
            xfbBuffers[b].contains16BitType = true;
        // TODO: 4.4 link: enhanced layouts: compare ranges
    }

    MERGE_TRUE(multiStream);
    MERGE_TRUE(layoutOverrideCoverage);
    MERGE_TRUE(geoPassthroughEXT);

    for (unsigned int i = 0; i < unit.shiftBinding.size(); ++i) {
        if (unit.shiftBinding[i] > 0)
            setShiftBinding((TResourceType)i, unit.shiftBinding[i]);
    }

    for (unsigned int i = 0; i < unit.shiftBindingForSet.size(); ++i) {
        for (auto it = unit.shiftBindingForSet[i].begin(); it != unit.shiftBindingForSet[i].end(); ++it)
            setShiftBindingForSet((TResourceType)i, it->second, it->first);
    }

    resourceSetBinding.insert(resourceSetBinding.end(), unit.resourceSetBinding.begin(), unit.resourceSetBinding.end());

    MERGE_TRUE(autoMapBindings);
    MERGE_TRUE(autoMapLocations);
    MERGE_TRUE(invertY);
    MERGE_TRUE(dxPositionW);
    MERGE_TRUE(debugInfo);
    MERGE_TRUE(flattenUniformArrays);
    MERGE_TRUE(useUnknownFormat);
    MERGE_TRUE(hlslOffsets);
    MERGE_TRUE(useStorageBuffer);
    MERGE_TRUE(invariantAll);
    MERGE_TRUE(hlslIoMapping);

    // TODO: sourceFile
    // TODO: sourceText
    // TODO: processes

    MERGE_TRUE(needToLegalize);
    MERGE_TRUE(binaryDoubleOutput);
    MERGE_TRUE(usePhysicalStorageBuffer);
}

//
// Merge the 'unit' AST into 'this' AST.
// That includes rationalizing the unique IDs, which were set up independently,
// and might have overlaps that are not the same symbol, or might have different
// IDs for what should be the same shared symbol.
//
void TIntermediate::mergeTrees(TInfoSink& infoSink, TIntermediate& unit)
{
    if (unit.treeRoot == nullptr)
        return;

    if (treeRoot == nullptr) {
        treeRoot = unit.treeRoot;
        return;
    }

    // Getting this far means we have two existing trees to merge...
    numShaderRecordBlocks += unit.numShaderRecordBlocks;
    numTaskNVBlocks += unit.numTaskNVBlocks;

    // Get the top-level globals of each unit
    TIntermSequence& globals = treeRoot->getAsAggregate()->getSequence();
    TIntermSequence& unitGlobals = unit.treeRoot->getAsAggregate()->getSequence();

    // Get the linker-object lists
    TIntermSequence& linkerObjects = findLinkerObjects()->getSequence();
    const TIntermSequence& unitLinkerObjects = unit.findLinkerObjects()->getSequence();

    // Map by global name to unique ID to rationalize the same object having
    // differing IDs in different trees.
    TIdMaps idMaps;
    long long idShift;
    seedIdMap(idMaps, idShift);
    remapIds(idMaps, idShift + 1, unit);

    mergeBodies(infoSink, globals, unitGlobals);
    bool mergeExistingOnly = false;
    mergeGlobalUniformBlocks(infoSink, unit, mergeExistingOnly);
    mergeLinkerObjects(infoSink, linkerObjects, unitLinkerObjects, unit.getStage());
    ioAccessed.insert(unit.ioAccessed.begin(), unit.ioAccessed.end());
}

#endif

static const TString& getNameForIdMap(TIntermSymbol* symbol)
{
    TShaderInterface si = symbol->getType().getShaderInterface();
    if (si == EsiNone)
        return symbol->getName();
    else
        return symbol->getType().getTypeName();
}



// Traverser that seeds an ID map with all built-ins, and tracks the
// maximum ID used, currently using (maximum ID + 1) as new symbol id shift seed.
// Level id will keep same after shifting.
// (It would be nice to put this in a function, but that causes warnings
// on having no bodies for the copy-constructor/operator=.)
class TBuiltInIdTraverser : public TIntermTraverser {
public:
    TBuiltInIdTraverser(TIdMaps& idMaps) : idMaps(idMaps), idShift(0) { }
    // If it's a built in, add it to the map.
    virtual void visitSymbol(TIntermSymbol* symbol)
    {
        const TQualifier& qualifier = symbol->getType().getQualifier();
        if (qualifier.builtIn != EbvNone) {
            TShaderInterface si = symbol->getType().getShaderInterface();
            idMaps[si][getNameForIdMap(symbol)] = symbol->getId();
        }
        idShift = (symbol->getId() & ~TSymbolTable::uniqueIdMask) |
                std::max(idShift & TSymbolTable::uniqueIdMask,
                         symbol->getId() & TSymbolTable::uniqueIdMask);
    }
    long long getIdShift() const { return idShift; }
protected:
    TBuiltInIdTraverser(TBuiltInIdTraverser&);
    TBuiltInIdTraverser& operator=(TBuiltInIdTraverser&);
    TIdMaps& idMaps;
    long long idShift;
};

// Traverser that seeds an ID map with non-builtins.
// (It would be nice to put this in a function, but that causes warnings
// on having no bodies for the copy-constructor/operator=.)
class TUserIdTraverser : public TIntermTraverser {
public:
    TUserIdTraverser(TIdMaps& idMaps) : idMaps(idMaps) { }
    // If its a non-built-in global, add it to the map.
    virtual void visitSymbol(TIntermSymbol* symbol)
    {
        const TQualifier& qualifier = symbol->getType().getQualifier();
        if (qualifier.builtIn == EbvNone) {
            TShaderInterface si = symbol->getType().getShaderInterface();
            idMaps[si][getNameForIdMap(symbol)] = symbol->getId();
        }
    }

protected:
    TUserIdTraverser(TUserIdTraverser&);
    TUserIdTraverser& operator=(TUserIdTraverser&);
    TIdMaps& idMaps; // over biggest id
};

// Initialize the the ID map with what we know of 'this' AST.
void TIntermediate::seedIdMap(TIdMaps& idMaps, long long& idShift)
{
    // all built-ins everywhere need to align on IDs and contribute to the max ID
    TBuiltInIdTraverser builtInIdTraverser(idMaps);
    treeRoot->traverse(&builtInIdTraverser);
    idShift = builtInIdTraverser.getIdShift() & TSymbolTable::uniqueIdMask;

    // user variables in the linker object list need to align on ids
    TUserIdTraverser userIdTraverser(idMaps);
    findLinkerObjects()->traverse(&userIdTraverser);
}

// Traverser to map an AST ID to what was known from the seeding AST.
// (It would be nice to put this in a function, but that causes warnings
// on having no bodies for the copy-constructor/operator=.)
class TRemapIdTraverser : public TIntermTraverser {
public:
    TRemapIdTraverser(const TIdMaps& idMaps, long long idShift) : idMaps(idMaps), idShift(idShift) { }
    // Do the mapping:
    //  - if the same symbol, adopt the 'this' ID
    //  - otherwise, ensure a unique ID by shifting to a new space
    virtual void visitSymbol(TIntermSymbol* symbol)
    {
        const TQualifier& qualifier = symbol->getType().getQualifier();
        bool remapped = false;
        if (qualifier.isLinkable() || qualifier.builtIn != EbvNone) {
            TShaderInterface si = symbol->getType().getShaderInterface();
            auto it = idMaps[si].find(getNameForIdMap(symbol));
            if (it != idMaps[si].end()) {
                uint64_t id = (symbol->getId() & ~TSymbolTable::uniqueIdMask) |
                    (it->second & TSymbolTable::uniqueIdMask);
                symbol->changeId(id);
                remapped = true;
            }
        }
        if (!remapped)
            symbol->changeId(symbol->getId() + idShift);
    }
protected:
    TRemapIdTraverser(TRemapIdTraverser&);
    TRemapIdTraverser& operator=(TRemapIdTraverser&);
    const TIdMaps& idMaps;
    long long idShift;
};

void TIntermediate::remapIds(const TIdMaps& idMaps, long long idShift, TIntermediate& unit)
{
    // Remap all IDs to either share or be unique, as dictated by the idMap and idShift.
    TRemapIdTraverser idTraverser(idMaps, idShift);
    unit.getTreeRoot()->traverse(&idTraverser);
}

//
// Merge the function bodies and global-level initializers from unitGlobals into globals.
// Will error check duplication of function bodies for the same signature.
//
void TIntermediate::mergeBodies(TInfoSink& infoSink, TIntermSequence& globals, const TIntermSequence& unitGlobals)
{
    // TODO: link-time performance: Processing in alphabetical order will be faster

    // Error check the global objects, not including the linker objects
    for (unsigned int child = 0; child < globals.size() - 1; ++child) {
        for (unsigned int unitChild = 0; unitChild < unitGlobals.size() - 1; ++unitChild) {
            TIntermAggregate* body = globals[child]->getAsAggregate();
            TIntermAggregate* unitBody = unitGlobals[unitChild]->getAsAggregate();
            if (body && unitBody && body->getOp() == EOpFunction && unitBody->getOp() == EOpFunction && body->getName() == unitBody->getName()) {
                error(infoSink, "Multiple function bodies in multiple compilation units for the same signature in the same stage:");
                infoSink.info << "    " << globals[child]->getAsAggregate()->getName() << "\n";
            }
        }
    }

    // Merge the global objects, just in front of the linker objects
    globals.insert(globals.end() - 1, unitGlobals.begin(), unitGlobals.end() - 1);
}

static inline bool isSameInterface(TIntermSymbol* symbol, EShLanguage stage, TIntermSymbol* unitSymbol, EShLanguage unitStage) {
    return // 1) same stage and same shader interface
        (stage == unitStage && symbol->getType().getShaderInterface() == unitSymbol->getType().getShaderInterface()) ||
        // 2) accross stages and both are uniform or buffer
        (symbol->getQualifier().storage == EvqUniform  && unitSymbol->getQualifier().storage == EvqUniform) ||
        (symbol->getQualifier().storage == EvqBuffer   && unitSymbol->getQualifier().storage == EvqBuffer) ||
        // 3) in/out matched across stage boundary
        (stage < unitStage && symbol->getQualifier().storage == EvqVaryingOut  && unitSymbol->getQualifier().storage == EvqVaryingIn) ||
        (unitStage < stage && symbol->getQualifier().storage == EvqVaryingIn && unitSymbol->getQualifier().storage == EvqVaryingOut);
}

//
// Global Unfiform block stores any default uniforms (i.e. uniforms without a block)
// If two linked stages declare the same member, they are meant to be the same uniform
// and need to be in the same block
// merge the members of different stages to allow them to be linked properly
// as a single block
//
void TIntermediate::mergeGlobalUniformBlocks(TInfoSink& infoSink, TIntermediate& unit, bool mergeExistingOnly)
{
    TIntermSequence& linkerObjects = findLinkerObjects()->getSequence();
    TIntermSequence& unitLinkerObjects = unit.findLinkerObjects()->getSequence();

    // build lists of default blocks from the intermediates
    TIntermSequence defaultBlocks;
    TIntermSequence unitDefaultBlocks;

    auto filter = [](TIntermSequence& list, TIntermNode* node) {
        if (node->getAsSymbolNode()->getQualifier().defaultBlock) {
            list.push_back(node);
        }
    };

    std::for_each(linkerObjects.begin(), linkerObjects.end(),
        [&defaultBlocks, &filter](TIntermNode* node) {
            filter(defaultBlocks, node);
        });
    std::for_each(unitLinkerObjects.begin(), unitLinkerObjects.end(),
        [&unitDefaultBlocks, &filter](TIntermNode* node) {
            filter(unitDefaultBlocks, node);
    });

    auto itUnitBlock = unitDefaultBlocks.begin();
    for (; itUnitBlock != unitDefaultBlocks.end(); itUnitBlock++) {

        bool add = !mergeExistingOnly;
        auto itBlock = defaultBlocks.begin();

        for (; itBlock != defaultBlocks.end(); itBlock++) {
            TIntermSymbol* block = (*itBlock)->getAsSymbolNode();
            TIntermSymbol* unitBlock = (*itUnitBlock)->getAsSymbolNode();

            assert(block && unitBlock);

            // if the two default blocks match, then merge their definitions
            if (block->getType().getTypeName() == unitBlock->getType().getTypeName() &&
                block->getQualifier().storage == unitBlock->getQualifier().storage) {
                add = false;
                mergeBlockDefinitions(infoSink, block, unitBlock, &unit);
            }
        }
        if (add) {
            // push back on original list; won't change the size of the list we're iterating over
            linkerObjects.push_back(*itUnitBlock);
        }
    }
}

void TIntermediate::mergeBlockDefinitions(TInfoSink& infoSink, TIntermSymbol* block, TIntermSymbol* unitBlock, TIntermediate* unit) {

    if (block->getType().getTypeName() != unitBlock->getType().getTypeName() ||
        block->getType().getBasicType() != unitBlock->getType().getBasicType() ||
        block->getQualifier().storage != unitBlock->getQualifier().storage ||
        block->getQualifier().layoutSet != unitBlock->getQualifier().layoutSet) {
        // different block names likely means different blocks
        return;
    }

    // merge the struct
    // order of declarations doesn't matter and they matched based on member name
    TTypeList* memberList = block->getType().getWritableStruct();
    TTypeList* unitMemberList = unitBlock->getType().getWritableStruct();

    // keep track of which members have changed position
    // so we don't have to search the array again
    std::map<unsigned int, unsigned int> memberIndexUpdates;

    size_t memberListStartSize = memberList->size();
    for (unsigned int i = 0; i < unitMemberList->size(); ++i) {
        bool merge = true;
        for (unsigned int j = 0; j < memberListStartSize; ++j) {
            if ((*memberList)[j].type->getFieldName() == (*unitMemberList)[i].type->getFieldName()) {
                merge = false;
                const TType* memberType = (*memberList)[j].type;
                const TType* unitMemberType = (*unitMemberList)[i].type;

                // compare types
                // don't need as many checks as when merging symbols, since
                // initializers and most qualifiers are stripped when the member is moved into the block
                if ((*memberType) != (*unitMemberType)) {
                    error(infoSink, "Types must match:");
                    infoSink.info << "    " << memberType->getFieldName() << ": ";
                    infoSink.info << "\"" << memberType->getCompleteString() << "\" versus ";
                    infoSink.info << "\"" << unitMemberType->getCompleteString() << "\"\n";
                }

                memberIndexUpdates[i] = j;
            }
        }
        if (merge) {
            memberList->push_back((*unitMemberList)[i]);
            memberIndexUpdates[i] = (unsigned int)memberList->size() - 1;
        }
    }

    // update symbol node in unit tree,
    // and other nodes that may reference it
    class TMergeBlockTraverser : public TIntermTraverser {
    public:
        TMergeBlockTraverser(const TIntermSymbol* newSym)
            : newSymbol(newSym), newType(nullptr), unit(nullptr), memberIndexUpdates(nullptr)
        {
        }
        TMergeBlockTraverser(const TIntermSymbol* newSym, const glslang::TType* unitType, glslang::TIntermediate* unit,
                             const std::map<unsigned int, unsigned int>* memberIdxUpdates)
            : TIntermTraverser(false, true), newSymbol(newSym), newType(unitType), unit(unit), memberIndexUpdates(memberIdxUpdates)
        {
        }
        virtual ~TMergeBlockTraverser() {}

        const TIntermSymbol* newSymbol;
        const glslang::TType* newType; // shallow copy of the new type
        glslang::TIntermediate* unit;   // intermediate that is being updated
        const std::map<unsigned int, unsigned int>* memberIndexUpdates;

        virtual void visitSymbol(TIntermSymbol* symbol)
        {
            if (newSymbol->getAccessName() == symbol->getAccessName() &&
                newSymbol->getQualifier().getBlockStorage() == symbol->getQualifier().getBlockStorage()) {
                // Each symbol node may have a local copy of the block structure.
                // Update those structures to match the new one post-merge
                *(symbol->getWritableType().getWritableStruct()) = *(newSymbol->getType().getStruct());
            }
        }

        virtual bool visitBinary(TVisit, glslang::TIntermBinary* node)
        {
            if (!unit || !newType || !memberIndexUpdates || memberIndexUpdates->empty())
                return true;

            if (node->getOp() == EOpIndexDirectStruct && node->getLeft()->getType() == *newType) {
                // this is a dereference to a member of the block since the
                // member list changed, need to update this to point to the
                // right index
                assert(node->getRight()->getAsConstantUnion());

                glslang::TIntermConstantUnion* constNode = node->getRight()->getAsConstantUnion();
                unsigned int memberIdx = constNode->getConstArray()[0].getUConst();
                unsigned int newIdx = memberIndexUpdates->at(memberIdx);
                TIntermTyped* newConstNode = unit->addConstantUnion(newIdx, node->getRight()->getLoc());

                node->setRight(newConstNode);
                delete constNode;

                return true;
            }
            return true;
        }
    };

    // 'this' may have symbols that are using the old block structure, so traverse the tree to update those
    // in 'visitSymbol'
    TMergeBlockTraverser finalLinkTraverser(block);
    getTreeRoot()->traverse(&finalLinkTraverser);

    // The 'unit' intermediate needs the block structures update, but also structure entry indices
    // may have changed from the old block to the new one that it was merged into, so update those
    // in 'visitBinary'
    TType newType;
    newType.shallowCopy(block->getType());
    TMergeBlockTraverser unitFinalLinkTraverser(block, &newType, unit, &memberIndexUpdates);
    unit->getTreeRoot()->traverse(&unitFinalLinkTraverser);

    // update the member list
    (*unitMemberList) = (*memberList);
}

//
// Merge the linker objects from unitLinkerObjects into linkerObjects.
// Duplication is expected and filtered out, but contradictions are an error.
//
void TIntermediate::mergeLinkerObjects(TInfoSink& infoSink, TIntermSequence& linkerObjects, const TIntermSequence& unitLinkerObjects, EShLanguage unitStage)
{
    // Error check and merge the linker objects (duplicates should not be created)
    std::size_t initialNumLinkerObjects = linkerObjects.size();
    for (unsigned int unitLinkObj = 0; unitLinkObj < unitLinkerObjects.size(); ++unitLinkObj) {
        bool merge = true;
        for (std::size_t linkObj = 0; linkObj < initialNumLinkerObjects; ++linkObj) {
            TIntermSymbol* symbol = linkerObjects[linkObj]->getAsSymbolNode();
            TIntermSymbol* unitSymbol = unitLinkerObjects[unitLinkObj]->getAsSymbolNode();
            assert(symbol && unitSymbol);

            bool isSameSymbol = false;
            // If they are both blocks in the same shader interface,
            // match by the block-name, not the identifier name.
            if (symbol->getType().getBasicType() == EbtBlock && unitSymbol->getType().getBasicType() == EbtBlock) {
                if (isSameInterface(symbol, getStage(), unitSymbol, unitStage)) {
                    isSameSymbol = symbol->getType().getTypeName() == unitSymbol->getType().getTypeName();
                }
            }
            else if (symbol->getName() == unitSymbol->getName())
                isSameSymbol = true;

            if (isSameSymbol) {
                // filter out copy
                merge = false;

                // but if one has an initializer and the other does not, update
                // the initializer
                if (symbol->getConstArray().empty() && ! unitSymbol->getConstArray().empty())
                    symbol->setConstArray(unitSymbol->getConstArray());

                // Similarly for binding
                if (! symbol->getQualifier().hasBinding() && unitSymbol->getQualifier().hasBinding())
                    symbol->getQualifier().layoutBinding = unitSymbol->getQualifier().layoutBinding;

                // Similarly for location
                if (!symbol->getQualifier().hasLocation() && unitSymbol->getQualifier().hasLocation()) {
                    symbol->getQualifier().layoutLocation = unitSymbol->getQualifier().layoutLocation;
                }

                // Update implicit array sizes
                if (symbol->getWritableType().isImplicitlySizedArray() && unitSymbol->getType().isImplicitlySizedArray()) {
                    if (unitSymbol->getType().getImplicitArraySize() > symbol->getType().getImplicitArraySize()){
                        symbol->getWritableType().updateImplicitArraySize(unitSymbol->getType().getImplicitArraySize());
                    }
                }
                else if (symbol->getWritableType().isImplicitlySizedArray() && unitSymbol->getType().isSizedArray()) {
                    if (symbol->getWritableType().getImplicitArraySize() > unitSymbol->getType().getOuterArraySize())
                        error(infoSink, "Implicit size of unsized array doesn't match same symbol among multiple shaders.");
                }
                else if (unitSymbol->getType().isImplicitlySizedArray() && symbol->getWritableType().isSizedArray()) {
                    if (unitSymbol->getType().getImplicitArraySize() > symbol->getWritableType().getOuterArraySize())
                        error(infoSink, "Implicit size of unsized array doesn't match same symbol among multiple shaders.");
                }

                // Update implicit array sizes
                mergeImplicitArraySizes(symbol->getWritableType(), unitSymbol->getType());

                // Check for consistent types/qualification/initializers etc.
                mergeErrorCheck(infoSink, *symbol, *unitSymbol, unitStage);
            }
            // If different symbols, verify they arn't push_constant since there can only be one per stage
            else if (symbol->getQualifier().isPushConstant() && unitSymbol->getQualifier().isPushConstant() && getStage() == unitStage)
                error(infoSink, "Only one push_constant block is allowed per stage");
        }

        // Check conflicts between preset primitives and sizes of I/O variables among multiple geometry shaders
        if (language == EShLangGeometry && unitStage == EShLangGeometry)
        {
            TIntermSymbol* unitSymbol = unitLinkerObjects[unitLinkObj]->getAsSymbolNode();
            if (unitSymbol->isArray() && unitSymbol->getQualifier().storage == EvqVaryingIn && unitSymbol->getQualifier().builtIn == EbvNone)
                if ((unitSymbol->getArraySizes()->isImplicitlySized() &&
                        unitSymbol->getArraySizes()->getImplicitSize() != TQualifier::mapGeometryToSize(getInputPrimitive())) ||
                    (! unitSymbol->getArraySizes()->isImplicitlySized() &&
                        unitSymbol->getArraySizes()->getDimSize(0) != TQualifier::mapGeometryToSize(getInputPrimitive())))
                    error(infoSink, "Not all array sizes match across all geometry shaders in the program");
        }

        if (merge) {
            linkerObjects.push_back(unitLinkerObjects[unitLinkObj]);

            // for anonymous blocks, check that their members don't conflict with other names
            if (unitLinkerObjects[unitLinkObj]->getAsSymbolNode()->getBasicType() == EbtBlock &&
                IsAnonymous(unitLinkerObjects[unitLinkObj]->getAsSymbolNode()->getName())) {
                for (std::size_t linkObj = 0; linkObj < initialNumLinkerObjects; ++linkObj) {
                    TIntermSymbol* symbol = linkerObjects[linkObj]->getAsSymbolNode();
                    TIntermSymbol* unitSymbol = unitLinkerObjects[unitLinkObj]->getAsSymbolNode();
                    assert(symbol && unitSymbol);

                    auto checkName = [this, unitSymbol, &infoSink](const TString& name) {
                        for (unsigned int i = 0; i < unitSymbol->getType().getStruct()->size(); ++i) {
                            if (name == (*unitSymbol->getType().getStruct())[i].type->getFieldName()
                                && !((*unitSymbol->getType().getStruct())[i].type->getQualifier().hasLocation()
                                    || unitSymbol->getType().getQualifier().hasLocation())
                                ) {
                                error(infoSink, "Anonymous member name used for global variable or other anonymous member: ");
                                infoSink.info << (*unitSymbol->getType().getStruct())[i].type->getCompleteString() << "\n";
                            }
                        }
                    };

                    if (isSameInterface(symbol, getStage(), unitSymbol, unitStage)) {
                        checkName(symbol->getName());

                        // check members of other anonymous blocks
                        if (symbol->getBasicType() == EbtBlock && IsAnonymous(symbol->getName())) {
                            for (unsigned int i = 0; i < symbol->getType().getStruct()->size(); ++i) {
                                checkName((*symbol->getType().getStruct())[i].type->getFieldName());
                            }
                        }
                    }
                }
            }
        }
    }
}

// TODO 4.5 link functionality: cull distance array size checking

// Recursively merge the implicit array sizes through the objects' respective type trees.
void TIntermediate::mergeImplicitArraySizes(TType& type, const TType& unitType)
{
    if (type.isUnsizedArray()) {
        if (unitType.isUnsizedArray()) {
            type.updateImplicitArraySize(unitType.getImplicitArraySize());
            if (unitType.isArrayVariablyIndexed())
                type.setArrayVariablyIndexed();
        } else if (unitType.isSizedArray())
            type.changeOuterArraySize(unitType.getOuterArraySize());
    }

    // Type mismatches are caught and reported after this, just be careful for now.
    if (! type.isStruct() || ! unitType.isStruct() || type.getStruct()->size() != unitType.getStruct()->size())
        return;

    for (int i = 0; i < (int)type.getStruct()->size(); ++i)
        mergeImplicitArraySizes(*(*type.getStruct())[i].type, *(*unitType.getStruct())[i].type);
}

//
// Compare two global objects from two compilation units and see if they match
// well enough.  Rules can be different for intra- vs. cross-stage matching.
//
// This function only does one of intra- or cross-stage matching per call.
//
void TIntermediate::mergeErrorCheck(TInfoSink& infoSink, const TIntermSymbol& symbol, const TIntermSymbol& unitSymbol, EShLanguage unitStage)
{
#if !defined(GLSLANG_WEB)
    bool crossStage = getStage() != unitStage;
    bool writeTypeComparison = false;
    bool errorReported = false;
    bool printQualifiers = false;
    bool printPrecision = false;
    bool printType = false;

    // Types have to match
    {
        // but, we make an exception if one is an implicit array and the other is sized
        // or if the array sizes differ because of the extra array dimension on some in/out boundaries
        bool arraysMatch = false;
        if (isIoResizeArray(symbol.getType(), getStage()) || isIoResizeArray(unitSymbol.getType(), unitStage)) {
            // if the arrays have an extra dimension because of the stage.
            // compare dimensions while ignoring the outer dimension
            unsigned int firstDim = isIoResizeArray(symbol.getType(), getStage()) ? 1 : 0;
            unsigned int numDim = symbol.getArraySizes()
                ? symbol.getArraySizes()->getNumDims() : 0;
            unsigned int unitFirstDim = isIoResizeArray(unitSymbol.getType(), unitStage) ? 1 : 0;
            unsigned int unitNumDim = unitSymbol.getArraySizes()
                ? unitSymbol.getArraySizes()->getNumDims() : 0;
            arraysMatch = (numDim - firstDim) == (unitNumDim - unitFirstDim);
            // check that array sizes match as well
            for (unsigned int i = 0; i < (numDim - firstDim) && arraysMatch; i++) {
                if (symbol.getArraySizes()->getDimSize(firstDim + i) !=
                    unitSymbol.getArraySizes()->getDimSize(unitFirstDim + i)) {
                    arraysMatch = false;
                    break;
                }
            }
        }
        else {
            arraysMatch = symbol.getType().sameArrayness(unitSymbol.getType()) ||
                (symbol.getType().isArray() && unitSymbol.getType().isArray() &&
                 (symbol.getType().isImplicitlySizedArray() || unitSymbol.getType().isImplicitlySizedArray() ||
                  symbol.getType().isUnsizedArray() || unitSymbol.getType().isUnsizedArray()));
        }

        int lpidx = -1;
        int rpidx = -1;
        if (!symbol.getType().sameElementType(unitSymbol.getType(), &lpidx, &rpidx)) {
            if (lpidx >= 0 && rpidx >= 0) {
                error(infoSink, "Member names and types must match:", unitStage);
                infoSink.info << "    Block: " << symbol.getType().getTypeName() << "\n";
                infoSink.info << "        " << StageName(getStage()) << " stage: \""
                              << (*symbol.getType().getStruct())[lpidx].type->getCompleteString(true, false, false, true,
                                      (*symbol.getType().getStruct())[lpidx].type->getFieldName()) << "\"\n";
                infoSink.info << "        " << StageName(unitStage) << " stage: \""
                              << (*unitSymbol.getType().getStruct())[rpidx].type->getCompleteString(true, false, false, true,
                                      (*unitSymbol.getType().getStruct())[rpidx].type->getFieldName()) << "\"\n";
                errorReported = true;
            } else if (lpidx >= 0 && rpidx == -1) {
                  TString errmsg = StageName(getStage());
                  errmsg.append(" block member has no corresponding member in ").append(StageName(unitStage)).append(" block:");
                  error(infoSink, errmsg.c_str(), unitStage);
                  infoSink.info << "    " << StageName(getStage()) << " stage: Block: " << symbol.getType().getTypeName() << ", Member: "
                    << (*symbol.getType().getStruct())[lpidx].type->getFieldName() << "\n";
                  infoSink.info << "    " << StageName(unitStage) << " stage: Block: " << unitSymbol.getType().getTypeName() << ", Member: n/a \n";
                  errorReported = true;
            } else if (lpidx == -1 && rpidx >= 0) {
                  TString errmsg = StageName(unitStage);
                  errmsg.append(" block member has no corresponding member in ").append(StageName(getStage())).append(" block:");
                  error(infoSink, errmsg.c_str(), unitStage);
                  infoSink.info << "    " << StageName(unitStage) << " stage: Block: " << unitSymbol.getType().getTypeName() << ", Member: "
                    << (*unitSymbol.getType().getStruct())[rpidx].type->getFieldName() << "\n";
                  infoSink.info << "    " << StageName(getStage()) << " stage: Block: " << symbol.getType().getTypeName() << ", Member: n/a \n";
                  errorReported = true;
            } else {
                  error(infoSink, "Types must match:", unitStage);
                  writeTypeComparison = true;
                  printType = true;
            }
        } else if (!arraysMatch) {
            error(infoSink, "Array sizes must be compatible:", unitStage);
            writeTypeComparison = true;
            printType = true;
        } else if (!symbol.getType().sameTypeParameters(unitSymbol.getType())) {
            error(infoSink, "Type parameters must match:", unitStage);
            writeTypeComparison = true;
            printType = true;
        }
    }

    // Interface block  member-wise layout qualifiers have to match
    if (symbol.getType().getBasicType() == EbtBlock && unitSymbol.getType().getBasicType() == EbtBlock &&
        symbol.getType().getStruct() && unitSymbol.getType().getStruct() &&
        symbol.getType().sameStructType(unitSymbol.getType())) {
        unsigned int li = 0;
        unsigned int ri = 0;
        while (li < symbol.getType().getStruct()->size() && ri < unitSymbol.getType().getStruct()->size()) {
            if ((*symbol.getType().getStruct())[li].type->hiddenMember()) {
                ++li;
                continue;
            }
            if ((*unitSymbol.getType().getStruct())[ri].type->hiddenMember()) {
                ++ri;
                continue;
            }
            const TQualifier& qualifier = (*symbol.getType().getStruct())[li].type->getQualifier();
            const TQualifier & unitQualifier = (*unitSymbol.getType().getStruct())[ri].type->getQualifier();
            bool layoutQualifierError = false;
            if (qualifier.layoutMatrix != unitQualifier.layoutMatrix) {
                error(infoSink, "Interface block member layout matrix qualifier must match:", unitStage);
                layoutQualifierError = true;
            }
            if (qualifier.layoutOffset != unitQualifier.layoutOffset) {
                error(infoSink, "Interface block member layout offset qualifier must match:", unitStage);
                layoutQualifierError = true;
            }
            if (qualifier.layoutAlign != unitQualifier.layoutAlign) {
                error(infoSink, "Interface block member layout align qualifier must match:", unitStage);
                layoutQualifierError = true;
            }
            if (qualifier.layoutLocation != unitQualifier.layoutLocation) {
                error(infoSink, "Interface block member layout location qualifier must match:", unitStage);
                layoutQualifierError = true;
            }
            if (qualifier.layoutComponent != unitQualifier.layoutComponent) {
                error(infoSink, "Interface block member layout component qualifier must match:", unitStage);
                layoutQualifierError = true;
            }
            if (layoutQualifierError) {
                infoSink.info << "    " << StageName(getStage()) << " stage: Block: " << symbol.getType().getTypeName() << ", Member: "
                              << (*symbol.getType().getStruct())[li].type->getFieldName() << " \""
                              << (*symbol.getType().getStruct())[li].type->getCompleteString(true, true, false, false) << "\"\n";
                infoSink.info << "    " << StageName(unitStage) << " stage: Block: " << unitSymbol.getType().getTypeName() << ", Member: "
                              << (*unitSymbol.getType().getStruct())[ri].type->getFieldName() << " \""
                              << (*unitSymbol.getType().getStruct())[ri].type->getCompleteString(true, true, false, false) << "\"\n";
                errorReported = true;
            }
            ++li;
            ++ri;
        }
    }

    bool isInOut = crossStage &&
                   ((symbol.getQualifier().storage == EvqVaryingIn && unitSymbol.getQualifier().storage == EvqVaryingOut) ||
                   (symbol.getQualifier().storage == EvqVaryingOut && unitSymbol.getQualifier().storage == EvqVaryingIn));

    // Qualifiers have to (almost) match
    // Storage...
    if (!isInOut && symbol.getQualifier().storage != unitSymbol.getQualifier().storage) {
        error(infoSink, "Storage qualifiers must match:", unitStage);
        writeTypeComparison = true;
        printQualifiers = true;
    }

    // Uniform and buffer blocks must either both have an instance name, or
    // must both be anonymous. The names don't need to match though.
    if (symbol.getQualifier().isUniformOrBuffer() &&
        (IsAnonymous(symbol.getName()) != IsAnonymous(unitSymbol.getName()))) {
        error(infoSink, "Matched Uniform or Storage blocks must all be anonymous,"
                        " or all be named:", unitStage);
        writeTypeComparison = true;
    }

    if (symbol.getQualifier().storage == unitSymbol.getQualifier().storage &&
        (IsAnonymous(symbol.getName()) != IsAnonymous(unitSymbol.getName()) ||
         (!IsAnonymous(symbol.getName()) && symbol.getName() != unitSymbol.getName()))) {
        warn(infoSink, "Matched shader interfaces are using different instance names.", unitStage);
        writeTypeComparison = true;
    }

    // Precision...
    if (!isInOut && symbol.getQualifier().precision != unitSymbol.getQualifier().precision) {
        error(infoSink, "Precision qualifiers must match:", unitStage);
        writeTypeComparison = true;
        printPrecision = true;
    }

    // Invariance...
    if (! crossStage && symbol.getQualifier().invariant != unitSymbol.getQualifier().invariant) {
        error(infoSink, "Presence of invariant qualifier must match:", unitStage);
        writeTypeComparison = true;
        printQualifiers = true;
    }

    // Precise...
    if (! crossStage && symbol.getQualifier().isNoContraction() != unitSymbol.getQualifier().isNoContraction()) {
        error(infoSink, "Presence of precise qualifier must match:", unitStage);
        writeTypeComparison = true;
        printPrecision = true;
    }

    // Auxiliary and interpolation...
    // "interpolation qualification (e.g., flat) and auxiliary qualification (e.g. centroid) may differ.
    //  These mismatches are allowed between any pair of stages ...
    //  those provided in the fragment shader supersede those provided in previous stages."
    if (!crossStage &&
        (symbol.getQualifier().centroid  != unitSymbol.getQualifier().centroid ||
        symbol.getQualifier().smooth    != unitSymbol.getQualifier().smooth ||
        symbol.getQualifier().flat      != unitSymbol.getQualifier().flat ||
        symbol.getQualifier().isSample()!= unitSymbol.getQualifier().isSample() ||
        symbol.getQualifier().isPatch() != unitSymbol.getQualifier().isPatch() ||
        symbol.getQualifier().isNonPerspective() != unitSymbol.getQualifier().isNonPerspective())) {
        error(infoSink, "Interpolation and auxiliary storage qualifiers must match:", unitStage);
        writeTypeComparison = true;
        printQualifiers = true;
    }

    // Memory...
    bool memoryQualifierError = false;
    if (symbol.getQualifier().coherent != unitSymbol.getQualifier().coherent) {
        error(infoSink, "Memory coherent qualifier must match:", unitStage);
        memoryQualifierError = true;
    }
    if (symbol.getQualifier().devicecoherent != unitSymbol.getQualifier().devicecoherent) {
        error(infoSink, "Memory devicecoherent qualifier must match:", unitStage);
        memoryQualifierError = true;
    }
    if (symbol.getQualifier().queuefamilycoherent != unitSymbol.getQualifier().queuefamilycoherent) {
        error(infoSink, "Memory queuefamilycoherent qualifier must match:", unitStage);
        memoryQualifierError = true;
    }
    if (symbol.getQualifier().workgroupcoherent != unitSymbol.getQualifier().workgroupcoherent) {
        error(infoSink, "Memory workgroupcoherent qualifier must match:", unitStage);
        memoryQualifierError = true;
    }
    if (symbol.getQualifier().subgroupcoherent != unitSymbol.getQualifier().subgroupcoherent) {
        error(infoSink, "Memory subgroupcoherent qualifier must match:", unitStage);
        memoryQualifierError = true;
    }
    if (symbol.getQualifier().shadercallcoherent != unitSymbol.getQualifier().shadercallcoherent) {
        error(infoSink, "Memory shadercallcoherent qualifier must match:", unitStage);
        memoryQualifierError = true;
    }
    if (symbol.getQualifier().nonprivate != unitSymbol.getQualifier().nonprivate) {
        error(infoSink, "Memory nonprivate qualifier must match:", unitStage);
        memoryQualifierError = true;
    }
    if (symbol.getQualifier().volatil != unitSymbol.getQualifier().volatil) {
        error(infoSink, "Memory volatil qualifier must match:", unitStage);
        memoryQualifierError = true;
    }
    if (symbol.getQualifier().restrict != unitSymbol.getQualifier().restrict) {
        error(infoSink, "Memory restrict qualifier must match:", unitStage);
        memoryQualifierError = true;
    }
    if (symbol.getQualifier().readonly != unitSymbol.getQualifier().readonly) {
        error(infoSink, "Memory readonly qualifier must match:", unitStage);
        memoryQualifierError = true;
    }
    if (symbol.getQualifier().writeonly != unitSymbol.getQualifier().writeonly) {
        error(infoSink, "Memory writeonly qualifier must match:", unitStage);
        memoryQualifierError = true;
    }
    if (memoryQualifierError) {
          writeTypeComparison = true;
          printQualifiers = true;
    }

    // Layouts...
    // TODO: 4.4 enhanced layouts: Generalize to include offset/align: current spec
    //       requires separate user-supplied offset from actual computed offset, but
    //       current implementation only has one offset.
    bool layoutQualifierError = false;
    if (symbol.getQualifier().layoutMatrix != unitSymbol.getQualifier().layoutMatrix) {
        error(infoSink, "Layout matrix qualifier must match:", unitStage);
        layoutQualifierError = true;
    }
    if (symbol.getQualifier().layoutPacking != unitSymbol.getQualifier().layoutPacking) {
        error(infoSink, "Layout packing qualifier must match:", unitStage);
        layoutQualifierError = true;
    }
    if (symbol.getQualifier().hasLocation() && unitSymbol.getQualifier().hasLocation() && symbol.getQualifier().layoutLocation != unitSymbol.getQualifier().layoutLocation) {
        error(infoSink, "Layout location qualifier must match:", unitStage);
        layoutQualifierError = true;
    }
    if (symbol.getQualifier().layoutComponent != unitSymbol.getQualifier().layoutComponent) {
        error(infoSink, "Layout component qualifier must match:", unitStage);
        layoutQualifierError = true;
    }
    if (symbol.getQualifier().layoutIndex != unitSymbol.getQualifier().layoutIndex) {
        error(infoSink, "Layout index qualifier must match:", unitStage);
        layoutQualifierError = true;
    }
    if (symbol.getQualifier().hasBinding() && unitSymbol.getQualifier().hasBinding() && symbol.getQualifier().layoutBinding != unitSymbol.getQualifier().layoutBinding) {
        error(infoSink, "Layout binding qualifier must match:", unitStage);
        layoutQualifierError = true;
    }
    if (symbol.getQualifier().hasBinding() && (symbol.getQualifier().layoutOffset != unitSymbol.getQualifier().layoutOffset)) {
        error(infoSink, "Layout offset qualifier must match:", unitStage);
        layoutQualifierError = true;
    }
    if (layoutQualifierError) {
        writeTypeComparison = true;
        printQualifiers = true;
    }

    // Initializers have to match, if both are present, and if we don't already know the types don't match
    if (! writeTypeComparison && ! errorReported) {
        if (! symbol.getConstArray().empty() && ! unitSymbol.getConstArray().empty()) {
            if (symbol.getConstArray() != unitSymbol.getConstArray()) {
                error(infoSink, "Initializers must match:", unitStage);
                infoSink.info << "    " << symbol.getName() << "\n";
            }
        }
    }

    if (writeTypeComparison) {
        if (symbol.getType().getBasicType() == EbtBlock && unitSymbol.getType().getBasicType() == EbtBlock &&
            symbol.getType().getStruct() && unitSymbol.getType().getStruct()) {
          if (printType) {
            infoSink.info << "    " << StageName(getStage()) << " stage: \"" << symbol.getType().getCompleteString(true, printQualifiers, printPrecision,
                                                    printType, symbol.getName(), symbol.getType().getTypeName()) << "\"\n";
            infoSink.info << "    " << StageName(unitStage) << " stage: \"" << unitSymbol.getType().getCompleteString(true, printQualifiers, printPrecision,
                                                    printType, unitSymbol.getName(), unitSymbol.getType().getTypeName()) << "\"\n";
          } else {
            infoSink.info << "    " << StageName(getStage()) << " stage: Block: " << symbol.getType().getTypeName() << " Instance: " << symbol.getName()
              << ": \"" << symbol.getType().getCompleteString(true, printQualifiers, printPrecision, printType) << "\"\n";
            infoSink.info << "    " << StageName(unitStage) << " stage: Block: " << unitSymbol.getType().getTypeName() << " Instance: " << unitSymbol.getName()
              << ": \"" << unitSymbol.getType().getCompleteString(true, printQualifiers, printPrecision, printType) << "\"\n";
          }
        } else {
          if (printType) {
            infoSink.info << "    " << StageName(getStage()) << " stage: \""
              << symbol.getType().getCompleteString(true, printQualifiers, printPrecision, printType, symbol.getName()) << "\"\n";
            infoSink.info << "    " << StageName(unitStage) << " stage: \""
              << unitSymbol.getType().getCompleteString(true, printQualifiers, printPrecision, printType, unitSymbol.getName()) << "\"\n";
          } else {
            infoSink.info << "    " << StageName(getStage()) << " stage: " << symbol.getName() << " \""
              << symbol.getType().getCompleteString(true, printQualifiers, printPrecision, printType) << "\"\n";
            infoSink.info << "    " << StageName(unitStage) << " stage: " << unitSymbol.getName() << " \""
              << unitSymbol.getType().getCompleteString(true, printQualifiers, printPrecision, printType) << "\"\n";
          }
        }
    }
#endif
}

void TIntermediate::sharedBlockCheck(TInfoSink& infoSink)
{
    bool has_shared_block = false;
    bool has_shared_non_block = false;
    TIntermSequence& linkObjects = findLinkerObjects()->getSequence();
    for (size_t i = 0; i < linkObjects.size(); ++i) {
        const TType& type = linkObjects[i]->getAsTyped()->getType();
        const TQualifier& qualifier = type.getQualifier();
        if (qualifier.storage == glslang::EvqShared) {
            if (type.getBasicType() == glslang::EbtBlock)
                has_shared_block = true;
            else
                has_shared_non_block = true;
        }
    }
    if (has_shared_block && has_shared_non_block)
        error(infoSink, "cannot mix use of shared variables inside and outside blocks");
}

//
// Do final link-time error checking of a complete (merged) intermediate representation.
// (Much error checking was done during merging).
//
// Also, lock in defaults of things not set, including array sizes.
//
void TIntermediate::finalCheck(TInfoSink& infoSink, bool keepUncalled)
{
    if (getTreeRoot() == nullptr)
        return;

    if (numEntryPoints < 1) {
        if (getSource() == EShSourceGlsl)
            error(infoSink, "Missing entry point: Each stage requires one entry point");
        else
            warn(infoSink, "Entry point not found");
    }

    // recursion and missing body checking
    checkCallGraphCycles(infoSink);
    checkCallGraphBodies(infoSink, keepUncalled);

    // overlap/alias/missing I/O, etc.
    inOutLocationCheck(infoSink);

#ifndef GLSLANG_WEB
    if (getNumPushConstants() > 1)
        error(infoSink, "Only one push_constant block is allowed per stage");

    // invocations
    if (invocations == TQualifier::layoutNotSet)
        invocations = 1;

    if (inIoAccessed("gl_ClipDistance") && inIoAccessed("gl_ClipVertex"))
        error(infoSink, "Can only use one of gl_ClipDistance or gl_ClipVertex (gl_ClipDistance is preferred)");
    if (inIoAccessed("gl_CullDistance") && inIoAccessed("gl_ClipVertex"))
        error(infoSink, "Can only use one of gl_CullDistance or gl_ClipVertex (gl_ClipDistance is preferred)");

    if (userOutputUsed() && (inIoAccessed("gl_FragColor") || inIoAccessed("gl_FragData")))
        error(infoSink, "Cannot use gl_FragColor or gl_FragData when using user-defined outputs");
    if (inIoAccessed("gl_FragColor") && inIoAccessed("gl_FragData"))
        error(infoSink, "Cannot use both gl_FragColor and gl_FragData");

    for (size_t b = 0; b < xfbBuffers.size(); ++b) {
        if (xfbBuffers[b].contains64BitType)
            RoundToPow2(xfbBuffers[b].implicitStride, 8);
        else if (xfbBuffers[b].contains32BitType)
            RoundToPow2(xfbBuffers[b].implicitStride, 4);
        else if (xfbBuffers[b].contains16BitType)
            RoundToPow2(xfbBuffers[b].implicitStride, 2);

        // "It is a compile-time or link-time error to have
        // any xfb_offset that overflows xfb_stride, whether stated on declarations before or after the xfb_stride, or
        // in different compilation units. While xfb_stride can be declared multiple times for the same buffer, it is a
        // compile-time or link-time error to have different values specified for the stride for the same buffer."
        if (xfbBuffers[b].stride != TQualifier::layoutXfbStrideEnd && xfbBuffers[b].implicitStride > xfbBuffers[b].stride) {
            error(infoSink, "xfb_stride is too small to hold all buffer entries:");
            infoSink.info.prefix(EPrefixError);
            infoSink.info << "    xfb_buffer " << (unsigned int)b << ", xfb_stride " << xfbBuffers[b].stride << ", minimum stride needed: " << xfbBuffers[b].implicitStride << "\n";
        }
        if (xfbBuffers[b].stride == TQualifier::layoutXfbStrideEnd)
            xfbBuffers[b].stride = xfbBuffers[b].implicitStride;

        // "If the buffer is capturing any
        // outputs with double-precision or 64-bit integer components, the stride must be a multiple of 8, otherwise it must be a
        // multiple of 4, or a compile-time or link-time error results."
        if (xfbBuffers[b].contains64BitType && ! IsMultipleOfPow2(xfbBuffers[b].stride, 8)) {
            error(infoSink, "xfb_stride must be multiple of 8 for buffer holding a double or 64-bit integer:");
            infoSink.info.prefix(EPrefixError);
            infoSink.info << "    xfb_buffer " << (unsigned int)b << ", xfb_stride " << xfbBuffers[b].stride << "\n";
        } else if (xfbBuffers[b].contains32BitType && ! IsMultipleOfPow2(xfbBuffers[b].stride, 4)) {
            error(infoSink, "xfb_stride must be multiple of 4:");
            infoSink.info.prefix(EPrefixError);
            infoSink.info << "    xfb_buffer " << (unsigned int)b << ", xfb_stride " << xfbBuffers[b].stride << "\n";
        }
        // "If the buffer is capturing any
        // outputs with half-precision or 16-bit integer components, the stride must be a multiple of 2"
        else if (xfbBuffers[b].contains16BitType && ! IsMultipleOfPow2(xfbBuffers[b].stride, 2)) {
            error(infoSink, "xfb_stride must be multiple of 2 for buffer holding a half float or 16-bit integer:");
            infoSink.info.prefix(EPrefixError);
            infoSink.info << "    xfb_buffer " << (unsigned int)b << ", xfb_stride " << xfbBuffers[b].stride << "\n";
        }

        // "The resulting stride (implicit or explicit), when divided by 4, must be less than or equal to the
        // implementation-dependent constant gl_MaxTransformFeedbackInterleavedComponents."
        if (xfbBuffers[b].stride > (unsigned int)(4 * resources->maxTransformFeedbackInterleavedComponents)) {
            error(infoSink, "xfb_stride is too large:");
            infoSink.info.prefix(EPrefixError);
            infoSink.info << "    xfb_buffer " << (unsigned int)b << ", components (1/4 stride) needed are " << xfbBuffers[b].stride/4 << ", gl_MaxTransformFeedbackInterleavedComponents is " << resources->maxTransformFeedbackInterleavedComponents << "\n";
        }
    }

    switch (language) {
    case EShLangVertex:
        break;
    case EShLangTessControl:
        if (vertices == TQualifier::layoutNotSet)
            error(infoSink, "At least one shader must specify an output layout(vertices=...)");
        break;
    case EShLangTessEvaluation:
        if (getSource() == EShSourceGlsl) {
            if (inputPrimitive == ElgNone)
                error(infoSink, "At least one shader must specify an input layout primitive");
            if (vertexSpacing == EvsNone)
                vertexSpacing = EvsEqual;
            if (vertexOrder == EvoNone)
                vertexOrder = EvoCcw;
        }
        break;
    case EShLangGeometry:
        if (inputPrimitive == ElgNone)
            error(infoSink, "At least one shader must specify an input layout primitive");
        if (outputPrimitive == ElgNone)
            error(infoSink, "At least one shader must specify an output layout primitive");
        if (vertices == TQualifier::layoutNotSet)
            error(infoSink, "At least one shader must specify a layout(max_vertices = value)");
        break;
    case EShLangFragment:
        // for GL_ARB_post_depth_coverage, EarlyFragmentTest is set automatically in
        // ParseHelper.cpp. So if we reach here, this must be GL_EXT_post_depth_coverage
        // requiring explicit early_fragment_tests
        if (getPostDepthCoverage() && !getEarlyFragmentTests())
            error(infoSink, "post_depth_coverage requires early_fragment_tests");
        break;
    case EShLangCompute:
        sharedBlockCheck(infoSink);
        break;
    case EShLangRayGen:
    case EShLangIntersect:
    case EShLangAnyHit:
    case EShLangClosestHit:
    case EShLangMiss:
    case EShLangCallable:
        if (numShaderRecordBlocks > 1)
            error(infoSink, "Only one shaderRecordNV buffer block is allowed per stage");
        break;
    case EShLangMesh:
        // NV_mesh_shader doesn't allow use of both single-view and per-view builtins.
        if (inIoAccessed("gl_Position") && inIoAccessed("gl_PositionPerViewNV"))
            error(infoSink, "Can only use one of gl_Position or gl_PositionPerViewNV");
        if (inIoAccessed("gl_ClipDistance") && inIoAccessed("gl_ClipDistancePerViewNV"))
            error(infoSink, "Can only use one of gl_ClipDistance or gl_ClipDistancePerViewNV");
        if (inIoAccessed("gl_CullDistance") && inIoAccessed("gl_CullDistancePerViewNV"))
            error(infoSink, "Can only use one of gl_CullDistance or gl_CullDistancePerViewNV");
        if (inIoAccessed("gl_Layer") && inIoAccessed("gl_LayerPerViewNV"))
            error(infoSink, "Can only use one of gl_Layer or gl_LayerPerViewNV");
        if (inIoAccessed("gl_ViewportMask") && inIoAccessed("gl_ViewportMaskPerViewNV"))
            error(infoSink, "Can only use one of gl_ViewportMask or gl_ViewportMaskPerViewNV");
        if (outputPrimitive == ElgNone)
            error(infoSink, "At least one shader must specify an output layout primitive");
        if (vertices == TQualifier::layoutNotSet)
            error(infoSink, "At least one shader must specify a layout(max_vertices = value)");
        if (primitives == TQualifier::layoutNotSet)
            error(infoSink, "At least one shader must specify a layout(max_primitives = value)");
        // fall through
    case EShLangTask:
        if (numTaskNVBlocks > 1)
            error(infoSink, "Only one taskNV interface block is allowed per shader");
        if (numTaskEXTPayloads > 1)
            error(infoSink, "Only single variable of type taskPayloadSharedEXT is allowed per shader");
        sharedBlockCheck(infoSink);
        break;
    default:
        error(infoSink, "Unknown Stage.");
        break;
    }

    // Process the tree for any node-specific work.
    class TFinalLinkTraverser : public TIntermTraverser {
    public:
        TFinalLinkTraverser() { }
        virtual ~TFinalLinkTraverser() { }

        virtual void visitSymbol(TIntermSymbol* symbol)
        {
            // Implicitly size arrays.
            // If an unsized array is left as unsized, it effectively
            // becomes run-time sized.
            symbol->getWritableType().adoptImplicitArraySizes(false);
        }
    } finalLinkTraverser;

    treeRoot->traverse(&finalLinkTraverser);
#endif
}

//
// See if the call graph contains any static recursion, which is disallowed
// by the specification.
//
void TIntermediate::checkCallGraphCycles(TInfoSink& infoSink)
{
    // Clear fields we'll use for this.
    for (TGraph::iterator call = callGraph.begin(); call != callGraph.end(); ++call) {
        call->visited = false;
        call->currentPath = false;
        call->errorGiven = false;
    }

    //
    // Loop, looking for a new connected subgraph.  One subgraph is handled per loop iteration.
    //

    TCall* newRoot;
    do {
        // See if we have unvisited parts of the graph.
        newRoot = nullptr;
        for (TGraph::iterator call = callGraph.begin(); call != callGraph.end(); ++call) {
            if (! call->visited) {
                newRoot = &(*call);
                break;
            }
        }

        // If not, we are done.
        if (! newRoot)
            break;

        // Otherwise, we found a new subgraph, process it:
        // See what all can be reached by this new root, and if any of
        // that is recursive.  This is done by depth-first traversals, seeing
        // if a new call is found that was already in the currentPath (a back edge),
        // thereby detecting recursion.
        std::list<TCall*> stack;
        newRoot->currentPath = true; // currentPath will be true iff it is on the stack
        stack.push_back(newRoot);
        while (! stack.empty()) {
            // get a caller
            TCall* call = stack.back();

            // Add to the stack just one callee.
            // This algorithm always terminates, because only !visited and !currentPath causes a push
            // and all pushes change currentPath to true, and all pops change visited to true.
            TGraph::iterator child = callGraph.begin();
            for (; child != callGraph.end(); ++child) {

                // If we already visited this node, its whole subgraph has already been processed, so skip it.
                if (child->visited)
                    continue;

                if (call->callee == child->caller) {
                    if (child->currentPath) {
                        // Then, we found a back edge
                        if (! child->errorGiven) {
                            error(infoSink, "Recursion detected:");
                            infoSink.info << "    " << call->callee << " calling " << child->callee << "\n";
                            child->errorGiven = true;
                            recursive = true;
                        }
                    } else {
                        child->currentPath = true;
                        stack.push_back(&(*child));
                        break;
                    }
                }
            }
            if (child == callGraph.end()) {
                // no more callees, we bottomed out, never look at this node again
                stack.back()->currentPath = false;
                stack.back()->visited = true;
                stack.pop_back();
            }
        }  // end while, meaning nothing left to process in this subtree

    } while (newRoot);  // redundant loop check; should always exit via the 'break' above
}

//
// See which functions are reachable from the entry point and which have bodies.
// Reachable ones with missing bodies are errors.
// Unreachable bodies are dead code.
//
void TIntermediate::checkCallGraphBodies(TInfoSink& infoSink, bool keepUncalled)
{
    // Clear fields we'll use for this.
    for (TGraph::iterator call = callGraph.begin(); call != callGraph.end(); ++call) {
        call->visited = false;
        call->calleeBodyPosition = -1;
    }

    // The top level of the AST includes function definitions (bodies).
    // Compare these to function calls in the call graph.
    // We'll end up knowing which have bodies, and if so,
    // how to map the call-graph node to the location in the AST.
    TIntermSequence &functionSequence = getTreeRoot()->getAsAggregate()->getSequence();
    std::vector<bool> reachable(functionSequence.size(), true); // so that non-functions are reachable
    for (int f = 0; f < (int)functionSequence.size(); ++f) {
        glslang::TIntermAggregate* node = functionSequence[f]->getAsAggregate();
        if (node && (node->getOp() == glslang::EOpFunction)) {
            if (node->getName().compare(getEntryPointMangledName().c_str()) != 0)
                reachable[f] = false; // so that function bodies are unreachable, until proven otherwise
            for (TGraph::iterator call = callGraph.begin(); call != callGraph.end(); ++call) {
                if (call->callee == node->getName())
                    call->calleeBodyPosition = f;
            }
        }
    }

    // Start call-graph traversal by visiting the entry point nodes.
    for (TGraph::iterator call = callGraph.begin(); call != callGraph.end(); ++call) {
        if (call->caller.compare(getEntryPointMangledName().c_str()) == 0)
            call->visited = true;
    }

    // Propagate 'visited' through the call-graph to every part of the graph it
    // can reach (seeded with the entry-point setting above).
    bool changed;
    do {
        changed = false;
        for (auto call1 = callGraph.begin(); call1 != callGraph.end(); ++call1) {
            if (call1->visited) {
                for (TGraph::iterator call2 = callGraph.begin(); call2 != callGraph.end(); ++call2) {
                    if (! call2->visited) {
                        if (call1->callee == call2->caller) {
                            changed = true;
                            call2->visited = true;
                        }
                    }
                }
            }
        }
    } while (changed);

    // Any call-graph node set to visited but without a callee body is an error.
    for (TGraph::iterator call = callGraph.begin(); call != callGraph.end(); ++call) {
        if (call->visited) {
            if (call->calleeBodyPosition == -1) {
                error(infoSink, "No function definition (body) found: ");
                infoSink.info << "    " << call->callee << "\n";
            } else
                reachable[call->calleeBodyPosition] = true;
        }
    }

    // Bodies in the AST not reached by the call graph are dead;
    // clear them out, since they can't be reached and also can't
    // be translated further due to possibility of being ill defined.
    if (! keepUncalled) {
        for (int f = 0; f < (int)functionSequence.size(); ++f) {
            if (! reachable[f])
            {
                resetTopLevelUncalledStatus(functionSequence[f]->getAsAggregate()->getName());
                functionSequence[f] = nullptr;
            }
        }
        functionSequence.erase(std::remove(functionSequence.begin(), functionSequence.end(), nullptr), functionSequence.end());
    }
}

//
// Satisfy rules for location qualifiers on inputs and outputs
//
void TIntermediate::inOutLocationCheck(TInfoSink& infoSink)
{
    // ES 3.0 requires all outputs to have location qualifiers if there is more than one output
    bool fragOutWithNoLocation = false;
    int numFragOut = 0;

    // TODO: linker functionality: location collision checking

    TIntermSequence& linkObjects = findLinkerObjects()->getSequence();
    for (size_t i = 0; i < linkObjects.size(); ++i) {
        const TType& type = linkObjects[i]->getAsTyped()->getType();
        const TQualifier& qualifier = type.getQualifier();
        if (language == EShLangFragment) {
            if (qualifier.storage == EvqVaryingOut && qualifier.builtIn == EbvNone) {
                ++numFragOut;
                if (!qualifier.hasAnyLocation())
                    fragOutWithNoLocation = true;
            }
        }
    }

    if (isEsProfile()) {
        if (numFragOut > 1 && fragOutWithNoLocation)
            error(infoSink, "when more than one fragment shader output, all must have location qualifiers");
    }
}

TIntermAggregate* TIntermediate::findLinkerObjects() const
{
    // Get the top-level globals
    TIntermSequence& globals = treeRoot->getAsAggregate()->getSequence();

    // Get the last member of the sequences, expected to be the linker-object lists
    assert(globals.back()->getAsAggregate()->getOp() == EOpLinkerObjects);

    return globals.back()->getAsAggregate();
}

// See if a variable was both a user-declared output and used.
// Note: the spec discusses writing to one, but this looks at read or write, which
// is more useful, and perhaps the spec should be changed to reflect that.
bool TIntermediate::userOutputUsed() const
{
    const TIntermSequence& linkerObjects = findLinkerObjects()->getSequence();

    bool found = false;
    for (size_t i = 0; i < linkerObjects.size(); ++i) {
        const TIntermSymbol& symbolNode = *linkerObjects[i]->getAsSymbolNode();
        if (symbolNode.getQualifier().storage == EvqVaryingOut &&
            symbolNode.getName().compare(0, 3, "gl_") != 0 &&
            inIoAccessed(symbolNode.getName())) {
            found = true;
            break;
        }
    }

    return found;
}

// Accumulate locations used for inputs, outputs, and uniforms, payload, callable data, and tileImageEXT
// and check for collisions as the accumulation is done.
//
// Returns < 0 if no collision, >= 0 if collision and the value returned is a colliding value.
//
// typeCollision is set to true if there is no direct collision, but the types in the same location
// are different.
//
int TIntermediate::addUsedLocation(const TQualifier& qualifier, const TType& type, bool& typeCollision)
{
    typeCollision = false;

    int set;
    int setRT;
    if (qualifier.isPipeInput())
        set = 0;
    else if (qualifier.isPipeOutput())
        set = 1;
    else if (qualifier.storage == EvqUniform)
        set = 2;
    else if (qualifier.storage == EvqBuffer)
        set = 3;
    else if (qualifier.storage == EvqTileImageEXT)
        set = 4;
    else if (qualifier.isAnyPayload())
        setRT = 0;
    else if (qualifier.isAnyCallable())
        setRT = 1;
    else if (qualifier.isHitObjectAttrNV())
        setRT = 2;
    else
        return -1;

    int size;
    if (qualifier.isAnyPayload() || qualifier.isAnyCallable()) {
        size = 1;
    } else if (qualifier.isUniformOrBuffer() || qualifier.isTaskMemory()) {
        if (type.isSizedArray())
            size = type.getCumulativeArraySize();
        else
            size = 1;
    } else {
        // Strip off the outer array dimension for those having an extra one.
        if (type.isArray() && qualifier.isArrayedIo(language)) {
            TType elementType(type, 0);
            size = computeTypeLocationSize(elementType, language);
        } else
            size = computeTypeLocationSize(type, language);
    }

    // Locations, and components within locations.
    //
    // Almost always, dealing with components means a single location is involved.
    // The exception is a dvec3. From the spec:
    //
    // "A dvec3 will consume all four components of the first location and components 0 and 1 of
    // the second location. This leaves components 2 and 3 available for other component-qualified
    // declarations."
    //
    // That means, without ever mentioning a component, a component range
    // for a different location gets specified, if it's not a vertex shader input. (!)
    // (A vertex shader input will show using only one location, even for a dvec3/4.)
    //
    // So, for the case of dvec3, we need two independent ioRanges.
    //
    // For raytracing IO (payloads and callabledata) each declaration occupies a single
    // slot irrespective of type.
    int collision = -1; // no collision
#ifndef GLSLANG_WEB
    if (qualifier.isAnyPayload() || qualifier.isAnyCallable() || qualifier.isHitObjectAttrNV()) {
        TRange range(qualifier.layoutLocation, qualifier.layoutLocation);
        collision = checkLocationRT(setRT, qualifier.layoutLocation);
        if (collision < 0)
            usedIoRT[setRT].push_back(range);
    } else if (size == 2 && type.getBasicType() == EbtDouble && type.getVectorSize() == 3 &&
        (qualifier.isPipeInput() || qualifier.isPipeOutput())) {
        // Dealing with dvec3 in/out split across two locations.
        // Need two io-ranges.
        // The case where the dvec3 doesn't start at component 0 was previously caught as overflow.

        // First range:
        TRange locationRange(qualifier.layoutLocation, qualifier.layoutLocation);
        TRange componentRange(0, 3);
        TIoRange range(locationRange, componentRange, type.getBasicType(), 0);

        // check for collisions
        collision = checkLocationRange(set, range, type, typeCollision);
        if (collision < 0) {
            usedIo[set].push_back(range);

            // Second range:
            TRange locationRange2(qualifier.layoutLocation + 1, qualifier.layoutLocation + 1);
            TRange componentRange2(0, 1);
            TIoRange range2(locationRange2, componentRange2, type.getBasicType(), 0);

            // check for collisions
            collision = checkLocationRange(set, range2, type, typeCollision);
            if (collision < 0)
                usedIo[set].push_back(range2);
        }
    } else
#endif
    {
        // Not a dvec3 in/out split across two locations, generic path.
        // Need a single IO-range block.

        TRange locationRange(qualifier.layoutLocation, qualifier.layoutLocation + size - 1);
        TRange componentRange(0, 3);
        if (qualifier.hasComponent() || type.getVectorSize() > 0) {
            int consumedComponents = type.getVectorSize() * (type.getBasicType() == EbtDouble ? 2 : 1);
            if (qualifier.hasComponent())
                componentRange.start = qualifier.layoutComponent;
            componentRange.last  = componentRange.start + consumedComponents - 1;
        }

        // combine location and component ranges
        TBasicType basicTy = type.getBasicType();
        if (basicTy == EbtSampler && type.getSampler().isAttachmentEXT())
            basicTy = type.getSampler().type;
        TIoRange range(locationRange, componentRange, basicTy, qualifier.hasIndex() ? qualifier.getIndex() : 0);

        // check for collisions, except for vertex inputs on desktop targeting OpenGL
        if (! (!isEsProfile() && language == EShLangVertex && qualifier.isPipeInput()) || spvVersion.vulkan > 0)
            collision = checkLocationRange(set, range, type, typeCollision);

        if (collision < 0)
            usedIo[set].push_back(range);
    }

    return collision;
}

// Compare a new (the passed in) 'range' against the existing set, and see
// if there are any collisions.
//
// Returns < 0 if no collision, >= 0 if collision and the value returned is a colliding value.
//
int TIntermediate::checkLocationRange(int set, const TIoRange& range, const TType& type, bool& typeCollision)
{
    for (size_t r = 0; r < usedIo[set].size(); ++r) {
        if (range.overlap(usedIo[set][r])) {
            // there is a collision; pick one
            return std::max(range.location.start, usedIo[set][r].location.start);
        } else if (range.location.overlap(usedIo[set][r].location) && type.getBasicType() != usedIo[set][r].basicType) {
            // aliased-type mismatch
            typeCollision = true;
            return std::max(range.location.start, usedIo[set][r].location.start);
        }
    }

    // check typeCollision between tileImageEXT and out
    if (set == 4 || set == 1) {
      // if the set is "tileImageEXT", check against "out" and vice versa
      int againstSet = (set == 4) ? 1 : 4;
      for (size_t r = 0; r < usedIo[againstSet].size(); ++r) {
        if (range.location.overlap(usedIo[againstSet][r].location) && type.getBasicType() != usedIo[againstSet][r].basicType) {
            // aliased-type mismatch
            typeCollision = true;
            return std::max(range.location.start, usedIo[againstSet][r].location.start);
        }
      }
    }

    return -1; // no collision
}

int TIntermediate::checkLocationRT(int set, int location) {
    TRange range(location, location);
    for (size_t r = 0; r < usedIoRT[set].size(); ++r) {
        if (range.overlap(usedIoRT[set][r])) {
            return range.start;
        }
    }
    return -1; // no collision
}

// Accumulate bindings and offsets, and check for collisions
// as the accumulation is done.
//
// Returns < 0 if no collision, >= 0 if collision and the value returned is a colliding value.
//
int TIntermediate::addUsedOffsets(int binding, int offset, int numOffsets)
{
    TRange bindingRange(binding, binding);
    TRange offsetRange(offset, offset + numOffsets - 1);
    TOffsetRange range(bindingRange, offsetRange);

    // check for collisions, except for vertex inputs on desktop
    for (size_t r = 0; r < usedAtomics.size(); ++r) {
        if (range.overlap(usedAtomics[r])) {
            // there is a collision; pick one
            return std::max(offset, usedAtomics[r].offset.start);
        }
    }

    usedAtomics.push_back(range);

    return -1; // no collision
}

// Accumulate used constant_id values.
//
// Return false is one was already used.
bool TIntermediate::addUsedConstantId(int id)
{
    if (usedConstantId.find(id) != usedConstantId.end())
        return false;

    usedConstantId.insert(id);

    return true;
}

// Recursively figure out how many locations are used up by an input or output type.
// Return the size of type, as measured by "locations".
int TIntermediate::computeTypeLocationSize(const TType& type, EShLanguage stage)
{
    // "If the declared input is an array of size n and each element takes m locations, it will be assigned m * n
    // consecutive locations..."
    if (type.isArray()) {
        // TODO: perf: this can be flattened by using getCumulativeArraySize(), and a deref that discards all arrayness
        // TODO: are there valid cases of having an unsized array with a location?  If so, running this code too early.
        TType elementType(type, 0);
        if (type.isSizedArray() && !type.getQualifier().isPerView())
            return type.getOuterArraySize() * computeTypeLocationSize(elementType, stage);
        else {
#ifndef GLSLANG_WEB
            // unset perViewNV attributes for arrayed per-view outputs: "perviewNV vec4 v[MAX_VIEWS][3];"
            elementType.getQualifier().perViewNV = false;
#endif
            return computeTypeLocationSize(elementType, stage);
        }
    }

    // "The locations consumed by block and structure members are determined by applying the rules above
    // recursively..."
    if (type.isStruct()) {
        int size = 0;
        for (int member = 0; member < (int)type.getStruct()->size(); ++member) {
            TType memberType(type, member);
            size += computeTypeLocationSize(memberType, stage);
        }
        return size;
    }

    // ES: "If a shader input is any scalar or vector type, it will consume a single location."

    // Desktop: "If a vertex shader input is any scalar or vector type, it will consume a single location. If a non-vertex
    // shader input is a scalar or vector type other than dvec3 or dvec4, it will consume a single location, while
    // types dvec3 or dvec4 will consume two consecutive locations. Inputs of type double and dvec2 will
    // consume only a single location, in all stages."
    if (type.isScalar())
        return 1;
    if (type.isVector()) {
        if (stage == EShLangVertex && type.getQualifier().isPipeInput())
            return 1;
        if (type.getBasicType() == EbtDouble && type.getVectorSize() > 2)
            return 2;
        else
            return 1;
    }

    // "If the declared input is an n x m single- or double-precision matrix, ...
    // The number of locations assigned for each matrix will be the same as
    // for an n-element array of m-component vectors..."
    if (type.isMatrix()) {
        TType columnType(type, 0);
        return type.getMatrixCols() * computeTypeLocationSize(columnType, stage);
    }

    assert(0);
    return 1;
}

// Same as computeTypeLocationSize but for uniforms
int TIntermediate::computeTypeUniformLocationSize(const TType& type)
{
    // "Individual elements of a uniform array are assigned
    // consecutive locations with the first element taking location
    // location."
    if (type.isArray()) {
        // TODO: perf: this can be flattened by using getCumulativeArraySize(), and a deref that discards all arrayness
        TType elementType(type, 0);
        if (type.isSizedArray()) {
            return type.getOuterArraySize() * computeTypeUniformLocationSize(elementType);
        } else {
            // TODO: are there valid cases of having an implicitly-sized array with a location?  If so, running this code too early.
            return computeTypeUniformLocationSize(elementType);
        }
    }

    // "Each subsequent inner-most member or element gets incremental
    // locations for the entire structure or array."
    if (type.isStruct()) {
        int size = 0;
        for (int member = 0; member < (int)type.getStruct()->size(); ++member) {
            TType memberType(type, member);
            size += computeTypeUniformLocationSize(memberType);
        }
        return size;
    }

    return 1;
}

#ifndef GLSLANG_WEB

// Accumulate xfb buffer ranges and check for collisions as the accumulation is done.
//
// Returns < 0 if no collision, >= 0 if collision and the value returned is a colliding value.
//
int TIntermediate::addXfbBufferOffset(const TType& type)
{
    const TQualifier& qualifier = type.getQualifier();

    assert(qualifier.hasXfbOffset() && qualifier.hasXfbBuffer());
    TXfbBuffer& buffer = xfbBuffers[qualifier.layoutXfbBuffer];

    // compute the range
    unsigned int size = computeTypeXfbSize(type, buffer.contains64BitType, buffer.contains32BitType, buffer.contains16BitType);
    buffer.implicitStride = std::max(buffer.implicitStride, qualifier.layoutXfbOffset + size);
    TRange range(qualifier.layoutXfbOffset, qualifier.layoutXfbOffset + size - 1);

    // check for collisions
    for (size_t r = 0; r < buffer.ranges.size(); ++r) {
        if (range.overlap(buffer.ranges[r])) {
            // there is a collision; pick an example to return
            return std::max(range.start, buffer.ranges[r].start);
        }
    }

    buffer.ranges.push_back(range);

    return -1;  // no collision
}

// Recursively figure out how many bytes of xfb buffer are used by the given type.
// Return the size of type, in bytes.
// Sets contains64BitType to true if the type contains a 64-bit data type.
// Sets contains32BitType to true if the type contains a 32-bit data type.
// Sets contains16BitType to true if the type contains a 16-bit data type.
// N.B. Caller must set contains64BitType, contains32BitType, and contains16BitType to false before calling.
unsigned int TIntermediate::computeTypeXfbSize(const TType& type, bool& contains64BitType, bool& contains32BitType, bool& contains16BitType) const
{
    // "...if applied to an aggregate containing a double or 64-bit integer, the offset must also be a multiple of 8,
    // and the space taken in the buffer will be a multiple of 8.
    // ...within the qualified entity, subsequent components are each
    // assigned, in order, to the next available offset aligned to a multiple of
    // that component's size.  Aggregate types are flattened down to the component
    // level to get this sequence of components."

    if (type.isSizedArray()) {
        // TODO: perf: this can be flattened by using getCumulativeArraySize(), and a deref that discards all arrayness
        // Unsized array use to xfb should be a compile error.
        TType elementType(type, 0);
        return type.getOuterArraySize() * computeTypeXfbSize(elementType, contains64BitType, contains16BitType, contains16BitType);
    }

    if (type.isStruct()) {
        unsigned int size = 0;
        bool structContains64BitType = false;
        bool structContains32BitType = false;
        bool structContains16BitType = false;
        for (int member = 0; member < (int)type.getStruct()->size(); ++member) {
            TType memberType(type, member);
            // "... if applied to
            // an aggregate containing a double or 64-bit integer, the offset must also be a multiple of 8,
            // and the space taken in the buffer will be a multiple of 8."
            bool memberContains64BitType = false;
            bool memberContains32BitType = false;
            bool memberContains16BitType = false;
            int memberSize = computeTypeXfbSize(memberType, memberContains64BitType, memberContains32BitType, memberContains16BitType);
            if (memberContains64BitType) {
                structContains64BitType = true;
                RoundToPow2(size, 8);
            } else if (memberContains32BitType) {
                structContains32BitType = true;
                RoundToPow2(size, 4);
            } else if (memberContains16BitType) {
                structContains16BitType = true;
                RoundToPow2(size, 2);
            }
            size += memberSize;
        }

        if (structContains64BitType) {
            contains64BitType = true;
            RoundToPow2(size, 8);
        } else if (structContains32BitType) {
            contains32BitType = true;
            RoundToPow2(size, 4);
        } else if (structContains16BitType) {
            contains16BitType = true;
            RoundToPow2(size, 2);
        }
        return size;
    }

    int numComponents {0};
    if (type.isScalar())
        numComponents = 1;
    else if (type.isVector())
        numComponents = type.getVectorSize();
    else if (type.isMatrix())
        numComponents = type.getMatrixCols() * type.getMatrixRows();
    else {
        assert(0);
        numComponents = 1;
    }

    if (type.getBasicType() == EbtDouble || type.getBasicType() == EbtInt64 || type.getBasicType() == EbtUint64) {
        contains64BitType = true;
        return 8 * numComponents;
    } else if (type.getBasicType() == EbtFloat16 || type.getBasicType() == EbtInt16 || type.getBasicType() == EbtUint16) {
        contains16BitType = true;
        return 2 * numComponents;
    } else if (type.getBasicType() == EbtInt8 || type.getBasicType() == EbtUint8)
        return numComponents;
    else {
        contains32BitType = true;
        return 4 * numComponents;
    }
}

#endif

const int baseAlignmentVec4Std140 = 16;

// Return the size and alignment of a component of the given type.
// The size is returned in the 'size' parameter
// Return value is the alignment..
int TIntermediate::getBaseAlignmentScalar(const TType& type, int& size)
{
#ifdef GLSLANG_WEB
    size = 4; return 4;
#endif

    switch (type.getBasicType()) {
    case EbtInt64:
    case EbtUint64:
    case EbtDouble:  size = 8; return 8;
    case EbtFloat16: size = 2; return 2;
    case EbtInt8:
    case EbtUint8:   size = 1; return 1;
    case EbtInt16:
    case EbtUint16:  size = 2; return 2;
    case EbtReference: size = 8; return 8;
    case EbtSampler:
    {
        if (type.isBindlessImage() || type.isBindlessTexture()) {
            size = 8; return 8;
        }
        else {
            size = 4; return 4;
        }
    }
    default:         size = 4; return 4;
    }
}

// Implement base-alignment and size rules from section 7.6.2.2 Standard Uniform Block Layout
// Operates recursively.
//
// If std140 is true, it does the rounding up to vec4 size required by std140,
// otherwise it does not, yielding std430 rules.
//
// The size is returned in the 'size' parameter
//
// The stride is only non-0 for arrays or matrices, and is the stride of the
// top-level object nested within the type.  E.g., for an array of matrices,
// it is the distances needed between matrices, despite the rules saying the
// stride comes from the flattening down to vectors.
//
// Return value is the alignment of the type.
int TIntermediate::getBaseAlignment(const TType& type, int& size, int& stride, TLayoutPacking layoutPacking, bool rowMajor)
{
    int alignment;

    bool std140 = layoutPacking == glslang::ElpStd140;
    // When using the std140 storage layout, structures will be laid out in buffer
    // storage with its members stored in monotonically increasing order based on their
    // location in the declaration. A structure and each structure member have a base
    // offset and a base alignment, from which an aligned offset is computed by rounding
    // the base offset up to a multiple of the base alignment. The base offset of the first
    // member of a structure is taken from the aligned offset of the structure itself. The
    // base offset of all other structure members is derived by taking the offset of the
    // last basic machine unit consumed by the previous member and adding one. Each
    // structure member is stored in memory at its aligned offset. The members of a top-
    // level uniform block are laid out in buffer storage by treating the uniform block as
    // a structure with a base offset of zero.
    //
    //   1. If the member is a scalar consuming N basic machine units, the base alignment is N.
    //
    //   2. If the member is a two- or four-component vector with components consuming N basic
    //      machine units, the base alignment is 2N or 4N, respectively.
    //
    //   3. If the member is a three-component vector with components consuming N
    //      basic machine units, the base alignment is 4N.
    //
    //   4. If the member is an array of scalars or vectors, the base alignment and array
    //      stride are set to match the base alignment of a single array element, according
    //      to rules (1), (2), and (3), and rounded up to the base alignment of a vec4. The
    //      array may have padding at the end; the base offset of the member following
    //      the array is rounded up to the next multiple of the base alignment.
    //
    //   5. If the member is a column-major matrix with C columns and R rows, the
    //      matrix is stored identically to an array of C column vectors with R
    //      components each, according to rule (4).
    //
    //   6. If the member is an array of S column-major matrices with C columns and
    //      R rows, the matrix is stored identically to a row of S X C column vectors
    //      with R components each, according to rule (4).
    //
    //   7. If the member is a row-major matrix with C columns and R rows, the matrix
    //      is stored identically to an array of R row vectors with C components each,
    //      according to rule (4).
    //
    //   8. If the member is an array of S row-major matrices with C columns and R
    //      rows, the matrix is stored identically to a row of S X R row vectors with C
    //      components each, according to rule (4).
    //
    //   9. If the member is a structure, the base alignment of the structure is N , where
    //      N is the largest base alignment value of any    of its members, and rounded
    //      up to the base alignment of a vec4. The individual members of this substructure
    //      are then assigned offsets by applying this set of rules recursively,
    //      where the base offset of the first member of the sub-structure is equal to the
    //      aligned offset of the structure. The structure may have padding at the end;
    //      the base offset of the member following the sub-structure is rounded up to
    //      the next multiple of the base alignment of the structure.
    //
    //   10. If the member is an array of S structures, the S elements of the array are laid
    //       out in order, according to rule (9).
    //
    //   Assuming, for rule 10:  The stride is the same as the size of an element.

    stride = 0;
    int dummyStride;

    // rules 4, 6, 8, and 10
    if (type.isArray()) {
        // TODO: perf: this might be flattened by using getCumulativeArraySize(), and a deref that discards all arrayness
        TType derefType(type, 0);
        alignment = getBaseAlignment(derefType, size, dummyStride, layoutPacking, rowMajor);
        if (std140)
            alignment = std::max(baseAlignmentVec4Std140, alignment);
        RoundToPow2(size, alignment);
        stride = size;  // uses full matrix size for stride of an array of matrices (not quite what rule 6/8, but what's expected)
                        // uses the assumption for rule 10 in the comment above
        // use one element to represent the last member of SSBO which is unsized array
        int arraySize = (type.isUnsizedArray() && (type.getOuterArraySize() == 0)) ? 1 : type.getOuterArraySize();
        size = stride * arraySize;
        return alignment;
    }

    // rule 9
    if (type.getBasicType() == EbtStruct || type.getBasicType() == EbtBlock) {
        const TTypeList& memberList = *type.getStruct();

        size = 0;
        int maxAlignment = std140 ? baseAlignmentVec4Std140 : 0;
        for (size_t m = 0; m < memberList.size(); ++m) {
            int memberSize;
            // modify just the children's view of matrix layout, if there is one for this member
            TLayoutMatrix subMatrixLayout = memberList[m].type->getQualifier().layoutMatrix;
            int memberAlignment = getBaseAlignment(*memberList[m].type, memberSize, dummyStride, layoutPacking,
                                                   (subMatrixLayout != ElmNone) ? (subMatrixLayout == ElmRowMajor) : rowMajor);
            maxAlignment = std::max(maxAlignment, memberAlignment);
            RoundToPow2(size, memberAlignment);
            size += memberSize;
        }

        // The structure may have padding at the end; the base offset of
        // the member following the sub-structure is rounded up to the next
        // multiple of the base alignment of the structure.
        RoundToPow2(size, maxAlignment);

        return maxAlignment;
    }

    // rule 1
    if (type.isScalar())
        return getBaseAlignmentScalar(type, size);

    // rules 2 and 3
    if (type.isVector()) {
        int scalarAlign = getBaseAlignmentScalar(type, size);
        switch (type.getVectorSize()) {
        case 1: // HLSL has this, GLSL does not
            return scalarAlign;
        case 2:
            size *= 2;
            return 2 * scalarAlign;
        default:
            size *= type.getVectorSize();
            return 4 * scalarAlign;
        }
    }

    // rules 5 and 7
    if (type.isMatrix()) {
        // rule 5: deref to row, not to column, meaning the size of vector is num columns instead of num rows
        TType derefType(type, 0, rowMajor);

        alignment = getBaseAlignment(derefType, size, dummyStride, layoutPacking, rowMajor);
        if (std140)
            alignment = std::max(baseAlignmentVec4Std140, alignment);
        RoundToPow2(size, alignment);
        stride = size;  // use intra-matrix stride for stride of a just a matrix
        if (rowMajor)
            size = stride * type.getMatrixRows();
        else
            size = stride * type.getMatrixCols();

        return alignment;
    }

    assert(0);  // all cases should be covered above
    size = baseAlignmentVec4Std140;
    return baseAlignmentVec4Std140;
}

// To aid the basic HLSL rule about crossing vec4 boundaries.
bool TIntermediate::improperStraddle(const TType& type, int size, int offset)
{
    if (! type.isVector() || type.isArray())
        return false;

    return size <= 16 ? offset / 16 != (offset + size - 1) / 16
                      : offset % 16 != 0;
}

int TIntermediate::getScalarAlignment(const TType& type, int& size, int& stride, bool rowMajor)
{
    int alignment;

    stride = 0;
    int dummyStride;

    if (type.isArray()) {
        TType derefType(type, 0);
        alignment = getScalarAlignment(derefType, size, dummyStride, rowMajor);

        stride = size;
        RoundToPow2(stride, alignment);

        size = stride * (type.getOuterArraySize() - 1) + size;
        return alignment;
    }

    if (type.getBasicType() == EbtStruct) {
        const TTypeList& memberList = *type.getStruct();

        size = 0;
        int maxAlignment = 0;
        for (size_t m = 0; m < memberList.size(); ++m) {
            int memberSize;
            // modify just the children's view of matrix layout, if there is one for this member
            TLayoutMatrix subMatrixLayout = memberList[m].type->getQualifier().layoutMatrix;
            int memberAlignment = getScalarAlignment(*memberList[m].type, memberSize, dummyStride,
                                                     (subMatrixLayout != ElmNone) ? (subMatrixLayout == ElmRowMajor) : rowMajor);
            maxAlignment = std::max(maxAlignment, memberAlignment);
            RoundToPow2(size, memberAlignment);
            size += memberSize;
        }

        return maxAlignment;
    }

    if (type.isScalar())
        return getBaseAlignmentScalar(type, size);

    if (type.isVector()) {
        int scalarAlign = getBaseAlignmentScalar(type, size);

        size *= type.getVectorSize();
        return scalarAlign;
    }

    if (type.isMatrix()) {
        TType derefType(type, 0, rowMajor);

        alignment = getScalarAlignment(derefType, size, dummyStride, rowMajor);

        stride = size;  // use intra-matrix stride for stride of a just a matrix
        if (rowMajor)
            size = stride * type.getMatrixRows();
        else
            size = stride * type.getMatrixCols();

        return alignment;
    }

    assert(0);  // all cases should be covered above
    size = 1;
    return 1;
}

int TIntermediate::getMemberAlignment(const TType& type, int& size, int& stride, TLayoutPacking layoutPacking, bool rowMajor)
{
    if (layoutPacking == glslang::ElpScalar) {
        return getScalarAlignment(type, size, stride, rowMajor);
    } else {
        return getBaseAlignment(type, size, stride, layoutPacking, rowMajor);
    }
}

// shared calculation by getOffset and getOffsets
void TIntermediate::updateOffset(const TType& parentType, const TType& memberType, int& offset, int& memberSize)
{
    int dummyStride;

    // modify just the children's view of matrix layout, if there is one for this member
    TLayoutMatrix subMatrixLayout = memberType.getQualifier().layoutMatrix;
    int memberAlignment = getMemberAlignment(memberType, memberSize, dummyStride,
                                             parentType.getQualifier().layoutPacking,
                                             subMatrixLayout != ElmNone
                                                 ? subMatrixLayout == ElmRowMajor
                                                 : parentType.getQualifier().layoutMatrix == ElmRowMajor);
    RoundToPow2(offset, memberAlignment);
}

// Lookup or calculate the offset of a block member, using the recursively
// defined block offset rules.
int TIntermediate::getOffset(const TType& type, int index)
{
    const TTypeList& memberList = *type.getStruct();

    // Don't calculate offset if one is present, it could be user supplied
    // and different than what would be calculated.  That is, this is faster,
    // but not just an optimization.
    if (memberList[index].type->getQualifier().hasOffset())
        return memberList[index].type->getQualifier().layoutOffset;

    int memberSize = 0;
    int offset = 0;
    for (int m = 0; m <= index; ++m) {
        updateOffset(type, *memberList[m].type, offset, memberSize);

        if (m < index)
            offset += memberSize;
    }

    return offset;
}

// Calculate the block data size.
// Block arrayness is not taken into account, each element is backed by a separate buffer.
int TIntermediate::getBlockSize(const TType& blockType)
{
    const TTypeList& memberList = *blockType.getStruct();
    int lastIndex = (int)memberList.size() - 1;
    int lastOffset = getOffset(blockType, lastIndex);

    int lastMemberSize;
    int dummyStride;
    getMemberAlignment(*memberList[lastIndex].type, lastMemberSize, dummyStride,
                       blockType.getQualifier().layoutPacking,
                       blockType.getQualifier().layoutMatrix == ElmRowMajor);

    return lastOffset + lastMemberSize;
}

int TIntermediate::computeBufferReferenceTypeSize(const TType& type)
{
    assert(type.isReference());
    int size = getBlockSize(*type.getReferentType());

    int align = type.getBufferReferenceAlignment();

    if (align) {
        size = (size + align - 1) & ~(align-1);
    }

    return size;
}

#ifndef GLSLANG_WEB
bool TIntermediate::isIoResizeArray(const TType& type, EShLanguage language) {
    return type.isArray() &&
            ((language == EShLangGeometry    && type.getQualifier().storage == EvqVaryingIn) ||
            (language == EShLangTessControl && (type.getQualifier().storage == EvqVaryingIn || type.getQualifier().storage == EvqVaryingOut) &&
                ! type.getQualifier().patch) ||
            (language == EShLangTessEvaluation && type.getQualifier().storage == EvqVaryingIn) ||
            (language == EShLangFragment && type.getQualifier().storage == EvqVaryingIn &&
             (type.getQualifier().pervertexNV || type.getQualifier().pervertexEXT)) ||
            (language == EShLangMesh && type.getQualifier().storage == EvqVaryingOut &&
                !type.getQualifier().perTaskNV));
}
#endif // not GLSLANG_WEB

} // end namespace glslang
