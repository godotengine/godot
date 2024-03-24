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

#ifndef _REFLECTION_INCLUDED
#define _REFLECTION_INCLUDED

#include "../Public/ShaderLang.h"
#include "../Include/Types.h"

#include <list>
#include <set>

//
// A reflection database and its interface, consistent with the OpenGL API reflection queries.
//

namespace glslang {

class TIntermediate;
class TIntermAggregate;
class TReflectionTraverser;

// The full reflection database
class TReflection {
public:
    TReflection(EShReflectionOptions opts, EShLanguage first, EShLanguage last)
        : options(opts), firstStage(first), lastStage(last), badReflection(TObjectReflection::badReflection())
    { 
        for (int dim=0; dim<3; ++dim)
            localSize[dim] = 0;
    }

    virtual ~TReflection() {}

    // grow the reflection stage by stage
    bool addStage(EShLanguage, const TIntermediate&);

    // for mapping a uniform index to a uniform object's description
    int getNumUniforms() { return (int)indexToUniform.size(); }
    const TObjectReflection& getUniform(int i) const
    {
        if (i >= 0 && i < (int)indexToUniform.size())
            return indexToUniform[i];
        else
            return badReflection;
    }

    // for mapping a block index to the block's description
    int getNumUniformBlocks() const { return (int)indexToUniformBlock.size(); }
    const TObjectReflection& getUniformBlock(int i) const
    {
        if (i >= 0 && i < (int)indexToUniformBlock.size())
            return indexToUniformBlock[i];
        else
            return badReflection;
    }

    // for mapping an pipeline input index to the input's description
    int getNumPipeInputs() { return (int)indexToPipeInput.size(); }
    const TObjectReflection& getPipeInput(int i) const
    {
        if (i >= 0 && i < (int)indexToPipeInput.size())
            return indexToPipeInput[i];
        else
            return badReflection;
    }

    // for mapping an pipeline output index to the output's description
    int getNumPipeOutputs() { return (int)indexToPipeOutput.size(); }
    const TObjectReflection& getPipeOutput(int i) const
    {
        if (i >= 0 && i < (int)indexToPipeOutput.size())
            return indexToPipeOutput[i];
        else
            return badReflection;
    }

    // for mapping from an atomic counter to the uniform index
    int getNumAtomicCounters() const { return (int)atomicCounterUniformIndices.size(); }
    const TObjectReflection& getAtomicCounter(int i) const
    {
        if (i >= 0 && i < (int)atomicCounterUniformIndices.size())
            return getUniform(atomicCounterUniformIndices[i]);
        else
            return badReflection;
    }

    // for mapping a buffer variable index to a buffer variable object's description
    int getNumBufferVariables() { return (int)indexToBufferVariable.size(); }
    const TObjectReflection& getBufferVariable(int i) const
    {
        if (i >= 0 && i < (int)indexToBufferVariable.size())
            return indexToBufferVariable[i];
        else
            return badReflection;
    }
    
    // for mapping a storage block index to the storage block's description
    int getNumStorageBuffers() const { return (int)indexToBufferBlock.size(); }
    const TObjectReflection&  getStorageBufferBlock(int i) const
    {
        if (i >= 0 && i < (int)indexToBufferBlock.size())
            return indexToBufferBlock[i];
        else
            return badReflection;
    }

    // for mapping any name to its index (block names, uniform names and input/output names)
    int getIndex(const char* name) const
    {
        TNameToIndex::const_iterator it = nameToIndex.find(name);
        if (it == nameToIndex.end())
            return -1;
        else
            return it->second;
    }

    // see getIndex(const char*)
    int getIndex(const TString& name) const { return getIndex(name.c_str()); }


    // for mapping any name to its index (only pipe input/output names)
    int getPipeIOIndex(const char* name, const bool inOrOut) const
    {
        TNameToIndex::const_iterator it = inOrOut ? pipeInNameToIndex.find(name) : pipeOutNameToIndex.find(name);
        if (it == (inOrOut ? pipeInNameToIndex.end() : pipeOutNameToIndex.end()))
            return -1;
        else
            return it->second;
    }

    // see gePipeIOIndex(const char*, const bool)
    int getPipeIOIndex(const TString& name, const bool inOrOut) const { return getPipeIOIndex(name.c_str(), inOrOut); }

    // Thread local size
    unsigned getLocalSize(int dim) const { return dim <= 2 ? localSize[dim] : 0; }

    void dump();

protected:
    friend class glslang::TReflectionTraverser;

    void buildCounterIndices(const TIntermediate&);
    void buildUniformStageMask(const TIntermediate& intermediate);
    void buildAttributeReflection(EShLanguage, const TIntermediate&);

    // Need a TString hash: typedef std::unordered_map<TString, int> TNameToIndex;
    typedef std::map<std::string, int> TNameToIndex;
    typedef std::vector<TObjectReflection> TMapIndexToReflection;
    typedef std::vector<int> TIndices;

    TMapIndexToReflection& GetBlockMapForStorage(TStorageQualifier storage)
    {
        if ((options & EShReflectionSeparateBuffers) && storage == EvqBuffer)
            return indexToBufferBlock;
        return indexToUniformBlock;
    }
    TMapIndexToReflection& GetVariableMapForStorage(TStorageQualifier storage)
    {
        if ((options & EShReflectionSeparateBuffers) && storage == EvqBuffer)
            return indexToBufferVariable;
        return indexToUniform;
    }

    EShReflectionOptions options;

    EShLanguage firstStage;
    EShLanguage lastStage;

    TObjectReflection badReflection; // return for queries of -1 or generally out of range; has expected descriptions with in it for this
    TNameToIndex nameToIndex;        // maps names to indexes; can hold all types of data: uniform/buffer and which function names have been processed
    TNameToIndex pipeInNameToIndex;  // maps pipe in names to indexes, this is a fix to seperate pipe I/O from uniforms and buffers.
    TNameToIndex pipeOutNameToIndex; // maps pipe out names to indexes, this is a fix to seperate pipe I/O from uniforms and buffers.
    TMapIndexToReflection indexToUniform;
    TMapIndexToReflection indexToUniformBlock;
    TMapIndexToReflection indexToBufferVariable;
    TMapIndexToReflection indexToBufferBlock;
    TMapIndexToReflection indexToPipeInput;
    TMapIndexToReflection indexToPipeOutput;
    TIndices atomicCounterUniformIndices;

    unsigned int localSize[3];
};

} // end namespace glslang

#endif // _REFLECTION_INCLUDED
