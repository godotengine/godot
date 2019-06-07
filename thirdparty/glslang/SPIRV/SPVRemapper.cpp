//
// Copyright (C) 2015 LunarG, Inc.
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

#include "SPVRemapper.h"
#include "doc.h"

#if !defined (use_cpp11)
// ... not supported before C++11
#else // defined (use_cpp11)

#include <algorithm>
#include <cassert>
#include "../glslang/Include/Common.h"

namespace spv {

    // By default, just abort on error.  Can be overridden via RegisterErrorHandler
    spirvbin_t::errorfn_t spirvbin_t::errorHandler = [](const std::string&) { exit(5); };
    // By default, eat log messages.  Can be overridden via RegisterLogHandler
    spirvbin_t::logfn_t   spirvbin_t::logHandler   = [](const std::string&) { };

    // This can be overridden to provide other message behavior if needed
    void spirvbin_t::msg(int minVerbosity, int indent, const std::string& txt) const
    {
        if (verbose >= minVerbosity)
            logHandler(std::string(indent, ' ') + txt);
    }

    // hash opcode, with special handling for OpExtInst
    std::uint32_t spirvbin_t::asOpCodeHash(unsigned word)
    {
        const spv::Op opCode = asOpCode(word);

        std::uint32_t offset = 0;

        switch (opCode) {
        case spv::OpExtInst:
            offset += asId(word + 4); break;
        default:
            break;
        }

        return opCode * 19 + offset; // 19 = small prime
    }

    spirvbin_t::range_t spirvbin_t::literalRange(spv::Op opCode) const
    {
        static const int maxCount = 1<<30;

        switch (opCode) {
        case spv::OpTypeFloat:        // fall through...
        case spv::OpTypePointer:      return range_t(2, 3);
        case spv::OpTypeInt:          return range_t(2, 4);
        // TODO: case spv::OpTypeImage:
        // TODO: case spv::OpTypeSampledImage:
        case spv::OpTypeSampler:      return range_t(3, 8);
        case spv::OpTypeVector:       // fall through
        case spv::OpTypeMatrix:       // ...
        case spv::OpTypePipe:         return range_t(3, 4);
        case spv::OpConstant:         return range_t(3, maxCount);
        default:                      return range_t(0, 0);
        }
    }

    spirvbin_t::range_t spirvbin_t::typeRange(spv::Op opCode) const
    {
        static const int maxCount = 1<<30;

        if (isConstOp(opCode))
            return range_t(1, 2);

        switch (opCode) {
        case spv::OpTypeVector:       // fall through
        case spv::OpTypeMatrix:       // ...
        case spv::OpTypeSampler:      // ...
        case spv::OpTypeArray:        // ...
        case spv::OpTypeRuntimeArray: // ...
        case spv::OpTypePipe:         return range_t(2, 3);
        case spv::OpTypeStruct:       // fall through
        case spv::OpTypeFunction:     return range_t(2, maxCount);
        case spv::OpTypePointer:      return range_t(3, 4);
        default:                      return range_t(0, 0);
        }
    }

    spirvbin_t::range_t spirvbin_t::constRange(spv::Op opCode) const
    {
        static const int maxCount = 1<<30;

        switch (opCode) {
        case spv::OpTypeArray:         // fall through...
        case spv::OpTypeRuntimeArray:  return range_t(3, 4);
        case spv::OpConstantComposite: return range_t(3, maxCount);
        default:                       return range_t(0, 0);
        }
    }

    // Return the size of a type in 32-bit words.  This currently only
    // handles ints and floats, and is only invoked by queries which must be
    // integer types.  If ever needed, it can be generalized.
    unsigned spirvbin_t::typeSizeInWords(spv::Id id) const
    {
        const unsigned typeStart = idPos(id);
        const spv::Op  opCode    = asOpCode(typeStart);

        if (errorLatch)
            return 0;

        switch (opCode) {
        case spv::OpTypeInt:   // fall through...
        case spv::OpTypeFloat: return (spv[typeStart+2]+31)/32;
        default:
            return 0;
        }
    }

    // Looks up the type of a given const or variable ID, and
    // returns its size in 32-bit words.
    unsigned spirvbin_t::idTypeSizeInWords(spv::Id id) const
    {
        const auto tid_it = idTypeSizeMap.find(id);
        if (tid_it == idTypeSizeMap.end()) {
            error("type size for ID not found");
            return 0;
        }

        return tid_it->second;
    }

    // Is this an opcode we should remove when using --strip?
    bool spirvbin_t::isStripOp(spv::Op opCode) const
    {
        switch (opCode) {
        case spv::OpSource:
        case spv::OpSourceExtension:
        case spv::OpName:
        case spv::OpMemberName:
        case spv::OpLine:           return true;
        default:                    return false;
        }
    }

    // Return true if this opcode is flow control
    bool spirvbin_t::isFlowCtrl(spv::Op opCode) const
    {
        switch (opCode) {
        case spv::OpBranchConditional:
        case spv::OpBranch:
        case spv::OpSwitch:
        case spv::OpLoopMerge:
        case spv::OpSelectionMerge:
        case spv::OpLabel:
        case spv::OpFunction:
        case spv::OpFunctionEnd:    return true;
        default:                    return false;
        }
    }

    // Return true if this opcode defines a type
    bool spirvbin_t::isTypeOp(spv::Op opCode) const
    {
        switch (opCode) {
        case spv::OpTypeVoid:
        case spv::OpTypeBool:
        case spv::OpTypeInt:
        case spv::OpTypeFloat:
        case spv::OpTypeVector:
        case spv::OpTypeMatrix:
        case spv::OpTypeImage:
        case spv::OpTypeSampler:
        case spv::OpTypeArray:
        case spv::OpTypeRuntimeArray:
        case spv::OpTypeStruct:
        case spv::OpTypeOpaque:
        case spv::OpTypePointer:
        case spv::OpTypeFunction:
        case spv::OpTypeEvent:
        case spv::OpTypeDeviceEvent:
        case spv::OpTypeReserveId:
        case spv::OpTypeQueue:
        case spv::OpTypeSampledImage:
        case spv::OpTypePipe:         return true;
        default:                      return false;
        }
    }

    // Return true if this opcode defines a constant
    bool spirvbin_t::isConstOp(spv::Op opCode) const
    {
        switch (opCode) {
        case spv::OpConstantSampler:
            error("unimplemented constant type");
            return true;

        case spv::OpConstantNull:
        case spv::OpConstantTrue:
        case spv::OpConstantFalse:
        case spv::OpConstantComposite:
        case spv::OpConstant:
            return true;

        default:
            return false;
        }
    }

    const auto inst_fn_nop = [](spv::Op, unsigned) { return false; };
    const auto op_fn_nop   = [](spv::Id&)          { };

    // g++ doesn't like these defined in the class proper in an anonymous namespace.
    // Dunno why.  Also MSVC doesn't like the constexpr keyword.  Also dunno why.
    // Defining them externally seems to please both compilers, so, here they are.
    const spv::Id spirvbin_t::unmapped    = spv::Id(-10000);
    const spv::Id spirvbin_t::unused      = spv::Id(-10001);
    const int     spirvbin_t::header_size = 5;

    spv::Id spirvbin_t::nextUnusedId(spv::Id id)
    {
        while (isNewIdMapped(id))  // search for an unused ID
            ++id;

        return id;
    }

    spv::Id spirvbin_t::localId(spv::Id id, spv::Id newId)
    {
        //assert(id != spv::NoResult && newId != spv::NoResult);

        if (id > bound()) {
            error(std::string("ID out of range: ") + std::to_string(id));
            return spirvbin_t::unused;
        }

        if (id >= idMapL.size())
            idMapL.resize(id+1, unused);

        if (newId != unmapped && newId != unused) {
            if (isOldIdUnused(id)) {
                error(std::string("ID unused in module: ") + std::to_string(id));
                return spirvbin_t::unused;
            }

            if (!isOldIdUnmapped(id)) {
                error(std::string("ID already mapped: ") + std::to_string(id) + " -> "
                        + std::to_string(localId(id)));

                return spirvbin_t::unused;
            }

            if (isNewIdMapped(newId)) {
                error(std::string("ID already used in module: ") + std::to_string(newId));
                return spirvbin_t::unused;
            }

            msg(4, 4, std::string("map: ") + std::to_string(id) + " -> " + std::to_string(newId));
            setMapped(newId);
            largestNewId = std::max(largestNewId, newId);
        }

        return idMapL[id] = newId;
    }

    // Parse a literal string from the SPIR binary and return it as an std::string
    // Due to C++11 RValue references, this doesn't copy the result string.
    std::string spirvbin_t::literalString(unsigned word) const
    {
        std::string literal;

        literal.reserve(16);

        const char* bytes = reinterpret_cast<const char*>(spv.data() + word);

        while (bytes && *bytes)
            literal += *bytes++;

        return literal;
    }

    void spirvbin_t::applyMap()
    {
        msg(3, 2, std::string("Applying map: "));

        // Map local IDs through the ID map
        process(inst_fn_nop, // ignore instructions
            [this](spv::Id& id) {
                id = localId(id);

                if (errorLatch)
                    return;

                assert(id != unused && id != unmapped);
            }
        );
    }

    // Find free IDs for anything we haven't mapped
    void spirvbin_t::mapRemainder()
    {
        msg(3, 2, std::string("Remapping remainder: "));

        spv::Id     unusedId  = 1;  // can't use 0: that's NoResult
        spirword_t  maxBound  = 0;

        for (spv::Id id = 0; id < idMapL.size(); ++id) {
            if (isOldIdUnused(id))
                continue;

            // Find a new mapping for any used but unmapped IDs
            if (isOldIdUnmapped(id)) {
                localId(id, unusedId = nextUnusedId(unusedId));
                if (errorLatch)
                    return;
            }

            if (isOldIdUnmapped(id)) {
                error(std::string("old ID not mapped: ") + std::to_string(id));
                return;
            }

            // Track max bound
            maxBound = std::max(maxBound, localId(id) + 1);

            if (errorLatch)
                return;
        }

        bound(maxBound); // reset header ID bound to as big as it now needs to be
    }

    // Mark debug instructions for stripping
    void spirvbin_t::stripDebug()
    {
        // Strip instructions in the stripOp set: debug info.
        process(
            [&](spv::Op opCode, unsigned start) {
                // remember opcodes we want to strip later
                if (isStripOp(opCode))
                    stripInst(start);
                return true;
            },
            op_fn_nop);
    }

    // Mark instructions that refer to now-removed IDs for stripping
    void spirvbin_t::stripDeadRefs()
    {
        process(
            [&](spv::Op opCode, unsigned start) {
                // strip opcodes pointing to removed data
                switch (opCode) {
                case spv::OpName:
                case spv::OpMemberName:
                case spv::OpDecorate:
                case spv::OpMemberDecorate:
                    if (idPosR.find(asId(start+1)) == idPosR.end())
                        stripInst(start);
                    break;
                default:
                    break; // leave it alone
                }

                return true;
            },
            op_fn_nop);

        strip();
    }

    // Update local maps of ID, type, etc positions
    void spirvbin_t::buildLocalMaps()
    {
        msg(2, 2, std::string("build local maps: "));

        mapped.clear();
        idMapL.clear();
//      preserve nameMap, so we don't clear that.
        fnPos.clear();
        fnCalls.clear();
        typeConstPos.clear();
        idPosR.clear();
        entryPoint = spv::NoResult;
        largestNewId = 0;

        idMapL.resize(bound(), unused);

        int         fnStart = 0;
        spv::Id     fnRes   = spv::NoResult;

        // build local Id and name maps
        process(
            [&](spv::Op opCode, unsigned start) {
                unsigned word = start+1;
                spv::Id  typeId = spv::NoResult;

                if (spv::InstructionDesc[opCode].hasType())
                    typeId = asId(word++);

                // If there's a result ID, remember the size of its type
                if (spv::InstructionDesc[opCode].hasResult()) {
                    const spv::Id resultId = asId(word++);
                    idPosR[resultId] = start;

                    if (typeId != spv::NoResult) {
                        const unsigned idTypeSize = typeSizeInWords(typeId);

                        if (errorLatch)
                            return false;

                        if (idTypeSize != 0)
                            idTypeSizeMap[resultId] = idTypeSize;
                    }
                }

                if (opCode == spv::Op::OpName) {
                    const spv::Id    target = asId(start+1);
                    const std::string  name = literalString(start+2);
                    nameMap[name] = target;

                } else if (opCode == spv::Op::OpFunctionCall) {
                    ++fnCalls[asId(start + 3)];
                } else if (opCode == spv::Op::OpEntryPoint) {
                    entryPoint = asId(start + 2);
                } else if (opCode == spv::Op::OpFunction) {
                    if (fnStart != 0) {
                        error("nested function found");
                        return false;
                    }

                    fnStart = start;
                    fnRes   = asId(start + 2);
                } else if (opCode == spv::Op::OpFunctionEnd) {
                    assert(fnRes != spv::NoResult);
                    if (fnStart == 0) {
                        error("function end without function start");
                        return false;
                    }

                    fnPos[fnRes] = range_t(fnStart, start + asWordCount(start));
                    fnStart = 0;
                } else if (isConstOp(opCode)) {
                    if (errorLatch)
                        return false;

                    assert(asId(start + 2) != spv::NoResult);
                    typeConstPos.insert(start);
                } else if (isTypeOp(opCode)) {
                    assert(asId(start + 1) != spv::NoResult);
                    typeConstPos.insert(start);
                }

                return false;
            },

            [this](spv::Id& id) { localId(id, unmapped); }
        );
    }

    // Validate the SPIR header
    void spirvbin_t::validate() const
    {
        msg(2, 2, std::string("validating: "));

        if (spv.size() < header_size) {
            error("file too short: ");
            return;
        }

        if (magic() != spv::MagicNumber) {
            error("bad magic number");
            return;
        }

        // field 1 = version
        // field 2 = generator magic
        // field 3 = result <id> bound

        if (schemaNum() != 0) {
            error("bad schema, must be 0");
            return;
        }
    }

    int spirvbin_t::processInstruction(unsigned word, instfn_t instFn, idfn_t idFn)
    {
        const auto     instructionStart = word;
        const unsigned wordCount = asWordCount(instructionStart);
        const int      nextInst  = word++ + wordCount;
        spv::Op  opCode    = asOpCode(instructionStart);

        if (nextInst > int(spv.size())) {
            error("spir instruction terminated too early");
            return -1;
        }

        // Base for computing number of operands; will be updated as more is learned
        unsigned numOperands = wordCount - 1;

        if (instFn(opCode, instructionStart))
            return nextInst;

        // Read type and result ID from instruction desc table
        if (spv::InstructionDesc[opCode].hasType()) {
            idFn(asId(word++));
            --numOperands;
        }

        if (spv::InstructionDesc[opCode].hasResult()) {
            idFn(asId(word++));
            --numOperands;
        }

        // Extended instructions: currently, assume everything is an ID.
        // TODO: add whatever data we need for exceptions to that
        if (opCode == spv::OpExtInst) {
            word        += 2; // instruction set, and instruction from set
            numOperands -= 2;

            for (unsigned op=0; op < numOperands; ++op)
                idFn(asId(word++)); // ID

            return nextInst;
        }

        // Circular buffer so we can look back at previous unmapped values during the mapping pass.
        static const unsigned idBufferSize = 4;
        spv::Id idBuffer[idBufferSize];
        unsigned idBufferPos = 0;

        // Store IDs from instruction in our map
        for (int op = 0; numOperands > 0; ++op, --numOperands) {
            // SpecConstantOp is special: it includes the operands of another opcode which is
            // given as a literal in the 3rd word.  We will switch over to pretending that the
            // opcode being processed is the literal opcode value of the SpecConstantOp.  See the
            // SPIRV spec for details.  This way we will handle IDs and literals as appropriate for
            // the embedded op.
            if (opCode == spv::OpSpecConstantOp) {
                if (op == 0) {
                    opCode = asOpCode(word++);  // this is the opcode embedded in the SpecConstantOp.
                    --numOperands;
                }
            }

            switch (spv::InstructionDesc[opCode].operands.getClass(op)) {
            case spv::OperandId:
            case spv::OperandScope:
            case spv::OperandMemorySemantics:
                idBuffer[idBufferPos] = asId(word);
                idBufferPos = (idBufferPos + 1) % idBufferSize;
                idFn(asId(word++));
                break;

            case spv::OperandVariableIds:
                for (unsigned i = 0; i < numOperands; ++i)
                    idFn(asId(word++));
                return nextInst;

            case spv::OperandVariableLiterals:
                // for clarity
                // if (opCode == spv::OpDecorate && asDecoration(word - 1) == spv::DecorationBuiltIn) {
                //     ++word;
                //     --numOperands;
                // }
                // word += numOperands;
                return nextInst;

            case spv::OperandVariableLiteralId: {
                if (opCode == OpSwitch) {
                    // word-2 is the position of the selector ID.  OpSwitch Literals match its type.
                    // In case the IDs are currently being remapped, we get the word[-2] ID from
                    // the circular idBuffer.
                    const unsigned literalSizePos = (idBufferPos+idBufferSize-2) % idBufferSize;
                    const unsigned literalSize = idTypeSizeInWords(idBuffer[literalSizePos]);
                    const unsigned numLiteralIdPairs = (nextInst-word) / (1+literalSize);

                    if (errorLatch)
                        return -1;

                    for (unsigned arg=0; arg<numLiteralIdPairs; ++arg) {
                        word += literalSize;  // literal
                        idFn(asId(word++));   // label
                    }
                } else {
                    assert(0); // currentely, only OpSwitch uses OperandVariableLiteralId
                }

                return nextInst;
            }

            case spv::OperandLiteralString: {
                const int stringWordCount = literalStringWords(literalString(word));
                word += stringWordCount;
                numOperands -= (stringWordCount-1); // -1 because for() header post-decrements
                break;
            }

            // Execution mode might have extra literal operands.  Skip them.
            case spv::OperandExecutionMode:
                return nextInst;

            // Single word operands we simply ignore, as they hold no IDs
            case spv::OperandLiteralNumber:
            case spv::OperandSource:
            case spv::OperandExecutionModel:
            case spv::OperandAddressing:
            case spv::OperandMemory:
            case spv::OperandStorage:
            case spv::OperandDimensionality:
            case spv::OperandSamplerAddressingMode:
            case spv::OperandSamplerFilterMode:
            case spv::OperandSamplerImageFormat:
            case spv::OperandImageChannelOrder:
            case spv::OperandImageChannelDataType:
            case spv::OperandImageOperands:
            case spv::OperandFPFastMath:
            case spv::OperandFPRoundingMode:
            case spv::OperandLinkageType:
            case spv::OperandAccessQualifier:
            case spv::OperandFuncParamAttr:
            case spv::OperandDecoration:
            case spv::OperandBuiltIn:
            case spv::OperandSelect:
            case spv::OperandLoop:
            case spv::OperandFunction:
            case spv::OperandMemoryAccess:
            case spv::OperandGroupOperation:
            case spv::OperandKernelEnqueueFlags:
            case spv::OperandKernelProfilingInfo:
            case spv::OperandCapability:
                ++word;
                break;

            default:
                assert(0 && "Unhandled Operand Class");
                break;
            }
        }

        return nextInst;
    }

    // Make a pass over all the instructions and process them given appropriate functions
    spirvbin_t& spirvbin_t::process(instfn_t instFn, idfn_t idFn, unsigned begin, unsigned end)
    {
        // For efficiency, reserve name map space.  It can grow if needed.
        nameMap.reserve(32);

        // If begin or end == 0, use defaults
        begin = (begin == 0 ? header_size          : begin);
        end   = (end   == 0 ? unsigned(spv.size()) : end);

        // basic parsing and InstructionDesc table borrowed from SpvDisassemble.cpp...
        unsigned nextInst = unsigned(spv.size());

        for (unsigned word = begin; word < end; word = nextInst) {
            nextInst = processInstruction(word, instFn, idFn);

            if (errorLatch)
                return *this;
        }

        return *this;
    }

    // Apply global name mapping to a single module
    void spirvbin_t::mapNames()
    {
        static const std::uint32_t softTypeIdLimit = 3011;  // small prime.  TODO: get from options
        static const std::uint32_t firstMappedID   = 3019;  // offset into ID space

        for (const auto& name : nameMap) {
            std::uint32_t hashval = 1911;
            for (const char c : name.first)
                hashval = hashval * 1009 + c;

            if (isOldIdUnmapped(name.second)) {
                localId(name.second, nextUnusedId(hashval % softTypeIdLimit + firstMappedID));
                if (errorLatch)
                    return;
            }
        }
    }

    // Map fn contents to IDs of similar functions in other modules
    void spirvbin_t::mapFnBodies()
    {
        static const std::uint32_t softTypeIdLimit = 19071;  // small prime.  TODO: get from options
        static const std::uint32_t firstMappedID   =  6203;  // offset into ID space

        // Initial approach: go through some high priority opcodes first and assign them
        // hash values.

        spv::Id               fnId       = spv::NoResult;
        std::vector<unsigned> instPos;
        instPos.reserve(unsigned(spv.size()) / 16); // initial estimate; can grow if needed.

        // Build local table of instruction start positions
        process(
            [&](spv::Op, unsigned start) { instPos.push_back(start); return true; },
            op_fn_nop);

        if (errorLatch)
            return;

        // Window size for context-sensitive canonicalization values
        // Empirical best size from a single data set.  TODO: Would be a good tunable.
        // We essentially perform a little convolution around each instruction,
        // to capture the flavor of nearby code, to hopefully match to similar
        // code in other modules.
        static const unsigned windowSize = 2;

        for (unsigned entry = 0; entry < unsigned(instPos.size()); ++entry) {
            const unsigned start  = instPos[entry];
            const spv::Op  opCode = asOpCode(start);

            if (opCode == spv::OpFunction)
                fnId   = asId(start + 2);

            if (opCode == spv::OpFunctionEnd)
                fnId = spv::NoResult;

            if (fnId != spv::NoResult) { // if inside a function
                if (spv::InstructionDesc[opCode].hasResult()) {
                    const unsigned word    = start + (spv::InstructionDesc[opCode].hasType() ? 2 : 1);
                    const spv::Id  resId   = asId(word);
                    std::uint32_t  hashval = fnId * 17; // small prime

                    for (unsigned i = entry-1; i >= entry-windowSize; --i) {
                        if (asOpCode(instPos[i]) == spv::OpFunction)
                            break;
                        hashval = hashval * 30103 + asOpCodeHash(instPos[i]); // 30103 = semiarbitrary prime
                    }

                    for (unsigned i = entry; i <= entry + windowSize; ++i) {
                        if (asOpCode(instPos[i]) == spv::OpFunctionEnd)
                            break;
                        hashval = hashval * 30103 + asOpCodeHash(instPos[i]); // 30103 = semiarbitrary prime
                    }

                    if (isOldIdUnmapped(resId)) {
                        localId(resId, nextUnusedId(hashval % softTypeIdLimit + firstMappedID));
                        if (errorLatch)
                            return;
                    }

                }
            }
        }

        spv::Op          thisOpCode(spv::OpNop);
        std::unordered_map<int, int> opCounter;
        int              idCounter(0);
        fnId = spv::NoResult;

        process(
            [&](spv::Op opCode, unsigned start) {
                switch (opCode) {
                case spv::OpFunction:
                    // Reset counters at each function
                    idCounter = 0;
                    opCounter.clear();
                    fnId = asId(start + 2);
                    break;

                case spv::OpImageSampleImplicitLod:
                case spv::OpImageSampleExplicitLod:
                case spv::OpImageSampleDrefImplicitLod:
                case spv::OpImageSampleDrefExplicitLod:
                case spv::OpImageSampleProjImplicitLod:
                case spv::OpImageSampleProjExplicitLod:
                case spv::OpImageSampleProjDrefImplicitLod:
                case spv::OpImageSampleProjDrefExplicitLod:
                case spv::OpDot:
                case spv::OpCompositeExtract:
                case spv::OpCompositeInsert:
                case spv::OpVectorShuffle:
                case spv::OpLabel:
                case spv::OpVariable:

                case spv::OpAccessChain:
                case spv::OpLoad:
                case spv::OpStore:
                case spv::OpCompositeConstruct:
                case spv::OpFunctionCall:
                    ++opCounter[opCode];
                    idCounter = 0;
                    thisOpCode = opCode;
                    break;
                default:
                    thisOpCode = spv::OpNop;
                }

                return false;
            },

            [&](spv::Id& id) {
                if (thisOpCode != spv::OpNop) {
                    ++idCounter;
                    const std::uint32_t hashval = opCounter[thisOpCode] * thisOpCode * 50047 + idCounter + fnId * 117;

                    if (isOldIdUnmapped(id))
                        localId(id, nextUnusedId(hashval % softTypeIdLimit + firstMappedID));
                }
            });
    }

    // EXPERIMENTAL: forward IO and uniform load/stores into operands
    // This produces invalid Schema-0 SPIRV
    void spirvbin_t::forwardLoadStores()
    {
        idset_t fnLocalVars; // set of function local vars
        idmap_t idMap;       // Map of load result IDs to what they load

        // EXPERIMENTAL: Forward input and access chain loads into consumptions
        process(
            [&](spv::Op opCode, unsigned start) {
                // Add inputs and uniforms to the map
                if ((opCode == spv::OpVariable && asWordCount(start) == 4) &&
                    (spv[start+3] == spv::StorageClassUniform ||
                    spv[start+3] == spv::StorageClassUniformConstant ||
                    spv[start+3] == spv::StorageClassInput))
                    fnLocalVars.insert(asId(start+2));

                if (opCode == spv::OpAccessChain && fnLocalVars.count(asId(start+3)) > 0)
                    fnLocalVars.insert(asId(start+2));

                if (opCode == spv::OpLoad && fnLocalVars.count(asId(start+3)) > 0) {
                    idMap[asId(start+2)] = asId(start+3);
                    stripInst(start);
                }

                return false;
            },

            [&](spv::Id& id) { if (idMap.find(id) != idMap.end()) id = idMap[id]; }
        );

        if (errorLatch)
            return;

        // EXPERIMENTAL: Implicit output stores
        fnLocalVars.clear();
        idMap.clear();

        process(
            [&](spv::Op opCode, unsigned start) {
                // Add inputs and uniforms to the map
                if ((opCode == spv::OpVariable && asWordCount(start) == 4) &&
                    (spv[start+3] == spv::StorageClassOutput))
                    fnLocalVars.insert(asId(start+2));

                if (opCode == spv::OpStore && fnLocalVars.count(asId(start+1)) > 0) {
                    idMap[asId(start+2)] = asId(start+1);
                    stripInst(start);
                }

                return false;
            },
            op_fn_nop);

        if (errorLatch)
            return;

        process(
            inst_fn_nop,
            [&](spv::Id& id) { if (idMap.find(id) != idMap.end()) id = idMap[id]; }
        );

        if (errorLatch)
            return;

        strip();          // strip out data we decided to eliminate
    }

    // optimize loads and stores
    void spirvbin_t::optLoadStore()
    {
        idset_t    fnLocalVars;  // candidates for removal (only locals)
        idmap_t    idMap;        // Map of load result IDs to what they load
        blockmap_t blockMap;     // Map of IDs to blocks they first appear in
        int        blockNum = 0; // block count, to avoid crossing flow control

        // Find all the function local pointers stored at most once, and not via access chains
        process(
            [&](spv::Op opCode, unsigned start) {
                const int wordCount = asWordCount(start);

                // Count blocks, so we can avoid crossing flow control
                if (isFlowCtrl(opCode))
                    ++blockNum;

                // Add local variables to the map
                if ((opCode == spv::OpVariable && spv[start+3] == spv::StorageClassFunction && asWordCount(start) == 4)) {
                    fnLocalVars.insert(asId(start+2));
                    return true;
                }

                // Ignore process vars referenced via access chain
                if ((opCode == spv::OpAccessChain || opCode == spv::OpInBoundsAccessChain) && fnLocalVars.count(asId(start+3)) > 0) {
                    fnLocalVars.erase(asId(start+3));
                    idMap.erase(asId(start+3));
                    return true;
                }

                if (opCode == spv::OpLoad && fnLocalVars.count(asId(start+3)) > 0) {
                    const spv::Id varId = asId(start+3);

                    // Avoid loads before stores
                    if (idMap.find(varId) == idMap.end()) {
                        fnLocalVars.erase(varId);
                        idMap.erase(varId);
                    }

                    // don't do for volatile references
                    if (wordCount > 4 && (spv[start+4] & spv::MemoryAccessVolatileMask)) {
                        fnLocalVars.erase(varId);
                        idMap.erase(varId);
                    }

                    // Handle flow control
                    if (blockMap.find(varId) == blockMap.end()) {
                        blockMap[varId] = blockNum;  // track block we found it in.
                    } else if (blockMap[varId] != blockNum) {
                        fnLocalVars.erase(varId);  // Ignore if crosses flow control
                        idMap.erase(varId);
                    }

                    return true;
                }

                if (opCode == spv::OpStore && fnLocalVars.count(asId(start+1)) > 0) {
                    const spv::Id varId = asId(start+1);

                    if (idMap.find(varId) == idMap.end()) {
                        idMap[varId] = asId(start+2);
                    } else {
                        // Remove if it has more than one store to the same pointer
                        fnLocalVars.erase(varId);
                        idMap.erase(varId);
                    }

                    // don't do for volatile references
                    if (wordCount > 3 && (spv[start+3] & spv::MemoryAccessVolatileMask)) {
                        fnLocalVars.erase(asId(start+3));
                        idMap.erase(asId(start+3));
                    }

                    // Handle flow control
                    if (blockMap.find(varId) == blockMap.end()) {
                        blockMap[varId] = blockNum;  // track block we found it in.
                    } else if (blockMap[varId] != blockNum) {
                        fnLocalVars.erase(varId);  // Ignore if crosses flow control
                        idMap.erase(varId);
                    }

                    return true;
                }

                return false;
            },

            // If local var id used anywhere else, don't eliminate
            [&](spv::Id& id) {
                if (fnLocalVars.count(id) > 0) {
                    fnLocalVars.erase(id);
                    idMap.erase(id);
                }
            }
        );

        if (errorLatch)
            return;

        process(
            [&](spv::Op opCode, unsigned start) {
                if (opCode == spv::OpLoad && fnLocalVars.count(asId(start+3)) > 0)
                    idMap[asId(start+2)] = idMap[asId(start+3)];
                return false;
            },
            op_fn_nop);

        if (errorLatch)
            return;

        // Chase replacements to their origins, in case there is a chain such as:
        //   2 = store 1
        //   3 = load 2
        //   4 = store 3
        //   5 = load 4
        // We want to replace uses of 5 with 1.
        for (const auto& idPair : idMap) {
            spv::Id id = idPair.first;
            while (idMap.find(id) != idMap.end())  // Chase to end of chain
                id = idMap[id];

            idMap[idPair.first] = id;              // replace with final result
        }

        // Remove the load/store/variables for the ones we've discovered
        process(
            [&](spv::Op opCode, unsigned start) {
                if ((opCode == spv::OpLoad  && fnLocalVars.count(asId(start+3)) > 0) ||
                    (opCode == spv::OpStore && fnLocalVars.count(asId(start+1)) > 0) ||
                    (opCode == spv::OpVariable && fnLocalVars.count(asId(start+2)) > 0)) {

                    stripInst(start);
                    return true;
                }

                return false;
            },

            [&](spv::Id& id) {
                if (idMap.find(id) != idMap.end()) id = idMap[id];
            }
        );

        if (errorLatch)
            return;

        strip();          // strip out data we decided to eliminate
    }

    // remove bodies of uncalled functions
    void spirvbin_t::dceFuncs()
    {
        msg(3, 2, std::string("Removing Dead Functions: "));

        // TODO: There are more efficient ways to do this.
        bool changed = true;

        while (changed) {
            changed = false;

            for (auto fn = fnPos.begin(); fn != fnPos.end(); ) {
                if (fn->first == entryPoint) { // don't DCE away the entry point!
                    ++fn;
                    continue;
                }

                const auto call_it = fnCalls.find(fn->first);

                if (call_it == fnCalls.end() || call_it->second == 0) {
                    changed = true;
                    stripRange.push_back(fn->second);

                    // decrease counts of called functions
                    process(
                        [&](spv::Op opCode, unsigned start) {
                            if (opCode == spv::Op::OpFunctionCall) {
                                const auto call_it = fnCalls.find(asId(start + 3));
                                if (call_it != fnCalls.end()) {
                                    if (--call_it->second <= 0)
                                        fnCalls.erase(call_it);
                                }
                            }

                            return true;
                        },
                        op_fn_nop,
                        fn->second.first,
                        fn->second.second);

                    if (errorLatch)
                        return;

                    fn = fnPos.erase(fn);
                } else ++fn;
            }
        }
    }

    // remove unused function variables + decorations
    void spirvbin_t::dceVars()
    {
        msg(3, 2, std::string("DCE Vars: "));

        std::unordered_map<spv::Id, int> varUseCount;

        // Count function variable use
        process(
            [&](spv::Op opCode, unsigned start) {
                if (opCode == spv::OpVariable) {
                    ++varUseCount[asId(start+2)];
                    return true;
                } else if (opCode == spv::OpEntryPoint) {
                    const int wordCount = asWordCount(start);
                    for (int i = 4; i < wordCount; i++) {
                        ++varUseCount[asId(start+i)];
                    }
                    return true;
                } else
                    return false;
            },

            [&](spv::Id& id) { if (varUseCount[id]) ++varUseCount[id]; }
        );

        if (errorLatch)
            return;

        // Remove single-use function variables + associated decorations and names
        process(
            [&](spv::Op opCode, unsigned start) {
                spv::Id id = spv::NoResult;
                if (opCode == spv::OpVariable)
                    id = asId(start+2);
                if (opCode == spv::OpDecorate || opCode == spv::OpName)
                    id = asId(start+1);

                if (id != spv::NoResult && varUseCount[id] == 1)
                    stripInst(start);

                return true;
            },
            op_fn_nop);
    }

    // remove unused types
    void spirvbin_t::dceTypes()
    {
        std::vector<bool> isType(bound(), false);

        // for speed, make O(1) way to get to type query (map is log(n))
        for (const auto typeStart : typeConstPos)
            isType[asTypeConstId(typeStart)] = true;

        std::unordered_map<spv::Id, int> typeUseCount;

        // This is not the most efficient algorithm, but this is an offline tool, and
        // it's easy to write this way.  Can be improved opportunistically if needed.
        bool changed = true;
        while (changed) {
            changed = false;
            strip();
            typeUseCount.clear();

            // Count total type usage
            process(inst_fn_nop,
                    [&](spv::Id& id) { if (isType[id]) ++typeUseCount[id]; }
                    );

            if (errorLatch)
                return;

            // Remove single reference types
            for (const auto typeStart : typeConstPos) {
                const spv::Id typeId = asTypeConstId(typeStart);
                if (typeUseCount[typeId] == 1) {
                    changed = true;
                    --typeUseCount[typeId];
                    stripInst(typeStart);
                }
            }

            if (errorLatch)
                return;
        }
    }

#ifdef NOTDEF
    bool spirvbin_t::matchType(const spirvbin_t::globaltypes_t& globalTypes, spv::Id lt, spv::Id gt) const
    {
        // Find the local type id "lt" and global type id "gt"
        const auto lt_it = typeConstPosR.find(lt);
        if (lt_it == typeConstPosR.end())
            return false;

        const auto typeStart = lt_it->second;

        // Search for entry in global table
        const auto gtype = globalTypes.find(gt);
        if (gtype == globalTypes.end())
            return false;

        const auto& gdata = gtype->second;

        // local wordcount and opcode
        const int     wordCount   = asWordCount(typeStart);
        const spv::Op opCode      = asOpCode(typeStart);

        // no type match if opcodes don't match, or operand count doesn't match
        if (opCode != opOpCode(gdata[0]) || wordCount != opWordCount(gdata[0]))
            return false;

        const unsigned numOperands = wordCount - 2; // all types have a result

        const auto cmpIdRange = [&](range_t range) {
            for (int x=range.first; x<std::min(range.second, wordCount); ++x)
                if (!matchType(globalTypes, asId(typeStart+x), gdata[x]))
                    return false;
            return true;
        };

        const auto cmpConst   = [&]() { return cmpIdRange(constRange(opCode)); };
        const auto cmpSubType = [&]() { return cmpIdRange(typeRange(opCode));  };

        // Compare literals in range [start,end)
        const auto cmpLiteral = [&]() {
            const auto range = literalRange(opCode);
            return std::equal(spir.begin() + typeStart + range.first,
                spir.begin() + typeStart + std::min(range.second, wordCount),
                gdata.begin() + range.first);
        };

        assert(isTypeOp(opCode) || isConstOp(opCode));

        switch (opCode) {
        case spv::OpTypeOpaque:       // TODO: disable until we compare the literal strings.
        case spv::OpTypeQueue:        return false;
        case spv::OpTypeEvent:        // fall through...
        case spv::OpTypeDeviceEvent:  // ...
        case spv::OpTypeReserveId:    return false;
            // for samplers, we don't handle the optional parameters yet
        case spv::OpTypeSampler:      return cmpLiteral() && cmpConst() && cmpSubType() && wordCount == 8;
        default:                      return cmpLiteral() && cmpConst() && cmpSubType();
        }
    }

    // Look for an equivalent type in the globalTypes map
    spv::Id spirvbin_t::findType(const spirvbin_t::globaltypes_t& globalTypes, spv::Id lt) const
    {
        // Try a recursive type match on each in turn, and return a match if we find one
        for (const auto& gt : globalTypes)
            if (matchType(globalTypes, lt, gt.first))
                return gt.first;

        return spv::NoType;
    }
#endif // NOTDEF

    // Return start position in SPV of given Id.  error if not found.
    unsigned spirvbin_t::idPos(spv::Id id) const
    {
        const auto tid_it = idPosR.find(id);
        if (tid_it == idPosR.end()) {
            error("ID not found");
            return 0;
        }

        return tid_it->second;
    }

    // Hash types to canonical values.  This can return ID collisions (it's a bit
    // inevitable): it's up to the caller to handle that gracefully.
    std::uint32_t spirvbin_t::hashType(unsigned typeStart) const
    {
        const unsigned wordCount   = asWordCount(typeStart);
        const spv::Op  opCode      = asOpCode(typeStart);

        switch (opCode) {
        case spv::OpTypeVoid:         return 0;
        case spv::OpTypeBool:         return 1;
        case spv::OpTypeInt:          return 3 + (spv[typeStart+3]);
        case spv::OpTypeFloat:        return 5;
        case spv::OpTypeVector:
            return 6 + hashType(idPos(spv[typeStart+2])) * (spv[typeStart+3] - 1);
        case spv::OpTypeMatrix:
            return 30 + hashType(idPos(spv[typeStart+2])) * (spv[typeStart+3] - 1);
        case spv::OpTypeImage:
            return 120 + hashType(idPos(spv[typeStart+2])) +
                spv[typeStart+3] +            // dimensionality
                spv[typeStart+4] * 8 * 16 +   // depth
                spv[typeStart+5] * 4 * 16 +   // arrayed
                spv[typeStart+6] * 2 * 16 +   // multisampled
                spv[typeStart+7] * 1 * 16;    // format
        case spv::OpTypeSampler:
            return 500;
        case spv::OpTypeSampledImage:
            return 502;
        case spv::OpTypeArray:
            return 501 + hashType(idPos(spv[typeStart+2])) * spv[typeStart+3];
        case spv::OpTypeRuntimeArray:
            return 5000  + hashType(idPos(spv[typeStart+2]));
        case spv::OpTypeStruct:
            {
                std::uint32_t hash = 10000;
                for (unsigned w=2; w < wordCount; ++w)
                    hash += w * hashType(idPos(spv[typeStart+w]));
                return hash;
            }

        case spv::OpTypeOpaque:         return 6000 + spv[typeStart+2];
        case spv::OpTypePointer:        return 100000  + hashType(idPos(spv[typeStart+3]));
        case spv::OpTypeFunction:
            {
                std::uint32_t hash = 200000;
                for (unsigned w=2; w < wordCount; ++w)
                    hash += w * hashType(idPos(spv[typeStart+w]));
                return hash;
            }

        case spv::OpTypeEvent:           return 300000;
        case spv::OpTypeDeviceEvent:     return 300001;
        case spv::OpTypeReserveId:       return 300002;
        case spv::OpTypeQueue:           return 300003;
        case spv::OpTypePipe:            return 300004;
        case spv::OpConstantTrue:        return 300007;
        case spv::OpConstantFalse:       return 300008;
        case spv::OpConstantComposite:
            {
                std::uint32_t hash = 300011 + hashType(idPos(spv[typeStart+1]));
                for (unsigned w=3; w < wordCount; ++w)
                    hash += w * hashType(idPos(spv[typeStart+w]));
                return hash;
            }
        case spv::OpConstant:
            {
                std::uint32_t hash = 400011 + hashType(idPos(spv[typeStart+1]));
                for (unsigned w=3; w < wordCount; ++w)
                    hash += w * spv[typeStart+w];
                return hash;
            }
        case spv::OpConstantNull:
            {
                std::uint32_t hash = 500009 + hashType(idPos(spv[typeStart+1]));
                return hash;
            }
        case spv::OpConstantSampler:
            {
                std::uint32_t hash = 600011 + hashType(idPos(spv[typeStart+1]));
                for (unsigned w=3; w < wordCount; ++w)
                    hash += w * spv[typeStart+w];
                return hash;
            }

        default:
            error("unknown type opcode");
            return 0;
        }
    }

    void spirvbin_t::mapTypeConst()
    {
        globaltypes_t globalTypeMap;

        msg(3, 2, std::string("Remapping Consts & Types: "));

        static const std::uint32_t softTypeIdLimit = 3011; // small prime.  TODO: get from options
        static const std::uint32_t firstMappedID   = 8;    // offset into ID space

        for (auto& typeStart : typeConstPos) {
            const spv::Id       resId     = asTypeConstId(typeStart);
            const std::uint32_t hashval   = hashType(typeStart);

            if (errorLatch)
                return;

            if (isOldIdUnmapped(resId)) {
                localId(resId, nextUnusedId(hashval % softTypeIdLimit + firstMappedID));
                if (errorLatch)
                    return;
            }
        }
    }

    // Strip a single binary by removing ranges given in stripRange
    void spirvbin_t::strip()
    {
        if (stripRange.empty()) // nothing to do
            return;

        // Sort strip ranges in order of traversal
        std::sort(stripRange.begin(), stripRange.end());

        // Allocate a new binary big enough to hold old binary
        // We'll step this iterator through the strip ranges as we go through the binary
        auto strip_it = stripRange.begin();

        int strippedPos = 0;
        for (unsigned word = 0; word < unsigned(spv.size()); ++word) {
            while (strip_it != stripRange.end() && word >= strip_it->second)
                ++strip_it;

            if (strip_it == stripRange.end() || word < strip_it->first || word >= strip_it->second)
                spv[strippedPos++] = spv[word];
        }

        spv.resize(strippedPos);
        stripRange.clear();

        buildLocalMaps();
    }

    // Strip a single binary by removing ranges given in stripRange
    void spirvbin_t::remap(std::uint32_t opts)
    {
        options = opts;

        // Set up opcode tables from SpvDoc
        spv::Parameterize();

        validate();       // validate header
        buildLocalMaps(); // build ID maps

        msg(3, 4, std::string("ID bound: ") + std::to_string(bound()));

        if (options & STRIP)         stripDebug();
        if (errorLatch) return;

        strip();        // strip out data we decided to eliminate
        if (errorLatch) return;

        if (options & OPT_LOADSTORE) optLoadStore();
        if (errorLatch) return;

        if (options & OPT_FWD_LS)    forwardLoadStores();
        if (errorLatch) return;

        if (options & DCE_FUNCS)     dceFuncs();
        if (errorLatch) return;

        if (options & DCE_VARS)      dceVars();
        if (errorLatch) return;

        if (options & DCE_TYPES)     dceTypes();
        if (errorLatch) return;

        strip();         // strip out data we decided to eliminate
        if (errorLatch) return;

        stripDeadRefs(); // remove references to things we DCEed
        if (errorLatch) return;

        // after the last strip, we must clean any debug info referring to now-deleted data

        if (options & MAP_TYPES)     mapTypeConst();
        if (errorLatch) return;

        if (options & MAP_NAMES)     mapNames();
        if (errorLatch) return;

        if (options & MAP_FUNCS)     mapFnBodies();
        if (errorLatch) return;

        if (options & MAP_ALL) {
            mapRemainder(); // map any unmapped IDs
            if (errorLatch) return;

            applyMap();     // Now remap each shader to the new IDs we've come up with
            if (errorLatch) return;
        }
    }

    // remap from a memory image
    void spirvbin_t::remap(std::vector<std::uint32_t>& in_spv, std::uint32_t opts)
    {
        spv.swap(in_spv);
        remap(opts);
        spv.swap(in_spv);
    }

} // namespace SPV

#endif // defined (use_cpp11)

