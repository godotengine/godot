//
// Copyright (C) 2018 Google, Inc.
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
// Post-processing for SPIR-V IR, in internal form, not standard binary form.
//

#include <cassert>
#include <cstdlib>

#include <unordered_map>
#include <unordered_set>
#include <algorithm>

#include "SPIRV/spvIR.h"
#include "SpvBuilder.h"
#include "spirv.hpp11"
#include "spvUtil.h"

namespace spv {
    #include "GLSL.std.450.h"
    #include "GLSL.ext.KHR.h"
    #include "GLSL.ext.EXT.h"
    #include "GLSL.ext.AMD.h"
    #include "GLSL.ext.NV.h"
    #include "GLSL.ext.ARM.h"
    #include "GLSL.ext.QCOM.h"
}

namespace spv {

// Hook to visit each operand type and result type of an instruction.
// Will be called multiple times for one instruction, once for each typed
// operand and the result.
void Builder::postProcessType(const Instruction& inst, Id typeId)
{
    // Characterize the type being questioned
    Op basicTypeOp = getMostBasicTypeClass(typeId);
    int width = 0;
    if (basicTypeOp == Op::OpTypeFloat || basicTypeOp == Op::OpTypeInt)
        width = getScalarTypeWidth(typeId);

    // Do opcode-specific checks
    switch (inst.getOpCode()) {
    case Op::OpLoad:
    case Op::OpStore:
        if (basicTypeOp == Op::OpTypeStruct) {
            if (containsType(typeId, Op::OpTypeInt, 8))
                addCapability(Capability::Int8);
            if (containsType(typeId, Op::OpTypeInt, 16))
                addCapability(Capability::Int16);
            if (containsType(typeId, Op::OpTypeFloat, 16))
                addCapability(Capability::Float16);
        } else {
            StorageClass storageClass = getStorageClass(inst.getIdOperand(0));
            if (width == 8) {
                switch (storageClass) {
                case StorageClass::PhysicalStorageBufferEXT:
                case StorageClass::Uniform:
                case StorageClass::StorageBuffer:
                case StorageClass::PushConstant:
                    break;
                default:
                    addCapability(Capability::Int8);
                    break;
                }
            } else if (width == 16) {
                switch (storageClass) {
                case StorageClass::PhysicalStorageBufferEXT:
                case StorageClass::Uniform:
                case StorageClass::StorageBuffer:
                case StorageClass::PushConstant:
                case StorageClass::Input:
                case StorageClass::Output:
                    break;
                default:
                    if (basicTypeOp == Op::OpTypeInt)
                        addCapability(Capability::Int16);
                    if (basicTypeOp == Op::OpTypeFloat)
                        addCapability(Capability::Float16);
                    break;
                }
            }
        }
        break;
    case Op::OpCopyObject:
        break;
    case Op::OpFConvert:
    case Op::OpSConvert:
    case Op::OpUConvert:
        // Look for any 8/16-bit storage capabilities. If there are none, assume that
        // the convert instruction requires the Float16/Int8/16 capability.
        if (containsType(typeId, Op::OpTypeFloat, 16) || containsType(typeId, Op::OpTypeInt, 16)) {
            bool foundStorage = false;
            for (auto it = capabilities.begin(); it != capabilities.end(); ++it) {
                spv::Capability cap = *it;
                if (cap == spv::Capability::StorageInputOutput16 ||
                    cap == spv::Capability::StoragePushConstant16 ||
                    cap == spv::Capability::StorageUniformBufferBlock16 ||
                    cap == spv::Capability::StorageUniform16) {
                    foundStorage = true;
                    break;
                }
            }
            if (!foundStorage) {
                if (containsType(typeId, Op::OpTypeFloat, 16))
                    addCapability(Capability::Float16);
                if (containsType(typeId, Op::OpTypeInt, 16))
                    addCapability(Capability::Int16);
            }
        }
        if (containsType(typeId, Op::OpTypeInt, 8)) {
            bool foundStorage = false;
            for (auto it = capabilities.begin(); it != capabilities.end(); ++it) {
                spv::Capability cap = *it;
                if (cap == spv::Capability::StoragePushConstant8 ||
                    cap == spv::Capability::UniformAndStorageBuffer8BitAccess ||
                    cap == spv::Capability::StorageBuffer8BitAccess) {
                    foundStorage = true;
                    break;
                }
            }
            if (!foundStorage) {
                addCapability(Capability::Int8);
            }
        }
        break;
    case Op::OpExtInst:
        switch (inst.getImmediateOperand(1)) {
        case GLSLstd450Frexp:
        case GLSLstd450FrexpStruct:
            if (getSpvVersion() < spv::Spv_1_3 && containsType(typeId, Op::OpTypeInt, 16))
                addExtension(spv::E_SPV_AMD_gpu_shader_int16);
            break;
        case GLSLstd450InterpolateAtCentroid:
        case GLSLstd450InterpolateAtSample:
        case GLSLstd450InterpolateAtOffset:
            if (getSpvVersion() < spv::Spv_1_3 && containsType(typeId, Op::OpTypeFloat, 16))
                addExtension(spv::E_SPV_AMD_gpu_shader_half_float);
            break;
        default:
            break;
        }
        break;
    case Op::OpAccessChain:
    case Op::OpPtrAccessChain:
        if (isPointerType(typeId))
            break;
        if (basicTypeOp == Op::OpTypeInt) {
            if (width == 16)
                addCapability(Capability::Int16);
            else if (width == 8)
                addCapability(Capability::Int8);
        }
        break;
    default:
        if (basicTypeOp == Op::OpTypeInt) {
            if (width == 16)
                addCapability(Capability::Int16);
            else if (width == 8)
                addCapability(Capability::Int8);
            else if (width == 64)
                addCapability(Capability::Int64);
        } else if (basicTypeOp == Op::OpTypeFloat) {
            if (width == 16)
                addCapability(Capability::Float16);
            else if (width == 64)
                addCapability(Capability::Float64);
        }
        break;
    }
}

unsigned int Builder::postProcessGetLargestScalarSize(const Instruction& type)
{
    switch (type.getOpCode()) {
    case Op::OpTypeBool:
        return 1;
    case Op::OpTypeInt:
    case Op::OpTypeFloat:
        return type.getImmediateOperand(0) / 8;
    case Op::OpTypePointer:
        return 8;
    case Op::OpTypeVector:
    case Op::OpTypeMatrix:
    case Op::OpTypeArray:
    case Op::OpTypeRuntimeArray: {
        const Instruction* elem_type = module.getInstruction(type.getIdOperand(0));
        return postProcessGetLargestScalarSize(*elem_type);
    }
    case Op::OpTypeStruct: {
        unsigned int largest = 0;
        for (int i = 0; i < type.getNumOperands(); ++i) {
            const Instruction* elem_type = module.getInstruction(type.getIdOperand(i));
            unsigned int elem_size = postProcessGetLargestScalarSize(*elem_type);
            largest = std::max(largest, elem_size);
        }
        return largest;
    }
    default:
        return 0;
    }
}

// Called for each instruction that resides in a block.
void Builder::postProcess(Instruction& inst)
{
    // Add capabilities based simply on the opcode.
    switch (inst.getOpCode()) {
    case Op::OpExtInst:
        switch (inst.getImmediateOperand(1)) {
        case GLSLstd450InterpolateAtCentroid:
        case GLSLstd450InterpolateAtSample:
        case GLSLstd450InterpolateAtOffset:
            addCapability(Capability::InterpolationFunction);
            break;
        default:
            break;
        }
        break;
    case Op::OpDPdxFine:
    case Op::OpDPdyFine:
    case Op::OpFwidthFine:
    case Op::OpDPdxCoarse:
    case Op::OpDPdyCoarse:
    case Op::OpFwidthCoarse:
        addCapability(Capability::DerivativeControl);
        break;

    case Op::OpImageQueryLod:
    case Op::OpImageQuerySize:
    case Op::OpImageQuerySizeLod:
    case Op::OpImageQuerySamples:
    case Op::OpImageQueryLevels:
        addCapability(Capability::ImageQuery);
        break;

    case Op::OpGroupNonUniformPartitionNV:
        addExtension(E_SPV_NV_shader_subgroup_partitioned);
        addCapability(Capability::GroupNonUniformPartitionedNV);
        break;

    case Op::OpLoad:
    case Op::OpStore:
        {
            // For any load/store to a PhysicalStorageBufferEXT, walk the accesschain
            // index list to compute the misalignment. The pre-existing alignment value
            // (set via Builder::AccessChain::alignment) only accounts for the base of
            // the reference type and any scalar component selection in the accesschain,
            // and this function computes the rest from the SPIR-V Offset decorations.
            Instruction *accessChain = module.getInstruction(inst.getIdOperand(0));
            if (accessChain->getOpCode() == Op::OpAccessChain) {
                const Instruction* base = module.getInstruction(accessChain->getIdOperand(0));
                // Get the type of the base of the access chain. It must be a pointer type.
                Id typeId = base->getTypeId();
                Instruction *type = module.getInstruction(typeId);
                assert(type->getOpCode() == Op::OpTypePointer);
                if (type->getImmediateOperand(0) != StorageClass::PhysicalStorageBuffer) {
                    break;
                }
                // Get the pointee type.
                typeId = type->getIdOperand(1);
                type = module.getInstruction(typeId);
                // Walk the index list for the access chain. For each index, find any
                // misalignment that can apply when accessing the member/element via
                // Offset/ArrayStride/MatrixStride decorations, and bitwise OR them all
                // together.
                int alignment = 0;
                bool first_struct_elem = false;
                for (int i = 1; i < accessChain->getNumOperands(); ++i) {
                    Instruction *idx = module.getInstruction(accessChain->getIdOperand(i));
                    if (type->getOpCode() == Op::OpTypeStruct) {
                        assert(idx->getOpCode() == Op::OpConstant);
                        unsigned int c = idx->getImmediateOperand(0);

                        const auto function = [&](const std::unique_ptr<Instruction>& decoration) {
                            if (decoration.get()->getOpCode() == Op::OpMemberDecorate &&
                                decoration.get()->getIdOperand(0) == typeId &&
                                decoration.get()->getImmediateOperand(1) == c &&
                                (decoration.get()->getImmediateOperand(2) == Decoration::Offset ||
                                 decoration.get()->getImmediateOperand(2) == Decoration::MatrixStride)) {
                                unsigned int opernad_value = decoration.get()->getImmediateOperand(3);
                                alignment |= opernad_value;
                                if (opernad_value == 0 &&
                                    decoration.get()->getImmediateOperand(2) == Decoration::Offset) {
                                    first_struct_elem = true;
                                }
                            }
                        };
                        std::for_each(decorations.begin(), decorations.end(), function);
                        // get the next member type
                        typeId = type->getIdOperand(c);
                        type = module.getInstruction(typeId);
                    } else if (type->getOpCode() == Op::OpTypeArray ||
                               type->getOpCode() == Op::OpTypeRuntimeArray) {
                        const auto function = [&](const std::unique_ptr<Instruction>& decoration) {
                            if (decoration.get()->getOpCode() == Op::OpDecorate &&
                                decoration.get()->getIdOperand(0) == typeId &&
                                decoration.get()->getImmediateOperand(1) == Decoration::ArrayStride) {
                                alignment |= decoration.get()->getImmediateOperand(2);
                            }
                        };
                        std::for_each(decorations.begin(), decorations.end(), function);
                        // Get the element type
                        typeId = type->getIdOperand(0);
                        type = module.getInstruction(typeId);
                    } else {
                        // Once we get to any non-aggregate type, we're done.
                        break;
                    }
                }
                assert(inst.getNumOperands() >= 3);
                const bool is_store = inst.getOpCode() == Op::OpStore;
                auto const memoryAccess = (MemoryAccessMask)inst.getImmediateOperand(is_store ? 2 : 1);
                assert(anySet(memoryAccess, MemoryAccessMask::Aligned));
                static_cast<void>(memoryAccess);

                // Compute the index of the alignment operand.
                int alignmentIdx = 2;
                if (is_store)
                    alignmentIdx++;
                // Merge new and old (mis)alignment
                alignment |= inst.getImmediateOperand(alignmentIdx);

                if (!is_store) {
                    Instruction* inst_type = module.getInstruction(inst.getTypeId());
                    if (inst_type->getOpCode() == Op::OpTypePointer &&
                        inst_type->getImmediateOperand(0) == StorageClass::PhysicalStorageBuffer) {
                        // This means we are loading a pointer which means need to ensure it is at least 8-byte aligned
                        // See https://github.com/KhronosGroup/glslang/issues/4084
                        // In case the alignment is currently 4, need to ensure it is 8 before grabbing the LSB
                        alignment |= 8;
                        alignment &= 8;
                    }
                }

                // Pick the LSB
                alignment = alignment & ~(alignment & (alignment-1));

                // The edge case we find is when copying a struct to another struct, we never find the alignment anywhere,
                // so in this case, fallback to doing a full size lookup on the type
                if (alignment == 0 && first_struct_elem) {
                    // Quick get the struct type back
                    const Instruction* pointer_type = module.getInstruction(base->getTypeId());
                    const Instruction* struct_type = module.getInstruction(pointer_type->getIdOperand(1));
                    assert(struct_type->getOpCode() == Op::OpTypeStruct);

                    const Instruction* elem_type = module.getInstruction(struct_type->getIdOperand(0));
                    unsigned int largest_scalar = postProcessGetLargestScalarSize(*elem_type);
                    if (largest_scalar != 0) {
                        alignment = largest_scalar;
                    } else {
                        alignment = 16; // fallback if can't determine a godo alignment
                    }
                }
                // update the Aligned operand
                assert(alignment != 0);
                inst.setImmediateOperand(alignmentIdx, alignment);
            }
            break;
        }

    default:
        break;
    }

    // Checks based on type
    if (inst.getTypeId() != NoType)
        postProcessType(inst, inst.getTypeId());
    for (int op = 0; op < inst.getNumOperands(); ++op) {
        if (inst.isIdOperand(op)) {
            // In blocks, these are always result ids, but we are relying on
            // getTypeId() to return NoType for things like OpLabel.
            if (getTypeId(inst.getIdOperand(op)) != NoType)
                postProcessType(inst, getTypeId(inst.getIdOperand(op)));
        }
    }
}

// comment in header
void Builder::postProcessCFG()
{
    // reachableBlocks is the set of blockss reached via control flow, or which are
    // unreachable continue targert or unreachable merge.
    std::unordered_set<const Block*> reachableBlocks;
    std::unordered_map<Block*, Block*> headerForUnreachableContinue;
    std::unordered_set<Block*> unreachableMerges;
    std::unordered_set<Id> unreachableDefinitions;
    // Collect IDs defined in unreachable blocks. For each function, label the
    // reachable blocks first. Then for each unreachable block, collect the
    // result IDs of the instructions in it.
    for (auto fi = module.getFunctions().cbegin(); fi != module.getFunctions().cend(); fi++) {
        Function* f = *fi;
        Block* entry = f->getEntryBlock();
        inReadableOrder(entry,
            [&reachableBlocks, &unreachableMerges, &headerForUnreachableContinue]
            (Block* b, ReachReason why, Block* header) {
               reachableBlocks.insert(b);
               if (why == ReachDeadContinue) headerForUnreachableContinue[b] = header;
               if (why == ReachDeadMerge) unreachableMerges.insert(b);
            });
        for (auto bi = f->getBlocks().cbegin(); bi != f->getBlocks().cend(); bi++) {
            Block* b = *bi;
            if (unreachableMerges.count(b) != 0 || headerForUnreachableContinue.count(b) != 0) {
                auto ii = b->getInstructions().cbegin();
                ++ii; // Keep potential decorations on the label.
                for (; ii != b->getInstructions().cend(); ++ii)
                    unreachableDefinitions.insert(ii->get()->getResultId());
            } else if (reachableBlocks.count(b) == 0) {
                // The normal case for unreachable code.  All definitions are considered dead.
                for (auto ii = b->getInstructions().cbegin(); ii != b->getInstructions().cend(); ++ii)
                    unreachableDefinitions.insert(ii->get()->getResultId());
            }
        }
    }

    // Modify unreachable merge blocks and unreachable continue targets.
    // Delete their contents.
    for (auto mergeIter = unreachableMerges.begin(); mergeIter != unreachableMerges.end(); ++mergeIter) {
        (*mergeIter)->rewriteAsCanonicalUnreachableMerge();
    }
    for (auto continueIter = headerForUnreachableContinue.begin();
         continueIter != headerForUnreachableContinue.end();
         ++continueIter) {
        Block* continue_target = continueIter->first;
        Block* header = continueIter->second;
        continue_target->rewriteAsCanonicalUnreachableContinue(header);
    }

    // Remove unneeded decorations, for unreachable instructions
    for (auto decorationIter = decorations.begin(); decorationIter != decorations.end();) {
        Id decorationId = (*decorationIter)->getIdOperand(0);
        if (unreachableDefinitions.count(decorationId) != 0) {
            decorationIter = decorations.erase(decorationIter);
        } else {
            ++decorationIter;
        }
    }
}

// comment in header
void Builder::postProcessFeatures() {
    // Add per-instruction capabilities, extensions, etc.,

    // Look for any 8/16 bit type in physical storage buffer class, and set the
    // appropriate capability. This happens in createSpvVariable for other storage
    // classes, but there isn't always a variable for physical storage buffer.
    for (int t = 0; t < (int)groupedTypes[enumCast(Op::OpTypePointer)].size(); ++t) {
        Instruction* type = groupedTypes[enumCast(Op::OpTypePointer)][t];
        if (type->getImmediateOperand(0) == (unsigned)StorageClass::PhysicalStorageBufferEXT) {
            if (containsType(type->getIdOperand(1), Op::OpTypeInt, 8)) {
                addIncorporatedExtension(spv::E_SPV_KHR_8bit_storage, spv::Spv_1_5);
                addCapability(spv::Capability::StorageBuffer8BitAccess);
            }
            if (containsType(type->getIdOperand(1), Op::OpTypeInt, 16) ||
                containsType(type->getIdOperand(1), Op::OpTypeFloat, 16)) {
                addIncorporatedExtension(spv::E_SPV_KHR_16bit_storage, spv::Spv_1_3);
                addCapability(spv::Capability::StorageBuffer16BitAccess);
            }
        }
    }

    // process all block-contained instructions
    for (auto fi = module.getFunctions().cbegin(); fi != module.getFunctions().cend(); fi++) {
        Function* f = *fi;
        for (auto bi = f->getBlocks().cbegin(); bi != f->getBlocks().cend(); bi++) {
            Block* b = *bi;
            for (auto ii = b->getInstructions().cbegin(); ii != b->getInstructions().cend(); ii++)
                postProcess(*ii->get());

            // For all local variables that contain pointers to PhysicalStorageBufferEXT, check whether
            // there is an existing restrict/aliased decoration. If we don't find one, add Aliased as the
            // default.
            for (auto vi = b->getLocalVariables().cbegin(); vi != b->getLocalVariables().cend(); vi++) {
                const Instruction& inst = *vi->get();
                Id resultId = inst.getResultId();
                if (containsPhysicalStorageBufferOrArray(getDerefTypeId(resultId))) {
                    bool foundDecoration = false;
                    const auto function = [&](const std::unique_ptr<Instruction>& decoration) {
                        if (decoration.get()->getIdOperand(0) == resultId &&
                            decoration.get()->getOpCode() == Op::OpDecorate &&
                            (decoration.get()->getImmediateOperand(1) == spv::Decoration::AliasedPointerEXT ||
                             decoration.get()->getImmediateOperand(1) == spv::Decoration::RestrictPointerEXT)) {
                            foundDecoration = true;
                        }
                    };
                    std::for_each(decorations.begin(), decorations.end(), function);
                    if (!foundDecoration) {
                        addDecoration(resultId, spv::Decoration::AliasedPointerEXT);
                    }
                }
            }
        }
    }

    // If any Vulkan memory model-specific functionality is used, update the
    // OpMemoryModel to match.
    if (capabilities.find(spv::Capability::VulkanMemoryModelKHR) != capabilities.end()) {
        memoryModel = spv::MemoryModel::VulkanKHR;
        addIncorporatedExtension(spv::E_SPV_KHR_vulkan_memory_model, spv::Spv_1_5);
    }

    // Add Aliased decoration if there's more than one Workgroup Block variable.
    if (capabilities.find(spv::Capability::WorkgroupMemoryExplicitLayoutKHR) != capabilities.end()) {
        assert(entryPoints.size() == 1);
        auto &ep = entryPoints[0];

        std::vector<Id> workgroup_variables;
        for (int i = 0; i < (int)ep->getNumOperands(); i++) {
            if (!ep->isIdOperand(i))
                continue;

            const Id id = ep->getIdOperand(i);
            const Instruction *instr = module.getInstruction(id);
            if (instr->getOpCode() != spv::Op::OpVariable)
                continue;

            if (instr->getImmediateOperand(0) == spv::StorageClass::Workgroup)
                workgroup_variables.push_back(id);
        }

        if (workgroup_variables.size() > 1) {
            for (size_t i = 0; i < workgroup_variables.size(); i++)
                addDecoration(workgroup_variables[i], spv::Decoration::Aliased);
        }
    }
}

// SPIR-V requires that any instruction consuming the result of an OpSampledImage
// be in the same block as the OpSampledImage instruction. This pass goes finds
// uses of OpSampledImage where that is not the case and duplicates the
// OpSampledImage to be immediately before the instruction that consumes it.
// The old OpSampledImage is left in place, potentially with no users.
void Builder::postProcessSamplers()
{
    // first, find all OpSampledImage instructions and store them in a map.
    std::map<Id, Instruction*> sampledImageInstrs;
    for (auto f: module.getFunctions()) {
	for (auto b: f->getBlocks()) {
	    for (auto &i: b->getInstructions()) {
        if (i->getOpCode() == spv::Op::OpSampledImage) {
		    sampledImageInstrs[i->getResultId()] = i.get();
		}
	    }
	}
    }
    // next find all uses of the given ids and rewrite them if needed.
    for (auto f: module.getFunctions()) {
	for (auto b: f->getBlocks()) {
            auto &instrs = b->getInstructions();
            for (size_t idx = 0; idx < instrs.size(); idx++) {
                Instruction *i = instrs[idx].get();
                for (int opnum = 0; opnum < i->getNumOperands(); opnum++) {
                    // Is this operand of the current instruction the result of an OpSampledImage?
                    if (i->isIdOperand(opnum) &&
                        sampledImageInstrs.count(i->getIdOperand(opnum)))
                    {
                        Instruction *opSampImg = sampledImageInstrs[i->getIdOperand(opnum)];
                        if (i->getBlock() != opSampImg->getBlock()) {
                            Instruction *newInstr = new Instruction(getUniqueId(),
                                                                    opSampImg->getTypeId(),
                                                                    spv::Op::OpSampledImage);
                            newInstr->addIdOperand(opSampImg->getIdOperand(0));
                            newInstr->addIdOperand(opSampImg->getIdOperand(1));
                            newInstr->setBlock(b);

                            // rewrite the user of the OpSampledImage to use the new instruction.
                            i->setIdOperand(opnum, newInstr->getResultId());
                            // insert the new OpSampledImage right before the current instruction.
                            instrs.insert(instrs.begin() + idx,
                                    std::unique_ptr<Instruction>(newInstr));
                            idx++;
                        }
                    }
                }
            }
	}
    }
}

// comment in header
void Builder::postProcess(bool compileOnly)
{
    // postProcessCFG needs an entrypoint to determine what is reachable, but if we are not creating an "executable" shader, we don't have an entrypoint
    if (!compileOnly)
        postProcessCFG();

    postProcessFeatures();
    postProcessSamplers();
}

} // end spv namespace
