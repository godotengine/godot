///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// PixPassHelpers.h  														 //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

//#define PIX_DEBUG_DUMP_HELPER

#ifdef PIX_DEBUG_DUMP_HELPER
#include "dxc/Support/Global.h"
#endif

namespace PIXPassHelpers
{
	bool IsAllocateRayQueryInstruction(llvm::Value* Val);
    llvm::CallInst* CreateUAV(hlsl::DxilModule& DM, llvm::IRBuilder<>& Builder,
                                  unsigned int registerId, const char *name);
    llvm::CallInst* CreateHandleForResource(hlsl::DxilModule& DM, llvm::IRBuilder<>& Builder,
        hlsl::DxilResourceBase * resource,
        const char* name);
    llvm::Function* GetEntryFunction(hlsl::DxilModule& DM);
    std::vector<llvm::BasicBlock*> GetAllBlocks(hlsl::DxilModule& DM);
#ifdef PIX_DEBUG_DUMP_HELPER
    void Log(const char* format, ...);
    void LogPartialLine(const char* format, ...);
    void IncreaseLogIndent();
    void DecreaseLogIndent();
    void DumpFullType(llvm::DIType const* type);
#else
    inline void DumpFullType(llvm::DIType const*) {}
    inline void Log(const char* , ...) {}
    inline void LogPartialLine(const char* format, ...) {}
    inline void IncreaseLogIndent() {}
    inline void DecreaseLogIndent() {}
#endif
    class ScopedIndenter
    {
    public:
        ScopedIndenter() { IncreaseLogIndent(); }
        ~ScopedIndenter() { DecreaseLogIndent(); }
    };

    struct ExpandedStruct {
      llvm::Type *ExpandedPayloadStructType = nullptr;
      llvm::Type *ExpandedPayloadStructPtrType = nullptr;
    };

    ExpandedStruct ExpandStructType(llvm::LLVMContext& Ctx,
        llvm::Type* OriginalPayloadStructType);
    void ReplaceAllUsesOfInstructionWithNewValueAndDeleteInstruction(
        llvm::Instruction* Instr, llvm::Value* newValue, llvm::Type* newType);
}