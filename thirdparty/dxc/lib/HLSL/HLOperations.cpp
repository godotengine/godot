///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLOperations.cpp                                                          //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Implementation of DXIL operations.                                        //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/HLSL/HLOperations.h"
#include "dxc/HlslIntrinsicOp.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

using namespace hlsl;
using namespace llvm;

namespace hlsl {

const char HLPrefixStr [] = "dx.hl";
const char * const HLPrefix = HLPrefixStr;
static const char HLLowerStrategyStr[] = "dx.hlls";
static const char * const HLLowerStrategy = HLLowerStrategyStr;

static const char HLWaveSensitiveStr[] = "dx.wave-sensitive";
static const char * const HLWaveSensitive = HLWaveSensitiveStr;

static StringRef HLOpcodeGroupNames[]{
    "notHLDXIL",   // NotHL,
    "<ext>",       // HLExtIntrinsic - should always refer through extension
    "op",          // HLIntrinsic,
    "cast",        // HLCast,
    "init",        // HLInit,
    "binop",       // HLBinOp,
    "unop",        // HLUnOp,
    "subscript",   // HLSubscript,
    "matldst",     // HLMatLoadStore,
    "select",      // HLSelect,
    "createhandle",// HLCreateHandle,
    "annotatehandle" // HLAnnotateHandle,
    "numOfHLDXIL", // NumOfHLOps
};

static StringRef HLOpcodeGroupFullNames[]{
    "notHLDXIL",       // NotHL,
    "<ext>",           // HLExtIntrinsic - should aways refer through extension
    "dx.hl.op",        // HLIntrinsic,
    "dx.hl.cast",      // HLCast,
    "dx.hl.init",      // HLInit,
    "dx.hl.binop",     // HLBinOp,
    "dx.hl.unop",      // HLUnOp,
    "dx.hl.subscript", // HLSubscript,
    "dx.hl.matldst",   // HLMatLoadStore,
    "dx.hl.select",    // HLSelect,
    "dx.hl.createhandle",  // HLCreateHandle,
    "dx.hl.annotatehandle",      // HLAnnotateHandle,
    "numOfHLDXIL",     // NumOfHLOps
};

static HLOpcodeGroup GetHLOpcodeGroupInternal(StringRef group) {
  if (!group.empty()) {
    switch (group[0]) {
    case 'o': // op
      return HLOpcodeGroup::HLIntrinsic;
    case 'c': // cast
      switch (group[1]) {
      case 'a': // cast
        return HLOpcodeGroup::HLCast;
      case 'r': // createhandle
        return HLOpcodeGroup::HLCreateHandle;
      }
    case 'i': // init
      return HLOpcodeGroup::HLInit;
    case 'b': // binaryOp
      return HLOpcodeGroup::HLBinOp;
    case 'u': // unaryOp
      return HLOpcodeGroup::HLUnOp;
    case 's': // subscript
      switch (group[1]) {
      case 'u':
        return HLOpcodeGroup::HLSubscript;
      case 'e':
        return HLOpcodeGroup::HLSelect;
      }
    case 'm': // matldst
      return HLOpcodeGroup::HLMatLoadStore;
    case 'a': // annotatehandle
      return HLOpcodeGroup::HLAnnotateHandle;
    }
  }
  return HLOpcodeGroup::NotHL;
}
// GetHLOpGroup by function name.
HLOpcodeGroup GetHLOpcodeGroupByName(const Function *F) {
  StringRef name = F->getName();

  if (!name.startswith(HLPrefix)) {
    // This could be an external intrinsic, but this function
    // won't recognize those as such. Use GetHLOpcodeGroupByName
    // to make that distinction.
    return HLOpcodeGroup::NotHL;
  }

  const unsigned prefixSize = sizeof(HLPrefixStr);

  StringRef group = name.substr(prefixSize);
  return GetHLOpcodeGroupInternal(group);
}

HLOpcodeGroup GetHLOpcodeGroup(llvm::Function *F) {
  llvm::StringRef name = GetHLOpcodeGroupNameByAttr(F);
  HLOpcodeGroup result = GetHLOpcodeGroupInternal(name);
  if (result == HLOpcodeGroup::NotHL) {
    result = name.empty() ? result : HLOpcodeGroup::HLExtIntrinsic;
  }
  if (result == HLOpcodeGroup::NotHL) {
    result = GetHLOpcodeGroupByName(F);
  }
  return result;
}

llvm::StringRef GetHLOpcodeGroupNameByAttr(llvm::Function *F) {
  Attribute groupAttr = F->getFnAttribute(hlsl::HLPrefix);
  StringRef group = groupAttr.getValueAsString();
  return group;
}

StringRef GetHLOpcodeGroupName(HLOpcodeGroup op) {
  switch (op) {
  case HLOpcodeGroup::HLCast:
  case HLOpcodeGroup::HLInit:
  case HLOpcodeGroup::HLBinOp:
  case HLOpcodeGroup::HLUnOp:
  case HLOpcodeGroup::HLIntrinsic:
  case HLOpcodeGroup::HLSubscript:
  case HLOpcodeGroup::HLMatLoadStore:
  case HLOpcodeGroup::HLSelect:
  case HLOpcodeGroup::HLCreateHandle:
  case HLOpcodeGroup::HLAnnotateHandle:
    return HLOpcodeGroupNames[static_cast<unsigned>(op)];
  default:
    llvm_unreachable("invalid op");
    
    return "";
  }
}
StringRef GetHLOpcodeGroupFullName(HLOpcodeGroup op) {
  switch (op) {
  case HLOpcodeGroup::HLCast:
  case HLOpcodeGroup::HLInit:
  case HLOpcodeGroup::HLBinOp:
  case HLOpcodeGroup::HLUnOp:
  case HLOpcodeGroup::HLIntrinsic:
  case HLOpcodeGroup::HLSubscript:
  case HLOpcodeGroup::HLMatLoadStore:
  case HLOpcodeGroup::HLSelect:
  case HLOpcodeGroup::HLCreateHandle:
  case HLOpcodeGroup::HLAnnotateHandle:
    return HLOpcodeGroupFullNames[static_cast<unsigned>(op)];
  default:
    llvm_unreachable("invalid op");
    return "";
  }
}

llvm::StringRef GetHLOpcodeName(HLUnaryOpcode Op) {
  switch (Op) {
  case HLUnaryOpcode::PostInc: return "++";
  case HLUnaryOpcode::PostDec: return "--";
  case HLUnaryOpcode::PreInc:  return "++";
  case HLUnaryOpcode::PreDec:  return "--";
  case HLUnaryOpcode::Plus:    return "+";
  case HLUnaryOpcode::Minus:   return "-";
  case HLUnaryOpcode::Not:     return "~";
  case HLUnaryOpcode::LNot:    return "!";
  case HLUnaryOpcode::Invalid:
  case HLUnaryOpcode::NumOfUO:
    // Invalid Unary Ops
    break;
  }
  llvm_unreachable("Unknown unary operator");

}

llvm::StringRef GetHLOpcodeName(HLBinaryOpcode Op) {
  switch (Op) {
  case HLBinaryOpcode::Mul:       return "*";
  case HLBinaryOpcode::UDiv:
  case HLBinaryOpcode::Div:       return "/";
  case HLBinaryOpcode::URem:
  case HLBinaryOpcode::Rem:       return "%";
  case HLBinaryOpcode::Add:       return "+";
  case HLBinaryOpcode::Sub:       return "-";
  case HLBinaryOpcode::Shl:       return "<<";
  case HLBinaryOpcode::UShr:
  case HLBinaryOpcode::Shr:       return ">>";
  case HLBinaryOpcode::ULT:
  case HLBinaryOpcode::LT:        return "<";
  case HLBinaryOpcode::UGT:
  case HLBinaryOpcode::GT:        return ">";
  case HLBinaryOpcode::ULE:
  case HLBinaryOpcode::LE:        return "<=";
  case HLBinaryOpcode::UGE:
  case HLBinaryOpcode::GE:        return ">=";
  case HLBinaryOpcode::EQ:        return "==";
  case HLBinaryOpcode::NE:        return "!=";
  case HLBinaryOpcode::And:       return "&";
  case HLBinaryOpcode::Xor:       return "^";
  case HLBinaryOpcode::Or:        return "|";
  case HLBinaryOpcode::LAnd:      return "&&";
  case HLBinaryOpcode::LOr:       return "||";
  case HLBinaryOpcode::Invalid:
  case HLBinaryOpcode::NumOfBO:
    // Invalid Binary Ops
    break;
  }

  llvm_unreachable("Invalid OpCode!");
}

llvm::StringRef GetHLOpcodeName(HLSubscriptOpcode Op) {
  switch (Op) {
  case HLSubscriptOpcode::DefaultSubscript:
    return "[]";
  case HLSubscriptOpcode::ColMatSubscript:
    return "colMajor[]";
  case HLSubscriptOpcode::RowMatSubscript:
    return "rowMajor[]";
  case HLSubscriptOpcode::ColMatElement:
    return "colMajor_m";
  case HLSubscriptOpcode::RowMatElement:
    return "rowMajor_m";
  case HLSubscriptOpcode::DoubleSubscript:
    return "[][]";
  case HLSubscriptOpcode::CBufferSubscript:
    return "cb";
  case HLSubscriptOpcode::VectorSubscript:
    return "vector[]";
  }
  return "";
}

llvm::StringRef GetHLOpcodeName(HLCastOpcode Op) {
  switch (Op) {
  case HLCastOpcode::DefaultCast:
    return "default";
  case HLCastOpcode::ToUnsignedCast:
    return "toUnsigned";
  case HLCastOpcode::FromUnsignedCast:
    return "fromUnsigned";
  case HLCastOpcode::UnsignedUnsignedCast:
    return "unsignedUnsigned";
  case HLCastOpcode::ColMatrixToVecCast:
    return "colMatToVec";
  case HLCastOpcode::RowMatrixToVecCast:
    return "rowMatToVec";
  case HLCastOpcode::ColMatrixToRowMatrix:
    return "colMatToRowMat";
  case HLCastOpcode::RowMatrixToColMatrix:
    return "rowMatToColMat";
  case HLCastOpcode::HandleToResCast:
    return "handleToRes";
  }
  return "";
}

llvm::StringRef GetHLOpcodeName(HLMatLoadStoreOpcode Op) {
  switch (Op) {
  case HLMatLoadStoreOpcode::ColMatLoad:
    return "colLoad";
  case HLMatLoadStoreOpcode::ColMatStore:
    return "colStore";
  case HLMatLoadStoreOpcode::RowMatLoad:
    return "rowLoad";
  case HLMatLoadStoreOpcode::RowMatStore:
    return "rowStore";
  }
  llvm_unreachable("invalid matrix load store operator");
}

StringRef GetHLLowerStrategy(Function *F) {
  llvm::Attribute A = F->getFnAttribute(HLLowerStrategy);
  llvm::StringRef LowerStrategy = A.getValueAsString();
  return LowerStrategy;
}

void SetHLLowerStrategy(Function *F, StringRef S) {
  F->addFnAttr(HLLowerStrategy, S);
}

// Set function attribute indicating wave-sensitivity
void SetHLWaveSensitive(Function *F) {
  F->addFnAttr(HLWaveSensitive, "y");
}

// Return if this Function is dependent on other wave members indicated by attribute
bool IsHLWaveSensitive(Function *F) {
  AttributeSet attrSet = F->getAttributes();
  return attrSet.hasAttribute(AttributeSet::FunctionIndex, HLWaveSensitive);
}

std::string GetHLFullName(HLOpcodeGroup op, unsigned opcode) {
  assert(op != HLOpcodeGroup::HLExtIntrinsic && "else table name should be used");
  std::string opName = GetHLOpcodeGroupFullName(op).str() + ".";

  switch (op) {
  case HLOpcodeGroup::HLBinOp: {
    HLBinaryOpcode binOp = static_cast<HLBinaryOpcode>(opcode);
    return opName + GetHLOpcodeName(binOp).str();
  }
  case HLOpcodeGroup::HLUnOp: {
    HLUnaryOpcode unOp = static_cast<HLUnaryOpcode>(opcode);
    return opName + GetHLOpcodeName(unOp).str();
  }
  case HLOpcodeGroup::HLIntrinsic: {
    // intrinsic with same signature will share the funciton now
    // The opcode is in arg0.
    return opName;
  }
  case HLOpcodeGroup::HLMatLoadStore: {
    HLMatLoadStoreOpcode matOp = static_cast<HLMatLoadStoreOpcode>(opcode);
    return opName + GetHLOpcodeName(matOp).str();
  }
  case HLOpcodeGroup::HLSubscript: {
    HLSubscriptOpcode subOp = static_cast<HLSubscriptOpcode>(opcode);
    return opName + GetHLOpcodeName(subOp).str();
  }
  case HLOpcodeGroup::HLCast: {
    HLCastOpcode castOp = static_cast<HLCastOpcode>(opcode);
    return opName + GetHLOpcodeName(castOp).str();
  }
  default:
    return opName;
  }
}

// Get opcode from arg0 of function call.
unsigned  GetHLOpcode(const CallInst *CI) {
  Value *idArg = CI->getArgOperand(HLOperandIndex::kOpcodeIdx);
  Constant *idConst = cast<Constant>(idArg);
  return idConst->getUniqueInteger().getLimitedValue();
}

unsigned  GetRowMajorOpcode(HLOpcodeGroup group, unsigned opcode) {
  switch (group) {
  case HLOpcodeGroup::HLMatLoadStore: {
    HLMatLoadStoreOpcode matOp = static_cast<HLMatLoadStoreOpcode>(opcode);
    switch (matOp) {
    case HLMatLoadStoreOpcode::ColMatLoad:
      return static_cast<unsigned>(HLMatLoadStoreOpcode::RowMatLoad);
    case HLMatLoadStoreOpcode::ColMatStore:
      return static_cast<unsigned>(HLMatLoadStoreOpcode::RowMatStore);
    default:
      return opcode;
    }
  } break;
  case HLOpcodeGroup::HLSubscript: {
    HLSubscriptOpcode subOp = static_cast<HLSubscriptOpcode>(opcode);
    switch (subOp) {
    case HLSubscriptOpcode::ColMatElement:
      return static_cast<unsigned>(HLSubscriptOpcode::RowMatElement);
    case HLSubscriptOpcode::ColMatSubscript:
      return static_cast<unsigned>(HLSubscriptOpcode::RowMatSubscript);
    default:
      return opcode;
    }
  } break;
  default:
    return opcode;
  }
}

unsigned GetUnsignedOpcode(unsigned opcode) {
  return GetUnsignedIntrinsicOpcode(static_cast<IntrinsicOp>(opcode));
}

// For HLBinaryOpcode
bool HasUnsignedOpcode(HLBinaryOpcode opcode) {
  switch (opcode) {
  case HLBinaryOpcode::Div:
  case HLBinaryOpcode::Rem:
  case HLBinaryOpcode::Shr:
  case HLBinaryOpcode::LT:
  case HLBinaryOpcode::GT:
  case HLBinaryOpcode::LE:
  case HLBinaryOpcode::GE:
    return true;
  default:
    return false;
  }
}

HLBinaryOpcode GetUnsignedOpcode(HLBinaryOpcode opcode) {
  switch (opcode) {
  case HLBinaryOpcode::Div:
    return HLBinaryOpcode::UDiv;
  case HLBinaryOpcode::Rem:
    return HLBinaryOpcode::URem;
  case HLBinaryOpcode::Shr:
    return HLBinaryOpcode::UShr;
  case HLBinaryOpcode::LT:
    return HLBinaryOpcode::ULT;
  case HLBinaryOpcode::GT:
    return HLBinaryOpcode::UGT;
  case HLBinaryOpcode::LE:
    return HLBinaryOpcode::ULE;
  case HLBinaryOpcode::GE:
    return HLBinaryOpcode::UGE;
  default:
    return opcode;
  }
}

static void SetHLFunctionAttribute(Function *F, HLOpcodeGroup group,
                                       unsigned opcode) {
  F->addFnAttr(Attribute::NoUnwind);

  switch (group) {
  case HLOpcodeGroup::HLUnOp:
  case HLOpcodeGroup::HLBinOp:
  case HLOpcodeGroup::HLCast:
  case HLOpcodeGroup::HLSubscript:
    if (!F->hasFnAttribute(Attribute::ReadNone)) {
      F->addFnAttr(Attribute::ReadNone);
    }
    break;
  case HLOpcodeGroup::HLInit:
    if (!F->hasFnAttribute(Attribute::ReadNone))
      if (!F->getReturnType()->isVoidTy()) {
        F->addFnAttr(Attribute::ReadNone);
      }
    break;
  case HLOpcodeGroup::HLMatLoadStore: {
    HLMatLoadStoreOpcode matOp = static_cast<HLMatLoadStoreOpcode>(opcode);
    if (matOp == HLMatLoadStoreOpcode::ColMatLoad ||
        matOp == HLMatLoadStoreOpcode::RowMatLoad)
      if (!F->hasFnAttribute(Attribute::ReadOnly)) {
        F->addFnAttr(Attribute::ReadOnly);
      }
  } break;
  case HLOpcodeGroup::HLCreateHandle: {
    F->addFnAttr(Attribute::ReadNone);
  } break;
  case HLOpcodeGroup::HLAnnotateHandle: {
    F->addFnAttr(Attribute::ReadNone);
  } break;
  case HLOpcodeGroup::HLIntrinsic: {
    IntrinsicOp intrinsicOp = static_cast<IntrinsicOp>(opcode);
    switch (intrinsicOp) {
    default:
      break;
    case IntrinsicOp::IOP_DeviceMemoryBarrierWithGroupSync:
    case IntrinsicOp::IOP_DeviceMemoryBarrier:
    case IntrinsicOp::IOP_GroupMemoryBarrierWithGroupSync:
    case IntrinsicOp::IOP_GroupMemoryBarrier:
    case IntrinsicOp::IOP_AllMemoryBarrierWithGroupSync:
    case IntrinsicOp::IOP_AllMemoryBarrier:
      F->addFnAttr(Attribute::NoDuplicate);
      break;
    }
  } break;
  case HLOpcodeGroup::NotHL:
  case HLOpcodeGroup::HLExtIntrinsic:
  case HLOpcodeGroup::HLSelect:
  case HLOpcodeGroup::NumOfHLOps:
    // No default attributes for these opcodes.
    break;
  }
}


Function *GetOrCreateHLFunction(Module &M, FunctionType *funcTy,
                                HLOpcodeGroup group, unsigned opcode) {
  AttributeSet attribs;
  return GetOrCreateHLFunction(M, funcTy, group, nullptr, nullptr, opcode, attribs);
}

Function *GetOrCreateHLFunction(Module &M, FunctionType *funcTy,
                                HLOpcodeGroup group, StringRef *groupName,
                                StringRef *fnName, unsigned opcode) {
  AttributeSet attribs;
  return GetOrCreateHLFunction(M, funcTy, group, groupName, fnName, opcode, attribs);
}

Function *GetOrCreateHLFunction(Module &M, FunctionType *funcTy,
                                HLOpcodeGroup group, unsigned opcode,
                                const AttributeSet &attribs) {
  return GetOrCreateHLFunction(M, funcTy, group, nullptr, nullptr, opcode, attribs);
}

Function *GetOrCreateHLFunction(Module &M, FunctionType *funcTy,
                                HLOpcodeGroup group, StringRef *groupName,
                                StringRef *fnName, unsigned opcode,
                                const AttributeSet &attribs) {
  std::string mangledName;
  raw_string_ostream mangledNameStr(mangledName);
  if (group == HLOpcodeGroup::HLExtIntrinsic) {
    assert(groupName && "else intrinsic should have been rejected");
    assert(fnName && "else intrinsic should have been rejected");
    mangledNameStr << *groupName;
    mangledNameStr << '.';
    mangledNameStr << *fnName;
  }
  else {
    mangledNameStr << GetHLFullName(group, opcode);
    // Need to add wave sensitivity to name to prevent clashes with non-wave intrinsic
    if(attribs.hasAttribute(AttributeSet::FunctionIndex, HLWaveSensitive))
        mangledNameStr << "wave";
    mangledNameStr << '.';
    funcTy->print(mangledNameStr);
  }

  mangledNameStr.flush();

  Function *F = cast<Function>(M.getOrInsertFunction(mangledName, funcTy));
  if (group == HLOpcodeGroup::HLExtIntrinsic) {
    F->addFnAttr(hlsl::HLPrefix, *groupName);
  }

  SetHLFunctionAttribute(F, group, opcode);

  // Copy attributes
  if (attribs.hasAttribute(AttributeSet::FunctionIndex, Attribute::ReadNone))
    F->addFnAttr(Attribute::ReadNone);
  if (attribs.hasAttribute(AttributeSet::FunctionIndex, Attribute::ReadOnly))
    F->addFnAttr(Attribute::ReadOnly);
  if (attribs.hasAttribute(AttributeSet::FunctionIndex, HLWaveSensitive))
    F->addFnAttr(HLWaveSensitive, "y");

  return F;
}

// HLFunction with body cannot share with HLFunction without body.
// So need add name.
Function *GetOrCreateHLFunctionWithBody(Module &M, FunctionType *funcTy,
                                        HLOpcodeGroup group, unsigned opcode,
                                        StringRef name) {
  std::string operatorName = GetHLFullName(group, opcode);
  std::string mangledName = operatorName + "." + name.str();
  raw_string_ostream mangledNameStr(mangledName);
  funcTy->print(mangledNameStr);
  mangledNameStr.flush();

  Function *F = cast<Function>(M.getOrInsertFunction(mangledName, funcTy));

  SetHLFunctionAttribute(F, group, opcode);

  F->setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);

  return F;
}

Value *callHLFunction(Module &Module, HLOpcodeGroup OpcodeGroup, unsigned Opcode,
      Type *RetTy, ArrayRef<Value*> Args, IRBuilder<> &Builder) {
  AttributeSet attribs;
  return callHLFunction(Module, OpcodeGroup, Opcode, RetTy, Args, attribs, Builder);
}

Value *callHLFunction(Module &Module, HLOpcodeGroup OpcodeGroup, unsigned Opcode,
      Type *RetTy, ArrayRef<Value*> Args, const AttributeSet &attribs, IRBuilder<> &Builder) {
  SmallVector<Type*, 4> ArgTys;
  ArgTys.reserve(Args.size());
  for (Value *Arg : Args)
    ArgTys.emplace_back(Arg->getType());

  FunctionType *FuncTy = FunctionType::get(RetTy, ArgTys, /* isVarArg */ false);
  Function *Func = GetOrCreateHLFunction(Module, FuncTy, OpcodeGroup, Opcode, attribs);

  return Builder.CreateCall(Func, Args);
}

} // namespace hlsl
