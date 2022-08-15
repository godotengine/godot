//===-- AutoUpgrade.cpp - Implement auto-upgrade helper functions ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the auto-upgrade helper functions.
// This is where deprecated IR intrinsics and other IR features are updated to
// current specifications.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstring>
using namespace llvm;

#if 0 // HLSL Change - remove platform intrinsics
// Upgrade the declarations of the SSE4.1 functions whose arguments have
// changed their type from v4f32 to v2i64.
static bool UpgradeSSE41Function(Function* F, Intrinsic::ID IID,
                                 Function *&NewFn) {
  // Check whether this is an old version of the function, which received
  // v4f32 arguments.
  Type *Arg0Type = F->getFunctionType()->getParamType(0);
  if (Arg0Type != VectorType::get(Type::getFloatTy(F->getContext()), 4))
    return false;

  // Yes, it's old, replace it with new version.
  F->setName(F->getName() + ".old");
  NewFn = Intrinsic::getDeclaration(F->getParent(), IID);
  return true;
}

// Upgrade the declarations of intrinsic functions whose 8-bit immediate mask
// arguments have changed their type from i32 to i8.
static bool UpgradeX86IntrinsicsWith8BitMask(Function *F, Intrinsic::ID IID,
                                             Function *&NewFn) {
  // Check that the last argument is an i32.
  Type *LastArgType = F->getFunctionType()->getParamType(
     F->getFunctionType()->getNumParams() - 1);
  if (!LastArgType->isIntegerTy(32))
    return false;

  // Move this function aside and map down.
  F->setName(F->getName() + ".old");
  NewFn = Intrinsic::getDeclaration(F->getParent(), IID);
  return true;
}
#endif // HLSL Change - remove platform intrinsics

static bool UpgradeIntrinsicFunction1(Function *F, Function *&NewFn) {
  assert(F && "Illegal to upgrade a non-existent Function.");

  // Quickly eliminate it, if it's not a candidate.
  StringRef Name = F->getName();
  if (Name.size() <= 8 || !Name.startswith("llvm."))
    return false;
  Name = Name.substr(5); // Strip off "llvm."

  switch (Name[0]) {
  default: break;
  case 'a': {
    if (Name.startswith("arm.neon.vclz")) {
      Type* args[2] = {
        F->arg_begin()->getType(),
        Type::getInt1Ty(F->getContext())
      };
      // Can't use Intrinsic::getDeclaration here as it adds a ".i1" to
      // the end of the name. Change name from llvm.arm.neon.vclz.* to
      //  llvm.ctlz.*
      FunctionType* fType = FunctionType::get(F->getReturnType(), args, false);
      NewFn = Function::Create(fType, F->getLinkage(),
                               "llvm.ctlz." + Name.substr(14), F->getParent());
      return true;
    }
    if (Name.startswith("arm.neon.vcnt")) {
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::ctpop,
                                        F->arg_begin()->getType());
      return true;
    }
    break;
  }
  case 'c': {
    if (Name.startswith("ctlz.") && F->arg_size() == 1) {
      F->setName(Name + ".old");
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::ctlz,
                                        F->arg_begin()->getType());
      return true;
    }
    if (Name.startswith("cttz.") && F->arg_size() == 1) {
      F->setName(Name + ".old");
      NewFn = Intrinsic::getDeclaration(F->getParent(), Intrinsic::cttz,
                                        F->arg_begin()->getType());
      return true;
    }
    break;
  }

  case 'o':
    // We only need to change the name to match the mangling including the
    // address space.
    if (F->arg_size() == 2 && Name.startswith("objectsize.")) {
      Type *Tys[2] = { F->getReturnType(), F->arg_begin()->getType() };
      if (F->getName() != Intrinsic::getName(Intrinsic::objectsize, Tys)) {
        F->setName(Name + ".old");
        NewFn = Intrinsic::getDeclaration(F->getParent(),
                                          Intrinsic::objectsize, Tys);
        return true;
      }
    }
    break;

#if 0 // HLSL Change - remove platform intrinsics
  case 'x': {
    if (Name.startswith("x86.sse2.pcmpeq.") ||
        Name.startswith("x86.sse2.pcmpgt.") ||
        Name.startswith("x86.avx2.pcmpeq.") ||
        Name.startswith("x86.avx2.pcmpgt.") ||
        Name.startswith("x86.avx.vpermil.") ||
        Name == "x86.avx.vinsertf128.pd.256" ||
        Name == "x86.avx.vinsertf128.ps.256" ||
        Name == "x86.avx.vinsertf128.si.256" ||
        Name == "x86.avx2.vinserti128" ||
        Name == "x86.avx.vextractf128.pd.256" ||
        Name == "x86.avx.vextractf128.ps.256" ||
        Name == "x86.avx.vextractf128.si.256" ||
        Name == "x86.avx2.vextracti128" ||
        Name == "x86.avx.movnt.dq.256" ||
        Name == "x86.avx.movnt.pd.256" ||
        Name == "x86.avx.movnt.ps.256" ||
        Name == "x86.sse42.crc32.64.8" ||
        Name == "x86.avx.vbroadcast.ss" ||
        Name == "x86.avx.vbroadcast.ss.256" ||
        Name == "x86.avx.vbroadcast.sd.256" ||
        Name == "x86.sse2.psll.dq" ||
        Name == "x86.sse2.psrl.dq" ||
        Name == "x86.avx2.psll.dq" ||
        Name == "x86.avx2.psrl.dq" ||
        Name == "x86.sse2.psll.dq.bs" ||
        Name == "x86.sse2.psrl.dq.bs" ||
        Name == "x86.avx2.psll.dq.bs" ||
        Name == "x86.avx2.psrl.dq.bs" ||
        Name == "x86.sse41.pblendw" ||
        Name == "x86.sse41.blendpd" ||
        Name == "x86.sse41.blendps" ||
        Name == "x86.avx.blend.pd.256" ||
        Name == "x86.avx.blend.ps.256" ||
        Name == "x86.avx2.pblendw" ||
        Name == "x86.avx2.pblendd.128" ||
        Name == "x86.avx2.pblendd.256" ||
        Name == "x86.avx2.vbroadcasti128" ||
        (Name.startswith("x86.xop.vpcom") && F->arg_size() == 2)) {
      NewFn = nullptr;
      return true;
    }
    // SSE4.1 ptest functions may have an old signature.
    if (Name.startswith("x86.sse41.ptest")) {
      if (Name == "x86.sse41.ptestc")
        return UpgradeSSE41Function(F, Intrinsic::x86_sse41_ptestc, NewFn);
      if (Name == "x86.sse41.ptestz")
        return UpgradeSSE41Function(F, Intrinsic::x86_sse41_ptestz, NewFn);
      if (Name == "x86.sse41.ptestnzc")
        return UpgradeSSE41Function(F, Intrinsic::x86_sse41_ptestnzc, NewFn);
    }
    // Several blend and other instructions with masks used the wrong number of
    // bits.
    if (Name == "x86.sse41.insertps")
      return UpgradeX86IntrinsicsWith8BitMask(F, Intrinsic::x86_sse41_insertps,
                                              NewFn);
    if (Name == "x86.sse41.dppd")
      return UpgradeX86IntrinsicsWith8BitMask(F, Intrinsic::x86_sse41_dppd,
                                              NewFn);
    if (Name == "x86.sse41.dpps")
      return UpgradeX86IntrinsicsWith8BitMask(F, Intrinsic::x86_sse41_dpps,
                                              NewFn);
    if (Name == "x86.sse41.mpsadbw")
      return UpgradeX86IntrinsicsWith8BitMask(F, Intrinsic::x86_sse41_mpsadbw,
                                              NewFn);
    if (Name == "x86.avx.dp.ps.256")
      return UpgradeX86IntrinsicsWith8BitMask(F, Intrinsic::x86_avx_dp_ps_256,
                                              NewFn);
    if (Name == "x86.avx2.mpsadbw")
      return UpgradeX86IntrinsicsWith8BitMask(F, Intrinsic::x86_avx2_mpsadbw,
                                              NewFn);

    // frcz.ss/sd may need to have an argument dropped
    if (Name.startswith("x86.xop.vfrcz.ss") && F->arg_size() == 2) {
      F->setName(Name + ".old");
      NewFn = Intrinsic::getDeclaration(F->getParent(),
                                        Intrinsic::x86_xop_vfrcz_ss);
      return true;
    }
    if (Name.startswith("x86.xop.vfrcz.sd") && F->arg_size() == 2) {
      F->setName(Name + ".old");
      NewFn = Intrinsic::getDeclaration(F->getParent(),
                                        Intrinsic::x86_xop_vfrcz_sd);
      return true;
    }
    // Fix the FMA4 intrinsics to remove the 4
    if (Name.startswith("x86.fma4.")) {
      F->setName("llvm.x86.fma" + Name.substr(8));
      NewFn = F;
      return true;
    }
    break;
  }
#endif // HLSL Change - remove platform intrinsics
  }

  //  This may not belong here. This function is effectively being overloaded
  //  to both detect an intrinsic which needs upgrading, and to provide the
  //  upgraded form of the intrinsic. We should perhaps have two separate
  //  functions for this.
  return false;
}

bool llvm::UpgradeIntrinsicFunction(Function *F, Function *&NewFn) {
  NewFn = nullptr;
  bool Upgraded = UpgradeIntrinsicFunction1(F, NewFn);
  assert(F != NewFn && "Intrinsic function upgraded to the same function");

  // Upgrade intrinsic attributes.  This does not change the function.
  if (NewFn)
    F = NewFn;
  if (Intrinsic::ID id = F->getIntrinsicID())
    F->setAttributes(Intrinsic::getAttributes(F->getContext(), id));
  return Upgraded;
}

bool llvm::UpgradeGlobalVariable(GlobalVariable *GV) {
  // Nothing to do yet.
  return false;
}

#if 0 // HLSL Change - remove platform intrinsics
// Handles upgrading SSE2 and AVX2 PSLLDQ intrinsics by converting them
// to byte shuffles.
static Value *UpgradeX86PSLLDQIntrinsics(IRBuilder<> &Builder, LLVMContext &C,
                                         Value *Op, unsigned NumLanes,
                                         unsigned Shift) {
  // Each lane is 16 bytes.
  unsigned NumElts = NumLanes * 16;

  // Bitcast from a 64-bit element type to a byte element type.
  Op = Builder.CreateBitCast(Op,
                             VectorType::get(Type::getInt8Ty(C), NumElts),
                             "cast");
  // We'll be shuffling in zeroes.
  Value *Res = ConstantVector::getSplat(NumElts, Builder.getInt8(0));

  // If shift is less than 16, emit a shuffle to move the bytes. Otherwise,
  // we'll just return the zero vector.
  if (Shift < 16) {
    SmallVector<Constant*, 32> Idxs;
    // 256-bit version is split into two 16-byte lanes.
    for (unsigned l = 0; l != NumElts; l += 16)
      for (unsigned i = 0; i != 16; ++i) {
        unsigned Idx = NumElts + i - Shift;
        if (Idx < NumElts)
          Idx -= NumElts - 16; // end of lane, switch operand.
        Idxs.push_back(Builder.getInt32(Idx + l));
      }

    Res = Builder.CreateShuffleVector(Res, Op, ConstantVector::get(Idxs));
  }

  // Bitcast back to a 64-bit element type.
  return Builder.CreateBitCast(Res,
                               VectorType::get(Type::getInt64Ty(C), 2*NumLanes),
                               "cast");
}

// Handles upgrading SSE2 and AVX2 PSRLDQ intrinsics by converting them
// to byte shuffles.
static Value *UpgradeX86PSRLDQIntrinsics(IRBuilder<> &Builder, LLVMContext &C,
                                         Value *Op, unsigned NumLanes,
                                         unsigned Shift) {
  // Each lane is 16 bytes.
  unsigned NumElts = NumLanes * 16;

  // Bitcast from a 64-bit element type to a byte element type.
  Op = Builder.CreateBitCast(Op,
                             VectorType::get(Type::getInt8Ty(C), NumElts),
                             "cast");
  // We'll be shuffling in zeroes.
  Value *Res = ConstantVector::getSplat(NumElts, Builder.getInt8(0));

  // If shift is less than 16, emit a shuffle to move the bytes. Otherwise,
  // we'll just return the zero vector.
  if (Shift < 16) {
    SmallVector<Constant*, 32> Idxs;
    // 256-bit version is split into two 16-byte lanes.
    for (unsigned l = 0; l != NumElts; l += 16)
      for (unsigned i = 0; i != 16; ++i) {
        unsigned Idx = i + Shift;
        if (Idx >= 16)
          Idx += NumElts - 16; // end of lane, switch operand.
        Idxs.push_back(Builder.getInt32(Idx + l));
      }

    Res = Builder.CreateShuffleVector(Op, Res, ConstantVector::get(Idxs));
  }

  // Bitcast back to a 64-bit element type.
  return Builder.CreateBitCast(Res,
                               VectorType::get(Type::getInt64Ty(C), 2*NumLanes),
                               "cast");
}
#endif // HLSL Change - remove platform intrinsics

// UpgradeIntrinsicCall - Upgrade a call to an old intrinsic to be a call the
// upgraded intrinsic. All argument and return casting must be provided in
// order to seamlessly integrate with existing context.
void llvm::UpgradeIntrinsicCall(CallInst *CI, Function *NewFn) {
  Function *F = CI->getCalledFunction();
  LLVMContext &C = CI->getContext();
  IRBuilder<> Builder(C);
  Builder.SetInsertPoint(CI->getParent(), CI);

  assert(F && "Intrinsic call is not direct?");

  if (!NewFn) {
    (void)F;  // HLSL Change - unused local variable
#if 0 // HLSL Change - remove platform intrinsics
    // Get the Function's name.
    StringRef Name = F->getName();

    Value *Rep;
    // Upgrade packed integer vector compares intrinsics to compare instructions
    if (Name.startswith("llvm.x86.sse2.pcmpeq.") ||
        Name.startswith("llvm.x86.avx2.pcmpeq.")) {
      Rep = Builder.CreateICmpEQ(CI->getArgOperand(0), CI->getArgOperand(1),
                                 "pcmpeq");
      // need to sign extend since icmp returns vector of i1
      Rep = Builder.CreateSExt(Rep, CI->getType(), "");
    } else if (Name.startswith("llvm.x86.sse2.pcmpgt.") ||
               Name.startswith("llvm.x86.avx2.pcmpgt.")) {
      Rep = Builder.CreateICmpSGT(CI->getArgOperand(0), CI->getArgOperand(1),
                                  "pcmpgt");
      // need to sign extend since icmp returns vector of i1
      Rep = Builder.CreateSExt(Rep, CI->getType(), "");
    } else if (Name == "llvm.x86.avx.movnt.dq.256" ||
               Name == "llvm.x86.avx.movnt.ps.256" ||
               Name == "llvm.x86.avx.movnt.pd.256") {
      IRBuilder<> Builder(C);
      Builder.SetInsertPoint(CI->getParent(), CI);

      Module *M = F->getParent();
      SmallVector<Metadata *, 1> Elts;
      Elts.push_back(
          ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(C), 1)));
      MDNode *Node = MDNode::get(C, Elts);

      Value *Arg0 = CI->getArgOperand(0);
      Value *Arg1 = CI->getArgOperand(1);

      // Convert the type of the pointer to a pointer to the stored type.
      Value *BC = Builder.CreateBitCast(Arg0,
                                        PointerType::getUnqual(Arg1->getType()),
                                        "cast");
      StoreInst *SI = Builder.CreateStore(Arg1, BC);
      SI->setMetadata(M->getMDKindID("nontemporal"), Node);
      SI->setAlignment(16);

      // Remove intrinsic.
      CI->eraseFromParent();
      return;
    } else if (Name.startswith("llvm.x86.xop.vpcom")) {
      Intrinsic::ID intID;
      if (Name.endswith("ub"))
        intID = Intrinsic::x86_xop_vpcomub;
      else if (Name.endswith("uw"))
        intID = Intrinsic::x86_xop_vpcomuw;
      else if (Name.endswith("ud"))
        intID = Intrinsic::x86_xop_vpcomud;
      else if (Name.endswith("uq"))
        intID = Intrinsic::x86_xop_vpcomuq;
      else if (Name.endswith("b"))
        intID = Intrinsic::x86_xop_vpcomb;
      else if (Name.endswith("w"))
        intID = Intrinsic::x86_xop_vpcomw;
      else if (Name.endswith("d"))
        intID = Intrinsic::x86_xop_vpcomd;
      else if (Name.endswith("q"))
        intID = Intrinsic::x86_xop_vpcomq;
      else
        llvm_unreachable("Unknown suffix");

      Name = Name.substr(18); // strip off "llvm.x86.xop.vpcom"
      unsigned Imm;
      if (Name.startswith("lt"))
        Imm = 0;
      else if (Name.startswith("le"))
        Imm = 1;
      else if (Name.startswith("gt"))
        Imm = 2;
      else if (Name.startswith("ge"))
        Imm = 3;
      else if (Name.startswith("eq"))
        Imm = 4;
      else if (Name.startswith("ne"))
        Imm = 5;
      else if (Name.startswith("false"))
        Imm = 6;
      else if (Name.startswith("true"))
        Imm = 7;
      else
        llvm_unreachable("Unknown condition");

      Function *VPCOM = Intrinsic::getDeclaration(F->getParent(), intID);
      Rep =
          Builder.CreateCall(VPCOM, {CI->getArgOperand(0), CI->getArgOperand(1),
                                     Builder.getInt8(Imm)});
    } else if (Name == "llvm.x86.sse42.crc32.64.8") {
      Function *CRC32 = Intrinsic::getDeclaration(F->getParent(),
                                               Intrinsic::x86_sse42_crc32_32_8);
      Value *Trunc0 = Builder.CreateTrunc(CI->getArgOperand(0), Type::getInt32Ty(C));
      Rep = Builder.CreateCall(CRC32, {Trunc0, CI->getArgOperand(1)});
      Rep = Builder.CreateZExt(Rep, CI->getType(), "");
    } else if (Name.startswith("llvm.x86.avx.vbroadcast")) {
      // Replace broadcasts with a series of insertelements.
      Type *VecTy = CI->getType();
      Type *EltTy = VecTy->getVectorElementType();
      unsigned EltNum = VecTy->getVectorNumElements();
      Value *Cast = Builder.CreateBitCast(CI->getArgOperand(0),
                                          EltTy->getPointerTo());
      Value *Load = Builder.CreateLoad(EltTy, Cast);
      Type *I32Ty = Type::getInt32Ty(C);
      Rep = UndefValue::get(VecTy);
      for (unsigned I = 0; I < EltNum; ++I)
        Rep = Builder.CreateInsertElement(Rep, Load,
                                          ConstantInt::get(I32Ty, I));
    } else if (Name == "llvm.x86.avx2.vbroadcasti128") {
      // Replace vbroadcasts with a vector shuffle.
      Type *VT = VectorType::get(Type::getInt64Ty(C), 2);
      Value *Op = Builder.CreatePointerCast(CI->getArgOperand(0),
                                            PointerType::getUnqual(VT));
      Value *Load = Builder.CreateLoad(VT, Op);
      const int Idxs[4] = { 0, 1, 0, 1 };
      Rep = Builder.CreateShuffleVector(Load, UndefValue::get(Load->getType()),
                                        Idxs);
    } else if (Name == "llvm.x86.sse2.psll.dq") {
      // 128-bit shift left specified in bits.
      unsigned Shift = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
      Rep = UpgradeX86PSLLDQIntrinsics(Builder, C, CI->getArgOperand(0), 1,
                                       Shift / 8); // Shift is in bits.
    } else if (Name == "llvm.x86.sse2.psrl.dq") {
      // 128-bit shift right specified in bits.
      unsigned Shift = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
      Rep = UpgradeX86PSRLDQIntrinsics(Builder, C, CI->getArgOperand(0), 1,
                                       Shift / 8); // Shift is in bits.
    } else if (Name == "llvm.x86.avx2.psll.dq") {
      // 256-bit shift left specified in bits.
      unsigned Shift = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
      Rep = UpgradeX86PSLLDQIntrinsics(Builder, C, CI->getArgOperand(0), 2,
                                       Shift / 8); // Shift is in bits.
    } else if (Name == "llvm.x86.avx2.psrl.dq") {
      // 256-bit shift right specified in bits.
      unsigned Shift = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
      Rep = UpgradeX86PSRLDQIntrinsics(Builder, C, CI->getArgOperand(0), 2,
                                       Shift / 8); // Shift is in bits.
    } else if (Name == "llvm.x86.sse2.psll.dq.bs") {
      // 128-bit shift left specified in bytes.
      unsigned Shift = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
      Rep = UpgradeX86PSLLDQIntrinsics(Builder, C, CI->getArgOperand(0), 1,
                                       Shift);
    } else if (Name == "llvm.x86.sse2.psrl.dq.bs") {
      // 128-bit shift right specified in bytes.
      unsigned Shift = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
      Rep = UpgradeX86PSRLDQIntrinsics(Builder, C, CI->getArgOperand(0), 1,
                                       Shift);
    } else if (Name == "llvm.x86.avx2.psll.dq.bs") {
      // 256-bit shift left specified in bytes.
      unsigned Shift = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
      Rep = UpgradeX86PSLLDQIntrinsics(Builder, C, CI->getArgOperand(0), 2,
                                       Shift);
    } else if (Name == "llvm.x86.avx2.psrl.dq.bs") {
      // 256-bit shift right specified in bytes.
      unsigned Shift = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
      Rep = UpgradeX86PSRLDQIntrinsics(Builder, C, CI->getArgOperand(0), 2,
                                       Shift);
    } else if (Name == "llvm.x86.sse41.pblendw" ||
               Name == "llvm.x86.sse41.blendpd" ||
               Name == "llvm.x86.sse41.blendps" ||
               Name == "llvm.x86.avx.blend.pd.256" ||
               Name == "llvm.x86.avx.blend.ps.256" ||
               Name == "llvm.x86.avx2.pblendw" ||
               Name == "llvm.x86.avx2.pblendd.128" ||
               Name == "llvm.x86.avx2.pblendd.256") {
      Value *Op0 = CI->getArgOperand(0);
      Value *Op1 = CI->getArgOperand(1);
      unsigned Imm = cast <ConstantInt>(CI->getArgOperand(2))->getZExtValue();
      VectorType *VecTy = cast<VectorType>(CI->getType());
      unsigned NumElts = VecTy->getNumElements();

      SmallVector<Constant*, 16> Idxs;
      for (unsigned i = 0; i != NumElts; ++i) {
        unsigned Idx = ((Imm >> (i%8)) & 1) ? i + NumElts : i;
        Idxs.push_back(Builder.getInt32(Idx));
      }

      Rep = Builder.CreateShuffleVector(Op0, Op1, ConstantVector::get(Idxs));
    } else if (Name == "llvm.x86.avx.vinsertf128.pd.256" ||
               Name == "llvm.x86.avx.vinsertf128.ps.256" ||
               Name == "llvm.x86.avx.vinsertf128.si.256" ||
               Name == "llvm.x86.avx2.vinserti128") {
      Value *Op0 = CI->getArgOperand(0);
      Value *Op1 = CI->getArgOperand(1);
      unsigned Imm = cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue();
      VectorType *VecTy = cast<VectorType>(CI->getType());
      unsigned NumElts = VecTy->getNumElements();
      
      // Mask off the high bits of the immediate value; hardware ignores those.
      Imm = Imm & 1;
      
      // Extend the second operand into a vector that is twice as big.
      Value *UndefV = UndefValue::get(Op1->getType());
      SmallVector<Constant*, 8> Idxs;
      for (unsigned i = 0; i != NumElts; ++i) {
        Idxs.push_back(Builder.getInt32(i));
      }
      Rep = Builder.CreateShuffleVector(Op1, UndefV, ConstantVector::get(Idxs));

      // Insert the second operand into the first operand.

      // Note that there is no guarantee that instruction lowering will actually
      // produce a vinsertf128 instruction for the created shuffles. In
      // particular, the 0 immediate case involves no lane changes, so it can
      // be handled as a blend.

      // Example of shuffle mask for 32-bit elements:
      // Imm = 1  <i32 0, i32 1, i32 2,  i32 3,  i32 8, i32 9, i32 10, i32 11>
      // Imm = 0  <i32 8, i32 9, i32 10, i32 11, i32 4, i32 5, i32 6,  i32 7 >

      SmallVector<Constant*, 8> Idxs2;
      // The low half of the result is either the low half of the 1st operand
      // or the low half of the 2nd operand (the inserted vector).
      for (unsigned i = 0; i != NumElts / 2; ++i) {
        unsigned Idx = Imm ? i : (i + NumElts);
        Idxs2.push_back(Builder.getInt32(Idx));
      }
      // The high half of the result is either the low half of the 2nd operand
      // (the inserted vector) or the high half of the 1st operand.
      for (unsigned i = NumElts / 2; i != NumElts; ++i) {
        unsigned Idx = Imm ? (i + NumElts / 2) : i;
        Idxs2.push_back(Builder.getInt32(Idx));
      }
      Rep = Builder.CreateShuffleVector(Op0, Rep, ConstantVector::get(Idxs2));
    } else if (Name == "llvm.x86.avx.vextractf128.pd.256" ||
               Name == "llvm.x86.avx.vextractf128.ps.256" ||
               Name == "llvm.x86.avx.vextractf128.si.256" ||
               Name == "llvm.x86.avx2.vextracti128") {
      Value *Op0 = CI->getArgOperand(0);
      unsigned Imm = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
      VectorType *VecTy = cast<VectorType>(CI->getType());
      unsigned NumElts = VecTy->getNumElements();
      
      // Mask off the high bits of the immediate value; hardware ignores those.
      Imm = Imm & 1;

      // Get indexes for either the high half or low half of the input vector.
      SmallVector<Constant*, 4> Idxs(NumElts);
      for (unsigned i = 0; i != NumElts; ++i) {
        unsigned Idx = Imm ? (i + NumElts) : i;
        Idxs[i] = Builder.getInt32(Idx);
      }

      Value *UndefV = UndefValue::get(Op0->getType());
      Rep = Builder.CreateShuffleVector(Op0, UndefV, ConstantVector::get(Idxs));
    } else {
      bool PD128 = false, PD256 = false, PS128 = false, PS256 = false;
      if (Name == "llvm.x86.avx.vpermil.pd.256")
        PD256 = true;
      else if (Name == "llvm.x86.avx.vpermil.pd")
        PD128 = true;
      else if (Name == "llvm.x86.avx.vpermil.ps.256")
        PS256 = true;
      else if (Name == "llvm.x86.avx.vpermil.ps")
        PS128 = true;

      if (PD256 || PD128 || PS256 || PS128) {
        Value *Op0 = CI->getArgOperand(0);
        unsigned Imm = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
        SmallVector<Constant*, 8> Idxs;

        if (PD128)
          for (unsigned i = 0; i != 2; ++i)
            Idxs.push_back(Builder.getInt32((Imm >> i) & 0x1));
        else if (PD256)
          for (unsigned l = 0; l != 4; l+=2)
            for (unsigned i = 0; i != 2; ++i)
              Idxs.push_back(Builder.getInt32(((Imm >> (l+i)) & 0x1) + l));
        else if (PS128)
          for (unsigned i = 0; i != 4; ++i)
            Idxs.push_back(Builder.getInt32((Imm >> (2 * i)) & 0x3));
        else if (PS256)
          for (unsigned l = 0; l != 8; l+=4)
            for (unsigned i = 0; i != 4; ++i)
              Idxs.push_back(Builder.getInt32(((Imm >> (2 * i)) & 0x3) + l));
        else
          llvm_unreachable("Unexpected function");

        Rep = Builder.CreateShuffleVector(Op0, Op0, ConstantVector::get(Idxs));
      } else {
        llvm_unreachable("Unknown function for CallInst upgrade.");
      }
    }

    CI->replaceAllUsesWith(Rep);
    CI->eraseFromParent();

#endif // HLSL Change - remove platform intrinsics
    llvm_unreachable("HLSL - should not be upgrading platform intrinsics."); // HLSL Change - remove platform intrinsics

    return;
  }

  std::string Name = CI->getName();
  if (!Name.empty())
    CI->setName(Name + ".old");

  switch (NewFn->getIntrinsicID()) {
  default:
    llvm_unreachable("Unknown function for CallInst upgrade.");

  case Intrinsic::ctlz:
  case Intrinsic::cttz:
    assert(CI->getNumArgOperands() == 1 &&
           "Mismatch between function args and call args");
    CI->replaceAllUsesWith(Builder.CreateCall(
        NewFn, {CI->getArgOperand(0), Builder.getFalse()}, Name));
    CI->eraseFromParent();
    return;

  case Intrinsic::objectsize:
    CI->replaceAllUsesWith(Builder.CreateCall(
        NewFn, {CI->getArgOperand(0), CI->getArgOperand(1)}, Name));
    CI->eraseFromParent();
    return;

  case Intrinsic::ctpop: {
    CI->replaceAllUsesWith(Builder.CreateCall(NewFn, {CI->getArgOperand(0)}));
    CI->eraseFromParent();
    return;
  }

#if 0 // HLSL Change - remove platform intrinsics
  case Intrinsic::x86_xop_vfrcz_ss:
  case Intrinsic::x86_xop_vfrcz_sd:
    CI->replaceAllUsesWith(
        Builder.CreateCall(NewFn, {CI->getArgOperand(1)}, Name));
    CI->eraseFromParent();
    return;

  case Intrinsic::x86_sse41_ptestc:
  case Intrinsic::x86_sse41_ptestz:
  case Intrinsic::x86_sse41_ptestnzc: {
    // The arguments for these intrinsics used to be v4f32, and changed
    // to v2i64. This is purely a nop, since those are bitwise intrinsics.
    // So, the only thing required is a bitcast for both arguments.
    // First, check the arguments have the old type.
    Value *Arg0 = CI->getArgOperand(0);
    if (Arg0->getType() != VectorType::get(Type::getFloatTy(C), 4))
      return;

    // Old intrinsic, add bitcasts
    Value *Arg1 = CI->getArgOperand(1);

    Type *NewVecTy = VectorType::get(Type::getInt64Ty(C), 2);

    Value *BC0 = Builder.CreateBitCast(Arg0, NewVecTy, "cast");
    Value *BC1 = Builder.CreateBitCast(Arg1, NewVecTy, "cast");

    CallInst *NewCall = Builder.CreateCall(NewFn, {BC0, BC1}, Name);
    CI->replaceAllUsesWith(NewCall);
    CI->eraseFromParent();
    return;
  }

  case Intrinsic::x86_sse41_insertps:
  case Intrinsic::x86_sse41_dppd:
  case Intrinsic::x86_sse41_dpps:
  case Intrinsic::x86_sse41_mpsadbw:
  case Intrinsic::x86_avx_dp_ps_256:
  case Intrinsic::x86_avx2_mpsadbw: {
    // Need to truncate the last argument from i32 to i8 -- this argument models
    // an inherently 8-bit immediate operand to these x86 instructions.
    SmallVector<Value *, 4> Args(CI->arg_operands().begin(),
                                 CI->arg_operands().end());

    // Replace the last argument with a trunc.
    Args.back() = Builder.CreateTrunc(Args.back(), Type::getInt8Ty(C), "trunc");

    CallInst *NewCall = Builder.CreateCall(NewFn, Args);
    CI->replaceAllUsesWith(NewCall);
    CI->eraseFromParent();
    return;
  }
#endif // HLSL Change - remove platform intrinsics
  }
}

// This tests each Function to determine if it needs upgrading. When we find
// one we are interested in, we then upgrade all calls to reflect the new
// function.
void llvm::UpgradeCallsToIntrinsic(Function* F) {
  assert(F && "Illegal attempt to upgrade a non-existent intrinsic.");

  // Upgrade the function and check if it is a totaly new function.
  Function *NewFn;
  if (UpgradeIntrinsicFunction(F, NewFn)) {
    // Replace all uses to the old function with the new one if necessary.
    for (Value::user_iterator UI = F->user_begin(), UE = F->user_end();
         UI != UE;) {
      if (CallInst *CI = dyn_cast<CallInst>(*UI++))
        UpgradeIntrinsicCall(CI, NewFn);
    }
    // Remove old function, no longer used, from the module.
    F->eraseFromParent();
  }
}

void llvm::UpgradeInstWithTBAATag(Instruction *I) {
  MDNode *MD = I->getMetadata(LLVMContext::MD_tbaa);
  assert(MD && "UpgradeInstWithTBAATag should have a TBAA tag");
  // Check if the tag uses struct-path aware TBAA format.
  if (isa<MDNode>(MD->getOperand(0)) && MD->getNumOperands() >= 3)
    return;

  if (MD->getNumOperands() == 3) {
    Metadata *Elts[] = {MD->getOperand(0), MD->getOperand(1)};
    MDNode *ScalarType = MDNode::get(I->getContext(), Elts);
    // Create a MDNode <ScalarType, ScalarType, offset 0, const>
    Metadata *Elts2[] = {ScalarType, ScalarType,
                         ConstantAsMetadata::get(Constant::getNullValue(
                             Type::getInt64Ty(I->getContext()))),
                         MD->getOperand(2)};
    I->setMetadata(LLVMContext::MD_tbaa, MDNode::get(I->getContext(), Elts2));
  } else {
    // Create a MDNode <MD, MD, offset 0>
    Metadata *Elts[] = {MD, MD, ConstantAsMetadata::get(Constant::getNullValue(
                                    Type::getInt64Ty(I->getContext())))};
    I->setMetadata(LLVMContext::MD_tbaa, MDNode::get(I->getContext(), Elts));
  }
}

Instruction *llvm::UpgradeBitCastInst(unsigned Opc, Value *V, Type *DestTy,
                                      Instruction *&Temp) {
  if (Opc != Instruction::BitCast)
    return nullptr;

  Temp = nullptr;
  Type *SrcTy = V->getType();
  if (SrcTy->isPtrOrPtrVectorTy() && DestTy->isPtrOrPtrVectorTy() &&
      SrcTy->getPointerAddressSpace() != DestTy->getPointerAddressSpace()) {
    LLVMContext &Context = V->getContext();

    // We have no information about target data layout, so we assume that
    // the maximum pointer size is 64bit.
    Type *MidTy = Type::getInt64Ty(Context);
    Temp = CastInst::Create(Instruction::PtrToInt, V, MidTy);

    return CastInst::Create(Instruction::IntToPtr, Temp, DestTy);
  }

  return nullptr;
}

Value *llvm::UpgradeBitCastExpr(unsigned Opc, Constant *C, Type *DestTy) {
  if (Opc != Instruction::BitCast)
    return nullptr;

  Type *SrcTy = C->getType();
  if (SrcTy->isPtrOrPtrVectorTy() && DestTy->isPtrOrPtrVectorTy() &&
      SrcTy->getPointerAddressSpace() != DestTy->getPointerAddressSpace()) {
    LLVMContext &Context = C->getContext();

    // We have no information about target data layout, so we assume that
    // the maximum pointer size is 64bit.
    Type *MidTy = Type::getInt64Ty(Context);

    return ConstantExpr::getIntToPtr(ConstantExpr::getPtrToInt(C, MidTy),
                                     DestTy);
  }

  return nullptr;
}

/// Check the debug info version number, if it is out-dated, drop the debug
/// info. Return true if module is modified.
bool llvm::UpgradeDebugInfo(Module &M) {
  unsigned Version = getDebugMetadataVersionFromModule(M);
  if (Version == DEBUG_METADATA_VERSION)
    return false;

  bool RetCode = StripDebugInfo(M);
  if (RetCode) {
    DiagnosticInfoDebugMetadataVersion DiagVersion(M, Version);
    M.getContext().diagnose(DiagVersion);
  }
  return RetCode;
}

void llvm::UpgradeMDStringConstant(std::string &String) {
  const std::string OldPrefix = "llvm.vectorizer.";
  if (String == "llvm.vectorizer.unroll") {
    String = "llvm.loop.interleave.count";
  } else if (String.find(OldPrefix) == 0) {
    String.replace(0, OldPrefix.size(), "llvm.loop.vectorize.");
  }
}
