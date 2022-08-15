///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilPIXVirtualRegisters.cpp                                               //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Defines functions for dealing with the virtual register annotations in    //
// DXIL instructions.                                                        //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DxilPIXPasses/DxilPIXVirtualRegisters.h"

#include "dxc/Support/Global.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Type.h"

void pix_dxil::PixDxilInstNum::AddMD(llvm::LLVMContext &Ctx,
                                     llvm::Instruction *pI,
                                     std::uint32_t InstNum) {
  llvm::IRBuilder<> B(Ctx);
  pI->setMetadata(
      llvm::StringRef(MDName),
      llvm::MDNode::get(Ctx,
                        {llvm::ConstantAsMetadata::get(B.getInt32(ID)),
                         llvm::ConstantAsMetadata::get(B.getInt32(InstNum))}));
}

bool pix_dxil::PixDxilInstNum::FromInst(llvm::Instruction *pI,
                                        std::uint32_t *pInstNum) {
  *pInstNum = 0;

  auto *mdNodes = pI->getMetadata(MDName);

  if (mdNodes == nullptr) {
    return false;
  }

  if (mdNodes->getNumOperands() != 2) {
    return false;
  }

  auto *mdID =
      llvm::mdconst::dyn_extract<llvm::ConstantInt>(mdNodes->getOperand(0));
  if (mdID == nullptr || mdID->getLimitedValue() != ID) {
    return false;
  }

  auto *mdInstNum =
      llvm::mdconst::dyn_extract<llvm::ConstantInt>(mdNodes->getOperand(1));
  if (mdInstNum == nullptr) {
    return false;
  }

  *pInstNum = mdInstNum->getLimitedValue();
  return true;
}

void pix_dxil::PixDxilReg::AddMD(llvm::LLVMContext &Ctx, llvm::Instruction *pI,
                                 std::uint32_t RegNum) {
  llvm::IRBuilder<> B(Ctx);
  pI->setMetadata(
      llvm::StringRef(MDName),
      llvm::MDNode::get(Ctx,
                        {llvm::ConstantAsMetadata::get(B.getInt32(ID)),
                         llvm::ConstantAsMetadata::get(B.getInt32(RegNum))}));
}

bool pix_dxil::PixDxilReg::FromInst(llvm::Instruction *pI,
                                    std::uint32_t *pRegNum) {
  *pRegNum = 0;

  auto *mdNodes = pI->getMetadata(MDName);

  if (mdNodes == nullptr) {
    return false;
  }

  if (mdNodes->getNumOperands() != 2) {
    return false;
  }

  auto *mdID =
      llvm::mdconst::dyn_extract<llvm::ConstantInt>(mdNodes->getOperand(0));
  if (mdID == nullptr || mdID->getLimitedValue() != ID) {
    return false;
  }

  auto *mdRegNum =
      llvm::mdconst::dyn_extract<llvm::ConstantInt>(mdNodes->getOperand(1));
  if (mdRegNum == nullptr) {
    return false;
  }

  *pRegNum = mdRegNum->getLimitedValue();
  return true;
}

static bool ParsePixAllocaReg(llvm::MDNode *MD, std::uint32_t *RegNum,
                              std::uint32_t *Count) {
  if (MD->getNumOperands() != 3) {
    return false;
  }

  auto *mdID = llvm::mdconst::dyn_extract<llvm::ConstantInt>(MD->getOperand(0));
  if (mdID == nullptr ||
      mdID->getLimitedValue() != pix_dxil::PixAllocaReg::ID) {
    return false;
  }

  auto *mdRegNum =
      llvm::mdconst::dyn_extract<llvm::ConstantInt>(MD->getOperand(1));
  auto *mdCount =
      llvm::mdconst::dyn_extract<llvm::ConstantInt>(MD->getOperand(2));

  if (mdRegNum == nullptr || mdCount == nullptr) {
    return false;
  }

  *RegNum = mdRegNum->getLimitedValue();
  *Count = mdCount->getLimitedValue();
  return true;
}

void pix_dxil::PixAllocaReg::AddMD(llvm::LLVMContext &Ctx,
                                   llvm::AllocaInst *pAlloca,
                                   std::uint32_t RegNum, std::uint32_t Count) {
  llvm::IRBuilder<> B(Ctx);
  pAlloca->setMetadata(
      llvm::StringRef(MDName),
      llvm::MDNode::get(Ctx,
                        {llvm::ConstantAsMetadata::get(B.getInt32(ID)),
                         llvm::ConstantAsMetadata::get(B.getInt32(RegNum)),
                         llvm::ConstantAsMetadata::get(B.getInt32(Count))}));
}

bool pix_dxil::PixAllocaReg::FromInst(llvm::AllocaInst *pAlloca,
                                      std::uint32_t *pRegBase,
                                      std::uint32_t *pRegSize) {
  *pRegBase = 0;
  *pRegSize = 0;

  auto *mdNodes = pAlloca->getMetadata(MDName);
  if (mdNodes == nullptr) {
    return false;
  }

  return ParsePixAllocaReg(mdNodes, pRegBase, pRegSize);
}

namespace pix_dxil {
namespace PixAllocaRegWrite {
static constexpr uint32_t IndexIsConst = 1;
static constexpr uint32_t IndexIsPixInst = 2;
} // namespace PixAllocaRegWrite
} // namespace pix_dxil

void pix_dxil::PixAllocaRegWrite::AddMD(llvm::LLVMContext &Ctx,
                                        llvm::StoreInst *pSt,
                                        llvm::MDNode *pAllocaReg,
                                        llvm::Value *Index) {
  llvm::IRBuilder<> B(Ctx);
  if (auto *C = llvm::dyn_cast<llvm::ConstantInt>(Index)) {
    pSt->setMetadata(
        llvm::StringRef(MDName),
        llvm::MDNode::get(
            Ctx, {llvm::ConstantAsMetadata::get(B.getInt32(ID)), pAllocaReg,
                  llvm::ConstantAsMetadata::get(B.getInt32(IndexIsConst)),
                  llvm::ConstantAsMetadata::get(C)}));
  }

  if (auto *I = llvm::dyn_cast<llvm::Instruction>(Index)) {
    std::uint32_t InstNum;
    if (!PixDxilInstNum::FromInst(I, &InstNum)) {
      return;
    }
    pSt->setMetadata(
        llvm::StringRef(MDName),
        llvm::MDNode::get(
            Ctx, {llvm::ConstantAsMetadata::get(B.getInt32(ID)), pAllocaReg,
                  llvm::ConstantAsMetadata::get(B.getInt32(IndexIsPixInst)),
                  llvm::ConstantAsMetadata::get(B.getInt32(InstNum))}));
  }
}

bool pix_dxil::PixAllocaRegWrite::FromInst(llvm::StoreInst *pI,
                                           std::uint32_t *pRegBase,
                                           std::uint32_t *pRegSize,
                                           llvm::Value **pIndex) {
  *pRegBase = 0;
  *pRegSize = 0;
  *pIndex = nullptr;

  auto *mdNodes = pI->getMetadata(MDName);
  if (mdNodes == nullptr || mdNodes->getNumOperands() != 4) {
    return false;
  }

  auto *mdID =
      llvm::mdconst::dyn_extract<llvm::ConstantInt>(mdNodes->getOperand(0));
  if (mdID == nullptr || mdID->getLimitedValue() != ID) {
    return false;
  }

  auto *mdAllocaReg = llvm::dyn_cast<llvm::MDNode>(mdNodes->getOperand(1));
  if (mdAllocaReg == nullptr ||
      !ParsePixAllocaReg(mdAllocaReg, pRegBase, pRegSize)) {
    return false;
  }

  auto *mdIndexType =
      llvm::dyn_cast<llvm::ConstantAsMetadata>(mdNodes->getOperand(2));
  if (mdIndexType == nullptr) {
    return false;
  }

  auto *cIndexType = llvm::dyn_cast<llvm::ConstantInt>(mdIndexType->getValue());
  if (cIndexType == nullptr) {
    return false;
  }

  auto *mdIndex =
      llvm::dyn_cast<llvm::ConstantAsMetadata>(mdNodes->getOperand(3));
  if (mdIndex == nullptr) {
    return false;
  }

  auto *cIndex = llvm::dyn_cast<llvm::ConstantInt>(mdIndex->getValue());
  if (cIndex == nullptr) {
    return false;
  }

  switch (cIndexType->getLimitedValue()) {
  default:
    return false;

  case IndexIsConst: {
    *pIndex = cIndex;
    return true;
  }

  case IndexIsPixInst: {
    for (llvm::Instruction &I :
         llvm::inst_range(pI->getParent()->getParent())) {
      uint32_t InstNum;
      if (PixDxilInstNum::FromInst(&I, &InstNum)) {
        *pIndex = &I;
        if (InstNum == cIndex->getLimitedValue()) {
          return true;
        }
      }
    }
    return false;
  }
  }

  return false;
}
