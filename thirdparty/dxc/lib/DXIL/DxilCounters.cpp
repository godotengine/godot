///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilCounters.cpp                                                           //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DXIL/DxilCounters.h"
#include "dxc/Support/Global.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Module.h"
#include "llvm/ADT/DenseMap.h"

#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilInstructions.h"

using namespace llvm;
using namespace hlsl;
using namespace hlsl::DXIL;

namespace hlsl {

namespace {

struct PointerInfo {
  enum class MemType : unsigned {
    Unknown = 0,
    Global_Static,
    Global_TGSM,
    Alloca
  };

  MemType memType : 2;
  bool isArray : 1;

  PointerInfo() :
    memType(MemType::Unknown),
    isArray(false)
  {}
};

typedef DenseMap<Value*, PointerInfo> PointerInfoMap;

PointerInfo GetPointerInfo(Value* V, PointerInfoMap &ptrInfoMap) {
  auto it = ptrInfoMap.find(V);
  if (it != ptrInfoMap.end())
    return it->second;

  Type *Ty = V->getType()->getPointerElementType();
  ptrInfoMap[V].isArray = Ty->isArrayTy();

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {
    if (GV->getType()->getPointerAddressSpace() == DXIL::kTGSMAddrSpace)
      ptrInfoMap[V].memType = PointerInfo::MemType::Global_TGSM;
    else if (!GV->isConstant() &&
             GV->getLinkage() == GlobalVariable::LinkageTypes::InternalLinkage &&
             GV->getType()->getPointerAddressSpace() == DXIL::kDefaultAddrSpace)
      ptrInfoMap[V].memType = PointerInfo::MemType::Global_Static;
  } else if (AllocaInst *AI = dyn_cast<AllocaInst>(V)) {
      ptrInfoMap[V].memType = PointerInfo::MemType::Alloca;
  } else if (GEPOperator *GEP = dyn_cast<GEPOperator>(V)) {
    ptrInfoMap[V] = GetPointerInfo(GEP->getPointerOperand(), ptrInfoMap);
  } else if (BitCastOperator *BC = dyn_cast<BitCastOperator>(V)) {
    ptrInfoMap[V] = GetPointerInfo(BC->getOperand(0), ptrInfoMap);
  } else if (AddrSpaceCastInst *AC = dyn_cast<AddrSpaceCastInst>(V)) {
    ptrInfoMap[V] = GetPointerInfo(AC->getOperand(0), ptrInfoMap);
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    if (CE->getOpcode() == LLVMAddrSpaceCast)
      ptrInfoMap[V] = GetPointerInfo(AC->getOperand(0), ptrInfoMap);
  //} else if (PHINode *PN = dyn_cast<PHINode>(V)) {
  //  for (auto it = PN->value_op_begin(), e = PN->value_op_end(); it != e; ++it) {
  //    PI = GetPointerInfo(*it, ptrInfoMap);
  //    if (PI.memType != PointerInfo::MemType::Unknown)
  //      break;
  //  }
  }
  return ptrInfoMap[V];
}

struct ValueInfo {
  bool isCbuffer : 1;
  bool isConstant : 1;

  ValueInfo() :
    isCbuffer(false),
    isConstant(false)
  {}

  ValueInfo Combine(const ValueInfo &other) const {
    ValueInfo R;
    R.isCbuffer = isCbuffer && other.isCbuffer;
    R.isConstant = isConstant && other.isConstant;
    return R;
  }
};

/*<py>

def tab_lines(text):
  return ['  ' + line for line in text.splitlines()]

def gen_count_dxil_op(counter):
  return (['bool CountDxilOp_%s(unsigned op) {' % counter] +
          tab_lines(
            hctdb_instrhelp.get_instrs_pred("op", hctdb_instrhelp.counter_pred(counter, True))) +
          ['}'])

def gen_count_llvm_op(counter):
  return (['bool CountLlvmOp_%s(unsigned op) {' % counter] +
          tab_lines(
            hctdb_instrhelp.get_instrs_pred("op", hctdb_instrhelp.counter_pred(counter, False), 'llvm_id')) +
          ['}'])

def gen_counter_functions():
  lines = ['// Counter functions for Dxil ops:']
  for counter in hctdb_instrhelp.get_dxil_op_counters():
    lines += gen_count_dxil_op(counter)
  lines.append('// Counter functions for llvm ops:')
  for counter in hctdb_instrhelp.get_llvm_op_counters():
    lines += gen_count_llvm_op(counter)
  return lines

</py>*/

// <py::lines('OPCODE-COUNTERS')>gen_counter_functions()</py>
// OPCODE-COUNTERS:BEGIN
// Counter functions for Dxil ops:
bool CountDxilOp_atomic(unsigned op) {
  // Instructions: BufferUpdateCounter=70, AtomicBinOp=78,
  // AtomicCompareExchange=79
  return op == 70 || (78 <= op && op <= 79);
}
bool CountDxilOp_barrier(unsigned op) {
  // Instructions: Barrier=80
  return op == 80;
}
bool CountDxilOp_floats(unsigned op) {
  // Instructions: FAbs=6, Saturate=7, IsNaN=8, IsInf=9, IsFinite=10,
  // IsNormal=11, Cos=12, Sin=13, Tan=14, Acos=15, Asin=16, Atan=17, Hcos=18,
  // Hsin=19, Htan=20, Exp=21, Frc=22, Log=23, Sqrt=24, Rsqrt=25, Round_ne=26,
  // Round_ni=27, Round_pi=28, Round_z=29, FMax=35, FMin=36, Fma=47, Dot2=54,
  // Dot3=55, Dot4=56, Dot2AddHalf=162
  return (6 <= op && op <= 29) || (35 <= op && op <= 36) || op == 47 || (54 <= op && op <= 56) || op == 162;
}
bool CountDxilOp_gs_cut(unsigned op) {
  // Instructions: CutStream=98, EmitThenCutStream=99
  return (98 <= op && op <= 99);
}
bool CountDxilOp_gs_emit(unsigned op) {
  // Instructions: EmitStream=97, EmitThenCutStream=99
  return op == 97 || op == 99;
}
bool CountDxilOp_ints(unsigned op) {
  // Instructions: IMax=37, IMin=38, IMul=41, IMad=48, Ibfe=51,
  // Dot4AddI8Packed=163
  return (37 <= op && op <= 38) || op == 41 || op == 48 || op == 51 || op == 163;
}
bool CountDxilOp_sig_ld(unsigned op) {
  // Instructions: LoadInput=4, LoadOutputControlPoint=103, LoadPatchConstant=104
  return op == 4 || (103 <= op && op <= 104);
}
bool CountDxilOp_sig_st(unsigned op) {
  // Instructions: StoreOutput=5, StorePatchConstant=106, StoreVertexOutput=171,
  // StorePrimitiveOutput=172
  return op == 5 || op == 106 || (171 <= op && op <= 172);
}
bool CountDxilOp_tex_bias(unsigned op) {
  // Instructions: SampleBias=61
  return op == 61;
}
bool CountDxilOp_tex_cmp(unsigned op) {
  // Instructions: SampleCmp=64, SampleCmpLevelZero=65, TextureGatherCmp=74,
  // SampleCmpLevel=224
  return (64 <= op && op <= 65) || op == 74 || op == 224;
}
bool CountDxilOp_tex_grad(unsigned op) {
  // Instructions: SampleGrad=63
  return op == 63;
}
bool CountDxilOp_tex_load(unsigned op) {
  // Instructions: TextureLoad=66, BufferLoad=68, RawBufferLoad=139
  return op == 66 || op == 68 || op == 139;
}
bool CountDxilOp_tex_norm(unsigned op) {
  // Instructions: Sample=60, SampleLevel=62, TextureGather=73,
  // TextureGatherRaw=223
  return op == 60 || op == 62 || op == 73 || op == 223;
}
bool CountDxilOp_tex_store(unsigned op) {
  // Instructions: TextureStore=67, BufferStore=69, RawBufferStore=140,
  // WriteSamplerFeedback=174, WriteSamplerFeedbackBias=175,
  // WriteSamplerFeedbackLevel=176, WriteSamplerFeedbackGrad=177,
  // TextureStoreSample=225
  return op == 67 || op == 69 || op == 140 || (174 <= op && op <= 177) || op == 225;
}
bool CountDxilOp_uints(unsigned op) {
  // Instructions: Bfrev=30, Countbits=31, FirstbitLo=32, FirstbitHi=33,
  // FirstbitSHi=34, UMax=39, UMin=40, UMul=42, UDiv=43, UAddc=44, USubb=45,
  // UMad=49, Msad=50, Ubfe=52, Bfi=53, Dot4AddU8Packed=164
  return (30 <= op && op <= 34) || (39 <= op && op <= 40) || (42 <= op && op <= 45) || (49 <= op && op <= 50) || (52 <= op && op <= 53) || op == 164;
}
// Counter functions for llvm ops:
bool CountLlvmOp_atomic(unsigned op) {
  // Instructions: AtomicCmpXchg=31, AtomicRMW=32
  return (31 <= op && op <= 32);
}
bool CountLlvmOp_fence(unsigned op) {
  // Instructions: Fence=30
  return op == 30;
}
bool CountLlvmOp_floats(unsigned op) {
  // Instructions: FAdd=9, FSub=11, FMul=13, FDiv=16, FRem=19, FPToUI=36,
  // FPToSI=37, UIToFP=38, SIToFP=39, FPTrunc=40, FPExt=41, FCmp=47
  return op == 9 || op == 11 || op == 13 || op == 16 || op == 19 || (36 <= op && op <= 41) || op == 47;
}
bool CountLlvmOp_ints(unsigned op) {
  // Instructions: Add=8, Sub=10, Mul=12, SDiv=15, SRem=18, AShr=22, Trunc=33,
  // SExt=35, ICmp=46
  return op == 8 || op == 10 || op == 12 || op == 15 || op == 18 || op == 22 || op == 33 || op == 35 || op == 46;
}
bool CountLlvmOp_uints(unsigned op) {
  // Instructions: UDiv=14, URem=17, Shl=20, LShr=21, And=23, Or=24, Xor=25,
  // ZExt=34
  return op == 14 || op == 17 || (20 <= op && op <= 21) || (23 <= op && op <= 25) || op == 34;
}
// OPCODE-COUNTERS:END

void CountDxilOp(unsigned op, DxilCounters &counters) {
  // <py::lines('COUNT-DXIL-OPS')>['if (CountDxilOp_%s(op)) ++counters.%s;' % (c,c) for c in hctdb_instrhelp.get_dxil_op_counters()]</py>
  // COUNT-DXIL-OPS:BEGIN
  if (CountDxilOp_atomic(op)) ++counters.atomic;
  if (CountDxilOp_barrier(op)) ++counters.barrier;
  if (CountDxilOp_floats(op)) ++counters.floats;
  if (CountDxilOp_gs_cut(op)) ++counters.gs_cut;
  if (CountDxilOp_gs_emit(op)) ++counters.gs_emit;
  if (CountDxilOp_ints(op)) ++counters.ints;
  if (CountDxilOp_sig_ld(op)) ++counters.sig_ld;
  if (CountDxilOp_sig_st(op)) ++counters.sig_st;
  if (CountDxilOp_tex_bias(op)) ++counters.tex_bias;
  if (CountDxilOp_tex_cmp(op)) ++counters.tex_cmp;
  if (CountDxilOp_tex_grad(op)) ++counters.tex_grad;
  if (CountDxilOp_tex_load(op)) ++counters.tex_load;
  if (CountDxilOp_tex_norm(op)) ++counters.tex_norm;
  if (CountDxilOp_tex_store(op)) ++counters.tex_store;
  if (CountDxilOp_uints(op)) ++counters.uints;
  // COUNT-DXIL-OPS:END
}

void CountLlvmOp(unsigned op, DxilCounters &counters) {
  // <py::lines('COUNT-LLVM-OPS')>['if (CountLlvmOp_%s(op)) ++counters.%s;' % (c,c) for c in hctdb_instrhelp.get_llvm_op_counters()]</py>
  // COUNT-LLVM-OPS:BEGIN
  if (CountLlvmOp_atomic(op)) ++counters.atomic;
  if (CountLlvmOp_fence(op)) ++counters.fence;
  if (CountLlvmOp_floats(op)) ++counters.floats;
  if (CountLlvmOp_ints(op)) ++counters.ints;
  if (CountLlvmOp_uints(op)) ++counters.uints;
  // COUNT-LLVM-OPS:END
}

} // namespace

void CountInstructions(llvm::Module &M, DxilCounters& counters) {
  const DataLayout &DL = M.getDataLayout();
  PointerInfoMap ptrInfoMap;

  for (auto &GV : M.globals()) {
    PointerInfo PI = GetPointerInfo(&GV, ptrInfoMap);
    if (PI.isArray) {
      // Count number of bytes used in global arrays.
      Type *pTy = GV.getType()->getPointerElementType();
      uint32_t size = DL.getTypeAllocSize(pTy);
      switch (PI.memType) {
      case PointerInfo::MemType::Global_Static:  counters.array_static_bytes += size;  break;
      case PointerInfo::MemType::Global_TGSM:    counters.array_tgsm_bytes += size;    break;
      default: break;
      }
    }
  }

  for (auto &F : M.functions()) {
    if (F.isDeclaration())
      continue;
    for (auto itBlock = F.begin(), endBlock = F.end(); itBlock != endBlock; ++itBlock) {
      for (auto itInst = itBlock->begin(), endInst = itBlock->end(); itInst != endInst; ++itInst) {
        Instruction* I = itInst;
        ++counters.insts;
        if (AllocaInst *AI = dyn_cast<AllocaInst>(I)) {
          Type *pTy = AI->getType()->getPointerElementType();
          // Count number of bytes used in alloca arrays.
          if (pTy->isArrayTy()) {
            counters.array_local_bytes += DL.getTypeAllocSize(pTy);
          }
        } else if (CallInst *CI = dyn_cast<CallInst>(I)) {
          if (hlsl::OP::IsDxilOpFuncCallInst(CI)) {
            unsigned opcode = (unsigned)llvm::cast<llvm::ConstantInt>(I->getOperand(0))->getZExtValue();
            CountDxilOp(opcode, counters);
          }
        } else if (isa<LoadInst>(I) || isa<StoreInst>(I)) {
          LoadInst  *LI = dyn_cast<LoadInst>(I);
          StoreInst *SI = dyn_cast<StoreInst>(I);
          Value *PtrOp = LI ? LI->getPointerOperand() : SI->getPointerOperand();
          PointerInfo PI = GetPointerInfo(PtrOp, ptrInfoMap);
          // Count load/store on array elements.
          if (PI.isArray) {
            switch (PI.memType) {
            case PointerInfo::MemType::Alloca:         ++counters.array_local_ldst;        break;
            case PointerInfo::MemType::Global_Static:  ++counters.array_static_ldst; break;
            case PointerInfo::MemType::Global_TGSM:    ++counters.array_tgsm_ldst;   break;
            default: break;
            }
          }
        } else if (BranchInst *BI = dyn_cast<BranchInst>(I)) {
          if (BI->getNumSuccessors() > 1) {
            // TODO: More sophisticated analysis to separate dynamic from static branching?
            ++counters.branches;
          }
        } else {
          // Count llvm ops:
          CountLlvmOp(I->getOpcode(), counters);
        }
      }
    }
  }
}

struct CounterOffsetByName {
  StringRef name;
  uint32_t DxilCounters::*ptr;
};

// Must be sorted case-sensitive:
static const CounterOffsetByName CountersByName[] = {
  // <py::lines('COUNTER-MEMBER-PTRS')>['{ "%s", &DxilCounters::%s },' % (c,c) for c in hctdb_instrhelp.get_counters()]</py>
  // COUNTER-MEMBER-PTRS:BEGIN
  { "array_local_bytes", &DxilCounters::array_local_bytes },
  { "array_local_ldst", &DxilCounters::array_local_ldst },
  { "array_static_bytes", &DxilCounters::array_static_bytes },
  { "array_static_ldst", &DxilCounters::array_static_ldst },
  { "array_tgsm_bytes", &DxilCounters::array_tgsm_bytes },
  { "array_tgsm_ldst", &DxilCounters::array_tgsm_ldst },
  { "atomic", &DxilCounters::atomic },
  { "barrier", &DxilCounters::barrier },
  { "branches", &DxilCounters::branches },
  { "fence", &DxilCounters::fence },
  { "floats", &DxilCounters::floats },
  { "gs_cut", &DxilCounters::gs_cut },
  { "gs_emit", &DxilCounters::gs_emit },
  { "insts", &DxilCounters::insts },
  { "ints", &DxilCounters::ints },
  { "sig_ld", &DxilCounters::sig_ld },
  { "sig_st", &DxilCounters::sig_st },
  { "tex_bias", &DxilCounters::tex_bias },
  { "tex_cmp", &DxilCounters::tex_cmp },
  { "tex_grad", &DxilCounters::tex_grad },
  { "tex_load", &DxilCounters::tex_load },
  { "tex_norm", &DxilCounters::tex_norm },
  { "tex_store", &DxilCounters::tex_store },
  { "uints", &DxilCounters::uints },
  // COUNTER-MEMBER-PTRS:END
};

static int CounterOffsetByNameLess(const CounterOffsetByName &a, const CounterOffsetByName &b) {
  return a.name < b.name;
}

uint32_t *LookupByName(llvm::StringRef name, DxilCounters& counters) {
  CounterOffsetByName key = {name, nullptr};
  static const CounterOffsetByName *CounterEnd = CountersByName +_countof(CountersByName);
  auto result = std::lower_bound(CountersByName, CounterEnd, key, CounterOffsetByNameLess);
  if (result != CounterEnd && result->name == key.name)
    return &(counters.*(result->ptr));
  return nullptr;
}


} // namespace hlsl
