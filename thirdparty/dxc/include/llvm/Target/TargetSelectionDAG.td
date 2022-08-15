//===- TargetSelectionDAG.td - Common code for DAG isels ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the target-independent interfaces used by SelectionDAG
// instruction selection generators.
//
//===----------------------------------------------------------------------===//

class SDTypeConstraint<int opnum> {
  int OperandNum = opnum;
}

// SDTCisVT - The specified operand has exactly this VT.
class SDTCisVT<int OpNum, ValueType vt> : SDTypeConstraint<OpNum> {
  ValueType VT = vt;
}

class SDTCisPtrTy<int OpNum> : SDTypeConstraint<OpNum>;

// SDTCisInt - The specified operand has integer type.
class SDTCisInt<int OpNum> : SDTypeConstraint<OpNum>;

// SDTCisFP - The specified operand has floating-point type.
class SDTCisFP<int OpNum> : SDTypeConstraint<OpNum>;

// SDTCisVec - The specified operand has a vector type.
class SDTCisVec<int OpNum> : SDTypeConstraint<OpNum>;

// SDTCisSameAs - The two specified operands have identical types.
class SDTCisSameAs<int OpNum, int OtherOp> : SDTypeConstraint<OpNum> {
  int OtherOperandNum = OtherOp;
}

// SDTCisVTSmallerThanOp - The specified operand is a VT SDNode, and its type is
// smaller than the 'Other' operand.
class SDTCisVTSmallerThanOp<int OpNum, int OtherOp> : SDTypeConstraint<OpNum> {
  int OtherOperandNum = OtherOp;
}

class SDTCisOpSmallerThanOp<int SmallOp, int BigOp> : SDTypeConstraint<SmallOp>{
  int BigOperandNum = BigOp;
}

/// SDTCisEltOfVec - This indicates that ThisOp is a scalar type of the same
/// type as the element type of OtherOp, which is a vector type.
class SDTCisEltOfVec<int ThisOp, int OtherOp>
  : SDTypeConstraint<ThisOp> {
  int OtherOpNum = OtherOp;
}

/// SDTCisSubVecOfVec - This indicates that ThisOp is a vector type
/// with length less that of OtherOp, which is a vector type.
class SDTCisSubVecOfVec<int ThisOp, int OtherOp>
  : SDTypeConstraint<ThisOp> {
  int OtherOpNum = OtherOp;
}

// SDTCVecEltisVT - The specified operand is vector type with element type
// of VT.
class SDTCVecEltisVT<int OpNum, ValueType vt> : SDTypeConstraint<OpNum> {
  ValueType VT = vt;
}

// SDTCisSameNumEltsAs - The two specified operands have identical number
// of elements.
class SDTCisSameNumEltsAs<int OpNum, int OtherOp> : SDTypeConstraint<OpNum> {
  int OtherOperandNum = OtherOp;
}

//===----------------------------------------------------------------------===//
// Selection DAG Type Profile definitions.
//
// These use the constraints defined above to describe the type requirements of
// the various nodes.  These are not hard coded into tblgen, allowing targets to
// add their own if needed.
//

// SDTypeProfile - This profile describes the type requirements of a Selection
// DAG node.
class SDTypeProfile<int numresults, int numoperands,
                    list<SDTypeConstraint> constraints> {
  int NumResults = numresults;
  int NumOperands = numoperands;
  list<SDTypeConstraint> Constraints = constraints;
}

// Builtin profiles.
def SDTIntLeaf: SDTypeProfile<1, 0, [SDTCisInt<0>]>;         // for 'imm'.
def SDTFPLeaf : SDTypeProfile<1, 0, [SDTCisFP<0>]>;          // for 'fpimm'.
def SDTPtrLeaf: SDTypeProfile<1, 0, [SDTCisPtrTy<0>]>;       // for '&g'.
def SDTOther  : SDTypeProfile<1, 0, [SDTCisVT<0, OtherVT>]>; // for 'vt'.
def SDTUNDEF  : SDTypeProfile<1, 0, []>;                     // for 'undef'.
def SDTUnaryOp  : SDTypeProfile<1, 1, []>;                   // for bitconvert.

def SDTIntBinOp : SDTypeProfile<1, 2, [     // add, and, or, xor, udiv, etc.
  SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisInt<0>
]>;
def SDTIntShiftOp : SDTypeProfile<1, 2, [   // shl, sra, srl
  SDTCisSameAs<0, 1>, SDTCisInt<0>, SDTCisInt<2>
]>;
def SDTIntBinHiLoOp : SDTypeProfile<2, 2, [ // mulhi, mullo, sdivrem, udivrem
  SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisSameAs<0, 3>,SDTCisInt<0>
]>;

def SDTFPBinOp : SDTypeProfile<1, 2, [      // fadd, fmul, etc.
  SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisFP<0>
]>;
def SDTFPSignOp : SDTypeProfile<1, 2, [     // fcopysign.
  SDTCisSameAs<0, 1>, SDTCisFP<0>, SDTCisFP<2>
]>;
def SDTFPTernaryOp : SDTypeProfile<1, 3, [  // fmadd, fnmsub, etc.
  SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisSameAs<0, 3>, SDTCisFP<0>
]>;
def SDTIntUnaryOp : SDTypeProfile<1, 1, [   // ctlz
  SDTCisSameAs<0, 1>, SDTCisInt<0>
]>;
def SDTIntExtendOp : SDTypeProfile<1, 1, [  // sext, zext, anyext
  SDTCisInt<0>, SDTCisInt<1>, SDTCisOpSmallerThanOp<1, 0>
]>;
def SDTIntTruncOp  : SDTypeProfile<1, 1, [  // trunc
  SDTCisInt<0>, SDTCisInt<1>, SDTCisOpSmallerThanOp<0, 1>
]>;
def SDTFPUnaryOp  : SDTypeProfile<1, 1, [   // fneg, fsqrt, etc
  SDTCisSameAs<0, 1>, SDTCisFP<0>
]>;
def SDTFPRoundOp  : SDTypeProfile<1, 1, [   // fround
  SDTCisFP<0>, SDTCisFP<1>, SDTCisOpSmallerThanOp<0, 1>
]>;
def SDTFPExtendOp  : SDTypeProfile<1, 1, [  // fextend
  SDTCisFP<0>, SDTCisFP<1>, SDTCisOpSmallerThanOp<1, 0>
]>;
def SDTIntToFPOp : SDTypeProfile<1, 1, [    // [su]int_to_fp
  SDTCisFP<0>, SDTCisInt<1>
]>;
def SDTFPToIntOp : SDTypeProfile<1, 1, [    // fp_to_[su]int
  SDTCisInt<0>, SDTCisFP<1>
]>;
def SDTExtInreg : SDTypeProfile<1, 2, [     // sext_inreg
  SDTCisSameAs<0, 1>, SDTCisInt<0>, SDTCisVT<2, OtherVT>,
  SDTCisVTSmallerThanOp<2, 1>
]>;

def SDTSetCC : SDTypeProfile<1, 3, [        // setcc
  SDTCisInt<0>, SDTCisSameAs<1, 2>, SDTCisVT<3, OtherVT>
]>;

def SDTSelect : SDTypeProfile<1, 3, [       // select
  SDTCisInt<1>, SDTCisSameAs<0, 2>, SDTCisSameAs<2, 3>
]>;

def SDTVSelect : SDTypeProfile<1, 3, [       // vselect
  SDTCisInt<1>, SDTCisSameAs<0, 2>, SDTCisSameAs<2, 3>
]>;

def SDTSelectCC : SDTypeProfile<1, 5, [     // select_cc
  SDTCisSameAs<1, 2>, SDTCisSameAs<3, 4>, SDTCisSameAs<0, 3>,
  SDTCisVT<5, OtherVT>
]>;

def SDTBr : SDTypeProfile<0, 1, [           // br
  SDTCisVT<0, OtherVT>
]>;

def SDTBrCC : SDTypeProfile<0, 4, [       // brcc
  SDTCisVT<0, OtherVT>, SDTCisSameAs<1, 2>, SDTCisVT<3, OtherVT>
]>;

def SDTBrcond : SDTypeProfile<0, 2, [       // brcond
  SDTCisInt<0>, SDTCisVT<1, OtherVT>
]>;

def SDTBrind : SDTypeProfile<0, 1, [        // brind
  SDTCisPtrTy<0>
]>;

def SDTNone : SDTypeProfile<0, 0, []>;      // ret, trap

def SDTLoad : SDTypeProfile<1, 1, [         // load
  SDTCisPtrTy<1>
]>;

def SDTStore : SDTypeProfile<0, 2, [        // store
  SDTCisPtrTy<1>
]>;

def SDTIStore : SDTypeProfile<1, 3, [       // indexed store
  SDTCisSameAs<0, 2>, SDTCisPtrTy<0>, SDTCisPtrTy<3>
]>;

def SDTMaskedStore: SDTypeProfile<0, 3, [       // masked store
  SDTCisPtrTy<0>, SDTCisVec<1>, SDTCisVec<2>
]>;

def SDTMaskedLoad: SDTypeProfile<1, 3, [       // masked load
  SDTCisVec<0>, SDTCisPtrTy<1>, SDTCisVec<2>, SDTCisSameAs<0, 3>
]>;

def SDTMaskedGather: SDTypeProfile<2, 3, [       // masked gather
  SDTCisVec<0>, SDTCisVec<1>, SDTCisSameAs<0, 2>, SDTCisSameAs<1, 3>,
  SDTCisPtrTy<4>, SDTCVecEltisVT<1, i1>, SDTCisSameNumEltsAs<0, 1>
]>;

def SDTMaskedScatter: SDTypeProfile<1, 3, [       // masked scatter
  SDTCisVec<0>, SDTCisVec<1>, SDTCisSameAs<0, 2>, SDTCisSameNumEltsAs<0, 1>,
  SDTCVecEltisVT<0, i1>, SDTCisPtrTy<3>
]>;

def SDTVecShuffle : SDTypeProfile<1, 2, [
  SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>
]>;
def SDTVecExtract : SDTypeProfile<1, 2, [   // vector extract
  SDTCisEltOfVec<0, 1>, SDTCisPtrTy<2>
]>;
def SDTVecInsert : SDTypeProfile<1, 3, [    // vector insert
  SDTCisEltOfVec<2, 1>, SDTCisSameAs<0, 1>, SDTCisPtrTy<3>
]>;

def SDTSubVecExtract : SDTypeProfile<1, 2, [// subvector extract
  SDTCisSubVecOfVec<0,1>, SDTCisInt<2>
]>;
def SDTSubVecInsert : SDTypeProfile<1, 3, [ // subvector insert
  SDTCisSubVecOfVec<2, 1>, SDTCisSameAs<0,1>, SDTCisInt<3>
]>;

def SDTPrefetch : SDTypeProfile<0, 4, [     // prefetch
  SDTCisPtrTy<0>, SDTCisSameAs<1, 2>, SDTCisSameAs<1, 3>, SDTCisInt<1>
]>;

def SDTMemBarrier : SDTypeProfile<0, 5, [   // memory barrier
  SDTCisSameAs<0,1>,  SDTCisSameAs<0,2>,  SDTCisSameAs<0,3>, SDTCisSameAs<0,4>,
  SDTCisInt<0>
]>;
def SDTAtomicFence : SDTypeProfile<0, 2, [
  SDTCisSameAs<0,1>, SDTCisPtrTy<0>
]>;
def SDTAtomic3 : SDTypeProfile<1, 3, [
  SDTCisSameAs<0,2>,  SDTCisSameAs<0,3>, SDTCisInt<0>, SDTCisPtrTy<1>
]>;
def SDTAtomic2 : SDTypeProfile<1, 2, [
  SDTCisSameAs<0,2>, SDTCisInt<0>, SDTCisPtrTy<1>
]>;
def SDTAtomicStore : SDTypeProfile<0, 2, [
  SDTCisPtrTy<0>, SDTCisInt<1>
]>;
def SDTAtomicLoad : SDTypeProfile<1, 1, [
  SDTCisInt<0>, SDTCisPtrTy<1>
]>;

def SDTConvertOp : SDTypeProfile<1, 5, [ //cvtss, su, us, uu, ff, fs, fu, sf, su
  SDTCisVT<2, OtherVT>, SDTCisVT<3, OtherVT>, SDTCisPtrTy<4>, SDTCisPtrTy<5>
]>;

class SDCallSeqStart<list<SDTypeConstraint> constraints> :
        SDTypeProfile<0, 1, constraints>;
class SDCallSeqEnd<list<SDTypeConstraint> constraints> :
        SDTypeProfile<0, 2, constraints>;

//===----------------------------------------------------------------------===//
// Selection DAG Node Properties.
//
// Note: These are hard coded into tblgen.
//
class SDNodeProperty;
def SDNPCommutative : SDNodeProperty;   // X op Y == Y op X
def SDNPAssociative : SDNodeProperty;   // (X op Y) op Z == X op (Y op Z)
def SDNPHasChain    : SDNodeProperty;   // R/W chain operand and result
def SDNPOutGlue     : SDNodeProperty;   // Write a flag result
def SDNPInGlue      : SDNodeProperty;   // Read a flag operand
def SDNPOptInGlue   : SDNodeProperty;   // Optionally read a flag operand
def SDNPMayStore    : SDNodeProperty;   // May write to memory, sets 'mayStore'.
def SDNPMayLoad     : SDNodeProperty;   // May read memory, sets 'mayLoad'.
def SDNPSideEffect  : SDNodeProperty;   // Sets 'HasUnmodelledSideEffects'.
def SDNPMemOperand  : SDNodeProperty;   // Touches memory, has assoc MemOperand
def SDNPVariadic    : SDNodeProperty;   // Node has variable arguments.
def SDNPWantRoot    : SDNodeProperty;   // ComplexPattern gets the root of match
def SDNPWantParent  : SDNodeProperty;   // ComplexPattern gets the parent

//===----------------------------------------------------------------------===//
// Selection DAG Pattern Operations
class SDPatternOperator;

//===----------------------------------------------------------------------===//
// Selection DAG Node definitions.
//
class SDNode<string opcode, SDTypeProfile typeprof,
             list<SDNodeProperty> props = [], string sdclass = "SDNode">
             : SDPatternOperator {
  string Opcode  = opcode;
  string SDClass = sdclass;
  list<SDNodeProperty> Properties = props;
  SDTypeProfile TypeProfile = typeprof;
}

// Special TableGen-recognized dag nodes
def set;
def implicit;
def node;
def srcvalue;

def imm        : SDNode<"ISD::Constant"  , SDTIntLeaf , [], "ConstantSDNode">;
def timm       : SDNode<"ISD::TargetConstant",SDTIntLeaf, [], "ConstantSDNode">;
def fpimm      : SDNode<"ISD::ConstantFP", SDTFPLeaf  , [], "ConstantFPSDNode">;
def vt         : SDNode<"ISD::VALUETYPE" , SDTOther   , [], "VTSDNode">;
def bb         : SDNode<"ISD::BasicBlock", SDTOther   , [], "BasicBlockSDNode">;
def cond       : SDNode<"ISD::CONDCODE"  , SDTOther   , [], "CondCodeSDNode">;
def undef      : SDNode<"ISD::UNDEF"     , SDTUNDEF   , []>;
def globaladdr : SDNode<"ISD::GlobalAddress",         SDTPtrLeaf, [],
                        "GlobalAddressSDNode">;
def tglobaladdr : SDNode<"ISD::TargetGlobalAddress",  SDTPtrLeaf, [],
                         "GlobalAddressSDNode">;
def globaltlsaddr : SDNode<"ISD::GlobalTLSAddress",         SDTPtrLeaf, [],
                          "GlobalAddressSDNode">;
def tglobaltlsaddr : SDNode<"ISD::TargetGlobalTLSAddress",  SDTPtrLeaf, [],
                           "GlobalAddressSDNode">;
def constpool   : SDNode<"ISD::ConstantPool",         SDTPtrLeaf, [],
                         "ConstantPoolSDNode">;
def tconstpool  : SDNode<"ISD::TargetConstantPool",   SDTPtrLeaf, [],
                         "ConstantPoolSDNode">;
def jumptable   : SDNode<"ISD::JumpTable",            SDTPtrLeaf, [],
                         "JumpTableSDNode">;
def tjumptable  : SDNode<"ISD::TargetJumpTable",      SDTPtrLeaf, [],
                         "JumpTableSDNode">;
def frameindex  : SDNode<"ISD::FrameIndex",           SDTPtrLeaf, [],
                         "FrameIndexSDNode">;
def tframeindex : SDNode<"ISD::TargetFrameIndex",     SDTPtrLeaf, [],
                         "FrameIndexSDNode">;
def externalsym : SDNode<"ISD::ExternalSymbol",       SDTPtrLeaf, [],
                         "ExternalSymbolSDNode">;
def texternalsym: SDNode<"ISD::TargetExternalSymbol", SDTPtrLeaf, [],
                         "ExternalSymbolSDNode">;
def mcsym: SDNode<"ISD::MCSymbol", SDTPtrLeaf, [], "MCSymbolSDNode">;
def blockaddress : SDNode<"ISD::BlockAddress",        SDTPtrLeaf, [],
                         "BlockAddressSDNode">;
def tblockaddress: SDNode<"ISD::TargetBlockAddress",  SDTPtrLeaf, [],
                         "BlockAddressSDNode">;

def add        : SDNode<"ISD::ADD"       , SDTIntBinOp   ,
                        [SDNPCommutative, SDNPAssociative]>;
def sub        : SDNode<"ISD::SUB"       , SDTIntBinOp>;
def mul        : SDNode<"ISD::MUL"       , SDTIntBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def mulhs      : SDNode<"ISD::MULHS"     , SDTIntBinOp, [SDNPCommutative]>;
def mulhu      : SDNode<"ISD::MULHU"     , SDTIntBinOp, [SDNPCommutative]>;
def smullohi   : SDNode<"ISD::SMUL_LOHI" , SDTIntBinHiLoOp, [SDNPCommutative]>;
def umullohi   : SDNode<"ISD::UMUL_LOHI" , SDTIntBinHiLoOp, [SDNPCommutative]>;
def sdiv       : SDNode<"ISD::SDIV"      , SDTIntBinOp>;
def udiv       : SDNode<"ISD::UDIV"      , SDTIntBinOp>;
def srem       : SDNode<"ISD::SREM"      , SDTIntBinOp>;
def urem       : SDNode<"ISD::UREM"      , SDTIntBinOp>;
def sdivrem    : SDNode<"ISD::SDIVREM"   , SDTIntBinHiLoOp>;
def udivrem    : SDNode<"ISD::UDIVREM"   , SDTIntBinHiLoOp>;
def srl        : SDNode<"ISD::SRL"       , SDTIntShiftOp>;
def sra        : SDNode<"ISD::SRA"       , SDTIntShiftOp>;
def shl        : SDNode<"ISD::SHL"       , SDTIntShiftOp>;
def rotl       : SDNode<"ISD::ROTL"      , SDTIntShiftOp>;
def rotr       : SDNode<"ISD::ROTR"      , SDTIntShiftOp>;
def and        : SDNode<"ISD::AND"       , SDTIntBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def or         : SDNode<"ISD::OR"        , SDTIntBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def xor        : SDNode<"ISD::XOR"       , SDTIntBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def addc       : SDNode<"ISD::ADDC"      , SDTIntBinOp,
                        [SDNPCommutative, SDNPOutGlue]>;
def adde       : SDNode<"ISD::ADDE"      , SDTIntBinOp,
                        [SDNPCommutative, SDNPOutGlue, SDNPInGlue]>;
def subc       : SDNode<"ISD::SUBC"      , SDTIntBinOp,
                        [SDNPOutGlue]>;
def sube       : SDNode<"ISD::SUBE"      , SDTIntBinOp,
                        [SDNPOutGlue, SDNPInGlue]>;
def smin       : SDNode<"ISD::SMIN"      , SDTIntBinOp>;
def smax       : SDNode<"ISD::SMAX"      , SDTIntBinOp>;
def umin       : SDNode<"ISD::UMIN"      , SDTIntBinOp>;
def umax       : SDNode<"ISD::UMAX"      , SDTIntBinOp>;

def sext_inreg : SDNode<"ISD::SIGN_EXTEND_INREG", SDTExtInreg>;
def bswap      : SDNode<"ISD::BSWAP"      , SDTIntUnaryOp>;
def ctlz       : SDNode<"ISD::CTLZ"       , SDTIntUnaryOp>;
def cttz       : SDNode<"ISD::CTTZ"       , SDTIntUnaryOp>;
def ctpop      : SDNode<"ISD::CTPOP"      , SDTIntUnaryOp>;
def ctlz_zero_undef : SDNode<"ISD::CTLZ_ZERO_UNDEF", SDTIntUnaryOp>;
def cttz_zero_undef : SDNode<"ISD::CTTZ_ZERO_UNDEF", SDTIntUnaryOp>;
def sext       : SDNode<"ISD::SIGN_EXTEND", SDTIntExtendOp>;
def zext       : SDNode<"ISD::ZERO_EXTEND", SDTIntExtendOp>;
def anyext     : SDNode<"ISD::ANY_EXTEND" , SDTIntExtendOp>;
def trunc      : SDNode<"ISD::TRUNCATE"   , SDTIntTruncOp>;
def bitconvert : SDNode<"ISD::BITCAST"    , SDTUnaryOp>;
def addrspacecast : SDNode<"ISD::ADDRSPACECAST", SDTUnaryOp>;
def extractelt : SDNode<"ISD::EXTRACT_VECTOR_ELT", SDTVecExtract>;
def insertelt  : SDNode<"ISD::INSERT_VECTOR_ELT", SDTVecInsert>;

def fadd       : SDNode<"ISD::FADD"       , SDTFPBinOp, [SDNPCommutative]>;
def fsub       : SDNode<"ISD::FSUB"       , SDTFPBinOp>;
def fmul       : SDNode<"ISD::FMUL"       , SDTFPBinOp, [SDNPCommutative]>;
def fdiv       : SDNode<"ISD::FDIV"       , SDTFPBinOp>;
def frem       : SDNode<"ISD::FREM"       , SDTFPBinOp>;
def fma        : SDNode<"ISD::FMA"        , SDTFPTernaryOp>;
def fmad       : SDNode<"ISD::FMAD"       , SDTFPTernaryOp>;
def fabs       : SDNode<"ISD::FABS"       , SDTFPUnaryOp>;
def fminnum    : SDNode<"ISD::FMINNUM"    , SDTFPBinOp>;
def fmaxnum    : SDNode<"ISD::FMAXNUM"    , SDTFPBinOp>;
def fgetsign   : SDNode<"ISD::FGETSIGN"   , SDTFPToIntOp>;
def fneg       : SDNode<"ISD::FNEG"       , SDTFPUnaryOp>;
def fsqrt      : SDNode<"ISD::FSQRT"      , SDTFPUnaryOp>;
def fsin       : SDNode<"ISD::FSIN"       , SDTFPUnaryOp>;
def fcos       : SDNode<"ISD::FCOS"       , SDTFPUnaryOp>;
def fexp2      : SDNode<"ISD::FEXP2"      , SDTFPUnaryOp>;
def fpow       : SDNode<"ISD::FPOW"       , SDTFPBinOp>;
def flog2      : SDNode<"ISD::FLOG2"      , SDTFPUnaryOp>;
def frint      : SDNode<"ISD::FRINT"      , SDTFPUnaryOp>;
def ftrunc     : SDNode<"ISD::FTRUNC"     , SDTFPUnaryOp>;
def fceil      : SDNode<"ISD::FCEIL"      , SDTFPUnaryOp>;
def ffloor     : SDNode<"ISD::FFLOOR"     , SDTFPUnaryOp>;
def fnearbyint : SDNode<"ISD::FNEARBYINT" , SDTFPUnaryOp>;
def frnd       : SDNode<"ISD::FROUND"     , SDTFPUnaryOp>;

def fround     : SDNode<"ISD::FP_ROUND"   , SDTFPRoundOp>;
def fextend    : SDNode<"ISD::FP_EXTEND"  , SDTFPExtendOp>;
def fcopysign  : SDNode<"ISD::FCOPYSIGN"  , SDTFPSignOp>;

def sint_to_fp : SDNode<"ISD::SINT_TO_FP" , SDTIntToFPOp>;
def uint_to_fp : SDNode<"ISD::UINT_TO_FP" , SDTIntToFPOp>;
def fp_to_sint : SDNode<"ISD::FP_TO_SINT" , SDTFPToIntOp>;
def fp_to_uint : SDNode<"ISD::FP_TO_UINT" , SDTFPToIntOp>;
def f16_to_fp  : SDNode<"ISD::FP16_TO_FP" , SDTIntToFPOp>;
def fp_to_f16  : SDNode<"ISD::FP_TO_FP16" , SDTFPToIntOp>;

def setcc      : SDNode<"ISD::SETCC"      , SDTSetCC>;
def select     : SDNode<"ISD::SELECT"     , SDTSelect>;
def vselect    : SDNode<"ISD::VSELECT"    , SDTVSelect>;
def selectcc   : SDNode<"ISD::SELECT_CC"  , SDTSelectCC>;

def brcc       : SDNode<"ISD::BR_CC"      , SDTBrCC,   [SDNPHasChain]>;
def brcond     : SDNode<"ISD::BRCOND"     , SDTBrcond, [SDNPHasChain]>;
def brind      : SDNode<"ISD::BRIND"      , SDTBrind,  [SDNPHasChain]>;
def br         : SDNode<"ISD::BR"         , SDTBr,     [SDNPHasChain]>;
def trap       : SDNode<"ISD::TRAP"       , SDTNone,
                        [SDNPHasChain, SDNPSideEffect]>;
def debugtrap  : SDNode<"ISD::DEBUGTRAP"  , SDTNone,
                        [SDNPHasChain, SDNPSideEffect]>;

def prefetch   : SDNode<"ISD::PREFETCH"   , SDTPrefetch,
                        [SDNPHasChain, SDNPMayLoad, SDNPMayStore,
                         SDNPMemOperand]>;

def readcyclecounter : SDNode<"ISD::READCYCLECOUNTER", SDTIntLeaf,
                     [SDNPHasChain, SDNPSideEffect]>;

def atomic_fence : SDNode<"ISD::ATOMIC_FENCE" , SDTAtomicFence,
                          [SDNPHasChain, SDNPSideEffect]>;

def atomic_cmp_swap : SDNode<"ISD::ATOMIC_CMP_SWAP" , SDTAtomic3,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_add : SDNode<"ISD::ATOMIC_LOAD_ADD" , SDTAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_swap     : SDNode<"ISD::ATOMIC_SWAP", SDTAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_sub : SDNode<"ISD::ATOMIC_LOAD_SUB" , SDTAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_and : SDNode<"ISD::ATOMIC_LOAD_AND" , SDTAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_or  : SDNode<"ISD::ATOMIC_LOAD_OR" , SDTAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_xor : SDNode<"ISD::ATOMIC_LOAD_XOR" , SDTAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_nand: SDNode<"ISD::ATOMIC_LOAD_NAND", SDTAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_min : SDNode<"ISD::ATOMIC_LOAD_MIN", SDTAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_max : SDNode<"ISD::ATOMIC_LOAD_MAX", SDTAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_umin : SDNode<"ISD::ATOMIC_LOAD_UMIN", SDTAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_umax : SDNode<"ISD::ATOMIC_LOAD_UMAX", SDTAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load      : SDNode<"ISD::ATOMIC_LOAD", SDTAtomicLoad,
                    [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def atomic_store     : SDNode<"ISD::ATOMIC_STORE", SDTAtomicStore,
                    [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

def masked_store : SDNode<"ISD::MSTORE",  SDTMaskedStore,
                       [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def masked_load  : SDNode<"ISD::MLOAD",  SDTMaskedLoad,
                       [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def masked_scatter : SDNode<"ISD::MSCATTER",  SDTMaskedScatter,
                       [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def masked_gather  : SDNode<"ISD::MGATHER",  SDTMaskedGather,
                       [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;

// Do not use ld, st directly. Use load, extload, sextload, zextload, store,
// and truncst (see below).
def ld         : SDNode<"ISD::LOAD"       , SDTLoad,
                        [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def st         : SDNode<"ISD::STORE"      , SDTStore,
                        [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def ist        : SDNode<"ISD::STORE"      , SDTIStore,
                        [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

def vector_shuffle : SDNode<"ISD::VECTOR_SHUFFLE", SDTVecShuffle, []>;
def build_vector : SDNode<"ISD::BUILD_VECTOR", SDTypeProfile<1, -1, []>, []>;
def scalar_to_vector : SDNode<"ISD::SCALAR_TO_VECTOR", SDTypeProfile<1, 1, []>,
                              []>;
def vector_extract : SDNode<"ISD::EXTRACT_VECTOR_ELT",
    SDTypeProfile<1, 2, [SDTCisPtrTy<2>]>, []>;
def vector_insert : SDNode<"ISD::INSERT_VECTOR_ELT",
    SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>, SDTCisPtrTy<3>]>, []>;
def concat_vectors : SDNode<"ISD::CONCAT_VECTORS",
    SDTypeProfile<1, 2, [SDTCisSubVecOfVec<1, 0>, SDTCisSameAs<1, 2>]>,[]>;

// This operator does not do subvector type checking.  The ARM
// backend, at least, needs it.
def vector_extract_subvec : SDNode<"ISD::EXTRACT_SUBVECTOR",
    SDTypeProfile<1, 2, [SDTCisInt<2>, SDTCisVec<1>, SDTCisVec<0>]>, 
    []>;

// This operator does subvector type checking.
def extract_subvector : SDNode<"ISD::EXTRACT_SUBVECTOR", SDTSubVecExtract, []>;
def insert_subvector : SDNode<"ISD::INSERT_SUBVECTOR", SDTSubVecInsert, []>;

// Nodes for intrinsics, you should use the intrinsic itself and let tblgen use
// these internally.  Don't reference these directly.
def intrinsic_void : SDNode<"ISD::INTRINSIC_VOID",
                            SDTypeProfile<0, -1, [SDTCisPtrTy<0>]>,
                            [SDNPHasChain]>;
def intrinsic_w_chain : SDNode<"ISD::INTRINSIC_W_CHAIN",
                               SDTypeProfile<1, -1, [SDTCisPtrTy<1>]>,
                               [SDNPHasChain]>;
def intrinsic_wo_chain : SDNode<"ISD::INTRINSIC_WO_CHAIN",
                                SDTypeProfile<1, -1, [SDTCisPtrTy<1>]>, []>;

// Do not use cvt directly. Use cvt forms below
def cvt : SDNode<"ISD::CONVERT_RNDSAT", SDTConvertOp>;

def SDT_assertext : SDTypeProfile<1, 1,
  [SDTCisInt<0>, SDTCisInt<1>, SDTCisSameAs<1, 0>]>;
def assertsext : SDNode<"ISD::AssertSext", SDT_assertext>;
def assertzext : SDNode<"ISD::AssertZext", SDT_assertext>;


//===----------------------------------------------------------------------===//
// Selection DAG Condition Codes

class CondCode; // ISD::CondCode enums
def SETOEQ : CondCode; def SETOGT : CondCode;
def SETOGE : CondCode; def SETOLT : CondCode; def SETOLE : CondCode;
def SETONE : CondCode; def SETO   : CondCode; def SETUO  : CondCode;
def SETUEQ : CondCode; def SETUGT : CondCode; def SETUGE : CondCode;
def SETULT : CondCode; def SETULE : CondCode; def SETUNE : CondCode;

def SETEQ : CondCode; def SETGT : CondCode; def SETGE : CondCode;
def SETLT : CondCode; def SETLE : CondCode; def SETNE : CondCode;


//===----------------------------------------------------------------------===//
// Selection DAG Node Transformation Functions.
//
// This mechanism allows targets to manipulate nodes in the output DAG once a
// match has been formed.  This is typically used to manipulate immediate
// values.
//
class SDNodeXForm<SDNode opc, code xformFunction> {
  SDNode Opcode = opc;
  code XFormFunction = xformFunction;
}

def NOOP_SDNodeXForm : SDNodeXForm<imm, [{}]>;

//===----------------------------------------------------------------------===//
// PatPred Subclasses.
//
// These allow specifying different sorts of predicates that control whether a
// node is matched.
//
class PatPred;

class CodePatPred<code predicate> : PatPred {
  code PredicateCode = predicate;
}


//===----------------------------------------------------------------------===//
// Selection DAG Pattern Fragments.
//
// Pattern fragments are reusable chunks of dags that match specific things.
// They can take arguments and have C++ predicates that control whether they
// match.  They are intended to make the patterns for common instructions more
// compact and readable.
//

/// PatFrag - Represents a pattern fragment.  This can match something on the
/// DAG, from a single node to multiple nested other fragments.
///
class PatFrag<dag ops, dag frag, code pred = [{}],
              SDNodeXForm xform = NOOP_SDNodeXForm> : SDPatternOperator {
  dag Operands = ops;
  dag Fragment = frag;
  code PredicateCode = pred;
  code ImmediateCode = [{}];
  SDNodeXForm OperandTransform = xform;
}

// OutPatFrag is a pattern fragment that is used as part of an output pattern
// (not an input pattern). These do not have predicates or transforms, but are
// used to avoid repeated subexpressions in output patterns.
class OutPatFrag<dag ops, dag frag>
 : PatFrag<ops, frag, [{}], NOOP_SDNodeXForm>;

// PatLeaf's are pattern fragments that have no operands.  This is just a helper
// to define immediates and other common things concisely.
class PatLeaf<dag frag, code pred = [{}], SDNodeXForm xform = NOOP_SDNodeXForm>
 : PatFrag<(ops), frag, pred, xform>;


// ImmLeaf is a pattern fragment with a constraint on the immediate.  The
// constraint is a function that is run on the immediate (always with the value
// sign extended out to an int64_t) as Imm.  For example:
//
//  def immSExt8 : ImmLeaf<i16, [{ return (char)Imm == Imm; }]>;
//
// this is a more convenient form to match 'imm' nodes in than PatLeaf and also
// is preferred over using PatLeaf because it allows the code generator to
// reason more about the constraint.
//
// If FastIsel should ignore all instructions that have an operand of this type,
// the FastIselShouldIgnore flag can be set.  This is an optimization to reduce
// the code size of the generated fast instruction selector.
class ImmLeaf<ValueType vt, code pred, SDNodeXForm xform = NOOP_SDNodeXForm>
  : PatFrag<(ops), (vt imm), [{}], xform> {
  let ImmediateCode = pred;
  bit FastIselShouldIgnore = 0;
}


// Leaf fragments.

def vtInt      : PatLeaf<(vt),  [{ return N->getVT().isInteger(); }]>;
def vtFP       : PatLeaf<(vt),  [{ return N->getVT().isFloatingPoint(); }]>;

def immAllOnesV: PatLeaf<(build_vector), [{
  return ISD::isBuildVectorAllOnes(N);
}]>;
def immAllZerosV: PatLeaf<(build_vector), [{
  return ISD::isBuildVectorAllZeros(N);
}]>;



// Other helper fragments.
def not  : PatFrag<(ops node:$in), (xor node:$in, -1)>;
def vnot : PatFrag<(ops node:$in), (xor node:$in, immAllOnesV)>;
def ineg : PatFrag<(ops node:$in), (sub 0, node:$in)>;

// null_frag - The null pattern operator is used in multiclass instantiations
// which accept an SDPatternOperator for use in matching patterns for internal
// definitions. When expanding a pattern, if the null fragment is referenced
// in the expansion, the pattern is discarded and it is as-if '[]' had been
// specified. This allows multiclasses to have the isel patterns be optional.
def null_frag : SDPatternOperator;

// load fragments.
def unindexedload : PatFrag<(ops node:$ptr), (ld node:$ptr), [{
  return cast<LoadSDNode>(N)->getAddressingMode() == ISD::UNINDEXED;
}]>;
def load : PatFrag<(ops node:$ptr), (unindexedload node:$ptr), [{
  return cast<LoadSDNode>(N)->getExtensionType() == ISD::NON_EXTLOAD;
}]>;

// extending load fragments.
def extload   : PatFrag<(ops node:$ptr), (unindexedload node:$ptr), [{
  return cast<LoadSDNode>(N)->getExtensionType() == ISD::EXTLOAD;
}]>;
def sextload  : PatFrag<(ops node:$ptr), (unindexedload node:$ptr), [{
  return cast<LoadSDNode>(N)->getExtensionType() == ISD::SEXTLOAD;
}]>;
def zextload  : PatFrag<(ops node:$ptr), (unindexedload node:$ptr), [{
  return cast<LoadSDNode>(N)->getExtensionType() == ISD::ZEXTLOAD;
}]>;

def extloadi1  : PatFrag<(ops node:$ptr), (extload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i1;
}]>;
def extloadi8  : PatFrag<(ops node:$ptr), (extload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i8;
}]>;
def extloadi16 : PatFrag<(ops node:$ptr), (extload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i16;
}]>;
def extloadi32 : PatFrag<(ops node:$ptr), (extload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i32;
}]>;
def extloadf32 : PatFrag<(ops node:$ptr), (extload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::f32;
}]>;
def extloadf64 : PatFrag<(ops node:$ptr), (extload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::f64;
}]>;

def sextloadi1  : PatFrag<(ops node:$ptr), (sextload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i1;
}]>;
def sextloadi8  : PatFrag<(ops node:$ptr), (sextload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i8;
}]>;
def sextloadi16 : PatFrag<(ops node:$ptr), (sextload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i16;
}]>;
def sextloadi32 : PatFrag<(ops node:$ptr), (sextload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i32;
}]>;

def zextloadi1  : PatFrag<(ops node:$ptr), (zextload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i1;
}]>;
def zextloadi8  : PatFrag<(ops node:$ptr), (zextload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i8;
}]>;
def zextloadi16 : PatFrag<(ops node:$ptr), (zextload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i16;
}]>;
def zextloadi32 : PatFrag<(ops node:$ptr), (zextload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i32;
}]>;

def extloadvi1  : PatFrag<(ops node:$ptr), (extload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT().getScalarType() == MVT::i1;
}]>;
def extloadvi8  : PatFrag<(ops node:$ptr), (extload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT().getScalarType() == MVT::i8;
}]>;
def extloadvi16 : PatFrag<(ops node:$ptr), (extload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT().getScalarType() == MVT::i16;
}]>;
def extloadvi32 : PatFrag<(ops node:$ptr), (extload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT().getScalarType() == MVT::i32;
}]>;
def extloadvf32 : PatFrag<(ops node:$ptr), (extload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT().getScalarType() == MVT::f32;
}]>;
def extloadvf64 : PatFrag<(ops node:$ptr), (extload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT().getScalarType() == MVT::f64;
}]>;

def sextloadvi1  : PatFrag<(ops node:$ptr), (sextload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT().getScalarType() == MVT::i1;
}]>;
def sextloadvi8  : PatFrag<(ops node:$ptr), (sextload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT().getScalarType() == MVT::i8;
}]>;
def sextloadvi16 : PatFrag<(ops node:$ptr), (sextload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT().getScalarType() == MVT::i16;
}]>;
def sextloadvi32 : PatFrag<(ops node:$ptr), (sextload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT().getScalarType() == MVT::i32;
}]>;

def zextloadvi1  : PatFrag<(ops node:$ptr), (zextload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT().getScalarType() == MVT::i1;
}]>;
def zextloadvi8  : PatFrag<(ops node:$ptr), (zextload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT().getScalarType() == MVT::i8;
}]>;
def zextloadvi16 : PatFrag<(ops node:$ptr), (zextload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT().getScalarType() == MVT::i16;
}]>;
def zextloadvi32 : PatFrag<(ops node:$ptr), (zextload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT().getScalarType() == MVT::i32;
}]>;

// store fragments.
def unindexedstore : PatFrag<(ops node:$val, node:$ptr),
                             (st node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getAddressingMode() == ISD::UNINDEXED;
}]>;
def store : PatFrag<(ops node:$val, node:$ptr),
                    (unindexedstore node:$val, node:$ptr), [{
  return !cast<StoreSDNode>(N)->isTruncatingStore();
}]>;

// truncstore fragments.
def truncstore : PatFrag<(ops node:$val, node:$ptr),
                         (unindexedstore node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->isTruncatingStore();
}]>;
def truncstorei8 : PatFrag<(ops node:$val, node:$ptr),
                           (truncstore node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::i8;
}]>;
def truncstorei16 : PatFrag<(ops node:$val, node:$ptr),
                            (truncstore node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::i16;
}]>;
def truncstorei32 : PatFrag<(ops node:$val, node:$ptr),
                            (truncstore node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::i32;
}]>;
def truncstoref32 : PatFrag<(ops node:$val, node:$ptr),
                            (truncstore node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::f32;
}]>;
def truncstoref64 : PatFrag<(ops node:$val, node:$ptr),
                            (truncstore node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::f64;
}]>;

// indexed store fragments.
def istore : PatFrag<(ops node:$val, node:$base, node:$offset),
                     (ist node:$val, node:$base, node:$offset), [{
  return !cast<StoreSDNode>(N)->isTruncatingStore();
}]>;

def pre_store : PatFrag<(ops node:$val, node:$base, node:$offset),
                        (istore node:$val, node:$base, node:$offset), [{
  ISD::MemIndexedMode AM = cast<StoreSDNode>(N)->getAddressingMode();
  return AM == ISD::PRE_INC || AM == ISD::PRE_DEC;
}]>;

def itruncstore : PatFrag<(ops node:$val, node:$base, node:$offset),
                          (ist node:$val, node:$base, node:$offset), [{
  return cast<StoreSDNode>(N)->isTruncatingStore();
}]>;
def pre_truncst : PatFrag<(ops node:$val, node:$base, node:$offset),
                          (itruncstore node:$val, node:$base, node:$offset), [{
  ISD::MemIndexedMode AM = cast<StoreSDNode>(N)->getAddressingMode();
  return AM == ISD::PRE_INC || AM == ISD::PRE_DEC;
}]>;
def pre_truncsti1 : PatFrag<(ops node:$val, node:$base, node:$offset),
                            (pre_truncst node:$val, node:$base, node:$offset), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::i1;
}]>;
def pre_truncsti8 : PatFrag<(ops node:$val, node:$base, node:$offset),
                            (pre_truncst node:$val, node:$base, node:$offset), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::i8;
}]>;
def pre_truncsti16 : PatFrag<(ops node:$val, node:$base, node:$offset),
                             (pre_truncst node:$val, node:$base, node:$offset), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::i16;
}]>;
def pre_truncsti32 : PatFrag<(ops node:$val, node:$base, node:$offset),
                             (pre_truncst node:$val, node:$base, node:$offset), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::i32;
}]>;
def pre_truncstf32 : PatFrag<(ops node:$val, node:$base, node:$offset),
                             (pre_truncst node:$val, node:$base, node:$offset), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::f32;
}]>;

def post_store : PatFrag<(ops node:$val, node:$ptr, node:$offset),
                         (istore node:$val, node:$ptr, node:$offset), [{
  ISD::MemIndexedMode AM = cast<StoreSDNode>(N)->getAddressingMode();
  return AM == ISD::POST_INC || AM == ISD::POST_DEC;
}]>;

def post_truncst : PatFrag<(ops node:$val, node:$base, node:$offset),
                           (itruncstore node:$val, node:$base, node:$offset), [{
  ISD::MemIndexedMode AM = cast<StoreSDNode>(N)->getAddressingMode();
  return AM == ISD::POST_INC || AM == ISD::POST_DEC;
}]>;
def post_truncsti1 : PatFrag<(ops node:$val, node:$base, node:$offset),
                             (post_truncst node:$val, node:$base, node:$offset), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::i1;
}]>;
def post_truncsti8 : PatFrag<(ops node:$val, node:$base, node:$offset),
                             (post_truncst node:$val, node:$base, node:$offset), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::i8;
}]>;
def post_truncsti16 : PatFrag<(ops node:$val, node:$base, node:$offset),
                              (post_truncst node:$val, node:$base, node:$offset), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::i16;
}]>;
def post_truncsti32 : PatFrag<(ops node:$val, node:$base, node:$offset),
                              (post_truncst node:$val, node:$base, node:$offset), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::i32;
}]>;
def post_truncstf32 : PatFrag<(ops node:$val, node:$base, node:$offset),
                              (post_truncst node:$val, node:$base, node:$offset), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::f32;
}]>;

// setcc convenience fragments.
def setoeq : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETOEQ)>;
def setogt : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETOGT)>;
def setoge : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETOGE)>;
def setolt : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETOLT)>;
def setole : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETOLE)>;
def setone : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETONE)>;
def seto   : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETO)>;
def setuo  : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETUO)>;
def setueq : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETUEQ)>;
def setugt : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETUGT)>;
def setuge : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETUGE)>;
def setult : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETULT)>;
def setule : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETULE)>;
def setune : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETUNE)>;
def seteq  : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETEQ)>;
def setgt  : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETGT)>;
def setge  : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETGE)>;
def setlt  : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETLT)>;
def setle  : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETLE)>;
def setne  : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETNE)>;

def atomic_cmp_swap_8 :
  PatFrag<(ops node:$ptr, node:$cmp, node:$swap),
          (atomic_cmp_swap node:$ptr, node:$cmp, node:$swap), [{
  return cast<AtomicSDNode>(N)->getMemoryVT() == MVT::i8;
}]>;
def atomic_cmp_swap_16 :
  PatFrag<(ops node:$ptr, node:$cmp, node:$swap),
          (atomic_cmp_swap node:$ptr, node:$cmp, node:$swap), [{
  return cast<AtomicSDNode>(N)->getMemoryVT() == MVT::i16;
}]>;
def atomic_cmp_swap_32 :
  PatFrag<(ops node:$ptr, node:$cmp, node:$swap),
          (atomic_cmp_swap node:$ptr, node:$cmp, node:$swap), [{
  return cast<AtomicSDNode>(N)->getMemoryVT() == MVT::i32;
}]>;
def atomic_cmp_swap_64 :
  PatFrag<(ops node:$ptr, node:$cmp, node:$swap),
          (atomic_cmp_swap node:$ptr, node:$cmp, node:$swap), [{
  return cast<AtomicSDNode>(N)->getMemoryVT() == MVT::i64;
}]>;

multiclass binary_atomic_op<SDNode atomic_op> {
  def _8 : PatFrag<(ops node:$ptr, node:$val),
                   (atomic_op node:$ptr, node:$val), [{
    return cast<AtomicSDNode>(N)->getMemoryVT() == MVT::i8;
  }]>;
  def _16 : PatFrag<(ops node:$ptr, node:$val),
                   (atomic_op node:$ptr, node:$val), [{
    return cast<AtomicSDNode>(N)->getMemoryVT() == MVT::i16;
  }]>;
  def _32 : PatFrag<(ops node:$ptr, node:$val),
                   (atomic_op node:$ptr, node:$val), [{
    return cast<AtomicSDNode>(N)->getMemoryVT() == MVT::i32;
  }]>;
  def _64 : PatFrag<(ops node:$ptr, node:$val),
                   (atomic_op node:$ptr, node:$val), [{
    return cast<AtomicSDNode>(N)->getMemoryVT() == MVT::i64;
  }]>;
}

defm atomic_load_add  : binary_atomic_op<atomic_load_add>;
defm atomic_swap      : binary_atomic_op<atomic_swap>;
defm atomic_load_sub  : binary_atomic_op<atomic_load_sub>;
defm atomic_load_and  : binary_atomic_op<atomic_load_and>;
defm atomic_load_or   : binary_atomic_op<atomic_load_or>;
defm atomic_load_xor  : binary_atomic_op<atomic_load_xor>;
defm atomic_load_nand : binary_atomic_op<atomic_load_nand>;
defm atomic_load_min  : binary_atomic_op<atomic_load_min>;
defm atomic_load_max  : binary_atomic_op<atomic_load_max>;
defm atomic_load_umin : binary_atomic_op<atomic_load_umin>;
defm atomic_load_umax : binary_atomic_op<atomic_load_umax>;
defm atomic_store     : binary_atomic_op<atomic_store>;

def atomic_load_8 :
  PatFrag<(ops node:$ptr),
          (atomic_load node:$ptr), [{
  return cast<AtomicSDNode>(N)->getMemoryVT() == MVT::i8;
}]>;
def atomic_load_16 :
  PatFrag<(ops node:$ptr),
          (atomic_load node:$ptr), [{
  return cast<AtomicSDNode>(N)->getMemoryVT() == MVT::i16;
}]>;
def atomic_load_32 :
  PatFrag<(ops node:$ptr),
          (atomic_load node:$ptr), [{
  return cast<AtomicSDNode>(N)->getMemoryVT() == MVT::i32;
}]>;
def atomic_load_64 :
  PatFrag<(ops node:$ptr),
          (atomic_load node:$ptr), [{
  return cast<AtomicSDNode>(N)->getMemoryVT() == MVT::i64;
}]>;

//===----------------------------------------------------------------------===//
// Selection DAG CONVERT_RNDSAT patterns

def cvtff : PatFrag<(ops node:$val, node:$dty, node:$sty, node:$rd, node:$sat),
    (cvt node:$val, node:$dty, node:$sty, node:$rd, node:$sat), [{
       return cast<CvtRndSatSDNode>(N)->getCvtCode() == ISD::CVT_FF;
    }]>;

def cvtss : PatFrag<(ops node:$val, node:$dty, node:$sty, node:$rd, node:$sat),
    (cvt node:$val, node:$dty, node:$sty, node:$rd, node:$sat), [{
       return cast<CvtRndSatSDNode>(N)->getCvtCode() == ISD::CVT_SS;
    }]>;

def cvtsu : PatFrag<(ops node:$val, node:$dty, node:$sty, node:$rd, node:$sat),
    (cvt node:$val, node:$dty, node:$sty, node:$rd, node:$sat), [{
       return cast<CvtRndSatSDNode>(N)->getCvtCode() == ISD::CVT_SU;
    }]>;

def cvtus : PatFrag<(ops node:$val, node:$dty, node:$sty, node:$rd, node:$sat),
    (cvt node:$val, node:$dty, node:$sty, node:$rd, node:$sat), [{
       return cast<CvtRndSatSDNode>(N)->getCvtCode() == ISD::CVT_US;
    }]>;

def cvtuu : PatFrag<(ops node:$val, node:$dty, node:$sty, node:$rd, node:$sat),
    (cvt node:$val, node:$dty, node:$sty, node:$rd, node:$sat), [{
       return cast<CvtRndSatSDNode>(N)->getCvtCode() == ISD::CVT_UU;
    }]>;

def cvtsf : PatFrag<(ops node:$val, node:$dty, node:$sty, node:$rd, node:$sat),
    (cvt node:$val, node:$dty, node:$sty, node:$rd, node:$sat), [{
       return cast<CvtRndSatSDNode>(N)->getCvtCode() == ISD::CVT_SF;
    }]>;

def cvtuf : PatFrag<(ops node:$val, node:$dty, node:$sty, node:$rd, node:$sat),
    (cvt node:$val, node:$dty, node:$sty, node:$rd, node:$sat), [{
       return cast<CvtRndSatSDNode>(N)->getCvtCode() == ISD::CVT_UF;
    }]>;

def cvtfs : PatFrag<(ops node:$val, node:$dty, node:$sty, node:$rd, node:$sat),
    (cvt node:$val, node:$dty, node:$sty, node:$rd, node:$sat), [{
       return cast<CvtRndSatSDNode>(N)->getCvtCode() == ISD::CVT_FS;
    }]>;

def cvtfu : PatFrag<(ops node:$val, node:$dty, node:$sty, node:$rd, node:$sat),
    (cvt node:$val, node:$dty, node:$sty, node:$rd, node:$sat), [{
       return cast<CvtRndSatSDNode>(N)->getCvtCode() == ISD::CVT_FU;
    }]>;

//===----------------------------------------------------------------------===//
// Selection DAG Pattern Support.
//
// Patterns are what are actually matched against by the target-flavored
// instruction selection DAG.  Instructions defined by the target implicitly
// define patterns in most cases, but patterns can also be explicitly added when
// an operation is defined by a sequence of instructions (e.g. loading a large
// immediate value on RISC targets that do not support immediates as large as
// their GPRs).
//

class Pattern<dag patternToMatch, list<dag> resultInstrs> {
  dag             PatternToMatch  = patternToMatch;
  list<dag>       ResultInstrs    = resultInstrs;
  list<Predicate> Predicates      = [];  // See class Instruction in Target.td.
  int             AddedComplexity = 0;   // See class Instruction in Target.td.
}

// Pat - A simple (but common) form of a pattern, which produces a simple result
// not needing a full list.
class Pat<dag pattern, dag result> : Pattern<pattern, [result]>;

//===----------------------------------------------------------------------===//
// Complex pattern definitions.
//

// Complex patterns, e.g. X86 addressing mode, requires pattern matching code
// in C++. NumOperands is the number of operands returned by the select function;
// SelectFunc is the name of the function used to pattern match the max. pattern;
// RootNodes are the list of possible root nodes of the sub-dags to match.
// e.g. X86 addressing mode - def addr : ComplexPattern<4, "SelectAddr", [add]>;
//
class ComplexPattern<ValueType ty, int numops, string fn,
                     list<SDNode> roots = [], list<SDNodeProperty> props = []> {
  ValueType Ty = ty;
  int NumOperands = numops;
  string SelectFunc = fn;
  list<SDNode> RootNodes = roots;
  list<SDNodeProperty> Properties = props;
}
