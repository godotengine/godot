//===-- LLParser.h - Parser Class -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the parser class for .ll files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_ASMPARSER_LLPARSER_H
#define LLVM_LIB_ASMPARSER_LLPARSER_H

#include "LLLexer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/ValueHandle.h"
#include <map>

namespace llvm {
  class Module;
  class OpaqueType;
  class Function;
  class Value;
  class BasicBlock;
  class Instruction;
  class Constant;
  class GlobalValue;
  class Comdat;
  class MDString;
  class MDNode;
  struct SlotMapping;
  class StructType;

  /// ValID - Represents a reference of a definition of some sort with no type.
  /// There are several cases where we have to parse the value but where the
  /// type can depend on later context.  This may either be a numeric reference
  /// or a symbolic (%var) reference.  This is just a discriminated union.
  struct ValID {
    enum {
      t_LocalID, t_GlobalID,      // ID in UIntVal.
      t_LocalName, t_GlobalName,  // Name in StrVal.
      t_APSInt, t_APFloat,        // Value in APSIntVal/APFloatVal.
      t_Null, t_Undef, t_Zero,    // No value.
      t_EmptyArray,               // No value:  []
      t_Constant,                 // Value in ConstantVal.
      t_InlineAsm,                // Value in StrVal/StrVal2/UIntVal.
      t_ConstantStruct,           // Value in ConstantStructElts.
      t_PackedConstantStruct      // Value in ConstantStructElts.
    } Kind;

    LLLexer::LocTy Loc;
    unsigned UIntVal;
    std::string StrVal, StrVal2;
    APSInt APSIntVal;
    APFloat APFloatVal;
    Constant *ConstantVal;
    Constant **ConstantStructElts;

    ValID() : Kind(t_LocalID), APFloatVal(0.0) {}
    ~ValID() {
      if (Kind == t_ConstantStruct || Kind == t_PackedConstantStruct)
        delete [] ConstantStructElts;
    }

    bool operator<(const ValID &RHS) const {
      if (Kind == t_LocalID || Kind == t_GlobalID)
        return UIntVal < RHS.UIntVal;
      assert((Kind == t_LocalName || Kind == t_GlobalName ||
              Kind == t_ConstantStruct || Kind == t_PackedConstantStruct) &&
             "Ordering not defined for this ValID kind yet");
      return StrVal < RHS.StrVal;
    }
  };

  class LLParser {
  public:
    typedef LLLexer::LocTy LocTy;
  private:
    LLVMContext &Context;
    LLLexer Lex;
    Module *M;
    SlotMapping *Slots;

    // Instruction metadata resolution.  Each instruction can have a list of
    // MDRef info associated with them.
    //
    // The simpler approach of just creating temporary MDNodes and then calling
    // RAUW on them when the definition is processed doesn't work because some
    // instruction metadata kinds, such as dbg, get stored in the IR in an
    // "optimized" format which doesn't participate in the normal value use
    // lists. This means that RAUW doesn't work, even on temporary MDNodes
    // which otherwise support RAUW. Instead, we defer resolving MDNode
    // references until the definitions have been processed.
    struct MDRef {
      SMLoc Loc;
      unsigned MDKind, MDSlot;
    };

    SmallVector<Instruction*, 64> InstsWithTBAATag;

    // Type resolution handling data structures.  The location is set when we
    // have processed a use of the type but not a definition yet.
    StringMap<std::pair<Type*, LocTy> > NamedTypes;
    std::map<unsigned, std::pair<Type*, LocTy> > NumberedTypes;

    std::map<unsigned, TrackingMDNodeRef> NumberedMetadata;
    std::map<unsigned, std::pair<TempMDTuple, LocTy>> ForwardRefMDNodes;

    // Global Value reference information.
    std::map<std::string, std::pair<GlobalValue*, LocTy> > ForwardRefVals;
    std::map<unsigned, std::pair<GlobalValue*, LocTy> > ForwardRefValIDs;
    std::vector<GlobalValue*> NumberedVals;

    // Comdat forward reference information.
    std::map<std::string, LocTy> ForwardRefComdats;

    // References to blockaddress.  The key is the function ValID, the value is
    // a list of references to blocks in that function.
    std::map<ValID, std::map<ValID, GlobalValue *>> ForwardRefBlockAddresses;
    class PerFunctionState;
    /// Reference to per-function state to allow basic blocks to be
    /// forward-referenced by blockaddress instructions within the same
    /// function.
    PerFunctionState *BlockAddressPFS;

    // Attribute builder reference information.
    std::map<Value*, std::vector<unsigned> > ForwardRefAttrGroups;
    std::map<unsigned, AttrBuilder> NumberedAttrBuilders;

  public:
    LLParser(StringRef F, SourceMgr &SM, SMDiagnostic &Err, Module *M,
             SlotMapping *Slots = nullptr)
        : Context(M->getContext()), Lex(F, SM, Err, M->getContext()), M(M),
          Slots(Slots), BlockAddressPFS(nullptr) {}
    bool Run();

    LLVMContext &getContext() { return Context; }

  private:

    bool Error(LocTy L, const Twine &Msg) const {
      return Lex.Error(L, Msg);
    }
    bool TokError(const Twine &Msg) const {
      return Error(Lex.getLoc(), Msg);
    }

    /// GetGlobalVal - Get a value with the specified name or ID, creating a
    /// forward reference record if needed.  This can return null if the value
    /// exists but does not have the right type.
    GlobalValue *GetGlobalVal(const std::string &N, Type *Ty, LocTy Loc);
    GlobalValue *GetGlobalVal(unsigned ID, Type *Ty, LocTy Loc);

    /// Get a Comdat with the specified name, creating a forward reference
    /// record if needed.
    Comdat *getComdat(const std::string &N, LocTy Loc);

    // Helper Routines.
    bool ParseToken(lltok::Kind T, const char *ErrMsg);
    bool EatIfPresent(lltok::Kind T) {
      if (Lex.getKind() != T) return false;
      Lex.Lex();
      return true;
    }

    FastMathFlags EatFastMathFlagsIfPresent() {
      FastMathFlags FMF;
      while (true)
        switch (Lex.getKind()) {
        case lltok::kw_fast: FMF.setUnsafeAlgebra();   Lex.Lex(); continue;
        case lltok::kw_nnan: FMF.setNoNaNs();          Lex.Lex(); continue;
        case lltok::kw_ninf: FMF.setNoInfs();          Lex.Lex(); continue;
        case lltok::kw_nsz:  FMF.setNoSignedZeros();   Lex.Lex(); continue;
        case lltok::kw_arcp: FMF.setAllowReciprocal(); Lex.Lex(); continue;
        default: return FMF;
        }
      return FMF;
    }

    bool ParseOptionalToken(lltok::Kind T, bool &Present,
                            LocTy *Loc = nullptr) {
      if (Lex.getKind() != T) {
        Present = false;
      } else {
        if (Loc)
          *Loc = Lex.getLoc();
        Lex.Lex();
        Present = true;
      }
      return false;
    }
    bool ParseStringConstant(std::string &Result);
    bool ParseUInt32(unsigned &Val);
    bool ParseUInt32(unsigned &Val, LocTy &Loc) {
      Loc = Lex.getLoc();
      return ParseUInt32(Val);
    }
    bool ParseUInt64(uint64_t &Val);
    bool ParseUInt64(uint64_t &Val, LocTy &Loc) {
      Loc = Lex.getLoc();
      return ParseUInt64(Val);
    }

    bool ParseTLSModel(GlobalVariable::ThreadLocalMode &TLM);
    bool ParseOptionalThreadLocal(GlobalVariable::ThreadLocalMode &TLM);
    bool parseOptionalUnnamedAddr(bool &UnnamedAddr) {
      return ParseOptionalToken(lltok::kw_unnamed_addr, UnnamedAddr);
    }
    bool ParseOptionalAddrSpace(unsigned &AddrSpace);
    bool ParseOptionalParamAttrs(AttrBuilder &B);
    bool ParseOptionalReturnAttrs(AttrBuilder &B);
    bool ParseOptionalLinkage(unsigned &Linkage, bool &HasLinkage);
    bool ParseOptionalLinkage(unsigned &Linkage) {
      bool HasLinkage; return ParseOptionalLinkage(Linkage, HasLinkage);
    }
    bool ParseOptionalVisibility(unsigned &Visibility);
    bool ParseOptionalDLLStorageClass(unsigned &DLLStorageClass);
    bool ParseOptionalCallingConv(unsigned &CC);
    bool ParseOptionalAlignment(unsigned &Alignment);
    bool ParseOptionalDerefAttrBytes(lltok::Kind AttrKind, uint64_t &Bytes);
    bool ParseScopeAndOrdering(bool isAtomic, SynchronizationScope &Scope,
                               AtomicOrdering &Ordering);
    bool ParseOrdering(AtomicOrdering &Ordering);
    bool ParseOptionalStackAlignment(unsigned &Alignment);
    bool ParseOptionalCommaAlign(unsigned &Alignment, bool &AteExtraComma);
    bool ParseOptionalCommaInAlloca(bool &IsInAlloca);
    bool ParseIndexList(SmallVectorImpl<unsigned> &Indices,bool &AteExtraComma);
    bool ParseIndexList(SmallVectorImpl<unsigned> &Indices) {
      bool AteExtraComma;
      if (ParseIndexList(Indices, AteExtraComma)) return true;
      if (AteExtraComma)
        return TokError("expected index");
      return false;
    }

    // Top-Level Entities
    bool ParseTopLevelEntities();
    bool ValidateEndOfModule();
    bool ParseTargetDefinition();
    bool ParseModuleAsm();
    bool ParseDepLibs();        // FIXME: Remove in 4.0.
    bool ParseUnnamedType();
    bool ParseNamedType();
    bool ParseDeclare();
    bool ParseDefine();

    bool ParseGlobalType(bool &IsConstant);
    bool ParseUnnamedGlobal();
    bool ParseNamedGlobal();
    bool ParseGlobal(const std::string &Name, LocTy Loc, unsigned Linkage,
                     bool HasLinkage, unsigned Visibility,
                     unsigned DLLStorageClass,
                     GlobalVariable::ThreadLocalMode TLM, bool UnnamedAddr);
    bool ParseAlias(const std::string &Name, LocTy Loc, unsigned Linkage,
                    unsigned Visibility, unsigned DLLStorageClass,
                    GlobalVariable::ThreadLocalMode TLM, bool UnnamedAddr);
    bool parseComdat();
    bool ParseStandaloneMetadata();
    bool ParseNamedMetadata();
    bool ParseMDString(MDString *&Result);
    bool ParseMDNodeID(MDNode *&Result);
    bool ParseUnnamedAttrGrp();
    bool ParseFnAttributeValuePairs(AttrBuilder &B,
                                    std::vector<unsigned> &FwdRefAttrGrps,
                                    bool inAttrGrp, LocTy &BuiltinLoc);

    // Type Parsing.
    bool ParseType(Type *&Result, const Twine &Msg, bool AllowVoid = false);
    bool ParseType(Type *&Result, bool AllowVoid = false) {
      return ParseType(Result, "expected type", AllowVoid);
    }
    bool ParseType(Type *&Result, const Twine &Msg, LocTy &Loc,
                   bool AllowVoid = false) {
      Loc = Lex.getLoc();
      return ParseType(Result, Msg, AllowVoid);
    }
    bool ParseType(Type *&Result, LocTy &Loc, bool AllowVoid = false) {
      Loc = Lex.getLoc();
      return ParseType(Result, AllowVoid);
    }
    bool ParseAnonStructType(Type *&Result, bool Packed);
    bool ParseStructBody(SmallVectorImpl<Type*> &Body);
    bool ParseStructDefinition(SMLoc TypeLoc, StringRef Name,
                               std::pair<Type*, LocTy> &Entry,
                               Type *&ResultTy);

    bool ParseArrayVectorType(Type *&Result, bool isVector);
    bool ParseFunctionType(Type *&Result);

    // Function Semantic Analysis.
    class PerFunctionState {
      LLParser &P;
      Function &F;
      std::map<std::string, std::pair<Value*, LocTy> > ForwardRefVals;
      std::map<unsigned, std::pair<Value*, LocTy> > ForwardRefValIDs;
      std::vector<Value*> NumberedVals;

      /// FunctionNumber - If this is an unnamed function, this is the slot
      /// number of it, otherwise it is -1.
      int FunctionNumber;
    public:
      PerFunctionState(LLParser &p, Function &f, int FunctionNumber);
      ~PerFunctionState();

      Function &getFunction() const { return F; }

      bool FinishFunction();

      /// GetVal - Get a value with the specified name or ID, creating a
      /// forward reference record if needed.  This can return null if the value
      /// exists but does not have the right type.
      Value *GetVal(const std::string &Name, Type *Ty, LocTy Loc);
      Value *GetVal(unsigned ID, Type *Ty, LocTy Loc);

      /// SetInstName - After an instruction is parsed and inserted into its
      /// basic block, this installs its name.
      bool SetInstName(int NameID, const std::string &NameStr, LocTy NameLoc,
                       Instruction *Inst);

      /// GetBB - Get a basic block with the specified name or ID, creating a
      /// forward reference record if needed.  This can return null if the value
      /// is not a BasicBlock.
      BasicBlock *GetBB(const std::string &Name, LocTy Loc);
      BasicBlock *GetBB(unsigned ID, LocTy Loc);

      /// DefineBB - Define the specified basic block, which is either named or
      /// unnamed.  If there is an error, this returns null otherwise it returns
      /// the block being defined.
      BasicBlock *DefineBB(const std::string &Name, LocTy Loc);

      bool resolveForwardRefBlockAddresses();
    };

    bool ConvertValIDToValue(Type *Ty, ValID &ID, Value *&V,
                             PerFunctionState *PFS);

    bool ParseValue(Type *Ty, Value *&V, PerFunctionState *PFS);
    bool ParseValue(Type *Ty, Value *&V, PerFunctionState &PFS) {
      return ParseValue(Ty, V, &PFS);
    }
    bool ParseValue(Type *Ty, Value *&V, LocTy &Loc,
                    PerFunctionState &PFS) {
      Loc = Lex.getLoc();
      return ParseValue(Ty, V, &PFS);
    }

    bool ParseTypeAndValue(Value *&V, PerFunctionState *PFS);
    bool ParseTypeAndValue(Value *&V, PerFunctionState &PFS) {
      return ParseTypeAndValue(V, &PFS);
    }
    bool ParseTypeAndValue(Value *&V, LocTy &Loc, PerFunctionState &PFS) {
      Loc = Lex.getLoc();
      return ParseTypeAndValue(V, PFS);
    }
    bool ParseTypeAndBasicBlock(BasicBlock *&BB, LocTy &Loc,
                                PerFunctionState &PFS);
    bool ParseTypeAndBasicBlock(BasicBlock *&BB, PerFunctionState &PFS) {
      LocTy Loc;
      return ParseTypeAndBasicBlock(BB, Loc, PFS);
    }


    struct ParamInfo {
      LocTy Loc;
      Value *V;
      AttributeSet Attrs;
      ParamInfo(LocTy loc, Value *v, AttributeSet attrs)
        : Loc(loc), V(v), Attrs(attrs) {}
    };
    bool ParseParameterList(SmallVectorImpl<ParamInfo> &ArgList,
                            PerFunctionState &PFS,
                            bool IsMustTailCall = false,
                            bool InVarArgsFunc = false);

    // Constant Parsing.
    bool ParseValID(ValID &ID, PerFunctionState *PFS = nullptr);
    bool ParseGlobalValue(Type *Ty, Constant *&V);
    bool ParseGlobalTypeAndValue(Constant *&V);
    bool ParseGlobalValueVector(SmallVectorImpl<Constant *> &Elts);
    bool parseOptionalComdat(StringRef GlobalName, Comdat *&C);
    bool ParseMetadataAsValue(Value *&V, PerFunctionState &PFS);
    bool ParseValueAsMetadata(Metadata *&MD, const Twine &TypeMsg,
                              PerFunctionState *PFS);
    bool ParseMetadata(Metadata *&MD, PerFunctionState *PFS);
    bool ParseMDTuple(MDNode *&MD, bool IsDistinct = false);
    bool ParseMDNode(MDNode *&MD);
    bool ParseMDNodeTail(MDNode *&MD);
    bool ParseMDNodeVector(SmallVectorImpl<Metadata *> &MDs);
    bool ParseMetadataAttachment(unsigned &Kind, MDNode *&MD);
    bool ParseInstructionMetadata(Instruction &Inst);
    bool ParseOptionalFunctionMetadata(Function &F);

    template <class FieldTy>
    bool ParseMDField(LocTy Loc, StringRef Name, FieldTy &Result);
    template <class FieldTy> bool ParseMDField(StringRef Name, FieldTy &Result);
    template <class ParserTy>
    bool ParseMDFieldsImplBody(ParserTy parseField);
    template <class ParserTy>
    bool ParseMDFieldsImpl(ParserTy parseField, LocTy &ClosingLoc);
    bool ParseSpecializedMDNode(MDNode *&N, bool IsDistinct = false);

#define HANDLE_SPECIALIZED_MDNODE_LEAF(CLASS)                                  \
  bool Parse##CLASS(MDNode *&Result, bool IsDistinct);
#include "llvm/IR/Metadata.def"

    // Function Parsing.
    struct ArgInfo {
      LocTy Loc;
      Type *Ty;
      AttributeSet Attrs;
      std::string Name;
      ArgInfo(LocTy L, Type *ty, AttributeSet Attr, const std::string &N)
        : Loc(L), Ty(ty), Attrs(Attr), Name(N) {}
    };
    bool ParseArgumentList(SmallVectorImpl<ArgInfo> &ArgList, bool &isVarArg);
    bool ParseFunctionHeader(Function *&Fn, bool isDefine);
    bool ParseFunctionBody(Function &Fn);
    bool ParseBasicBlock(PerFunctionState &PFS);

    enum TailCallType { TCT_None, TCT_Tail, TCT_MustTail };

    // Instruction Parsing.  Each instruction parsing routine can return with a
    // normal result, an error result, or return having eaten an extra comma.
    enum InstResult { InstNormal = 0, InstError = 1, InstExtraComma = 2 };
    int ParseInstruction(Instruction *&Inst, BasicBlock *BB,
                         PerFunctionState &PFS);
    bool ParseCmpPredicate(unsigned &Pred, unsigned Opc);

    bool ParseRet(Instruction *&Inst, BasicBlock *BB, PerFunctionState &PFS);
    bool ParseBr(Instruction *&Inst, PerFunctionState &PFS);
    bool ParseSwitch(Instruction *&Inst, PerFunctionState &PFS);
    bool ParseIndirectBr(Instruction *&Inst, PerFunctionState &PFS);
    bool ParseInvoke(Instruction *&Inst, PerFunctionState &PFS);
    bool ParseResume(Instruction *&Inst, PerFunctionState &PFS);

    bool ParseArithmetic(Instruction *&I, PerFunctionState &PFS, unsigned Opc,
                         unsigned OperandType);
    bool ParseLogical(Instruction *&I, PerFunctionState &PFS, unsigned Opc);
    bool ParseCompare(Instruction *&I, PerFunctionState &PFS, unsigned Opc);
    bool ParseCast(Instruction *&I, PerFunctionState &PFS, unsigned Opc);
    bool ParseSelect(Instruction *&I, PerFunctionState &PFS);
    bool ParseVA_Arg(Instruction *&I, PerFunctionState &PFS);
    bool ParseExtractElement(Instruction *&I, PerFunctionState &PFS);
    bool ParseInsertElement(Instruction *&I, PerFunctionState &PFS);
    bool ParseShuffleVector(Instruction *&I, PerFunctionState &PFS);
    int ParsePHI(Instruction *&I, PerFunctionState &PFS);
    bool ParseLandingPad(Instruction *&I, PerFunctionState &PFS);
    bool ParseCall(Instruction *&I, PerFunctionState &PFS,
                   CallInst::TailCallKind IsTail);
    int ParseAlloc(Instruction *&I, PerFunctionState &PFS);
    int ParseLoad(Instruction *&I, PerFunctionState &PFS);
    int ParseStore(Instruction *&I, PerFunctionState &PFS);
    int ParseCmpXchg(Instruction *&I, PerFunctionState &PFS);
    int ParseAtomicRMW(Instruction *&I, PerFunctionState &PFS);
    int ParseFence(Instruction *&I, PerFunctionState &PFS);
    int ParseGetElementPtr(Instruction *&I, PerFunctionState &PFS);
    int ParseExtractValue(Instruction *&I, PerFunctionState &PFS);
    int ParseInsertValue(Instruction *&I, PerFunctionState &PFS);

    // Use-list order directives.
    bool ParseUseListOrder(PerFunctionState *PFS = nullptr);
    bool ParseUseListOrderBB();
    bool ParseUseListOrderIndexes(SmallVectorImpl<unsigned> &Indexes);
    bool sortUseListOrder(Value *V, ArrayRef<unsigned> Indexes, SMLoc Loc);
  };
} // End llvm namespace

#endif
