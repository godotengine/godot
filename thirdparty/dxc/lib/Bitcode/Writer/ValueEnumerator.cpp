//===-- ValueEnumerator.cpp - Number values and types for bitcode writer --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ValueEnumerator class.
//
//===----------------------------------------------------------------------===//

#include "ValueEnumerator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/UseListOrder.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace llvm;

namespace {
struct OrderMap {
  DenseMap<const Value *, std::pair<unsigned, bool>> IDs;
  unsigned LastGlobalConstantID;
  unsigned LastGlobalValueID;

  OrderMap() : LastGlobalConstantID(0), LastGlobalValueID(0) {}

  bool isGlobalConstant(unsigned ID) const {
    return ID <= LastGlobalConstantID;
  }
  bool isGlobalValue(unsigned ID) const {
    return ID <= LastGlobalValueID && !isGlobalConstant(ID);
  }

  unsigned size() const { return IDs.size(); }
  std::pair<unsigned, bool> &operator[](const Value *V) { return IDs[V]; }
  std::pair<unsigned, bool> lookup(const Value *V) const {
    return IDs.lookup(V);
  }
  void index(const Value *V) {
    // Explicitly sequence get-size and insert-value operations to avoid UB.
    unsigned ID = IDs.size() + 1;
    IDs[V].first = ID;
  }
};
}

static void orderValue(const Value *V, OrderMap &OM) {
  if (OM.lookup(V).first)
    return;

  if (const Constant *C = dyn_cast<Constant>(V))
    if (C->getNumOperands() && !isa<GlobalValue>(C))
      for (const Value *Op : C->operands())
        if (!isa<BasicBlock>(Op) && !isa<GlobalValue>(Op))
          orderValue(Op, OM);

  // Note: we cannot cache this lookup above, since inserting into the map
  // changes the map's size, and thus affects the other IDs.
  OM.index(V);
}

static OrderMap orderModule(const Module &M) {
  // This needs to match the order used by ValueEnumerator::ValueEnumerator()
  // and ValueEnumerator::incorporateFunction().
  OrderMap OM;

  // In the reader, initializers of GlobalValues are set *after* all the
  // globals have been read.  Rather than awkwardly modeling this behaviour
  // directly in predictValueUseListOrderImpl(), just assign IDs to
  // initializers of GlobalValues before GlobalValues themselves to model this
  // implicitly.
  for (const GlobalVariable &G : M.globals())
    if (G.hasInitializer())
      if (!isa<GlobalValue>(G.getInitializer()))
        orderValue(G.getInitializer(), OM);
  for (const GlobalAlias &A : M.aliases())
    if (!isa<GlobalValue>(A.getAliasee()))
      orderValue(A.getAliasee(), OM);
  for (const Function &F : M) {
    if (F.hasPrefixData())
      if (!isa<GlobalValue>(F.getPrefixData()))
        orderValue(F.getPrefixData(), OM);
    if (F.hasPrologueData())
      if (!isa<GlobalValue>(F.getPrologueData()))
        orderValue(F.getPrologueData(), OM);
    if (F.hasPersonalityFn())
      if (!isa<GlobalValue>(F.getPersonalityFn()))
        orderValue(F.getPersonalityFn(), OM);
  }
  OM.LastGlobalConstantID = OM.size();

  // Initializers of GlobalValues are processed in
  // BitcodeReader::ResolveGlobalAndAliasInits().  Match the order there rather
  // than ValueEnumerator, and match the code in predictValueUseListOrderImpl()
  // by giving IDs in reverse order.
  //
  // Since GlobalValues never reference each other directly (just through
  // initializers), their relative IDs only matter for determining order of
  // uses in their initializers.
  for (const Function &F : M)
    orderValue(&F, OM);
  for (const GlobalAlias &A : M.aliases())
    orderValue(&A, OM);
  for (const GlobalVariable &G : M.globals())
    orderValue(&G, OM);
  OM.LastGlobalValueID = OM.size();

  for (const Function &F : M) {
    if (F.isDeclaration())
      continue;
    // Here we need to match the union of ValueEnumerator::incorporateFunction()
    // and WriteFunction().  Basic blocks are implicitly declared before
    // anything else (by declaring their size).
    for (const BasicBlock &BB : F)
      orderValue(&BB, OM);
    for (const Argument &A : F.args())
      orderValue(&A, OM);
    for (const BasicBlock &BB : F)
      for (const Instruction &I : BB)
        for (const Value *Op : I.operands())
          if ((isa<Constant>(*Op) && !isa<GlobalValue>(*Op)) ||
              isa<InlineAsm>(*Op))
            orderValue(Op, OM);
    for (const BasicBlock &BB : F)
      for (const Instruction &I : BB)
        orderValue(&I, OM);
  }
  return OM;
}

static void predictValueUseListOrderImpl(const Value *V, const Function *F,
                                         unsigned ID, const OrderMap &OM,
                                         UseListOrderStack &Stack) {
  // Predict use-list order for this one.
  typedef std::pair<const Use *, unsigned> Entry;
  SmallVector<Entry, 64> List;
  for (const Use &U : V->uses())
    // Check if this user will be serialized.
    if (OM.lookup(U.getUser()).first)
      List.push_back(std::make_pair(&U, List.size()));

  if (List.size() < 2)
    // We may have lost some users.
    return;

  bool IsGlobalValue = OM.isGlobalValue(ID);
  std::sort(List.begin(), List.end(), [&](const Entry &L, const Entry &R) {
    const Use *LU = L.first;
    const Use *RU = R.first;
    if (LU == RU)
      return false;

    auto LID = OM.lookup(LU->getUser()).first;
    auto RID = OM.lookup(RU->getUser()).first;

    // Global values are processed in reverse order.
    //
    // Moreover, initializers of GlobalValues are set *after* all the globals
    // have been read (despite having earlier IDs).  Rather than awkwardly
    // modeling this behaviour here, orderModule() has assigned IDs to
    // initializers of GlobalValues before GlobalValues themselves.
    if (OM.isGlobalValue(LID) && OM.isGlobalValue(RID))
      return LID < RID;

    // If ID is 4, then expect: 7 6 5 1 2 3.
    if (LID < RID) {
      if (RID <= ID)
        if (!IsGlobalValue) // GlobalValue uses don't get reversed.
          return true;
      return false;
    }
    if (RID < LID) {
      if (LID <= ID)
        if (!IsGlobalValue) // GlobalValue uses don't get reversed.
          return false;
      return true;
    }

    // LID and RID are equal, so we have different operands of the same user.
    // Assume operands are added in order for all instructions.
    if (LID <= ID)
      if (!IsGlobalValue) // GlobalValue uses don't get reversed.
        return LU->getOperandNo() < RU->getOperandNo();
    return LU->getOperandNo() > RU->getOperandNo();
  });

  if (std::is_sorted(
          List.begin(), List.end(),
          [](const Entry &L, const Entry &R) { return L.second < R.second; }))
    // Order is already correct.
    return;

  // Store the shuffle.
  Stack.emplace_back(V, F, List.size());
  assert(List.size() == Stack.back().Shuffle.size() && "Wrong size");
  for (size_t I = 0, E = List.size(); I != E; ++I)
    Stack.back().Shuffle[I] = List[I].second;
}

static void predictValueUseListOrder(const Value *V, const Function *F,
                                     OrderMap &OM, UseListOrderStack &Stack) {
  auto &IDPair = OM[V];
  assert(IDPair.first && "Unmapped value");
  if (IDPair.second)
    // Already predicted.
    return;

  // Do the actual prediction.
  IDPair.second = true;
  if (!V->use_empty() && std::next(V->use_begin()) != V->use_end())
    predictValueUseListOrderImpl(V, F, IDPair.first, OM, Stack);

  // Recursive descent into constants.
  if (const Constant *C = dyn_cast<Constant>(V))
    if (C->getNumOperands()) // Visit GlobalValues.
      for (const Value *Op : C->operands())
        if (isa<Constant>(Op)) // Visit GlobalValues.
          predictValueUseListOrder(Op, F, OM, Stack);
}

static UseListOrderStack predictUseListOrder(const Module &M) {
  OrderMap OM = orderModule(M);

  // Use-list orders need to be serialized after all the users have been added
  // to a value, or else the shuffles will be incomplete.  Store them per
  // function in a stack.
  //
  // Aside from function order, the order of values doesn't matter much here.
  UseListOrderStack Stack;

  // We want to visit the functions backward now so we can list function-local
  // constants in the last Function they're used in.  Module-level constants
  // have already been visited above.
  for (auto I = M.rbegin(), E = M.rend(); I != E; ++I) {
    const Function &F = *I;
    if (F.isDeclaration())
      continue;
    for (const BasicBlock &BB : F)
      predictValueUseListOrder(&BB, &F, OM, Stack);
    for (const Argument &A : F.args())
      predictValueUseListOrder(&A, &F, OM, Stack);
    for (const BasicBlock &BB : F)
      for (const Instruction &I : BB)
        for (const Value *Op : I.operands())
          if (isa<Constant>(*Op) || isa<InlineAsm>(*Op)) // Visit GlobalValues.
            predictValueUseListOrder(Op, &F, OM, Stack);
    for (const BasicBlock &BB : F)
      for (const Instruction &I : BB)
        predictValueUseListOrder(&I, &F, OM, Stack);
  }

  // Visit globals last, since the module-level use-list block will be seen
  // before the function bodies are processed.
  for (const GlobalVariable &G : M.globals())
    predictValueUseListOrder(&G, nullptr, OM, Stack);
  for (const Function &F : M)
    predictValueUseListOrder(&F, nullptr, OM, Stack);
  for (const GlobalAlias &A : M.aliases())
    predictValueUseListOrder(&A, nullptr, OM, Stack);
  for (const GlobalVariable &G : M.globals())
    if (G.hasInitializer())
      predictValueUseListOrder(G.getInitializer(), nullptr, OM, Stack);
  for (const GlobalAlias &A : M.aliases())
    predictValueUseListOrder(A.getAliasee(), nullptr, OM, Stack);
  for (const Function &F : M) {
    if (F.hasPrefixData())
      predictValueUseListOrder(F.getPrefixData(), nullptr, OM, Stack);
    if (F.hasPrologueData())
      predictValueUseListOrder(F.getPrologueData(), nullptr, OM, Stack);
    if (F.hasPersonalityFn())
      predictValueUseListOrder(F.getPersonalityFn(), nullptr, OM, Stack);
  }

  return Stack;
}

static bool isIntOrIntVectorValue(const std::pair<const Value*, unsigned> &V) {
  return V.first->getType()->isIntOrIntVectorTy();
}

ValueEnumerator::ValueEnumerator(const Module &M,
                                 bool ShouldPreserveUseListOrder)
    : HasMDString(false), HasDILocation(false), HasGenericDINode(false),
      ShouldPreserveUseListOrder(ShouldPreserveUseListOrder) {
  if (ShouldPreserveUseListOrder)
    UseListOrders = predictUseListOrder(M);

  // Enumerate the global variables.
  for (const GlobalVariable &GV : M.globals())
    EnumerateValue(&GV);

  // Enumerate the functions.
  for (const Function & F : M) {
    EnumerateValue(&F);
    EnumerateAttributes(F.getAttributes());
  }

  // Enumerate the aliases.
  for (const GlobalAlias &GA : M.aliases())
    EnumerateValue(&GA);

  // Remember what is the cutoff between globalvalue's and other constants.
  unsigned FirstConstant = Values.size();

  // Enumerate the global variable initializers.
  for (const GlobalVariable &GV : M.globals())
    if (GV.hasInitializer())
      EnumerateValue(GV.getInitializer());

  // Enumerate the aliasees.
  for (const GlobalAlias &GA : M.aliases())
    EnumerateValue(GA.getAliasee());

  // Enumerate the prefix data constants.
  for (const Function &F : M)
    if (F.hasPrefixData())
      EnumerateValue(F.getPrefixData());

  // Enumerate the prologue data constants.
  for (const Function &F : M)
    if (F.hasPrologueData())
      EnumerateValue(F.getPrologueData());

  // Enumerate the personality functions.
  for (Module::const_iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (I->hasPersonalityFn())
      EnumerateValue(I->getPersonalityFn());

  // Enumerate the metadata type.
  //
  // TODO: Move this to ValueEnumerator::EnumerateOperandType() once bitcode
  // only encodes the metadata type when it's used as a value.
  EnumerateType(Type::getMetadataTy(M.getContext()));

  // Insert constants and metadata that are named at module level into the slot
  // pool so that the module symbol table can refer to them...
  EnumerateValueSymbolTable(M.getValueSymbolTable());
  EnumerateNamedMetadata(M);

  SmallVector<std::pair<unsigned, MDNode *>, 8> MDs;

  // Enumerate types used by function bodies and argument lists.
  for (const Function &F : M) {
    for (const Argument &A : F.args())
      EnumerateType(A.getType());

    // Enumerate metadata attached to this function.
    F.getAllMetadata(MDs);
    for (const auto &I : MDs)
      EnumerateMetadata(I.second);

    for (const BasicBlock &BB : F)
      for (const Instruction &I : BB) {
        for (const Use &Op : I.operands()) {
          auto *MD = dyn_cast<MetadataAsValue>(&Op);
          if (!MD) {
            EnumerateOperandType(Op);
            continue;
          }

          // Local metadata is enumerated during function-incorporation.
          if (isa<LocalAsMetadata>(MD->getMetadata()))
            continue;

          EnumerateMetadata(MD->getMetadata());
        }
        EnumerateType(I.getType());
        if (const CallInst *CI = dyn_cast<CallInst>(&I))
          EnumerateAttributes(CI->getAttributes());
        else if (const InvokeInst *II = dyn_cast<InvokeInst>(&I))
          EnumerateAttributes(II->getAttributes());

        // Enumerate metadata attached with this instruction.
        MDs.clear();
        I.getAllMetadataOtherThanDebugLoc(MDs);
        for (unsigned i = 0, e = MDs.size(); i != e; ++i)
          EnumerateMetadata(MDs[i].second);

        // Don't enumerate the location directly -- it has a special record
        // type -- but enumerate its operands.
        if (DILocation *L = I.getDebugLoc())
          EnumerateMDNodeOperands(L);
      }
  }

  // Optimize constant ordering.
  OptimizeConstants(FirstConstant, Values.size());
}

unsigned ValueEnumerator::getInstructionID(const Instruction *Inst) const {
  InstructionMapType::const_iterator I = InstructionMap.find(Inst);
  assert(I != InstructionMap.end() && "Instruction is not mapped!");
  return I->second;
}

unsigned ValueEnumerator::getComdatID(const Comdat *C) const {
  unsigned ComdatID = Comdats.idFor(C);
  assert(ComdatID && "Comdat not found!");
  return ComdatID;
}

void ValueEnumerator::setInstructionID(const Instruction *I) {
  InstructionMap[I] = InstructionCount++;
}

unsigned ValueEnumerator::getValueID(const Value *V) const {
  if (auto *MD = dyn_cast<MetadataAsValue>(V))
    return getMetadataID(MD->getMetadata());

  ValueMapType::const_iterator I = ValueMap.find(V);
  assert(I != ValueMap.end() && "Value not in slotcalculator!");
  return I->second-1;
}

void ValueEnumerator::dump() const {
  print(dbgs(), ValueMap, "Default");
  dbgs() << '\n';
  print(dbgs(), MDValueMap, "MetaData");
  dbgs() << '\n';
}

void ValueEnumerator::print(raw_ostream &OS, const ValueMapType &Map,
                            const char *Name) const {

  OS << "Map Name: " << Name << "\n";
  OS << "Size: " << Map.size() << "\n";
  for (ValueMapType::const_iterator I = Map.begin(),
         E = Map.end(); I != E; ++I) {

    const Value *V = I->first;
    if (V->hasName())
      OS << "Value: " << V->getName();
    else
      OS << "Value: [null]\n";
    V->dump();

    OS << " Uses(" << std::distance(V->use_begin(),V->use_end()) << "):";
    for (const Use &U : V->uses()) {
      if (&U != &*V->use_begin())
        OS << ",";
      if(U->hasName())
        OS << " " << U->getName();
      else
        OS << " [null]";

    }
    OS <<  "\n\n";
  }
}

void ValueEnumerator::print(raw_ostream &OS, const MetadataMapType &Map,
                            const char *Name) const {

  OS << "Map Name: " << Name << "\n";
  OS << "Size: " << Map.size() << "\n";
  for (auto I = Map.begin(), E = Map.end(); I != E; ++I) {
    const Metadata *MD = I->first;
    OS << "Metadata: slot = " << I->second << "\n";
    MD->print(OS);
  }
}

/// OptimizeConstants - Reorder constant pool for denser encoding.
void ValueEnumerator::OptimizeConstants(unsigned CstStart, unsigned CstEnd) {
  if (CstStart == CstEnd || CstStart+1 == CstEnd) return;

  if (ShouldPreserveUseListOrder)
    // Optimizing constants makes the use-list order difficult to predict.
    // Disable it for now when trying to preserve the order.
    return;

  std::stable_sort(Values.begin() + CstStart, Values.begin() + CstEnd,
                   [this](const std::pair<const Value *, unsigned> &LHS,
                          const std::pair<const Value *, unsigned> &RHS) {
    // Sort by plane.
    if (LHS.first->getType() != RHS.first->getType())
      return getTypeID(LHS.first->getType()) < getTypeID(RHS.first->getType());
    // Then by frequency.
    return LHS.second > RHS.second;
  });

  // Ensure that integer and vector of integer constants are at the start of the
  // constant pool.  This is important so that GEP structure indices come before
  // gep constant exprs.
  std::partition(Values.begin()+CstStart, Values.begin()+CstEnd,
                 isIntOrIntVectorValue);

  // Rebuild the modified portion of ValueMap.
  for (; CstStart != CstEnd; ++CstStart)
    ValueMap[Values[CstStart].first] = CstStart+1;
}


/// EnumerateValueSymbolTable - Insert all of the values in the specified symbol
/// table into the values table.
void ValueEnumerator::EnumerateValueSymbolTable(const ValueSymbolTable &VST) {
  for (ValueSymbolTable::const_iterator VI = VST.begin(), VE = VST.end();
       VI != VE; ++VI)
    EnumerateValue(VI->getValue());
}

/// Insert all of the values referenced by named metadata in the specified
/// module.
void ValueEnumerator::EnumerateNamedMetadata(const Module &M) {
  for (Module::const_named_metadata_iterator I = M.named_metadata_begin(),
                                             E = M.named_metadata_end();
       I != E; ++I)
    EnumerateNamedMDNode(I);
}

void ValueEnumerator::EnumerateNamedMDNode(const NamedMDNode *MD) {
  for (unsigned i = 0, e = MD->getNumOperands(); i != e; ++i)
    EnumerateMetadata(MD->getOperand(i));
}

/// EnumerateMDNodeOperands - Enumerate all non-function-local values
/// and types referenced by the given MDNode.
void ValueEnumerator::EnumerateMDNodeOperands(const MDNode *N) {
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    Metadata *MD = N->getOperand(i);
    if (!MD)
      continue;
    assert(!isa<LocalAsMetadata>(MD) && "MDNodes cannot be function-local");
    EnumerateMetadata(MD);
  }
}

void ValueEnumerator::EnumerateMetadata(const Metadata *MD) {
  assert(
      (isa<MDNode>(MD) || isa<MDString>(MD) || isa<ConstantAsMetadata>(MD)) &&
      "Invalid metadata kind");

  // Insert a dummy ID to block the co-recursive call to
  // EnumerateMDNodeOperands() from re-visiting MD in a cyclic graph.
  //
  // Return early if there's already an ID.
  if (!MDValueMap.insert(std::make_pair(MD, 0)).second)
    return;

  // Visit operands first to minimize RAUW.
  if (auto *N = dyn_cast<MDNode>(MD))
    EnumerateMDNodeOperands(N);
  else if (auto *C = dyn_cast<ConstantAsMetadata>(MD))
    EnumerateValue(C->getValue());

  HasMDString |= isa<MDString>(MD);
  HasDILocation |= isa<DILocation>(MD);
  HasGenericDINode |= isa<GenericDINode>(MD);

  // Replace the dummy ID inserted above with the correct one.  MDValueMap may
  // have changed by inserting operands, so we need a fresh lookup here.
  MDs.push_back(MD);
  MDValueMap[MD] = MDs.size();
}

/// EnumerateFunctionLocalMetadataa - Incorporate function-local metadata
/// information reachable from the metadata.
void ValueEnumerator::EnumerateFunctionLocalMetadata(
    const LocalAsMetadata *Local) {
  // Check to see if it's already in!
  unsigned &MDValueID = MDValueMap[Local];
  if (MDValueID)
    return;

  MDs.push_back(Local);
  MDValueID = MDs.size();

  EnumerateValue(Local->getValue());

  // Also, collect all function-local metadata for easy access.
  FunctionLocalMDs.push_back(Local);
}

void ValueEnumerator::EnumerateValue(const Value *V) {
  assert(!V->getType()->isVoidTy() && "Can't insert void values!");
  assert(!isa<MetadataAsValue>(V) && "EnumerateValue doesn't handle Metadata!");

  // Check to see if it's already in!
  unsigned &ValueID = ValueMap[V];
  if (ValueID) {
    // Increment use count.
    Values[ValueID-1].second++;
    return;
  }

  if (auto *GO = dyn_cast<GlobalObject>(V))
    if (const Comdat *C = GO->getComdat())
      Comdats.insert(C);

  // Enumerate the type of this value.
  EnumerateType(V->getType());

  if (const Constant *C = dyn_cast<Constant>(V)) {
    if (isa<GlobalValue>(C)) {
      // Initializers for globals are handled explicitly elsewhere.
    } else if (C->getNumOperands()) {
      // If a constant has operands, enumerate them.  This makes sure that if a
      // constant has uses (for example an array of const ints), that they are
      // inserted also.

      // We prefer to enumerate them with values before we enumerate the user
      // itself.  This makes it more likely that we can avoid forward references
      // in the reader.  We know that there can be no cycles in the constants
      // graph that don't go through a global variable.
      for (User::const_op_iterator I = C->op_begin(), E = C->op_end();
           I != E; ++I)
        if (!isa<BasicBlock>(*I)) // Don't enumerate BB operand to BlockAddress.
          EnumerateValue(*I);

      // Finally, add the value.  Doing this could make the ValueID reference be
      // dangling, don't reuse it.
      Values.push_back(std::make_pair(V, 1U));
      ValueMap[V] = Values.size();
      return;
    }
  }

  // Add the value.
  Values.push_back(std::make_pair(V, 1U));
  ValueID = Values.size();
}


void ValueEnumerator::EnumerateType(Type *Ty) {
  unsigned *TypeID = &TypeMap[Ty];

  // We've already seen this type.
  if (*TypeID)
    return;

  // If it is a non-anonymous struct, mark the type as being visited so that we
  // don't recursively visit it.  This is safe because we allow forward
  // references of these in the bitcode reader.
  if (StructType *STy = dyn_cast<StructType>(Ty))
    if (!STy->isLiteral())
      *TypeID = ~0U;

  // Enumerate all of the subtypes before we enumerate this type.  This ensures
  // that the type will be enumerated in an order that can be directly built.
  for (Type *SubTy : Ty->subtypes())
    EnumerateType(SubTy);

  // Refresh the TypeID pointer in case the table rehashed.
  TypeID = &TypeMap[Ty];

  // Check to see if we got the pointer another way.  This can happen when
  // enumerating recursive types that hit the base case deeper than they start.
  //
  // If this is actually a struct that we are treating as forward ref'able,
  // then emit the definition now that all of its contents are available.
  if (*TypeID && *TypeID != ~0U)
    return;

  // Add this type now that its contents are all happily enumerated.
  Types.push_back(Ty);

  *TypeID = Types.size();
}

// Enumerate the types for the specified value.  If the value is a constant,
// walk through it, enumerating the types of the constant.
void ValueEnumerator::EnumerateOperandType(const Value *V) {
  EnumerateType(V->getType());

  if (auto *MD = dyn_cast<MetadataAsValue>(V)) {
    assert(!isa<LocalAsMetadata>(MD->getMetadata()) &&
           "Function-local metadata should be left for later");

    EnumerateMetadata(MD->getMetadata());
    return;
  }

  const Constant *C = dyn_cast<Constant>(V);
  if (!C)
    return;

  // If this constant is already enumerated, ignore it, we know its type must
  // be enumerated.
  if (ValueMap.count(C))
    return;

  // This constant may have operands, make sure to enumerate the types in
  // them.
  for (const Value *Op : C->operands()) {
    // Don't enumerate basic blocks here, this happens as operands to
    // blockaddress.
    if (isa<BasicBlock>(Op))
      continue;

    EnumerateOperandType(Op);
  }
}

void ValueEnumerator::EnumerateAttributes(AttributeSet PAL) {
  if (PAL.isEmpty()) return;  // null is always 0.

  // Do a lookup.
  unsigned &Entry = AttributeMap[PAL];
  if (Entry == 0) {
    // Never saw this before, add it.
    Attribute.push_back(PAL);
    Entry = Attribute.size();
  }

  // Do lookups for all attribute groups.
  for (unsigned i = 0, e = PAL.getNumSlots(); i != e; ++i) {
    AttributeSet AS = PAL.getSlotAttributes(i);
    unsigned &Entry = AttributeGroupMap[AS];
    if (Entry == 0) {
      AttributeGroups.push_back(AS);
      Entry = AttributeGroups.size();
    }
  }
}

void ValueEnumerator::incorporateFunction(const Function &F) {
  InstructionCount = 0;
  NumModuleValues = Values.size();
  NumModuleMDs = MDs.size();

  // Adding function arguments to the value table.
  for (Function::const_arg_iterator I = F.arg_begin(), E = F.arg_end();
       I != E; ++I)
    EnumerateValue(I);

  FirstFuncConstantID = Values.size();

  // Add all function-level constants to the value table.
  for (Function::const_iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I!=E; ++I)
      for (User::const_op_iterator OI = I->op_begin(), E = I->op_end();
           OI != E; ++OI) {
        if ((isa<Constant>(*OI) && !isa<GlobalValue>(*OI)) ||
            isa<InlineAsm>(*OI))
          EnumerateValue(*OI);
      }
    BasicBlocks.push_back(BB);
    ValueMap[BB] = BasicBlocks.size();
  }

  // Optimize the constant layout.
  OptimizeConstants(FirstFuncConstantID, Values.size());

  // Add the function's parameter attributes so they are available for use in
  // the function's instruction.
  EnumerateAttributes(F.getAttributes());

  FirstInstID = Values.size();

  SmallVector<LocalAsMetadata *, 8> FnLocalMDVector;
  // Add all of the instructions.
  for (Function::const_iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I!=E; ++I) {
      for (User::const_op_iterator OI = I->op_begin(), E = I->op_end();
           OI != E; ++OI) {
        if (auto *MD = dyn_cast<MetadataAsValue>(&*OI))
          if (auto *Local = dyn_cast<LocalAsMetadata>(MD->getMetadata()))
            // Enumerate metadata after the instructions they might refer to.
            FnLocalMDVector.push_back(Local);
      }

      if (!I->getType()->isVoidTy())
        EnumerateValue(I);
    }
  }

  // Add all of the function-local metadata.
  for (unsigned i = 0, e = FnLocalMDVector.size(); i != e; ++i)
    EnumerateFunctionLocalMetadata(FnLocalMDVector[i]);
}

void ValueEnumerator::purgeFunction() {
  /// Remove purged values from the ValueMap.
  for (unsigned i = NumModuleValues, e = Values.size(); i != e; ++i)
    ValueMap.erase(Values[i].first);
  for (unsigned i = NumModuleMDs, e = MDs.size(); i != e; ++i)
    MDValueMap.erase(MDs[i]);
  for (unsigned i = 0, e = BasicBlocks.size(); i != e; ++i)
    ValueMap.erase(BasicBlocks[i]);

  Values.resize(NumModuleValues);
  MDs.resize(NumModuleMDs);
  BasicBlocks.clear();
  FunctionLocalMDs.clear();
}

static void IncorporateFunctionInfoGlobalBBIDs(const Function *F,
                                 DenseMap<const BasicBlock*, unsigned> &IDMap) {
  unsigned Counter = 0;
  for (Function::const_iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
    IDMap[BB] = ++Counter;
}

/// getGlobalBasicBlockID - This returns the function-specific ID for the
/// specified basic block.  This is relatively expensive information, so it
/// should only be used by rare constructs such as address-of-label.
unsigned ValueEnumerator::getGlobalBasicBlockID(const BasicBlock *BB) const {
  unsigned &Idx = GlobalBasicBlockIDs[BB];
  if (Idx != 0)
    return Idx-1;

  IncorporateFunctionInfoGlobalBBIDs(BB->getParent(), GlobalBasicBlockIDs);
  return getGlobalBasicBlockID(BB);
}

uint64_t ValueEnumerator::computeBitsRequiredForTypeIndicies() const {
  return Log2_32_Ceil(getTypes().size() + 1);
}
