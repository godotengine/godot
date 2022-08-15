#include "StateFunctionTransform.h"

#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"

#include "FunctionBuilder.h"
#include "LiveValues.h"
#include "LLVMUtils.h"
#include "Reducibility.h"

#define DBGS dbgs
//#define DBGS errs

using namespace llvm;

static const char* CALL_INDIRECT_NAME = "\x1?Fallback_CallIndirect@@YAXH@Z";
static const char* SET_PENDING_ATTR_PREFIX = "\x1?Fallback_SetPendingAttr@@";


// Create a string with printf-like arguments
inline std::string stringf(const char* fmt, ...)
{
  va_list args;
  va_start(args, fmt);
#ifdef WIN32
  int size = _vscprintf(fmt, args);
#else
  int size = vsnprintf(0, 0, fmt, args);
#endif
  va_end(args);

  std::string ret;
  if (size > 0)
  {
    ret.resize(size);
    va_start(args, fmt);
    vsnprintf(const_cast<char*>(ret.data()), size + 1, fmt, args);
    va_end(args);
  }
  return ret;
}


// Remove ELF mangling
static std::string cleanName(StringRef name)
{
  if (!name.startswith("\x1?"))
    return name;

  size_t pos = name.find("@@");
  if (pos == name.npos)
    return name;

  std::string newName = name.substr(2, pos - 2);
  return newName;
}


// Utility to append the suffix to the name of the value, but returns
// an empty string if name is empty.  This is to avoid names like ".ptr".
static std::string addSuffix(StringRef valueName, StringRef suffix)
{
  if (!valueName.empty())
  {

    if (valueName.back() == '.' && suffix.front() == '.') // avoid double dots
      return (valueName + suffix.substr(1)).str();
    else
      return (valueName + suffix).str();
  }
  else
    return valueName.str();
}


// Remove suffix from name.
static std::string stripSuffix(StringRef name, StringRef suffix)
{
  size_t pos = name.rfind(suffix);
  if (pos != name.npos)
    return name.substr(0, pos).str();
  else
    return name.str();
}


// Insert str before the final "." in filename.
static std::string insertBeforeExtension(const std::string& filename, const std::string& str)
{
  std::string ret = filename;
  size_t      pos = filename.rfind('.');
  if (pos != std::string::npos)
    ret.insert(pos, str);
  else
    ret += str;

  return ret;
}


// Inserts <functionName>-<id>-<suffix> before the extension in baseName
static std::string createDumpPath(
  const std::string& baseName,
  unsigned           id,
  const std::string& suffix,
  const std::string& functionName)
{
  std::string s;
  if (!functionName.empty())
    s = "-" + functionName;
  s += stringf("-%02d-", id) + suffix;
  return insertBeforeExtension(baseName, s);
}


// Return byte offset aligned to the alignment required by inst.
static uint64_t align(uint64_t offset, Instruction* inst, DataLayout& DL)
{
  unsigned alignment = 0;
  if (AllocaInst* ai = dyn_cast<AllocaInst>(inst))
    alignment = ai->getAlignment();

  if (alignment == 0)
    alignment = DL.getPrefTypeAlignment(inst->getType());

  return RoundUpToAlignment(offset, alignment);
}


template <class T>  // T can be Value* or Instruction*
T createCastForStack(T ptr, llvm::Type* targetPtrElemType, llvm::Instruction* insertBefore)
{
  llvm::PointerType* requiredType = llvm::PointerType::get(targetPtrElemType, ptr->getType()->getPointerAddressSpace());
  if (ptr->getType() == requiredType)
    return ptr;

  return new llvm::BitCastInst(ptr, requiredType, ptr->getName(), insertBefore);
}


static Value* createCastToInt(Value* val, Instruction* insertBefore)
{
  Type* i32Ty = Type::getInt32Ty(val->getContext());
  if (val->getType() == i32Ty)
    return val;

  if (val->getType() == Type::getInt1Ty(val->getContext()))
    return new ZExtInst(val, i32Ty, addSuffix(val->getName(), ".int"), insertBefore);

  Value* intVal = new BitCastInst(val, i32Ty, addSuffix(val->getName(), ".int"), insertBefore);
  return intVal;
}


static Value* createCastFromInt(Value* intVal, Type* ty, Instruction* insertBefore)
{
  Type* i32Ty = Type::getInt32Ty(intVal->getContext());
  if (ty == i32Ty)
    return intVal;

  std::string name = intVal->getName();
  intVal->setName(addSuffix(name, ".int"));

  // Create boolean with compare
  if (ty == Type::getInt1Ty(intVal->getContext()))
    return new ICmpInst(insertBefore, CmpInst::ICMP_SGT, intVal, makeInt32(0, intVal->getContext()), name);

  return new BitCastInst(intVal, ty, name, insertBefore);
}


// Gives every value in the given function a name. This can aid in debugging.
static void dbgNameUnnamedVals(Function* func)
{
  Type* voidTy = Type::getVoidTy(func->getContext());
  for (auto& I : inst_range(func))
  {
    if (!I.hasName() && I.getType() != voidTy)
      I.setName("v"); // LLVM will uniquify the name by adding a numeric suffix
  }
}


// Returns an iterator for the instruction after the last alloca in the entry block
// (assuming that allocas are at the top of the entry block).
static BasicBlock::iterator afterEntryBlockAllocas(Function* function)
{
  BasicBlock::iterator insertBefore = function->getEntryBlock().begin();
  while (isa<AllocaInst>(insertBefore))
    ++insertBefore;
  return insertBefore;
}


// Return all the blocks reachable from entryBlock.
static BasicBlockVector getReachableBlocks(BasicBlock* entryBlock)
{
  BasicBlockVector        blocks;
  std::deque<BasicBlock*> stack = { entryBlock };
  ::BasicBlockSet         visited = { entryBlock };
  while (!stack.empty())
  {
    BasicBlock* block = stack.front();
    stack.pop_front();

    blocks.push_back(block);

    TerminatorInst* termInst = block->getTerminator();
    for (unsigned int succ = 0, succEnd = termInst->getNumSuccessors(); succ != succEnd; ++succ)
    {
      BasicBlock* succBlock = termInst->getSuccessor(succ);
      if (visited.insert(succBlock).second)
        stack.push_front(succBlock);
    }
  }

  return blocks;
}


// Creates a new function with the same arguments and attributes as oldFunction
static Function* cloneFunctionPrototype(const Function* oldFunction, ValueToValueMapTy& VMap)
{
  std::vector<Type*> argTypes;
  for (auto I = oldFunction->arg_begin(), E = oldFunction->arg_end(); I != E; ++I)
    argTypes.push_back(I->getType());

  FunctionType* FTy = FunctionType::get(oldFunction->getFunctionType()->getReturnType(), argTypes,
    oldFunction->getFunctionType()->isVarArg());
  Function* newFunction = Function::Create(FTy, oldFunction->getLinkage(), oldFunction->getName());

  Function::arg_iterator destI = newFunction->arg_begin();
  for (auto I = oldFunction->arg_begin(), E = oldFunction->arg_end(); I != E; ++I, ++destI)
  {
    destI->setName(I->getName());
    VMap[I] = destI;
  }

  AttributeSet oldAttrs = oldFunction->getAttributes();
  for (auto I = oldFunction->arg_begin(), E = oldFunction->arg_end(); I != E; ++I)
  {
    if (Argument* Anew = dyn_cast<Argument>(VMap[I]))
    {
      AttributeSet attrs = oldAttrs.getParamAttributes(I->getArgNo() + 1);
      if (attrs.getNumSlots() > 0)
        Anew->addAttr(attrs);
    }
  }

  newFunction->setAttributes(newFunction->getAttributes().addAttributes(newFunction->getContext(), AttributeSet::ReturnIndex,
    oldAttrs.getRetAttributes()));
  newFunction->setAttributes(newFunction->getAttributes().addAttributes(newFunction->getContext(), AttributeSet::FunctionIndex,
    oldAttrs.getFnAttributes()));
  return newFunction;
}


// Creates a new function by cloning blocks reachable from entryBlock
static Function* cloneBlocksReachableFrom(BasicBlock* entryBlock, ValueToValueMapTy& VMap)
{
  Function* oldFunction = entryBlock->getParent();
  Function* newFunction = cloneFunctionPrototype(oldFunction, VMap);

  // Insert a clone of the entry block into the function.
  BasicBlock* newEntry = CloneBasicBlock(entryBlock, VMap, "", newFunction);
  VMap[entryBlock] = newEntry;

  // Clone all other blocks.
  BasicBlockVector blocks = getReachableBlocks(entryBlock);
  for (auto block : blocks)
  {
    if (block == entryBlock)
      continue;
    BasicBlock* clonedBlock = CloneBasicBlock(block, VMap, "", newFunction);
    VMap[block] = clonedBlock;
  }

  // Remap new instructions to reference blocks and instructions of the new function.
  for (auto block : blocks)
  {
    auto clonedBlock = cast<BasicBlock>(VMap[block]);
    for (BasicBlock::iterator I = clonedBlock->begin(); I != clonedBlock->end(); ++I)
    {
      RemapInstruction(I, VMap, RF_NoModuleLevelChanges | RF_IgnoreMissingEntries);
    }
  }

  // Remove phi operands incoming from blocks that are not present in the new function anymore.
  for (auto& block : *newFunction)
  {
    PHINode* firstPHI = dyn_cast<PHINode>(block.begin());
    if (firstPHI == nullptr)
      continue; // phi instructions only at beginning

    // Create set of actual predecessors
    BasicBlockSet preds(pred_begin(&block), pred_end(&block));
    if (preds.size() == firstPHI->getNumIncomingValues())
      continue;

    // Remove phi incoming blocks not in preds
    for (auto iter = block.begin(); isa<PHINode>(iter); ++iter)
    {
      std::vector<unsigned int> toRemove;
      PHINode*                  phi = cast<PHINode>(iter);
      for (unsigned int op = 0, opEnd = phi->getNumIncomingValues(); op != opEnd; ++op)
      {
        BasicBlock* pred = phi->getIncomingBlock(op);
        if (preds.count(pred) == 0)
        {
          toRemove.push_back(op);
        }
      }
      for (auto I = toRemove.rbegin(), E = toRemove.rend(); I != E; ++I)
        phi->removeIncomingValue(*I, false);
    }
  }

  return newFunction;
}


// Replace and remove calls to func with val
static void replaceValAndRemoveUnusedDummyFunc(Value* oldVal, Value* newVal, Function* caller)
{
  CallInst* call = dyn_cast<CallInst>(oldVal);
  assert(call != nullptr && "Must be a call");
  Function* func = call->getCalledFunction();
  for (CallInst* CI : getCallsToFunction(func, caller))
  {
    CI->replaceAllUsesWith(newVal);
    CI->eraseFromParent();
  }
  if (func->getNumUses() == 0)
    func->eraseFromParent();
}


// Get the integer value of val. If val is not a ConstantInt return false.
static bool getConstantValue(int& constant, const Value* val)
{
  const ConstantInt* CI = dyn_cast<ConstantInt>(val);
  if (!CI)
    return false;

  if (CI->getBitWidth() > 32)
    return false;

  constant = static_cast<int>(CI->getSExtValue());
  return true;
}

static int getConstantValue(const Value* val)
{
    const ConstantInt* CI = dyn_cast<ConstantInt>(val);
    assert(CI && CI->getBitWidth() <= 32);
    return static_cast<int>(CI->getSExtValue());
}


struct StoreInfo
{
  Function* stackIntPtrFunc;
  Value* runtimeDataArg;
  Value* baseOffset;
  Instruction* insertBefore;

  Value* val;
  std::vector<Value*> idxList;
};

// Takes the offset at which to store the next value.
// Returns the next available offset.
static int store(int offset, StoreInfo& SI, Type* ty)
{
  if (StructType* STy = dyn_cast<StructType>(ty))
  {
    SI.idxList.push_back(nullptr);
    int elIdx = 0;
    for (auto& elTy : STy->elements())
    {
      SI.idxList.back() = makeInt32(elIdx++, ty->getContext());
      offset = store(offset, SI, elTy);
    }
    SI.idxList.pop_back();
  }
  else if (ArrayType* ATy = dyn_cast<ArrayType>(ty))
  {
    Type* elTy = ATy->getArrayElementType();
    SI.idxList.push_back(nullptr);
    for (int elIdx = 0; elIdx < (int)ATy->getArrayNumElements(); ++elIdx)
    {
      SI.idxList.back() = makeInt32(elIdx, ty->getContext());
      offset = store(offset, SI, elTy);
    }
    SI.idxList.pop_back();
  }
  else if (PointerType* PTy = dyn_cast<PointerType>(ty))
  {
    SI.idxList.push_back(makeInt32(0, ty->getContext()));
    offset = store(offset, SI, PTy->getPointerElementType());
    SI.idxList.pop_back();
  }
  else
  {
    Value* val = SI.val;
    if (!SI.idxList.empty())
    {
      Value* gep = GetElementPtrInst::CreateInBounds(SI.val, SI.idxList, "", SI.insertBefore);
      val = new LoadInst(gep, "", SI.insertBefore);
    }
    if (VectorType* VTy = dyn_cast<VectorType>(ty))
    {
      std::vector<Value*>idxList = std::move(SI.idxList);
      Type* elTy = VTy->getVectorElementType();
      for (int elIdx = 0; elIdx < (int)VTy->getVectorNumElements(); ++elIdx)
      {
        Value* idxVal = makeInt32(elIdx, ty->getContext());
        Value* el = ExtractElementInst::Create(val, idxVal, "", SI.insertBefore);
        SI.val = el;
        offset = store(offset, SI, elTy);
      }
      SI.idxList = std::move(idxList);
    }
    else
    {
      Value* idxVal = makeInt32(offset, val->getContext());
      Value* intVal = createCastToInt(val, SI.insertBefore);
      Value* intPtr = CallInst::Create(SI.stackIntPtrFunc, { SI.runtimeDataArg, SI.baseOffset, idxVal }, addSuffix(val->getName(), ".ptr"), SI.insertBefore);
      new StoreInst(intVal, intPtr, SI.insertBefore);
      offset += 1;
    }
  }
  return offset;
}

// Store value to the stack at given baseOffset + offset. Will flatten aggregates and vectors.
// Returns the offset where writing left off. For pointer vals stores what is pointed to.
static int store(Value* val, Function* stackIntPtrFunc, Value* runtimeDataArg, Value* baseOffset, int offset, Instruction* insertBefore)
{
  StoreInfo SI;
  SI.stackIntPtrFunc = stackIntPtrFunc;
  SI.runtimeDataArg = runtimeDataArg;
  SI.baseOffset = baseOffset;
  SI.insertBefore = insertBefore;
  SI.val = val;

  return store(offset, SI, val->getType());
}


static Value* load(llvm::Function* m_stackIntPtrFunc, Value* runtimeDataArg, Value* offset, Value* idx, const std::string& name, Type* ty, Instruction* insertBefore)
{
  if (VectorType* VTy = dyn_cast<VectorType>(ty))
  {
    LLVMContext& C = ty->getContext();
    int baseIdx = getConstantValue(idx);
    Type* elTy = VTy->getVectorElementType();
    Value* vec = UndefValue::get(VTy);
    for (int i = 0; i < (int)VTy->getVectorNumElements(); ++i)
    {
      std::string elName = stringf("el%d.", i);
      Value* intPtr = CallInst::Create(m_stackIntPtrFunc, { runtimeDataArg, offset, makeInt32(baseIdx + i, C) }, elName + "ptr", insertBefore);
      Value* intEl = new LoadInst(intPtr, elName, insertBefore);
      Value* el = createCastFromInt(intEl, elTy, insertBefore);
      vec = InsertElementInst::Create(vec, el, makeInt32(i, C), "tmpvec", insertBefore);
    }
    vec->setName(name);
    return vec;
  }
  else
  {
    Value* intPtr = CallInst::Create(m_stackIntPtrFunc, { runtimeDataArg, offset, idx }, addSuffix(name, ".ptr"), insertBefore);
    Value* intVal = new LoadInst(intPtr, name, insertBefore);
    Value* val = createCastFromInt(intVal, ty, insertBefore); 
    return val;
  }
}

static void reg2Mem(DenseMap<Instruction*, AllocaInst*>& valToAlloca, DenseMap<AllocaInst*, Instruction*>& allocaToVal, Instruction* inst)
{
  if (valToAlloca.count(inst))
    return;

  // Convert the value to an alloca
  AllocaInst*  allocaPtr = DemoteRegToStack(*inst, false);
  if (allocaPtr)
  {
    valToAlloca[inst] = allocaPtr;
    allocaToVal[allocaPtr] = inst;
  }
}


// Utility class for rematerializing values at a callsite
class Rematerializer
{
public:
  Rematerializer(
    DenseMap<AllocaInst*, Instruction*>& allocaToVal,
    const InstructionSetVector& liveHere,
    const std::set<Value*>& resources
  )
    : m_allocaToVal(allocaToVal)
    , m_liveHere(liveHere)
    , m_resources(resources)
  {}


  // Returns true if inst can be rematerialized.
  bool canRematerialize(Instruction* inst)
  {
    if (CallInst* call = dyn_cast<CallInst>(inst))
    {
      StringRef funcName = call->getCalledFunction()->getName();
      if (funcName.startswith("dummyStackFrameSize"))
        return true;
      if (funcName.startswith("stack.ptr"))
        return true;
      if (funcName.startswith("stack.load"))
        return true;
      if (funcName.startswith("dx.op.createHandle"))
        return true;
    }
    else if (LoadInst* load = dyn_cast<LoadInst>(inst))
    {
      Value* op = load->getOperand(0);
      if (GetElementPtrInst* gep = dyn_cast<GetElementPtrInst>(op)) // for descriptor tables
        op = gep->getOperand(0);
      if (m_resources.count(op))
        return true;
    }
    else if (GetElementPtrInst* gep = dyn_cast<GetElementPtrInst>(inst))
    {
      assert(gep->hasAllConstantIndices() && "Unhandled non-constant index"); // Should have been changed to stack.ptr
      return true;
    }

    return false;
  }


  // Rematerialize the given instruction and its dependency graph, adding 
  // any nonrematerializable values that are live in the function, but not 
  // at this callsite to the work list to insure that their values are restored.
  Instruction* rematerialize(Instruction* inst, std::vector<Instruction *> workList, Instruction* insertBefore, int depth = 0)
  {
    // Signal if we hit a complex case. Deep rematerialization needs more analysis.
    // To make this robust we would need to make it possible to run the current
    // value through the live value handling pipeline: figure out where it is live,
    // reg2mem, save/restore at appropriate callsites, etc.
    assert(depth < 8);

    // Reuse an already rematerialized value?
    auto it = m_rematMap.find(inst);
    if (it != m_rematMap.end())
      return it->second;

    // Handle allocas
    if (AllocaInst* alloc = dyn_cast<AllocaInst>(inst))
    {
      assert(depth > 0); // Should only be an operand to another rematerialized value
      auto it = m_allocaToVal.find(alloc);
      if (it != m_allocaToVal.end()) // Is it a value that is live at some callsite (and reg2mem'd)?
      {
        Instruction* val = it->second;
        if (canRematerialize(val))
        {
          // Rematerialize here and store to the alloca. We may have already rematerialized a load
          // from the alloca. Any future uses will use the rematerialized value directly.
          Instruction* remat = rematerialize(val, workList, insertBefore, depth + 1);
          new StoreInst(remat, alloc, insertBefore);
        }
        else
        {
          // Value has to be restored, but it rematerialization may have extended
          // the liveness of this value to this callsite. Make sure it gets restored.
          if (!m_liveHere.count(val))
            workList.push_back(val);
        }
      }

      // Allocas are not cloned.
      return inst;
    }

    Instruction* clone = inst->clone();
    clone->setName(addSuffix(inst->getName(), ".remat"));
    for (unsigned i = 0; i < inst->getNumOperands(); ++i)
    {
      Value* op = inst->getOperand(i);
      if (Instruction* opInst = dyn_cast<Instruction>(op))
        clone->setOperand(i, rematerialize(opInst, workList, insertBefore, depth + 1));
      else
        clone->setOperand(i, op);
    }
    clone->insertBefore(insertBefore); // insert after any instructions cloned for operands
    m_rematMap[inst] = clone;
    return clone;
  }


  Instruction* getRematerializedValueFor(Instruction* val)
  {
    auto it = m_rematMap.find(val);
    if (it != m_rematMap.end())
      return it->second;
    else
      return nullptr;
  }


private:
  DenseMap<Instruction*, Instruction*> m_rematMap;    // Map instructions to their rematerialized counterparts
  DenseMap<AllocaInst*, Instruction*>& m_allocaToVal; // Map allocas for reg2mem'd live values back to the value
  const InstructionSetVector& m_liveHere;             // Values live at this callsite
  const std::set<Value*>& m_resources;                // Values for resources like SRVs, UAVs, etc.
};



StateFunctionTransform::StateFunctionTransform(Function* func, const std::vector<std::string>& candidateFuncNames, Type* runtimeDataArgTy)
  : m_function(func)
  , m_candidateFuncNames(candidateFuncNames)
  , m_runtimeDataArgTy(runtimeDataArgTy)
{
  m_functionName = cleanName(m_function->getName());
  auto it = std::find(m_candidateFuncNames.begin(), m_candidateFuncNames.end(), m_functionName);
  assert(it != m_candidateFuncNames.end());
  m_functionIdx = it - m_candidateFuncNames.begin();
}

void StateFunctionTransform::setAttributeSize(int size)
{
  m_attributeSizeInBytes = size;
}

void StateFunctionTransform::setParameterInfo(const std::vector<ParameterSemanticType>& paramTypes, bool useCommittedAttr)
{
  m_paramTypes = paramTypes;
  m_useCommittedAttr = useCommittedAttr;
}

void StateFunctionTransform::setResourceGlobals(const std::set<llvm::Value*>& resources)
{
  m_resources = &resources;
}

Function* StateFunctionTransform::createDummyRuntimeDataArgFunc(Module* mod, Type* runtimeDataArgTy)
{
  return FunctionBuilder(mod, "dummyRuntimeDataArg").type(runtimeDataArgTy).build();
}

void StateFunctionTransform::setVerbose(bool val)
{
  m_verbose = val;
}

void StateFunctionTransform::setDumpFilename(const std::string& dumpFilename)
{
  m_dumpFilename = dumpFilename;
}

void StateFunctionTransform::run(std::vector<Function*>& stateFunctions, _Out_ unsigned int &shaderStackSize)
{
  printFunction("Initial");

  init();
  printFunction("AfterInit");

  changeCallingConvention();
  printFunction("AfterCallingConvention");

  preserveLiveValuesAcrossCallsites(shaderStackSize);
  printFunction("AfterPreserveLiveValues");

  createSubstateFunctions(stateFunctions);
  printFunctions(stateFunctions, "AfterSubstateFunctions");

  lowerStackFuncs();
  printFunctions(stateFunctions, "AfterLowerStackFuncs");
}

void StateFunctionTransform::finalizeStateIds(llvm::Module* mod, const std::vector<int>& candidateFuncEntryStateIds)
{
  LLVMContext& context = mod->getContext();
  Function* func = mod->getFunction("dummyStateId");
  if (!func)
    return;

  std::vector<Instruction*> toRemove;
  for (User* U : func->users())
  {
    CallInst* call = dyn_cast<CallInst>(U);
    if (!call)
      continue;

    int  functionIdx = 0;
    int  substate = 0;
    getConstantValue(functionIdx, call->getArgOperand(0));
    getConstantValue(substate, call->getArgOperand(1));
    int stateId = candidateFuncEntryStateIds[functionIdx] + substate;

    call->replaceAllUsesWith(makeInt32(stateId, context));
    toRemove.push_back(call);
  }

  for (Instruction* v : toRemove)
    v->eraseFromParent();
  func->eraseFromParent();

}

void StateFunctionTransform::init()
{
  Module* mod = m_function->getParent();
  m_function->setName(cleanName(m_function->getName()));

  // Run preparatory passes
  runPasses(m_function, {
    //createBreakCriticalEdgesPass(),
    //createLoopSimplifyPass(),
    //createLCSSAPass(),
    createPromoteMemoryToRegisterPass()
  });

  // Make debugging a little easier by giving things names
  dbgNameUnnamedVals(m_function);


  findCallSitesIntrinsicsAndReturns();


  // Create a bunch of functions that we are going to need
  m_stackIntPtrFunc = FunctionBuilder(mod, "stackIntPtr").i32Ptr().type(m_runtimeDataArgTy, "runtimeData").i32("baseOffset").i32("offset").build();

  Instruction* insertBefore = afterEntryBlockAllocas(m_function);
  Function* runtimeDataArgFunc = createDummyRuntimeDataArgFunc(mod, m_runtimeDataArgTy);
  m_runtimeDataArg = CallInst::Create(runtimeDataArgFunc, "runtimeData", insertBefore);

  Function* stackFrameSizeFunc = FunctionBuilder(mod, "dummyStackFrameSize").i32().build();
  m_stackFrameSizeVal = CallInst::Create(stackFrameSizeFunc, "stackFrame.size", insertBefore);

  // TODO only create the values that are actually needed
  Function* payloadOffsetFunc = FunctionBuilder(mod, "payloadOffset").i32().type(m_runtimeDataArgTy, "runtimeData").build();
  m_payloadOffset = CallInst::Create(payloadOffsetFunc, { m_runtimeDataArg }, "payload.offset", insertBefore);

  Function* committedAttrOffsetFunc = FunctionBuilder(mod, "committedAttrOffset").i32().type(m_runtimeDataArgTy, "runtimeData").build();
  m_committedAttrOffset = CallInst::Create(committedAttrOffsetFunc, { m_runtimeDataArg }, "committedAttr.offset", insertBefore);

  Function* pendingAttrOffsetFunc = FunctionBuilder(mod, "pendingAttrOffset").i32().type(m_runtimeDataArgTy, "runtimeData").build();
  m_pendingAttrOffset = CallInst::Create(pendingAttrOffsetFunc, { m_runtimeDataArg }, "pendingAttr.offset", insertBefore);

  Function* stackFrameOffsetFunc = FunctionBuilder(mod, "stackFrameOffset").i32().type(m_runtimeDataArgTy, "runtimeData").build();
  m_stackFrameOffset = CallInst::Create(stackFrameOffsetFunc, { m_runtimeDataArg }, "stackFrame.offset", insertBefore);


  // lower SetPendingAttr() now
  for (CallInst* call : m_setPendingAttrCalls)
  {
    // Get the current pending attribute offset. It can change when a hit is committed
    Instruction* insertBefore = call;
    Value* currentPendingAttrOffset = CallInst::Create(pendingAttrOffsetFunc, { m_runtimeDataArg }, "cur.pendingAttr.offset", insertBefore);
    Value* attr = call->getArgOperand(0);
    createStackStore(currentPendingAttrOffset, attr, 0, insertBefore);
    call->eraseFromParent();
  }
}

void StateFunctionTransform::findCallSitesIntrinsicsAndReturns()
{
  // Create a map for log N lookup
  std::map<std::string, int> candidateFuncMap;
  for (int i = 0; i < (int)m_candidateFuncNames.size(); ++i)
    candidateFuncMap[m_candidateFuncNames[i]] = i;

  for (auto& I : inst_range(m_function))
  {
    if (CallInst* call = dyn_cast<CallInst>(&I))
    {
      StringRef calledFuncName = call->getCalledFunction()->getName();
      if (calledFuncName.startswith(SET_PENDING_ATTR_PREFIX))
        m_setPendingAttrCalls.push_back(call);
      else if (calledFuncName.startswith("movePayloadToStack"))
        m_movePayloadToStackCalls.push_back(call);
      else if (calledFuncName == CALL_INDIRECT_NAME)
        m_callSites.push_back(call);
      else
      {
        auto it = candidateFuncMap.find(cleanName(calledFuncName));
        if (it == candidateFuncMap.end())
          continue;

        assert(call->getCalledFunction()->getReturnType() == Type::getVoidTy(call->getContext()) && "Continuations with returns not supported");
        m_callSites.push_back(call);
        m_callSiteFunctionIdx.push_back(it->second);
      }
    }
    else if (ReturnInst* ret = dyn_cast<ReturnInst>(&I))
    {
      m_returns.push_back(ret);
    }
  }
}

void StateFunctionTransform::changeCallingConvention()
{
  if (!m_callSites.empty() || m_attributeSizeInBytes >= 0)
    allocateStackFrame();

  if (m_attributeSizeInBytes >= 0)
    allocateTraceFrame();

  createArgFrames();

  changeFunctionSignature();
}

static bool isCallToStackPtr(Value* inst)
{
  CallInst* call = dyn_cast<CallInst>(inst);
  if (call && call->getCalledFunction()->getName().startswith("stack.ptr"))
    return true;

  return false;
}

static void extendAllocaLifetimes(LiveValues& lv)
{
  for (Instruction* inst : lv.getAllLiveValues())
  {
    if (!inst->getType()->isPointerTy())
      continue;

    if (isa<AllocaInst>(inst) || isCallToStackPtr(inst))
      continue;

    GetElementPtrInst* gep = dyn_cast<GetElementPtrInst>(inst);
    assert(gep && "Unhandled live pointer");
    Value* ptr = gep->getPointerOperand();
    if (isCallToStackPtr(ptr))
      continue;
    AllocaInst* alloc = dyn_cast<AllocaInst>(gep->getPointerOperand());
    assert(alloc && "GEP of non-alloca pointer");

    // TODO: We need to set indices of the uses of the gep, not the gep itself
    const LiveValues::Indices* gepIndices = lv.getIndicesWhereLive(gep);
    const LiveValues::Indices* allocIndices = lv.getIndicesWhereLive(alloc);
    if (!allocIndices || *allocIndices != *gepIndices)
      lv.setIndicesWhereLive(alloc, gepIndices);
  }
}


void StateFunctionTransform::preserveLiveValuesAcrossCallsites(_Out_ unsigned int &shaderStackSize)
{
  if (m_callSites.empty())
  {
    // No stack frame. Nothing to do.
    rewriteDummyStackSize(0);
    return;
  }

  SetVector<Instruction*> stackOffsets;
  stackOffsets.insert(m_stackFrameOffset);
  if (m_payloadOffset && !m_payloadOffset->user_empty())
    stackOffsets.insert(m_payloadOffset);
  if (m_committedAttrOffset && !m_committedAttrOffset->user_empty())
    stackOffsets.insert(m_committedAttrOffset);
  if (m_pendingAttrOffset && !m_pendingAttrOffset->user_empty())
    stackOffsets.insert(m_pendingAttrOffset);

  // Do liveness analysis
  ArrayRef<Instruction*> instructions((Instruction**)m_callSites.data(), m_callSites.size());
  LiveValues lv(instructions);
  lv.run();

  // Make sure alloca lifetimes match their uses
  extendAllocaLifetimes(lv);

  // Make sure stack offsets get included
  for (auto o : stackOffsets)
    lv.setLiveAtAllIndices(o, true);

  // Add payload allocas, if any
  for (CallInst* call : m_movePayloadToStackCalls)
  {
    if (AllocaInst* payloadAlloca = dyn_cast<AllocaInst>(call->getArgOperand(0)))
      lv.setLiveAtAllIndices(payloadAlloca, true);
  }

  printSet(lv.getAllLiveValues(), "live values");



  //
  // Carve up the stack frame. 
  //
  uint64_t offsetInBytes = 0;

  // ... argument frame
  offsetInBytes += m_maxCallerArgFrameSizeInBytes;


  // ... live allocas. 
  Module* mod = m_function->getParent();
  DataLayout DL(mod);
  DenseMap<Instruction*, Instruction*> allocaToStack;
  Instruction* insertBefore = getInstructionAfter(m_stackFrameOffset);
  for (Instruction* inst : lv.getAllLiveValues())
  {
    AllocaInst* alloc = dyn_cast<AllocaInst>(inst);
    if (!alloc)
      continue;

    // Allocate a slot in the stack frame for the alloca
    offsetInBytes = align(offsetInBytes, inst, DL);
    Instruction* stackAlloca = createStackPtr(m_stackFrameOffset, alloc, offsetInBytes, insertBefore);
    alloc->replaceAllUsesWith(stackAlloca);
    allocaToStack[inst] = stackAlloca;

    offsetInBytes += DL.getTypeAllocSize(alloc->getAllocatedType());
  }
  lv.remapLiveValues(allocaToStack); // replace old allocas with stackAllocas
  for (auto& kv : allocaToStack)
    kv.first->eraseFromParent(); // delete old allocas

  // Set payload offsets now that they are all on the stack
  for (CallInst* call : m_movePayloadToStackCalls)
  {
    CallInst* payloadStackPtr = dyn_cast<CallInst>(call->getArgOperand(0));
    assert(payloadStackPtr->getCalledFunction()->getName().startswith("stack.ptr"));
    Value* baseOffset = payloadStackPtr->getArgOperand(0);
    Value* idx = payloadStackPtr->getArgOperand(1);
    Value* payloadOffset = BinaryOperator::Create(Instruction::Add, baseOffset, idx, "", call);
    call->replaceAllUsesWith(payloadOffset);
    payloadOffset->takeName(call);
    call->eraseFromParent();
  }
  //printFunction("AfterStackAllocas");


  // ... saves/restores for each call site
  // Create allocas for live values. This makes it easier to generate code because
  // we don't have to maintain the use-def chains of SSA form. We can just
  // load/store from/to the alloca for a particular value. A subsequent mem2reg
  // pass will rebuild the SSA form.
  DenseMap<Instruction*, AllocaInst*> valToAlloca;
  DenseMap<AllocaInst*, Instruction*> allocaToVal;
  for (Instruction* inst : lv.getAllLiveValues())
    reg2Mem(valToAlloca, allocaToVal, inst);
  //printFunction("AfterReg2Mem");

  uint64_t baseOffsetInBytes = offsetInBytes;
  uint64_t maxOffsetInBytes = offsetInBytes;
  for (size_t i = 0; i < m_callSites.size(); ++i)
  {
    offsetInBytes = baseOffsetInBytes;

    const InstructionSetVector& liveHere = lv.getLiveValues(i);
    std::vector<Instruction*> workList(liveHere.begin(), liveHere.end());
    std::set<Instruction*> visited;
    Rematerializer R(allocaToVal, liveHere, *m_resources);
    Instruction* saveInsertBefore = m_callSites[i];
    Instruction* restoreInsertBefore = getInstructionAfter(m_callSites[i]);
    Instruction* rematInsertBefore = nullptr; // create only if needed

    // Rematerialize stack offsets after the continuation before other restores
    for (Instruction* inst : stackOffsets)
    {
      visited.insert(inst);
      Instruction* remat = R.rematerialize(inst, workList, restoreInsertBefore);
      new StoreInst(remat, valToAlloca[inst], restoreInsertBefore);
    }
    Instruction* saveStackFrameOffset = new LoadInst(valToAlloca[m_stackFrameOffset], "stackFrame.offset", saveInsertBefore);
    Instruction* restoreStackFrameOffset = R.getRematerializedValueFor(m_stackFrameOffset);

    while (!workList.empty())
    {
      Instruction* inst = workList.back();
      workList.pop_back();
      if (!visited.insert(inst).second)
        continue;

      if (!R.canRematerialize(inst))
      {
        assert(!inst->getType()->isPointerTy() && "Can not save pointers");

        offsetInBytes = align(offsetInBytes, inst, DL);
        AllocaInst* alloca = valToAlloca[inst];

        Value* saveVal = new LoadInst(alloca, addSuffix(inst->getName(), ".save"), saveInsertBefore);
        createStackStore(saveStackFrameOffset, saveVal, offsetInBytes, saveInsertBefore);

        Value* restoreVal = createStackLoad(restoreStackFrameOffset, inst, offsetInBytes, restoreInsertBefore);
        new StoreInst(restoreVal, alloca, restoreInsertBefore);

        offsetInBytes += DL.getTypeAllocSize(inst->getType());
      }
      else if (R.getRematerializedValueFor(inst) == nullptr)
      {
        if (!rematInsertBefore)
        {
          // Create a new block after restores for rematerialized values. This 
          // ensures that we can use restored values (through their allocas) even
          // if we haven't generated the actual restore yet.
          rematInsertBefore = restoreInsertBefore->getParent()->splitBasicBlock(restoreInsertBefore, "remat_begin")->begin();
          restoreInsertBefore = m_callSites[i]->getParent()->getTerminator();
        }
        Instruction* remat = R.rematerialize(inst, workList, rematInsertBefore);
        new StoreInst(remat, valToAlloca[inst], rematInsertBefore);
      }
    }

    // Take the max offset over all call sites
    maxOffsetInBytes = std::max(maxOffsetInBytes, offsetInBytes);
  }


  // ... traceFrame (if any)
  maxOffsetInBytes += m_traceFrameSizeInBytes;


  // Set the stack size
  rewriteDummyStackSize(maxOffsetInBytes);
  shaderStackSize = maxOffsetInBytes;
}

void StateFunctionTransform::createSubstateFunctions(std::vector<Function*>& stateFunctions)
{
  // The runtime perf of split() depends on the number of blocks in the function.
  // Simplifying the CFG before the split helps reduce the cost of that operation.
  runPasses(m_function, {
    createCFGSimplificationPass()
  });

  stateFunctions.resize(m_callSites.size() + 1);
  BasicBlockVector substateEntryBlocks = replaceCallSites();
  for (size_t i = 0, e = stateFunctions.size(); i < e; ++i)
  {
    stateFunctions[i] = split(m_function, substateEntryBlocks[i], i);

    // Add an attribute so we can detect when an intrinsic is not being called
    // from a state function, and thus doesn't have access to the runtimeData pointer.
    stateFunctions[i]->addFnAttr("state_function", "true");
  }

  // Erase base function
  m_function->eraseFromParent();
  m_function = nullptr;
}

void StateFunctionTransform::allocateStackFrame()
{
  Module* mod = m_function->getParent();

  // Push stack frame in entry block. 
  Instruction* insertBefore = m_stackFrameOffset;
  Function* stackFramePushFunc = FunctionBuilder(mod, "stackFramePush").voidTy().type(m_runtimeDataArgTy, "runtimeData").i32("size").build();
  m_stackFramePush = CallInst::Create(stackFramePushFunc, { m_runtimeDataArg, m_stackFrameSizeVal }, "", insertBefore);

  // Pop the stack frame just before returns.
  Function* stackFramePop = FunctionBuilder(mod, "stackFramePop").voidTy().type(m_runtimeDataArgTy, "runtimeData").i32("size").build();
  for (Instruction* insertBefore : m_returns)
    CallInst::Create(stackFramePop, { m_runtimeDataArg, m_stackFrameSizeVal }, "", insertBefore);
}

void StateFunctionTransform::allocateTraceFrame()
{
  assert(m_attributeSizeInBytes >= 0 && "Attribute size has not been specified");

  m_traceFrameSizeInBytes =
      2 * m_attributeSizeInBytes // committed and pending attributes
    + 2 * sizeof(int);           // old committed/pending attribute offsets
  int attrSizeInInts = m_attributeSizeInBytes / sizeof(int);

  // Push the trace frame first thing so that the runtime 
  // can do setup relative to the entry stack offset.
  Module* mod = m_function->getParent();
  Instruction* insertBefore = afterEntryBlockAllocas(m_function);
  Value* attrSize = makeInt32(attrSizeInInts, mod->getContext());
  Function* traceFramePushFunc = FunctionBuilder(mod, "traceFramePush").voidTy().type(m_runtimeDataArgTy, "runtimeData").i32("attrSize").build();
  CallInst::Create(traceFramePushFunc, { m_runtimeDataArg, attrSize }, "", insertBefore);

  // Pop the trace frame just before returns.
  Function* traceFramePopFunc = FunctionBuilder(mod, "traceFramePop").voidTy().type(m_runtimeDataArgTy, "runtimeData").build();
  for (Instruction* insertBefore : m_returns)
    CallInst::Create(traceFramePopFunc, { m_runtimeDataArg }, "", insertBefore);
}

bool isTemporaryAlloca(Value* op)
{
  // TODO: Need to some analysis to figure this out. We can put the alloca on
  // the caller stack if:
  //  there is only a single callsite OR
  //  if no callsite between stores/loads and this callsite
  return true;
}

void StateFunctionTransform::createArgFrames()
{
  Module* mod = m_function->getParent();
  DataLayout DL(mod);
  Instruction* stackAllocaInsertBefore = getInstructionAfter(m_stackFrameOffset);

  // Retrieve this function's arguments from the stack
  if (m_function->getFunctionType()->getNumParams() > 0)
  {
    if (m_paramTypes.empty())
      m_paramTypes.assign(m_function->getFunctionType()->getNumParams(), PST_NONE); // assume standard argument types

    static_assert(PST_COUNT == 3, "Expected 3 parameter semantic types");
    int offsetInBytes[PST_COUNT] = { 0, 0, 0 };
    Value* baseOffset[PST_COUNT] = { nullptr, nullptr, nullptr };

    Instruction* insertBefore = stackAllocaInsertBefore;
    for (auto pst : m_paramTypes)
    {
      if (baseOffset[pst])
        continue;

      if (pst == PST_NONE)
      {
        baseOffset[pst] = BinaryOperator::Create(Instruction::Add, m_stackFrameOffset, m_stackFrameSizeVal, "callerArgFrame.offset", insertBefore);
        offsetInBytes[pst] = sizeof(int); // skip the first element in caller arg frame (returnStateID)
      }
      else if (pst == PST_PAYLOAD)
      {
        baseOffset[pst] = m_payloadOffset;
      }
      else if (pst == PST_ATTRIBUTE)
      {
        baseOffset[pst] = (m_useCommittedAttr) ? m_committedAttrOffset : m_pendingAttrOffset;
      }
      else
      {
        assert(0 && "Bad parameter type");
      }
    }

    int argIdx = 0;
    for (auto& arg : m_function->args())
    {
      ParameterSemanticType pst = m_paramTypes[argIdx];
      Value* val = nullptr;
      if (arg.getType()->isPointerTy())
      {
        // Assume that pointed to memory is on the stack.
        val = createStackPtr(baseOffset[pst], &arg, offsetInBytes[pst], insertBefore);
        offsetInBytes[pst] += DL.getTypeAllocSize(arg.getType()->getPointerElementType());
      }
      else
      {
        val = createStackLoad(baseOffset[pst], &arg, offsetInBytes[pst], insertBefore);
        offsetInBytes[pst] += DL.getTypeAllocSize(arg.getType());
      }

      // Replace use of the argument with the loaded value
      if (arg.hasName())
        val->takeName(&arg);
      else
        val->setName("arg" + std::to_string(argIdx));
      arg.replaceAllUsesWith(val);

      argIdx++;
    }
  }


  // Process function arguments for each call site
  m_maxCallerArgFrameSizeInBytes = 0;
  for (size_t i = 0; i < m_callSites.size(); ++i)
  {
    int offsetInBytes = 0;
    CallInst* call = m_callSites[i];
    FunctionType* FT = call->getCalledFunction()->getFunctionType();
    StringRef calledFuncName = call->getCalledFunction()->getName();

    Instruction* insertBefore = call;

    // Set the return stateId (next substate of this function)
    int nextSubstate = i + 1;
    Value* nextStateId = getDummyStateId(m_functionIdx, nextSubstate, insertBefore);
    createStackStore(m_stackFrameOffset, nextStateId, offsetInBytes, insertBefore);
    offsetInBytes += DL.getTypeAllocSize(nextStateId->getType());
    if (FT->getNumParams() && calledFuncName != CALL_INDIRECT_NAME)
    {
      for (unsigned index = 0; index < FT->getNumParams(); ++index)
      {
        // Save the argument from the argFrame
        Value* op = call->getArgOperand(index);
        Type* opTy = op->getType();
        if (opTy->isPointerTy())
        {
          // TODO: Until we have callable shaders we should not get here except
          // in tests.
          if (isTemporaryAlloca(op))
          {
            // We can just replace the alloca with space in the arg frame
            assert(isa<AllocaInst>(op));
            Value* stackAlloca = createStackPtr(m_stackFrameOffset, op, offsetInBytes, stackAllocaInsertBefore);
            op->replaceAllUsesWith(stackAlloca);
            cast<AllocaInst>(op)->eraseFromParent();
          }
          else
          {
            // copy in/out
            assert(0);
          }
          offsetInBytes += DL.getTypeAllocSize(opTy->getPointerElementType());
        }
        else
        {
          createStackStore(m_stackFrameOffset, op, offsetInBytes, insertBefore);
          offsetInBytes += DL.getTypeAllocSize(opTy);
        }

        // Replace use of the argument with undef
        call->setArgOperand(index, UndefValue::get(opTy));

      }
    }

    if (offsetInBytes > m_maxCallerArgFrameSizeInBytes)
      m_maxCallerArgFrameSizeInBytes = offsetInBytes;
  }
}

void StateFunctionTransform::changeFunctionSignature()
{
  // Create a new function that takes a state object pointer and returns next state ID
  // and splice in the body of the old function into the new one.
  Function* newFunc = FunctionBuilder(m_function->getParent(), m_functionName + "_tmp").i32().type(m_runtimeDataArgTy, "runtimeData").build();
  newFunc->getBasicBlockList().splice(newFunc->begin(), m_function->getBasicBlockList());
  m_function = newFunc;

  // Set the runtime data pointer and remove the dummy function .
  Value* runtimeDataArg = m_function->arg_begin();
  replaceValAndRemoveUnusedDummyFunc(m_runtimeDataArg, runtimeDataArg, m_function);
  m_runtimeDataArg = runtimeDataArg;

  // Get return stateID from stack on each return.
  LLVMContext& context = m_function->getContext();
  Value* zero = makeInt32(0, context);
  CallInst* retStackFrameOffset = m_stackFrameOffset;
  for (ReturnInst*& ret : m_returns)
  {
    Instruction* insertBefore = ret;
    if (m_stackFramePush)
      retStackFrameOffset = CallInst::Create(m_stackFrameOffset->getCalledFunction(), { m_runtimeDataArg }, "ret.stackFrame.offset", insertBefore);
    Instruction* returnStateIdPtr = CallInst::Create(m_stackIntPtrFunc, { m_runtimeDataArg, retStackFrameOffset, zero }, "ret.stateId.ptr", insertBefore);
    Value* returnStateId = new LoadInst(returnStateIdPtr, "ret.stateId", insertBefore);
    ReturnInst* newRet = ReturnInst::Create(context, returnStateId);
    ReplaceInstWithInst(ret, newRet);
    ret = newRet; // update reference
  }
}


void StateFunctionTransform::rewriteDummyStackSize(uint64_t frameSizeInBytes)
{
  assert(frameSizeInBytes % sizeof(int) == 0);
  Value*   frameSizeVal = makeInt32(frameSizeInBytes / sizeof(int), m_function->getContext());
  replaceValAndRemoveUnusedDummyFunc(m_stackFrameSizeVal, frameSizeVal, m_function);
  m_stackFrameSizeVal = frameSizeVal;
}

void StateFunctionTransform::createStackStore(Value* baseOffset, Value* val, int offsetInBytes, Instruction* insertBefore)
{
  assert(offsetInBytes % sizeof(int) == 0);
  Value* intIndex = makeInt32(offsetInBytes / sizeof(int), insertBefore->getContext());
  Value* args[] = { val, baseOffset, intIndex };
  Type* argTypes[] = { args[0]->getType(), args[1]->getType(), args[2]->getType() };
  FunctionType* FT = FunctionType::get(Type::getVoidTy(val->getContext()), argTypes, false);
  Function* F = getOrCreateFunction("stack.store", insertBefore->getModule(), FT, m_stackStoreFuncs);
  CallInst::Create(F, args, "", insertBefore);
}

Instruction* StateFunctionTransform::createStackLoad(Value* baseOffset, Value* val, int offsetInBytes, Instruction* insertBefore)
{
  assert(offsetInBytes % sizeof(int) == 0);
  Value* intIndex = makeInt32(offsetInBytes / sizeof(int), insertBefore->getContext());
  Value* args[] = { baseOffset, intIndex };
  Type* argTypes[] = { args[0]->getType(), args[1]->getType() };
  FunctionType* FT = FunctionType::get(val->getType(), argTypes, false);
  Function* F = getOrCreateFunction("stack.load", insertBefore->getModule(), FT, m_stackLoadFuncs);
  return CallInst::Create(F, args, addSuffix(val->getName(), ".restore"), insertBefore);
}

Instruction* StateFunctionTransform::createStackPtr(Value* baseOffset, Type* valTy, Value* intIndex, Instruction* insertBefore)
{
  Value* args[] = { baseOffset, intIndex };
  Type* argTypes[] = { args[0]->getType(), args[1]->getType() };
  FunctionType* FT = FunctionType::get(valTy, argTypes, false);
  Function* F = getOrCreateFunction("stack.ptr", insertBefore->getModule(), FT, m_stackPtrFuncs);
  CallInst* call = CallInst::Create(F, args, "", insertBefore);
  return call;
}

Instruction* StateFunctionTransform::createStackPtr(Value* baseOffset, Value* val, int offsetInBytes, Instruction* insertBefore)
{
  assert(offsetInBytes % sizeof(int) == 0);
  Value* intIndex = makeInt32(offsetInBytes / sizeof(int), insertBefore->getContext());
  Instruction* ptr = createStackPtr(baseOffset, val->getType(), intIndex, insertBefore);
  ptr->takeName(val);
  return ptr;
}

static bool isStackIntPtr(Value* val)
{
  CallInst* call = dyn_cast<CallInst>(val);
  return call && call->getCalledFunction()->getName().startswith("stack.ptr");
}

// This code adapted from GetElementPtrInst::accumulateConstantOffset(). 
// TODO: Use a single function for both constant and dynamic offsets? Could do
// some constant folding along the way for dynamic offsets.
Value* accumulateDynamicOffset(GetElementPtrInst* gep, const DataLayout &DL)
{
  LLVMContext& C = gep->getContext();
  Instruction* insertBefore = gep;
  Value* offset = makeInt32(0, C);
  for (gep_type_iterator GTI = gep_type_begin(gep), GTE = gep_type_end(gep); GTI != GTE; ++GTI)
  {
    ConstantInt *OpC = dyn_cast<ConstantInt>(GTI.getOperand());
    if (OpC && OpC->isZero())
      continue;

    // Handle a struct index, which adds its field offset to the pointer.
    Value* elementOffset = nullptr;
    if (StructType *STy = dyn_cast<StructType>(*GTI))
    {
      assert(OpC && "Structure indices must be constant");
      unsigned ElementIdx = OpC->getZExtValue();
      const StructLayout *SL = DL.getStructLayout(STy);
      elementOffset = makeInt32(SL->getElementOffset(ElementIdx) / sizeof(int), C);
    }
    else
    {
      // For array or vector indices, scale the index by the size of the type.
      Value* stride = makeInt32(DL.getTypeAllocSize(GTI.getIndexedType()) / sizeof(int), C);
      elementOffset = BinaryOperator::Create(Instruction::Mul, GTI.getOperand(), stride, "elOffs", insertBefore);
    }

    offset = BinaryOperator::Create(Instruction::Add, offset, elementOffset, "offs", insertBefore);
  }
  return offset;
}


// Adds gep offset to offsetVal and returns the result
static Value* accumulateGepOffset(GetElementPtrInst* gep, Value* offsetVal)
{
  Module* M = gep->getModule();
  const DataLayout& DL = M->getDataLayout();

  Value* elementOffsetVal = nullptr;
  APInt constOffset(DL.getPointerSizeInBits(), 0);
  if (gep->accumulateConstantOffset(DL, constOffset))
    elementOffsetVal = makeInt32((int)constOffset.getZExtValue() / sizeof(int), M->getContext());
  else
    elementOffsetVal = accumulateDynamicOffset(gep, DL);
  elementOffsetVal = BinaryOperator::Create(Instruction::Add, offsetVal, elementOffsetVal, "offs", gep);

  return elementOffsetVal;
}

// Turn GEPs on a stack.ptr of aggregate type into stack.ptrs of scalar type
void StateFunctionTransform::flattenGepsOnValue(Value* val, Value* baseOffset, Value* offsetVal)
{
  for (auto U = val->user_begin(), UE = val->user_end(); U != UE;)
  {
    User* user = *U++;
    if (CallInst* call = dyn_cast<CallInst>(user))
    {
      // inline the call to expose GEPs and restart the loop. 
      InlineFunctionInfo IFI;
      bool success = InlineFunction(call, IFI, false);
      assert(success);
      (void)success; 

      U = val->user_begin();
      UE = val->user_end();
      continue;
    }

    GetElementPtrInst* gep = dyn_cast<GetElementPtrInst>(user);
    if (!gep)
      continue;

    Value* elementOffsetVal = accumulateGepOffset(gep, offsetVal);
    Type* gepElTy = gep->getType()->getPointerElementType();
    if (gepElTy->isAggregateType())
    {
      // flatten geps on this gep
      flattenGepsOnValue(gep, baseOffset, elementOffsetVal);
    }
    else if (isa<VectorType>(gepElTy))
      scalarizeVectorStackAccess(gep, baseOffset, elementOffsetVal);
    else 
    {
      Value* ptr = createStackPtr(baseOffset, gep->getType(), elementOffsetVal, gep);
      ptr->takeName(gep); // could use a name that encodes the gep type and indices
      gep->replaceAllUsesWith(ptr);
    }

    gep->eraseFromParent();
  }
}


void StateFunctionTransform::scalarizeVectorStackAccess(Instruction* vecPtr, Value* baseOffset, Value* offsetVal)
{
  std::vector<Value*> elPtrs;
  Type* VTy = vecPtr->getType()->getPointerElementType();
  Type* elTy = VTy->getVectorElementType();
  LLVMContext& C = vecPtr->getContext();
  Value* curOffsetVal = offsetVal;
  Value* one = makeInt32(1, C);
  offsetVal->setName("offs0.");
  for (unsigned i = 0; i < VTy->getVectorNumElements(); ++i)
  {
    // TODO: If offsetVal is a constant we could just create constants instead of add instructions
    if (i > 0)
      curOffsetVal = BinaryOperator::Create(Instruction::Add, curOffsetVal, one, stringf("offs%d.", i), vecPtr);
    elPtrs.push_back(createStackPtr(baseOffset, elTy->getPointerTo(), curOffsetVal, vecPtr));
    elPtrs.back()->setName(addSuffix(vecPtr->getName(), stringf(".el%d.", i)));
  }

  // Scalarize load/stores
  for (auto U = vecPtr->user_begin(), UE = vecPtr->user_end(); U != UE;)
  {
    User* user = *U++;
    if (LoadInst* load = dyn_cast<LoadInst>(user))
    {
      Value* vec = UndefValue::get(VTy);
      for (size_t i = 0; i < elPtrs.size(); ++i)
      {
        Value* el = new LoadInst(elPtrs[i], stringf("el%d.", i), load);
        vec = InsertElementInst::Create(vec, el, makeInt32(i, C), "vec", load);
      }
      load->replaceAllUsesWith(vec);
      load->eraseFromParent();
    }
    else if (StoreInst* store = dyn_cast<StoreInst>(user))
    {
      Value* vec = store->getOperand(0);
      for (size_t i = 0; i < elPtrs.size(); ++i)
      {
        Value* el = ExtractElementInst::Create(vec, makeInt32(i, C), stringf("el%d.", i), store);
        new StoreInst(el, elPtrs[i], store);
      }
      store->eraseFromParent();
    }
    else
    {
      assert(0 && "Unhandled user");
    }
  }
}


void StateFunctionTransform::lowerStackFuncs()
{
  LLVMContext& C = m_stackIntPtrFunc->getContext();
  const DataLayout& DL = m_stackIntPtrFunc->getParent()->getDataLayout();

  // stack.store functions
  for (auto& kv : m_stackStoreFuncs)
  {
    Function* F = kv.second;
    for (auto U = F->user_begin(); U != F->user_end(); )
    {
      CallInst* call = dyn_cast<CallInst>(*(U++));
      assert(call);

      Value* runtimeDataArg = call->getParent()->getParent()->arg_begin();
      Value* val = call->getArgOperand(0);
      Value* offset = call->getArgOperand(1);
      int idx = getConstantValue(call->getArgOperand(2));

      Instruction* insertBefore = call;
      if (isStackIntPtr(val))
      {
        // Copy from one part of the stack to another
        CallInst* valCall = dyn_cast<CallInst>(val);
        Value* srcOffset = valCall->getArgOperand(0);
        int srcIdx = getConstantValue(valCall->getArgOperand(1));
        Value* dstOffset = offset;
        int dstIdx = idx;
        int intCount = (int)DL.getTypeAllocSize(val->getType()->getPointerElementType()) / sizeof(int);
        for (int i = 0; i < intCount; ++i)
        {
          std::string idxStr = stringf("%d.", i);
          Value* srcPtr = CallInst::Create(m_stackIntPtrFunc, { runtimeDataArg, srcOffset, makeInt32(srcIdx + i, C) }, addSuffix(val->getName(), ".ptr" + idxStr), insertBefore);
          Value* dstPtr = CallInst::Create(m_stackIntPtrFunc, { runtimeDataArg, dstOffset, makeInt32(dstIdx + i, C) }, "dst.ptr" + idxStr, insertBefore);
          Value* intVal = new LoadInst(srcPtr, "copy.val" + idxStr, insertBefore);
          new StoreInst(intVal, dstPtr, insertBefore);
        }
      }
      else
      {
        store(val, m_stackIntPtrFunc, runtimeDataArg, offset, idx, insertBefore);
      }

      call->eraseFromParent();
    }
    F->eraseFromParent();
  }

  // stack.load functions
  for (auto& kv : m_stackLoadFuncs)
  {
    Function* F = kv.second;
    for (auto U = F->user_begin(); U != F->user_end(); )
    {
      CallInst* call = dyn_cast<CallInst>(*(U++));
      assert(call);

      std::string name = stripSuffix(call->getName(), ".restore");
      call->setName("");
      Value* runtimeDataArg = call->getParent()->getParent()->arg_begin();
      Value* offset = call->getArgOperand(0);
      Value* idx = call->getArgOperand(1);

      Instruction* insertBefore = call;
      Value* val = load(m_stackIntPtrFunc, runtimeDataArg, offset, idx, name, call->getType(), insertBefore);
      call->replaceAllUsesWith(val);
      call->eraseFromParent();
    }
    F->eraseFromParent();
  }


  // Scalarize accesses based on a stack.ptr func
  for (auto& kv : m_stackPtrFuncs)
  {
    Function* F = kv.second;
    if (!F->getReturnType()->getPointerElementType()->isAggregateType())
      continue;
    for (auto U = F->user_begin(), UE = F->user_end(); U != UE; )
    {
      CallInst* call = dyn_cast<CallInst>(*(U++));
      assert(call);

      Value* offset = call->getArgOperand(0);
      Value* idx = call->getArgOperand(1);
      flattenGepsOnValue(call, offset, idx);
      call->eraseFromParent();
    }
  }


  // stack.ptr functions
  for (auto& kv : m_stackPtrFuncs)
  {
    Function* F = kv.second;
    for (auto U = F->user_begin(); U != F->user_end(); )
    {
      CallInst* call = dyn_cast<CallInst>(*(U++));
      assert(call);

      std::string name = call->getName();
      Value* runtimeDataArg = call->getParent()->getParent()->arg_begin();
      Value* offset = call->getArgOperand(0);
      Value* idx = call->getArgOperand(1);

      Instruction* insertBefore = call;
      Value* ptr = CallInst::Create(m_stackIntPtrFunc, { runtimeDataArg, offset, idx }, addSuffix(name, ".ptr"), insertBefore);
      if (ptr->getType() != call->getType())
        ptr = new BitCastInst(ptr, call->getType(), "", insertBefore);
      ptr->takeName(call);
      call->replaceAllUsesWith(ptr);
      call->eraseFromParent();
    }
    F->eraseFromParent();
  }
}

Function* StateFunctionTransform::split(Function* baseFunc, BasicBlock* substateEntryBlock, int substateIndex)
{
  ValueToValueMapTy VMap;
  Function*         substateFunc = cloneBlocksReachableFrom(substateEntryBlock, VMap);
  Module*           mod = baseFunc->getParent();
  mod->getFunctionList().push_back(substateFunc);
  substateFunc->setName(m_functionName + ".ss_" + std::to_string(substateIndex));

  if (substateIndex != 0)
  {
    // Collect allocas from entry block
    SmallVector<Instruction*, 16> allocasToClone;
    for (auto& I : baseFunc->getEntryBlock().getInstList())
    {
      if (isa<AllocaInst>(&I))
        allocasToClone.push_back(&I);
    }

    // Clone collected allocas
    BasicBlock* newEntryBlock = &substateFunc->getEntryBlock();
    for (auto I : allocasToClone)
    {
      // Collect users of original instruction in substateFunc
      std::vector<Instruction*> users;
      for (auto U : I->users())
      {
        Instruction* inst = dyn_cast<Instruction>(U);
        if (inst->getParent()->getParent() == substateFunc)
          users.push_back(inst);
      }

      if (users.empty())
        continue;

      // Clone instruction
      Instruction* clone = I->clone();
      if (I->hasName())
        clone->setName(I->getName());
      clone->insertBefore(newEntryBlock->getFirstInsertionPt()); // allocas first in entry block
      RemapInstruction(clone, VMap, RF_NoModuleLevelChanges | RF_IgnoreMissingEntries);

      // Replaces uses
      for (auto user : users)
        user->replaceUsesOfWith(I, clone);
    }
  }

  //printFunction( substateFunc, substateFunc->getName().str() + "-BeforeSplittingOpt", m_dumpId++ );

  makeReducible(substateFunc);

  // Undo the reg2mem done in preserveLiveValuesAcrossCallSites()
  runPasses(substateFunc, {
    createVerifierPass(),
    createPromoteMemoryToRegisterPass()
  });

  //printFunction( substateFunc, substateFunc->getName().str() + "-AfterSplitting", m_dumpId++ );

  return substateFunc;
}

BasicBlockVector StateFunctionTransform::replaceCallSites()
{
  LLVMContext& context = m_function->getContext();

  BasicBlockVector substateEntryPoints{ &m_function->getEntryBlock() };
  substateEntryPoints[0]->setName(m_functionName + ".BB0");

  // Add other substates by splitting blocks at call sites.
  for (size_t i = 0; i < m_callSites.size(); ++i)
  {
    CallInst*   call = m_callSites[i];
    BasicBlock* block = call->getParent();
    StringRef calledFuncName = call->getCalledFunction()->getName();

    BasicBlock* nextBlock =
      block->splitBasicBlock(call->getNextNode(), m_functionName + ".BB" + std::to_string(i + 1) + ".from."
        + cleanName(calledFuncName));
    substateEntryPoints.push_back(nextBlock);

    // Return state id for entry state of the function being called
    Instruction* insertBefore = call;
    Value* returnStateId = nullptr;
    if (calledFuncName == CALL_INDIRECT_NAME)
      returnStateId = call->getArgOperand(0);
    else
      returnStateId = getDummyStateId(m_callSiteFunctionIdx[i], 0, insertBefore);
    ReplaceInstWithInst(call->getParent()->getTerminator(), ReturnInst::Create(context, returnStateId));
    call->eraseFromParent();
  }
  return substateEntryPoints;
}

llvm::Value* StateFunctionTransform::getDummyStateId(int functionIdx, int substate, llvm::Instruction* insertBefore)
{
  if (!m_dummyStateIdFunc)
  {
    Module* M = m_function->getParent();
    m_dummyStateIdFunc = FunctionBuilder(M, "dummyStateId").i32().i32("functionIdx").i32("substate").build();
  }
  LLVMContext& context = insertBefore->getContext();
  Value* functionIdxVal = makeInt32(functionIdx, context);
  Value* substateVal = makeInt32(substate, context);
  return CallInst::Create(m_dummyStateIdFunc, { functionIdxVal, substateVal }, "stateId", insertBefore);
}

raw_ostream& StateFunctionTransform::getOutputStream(const std::string functionName, const std::string& suffix, unsigned int dumpId)
{
  if (m_dumpFilename.empty())
    return DBGS();

  const std::string filename = createDumpPath(m_dumpFilename, dumpId, suffix, functionName);
  std::error_code  errorCode;
  raw_ostream* out = new raw_fd_ostream(filename, errorCode, sys::fs::OpenFlags::F_None);
  if (errorCode)
  {
    DBGS() << "Failed to open " << filename << " for writing sft output. " << errorCode.message() << "\n";
    delete out;
    return DBGS();
  }

  return *out;
}

void StateFunctionTransform::printFunction(const Function* function, const std::string& suffix, unsigned int dumpId)
{
  if (!m_verbose)
    return;

  raw_ostream& out = getOutputStream(m_functionName, suffix, dumpId);
  out << "; ########################### " << suffix << "\n";
  out << *function << "\n";
  if (&out != &DBGS())
    delete &out;
}

void StateFunctionTransform::printFunction(const std::string& suffix)
{
  printFunction(m_function, suffix, m_dumpId++);
}

void StateFunctionTransform::printFunctions(const std::vector<Function*>& funcs, const char* suffix)
{
  if (!m_verbose)
    return;

  raw_ostream& out = getOutputStream(m_functionName, suffix, m_dumpId++);
  out << "; ########################### " << suffix << "\n";
  for (Function* F : funcs)
    out << *F << "\n";
  if (&out != &DBGS())
    delete &out;
}

void StateFunctionTransform::printModule(const Module* mod, const std::string& suffix)
{
  if (!m_verbose)
    return;

  raw_ostream& out = getOutputStream("module", suffix, m_dumpId++);
  out << "; ########################### " << suffix << "\n";
  out << *mod << "\n";
}

void StateFunctionTransform::printSet(const InstructionSetVector& vals, const char* msg)
{
  if (!m_verbose)
    return;

  raw_ostream& out = DBGS();
  if (msg)
    out << msg << " --------------------\n";

  uint64_t totalBytes = 0;
  if (vals.size() > 0)
  {
    Module*    mod = m_function->getParent();
    DataLayout DL(mod);
    for (InstructionSetVector::const_iterator I = vals.begin(), IE = vals.end(); I != IE; ++I)
    {
      const Instruction* inst = *I;
      uint64_t           size = DL.getTypeAllocSize(inst->getType());
      out << stringf("%3dB: ", size) << *inst << '\n';
      totalBytes += size;
    }
  }
  out << "Count:" << vals.size() << "  Bytes:" << totalBytes << "\n\n";
}
