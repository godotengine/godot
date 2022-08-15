#include "dxc/DxrFallback/DxrFallbackCompiler.h"

#include "dxc/Support/Global.h"
#include "dxc/Support/Unicode.h"
#include "dxc/Support/WinIncludes.h"
#include "dxc/Support/FileIOHelper.h"
#include "dxc/dxcapi.h"
#include "dxc/dxcdxrfallbackcompiler.h"
#include "dxc/Support/dxcapi.use.h"
#include "dxc/Support/dxcapi.impl.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/HLSL/DxilLinker.h"
#include "dxc/DXIL/DxilFunctionProps.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilInstructions.h"

#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "FunctionBuilder.h"
#include "LLVMUtils.h"
#include "runtime.h"
#include "StateFunctionTransform.h"

#include <queue>

using namespace hlsl;
using namespace llvm;

static std::vector<Function*> getFunctionsWithPrefix(Module* mod, const std::string& prefix)
{
  std::vector<Function*> functions;
  for (auto F = mod->begin(), E = mod->end(); F != E; ++F)
  {
    StringRef name = F->getName();
    if (name.startswith(prefix))
      functions.push_back(F);
  }
  return functions;
}


static bool inlineFunc(CallInst* call, Function* Fimpl)
{
  // Note. LLVM inlining may not be sufficient if the function references DX 
  // resources because the corresponding metadata is not created if the function
  // comes from another module.

  // Make sure that we have a definition for the called function in this module
  Function* F = call->getCalledFunction();
  Module* dstM = F->getParent();
  if (F->isDeclaration())
  {
    // Map called functions in impl module to functions in this one (because the
    // cloning step doesn't do this automatically)
    ValueToValueMapTy VMap;
    for (auto& I : inst_range(Fimpl))
    {
      if (CallInst* c = dyn_cast<CallInst>(&I))
      {
        Function* calledFimpl = c->getCalledFunction();
        if (VMap.count(calledFimpl))
          continue;

        Constant* calledF = dstM->getOrInsertFunction(calledFimpl->getName(), calledFimpl->getFunctionType(), calledFimpl->getAttributes());
        VMap[calledFimpl] = calledF;
      }
    }

    // Map arguments
    for (auto SI = Fimpl->arg_begin(), SE = Fimpl->arg_end(), DI = F->arg_begin(); SI != SE; ++SI, ++DI)
      VMap[SI] = DI;

    SmallVector<ReturnInst*, 4> returns;
    CloneFunctionInto(F, Fimpl, VMap, true, returns);
    F->setLinkage(GlobalValue::InternalLinkage);
  }

  InlineFunctionInfo IFI;
  return InlineFunction(call, IFI, false);
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


static inline Function* getOrInsertFunction(Module* mod, Function* F)
{
  return dyn_cast<Function>(mod->getOrInsertFunction(F->getName(), F->getFunctionType()));
}


template<typename K, typename V>
V get(std::map<K, V>& theMap, const K& key, V defaultVal = static_cast<V>(nullptr))
{
  auto it = theMap.find(key);
  if (it == theMap.end())
    return defaultVal;
  else
    return it->second;
}


DxrFallbackCompiler::DxrFallbackCompiler(llvm::Module* mod, const std::vector<std::string>& shaderNames, unsigned maxAttributeSize, unsigned stackSizeInBytes, bool findCalledShaders /*= false*/)
  : m_module(mod)
  , m_entryShaderNames(shaderNames)
  , m_stackSizeInBytes(stackSizeInBytes)
  , m_maxAttributeSize(maxAttributeSize)
  , m_findCalledShaders(findCalledShaders)
{}

void DxrFallbackCompiler::compile(std::vector<int>& shaderEntryStateIds, std::vector<unsigned int> &shaderStackSizes, IntToFuncNameMap *pCachedMap)
{
  std::vector<std::string> shaderNames = m_entryShaderNames;
  initShaderMap(shaderNames);

  // Bring in runtime so we can get the runtime data type
  linkRuntime();
  Type* runtimeDataArgTy = getRuntimeDataArgType();
  
  // Make sure all calls to intrinsics and shaders are at function scope and 
  // fix up control flow.
  lowerAnyHitControlFlowFuncs();
  lowerReportHit();
  lowerTraceRay(runtimeDataArgTy);
  
  // Create state functions
  IntToFuncMap stateFunctionMap; // stateID -> state function
  const int baseStateId = 1000;  // could be anything but this makes stateIds more recognizable 
  createStateFunctions(stateFunctionMap, shaderEntryStateIds, shaderStackSizes, baseStateId, shaderNames, runtimeDataArgTy);

  if (pCachedMap)
  {
      for (auto &entry : stateFunctionMap)
      {
          (*pCachedMap)[entry.first] = entry.second->getName().str();
      }
  }
}

void DxrFallbackCompiler::link(std::vector<int>& shaderEntryStateIds, std::vector<unsigned int> &shaderStackSizes, IntToFuncNameMap *pCachedMap)
{
    IntToFuncMap stateFunctionMap; // stateID -> state function
    if (pCachedMap)
    {
        for (auto entry : *pCachedMap)
        {
            stateFunctionMap[entry.first] = m_module->getFunction(entry.second);
        }
    }
    else
    {
        for (UINT i = 0; i < shaderEntryStateIds.size(); i++)
        {
            UINT substateIndex = 0;
            UINT baseStateId = shaderEntryStateIds[i];
            while (true)
            {
                auto substateName = m_entryShaderNames[i] + ".ss_" + std::to_string(substateIndex);

                auto function = m_module->getFunction(substateName);
                if (!function) break;
                stateFunctionMap[baseStateId + substateIndex] = m_module->getFunction(substateName);
                substateIndex++;
            }
        }
    }
    
    // Fix up scheduler
    Function* schedulerFunc = m_module->getFunction("fb_Fallback_Scheduler");
    createLaunchParams(schedulerFunc);

    Type* runtimeDataArgTy = getRuntimeDataArgType();
    createStateDispatch(schedulerFunc, stateFunctionMap, runtimeDataArgTy);
    createStack(schedulerFunc);

    lowerIntrinsics();
}


void DxrFallbackCompiler::setDebugOutputLevel(int val)
{
  m_debugOutputLevel = val;
}

static bool isShader(Function* F)
{
  if (F->hasFnAttribute("exp-shader"))
    return true;

  DxilModule& DM = F->getParent()->GetDxilModule();
  return (DM.HasDxilFunctionProps(F) && DM.GetDxilFunctionProps(F).IsRay());
}

DXIL::ShaderKind getRayShaderKind(Function* F)
{
  if (F->hasFnAttribute("exp-shader"))
    return DXIL::ShaderKind::RayGeneration;

  DxilModule& DM = F->getParent()->GetDxilModule();
  if (DM.HasDxilFunctionProps(F) && DM.GetDxilFunctionProps(F).IsRay())
    return DM.GetDxilFunctionProps(F).shaderKind;

  return DXIL::ShaderKind::Invalid;
}


// Some shaders should use the "pending" values of intrinsics instead of the 
// committed ones. In particular anyhit and intersection shaders use the
// pending values with the exception that the committed rayTCurrent should be
// used in intersection.
static bool shouldUsePendingValue(Function* F, StringRef instrinsicName)
{
  DxilModule& DM = F->getParent()->GetDxilModule();
  if (!DM.HasDxilFunctionProps(F))
    return false;
  const hlsl::DxilFunctionProps& props = DM.GetDxilFunctionProps(F);

  return props.IsAnyHit() || (props.IsIntersection() && instrinsicName != "rayTCurrent");
}

void DxrFallbackCompiler::initShaderMap(std::vector<std::string>& shaderNames)
{
  // Clean names and initialize shaderMap
  StringToFuncMap allShadersMap;
  for (Function& F : m_module->functions())
  {
    if (isShader(&F))
    {
      if (!F.isDeclaration())
        allShadersMap[cleanName(F.getName())] = &F;
    }

    F.removeFnAttr(Attribute::NoInline);
  }


  for (auto& name : shaderNames)
    m_shaderMap[name] = allShadersMap[name];


  if (!m_findCalledShaders)
    return;


  // Create a map from shader name to CallGraphNode
  CallGraph callGraph(*m_module);
  std::map<std::string, CallGraphNode*> allShaderNodes;
  for (auto& kv : m_shaderMap)
  {
    const std::string& name = kv.first;
    Function* func = kv.second;
    allShaderNodes[name] = callGraph[func];
  }

  // Start traversing the call graph from given shaderNames
  std::deque<CallGraphNode*> workList;
  for (auto& name : shaderNames)
    workList.push_back(allShaderNodes[name]);
  while (!workList.empty())
  {
    CallGraphNode* cur = workList.front();
    workList.pop_front();
    for (size_t i = 0; i < cur->size(); ++i)
    {
      Function* nextFunc = (*cur)[i]->getFunction();
      if (!nextFunc)
        continue;
      if (isShader(nextFunc))
      {
        const std::string nextName = cleanName(nextFunc->getName());
        if (m_shaderMap.count(nextName) == 0) // not in the shaderMap yet?
        {
          workList.push_back(allShaderNodes[nextName]);
          shaderNames.push_back(nextName);
          m_shaderMap[nextName] = workList.back()->getFunction();
        }
      }
    }
  }
}

void DxrFallbackCompiler::linkRuntime()
{
  Linker linker(m_module);
  std::unique_ptr<Module> runtimeModule = loadModuleFromAsmString(m_module->getContext(), getRuntimeString());
  bool linkErr = linker.linkInModule(runtimeModule.get());
  assert(!linkErr && "Error linking runtime");
  UNREFERENCED_PARAMETER(linkErr);

}

static void inlineFuncAndAddRet(CallInst* call, Function*F)
{
  // Add a return after the function call.
  // Should be followed immediately by "unreachable". Turn that into a "ret void".
  Instruction* ret = ReturnInst::Create(call->getContext());
  ReplaceInstWithInst(call->getParent()->getTerminator(), ret);

  bool success = inlineFunc(call, F);
  assert(success);
  UNREFERENCED_PARAMETER(success);
}

void DxrFallbackCompiler::lowerAnyHitControlFlowFuncs()
{
  std::vector<CallInst*> callsToIgnoreHit = getCallsInShadersToFunction("dx.op.ignoreHit");
  if (!callsToIgnoreHit.empty())
  {
    Function* ignoreHitFunc = m_module->getFunction("\x1?Fallback_IgnoreHit@@YAXXZ");
    assert(ignoreHitFunc && "IgnoreHit() implementation not found");
    for (CallInst* call : callsToIgnoreHit)
      inlineFuncAndAddRet(call, ignoreHitFunc);
  }

  std::vector<CallInst*> callsToAcceptHitAndEndSearch = getCallsInShadersToFunction("dx.op.acceptHitAndEndSearch");
  if (!callsToAcceptHitAndEndSearch.empty())
  {
    Function* acceptHitAndEndSearchFunc = m_module->getFunction("\x1?Fallback_AcceptHitAndEndSearch@@YAXXZ");
    assert(acceptHitAndEndSearchFunc && "AcceptHitAndEndSearch() implementation not found");
    for (CallInst* call : callsToAcceptHitAndEndSearch)
      inlineFuncAndAddRet(call, acceptHitAndEndSearchFunc);
  }
}

void DxrFallbackCompiler::lowerReportHit()
{
  std::vector<CallInst*> callsToReportHit = getCallsInShadersToFunctionWithPrefix("dx.op.reportHit");
  if (callsToReportHit.empty())
    return;

  Function* reportHitFunc = m_module->getFunction("\x1?Fallback_ReportHit@@YAHMI@Z");
  assert(reportHitFunc && "ReportHit() implementation not found");

  LLVMContext& C = m_module->getContext();
  for (CallInst* call : callsToReportHit)
  {
    // Wrap attribute arguments in Fallback_SetPendingAttr() call
    Instruction* insertBefore = call;
    hlsl::DxilInst_ReportHit reportHitCall(call);

    Value* attr = reportHitCall.get_Attributes();
    Function* setPendingAttrFunc = FunctionBuilder(m_module, "\x1?Fallback_SetPendingAttr@@").voidTy().type(attr->getType(), "attr").build();
    CallInst::Create(setPendingAttrFunc, { attr }, "", insertBefore);

    // Make call to implementation and load result
    CallInst* callImpl = CallInst::Create(reportHitFunc, { reportHitCall.get_THit(), reportHitCall.get_HitKind() }, "reportHit.result", insertBefore);
    Value* result = callImpl;

    // Result < 0 ==> ret
    Value* zero = makeInt32(0, C);
    Value* ltz = new ICmpInst(insertBefore, CmpInst::ICMP_SLT, result, zero, "endSearch");
    BasicBlock* prevBlock = call->getParent();
    BasicBlock* retBlock = prevBlock->splitBasicBlock(call, "endSearch");
    BasicBlock* nextBlock = retBlock->splitBasicBlock(call, "afterReportHit");
    ReplaceInstWithInst(prevBlock->getTerminator(), BranchInst::Create(retBlock, nextBlock, ltz));
    ReplaceInstWithInst(retBlock->getTerminator(), ReturnInst::Create(C));

    // Compare result to zero and store into original result
    Value* gtz = new ICmpInst(insertBefore, CmpInst::ICMP_SGT, result, zero, "accepted");
    call->replaceAllUsesWith(gtz);

    bool success = inlineFunc(callImpl, reportHitFunc);
    assert(success);
    (void)success;

    call->eraseFromParent();
  }
}

void DxrFallbackCompiler::lowerTraceRay(Type* runtimeDataArgTy)
{
  std::vector<CallInst*> callsToTraceRay = getCallsInShadersToFunctionWithPrefix("dx.op.traceRay");
  if (callsToTraceRay.empty())
  {
    // TODO: It might be worth dropping this from the tests eventually
    callsToTraceRay = getCallsInShadersToFunctionWithPrefix("\x1?TraceRayTest@@");
    if (callsToTraceRay.empty())
      return;
  }

  std::vector<Function*> traceRayImpl = getFunctionsWithPrefix(m_module, "\x1?Fallback_TraceRay@@");
  assert(traceRayImpl.size() == 1 && "Could not find Fallback_TraceRay() implementation");

  enum { CLOSEST_HIT = 0, MISS = 1 };
  Function* traceRaySave[] = { m_module->getFunction("traceRaySave_ClosestHit"), m_module->getFunction("traceRaySave_Miss") };
  Function* traceRayRestore[] = { m_module->getFunction("traceRayRestore_ClosestHit"), m_module->getFunction("traceRayRestore_Miss") };
  assert(traceRaySave[CLOSEST_HIT] && traceRayRestore[CLOSEST_HIT] && traceRaySave[MISS] && traceRayRestore[MISS] &&
    "Could not find TraceRay spill functions");

  Function* dummyRuntimeDataArgFunc = StateFunctionTransform::createDummyRuntimeDataArgFunc(m_module, runtimeDataArgTy);
  assert(dummyRuntimeDataArgFunc && "dummyRuntimeDataArg function could not be created.");

  // Process calls
  LLVMContext& C = m_module->getContext();
  Type* int32Ty = Type::getInt32Ty(C);
  std::map<FunctionType*, Function*> movePayloadToStackFuncs;
  std::map<Function*, AllocaInst*> funcToSpillAlloca;
  for (CallInst* call : callsToTraceRay)
  {
    Instruction* insertBefore = call;

    
    // Spill runtime data values, if necessary (closesthit and miss shaders)
    Function* caller = call->getParent()->getParent();
    DXIL::ShaderKind kind = getRayShaderKind(caller);
    if (kind == DXIL::ShaderKind::ClosestHit || kind == DXIL::ShaderKind::Miss)
    {
      int sh = (kind == DXIL::ShaderKind::ClosestHit) ? CLOSEST_HIT : MISS;
      AllocaInst* spillAlloca = get(funcToSpillAlloca, caller);
      if (!spillAlloca)
      {
        Argument* spillAllocaArg = (++traceRaySave[sh]->arg_begin());
        Type* spillAllocaTy = spillAllocaArg->getType()->getPointerElementType();
        spillAlloca = new AllocaInst(spillAllocaTy, "spill.alloca", caller->getEntryBlock().begin());
        funcToSpillAlloca[caller] = spillAlloca;
      }
      
      // Create calls. SFT will inline them.
      Value* runtimeDataArg = CallInst::Create(dummyRuntimeDataArgFunc, "runtimeData", insertBefore);
      CallInst::Create(traceRaySave[sh], {runtimeDataArg, spillAlloca}, "", insertBefore);
      CallInst::Create(traceRayRestore[sh], {runtimeDataArg, spillAlloca}, "", getInstructionAfter(call));    
    }

    
    // Get the payload offset to pass to trace implementation
    //hlsl::DxilInst_TraceRay traceRayCall(call);
    // TODO: Avoiding the intrinsic to support the test's use of TraceRayTest
    Value* payload = call->getOperand(call->getNumArgOperands() - 1);
    FunctionType* funcType = FunctionType::get(int32Ty, { payload->getType() }, false);
    Function* movePayloadToStackFunc = getOrCreateFunction("movePayloadToStack", m_module, funcType, movePayloadToStackFuncs);
    Value* newPayloadOffset = CallInst::Create(movePayloadToStackFunc, { payload }, "new.payload.offset", insertBefore);

    // Call implementation
    unsigned i = 0;
    if (call->getCalledFunction()->getName().startswith("dx.op"))
      i += 2; // skip intrinsic number and acceleration structure (for now)
    std::vector<Value*> args;
    for (; i < call->getNumArgOperands() - 1; ++i)
      args.push_back(call->getArgOperand(i));
    args.push_back(newPayloadOffset);
    CallInst::Create(traceRayImpl[0], args, "", insertBefore);

    call->eraseFromParent();
  }
}

static std::vector<StateFunctionTransform::ParameterSemanticType> getParameterTypes(Function* F, DXIL::ShaderKind shaderKind)
{
  std::vector<StateFunctionTransform::ParameterSemanticType> paramTypes;
  if (shaderKind == DXIL::ShaderKind::AnyHit || shaderKind == DXIL::ShaderKind::ClosestHit)
  {
    paramTypes.push_back(StateFunctionTransform::PST_PAYLOAD);
    paramTypes.push_back(StateFunctionTransform::PST_ATTRIBUTE);
  }
  else if (shaderKind == DXIL::ShaderKind::Miss)
  {
    paramTypes.push_back(StateFunctionTransform::PST_PAYLOAD);
  }
  else
  {
    paramTypes.assign(F->getNumOperands(), StateFunctionTransform::PST_NONE);
  }
  return paramTypes;
}

static void collectResources(DxilModule& DM, std::set<Value*>& resources)
{
  for (auto& r : DM.GetCBuffers())
    resources.insert(r->GetGlobalSymbol());
  for (auto& r : DM.GetUAVs())
    resources.insert(r->GetGlobalSymbol());
  for (auto& r : DM.GetSRVs())
    resources.insert(r->GetGlobalSymbol());
  for (auto& r : DM.GetSamplers())
    resources.insert(r->GetGlobalSymbol());
}


void DxrFallbackCompiler::createStateFunctions(
  IntToFuncMap& stateFunctionMap,
  std::vector<int>& shaderEntryStateIds,
  std::vector<unsigned int>& shaderStackSizes,
  int baseStateId,
  const std::vector<std::string>& shaderNames,
  Type* runtimeDataArgTy
)
{
  for (auto& kv : m_shaderMap)
  {
    if (kv.second == nullptr)
      errs() << "Function not found for shader " << kv.first << "\n";
  }

  DxilModule& DM = m_module->GetOrCreateDxilModule();
  std::set<Value*> resources;
  collectResources(DM, resources);

  shaderEntryStateIds.clear();
  shaderStackSizes.clear();
  int stateId = baseStateId;
  for (auto& shader : shaderNames)
  {
    std::vector<Function*> stateFunctions;
    Function* F = m_shaderMap[shader];
    StateFunctionTransform sft(F, shaderNames, runtimeDataArgTy);
    if (m_debugOutputLevel >= 2)
      sft.setVerbose(true);
    if (m_debugOutputLevel >= 3)
      sft.setDumpFilename("dump.ll");
    if (shader == "Fallback_TraceRay")
      sft.setAttributeSize(m_maxAttributeSize);
    DXIL::ShaderKind shaderKind = getRayShaderKind(F);
    if (shaderKind != DXIL::ShaderKind::Invalid)
      sft.setParameterInfo(getParameterTypes(F, shaderKind), shaderKind == DXIL::ShaderKind::ClosestHit);
    sft.setResourceGlobals(resources);
    UINT shaderStackSize = 0;
    sft.run(stateFunctions, shaderStackSize);

    shaderEntryStateIds.push_back(stateId);
    shaderStackSizes.push_back(shaderStackSize);
    for (Function* stateF : stateFunctions)
    {
      stateFunctionMap[stateId++] = stateF;
      if (DM.HasDxilFunctionProps(F)) {
        DM.CloneDxilEntryProps(F, stateF);
      }
    }
  }

  StateFunctionTransform::finalizeStateIds(m_module, shaderEntryStateIds);
}

void DxrFallbackCompiler::createLaunchParams(Function* func)
{
  Module* mod = func->getParent();
  Function* rewrite_setLaunchParams = mod->getFunction("rewrite_setLaunchParams");
  CallInst* call = dyn_cast<CallInst>(*rewrite_setLaunchParams->user_begin());

  LLVMContext& context = mod->getContext();
  Instruction* insertBefore = call;

  Function* DTidFunc = FunctionBuilder(mod, "dx.op.threadId.i32").i32().i32().i32().build();
  Value* DTidx = CallInst::Create(DTidFunc, { makeInt32((int)hlsl::OP::OpCode::ThreadId, context), makeInt32(0, context) }, "DTidx", insertBefore);
  Value* DTidy = CallInst::Create(DTidFunc, { makeInt32((int)hlsl::OP::OpCode::ThreadId, context), makeInt32(1, context) }, "DTidy", insertBefore);

  Value* dimx = call->getArgOperand(1);
  Value* dimy = call->getArgOperand(2);

  Function* groupIndexFunc = FunctionBuilder(mod, "dx.op.flattenedThreadIdInGroup.i32").i32().i32().build();
  Value* groupIndex = CallInst::Create(groupIndexFunc, { makeInt32(96, context) }, "groupIndex", insertBefore);

  Function* fb_setLaunchParams = mod->getFunction("fb_Fallback_SetLaunchParams");
  Value* runtimeDataArg = call->getArgOperand(0);
  CallInst::Create(fb_setLaunchParams, { runtimeDataArg, DTidx, DTidy, dimx, dimy, groupIndex }, "", insertBefore);

  call->eraseFromParent();
  rewrite_setLaunchParams->eraseFromParent();
}

void DxrFallbackCompiler::createStateDispatch(Function* func, const IntToFuncMap& stateFunctionMap, Type* runtimeDataArgTy)
{
  Module* mod = func->getParent();
  Function* dispatchFunc = createDispatchFunction(stateFunctionMap, runtimeDataArgTy);
  Function* rewrite_dispatchFunc = mod->getFunction("rewrite_dispatch");
  rewrite_dispatchFunc->replaceAllUsesWith(dispatchFunc);
  rewrite_dispatchFunc->eraseFromParent();
}

void DxrFallbackCompiler::createStack(Function* func)
{
  LLVMContext& context = func->getContext();

  // We would like to allocate the properly sized stack here, but DXIL doesn't
  // allow bitcasts between objects of different sizes. So we have to use the
  // default size from the runtime and replace all the accesses later.
  Function* rewrite_createStack = m_module->getFunction("rewrite_createStack");
  CallInst* call = dyn_cast<CallInst>(*rewrite_createStack->user_begin());
  AllocaInst* stack = new AllocaInst(call->getType()->getPointerElementType(), "theStack", call);
  stack->setAlignment(sizeof(int));
  call->replaceAllUsesWith(stack);
  call->eraseFromParent();
  rewrite_createStack->eraseFromParent();

  if (m_stackSizeInBytes == 0) // Take the default
    m_stackSizeInBytes = stack->getType()->getPointerElementType()->getArrayNumElements() * sizeof(int);
  Function* rewrite_getStackSize = m_module->getFunction("rewrite_getStackSize");
  call = dyn_cast<CallInst>(*rewrite_getStackSize->user_begin());
  Value* stackSizeVal = makeInt32(m_stackSizeInBytes, context);
  call->replaceAllUsesWith(stackSizeVal);
  call->eraseFromParent();
  rewrite_getStackSize->eraseFromParent();
}

// WAR to avoid crazy <3 x float> code emitted by vanilla clang in the runtime
static bool expandFloat3(std::vector<Value*>& args, Value* arg, Instruction* insertBefore)
{
  VectorType* argTy = dyn_cast<VectorType>(arg->getType());
  if (!argTy || argTy->getVectorNumElements() != 3)
    return false;

  LLVMContext& C = arg->getContext();
  args.push_back(ExtractElementInst::Create(arg, makeInt32(0, C), "vec.x", insertBefore));
  args.push_back(ExtractElementInst::Create(arg, makeInt32(1, C), "vec.y", insertBefore));
  args.push_back(ExtractElementInst::Create(arg, makeInt32(2, C), "vec.z", insertBefore));

  return true;
}

static bool float3x4ToFloat12(std::vector<Value*>& args, Value* arg, Instruction* insertBefore)
{
  StructType* STy = dyn_cast<StructType>(arg->getType());
  if (!STy || STy->getName() != "class.matrix.float.3.4")
    return false;

  BasicBlock& entryBlock = insertBefore->getParent()->getParent()->getEntryBlock();
  AllocaInst* alloca = new AllocaInst(arg->getType(), "tmp", entryBlock.begin());
  new StoreInst(arg, alloca, insertBefore);
  VectorType* VTy = VectorType::get(Type::getFloatTy(arg->getContext()), 12);
  Value* vec12Ptr = new BitCastInst(alloca, VTy->getPointerTo(), "vec12.ptr", insertBefore);
  Value* vec12 = new LoadInst(vec12Ptr, "vec12.", insertBefore);
  args.push_back(vec12);

  return true;
}

void DxrFallbackCompiler::lowerIntrinsics()
{
  std::vector<Function*> intrinsics = getFunctionsWithPrefix(m_module, "fb_");
  assert(intrinsics.size() > 0);


  // Replace intrinsics in anyhit shaders with their pending versions
  LLVMContext& C = m_module->getContext();
  std::map<std::string, Function*> pendingIntrinsics;
  std::string pendingPrefixes[] = { "fb_dxop_pending_",  "fb_Fallback_Pending" };
  for (auto& F : intrinsics)
  {
    std::string intrinsicName;
    if (F->getName().startswith(pendingPrefixes[0]))
      intrinsicName = F->getName().substr(pendingPrefixes[0].length());
    else if (F->getName().startswith(pendingPrefixes[1]))
      intrinsicName = "Fallback_" + F->getName().substr(pendingPrefixes[1].length()).str();
    else
      continue;

    pendingIntrinsics[intrinsicName] = F;
  }

  for (Function* func : intrinsics)
  {
    StringRef intrinsicName;
    std::string name;
    bool isDxilOp = false;
    if (func->getName().startswith("fb_Fallback_"))
    {
      intrinsicName = func->getName().substr(3); // after the "fb_" prefix
      name = "\x1?" + intrinsicName.str();
    }
    else if (func->getName().startswith("fb_dxop_"))
    {
      intrinsicName = func->getName().substr(8);
      name = "dx.op." + intrinsicName.str();
      isDxilOp = true;
    }
    else
    {
      assert(0 && "Bad intrinsic");
    }
    std::vector<Function*> calledFunc = getFunctionsWithPrefix(m_module, name);
    if (calledFunc.empty())
      continue;
    std::vector<CallInst*> calls = getCallsToFunction(calledFunc[0]);
    if (calls.empty())
      continue;


    bool needsRuntimeDataArg = (intrinsicName != "Fallback_Scheduler");
    Function* pendingFunc = get(pendingIntrinsics, intrinsicName.str());
    Function* funcInModule = nullptr;
    Function* pendingFuncInModule = nullptr;
    for (CallInst* call : calls)
    {
      Function* caller = call->getParent()->getParent();
      if (needsRuntimeDataArg && !caller->hasFnAttribute("state_function"))
        continue;

      Function* F = nullptr;
      if (pendingFunc && shouldUsePendingValue(caller, intrinsicName))
      {
        if (!pendingFuncInModule)
          pendingFuncInModule = getOrInsertFunction(m_module, pendingFunc);
        F = pendingFuncInModule;
      }
      else
      {
        if (!funcInModule)
          funcInModule = getOrInsertFunction(m_module, func);
        F = funcInModule;
      }

      // insert runtime data and the rest of the arguments
      std::vector<Value*> args;
      if (needsRuntimeDataArg)
        args.push_back(caller->arg_begin());
      int argIdx = 0;
      for (auto& arg : call->arg_operands())
      {
        if (argIdx++ == 0 && isDxilOp)
          continue; // skip the intrinsic number
        if (!expandFloat3(args, arg, call) && !float3x4ToFloat12(args, arg, call))
          args.push_back(arg);
      }

      CallInst* newCall = CallInst::Create(F, args, "", call);
      if (F->getFunctionType()->getReturnType() != Type::getVoidTy(C))
      {
        newCall->takeName(call);
        call->replaceAllUsesWith(newCall);
      }
      call->eraseFromParent();
    }
  }
}

Type* DxrFallbackCompiler::getRuntimeDataArgType()
{
  // Get the first argument from a known runtime function (assuming the runtime
  // has already been linked in).
  Function* F = m_module->getFunction("stackIntPtr");
  return F->arg_begin()->getType();
}

Function* DxrFallbackCompiler::createDispatchFunction(const IntToFuncMap &stateFunctionMap, Type* runtimeDataArgTy)
{
  LLVMContext& context = m_module->getContext();
  FunctionType* stateFuncTy = FunctionType::get(Type::getInt32Ty(context), { runtimeDataArgTy }, false);

  Function* dispatchFunc = FunctionBuilder(m_module, "dispatch").i32().type(runtimeDataArgTy, "runtimeData").i32("stateID").build();
  Value* runtimeDataArg = dispatchFunc->arg_begin();
  Value* stateIdArg = ++dispatchFunc->arg_begin();
  BasicBlock* entryBlock = BasicBlock::Create(context, "entry", dispatchFunc);
  BasicBlock* badBlock = BasicBlock::Create(context, "badStateID", dispatchFunc);
  IRBuilder<> builder(badBlock);
  builder.SetInsertPoint(badBlock);
  builder.CreateRet(makeInt32(-3, context)); // return an error value

  builder.SetInsertPoint(entryBlock);
  SwitchInst* switchInst = builder.CreateSwitch(stateIdArg, badBlock, stateFunctionMap.size());
  BasicBlock* endBlock = badBlock;
  for (auto& kv : stateFunctionMap)
  {
    int stateId = kv.first;
    Function* stateFunc = kv.second;

    Value* stateFuncInModule = m_module->getOrInsertFunction(stateFunc->getName(), stateFuncTy);
    BasicBlock* block = BasicBlock::Create(context, "state_" + Twine(stateId) + "." + stateFunc->getName(), dispatchFunc, endBlock);
    builder.SetInsertPoint(block);
    Value* nextStateId = builder.CreateCall(stateFuncInModule, { runtimeDataArg }, "nextStateId");
    builder.CreateRet(nextStateId);

    switchInst->addCase(makeInt32(stateId, context), block);
  }

  return dispatchFunc;
}

std::vector<CallInst*> DxrFallbackCompiler::getCallsInShadersToFunction(const std::string& funcName)
{
  std::vector<CallInst*> calls;
  Function* F = m_module->getFunction(funcName);
  if (!F)
    return calls;

  for (User* U : F->users())
  {
    CallInst* call = dyn_cast<CallInst>(U);
    if (!call)
      continue;

    Function* caller = call->getParent()->getParent();
    auto it = m_shaderMap.find(cleanName(caller->getName()));
    if (it != m_shaderMap.end())
      calls.push_back(call);
  }
  return calls;
}

std::vector<CallInst*> DxrFallbackCompiler::getCallsInShadersToFunctionWithPrefix(const std::string& funcNamePrefix)
{
  std::vector<CallInst*> calls;
  for (Function* F : getFunctionsWithPrefix(m_module, funcNamePrefix))
  {
    for (User* U : F->users())
    {
      CallInst* call = dyn_cast<CallInst>(U);
      if (!call)
        continue;

      Function* caller = call->getParent()->getParent();
      if (m_shaderMap.count(cleanName(caller->getName())))
        calls.push_back(call);
    }
  }
  return calls;
}

void DxrFallbackCompiler::resizeStack(Function* F, unsigned sizeInBytes)
{
  // Find the stack
  AllocaInst* stack = nullptr;
  for (auto& I : F->getEntryBlock().getInstList())
  {
    AllocaInst* alloc = dyn_cast<AllocaInst>(&I);
    if (alloc && alloc->getName().startswith("theStack"))
    {
      stack = alloc;
      break;
    }
  }
  if (!stack)
    return;

  // Create a new stack
  LLVMContext& C = F->getContext();
  ArrayType* newStackTy = ArrayType::get(Type::getInt32Ty(C), sizeInBytes / sizeof(int));
  AllocaInst* newStack = new AllocaInst(newStackTy, "", stack);
  newStack->takeName(stack);

  // Remap all GEPs - replaceAllUsesWith() won't change types
  for (auto U = stack->user_begin(), UE = stack->user_end(); U != UE; )
  {
    GetElementPtrInst* gep = dyn_cast<GetElementPtrInst>(*U++);
    assert(gep && "theStack has non-gep user.");

    std::vector<Value*> idxList(gep->idx_begin(), gep->idx_end());
    GetElementPtrInst* newGep = GetElementPtrInst::CreateInBounds(newStack, idxList, "", gep);
    newGep->takeName(gep);
    gep->replaceAllUsesWith(newGep);
    gep->eraseFromParent();
  }

  stack->eraseFromParent();
}