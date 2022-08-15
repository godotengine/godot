#include "llvm/Analysis/CFGPrinter.h"  // needed for DOTGraphTraits<const Function*>
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/GraphWriter.h"


using namespace llvm;

std::vector<CallInst*> getCallsToFunction(Function* callee, const Function* caller)
{
  std::vector<CallInst*> calls;
  if (callee == nullptr)
    return calls;

  for (auto U = callee->user_begin(), UE = callee->user_end(); U != UE; ++U)
  {
    CallInst* CI = dyn_cast<CallInst>(*U);
    if (!CI) // We are not interested in uses that are not calls 
      continue;
    assert(CI->getCalledFunction() == callee);

    if (caller == nullptr || CI->getParent()->getParent() == caller)
      calls.push_back(CI);
  }
  return calls;
}

ConstantInt* makeInt32(int val, LLVMContext& context)
{
  return ConstantInt::get(Type::getInt32Ty(context), val);
}

Instruction* getInstructionAfter(Instruction* inst)
{
  return ++BasicBlock::iterator(inst);
}

std::unique_ptr<Module> loadModuleFromAsmFile(LLVMContext& context, const std::string& filename)
{
  SMDiagnostic err;
  std::unique_ptr<Module> mod = parseIRFile(filename, err, context);
  if (!mod)
  {
    err.print(filename.c_str(), errs());
    exit(1);
  }

  return mod;
}

std::unique_ptr<Module> loadModuleFromAsmString(LLVMContext& context, const std::string& str)
{
  SMDiagnostic  err;
  MemoryBufferRef memBuffer(str, "id");
  std::unique_ptr<Module> mod = parseIR(memBuffer, err, context);
  return mod;
}

void saveModuleToAsmFile(const llvm::Module* mod, const std::string& filename)
{
  std::error_code EC;
  raw_fd_ostream out(filename, EC, sys::fs::F_Text);
  if (!out.has_error())
  {
    mod->print(out, 0);
    out.close();
  }
  if (out.has_error())
  {
    errs() << "Error saving to " << filename << "\n";
    exit(1);
  }
}


void dumpCFG(const Function* F, const std::string& suffix)
{
  std::string filename = ("cfg." + F->getName() + "." + suffix + ".dot").str();

  std::error_code EC;
  raw_fd_ostream out(filename, EC, sys::fs::F_Text);
  if (!out.has_error())
  {
    errs() << "Writing '" << filename << "'...\n";
    WriteGraph(out, F, true, F->getName());
    out.close();
  }
  if (out.has_error())
  {
    errs() << "Error saving to " << filename << "\n";
    exit(1);
  }
}

Function* getOrCreateFunction(const std::string& name, Module* mod, FunctionType* funcType, std::map<FunctionType*, Function*>& typeToFuncMap)
{
  auto it = typeToFuncMap.find(funcType);
  if (it != typeToFuncMap.end())
    return it->second;

  // Give name a numerical suffix to make it unique 
  std::string uniqueName = name + std::to_string(typeToFuncMap.size());
  Function* F = dyn_cast<Function>(mod->getOrInsertFunction(uniqueName, funcType));
  typeToFuncMap[funcType] = F;
  return F;
}

void runPasses(llvm::Function* F, const std::vector<llvm::Pass*>& passes)
{
  legacy::FunctionPassManager FPM(F->getParent());
  for (Pass* pass : passes)
    FPM.add(pass);
  FPM.doInitialization();
  FPM.run(*F);
  FPM.doFinalization();
}
