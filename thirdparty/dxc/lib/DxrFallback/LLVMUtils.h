#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace llvm
{
  class CallInst;
  class ConstantInt;
  class Function;
  class FunctionType;
  class Instruction;
  class LLVMContext;
  class Module;
  class Pass;
}

std::vector<llvm::CallInst*> getCallsToFunction(llvm::Function* callee, const llvm::Function* caller = nullptr);

llvm::Function* getOrCreateFunction(const std::string& name, llvm::Module* module, llvm::FunctionType* funcType, std::map<llvm::FunctionType*, llvm::Function*>& typeToFuncMap);

llvm::ConstantInt* makeInt32(int val, llvm::LLVMContext& context);

llvm::Instruction* getInstructionAfter(llvm::Instruction* inst);

std::unique_ptr<llvm::Module> loadModuleFromAsmFile(llvm::LLVMContext& context, const std::string& filename);
std::unique_ptr<llvm::Module> loadModuleFromAsmString(llvm::LLVMContext& context, const std::string& str);
void saveModuleToAsmFile(const llvm::Module* module, const std::string& filename);

void dumpCFG(const llvm::Function* F, const std::string& suffix);

void runPasses(llvm::Function*, const std::vector<llvm::Pass*>& passes);
