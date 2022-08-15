#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace llvm
{
  class CallInst;
  class Function;
  class Module;
  class Type;
}

// Combines DXIL raytracing shaders together into a compute shader.
//
// The incoming module should contain the following functions if the corresponding
// intrinsic are called by the specified shaders,
// if called:
//    Fallback_TraceRay() 
//    Fallback_Ignore()
//    Fallback_AcceptHitAndEndSearch()
//    Fallback_ReportHit()
//
// Fallback_TraceRay() will be called with the original arguments, substituting
// the offset of the payload on the stack for the actual payload. 
// Fallback_TraceRay() will also be used to replace calls to TraceRayTest().
//
// ReportHit() returns a boolean. But to handle the abort of the intersection
// shader when AcceptHitAndEndSearch() is called we need a third return value.
// Fallback_ReportHit() should return an integer < 0 for end search, 0 for ignore, 
// and > 0 for accept.
//
// The module should also contain a single call to Fallback_Scheduler() in the
// entry shader for the raytracing compute shader.
//
// resizeStack() needs to be called after inlining everything in the compute 
// shader.
//
// Currently the main scheduling loop and the implementation for intrinsic 
// functions come from an internal runtime module.
class DxrFallbackCompiler
{
public:
  typedef std::map<int, std::string> IntToFuncNameMap;

  // If findCalledShaders is true, then the list of shaderNames is expanded to 
  // include shader functions (functions with attribute "exp-shader") that are 
  // called by functions in shaderNames. Shader entry state IDs are still
  // returned only for those originally in shaderNames. findCalledShaders used 
  // for testing.
  DxrFallbackCompiler(llvm::Module* mod, const std::vector<std::string>& shaderNames, unsigned maxAttributeSize, unsigned stackSizeInBytes, bool findCalledShaders = false);

  // 0 - no debug output
  // 1 - dump initial combined module, compiled module, and final linked module
  // 2 - dump intermediate stages of SFT to console
  // 3 - dump intermediate stages of SFT to file
  void setDebugOutputLevel(int val);

  // Returns the entry state id for each of shaderNames. The transformations 
  // are performed in place on the module.
  void compile(std::vector<int>& shaderEntryStateIds, std::vector<unsigned int> &shaderStackSizes, IntToFuncNameMap *pCachedMap);
  void link(std::vector<int>& shaderEntryStateIds, std::vector<unsigned int> &shaderStackSizes, IntToFuncNameMap *pCachedMap);
  // TODO: Ideally we would run this after inlining everything at the end of compile.
  // Until we figure out to do this, we will call the function after the final link.
  static void resizeStack(llvm::Function* F, unsigned stackSizeInBytes);
private:
  typedef std::map<int, llvm::Function*> IntToFuncMap;
  typedef std::map<std::string, llvm::Function*> StringToFuncMap;

  llvm::Module* m_module = nullptr;
  const std::vector<std::string>& m_entryShaderNames;
  unsigned m_stackSizeInBytes = 0;
  unsigned m_maxAttributeSize = 0;
  bool m_findCalledShaders = false;
  int m_debugOutputLevel = 0;

  StringToFuncMap m_shaderMap;

  void initShaderMap(std::vector<std::string>& shaderNames);
  void linkRuntime();
  void lowerAnyHitControlFlowFuncs();
  void lowerReportHit();
  void lowerTraceRay(llvm::Type* runtimeDataArgTy);
  void createStateFunctions(IntToFuncMap& stateFunctionMap, std::vector<int>& shaderEntryStateIds, std::vector<unsigned int>& shaderStackSizes, int baseStateId, const std::vector<std::string>& shaderNames, llvm::Type* runtimeDataArgTy);
  void createLaunchParams(llvm::Function* func);
  void createStack(llvm::Function* func);
  void createStateDispatch(llvm::Function* func, const IntToFuncMap& stateFunctionMap, llvm::Type* runtimeDataArgTy);
  void lowerIntrinsics();

  llvm::Type* getRuntimeDataArgType();
  llvm::Function* createDispatchFunction(const IntToFuncMap &stateFunctionMap, llvm::Type* runtimeDataArgTy);

  // These functions return calls only in shaders in m_shaderMap.
  std::vector<llvm::CallInst*> getCallsInShadersToFunction(const std::string& funcName);
  std::vector<llvm::CallInst*> getCallsInShadersToFunctionWithPrefix(const std::string& funcNamePrefix);

};
