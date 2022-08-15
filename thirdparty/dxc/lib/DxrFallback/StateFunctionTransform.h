#pragma once

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

#include <map>
#include <string>
#include <vector>

namespace llvm
{
  class AllocaInst;
  class BasicBlock;
  class CallInst;
  class Function;
  class FunctionType;
  class Instruction;
  class Module;
  class raw_ostream;
  class ReturnInst;
  class StructType;
  class Type;
  class Value;
}

class LiveValues;

typedef std::vector<llvm::BasicBlock*>  BasicBlockVector;
typedef llvm::SetVector<llvm::Instruction*> InstructionSetVector;


//==============================================================================
// Transforms the given function into a number of state functions to be 
// used in a state machine. 
//
// State functions have the following signature: 
//    int (<RuntimeDataTy> runtimeData). 
// They take an runtime data argument with a given type used by the runtime and 
// return the state ID of the next state. If the function contains calls to other  
// candidate functions that are to be transformed into state functions, the 
// function is split into multiple substate functions at call sites and the calls 
// are replaced with continuations. For example candidate funcA() calling candidate 
// funcB():
//   void funcA(int param0)
//   {
//      // code moved to funcA_ss0()
//      int foo = 10;
//      ...
//
//      funcB(arg0, arg1); 
//
//      // code moved to funcA_ss1()
//      int bar = someFunc(foo);
//      
//   } 
// will be split into two substate functions, funcA_ss0() and funcA_ss1(). 
// funcA_ss0() pushes the stateID for funcA_ss1() onto the stack, and
// returns the state ID for the entry substate of funcB, funcB_ss0(). 
// A substate of funcB will eventually pop the stack and return the state ID
// for funcA_ss1(). funcA_ss1() in turn pops the stack to get the state ID
// placed there by its caller. 
//
// If candidate functions, like funcB(), have arguments they are moved to the stack.
// Any values that are live across continuations, like foo in this example,
// must also be saved to the stack before the continuation and restored before use. 
// Some values, like DXIL buffer handles should not be saved and must be 
// rematerialized after a continuation. The stack frame in a state function has
// the following layout:
//   
//   |               |
//   +---------------+  
//   | argN          |  
//   | ...           |   
//   | arg0          |  
//   | returnStateID | caller arg frame
//   +---------------+ <-- entry stack pointer
//   |               |
//   | saved values  |
//   |               |
//   +---------------+
//   | argN          |
//   | ...           |
//   | arg0          |
//   | returnStateID | callee arg frame
//   +---------------+ <-- stack frame pointer
//           |
//           V stack grows downward towards smaller addresses
//
// The return state ID is stored at the base of the argument frame, followed by
// function arguments, if any. The saved values follow the argument frame. Instead
// of adjusting the size of the stack frame for the saved values and argument
// frames of each continuation a single allocation is made with enough space to
// accommodate all continuations in the function.
//
// Several placeholder functions are used during the process of the state function
// transform to break dependency cycles. A placeholder for the runtime data pointer
// is used to allocate the stack frame before the function signature is changed
// and the pointer parameter is created. The stack frame is also allocated before
// its size has been determined, so a placeholder is used. The state IDs corresponding
// to function entry substates may also not be known before the transform has been 
// run on all the candidate functions. Therefore a placeholder is used for state 
// IDs as well. These are replaced by calling StateFunctionTransform::finalizeStateIds()
// after all the candidate functions have been transformed.
//
// If the intrinsic Internal_CallIndirect(int stateId) appears in the body of
// the function then it is treated as a continuation with a transition to the
// specified stateId.
//
// When an attribute size is specified, space is allocated on the stack frame for
// committed/pending attributes, as well as the previous offsets for the committed/
// pending attributes. The attribute size should be set if the 
// function is TraceRay(). The payload offset needs to be set by the caller. The 
// stack frame for TraceRay() has the following layout:
//
//   |                         |
//   +-------------------------+ 
//   |                         |
//   | TraceRay() args         |
//   |                         |
//   +-------------------------+
//   | returnStateID           | caller arg frame
//   +-------------------------+ <-- entry stack offset
//   | old committed attr offs |
//   | old pending attr offset |
//   +-------------------------+ 
//   |                         |
//   | committed attributes    |
//   |                         |
//   +-------------------------+ <-- new committed attribute offset
//   |                         |
//   | pending attributes      |
//   |                         |
//   +-------------------------+ <-- new pending attribute offset
//   |                         |
//   | saved values            |
//   |                         |
//   +-------------------------+
//   | argN                    |
//   | ...                     |
//   | arg0                    |
//   | returnStateID           | callee arg frame
//   +-------------------------+ <-- stack frame offset
//      
// The arguments to some functions (e.g. closesthit, anyhit, and miss shaders)
// come from the payload or attributes. The positions of these arguments can be 
// specified to SFT, which will redirect the defs from the args to corresponding
// values on the stack.
//
// The following runtime (LLVM) functions are used by SFT (all sizes and offsets
// are in terms of ints):
//   void stackFramePush(<RuntimeDataTy> runtimeData, i32 size)
//   void stackFramePop(<RuntimeDataTy> runtimeData, i32 size)
//
//   i32 stackFrameOffset(<RuntimeDataTy> runtimeData)
//   i32 payloadOffset(<RuntimeDataTy> runtimeData) 
//   i32 committedAttrOffset(<RuntimeDataTy> runtimeData)
//   i32 pendingAttrOffset(<RuntimeDataTy> runtimeData)
//
//   i32* stackIntPtr(<RuntimeDataTy> runtimeData, i32 baseOffset, i32 offset)
//   
// Called before/after stackFramePush()/stackFramePop():
//   void traceFramePush(<RuntimeDataTy> runtimeData, i32 attrSize) 
//   void traceFramePop(<RuntimeDataTy> runtimeData)               

class StateFunctionTransform
{
public:
  enum ParameterSemanticType
  {
    PST_NONE = 0,
    PST_PAYLOAD,
    PST_ATTRIBUTE,

    PST_COUNT
  };

  // func is the function to be transformed. candidateFuncNames is a list of all 
  // functions that which have been or will be transformed to state functions, 
  // including func. The runtimeDataArgTy is the type to use for the first argument
  // in state functions.
  StateFunctionTransform(llvm::Function* func, const std::vector<std::string>& candidateFuncNames, llvm::Type* runtimeDataArgTy);

  // Optional parameters to be specified before run()
  void setAttributeSize(int sizeInBytes); // needed for TraceRay()
  void setParameterInfo(const std::vector<ParameterSemanticType>& paramTypes, bool useCommittedAttr = true);
  void setResourceGlobals(const std::set<llvm::Value*>& resources);

  static llvm::Function* createDummyRuntimeDataArgFunc(llvm::Module* M, llvm::Type* runtimeDataArgTy);

  // Generates state functions from func into the same module. The original function
  // is left only as a declaration.
  void run(std::vector<llvm::Function*>& stateFunctions, _Out_ unsigned int &shaderStackSize);

  // candidateFuncEntryStateIds corresponding to the candidateFuncNames passed to
  // the constructor. stateIDs are computed as candidateFuncEntryStateIds[functionIdx]
  // + substateIdx, where functionIdx and substateIdx come from the arguments to
  // the placeholder stateID function.
  static void finalizeStateIds(llvm::Module* module, const std::vector<int>& candidateFuncEntryStateIds);

  // Outputs detailed diagnostic information if set to true.
  void setVerbose(bool val);

  void setDumpFilename(const std::string& dumpFilename);


private:
  // Function to transform
  llvm::Function* m_function = nullptr;

  // Name of the function to transform
  std::string m_functionName;

  // Index of the function to transform in m_candidateFuncNames
  int m_functionIdx = 0;

  // cadidateFuncNames is a list of all functions that which have been or will 
  // be transformed to state functions. Used to create function index used
  // by the stateID placeholder function.
  const std::vector<std::string>& m_candidateFuncNames;

  llvm::Type* m_runtimeDataArgTy = nullptr;
  llvm::Value* m_runtimeDataArg = nullptr;     // set in init() and changeFunctionSignature()
  llvm::Value* m_stackFrameSizeVal = nullptr;  // set in init() and preserveLiveValuesAcrossCallsites()

  int m_attributeSizeInBytes = -1;
  std::vector<ParameterSemanticType> m_paramTypes;
  bool m_useCommittedAttr = false;
  const std::set<llvm::Value*>* m_resources;

  std::vector<llvm::CallInst*> m_callSites;
  std::vector<int> m_callSiteFunctionIdx;
  std::vector<llvm::CallInst*> m_movePayloadToStackCalls;
  std::vector<llvm::CallInst*> m_setPendingAttrCalls;
  std::vector<llvm::ReturnInst*> m_returns;

  bool m_verbose = false;
  std::string m_dumpFilename;
  unsigned int m_dumpId = 0;

  llvm::Function* m_stackIntPtrFunc = nullptr;

  llvm::CallInst* m_stackFramePush = nullptr;
  llvm::CallInst* m_stackFrameOffset = nullptr;
  llvm::CallInst* m_payloadOffset = nullptr;          // Offset at beginning of function
  llvm::CallInst* m_committedAttrOffset = nullptr;    // Offset at beginning of function
  llvm::CallInst* m_pendingAttrOffset = nullptr;      // Offset at beginning of function

  // Placeholder function taking constant values functionIdx and substate. 
  // These are later translated to a stateId by finalizeStateIds().
  llvm::Function* m_dummyStateIdFunc = nullptr;

  int m_maxCallerArgFrameSizeInBytes = 0;
  int m_traceFrameSizeInBytes = 0;

  // Functions used to abstract stack operations. These make intermediate stages
  // in the transform a little bit cleaner. 
  std::map<llvm::FunctionType*, llvm::Function*> m_stackStoreFuncs;
  std::map<llvm::FunctionType*, llvm::Function*> m_stackLoadFuncs;
  std::map<llvm::FunctionType*, llvm::Function*> m_stackPtrFuncs;

  // Main stages of the transformation 
  void init();
  void findCallSitesIntrinsicsAndReturns();
  void changeCallingConvention();
  void preserveLiveValuesAcrossCallsites(_Out_ unsigned int &shaderStackSize);
  void createSubstateFunctions(std::vector<llvm::Function*>& stateFunctions);
  void lowerStackFuncs();

  llvm::Value* getDummyStateId(int functionIdx, int substate, llvm::Instruction* insertBefore);

  void allocateStackFrame();
  void allocateTraceFrame();
  void createArgFrames();
  void changeFunctionSignature();

  void createStackStore(llvm::Value* baseOffset, llvm::Value* val, int offsetInBytes, llvm::Instruction* insertBefore);
  llvm::Instruction* createStackLoad(llvm::Value* baseOffset, llvm::Value* val, int offsetInBytes, llvm::Instruction* insertBefore);
  llvm::Instruction* createStackPtr(llvm::Value* baseOffset, llvm::Value* val, int offsetInBytes, llvm::Instruction* insertBefore);
  llvm::Instruction* createStackPtr(llvm::Value* baseOffset, llvm::Type* valTy, llvm::Value* intIndex, llvm::Instruction* insertBefore);
  void rewriteDummyStackSize(uint64_t frameSizeInBytes);

  BasicBlockVector replaceCallSites();
  llvm::Function* split(llvm::Function* baseFunc, llvm::BasicBlock* subStateEntryBlock, int substateIndex);

  void flattenGepsOnValue(llvm::Value* val, llvm::Value* baseOffset, llvm::Value* offset);
  void scalarizeVectorStackAccess(llvm::Instruction* vecPtr, llvm::Value* baseOffset, llvm::Value* offsetVal);

  // Diagnostic printing functions
  llvm::raw_ostream& getOutputStream(const std::string functionName, const std::string& suffix, unsigned int dumpId);
  void printFunction(const llvm::Function* function, const std::string& suffix, unsigned int dumpId);
  void printFunction(const std::string& suffix);
  void printFunctions(const std::vector<llvm::Function*>& funcs, const char* suffix);
  void printModule(const llvm::Module* module, const std::string& suffix);
  void printSet(const InstructionSetVector& vals, const char* msg = nullptr);
};
