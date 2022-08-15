#pragma once

#include "llvm/IR/Module.h"

#include <string>
#include <vector>

//==============================================================================
// Simplifies the creation of functions.
//
// To create a function 'void foo( userType, i32, float* )' use the following
// code:
//   FunctionBuilder(module, "foo").voidTy().type(userType).i32().floatPtr().build()
//
// The first type specified is the return type.
class FunctionBuilder
{
public:
  FunctionBuilder(llvm::Module* mod, const std::string& name)
    : m_context(mod->getContext())
    , m_module(mod)
    , m_name(name)
  {}

  FunctionBuilder& voidTy()
  {
    m_argNames.push_back("");
    m_types.push_back(llvm::Type::getVoidTy(m_context));
    return *this;
  }
  FunctionBuilder& floatTy(const std::string& argName = "")
  {
    m_argNames.push_back(argName);
    m_types.push_back(llvm::Type::getFloatTy(m_context));
    return *this;
  }
  FunctionBuilder& floatPtr(const std::string& argName = "")
  {
    m_argNames.push_back(argName);
    m_types.push_back(llvm::Type::getFloatPtrTy(m_context));
    return *this;
  }
  FunctionBuilder& doubleTy(const std::string& argName = "")
  {
    m_argNames.push_back(argName);
    m_types.push_back(llvm::Type::getDoubleTy(m_context));
    return *this;
  }
  FunctionBuilder& doublePtr(const std::string& argName = "")
  {
    m_argNames.push_back(argName);
    m_types.push_back(llvm::Type::getDoublePtrTy(m_context));
    return *this;
  }
  FunctionBuilder& i32(const std::string& argName = "")
  {
    m_argNames.push_back(argName);
    m_types.push_back(llvm::Type::getInt32Ty(m_context));
    return *this;
  }
  FunctionBuilder& i32Ptr(const std::string& argName = "")
  {
    m_argNames.push_back(argName);
    m_types.push_back(llvm::Type::getInt32PtrTy(m_context));
    return *this;
  }
  FunctionBuilder& i16(const std::string& argName = "")
  {
    m_argNames.push_back(argName);
    m_types.push_back(llvm::Type::getInt16Ty(m_context));
    return *this;
  }
  FunctionBuilder& i16Ptr(const std::string& argName = "")
  {
    m_argNames.push_back(argName);
    m_types.push_back(llvm::Type::getInt16PtrTy(m_context));
    return *this;
  }
  FunctionBuilder& i8(const std::string& argName = "")
  {
    m_argNames.push_back(argName);
    m_types.push_back(llvm::Type::getInt8Ty(m_context));
    return *this;
  }
  FunctionBuilder& i8Ptr(const std::string& argName = "")
  {
    m_argNames.push_back(argName);
    m_types.push_back(llvm::Type::getInt8PtrTy(m_context));
    return *this;
  }
  FunctionBuilder& i1(const std::string& argName = "")
  {
    m_argNames.push_back(argName);
    m_types.push_back(llvm::Type::getInt1Ty(m_context));
    return *this;
  }
  FunctionBuilder& i1Ptr(const std::string& argName = "")
  {
    m_argNames.push_back(argName);
    m_types.push_back(llvm::Type::getInt1PtrTy(m_context));
    return *this;
  }

  FunctionBuilder& type(llvm::Type* ty, const std::string& argName = "")
  {
    m_argNames.push_back(argName);
    m_types.push_back(ty);
    return *this;
  }
  FunctionBuilder& types(const std::vector<llvm::Type*>& ty, const std::vector<std::string>& argNames)
  {
    if (argNames.empty())
      for (size_t i = 0; i < ty.size(); ++i)
        m_argNames.push_back("");
    m_types.insert(m_types.end(), ty.begin(), ty.end());
    return *this;
  }

  llvm::Function* build()
  {
    using namespace llvm;

    Type*        retTy = m_types[0];
    AttributeSet attributes;
    Type**       argsBegin = (&m_types[0]) + 1;
    Type**       argsEnd = argsBegin + m_types.size() - 1;
    Constant*    funcC =
      m_module->getOrInsertFunction(m_name, FunctionType::get(retTy, ArrayRef<Type*>(argsBegin, argsEnd), false), attributes);
    Function* func = cast<Function>(funcC);

    std::string* argNamePtr = m_argNames.data() + 1;
    for (auto& arg : func->args())
      arg.setName(*argNamePtr++);

    return func;
  }

private:
  llvm::LLVMContext&       m_context;
  llvm::Module*            m_module = nullptr;
  std::string              m_name;
  std::vector<std::string> m_argNames;
  std::vector<llvm::Type*> m_types;

  // forbidden
  FunctionBuilder();
  FunctionBuilder(const FunctionBuilder&);
};
