///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLSLExtensionsCodegenHelper.h                                             //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Codegen support for hlsl extensions.                                      //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include "dxc/DXIL/DxilOperations.h"
#include <vector>
#include <string>

namespace clang {
class CodeGenOptions;
}

namespace llvm {
class CallInst;
class Value;
class Module;
}

namespace hlsl {

// Provide DXIL codegen support for private HLSL extensions.
// The HLSL extension mechanism has hooks for two cases:
//
//  1. You can mark certain defines as "semantic" defines which
//     will be preserved as metadata in the final DXIL.
//  2. You can add new HLSL intrinsic functions.
//  3. You can read a root signature from a custom define.
//
// This class provides an interface for generating the DXIL bitcode
// needed for the types of extensions above.
//  
class HLSLExtensionsCodegenHelper {
public:
  // Used to indicate a semantic define was used incorrectly.
  // Since semantic defines have semantic meaning it is possible
  // that a programmer can use them incorrectly. This class provides
  // a way to signal the error to the programmer. Semantic define
  // errors will be propagated as errors to the clang frontend.
  class SemanticDefineError {
  public:
    enum class Level { Warning, Error };
    SemanticDefineError(unsigned location, Level level, const std::string &message)
    :  m_location(location)
    ,  m_level(level)
    ,  m_message(message)
    { }

    unsigned Location() const { return m_location; }
    bool IsWarning() const { return m_level == Level::Warning; }
    const std::string &Message() const { return m_message; }

  private:
    unsigned m_location; // Use an encoded clang::SourceLocation to avoid a clang include dependency.
    Level m_level;
    std::string m_message;
  };
  typedef std::vector<SemanticDefineError> SemanticDefineErrorList;

  // Write semantic defines as metadata in the module.
  virtual void WriteSemanticDefines(llvm::Module *M) = 0;
  virtual void UpdateCodeGenOptions(clang::CodeGenOptions &CGO) = 0;
  // Query the named option enable
  // Needed because semantic defines may have set it since options were copied
  virtual bool IsOptionEnabled(std::string option) = 0;

  // Get the name to use for the dxil intrinsic function.
  virtual std::string GetIntrinsicName(unsigned opcode) = 0;

  // Get the dxil opcode the extension should use when lowering with
  // dxil lowering strategy.
  //
  // Returns true if the opcode was successfully mapped to a dxil opcode.
  virtual bool GetDxilOpcode(unsigned opcode, OP::OpCode &dxilOpcode) = 0;

  // Struct to hold a root signature that is read from a define.
  struct CustomRootSignature {
    std::string RootSignature;
    unsigned  EncodedSourceLocation;
    enum Status { NOT_FOUND = 0, FOUND };
  };

  // Get custom defined root signature.
  virtual CustomRootSignature::Status GetCustomRootSignature(CustomRootSignature *out) = 0;

  // Virtual destructor.
  virtual ~HLSLExtensionsCodegenHelper() {};
};
}
