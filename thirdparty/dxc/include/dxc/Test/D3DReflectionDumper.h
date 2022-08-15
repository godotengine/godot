///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// D3DReflectionDumper.h                                                     //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Use this to dump D3D Reflection data for testing.                         //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "dxc/Support/Global.h"
#include "DumpContext.h"
#include "dxc/Support/WinIncludes.h"
#include <d3d12shader.h>

namespace hlsl {
namespace dump {

LPCSTR ToString(D3D_CBUFFER_TYPE CBType);
LPCSTR ToString(D3D_SHADER_INPUT_TYPE Type);
LPCSTR ToString(D3D_RESOURCE_RETURN_TYPE ReturnType);
LPCSTR ToString(D3D_SRV_DIMENSION Dimension);
LPCSTR ToString(D3D_PRIMITIVE_TOPOLOGY GSOutputTopology);
LPCSTR ToString(D3D_PRIMITIVE InputPrimitive);
LPCSTR ToString(D3D_TESSELLATOR_OUTPUT_PRIMITIVE HSOutputPrimitive);
LPCSTR ToString(D3D_TESSELLATOR_PARTITIONING HSPartitioning);
LPCSTR ToString(D3D_TESSELLATOR_DOMAIN TessellatorDomain);
LPCSTR ToString(D3D_SHADER_VARIABLE_CLASS Class);
LPCSTR ToString(D3D_SHADER_VARIABLE_TYPE Type);
LPCSTR ToString(D3D_SHADER_VARIABLE_FLAGS Flag);
LPCSTR ToString(D3D_SHADER_INPUT_FLAGS Flag);
LPCSTR ToString(D3D_SHADER_CBUFFER_FLAGS Flag);
LPCSTR ToString(D3D_PARAMETER_FLAGS Flag);
LPCSTR ToString(D3D_NAME Name);
LPCSTR ToString(D3D_REGISTER_COMPONENT_TYPE CompTy);
LPCSTR ToString(D3D_MIN_PRECISION MinPrec);
LPCSTR CompMaskToString(unsigned CompMask);

class D3DReflectionDumper : public DumpContext {
private:
  bool m_bCheckByName = false;
  const char *m_LastName = nullptr;
  void SetLastName(const char *Name = nullptr) { m_LastName = Name ? Name : "<nullptr>"; }

public:
  D3DReflectionDumper(std::ostream &outStream) : DumpContext(outStream) {}
  void SetCheckByName(bool bCheckByName) { m_bCheckByName = bCheckByName; }

  void DumpShaderVersion(UINT Version);
  void DumpDefaultValue(LPCVOID pDefaultValue, UINT Size);

  void Dump(D3D12_SHADER_TYPE_DESC &tyDesc);
  void Dump(D3D12_SHADER_VARIABLE_DESC &varDesc);
  void Dump(D3D12_SHADER_BUFFER_DESC &Desc);
  void Dump(D3D12_SHADER_INPUT_BIND_DESC &resDesc);
  void Dump(D3D12_SIGNATURE_PARAMETER_DESC &elDesc);
  void Dump(D3D12_SHADER_DESC &Desc);
  void Dump(D3D12_FUNCTION_DESC &Desc);
  void Dump(D3D12_LIBRARY_DESC &Desc);

  void Dump(ID3D12ShaderReflectionType *pType);
  void Dump(ID3D12ShaderReflectionVariable *pVar);

  void Dump(ID3D12ShaderReflectionConstantBuffer *pCBReflection);

  void Dump(ID3D12ShaderReflection *pShaderReflection);
  void Dump(ID3D12FunctionReflection *pFunctionReflection);
  void Dump(ID3D12LibraryReflection *pLibraryReflection);

};

} // namespace dump
} // namespace hlsl
