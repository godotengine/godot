///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilTypeSystem.h                                                          //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// DXIL extension to LLVM type system.                                       //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/MapVector.h"
#include "dxc/DXIL/DxilConstants.h"
#include "dxc/DXIL/DxilCompType.h"
#include "dxc/DXIL/DxilInterpolationMode.h"

#include <memory>
#include <string>
#include <vector>

namespace llvm {
class LLVMContext;
class Module;
class Function;
class MDNode;
class Type;
class StructType;
}


namespace hlsl {

enum class MatrixOrientation { Undefined = 0, RowMajor, ColumnMajor, LastEntry };

struct DxilMatrixAnnotation {
  unsigned Rows;
  unsigned Cols;
  MatrixOrientation Orientation;

  DxilMatrixAnnotation();
};

/// Use this class to represent type annotation for structure field.
class DxilFieldAnnotation {
public:
  DxilFieldAnnotation();
  
  bool IsPrecise() const;
  void SetPrecise(bool b = true);

  bool HasMatrixAnnotation() const;
  const DxilMatrixAnnotation &GetMatrixAnnotation() const;
  void SetMatrixAnnotation(const DxilMatrixAnnotation &MA);

  bool HasResourceAttribute() const;
  llvm::MDNode *GetResourceAttribute() const;
  void SetResourceAttribute(llvm::MDNode *MD);

  bool HasCBufferOffset() const;
  unsigned GetCBufferOffset() const;
  void SetCBufferOffset(unsigned Offset);

  bool HasCompType() const;
  const CompType &GetCompType() const;
  void SetCompType(CompType::Kind kind);

  bool HasSemanticString() const;
  const std::string &GetSemanticString() const;
  llvm::StringRef GetSemanticStringRef() const;
  void SetSemanticString(const std::string &SemString);

  bool HasInterpolationMode() const;
  const InterpolationMode &GetInterpolationMode() const;
  void SetInterpolationMode(const InterpolationMode &IM);

  bool HasFieldName() const;
  const std::string &GetFieldName() const;
  void SetFieldName(const std::string &FieldName);

  bool IsCBVarUsed() const;
  void SetCBVarUsed(bool used);

private:
  bool m_bPrecise;
  CompType m_CompType;
  DxilMatrixAnnotation m_Matrix;
  llvm::MDNode *m_ResourceAttribute;
  unsigned m_CBufferOffset;
  std::string m_Semantic;
  InterpolationMode m_InterpMode;
  std::string m_FieldName;
  bool m_bCBufferVarUsed; // true if this field represents a top level variable in CB structure, and it is used.
};

class DxilTemplateArgAnnotation {
public:
  DxilTemplateArgAnnotation();

  bool IsType() const;
  const llvm::Type *GetType() const;
  void SetType(const llvm::Type *pType);

  bool IsIntegral() const;
  int64_t GetIntegral() const;
  void SetIntegral(int64_t i64);

private:
  const llvm::Type *m_Type;
  int64_t m_Integral;
};

/// Use this class to represent LLVM structure annotation.
class DxilStructAnnotation {
  friend class DxilTypeSystem;

public:
  unsigned GetNumFields() const;
  DxilFieldAnnotation &GetFieldAnnotation(unsigned FieldIdx);
  const DxilFieldAnnotation &GetFieldAnnotation(unsigned FieldIdx) const;
  const llvm::StructType *GetStructType() const;
  void SetStructType(const llvm::StructType *Ty);
  unsigned GetCBufferSize() const;
  void SetCBufferSize(unsigned size);
  void MarkEmptyStruct();
  bool IsEmptyStruct();
  // Since resources don't take real space, IsEmptyBesidesResources
  // determines if the structure is empty or contains only resources.
  bool IsEmptyBesidesResources();
  bool ContainsResources() const;

  // For template args, GetNumTemplateArgs() will return 0 if not a template
  unsigned GetNumTemplateArgs() const;
  void SetNumTemplateArgs(unsigned count);
  DxilTemplateArgAnnotation &GetTemplateArgAnnotation(unsigned argIdx);
  const DxilTemplateArgAnnotation &GetTemplateArgAnnotation(unsigned argIdx) const;

private:
  const llvm::StructType *m_pStructType = nullptr;
  std::vector<DxilFieldAnnotation> m_FieldAnnotations;
  unsigned m_CBufferSize = 0;  // The size of struct if inside constant buffer.
  std::vector<DxilTemplateArgAnnotation> m_TemplateAnnotations;

  // m_ResourcesContained property not stored to metadata
  void SetContainsResources();
  // HasResources::Only will be set on MarkEmptyStruct() when HasResources::True
  enum class HasResources { True, False, Only } m_ResourcesContained = HasResources::False;
};


/// Use this class to represent type annotation for DXR payload field.
class DxilPayloadFieldAnnotation {
public:

  static unsigned GetBitOffsetForShaderStage(DXIL::PayloadAccessShaderStage shaderStage);

  DxilPayloadFieldAnnotation() = default;

  bool HasCompType() const;
  const CompType &GetCompType() const;
  void SetCompType(CompType::Kind kind);

  uint32_t GetPayloadFieldQualifierMask() const;
  void SetPayloadFieldQualifierMask(uint32_t fieldBitmask);
  void AddPayloadFieldQualifier(DXIL::PayloadAccessShaderStage shaderStage, DXIL::PayloadAccessQualifier qualifier);
  DXIL::PayloadAccessQualifier GetPayloadFieldQualifier(DXIL::PayloadAccessShaderStage shaderStage) const;
  bool HasAnnotations() const;

private:
  CompType m_CompType;
  unsigned m_bitmask = 0;
};

/// Use this class to represent DXR payload structures.
class DxilPayloadAnnotation {
  friend class DxilTypeSystem;

public:
  unsigned GetNumFields() const;
  DxilPayloadFieldAnnotation &GetFieldAnnotation(unsigned FieldIdx);
  const DxilPayloadFieldAnnotation &GetFieldAnnotation(unsigned FieldIdx) const;
  const llvm::StructType *GetStructType() const;
  void SetStructType(const llvm::StructType *Ty);

private:
  const llvm::StructType *m_pStructType;
  std::vector<DxilPayloadFieldAnnotation> m_FieldAnnotations;
};


enum class DxilParamInputQual {
  In,
  Out,
  Inout,
  InputPatch,
  OutputPatch,
  OutStream0,
  OutStream1,
  OutStream2,
  OutStream3,
  InputPrimitive,
  OutIndices,
  OutVertices,
  OutPrimitives,
  InPayload,
};

/// Use this class to represent type annotation for function parameter.
class DxilParameterAnnotation : public DxilFieldAnnotation {
public:
  DxilParameterAnnotation();
  DxilParamInputQual GetParamInputQual() const;
  void SetParamInputQual(DxilParamInputQual qual);
  const std::vector<unsigned> &GetSemanticIndexVec() const;
  void SetSemanticIndexVec(const std::vector<unsigned> &Vec);
  void AppendSemanticIndex(unsigned SemIdx);
private:
  DxilParamInputQual m_inputQual;
  std::vector<unsigned> m_semanticIndex;
};

/// Use this class to represent LLVM function annotation.
class DxilFunctionAnnotation {
  friend class DxilTypeSystem;

public:
  unsigned GetNumParameters() const;
  DxilParameterAnnotation &GetParameterAnnotation(unsigned ParamIdx);
  const DxilParameterAnnotation &GetParameterAnnotation(unsigned ParamIdx) const;
  const llvm::Function *GetFunction() const;
  DxilParameterAnnotation &GetRetTypeAnnotation();
  const DxilParameterAnnotation &GetRetTypeAnnotation() const;

  bool ContainsResourceArgs() { return m_bContainsResourceArgs; }

private:
  const llvm::Function *m_pFunction;
  std::vector<DxilParameterAnnotation> m_parameterAnnotations;
  DxilParameterAnnotation m_retTypeAnnotation;

  // m_bContainsResourceArgs property not stored to metadata
  void SetContainsResourceArgs() { m_bContainsResourceArgs = true; }
  bool m_bContainsResourceArgs = false;
};

/// Use this class to represent structure type annotations in HL and DXIL.
class DxilTypeSystem {
public:
  using StructAnnotationMap = llvm::MapVector<const llvm::StructType *, std::unique_ptr<DxilStructAnnotation> >;
  using PayloadAnnotationMap = llvm::MapVector<const llvm::StructType *, std::unique_ptr<DxilPayloadAnnotation> >;
  using FunctionAnnotationMap = llvm::MapVector<const llvm::Function *, std::unique_ptr<DxilFunctionAnnotation> >;

  DxilTypeSystem(llvm::Module *pModule);

  DxilStructAnnotation *AddStructAnnotation(const llvm::StructType *pStructType, unsigned numTemplateArgs = 0);
  void FinishStructAnnotation(DxilStructAnnotation &SA);
  DxilStructAnnotation *GetStructAnnotation(const llvm::StructType *pStructType);
  const DxilStructAnnotation *GetStructAnnotation(const llvm::StructType *pStructType) const;
  void EraseStructAnnotation(const llvm::StructType *pStructType);
  void EraseUnusedStructAnnotations();

  StructAnnotationMap &GetStructAnnotationMap();
  const StructAnnotationMap &GetStructAnnotationMap() const;

  DxilPayloadAnnotation *AddPayloadAnnotation(const llvm::StructType *pStructType);
  DxilPayloadAnnotation *GetPayloadAnnotation(const llvm::StructType *pStructType);
  const DxilPayloadAnnotation *GetPayloadAnnotation(const llvm::StructType *pStructType) const;
  void ErasePayloadAnnotation(const llvm::StructType *pStructType);

  PayloadAnnotationMap &GetPayloadAnnotationMap();
  const PayloadAnnotationMap &GetPayloadAnnotationMap() const;

  DxilFunctionAnnotation *AddFunctionAnnotation(const llvm::Function *pFunction);
  void FinishFunctionAnnotation(DxilFunctionAnnotation &FA);
  DxilFunctionAnnotation *GetFunctionAnnotation(const llvm::Function *pFunction);
  const DxilFunctionAnnotation *GetFunctionAnnotation(const llvm::Function *pFunction) const;
  void EraseFunctionAnnotation(const llvm::Function *pFunction);

  FunctionAnnotationMap &GetFunctionAnnotationMap();

  // Utility methods to create stand-alone SNORM and UNORM.
  // We may want to move them to a more centralized place for most utilities.
  llvm::StructType *GetSNormF32Type(unsigned NumComps);
  llvm::StructType *GetUNormF32Type(unsigned NumComps);

  // Methods to copy annotation from another DxilTypeSystem.
  void CopyTypeAnnotation(const llvm::Type *Ty, const DxilTypeSystem &src);
  void CopyFunctionAnnotation(const llvm::Function *pDstFunction,
                              const llvm::Function *pSrcFunction,
                              const DxilTypeSystem &src);

  bool UseMinPrecision();
  void SetMinPrecision(bool bMinPrecision);

  // Determines whether type is a resource or contains a resource
  bool IsResourceContained(llvm::Type *Ty);

private:
  llvm::Module *m_pModule;
  StructAnnotationMap m_StructAnnotations;
  PayloadAnnotationMap m_PayloadAnnotations;
  FunctionAnnotationMap m_FunctionAnnotations;

  DXIL::LowPrecisionMode m_LowPrecisionMode;

  llvm::StructType *GetNormFloatType(CompType CT, unsigned NumComps);
};

DXIL::SigPointKind SigPointFromInputQual(DxilParamInputQual Q, DXIL::ShaderKind SK, bool isPC);

void RemapObsoleteSemantic(DxilParameterAnnotation &paramInfo, DXIL::SigPointKind sigPoint, llvm::LLVMContext &Context);

class DxilStructTypeIterator
    : public std::iterator<std::input_iterator_tag,
                           std::pair<llvm::Type *, DxilFieldAnnotation *>> {
private:
  llvm::StructType *STy;
  DxilStructAnnotation *SAnnotation;
  unsigned index;

public:
  DxilStructTypeIterator(llvm::StructType *sTy,
                         DxilStructAnnotation *sAnnotation, unsigned idx = 0);
  // prefix
  DxilStructTypeIterator &operator++();
  // postfix
  DxilStructTypeIterator operator++(int);

  bool operator==(DxilStructTypeIterator iter);
  bool operator!=(DxilStructTypeIterator iter);
  std::pair<llvm::Type *, DxilFieldAnnotation *> operator*();
};

DxilStructTypeIterator begin(llvm::StructType *STy,
                             DxilStructAnnotation *SAnno);
DxilStructTypeIterator end(llvm::StructType *STy, DxilStructAnnotation *SAnno);

} // namespace hlsl
