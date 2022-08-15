///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxcPixEntrypoints.cpp                                                     //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Defines all of the entrypoints for DXC's PIX interfaces for dealing with  //
// debug info. These entrypoints are responsible for setting up a common,    //
// sane environment -- e.g., exceptions are caught and returned as an        //
// HRESULT -- deferring to the real implementations defined elsewhere in     //
// this library.                                                             //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/Support/WinIncludes.h"

#include "dxc/dxcapi.h"
#include "dxc/dxcpix.h"
#include "DxilDiaSession.h"

#include "dxc/Support/Global.h"
#include "dxc/Support/microcom.h"
#include "dxc/Support/FileIOHelper.h"
#include "llvm/Support/MSFileSystem.h"
#include "llvm/Support/FileSystem.h"

#include "DxcPixBase.h"
#include "DxcPixDxilDebugInfo.h"
#include "DxcPixCompilationInfo.h"

#include <functional>

namespace dxil_debug_info
{
namespace entrypoints
{
// OutParam/OutParamImpl provides a mechanism that entrypoints
// use to tag method arguments as OutParams. OutParams are
// automatically zero-initialized.
template <typename T>
class OutParamImpl
{
public:
  OutParamImpl(T *V) : m_V(V)
  {
    if (m_V != nullptr)
    {
      *m_V = T();
    }
  }

  operator T *() const
  {
    return m_V;
  }

private:
  T *m_V;
};

template <typename T>
OutParamImpl<T> OutParam(T *V)
{
  return OutParamImpl<T>(V);
}


// InParam/InParamImpl provides a mechanism that entrypoints
// use to tag method arguments as InParams. InParams are
// not zero-initialized.
template <typename T>
struct InParamImpl
{
public:
  InParamImpl(const T *V) : m_V(V)
  {
  }

  operator const T *() const
  {
    return m_V;
  }

private:
  const T *m_V;
};

template <typename T>
InParamImpl<T> InParam(T *V)
{
  return InParamImpl<T>(V);
}


// ThisPtr/ThisPtrImpl provides a mechanism that entrypoints
// use to tag method arguments as c++'s this. This values
// are not checked/initialized.
template <typename T>
struct ThisPtrImpl
{
public:
  ThisPtrImpl(T *V) : m_V(V)
  {
  }

  operator T *() const
  {
    return m_V;
  }

private:
  T *m_V;
};

template <typename T>
ThisPtrImpl<T> ThisPtr(T *V)
{
  return ThisPtrImpl<T>(V);
}


// CheckNotNull/CheckNotNullImpl provide a mechanism that entrypoints
// can use for automatic parameter validation. They will throw an
// exception if the given parameter is null.
template <typename T>
class CheckNotNullImpl;

template <typename T>
class CheckNotNullImpl<OutParamImpl<T>>
{
public:
  explicit CheckNotNullImpl(OutParamImpl<T> V) : m_V(V)
  {
  }

  operator T *() const
  {
    if (m_V == nullptr)
    {
      throw hlsl::Exception(E_POINTER);
    }
    return m_V;
  }

private:
  T *m_V;
};

template <typename T>
class CheckNotNullImpl<InParamImpl<T>>
{
public:
  explicit CheckNotNullImpl(InParamImpl<T> V) : m_V(V)
  {
  }

  operator T *() const
  {
    if (m_V == nullptr)
    {
      throw hlsl::Exception(E_POINTER);
    }
    return m_V;
  }

private:
  T *m_V;
};

template <typename T>
CheckNotNullImpl<T> CheckNotNull(T V)
{
  return CheckNotNullImpl<T>(V);
}

// WrapOutParams will wrap any OutParams<T> that the
// entrypoints provide to SetupAndRun -- which is essentially
// the method that runs the actual methods.
void WrapOutParams(IMalloc *)
{
}

template <typename T> struct EntrypointWrapper;

// WrapOutParams' specialization that detects OutParams that
// inherit from IUnknown. Any OutParams inheriting from IUnknown
// should be wrapped by one of the classes in this file so that
// user calls will be safely run.
template <
    typename T,
    typename =
        typename std::enable_if<std::is_base_of<IUnknown, T>::value>::type,
    typename... O>
void WrapOutParams(
    IMalloc* M,
    CheckNotNullImpl<OutParamImpl<T *>> ppOut,
    O... Others
)
{
  if (*ppOut)
  {
    NewDxcPixDxilDebugInfoObjectOrThrow<typename EntrypointWrapper<T*>::type>(
        (T**)ppOut,
        M,
        *ppOut);
  }

  WrapOutParams(M, Others...);
}

template <typename T, typename... O>
void WrapOutParams(IMalloc *M, T, O... Others)
{
  WrapOutParams(M, Others...);
}

// DEFINE_ENTRYPOINT_WRAPPER_TRAIT is a helper macro that every entrypoint
// should use in order to define the EntrypointWrapper traits class for
// the interface it implements.
#define DEFINE_ENTRYPOINT_WRAPPER_TRAIT(IInterface)                         \
template <> struct EntrypointWrapper<IInterface *>                          \
{                                                                           \
 using type = IInterface ## Entrypoint;                                     \
};

// IsValidArgType exposes a static method named check(), which returns
// true if <T> is a valid type for a method argument, and false otherwise.
// This is trying to ensure all pointers are checked, and all out params
// are default-initialized.
template <typename T>
struct IsValidArgType
{
    static void check() {}
};

template <typename T>
struct IsValidArgType<T *>
{
    static void check()
    {
        static_assert(
            false,
            "Pointer arguments should be checked and wrapped"
            " with InParam/OutParam, or marked with ThisPtr()");
    }
};


template <typename T>
struct IsValidArgType<InParamImpl<T>>
{
    static void check()
    {
        static_assert(
            false,
            "InParams should be checked for nullptrs");
    }
};

template <typename T>
struct IsValidArgType<OutParamImpl<T>>
{
    static void check()
    {
        static_assert(
            false,
            "InParams should be checked for nullptrs");
    }
};

void EnsureAllPointersAreChecked()
{
}

template <typename T, typename... O>
void EnsureAllPointersAreChecked(T, O... o)
{
  IsValidArgType<T>::check();
  EnsureAllPointersAreChecked(o...);
}

// SetupAndRun is the function that sets up the environment
// in which all of the user requests to the DxcPix library
// run under.
template<typename H, typename... A>
HRESULT SetupAndRun(
    IMalloc* M,
    H Handler,
    A... Args
)
{
  DxcThreadMalloc TM(M);

  HRESULT hr = E_FAIL;
  try
  {
    ::llvm::sys::fs::MSFileSystem* msfPtr;
    IFT(CreateMSFileSystemForDisk(&msfPtr));
    std::unique_ptr<::llvm::sys::fs::MSFileSystem> msf(msfPtr);

    ::llvm::sys::fs::AutoPerThreadSystem pts(msf.get());
    IFTLLVM(pts.error_code());

    EnsureAllPointersAreChecked(Args...);
    hr = Handler(Args...);
    WrapOutParams(M, Args...);
  }
  catch (const hlsl::Exception &e)
  {
      hr = e.hr;
  }
  catch (const std::bad_alloc &)
  {
      hr = E_OUTOFMEMORY;
  }
  catch (const std::exception &)
  {
      hr = E_FAIL;
  }

  return hr;
}

HRESULT CreateEntrypointWrapper(
    IMalloc *pMalloc,
    IUnknown *pReal,
    REFIID iid,
    void **ppvObject);

// Entrypoint is the base class for all entrypoints, providing
// the default QueryInterface implementation, as well as a
// more convenient way of calling SetupAndRun.
template <typename I>
class Entrypoint : public I
{
protected:
  using IInterface = I;

  Entrypoint(
      IMalloc *pMalloc,
      IInterface *pI
  ) : m_pMalloc(pMalloc)
    , m_pReal(pI)
  {
  }

  DXC_MICROCOM_TM_REF_FIELDS();
  CComPtr<IInterface> m_pReal;

  template <typename F, typename... A>
  HRESULT InvokeOnReal(F pFn, A... Args)
  {
    return ::SetupAndRun(m_pMalloc, std::mem_fn(pFn), m_pReal, Args...);
  }

public:
  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL();

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** ppvObject) override final
  {
    return ::SetupAndRun(
        m_pMalloc,
        std::mem_fn(&Entrypoint<IInterface>::QueryInterfaceImpl),
        ThisPtr(this),
        iid,
        CheckNotNull(OutParam(ppvObject)));
  }

  HRESULT STDMETHODCALLTYPE QueryInterfaceImpl(REFIID iid, void** ppvObject)
  {
    // Special-casing so we don't need to create a new wrapper.
    if (iid == __uuidof(IInterface) ||
        iid == __uuidof(IUnknown)   ||
        iid == __uuidof(INoMarshal))
    {
      this->AddRef();
      *ppvObject = this;
      return S_OK;
    }

    CComPtr<IUnknown> RealQI;
    IFR(m_pReal->QueryInterface(iid, (void**)&RealQI));
    return CreateEntrypointWrapper(m_pMalloc, RealQI, iid, ppvObject);
  }
};

#define DEFINE_ENTRYPOINT_BOILERPLATE(Name)                                  \
    Name(IMalloc *M, IInterface *pI) : Entrypoint<IInterface>(M, pI){}       \
    DXC_MICROCOM_TM_ALLOC(Name)

struct IUnknownEntrypoint : public Entrypoint<IUnknown>
{
  DEFINE_ENTRYPOINT_BOILERPLATE(IUnknownEntrypoint);
};
DEFINE_ENTRYPOINT_WRAPPER_TRAIT(IUnknown);

struct IDxcPixTypeEntrypoint : public Entrypoint<IDxcPixType>
{
  DEFINE_ENTRYPOINT_BOILERPLATE(IDxcPixTypeEntrypoint);

  STDMETHODIMP GetName(
      _Outptr_result_z_ BSTR *Name) override
  {
    return InvokeOnReal(&IInterface::GetName, CheckNotNull(OutParam(Name)));
  }

  STDMETHODIMP GetSizeInBits(
      _Outptr_result_z_ DWORD *SizeInBits) override
  {
    return InvokeOnReal(&IInterface::GetSizeInBits, CheckNotNull(OutParam(SizeInBits)));
  }

  STDMETHODIMP UnAlias(
      _Outptr_result_z_ IDxcPixType** ppBaseType) override
  {
    return InvokeOnReal(&IInterface::UnAlias, CheckNotNull(OutParam(ppBaseType)));
  }
};
DEFINE_ENTRYPOINT_WRAPPER_TRAIT(IDxcPixType);

struct IDxcPixConstTypeEntrypoint : public Entrypoint<IDxcPixConstType>
{
  DEFINE_ENTRYPOINT_BOILERPLATE(IDxcPixConstTypeEntrypoint);

  STDMETHODIMP GetName(
      _Outptr_result_z_ BSTR *Name) override
  {
    return InvokeOnReal(&IInterface::GetName, CheckNotNull(OutParam(Name)));
  }

  STDMETHODIMP GetSizeInBits(
      _Outptr_result_z_ DWORD *SizeInBits) override
  {
    return InvokeOnReal(&IInterface::GetSizeInBits, CheckNotNull(OutParam(SizeInBits)));
  }

  STDMETHODIMP UnAlias(
      _Outptr_result_z_ IDxcPixType** ppBaseType) override
  {
    return InvokeOnReal(&IInterface::UnAlias, CheckNotNull(OutParam(ppBaseType)));
  }
};
DEFINE_ENTRYPOINT_WRAPPER_TRAIT(IDxcPixConstType);

struct IDxcPixTypedefTypeEntrypoint : public Entrypoint<IDxcPixTypedefType>
{
  DEFINE_ENTRYPOINT_BOILERPLATE(IDxcPixTypedefTypeEntrypoint);

  STDMETHODIMP GetName(
      _Outptr_result_z_ BSTR *Name) override
  {
    return InvokeOnReal(&IInterface::GetName, CheckNotNull(OutParam(Name)));
  }

  STDMETHODIMP GetSizeInBits(
      _Outptr_result_z_ DWORD *SizeInBits) override
  {
    return InvokeOnReal(&IInterface::GetSizeInBits, CheckNotNull(OutParam(SizeInBits)));
  }

  STDMETHODIMP UnAlias(
      _Outptr_result_z_ IDxcPixType** ppBaseType) override
  {
    return InvokeOnReal(&IInterface::UnAlias, CheckNotNull(OutParam(ppBaseType)));
  }
};
DEFINE_ENTRYPOINT_WRAPPER_TRAIT(IDxcPixTypedefType);

struct IDxcPixScalarTypeEntrypoint : public Entrypoint<IDxcPixScalarType>
{
  DEFINE_ENTRYPOINT_BOILERPLATE(IDxcPixScalarTypeEntrypoint);

  STDMETHODIMP GetName(
      _Outptr_result_z_ BSTR *Name) override
  {
    return InvokeOnReal(&IInterface::GetName, CheckNotNull(OutParam(Name)));
  }

  STDMETHODIMP GetSizeInBits(
      _Outptr_result_z_ DWORD *SizeInBits) override
  {
    return InvokeOnReal(&IInterface::GetSizeInBits, CheckNotNull(OutParam(SizeInBits)));
  }

  STDMETHODIMP UnAlias(
      _Outptr_result_z_ IDxcPixType** ppBaseType) override
  {
    return InvokeOnReal(&IInterface::UnAlias, CheckNotNull(OutParam(ppBaseType)));
  }
};
DEFINE_ENTRYPOINT_WRAPPER_TRAIT(IDxcPixScalarType);

struct IDxcPixArrayTypeEntrypoint : public Entrypoint<IDxcPixArrayType>
{
  DEFINE_ENTRYPOINT_BOILERPLATE(IDxcPixArrayTypeEntrypoint);

  STDMETHODIMP GetName(
      _Outptr_result_z_ BSTR *Name) override
  {
    return InvokeOnReal(&IInterface::GetName, CheckNotNull(OutParam(Name)));
  }

  STDMETHODIMP GetSizeInBits(
      _Outptr_result_z_ DWORD *SizeInBits) override
  {
    return InvokeOnReal(&IInterface::GetSizeInBits, CheckNotNull(OutParam(SizeInBits)));
  }

  STDMETHODIMP UnAlias(
      _Outptr_result_z_ IDxcPixType** ppBaseType) override
  {
    return InvokeOnReal(&IInterface::UnAlias, CheckNotNull(OutParam(ppBaseType)));
  }

  STDMETHODIMP GetNumElements(
      _Outptr_result_z_ DWORD *ppNumElements) override
  {
    return InvokeOnReal(&IInterface::GetNumElements, CheckNotNull(OutParam(ppNumElements)));
  }

  STDMETHODIMP GetIndexedType(
      _Outptr_result_z_ IDxcPixType **ppElementType) override
  {
    return InvokeOnReal(&IInterface::GetIndexedType, CheckNotNull(OutParam(ppElementType)));
  }

  STDMETHODIMP GetElementType(
      _Outptr_result_z_ IDxcPixType** ppElementType) override
  {
    return InvokeOnReal(&IInterface::GetElementType, CheckNotNull(OutParam(ppElementType)));
  }
};
DEFINE_ENTRYPOINT_WRAPPER_TRAIT(IDxcPixArrayType);

struct IDxcPixStructFieldEntrypoint : public Entrypoint<IDxcPixStructField>
{
  DEFINE_ENTRYPOINT_BOILERPLATE(IDxcPixStructFieldEntrypoint);

  STDMETHODIMP GetName(
      _Outptr_result_z_ BSTR *Name) override
  {
    return InvokeOnReal(&IInterface::GetName, CheckNotNull(OutParam(Name)));
  }

  STDMETHODIMP GetType(
      _Outptr_result_z_ IDxcPixType** ppType) override
  {
    return InvokeOnReal(&IInterface::GetType, CheckNotNull(OutParam(ppType)));
  }

  STDMETHODIMP GetOffsetInBits(
      _Outptr_result_z_ DWORD *pOffsetInBits) override
  {
    return InvokeOnReal(&IInterface::GetOffsetInBits, CheckNotNull(OutParam(pOffsetInBits)));
  }
};
DEFINE_ENTRYPOINT_WRAPPER_TRAIT(IDxcPixStructField);

struct IDxcPixStructTypeEntrypoint : public Entrypoint<IDxcPixStructType>
{
  DEFINE_ENTRYPOINT_BOILERPLATE(IDxcPixStructTypeEntrypoint);

  STDMETHODIMP GetName(
      _Outptr_result_z_ BSTR *Name) override
  {
    return InvokeOnReal(&IInterface::GetName, CheckNotNull(OutParam(Name)));
  }

  STDMETHODIMP GetSizeInBits(
      _Outptr_result_z_ DWORD *SizeInBits) override
  {
    return InvokeOnReal(&IInterface::GetSizeInBits, CheckNotNull(OutParam(SizeInBits)));
  }

  STDMETHODIMP UnAlias(
      _Outptr_result_z_ IDxcPixType** ppBaseType) override
  {
    return InvokeOnReal(&IInterface::UnAlias, CheckNotNull(OutParam(ppBaseType)));
  }

  STDMETHODIMP GetNumFields(
      _Outptr_result_z_ DWORD* ppNumFields) override
  {
      return InvokeOnReal(&IInterface::GetNumFields, CheckNotNull(OutParam(ppNumFields)));
  }

  STDMETHODIMP GetFieldByIndex(
      DWORD dwIndex,
      _Outptr_result_z_ IDxcPixStructField **ppField) override
  {
    return InvokeOnReal(&IInterface::GetFieldByIndex, dwIndex, CheckNotNull(OutParam(ppField)));
  }

  STDMETHODIMP GetFieldByName(
      _In_ LPCWSTR lpName,
      _Outptr_result_z_ IDxcPixStructField** ppField) override
  {
    return InvokeOnReal(&IInterface::GetFieldByName, CheckNotNull(InParam(lpName)), CheckNotNull(OutParam(ppField)));
  }
};
DEFINE_ENTRYPOINT_WRAPPER_TRAIT(IDxcPixStructType);

struct IDxcPixDxilStorageEntrypoint : public Entrypoint<IDxcPixDxilStorage>
{
  DEFINE_ENTRYPOINT_BOILERPLATE(IDxcPixDxilStorageEntrypoint);

  STDMETHODIMP AccessField(
      _In_ LPCWSTR Name,
      _COM_Outptr_ IDxcPixDxilStorage** ppResult) override
  {
    return InvokeOnReal(&IInterface::AccessField, CheckNotNull(InParam(Name)), CheckNotNull(OutParam(ppResult)));
  }

  STDMETHODIMP Index(
      _In_ DWORD Index,
      _COM_Outptr_ IDxcPixDxilStorage** ppResult) override
  {
    return InvokeOnReal(&IInterface::Index, Index, CheckNotNull(OutParam(ppResult)));
  }

  STDMETHODIMP GetRegisterNumber(
      _Outptr_result_z_ DWORD *pRegNum) override
  {
    return InvokeOnReal(&IInterface::GetRegisterNumber, CheckNotNull(OutParam(pRegNum)));
  }

  STDMETHODIMP GetIsAlive() override
  {
    return InvokeOnReal(&IInterface::GetIsAlive);
  }

  STDMETHODIMP GetType(
      _Outptr_result_z_ IDxcPixType** ppType) override
  {
    return InvokeOnReal(&IInterface::GetType, CheckNotNull(OutParam(ppType)));
  }
};
DEFINE_ENTRYPOINT_WRAPPER_TRAIT(IDxcPixDxilStorage);

struct IDxcPixVariableEntrypoint : public Entrypoint<IDxcPixVariable>
{
  DEFINE_ENTRYPOINT_BOILERPLATE(IDxcPixVariableEntrypoint);

  STDMETHODIMP GetName(
      _Outptr_result_z_ BSTR *Name) override
  {
    return InvokeOnReal(&IInterface::GetName, CheckNotNull(OutParam(Name)));
  }

  STDMETHODIMP GetType(
      _Outptr_result_z_ IDxcPixType **ppType) override
  {
    return InvokeOnReal(&IInterface::GetType, CheckNotNull(OutParam(ppType)));
  }

  STDMETHODIMP GetStorage(
      _COM_Outptr_ IDxcPixDxilStorage **ppStorage) override
  {
      return InvokeOnReal(&IInterface::GetStorage, CheckNotNull(OutParam(ppStorage)));
  }
};
DEFINE_ENTRYPOINT_WRAPPER_TRAIT(IDxcPixVariable);

struct IDxcPixDxilLiveVariablesEntrypoint : public Entrypoint<IDxcPixDxilLiveVariables>
{
  DEFINE_ENTRYPOINT_BOILERPLATE(IDxcPixDxilLiveVariablesEntrypoint);

  STDMETHODIMP GetCount(
      _Outptr_ DWORD *dwSize) override
  {
    return InvokeOnReal(&IInterface::GetCount, CheckNotNull(OutParam(dwSize)));
  }

  STDMETHODIMP GetVariableByIndex(
      _In_ DWORD Index,
      _Outptr_result_z_ IDxcPixVariable ** ppVariable) override
  {
    return InvokeOnReal(&IInterface::GetVariableByIndex, Index, CheckNotNull(OutParam(ppVariable)));
  }

  STDMETHODIMP GetVariableByName(
      _In_ LPCWSTR Name,
      _Outptr_result_z_ IDxcPixVariable** ppVariable) override
  {
    return InvokeOnReal(&IInterface::GetVariableByName, CheckNotNull(InParam(Name)), CheckNotNull(OutParam(ppVariable)));
  }
};
DEFINE_ENTRYPOINT_WRAPPER_TRAIT(IDxcPixDxilLiveVariables);

struct IDxcPixDxilDebugInfoEntrypoint : public Entrypoint<IDxcPixDxilDebugInfo>
{
  DEFINE_ENTRYPOINT_BOILERPLATE(IDxcPixDxilDebugInfoEntrypoint);

  STDMETHODIMP GetLiveVariablesAt(
      _In_ DWORD InstructionOffset,
      _COM_Outptr_ IDxcPixDxilLiveVariables **ppLiveVariables) override
  {
    return InvokeOnReal(&IInterface::GetLiveVariablesAt, InstructionOffset, CheckNotNull(OutParam(ppLiveVariables)));
  }

  STDMETHODIMP IsVariableInRegister(
      _In_ DWORD InstructionOffset,
      _In_ const wchar_t *VariableName) override
  {
    return InvokeOnReal(&IInterface::IsVariableInRegister, InstructionOffset, CheckNotNull(InParam(VariableName)));
  }

  STDMETHODIMP GetFunctionName(
      _In_ DWORD InstructionOffset,
      _Outptr_result_z_ BSTR *ppFunctionName) override
  {
    return InvokeOnReal(&IInterface::GetFunctionName, InstructionOffset, CheckNotNull(OutParam(ppFunctionName)));
  }

  STDMETHODIMP GetStackDepth(
      _In_ DWORD InstructionOffset,
      _Outptr_ DWORD *StackDepth) override
  {
    return InvokeOnReal(&IInterface::GetStackDepth, InstructionOffset, CheckNotNull(OutParam(StackDepth)));
  }

  STDMETHODIMP InstructionOffsetsFromSourceLocation(
      _In_ const wchar_t* FileName,
      _In_ DWORD SourceLine,
      _In_ DWORD SourceColumn,
      _COM_Outptr_ IDxcPixDxilInstructionOffsets** ppOffsets) override
  {
    return InvokeOnReal(&IInterface::InstructionOffsetsFromSourceLocation, CheckNotNull(InParam(FileName)), SourceLine, SourceColumn, CheckNotNull(OutParam(ppOffsets)));
  }

  STDMETHODIMP SourceLocationsFromInstructionOffset(
      _In_ DWORD InstructionOffset,
      _COM_Outptr_ IDxcPixDxilSourceLocations** ppSourceLocations) override
  {
      return InvokeOnReal(&IInterface::SourceLocationsFromInstructionOffset, InstructionOffset, CheckNotNull(OutParam(ppSourceLocations)));
  }
};
DEFINE_ENTRYPOINT_WRAPPER_TRAIT(IDxcPixDxilDebugInfo);

struct IDxcPixDxilInstructionOffsetsEntrypoint : public Entrypoint<IDxcPixDxilInstructionOffsets>
{
  DEFINE_ENTRYPOINT_BOILERPLATE(IDxcPixDxilInstructionOffsetsEntrypoint);

  STDMETHODIMP_(DWORD) GetCount() override
  {
    return InvokeOnReal(&IInterface::GetCount);
  }

  STDMETHODIMP_(DWORD) GetOffsetByIndex(_In_ DWORD Index) override
  {
    return InvokeOnReal(&IInterface::GetOffsetByIndex, Index);
  }
};
DEFINE_ENTRYPOINT_WRAPPER_TRAIT(IDxcPixDxilInstructionOffsets);

struct IDxcPixDxilSourceLocationsEntrypoint : public Entrypoint<IDxcPixDxilSourceLocations>
{
  DEFINE_ENTRYPOINT_BOILERPLATE(IDxcPixDxilSourceLocationsEntrypoint);

  STDMETHODIMP_(DWORD) GetCount() override
  {
      return InvokeOnReal(&IInterface::GetCount);
  }

  STDMETHODIMP_(DWORD) GetLineNumberByIndex(_In_ DWORD Index) override
  {
    return InvokeOnReal(&IInterface::GetLineNumberByIndex, Index);
  }

  STDMETHODIMP_(DWORD) GetColumnByIndex(_In_ DWORD Index) override
  {
    return InvokeOnReal(&IInterface::GetColumnByIndex, Index);
  }
  STDMETHODIMP GetFileNameByIndex(_In_ DWORD Index, _Outptr_result_z_ BSTR* Name) override
  {
    return InvokeOnReal(&IInterface::GetFileNameByIndex, Index, CheckNotNull(OutParam(Name)));
  }
};
DEFINE_ENTRYPOINT_WRAPPER_TRAIT(IDxcPixDxilSourceLocations);

struct IDxcPixCompilationInfoEntrypoint
    : public Entrypoint<IDxcPixCompilationInfo>
{
  DEFINE_ENTRYPOINT_BOILERPLATE(IDxcPixCompilationInfoEntrypoint);
  virtual STDMETHODIMP
  GetSourceFile(_In_ DWORD SourceFileOrdinal,
                _Outptr_result_z_ BSTR *pSourceName,
                _Outptr_result_z_ BSTR *pSourceContents) override {
    return InvokeOnReal(&IInterface::GetSourceFile, SourceFileOrdinal,
                        CheckNotNull(OutParam(pSourceName)),
                        CheckNotNull(OutParam(pSourceContents))
      );
  }

  virtual STDMETHODIMP GetArguments(_Outptr_result_z_ BSTR* pArguments) override 
  {
    return InvokeOnReal(&IInterface::GetArguments, CheckNotNull(OutParam(pArguments)));
  }
  virtual STDMETHODIMP
    GetMacroDefinitions(_Outptr_result_z_ BSTR* pMacroDefinitions) override 
  {
    return InvokeOnReal(&IInterface::GetMacroDefinitions,
                        CheckNotNull(OutParam(pMacroDefinitions)));

  }
  virtual STDMETHODIMP
    GetEntryPointFile(_Outptr_result_z_ BSTR* pEntryPointFile) override {
    return InvokeOnReal(&IInterface::GetEntryPointFile,
                        CheckNotNull(OutParam(pEntryPointFile)));

  }
  virtual STDMETHODIMP
    GetHlslTarget(_Outptr_result_z_ BSTR* pHlslTarget) override {
    return InvokeOnReal(&IInterface::GetHlslTarget,
                        CheckNotNull(OutParam(pHlslTarget)));

  }
  virtual STDMETHODIMP
    GetEntryPoint(_Outptr_result_z_ BSTR* pEntryPoint) override
  {
    return InvokeOnReal(&IInterface::GetEntryPoint,
                        CheckNotNull(OutParam(pEntryPoint)));

  }
};
DEFINE_ENTRYPOINT_WRAPPER_TRAIT(IDxcPixCompilationInfo);


HRESULT CreateEntrypointWrapper(
    IMalloc* pMalloc,
    IUnknown* pReal,
    REFIID riid,
    void** ppvObject)
{
#define HANDLE_INTERFACE(IInterface)                                         \
    if (__uuidof(IInterface) == riid)                                        \
    {                                                                        \
        return NewDxcPixDxilDebugInfoObjectOrThrow<IInterface##Entrypoint>(  \
            (IInterface **) ppvObject,                                       \
            pMalloc,                                                         \
            (IInterface *) pReal);                                           \
    } (void)0

  HANDLE_INTERFACE(IUnknown);
  HANDLE_INTERFACE(IDxcPixType);
  HANDLE_INTERFACE(IDxcPixConstType);
  HANDLE_INTERFACE(IDxcPixTypedefType);
  HANDLE_INTERFACE(IDxcPixScalarType);
  HANDLE_INTERFACE(IDxcPixArrayType);
  HANDLE_INTERFACE(IDxcPixStructField);
  HANDLE_INTERFACE(IDxcPixStructType);
  HANDLE_INTERFACE(IDxcPixDxilStorage);
  HANDLE_INTERFACE(IDxcPixVariable);
  HANDLE_INTERFACE(IDxcPixDxilLiveVariables);
  HANDLE_INTERFACE(IDxcPixDxilDebugInfo);
  HANDLE_INTERFACE(IDxcPixCompilationInfo);

  return E_FAIL;
}

}  // namespace entrypoints
}  // namespace dxil_debug_info

#include "DxilDiaSession.h"

using namespace dxil_debug_info::entrypoints;
static STDMETHODIMP NewDxcPixDxilDebugInfoImpl(
    IMalloc *pMalloc,
    dxil_dia::Session *pSession,
    IDxcPixDxilDebugInfo** ppDxilDebugInfo
)
{
  return dxil_debug_info::NewDxcPixDxilDebugInfoObjectOrThrow<dxil_debug_info::DxcPixDxilDebugInfo>(
      ppDxilDebugInfo,
      pMalloc,
      pSession);
}

STDMETHODIMP dxil_dia::Session::NewDxcPixDxilDebugInfo(
    _COM_Outptr_ IDxcPixDxilDebugInfo** ppDxilDebugInfo)
{
  return SetupAndRun(
      m_pMalloc,
      &NewDxcPixDxilDebugInfoImpl,
      m_pMalloc,
      ThisPtr(this),
      CheckNotNull(OutParam(ppDxilDebugInfo)));
}

static STDMETHODIMP NewDxcPixCompilationInfoImpl(
    IMalloc *pMalloc,
    dxil_dia::Session *pSession,
    IDxcPixCompilationInfo** ppCompilationInfo
)
{
  return dxil_debug_info::CreateDxilCompilationInfo(
      pMalloc,
      pSession,
      ppCompilationInfo);
}

STDMETHODIMP dxil_dia::Session::NewDxcPixCompilationInfo(
    _COM_Outptr_ IDxcPixCompilationInfo **ppCompilationInfo)
{
  return SetupAndRun(
      m_pMalloc,
      &NewDxcPixCompilationInfoImpl,
      m_pMalloc,
      ThisPtr(this), 
      CheckNotNull(OutParam(ppCompilationInfo)));
}
