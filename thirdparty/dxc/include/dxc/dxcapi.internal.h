///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// dxcapi.internal.h                                                         //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides non-public declarations for the DirectX Compiler component.      //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#ifndef __DXC_API_INTERNAL__
#define __DXC_API_INTERNAL__

#include "dxcapi.h"

///////////////////////////////////////////////////////////////////////////////
// Forward declarations.
typedef struct ITextFont ITextFont;
typedef struct IEnumSTATSTG IEnumSTATSTG;
typedef struct ID3D10Blob ID3D10Blob;

///////////////////////////////////////////////////////////////////////////////
// Intrinsic definitions.
#define AR_QUAL_IN             0x0000000000000010ULL
#define AR_QUAL_OUT            0x0000000000000020ULL
#define AR_QUAL_CONST          0x0000000000000200ULL
#define AR_QUAL_ROWMAJOR       0x0000000000000400ULL
#define AR_QUAL_COLMAJOR       0x0000000000000800ULL

#define AR_QUAL_IN_OUT (AR_QUAL_IN | AR_QUAL_OUT)

static const BYTE INTRIN_TEMPLATE_FROM_TYPE = 0xff;
static const BYTE INTRIN_TEMPLATE_VARARGS = 0xfe;
static const BYTE INTRIN_TEMPLATE_FROM_FUNCTION = 0xfd;

// Use this enumeration to describe allowed templates (layouts) in intrinsics.
enum LEGAL_INTRINSIC_TEMPLATES {
  LITEMPLATE_VOID   = 0,  // No return type.
  LITEMPLATE_SCALAR = 1,  // Scalar types.
  LITEMPLATE_VECTOR = 2,  // Vector types (eg. float3).
  LITEMPLATE_MATRIX = 3,  // Matrix types (eg. float3x3).
  LITEMPLATE_ANY    = 4,  // Any one of scalar, vector or matrix types (but not object).
  LITEMPLATE_OBJECT = 5,  // Object types.

  LITEMPLATE_COUNT = 6
};

// INTRIN_COMPTYPE_FROM_TYPE_ELT0 is for object method intrinsics to indicate
// that the component type of the type is taken from the first subelement of the
// object's template type; see for example Texture2D.Gather
static const BYTE INTRIN_COMPTYPE_FROM_TYPE_ELT0 = 0xff;

enum LEGAL_INTRINSIC_COMPTYPES {
  LICOMPTYPE_VOID = 0,            // void, used for function returns
  LICOMPTYPE_BOOL = 1,            // bool
  LICOMPTYPE_INT = 2,             // i32, int-literal
  LICOMPTYPE_UINT = 3,            // u32, int-literal
  LICOMPTYPE_ANY_INT = 4,         // i32, u32, i64, u64, int-literal
  LICOMPTYPE_ANY_INT32 = 5,       // i32, u32, int-literal
  LICOMPTYPE_UINT_ONLY = 6,       // u32, u64, int-literal; no casts allowed
  LICOMPTYPE_FLOAT = 7,           // f32, partial-precision-f32, float-literal
  LICOMPTYPE_ANY_FLOAT = 8,       // f32, partial-precision-f32, f64, float-literal, min10-float, min16-float, half
  LICOMPTYPE_FLOAT_LIKE = 9,      // f32, partial-precision-f32, float-literal, min10-float, min16-float, half
  LICOMPTYPE_FLOAT_DOUBLE = 10,   // f32, partial-precision-f32, f64, float-literal
  LICOMPTYPE_DOUBLE = 11,         // f64, float-literal
  LICOMPTYPE_DOUBLE_ONLY = 12,    // f64; no casts allowed
  LICOMPTYPE_NUMERIC = 13,        // float-literal, f32, partial-precision-f32, f64, min10-float, min16-float, int-literal, i32, u32, min12-int, min16-int, min16-uint, i64, u64
  LICOMPTYPE_NUMERIC32 = 14,      // float-literal, f32, partial-precision-f32, int-literal, i32, u32
  LICOMPTYPE_NUMERIC32_ONLY = 15, // float-literal, f32, partial-precision-f32, int-literal, i32, u32; no casts allowed
  LICOMPTYPE_ANY = 16,            // float-literal, f32, partial-precision-f32, f64, min10-float, min16-float, int-literal, i32, u32, min12-int, min16-int, min16-uint, bool, i64, u64
  LICOMPTYPE_SAMPLER1D = 17,
  LICOMPTYPE_SAMPLER2D = 18,
  LICOMPTYPE_SAMPLER3D = 19,
  LICOMPTYPE_SAMPLERCUBE = 20,
  LICOMPTYPE_SAMPLERCMP = 21,
  LICOMPTYPE_SAMPLER = 22,
  LICOMPTYPE_STRING = 23,
  LICOMPTYPE_WAVE = 24,
  LICOMPTYPE_UINT64 = 25,         // u64, int-literal
  LICOMPTYPE_FLOAT16 = 26,
  LICOMPTYPE_INT16 = 27,
  LICOMPTYPE_UINT16 = 28,
  LICOMPTYPE_NUMERIC16_ONLY = 29,

  LICOMPTYPE_RAYDESC = 30,
  LICOMPTYPE_ACCELERATION_STRUCT = 31,
  LICOMPTYPE_USER_DEFINED_TYPE = 32,

  LICOMPTYPE_TEXTURE2D = 33,
  LICOMPTYPE_TEXTURE2DARRAY = 34,
  LICOMPTYPE_RESOURCE = 35,
  LICOMPTYPE_INT32_ONLY = 36,
  LICOMPTYPE_INT64_ONLY = 37,
  LICOMPTYPE_ANY_INT64 = 38,
  LICOMPTYPE_FLOAT32_ONLY = 39,
  LICOMPTYPE_INT8_4PACKED = 40,
  LICOMPTYPE_UINT8_4PACKED = 41,
  LICOMPTYPE_ANY_INT16_OR_32 = 42,
  LICOMPTYPE_SINT16_OR_32_ONLY = 43,
  LICOMPTYPE_COUNT = 44
};

static const BYTE IA_SPECIAL_BASE = 0xf0;
static const BYTE IA_R = 0xf0;
static const BYTE IA_C = 0xf1;
static const BYTE IA_R2 = 0xf2;
static const BYTE IA_C2 = 0xf3;
static const BYTE IA_SPECIAL_SLOTS = 4;

struct HLSL_INTRINSIC_ARGUMENT {
  LPCSTR pName;               // Name of the argument; the first argument has the function name.
  UINT64 qwUsage;             // A combination of AR_QUAL_IN|AR_QUAL_OUT|AR_QUAL_COLMAJOR|AR_QUAL_ROWMAJOR in parameter tables; other values possible elsewhere.

  BYTE uTemplateId;           // One of INTRIN_TEMPLATE_FROM_TYPE, INTRIN_TEMPLATE_VARARGS or the argument # the template (layout) must match (trivially itself).
  BYTE uLegalTemplates;       // A LEGAL_INTRINSIC_TEMPLATES value for allowed templates.
  BYTE uComponentTypeId;      // INTRIN_COMPTYPE_FROM_TYPE_ELT0, or the argument # the component (element type) must match (trivially itself).
  BYTE uLegalComponentTypes;  // A LEGAL_INTRINSIC_COMPTYPES value for allowed components.

  BYTE uRows;                 // Required number of rows, or one of IA_R/IA_C/IA_R2/IA_C2 for matching input constraints.
  BYTE uCols;                 // Required number of cols, or one of IA_R/IA_C/IA_R2/IA_C2 for matching input constraints.
};

struct HLSL_INTRINSIC {
  UINT Op;                              // Intrinsic Op ID
  BOOL bReadOnly;                       // Only read memory
  BOOL bReadNone;                       // Not read memory
  BOOL bIsWave;                         // Is a wave-sensitive op
  INT  iOverloadParamIndex;             // Parameter decide the overload type, -1 means ret type
  UINT uNumArgs;                        // Count of arguments in pArgs.
  const HLSL_INTRINSIC_ARGUMENT* pArgs; // Pointer to first argument.
};

///////////////////////////////////////////////////////////////////////////////
// Interfaces.
CROSS_PLATFORM_UUIDOF(IDxcIntrinsicTable, "f0d4da3f-f863-4660-b8b4-dfd94ded6215")
struct IDxcIntrinsicTable : public IUnknown
{
public:
  virtual HRESULT STDMETHODCALLTYPE GetTableName(_Outptr_ LPCSTR *pTableName) = 0;
  virtual HRESULT STDMETHODCALLTYPE LookupIntrinsic(
    LPCWSTR typeName, LPCWSTR functionName,
    const HLSL_INTRINSIC** pIntrinsic,
    _Inout_ UINT64* pLookupCookie) = 0;

  // Get the lowering strategy for an hlsl extension intrinsic.
  virtual HRESULT STDMETHODCALLTYPE GetLoweringStrategy(UINT opcode, LPCSTR *pStrategy) = 0;

  // Callback to support custom naming of hlsl extension intrinsic functions in dxil.
  // Return the empty string to get the default intrinsic name, which is the mangled
  // name of the high level intrinsic function.
  //
  // Overloaded intrinsics are supported by use of an overload place holder in the
  // name. The string "$o" in the name will be replaced by the return type of the
  // intrinsic.
  virtual HRESULT STDMETHODCALLTYPE GetIntrinsicName(UINT opcode, LPCSTR *pName) = 0;

  // Callback to support the 'dxil' lowering strategy.
  // Returns the dxil opcode that the intrinsic should use for lowering.
  virtual HRESULT STDMETHODCALLTYPE GetDxilOpCode(UINT opcode, UINT *pDxilOpcode) = 0;
};

CROSS_PLATFORM_UUIDOF(IDxcSemanticDefineValidator, "1d063e4f-515a-4d57-a12a-431f6a44cfb9")
struct IDxcSemanticDefineValidator : public IUnknown
{
public:
  virtual HRESULT STDMETHODCALLTYPE GetSemanticDefineWarningsAndErrors(LPCSTR pName, LPCSTR pValue, IDxcBlobEncoding **ppWarningBlob, IDxcBlobEncoding **ppErrorBlob) = 0;
};

CROSS_PLATFORM_UUIDOF(IDxcLangExtensions, "282a56b4-3f56-4360-98c7-9ea04a752272")
struct IDxcLangExtensions : public IUnknown
{
public:
  /// <summary>
  /// Registers the name of a preprocessor define that has semantic meaning
  /// and should be preserved for downstream consumers.
  /// </summary>
  virtual HRESULT STDMETHODCALLTYPE RegisterSemanticDefine(LPCWSTR name) = 0;
  /// <summary>Registers a name to exclude from semantic defines.</summary>
  virtual HRESULT STDMETHODCALLTYPE RegisterSemanticDefineExclusion(LPCWSTR name) = 0;
  /// <summary>Registers a definition for compilation.</summary>
  virtual HRESULT STDMETHODCALLTYPE RegisterDefine(LPCWSTR name) = 0;
  /// <summary>Registers a table of built-in intrinsics.</summary>
  virtual HRESULT STDMETHODCALLTYPE RegisterIntrinsicTable(_In_ IDxcIntrinsicTable* pTable) = 0;
  /// <summary>Sets an (optional) validator for parsed semantic defines.<summary>
  /// This provides a hook to check that the semantic defines present in the source
  /// contain valid data. One validator is used to validate all parsed semantic defines.
  virtual HRESULT STDMETHODCALLTYPE SetSemanticDefineValidator(_In_ IDxcSemanticDefineValidator* pValidator) = 0;
  /// <summary>Sets the name for the root metadata node used in DXIL to hold the semantic defines.</summary>
  virtual HRESULT STDMETHODCALLTYPE SetSemanticDefineMetaDataName(LPCSTR name) = 0;
};

CROSS_PLATFORM_UUIDOF(IDxcLangExtensions2, "2490C368-89EE-4491-A4B2-C6547B6C9381")
struct IDxcLangExtensions2 : public IDxcLangExtensions {
public:
  virtual HRESULT STDMETHODCALLTYPE SetTargetTriple(LPCSTR name) = 0;
};

CROSS_PLATFORM_UUIDOF(IDxcLangExtensions3, "A1B19880-FB1F-4920-9BC5-50356483BAC1")
struct IDxcLangExtensions3 : public IDxcLangExtensions2 {
public:
  /// Registers a semantic define which cannot be overriden using the flag -override-opt-semdefs
  virtual HRESULT STDMETHODCALLTYPE RegisterNonOptSemanticDefine(LPCWSTR name) = 0;
};

CROSS_PLATFORM_UUIDOF(IDxcSystemAccess, "454b764f-3549-475b-958c-a7a6fcd05fbc")
struct IDxcSystemAccess : public IUnknown
{
public:
  virtual HRESULT STDMETHODCALLTYPE EnumFiles(LPCWSTR fileName, IEnumSTATSTG** pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE OpenStorage(
    _In_      LPCWSTR lpFileName,
    _In_      DWORD dwDesiredAccess,
    _In_      DWORD dwShareMode,
    _In_      DWORD dwCreationDisposition,
    _In_      DWORD dwFlagsAndAttributes, IUnknown** pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE SetStorageTime(_In_ IUnknown* storage,
    _In_opt_  const FILETIME *lpCreationTime,
    _In_opt_  const FILETIME *lpLastAccessTime,
    _In_opt_  const FILETIME *lpLastWriteTime) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetFileInformationForStorage(_In_ IUnknown* storage, _Out_ LPBY_HANDLE_FILE_INFORMATION lpFileInformation) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetFileTypeForStorage(_In_ IUnknown* storage, _Out_ DWORD* fileType) = 0;
  virtual HRESULT STDMETHODCALLTYPE CreateHardLinkInStorage(_In_ LPCWSTR lpFileName, _In_ LPCWSTR lpExistingFileName) = 0;
  virtual HRESULT STDMETHODCALLTYPE MoveStorage(_In_ LPCWSTR lpExistingFileName, _In_opt_ LPCWSTR lpNewFileName, _In_ DWORD dwFlags) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetFileAttributesForStorage(_In_ LPCWSTR lpFileName, _Out_ DWORD* pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE DeleteStorage(_In_ LPCWSTR lpFileName) = 0;
  virtual HRESULT STDMETHODCALLTYPE RemoveDirectoryStorage(LPCWSTR lpFileName) = 0;
  virtual HRESULT STDMETHODCALLTYPE CreateDirectoryStorage(_In_ LPCWSTR lpPathName) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetCurrentDirectoryForStorage(DWORD nBufferLength, _Out_writes_(nBufferLength) LPWSTR lpBuffer, _Out_ DWORD* written) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetMainModuleFileNameW(DWORD nBufferLength, _Out_writes_(nBufferLength) LPWSTR lpBuffer, _Out_ DWORD* written) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetTempStoragePath(DWORD nBufferLength, _Out_writes_(nBufferLength) LPWSTR lpBuffer, _Out_ DWORD* written) = 0;
  virtual HRESULT STDMETHODCALLTYPE SupportsCreateSymbolicLink(_Out_ BOOL* pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE CreateSymbolicLinkInStorage(_In_ LPCWSTR lpSymlinkFileName, _In_ LPCWSTR lpTargetFileName, DWORD dwFlags) = 0;
  virtual HRESULT STDMETHODCALLTYPE CreateStorageMapping(
    _In_      IUnknown* hFile,
    _In_      DWORD flProtect,
    _In_      DWORD dwMaximumSizeHigh,
    _In_      DWORD dwMaximumSizeLow,
    _Outptr_  IUnknown** pResult) = 0;
  virtual HRESULT MapViewOfFile(
    _In_  IUnknown* hFileMappingObject,
    _In_  DWORD dwDesiredAccess,
    _In_  DWORD dwFileOffsetHigh,
    _In_  DWORD dwFileOffsetLow,
    _In_  SIZE_T dwNumberOfBytesToMap,
    _Outptr_ ID3D10Blob** pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE OpenStdStorage(int standardFD, _Outptr_ IUnknown** pResult) = 0;
  virtual HRESULT STDMETHODCALLTYPE GetStreamDisplay(_COM_Outptr_result_maybenull_ ITextFont** textFont, _Out_ unsigned* columnCount) = 0;
};

CROSS_PLATFORM_UUIDOF(IDxcContainerEventsHandler, "e991ca8d-2045-413c-a8b8-788b2c06e14d")
struct IDxcContainerEventsHandler : public IUnknown
{
public:
  virtual HRESULT STDMETHODCALLTYPE OnDxilContainerBuilt(_In_ IDxcBlob *pSource, _Out_ IDxcBlob **ppTarget) = 0;
};

CROSS_PLATFORM_UUIDOF(IDxcContainerEvent, "0cfc5058-342b-4ff2-83f7-04c12aad3d01")
struct IDxcContainerEvent : public IUnknown
{
public:
  virtual HRESULT STDMETHODCALLTYPE RegisterDxilContainerEventHandler(IDxcContainerEventsHandler *pHandler, UINT64 *pCookie) = 0;
  virtual HRESULT STDMETHODCALLTYPE UnRegisterDxilContainerEventHandler(UINT64 cookie) = 0;
};
#endif
