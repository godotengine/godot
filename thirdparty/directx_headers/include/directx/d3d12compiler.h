/*-------------------------------------------------------------------------------------
 *
 * Copyright (c) Microsoft Corporation
 * Licensed under the MIT license
 *
 *-------------------------------------------------------------------------------------*/


/* this ALWAYS GENERATED file contains the definitions for the interfaces */


 /* File created by MIDL compiler version 8.01.0628 */



/* verify that the <rpcndr.h> version is high enough to compile this file*/
#ifndef __REQUIRED_RPCNDR_H_VERSION__
#define __REQUIRED_RPCNDR_H_VERSION__ 500
#endif

/* verify that the <rpcsal.h> version is high enough to compile this file*/
#ifndef __REQUIRED_RPCSAL_H_VERSION__
#define __REQUIRED_RPCSAL_H_VERSION__ 100
#endif

#include "rpc.h"
#include "rpcndr.h"

#ifndef __RPCNDR_H_VERSION__
#error this stub requires an updated version of <rpcndr.h>
#endif /* __RPCNDR_H_VERSION__ */

#ifndef COM_NO_WINDOWS_H
#include "windows.h"
#include "ole2.h"
#endif /*COM_NO_WINDOWS_H*/

#ifndef __d3d12compiler_h__
#define __d3d12compiler_h__

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#endif

#ifndef DECLSPEC_XFGVIRT
#if defined(_CONTROL_FLOW_GUARD_XFG)
#define DECLSPEC_XFGVIRT(base, func) __declspec(xfg_virtual(base, func))
#else
#define DECLSPEC_XFGVIRT(base, func)
#endif
#endif

/* Forward Declarations */ 

#ifndef __ID3D12CompilerFactoryChild_FWD_DEFINED__
#define __ID3D12CompilerFactoryChild_FWD_DEFINED__
typedef interface ID3D12CompilerFactoryChild ID3D12CompilerFactoryChild;

#endif 	/* __ID3D12CompilerFactoryChild_FWD_DEFINED__ */


#ifndef __ID3D12CompilerCacheSession_FWD_DEFINED__
#define __ID3D12CompilerCacheSession_FWD_DEFINED__
typedef interface ID3D12CompilerCacheSession ID3D12CompilerCacheSession;

#endif 	/* __ID3D12CompilerCacheSession_FWD_DEFINED__ */


#ifndef __ID3D12CompilerStateObject_FWD_DEFINED__
#define __ID3D12CompilerStateObject_FWD_DEFINED__
typedef interface ID3D12CompilerStateObject ID3D12CompilerStateObject;

#endif 	/* __ID3D12CompilerStateObject_FWD_DEFINED__ */


#ifndef __ID3D12Compiler_FWD_DEFINED__
#define __ID3D12Compiler_FWD_DEFINED__
typedef interface ID3D12Compiler ID3D12Compiler;

#endif 	/* __ID3D12Compiler_FWD_DEFINED__ */


#ifndef __ID3D12CompilerFactory_FWD_DEFINED__
#define __ID3D12CompilerFactory_FWD_DEFINED__
typedef interface ID3D12CompilerFactory ID3D12CompilerFactory;

#endif 	/* __ID3D12CompilerFactory_FWD_DEFINED__ */


/* header files for imported files */
#include "oaidl.h"
#include "ocidl.h"
#include "dxgicommon.h"
#include "d3d12.h"

#ifdef __cplusplus
extern "C"{
#endif 


/* interface __MIDL_itf_d3d12compiler_0000_0000 */
/* [local] */ 

#include <winapifamily.h>
#pragma region App Family
#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP | WINAPI_PARTITION_GAMES)
typedef struct D3D12_ADAPTER_FAMILY
    {
    WCHAR szAdapterFamily[ 128 ];
    } 	D3D12_ADAPTER_FAMILY;

typedef HRESULT (WINAPI* PFN_D3D12_COMPILER_CREATE_FACTORY)( 
    _In_ LPCWSTR pPluginCompilerDllPath,
    _In_ REFIID riid,
    _COM_Outptr_opt_ void** ppFactory );

HRESULT WINAPI D3D12CompilerCreateFactory(
    _In_ LPCWSTR pPluginCompilerDllPath,
    _In_ REFIID riid, // Expected: ID3D12CompilerFactory
    _COM_Outptr_opt_ void** ppFactory );

typedef HRESULT (WINAPI* PFN_D3D12_COMPILER_SERIALIZE_VERSIONED_ROOT_SIGNATURE)(
                            _In_ const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* pRootSignature,
                            _Out_ ID3DBlob** ppBlob,
                            _Always_(_Outptr_opt_result_maybenull_) ID3DBlob** ppErrorBlob);

HRESULT WINAPI D3D12CompilerSerializeVersionedRootSignature(
                            _In_ const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* pRootSignature,
                            _Out_ ID3DBlob** ppBlob,
                            _Always_(_Outptr_opt_result_maybenull_) ID3DBlob** ppErrorBlob);



extern RPC_IF_HANDLE __MIDL_itf_d3d12compiler_0000_0000_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12compiler_0000_0000_v0_0_s_ifspec;

#ifndef __ID3D12CompilerFactoryChild_INTERFACE_DEFINED__
#define __ID3D12CompilerFactoryChild_INTERFACE_DEFINED__

/* interface ID3D12CompilerFactoryChild */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12CompilerFactoryChild;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("e0d06420-9f31-47e8-ae9a-dd2ba25ac0bc")
    ID3D12CompilerFactoryChild : public IUnknown
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE GetFactory( 
            _In_  REFIID riid,
            _COM_Outptr_  void **ppCompilerFactory) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12CompilerFactoryChildVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12CompilerFactoryChild * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12CompilerFactoryChild * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12CompilerFactoryChild * This);
        
        DECLSPEC_XFGVIRT(ID3D12CompilerFactoryChild, GetFactory)
        HRESULT ( STDMETHODCALLTYPE *GetFactory )( 
            ID3D12CompilerFactoryChild * This,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppCompilerFactory);
        
        END_INTERFACE
    } ID3D12CompilerFactoryChildVtbl;

    interface ID3D12CompilerFactoryChild
    {
        CONST_VTBL struct ID3D12CompilerFactoryChildVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12CompilerFactoryChild_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12CompilerFactoryChild_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12CompilerFactoryChild_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12CompilerFactoryChild_GetFactory(This,riid,ppCompilerFactory)	\
    ( (This)->lpVtbl -> GetFactory(This,riid,ppCompilerFactory) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12CompilerFactoryChild_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12compiler_0000_0001 */
/* [local] */ 

typedef 
enum D3D12_COMPILER_VALUE_TYPE
    {
        D3D12_COMPILER_VALUE_TYPE_OBJECT_CODE	= 0,
        D3D12_COMPILER_VALUE_TYPE_METADATA	= 1,
        D3D12_COMPILER_VALUE_TYPE_DEBUG_PDB	= 2,
        D3D12_COMPILER_VALUE_TYPE_PERFORMANCE_DATA	= 3
    } 	D3D12_COMPILER_VALUE_TYPE;

typedef 
enum D3D12_COMPILER_VALUE_TYPE_FLAGS
    {
        D3D12_COMPILER_VALUE_TYPE_FLAGS_NONE	= 0,
        D3D12_COMPILER_VALUE_TYPE_FLAGS_OBJECT_CODE	= ( 1 << D3D12_COMPILER_VALUE_TYPE_OBJECT_CODE ) ,
        D3D12_COMPILER_VALUE_TYPE_FLAGS_METADATA	= ( 1 << D3D12_COMPILER_VALUE_TYPE_METADATA ) ,
        D3D12_COMPILER_VALUE_TYPE_FLAGS_DEBUG_PDB	= ( 1 << D3D12_COMPILER_VALUE_TYPE_DEBUG_PDB ) ,
        D3D12_COMPILER_VALUE_TYPE_FLAGS_PERFORMANCE_DATA	= ( 1 << D3D12_COMPILER_VALUE_TYPE_PERFORMANCE_DATA ) 
    } 	D3D12_COMPILER_VALUE_TYPE_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS( D3D12_COMPILER_VALUE_TYPE_FLAGS )
typedef struct D3D12_COMPILER_DATABASE_PATH
    {
    D3D12_COMPILER_VALUE_TYPE_FLAGS Types;
    LPCWSTR pPath;
    } 	D3D12_COMPILER_DATABASE_PATH;

typedef struct D3D12_COMPILER_CACHE_GROUP_KEY
    {
    _Field_size_bytes_full_(KeySize)  const void *pKey;
    UINT KeySize;
    } 	D3D12_COMPILER_CACHE_GROUP_KEY;

typedef struct D3D12_COMPILER_CACHE_VALUE_KEY
    {
    _Field_size_bytes_full_(KeySize)  const void *pKey;
    UINT KeySize;
    } 	D3D12_COMPILER_CACHE_VALUE_KEY;

typedef struct D3D12_COMPILER_CACHE_VALUE
    {
    _Field_size_bytes_full_(ValueSize)  void *pValue;
    UINT ValueSize;
    } 	D3D12_COMPILER_CACHE_VALUE;

typedef struct D3D12_COMPILER_CACHE_TYPED_VALUE
    {
    D3D12_COMPILER_VALUE_TYPE Type;
    D3D12_COMPILER_CACHE_VALUE Value;
    } 	D3D12_COMPILER_CACHE_TYPED_VALUE;

typedef struct D3D12_COMPILER_CACHE_CONST_VALUE
    {
    _Field_size_bytes_full_(ValueSize)  const void *pValue;
    UINT ValueSize;
    } 	D3D12_COMPILER_CACHE_CONST_VALUE;

typedef struct D3D12_COMPILER_CACHE_TYPED_CONST_VALUE
    {
    D3D12_COMPILER_VALUE_TYPE Type;
    D3D12_COMPILER_CACHE_CONST_VALUE Value;
    } 	D3D12_COMPILER_CACHE_TYPED_CONST_VALUE;

typedef struct D3D12_COMPILER_TARGET
    {
    UINT AdapterFamilyIndex;
    UINT64 ABIVersion;
    } 	D3D12_COMPILER_TARGET;

typedef void *( __stdcall *D3D12CompilerCacheSessionAllocationFunc )( 
    SIZE_T SizeInBytes,
    _Inout_opt_  void *pContext);

typedef void ( __stdcall *D3D12CompilerCacheSessionGroupValueKeysFunc )( 
    _In_  const D3D12_COMPILER_CACHE_VALUE_KEY *pValueKey,
    _Inout_opt_  void *pContext);

typedef void ( __stdcall *D3D12CompilerCacheSessionGroupValuesFunc )( 
    UINT ValueKeyIndex,
    _In_  const D3D12_COMPILER_CACHE_TYPED_CONST_VALUE *pTypedValue,
    _Inout_opt_  void *pContext);



extern RPC_IF_HANDLE __MIDL_itf_d3d12compiler_0000_0001_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12compiler_0000_0001_v0_0_s_ifspec;

#ifndef __ID3D12CompilerCacheSession_INTERFACE_DEFINED__
#define __ID3D12CompilerCacheSession_INTERFACE_DEFINED__

/* interface ID3D12CompilerCacheSession */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12CompilerCacheSession;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("5704e5e6-054b-4738-b661-7b0d68d8dde2")
    ID3D12CompilerCacheSession : public ID3D12CompilerFactoryChild
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE FindGroup( 
            _In_  const D3D12_COMPILER_CACHE_GROUP_KEY *pGroupKey,
            _Out_opt_  UINT *pGroupVersion) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE FindGroupValueKeys( 
            _In_  const D3D12_COMPILER_CACHE_GROUP_KEY *pGroupKey,
            _In_opt_  const UINT *pExpectedGroupVersion,
            _In_  D3D12CompilerCacheSessionGroupValueKeysFunc CallbackFunc,
            _Inout_opt_  void *pContext) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE FindGroupValues( 
            _In_  const D3D12_COMPILER_CACHE_GROUP_KEY *pGroupKey,
            _In_opt_  const UINT *pExpectedGroupVersion,
            D3D12_COMPILER_VALUE_TYPE_FLAGS ValueTypeFlags,
            _In_opt_  D3D12CompilerCacheSessionGroupValuesFunc CallbackFunc,
            _Inout_opt_  void *pContext) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE FindValue( 
            _In_  const D3D12_COMPILER_CACHE_VALUE_KEY *pValueKey,
            _Inout_count_(NumTypedValues)  D3D12_COMPILER_CACHE_TYPED_VALUE *pTypedValues,
            UINT NumTypedValues,
            _In_opt_  D3D12CompilerCacheSessionAllocationFunc pCallbackFunc,
            _Inout_opt_  void *pContext) = 0;
        
        virtual const D3D12_APPLICATION_DESC *STDMETHODCALLTYPE GetApplicationDesc( void) = 0;
        
#if defined(_MSC_VER) || !defined(_WIN32)
        virtual D3D12_COMPILER_TARGET STDMETHODCALLTYPE GetCompilerTarget( void) = 0;
#else
        virtual D3D12_COMPILER_TARGET *STDMETHODCALLTYPE GetCompilerTarget( 
            D3D12_COMPILER_TARGET * RetVal) = 0;
#endif
        
        virtual D3D12_COMPILER_VALUE_TYPE_FLAGS STDMETHODCALLTYPE GetValueTypes( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE StoreGroupValueKeys( 
            _In_  const D3D12_COMPILER_CACHE_GROUP_KEY *pGroupKey,
            UINT GroupVersion,
            _In_reads_(NumValueKeys)  const D3D12_COMPILER_CACHE_VALUE_KEY *pValueKeys,
            UINT NumValueKeys) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE StoreValue( 
            _In_  const D3D12_COMPILER_CACHE_VALUE_KEY *pValueKey,
            _In_reads_(NumTypedValues)  const D3D12_COMPILER_CACHE_TYPED_CONST_VALUE *pTypedValues,
            UINT NumTypedValues) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12CompilerCacheSessionVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12CompilerCacheSession * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12CompilerCacheSession * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12CompilerCacheSession * This);
        
        DECLSPEC_XFGVIRT(ID3D12CompilerFactoryChild, GetFactory)
        HRESULT ( STDMETHODCALLTYPE *GetFactory )( 
            ID3D12CompilerCacheSession * This,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppCompilerFactory);
        
        DECLSPEC_XFGVIRT(ID3D12CompilerCacheSession, FindGroup)
        HRESULT ( STDMETHODCALLTYPE *FindGroup )( 
            ID3D12CompilerCacheSession * This,
            _In_  const D3D12_COMPILER_CACHE_GROUP_KEY *pGroupKey,
            _Out_opt_  UINT *pGroupVersion);
        
        DECLSPEC_XFGVIRT(ID3D12CompilerCacheSession, FindGroupValueKeys)
        HRESULT ( STDMETHODCALLTYPE *FindGroupValueKeys )( 
            ID3D12CompilerCacheSession * This,
            _In_  const D3D12_COMPILER_CACHE_GROUP_KEY *pGroupKey,
            _In_opt_  const UINT *pExpectedGroupVersion,
            _In_  D3D12CompilerCacheSessionGroupValueKeysFunc CallbackFunc,
            _Inout_opt_  void *pContext);
        
        DECLSPEC_XFGVIRT(ID3D12CompilerCacheSession, FindGroupValues)
        HRESULT ( STDMETHODCALLTYPE *FindGroupValues )( 
            ID3D12CompilerCacheSession * This,
            _In_  const D3D12_COMPILER_CACHE_GROUP_KEY *pGroupKey,
            _In_opt_  const UINT *pExpectedGroupVersion,
            D3D12_COMPILER_VALUE_TYPE_FLAGS ValueTypeFlags,
            _In_opt_  D3D12CompilerCacheSessionGroupValuesFunc CallbackFunc,
            _Inout_opt_  void *pContext);
        
        DECLSPEC_XFGVIRT(ID3D12CompilerCacheSession, FindValue)
        HRESULT ( STDMETHODCALLTYPE *FindValue )( 
            ID3D12CompilerCacheSession * This,
            _In_  const D3D12_COMPILER_CACHE_VALUE_KEY *pValueKey,
            _Inout_count_(NumTypedValues)  D3D12_COMPILER_CACHE_TYPED_VALUE *pTypedValues,
            UINT NumTypedValues,
            _In_opt_  D3D12CompilerCacheSessionAllocationFunc pCallbackFunc,
            _Inout_opt_  void *pContext);
        
        DECLSPEC_XFGVIRT(ID3D12CompilerCacheSession, GetApplicationDesc)
        const D3D12_APPLICATION_DESC *( STDMETHODCALLTYPE *GetApplicationDesc )( 
            ID3D12CompilerCacheSession * This);
        
        DECLSPEC_XFGVIRT(ID3D12CompilerCacheSession, GetCompilerTarget)
#if !defined(_WIN32)
        D3D12_COMPILER_TARGET ( STDMETHODCALLTYPE *GetCompilerTarget )( 
            ID3D12CompilerCacheSession * This);
        
#else
        D3D12_COMPILER_TARGET *( STDMETHODCALLTYPE *GetCompilerTarget )( 
            ID3D12CompilerCacheSession * This,
            D3D12_COMPILER_TARGET * RetVal);
        
#endif
        
        DECLSPEC_XFGVIRT(ID3D12CompilerCacheSession, GetValueTypes)
        D3D12_COMPILER_VALUE_TYPE_FLAGS ( STDMETHODCALLTYPE *GetValueTypes )( 
            ID3D12CompilerCacheSession * This);
        
        DECLSPEC_XFGVIRT(ID3D12CompilerCacheSession, StoreGroupValueKeys)
        HRESULT ( STDMETHODCALLTYPE *StoreGroupValueKeys )( 
            ID3D12CompilerCacheSession * This,
            _In_  const D3D12_COMPILER_CACHE_GROUP_KEY *pGroupKey,
            UINT GroupVersion,
            _In_reads_(NumValueKeys)  const D3D12_COMPILER_CACHE_VALUE_KEY *pValueKeys,
            UINT NumValueKeys);
        
        DECLSPEC_XFGVIRT(ID3D12CompilerCacheSession, StoreValue)
        HRESULT ( STDMETHODCALLTYPE *StoreValue )( 
            ID3D12CompilerCacheSession * This,
            _In_  const D3D12_COMPILER_CACHE_VALUE_KEY *pValueKey,
            _In_reads_(NumTypedValues)  const D3D12_COMPILER_CACHE_TYPED_CONST_VALUE *pTypedValues,
            UINT NumTypedValues);
        
        END_INTERFACE
    } ID3D12CompilerCacheSessionVtbl;

    interface ID3D12CompilerCacheSession
    {
        CONST_VTBL struct ID3D12CompilerCacheSessionVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12CompilerCacheSession_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12CompilerCacheSession_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12CompilerCacheSession_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12CompilerCacheSession_GetFactory(This,riid,ppCompilerFactory)	\
    ( (This)->lpVtbl -> GetFactory(This,riid,ppCompilerFactory) ) 


#define ID3D12CompilerCacheSession_FindGroup(This,pGroupKey,pGroupVersion)	\
    ( (This)->lpVtbl -> FindGroup(This,pGroupKey,pGroupVersion) ) 

#define ID3D12CompilerCacheSession_FindGroupValueKeys(This,pGroupKey,pExpectedGroupVersion,CallbackFunc,pContext)	\
    ( (This)->lpVtbl -> FindGroupValueKeys(This,pGroupKey,pExpectedGroupVersion,CallbackFunc,pContext) ) 

#define ID3D12CompilerCacheSession_FindGroupValues(This,pGroupKey,pExpectedGroupVersion,ValueTypeFlags,CallbackFunc,pContext)	\
    ( (This)->lpVtbl -> FindGroupValues(This,pGroupKey,pExpectedGroupVersion,ValueTypeFlags,CallbackFunc,pContext) ) 

#define ID3D12CompilerCacheSession_FindValue(This,pValueKey,pTypedValues,NumTypedValues,pCallbackFunc,pContext)	\
    ( (This)->lpVtbl -> FindValue(This,pValueKey,pTypedValues,NumTypedValues,pCallbackFunc,pContext) ) 

#define ID3D12CompilerCacheSession_GetApplicationDesc(This)	\
    ( (This)->lpVtbl -> GetApplicationDesc(This) ) 
#if !defined(_WIN32)

#define ID3D12CompilerCacheSession_GetCompilerTarget(This)	\
    ( (This)->lpVtbl -> GetCompilerTarget(This) ) 
#else
#define ID3D12CompilerCacheSession_GetCompilerTarget(This,RetVal)	\
    ( (This)->lpVtbl -> GetCompilerTarget(This,RetVal) ) 
#endif

#define ID3D12CompilerCacheSession_GetValueTypes(This)	\
    ( (This)->lpVtbl -> GetValueTypes(This) ) 

#define ID3D12CompilerCacheSession_StoreGroupValueKeys(This,pGroupKey,GroupVersion,pValueKeys,NumValueKeys)	\
    ( (This)->lpVtbl -> StoreGroupValueKeys(This,pGroupKey,GroupVersion,pValueKeys,NumValueKeys) ) 

#define ID3D12CompilerCacheSession_StoreValue(This,pValueKey,pTypedValues,NumTypedValues)	\
    ( (This)->lpVtbl -> StoreValue(This,pValueKey,pTypedValues,NumTypedValues) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12CompilerCacheSession_INTERFACE_DEFINED__ */


#ifndef __ID3D12CompilerStateObject_INTERFACE_DEFINED__
#define __ID3D12CompilerStateObject_INTERFACE_DEFINED__

/* interface ID3D12CompilerStateObject */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12CompilerStateObject;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("5981cca4-e8ae-44ca-9b92-4fa86f5a3a3a")
    ID3D12CompilerStateObject : public IUnknown
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE GetCompiler( 
            _In_  REFIID riid,
            _COM_Outptr_  void **ppCompiler) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12CompilerStateObjectVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12CompilerStateObject * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12CompilerStateObject * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12CompilerStateObject * This);
        
        DECLSPEC_XFGVIRT(ID3D12CompilerStateObject, GetCompiler)
        HRESULT ( STDMETHODCALLTYPE *GetCompiler )( 
            ID3D12CompilerStateObject * This,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppCompiler);
        
        END_INTERFACE
    } ID3D12CompilerStateObjectVtbl;

    interface ID3D12CompilerStateObject
    {
        CONST_VTBL struct ID3D12CompilerStateObjectVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12CompilerStateObject_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12CompilerStateObject_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12CompilerStateObject_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12CompilerStateObject_GetCompiler(This,riid,ppCompiler)	\
    ( (This)->lpVtbl -> GetCompiler(This,riid,ppCompiler) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12CompilerStateObject_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12compiler_0000_0003 */
/* [local] */ 

typedef struct D3D12_COMPILER_EXISTING_COLLECTION_DESC
    {
    ID3D12CompilerStateObject *pExistingCollection;
    UINT NumExports;
    _In_reads_(NumExports)  const D3D12_EXPORT_DESC *pExports;
    } 	D3D12_COMPILER_EXISTING_COLLECTION_DESC;



extern RPC_IF_HANDLE __MIDL_itf_d3d12compiler_0000_0003_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12compiler_0000_0003_v0_0_s_ifspec;

#ifndef __ID3D12Compiler_INTERFACE_DEFINED__
#define __ID3D12Compiler_INTERFACE_DEFINED__

/* interface ID3D12Compiler */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12Compiler;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("8c403c12-993b-4583-80f1-6824138fa68e")
    ID3D12Compiler : public ID3D12CompilerFactoryChild
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE CompilePipelineState( 
            _In_  const D3D12_COMPILER_CACHE_GROUP_KEY *pGroupKey,
            UINT GroupVersion,
            _In_  const D3D12_PIPELINE_STATE_STREAM_DESC *pDesc) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE CompileStateObject( 
            _In_  const D3D12_COMPILER_CACHE_GROUP_KEY *pGroupKey,
            UINT GroupVersion,
            _In_  const D3D12_STATE_OBJECT_DESC *pDesc,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppCompilerStateObject) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE CompileAddToStateObject( 
            _In_  const D3D12_COMPILER_CACHE_GROUP_KEY *pGroupKey,
            UINT GroupVersion,
            _In_  const D3D12_STATE_OBJECT_DESC *pAddition,
            _In_  ID3D12CompilerStateObject *pCompilerStateObjectToGrowFrom,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppNewCompilerStateObject) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetCacheSession( 
            _In_  REFIID riid,
            _COM_Outptr_  void **ppCompilerCacheSession) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12CompilerVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12Compiler * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12Compiler * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12Compiler * This);
        
        DECLSPEC_XFGVIRT(ID3D12CompilerFactoryChild, GetFactory)
        HRESULT ( STDMETHODCALLTYPE *GetFactory )( 
            ID3D12Compiler * This,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppCompilerFactory);
        
        DECLSPEC_XFGVIRT(ID3D12Compiler, CompilePipelineState)
        HRESULT ( STDMETHODCALLTYPE *CompilePipelineState )( 
            ID3D12Compiler * This,
            _In_  const D3D12_COMPILER_CACHE_GROUP_KEY *pGroupKey,
            UINT GroupVersion,
            _In_  const D3D12_PIPELINE_STATE_STREAM_DESC *pDesc);
        
        DECLSPEC_XFGVIRT(ID3D12Compiler, CompileStateObject)
        HRESULT ( STDMETHODCALLTYPE *CompileStateObject )( 
            ID3D12Compiler * This,
            _In_  const D3D12_COMPILER_CACHE_GROUP_KEY *pGroupKey,
            UINT GroupVersion,
            _In_  const D3D12_STATE_OBJECT_DESC *pDesc,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppCompilerStateObject);
        
        DECLSPEC_XFGVIRT(ID3D12Compiler, CompileAddToStateObject)
        HRESULT ( STDMETHODCALLTYPE *CompileAddToStateObject )( 
            ID3D12Compiler * This,
            _In_  const D3D12_COMPILER_CACHE_GROUP_KEY *pGroupKey,
            UINT GroupVersion,
            _In_  const D3D12_STATE_OBJECT_DESC *pAddition,
            _In_  ID3D12CompilerStateObject *pCompilerStateObjectToGrowFrom,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppNewCompilerStateObject);
        
        DECLSPEC_XFGVIRT(ID3D12Compiler, GetCacheSession)
        HRESULT ( STDMETHODCALLTYPE *GetCacheSession )( 
            ID3D12Compiler * This,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppCompilerCacheSession);
        
        END_INTERFACE
    } ID3D12CompilerVtbl;

    interface ID3D12Compiler
    {
        CONST_VTBL struct ID3D12CompilerVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12Compiler_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12Compiler_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12Compiler_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12Compiler_GetFactory(This,riid,ppCompilerFactory)	\
    ( (This)->lpVtbl -> GetFactory(This,riid,ppCompilerFactory) ) 


#define ID3D12Compiler_CompilePipelineState(This,pGroupKey,GroupVersion,pDesc)	\
    ( (This)->lpVtbl -> CompilePipelineState(This,pGroupKey,GroupVersion,pDesc) ) 

#define ID3D12Compiler_CompileStateObject(This,pGroupKey,GroupVersion,pDesc,riid,ppCompilerStateObject)	\
    ( (This)->lpVtbl -> CompileStateObject(This,pGroupKey,GroupVersion,pDesc,riid,ppCompilerStateObject) ) 

#define ID3D12Compiler_CompileAddToStateObject(This,pGroupKey,GroupVersion,pAddition,pCompilerStateObjectToGrowFrom,riid,ppNewCompilerStateObject)	\
    ( (This)->lpVtbl -> CompileAddToStateObject(This,pGroupKey,GroupVersion,pAddition,pCompilerStateObjectToGrowFrom,riid,ppNewCompilerStateObject) ) 

#define ID3D12Compiler_GetCacheSession(This,riid,ppCompilerCacheSession)	\
    ( (This)->lpVtbl -> GetCacheSession(This,riid,ppCompilerCacheSession) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12Compiler_INTERFACE_DEFINED__ */


#ifndef __ID3D12CompilerFactory_INTERFACE_DEFINED__
#define __ID3D12CompilerFactory_INTERFACE_DEFINED__

/* interface ID3D12CompilerFactory */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12CompilerFactory;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("c1ee4b59-3f59-47a5-9b4e-a855c858a878")
    ID3D12CompilerFactory : public IUnknown
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE EnumerateAdapterFamilies( 
            UINT AdapterFamilyIndex,
            _Out_  D3D12_ADAPTER_FAMILY *pAdapterFamily) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE EnumerateAdapterFamilyABIVersions( 
            UINT AdapterFamilyIndex,
            _Inout_  UINT32 *pNumABIVersions,
            _Out_writes_opt_(*pNumABIVersions)  UINT64 *pABIVersions) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE EnumerateAdapterFamilyCompilerVersion( 
            UINT AdapterFamilyIndex,
            _Out_  D3D12_VERSION_NUMBER *pCompilerVersion) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetApplicationProfileVersion( 
            _In_  const D3D12_COMPILER_TARGET *pTarget,
            _In_  const D3D12_APPLICATION_DESC *pApplicationDesc,
            _Out_  D3D12_VERSION_NUMBER *pApplicationProfileVersion) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE CreateCompilerCacheSession( 
            _In_reads_(NumPaths)  const D3D12_COMPILER_DATABASE_PATH *pPaths,
            UINT NumPaths,
            _In_opt_  const D3D12_COMPILER_TARGET *pTarget,
            _In_opt_  const D3D12_APPLICATION_DESC *pApplicationDesc,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppCompilerCacheSession) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE CreateCompiler( 
            _In_  ID3D12CompilerCacheSession *pCompilerCacheSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppCompiler) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12CompilerFactoryVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12CompilerFactory * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12CompilerFactory * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12CompilerFactory * This);
        
        DECLSPEC_XFGVIRT(ID3D12CompilerFactory, EnumerateAdapterFamilies)
        HRESULT ( STDMETHODCALLTYPE *EnumerateAdapterFamilies )( 
            ID3D12CompilerFactory * This,
            UINT AdapterFamilyIndex,
            _Out_  D3D12_ADAPTER_FAMILY *pAdapterFamily);
        
        DECLSPEC_XFGVIRT(ID3D12CompilerFactory, EnumerateAdapterFamilyABIVersions)
        HRESULT ( STDMETHODCALLTYPE *EnumerateAdapterFamilyABIVersions )( 
            ID3D12CompilerFactory * This,
            UINT AdapterFamilyIndex,
            _Inout_  UINT32 *pNumABIVersions,
            _Out_writes_opt_(*pNumABIVersions)  UINT64 *pABIVersions);
        
        DECLSPEC_XFGVIRT(ID3D12CompilerFactory, EnumerateAdapterFamilyCompilerVersion)
        HRESULT ( STDMETHODCALLTYPE *EnumerateAdapterFamilyCompilerVersion )( 
            ID3D12CompilerFactory * This,
            UINT AdapterFamilyIndex,
            _Out_  D3D12_VERSION_NUMBER *pCompilerVersion);
        
        DECLSPEC_XFGVIRT(ID3D12CompilerFactory, GetApplicationProfileVersion)
        HRESULT ( STDMETHODCALLTYPE *GetApplicationProfileVersion )( 
            ID3D12CompilerFactory * This,
            _In_  const D3D12_COMPILER_TARGET *pTarget,
            _In_  const D3D12_APPLICATION_DESC *pApplicationDesc,
            _Out_  D3D12_VERSION_NUMBER *pApplicationProfileVersion);
        
        DECLSPEC_XFGVIRT(ID3D12CompilerFactory, CreateCompilerCacheSession)
        HRESULT ( STDMETHODCALLTYPE *CreateCompilerCacheSession )( 
            ID3D12CompilerFactory * This,
            _In_reads_(NumPaths)  const D3D12_COMPILER_DATABASE_PATH *pPaths,
            UINT NumPaths,
            _In_opt_  const D3D12_COMPILER_TARGET *pTarget,
            _In_opt_  const D3D12_APPLICATION_DESC *pApplicationDesc,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppCompilerCacheSession);
        
        DECLSPEC_XFGVIRT(ID3D12CompilerFactory, CreateCompiler)
        HRESULT ( STDMETHODCALLTYPE *CreateCompiler )( 
            ID3D12CompilerFactory * This,
            _In_  ID3D12CompilerCacheSession *pCompilerCacheSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppCompiler);
        
        END_INTERFACE
    } ID3D12CompilerFactoryVtbl;

    interface ID3D12CompilerFactory
    {
        CONST_VTBL struct ID3D12CompilerFactoryVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12CompilerFactory_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12CompilerFactory_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12CompilerFactory_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12CompilerFactory_EnumerateAdapterFamilies(This,AdapterFamilyIndex,pAdapterFamily)	\
    ( (This)->lpVtbl -> EnumerateAdapterFamilies(This,AdapterFamilyIndex,pAdapterFamily) ) 

#define ID3D12CompilerFactory_EnumerateAdapterFamilyABIVersions(This,AdapterFamilyIndex,pNumABIVersions,pABIVersions)	\
    ( (This)->lpVtbl -> EnumerateAdapterFamilyABIVersions(This,AdapterFamilyIndex,pNumABIVersions,pABIVersions) ) 

#define ID3D12CompilerFactory_EnumerateAdapterFamilyCompilerVersion(This,AdapterFamilyIndex,pCompilerVersion)	\
    ( (This)->lpVtbl -> EnumerateAdapterFamilyCompilerVersion(This,AdapterFamilyIndex,pCompilerVersion) ) 

#define ID3D12CompilerFactory_GetApplicationProfileVersion(This,pTarget,pApplicationDesc,pApplicationProfileVersion)	\
    ( (This)->lpVtbl -> GetApplicationProfileVersion(This,pTarget,pApplicationDesc,pApplicationProfileVersion) ) 

#define ID3D12CompilerFactory_CreateCompilerCacheSession(This,pPaths,NumPaths,pTarget,pApplicationDesc,riid,ppCompilerCacheSession)	\
    ( (This)->lpVtbl -> CreateCompilerCacheSession(This,pPaths,NumPaths,pTarget,pApplicationDesc,riid,ppCompilerCacheSession) ) 

#define ID3D12CompilerFactory_CreateCompiler(This,pCompilerCacheSession,riid,ppCompiler)	\
    ( (This)->lpVtbl -> CreateCompiler(This,pCompilerCacheSession,riid,ppCompiler) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12CompilerFactory_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12compiler_0000_0005 */
/* [local] */ 

#endif /* WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP | WINAPI_PARTITION_GAMES) */
#pragma endregion
DEFINE_GUID(IID_ID3D12CompilerFactoryChild,0xe0d06420,0x9f31,0x47e8,0xae,0x9a,0xdd,0x2b,0xa2,0x5a,0xc0,0xbc);
DEFINE_GUID(IID_ID3D12CompilerCacheSession,0x5704e5e6,0x054b,0x4738,0xb6,0x61,0x7b,0x0d,0x68,0xd8,0xdd,0xe2);
DEFINE_GUID(IID_ID3D12CompilerStateObject,0x5981cca4,0xe8ae,0x44ca,0x9b,0x92,0x4f,0xa8,0x6f,0x5a,0x3a,0x3a);
DEFINE_GUID(IID_ID3D12Compiler,0x8c403c12,0x993b,0x4583,0x80,0xf1,0x68,0x24,0x13,0x8f,0xa6,0x8e);
DEFINE_GUID(IID_ID3D12CompilerFactory,0xc1ee4b59,0x3f59,0x47a5,0x9b,0x4e,0xa8,0x55,0xc8,0x58,0xa8,0x78);


extern RPC_IF_HANDLE __MIDL_itf_d3d12compiler_0000_0005_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12compiler_0000_0005_v0_0_s_ifspec;

/* Additional Prototypes for ALL interfaces */

/* end of Additional Prototypes */

#ifdef __cplusplus
}
#endif

#endif


