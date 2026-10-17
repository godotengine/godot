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

#ifndef __d3dshadercacheregistration_h__
#define __d3dshadercacheregistration_h__

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

#ifndef __ID3DShaderCacheInstallerClient_FWD_DEFINED__
#define __ID3DShaderCacheInstallerClient_FWD_DEFINED__
typedef interface ID3DShaderCacheInstallerClient ID3DShaderCacheInstallerClient;

#endif 	/* __ID3DShaderCacheInstallerClient_FWD_DEFINED__ */


#ifndef __ID3DShaderCacheComponent_FWD_DEFINED__
#define __ID3DShaderCacheComponent_FWD_DEFINED__
typedef interface ID3DShaderCacheComponent ID3DShaderCacheComponent;

#endif 	/* __ID3DShaderCacheComponent_FWD_DEFINED__ */


#ifndef __ID3DShaderCacheApplication_FWD_DEFINED__
#define __ID3DShaderCacheApplication_FWD_DEFINED__
typedef interface ID3DShaderCacheApplication ID3DShaderCacheApplication;

#endif 	/* __ID3DShaderCacheApplication_FWD_DEFINED__ */


#ifndef __ID3DShaderCacheInstaller_FWD_DEFINED__
#define __ID3DShaderCacheInstaller_FWD_DEFINED__
typedef interface ID3DShaderCacheInstaller ID3DShaderCacheInstaller;

#endif 	/* __ID3DShaderCacheInstaller_FWD_DEFINED__ */


#ifndef __ID3DShaderCacheExplorer_FWD_DEFINED__
#define __ID3DShaderCacheExplorer_FWD_DEFINED__
typedef interface ID3DShaderCacheExplorer ID3DShaderCacheExplorer;

#endif 	/* __ID3DShaderCacheExplorer_FWD_DEFINED__ */


#ifndef __ID3DShaderCacheInstallerFactory_FWD_DEFINED__
#define __ID3DShaderCacheInstallerFactory_FWD_DEFINED__
typedef interface ID3DShaderCacheInstallerFactory ID3DShaderCacheInstallerFactory;

#endif 	/* __ID3DShaderCacheInstallerFactory_FWD_DEFINED__ */


/* header files for imported files */
#include "oaidl.h"
#include "ocidl.h"

#ifdef __cplusplus
extern "C"{
#endif 


/* interface __MIDL_itf_d3dshadercacheregistration_0000_0000 */
/* [local] */ 

#pragma once
DEFINE_GUID(CLSID_D3DShaderCacheInstallerFactory,    0x16195a0b, 0x607c, 0x41f1, 0xbf, 0x03, 0xc7, 0x69, 0x4d, 0x60, 0xa8, 0xd4);
typedef 
enum D3D_SHADER_CACHE_APP_REGISTRATION_SCOPE
    {
        D3D_SHADER_CACHE_APP_REGISTRATION_SCOPE_USER	= 0,
        D3D_SHADER_CACHE_APP_REGISTRATION_SCOPE_SYSTEM	= ( D3D_SHADER_CACHE_APP_REGISTRATION_SCOPE_USER + 1 ) 
    } 	D3D_SHADER_CACHE_APP_REGISTRATION_SCOPE;




extern RPC_IF_HANDLE __MIDL_itf_d3dshadercacheregistration_0000_0000_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3dshadercacheregistration_0000_0000_v0_0_s_ifspec;

#ifndef __ID3DShaderCacheInstallerClient_INTERFACE_DEFINED__
#define __ID3DShaderCacheInstallerClient_INTERFACE_DEFINED__

/* interface ID3DShaderCacheInstallerClient */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3DShaderCacheInstallerClient;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("a16ee930-d9f6-4222-a514-244473e5d266")
    ID3DShaderCacheInstallerClient
    {
    public:
        BEGIN_INTERFACE
        virtual HRESULT STDMETHODCALLTYPE GetInstallerName( 
            _Inout_  SIZE_T *pNameLength,
            _Out_writes_opt_(*pNameLength)  wchar_t *pName) = 0;
        
        virtual D3D_SHADER_CACHE_APP_REGISTRATION_SCOPE STDMETHODCALLTYPE GetInstallerScope( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE HandleDriverUpdate( 
            _In_  ID3DShaderCacheInstaller *pInstaller) = 0;
        
        END_INTERFACE
    };
    
    
#else 	/* C style interface */

    typedef struct ID3DShaderCacheInstallerClientVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheInstallerClient, GetInstallerName)
        HRESULT ( STDMETHODCALLTYPE *GetInstallerName )( 
            ID3DShaderCacheInstallerClient * This,
            _Inout_  SIZE_T *pNameLength,
            _Out_writes_opt_(*pNameLength)  wchar_t *pName);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheInstallerClient, GetInstallerScope)
        D3D_SHADER_CACHE_APP_REGISTRATION_SCOPE ( STDMETHODCALLTYPE *GetInstallerScope )( 
            ID3DShaderCacheInstallerClient * This);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheInstallerClient, HandleDriverUpdate)
        HRESULT ( STDMETHODCALLTYPE *HandleDriverUpdate )( 
            ID3DShaderCacheInstallerClient * This,
            _In_  ID3DShaderCacheInstaller *pInstaller);
        
        END_INTERFACE
    } ID3DShaderCacheInstallerClientVtbl;

    interface ID3DShaderCacheInstallerClient
    {
        CONST_VTBL struct ID3DShaderCacheInstallerClientVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3DShaderCacheInstallerClient_GetInstallerName(This,pNameLength,pName)	\
    ( (This)->lpVtbl -> GetInstallerName(This,pNameLength,pName) ) 

#define ID3DShaderCacheInstallerClient_GetInstallerScope(This)	\
    ( (This)->lpVtbl -> GetInstallerScope(This) ) 

#define ID3DShaderCacheInstallerClient_HandleDriverUpdate(This,pInstaller)	\
    ( (This)->lpVtbl -> HandleDriverUpdate(This,pInstaller) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3DShaderCacheInstallerClient_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3dshadercacheregistration_0000_0001 */
/* [local] */ 

typedef struct D3D_SHADER_CACHE_PSDB_PROPERTIES
    {
    const wchar_t *pAdapterFamily;
    const wchar_t *pPsdbPath;
    } 	D3D_SHADER_CACHE_PSDB_PROPERTIES;



extern RPC_IF_HANDLE __MIDL_itf_d3dshadercacheregistration_0000_0001_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3dshadercacheregistration_0000_0001_v0_0_s_ifspec;

#ifndef __ID3DShaderCacheComponent_INTERFACE_DEFINED__
#define __ID3DShaderCacheComponent_INTERFACE_DEFINED__

/* interface ID3DShaderCacheComponent */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3DShaderCacheComponent;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("eed1bf00-f5c7-4cf7-885c-d0f9c0cb4828")
    ID3DShaderCacheComponent : public IUnknown
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE GetComponentName( 
            _Out_  const wchar_t **pName) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetStateObjectDatabasePath( 
            _Out_  const wchar_t **pPath) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetPrecompiledCachePath( 
            _In_  const wchar_t *pAdapterFamily,
            _Inout_  const wchar_t **pPath) = 0;
        
        virtual UINT STDMETHODCALLTYPE GetPrecompiledShaderDatabaseCount( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetPrecompiledShaderDatabases( 
            UINT ArraySize,
            _Out_writes_(ArraySize)  D3D_SHADER_CACHE_PSDB_PROPERTIES *pPSDBs) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3DShaderCacheComponentVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3DShaderCacheComponent * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3DShaderCacheComponent * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3DShaderCacheComponent * This);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheComponent, GetComponentName)
        HRESULT ( STDMETHODCALLTYPE *GetComponentName )( 
            ID3DShaderCacheComponent * This,
            _Out_  const wchar_t **pName);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheComponent, GetStateObjectDatabasePath)
        HRESULT ( STDMETHODCALLTYPE *GetStateObjectDatabasePath )( 
            ID3DShaderCacheComponent * This,
            _Out_  const wchar_t **pPath);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheComponent, GetPrecompiledCachePath)
        HRESULT ( STDMETHODCALLTYPE *GetPrecompiledCachePath )( 
            ID3DShaderCacheComponent * This,
            _In_  const wchar_t *pAdapterFamily,
            _Inout_  const wchar_t **pPath);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheComponent, GetPrecompiledShaderDatabaseCount)
        UINT ( STDMETHODCALLTYPE *GetPrecompiledShaderDatabaseCount )( 
            ID3DShaderCacheComponent * This);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheComponent, GetPrecompiledShaderDatabases)
        HRESULT ( STDMETHODCALLTYPE *GetPrecompiledShaderDatabases )( 
            ID3DShaderCacheComponent * This,
            UINT ArraySize,
            _Out_writes_(ArraySize)  D3D_SHADER_CACHE_PSDB_PROPERTIES *pPSDBs);
        
        END_INTERFACE
    } ID3DShaderCacheComponentVtbl;

    interface ID3DShaderCacheComponent
    {
        CONST_VTBL struct ID3DShaderCacheComponentVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3DShaderCacheComponent_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3DShaderCacheComponent_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3DShaderCacheComponent_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3DShaderCacheComponent_GetComponentName(This,pName)	\
    ( (This)->lpVtbl -> GetComponentName(This,pName) ) 

#define ID3DShaderCacheComponent_GetStateObjectDatabasePath(This,pPath)	\
    ( (This)->lpVtbl -> GetStateObjectDatabasePath(This,pPath) ) 

#define ID3DShaderCacheComponent_GetPrecompiledCachePath(This,pAdapterFamily,pPath)	\
    ( (This)->lpVtbl -> GetPrecompiledCachePath(This,pAdapterFamily,pPath) ) 

#define ID3DShaderCacheComponent_GetPrecompiledShaderDatabaseCount(This)	\
    ( (This)->lpVtbl -> GetPrecompiledShaderDatabaseCount(This) ) 

#define ID3DShaderCacheComponent_GetPrecompiledShaderDatabases(This,ArraySize,pPSDBs)	\
    ( (This)->lpVtbl -> GetPrecompiledShaderDatabases(This,ArraySize,pPSDBs) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3DShaderCacheComponent_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3dshadercacheregistration_0000_0002 */
/* [local] */ 

typedef 
enum D3D_SHADER_CACHE_TARGET_FLAGS
    {
        D3D_SHADER_CACHE_TARGET_FLAG_NONE	= 0
    } 	D3D_SHADER_CACHE_TARGET_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS( D3D_SHADER_CACHE_TARGET_FLAGS )
typedef union D3D_VERSION_NUMBER
    {
    UINT64 Version;
    UINT16 VersionParts[ 4 ];
    } 	D3D_VERSION_NUMBER;

typedef struct D3D_SHADER_CACHE_COMPILER_PROPERTIES
    {
    wchar_t szAdapterFamily[ 128 ];
    UINT64 MinimumABISupportVersion;
    UINT64 MaximumABISupportVersion;
    D3D_VERSION_NUMBER CompilerVersion;
    D3D_VERSION_NUMBER ApplicationProfileVersion;
    } 	D3D_SHADER_CACHE_COMPILER_PROPERTIES;

typedef struct D3D_SHADER_CACHE_APPLICATION_DESC
    {
    const wchar_t *pExeFilename;
    const wchar_t *pName;
    D3D_VERSION_NUMBER Version;
    const wchar_t *pEngineName;
    D3D_VERSION_NUMBER EngineVersion;
    } 	D3D_SHADER_CACHE_APPLICATION_DESC;



extern RPC_IF_HANDLE __MIDL_itf_d3dshadercacheregistration_0000_0002_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3dshadercacheregistration_0000_0002_v0_0_s_ifspec;

#ifndef __ID3DShaderCacheApplication_INTERFACE_DEFINED__
#define __ID3DShaderCacheApplication_INTERFACE_DEFINED__

/* interface ID3DShaderCacheApplication */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3DShaderCacheApplication;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("fc688ee2-1b35-4913-93be-1ca3fa7df39e")
    ID3DShaderCacheApplication : public IUnknown
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE GetExePath( 
            _Out_  const wchar_t **pExePath) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetDesc( 
            _Out_  D3D_SHADER_CACHE_APPLICATION_DESC *pApplicationDesc) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE RegisterComponent( 
            _In_  const wchar_t *pName,
            _In_  const wchar_t *pStateObjectDBPath,
            _In_  UINT NumPSDB,
            _In_reads_(NumPSDB)  const D3D_SHADER_CACHE_PSDB_PROPERTIES *pPSDBs,
            REFIID riid,
            _COM_Outptr_  void **ppvComponent) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE RemoveComponent( 
            _In_  ID3DShaderCacheComponent *pComponent) = 0;
        
        virtual UINT STDMETHODCALLTYPE GetComponentCount( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetComponent( 
            _In_  UINT index,
            REFIID riid,
            _COM_Outptr_  void **ppvComponent) = 0;
        
        virtual UINT STDMETHODCALLTYPE GetPrecompileTargetCount( 
            D3D_SHADER_CACHE_TARGET_FLAGS flags) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetPrecompileTargets( 
            _In_  UINT ArraySize,
            _In_reads_(ArraySize)  D3D_SHADER_CACHE_COMPILER_PROPERTIES *pArray,
            D3D_SHADER_CACHE_TARGET_FLAGS flags) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetInstallerName( 
            _Out_  const wchar_t **pInstallerName) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3DShaderCacheApplicationVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3DShaderCacheApplication * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3DShaderCacheApplication * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3DShaderCacheApplication * This);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheApplication, GetExePath)
        HRESULT ( STDMETHODCALLTYPE *GetExePath )( 
            ID3DShaderCacheApplication * This,
            _Out_  const wchar_t **pExePath);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheApplication, GetDesc)
        HRESULT ( STDMETHODCALLTYPE *GetDesc )( 
            ID3DShaderCacheApplication * This,
            _Out_  D3D_SHADER_CACHE_APPLICATION_DESC *pApplicationDesc);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheApplication, RegisterComponent)
        HRESULT ( STDMETHODCALLTYPE *RegisterComponent )( 
            ID3DShaderCacheApplication * This,
            _In_  const wchar_t *pName,
            _In_  const wchar_t *pStateObjectDBPath,
            _In_  UINT NumPSDB,
            _In_reads_(NumPSDB)  const D3D_SHADER_CACHE_PSDB_PROPERTIES *pPSDBs,
            REFIID riid,
            _COM_Outptr_  void **ppvComponent);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheApplication, RemoveComponent)
        HRESULT ( STDMETHODCALLTYPE *RemoveComponent )( 
            ID3DShaderCacheApplication * This,
            _In_  ID3DShaderCacheComponent *pComponent);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheApplication, GetComponentCount)
        UINT ( STDMETHODCALLTYPE *GetComponentCount )( 
            ID3DShaderCacheApplication * This);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheApplication, GetComponent)
        HRESULT ( STDMETHODCALLTYPE *GetComponent )( 
            ID3DShaderCacheApplication * This,
            _In_  UINT index,
            REFIID riid,
            _COM_Outptr_  void **ppvComponent);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheApplication, GetPrecompileTargetCount)
        UINT ( STDMETHODCALLTYPE *GetPrecompileTargetCount )( 
            ID3DShaderCacheApplication * This,
            D3D_SHADER_CACHE_TARGET_FLAGS flags);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheApplication, GetPrecompileTargets)
        HRESULT ( STDMETHODCALLTYPE *GetPrecompileTargets )( 
            ID3DShaderCacheApplication * This,
            _In_  UINT ArraySize,
            _In_reads_(ArraySize)  D3D_SHADER_CACHE_COMPILER_PROPERTIES *pArray,
            D3D_SHADER_CACHE_TARGET_FLAGS flags);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheApplication, GetInstallerName)
        HRESULT ( STDMETHODCALLTYPE *GetInstallerName )( 
            ID3DShaderCacheApplication * This,
            _Out_  const wchar_t **pInstallerName);
        
        END_INTERFACE
    } ID3DShaderCacheApplicationVtbl;

    interface ID3DShaderCacheApplication
    {
        CONST_VTBL struct ID3DShaderCacheApplicationVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3DShaderCacheApplication_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3DShaderCacheApplication_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3DShaderCacheApplication_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3DShaderCacheApplication_GetExePath(This,pExePath)	\
    ( (This)->lpVtbl -> GetExePath(This,pExePath) ) 

#define ID3DShaderCacheApplication_GetDesc(This,pApplicationDesc)	\
    ( (This)->lpVtbl -> GetDesc(This,pApplicationDesc) ) 

#define ID3DShaderCacheApplication_RegisterComponent(This,pName,pStateObjectDBPath,NumPSDB,pPSDBs,riid,ppvComponent)	\
    ( (This)->lpVtbl -> RegisterComponent(This,pName,pStateObjectDBPath,NumPSDB,pPSDBs,riid,ppvComponent) ) 

#define ID3DShaderCacheApplication_RemoveComponent(This,pComponent)	\
    ( (This)->lpVtbl -> RemoveComponent(This,pComponent) ) 

#define ID3DShaderCacheApplication_GetComponentCount(This)	\
    ( (This)->lpVtbl -> GetComponentCount(This) ) 

#define ID3DShaderCacheApplication_GetComponent(This,index,riid,ppvComponent)	\
    ( (This)->lpVtbl -> GetComponent(This,index,riid,ppvComponent) ) 

#define ID3DShaderCacheApplication_GetPrecompileTargetCount(This,flags)	\
    ( (This)->lpVtbl -> GetPrecompileTargetCount(This,flags) ) 

#define ID3DShaderCacheApplication_GetPrecompileTargets(This,ArraySize,pArray,flags)	\
    ( (This)->lpVtbl -> GetPrecompileTargets(This,ArraySize,pArray,flags) ) 

#define ID3DShaderCacheApplication_GetInstallerName(This,pInstallerName)	\
    ( (This)->lpVtbl -> GetInstallerName(This,pInstallerName) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3DShaderCacheApplication_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3dshadercacheregistration_0000_0003 */
/* [local] */ 

typedef struct SC_HANDLE__ *SC_HANDLE;



extern RPC_IF_HANDLE __MIDL_itf_d3dshadercacheregistration_0000_0003_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3dshadercacheregistration_0000_0003_v0_0_s_ifspec;

#ifndef __ID3DShaderCacheInstaller_INTERFACE_DEFINED__
#define __ID3DShaderCacheInstaller_INTERFACE_DEFINED__

/* interface ID3DShaderCacheInstaller */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3DShaderCacheInstaller;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("bbe30de1-6318-4526-ae17-776693191bb4")
    ID3DShaderCacheInstaller : public IUnknown
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE RegisterDriverUpdateListener( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE UnregisterDriverUpdateListener( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE RegisterServiceDriverUpdateTrigger( 
            SC_HANDLE hServiceHandle) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE UnregisterServiceDriverUpdateTrigger( 
            SC_HANDLE hServiceHandle) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE RegisterApplication( 
            _In_  const wchar_t *pExePath,
            _In_  const D3D_SHADER_CACHE_APPLICATION_DESC *pApplicationDesc,
            REFIID riid,
            _COM_Outptr_  void **ppvApp) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE RemoveApplication( 
            _In_  ID3DShaderCacheApplication *pApplication) = 0;
        
        virtual UINT STDMETHODCALLTYPE GetApplicationCount( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetApplication( 
            _In_  UINT index,
            REFIID riid,
            _COM_Outptr_  void **ppvApp) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE ClearAllState( void) = 0;
        
        virtual UINT STDMETHODCALLTYPE GetMaxPrecompileTargetCount( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetPrecompileTargets( 
            _In_opt_  const D3D_SHADER_CACHE_APPLICATION_DESC *pApplicationDesc,
            _Inout_  UINT *pArraySize,
            _Out_writes_(*pArraySize)  D3D_SHADER_CACHE_COMPILER_PROPERTIES *pArray,
            D3D_SHADER_CACHE_TARGET_FLAGS flags) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3DShaderCacheInstallerVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3DShaderCacheInstaller * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3DShaderCacheInstaller * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3DShaderCacheInstaller * This);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheInstaller, RegisterDriverUpdateListener)
        HRESULT ( STDMETHODCALLTYPE *RegisterDriverUpdateListener )( 
            ID3DShaderCacheInstaller * This);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheInstaller, UnregisterDriverUpdateListener)
        HRESULT ( STDMETHODCALLTYPE *UnregisterDriverUpdateListener )( 
            ID3DShaderCacheInstaller * This);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheInstaller, RegisterServiceDriverUpdateTrigger)
        HRESULT ( STDMETHODCALLTYPE *RegisterServiceDriverUpdateTrigger )( 
            ID3DShaderCacheInstaller * This,
            SC_HANDLE hServiceHandle);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheInstaller, UnregisterServiceDriverUpdateTrigger)
        HRESULT ( STDMETHODCALLTYPE *UnregisterServiceDriverUpdateTrigger )( 
            ID3DShaderCacheInstaller * This,
            SC_HANDLE hServiceHandle);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheInstaller, RegisterApplication)
        HRESULT ( STDMETHODCALLTYPE *RegisterApplication )( 
            ID3DShaderCacheInstaller * This,
            _In_  const wchar_t *pExePath,
            _In_  const D3D_SHADER_CACHE_APPLICATION_DESC *pApplicationDesc,
            REFIID riid,
            _COM_Outptr_  void **ppvApp);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheInstaller, RemoveApplication)
        HRESULT ( STDMETHODCALLTYPE *RemoveApplication )( 
            ID3DShaderCacheInstaller * This,
            _In_  ID3DShaderCacheApplication *pApplication);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheInstaller, GetApplicationCount)
        UINT ( STDMETHODCALLTYPE *GetApplicationCount )( 
            ID3DShaderCacheInstaller * This);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheInstaller, GetApplication)
        HRESULT ( STDMETHODCALLTYPE *GetApplication )( 
            ID3DShaderCacheInstaller * This,
            _In_  UINT index,
            REFIID riid,
            _COM_Outptr_  void **ppvApp);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheInstaller, ClearAllState)
        HRESULT ( STDMETHODCALLTYPE *ClearAllState )( 
            ID3DShaderCacheInstaller * This);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheInstaller, GetMaxPrecompileTargetCount)
        UINT ( STDMETHODCALLTYPE *GetMaxPrecompileTargetCount )( 
            ID3DShaderCacheInstaller * This);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheInstaller, GetPrecompileTargets)
        HRESULT ( STDMETHODCALLTYPE *GetPrecompileTargets )( 
            ID3DShaderCacheInstaller * This,
            _In_opt_  const D3D_SHADER_CACHE_APPLICATION_DESC *pApplicationDesc,
            _Inout_  UINT *pArraySize,
            _Out_writes_(*pArraySize)  D3D_SHADER_CACHE_COMPILER_PROPERTIES *pArray,
            D3D_SHADER_CACHE_TARGET_FLAGS flags);
        
        END_INTERFACE
    } ID3DShaderCacheInstallerVtbl;

    interface ID3DShaderCacheInstaller
    {
        CONST_VTBL struct ID3DShaderCacheInstallerVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3DShaderCacheInstaller_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3DShaderCacheInstaller_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3DShaderCacheInstaller_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3DShaderCacheInstaller_RegisterDriverUpdateListener(This)	\
    ( (This)->lpVtbl -> RegisterDriverUpdateListener(This) ) 

#define ID3DShaderCacheInstaller_UnregisterDriverUpdateListener(This)	\
    ( (This)->lpVtbl -> UnregisterDriverUpdateListener(This) ) 

#define ID3DShaderCacheInstaller_RegisterServiceDriverUpdateTrigger(This,hServiceHandle)	\
    ( (This)->lpVtbl -> RegisterServiceDriverUpdateTrigger(This,hServiceHandle) ) 

#define ID3DShaderCacheInstaller_UnregisterServiceDriverUpdateTrigger(This,hServiceHandle)	\
    ( (This)->lpVtbl -> UnregisterServiceDriverUpdateTrigger(This,hServiceHandle) ) 

#define ID3DShaderCacheInstaller_RegisterApplication(This,pExePath,pApplicationDesc,riid,ppvApp)	\
    ( (This)->lpVtbl -> RegisterApplication(This,pExePath,pApplicationDesc,riid,ppvApp) ) 

#define ID3DShaderCacheInstaller_RemoveApplication(This,pApplication)	\
    ( (This)->lpVtbl -> RemoveApplication(This,pApplication) ) 

#define ID3DShaderCacheInstaller_GetApplicationCount(This)	\
    ( (This)->lpVtbl -> GetApplicationCount(This) ) 

#define ID3DShaderCacheInstaller_GetApplication(This,index,riid,ppvApp)	\
    ( (This)->lpVtbl -> GetApplication(This,index,riid,ppvApp) ) 

#define ID3DShaderCacheInstaller_ClearAllState(This)	\
    ( (This)->lpVtbl -> ClearAllState(This) ) 

#define ID3DShaderCacheInstaller_GetMaxPrecompileTargetCount(This)	\
    ( (This)->lpVtbl -> GetMaxPrecompileTargetCount(This) ) 

#define ID3DShaderCacheInstaller_GetPrecompileTargets(This,pApplicationDesc,pArraySize,pArray,flags)	\
    ( (This)->lpVtbl -> GetPrecompileTargets(This,pApplicationDesc,pArraySize,pArray,flags) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3DShaderCacheInstaller_INTERFACE_DEFINED__ */


#ifndef __ID3DShaderCacheExplorer_INTERFACE_DEFINED__
#define __ID3DShaderCacheExplorer_INTERFACE_DEFINED__

/* interface ID3DShaderCacheExplorer */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3DShaderCacheExplorer;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("90432322-32f5-487f-9264-e9390fa58b2a")
    ID3DShaderCacheExplorer : public IUnknown
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE GetApplicationFromExePath( 
            _In_  const wchar_t *pFullExePath,
            REFIID riid,
            _COM_Outptr_  void **ppvApp) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3DShaderCacheExplorerVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3DShaderCacheExplorer * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3DShaderCacheExplorer * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3DShaderCacheExplorer * This);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheExplorer, GetApplicationFromExePath)
        HRESULT ( STDMETHODCALLTYPE *GetApplicationFromExePath )( 
            ID3DShaderCacheExplorer * This,
            _In_  const wchar_t *pFullExePath,
            REFIID riid,
            _COM_Outptr_  void **ppvApp);
        
        END_INTERFACE
    } ID3DShaderCacheExplorerVtbl;

    interface ID3DShaderCacheExplorer
    {
        CONST_VTBL struct ID3DShaderCacheExplorerVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3DShaderCacheExplorer_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3DShaderCacheExplorer_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3DShaderCacheExplorer_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3DShaderCacheExplorer_GetApplicationFromExePath(This,pFullExePath,riid,ppvApp)	\
    ( (This)->lpVtbl -> GetApplicationFromExePath(This,pFullExePath,riid,ppvApp) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3DShaderCacheExplorer_INTERFACE_DEFINED__ */


#ifndef __ID3DShaderCacheInstallerFactory_INTERFACE_DEFINED__
#define __ID3DShaderCacheInstallerFactory_INTERFACE_DEFINED__

/* interface ID3DShaderCacheInstallerFactory */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3DShaderCacheInstallerFactory;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("09b2dfe4-840f-401a-804c-0dd8aadc9e9f")
    ID3DShaderCacheInstallerFactory : public IUnknown
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE CreateInstaller( 
            _In_  ID3DShaderCacheInstallerClient *pClient,
            REFIID riid,
            _COM_Outptr_  void **ppvInstaller) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE CreateExplorer( 
            IUnknown *pUnknown,
            REFIID riid,
            _COM_Outptr_  void **ppvExplorer) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3DShaderCacheInstallerFactoryVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3DShaderCacheInstallerFactory * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3DShaderCacheInstallerFactory * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3DShaderCacheInstallerFactory * This);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheInstallerFactory, CreateInstaller)
        HRESULT ( STDMETHODCALLTYPE *CreateInstaller )( 
            ID3DShaderCacheInstallerFactory * This,
            _In_  ID3DShaderCacheInstallerClient *pClient,
            REFIID riid,
            _COM_Outptr_  void **ppvInstaller);
        
        DECLSPEC_XFGVIRT(ID3DShaderCacheInstallerFactory, CreateExplorer)
        HRESULT ( STDMETHODCALLTYPE *CreateExplorer )( 
            ID3DShaderCacheInstallerFactory * This,
            IUnknown *pUnknown,
            REFIID riid,
            _COM_Outptr_  void **ppvExplorer);
        
        END_INTERFACE
    } ID3DShaderCacheInstallerFactoryVtbl;

    interface ID3DShaderCacheInstallerFactory
    {
        CONST_VTBL struct ID3DShaderCacheInstallerFactoryVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3DShaderCacheInstallerFactory_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3DShaderCacheInstallerFactory_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3DShaderCacheInstallerFactory_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3DShaderCacheInstallerFactory_CreateInstaller(This,pClient,riid,ppvInstaller)	\
    ( (This)->lpVtbl -> CreateInstaller(This,pClient,riid,ppvInstaller) ) 

#define ID3DShaderCacheInstallerFactory_CreateExplorer(This,pUnknown,riid,ppvExplorer)	\
    ( (This)->lpVtbl -> CreateExplorer(This,pUnknown,riid,ppvExplorer) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3DShaderCacheInstallerFactory_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3dshadercacheregistration_0000_0006 */
/* [local] */ 

DEFINE_GUID(IID_ID3DShaderCacheInstallerClient,0xa16ee930,0xd9f6,0x4222,0xa5,0x14,0x24,0x44,0x73,0xe5,0xd2,0x66);
DEFINE_GUID(IID_ID3DShaderCacheComponent,0xeed1bf00,0xf5c7,0x4cf7,0x88,0x5c,0xd0,0xf9,0xc0,0xcb,0x48,0x28);
DEFINE_GUID(IID_ID3DShaderCacheApplication,0xfc688ee2,0x1b35,0x4913,0x93,0xbe,0x1c,0xa3,0xfa,0x7d,0xf3,0x9e);
DEFINE_GUID(IID_ID3DShaderCacheInstaller,0xbbe30de1,0x6318,0x4526,0xae,0x17,0x77,0x66,0x93,0x19,0x1b,0xb4);
DEFINE_GUID(IID_ID3DShaderCacheExplorer,0x90432322,0x32f5,0x487f,0x92,0x64,0xe9,0x39,0x0f,0xa5,0x8b,0x2a);
DEFINE_GUID(IID_ID3DShaderCacheInstallerFactory,0x09b2dfe4,0x840f,0x401a,0x80,0x4c,0x0d,0xd8,0xaa,0xdc,0x9e,0x9f);


extern RPC_IF_HANDLE __MIDL_itf_d3dshadercacheregistration_0000_0006_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3dshadercacheregistration_0000_0006_v0_0_s_ifspec;

/* Additional Prototypes for ALL interfaces */

/* end of Additional Prototypes */

#ifdef __cplusplus
}
#endif

#endif


