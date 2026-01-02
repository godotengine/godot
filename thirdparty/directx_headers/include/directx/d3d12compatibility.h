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

#ifndef __d3d12compatibility_h__
#define __d3d12compatibility_h__

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

#ifndef __ID3D12CompatibilityDevice_FWD_DEFINED__
#define __ID3D12CompatibilityDevice_FWD_DEFINED__
typedef interface ID3D12CompatibilityDevice ID3D12CompatibilityDevice;

#endif 	/* __ID3D12CompatibilityDevice_FWD_DEFINED__ */


#ifndef __D3D11On12CreatorID_FWD_DEFINED__
#define __D3D11On12CreatorID_FWD_DEFINED__
typedef interface D3D11On12CreatorID D3D11On12CreatorID;

#endif 	/* __D3D11On12CreatorID_FWD_DEFINED__ */


#ifndef __D3D9On12CreatorID_FWD_DEFINED__
#define __D3D9On12CreatorID_FWD_DEFINED__
typedef interface D3D9On12CreatorID D3D9On12CreatorID;

#endif 	/* __D3D9On12CreatorID_FWD_DEFINED__ */


#ifndef __OpenGLOn12CreatorID_FWD_DEFINED__
#define __OpenGLOn12CreatorID_FWD_DEFINED__
typedef interface OpenGLOn12CreatorID OpenGLOn12CreatorID;

#endif 	/* __OpenGLOn12CreatorID_FWD_DEFINED__ */


#ifndef __OpenCLOn12CreatorID_FWD_DEFINED__
#define __OpenCLOn12CreatorID_FWD_DEFINED__
typedef interface OpenCLOn12CreatorID OpenCLOn12CreatorID;

#endif 	/* __OpenCLOn12CreatorID_FWD_DEFINED__ */


#ifndef __VulkanOn12CreatorID_FWD_DEFINED__
#define __VulkanOn12CreatorID_FWD_DEFINED__
typedef interface VulkanOn12CreatorID VulkanOn12CreatorID;

#endif 	/* __VulkanOn12CreatorID_FWD_DEFINED__ */


#ifndef __DirectMLTensorFlowCreatorID_FWD_DEFINED__
#define __DirectMLTensorFlowCreatorID_FWD_DEFINED__
typedef interface DirectMLTensorFlowCreatorID DirectMLTensorFlowCreatorID;

#endif 	/* __DirectMLTensorFlowCreatorID_FWD_DEFINED__ */


#ifndef __DirectMLPyTorchCreatorID_FWD_DEFINED__
#define __DirectMLPyTorchCreatorID_FWD_DEFINED__
typedef interface DirectMLPyTorchCreatorID DirectMLPyTorchCreatorID;

#endif 	/* __DirectMLPyTorchCreatorID_FWD_DEFINED__ */


#ifndef __DirectMLWebNNCreatorID_FWD_DEFINED__
#define __DirectMLWebNNCreatorID_FWD_DEFINED__
typedef interface DirectMLWebNNCreatorID DirectMLWebNNCreatorID;

#endif 	/* __DirectMLWebNNCreatorID_FWD_DEFINED__ */


/* header files for imported files */
#include "oaidl.h"
#include "ocidl.h"
#include "d3d11on12.h"

#ifdef __cplusplus
extern "C"{
#endif 


/* interface __MIDL_itf_d3d12compatibility_0000_0000 */
/* [local] */ 

#include <winapifamily.h>
#pragma region Desktop Family
#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP | WINAPI_PARTITION_GAMES)
typedef 
enum D3D12_COMPATIBILITY_SHARED_FLAGS
    {
        D3D12_COMPATIBILITY_SHARED_FLAG_NONE	= 0,
        D3D12_COMPATIBILITY_SHARED_FLAG_NON_NT_HANDLE	= 0x1,
        D3D12_COMPATIBILITY_SHARED_FLAG_KEYED_MUTEX	= 0x2,
        D3D12_COMPATIBILITY_SHARED_FLAG_9_ON_12	= 0x4
    } 	D3D12_COMPATIBILITY_SHARED_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS( D3D12_COMPATIBILITY_SHARED_FLAGS )
typedef 
enum D3D12_REFLECT_SHARED_PROPERTY
    {
        D3D12_REFLECT_SHARED_PROPERTY_D3D11_RESOURCE_FLAGS	= 0,
        D3D12_REFELCT_SHARED_PROPERTY_COMPATIBILITY_SHARED_FLAGS	= ( D3D12_REFLECT_SHARED_PROPERTY_D3D11_RESOURCE_FLAGS + 1 ) ,
        D3D12_REFLECT_SHARED_PROPERTY_NON_NT_SHARED_HANDLE	= ( D3D12_REFELCT_SHARED_PROPERTY_COMPATIBILITY_SHARED_FLAGS + 1 ) 
    } 	D3D12_REFLECT_SHARED_PROPERTY;



extern RPC_IF_HANDLE __MIDL_itf_d3d12compatibility_0000_0000_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12compatibility_0000_0000_v0_0_s_ifspec;

#ifndef __ID3D12CompatibilityDevice_INTERFACE_DEFINED__
#define __ID3D12CompatibilityDevice_INTERFACE_DEFINED__

/* interface ID3D12CompatibilityDevice */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12CompatibilityDevice;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("8f1c0e3c-fae3-4a82-b098-bfe1708207ff")
    ID3D12CompatibilityDevice : public IUnknown
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE CreateSharedResource( 
            _In_  const D3D12_HEAP_PROPERTIES *pHeapProperties,
            D3D12_HEAP_FLAGS HeapFlags,
            _In_  const D3D12_RESOURCE_DESC *pDesc,
            D3D12_RESOURCE_STATES InitialResourceState,
            _In_opt_  const D3D12_CLEAR_VALUE *pOptimizedClearValue,
            _In_opt_  const D3D11_RESOURCE_FLAGS *pFlags11,
            D3D12_COMPATIBILITY_SHARED_FLAGS CompatibilityFlags,
            _In_opt_  ID3D12LifetimeTracker *pLifetimeTracker,
            _In_opt_  ID3D12SwapChainAssistant *pOwningSwapchain,
            REFIID riid,
            _COM_Outptr_opt_  void **ppResource) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE CreateSharedHeap( 
            _In_  const D3D12_HEAP_DESC *pHeapDesc,
            D3D12_COMPATIBILITY_SHARED_FLAGS CompatibilityFlags,
            REFIID riid,
            _COM_Outptr_opt_  void **ppHeap) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE ReflectSharedProperties( 
            _In_  ID3D12Object *pHeapOrResource,
            D3D12_REFLECT_SHARED_PROPERTY ReflectType,
            _Out_writes_bytes_(DataSize)  void *pData,
            UINT DataSize) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12CompatibilityDeviceVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12CompatibilityDevice * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12CompatibilityDevice * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12CompatibilityDevice * This);
        
        DECLSPEC_XFGVIRT(ID3D12CompatibilityDevice, CreateSharedResource)
        HRESULT ( STDMETHODCALLTYPE *CreateSharedResource )( 
            ID3D12CompatibilityDevice * This,
            _In_  const D3D12_HEAP_PROPERTIES *pHeapProperties,
            D3D12_HEAP_FLAGS HeapFlags,
            _In_  const D3D12_RESOURCE_DESC *pDesc,
            D3D12_RESOURCE_STATES InitialResourceState,
            _In_opt_  const D3D12_CLEAR_VALUE *pOptimizedClearValue,
            _In_opt_  const D3D11_RESOURCE_FLAGS *pFlags11,
            D3D12_COMPATIBILITY_SHARED_FLAGS CompatibilityFlags,
            _In_opt_  ID3D12LifetimeTracker *pLifetimeTracker,
            _In_opt_  ID3D12SwapChainAssistant *pOwningSwapchain,
            REFIID riid,
            _COM_Outptr_opt_  void **ppResource);
        
        DECLSPEC_XFGVIRT(ID3D12CompatibilityDevice, CreateSharedHeap)
        HRESULT ( STDMETHODCALLTYPE *CreateSharedHeap )( 
            ID3D12CompatibilityDevice * This,
            _In_  const D3D12_HEAP_DESC *pHeapDesc,
            D3D12_COMPATIBILITY_SHARED_FLAGS CompatibilityFlags,
            REFIID riid,
            _COM_Outptr_opt_  void **ppHeap);
        
        DECLSPEC_XFGVIRT(ID3D12CompatibilityDevice, ReflectSharedProperties)
        HRESULT ( STDMETHODCALLTYPE *ReflectSharedProperties )( 
            ID3D12CompatibilityDevice * This,
            _In_  ID3D12Object *pHeapOrResource,
            D3D12_REFLECT_SHARED_PROPERTY ReflectType,
            _Out_writes_bytes_(DataSize)  void *pData,
            UINT DataSize);
        
        END_INTERFACE
    } ID3D12CompatibilityDeviceVtbl;

    interface ID3D12CompatibilityDevice
    {
        CONST_VTBL struct ID3D12CompatibilityDeviceVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12CompatibilityDevice_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12CompatibilityDevice_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12CompatibilityDevice_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12CompatibilityDevice_CreateSharedResource(This,pHeapProperties,HeapFlags,pDesc,InitialResourceState,pOptimizedClearValue,pFlags11,CompatibilityFlags,pLifetimeTracker,pOwningSwapchain,riid,ppResource)	\
    ( (This)->lpVtbl -> CreateSharedResource(This,pHeapProperties,HeapFlags,pDesc,InitialResourceState,pOptimizedClearValue,pFlags11,CompatibilityFlags,pLifetimeTracker,pOwningSwapchain,riid,ppResource) ) 

#define ID3D12CompatibilityDevice_CreateSharedHeap(This,pHeapDesc,CompatibilityFlags,riid,ppHeap)	\
    ( (This)->lpVtbl -> CreateSharedHeap(This,pHeapDesc,CompatibilityFlags,riid,ppHeap) ) 

#define ID3D12CompatibilityDevice_ReflectSharedProperties(This,pHeapOrResource,ReflectType,pData,DataSize)	\
    ( (This)->lpVtbl -> ReflectSharedProperties(This,pHeapOrResource,ReflectType,pData,DataSize) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12CompatibilityDevice_INTERFACE_DEFINED__ */


#ifndef __D3D11On12CreatorID_INTERFACE_DEFINED__
#define __D3D11On12CreatorID_INTERFACE_DEFINED__

/* interface D3D11On12CreatorID */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_D3D11On12CreatorID;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("edbf5678-2960-4e81-8429-99d4b2630c4e")
    D3D11On12CreatorID : public IUnknown
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct D3D11On12CreatorIDVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            D3D11On12CreatorID * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            D3D11On12CreatorID * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            D3D11On12CreatorID * This);
        
        END_INTERFACE
    } D3D11On12CreatorIDVtbl;

    interface D3D11On12CreatorID
    {
        CONST_VTBL struct D3D11On12CreatorIDVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define D3D11On12CreatorID_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define D3D11On12CreatorID_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define D3D11On12CreatorID_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __D3D11On12CreatorID_INTERFACE_DEFINED__ */


#ifndef __D3D9On12CreatorID_INTERFACE_DEFINED__
#define __D3D9On12CreatorID_INTERFACE_DEFINED__

/* interface D3D9On12CreatorID */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_D3D9On12CreatorID;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("fffcbb7f-15d3-42a2-841e-9d8d32f37ddd")
    D3D9On12CreatorID : public IUnknown
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct D3D9On12CreatorIDVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            D3D9On12CreatorID * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            D3D9On12CreatorID * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            D3D9On12CreatorID * This);
        
        END_INTERFACE
    } D3D9On12CreatorIDVtbl;

    interface D3D9On12CreatorID
    {
        CONST_VTBL struct D3D9On12CreatorIDVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define D3D9On12CreatorID_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define D3D9On12CreatorID_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define D3D9On12CreatorID_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __D3D9On12CreatorID_INTERFACE_DEFINED__ */


#ifndef __OpenGLOn12CreatorID_INTERFACE_DEFINED__
#define __OpenGLOn12CreatorID_INTERFACE_DEFINED__

/* interface OpenGLOn12CreatorID */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_OpenGLOn12CreatorID;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("6bb3cd34-0d19-45ab-97ed-d720ba3dfc80")
    OpenGLOn12CreatorID : public IUnknown
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct OpenGLOn12CreatorIDVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            OpenGLOn12CreatorID * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            OpenGLOn12CreatorID * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            OpenGLOn12CreatorID * This);
        
        END_INTERFACE
    } OpenGLOn12CreatorIDVtbl;

    interface OpenGLOn12CreatorID
    {
        CONST_VTBL struct OpenGLOn12CreatorIDVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define OpenGLOn12CreatorID_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define OpenGLOn12CreatorID_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define OpenGLOn12CreatorID_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __OpenGLOn12CreatorID_INTERFACE_DEFINED__ */


#ifndef __OpenCLOn12CreatorID_INTERFACE_DEFINED__
#define __OpenCLOn12CreatorID_INTERFACE_DEFINED__

/* interface OpenCLOn12CreatorID */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_OpenCLOn12CreatorID;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("3f76bb74-91b5-4a88-b126-20ca0331cd60")
    OpenCLOn12CreatorID : public IUnknown
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct OpenCLOn12CreatorIDVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            OpenCLOn12CreatorID * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            OpenCLOn12CreatorID * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            OpenCLOn12CreatorID * This);
        
        END_INTERFACE
    } OpenCLOn12CreatorIDVtbl;

    interface OpenCLOn12CreatorID
    {
        CONST_VTBL struct OpenCLOn12CreatorIDVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define OpenCLOn12CreatorID_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define OpenCLOn12CreatorID_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define OpenCLOn12CreatorID_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __OpenCLOn12CreatorID_INTERFACE_DEFINED__ */


#ifndef __VulkanOn12CreatorID_INTERFACE_DEFINED__
#define __VulkanOn12CreatorID_INTERFACE_DEFINED__

/* interface VulkanOn12CreatorID */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_VulkanOn12CreatorID;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("bc806e01-3052-406c-a3e8-9fc07f048f98")
    VulkanOn12CreatorID : public IUnknown
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct VulkanOn12CreatorIDVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            VulkanOn12CreatorID * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            VulkanOn12CreatorID * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            VulkanOn12CreatorID * This);
        
        END_INTERFACE
    } VulkanOn12CreatorIDVtbl;

    interface VulkanOn12CreatorID
    {
        CONST_VTBL struct VulkanOn12CreatorIDVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define VulkanOn12CreatorID_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define VulkanOn12CreatorID_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define VulkanOn12CreatorID_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __VulkanOn12CreatorID_INTERFACE_DEFINED__ */


#ifndef __DirectMLTensorFlowCreatorID_INTERFACE_DEFINED__
#define __DirectMLTensorFlowCreatorID_INTERFACE_DEFINED__

/* interface DirectMLTensorFlowCreatorID */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_DirectMLTensorFlowCreatorID;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("cb7490ac-8a0f-44ec-9b7b-6f4cafe8e9ab")
    DirectMLTensorFlowCreatorID : public IUnknown
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct DirectMLTensorFlowCreatorIDVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            DirectMLTensorFlowCreatorID * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            DirectMLTensorFlowCreatorID * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            DirectMLTensorFlowCreatorID * This);
        
        END_INTERFACE
    } DirectMLTensorFlowCreatorIDVtbl;

    interface DirectMLTensorFlowCreatorID
    {
        CONST_VTBL struct DirectMLTensorFlowCreatorIDVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define DirectMLTensorFlowCreatorID_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define DirectMLTensorFlowCreatorID_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define DirectMLTensorFlowCreatorID_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __DirectMLTensorFlowCreatorID_INTERFACE_DEFINED__ */


#ifndef __DirectMLPyTorchCreatorID_INTERFACE_DEFINED__
#define __DirectMLPyTorchCreatorID_INTERFACE_DEFINED__

/* interface DirectMLPyTorchCreatorID */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_DirectMLPyTorchCreatorID;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("af029192-fba1-4b05-9116-235e06560354")
    DirectMLPyTorchCreatorID : public IUnknown
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct DirectMLPyTorchCreatorIDVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            DirectMLPyTorchCreatorID * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            DirectMLPyTorchCreatorID * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            DirectMLPyTorchCreatorID * This);
        
        END_INTERFACE
    } DirectMLPyTorchCreatorIDVtbl;

    interface DirectMLPyTorchCreatorID
    {
        CONST_VTBL struct DirectMLPyTorchCreatorIDVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define DirectMLPyTorchCreatorID_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define DirectMLPyTorchCreatorID_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define DirectMLPyTorchCreatorID_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __DirectMLPyTorchCreatorID_INTERFACE_DEFINED__ */


#ifndef __DirectMLWebNNCreatorID_INTERFACE_DEFINED__
#define __DirectMLWebNNCreatorID_INTERFACE_DEFINED__

/* interface DirectMLWebNNCreatorID */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_DirectMLWebNNCreatorID;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("fdf01a76-1e11-450f-902b-74f04ea08094")
    DirectMLWebNNCreatorID : public IUnknown
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct DirectMLWebNNCreatorIDVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            DirectMLWebNNCreatorID * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            DirectMLWebNNCreatorID * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            DirectMLWebNNCreatorID * This);
        
        END_INTERFACE
    } DirectMLWebNNCreatorIDVtbl;

    interface DirectMLWebNNCreatorID
    {
        CONST_VTBL struct DirectMLWebNNCreatorIDVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define DirectMLWebNNCreatorID_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define DirectMLWebNNCreatorID_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define DirectMLWebNNCreatorID_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __DirectMLWebNNCreatorID_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12compatibility_0000_0009 */
/* [local] */ 

#endif /* WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP | WINAPI_PARTITION_GAMES) */
#pragma endregion
DEFINE_GUID(IID_ID3D12CompatibilityDevice,0x8f1c0e3c,0xfae3,0x4a82,0xb0,0x98,0xbf,0xe1,0x70,0x82,0x07,0xff);
DEFINE_GUID(IID_D3D11On12CreatorID,0xedbf5678,0x2960,0x4e81,0x84,0x29,0x99,0xd4,0xb2,0x63,0x0c,0x4e);
DEFINE_GUID(IID_D3D9On12CreatorID,0xfffcbb7f,0x15d3,0x42a2,0x84,0x1e,0x9d,0x8d,0x32,0xf3,0x7d,0xdd);
DEFINE_GUID(IID_OpenGLOn12CreatorID,0x6bb3cd34,0x0d19,0x45ab,0x97,0xed,0xd7,0x20,0xba,0x3d,0xfc,0x80);
DEFINE_GUID(IID_OpenCLOn12CreatorID,0x3f76bb74,0x91b5,0x4a88,0xb1,0x26,0x20,0xca,0x03,0x31,0xcd,0x60);
DEFINE_GUID(IID_VulkanOn12CreatorID,0xbc806e01,0x3052,0x406c,0xa3,0xe8,0x9f,0xc0,0x7f,0x04,0x8f,0x98);
DEFINE_GUID(IID_DirectMLTensorFlowCreatorID,0xcb7490ac,0x8a0f,0x44ec,0x9b,0x7b,0x6f,0x4c,0xaf,0xe8,0xe9,0xab);
DEFINE_GUID(IID_DirectMLPyTorchCreatorID,0xaf029192,0xfba1,0x4b05,0x91,0x16,0x23,0x5e,0x06,0x56,0x03,0x54);
DEFINE_GUID(IID_DirectMLWebNNCreatorID,0xfdf01a76,0x1e11,0x450f,0x90,0x2b,0x74,0xf0,0x4e,0xa0,0x80,0x94);


extern RPC_IF_HANDLE __MIDL_itf_d3d12compatibility_0000_0009_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12compatibility_0000_0009_v0_0_s_ifspec;

/* Additional Prototypes for ALL interfaces */

/* end of Additional Prototypes */

#ifdef __cplusplus
}
#endif

#endif


