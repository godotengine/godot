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

#ifndef __d3d12sdklayers_h__
#define __d3d12sdklayers_h__

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

#ifndef __ID3D12Debug_FWD_DEFINED__
#define __ID3D12Debug_FWD_DEFINED__
typedef interface ID3D12Debug ID3D12Debug;

#endif 	/* __ID3D12Debug_FWD_DEFINED__ */


#ifndef __ID3D12Debug1_FWD_DEFINED__
#define __ID3D12Debug1_FWD_DEFINED__
typedef interface ID3D12Debug1 ID3D12Debug1;

#endif 	/* __ID3D12Debug1_FWD_DEFINED__ */


#ifndef __ID3D12Debug2_FWD_DEFINED__
#define __ID3D12Debug2_FWD_DEFINED__
typedef interface ID3D12Debug2 ID3D12Debug2;

#endif 	/* __ID3D12Debug2_FWD_DEFINED__ */


#ifndef __ID3D12Debug3_FWD_DEFINED__
#define __ID3D12Debug3_FWD_DEFINED__
typedef interface ID3D12Debug3 ID3D12Debug3;

#endif 	/* __ID3D12Debug3_FWD_DEFINED__ */


#ifndef __ID3D12Debug4_FWD_DEFINED__
#define __ID3D12Debug4_FWD_DEFINED__
typedef interface ID3D12Debug4 ID3D12Debug4;

#endif 	/* __ID3D12Debug4_FWD_DEFINED__ */


#ifndef __ID3D12Debug5_FWD_DEFINED__
#define __ID3D12Debug5_FWD_DEFINED__
typedef interface ID3D12Debug5 ID3D12Debug5;

#endif 	/* __ID3D12Debug5_FWD_DEFINED__ */


#ifndef __ID3D12Debug6_FWD_DEFINED__
#define __ID3D12Debug6_FWD_DEFINED__
typedef interface ID3D12Debug6 ID3D12Debug6;

#endif 	/* __ID3D12Debug6_FWD_DEFINED__ */


#ifndef __ID3D12DebugDevice1_FWD_DEFINED__
#define __ID3D12DebugDevice1_FWD_DEFINED__
typedef interface ID3D12DebugDevice1 ID3D12DebugDevice1;

#endif 	/* __ID3D12DebugDevice1_FWD_DEFINED__ */


#ifndef __ID3D12DebugDevice_FWD_DEFINED__
#define __ID3D12DebugDevice_FWD_DEFINED__
typedef interface ID3D12DebugDevice ID3D12DebugDevice;

#endif 	/* __ID3D12DebugDevice_FWD_DEFINED__ */


#ifndef __ID3D12DebugDevice2_FWD_DEFINED__
#define __ID3D12DebugDevice2_FWD_DEFINED__
typedef interface ID3D12DebugDevice2 ID3D12DebugDevice2;

#endif 	/* __ID3D12DebugDevice2_FWD_DEFINED__ */


#ifndef __ID3D12DebugCommandQueue_FWD_DEFINED__
#define __ID3D12DebugCommandQueue_FWD_DEFINED__
typedef interface ID3D12DebugCommandQueue ID3D12DebugCommandQueue;

#endif 	/* __ID3D12DebugCommandQueue_FWD_DEFINED__ */


#ifndef __ID3D12DebugCommandQueue1_FWD_DEFINED__
#define __ID3D12DebugCommandQueue1_FWD_DEFINED__
typedef interface ID3D12DebugCommandQueue1 ID3D12DebugCommandQueue1;

#endif 	/* __ID3D12DebugCommandQueue1_FWD_DEFINED__ */


#ifndef __ID3D12DebugCommandList1_FWD_DEFINED__
#define __ID3D12DebugCommandList1_FWD_DEFINED__
typedef interface ID3D12DebugCommandList1 ID3D12DebugCommandList1;

#endif 	/* __ID3D12DebugCommandList1_FWD_DEFINED__ */


#ifndef __ID3D12DebugCommandList_FWD_DEFINED__
#define __ID3D12DebugCommandList_FWD_DEFINED__
typedef interface ID3D12DebugCommandList ID3D12DebugCommandList;

#endif 	/* __ID3D12DebugCommandList_FWD_DEFINED__ */


#ifndef __ID3D12DebugCommandList2_FWD_DEFINED__
#define __ID3D12DebugCommandList2_FWD_DEFINED__
typedef interface ID3D12DebugCommandList2 ID3D12DebugCommandList2;

#endif 	/* __ID3D12DebugCommandList2_FWD_DEFINED__ */


#ifndef __ID3D12DebugCommandList3_FWD_DEFINED__
#define __ID3D12DebugCommandList3_FWD_DEFINED__
typedef interface ID3D12DebugCommandList3 ID3D12DebugCommandList3;

#endif 	/* __ID3D12DebugCommandList3_FWD_DEFINED__ */


#ifndef __ID3D12SharingContract_FWD_DEFINED__
#define __ID3D12SharingContract_FWD_DEFINED__
typedef interface ID3D12SharingContract ID3D12SharingContract;

#endif 	/* __ID3D12SharingContract_FWD_DEFINED__ */


#ifndef __ID3D12ManualWriteTrackingResource_FWD_DEFINED__
#define __ID3D12ManualWriteTrackingResource_FWD_DEFINED__
typedef interface ID3D12ManualWriteTrackingResource ID3D12ManualWriteTrackingResource;

#endif 	/* __ID3D12ManualWriteTrackingResource_FWD_DEFINED__ */


#ifndef __ID3D12InfoQueue_FWD_DEFINED__
#define __ID3D12InfoQueue_FWD_DEFINED__
typedef interface ID3D12InfoQueue ID3D12InfoQueue;

#endif 	/* __ID3D12InfoQueue_FWD_DEFINED__ */


#ifndef __ID3D12InfoQueue1_FWD_DEFINED__
#define __ID3D12InfoQueue1_FWD_DEFINED__
typedef interface ID3D12InfoQueue1 ID3D12InfoQueue1;

#endif 	/* __ID3D12InfoQueue1_FWD_DEFINED__ */


/* header files for imported files */
#include "oaidl.h"
#include "ocidl.h"
#include "d3d12.h"

#ifdef __cplusplus
extern "C"{
#endif 


/* interface __MIDL_itf_d3d12sdklayers_0000_0000 */
/* [local] */ 

#include <winapifamily.h>
#ifdef _MSC_VER
#pragma region App Family
#endif
#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP | WINAPI_PARTITION_GAMES)


extern RPC_IF_HANDLE __MIDL_itf_d3d12sdklayers_0000_0000_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12sdklayers_0000_0000_v0_0_s_ifspec;

#ifndef __ID3D12Debug_INTERFACE_DEFINED__
#define __ID3D12Debug_INTERFACE_DEFINED__

/* interface ID3D12Debug */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12Debug;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("344488b7-6846-474b-b989-f027448245e0")
    ID3D12Debug : public IUnknown
    {
    public:
        virtual void STDMETHODCALLTYPE EnableDebugLayer( void) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12DebugVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12Debug * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12Debug * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12Debug * This);
        
        DECLSPEC_XFGVIRT(ID3D12Debug, EnableDebugLayer)
        void ( STDMETHODCALLTYPE *EnableDebugLayer )( 
            ID3D12Debug * This);
        
        END_INTERFACE
    } ID3D12DebugVtbl;

    interface ID3D12Debug
    {
        CONST_VTBL struct ID3D12DebugVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12Debug_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12Debug_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12Debug_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12Debug_EnableDebugLayer(This)	\
    ( (This)->lpVtbl -> EnableDebugLayer(This) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12Debug_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12sdklayers_0000_0001 */
/* [local] */ 

typedef 
enum D3D12_GPU_BASED_VALIDATION_FLAGS
    {
        D3D12_GPU_BASED_VALIDATION_FLAGS_NONE	= 0,
        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING	= 0x1
    } 	D3D12_GPU_BASED_VALIDATION_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_GPU_BASED_VALIDATION_FLAGS)


extern RPC_IF_HANDLE __MIDL_itf_d3d12sdklayers_0000_0001_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12sdklayers_0000_0001_v0_0_s_ifspec;

#ifndef __ID3D12Debug1_INTERFACE_DEFINED__
#define __ID3D12Debug1_INTERFACE_DEFINED__

/* interface ID3D12Debug1 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12Debug1;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("affaa4ca-63fe-4d8e-b8ad-159000af4304")
    ID3D12Debug1 : public IUnknown
    {
    public:
        virtual void STDMETHODCALLTYPE EnableDebugLayer( void) = 0;
        
        virtual void STDMETHODCALLTYPE SetEnableGPUBasedValidation( 
            BOOL Enable) = 0;
        
        virtual void STDMETHODCALLTYPE SetEnableSynchronizedCommandQueueValidation( 
            BOOL Enable) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12Debug1Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12Debug1 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12Debug1 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12Debug1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Debug1, EnableDebugLayer)
        void ( STDMETHODCALLTYPE *EnableDebugLayer )( 
            ID3D12Debug1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Debug1, SetEnableGPUBasedValidation)
        void ( STDMETHODCALLTYPE *SetEnableGPUBasedValidation )( 
            ID3D12Debug1 * This,
            BOOL Enable);
        
        DECLSPEC_XFGVIRT(ID3D12Debug1, SetEnableSynchronizedCommandQueueValidation)
        void ( STDMETHODCALLTYPE *SetEnableSynchronizedCommandQueueValidation )( 
            ID3D12Debug1 * This,
            BOOL Enable);
        
        END_INTERFACE
    } ID3D12Debug1Vtbl;

    interface ID3D12Debug1
    {
        CONST_VTBL struct ID3D12Debug1Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12Debug1_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12Debug1_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12Debug1_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12Debug1_EnableDebugLayer(This)	\
    ( (This)->lpVtbl -> EnableDebugLayer(This) ) 

#define ID3D12Debug1_SetEnableGPUBasedValidation(This,Enable)	\
    ( (This)->lpVtbl -> SetEnableGPUBasedValidation(This,Enable) ) 

#define ID3D12Debug1_SetEnableSynchronizedCommandQueueValidation(This,Enable)	\
    ( (This)->lpVtbl -> SetEnableSynchronizedCommandQueueValidation(This,Enable) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12Debug1_INTERFACE_DEFINED__ */


#ifndef __ID3D12Debug2_INTERFACE_DEFINED__
#define __ID3D12Debug2_INTERFACE_DEFINED__

/* interface ID3D12Debug2 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12Debug2;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("93a665c4-a3b2-4e5d-b692-a26ae14e3374")
    ID3D12Debug2 : public IUnknown
    {
    public:
        virtual void STDMETHODCALLTYPE SetGPUBasedValidationFlags( 
            D3D12_GPU_BASED_VALIDATION_FLAGS Flags) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12Debug2Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12Debug2 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12Debug2 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12Debug2 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Debug2, SetGPUBasedValidationFlags)
        void ( STDMETHODCALLTYPE *SetGPUBasedValidationFlags )( 
            ID3D12Debug2 * This,
            D3D12_GPU_BASED_VALIDATION_FLAGS Flags);
        
        END_INTERFACE
    } ID3D12Debug2Vtbl;

    interface ID3D12Debug2
    {
        CONST_VTBL struct ID3D12Debug2Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12Debug2_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12Debug2_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12Debug2_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12Debug2_SetGPUBasedValidationFlags(This,Flags)	\
    ( (This)->lpVtbl -> SetGPUBasedValidationFlags(This,Flags) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12Debug2_INTERFACE_DEFINED__ */


#ifndef __ID3D12Debug3_INTERFACE_DEFINED__
#define __ID3D12Debug3_INTERFACE_DEFINED__

/* interface ID3D12Debug3 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12Debug3;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("5cf4e58f-f671-4ff1-a542-3686e3d153d1")
    ID3D12Debug3 : public ID3D12Debug
    {
    public:
        virtual void STDMETHODCALLTYPE SetEnableGPUBasedValidation( 
            BOOL Enable) = 0;
        
        virtual void STDMETHODCALLTYPE SetEnableSynchronizedCommandQueueValidation( 
            BOOL Enable) = 0;
        
        virtual void STDMETHODCALLTYPE SetGPUBasedValidationFlags( 
            D3D12_GPU_BASED_VALIDATION_FLAGS Flags) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12Debug3Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12Debug3 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12Debug3 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12Debug3 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Debug, EnableDebugLayer)
        void ( STDMETHODCALLTYPE *EnableDebugLayer )( 
            ID3D12Debug3 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Debug3, SetEnableGPUBasedValidation)
        void ( STDMETHODCALLTYPE *SetEnableGPUBasedValidation )( 
            ID3D12Debug3 * This,
            BOOL Enable);
        
        DECLSPEC_XFGVIRT(ID3D12Debug3, SetEnableSynchronizedCommandQueueValidation)
        void ( STDMETHODCALLTYPE *SetEnableSynchronizedCommandQueueValidation )( 
            ID3D12Debug3 * This,
            BOOL Enable);
        
        DECLSPEC_XFGVIRT(ID3D12Debug3, SetGPUBasedValidationFlags)
        void ( STDMETHODCALLTYPE *SetGPUBasedValidationFlags )( 
            ID3D12Debug3 * This,
            D3D12_GPU_BASED_VALIDATION_FLAGS Flags);
        
        END_INTERFACE
    } ID3D12Debug3Vtbl;

    interface ID3D12Debug3
    {
        CONST_VTBL struct ID3D12Debug3Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12Debug3_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12Debug3_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12Debug3_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12Debug3_EnableDebugLayer(This)	\
    ( (This)->lpVtbl -> EnableDebugLayer(This) ) 


#define ID3D12Debug3_SetEnableGPUBasedValidation(This,Enable)	\
    ( (This)->lpVtbl -> SetEnableGPUBasedValidation(This,Enable) ) 

#define ID3D12Debug3_SetEnableSynchronizedCommandQueueValidation(This,Enable)	\
    ( (This)->lpVtbl -> SetEnableSynchronizedCommandQueueValidation(This,Enable) ) 

#define ID3D12Debug3_SetGPUBasedValidationFlags(This,Flags)	\
    ( (This)->lpVtbl -> SetGPUBasedValidationFlags(This,Flags) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12Debug3_INTERFACE_DEFINED__ */


#ifndef __ID3D12Debug4_INTERFACE_DEFINED__
#define __ID3D12Debug4_INTERFACE_DEFINED__

/* interface ID3D12Debug4 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12Debug4;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("014b816e-9ec5-4a2f-a845-ffbe441ce13a")
    ID3D12Debug4 : public ID3D12Debug3
    {
    public:
        virtual void STDMETHODCALLTYPE DisableDebugLayer( void) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12Debug4Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12Debug4 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12Debug4 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12Debug4 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Debug, EnableDebugLayer)
        void ( STDMETHODCALLTYPE *EnableDebugLayer )( 
            ID3D12Debug4 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Debug3, SetEnableGPUBasedValidation)
        void ( STDMETHODCALLTYPE *SetEnableGPUBasedValidation )( 
            ID3D12Debug4 * This,
            BOOL Enable);
        
        DECLSPEC_XFGVIRT(ID3D12Debug3, SetEnableSynchronizedCommandQueueValidation)
        void ( STDMETHODCALLTYPE *SetEnableSynchronizedCommandQueueValidation )( 
            ID3D12Debug4 * This,
            BOOL Enable);
        
        DECLSPEC_XFGVIRT(ID3D12Debug3, SetGPUBasedValidationFlags)
        void ( STDMETHODCALLTYPE *SetGPUBasedValidationFlags )( 
            ID3D12Debug4 * This,
            D3D12_GPU_BASED_VALIDATION_FLAGS Flags);
        
        DECLSPEC_XFGVIRT(ID3D12Debug4, DisableDebugLayer)
        void ( STDMETHODCALLTYPE *DisableDebugLayer )( 
            ID3D12Debug4 * This);
        
        END_INTERFACE
    } ID3D12Debug4Vtbl;

    interface ID3D12Debug4
    {
        CONST_VTBL struct ID3D12Debug4Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12Debug4_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12Debug4_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12Debug4_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12Debug4_EnableDebugLayer(This)	\
    ( (This)->lpVtbl -> EnableDebugLayer(This) ) 


#define ID3D12Debug4_SetEnableGPUBasedValidation(This,Enable)	\
    ( (This)->lpVtbl -> SetEnableGPUBasedValidation(This,Enable) ) 

#define ID3D12Debug4_SetEnableSynchronizedCommandQueueValidation(This,Enable)	\
    ( (This)->lpVtbl -> SetEnableSynchronizedCommandQueueValidation(This,Enable) ) 

#define ID3D12Debug4_SetGPUBasedValidationFlags(This,Flags)	\
    ( (This)->lpVtbl -> SetGPUBasedValidationFlags(This,Flags) ) 


#define ID3D12Debug4_DisableDebugLayer(This)	\
    ( (This)->lpVtbl -> DisableDebugLayer(This) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12Debug4_INTERFACE_DEFINED__ */


#ifndef __ID3D12Debug5_INTERFACE_DEFINED__
#define __ID3D12Debug5_INTERFACE_DEFINED__

/* interface ID3D12Debug5 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12Debug5;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("548d6b12-09fa-40e0-9069-5dcd589a52c9")
    ID3D12Debug5 : public ID3D12Debug4
    {
    public:
        virtual void STDMETHODCALLTYPE SetEnableAutoName( 
            BOOL Enable) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12Debug5Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12Debug5 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12Debug5 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12Debug5 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Debug, EnableDebugLayer)
        void ( STDMETHODCALLTYPE *EnableDebugLayer )( 
            ID3D12Debug5 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Debug3, SetEnableGPUBasedValidation)
        void ( STDMETHODCALLTYPE *SetEnableGPUBasedValidation )( 
            ID3D12Debug5 * This,
            BOOL Enable);
        
        DECLSPEC_XFGVIRT(ID3D12Debug3, SetEnableSynchronizedCommandQueueValidation)
        void ( STDMETHODCALLTYPE *SetEnableSynchronizedCommandQueueValidation )( 
            ID3D12Debug5 * This,
            BOOL Enable);
        
        DECLSPEC_XFGVIRT(ID3D12Debug3, SetGPUBasedValidationFlags)
        void ( STDMETHODCALLTYPE *SetGPUBasedValidationFlags )( 
            ID3D12Debug5 * This,
            D3D12_GPU_BASED_VALIDATION_FLAGS Flags);
        
        DECLSPEC_XFGVIRT(ID3D12Debug4, DisableDebugLayer)
        void ( STDMETHODCALLTYPE *DisableDebugLayer )( 
            ID3D12Debug5 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Debug5, SetEnableAutoName)
        void ( STDMETHODCALLTYPE *SetEnableAutoName )( 
            ID3D12Debug5 * This,
            BOOL Enable);
        
        END_INTERFACE
    } ID3D12Debug5Vtbl;

    interface ID3D12Debug5
    {
        CONST_VTBL struct ID3D12Debug5Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12Debug5_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12Debug5_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12Debug5_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12Debug5_EnableDebugLayer(This)	\
    ( (This)->lpVtbl -> EnableDebugLayer(This) ) 


#define ID3D12Debug5_SetEnableGPUBasedValidation(This,Enable)	\
    ( (This)->lpVtbl -> SetEnableGPUBasedValidation(This,Enable) ) 

#define ID3D12Debug5_SetEnableSynchronizedCommandQueueValidation(This,Enable)	\
    ( (This)->lpVtbl -> SetEnableSynchronizedCommandQueueValidation(This,Enable) ) 

#define ID3D12Debug5_SetGPUBasedValidationFlags(This,Flags)	\
    ( (This)->lpVtbl -> SetGPUBasedValidationFlags(This,Flags) ) 


#define ID3D12Debug5_DisableDebugLayer(This)	\
    ( (This)->lpVtbl -> DisableDebugLayer(This) ) 


#define ID3D12Debug5_SetEnableAutoName(This,Enable)	\
    ( (This)->lpVtbl -> SetEnableAutoName(This,Enable) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12Debug5_INTERFACE_DEFINED__ */


#ifndef __ID3D12Debug6_INTERFACE_DEFINED__
#define __ID3D12Debug6_INTERFACE_DEFINED__

/* interface ID3D12Debug6 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12Debug6;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("82a816d6-5d01-4157-97d0-4975463fd1ed")
    ID3D12Debug6 : public ID3D12Debug5
    {
    public:
        virtual void STDMETHODCALLTYPE SetForceLegacyBarrierValidation( 
            BOOL Enable) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12Debug6Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12Debug6 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12Debug6 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12Debug6 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Debug, EnableDebugLayer)
        void ( STDMETHODCALLTYPE *EnableDebugLayer )( 
            ID3D12Debug6 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Debug3, SetEnableGPUBasedValidation)
        void ( STDMETHODCALLTYPE *SetEnableGPUBasedValidation )( 
            ID3D12Debug6 * This,
            BOOL Enable);
        
        DECLSPEC_XFGVIRT(ID3D12Debug3, SetEnableSynchronizedCommandQueueValidation)
        void ( STDMETHODCALLTYPE *SetEnableSynchronizedCommandQueueValidation )( 
            ID3D12Debug6 * This,
            BOOL Enable);
        
        DECLSPEC_XFGVIRT(ID3D12Debug3, SetGPUBasedValidationFlags)
        void ( STDMETHODCALLTYPE *SetGPUBasedValidationFlags )( 
            ID3D12Debug6 * This,
            D3D12_GPU_BASED_VALIDATION_FLAGS Flags);
        
        DECLSPEC_XFGVIRT(ID3D12Debug4, DisableDebugLayer)
        void ( STDMETHODCALLTYPE *DisableDebugLayer )( 
            ID3D12Debug6 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Debug5, SetEnableAutoName)
        void ( STDMETHODCALLTYPE *SetEnableAutoName )( 
            ID3D12Debug6 * This,
            BOOL Enable);
        
        DECLSPEC_XFGVIRT(ID3D12Debug6, SetForceLegacyBarrierValidation)
        void ( STDMETHODCALLTYPE *SetForceLegacyBarrierValidation )( 
            ID3D12Debug6 * This,
            BOOL Enable);
        
        END_INTERFACE
    } ID3D12Debug6Vtbl;

    interface ID3D12Debug6
    {
        CONST_VTBL struct ID3D12Debug6Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12Debug6_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12Debug6_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12Debug6_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12Debug6_EnableDebugLayer(This)	\
    ( (This)->lpVtbl -> EnableDebugLayer(This) ) 


#define ID3D12Debug6_SetEnableGPUBasedValidation(This,Enable)	\
    ( (This)->lpVtbl -> SetEnableGPUBasedValidation(This,Enable) ) 

#define ID3D12Debug6_SetEnableSynchronizedCommandQueueValidation(This,Enable)	\
    ( (This)->lpVtbl -> SetEnableSynchronizedCommandQueueValidation(This,Enable) ) 

#define ID3D12Debug6_SetGPUBasedValidationFlags(This,Flags)	\
    ( (This)->lpVtbl -> SetGPUBasedValidationFlags(This,Flags) ) 


#define ID3D12Debug6_DisableDebugLayer(This)	\
    ( (This)->lpVtbl -> DisableDebugLayer(This) ) 


#define ID3D12Debug6_SetEnableAutoName(This,Enable)	\
    ( (This)->lpVtbl -> SetEnableAutoName(This,Enable) ) 


#define ID3D12Debug6_SetForceLegacyBarrierValidation(This,Enable)	\
    ( (This)->lpVtbl -> SetForceLegacyBarrierValidation(This,Enable) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12Debug6_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12sdklayers_0000_0007 */
/* [local] */ 

DEFINE_GUID(WKPDID_D3DAutoDebugObjectNameW, 0xd4902e36, 0x757a, 0x4942, 0x95, 0x94, 0xb6, 0x76, 0x9a, 0xfa, 0x43, 0xcd);
typedef 
enum D3D12_RLDO_FLAGS
    {
        D3D12_RLDO_NONE	= 0,
        D3D12_RLDO_SUMMARY	= 0x1,
        D3D12_RLDO_DETAIL	= 0x2,
        D3D12_RLDO_IGNORE_INTERNAL	= 0x4
    } 	D3D12_RLDO_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_RLDO_FLAGS)
typedef 
enum D3D12_DEBUG_DEVICE_PARAMETER_TYPE
    {
        D3D12_DEBUG_DEVICE_PARAMETER_FEATURE_FLAGS	= 0,
        D3D12_DEBUG_DEVICE_PARAMETER_GPU_BASED_VALIDATION_SETTINGS	= ( D3D12_DEBUG_DEVICE_PARAMETER_FEATURE_FLAGS + 1 ) ,
        D3D12_DEBUG_DEVICE_PARAMETER_GPU_SLOWDOWN_PERFORMANCE_FACTOR	= ( D3D12_DEBUG_DEVICE_PARAMETER_GPU_BASED_VALIDATION_SETTINGS + 1 ) 
    } 	D3D12_DEBUG_DEVICE_PARAMETER_TYPE;

typedef 
enum D3D12_DEBUG_FEATURE
    {
        D3D12_DEBUG_FEATURE_NONE	= 0,
        D3D12_DEBUG_FEATURE_ALLOW_BEHAVIOR_CHANGING_DEBUG_AIDS	= 0x1,
        D3D12_DEBUG_FEATURE_CONSERVATIVE_RESOURCE_STATE_TRACKING	= 0x2,
        D3D12_DEBUG_FEATURE_DISABLE_VIRTUALIZED_BUNDLES_VALIDATION	= 0x4,
        D3D12_DEBUG_FEATURE_EMULATE_WINDOWS7	= 0x8
    } 	D3D12_DEBUG_FEATURE;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_DEBUG_FEATURE)
typedef 
enum D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODE
    {
        D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODE_NONE	= 0,
        D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODE_STATE_TRACKING_ONLY	= ( D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODE_NONE + 1 ) ,
        D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODE_UNGUARDED_VALIDATION	= ( D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODE_STATE_TRACKING_ONLY + 1 ) ,
        D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODE_GUARDED_VALIDATION	= ( D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODE_UNGUARDED_VALIDATION + 1 ) ,
        NUM_D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODES	= ( D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODE_GUARDED_VALIDATION + 1 ) 
    } 	D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODE;

typedef 
enum D3D12_GPU_BASED_VALIDATION_PIPELINE_STATE_CREATE_FLAGS
    {
        D3D12_GPU_BASED_VALIDATION_PIPELINE_STATE_CREATE_FLAG_NONE	= 0,
        D3D12_GPU_BASED_VALIDATION_PIPELINE_STATE_CREATE_FLAG_FRONT_LOAD_CREATE_TRACKING_ONLY_SHADERS	= 0x1,
        D3D12_GPU_BASED_VALIDATION_PIPELINE_STATE_CREATE_FLAG_FRONT_LOAD_CREATE_UNGUARDED_VALIDATION_SHADERS	= 0x2,
        D3D12_GPU_BASED_VALIDATION_PIPELINE_STATE_CREATE_FLAG_FRONT_LOAD_CREATE_GUARDED_VALIDATION_SHADERS	= 0x4,
        D3D12_GPU_BASED_VALIDATION_PIPELINE_STATE_CREATE_FLAGS_VALID_MASK	= 0x7
    } 	D3D12_GPU_BASED_VALIDATION_PIPELINE_STATE_CREATE_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_GPU_BASED_VALIDATION_PIPELINE_STATE_CREATE_FLAGS)
typedef struct D3D12_DEBUG_DEVICE_GPU_BASED_VALIDATION_SETTINGS
    {
    UINT MaxMessagesPerCommandList;
    D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODE DefaultShaderPatchMode;
    D3D12_GPU_BASED_VALIDATION_PIPELINE_STATE_CREATE_FLAGS PipelineStateCreateFlags;
    } 	D3D12_DEBUG_DEVICE_GPU_BASED_VALIDATION_SETTINGS;

typedef struct D3D12_DEBUG_DEVICE_GPU_SLOWDOWN_PERFORMANCE_FACTOR
    {
    FLOAT SlowdownFactor;
    } 	D3D12_DEBUG_DEVICE_GPU_SLOWDOWN_PERFORMANCE_FACTOR;



extern RPC_IF_HANDLE __MIDL_itf_d3d12sdklayers_0000_0007_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12sdklayers_0000_0007_v0_0_s_ifspec;

#ifndef __ID3D12DebugDevice1_INTERFACE_DEFINED__
#define __ID3D12DebugDevice1_INTERFACE_DEFINED__

/* interface ID3D12DebugDevice1 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12DebugDevice1;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("a9b71770-d099-4a65-a698-3dee10020f88")
    ID3D12DebugDevice1 : public IUnknown
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE SetDebugParameter( 
            D3D12_DEBUG_DEVICE_PARAMETER_TYPE Type,
            _In_reads_bytes_(DataSize)  const void *pData,
            UINT DataSize) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetDebugParameter( 
            D3D12_DEBUG_DEVICE_PARAMETER_TYPE Type,
            _Out_writes_bytes_(DataSize)  void *pData,
            UINT DataSize) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE ReportLiveDeviceObjects( 
            D3D12_RLDO_FLAGS Flags) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12DebugDevice1Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12DebugDevice1 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12DebugDevice1 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12DebugDevice1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12DebugDevice1, SetDebugParameter)
        HRESULT ( STDMETHODCALLTYPE *SetDebugParameter )( 
            ID3D12DebugDevice1 * This,
            D3D12_DEBUG_DEVICE_PARAMETER_TYPE Type,
            _In_reads_bytes_(DataSize)  const void *pData,
            UINT DataSize);
        
        DECLSPEC_XFGVIRT(ID3D12DebugDevice1, GetDebugParameter)
        HRESULT ( STDMETHODCALLTYPE *GetDebugParameter )( 
            ID3D12DebugDevice1 * This,
            D3D12_DEBUG_DEVICE_PARAMETER_TYPE Type,
            _Out_writes_bytes_(DataSize)  void *pData,
            UINT DataSize);
        
        DECLSPEC_XFGVIRT(ID3D12DebugDevice1, ReportLiveDeviceObjects)
        HRESULT ( STDMETHODCALLTYPE *ReportLiveDeviceObjects )( 
            ID3D12DebugDevice1 * This,
            D3D12_RLDO_FLAGS Flags);
        
        END_INTERFACE
    } ID3D12DebugDevice1Vtbl;

    interface ID3D12DebugDevice1
    {
        CONST_VTBL struct ID3D12DebugDevice1Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12DebugDevice1_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12DebugDevice1_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12DebugDevice1_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12DebugDevice1_SetDebugParameter(This,Type,pData,DataSize)	\
    ( (This)->lpVtbl -> SetDebugParameter(This,Type,pData,DataSize) ) 

#define ID3D12DebugDevice1_GetDebugParameter(This,Type,pData,DataSize)	\
    ( (This)->lpVtbl -> GetDebugParameter(This,Type,pData,DataSize) ) 

#define ID3D12DebugDevice1_ReportLiveDeviceObjects(This,Flags)	\
    ( (This)->lpVtbl -> ReportLiveDeviceObjects(This,Flags) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12DebugDevice1_INTERFACE_DEFINED__ */


#ifndef __ID3D12DebugDevice_INTERFACE_DEFINED__
#define __ID3D12DebugDevice_INTERFACE_DEFINED__

/* interface ID3D12DebugDevice */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12DebugDevice;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("3febd6dd-4973-4787-8194-e45f9e28923e")
    ID3D12DebugDevice : public IUnknown
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE SetFeatureMask( 
            D3D12_DEBUG_FEATURE Mask) = 0;
        
        virtual D3D12_DEBUG_FEATURE STDMETHODCALLTYPE GetFeatureMask( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE ReportLiveDeviceObjects( 
            D3D12_RLDO_FLAGS Flags) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12DebugDeviceVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12DebugDevice * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12DebugDevice * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12DebugDevice * This);
        
        DECLSPEC_XFGVIRT(ID3D12DebugDevice, SetFeatureMask)
        HRESULT ( STDMETHODCALLTYPE *SetFeatureMask )( 
            ID3D12DebugDevice * This,
            D3D12_DEBUG_FEATURE Mask);
        
        DECLSPEC_XFGVIRT(ID3D12DebugDevice, GetFeatureMask)
        D3D12_DEBUG_FEATURE ( STDMETHODCALLTYPE *GetFeatureMask )( 
            ID3D12DebugDevice * This);
        
        DECLSPEC_XFGVIRT(ID3D12DebugDevice, ReportLiveDeviceObjects)
        HRESULT ( STDMETHODCALLTYPE *ReportLiveDeviceObjects )( 
            ID3D12DebugDevice * This,
            D3D12_RLDO_FLAGS Flags);
        
        END_INTERFACE
    } ID3D12DebugDeviceVtbl;

    interface ID3D12DebugDevice
    {
        CONST_VTBL struct ID3D12DebugDeviceVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12DebugDevice_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12DebugDevice_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12DebugDevice_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12DebugDevice_SetFeatureMask(This,Mask)	\
    ( (This)->lpVtbl -> SetFeatureMask(This,Mask) ) 

#define ID3D12DebugDevice_GetFeatureMask(This)	\
    ( (This)->lpVtbl -> GetFeatureMask(This) ) 

#define ID3D12DebugDevice_ReportLiveDeviceObjects(This,Flags)	\
    ( (This)->lpVtbl -> ReportLiveDeviceObjects(This,Flags) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12DebugDevice_INTERFACE_DEFINED__ */


#ifndef __ID3D12DebugDevice2_INTERFACE_DEFINED__
#define __ID3D12DebugDevice2_INTERFACE_DEFINED__

/* interface ID3D12DebugDevice2 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12DebugDevice2;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("60eccbc1-378d-4df1-894c-f8ac5ce4d7dd")
    ID3D12DebugDevice2 : public ID3D12DebugDevice
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE SetDebugParameter( 
            D3D12_DEBUG_DEVICE_PARAMETER_TYPE Type,
            _In_reads_bytes_(DataSize)  const void *pData,
            UINT DataSize) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetDebugParameter( 
            D3D12_DEBUG_DEVICE_PARAMETER_TYPE Type,
            _Out_writes_bytes_(DataSize)  void *pData,
            UINT DataSize) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12DebugDevice2Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12DebugDevice2 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12DebugDevice2 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12DebugDevice2 * This);
        
        DECLSPEC_XFGVIRT(ID3D12DebugDevice, SetFeatureMask)
        HRESULT ( STDMETHODCALLTYPE *SetFeatureMask )( 
            ID3D12DebugDevice2 * This,
            D3D12_DEBUG_FEATURE Mask);
        
        DECLSPEC_XFGVIRT(ID3D12DebugDevice, GetFeatureMask)
        D3D12_DEBUG_FEATURE ( STDMETHODCALLTYPE *GetFeatureMask )( 
            ID3D12DebugDevice2 * This);
        
        DECLSPEC_XFGVIRT(ID3D12DebugDevice, ReportLiveDeviceObjects)
        HRESULT ( STDMETHODCALLTYPE *ReportLiveDeviceObjects )( 
            ID3D12DebugDevice2 * This,
            D3D12_RLDO_FLAGS Flags);
        
        DECLSPEC_XFGVIRT(ID3D12DebugDevice2, SetDebugParameter)
        HRESULT ( STDMETHODCALLTYPE *SetDebugParameter )( 
            ID3D12DebugDevice2 * This,
            D3D12_DEBUG_DEVICE_PARAMETER_TYPE Type,
            _In_reads_bytes_(DataSize)  const void *pData,
            UINT DataSize);
        
        DECLSPEC_XFGVIRT(ID3D12DebugDevice2, GetDebugParameter)
        HRESULT ( STDMETHODCALLTYPE *GetDebugParameter )( 
            ID3D12DebugDevice2 * This,
            D3D12_DEBUG_DEVICE_PARAMETER_TYPE Type,
            _Out_writes_bytes_(DataSize)  void *pData,
            UINT DataSize);
        
        END_INTERFACE
    } ID3D12DebugDevice2Vtbl;

    interface ID3D12DebugDevice2
    {
        CONST_VTBL struct ID3D12DebugDevice2Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12DebugDevice2_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12DebugDevice2_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12DebugDevice2_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12DebugDevice2_SetFeatureMask(This,Mask)	\
    ( (This)->lpVtbl -> SetFeatureMask(This,Mask) ) 

#define ID3D12DebugDevice2_GetFeatureMask(This)	\
    ( (This)->lpVtbl -> GetFeatureMask(This) ) 

#define ID3D12DebugDevice2_ReportLiveDeviceObjects(This,Flags)	\
    ( (This)->lpVtbl -> ReportLiveDeviceObjects(This,Flags) ) 


#define ID3D12DebugDevice2_SetDebugParameter(This,Type,pData,DataSize)	\
    ( (This)->lpVtbl -> SetDebugParameter(This,Type,pData,DataSize) ) 

#define ID3D12DebugDevice2_GetDebugParameter(This,Type,pData,DataSize)	\
    ( (This)->lpVtbl -> GetDebugParameter(This,Type,pData,DataSize) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12DebugDevice2_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12sdklayers_0000_0010 */
/* [local] */ 

DEFINE_GUID(DXGI_DEBUG_D3D12, 0xcf59a98c, 0xa950, 0x4326, 0x91, 0xef, 0x9b, 0xba, 0xa1, 0x7b, 0xfd, 0x95);


extern RPC_IF_HANDLE __MIDL_itf_d3d12sdklayers_0000_0010_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12sdklayers_0000_0010_v0_0_s_ifspec;

#ifndef __ID3D12DebugCommandQueue_INTERFACE_DEFINED__
#define __ID3D12DebugCommandQueue_INTERFACE_DEFINED__

/* interface ID3D12DebugCommandQueue */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12DebugCommandQueue;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("09e0bf36-54ac-484f-8847-4baeeab6053a")
    ID3D12DebugCommandQueue : public IUnknown
    {
    public:
        virtual BOOL STDMETHODCALLTYPE AssertResourceState( 
            _In_  ID3D12Resource *pResource,
            UINT Subresource,
            UINT State) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12DebugCommandQueueVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12DebugCommandQueue * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12DebugCommandQueue * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12DebugCommandQueue * This);
        
        DECLSPEC_XFGVIRT(ID3D12DebugCommandQueue, AssertResourceState)
        BOOL ( STDMETHODCALLTYPE *AssertResourceState )( 
            ID3D12DebugCommandQueue * This,
            _In_  ID3D12Resource *pResource,
            UINT Subresource,
            UINT State);
        
        END_INTERFACE
    } ID3D12DebugCommandQueueVtbl;

    interface ID3D12DebugCommandQueue
    {
        CONST_VTBL struct ID3D12DebugCommandQueueVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12DebugCommandQueue_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12DebugCommandQueue_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12DebugCommandQueue_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12DebugCommandQueue_AssertResourceState(This,pResource,Subresource,State)	\
    ( (This)->lpVtbl -> AssertResourceState(This,pResource,Subresource,State) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12DebugCommandQueue_INTERFACE_DEFINED__ */


#ifndef __ID3D12DebugCommandQueue1_INTERFACE_DEFINED__
#define __ID3D12DebugCommandQueue1_INTERFACE_DEFINED__

/* interface ID3D12DebugCommandQueue1 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12DebugCommandQueue1;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("16be35a2-bfd6-49f2-bcae-eaae4aff862d")
    ID3D12DebugCommandQueue1 : public ID3D12DebugCommandQueue
    {
    public:
        virtual void STDMETHODCALLTYPE AssertResourceAccess( 
            _In_  ID3D12Resource *pResource,
            UINT Subresource,
            D3D12_BARRIER_ACCESS Access) = 0;
        
        virtual void STDMETHODCALLTYPE AssertTextureLayout( 
            _In_  ID3D12Resource *pResource,
            UINT Subresource,
            D3D12_BARRIER_LAYOUT Layout) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12DebugCommandQueue1Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12DebugCommandQueue1 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12DebugCommandQueue1 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12DebugCommandQueue1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12DebugCommandQueue, AssertResourceState)
        BOOL ( STDMETHODCALLTYPE *AssertResourceState )( 
            ID3D12DebugCommandQueue1 * This,
            _In_  ID3D12Resource *pResource,
            UINT Subresource,
            UINT State);
        
        DECLSPEC_XFGVIRT(ID3D12DebugCommandQueue1, AssertResourceAccess)
        void ( STDMETHODCALLTYPE *AssertResourceAccess )( 
            ID3D12DebugCommandQueue1 * This,
            _In_  ID3D12Resource *pResource,
            UINT Subresource,
            D3D12_BARRIER_ACCESS Access);
        
        DECLSPEC_XFGVIRT(ID3D12DebugCommandQueue1, AssertTextureLayout)
        void ( STDMETHODCALLTYPE *AssertTextureLayout )( 
            ID3D12DebugCommandQueue1 * This,
            _In_  ID3D12Resource *pResource,
            UINT Subresource,
            D3D12_BARRIER_LAYOUT Layout);
        
        END_INTERFACE
    } ID3D12DebugCommandQueue1Vtbl;

    interface ID3D12DebugCommandQueue1
    {
        CONST_VTBL struct ID3D12DebugCommandQueue1Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12DebugCommandQueue1_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12DebugCommandQueue1_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12DebugCommandQueue1_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12DebugCommandQueue1_AssertResourceState(This,pResource,Subresource,State)	\
    ( (This)->lpVtbl -> AssertResourceState(This,pResource,Subresource,State) ) 


#define ID3D12DebugCommandQueue1_AssertResourceAccess(This,pResource,Subresource,Access)	\
    ( (This)->lpVtbl -> AssertResourceAccess(This,pResource,Subresource,Access) ) 

#define ID3D12DebugCommandQueue1_AssertTextureLayout(This,pResource,Subresource,Layout)	\
    ( (This)->lpVtbl -> AssertTextureLayout(This,pResource,Subresource,Layout) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12DebugCommandQueue1_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12sdklayers_0000_0012 */
/* [local] */ 

typedef 
enum D3D12_DEBUG_COMMAND_LIST_PARAMETER_TYPE
    {
        D3D12_DEBUG_COMMAND_LIST_PARAMETER_GPU_BASED_VALIDATION_SETTINGS	= 0
    } 	D3D12_DEBUG_COMMAND_LIST_PARAMETER_TYPE;

typedef struct D3D12_DEBUG_COMMAND_LIST_GPU_BASED_VALIDATION_SETTINGS
    {
    D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODE ShaderPatchMode;
    } 	D3D12_DEBUG_COMMAND_LIST_GPU_BASED_VALIDATION_SETTINGS;



extern RPC_IF_HANDLE __MIDL_itf_d3d12sdklayers_0000_0012_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12sdklayers_0000_0012_v0_0_s_ifspec;

#ifndef __ID3D12DebugCommandList1_INTERFACE_DEFINED__
#define __ID3D12DebugCommandList1_INTERFACE_DEFINED__

/* interface ID3D12DebugCommandList1 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12DebugCommandList1;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("102ca951-311b-4b01-b11f-ecb83e061b37")
    ID3D12DebugCommandList1 : public IUnknown
    {
    public:
        virtual BOOL STDMETHODCALLTYPE AssertResourceState( 
            _In_  ID3D12Resource *pResource,
            UINT Subresource,
            UINT State) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE SetDebugParameter( 
            D3D12_DEBUG_COMMAND_LIST_PARAMETER_TYPE Type,
            _In_reads_bytes_(DataSize)  const void *pData,
            UINT DataSize) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetDebugParameter( 
            D3D12_DEBUG_COMMAND_LIST_PARAMETER_TYPE Type,
            _Out_writes_bytes_(DataSize)  void *pData,
            UINT DataSize) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12DebugCommandList1Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12DebugCommandList1 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12DebugCommandList1 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12DebugCommandList1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12DebugCommandList1, AssertResourceState)
        BOOL ( STDMETHODCALLTYPE *AssertResourceState )( 
            ID3D12DebugCommandList1 * This,
            _In_  ID3D12Resource *pResource,
            UINT Subresource,
            UINT State);
        
        DECLSPEC_XFGVIRT(ID3D12DebugCommandList1, SetDebugParameter)
        HRESULT ( STDMETHODCALLTYPE *SetDebugParameter )( 
            ID3D12DebugCommandList1 * This,
            D3D12_DEBUG_COMMAND_LIST_PARAMETER_TYPE Type,
            _In_reads_bytes_(DataSize)  const void *pData,
            UINT DataSize);
        
        DECLSPEC_XFGVIRT(ID3D12DebugCommandList1, GetDebugParameter)
        HRESULT ( STDMETHODCALLTYPE *GetDebugParameter )( 
            ID3D12DebugCommandList1 * This,
            D3D12_DEBUG_COMMAND_LIST_PARAMETER_TYPE Type,
            _Out_writes_bytes_(DataSize)  void *pData,
            UINT DataSize);
        
        END_INTERFACE
    } ID3D12DebugCommandList1Vtbl;

    interface ID3D12DebugCommandList1
    {
        CONST_VTBL struct ID3D12DebugCommandList1Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12DebugCommandList1_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12DebugCommandList1_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12DebugCommandList1_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12DebugCommandList1_AssertResourceState(This,pResource,Subresource,State)	\
    ( (This)->lpVtbl -> AssertResourceState(This,pResource,Subresource,State) ) 

#define ID3D12DebugCommandList1_SetDebugParameter(This,Type,pData,DataSize)	\
    ( (This)->lpVtbl -> SetDebugParameter(This,Type,pData,DataSize) ) 

#define ID3D12DebugCommandList1_GetDebugParameter(This,Type,pData,DataSize)	\
    ( (This)->lpVtbl -> GetDebugParameter(This,Type,pData,DataSize) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12DebugCommandList1_INTERFACE_DEFINED__ */


#ifndef __ID3D12DebugCommandList_INTERFACE_DEFINED__
#define __ID3D12DebugCommandList_INTERFACE_DEFINED__

/* interface ID3D12DebugCommandList */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12DebugCommandList;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("09e0bf36-54ac-484f-8847-4baeeab6053f")
    ID3D12DebugCommandList : public IUnknown
    {
    public:
        virtual BOOL STDMETHODCALLTYPE AssertResourceState( 
            _In_  ID3D12Resource *pResource,
            UINT Subresource,
            UINT State) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE SetFeatureMask( 
            D3D12_DEBUG_FEATURE Mask) = 0;
        
        virtual D3D12_DEBUG_FEATURE STDMETHODCALLTYPE GetFeatureMask( void) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12DebugCommandListVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12DebugCommandList * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12DebugCommandList * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12DebugCommandList * This);
        
        DECLSPEC_XFGVIRT(ID3D12DebugCommandList, AssertResourceState)
        BOOL ( STDMETHODCALLTYPE *AssertResourceState )( 
            ID3D12DebugCommandList * This,
            _In_  ID3D12Resource *pResource,
            UINT Subresource,
            UINT State);
        
        DECLSPEC_XFGVIRT(ID3D12DebugCommandList, SetFeatureMask)
        HRESULT ( STDMETHODCALLTYPE *SetFeatureMask )( 
            ID3D12DebugCommandList * This,
            D3D12_DEBUG_FEATURE Mask);
        
        DECLSPEC_XFGVIRT(ID3D12DebugCommandList, GetFeatureMask)
        D3D12_DEBUG_FEATURE ( STDMETHODCALLTYPE *GetFeatureMask )( 
            ID3D12DebugCommandList * This);
        
        END_INTERFACE
    } ID3D12DebugCommandListVtbl;

    interface ID3D12DebugCommandList
    {
        CONST_VTBL struct ID3D12DebugCommandListVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12DebugCommandList_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12DebugCommandList_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12DebugCommandList_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12DebugCommandList_AssertResourceState(This,pResource,Subresource,State)	\
    ( (This)->lpVtbl -> AssertResourceState(This,pResource,Subresource,State) ) 

#define ID3D12DebugCommandList_SetFeatureMask(This,Mask)	\
    ( (This)->lpVtbl -> SetFeatureMask(This,Mask) ) 

#define ID3D12DebugCommandList_GetFeatureMask(This)	\
    ( (This)->lpVtbl -> GetFeatureMask(This) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12DebugCommandList_INTERFACE_DEFINED__ */


#ifndef __ID3D12DebugCommandList2_INTERFACE_DEFINED__
#define __ID3D12DebugCommandList2_INTERFACE_DEFINED__

/* interface ID3D12DebugCommandList2 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12DebugCommandList2;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("aeb575cf-4e06-48be-ba3b-c450fc96652e")
    ID3D12DebugCommandList2 : public ID3D12DebugCommandList
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE SetDebugParameter( 
            D3D12_DEBUG_COMMAND_LIST_PARAMETER_TYPE Type,
            _In_reads_bytes_(DataSize)  const void *pData,
            UINT DataSize) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetDebugParameter( 
            D3D12_DEBUG_COMMAND_LIST_PARAMETER_TYPE Type,
            _Out_writes_bytes_(DataSize)  void *pData,
            UINT DataSize) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12DebugCommandList2Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12DebugCommandList2 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12DebugCommandList2 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12DebugCommandList2 * This);
        
        DECLSPEC_XFGVIRT(ID3D12DebugCommandList, AssertResourceState)
        BOOL ( STDMETHODCALLTYPE *AssertResourceState )( 
            ID3D12DebugCommandList2 * This,
            _In_  ID3D12Resource *pResource,
            UINT Subresource,
            UINT State);
        
        DECLSPEC_XFGVIRT(ID3D12DebugCommandList, SetFeatureMask)
        HRESULT ( STDMETHODCALLTYPE *SetFeatureMask )( 
            ID3D12DebugCommandList2 * This,
            D3D12_DEBUG_FEATURE Mask);
        
        DECLSPEC_XFGVIRT(ID3D12DebugCommandList, GetFeatureMask)
        D3D12_DEBUG_FEATURE ( STDMETHODCALLTYPE *GetFeatureMask )( 
            ID3D12DebugCommandList2 * This);
        
        DECLSPEC_XFGVIRT(ID3D12DebugCommandList2, SetDebugParameter)
        HRESULT ( STDMETHODCALLTYPE *SetDebugParameter )( 
            ID3D12DebugCommandList2 * This,
            D3D12_DEBUG_COMMAND_LIST_PARAMETER_TYPE Type,
            _In_reads_bytes_(DataSize)  const void *pData,
            UINT DataSize);
        
        DECLSPEC_XFGVIRT(ID3D12DebugCommandList2, GetDebugParameter)
        HRESULT ( STDMETHODCALLTYPE *GetDebugParameter )( 
            ID3D12DebugCommandList2 * This,
            D3D12_DEBUG_COMMAND_LIST_PARAMETER_TYPE Type,
            _Out_writes_bytes_(DataSize)  void *pData,
            UINT DataSize);
        
        END_INTERFACE
    } ID3D12DebugCommandList2Vtbl;

    interface ID3D12DebugCommandList2
    {
        CONST_VTBL struct ID3D12DebugCommandList2Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12DebugCommandList2_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12DebugCommandList2_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12DebugCommandList2_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12DebugCommandList2_AssertResourceState(This,pResource,Subresource,State)	\
    ( (This)->lpVtbl -> AssertResourceState(This,pResource,Subresource,State) ) 

#define ID3D12DebugCommandList2_SetFeatureMask(This,Mask)	\
    ( (This)->lpVtbl -> SetFeatureMask(This,Mask) ) 

#define ID3D12DebugCommandList2_GetFeatureMask(This)	\
    ( (This)->lpVtbl -> GetFeatureMask(This) ) 


#define ID3D12DebugCommandList2_SetDebugParameter(This,Type,pData,DataSize)	\
    ( (This)->lpVtbl -> SetDebugParameter(This,Type,pData,DataSize) ) 

#define ID3D12DebugCommandList2_GetDebugParameter(This,Type,pData,DataSize)	\
    ( (This)->lpVtbl -> GetDebugParameter(This,Type,pData,DataSize) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12DebugCommandList2_INTERFACE_DEFINED__ */


#ifndef __ID3D12DebugCommandList3_INTERFACE_DEFINED__
#define __ID3D12DebugCommandList3_INTERFACE_DEFINED__

/* interface ID3D12DebugCommandList3 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12DebugCommandList3;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("197d5e15-4d37-4d34-af78-724cd70fdb1f")
    ID3D12DebugCommandList3 : public ID3D12DebugCommandList2
    {
    public:
        virtual void STDMETHODCALLTYPE AssertResourceAccess( 
            _In_  ID3D12Resource *pResource,
            UINT Subresource,
            D3D12_BARRIER_ACCESS Access) = 0;
        
        virtual void STDMETHODCALLTYPE AssertTextureLayout( 
            _In_  ID3D12Resource *pResource,
            UINT Subresource,
            D3D12_BARRIER_LAYOUT Layout) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12DebugCommandList3Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12DebugCommandList3 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12DebugCommandList3 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12DebugCommandList3 * This);
        
        DECLSPEC_XFGVIRT(ID3D12DebugCommandList, AssertResourceState)
        BOOL ( STDMETHODCALLTYPE *AssertResourceState )( 
            ID3D12DebugCommandList3 * This,
            _In_  ID3D12Resource *pResource,
            UINT Subresource,
            UINT State);
        
        DECLSPEC_XFGVIRT(ID3D12DebugCommandList, SetFeatureMask)
        HRESULT ( STDMETHODCALLTYPE *SetFeatureMask )( 
            ID3D12DebugCommandList3 * This,
            D3D12_DEBUG_FEATURE Mask);
        
        DECLSPEC_XFGVIRT(ID3D12DebugCommandList, GetFeatureMask)
        D3D12_DEBUG_FEATURE ( STDMETHODCALLTYPE *GetFeatureMask )( 
            ID3D12DebugCommandList3 * This);
        
        DECLSPEC_XFGVIRT(ID3D12DebugCommandList2, SetDebugParameter)
        HRESULT ( STDMETHODCALLTYPE *SetDebugParameter )( 
            ID3D12DebugCommandList3 * This,
            D3D12_DEBUG_COMMAND_LIST_PARAMETER_TYPE Type,
            _In_reads_bytes_(DataSize)  const void *pData,
            UINT DataSize);
        
        DECLSPEC_XFGVIRT(ID3D12DebugCommandList2, GetDebugParameter)
        HRESULT ( STDMETHODCALLTYPE *GetDebugParameter )( 
            ID3D12DebugCommandList3 * This,
            D3D12_DEBUG_COMMAND_LIST_PARAMETER_TYPE Type,
            _Out_writes_bytes_(DataSize)  void *pData,
            UINT DataSize);
        
        DECLSPEC_XFGVIRT(ID3D12DebugCommandList3, AssertResourceAccess)
        void ( STDMETHODCALLTYPE *AssertResourceAccess )( 
            ID3D12DebugCommandList3 * This,
            _In_  ID3D12Resource *pResource,
            UINT Subresource,
            D3D12_BARRIER_ACCESS Access);
        
        DECLSPEC_XFGVIRT(ID3D12DebugCommandList3, AssertTextureLayout)
        void ( STDMETHODCALLTYPE *AssertTextureLayout )( 
            ID3D12DebugCommandList3 * This,
            _In_  ID3D12Resource *pResource,
            UINT Subresource,
            D3D12_BARRIER_LAYOUT Layout);
        
        END_INTERFACE
    } ID3D12DebugCommandList3Vtbl;

    interface ID3D12DebugCommandList3
    {
        CONST_VTBL struct ID3D12DebugCommandList3Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12DebugCommandList3_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12DebugCommandList3_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12DebugCommandList3_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12DebugCommandList3_AssertResourceState(This,pResource,Subresource,State)	\
    ( (This)->lpVtbl -> AssertResourceState(This,pResource,Subresource,State) ) 

#define ID3D12DebugCommandList3_SetFeatureMask(This,Mask)	\
    ( (This)->lpVtbl -> SetFeatureMask(This,Mask) ) 

#define ID3D12DebugCommandList3_GetFeatureMask(This)	\
    ( (This)->lpVtbl -> GetFeatureMask(This) ) 


#define ID3D12DebugCommandList3_SetDebugParameter(This,Type,pData,DataSize)	\
    ( (This)->lpVtbl -> SetDebugParameter(This,Type,pData,DataSize) ) 

#define ID3D12DebugCommandList3_GetDebugParameter(This,Type,pData,DataSize)	\
    ( (This)->lpVtbl -> GetDebugParameter(This,Type,pData,DataSize) ) 


#define ID3D12DebugCommandList3_AssertResourceAccess(This,pResource,Subresource,Access)	\
    ( (This)->lpVtbl -> AssertResourceAccess(This,pResource,Subresource,Access) ) 

#define ID3D12DebugCommandList3_AssertTextureLayout(This,pResource,Subresource,Layout)	\
    ( (This)->lpVtbl -> AssertTextureLayout(This,pResource,Subresource,Layout) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12DebugCommandList3_INTERFACE_DEFINED__ */


#ifndef __ID3D12SharingContract_INTERFACE_DEFINED__
#define __ID3D12SharingContract_INTERFACE_DEFINED__

/* interface ID3D12SharingContract */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12SharingContract;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("0adf7d52-929c-4e61-addb-ffed30de66ef")
    ID3D12SharingContract : public IUnknown
    {
    public:
        virtual void STDMETHODCALLTYPE Present( 
            _In_  ID3D12Resource *pResource,
            UINT Subresource,
            _In_  HWND window) = 0;
        
        virtual void STDMETHODCALLTYPE SharedFenceSignal( 
            _In_  ID3D12Fence *pFence,
            UINT64 FenceValue) = 0;
        
        virtual void STDMETHODCALLTYPE BeginCapturableWork( 
            _In_  REFGUID guid) = 0;
        
        virtual void STDMETHODCALLTYPE EndCapturableWork( 
            _In_  REFGUID guid) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12SharingContractVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12SharingContract * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12SharingContract * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12SharingContract * This);
        
        DECLSPEC_XFGVIRT(ID3D12SharingContract, Present)
        void ( STDMETHODCALLTYPE *Present )( 
            ID3D12SharingContract * This,
            _In_  ID3D12Resource *pResource,
            UINT Subresource,
            _In_  HWND window);
        
        DECLSPEC_XFGVIRT(ID3D12SharingContract, SharedFenceSignal)
        void ( STDMETHODCALLTYPE *SharedFenceSignal )( 
            ID3D12SharingContract * This,
            _In_  ID3D12Fence *pFence,
            UINT64 FenceValue);
        
        DECLSPEC_XFGVIRT(ID3D12SharingContract, BeginCapturableWork)
        void ( STDMETHODCALLTYPE *BeginCapturableWork )( 
            ID3D12SharingContract * This,
            _In_  REFGUID guid);
        
        DECLSPEC_XFGVIRT(ID3D12SharingContract, EndCapturableWork)
        void ( STDMETHODCALLTYPE *EndCapturableWork )( 
            ID3D12SharingContract * This,
            _In_  REFGUID guid);
        
        END_INTERFACE
    } ID3D12SharingContractVtbl;

    interface ID3D12SharingContract
    {
        CONST_VTBL struct ID3D12SharingContractVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12SharingContract_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12SharingContract_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12SharingContract_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12SharingContract_Present(This,pResource,Subresource,window)	\
    ( (This)->lpVtbl -> Present(This,pResource,Subresource,window) ) 

#define ID3D12SharingContract_SharedFenceSignal(This,pFence,FenceValue)	\
    ( (This)->lpVtbl -> SharedFenceSignal(This,pFence,FenceValue) ) 

#define ID3D12SharingContract_BeginCapturableWork(This,guid)	\
    ( (This)->lpVtbl -> BeginCapturableWork(This,guid) ) 

#define ID3D12SharingContract_EndCapturableWork(This,guid)	\
    ( (This)->lpVtbl -> EndCapturableWork(This,guid) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12SharingContract_INTERFACE_DEFINED__ */


#ifndef __ID3D12ManualWriteTrackingResource_INTERFACE_DEFINED__
#define __ID3D12ManualWriteTrackingResource_INTERFACE_DEFINED__

/* interface ID3D12ManualWriteTrackingResource */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12ManualWriteTrackingResource;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("86ca3b85-49ad-4b6e-aed5-eddb18540f41")
    ID3D12ManualWriteTrackingResource : public IUnknown
    {
    public:
        virtual void STDMETHODCALLTYPE TrackWrite( 
            UINT Subresource,
            _In_opt_  const D3D12_RANGE *pWrittenRange) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12ManualWriteTrackingResourceVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12ManualWriteTrackingResource * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12ManualWriteTrackingResource * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12ManualWriteTrackingResource * This);
        
        DECLSPEC_XFGVIRT(ID3D12ManualWriteTrackingResource, TrackWrite)
        void ( STDMETHODCALLTYPE *TrackWrite )( 
            ID3D12ManualWriteTrackingResource * This,
            UINT Subresource,
            _In_opt_  const D3D12_RANGE *pWrittenRange);
        
        END_INTERFACE
    } ID3D12ManualWriteTrackingResourceVtbl;

    interface ID3D12ManualWriteTrackingResource
    {
        CONST_VTBL struct ID3D12ManualWriteTrackingResourceVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12ManualWriteTrackingResource_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12ManualWriteTrackingResource_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12ManualWriteTrackingResource_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12ManualWriteTrackingResource_TrackWrite(This,Subresource,pWrittenRange)	\
    ( (This)->lpVtbl -> TrackWrite(This,Subresource,pWrittenRange) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12ManualWriteTrackingResource_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12sdklayers_0000_0018 */
/* [local] */ 

typedef 
enum D3D12_MESSAGE_CATEGORY
    {
        D3D12_MESSAGE_CATEGORY_APPLICATION_DEFINED	= 0,
        D3D12_MESSAGE_CATEGORY_MISCELLANEOUS	= ( D3D12_MESSAGE_CATEGORY_APPLICATION_DEFINED + 1 ) ,
        D3D12_MESSAGE_CATEGORY_INITIALIZATION	= ( D3D12_MESSAGE_CATEGORY_MISCELLANEOUS + 1 ) ,
        D3D12_MESSAGE_CATEGORY_CLEANUP	= ( D3D12_MESSAGE_CATEGORY_INITIALIZATION + 1 ) ,
        D3D12_MESSAGE_CATEGORY_COMPILATION	= ( D3D12_MESSAGE_CATEGORY_CLEANUP + 1 ) ,
        D3D12_MESSAGE_CATEGORY_STATE_CREATION	= ( D3D12_MESSAGE_CATEGORY_COMPILATION + 1 ) ,
        D3D12_MESSAGE_CATEGORY_STATE_SETTING	= ( D3D12_MESSAGE_CATEGORY_STATE_CREATION + 1 ) ,
        D3D12_MESSAGE_CATEGORY_STATE_GETTING	= ( D3D12_MESSAGE_CATEGORY_STATE_SETTING + 1 ) ,
        D3D12_MESSAGE_CATEGORY_RESOURCE_MANIPULATION	= ( D3D12_MESSAGE_CATEGORY_STATE_GETTING + 1 ) ,
        D3D12_MESSAGE_CATEGORY_EXECUTION	= ( D3D12_MESSAGE_CATEGORY_RESOURCE_MANIPULATION + 1 ) ,
        D3D12_MESSAGE_CATEGORY_SHADER	= ( D3D12_MESSAGE_CATEGORY_EXECUTION + 1 ) 
    } 	D3D12_MESSAGE_CATEGORY;

typedef 
enum D3D12_MESSAGE_SEVERITY
    {
        D3D12_MESSAGE_SEVERITY_CORRUPTION	= 0,
        D3D12_MESSAGE_SEVERITY_ERROR	= ( D3D12_MESSAGE_SEVERITY_CORRUPTION + 1 ) ,
        D3D12_MESSAGE_SEVERITY_WARNING	= ( D3D12_MESSAGE_SEVERITY_ERROR + 1 ) ,
        D3D12_MESSAGE_SEVERITY_INFO	= ( D3D12_MESSAGE_SEVERITY_WARNING + 1 ) ,
        D3D12_MESSAGE_SEVERITY_MESSAGE	= ( D3D12_MESSAGE_SEVERITY_INFO + 1 ) 
    } 	D3D12_MESSAGE_SEVERITY;

typedef 
enum D3D12_MESSAGE_ID
    {
        D3D12_MESSAGE_ID_UNKNOWN	= 0,
        D3D12_MESSAGE_ID_STRING_FROM_APPLICATION	= 1,
        D3D12_MESSAGE_ID_CORRUPTED_THIS	= 2,
        D3D12_MESSAGE_ID_CORRUPTED_PARAMETER1	= 3,
        D3D12_MESSAGE_ID_CORRUPTED_PARAMETER2	= 4,
        D3D12_MESSAGE_ID_CORRUPTED_PARAMETER3	= 5,
        D3D12_MESSAGE_ID_CORRUPTED_PARAMETER4	= 6,
        D3D12_MESSAGE_ID_CORRUPTED_PARAMETER5	= 7,
        D3D12_MESSAGE_ID_CORRUPTED_PARAMETER6	= 8,
        D3D12_MESSAGE_ID_CORRUPTED_PARAMETER7	= 9,
        D3D12_MESSAGE_ID_CORRUPTED_PARAMETER8	= 10,
        D3D12_MESSAGE_ID_CORRUPTED_PARAMETER9	= 11,
        D3D12_MESSAGE_ID_CORRUPTED_PARAMETER10	= 12,
        D3D12_MESSAGE_ID_CORRUPTED_PARAMETER11	= 13,
        D3D12_MESSAGE_ID_CORRUPTED_PARAMETER12	= 14,
        D3D12_MESSAGE_ID_CORRUPTED_PARAMETER13	= 15,
        D3D12_MESSAGE_ID_CORRUPTED_PARAMETER14	= 16,
        D3D12_MESSAGE_ID_CORRUPTED_PARAMETER15	= 17,
        D3D12_MESSAGE_ID_CORRUPTED_MULTITHREADING	= 18,
        D3D12_MESSAGE_ID_MESSAGE_REPORTING_OUTOFMEMORY	= 19,
        D3D12_MESSAGE_ID_GETPRIVATEDATA_MOREDATA	= 20,
        D3D12_MESSAGE_ID_SETPRIVATEDATA_INVALIDFREEDATA	= 21,
        D3D12_MESSAGE_ID_SETPRIVATEDATA_CHANGINGPARAMS	= 24,
        D3D12_MESSAGE_ID_SETPRIVATEDATA_OUTOFMEMORY	= 25,
        D3D12_MESSAGE_ID_CREATESHADERRESOURCEVIEW_UNRECOGNIZEDFORMAT	= 26,
        D3D12_MESSAGE_ID_CREATESHADERRESOURCEVIEW_INVALIDDESC	= 27,
        D3D12_MESSAGE_ID_CREATESHADERRESOURCEVIEW_INVALIDFORMAT	= 28,
        D3D12_MESSAGE_ID_CREATESHADERRESOURCEVIEW_INVALIDVIDEOPLANESLICE	= 29,
        D3D12_MESSAGE_ID_CREATESHADERRESOURCEVIEW_INVALIDPLANESLICE	= 30,
        D3D12_MESSAGE_ID_CREATESHADERRESOURCEVIEW_INVALIDDIMENSIONS	= 31,
        D3D12_MESSAGE_ID_CREATESHADERRESOURCEVIEW_INVALIDRESOURCE	= 32,
        D3D12_MESSAGE_ID_CREATERENDERTARGETVIEW_UNRECOGNIZEDFORMAT	= 35,
        D3D12_MESSAGE_ID_CREATERENDERTARGETVIEW_UNSUPPORTEDFORMAT	= 36,
        D3D12_MESSAGE_ID_CREATERENDERTARGETVIEW_INVALIDDESC	= 37,
        D3D12_MESSAGE_ID_CREATERENDERTARGETVIEW_INVALIDFORMAT	= 38,
        D3D12_MESSAGE_ID_CREATERENDERTARGETVIEW_INVALIDVIDEOPLANESLICE	= 39,
        D3D12_MESSAGE_ID_CREATERENDERTARGETVIEW_INVALIDPLANESLICE	= 40,
        D3D12_MESSAGE_ID_CREATERENDERTARGETVIEW_INVALIDDIMENSIONS	= 41,
        D3D12_MESSAGE_ID_CREATERENDERTARGETVIEW_INVALIDRESOURCE	= 42,
        D3D12_MESSAGE_ID_CREATEDEPTHSTENCILVIEW_UNRECOGNIZEDFORMAT	= 45,
        D3D12_MESSAGE_ID_CREATEDEPTHSTENCILVIEW_INVALIDDESC	= 46,
        D3D12_MESSAGE_ID_CREATEDEPTHSTENCILVIEW_INVALIDFORMAT	= 47,
        D3D12_MESSAGE_ID_CREATEDEPTHSTENCILVIEW_INVALIDDIMENSIONS	= 48,
        D3D12_MESSAGE_ID_CREATEDEPTHSTENCILVIEW_INVALIDRESOURCE	= 49,
        D3D12_MESSAGE_ID_CREATEINPUTLAYOUT_OUTOFMEMORY	= 52,
        D3D12_MESSAGE_ID_CREATEINPUTLAYOUT_TOOMANYELEMENTS	= 53,
        D3D12_MESSAGE_ID_CREATEINPUTLAYOUT_INVALIDFORMAT	= 54,
        D3D12_MESSAGE_ID_CREATEINPUTLAYOUT_INCOMPATIBLEFORMAT	= 55,
        D3D12_MESSAGE_ID_CREATEINPUTLAYOUT_INVALIDSLOT	= 56,
        D3D12_MESSAGE_ID_CREATEINPUTLAYOUT_INVALIDINPUTSLOTCLASS	= 57,
        D3D12_MESSAGE_ID_CREATEINPUTLAYOUT_STEPRATESLOTCLASSMISMATCH	= 58,
        D3D12_MESSAGE_ID_CREATEINPUTLAYOUT_INVALIDSLOTCLASSCHANGE	= 59,
        D3D12_MESSAGE_ID_CREATEINPUTLAYOUT_INVALIDSTEPRATECHANGE	= 60,
        D3D12_MESSAGE_ID_CREATEINPUTLAYOUT_INVALIDALIGNMENT	= 61,
        D3D12_MESSAGE_ID_CREATEINPUTLAYOUT_DUPLICATESEMANTIC	= 62,
        D3D12_MESSAGE_ID_CREATEINPUTLAYOUT_UNPARSEABLEINPUTSIGNATURE	= 63,
        D3D12_MESSAGE_ID_CREATEINPUTLAYOUT_NULLSEMANTIC	= 64,
        D3D12_MESSAGE_ID_CREATEINPUTLAYOUT_MISSINGELEMENT	= 65,
        D3D12_MESSAGE_ID_CREATEVERTEXSHADER_OUTOFMEMORY	= 66,
        D3D12_MESSAGE_ID_CREATEVERTEXSHADER_INVALIDSHADERBYTECODE	= 67,
        D3D12_MESSAGE_ID_CREATEVERTEXSHADER_INVALIDSHADERTYPE	= 68,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADER_OUTOFMEMORY	= 69,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADER_INVALIDSHADERBYTECODE	= 70,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADER_INVALIDSHADERTYPE	= 71,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_OUTOFMEMORY	= 72,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_INVALIDSHADERBYTECODE	= 73,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_INVALIDSHADERTYPE	= 74,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_INVALIDNUMENTRIES	= 75,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_OUTPUTSTREAMSTRIDEUNUSED	= 76,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_OUTPUTSLOT0EXPECTED	= 79,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_INVALIDOUTPUTSLOT	= 80,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_ONLYONEELEMENTPERSLOT	= 81,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_INVALIDCOMPONENTCOUNT	= 82,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_INVALIDSTARTCOMPONENTANDCOMPONENTCOUNT	= 83,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_INVALIDGAPDEFINITION	= 84,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_REPEATEDOUTPUT	= 85,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_INVALIDOUTPUTSTREAMSTRIDE	= 86,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_MISSINGSEMANTIC	= 87,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_MASKMISMATCH	= 88,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_CANTHAVEONLYGAPS	= 89,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_DECLTOOCOMPLEX	= 90,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_MISSINGOUTPUTSIGNATURE	= 91,
        D3D12_MESSAGE_ID_CREATEPIXELSHADER_OUTOFMEMORY	= 92,
        D3D12_MESSAGE_ID_CREATEPIXELSHADER_INVALIDSHADERBYTECODE	= 93,
        D3D12_MESSAGE_ID_CREATEPIXELSHADER_INVALIDSHADERTYPE	= 94,
        D3D12_MESSAGE_ID_CREATERASTERIZERSTATE_INVALIDFILLMODE	= 95,
        D3D12_MESSAGE_ID_CREATERASTERIZERSTATE_INVALIDCULLMODE	= 96,
        D3D12_MESSAGE_ID_CREATERASTERIZERSTATE_INVALIDDEPTHBIASCLAMP	= 97,
        D3D12_MESSAGE_ID_CREATERASTERIZERSTATE_INVALIDSLOPESCALEDDEPTHBIAS	= 98,
        D3D12_MESSAGE_ID_CREATEDEPTHSTENCILSTATE_INVALIDDEPTHWRITEMASK	= 100,
        D3D12_MESSAGE_ID_CREATEDEPTHSTENCILSTATE_INVALIDDEPTHFUNC	= 101,
        D3D12_MESSAGE_ID_CREATEDEPTHSTENCILSTATE_INVALIDFRONTFACESTENCILFAILOP	= 102,
        D3D12_MESSAGE_ID_CREATEDEPTHSTENCILSTATE_INVALIDFRONTFACESTENCILZFAILOP	= 103,
        D3D12_MESSAGE_ID_CREATEDEPTHSTENCILSTATE_INVALIDFRONTFACESTENCILPASSOP	= 104,
        D3D12_MESSAGE_ID_CREATEDEPTHSTENCILSTATE_INVALIDFRONTFACESTENCILFUNC	= 105,
        D3D12_MESSAGE_ID_CREATEDEPTHSTENCILSTATE_INVALIDBACKFACESTENCILFAILOP	= 106,
        D3D12_MESSAGE_ID_CREATEDEPTHSTENCILSTATE_INVALIDBACKFACESTENCILZFAILOP	= 107,
        D3D12_MESSAGE_ID_CREATEDEPTHSTENCILSTATE_INVALIDBACKFACESTENCILPASSOP	= 108,
        D3D12_MESSAGE_ID_CREATEDEPTHSTENCILSTATE_INVALIDBACKFACESTENCILFUNC	= 109,
        D3D12_MESSAGE_ID_CREATEBLENDSTATE_INVALIDSRCBLEND	= 111,
        D3D12_MESSAGE_ID_CREATEBLENDSTATE_INVALIDDESTBLEND	= 112,
        D3D12_MESSAGE_ID_CREATEBLENDSTATE_INVALIDBLENDOP	= 113,
        D3D12_MESSAGE_ID_CREATEBLENDSTATE_INVALIDSRCBLENDALPHA	= 114,
        D3D12_MESSAGE_ID_CREATEBLENDSTATE_INVALIDDESTBLENDALPHA	= 115,
        D3D12_MESSAGE_ID_CREATEBLENDSTATE_INVALIDBLENDOPALPHA	= 116,
        D3D12_MESSAGE_ID_CREATEBLENDSTATE_INVALIDRENDERTARGETWRITEMASK	= 117,
        D3D12_MESSAGE_ID_GET_PROGRAM_IDENTIFIER_ERROR	= 118,
        D3D12_MESSAGE_ID_GET_WORK_GRAPH_PROPERTIES_ERROR	= 119,
        D3D12_MESSAGE_ID_SET_PROGRAM_ERROR	= 120,
        D3D12_MESSAGE_ID_CLEARDEPTHSTENCILVIEW_INVALID	= 135,
        D3D12_MESSAGE_ID_COMMAND_LIST_DRAW_ROOT_SIGNATURE_NOT_SET	= 200,
        D3D12_MESSAGE_ID_COMMAND_LIST_DRAW_ROOT_SIGNATURE_MISMATCH	= 201,
        D3D12_MESSAGE_ID_COMMAND_LIST_DRAW_VERTEX_BUFFER_NOT_SET	= 202,
        D3D12_MESSAGE_ID_COMMAND_LIST_DRAW_VERTEX_BUFFER_STRIDE_TOO_SMALL	= 209,
        D3D12_MESSAGE_ID_COMMAND_LIST_DRAW_VERTEX_BUFFER_TOO_SMALL	= 210,
        D3D12_MESSAGE_ID_COMMAND_LIST_DRAW_INDEX_BUFFER_NOT_SET	= 211,
        D3D12_MESSAGE_ID_COMMAND_LIST_DRAW_INDEX_BUFFER_FORMAT_INVALID	= 212,
        D3D12_MESSAGE_ID_COMMAND_LIST_DRAW_INDEX_BUFFER_TOO_SMALL	= 213,
        D3D12_MESSAGE_ID_COMMAND_LIST_DRAW_INVALID_PRIMITIVETOPOLOGY	= 219,
        D3D12_MESSAGE_ID_COMMAND_LIST_DRAW_VERTEX_STRIDE_UNALIGNED	= 221,
        D3D12_MESSAGE_ID_COMMAND_LIST_DRAW_INDEX_OFFSET_UNALIGNED	= 222,
        D3D12_MESSAGE_ID_DEVICE_REMOVAL_PROCESS_AT_FAULT	= 232,
        D3D12_MESSAGE_ID_DEVICE_REMOVAL_PROCESS_POSSIBLY_AT_FAULT	= 233,
        D3D12_MESSAGE_ID_DEVICE_REMOVAL_PROCESS_NOT_AT_FAULT	= 234,
        D3D12_MESSAGE_ID_CREATEINPUTLAYOUT_TRAILING_DIGIT_IN_SEMANTIC	= 239,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_TRAILING_DIGIT_IN_SEMANTIC	= 240,
        D3D12_MESSAGE_ID_CREATEINPUTLAYOUT_TYPE_MISMATCH	= 245,
        D3D12_MESSAGE_ID_CREATEINPUTLAYOUT_EMPTY_LAYOUT	= 253,
        D3D12_MESSAGE_ID_LIVE_OBJECT_SUMMARY	= 255,
        D3D12_MESSAGE_ID_LIVE_DEVICE	= 274,
        D3D12_MESSAGE_ID_LIVE_SWAPCHAIN	= 275,
        D3D12_MESSAGE_ID_CREATEDEPTHSTENCILVIEW_INVALIDFLAGS	= 276,
        D3D12_MESSAGE_ID_CREATEVERTEXSHADER_INVALIDCLASSLINKAGE	= 277,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADER_INVALIDCLASSLINKAGE	= 278,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_INVALIDSTREAMTORASTERIZER	= 280,
        D3D12_MESSAGE_ID_CREATEPIXELSHADER_INVALIDCLASSLINKAGE	= 283,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_INVALIDSTREAM	= 284,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_UNEXPECTEDENTRIES	= 285,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_UNEXPECTEDSTRIDES	= 286,
        D3D12_MESSAGE_ID_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_INVALIDNUMSTRIDES	= 287,
        D3D12_MESSAGE_ID_CREATEHULLSHADER_OUTOFMEMORY	= 289,
        D3D12_MESSAGE_ID_CREATEHULLSHADER_INVALIDSHADERBYTECODE	= 290,
        D3D12_MESSAGE_ID_CREATEHULLSHADER_INVALIDSHADERTYPE	= 291,
        D3D12_MESSAGE_ID_CREATEHULLSHADER_INVALIDCLASSLINKAGE	= 292,
        D3D12_MESSAGE_ID_CREATEDOMAINSHADER_OUTOFMEMORY	= 294,
        D3D12_MESSAGE_ID_CREATEDOMAINSHADER_INVALIDSHADERBYTECODE	= 295,
        D3D12_MESSAGE_ID_CREATEDOMAINSHADER_INVALIDSHADERTYPE	= 296,
        D3D12_MESSAGE_ID_CREATEDOMAINSHADER_INVALIDCLASSLINKAGE	= 297,
        D3D12_MESSAGE_ID_RESOURCE_UNMAP_NOTMAPPED	= 310,
        D3D12_MESSAGE_ID_DEVICE_CHECKFEATURESUPPORT_MISMATCHED_DATA_SIZE	= 318,
        D3D12_MESSAGE_ID_CREATECOMPUTESHADER_OUTOFMEMORY	= 321,
        D3D12_MESSAGE_ID_CREATECOMPUTESHADER_INVALIDSHADERBYTECODE	= 322,
        D3D12_MESSAGE_ID_CREATECOMPUTESHADER_INVALIDCLASSLINKAGE	= 323,
        D3D12_MESSAGE_ID_DEVICE_CREATEVERTEXSHADER_DOUBLEFLOATOPSNOTSUPPORTED	= 331,
        D3D12_MESSAGE_ID_DEVICE_CREATEHULLSHADER_DOUBLEFLOATOPSNOTSUPPORTED	= 332,
        D3D12_MESSAGE_ID_DEVICE_CREATEDOMAINSHADER_DOUBLEFLOATOPSNOTSUPPORTED	= 333,
        D3D12_MESSAGE_ID_DEVICE_CREATEGEOMETRYSHADER_DOUBLEFLOATOPSNOTSUPPORTED	= 334,
        D3D12_MESSAGE_ID_DEVICE_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_DOUBLEFLOATOPSNOTSUPPORTED	= 335,
        D3D12_MESSAGE_ID_DEVICE_CREATEPIXELSHADER_DOUBLEFLOATOPSNOTSUPPORTED	= 336,
        D3D12_MESSAGE_ID_DEVICE_CREATECOMPUTESHADER_DOUBLEFLOATOPSNOTSUPPORTED	= 337,
        D3D12_MESSAGE_ID_CREATEUNORDEREDACCESSVIEW_INVALIDRESOURCE	= 340,
        D3D12_MESSAGE_ID_CREATEUNORDEREDACCESSVIEW_INVALIDDESC	= 341,
        D3D12_MESSAGE_ID_CREATEUNORDEREDACCESSVIEW_INVALIDFORMAT	= 342,
        D3D12_MESSAGE_ID_CREATEUNORDEREDACCESSVIEW_INVALIDVIDEOPLANESLICE	= 343,
        D3D12_MESSAGE_ID_CREATEUNORDEREDACCESSVIEW_INVALIDPLANESLICE	= 344,
        D3D12_MESSAGE_ID_CREATEUNORDEREDACCESSVIEW_INVALIDDIMENSIONS	= 345,
        D3D12_MESSAGE_ID_CREATEUNORDEREDACCESSVIEW_UNRECOGNIZEDFORMAT	= 346,
        D3D12_MESSAGE_ID_CREATEUNORDEREDACCESSVIEW_INVALIDFLAGS	= 354,
        D3D12_MESSAGE_ID_CREATERASTERIZERSTATE_INVALIDFORCEDSAMPLECOUNT	= 401,
        D3D12_MESSAGE_ID_CREATEBLENDSTATE_INVALIDLOGICOPS	= 403,
        D3D12_MESSAGE_ID_DEVICE_CREATEVERTEXSHADER_DOUBLEEXTENSIONSNOTSUPPORTED	= 410,
        D3D12_MESSAGE_ID_DEVICE_CREATEHULLSHADER_DOUBLEEXTENSIONSNOTSUPPORTED	= 412,
        D3D12_MESSAGE_ID_DEVICE_CREATEDOMAINSHADER_DOUBLEEXTENSIONSNOTSUPPORTED	= 414,
        D3D12_MESSAGE_ID_DEVICE_CREATEGEOMETRYSHADER_DOUBLEEXTENSIONSNOTSUPPORTED	= 416,
        D3D12_MESSAGE_ID_DEVICE_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_DOUBLEEXTENSIONSNOTSUPPORTED	= 418,
        D3D12_MESSAGE_ID_DEVICE_CREATEPIXELSHADER_DOUBLEEXTENSIONSNOTSUPPORTED	= 420,
        D3D12_MESSAGE_ID_DEVICE_CREATECOMPUTESHADER_DOUBLEEXTENSIONSNOTSUPPORTED	= 422,
        D3D12_MESSAGE_ID_DEVICE_CREATEVERTEXSHADER_UAVSNOTSUPPORTED	= 425,
        D3D12_MESSAGE_ID_DEVICE_CREATEHULLSHADER_UAVSNOTSUPPORTED	= 426,
        D3D12_MESSAGE_ID_DEVICE_CREATEDOMAINSHADER_UAVSNOTSUPPORTED	= 427,
        D3D12_MESSAGE_ID_DEVICE_CREATEGEOMETRYSHADER_UAVSNOTSUPPORTED	= 428,
        D3D12_MESSAGE_ID_DEVICE_CREATEGEOMETRYSHADERWITHSTREAMOUTPUT_UAVSNOTSUPPORTED	= 429,
        D3D12_MESSAGE_ID_DEVICE_CREATEPIXELSHADER_UAVSNOTSUPPORTED	= 430,
        D3D12_MESSAGE_ID_DEVICE_CREATECOMPUTESHADER_UAVSNOTSUPPORTED	= 431,
        D3D12_MESSAGE_ID_DEVICE_CLEARVIEW_INVALIDSOURCERECT	= 447,
        D3D12_MESSAGE_ID_DEVICE_CLEARVIEW_EMPTYRECT	= 448,
        D3D12_MESSAGE_ID_UPDATETILEMAPPINGS_INVALID_PARAMETER	= 493,
        D3D12_MESSAGE_ID_COPYTILEMAPPINGS_INVALID_PARAMETER	= 494,
        D3D12_MESSAGE_ID_CREATEDEVICE_INVALIDARGS	= 506,
        D3D12_MESSAGE_ID_CREATEDEVICE_WARNING	= 507,
        D3D12_MESSAGE_ID_RESOURCE_BARRIER_INVALID_TYPE	= 519,
        D3D12_MESSAGE_ID_RESOURCE_BARRIER_NULL_POINTER	= 520,
        D3D12_MESSAGE_ID_RESOURCE_BARRIER_INVALID_SUBRESOURCE	= 521,
        D3D12_MESSAGE_ID_RESOURCE_BARRIER_RESERVED_BITS	= 522,
        D3D12_MESSAGE_ID_RESOURCE_BARRIER_MISSING_BIND_FLAGS	= 523,
        D3D12_MESSAGE_ID_RESOURCE_BARRIER_MISMATCHING_MISC_FLAGS	= 524,
        D3D12_MESSAGE_ID_RESOURCE_BARRIER_MATCHING_STATES	= 525,
        D3D12_MESSAGE_ID_RESOURCE_BARRIER_INVALID_COMBINATION	= 526,
        D3D12_MESSAGE_ID_RESOURCE_BARRIER_BEFORE_AFTER_MISMATCH	= 527,
        D3D12_MESSAGE_ID_RESOURCE_BARRIER_INVALID_RESOURCE	= 528,
        D3D12_MESSAGE_ID_RESOURCE_BARRIER_SAMPLE_COUNT	= 529,
        D3D12_MESSAGE_ID_RESOURCE_BARRIER_INVALID_FLAGS	= 530,
        D3D12_MESSAGE_ID_RESOURCE_BARRIER_INVALID_COMBINED_FLAGS	= 531,
        D3D12_MESSAGE_ID_RESOURCE_BARRIER_INVALID_FLAGS_FOR_FORMAT	= 532,
        D3D12_MESSAGE_ID_RESOURCE_BARRIER_INVALID_SPLIT_BARRIER	= 533,
        D3D12_MESSAGE_ID_RESOURCE_BARRIER_UNMATCHED_END	= 534,
        D3D12_MESSAGE_ID_RESOURCE_BARRIER_UNMATCHED_BEGIN	= 535,
        D3D12_MESSAGE_ID_RESOURCE_BARRIER_INVALID_FLAG	= 536,
        D3D12_MESSAGE_ID_RESOURCE_BARRIER_INVALID_COMMAND_LIST_TYPE	= 537,
        D3D12_MESSAGE_ID_INVALID_SUBRESOURCE_STATE	= 538,
        D3D12_MESSAGE_ID_COMMAND_ALLOCATOR_CONTENTION	= 540,
        D3D12_MESSAGE_ID_COMMAND_ALLOCATOR_RESET	= 541,
        D3D12_MESSAGE_ID_COMMAND_ALLOCATOR_RESET_BUNDLE	= 542,
        D3D12_MESSAGE_ID_COMMAND_ALLOCATOR_CANNOT_RESET	= 543,
        D3D12_MESSAGE_ID_COMMAND_LIST_OPEN	= 544,
        D3D12_MESSAGE_ID_INVALID_BUNDLE_API	= 546,
        D3D12_MESSAGE_ID_COMMAND_LIST_CLOSED	= 547,
        D3D12_MESSAGE_ID_WRONG_COMMAND_ALLOCATOR_TYPE	= 549,
        D3D12_MESSAGE_ID_COMMAND_ALLOCATOR_SYNC	= 552,
        D3D12_MESSAGE_ID_COMMAND_LIST_SYNC	= 553,
        D3D12_MESSAGE_ID_SET_DESCRIPTOR_HEAP_INVALID	= 554,
        D3D12_MESSAGE_ID_CREATE_COMMANDQUEUE	= 557,
        D3D12_MESSAGE_ID_CREATE_COMMANDALLOCATOR	= 558,
        D3D12_MESSAGE_ID_CREATE_PIPELINESTATE	= 559,
        D3D12_MESSAGE_ID_CREATE_COMMANDLIST12	= 560,
        D3D12_MESSAGE_ID_CREATE_RESOURCE	= 562,
        D3D12_MESSAGE_ID_CREATE_DESCRIPTORHEAP	= 563,
        D3D12_MESSAGE_ID_CREATE_ROOTSIGNATURE	= 564,
        D3D12_MESSAGE_ID_CREATE_LIBRARY	= 565,
        D3D12_MESSAGE_ID_CREATE_HEAP	= 566,
        D3D12_MESSAGE_ID_CREATE_MONITOREDFENCE	= 567,
        D3D12_MESSAGE_ID_CREATE_QUERYHEAP	= 568,
        D3D12_MESSAGE_ID_CREATE_COMMANDSIGNATURE	= 569,
        D3D12_MESSAGE_ID_LIVE_COMMANDQUEUE	= 570,
        D3D12_MESSAGE_ID_LIVE_COMMANDALLOCATOR	= 571,
        D3D12_MESSAGE_ID_LIVE_PIPELINESTATE	= 572,
        D3D12_MESSAGE_ID_LIVE_COMMANDLIST12	= 573,
        D3D12_MESSAGE_ID_LIVE_RESOURCE	= 575,
        D3D12_MESSAGE_ID_LIVE_DESCRIPTORHEAP	= 576,
        D3D12_MESSAGE_ID_LIVE_ROOTSIGNATURE	= 577,
        D3D12_MESSAGE_ID_LIVE_LIBRARY	= 578,
        D3D12_MESSAGE_ID_LIVE_HEAP	= 579,
        D3D12_MESSAGE_ID_LIVE_MONITOREDFENCE	= 580,
        D3D12_MESSAGE_ID_LIVE_QUERYHEAP	= 581,
        D3D12_MESSAGE_ID_LIVE_COMMANDSIGNATURE	= 582,
        D3D12_MESSAGE_ID_DESTROY_COMMANDQUEUE	= 583,
        D3D12_MESSAGE_ID_DESTROY_COMMANDALLOCATOR	= 584,
        D3D12_MESSAGE_ID_DESTROY_PIPELINESTATE	= 585,
        D3D12_MESSAGE_ID_DESTROY_COMMANDLIST12	= 586,
        D3D12_MESSAGE_ID_DESTROY_RESOURCE	= 588,
        D3D12_MESSAGE_ID_DESTROY_DESCRIPTORHEAP	= 589,
        D3D12_MESSAGE_ID_DESTROY_ROOTSIGNATURE	= 590,
        D3D12_MESSAGE_ID_DESTROY_LIBRARY	= 591,
        D3D12_MESSAGE_ID_DESTROY_HEAP	= 592,
        D3D12_MESSAGE_ID_DESTROY_MONITOREDFENCE	= 593,
        D3D12_MESSAGE_ID_DESTROY_QUERYHEAP	= 594,
        D3D12_MESSAGE_ID_DESTROY_COMMANDSIGNATURE	= 595,
        D3D12_MESSAGE_ID_CREATERESOURCE_INVALIDDIMENSIONS	= 597,
        D3D12_MESSAGE_ID_CREATERESOURCE_INVALIDMISCFLAGS	= 599,
        D3D12_MESSAGE_ID_CREATERESOURCE_INVALIDARG_RETURN	= 602,
        D3D12_MESSAGE_ID_CREATERESOURCE_OUTOFMEMORY_RETURN	= 603,
        D3D12_MESSAGE_ID_CREATERESOURCE_INVALIDDESC	= 604,
        D3D12_MESSAGE_ID_POSSIBLY_INVALID_SUBRESOURCE_STATE	= 607,
        D3D12_MESSAGE_ID_INVALID_USE_OF_NON_RESIDENT_RESOURCE	= 608,
        D3D12_MESSAGE_ID_POSSIBLE_INVALID_USE_OF_NON_RESIDENT_RESOURCE	= 609,
        D3D12_MESSAGE_ID_BUNDLE_PIPELINE_STATE_MISMATCH	= 610,
        D3D12_MESSAGE_ID_PRIMITIVE_TOPOLOGY_MISMATCH_PIPELINE_STATE	= 611,
        D3D12_MESSAGE_ID_RENDER_TARGET_FORMAT_MISMATCH_PIPELINE_STATE	= 613,
        D3D12_MESSAGE_ID_RENDER_TARGET_SAMPLE_DESC_MISMATCH_PIPELINE_STATE	= 614,
        D3D12_MESSAGE_ID_DEPTH_STENCIL_FORMAT_MISMATCH_PIPELINE_STATE	= 615,
        D3D12_MESSAGE_ID_DEPTH_STENCIL_SAMPLE_DESC_MISMATCH_PIPELINE_STATE	= 616,
        D3D12_MESSAGE_ID_CREATESHADER_INVALIDBYTECODE	= 622,
        D3D12_MESSAGE_ID_CREATEHEAP_NULLDESC	= 623,
        D3D12_MESSAGE_ID_CREATEHEAP_INVALIDSIZE	= 624,
        D3D12_MESSAGE_ID_CREATEHEAP_UNRECOGNIZEDHEAPTYPE	= 625,
        D3D12_MESSAGE_ID_CREATEHEAP_UNRECOGNIZEDCPUPAGEPROPERTIES	= 626,
        D3D12_MESSAGE_ID_CREATEHEAP_UNRECOGNIZEDMEMORYPOOL	= 627,
        D3D12_MESSAGE_ID_CREATEHEAP_INVALIDPROPERTIES	= 628,
        D3D12_MESSAGE_ID_CREATEHEAP_INVALIDALIGNMENT	= 629,
        D3D12_MESSAGE_ID_CREATEHEAP_UNRECOGNIZEDMISCFLAGS	= 630,
        D3D12_MESSAGE_ID_CREATEHEAP_INVALIDMISCFLAGS	= 631,
        D3D12_MESSAGE_ID_CREATEHEAP_INVALIDARG_RETURN	= 632,
        D3D12_MESSAGE_ID_CREATEHEAP_OUTOFMEMORY_RETURN	= 633,
        D3D12_MESSAGE_ID_CREATERESOURCEANDHEAP_NULLHEAPPROPERTIES	= 634,
        D3D12_MESSAGE_ID_CREATERESOURCEANDHEAP_UNRECOGNIZEDHEAPTYPE	= 635,
        D3D12_MESSAGE_ID_CREATERESOURCEANDHEAP_UNRECOGNIZEDCPUPAGEPROPERTIES	= 636,
        D3D12_MESSAGE_ID_CREATERESOURCEANDHEAP_UNRECOGNIZEDMEMORYPOOL	= 637,
        D3D12_MESSAGE_ID_CREATERESOURCEANDHEAP_INVALIDHEAPPROPERTIES	= 638,
        D3D12_MESSAGE_ID_CREATERESOURCEANDHEAP_UNRECOGNIZEDHEAPMISCFLAGS	= 639,
        D3D12_MESSAGE_ID_CREATERESOURCEANDHEAP_INVALIDHEAPMISCFLAGS	= 640,
        D3D12_MESSAGE_ID_CREATERESOURCEANDHEAP_INVALIDARG_RETURN	= 641,
        D3D12_MESSAGE_ID_CREATERESOURCEANDHEAP_OUTOFMEMORY_RETURN	= 642,
        D3D12_MESSAGE_ID_GETCUSTOMHEAPPROPERTIES_UNRECOGNIZEDHEAPTYPE	= 643,
        D3D12_MESSAGE_ID_GETCUSTOMHEAPPROPERTIES_INVALIDHEAPTYPE	= 644,
        D3D12_MESSAGE_ID_CREATE_DESCRIPTOR_HEAP_INVALID_DESC	= 645,
        D3D12_MESSAGE_ID_INVALID_DESCRIPTOR_HANDLE	= 646,
        D3D12_MESSAGE_ID_CREATERASTERIZERSTATE_INVALID_CONSERVATIVERASTERMODE	= 647,
        D3D12_MESSAGE_ID_CREATE_CONSTANT_BUFFER_VIEW_INVALID_RESOURCE	= 649,
        D3D12_MESSAGE_ID_CREATE_CONSTANT_BUFFER_VIEW_INVALID_DESC	= 650,
        D3D12_MESSAGE_ID_CREATE_UNORDEREDACCESS_VIEW_INVALID_COUNTER_USAGE	= 652,
        D3D12_MESSAGE_ID_COPY_DESCRIPTORS_INVALID_RANGES	= 653,
        D3D12_MESSAGE_ID_COPY_DESCRIPTORS_WRITE_ONLY_DESCRIPTOR	= 654,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_RTV_FORMAT_NOT_UNKNOWN	= 655,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_INVALID_RENDER_TARGET_COUNT	= 656,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_VERTEX_SHADER_NOT_SET	= 657,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_INPUTLAYOUT_NOT_SET	= 658,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_SHADER_LINKAGE_HS_DS_SIGNATURE_MISMATCH	= 659,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_SHADER_LINKAGE_REGISTERINDEX	= 660,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_SHADER_LINKAGE_COMPONENTTYPE	= 661,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_SHADER_LINKAGE_REGISTERMASK	= 662,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_SHADER_LINKAGE_SYSTEMVALUE	= 663,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_SHADER_LINKAGE_NEVERWRITTEN_ALWAYSREADS	= 664,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_SHADER_LINKAGE_MINPRECISION	= 665,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_SHADER_LINKAGE_SEMANTICNAME_NOT_FOUND	= 666,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_HS_XOR_DS_MISMATCH	= 667,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_HULL_SHADER_INPUT_TOPOLOGY_MISMATCH	= 668,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_HS_DS_CONTROL_POINT_COUNT_MISMATCH	= 669,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_HS_DS_TESSELLATOR_DOMAIN_MISMATCH	= 670,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_INVALID_USE_OF_CENTER_MULTISAMPLE_PATTERN	= 671,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_INVALID_USE_OF_FORCED_SAMPLE_COUNT	= 672,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_INVALID_PRIMITIVETOPOLOGY	= 673,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_INVALID_SYSTEMVALUE	= 674,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_OM_DUAL_SOURCE_BLENDING_CAN_ONLY_HAVE_RENDER_TARGET_0	= 675,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_OM_RENDER_TARGET_DOES_NOT_SUPPORT_BLENDING	= 676,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_PS_OUTPUT_TYPE_MISMATCH	= 677,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_OM_RENDER_TARGET_DOES_NOT_SUPPORT_LOGIC_OPS	= 678,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_RENDERTARGETVIEW_NOT_SET	= 679,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_DEPTHSTENCILVIEW_NOT_SET	= 680,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_GS_INPUT_PRIMITIVE_MISMATCH	= 681,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_POSITION_NOT_PRESENT	= 682,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_MISSING_ROOT_SIGNATURE_FLAGS	= 683,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_INVALID_INDEX_BUFFER_PROPERTIES	= 684,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_INVALID_SAMPLE_DESC	= 685,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_HS_ROOT_SIGNATURE_MISMATCH	= 686,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_DS_ROOT_SIGNATURE_MISMATCH	= 687,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_VS_ROOT_SIGNATURE_MISMATCH	= 688,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_GS_ROOT_SIGNATURE_MISMATCH	= 689,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_PS_ROOT_SIGNATURE_MISMATCH	= 690,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_MISSING_ROOT_SIGNATURE	= 691,
        D3D12_MESSAGE_ID_EXECUTE_BUNDLE_OPEN_BUNDLE	= 692,
        D3D12_MESSAGE_ID_EXECUTE_BUNDLE_DESCRIPTOR_HEAP_MISMATCH	= 693,
        D3D12_MESSAGE_ID_EXECUTE_BUNDLE_TYPE	= 694,
        D3D12_MESSAGE_ID_DRAW_EMPTY_SCISSOR_RECTANGLE	= 695,
        D3D12_MESSAGE_ID_CREATE_ROOT_SIGNATURE_BLOB_NOT_FOUND	= 696,
        D3D12_MESSAGE_ID_CREATE_ROOT_SIGNATURE_DESERIALIZE_FAILED	= 697,
        D3D12_MESSAGE_ID_CREATE_ROOT_SIGNATURE_INVALID_CONFIGURATION	= 698,
        D3D12_MESSAGE_ID_CREATE_ROOT_SIGNATURE_NOT_SUPPORTED_ON_DEVICE	= 699,
        D3D12_MESSAGE_ID_CREATERESOURCEANDHEAP_NULLRESOURCEPROPERTIES	= 700,
        D3D12_MESSAGE_ID_CREATERESOURCEANDHEAP_NULLHEAP	= 701,
        D3D12_MESSAGE_ID_GETRESOURCEALLOCATIONINFO_INVALIDRDESCS	= 702,
        D3D12_MESSAGE_ID_MAKERESIDENT_NULLOBJECTARRAY	= 703,
        D3D12_MESSAGE_ID_EVICT_NULLOBJECTARRAY	= 705,
        D3D12_MESSAGE_ID_SET_DESCRIPTOR_TABLE_INVALID	= 708,
        D3D12_MESSAGE_ID_SET_ROOT_CONSTANT_INVALID	= 709,
        D3D12_MESSAGE_ID_SET_ROOT_CONSTANT_BUFFER_VIEW_INVALID	= 710,
        D3D12_MESSAGE_ID_SET_ROOT_SHADER_RESOURCE_VIEW_INVALID	= 711,
        D3D12_MESSAGE_ID_SET_ROOT_UNORDERED_ACCESS_VIEW_INVALID	= 712,
        D3D12_MESSAGE_ID_SET_VERTEX_BUFFERS_INVALID_DESC	= 713,
        D3D12_MESSAGE_ID_SET_INDEX_BUFFER_INVALID_DESC	= 715,
        D3D12_MESSAGE_ID_SET_STREAM_OUTPUT_BUFFERS_INVALID_DESC	= 717,
        D3D12_MESSAGE_ID_CREATERESOURCE_UNRECOGNIZEDDIMENSIONALITY	= 718,
        D3D12_MESSAGE_ID_CREATERESOURCE_UNRECOGNIZEDLAYOUT	= 719,
        D3D12_MESSAGE_ID_CREATERESOURCE_INVALIDDIMENSIONALITY	= 720,
        D3D12_MESSAGE_ID_CREATERESOURCE_INVALIDALIGNMENT	= 721,
        D3D12_MESSAGE_ID_CREATERESOURCE_INVALIDMIPLEVELS	= 722,
        D3D12_MESSAGE_ID_CREATERESOURCE_INVALIDSAMPLEDESC	= 723,
        D3D12_MESSAGE_ID_CREATERESOURCE_INVALIDLAYOUT	= 724,
        D3D12_MESSAGE_ID_SET_INDEX_BUFFER_INVALID	= 725,
        D3D12_MESSAGE_ID_SET_VERTEX_BUFFERS_INVALID	= 726,
        D3D12_MESSAGE_ID_SET_STREAM_OUTPUT_BUFFERS_INVALID	= 727,
        D3D12_MESSAGE_ID_SET_RENDER_TARGETS_INVALID	= 728,
        D3D12_MESSAGE_ID_CREATEQUERY_HEAP_INVALID_PARAMETERS	= 729,
        D3D12_MESSAGE_ID_BEGIN_END_QUERY_INVALID_PARAMETERS	= 731,
        D3D12_MESSAGE_ID_CLOSE_COMMAND_LIST_OPEN_QUERY	= 732,
        D3D12_MESSAGE_ID_RESOLVE_QUERY_DATA_INVALID_PARAMETERS	= 733,
        D3D12_MESSAGE_ID_SET_PREDICATION_INVALID_PARAMETERS	= 734,
        D3D12_MESSAGE_ID_TIMESTAMPS_NOT_SUPPORTED	= 735,
        D3D12_MESSAGE_ID_CREATERESOURCE_UNRECOGNIZEDFORMAT	= 737,
        D3D12_MESSAGE_ID_CREATERESOURCE_INVALIDFORMAT	= 738,
        D3D12_MESSAGE_ID_GETCOPYABLEFOOTPRINTS_INVALIDSUBRESOURCERANGE	= 739,
        D3D12_MESSAGE_ID_GETCOPYABLEFOOTPRINTS_INVALIDBASEOFFSET	= 740,
        D3D12_MESSAGE_ID_GETCOPYABLELAYOUT_INVALIDSUBRESOURCERANGE	= 739,
        D3D12_MESSAGE_ID_GETCOPYABLELAYOUT_INVALIDBASEOFFSET	= 740,
        D3D12_MESSAGE_ID_RESOURCE_BARRIER_INVALID_HEAP	= 741,
        D3D12_MESSAGE_ID_CREATE_SAMPLER_INVALID	= 742,
        D3D12_MESSAGE_ID_CREATECOMMANDSIGNATURE_INVALID	= 743,
        D3D12_MESSAGE_ID_EXECUTE_INDIRECT_INVALID_PARAMETERS	= 744,
        D3D12_MESSAGE_ID_GETGPUVIRTUALADDRESS_INVALID_RESOURCE_DIMENSION	= 745,
        D3D12_MESSAGE_ID_CREATERESOURCE_INVALIDCLEARVALUE	= 815,
        D3D12_MESSAGE_ID_CREATERESOURCE_UNRECOGNIZEDCLEARVALUEFORMAT	= 816,
        D3D12_MESSAGE_ID_CREATERESOURCE_INVALIDCLEARVALUEFORMAT	= 817,
        D3D12_MESSAGE_ID_CREATERESOURCE_CLEARVALUEDENORMFLUSH	= 818,
        D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE	= 820,
        D3D12_MESSAGE_ID_CLEARDEPTHSTENCILVIEW_MISMATCHINGCLEARVALUE	= 821,
        D3D12_MESSAGE_ID_MAP_INVALIDHEAP	= 822,
        D3D12_MESSAGE_ID_UNMAP_INVALIDHEAP	= 823,
        D3D12_MESSAGE_ID_MAP_INVALIDRESOURCE	= 824,
        D3D12_MESSAGE_ID_UNMAP_INVALIDRESOURCE	= 825,
        D3D12_MESSAGE_ID_MAP_INVALIDSUBRESOURCE	= 826,
        D3D12_MESSAGE_ID_UNMAP_INVALIDSUBRESOURCE	= 827,
        D3D12_MESSAGE_ID_MAP_INVALIDRANGE	= 828,
        D3D12_MESSAGE_ID_UNMAP_INVALIDRANGE	= 829,
        D3D12_MESSAGE_ID_MAP_INVALIDDATAPOINTER	= 832,
        D3D12_MESSAGE_ID_MAP_INVALIDARG_RETURN	= 833,
        D3D12_MESSAGE_ID_MAP_OUTOFMEMORY_RETURN	= 834,
        D3D12_MESSAGE_ID_EXECUTECOMMANDLISTS_BUNDLENOTSUPPORTED	= 835,
        D3D12_MESSAGE_ID_EXECUTECOMMANDLISTS_COMMANDLISTMISMATCH	= 836,
        D3D12_MESSAGE_ID_EXECUTECOMMANDLISTS_OPENCOMMANDLIST	= 837,
        D3D12_MESSAGE_ID_EXECUTECOMMANDLISTS_FAILEDCOMMANDLIST	= 838,
        D3D12_MESSAGE_ID_COPYBUFFERREGION_NULLDST	= 839,
        D3D12_MESSAGE_ID_COPYBUFFERREGION_INVALIDDSTRESOURCEDIMENSION	= 840,
        D3D12_MESSAGE_ID_COPYBUFFERREGION_DSTRANGEOUTOFBOUNDS	= 841,
        D3D12_MESSAGE_ID_COPYBUFFERREGION_NULLSRC	= 842,
        D3D12_MESSAGE_ID_COPYBUFFERREGION_INVALIDSRCRESOURCEDIMENSION	= 843,
        D3D12_MESSAGE_ID_COPYBUFFERREGION_SRCRANGEOUTOFBOUNDS	= 844,
        D3D12_MESSAGE_ID_COPYBUFFERREGION_INVALIDCOPYFLAGS	= 845,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_NULLDST	= 846,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_UNRECOGNIZEDDSTTYPE	= 847,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_INVALIDDSTRESOURCEDIMENSION	= 848,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_INVALIDDSTRESOURCE	= 849,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_INVALIDDSTSUBRESOURCE	= 850,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_INVALIDDSTOFFSET	= 851,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_UNRECOGNIZEDDSTFORMAT	= 852,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_INVALIDDSTFORMAT	= 853,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_INVALIDDSTDIMENSIONS	= 854,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_INVALIDDSTROWPITCH	= 855,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_INVALIDDSTPLACEMENT	= 856,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_INVALIDDSTDSPLACEDFOOTPRINTFORMAT	= 857,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_DSTREGIONOUTOFBOUNDS	= 858,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_NULLSRC	= 859,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_UNRECOGNIZEDSRCTYPE	= 860,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_INVALIDSRCRESOURCEDIMENSION	= 861,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_INVALIDSRCRESOURCE	= 862,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_INVALIDSRCSUBRESOURCE	= 863,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_INVALIDSRCOFFSET	= 864,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_UNRECOGNIZEDSRCFORMAT	= 865,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_INVALIDSRCFORMAT	= 866,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_INVALIDSRCDIMENSIONS	= 867,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_INVALIDSRCROWPITCH	= 868,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_INVALIDSRCPLACEMENT	= 869,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_INVALIDSRCDSPLACEDFOOTPRINTFORMAT	= 870,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_SRCREGIONOUTOFBOUNDS	= 871,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_INVALIDDSTCOORDINATES	= 872,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_INVALIDSRCBOX	= 873,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_FORMATMISMATCH	= 874,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_EMPTYBOX	= 875,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_INVALIDCOPYFLAGS	= 876,
        D3D12_MESSAGE_ID_RESOLVESUBRESOURCE_INVALID_SUBRESOURCE_INDEX	= 877,
        D3D12_MESSAGE_ID_RESOLVESUBRESOURCE_INVALID_FORMAT	= 878,
        D3D12_MESSAGE_ID_RESOLVESUBRESOURCE_RESOURCE_MISMATCH	= 879,
        D3D12_MESSAGE_ID_RESOLVESUBRESOURCE_INVALID_SAMPLE_COUNT	= 880,
        D3D12_MESSAGE_ID_CREATECOMPUTEPIPELINESTATE_INVALID_SHADER	= 881,
        D3D12_MESSAGE_ID_CREATECOMPUTEPIPELINESTATE_CS_ROOT_SIGNATURE_MISMATCH	= 882,
        D3D12_MESSAGE_ID_CREATECOMPUTEPIPELINESTATE_MISSING_ROOT_SIGNATURE	= 883,
        D3D12_MESSAGE_ID_CREATEPIPELINESTATE_INVALIDCACHEDBLOB	= 884,
        D3D12_MESSAGE_ID_CREATEPIPELINESTATE_CACHEDBLOBADAPTERMISMATCH	= 885,
        D3D12_MESSAGE_ID_CREATEPIPELINESTATE_CACHEDBLOBDRIVERVERSIONMISMATCH	= 886,
        D3D12_MESSAGE_ID_CREATEPIPELINESTATE_CACHEDBLOBDESCMISMATCH	= 887,
        D3D12_MESSAGE_ID_CREATEPIPELINESTATE_CACHEDBLOBIGNORED	= 888,
        D3D12_MESSAGE_ID_WRITETOSUBRESOURCE_INVALIDHEAP	= 889,
        D3D12_MESSAGE_ID_WRITETOSUBRESOURCE_INVALIDRESOURCE	= 890,
        D3D12_MESSAGE_ID_WRITETOSUBRESOURCE_INVALIDBOX	= 891,
        D3D12_MESSAGE_ID_WRITETOSUBRESOURCE_INVALIDSUBRESOURCE	= 892,
        D3D12_MESSAGE_ID_WRITETOSUBRESOURCE_EMPTYBOX	= 893,
        D3D12_MESSAGE_ID_READFROMSUBRESOURCE_INVALIDHEAP	= 894,
        D3D12_MESSAGE_ID_READFROMSUBRESOURCE_INVALIDRESOURCE	= 895,
        D3D12_MESSAGE_ID_READFROMSUBRESOURCE_INVALIDBOX	= 896,
        D3D12_MESSAGE_ID_READFROMSUBRESOURCE_INVALIDSUBRESOURCE	= 897,
        D3D12_MESSAGE_ID_READFROMSUBRESOURCE_EMPTYBOX	= 898,
        D3D12_MESSAGE_ID_TOO_MANY_NODES_SPECIFIED	= 899,
        D3D12_MESSAGE_ID_INVALID_NODE_INDEX	= 900,
        D3D12_MESSAGE_ID_GETHEAPPROPERTIES_INVALIDRESOURCE	= 901,
        D3D12_MESSAGE_ID_NODE_MASK_MISMATCH	= 902,
        D3D12_MESSAGE_ID_COMMAND_LIST_OUTOFMEMORY	= 903,
        D3D12_MESSAGE_ID_COMMAND_LIST_MULTIPLE_SWAPCHAIN_BUFFER_REFERENCES	= 904,
        D3D12_MESSAGE_ID_COMMAND_LIST_TOO_MANY_SWAPCHAIN_REFERENCES	= 905,
        D3D12_MESSAGE_ID_COMMAND_QUEUE_TOO_MANY_SWAPCHAIN_REFERENCES	= 906,
        D3D12_MESSAGE_ID_EXECUTECOMMANDLISTS_WRONGSWAPCHAINBUFFERREFERENCE	= 907,
        D3D12_MESSAGE_ID_COMMAND_LIST_SETRENDERTARGETS_INVALIDNUMRENDERTARGETS	= 908,
        D3D12_MESSAGE_ID_CREATE_QUEUE_INVALID_TYPE	= 909,
        D3D12_MESSAGE_ID_CREATE_QUEUE_INVALID_FLAGS	= 910,
        D3D12_MESSAGE_ID_CREATESHAREDRESOURCE_INVALIDFLAGS	= 911,
        D3D12_MESSAGE_ID_CREATESHAREDRESOURCE_INVALIDFORMAT	= 912,
        D3D12_MESSAGE_ID_CREATESHAREDHEAP_INVALIDFLAGS	= 913,
        D3D12_MESSAGE_ID_REFLECTSHAREDPROPERTIES_UNRECOGNIZEDPROPERTIES	= 914,
        D3D12_MESSAGE_ID_REFLECTSHAREDPROPERTIES_INVALIDSIZE	= 915,
        D3D12_MESSAGE_ID_REFLECTSHAREDPROPERTIES_INVALIDOBJECT	= 916,
        D3D12_MESSAGE_ID_KEYEDMUTEX_INVALIDOBJECT	= 917,
        D3D12_MESSAGE_ID_KEYEDMUTEX_INVALIDKEY	= 918,
        D3D12_MESSAGE_ID_KEYEDMUTEX_WRONGSTATE	= 919,
        D3D12_MESSAGE_ID_CREATE_QUEUE_INVALID_PRIORITY	= 920,
        D3D12_MESSAGE_ID_OBJECT_DELETED_WHILE_STILL_IN_USE	= 921,
        D3D12_MESSAGE_ID_CREATEPIPELINESTATE_INVALID_FLAGS	= 922,
        D3D12_MESSAGE_ID_HEAP_ADDRESS_RANGE_HAS_NO_RESOURCE	= 923,
        D3D12_MESSAGE_ID_COMMAND_LIST_DRAW_RENDER_TARGET_DELETED	= 924,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_ALL_RENDER_TARGETS_HAVE_UNKNOWN_FORMAT	= 925,
        D3D12_MESSAGE_ID_HEAP_ADDRESS_RANGE_INTERSECTS_MULTIPLE_BUFFERS	= 926,
        D3D12_MESSAGE_ID_EXECUTECOMMANDLISTS_GPU_WRITTEN_READBACK_RESOURCE_MAPPED	= 927,
        D3D12_MESSAGE_ID_UNMAP_RANGE_NOT_EMPTY	= 929,
        D3D12_MESSAGE_ID_MAP_INVALID_NULLRANGE	= 930,
        D3D12_MESSAGE_ID_UNMAP_INVALID_NULLRANGE	= 931,
        D3D12_MESSAGE_ID_NO_GRAPHICS_API_SUPPORT	= 932,
        D3D12_MESSAGE_ID_NO_COMPUTE_API_SUPPORT	= 933,
        D3D12_MESSAGE_ID_RESOLVESUBRESOURCE_RESOURCE_FLAGS_NOT_SUPPORTED	= 934,
        D3D12_MESSAGE_ID_GPU_BASED_VALIDATION_ROOT_ARGUMENT_UNINITIALIZED	= 935,
        D3D12_MESSAGE_ID_GPU_BASED_VALIDATION_DESCRIPTOR_HEAP_INDEX_OUT_OF_BOUNDS	= 936,
        D3D12_MESSAGE_ID_GPU_BASED_VALIDATION_DESCRIPTOR_TABLE_REGISTER_INDEX_OUT_OF_BOUNDS	= 937,
        D3D12_MESSAGE_ID_GPU_BASED_VALIDATION_DESCRIPTOR_UNINITIALIZED	= 938,
        D3D12_MESSAGE_ID_GPU_BASED_VALIDATION_DESCRIPTOR_TYPE_MISMATCH	= 939,
        D3D12_MESSAGE_ID_GPU_BASED_VALIDATION_SRV_RESOURCE_DIMENSION_MISMATCH	= 940,
        D3D12_MESSAGE_ID_GPU_BASED_VALIDATION_UAV_RESOURCE_DIMENSION_MISMATCH	= 941,
        D3D12_MESSAGE_ID_GPU_BASED_VALIDATION_INCOMPATIBLE_RESOURCE_STATE	= 942,
        D3D12_MESSAGE_ID_COPYRESOURCE_NULLDST	= 943,
        D3D12_MESSAGE_ID_COPYRESOURCE_INVALIDDSTRESOURCE	= 944,
        D3D12_MESSAGE_ID_COPYRESOURCE_NULLSRC	= 945,
        D3D12_MESSAGE_ID_COPYRESOURCE_INVALIDSRCRESOURCE	= 946,
        D3D12_MESSAGE_ID_RESOLVESUBRESOURCE_NULLDST	= 947,
        D3D12_MESSAGE_ID_RESOLVESUBRESOURCE_INVALIDDSTRESOURCE	= 948,
        D3D12_MESSAGE_ID_RESOLVESUBRESOURCE_NULLSRC	= 949,
        D3D12_MESSAGE_ID_RESOLVESUBRESOURCE_INVALIDSRCRESOURCE	= 950,
        D3D12_MESSAGE_ID_PIPELINE_STATE_TYPE_MISMATCH	= 951,
        D3D12_MESSAGE_ID_COMMAND_LIST_DISPATCH_ROOT_SIGNATURE_NOT_SET	= 952,
        D3D12_MESSAGE_ID_COMMAND_LIST_DISPATCH_ROOT_SIGNATURE_MISMATCH	= 953,
        D3D12_MESSAGE_ID_RESOURCE_BARRIER_ZERO_BARRIERS	= 954,
        D3D12_MESSAGE_ID_BEGIN_END_EVENT_MISMATCH	= 955,
        D3D12_MESSAGE_ID_RESOURCE_BARRIER_POSSIBLE_BEFORE_AFTER_MISMATCH	= 956,
        D3D12_MESSAGE_ID_RESOURCE_BARRIER_MISMATCHING_BEGIN_END	= 957,
        D3D12_MESSAGE_ID_GPU_BASED_VALIDATION_INVALID_RESOURCE	= 958,
        D3D12_MESSAGE_ID_USE_OF_ZERO_REFCOUNT_OBJECT	= 959,
        D3D12_MESSAGE_ID_OBJECT_EVICTED_WHILE_STILL_IN_USE	= 960,
        D3D12_MESSAGE_ID_GPU_BASED_VALIDATION_ROOT_DESCRIPTOR_ACCESS_OUT_OF_BOUNDS	= 961,
        D3D12_MESSAGE_ID_CREATEPIPELINELIBRARY_INVALIDLIBRARYBLOB	= 962,
        D3D12_MESSAGE_ID_CREATEPIPELINELIBRARY_DRIVERVERSIONMISMATCH	= 963,
        D3D12_MESSAGE_ID_CREATEPIPELINELIBRARY_ADAPTERVERSIONMISMATCH	= 964,
        D3D12_MESSAGE_ID_CREATEPIPELINELIBRARY_UNSUPPORTED	= 965,
        D3D12_MESSAGE_ID_CREATE_PIPELINELIBRARY	= 966,
        D3D12_MESSAGE_ID_LIVE_PIPELINELIBRARY	= 967,
        D3D12_MESSAGE_ID_DESTROY_PIPELINELIBRARY	= 968,
        D3D12_MESSAGE_ID_STOREPIPELINE_NONAME	= 969,
        D3D12_MESSAGE_ID_STOREPIPELINE_DUPLICATENAME	= 970,
        D3D12_MESSAGE_ID_LOADPIPELINE_NAMENOTFOUND	= 971,
        D3D12_MESSAGE_ID_LOADPIPELINE_INVALIDDESC	= 972,
        D3D12_MESSAGE_ID_PIPELINELIBRARY_SERIALIZE_NOTENOUGHMEMORY	= 973,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_PS_OUTPUT_RT_OUTPUT_MISMATCH	= 974,
        D3D12_MESSAGE_ID_SETEVENTONMULTIPLEFENCECOMPLETION_INVALIDFLAGS	= 975,
        D3D12_MESSAGE_ID_CREATE_QUEUE_VIDEO_NOT_SUPPORTED	= 976,
        D3D12_MESSAGE_ID_CREATE_COMMAND_ALLOCATOR_VIDEO_NOT_SUPPORTED	= 977,
        D3D12_MESSAGE_ID_CREATEQUERY_HEAP_VIDEO_DECODE_STATISTICS_NOT_SUPPORTED	= 978,
        D3D12_MESSAGE_ID_CREATE_VIDEODECODECOMMANDLIST	= 979,
        D3D12_MESSAGE_ID_CREATE_VIDEODECODER	= 980,
        D3D12_MESSAGE_ID_CREATE_VIDEODECODESTREAM	= 981,
        D3D12_MESSAGE_ID_LIVE_VIDEODECODECOMMANDLIST	= 982,
        D3D12_MESSAGE_ID_LIVE_VIDEODECODER	= 983,
        D3D12_MESSAGE_ID_LIVE_VIDEODECODESTREAM	= 984,
        D3D12_MESSAGE_ID_DESTROY_VIDEODECODECOMMANDLIST	= 985,
        D3D12_MESSAGE_ID_DESTROY_VIDEODECODER	= 986,
        D3D12_MESSAGE_ID_DESTROY_VIDEODECODESTREAM	= 987,
        D3D12_MESSAGE_ID_DECODE_FRAME_INVALID_PARAMETERS	= 988,
        D3D12_MESSAGE_ID_DEPRECATED_API	= 989,
        D3D12_MESSAGE_ID_RESOURCE_BARRIER_MISMATCHING_COMMAND_LIST_TYPE	= 990,
        D3D12_MESSAGE_ID_COMMAND_LIST_DESCRIPTOR_TABLE_NOT_SET	= 991,
        D3D12_MESSAGE_ID_COMMAND_LIST_ROOT_CONSTANT_BUFFER_VIEW_NOT_SET	= 992,
        D3D12_MESSAGE_ID_COMMAND_LIST_ROOT_SHADER_RESOURCE_VIEW_NOT_SET	= 993,
        D3D12_MESSAGE_ID_COMMAND_LIST_ROOT_UNORDERED_ACCESS_VIEW_NOT_SET	= 994,
        D3D12_MESSAGE_ID_DISCARD_INVALID_SUBRESOURCE_RANGE	= 995,
        D3D12_MESSAGE_ID_DISCARD_ONE_SUBRESOURCE_FOR_MIPS_WITH_RECTS	= 996,
        D3D12_MESSAGE_ID_DISCARD_NO_RECTS_FOR_NON_TEXTURE2D	= 997,
        D3D12_MESSAGE_ID_COPY_ON_SAME_SUBRESOURCE	= 998,
        D3D12_MESSAGE_ID_SETRESIDENCYPRIORITY_INVALID_PAGEABLE	= 999,
        D3D12_MESSAGE_ID_GPU_BASED_VALIDATION_UNSUPPORTED	= 1000,
        D3D12_MESSAGE_ID_STATIC_DESCRIPTOR_INVALID_DESCRIPTOR_CHANGE	= 1001,
        D3D12_MESSAGE_ID_DATA_STATIC_DESCRIPTOR_INVALID_DATA_CHANGE	= 1002,
        D3D12_MESSAGE_ID_DATA_STATIC_WHILE_SET_AT_EXECUTE_DESCRIPTOR_INVALID_DATA_CHANGE	= 1003,
        D3D12_MESSAGE_ID_EXECUTE_BUNDLE_STATIC_DESCRIPTOR_DATA_STATIC_NOT_SET	= 1004,
        D3D12_MESSAGE_ID_GPU_BASED_VALIDATION_RESOURCE_ACCESS_OUT_OF_BOUNDS	= 1005,
        D3D12_MESSAGE_ID_GPU_BASED_VALIDATION_SAMPLER_MODE_MISMATCH	= 1006,
        D3D12_MESSAGE_ID_CREATE_FENCE_INVALID_FLAGS	= 1007,
        D3D12_MESSAGE_ID_RESOURCE_BARRIER_DUPLICATE_SUBRESOURCE_TRANSITIONS	= 1008,
        D3D12_MESSAGE_ID_SETRESIDENCYPRIORITY_INVALID_PRIORITY	= 1009,
        D3D12_MESSAGE_ID_CREATE_DESCRIPTOR_HEAP_LARGE_NUM_DESCRIPTORS	= 1013,
        D3D12_MESSAGE_ID_BEGIN_EVENT	= 1014,
        D3D12_MESSAGE_ID_END_EVENT	= 1015,
        D3D12_MESSAGE_ID_CREATEDEVICE_DEBUG_LAYER_STARTUP_OPTIONS	= 1016,
        D3D12_MESSAGE_ID_CREATEDEPTHSTENCILSTATE_DEPTHBOUNDSTEST_UNSUPPORTED	= 1017,
        D3D12_MESSAGE_ID_CREATEPIPELINESTATE_DUPLICATE_SUBOBJECT	= 1018,
        D3D12_MESSAGE_ID_CREATEPIPELINESTATE_UNKNOWN_SUBOBJECT	= 1019,
        D3D12_MESSAGE_ID_CREATEPIPELINESTATE_ZERO_SIZE_STREAM	= 1020,
        D3D12_MESSAGE_ID_CREATEPIPELINESTATE_INVALID_STREAM	= 1021,
        D3D12_MESSAGE_ID_CREATEPIPELINESTATE_CANNOT_DEDUCE_TYPE	= 1022,
        D3D12_MESSAGE_ID_COMMAND_LIST_STATIC_DESCRIPTOR_RESOURCE_DIMENSION_MISMATCH	= 1023,
        D3D12_MESSAGE_ID_CREATE_COMMAND_QUEUE_INSUFFICIENT_PRIVILEGE_FOR_GLOBAL_REALTIME	= 1024,
        D3D12_MESSAGE_ID_CREATE_COMMAND_QUEUE_INSUFFICIENT_HARDWARE_SUPPORT_FOR_GLOBAL_REALTIME	= 1025,
        D3D12_MESSAGE_ID_ATOMICCOPYBUFFER_INVALID_ARCHITECTURE	= 1026,
        D3D12_MESSAGE_ID_ATOMICCOPYBUFFER_NULL_DST	= 1027,
        D3D12_MESSAGE_ID_ATOMICCOPYBUFFER_INVALID_DST_RESOURCE_DIMENSION	= 1028,
        D3D12_MESSAGE_ID_ATOMICCOPYBUFFER_DST_RANGE_OUT_OF_BOUNDS	= 1029,
        D3D12_MESSAGE_ID_ATOMICCOPYBUFFER_NULL_SRC	= 1030,
        D3D12_MESSAGE_ID_ATOMICCOPYBUFFER_INVALID_SRC_RESOURCE_DIMENSION	= 1031,
        D3D12_MESSAGE_ID_ATOMICCOPYBUFFER_SRC_RANGE_OUT_OF_BOUNDS	= 1032,
        D3D12_MESSAGE_ID_ATOMICCOPYBUFFER_INVALID_OFFSET_ALIGNMENT	= 1033,
        D3D12_MESSAGE_ID_ATOMICCOPYBUFFER_NULL_DEPENDENT_RESOURCES	= 1034,
        D3D12_MESSAGE_ID_ATOMICCOPYBUFFER_NULL_DEPENDENT_SUBRESOURCE_RANGES	= 1035,
        D3D12_MESSAGE_ID_ATOMICCOPYBUFFER_INVALID_DEPENDENT_RESOURCE	= 1036,
        D3D12_MESSAGE_ID_ATOMICCOPYBUFFER_INVALID_DEPENDENT_SUBRESOURCE_RANGE	= 1037,
        D3D12_MESSAGE_ID_ATOMICCOPYBUFFER_DEPENDENT_SUBRESOURCE_OUT_OF_BOUNDS	= 1038,
        D3D12_MESSAGE_ID_ATOMICCOPYBUFFER_DEPENDENT_RANGE_OUT_OF_BOUNDS	= 1039,
        D3D12_MESSAGE_ID_ATOMICCOPYBUFFER_ZERO_DEPENDENCIES	= 1040,
        D3D12_MESSAGE_ID_DEVICE_CREATE_SHARED_HANDLE_INVALIDARG	= 1041,
        D3D12_MESSAGE_ID_DESCRIPTOR_HANDLE_WITH_INVALID_RESOURCE	= 1042,
        D3D12_MESSAGE_ID_SETDEPTHBOUNDS_INVALIDARGS	= 1043,
        D3D12_MESSAGE_ID_GPU_BASED_VALIDATION_RESOURCE_STATE_IMPRECISE	= 1044,
        D3D12_MESSAGE_ID_COMMAND_LIST_PIPELINE_STATE_NOT_SET	= 1045,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_SHADER_MODEL_MISMATCH	= 1046,
        D3D12_MESSAGE_ID_OBJECT_ACCESSED_WHILE_STILL_IN_USE	= 1047,
        D3D12_MESSAGE_ID_PROGRAMMABLE_MSAA_UNSUPPORTED	= 1048,
        D3D12_MESSAGE_ID_SETSAMPLEPOSITIONS_INVALIDARGS	= 1049,
        D3D12_MESSAGE_ID_RESOLVESUBRESOURCEREGION_INVALID_RECT	= 1050,
        D3D12_MESSAGE_ID_CREATE_VIDEODECODECOMMANDQUEUE	= 1051,
        D3D12_MESSAGE_ID_CREATE_VIDEOPROCESSCOMMANDLIST	= 1052,
        D3D12_MESSAGE_ID_CREATE_VIDEOPROCESSCOMMANDQUEUE	= 1053,
        D3D12_MESSAGE_ID_LIVE_VIDEODECODECOMMANDQUEUE	= 1054,
        D3D12_MESSAGE_ID_LIVE_VIDEOPROCESSCOMMANDLIST	= 1055,
        D3D12_MESSAGE_ID_LIVE_VIDEOPROCESSCOMMANDQUEUE	= 1056,
        D3D12_MESSAGE_ID_DESTROY_VIDEODECODECOMMANDQUEUE	= 1057,
        D3D12_MESSAGE_ID_DESTROY_VIDEOPROCESSCOMMANDLIST	= 1058,
        D3D12_MESSAGE_ID_DESTROY_VIDEOPROCESSCOMMANDQUEUE	= 1059,
        D3D12_MESSAGE_ID_CREATE_VIDEOPROCESSOR	= 1060,
        D3D12_MESSAGE_ID_CREATE_VIDEOPROCESSSTREAM	= 1061,
        D3D12_MESSAGE_ID_LIVE_VIDEOPROCESSOR	= 1062,
        D3D12_MESSAGE_ID_LIVE_VIDEOPROCESSSTREAM	= 1063,
        D3D12_MESSAGE_ID_DESTROY_VIDEOPROCESSOR	= 1064,
        D3D12_MESSAGE_ID_DESTROY_VIDEOPROCESSSTREAM	= 1065,
        D3D12_MESSAGE_ID_PROCESS_FRAME_INVALID_PARAMETERS	= 1066,
        D3D12_MESSAGE_ID_COPY_INVALIDLAYOUT	= 1067,
        D3D12_MESSAGE_ID_CREATE_CRYPTO_SESSION	= 1068,
        D3D12_MESSAGE_ID_CREATE_CRYPTO_SESSION_POLICY	= 1069,
        D3D12_MESSAGE_ID_CREATE_PROTECTED_RESOURCE_SESSION	= 1070,
        D3D12_MESSAGE_ID_LIVE_CRYPTO_SESSION	= 1071,
        D3D12_MESSAGE_ID_LIVE_CRYPTO_SESSION_POLICY	= 1072,
        D3D12_MESSAGE_ID_LIVE_PROTECTED_RESOURCE_SESSION	= 1073,
        D3D12_MESSAGE_ID_DESTROY_CRYPTO_SESSION	= 1074,
        D3D12_MESSAGE_ID_DESTROY_CRYPTO_SESSION_POLICY	= 1075,
        D3D12_MESSAGE_ID_DESTROY_PROTECTED_RESOURCE_SESSION	= 1076,
        D3D12_MESSAGE_ID_PROTECTED_RESOURCE_SESSION_UNSUPPORTED	= 1077,
        D3D12_MESSAGE_ID_FENCE_INVALIDOPERATION	= 1078,
        D3D12_MESSAGE_ID_CREATEQUERY_HEAP_COPY_QUEUE_TIMESTAMPS_NOT_SUPPORTED	= 1079,
        D3D12_MESSAGE_ID_SAMPLEPOSITIONS_MISMATCH_DEFERRED	= 1080,
        D3D12_MESSAGE_ID_SAMPLEPOSITIONS_MISMATCH_RECORDTIME_ASSUMEDFROMFIRSTUSE	= 1081,
        D3D12_MESSAGE_ID_SAMPLEPOSITIONS_MISMATCH_RECORDTIME_ASSUMEDFROMCLEAR	= 1082,
        D3D12_MESSAGE_ID_CREATE_VIDEODECODERHEAP	= 1083,
        D3D12_MESSAGE_ID_LIVE_VIDEODECODERHEAP	= 1084,
        D3D12_MESSAGE_ID_DESTROY_VIDEODECODERHEAP	= 1085,
        D3D12_MESSAGE_ID_OPENEXISTINGHEAP_INVALIDARG_RETURN	= 1086,
        D3D12_MESSAGE_ID_OPENEXISTINGHEAP_OUTOFMEMORY_RETURN	= 1087,
        D3D12_MESSAGE_ID_OPENEXISTINGHEAP_INVALIDADDRESS	= 1088,
        D3D12_MESSAGE_ID_OPENEXISTINGHEAP_INVALIDHANDLE	= 1089,
        D3D12_MESSAGE_ID_WRITEBUFFERIMMEDIATE_INVALID_DEST	= 1090,
        D3D12_MESSAGE_ID_WRITEBUFFERIMMEDIATE_INVALID_MODE	= 1091,
        D3D12_MESSAGE_ID_WRITEBUFFERIMMEDIATE_INVALID_ALIGNMENT	= 1092,
        D3D12_MESSAGE_ID_WRITEBUFFERIMMEDIATE_NOT_SUPPORTED	= 1093,
        D3D12_MESSAGE_ID_SETVIEWINSTANCEMASK_INVALIDARGS	= 1094,
        D3D12_MESSAGE_ID_VIEW_INSTANCING_UNSUPPORTED	= 1095,
        D3D12_MESSAGE_ID_VIEW_INSTANCING_INVALIDARGS	= 1096,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_MISMATCH_DECODE_REFERENCE_ONLY_FLAG	= 1097,
        D3D12_MESSAGE_ID_COPYRESOURCE_MISMATCH_DECODE_REFERENCE_ONLY_FLAG	= 1098,
        D3D12_MESSAGE_ID_CREATE_VIDEO_DECODE_HEAP_CAPS_FAILURE	= 1099,
        D3D12_MESSAGE_ID_CREATE_VIDEO_DECODE_HEAP_CAPS_UNSUPPORTED	= 1100,
        D3D12_MESSAGE_ID_VIDEO_DECODE_SUPPORT_INVALID_INPUT	= 1101,
        D3D12_MESSAGE_ID_CREATE_VIDEO_DECODER_UNSUPPORTED	= 1102,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_METADATA_ERROR	= 1103,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_VIEW_INSTANCING_VERTEX_SIZE_EXCEEDED	= 1104,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_RUNTIME_INTERNAL_ERROR	= 1105,
        D3D12_MESSAGE_ID_NO_VIDEO_API_SUPPORT	= 1106,
        D3D12_MESSAGE_ID_VIDEO_PROCESS_SUPPORT_INVALID_INPUT	= 1107,
        D3D12_MESSAGE_ID_CREATE_VIDEO_PROCESSOR_CAPS_FAILURE	= 1108,
        D3D12_MESSAGE_ID_VIDEO_PROCESS_SUPPORT_UNSUPPORTED_FORMAT	= 1109,
        D3D12_MESSAGE_ID_VIDEO_DECODE_FRAME_INVALID_ARGUMENT	= 1110,
        D3D12_MESSAGE_ID_ENQUEUE_MAKE_RESIDENT_INVALID_FLAGS	= 1111,
        D3D12_MESSAGE_ID_OPENEXISTINGHEAP_UNSUPPORTED	= 1112,
        D3D12_MESSAGE_ID_VIDEO_PROCESS_FRAMES_INVALID_ARGUMENT	= 1113,
        D3D12_MESSAGE_ID_VIDEO_DECODE_SUPPORT_UNSUPPORTED	= 1114,
        D3D12_MESSAGE_ID_CREATE_COMMANDRECORDER	= 1115,
        D3D12_MESSAGE_ID_LIVE_COMMANDRECORDER	= 1116,
        D3D12_MESSAGE_ID_DESTROY_COMMANDRECORDER	= 1117,
        D3D12_MESSAGE_ID_CREATE_COMMAND_RECORDER_VIDEO_NOT_SUPPORTED	= 1118,
        D3D12_MESSAGE_ID_CREATE_COMMAND_RECORDER_INVALID_SUPPORT_FLAGS	= 1119,
        D3D12_MESSAGE_ID_CREATE_COMMAND_RECORDER_INVALID_FLAGS	= 1120,
        D3D12_MESSAGE_ID_CREATE_COMMAND_RECORDER_MORE_RECORDERS_THAN_LOGICAL_PROCESSORS	= 1121,
        D3D12_MESSAGE_ID_CREATE_COMMANDPOOL	= 1122,
        D3D12_MESSAGE_ID_LIVE_COMMANDPOOL	= 1123,
        D3D12_MESSAGE_ID_DESTROY_COMMANDPOOL	= 1124,
        D3D12_MESSAGE_ID_CREATE_COMMAND_POOL_INVALID_FLAGS	= 1125,
        D3D12_MESSAGE_ID_CREATE_COMMAND_LIST_VIDEO_NOT_SUPPORTED	= 1126,
        D3D12_MESSAGE_ID_COMMAND_RECORDER_SUPPORT_FLAGS_MISMATCH	= 1127,
        D3D12_MESSAGE_ID_COMMAND_RECORDER_CONTENTION	= 1128,
        D3D12_MESSAGE_ID_COMMAND_RECORDER_USAGE_WITH_CREATECOMMANDLIST_COMMAND_LIST	= 1129,
        D3D12_MESSAGE_ID_COMMAND_ALLOCATOR_USAGE_WITH_CREATECOMMANDLIST1_COMMAND_LIST	= 1130,
        D3D12_MESSAGE_ID_CANNOT_EXECUTE_EMPTY_COMMAND_LIST	= 1131,
        D3D12_MESSAGE_ID_CANNOT_RESET_COMMAND_POOL_WITH_OPEN_COMMAND_LISTS	= 1132,
        D3D12_MESSAGE_ID_CANNOT_USE_COMMAND_RECORDER_WITHOUT_CURRENT_TARGET	= 1133,
        D3D12_MESSAGE_ID_CANNOT_CHANGE_COMMAND_RECORDER_TARGET_WHILE_RECORDING	= 1134,
        D3D12_MESSAGE_ID_COMMAND_POOL_SYNC	= 1135,
        D3D12_MESSAGE_ID_EVICT_UNDERFLOW	= 1136,
        D3D12_MESSAGE_ID_CREATE_META_COMMAND	= 1137,
        D3D12_MESSAGE_ID_LIVE_META_COMMAND	= 1138,
        D3D12_MESSAGE_ID_DESTROY_META_COMMAND	= 1139,
        D3D12_MESSAGE_ID_COPYBUFFERREGION_INVALID_DST_RESOURCE	= 1140,
        D3D12_MESSAGE_ID_COPYBUFFERREGION_INVALID_SRC_RESOURCE	= 1141,
        D3D12_MESSAGE_ID_ATOMICCOPYBUFFER_INVALID_DST_RESOURCE	= 1142,
        D3D12_MESSAGE_ID_ATOMICCOPYBUFFER_INVALID_SRC_RESOURCE	= 1143,
        D3D12_MESSAGE_ID_CREATEPLACEDRESOURCEONBUFFER_NULL_BUFFER	= 1144,
        D3D12_MESSAGE_ID_CREATEPLACEDRESOURCEONBUFFER_NULL_RESOURCE_DESC	= 1145,
        D3D12_MESSAGE_ID_CREATEPLACEDRESOURCEONBUFFER_UNSUPPORTED	= 1146,
        D3D12_MESSAGE_ID_CREATEPLACEDRESOURCEONBUFFER_INVALID_BUFFER_DIMENSION	= 1147,
        D3D12_MESSAGE_ID_CREATEPLACEDRESOURCEONBUFFER_INVALID_BUFFER_FLAGS	= 1148,
        D3D12_MESSAGE_ID_CREATEPLACEDRESOURCEONBUFFER_INVALID_BUFFER_OFFSET	= 1149,
        D3D12_MESSAGE_ID_CREATEPLACEDRESOURCEONBUFFER_INVALID_RESOURCE_DIMENSION	= 1150,
        D3D12_MESSAGE_ID_CREATEPLACEDRESOURCEONBUFFER_INVALID_RESOURCE_FLAGS	= 1151,
        D3D12_MESSAGE_ID_CREATEPLACEDRESOURCEONBUFFER_OUTOFMEMORY_RETURN	= 1152,
        D3D12_MESSAGE_ID_CANNOT_CREATE_GRAPHICS_AND_VIDEO_COMMAND_RECORDER	= 1153,
        D3D12_MESSAGE_ID_UPDATETILEMAPPINGS_POSSIBLY_MISMATCHING_PROPERTIES	= 1154,
        D3D12_MESSAGE_ID_CREATE_COMMAND_LIST_INVALID_COMMAND_LIST_TYPE	= 1155,
        D3D12_MESSAGE_ID_CLEARUNORDEREDACCESSVIEW_INCOMPATIBLE_WITH_STRUCTURED_BUFFERS	= 1156,
        D3D12_MESSAGE_ID_COMPUTE_ONLY_DEVICE_OPERATION_UNSUPPORTED	= 1157,
        D3D12_MESSAGE_ID_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INVALID	= 1158,
        D3D12_MESSAGE_ID_EMIT_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_INVALID	= 1159,
        D3D12_MESSAGE_ID_COPY_RAYTRACING_ACCELERATION_STRUCTURE_INVALID	= 1160,
        D3D12_MESSAGE_ID_DISPATCH_RAYS_INVALID	= 1161,
        D3D12_MESSAGE_ID_GET_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO_INVALID	= 1162,
        D3D12_MESSAGE_ID_CREATE_LIFETIMETRACKER	= 1163,
        D3D12_MESSAGE_ID_LIVE_LIFETIMETRACKER	= 1164,
        D3D12_MESSAGE_ID_DESTROY_LIFETIMETRACKER	= 1165,
        D3D12_MESSAGE_ID_DESTROYOWNEDOBJECT_OBJECTNOTOWNED	= 1166,
        D3D12_MESSAGE_ID_CREATE_TRACKEDWORKLOAD	= 1167,
        D3D12_MESSAGE_ID_LIVE_TRACKEDWORKLOAD	= 1168,
        D3D12_MESSAGE_ID_DESTROY_TRACKEDWORKLOAD	= 1169,
        D3D12_MESSAGE_ID_RENDER_PASS_ERROR	= 1170,
        D3D12_MESSAGE_ID_META_COMMAND_ID_INVALID	= 1171,
        D3D12_MESSAGE_ID_META_COMMAND_UNSUPPORTED_PARAMS	= 1172,
        D3D12_MESSAGE_ID_META_COMMAND_FAILED_ENUMERATION	= 1173,
        D3D12_MESSAGE_ID_META_COMMAND_PARAMETER_SIZE_MISMATCH	= 1174,
        D3D12_MESSAGE_ID_UNINITIALIZED_META_COMMAND	= 1175,
        D3D12_MESSAGE_ID_META_COMMAND_INVALID_GPU_VIRTUAL_ADDRESS	= 1176,
        D3D12_MESSAGE_ID_CREATE_VIDEOENCODECOMMANDLIST	= 1177,
        D3D12_MESSAGE_ID_LIVE_VIDEOENCODECOMMANDLIST	= 1178,
        D3D12_MESSAGE_ID_DESTROY_VIDEOENCODECOMMANDLIST	= 1179,
        D3D12_MESSAGE_ID_CREATE_VIDEOENCODECOMMANDQUEUE	= 1180,
        D3D12_MESSAGE_ID_LIVE_VIDEOENCODECOMMANDQUEUE	= 1181,
        D3D12_MESSAGE_ID_DESTROY_VIDEOENCODECOMMANDQUEUE	= 1182,
        D3D12_MESSAGE_ID_CREATE_VIDEOMOTIONESTIMATOR	= 1183,
        D3D12_MESSAGE_ID_LIVE_VIDEOMOTIONESTIMATOR	= 1184,
        D3D12_MESSAGE_ID_DESTROY_VIDEOMOTIONESTIMATOR	= 1185,
        D3D12_MESSAGE_ID_CREATE_VIDEOMOTIONVECTORHEAP	= 1186,
        D3D12_MESSAGE_ID_LIVE_VIDEOMOTIONVECTORHEAP	= 1187,
        D3D12_MESSAGE_ID_DESTROY_VIDEOMOTIONVECTORHEAP	= 1188,
        D3D12_MESSAGE_ID_MULTIPLE_TRACKED_WORKLOADS	= 1189,
        D3D12_MESSAGE_ID_MULTIPLE_TRACKED_WORKLOAD_PAIRS	= 1190,
        D3D12_MESSAGE_ID_OUT_OF_ORDER_TRACKED_WORKLOAD_PAIR	= 1191,
        D3D12_MESSAGE_ID_CANNOT_ADD_TRACKED_WORKLOAD	= 1192,
        D3D12_MESSAGE_ID_INCOMPLETE_TRACKED_WORKLOAD_PAIR	= 1193,
        D3D12_MESSAGE_ID_CREATE_STATE_OBJECT_ERROR	= 1194,
        D3D12_MESSAGE_ID_GET_SHADER_IDENTIFIER_ERROR	= 1195,
        D3D12_MESSAGE_ID_GET_SHADER_STACK_SIZE_ERROR	= 1196,
        D3D12_MESSAGE_ID_GET_PIPELINE_STACK_SIZE_ERROR	= 1197,
        D3D12_MESSAGE_ID_SET_PIPELINE_STACK_SIZE_ERROR	= 1198,
        D3D12_MESSAGE_ID_GET_SHADER_IDENTIFIER_SIZE_INVALID	= 1199,
        D3D12_MESSAGE_ID_CHECK_DRIVER_MATCHING_IDENTIFIER_INVALID	= 1200,
        D3D12_MESSAGE_ID_CHECK_DRIVER_MATCHING_IDENTIFIER_DRIVER_REPORTED_ISSUE	= 1201,
        D3D12_MESSAGE_ID_RENDER_PASS_INVALID_RESOURCE_BARRIER	= 1202,
        D3D12_MESSAGE_ID_RENDER_PASS_DISALLOWED_API_CALLED	= 1203,
        D3D12_MESSAGE_ID_RENDER_PASS_CANNOT_NEST_RENDER_PASSES	= 1204,
        D3D12_MESSAGE_ID_RENDER_PASS_CANNOT_END_WITHOUT_BEGIN	= 1205,
        D3D12_MESSAGE_ID_RENDER_PASS_CANNOT_CLOSE_COMMAND_LIST	= 1206,
        D3D12_MESSAGE_ID_RENDER_PASS_GPU_WORK_WHILE_SUSPENDED	= 1207,
        D3D12_MESSAGE_ID_RENDER_PASS_MISMATCHING_SUSPEND_RESUME	= 1208,
        D3D12_MESSAGE_ID_RENDER_PASS_NO_PRIOR_SUSPEND_WITHIN_EXECUTECOMMANDLISTS	= 1209,
        D3D12_MESSAGE_ID_RENDER_PASS_NO_SUBSEQUENT_RESUME_WITHIN_EXECUTECOMMANDLISTS	= 1210,
        D3D12_MESSAGE_ID_TRACKED_WORKLOAD_COMMAND_QUEUE_MISMATCH	= 1211,
        D3D12_MESSAGE_ID_TRACKED_WORKLOAD_NOT_SUPPORTED	= 1212,
        D3D12_MESSAGE_ID_RENDER_PASS_MISMATCHING_NO_ACCESS	= 1213,
        D3D12_MESSAGE_ID_RENDER_PASS_UNSUPPORTED_RESOLVE	= 1214,
        D3D12_MESSAGE_ID_CLEARUNORDEREDACCESSVIEW_INVALID_RESOURCE_PTR	= 1215,
        D3D12_MESSAGE_ID_WINDOWS7_FENCE_OUTOFORDER_SIGNAL	= 1216,
        D3D12_MESSAGE_ID_WINDOWS7_FENCE_OUTOFORDER_WAIT	= 1217,
        D3D12_MESSAGE_ID_VIDEO_CREATE_MOTION_ESTIMATOR_INVALID_ARGUMENT	= 1218,
        D3D12_MESSAGE_ID_VIDEO_CREATE_MOTION_VECTOR_HEAP_INVALID_ARGUMENT	= 1219,
        D3D12_MESSAGE_ID_ESTIMATE_MOTION_INVALID_ARGUMENT	= 1220,
        D3D12_MESSAGE_ID_RESOLVE_MOTION_VECTOR_HEAP_INVALID_ARGUMENT	= 1221,
        D3D12_MESSAGE_ID_GETGPUVIRTUALADDRESS_INVALID_HEAP_TYPE	= 1222,
        D3D12_MESSAGE_ID_SET_BACKGROUND_PROCESSING_MODE_INVALID_ARGUMENT	= 1223,
        D3D12_MESSAGE_ID_CREATE_COMMAND_LIST_INVALID_COMMAND_LIST_TYPE_FOR_FEATURE_LEVEL	= 1224,
        D3D12_MESSAGE_ID_CREATE_VIDEOEXTENSIONCOMMAND	= 1225,
        D3D12_MESSAGE_ID_LIVE_VIDEOEXTENSIONCOMMAND	= 1226,
        D3D12_MESSAGE_ID_DESTROY_VIDEOEXTENSIONCOMMAND	= 1227,
        D3D12_MESSAGE_ID_INVALID_VIDEO_EXTENSION_COMMAND_ID	= 1228,
        D3D12_MESSAGE_ID_VIDEO_EXTENSION_COMMAND_INVALID_ARGUMENT	= 1229,
        D3D12_MESSAGE_ID_CREATE_ROOT_SIGNATURE_NOT_UNIQUE_IN_DXIL_LIBRARY	= 1230,
        D3D12_MESSAGE_ID_VARIABLE_SHADING_RATE_NOT_ALLOWED_WITH_TIR	= 1231,
        D3D12_MESSAGE_ID_GEOMETRY_SHADER_OUTPUTTING_BOTH_VIEWPORT_ARRAY_INDEX_AND_SHADING_RATE_NOT_SUPPORTED_ON_DEVICE	= 1232,
        D3D12_MESSAGE_ID_RSSETSHADING_RATE_INVALID_SHADING_RATE	= 1233,
        D3D12_MESSAGE_ID_RSSETSHADING_RATE_SHADING_RATE_NOT_PERMITTED_BY_CAP	= 1234,
        D3D12_MESSAGE_ID_RSSETSHADING_RATE_INVALID_COMBINER	= 1235,
        D3D12_MESSAGE_ID_RSSETSHADINGRATEIMAGE_REQUIRES_TIER_2	= 1236,
        D3D12_MESSAGE_ID_RSSETSHADINGRATE_REQUIRES_TIER_1	= 1237,
        D3D12_MESSAGE_ID_SHADING_RATE_IMAGE_INCORRECT_FORMAT	= 1238,
        D3D12_MESSAGE_ID_SHADING_RATE_IMAGE_INCORRECT_ARRAY_SIZE	= 1239,
        D3D12_MESSAGE_ID_SHADING_RATE_IMAGE_INCORRECT_MIP_LEVEL	= 1240,
        D3D12_MESSAGE_ID_SHADING_RATE_IMAGE_INCORRECT_SAMPLE_COUNT	= 1241,
        D3D12_MESSAGE_ID_SHADING_RATE_IMAGE_INCORRECT_SAMPLE_QUALITY	= 1242,
        D3D12_MESSAGE_ID_NON_RETAIL_SHADER_MODEL_WONT_VALIDATE	= 1243,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_AS_ROOT_SIGNATURE_MISMATCH	= 1244,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_MS_ROOT_SIGNATURE_MISMATCH	= 1245,
        D3D12_MESSAGE_ID_ADD_TO_STATE_OBJECT_ERROR	= 1246,
        D3D12_MESSAGE_ID_CREATE_PROTECTED_RESOURCE_SESSION_INVALID_ARGUMENT	= 1247,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_MS_PSO_DESC_MISMATCH	= 1248,
        D3D12_MESSAGE_ID_CREATEPIPELINESTATE_MS_INCOMPLETE_TYPE	= 1249,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_AS_NOT_MS_MISMATCH	= 1250,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_MS_NOT_PS_MISMATCH	= 1251,
        D3D12_MESSAGE_ID_NONZERO_SAMPLER_FEEDBACK_MIP_REGION_WITH_INCOMPATIBLE_FORMAT	= 1252,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_INPUTLAYOUT_SHADER_MISMATCH	= 1253,
        D3D12_MESSAGE_ID_EMPTY_DISPATCH	= 1254,
        D3D12_MESSAGE_ID_RESOURCE_FORMAT_REQUIRES_SAMPLER_FEEDBACK_CAPABILITY	= 1255,
        D3D12_MESSAGE_ID_SAMPLER_FEEDBACK_MAP_INVALID_MIP_REGION	= 1256,
        D3D12_MESSAGE_ID_SAMPLER_FEEDBACK_MAP_INVALID_DIMENSION	= 1257,
        D3D12_MESSAGE_ID_SAMPLER_FEEDBACK_MAP_INVALID_SAMPLE_COUNT	= 1258,
        D3D12_MESSAGE_ID_SAMPLER_FEEDBACK_MAP_INVALID_SAMPLE_QUALITY	= 1259,
        D3D12_MESSAGE_ID_SAMPLER_FEEDBACK_MAP_INVALID_LAYOUT	= 1260,
        D3D12_MESSAGE_ID_SAMPLER_FEEDBACK_MAP_REQUIRES_UNORDERED_ACCESS_FLAG	= 1261,
        D3D12_MESSAGE_ID_SAMPLER_FEEDBACK_CREATE_UAV_NULL_ARGUMENTS	= 1262,
        D3D12_MESSAGE_ID_SAMPLER_FEEDBACK_UAV_REQUIRES_SAMPLER_FEEDBACK_CAPABILITY	= 1263,
        D3D12_MESSAGE_ID_SAMPLER_FEEDBACK_CREATE_UAV_REQUIRES_FEEDBACK_MAP_FORMAT	= 1264,
        D3D12_MESSAGE_ID_CREATEMESHSHADER_INVALIDSHADERBYTECODE	= 1265,
        D3D12_MESSAGE_ID_CREATEMESHSHADER_OUTOFMEMORY	= 1266,
        D3D12_MESSAGE_ID_CREATEMESHSHADERWITHSTREAMOUTPUT_INVALIDSHADERTYPE	= 1267,
        D3D12_MESSAGE_ID_RESOLVESUBRESOURCE_SAMPLER_FEEDBACK_TRANSCODE_INVALID_FORMAT	= 1268,
        D3D12_MESSAGE_ID_RESOLVESUBRESOURCE_SAMPLER_FEEDBACK_INVALID_MIP_LEVEL_COUNT	= 1269,
        D3D12_MESSAGE_ID_RESOLVESUBRESOURCE_SAMPLER_FEEDBACK_TRANSCODE_ARRAY_SIZE_MISMATCH	= 1270,
        D3D12_MESSAGE_ID_SAMPLER_FEEDBACK_CREATE_UAV_MISMATCHING_TARGETED_RESOURCE	= 1271,
        D3D12_MESSAGE_ID_CREATEMESHSHADER_OUTPUTEXCEEDSMAXSIZE	= 1272,
        D3D12_MESSAGE_ID_CREATEMESHSHADER_GROUPSHAREDEXCEEDSMAXSIZE	= 1273,
        D3D12_MESSAGE_ID_VERTEX_SHADER_OUTPUTTING_BOTH_VIEWPORT_ARRAY_INDEX_AND_SHADING_RATE_NOT_SUPPORTED_ON_DEVICE	= 1274,
        D3D12_MESSAGE_ID_MESH_SHADER_OUTPUTTING_BOTH_VIEWPORT_ARRAY_INDEX_AND_SHADING_RATE_NOT_SUPPORTED_ON_DEVICE	= 1275,
        D3D12_MESSAGE_ID_CREATEMESHSHADER_MISMATCHEDASMSPAYLOADSIZE	= 1276,
        D3D12_MESSAGE_ID_CREATE_ROOT_SIGNATURE_UNBOUNDED_STATIC_DESCRIPTORS	= 1277,
        D3D12_MESSAGE_ID_CREATEAMPLIFICATIONSHADER_INVALIDSHADERBYTECODE	= 1278,
        D3D12_MESSAGE_ID_CREATEAMPLIFICATIONSHADER_OUTOFMEMORY	= 1279,
        D3D12_MESSAGE_ID_CREATE_SHADERCACHESESSION	= 1280,
        D3D12_MESSAGE_ID_LIVE_SHADERCACHESESSION	= 1281,
        D3D12_MESSAGE_ID_DESTROY_SHADERCACHESESSION	= 1282,
        D3D12_MESSAGE_ID_CREATESHADERCACHESESSION_INVALIDARGS	= 1283,
        D3D12_MESSAGE_ID_CREATESHADERCACHESESSION_DISABLED	= 1284,
        D3D12_MESSAGE_ID_CREATESHADERCACHESESSION_ALREADYOPEN	= 1285,
        D3D12_MESSAGE_ID_SHADERCACHECONTROL_DEVELOPERMODE	= 1286,
        D3D12_MESSAGE_ID_SHADERCACHECONTROL_INVALIDFLAGS	= 1287,
        D3D12_MESSAGE_ID_SHADERCACHECONTROL_STATEALREADYSET	= 1288,
        D3D12_MESSAGE_ID_SHADERCACHECONTROL_IGNOREDFLAG	= 1289,
        D3D12_MESSAGE_ID_SHADERCACHESESSION_STOREVALUE_ALREADYPRESENT	= 1290,
        D3D12_MESSAGE_ID_SHADERCACHESESSION_STOREVALUE_HASHCOLLISION	= 1291,
        D3D12_MESSAGE_ID_SHADERCACHESESSION_STOREVALUE_CACHEFULL	= 1292,
        D3D12_MESSAGE_ID_SHADERCACHESESSION_FINDVALUE_NOTFOUND	= 1293,
        D3D12_MESSAGE_ID_SHADERCACHESESSION_CORRUPT	= 1294,
        D3D12_MESSAGE_ID_SHADERCACHESESSION_DISABLED	= 1295,
        D3D12_MESSAGE_ID_OVERSIZED_DISPATCH	= 1296,
        D3D12_MESSAGE_ID_CREATE_VIDEOENCODER	= 1297,
        D3D12_MESSAGE_ID_LIVE_VIDEOENCODER	= 1298,
        D3D12_MESSAGE_ID_DESTROY_VIDEOENCODER	= 1299,
        D3D12_MESSAGE_ID_CREATE_VIDEOENCODERHEAP	= 1300,
        D3D12_MESSAGE_ID_LIVE_VIDEOENCODERHEAP	= 1301,
        D3D12_MESSAGE_ID_DESTROY_VIDEOENCODERHEAP	= 1302,
        D3D12_MESSAGE_ID_COPYTEXTUREREGION_MISMATCH_ENCODE_REFERENCE_ONLY_FLAG	= 1303,
        D3D12_MESSAGE_ID_COPYRESOURCE_MISMATCH_ENCODE_REFERENCE_ONLY_FLAG	= 1304,
        D3D12_MESSAGE_ID_ENCODE_FRAME_INVALID_PARAMETERS	= 1305,
        D3D12_MESSAGE_ID_ENCODE_FRAME_UNSUPPORTED_PARAMETERS	= 1306,
        D3D12_MESSAGE_ID_RESOLVE_ENCODER_OUTPUT_METADATA_INVALID_PARAMETERS	= 1307,
        D3D12_MESSAGE_ID_RESOLVE_ENCODER_OUTPUT_METADATA_UNSUPPORTED_PARAMETERS	= 1308,
        D3D12_MESSAGE_ID_CREATE_VIDEO_ENCODER_INVALID_PARAMETERS	= 1309,
        D3D12_MESSAGE_ID_CREATE_VIDEO_ENCODER_UNSUPPORTED_PARAMETERS	= 1310,
        D3D12_MESSAGE_ID_CREATE_VIDEO_ENCODER_HEAP_INVALID_PARAMETERS	= 1311,
        D3D12_MESSAGE_ID_CREATE_VIDEO_ENCODER_HEAP_UNSUPPORTED_PARAMETERS	= 1312,
        D3D12_MESSAGE_ID_CREATECOMMANDLIST_NULL_COMMANDALLOCATOR	= 1313,
        D3D12_MESSAGE_ID_CLEAR_UNORDERED_ACCESS_VIEW_INVALID_DESCRIPTOR_HANDLE	= 1314,
        D3D12_MESSAGE_ID_DESCRIPTOR_HEAP_NOT_SHADER_VISIBLE	= 1315,
        D3D12_MESSAGE_ID_CREATEBLENDSTATE_BLENDOP_WARNING	= 1316,
        D3D12_MESSAGE_ID_CREATEBLENDSTATE_BLENDOPALPHA_WARNING	= 1317,
        D3D12_MESSAGE_ID_WRITE_COMBINE_PERFORMANCE_WARNING	= 1318,
        D3D12_MESSAGE_ID_RESOLVE_QUERY_INVALID_QUERY_STATE	= 1319,
        D3D12_MESSAGE_ID_SETPRIVATEDATA_NO_ACCESS	= 1320,
        D3D12_MESSAGE_ID_COMMAND_LIST_STATIC_DESCRIPTOR_SAMPLER_MODE_MISMATCH	= 1321,
        D3D12_MESSAGE_ID_GETCOPYABLEFOOTPRINTS_UNSUPPORTED_BUFFER_WIDTH	= 1322,
        D3D12_MESSAGE_ID_CREATEMESHSHADER_TOPOLOGY_MISMATCH	= 1323,
        D3D12_MESSAGE_ID_VRS_SUM_COMBINER_REQUIRES_CAPABILITY	= 1324,
        D3D12_MESSAGE_ID_SETTING_SHADING_RATE_FROM_MS_REQUIRES_CAPABILITY	= 1325,
        D3D12_MESSAGE_ID_SHADERCACHESESSION_SHADERCACHEDELETE_NOTSUPPORTED	= 1326,
        D3D12_MESSAGE_ID_SHADERCACHECONTROL_SHADERCACHECLEAR_NOTSUPPORTED	= 1327,
        D3D12_MESSAGE_ID_CREATERESOURCE_STATE_IGNORED	= 1328,
        D3D12_MESSAGE_ID_UNUSED_CROSS_EXECUTE_SPLIT_BARRIER	= 1329,
        D3D12_MESSAGE_ID_DEVICE_OPEN_SHARED_HANDLE_ACCESS_DENIED	= 1330,
        D3D12_MESSAGE_ID_INCOMPATIBLE_BARRIER_VALUES	= 1331,
        D3D12_MESSAGE_ID_INCOMPATIBLE_BARRIER_ACCESS	= 1332,
        D3D12_MESSAGE_ID_INCOMPATIBLE_BARRIER_SYNC	= 1333,
        D3D12_MESSAGE_ID_INCOMPATIBLE_BARRIER_LAYOUT	= 1334,
        D3D12_MESSAGE_ID_INCOMPATIBLE_BARRIER_TYPE	= 1335,
        D3D12_MESSAGE_ID_OUT_OF_BOUNDS_BARRIER_SUBRESOURCE_RANGE	= 1336,
        D3D12_MESSAGE_ID_INCOMPATIBLE_BARRIER_RESOURCE_DIMENSION	= 1337,
        D3D12_MESSAGE_ID_SET_SCISSOR_RECTS_INVALID_RECT	= 1338,
        D3D12_MESSAGE_ID_SHADING_RATE_SOURCE_REQUIRES_DIMENSION_TEXTURE2D	= 1339,
        D3D12_MESSAGE_ID_BUFFER_BARRIER_SUBREGION_OUT_OF_BOUNDS	= 1340,
        D3D12_MESSAGE_ID_UNSUPPORTED_BARRIER_LAYOUT	= 1341,
        D3D12_MESSAGE_ID_CREATERESOURCEANDHEAP_INVALID_PARAMETERS	= 1342,
        D3D12_MESSAGE_ID_ENHANCED_BARRIERS_NOT_SUPPORTED	= 1343,
        D3D12_MESSAGE_ID_LEGACY_BARRIER_VALIDATION_FORCED_ON	= 1346,
        D3D12_MESSAGE_ID_EMPTY_ROOT_DESCRIPTOR_TABLE	= 1347,
        D3D12_MESSAGE_ID_COMMAND_LIST_DRAW_ELEMENT_OFFSET_UNALIGNED	= 1348,
        D3D12_MESSAGE_ID_ALPHA_BLEND_FACTOR_NOT_SUPPORTED	= 1349,
        D3D12_MESSAGE_ID_BARRIER_INTEROP_INVALID_LAYOUT	= 1350,
        D3D12_MESSAGE_ID_BARRIER_INTEROP_INVALID_STATE	= 1351,
        D3D12_MESSAGE_ID_GRAPHICS_PIPELINE_STATE_DESC_ZERO_SAMPLE_MASK	= 1352,
        D3D12_MESSAGE_ID_INDEPENDENT_STENCIL_REF_NOT_SUPPORTED	= 1353,
        D3D12_MESSAGE_ID_CREATEDEPTHSTENCILSTATE_INDEPENDENT_MASKS_UNSUPPORTED	= 1354,
        D3D12_MESSAGE_ID_TEXTURE_BARRIER_SUBRESOURCES_OUT_OF_BOUNDS	= 1355,
        D3D12_MESSAGE_ID_NON_OPTIMAL_BARRIER_ONLY_EXECUTE_COMMAND_LISTS	= 1356,
        D3D12_MESSAGE_ID_EXECUTE_INDIRECT_ZERO_COMMAND_COUNT	= 1357,
        D3D12_MESSAGE_ID_GPU_BASED_VALIDATION_INCOMPATIBLE_TEXTURE_LAYOUT	= 1358,
        D3D12_MESSAGE_ID_DYNAMIC_INDEX_BUFFER_STRIP_CUT_NOT_SUPPORTED	= 1359,
        D3D12_MESSAGE_ID_PRIMITIVE_TOPOLOGY_TRIANGLE_FANS_NOT_SUPPORTED	= 1360,
        D3D12_MESSAGE_ID_CREATE_SAMPLER_COMPARISON_FUNC_IGNORED	= 1361,
        D3D12_MESSAGE_ID_CREATEHEAP_INVALIDHEAPTYPE	= 1362,
        D3D12_MESSAGE_ID_CREATERESOURCEANDHEAP_INVALIDHEAPTYPE	= 1363,
        D3D12_MESSAGE_ID_DYNAMIC_DEPTH_BIAS_NOT_SUPPORTED	= 1364,
        D3D12_MESSAGE_ID_CREATERASTERIZERSTATE_NON_WHOLE_DYNAMIC_DEPTH_BIAS	= 1365,
        D3D12_MESSAGE_ID_DYNAMIC_DEPTH_BIAS_FLAG_MISSING	= 1366,
        D3D12_MESSAGE_ID_DYNAMIC_DEPTH_BIAS_NO_PIPELINE	= 1367,
        D3D12_MESSAGE_ID_DYNAMIC_INDEX_BUFFER_STRIP_CUT_FLAG_MISSING	= 1368,
        D3D12_MESSAGE_ID_DYNAMIC_INDEX_BUFFER_STRIP_CUT_NO_PIPELINE	= 1369,
        D3D12_MESSAGE_ID_NONNORMALIZED_COORDINATE_SAMPLING_NOT_SUPPORTED	= 1370,
        D3D12_MESSAGE_ID_INVALID_CAST_TARGET	= 1371,
        D3D12_MESSAGE_ID_RENDER_PASS_COMMANDLIST_INVALID_END_STATE	= 1372,
        D3D12_MESSAGE_ID_RENDER_PASS_COMMANDLIST_INVALID_START_STATE	= 1373,
        D3D12_MESSAGE_ID_RENDER_PASS_MISMATCHING_ACCESS	= 1374,
        D3D12_MESSAGE_ID_RENDER_PASS_MISMATCHING_LOCAL_PRESERVE_PARAMETERS	= 1375,
        D3D12_MESSAGE_ID_RENDER_PASS_LOCAL_PRESERVE_RENDER_PARAMETERS_ERROR	= 1376,
        D3D12_MESSAGE_ID_RENDER_PASS_LOCAL_DEPTH_STENCIL_ERROR	= 1377,
        D3D12_MESSAGE_ID_DRAW_POTENTIALLY_OUTSIDE_OF_VALID_RENDER_AREA	= 1378,
        D3D12_MESSAGE_ID_CREATERASTERIZERSTATE_INVALID_LINERASTERIZATIONMODE	= 1379,
        D3D12_MESSAGE_ID_CREATERESOURCE_INVALIDALIGNMENT_SMALLRESOURCE	= 1380,
        D3D12_MESSAGE_ID_GENERIC_DEVICE_OPERATION_UNSUPPORTED	= 1381,
        D3D12_MESSAGE_ID_CREATEGRAPHICSPIPELINESTATE_RENDER_TARGET_WRONG_WRITE_MASK	= 1382,
        D3D12_MESSAGE_ID_PROBABLE_PIX_EVENT_LEAK	= 1383,
        D3D12_MESSAGE_ID_PIX_EVENT_UNDERFLOW	= 1384,
        D3D12_MESSAGE_ID_RECREATEAT_INVALID_TARGET	= 1385,
        D3D12_MESSAGE_ID_RECREATEAT_INSUFFICIENT_SUPPORT	= 1386,
        D3D12_MESSAGE_ID_GPU_BASED_VALIDATION_STRUCTURED_BUFFER_STRIDE_MISMATCH	= 1387,
        D3D12_MESSAGE_ID_DISPATCH_GRAPH_INVALID	= 1388,
        D3D12_MESSAGE_ID_DIRECTSR_SUPERRES_TARGET_FORMAT_INVALID	= 1389,
        D3D12_MESSAGE_ID_DIRECTSR_SUPERRES_TARGET_DIMENSION_INVALID	= 1390,
        D3D12_MESSAGE_ID_DIRECTSR_SUPERRES_SOURCE_COLOR_FORMAT_INVALID	= 1391,
        D3D12_MESSAGE_ID_DIRECTSR_SUPERRES_SOURCE_DEPTH_FORMAT_INVALID	= 1392,
        D3D12_MESSAGE_ID_DIRECTSR_SUPERRES_EXPOSURE_SCALE_FORMAT_INVALID	= 1393,
        D3D12_MESSAGE_ID_DIRECTSR_SUPERRES_ENGINE_CREATE_FLAGS_INVALID	= 1394,
        D3D12_MESSAGE_ID_DIRECTSR_SUPERRES_EXTENSION_INTERNAL_LOAD_FAILURE	= 1395,
        D3D12_MESSAGE_ID_DIRECTSR_SUPERRES_EXTENSION_INTERNAL_ENGINE_CREATION_ERROR	= 1396,
        D3D12_MESSAGE_ID_DIRECTSR_SUPERRES_EXTENSION_INTERNAL_UPSCALER_CREATION_ERROR	= 1397,
        D3D12_MESSAGE_ID_DIRECTSR_SUPERRES_EXTENSION_INTERNAL_UPSCALER_EXECUTION_ERROR	= 1398,
        D3D12_MESSAGE_ID_DIRECTSR_SUPERRES_UPSCALER_EXECUTE_REGION_INVALID	= 1399,
        D3D12_MESSAGE_ID_DIRECTSR_SUPERRES_UPSCALER_EXECUTE_TIME_DELTA_INVALID	= 1400,
        D3D12_MESSAGE_ID_DIRECTSR_SUPERRES_UPSCALER_EXECUTE_REQUIRED_TEXTURE_IS_NULL	= 1401,
        D3D12_MESSAGE_ID_DIRECTSR_SUPERRES_UPSCALER_EXECUTE_MOTION_VECTORS_FORMAT_INVALID	= 1402,
        D3D12_MESSAGE_ID_DIRECTSR_SUPERRES_UPSCALER_EXECUTE_FLAGS_INVALID	= 1403,
        D3D12_MESSAGE_ID_DIRECTSR_SUPERRES_UPSCALER_EXECUTE_FORMAT_INVALID	= 1404,
        D3D12_MESSAGE_ID_DIRECTSR_SUPERRES_UPSCALER_EXECUTE_EXPOSURE_SCALE_TEXTURE_SIZE_INVALID	= 1405,
        D3D12_MESSAGE_ID_DIRECTSR_SUPERRES_VARIANT_INDEX_OUT_OF_BOUNDS	= 1406,
        D3D12_MESSAGE_ID_DIRECTSR_SUPERRES_VARIANT_ID_NOT_FOUND	= 1407,
        D3D12_MESSAGE_ID_DIRECTSR_SUPERRES_DUPLICATE_VARIANT_ID	= 1408,
        D3D12_MESSAGE_ID_DIRECTSR_OUT_OF_MEMORY	= 1409,
        D3D12_MESSAGE_ID_DIRECTSR_SUPERRES_UPSCALER_EXECUTE_UNEXPECTED_TEXTURE_IS_IGNORED	= 1410,
        D3D12_MESSAGE_ID_DIRECTSR_SUPERRES_UPSCALER_EVICT_UNDERFLOW	= 1411,
        D3D12_MESSAGE_ID_DIRECTSR_SUPERRES_UPSCALER_EXECUTE_OPTIONAL_TEXTURE_IS_NULL	= 1412,
        D3D12_MESSAGE_ID_DIRECTSR_SUPERRES_UPSCALER_EXECUTE_INVALID_CAMERA_JITTER	= 1413,
        D3D12_MESSAGE_ID_CREATE_STATE_OBJECT_WARNING	= 1414,
        D3D12_MESSAGE_ID_GUID_TEXTURE_LAYOUT_UNSUPPORTED	= 1415,
        D3D12_MESSAGE_ID_RESOLVE_ENCODER_INPUT_PARAM_LAYOUT_INVALID_PARAMETERS	= 1416,
        D3D12_MESSAGE_ID_INVALID_BARRIER_ACCESS	= 1417,
        D3D12_MESSAGE_ID_COMMAND_LIST_DRAW_INSTANCE_COUNT_ZERO	= 1418,
        D3D12_MESSAGE_ID_DESCRIPTOR_HEAP_NOT_SET_BEFORE_ROOT_SIGNATURE_WITH_DIRECTLY_INDEXED_FLAG	= 1419,
        D3D12_MESSAGE_ID_DIFFERENT_DESCRIPTOR_HEAP_SET_AFTER_ROOT_SIGNATURE_WITH_DIRECTLY_INDEXED_FLAG	= 1420,
        D3D12_MESSAGE_ID_D3D12_MESSAGES_END	= ( D3D12_MESSAGE_ID_DIFFERENT_DESCRIPTOR_HEAP_SET_AFTER_ROOT_SIGNATURE_WITH_DIRECTLY_INDEXED_FLAG + 1 ) 
    } 	D3D12_MESSAGE_ID;

typedef struct D3D12_MESSAGE
    {
    D3D12_MESSAGE_CATEGORY Category;
    D3D12_MESSAGE_SEVERITY Severity;
    D3D12_MESSAGE_ID ID;
    _Field_size_(DescriptionByteLength)  const char *pDescription;
    SIZE_T DescriptionByteLength;
    } 	D3D12_MESSAGE;

typedef struct D3D12_INFO_QUEUE_FILTER_DESC
    {
    UINT NumCategories;
    _Field_size_(NumCategories)  D3D12_MESSAGE_CATEGORY *pCategoryList;
    UINT NumSeverities;
    _Field_size_(NumSeverities)  D3D12_MESSAGE_SEVERITY *pSeverityList;
    UINT NumIDs;
    _Field_size_(NumIDs)  D3D12_MESSAGE_ID *pIDList;
    } 	D3D12_INFO_QUEUE_FILTER_DESC;

typedef struct D3D12_INFO_QUEUE_FILTER
    {
    D3D12_INFO_QUEUE_FILTER_DESC AllowList;
    D3D12_INFO_QUEUE_FILTER_DESC DenyList;
    } 	D3D12_INFO_QUEUE_FILTER;

#define D3D12_INFO_QUEUE_DEFAULT_MESSAGE_COUNT_LIMIT 1024


extern RPC_IF_HANDLE __MIDL_itf_d3d12sdklayers_0000_0018_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12sdklayers_0000_0018_v0_0_s_ifspec;

#ifndef __ID3D12InfoQueue_INTERFACE_DEFINED__
#define __ID3D12InfoQueue_INTERFACE_DEFINED__

/* interface ID3D12InfoQueue */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12InfoQueue;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("0742a90b-c387-483f-b946-30a7e4e61458")
    ID3D12InfoQueue : public IUnknown
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE SetMessageCountLimit( 
            _In_  UINT64 MessageCountLimit) = 0;
        
        virtual void STDMETHODCALLTYPE ClearStoredMessages( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetMessage( 
            _In_  UINT64 MessageIndex,
            _Out_writes_bytes_opt_(*pMessageByteLength)  D3D12_MESSAGE *pMessage,
            _Inout_  SIZE_T *pMessageByteLength) = 0;
        
        virtual UINT64 STDMETHODCALLTYPE GetNumMessagesAllowedByStorageFilter( void) = 0;
        
        virtual UINT64 STDMETHODCALLTYPE GetNumMessagesDeniedByStorageFilter( void) = 0;
        
        virtual UINT64 STDMETHODCALLTYPE GetNumStoredMessages( void) = 0;
        
        virtual UINT64 STDMETHODCALLTYPE GetNumStoredMessagesAllowedByRetrievalFilter( void) = 0;
        
        virtual UINT64 STDMETHODCALLTYPE GetNumMessagesDiscardedByMessageCountLimit( void) = 0;
        
        virtual UINT64 STDMETHODCALLTYPE GetMessageCountLimit( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE AddStorageFilterEntries( 
            _In_  D3D12_INFO_QUEUE_FILTER *pFilter) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetStorageFilter( 
            _Out_writes_bytes_opt_(*pFilterByteLength)  D3D12_INFO_QUEUE_FILTER *pFilter,
            _Inout_  SIZE_T *pFilterByteLength) = 0;
        
        virtual void STDMETHODCALLTYPE ClearStorageFilter( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE PushEmptyStorageFilter( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE PushCopyOfStorageFilter( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE PushStorageFilter( 
            _In_  D3D12_INFO_QUEUE_FILTER *pFilter) = 0;
        
        virtual void STDMETHODCALLTYPE PopStorageFilter( void) = 0;
        
        virtual UINT STDMETHODCALLTYPE GetStorageFilterStackSize( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE AddRetrievalFilterEntries( 
            _In_  D3D12_INFO_QUEUE_FILTER *pFilter) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetRetrievalFilter( 
            _Out_writes_bytes_opt_(*pFilterByteLength)  D3D12_INFO_QUEUE_FILTER *pFilter,
            _Inout_  SIZE_T *pFilterByteLength) = 0;
        
        virtual void STDMETHODCALLTYPE ClearRetrievalFilter( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE PushEmptyRetrievalFilter( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE PushCopyOfRetrievalFilter( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE PushRetrievalFilter( 
            _In_  D3D12_INFO_QUEUE_FILTER *pFilter) = 0;
        
        virtual void STDMETHODCALLTYPE PopRetrievalFilter( void) = 0;
        
        virtual UINT STDMETHODCALLTYPE GetRetrievalFilterStackSize( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE AddMessage( 
            _In_  D3D12_MESSAGE_CATEGORY Category,
            _In_  D3D12_MESSAGE_SEVERITY Severity,
            _In_  D3D12_MESSAGE_ID ID,
            _In_  LPCSTR pDescription) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE AddApplicationMessage( 
            _In_  D3D12_MESSAGE_SEVERITY Severity,
            _In_  LPCSTR pDescription) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE SetBreakOnCategory( 
            _In_  D3D12_MESSAGE_CATEGORY Category,
            _In_  BOOL bEnable) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE SetBreakOnSeverity( 
            _In_  D3D12_MESSAGE_SEVERITY Severity,
            _In_  BOOL bEnable) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE SetBreakOnID( 
            _In_  D3D12_MESSAGE_ID ID,
            _In_  BOOL bEnable) = 0;
        
        virtual BOOL STDMETHODCALLTYPE GetBreakOnCategory( 
            _In_  D3D12_MESSAGE_CATEGORY Category) = 0;
        
        virtual BOOL STDMETHODCALLTYPE GetBreakOnSeverity( 
            _In_  D3D12_MESSAGE_SEVERITY Severity) = 0;
        
        virtual BOOL STDMETHODCALLTYPE GetBreakOnID( 
            _In_  D3D12_MESSAGE_ID ID) = 0;
        
        virtual void STDMETHODCALLTYPE SetMuteDebugOutput( 
            _In_  BOOL bMute) = 0;
        
        virtual BOOL STDMETHODCALLTYPE GetMuteDebugOutput( void) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12InfoQueueVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12InfoQueue * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12InfoQueue * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12InfoQueue * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, SetMessageCountLimit)
        HRESULT ( STDMETHODCALLTYPE *SetMessageCountLimit )( 
            ID3D12InfoQueue * This,
            _In_  UINT64 MessageCountLimit);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, ClearStoredMessages)
        void ( STDMETHODCALLTYPE *ClearStoredMessages )( 
            ID3D12InfoQueue * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetMessage)
        HRESULT ( STDMETHODCALLTYPE *GetMessage )( 
            ID3D12InfoQueue * This,
            _In_  UINT64 MessageIndex,
            _Out_writes_bytes_opt_(*pMessageByteLength)  D3D12_MESSAGE *pMessage,
            _Inout_  SIZE_T *pMessageByteLength);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetNumMessagesAllowedByStorageFilter)
        UINT64 ( STDMETHODCALLTYPE *GetNumMessagesAllowedByStorageFilter )( 
            ID3D12InfoQueue * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetNumMessagesDeniedByStorageFilter)
        UINT64 ( STDMETHODCALLTYPE *GetNumMessagesDeniedByStorageFilter )( 
            ID3D12InfoQueue * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetNumStoredMessages)
        UINT64 ( STDMETHODCALLTYPE *GetNumStoredMessages )( 
            ID3D12InfoQueue * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetNumStoredMessagesAllowedByRetrievalFilter)
        UINT64 ( STDMETHODCALLTYPE *GetNumStoredMessagesAllowedByRetrievalFilter )( 
            ID3D12InfoQueue * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetNumMessagesDiscardedByMessageCountLimit)
        UINT64 ( STDMETHODCALLTYPE *GetNumMessagesDiscardedByMessageCountLimit )( 
            ID3D12InfoQueue * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetMessageCountLimit)
        UINT64 ( STDMETHODCALLTYPE *GetMessageCountLimit )( 
            ID3D12InfoQueue * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, AddStorageFilterEntries)
        HRESULT ( STDMETHODCALLTYPE *AddStorageFilterEntries )( 
            ID3D12InfoQueue * This,
            _In_  D3D12_INFO_QUEUE_FILTER *pFilter);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetStorageFilter)
        HRESULT ( STDMETHODCALLTYPE *GetStorageFilter )( 
            ID3D12InfoQueue * This,
            _Out_writes_bytes_opt_(*pFilterByteLength)  D3D12_INFO_QUEUE_FILTER *pFilter,
            _Inout_  SIZE_T *pFilterByteLength);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, ClearStorageFilter)
        void ( STDMETHODCALLTYPE *ClearStorageFilter )( 
            ID3D12InfoQueue * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, PushEmptyStorageFilter)
        HRESULT ( STDMETHODCALLTYPE *PushEmptyStorageFilter )( 
            ID3D12InfoQueue * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, PushCopyOfStorageFilter)
        HRESULT ( STDMETHODCALLTYPE *PushCopyOfStorageFilter )( 
            ID3D12InfoQueue * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, PushStorageFilter)
        HRESULT ( STDMETHODCALLTYPE *PushStorageFilter )( 
            ID3D12InfoQueue * This,
            _In_  D3D12_INFO_QUEUE_FILTER *pFilter);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, PopStorageFilter)
        void ( STDMETHODCALLTYPE *PopStorageFilter )( 
            ID3D12InfoQueue * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetStorageFilterStackSize)
        UINT ( STDMETHODCALLTYPE *GetStorageFilterStackSize )( 
            ID3D12InfoQueue * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, AddRetrievalFilterEntries)
        HRESULT ( STDMETHODCALLTYPE *AddRetrievalFilterEntries )( 
            ID3D12InfoQueue * This,
            _In_  D3D12_INFO_QUEUE_FILTER *pFilter);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetRetrievalFilter)
        HRESULT ( STDMETHODCALLTYPE *GetRetrievalFilter )( 
            ID3D12InfoQueue * This,
            _Out_writes_bytes_opt_(*pFilterByteLength)  D3D12_INFO_QUEUE_FILTER *pFilter,
            _Inout_  SIZE_T *pFilterByteLength);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, ClearRetrievalFilter)
        void ( STDMETHODCALLTYPE *ClearRetrievalFilter )( 
            ID3D12InfoQueue * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, PushEmptyRetrievalFilter)
        HRESULT ( STDMETHODCALLTYPE *PushEmptyRetrievalFilter )( 
            ID3D12InfoQueue * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, PushCopyOfRetrievalFilter)
        HRESULT ( STDMETHODCALLTYPE *PushCopyOfRetrievalFilter )( 
            ID3D12InfoQueue * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, PushRetrievalFilter)
        HRESULT ( STDMETHODCALLTYPE *PushRetrievalFilter )( 
            ID3D12InfoQueue * This,
            _In_  D3D12_INFO_QUEUE_FILTER *pFilter);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, PopRetrievalFilter)
        void ( STDMETHODCALLTYPE *PopRetrievalFilter )( 
            ID3D12InfoQueue * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetRetrievalFilterStackSize)
        UINT ( STDMETHODCALLTYPE *GetRetrievalFilterStackSize )( 
            ID3D12InfoQueue * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, AddMessage)
        HRESULT ( STDMETHODCALLTYPE *AddMessage )( 
            ID3D12InfoQueue * This,
            _In_  D3D12_MESSAGE_CATEGORY Category,
            _In_  D3D12_MESSAGE_SEVERITY Severity,
            _In_  D3D12_MESSAGE_ID ID,
            _In_  LPCSTR pDescription);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, AddApplicationMessage)
        HRESULT ( STDMETHODCALLTYPE *AddApplicationMessage )( 
            ID3D12InfoQueue * This,
            _In_  D3D12_MESSAGE_SEVERITY Severity,
            _In_  LPCSTR pDescription);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, SetBreakOnCategory)
        HRESULT ( STDMETHODCALLTYPE *SetBreakOnCategory )( 
            ID3D12InfoQueue * This,
            _In_  D3D12_MESSAGE_CATEGORY Category,
            _In_  BOOL bEnable);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, SetBreakOnSeverity)
        HRESULT ( STDMETHODCALLTYPE *SetBreakOnSeverity )( 
            ID3D12InfoQueue * This,
            _In_  D3D12_MESSAGE_SEVERITY Severity,
            _In_  BOOL bEnable);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, SetBreakOnID)
        HRESULT ( STDMETHODCALLTYPE *SetBreakOnID )( 
            ID3D12InfoQueue * This,
            _In_  D3D12_MESSAGE_ID ID,
            _In_  BOOL bEnable);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetBreakOnCategory)
        BOOL ( STDMETHODCALLTYPE *GetBreakOnCategory )( 
            ID3D12InfoQueue * This,
            _In_  D3D12_MESSAGE_CATEGORY Category);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetBreakOnSeverity)
        BOOL ( STDMETHODCALLTYPE *GetBreakOnSeverity )( 
            ID3D12InfoQueue * This,
            _In_  D3D12_MESSAGE_SEVERITY Severity);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetBreakOnID)
        BOOL ( STDMETHODCALLTYPE *GetBreakOnID )( 
            ID3D12InfoQueue * This,
            _In_  D3D12_MESSAGE_ID ID);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, SetMuteDebugOutput)
        void ( STDMETHODCALLTYPE *SetMuteDebugOutput )( 
            ID3D12InfoQueue * This,
            _In_  BOOL bMute);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetMuteDebugOutput)
        BOOL ( STDMETHODCALLTYPE *GetMuteDebugOutput )( 
            ID3D12InfoQueue * This);
        
        END_INTERFACE
    } ID3D12InfoQueueVtbl;

    interface ID3D12InfoQueue
    {
        CONST_VTBL struct ID3D12InfoQueueVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12InfoQueue_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12InfoQueue_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12InfoQueue_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12InfoQueue_SetMessageCountLimit(This,MessageCountLimit)	\
    ( (This)->lpVtbl -> SetMessageCountLimit(This,MessageCountLimit) ) 

#define ID3D12InfoQueue_ClearStoredMessages(This)	\
    ( (This)->lpVtbl -> ClearStoredMessages(This) ) 

#define ID3D12InfoQueue_GetMessage(This,MessageIndex,pMessage,pMessageByteLength)	\
    ( (This)->lpVtbl -> GetMessage(This,MessageIndex,pMessage,pMessageByteLength) ) 

#define ID3D12InfoQueue_GetNumMessagesAllowedByStorageFilter(This)	\
    ( (This)->lpVtbl -> GetNumMessagesAllowedByStorageFilter(This) ) 

#define ID3D12InfoQueue_GetNumMessagesDeniedByStorageFilter(This)	\
    ( (This)->lpVtbl -> GetNumMessagesDeniedByStorageFilter(This) ) 

#define ID3D12InfoQueue_GetNumStoredMessages(This)	\
    ( (This)->lpVtbl -> GetNumStoredMessages(This) ) 

#define ID3D12InfoQueue_GetNumStoredMessagesAllowedByRetrievalFilter(This)	\
    ( (This)->lpVtbl -> GetNumStoredMessagesAllowedByRetrievalFilter(This) ) 

#define ID3D12InfoQueue_GetNumMessagesDiscardedByMessageCountLimit(This)	\
    ( (This)->lpVtbl -> GetNumMessagesDiscardedByMessageCountLimit(This) ) 

#define ID3D12InfoQueue_GetMessageCountLimit(This)	\
    ( (This)->lpVtbl -> GetMessageCountLimit(This) ) 

#define ID3D12InfoQueue_AddStorageFilterEntries(This,pFilter)	\
    ( (This)->lpVtbl -> AddStorageFilterEntries(This,pFilter) ) 

#define ID3D12InfoQueue_GetStorageFilter(This,pFilter,pFilterByteLength)	\
    ( (This)->lpVtbl -> GetStorageFilter(This,pFilter,pFilterByteLength) ) 

#define ID3D12InfoQueue_ClearStorageFilter(This)	\
    ( (This)->lpVtbl -> ClearStorageFilter(This) ) 

#define ID3D12InfoQueue_PushEmptyStorageFilter(This)	\
    ( (This)->lpVtbl -> PushEmptyStorageFilter(This) ) 

#define ID3D12InfoQueue_PushCopyOfStorageFilter(This)	\
    ( (This)->lpVtbl -> PushCopyOfStorageFilter(This) ) 

#define ID3D12InfoQueue_PushStorageFilter(This,pFilter)	\
    ( (This)->lpVtbl -> PushStorageFilter(This,pFilter) ) 

#define ID3D12InfoQueue_PopStorageFilter(This)	\
    ( (This)->lpVtbl -> PopStorageFilter(This) ) 

#define ID3D12InfoQueue_GetStorageFilterStackSize(This)	\
    ( (This)->lpVtbl -> GetStorageFilterStackSize(This) ) 

#define ID3D12InfoQueue_AddRetrievalFilterEntries(This,pFilter)	\
    ( (This)->lpVtbl -> AddRetrievalFilterEntries(This,pFilter) ) 

#define ID3D12InfoQueue_GetRetrievalFilter(This,pFilter,pFilterByteLength)	\
    ( (This)->lpVtbl -> GetRetrievalFilter(This,pFilter,pFilterByteLength) ) 

#define ID3D12InfoQueue_ClearRetrievalFilter(This)	\
    ( (This)->lpVtbl -> ClearRetrievalFilter(This) ) 

#define ID3D12InfoQueue_PushEmptyRetrievalFilter(This)	\
    ( (This)->lpVtbl -> PushEmptyRetrievalFilter(This) ) 

#define ID3D12InfoQueue_PushCopyOfRetrievalFilter(This)	\
    ( (This)->lpVtbl -> PushCopyOfRetrievalFilter(This) ) 

#define ID3D12InfoQueue_PushRetrievalFilter(This,pFilter)	\
    ( (This)->lpVtbl -> PushRetrievalFilter(This,pFilter) ) 

#define ID3D12InfoQueue_PopRetrievalFilter(This)	\
    ( (This)->lpVtbl -> PopRetrievalFilter(This) ) 

#define ID3D12InfoQueue_GetRetrievalFilterStackSize(This)	\
    ( (This)->lpVtbl -> GetRetrievalFilterStackSize(This) ) 

#define ID3D12InfoQueue_AddMessage(This,Category,Severity,ID,pDescription)	\
    ( (This)->lpVtbl -> AddMessage(This,Category,Severity,ID,pDescription) ) 

#define ID3D12InfoQueue_AddApplicationMessage(This,Severity,pDescription)	\
    ( (This)->lpVtbl -> AddApplicationMessage(This,Severity,pDescription) ) 

#define ID3D12InfoQueue_SetBreakOnCategory(This,Category,bEnable)	\
    ( (This)->lpVtbl -> SetBreakOnCategory(This,Category,bEnable) ) 

#define ID3D12InfoQueue_SetBreakOnSeverity(This,Severity,bEnable)	\
    ( (This)->lpVtbl -> SetBreakOnSeverity(This,Severity,bEnable) ) 

#define ID3D12InfoQueue_SetBreakOnID(This,ID,bEnable)	\
    ( (This)->lpVtbl -> SetBreakOnID(This,ID,bEnable) ) 

#define ID3D12InfoQueue_GetBreakOnCategory(This,Category)	\
    ( (This)->lpVtbl -> GetBreakOnCategory(This,Category) ) 

#define ID3D12InfoQueue_GetBreakOnSeverity(This,Severity)	\
    ( (This)->lpVtbl -> GetBreakOnSeverity(This,Severity) ) 

#define ID3D12InfoQueue_GetBreakOnID(This,ID)	\
    ( (This)->lpVtbl -> GetBreakOnID(This,ID) ) 

#define ID3D12InfoQueue_SetMuteDebugOutput(This,bMute)	\
    ( (This)->lpVtbl -> SetMuteDebugOutput(This,bMute) ) 

#define ID3D12InfoQueue_GetMuteDebugOutput(This)	\
    ( (This)->lpVtbl -> GetMuteDebugOutput(This) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12InfoQueue_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12sdklayers_0000_0019 */
/* [local] */ 

typedef 
enum D3D12_MESSAGE_CALLBACK_FLAGS
    {
        D3D12_MESSAGE_CALLBACK_FLAG_NONE	= 0,
        D3D12_MESSAGE_CALLBACK_IGNORE_FILTERS	= 0x1
    } 	D3D12_MESSAGE_CALLBACK_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_MESSAGE_CALLBACK_FLAGS)
typedef void ( __stdcall *D3D12MessageFunc )( 
    D3D12_MESSAGE_CATEGORY Category,
    D3D12_MESSAGE_SEVERITY Severity,
    D3D12_MESSAGE_ID ID,
    LPCSTR pDescription,
    void *pContext);



extern RPC_IF_HANDLE __MIDL_itf_d3d12sdklayers_0000_0019_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12sdklayers_0000_0019_v0_0_s_ifspec;

#ifndef __ID3D12InfoQueue1_INTERFACE_DEFINED__
#define __ID3D12InfoQueue1_INTERFACE_DEFINED__

/* interface ID3D12InfoQueue1 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12InfoQueue1;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("2852dd88-b484-4c0c-b6b1-67168500e600")
    ID3D12InfoQueue1 : public ID3D12InfoQueue
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE RegisterMessageCallback( 
            _In_  D3D12MessageFunc CallbackFunc,
            _In_  D3D12_MESSAGE_CALLBACK_FLAGS CallbackFilterFlags,
            _Inout_  void *pContext,
            _Inout_  DWORD *pCallbackCookie) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE UnregisterMessageCallback( 
            _In_  DWORD CallbackCookie) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12InfoQueue1Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12InfoQueue1 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12InfoQueue1 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12InfoQueue1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, SetMessageCountLimit)
        HRESULT ( STDMETHODCALLTYPE *SetMessageCountLimit )( 
            ID3D12InfoQueue1 * This,
            _In_  UINT64 MessageCountLimit);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, ClearStoredMessages)
        void ( STDMETHODCALLTYPE *ClearStoredMessages )( 
            ID3D12InfoQueue1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetMessage)
        HRESULT ( STDMETHODCALLTYPE *GetMessage )( 
            ID3D12InfoQueue1 * This,
            _In_  UINT64 MessageIndex,
            _Out_writes_bytes_opt_(*pMessageByteLength)  D3D12_MESSAGE *pMessage,
            _Inout_  SIZE_T *pMessageByteLength);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetNumMessagesAllowedByStorageFilter)
        UINT64 ( STDMETHODCALLTYPE *GetNumMessagesAllowedByStorageFilter )( 
            ID3D12InfoQueue1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetNumMessagesDeniedByStorageFilter)
        UINT64 ( STDMETHODCALLTYPE *GetNumMessagesDeniedByStorageFilter )( 
            ID3D12InfoQueue1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetNumStoredMessages)
        UINT64 ( STDMETHODCALLTYPE *GetNumStoredMessages )( 
            ID3D12InfoQueue1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetNumStoredMessagesAllowedByRetrievalFilter)
        UINT64 ( STDMETHODCALLTYPE *GetNumStoredMessagesAllowedByRetrievalFilter )( 
            ID3D12InfoQueue1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetNumMessagesDiscardedByMessageCountLimit)
        UINT64 ( STDMETHODCALLTYPE *GetNumMessagesDiscardedByMessageCountLimit )( 
            ID3D12InfoQueue1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetMessageCountLimit)
        UINT64 ( STDMETHODCALLTYPE *GetMessageCountLimit )( 
            ID3D12InfoQueue1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, AddStorageFilterEntries)
        HRESULT ( STDMETHODCALLTYPE *AddStorageFilterEntries )( 
            ID3D12InfoQueue1 * This,
            _In_  D3D12_INFO_QUEUE_FILTER *pFilter);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetStorageFilter)
        HRESULT ( STDMETHODCALLTYPE *GetStorageFilter )( 
            ID3D12InfoQueue1 * This,
            _Out_writes_bytes_opt_(*pFilterByteLength)  D3D12_INFO_QUEUE_FILTER *pFilter,
            _Inout_  SIZE_T *pFilterByteLength);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, ClearStorageFilter)
        void ( STDMETHODCALLTYPE *ClearStorageFilter )( 
            ID3D12InfoQueue1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, PushEmptyStorageFilter)
        HRESULT ( STDMETHODCALLTYPE *PushEmptyStorageFilter )( 
            ID3D12InfoQueue1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, PushCopyOfStorageFilter)
        HRESULT ( STDMETHODCALLTYPE *PushCopyOfStorageFilter )( 
            ID3D12InfoQueue1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, PushStorageFilter)
        HRESULT ( STDMETHODCALLTYPE *PushStorageFilter )( 
            ID3D12InfoQueue1 * This,
            _In_  D3D12_INFO_QUEUE_FILTER *pFilter);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, PopStorageFilter)
        void ( STDMETHODCALLTYPE *PopStorageFilter )( 
            ID3D12InfoQueue1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetStorageFilterStackSize)
        UINT ( STDMETHODCALLTYPE *GetStorageFilterStackSize )( 
            ID3D12InfoQueue1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, AddRetrievalFilterEntries)
        HRESULT ( STDMETHODCALLTYPE *AddRetrievalFilterEntries )( 
            ID3D12InfoQueue1 * This,
            _In_  D3D12_INFO_QUEUE_FILTER *pFilter);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetRetrievalFilter)
        HRESULT ( STDMETHODCALLTYPE *GetRetrievalFilter )( 
            ID3D12InfoQueue1 * This,
            _Out_writes_bytes_opt_(*pFilterByteLength)  D3D12_INFO_QUEUE_FILTER *pFilter,
            _Inout_  SIZE_T *pFilterByteLength);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, ClearRetrievalFilter)
        void ( STDMETHODCALLTYPE *ClearRetrievalFilter )( 
            ID3D12InfoQueue1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, PushEmptyRetrievalFilter)
        HRESULT ( STDMETHODCALLTYPE *PushEmptyRetrievalFilter )( 
            ID3D12InfoQueue1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, PushCopyOfRetrievalFilter)
        HRESULT ( STDMETHODCALLTYPE *PushCopyOfRetrievalFilter )( 
            ID3D12InfoQueue1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, PushRetrievalFilter)
        HRESULT ( STDMETHODCALLTYPE *PushRetrievalFilter )( 
            ID3D12InfoQueue1 * This,
            _In_  D3D12_INFO_QUEUE_FILTER *pFilter);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, PopRetrievalFilter)
        void ( STDMETHODCALLTYPE *PopRetrievalFilter )( 
            ID3D12InfoQueue1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetRetrievalFilterStackSize)
        UINT ( STDMETHODCALLTYPE *GetRetrievalFilterStackSize )( 
            ID3D12InfoQueue1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, AddMessage)
        HRESULT ( STDMETHODCALLTYPE *AddMessage )( 
            ID3D12InfoQueue1 * This,
            _In_  D3D12_MESSAGE_CATEGORY Category,
            _In_  D3D12_MESSAGE_SEVERITY Severity,
            _In_  D3D12_MESSAGE_ID ID,
            _In_  LPCSTR pDescription);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, AddApplicationMessage)
        HRESULT ( STDMETHODCALLTYPE *AddApplicationMessage )( 
            ID3D12InfoQueue1 * This,
            _In_  D3D12_MESSAGE_SEVERITY Severity,
            _In_  LPCSTR pDescription);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, SetBreakOnCategory)
        HRESULT ( STDMETHODCALLTYPE *SetBreakOnCategory )( 
            ID3D12InfoQueue1 * This,
            _In_  D3D12_MESSAGE_CATEGORY Category,
            _In_  BOOL bEnable);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, SetBreakOnSeverity)
        HRESULT ( STDMETHODCALLTYPE *SetBreakOnSeverity )( 
            ID3D12InfoQueue1 * This,
            _In_  D3D12_MESSAGE_SEVERITY Severity,
            _In_  BOOL bEnable);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, SetBreakOnID)
        HRESULT ( STDMETHODCALLTYPE *SetBreakOnID )( 
            ID3D12InfoQueue1 * This,
            _In_  D3D12_MESSAGE_ID ID,
            _In_  BOOL bEnable);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetBreakOnCategory)
        BOOL ( STDMETHODCALLTYPE *GetBreakOnCategory )( 
            ID3D12InfoQueue1 * This,
            _In_  D3D12_MESSAGE_CATEGORY Category);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetBreakOnSeverity)
        BOOL ( STDMETHODCALLTYPE *GetBreakOnSeverity )( 
            ID3D12InfoQueue1 * This,
            _In_  D3D12_MESSAGE_SEVERITY Severity);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetBreakOnID)
        BOOL ( STDMETHODCALLTYPE *GetBreakOnID )( 
            ID3D12InfoQueue1 * This,
            _In_  D3D12_MESSAGE_ID ID);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, SetMuteDebugOutput)
        void ( STDMETHODCALLTYPE *SetMuteDebugOutput )( 
            ID3D12InfoQueue1 * This,
            _In_  BOOL bMute);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue, GetMuteDebugOutput)
        BOOL ( STDMETHODCALLTYPE *GetMuteDebugOutput )( 
            ID3D12InfoQueue1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue1, RegisterMessageCallback)
        HRESULT ( STDMETHODCALLTYPE *RegisterMessageCallback )( 
            ID3D12InfoQueue1 * This,
            _In_  D3D12MessageFunc CallbackFunc,
            _In_  D3D12_MESSAGE_CALLBACK_FLAGS CallbackFilterFlags,
            _Inout_  void *pContext,
            _Inout_  DWORD *pCallbackCookie);
        
        DECLSPEC_XFGVIRT(ID3D12InfoQueue1, UnregisterMessageCallback)
        HRESULT ( STDMETHODCALLTYPE *UnregisterMessageCallback )( 
            ID3D12InfoQueue1 * This,
            _In_  DWORD CallbackCookie);
        
        END_INTERFACE
    } ID3D12InfoQueue1Vtbl;

    interface ID3D12InfoQueue1
    {
        CONST_VTBL struct ID3D12InfoQueue1Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12InfoQueue1_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12InfoQueue1_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12InfoQueue1_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12InfoQueue1_SetMessageCountLimit(This,MessageCountLimit)	\
    ( (This)->lpVtbl -> SetMessageCountLimit(This,MessageCountLimit) ) 

#define ID3D12InfoQueue1_ClearStoredMessages(This)	\
    ( (This)->lpVtbl -> ClearStoredMessages(This) ) 

#define ID3D12InfoQueue1_GetMessage(This,MessageIndex,pMessage,pMessageByteLength)	\
    ( (This)->lpVtbl -> GetMessage(This,MessageIndex,pMessage,pMessageByteLength) ) 

#define ID3D12InfoQueue1_GetNumMessagesAllowedByStorageFilter(This)	\
    ( (This)->lpVtbl -> GetNumMessagesAllowedByStorageFilter(This) ) 

#define ID3D12InfoQueue1_GetNumMessagesDeniedByStorageFilter(This)	\
    ( (This)->lpVtbl -> GetNumMessagesDeniedByStorageFilter(This) ) 

#define ID3D12InfoQueue1_GetNumStoredMessages(This)	\
    ( (This)->lpVtbl -> GetNumStoredMessages(This) ) 

#define ID3D12InfoQueue1_GetNumStoredMessagesAllowedByRetrievalFilter(This)	\
    ( (This)->lpVtbl -> GetNumStoredMessagesAllowedByRetrievalFilter(This) ) 

#define ID3D12InfoQueue1_GetNumMessagesDiscardedByMessageCountLimit(This)	\
    ( (This)->lpVtbl -> GetNumMessagesDiscardedByMessageCountLimit(This) ) 

#define ID3D12InfoQueue1_GetMessageCountLimit(This)	\
    ( (This)->lpVtbl -> GetMessageCountLimit(This) ) 

#define ID3D12InfoQueue1_AddStorageFilterEntries(This,pFilter)	\
    ( (This)->lpVtbl -> AddStorageFilterEntries(This,pFilter) ) 

#define ID3D12InfoQueue1_GetStorageFilter(This,pFilter,pFilterByteLength)	\
    ( (This)->lpVtbl -> GetStorageFilter(This,pFilter,pFilterByteLength) ) 

#define ID3D12InfoQueue1_ClearStorageFilter(This)	\
    ( (This)->lpVtbl -> ClearStorageFilter(This) ) 

#define ID3D12InfoQueue1_PushEmptyStorageFilter(This)	\
    ( (This)->lpVtbl -> PushEmptyStorageFilter(This) ) 

#define ID3D12InfoQueue1_PushCopyOfStorageFilter(This)	\
    ( (This)->lpVtbl -> PushCopyOfStorageFilter(This) ) 

#define ID3D12InfoQueue1_PushStorageFilter(This,pFilter)	\
    ( (This)->lpVtbl -> PushStorageFilter(This,pFilter) ) 

#define ID3D12InfoQueue1_PopStorageFilter(This)	\
    ( (This)->lpVtbl -> PopStorageFilter(This) ) 

#define ID3D12InfoQueue1_GetStorageFilterStackSize(This)	\
    ( (This)->lpVtbl -> GetStorageFilterStackSize(This) ) 

#define ID3D12InfoQueue1_AddRetrievalFilterEntries(This,pFilter)	\
    ( (This)->lpVtbl -> AddRetrievalFilterEntries(This,pFilter) ) 

#define ID3D12InfoQueue1_GetRetrievalFilter(This,pFilter,pFilterByteLength)	\
    ( (This)->lpVtbl -> GetRetrievalFilter(This,pFilter,pFilterByteLength) ) 

#define ID3D12InfoQueue1_ClearRetrievalFilter(This)	\
    ( (This)->lpVtbl -> ClearRetrievalFilter(This) ) 

#define ID3D12InfoQueue1_PushEmptyRetrievalFilter(This)	\
    ( (This)->lpVtbl -> PushEmptyRetrievalFilter(This) ) 

#define ID3D12InfoQueue1_PushCopyOfRetrievalFilter(This)	\
    ( (This)->lpVtbl -> PushCopyOfRetrievalFilter(This) ) 

#define ID3D12InfoQueue1_PushRetrievalFilter(This,pFilter)	\
    ( (This)->lpVtbl -> PushRetrievalFilter(This,pFilter) ) 

#define ID3D12InfoQueue1_PopRetrievalFilter(This)	\
    ( (This)->lpVtbl -> PopRetrievalFilter(This) ) 

#define ID3D12InfoQueue1_GetRetrievalFilterStackSize(This)	\
    ( (This)->lpVtbl -> GetRetrievalFilterStackSize(This) ) 

#define ID3D12InfoQueue1_AddMessage(This,Category,Severity,ID,pDescription)	\
    ( (This)->lpVtbl -> AddMessage(This,Category,Severity,ID,pDescription) ) 

#define ID3D12InfoQueue1_AddApplicationMessage(This,Severity,pDescription)	\
    ( (This)->lpVtbl -> AddApplicationMessage(This,Severity,pDescription) ) 

#define ID3D12InfoQueue1_SetBreakOnCategory(This,Category,bEnable)	\
    ( (This)->lpVtbl -> SetBreakOnCategory(This,Category,bEnable) ) 

#define ID3D12InfoQueue1_SetBreakOnSeverity(This,Severity,bEnable)	\
    ( (This)->lpVtbl -> SetBreakOnSeverity(This,Severity,bEnable) ) 

#define ID3D12InfoQueue1_SetBreakOnID(This,ID,bEnable)	\
    ( (This)->lpVtbl -> SetBreakOnID(This,ID,bEnable) ) 

#define ID3D12InfoQueue1_GetBreakOnCategory(This,Category)	\
    ( (This)->lpVtbl -> GetBreakOnCategory(This,Category) ) 

#define ID3D12InfoQueue1_GetBreakOnSeverity(This,Severity)	\
    ( (This)->lpVtbl -> GetBreakOnSeverity(This,Severity) ) 

#define ID3D12InfoQueue1_GetBreakOnID(This,ID)	\
    ( (This)->lpVtbl -> GetBreakOnID(This,ID) ) 

#define ID3D12InfoQueue1_SetMuteDebugOutput(This,bMute)	\
    ( (This)->lpVtbl -> SetMuteDebugOutput(This,bMute) ) 

#define ID3D12InfoQueue1_GetMuteDebugOutput(This)	\
    ( (This)->lpVtbl -> GetMuteDebugOutput(This) ) 


#define ID3D12InfoQueue1_RegisterMessageCallback(This,CallbackFunc,CallbackFilterFlags,pContext,pCallbackCookie)	\
    ( (This)->lpVtbl -> RegisterMessageCallback(This,CallbackFunc,CallbackFilterFlags,pContext,pCallbackCookie) ) 

#define ID3D12InfoQueue1_UnregisterMessageCallback(This,CallbackCookie)	\
    ( (This)->lpVtbl -> UnregisterMessageCallback(This,CallbackCookie) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12InfoQueue1_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12sdklayers_0000_0020 */
/* [local] */ 

#endif /* WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP | WINAPI_PARTITION_GAMES) */
#ifdef _MSC_VER
#pragma endregion
#endif
DEFINE_GUID(IID_ID3D12Debug,0x344488b7,0x6846,0x474b,0xb9,0x89,0xf0,0x27,0x44,0x82,0x45,0xe0);
DEFINE_GUID(IID_ID3D12Debug1,0xaffaa4ca,0x63fe,0x4d8e,0xb8,0xad,0x15,0x90,0x00,0xaf,0x43,0x04);
DEFINE_GUID(IID_ID3D12Debug2,0x93a665c4,0xa3b2,0x4e5d,0xb6,0x92,0xa2,0x6a,0xe1,0x4e,0x33,0x74);
DEFINE_GUID(IID_ID3D12Debug3,0x5cf4e58f,0xf671,0x4ff1,0xa5,0x42,0x36,0x86,0xe3,0xd1,0x53,0xd1);
DEFINE_GUID(IID_ID3D12Debug4,0x014b816e,0x9ec5,0x4a2f,0xa8,0x45,0xff,0xbe,0x44,0x1c,0xe1,0x3a);
DEFINE_GUID(IID_ID3D12Debug5,0x548d6b12,0x09fa,0x40e0,0x90,0x69,0x5d,0xcd,0x58,0x9a,0x52,0xc9);
DEFINE_GUID(IID_ID3D12Debug6,0x82a816d6,0x5d01,0x4157,0x97,0xd0,0x49,0x75,0x46,0x3f,0xd1,0xed);
DEFINE_GUID(IID_ID3D12DebugDevice1,0xa9b71770,0xd099,0x4a65,0xa6,0x98,0x3d,0xee,0x10,0x02,0x0f,0x88);
DEFINE_GUID(IID_ID3D12DebugDevice,0x3febd6dd,0x4973,0x4787,0x81,0x94,0xe4,0x5f,0x9e,0x28,0x92,0x3e);
DEFINE_GUID(IID_ID3D12DebugDevice2,0x60eccbc1,0x378d,0x4df1,0x89,0x4c,0xf8,0xac,0x5c,0xe4,0xd7,0xdd);
DEFINE_GUID(IID_ID3D12DebugCommandQueue,0x09e0bf36,0x54ac,0x484f,0x88,0x47,0x4b,0xae,0xea,0xb6,0x05,0x3a);
DEFINE_GUID(IID_ID3D12DebugCommandQueue1,0x16be35a2,0xbfd6,0x49f2,0xbc,0xae,0xea,0xae,0x4a,0xff,0x86,0x2d);
DEFINE_GUID(IID_ID3D12DebugCommandList1,0x102ca951,0x311b,0x4b01,0xb1,0x1f,0xec,0xb8,0x3e,0x06,0x1b,0x37);
DEFINE_GUID(IID_ID3D12DebugCommandList,0x09e0bf36,0x54ac,0x484f,0x88,0x47,0x4b,0xae,0xea,0xb6,0x05,0x3f);
DEFINE_GUID(IID_ID3D12DebugCommandList2,0xaeb575cf,0x4e06,0x48be,0xba,0x3b,0xc4,0x50,0xfc,0x96,0x65,0x2e);
DEFINE_GUID(IID_ID3D12DebugCommandList3,0x197d5e15,0x4d37,0x4d34,0xaf,0x78,0x72,0x4c,0xd7,0x0f,0xdb,0x1f);
DEFINE_GUID(IID_ID3D12SharingContract,0x0adf7d52,0x929c,0x4e61,0xad,0xdb,0xff,0xed,0x30,0xde,0x66,0xef);
DEFINE_GUID(IID_ID3D12ManualWriteTrackingResource,0x86ca3b85,0x49ad,0x4b6e,0xae,0xd5,0xed,0xdb,0x18,0x54,0x0f,0x41);
DEFINE_GUID(IID_ID3D12InfoQueue,0x0742a90b,0xc387,0x483f,0xb9,0x46,0x30,0xa7,0xe4,0xe6,0x14,0x58);
DEFINE_GUID(IID_ID3D12InfoQueue1,0x2852dd88,0xb484,0x4c0c,0xb6,0xb1,0x67,0x16,0x85,0x00,0xe6,0x00);


extern RPC_IF_HANDLE __MIDL_itf_d3d12sdklayers_0000_0020_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12sdklayers_0000_0020_v0_0_s_ifspec;

/* Additional Prototypes for ALL interfaces */

/* end of Additional Prototypes */

#ifdef __cplusplus
}
#endif

#endif


