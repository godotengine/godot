using System;
using System.Runtime.InteropServices;
using Godot.NativeInterop;

namespace Godot.Bridge
{
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct ManagedCallbacks
    {
        // @formatter:off
        public delegate* unmanaged<nint, godot_variant**, int, godot_bool*, void> SignalAwaiter_SignalCallback;
        public delegate* unmanaged<nint, void*, godot_variant**, int, godot_variant*, void> DelegateUtils_InvokeWithVariantArgs;
        public delegate* unmanaged<nint, nint, godot_bool> DelegateUtils_DelegateEquals;
        public delegate* unmanaged<nint, godot_array*, godot_bool> DelegateUtils_TrySerializeDelegateWithGCHandle;
        public delegate* unmanaged<godot_array*, nint*, godot_bool> DelegateUtils_TryDeserializeDelegateWithGCHandle;
        public delegate* unmanaged<void> ScriptManagerBridge_FrameCallback;
        public delegate* unmanaged<godot_string_name*, nint, nint> ScriptManagerBridge_CreateManagedForGodotObjectBinding;
        public delegate* unmanaged<nint, nint, godot_variant**, int, godot_bool> ScriptManagerBridge_CreateManagedForGodotObjectScriptInstance;
        public delegate* unmanaged<nint, godot_string_name*, void> ScriptManagerBridge_GetScriptNativeName;
        public delegate* unmanaged<nint, nint, void> ScriptManagerBridge_SetGodotObjectPtr;
        public delegate* unmanaged<nint, godot_string_name*, godot_variant**, int, godot_bool*, void> ScriptManagerBridge_RaiseEventSignal;
        public delegate* unmanaged<nint, nint, godot_bool> ScriptManagerBridge_ScriptIsOrInherits;
        public delegate* unmanaged<nint, godot_string*, godot_bool> ScriptManagerBridge_AddScriptBridge;
        public delegate* unmanaged<godot_string*, godot_ref*, void> ScriptManagerBridge_GetOrCreateScriptBridgeForPath;
        public delegate* unmanaged<nint, void> ScriptManagerBridge_RemoveScriptBridge;
        public delegate* unmanaged<nint, godot_bool> ScriptManagerBridge_TryReloadRegisteredScriptWithClass;
        public delegate* unmanaged<nint, godot_bool*, godot_array*, godot_dictionary*, godot_dictionary*, godot_ref*, void> ScriptManagerBridge_UpdateScriptClassInfo;
        public delegate* unmanaged<nint, nint*, godot_bool, godot_bool> ScriptManagerBridge_SwapGCHandleForType;
        public delegate* unmanaged<nint, delegate* unmanaged<nint, godot_string*, void*, int, void>, void> ScriptManagerBridge_GetPropertyInfoList;
        public delegate* unmanaged<nint, delegate* unmanaged<nint, void*, int, void>, void> ScriptManagerBridge_GetPropertyDefaultValues;
        public delegate* unmanaged<nint, godot_string_name*, godot_variant**, int, godot_variant_call_error*, godot_variant*, godot_bool> CSharpInstanceBridge_Call;
        public delegate* unmanaged<nint, godot_string_name*, godot_variant*, godot_bool> CSharpInstanceBridge_Set;
        public delegate* unmanaged<nint, godot_string_name*, godot_variant*, godot_bool> CSharpInstanceBridge_Get;
        public delegate* unmanaged<nint, godot_bool, void> CSharpInstanceBridge_CallDispose;
        public delegate* unmanaged<nint, godot_string*, godot_bool*, void> CSharpInstanceBridge_CallToString;
        public delegate* unmanaged<nint, godot_string_name*, godot_bool> CSharpInstanceBridge_HasMethodUnknownParams;
        public delegate* unmanaged<nint, godot_dictionary*, godot_dictionary*, void> CSharpInstanceBridge_SerializeState;
        public delegate* unmanaged<nint, godot_dictionary*, godot_dictionary*, void> CSharpInstanceBridge_DeserializeState;
        public delegate* unmanaged<nint, void> GCHandleBridge_FreeGCHandle;
        public delegate* unmanaged<void*, void> DebuggingUtils_GetCurrentStackInfo;
        public delegate* unmanaged<void> DisposablesTracker_OnGodotShuttingDown;
        public delegate* unmanaged<godot_bool, void> GD_OnCoreApiAssemblyLoaded;
        // @formatter:on

        public static ManagedCallbacks Create()
        {
            return new()
            {
                // @formatter:off
                SignalAwaiter_SignalCallback = &SignalAwaiter.SignalCallback,
                DelegateUtils_InvokeWithVariantArgs = &DelegateUtils.InvokeWithVariantArgs,
                DelegateUtils_DelegateEquals = &DelegateUtils.DelegateEquals,
                DelegateUtils_TrySerializeDelegateWithGCHandle = &DelegateUtils.TrySerializeDelegateWithGCHandle,
                DelegateUtils_TryDeserializeDelegateWithGCHandle = &DelegateUtils.TryDeserializeDelegateWithGCHandle,
                ScriptManagerBridge_FrameCallback = &ScriptManagerBridge.FrameCallback,
                ScriptManagerBridge_CreateManagedForGodotObjectBinding = &ScriptManagerBridge.CreateManagedForGodotObjectBinding,
                ScriptManagerBridge_CreateManagedForGodotObjectScriptInstance = &ScriptManagerBridge.CreateManagedForGodotObjectScriptInstance,
                ScriptManagerBridge_GetScriptNativeName = &ScriptManagerBridge.GetScriptNativeName,
                ScriptManagerBridge_SetGodotObjectPtr = &ScriptManagerBridge.SetGodotObjectPtr,
                ScriptManagerBridge_RaiseEventSignal = &ScriptManagerBridge.RaiseEventSignal,
                ScriptManagerBridge_ScriptIsOrInherits = &ScriptManagerBridge.ScriptIsOrInherits,
                ScriptManagerBridge_AddScriptBridge = &ScriptManagerBridge.AddScriptBridge,
                ScriptManagerBridge_GetOrCreateScriptBridgeForPath = &ScriptManagerBridge.GetOrCreateScriptBridgeForPath,
                ScriptManagerBridge_RemoveScriptBridge = &ScriptManagerBridge.RemoveScriptBridge,
                ScriptManagerBridge_TryReloadRegisteredScriptWithClass = &ScriptManagerBridge.TryReloadRegisteredScriptWithClass,
                ScriptManagerBridge_UpdateScriptClassInfo = &ScriptManagerBridge.UpdateScriptClassInfo,
                ScriptManagerBridge_SwapGCHandleForType = &ScriptManagerBridge.SwapGCHandleForType,
                ScriptManagerBridge_GetPropertyInfoList = &ScriptManagerBridge.GetPropertyInfoList,
                ScriptManagerBridge_GetPropertyDefaultValues = &ScriptManagerBridge.GetPropertyDefaultValues,
                CSharpInstanceBridge_Call = &CSharpInstanceBridge.Call,
                CSharpInstanceBridge_Set = &CSharpInstanceBridge.Set,
                CSharpInstanceBridge_Get = &CSharpInstanceBridge.Get,
                CSharpInstanceBridge_CallDispose = &CSharpInstanceBridge.CallDispose,
                CSharpInstanceBridge_CallToString = &CSharpInstanceBridge.CallToString,
                CSharpInstanceBridge_HasMethodUnknownParams = &CSharpInstanceBridge.HasMethodUnknownParams,
                CSharpInstanceBridge_SerializeState = &CSharpInstanceBridge.SerializeState,
                CSharpInstanceBridge_DeserializeState = &CSharpInstanceBridge.DeserializeState,
                GCHandleBridge_FreeGCHandle = &GCHandleBridge.FreeGCHandle,
                DebuggingUtils_GetCurrentStackInfo = &DebuggingUtils.GetCurrentStackInfo,
                DisposablesTracker_OnGodotShuttingDown = &DisposablesTracker.OnGodotShuttingDown,
                GD_OnCoreApiAssemblyLoaded = &GD.OnCoreApiAssemblyLoaded,
                // @formatter:on
            };
        }

        public static void Create(nint outManagedCallbacks)
            => *(ManagedCallbacks*)outManagedCallbacks = Create();
    }
}
