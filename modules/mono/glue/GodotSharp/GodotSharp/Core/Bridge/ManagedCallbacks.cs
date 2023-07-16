using System;
using System.Runtime.InteropServices;
using Godot.NativeInterop;

namespace Godot.Bridge
{
    /// <summary>
    /// Collection of managed callbacks to handle a dotnet hot-reload environment.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct ManagedCallbacks
    {
        // @formatter:off
        /// <summary>
        /// Delegate for <see cref="SignalAwaiter.SignalCallback"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, godot_variant**, int, godot_bool*, void> SignalAwaiter_SignalCallback;
        /// <summary>
        /// Delegate for <see cref="DelegateUtils.InvokeWithVariantArgs"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, void*, godot_variant**, int, godot_variant*, void> DelegateUtils_InvokeWithVariantArgs;
        /// <summary>
        /// Delegate for <see cref="DelegateUtils.DelegateEquals"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, IntPtr, godot_bool> DelegateUtils_DelegateEquals;
        /// <summary>
        /// Delegate for <see cref="DelegateUtils.DelegateHash"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, int> DelegateUtils_DelegateHash;
        /// <summary>
        /// Delegate for <see cref="DelegateUtils.TrySerializeDelegateWithGCHandle"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, godot_array*, godot_bool> DelegateUtils_TrySerializeDelegateWithGCHandle;
        /// <summary>
        /// Delegate for <see cref="DelegateUtils.TryDeserializeDelegateWithGCHandle"/>
        /// </summary>
        public delegate* unmanaged<godot_array*, IntPtr*, godot_bool> DelegateUtils_TryDeserializeDelegateWithGCHandle;
        /// <summary>
        /// Delegate for <see cref="ScriptManagerBridge.FrameCallback"/>
        /// </summary>
        public delegate* unmanaged<void> ScriptManagerBridge_FrameCallback;
        /// <summary>
        /// Delegate for <see cref="ScriptManagerBridge.CreateManagedForGodotObjectBinding"/>
        /// </summary>
        public delegate* unmanaged<godot_string_name*, IntPtr, IntPtr> ScriptManagerBridge_CreateManagedForGodotObjectBinding;
        /// <summary>
        /// Delegate for <see cref="ScriptManagerBridge.CreateManagedForGodotObjectScriptInstance"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, IntPtr, godot_variant**, int, godot_bool> ScriptManagerBridge_CreateManagedForGodotObjectScriptInstance;
        /// <summary>
        /// Delegate for <see cref="ScriptManagerBridge.GetScriptNativeName"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, godot_string_name*, void> ScriptManagerBridge_GetScriptNativeName;
        /// <summary>
        /// Delegate for <see cref="ScriptManagerBridge.SetGodotObjectPtr"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, IntPtr, void> ScriptManagerBridge_SetGodotObjectPtr;
        /// <summary>
        /// Delegate for <see cref="ScriptManagerBridge.RaiseEventSignal"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, godot_string_name*, godot_variant**, int, godot_bool*, void> ScriptManagerBridge_RaiseEventSignal;
        /// <summary>
        /// Delegate for <see cref="ScriptManagerBridge.ScriptIsOrInherits"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, IntPtr, godot_bool> ScriptManagerBridge_ScriptIsOrInherits;
        /// <summary>
        /// Delegate for <see cref="ScriptManagerBridge.AddScriptBridge"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, godot_string*, godot_bool> ScriptManagerBridge_AddScriptBridge;
        /// <summary>
        /// Delegate for <see cref="ScriptManagerBridge.GetOrCreateScriptBridgeForPath"/>
        /// </summary>
        public delegate* unmanaged<godot_string*, godot_ref*, void> ScriptManagerBridge_GetOrCreateScriptBridgeForPath;
        /// <summary>
        /// Delegate for <see cref="ScriptManagerBridge.RemoveScriptBridge"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, void> ScriptManagerBridge_RemoveScriptBridge;
        /// <summary>
        /// Delegate for <see cref="ScriptManagerBridge.TryReloadRegisteredScriptWithClass"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, godot_bool> ScriptManagerBridge_TryReloadRegisteredScriptWithClass;
        /// <summary>
        /// Delegate for <see cref="ScriptManagerBridge.UpdateScriptClassInfo"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, godot_string*, godot_bool*, godot_bool*, godot_string*, godot_array*, godot_dictionary*, godot_dictionary*, godot_ref*, void> ScriptManagerBridge_UpdateScriptClassInfo;
        /// <summary>
        /// Delegate for <see cref="ScriptManagerBridge.SwapGCHandleForType"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, IntPtr*, godot_bool, godot_bool> ScriptManagerBridge_SwapGCHandleForType;
        /// <summary>
        /// Delegate for <see cref="ScriptManagerBridge.GetPropertyInfoList"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, delegate* unmanaged<IntPtr, godot_string*, void*, int, void>, void> ScriptManagerBridge_GetPropertyInfoList;
        /// <summary>
        /// Delegate for <see cref="ScriptManagerBridge.GetPropertyDefaultValues"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, delegate* unmanaged<IntPtr, void*, int, void>, void> ScriptManagerBridge_GetPropertyDefaultValues;
        /// <summary>
        /// Delegate for <see cref="CSharpInstanceBridge.Call"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, godot_string_name*, godot_variant**, int, godot_variant_call_error*, godot_variant*, godot_bool> CSharpInstanceBridge_Call;
        /// <summary>
        /// Delegate for <see cref="CSharpInstanceBridge.Set"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, godot_string_name*, godot_variant*, godot_bool> CSharpInstanceBridge_Set;
        /// <summary>
        /// Delegate for <see cref="CSharpInstanceBridge.Get"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, godot_string_name*, godot_variant*, godot_bool> CSharpInstanceBridge_Get;
        /// <summary>
        /// Delegate for <see cref="CSharpInstanceBridge.CallDispose"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, godot_bool, void> CSharpInstanceBridge_CallDispose;
        /// <summary>
        /// Delegate for <see cref="CSharpInstanceBridge.CallToString"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, godot_string*, godot_bool*, void> CSharpInstanceBridge_CallToString;
        /// <summary>
        /// Delegate for <see cref="CSharpInstanceBridge.HasMethodUnknownParams"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, godot_string_name*, godot_bool> CSharpInstanceBridge_HasMethodUnknownParams;
        /// <summary>
        /// Delegate for <see cref="CSharpInstanceBridge.SerializeState"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, godot_dictionary*, godot_dictionary*, void> CSharpInstanceBridge_SerializeState;
        /// <summary>
        /// Delegate for <see cref="CSharpInstanceBridge.DeserializeState"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, godot_dictionary*, godot_dictionary*, void> CSharpInstanceBridge_DeserializeState;
        /// <summary>
        /// Delegate for <see cref="GCHandleBridge.FreeGCHandle"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, void> GCHandleBridge_FreeGCHandle;
        /// <summary>
        /// Delegate for <see cref="GCHandleBridge.GCHandleIsTargetCollectible"/>
        /// </summary>
        public delegate* unmanaged<IntPtr, godot_bool> GCHandleBridge_GCHandleIsTargetCollectible;
        /// <summary>
        /// Delegate for <see cref="DebuggingUtils.GetCurrentStackInfo"/>
        /// </summary>
        public delegate* unmanaged<void*, void> DebuggingUtils_GetCurrentStackInfo;
        /// <summary>
        /// Delegate for <see cref="DisposablesTracker.OnGodotShuttingDown"/>
        /// </summary>
        public delegate* unmanaged<void> DisposablesTracker_OnGodotShuttingDown;
        /// <summary>
        /// Delegate for <see cref="GD.OnCoreApiAssemblyLoaded"/>
        /// </summary>
        public delegate* unmanaged<godot_bool, void> GD_OnCoreApiAssemblyLoaded;
        // @formatter:on

        /// <summary>
        /// Initializes various delegates for managed callbacks.
        /// </summary>
        /// <returns>The initialized delegates.</returns>
        public static ManagedCallbacks Create()
        {
            return new()
            {
                // @formatter:off
                SignalAwaiter_SignalCallback = &SignalAwaiter.SignalCallback,
                DelegateUtils_InvokeWithVariantArgs = &DelegateUtils.InvokeWithVariantArgs,
                DelegateUtils_DelegateEquals = &DelegateUtils.DelegateEquals,
                DelegateUtils_DelegateHash = &DelegateUtils.DelegateHash,
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
                GCHandleBridge_GCHandleIsTargetCollectible = &GCHandleBridge.GCHandleIsTargetCollectible,
                DebuggingUtils_GetCurrentStackInfo = &DebuggingUtils.GetCurrentStackInfo,
                DisposablesTracker_OnGodotShuttingDown = &DisposablesTracker.OnGodotShuttingDown,
                GD_OnCoreApiAssemblyLoaded = &GD.OnCoreApiAssemblyLoaded,
                // @formatter:on
            };
        }

        /// <summary>
        /// Initializes various delegates for managed callbacks.
        /// </summary>
        /// <param name="outManagedCallbacks">The initialized delegates.</param>
        public static void Create(IntPtr outManagedCallbacks)
            => *(ManagedCallbacks*)outManagedCallbacks = Create();
    }
}
