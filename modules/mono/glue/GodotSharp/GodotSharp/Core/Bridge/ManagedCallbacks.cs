using System;
using System.Runtime.InteropServices;
using Godot.NativeInterop;

namespace Godot.Bridge
{
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct ManagedCallbacks
    {
        // @formatter:off
        public delegate* unmanaged<IntPtr, godot_variant**, int, godot_bool*, void> SignalAwaiter_SignalCallback;
        public delegate* unmanaged<IntPtr, godot_variant**, uint, godot_variant*, void> DelegateUtils_InvokeWithVariantArgs;
        public delegate* unmanaged<IntPtr, IntPtr, godot_bool> DelegateUtils_DelegateEquals;
        public delegate* unmanaged<void> ScriptManagerBridge_FrameCallback;
        public delegate* unmanaged<godot_string_name*, IntPtr, IntPtr> ScriptManagerBridge_CreateManagedForGodotObjectBinding;
        public delegate* unmanaged<IntPtr, IntPtr, godot_variant**, int, godot_bool> ScriptManagerBridge_CreateManagedForGodotObjectScriptInstance;
        public delegate* unmanaged<IntPtr, godot_string_name*, void> ScriptManagerBridge_GetScriptNativeName;
        public delegate* unmanaged<IntPtr, IntPtr, void> ScriptManagerBridge_SetGodotObjectPtr;
        public delegate* unmanaged<IntPtr, godot_string_name*, godot_variant**, int, godot_bool*, void> ScriptManagerBridge_RaiseEventSignal;
        public delegate* unmanaged<IntPtr, godot_dictionary*, void> ScriptManagerBridge_GetScriptSignalList;
        public delegate* unmanaged<IntPtr, godot_string*, godot_bool> ScriptManagerBridge_HasScriptSignal;
        public delegate* unmanaged<IntPtr, IntPtr, godot_bool> ScriptManagerBridge_ScriptIsOrInherits;
        public delegate* unmanaged<IntPtr, godot_string*, godot_bool> ScriptManagerBridge_AddScriptBridge;
        public delegate* unmanaged<IntPtr, void> ScriptManagerBridge_RemoveScriptBridge;
        public delegate* unmanaged<IntPtr, godot_bool*, godot_dictionary*, void> ScriptManagerBridge_UpdateScriptClassInfo;
        public delegate* unmanaged<IntPtr, IntPtr*, godot_bool, godot_bool> ScriptManagerBridge_SwapGCHandleForType;
        public delegate* unmanaged<IntPtr, godot_string_name*, godot_variant**, int, godot_variant_call_error*, godot_variant*, godot_bool> CSharpInstanceBridge_Call;
        public delegate* unmanaged<IntPtr, godot_string_name*, godot_variant*, godot_bool> CSharpInstanceBridge_Set;
        public delegate* unmanaged<IntPtr, godot_string_name*, godot_variant*, godot_bool> CSharpInstanceBridge_Get;
        public delegate* unmanaged<IntPtr, godot_bool, void> CSharpInstanceBridge_CallDispose;
        public delegate* unmanaged<IntPtr, godot_string*, godot_bool*, void> CSharpInstanceBridge_CallToString;
        public delegate* unmanaged<IntPtr, godot_string_name*, godot_bool> CSharpInstanceBridge_HasMethodUnknownParams;
        public delegate* unmanaged<IntPtr, void> GCHandleBridge_FreeGCHandle;
        public delegate* unmanaged<void> DebuggingUtils_InstallTraceListener;
        public delegate* unmanaged<void> Dispatcher_InitializeDefaultGodotTaskScheduler;
        public delegate* unmanaged<void> DisposablesTracker_OnGodotShuttingDown;
        // @formatter:on

        public static ManagedCallbacks Create()
        {
            return new()
            {
                // @formatter:off
                SignalAwaiter_SignalCallback = &SignalAwaiter.SignalCallback,
                DelegateUtils_InvokeWithVariantArgs = &DelegateUtils.InvokeWithVariantArgs,
                DelegateUtils_DelegateEquals = &DelegateUtils.DelegateEquals,
                ScriptManagerBridge_FrameCallback = &ScriptManagerBridge.FrameCallback,
                ScriptManagerBridge_CreateManagedForGodotObjectBinding = &ScriptManagerBridge.CreateManagedForGodotObjectBinding,
                ScriptManagerBridge_CreateManagedForGodotObjectScriptInstance = &ScriptManagerBridge.CreateManagedForGodotObjectScriptInstance,
                ScriptManagerBridge_GetScriptNativeName = &ScriptManagerBridge.GetScriptNativeName,
                ScriptManagerBridge_SetGodotObjectPtr = &ScriptManagerBridge.SetGodotObjectPtr,
                ScriptManagerBridge_RaiseEventSignal = &ScriptManagerBridge.RaiseEventSignal,
                ScriptManagerBridge_GetScriptSignalList = &ScriptManagerBridge.GetScriptSignalList,
                ScriptManagerBridge_HasScriptSignal = &ScriptManagerBridge.HasScriptSignal,
                ScriptManagerBridge_ScriptIsOrInherits = &ScriptManagerBridge.ScriptIsOrInherits,
                ScriptManagerBridge_AddScriptBridge = &ScriptManagerBridge.AddScriptBridge,
                ScriptManagerBridge_RemoveScriptBridge = &ScriptManagerBridge.RemoveScriptBridge,
                ScriptManagerBridge_UpdateScriptClassInfo = &ScriptManagerBridge.UpdateScriptClassInfo,
                ScriptManagerBridge_SwapGCHandleForType = &ScriptManagerBridge.SwapGCHandleForType,
                CSharpInstanceBridge_Call = &CSharpInstanceBridge.Call,
                CSharpInstanceBridge_Set = &CSharpInstanceBridge.Set,
                CSharpInstanceBridge_Get = &CSharpInstanceBridge.Get,
                CSharpInstanceBridge_CallDispose = &CSharpInstanceBridge.CallDispose,
                CSharpInstanceBridge_CallToString = &CSharpInstanceBridge.CallToString,
                CSharpInstanceBridge_HasMethodUnknownParams = &CSharpInstanceBridge.HasMethodUnknownParams,
                GCHandleBridge_FreeGCHandle = &GCHandleBridge.FreeGCHandle,
                DebuggingUtils_InstallTraceListener = &DebuggingUtils.InstallTraceListener,
                Dispatcher_InitializeDefaultGodotTaskScheduler = &Dispatcher.InitializeDefaultGodotTaskScheduler,
                DisposablesTracker_OnGodotShuttingDown = &DisposablesTracker.OnGodotShuttingDown,
                // @formatter:on
            };
        }

        public static void Create(IntPtr outManagedCallbacks)
            => *(ManagedCallbacks*)outManagedCallbacks = Create();
    }
}
