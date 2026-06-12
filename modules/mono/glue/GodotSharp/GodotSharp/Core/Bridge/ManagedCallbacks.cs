using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Godot.NativeInterop;
using JetBrains.Annotations;

namespace Godot.Bridge
{
    [StructLayout(LayoutKind.Sequential)]
    file unsafe struct ManagedCallbacksInitContext(
        ManagedCallbacks* managedCallbacks,
        ToolsBuildManagedCallbacks* toolsBuildManagedCallbacks)
    {
        public ManagedCallbacks* ManagedCallbacks = managedCallbacks;
        public ToolsBuildManagedCallbacks* ToolsBuildManagedCallbacks = toolsBuildManagedCallbacks;
    }

    [StructLayout(LayoutKind.Sequential)]
    file unsafe struct ToolsBuildManagedCallbacks
    {
        public delegate* unmanaged<IntPtr, godot_array*, godot_bool> DelegateUtils_TrySerializeDelegateWithGCHandle;
        public delegate* unmanaged<godot_array*, IntPtr*, godot_bool> DelegateUtils_TryDeserializeDelegateWithGCHandle;
        public delegate* unmanaged<IntPtr, godot_bool> ScriptManagerBridge_TryReloadRegisteredScriptWithClass;

        public delegate* unmanaged<IntPtr, godot_dictionary*, godot_dictionary*, void>
            CSharpInstanceBridge_SerializeState;

        public delegate* unmanaged<IntPtr, godot_dictionary*, godot_dictionary*, void>
            CSharpInstanceBridge_DeserializeState;

        [RequiresUnreferencedCode(
            "This method is for internal use by the Godot editor only. "
            + "It returns delegates that point to methods that are not compatible with trimming.")]
        public static ToolsBuildManagedCallbacks Create()
        {
            return new()
            {
                DelegateUtils_TrySerializeDelegateWithGCHandle =
                    DelegateUtils.ToolsBuildUnmanagedCallables.GetAddressOfTrySerializeDelegateWithGCHandle(),
                DelegateUtils_TryDeserializeDelegateWithGCHandle =
                    DelegateUtils.ToolsBuildUnmanagedCallables.GetAddressOfTryDeserializeDelegateWithGCHandle(),
                ScriptManagerBridge_TryReloadRegisteredScriptWithClass =
                    ScriptManagerBridge.ToolsBuildUnmanagedCallables.GetAddressOfTryReloadRegisteredScriptWithClass(),
                CSharpInstanceBridge_SerializeState =
                    CSharpInstanceBridge.ToolsBuildUnmanagedCallables.GetAddressOfSerializeState(),
                CSharpInstanceBridge_DeserializeState =
                    CSharpInstanceBridge.ToolsBuildUnmanagedCallables.GetAddressOfDeserializeState(),
            };
        }
    }

    /// <summary>
    /// Represents a collection of unmanaged function pointers that serve
    /// as callbacks for various operations in the Godot C# integration.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct ManagedCallbacks
    {
        // @formatter:off
#pragma warning disable CS1591 // Missing XML comment for publicly visible type or member
        public delegate* unmanaged<IntPtr, godot_variant**, int, godot_bool*, void> SignalAwaiter_SignalCallback;
        public delegate* unmanaged<IntPtr, void*, godot_variant**, int, godot_variant*, void> DelegateUtils_InvokeWithVariantArgs;
        public delegate* unmanaged<IntPtr, IntPtr, godot_bool> DelegateUtils_DelegateEquals;
        public delegate* unmanaged<IntPtr, int> DelegateUtils_DelegateHash;
        public delegate* unmanaged<IntPtr, godot_bool*, int> DelegateUtils_GetArgumentCount;
        public delegate* unmanaged<void> ScriptManagerBridge_FrameCallback;
        public delegate* unmanaged<godot_string_name*, IntPtr, IntPtr> ScriptManagerBridge_CreateManagedForGodotObjectBinding;
        public delegate* unmanaged<IntPtr, IntPtr, godot_variant**, int, godot_bool> ScriptManagerBridge_LegacyCreateManagedForGodotObjectScriptInstance;
        public delegate* unmanaged<ConstructorTrampolineDelegate, IntPtr, godot_variant**, int, godot_bool> ScriptManagerBridge_CreateManagedForGodotObjectScriptInstanceWithTrampoline;
        public delegate* unmanaged<godot_string*, godot_string*, godot_string*, godot_bool*, godot_bool*, godot_string*, void> ScriptManagerBridge_GetGlobalClassName;
        public delegate* unmanaged<IntPtr, IntPtr, void> ScriptManagerBridge_SetGodotObjectPtr;
        public delegate* unmanaged<IntPtr, godot_string_name*, godot_variant**, int, godot_bool*, void> ScriptManagerBridge_LegacyRaiseEventSignal;
        public delegate* unmanaged<RaiseSignalTrampolineDelegate, IntPtr, godot_variant**, int, godot_bool*, void> ScriptManagerBridge_RaiseEventSignalViaTrampoline;
        public delegate* unmanaged<IntPtr, IntPtr, godot_bool> ScriptManagerBridge_ScriptIsOrInherits;
        public delegate* unmanaged<IntPtr, godot_string*, godot_bool> ScriptManagerBridge_AddScriptBridge;
        public delegate* unmanaged<godot_string*, godot_ref*, void> ScriptManagerBridge_GetOrCreateScriptBridgeForPath;
        public delegate* unmanaged<IntPtr, void> ScriptManagerBridge_RemoveScriptBridge;
        public delegate* unmanaged<IntPtr, godot_bool*,
            delegate* unmanaged<IntPtr, int, void*, void>,
            delegate* unmanaged<IntPtr, godot_string_name*, int, void*, godot_bool, void>,
            delegate* unmanaged<IntPtr, godot_string_name*, void*, void*, void>,
            delegate* unmanaged<IntPtr, godot_string_name*, int, void*, void>,
            void> ScriptManagerBridge_UpdateScriptTrampolines;
        public delegate* unmanaged<IntPtr, godot_csharp_type_info*, godot_array*, godot_dictionary*, godot_dictionary*, godot_ref*, void> ScriptManagerBridge_UpdateScriptClassInfo;
        public delegate* unmanaged<IntPtr, IntPtr*, godot_bool, godot_bool> ScriptManagerBridge_SwapGCHandleForType;
        public delegate* unmanaged<IntPtr, delegate* unmanaged<IntPtr, godot_string*, void*, int, void>, void> ScriptManagerBridge_GetPropertyInfoList;
        public delegate* unmanaged<IntPtr, delegate* unmanaged<IntPtr, void*, int, void>, void> ScriptManagerBridge_GetPropertyDefaultValues;
        public delegate* unmanaged<IntPtr, godot_string_name*, godot_variant**, int, godot_variant_call_error*, godot_variant*, godot_bool> ScriptManagerBridge_LegacyCallStatic;
        public delegate* unmanaged<MethodTrampolineDelegate, godot_variant**, int, godot_variant_call_error*, godot_variant*, godot_bool> ScriptManagerBridge_CallStaticWithTrampoline;
        public delegate* unmanaged<IntPtr, godot_string_name*, godot_variant**, int, godot_variant_call_error*, godot_variant*, godot_bool> CSharpInstanceBridge_LegacyCall;
        public delegate* unmanaged<IntPtr, godot_string_name*, godot_variant*, godot_bool> CSharpInstanceBridge_LegacySet;
        public delegate* unmanaged<IntPtr, godot_string_name*, godot_variant*, godot_bool> CSharpInstanceBridge_LegacyGet;
        public delegate* unmanaged<MethodTrampolineDelegate, IntPtr, godot_variant**, int, godot_variant_call_error*, godot_variant*, godot_bool> CSharpInstanceBridge_CallViaTrampoline;
        public delegate* unmanaged<PropertySetterTrampolineDelegate, IntPtr, godot_variant*, godot_bool> CSharpInstanceBridge_SetViaTrampoline;
        public delegate* unmanaged<PropertyGetterTrampolineDelegate, IntPtr, godot_variant*, godot_bool> CSharpInstanceBridge_GetViaTrampoline;
        public delegate* unmanaged<IntPtr, godot_bool, void> CSharpInstanceBridge_CallDispose;
        public delegate* unmanaged<IntPtr, godot_string*, godot_bool*, void> CSharpInstanceBridge_CallToString;
        public delegate* unmanaged<IntPtr, godot_string_name*, godot_bool> CSharpInstanceBridge_LegacyHasMethodUnknownParams;
        public delegate* unmanaged<IntPtr, void> GCHandleBridge_FreeGCHandle;
        public delegate* unmanaged<IntPtr, godot_bool> GCHandleBridge_GCHandleIsTargetCollectible;
        public delegate* unmanaged<void*, void> DebuggingUtils_GetCurrentStackInfo;
        public delegate* unmanaged<void> DisposablesTracker_OnGodotShuttingDown;
        public delegate* unmanaged<godot_bool, void> GD_OnCoreApiAssemblyLoaded;
        // @formatter:on
#pragma warning restore CS1591 // Missing XML comment for publicly visible type or member

        private static ManagedCallbacks CreateWithoutLegacy()
        {
            return new()
            {
                // @formatter:off
                SignalAwaiter_SignalCallback = &SignalAwaiter.SignalCallback,
                DelegateUtils_InvokeWithVariantArgs = &DelegateUtils.InvokeWithVariantArgs,
                DelegateUtils_DelegateEquals = &DelegateUtils.DelegateEquals,
                DelegateUtils_DelegateHash = &DelegateUtils.DelegateHash,
                DelegateUtils_GetArgumentCount = &DelegateUtils.GetArgumentCount,
                ScriptManagerBridge_FrameCallback = &ScriptManagerBridge.FrameCallback,
                ScriptManagerBridge_CreateManagedForGodotObjectBinding = &ScriptManagerBridge.CreateManagedForGodotObjectBinding,
                ScriptManagerBridge_CreateManagedForGodotObjectScriptInstanceWithTrampoline = &ScriptManagerBridge.CreateManagedForGodotObjectScriptInstanceWithTrampoline,
                ScriptManagerBridge_GetGlobalClassName = &ScriptManagerBridge.GetGlobalClassName,
                ScriptManagerBridge_SetGodotObjectPtr = &ScriptManagerBridge.SetGodotObjectPtr,
                ScriptManagerBridge_RaiseEventSignalViaTrampoline = &ScriptManagerBridge.RaiseEventSignalViaTrampoline,
                ScriptManagerBridge_ScriptIsOrInherits = &ScriptManagerBridge.ScriptIsOrInherits,
                ScriptManagerBridge_AddScriptBridge = &ScriptManagerBridge.AddScriptBridge,
                ScriptManagerBridge_GetOrCreateScriptBridgeForPath = &ScriptManagerBridge.GetOrCreateScriptBridgeForPath,
                ScriptManagerBridge_RemoveScriptBridge = &ScriptManagerBridge.RemoveScriptBridge,
                ScriptManagerBridge_UpdateScriptTrampolines = &ScriptManagerBridge.UpdateScriptTrampolines,
                ScriptManagerBridge_UpdateScriptClassInfo = &ScriptManagerBridge.UpdateScriptClassInfo,
                ScriptManagerBridge_SwapGCHandleForType = &ScriptManagerBridge.SwapGCHandleForType,
                ScriptManagerBridge_GetPropertyInfoList = &ScriptManagerBridge.GetPropertyInfoList,
                ScriptManagerBridge_GetPropertyDefaultValues = &ScriptManagerBridge.GetPropertyDefaultValues,
                ScriptManagerBridge_CallStaticWithTrampoline = &ScriptManagerBridge.CallStaticWithTrampoline,
                CSharpInstanceBridge_CallViaTrampoline = &CSharpInstanceBridge.CallViaTrampoline,
                CSharpInstanceBridge_SetViaTrampoline = &CSharpInstanceBridge.SetViaTrampoline,
                CSharpInstanceBridge_GetViaTrampoline = &CSharpInstanceBridge.GetViaTrampoline,
                CSharpInstanceBridge_CallDispose = &CSharpInstanceBridge.CallDispose,
                CSharpInstanceBridge_CallToString = &CSharpInstanceBridge.CallToString,
                GCHandleBridge_FreeGCHandle = &GCHandleBridge.FreeGCHandle,
                GCHandleBridge_GCHandleIsTargetCollectible = &GCHandleBridge.GCHandleIsTargetCollectible,
                DebuggingUtils_GetCurrentStackInfo = &DebuggingUtils.GetCurrentStackInfo,
                DisposablesTracker_OnGodotShuttingDown = &DisposablesTracker.OnGodotShuttingDown,
                GD_OnCoreApiAssemblyLoaded = &GD.OnCoreApiAssemblyLoaded,
                // @formatter:on
            };
        }

        private static void AddLegacyCallbacks(ref ManagedCallbacks managedCallbacks)
        {
            // @formatter:off
            managedCallbacks.ScriptManagerBridge_LegacyCreateManagedForGodotObjectScriptInstance = &ScriptManagerBridge.LegacyCreateManagedForGodotObjectScriptInstance;
            managedCallbacks.ScriptManagerBridge_LegacyRaiseEventSignal = &ScriptManagerBridge.LegacyRaiseEventSignal;
            managedCallbacks.ScriptManagerBridge_LegacyCallStatic = &ScriptManagerBridge.LegacyCallStatic;
            managedCallbacks.CSharpInstanceBridge_LegacyCall = &CSharpInstanceBridge.LegacyCall;
            managedCallbacks.CSharpInstanceBridge_LegacySet = &CSharpInstanceBridge.LegacySet;
            managedCallbacks.CSharpInstanceBridge_LegacyGet = &CSharpInstanceBridge.LegacyGet;
            managedCallbacks.CSharpInstanceBridge_LegacyHasMethodUnknownParams = &CSharpInstanceBridge.LegacyHasMethodUnknownParams;
            // @formatter:on
        }

        /// <summary>
        /// Creates a new instance of <see cref="ManagedCallbacks"/> with all callbacks initialized to their respective methods.
        /// </summary>
        [PublicAPI("ABI compatibility with legacy code.")]
        [Obsolete("Use Create(IntPtr, bool) or CreateForToolsBuild(IntPtr).")]
        public static ManagedCallbacks Create()
        {
            var managedCallbacks = CreateWithoutLegacy();

            // Since Create() is a public API that's kept for compatibility with legacy code,
            // we must assume that the legacy callbacks are needed.
            AddLegacyCallbacks(ref managedCallbacks);

            return managedCallbacks;
        }

        /// <summary>
        /// Initializes the managed callbacks by writing them to the provided pointer, excluding legacy callbacks.
        /// Only use this method if you are certain that your project and all its dependencies do not require the legacy callbacks.
        /// </summary>
        /// <param name="outManagedCallbacks">
        /// A pointer to the location where the managed callbacks will be written.
        /// </param>
        /// <remarks>
        /// <para><paramref name="outManagedCallbacks"/> must be the pointer
        /// that is provided by Godot to the initialization function.</para>
        /// <para>When running from the Godot editor, the managed callbacks must be initialized
        /// using <see cref="CreateForToolsBuild"/> instead, which initializes both the regular
        /// managed callbacks and the tools build specific managed callbacks.</para>
        /// <para>This method excludes legacy callbacks, which are used for compatibility with assemblies
        /// built against older versions of the Godot C# integration. If your project does not have any
        /// dependencies that use or override legacy virtual methods, you can use this method to improve
        /// trimming and reduce the size of the resulting binary. However, if you are unsure, it is recommended
        /// to use <see cref="CreateIncludingLegacyCallbacks"/> instead to ensure compatibility.</para>
        /// <para>The legacy virtual methods that require the legacy callbacks are:
        /// <list type="bullet">
        ///     <item>
        ///         <description><see cref="Godot.GodotObject.InvokeGodotClassMethod"/></description>
        ///     </item>
        ///     <item>
        ///         <description><see cref="Godot.GodotObject.HasGodotClassMethod"/></description>
        ///     </item>
        ///     <item>
        ///         <description><see cref="Godot.GodotObject.HasGodotClassSignal"/></description>
        ///     </item>
        ///     <item>
        ///         <description><see cref="Godot.GodotObject.RaiseGodotClassSignalCallbacks"/></description>
        ///     </item>
        ///     <item>
        ///         <description><see cref="Godot.GodotObject.SetGodotClassPropertyValue"/></description>
        ///     </item>
        ///     <item>
        ///         <description><see cref="Godot.GodotObject.GetGodotClassPropertyValue"/></description>
        ///     </item>
        /// </list></para>
        /// </remarks>
        [PublicAPI]
        public static void CreateExcludingLegacyCallbacks(IntPtr outManagedCallbacks)
        {
            var initContext = *(ManagedCallbacksInitContext*)outManagedCallbacks;

            ArgumentNullException.ThrowIfNull(initContext.ManagedCallbacks);

            *initContext.ManagedCallbacks = CreateWithoutLegacy();
        }

        /// <summary>
        /// Initializes the managed callbacks by writing them to the provided pointer,
        /// including legacy callbacks for compatibility with older assemblies.
        /// </summary>
        /// <param name="outManagedCallbacks">
        /// A pointer to the location where the managed callbacks will be written.
        /// </param>
        /// <remarks>
        /// <para><paramref name="outManagedCallbacks"/> must be the pointer
        /// that is provided by Godot to the initialization function.</para>
        /// <para>When running from the Godot editor, the managed callbacks must be initialized
        /// using <see cref="CreateForToolsBuild"/> instead, which initializes both the regular
        /// managed callbacks and the tools build specific managed callbacks.</para>
        /// <para>>This method includes legacy callbacks, which are used for compatibility with assemblies
        /// built against older versions of the Godot C# integration. If your project has dependencies that
        /// use or override legacy virtual methods, you should use this method to ensure compatibility.
        /// However, if you are certain that neither your project nor any of its dependencies use or
        /// override legacy virtual methods, you can use <see cref="CreateExcludingLegacyCallbacks"/>
        /// instead to improve trimming and reduce the size of the resulting binary. If you are unsure,
        /// it is recommended to use this method to ensure compatibility.</para>
        /// <para>The legacy virtual methods that require the legacy callbacks are:
        /// <list type="bullet">
        ///     <item>
        ///         <description><see cref="Godot.GodotObject.InvokeGodotClassMethod"/></description>
        ///     </item>
        ///     <item>
        ///         <description><see cref="Godot.GodotObject.HasGodotClassMethod"/></description>
        ///     </item>
        ///     <item>
        ///         <description><see cref="Godot.GodotObject.HasGodotClassSignal"/></description>
        ///     </item>
        ///     <item>
        ///         <description><see cref="Godot.GodotObject.RaiseGodotClassSignalCallbacks"/></description>
        ///     </item>
        ///     <item>
        ///         <description><see cref="Godot.GodotObject.SetGodotClassPropertyValue"/></description>
        ///     </item>
        ///     <item>
        ///         <description><see cref="Godot.GodotObject.GetGodotClassPropertyValue"/></description>
        ///     </item>
        /// </list></para>
        /// </remarks>
        [PublicAPI]
        public static void CreateIncludingLegacyCallbacks(IntPtr outManagedCallbacks)
        {
            var initContext = *(ManagedCallbacksInitContext*)outManagedCallbacks;

            ArgumentNullException.ThrowIfNull(initContext.ManagedCallbacks);

            *initContext.ManagedCallbacks = CreateWithoutLegacy();

            AddLegacyCallbacks(ref Unsafe.AsRef<ManagedCallbacks>(initContext.ManagedCallbacks));
        }

        /// <summary>
        /// Initializes the managed callbacks for tools build by writing them to the provided pointer.
        /// </summary>
        /// <param name="outManagedCallbacks">A pointer to the location where the managed callbacks will be written.</param>
        /// <remarks>
        /// <para><paramref name="outManagedCallbacks"/> must be the pointer
        /// that is provided by Godot to the initialization function for tools build.</para>
        /// <para>Use this method when running from the Godot editor, as the editor requires
        /// additional callbacks for tools build. This method initializes both the regular
        /// managed callbacks and the tools build specific managed callbacks.</para>
        /// </remarks>
        [RequiresUnreferencedCode(
            "This method is for internal use by the Godot editor only. "
            + "It populates the output's delegates with methods that are not compatible with trimming.")]
        [PublicAPI]
        public static void CreateForToolsBuild(IntPtr outManagedCallbacks) // For the Godot editor
        {
            var initContext = *(ManagedCallbacksInitContext*)outManagedCallbacks;

            ArgumentNullException.ThrowIfNull(initContext.ManagedCallbacks);
            ArgumentNullException.ThrowIfNull(initContext.ToolsBuildManagedCallbacks);

            *initContext.ManagedCallbacks = CreateWithoutLegacy();
            *initContext.ToolsBuildManagedCallbacks = ToolsBuildManagedCallbacks.Create();

            // Tools build can load assemblies that use or override legacy virtual methods, so the legacy callbacks are necessary.
            AddLegacyCallbacks(ref Unsafe.AsRef<ManagedCallbacks>(initContext.ManagedCallbacks));
        }
    }
}
