using System;
using Godot.NativeInterop;
using JetBrains.Annotations;

namespace Godot.Bridge;

using PropertyTrampolines = (PropertyGetterTrampoline getterTramp, PropertySetterTrampoline setterTramp);

/// <summary>
/// This is used to collect the RPC methods for a script.
/// </summary>
/// <remarks>
/// The user script can implement a static method "GetGodotClassRpcMethods"
/// that receives an instance of this class. The user script can then call the
/// <see cref="TryAdd"/> method of this class to add RPC methods to the script.<br/>
/// </remarks>
[PublicAPI]
public class RpcMethodCollector
{
    internal Collections.Dictionary RpcMethodsDict { get; private set; }

    internal RpcMethodCollector()
    {
    }

    internal void Reset() => RpcMethodsDict = null;

    /// <summary>
    /// Whether RPC methods from ancestor classes should also be collected.
    /// </summary>
    /// <remarks>
    /// If true, the collection method of each ancestor class must be called
    /// after the collection method of the current class.
    /// </remarks>
    public bool IncludeAncestors { get; internal set; } = true;

    private static class RpcConfigKeys
    {
        private static readonly Collections.Array _rpcConfigKeyCache =
        [
            "rpc_mode",
            "call_local",
            "transfer_mode",
            "channel"
        ];

        private static godot_variant BorrowElementAt(int index)
        {
            _rpcConfigKeyCache.GetVariantBorrowElementAtUnchecked(index, out var elem);
            return elem;
        }

        /// <summary>
        /// The returned <see cref="Godot.NativeInterop.godot_variant"/> is borrowed. Do not free it.
        /// </summary>
        public static godot_variant RpcMode => BorrowElementAt(0);

        /// <summary>
        /// The returned <see cref="Godot.NativeInterop.godot_variant"/> is borrowed. Do not free it.
        /// </summary>
        public static godot_variant CallLocal => BorrowElementAt(1);

        /// <summary>
        /// The returned <see cref="Godot.NativeInterop.godot_variant"/> is borrowed. Do not free it.
        /// </summary>
        public static godot_variant TransferMode => BorrowElementAt(2);

        /// <summary>
        /// The returned <see cref="Godot.NativeInterop.godot_variant"/> is borrowed. Do not free it.
        /// </summary>
        public static godot_variant Channel => BorrowElementAt(3);
    }

    /// <summary>
    /// Adds a raise signal trampoline to the script if a trampoline for the given signal key doesn't already exist.
    /// </summary>
    public void TryAdd(StringName methodName, MultiplayerApi.RpcMode? rpcMode, bool? callLocal,
        MultiplayerPeer.TransferModeEnum? transferMode, int? transferChannel)
    {
        if (RpcMethodsDict?.ContainsKey(methodName) ?? false)
            return;

        RpcMethodsDict ??= new();

        // No-GC version of:
        // var rpcConfig = new Collections.Dictionary();
        // rpcConfig[...] = ...;
        // _rpcFunctionsDict.Add(methodName, rpcConfig);

        var rpcConfig = NativeFuncs.godotsharp_dictionary_new();
        ref var rpcConfigSelf = ref rpcConfig; // To avoid CS0728 warning.
        using (rpcConfig)
        {
            NativeFuncs.godotsharp_dictionary_set_value(ref rpcConfigSelf, RpcConfigKeys.RpcMode,
                VariantUtils.CreateFromInt((long)(rpcMode ?? MultiplayerApi.RpcMode.Authority)));
            NativeFuncs.godotsharp_dictionary_set_value(ref rpcConfigSelf, RpcConfigKeys.CallLocal,
                VariantUtils.CreateFromBool(callLocal ?? false));
            NativeFuncs.godotsharp_dictionary_set_value(ref rpcConfigSelf, RpcConfigKeys.TransferMode,
                VariantUtils.CreateFromInt((long)(transferMode ?? MultiplayerPeer.TransferModeEnum.Reliable)));
            NativeFuncs.godotsharp_dictionary_set_value(ref rpcConfigSelf, RpcConfigKeys.Channel,
                VariantUtils.CreateFromInt(transferChannel ?? 0));

            var rpcFunctionsSelf = (godot_dictionary)RpcMethodsDict.NativeValue;
            NativeFuncs.godotsharp_dictionary_set_value(ref rpcFunctionsSelf,
                VariantUtils.CreateFromStringName(methodName),
                VariantUtils.CreateFromDictionary(rpcConfig));
        }
    }
}

/// <summary>
/// This is used to collect the constructor trampolines for a script.
/// </summary>
/// <remarks>
/// The user script can implement a static method "GetGodotClassTrampolines"
/// that receives an instance of <see cref="TrampolineCollectors"/> which
/// contains an instance of this class. The user script can then call the
/// <see cref="TryAdd"/> method of this class to add constructors to the script.<br/>
/// </remarks>
[PublicAPI]
public class ConstructorTrampolineCollector
{
    private IntPtr _scriptPtr;
    private unsafe TryAddConstructorTrampolineDelegate _tryAddDelegate;

    internal unsafe ConstructorTrampolineCollector(IntPtr scriptPtr,
        TryAddConstructorTrampolineDelegate tryAddDelegate)
    {
        _scriptPtr = scriptPtr;
        _tryAddDelegate = tryAddDelegate;
    }

    internal unsafe void Update(IntPtr scriptPtr, TryAddConstructorTrampolineDelegate tryAddDelegate)
    {
        _scriptPtr = scriptPtr;
        _tryAddDelegate = tryAddDelegate;
    }

    /// <summary>
    /// Adds a method trampoline to the script if a trampoline for the given method key doesn't already exist.
    /// </summary>
    public unsafe void TryAdd(int argumentCount, ConstructorTrampoline trampoline)
        => _tryAddDelegate(_scriptPtr, argumentCount, trampoline.TrampolineDelegate);
}

/// <summary>
/// This is used to collect the method trampolines for a script.
/// </summary>
/// <remarks>
/// The user script can implement a static method "GetGodotClassTrampolines"
/// that receives an instance of <see cref="TrampolineCollectors"/> which
/// contains an instance of this class. The user script can then call the
/// <see cref="TryAdd"/> method of this class to add methods to the script.<br/>
/// <br/>
/// "GetGodotClassTrampolines" must be called on the most derived class first,
/// and after collecting the members for that class, its implementation must call the base
/// implementation to also collect members from the base classes in inheritance order.<br/>
/// As a result, <see cref="TryAdd"/> will not add a method from a base class if the derived class
/// has already added a method with the same MethodKey.
/// </remarks>
[PublicAPI]
public class MethodTrampolineCollector
{
    private IntPtr _scriptPtr;
    private unsafe TryAddMethodTrampolineDelegate _tryAddDelegate;

    internal unsafe MethodTrampolineCollector(IntPtr scriptPtr, TryAddMethodTrampolineDelegate tryAddDelegate)
    {
        _scriptPtr = scriptPtr;
        _tryAddDelegate = tryAddDelegate;
    }

    internal unsafe void Update(IntPtr scriptPtr, TryAddMethodTrampolineDelegate tryAddDelegate)
    {
        _scriptPtr = scriptPtr;
        _tryAddDelegate = tryAddDelegate;
    }

    /// <summary>
    /// Adds a method trampoline to the script if a trampoline for the given method key doesn't already exist.
    /// </summary>
    public unsafe void TryAdd(MethodKey methodKey, MethodTrampoline trampoline)
    {
        var nameSelf = (godot_string_name)methodKey.Name.NativeValue;
        _tryAddDelegate(_scriptPtr, &nameSelf, methodKey.ArgumentCount,
            trampoline.TrampolineDelegate, trampoline.IsStatic.ToGodotBool());
    }
}

/// <summary>
/// This is used to collect the property trampolines for a script.
/// </summary>
/// <remarks>
/// The user script can implement a static method "GetGodotClassTrampolines"
/// that receives an instance of <see cref="TrampolineCollectors"/> which
/// contains an instance of this class. The user script can then call the
/// <see cref="TryAdd"/> method of this class to add properties to the script.<br/>
/// <br/>
/// "GetGodotClassTrampolines" must be called on the most derived class first,
/// and after collecting the members for that class, its implementation must call the base
/// implementation to also collect members from the base classes in inheritance order.<br/>
/// As a result, <see cref="TryAdd"/> will not add a property from a base class if the derived class
/// has already added a property with the same name.
/// </remarks>
[PublicAPI]
public class PropertyTrampolineCollector
{
    private IntPtr _scriptPtr;
    private unsafe TryAddPropertyTrampolineDelegate _tryAddDelegate;

    internal unsafe PropertyTrampolineCollector(IntPtr scriptPtr,
        TryAddPropertyTrampolineDelegate tryAddDelegate)
    {
        _scriptPtr = scriptPtr;
        _tryAddDelegate = tryAddDelegate;
    }

    internal unsafe void Update(IntPtr scriptPtr, TryAddPropertyTrampolineDelegate tryAddDelegate)
    {
        _scriptPtr = scriptPtr;
        _tryAddDelegate = tryAddDelegate;
    }

    /// <summary>
    /// Adds property trampolines to the script if trampolines for the given property name don't already exist.
    /// If trampolines for the property already exist, but one of the getter or setter trampolines is missing,
    /// the missing trampoline will be added if provided. This allows a derived class to introduce a readonly
    /// or write-only property with the same name as a property in the base class, overriding that specific
    /// accessor while inheriting the other one. This is done only to match the behavior of the old
    /// trampoline system (SetGodotClassPropertyTrampoline and GetGodotClassPropertyTrampoline).
    /// </summary>
    public unsafe void TryAdd(StringName propertyName, PropertyTrampolines trampolines)
    {
        var propertyNameSelf = (godot_string_name)propertyName.NativeValue;
        _tryAddDelegate(_scriptPtr, &propertyNameSelf,
            trampolines.getterTramp.TrampolineDelegate,
            trampolines.setterTramp.TrampolineDelegate);
    }
}

/// <summary>
/// This is used to collect the raise signal trampolines for a script.
/// </summary>
/// <remarks>
/// The user script can implement a static method "GetGodotClassTrampolines"
/// that receives an instance of <see cref="TrampolineCollectors"/> which
/// contains an instance of this class. The user script can then call the
/// <see cref="TryAdd"/> method of this class to add signals to the script.<br/>
/// <br/>
/// "GetGodotClassTrampolines" must be called on the most derived class first,
/// and after collecting the members for that class, its implementation must call the base
/// implementation to also collect members from the base classes in inheritance order.<br/>
/// As a result, <see cref="TryAdd"/> will not add a signal from a base class if the derived class
/// has already added a signal with the same SignalKey.
/// </remarks>
[PublicAPI]
public class RaiseSignalTrampolineCollector
{
    private IntPtr _scriptPtr;
    private unsafe TryAddRaiseSignalTrampolineDelegate _tryAddDelegate;

    internal unsafe RaiseSignalTrampolineCollector(IntPtr scriptPtr,
        TryAddRaiseSignalTrampolineDelegate tryAddDelegate)
    {
        _scriptPtr = scriptPtr;
        _tryAddDelegate = tryAddDelegate;
    }

    internal unsafe void Update(IntPtr scriptPtr, TryAddRaiseSignalTrampolineDelegate tryAddDelegate)
    {
        _scriptPtr = scriptPtr;
        _tryAddDelegate = tryAddDelegate;
    }

    /// <summary>
    /// Adds a raise signal trampoline to the script if a trampoline for the given signal key doesn't already exist.
    /// </summary>
    public unsafe void TryAdd(SignalKey signalKey, RaiseSignalTrampoline trampoline)
    {
        var nameSelf = (godot_string_name)signalKey.Name.NativeValue;
        _tryAddDelegate(_scriptPtr, &nameSelf, signalKey.ArgumentCount, trampoline.TrampolineDelegate);
    }
}

/// <summary>
/// Group of collectors passed to the user script to collect trampolines
/// and method name to proxy name mappings for a script.
/// </summary>
[PublicAPI]
public record TrampolineCollectors(
    ConstructorTrampolineCollector ConstructorTrampolineCollector,
    MethodTrampolineCollector MethodTrampolineCollector,
    PropertyTrampolineCollector PropertyTrampolineCollector,
    RaiseSignalTrampolineCollector RaiseSignalTrampolineCollector)
{
    internal unsafe void UpdateCollectors(IntPtr scriptPtr,
        TryAddConstructorTrampolineDelegate tryAddConstructorTrampoline,
        TryAddMethodTrampolineDelegate tryAddMethodTrampoline,
        TryAddPropertyTrampolineDelegate tryAddPropertyTrampoline,
        TryAddRaiseSignalTrampolineDelegate tryAddRaiseSignalTrampoline)
    {
        ConstructorTrampolineCollector.Update(scriptPtr, tryAddConstructorTrampoline);
        MethodTrampolineCollector.Update(scriptPtr, tryAddMethodTrampoline);
        PropertyTrampolineCollector.Update(scriptPtr, tryAddPropertyTrampoline);
        RaiseSignalTrampolineCollector.Update(scriptPtr, tryAddRaiseSignalTrampoline);
    }
}

/// <summary>
/// Options for collecting trampolines for a script.
/// </summary>
/// <param name="IncludeAncestors">
/// Whether trampolines from ancestor classes should also be collected.
/// If true, the trampoline collection method of each ancestor class must be called
/// after the trampoline collection method of the current class.
/// </param>
[PublicAPI]
public record TrampolineCollectionOptions(bool IncludeAncestors)
{
    /// <summary>
    /// Whether constructors should also be collected.
    /// </summary>
    /// <remarks>
    /// Implementations of GetGodotClassTrampolines must set this to <see langword="false"/>
    /// before calling GetGodotClassTrampolines on its ancestor.
    /// </remarks>
    public bool CollectConstructors { get; set; }
}
