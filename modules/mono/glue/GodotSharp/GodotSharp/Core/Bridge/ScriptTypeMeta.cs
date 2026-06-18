using System;
using System.Collections.Generic;
using Godot.NativeInterop;
using JetBrains.Annotations;

namespace Godot.Bridge;

#nullable enable

internal delegate bool LegacyInvokeGodotClassStaticMethodDelegate(
    in godot_string_name method, NativeVariantPtrArgs args, out godot_variant ret);

internal delegate bool LegacyCreateManagedForGodotObjectScriptInstanceDelegate(
    IntPtr godotObjectPtr, NativeVariantPtrArgs args);

/// <summary>
/// Represents metadata and callbacks related to a script type, used by <see cref="ScriptManagerBridge"/>
/// to manage script types and their interactions with Godot.
/// </summary>
/// <remarks>
/// Includes information about the managed type, the corresponding native type and name,
/// as well as callbacks for collecting trampolines, methods, signals, properties,
/// RPC methods, and property default values.
/// </remarks>
[PublicAPI]
public record ScriptTypeMeta
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ScriptTypeMeta"/> record with the specified type information and callbacks.
    /// </summary>
    /// <param name="Type">The managed type of the script.</param>
    /// <param name="NativeType">The native Godot class managed type.</param>
    /// <param name="NativeName">
    /// <para>The native Godot class name.</para>
    /// </param>
    /// <remarks>
    /// The <paramref name="NativeName"/> is the C++ class name, which may differ from the C# class name.
    /// This name is available via the <see cref="NativeName"/> property of the C# class.
    /// </remarks>
    public ScriptTypeMeta(Type Type, Type NativeType, StringName NativeName)
    {
        this.Type = Type;
        this.NativeType = NativeType;
        this.NativeName = NativeName;
    }

    /// <summary>
    /// The managed type of the script.
    /// </summary>
    public Type Type { get; init; }

    /// <summary>
    /// The native Godot class managed type.
    /// </summary>
    public Type NativeType { get; init; }

    /// <summary>
    /// <para>The native Godot class name.</para>
    /// </summary>
    /// <remarks>
    /// This is the C++ class name, which may differ from the C# class name.
    /// This name is available via the <see cref="NativeName"/> property of the C# class.
    /// </remarks>
    public StringName NativeName { get; init; }

    /// <summary>
    /// Callbacks for collecting constructor, method, property, and signal trampolines for the script type.
    /// </summary>
    public Action<TrampolineCollectors, TrampolineCollectionOptions>? GetGodotClassTrampolines { get; set; }

    /// <summary>
    /// Callback for collecting the list of <see cref="MethodInfo"/> that describe
    /// the methods that are exposed to Godot for the script type.
    /// </summary>
    public Func<List<MethodInfo>?>? GetGodotMethodList { get; set; }

    /// <summary>
    /// Callback for collecting the list of <see cref="MethodInfo"/> that describe
    /// the signals that are exposed to Godot for the script type.
    /// </summary>
    public Func<List<MethodInfo>?>? GetGodotSignalList { get; set; }

    /// <summary>
    /// Callback for collecting the list of <see cref="PropertyInfo"/> that describe
    /// the properties that are exposed to Godot for the script type.
    /// </summary>
    public Func<List<PropertyInfo>?>? GetGodotPropertyList { get; set; }

    /// <summary>
    /// Callback for collecting metadata about RPC methods that are exposed to Godot for the script type.
    /// </summary>
    public Action<RpcMethodCollector>? GetGodotRpcMethods { get; set; }

    /// <summary>
    /// Callback for collecting the default values of properties that are exposed to Godot for the script type.
    /// </summary>
    /// <remarks>
    /// This information is required to know when a property value has been changed from its default,
    /// which is important for serialization and editor integration.
    /// </remarks>
    public Func<Dictionary<StringName, Variant>?>? GetGodotPropertyDefaultValues { get; set; }

    internal bool ShouldFallbackToLegacyTrampolines { get; set; }
    internal Func<Dictionary<StringName, object?>>? LegacyGetGodotPropertyDefaultValues { get; set; }
    internal LegacyInvokeGodotClassStaticMethodDelegate? LegacyInvokeGodotClassStaticMethod { get; set; }

    internal LegacyCreateManagedForGodotObjectScriptInstanceDelegate? LegacyCreateManagedForGodotObjectScriptInstance
    {
        get;
        set;
    }
}
