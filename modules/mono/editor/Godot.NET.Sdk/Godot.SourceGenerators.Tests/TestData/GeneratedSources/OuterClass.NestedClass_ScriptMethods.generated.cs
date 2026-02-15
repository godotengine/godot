using Godot;
using Godot.NativeInterop;
using Godot.Bridge;
using System.Runtime.CompilerServices;

partial struct OuterClass
{
partial class NestedClass
{
#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword
    /// <summary>
    /// Cached StringNames for the methods contained in this class, for fast lookup.
    /// </summary>
    public new class MethodName : global::Godot.RefCounted.MethodName
    {
        /// <summary>
        /// Cached name for the '_Get' method.
        /// </summary>
        public new static readonly global::Godot.StringName @_Get = "_Get";
    }
    /// <summary>
    /// Get the method information for all the methods declared in this class.
    /// This method is used by Godot to register the available methods in the editor.
    /// Do not call this method.
    /// </summary>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    internal new static global::System.Collections.Generic.List<global::Godot.Bridge.MethodInfo> GetGodotMethodList()
    {
        var methods = new global::System.Collections.Generic.List<global::Godot.Bridge.MethodInfo>(1);
        methods.Add(new(name: MethodName.@_Get, returnVal: new(type: (global::Godot.Variant.Type)0, name: "", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)131078, exported: false), flags: (global::Godot.MethodFlags)1, arguments: new() { new(type: (global::Godot.Variant.Type)21, name: "property", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false), }, defaultArguments: null));
        return methods;
    }
#pragma warning restore CS0109

    protected new static readonly ScriptMethodRegistry<NestedClass> MethodRegistry = new ScriptMethodRegistry<NestedClass>()
        .Register(global::Godot.RefCounted.MethodRegistry)
        .Register(MethodName._Get, 1, ScriptMethodDispatchHelper.CreateScriptMethod__Get1())
        .Compile();

    private sealed class ScriptMethodDispatchHelper
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ScriptMethod<GodotObject> CreateScriptMethod__Get1()
        {
            return [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in NativeVariantPtrArgs args) =>
            {
                var callRet = Unsafe.As<GodotObject, NestedClass>(ref scriptInstance)._Get(global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.StringName>(args[0]));
                var ret = global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Variant>(callRet);
                return ret;
            };
        }
    }

    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    public override ref readonly ScriptMethod<GodotObject> TryGetGodotClassMethod(in godot_string_name method, int argc)
    {
        return ref MethodRegistry.TryGetMethodFast(in method, argc);
    }

    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    protected override bool InvokeGodotClassMethod(in godot_string_name method, NativeVariantPtrArgs args, out godot_variant ret)
    {
        if (MethodRegistry.TryGetMethod(in method, args.Count, out var scriptMethod))
        {
            ret = scriptMethod(this, args);
            return true;
        }

        ret = new godot_variant();
        return false;
    }

    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    protected override bool HasGodotClassMethod(in godot_string_name method)
    {
        return MethodRegistry.ContainsMethod(method);
    }
}
}
