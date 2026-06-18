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

#pragma warning disable CS0618 // Type or member is obsolete
    protected new static readonly ScriptMethodRegistry<NestedClass> MethodRegistry = new ScriptMethodRegistry<NestedClass>()
        .Register(global::Godot.RefCounted.MethodRegistry)
        .Register(MethodName.@_Get, 1, ScriptMethodDispatchHelper.CreateScriptMethod__Get1())
        .Build();

    private sealed class ScriptMethodDispatchHelper
    {
        public static ScriptMethod CreateScriptMethod__Get1()
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
#pragma warning restore CS0618 // Type or member is obsolete

    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    public override ref readonly ScriptMethod GetGodotClassMethodOrNullRef(in godot_string_name method, int argCount)
    {
        return ref MethodRegistry.GetMethodOrNullRef(in method, argCount);
    }

    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    protected override bool InvokeGodotClassMethod(in godot_string_name method, NativeVariantPtrArgs args, out godot_variant ret)
    {
        ref readonly var scriptMethod = ref GetGodotClassMethodOrNullRef(in method, args.Count);
        if (!Unsafe.IsNullRef(in scriptMethod))
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
        return MethodRegistry.ContainsName(method);
    }
}
}
