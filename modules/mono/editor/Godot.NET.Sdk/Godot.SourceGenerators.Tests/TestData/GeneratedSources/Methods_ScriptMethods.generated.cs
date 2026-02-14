using Godot;
using Godot.NativeInterop;
using Godot.Bridge;
using System.Runtime.CompilerServices;

partial class Methods
{
#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword
    /// <summary>
    /// Cached StringNames for the methods contained in this class, for fast lookup.
    /// </summary>
    public new class MethodName : global::Godot.GodotObject.MethodName
    {
        /// <summary>
        /// Cached name for the 'MethodWithOverload' method.
        /// </summary>
        public new static readonly global::Godot.StringName @MethodWithOverload = "MethodWithOverload";
    }
    /// <summary>
    /// Get the method information for all the methods declared in this class.
    /// This method is used by Godot to register the available methods in the editor.
    /// Do not call this method.
    /// </summary>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    internal new static global::System.Collections.Generic.List<global::Godot.Bridge.MethodInfo> GetGodotMethodList()
    {
        var methods = new global::System.Collections.Generic.List<global::Godot.Bridge.MethodInfo>(3);
        methods.Add(new(name: MethodName.@MethodWithOverload, returnVal: new(type: (global::Godot.Variant.Type)0, name: "", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false), flags: (global::Godot.MethodFlags)1, arguments: null, defaultArguments: null));
        methods.Add(new(name: MethodName.@MethodWithOverload, returnVal: new(type: (global::Godot.Variant.Type)0, name: "", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false), flags: (global::Godot.MethodFlags)1, arguments: new() { new(type: (global::Godot.Variant.Type)2, name: "a", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false), }, defaultArguments: null));
        methods.Add(new(name: MethodName.@MethodWithOverload, returnVal: new(type: (global::Godot.Variant.Type)0, name: "", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false), flags: (global::Godot.MethodFlags)1, arguments: new() { new(type: (global::Godot.Variant.Type)2, name: "a", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false), new(type: (global::Godot.Variant.Type)2, name: "b", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false), }, defaultArguments: null));
        return methods;
    }
#pragma warning restore CS0109

    protected new static readonly ScriptMethodRegistry<Methods> MethodRegistry = new ScriptMethodRegistry<Methods>()
        .Register(global::Godot.GodotObject.MethodRegistry)
        .Register(MethodName.MethodWithOverload, 0, ScriptMethodDispatchHelper.CreateScriptMethod_MethodWithOverload0())
        .Register(MethodName.MethodWithOverload, 1, ScriptMethodDispatchHelper.CreateScriptMethod_MethodWithOverload1())
        .Register(MethodName.MethodWithOverload, 2, ScriptMethodDispatchHelper.CreateScriptMethod_MethodWithOverload2())
        .Compile();

    private sealed class ScriptMethodDispatchHelper
    {
        public static ScriptMethod<GodotObject> CreateScriptMethod_MethodWithOverload0()
        {
            static godot_variant Impl(GodotObject scriptInstance, scoped in NativeVariantPtrArgs args)
            {
                Unsafe.As<GodotObject, Methods>(ref scriptInstance).MethodWithOverload();
                godot_variant ret = default;
                return ret;
            }

            // Wrap static method into ScriptMethodPtr
            //return ScriptMethodPtr.Create<Methods>(&Impl);
            return Impl;
        }
        public static ScriptMethod<GodotObject> CreateScriptMethod_MethodWithOverload1()
        {
            static godot_variant Impl(GodotObject scriptInstance, scoped in NativeVariantPtrArgs args)
            {
                Unsafe.As<GodotObject, Methods>(ref scriptInstance).MethodWithOverload(global::Godot.NativeInterop.VariantUtils.ConvertTo<int>(args[0]));
                godot_variant ret = default;
                return ret;
            }

            // Wrap static method into ScriptMethodPtr
            //return ScriptMethodPtr.Create<Methods>(&Impl);
            return Impl;
        }
        public static ScriptMethod<GodotObject> CreateScriptMethod_MethodWithOverload2()
        {
            static godot_variant Impl(GodotObject scriptInstance, scoped in NativeVariantPtrArgs args)
            {
                Unsafe.As<GodotObject, Methods>(ref scriptInstance).MethodWithOverload(global::Godot.NativeInterop.VariantUtils.ConvertTo<int>(args[0]), global::Godot.NativeInterop.VariantUtils.ConvertTo<int>(args[1]));
                godot_variant ret = default;
                return ret;
            }

            // Wrap static method into ScriptMethodPtr
            //return ScriptMethodPtr.Create<Methods>(&Impl);
            return Impl;
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
