using Godot;
using Godot.NativeInterop;
using Godot.Bridge;
using System.Runtime.CompilerServices;

partial class ScriptBoilerplate
{
#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword
    /// <summary>
    /// Cached StringNames for the methods contained in this class, for fast lookup.
    /// </summary>
    public new class MethodName : global::Godot.Node.MethodName
    {
        /// <summary>
        /// Cached name for the '_Process' method.
        /// </summary>
        public new static readonly global::Godot.StringName @_Process = "_Process";
        /// <summary>
        /// Cached name for the 'Bazz' method.
        /// </summary>
        public new static readonly global::Godot.StringName @Bazz = "Bazz";
    }
    /// <summary>
    /// Get the method information for all the methods declared in this class.
    /// This method is used by Godot to register the available methods in the editor.
    /// Do not call this method.
    /// </summary>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    internal new static global::System.Collections.Generic.List<global::Godot.Bridge.MethodInfo> GetGodotMethodList()
    {
        var methods = new global::System.Collections.Generic.List<global::Godot.Bridge.MethodInfo>(2);
        methods.Add(new(name: MethodName.@_Process, returnVal: new(type: (global::Godot.Variant.Type)0, name: "", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false), flags: (global::Godot.MethodFlags)1, arguments: new() { new(type: (global::Godot.Variant.Type)3, name: "delta", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false), }, defaultArguments: null));
        methods.Add(new(name: MethodName.@Bazz, returnVal: new(type: (global::Godot.Variant.Type)2, name: "", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false), flags: (global::Godot.MethodFlags)1, arguments: new() { new(type: (global::Godot.Variant.Type)21, name: "name", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false), }, defaultArguments: null));
        return methods;
    }
#pragma warning restore CS0109

    protected new static readonly ScriptMethodRegistry<ScriptBoilerplate> MethodRegistry = new ScriptMethodRegistry<ScriptBoilerplate>()
        .Register(global::Godot.Node.MethodRegistry)
        .Register(MethodName._Process, 1, ScriptMethodDispatchHelper.CreateScriptMethod__Process1())
        .Register(MethodName.Bazz, 1, ScriptMethodDispatchHelper.CreateScriptMethod_Bazz1())
        .Compile();

    private sealed class ScriptMethodDispatchHelper
    {
        public static ScriptMethod<GodotObject> CreateScriptMethod__Process1()
        {
            static godot_variant Impl(GodotObject scriptInstance, scoped in NativeVariantPtrArgs args)
            {
                Unsafe.As<GodotObject, ScriptBoilerplate>(ref scriptInstance)._Process(global::Godot.NativeInterop.VariantUtils.ConvertTo<double>(args[0]));
                godot_variant ret = default;
                return ret;
            }

            // Wrap static method into ScriptMethodPtr
            //return ScriptMethodPtr.Create<ScriptBoilerplate>(&Impl);
            return Impl;
        }
        public static ScriptMethod<GodotObject> CreateScriptMethod_Bazz1()
        {
            static godot_variant Impl(GodotObject scriptInstance, scoped in NativeVariantPtrArgs args)
            {
                var callRet = Unsafe.As<GodotObject, ScriptBoilerplate>(ref scriptInstance).Bazz(global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.StringName>(args[0]));
                var ret = global::Godot.NativeInterop.VariantUtils.CreateFrom<int>(callRet);
                return ret;
            }

            // Wrap static method into ScriptMethodPtr
            //return ScriptMethodPtr.Create<ScriptBoilerplate>(&Impl);
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
