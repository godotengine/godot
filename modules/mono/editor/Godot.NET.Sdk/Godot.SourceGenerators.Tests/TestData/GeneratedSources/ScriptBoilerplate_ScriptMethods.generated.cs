using Godot;
using Godot.NativeInterop;
using Godot.Bridge;

partial class ScriptBoilerplate
{
#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword
    /// <summary>
    /// Cached StringNames for the methods contained in this class, for fast lookup.
    /// </summary>
    public new class MethodName : global::Godot.Node.MethodName {
        /// <summary>
        /// Cached name for the '_Process' method.
        /// </summary>
        public new static readonly global::Godot.StringName _Process = "_Process";
        /// <summary>
        /// Cached name for the 'Bazz' method.
        /// </summary>
        public new static readonly global::Godot.StringName Bazz = "Bazz";
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
        methods.Add(new(name: MethodName._Process, returnVal: new(type: (global::Godot.Variant.Type)0, name: "", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false), flags: (global::Godot.MethodFlags)1, arguments: new() { new(type: (global::Godot.Variant.Type)3, name: "delta", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false),  }, defaultArguments: null));
        methods.Add(new(name: MethodName.Bazz, returnVal: new(type: (global::Godot.Variant.Type)2, name: "", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false), flags: (global::Godot.MethodFlags)1, arguments: new() { new(type: (global::Godot.Variant.Type)21, name: "name", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false),  }, defaultArguments: null));
        return methods;
    }
#pragma warning restore CS0109

    public new static readonly ScriptMethodRegistry<ScriptBoilerplate> MethodRegistry = new ScriptMethodRegistry<ScriptBoilerplate>()
        .Register(global::Godot.Node.MethodRegistry)
        .Register(MethodName._Process, 1, (ScriptBoilerplate scriptInstance, NativeVariantPtrArgs args, out godot_variant ret) => 
        {
            scriptInstance._Process(global::Godot.NativeInterop.VariantUtils.ConvertTo<double>(args[0]));
            ret = default;
        })
        .Register(MethodName.Bazz, 1, (ScriptBoilerplate scriptInstance, NativeVariantPtrArgs args, out godot_variant ret) => 
        {
            var callRet = scriptInstance.Bazz(global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.StringName>(args[0]));
            ret = global::Godot.NativeInterop.VariantUtils.CreateFrom<int>(callRet);
        })
        .Compile();

    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    protected override bool InvokeGodotClassMethod(in godot_string_name method, NativeVariantPtrArgs args, out godot_variant ret)
    {
        if (MethodRegistry.TryGetMethod(in method, args.Count, out var scriptMethod))
        {
            scriptMethod(this, args, out ret);
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
