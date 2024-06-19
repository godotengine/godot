using Godot;
using Godot.NativeInterop;

partial class Generic<T>
{
#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword
    /// <summary>
    /// Cached StringNames for the methods contained in this class, for fast lookup.
    /// </summary>
    public new class MethodName : global::Godot.GodotObject.MethodName {
        /// <summary>
        /// Cached name for the 'GenericMethod' method.
        /// </summary>
        public new static readonly global::Godot.StringName GenericMethod = "GenericMethod";
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
        methods.Add(new(name: MethodName.GenericMethod, returnVal: global::Godot.Bridge.GenericUtils.PropertyInfoFromGenericType<T>(name: "", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false), flags: (global::Godot.MethodFlags)1, arguments: new() { global::Godot.Bridge.GenericUtils.PropertyInfoFromGenericType<T>(name: "var", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false),  }, defaultArguments: null));
        return methods;
    }
#pragma warning restore CS0109
    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    protected override bool InvokeGodotClassMethod(in godot_string_name method, NativeVariantPtrArgs args, out godot_variant ret)
    {
        if (method == MethodName.GenericMethod && args.Count == 1) {
            var callRet = GenericMethod(global::Godot.NativeInterop.VariantUtils.ConvertTo<T>(args[0]));
            ret = global::Godot.NativeInterop.VariantUtils.CreateFrom<T>(callRet);
            return true;
        }
        return base.InvokeGodotClassMethod(method, args, out ret);
    }
    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    protected override bool HasGodotClassMethod(in godot_string_name method)
    {
        if (method == MethodName.GenericMethod) {
           return true;
        }
        return base.HasGodotClassMethod(method);
    }
}
