using Godot;
using Godot.NativeInterop;

partial class Generic<T>
{
#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword
    /// <summary>
    /// Cached StringNames for the signals contained in this class, for fast lookup.
    /// </summary>
    public new class SignalName : global::Godot.GodotObject.SignalName {
        /// <summary>
        /// Cached name for the 'GenericSignal' signal.
        /// </summary>
        public new static readonly global::Godot.StringName GenericSignal = "GenericSignal";
    }
    /// <summary>
    /// Get the signal information for all the signals declared in this class.
    /// This method is used by Godot to register the available signals in the editor.
    /// Do not call this method.
    /// </summary>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    internal new static global::System.Collections.Generic.List<global::Godot.Bridge.MethodInfo> GetGodotSignalList()
    {
        var signals = new global::System.Collections.Generic.List<global::Godot.Bridge.MethodInfo>(1);
        signals.Add(new(name: SignalName.GenericSignal, returnVal: global::Godot.Bridge.GenericUtils.PropertyInfoFromGenericType<T>(name: "", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false), flags: (global::Godot.MethodFlags)1, arguments: new() { global::Godot.Bridge.GenericUtils.PropertyInfoFromGenericType<T>(name: "var", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false),  }, defaultArguments: null));
        return signals;
    }
#pragma warning restore CS0109
    private global::Generic<T>.GenericSignalEventHandler backing_GenericSignal;
    /// <inheritdoc cref="global::Generic{T}.GenericSignalEventHandler"/>
    public event global::Generic<T>.GenericSignalEventHandler GenericSignal {
        add => backing_GenericSignal += value;
        remove => backing_GenericSignal -= value;
}
    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    protected override void RaiseGodotClassSignalCallbacks(in godot_string_name signal, NativeVariantPtrArgs args)
    {
        if (signal == SignalName.GenericSignal && args.Count == 1) {
            backing_GenericSignal?.Invoke(global::Godot.NativeInterop.VariantUtils.ConvertTo<T>(args[0]));
            return;
        }
        base.RaiseGodotClassSignalCallbacks(signal, args);
    }
    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    protected override bool HasGodotClassSignal(in godot_string_name signal)
    {
        if (signal == SignalName.GenericSignal) {
           return true;
        }
        return base.HasGodotClassSignal(signal);
    }
}
