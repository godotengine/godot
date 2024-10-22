using Godot;
using Godot.NativeInterop;

partial class EventSignals
{
#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword
    /// <summary>
    /// Cached StringNames for the signals contained in this class, for fast lookup.
    /// </summary>
    public new class SignalName : global::Godot.GodotObject.SignalName {
        /// <summary>
        /// Cached name for the 'MySignal' signal.
        /// </summary>
        public new static readonly global::Godot.StringName @MySignal = "MySignal";
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
        signals.Add(new(name: SignalName.@MySignal, returnVal: new(type: (global::Godot.Variant.Type)0, name: "", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false), flags: (global::Godot.MethodFlags)1, arguments: new() { new(type: (global::Godot.Variant.Type)4, name: "str", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false), new(type: (global::Godot.Variant.Type)2, name: "num", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false),  }, defaultArguments: null));
        return signals;
    }
#pragma warning restore CS0109
    private global::EventSignals.MySignalEventHandler backing_MySignal;
    /// <inheritdoc cref="global::EventSignals.MySignalEventHandler"/>
    public event global::EventSignals.MySignalEventHandler @MySignal {
        add => backing_MySignal += value;
        remove => backing_MySignal -= value;
}
    protected void EmitSignalMySignal(string str, int num)
    {
        EmitSignal(SignalName.MySignal, str, num);
    }
    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    protected override void RaiseGodotClassSignalCallbacks(in godot_string_name signal, NativeVariantPtrArgs args)
    {
        if (signal == SignalName.@MySignal && args.Count == 2) {
            backing_MySignal?.Invoke(global::Godot.NativeInterop.VariantUtils.ConvertTo<string>(args[0]), global::Godot.NativeInterop.VariantUtils.ConvertTo<int>(args[1]));
            return;
        }
        base.RaiseGodotClassSignalCallbacks(signal, args);
    }
    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    protected override bool HasGodotClassSignal(in godot_string_name signal)
    {
        if (signal == SignalName.@MySignal) {
           return true;
        }
        return base.HasGodotClassSignal(signal);
    }
}
