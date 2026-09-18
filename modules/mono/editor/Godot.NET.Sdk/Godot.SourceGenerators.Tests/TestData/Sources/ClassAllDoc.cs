using Godot;

/// <summary>
/// class description
/// test
/// </summary>
public partial class ClassAllDoc : GodotObject
{
    /// <summary>
    /// field description <c>true</c>
    /// test <see cref="ClassAllDoc"/>
    /// </summary>
    [Export]
    private int _fieldDocTest = 1;

    /// <summary>
    /// property description
    /// test <see cref="ClassAllDoc"/>
    /// </summary>
    [Export]
    public int PropertyDocTest { get; set; }

    /// <summary>
    /// signal description ~!@#$%^*()_+{}|
    /// test <see cref="ClassAllDoc"/>
    /// </summary>
    /// <param name="num"></param>
    [Signal]
    public delegate void SignalDocTestEventHandler(int num);
}
