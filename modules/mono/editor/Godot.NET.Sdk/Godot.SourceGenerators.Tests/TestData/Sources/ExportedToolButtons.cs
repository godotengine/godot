using Godot;
using System;

[Tool]
public partial class ExportedToolButtons : GodotObject
{
    [Description("Run the preview.")]
    [ExportToolButton("Click me!")]
    public Callable MyButton1 => Callable.From(() => { GD.Print("Clicked MyButton1!"); });

    [ExportToolButton("Click me!", Icon = "ColorRect")]
    public Callable MyButton2 => Callable.From(() => { GD.Print("Clicked MyButton2!"); });
}
