using Godot;

[GlobalClass]
public partial class CustomGlobalClass1 : GodotObject // This works because it inherits from GodotObject
{

}

[GlobalClass]
public partial class CustomGlobalClass2 : Node // This works because it inherits from an object that inherits from GodotObject
{

}

[GlobalClass]
public partial class CustomGlobalClass3 // TThis raises a GD0401 diagnositc error: global classes must inherit from GodotObject
{

}
