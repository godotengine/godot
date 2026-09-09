using Godot;

// This works because it inherits from GodotObject.
[GlobalClass]
public partial class CustomGlobalClass1 : GodotObject
{

}

// This works because it inherits from an object that inherits from GodotObject
[GlobalClass]
public partial class CustomGlobalClass2 : Node
{

}

// This raises a GD0401 diagnostic error: global classes must inherit from GodotObject
[GlobalClass]
public partial class {|GD0401:CustomGlobalClass3|}
{

}
