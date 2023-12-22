using Godot;

[GlobalClass]
public partial class CustomGlobalClass : GodotObject // This works because it inherits from GodotObject and it doesn't have any generic type parameter
{

}

[GlobalClass]
public partial class CustomGlobalClass<T> : GodotObject // This raises a GD0402 diagnositc error: global classes can't have any generic type parameter
{

}