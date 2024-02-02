using Godot;

// This works because it inherits from GodotObject and it doesn't have any generic type parameter.
[GlobalClass]
public partial class CustomGlobalClass : GodotObject 
{

}

// This raises a GD0402 diagnostic error: global classes can't have any generic type parameter
{|GD0402:[GlobalClass]
public partial class CustomGlobalClass<T> : GodotObject
{

}|}
