using Godot;

partial class Bar : GodotObject
{
}

// Foo in another file
partial class Foo
{
}

partial class NotSameNameAsFile : GodotObject
{
}
