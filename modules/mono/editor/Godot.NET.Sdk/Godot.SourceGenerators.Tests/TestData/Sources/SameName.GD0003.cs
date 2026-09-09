using Godot;

namespace NamespaceA
{
    partial class SameName : GodotObject
    {
        private int _field;
    }
}

// SameName again but different namespace
namespace NamespaceB
{
    partial class {|GD0003:SameName|} : GodotObject
    {
        private int _field;
    }
}
