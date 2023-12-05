using System;

namespace Godot.SourceGenerators.Sample
{
    public partial class AllWriteOnly : GodotObject
    {
        bool writeonly_backing_field = false;
        public bool writeonly_property { set => writeonly_backing_field = value; }
    }
}
