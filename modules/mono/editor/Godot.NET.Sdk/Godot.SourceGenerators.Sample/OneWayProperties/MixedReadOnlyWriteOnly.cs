namespace Godot.SourceGenerators.Sample
{
    public partial class MixedReadonlyWriteOnly : GodotObject
    {
        public readonly string readonly_field = "foo";
        public string readonly_auto_property { get; } = "foo";
        public string readonly_property { get => "foo"; }
        public string initonly_auto_property { get; init; }

        bool writeonly_backing_field = false;
        public bool writeonly_property { set => writeonly_backing_field = value; }
    }
}
