namespace Godot.SourceGenerators.Sample
{
    public partial class AllReadOnly : GodotObject
    {
        public readonly string readonly_field = "foo";
        public string readonly_auto_property { get; } = "foo";
        public string readonly_property { get => "foo"; }
        public string initonly_auto_property { get; init; }
    }
}
