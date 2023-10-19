using System;

namespace Godot.SourceGenerators.Sample;

public partial class NestedClass : GodotObject
{
    public partial class NestedClass2 : GodotObject
    {
        public partial class NestedClass3 : GodotObject
        {
            [Signal]
            public delegate void MySignalEventHandler(string str, int num);

            [Export] private String field_String = "foo";
            [Export] private String property_String { get; set; } = "foo";

            private void Method()
            {
            }
        }
    }
}
