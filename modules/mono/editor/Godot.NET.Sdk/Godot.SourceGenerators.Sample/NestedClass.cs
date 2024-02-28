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

            [Export] private String _fieldString = "foo";
            [Export] private String PropertyString { get; set; } = "foo";

            private void Method()
            {
            }
        }
    }
}
