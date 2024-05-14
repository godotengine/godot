using Godot.Collections;
using System;

namespace Godot.SourceGenerators.Sample;

public partial class GenericExports<[MustBeVariant] T> : GodotObject
{
    [Export] public T RegularField;
    [Export] public T RegularProperty { get; set; }

    [Export] public Array<T> ArrayProperty { get; set; }

    [Signal]
    public delegate T GenericSignalEventHandler(T var);

    public T GenericMethod(T var)
    {
        return var;
    }
}

public partial class GenericExportsVector2 : GenericExports<Vector2>
{
}

public partial class GenericExportsRect2 : GenericExports<Rect2>
{
}

public partial class GenericExportsMultiple<TSome, [MustBeVariant] TOther> : GodotObject
{
    [Export] public Array<TOther> ArrayExport { get; set; }

    // This is not valid because TSome is not [MustBeVariant]
    // [Export] public TSome AnotherArray { get; set; }

    // You can still use TSome, just not exported.
    public TSome NonExportField;
}

// Doesn't require the [MustBeVariant] attribute because T is constrained to GodotObject or Enum already.
public partial class InferredVariantConstraintGodotObject<T> : GodotObject
    where T : GodotObject
{
    [Export] public T Exported;
}

public partial class InferredVariantConstraintEnum<T> : GodotObject
    where T : Enum
{
    [Export] public T Exported;
}
