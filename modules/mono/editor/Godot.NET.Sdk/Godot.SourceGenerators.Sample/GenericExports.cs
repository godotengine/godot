using Godot.Collections;

namespace Godot.SourceGenerators.Sample;

public partial class GenericExports<[MustBeVariant] T> : GodotObject
{
    [Export] public T RegularField;
    [Export] public T RegularProperty { get; set; }

    [Export] public Array<T> ArrayProperty { get; set; }

    [Signal]
    public delegate T GenericEventHandler(T var);

    public T GenericMethod(T var)
    {
        return var;
    }
}

public partial class GenericExports : GenericExports<Vector2>
{
}

public partial class GenericExportsRect2 : GenericExports<Rect2>
{
}

public partial class GenericExports<TSome, [MustBeVariant] TOther> : GodotObject
{
    [Export] public Array<TOther> ArrayExport { get; set; }

    // This is not valid because TSome is not [MustBeVariant]
    // [Export] public TSome AnotherArray { get; set; }

    // You can still use TSome, just not exported.
    public TSome NonExportField;
}

public partial class GenericArrayExport<[MustBeVariant] T> : GodotObject
{
    [Export] public T[] ArrayOfT;
}

public partial class GenericArrayExportInt : GenericArrayExport<int>
{
}

// This is not valid because it results in attempting to export a Plane[], which is not a valid Variant type.
// An error is generated suggesting to use a Godot array instead of a C# array.
// public partial class GenericArrayExportPlane : GenericArrayExport<Plane>
// {
// }

// Doesn't require the [MustBeVariant] attribute because T is constrained to GodotObject already.
public partial class InferredVariantCompatible<T> : Resource
    where T : GodotObject
{
    [Export] public T Exported;
}
