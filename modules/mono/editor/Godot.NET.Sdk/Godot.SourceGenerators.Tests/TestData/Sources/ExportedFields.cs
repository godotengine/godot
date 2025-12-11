using Godot;
using System;
using System.Collections.Generic;
using static Godot.Mathf;

public partial class ExportedFields : GodotObject
{
    [Export] private Boolean _fieldBoolean = true;
    [Export] private Char _fieldChar = 'f';
    [Export] private SByte _fieldSByte = 10;
    [Export] private Int16 _fieldInt16 = 10;
    [Export] private Int32 _fieldInt32 = 10;
    [Export] private Int64 _fieldInt64 = -10_000;
    [Export] private Byte _fieldByte = 10;
    [Export] private UInt16 _fieldUInt16 = 10;
    [Export] private UInt32 _fieldUInt32 = 10;
    [Export] private UInt64 _fieldUInt64 = 10;
    [Export] private Single _fieldSingle = 10;
    [Export] private Double _fieldDouble = 10;
    [Export] private String _fieldString = "foo";

    // Static import
    [Export] private Single _fieldStaticImport = RadToDeg(2 * Pi);

    // Godot structs
    [Export] private Vector2 _fieldVector2 = new(10f, 10f);
    [Export] private Vector2I _fieldVector2I = Vector2I.Up;
    [Export] private Rect2 _fieldRect2 = new(new Vector2(10f, 10f), new Vector2(10f, 10f));
    [Export] private Rect2I _fieldRect2I = new(new Vector2I(10, 10), new Vector2I(10, 10));
    [Export] private Transform2D _fieldTransform2D = Transform2D.Identity;
    [Export] private Vector3 _fieldVector3 = new(10f, 10f, 10f);
    [Export] private Vector3I _fieldVector3I = Vector3I.Back;
    [Export] private Basis _fieldBasis = new Basis(Quaternion.Identity);
    [Export] private Quaternion _fieldQuaternion = new Quaternion(Basis.Identity);
    [Export] private Transform3D _fieldTransform3D = Transform3D.Identity;
    [Export] private Vector4 _fieldVector4 = new(10f, 10f, 10f, 10f);
    [Export] private Vector4I _fieldVector4I = Vector4I.One;
    [Export] private Projection _fieldProjection = Projection.Identity;
    [Export] private Aabb _fieldAabb = new Aabb(10f, 10f, 10f, new Vector3(1f, 1f, 1f));
    [Export] private Color _fieldColor = Colors.Aquamarine;
    [Export] private Plane _fieldPlane = Plane.PlaneXZ;
    [Export] private Callable _fieldCallable = new Callable(Engine.GetMainLoop(), "_process");
    [Export] private Signal _fieldSignal = new Signal(Engine.GetMainLoop(), "property_list_changed");

    // Enums
    public enum MyEnum
    {
        A,
        B,
        C
    }

    [Export] private MyEnum _fieldEnum = MyEnum.C;

    [Flags]
    public enum MyFlagsEnum
    {
        A,
        B,
        C
    }

    [Export] private MyFlagsEnum _fieldFlagsEnum = MyFlagsEnum.C;

    // Arrays
    [Export] private Byte[] _fieldByteArray = { 0, 1, 2, 3, 4, 5, 6 };
    [Export] private Int32[] _fieldInt32Array = { 0, 1, 2, 3, 4, 5, 6 };
    [Export] private Int64[] _fieldInt64Array = { 0, 1, 2, 3, 4, 5, 6 };
    [Export] private Single[] _fieldSingleArray = { 0f, 1f, 2f, 3f, 4f, 5f, 6f };
    [Export] private Double[] _fieldDoubleArray = { 0d, 1d, 2d, 3d, 4d, 5d, 6d };
    [Export] private String[] _fieldStringArray = { "foo", "bar" };
    [Export(PropertyHint.Enum, "A,B,C")] private String[] _fieldStringArrayEnum = { "foo", "bar" };
    [Export] private Vector2[] _fieldVector2Array = { Vector2.Up, Vector2.Down, Vector2.Left, Vector2.Right };
    [Export] private Vector3[] _fieldVector3Array = { Vector3.Up, Vector3.Down, Vector3.Left, Vector3.Right };
    [Export] private Color[] _fieldColorArray = { Colors.Aqua, Colors.Aquamarine, Colors.Azure, Colors.Beige };
    [Export] private GodotObject[] _fieldGodotObjectOrDerivedArray = { null };
    [Export] private StringName[] _fieldStringNameArray = { "foo", "bar" };
    [Export] private NodePath[] _fieldNodePathArray = { "foo", "bar" };
    [Export] private Rid[] _fieldRidArray = { default, default, default };
    // Note we use Array and not System.Array. This tests the generated namespace qualification.
    [Export] private Int32[] _fieldEmptyInt32Array = Array.Empty<Int32>();
    // Note we use List and not System.Collections.Generic.
    [Export] private int[] _fieldArrayFromList = new List<int>(Array.Empty<int>()).ToArray();

    // Variant
    [Export] private Variant _fieldVariant = "foo";

    // Classes
    [Export] private GodotObject _fieldGodotObjectOrDerived;
    [Export] private Godot.Texture _fieldGodotResourceTexture;
    [Export] private Godot.Texture _fieldGodotResourceTextureWithInitializer = new() { ResourceName = "" };
    [Export] private StringName _fieldStringName = new StringName("foo");
    [Export] private NodePath _fieldNodePath = new NodePath("foo");
    [Export] private Rid _fieldRid;

    [Export]
    private Godot.Collections.Dictionary _fieldGodotDictionary = new() { { "foo", 10 }, { Vector2.Up, Colors.Chocolate } };

    [Export]
    private Godot.Collections.Array _fieldGodotArray = new() { "foo", 10, Vector2.Up, Colors.Chocolate };

    [Export]
    private Godot.Collections.Dictionary<string, bool> _fieldGodotGenericDictionary = new() { { "foo", true }, { "bar", false } };

    [Export]
    private Godot.Collections.Array<int> _fieldGodotGenericArray = new() { 0, 1, 2, 3, 4, 5, 6 };
}
