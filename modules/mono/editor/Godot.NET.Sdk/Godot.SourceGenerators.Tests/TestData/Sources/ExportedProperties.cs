using Godot;
using System;

public partial class ExportedProperties : GodotObject
{
    // Do not generate default value
    private String _notGeneratePropertyString = new string("not generate");
    [Export]
    public String NotGenerateComplexLamdaProperty
    {
        get => _notGeneratePropertyString + Convert.ToInt32("1");
        set => _notGeneratePropertyString = value;
    }

    [Export]
    public String NotGenerateLamdaNoFieldProperty
    {
        get => new string("not generate");
        set => _notGeneratePropertyString = value;
    }

    [Export]
    public String NotGenerateComplexReturnProperty
    {
        get
        {
            return _notGeneratePropertyString + Convert.ToInt32("1");
        }
        set
        {
            _notGeneratePropertyString = value;
        }
    }

    private int _notGeneratePropertyInt = 1;
    [Export]
    public string NotGenerateReturnsProperty
    {
        get
        {
            if (_notGeneratePropertyInt == 1)
            {
                return "a";
            }
            else
            {
                return "b";
            }
        }
        set
        {
            _notGeneratePropertyInt = value == "a" ? 1 : 2;
        }
    }

    // Full Property
    private String _fullPropertyString = "FullPropertyString";
    [Export]
    public String FullPropertyString
    {
        get
        {
            return _fullPropertyString;
        }
        set
        {
            _fullPropertyString = value;
        }
    }

    private String _fullPropertyStringComplex = new string("FullPropertyString_Complex") + Convert.ToInt32("1");
    [Export]
    public String FullPropertyString_Complex
    {
        get
        {
            return _fullPropertyStringComplex;
        }
        set
        {
            _fullPropertyStringComplex = value;
        }
    }

    // Lambda Property
    private String _lamdaPropertyString = "LamdaPropertyString";
    [Export]
    public String LamdaPropertyString
    {
        get => _lamdaPropertyString;
        set => _lamdaPropertyString = value;
    }

    // Auto Property
    [Export] private Boolean PropertyBoolean { get; set; } = true;
    [Export] private Char PropertyChar { get; set; } = 'f';
    [Export] private SByte PropertySByte { get; set; } = 10;
    [Export] private Int16 PropertyInt16 { get; set; } = 10;
    [Export] private Int32 PropertyInt32 { get; set; } = 10;
    [Export] private Int64 PropertyInt64 { get; set; } = 10;
    [Export] private Byte PropertyByte { get; set; } = 10;
    [Export] private UInt16 PropertyUInt16 { get; set; } = 10;
    [Export] private UInt32 PropertyUInt32 { get; set; } = 10;
    [Export] private UInt64 PropertyUInt64 { get; set; } = 10;
    [Export] private Single PropertySingle { get; set; } = 10;
    [Export] private Double PropertyDouble { get; set; } = 10;
    [Export] private String PropertyString { get; set; } = "foo";

    // Godot structs
    [Export] private Vector2 PropertyVector2 { get; set; } = new(10f, 10f);
    [Export] private Vector2I PropertyVector2I { get; set; } = Vector2I.Up;
    [Export] private Rect2 PropertyRect2 { get; set; } = new(new Vector2(10f, 10f), new Vector2(10f, 10f));
    [Export] private Rect2I PropertyRect2I { get; set; } = new(new Vector2I(10, 10), new Vector2I(10, 10));
    [Export] private Transform2D PropertyTransform2D { get; set; } = Transform2D.Identity;
    [Export] private Vector3 PropertyVector3 { get; set; } = new(10f, 10f, 10f);
    [Export] private Vector3I PropertyVector3I { get; set; } = Vector3I.Back;
    [Export] private Basis PropertyBasis { get; set; } = new Basis(Quaternion.Identity);
    [Export] private Quaternion PropertyQuaternion { get; set; } = new Quaternion(Basis.Identity);
    [Export] private Transform3D PropertyTransform3D { get; set; } = Transform3D.Identity;
    [Export] private Vector4 PropertyVector4 { get; set; } = new(10f, 10f, 10f, 10f);
    [Export] private Vector4I PropertyVector4I { get; set; } = Vector4I.One;
    [Export] private Projection PropertyProjection { get; set; } = Projection.Identity;
    [Export] private Aabb PropertyAabb { get; set; } = new Aabb(10f, 10f, 10f, new Vector3(1f, 1f, 1f));
    [Export] private Color PropertyColor { get; set; } = Colors.Aquamarine;
    [Export] private Plane PropertyPlane { get; set; } = Plane.PlaneXZ;
    [Export] private Callable PropertyCallable { get; set; } = new Callable(Engine.GetMainLoop(), "_process");
    [Export] private Signal PropertySignal { get; set; } = new Signal(Engine.GetMainLoop(), "Propertylist_changed");

    // Enums
    public enum MyEnum
    {
        A,
        B,
        C
    }

    [Export] private MyEnum PropertyEnum { get; set; } = MyEnum.C;

    [Flags]
    public enum MyFlagsEnum
    {
        A,
        B,
        C
    }

    [Export] private MyFlagsEnum PropertyFlagsEnum { get; set; } = MyFlagsEnum.C;

    // Arrays
    [Export] private Byte[] PropertyByteArray { get; set; } = { 0, 1, 2, 3, 4, 5, 6 };
    [Export] private Int32[] PropertyInt32Array { get; set; } = { 0, 1, 2, 3, 4, 5, 6 };
    [Export] private Int64[] PropertyInt64Array { get; set; } = { 0, 1, 2, 3, 4, 5, 6 };
    [Export] private Single[] PropertySingleArray { get; set; } = { 0f, 1f, 2f, 3f, 4f, 5f, 6f };
    [Export] private Double[] PropertyDoubleArray { get; set; } = { 0d, 1d, 2d, 3d, 4d, 5d, 6d };
    [Export] private String[] PropertyStringArray { get; set; } = { "foo", "bar" };
    [Export(PropertyHint.Enum, "A,B,C")] private String[] PropertyStringArrayEnum { get; set; } = { "foo", "bar" };
    [Export] private Vector2[] PropertyVector2Array { get; set; } = { Vector2.Up, Vector2.Down, Vector2.Left, Vector2.Right };
    [Export] private Vector3[] PropertyVector3Array { get; set; } = { Vector3.Up, Vector3.Down, Vector3.Left, Vector3.Right };
    [Export] private Color[] PropertyColorArray { get; set; } = { Colors.Aqua, Colors.Aquamarine, Colors.Azure, Colors.Beige };
    [Export] private GodotObject[] PropertyGodotObjectOrDerivedArray { get; set; } = { null };
    [Export] private StringName[] field_StringNameArray { get; set; } = { "foo", "bar" };
    [Export] private NodePath[] field_NodePathArray { get; set; } = { "foo", "bar" };
    [Export] private Rid[] field_RidArray { get; set; } = { default, default, default };

    // Variant
    [Export] private Variant PropertyVariant { get; set; } = "foo";

    // Classes
    [Export] private GodotObject PropertyGodotObjectOrDerived { get; set; }
    [Export] private Godot.Texture PropertyGodotResourceTexture { get; set; }
    [Export] private StringName PropertyStringName { get; set; } = new StringName("foo");
    [Export] private NodePath PropertyNodePath { get; set; } = new NodePath("foo");
    [Export] private Rid PropertyRid { get; set; }

    [Export]
    private Godot.Collections.Dictionary PropertyGodotDictionary { get; set; } = new() { { "foo", 10 }, { Vector2.Up, Colors.Chocolate } };

    [Export]
    private Godot.Collections.Array PropertyGodotArray { get; set; } = new() { "foo", 10, Vector2.Up, Colors.Chocolate };

    [Export]
    private Godot.Collections.Dictionary<string, bool> PropertyGodotGenericDictionary { get; set; } = new() { { "foo", true }, { "bar", false } };

    [Export]
    private Godot.Collections.Array<int> PropertyGodotGenericArray { get; set; } = new() { 0, 1, 2, 3, 4, 5, 6 };
}
