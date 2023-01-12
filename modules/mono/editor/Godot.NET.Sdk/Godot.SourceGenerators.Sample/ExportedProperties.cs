using System;
using System.Diagnostics.CodeAnalysis;

#pragma warning disable CS0169
#pragma warning disable CS0414

namespace Godot.SourceGenerators.Sample
{
    [SuppressMessage("ReSharper", "BuiltInTypeReferenceStyle")]
    [SuppressMessage("ReSharper", "RedundantNameQualifier")]
    [SuppressMessage("ReSharper", "ArrangeObjectCreationWhenTypeEvident")]
    [SuppressMessage("ReSharper", "InconsistentNaming")]
    public partial class ExportedProperties : Godot.Object
    {
        // Do not generate default value
        private String _notGenerate_Property_String = new string("not generate");
        [Export]
        public String NotGenerate_Complex_Lamda_Property
        {
            get => _notGenerate_Property_String + Convert.ToInt32("1");
            set => _notGenerate_Property_String = value;
        }

        [Export]
        public String NotGenerate_Lamda_NoField_Property
        {
            get => new string("not generate");
            set => _notGenerate_Property_String = value;
        }

        [Export]
        public String NotGenerate_Complex_Return_Property
        {
            get
            {
                return _notGenerate_Property_String + Convert.ToInt32("1");
            }
            set
            {
                _notGenerate_Property_String = value;
            }
        }

        private int _notGenerate_Property_Int = 1;
        [Export]
        public string NotGenerate_Returns_Property
        {
            get
            {
                if (_notGenerate_Property_Int == 1)
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
                _notGenerate_Property_Int = value == "a" ? 1 : 2;
            }
        }

        // Full Property
        private String _fullProperty_String = "FullProperty_String";
        [Export]
        public String FullProperty_String
        {
            get
            {
                return _fullProperty_String;
            }
            set
            {
                _fullProperty_String = value;
            }
        }

        private String _fullProperty_String_Complex = new string("FullProperty_String_Complex") + Convert.ToInt32("1");
        [Export]
        public String FullProperty_String_Complex
        {
            get
            {
                return _fullProperty_String_Complex;
            }
            set
            {
                _fullProperty_String_Complex = value;
            }
        }

        // Lambda Property
        private String _lamdaProperty_String = "LamdaProperty_String";
        [Export]
        public String LamdaProperty_String
        {
            get => _lamdaProperty_String;
            set => _lamdaProperty_String = value;
        }

        // Auto Property
        [Export] private Boolean property_Boolean { get; set; } = true;
        [Export] private Char property_Char { get; set; } = 'f';
        [Export] private SByte property_SByte { get; set; } = 10;
        [Export] private Int16 property_Int16 { get; set; } = 10;
        [Export] private Int32 property_Int32 { get; set; } = 10;
        [Export] private Int64 property_Int64 { get; set; } = 10;
        [Export] private Byte property_Byte { get; set; } = 10;
        [Export] private UInt16 property_UInt16 { get; set; } = 10;
        [Export] private UInt32 property_UInt32 { get; set; } = 10;
        [Export] private UInt64 property_UInt64 { get; set; } = 10;
        [Export] private Single property_Single { get; set; } = 10;
        [Export] private Double property_Double { get; set; } = 10;
        [Export] private String property_String { get; set; } = "foo";

        // Godot structs
        [Export] private Vector2 property_Vector2 { get; set; } = new(10f, 10f);
        [Export] private Vector2i property_Vector2i { get; set; } = Vector2i.Up;
        [Export] private Rect2 property_Rect2 { get; set; } = new(new Vector2(10f, 10f), new Vector2(10f, 10f));
        [Export] private Rect2i property_Rect2i { get; set; } = new(new Vector2i(10, 10), new Vector2i(10, 10));
        [Export] private Transform2D property_Transform2D { get; set; } = Transform2D.Identity;
        [Export] private Vector3 property_Vector3 { get; set; } = new(10f, 10f, 10f);
        [Export] private Vector3i property_Vector3i { get; set; } = Vector3i.Back;
        [Export] private Basis property_Basis { get; set; } = new Basis(Quaternion.Identity);
        [Export] private Quaternion property_Quaternion { get; set; } = new Quaternion(Basis.Identity);
        [Export] private Transform3D property_Transform3D { get; set; } = Transform3D.Identity;
        [Export] private Vector4 property_Vector4 { get; set; } = new(10f, 10f, 10f, 10f);
        [Export] private Vector4i property_Vector4i { get; set; } = Vector4i.One;
        [Export] private Projection property_Projection { get; set; } = Projection.Identity;
        [Export] private AABB property_AABB { get; set; } = new AABB(10f, 10f, 10f, new Vector3(1f, 1f, 1f));
        [Export] private Color property_Color { get; set; } = Colors.Aquamarine;
        [Export] private Plane property_Plane { get; set; } = Plane.PlaneXZ;
        [Export] private Callable property_Callable { get; set; } = new Callable(Engine.GetMainLoop(), "_process");
        [Export] private Signal property_Signal { get; set; } = new Signal(Engine.GetMainLoop(), "property_list_changed");

        // Enums
        [SuppressMessage("ReSharper", "UnusedMember.Local")]
        enum MyEnum
        {
            A,
            B,
            C
        }

        [Export] private MyEnum property_Enum { get; set; } = MyEnum.C;

        [Flags]
        [SuppressMessage("ReSharper", "UnusedMember.Local")]
        enum MyFlagsEnum
        {
            A,
            B,
            C
        }

        [Export] private MyFlagsEnum property_FlagsEnum { get; set; } = MyFlagsEnum.C;

        // Arrays
        [Export] private Byte[] property_ByteArray { get; set; } = { 0, 1, 2, 3, 4, 5, 6 };
        [Export] private Int32[] property_Int32Array { get; set; } = { 0, 1, 2, 3, 4, 5, 6 };
        [Export] private Int64[] property_Int64Array { get; set; } = { 0, 1, 2, 3, 4, 5, 6 };
        [Export] private Single[] property_SingleArray { get; set; } = { 0f, 1f, 2f, 3f, 4f, 5f, 6f };
        [Export] private Double[] property_DoubleArray { get; set; } = { 0d, 1d, 2d, 3d, 4d, 5d, 6d };
        [Export] private String[] property_StringArray { get; set; } = { "foo", "bar" };
        [Export(PropertyHint.Enum, "A,B,C")] private String[] property_StringArrayEnum { get; set; } = { "foo", "bar" };
        [Export] private Vector2[] property_Vector2Array { get; set; } = { Vector2.Up, Vector2.Down, Vector2.Left, Vector2.Right };
        [Export] private Vector3[] property_Vector3Array { get; set; } = { Vector3.Up, Vector3.Down, Vector3.Left, Vector3.Right };
        [Export] private Color[] property_ColorArray { get; set; } = { Colors.Aqua, Colors.Aquamarine, Colors.Azure, Colors.Beige };
        [Export] private Godot.Object[] property_GodotObjectOrDerivedArray { get; set; } = { null };
        [Export] private StringName[] field_StringNameArray { get; set; } = { "foo", "bar" };
        [Export] private NodePath[] field_NodePathArray { get; set; } = { "foo", "bar" };
        [Export] private RID[] field_RIDArray { get; set; } = { default, default, default };

        // Variant
        [Export] private Variant property_Variant { get; set; } = "foo";

        // Classes
        [Export] private Godot.Object property_GodotObjectOrDerived { get; set; }
        [Export] private Godot.Texture property_GodotResourceTexture { get; set; }
        [Export] private StringName property_StringName { get; set; } = new StringName("foo");
        [Export] private NodePath property_NodePath { get; set; } = new NodePath("foo");
        [Export] private RID property_RID { get; set; }

        [Export]
        private Godot.Collections.Dictionary property_GodotDictionary { get; set; } =
            new() { { "foo", 10 }, { Vector2.Up, Colors.Chocolate } };

        [Export]
        private Godot.Collections.Array property_GodotArray { get; set; } =
            new() { "foo", 10, Vector2.Up, Colors.Chocolate };

        [Export]
        private Godot.Collections.Dictionary<string, bool> property_GodotGenericDictionary { get; set; } =
            new() { { "foo", true }, { "bar", false } };

        [Export]
        private Godot.Collections.Array<int> property_GodotGenericArray { get; set; } =
            new() { 0, 1, 2, 3, 4, 5, 6 };
    }
}
