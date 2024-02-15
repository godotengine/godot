partial class ExportedFields
{
#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword
#if TOOLS
    /// <summary>
    /// Get the default values for all properties declared in this class.
    /// This method is used by Godot to determine the value that will be
    /// used by the inspector when resetting properties.
    /// Do not call this method.
    /// </summary>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    internal new static global::System.Collections.Generic.Dictionary<global::Godot.StringName, global::Godot.Variant> GetGodotPropertyDefaultValues()
    {
        var values = new global::System.Collections.Generic.Dictionary<global::Godot.StringName, global::Godot.Variant>(60);
        bool __field_Boolean_default_value = true;
        values.Add(PropertyName.field_Boolean, global::Godot.Variant.From<bool>(__field_Boolean_default_value));
        char __field_Char_default_value = 'f';
        values.Add(PropertyName.field_Char, global::Godot.Variant.From<char>(__field_Char_default_value));
        sbyte __field_SByte_default_value = 10;
        values.Add(PropertyName.field_SByte, global::Godot.Variant.From<sbyte>(__field_SByte_default_value));
        short __field_Int16_default_value = 10;
        values.Add(PropertyName.field_Int16, global::Godot.Variant.From<short>(__field_Int16_default_value));
        int __field_Int32_default_value = 10;
        values.Add(PropertyName.field_Int32, global::Godot.Variant.From<int>(__field_Int32_default_value));
        long __field_Int64_default_value = 10;
        values.Add(PropertyName.field_Int64, global::Godot.Variant.From<long>(__field_Int64_default_value));
        byte __field_Byte_default_value = 10;
        values.Add(PropertyName.field_Byte, global::Godot.Variant.From<byte>(__field_Byte_default_value));
        ushort __field_UInt16_default_value = 10;
        values.Add(PropertyName.field_UInt16, global::Godot.Variant.From<ushort>(__field_UInt16_default_value));
        uint __field_UInt32_default_value = 10;
        values.Add(PropertyName.field_UInt32, global::Godot.Variant.From<uint>(__field_UInt32_default_value));
        ulong __field_UInt64_default_value = 10;
        values.Add(PropertyName.field_UInt64, global::Godot.Variant.From<ulong>(__field_UInt64_default_value));
        float __field_Single_default_value = 10;
        values.Add(PropertyName.field_Single, global::Godot.Variant.From<float>(__field_Single_default_value));
        double __field_Double_default_value = 10;
        values.Add(PropertyName.field_Double, global::Godot.Variant.From<double>(__field_Double_default_value));
        string __field_String_default_value = "foo";
        values.Add(PropertyName.field_String, global::Godot.Variant.From<string>(__field_String_default_value));
        global::Godot.Vector2 __field_Vector2_default_value = new(10f, 10f);
        values.Add(PropertyName.field_Vector2, global::Godot.Variant.From<global::Godot.Vector2>(__field_Vector2_default_value));
        global::Godot.Vector2I __field_Vector2I_default_value = global::Godot.Vector2I.Up;
        values.Add(PropertyName.field_Vector2I, global::Godot.Variant.From<global::Godot.Vector2I>(__field_Vector2I_default_value));
        global::Godot.Rect2 __field_Rect2_default_value = new(new global::Godot.Vector2(10f, 10f), new global::Godot.Vector2(10f, 10f));
        values.Add(PropertyName.field_Rect2, global::Godot.Variant.From<global::Godot.Rect2>(__field_Rect2_default_value));
        global::Godot.Rect2I __field_Rect2I_default_value = new(new global::Godot.Vector2I(10, 10), new global::Godot.Vector2I(10, 10));
        values.Add(PropertyName.field_Rect2I, global::Godot.Variant.From<global::Godot.Rect2I>(__field_Rect2I_default_value));
        global::Godot.Transform2D __field_Transform2D_default_value = global::Godot.Transform2D.Identity;
        values.Add(PropertyName.field_Transform2D, global::Godot.Variant.From<global::Godot.Transform2D>(__field_Transform2D_default_value));
        global::Godot.Vector3 __field_Vector3_default_value = new(10f, 10f, 10f);
        values.Add(PropertyName.field_Vector3, global::Godot.Variant.From<global::Godot.Vector3>(__field_Vector3_default_value));
        global::Godot.Vector3I __field_Vector3I_default_value = global::Godot.Vector3I.Back;
        values.Add(PropertyName.field_Vector3I, global::Godot.Variant.From<global::Godot.Vector3I>(__field_Vector3I_default_value));
        global::Godot.Basis __field_Basis_default_value = new global::Godot.Basis(global::Godot.Quaternion.Identity);
        values.Add(PropertyName.field_Basis, global::Godot.Variant.From<global::Godot.Basis>(__field_Basis_default_value));
        global::Godot.Quaternion __field_Quaternion_default_value = new global::Godot.Quaternion(global::Godot.Basis.Identity);
        values.Add(PropertyName.field_Quaternion, global::Godot.Variant.From<global::Godot.Quaternion>(__field_Quaternion_default_value));
        global::Godot.Transform3D __field_Transform3D_default_value = global::Godot.Transform3D.Identity;
        values.Add(PropertyName.field_Transform3D, global::Godot.Variant.From<global::Godot.Transform3D>(__field_Transform3D_default_value));
        global::Godot.Vector4 __field_Vector4_default_value = new(10f, 10f, 10f, 10f);
        values.Add(PropertyName.field_Vector4, global::Godot.Variant.From<global::Godot.Vector4>(__field_Vector4_default_value));
        global::Godot.Vector4I __field_Vector4I_default_value = global::Godot.Vector4I.One;
        values.Add(PropertyName.field_Vector4I, global::Godot.Variant.From<global::Godot.Vector4I>(__field_Vector4I_default_value));
        global::Godot.Projection __field_Projection_default_value = global::Godot.Projection.Identity;
        values.Add(PropertyName.field_Projection, global::Godot.Variant.From<global::Godot.Projection>(__field_Projection_default_value));
        global::Godot.Aabb __field_Aabb_default_value = new global::Godot.Aabb(10f, 10f, 10f, new global::Godot.Vector3(1f, 1f, 1f));
        values.Add(PropertyName.field_Aabb, global::Godot.Variant.From<global::Godot.Aabb>(__field_Aabb_default_value));
        global::Godot.Color __field_Color_default_value = global::Godot.Colors.Aquamarine;
        values.Add(PropertyName.field_Color, global::Godot.Variant.From<global::Godot.Color>(__field_Color_default_value));
        global::Godot.Plane __field_Plane_default_value = global::Godot.Plane.PlaneXZ;
        values.Add(PropertyName.field_Plane, global::Godot.Variant.From<global::Godot.Plane>(__field_Plane_default_value));
        global::Godot.Callable __field_Callable_default_value = new global::Godot.Callable(global::Godot.Engine.GetMainLoop(), "_process");
        values.Add(PropertyName.field_Callable, global::Godot.Variant.From<global::Godot.Callable>(__field_Callable_default_value));
        global::Godot.Signal __field_Signal_default_value = new global::Godot.Signal(global::Godot.Engine.GetMainLoop(), "property_list_changed");
        values.Add(PropertyName.field_Signal, global::Godot.Variant.From<global::Godot.Signal>(__field_Signal_default_value));
        global::ExportedFields.MyEnum __field_Enum_default_value = global::ExportedFields.MyEnum.C;
        values.Add(PropertyName.field_Enum, global::Godot.Variant.From<global::ExportedFields.MyEnum>(__field_Enum_default_value));
        global::ExportedFields.MyFlagsEnum __field_FlagsEnum_default_value = global::ExportedFields.MyFlagsEnum.C;
        values.Add(PropertyName.field_FlagsEnum, global::Godot.Variant.From<global::ExportedFields.MyFlagsEnum>(__field_FlagsEnum_default_value));
        byte[] __field_ByteArray_default_value = { 0, 1, 2, 3, 4, 5, 6  };
        values.Add(PropertyName.field_ByteArray, global::Godot.Variant.From<byte[]>(__field_ByteArray_default_value));
        int[] __field_Int32Array_default_value = { 0, 1, 2, 3, 4, 5, 6  };
        values.Add(PropertyName.field_Int32Array, global::Godot.Variant.From<int[]>(__field_Int32Array_default_value));
        long[] __field_Int64Array_default_value = { 0, 1, 2, 3, 4, 5, 6  };
        values.Add(PropertyName.field_Int64Array, global::Godot.Variant.From<long[]>(__field_Int64Array_default_value));
        float[] __field_SingleArray_default_value = { 0f, 1f, 2f, 3f, 4f, 5f, 6f  };
        values.Add(PropertyName.field_SingleArray, global::Godot.Variant.From<float[]>(__field_SingleArray_default_value));
        double[] __field_DoubleArray_default_value = { 0d, 1d, 2d, 3d, 4d, 5d, 6d  };
        values.Add(PropertyName.field_DoubleArray, global::Godot.Variant.From<double[]>(__field_DoubleArray_default_value));
        string[] __field_StringArray_default_value = { "foo", "bar"  };
        values.Add(PropertyName.field_StringArray, global::Godot.Variant.From<string[]>(__field_StringArray_default_value));
        string[] __field_StringArrayEnum_default_value = { "foo", "bar"  };
        values.Add(PropertyName.field_StringArrayEnum, global::Godot.Variant.From<string[]>(__field_StringArrayEnum_default_value));
        global::Godot.Vector2[] __field_Vector2Array_default_value = { global::Godot.Vector2.Up, global::Godot.Vector2.Down, global::Godot.Vector2.Left, global::Godot.Vector2.Right   };
        values.Add(PropertyName.field_Vector2Array, global::Godot.Variant.From<global::Godot.Vector2[]>(__field_Vector2Array_default_value));
        global::Godot.Vector3[] __field_Vector3Array_default_value = { global::Godot.Vector3.Up, global::Godot.Vector3.Down, global::Godot.Vector3.Left, global::Godot.Vector3.Right   };
        values.Add(PropertyName.field_Vector3Array, global::Godot.Variant.From<global::Godot.Vector3[]>(__field_Vector3Array_default_value));
        global::Godot.Color[] __field_ColorArray_default_value = { global::Godot.Colors.Aqua, global::Godot.Colors.Aquamarine, global::Godot.Colors.Azure, global::Godot.Colors.Beige   };
        values.Add(PropertyName.field_ColorArray, global::Godot.Variant.From<global::Godot.Color[]>(__field_ColorArray_default_value));
        global::Godot.GodotObject[] __field_GodotObjectOrDerivedArray_default_value = { null  };
        values.Add(PropertyName.field_GodotObjectOrDerivedArray, global::Godot.Variant.CreateFrom(__field_GodotObjectOrDerivedArray_default_value));
        global::Godot.StringName[] __field_StringNameArray_default_value = { "foo", "bar"  };
        values.Add(PropertyName.field_StringNameArray, global::Godot.Variant.From<global::Godot.StringName[]>(__field_StringNameArray_default_value));
        global::Godot.NodePath[] __field_NodePathArray_default_value = { "foo", "bar"  };
        values.Add(PropertyName.field_NodePathArray, global::Godot.Variant.From<global::Godot.NodePath[]>(__field_NodePathArray_default_value));
        global::Godot.Rid[] __field_RidArray_default_value = { default, default, default  };
        values.Add(PropertyName.field_RidArray, global::Godot.Variant.From<global::Godot.Rid[]>(__field_RidArray_default_value));
        int[] __field_empty_Int32Array_default_value = global::System.Array.Empty<int>();
        values.Add(PropertyName.field_empty_Int32Array, global::Godot.Variant.From<int[]>(__field_empty_Int32Array_default_value));
        int[] __field_array_from_list_default_value = new global::System.Collections.Generic.List<int>(global::System.Array.Empty<int>()).ToArray();
        values.Add(PropertyName.field_array_from_list, global::Godot.Variant.From<int[]>(__field_array_from_list_default_value));
        global::Godot.Variant __field_Variant_default_value = "foo";
        values.Add(PropertyName.field_Variant, global::Godot.Variant.From<global::Godot.Variant>(__field_Variant_default_value));
        global::Godot.GodotObject __field_GodotObjectOrDerived_default_value = default;
        values.Add(PropertyName.field_GodotObjectOrDerived, global::Godot.Variant.From<global::Godot.GodotObject>(__field_GodotObjectOrDerived_default_value));
        global::Godot.Texture __field_GodotResourceTexture_default_value = default;
        values.Add(PropertyName.field_GodotResourceTexture, global::Godot.Variant.From<global::Godot.Texture>(__field_GodotResourceTexture_default_value));
        global::Godot.StringName __field_StringName_default_value = new global::Godot.StringName("foo");
        values.Add(PropertyName.field_StringName, global::Godot.Variant.From<global::Godot.StringName>(__field_StringName_default_value));
        global::Godot.NodePath __field_NodePath_default_value = new global::Godot.NodePath("foo");
        values.Add(PropertyName.field_NodePath, global::Godot.Variant.From<global::Godot.NodePath>(__field_NodePath_default_value));
        global::Godot.Rid __field_Rid_default_value = default;
        values.Add(PropertyName.field_Rid, global::Godot.Variant.From<global::Godot.Rid>(__field_Rid_default_value));
        global::Godot.Collections.Dictionary __field_GodotDictionary_default_value = new()  { { "foo", 10  }, { global::Godot.Vector2.Up, global::Godot.Colors.Chocolate   }  };
        values.Add(PropertyName.field_GodotDictionary, global::Godot.Variant.From<global::Godot.Collections.Dictionary>(__field_GodotDictionary_default_value));
        global::Godot.Collections.Array __field_GodotArray_default_value = new()  { "foo", 10, global::Godot.Vector2.Up, global::Godot.Colors.Chocolate   };
        values.Add(PropertyName.field_GodotArray, global::Godot.Variant.From<global::Godot.Collections.Array>(__field_GodotArray_default_value));
        global::Godot.Collections.Dictionary<string, bool> __field_GodotGenericDictionary_default_value = new()  { { "foo", true  }, { "bar", false  }  };
        values.Add(PropertyName.field_GodotGenericDictionary, global::Godot.Variant.CreateFrom(__field_GodotGenericDictionary_default_value));
        global::Godot.Collections.Array<int> __field_GodotGenericArray_default_value = new()  { 0, 1, 2, 3, 4, 5, 6  };
        values.Add(PropertyName.field_GodotGenericArray, global::Godot.Variant.CreateFrom(__field_GodotGenericArray_default_value));
        long[] __field_empty_Int64Array_default_value = global::System.Array.Empty<long>();
        values.Add(PropertyName.field_empty_Int64Array, global::Godot.Variant.From<long[]>(__field_empty_Int64Array_default_value));
        return values;
    }
#endif // TOOLS
#pragma warning restore CS0109
}
