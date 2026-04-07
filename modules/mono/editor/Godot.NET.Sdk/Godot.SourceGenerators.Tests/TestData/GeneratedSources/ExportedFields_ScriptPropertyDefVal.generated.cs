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
        var values = new global::System.Collections.Generic.Dictionary<global::Godot.StringName, global::Godot.Variant>(62);
        bool ___fieldBoolean_default_value = true;
        values.Add(PropertyName.@_fieldBoolean, global::Godot.Variant.From<bool>(___fieldBoolean_default_value));
        char ___fieldChar_default_value = 'f';
        values.Add(PropertyName.@_fieldChar, global::Godot.Variant.From<char>(___fieldChar_default_value));
        sbyte ___fieldSByte_default_value = 10;
        values.Add(PropertyName.@_fieldSByte, global::Godot.Variant.From<sbyte>(___fieldSByte_default_value));
        short ___fieldInt16_default_value = 10;
        values.Add(PropertyName.@_fieldInt16, global::Godot.Variant.From<short>(___fieldInt16_default_value));
        int ___fieldInt32_default_value = 10;
        values.Add(PropertyName.@_fieldInt32, global::Godot.Variant.From<int>(___fieldInt32_default_value));
        long ___fieldInt64_default_value = -10_000;
        values.Add(PropertyName.@_fieldInt64, global::Godot.Variant.From<long>(___fieldInt64_default_value));
        byte ___fieldByte_default_value = 10;
        values.Add(PropertyName.@_fieldByte, global::Godot.Variant.From<byte>(___fieldByte_default_value));
        ushort ___fieldUInt16_default_value = 10;
        values.Add(PropertyName.@_fieldUInt16, global::Godot.Variant.From<ushort>(___fieldUInt16_default_value));
        uint ___fieldUInt32_default_value = 10;
        values.Add(PropertyName.@_fieldUInt32, global::Godot.Variant.From<uint>(___fieldUInt32_default_value));
        ulong ___fieldUInt64_default_value = 10;
        values.Add(PropertyName.@_fieldUInt64, global::Godot.Variant.From<ulong>(___fieldUInt64_default_value));
        float ___fieldSingle_default_value = 10;
        values.Add(PropertyName.@_fieldSingle, global::Godot.Variant.From<float>(___fieldSingle_default_value));
        double ___fieldDouble_default_value = 10;
        values.Add(PropertyName.@_fieldDouble, global::Godot.Variant.From<double>(___fieldDouble_default_value));
        string ___fieldString_default_value = "foo";
        values.Add(PropertyName.@_fieldString, global::Godot.Variant.From<string>(___fieldString_default_value));
        float ___fieldStaticImport_default_value = global::Godot.Mathf.RadToDeg(2  * global::Godot.Mathf.Pi);
        values.Add(PropertyName.@_fieldStaticImport, global::Godot.Variant.From<float>(___fieldStaticImport_default_value));
        global::Godot.Vector2 ___fieldVector2_default_value = new(10f, 10f);
        values.Add(PropertyName.@_fieldVector2, global::Godot.Variant.From<global::Godot.Vector2>(___fieldVector2_default_value));
        global::Godot.Vector2I ___fieldVector2I_default_value = global::Godot.Vector2I.Up;
        values.Add(PropertyName.@_fieldVector2I, global::Godot.Variant.From<global::Godot.Vector2I>(___fieldVector2I_default_value));
        global::Godot.Rect2 ___fieldRect2_default_value = new(new global::Godot.Vector2(10f, 10f), new global::Godot.Vector2(10f, 10f));
        values.Add(PropertyName.@_fieldRect2, global::Godot.Variant.From<global::Godot.Rect2>(___fieldRect2_default_value));
        global::Godot.Rect2I ___fieldRect2I_default_value = new(new global::Godot.Vector2I(10, 10), new global::Godot.Vector2I(10, 10));
        values.Add(PropertyName.@_fieldRect2I, global::Godot.Variant.From<global::Godot.Rect2I>(___fieldRect2I_default_value));
        global::Godot.Transform2D ___fieldTransform2D_default_value = global::Godot.Transform2D.Identity;
        values.Add(PropertyName.@_fieldTransform2D, global::Godot.Variant.From<global::Godot.Transform2D>(___fieldTransform2D_default_value));
        global::Godot.Vector3 ___fieldVector3_default_value = new(10f, 10f, 10f);
        values.Add(PropertyName.@_fieldVector3, global::Godot.Variant.From<global::Godot.Vector3>(___fieldVector3_default_value));
        global::Godot.Vector3I ___fieldVector3I_default_value = global::Godot.Vector3I.Back;
        values.Add(PropertyName.@_fieldVector3I, global::Godot.Variant.From<global::Godot.Vector3I>(___fieldVector3I_default_value));
        global::Godot.Basis ___fieldBasis_default_value = new global::Godot.Basis(global::Godot.Quaternion.Identity);
        values.Add(PropertyName.@_fieldBasis, global::Godot.Variant.From<global::Godot.Basis>(___fieldBasis_default_value));
        global::Godot.Quaternion ___fieldQuaternion_default_value = new global::Godot.Quaternion(global::Godot.Basis.Identity);
        values.Add(PropertyName.@_fieldQuaternion, global::Godot.Variant.From<global::Godot.Quaternion>(___fieldQuaternion_default_value));
        global::Godot.Transform3D ___fieldTransform3D_default_value = global::Godot.Transform3D.Identity;
        values.Add(PropertyName.@_fieldTransform3D, global::Godot.Variant.From<global::Godot.Transform3D>(___fieldTransform3D_default_value));
        global::Godot.Vector4 ___fieldVector4_default_value = new(10f, 10f, 10f, 10f);
        values.Add(PropertyName.@_fieldVector4, global::Godot.Variant.From<global::Godot.Vector4>(___fieldVector4_default_value));
        global::Godot.Vector4I ___fieldVector4I_default_value = global::Godot.Vector4I.One;
        values.Add(PropertyName.@_fieldVector4I, global::Godot.Variant.From<global::Godot.Vector4I>(___fieldVector4I_default_value));
        global::Godot.Projection ___fieldProjection_default_value = global::Godot.Projection.Identity;
        values.Add(PropertyName.@_fieldProjection, global::Godot.Variant.From<global::Godot.Projection>(___fieldProjection_default_value));
        global::Godot.Aabb ___fieldAabb_default_value = new global::Godot.Aabb(10f, 10f, 10f, new global::Godot.Vector3(1f, 1f, 1f));
        values.Add(PropertyName.@_fieldAabb, global::Godot.Variant.From<global::Godot.Aabb>(___fieldAabb_default_value));
        global::Godot.Color ___fieldColor_default_value = global::Godot.Colors.Aquamarine;
        values.Add(PropertyName.@_fieldColor, global::Godot.Variant.From<global::Godot.Color>(___fieldColor_default_value));
        global::Godot.Plane ___fieldPlane_default_value = global::Godot.Plane.PlaneXZ;
        values.Add(PropertyName.@_fieldPlane, global::Godot.Variant.From<global::Godot.Plane>(___fieldPlane_default_value));
        global::Godot.Callable ___fieldCallable_default_value = new global::Godot.Callable(global::Godot.Engine.GetMainLoop(), "_process");
        values.Add(PropertyName.@_fieldCallable, global::Godot.Variant.From<global::Godot.Callable>(___fieldCallable_default_value));
        global::Godot.Signal ___fieldSignal_default_value = new global::Godot.Signal(global::Godot.Engine.GetMainLoop(), "property_list_changed");
        values.Add(PropertyName.@_fieldSignal, global::Godot.Variant.From<global::Godot.Signal>(___fieldSignal_default_value));
        global::ExportedFields.MyEnum ___fieldEnum_default_value = global::ExportedFields.MyEnum.C;
        values.Add(PropertyName.@_fieldEnum, global::Godot.Variant.From<global::ExportedFields.MyEnum>(___fieldEnum_default_value));
        global::ExportedFields.MyFlagsEnum ___fieldFlagsEnum_default_value = global::ExportedFields.MyFlagsEnum.C;
        values.Add(PropertyName.@_fieldFlagsEnum, global::Godot.Variant.From<global::ExportedFields.MyFlagsEnum>(___fieldFlagsEnum_default_value));
        byte[] ___fieldByteArray_default_value = { 0, 1, 2, 3, 4, 5, 6  };
        values.Add(PropertyName.@_fieldByteArray, global::Godot.Variant.From<byte[]>(___fieldByteArray_default_value));
        int[] ___fieldInt32Array_default_value = { 0, 1, 2, 3, 4, 5, 6  };
        values.Add(PropertyName.@_fieldInt32Array, global::Godot.Variant.From<int[]>(___fieldInt32Array_default_value));
        long[] ___fieldInt64Array_default_value = { 0, 1, 2, 3, 4, 5, 6  };
        values.Add(PropertyName.@_fieldInt64Array, global::Godot.Variant.From<long[]>(___fieldInt64Array_default_value));
        float[] ___fieldSingleArray_default_value = { 0f, 1f, 2f, 3f, 4f, 5f, 6f  };
        values.Add(PropertyName.@_fieldSingleArray, global::Godot.Variant.From<float[]>(___fieldSingleArray_default_value));
        double[] ___fieldDoubleArray_default_value = { 0d, 1d, 2d, 3d, 4d, 5d, 6d  };
        values.Add(PropertyName.@_fieldDoubleArray, global::Godot.Variant.From<double[]>(___fieldDoubleArray_default_value));
        string[] ___fieldStringArray_default_value = { "foo", "bar"  };
        values.Add(PropertyName.@_fieldStringArray, global::Godot.Variant.From<string[]>(___fieldStringArray_default_value));
        string[] ___fieldStringArrayEnum_default_value = { "foo", "bar"  };
        values.Add(PropertyName.@_fieldStringArrayEnum, global::Godot.Variant.From<string[]>(___fieldStringArrayEnum_default_value));
        global::Godot.Vector2[] ___fieldVector2Array_default_value = { global::Godot.Vector2.Up, global::Godot.Vector2.Down, global::Godot.Vector2.Left, global::Godot.Vector2.Right   };
        values.Add(PropertyName.@_fieldVector2Array, global::Godot.Variant.From<global::Godot.Vector2[]>(___fieldVector2Array_default_value));
        global::Godot.Vector3[] ___fieldVector3Array_default_value = { global::Godot.Vector3.Up, global::Godot.Vector3.Down, global::Godot.Vector3.Left, global::Godot.Vector3.Right   };
        values.Add(PropertyName.@_fieldVector3Array, global::Godot.Variant.From<global::Godot.Vector3[]>(___fieldVector3Array_default_value));
        global::Godot.Color[] ___fieldColorArray_default_value = { global::Godot.Colors.Aqua, global::Godot.Colors.Aquamarine, global::Godot.Colors.Azure, global::Godot.Colors.Beige   };
        values.Add(PropertyName.@_fieldColorArray, global::Godot.Variant.From<global::Godot.Color[]>(___fieldColorArray_default_value));
        global::Godot.GodotObject[] ___fieldGodotObjectOrDerivedArray_default_value = { null  };
        values.Add(PropertyName.@_fieldGodotObjectOrDerivedArray, global::Godot.Variant.CreateFrom(___fieldGodotObjectOrDerivedArray_default_value));
        global::Godot.StringName[] ___fieldStringNameArray_default_value = { "foo", "bar"  };
        values.Add(PropertyName.@_fieldStringNameArray, global::Godot.Variant.From<global::Godot.StringName[]>(___fieldStringNameArray_default_value));
        global::Godot.NodePath[] ___fieldNodePathArray_default_value = { "foo", "bar"  };
        values.Add(PropertyName.@_fieldNodePathArray, global::Godot.Variant.From<global::Godot.NodePath[]>(___fieldNodePathArray_default_value));
        global::Godot.Rid[] ___fieldRidArray_default_value = { default, default, default  };
        values.Add(PropertyName.@_fieldRidArray, global::Godot.Variant.From<global::Godot.Rid[]>(___fieldRidArray_default_value));
        int[] ___fieldEmptyInt32Array_default_value = global::System.Array.Empty<int>();
        values.Add(PropertyName.@_fieldEmptyInt32Array, global::Godot.Variant.From<int[]>(___fieldEmptyInt32Array_default_value));
        int[] ___fieldArrayFromList_default_value = new global::System.Collections.Generic.List<int>(global::System.Array.Empty<int>()).ToArray();
        values.Add(PropertyName.@_fieldArrayFromList, global::Godot.Variant.From<int[]>(___fieldArrayFromList_default_value));
        global::Godot.Variant ___fieldVariant_default_value = "foo";
        values.Add(PropertyName.@_fieldVariant, global::Godot.Variant.From<global::Godot.Variant>(___fieldVariant_default_value));
        global::Godot.GodotObject ___fieldGodotObjectOrDerived_default_value = default;
        values.Add(PropertyName.@_fieldGodotObjectOrDerived, global::Godot.Variant.From<global::Godot.GodotObject>(___fieldGodotObjectOrDerived_default_value));
        global::Godot.Texture ___fieldGodotResourceTexture_default_value = default;
        values.Add(PropertyName.@_fieldGodotResourceTexture, global::Godot.Variant.From<global::Godot.Texture>(___fieldGodotResourceTexture_default_value));
        global::Godot.Texture ___fieldGodotResourceTextureWithInitializer_default_value = new()  { ResourceName  = ""   };
        values.Add(PropertyName.@_fieldGodotResourceTextureWithInitializer, global::Godot.Variant.From<global::Godot.Texture>(___fieldGodotResourceTextureWithInitializer_default_value));
        global::Godot.StringName ___fieldStringName_default_value = new global::Godot.StringName("foo");
        values.Add(PropertyName.@_fieldStringName, global::Godot.Variant.From<global::Godot.StringName>(___fieldStringName_default_value));
        global::Godot.NodePath ___fieldNodePath_default_value = new global::Godot.NodePath("foo");
        values.Add(PropertyName.@_fieldNodePath, global::Godot.Variant.From<global::Godot.NodePath>(___fieldNodePath_default_value));
        global::Godot.Rid ___fieldRid_default_value = default;
        values.Add(PropertyName.@_fieldRid, global::Godot.Variant.From<global::Godot.Rid>(___fieldRid_default_value));
        global::Godot.Collections.Dictionary ___fieldGodotDictionary_default_value = new()  { { "foo", 10  }, { global::Godot.Vector2.Up, global::Godot.Colors.Chocolate   }  };
        values.Add(PropertyName.@_fieldGodotDictionary, global::Godot.Variant.From<global::Godot.Collections.Dictionary>(___fieldGodotDictionary_default_value));
        global::Godot.Collections.Array ___fieldGodotArray_default_value = new()  { "foo", 10, global::Godot.Vector2.Up, global::Godot.Colors.Chocolate   };
        values.Add(PropertyName.@_fieldGodotArray, global::Godot.Variant.From<global::Godot.Collections.Array>(___fieldGodotArray_default_value));
        global::Godot.Collections.Dictionary<string, bool> ___fieldGodotGenericDictionary_default_value = new()  { { "foo", true  }, { "bar", false  }  };
        values.Add(PropertyName.@_fieldGodotGenericDictionary, global::Godot.Variant.CreateFrom(___fieldGodotGenericDictionary_default_value));
        global::Godot.Collections.Array<int> ___fieldGodotGenericArray_default_value = new()  { 0, 1, 2, 3, 4, 5, 6  };
        values.Add(PropertyName.@_fieldGodotGenericArray, global::Godot.Variant.CreateFrom(___fieldGodotGenericArray_default_value));
        long[] ___fieldEmptyInt64Array_default_value = global::System.Array.Empty<long>();
        values.Add(PropertyName.@_fieldEmptyInt64Array, global::Godot.Variant.From<long[]>(___fieldEmptyInt64Array_default_value));
        return values;
    }
#endif // TOOLS
#pragma warning restore CS0109
}
