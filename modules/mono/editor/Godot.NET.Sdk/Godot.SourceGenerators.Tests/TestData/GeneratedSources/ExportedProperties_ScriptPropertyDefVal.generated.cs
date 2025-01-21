partial class ExportedProperties
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
        var values = new global::System.Collections.Generic.Dictionary<global::Godot.StringName, global::Godot.Variant>(64);
        string __NotGenerateComplexLamdaProperty_default_value = default;
        values.Add(PropertyName.@NotGenerateComplexLamdaProperty, global::Godot.Variant.From<string>(__NotGenerateComplexLamdaProperty_default_value));
        string __NotGenerateLamdaNoFieldProperty_default_value = default;
        values.Add(PropertyName.@NotGenerateLamdaNoFieldProperty, global::Godot.Variant.From<string>(__NotGenerateLamdaNoFieldProperty_default_value));
        string __NotGenerateComplexReturnProperty_default_value = default;
        values.Add(PropertyName.@NotGenerateComplexReturnProperty, global::Godot.Variant.From<string>(__NotGenerateComplexReturnProperty_default_value));
        string __NotGenerateReturnsProperty_default_value = default;
        values.Add(PropertyName.@NotGenerateReturnsProperty, global::Godot.Variant.From<string>(__NotGenerateReturnsProperty_default_value));
        string __FullPropertyString_default_value = "FullPropertyString";
        values.Add(PropertyName.@FullPropertyString, global::Godot.Variant.From<string>(__FullPropertyString_default_value));
        string __FullPropertyString_Complex_default_value = new string("FullPropertyString_Complex")   + global::System.Convert.ToInt32("1");
        values.Add(PropertyName.@FullPropertyString_Complex, global::Godot.Variant.From<string>(__FullPropertyString_Complex_default_value));
        string __LamdaPropertyString_default_value = "LamdaPropertyString";
        values.Add(PropertyName.@LamdaPropertyString, global::Godot.Variant.From<string>(__LamdaPropertyString_default_value));
        bool __PropertyBoolean_default_value = true;
        values.Add(PropertyName.@PropertyBoolean, global::Godot.Variant.From<bool>(__PropertyBoolean_default_value));
        char __PropertyChar_default_value = 'f';
        values.Add(PropertyName.@PropertyChar, global::Godot.Variant.From<char>(__PropertyChar_default_value));
        sbyte __PropertySByte_default_value = 10;
        values.Add(PropertyName.@PropertySByte, global::Godot.Variant.From<sbyte>(__PropertySByte_default_value));
        short __PropertyInt16_default_value = 10;
        values.Add(PropertyName.@PropertyInt16, global::Godot.Variant.From<short>(__PropertyInt16_default_value));
        int __PropertyInt32_default_value = 10;
        values.Add(PropertyName.@PropertyInt32, global::Godot.Variant.From<int>(__PropertyInt32_default_value));
        long __PropertyInt64_default_value = 10;
        values.Add(PropertyName.@PropertyInt64, global::Godot.Variant.From<long>(__PropertyInt64_default_value));
        byte __PropertyByte_default_value = 10;
        values.Add(PropertyName.@PropertyByte, global::Godot.Variant.From<byte>(__PropertyByte_default_value));
        ushort __PropertyUInt16_default_value = 10;
        values.Add(PropertyName.@PropertyUInt16, global::Godot.Variant.From<ushort>(__PropertyUInt16_default_value));
        uint __PropertyUInt32_default_value = 10;
        values.Add(PropertyName.@PropertyUInt32, global::Godot.Variant.From<uint>(__PropertyUInt32_default_value));
        ulong __PropertyUInt64_default_value = 10;
        values.Add(PropertyName.@PropertyUInt64, global::Godot.Variant.From<ulong>(__PropertyUInt64_default_value));
        float __PropertySingle_default_value = 10;
        values.Add(PropertyName.@PropertySingle, global::Godot.Variant.From<float>(__PropertySingle_default_value));
        double __PropertyDouble_default_value = 10;
        values.Add(PropertyName.@PropertyDouble, global::Godot.Variant.From<double>(__PropertyDouble_default_value));
        string __PropertyString_default_value = "foo";
        values.Add(PropertyName.@PropertyString, global::Godot.Variant.From<string>(__PropertyString_default_value));
        global::Godot.Vector2 __PropertyVector2_default_value = new(10f, 10f);
        values.Add(PropertyName.@PropertyVector2, global::Godot.Variant.From<global::Godot.Vector2>(__PropertyVector2_default_value));
        global::Godot.Vector2I __PropertyVector2I_default_value = global::Godot.Vector2I.Up;
        values.Add(PropertyName.@PropertyVector2I, global::Godot.Variant.From<global::Godot.Vector2I>(__PropertyVector2I_default_value));
        global::Godot.Rect2 __PropertyRect2_default_value = new(new global::Godot.Vector2(10f, 10f), new global::Godot.Vector2(10f, 10f));
        values.Add(PropertyName.@PropertyRect2, global::Godot.Variant.From<global::Godot.Rect2>(__PropertyRect2_default_value));
        global::Godot.Rect2I __PropertyRect2I_default_value = new(new global::Godot.Vector2I(10, 10), new global::Godot.Vector2I(10, 10));
        values.Add(PropertyName.@PropertyRect2I, global::Godot.Variant.From<global::Godot.Rect2I>(__PropertyRect2I_default_value));
        global::Godot.Transform2D __PropertyTransform2D_default_value = global::Godot.Transform2D.Identity;
        values.Add(PropertyName.@PropertyTransform2D, global::Godot.Variant.From<global::Godot.Transform2D>(__PropertyTransform2D_default_value));
        global::Godot.Vector3 __PropertyVector3_default_value = new(10f, 10f, 10f);
        values.Add(PropertyName.@PropertyVector3, global::Godot.Variant.From<global::Godot.Vector3>(__PropertyVector3_default_value));
        global::Godot.Vector3I __PropertyVector3I_default_value = global::Godot.Vector3I.Back;
        values.Add(PropertyName.@PropertyVector3I, global::Godot.Variant.From<global::Godot.Vector3I>(__PropertyVector3I_default_value));
        global::Godot.Basis __PropertyBasis_default_value = new global::Godot.Basis(global::Godot.Quaternion.Identity);
        values.Add(PropertyName.@PropertyBasis, global::Godot.Variant.From<global::Godot.Basis>(__PropertyBasis_default_value));
        global::Godot.Quaternion __PropertyQuaternion_default_value = new global::Godot.Quaternion(global::Godot.Basis.Identity);
        values.Add(PropertyName.@PropertyQuaternion, global::Godot.Variant.From<global::Godot.Quaternion>(__PropertyQuaternion_default_value));
        global::Godot.Transform3D __PropertyTransform3D_default_value = global::Godot.Transform3D.Identity;
        values.Add(PropertyName.@PropertyTransform3D, global::Godot.Variant.From<global::Godot.Transform3D>(__PropertyTransform3D_default_value));
        global::Godot.Vector4 __PropertyVector4_default_value = new(10f, 10f, 10f, 10f);
        values.Add(PropertyName.@PropertyVector4, global::Godot.Variant.From<global::Godot.Vector4>(__PropertyVector4_default_value));
        global::Godot.Vector4I __PropertyVector4I_default_value = global::Godot.Vector4I.One;
        values.Add(PropertyName.@PropertyVector4I, global::Godot.Variant.From<global::Godot.Vector4I>(__PropertyVector4I_default_value));
        global::Godot.Projection __PropertyProjection_default_value = global::Godot.Projection.Identity;
        values.Add(PropertyName.@PropertyProjection, global::Godot.Variant.From<global::Godot.Projection>(__PropertyProjection_default_value));
        global::Godot.Aabb __PropertyAabb_default_value = new global::Godot.Aabb(10f, 10f, 10f, new global::Godot.Vector3(1f, 1f, 1f));
        values.Add(PropertyName.@PropertyAabb, global::Godot.Variant.From<global::Godot.Aabb>(__PropertyAabb_default_value));
        global::Godot.Color __PropertyColor_default_value = global::Godot.Colors.Aquamarine;
        values.Add(PropertyName.@PropertyColor, global::Godot.Variant.From<global::Godot.Color>(__PropertyColor_default_value));
        global::Godot.Plane __PropertyPlane_default_value = global::Godot.Plane.PlaneXZ;
        values.Add(PropertyName.@PropertyPlane, global::Godot.Variant.From<global::Godot.Plane>(__PropertyPlane_default_value));
        global::Godot.Callable __PropertyCallable_default_value = new global::Godot.Callable(global::Godot.Engine.GetMainLoop(), "_process");
        values.Add(PropertyName.@PropertyCallable, global::Godot.Variant.From<global::Godot.Callable>(__PropertyCallable_default_value));
        global::Godot.Signal __PropertySignal_default_value = new global::Godot.Signal(global::Godot.Engine.GetMainLoop(), "Propertylist_changed");
        values.Add(PropertyName.@PropertySignal, global::Godot.Variant.From<global::Godot.Signal>(__PropertySignal_default_value));
        global::ExportedProperties.MyEnum __PropertyEnum_default_value = global::ExportedProperties.MyEnum.C;
        values.Add(PropertyName.@PropertyEnum, global::Godot.Variant.From<global::ExportedProperties.MyEnum>(__PropertyEnum_default_value));
        global::ExportedProperties.MyFlagsEnum __PropertyFlagsEnum_default_value = global::ExportedProperties.MyFlagsEnum.C;
        values.Add(PropertyName.@PropertyFlagsEnum, global::Godot.Variant.From<global::ExportedProperties.MyFlagsEnum>(__PropertyFlagsEnum_default_value));
        byte[] __PropertyByteArray_default_value = { 0, 1, 2, 3, 4, 5, 6  };
        values.Add(PropertyName.@PropertyByteArray, global::Godot.Variant.From<byte[]>(__PropertyByteArray_default_value));
        int[] __PropertyInt32Array_default_value = { 0, 1, 2, 3, 4, 5, 6  };
        values.Add(PropertyName.@PropertyInt32Array, global::Godot.Variant.From<int[]>(__PropertyInt32Array_default_value));
        long[] __PropertyInt64Array_default_value = { 0, 1, 2, 3, 4, 5, 6  };
        values.Add(PropertyName.@PropertyInt64Array, global::Godot.Variant.From<long[]>(__PropertyInt64Array_default_value));
        float[] __PropertySingleArray_default_value = { 0f, 1f, 2f, 3f, 4f, 5f, 6f  };
        values.Add(PropertyName.@PropertySingleArray, global::Godot.Variant.From<float[]>(__PropertySingleArray_default_value));
        double[] __PropertyDoubleArray_default_value = { 0d, 1d, 2d, 3d, 4d, 5d, 6d  };
        values.Add(PropertyName.@PropertyDoubleArray, global::Godot.Variant.From<double[]>(__PropertyDoubleArray_default_value));
        string[] __PropertyStringArray_default_value = { "foo", "bar"  };
        values.Add(PropertyName.@PropertyStringArray, global::Godot.Variant.From<string[]>(__PropertyStringArray_default_value));
        string[] __PropertyStringArrayEnum_default_value = { "foo", "bar"  };
        values.Add(PropertyName.@PropertyStringArrayEnum, global::Godot.Variant.From<string[]>(__PropertyStringArrayEnum_default_value));
        global::Godot.Vector2[] __PropertyVector2Array_default_value = { global::Godot.Vector2.Up, global::Godot.Vector2.Down, global::Godot.Vector2.Left, global::Godot.Vector2.Right   };
        values.Add(PropertyName.@PropertyVector2Array, global::Godot.Variant.From<global::Godot.Vector2[]>(__PropertyVector2Array_default_value));
        global::Godot.Vector3[] __PropertyVector3Array_default_value = { global::Godot.Vector3.Up, global::Godot.Vector3.Down, global::Godot.Vector3.Left, global::Godot.Vector3.Right   };
        values.Add(PropertyName.@PropertyVector3Array, global::Godot.Variant.From<global::Godot.Vector3[]>(__PropertyVector3Array_default_value));
        global::Godot.Color[] __PropertyColorArray_default_value = { global::Godot.Colors.Aqua, global::Godot.Colors.Aquamarine, global::Godot.Colors.Azure, global::Godot.Colors.Beige   };
        values.Add(PropertyName.@PropertyColorArray, global::Godot.Variant.From<global::Godot.Color[]>(__PropertyColorArray_default_value));
        global::Godot.GodotObject[] __PropertyGodotObjectOrDerivedArray_default_value = { null  };
        values.Add(PropertyName.@PropertyGodotObjectOrDerivedArray, global::Godot.Variant.CreateFrom(__PropertyGodotObjectOrDerivedArray_default_value));
        global::Godot.StringName[] __field_StringNameArray_default_value = { "foo", "bar"  };
        values.Add(PropertyName.@field_StringNameArray, global::Godot.Variant.From<global::Godot.StringName[]>(__field_StringNameArray_default_value));
        global::Godot.NodePath[] __field_NodePathArray_default_value = { "foo", "bar"  };
        values.Add(PropertyName.@field_NodePathArray, global::Godot.Variant.From<global::Godot.NodePath[]>(__field_NodePathArray_default_value));
        global::Godot.Rid[] __field_RidArray_default_value = { default, default, default  };
        values.Add(PropertyName.@field_RidArray, global::Godot.Variant.From<global::Godot.Rid[]>(__field_RidArray_default_value));
        global::Godot.Variant __PropertyVariant_default_value = "foo";
        values.Add(PropertyName.@PropertyVariant, global::Godot.Variant.From<global::Godot.Variant>(__PropertyVariant_default_value));
        global::Godot.GodotObject __PropertyGodotObjectOrDerived_default_value = default;
        values.Add(PropertyName.@PropertyGodotObjectOrDerived, global::Godot.Variant.From<global::Godot.GodotObject>(__PropertyGodotObjectOrDerived_default_value));
        global::Godot.Texture __PropertyGodotResourceTexture_default_value = default;
        values.Add(PropertyName.@PropertyGodotResourceTexture, global::Godot.Variant.From<global::Godot.Texture>(__PropertyGodotResourceTexture_default_value));
        global::Godot.StringName __PropertyStringName_default_value = new global::Godot.StringName("foo");
        values.Add(PropertyName.@PropertyStringName, global::Godot.Variant.From<global::Godot.StringName>(__PropertyStringName_default_value));
        global::Godot.NodePath __PropertyNodePath_default_value = new global::Godot.NodePath("foo");
        values.Add(PropertyName.@PropertyNodePath, global::Godot.Variant.From<global::Godot.NodePath>(__PropertyNodePath_default_value));
        global::Godot.Rid __PropertyRid_default_value = default;
        values.Add(PropertyName.@PropertyRid, global::Godot.Variant.From<global::Godot.Rid>(__PropertyRid_default_value));
        global::Godot.Collections.Dictionary __PropertyGodotDictionary_default_value = new()  { { "foo", 10  }, { global::Godot.Vector2.Up, global::Godot.Colors.Chocolate   }  };
        values.Add(PropertyName.@PropertyGodotDictionary, global::Godot.Variant.From<global::Godot.Collections.Dictionary>(__PropertyGodotDictionary_default_value));
        global::Godot.Collections.Array __PropertyGodotArray_default_value = new()  { "foo", 10, global::Godot.Vector2.Up, global::Godot.Colors.Chocolate   };
        values.Add(PropertyName.@PropertyGodotArray, global::Godot.Variant.From<global::Godot.Collections.Array>(__PropertyGodotArray_default_value));
        global::Godot.Collections.Dictionary<string, bool> __PropertyGodotGenericDictionary_default_value = new()  { { "foo", true  }, { "bar", false  }  };
        values.Add(PropertyName.@PropertyGodotGenericDictionary, global::Godot.Variant.CreateFrom(__PropertyGodotGenericDictionary_default_value));
        global::Godot.Collections.Array<int> __PropertyGodotGenericArray_default_value = new()  { 0, 1, 2, 3, 4, 5, 6  };
        values.Add(PropertyName.@PropertyGodotGenericArray, global::Godot.Variant.CreateFrom(__PropertyGodotGenericArray_default_value));
        return values;
    }
#endif // TOOLS
#pragma warning restore CS0109
}
