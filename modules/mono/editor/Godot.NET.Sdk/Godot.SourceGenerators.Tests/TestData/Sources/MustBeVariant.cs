using System;
using Godot;
using Godot.Collections;
using Array = Godot.Collections.Array;


[AttributeUsage(AttributeTargets.Class | AttributeTargets.Method, AllowMultiple = true)]
public class GenericTypeAttribute<[MustBeVariant] T> : Attribute
{
}

public class MustBeVariantMethods
{
    public MustBeVariantMethods()
    {
        Method<bool>();
        Method<char>();
        Method<sbyte>();
        Method<byte>();
        Method<short>();
        Method<ushort>();
        Method<int>();
        Method<uint>();
        Method<long>();
        Method<ulong>();
        Method<float>();
        Method<double>();
        Method<string>();
        Method<Vector2>();
        Method<Vector2I>();
        Method<Rect2>();
        Method<Rect2I>();
        Method<Transform2D>();
        Method<Vector3>();
        Method<Vector3I>();
        Method<Vector4>();
        Method<Vector4I>();
        Method<Basis>();
        Method<Quaternion>();
        Method<Transform3D>();
        Method<Projection>();
        Method<Aabb>();
        Method<Color>();
        Method<Plane>();
        Method<Callable>();
        Method<Signal>();
        Method<GodotObject>();
        Method<StringName>();
        Method<NodePath>();
        Method<Rid>();
        Method<Dictionary>();
        Method<Array>();
        Method<bool[]>();
        Method<char[]>();
        Method<sbyte[]>();
        Method<byte[]>();
        Method<short[]>();
        Method<ushort[]>();
        Method<int[]>();
        Method<uint[]>();
        Method<long[]>();
        Method<ulong[]>();
        Method<float[]>();
        Method<double[]>();
        Method<string[]>();
        Method<Vector2[]>();
        Method<Vector3[]>();
        Method<Color[]>();
        Method<GodotObject[]>();
        Method<StringName[]>();
        Method<NodePath[]>();
        Method<Rid[]>();
    }

    public void Method<[MustBeVariant] T>()
    {
    }
}

public class MustBeVariantAnnotatedMethods
{
    [GenericType<bool>()]
    public void MethodWithAttribute_bool()
    {
    }

    [GenericType<char>()]
    public void MethodWithAttribute_char()
    {
    }

    [GenericType<sbyte>()]
    public void MethodWithAttribute_sbyte()
    {
    }

    [GenericType<byte>()]
    public void MethodWithAttribute_byte()
    {
    }

    [GenericType<short>()]
    public void MethodWithAttribute_short()
    {
    }

    [GenericType<ushort>()]
    public void MethodWithAttribute_ushort()
    {
    }

    [GenericType<int>()]
    public void MethodWithAttribute_int()
    {
    }

    [GenericType<uint>()]
    public void MethodWithAttribute_uint()
    {
    }

    [GenericType<long>()]
    public void MethodWithAttribute_long()
    {
    }

    [GenericType<ulong>()]
    public void MethodWithAttribute_ulong()
    {
    }

    [GenericType<float>()]
    public void MethodWithAttribute_float()
    {
    }

    [GenericType<double>()]
    public void MethodWithAttribute_double()
    {
    }

    [GenericType<string>()]
    public void MethodWithAttribute_string()
    {
    }

    [GenericType<Vector2>()]
    public void MethodWithAttribute_Vector2()
    {
    }

    [GenericType<Vector2I>()]
    public void MethodWithAttribute_Vector2I()
    {
    }

    [GenericType<Rect2>()]
    public void MethodWithAttribute_Rect2()
    {
    }

    [GenericType<Rect2I>()]
    public void MethodWithAttribute_Rect2I()
    {
    }

    [GenericType<Transform2D>()]
    public void MethodWithAttribute_Transform2D()
    {
    }

    [GenericType<Vector3>()]
    public void MethodWithAttribute_Vector3()
    {
    }

    [GenericType<Vector3I>()]
    public void MethodWithAttribute_Vector3I()
    {
    }

    [GenericType<Vector4>()]
    public void MethodWithAttribute_Vector4()
    {
    }

    [GenericType<Vector4I>()]
    public void MethodWithAttribute_Vector4I()
    {
    }

    [GenericType<Basis>()]
    public void MethodWithAttribute_Basis()
    {
    }

    [GenericType<Quaternion>()]
    public void MethodWithAttribute_Quaternion()
    {
    }

    [GenericType<Transform3D>()]
    public void MethodWithAttribute_Transform3D()
    {
    }

    [GenericType<Projection>()]
    public void MethodWithAttribute_Projection()
    {
    }

    [GenericType<Aabb>()]
    public void MethodWithAttribute_Aabb()
    {
    }

    [GenericType<Color>()]
    public void MethodWithAttribute_Color()
    {
    }

    [GenericType<Plane>()]
    public void MethodWithAttribute_Plane()
    {
    }

    [GenericType<GodotObject>()]
    public void MethodWithAttribute_GodotObject()
    {
    }

    [GenericType<StringName>()]
    public void MethodWithAttribute_StringName()
    {
    }

    [GenericType<NodePath>()]
    public void MethodWithAttribute_NodePath()
    {
    }

    [GenericType<Rid>()]
    public void MethodWithAttribute_Rid()
    {
    }

    [GenericType<Dictionary>()]
    public void MethodWithAttribute_Dictionary()
    {
    }

    [GenericType<Array>()]
    public void MethodWithAttribute_Array()
    {
    }

    [GenericType<Callable>()]
    public void MethodWithAttribute_Callable()
    {
    }

    [GenericType<Signal>()]
    public void MethodWithAttribute_Signal()
    {
    }

    [GenericType<bool[]>()]
    public void MethodWithAttribute_boolArray()
    {
    }

    [GenericType<char[]>()]
    public void MethodWithAttribute_charArray()
    {
    }

    [GenericType<sbyte[]>()]
    public void MethodWithAttribute_sbyteArray()
    {
    }

    [GenericType<byte[]>()]
    public void MethodWithAttribute_byteArray()
    {
    }

    [GenericType<short[]>()]
    public void MethodWithAttribute_shortArray()
    {
    }

    [GenericType<ushort[]>()]
    public void MethodWithAttribute_ushortArray()
    {
    }

    [GenericType<int[]>()]
    public void MethodWithAttribute_intArray()
    {
    }

    [GenericType<uint[]>()]
    public void MethodWithAttribute_uintArray()
    {
    }

    [GenericType<long[]>()]
    public void MethodWithAttribute_longArray()
    {
    }

    [GenericType<ulong[]>()]
    public void MethodWithAttribute_ulongArray()
    {
    }

    [GenericType<float[]>()]
    public void MethodWithAttribute_floatArray()
    {
    }

    [GenericType<double[]>()]
    public void MethodWithAttribute_doubleArray()
    {
    }

    [GenericType<string[]>()]
    public void MethodWithAttribute_stringArray()
    {
    }

    [GenericType<Vector2[]>()]
    public void MethodWithAttribute_Vector2Array()
    {
    }

    [GenericType<Vector3[]>()]
    public void MethodWithAttribute_Vector3Array()
    {
    }

    [GenericType<Color[]>()]
    public void MethodWithAttribute_ColorArray()
    {
    }

    [GenericType<GodotObject[]>()]
    public void MethodWithAttribute_GodotObjectArray()
    {
    }

    [GenericType<StringName[]>()]
    public void MethodWithAttribute_StringNameArray()
    {
    }

    [GenericType<NodePath[]>()]
    public void MethodWithAttribute_NodePathArray()
    {
    }

    [GenericType<Rid[]>()]
    public void MethodWithAttribute_RidArray()
    {
    }
}

[GenericType<bool>()]
public class ClassVariantAnnotated_bool
{
}

[GenericType<char>]
public class ClassVariantAnnotated_char
{
}

[GenericType<sbyte>]
public class ClassVariantAnnotated_sbyte
{
}

[GenericType<byte>]
public class ClassVariantAnnotated_byte
{
}

[GenericType<short>]
public class ClassVariantAnnotated_short
{
}

[GenericType<ushort>]
public class ClassVariantAnnotated_ushort
{
}

[GenericType<int>]
public class ClassVariantAnnotated_int
{
}

[GenericType<uint>]
public class ClassVariantAnnotated_uint
{
}

[GenericType<long>]
public class ClassVariantAnnotated_long
{
}

[GenericType<ulong>]
public class ClassVariantAnnotated_ulong
{
}

[GenericType<float>]
public class ClassVariantAnnotated_float
{
}

[GenericType<double>]
public class ClassVariantAnnotated_double
{
}

[GenericType<string>]
public class ClassVariantAnnotated_string
{
}

[GenericType<Vector2>]
public class ClassVariantAnnotated_Vector2
{
}

[GenericType<Vector2I>]
public class ClassVariantAnnotated_Vector2I
{
}

[GenericType<Rect2>]
public class ClassVariantAnnotated_Rect2
{
}

[GenericType<Rect2I>]
public class ClassVariantAnnotated_Rect2I
{
}

[GenericType<Transform2D>]
public class ClassVariantAnnotated_Transform2D
{
}

[GenericType<Vector3>]
public class ClassVariantAnnotated_Vector3
{
}

[GenericType<Vector3I>]
public class ClassVariantAnnotated_Vector3I
{
}

[GenericType<Vector4>]
public class ClassVariantAnnotated_Vector4
{
}

[GenericType<Vector4I>]
public class ClassVariantAnnotated_Vector4I
{
}

[GenericType<Basis>]
public class ClassVariantAnnotated_Basis
{
}

[GenericType<Quaternion>]
public class ClassVariantAnnotated_Quaternion
{
}

[GenericType<Transform3D>]
public class ClassVariantAnnotated_Transform3D
{
}

[GenericType<Projection>]
public class ClassVariantAnnotated_Projection
{
}

[GenericType<Aabb>]
public class ClassVariantAnnotated_Aabb
{
}

[GenericType<Color>]
public class ClassVariantAnnotated_Color
{
}

[GenericType<Plane>]
public class ClassVariantAnnotated_Plane
{
}

[GenericType<Callable>]
public class ClassVariantAnnotated_Callable
{
}

[GenericType<Signal>]
public class ClassVariantAnnotated_Signal
{
}

[GenericType<GodotObject>]
public class ClassVariantAnnotated_GodotObject
{
}

[GenericType<StringName>]
public class ClassVariantAnnotated_StringName
{
}

[GenericType<NodePath>]
public class ClassVariantAnnotated_NodePath
{
}

[GenericType<Rid>]
public class ClassVariantAnnotated_Rid
{
}

[GenericType<Dictionary>]
public class ClassVariantAnnotated_Dictionary
{
}

[GenericType<Array>]
public class ClassVariantAnnotated_Array
{
}

[GenericType<bool[]>]
public class ClassVariantAnnotated_boolArray
{
}

[GenericType<char[]>]
public class ClassVariantAnnotated_charArray
{
}

[GenericType<sbyte[]>]
public class ClassVariantAnnotated_sbyteArray
{
}

[GenericType<byte[]>]
public class ClassVariantAnnotated_byteArray
{
}

[GenericType<short[]>]
public class ClassVariantAnnotated_shortArray
{
}

[GenericType<ushort[]>]
public class ClassVariantAnnotated_ushortArray
{
}

[GenericType<int[]>]
public class ClassVariantAnnotated_intArray
{
}

[GenericType<uint[]>]
public class ClassVariantAnnotated_uintArray
{
}

[GenericType<long[]>]
public class ClassVariantAnnotated_longArray
{
}

[GenericType<ulong[]>]
public class ClassVariantAnnotated_ulongArray
{
}

[GenericType<float[]>]
public class ClassVariantAnnotated_floatArray
{
}

[GenericType<double[]>]
public class ClassVariantAnnotated_doubleArray
{
}

[GenericType<string[]>]
public class ClassVariantAnnotated_stringArray
{
}

[GenericType<Vector2[]>]
public class ClassVariantAnnotated_Vector2Array
{
}

[GenericType<Vector3[]>]
public class ClassVariantAnnotated_Vector3Array
{
}

[GenericType<Color[]>]
public class ClassVariantAnnotated_ColorArray
{
}

[GenericType<GodotObject[]>]
public class ClassVariantAnnotated_GodotObjectArray
{
}

[GenericType<StringName[]>]
public class ClassVariantAnnotated_StringNameArray
{
}

[GenericType<NodePath[]>]
public class ClassVariantAnnotated_NodePathArray
{
}

[GenericType<Rid[]>]
public class ClassVariantAnnotated_RidArray
{
}
