using System;
using Godot.Collections;
using Array = Godot.Collections.Array;

namespace Godot.SourceGenerators.Sample;

public class MustBeVariantMethods
{
    public void MustBeVariantMethodCalls()
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
        Method<byte[]>();
        Method<int[]>();
        Method<long[]>();
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

        // This call fails because generic type is not Variant-compatible.
        //Method<object>();
    }

    public void Method<[MustBeVariant] T>()
    {
    }

    public void MustBeVariantClasses()
    {
        new ClassWithGenericVariant<bool>();
        new ClassWithGenericVariant<char>();
        new ClassWithGenericVariant<sbyte>();
        new ClassWithGenericVariant<byte>();
        new ClassWithGenericVariant<short>();
        new ClassWithGenericVariant<ushort>();
        new ClassWithGenericVariant<int>();
        new ClassWithGenericVariant<uint>();
        new ClassWithGenericVariant<long>();
        new ClassWithGenericVariant<ulong>();
        new ClassWithGenericVariant<float>();
        new ClassWithGenericVariant<double>();
        new ClassWithGenericVariant<string>();
        new ClassWithGenericVariant<Vector2>();
        new ClassWithGenericVariant<Vector2I>();
        new ClassWithGenericVariant<Rect2>();
        new ClassWithGenericVariant<Rect2I>();
        new ClassWithGenericVariant<Transform2D>();
        new ClassWithGenericVariant<Vector3>();
        new ClassWithGenericVariant<Vector3I>();
        new ClassWithGenericVariant<Vector4>();
        new ClassWithGenericVariant<Vector4I>();
        new ClassWithGenericVariant<Basis>();
        new ClassWithGenericVariant<Quaternion>();
        new ClassWithGenericVariant<Transform3D>();
        new ClassWithGenericVariant<Projection>();
        new ClassWithGenericVariant<Aabb>();
        new ClassWithGenericVariant<Color>();
        new ClassWithGenericVariant<Plane>();
        new ClassWithGenericVariant<Callable>();
        new ClassWithGenericVariant<Signal>();
        new ClassWithGenericVariant<GodotObject>();
        new ClassWithGenericVariant<StringName>();
        new ClassWithGenericVariant<NodePath>();
        new ClassWithGenericVariant<Rid>();
        new ClassWithGenericVariant<Dictionary>();
        new ClassWithGenericVariant<Array>();
        new ClassWithGenericVariant<byte[]>();
        new ClassWithGenericVariant<int[]>();
        new ClassWithGenericVariant<long[]>();
        new ClassWithGenericVariant<float[]>();
        new ClassWithGenericVariant<double[]>();
        new ClassWithGenericVariant<string[]>();
        new ClassWithGenericVariant<Vector2[]>();
        new ClassWithGenericVariant<Vector3[]>();
        new ClassWithGenericVariant<Color[]>();
        new ClassWithGenericVariant<GodotObject[]>();
        new ClassWithGenericVariant<StringName[]>();
        new ClassWithGenericVariant<NodePath[]>();
        new ClassWithGenericVariant<Rid[]>();

        // This class fails because generic type is not Variant-compatible.
        //new ClassWithGenericVariant<object>();
    }
}

public class ClassWithGenericVariant<[MustBeVariant] T>
{
}

public class MustBeVariantAnnotatedMethods
{
    [GenericTypeAttribute<string>()]
    public void MethodWithAttributeOk()
    {
    }

    // This method definition fails because generic type is not Variant-compatible.
    /*
    [GenericTypeAttribute<object>()]
    public void MethodWithWrongAttribute()
    {
    }
    */
}

[GenericTypeAttribute<string>()]
public class ClassVariantAnnotated
{
}

// This class definition fails because generic type is not Variant-compatible.
/*
[GenericTypeAttribute<object>()]
public class ClassNonVariantAnnotated
{
}
*/

[AttributeUsage(AttributeTargets.Class | AttributeTargets.Method, AllowMultiple = true)]
public class GenericTypeAttribute<[MustBeVariant] T> : Attribute
{
}
