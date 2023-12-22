using System;
using Godot;
using Godot.Collections;
using Array = Godot.Collections.Array;

public class MustBeVariantGD0301
{
    public void MethodCallsError()
    {
        // object is not Variant and Method<T> requires a generic type
        Method<object>();
    }
    
    public void MethodCallsOk()
    {
        // All these call are valid because they are Variant-compatible types
        Method<bool>();
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
    }

    public void Method<[MustBeVariant] T>()
    {
    }
}