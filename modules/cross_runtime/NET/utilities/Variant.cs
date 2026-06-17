/*
    Adapted Variant container for your runtime types.

    This mirrors the shape of Godot's Variant API, but it stores and converts
    your own managed types directly.
*/
using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Godot;

#nullable enable

namespace Godot
{
public partial struct Variant : IDisposable
{
    public enum Type : int
    {
        Nil = 0,
        Bool = 1,
        Int = 2,
        Float = 3,
        String = 4,
        Vector2 = 5,
        Vector2i = 6,
        Rect2 = 7,
        Rect2i = 8,
        Vector3 = 9,
        Vector3i = 10,
        Transform2D = 11,
        Vector4 = 12,
        Vector4i = 13,
        Plane = 14,
        Quaternion = 15,
        Aabb = 16,
        Basis = 17,
        Transform3D = 18,
        Projection = 19,
        Color = 20,
        Object = 24,
        Callable = 25,
        Signal = 26,
        Dictionary = 27,
        Array = 28,
        PackedByteArray = 29,
        PackedInt32Array = 30,
        PackedInt64Array = 31,
        PackedFloat32Array = 32,
        PackedFloat64Array = 33,
        PackedStringArray = 34,
        PackedVector2Array = 35,
        PackedVector3Array = 36,
        PackedVector4Array = 37,
        PackedColorArray = 38
    }

    private object? _value;

    private Variant(object? value)
    {
        _value = value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant CreateFrom(object? value) => new(value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Variant From(object? value) => new(value);

    public void Dispose()
    {
        _value = null;
    }

    public Type VariantType => GetVariantType(_value);

    public object? Obj => _value;

    public override string ToString() => AsString();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T As<T>()
    {
        if (_value is T typed)
            return typed;

        throw new InvalidCastException($"Variant does not contain a value of type {typeof(T)}.");
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool AsBool() => _value is bool b ? b : throw new InvalidCastException();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public char AsChar() => (char)AsInt32();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public sbyte AsSByte() => checked((sbyte)AsInt64());

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public short AsInt16() => checked((short)AsInt64());

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int AsInt32() => checked((int)AsInt64());

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public long AsInt64()
    {
        return _value switch
        {
            byte v => v,
            sbyte v => v,
            short v => v,
            ushort v => v,
            int v => v,
            uint v => v,
            long v => v,
            ulong v => checked((long)v),
            _ => throw new InvalidCastException()
        };
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public byte AsByte() => checked((byte)AsInt64());

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ushort AsUInt16() => checked((ushort)AsInt64());

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public uint AsUInt32() => checked((uint)AsInt64());

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ulong AsUInt64()
    {
        return _value switch
        {
            byte v => v,
            sbyte v => checked((ulong)v),
            short v => checked((ulong)v),
            ushort v => v,
            int v => checked((ulong)v),
            uint v => v,
            long v => unchecked((ulong)v),
            ulong v => v,
            _ => throw new InvalidCastException()
        };
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float AsSingle() => _value switch
    {
        float v => v,
        double v => (float)v,
        byte v => v,
        sbyte v => v,
        short v => v,
        ushort v => v,
        int v => v,
        uint v => v,
        long v => v,
        ulong v => v,
        _ => throw new InvalidCastException()
    };

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double AsDouble() => _value switch
    {
        double v => v,
        float v => v,
        byte v => v,
        sbyte v => v,
        short v => v,
        ushort v => v,
        int v => v,
        uint v => v,
        long v => v,
        ulong v => v,
        _ => throw new InvalidCastException()
    };

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public string AsString() => _value switch
    {
        null => string.Empty,
        string v => v,
        _ => _value.ToString() ?? string.Empty
    };

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector2 AsVector2() => As<Vector2>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector2i AsVector2i() => As<Vector2i>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Rect2 AsRect2() => As<Rect2>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Rect2i AsRect2i() => As<Rect2i>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3 AsVector3() => As<Vector3>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3i AsVector3i() => As<Vector3i>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Transform2D AsTransform2D() => As<Transform2D>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector4 AsVector4() => As<Vector4>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector4i AsVector4i() => As<Vector4i>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Plane AsPlane() => As<Plane>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Quaternion AsQuaternion() => As<Quaternion>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public AABB AsAabb() => As<AABB>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Basis AsBasis() => As<Basis>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Transform3D AsTransform3D() => As<Transform3D>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Projection AsProjection() => As<Projection>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Color AsColor() => As<Color>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public GodotObject AsGodotObject() => As<GodotObject>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Callable AsCallable() => As<Callable>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Signal AsSignal() => As<Signal>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Dictionary<object, object> AsDictionary() => As<Dictionary<object, object>>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public object[] AsArray() => As<object[]>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public byte[] AsByteArray() => As<byte[]>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int[] AsInt32Array() => As<int[]>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public long[] AsInt64Array() => As<long[]>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float[] AsFloat32Array() => As<float[]>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double[] AsFloat64Array() => As<double[]>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public string[] AsStringArray() => As<string[]>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector2[] AsVector2Array() => As<Vector2[]>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector3[] AsVector3Array() => As<Vector3[]>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector4[] AsVector4Array() => As<Vector4[]>();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Color[] AsColorArray() => As<Color[]>();

    public static Variant CreateFrom(bool from) => from;
    public static Variant CreateFrom(char from) => from;
    public static Variant CreateFrom(sbyte from) => from;
    public static Variant CreateFrom(short from) => from;
    public static Variant CreateFrom(int from) => from;
    public static Variant CreateFrom(long from) => from;
    public static Variant CreateFrom(ulong from) => from;
    public static Variant CreateFrom(byte from) => from;
    public static Variant CreateFrom(ushort from) => from;
    public static Variant CreateFrom(uint from) => from;
    public static Variant CreateFrom(float from) => from;
    public static Variant CreateFrom(double from) => from;
    public static Variant CreateFrom(string from) => from;

    public static Variant CreateFrom(Vector2 from) => from;
    public static Variant CreateFrom(Vector2i from) => from;
    public static Variant CreateFrom(Rect2 from) => from;
    public static Variant CreateFrom(Rect2i from) => from;
    public static Variant CreateFrom(Transform2D from) => from;
    public static Variant CreateFrom(Vector3 from) => from;
    public static Variant CreateFrom(Vector3i from) => from;
    public static Variant CreateFrom(Basis from) => from;
    public static Variant CreateFrom(Quaternion from) => from;
    public static Variant CreateFrom(Transform3D from) => from;
    public static Variant CreateFrom(Vector4 from) => from;
    public static Variant CreateFrom(Vector4i from) => from;
    public static Variant CreateFrom(Projection from) => from;
    public static Variant CreateFrom(AABB from) => from;
    public static Variant CreateFrom(Color from) => from;
    public static Variant CreateFrom(Plane from) => from;

    public static Variant CreateFrom(Callable from) => from;
    public static Variant CreateFrom(Signal from) => from;
    public static Variant CreateFrom(GodotObject from) => from;

    public static Variant CreateFrom(Dictionary<object, object> from) => from;
    public static Variant CreateFrom(object[] from) => from;

    public static Variant CreateFrom(byte[] from) => from;
    public static Variant CreateFrom(int[] from) => from;
    public static Variant CreateFrom(long[] from) => from;
    public static Variant CreateFrom(float[] from) => from;
    public static Variant CreateFrom(double[] from) => from;
    public static Variant CreateFrom(string[] from) => from;
    public static Variant CreateFrom(Vector2[] from) => from;
    public static Variant CreateFrom(Vector3[] from) => from;
    public static Variant CreateFrom(Vector4[] from) => from;
    public static Variant CreateFrom(Color[] from) => from;

    public static implicit operator Variant(bool from) => CreateFrom(from);
    public static implicit operator Variant(char from) => CreateFrom(from);
    public static implicit operator Variant(sbyte from) => CreateFrom(from);
    public static implicit operator Variant(short from) => CreateFrom(from);
    public static implicit operator Variant(int from) => CreateFrom(from);
    public static implicit operator Variant(long from) => CreateFrom(from);
    public static implicit operator Variant(ulong from) => CreateFrom(from);
    public static implicit operator Variant(byte from) => CreateFrom(from);
    public static implicit operator Variant(ushort from) => CreateFrom(from);
    public static implicit operator Variant(uint from) => CreateFrom(from);
    public static implicit operator Variant(float from) => CreateFrom(from);
    public static implicit operator Variant(double from) => CreateFrom(from);
    public static implicit operator Variant(string from) => CreateFrom(from);

    public static implicit operator Variant(Vector2 from) => CreateFrom(from);
    public static implicit operator Variant(Vector2i from) => CreateFrom(from);
    public static implicit operator Variant(Rect2 from) => CreateFrom(from);
    public static implicit operator Variant(Rect2i from) => CreateFrom(from);
    public static implicit operator Variant(Transform2D from) => CreateFrom(from);
    public static implicit operator Variant(Vector3 from) => CreateFrom(from);
    public static implicit operator Variant(Vector3i from) => CreateFrom(from);
    public static implicit operator Variant(Basis from) => CreateFrom(from);
    public static implicit operator Variant(Quaternion from) => CreateFrom(from);
    public static implicit operator Variant(Transform3D from) => CreateFrom(from);
    public static implicit operator Variant(Vector4 from) => CreateFrom(from);
    public static implicit operator Variant(Vector4i from) => CreateFrom(from);
    public static implicit operator Variant(Projection from) => CreateFrom(from);
    public static implicit operator Variant(AABB from) => CreateFrom(from);
    public static implicit operator Variant(Color from) => CreateFrom(from);
    public static implicit operator Variant(Plane from) => CreateFrom(from);

    public static implicit operator Variant(Callable from) => CreateFrom(from);
    public static implicit operator Variant(Signal from) => CreateFrom(from);
    public static implicit operator Variant(GodotObject from) => CreateFrom(from);

    public static implicit operator Variant(Dictionary<object, object> from) => CreateFrom(from);
    public static implicit operator Variant(object[] from) => CreateFrom(from);

    public static implicit operator Variant(byte[] from) => CreateFrom(from);
    public static implicit operator Variant(int[] from) => CreateFrom(from);
    public static implicit operator Variant(long[] from) => CreateFrom(from);
    public static implicit operator Variant(float[] from) => CreateFrom(from);
    public static implicit operator Variant(double[] from) => CreateFrom(from);
    public static implicit operator Variant(string[] from) => CreateFrom(from);
    public static implicit operator Variant(Vector2[] from) => CreateFrom(from);
    public static implicit operator Variant(Vector3[] from) => CreateFrom(from);
    public static implicit operator Variant(Vector4[] from) => CreateFrom(from);
    public static implicit operator Variant(Color[] from) => CreateFrom(from);

    public static explicit operator bool(Variant from) => from.AsBool();
    public static explicit operator char(Variant from) => from.AsChar();
    public static explicit operator sbyte(Variant from) => from.AsSByte();
    public static explicit operator short(Variant from) => from.AsInt16();
    public static explicit operator int(Variant from) => from.AsInt32();
    public static explicit operator long(Variant from) => from.AsInt64();
    public static explicit operator ulong(Variant from) => from.AsUInt64();
    public static explicit operator byte(Variant from) => from.AsByte();
    public static explicit operator ushort(Variant from) => from.AsUInt16();
    public static explicit operator uint(Variant from) => from.AsUInt32();
    public static explicit operator float(Variant from) => from.AsSingle();
    public static explicit operator double(Variant from) => from.AsDouble();
    public static explicit operator string(Variant from) => from.AsString();

    public static explicit operator Vector2(Variant from) => from.AsVector2();
    public static explicit operator Vector2i(Variant from) => from.AsVector2i();
    public static explicit operator Rect2(Variant from) => from.AsRect2();
    public static explicit operator Rect2i(Variant from) => from.AsRect2i();
    public static explicit operator Transform2D(Variant from) => from.AsTransform2D();
    public static explicit operator Vector3(Variant from) => from.AsVector3();
    public static explicit operator Vector3i(Variant from) => from.AsVector3i();
    public static explicit operator Basis(Variant from) => from.AsBasis();
    public static explicit operator Quaternion(Variant from) => from.AsQuaternion();
    public static explicit operator Transform3D(Variant from) => from.AsTransform3D();
    public static explicit operator Vector4(Variant from) => from.AsVector4();
    public static explicit operator Vector4i(Variant from) => from.AsVector4i();
    public static explicit operator Projection(Variant from) => from.AsProjection();
    public static explicit operator AABB(Variant from) => from.AsAabb();
    public static explicit operator Color(Variant from) => from.AsColor();
    public static explicit operator Plane(Variant from) => from.AsPlane();

    public static explicit operator Callable(Variant from) => from.AsCallable();
    public static explicit operator Signal(Variant from) => from.AsSignal();
    public static explicit operator GodotObject(Variant from) => from.AsGodotObject();

    public static explicit operator Dictionary<object, object>(Variant from) => from.AsDictionary();
    public static explicit operator object[](Variant from) => from.AsArray();

    public static explicit operator byte[](Variant from) => from.AsByteArray();
    public static explicit operator int[](Variant from) => from.AsInt32Array();
    public static explicit operator long[](Variant from) => from.AsInt64Array();
    public static explicit operator float[](Variant from) => from.AsFloat32Array();
    public static explicit operator double[](Variant from) => from.AsFloat64Array();
    public static explicit operator string[](Variant from) => from.AsStringArray();
    public static explicit operator Vector2[](Variant from) => from.AsVector2Array();
    public static explicit operator Vector3[](Variant from) => from.AsVector3Array();
    public static explicit operator Vector4[](Variant from) => from.AsVector4Array();
    public static explicit operator Color[](Variant from) => from.AsColorArray();

    private static Type GetVariantType(object? value)
    {
        if (value is null)
            return Type.Nil;

        if (value is bool)
            return Type.Bool;

        if (value is byte or sbyte or short or ushort or int or uint or long or ulong)
            return Type.Int;

        if (value is float or double)
            return Type.Float;

        if (value is string)
            return Type.String;

        if (value is Vector2)
            return Type.Vector2;
        if (value is Vector2i)
            return Type.Vector2i;
        if (value is Rect2)
            return Type.Rect2;
        if (value is Rect2i)
            return Type.Rect2i;
        if (value is Vector3)
            return Type.Vector3;
        if (value is Vector3i)
            return Type.Vector3i;
        if (value is Transform2D)
            return Type.Transform2D;
        if (value is Vector4)
            return Type.Vector4;
        if (value is Vector4i)
            return Type.Vector4i;
        if (value is Plane)
            return Type.Plane;
        if (value is Quaternion)
            return Type.Quaternion;
        if (value is AABB)
            return Type.Aabb;
        if (value is Basis)
            return Type.Basis;
        if (value is Transform3D)
            return Type.Transform3D;
        if (value is Projection)
            return Type.Projection;
        if (value is Color)
            return Type.Color;

        if (value is Callable)
            return Type.Callable;
        if (value is Signal)
            return Type.Signal;
        if (value is GodotObject)
            return Type.Object;

        if (value is Dictionary<object, object>)
            return Type.Dictionary;

        if (value is byte[])
            return Type.PackedByteArray;
        if (value is int[])
            return Type.PackedInt32Array;
        if (value is long[])
            return Type.PackedInt64Array;
        if (value is float[])
            return Type.PackedFloat32Array;
        if (value is double[])
            return Type.PackedFloat64Array;
        if (value is string[])
            return Type.PackedStringArray;
        if (value is Vector2[])
            return Type.PackedVector2Array;
        if (value is Vector3[])
            return Type.PackedVector3Array;
        if (value is Vector4[])
            return Type.PackedVector4Array;
        if (value is Color[])
            return Type.PackedColorArray;
        if (value is object[])
            return Type.Array;

        return Type.Nil;
    }

    public Variant ConvertTo(Type type)
    {
        if (VariantType == type)
            return this;

        return type switch
        {
            Type.Nil => Variant.CreateFrom((object?)null),

            Type.Bool => Variant.CreateFrom(AsBool()),
            Type.Int => Variant.CreateFrom((long)AsDouble()),
            Type.Float => Variant.CreateFrom(AsDouble()),
            Type.String => Variant.CreateFrom(AsString()),

            Type.Vector2 => Variant.CreateFrom(AsVector2()),
            Type.Vector2i => Variant.CreateFrom(AsVector2i()),
            Type.Rect2 => Variant.CreateFrom(AsRect2()),
            Type.Rect2i => Variant.CreateFrom(AsRect2i()),
            Type.Vector3 => Variant.CreateFrom(AsVector3()),
            Type.Vector3i => Variant.CreateFrom(AsVector3i()),
            Type.Transform2D => Variant.CreateFrom(AsTransform2D()),
            Type.Vector4 => Variant.CreateFrom(AsVector4()),
            Type.Vector4i => Variant.CreateFrom(AsVector4i()),
            Type.Plane => Variant.CreateFrom(AsPlane()),
            Type.Quaternion => Variant.CreateFrom(AsQuaternion()),
            Type.Aabb => Variant.CreateFrom(AsAabb()),
            Type.Basis => Variant.CreateFrom(AsBasis()),
            Type.Transform3D => Variant.CreateFrom(AsTransform3D()),
            Type.Projection => Variant.CreateFrom(AsProjection()),
            Type.Color => Variant.CreateFrom(AsColor()),

            Type.Object => Variant.CreateFrom(AsGodotObject()),
            Type.Callable => Variant.CreateFrom(AsCallable()),
            Type.Signal => Variant.CreateFrom(AsSignal()),

            Type.Dictionary => Variant.CreateFrom(AsDictionary()),
            Type.Array => Variant.CreateFrom(AsArray()),

            Type.PackedByteArray => Variant.CreateFrom(AsByteArray()),
            Type.PackedInt32Array => Variant.CreateFrom(AsInt32Array()),
            Type.PackedInt64Array => Variant.CreateFrom(AsInt64Array()),
            Type.PackedFloat32Array => Variant.CreateFrom(AsFloat32Array()),
            Type.PackedFloat64Array => Variant.CreateFrom(AsFloat64Array()),
            Type.PackedStringArray => Variant.CreateFrom(AsStringArray()),
            Type.PackedVector2Array => Variant.CreateFrom(AsVector2Array()),
            Type.PackedVector3Array => Variant.CreateFrom(AsVector3Array()),
            Type.PackedVector4Array => Variant.CreateFrom(AsVector4Array()),
            Type.PackedColorArray => Variant.CreateFrom(AsColorArray()),

            _ => this
        };
    }
}
}