/*
    Minimal Variant container for the cross‑runtime bridge.

    Only two jobs:
    - Hold an arbitrary object and report its Godot variant type.
    - Provide a static factory (`CreateFrom`) so the generated code can
      wrap return values into a common envelope when needed (e.g. Array
      and Dictionary constructors).

    No conversion methods, no operators, no implicit casts – the JS
    decoder already hands back concrete .NET types, so the bridge never
    uses these facilities.
*/
using System;
using System.Collections.Generic;

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

		private Variant(object? value) => _value = value;

		/// <summary>Public factory – required by generated code.</summary>
		public static Variant CreateFrom(object? value) => new(value);

		/// <summary>The wrapped object – used by Array/Dictionary flattening.</summary>
		public readonly object? Obj => _value;

		/// <summary>Godot variant type of the wrapped value.</summary>
		public readonly Type VariantType => GetVariantType(_value);

		/// <summary>Releases the reference; no native resources to free.</summary>
		public void Dispose() => _value = null;

		public override readonly string ToString() => _value?.ToString() ?? string.Empty;

		// ── type tag mapping ────────────────────────────────────────────────
		private static Type GetVariantType(object? value)
		{
			if (value is null) return Type.Nil;

			if (value is bool) return Type.Bool;

			if (value is byte or sbyte or short or ushort or int or uint or long or ulong)
				return Type.Int;

			if (value is float or double) return Type.Float;

			if (value is string) return Type.String;

			if (value is Vector2) return Type.Vector2;
			if (value is Vector2i) return Type.Vector2i;
			if (value is Rect2) return Type.Rect2;
			if (value is Rect2i) return Type.Rect2i;
			if (value is Vector3) return Type.Vector3;
			if (value is Vector3i) return Type.Vector3i;
			if (value is Transform2D) return Type.Transform2D;
			if (value is Vector4) return Type.Vector4;
			if (value is Vector4i) return Type.Vector4i;
			if (value is Plane) return Type.Plane;
			if (value is Quaternion) return Type.Quaternion;
			if (value is AABB) return Type.Aabb;
			if (value is Basis) return Type.Basis;
			if (value is Transform3D) return Type.Transform3D;
			if (value is Projection) return Type.Projection;
			if (value is Color) return Type.Color;

			if (value is Callable) return Type.Callable;
			if (value is Signal) return Type.Signal;
			if (value is GodotObject) return Type.Object;

			if (value is Dictionary<object, object>) return Type.Dictionary;

			if (value is byte[]) return Type.PackedByteArray;
			if (value is int[]) return Type.PackedInt32Array;
			if (value is long[]) return Type.PackedInt64Array;
			if (value is float[]) return Type.PackedFloat32Array;
			if (value is double[]) return Type.PackedFloat64Array;
			if (value is string[]) return Type.PackedStringArray;
			if (value is Vector2[]) return Type.PackedVector2Array;
			if (value is Vector3[]) return Type.PackedVector3Array;
			if (value is Vector4[]) return Type.PackedVector4Array;
			if (value is Color[]) return Type.PackedColorArray;
			if (value is object[]) return Type.Array;

			return Type.Nil;
		}
	}
}
