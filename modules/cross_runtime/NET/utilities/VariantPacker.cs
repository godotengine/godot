using System;

// VariantPacker.cs - flattens C# Godot types into JS-transferable object[]
// for varargs calls across the __callGodot bridge.
//
// The JS side (variant_construct.js) reads these tagged arrays and reconstructs
// the corresponding heap-allocated Variant* before dispatching into Godot WASM.
// Every tagged array has the form: [tag, ...fields] where fields are all double.
// Unsupported types produce null, which variant_construct.js decodes as Nil.
namespace Godot
{
	public static class VariantPacker
	{
		// Entry point for varargs marshaling. Each element is flattened independently;
		// the resulting array is passed as the trailing varargs object[] to CallGodot.
		public static object[] Flatten(object[] args)
		{
			var out_ = new object[args.Length];
			for (int i = 0; i < args.Length; i++)
				out_[i] = FlattenOne(args[i]);
			return out_;
		}

		private static object FlattenOne(object v)
		{
			switch (v)
			{
				// Primitives cross the boundary as-is.
				case null: return null;
				case bool b: return b;
				case string s: return s;
				case double d: return d;

				// All integer and float types are widened to double — the only
				// numeric type the JS bridge accepts without coercion errors.
				case float f: return (double)f;
				case int i32: return (double)i32;
				case long i64: return (double)i64;
				case ulong u64: return (double)u64;

				// ── Vectors ──────────────────────────────────────────────────
				case Vector2 v2: return new object[] { "Vector2", (double)v2.X, (double)v2.Y };
				case Vector2i v2i: return new object[] { "Vector2i", (double)v2i.X, (double)v2i.Y };
				case Vector3 v3: return new object[] { "Vector3", (double)v3.X, (double)v3.Y, (double)v3.Z };
				case Vector3i v3i: return new object[] { "Vector3i", (double)v3i.X, (double)v3i.Y, (double)v3i.Z };
				case Vector4 v4: return new object[] { "Vector4", (double)v4.X, (double)v4.Y, (double)v4.Z, (double)v4.W };
				case Vector4i v4i: return new object[] { "Vector4i", (double)v4i.X, (double)v4i.Y, (double)v4i.Z, (double)v4i.W };

				// ── Geometry ─────────────────────────────────────────────────
				case Color c:
					return new object[] { "Color", (double)c.R, (double)c.G, (double)c.B, (double)c.A };

				case Rect2 r2:
					return new object[] { "Rect2",
						(double)r2.Position.X, (double)r2.Position.Y,
						(double)r2.Size.X,     (double)r2.Size.Y };
				case Rect2i r2i:
					return new object[] { "Rect2i",
						(double)r2i.Position.X, (double)r2i.Position.Y,
						(double)r2i.Size.X,     (double)r2i.Size.Y };

				case AABB aabb:
					return new object[] { "AABB",
						(double)aabb.Position.X, (double)aabb.Position.Y, (double)aabb.Position.Z,
						(double)aabb.Size.X,     (double)aabb.Size.Y,     (double)aabb.Size.Z };

				case Plane pl:
					return new object[] { "Plane",
						(double)pl.Normal.X, (double)pl.Normal.Y, (double)pl.Normal.Z, (double)pl.D };

				// Transforms
				case Quaternion q:
					return new object[] { "Quaternion", (double)q.X, (double)q.Y, (double)q.Z, (double)q.W };

				case Basis ba:
					return new object[] { "Basis",
						(double)ba.X.X, (double)ba.X.Y, (double)ba.X.Z,
						(double)ba.Y.X, (double)ba.Y.Y, (double)ba.Y.Z,
						(double)ba.Z.X, (double)ba.Z.Y, (double)ba.Z.Z };

				case Transform2D t2:
					return new object[] { "Transform2D",
						(double)t2.X.X, (double)t2.X.Y,
						(double)t2.Y.X, (double)t2.Y.Y,
						(double)t2.Origin.X, (double)t2.Origin.Y };

				case Transform3D t3:
					return new object[] { "Transform3D",
						(double)t3.Basis.X.X, (double)t3.Basis.X.Y, (double)t3.Basis.X.Z,
						(double)t3.Basis.Y.X, (double)t3.Basis.Y.Y, (double)t3.Basis.Y.Z,
						(double)t3.Basis.Z.X, (double)t3.Basis.Z.Y, (double)t3.Basis.Z.Z,
						(double)t3.Origin.X,  (double)t3.Origin.Y,  (double)t3.Origin.Z };


				case RID rid:
					return new object[] { "RID", (double)(ulong)rid.Id };

				case GodotObject go:
					return new object[] { "ObjectId", (double)(ulong)go.Id };

				default:
					return null; // unsupported type → Nil variant on the JS side
			}
		}
	}
}
