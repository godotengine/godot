using System;
using System.Runtime.InteropServices;

namespace Godot
{
	public struct AudioFrame
	{
		public float left;
		public float right;
	}

	public struct CaretInfo
	{
		public Rect2 leading_caret;
		public Rect2 trailing_caret;
		public ulong leading_direction;
		public ulong trailing_direction;
	}

	public struct Glyph
	{
		public int start = -1;
		public int end = -1;
		public byte count = 0;
		public byte repeat = 1;
		public ushort flags = 0;
		public float x_off = 0;
		public float y_off = 0;
		public float advance = 0;
		public RID font_rid;
		public int font_size = 0;
		public int index = 0;

		public Glyph() { }
	}

	public struct ObjectID
	{
		public ulong id = 0;

		public ObjectID() { }
	}

	public struct PhysicsServer2DExtensionMotionResult
	{
		public Vector2 travel;
		public Vector2 remainder;
		public Vector2 collision_point;
		public Vector2 collision_normal;
		public Vector2 collider_velocity;
		public double collision_depth;
		public double collision_safe_fraction;
		public double collision_unsafe_fraction;
		public int collision_local_shape;
		public ulong collider_id;
		public RID collider;
		public int collider_shape;
	}

	public struct PhysicsServer2DExtensionRayResult
	{
		public Vector2 position;
		public Vector2 normal;
		public RID rid;
		public ulong collider_id;
		public nint collider;
		public int shape;
	}

	public struct PhysicsServer2DExtensionShapeRestInfo
	{
		public Vector2 point;
		public Vector2 normal;
		public RID rid;
		public ulong collider_id;
		public int shape;
		public Vector2 linear_velocity;
	}

	public struct PhysicsServer2DExtensionShapeResult
	{
		public RID rid;
		public ulong collider_id;
		public nint collider;
		public int shape;
	}

	public struct PhysicsServer3DExtensionMotionCollision
	{
		public Vector3 position;
		public Vector3 normal;
		public Vector3 collider_velocity;
		public Vector3 collider_angular_velocity;
		public double depth;
		public int local_shape;
		public ulong collider_id;
		public RID collider;
		public int collider_shape;
	}

	public struct PhysicsServer3DExtensionMotionResult
	{
		public Vector3 travel;
		public Vector3 remainder;
		public double collision_depth;
		public double collision_safe_fraction;
		public double collision_unsafe_fraction;
		public ulong[] collisions = new ulong[32];
		public int collision_count;

		public PhysicsServer3DExtensionMotionResult() { }
	}

	public struct PhysicsServer3DExtensionRayResult
	{
		public Vector3 position;
		public Vector3 normal;
		public RID rid;
		public ulong collider_id;
		public nint collider;
		public int shape;
		public int face_index;
	}

	public struct PhysicsServer3DExtensionShapeRestInfo
	{
		public Vector3 point;
		public Vector3 normal;
		public RID rid;
		public ulong collider_id;
		public int shape;
		public Vector3 linear_velocity;
	}

	public struct PhysicsServer3DExtensionShapeResult
	{
		public RID rid;
		public ulong collider_id;
		public nint collider;
		public int shape;
	}

	public struct ScriptLanguageExtensionProfilingInfo
	{
		public string signature;
		public ulong call_count;
		public ulong total_time;
		public ulong self_time;
	}

}
