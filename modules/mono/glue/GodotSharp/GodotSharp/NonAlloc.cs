// This file contains methods that are specifically written to return a non-allocating struct as opposed to a Dictionary (or Array of Dictionaries) to avoid allocations.

using System.Collections.Generic;
using Godot.NativeInterop;

namespace Godot;

#pragma warning disable IDE0040 // Add accessibility modifiers.

partial class Node
{
    /// <summary>
    /// <para>Stores all children of this node into the provided <paramref name="results"/> list.</para>
    /// <para>If <paramref name="includeInternal"/> is <see langword="false"/>, excludes internal children when populating the list (see <see cref="Godot.Node.AddChild(Node, bool, Node.InternalMode)"/>'s <c>internal</c> parameter).</para>
    /// </summary>
    public unsafe void GetChildrenNonAlloc(List<Node> results, bool includeInternal = false)
    {
        var ptr = GodotObject.GetPtr(this);
        ExceptionUtils.ThrowIfNullPtr(ptr);
        godot_array ret = default;
        void** call_args = stackalloc void*[1] { &includeInternal };
        NativeFuncs.godotsharp_method_bind_ptrcall(MethodBind9, ptr, call_args, &ret);
        for (var i = 0; i < ret.Size; i++)
        {
            var item = ret.Elements[i];
            var obj = VariantUtils.ConvertToGodotObject(item);
            results.Add((Node)obj);
        }
    }
}

partial class PhysicsDirectSpaceState2D
{
    /// <summary>
    /// Stores the result of the <see cref="IntersectPointNonAlloc"/> method.
    /// </summary>
    /// <param name="Collider">The colliding object.</param>
    /// <param name="ColliderId">The colliding object's ID.</param>
    /// <param name="Rid">The intersecting object's <see cref="Godot.Rid"/>.</param>
    /// <param name="Shape">The shape index of the colliding shape.</param>
    public record struct IntersectPointResult(
        GodotObject Collider,
        long ColliderId,
        Rid Rid,
        long Shape
    )
    {
        internal static readonly Variant ColliderKey = "collider";
        internal static readonly Variant ColliderIdKey = "collider_id";
        internal static readonly Variant RidKey = "rid";
        internal static readonly Variant ShapeKey = "shape";
    }

    /// <summary>
    /// Checks whether a point is inside any solid shape. Position and other parameters are defined through <see cref="Godot.PhysicsPointQueryParameters2D"/>. The shapes the point is inside of are filled into the <paramref name="results"/> list.
    /// </summary>
    /// <remarks>
    /// <see cref="Godot.ConcavePolygonShape2D"/>s and <see cref="Godot.CollisionPolygon2D"/>s in <c>Segments</c> build mode are not solid shapes. Therefore, they will not be detected.
    /// </remarks>
    public unsafe void IntersectPointNonAlloc(PhysicsPointQueryParameters2D parameters, List<IntersectPointResult> results, int maxResults = 32)
    {
        var method = MethodBind0;
        var ptr = GodotObject.GetPtr(this);
        var arg1 = GodotObject.GetPtr(parameters);
        var arg2 = maxResults;
        ExceptionUtils.ThrowIfNullPtr(ptr);
        godot_array ret = default;
        long arg2_in = arg2;
        void** call_args = stackalloc void*[2] { &arg1, &arg2_in };
        NativeFuncs.godotsharp_method_bind_ptrcall(method, ptr, call_args, &ret);
        for (int i = 0; i < ret.Size; i++)
        {
            if (results.Count >= maxResults) break;
            var item = ret.Elements[i];
            if (item.Type != Variant.Type.Dictionary) continue;

            var dict = item.Dictionary;
            if (!NativeFuncs.godotsharp_dictionary_try_get_value(ref dict, (godot_variant)IntersectPointResult.ColliderKey.NativeVar, out var colliderValue).ToBool()
                || !NativeFuncs.godotsharp_dictionary_try_get_value(ref dict, (godot_variant)IntersectPointResult.ColliderIdKey.NativeVar, out var colliderIdValue).ToBool()
                || !NativeFuncs.godotsharp_dictionary_try_get_value(ref dict, (godot_variant)IntersectPointResult.RidKey.NativeVar, out var ridValue).ToBool()
                || !NativeFuncs.godotsharp_dictionary_try_get_value(ref dict, (godot_variant)IntersectPointResult.ShapeKey.NativeVar, out var shapeValue).ToBool())
                continue;

            results.Add(new IntersectPointResult(
                VariantUtils.ConvertToGodotObject(colliderValue),
                colliderIdValue.Int,
                ridValue.Rid,
                shapeValue.Int
            ));
        }
        NativeFuncs.godotsharp_array_destroy(ref ret);
    }


    /// <summary>
    /// Stores the result of the <see cref="IntersectRayNonAlloc"/> method.
    /// </summary>
    /// <param name="Collider">The colliding object.</param>
    /// <param name="ColliderId">The colliding object's ID.</param>
    /// <param name="Normal">The object's surface normal at the intersection point, or <c>Vector2(0, 0)</c> if the ray starts inside the shape and <see cref="Godot.PhysicsRayQueryParameters2D.HitFromInside"/> is <see langword="true"/>.</param>
    /// <param name="Position">The intersection point.</param>
    /// <param name="Rid">The intersecting object's <see cref="Godot.Rid"/>.</param>
    /// <param name="Shape">The shape index of the colliding shape.</param>
    public record struct IntersectRayResult(
        GodotObject Collider,
        long ColliderId,
        Vector2 Normal,
        Vector2 Position,
        Rid Rid,
        long Shape
    )
    {
        internal static readonly Variant ColliderKey = "collider";
        internal static readonly Variant ColliderIdKey = "collider_id";
        internal static readonly Variant NormalKey = "normal";
        internal static readonly Variant PositionKey = "position";
        internal static readonly Variant RidKey = "rid";
        internal static readonly Variant ShapeKey = "shape";
    }

    /// <summary>
    /// Intersects a ray in a given space. Ray position and other parameters are defined through <see cref="Godot.PhysicsRayQueryParameters2D"/>.
    /// </summary>
    /// <returns>
    /// A <see cref="IntersectRayResult"/> struct containing the result of the ray intersection, or <see langword="null"/> if no intersection occurred.
    /// </returns>
    public unsafe IntersectRayResult? IntersectRayNonAlloc(PhysicsRayQueryParameters2D parameters)
    {
        var method = MethodBind1;
        var ptr = GodotObject.GetPtr(this);
        var arg1 = GodotObject.GetPtr(parameters);
        ExceptionUtils.ThrowIfNullPtr(ptr);
        godot_dictionary ret = default;
        void** call_args = stackalloc void*[1] { &arg1 };
        NativeFuncs.godotsharp_method_bind_ptrcall(method, ptr, call_args, &ret);

        if (NativeFuncs.godotsharp_dictionary_count(ref ret) == 0) return null;

        if (!NativeFuncs.godotsharp_dictionary_try_get_value(ref ret, (godot_variant)IntersectRayResult.ColliderKey.NativeVar, out godot_variant colliderValue).ToBool()
        || !NativeFuncs.godotsharp_dictionary_try_get_value(ref ret, (godot_variant)IntersectRayResult.ColliderIdKey.NativeVar, out godot_variant colliderIdValue).ToBool()
        || !NativeFuncs.godotsharp_dictionary_try_get_value(ref ret, (godot_variant)IntersectRayResult.NormalKey.NativeVar, out godot_variant normalValue).ToBool()
        || !NativeFuncs.godotsharp_dictionary_try_get_value(ref ret, (godot_variant)IntersectRayResult.PositionKey.NativeVar, out godot_variant positionValue).ToBool()
        || !NativeFuncs.godotsharp_dictionary_try_get_value(ref ret, (godot_variant)IntersectRayResult.RidKey.NativeVar, out godot_variant ridValue).ToBool()
        || !NativeFuncs.godotsharp_dictionary_try_get_value(ref ret, (godot_variant)IntersectRayResult.ShapeKey.NativeVar, out godot_variant shapeValue).ToBool())
        {
            NativeFuncs.godotsharp_dictionary_destroy(ref ret);
            return null;
        }

        var returnValue = new IntersectRayResult(
            VariantUtils.ConvertToGodotObject(colliderValue),
            colliderIdValue.Int,
            normalValue.Vector2,
            positionValue.Vector2,
            ridValue.Rid,
            shapeValue.Int
        );

        NativeFuncs.godotsharp_dictionary_destroy(ref ret);
        return returnValue;
    }


    /// <summary>
    /// Stores the result of the <see cref="IntersectShapeNonAlloc"/> method.
    /// </summary>
    /// <param name="Collider">The colliding object.</param>
    /// <param name="ColliderId">The colliding object's ID.</param>
    /// <param name="Rid">The intersecting object's <see cref="Godot.Rid"/>.</param>
    /// <param name="Shape">The shape index of the colliding shape.</param>
    public record struct IntersectShapeResult(
        GodotObject Collider,
        long ColliderId,
        Rid Rid,
        long Shape
    )
    {
        internal static readonly Variant ColliderKey = "collider";
        internal static readonly Variant ColliderIdKey = "collider_id";
        internal static readonly Variant RidKey = "rid";
        internal static readonly Variant ShapeKey = "shape";
    }

    /// <summary>
    /// Checks the intersections of a shape, given through a <see cref="Godot.PhysicsShapeQueryParameters2D"/> object, against the space. The intersected shapes are filled into the <paramref name="results"/> list.
    /// </summary>
    public unsafe void IntersectShapeNonAlloc(PhysicsShapeQueryParameters2D parameters, List<IntersectShapeResult> results, int maxResults = 32)
    {
        var method = MethodBind2;
        var ptr = GodotObject.GetPtr(this);
        var arg1 = GodotObject.GetPtr(parameters);
        var arg2 = maxResults;
        ExceptionUtils.ThrowIfNullPtr(ptr);
        godot_array ret = default;
        long arg2_in = arg2;
        void** call_args = stackalloc void*[2] { &arg1, &arg2_in };
        NativeFuncs.godotsharp_method_bind_ptrcall(method, ptr, call_args, &ret);
        for (int i = 0; i < ret.Size; i++)
        {
            if (results.Count >= maxResults) break;
            var item = ret.Elements[i];
            if (item.Type != Variant.Type.Dictionary) continue;

            var dict = item.Dictionary;
            if (!NativeFuncs.godotsharp_dictionary_try_get_value(ref dict, (godot_variant)IntersectShapeResult.ColliderKey.NativeVar, out var colliderValue).ToBool()
                || !NativeFuncs.godotsharp_dictionary_try_get_value(ref dict, (godot_variant)IntersectShapeResult.ColliderIdKey.NativeVar, out var colliderIdValue).ToBool()
                || !NativeFuncs.godotsharp_dictionary_try_get_value(ref dict, (godot_variant)IntersectShapeResult.RidKey.NativeVar, out var ridValue).ToBool()
                || !NativeFuncs.godotsharp_dictionary_try_get_value(ref dict, (godot_variant)IntersectShapeResult.ShapeKey.NativeVar, out var shapeValue).ToBool())
                continue;

            results.Add(new IntersectShapeResult(
                VariantUtils.ConvertToGodotObject(colliderValue),
                colliderIdValue.Int,
                ridValue.Rid,
                shapeValue.Int
            ));
        }
        NativeFuncs.godotsharp_array_destroy(ref ret);
    }


    /// <summary>
    /// Stores the result of the <see cref="CastMotionNonAlloc"/> method.
    /// </summary>
    /// <param name="Safe">The maximum fraction of the motion that can be made without a collision.</param>
    /// <param name="Unsafe">The minimum fraction of the distance that must be moved for a collision.</param>
    public record struct CastMotionResult(
        float Safe,
        float Unsafe
    );

    /// <summary>
    /// Checks how far a <see cref="Godot.Shape2D"/> can move without colliding. All the parameters for the query, including the shape and the motion, are supplied through a <see cref="Godot.PhysicsShapeQueryParameters2D"/> object.
    /// </summary>
    /// <returns>A <see cref="CastMotionResult"/> struct containing the safe and unsafe proportions of the motion, or <c>[1.0, 1.0]</c> if no collision is detected.</returns>
    /// <remarks>
    /// Any <see cref="Godot.Shape2D"/>s that the shape is already colliding with e.g. inside of, will be ignored. Use <see cref="Godot.PhysicsDirectSpaceState2D.CollideShape(PhysicsShapeQueryParameters2D, int)"/> to determine the <see cref="Godot.Shape2D"/>s that the shape is already colliding with.
    /// </remarks>
    public unsafe CastMotionResult CastMotionNonAlloc(PhysicsShapeQueryParameters2D parameters)
    {
        var method = MethodBind3;
        var ptr = GodotObject.GetPtr(this);
        var arg1 = GodotObject.GetPtr(parameters);
        ExceptionUtils.ThrowIfNullPtr(ptr);
        godot_packed_float32_array ret = default;
        void** call_args = stackalloc void*[1] { &arg1 };
        NativeFuncs.godotsharp_method_bind_ptrcall(method, ptr, call_args, &ret);
        if (ret.Size != 2)
        {
            NativeFuncs.godotsharp_packed_float32_array_destroy(ref ret);
            return new CastMotionResult(1.0f, 1.0f);
        }

        var safeValue = ret.Buffer[0];
        var unsafeValue = ret.Buffer[1];
        NativeFuncs.godotsharp_packed_float32_array_destroy(ref ret);
        return new CastMotionResult(safeValue, unsafeValue);
    }


    /// <summary>
    /// Stores the result of the <see cref="GetRestInfoNonAlloc"/> method.
    /// </summary>
    /// <param name="ColliderId">The colliding object's ID.</param>
    /// <param name="LinearVelocity">The colliding object's velocity <see cref="Godot.Vector2"/>. If the object is an <see cref="Godot.Area2D"/>, the result is <c>(0, 0)</c>.</param>
    /// <param name="Normal">The collision normal of the query shape at the intersection point, pointing away from the intersecting object.</param>
    /// <param name="Point">The intersection point.</param>
    /// <param name="Rid">The intersecting object's <see cref="Godot.Rid"/>.</param>
    /// <param name="Shape">The shape index of the colliding shape.</param>
    public record struct GetRestInfoResult(
        long ColliderId,
        Vector2 LinearVelocity,
        Vector2 Normal,
        Vector2 Point,
        Rid Rid,
        long Shape
    )
    {
        internal static readonly Variant ColliderIdKey = "collider_id";
        internal static readonly Variant LinearVelocityKey = "linear_velocity";
        internal static readonly Variant NormalKey = "normal";
        internal static readonly Variant PointKey = "point";
        internal static readonly Variant RidKey = "rid";
        internal static readonly Variant ShapeKey = "shape";
    }

    /// <summary>
    /// Checks the intersections of a shape, given through a <see cref="Godot.PhysicsShapeQueryParameters2D"/> object, against the space. If it collides with more than one shape, the nearest one is selected.
    /// </summary>
    /// <returns>A <see cref="GetRestInfoResult"/> struct containing the result of the shape intersection, or <see langword="null"/> if no intersection occurred.</returns>
    /// <remarks>
    /// This method does not take into account the <c>motion</c> property of the object.
    /// </remarks>
    public unsafe GetRestInfoResult? GetRestInfoNonAlloc(PhysicsShapeQueryParameters2D parameters)
    {
        var method = MethodBind2;
        var ptr = GodotObject.GetPtr(this);
        var arg1 = GodotObject.GetPtr(parameters);
        ExceptionUtils.ThrowIfNullPtr(ptr);
        godot_dictionary ret = default;
        void** call_args = stackalloc void*[1] { &arg1 };
        NativeFuncs.godotsharp_method_bind_ptrcall(method, ptr, call_args, &ret);

        if (NativeFuncs.godotsharp_dictionary_count(ref ret) == 0)
        {
            NativeFuncs.godotsharp_dictionary_destroy(ref ret);
            return null;
        }

        if (!NativeFuncs.godotsharp_dictionary_try_get_value(ref ret, (godot_variant)GetRestInfoResult.ColliderIdKey.NativeVar, out godot_variant colliderIdValue).ToBool()
        || !NativeFuncs.godotsharp_dictionary_try_get_value(ref ret, (godot_variant)GetRestInfoResult.LinearVelocityKey.NativeVar, out godot_variant linearVelocityValue).ToBool()
        || !NativeFuncs.godotsharp_dictionary_try_get_value(ref ret, (godot_variant)GetRestInfoResult.NormalKey.NativeVar, out godot_variant normalValue).ToBool()
        || !NativeFuncs.godotsharp_dictionary_try_get_value(ref ret, (godot_variant)GetRestInfoResult.PointKey.NativeVar, out godot_variant pointValue).ToBool()
        || !NativeFuncs.godotsharp_dictionary_try_get_value(ref ret, (godot_variant)GetRestInfoResult.RidKey.NativeVar, out godot_variant ridValue).ToBool()
        || !NativeFuncs.godotsharp_dictionary_try_get_value(ref ret, (godot_variant)GetRestInfoResult.ShapeKey.NativeVar, out godot_variant shapeValue).ToBool())
        {
            NativeFuncs.godotsharp_dictionary_destroy(ref ret);
            return null;
        }

        var returnValue = new GetRestInfoResult(
            colliderIdValue.Int,
            linearVelocityValue.Vector2,
            normalValue.Vector2,
            pointValue.Vector2,
            ridValue.Rid,
            shapeValue.Int
        );

        NativeFuncs.godotsharp_dictionary_destroy(ref ret);
        return returnValue;
    }
}

partial class PhysicsDirectSpaceState3D
{
    /// <summary>
    /// Stores the shapes the point is inside of when calling <see cref="IntersectPointNonAlloc"/>.
    /// </summary>
    /// <param name="Collider">The colliding object.</param>
    /// <param name="ColliderId">The colliding object's ID.</param>
    /// <param name="Rid">The intersecting object's <see cref="Godot.Rid"/>.</param>
    /// <param name="Shape">The shape index of the colliding shape.</param>
    public record struct IntersectPointResult(
        GodotObject Collider,
        long ColliderId,
        Rid Rid,
        long Shape
    )
    {
        internal static readonly Variant ColliderKey = "collider";
        internal static readonly Variant ColliderIdKey = "collider_id";
        internal static readonly Variant RidKey = "rid";
        internal static readonly Variant ShapeKey = "shape";
    }

    /// <summary>
    /// Checks whether a point is inside any solid shape. Position and other parameters are defined through <see cref="Godot.PhysicsPointQueryParameters3D"/>. The shapes the point is inside of are filled into the <paramref name="results"/> list.
    /// </summary>
    /// <remarks>
    /// The number of intersections can be limited with the <paramref name="maxResults"/> parameter, to reduce the processing time.
    /// </remarks>
    public unsafe void IntersectPointNonAlloc(PhysicsPointQueryParameters3D parameters, List<IntersectPointResult> results, int maxResults = 32)
    {
        var method = MethodBind0;
        var ptr = GodotObject.GetPtr(this);
        var arg1 = GodotObject.GetPtr(parameters);
        var arg2 = maxResults;
        ExceptionUtils.ThrowIfNullPtr(ptr);
        godot_array ret = default;
        long arg2_in = arg2;
        void** call_args = stackalloc void*[2] { &arg1, &arg2_in };
        NativeFuncs.godotsharp_method_bind_ptrcall(method, ptr, call_args, &ret);
        for (int i = 0; i < ret.Size; i++)
        {
            if (results.Count >= maxResults) break;
            var item = ret.Elements[i];
            if (item.Type != Variant.Type.Dictionary) continue;
            var dict = item.Dictionary;
            if (!NativeFuncs.godotsharp_dictionary_try_get_value(ref dict, (godot_variant)IntersectPointResult.ColliderKey.NativeVar, out var colliderValue).ToBool()
                || !NativeFuncs.godotsharp_dictionary_try_get_value(ref dict, (godot_variant)IntersectPointResult.ColliderIdKey.NativeVar, out var colliderIdValue).ToBool()
                || !NativeFuncs.godotsharp_dictionary_try_get_value(ref dict, (godot_variant)IntersectPointResult.RidKey.NativeVar, out var ridValue).ToBool()
                || !NativeFuncs.godotsharp_dictionary_try_get_value(ref dict, (godot_variant)IntersectPointResult.ShapeKey.NativeVar, out var shapeValue).ToBool())
                continue;
            results.Add(new IntersectPointResult(
                VariantUtils.ConvertToGodotObject(colliderValue),
                colliderIdValue.Int,
                ridValue.Rid,
                shapeValue.Int
            ));
        }
        NativeFuncs.godotsharp_array_destroy(ref ret);
    }

    /// <summary>
    /// Stores the result of the <see cref="IntersectRayNonAlloc"/> method.
    /// </summary>
    /// <param name="Collider">The colliding object.</param>
    /// <param name="ColliderId">The colliding object's ID.</param>
    /// <param name="Normal">The object's surface normal at the intersection point, or <c>Vector3(0, 0, 0)</c> if the ray starts inside the shape and <see cref="Godot.PhysicsRayQueryParameters3D.HitFromInside"/> is <see langword="true"/>.</param>
    /// <param name="Position">The intersection point.</param>
    /// <param name="FaceIndex">The face index at the intersection point, or <c>-1</c> if the ray did not hit a face.</param>
    /// <param name="Rid">The intersecting object's <see cref="Godot.Rid"/>.</param>
    /// <param name="Shape">The shape index of the colliding shape.</param>
    public record struct IntersectRayResult(
        GodotObject Collider,
        long ColliderId,
        Vector3 Normal,
        Vector3 Position,
        long FaceIndex,
        Rid Rid,
        long Shape
    )
    {
        internal static readonly Variant ColliderKey = "collider";
        internal static readonly Variant ColliderIdKey = "collider_id";
        internal static readonly Variant NormalKey = "normal";
        internal static readonly Variant PositionKey = "position";
        internal static readonly Variant FaceIndexKey = "face_index";
        internal static readonly Variant RidKey = "rid";
        internal static readonly Variant ShapeKey = "shape";
    }

    /// <summary>
    /// Intersects a ray in a given space. Ray position and other parameters are defined through <see cref="Godot.PhysicsRayQueryParameters3D"/>.
    /// </summary>
    /// <returns>
    /// A <see cref="IntersectRayResult"/> struct containing the result of the ray intersection, or <see langword="null"/> if no intersection occurred.
    /// </returns>
    public unsafe IntersectRayResult? IntersectRayNonAlloc(PhysicsRayQueryParameters3D parameters)
    {
        var method = MethodBind1;
        var ptr = GodotObject.GetPtr(this);
        var arg1 = GodotObject.GetPtr(parameters);
        ExceptionUtils.ThrowIfNullPtr(ptr);
        godot_dictionary ret = default;
        void** call_args = stackalloc void*[1] { &arg1 };
        NativeFuncs.godotsharp_method_bind_ptrcall(method, ptr, call_args, &ret);
        if (NativeFuncs.godotsharp_dictionary_count(ref ret) == 0)
        {
            NativeFuncs.godotsharp_dictionary_destroy(ref ret);
            return null;
        }
        if (!NativeFuncs.godotsharp_dictionary_try_get_value(ref ret, (godot_variant)IntersectRayResult.ColliderKey.NativeVar, out godot_variant colliderValue).ToBool()
            || !NativeFuncs.godotsharp_dictionary_try_get_value(ref ret, (godot_variant)IntersectRayResult.ColliderIdKey.NativeVar, out godot_variant colliderIdValue).ToBool()
            || !NativeFuncs.godotsharp_dictionary_try_get_value(ref ret, (godot_variant)IntersectRayResult.NormalKey.NativeVar, out godot_variant normalValue).ToBool()
            || !NativeFuncs.godotsharp_dictionary_try_get_value(ref ret, (godot_variant)IntersectRayResult.PositionKey.NativeVar, out godot_variant positionValue).ToBool()
            || !NativeFuncs.godotsharp_dictionary_try_get_value(ref ret, (godot_variant)IntersectRayResult.FaceIndexKey.NativeVar, out godot_variant faceIndexValue).ToBool()
            || !NativeFuncs.godotsharp_dictionary_try_get_value(ref ret, (godot_variant)IntersectRayResult.RidKey.NativeVar, out godot_variant ridValue).ToBool()
            || !NativeFuncs.godotsharp_dictionary_try_get_value(ref ret, (godot_variant)IntersectRayResult.ShapeKey.NativeVar, out godot_variant shapeValue).ToBool())
        {
            NativeFuncs.godotsharp_dictionary_destroy(ref ret);
            return null;
        }
        var returnValue = new IntersectRayResult(
            VariantUtils.ConvertToGodotObject(colliderValue),
            colliderIdValue.Int,
            normalValue.Vector3,
            positionValue.Vector3,
            faceIndexValue.Int,
            ridValue.Rid,
            shapeValue.Int
        );
        NativeFuncs.godotsharp_dictionary_destroy(ref ret);
        return returnValue;
    }

    /// <summary>
    /// Stores the shapes the ray intersects with when calling <see cref="IntersectShapeNonAlloc"/>.
    /// </summary>
    /// <param name="Collider">The colliding object.</param>
    /// <param name="ColliderId">The colliding object's ID.</param>
    /// <param name="Rid">The intersecting object's <see cref="Godot.Rid"/>.</param>
    /// <param name="Shape">The shape index of the colliding shape.</param>
    public record struct IntersectShapeResult(
        GodotObject Collider,
        long ColliderId,
        Rid Rid,
        long Shape
    )
    {
        internal static readonly Variant ColliderKey = "collider";
        internal static readonly Variant ColliderIdKey = "collider_id";
        internal static readonly Variant RidKey = "rid";
        internal static readonly Variant ShapeKey = "shape";
    }

    /// <summary>
    /// Checks the intersections of a shape, given through a <see cref="Godot.PhysicsShapeQueryParameters3D"/> object, against the space. The intersected shapes are filled into the <paramref name="results"/> list.
    /// </summary>
    /// <remarks>
    /// The number of intersections can be limited with the <paramref name="maxResults"/> parameter, to reduce the processing time.
    /// </remarks>
    /// <remarks>
    /// This method does not take into account the <c>motion</c> property of the object.
    /// </remarks>
    public unsafe void IntersectShapeNonAlloc(PhysicsShapeQueryParameters3D parameters, List<IntersectShapeResult> results, int maxResults = 32)
    {
        var method = MethodBind2;
        var ptr = GodotObject.GetPtr(this);
        var arg1 = GodotObject.GetPtr(parameters);
        var arg2 = maxResults;
        ExceptionUtils.ThrowIfNullPtr(ptr);
        godot_array ret = default;
        long arg2_in = arg2;
        void** call_args = stackalloc void*[2] { &arg1, &arg2_in };
        NativeFuncs.godotsharp_method_bind_ptrcall(method, ptr, call_args, &ret);
        for (int i = 0; i < ret.Size; i++)
        {
            if (results.Count >= maxResults) break;
            var item = ret.Elements[i];
            if (item.Type != Variant.Type.Dictionary) continue;
            var dict = item.Dictionary;
            if (!NativeFuncs.godotsharp_dictionary_try_get_value(ref dict, (godot_variant)IntersectShapeResult.ColliderKey.NativeVar, out var colliderValue).ToBool()
                || !NativeFuncs.godotsharp_dictionary_try_get_value(ref dict, (godot_variant)IntersectShapeResult.ColliderIdKey.NativeVar, out var colliderIdValue).ToBool()
                || !NativeFuncs.godotsharp_dictionary_try_get_value(ref dict, (godot_variant)IntersectShapeResult.RidKey.NativeVar, out var ridValue).ToBool()
                || !NativeFuncs.godotsharp_dictionary_try_get_value(ref dict, (godot_variant)IntersectShapeResult.ShapeKey.NativeVar, out var shapeValue).ToBool())
                continue;
            results.Add(new IntersectShapeResult(
                VariantUtils.ConvertToGodotObject(colliderValue),
                colliderIdValue.Int,
                ridValue.Rid,
                shapeValue.Int
            ));
        }
        NativeFuncs.godotsharp_array_destroy(ref ret);
    }

    /// <summary>
    /// Stores the result of the <see cref="CastMotionNonAlloc"/> method.
    /// </summary>
    /// <param name="Safe">The maximum fraction of the motion that can be made without a collision.</param>
    /// <param name="Unsafe">The minimum fraction of the distance that must be moved for a collision.</param>
    public record struct CastMotionResult(
        float Safe,
        float Unsafe
    );

    /// <summary>
    /// Checks how far a <see cref="Godot.Shape3D"/> can move without colliding. All the parameters for the query, including the shape and the motion, are supplied through a <see cref="Godot.PhysicsShapeQueryParameters3D"/> object.
    /// </summary>
    /// <returns>A <see cref="CastMotionResult"/> struct containing the safe and unsafe proportions of the motion, or <c>[1.0, 1.0]</c> if no collision is detected.</returns>
    /// <remarks>
    /// Any <see cref="Godot.Shape3D"/>s that the shape is already colliding with e.g. inside of, will be ignored. Use <see cref="Godot.PhysicsDirectSpaceState3D.CollideShape(PhysicsShapeQueryParameters3D, int)"/> to determine the <see cref="Godot.Shape3D"/>s that the shape is already colliding with.
    /// </remarks>
    public unsafe CastMotionResult CastMotionNonAlloc(PhysicsShapeQueryParameters3D parameters)
    {
        var method = MethodBind3;
        var ptr = GodotObject.GetPtr(this);
        var arg1 = GodotObject.GetPtr(parameters);
        ExceptionUtils.ThrowIfNullPtr(ptr);
        godot_packed_float32_array ret = default;
        void** call_args = stackalloc void*[1] { &arg1 };
        NativeFuncs.godotsharp_method_bind_ptrcall(method, ptr, call_args, &ret);
        if (ret.Size != 2)
        {
            NativeFuncs.godotsharp_packed_float32_array_destroy(ref ret);
            return new CastMotionResult(1.0f, 1.0f);
        }
        var safeValue = ret.Buffer[0];
        var unsafeValue = ret.Buffer[1];
        NativeFuncs.godotsharp_packed_float32_array_destroy(ref ret);
        return new CastMotionResult(safeValue, unsafeValue);
    }

    /// <summary>
    /// Checks the intersections of a shape, given through a <see cref="Godot.PhysicsShapeQueryParameters3D"/> object, against the space. The points where the shape intersects another are filled into the <paramref name="results"/> list.
    /// </summary>
    /// <remarks>
    /// The number of intersections can be limited with the <paramref name="maxResults"/> parameter, to reduce the processing time.
    /// </remarks>
    /// <remarks>
    /// This method does not take into account the <c>motion</c> property of the object.
    /// </remarks>
    public unsafe void CollideShapeNonAlloc(PhysicsShapeQueryParameters3D parameters, List<Vector3> results, int maxResults = 32)
    {
        var method = MethodBind4;
        var ptr = GodotObject.GetPtr(this);
        var arg1 = GodotObject.GetPtr(parameters);
        var arg2 = maxResults;
        ExceptionUtils.ThrowIfNullPtr(ptr);
        godot_array ret = default;
        long arg2_in = arg2;
        void** call_args = stackalloc void*[2] { &arg1, &arg2_in };
        NativeFuncs.godotsharp_method_bind_ptrcall(method, ptr, call_args, &ret);
        for (int i = 0; i < ret.Size; i++)
        {
            if (results.Count >= maxResults) break;
            var item = ret.Elements[i];
            if (item.Type != Variant.Type.Vector3) continue;
            results.Add(item.Vector3);
        }
        NativeFuncs.godotsharp_array_destroy(ref ret);
    }

    /// <summary>
    /// Stores the result of the <see cref="GetRestInfoNonAlloc"/> method.
    /// </summary>
    /// <param name="ColliderId">The colliding object's ID.</param>
    /// <param name="LinearVelocity">The colliding object's velocity <see cref="Godot.Vector3"/>. If the object is an <see cref="Godot.Area3D"/>, the result is <c>(0, 0, 0)</c>.</param>
    /// <param name="Normal">The collision normal of the query shape at the intersection point, pointing away from the intersecting object.</param>
    /// <param name="Point">The intersection point.</param>
    /// <param name="Rid">The intersecting object's <see cref="Godot.Rid"/>.</param>
    /// <param name="Shape">The shape index of the colliding shape.</param>
    public record struct GetRestInfoResult(
        long ColliderId,
        Vector3 LinearVelocity,
        Vector3 Normal,
        Vector3 Point,
        Rid Rid,
        long Shape
    )
    {
        internal static readonly Variant ColliderIdKey = "collider_id";
        internal static readonly Variant LinearVelocityKey = "linear_velocity";
        internal static readonly Variant NormalKey = "normal";
        internal static readonly Variant PointKey = "point";
        internal static readonly Variant RidKey = "rid";
        internal static readonly Variant ShapeKey = "shape";
    }

    /// <summary>
    /// Checks the intersections of a shape, given through a <see cref="Godot.PhysicsShapeQueryParameters3D"/> object, against the space. If it collides with more than one shape, the nearest one is selected.
    /// </summary>
    /// <returns>
    /// A <see cref="GetRestInfoResult"/> struct containing the result of the shape intersection, or <see langword="null"/> if no intersection occurred.
    /// </returns>
    /// <remarks>
    /// This method does not take into account the <c>motion</c> property of the object.
    /// </remarks>
    public unsafe GetRestInfoResult? GetRestInfoNonAlloc(PhysicsShapeQueryParameters3D parameters)
    {
        var method = MethodBind5;
        var ptr = GodotObject.GetPtr(this);
        var arg1 = GodotObject.GetPtr(parameters);
        ExceptionUtils.ThrowIfNullPtr(ptr);
        godot_dictionary ret = default;
        void** call_args = stackalloc void*[1] { &arg1 };
        NativeFuncs.godotsharp_method_bind_ptrcall(method, ptr, call_args, &ret);
        if (NativeFuncs.godotsharp_dictionary_count(ref ret) == 0)
        {
            NativeFuncs.godotsharp_dictionary_destroy(ref ret);
            return null;
        }
        if (!NativeFuncs.godotsharp_dictionary_try_get_value(ref ret, (godot_variant)GetRestInfoResult.ColliderIdKey.NativeVar, out godot_variant colliderIdValue).ToBool()
            || !NativeFuncs.godotsharp_dictionary_try_get_value(ref ret, (godot_variant)GetRestInfoResult.LinearVelocityKey.NativeVar, out godot_variant linearVelocityValue).ToBool()
            || !NativeFuncs.godotsharp_dictionary_try_get_value(ref ret, (godot_variant)GetRestInfoResult.NormalKey.NativeVar, out godot_variant normalValue).ToBool()
            || !NativeFuncs.godotsharp_dictionary_try_get_value(ref ret, (godot_variant)GetRestInfoResult.PointKey.NativeVar, out godot_variant pointValue).ToBool()
            || !NativeFuncs.godotsharp_dictionary_try_get_value(ref ret, (godot_variant)GetRestInfoResult.RidKey.NativeVar, out godot_variant ridValue).ToBool()
            || !NativeFuncs.godotsharp_dictionary_try_get_value(ref ret, (godot_variant)GetRestInfoResult.ShapeKey.NativeVar, out godot_variant shapeValue).ToBool())
        {
            NativeFuncs.godotsharp_dictionary_destroy(ref ret);
            return null;
        }
        var returnValue = new GetRestInfoResult(
            colliderIdValue.Int,
            linearVelocityValue.Vector3,
            normalValue.Vector3,
            pointValue.Vector3,
            ridValue.Rid,
            shapeValue.Int
        );
        NativeFuncs.godotsharp_dictionary_destroy(ref ret);
        return returnValue;
    }
}
