using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Godot.NativeInterop;

namespace Godot;

#pragma warning disable IDE0040

file static class NativeDictionary
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsEmpty(scoped ref godot_dictionary dictionary) =>
        NativeFuncs.godotsharp_dictionary_count(ref dictionary) == 0;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool TryGetValue(scoped ref godot_dictionary dictionary, Variant key, out godot_variant value) =>
        NativeFuncs.godotsharp_dictionary_try_get_value(ref dictionary, key.NativeVar.DangerousSelfRef, out value).ToBool();
}

file static class NativeArray
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void Fill<[MustBeVariant] T>(scoped ref godot_array array, List<T> results)
    {
        ArgumentNullException.ThrowIfNull(results);

        results.EnsureCapacity(results.Count + array.Size);

        for (var i = 0; i < array.Size; i++)
        {
            results.Add(VariantUtils.ConvertTo<T>(array.Elements[i]));
        }
    }
}

partial class Node
{
    /// <summary>
    /// <para>Stores all children of this node into the provided <paramref name="results"/> list.</para>
    /// <para>If <paramref name="includeInternal"/> is <see langword="false"/>, excludes internal children when populating the list (see <see cref="Godot.Node.AddChild(Node, bool, Node.InternalMode)"/>'s <c>internal</c> parameter).</para>
    /// </summary>
    public unsafe void GetChildrenNonAlloc(List<Node> results, bool includeInternal = false)
    {
        var instancePtr = GetPtr(this);
        ExceptionUtils.ThrowIfNullPtr(instancePtr);
        var callArgs = stackalloc void*[1] { &includeInternal };
        godot_array ret = default;
        NativeFuncs.godotsharp_method_bind_ptrcall(MethodBind9, instancePtr, callArgs, &ret);
        try
        {
            NativeArray.Fill(ref ret, results);
        }
        finally
        {
            ret.Dispose();
        }
    }
}

partial class TileMap
{
    /// <summary>
    /// Stores the list of all neighboring cells to the one at <paramref name="coords"/> into the provided <paramref name="results"/> list.
    /// </summary>
    public unsafe void GetSurroundingCellsNonAlloc(Vector2I coords, List<Vector2I> results)
    {
        var instancePtr = GetPtr(this);
        ExceptionUtils.ThrowIfNullPtr(instancePtr);
        var coords_in = coords;
        var callArgs = stackalloc void*[1] { &coords_in };
        godot_array ret = default;
        NativeFuncs.godotsharp_method_bind_ptrcall(MethodBind54, instancePtr, callArgs, &ret);
        try
        {
            NativeArray.Fill(ref ret, results);
        }
        finally
        {
            ret.Dispose();
        }
    }

    /// <summary>
    /// Stores the positions of all cells containing a tile in the given <paramref name="layer"/> into the provided <paramref name="results"/> list.
    /// </summary>
    public unsafe void GetUsedCellsNonAlloc(int layer, List<Vector2I> results)
    {
        var instancePtr = GetPtr(this);
        ExceptionUtils.ThrowIfNullPtr(instancePtr);
        long layer_in = layer;
        var callArgs = stackalloc void*[1] { &layer_in };
        godot_array ret = default;
        NativeFuncs.godotsharp_method_bind_ptrcall(MethodBind55, instancePtr, callArgs, &ret);
        try
        {
            NativeArray.Fill(ref ret, results);
        }
        finally
        {
            ret.Dispose();
        }
    }

    /// <summary>
    /// Stores the positions of all cells containing a tile in the given <paramref name="layer"/> that match the provided filters into the provided <paramref name="results"/> list.
    /// Tiles may be filtered according to their source, atlas coordinates, or alternative ID.
    /// </summary>
    public unsafe void GetUsedCellsByIdNonAlloc(int layer, List<Vector2I> results, int sourceId = -1, Vector2I? atlasCoords = null, int alternativeTile = -1)
    {
        var instancePtr = GetPtr(this);
        ExceptionUtils.ThrowIfNullPtr(instancePtr);
        long layer_in = layer;
        long source_id_in = sourceId;
        var atlas_coords_in = atlasCoords ?? new Vector2I(-1, -1);
        long alternative_tile_in = alternativeTile;
        var callArgs = stackalloc void*[4] { &layer_in, &source_id_in, &atlas_coords_in, &alternative_tile_in };
        godot_array ret = default;
        NativeFuncs.godotsharp_method_bind_ptrcall(MethodBind56, instancePtr, callArgs, &ret);
        try
        {
            NativeArray.Fill(ref ret, results);
        }
        finally
        {
            ret.Dispose();
        }
    }
}

partial class TileMapLayer
{
    /// <summary>
    /// Stores the list of all neighboring cells to the one at <paramref name="coords"/> into the provided <paramref name="results"/> list.
    /// </summary>
    public unsafe void GetSurroundingCellsNonAlloc(Vector2I coords, List<Vector2I> results)
    {
        var instancePtr = GetPtr(this);
        ExceptionUtils.ThrowIfNullPtr(instancePtr);
        var coords_in = coords;
        var callArgs = stackalloc void*[1] { &coords_in };
        godot_array ret = default;
        NativeFuncs.godotsharp_method_bind_ptrcall(MethodBind23, instancePtr, callArgs, &ret);
        try
        {
            NativeArray.Fill(ref ret, results);
        }
        finally
        {
            ret.Dispose();
        }
    }

    /// <summary>
    /// Stores the positions of all cells containing a tile in this layer into the provided <paramref name="results"/> list.
    /// </summary>
    public unsafe void GetUsedCellsNonAlloc(List<Vector2I> results)
    {
        var instancePtr = GetPtr(this);
        ExceptionUtils.ThrowIfNullPtr(instancePtr);
        godot_array ret = default;
        NativeFuncs.godotsharp_method_bind_ptrcall(MethodBind11, instancePtr, null, &ret);
        try
        {
            NativeArray.Fill(ref ret, results);
        }
        finally
        {
            ret.Dispose();
        }
    }

    /// <summary>
    /// Stores the positions of all cells containing a tile in this layer that match the provided filters into the provided <paramref name="results"/> list.
    /// Tiles may be filtered according to their source, atlas coordinates, or alternative ID.
    /// </summary>
    public unsafe void GetUsedCellsByIdNonAlloc(List<Vector2I> results, int sourceId = -1, Vector2I? atlasCoords = null, int alternativeTile = -1)
    {
        var instancePtr = GetPtr(this);
        ExceptionUtils.ThrowIfNullPtr(instancePtr);
        long source_id_in = sourceId;
        var atlas_coords_in = atlasCoords ?? new Vector2I(-1, -1);
        long alternative_tile_in = alternativeTile;
        var callArgs = stackalloc void*[3] { &source_id_in, &atlas_coords_in, &alternative_tile_in };
        godot_array ret = default;
        NativeFuncs.godotsharp_method_bind_ptrcall(MethodBind12, instancePtr, callArgs, &ret);
        try
        {
            NativeArray.Fill(ref ret, results);
        }
        finally
        {
            ret.Dispose();
        }
    }
}

partial class TileMapPattern
{
    /// <summary>
    /// Stores the list of used cell coordinates in this pattern into the provided <paramref name="results"/> list.
    /// </summary>
    public unsafe void GetUsedCellsNonAlloc(List<Vector2I> results)
    {
        var instancePtr = GetPtr(this);
        ExceptionUtils.ThrowIfNullPtr(instancePtr);
        godot_array ret = default;
        NativeFuncs.godotsharp_method_bind_ptrcall(MethodBind6, instancePtr, null, &ret);
        try
        {
            NativeArray.Fill(ref ret, results);
        }
        finally
        {
            ret.Dispose();
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
        ArgumentNullException.ThrowIfNull(results);
        var instancePtr = GetPtr(this);
        ExceptionUtils.ThrowIfNullPtr(instancePtr);
        var arg1 = GetPtr(parameters);
        long arg2 = maxResults;
        var callArgs = stackalloc void*[2] { &arg1, &arg2 };
        godot_array ret = default;
        NativeFuncs.godotsharp_method_bind_ptrcall(MethodBind0, instancePtr, callArgs, &ret);
        try
        {
            results.EnsureCapacity(results.Count + ret.Size);
            for (int i = 0; i < ret.Size; i++)
            {
                var dict = ret.Elements[i].Dictionary;
                if (!NativeDictionary.TryGetValue(ref dict, IntersectPointResult.ColliderKey, out var colliderValue)
                    || !NativeDictionary.TryGetValue(ref dict, IntersectPointResult.ColliderIdKey, out var colliderIdValue)
                    || !NativeDictionary.TryGetValue(ref dict, IntersectPointResult.RidKey, out var ridValue)
                    || !NativeDictionary.TryGetValue(ref dict, IntersectPointResult.ShapeKey, out var shapeValue))
                {
                    throw new InvalidOperationException($"IntersectPoint result item #{i} was invalid");
                }

                results.Add(new IntersectPointResult(
                    VariantUtils.ConvertToGodotObject(colliderValue),
                    colliderIdValue.Int,
                    ridValue.Rid,
                    shapeValue.Int
                ));
            }
        }
        finally
        {
            ret.Dispose();
        }
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
        var instancePtr = GetPtr(this);
        ExceptionUtils.ThrowIfNullPtr(instancePtr);
        var arg1 = GetPtr(parameters);
        var callArgs = stackalloc void*[1] { &arg1 };
        godot_dictionary ret = default;
        NativeFuncs.godotsharp_method_bind_ptrcall(MethodBind1, instancePtr, callArgs, &ret);
        try
        {
            if (NativeDictionary.IsEmpty(ref ret))
            {
                return null;
            }

            if (!NativeDictionary.TryGetValue(ref ret, IntersectRayResult.ColliderKey, out var colliderValue)
                || !NativeDictionary.TryGetValue(ref ret, IntersectRayResult.ColliderIdKey, out var colliderIdValue)
                || !NativeDictionary.TryGetValue(ref ret, IntersectRayResult.NormalKey, out var normalValue)
                || !NativeDictionary.TryGetValue(ref ret, IntersectRayResult.PositionKey, out var positionValue)
                || !NativeDictionary.TryGetValue(ref ret, IntersectRayResult.RidKey, out var ridValue)
                || !NativeDictionary.TryGetValue(ref ret, IntersectRayResult.ShapeKey, out var shapeValue))
            {
                throw new InvalidOperationException("IntersectRay result was invalid");
            }

            return new IntersectRayResult(
                VariantUtils.ConvertToGodotObject(colliderValue),
                colliderIdValue.Int,
                normalValue.Vector2,
                positionValue.Vector2,
                ridValue.Rid,
                shapeValue.Int
            );
        }
        finally
        {
            ret.Dispose();
        }
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
        ArgumentNullException.ThrowIfNull(results);
        var instancePtr = GetPtr(this);
        ExceptionUtils.ThrowIfNullPtr(instancePtr);
        var arg1 = GetPtr(parameters);
        long arg2 = maxResults;
        var callArgs = stackalloc void*[2] { &arg1, &arg2 };
        godot_array ret = default;
        NativeFuncs.godotsharp_method_bind_ptrcall(MethodBind2, instancePtr, callArgs, &ret);
        try
        {
            results.EnsureCapacity(results.Count + ret.Size);
            for (int i = 0; i < ret.Size; i++)
            {
                var dict = ret.Elements[i].Dictionary;
                if (!NativeDictionary.TryGetValue(ref dict, IntersectShapeResult.ColliderKey, out var colliderValue)
                    || !NativeDictionary.TryGetValue(ref dict, IntersectShapeResult.ColliderIdKey, out var colliderIdValue)
                    || !NativeDictionary.TryGetValue(ref dict, IntersectShapeResult.RidKey, out var ridValue)
                    || !NativeDictionary.TryGetValue(ref dict, IntersectShapeResult.ShapeKey, out var shapeValue))
                {
                    throw new InvalidOperationException($"IntersectShape result item #{i} was invalid");
                }

                results.Add(new IntersectShapeResult(
                    VariantUtils.ConvertToGodotObject(colliderValue),
                    colliderIdValue.Int,
                    ridValue.Rid,
                    shapeValue.Int
                ));
            }
        }
        finally
        {
            ret.Dispose();
        }
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
        var instancePtr = GetPtr(this);
        ExceptionUtils.ThrowIfNullPtr(instancePtr);
        var arg1 = GetPtr(parameters);
        var callArgs = stackalloc void*[1] { &arg1 };
        godot_packed_float32_array ret = default;
        NativeFuncs.godotsharp_method_bind_ptrcall(MethodBind3, instancePtr, callArgs, &ret);
        try
        {
            if (ret.Size < 2)
            {
                return new CastMotionResult(1.0f, 1.0f);
            }

            return new CastMotionResult(ret.Buffer[0], ret.Buffer[1]);
        }
        finally
        {
            ret.Dispose();
        }
    }

    /// <summary>
    /// Checks the intersections of a shape, given through a <see cref="Godot.PhysicsShapeQueryParameters2D"/> object, against the space. The points where the shape intersects another are filled into the <paramref name="results"/> list.
    /// </summary>
    /// <remarks>
    /// The number of intersections can be limited with the <paramref name="maxResults"/> parameter, to reduce the processing time.
    /// </remarks>
    /// <remarks>
    /// This method does not take into account the <c>motion</c> property of the object.
    /// </remarks>
    public unsafe void CollideShapeNonAlloc(PhysicsShapeQueryParameters2D parameters, List<Vector2> results, int maxResults = 32)
    {
        var instancePtr = GetPtr(this);
        ExceptionUtils.ThrowIfNullPtr(instancePtr);
        var arg1 = GetPtr(parameters);
        long arg2 = maxResults;
        var callArgs = stackalloc void*[2] { &arg1, &arg2 };
        godot_array ret = default;
        NativeFuncs.godotsharp_method_bind_ptrcall(MethodBind4, instancePtr, callArgs, &ret);
        try
        {
            NativeArray.Fill(ref ret, results);
        }
        finally
        {
            ret.Dispose();
        }
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
        var instancePtr = GetPtr(this);
        ExceptionUtils.ThrowIfNullPtr(instancePtr);
        var arg1 = GetPtr(parameters);
        var callArgs = stackalloc void*[1] { &arg1 };
        godot_dictionary ret = default;
        NativeFuncs.godotsharp_method_bind_ptrcall(MethodBind5, instancePtr, callArgs, &ret);
        try
        {
            if (NativeDictionary.IsEmpty(ref ret))
            {
                return null;
            }

            if (!NativeDictionary.TryGetValue(ref ret, GetRestInfoResult.ColliderIdKey, out var colliderIdValue)
                || !NativeDictionary.TryGetValue(ref ret, GetRestInfoResult.LinearVelocityKey, out var linearVelocityValue)
                || !NativeDictionary.TryGetValue(ref ret, GetRestInfoResult.NormalKey, out var normalValue)
                || !NativeDictionary.TryGetValue(ref ret, GetRestInfoResult.PointKey, out var pointValue)
                || !NativeDictionary.TryGetValue(ref ret, GetRestInfoResult.RidKey, out var ridValue)
                || !NativeDictionary.TryGetValue(ref ret, GetRestInfoResult.ShapeKey, out var shapeValue))
            {
                throw new InvalidOperationException("GetRestInfo result was invalid");
            }

            return new GetRestInfoResult(
                colliderIdValue.Int,
                linearVelocityValue.Vector2,
                normalValue.Vector2,
                pointValue.Vector2,
                ridValue.Rid,
                shapeValue.Int
            );
        }
        finally
        {
            ret.Dispose();
        }
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
        ArgumentNullException.ThrowIfNull(results);
        var instancePtr = GetPtr(this);
        ExceptionUtils.ThrowIfNullPtr(instancePtr);
        var arg1 = GetPtr(parameters);
        long arg2 = maxResults;
        var callArgs = stackalloc void*[2] { &arg1, &arg2 };
        godot_array ret = default;
        NativeFuncs.godotsharp_method_bind_ptrcall(MethodBind0, instancePtr, callArgs, &ret);
        try
        {
            results.EnsureCapacity(results.Count + ret.Size);
            for (int i = 0; i < ret.Size; i++)
            {
                var dict = ret.Elements[i].Dictionary;
                if (!NativeDictionary.TryGetValue(ref dict, IntersectPointResult.ColliderKey, out var colliderValue)
                    || !NativeDictionary.TryGetValue(ref dict, IntersectPointResult.ColliderIdKey, out var colliderIdValue)
                    || !NativeDictionary.TryGetValue(ref dict, IntersectPointResult.RidKey, out var ridValue)
                    || !NativeDictionary.TryGetValue(ref dict, IntersectPointResult.ShapeKey, out var shapeValue))
                {
                    throw new InvalidOperationException($"IntersectPoint result item #{i} was invalid");
                }

                results.Add(new IntersectPointResult(
                    VariantUtils.ConvertToGodotObject(colliderValue),
                    colliderIdValue.Int,
                    ridValue.Rid,
                    shapeValue.Int
                ));
            }
        }
        finally
        {
            ret.Dispose();
        }
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
        var instancePtr = GetPtr(this);
        ExceptionUtils.ThrowIfNullPtr(instancePtr);
        var arg1 = GetPtr(parameters);
        var callArgs = stackalloc void*[1] { &arg1 };
        godot_dictionary ret = default;
        NativeFuncs.godotsharp_method_bind_ptrcall(MethodBind1, instancePtr, callArgs, &ret);
        try
        {
            if (NativeDictionary.IsEmpty(ref ret))
            {
                return null;
            }

            if (!NativeDictionary.TryGetValue(ref ret, IntersectRayResult.ColliderKey, out var colliderValue)
                || !NativeDictionary.TryGetValue(ref ret, IntersectRayResult.ColliderIdKey, out var colliderIdValue)
                || !NativeDictionary.TryGetValue(ref ret, IntersectRayResult.NormalKey, out var normalValue)
                || !NativeDictionary.TryGetValue(ref ret, IntersectRayResult.PositionKey, out var positionValue)
                || !NativeDictionary.TryGetValue(ref ret, IntersectRayResult.FaceIndexKey, out var faceIndexValue)
                || !NativeDictionary.TryGetValue(ref ret, IntersectRayResult.RidKey, out var ridValue)
                || !NativeDictionary.TryGetValue(ref ret, IntersectRayResult.ShapeKey, out var shapeValue))
            {
                throw new InvalidOperationException("IntersectRay result was invalid");
            }

            return new IntersectRayResult(
                VariantUtils.ConvertToGodotObject(colliderValue),
                colliderIdValue.Int,
                normalValue.Vector3,
                positionValue.Vector3,
                faceIndexValue.Int,
                ridValue.Rid,
                shapeValue.Int
            );
        }
        finally
        {
            ret.Dispose();
        }
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
        ArgumentNullException.ThrowIfNull(results);
        var instancePtr = GetPtr(this);
        ExceptionUtils.ThrowIfNullPtr(instancePtr);
        var arg1 = GetPtr(parameters);
        long arg2 = maxResults;
        var callArgs = stackalloc void*[2] { &arg1, &arg2 };
        godot_array ret = default;
        NativeFuncs.godotsharp_method_bind_ptrcall(MethodBind2, instancePtr, callArgs, &ret);
        try
        {
            results.EnsureCapacity(results.Count + ret.Size);
            for (int i = 0; i < ret.Size; i++)
            {
                var dict = ret.Elements[i].Dictionary;
                if (!NativeDictionary.TryGetValue(ref dict, IntersectShapeResult.ColliderKey, out var colliderValue)
                    || !NativeDictionary.TryGetValue(ref dict, IntersectShapeResult.ColliderIdKey, out var colliderIdValue)
                    || !NativeDictionary.TryGetValue(ref dict, IntersectShapeResult.RidKey, out var ridValue)
                    || !NativeDictionary.TryGetValue(ref dict, IntersectShapeResult.ShapeKey, out var shapeValue))
                {
                    throw new InvalidOperationException($"IntersectShape result item #{i} was invalid");
                }

                results.Add(new IntersectShapeResult(
                    VariantUtils.ConvertToGodotObject(colliderValue),
                    colliderIdValue.Int,
                    ridValue.Rid,
                    shapeValue.Int
                ));
            }
        }
        finally
        {
            ret.Dispose();
        }
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
        var instancePtr = GetPtr(this);
        ExceptionUtils.ThrowIfNullPtr(instancePtr);
        var arg1 = GetPtr(parameters);
        var callArgs = stackalloc void*[1] { &arg1 };
        godot_packed_float32_array ret = default;
        NativeFuncs.godotsharp_method_bind_ptrcall(MethodBind3, instancePtr, callArgs, &ret);
        try
        {
            if (ret.Size < 2)
            {
                return new CastMotionResult(1.0f, 1.0f);
            }

            return new CastMotionResult(ret.Buffer[0], ret.Buffer[1]);
        }
        finally
        {
            ret.Dispose();
        }
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
        var instancePtr = GetPtr(this);
        ExceptionUtils.ThrowIfNullPtr(instancePtr);
        var arg1 = GetPtr(parameters);
        long arg2 = maxResults;
        var callArgs = stackalloc void*[2] { &arg1, &arg2 };
        godot_array ret = default;
        NativeFuncs.godotsharp_method_bind_ptrcall(MethodBind4, instancePtr, callArgs, &ret);
        try
        {
            NativeArray.Fill(ref ret, results);
        }
        finally
        {
            ret.Dispose();
        }
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
        var instancePtr = GetPtr(this);
        ExceptionUtils.ThrowIfNullPtr(instancePtr);
        var arg1 = GetPtr(parameters);
        var callArgs = stackalloc void*[1] { &arg1 };
        godot_dictionary ret = default;
        NativeFuncs.godotsharp_method_bind_ptrcall(MethodBind5, instancePtr, callArgs, &ret);
        try
        {
            if (NativeDictionary.IsEmpty(ref ret))
            {
                return null;
            }

            if (!NativeDictionary.TryGetValue(ref ret, GetRestInfoResult.ColliderIdKey, out var colliderIdValue)
                || !NativeDictionary.TryGetValue(ref ret, GetRestInfoResult.LinearVelocityKey, out var linearVelocityValue)
                || !NativeDictionary.TryGetValue(ref ret, GetRestInfoResult.NormalKey, out var normalValue)
                || !NativeDictionary.TryGetValue(ref ret, GetRestInfoResult.PointKey, out var pointValue)
                || !NativeDictionary.TryGetValue(ref ret, GetRestInfoResult.RidKey, out var ridValue)
                || !NativeDictionary.TryGetValue(ref ret, GetRestInfoResult.ShapeKey, out var shapeValue))
            {
                throw new InvalidOperationException("GetRestInfo result was invalid");
            }

            return new GetRestInfoResult(
                colliderIdValue.Int,
                linearVelocityValue.Vector3,
                normalValue.Vector3,
                pointValue.Vector3,
                ridValue.Rid,
                shapeValue.Int
            );
        }
        finally
        {
            ret.Dispose();
        }
    }
}

partial class PhysicsBody2D
{
    /// <summary>
    /// Stores the result of the <see cref="MoveAndCollideNonAlloc"/> method.
    /// </summary>
    /// <param name="ColliderId">The colliding body's attached <see cref="Godot.GodotObject"/> instance ID.</param>
    /// <param name="ColliderShapeIndex">The colliding body's shape index.</param>
    /// <param name="ColliderVelocity">The colliding body's velocity.</param>
    /// <param name="Depth">The colliding body's length of overlap along the collision normal.</param>
    /// <param name="Normal">The colliding body's shape's normal at the point of collision.</param>
    /// <param name="Position">The point of collision in global coordinates.</param>
    /// <param name="Remainder">The moving object's remaining movement vector.</param>
    /// <param name="Travel">The moving object's travel before collision.</param>
    public record struct KinematicCollision2DResult(
        ulong ColliderId,
        int ColliderShapeIndex,
        Vector2 ColliderVelocity,
        float Depth,
        Vector2 Normal,
        Vector2 Position,
        Vector2 Remainder,
        Vector2 Travel
    );

    private static nint KinematicCollision2D_MethodBind0_Backing;
    private static nint KinematicCollision2D_MethodBind0
    {
        get
        {
            if (KinematicCollision2D_MethodBind0_Backing == 0)
            {
                KinematicCollision2D_MethodBind0_Backing = KinematicCollision2D_MethodBind0(null!);
            }
            return KinematicCollision2D_MethodBind0_Backing;

            [UnsafeAccessor(UnsafeAccessorKind.StaticField, Name = "MethodBind0")]
            static extern ref nint KinematicCollision2D_MethodBind0(KinematicCollision2D _);
        }
    }

    private static nint KinematicCollision2D_MethodBind1_Backing;
    private static nint KinematicCollision2D_MethodBind1
    {
        get
        {
            if (KinematicCollision2D_MethodBind1_Backing == 0)
            {
                KinematicCollision2D_MethodBind1_Backing = KinematicCollision2D_MethodBind1(null!);
            }
            return KinematicCollision2D_MethodBind1_Backing;

            [UnsafeAccessor(UnsafeAccessorKind.StaticField, Name = "MethodBind1")]
            static extern ref nint KinematicCollision2D_MethodBind1(KinematicCollision2D _);
        }
    }

    private static nint KinematicCollision2D_MethodBind2_Backing;
    private static nint KinematicCollision2D_MethodBind2
    {
        get
        {
            if (KinematicCollision2D_MethodBind2_Backing == 0)
            {
                KinematicCollision2D_MethodBind2_Backing = KinematicCollision2D_MethodBind2(null!);
            }
            return KinematicCollision2D_MethodBind2_Backing;

            [UnsafeAccessor(UnsafeAccessorKind.StaticField, Name = "MethodBind2")]
            static extern ref nint KinematicCollision2D_MethodBind2(KinematicCollision2D _);
        }
    }

    private static nint KinematicCollision2D_MethodBind3_Backing;
    private static nint KinematicCollision2D_MethodBind3
    {
        get
        {
            if (KinematicCollision2D_MethodBind3_Backing == 0)
            {
                KinematicCollision2D_MethodBind3_Backing = KinematicCollision2D_MethodBind3(null!);
            }
            return KinematicCollision2D_MethodBind3_Backing;

            [UnsafeAccessor(UnsafeAccessorKind.StaticField, Name = "MethodBind3")]
            static extern ref nint KinematicCollision2D_MethodBind3(KinematicCollision2D _);
        }
    }

    private static nint KinematicCollision2D_MethodBind5_Backing;
    private static nint KinematicCollision2D_MethodBind5
    {
        get
        {
            if (KinematicCollision2D_MethodBind5_Backing == 0)
            {
                KinematicCollision2D_MethodBind5_Backing = KinematicCollision2D_MethodBind5(null!);
            }
            return KinematicCollision2D_MethodBind5_Backing;

            [UnsafeAccessor(UnsafeAccessorKind.StaticField, Name = "MethodBind5")]
            static extern ref nint KinematicCollision2D_MethodBind5(KinematicCollision2D _);
        }
    }

    private static nint KinematicCollision2D_MethodBind8_Backing;
    private static nint KinematicCollision2D_MethodBind8
    {
        get
        {
            if (KinematicCollision2D_MethodBind8_Backing == 0)
            {
                KinematicCollision2D_MethodBind8_Backing = KinematicCollision2D_MethodBind8(null!);
            }
            return KinematicCollision2D_MethodBind8_Backing;

            [UnsafeAccessor(UnsafeAccessorKind.StaticField, Name = "MethodBind8")]
            static extern ref nint KinematicCollision2D_MethodBind8(KinematicCollision2D _);
        }
    }

    private static nint KinematicCollision2D_MethodBind11_Backing;
    private static nint KinematicCollision2D_MethodBind11
    {
        get
        {
            if (KinematicCollision2D_MethodBind11_Backing == 0)
            {
                KinematicCollision2D_MethodBind11_Backing = KinematicCollision2D_MethodBind11(null!);
            }
            return KinematicCollision2D_MethodBind11_Backing;

            [UnsafeAccessor(UnsafeAccessorKind.StaticField, Name = "MethodBind11")]
            static extern ref nint KinematicCollision2D_MethodBind11(KinematicCollision2D _);
        }
    }

    private static nint KinematicCollision2D_MethodBind12_Backing;
    private static nint KinematicCollision2D_MethodBind12
    {
        get
        {
            if (KinematicCollision2D_MethodBind12_Backing == 0)
            {
                KinematicCollision2D_MethodBind12_Backing = KinematicCollision2D_MethodBind12(null!);
            }
            return KinematicCollision2D_MethodBind12_Backing;

            [UnsafeAccessor(UnsafeAccessorKind.StaticField, Name = "MethodBind12")]
            static extern ref nint KinematicCollision2D_MethodBind12(KinematicCollision2D _);
        }
    }

    /// <summary>
    /// Moves the body along a motion vector using the physics server, without creating a managed <see cref="Godot.KinematicCollision2D"/> wrapper.
    /// If the body collides with another, all collision data is returned in a <see cref="KinematicCollision2DResult"/> struct.
    /// </summary>
    /// <returns>A <see cref="KinematicCollision2DResult"/> struct containing the collision data, or <see langword="null"/> if no collision occurred.</returns>
    public unsafe KinematicCollision2DResult? MoveAndCollideNonAlloc(Vector2 motion, bool testOnly = false, float safeMargin = 0.08f, bool recoveryAsCollision = false)
    {
        var instancePtr = GetPtr(this);
        ExceptionUtils.ThrowIfNullPtr(instancePtr);
        var arg1 = motion;
        godot_bool arg2 = testOnly.ToGodotBool();
        double arg3 = safeMargin;
        godot_bool arg4 = recoveryAsCollision.ToGodotBool();
        var callArgs = stackalloc void*[4] { &arg1, &arg2, &arg3, &arg4 };
        godot_ref ret = default;
        NativeFuncs.godotsharp_method_bind_ptrcall(MethodBind0, instancePtr, callArgs, &ret);
        try
        {
            if (ret.IsNull)
            {
                return null;
            }

            nint nativeRef = ret.Reference;
            Vector2 position;
            NativeFuncs.godotsharp_method_bind_ptrcall(KinematicCollision2D_MethodBind0, nativeRef, null, &position);
            Vector2 normal;
            NativeFuncs.godotsharp_method_bind_ptrcall(KinematicCollision2D_MethodBind1, nativeRef, null, &normal);
            Vector2 travel;
            NativeFuncs.godotsharp_method_bind_ptrcall(KinematicCollision2D_MethodBind2, nativeRef, null, &travel);
            Vector2 remainder;
            NativeFuncs.godotsharp_method_bind_ptrcall(KinematicCollision2D_MethodBind3, nativeRef, null, &remainder);
            float depth;
            NativeFuncs.godotsharp_method_bind_ptrcall(KinematicCollision2D_MethodBind5, nativeRef, null, &depth);
            ulong colliderId;
            NativeFuncs.godotsharp_method_bind_ptrcall(KinematicCollision2D_MethodBind8, nativeRef, null, &colliderId);
            int colliderShapeIndex;
            NativeFuncs.godotsharp_method_bind_ptrcall(KinematicCollision2D_MethodBind11, nativeRef, null, &colliderShapeIndex);
            Vector2 colliderVelocity;
            NativeFuncs.godotsharp_method_bind_ptrcall(KinematicCollision2D_MethodBind12, nativeRef, null, &colliderVelocity);

            return new KinematicCollision2DResult(
                colliderId,
                colliderShapeIndex,
                colliderVelocity,
                depth,
                normal,
                position,
                remainder,
                travel
            );
        }
        finally
        {
            ret.Dispose();
        }
    }
}

partial class PhysicsBody3D
{
    /// <summary>
    /// Stores a single collision data entry for the <see cref="MoveAndCollideNonAlloc"/> method.
    /// </summary>
    /// <param name="ColliderId">The colliding body's attached <see cref="Godot.GodotObject"/> instance ID.</param>
    /// <param name="ColliderShapeIndex">The colliding body's shape index.</param>
    /// <param name="ColliderVelocity">The colliding body's velocity.</param>
    /// <param name="Normal">The colliding body's shape's normal at the point of collision.</param>
    /// <param name="Position">The point of collision in global coordinates.</param>
    public record struct KinematicCollision3DCollisionData(
        ulong ColliderId,
        int ColliderShapeIndex,
        Vector3 ColliderVelocity,
        Vector3 Normal,
        Vector3 Position
    );

    /// <summary>
    /// Stores the non-indexed result of the <see cref="MoveAndCollideNonAlloc"/> method.
    /// </summary>
    /// <param name="Depth">The colliding body's length of overlap along the collision normal.</param>
    /// <param name="Remainder">The moving object's remaining movement vector.</param>
    /// <param name="Travel">The moving object's travel before collision.</param>
    /// <param name="CollisionCount">The number of detected collisions.</param>
    public record struct KinematicCollision3DResult(
        float Depth,
        Vector3 Remainder,
        Vector3 Travel,
        int CollisionCount
    );

    private static nint KinematicCollision3D_MethodBind0_Backing;
    private static nint KinematicCollision3D_MethodBind0
    {
        get
        {
            if (KinematicCollision3D_MethodBind0_Backing == 0)
            {
                KinematicCollision3D_MethodBind0_Backing = KinematicCollision3D_MethodBind0(null!);
            }
            return KinematicCollision3D_MethodBind0_Backing;

            [UnsafeAccessor(UnsafeAccessorKind.StaticField, Name = "MethodBind0")]
            static extern ref nint KinematicCollision3D_MethodBind0(KinematicCollision3D _);
        }
    }

    private static nint KinematicCollision3D_MethodBind1_Backing;
    private static nint KinematicCollision3D_MethodBind1
    {
        get
        {
            if (KinematicCollision3D_MethodBind1_Backing == 0)
            {
                KinematicCollision3D_MethodBind1_Backing = KinematicCollision3D_MethodBind1(null!);
            }
            return KinematicCollision3D_MethodBind1_Backing;

            [UnsafeAccessor(UnsafeAccessorKind.StaticField, Name = "MethodBind1")]
            static extern ref nint KinematicCollision3D_MethodBind1(KinematicCollision3D _);
        }
    }

    private static nint KinematicCollision3D_MethodBind2_Backing;
    private static nint KinematicCollision3D_MethodBind2
    {
        get
        {
            if (KinematicCollision3D_MethodBind2_Backing == 0)
            {
                KinematicCollision3D_MethodBind2_Backing = KinematicCollision3D_MethodBind2(null!);
            }
            return KinematicCollision3D_MethodBind2_Backing;

            [UnsafeAccessor(UnsafeAccessorKind.StaticField, Name = "MethodBind2")]
            static extern ref nint KinematicCollision3D_MethodBind2(KinematicCollision3D _);
        }
    }

    private static nint KinematicCollision3D_MethodBind3_Backing;
    private static nint KinematicCollision3D_MethodBind3
    {
        get
        {
            if (KinematicCollision3D_MethodBind3_Backing == 0)
            {
                KinematicCollision3D_MethodBind3_Backing = KinematicCollision3D_MethodBind3(null!);
            }
            return KinematicCollision3D_MethodBind3_Backing;

            [UnsafeAccessor(UnsafeAccessorKind.StaticField, Name = "MethodBind3")]
            static extern ref nint KinematicCollision3D_MethodBind3(KinematicCollision3D _);
        }
    }

    private static nint KinematicCollision3D_MethodBind4_Backing;
    private static nint KinematicCollision3D_MethodBind4
    {
        get
        {
            if (KinematicCollision3D_MethodBind4_Backing == 0)
            {
                KinematicCollision3D_MethodBind4_Backing = KinematicCollision3D_MethodBind4(null!);
            }
            return KinematicCollision3D_MethodBind4_Backing;

            [UnsafeAccessor(UnsafeAccessorKind.StaticField, Name = "MethodBind4")]
            static extern ref nint KinematicCollision3D_MethodBind4(KinematicCollision3D _);
        }
    }

    private static nint KinematicCollision3D_MethodBind5_Backing;
    private static nint KinematicCollision3D_MethodBind5
    {
        get
        {
            if (KinematicCollision3D_MethodBind5_Backing == 0)
            {
                KinematicCollision3D_MethodBind5_Backing = KinematicCollision3D_MethodBind5(null!);
            }
            return KinematicCollision3D_MethodBind5_Backing;

            [UnsafeAccessor(UnsafeAccessorKind.StaticField, Name = "MethodBind5")]
            static extern ref nint KinematicCollision3D_MethodBind5(KinematicCollision3D _);
        }
    }

    private static nint KinematicCollision3D_MethodBind9_Backing;
    private static nint KinematicCollision3D_MethodBind9
    {
        get
        {
            if (KinematicCollision3D_MethodBind9_Backing == 0)
            {
                KinematicCollision3D_MethodBind9_Backing = KinematicCollision3D_MethodBind9(null!);
            }
            return KinematicCollision3D_MethodBind9_Backing;

            [UnsafeAccessor(UnsafeAccessorKind.StaticField, Name = "MethodBind9")]
            static extern ref nint KinematicCollision3D_MethodBind9(KinematicCollision3D _);
        }
    }

    private static nint KinematicCollision3D_MethodBind12_Backing;
    private static nint KinematicCollision3D_MethodBind12
    {
        get
        {
            if (KinematicCollision3D_MethodBind12_Backing == 0)
            {
                KinematicCollision3D_MethodBind12_Backing = KinematicCollision3D_MethodBind12(null!);
            }
            return KinematicCollision3D_MethodBind12_Backing;

            [UnsafeAccessor(UnsafeAccessorKind.StaticField, Name = "MethodBind12")]
            static extern ref nint KinematicCollision3D_MethodBind12(KinematicCollision3D _);
        }
    }

    private static nint KinematicCollision3D_MethodBind13_Backing;
    private static nint KinematicCollision3D_MethodBind13
    {
        get
        {
            if (KinematicCollision3D_MethodBind13_Backing == 0)
            {
                KinematicCollision3D_MethodBind13_Backing = KinematicCollision3D_MethodBind13(null!);
            }
            return KinematicCollision3D_MethodBind13_Backing;

            [UnsafeAccessor(UnsafeAccessorKind.StaticField, Name = "MethodBind13")]
            static extern ref nint KinematicCollision3D_MethodBind13(KinematicCollision3D _);
        }
    }

    /// <summary>
    /// Moves the body along a motion vector using the physics server, without creating a managed <see cref="Godot.KinematicCollision3D"/> wrapper.
    /// If the body collides with another, all collision data is returned in a <see cref="KinematicCollision3DResult"/> struct and individual collisions are filled into <paramref name="collisions"/>.
    /// </summary>
    /// <returns>A <see cref="KinematicCollision3DResult"/> struct containing the non-indexed collision data, or <see langword="null"/> if no collision occurred.</returns>
    public unsafe KinematicCollision3DResult? MoveAndCollideNonAlloc(Vector3 motion, bool testOnly = false, float safeMargin = 0.001f, bool recoveryAsCollision = false, int maxCollisions = 1, List<KinematicCollision3DCollisionData> collisions = null)
    {
        var instancePtr = GetPtr(this);
        ExceptionUtils.ThrowIfNullPtr(instancePtr);
        var arg1 = motion;
        godot_bool arg2 = testOnly.ToGodotBool();
        double arg3 = safeMargin;
        godot_bool arg4 = recoveryAsCollision.ToGodotBool();
        long arg5 = maxCollisions;
        var callArgs = stackalloc void*[5] { &arg1, &arg2, &arg3, &arg4, &arg5 };
        godot_ref ret = default;
        NativeFuncs.godotsharp_method_bind_ptrcall(MethodBind0, instancePtr, callArgs, &ret);
        try
        {
            if (ret.IsNull)
            {
                return null;
            }

            nint nativeRef = ret.Reference;
            Vector3 travel;
            NativeFuncs.godotsharp_method_bind_ptrcall(KinematicCollision3D_MethodBind0, nativeRef, null, &travel);
            Vector3 remainder;
            NativeFuncs.godotsharp_method_bind_ptrcall(KinematicCollision3D_MethodBind1, nativeRef, null, &remainder);
            float depth;
            NativeFuncs.godotsharp_method_bind_ptrcall(KinematicCollision3D_MethodBind2, nativeRef, null, &depth);
            int collisionCount;
            NativeFuncs.godotsharp_method_bind_ptrcall(KinematicCollision3D_MethodBind3, nativeRef, null, &collisionCount);

            if (collisions != null)
            {
                int count = Math.Min(collisionCount, maxCollisions);
                collisions.Clear();
                collisions.EnsureCapacity(count);
                long idx = 0;
                var idxArgs = stackalloc void*[1] { &idx };
                for (int i = 0; i < count; i++)
                {
                    idx = i;

                    Vector3 position;
                    NativeFuncs.godotsharp_method_bind_ptrcall(KinematicCollision3D_MethodBind4, nativeRef, idxArgs, &position);
                    Vector3 normal;
                    NativeFuncs.godotsharp_method_bind_ptrcall(KinematicCollision3D_MethodBind5, nativeRef, idxArgs, &normal);
                    ulong colliderId;
                    NativeFuncs.godotsharp_method_bind_ptrcall(KinematicCollision3D_MethodBind9, nativeRef, idxArgs, &colliderId);
                    int colliderShapeIndex;
                    NativeFuncs.godotsharp_method_bind_ptrcall(KinematicCollision3D_MethodBind12, nativeRef, idxArgs, &colliderShapeIndex);
                    Vector3 colliderVelocity;
                    NativeFuncs.godotsharp_method_bind_ptrcall(KinematicCollision3D_MethodBind13, nativeRef, idxArgs, &colliderVelocity);

                    collisions.Add(new KinematicCollision3DCollisionData(
                        colliderId,
                        colliderShapeIndex,
                        colliderVelocity,
                        normal,
                        position
                    ));
                }
            }

            return new KinematicCollision3DResult(
                depth,
                remainder,
                travel,
                collisionCount
            );
        }
        finally
        {
            ret.Dispose();
        }
    }
}

partial class SceneMultiplayer
{
    /// <summary>
    /// Represents the method that handles the <see cref="PeerPacketNonAlloc"/> event of a <see cref="SceneMultiplayer"/> class.
    /// </summary>
    /// <param name="id">The peer ID of the peer that sent the packet.</param>
    /// <param name="packet">
    /// The packet payload. Only valid for the duration of the handler; do not store or return this span.
    /// </param>
#pragma warning disable CA1711
    public delegate void PeerPacketNonAllocEventHandler(long id, ReadOnlySpan<byte> packet);
#pragma warning restore CA1711

    private static unsafe void PeerPacketNonAllocTrampoline(object delegateObj, NativeVariantPtrArgs args, out godot_variant ret)
    {
        Callable.ThrowIfArgCountMismatch(args, 2);
        long id = VariantUtils.ConvertTo<long>(args[0]);

        godot_packed_byte_array packed = NativeFuncs.godotsharp_variant_as_packed_byte_array(args[1]);
        try
        {
            ReadOnlySpan<byte> packet = packed.Size == 0
                ? ReadOnlySpan<byte>.Empty
                : new ReadOnlySpan<byte>(packed.Buffer, packed.Size);
            ((PeerPacketNonAllocEventHandler)delegateObj)(id, packet);
        }
        finally
        {
            packed.Dispose();
        }

        ret = default;
    }

    /// <summary>
    /// <para>Emitted when this MultiplayerAPI's <see cref="MultiplayerApi.MultiplayerPeer"/> receives a <c>packet</c> with custom data (see <see cref="SendBytes(byte[], int, MultiplayerPeer.TransferModeEnum, int)"/>). ID is the peer ID of the peer that sent the packet.</para>
    /// <para>Unlike <see cref="PeerPacket"/>, this event delivers the payload as a <see cref="ReadOnlySpan{T}"/> over the native buffer and does not allocate a managed <c>byte[]</c>. The span is only valid for the duration of the handler.</para>
    /// </summary>
    public unsafe event PeerPacketNonAllocEventHandler PeerPacketNonAlloc
    {
        add => Connect(SignalName.PeerPacket, Callable.CreateWithUnsafeTrampoline(value, &PeerPacketNonAllocTrampoline));
        remove => Disconnect(SignalName.PeerPacket, Callable.CreateWithUnsafeTrampoline(value, &PeerPacketNonAllocTrampoline));
    }
}
