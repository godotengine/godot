using System;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Runtime.InteropServices;

#nullable enable
namespace Godot
{
[Serializable]
[StructLayout(LayoutKind.Sequential)]
public struct Projection : IEquatable<Projection>
{
    public enum Planes
    {
        Near,
        Far,
        Left,
        Top,
        Right,
        Bottom,
    }

    public Vector4 X;
    public Vector4 Y;
    public Vector4 Z;
    public Vector4 W;

    public Vector4 this[int column]
    {
        readonly get
        {
            return column switch
            {
                0 => X,
                1 => Y,
                2 => Z,
                3 => W,
                _ => throw new ArgumentOutOfRangeException(nameof(column))
            };
        }
        set
        {
            switch (column)
            {
                case 0: X = value; return;
                case 1: Y = value; return;
                case 2: Z = value; return;
                case 3: W = value; return;
                default: throw new ArgumentOutOfRangeException(nameof(column));
            }
        }
    }

    public float this[int column, int row]
    {
        readonly get
        {
            return column switch
            {
                0 => GetVector4Component(X, row),
                1 => GetVector4Component(Y, row),
                2 => GetVector4Component(Z, row),
                3 => GetVector4Component(W, row),
                _ => throw new ArgumentOutOfRangeException(nameof(column))
            };
        }
        set
        {
            switch (column)
            {
                case 0: SetVector4Component(ref X, row, value); return;
                case 1: SetVector4Component(ref Y, row, value); return;
                case 2: SetVector4Component(ref Z, row, value); return;
                case 3: SetVector4Component(ref W, row, value); return;
                default: throw new ArgumentOutOfRangeException(nameof(column));
            }
        }
    }

    public static Projection CreateDepthCorrection(bool flipY)
    {
        return new Projection(
            new Vector4(1f, 0f, 0f, 0f),
            new Vector4(0f, flipY ? -1f : 1f, 0f, 0f),
            new Vector4(0f, 0f, 0.5f, 0f),
            new Vector4(0f, 0f, 0.5f, 1f)
        );
    }

    public static Projection CreateFitAabb(AABB aabb)
    {
        Vector3 min = aabb.Position;
        Vector3 max = Add(aabb.Position, aabb.Size);

        float dx = max.X - min.X;
        float dy = max.Y - min.Y;
        float dz = max.Z - min.Z;

        return new Projection(
            new Vector4(2f / dx, 0f, 0f, 0f),
            new Vector4(0f, 2f / dy, 0f, 0f),
            new Vector4(0f, 0f, 2f / dz, 0f),
            new Vector4(
                -(max.X + min.X) / dx,
                -(max.Y + min.Y) / dy,
                -(max.Z + min.Z) / dz,
                1f
            )
        );
    }

    public static Projection CreateForHmd(int eye, float aspect, float intraocularDist, float displayWidth, float displayToLens, float oversample, float zNear, float zFar)
    {
        float f1 = (intraocularDist * 0.5f) / displayToLens;
        float f2 = ((displayWidth - intraocularDist) * 0.5f) / displayToLens;
        float f3 = (displayWidth * 0.25f) / displayToLens;

        float add = ((f1 + f2) * (oversample - 1f)) / 2f;
        f1 += add;
        f2 += add;
        f3 *= oversample;
        f3 /= aspect;

        return eye switch
        {
            1 => CreateFrustum(-f2 * zNear, f1 * zNear, -f3 * zNear, f3 * zNear, zNear, zFar),
            2 => CreateFrustum(-f1 * zNear, f2 * zNear, -f3 * zNear, f3 * zNear, zNear, zFar),
            _ => Zero
        };
    }

    public static Projection CreateFrustum(float left, float right, float bottom, float top, float depthNear, float depthFar)
    {
        if (right <= left)
            throw new ArgumentException("right is less or equal to left.");
        if (top <= bottom)
            throw new ArgumentException("top is less or equal to bottom.");
        if (depthFar <= depthNear)
            throw new ArgumentException("far is less or equal to near.");

        float x = 2f * depthNear / (right - left);
        float y = 2f * depthNear / (top - bottom);

        float a = (right + left) / (right - left);
        float b = (top + bottom) / (top - bottom);
        float c = -(depthFar + depthNear) / (depthFar - depthNear);
        float d = -2f * depthFar * depthNear / (depthFar - depthNear);

        return new Projection(
            new Vector4(x, 0f, 0f, 0f),
            new Vector4(0f, y, 0f, 0f),
            new Vector4(a, b, c, -1f),
            new Vector4(0f, 0f, d, 0f)
        );
    }

    public static Projection CreateFrustumAspect(float size, float aspect, Vector2 offset, float depthNear, float depthFar, bool flipFov)
    {
        if (!flipFov)
        {
            size *= aspect;
        }

        return CreateFrustum(
            -size / 2f + offset.X,
            +size / 2f + offset.X,
            -size / aspect / 2f + offset.Y,
            +size / aspect / 2f + offset.Y,
            depthNear,
            depthFar
        );
    }

    public static Projection CreateLightAtlasRect(Rect2 rect)
    {
        return new Projection(
            new Vector4(rect.Size.X, 0f, 0f, 0f),
            new Vector4(0f, rect.Size.Y, 0f, 0f),
            new Vector4(0f, 0f, 1f, 0f),
            new Vector4(rect.Position.X, rect.Position.Y, 0f, 1f)
        );
    }

    public static Projection CreateOrthogonal(float left, float right, float bottom, float top, float zNear, float zFar)
    {
        Projection proj = Identity;
        proj.X.X = 2f / (right - left);
        proj.W.X = -((right + left) / (right - left));
        proj.Y.Y = 2f / (top - bottom);
        proj.W.Y = -((top + bottom) / (top - bottom));
        proj.Z.Z = -2f / (zFar - zNear);
        proj.W.Z = -((zFar + zNear) / (zFar - zNear));
        proj.W.W = 1f;
        return proj;
    }

    public static Projection CreateOrthogonalAspect(float size, float aspect, float zNear, float zFar, bool flipFov)
    {
        if (!flipFov)
        {
            size *= aspect;
        }

        return CreateOrthogonal(
            -size / 2f,
            +size / 2f,
            -size / aspect / 2f,
            +size / aspect / 2f,
            zNear,
            zFar
        );
    }

    public static Projection CreatePerspective(float fovyDegrees, float aspect, float zNear, float zFar, bool flipFov)
    {
        if (flipFov)
        {
            fovyDegrees = GetFovy(fovyDegrees, 1f / aspect);
        }

        float radians = DegToRad(fovyDegrees / 2f);
        float deltaZ = zFar - zNear;
        float sin = MathF.Sin(radians);
        float cos = MathF.Cos(radians);

        if (deltaZ == 0f || sin == 0f || aspect == 0f)
            return Zero;

        float cotangent = cos / sin;

        Projection proj = Identity;
        proj.X.X = cotangent / aspect;
        proj.Y.Y = cotangent;
        proj.Z.Z = -(zFar + zNear) / deltaZ;
        proj.Z.W = -1f;
        proj.W.Z = -2f * zNear * zFar / deltaZ;
        proj.W.W = 0f;
        return proj;
    }

    public static Projection CreatePerspectiveHmd(float fovyDegrees, float aspect, float zNear, float zFar, bool flipFov, int eye, float intraocularDist, float convergenceDist)
    {
        if (flipFov)
        {
            fovyDegrees = GetFovy(fovyDegrees, 1f / aspect);
        }

        float ymax = zNear * MathF.Tan(DegToRad(fovyDegrees / 2f));
        float xmax = ymax * aspect;
        float frustumshift = (intraocularDist / 2f) * zNear / convergenceDist;

        float left;
        float right;
        float modeltranslation;

        switch (eye)
        {
            case 1:
                left = -xmax + frustumshift;
                right = xmax + frustumshift;
                modeltranslation = intraocularDist / 2f;
                break;
            case 2:
                left = -xmax - frustumshift;
                right = xmax - frustumshift;
                modeltranslation = -intraocularDist / 2f;
                break;
            default:
                left = -xmax;
                right = xmax;
                modeltranslation = 0f;
                break;
        }

        Projection proj = CreateFrustum(left, right, -ymax, ymax, zNear, zFar);
        Projection cm = Identity;
        cm.W.X = modeltranslation;
        return proj * cm;
    }

    public readonly float Determinant()
    {
        return X.W * Y.Z * Z.Y * W.X - X.Z * Y.W * Z.Y * W.X -
               X.W * Y.Y * Z.Z * W.X + X.Y * Y.W * Z.Z * W.X +
               X.Z * Y.Y * Z.W * W.X - X.Y * Y.Z * Z.W * W.X -
               X.W * Y.Z * Z.X * W.Y + X.Z * Y.W * Z.X * W.Y +
               X.W * Y.X * Z.Z * W.Y - X.X * Y.W * Z.Z * W.Y -
               X.Z * Y.X * Z.W * W.Y + X.X * Y.Z * Z.W * W.Y +
               X.W * Y.Y * Z.X * W.Z - X.Y * Y.W * Z.X * W.Z -
               X.W * Y.X * Z.Y * W.Z + X.X * Y.W * Z.Y * W.Z +
               X.Y * Y.X * Z.W * W.Z - X.X * Y.Y * Z.W * W.Z -
               X.Z * Y.Y * Z.X * W.W + X.Y * Y.Z * Z.X * W.W +
               X.Z * Y.X * Z.Y * W.W - X.X * Y.Z * Z.Y * W.W -
               X.Y * Y.X * Z.Z * W.W + X.X * Y.Y * Z.Z * W.W;
    }

    public readonly float GetAspect()
    {
        Vector2 vpHe = GetViewportHalfExtents();
        return vpHe.X / vpHe.Y;
    }

    public readonly float GetFov()
    {
        Plane rightPlane = new Plane(
            X.W - X.X,
            Y.W - Y.X,
            Z.W - Z.X,
            -W.W + W.X
        ).Normalized();

        if (Z.X == 0f && Z.Y == 0f)
        {
            return RadToDeg(MathF.Acos(MathF.Abs(rightPlane.Normal.X))) * 2f;
        }
        else
        {
            Plane leftPlane = new Plane(
                X.W + X.X,
                Y.W + Y.X,
                Z.W + Z.X,
                W.W + W.X
            ).Normalized();

            return RadToDeg(MathF.Acos(MathF.Abs(leftPlane.Normal.X))) +
                   RadToDeg(MathF.Acos(MathF.Abs(rightPlane.Normal.X)));
        }
    }

    public static float GetFovy(float fovx, float aspect)
    {
        return RadToDeg(MathF.Atan(aspect * MathF.Tan(DegToRad(fovx) * 0.5f)) * 2f);
    }

    public readonly float GetLodMultiplier()
    {
        if (IsOrthogonal())
        {
            return GetViewportHalfExtents().X;
        }
        else
        {
            float zn = GetZNear();
            float width = GetViewportHalfExtents().X * 2f;
            return 1f / (zn / width);
        }
    }

    public readonly int GetPixelsPerMeter(int forPixelWidth)
    {
        Vector3 result = this * new Vector3(1f, 0f, -1f);
        return (int)((result.X * 0.5f + 0.5f) * forPixelWidth);
    }

    public readonly Plane GetProjectionPlane(Planes plane)
    {
        Plane newPlane = plane switch
        {
            Planes.Near => new Plane(X.W + X.Z, Y.W + Y.Z, Z.W + Z.Z, W.W + W.Z),
            Planes.Far => new Plane(X.W - X.Z, Y.W - Y.Z, Z.W - Z.Z, W.W - W.Z),
            Planes.Left => new Plane(X.W + X.X, Y.W + Y.X, Z.W + Z.X, W.W + W.X),
            Planes.Top => new Plane(X.W - X.Y, Y.W - Y.Y, Z.W - Z.Y, W.W - W.Y),
            Planes.Right => new Plane(X.W - X.X, Y.W - Y.X, Z.W - Z.X, W.W - W.X),
            Planes.Bottom => new Plane(X.W + X.Y, Y.W + Y.Y, Z.W + Z.Y, W.W + W.Y),
            _ => new Plane()
        };

        newPlane = -newPlane;
        return newPlane.Normalized();
    }

    public readonly Vector2 GetFarPlaneHalfExtents()
    {
        Vector3? res = GetProjectionPlane(Planes.Far).Intersect3(
            GetProjectionPlane(Planes.Right),
            GetProjectionPlane(Planes.Top)
        );

        return res is null ? default : new Vector2(res.Value.X, res.Value.Y);
    }

    public readonly Vector2 GetViewportHalfExtents()
    {
        Vector3? res = GetProjectionPlane(Planes.Near).Intersect3(
            GetProjectionPlane(Planes.Right),
            GetProjectionPlane(Planes.Top)
        );

        return res is null ? default : new Vector2(res.Value.X, res.Value.Y);
    }

    public readonly float GetZFar()
    {
        return GetProjectionPlane(Planes.Far).D;
    }

    public readonly float GetZNear()
    {
        return -GetProjectionPlane(Planes.Near).D;
    }

    public readonly Projection FlippedY()
    {
        Projection proj = this;
        proj.Y = Negate(proj.Y);
        return proj;
    }

    public readonly Projection PerspectiveZNearAdjusted(float newZNear)
    {
        Projection proj = this;
        float zFar = GetZFar();
        float zNear = newZNear;
        float deltaZ = zFar - zNear;
        proj.Z.Z = -(zFar + zNear) / deltaZ;
        proj.W.Z = -2f * zNear * zFar / deltaZ;
        return proj;
    }

    public readonly Projection JitterOffseted(Vector2 offset)
    {
        Projection proj = this;
        proj.W.X += offset.X;
        proj.W.Y += offset.Y;
        return proj;
    }

    public readonly Projection Inverse()
    {
        Projection proj = this;
        int i, j, k;
        Span<int> pvt_i = stackalloc int[4];
        Span<int> pvt_j = stackalloc int[4];
        float pvt_val;
        float hold;
        float determinant = 1f;

        for (k = 0; k < 4; k++)
        {
            pvt_val = proj[k][k];
            pvt_i[k] = k;
            pvt_j[k] = k;

            for (i = k; i < 4; i++)
            {
                for (j = k; j < 4; j++)
                {
                    if (MathF.Abs(proj[i][j]) > MathF.Abs(pvt_val))
                    {
                        pvt_i[k] = i;
                        pvt_j[k] = j;
                        pvt_val = proj[i][j];
                    }
                }
            }

            determinant *= pvt_val;
            if (IsZeroApprox(determinant))
            {
                return Zero;
            }

            i = pvt_i[k];
            if (i != k)
            {
                for (j = 0; j < 4; j++)
                {
                    hold = -proj[k][j];
                    proj[k, j] = proj[i][j];
                    proj[i, j] = hold;
                }
            }

            j = pvt_j[k];
            if (j != k)
            {
                for (i = 0; i < 4; i++)
                {
                    hold = -proj[i][k];
                    proj[i, k] = proj[i][j];
                    proj[i, j] = hold;
                }
            }

            for (i = 0; i < 4; i++)
            {
                if (i != k)
                {
                    proj[i, k] /= -pvt_val;
                }
            }

            for (i = 0; i < 4; i++)
            {
                hold = proj[i][k];
                for (j = 0; j < 4; j++)
                {
                    if (i != k && j != k)
                    {
                        proj[i, j] += hold * proj[k][j];
                    }
                }
            }

            for (j = 0; j < 4; j++)
            {
                if (j != k)
                {
                    proj[k, j] /= pvt_val;
                }
            }

            proj[k, k] = 1f / pvt_val;
        }

        for (k = 2; k >= 0; k--)
        {
            i = pvt_j[k];
            if (i != k)
            {
                for (j = 0; j < 4; j++)
                {
                    hold = proj[k][j];
                    proj[k, j] = -proj[i][j];
                    proj[i, j] = hold;
                }
            }

            j = pvt_i[k];
            if (j != k)
            {
                for (i = 0; i < 4; i++)
                {
                    hold = proj[i][k];
                    proj[i, k] = -proj[i][j];
                    proj[i, j] = hold;
                }
            }
        }

        return proj;
    }

    public readonly bool IsOrthogonal()
    {
        return W.W == 1f;
    }

    private static readonly Projection _zero = new Projection(
        new Vector4(0f, 0f, 0f, 0f),
        new Vector4(0f, 0f, 0f, 0f),
        new Vector4(0f, 0f, 0f, 0f),
        new Vector4(0f, 0f, 0f, 0f)
    );

    private static readonly Projection _identity = new Projection(
        new Vector4(1f, 0f, 0f, 0f),
        new Vector4(0f, 1f, 0f, 0f),
        new Vector4(0f, 0f, 1f, 0f),
        new Vector4(0f, 0f, 0f, 1f)
    );

    public static Projection Zero => _zero;
    public static Projection Identity => _identity;

    public Projection(Vector4 x, Vector4 y, Vector4 z, Vector4 w)
    {
        X = x;
        Y = y;
        Z = z;
        W = w;
    }

    public Projection(
        float xx, float xy, float xz, float xw,
        float yx, float yy, float yz, float yw,
        float zx, float zy, float zz, float zw,
        float wx, float wy, float wz, float ww)
    {
        X = new Vector4(xx, xy, xz, xw);
        Y = new Vector4(yx, yy, yz, yw);
        Z = new Vector4(zx, zy, zz, zw);
        W = new Vector4(wx, wy, wz, ww);
    }

    public Projection(Transform3D transform)
    {
        X = new Vector4(transform.Basis.Row0.X, transform.Basis.Row1.X, transform.Basis.Row2.X, 0f);
        Y = new Vector4(transform.Basis.Row0.Y, transform.Basis.Row1.Y, transform.Basis.Row2.Y, 0f);
        Z = new Vector4(transform.Basis.Row0.Z, transform.Basis.Row1.Z, transform.Basis.Row2.Z, 0f);
        W = new Vector4(transform.Origin.X, transform.Origin.Y, transform.Origin.Z, 1f);
    }

    public static Projection operator *(Projection left, Projection right)
    {
        return new Projection(
            left * right.X,
            left * right.Y,
            left * right.Z,
            left * right.W
        );
    }

    public static Vector4 operator *(Projection proj, Vector4 vector)
    {
        return new Vector4(
            proj.X.X * vector.X + proj.Y.X * vector.Y + proj.Z.X * vector.Z + proj.W.X * vector.W,
            proj.X.Y * vector.X + proj.Y.Y * vector.Y + proj.Z.Y * vector.Z + proj.W.Y * vector.W,
            proj.X.Z * vector.X + proj.Y.Z * vector.Y + proj.Z.Z * vector.Z + proj.W.Z * vector.W,
            proj.X.W * vector.X + proj.Y.W * vector.Y + proj.Z.W * vector.Z + proj.W.W * vector.W
        );
    }

    public static Vector4 operator *(Vector4 vector, Projection proj)
    {
        return new Vector4(
            proj.X.X * vector.X + proj.X.Y * vector.Y + proj.X.Z * vector.Z + proj.X.W * vector.W,
            proj.Y.X * vector.X + proj.Y.Y * vector.Y + proj.Y.Z * vector.Z + proj.Y.W * vector.W,
            proj.Z.X * vector.X + proj.Z.Y * vector.Y + proj.Z.Z * vector.Z + proj.Z.W * vector.W,
            proj.W.X * vector.X + proj.W.Y * vector.Y + proj.W.Z * vector.Z + proj.W.W * vector.W
        );
    }

    public static Vector3 operator *(Projection proj, Vector3 vector)
    {
        Vector3 ret = new Vector3(
            proj.X.X * vector.X + proj.Y.X * vector.Y + proj.Z.X * vector.Z + proj.W.X,
            proj.X.Y * vector.X + proj.Y.Y * vector.Y + proj.Z.Y * vector.Z + proj.W.Y,
            proj.X.Z * vector.X + proj.Y.Z * vector.Y + proj.Z.Z * vector.Z + proj.W.Z
        );

        float w = proj.X.W * vector.X + proj.Y.W * vector.Y + proj.Z.W * vector.Z + proj.W.W;
        return Div(ret, w);
    }

    public static bool operator ==(Projection left, Projection right) => left.Equals(right);
    public static bool operator !=(Projection left, Projection right) => !left.Equals(right);

    public static explicit operator Transform3D(Projection proj)
    {
        return new Transform3D(
            new Basis(
                new Vector3(proj.X.X, proj.X.Y, proj.X.Z),
                new Vector3(proj.Y.X, proj.Y.Y, proj.Y.Z),
                new Vector3(proj.Z.X, proj.Z.Y, proj.Z.Z)
            ),
            new Vector3(proj.W.X, proj.W.Y, proj.W.Z)
        );
    }

    public override readonly bool Equals([NotNullWhen(true)] object? obj)
    {
        return obj is Projection other && Equals(other);
    }

    public readonly bool Equals(Projection other)
    {
        return X.X == other.X.X && X.Y == other.X.Y && X.Z == other.X.Z && X.W == other.X.W &&
               Y.X == other.Y.X && Y.Y == other.Y.Y && Y.Z == other.Y.Z && Y.W == other.Y.W &&
               Z.X == other.Z.X && Z.Y == other.Z.Y && Z.Z == other.Z.Z && Z.W == other.Z.W &&
               W.X == other.W.X && W.Y == other.W.Y && W.Z == other.W.Z && W.W == other.W.W;
    }

    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(X);
        hash.Add(Y);
        hash.Add(Z);
        hash.Add(W);
        return hash.ToHashCode();
    }

    public override readonly string ToString() => ToString(null);

    public readonly string ToString(string? format)
    {
        string f = string.IsNullOrEmpty(format) ? "G" : format!;

        return
            $"{X.X.ToString(f, CultureInfo.InvariantCulture)}, {X.Y.ToString(f, CultureInfo.InvariantCulture)}, {X.Z.ToString(f, CultureInfo.InvariantCulture)}, {X.W.ToString(f, CultureInfo.InvariantCulture)}\n" +
            $"{Y.X.ToString(f, CultureInfo.InvariantCulture)}, {Y.Y.ToString(f, CultureInfo.InvariantCulture)}, {Y.Z.ToString(f, CultureInfo.InvariantCulture)}, {Y.W.ToString(f, CultureInfo.InvariantCulture)}\n" +
            $"{Z.X.ToString(f, CultureInfo.InvariantCulture)}, {Z.Y.ToString(f, CultureInfo.InvariantCulture)}, {Z.Z.ToString(f, CultureInfo.InvariantCulture)}, {Z.W.ToString(f, CultureInfo.InvariantCulture)}\n" +
            $"{W.X.ToString(f, CultureInfo.InvariantCulture)}, {W.Y.ToString(f, CultureInfo.InvariantCulture)}, {W.Z.ToString(f, CultureInfo.InvariantCulture)}, {W.W.ToString(f, CultureInfo.InvariantCulture)}";
    }

    private static float DegToRad(float degrees) => degrees * (MathF.PI / 180f);
    private static float RadToDeg(float radians) => radians * (180f / MathF.PI);

    private static bool IsZeroApprox(float v) => MathF.Abs(v) <= 0.00001f;

    private static float GetVector4Component(Vector4 v, int row)
    {
        return row switch
        {
            0 => v.X,
            1 => v.Y,
            2 => v.Z,
            3 => v.W,
            _ => throw new ArgumentOutOfRangeException(nameof(row))
        };
    }

    private static void SetVector4Component(ref Vector4 v, int row, float value)
    {
        switch (row)
        {
            case 0: v.X = value; return;
            case 1: v.Y = value; return;
            case 2: v.Z = value; return;
            case 3: v.W = value; return;
            default: throw new ArgumentOutOfRangeException(nameof(row));
        }
    }

    private static Vector4 Negate(Vector4 v)
    {
        return new Vector4(-v.X, -v.Y, -v.Z, -v.W);
    }

    private static Vector3 Add(Vector3 a, Vector3 b)
    {
        return new Vector3(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
    }

    private static Vector3 Div(Vector3 v, float s)
    {
        return new Vector3(v.X / s, v.Y / s, v.Z / s);
    }
}

}