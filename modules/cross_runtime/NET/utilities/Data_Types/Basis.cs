using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;
using System.ComponentModel;

#nullable enable
namespace Godot
{


/// <summary>
/// 3×3 matrix used for 3D rotation and scale.
/// </summary>
[Serializable]
[StructLayout(LayoutKind.Sequential)]
public struct Basis : IEquatable<Basis>
{
    // X / Y / Z are public accessors, like Godot's official API.
    public Vector3 X
    {
        readonly get => Column0;
        set => Column0 = value;
    }

    public Vector3 Y
    {
        readonly get => Column1;
        set => Column1 = value;
    }

    public Vector3 Z
    {
        readonly get => Column2;
        set => Column2 = value;
    }

    public Vector3 Row0;
    public Vector3 Row1;
    public Vector3 Row2;

    public Vector3 Column0
    {
        readonly get => new Vector3(Row0.X, Row1.X, Row2.X);
        set
        {
            Row0.X = value.X;
            Row1.X = value.Y;
            Row2.X = value.Z;
        }
    }

    public Vector3 Column1
    {
        readonly get => new Vector3(Row0.Y, Row1.Y, Row2.Y);
        set
        {
            Row0.Y = value.X;
            Row1.Y = value.Y;
            Row2.Y = value.Z;
        }
    }

    public Vector3 Column2
    {
        readonly get => new Vector3(Row0.Z, Row1.Z, Row2.Z);
        set
        {
            Row0.Z = value.X;
            Row1.Z = value.Y;
            Row2.Z = value.Z;
        }
    }

    public readonly Vector3 Scale
    {
        get
        {
            float detSign = Sign(Determinant());
            return Mul(new Vector3(
                Length(Column0),
                Length(Column1),
                Length(Column2)
            ), detSign);
        }
    }

    public Vector3 this[int column]
    {
        readonly get
        {
            switch (column)
            {
                case 0: return Column0;
                case 1: return Column1;
                case 2: return Column2;
                default: throw new ArgumentOutOfRangeException(nameof(column));
            }
        }
        set
        {
            switch (column)
            {
                case 0: Column0 = value; return;
                case 1: Column1 = value; return;
                case 2: Column2 = value; return;
                default: throw new ArgumentOutOfRangeException(nameof(column));
            }
        }
    }
    

    public float this[int column, int row]
    {
        readonly get => GetComponent(this[column], row);
        set
        {
            Vector3 c = this[column];
            SetComponent(ref c, row, value);
            this[column] = c;
        }
    }

    internal void SetQuaternionScale(Quaternion quaternion, Vector3 scale)
    {
        SetDiagonal(scale);
        Rotate(quaternion);
    }

    private void Rotate(Quaternion quaternion)
    {
        this = new Basis(quaternion) * this;
    }

    private void SetDiagonal(Vector3 diagonal)
    {
        Row0 = new Vector3(diagonal.X, 0f, 0f);
        Row1 = new Vector3(0f, diagonal.Y, 0f);
        Row2 = new Vector3(0f, 0f, diagonal.Z);
    }

    public readonly float Determinant()
    {
        float cofac00 = Row1.Y * Row2.Z - Row1.Z * Row2.Y;
        float cofac10 = Row1.Z * Row2.X - Row1.X * Row2.Z;
        float cofac20 = Row1.X * Row2.Y - Row1.Y * Row2.X;

        return Row0.X * cofac00 + Row0.Y * cofac10 + Row0.Z * cofac20;
    }

    public readonly Vector3 GetEuler(EulerOrder order = EulerOrder.EULER_ORDER_YXZ)
    {
        switch (order)
        {
            case EulerOrder.EULER_ORDER_XYZ:
            {
                Vector3 euler;
                float sy = Row0.Z;
                if (sy < (1.0f - Epsilon))
                {
                    if (sy > -(1.0f - Epsilon))
                    {
                        if (IsPureYRotation_XYZ())
                        {
                            euler.X = 0f;
                            euler.Y = Atan2(Row0.Z, Row0.X);
                            euler.Z = 0f;
                        }
                        else
                        {
                            euler.X = Atan2(-Row1.Z, Row2.Z);
                            euler.Y = Asin(sy);
                            euler.Z = Atan2(-Row0.Y, Row0.X);
                        }
                    }
                    else
                    {
                        euler.X = Atan2(Row2.Y, Row1.Y);
                        euler.Y = -Tau / 4.0f;
                        euler.Z = 0.0f;
                    }
                }
                else
                {
                    euler.X = Atan2(Row2.Y, Row1.Y);
                    euler.Y = Tau / 4.0f;
                    euler.Z = 0.0f;
                }
                return euler;
            }

            case EulerOrder.EULER_ORDER_XZY:
            {
                Vector3 euler;
                float sz = Row0.Y;
                if (sz < (1.0f - Epsilon))
                {
                    if (sz > -(1.0f - Epsilon))
                    {
                        euler.X = Atan2(Row2.Y, Row1.Y);
                        euler.Y = Atan2(Row0.Z, Row0.X);
                        euler.Z = Asin(-sz);
                    }
                    else
                    {
                        euler.X = -Atan2(Row1.Z, Row2.Z);
                        euler.Y = 0.0f;
                        euler.Z = Tau / 4.0f;
                    }
                }
                else
                {
                    euler.X = -Atan2(Row1.Z, Row2.Z);
                    euler.Y = 0.0f;
                    euler.Z = -Tau / 4.0f;
                }
                return euler;
            }

            case EulerOrder.EULER_ORDER_YXZ:
            {
                Vector3 euler;
                float m12 = Row1.Z;
                if (m12 < (1.0f - Epsilon))
                {
                    if (m12 > -(1.0f - Epsilon))
                    {
                        if (IsPureXRotation_YXZ())
                        {
                            euler.X = Atan2(-m12, Row1.Y);
                            euler.Y = 0f;
                            euler.Z = 0f;
                        }
                        else
                        {
                            euler.X = Asin(-m12);
                            euler.Y = Atan2(Row0.Z, Row2.Z);
                            euler.Z = Atan2(Row1.X, Row1.Y);
                        }
                    }
                    else
                    {
                        euler.X = Tau / 4.0f;
                        euler.Y = Atan2(Row0.Y, Row0.X);
                        euler.Z = 0f;
                    }
                }
                else
                {
                    euler.X = -Tau / 4.0f;
                    euler.Y = -Atan2(Row0.Y, Row0.X);
                    euler.Z = 0f;
                }
                return euler;
            }

            case EulerOrder.EULER_ORDER_YZX:
            {
                Vector3 euler;
                float sz = Row1.X;
                if (sz < (1.0f - Epsilon))
                {
                    if (sz > -(1.0f - Epsilon))
                    {
                        euler.X = Atan2(-Row1.Z, Row1.Y);
                        euler.Y = Atan2(-Row2.X, Row0.X);
                        euler.Z = Asin(sz);
                    }
                    else
                    {
                        euler.X = Atan2(Row2.Y, Row2.Z);
                        euler.Y = 0.0f;
                        euler.Z = -Tau / 4.0f;
                    }
                }
                else
                {
                    euler.X = Atan2(Row2.Y, Row2.Z);
                    euler.Y = 0.0f;
                    euler.Z = Tau / 4.0f;
                }
                return euler;
            }

            case EulerOrder.EULER_ORDER_ZXY:
            {
                Vector3 euler;
                float sx = Row2.Y;
                if (sx < (1.0f - Epsilon))
                {
                    if (sx > -(1.0f - Epsilon))
                    {
                        euler.X = Asin(sx);
                        euler.Y = Atan2(-Row2.X, Row2.Z);
                        euler.Z = Atan2(-Row0.Y, Row1.Y);
                    }
                    else
                    {
                        euler.X = -Tau / 4.0f;
                        euler.Y = Atan2(Row0.Z, Row0.X);
                        euler.Z = 0f;
                    }
                }
                else
                {
                    euler.X = Tau / 4.0f;
                    euler.Y = Atan2(Row0.Z, Row0.X);
                    euler.Z = 0f;
                }
                return euler;
            }

            case EulerOrder.EULER_ORDER_ZYX:
            {
                Vector3 euler;
                float sy = Row2.X;
                if (sy < (1.0f - Epsilon))
                {
                    if (sy > -(1.0f - Epsilon))
                    {
                        euler.X = Atan2(Row2.Y, Row2.Z);
                        euler.Y = Asin(-sy);
                        euler.Z = Atan2(Row1.X, Row0.X);
                    }
                    else
                    {
                        euler.X = 0f;
                        euler.Y = Tau / 4.0f;
                        euler.Z = -Atan2(Row0.Y, Row1.Y);
                    }
                }
                else
                {
                    euler.X = 0f;
                    euler.Y = -Tau / 4.0f;
                    euler.Z = -Atan2(Row0.Y, Row1.Y);
                }
                return euler;
            }

            default:
                throw new ArgumentOutOfRangeException(nameof(order));
        }
    }

    internal readonly Quaternion GetQuaternion()
    {
        float trace = Row0.X + Row1.Y + Row2.Z;

        if (trace > 0.0f)
        {
            float s = Sqrt(trace + 1.0f) * 2f;
            float invS = 1f / s;
            return new Quaternion(
                (Row2.Y - Row1.Z) * invS,
                (Row0.Z - Row2.X) * invS,
                (Row1.X - Row0.Y) * invS,
                s * 0.25f
            );
        }

        if (Row0.X > Row1.Y && Row0.X > Row2.Z)
        {
            float s = Sqrt(Row0.X - Row1.Y - Row2.Z + 1.0f) * 2f;
            float invS = 1f / s;
            return new Quaternion(
                s * 0.25f,
                (Row0.Y + Row1.X) * invS,
                (Row0.Z + Row2.X) * invS,
                (Row2.Y - Row1.Z) * invS
            );
        }

        if (Row1.Y > Row2.Z)
        {
            float s = Sqrt(-Row0.X + Row1.Y - Row2.Z + 1.0f) * 2f;
            float invS = 1f / s;
            return new Quaternion(
                (Row0.Y + Row1.X) * invS,
                s * 0.25f,
                (Row1.Z + Row2.Y) * invS,
                (Row0.Z - Row2.X) * invS
            );
        }
        else
        {
            float s = Sqrt(-Row0.X - Row1.Y + Row2.Z + 1.0f) * 2f;
            float invS = 1f / s;
            return new Quaternion(
                (Row0.Z + Row2.X) * invS,
                (Row1.Z + Row2.Y) * invS,
                s * 0.25f,
                (Row1.X - Row0.Y) * invS
            );
        }
    }

    public readonly Quaternion GetRotationQuaternion()
    {
        Basis orthonormalizedBasis = Orthonormalized();
        float det = orthonormalizedBasis.Determinant();
        if (det < 0f)
        {
            orthonormalizedBasis = orthonormalizedBasis.Scaled(new Vector3(-1f, -1f, -1f));
        }

        return orthonormalizedBasis.GetQuaternion();
    }

    public readonly Basis Inverse()
    {
        float cofac00 = Row1.Y * Row2.Z - Row1.Z * Row2.Y;
        float cofac10 = Row1.Z * Row2.X - Row1.X * Row2.Z;
        float cofac20 = Row1.X * Row2.Y - Row1.Y * Row2.X;

        float det = Row0.X * cofac00 + Row0.Y * cofac10 + Row0.Z * cofac20;

        if (det == 0f)
            throw new InvalidOperationException("Matrix determinant is zero and cannot be inverted.");

        float detInv = 1.0f / det;

        float cofac01 = Row0.Z * Row2.Y - Row0.Y * Row2.Z;
        float cofac02 = Row0.Y * Row1.Z - Row0.Z * Row1.Y;
        float cofac11 = Row0.X * Row2.Z - Row0.Z * Row2.X;
        float cofac12 = Row0.Z * Row1.X - Row0.X * Row1.Z;
        float cofac21 = Row0.Y * Row2.X - Row0.X * Row2.Y;
        float cofac22 = Row0.X * Row1.Y - Row0.Y * Row1.X;

        return new Basis(
            cofac00 * detInv, cofac01 * detInv, cofac02 * detInv,
            cofac10 * detInv, cofac11 * detInv, cofac12 * detInv,
            cofac20 * detInv, cofac21 * detInv, cofac22 * detInv
        );
    }

    public readonly bool IsFinite()
    {
        return IsFinite(Row0.X) && IsFinite(Row0.Y) && IsFinite(Row0.Z) &&
               IsFinite(Row1.X) && IsFinite(Row1.Y) && IsFinite(Row1.Z) &&
               IsFinite(Row2.X) && IsFinite(Row2.Y) && IsFinite(Row2.Z);
    }

    internal readonly Basis Lerp(Basis to, float weight)
    {
        Basis b = this;
        b.Row0 = Lerp(Row0, to.Row0, weight);
        b.Row1 = Lerp(Row1, to.Row1, weight);
        b.Row2 = Lerp(Row2, to.Row2, weight);
        return b;
    }
    private static real_t Neg(real_t value) => -value;
    public static Basis LookingAt(Vector3 target, Vector3? up = null, bool useModelFront = false)
    {
        up ??= new Vector3(0f, 1f, 0f);
    
    #if DEBUG
        if (IsZeroApprox(target))
            throw new ArgumentException("The vector can't be zero.", nameof(target));
        if (IsZeroApprox(up.Value))
            throw new ArgumentException("The vector can't be zero.", nameof(up));
    #endif
    
        Vector3 column2 = Normalize(target);
        if (!useModelFront)
            column2 = new Vector3(-column2.X, -column2.Y, -column2.Z);
    
        Vector3 column0 = Cross(up.Value, column2);
        if (IsZeroApprox(column0))
        {
            throw new ArgumentException(
                "Target and up vectors are colinear. This is not advised as it may cause unwanted rotation around local Z axis.");
        }
    
        column0 = Normalize(column0);
        Vector3 column1 = Cross(column2, column0);
    
        return new Basis(column0, column1, column2);
    }

    [EditorBrowsable(EditorBrowsableState.Never)]
    public static Basis LookingAt(Vector3 target, Vector3 up)
    {
        return LookingAt(target, up, false);
    }

    public readonly Basis Orthonormalized()
    {
        Vector3 column0 = this[0];
        Vector3 column1 = this[1];
        Vector3 column2 = this[2];

        column0 = Normalize(column0);
        column1 = Sub(column1, Mul(column0, Dot(column0, column1)));
        column1 = Normalize(column1);
        column2 = Sub(column2, Add(Mul(column0, Dot(column0, column2)), Mul(column1, Dot(column1, column2))));
        column2 = Normalize(column2);

        return new Basis(column0, column1, column2);
    }

    public readonly Basis Rotated(Vector3 axis, real_t angle)
    {
        return new Basis(axis, angle) * this;
    }

    public readonly Basis Scaled(Vector3 scale)
    {
        Basis b = this;
        b.Row0 = Mul(b.Row0, scale.X);
        b.Row1 = Mul(b.Row1, scale.Y);
        b.Row2 = Mul(b.Row2, scale.Z);
        return b;
    }

    public readonly Basis ScaledLocal(Vector3 scale)
    {
        Basis b = this;
        b.Row0 = Mul(b.Row0, scale);
        b.Row1 = Mul(b.Row1, scale);
        b.Row2 = Mul(b.Row2, scale);
        return b;
    }

    public readonly Basis Slerp(Basis target, float weight)
    {
        Quaternion from = GetQuaternion();
        Quaternion to = target.GetQuaternion();

        Quaternion q = SlerpQuaternion(from, to, weight);

        Basis b = new Basis(q);
        b.Row0 = Mul(b.Row0, Lerp(Length(Row0), Length(target.Row0), weight));
        b.Row1 = Mul(b.Row1, Lerp(Length(Row1), Length(target.Row1), weight));
        b.Row2 = Mul(b.Row2, Lerp(Length(Row2), Length(target.Row2), weight));

        return b;
    }

    public readonly float Tdotx(Vector3 with)
    {
        return Row0.X * with.X + Row1.X * with.Y + Row2.X * with.Z;
    }

    public readonly float Tdoty(Vector3 with)
    {
        return Row0.Y * with.X + Row1.Y * with.Y + Row2.Y * with.Z;
    }

    public readonly float Tdotz(Vector3 with)
    {
        return Row0.Z * with.X + Row1.Z * with.Y + Row2.Z * with.Z;
    }

    public readonly Basis Transposed()
    {
        Basis tr = this;

        tr.Row0.Y = Row1.X;
        tr.Row1.X = Row0.Y;

        tr.Row0.Z = Row2.X;
        tr.Row2.X = Row0.Z;

        tr.Row1.Z = Row2.Y;
        tr.Row2.Y = Row1.Z;

        return tr;
    }

    private static readonly Basis _identity = new Basis(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f);
    private static readonly Basis _flipX = new Basis(-1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f);
    private static readonly Basis _flipY = new Basis(1f, 0f, 0f, 0f, -1f, 0f, 0f, 0f, 1f);
    private static readonly Basis _flipZ = new Basis(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, -1f);

    public static Basis Identity => _identity;
    public static Basis FlipX => _flipX;
    public static Basis FlipY => _flipY;
    public static Basis FlipZ => _flipZ;

    public Basis(Quaternion q)
    {
        // standard quaternion-to-basis conversion
        real_t xx = q.X * q.X;
        real_t yy = q.Y * q.Y;
        real_t zz = q.Z * q.Z;
        real_t xy = q.X * q.Y;
        real_t xz = q.X * q.Z;
        real_t yz = q.Y * q.Z;
        real_t wx = q.W * q.X;
        real_t wy = q.W * q.Y;
        real_t wz = q.W * q.Z;
    
        X = new Vector3(1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy));
        Y = new Vector3(2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx));
        Z = new Vector3(2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy));
    }

    public Basis(Vector3 axis, real_t angle)
    {
        float axisLenSq = Dot(axis, axis);
        if (axisLenSq == 0f)
            throw new ArgumentException("Axis cannot be zero.", nameof(axis));

        Vector3 n = Normalize(axis);
        float x = n.X;
        float y = n.Y;
        float z = n.Z;

        float sin = Sin(angle);
        float cos = Cos(angle);
        float t = 1.0f - cos;

        Row0 = new Vector3(
            t * x * x + cos,
            t * x * y - sin * z,
            t * x * z + sin * y
        );

        Row1 = new Vector3(
            t * x * y + sin * z,
            t * y * y + cos,
            t * y * z - sin * x
        );

        Row2 = new Vector3(
            t * x * z - sin * y,
            t * y * z + sin * x,
            t * z * z + cos
        );
    }

    public Basis(Vector3 column0, Vector3 column1, Vector3 column2)
    {
        Row0 = new Vector3(column0.X, column1.X, column2.X);
        Row1 = new Vector3(column0.Y, column1.Y, column2.Y);
        Row2 = new Vector3(column0.Z, column1.Z, column2.Z);
    }

    public Basis(float xx, float yx, float zx, float xy, float yy, float zy, float xz, float yz, float zz)
    {
        Row0 = new Vector3(xx, yx, zx);
        Row1 = new Vector3(xy, yy, zy);
        Row2 = new Vector3(xz, yz, zz);
    }

    public static Basis FromEuler(Vector3 euler, EulerOrder order = EulerOrder.EULER_ORDER_YXZ)
    {
        float sin;
        float cos;

        SinCos(euler.X, out sin, out cos);
        Basis xmat = new Basis(
            new Vector3(1f, 0f, 0f),
            new Vector3(0f, cos, sin),
            new Vector3(0f, -sin, cos)
        );

        SinCos(euler.Y, out sin, out cos);
        Basis ymat = new Basis(
            new Vector3(cos, 0f, -sin),
            new Vector3(0f, 1f, 0f),
            new Vector3(sin, 0f, cos)
        );

        SinCos(euler.Z, out sin, out cos);
        Basis zmat = new Basis(
            new Vector3(cos, sin, 0f),
            new Vector3(-sin, cos, 0f),
            new Vector3(0f, 0f, 1f)
        );

        return order switch
        {
            EulerOrder.EULER_ORDER_XYZ => xmat * ymat * zmat,
            EulerOrder.EULER_ORDER_XZY => xmat * zmat * ymat,
            EulerOrder.EULER_ORDER_YXZ => ymat * xmat * zmat,
            EulerOrder.EULER_ORDER_YZX => ymat * zmat * xmat,
            EulerOrder.EULER_ORDER_ZXY => zmat * xmat * ymat,
            EulerOrder.EULER_ORDER_ZYX => zmat * ymat * xmat,
            _ => throw new ArgumentOutOfRangeException(nameof(order))
        };
    }

    public static Basis FromScale(Vector3 scale)
    {
        return new Basis(
            scale.X, 0f, 0f,
            0f, scale.Y, 0f,
            0f, 0f, scale.Z
        );
    }

    public static Basis operator *(Basis left, Basis right)
    {
        return new Basis
        (
            right.Tdotx(left.Row0), right.Tdoty(left.Row0), right.Tdotz(left.Row0),
            right.Tdotx(left.Row1), right.Tdoty(left.Row1), right.Tdotz(left.Row1),
            right.Tdotx(left.Row2), right.Tdoty(left.Row2), right.Tdotz(left.Row2)
        );
    }

    public static Vector3 operator *(Basis basis, Vector3 vector)
    {
        return new Vector3
        (
            Dot(basis.Row0, vector),
            Dot(basis.Row1, vector),
            Dot(basis.Row2, vector)
        );
    }

    public static Vector3 operator *(Vector3 vector, Basis basis)
    {
        return new Vector3
        (
            basis.Row0.X * vector.X + basis.Row1.X * vector.Y + basis.Row2.X * vector.Z,
            basis.Row0.Y * vector.X + basis.Row1.Y * vector.Y + basis.Row2.Y * vector.Z,
            basis.Row0.Z * vector.X + basis.Row1.Z * vector.Y + basis.Row2.Z * vector.Z
        );
    }

    public static bool operator ==(Basis left, Basis right) => left.Equals(right);
    public static bool operator !=(Basis left, Basis right) => !left.Equals(right);

    public override readonly bool Equals([NotNullWhen(true)] object? obj)
    {
        return obj is Basis other && Equals(other);
    }

    public readonly bool Equals(Basis other)
    {
        return Row0.Equals(other.Row0) &&
               Row1.Equals(other.Row1) &&
               Row2.Equals(other.Row2);
    }

    public readonly bool IsEqualApprox(Basis other)
    {
        return IsEqualApprox(Row0, other.Row0) &&
               IsEqualApprox(Row1, other.Row1) &&
               IsEqualApprox(Row2, other.Row2);
    }

    public override readonly int GetHashCode()
    {
        return HashCode.Combine(Row0, Row1, Row2);
    }

    public override readonly string ToString() => ToString(null);

    public readonly string ToString(string? format)
    {
        return $"[X: {FormatVector3(X, format)}, Y: {FormatVector3(Y, format)}, Z: {FormatVector3(Z, format)}]";
    }

    private static void SinCos(float angle, out float sin, out float cos)
    {
        sin = Sin(angle);
        cos = Cos(angle);
    }

    private static float Sin(float v) => MathF.Sin(v);
    private static float Cos(float v) => MathF.Cos(v);
    private static float Tan(float v) => MathF.Tan(v);
    private static float Asin(float v) => MathF.Asin(v);
    private static float Atan2(float y, float x) => MathF.Atan2(y, x);
    private static float Sqrt(float v) => MathF.Sqrt(v);
    private static float Abs(float v) => MathF.Abs(v);

    private static float Tau => MathF.PI * 2f;
    private static float Epsilon => 0.00001f;

    private static float Dot(Vector3 a, Vector3 b)
    {
        return a.X * b.X + a.Y * b.Y + a.Z * b.Z;
    }

    private static Vector3 Cross(Vector3 a, Vector3 b)
    {
        return new Vector3(
            a.Y * b.Z - a.Z * b.Y,
            a.Z * b.X - a.X * b.Z,
            a.X * b.Y - a.Y * b.X
        );
    }

    private static float LengthSquared(Vector3 v)
    {
        return Dot(v, v);
    }

    private static float Length(Vector3 v)
    {
        return Sqrt(LengthSquared(v));
    }

    private static Vector3 Normalize(Vector3 v)
    {
        float len = Length(v);
        if (len == 0f)
            return new Vector3(0f, 0f, 0f);

        float inv = 1f / len;
        return Mul(v, inv);
    }

    private static Vector3 Add(Vector3 a, Vector3 b)
    {
        return new Vector3(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
    }

    private static Vector3 Sub(Vector3 a, Vector3 b)
    {
        return new Vector3(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
    }

    private static Vector3 Mul(Vector3 v, float s)
    {
        return new Vector3(v.X * s, v.Y * s, v.Z * s);
    }

    private static Vector3 Mul(Vector3 a, Vector3 b)
    {
        return new Vector3(a.X * b.X, a.Y * b.Y, a.Z * b.Z);
    }

    private static Vector3 Lerp(Vector3 from, Vector3 to, float weight)
    {
        return new Vector3(
            from.X + (to.X - from.X) * weight,
            from.Y + (to.Y - from.Y) * weight,
            from.Z + (to.Z - from.Z) * weight
        );
    }

    private static float Lerp(float from, float to, float weight)
    {
        return from + (to - from) * weight;
    }

    private static bool IsFinite(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value);
    }

    private static bool IsZeroApprox(Vector3 v)
    {
        return Abs(v.X) <= Epsilon && Abs(v.Y) <= Epsilon && Abs(v.Z) <= Epsilon;
    }

    private static bool IsEqualApprox(Vector3 a, Vector3 b)
    {
        return Abs(a.X - b.X) <= Epsilon &&
               Abs(a.Y - b.Y) <= Epsilon &&
               Abs(a.Z - b.Z) <= Epsilon;
    }

    private static float Sign(float v)
    {
        if (v > 0f) return 1f;
        if (v < 0f) return -1f;
        return 0f;
    }

    private static float GetComponent(Vector3 v, int index)
    {
        return index switch
        {
            0 => v.X,
            1 => v.Y,
            2 => v.Z,
            _ => throw new ArgumentOutOfRangeException(nameof(index))
        };
    }

    private static void SetComponent(ref Vector3 v, int index, float value)
    {
        switch (index)
        {
            case 0: v.X = value; return;
            case 1: v.Y = value; return;
            case 2: v.Z = value; return;
            default: throw new ArgumentOutOfRangeException(nameof(index));
        }
    }

    private static string FormatVector3(Vector3 v, string? format)
    {
        string fmt = string.IsNullOrEmpty(format) ? "G" : format!;
        return $"({v.X.ToString(fmt)}, {v.Y.ToString(fmt)}, {v.Z.ToString(fmt)})";
    }

    private static bool IsPureYRotation_XYZ()
    {
        return false;
    }

    private static bool IsPureXRotation_YXZ()
    {
        return false;
    }

    private static Quaternion SlerpQuaternion(Quaternion from, Quaternion to, float weight)
    {
        float dot = from.X * to.X + from.Y * to.Y + from.Z * to.Z + from.W * to.W;

        if (dot < 0f)
        {
            to = new Quaternion(-to.X, -to.Y, -to.Z, -to.W);
            dot = -dot;
        }

        const float threshold = 0.9995f;
        if (dot > threshold)
        {
            Quaternion result = new Quaternion(
                from.X + (to.X - from.X) * weight,
                from.Y + (to.Y - from.Y) * weight,
                from.Z + (to.Z - from.Z) * weight,
                from.W + (to.W - from.W) * weight
            );

            float len = Sqrt(result.X * result.X + result.Y * result.Y + result.Z * result.Z + result.W * result.W);
            if (len == 0f)
                return new Quaternion(0f, 0f, 0f, 1f);

            float inv = 1f / len;
            return new Quaternion(result.X * inv, result.Y * inv, result.Z * inv, result.W * inv);
        }

        float theta0 = MathF.Acos(dot);
        float theta = theta0 * weight;

        float sinTheta0 = MathF.Sin(theta0);
        float sinTheta = MathF.Sin(theta);

        float s0 = MathF.Cos(theta) - dot * sinTheta / sinTheta0;
        float s1 = sinTheta / sinTheta0;

        return new Quaternion(
            from.X * s0 + to.X * s1,
            from.Y * s0 + to.Y * s1,
            from.Z * s0 + to.Z * s1,
            from.W * s0 + to.W * s1
        );
    }
}
}