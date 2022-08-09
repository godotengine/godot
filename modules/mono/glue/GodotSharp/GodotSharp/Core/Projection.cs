#if REAL_T_IS_DOUBLE
using real_t = System.Double;
#else
using real_t = System.Single;
#endif
using System;
using System.Runtime.InteropServices;

namespace Godot
{
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Projection : IEquatable<Projection>
    {
        /// <summary>
        /// Enumerated index values for the planes.
        /// </summary>
        public enum Planes
        {
            /// <summary>
            /// The projection's near plane.
            /// </summary>
            Near,
            /// <summary>
            /// The projection's far plane.
            /// </summary>
            Far,
            /// <summary>
            /// The projection's left plane.
            /// </summary>
            Left,
            /// <summary>
            /// The projection's top plane.
            /// </summary>
            Top,
            /// <summary>
            /// The projection's right plane.
            /// </summary>
            Right,
            /// <summary>
            /// The projection's bottom plane.
            /// </summary>
            Bottom,
        }

        /// <summary>
        /// The projections's X column. Also accessible by using the index position <c>[0]</c>.
        /// </summary>
        public Vector4 x;

        /// <summary>
        /// The projections's Y column. Also accessible by using the index position <c>[1]</c>.
        /// </summary>
        public Vector4 y;

        /// <summary>
        /// The projections's Z column. Also accessible by using the index position <c>[2]</c>.
        /// </summary>
        public Vector4 z;

        /// <summary>
        /// The projections's W column. Also accessible by using the index position <c>[3]</c>.
        /// </summary>
        public Vector4 w;

        /// <summary>
        /// Constructs a projection from 4 vectors (matrix columns).
        /// </summary>
        /// <param name="x">The X column, or column index 0.</param>
        /// <param name="y">The Y column, or column index 1.</param>
        /// <param name="z">The Z column, or column index 2.</param>
        /// <param name="w">The W column, or column index 3.</param>
        public Projection(Vector4 x, Vector4 y, Vector4 z, Vector4 w)
        {
            this.x = x;
            this.y = y;
            this.z = z;
            this.w = w;
        }

        /// <summary>
        /// Constructs a new <see cref="Projection"/> from an existing <see cref="Projection"/>.
        /// </summary>
        /// <param name="proj">The existing <see cref="Projection"/>.</param>
        public Projection(Projection proj)
        {
            x = proj.x;
            y = proj.y;
            z = proj.z;
            w = proj.w;
        }

        /// <summary>
        /// Constructs a new <see cref="Projection"/> from a <see cref="Transform3D"/>.
        /// </summary>
        /// <param name="transform">The <see cref="Transform3D"/>.</param>
        public Projection(Transform3D transform)
        {
            x = new Vector4(transform.basis.Row0.x, transform.basis.Row1.x, transform.basis.Row2.x, 0);
            y = new Vector4(transform.basis.Row0.y, transform.basis.Row1.y, transform.basis.Row2.y, 0);
            z = new Vector4(transform.basis.Row0.z, transform.basis.Row1.z, transform.basis.Row2.z, 0);
            w = new Vector4(transform.origin.x, transform.origin.y, transform.origin.z, 1);
        }

        /// <summary>
        /// Constructs a new <see cref="Transform3D"/> from the <see cref="Projection"/>.
        /// </summary>
        /// <param name="proj">The <see cref="Projection"/>.</param>
        public static explicit operator Transform3D(Projection proj)
        {
            return new Transform3D(
                new Basis(
                    new Vector3(proj.x.x, proj.x.y, proj.x.z),
                    new Vector3(proj.y.x, proj.y.y, proj.y.z),
                    new Vector3(proj.z.x, proj.z.y, proj.z.z)
                ),
                new Vector3(proj.w.x, proj.w.y, proj.w.z)
            );
        }

        public static Projection CreateDepthCorrection(bool flipY)
        {
            return new Projection(
                new Vector4(1, 0, 0, 0),
                new Vector4(0, flipY ? -1 : 1, 0, 0),
                new Vector4(0, 0, (real_t)0.5, 0),
                new Vector4(0, 0, (real_t)0.5, 1)
            );
        }

        public static Projection CreateFitAabb(AABB aabb)
        {
            Vector3 min = aabb.Position;
            Vector3 max = aabb.Position + aabb.Size;

            return new Projection(
                new Vector4(2 / (max.x - min.x), 0, 0, 0),
                new Vector4(0, 2 / (max.y - min.y), 0, 0),
                new Vector4(0, 0, 2 / (max.z - min.z), 0),
                new Vector4(-(max.x + min.x) / (max.x - min.x), -(max.y + min.y) / (max.y - min.y), -(max.z + min.z) / (max.z - min.z), 1)
            );
        }

        public static Projection CreateForHmd(int eye, real_t aspect, real_t intraocularDist, real_t displayWidth, real_t displayToLens, real_t oversample, real_t zNear, real_t zFar)
        {
            real_t f1 = (intraocularDist * (real_t)0.5) / displayToLens;
            real_t f2 = ((displayWidth - intraocularDist) * (real_t)0.5) / displayToLens;
            real_t f3 = (displayWidth / (real_t)4.0) / displayToLens;

            real_t add = ((f1 + f2) * (oversample - (real_t)1.0)) / (real_t)2.0;
            f1 += add;
            f2 += add;
            f3 *= oversample;

            f3 /= aspect;

            switch (eye)
            {
                case 1:
                    return CreateFrustum(-f2 * zNear, f1 * zNear, -f3 * zNear, f3 * zNear, zNear, zFar);
                case 2:
                    return CreateFrustum(-f1 * zNear, f2 * zNear, -f3 * zNear, f3 * zNear, zNear, zFar);
                default:
                    return Zero;
            }
        }

        public static Projection CreateFrustum(real_t left, real_t right, real_t bottom, real_t top, real_t near, real_t far)
        {
            if (right <= left)
            {
                throw new ArgumentException("right is less or equal to left.");
            }
            if (top <= bottom)
            {
                throw new ArgumentException("top is less or equal to bottom.");
            }
            if (far <= near)
            {
                throw new ArgumentException("far is less or equal to near.");
            }

            real_t x = 2 * near / (right - left);
            real_t y = 2 * near / (top - bottom);

            real_t a = (right + left) / (right - left);
            real_t b = (top + bottom) / (top - bottom);
            real_t c = -(far + near) / (far - near);
            real_t d = -2 * far * near / (far - near);

            return new Projection(
                new Vector4(x, 0, 0, 0),
                new Vector4(0, y, 0, 0),
                new Vector4(a, b, c, -1),
                new Vector4(0, 0, d, 0)
            );
        }

        public static Projection CreateFrustumAspect(real_t size, real_t aspect, Vector2 offset, real_t near, real_t far, bool flipFov)
        {
            if (!flipFov)
            {
                size *= aspect;
            }
            return CreateFrustum(-size / 2 + offset.x, +size / 2 + offset.x, -size / aspect / 2 + offset.y, +size / aspect / 2 + offset.y, near, far);
        }

        public static Projection CreateLightAtlasRect(Rect2 rect)
        {
            return new Projection(
                new Vector4(rect.Size.x, 0, 0, 0),
                new Vector4(0, rect.Size.y, 0, 0),
                new Vector4(0, 0, 1, 0),
                new Vector4(rect.Position.x, rect.Position.y, 0, 1)
            );
        }

        public static Projection CreateOrthogonal(real_t left, real_t right, real_t bottom, real_t top, real_t zNear, real_t zFar)
        {
            Projection proj = Projection.Identity;
            proj.x.x = (real_t)2.0 / (right - left);
            proj.w.x = -((right + left) / (right - left));
            proj.y.y = (real_t)2.0 / (top - bottom);
            proj.w.y = -((top + bottom) / (top - bottom));
            proj.z.z = (real_t)(-2.0) / (zFar - zNear);
            proj.w.z = -((zFar + zNear) / (zFar - zNear));
            proj.w.w = (real_t)1.0;
            return proj;
        }

        public static Projection CreateOrthogonalAspect(real_t size, real_t aspect, real_t zNear, real_t zFar, bool flipFov)
        {
            if (!flipFov)
            {
                size *= aspect;
            }
            return CreateOrthogonal(-size / 2, +size / 2, -size / aspect / 2, +size / aspect / 2, zNear, zFar);
        }

        public static Projection CreatePerspective(real_t fovyDegrees, real_t aspect, real_t zNear, real_t zFar, bool flipFov)
        {
            if (flipFov)
            {
                fovyDegrees = GetFovy(fovyDegrees, (real_t)1.0 / aspect);
            }
            real_t radians = Mathf.Deg2Rad(fovyDegrees / (real_t)2.0);
            real_t deltaZ = zFar - zNear;
            real_t sine = Mathf.Sin(radians);

            if ((deltaZ == 0) || (sine == 0) || (aspect == 0))
            {
                return Zero;
            }

            real_t cotangent = Mathf.Cos(radians) / sine;

            Projection proj = Projection.Identity;

            proj.x.x = cotangent / aspect;
            proj.y.y = cotangent;
            proj.z.z = -(zFar + zNear) / deltaZ;
            proj.z.w = -1;
            proj.w.z = -2 * zNear * zFar / deltaZ;
            proj.w.w = 0;

            return proj;
        }

        public static Projection CreatePerspectiveHmd(real_t fovyDegrees, real_t aspect, real_t zNear, real_t zFar, bool flipFov, int eye, real_t intraocularDist, real_t convergenceDist)
        {
            if (flipFov)
            {
                fovyDegrees = GetFovy(fovyDegrees, (real_t)1.0 / aspect);
            }

            real_t ymax = zNear * Mathf.Tan(Mathf.Deg2Rad(fovyDegrees / (real_t)2.0));
            real_t xmax = ymax * aspect;
            real_t frustumshift = (intraocularDist / (real_t)2.0) * zNear / convergenceDist;
            real_t left;
            real_t right;
            real_t modeltranslation;
            switch (eye)
            {
                case 1:
                    left = -xmax + frustumshift;
                    right = xmax + frustumshift;
                    modeltranslation = intraocularDist / (real_t)2.0;
                    break;
                case 2:
                    left = -xmax - frustumshift;
                    right = xmax - frustumshift;
                    modeltranslation = -intraocularDist / (real_t)2.0;
                    break;
                default:
                    left = -xmax;
                    right = xmax;
                    modeltranslation = (real_t)0.0;
                    break;
            }
            Projection proj = CreateFrustum(left, right, -ymax, ymax, zNear, zFar);
            Projection cm = Projection.Identity;
            cm.w.x = modeltranslation;
            return proj * cm;
        }

        public real_t Determinant()
        {
            return x.w * y.z * z.y * w.x - x.z * y.w * z.y * w.x -
                   x.w * y.y * z.z * w.x + x.y * y.w * z.z * w.x +
                   x.z * y.y * z.w * w.x - x.y * y.z * z.w * w.x -
                   x.w * y.z * z.x * w.y + x.z * y.w * z.x * w.y +
                   x.w * y.x * z.z * w.y - x.x * y.w * z.z * w.y -
                   x.z * y.x * z.w * w.y + x.x * y.z * z.w * w.y +
                   x.w * y.y * z.x * w.z - x.y * y.w * z.x * w.z -
                   x.w * y.x * z.y * w.z + x.x * y.w * z.y * w.z +
                   x.y * y.x * z.w * w.z - x.x * y.y * z.w * w.z -
                   x.z * y.y * z.x * w.w + x.y * y.z * z.x * w.w +
                   x.z * y.x * z.y * w.w - x.x * y.z * z.y * w.w -
                   x.y * y.x * z.z * w.w + x.x * y.y * z.z * w.w;
        }

        public real_t GetAspect()
        {
            Vector2 vpHe = GetViewportHalfExtents();
            return vpHe.x / vpHe.y;
        }

        public real_t GetFov()
        {
            Plane rightPlane = new Plane(x.w - x.x, y.w - y.x, z.w - z.x, -w.w + w.x).Normalized();
            if (z.x == 0 && z.y == 0)
            {
                return Mathf.Rad2Deg(Mathf.Acos(Mathf.Abs(rightPlane.Normal.x))) * (real_t)2.0;
            }
            else
            {
                Plane leftPlane = new Plane(x.w + x.x, y.w + y.x, z.w + z.x, w.w + w.x).Normalized();
                return Mathf.Rad2Deg(Mathf.Acos(Mathf.Abs(leftPlane.Normal.x))) + Mathf.Rad2Deg(Mathf.Acos(Mathf.Abs(rightPlane.Normal.x)));
            }
        }

        public static real_t GetFovy(real_t fovx, real_t aspect)
        {
            return Mathf.Rad2Deg(Mathf.Atan(aspect * Mathf.Tan(Mathf.Deg2Rad(fovx) * (real_t)0.5)) * (real_t)2.0);
        }

        public real_t GetLodMultiplier()
        {
            if (IsOrthogonal())
            {
                return GetViewportHalfExtents().x;
            }
            else
            {
                real_t zn = GetZNear();
                real_t width = GetViewportHalfExtents().x * (real_t)2.0;
                return (real_t)1.0 / (zn / width);
            }
        }

        public int GetPixelsPerMeter(int forPixelWidth)
        {
            Vector3 result = Xform(new Vector3(1, 0, -1));

            return (int)((result.x * (real_t)0.5 + (real_t)0.5) * forPixelWidth);
        }

        public Plane GetProjectionPlane(Planes plane)
        {
            Plane newPlane = plane switch
            {
                Planes.Near => new Plane(x.w + x.z, y.w + y.z, z.w + z.z, w.w + w.z),
                Planes.Far => new Plane(x.w - x.z, y.w - y.z, z.w - z.z, w.w - w.z),
                Planes.Left => new Plane(x.w + x.x, y.w + y.x, z.w + z.x, w.w + w.x),
                Planes.Top => new Plane(x.w - x.y, y.w - y.y, z.w - z.y, w.w - w.y),
                Planes.Right => new Plane(x.w - x.x, y.w - y.x, z.w - z.x, w.w - w.x),
                Planes.Bottom => new Plane(x.w + x.y, y.w + y.y, z.w + z.y, w.w + w.y),
                _ => new Plane(),
            };
            newPlane.Normal = -newPlane.Normal;
            return newPlane.Normalized();
        }

        public Vector2 GetFarPlaneHalfExtents()
        {
            var res = GetProjectionPlane(Planes.Far).Intersect3(GetProjectionPlane(Planes.Right), GetProjectionPlane(Planes.Top));
            return new Vector2(res.Value.x, res.Value.y);
        }

        public Vector2 GetViewportHalfExtents()
        {
            var res = GetProjectionPlane(Planes.Near).Intersect3(GetProjectionPlane(Planes.Right), GetProjectionPlane(Planes.Top));
            return new Vector2(res.Value.x, res.Value.y);
        }

        public real_t GetZFar()
        {
            return GetProjectionPlane(Planes.Far).D;
        }

        public real_t GetZNear()
        {
            return -GetProjectionPlane(Planes.Near).D;
        }

        public Projection FlippedY()
        {
            Projection proj = this;
            proj.y = -proj.y;
            return proj;
        }

        public Projection PerspectiveZNearAdjusted(real_t newZNear)
        {
            Projection proj = this;
            real_t zFar = GetZFar();
            real_t zNear = newZNear;
            real_t deltaZ = zFar - zNear;
            proj.z.z = -(zFar + zNear) / deltaZ;
            proj.w.z = -2 * zNear * zFar / deltaZ;
            return proj;
        }

        public Projection JitterOffseted(Vector2 offset)
        {
            Projection proj = this;
            proj.w.x += offset.x;
            proj.w.y += offset.y;
            return proj;
        }

        public Projection Inverse()
        {
            Projection proj = this;
            int i, j, k;
            int[] pvt_i = new int[4];
            int[] pvt_j = new int[4]; /* Locations of pivot matrix */
            real_t pvt_val; /* Value of current pivot element */
            real_t hold; /* Temporary storage */
            real_t determinant = 1.0f;
            for (k = 0; k < 4; k++)
            {
                /* Locate k'th pivot element */
                pvt_val = proj[k][k]; /* Initialize for search */
                pvt_i[k] = k;
                pvt_j[k] = k;
                for (i = k; i < 4; i++)
                {
                    for (j = k; j < 4; j++)
                    {
                        if (Mathf.Abs(proj[i][j]) > Mathf.Abs(pvt_val))
                        {
                            pvt_i[k] = i;
                            pvt_j[k] = j;
                            pvt_val = proj[i][j];
                        }
                    }
                }

                /* Product of pivots, gives determinant when finished */
                determinant *= pvt_val;
                if (Mathf.IsZeroApprox(determinant))
                {
                    return Zero;
                }

                /* "Interchange" rows (with sign change stuff) */
                i = pvt_i[k];
                if (i != k)
                { /* If rows are different */
                    for (j = 0; j < 4; j++)
                    {
                        hold = -proj[k][j];
                        proj[k, j] = proj[i][j];
                        proj[i, j] = hold;
                    }
                }

                /* "Interchange" columns */
                j = pvt_j[k];
                if (j != k)
                { /* If columns are different */
                    for (i = 0; i < 4; i++)
                    {
                        hold = -proj[i][k];
                        proj[i, k] = proj[i][j];
                        proj[i, j] = hold;
                    }
                }

                /* Divide column by minus pivot value */
                for (i = 0; i < 4; i++)
                {
                    if (i != k)
                    {
                        proj[i, k] /= (-pvt_val);
                    }
                }

                /* Reduce the matrix */
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

                /* Divide row by pivot */
                for (j = 0; j < 4; j++)
                {
                    if (j != k)
                    {
                        proj[k, j] /= pvt_val;
                    }
                }

                /* Replace pivot by reciprocal (at last we can touch it). */
                proj[k, k] = (real_t)1.0 / pvt_val;
            }

            /* That was most of the work, one final pass of row/column interchange */
            /* to finish */
            for (k = 4 - 2; k >= 0; k--)
            { /* Don't need to work with 1 by 1 corner*/
                i = pvt_j[k]; /* Rows to swap correspond to pivot COLUMN */
                if (i != k)
                { /* If rows are different */
                    for (j = 0; j < 4; j++)
                    {
                        hold = proj[k][j];
                        proj[k, j] = -proj[i][j];
                        proj[i, j] = hold;
                    }
                }

                j = pvt_i[k]; /* Columns to swap correspond to pivot ROW */
                if (j != k)
                { /* If columns are different */
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

        public bool IsOrthogonal()
        {
            return w.w == (real_t)1.0;
        }

        /// <summary>
        /// Composes these two projections by multiplying them
        /// together. This has the effect of applying the right
        /// and then the left projection.
        /// </summary>
        /// <param name="left">The parent transform.</param>
        /// <param name="right">The child transform.</param>
        /// <returns>The composed projection.</returns>
        public static Projection operator *(Projection left, Projection right)
        {
            return new Projection(
                new Vector4(
                    left.x.x * right.x.x + left.y.x * right.x.y + left.z.x * right.x.z + left.w.x * right.x.w,
                    left.x.y * right.x.x + left.y.y * right.x.y + left.z.y * right.x.z + left.w.y * right.x.w,
                    left.x.z * right.x.x + left.y.z * right.x.y + left.z.z * right.x.z + left.w.z * right.x.w,
                    left.x.w * right.x.x + left.y.w * right.x.y + left.z.w * right.x.z + left.w.w * right.x.w
                ), new Vector4(
                    left.x.x * right.y.x + left.y.x * right.y.y + left.z.x * right.y.z + left.w.x * right.y.w,
                    left.x.y * right.y.x + left.y.y * right.y.y + left.z.y * right.y.z + left.w.y * right.y.w,
                    left.x.z * right.y.x + left.y.z * right.y.y + left.z.z * right.y.z + left.w.z * right.y.w,
                    left.x.w * right.y.x + left.y.w * right.y.y + left.z.w * right.y.z + left.w.w * right.y.w
                ), new Vector4(
                    left.x.x * right.z.x + left.y.x * right.z.y + left.z.x * right.z.z + left.w.x * right.z.w,
                    left.x.y * right.z.x + left.y.y * right.z.y + left.z.y * right.z.z + left.w.y * right.z.w,
                    left.x.z * right.z.x + left.y.z * right.z.y + left.z.z * right.z.z + left.w.z * right.z.w,
                    left.x.w * right.z.x + left.y.w * right.z.y + left.z.w * right.z.z + left.w.w * right.z.w
                ), new Vector4(
                    left.x.x * right.w.x + left.y.x * right.w.y + left.z.x * right.w.z + left.w.x * right.w.w,
                    left.x.y * right.w.x + left.y.y * right.w.y + left.z.y * right.w.z + left.w.y * right.w.w,
                    left.x.z * right.w.x + left.y.z * right.w.y + left.z.z * right.w.z + left.w.z * right.w.w,
                    left.x.w * right.w.x + left.y.w * right.w.y + left.z.w * right.w.z + left.w.w * right.w.w
                )
            );
        }

        /// <summary>
        /// Returns a vector transformed (multiplied) by this projection.
        /// </summary>
        /// <param name="proj">The projection to apply.</param>
        /// <param name="v">A vector to transform.</param>
        /// <returns>The transformed vector.</returns>
        public static Vector4 operator *(Projection proj, Vector4 v)
        {
            return new Vector4(
                proj.x.x * v.x + proj.y.x * v.y + proj.z.x * v.z + proj.w.x * v.w,
                proj.x.y * v.x + proj.y.y * v.y + proj.z.y * v.z + proj.w.y * v.w,
                proj.x.z * v.x + proj.y.z * v.y + proj.z.z * v.z + proj.w.z * v.w,
                proj.x.w * v.x + proj.y.w * v.y + proj.z.w * v.z + proj.w.w * v.w
            );
        }

        /// <summary>
        /// Returns <see langword="true"/> if the projections are exactly equal.
        /// </summary>
        /// <param name="left">The left projection.</param>
        /// <param name="right">The right projection.</param>
        /// <returns>Whether or not the projections are exactly equal.</returns>
        public static bool operator ==(Projection left, Projection right)
        {
            return left.Equals(right);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the projections are not exactly equal.
        /// </summary>
        /// <param name="left">The left projection.</param>
        /// <param name="right">The right projection.</param>
        /// <returns>Whether or not the projections are not exactly equal.</returns>
        public static bool operator !=(Projection left, Projection right)
        {
            return !left.Equals(right);
        }

        /// <summary>
        /// Access whole columns in the form of <see cref="Vector4"/>.
        /// </summary>
        /// <param name="column">Which column vector.</param>
        public Vector4 this[int column]
        {
            get
            {
                switch (column)
                {
                    case 0:
                        return x;
                    case 1:
                        return y;
                    case 2:
                        return z;
                    case 3:
                        return w;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
            set
            {
                switch (column)
                {
                    case 0:
                        x = value;
                        return;
                    case 1:
                        y = value;
                        return;
                    case 2:
                        z = value;
                        return;
                    case 3:
                        w = value;
                        return;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        /// <summary>
        /// Access single values.
        /// </summary>
        /// <param name="column">Which column vector.</param>
        /// <param name="row">Which row of the column.</param>
        public real_t this[int column, int row]
        {
            get
            {
                switch (column)
                {
                    case 0:
                        return x[row];
                    case 1:
                        return y[row];
                    case 2:
                        return z[row];
                    case 3:
                        return w[row];
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
            set
            {
                switch (column)
                {
                    case 0:
                        x[row] = value;
                        return;
                    case 1:
                        y[row] = value;
                        return;
                    case 2:
                        z[row] = value;
                        return;
                    case 3:
                        w[row] = value;
                        return;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        /// <summary>
        /// Returns a vector transformed (multiplied) by this projection.
        /// </summary>
        /// <param name="v">A vector to transform.</param>
        /// <returns>The transformed vector.</returns>
        private Vector3 Xform(Vector3 v)
        {
            Vector3 ret = new Vector3(
                x.x * v.x + y.x * v.y + z.x * v.z + w.x,
                x.y * v.x + y.y * v.y + z.y * v.z + w.y,
                x.z * v.x + y.z * v.y + z.z * v.z + w.z
            );
            return ret / (x.w * v.x + y.w * v.y + z.w * v.z + w.w);
        }

        // Constants
        private static readonly Projection _zero = new Projection(
            new Vector4(0, 0, 0, 0),
            new Vector4(0, 0, 0, 0),
            new Vector4(0, 0, 0, 0),
            new Vector4(0, 0, 0, 0)
        );
        private static readonly Projection _identity = new Projection(
            new Vector4(1, 0, 0, 0),
            new Vector4(0, 1, 0, 0),
            new Vector4(0, 0, 1, 0),
            new Vector4(0, 0, 0, 1)
        );

        /// <summary>
        /// Zero projection, a projection with all components set to <c>0</c>.
        /// </summary>
        /// <value>Equivalent to <c>new Projection(Vector4.Zero, Vector4.Zero, Vector4.Zero, Vector4.Zero)</c>.</value>
        public static Projection Zero { get { return _zero; } }

        /// <summary>
        /// The identity projection, with no distortion applied.
        /// This is used as a replacement for <c>Projection()</c> in GDScript.
        /// Do not use <c>new Projection()</c> with no arguments in C#, because it sets all values to zero.
        /// </summary>
        /// <value>Equivalent to <c>new Projection(new Vector4(1, 0, 0, 0), new Vector4(0, 1, 0, 0), new Vector4(0, 0, 1, 0), new Vector4(0, 0, 0, 1))</c>.</value>
        public static Projection Identity { get { return _identity; } }

        /// <summary>
        /// Serves as the hash function for <see cref="Projection"/>.
        /// </summary>
        /// <returns>A hash code for this projection.</returns>
        public override int GetHashCode()
        {
            return y.GetHashCode() ^ x.GetHashCode() ^ z.GetHashCode() ^ w.GetHashCode();
        }

        /// <summary>
        /// Converts this <see cref="Projection"/> to a string.
        /// </summary>
        /// <returns>A string representation of this projection.</returns>
        public override string ToString()
        {
            return $"{x.x}, {x.y}, {x.z}, {x.w}\n{y.x}, {y.y}, {y.z}, {y.w}\n{z.x}, {z.y}, {z.z}, {z.w}\n{w.x}, {w.y}, {w.z}, {w.w}\n";
        }

        /// <summary>
        /// Converts this <see cref="Projection"/> to a string with the given <paramref name="format"/>.
        /// </summary>
        /// <returns>A string representation of this projection.</returns>
        public string ToString(string format)
        {
            return $"{x.x.ToString(format)}, {x.y.ToString(format)}, {x.z.ToString(format)}, {x.w.ToString(format)}\n" +
                $"{y.x.ToString(format)}, {y.y.ToString(format)}, {y.z.ToString(format)}, {y.w.ToString(format)}\n" +
                $"{z.x.ToString(format)}, {z.y.ToString(format)}, {z.z.ToString(format)}, {z.w.ToString(format)}\n" +
                $"{w.x.ToString(format)}, {w.y.ToString(format)}, {w.z.ToString(format)}, {w.w.ToString(format)}\n";
        }

        /// <summary>
        /// Returns <see langword="true"/> if the projection is exactly equal
        /// to the given object (<see paramref="obj"/>).
        /// </summary>
        /// <param name="obj">The object to compare with.</param>
        /// <returns>Whether or not the vector and the object are equal.</returns>
        public override bool Equals(object obj)
        {
            if (obj is Projection)
            {
                return Equals((Projection)obj);
            }
            return false;
        }

        /// <summary>
        /// Returns <see langword="true"/> if the projections are exactly equal.
        /// </summary>
        /// <param name="other">The other projection.</param>
        /// <returns>Whether or not the projections are exactly equal.</returns>
        public bool Equals(Projection other)
        {
            return x == other.x && y == other.y && z == other.z && w == other.w;
        }
    }
}
