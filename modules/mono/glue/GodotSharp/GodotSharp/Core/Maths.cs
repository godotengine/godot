using System;

namespace Godot
{
    /// <summary>
    /// Provides constants and static methods for common mathematical functions with single precision.
    /// </summary>
#if REAL_T_IS_DOUBLE
    public static partial class Maths
#else
    public static partial class Mathf
#endif
    {
        // Define constants with Decimal precision and cast down to double or float.

        /// <include file="Math.xml" path='doc/members/member[@name="Tau"]/*' />
        public const float Tau = (float)6.2831853071795864769252867666M;


        /// <include file="Math.xml" path='doc/members/member[@name="Pi"]/*' />
        public const float Pi = (float)3.1415926535897932384626433833M;

        /// <include file="Math.xml" path='doc/members/member[@name="Inf"]/*' />
        public const float Inf = float.PositiveInfinity;

        /// <include file="Math.xml" path='doc/members/member[@name="NaN"]/*' />
        public const float NaN = float.NaN;

        // 0.0174532924f and 0.0174532925199433
        private const float _degToRadConst = (float)0.0174532925199432957692369077M;
        // 57.29578f and 57.2957795130823
        private const float _radToDegConst = (float)57.295779513082320876798154814M;

        /// <include file="Math.xml" path='doc/members/member[@name="Abs"]/*' />
        public static float Abs(float s)
        {
            return Math.Abs(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Acos"]/*' />
        public static float Acos(float s)
        {
            return (float)Math.Acos(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Asin"]/*' />
        public static float Asin(float s)
        {
            return (float)Math.Asin(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Atan"]/*' />
        public static float Atan(float s)
        {
            return (float)Math.Atan(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Atan2"]/*' />
        public static float Atan2(float y, float x)
        {
            return (float)Math.Atan2(y, x);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Ceil"]/*' />
        public static float Ceil(float s)
        {
            return (float)Math.Ceiling(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Clamp"]/*' />
        public static float Clamp(float value, float min, float max)
        {
            return value < min ? min : value > max ? max : value;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Cos"]/*' />
        public static float Cos(float s)
        {
            return (float)Math.Cos(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Cosh"]/*' />
        public static float Cosh(float s)
        {
            return (float)Math.Cosh(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="CubicInterpolate"]/*' />
        public static float CubicInterpolate(float from, float to, float pre, float post, float weight)
        {
            return 0.5f *
                    ((from * 2.0f) +
                            (-pre + to) * weight +
                            (2.0f * pre - 5.0f * from + 4.0f * to - post) * (weight * weight) +
                            (-pre + 3.0f * from - 3.0f * to + post) * (weight * weight * weight));
        }

        /// <include file="Math.xml" path='doc/members/member[@name="CubicInterpolateAngle"]/*' />
        public static float CubicInterpolateAngle(float from, float to, float pre, float post, float weight)
        {
            float fromRot = from % Mathf.Tau;

            float preDiff = (pre - fromRot) % Mathf.Tau;
            float preRot = fromRot + (2.0f * preDiff) % Mathf.Tau - preDiff;

            float toDiff = (to - fromRot) % Mathf.Tau;
            float toRot = fromRot + (2.0f * toDiff) % Mathf.Tau - toDiff;

            float postDiff = (post - toRot) % Mathf.Tau;
            float postRot = toRot + (2.0f * postDiff) % Mathf.Tau - postDiff;

            return CubicInterpolate(fromRot, toRot, preRot, postRot, weight);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="CubicInterpolateInTime"]/*' />
        public static float CubicInterpolateInTime(float from, float to, float pre, float post, float weight, float toT, float preT, float postT)
        {
            /* Barry-Goldman method */
            float t = Lerp(0.0f, toT, weight);
            float a1 = Lerp(pre, from, preT == 0 ? 0.0f : (t - preT) / -preT);
            float a2 = Lerp(from, to, toT == 0 ? 0.5f : t / toT);
            float a3 = Lerp(to, post, postT - toT == 0 ? 1.0f : (t - toT) / (postT - toT));
            float b1 = Lerp(a1, a2, toT - preT == 0 ? 0.0f : (t - preT) / (toT - preT));
            float b2 = Lerp(a2, a3, postT == 0 ? 1.0f : t / postT);
            return Lerp(b1, b2, toT == 0 ? 0.5f : t / toT);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="CubicInterpolateAngleInTime"]/*' />
        public static float CubicInterpolateAngleInTime(float from, float to, float pre, float post, float weight,
                    float toT, float preT, float postT)
        {
            float fromRot = from % Mathf.Tau;

            float preDiff = (pre - fromRot) % Mathf.Tau;
            float preRot = fromRot + (2.0f * preDiff) % Mathf.Tau - preDiff;

            float toDiff = (to - fromRot) % Mathf.Tau;
            float toRot = fromRot + (2.0f * toDiff) % Mathf.Tau - toDiff;

            float postDiff = (post - toRot) % Mathf.Tau;
            float postRot = toRot + (2.0f * postDiff) % Mathf.Tau - postDiff;

            return CubicInterpolateInTime(fromRot, toRot, preRot, postRot, weight, toT, preT, postT);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="BezierInterpolate"]/*' />
        public static float BezierInterpolate(float start, float control1, float control2, float end, float t)
        {
            // Formula from Wikipedia article on Bezier curves
            float omt = 1 - t;
            float omt2 = omt * omt;
            float omt3 = omt2 * omt;
            float t2 = t * t;
            float t3 = t2 * t;

            return start * omt3 + control1 * omt2 * t * 3 + control2 * omt * t2 * 3 + end * t3;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="DegToRad"]/*' />
        public static float DegToRad(float deg)
        {
            return deg * _degToRadConst;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Ease"]/*' />
        public static float Ease(float s, float curve)
        {
            if (s < 0f)
            {
                s = 0f;
            }
            else if (s > 1.0f)
            {
                s = 1.0f;
            }

            if (curve > 0f)
            {
                if (curve < 1.0f)
                {
                    return 1.0f - Pow(1.0f - s, 1.0f / curve);
                }

                return Pow(s, curve);
            }

            if (curve < 0f)
            {
                if (s < 0.5f)
                {
                    return Pow(s * 2.0f, -curve) * 0.5f;
                }

                return ((1.0f - Pow(1.0f - ((s - 0.5f) * 2.0f), -curve)) * 0.5f) + 0.5f;
            }

            return 0f;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Exp"]/*' />
        public static float Exp(float s)
        {
            return (float)Math.Exp(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Floor"]/*' />
        public static float Floor(float s)
        {
            return (float)Math.Floor(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="InverseLerp"]/*' />
        public static float InverseLerp(float from, float to, float weight)
        {
            return (weight - from) / (to - from);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="IsEqualApprox"]/*' />
        public static bool IsEqualApprox(float a, float b)
        {
            // Check for exact equality first, required to handle "infinity" values.
            if (a == b)
            {
                return true;
            }
            // Then check for approximate equality.
            float tolerance = Epsilon * Abs(a);
            if (tolerance < Epsilon)
            {
                tolerance = Epsilon;
            }
            return Abs(a - b) < tolerance;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="IsInf"]/*' />
        public static bool IsInf(float s)
        {
            return float.IsInfinity(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="IsNaN"]/*' />
        public static bool IsNaN(float s)
        {
            return float.IsNaN(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="IsZeroApprox"]/*' />
        public static bool IsZeroApprox(float s)
        {
            return Abs(s) < Epsilon;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Lerp"]/*' />
        public static float Lerp(float from, float to, float weight)
        {
            return from + ((to - from) * weight);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="LerpAngle"]/*' />
        public static float LerpAngle(float from, float to, float weight)
        {
            float difference = (to - from) % Mathf.Tau;
            float distance = ((2 * difference) % Mathf.Tau) - difference;
            return from + (distance * weight);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Log"]/*' />
        public static float Log(float s)
        {
            return (float)Math.Log(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Max"]/*' />
        public static float Max(float a, float b)
        {
            return a > b ? a : b;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Min"]/*' />
        public static float Min(float a, float b)
        {
            return a < b ? a : b;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="MoveToward"]/*' />
        public static float MoveToward(float from, float to, float delta)
        {
            if (Abs(to - from) <= delta)
                return to;

            return from + (Sign(to - from) * delta);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="PosMod"]/*' />
        public static float PosMod(float a, float b)
        {
            float c = a % b;
            if ((c < 0 && b > 0) || (c > 0 && b < 0))
            {
                c += b;
            }
            return c;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Pow"]/*' />
        public static float Pow(float x, float y)
        {
            return (float)Math.Pow(x, y);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="RadToDeg"]/*' />
        public static float RadToDeg(float rad)
        {
            return rad * _radToDegConst;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Remap"]/*' />
        public static float Remap(float value, float inFrom, float inTo, float outFrom, float outTo)
        {
            return Lerp(outFrom, outTo, InverseLerp(inFrom, inTo, value));
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Round"]/*' />
        public static float Round(float s)
        {
            return (float)Math.Round(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Sign"]/*' />
        public static int Sign(float s)
        {
            if (s == 0)
                return 0;
            return s < 0 ? -1 : 1;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Sin"]/*' />
        public static float Sin(float s)
        {
            return (float)Math.Sin(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Sinh"]/*' />
        public static float Sinh(float s)
        {
            return (float)Math.Sinh(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="SmoothStep"]/*' />
        public static float SmoothStep(float from, float to, float weight)
        {
            if (IsEqualApprox(from, to))
            {
                return from;
            }
            float x = Clamp((weight - from) / (to - from), (float)0.0, (float)1.0);
            return x * x * (3 - (2 * x));
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Sqrt"]/*' />
        public static float Sqrt(float s)
        {
            return (float)Math.Sqrt(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="StepDecimals"]/*' />
        public static int StepDecimals(float step)
        {
            double[] sd = new double[]
            {
                0.9999,
                0.09999,
                0.009999,
                0.0009999,
                0.00009999,
                0.000009999,
                0.0000009999,
                0.00000009999,
                0.000000009999,
            };
            double abs = Abs(step);
            double decs = abs - (int)abs; // Strip away integer part
            for (int i = 0; i < sd.Length; i++)
            {
                if (decs >= sd[i])
                {
                    return i;
                }
            }
            return 0;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Snapped"]/*' />
        public static float Snapped(float s, float step)
        {
            if (step != 0f)
            {
                return Floor((s / step) + 0.5f) * step;
            }

            return s;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Tan"]/*' />
        public static float Tan(float s)
        {
            return (float)Math.Tan(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Tanh"]/*' />
        public static float Tanh(float s)
        {
            return (float)Math.Tanh(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Wrap"]/*' />
        public static float Wrap(float value, float min, float max)
        {
            float range = max - min;
            if (IsZeroApprox(range))
            {
                return min;
            }
            return min + ((((value - min) % range) + range) % range);
        }

        private static float Fract(float value)
        {
            return value - (float)Math.Floor(value);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="PingPong"]/*' />
        public static float PingPong(float value, float length)
        {
            return (length != (float)0.0) ? Abs(Fract((value - length) / (length * (float)2.0)) * length * (float)2.0 - length) : (float)0.0;
        }
    }
}
