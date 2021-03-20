using System;
#if REAL_T_IS_DOUBLE
using real_t = System.Double;
#else
using real_t = System.Single;
#endif

namespace Godot
{
    public static partial class Mathf
    {
        // Define constants with Decimal precision and cast down to double or float.

        /// <summary>
        /// The circle constant, the circumference of the unit circle in radians.
        /// </summary>
        public const real_t Tau = (real_t) 6.2831853071795864769252867666M; // 6.2831855f and 6.28318530717959

        /// <summary>
        /// Constant that represents how many times the diameter of a circle
        /// fits around its perimeter. This is equivalent to `Mathf.Tau / 2`.
        /// </summary>
        public const real_t Pi = (real_t) 3.1415926535897932384626433833M; // 3.1415927f and 3.14159265358979

        /// <summary>
        /// Positive infinity. For negative infinity, use `-Mathf.Inf`.
        /// </summary>
        public const real_t Inf = real_t.PositiveInfinity;

        /// <summary>
        /// "Not a Number", an invalid value. `NaN` has special properties, including
        /// that it is not equal to itself. It is output by some invalid operations,
        /// such as dividing zero by zero.
        /// </summary>
        public const real_t NaN = real_t.NaN;

        private const real_t Deg2RadConst = (real_t) 0.0174532925199432957692369077M; // 0.0174532924f and 0.0174532925199433
        private const real_t Rad2DegConst = (real_t) 57.295779513082320876798154814M; // 57.29578f and 57.2957795130823

        /// <summary>
        /// Returns the absolute value of `s` (i.e. positive value).
        /// </summary>
        /// <param name="s">The input number.</param>
        /// <returns>The absolute value of `s`.</returns>
        public static int Abs(int s)
        {
            return Math.Abs(s);
        }

        /// <summary>
        /// Returns the absolute value of `s` (i.e. positive value).
        /// </summary>
        /// <param name="s">The input number.</param>
        /// <returns>The absolute value of `s`.</returns>
        public static real_t Abs(real_t s)
        {
            return Math.Abs(s);
        }

        /// <summary>
        /// Returns the arc cosine of `s` in radians. Use to get the angle of cosine s.
        /// </summary>
        /// <param name="s">The input cosine value. Must be on the range of -1.0 to 1.0.</param>
        /// <returns>An angle that would result in the given cosine value. On the range `0` to `Tau/2`.</returns>
        public static real_t Acos(real_t s)
        {
            return (real_t)Math.Acos(s);
        }

        /// <summary>
        /// Returns the arc sine of `s` in radians. Use to get the angle of sine s.
        /// </summary>
        /// <param name="s">The input sine value. Must be on the range of -1.0 to 1.0.</param>
        /// <returns>An angle that would result in the given sine value. On the range `-Tau/4` to `Tau/4`.</returns>
        public static real_t Asin(real_t s)
        {
            return (real_t)Math.Asin(s);
        }

        /// <summary>
        /// Returns the arc tangent of `s` in radians. Use to get the angle of tangent s.
        ///
        /// The method cannot know in which quadrant the angle should fall.
        /// See <see cref="Atan2(real_t, real_t)"/> if you have both `y` and `x`.
        /// </summary>
        /// <param name="s">The input tangent value.</param>
        /// <returns>An angle that would result in the given tangent value. On the range `-Tau/4` to `Tau/4`.</returns>
        public static real_t Atan(real_t s)
        {
            return (real_t)Math.Atan(s);
        }

        /// <summary>
        /// Returns the arc tangent of `y` and `x` in radians. Use to get the angle
        /// of the tangent of `y/x`. To compute the value, the method takes into
        /// account the sign of both arguments in order to determine the quadrant.
        ///
        /// Important note: The Y coordinate comes first, by convention.
        /// </summary>
        /// <param name="y">The Y coordinate of the point to find the angle to.</param>
        /// <param name="x">The X coordinate of the point to find the angle to.</param>
        /// <returns>An angle that would result in the given tangent value. On the range `-Tau/2` to `Tau/2`.</returns>
        public static real_t Atan2(real_t y, real_t x)
        {
            return (real_t)Math.Atan2(y, x);
        }

        /// <summary>
        /// Converts a 2D point expressed in the cartesian coordinate
        /// system (X and Y axis) to the polar coordinate system
        /// (a distance from the origin and an angle).
        /// </summary>
        /// <param name="x">The input X coordinate.</param>
        /// <param name="y">The input Y coordinate.</param>
        /// <returns>A <see cref="Vector2"/> with X representing the distance and Y representing the angle.</returns>
        public static Vector2 Cartesian2Polar(real_t x, real_t y)
        {
            return new Vector2(Sqrt(x * x + y * y), Atan2(y, x));
        }

        /// <summary>
        /// Rounds `s` upward (towards positive infinity).
        /// </summary>
        /// <param name="s">The number to ceil.</param>
        /// <returns>The smallest whole number that is not less than `s`.</returns>
        public static real_t Ceil(real_t s)
        {
            return (real_t)Math.Ceiling(s);
        }

        /// <summary>
        /// Clamps a `value` so that it is not less than `min` and not more than `max`.
        /// </summary>
        /// <param name="value">The value to clamp.</param>
        /// <param name="min">The minimum allowed value.</param>
        /// <param name="max">The maximum allowed value.</param>
        /// <returns>The clamped value.</returns>
        public static int Clamp(int value, int min, int max)
        {
            return value < min ? min : value > max ? max : value;
        }

        /// <summary>
        /// Clamps a `value` so that it is not less than `min` and not more than `max`.
        /// </summary>
        /// <param name="value">The value to clamp.</param>
        /// <param name="min">The minimum allowed value.</param>
        /// <param name="max">The maximum allowed value.</param>
        /// <returns>The clamped value.</returns>
        public static real_t Clamp(real_t value, real_t min, real_t max)
        {
            return value < min ? min : value > max ? max : value;
        }

        /// <summary>
        /// Returns the cosine of angle `s` in radians.
        /// </summary>
        /// <param name="s">The angle in radians.</param>
        /// <returns>The cosine of that angle.</returns>
        public static real_t Cos(real_t s)
        {
            return (real_t)Math.Cos(s);
        }

        /// <summary>
        /// Returns the hyperbolic cosine of angle `s` in radians.
        /// </summary>
        /// <param name="s">The angle in radians.</param>
        /// <returns>The hyperbolic cosine of that angle.</returns>
        public static real_t Cosh(real_t s)
        {
            return (real_t)Math.Cosh(s);
        }

        /// <summary>
        /// Converts an angle expressed in degrees to radians.
        /// </summary>
        /// <param name="deg">An angle expressed in degrees.</param>
        /// <returns>The same angle expressed in radians.</returns>
        public static real_t Deg2Rad(real_t deg)
        {
            return deg * Deg2RadConst;
        }

        /// <summary>
        /// Easing function, based on exponent. The curve values are:
        /// `0` is constant, `1` is linear, `0` to `1` is ease-in, `1` or more is ease-out.
        /// Negative values are in-out/out-in.
        /// </summary>
        /// <param name="s">The value to ease.</param>
        /// <param name="curve">`0` is constant, `1` is linear, `0` to `1` is ease-in, `1` or more is ease-out.</param>
        /// <returns>The eased value.</returns>
        public static real_t Ease(real_t s, real_t curve)
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

                return (1.0f - Pow(1.0f - (s - 0.5f) * 2.0f, -curve)) * 0.5f + 0.5f;
            }

            return 0f;
        }

        /// <summary>
        /// The natural exponential function. It raises the mathematical
        /// constant `e` to the power of `s` and returns it.
        /// </summary>
        /// <param name="s">The exponent to raise `e` to.</param>
        /// <returns>`e` raised to the power of `s`.</returns>
        public static real_t Exp(real_t s)
        {
            return (real_t)Math.Exp(s);
        }

        /// <summary>
        /// Rounds `s` downward (towards negative infinity).
        /// </summary>
        /// <param name="s">The number to floor.</param>
        /// <returns>The largest whole number that is not more than `s`.</returns>
        public static real_t Floor(real_t s)
        {
            return (real_t)Math.Floor(s);
        }

        /// <summary>
        /// Returns a normalized value considering the given range.
        /// This is the opposite of <see cref="Lerp(real_t, real_t, real_t)"/>.
        /// </summary>
        /// <param name="from">The interpolated value.</param>
        /// <param name="to">The destination value for interpolation.</param>
        /// <param name="weight">A value on the range of 0.0 to 1.0, representing the amount of interpolation.</param>
        /// <returns>The resulting value of the inverse interpolation.</returns>
        public static real_t InverseLerp(real_t from, real_t to, real_t weight)
        {
            return (weight - from) / (to - from);
        }

        /// <summary>
        /// Returns true if `a` and `b` are approximately equal to each other.
        /// The comparison is done using a tolerance calculation with <see cref="Epsilon"/>.
        /// </summary>
        /// <param name="a">One of the values.</param>
        /// <param name="b">The other value.</param>
        /// <returns>A bool for whether or not the two values are approximately equal.</returns>
        public static bool IsEqualApprox(real_t a, real_t b)
        {
            // Check for exact equality first, required to handle "infinity" values.
            if (a == b)
            {
                return true;
            }
            // Then check for approximate equality.
            real_t tolerance = Epsilon * Abs(a);
            if (tolerance < Epsilon)
            {
                tolerance = Epsilon;
            }
            return Abs(a - b) < tolerance;
        }

        /// <summary>
        /// Returns whether `s` is an infinity value (either positive infinity or negative infinity).
        /// </summary>
        /// <param name="s">The value to check.</param>
        /// <returns>A bool for whether or not the value is an infinity value.</returns>
        public static bool IsInf(real_t s)
        {
            return real_t.IsInfinity(s);
        }

        /// <summary>
        /// Returns whether `s` is a `NaN` ("Not a Number" or invalid) value.
        /// </summary>
        /// <param name="s">The value to check.</param>
        /// <returns>A bool for whether or not the value is a `NaN` value.</returns>
        public static bool IsNaN(real_t s)
        {
            return real_t.IsNaN(s);
        }

        /// <summary>
        /// Returns true if `s` is approximately zero.
        /// The comparison is done using a tolerance calculation with <see cref="Epsilon"/>.
        ///
        /// This method is faster than using <see cref="IsEqualApprox(real_t, real_t)"/> with one value as zero.
        /// </summary>
        /// <param name="s">The value to check.</param>
        /// <returns>A bool for whether or not the value is nearly zero.</returns>
        public static bool IsZeroApprox(real_t s)
        {
            return Abs(s) < Epsilon;
        }

        /// <summary>
        /// Linearly interpolates between two values by a normalized value.
        /// This is the opposite <see cref="InverseLerp(real_t, real_t, real_t)"/>.
        /// </summary>
        /// <param name="from">The start value for interpolation.</param>
        /// <param name="to">The destination value for interpolation.</param>
        /// <param name="weight">A value on the range of 0.0 to 1.0, representing the amount of interpolation.</param>
        /// <returns>The resulting value of the interpolation.</returns>
        public static real_t Lerp(real_t from, real_t to, real_t weight)
        {
            return from + (to - from) * weight;
        }

        /// <summary>
        /// Linearly interpolates between two angles (in radians) by a normalized value.
        ///
        /// Similar to <see cref="Lerp(real_t, real_t, real_t)"/>,
        /// but interpolates correctly when the angles wrap around <see cref="Tau"/>.
        /// </summary>
        /// <param name="from">The start angle for interpolation.</param>
        /// <param name="to">The destination angle for interpolation.</param>
        /// <param name="weight">A value on the range of 0.0 to 1.0, representing the amount of interpolation.</param>
        /// <returns>The resulting angle of the interpolation.</returns>
        public static real_t LerpAngle(real_t from, real_t to, real_t weight)
        {
            real_t difference = (to - from) % Mathf.Tau;
            real_t distance = ((2 * difference) % Mathf.Tau) - difference;
            return from + distance * weight;
        }

        /// <summary>
        /// Natural logarithm. The amount of time needed to reach a certain level of continuous growth.
        ///
        /// Note: This is not the same as the "log" function on most calculators, which uses a base 10 logarithm.
        /// </summary>
        /// <param name="s">The input value.</param>
        /// <returns>The natural log of `s`.</returns>
        public static real_t Log(real_t s)
        {
            return (real_t)Math.Log(s);
        }

        /// <summary>
        /// Returns the maximum of two values.
        /// </summary>
        /// <param name="a">One of the values.</param>
        /// <param name="b">The other value.</param>
        /// <returns>Whichever of the two values is higher.</returns>
        public static int Max(int a, int b)
        {
            return a > b ? a : b;
        }

        /// <summary>
        /// Returns the maximum of two values.
        /// </summary>
        /// <param name="a">One of the values.</param>
        /// <param name="b">The other value.</param>
        /// <returns>Whichever of the two values is higher.</returns>
        public static real_t Max(real_t a, real_t b)
        {
            return a > b ? a : b;
        }

        /// <summary>
        /// Returns the minimum of two values.
        /// </summary>
        /// <param name="a">One of the values.</param>
        /// <param name="b">The other value.</param>
        /// <returns>Whichever of the two values is lower.</returns>
        public static int Min(int a, int b)
        {
            return a < b ? a : b;
        }

        /// <summary>
        /// Returns the minimum of two values.
        /// </summary>
        /// <param name="a">One of the values.</param>
        /// <param name="b">The other value.</param>
        /// <returns>Whichever of the two values is lower.</returns>
        public static real_t Min(real_t a, real_t b)
        {
            return a < b ? a : b;
        }

        /// <summary>
        /// Moves `from` toward `to` by the `delta` value.
        ///
        /// Use a negative delta value to move away.
        /// </summary>
        /// <param name="from">The start value.</param>
        /// <param name="to">The value to move towards.</param>
        /// <param name="delta">The amount to move by.</param>
        /// <returns>The value after moving.</returns>
        public static real_t MoveToward(real_t from, real_t to, real_t delta)
        {
            return Abs(to - from) <= delta ? to : from + Sign(to - from) * delta;
        }

        /// <summary>
        /// Returns the nearest larger power of 2 for the integer `value`.
        /// </summary>
        /// <param name="value">The input value.</param>
        /// <returns>The nearest larger power of 2.</returns>
        public static int NearestPo2(int value)
        {
            value--;
            value |= value >> 1;
            value |= value >> 2;
            value |= value >> 4;
            value |= value >> 8;
            value |= value >> 16;
            value++;
            return value;
        }

        /// <summary>
        /// Converts a 2D point expressed in the polar coordinate
        /// system (a distance from the origin `r` and an angle `th`)
        /// to the cartesian coordinate system (X and Y axis).
        /// </summary>
        /// <param name="r">The distance from the origin.</param>
        /// <param name="th">The angle of the point.</param>
        /// <returns>A <see cref="Vector2"/> representing the cartesian coordinate.</returns>
        public static Vector2 Polar2Cartesian(real_t r, real_t th)
        {
            return new Vector2(r * Cos(th), r * Sin(th));
        }

        /// <summary>
        /// Performs a canonical Modulus operation, where the output is on the range `[0, b)`.
        /// </summary>
        /// <param name="a">The dividend, the primary input.</param>
        /// <param name="b">The divisor. The output is on the range `[0, b)`.</param>
        /// <returns>The resulting output.</returns>
        public static int PosMod(int a, int b)
        {
            int c = a % b;
            if ((c < 0 && b > 0) || (c > 0 && b < 0))
            {
                c += b;
            }
            return c;
        }

        /// <summary>
        /// Performs a canonical Modulus operation, where the output is on the range `[0, b)`.
        /// </summary>
        /// <param name="a">The dividend, the primary input.</param>
        /// <param name="b">The divisor. The output is on the range `[0, b)`.</param>
        /// <returns>The resulting output.</returns>
        public static real_t PosMod(real_t a, real_t b)
        {
            real_t c = a % b;
            if ((c < 0 && b > 0) || (c > 0 && b < 0))
            {
                c += b;
            }
            return c;
        }

        /// <summary>
        /// Returns the result of `x` raised to the power of `y`.
        /// </summary>
        /// <param name="x">The base.</param>
        /// <param name="y">The exponent.</param>
        /// <returns>`x` raised to the power of `y`.</returns>
        public static real_t Pow(real_t x, real_t y)
        {
            return (real_t)Math.Pow(x, y);
        }

        /// <summary>
        /// Converts an angle expressed in radians to degrees.
        /// </summary>
        /// <param name="rad">An angle expressed in radians.</param>
        /// <returns>The same angle expressed in degrees.</returns>
        public static real_t Rad2Deg(real_t rad)
        {
            return rad * Rad2DegConst;
        }

        /// <summary>
        /// Rounds `s` to the nearest whole number,
        /// with halfway cases rounded towards the nearest multiple of two.
        /// </summary>
        /// <param name="s">The number to round.</param>
        /// <returns>The rounded number.</returns>
        public static real_t Round(real_t s)
        {
            return (real_t)Math.Round(s);
        }

        /// <summary>
        /// Returns the sign of `s`: `-1` or `1`. Returns `0` if `s` is `0`.
        /// </summary>
        /// <param name="s">The input number.</param>
        /// <returns>One of three possible values: `1`, `-1`, or `0`.</returns>
        public static int Sign(int s)
        {
            if (s == 0) return 0;
            return s < 0 ? -1 : 1;
        }

        /// <summary>
        /// Returns the sign of `s`: `-1` or `1`. Returns `0` if `s` is `0`.
        /// </summary>
        /// <param name="s">The input number.</param>
        /// <returns>One of three possible values: `1`, `-1`, or `0`.</returns>
        public static int Sign(real_t s)
        {
            if (s == 0) return 0;
            return s < 0 ? -1 : 1;
        }

        /// <summary>
        /// Returns the sine of angle `s` in radians.
        /// </summary>
        /// <param name="s">The angle in radians.</param>
        /// <returns>The sine of that angle.</returns>
        public static real_t Sin(real_t s)
        {
            return (real_t)Math.Sin(s);
        }

        /// <summary>
        /// Returns the hyperbolic sine of angle `s` in radians.
        /// </summary>
        /// <param name="s">The angle in radians.</param>
        /// <returns>The hyperbolic sine of that angle.</returns>
        public static real_t Sinh(real_t s)
        {
            return (real_t)Math.Sinh(s);
        }

        /// <summary>
        /// Returns a number smoothly interpolated between `from` and `to`,
        /// based on the `weight`. Similar to <see cref="Lerp(real_t, real_t, real_t)"/>,
        /// but interpolates faster at the beginning and slower at the end.
        /// </summary>
        /// <param name="from">The start value for interpolation.</param>
        /// <param name="to">The destination value for interpolation.</param>
        /// <param name="weight">A value representing the amount of interpolation.</param>
        /// <returns>The resulting value of the interpolation.</returns>
        public static real_t SmoothStep(real_t from, real_t to, real_t weight)
        {
            if (IsEqualApprox(from, to))
            {
                return from;
            }
            real_t x = Clamp((weight - from) / (to - from), (real_t)0.0, (real_t)1.0);
            return x * x * (3 - 2 * x);
        }

        /// <summary>
        /// Returns the square root of `s`, where `s` is a non-negative number.
        ///
        /// If you need negative inputs, use `System.Numerics.Complex`.
        /// </summary>
        /// <param name="s">The input number. Must not be negative.</param>
        /// <returns>The square root of `s`.</returns>
        public static real_t Sqrt(real_t s)
        {
            return (real_t)Math.Sqrt(s);
        }

        /// <summary>
        /// Returns the position of the first non-zero digit, after the
        /// decimal point. Note that the maximum return value is 10,
        /// which is a design decision in the implementation.
        /// </summary>
        /// <param name="step">The input value.</param>
        /// <returns>The position of the first non-zero digit.</returns>
        public static int StepDecimals(real_t step)
        {
            double[] sd = new double[] {
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
            double abs = Mathf.Abs(step);
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

        /// <summary>
        /// Snaps float value `s` to a given `step`.
        /// This can also be used to round a floating point
        /// number to an arbitrary number of decimals.
        /// </summary>
        /// <param name="s">The value to snap.</param>
        /// <param name="step">The step size to snap to.</param>
        /// <returns></returns>
        public static real_t Snapped(real_t s, real_t step)
        {
            if (step != 0f)
            {
                return Floor(s / step + 0.5f) * step;
            }

            return s;
        }

        /// <summary>
        /// Returns the tangent of angle `s` in radians.
        /// </summary>
        /// <param name="s">The angle in radians.</param>
        /// <returns>The tangent of that angle.</returns>
        public static real_t Tan(real_t s)
        {
            return (real_t)Math.Tan(s);
        }

        /// <summary>
        /// Returns the hyperbolic tangent of angle `s` in radians.
        /// </summary>
        /// <param name="s">The angle in radians.</param>
        /// <returns>The hyperbolic tangent of that angle.</returns>
        public static real_t Tanh(real_t s)
        {
            return (real_t)Math.Tanh(s);
        }

        /// <summary>
        /// Wraps `value` between `min` and `max`. Usable for creating loop-alike
        /// behavior or infinite surfaces. If `min` is `0`, this is equivalent
        /// to <see cref="PosMod(int, int)"/>, so prefer using that instead.
        /// </summary>
        /// <param name="value">The value to wrap.</param>
        /// <param name="min">The minimum allowed value and lower bound of the range.</param>
        /// <param name="max">The maximum allowed value and upper bound of the range.</param>
        /// <returns>The wrapped value.</returns>
        public static int Wrap(int value, int min, int max)
        {
            int range = max - min;
            return range == 0 ? min : min + ((value - min) % range + range) % range;
        }

        /// <summary>
        /// Wraps `value` between `min` and `max`. Usable for creating loop-alike
        /// behavior or infinite surfaces. If `min` is `0`, this is equivalent
        /// to <see cref="PosMod(real_t, real_t)"/>, so prefer using that instead.
        /// </summary>
        /// <param name="value">The value to wrap.</param>
        /// <param name="min">The minimum allowed value and lower bound of the range.</param>
        /// <param name="max">The maximum allowed value and upper bound of the range.</param>
        /// <returns>The wrapped value.</returns>
        public static real_t Wrap(real_t value, real_t min, real_t max)
        {
            real_t range = max - min;
            return IsZeroApprox(range) ? min : min + ((value - min) % range + range) % range;
        }
    }
}
