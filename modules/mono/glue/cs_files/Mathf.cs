using System;

namespace Godot
{
    public static class Mathf
    {
        public const float PI = 3.14159274f;
        public const float Epsilon = 1e-06f;

        private const float Deg2RadConst = 0.0174532924f;
        private const float Rad2DegConst = 57.29578f;

        public static float abs(float s)
        {
            return Math.Abs(s);
        }

        public static float acos(float s)
        {
            return (float)Math.Acos(s);
        }

        public static float asin(float s)
        {
            return (float)Math.Asin(s);
        }

        public static float atan(float s)
        {
            return (float)Math.Atan(s);
        }

        public static float atan2(float x, float y)
        {
            return (float)Math.Atan2(x, y);
        }

        public static float ceil(float s)
        {
            return (float)Math.Ceiling(s);
        }

        public static float clamp(float val, float min, float max)
        {
            if (val < min)
            {
                return min;
            }
            else if (val > max)
            {
                return max;
            }

            return val;
        }

        public static float cos(float s)
        {
            return (float)Math.Cos(s);
        }

        public static float cosh(float s)
        {
            return (float)Math.Cosh(s);
        }

        public static int decimals(float step)
        {
            return decimals(step);
        }

        public static int decimals(decimal step)
        {
            return BitConverter.GetBytes(decimal.GetBits(step)[3])[2];
        }

        public static float deg2rad(float deg)
        {
            return deg * Deg2RadConst;
        }

        public static float ease(float s, float curve)
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
                    return 1.0f - pow(1.0f - s, 1.0f / curve);
                }

                return pow(s, curve);
            }
            else if (curve < 0f)
            {
                if (s < 0.5f)
                {
                    return pow(s * 2.0f, -curve) * 0.5f;
                }

                return (1.0f - pow(1.0f - (s - 0.5f) * 2.0f, -curve)) * 0.5f + 0.5f;
            }

            return 0f;
        }

        public static float exp(float s)
        {
            return (float)Math.Exp(s);
        }

        public static float floor(float s)
        {
            return (float)Math.Floor(s);
        }

        public static float fposmod(float x, float y)
        {
            if (x >= 0f)
            {
                return x % y;
            }
            else
            {
                return y - (-x % y);
            }
        }

        public static float lerp(float from, float to, float weight)
        {
            return from + (to - from) * clamp(weight, 0f, 1f);
        }

        public static float log(float s)
        {
            return (float)Math.Log(s);
        }

        public static int max(int a, int b)
        {
            return (a > b) ? a : b;
        }

        public static float max(float a, float b)
        {
            return (a > b) ? a : b;
        }

        public static int min(int a, int b)
        {
            return (a < b) ? a : b;
        }

        public static float min(float a, float b)
        {
            return (a < b) ? a : b;
        }

        public static int nearest_po2(int val)
        {
            val--;
            val |= val >> 1;
            val |= val >> 2;
            val |= val >> 4;
            val |= val >> 8;
            val |= val >> 16;
            val++;
            return val;
        }

        public static float pow(float x, float y)
        {
            return (float)Math.Pow(x, y);
        }

        public static float rad2deg(float rad)
        {
            return rad * Rad2DegConst;
        }

        public static float round(float s)
        {
            return (float)Math.Round(s);
        }

        public static float sign(float s)
        {
            return (s < 0f) ? -1f : 1f;
        }

        public static float sin(float s)
        {
            return (float)Math.Sin(s);
        }

        public static float sinh(float s)
        {
            return (float)Math.Sinh(s);
        }

        public static float sqrt(float s)
        {
            return (float)Math.Sqrt(s);
        }

        public static float stepify(float s, float step)
        {
            if (step != 0f)
            {
                s = floor(s / step + 0.5f) * step;
            }

            return s;
        }

        public static float tan(float s)
        {
            return (float)Math.Tan(s);
        }

        public static float tanh(float s)
        {
            return (float)Math.Tanh(s);
        }
    }
}
