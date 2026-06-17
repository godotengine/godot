using System;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Runtime.InteropServices;

#nullable enable
namespace Godot
{
/// <summary>
/// A color represented by red, green, blue, and alpha (RGBA) components.
/// </summary>
[Serializable]
[StructLayout(LayoutKind.Sequential)]
public struct Color : IEquatable<Color>
{
    public float R;
    public float G;
    public float B;
    public float A;

    public int R8
    {
        readonly get => (int)Math.Round(R * 255.0f);
        set => R = value / 255.0f;
    }

    public int G8
    {
        readonly get => (int)Math.Round(G * 255.0f);
        set => G = value / 255.0f;
    }

    public int B8
    {
        readonly get => (int)Math.Round(B * 255.0f);
        set => B = value / 255.0f;
    }

    public int A8
    {
        readonly get => (int)Math.Round(A * 255.0f);
        set => A = value / 255.0f;
    }

    public float H
    {
        readonly get
        {
            float max = Math.Max(R, Math.Max(G, B));
            float min = Math.Min(R, Math.Min(G, B));
            float delta = max - min;

            if (delta == 0f)
                return 0f;

            float h;

            if (R == max)
                h = (G - B) / delta;
            else if (G == max)
                h = 2f + ((B - R) / delta);
            else
                h = 4f + ((R - G) / delta);

            h /= 6.0f;

            if (h < 0f)
                h += 1.0f;

            return h;
        }
        set
        {
            this = FromHsv(value, S, V, A);
        }
    }

    public float S
    {
        readonly get
        {
            float max = Math.Max(R, Math.Max(G, B));
            float min = Math.Min(R, Math.Min(G, B));
            float delta = max - min;

            return max == 0f ? 0f : delta / max;
        }
        set
        {
            this = FromHsv(H, value, V, A);
        }
    }

    public float V
    {
        readonly get => Math.Max(R, Math.Max(G, B));
        set
        {
            this = FromHsv(H, S, value, A);
        }
    }

    public readonly float Luminance
    {
        get => 0.2126f * R + 0.7152f * G + 0.0722f * B;
    }

    public float this[int index]
    {
        readonly get
        {
            switch (index)
            {
                case 0: return R;
                case 1: return G;
                case 2: return B;
                case 3: return A;
                default: throw new ArgumentOutOfRangeException(nameof(index));
            }
        }
        set
        {
            switch (index)
            {
                case 0: R = value; return;
                case 1: G = value; return;
                case 2: B = value; return;
                case 3: A = value; return;
                default: throw new ArgumentOutOfRangeException(nameof(index));
            }
        }
    }

    public readonly Color Blend(Color over)
    {
        Color res;
        float sa = 1.0f - over.A;
        res.A = (A * sa) + over.A;

        if (res.A == 0f)
            return new Color(0f, 0f, 0f, 0f);

        res.R = ((R * A * sa) + (over.R * over.A)) / res.A;
        res.G = ((G * A * sa) + (over.G * over.A)) / res.A;
        res.B = ((B * A * sa) + (over.B * over.A)) / res.A;

        return res;
    }

    public readonly Color Clamp(Color? min = null, Color? max = null)
    {
        Color minimum = min ?? new Color(0f, 0f, 0f, 0f);
        Color maximum = max ?? new Color(1f, 1f, 1f, 1f);

        return new Color(
            Math.Clamp(R, minimum.R, maximum.R),
            Math.Clamp(G, minimum.G, maximum.G),
            Math.Clamp(B, minimum.B, maximum.B),
            Math.Clamp(A, minimum.A, maximum.A)
        );
    }

    public readonly Color Darkened(float amount)
    {
        Color res = this;
        res.R *= 1.0f - amount;
        res.G *= 1.0f - amount;
        res.B *= 1.0f - amount;
        return res;
    }

    public readonly Color Inverted()
    {
        return new Color(1.0f - R, 1.0f - G, 1.0f - B, A);
    }

    public readonly Color Lightened(float amount)
    {
        Color res = this;
        res.R += (1.0f - res.R) * amount;
        res.G += (1.0f - res.G) * amount;
        res.B += (1.0f - res.B) * amount;
        return res;
    }

    public readonly Color Lerp(Color to, float weight)
    {
        return new Color(
            Lerp(R, to.R, weight),
            Lerp(G, to.G, weight),
            Lerp(B, to.B, weight),
            Lerp(A, to.A, weight)
        );
    }

    public readonly Color LinearToSrgb()
    {
        return new Color(
            R < 0.0031308f ? 12.92f * R : (1.0f + 0.055f) * MathF.Pow(R, 1.0f / 2.4f) - 0.055f,
            G < 0.0031308f ? 12.92f * G : (1.0f + 0.055f) * MathF.Pow(G, 1.0f / 2.4f) - 0.055f,
            B < 0.0031308f ? 12.92f * B : (1.0f + 0.055f) * MathF.Pow(B, 1.0f / 2.4f) - 0.055f,
            A
        );
    }

    public readonly Color SrgbToLinear()
    {
        return new Color(
            R < 0.04045f ? R * (1.0f / 12.92f) : MathF.Pow((R + 0.055f) * (1.0f / 1.055f), 2.4f),
            G < 0.04045f ? G * (1.0f / 12.92f) : MathF.Pow((G + 0.055f) * (1.0f / 1.055f), 2.4f),
            B < 0.04045f ? B * (1.0f / 12.92f) : MathF.Pow((B + 0.055f) * (1.0f / 1.055f), 2.4f),
            A
        );
    }

    public readonly uint ToAbgr32()
    {
        uint c = (byte)Math.Round(A * 255);
        c <<= 8;
        c |= (byte)Math.Round(B * 255);
        c <<= 8;
        c |= (byte)Math.Round(G * 255);
        c <<= 8;
        c |= (byte)Math.Round(R * 255);
        return c;
    }

    public readonly long ToAbgr64()
    {
        long c = (ushort)Math.Round(A * 65535);
        c <<= 16;
        c |= (ushort)Math.Round(B * 65535);
        c <<= 16;
        c |= (ushort)Math.Round(G * 65535);
        c <<= 16;
        c |= (ushort)Math.Round(R * 65535);
        return c;
    }

    public readonly uint ToArgb32()
    {
        uint c = (byte)Math.Round(A * 255);
        c <<= 8;
        c |= (byte)Math.Round(R * 255);
        c <<= 8;
        c |= (byte)Math.Round(G * 255);
        c <<= 8;
        c |= (byte)Math.Round(B * 255);
        return c;
    }

    public readonly long ToArgb64()
    {
        long c = (ushort)Math.Round(A * 65535);
        c <<= 16;
        c |= (ushort)Math.Round(R * 65535);
        c <<= 16;
        c |= (ushort)Math.Round(G * 65535);
        c <<= 16;
        c |= (ushort)Math.Round(B * 65535);
        return c;
    }

    public readonly uint ToRgba32()
    {
        uint c = (byte)Math.Round(R * 255);
        c <<= 8;
        c |= (byte)Math.Round(G * 255);
        c <<= 8;
        c |= (byte)Math.Round(B * 255);
        c <<= 8;
        c |= (byte)Math.Round(A * 255);
        return c;
    }

    public readonly long ToRgba64()
    {
        long c = (ushort)Math.Round(R * 65535);
        c <<= 16;
        c |= (ushort)Math.Round(G * 65535);
        c <<= 16;
        c |= (ushort)Math.Round(B * 65535);
        c <<= 16;
        c |= (ushort)Math.Round(A * 65535);
        return c;
    }

    public readonly string ToHtml(bool includeAlpha = true)
    {
        string txt = string.Empty;
        txt += ToHex32(R);
        txt += ToHex32(G);
        txt += ToHex32(B);

        if (includeAlpha)
            txt += ToHex32(A);

        return txt;
    }

    public readonly void ToHsv(out float hue, out float saturation, out float value)
    {
        float max = Math.Max(R, Math.Max(G, B));
        float min = Math.Min(R, Math.Min(G, B));
        float delta = max - min;

        if (delta == 0f)
        {
            hue = 0f;
        }
        else if (R == max)
        {
            hue = (G - B) / delta;
            hue /= 6.0f;
            if (hue < 0f) hue += 1.0f;
        }
        else if (G == max)
        {
            hue = 2f + ((B - R) / delta);
            hue /= 6.0f;
        }
        else
        {
            hue = 4f + ((R - G) / delta);
            hue /= 6.0f;
        }

        saturation = max == 0f ? 0f : 1f - (min / max);
        value = max;
    }

    public Color(float r, float g, float b, float a = 1.0f)
    {
        R = r;
        G = g;
        B = b;
        A = a;
    }

    public Color(Color c, float a = 1.0f)
    {
        R = c.R;
        G = c.G;
        B = c.B;
        A = a;
    }

    public Color(uint rgba)
    {
        A = (rgba & 0xFF) / 255.0f;
        rgba >>= 8;
        B = (rgba & 0xFF) / 255.0f;
        rgba >>= 8;
        G = (rgba & 0xFF) / 255.0f;
        rgba >>= 8;
        R = (rgba & 0xFF) / 255.0f;
    }

    public Color(long rgba)
    {
        A = (rgba & 0xFFFF) / 65535.0f;
        rgba >>= 16;
        B = (rgba & 0xFFFF) / 65535.0f;
        rgba >>= 16;
        G = (rgba & 0xFFFF) / 65535.0f;
        rgba >>= 16;
        R = (rgba & 0xFFFF) / 65535.0f;
    }

    public Color(string code)
    {
        if (HtmlIsValid(code))
        {
            this = FromHtml(code);
        }
        else
        {
            throw new ArgumentOutOfRangeException(nameof(code), $"Invalid color code: {code}");
        }
    }

    public Color(string code, float alpha)
    {
        this = new Color(code);
        A = alpha;
    }

    public static Color Color8(byte r8, byte g8, byte b8, byte a8 = 255)
    {
        return new Color(r8 / 255f, g8 / 255f, b8 / 255f, a8 / 255f);
    }

    public static Color FromHtml(ReadOnlySpan<char> rgba)
    {
        Color c;
        if (rgba.Length == 0)
        {
            c.R = 0f;
            c.G = 0f;
            c.B = 0f;
            c.A = 1.0f;
            return c;
        }

        if (rgba[0] == '#')
            rgba = rgba.Slice(1);

        bool alpha;
        bool isShorthand;

        if (rgba.Length == 8)
        {
            alpha = true;
            isShorthand = false;
        }
        else if (rgba.Length == 6)
        {
            alpha = false;
            isShorthand = false;
        }
        else if (rgba.Length == 4)
        {
            alpha = true;
            isShorthand = true;
        }
        else if (rgba.Length == 3)
        {
            alpha = false;
            isShorthand = true;
        }
        else
        {
            throw new ArgumentOutOfRangeException(nameof(rgba),
                $"Invalid color code. Length is {rgba.Length}, but a length of 3, 4, 6 or 8 is expected.");
        }

        c.A = 1.0f;

        if (isShorthand)
        {
            c.R = ParseCol4(rgba, 0) / 15f;
            c.G = ParseCol4(rgba, 1) / 15f;
            c.B = ParseCol4(rgba, 2) / 15f;
            if (alpha)
                c.A = ParseCol4(rgba, 3) / 15f;
        }
        else
        {
            c.R = ParseCol8(rgba, 0) / 255f;
            c.G = ParseCol8(rgba, 2) / 255f;
            c.B = ParseCol8(rgba, 4) / 255f;
            if (alpha)
                c.A = ParseCol8(rgba, 6) / 255f;
        }

        if (c.R < 0 || c.G < 0 || c.B < 0 || c.A < 0)
            throw new ArgumentOutOfRangeException(nameof(rgba), $"Invalid color code: {rgba}");

        return c;
    }

    public static Color FromHsv(float hue, float saturation, float value, float alpha = 1.0f)
    {
        if (saturation == 0f)
            return new Color(value, value, value, alpha);

        hue *= 6.0f;
        hue %= 6f;
        int i = (int)hue;

        float f = hue - i;
        float p = value * (1 - saturation);
        float q = value * (1 - (saturation * f));
        float t = value * (1 - (saturation * (1 - f)));

        return i switch
        {
            0 => new Color(value, t, p, alpha),
            1 => new Color(q, value, p, alpha),
            2 => new Color(p, value, t, alpha),
            3 => new Color(p, q, value, alpha),
            4 => new Color(t, p, value, alpha),
            _ => new Color(value, p, q, alpha),
        };
    }

    public static Color FromString(string str, Color @default)
    {
        return HtmlIsValid(str) ? FromHtml(str) : @default;
    }

    public static bool HtmlIsValid(ReadOnlySpan<char> color)
    {
        if (color.IsEmpty)
            return false;

        if (color[0] == '#')
            color = color.Slice(1);

        int len = color.Length;
        if (!(len == 3 || len == 4 || len == 6 || len == 8))
            return false;

        for (int i = 0; i < len; i++)
        {
            if (ParseCol4(color, i) == -1)
                return false;
        }

        return true;
    }

    public static Color operator +(Color left, Color right)
    {
        left.R += right.R;
        left.G += right.G;
        left.B += right.B;
        left.A += right.A;
        return left;
    }

    public static Color operator -(Color left, Color right)
    {
        left.R -= right.R;
        left.G -= right.G;
        left.B -= right.B;
        left.A -= right.A;
        return left;
    }

    public static Color operator -(Color color)
    {
        return new Color(1f - color.R, 1f - color.G, 1f - color.B, 1f - color.A);
    }

    public static Color operator *(Color color, float scale)
    {
        color.R *= scale;
        color.G *= scale;
        color.B *= scale;
        color.A *= scale;
        return color;
    }

    public static Color operator *(float scale, Color color) => color * scale;

    public static Color operator *(Color left, Color right)
    {
        left.R *= right.R;
        left.G *= right.G;
        left.B *= right.B;
        left.A *= right.A;
        return left;
    }

    public static Color operator /(Color color, float scale)
    {
        color.R /= scale;
        color.G /= scale;
        color.B /= scale;
        color.A /= scale;
        return color;
    }

    public static Color operator /(Color left, Color right)
    {
        left.R /= right.R;
        left.G /= right.G;
        left.B /= right.B;
        left.A /= right.A;
        return left;
    }

    public static bool operator ==(Color left, Color right) => left.Equals(right);
    public static bool operator !=(Color left, Color right) => !left.Equals(right);

    public static bool operator <(Color left, Color right)
    {
        if (left.R == right.R)
        {
            if (left.G == right.G)
            {
                if (left.B == right.B)
                    return left.A < right.A;

                return left.B < right.B;
            }

            return left.G < right.G;
        }

        return left.R < right.R;
    }

    public static bool operator >(Color left, Color right)
    {
        if (left.R == right.R)
        {
            if (left.G == right.G)
            {
                if (left.B == right.B)
                    return left.A > right.A;

                return left.B > right.B;
            }

            return left.G > right.G;
        }

        return left.R > right.R;
    }

    public static bool operator <=(Color left, Color right)
    {
        if (left.R == right.R)
        {
            if (left.G == right.G)
            {
                if (left.B == right.B)
                    return left.A <= right.A;

                return left.B < right.B;
            }

            return left.G < right.G;
        }

        return left.R < right.R;
    }

    public static bool operator >=(Color left, Color right)
    {
        if (left.R == right.R)
        {
            if (left.G == right.G)
            {
                if (left.B == right.B)
                    return left.A >= right.A;

                return left.B > right.B;
            }

            return left.G > right.G;
        }

        return left.R > right.R;
    }

    public override readonly bool Equals([NotNullWhen(true)] object? obj)
    {
        return obj is Color other && Equals(other);
    }

    public readonly bool Equals(Color other)
    {
        return R == other.R && G == other.G && B == other.B && A == other.A;
    }

    public readonly bool IsEqualApprox(Color other)
    {
        return IsEqualApprox(R, other.R) &&
               IsEqualApprox(G, other.G) &&
               IsEqualApprox(B, other.B) &&
               IsEqualApprox(A, other.A);
    }

    public override readonly int GetHashCode()
    {
        return HashCode.Combine(R, G, B, A);
    }

    public override readonly string ToString() => ToString(null);

    public readonly string ToString(string? format)
    {
        string f = string.IsNullOrEmpty(format) ? "G" : format!;
        return $"({R.ToString(f, CultureInfo.InvariantCulture)}, {G.ToString(f, CultureInfo.InvariantCulture)}, {B.ToString(f, CultureInfo.InvariantCulture)}, {A.ToString(f, CultureInfo.InvariantCulture)})";
    }

    private static int ParseCol4(ReadOnlySpan<char> str, int index)
    {
        char character = str[index];

        if (character >= '0' && character <= '9')
            return character - '0';
        if (character >= 'a' && character <= 'f')
            return character + (10 - 'a');
        if (character >= 'A' && character <= 'F')
            return character + (10 - 'A');

        return -1;
    }

    private static int ParseCol8(ReadOnlySpan<char> str, int index)
    {
        return ParseCol4(str, index) * 16 + ParseCol4(str, index + 1);
    }

    private static float Lerp(float from, float to, float weight)
    {
        return from + (to - from) * weight;
    }

    private static float ToHexComponent(float val)
    {
        return Math.Clamp(val * 255f, 0f, 255f);
    }

    private static string ToHex32(float val)
    {
        byte b = (byte)Math.Round(ToHexComponent(val));
        return b.ToString("X2", CultureInfo.InvariantCulture);
    }

    private static bool IsEqualApprox(float a, float b)
    {
        return MathF.Abs(a - b) <= 0.00001f;
    }
}
}