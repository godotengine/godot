using System;
using System.Runtime.InteropServices;

namespace Godot
{
    /// <summary>
    /// A color represented by red, green, blue, and alpha (RGBA) components.
    /// The alpha component is often used for transparency.
    /// Values are in floating-point and usually range from 0 to 1.
    /// Some properties (such as <see cref="CanvasItem.Modulate"/>) may accept values
    /// greater than 1 (overbright or HDR colors).
    ///
    /// If you want to supply values in a range of 0 to 255, you should use
    /// <see cref="Color8"/> and the <c>r8</c>/<c>g8</c>/<c>b8</c>/<c>a8</c> properties.
    /// </summary>
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Color : IEquatable<Color>
    {
        /// <summary>
        /// The color's red component, typically on the range of 0 to 1.
        /// </summary>
        public float r;

        /// <summary>
        /// The color's green component, typically on the range of 0 to 1.
        /// </summary>
        public float g;

        /// <summary>
        /// The color's blue component, typically on the range of 0 to 1.
        /// </summary>
        public float b;

        /// <summary>
        /// The color's alpha (transparency) component, typically on the range of 0 to 1.
        /// </summary>
        public float a;

        /// <summary>
        /// Wrapper for <see cref="r"/> that uses the range 0 to 255 instead of 0 to 1.
        /// </summary>
        /// <value>Getting is equivalent to multiplying by 255 and rounding. Setting is equivalent to dividing by 255.</value>
        public int r8
        {
            get
            {
                return (int)Math.Round(r * 255.0f);
            }
            set
            {
                r = value / 255.0f;
            }
        }

        /// <summary>
        /// Wrapper for <see cref="g"/> that uses the range 0 to 255 instead of 0 to 1.
        /// </summary>
        /// <value>Getting is equivalent to multiplying by 255 and rounding. Setting is equivalent to dividing by 255.</value>
        public int g8
        {
            get
            {
                return (int)Math.Round(g * 255.0f);
            }
            set
            {
                g = value / 255.0f;
            }
        }

        /// <summary>
        /// Wrapper for <see cref="b"/> that uses the range 0 to 255 instead of 0 to 1.
        /// </summary>
        /// <value>Getting is equivalent to multiplying by 255 and rounding. Setting is equivalent to dividing by 255.</value>
        public int b8
        {
            get
            {
                return (int)Math.Round(b * 255.0f);
            }
            set
            {
                b = value / 255.0f;
            }
        }

        /// <summary>
        /// Wrapper for <see cref="a"/> that uses the range 0 to 255 instead of 0 to 1.
        /// </summary>
        /// <value>Getting is equivalent to multiplying by 255 and rounding. Setting is equivalent to dividing by 255.</value>
        public int a8
        {
            get
            {
                return (int)Math.Round(a * 255.0f);
            }
            set
            {
                a = value / 255.0f;
            }
        }

        /// <summary>
        /// The HSV hue of this color, on the range 0 to 1.
        /// </summary>
        /// <value>Getting is a long process, refer to the source code for details. Setting uses <see cref="FromHsv"/>.</value>
        public float h
        {
            get
            {
                float max = Math.Max(r, Math.Max(g, b));
                float min = Math.Min(r, Math.Min(g, b));

                float delta = max - min;

                if (delta == 0)
                {
                    return 0;
                }

                float h;

                if (r == max)
                {
                    h = (g - b) / delta; // Between yellow & magenta
                }
                else if (g == max)
                {
                    h = 2 + ((b - r) / delta); // Between cyan & yellow
                }
                else
                {
                    h = 4 + ((r - g) / delta); // Between magenta & cyan
                }

                h /= 6.0f;

                if (h < 0)
                {
                    h += 1.0f;
                }

                return h;
            }
            set
            {
                this = FromHsv(value, s, v, a);
            }
        }

        /// <summary>
        /// The HSV saturation of this color, on the range 0 to 1.
        /// </summary>
        /// <value>Getting is equivalent to the ratio between the min and max RGB value. Setting uses <see cref="FromHsv"/>.</value>
        public float s
        {
            get
            {
                float max = Math.Max(r, Math.Max(g, b));
                float min = Math.Min(r, Math.Min(g, b));

                float delta = max - min;

                return max == 0 ? 0 : delta / max;
            }
            set
            {
                this = FromHsv(h, value, v, a);
            }
        }

        /// <summary>
        /// The HSV value (brightness) of this color, on the range 0 to 1.
        /// </summary>
        /// <value>Getting is equivalent to using <see cref="Math.Max(float, float)"/> on the RGB components. Setting uses <see cref="FromHsv"/>.</value>
        public float v
        {
            get
            {
                return Math.Max(r, Math.Max(g, b));
            }
            set
            {
                this = FromHsv(h, s, value, a);
            }
        }

        /// <summary>
        /// Returns a color according to the standardized name, with the
        /// specified alpha value. Supported color names are the same as
        /// the constants defined in <see cref="Colors"/>.
        /// </summary>
        /// <param name="name">The name of the color.</param>
        /// <param name="alpha">The alpha (transparency) component represented on the range of 0 to 1. Default: 1.</param>
        /// <returns>The constructed color.</returns>
        public static Color ColorN(string name, float alpha = 1f)
        {
            name = name.Replace(" ", String.Empty);
            name = name.Replace("-", String.Empty);
            name = name.Replace("_", String.Empty);
            name = name.Replace("'", String.Empty);
            name = name.Replace(".", String.Empty);
            name = name.ToLower();

            if (!Colors.namedColors.ContainsKey(name))
            {
                throw new ArgumentOutOfRangeException($"Invalid Color Name: {name}");
            }

            Color color = Colors.namedColors[name];
            color.a = alpha;
            return color;
        }

        /// <summary>
        /// Access color components using their index.
        /// </summary>
        /// <value>
        /// <c>[0]</c> is equivalent to <see cref="r"/>,
        /// <c>[1]</c> is equivalent to <see cref="g"/>,
        /// <c>[2]</c> is equivalent to <see cref="b"/>,
        /// <c>[3]</c> is equivalent to <see cref="a"/>.
        /// </value>
        public float this[int index]
        {
            get
            {
                switch (index)
                {
                    case 0:
                        return r;
                    case 1:
                        return g;
                    case 2:
                        return b;
                    case 3:
                        return a;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
            set
            {
                switch (index)
                {
                    case 0:
                        r = value;
                        return;
                    case 1:
                        g = value;
                        return;
                    case 2:
                        b = value;
                        return;
                    case 3:
                        a = value;
                        return;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        /// <summary>
        /// Converts a color to HSV values. This is equivalent to using each of
        /// the <c>h</c>/<c>s</c>/<c>v</c> properties, but much more efficient.
        /// </summary>
        /// <param name="hue">Output parameter for the HSV hue.</param>
        /// <param name="saturation">Output parameter for the HSV saturation.</param>
        /// <param name="value">Output parameter for the HSV value.</param>
        public void ToHsv(out float hue, out float saturation, out float value)
        {
            float max = (float)Mathf.Max(r, Mathf.Max(g, b));
            float min = (float)Mathf.Min(r, Mathf.Min(g, b));

            float delta = max - min;

            if (delta == 0)
            {
                hue = 0;
            }
            else
            {
                if (r == max)
                    hue = (g - b) / delta; // Between yellow & magenta
                else if (g == max)
                    hue = 2 + (b - r) / delta; // Between cyan & yellow
                else
                    hue = 4 + (r - g) / delta; // Between magenta & cyan

                hue /= 6.0f;

                if (hue < 0)
                    hue += 1.0f;
            }

            saturation = max == 0 ? 0 : 1f - 1f * min / max;
            value = max;
        }

        /// <summary>
        /// Constructs a color from an HSV profile, with values on the
        /// range of 0 to 1. This is equivalent to using each of
        /// the <c>h</c>/<c>s</c>/<c>v</c> properties, but much more efficient.
        /// </summary>
        /// <param name="hue">The HSV hue, typically on the range of 0 to 1.</param>
        /// <param name="saturation">The HSV saturation, typically on the range of 0 to 1.</param>
        /// <param name="value">The HSV value (brightness), typically on the range of 0 to 1.</param>
        /// <param name="alpha">The alpha (transparency) value, typically on the range of 0 to 1.</param>
        /// <returns>The constructed color.</returns>
        public static Color FromHsv(float hue, float saturation, float value, float alpha = 1.0f)
        {
            if (saturation == 0)
            {
                // acp_hromatic (grey)
                return new Color(value, value, value, alpha);
            }

            int i;
            float f, p, q, t;

            hue *= 6.0f;
            hue %= 6f;
            i = (int)hue;

            f = hue - i;
            p = value * (1 - saturation);
            q = value * (1 - saturation * f);
            t = value * (1 - saturation * (1 - f));

            switch (i)
            {
                case 0: // Red is the dominant color
                    return new Color(value, t, p, alpha);
                case 1: // Green is the dominant color
                    return new Color(q, value, p, alpha);
                case 2:
                    return new Color(p, value, t, alpha);
                case 3: // Blue is the dominant color
                    return new Color(p, q, value, alpha);
                case 4:
                    return new Color(t, p, value, alpha);
                default: // (5) Red is the dominant color
                    return new Color(value, p, q, alpha);
            }
        }

        /// <summary>
        /// Returns a new color resulting from blending this color over another.
        /// If the color is opaque, the result is also opaque.
        /// The second color may have a range of alpha values.
        /// </summary>
        /// <param name="over">The color to blend over.</param>
        /// <returns>This color blended over <paramref name="over"/>.</returns>
        public Color Blend(Color over)
        {
            Color res;

            float sa = 1.0f - over.a;
            res.a = (a * sa) + over.a;

            if (res.a == 0)
            {
                return new Color(0, 0, 0, 0);
            }

            res.r = ((r * a * sa) + (over.r * over.a)) / res.a;
            res.g = ((g * a * sa) + (over.g * over.a)) / res.a;
            res.b = ((b * a * sa) + (over.b * over.a)) / res.a;

            return res;
        }

        /// <summary>
        /// Returns the most contrasting color.
        /// </summary>
        /// <returns>The most contrasting color</returns>
        public Color Contrasted()
        {
            return new Color(
                (r + 0.5f) % 1.0f,
                (g + 0.5f) % 1.0f,
                (b + 0.5f) % 1.0f,
                a
            );
        }

        /// <summary>
        /// Returns a new color resulting from making this color darker
        /// by the specified ratio (on the range of 0 to 1).
        /// </summary>
        /// <param name="amount">The ratio to darken by.</param>
        /// <returns>The darkened color.</returns>
        public Color Darkened(float amount)
        {
            Color res = this;
            res.r *= 1.0f - amount;
            res.g *= 1.0f - amount;
            res.b *= 1.0f - amount;
            return res;
        }

        /// <summary>
        /// Returns the inverted color: <c>(1 - r, 1 - g, 1 - b, a)</c>.
        /// </summary>
        /// <returns>The inverted color.</returns>
        public Color Inverted()
        {
            return new Color(
                1.0f - r,
                1.0f - g,
                1.0f - b,
                a
            );
        }

        /// <summary>
        /// Returns a new color resulting from making this color lighter
        /// by the specified ratio (on the range of 0 to 1).
        /// </summary>
        /// <param name="amount">The ratio to lighten by.</param>
        /// <returns>The darkened color.</returns>
        public Color Lightened(float amount)
        {
            Color res = this;
            res.r += (1.0f - res.r) * amount;
            res.g += (1.0f - res.g) * amount;
            res.b += (1.0f - res.b) * amount;
            return res;
        }

        /// <summary>
        /// Returns the result of the linear interpolation between
        /// this color and <paramref name="to"/> by amount <paramref name="weight"/>.
        /// </summary>
        /// <param name="to">The destination color for interpolation.</param>
        /// <param name="weight">A value on the range of 0.0 to 1.0, representing the amount of interpolation.</param>
        /// <returns>The resulting color of the interpolation.</returns>
        public Color LinearInterpolate(Color to, float weight)
        {
            return new Color
            (
                Mathf.Lerp(r, to.r, weight),
                Mathf.Lerp(g, to.g, weight),
                Mathf.Lerp(b, to.b, weight),
                Mathf.Lerp(a, to.a, weight)
            );
        }

        /// <summary>
        /// Returns the result of the linear interpolation between
        /// this color and <paramref name="to"/> by color amount <paramref name="weight"/>.
        /// </summary>
        /// <param name="to">The destination color for interpolation.</param>
        /// <param name="weight">A color with components on the range of 0.0 to 1.0, representing the amount of interpolation.</param>
        /// <returns>The resulting color of the interpolation.</returns>
        public Color LinearInterpolate(Color to, Color weight)
        {
            return new Color
            (
                Mathf.Lerp(r, to.r, weight.r),
                Mathf.Lerp(g, to.g, weight.g),
                Mathf.Lerp(b, to.b, weight.b),
                Mathf.Lerp(a, to.a, weight.a)
            );
        }

        /// <summary>
        /// Returns the color converted to a 32-bit integer in ABGR
        /// format (each byte represents a color channel).
        /// ABGR is the reversed version of the default format.
        /// </summary>
        /// <returns>A <see langword="int"/> representing this color in ABGR32 format.</returns>
        public int ToAbgr32()
        {
            int c = (byte)Math.Round(a * 255);
            c <<= 8;
            c |= (byte)Math.Round(b * 255);
            c <<= 8;
            c |= (byte)Math.Round(g * 255);
            c <<= 8;
            c |= (byte)Math.Round(r * 255);

            return c;
        }

        /// <summary>
        /// Returns the color converted to a 64-bit integer in ABGR
        /// format (each word represents a color channel).
        /// ABGR is the reversed version of the default format.
        /// </summary>
        /// <returns>A <see langword="long"/> representing this color in ABGR64 format.</returns>
        public long ToAbgr64()
        {
            long c = (ushort)Math.Round(a * 65535);
            c <<= 16;
            c |= (ushort)Math.Round(b * 65535);
            c <<= 16;
            c |= (ushort)Math.Round(g * 65535);
            c <<= 16;
            c |= (ushort)Math.Round(r * 65535);

            return c;
        }

        /// <summary>
        /// Returns the color converted to a 32-bit integer in ARGB
        /// format (each byte represents a color channel).
        /// ARGB is more compatible with DirectX, but not used much in Godot.
        /// </summary>
        /// <returns>A <see langword="int"/> representing this color in ARGB32 format.</returns>
        public int ToArgb32()
        {
            int c = (byte)Math.Round(a * 255);
            c <<= 8;
            c |= (byte)Math.Round(r * 255);
            c <<= 8;
            c |= (byte)Math.Round(g * 255);
            c <<= 8;
            c |= (byte)Math.Round(b * 255);

            return c;
        }

        /// <summary>
        /// Returns the color converted to a 64-bit integer in ARGB
        /// format (each word represents a color channel).
        /// ARGB is more compatible with DirectX, but not used much in Godot.
        /// </summary>
        /// <returns>A <see langword="long"/> representing this color in ARGB64 format.</returns>
        public long ToArgb64()
        {
            long c = (ushort)Math.Round(a * 65535);
            c <<= 16;
            c |= (ushort)Math.Round(r * 65535);
            c <<= 16;
            c |= (ushort)Math.Round(g * 65535);
            c <<= 16;
            c |= (ushort)Math.Round(b * 65535);

            return c;
        }

        /// <summary>
        /// Returns the color converted to a 32-bit integer in RGBA
        /// format (each byte represents a color channel).
        /// RGBA is Godot's default and recommended format.
        /// </summary>
        /// <returns>A <see langword="int"/> representing this color in RGBA32 format.</returns>
        public int ToRgba32()
        {
            int c = (byte)Math.Round(r * 255);
            c <<= 8;
            c |= (byte)Math.Round(g * 255);
            c <<= 8;
            c |= (byte)Math.Round(b * 255);
            c <<= 8;
            c |= (byte)Math.Round(a * 255);

            return c;
        }

        /// <summary>
        /// Returns the color converted to a 64-bit integer in RGBA
        /// format (each word represents a color channel).
        /// RGBA is Godot's default and recommended format.
        /// </summary>
        /// <returns>A <see langword="long"/> representing this color in RGBA64 format.</returns>
        public long ToRgba64()
        {
            long c = (ushort)Math.Round(r * 65535);
            c <<= 16;
            c |= (ushort)Math.Round(g * 65535);
            c <<= 16;
            c |= (ushort)Math.Round(b * 65535);
            c <<= 16;
            c |= (ushort)Math.Round(a * 65535);

            return c;
        }

        /// <summary>
        /// Returns the color's HTML hexadecimal color string in RGBA format.
        /// </summary>
        /// <param name="includeAlpha">
        /// Whether or not to include alpha. If <see langword="false"/>, the color is RGB instead of RGBA.
        /// </param>
        /// <returns>A string for the HTML hexadecimal representation of this color.</returns>
        public string ToHtml(bool includeAlpha = true)
        {
            var txt = string.Empty;

            txt += ToHex32(r);
            txt += ToHex32(g);
            txt += ToHex32(b);

            if (includeAlpha)
                txt = ToHex32(a) + txt;

            return txt;
        }

        /// <summary>
        /// Constructs a <see cref="Color"/> from RGBA values, typically on the range of 0 to 1.
        /// </summary>
        /// <param name="r">The color's red component, typically on the range of 0 to 1.</param>
        /// <param name="g">The color's green component, typically on the range of 0 to 1.</param>
        /// <param name="b">The color's blue component, typically on the range of 0 to 1.</param>
        /// <param name="a">The color's alpha (transparency) value, typically on the range of 0 to 1. Default: 1.</param>
        public Color(float r, float g, float b, float a = 1.0f)
        {
            this.r = r;
            this.g = g;
            this.b = b;
            this.a = a;
        }

        /// <summary>
        /// Constructs a <see cref="Color"/> from an existing color and an alpha value.
        /// </summary>
        /// <param name="c">The color to construct from. Only its RGB values are used.</param>
        /// <param name="a">The color's alpha (transparency) value, typically on the range of 0 to 1. Default: 1.</param>
        public Color(Color c, float a = 1.0f)
        {
            r = c.r;
            g = c.g;
            b = c.b;
            this.a = a;
        }

        /// <summary>
        /// Constructs a <see cref="Color"/> from a 32-bit integer in RGBA format
        /// (each byte represents a color channel).
        /// </summary>
        /// <param name="rgba">The <see langword="int"/> representing the color.</param>
        public Color(int rgba)
        {
            a = (rgba & 0xFF) / 255.0f;
            rgba >>= 8;
            b = (rgba & 0xFF) / 255.0f;
            rgba >>= 8;
            g = (rgba & 0xFF) / 255.0f;
            rgba >>= 8;
            r = (rgba & 0xFF) / 255.0f;
        }

        /// <summary>
        /// Constructs a <see cref="Color"/> from a 64-bit integer in RGBA format
        /// (each word represents a color channel).
        /// </summary>
        /// <param name="rgba">The <see langword="long"/> representing the color.</param>
        public Color(long rgba)
        {
            a = (rgba & 0xFFFF) / 65535.0f;
            rgba >>= 16;
            b = (rgba & 0xFFFF) / 65535.0f;
            rgba >>= 16;
            g = (rgba & 0xFFFF) / 65535.0f;
            rgba >>= 16;
            r = (rgba & 0xFFFF) / 65535.0f;
        }

        private static int ParseCol8(string str, int ofs)
        {
            int ig = 0;

            for (int i = 0; i < 2; i++)
            {
                int c = str[i + ofs];
                int v;

                if (c >= '0' && c <= '9')
                {
                    v = c - '0';
                }
                else if (c >= 'a' && c <= 'f')
                {
                    v = c - 'a';
                    v += 10;
                }
                else if (c >= 'A' && c <= 'F')
                {
                    v = c - 'A';
                    v += 10;
                }
                else
                {
                    return -1;
                }

                if (i == 0)
                {
                    ig += v * 16;
                }
                else
                {
                    ig += v;
                }
            }

            return ig;
        }

        private string ToHex32(float val)
        {
            byte b = (byte)Mathf.RoundToInt(Mathf.Clamp(val * 255, 0, 255));
            return b.HexEncode();
        }

        internal static bool HtmlIsValid(string color)
        {
            if (color.Length == 0)
            {
                return false;
            }

            if (color[0] == '#')
            {
                color = color.Substring(1, color.Length - 1);
            }

            bool alpha;

            switch (color.Length)
            {
                case 8:
                    alpha = true;
                    break;
                case 6:
                    alpha = false;
                    break;
                default:
                    return false;
            }

            if (alpha)
            {
                if (ParseCol8(color, 0) < 0)
                {
                    return false;
                }
            }

            int from = alpha ? 2 : 0;

            if (ParseCol8(color, from + 0) < 0)
                return false;
            if (ParseCol8(color, from + 2) < 0)
                return false;
            if (ParseCol8(color, from + 4) < 0)
                return false;

            return true;
        }

        /// <summary>
        /// Returns a color constructed from integer red, green, blue, and alpha channels.
        /// Each channel should have 8 bits of information ranging from 0 to 255.
        /// </summary>
        /// <param name="r8">The red component represented on the range of 0 to 255.</param>
        /// <param name="g8">The green component represented on the range of 0 to 255.</param>
        /// <param name="b8">The blue component represented on the range of 0 to 255.</param>
        /// <param name="a8">The alpha (transparency) component represented on the range of 0 to 255.</param>
        /// <returns>The constructed color.</returns>
        public static Color Color8(byte r8, byte g8, byte b8, byte a8 = 255)
        {
            return new Color(r8 / 255f, g8 / 255f, b8 / 255f, a8 / 255f);
        }

        /// <summary>
        /// Constructs a <see cref="Color"/> from the HTML hexadecimal color string in RGBA format.
        /// </summary>
        /// <param name="rgba">A string for the HTML hexadecimal representation of this color.</param>
        /// <exception name="ArgumentOutOfRangeException">
        /// Thrown when the given <paramref name="rgba"/> color code is invalid.
        /// </exception>
        public Color(string rgba)
        {
            if (rgba.Length == 0)
            {
                r = 0f;
                g = 0f;
                b = 0f;
                a = 1.0f;
                return;
            }

            if (rgba[0] == '#')
                rgba = rgba.Substring(1);

            bool alpha;

            if (rgba.Length == 8)
            {
                alpha = true;
            }
            else if (rgba.Length == 6)
            {
                alpha = false;
            }
            else
            {
                throw new ArgumentOutOfRangeException("Invalid color code. Length is " + rgba.Length + " but a length of 6 or 8 is expected: " + rgba);
            }

            if (alpha)
            {
                a = ParseCol8(rgba, 0) / 255f;

                if (a < 0)
                    throw new ArgumentOutOfRangeException("Invalid color code. Alpha part is not valid hexadecimal: " + rgba);
            }
            else
            {
                a = 1.0f;
            }

            int from = alpha ? 2 : 0;

            r = ParseCol8(rgba, from + 0) / 255f;

            if (r < 0)
                throw new ArgumentOutOfRangeException("Invalid color code. Red part is not valid hexadecimal: " + rgba);

            g = ParseCol8(rgba, from + 2) / 255f;

            if (g < 0)
                throw new ArgumentOutOfRangeException("Invalid color code. Green part is not valid hexadecimal: " + rgba);

            b = ParseCol8(rgba, from + 4) / 255f;

            if (b < 0)
                throw new ArgumentOutOfRangeException("Invalid color code. Blue part is not valid hexadecimal: " + rgba);
        }

        /// <summary>
        /// Adds each component of the <see cref="Color"/>
        /// with the components of the given <see cref="Color"/>.
        /// </summary>
        /// <param name="left">The left color.</param>
        /// <param name="right">The right color.</param>
        /// <returns>The added color.</returns>
        public static Color operator +(Color left, Color right)
        {
            left.r += right.r;
            left.g += right.g;
            left.b += right.b;
            left.a += right.a;
            return left;
        }

        /// <summary>
        /// Subtracts each component of the <see cref="Color"/>
        /// by the components of the given <see cref="Color"/>.
        /// </summary>
        /// <param name="left">The left color.</param>
        /// <param name="right">The right color.</param>
        /// <returns>The subtracted color.</returns>
        public static Color operator -(Color left, Color right)
        {
            left.r -= right.r;
            left.g -= right.g;
            left.b -= right.b;
            left.a -= right.a;
            return left;
        }

        /// <summary>
        /// Inverts the given color. This is equivalent to
        /// <c>Colors.White - c</c> or
        /// <c>new Color(1 - c.r, 1 - c.g, 1 - c.b, 1 - c.a)</c>.
        /// </summary>
        /// <param name="color">The color to invert.</param>
        /// <returns>The inverted color</returns>
        public static Color operator -(Color color)
        {
            return Colors.White - color;
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Color"/>
        /// by the given <see langword="float"/>.
        /// </summary>
        /// <param name="color">The color to multiply.</param>
        /// <param name="scale">The value to multiply by.</param>
        /// <returns>The multiplied color.</returns>
        public static Color operator *(Color color, float scale)
        {
            color.r *= scale;
            color.g *= scale;
            color.b *= scale;
            color.a *= scale;
            return color;
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Color"/>
        /// by the given <see langword="float"/>.
        /// </summary>
        /// <param name="scale">The value to multiply by.</param>
        /// <param name="color">The color to multiply.</param>
        /// <returns>The multiplied color.</returns>
        public static Color operator *(float scale, Color color)
        {
            color.r *= scale;
            color.g *= scale;
            color.b *= scale;
            color.a *= scale;
            return color;
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Color"/>
        /// by the components of the given <see cref="Color"/>.
        /// </summary>
        /// <param name="left">The left color.</param>
        /// <param name="right">The right color.</param>
        /// <returns>The multiplied color.</returns>
        public static Color operator *(Color left, Color right)
        {
            left.r *= right.r;
            left.g *= right.g;
            left.b *= right.b;
            left.a *= right.a;
            return left;
        }

        /// <summary>
        /// Divides each component of the <see cref="Color"/>
        /// by the given <see langword="float"/>.
        /// </summary>
        /// <param name="color">The dividend vector.</param>
        /// <param name="scale">The divisor value.</param>
        /// <returns>The divided color.</returns>
        public static Color operator /(Color color, float scale)
        {
            color.r /= scale;
            color.g /= scale;
            color.b /= scale;
            color.a /= scale;
            return color;
        }

        /// <summary>
        /// Divides each component of the <see cref="Color"/>
        /// by the components of the given <see cref="Color"/>.
        /// </summary>
        /// <param name="left">The dividend color.</param>
        /// <param name="right">The divisor color.</param>
        /// <returns>The divided color.</returns>
        public static Color operator /(Color left, Color right)
        {
            left.r /= right.r;
            left.g /= right.g;
            left.b /= right.b;
            left.a /= right.a;
            return left;
        }

        /// <summary>
        /// Returns <see langword="true"/> if the colors are exactly equal.
        /// Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="left">The left color.</param>
        /// <param name="right">The right color.</param>
        /// <returns>Whether or not the colors are equal.</returns>
        public static bool operator ==(Color left, Color right)
        {
            return left.Equals(right);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the colors are not equal.
        /// Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="left">The left color.</param>
        /// <param name="right">The right color.</param>
        /// <returns>Whether or not the colors are equal.</returns>
        public static bool operator !=(Color left, Color right)
        {
            return !left.Equals(right);
        }

        /// <summary>
        /// Compares two <see cref="Color"/>s by first checking if
        /// the red value of the <paramref name="left"/> color is less than
        /// the red value of the <paramref name="right"/> color.
        /// If the red values are exactly equal, then it repeats this check
        /// with the green values of the two colors, then with the blue values,
        /// and then with the alpha value.
        /// This operator is useful for sorting colors.
        /// </summary>
        /// <param name="left">The left color.</param>
        /// <param name="right">The right color.</param>
        /// <returns>Whether or not the left is less than the right.</returns>
        public static bool operator <(Color left, Color right)
        {
            if (Mathf.IsEqualApprox(left.r, right.r))
            {
                if (Mathf.IsEqualApprox(left.g, right.g))
                {
                    if (Mathf.IsEqualApprox(left.b, right.b))
                    {
                        return left.a < right.a;
                    }
                    return left.b < right.b;
                }
                return left.g < right.g;
            }
            return left.r < right.r;
        }

        /// <summary>
        /// Compares two <see cref="Color"/>s by first checking if
        /// the red value of the <paramref name="left"/> color is greater than
        /// the red value of the <paramref name="right"/> color.
        /// If the red values are exactly equal, then it repeats this check
        /// with the green values of the two colors, then with the blue values,
        /// and then with the alpha value.
        /// This operator is useful for sorting colors.
        /// </summary>
        /// <param name="left">The left color.</param>
        /// <param name="right">The right color.</param>
        /// <returns>Whether or not the left is greater than the right.</returns>
        public static bool operator >(Color left, Color right)
        {
            if (Mathf.IsEqualApprox(left.r, right.r))
            {
                if (Mathf.IsEqualApprox(left.g, right.g))
                {
                    if (Mathf.IsEqualApprox(left.b, right.b))
                    {
                        return left.a > right.a;
                    }
                    return left.b > right.b;
                }
                return left.g > right.g;
            }
            return left.r > right.r;
        }

        /// <summary>
        /// Compares two <see cref="Color"/>s by first checking if
        /// the red value of the <paramref name="left"/> color is less than
        /// or equal to the red value of the <paramref name="right"/> color.
        /// If the red values are exactly equal, then it repeats this check
        /// with the green values of the two colors, then with the blue values,
        /// and then with the alpha value.
        /// This operator is useful for sorting colors.
        /// </summary>
        /// <param name="left">The left color.</param>
        /// <param name="right">The right color.</param>
        /// <returns>Whether or not the left is less than or equal to the right.</returns>
        public static bool operator <=(Color left, Color right)
        {
            if (left.r == right.r)
            {
                if (left.g == right.g)
                {
                    if (left.b == right.b)
                    {
                        return left.a <= right.a;
                    }
                    return left.b < right.b;
                }
                return left.g < right.g;
            }
            return left.r < right.r;
        }

        /// <summary>
        /// Compares two <see cref="Color"/>s by first checking if
        /// the red value of the <paramref name="left"/> color is greater than
        /// or equal to the red value of the <paramref name="right"/> color.
        /// If the red values are exactly equal, then it repeats this check
        /// with the green values of the two colors, then with the blue values,
        /// and then with the alpha value.
        /// This operator is useful for sorting colors.
        /// </summary>
        /// <param name="left">The left color.</param>
        /// <param name="right">The right color.</param>
        /// <returns>Whether or not the left is greater than or equal to the right.</returns>
        public static bool operator >=(Color left, Color right)
        {
            if (left.r == right.r)
            {
                if (left.g == right.g)
                {
                    if (left.b == right.b)
                    {
                        return left.a >= right.a;
                    }
                    return left.b > right.b;
                }
                return left.g > right.g;
            }
            return left.r > right.r;
        }

        /// <summary>
        /// Returns <see langword="true"/> if this color and <paramref name="obj"/> are equal.
        /// </summary>
        /// <param name="obj">The other object to compare.</param>
        /// <returns>Whether or not the color and the other object are equal.</returns>
        public override bool Equals(object obj)
        {
            if (obj is Color)
            {
                return Equals((Color)obj);
            }

            return false;
        }

        /// <summary>
        /// Returns <see langword="true"/> if the colors are exactly equal.
        /// Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="other">The other color.</param>
        /// <returns>Whether or not the colors are equal.</returns>
        public bool Equals(Color other)
        {
            return r == other.r && g == other.g && b == other.b && a == other.a;
        }

        /// <summary>
        /// Returns <see langword="true"/> if this color and <paramref name="other"/> are approximately equal,
        /// by running <see cref="Mathf.IsEqualApprox(float, float)"/> on each component.
        /// </summary>
        /// <param name="other">The other color to compare.</param>
        /// <returns>Whether or not the colors are approximately equal.</returns>
        public bool IsEqualApprox(Color other)
        {
            return Mathf.IsEqualApprox(r, other.r) && Mathf.IsEqualApprox(g, other.g) && Mathf.IsEqualApprox(b, other.b) && Mathf.IsEqualApprox(a, other.a);
        }

        /// <summary>
        /// Serves as the hash function for <see cref="Color"/>.
        /// </summary>
        /// <returns>A hash code for this color.</returns>
        public override int GetHashCode()
        {
            return r.GetHashCode() ^ g.GetHashCode() ^ b.GetHashCode() ^ a.GetHashCode();
        }

        /// <summary>
        /// Converts this <see cref="Color"/> to a string.
        /// </summary>
        /// <returns>A string representation of this color.</returns>
        public override string ToString()
        {
            return String.Format("{0},{1},{2},{3}", r.ToString(), g.ToString(), b.ToString(), a.ToString());
        }

        /// <summary>
        /// Converts this <see cref="Color"/> to a string with the given <paramref name="format"/>.
        /// </summary>
        /// <returns>A string representation of this color.</returns>
        public string ToString(string format)
        {
            return String.Format("{0},{1},{2},{3}", r.ToString(format), g.ToString(format), b.ToString(format), a.ToString(format));
        }
    }
}
