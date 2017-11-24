using System;

namespace Godot
{
    public struct Color : IEquatable<Color>
    {
        public float r;
        public float g;
        public float b;
        public float a;

        public int r8
        {
            get
            {
                return (int)(r * 255.0f);
            }
        }

        public int g8
        {
            get
            {
                return (int)(g * 255.0f);
            }
        }

        public int b8
        {
            get
            {
                return (int)(b * 255.0f);
            }
        }

        public int a8
        {
            get
            {
                return (int)(a * 255.0f);
            }
        }

        public float h
        {
            get
            {
                float max = Mathf.Max(r, Mathf.Max(g, b));
                float min = Mathf.Min(r, Mathf.Min(g, b));

                float delta = max - min;

                if (delta == 0)
                    return 0;

                float h;

                if (r == max)
                    h = (g - b) / delta; // Between yellow & magenta
                else if (g == max)
                    h = 2 + (b - r) / delta; // Between cyan & yellow
                else
                    h = 4 + (r - g) / delta; // Between magenta & cyan

                h /= 6.0f;

                if (h < 0)
                    h += 1.0f;

                return h;
            }
            set
            {
                this = FromHsv(value, s, v);
            }
        }

        public float s
        {
            get
            {
                float max = Mathf.Max(r, Mathf.Max(g, b));
                float min = Mathf.Min(r, Mathf.Min(g, b));

                float delta = max - min;

                return max != 0 ? delta / max : 0;
            }
            set
            {
                this = FromHsv(h, value, v);
            }
        }

        public float v
        {
            get
            {
                return Mathf.Max(r, Mathf.Max(g, b));
            }
            set
            {
                this = FromHsv(h, s, value);
            }
        }

        private static readonly Color black = new Color(0f, 0f, 0f, 1.0f);

        public Color Black
        {
            get
            {
                return black;
            }
        }

        public float this [int index]
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

        public static void ToHsv(Color color, out float hue, out float saturation, out float value)
        {
            int max = Mathf.Max(color.r8, Mathf.Max(color.g8, color.b8));
            int min = Mathf.Min(color.r8, Mathf.Min(color.g8, color.b8));

            float delta = max - min;

            if (delta == 0)
            {
                hue = 0;
            }
            else
            {
                if (color.r == max)
                    hue = (color.g - color.b) / delta; // Between yellow & magenta
                else if (color.g == max)
                    hue = 2 + (color.b - color.r) / delta; // Between cyan & yellow
                else
                    hue = 4 + (color.r - color.g) / delta; // Between magenta & cyan

                hue /= 6.0f;

                if (hue < 0)
                    hue += 1.0f;
            }

            saturation = (max == 0) ? 0 : 1f - (1f * min / max);
            value = max / 255f;
        }

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

        public Color Blend(Color over)
        {
            Color res;

            float sa = 1.0f - over.a;
            res.a = a * sa + over.a;

            if (res.a == 0)
            {
                return new Color(0, 0, 0, 0);
            }
            else
            {
                res.r = (r * a * sa + over.r * over.a) / res.a;
                res.g = (g * a * sa + over.g * over.a) / res.a;
                res.b = (b * a * sa + over.b * over.a) / res.a;
            }

            return res;
        }

        public Color Contrasted()
        {
            return new Color(
                (r + 0.5f) % 1.0f,
                (g + 0.5f) % 1.0f,
                (b + 0.5f) % 1.0f
            );
        }

        public float Gray()
        {
            return (r + g + b) / 3.0f;
        }

        public Color Inverted()
        {
            return new Color(
                1.0f - r,
                1.0f - g,
                1.0f - b
            );
        }

        public Color LinearInterpolate(Color b, float t)
        {
            Color res = this;

            res.r += (t * (b.r - this.r));
            res.g += (t * (b.g - this.g));
            res.b += (t * (b.b - this.b));
            res.a += (t * (b.a - this.a));

            return res;
        }

        public int To32()
        {
            int c = (byte)(a * 255);
            c <<= 8;
            c |= (byte)(r * 255);
            c <<= 8;
            c |= (byte)(g * 255);
            c <<= 8;
            c |= (byte)(b * 255);

            return c;
        }

        public int ToArgb32()
        {
            int c = (byte)(a * 255);
            c <<= 8;
            c |= (byte)(r * 255);
            c <<= 8;
            c |= (byte)(g * 255);
            c <<= 8;
            c |= (byte)(b * 255);

            return c;
        }

        public string ToHtml(bool include_alpha = true)
        {
            String txt = string.Empty;

            txt += _to_hex(r);
            txt += _to_hex(g);
            txt += _to_hex(b);

            if (include_alpha)
                txt = _to_hex(a) + txt;

            return txt;
        }

        public Color(float r, float g, float b, float a = 1.0f)
        {
            this.r = r;
            this.g = g;
            this.b = b;
            this.a = a;
        }

        public Color(int rgba)
        {
            this.a = (rgba & 0xFF) / 255.0f;
            rgba >>= 8;
            this.b = (rgba & 0xFF) / 255.0f;
            rgba >>= 8;
            this.g = (rgba & 0xFF) / 255.0f;
            rgba >>= 8;
            this.r = (rgba & 0xFF) / 255.0f;
        }

        private static float _parse_col(string str, int ofs)
        {
            int ig = 0;

            for (int i = 0; i < 2; i++)
            {
                int c = str[i + ofs];
                int v = 0;

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
                    ig += v * 16;
                else
                    ig += v;
            }

            return ig;
        }

        private String _to_hex(float val)
        {
            int v = (int)Mathf.Clamp(val * 255.0f, 0, 255);

            string ret = string.Empty;

            for (int i = 0; i < 2; i++)
            {
                char[] c = { (char)0, (char)0 };
                int lv = v & 0xF;

                if (lv < 10)
                    c[0] = (char)('0' + lv);
                else
                    c[0] = (char)('a' + lv - 10);

                v >>= 4;
                ret = c + ret;
            }

            return ret;
        }

        internal static bool HtmlIsValid(string color)
        {
            if (color.Length == 0)
                return false;

            if (color[0] == '#')
                color = color.Substring(1, color.Length - 1);

            bool alpha = false;

            if (color.Length == 8)
                alpha = true;
            else if (color.Length == 6)
                alpha = false;
            else
                return false;

            if (alpha)
            {
                if ((int)_parse_col(color, 0) < 0)
                    return false;
            }

            int from = alpha ? 2 : 0;

            if ((int)_parse_col(color, from + 0) < 0)
                return false;
            if ((int)_parse_col(color, from + 2) < 0)
                return false;
            if ((int)_parse_col(color, from + 4) < 0)
                return false;

            return true;
        }

        public static Color Color8(byte r8, byte g8, byte b8, byte a8)
        {
            return new Color((float)r8 / 255f, (float)g8 / 255f, (float)b8 / 255f, (float)a8 / 255f);
        }

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

            bool alpha = false;

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
                a = _parse_col(rgba, 0);

                if (a < 0)
                    throw new ArgumentOutOfRangeException("Invalid color code. Alpha is " + a + " but zero or greater is expected: " + rgba);
            }
            else
            {
                a = 1.0f;
            }

            int from = alpha ? 2 : 0;

            r = _parse_col(rgba, from + 0);

            if (r < 0)
                throw new ArgumentOutOfRangeException("Invalid color code. Red is " + r + " but zero or greater is expected: " + rgba);

            g = _parse_col(rgba, from + 2);

            if (g < 0)
                throw new ArgumentOutOfRangeException("Invalid color code. Green is " + g + " but zero or greater is expected: " + rgba);

            b = _parse_col(rgba, from + 4);

            if (b < 0)
                throw new ArgumentOutOfRangeException("Invalid color code. Blue is " + b + " but zero or greater is expected: " + rgba);
        }

        public static bool operator ==(Color left, Color right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Color left, Color right)
        {
            return !left.Equals(right);
        }

        public static bool operator <(Color left, Color right)
        {
            if (left.r == right.r)
            {
                if (left.g == right.g)
                {
                    if (left.b == right.b)
                        return (left.a < right.a);
                    else
                        return (left.b < right.b);
                }
                else
                {
                    return left.g < right.g;
                }
            }

            return left.r < right.r;
        }

        public static bool operator >(Color left, Color right)
        {
            if (left.r == right.r)
            {
                if (left.g == right.g)
                {
                    if (left.b == right.b)
                        return (left.a > right.a);
                    else
                        return (left.b > right.b);
                }
                else
                {
                    return left.g > right.g;
                }
            }

            return left.r > right.r;
        }

        public override bool Equals(object obj)
        {
            if (obj is Color)
            {
                return Equals((Color)obj);
            }

            return false;
        }

        public bool Equals(Color other)
        {
            return r == other.r && g == other.g && b == other.b && a == other.a;
        }

        public override int GetHashCode()
        {
            return r.GetHashCode() ^ g.GetHashCode() ^ b.GetHashCode() ^ a.GetHashCode();
        }

        public override string ToString()
        {
            return String.Format("{0},{1},{2},{3}", new object[]
                {
                    this.r.ToString(),
                    this.g.ToString(),
                    this.b.ToString(),
                    this.a.ToString()
                });
        }

        public string ToString(string format)
        {
            return String.Format("{0},{1},{2},{3}", new object[]
                {
                    this.r.ToString(format),
                    this.g.ToString(format),
                    this.b.ToString(format),
                    this.a.ToString(format)
                });
        }
    }
}
