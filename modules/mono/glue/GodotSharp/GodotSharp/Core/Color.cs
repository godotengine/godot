using System;
using System.Runtime.InteropServices;

namespace Godot
{
    /// <summary>
    /// 由红色、绿色、蓝色和 alpha (RGBA) 分量表示的颜色。
    /// alpha分量常用于透明度。
    /// 值是浮点数，通常在 0 到 1 之间。
    /// 一些属性（例如<see cref="CanvasItem.Modulate"/>）可以接受值
    /// 大于 1（过亮或 HDR 颜色）。
    ///
    /// 如果你想提供 0 到 255 范围内的值，你应该使用
    /// <see cref="Color8"/> 和 <c>r8</c>/<c>g8</c>/<c>b8</c>/<c>a8</c> 属性 .
    /// </summary>
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Color : IEquatable<Color>
    {
        /// <summary>
        /// 颜色的红色分量，通常在 0 到 1 的范围内。
        /// </summary>
        public float r;

        /// <summary>
        /// 颜色的绿色分量，通常在 0 到 1 的范围内。
        /// </summary>
        public float g;

        /// <summary>
        /// 颜色的蓝色分量，通常在 0 到 1 的范围内。
        /// </summary>
        public float b;

        /// <summary>
        /// 颜色的 alpha（透明度）分量，通常在 0 到 1 的范围内。
        /// </summary>
        public float a;

        /// <summary>
        /// <see cref="r"/> 的包装器，它使用范围 0 到 255 而不是 0 到 1。
        /// </summary>
        /// <value>获取相当于乘以 255 并四舍五入。 设置相当于除以255.</value>
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
        /// <see cref="g"/> 的包装器，它使用范围 0 到 255 而不是 0 到 1。
        /// </summary>
        /// <value>获取相当于乘以 255 并四舍五入。 设置相当于除以255.</value>
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
        /// <see cref="b"/> 的包装器，它使用范围 0 到 255 而不是 0 到 1。
        /// </summary>
        /// <value>获取相当于乘以 255 并四舍五入。 设置相当于除以255.</value>
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
        /// <see cref="a"/> 的包装器，它使用范围 0 到 255 而不是 0 到 1。
        /// </summary>
        /// <value>获取相当于乘以 255 并四舍五入。 设置相当于除以255.</value>
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
        /// 此颜色的 HSV 色调，范围为 0 到 1。
        /// </summary>
        /// <value>获取是一个漫长的过程，详情请参考源码。 设置使用 <see cref="FromHsv"/>.</value>
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
        /// 此颜色的 HSV 饱和度，范围为 0 到 1.
        /// </summary>
        /// <value>获取相当于最小和最大 RGB 值之间的比率。 设置使用 <see cref="FromHsv"/>.</value>
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
        /// 此颜色的 HSV 值（亮度），范围为 0 到 1.
        /// </summary>
        /// <value>获取等效于在 RGB 组件上使用 <see cref="Math.Max(float, float)"/>。 设置使用 <see cref="FromHsv"/>.</value>
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
        /// 根据标准化名称返回颜色，带有
        /// 指定的alpha值。 支持的颜色名称与
        /// <see cref="Colors"/> 中定义的常量。
        /// </summary>
        /// <param name="name">颜色的名称.</param>
        /// <param name="alpha">在 0 到 1 的范围内表示的 alpha（透明度）分量。默认值：1.</param>
        /// <returns>构造的颜色.</returns>
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
        /// 使用它们的索引访问颜色组件。
        /// </summary>
        /// <value>
        /// <c>[0]</c> 相当于 <see cref="r"/>,
        /// <c>[1]</c> 相当于 <see cref="g"/>,
        /// <c>[2]</c> 相当于 <see cref="b"/>,
        /// <c>[3]</c> 相当于 <see cref="a"/>.
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
        /// 将颜色转换为 HSV 值。 这相当于使用每个
        /// <c>h</c>/<c>s</c>/<c>v</c> 属性，但效率更高。
        /// </summary>
        /// <param name="hue">HSV 色调的输出参数.</param>
        /// <param name="saturation">HSV 饱和的输出参数.</param>
        /// <param name="value">HSV 值的输出参数.</param>
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
        /// 从 HSV 配置文件构造颜色，其值位于
        /// 0到1的范围。这相当于使用每个
        /// <c>h</c>/<c>s</c>/<c>v</c> 属性，但效率更高。
        /// </summary>
        /// <param name="hue">HSV 色调，通常在 0 到 1 的范围内.</param>
        /// <param name="saturation">HSV 饱和度，通常在 0 到 1 的范围内.</param>
        /// <param name="value">HSV 值（亮度），通常在 0 到 1 的范围内.</param>
        /// <param name="alpha">alpha（透明度）值，通常在 0 到 1 的范围内.</param>
        /// <returns>构造的颜色.</returns>
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
        /// 返回一种新颜色，这种颜色是由这种颜色与另一种颜色混合而成的。
        /// 如果颜色是不透明的，结果也是不透明的。
        /// 第二种颜色可能有一个 alpha 值范围。
        /// </summary>
        /// <param name="over">要混合的颜色.</param>
        /// <returns>这种颜色混合了<paramref name="over"/>.</returns>
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
        /// 返回最具对比的颜色。
        /// </summary>
        /// <returns>最反差的颜色</returns>
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
        /// 返回由于使该颜色变暗而产生的新颜色
        /// 按指定的比率（在 0 到 1 的范围内）。
        /// </summary>
        /// <param name="amount">变暗的比例.</param>
        /// <returns>变暗的颜色.</returns>
        public Color Darkened(float amount)
        {
            Color res = this;
            res.r *= 1.0f - amount;
            res.g *= 1.0f - amount;
            res.b *= 1.0f - amount;
            return res;
        }

        /// <summary>
        /// 返回反转颜色：<c>(1 - r, 1 - g, 1 - b, a)</c>。
        /// </summary>
        /// <returns>反转的颜色.</returns>
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
        /// 返回由于使该颜色变亮而产生的新颜色
        /// 按指定的比率（在 0 到 1 的范围内）。
        /// </summary>
        /// <param name="amount">减仓比例.</param>
        /// <returns>变暗的颜色.</returns>
        public Color Lightened(float amount)
        {
            Color res = this;
            res.r += (1.0f - res.r) * amount;
            res.g += (1.0f - res.g) * amount;
            res.b += (1.0f - res.b) * amount;
            return res;
        }

        /// <summary>
        /// 返回之间的线性插值结果
        /// 这个颜色和 <paramref name="to"/> 的数量 <paramref name="weight"/>.
        /// </summary>
        /// <param name="to">插值的目标颜色.</param>
        /// <param name="weight">0.0 到 1.0 范围内的一个值，表示插值量.</param>
        /// <returns>插值的结果颜色.</returns>
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
        /// 返回之间的线性插值结果
        /// 这个颜色和 <paramref name="to"/> 按颜色量 <paramref name="weight"/>.
        /// </summary>
        /// <param name="to">插值的目标颜色.</param>
        /// <param name="weight">分量在 0.0 到 1.0 范围内的颜色，表示插值量.</param>
        /// <returns>插值的结果颜色.</returns>
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
        /// 返回在 ABGR 中转换为 32 位整数的颜色
        /// 格式（每个字节代表一个颜色通道）。
        /// ABGR 是默认格式的反转版本。
        /// </summary>
        /// <returns>A <see langword="int"/>以 ABGR32 格式表示此颜色.</returns>
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
        /// 返回在 ABGR 中转换为 64 位整数的颜色
        /// 格式（每个单词代表一个颜色通道）。
        /// ABGR 是默认格式的反转版本。
        /// </summary>
        /// <returns>A <see langword="long"/> 以 ABGR64 格式表示此颜色.</returns>
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
        /// 返回转换为 ARGB 中 32 位整数的颜色
        /// 格式（每个字节代表一个颜色通道）。
        /// ARGB 与 DirectX 更兼容，但在 Godot 中用得不多。
        /// </summary>
        /// <returns>A <see langword="int"/> 以 ARGB32 格式表示此颜色.</returns>
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
        /// 返回转换为 ARGB 中 64 位整数的颜色
        /// 格式（每个单词代表一个颜色通道）。
        /// ARGB 与 DirectX 更兼容，但在 Godot 中用得不多。
        /// </summary>
        /// <returns>A <see langword="long"/> 以 ARGB64 格式表示此颜色.</returns>
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
        /// 返回转换为 RGBA 中的 32 位整数的颜色
        /// 格式（每个字节代表一个颜色通道）。
        /// RGBA 是 Godot 的默认和推荐格式。
        /// </summary>
        /// <returns>A <see langword="int"/> 以 RGBA32 格式表示此颜色.</returns>
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
        /// 返回转换为 RGBA 中 64 位整数的颜色
        /// 格式（每个单词代表一个颜色通道）。
        /// RGBA 是 Godot 的默认和推荐格式。
        /// </summary>
        /// <returns>A <see langword="long"/> 以 RGBA64 格式表示此颜色.</returns>
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
        /// 以 RGBA 格式返回颜色的 HTML 十六进制颜色字符串。
        /// </summary>
        /// <param name="includeAlpha">
        /// 是否包含 alpha。 如果 <see langword="false"/>，颜色是 RGB 而不是 RGBA。
        /// </param>
        /// <returns>此颜色的 HTML 十六进制表示的字符串.</returns>
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
        /// 从 RGBA 值构造一个 <see cref="Color"/>，通常在 0 到 1 的范围内。
        /// </summary>
        /// <param name="r">颜色的红色分量，通常在 0 到 1 的范围内.</param>
        /// <param name="g">颜色的绿色分量，通常在 0 到 1 的范围内.</param>
        /// <param name="b">颜色的蓝色分量，通常在 0 到 1 的范围内.</param>
        /// <param name="a">颜色的 alpha（透明度）值，通常在 0 到 1 的范围内。默认值：1.</param>
        public Color(float r, float g, float b, float a = 1.0f)
        {
            this.r = r;
            this.g = g;
            this.b = b;
            this.a = a;
        }

        /// <summary>
        /// 从现有颜色和 alpha 值构造 <see cref="Color"/>。
        /// </summary>
        /// <param name="c">要构造的颜色。 仅使用其 RGB 值.</param>
        /// <param name="a">颜色的 alpha（透明度）值，通常在 0 到 1 的范围内。默认值：1.</param>
        public Color(Color c, float a = 1.0f)
        {
            r = c.r;
            g = c.g;
            b = c.b;
            this.a = a;
        }

        /// <summary>
        /// 从 RGBA 格式的 32 位整数构造 <see cref="Color"/>
        ///（每个字节代表一个颜色通道）。
        /// </summary>
        /// <param name="rgba">The <see langword="int"/> 代表颜色.</param>
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
        /// 从 RGBA 格式的 64 位整数构造 <see cref="Color"/>
        ///（每个单词代表一个颜色通道）。
        /// </summary>
        /// <param name="rgba"><see langword="long"/> 代表颜色.</param>
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
        /// 返回由整数红色、绿色、蓝色和 alpha 通道构成的颜色。
        /// 每个通道应该有 8 位信息，范围从 0 到 255。
        /// </summary>
        /// <param name="r8">红色分量表示在 0 到 255 的范围内.</param>
        /// <param name="g8">绿色分量表示在 0 到 255 的范围内.</param>
        /// <param name="b8">蓝色分量表示在 0 到 255 的范围内.</param>
        /// <param name="a8">在 0 到 255 范围内表示的 alpha（透明度）分量.</param>
        /// <returns>构造的颜色.</returns>
        public static Color Color8(byte r8, byte g8, byte b8, byte a8 = 255)
        {
            return new Color(r8 / 255f, g8 / 255f, b8 / 255f, a8 / 255f);
        }

        /// <summary>
        /// 从 RGBA 格式的 HTML 十六进制颜色字符串构造 <see cref="Color"/>。
        /// </summary>
        /// <param name="rgba">此颜色的 HTML 十六进制表示的字符串.</param>
        /// <exception name="ArgumentOutOfRangeException">
        /// 当给定的 <paramref name="rgba"/> 颜色代码无效时抛出。
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
        /// 如果此颜色和 <paramref name="obj"/> 相等，则返回 <see langword="true"/>。
        /// </summary>
        /// <param name="obj">另一个要比较的对象.</param>
        /// <returns>颜色与其他对象是否相等.</returns>
        public override bool Equals(object obj)
        {
            if (obj is Color)
            {
                return Equals((Color)obj);
            }

            return false;
        }

        /// <summary>

        /// 如果此颜色和 <paramref name="other"/> 相等，则返回 <see langword="true"/>
        /// </summary>
        /// <param name="other">要比较的其他颜色.</param>
        /// <returns>颜色是否相等.</returns>
        public bool Equals(Color other)
        {
            return r == other.r && g == other.g && b == other.b && a == other.a;
        }

        /// <summary>
        /// 如果此颜色和 <paramref name="other"/> 大致相等，则返回 <see langword="true"/>，
        /// by running <see cref="Mathf.IsEqualApprox(float, float)"/> on each component.
        /// </summary>
        /// <param name="other">要比较的其他颜色.</param>
        /// <returns>颜色是否大致相等.</returns>
        public bool IsEqualApprox(Color other)
        {
            return Mathf.IsEqualApprox(r, other.r) && Mathf.IsEqualApprox(g, other.g) && Mathf.IsEqualApprox(b, other.b) && Mathf.IsEqualApprox(a, other.a);
        }

        /// <summary>
        /// 用作 <see cref="Color"/> 的散列函数.
        /// </summary>
        /// <returns>此颜色的哈希码.</returns>
        public override int GetHashCode()
        {
            return r.GetHashCode() ^ g.GetHashCode() ^ b.GetHashCode() ^ a.GetHashCode();
        }

        /// <summary>
        /// 将此 <see cref="Color"/> 转换为字符串.
        /// </summary>
        /// <returns>此颜色的字符串表示.</returns>
        public override string ToString()
        {
            return String.Format("{0},{1},{2},{3}", r.ToString(), g.ToString(), b.ToString(), a.ToString());
        }

        /// <summary>
        /// 将此 <see cref="Color"/> 转换为具有给定 <paramref name="format"/> 的字符串.
        /// </summary>
        /// <returns>此颜色的字符串表示.</returns>
        public string ToString(string format)
        {
            return String.Format("{0},{1},{2},{3}", r.ToString(format), g.ToString(format), b.ToString(format), a.ToString(format));
        }
    }
}
