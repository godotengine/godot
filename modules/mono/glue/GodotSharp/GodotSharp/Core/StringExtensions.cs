using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Security;
using System.Text;
using System.Text.RegularExpressions;

namespace Godot
{
    /// <summary>
    /// 操作字符串的扩展方法。
    /// </summary>
    public static class StringExtensions
    {
        private static int GetSliceCount(this string instance, string splitter)
        {
            if (instance.Empty() || splitter.Empty())
                return 0;

            int pos = 0;
            int slices = 1;

            while ((pos = instance.Find(splitter, pos, caseSensitive: true)) >= 0)
            {
                slices++;
                pos += splitter.Length;
            }

            return slices;
        }

        private static string GetSliceCharacter(this string instance, char splitter, int slice)
        {
            if (!instance.Empty() && slice >= 0)
            {
                int i = 0;
                int prev = 0;
                int count = 0;

                while (true)
                {
                    bool end = instance.Length <= i;

                    if (end || instance[i] == splitter)
                    {
                        if (slice == count)
                        {
                            return instance.Substring(prev, i - prev);
                        }
                        else if (end)
                        {
                            return string.Empty;
                        }

                        count++;
                        prev = i + 1;
                    }

                    i++;
                }
            }

            return string.Empty;
        }

        /// <summary>
        /// 如果字符串是文件路径，则返回不带扩展名的文件路径。
        /// </summary>
        /// <seealso cref="Extension(string)"/>
        /// <seealso cref="GetBaseDir(string)"/>
        /// <seealso cref="GetFile(string)"/>
        /// <param name="instance">文件路径</param>
        /// <returns>不带扩展名的文件路径。</returns>
        public static string BaseName(this string instance)
        {
            int index = instance.LastIndexOf('.');

            if (index > 0)
                return instance.Substring(0, index);

            return instance;
        }

        /// <summary>
        /// 如果字符串开始，则返回 <see langword="true"/>
        /// 使用给定的字符串 <paramref name="text"/>。
        /// </summary>
        /// <param name="instance">要检查的字符串。</param>
        /// <param name="text">开始字符串。</param>
        /// <returns>如果字符串以给定字符串开头。</returns>
        public static bool BeginsWith(this string instance, string text)
        {
            return instance.StartsWith(text);
        }

        /// <summary>
        /// 返回此字符串的二元组（连续字母对）。
        /// </summary>
        /// <param name="instance">将使用的字符串。</param>
        /// <returns>这个字符串的二元组。</returns>
        public static string[] Bigrams(this string instance)
        {
            string[] b = new string[instance.Length - 1];

            for (int i = 0; i < b.Length; i++)
            {
                b[i] = instance.Substring(i, 2);
            }

            return b;
        }

        /// <summary>
        /// 将包含二进制数的字符串转换为整数。
        /// 二进制字符串可以以 <c>0b</c> 为前缀，也可以不加前缀，
        /// 并且它们也可以在可选前缀之前以 <c>-</c> 开头。
        /// </summary>
        /// <param name="instance">要转换的字符串。</param>
        /// <returns>转换后的字符串。</returns>
        public static int BinToInt(this string instance)
        {
            if (instance.Length == 0)
            {
                return 0;
            }

            int sign = 1;

            if (instance[0] == '-')
            {
                sign = -1;
                instance = instance.Substring(1);
            }

            if (instance.StartsWith("0b"))
            {
                instance = instance.Substring(2);
            }

            return sign * Convert.ToInt32(instance, 2);
        }

        /// <summary>
        /// 返回字符串中子字符串 <paramref name="what"/> 的数量。
        /// </summary>
        /// <param name="instance">要搜索子字符串的字符串。</param>
        /// <param name="what">将被计算的子字符串。</param>
        /// <param name="caseSensitive">如果搜索区分大小写。</param>
        /// <param name="from">开始搜索的索引。</param>
        /// <param name="to">停止搜索的索引。</param>
        /// <returns>字符串中子串的数量。</returns>
        public static int Count(this string instance, string what, bool caseSensitive = true, int from = 0, int to = 0)
        {
            if (what.Length == 0)
            {
                return 0;
            }

            int len = instance.Length;
            int slen = what.Length;

            if (len < slen)
            {
                return 0;
            }

            string str;

            if (from >= 0 && to >= 0)
            {
                if (to == 0)
                {
                    to = len;
                }
                else if (from >= to)
                {
                    return 0;
                }
                if (from == 0 && to == len)
                {
                    str = instance;
                }
                else
                {
                    str = instance.Substring(from, to - from);
                }
            }
            else
            {
                return 0;
            }

            int c = 0;
            int idx;

            do
            {
                idx = str.IndexOf(what, caseSensitive ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase);
                if (idx != -1)
                {
                    str = str.Substring(idx + slen);
                    ++c;
                }
            } while (idx != -1);

            return c;
        }

        /// <summary>
        /// 返回使用 C 语言标准转义的特殊字符的字符串副本。
        /// </summary>
        /// <param name="instance">要转义的字符串。</param>
        /// <returns>转义的字符串。</returns>
        public static string CEscape(this string instance)
        {
            var sb = new StringBuilder(string.Copy(instance));

            sb.Replace("\\", "\\\\");
            sb.Replace("\a", "\\a");
            sb.Replace("\b", "\\b");
            sb.Replace("\f", "\\f");
            sb.Replace("\n", "\\n");
            sb.Replace("\r", "\\r");
            sb.Replace("\t", "\\t");
            sb.Replace("\v", "\\v");
            sb.Replace("\'", "\\'");
            sb.Replace("\"", "\\\"");
            sb.Replace("?", "\\?");

            return sb.ToString();
        }

        /// <summary>
        /// 返回字符串的副本，其中转义字符替换为它们的含义
        /// 根据 C 语言标准。
        /// </summary>
        /// <param name="instance">要取消转义的字符串。</param>
        /// <returns>未转义的字符串。</returns>
        public static string CUnescape(this string instance)
        {
            var sb = new StringBuilder(string.Copy(instance));

            sb.Replace("\\a", "\a");
            sb.Replace("\\b", "\b");
            sb.Replace("\\f", "\f");
            sb.Replace("\\n", "\n");
            sb.Replace("\\r", "\r");
            sb.Replace("\\t", "\t");
            sb.Replace("\\v", "\v");
            sb.Replace("\\'", "\'");
            sb.Replace("\\\"", "\"");
            sb.Replace("\\?", "?");
            sb.Replace("\\\\", "\\");

            return sb.ToString();
        }

        /// <summary>
        /// 改变一些字母的大小写。 用空格替换下划线，转换所有字母
        /// 小写然后大写第一个和空格字符后面的每个字母。
        /// 对于 <c>capitalize camelCase mixed_with_underscores</c> 它将返回
        /// <c>大写 Camelcase 与下划线混合</c>.
        /// </summary>
        /// <param name="instance">要大写的字符串。</param>
        /// <returns>大写字符串。</returns>
        public static string Capitalize(this string instance)
        {
            string aux = instance.CamelcaseToUnderscore(true).Replace("_", " ").Trim();
            string cap = string.Empty;

            for (int i = 0; i < aux.GetSliceCount(" "); i++)
            {
                string slice = aux.GetSliceCharacter(' ', i);
                if (slice.Length > 0)
                {
                    slice = char.ToUpper(slice[0]) + slice.Substring(1);
                    if (i > 0)
                        cap += " ";
                    cap += slice;
                }
            }

            return cap;
        }

        private static string CamelcaseToUnderscore(this string instance, bool lowerCase)
        {
            string newString = string.Empty;
            int startIndex = 0;

            for (int i = 1; i < instance.Length; i++)
            {
                bool isUpper = char.IsUpper(instance[i]);
                bool isNumber = char.IsDigit(instance[i]);

                bool areNext2Lower = false;
                bool isNextLower = false;
                bool isNextNumber = false;
                bool wasPrecedentUpper = char.IsUpper(instance[i - 1]);
                bool wasPrecedentNumber = char.IsDigit(instance[i - 1]);

                if (i + 2 < instance.Length)
                {
                    areNext2Lower = char.IsLower(instance[i + 1]) && char.IsLower(instance[i + 2]);
                }

                if (i + 1 < instance.Length)
                {
                    isNextLower = char.IsLower(instance[i + 1]);
                    isNextNumber = char.IsDigit(instance[i + 1]);
                }

                bool condA = isUpper && !wasPrecedentUpper && !wasPrecedentNumber;
                bool condB = wasPrecedentUpper && isUpper && areNext2Lower;
                bool condC = isNumber && !wasPrecedentNumber;
                bool canBreakNumberLetter = isNumber && !wasPrecedentNumber && isNextLower;
                bool canBreakLetterNumber = !isNumber && wasPrecedentNumber && (isNextLower || isNextNumber);

                bool shouldSplit = condA || condB || condC || canBreakNumberLetter || canBreakLetterNumber;
                if (shouldSplit)
                {
                    newString += instance.Substring(startIndex, i - startIndex) + "_";
                    startIndex = i;
                }
            }

            newString += instance.Substring(startIndex, instance.Length - startIndex);
            return lowerCase ? newString.ToLower() : newString;
        }

        /// <summary>
        /// 与另一个字符串进行区分大小写的比较，如果小于则返回-1，如果相等则返回0，如果大于则+1。
        /// </summary>
        /// <seealso cref="NocasecmpTo(string, string)"/>
        /// <seealso cref="CompareTo(string, string, bool)"/>
        /// <param name="instance">要比较的字符串。</param>
        /// <param name="to">要比较的另一个字符串。</param>
        /// <returns>-1 如果小于，0 如果相等，+1 如果大于。</returns>
        public static int CasecmpTo(this string instance, string to)
        {
            return instance.CompareTo(to, caseSensitive: true);
        }

        /// <summary>
        /// 与另一个字符串进行比较，小于返回-1，相等返回0，大于返回+1。
        /// </summary>
        /// <param name="instance">要比较的字符串。</param>
        /// <param name="to">要比较的另一个字符串。</param>
        /// <param name="caseSensitive">
        /// 如果<see langword="true"/>，比较区分大小写。
        /// </param>
        /// <returns>-1 如果小于，0 如果相等，+1 如果大于。</returns>
        public static int CompareTo(this string instance, string to, bool caseSensitive = true)
        {
            if (instance.Empty())
                return to.Empty() ? 0 : -1;

            if (to.Empty())
                return 1;

            int instanceIndex = 0;
            int toIndex = 0;

            if (caseSensitive) // Outside while loop to avoid checking multiple times, despite some code duplication.
            {
                while (true)
                {
                    if (to[toIndex] == 0 && instance[instanceIndex] == 0)
                        return 0; // We're equal
                    if (instance[instanceIndex] == 0)
                        return -1; // If this is empty, and the other one is not, then we're less... I think?
                    if (to[toIndex] == 0)
                        return 1; // Otherwise the other one is smaller...
                    if (instance[instanceIndex] < to[toIndex]) // More than
                        return -1;
                    if (instance[instanceIndex] > to[toIndex]) // Less than
                        return 1;

                    instanceIndex++;
                    toIndex++;
                }
            }
            else
            {
                while (true)
                {
                    if (to[toIndex] == 0 && instance[instanceIndex] == 0)
                        return 0; // We're equal
                    if (instance[instanceIndex] == 0)
                        return -1; // If this is empty, and the other one is not, then we're less... I think?
                    if (to[toIndex] == 0)
                        return 1; // Otherwise the other one is smaller..
                    if (char.ToUpper(instance[instanceIndex]) < char.ToUpper(to[toIndex])) // More than
                        return -1;
                    if (char.ToUpper(instance[instanceIndex]) > char.ToUpper(to[toIndex])) // Less than
                        return 1;

                    instanceIndex++;
                    toIndex++;
                }
            }
        }

        /// <summary>
        /// 如果字符串为空，则返回 <see langword="true"/>。
        /// </summary>
        public static bool Empty(this string instance)
        {
            return string.IsNullOrEmpty(instance);
        }

        /// <summary>
        /// 如果字符串结束，则返回 <see langword="true"/>
        /// 使用给定的字符串 <paramref name="text"/>。
        /// </summary>
        /// <param name="instance">要检查的字符串。</param>
        /// <param name="text">结束字符串。</param>
        /// <returns>如果字符串以给定的字符串结尾。</returns>
        public static bool EndsWith(this string instance, string text)
        {
            return instance.EndsWith(text);
        }

        /// <summary>
        /// 从 <paramref name="pos"/> 开始的字符串中删除 <paramref name="chars"/> 字符。
        /// </summary>
        /// <param name="instance">要修改的字符串。</param>
        /// <param name="pos">擦除的起始位置。</param>
        /// <param name="chars">要擦除的字符数。</param>
        public static void Erase(this StringBuilder instance, int pos, int chars)
        {
            instance.Remove(pos, chars);
        }

        /// <summary>
        /// 返回没有前导句点字符的扩展名 (<c>.</c>)
        /// 如果字符串是有效的文件名或路径。 如果字符串不包含
        /// 一个扩展，而是返回一个空字符串。
        /// </summary>
        /// <example>
        /// <code>
        /// GD.Print("/path/to/file.txt".GetExtension())  // "txt"
        /// GD.Print("file.txt".GetExtension())  // "txt"
        /// GD.Print("file.sample.txt".GetExtension())  // "txt"
        /// GD.Print(".txt".GetExtension())  // "txt"
        /// GD.Print("file.txt.".GetExtension())  // "" (empty string)
        /// GD.Print("file.txt..".GetExtension())  // "" (empty string)
        /// GD.Print("txt".GetExtension())  // "" (empty string)
        /// GD.Print("".GetExtension())  // "" (empty string)
        /// </code>
        /// </example>
        /// <seealso cref="BaseName(string)"/>
        /// <seealso cref="GetBaseDir(string)"/>
        /// <seealso cref="GetFile(string)"/>
        /// <param name="instance">文件的路径.</param>
        /// <returns>文件的扩展名或空字符串。</returns>
        public static string Extension(this string instance)
        {
            int pos = instance.FindLast(".");

            if (pos < 0)
                return instance;

            return instance.Substring(pos + 1);
        }

        /// <summary>
        /// 查找第一次出现的子字符串。 可选地，可以传递搜索起始位置。
        /// </summary>
         /// <seealso cref="Find(string, char, int, bool)"/>
        /// <seealso cref="FindLast(string, string, bool)"/>
        /// <seealso cref="FindLast(string, string, int, bool)"/>
        /// <seealso cref="FindN(string, string, int)"/>
        /// <param name="instance">将被搜索的字符串。</param>
        /// <param name="what">要查找的子字符串。</param>
        /// <param name="from">搜索起始位置。</param>
        /// <param name="caseSensitive">如果<see langword="true"/>，搜索区分大小写。</param>
        /// <returns>子字符串的起始位置，如果没有找到则为-1。</returns>
        public static int Find(this string instance, string what, int from = 0, bool caseSensitive = true)
        {
            return instance.IndexOf(what, from, caseSensitive ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// 查找第一次出现的字符。 可选地，可以传递搜索起始位置。
        /// </summary>
        /// <seealso cref="Find(string, string, int, bool)"/>
        /// <seealso cref="FindLast(string, string, bool)"/>
        /// <seealso cref="FindLast(string, string, int, bool)"/>
        /// <seealso cref="FindN(string, string, int)"/>
        /// <param name="instance">将被搜索的字符串。</param>
        /// <param name="what">要查找的子字符串。</param>
        /// <param name="from">搜索起始位置。</param>
        /// <param name="caseSensitive">如果<see langword="true"/>，搜索区分大小写。</param>
        /// <returns>char的第一个实例，如果没有找到则为-1。</returns>
        public static int Find(this string instance, char what, int from = 0, bool caseSensitive = true)
        {
            // TODO: Could be more efficient if we get a char version of `IndexOf`.
            // See https://github.com/dotnet/runtime/issues/44116
            return instance.IndexOf(what.ToString(), from, caseSensitive ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>查找子字符串的最后一次出现.</summary>
        /// <seealso cref="Find(string, string, int, bool)"/>
        /// <seealso cref="Find(string, char, int, bool)"/>
        /// <seealso cref="FindLast(string, string, int, bool)"/>
        /// <seealso cref="FindN(string, string, int)"/>
        /// <param name="instance">将被搜索的字符串。</param>
        /// <param name="what">要查找的子字符串。</param>
        /// <param name="caseSensitive">如果<see langword="true"/>，搜索区分大小写。</param>
        /// <returns>子字符串的起始位置，如果没有找到则为-1。</returns>
        public static int FindLast(this string instance, string what, bool caseSensitive = true)
        {
            return instance.FindLast(what, instance.Length - 1, caseSensitive);
        }

        /// <summary>查找指定搜索起始位置的子字符串的最后一次出现.</summary>
        /// <seealso cref="Find(string, string, int, bool)"/>
        /// <seealso cref="Find(string, char, int, bool)"/>
        /// <seealso cref="FindLast(string, string, bool)"/>
        /// <seealso cref="FindN(string, string, int)"/>
        /// <param name="instance">将被搜索的字符串。</param>
        /// <param name="what">要查找的子字符串。</param>
        /// <param name="from">搜索起始位置。</param>
        /// <param name="caseSensitive">如果<see langword="true"/>，搜索区分大小写。</param>
        /// <returns>子字符串的起始位置，如果没有找到则为-1。</returns>
        public static int FindLast(this string instance, string what, int from, bool caseSensitive = true)
        {
            return instance.LastIndexOf(what, from, caseSensitive ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// 查找第一次出现的子字符串，但不区分大小写。
        /// 可选地，可以传递搜索起始位置。
        /// </summary>
        /// <seealso cref="Find(string, string, int, bool)"/>
        /// <seealso cref="Find(string, char, int, bool)"/>
        /// <seealso cref="FindLast(string, string, bool)"/>
        /// <seealso cref="FindLast(string, string, int, bool)"/>
        /// <param name="instance">将被搜索的字符串。</param>
        /// <param name="what">要查找的子字符串。</param>
        /// <param name="from">搜索起始位置。</param>
        /// <returns>子字符串的起始位置，如果没有找到则为-1。</returns>
        public static int FindN(this string instance, string what, int from = 0)
        {
            return instance.IndexOf(what, from, StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// 如果字符串是文件路径，则返回基目录。
        /// </summary>
        /// <seealso cref="BaseName(string)"/>
        /// <seealso cref="Extension(string)"/>
        /// <seealso cref="GetFile(string)"/>
        /// <param name="instance">文件路径</param>
        /// <returns>基本目录。</returns>
        public static string GetBaseDir(this string instance)
        {
            int basepos = instance.Find("://");

            string rs;
            string directory = string.Empty;

            if (basepos != -1)
            {
                int end = basepos + 3;
                rs = instance.Substring(end);
                directory = instance.Substring(0, end);
            }
            else
            {
                if (instance.BeginsWith("/"))
                {
                    rs = instance.Substring(1);
                    directory = "/";
                }
                else
                {
                    rs = instance;
                }
            }

            int sep = Mathf.Max(rs.FindLast("/"), rs.FindLast("\\"));

            if (sep == -1)
                return directory;

            return directory + rs.Substr(0, sep);
        }

        /// <summary>
        /// 如果字符串是文件的路径，则返回文件并忽略基目录。
        /// </summary>
        /// <seealso cref="BaseName(string)"/>
        /// <seealso cref="Extension(string)"/>
        /// <seealso cref="GetBaseDir(string)"/>
        /// <param name="instance">文件路径</param>
        /// <returns>文件名。</returns>
        public static string GetFile(this string instance)
        {
            int sep = Mathf.Max(instance.FindLast("/"), instance.FindLast("\\"));

            if (sep == -1)
                return instance;

            return instance.Substring(sep + 1);
        }

        /// <summary>
        /// 将给定的 ASCII 编码文本字节数组转换为字符串。
        /// <see cref="GetStringFromUTF8"/> 如果
        /// 内容仅为 ASCII。 与 UTF-8 函数不同，此函数
        /// 将每个字节映射到数组中的一个字符。 多字节序列
        /// 将无法正确解释。 始终解析用户输入
        /// 使用 <see cref="GetStringFromUTF8"/>。
        /// </summary>
        /// <param name="bytes">ASCII 字符的字节数组（范围为 0-127）。</param>
        /// <returns>从字节创建的字符串。</returns>
        public static string GetStringFromASCII(this byte[] bytes)
        {
            return Encoding.ASCII.GetString(bytes);
        }

        /// <summary>
        /// 将给定的 UTF-8 编码文本字节数组转换为字符串。
        /// 比 <see cref="GetStringFromASCII"/> 慢，但支持 UTF-8
        /// 编码数据。 如果您不确定
        /// 数据来源。 对于用户输入此功能
        /// 应该始终是首选。
        /// </summary>
        /// <param name="bytes">一个UTF-8字符的字节数组（一个字符可能占用多个字节）。</param>
        /// <returns>从字节创建的字符串。</returns>
        public static string GetStringFromUTF8(this byte[] bytes)
        {
            return Encoding.UTF8.GetString(bytes);
        }

        /// <summary>
        /// 散列字符串并返回一个 32 位无符号整数。
        /// </summary>
        /// <param name="instance">要散列的字符串。</param>
        /// <returns>计算得到的字符串哈希值。</returns>
        public static uint Hash(this string instance)
        {
            uint hash = 5381;

            foreach (uint c in instance)
            {
                hash = (hash << 5) + hash + c; // hash * 33 + c
            }

            return hash;
        }

        /// <summary>
        /// 以字符串形式返回此字节的十六进制表示。
        /// </summary>
        /// <param name="b">要编码的字节。</param>
        /// <returns>这个字节的十六进制表示。</returns>
        internal static string HexEncode(this byte b)
        {
            string ret = string.Empty;

            for (int i = 0; i < 2; i++)
            {
                char c;
                int lv = b & 0xF;

                if (lv < 10)
                {
                    c = (char)('0' + lv);
                }
                else
                {
                    c = (char)('a' + lv - 10);
                }

                b >>= 4;
                ret = c + ret;
            }

            return ret;
        }

        /// <summary>
        /// 以字符串形式返回此字节数组的十六进制表示。
        /// </summary>
        /// <param name="bytes">要编码的字节数组。</param>
        /// <returns>这个字节数组的十六进制表示。</returns>
        public static string HexEncode(this byte[] bytes)
        {
            string ret = string.Empty;

            foreach (byte b in bytes)
            {
                ret += b.HexEncode();
            }

            return ret;
        }

        /// <summary>
        /// 将包含十六进制数的字符串转换为整数。
        /// 十六进制字符串可以加前缀<c>0x</c>，也可以不加前缀，
        /// 并且它们也可以在可选前缀之前以 <c>-</c> 开头。
        /// </summary>
        /// <param name="instance">要转换的字符串。</param>
        /// <returns>转换后的字符串。</returns>
        public static int HexToInt(this string instance)
        {
            if (instance.Length == 0)
            {
                return 0;
            }

            int sign = 1;

            if (instance[0] == '-')
            {
                sign = -1;
                instance = instance.Substring(1);
            }

            if (instance.StartsWith("0x"))
            {
                instance = instance.Substring(2);
            }

            return sign * int.Parse(instance, NumberStyles.HexNumber);
        }

        /// <summary>
        /// 在给定位置插入子字符串。
        /// </summary>
        /// <param name="instance">要修改的字符串。</param>
        /// <param name="pos">插入子串的位置。</param>
        /// <param name="what">要插入的子字符串</param>
        /// <returns>
        /// 在给定位置插入 <paramref name="what"/> 的字符串
        /// 位置 <paramref name="pos"/>.
        /// </returns>
        public static string Insert(this string instance, int pos, string what)
        {
            return instance.Insert(pos, what);
        }

        /// <summary>
        /// 如果字符串是文件的路径，则返回 <see langword="true"/>
        /// 目录和它的起点是明确定义的。 这包括
        /// <c>res://</c>、<c>user://</c>、<c>C:\</c>、<c>/</c>等
        /// </summary>
        /// <seealso cref="IsRelPath(string)"/>
        /// <param name="instance">要检查的字符串。</param>
        /// <returns>如果字符串是绝对路径。</returns>
        public static bool IsAbsPath(this string instance)
        {
            if (string.IsNullOrEmpty(instance))
                return false;
            else if (instance.Length > 1)
                return instance[0] == '/' || instance[0] == '\\' || instance.Contains(":/") || instance.Contains(":\\");
            else
                return instance[0] == '/' || instance[0] == '\\';
        }

        /// <summary>
        /// 如果字符串是文件的路径，则返回 <see langword="true"/>
        /// 目录及其起点在
        /// 正在使用的上下文。 起点可参考当前
        /// 目录 (<c>./</c>)，或当前 <see cref="Node"/>。
        /// </summary>
        /// <seealso cref="IsAbsPath(string)"/>
        /// <param name="instance">要检查的字符串。</param>
        /// <returns>如果字符串是相对路径。</returns>
        public static bool IsRelPath(this string instance)
        {
            return !IsAbsPath(instance);
        }

        /// <summary>
        /// 检查这个字符串是否是给定字符串的子序列。
        /// </summary>
        /// <seealso cref="IsSubsequenceOfI(string, string)"/>
        /// <param name="instance">要搜索的子序列。</param>
        /// <param name="text">包含子序列的字符串。</param>
        /// <param name="caseSensitive">如果<see langword="true"/>，检查区分大小写。</param>
        /// <returns>如果字符串是给定字符串的子序列。</returns>
        public static bool IsSubsequenceOf(this string instance, string text, bool caseSensitive = true)
        {
            int len = instance.Length;

            if (len == 0)
                return true; // Technically an empty string is subsequence of any string

            if (len > text.Length)
                return false;

            int source = 0;
            int target = 0;

            while (source < len && target < text.Length)
            {
                bool match;

                if (!caseSensitive)
                {
                    char sourcec = char.ToLower(instance[source]);
                    char targetc = char.ToLower(text[target]);
                    match = sourcec == targetc;
                }
                else
                {
                    match = instance[source] == text[target];
                }
                if (match)
                {
                    source++;
                    if (source >= len)
                        return true;
                }

                target++;
            }

            return false;
        }

        /// <summary>
        /// 检查这个字符串是否是给定字符串的子序列，忽略大小写差异。
        /// </summary>
        /// <seealso cref="IsSubsequenceOf(string, string, bool)"/>
        /// <param name="instance">要搜索的子序列。</param>
        /// <param name="text">包含子序列的字符串。</param>
        /// <returns>如果字符串是给定字符串的子序列。</returns>
        public static bool IsSubsequenceOfI(this string instance, string text)
        {
            return instance.IsSubsequenceOf(text, caseSensitive: false);
        }

        /// <summary>
        /// 检查字符串是否包含有效的<see langword="float"/>。
        /// </summary>
        /// <param name="instance">要检查的字符串。</param>
        /// <returns>如果字符串包含一个有效的浮点数。</returns>
        public static bool IsValidFloat(this string instance)
        {
            float f;
            return float.TryParse(instance, out f);
        }

        /// <summary>
        /// 检查字符串是否包含 HTML 表示法中的有效颜色。
        /// </summary>
        /// <param name="instance">要检查的字符串。</param>
        /// <returns>如果字符串包含有效的 HTML 颜色。</returns>
        public static bool IsValidHtmlColor(this string instance)
        {
            return Color.HtmlIsValid(instance);
        }

        /// <summary>
        /// 检查字符串是否为有效标识符。 正如在
        /// 编程语言，一个有效的标识符只能包含字母，
        /// 数字和下划线 (_) 并且第一个字符可能不是数字。
        /// </summary>
        /// <param name="instance">要检查的字符串。</param>
        /// <returns>如果字符串包含一个有效的标识符。</returns>
        public static bool IsValidIdentifier(this string instance)
        {
            int len = instance.Length;

            if (len == 0)
                return false;

            for (int i = 0; i < len; i++)
            {
                if (i == 0)
                {
                    if (instance[0] >= '0' && instance[0] <= '9')
                        return false; // Don't start with number plz
                }

                bool validChar = instance[i] == '_' ||
                    (instance[i] >= 'a' && instance[i] <= 'z') ||
                    (instance[i] >= 'A' && instance[i] <= 'Z') ||
                    (instance[i] >= '0' && instance[i] <= '9');

                if (!validChar)
                    return false;
            }

            return true;
        }

        /// <summary>
        /// 检查字符串是否包含有效整数。
        /// </summary>
        /// <param name="instance">要检查的字符串。</param>
        /// <returns>如果字符串包含一个有效的整数。</returns>
        public static bool IsValidInteger(this string instance)
        {
            int f;
            return int.TryParse(instance, out f);
        }

        /// <summary>
        /// 检查字符串是否包含有效的 IP 地址。
        /// </summary>
        /// <param name="instance">要检查的字符串。</param>
        /// <returns>如果字符串包含有效的IP地址。</returns>
        public static bool IsValidIPAddress(this string instance)
        {
            // TODO: Support IPv6 addresses
            string[] ip = instance.Split(".");

            if (ip.Length != 4)
                return false;

            for (int i = 0; i < ip.Length; i++)
            {
                string n = ip[i];
                if (!n.IsValidInteger())
                    return false;

                int val = n.ToInt();
                if (val < 0 || val > 255)
                    return false;
            }

            return true;
        }

        /// <summary>
        /// 返回使用 JSON 标准转义的特殊字符的字符串副本。
        /// </summary>
        /// <param name="instance">要转义的字符串。</param>
        /// <returns>转义的字符串。</returns>
        public static string JSONEscape(this string instance)
        {
            var sb = new StringBuilder(string.Copy(instance));

            sb.Replace("\\", "\\\\");
            sb.Replace("\b", "\\b");
            sb.Replace("\f", "\\f");
            sb.Replace("\n", "\\n");
            sb.Replace("\r", "\\r");
            sb.Replace("\t", "\\t");
            sb.Replace("\v", "\\v");
            sb.Replace("\"", "\\\"");

            return sb.ToString();
        }

        /// <summary>
        /// 返回字符串左边的字符数。
        /// </summary>
        /// <seealso cref="Right(string, int)"/>
        /// <param name="instance">原始字符串。</param>
        /// <param name="pos">字符串中左边结束的位置。</param>
        /// <returns>从给定位置开始的字符串左侧。</returns>
        public static string Left(this string instance, int pos)
        {
            if (pos <= 0)
                return string.Empty;

            if (pos >= instance.Length)
                return instance;

            return instance.Substring(0, pos);
        }

        /// <summary>
        /// 返回字符串的长度（以字符为单位）。
        /// </summary>
        /// <param name="instance">要检查的字符串。</param>
        /// <returns>字符串的长度。</returns>
        public static int Length(this string instance)
        {
            return instance.Length;
        }

        /// <summary>
        /// 返回字符串的副本，其中删除了左侧的字符。
        /// </summary>
        /// <seealso cref="RStrip(string, string)"/>
        /// <param name="instance">要从中删除字符的字符串。</param>
        /// <param name="chars">要删除的字符。</param>
        /// <returns>从左边删除字符的字符串副本。</returns>
        public static string LStrip(this string instance, string chars)
        {
            int len = instance.Length;
            int beg;

            for (beg = 0; beg < len; beg++)
            {
                if (chars.Find(instance[beg]) == -1)
                {
                    break;
                }
            }

            if (beg == 0)
            {
                return instance;
            }

            return instance.Substr(beg, len - beg);
        }

        /// <summary>
        /// 做一个简单的表达式匹配，其中 '*' 匹配零个或多个
        /// 任意字符和'?' 匹配除 '.' 之外的任何单个字符。
        /// </summary>
        /// <param name="instance">要检查的字符串。</param>
        /// <param name="expr">要检查的表达式。</param>
        /// <param name="caseSensitive">
        /// 如果 <see langword="true"/>，检查将区分大小写。
        /// </参数>
        /// <returns>如果表达式有任何匹配项。</returns>
        private static bool ExprMatch(this string instance, string expr, bool caseSensitive)
        {
            // case '\0':
            if (expr.Length == 0)
                return instance.Length == 0;

            switch (expr[0])
            {
                case '*':
                    return ExprMatch(instance, expr.Substring(1), caseSensitive) || (instance.Length > 0 && ExprMatch(instance.Substring(1), expr, caseSensitive));
                case '?':
                    return instance.Length > 0 && instance[0] != '.' && ExprMatch(instance.Substring(1), expr.Substring(1), caseSensitive);
                default:
                    if (instance.Length == 0)
                        return false;
                    if (caseSensitive)
                        return instance[0] == expr[0];
                    return (char.ToUpper(instance[0]) == char.ToUpper(expr[0])) && ExprMatch(instance.Substring(1), expr.Substring(1), caseSensitive);
            }
        }

        /// <summary>
        /// 做一个简单的区分大小写的表达式匹配，使用 ? 和 * 通配符
        ///（参见 <see cref="ExprMatch(string, string, bool)"/>）。
        /// </summary>
        /// <seealso cref="MatchN(string, string)"/>
        /// <param name="instance">要检查的字符串。</param>
        /// <param name="expr">要检查的表达式。</param>
        /// <param name="caseSensitive">
        /// 如果 <see langword="true"/>，检查将区分大小写。
        /// </param>
        /// <returns>如果表达式有任何匹配项。</returns>
        public static bool Match(this string instance, string expr, bool caseSensitive = true)
        {
            if (instance.Length == 0 || expr.Length == 0)
                return false;

            return instance.ExprMatch(expr, caseSensitive);
        }

        /// <summary>
        /// 做一个简单的不区分大小写的表达式匹配，使用 ? 和 * 通配符
        ///（参见 <see cref="ExprMatch(string, string, bool)"/>）。
        /// </summary>
        /// <seealso cref="Match(string, string, bool)"/>
        /// <param name="instance">要检查的字符串。</param>
        /// <param name="expr">要检查的表达式。</param>
        /// <returns>如果表达式有任何匹配项。</returns>
        public static bool MatchN(this string instance, string expr)
        {
            if (instance.Length == 0 || expr.Length == 0)
                return false;

            return instance.ExprMatch(expr, caseSensitive: false);
        }

        /// <summary>
        /// 以字节数组的形式返回字符串的 MD5 哈希值。
        /// </summary>
        /// <seealso cref="MD5Text(string)"/>
        /// <param name="instance">The string to hash.</param>
        /// <returns>The MD5 hash of the string.</returns>
        public static byte[] MD5Buffer(this string instance)
        {
            return godot_icall_String_md5_buffer(instance);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern byte[] godot_icall_String_md5_buffer(string str);

        /// <summary>
        /// 将字符串的 MD5 哈希作为字符串返回。
        /// </summary>
        /// <seealso cref="MD5Buffer(string)"/>
        /// <param name="instance">要散列的字符串。</param>
        /// <returns>字符串的 MD5 哈希值。</returns>
        public static string MD5Text(this string instance)
        {
            return godot_icall_String_md5_text(instance);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern string godot_icall_String_md5_text(string str);

        /// <summary>
        /// 与另一个字符串进行不区分大小写的比较，如果小于则返回-1，如果相等则返回0，如果大于则+1。
        /// </summary>
        /// <seealso cref="CasecmpTo(string, string)"/>
        /// <seealso cref="CompareTo(string, string, bool)"/>
        /// <param name="instance">要比较的字符串。</param>
        /// <param name="to">要比较的另一个字符串。</param>
        /// <returns>-1 如果小于，0 如果相等，+1 如果大于。</returns>
        public static int NocasecmpTo(this string instance, string to)
        {
            return instance.CompareTo(to, caseSensitive: false);
        }

        /// <summary>
        /// 返回位置 <paramref name="at"/> 处的字符代码。
        /// </summary>
        /// <param name="instance">要检查的字符串。</param>
        /// <param name="at">要检查的字符在字符串中的位置。</param>
        /// <returns>字符代码。</returns>
        public static int OrdAt(this string instance, int at)
        {
            return instance[at];
        }

        /// <summary>
        /// 将数字格式化为具有确切数字的 <paramref name="digits"/>
        /// 小数点后。
        /// </summary>
        /// <seealso cref="PadZeros(string, int)"/>
        /// <param name="instance">要填充的字符串。</param>
        /// <param name="digits">小数点后的位数。</param>
        /// <returns>用零填充的字符串。</returns>
        public static string PadDecimals(this string instance, int digits)
        {
            int c = instance.Find(".");

            if (c == -1)
            {
                if (digits <= 0)
                    return instance;

                instance += ".";
                c = instance.Length - 1;
            }
            else
            {
                if (digits <= 0)
                    return instance.Substring(0, c);
            }

            if (instance.Length - (c + 1) > digits)
            {
                instance = instance.Substring(0, c + digits + 1);
            }
            else
            {
                while (instance.Length - (c + 1) < digits)
                {
                    instance += "0";
                }
            }

            return instance;
        }

        /// <summary>
        /// 将数字格式化为具有确切数字的 <paramref name="digits"/>
        /// 小数点前。
        /// </summary>
        /// <seealso cref="PadDecimals(string, int)"/>
        /// <param name="instance">要填充的字符串。</param>
        /// <param name="digits">小数点前的位数。</param>
        /// <returns>用零填充的字符串。</returns>
        public static string PadZeros(this string instance, int digits)
        {
            string s = instance;
            int end = s.Find(".");

            if (end == -1)
                end = s.Length;

            if (end == 0)
                return s;

            int begin = 0;

            while (begin < end && (s[begin] < '0' || s[begin] > '9'))
            {
                begin++;
            }

            if (begin >= end)
                return s;

            while (end - begin < digits)
            {
                s = s.Insert(begin, "0");
                end++;
            }

            return s;
        }

        /// <summary>
        /// 解码百分比编码的字符串。 请参阅 <see cref="PercentEncode"/>。
        /// </summary>
        public static string PercentDecode(this string instance)
        {
            return Uri.UnescapeDataString(instance);
        }

        /// <summary>
        /// 对字符串进行百分比编码。 这是为了对 URL 中的参数进行编码
        /// 发送 HTTP GET 请求和 form-urlencoded POST 请求的主体时。
        /// </summary>
        /// <seealso cref="PercentDecode(string)"/>
        public static string PercentEncode(this string instance)
        {
            return Uri.EscapeDataString(instance);
        }

        /// <summary>
        /// 如果字符串是路径，则连接 <paramref name="file"/>
        /// 在字符串的末尾作为子路径。
        /// 例如 <c>"this/is".PlusFile("path") == "this/is/path"</c>.
        /// </summary>
        /// <param name="instance">将被连接的路径。</param>
        /// <param name="file">要与路径连接的文件名。</param>
        /// <returns>给定文件名的连接路径。</returns>
        public static string PlusFile(this string instance, string file)
        {
            if (instance.Length > 0 && instance[instance.Length - 1] == '/')
                return instance + file;
            return instance + "/" + file;
        }

        /// <summary>
        /// 为字符串中不同的子字符串替换出现的子字符串。
        /// </summary>
        /// <seealso cref="ReplaceN(string, string, string)"/>
        /// <param name="instance">要修改的字符串。</param>
        /// <param name="what">字符串中要替换的子字符串。</param>
        /// <param name="forwhat">替换<paramref name="what"/>的子字符串。</param>
        /// <returns>替换子串的字符串。</returns>
        public static string Replace(this string instance, string what, string forwhat)
        {
            return instance.Replace(what, forwhat);
        }

        /// <summary>
        /// 用字符串中不同的子字符串替换出现的子字符串，但不区分大小写。
        /// </summary>
        /// <seealso cref="Replace(string, string, string)"/>
        /// <param name="instance">要修改的字符串。</param>
        /// <param name="what">字符串中要替换的子字符串。</param>
        /// <param name="forwhat">替换<paramref name="what"/>的子字符串。</param>
        /// <returns>替换子串的字符串。</returns>
        public static string ReplaceN(this string instance, string what, string forwhat)
        {
            return Regex.Replace(instance, what, forwhat, RegexOptions.IgnoreCase);
        }

        /// <summary>
        /// 执行子字符串的搜索，但是从字符串的结尾而不是开头开始。
        /// </summary>
        /// <seealso cref="RFindN(string, string, int)"/>
        /// <param name="instance">将被搜索的字符串。</param>
        /// <param name="what">要在字符串中搜索的子字符串。</param>
        /// <param name="from">开始搜索的位置。</param>
        /// <returns>找到子字符串的位置，如果没有找到，则返回-1。</returns>
        public static int RFind(this string instance, string what, int from = -1)
        {
            return godot_icall_String_rfind(instance, what, from);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern int godot_icall_String_rfind(string str, string what, int from);

        /// <summary>
        /// 执行子字符串的搜索，但是从字符串的结尾而不是开头开始。
        /// 也搜索不区分大小写。
        /// </summary>
        /// <seealso cref="RFind(string, string, int)"/>
        /// <param name="instance">将被搜索的字符串。</param>
        /// <param name="what">要在字符串中搜索的子字符串。</param>
        /// <param name="from">开始搜索的位置。</param>
        /// <returns>找到子字符串的位置，如果没有找到，则返回-1。</returns>
        public static int RFindN(this string instance, string what, int from = -1)
        {
            return godot_icall_String_rfindn(instance, what, from);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern int godot_icall_String_rfindn(string str, string what, int from);

        /// <summary>
        /// 从给定位置返回字符串的右侧。
        /// </summary>
        /// <seealso cref="Left(string, int)"/>
        /// <param name="instance">原始字符串。</param>
        /// <param name="pos">字符串中右边开始的位置。</param>
        /// <returns>从给定位置开始的字符串的右侧。</returns>
        public static string Right(this string instance, int pos)
        {
            if (pos >= instance.Length)
                return instance;

            if (pos < 0)
                return string.Empty;

            return instance.Substring(pos, instance.Length - pos);
        }

        /// <summary>
        /// 返回字符串的副本，其中删除了右侧的字符。
        /// </summary>
        /// <seealso cref="LStrip(string, string)"/>
        /// <param name="instance">要从中删除字符的字符串。</param>
        /// <param name="chars">要删除的字符。</param>
        /// <returns>从右边删除字符的字符串的副本。</returns>
        public static string RStrip(this string instance, string chars)
        {
            int len = instance.Length;
            int end;

            for (end = len - 1; end >= 0; end--)
            {
                if (chars.Find(instance[end]) == -1)
                {
                    break;
                }
            }

            if (end == len - 1)
            {
                return instance;
            }

            return instance.Substr(0, end + 1);
        }

        /// <summary>
        /// 将字符串的 SHA-256 哈希作为字节数组返回。
        /// </summary>
        /// <seealso cref="SHA256Text(string)"/>
        /// <param name="instance">要散列的字符串。</param>
        /// <returns>字符串的 SHA-256 哈希值。</returns>
        public static byte[] SHA256Buffer(this string instance)
        {
            return godot_icall_String_sha256_buffer(instance);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern byte[] godot_icall_String_sha256_buffer(string str);

        /// <summary>
        /// 将字符串的 SHA-256 哈希作为字符串返回。
        /// </summary>
        /// <seealso cref="SHA256Buffer(string)"/>
        /// <param name="instance">要散列的字符串。</param>
        /// <returns>字符串的 SHA-256 哈希值。</returns>
        public static string SHA256Text(this string instance)
        {
            return godot_icall_String_sha256_text(instance);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern string godot_icall_String_sha256_text(string str);

        /// <summary>
        /// 返回文本与该字符串相比的相似度指数。
        /// 1 表示完全相似，0 表示完全不同。
        /// </summary>
        /// <param name="instance">要比较的字符串。</param>
        /// <param name="text">要比较的另一个字符串。</param>
        /// <returns>相似度指数。</returns>
        public static float Similarity(this string instance, string text)
        {
            if (instance == text)
            {
                // Equal strings are totally similar
                return 1.0f;
            }
            if (instance.Length < 2 || text.Length < 2)
            {
                // No way to calculate similarity without a single bigram
                return 0.0f;
            }

            string[] sourceBigrams = instance.Bigrams();
            string[] targetBigrams = text.Bigrams();

            int sourceSize = sourceBigrams.Length;
            int targetSize = targetBigrams.Length;

            float sum = sourceSize + targetSize;
            float inter = 0;

            for (int i = 0; i < sourceSize; i++)
            {
                for (int j = 0; j < targetSize; j++)
                {
                    if (sourceBigrams[i] == targetBigrams[j])
                    {
                        inter++;
                        break;
                    }
                }
            }

            return 2.0f * inter / sum;
        }

        /// <summary>
        ///返回简化的规范路径。
        /// </summary>
        public static string SimplifyPath(this string instance)
        {
            return godot_icall_String_simplify_path(instance);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern string godot_icall_String_simplify_path(string str);

        /// <summary>
        /// 将字符串除以除数字符串，返回子字符串数组。
        /// 示例 "One,Two,Three" 如果被 "," 分割，将返回 ["One","Two","Three"]。
        /// </summary>
        /// <seealso cref="SplitFloats(string, string, bool)"/>
        /// <param name="instance">要拆分的字符串。</param>
        /// <param name="divisor">分割字符串的除数字符串。</param>
        /// <param name="allowEmpty">
        /// 如果 <see langword="true"/>，数组可能包含空字符串。
        /// </参数>
        /// <returns>从字符串中拆分出来的字符串数组。</returns>
        public static string[] Split(this string instance, string divisor, bool allowEmpty = true)
        {
            return instance.Split(new[] { divisor }, allowEmpty ? StringSplitOptions.None : StringSplitOptions.RemoveEmptyEntries);
        }

        /// <summary>
        /// 使用除数字符串将字符串拆分为浮点数，返回子字符串数组。
        /// 示例 "1,2.5,3" 如果被 "," 分割，将返回 [1,2.5,3]。
        /// </summary>
        /// <seealso cref="Split(string, string, bool)"/>
        /// <param name="instance">要拆分的字符串。</param>
        /// <param name="divisor">分割字符串的除数字符串。</param>
        /// <param name="allowEmpty">
        /// 如果 <see langword="true"/>，数组可能包含空浮点数。
        /// </参数>
        /// <returns>从字符串中拆分出来的浮点数数组。</returns>
        public static float[] SplitFloats(this string instance, string divisor, bool allowEmpty = true)
        {
            var ret = new List<float>();
            int from = 0;
            int len = instance.Length;

            while (true)
            {
                int end = instance.Find(divisor, from, caseSensitive: true);
                if (end < 0)
                    end = len;
                if (allowEmpty || end > from)
                    ret.Add(float.Parse(instance.Substring(from)));
                if (end == len)
                    break;

                from = end + divisor.Length;
            }

            return ret.ToArray();
        }

        private static readonly char[] _nonPrintable =
        {
            (char)00, (char)01, (char)02, (char)03, (char)04, (char)05,
            (char)06, (char)07, (char)08, (char)09, (char)10, (char)11,
            (char)12, (char)13, (char)14, (char)15, (char)16, (char)17,
            (char)18, (char)19, (char)20, (char)21, (char)22, (char)23,
            (char)24, (char)25, (char)26, (char)27, (char)28, (char)29,
            (char)30, (char)31, (char)32
        };

        /// <summary>
        /// 返回在开头和结尾去除了任何不可打印字符的字符串的副本。
        /// 可选参数分别用于切换左右边缘的剥离。
        /// </summary>
        /// <param name="instance">要剥离的字符串。</param>
        /// <param name="left">如果左边应该被剥离。</param>
        /// <param name="right">如果右边应该被剥离。</param>
        /// <returns>去掉所有不可打印字符的字符串。</returns>
        public static string StripEdges(this string instance, bool left = true, bool right = true)
        {
            if (left)
            {
                if (right)
                    return instance.Trim(_nonPrintable);
                return instance.TrimStart(_nonPrintable);
            }

            return instance.TrimEnd(_nonPrintable);
        }

        /// <summary>
        ///从 <paramref name="from"/> 位置返回字符串的一部分，长度为 <paramref name="len"/>。
        /// </summary>
        /// <param name="instance">要切片的字符串。</param>
        /// <param name="from">字符串中部分开始的位置。</param>
        /// <param name="len">返回部分的长度。</param>
        /// <returns>
        /// 从位置 <paramref name="from"/> 开始的部分字符串，长度为 <paramref name="len"/>。
        /// </returns>
        public static string Substr(this string instance, int from, int len)
        {
            int max = instance.Length - from;
            return instance.Substring(from, len > max ? max : len);
        }

        /// <summary>
        /// 将 String（它是一个字符数组）转换为 PoolByteArray（它是一个字节数组）。
        /// 与 <see cref="ToUTF8(string)"/> 相比，转换速度更快，假设
        /// String 包含的所有字符都只是 ASCII 字符。
        /// </summary>
        /// <seealso cref="ToUTF8(string)"/>
        /// <param name="instance">要转换的字符串。</param>
        /// <returns>作为 ASCII 编码字节的字符串。</returns>
        public static byte[] ToAscii(this string instance)
        {
            return Encoding.ASCII.GetBytes(instance);
        }

        /// <summary>
        /// 将包含十进制数的字符串转换为 <see langword="float" />。
        /// </summary>
        /// <seealso cref="ToInt(string)"/>
        /// <param name="instance">要转换的字符串。</param>
        /// <returns>字符串的数字表示。</returns>
        public static float ToFloat(this string instance)
        {
            return float.Parse(instance);
        }

        /// <summary>
        /// 将包含整数的字符串转换为 <see langword="int" />。
        /// </summary>
        /// <seealso cref="ToFloat(string)"/>
        /// <param name="instance">要转换的字符串。</param>
        /// <returns>字符串的数字表示。</returns>
        public static int ToInt(this string instance)
        {
            return int.Parse(instance);
        }

        /// <summary>
        /// 返回转换为小写的字符串。
        /// </summary>
        /// <seealso cref="ToUpper(string)"/>
        /// <param name="instance">要转换的字符串。</param>
        /// <returns>转换为小写的字符串。</returns>
        public static string ToLower(this string instance)
        {
            return instance.ToLower();
        }

        /// <summary>
        /// 返回转换为大写的字符串。
        /// </summary>
        /// <seealso cref="ToLower(string)"/>
        /// <param name="instance">要转换的字符串。</param>
        /// <returns>转换为大写的字符串。</returns>
        public static string ToUpper(this string instance)
        {
            return instance.ToUpper();
        }

        /// <summary>
        /// 将 String（字符数组）转换为 PoolByteArray（字节数组）。
        /// 转换比 <see cref="ToAscii(string)"/> 慢一点，但支持所有 UTF-8 字符。
        /// 因此，您应该更喜欢这个函数而不是 <see cref="ToAscii(string)"/>。
        /// </summary>
        /// <seealso cref="ToAscii(string)"/>
        /// <param name="instance">要转换的字符串。</param>
        /// <returns>字符串为 UTF-8 编码字节。</returns>
        public static byte[] ToUTF8(this string instance)
        {
            return Encoding.UTF8.GetBytes(instance);
        }

        /// <summary>
        /// 返回带有使用 XML 标准转义的特殊字符的字符串的副本。
        /// </summary>
        /// <seealso cref="XMLUnescape(string)"/>
        /// <param name="instance">要转义的字符串。</param>
        /// <returns>转义的字符串。</returns>
        public static string XMLEscape(this string instance)
        {
            return SecurityElement.Escape(instance);
        }

        /// <summary>
        /// 返回字符串的副本，其中转义字符替换为它们的含义
        /// 根据 XML 标准。
        /// </summary>
        /// <seealso cref="XMLEscape(string)"/>
        /// <param name="instance">要取消转义的字符串。</param>
        /// <returns>未转义的字符串。</returns>
        public static string XMLUnescape(this string instance)
        {
            return SecurityElement.FromString(instance).Text;
        }
    }
}
