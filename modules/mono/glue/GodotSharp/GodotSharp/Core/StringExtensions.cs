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
    /// Extension methods to manipulate strings.
    /// </summary>
    public static class StringExtensions
    {
        private static int GetSliceCount(this string instance, string splitter)
        {
            if (string.IsNullOrEmpty(instance) || string.IsNullOrEmpty(splitter))
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
            if (!string.IsNullOrEmpty(instance) && slice >= 0)
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
        /// If the string is a path to a file, return the path to the file without the extension.
        /// </summary>
        /// <seealso cref="GetExtension(string)"/>
        /// <seealso cref="GetBaseDir(string)"/>
        /// <seealso cref="GetFile(string)"/>
        /// <param name="instance">The path to a file.</param>
        /// <returns>The path to the file without the extension.</returns>
        public static string GetBaseName(this string instance)
        {
            int index = instance.LastIndexOf('.');

            if (index > 0)
                return instance.Substring(0, index);

            return instance;
        }

        /// <summary>
        /// Returns <see langword="true"/> if the strings begins
        /// with the given string <paramref name="text"/>.
        /// </summary>
        /// <param name="instance">The string to check.</param>
        /// <param name="text">The beginning string.</param>
        /// <returns>If the string begins with the given string.</returns>
        public static bool BeginsWith(this string instance, string text)
        {
            return instance.StartsWith(text);
        }

        /// <summary>
        /// Returns the bigrams (pairs of consecutive letters) of this string.
        /// </summary>
        /// <param name="instance">The string that will be used.</param>
        /// <returns>The bigrams of this string.</returns>
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
        /// Converts a string containing a binary number into an integer.
        /// Binary strings can either be prefixed with <c>0b</c> or not,
        /// and they can also start with a <c>-</c> before the optional prefix.
        /// </summary>
        /// <param name="instance">The string to convert.</param>
        /// <returns>The converted string.</returns>
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
        /// Returns the amount of substrings <paramref name="what"/> in the string.
        /// </summary>
        /// <param name="instance">The string where the substring will be searched.</param>
        /// <param name="what">The substring that will be counted.</param>
        /// <param name="caseSensitive">If the search is case sensitive.</param>
        /// <param name="from">Index to start searching from.</param>
        /// <param name="to">Index to stop searching at.</param>
        /// <returns>Amount of substrings in the string.</returns>
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
        /// Returns a copy of the string with special characters escaped using the C language standard.
        /// </summary>
        /// <param name="instance">The string to escape.</param>
        /// <returns>The escaped string.</returns>
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
        /// Returns a copy of the string with escaped characters replaced by their meanings
        /// according to the C language standard.
        /// </summary>
        /// <param name="instance">The string to unescape.</param>
        /// <returns>The unescaped string.</returns>
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
        /// Changes the case of some letters. Replace underscores with spaces, convert all letters
        /// to lowercase then capitalize first and every letter following the space character.
        /// For <c>capitalize camelCase mixed_with_underscores</c> it will return
        /// <c>Capitalize Camelcase Mixed With Underscores</c>.
        /// </summary>
        /// <param name="instance">The string to capitalize.</param>
        /// <returns>The capitalized string.</returns>
        public static string Capitalize(this string instance)
        {
            string aux = instance.Replace("_", " ").ToLower();
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

        /// <summary>
        /// Performs a case-sensitive comparison to another string, return -1 if less, 0 if equal and +1 if greater.
        /// </summary>
        /// <seealso cref="NocasecmpTo(string, string)"/>
        /// <seealso cref="CompareTo(string, string, bool)"/>
        /// <param name="instance">The string to compare.</param>
        /// <param name="to">The other string to compare.</param>
        /// <returns>-1 if less, 0 if equal and +1 if greater.</returns>
        public static int CasecmpTo(this string instance, string to)
        {
            return instance.CompareTo(to, caseSensitive: true);
        }

        /// <summary>
        /// Performs a comparison to another string, return -1 if less, 0 if equal and +1 if greater.
        /// </summary>
        /// <param name="instance">The string to compare.</param>
        /// <param name="to">The other string to compare.</param>
        /// <param name="caseSensitive">
        /// If <see langword="true"/>, the comparison will be case sensitive.
        /// </param>
        /// <returns>-1 if less, 0 if equal and +1 if greater.</returns>
        public static int CompareTo(this string instance, string to, bool caseSensitive = true)
        {
            if (string.IsNullOrEmpty(instance))
                return string.IsNullOrEmpty(to) ? 0 : -1;

            if (string.IsNullOrEmpty(to))
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
        /// Returns <see langword="true"/> if the strings ends
        /// with the given string <paramref name="text"/>.
        /// </summary>
        /// <param name="instance">The string to check.</param>
        /// <param name="text">The ending string.</param>
        /// <returns>If the string ends with the given string.</returns>
        public static bool EndsWith(this string instance, string text)
        {
            return instance.EndsWith(text);
        }

        /// <summary>
        /// Erase <paramref name="chars"/> characters from the string starting from <paramref name="pos"/>.
        /// </summary>
        /// <param name="instance">The string to modify.</param>
        /// <param name="pos">Starting position from which to erase.</param>
        /// <param name="chars">Amount of characters to erase.</param>
        public static void Erase(this StringBuilder instance, int pos, int chars)
        {
            instance.Remove(pos, chars);
        }

        /// <summary>
        /// Returns the extension without the leading period character (<c>.</c>)
        /// if the string is a valid file name or path. If the string does not contain
        /// an extension, returns an empty string instead.
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
        /// <seealso cref="GetBaseName(string)"/>
        /// <seealso cref="GetBaseDir(string)"/>
        /// <seealso cref="GetFile(string)"/>
        /// <param name="instance">The path to a file.</param>
        /// <returns>The extension of the file or an empty string.</returns>
        public static string GetExtension(this string instance)
        {
            int pos = instance.FindLast(".");

            if (pos < 0)
                return instance;

            return instance.Substring(pos + 1);
        }

        /// <summary>
        /// Find the first occurrence of a substring. Optionally, the search starting position can be passed.
        /// </summary>
        /// <seealso cref="Find(string, char, int, bool)"/>
        /// <seealso cref="FindLast(string, string, bool)"/>
        /// <seealso cref="FindLast(string, string, int, bool)"/>
        /// <seealso cref="FindN(string, string, int)"/>
        /// <param name="instance">The string that will be searched.</param>
        /// <param name="what">The substring to find.</param>
        /// <param name="from">The search starting position.</param>
        /// <param name="caseSensitive">If <see langword="true"/>, the search is case sensitive.</param>
        /// <returns>The starting position of the substring, or -1 if not found.</returns>
        public static int Find(this string instance, string what, int from = 0, bool caseSensitive = true)
        {
            return instance.IndexOf(what, from, caseSensitive ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Find the first occurrence of a char. Optionally, the search starting position can be passed.
        /// </summary>
        /// <seealso cref="Find(string, string, int, bool)"/>
        /// <seealso cref="FindLast(string, string, bool)"/>
        /// <seealso cref="FindLast(string, string, int, bool)"/>
        /// <seealso cref="FindN(string, string, int)"/>
        /// <param name="instance">The string that will be searched.</param>
        /// <param name="what">The substring to find.</param>
        /// <param name="from">The search starting position.</param>
        /// <param name="caseSensitive">If <see langword="true"/>, the search is case sensitive.</param>
        /// <returns>The first instance of the char, or -1 if not found.</returns>
        public static int Find(this string instance, char what, int from = 0, bool caseSensitive = true)
        {
            // TODO: Could be more efficient if we get a char version of `IndexOf`.
            // See https://github.com/dotnet/runtime/issues/44116
            return instance.IndexOf(what.ToString(), from, caseSensitive ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>Find the last occurrence of a substring.</summary>
        /// <seealso cref="Find(string, string, int, bool)"/>
        /// <seealso cref="Find(string, char, int, bool)"/>
        /// <seealso cref="FindLast(string, string, int, bool)"/>
        /// <seealso cref="FindN(string, string, int)"/>
        /// <param name="instance">The string that will be searched.</param>
        /// <param name="what">The substring to find.</param>
        /// <param name="caseSensitive">If <see langword="true"/>, the search is case sensitive.</param>
        /// <returns>The starting position of the substring, or -1 if not found.</returns>
        public static int FindLast(this string instance, string what, bool caseSensitive = true)
        {
            return instance.FindLast(what, instance.Length - 1, caseSensitive);
        }

        /// <summary>Find the last occurrence of a substring specifying the search starting position.</summary>
        /// <seealso cref="Find(string, string, int, bool)"/>
        /// <seealso cref="Find(string, char, int, bool)"/>
        /// <seealso cref="FindLast(string, string, bool)"/>
        /// <seealso cref="FindN(string, string, int)"/>
        /// <param name="instance">The string that will be searched.</param>
        /// <param name="what">The substring to find.</param>
        /// <param name="from">The search starting position.</param>
        /// <param name="caseSensitive">If <see langword="true"/>, the search is case sensitive.</param>
        /// <returns>The starting position of the substring, or -1 if not found.</returns>
        public static int FindLast(this string instance, string what, int from, bool caseSensitive = true)
        {
            return instance.LastIndexOf(what, from, caseSensitive ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Find the first occurrence of a substring but search as case-insensitive.
        /// Optionally, the search starting position can be passed.
        /// </summary>
        /// <seealso cref="Find(string, string, int, bool)"/>
        /// <seealso cref="Find(string, char, int, bool)"/>
        /// <seealso cref="FindLast(string, string, bool)"/>
        /// <seealso cref="FindLast(string, string, int, bool)"/>
        /// <param name="instance">The string that will be searched.</param>
        /// <param name="what">The substring to find.</param>
        /// <param name="from">The search starting position.</param>
        /// <returns>The starting position of the substring, or -1 if not found.</returns>
        public static int FindN(this string instance, string what, int from = 0)
        {
            return instance.IndexOf(what, from, StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// If the string is a path to a file, return the base directory.
        /// </summary>
        /// <seealso cref="GetBaseName(string)"/>
        /// <seealso cref="GetExtension(string)"/>
        /// <seealso cref="GetFile(string)"/>
        /// <param name="instance">The path to a file.</param>
        /// <returns>The base directory.</returns>
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
        /// If the string is a path to a file, return the file and ignore the base directory.
        /// </summary>
        /// <seealso cref="GetBaseName(string)"/>
        /// <seealso cref="GetExtension(string)"/>
        /// <seealso cref="GetBaseDir(string)"/>
        /// <param name="instance">The path to a file.</param>
        /// <returns>The file name.</returns>
        public static string GetFile(this string instance)
        {
            int sep = Mathf.Max(instance.FindLast("/"), instance.FindLast("\\"));

            if (sep == -1)
                return instance;

            return instance.Substring(sep + 1);
        }

        /// <summary>
        /// Converts the given byte array of ASCII encoded text to a string.
        /// Faster alternative to <see cref="GetStringFromUTF8"/> if the
        /// content is ASCII-only. Unlike the UTF-8 function this function
        /// maps every byte to a character in the array. Multibyte sequences
        /// will not be interpreted correctly. For parsing user input always
        /// use <see cref="GetStringFromUTF8"/>.
        /// </summary>
        /// <param name="bytes">A byte array of ASCII characters (on the range of 0-127).</param>
        /// <returns>A string created from the bytes.</returns>
        public static string GetStringFromASCII(this byte[] bytes)
        {
            return Encoding.ASCII.GetString(bytes);
        }

        /// <summary>
        /// Converts the given byte array of UTF-8 encoded text to a string.
        /// Slower than <see cref="GetStringFromASCII"/> but supports UTF-8
        /// encoded data. Use this function if you are unsure about the
        /// source of the data. For user input this function
        /// should always be preferred.
        /// </summary>
        /// <param name="bytes">A byte array of UTF-8 characters (a character may take up multiple bytes).</param>
        /// <returns>A string created from the bytes.</returns>
        public static string GetStringFromUTF8(this byte[] bytes)
        {
            return Encoding.UTF8.GetString(bytes);
        }

        /// <summary>
        /// Hash the string and return a 32 bits unsigned integer.
        /// </summary>
        /// <param name="instance">The string to hash.</param>
        /// <returns>The calculated hash of the string.</returns>
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
        /// Returns a hexadecimal representation of this byte as a string.
        /// </summary>
        /// <param name="b">The byte to encode.</param>
        /// <returns>The hexadecimal representation of this byte.</returns>
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
        /// Returns a hexadecimal representation of this byte array as a string.
        /// </summary>
        /// <param name="bytes">The byte array to encode.</param>
        /// <returns>The hexadecimal representation of this byte array.</returns>
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
        /// Converts a string containing a hexadecimal number into an integer.
        /// Hexadecimal strings can either be prefixed with <c>0x</c> or not,
        /// and they can also start with a <c>-</c> before the optional prefix.
        /// </summary>
        /// <param name="instance">The string to convert.</param>
        /// <returns>The converted string.</returns>
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
        /// Inserts a substring at a given position.
        /// </summary>
        /// <param name="instance">The string to modify.</param>
        /// <param name="pos">Position at which to insert the substring.</param>
        /// <param name="what">Substring to insert.</param>
        /// <returns>
        /// The string with <paramref name="what"/> inserted at the given
        /// position <paramref name="pos"/>.
        /// </returns>
        public static string Insert(this string instance, int pos, string what)
        {
            return instance.Insert(pos, what);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the string is a path to a file or
        /// directory and its starting point is explicitly defined. This includes
        /// <c>res://</c>, <c>user://</c>, <c>C:\</c>, <c>/</c>, etc.
        /// </summary>
        /// <seealso cref="IsRelativePath(string)"/>
        /// <param name="instance">The string to check.</param>
        /// <returns>If the string is an absolute path.</returns>
        public static bool IsAbsolutePath(this string instance)
        {
            if (string.IsNullOrEmpty(instance))
                return false;
            else if (instance.Length > 1)
                return instance[0] == '/' || instance[0] == '\\' || instance.Contains(":/") || instance.Contains(":\\");
            else
                return instance[0] == '/' || instance[0] == '\\';
        }

        /// <summary>
        /// Returns <see langword="true"/> if the string is a path to a file or
        /// directory and its starting point is implicitly defined within the
        /// context it is being used. The starting point may refer to the current
        /// directory (<c>./</c>), or the current <see cref="Node"/>.
        /// </summary>
        /// <seealso cref="IsAbsolutePath(string)"/>
        /// <param name="instance">The string to check.</param>
        /// <returns>If the string is a relative path.</returns>
        public static bool IsRelativePath(this string instance)
        {
            return !IsAbsolutePath(instance);
        }

        /// <summary>
        /// Check whether this string is a subsequence of the given string.
        /// </summary>
        /// <seealso cref="IsSubsequenceOfN(string, string)"/>
        /// <param name="instance">The subsequence to search.</param>
        /// <param name="text">The string that contains the subsequence.</param>
        /// <param name="caseSensitive">If <see langword="true"/>, the check is case sensitive.</param>
        /// <returns>If the string is a subsequence of the given string.</returns>
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
        /// Check whether this string is a subsequence of the given string, ignoring case differences.
        /// </summary>
        /// <seealso cref="IsSubsequenceOf(string, string, bool)"/>
        /// <param name="instance">The subsequence to search.</param>
        /// <param name="text">The string that contains the subsequence.</param>
        /// <returns>If the string is a subsequence of the given string.</returns>
        public static bool IsSubsequenceOfN(this string instance, string text)
        {
            return instance.IsSubsequenceOf(text, caseSensitive: false);
        }

        /// <summary>
        /// Check whether the string contains a valid <see langword="float"/>.
        /// </summary>
        /// <param name="instance">The string to check.</param>
        /// <returns>If the string contains a valid floating point number.</returns>
        public static bool IsValidFloat(this string instance)
        {
            float f;
            return float.TryParse(instance, out f);
        }

        /// <summary>
        /// Check whether the string contains a valid color in HTML notation.
        /// </summary>
        /// <param name="instance">The string to check.</param>
        /// <returns>If the string contains a valid HTML color.</returns>
        public static bool IsValidHtmlColor(this string instance)
        {
            return Color.HtmlIsValid(instance);
        }

        /// <summary>
        /// Check whether the string is a valid identifier. As is common in
        /// programming languages, a valid identifier may contain only letters,
        /// digits and underscores (_) and the first character may not be a digit.
        /// </summary>
        /// <param name="instance">The string to check.</param>
        /// <returns>If the string contains a valid identifier.</returns>
        public static bool IsValidIdentifier(this string instance)
        {
            int len = instance.Length;

            if (len == 0)
                return false;

            if (instance[0] >= '0' && instance[0] <= '9')
                return false; // Identifiers cannot start with numbers.

            for (int i = 0; i < len; i++)
            {
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
        /// Check whether the string contains a valid integer.
        /// </summary>
        /// <param name="instance">The string to check.</param>
        /// <returns>If the string contains a valid integer.</returns>
        public static bool IsValidInteger(this string instance)
        {
            int f;
            return int.TryParse(instance, out f);
        }

        /// <summary>
        /// Check whether the string contains a valid IP address.
        /// </summary>
        /// <param name="instance">The string to check.</param>
        /// <returns>If the string contains a valid IP address.</returns>
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
        /// Returns a copy of the string with special characters escaped using the JSON standard.
        /// </summary>
        /// <param name="instance">The string to escape.</param>
        /// <returns>The escaped string.</returns>
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
        /// Returns an amount of characters from the left of the string.
        /// </summary>
        /// <seealso cref="Right(string, int)"/>
        /// <param name="instance">The original string.</param>
        /// <param name="pos">The position in the string where the left side ends.</param>
        /// <returns>The left side of the string from the given position.</returns>
        public static string Left(this string instance, int pos)
        {
            if (pos <= 0)
                return string.Empty;

            if (pos >= instance.Length)
                return instance;

            return instance.Substring(0, pos);
        }

        /// <summary>
        /// Returns the length of the string in characters.
        /// </summary>
        /// <param name="instance">The string to check.</param>
        /// <returns>The length of the string.</returns>
        public static int Length(this string instance)
        {
            return instance.Length;
        }

        /// <summary>
        /// Returns a copy of the string with characters removed from the left.
        /// </summary>
        /// <seealso cref="RStrip(string, string)"/>
        /// <param name="instance">The string to remove characters from.</param>
        /// <param name="chars">The characters to be removed.</param>
        /// <returns>A copy of the string with characters removed from the left.</returns>
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
        /// Do a simple expression match, where '*' matches zero or more
        /// arbitrary characters and '?' matches any single character except '.'.
        /// </summary>
        /// <param name="instance">The string to check.</param>
        /// <param name="expr">Expression to check.</param>
        /// <param name="caseSensitive">
        /// If <see langword="true"/>, the check will be case sensitive.
        /// </param>
        /// <returns>If the expression has any matches.</returns>
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
        /// Do a simple case sensitive expression match, using ? and * wildcards
        /// (see <see cref="ExprMatch(string, string, bool)"/>).
        /// </summary>
        /// <seealso cref="MatchN(string, string)"/>
        /// <param name="instance">The string to check.</param>
        /// <param name="expr">Expression to check.</param>
        /// <param name="caseSensitive">
        /// If <see langword="true"/>, the check will be case sensitive.
        /// </param>
        /// <returns>If the expression has any matches.</returns>
        public static bool Match(this string instance, string expr, bool caseSensitive = true)
        {
            if (instance.Length == 0 || expr.Length == 0)
                return false;

            return instance.ExprMatch(expr, caseSensitive);
        }

        /// <summary>
        /// Do a simple case insensitive expression match, using ? and * wildcards
        /// (see <see cref="ExprMatch(string, string, bool)"/>).
        /// </summary>
        /// <seealso cref="Match(string, string, bool)"/>
        /// <param name="instance">The string to check.</param>
        /// <param name="expr">Expression to check.</param>
        /// <returns>If the expression has any matches.</returns>
        public static bool MatchN(this string instance, string expr)
        {
            if (instance.Length == 0 || expr.Length == 0)
                return false;

            return instance.ExprMatch(expr, caseSensitive: false);
        }

        /// <summary>
        /// Returns the MD5 hash of the string as an array of bytes.
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
        /// Returns the MD5 hash of the string as a string.
        /// </summary>
        /// <seealso cref="MD5Buffer(string)"/>
        /// <param name="instance">The string to hash.</param>
        /// <returns>The MD5 hash of the string.</returns>
        public static string MD5Text(this string instance)
        {
            return godot_icall_String_md5_text(instance);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern string godot_icall_String_md5_text(string str);

        /// <summary>
        /// Perform a case-insensitive comparison to another string, return -1 if less, 0 if equal and +1 if greater.
        /// </summary>
        /// <seealso cref="CasecmpTo(string, string)"/>
        /// <seealso cref="CompareTo(string, string, bool)"/>
        /// <param name="instance">The string to compare.</param>
        /// <param name="to">The other string to compare.</param>
        /// <returns>-1 if less, 0 if equal and +1 if greater.</returns>
        public static int NocasecmpTo(this string instance, string to)
        {
            return instance.CompareTo(to, caseSensitive: false);
        }

        /// <summary>
        /// Returns the character code at position <paramref name="at"/>.
        /// </summary>
        /// <param name="instance">The string to check.</param>
        /// <param name="at">The position int the string for the character to check.</param>
        /// <returns>The character code.</returns>
        public static int OrdAt(this string instance, int at)
        {
            return instance[at];
        }

        /// <summary>
        /// Format a number to have an exact number of <paramref name="digits"/>
        /// after the decimal point.
        /// </summary>
        /// <seealso cref="PadZeros(string, int)"/>
        /// <param name="instance">The string to pad.</param>
        /// <param name="digits">Amount of digits after the decimal point.</param>
        /// <returns>The string padded with zeroes.</returns>
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
        /// Format a number to have an exact number of <paramref name="digits"/>
        /// before the decimal point.
        /// </summary>
        /// <seealso cref="PadDecimals(string, int)"/>
        /// <param name="instance">The string to pad.</param>
        /// <param name="digits">Amount of digits before the decimal point.</param>
        /// <returns>The string padded with zeroes.</returns>
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
        /// If the string is a path, this concatenates <paramref name="file"/>
        /// at the end of the string as a subpath.
        /// E.g. <c>"this/is".PlusFile("path") == "this/is/path"</c>.
        /// </summary>
        /// <param name="instance">The path that will be concatenated.</param>
        /// <param name="file">File name to concatenate with the path.</param>
        /// <returns>The concatenated path with the given file name.</returns>
        public static string PlusFile(this string instance, string file)
        {
            if (instance.Length > 0 && instance[instance.Length - 1] == '/')
                return instance + file;
            return instance + "/" + file;
        }

        /// <summary>
        /// Replace occurrences of a substring for different ones inside the string.
        /// </summary>
        /// <seealso cref="ReplaceN(string, string, string)"/>
        /// <param name="instance">The string to modify.</param>
        /// <param name="what">The substring to be replaced in the string.</param>
        /// <param name="forwhat">The substring that replaces <paramref name="what"/>.</param>
        /// <returns>The string with the substring occurrences replaced.</returns>
        public static string Replace(this string instance, string what, string forwhat)
        {
            return instance.Replace(what, forwhat);
        }

        /// <summary>
        /// Replace occurrences of a substring for different ones inside the string, but search case-insensitive.
        /// </summary>
        /// <seealso cref="Replace(string, string, string)"/>
        /// <param name="instance">The string to modify.</param>
        /// <param name="what">The substring to be replaced in the string.</param>
        /// <param name="forwhat">The substring that replaces <paramref name="what"/>.</param>
        /// <returns>The string with the substring occurrences replaced.</returns>
        public static string ReplaceN(this string instance, string what, string forwhat)
        {
            return Regex.Replace(instance, what, forwhat, RegexOptions.IgnoreCase);
        }

        /// <summary>
        /// Perform a search for a substring, but start from the end of the string instead of the beginning.
        /// </summary>
        /// <seealso cref="RFindN(string, string, int)"/>
        /// <param name="instance">The string that will be searched.</param>
        /// <param name="what">The substring to search in the string.</param>
        /// <param name="from">The position at which to start searching.</param>
        /// <returns>The position at which the substring was found, or -1 if not found.</returns>
        public static int RFind(this string instance, string what, int from = -1)
        {
            return godot_icall_String_rfind(instance, what, from);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern int godot_icall_String_rfind(string str, string what, int from);

        /// <summary>
        /// Perform a search for a substring, but start from the end of the string instead of the beginning.
        /// Also search case-insensitive.
        /// </summary>
        /// <seealso cref="RFind(string, string, int)"/>
        /// <param name="instance">The string that will be searched.</param>
        /// <param name="what">The substring to search in the string.</param>
        /// <param name="from">The position at which to start searching.</param>
        /// <returns>The position at which the substring was found, or -1 if not found.</returns>
        public static int RFindN(this string instance, string what, int from = -1)
        {
            return godot_icall_String_rfindn(instance, what, from);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern int godot_icall_String_rfindn(string str, string what, int from);

        /// <summary>
        /// Returns the right side of the string from a given position.
        /// </summary>
        /// <seealso cref="Left(string, int)"/>
        /// <param name="instance">The original string.</param>
        /// <param name="pos">The position in the string from which the right side starts.</param>
        /// <returns>The right side of the string from the given position.</returns>
        public static string Right(this string instance, int pos)
        {
            if (pos >= instance.Length)
                return instance;

            if (pos < 0)
                return string.Empty;

            return instance.Substring(pos, instance.Length - pos);
        }

        /// <summary>
        /// Returns a copy of the string with characters removed from the right.
        /// </summary>
        /// <seealso cref="LStrip(string, string)"/>
        /// <param name="instance">The string to remove characters from.</param>
        /// <param name="chars">The characters to be removed.</param>
        /// <returns>A copy of the string with characters removed from the right.</returns>
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
        /// Returns the SHA-256 hash of the string as an array of bytes.
        /// </summary>
        /// <seealso cref="SHA256Text(string)"/>
        /// <param name="instance">The string to hash.</param>
        /// <returns>The SHA-256 hash of the string.</returns>
        public static byte[] SHA256Buffer(this string instance)
        {
            return godot_icall_String_sha256_buffer(instance);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern byte[] godot_icall_String_sha256_buffer(string str);

        /// <summary>
        /// Returns the SHA-256 hash of the string as a string.
        /// </summary>
        /// <seealso cref="SHA256Buffer(string)"/>
        /// <param name="instance">The string to hash.</param>
        /// <returns>The SHA-256 hash of the string.</returns>
        public static string SHA256Text(this string instance)
        {
            return godot_icall_String_sha256_text(instance);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern string godot_icall_String_sha256_text(string str);

        /// <summary>
        /// Returns the similarity index of the text compared to this string.
        /// 1 means totally similar and 0 means totally dissimilar.
        /// </summary>
        /// <param name="instance">The string to compare.</param>
        /// <param name="text">The other string to compare.</param>
        /// <returns>The similarity index.</returns>
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
        /// Returns a simplified canonical path.
        /// </summary>
        public static string SimplifyPath(this string instance)
        {
            return godot_icall_String_simplify_path(instance);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern string godot_icall_String_simplify_path(string str);

        /// <summary>
        /// Split the string by a divisor string, return an array of the substrings.
        /// Example "One,Two,Three" will return ["One","Two","Three"] if split by ",".
        /// </summary>
        /// <seealso cref="SplitFloats(string, string, bool)"/>
        /// <param name="instance">The string to split.</param>
        /// <param name="divisor">The divisor string that splits the string.</param>
        /// <param name="allowEmpty">
        /// If <see langword="true"/>, the array may include empty strings.
        /// </param>
        /// <returns>The array of strings split from the string.</returns>
        public static string[] Split(this string instance, string divisor, bool allowEmpty = true)
        {
            return instance.Split(new[] { divisor }, allowEmpty ? StringSplitOptions.None : StringSplitOptions.RemoveEmptyEntries);
        }

        /// <summary>
        /// Split the string in floats by using a divisor string, return an array of the substrings.
        /// Example "1,2.5,3" will return [1,2.5,3] if split by ",".
        /// </summary>
        /// <seealso cref="Split(string, string, bool)"/>
        /// <param name="instance">The string to split.</param>
        /// <param name="divisor">The divisor string that splits the string.</param>
        /// <param name="allowEmpty">
        /// If <see langword="true"/>, the array may include empty floats.
        /// </param>
        /// <returns>The array of floats split from the string.</returns>
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
        /// Returns a copy of the string stripped of any non-printable character at the beginning and the end.
        /// The optional arguments are used to toggle stripping on the left and right edges respectively.
        /// </summary>
        /// <param name="instance">The string to strip.</param>
        /// <param name="left">If the left side should be stripped.</param>
        /// <param name="right">If the right side should be stripped.</param>
        /// <returns>The string stripped of any non-printable characters.</returns>
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
        /// Returns part of the string from the position <paramref name="from"/>, with length <paramref name="len"/>.
        /// </summary>
        /// <param name="instance">The string to slice.</param>
        /// <param name="from">The position in the string that the part starts from.</param>
        /// <param name="len">The length of the returned part.</param>
        /// <returns>
        /// Part of the string from the position <paramref name="from"/>, with length <paramref name="len"/>.
        /// </returns>
        public static string Substr(this string instance, int from, int len)
        {
            int max = instance.Length - from;
            return instance.Substring(from, len > max ? max : len);
        }

        /// <summary>
        /// Converts the String (which is a character array) to PackedByteArray (which is an array of bytes).
        /// The conversion is speeded up in comparison to <see cref="ToUTF8(string)"/> with the assumption
        /// that all the characters the String contains are only ASCII characters.
        /// </summary>
        /// <seealso cref="ToUTF8(string)"/>
        /// <param name="instance">The string to convert.</param>
        /// <returns>The string as ASCII encoded bytes.</returns>
        public static byte[] ToAscii(this string instance)
        {
            return Encoding.ASCII.GetBytes(instance);
        }

        /// <summary>
        /// Converts a string, containing a decimal number, into a <see langword="float" />.
        /// </summary>
        /// <seealso cref="ToInt(string)"/>
        /// <param name="instance">The string to convert.</param>
        /// <returns>The number representation of the string.</returns>
        public static float ToFloat(this string instance)
        {
            return float.Parse(instance);
        }

        /// <summary>
        /// Converts a string, containing an integer number, into an <see langword="int" />.
        /// </summary>
        /// <seealso cref="ToFloat(string)"/>
        /// <param name="instance">The string to convert.</param>
        /// <returns>The number representation of the string.</returns>
        public static int ToInt(this string instance)
        {
            return int.Parse(instance);
        }

        /// <summary>
        /// Returns the string converted to lowercase.
        /// </summary>
        /// <seealso cref="ToUpper(string)"/>
        /// <param name="instance">The string to convert.</param>
        /// <returns>The string converted to lowercase.</returns>
        public static string ToLower(this string instance)
        {
            return instance.ToLower();
        }

        /// <summary>
        /// Returns the string converted to uppercase.
        /// </summary>
        /// <seealso cref="ToLower(string)"/>
        /// <param name="instance">The string to convert.</param>
        /// <returns>The string converted to uppercase.</returns>
        public static string ToUpper(this string instance)
        {
            return instance.ToUpper();
        }

        /// <summary>
        /// Converts the String (which is an array of characters) to PackedByteArray (which is an array of bytes).
        /// The conversion is a bit slower than <see cref="ToAscii(string)"/>, but supports all UTF-8 characters.
        /// Therefore, you should prefer this function over <see cref="ToAscii(string)"/>.
        /// </summary>
        /// <seealso cref="ToAscii(string)"/>
        /// <param name="instance">The string to convert.</param>
        /// <returns>The string as UTF-8 encoded bytes.</returns>
        public static byte[] ToUTF8(this string instance)
        {
            return Encoding.UTF8.GetBytes(instance);
        }

        /// <summary>
        /// Decodes a string in URL encoded format. This is meant to
        /// decode parameters in a URL when receiving an HTTP request.
        /// This mostly wraps around <see cref="Uri.UnescapeDataString"/>,
        /// but also handles <c>+</c>.
        /// See <see cref="URIEncode"/> for encoding.
        /// </summary>
        /// <param name="instance">The string to decode.</param>
        /// <returns>The unescaped string.</returns>
        public static string URIDecode(this string instance)
        {
            return Uri.UnescapeDataString(instance.Replace("+", "%20"));
        }

        /// <summary>
        /// Encodes a string to URL friendly format. This is meant to
        /// encode parameters in a URL when sending an HTTP request.
        /// This wraps around <see cref="Uri.EscapeDataString"/>.
        /// See <see cref="URIDecode"/> for decoding.
        /// </summary>
        /// <param name="instance">The string to encode.</param>
        /// <returns>The escaped string.</returns>
        public static string URIEncode(this string instance)
        {
            return Uri.EscapeDataString(instance);
        }

        /// <summary>
        /// Returns a copy of the string with special characters escaped using the XML standard.
        /// </summary>
        /// <seealso cref="XMLUnescape(string)"/>
        /// <param name="instance">The string to escape.</param>
        /// <returns>The escaped string.</returns>
        public static string XMLEscape(this string instance)
        {
            return SecurityElement.Escape(instance);
        }

        /// <summary>
        /// Returns a copy of the string with escaped characters replaced by their meanings
        /// according to the XML standard.
        /// </summary>
        /// <seealso cref="XMLEscape(string)"/>
        /// <param name="instance">The string to unescape.</param>
        /// <returns>The unescaped string.</returns>
        public static string XMLUnescape(this string instance)
        {
            return SecurityElement.FromString(instance).Text;
        }
    }
}
