using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Security;
using System.Text;
using System.Text.RegularExpressions;

namespace Godot
{
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

        // <summary>
        // If the string is a path to a file, return the path to the file without the extension.
        // </summary>
        public static string BaseName(this string instance)
        {
            int index = instance.LastIndexOf('.');

            if (index > 0)
                return instance.Substring(0, index);

            return instance;
        }

        // <summary>
        // Return true if the strings begins with the given string.
        // </summary>
        public static bool BeginsWith(this string instance, string text)
        {
            return instance.StartsWith(text);
        }

        // <summary>
        // Return the bigrams (pairs of consecutive letters) of this string.
        // </summary>
        public static string[] Bigrams(this string instance)
        {
            var b = new string[instance.Length - 1];

            for (int i = 0; i < b.Length; i++)
            {
                b[i] = instance.Substring(i, 2);
            }

            return b;
        }

        // <summary>
        // Return the amount of substrings in string.
        // </summary>
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

        // <summary>
        // Return a copy of the string with special characters escaped using the C language standard.
        // </summary>
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

        // <summary>
        // Return a copy of the string with escaped characters replaced by their meanings according to the C language standard.
        // </summary>
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

        // <summary>
        // Change the case of some letters. Replace underscores with spaces, convert all letters to lowercase then capitalize first and every letter following the space character. For [code]capitalize camelCase mixed_with_underscores[/code] it will return [code]Capitalize Camelcase Mixed With Underscores[/code].
        // </summary>
        public static string Capitalize(this string instance)
        {
            string aux = instance.Replace("_", " ").ToLower();
            var cap = string.Empty;

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

        // <summary>
        // Perform a case-sensitive comparison to another string, return -1 if less, 0 if equal and +1 if greater.
        // </summary>
        public static int CasecmpTo(this string instance, string to)
        {
            return instance.CompareTo(to, caseSensitive: true);
        }

        // <summary>
        // Perform a comparison to another string, return -1 if less, 0 if equal and +1 if greater.
        // </summary>
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

        // <summary>
        // Return true if the strings ends with the given string.
        // </summary>
        public static bool EndsWith(this string instance, string text)
        {
            return instance.EndsWith(text);
        }

        // <summary>
        // Erase [code]chars[/code] characters from the string starting from [code]pos[/code].
        // </summary>
        public static void Erase(this StringBuilder instance, int pos, int chars)
        {
            instance.Remove(pos, chars);
        }

        // <summary>
        // If the string is a path to a file, return the extension.
        // </summary>
        public static string Extension(this string instance)
        {
            int pos = instance.FindLast(".");

            if (pos < 0)
                return instance;

            return instance.Substring(pos + 1);
        }

        /// <summary>Find the first occurrence of a substring. Optionally, the search starting position can be passed.</summary>
        /// <returns>The starting position of the substring, or -1 if not found.</returns>
        public static int Find(this string instance, string what, int from = 0, bool caseSensitive = true)
        {
            return instance.IndexOf(what, from, caseSensitive ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>Find the last occurrence of a substring.</summary>
        /// <returns>The starting position of the substring, or -1 if not found.</returns>
        public static int FindLast(this string instance, string what, bool caseSensitive = true)
        {
            return instance.FindLast(what, instance.Length - 1, caseSensitive);
        }

        /// <summary>Find the last occurrence of a substring specifying the search starting position.</summary>
        /// <returns>The starting position of the substring, or -1 if not found.</returns>
        public static int FindLast(this string instance, string what, int from, bool caseSensitive = true)
        {
            return instance.LastIndexOf(what, from, caseSensitive ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>Find the first occurrence of a substring but search as case-insensitive. Optionally, the search starting position can be passed.</summary>
        /// <returns>The starting position of the substring, or -1 if not found.</returns>
        public static int FindN(this string instance, string what, int from = 0)
        {
            return instance.IndexOf(what, from, StringComparison.OrdinalIgnoreCase);
        }

        // <summary>
        // If the string is a path to a file, return the base directory.
        // </summary>
        public static string GetBaseDir(this string instance)
        {
            int basepos = instance.Find("://");

            string rs;
            var @base = string.Empty;

            if (basepos != -1)
            {
                var end = basepos + 3;
                rs = instance.Substring(end);
                @base = instance.Substring(0, end);
            }
            else
            {
                if (instance.BeginsWith("/"))
                {
                    rs = instance.Substring(1);
                    @base = "/";
                }
                else
                {
                    rs = instance;
                }
            }

            int sep = Mathf.Max(rs.FindLast("/"), rs.FindLast("\\"));

            if (sep == -1)
                return @base;

            return @base + rs.Substr(0, sep);
        }

        // <summary>
        // If the string is a path to a file, return the file and ignore the base directory.
        // </summary>
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

        // <summary>
        // Hash the string and return a 32 bits integer.
        // </summary>
        public static int Hash(this string instance)
        {
            int index = 0;
            int hashv = 5381;
            int c;

            while ((c = instance[index++]) != 0)
                hashv = (hashv << 5) + hashv + c; // hash * 33 + c

            return hashv;
        }

        // <summary>
        // Convert a string containing an hexadecimal number into an int.
        // </summary>
        public static int HexToInt(this string instance)
        {
            int sign = 1;

            if (instance[0] == '-')
            {
                sign = -1;
                instance = instance.Substring(1);
            }

            if (!instance.StartsWith("0x"))
                return 0;

            return sign * int.Parse(instance.Substring(2), NumberStyles.HexNumber);
        }

        // <summary>
        // Insert a substring at a given position.
        // </summary>
        public static string Insert(this string instance, int pos, string what)
        {
            return instance.Insert(pos, what);
        }

        // <summary>
        // If the string is a path to a file or directory, return true if the path is absolute.
        // </summary>
        public static bool IsAbsPath(this string instance)
        {
            if (string.IsNullOrEmpty(instance))
                return false;
            else if (instance.Length > 1)
                return instance[0] == '/' || instance[0] == '\\' || instance.Contains(":/") || instance.Contains(":\\");
            else
                return instance[0] == '/' || instance[0] == '\\';
        }

        // <summary>
        // If the string is a path to a file or directory, return true if the path is relative.
        // </summary>
        public static bool IsRelPath(this string instance)
        {
            return !IsAbsPath(instance);
        }

        // <summary>
        // Check whether this string is a subsequence of the given string.
        // </summary>
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

        // <summary>
        // Check whether this string is a subsequence of the given string, ignoring case differences.
        // </summary>
        public static bool IsSubsequenceOfI(this string instance, string text)
        {
            return instance.IsSubsequenceOf(text, caseSensitive: false);
        }

        // <summary>
        // Check whether the string contains a valid float.
        // </summary>
        public static bool IsValidFloat(this string instance)
        {
            float f;
            return float.TryParse(instance, out f);
        }

        // <summary>
        // Check whether the string contains a valid color in HTML notation.
        // </summary>
        public static bool IsValidHtmlColor(this string instance)
        {
            return Color.HtmlIsValid(instance);
        }

        // <summary>
        // Check whether the string is a valid identifier. As is common in programming languages, a valid identifier may contain only letters, digits and underscores (_) and the first character may not be a digit.
        // </summary>
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

                bool validChar = instance[i] >= '0' &&
                                  instance[i] <= '9' || instance[i] >= 'a' &&
                                  instance[i] <= 'z' || instance[i] >= 'A' &&
                                  instance[i] <= 'Z' || instance[i] == '_';

                if (!validChar)
                    return false;
            }

            return true;
        }

        // <summary>
        // Check whether the string contains a valid integer.
        // </summary>
        public static bool IsValidInteger(this string instance)
        {
            int f;
            return int.TryParse(instance, out f);
        }

        // <summary>
        // Check whether the string contains a valid IP address.
        // </summary>
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

        // <summary>
        // Return a copy of the string with special characters escaped using the JSON standard.
        // </summary>
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

        // <summary>
        // Return an amount of characters from the left of the string.
        // </summary>
        public static string Left(this string instance, int pos)
        {
            if (pos <= 0)
                return string.Empty;

            if (pos >= instance.Length)
                return instance;

            return instance.Substring(0, pos);
        }

        /// <summary>
        /// Return the length of the string in characters.
        /// </summary>
        public static int Length(this string instance)
        {
            return instance.Length;
        }

        /// <summary>
        /// Do a simple expression match, where '*' matches zero or more arbitrary characters and '?' matches any single character except '.'.
        /// </summary>
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
                    if (instance.Length == 0) return false;
                    return (caseSensitive ? instance[0] == expr[0] : char.ToUpper(instance[0]) == char.ToUpper(expr[0])) && ExprMatch(instance.Substring(1), expr.Substring(1), caseSensitive);
            }
        }

        /// <summary>
        /// Do a simple case sensitive expression match, using ? and * wildcards (see [method expr_match]).
        /// </summary>
        public static bool Match(this string instance, string expr, bool caseSensitive = true)
        {
            if (instance.Length == 0 || expr.Length == 0)
                return false;

            return instance.ExprMatch(expr, caseSensitive);
        }

        /// <summary>
        /// Do a simple case insensitive expression match, using ? and * wildcards (see [method expr_match]).
        /// </summary>
        public static bool MatchN(this string instance, string expr)
        {
            if (instance.Length == 0 || expr.Length == 0)
                return false;

            return instance.ExprMatch(expr, caseSensitive: false);
        }

        // <summary>
        // Return the MD5 hash of the string as an array of bytes.
        // </summary>
        public static byte[] MD5Buffer(this string instance)
        {
            return godot_icall_String_md5_buffer(instance);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static byte[] godot_icall_String_md5_buffer(string str);

        // <summary>
        // Return the MD5 hash of the string as a string.
        // </summary>
        public static string MD5Text(this string instance)
        {
            return godot_icall_String_md5_text(instance);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static string godot_icall_String_md5_text(string str);

        // <summary>
        // Perform a case-insensitive comparison to another string, return -1 if less, 0 if equal and +1 if greater.
        // </summary>
        public static int NocasecmpTo(this string instance, string to)
        {
            return instance.CompareTo(to, caseSensitive: false);
        }

        // <summary>
        // Return the character code at position [code]at[/code].
        // </summary>
        public static int OrdAt(this string instance, int at)
        {
            return instance[at];
        }

        // <summary>
        // Format a number to have an exact number of [code]digits[/code] after the decimal point.
        // </summary>
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

        // <summary>
        // Format a number to have an exact number of [code]digits[/code] before the decimal point.
        // </summary>
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

        // <summary>
        // Decode a percent-encoded string. See [method percent_encode].
        // </summary>
        public static string PercentDecode(this string instance)
        {
            return Uri.UnescapeDataString(instance);
        }

        // <summary>
        // Percent-encode a string. This is meant to encode parameters in a URL when sending a HTTP GET request and bodies of form-urlencoded POST request.
        // </summary>
        public static string PercentEncode(this string instance)
        {
            return Uri.EscapeDataString(instance);
        }

        // <summary>
        // If the string is a path, this concatenates [code]file[/code] at the end of the string as a subpath. E.g. [code]"this/is".plus_file("path") == "this/is/path"[/code].
        // </summary>
        public static string PlusFile(this string instance, string file)
        {
            if (instance.Length > 0 && instance[instance.Length - 1] == '/')
                return instance + file;
            return instance + "/" + file;
        }

        // <summary>
        // Replace occurrences of a substring for different ones inside the string.
        // </summary>
        public static string Replace(this string instance, string what, string forwhat)
        {
            return instance.Replace(what, forwhat);
        }

        // <summary>
        // Replace occurrences of a substring for different ones inside the string, but search case-insensitive.
        // </summary>
        public static string ReplaceN(this string instance, string what, string forwhat)
        {
            return Regex.Replace(instance, what, forwhat, RegexOptions.IgnoreCase);
        }

        // <summary>
        // Perform a search for a substring, but start from the end of the string instead of the beginning.
        // </summary>
        public static int RFind(this string instance, string what, int from = -1)
        {
            return godot_icall_String_rfind(instance, what, from);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static int godot_icall_String_rfind(string str, string what, int from);

        // <summary>
        // Perform a search for a substring, but start from the end of the string instead of the beginning. Also search case-insensitive.
        // </summary>
        public static int RFindN(this string instance, string what, int from = -1)
        {
            return godot_icall_String_rfindn(instance, what, from);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static int godot_icall_String_rfindn(string str, string what, int from);

        // <summary>
        // Return the right side of the string from a given position.
        // </summary>
        public static string Right(this string instance, int pos)
        {
            if (pos >= instance.Length)
                return instance;

            if (pos < 0)
                return string.Empty;

            return instance.Substring(pos, instance.Length - pos);
        }

        public static byte[] SHA256Buffer(this string instance)
        {
            return godot_icall_String_sha256_buffer(instance);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static byte[] godot_icall_String_sha256_buffer(string str);

        // <summary>
        // Return the SHA-256 hash of the string as a string.
        // </summary>
        public static string SHA256Text(this string instance)
        {
            return godot_icall_String_sha256_text(instance);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static string godot_icall_String_sha256_text(string str);

        // <summary>
        // Return the similarity index of the text compared to this string. 1 means totally similar and 0 means totally dissimilar.
        // </summary>
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

        // <summary>
        // Split the string by a divisor string, return an array of the substrings. Example "One,Two,Three" will return ["One","Two","Three"] if split by ",".
        // </summary>
        public static string[] Split(this string instance, string divisor, bool allowEmpty = true)
        {
            return instance.Split(new[] { divisor }, StringSplitOptions.RemoveEmptyEntries);
        }

        // <summary>
        // Split the string in floats by using a divisor string, return an array of the substrings. Example "1,2.5,3" will return [1,2.5,3] if split by ",".
        // </summary>
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

        private static readonly char[] _nonPrintable = {
            (char)00, (char)01, (char)02, (char)03, (char)04, (char)05,
            (char)06, (char)07, (char)08, (char)09, (char)10, (char)11,
            (char)12, (char)13, (char)14, (char)15, (char)16, (char)17,
            (char)18, (char)19, (char)20, (char)21, (char)22, (char)23,
            (char)24, (char)25, (char)26, (char)27, (char)28, (char)29,
            (char)30, (char)31, (char)32
        };

        // <summary>
        // Return a copy of the string stripped of any non-printable character at the beginning and the end. The optional arguments are used to toggle stripping on the left and right edges respectively.
        // </summary>
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

        // <summary>
        // Return part of the string from the position [code]from[/code], with length [code]len[/code].
        // </summary>
        public static string Substr(this string instance, int from, int len)
        {
            int max = instance.Length - from;
            return instance.Substring(from, len > max ? max : len);
        }

        // <summary>
        // Convert the String (which is a character array) to PackedByteArray (which is an array of bytes). The conversion is speeded up in comparison to to_utf8() with the assumption that all the characters the String contains are only ASCII characters.
        // </summary>
        public static byte[] ToAscii(this string instance)
        {
            return Encoding.ASCII.GetBytes(instance);
        }

        // <summary>
        // Convert a string, containing a decimal number, into a [code]float[/code].
        // </summary>
        public static float ToFloat(this string instance)
        {
            return float.Parse(instance);
        }

        // <summary>
        // Convert a string, containing an integer number, into an [code]int[/code].
        // </summary>
        public static int ToInt(this string instance)
        {
            return int.Parse(instance);
        }

        // <summary>
        // Return the string converted to lowercase.
        // </summary>
        public static string ToLower(this string instance)
        {
            return instance.ToLower();
        }

        // <summary>
        // Return the string converted to uppercase.
        // </summary>
        public static string ToUpper(this string instance)
        {
            return instance.ToUpper();
        }

        // <summary>
        // Convert the String (which is an array of characters) to PackedByteArray (which is an array of bytes). The conversion is a bit slower than to_ascii(), but supports all UTF-8 characters. Therefore, you should prefer this function over to_ascii().
        // </summary>
        public static byte[] ToUTF8(this string instance)
        {
            return Encoding.UTF8.GetBytes(instance);
        }

        // <summary>
        // Return a copy of the string with special characters escaped using the XML standard.
        // </summary>
        public static string XMLEscape(this string instance)
        {
            return SecurityElement.Escape(instance);
        }

        // <summary>
        // Return a copy of the string with escaped characters replaced by their meanings according to the XML standard.
        // </summary>
        public static string XMLUnescape(this string instance)
        {
            return SecurityElement.FromString(instance).Text;
        }
    }
}
