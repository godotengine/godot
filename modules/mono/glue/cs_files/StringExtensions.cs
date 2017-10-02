//using System;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Security;
using System.Text;
using System.Text.RegularExpressions;

namespace Godot
{
    public static class StringExtensions
    {
        private static int get_slice_count(this string instance, string splitter)
        {
            if (instance.empty() || splitter.empty())
                return 0;

            int pos = 0;
            int slices = 1;

            while ((pos = instance.find(splitter, pos)) >= 0)
            {
                slices++;
                pos += splitter.Length;
            }

            return slices;
        }

        private static string get_slicec(this string instance, char splitter, int slice)
        {
            if (!instance.empty() && slice >= 0)
            {
                int i = 0;
                int prev = 0;
                int count = 0;

                while (true)
                {
                    if (instance[i] == 0 || instance[i] == splitter)
                    {
                        if (slice == count)
                        {
                            return instance.Substring(prev, i - prev);
                        }
                        else
                        {
                            count++;
                            prev = i + 1;
                        }
                    }

                    i++;
                }
            }

            return string.Empty;
        }

        // <summary>
        // If the string is a path to a file, return the path to the file without the extension.
        // </summary>
        public static string basename(this string instance)
        {
            int index = instance.LastIndexOf('.');

            if (index > 0)
                return instance.Substring(0, index);

            return instance;
        }

        // <summary>
        // Return true if the strings begins with the given string.
        // </summary>
        public static bool begins_with(this string instance, string text)
        {
            return instance.StartsWith(text);
        }

        // <summary>
        // Return the bigrams (pairs of consecutive letters) of this string.
        // </summary>
        public static string[] bigrams(this string instance)
        {
            string[] b = new string[instance.Length - 1];

            for (int i = 0; i < b.Length; i++)
            {
                b[i] = instance.Substring(i, 2);
            }

            return b;
        }

        // <summary>
        // Return a copy of the string with special characters escaped using the C language standard.
        // </summary>
        public static string c_escape(this string instance)
        {
            StringBuilder sb = new StringBuilder(string.Copy(instance));

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
        public static string c_unescape(this string instance)
        {
            StringBuilder sb = new StringBuilder(string.Copy(instance));

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
        public static string capitalize(this string instance)
        {
            string aux = instance.Replace("_", " ").ToLower();
            string cap = string.Empty;

            for (int i = 0; i < aux.get_slice_count(" "); i++)
            {
                string slice = aux.get_slicec(' ', i);
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
        public static int casecmp_to(this string instance, string to)
        {
            if (instance.empty())
                return to.empty() ? 0 : -1;

            if (to.empty())
                return 1;

            int instance_idx = 0;
            int to_idx = 0;

            while (true)
            {
                if (to[to_idx] == 0 && instance[instance_idx] == 0)
                    return 0; // We're equal
                else if (instance[instance_idx] == 0)
                    return -1; // If this is empty, and the other one is not, then we're less... I think?
                else if (to[to_idx] == 0)
                    return 1; // Otherwise the other one is smaller...
                else if (instance[instance_idx] < to[to_idx]) // More than
                    return -1;
                else if (instance[instance_idx] > to[to_idx]) // Less than
                    return 1;

                instance_idx++;
                to_idx++;
            }
        }

        // <summary>
        // Return true if the string is empty.
        // </summary>
        public static bool empty(this string instance)
        {
            return string.IsNullOrEmpty(instance);
        }

        // <summary>
        // Return true if the strings ends with the given string.
        // </summary>
        public static bool ends_with(this string instance, string text)
        {
            return instance.EndsWith(text);
        }

        // <summary>
        // Erase [code]chars[/code] characters from the string starting from [code]pos[/code].
        // </summary>
        public static void erase(this StringBuilder instance, int pos, int chars)
        {
            instance.Remove(pos, chars);
        }

        // <summary>
        // If the string is a path to a file, return the extension.
        // </summary>
        public static string extension(this string instance)
        {
            int pos = instance.find_last(".");

            if (pos < 0)
                return instance;

            return instance.Substring(pos + 1, instance.Length);
        }

        // <summary>
        // Find the first occurrence of a substring, return the starting position of the substring or -1 if not found. Optionally, the initial search index can be passed.
        // </summary>
        public static int find(this string instance, string what, int from = 0)
        {
            return instance.IndexOf(what, StringComparison.OrdinalIgnoreCase);
        }

        // <summary>
        // Find the last occurrence of a substring, return the starting position of the substring or -1 if not found. Optionally, the initial search index can be passed.
        // </summary>
        public static int find_last(this string instance, string what)
        {
            return instance.LastIndexOf(what, StringComparison.OrdinalIgnoreCase);
        }

        // <summary>
        // Find the first occurrence of a substring but search as case-insensitive, return the starting position of the substring or -1 if not found. Optionally, the initial search index can be passed.
        // </summary>
        public static int findn(this string instance, string what, int from = 0)
        {
            return instance.IndexOf(what, StringComparison.Ordinal);
        }

        // <summary>
        // If the string is a path to a file, return the base directory.
        // </summary>
        public static string get_base_dir(this string instance)
        {
            int basepos = instance.find("://");

            string rs = string.Empty;
            string @base = string.Empty;

            if (basepos != -1)
            {
                int end = basepos + 3;
                rs = instance.Substring(end, instance.Length);
                @base = instance.Substring(0, end);
            }
            else
            {
                if (instance.begins_with("/"))
                {
                    rs = instance.Substring(1, instance.Length);
                    @base = "/";
                }
                else
                {
                    rs = instance;
                }
            }

            int sep = Mathf.max(rs.find_last("/"), rs.find_last("\\"));

            if (sep == -1)
                return @base;

            return @base + rs.substr(0, sep);
        }

        // <summary>
        // If the string is a path to a file, return the file and ignore the base directory.
        // </summary>
        public static string get_file(this string instance)
        {
            int sep = Mathf.max(instance.find_last("/"), instance.find_last("\\"));

            if (sep == -1)
                return instance;

            return instance.Substring(sep + 1, instance.Length);
        }

        // <summary>
        // Hash the string and return a 32 bits integer.
        // </summary>
        public static int hash(this string instance)
        {
            int index = 0;
            int hashv = 5381;
            int c;

            while ((c = (int)instance[index++]) != 0)
                hashv = ((hashv << 5) + hashv) + c; // hash * 33 + c

            return hashv;
        }

        // <summary>
        // Convert a string containing an hexadecimal number into an int.
        // </summary>
        public static int hex_to_int(this string instance)
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
        public static string insert(this string instance, int pos, string what)
        {
            return instance.Insert(pos, what);
        }

        // <summary>
        // If the string is a path to a file or directory, return true if the path is absolute.
        // </summary>
        public static bool is_abs_path(this string instance)
        {
            return System.IO.Path.IsPathRooted(instance);
        }

        // <summary>
        // If the string is a path to a file or directory, return true if the path is relative.
        // </summary>
        public static bool is_rel_path(this string instance)
        {
            return !System.IO.Path.IsPathRooted(instance);
        }

        // <summary>
        // Check whether this string is a subsequence of the given string.
        // </summary>
        public static bool is_subsequence_of(this string instance, string text, bool case_insensitive)
        {
            int len = instance.Length;

            if (len == 0)
                return true; // Technically an empty string is subsequence of any string

            if (len > text.Length)
                return false;

            int src = 0;
            int tgt = 0;

            while (instance[src] != 0 && text[tgt] != 0)
            {
                bool match = false;

                if (case_insensitive)
                {
                    char srcc = char.ToLower(instance[src]);
                    char tgtc = char.ToLower(text[tgt]);
                    match = srcc == tgtc;
                }
                else
                {
                    match = instance[src] == text[tgt];
                }
                if (match)
                {
                    src++;
                    if (instance[src] == 0)
                        return true;
                }

                tgt++;
            }

            return false;
        }

        // <summary>
        // Check whether this string is a subsequence of the given string, considering case.
        // </summary>
        public static bool is_subsequence_of(this string instance, string text)
        {
            return instance.is_subsequence_of(text, false);
        }

        // <summary>
        // Check whether this string is a subsequence of the given string, without considering case.
        // </summary>
        public static bool is_subsequence_ofi(this string instance, string text)
        {
            return instance.is_subsequence_of(text, true);
        }

        // <summary>
        // Check whether the string contains a valid float.
        // </summary>
        public static bool is_valid_float(this string instance)
        {
            float f;
            return float.TryParse(instance, out f);
        }

        // <summary>
        // Check whether the string contains a valid color in HTML notation.
        // </summary>
        public static bool is_valid_html_color(this string instance)
        {
            return Color.html_is_valid(instance);
        }

        // <summary>
        // Check whether the string is a valid identifier. As is common in programming languages, a valid identifier may contain only letters, digits and underscores (_) and the first character may not be a digit.
        // </summary>
        public static bool is_valid_identifier(this string instance)
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

                bool valid_char = (instance[i] >= '0' && instance[i] <= '9') || (instance[i] >= 'a' && instance[i] <= 'z') || (instance[i] >= 'A' && instance[i] <= 'Z') || instance[i] == '_';

                if (!valid_char)
                    return false;
            }

            return true;
        }

        // <summary>
        // Check whether the string contains a valid integer.
        // </summary>
        public static bool is_valid_integer(this string instance)
        {
            int f;
            return int.TryParse(instance, out f);
        }

        // <summary>
        // Check whether the string contains a valid IP address.
        // </summary>
        public static bool is_valid_ip_address(this string instance)
        {
            string[] ip = instance.split(".");

            if (ip.Length != 4)
                return false;

            for (int i = 0; i < ip.Length; i++)
            {
                string n = ip[i];
                if (!n.is_valid_integer())
                    return false;

                int val = n.to_int();
                if (val < 0 || val > 255)
                    return false;
            }

            return true;
        }

        // <summary>
        // Return a copy of the string with special characters escaped using the JSON standard.
        // </summary>
        public static string json_escape(this string instance)
        {
            StringBuilder sb = new StringBuilder(string.Copy(instance));

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
        public static string left(this string instance, int pos)
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
        public static int length(this string instance)
        {
            return instance.Length;
        }

        // <summary>
        // Do a simple expression match, where '*' matches zero or more arbitrary characters and '?' matches any single character except '.'.
        // </summary>
        public static bool expr_match(this string instance, string expr, bool case_sensitive)
        {
            if (expr.Length == 0 || instance.Length == 0)
                return false;

            switch (expr[0])
            {
                case '\0':
                    return instance[0] == 0;
                case '*':
                    return expr_match(expr + 1, instance, case_sensitive) || (instance[0] != 0 && expr_match(expr, instance + 1, case_sensitive));
                case '?':
                    return instance[0] != 0 && instance[0] != '.' && expr_match(expr + 1, instance + 1, case_sensitive);
                default:
                    return (case_sensitive ? instance[0] == expr[0] : char.ToUpper(instance[0]) == char.ToUpper(expr[0])) &&
                                expr_match(expr + 1, instance + 1, case_sensitive);
            }
        }

        // <summary>
        // Do a simple case sensitive expression match, using ? and * wildcards (see [method expr_match]).
        // </summary>
        public static bool match(this string instance, string expr)
        {
            return instance.expr_match(expr, true);
        }

        // <summary>
        // Do a simple case insensitive expression match, using ? and * wildcards (see [method expr_match]).
        // </summary>
        public static bool matchn(this string instance, string expr)
        {
            return instance.expr_match(expr, false);
        }

        // <summary>
        // Return the MD5 hash of the string as an array of bytes.
        // </summary>
        public static byte[] md5_buffer(this string instance)
        {
			return NativeCalls.godot_icall_String_md5_buffer(instance);
        }

        // <summary>
        // Return the MD5 hash of the string as a string.
        // </summary>
        public static string md5_text(this string instance)
        {
			return NativeCalls.godot_icall_String_md5_text(instance);
        }

        // <summary>
        // Perform a case-insensitive comparison to another string, return -1 if less, 0 if equal and +1 if greater.
        // </summary>
        public static int nocasecmp_to(this string instance, string to)
        {
            if (instance.empty())
                return to.empty() ? 0 : -1;

            if (to.empty())
                return 1;

            int instance_idx = 0;
            int to_idx = 0;

            while (true)
            {
                if (to[to_idx] == 0 && instance[instance_idx] == 0)
                    return 0; // We're equal
                else if (instance[instance_idx] == 0)
                    return -1; // If this is empty, and the other one is not, then we're less... I think?
                else if (to[to_idx] == 0)
                    return 1; // Otherwise the other one is smaller..
                else if (char.ToUpper(instance[instance_idx]) < char.ToUpper(to[to_idx])) // More than
                    return -1;
                else if (char.ToUpper(instance[instance_idx]) > char.ToUpper(to[to_idx])) // Less than
                    return 1;

                instance_idx++;
                to_idx++;
            }
        }

        // <summary>
        // Return the character code at position [code]at[/code].
        // </summary>
        public static int ord_at(this string instance, int at)
        {
            return instance[at];
        }

        // <summary>
        // Format a number to have an exact number of [code]digits[/code] after the decimal point.
        // </summary>
        public static string pad_decimals(this string instance, int digits)
        {
            int c = instance.find(".");

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
        public static string pad_zeros(this string instance, int digits)
        {
            string s = instance;
            int end = s.find(".");

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
        public static string percent_decode(this string instance)
        {
            return Uri.UnescapeDataString(instance);
        }

        // <summary>
        // Percent-encode a string. This is meant to encode parameters in a URL when sending a HTTP GET request and bodies of form-urlencoded POST request.
        // </summary>
        public static string percent_encode(this string instance)
        {
            return Uri.EscapeDataString(instance);
        }

        // <summary>
        // If the string is a path, this concatenates [code]file[/code] at the end of the string as a subpath. E.g. [code]"this/is".plus_file("path") == "this/is/path"[/code].
        // </summary>
        public static string plus_file(this string instance, string file)
        {
            if (instance.Length > 0 && instance[instance.Length - 1] == '/')
                return instance + file;
            else
                return instance + "/" + file;
        }

        // <summary>
        // Replace occurrences of a substring for different ones inside the string.
        // </summary>
        public static string replace(this string instance, string what, string forwhat)
        {
            return instance.Replace(what, forwhat);
        }

        // <summary>
        // Replace occurrences of a substring for different ones inside the string, but search case-insensitive.
        // </summary>
        public static string replacen(this string instance, string what, string forwhat)
        {
            return Regex.Replace(instance, what, forwhat, RegexOptions.IgnoreCase);
        }

        // <summary>
        // Perform a search for a substring, but start from the end of the string instead of the beginning.
        // </summary>
        public static int rfind(this string instance, string what, int from = -1)
        {
			return NativeCalls.godot_icall_String_rfind(instance, what, from);
        }

        // <summary>
        // Perform a search for a substring, but start from the end of the string instead of the beginning. Also search case-insensitive.
        // </summary>
        public static int rfindn(this string instance, string what, int from = -1)
        {
			return NativeCalls.godot_icall_String_rfindn(instance, what, from);
        }

        // <summary>
        // Return the right side of the string from a given position.
        // </summary>
        public static string right(this string instance, int pos)
        {
            if (pos >= instance.Length)
                return instance;

            if (pos < 0)
                return string.Empty;

            return instance.Substring(pos, (instance.Length - pos));
        }

        public static byte[] sha256_buffer(this string instance)
        {
			return NativeCalls.godot_icall_String_sha256_buffer(instance);
        }

        // <summary>
        // Return the SHA-256 hash of the string as a string.
        // </summary>
        public static string sha256_text(this string instance)
        {
			return NativeCalls.godot_icall_String_sha256_text(instance);
        }

        // <summary>
        // Return the similarity index of the text compared to this string. 1 means totally similar and 0 means totally dissimilar.
        // </summary>
        public static float similarity(this string instance, string text)
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

            string[] src_bigrams = instance.bigrams();
            string[] tgt_bigrams = text.bigrams();

            int src_size = src_bigrams.Length;
            int tgt_size = tgt_bigrams.Length;

            float sum = src_size + tgt_size;
            float inter = 0;

            for (int i = 0; i < src_size; i++)
            {
                for (int j = 0; j < tgt_size; j++)
                {
                    if (src_bigrams[i] == tgt_bigrams[j])
                    {
                        inter++;
                        break;
                    }
                }
            }

            return (2.0f * inter) / sum;
        }

        // <summary>
        // Split the string by a divisor string, return an array of the substrings. Example "One,Two,Three" will return ["One","Two","Three"] if split by ",".
        // </summary>
        public static string[] split(this string instance, string divisor, bool allow_empty = true)
        {
            return instance.Split(new string[] { divisor }, StringSplitOptions.RemoveEmptyEntries);
        }

        // <summary>
        // Split the string in floats by using a divisor string, return an array of the substrings. Example "1,2.5,3" will return [1,2.5,3] if split by ",".
        // </summary>
        public static float[] split_floats(this string instance, string divisor, bool allow_empty = true)
        {
            List<float> ret = new List<float>();
            int from = 0;
            int len = instance.Length;

            while (true)
            {
                int end = instance.find(divisor, from);
                if (end < 0)
                    end = len;
                if (allow_empty || (end > from))
                    ret.Add(float.Parse(instance.Substring(from)));
                if (end == len)
                    break;

                from = end + divisor.Length;
            }

            return ret.ToArray();
        }

        private static readonly char[] non_printable = {
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
        public static string strip_edges(this string instance, bool left = true, bool right = true)
        {
            if (left)
            {
                if (right)
                    return instance.Trim(non_printable);
                else
                    return instance.TrimStart(non_printable);
            }
            else
            {
                return instance.TrimEnd(non_printable);
            }
        }

        // <summary>
        // Return part of the string from the position [code]from[/code], with length [code]len[/code].
        // </summary>
        public static string substr(this string instance, int from, int len)
        {
            return instance.Substring(from, len);
        }

        // <summary>
        // Convert the String (which is a character array) to PoolByteArray (which is an array of bytes). The conversion is speeded up in comparison to to_utf8() with the assumption that all the characters the String contains are only ASCII characters.
        // </summary>
        public static byte[] to_ascii(this string instance)
        {
            return Encoding.ASCII.GetBytes(instance);
        }

        // <summary>
        // Convert a string, containing a decimal number, into a [code]float[/code].
        // </summary>
        public static float to_float(this string instance)
        {
            return float.Parse(instance);
        }

        // <summary>
        // Convert a string, containing an integer number, into an [code]int[/code].
        // </summary>
        public static int to_int(this string instance)
        {
            return int.Parse(instance);
        }

        // <summary>
        // Return the string converted to lowercase.
        // </summary>
        public static string to_lower(this string instance)
        {
            return instance.ToLower();
        }

        // <summary>
        // Return the string converted to uppercase.
        // </summary>
        public static string to_upper(this string instance)
        {
            return instance.ToUpper();
        }

        // <summary>
        // Convert the String (which is an array of characters) to PoolByteArray (which is an array of bytes). The conversion is a bit slower than to_ascii(), but supports all UTF-8 characters. Therefore, you should prefer this function over to_ascii().
        // </summary>
        public static byte[] to_utf8(this string instance)
        {
            return Encoding.UTF8.GetBytes(instance);
        }

        // <summary>
        // Return a copy of the string with special characters escaped using the XML standard.
        // </summary>
        public static string xml_escape(this string instance)
        {
            return SecurityElement.Escape(instance);
        }

        // <summary>
        // Return a copy of the string with escaped characters replaced by their meanings according to the XML standard.
        // </summary>
        public static string xml_unescape(this string instance)
        {
            return SecurityElement.FromString(instance).Text;
        }
    }
}
