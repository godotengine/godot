using System;
using System.Collections.Generic;
using System.Globalization;
using System.Security;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;


#nullable enable
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

		public static string[] Bigrams(this string instance)
		{
			string[] b = new string[instance.Length - 1];

			for (int i = 0; i < b.Length; i++)
			{
				b[i] = instance.Substring(i, 2);
			}

			return b;
		}

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

			if (instance.StartsWith("0b", StringComparison.OrdinalIgnoreCase))
			{
				instance = instance.Substring(2);
			}

			return sign * Convert.ToInt32(instance, 2);
		}

		public static int Count(this string instance, string what, int from = 0, int to = 0, bool caseSensitive = true)
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

		public static int CountN(this string instance, string what, int from = 0, int to = 0)
		{
			return instance.Count(what, from, to, caseSensitive: false);
		}

		public static string Dedent(this string instance)
		{
			var sb = new StringBuilder();
			string indent = "";
			bool hasIndent = false;
			bool hasText = false;
			int lineStart = 0;
			int indentStop = -1;

			for (int i = 0; i < instance.Length; i++)
			{
				char c = instance[i];
				if (c == '\n')
				{
					if (hasText)
					{
						sb.Append(instance.AsSpan(indentStop, i - indentStop));
					}
					sb.Append('\n');
					hasText = false;
					lineStart = i + 1;
					indentStop = -1;
				}
				else if (!hasText)
				{
					if (c > 32)
					{
						hasText = true;
						if (!hasIndent)
						{
							hasIndent = true;
							indent = instance.Substring(lineStart, i - lineStart);
							indentStop = i;
						}
					}

					if (hasIndent && indentStop < 0)
					{
						int j = i - lineStart;
						if (j >= indent.Length || c != indent[j])
						{
							indentStop = i;
						}
					}
				}
			}

			if (hasText)
			{
				sb.Append(instance.AsSpan(indentStop, instance.Length - indentStop));
			}

			return sb.ToString();
		}

		public static string CEscape(this string instance)
		{
			var sb = new StringBuilder(instance);

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

			return sb.ToString();
		}

		public static string CUnescape(this string instance)
		{
			var sb = new StringBuilder(instance);

			sb.Replace("\\a", "\a");
			sb.Replace("\\b", "\b");
			sb.Replace("\\f", "\f");
			sb.Replace("\\n", "\n");
			sb.Replace("\\r", "\r");
			sb.Replace("\\t", "\t");
			sb.Replace("\\v", "\v");
			sb.Replace("\\'", "\'");
			sb.Replace("\\\"", "\"");
			sb.Replace("\\\\", "\\");

			return sb.ToString();
		}

		public static string Capitalize(this string instance)
		{
			if (string.IsNullOrEmpty(instance))
				return string.Empty;

			var sb = new StringBuilder(instance.Length);
			bool newWord = true;
			bool lastWasSpace = false;

			for (int i = 0; i < instance.Length; i++)
			{
				char c = instance[i];

				if (c == '_' || char.IsWhiteSpace(c))
				{
					if (!lastWasSpace && sb.Length > 0)
					{
						sb.Append(' ');
						lastWasSpace = true;
					}
					newWord = true;
					continue;
				}

				if (newWord)
				{
					sb.Append(char.ToUpperInvariant(c));
					newWord = false;
					lastWasSpace = false;
				}
				else
				{
					sb.Append(char.ToLowerInvariant(c));
					lastWasSpace = false;
				}
			}

			return sb.ToString().Trim();
		}

		private static bool IsWordSeparator(char c)
		{
			return c == '_' || c == ' ' || c == '-' || c == '.' || c == '/' || c == '\\' || c == ':';
		}

		private static List<string> SplitToWords(string instance)
		{
			var words = new List<string>();
			if (string.IsNullOrEmpty(instance))
				return words;

			var current = new StringBuilder();

			for (int i = 0; i < instance.Length; i++)
			{
				char c = instance[i];

				if (IsWordSeparator(c))
				{
					if (current.Length > 0)
					{
						words.Add(current.ToString());
						current.Clear();
					}
					continue;
				}

				bool boundary =
					current.Length > 0 &&
					char.IsUpper(c) &&
					(char.IsLower(current[^1]) || char.IsDigit(current[^1]));

				bool acronymBoundary =
					current.Length > 0 &&
					char.IsUpper(c) &&
					i + 1 < instance.Length &&
					char.IsLower(instance[i + 1]) &&
					char.IsUpper(current[^1]);

				if (boundary || acronymBoundary)
				{
					words.Add(current.ToString());
					current.Clear();
				}

				current.Append(c);
			}

			if (current.Length > 0)
				words.Add(current.ToString());

			return words;
		}

		public static string ToCamelCase(this string instance)
		{
			var words = SplitToWords(instance);
			if (words.Count == 0)
				return string.Empty;

			var sb = new StringBuilder(instance.Length);
			for (int i = 0; i < words.Count; i++)
			{
				string word = words[i];
				if (word.Length == 0)
					continue;

				if (i == 0)
					sb.Append(word.ToLowerInvariant());
				else
					sb.Append(char.ToUpperInvariant(word[0])).Append(word.Substring(1).ToLowerInvariant());
			}

			return sb.ToString();
		}

		public static string ToPascalCase(this string instance)
		{
			var words = SplitToWords(instance);
			if (words.Count == 0)
				return string.Empty;

			var sb = new StringBuilder(instance.Length);
			foreach (string word in words)
			{
				if (word.Length == 0)
					continue;

				sb.Append(char.ToUpperInvariant(word[0]));
				if (word.Length > 1)
					sb.Append(word.Substring(1).ToLowerInvariant());
			}

			return sb.ToString();
		}

		private static string ToSeparatedCase(string instance, char separator)
		{
			if (string.IsNullOrEmpty(instance))
				return string.Empty;

			var sb = new StringBuilder(instance.Length + 8);
			bool lastWasSeparator = true;

			for (int i = 0; i < instance.Length; i++)
			{
				char c = instance[i];

				if (IsWordSeparator(c))
				{
					if (!lastWasSeparator && sb.Length > 0)
					{
						sb.Append(separator);
						lastWasSeparator = true;
					}
					continue;
				}

				bool boundary =
					i > 0 &&
					char.IsUpper(c) &&
					(char.IsLower(instance[i - 1]) || char.IsDigit(instance[i - 1]));

				bool acronymBoundary =
					i > 0 &&
					char.IsUpper(c) &&
					i + 1 < instance.Length &&
					char.IsLower(instance[i + 1]) &&
					char.IsUpper(instance[i - 1]);

				if ((boundary || acronymBoundary) && !lastWasSeparator && sb.Length > 0 && sb[^1] != separator)
				{
					sb.Append(separator);
					lastWasSeparator = true;
				}

				sb.Append(char.ToLowerInvariant(c));
				lastWasSeparator = false;
			}

			return sb.ToString().Trim(separator);
		}

		public static string ToSnakeCase(this string instance) => ToSeparatedCase(instance, '_');
		public static string ToKebabCase(this string instance) => ToSeparatedCase(instance, '-');

		public static int RFind(this string instance, string what, int from = -1, bool caseSensitive = true)
		{
			if (from == -1)
				from = instance.Length - 1;

			return instance.LastIndexOf(
				what,
				from,
				caseSensitive ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase
			);
		}

		public static int RFind(this string instance, char what, int from = -1, bool caseSensitive = true)
		{
			if (from == -1)
				from = instance.Length - 1;

			return caseSensitive
				? instance.LastIndexOf(what, from)
				: CultureInfo.InvariantCulture.CompareInfo.LastIndexOf(instance, what, from, CompareOptions.OrdinalIgnoreCase);
		}

		public static string Substr(this string instance, int from, int len)
		{
			int max = instance.Length - from;
			return instance.Substring(from, len > max ? max : len);
		}

		public static int ToInt(this string instance)
		{
			return int.Parse(instance, CultureInfo.InvariantCulture);
		}

		public static int CasecmpTo(this string instance, string to)
		{
#pragma warning disable CA1309
			return string.Compare(instance, to, ignoreCase: false, null);
#pragma warning restore CA1309
		}

		[Obsolete("Use string.Compare instead.")]
		public static int CompareTo(this string instance, string to, bool caseSensitive = true)
		{
#pragma warning disable CA1309
			return string.Compare(instance, to, ignoreCase: !caseSensitive, null);
#pragma warning restore CA1309
		}

		public static string GetExtension(this string instance)
		{
			int pos = instance.RFind(".");

			if (pos < 0 || pos < Math.Max(instance.RFind("/"), instance.RFind("\\")))
				return string.Empty;

			return instance.Substring(pos + 1);
		}

		public static int Find(this string instance, string what, int from = 0, bool caseSensitive = true)
		{
			return instance.IndexOf(what, from, caseSensitive ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase);
		}

		public static int Find(this string instance, char what, int from = 0, bool caseSensitive = true)
		{
			if (caseSensitive)
				return instance.IndexOf(what, from);

			return CultureInfo.InvariantCulture.CompareInfo.IndexOf(instance, what, from, CompareOptions.OrdinalIgnoreCase);
		}

		public static int FindN(this string instance, string what, int from = 0)
		{
			return instance.IndexOf(what, from, StringComparison.OrdinalIgnoreCase);
		}

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
				if (instance.StartsWith('/'))
				{
					rs = instance.Substring(1);
					directory = "/";
				}
				else
				{
					rs = instance;
				}
			}

			int sep = Mathf.Max(rs.RFind("/"), rs.RFind("\\"));

			if (sep == -1)
				return directory;

			return directory + rs.Substr(0, sep);
		}

		public static string GetBaseName(this string instance)
		{
			int index = instance.RFind(".");

			if (index > 0)
				return instance.Substring(0, index);

			return instance;
		}

		public static string GetFile(this string instance)
		{
			int sep = Mathf.Max(instance.RFind("/"), instance.RFind("\\"));

			if (sep == -1)
				return instance;

			return instance.Substring(sep + 1);
		}

		public static string GetStringFromAscii(this byte[] bytes)
		{
			return Encoding.ASCII.GetString(bytes);
		}

		public static string GetStringFromUtf16(this byte[] bytes)
		{
			return Encoding.Unicode.GetString(bytes);
		}

		public static string GetStringFromUtf32(this byte[] bytes)
		{
			return Encoding.UTF32.GetString(bytes);
		}

		public static string GetStringFromUtf8(this byte[] bytes)
		{
			return Encoding.UTF8.GetString(bytes);
		}

		public static uint Hash(this string instance)
		{
			uint hash = 5381;

			foreach (uint c in instance)
			{
				hash = (hash << 5) + hash + c;
			}

			return hash;
		}

		public static byte[] HexDecode(this string instance)
		{
			if (instance.Length % 2 != 0)
			{
				throw new ArgumentException("Hexadecimal string of uneven length.", nameof(instance));
			}

			int len = instance.Length / 2;
			byte[] ret = new byte[len];

			for (int i = 0; i < len; i++)
			{
				ret[i] = (byte)int.Parse(instance.AsSpan(i * 2, 2), NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture);
			}

			return ret;
		}

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

		public static string HexEncode(this byte[] bytes)
		{
			string ret = string.Empty;

			foreach (byte b in bytes)
			{
				ret += b.HexEncode();
			}

			return ret;
		}

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

			if (instance.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
			{
				instance = instance.Substring(2);
			}

			return sign * int.Parse(instance, NumberStyles.HexNumber, CultureInfo.InvariantCulture);
		}

		public static string Indent(this string instance, string prefix)
		{
			var sb = new StringBuilder();
			int lineStart = 0;

			for (int i = 0; i < instance.Length; i++)
			{
				char c = instance[i];
				if (c == '\n')
				{
					if (i == lineStart)
					{
						sb.Append(c);
					}
					else
					{
						sb.Append(prefix);
						sb.Append(instance.AsSpan(lineStart, i - lineStart + 1));
					}

					lineStart = i + 1;
				}
			}

			if (lineStart != instance.Length)
			{
				sb.Append(prefix);
				sb.Append(instance.AsSpan(lineStart));
			}

			return sb.ToString();
		}

		public static bool IsAbsolutePath(this string instance)
		{
			if (string.IsNullOrEmpty(instance))
				return false;
			else if (instance.Length > 1)
				return instance[0] == '/' || instance[0] == '\\' || instance.Contains(":/", StringComparison.Ordinal) || instance.Contains(":\\", StringComparison.Ordinal);
			else
				return instance[0] == '/' || instance[0] == '\\';
		}

		public static bool IsRelativePath(this string instance)
		{
			return !IsAbsolutePath(instance);
		}

		public static bool IsSubsequenceOf(this string instance, string text, bool caseSensitive = true)
		{
			int len = instance.Length;

			if (len == 0)
				return true;

			if (len > text.Length)
				return false;

			int source = 0;
			int target = 0;

			while (source < len && target < text.Length)
			{
				bool match;

				if (!caseSensitive)
				{
					char sourcec = char.ToLowerInvariant(instance[source]);
					char targetc = char.ToLowerInvariant(text[target]);
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

		public static bool IsSubsequenceOfN(this string instance, string text)
		{
			return instance.IsSubsequenceOf(text, caseSensitive: false);
		}

		private static readonly char[] _invalidFileNameCharacters = { ':', '/', '\\', '?', '*', '"', '|', '%', '<', '>' };

		public static bool IsValidFileName(this string instance)
		{
			var stripped = instance.Trim();
			if (instance != stripped)
				return false;

			if (string.IsNullOrEmpty(stripped))
				return false;

			return instance.IndexOfAny(_invalidFileNameCharacters) == -1;
		}

		public static bool IsValidFloat(this string instance)
		{
			return float.TryParse(instance, out _);
		}

		public static bool IsValidHexNumber(this string instance, bool withPrefix = false)
		{
			if (string.IsNullOrEmpty(instance))
				return false;

			int from = 0;
			if (instance.Length != 1 && (instance[0] == '+' || instance[0] == '-'))
			{
				from++;
			}

			if (withPrefix)
			{
				if (instance.Length < 3)
					return false;
				if (instance[from] != '0' || (instance[from + 1] != 'x' && instance[from + 1] != 'X'))
					return false;
				from += 2;
			}

			if (from == instance.Length)
				return false;

			for (int i = from; i < instance.Length; i++)
			{
				char c = instance[i];
				if (char.IsAsciiHexDigit(c))
					continue;

				return false;
			}

			return true;
		}

		public static bool IsValidHtmlColor(this string instance)
		{
			return Color.HtmlIsValid(instance);
		}

		public static bool IsValidIdentifier(this string instance)
		{
			int len = instance.Length;

			if (len == 0)
				return false;

			if (instance[0] >= '0' && instance[0] <= '9')
				return false;

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

		public static bool IsValidInt(this string instance)
		{
			return int.TryParse(instance, out _);
		}

		public static bool IsValidIPAddress(this string instance)
		{
			if (instance.Contains(':', StringComparison.Ordinal))
			{
				string[] ip = instance.Split(':');

				for (int i = 0; i < ip.Length; i++)
				{
					string n = ip[i];
					if (n.Length == 0)
						continue;

					if (!n.IsValidHexNumber(withPrefix: false))
						return false;

					long nint = n.HexToInt();
					if (nint < 0 || nint > 0xffff)
						return false;
				}
			}
			else
			{
				string[] ip = instance.Split('.');

				if (ip.Length != 4)
					return false;

				for (int i = 0; i < ip.Length; i++)
				{
					string n = ip[i];
					if (!n.IsValidInt())
						return false;

					int val = n.ToInt();
					if (val < 0 || val > 255)
						return false;
				}
			}

			return true;
		}

		public static string JSONEscape(this string instance)
		{
			var sb = new StringBuilder(instance);

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

		public static string Left(this string instance, int pos)
		{
			if (pos <= 0)
				return string.Empty;

			if (pos >= instance.Length)
				return instance;

			return instance.Substring(0, pos);
		}
	}
}
