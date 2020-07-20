using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace GodotTools.ProjectEditor
{
    public static class IdentifierUtils
    {
        public static string SanitizeQualifiedIdentifier(string qualifiedIdentifier, bool allowEmptyIdentifiers)
        {
            if (string.IsNullOrEmpty(qualifiedIdentifier))
                throw new ArgumentException($"{nameof(qualifiedIdentifier)} cannot be empty", nameof(qualifiedIdentifier));

            string[] identifiers = qualifiedIdentifier.Split('.');

            for (int i = 0; i < identifiers.Length; i++)
            {
                identifiers[i] = SanitizeIdentifier(identifiers[i], allowEmpty: allowEmptyIdentifiers);
            }

            return string.Join(".", identifiers);
        }

        /// <summary>
        /// Skips invalid identifier characters including decimal digit numbers at the start of the identifier.
        /// </summary>
        private static void SkipInvalidCharacters(string source, int startIndex, StringBuilder outputBuilder)
        {
            for (int i = startIndex; i < source.Length; i++)
            {
                char @char = source[i];

                switch (char.GetUnicodeCategory(@char))
                {
                    case UnicodeCategory.UppercaseLetter:
                    case UnicodeCategory.LowercaseLetter:
                    case UnicodeCategory.TitlecaseLetter:
                    case UnicodeCategory.ModifierLetter:
                    case UnicodeCategory.LetterNumber:
                    case UnicodeCategory.OtherLetter:
                        outputBuilder.Append(@char);
                        break;
                    case UnicodeCategory.NonSpacingMark:
                    case UnicodeCategory.SpacingCombiningMark:
                    case UnicodeCategory.ConnectorPunctuation:
                    case UnicodeCategory.DecimalDigitNumber:
                        // Identifiers may start with underscore
                        if (outputBuilder.Length > startIndex || @char == '_')
                            outputBuilder.Append(@char);
                        break;
                }
            }
        }

        public static string SanitizeIdentifier(string identifier, bool allowEmpty)
        {
            if (string.IsNullOrEmpty(identifier))
            {
                if (allowEmpty)
                    return "Empty"; // Default value for empty identifiers

                throw new ArgumentException($"{nameof(identifier)} cannot be empty if {nameof(allowEmpty)} is false", nameof(identifier));
            }

            if (identifier.Length > 511)
                identifier = identifier.Substring(0, 511);

            var identifierBuilder = new StringBuilder();
            int startIndex = 0;

            if (identifier[0] == '@')
            {
                identifierBuilder.Append('@');
                startIndex += 1;
            }

            SkipInvalidCharacters(identifier, startIndex, identifierBuilder);

            if (identifierBuilder.Length == startIndex)
            {
                // All characters were invalid so now it's empty. Fill it with something.
                identifierBuilder.Append("Empty");
            }

            identifier = identifierBuilder.ToString();

            if (identifier[0] != '@' && IsKeyword(identifier, anyDoubleUnderscore: true))
                identifier = '@' + identifier;

            return identifier;
        }

        static bool IsKeyword(string value, bool anyDoubleUnderscore)
        {
            // Identifiers that start with double underscore are meant to be used for reserved keywords.
            // Only existing keywords are enforced, but it may be useful to forbid any identifier
            // that begins with double underscore to prevent issues with future C# versions.
            if (anyDoubleUnderscore)
            {
                if (value.Length > 2 && value[0] == '_' && value[1] == '_' && value[2] != '_')
                    return true;
            }
            else
            {
                if (DoubleUnderscoreKeywords.Contains(value))
                    return true;
            }

            return Keywords.Contains(value);
        }

        private static readonly HashSet<string> DoubleUnderscoreKeywords = new HashSet<string>
        {
            "__arglist",
            "__makeref",
            "__reftype",
            "__refvalue",
        };

        private static readonly HashSet<string> Keywords = new HashSet<string>
        {
            "as",
            "do",
            "if",
            "in",
            "is",
            "for",
            "int",
            "new",
            "out",
            "ref",
            "try",
            "base",
            "bool",
            "byte",
            "case",
            "char",
            "else",
            "enum",
            "goto",
            "lock",
            "long",
            "null",
            "this",
            "true",
            "uint",
            "void",
            "break",
            "catch",
            "class",
            "const",
            "event",
            "false",
            "fixed",
            "float",
            "sbyte",
            "short",
            "throw",
            "ulong",
            "using",
            "where",
            "while",
            "yield",
            "double",
            "extern",
            "object",
            "params",
            "public",
            "return",
            "sealed",
            "sizeof",
            "static",
            "string",
            "struct",
            "switch",
            "typeof",
            "unsafe",
            "ushort",
            "checked",
            "decimal",
            "default",
            "finally",
            "foreach",
            "partial",
            "private",
            "virtual",
            "abstract",
            "continue",
            "delegate",
            "explicit",
            "implicit",
            "internal",
            "operator",
            "override",
            "readonly",
            "volatile",
            "interface",
            "namespace",
            "protected",
            "unchecked",
            "stackalloc",
        };
    }
}
