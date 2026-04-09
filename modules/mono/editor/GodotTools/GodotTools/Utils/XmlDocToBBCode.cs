using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Xml.Linq;

namespace GodotTools.Utils
{
    internal static class XmlDocToBBCode
    {
        internal static string GetTypeDocumentationId(Type type)
        {
            Type xmlType = type.IsConstructedGenericType ? type.GetGenericTypeDefinition() : type;
            string fullName = xmlType.FullName ?? throw new InvalidOperationException($"Unable to resolve full name for type '{type}'.");
            return $"T:{fullName.Replace('+', '.')}";
        }

        internal static string GetPropertyDocumentationId(System.Reflection.PropertyInfo property)
            => $"P:{GetTypeDocumentationId(property.DeclaringType!).Substring(2)}.{property.Name}";

        internal static string GetFieldDocumentationId(System.Reflection.FieldInfo field)
            => $"F:{GetTypeDocumentationId(field.DeclaringType!).Substring(2)}.{field.Name}";

        internal static string? ResolveCrefToDocumentationId(
            Dictionary<string, XElement> members,
            Type contextType,
            string? rawCref)
        {
            if (string.IsNullOrWhiteSpace(rawCref))
                return null;

            string cref = rawCref.Trim();

            if (cref.Contains(':'))
                return members.ContainsKey(cref) ? cref : null;

            string contextTypeName = GetTypeDocumentationId(contextType).Substring(2);

            static IEnumerable<string> PrefixedCandidates(string value)
            {
                yield return "T:" + value;
                yield return "M:" + value;
                yield return "P:" + value;
                yield return "F:" + value;
                yield return "E:" + value;
            }

            if (cref.Contains('('))
            {
                foreach (string candidate in PrefixedCandidates(contextTypeName + "." + cref))
                {
                    if (members.ContainsKey(candidate))
                        return candidate;
                }

                foreach (string candidate in PrefixedCandidates(cref))
                {
                    if (members.ContainsKey(candidate))
                        return candidate;
                }

                return null;
            }

            foreach (string candidate in PrefixedCandidates(cref))
            {
                if (members.ContainsKey(candidate))
                    return candidate;
            }

            foreach (string candidate in PrefixedCandidates(contextTypeName + "." + cref))
            {
                if (members.ContainsKey(candidate))
                    return candidate;
            }

            return null;
        }

        internal static string RenderNodesToBbCode(Dictionary<string, XElement> members, Type contextType, IEnumerable<XNode> nodes)
        {
            var sb = new StringBuilder();

            void Render(XNode node)
            {
                switch (node)
                {
                    case XCData cdata:
                        sb.Append(cdata.Value);
                        break;
                    case XText text:
                        sb.Append(text.Value);
                        break;
                    case XElement element:
                    {
                        string elementName = element.Name.LocalName;

                        switch (elementName)
                        {
                            case "c":
                                sb.Append("[code]");
                                foreach (XNode child in element.Nodes())
                                    Render(child);
                                sb.Append("[/code]");
                                break;
                            case "code":
                                sb.Append("[codeblocks][csharp]");
                                sb.Append(element.Value.Trim());
                                sb.Append("[/csharp][/codeblocks]");
                                break;
                            case "para":
                                sb.AppendLine();
                                foreach (XNode child in element.Nodes())
                                    Render(child);
                                sb.AppendLine();
                                break;
                            case "see":
                            {
                                string? href = element.Attribute("href")?.Value;
                                if (!string.IsNullOrWhiteSpace(href))
                                {
                                    string label = element.Value.Trim();
                                    if (string.IsNullOrEmpty(label))
                                        label = href;
                                    sb.Append("[url=").Append(href).Append(']').Append(label).Append("[/url]");
                                    break;
                                }

                                string? langword = element.Attribute("langword")?.Value;
                                if (!string.IsNullOrWhiteSpace(langword))
                                {
                                    sb.Append("[code]").Append(langword).Append("[/code]");
                                    break;
                                }

                                string? cref = element.Attribute("cref")?.Value;
                                if (!string.IsNullOrWhiteSpace(cref))
                                {
                                    string cleaned = cref.Replace('{', '<').Replace('}', '>');
                                    sb.Append("[code]").Append(cleaned).Append("[/code]");
                                }

                                break;
                            }
                            case "paramref":
                            {
                                string? name = element.Attribute("name")?.Value;
                                if (!string.IsNullOrEmpty(name))
                                    sb.Append("[param ").Append(name).Append(']');
                                break;
                            }
                            case "typeparamref":
                            {
                                string? name = element.Attribute("name")?.Value;
                                if (!string.IsNullOrEmpty(name))
                                    sb.Append("[code]").Append(name).Append("[/code]");
                                break;
                            }
                            case "inheritdoc":
                            {
                                string? targetId = ResolveCrefToDocumentationId(members, contextType, element.Attribute("cref")?.Value);
                                if (targetId != null && members.TryGetValue(targetId, out XElement? inheritedMember))
                                {
                                    XElement? inheritedSummary = inheritedMember.Element("summary");
                                    if (inheritedSummary != null)
                                    {
                                        foreach (XNode child in inheritedSummary.Nodes())
                                            Render(child);
                                    }
                                }
                                break;
                            }
                            default:
                                foreach (XNode child in element.Nodes())
                                    Render(child);
                                break;
                        }

                        break;
                    }
                }
            }

            foreach (XNode node in nodes)
                Render(node);

            string rendered = sb.ToString();
            if (rendered.Length == 0)
                return rendered;

            // Trim leading whitespace on each line because XML doc has indentation
            string[] lines = rendered.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.None);

            StringBuilder sbNormalized = new StringBuilder();
            bool firstLine = true;
            foreach (string line in lines)
            {
                string trimmedLine = line.TrimStart();
                if (string.IsNullOrWhiteSpace(trimmedLine))
                    continue;

                if (!firstLine)
                    sbNormalized.Append("[br]");

                sbNormalized.Append(trimmedLine);
                firstLine = false;
            }

            return sbNormalized.ToString().Trim();
        }

        internal static string? TryGetTagBbCode(
            Dictionary<string, XElement> members,
            Type contextType,
            string memberId,
            string tagName,
            Func<string?>? implicitInheritResolver = null,
            HashSet<string>? visited = null)
        {
            visited ??= new HashSet<string>(StringComparer.Ordinal);
            string recursionKey = memberId + "|" + tagName;

            if (!visited.Add(recursionKey))
                return null;

            if (!members.TryGetValue(memberId, out XElement? memberElement))
                return null;

            XElement? tagElement = memberElement.Element(tagName);
            if (tagElement != null)
            {
                string text = RenderNodesToBbCode(members, contextType, tagElement.Nodes());
                if (!string.IsNullOrWhiteSpace(text))
                    return text;
            }

            string? inheritedTarget = ResolveCrefToDocumentationId(members, contextType, memberElement.Element("inheritdoc")?.Attribute("cref")?.Value);
            inheritedTarget ??= implicitInheritResolver?.Invoke();

            if (string.IsNullOrEmpty(inheritedTarget))
                return null;

            return TryGetTagBbCode(members, contextType, inheritedTarget, tagName, null, visited);
        }
    }
}
