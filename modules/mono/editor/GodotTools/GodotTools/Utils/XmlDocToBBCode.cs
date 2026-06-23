using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
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

        /// <summary>
        /// Strips an XML documentation <c>cref</c> value down to a human-readable name for
        /// plain-text fallback display. Removes the member-kind prefix (<c>T:</c>, <c>M:</c>,
        /// <c>P:</c>, <c>F:</c>, <c>E:</c>, or the unresolved <c>!:</c> marker), normalizes
        /// generic braces to angle brackets, and drops any method parameter list.
        /// </summary>
        private static string StripCrefPrefix(string cref)
        {
            string value = cref.Trim();

            // Drop the XML documentation prefix ("T:", "M:", "P:", "F:", "E:", "!:").
            int colon = value.IndexOf(':');
            if (colon >= 0 && colon <= 2)
                value = value.Substring(colon + 1);

            // Drop a method parameter list, e.g. "Ns.Type.Method(System.Int32)".
            int paren = value.IndexOf('(');
            if (paren >= 0)
                value = value.Substring(0, paren);

            // Generic type arguments use {} in cref; show them as <> for readability.
            value = value.Replace('{', '<').Replace('}', '>');

            return value.Trim();
        }

        /// <summary>
        /// Resolves a <c>cref</c> reference to a Godot BBCode cross-reference link when the
        /// target is a member of <paramref name="contextType"/> (the class currently being
        /// documented). Returns <see langword="null"/> for cross-class references, in which
        /// case the caller falls back to plain text.
        /// </summary>
        /// <remarks>
        /// Uses the same bare-tag syntax the Godot documentation itself uses (e.g.
        /// <c>[member Name]</c>, <c>[method Name]</c>, <c>[enum Name]</c>), which
        /// <c>_add_text_to_rt</c> (editor_help.cpp) parses into clickable meta links. Only
        /// members that actually appear on the current class's doc page are linked; anything
        /// else (foreign types, generated helpers like SignalName.X) stays as plain text.
        /// </remarks>
        private static string? TryResolveDocLink(Type contextType, string rawCref)
        {
            // Normalize the cref into a kind prefix (T/M/P/F/E) and a dotted member path.
            // The compiler may supply either a prefixed id ("P:Ns.T.Prop") or a bare name.
            char kind;
            string memberPath;
            string cref = rawCref.Trim();

            if (cref.Length > 2 && cref[1] == ':')
            {
                kind = cref[0];
                memberPath = StripCrefPrefix(cref);
            }
            else
            {
                // Bare name: infer the kind by reflecting on the current type.
                memberPath = cref;
                kind = InferMemberKind(contextType, cref);
            }

            if (kind == '\0')
                return null;

            // For references that include a dotted path, require the path to be a direct
            // member of the current class. We strip the current class prefix; any remaining dot
            // means the target lives on a nested type (e.g. SignalName.StatsChanged) and must
            // not be linked, since those helpers are not part of the documented API surface.
            if (memberPath.Contains('.'))
            {
                string? fullName = contextType.FullName;
                string shortName = contextType.Name;

                string? trimmed = null;
                if (fullName != null && (memberPath == fullName || memberPath.StartsWith(fullName + ".", StringComparison.Ordinal)))
                    trimmed = memberPath == fullName ? string.Empty : memberPath.Substring(fullName.Length + 1);
                else if (memberPath == shortName || memberPath.StartsWith(shortName + ".", StringComparison.Ordinal))
                    trimmed = memberPath == shortName ? string.Empty : memberPath.Substring(shortName.Length + 1);

                // trimmed now holds the member name relative to the current class. If it still
                // contains a dot, the target is on a nested type -> not a direct member -> skip.
                if (trimmed == null || trimmed.Contains('.'))
                    return null;
            }

            string simpleName = GetSimpleName(memberPath);

            switch (kind)
            {
                case 'M':
                    return $"[method {simpleName}]";
                case 'P':
                    return $"[member {simpleName}]";
                case 'E':
                    return $"[signal {simpleName}]";
                case 'F':
                    // If the constant is a value of a nested enum, link to that enum.
                    if (TryResolveEnumValue(contextType, simpleName, out string enumName))
                        return $"[enum {enumName}]";
                    return $"[constant {simpleName}]";
                case 'T':
                    // A nested enum of the current class links via [enum Name].
                    Type? nested = contextType.GetNestedType(simpleName, System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic);
                    if (nested != null && nested.IsEnum)
                        return $"[enum {simpleName}]";
                    return null;
                default:
                    return null;
            }
        }

        /// <summary>
        /// Infers the documentation kind of a bare cref name by reflecting on the current type,
        /// returning 'T'/'M'/'P'/'F'/'E'/'\0'. '\0' means the name was not found or refers to a
        /// source-generator helper that should not be linked (e.g. SignalName/MethodName/PropertyName
        /// nested classes, or members marked EditorBrowsableState.Never).
        /// </summary>
        private static char InferMemberKind(Type contextType, string cref)
        {
            const System.Reflection.BindingFlags Flags =
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic |
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Static |
                System.Reflection.BindingFlags.DeclaredOnly;

            string name = GetSimpleName(cref);

            // Nested type. Skip source-generator helper classes (SignalName, MethodName,
            // PropertyName) and anything hidden from the editor, since those are not part of
            // the documented API surface and links to them would not resolve.
            Type? nestedType = contextType.GetNestedType(name, System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic);
            if (nestedType != null)
            {
                if (IsGeneratedHelperType(nestedType))
                    return '\0';
                return 'T';
            }

            if (contextType.GetProperty(name, Flags) is { } prop && !IsHiddenMember(prop))
                return 'P';

            if (contextType.GetMethod(name, Flags) is { } method && !IsHiddenMember(method))
                return 'M';

            if (contextType.GetField(name, Flags) is { } field && !IsHiddenMember(field))
                return 'F';

            if (contextType.GetEvent(name, Flags) is { } evt && !IsHiddenMember(evt))
                return 'E';

            return '\0';
        }

        /// <summary>
        /// Returns <see langword="true"/> for the source-generator-emitted helper nested types
        /// (SignalName, MethodName, PropertyName) and any type marked EditorBrowsableState.Never,
        /// which must not be linked from documentation.
        /// </summary>
        private static bool IsGeneratedHelperType(Type type)
        {
            if (type.GetCustomAttribute<System.ComponentModel.EditorBrowsableAttribute>(inherit: false)?.State
                == System.ComponentModel.EditorBrowsableState.Never)
                return true;

            // The well-known generated nested classes used to reference exported members.
            return type.Name is "SignalName" or "MethodName" or "PropertyName";
        }

        /// <summary>
        /// Returns <see langword="true"/> for members the source generator marks as hidden from
        /// the editor (EditorBrowsableState.Never), so they are not linked from documentation.
        /// </summary>
        private static bool IsHiddenMember(System.Reflection.MemberInfo member)
        {
            return member.GetCustomAttribute<System.ComponentModel.EditorBrowsableAttribute>(inherit: false)?.State
                == System.ComponentModel.EditorBrowsableState.Never;
        }

        /// <summary>
        /// Returns the simple (last) segment of a dotted member path, with any method
        /// parameter list stripped.
        /// </summary>
        private static string GetSimpleName(string memberPath)
        {
            int paren = memberPath.IndexOf('(');
            string path = paren >= 0 ? memberPath.Substring(0, paren) : memberPath;
            int dot = path.LastIndexOf('.');
            return dot >= 0 ? path.Substring(dot + 1) : path;
        }

        /// <summary>
        /// If <paramref name="simpleName"/> names a static field of one of the current class's
        /// nested enums, returns that enum's name via <paramref name="enumName"/>.
        /// </summary>
        private static bool TryResolveEnumValue(Type contextType, string simpleName, out string enumName)
        {
            enumName = string.Empty;
            Type[] nestedTypes = contextType.GetNestedTypes(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic);
            foreach (Type nested in nestedTypes)
            {
                if (!nested.IsEnum)
                    continue;
                if (nested.GetField(simpleName) != null)
                {
                    enumName = nested.Name;
                    return true;
                }
            }
            return false;
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
                                    // Try to emit a clickable in-page link when the cref targets
                                    // a member of the class currently being documented. Otherwise
                                    // fall back to plain text showing the cleaned-up cref name.
                                    string? link = TryResolveDocLink(contextType, cref);
                                    if (link != null)
                                        sb.Append(link);
                                    else
                                        sb.Append("[code]").Append(StripCrefPrefix(cref)).Append("[/code]");
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

        /// <summary>
        /// Renders a full BBCode description for a documentation member, combining its
        /// <c>summary</c>, <c>remarks</c>, <c>param</c>, <c>returns</c> and <c>exception</c>
        /// tags into a single string using the same sectioned layout as the GDScript
        /// documentation (e.g. "[b]Parameters:[/b]", "[b]Exceptions:[/b]"). This lets C#
        /// XML documentation tags that were previously dropped (remarks, exception, etc.)
        /// surface in the inspector and built-in documentation.
        /// </summary>
        internal static string? TryGetFullDescriptionBbCode(
            Dictionary<string, XElement> members,
            Type contextType,
            string memberId,
            Func<string?>? implicitInheritResolver = null,
            HashSet<string>? visited = null)
        {
            visited ??= new HashSet<string>(StringComparer.Ordinal);
            string recursionKey = memberId + "|fulldesc";
            if (!visited.Add(recursionKey))
                return null;

            if (!members.TryGetValue(memberId, out XElement? memberElement))
                return null;

            var sections = new StringBuilder();

            // summary
            string? summary = TryGetTagBbCode(members, contextType, memberId, "summary", null, visited);
            if (!string.IsNullOrWhiteSpace(summary))
                sections.Append(summary);

            // remarks -> Note section
            string? remarks = TryGetTagBbCode(members, contextType, memberId, "remarks", null, visited);
            if (!string.IsNullOrWhiteSpace(remarks))
            {
                if (sections.Length > 0)
                    sections.Append("[br]");
                sections.Append("[b]Note:[/b] ").Append(remarks);
            }

            // param -> Parameters section
            var paramElements = memberElement.Elements("param").ToList();
            if (paramElements.Count > 0)
            {
                if (sections.Length > 0)
                    sections.Append("[br]");
                sections.Append("[b]Parameters:[/b]");
                foreach (XElement param in paramElements)
                {
                    string paramName = param.Attribute("name")?.Value ?? string.Empty;
                    string paramDesc = RenderNodesToBbCode(members, contextType, param.Nodes());
                    sections.Append("[br] • [b]").Append(paramName).Append("[/b]");
                    if (!string.IsNullOrWhiteSpace(paramDesc))
                        sections.Append(": ").Append(paramDesc);
                }
            }

            // returns -> Returns section
            string? returns = TryGetTagBbCode(members, contextType, memberId, "returns", null, visited);
            if (!string.IsNullOrWhiteSpace(returns))
            {
                if (sections.Length > 0)
                    sections.Append("[br]");
                sections.Append("[b]Returns:[/b][br] • ").Append(returns);
            }

            // exception -> Exceptions section
            var exceptionElements = memberElement.Elements("exception").ToList();
            if (exceptionElements.Count > 0)
            {
                if (sections.Length > 0)
                    sections.Append("[br]");
                sections.Append("[b]Exceptions:[/b]");
                foreach (XElement exception in exceptionElements)
                {
                    string cref = exception.Attribute("cref")?.Value ?? string.Empty;
                    string cleanedCref = cref.Replace('{', '<').Replace('}', '>');
                    string exceptionDesc = RenderNodesToBbCode(members, contextType, exception.Nodes());
                    sections.Append("[br] • [b]").Append(cleanedCref).Append("[/b]");
                    if (!string.IsNullOrWhiteSpace(exceptionDesc))
                        sections.Append(": ").Append(exceptionDesc);
                }
            }

            if (sections.Length > 0)
                return sections.ToString();

            // If this member has nothing of its own, follow inheritdoc / implicit inheritance.
            string? inheritedTarget = ResolveCrefToDocumentationId(members, contextType, memberElement.Element("inheritdoc")?.Attribute("cref")?.Value);
            inheritedTarget ??= implicitInheritResolver?.Invoke();

            if (string.IsNullOrEmpty(inheritedTarget))
                return null;

            return TryGetFullDescriptionBbCode(members, contextType, inheritedTarget, null, visited);
        }
    }
}
