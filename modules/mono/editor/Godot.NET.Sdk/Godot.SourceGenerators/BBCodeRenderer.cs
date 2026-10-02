using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace Godot.SourceGenerators
{
    internal static class BBCodeRenderer
    {
        private static readonly Regex SanitizeXmlDocumentationRegex = new(@" *\/\/\/ *", RegexOptions.Compiled);

        /// <summary>
        /// Render the specified raw C# XML documentation comment to BBCode.
        /// </summary>
        /// <returns>The rendered BBCode string.</returns>
        /// <exception cref="FormatException">Raised if the XML documentation is malformed.</exception>
        public static void RenderXmlDocumentationToBBCode(string xmlDocumentation, out string briefDescription,
            out string fullDescription)
        {
            var sanitizedDocumentation =
                SanitizeXmlDocumentationRegex
                .Replace(xmlDocumentation, string.Empty)
                .Replace("<c>", "[code]")
                .Replace("</c>", "[/code]");

            XmlNodeCollection root;
            try
            {
                root = Parse(sanitizedDocumentation);
            }
            catch (Exception e)
            {
                throw new FormatException("Failed to parse XML documentation.", e);
            }

            var briefDescriptionBuilder = new StringBuilder();
            root.RenderBriefBbCode(briefDescriptionBuilder);
            var fullDescriptionBuilder = new StringBuilder();
            root.RenderBbCode(fullDescriptionBuilder);

            briefDescription = briefDescriptionBuilder.ToString().Trim().ReplaceLineEndings("[br]").Replace("\"", "\"\"");
            fullDescription = fullDescriptionBuilder.ToString().Trim().ReplaceLineEndings("[br]").Replace("\"", "\"\"");
        }

        private enum NodeType
        {
            Root,
            Text,
            Summary,
            Remarks,
            Para,
            See,
            Code,
            Param,
            Returns,
            ParamRef,
            Exception,
            TypeParam,
            TypeParamRef,
            Unknown,
        }

        private class UnknownTypeXmlNode : XmlNode
        {
            public UnknownTypeXmlNode(string content) : base(NodeType.Unknown)
            {
                Content = content;
            }

            public override string ToString() => $"[Unknown: {Content}]";
            public override void RenderBbCode(StringBuilder builder) => builder.Append(Content);
            public string Content { get; }
        }

        private class TypeParamRefXmlNode : XmlNode
        {
            public TypeParamRefXmlNode(string name) : base(NodeType.TypeParamRef)
            {
                Name = name;
            }

            public override string ToString() => $"[TypeParamRef: {Name}]";
            public override void RenderBbCode(StringBuilder builder) => builder.Append($"[code]{Name}[/code]");
            public string Name { get; }
        }

        private class ParamRefXmlNode : XmlNode
        {
            public ParamRefXmlNode(string name) : base(NodeType.ParamRef)
            {
                Name = name;
            }

            public override string ToString() => $"[ParamRef: {Name}]";
            public override void RenderBbCode(StringBuilder builder) => builder.Append($"[param {Name}]");
            public string Name { get; }
        }

        private class TypeParamXmlNode : XmlNodeCollection
        {
            public TypeParamXmlNode(string name, List<XmlNode> nodes) : base(NodeType.TypeParam, nodes)
            {
                Name = name;
            }

            public override string ToString() => $"TypeParam: {Name} {string.Join(" ", Nodes)}";

            public override void RenderBbCode(StringBuilder builder)
            {
                builder.Append($"[b]{Name}[/b]: ");
                base.RenderBbCode(builder);
            }

            public string Name { get; }
        }

        private class ParamXmlNode : XmlNodeCollection
        {
            public ParamXmlNode(string name, List<XmlNode> nodes) : base(NodeType.Param, nodes)
            {
                Name = name;
            }

            public override string ToString() => $"Param: {Name} {string.Join(" ", Nodes)}";

            public override void RenderBbCode(StringBuilder builder)
            {
                builder.Append($"[b]{Name}[/b]: ");
                base.RenderBbCode(builder);
            }

            public string Name { get; }
        }

        private class ExceptionXmlNode : XmlNodeCollection
        {
            public ExceptionXmlNode(string cref, List<XmlNode> nodes) : base(NodeType.Exception, nodes)
            {
                Cref = cref;
            }

            public override string ToString() => $"Exception: {Cref} {string.Join(" ", Nodes)}";

            public override void RenderBbCode(StringBuilder builder)
            {
                builder.Append($"[b]{Cref}[/b]: ");
                base.RenderBbCode(builder);
            }

            public string Cref { get; }
        }

        private class CrefXmlNode : SeeXmlNode
        {
            public CrefXmlNode(string cref)
            {
                Cref = cref;
            }

            private string ToCleanCref() => Cref.Replace('{', '<').Replace('}', '>');
            public override string ToString() => $"[Cref: {ToCleanCref()}]";
            public override void RenderBbCode(StringBuilder builder) => builder.Append($"[code]{ToCleanCref()}[/code]");
            public string Cref { get; }
        }

        private class HrefXmlNode : SeeXmlNode
        {
            public HrefXmlNode(string href, string text)
            {
                Href = href;
                Text = text;
            }

            public override string ToString() => $"[Href: {Href} - {Text}]";
            public override void RenderBbCode(StringBuilder builder) => builder.Append($"[url={Href}]{Text}[/url]");
            public string Href { get; }
            public string Text { get; }
        }

        private abstract class SeeXmlNode : XmlNode
        {
            protected SeeXmlNode() : base(NodeType.See)
            {
            }
        }

        private class TextXmlNode : XmlNode
        {
            public TextXmlNode(string text) : base(NodeType.Text)
            {
                Text = text;
            }

            public override string ToString() => Text;
            public override void RenderBbCode(StringBuilder builder) => builder.Append(Text);
            public string Text { get; }
        }

        private class XmlNodeCollection : XmlNode
        {
            public XmlNodeCollection(NodeType type, List<XmlNode> nodes) : base(type)
            {
                Nodes = nodes;
            }

            public override string ToString()
            {
                var builder = new StringBuilder();
                builder.Append("--- ").Append(Type).AppendLine(" ---");
                var isFirst = true;
                foreach (var node in Nodes)
                {
                    if (Type == NodeType.Root) builder.AppendLine(node.ToString());
                    else
                    {
                        if (isFirst) isFirst = false;
                        else builder.Append(' ');
                        builder.Append(node);
                    }
                }

                return builder.ToString();
            }

            private enum RenderMode
            {
                Paragraph,
                List,
            }

            private void RenderCollection(StringBuilder builder, RenderMode renderMode) =>
                RenderCollectionCore(builder, Nodes, renderMode);

            private static void RenderCollectionCore(StringBuilder builder, IEnumerable<XmlNode> nodes, RenderMode renderMode, string prefix = "")
            {
                switch (renderMode)
                {
                    case RenderMode.Paragraph:
                    {
                        var isFirst = true;
                        builder.Append(prefix);
                        foreach (var node in nodes)
                        {
                            if (isFirst) isFirst = false;
                            else
                            {
                                if (node is not TextXmlNode textXmlNode || !textXmlNode.Text.StartsWith("."))
                                    builder.Append(' ');
                            }

                            node.RenderBbCode(builder);
                        }

                        break;
                    }
                    case RenderMode.List:
                    {
                        var isFirst = true;
                        foreach (var node in nodes)
                        {
                            if (isFirst) isFirst = false;
                            else builder.AppendLine();
                            builder.Append(prefix);
                            node.RenderBbCode(builder);
                        }

                        break;
                    }
                    default:
                        throw new ArgumentOutOfRangeException(nameof(renderMode), renderMode, null);
                }
            }

            public void RenderBriefBbCode(StringBuilder builder)
            {
                RenderCollectionCore(builder, Nodes.Where(x => x.Type == NodeType.Summary), RenderMode.List);
            }

            public override void RenderBbCode(StringBuilder builder)
            {
                switch (Type)
                {
                    case NodeType.Root:
                        var ordered = Nodes.GroupBy(x => x.Type).ToDictionary(x => x.Key);
                        if (ordered.TryGetValue(NodeType.Summary, out var summary))
                        {
                            RenderCollectionCore(builder, summary, RenderMode.List);
                            builder.AppendLine();
                        }

                        if (ordered.TryGetValue(NodeType.Param, out var para))
                        {
                            builder.AppendLine().AppendLine("[b]Parameters:[/b]");
                            RenderCollectionCore(builder, para, RenderMode.List, " • ");
                            builder.AppendLine();
                        }

                        if (ordered.TryGetValue(NodeType.Returns, out var returns))
                        {
                            builder.AppendLine().AppendLine("[b]Returns:[/b]");
                            RenderCollectionCore(builder, returns, RenderMode.List, " • ");
                            builder.AppendLine();
                        }

                        if (ordered.TryGetValue(NodeType.Exception, out var exception))
                        {
                            builder.AppendLine().AppendLine("[b]Exceptions:[/b]");
                            RenderCollectionCore(builder, exception, RenderMode.List, " • ");
                            builder.AppendLine();
                        }

                        if (ordered.TryGetValue(NodeType.Remarks, out var remarks))
                        {
                            builder.AppendLine().AppendLine("[b]Note:[/b]");
                            RenderCollectionCore(builder, remarks, RenderMode.List);
                            builder.AppendLine();
                        }

                        break;
                    case NodeType.Para:
                        builder.AppendLine();
                        RenderCollection(builder, RenderMode.Paragraph);
                        builder.AppendLine();
                        break;
                    case NodeType.Code:
                        builder.Append("[codeblocks][csharp]");
                        RenderCollection(builder, RenderMode.List);
                        builder.Append("[/csharp][/codeblocks]");
                        break;
                    default:
                        RenderCollection(builder, RenderMode.Paragraph);
                        break;
                }
            }

            public List<XmlNode> Nodes { get; }
        }

        private abstract class XmlNode
        {
            protected XmlNode(NodeType type)
            {
                Type = type;
            }

            public abstract void RenderBbCode(StringBuilder builder);
            public NodeType Type { get; }
        }

        private static XmlNodeCollection Parse(string documentation)
        {
            var root = new XmlNodeCollection(NodeType.Root, new List<XmlNode>());
            ParseNodes(documentation, false, root.Nodes);
            return root;
        }

        private static string ConditionalFormat(string sourceString, bool replaceLineEndings)
        {
            return (replaceLineEndings ? sourceString.ReplaceLineEndings(" ") : sourceString)
                .Replace("<br/>", "\n")
                .Trim();
        }

        private static string ReplaceLineEndings(this string input, string replacement)
        {
            return input.Replace("\r\n", replacement)
                       .Replace("\r", replacement)
                       .Replace("\n", replacement);
        }

        private static void ParseNodes(string content, bool replaceLineEndings, List<XmlNode> nodes)
        {
            var position = 0;

            while (position < content.Length)
            {
                var tagStart = content.IndexOf('<', position);

                if (tagStart == -1)
                {
                    // No more tags, add remaining text
                    var remainingText = content.Substring(position).Trim();
                    if (!string.IsNullOrEmpty(remainingText))
                    {
                        nodes.Add(new TextXmlNode(ConditionalFormat(remainingText, replaceLineEndings)));
                    }

                    break;
                }

                // Add text before tag
                if (tagStart > position)
                {
                    var textBefore = content.Substring(position, tagStart - position).Trim();
                    if (!string.IsNullOrEmpty(textBefore))
                    {
                        nodes.Add(new TextXmlNode(ConditionalFormat(textBefore, replaceLineEndings)));
                    }
                }

                var tagEnd = content.IndexOf('>', tagStart);
                if (tagEnd == -1) break;

                var tagContent = content.Substring(tagStart + 1, tagEnd - tagStart - 1);

                // Check if it's a self-closing tag or see tag
                if (tagContent.EndsWith("/") || tagContent.StartsWith("see ") || tagContent.StartsWith("paramref ") ||
                    tagContent.StartsWith("typeparamref "))
                {
                    if (tagContent.StartsWith("see "))
                    {
                        var crefMatch = ExtractCrefRegex.Match(tagContent);
                        if (crefMatch.Success)
                        {
                            nodes.Add(new CrefXmlNode(ConditionalFormat(crefMatch.Groups[1].Value, true)));
                        }
                    }
                    else if (tagContent.StartsWith("paramref "))
                    {
                        var nameMatch = ExtractNameRegex.Match(tagContent);
                        if (nameMatch.Success)
                        {
                            nodes.Add(new ParamRefXmlNode(ConditionalFormat(nameMatch.Groups[1].Value, true)));
                        }
                    }
                    else if (tagContent.StartsWith("typeparamref "))
                    {
                        var nameMatch = ExtractNameRegex.Match(tagContent);
                        if (nameMatch.Success)
                        {
                            nodes.Add(new TypeParamRefXmlNode(ConditionalFormat(nameMatch.Groups[1].Value, true)));
                        }
                    }
                    else
                    {
                        // Handle unknown self-closing tags
                        nodes.Add(new UnknownTypeXmlNode($"<{tagContent}>"));
                    }

                    position = tagEnd + 1;
                    continue;
                }

                var tagName = tagContent.Split(' ')[0];
                var nodeType = tagName switch
                {
                    "summary" => NodeType.Summary,
                    "remarks" => NodeType.Remarks,
                    "para" => NodeType.Para,
                    "code" => NodeType.Code,
                    "param" => NodeType.Param,
                    "returns" => NodeType.Returns,
                    "exception" => NodeType.Exception,
                    "typeparam" => NodeType.TypeParam,
                    "a" => NodeType.See,
                    _ => NodeType.Unknown
                };

                // Find closing tag
                var closingTag = $"</{tagName}>";
                var closingTagStart = content.IndexOf(closingTag, tagEnd + 1, StringComparison.Ordinal);

                if (closingTagStart == -1)
                {
                    position = tagEnd + 1;
                    continue;
                }

                var innerContent = content.Substring(tagEnd + 1, closingTagStart - tagEnd - 1);
                var childNodes = new List<XmlNode>();

                // Recursively parse inner content
                ParseNodes(innerContent, nodeType switch
                {
                    NodeType.Summary => true,
                    NodeType.Remarks => true,
                    NodeType.Para => true,
                    NodeType.Code => false,
                    NodeType.Param => true,
                    NodeType.Returns => true,
                    NodeType.Exception => true,
                    NodeType.TypeParam => true,
                    NodeType.See => true,
                    _ => false
                }, childNodes);

                switch (nodeType)
                {
                    case NodeType.Param:
                    {
                        var nameMatch = ExtractNameRegex.Match(tagContent);
                        var paramName = nameMatch.Success ? nameMatch.Groups[1].Value : string.Empty;
                        nodes.Add(new ParamXmlNode(paramName, childNodes));
                        break;
                    }
                    case NodeType.Exception:
                    {
                        var crefMatch = ExtractCrefRegex.Match(tagContent);
                        var crefValue = crefMatch.Success ? crefMatch.Groups[1].Value : string.Empty;
                        nodes.Add(new ExceptionXmlNode(crefValue, childNodes));
                        break;
                    }
                    case NodeType.TypeParam:
                    {
                        var nameMatch = ExtractNameRegex.Match(tagContent);
                        var paramName = nameMatch.Success ? nameMatch.Groups[1].Value : string.Empty;
                        nodes.Add(new TypeParamXmlNode(paramName, childNodes));
                        break;
                    }
                    case NodeType.See when tagName == "a":
                    {
                        var hrefMatch = ExtractHrefRegex.Match(tagContent);
                        var hrefValue = hrefMatch.Success ? hrefMatch.Groups[1].Value : string.Empty;
                        var linkText = innerContent.Trim();
                        nodes.Add(new HrefXmlNode(hrefValue, linkText));
                        break;
                    }
                    case NodeType.Unknown:
                    {
                        // Handle unknown tags - store the full tag with content
                        var fullTag = content.Substring(tagStart, closingTagStart + closingTag.Length - tagStart);
                        nodes.Add(new UnknownTypeXmlNode(fullTag));
                        break;
                    }
                    default:
                        nodes.Add(new XmlNodeCollection(nodeType, childNodes));
                        break;
                }

                position = closingTagStart + closingTag.Length;
            }
        }

        private static readonly Regex ExtractCrefRegex = new(@"cref=""([^""]+)""");

        private static readonly Regex ExtractNameRegex = new(@"name=""([^""]+)""");

        private static readonly Regex ExtractHrefRegex = new(@"href=""([^""]+)""");

    }
}
