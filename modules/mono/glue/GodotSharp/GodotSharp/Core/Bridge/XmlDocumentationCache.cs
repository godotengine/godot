#nullable enable

using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;
using System.Text.RegularExpressions;
using System.Xml;
using Godot.NativeInterop;

namespace Godot.Bridge
{
    /// <summary>
    /// Lazily reads and caches the XML documentation file emitted next to a managed
    /// assembly (e.g. MyGame.xml beside MyGame.dll), so the <c>/// &lt;summary&gt;</c>
    /// text of exported members can be surfaced in the Godot Inspector. The result is
    /// cached per assembly and invalidated when the XML file's last-write time changes.
    /// </summary>
    internal static class XmlDocumentationCache
    {
        private sealed class Entry
        {
            public DateTime WriteTimeUtc;
            public Dictionary<string, string> Summaries = new(); // doc id ("P:Ns.T.Prop") -> summary
        }

        // Keyed weakly so assemblies stay collectible; a strong Dictionary<Assembly, ...>
        // would pin the collectible load context and break hot-reload (see godot#78513).
        private static readonly ConditionalWeakTable<Assembly, Entry> _byAssembly = new();
        private static readonly object _lock = new();

        public static string? GetSummary(MemberInfo member)
            => GetSummaryById(member.DeclaringType?.Assembly, DocId(member));

        public static string? GetTypeSummary(Type type)
            => GetSummaryById(type.Assembly, "T:" + DocTypeName(type));

        private static string? GetSummaryById(Assembly? asm, string? id)
        {
            if (asm == null || string.IsNullOrEmpty(id))
                return null;

            Entry entry = GetEntry(asm);
            return entry.Summaries.TryGetValue(id, out string? summary) ? summary : null;
        }

        private static Entry GetEntry(Assembly asm)
        {
            string xmlPath = GetXmlPath(asm);

            lock (_lock)
            {
                bool exists = xmlPath.Length != 0 && File.Exists(xmlPath);
                DateTime writeTime = exists ? File.GetLastWriteTimeUtc(xmlPath) : DateTime.MinValue;

                if (_byAssembly.TryGetValue(asm, out Entry? cached) && cached.WriteTimeUtc == writeTime)
                    return cached; // Up to date (covers the "no file" -> MinValue case too).

                var entry = new Entry { WriteTimeUtc = writeTime };
                if (exists)
                {
                    try
                    {
                        Parse(xmlPath, entry.Summaries);
                    }
                    catch (Exception e)
                    {
                        // Invalid XML -> leave summaries empty, behavior stays unchanged.
                        ExceptionUtils.LogException(e);
                    }
                }

                _byAssembly.AddOrUpdate(asm, entry);
                return entry;
            }
        }

        // Resolves the path of the assembly's XML doc file (e.g. MyGame.xml).
        // Godot loads project assemblies from memory (see GodotPlugins.PluginLoadContext),
        // so Assembly.Location is empty; fall back to the load context's known on-disk path.
        private static string GetXmlPath(Assembly asm)
        {
            if (asm.IsDynamic)
                return string.Empty;

            string? dir = null;
            if (!string.IsNullOrEmpty(asm.Location))
            {
                dir = Path.GetDirectoryName(asm.Location);
            }
            else
            {
                AssemblyLoadContext? alc = AssemblyLoadContext.GetLoadContext(asm);
                // PluginLoadContext exposes the on-disk path it loaded from. Read it via
                // reflection to avoid a circular reference from GodotSharp to GodotPlugins.
                // All assemblies in a plugin context share the same bin directory.
                if (alc?.GetType().GetProperty("AssemblyLoadedPath")?.GetValue(alc) is string loadedPath
                    && !string.IsNullOrEmpty(loadedPath))
                {
                    dir = Path.GetDirectoryName(loadedPath);
                }
            }

            string? name = asm.GetName().Name;
            if (string.IsNullOrEmpty(dir) || string.IsNullOrEmpty(name))
                return string.Empty;

            return Path.Combine(dir, name + ".xml");
        }

        private static void Parse(string xmlPath, Dictionary<string, string> into)
        {
            var settings = new XmlReaderSettings { DtdProcessing = DtdProcessing.Prohibit, XmlResolver = null };
            using var reader = XmlReader.Create(xmlPath, settings);

            string? currentMember = null;
            while (reader.Read())
            {
                if (reader.NodeType == XmlNodeType.Element && reader.Name == "member")
                {
                    currentMember = reader.GetAttribute("name");
                }
                else if (reader.NodeType == XmlNodeType.Element && reader.Name == "summary" && currentMember != null)
                {
                    string raw = reader.ReadInnerXml(); // May contain <see cref="..."/> etc.
                    into[currentMember] = Normalize(raw);
                }
            }
        }

        // Collapse whitespace, drop tags, resolve <see cref="..."/> to its short name.
        private static string Normalize(string xml)
        {
            // <see cref="T:Ns.Type.Member"/> -> Member ; <paramref name="x"/> -> x
            xml = Regex.Replace(xml, "<see[^>]*cref=\"[A-Za-z]:(?<id>[^\"]+)\"[^>]*/?>",
                m => ShortName(m.Groups["id"].Value));
            xml = Regex.Replace(xml, "<(paramref|typeparamref)[^>]*name=\"(?<n>[^\"]+)\"[^>]*/?>",
                m => m.Groups["n"].Value);
            xml = Regex.Replace(xml, "<[^>]+>", string.Empty); // Strip any remaining tags.
            xml = System.Net.WebUtility.HtmlDecode(xml);
            xml = Regex.Replace(xml, "\\s+", " ").Trim(); // Normalize whitespace.
            return xml;
        }

        private static string ShortName(string id)
        {
            int paren = id.IndexOf('(');
            if (paren >= 0)
                id = id[..paren];
            int dot = id.LastIndexOf('.');
            return dot >= 0 ? id[(dot + 1)..] : id;
        }

        // Build the XML doc member id used by the compiler.
        private static string DocId(MemberInfo member)
        {
            if (member.DeclaringType == null)
                return string.Empty;

            string typeName = DocTypeName(member.DeclaringType);
            return member.MemberType switch
            {
                MemberTypes.Field => "F:" + typeName + "." + member.Name,
                MemberTypes.Property => "P:" + typeName + "." + member.Name,
                _ => string.Empty,
            };
        }

        // Namespace.Type, with the nested-type '+' separator replaced by '.' as the doc id format requires.
        private static string DocTypeName(Type type)
            => (type.FullName ?? type.Name).Replace('+', '.');
    }
}
