using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Text;
using System.Collections.Generic;
using System.Text;

namespace GodotSharp.SourceGenerators
{
    [Generator]
    internal class VectorSwizzlingGenerator : ISourceGenerator
    {
        private readonly char[] _members = new char[] { 'X', 'Y', 'Z', 'W' };

        public void Execute(GeneratorExecutionContext context)
        {
            // Vector Dimensions
            for (int d = 2; d <= 4; d++)
            {
                GenerateStruct(context, d, string.Empty);
                GenerateStruct(context, d, "I");
            }
        }

        private void GenerateStruct(GeneratorExecutionContext context, int d, string suffix)
        {
            StringBuilder sb = new();
            sb.AppendLine("using System.ComponentModel;");
            sb.AppendLine("using System.Diagnostics;");
            sb.AppendLine("");
            sb.AppendLine("namespace Godot");
            sb.AppendLine("{");
            sb.AppendLine($"\tpublic partial struct Vector{d}{suffix}");
            sb.AppendLine("\t{");
            foreach (char[] per in GetPermutations(d))
            {
                string name = new string(per);
                int retrLength = per.Length;
                string components = string.Join(", ", per);
                sb.AppendLine("\t\t/// <summary>");
                sb.AppendLine($"\t\t/// Swizzle operator returning a <see cref=\"Vector{retrLength}{suffix}\"/> with components {components} from this vector");
                sb.AppendLine("\t\t/// </summary>");
                sb.AppendLine("\t\t[EditorBrowsable(EditorBrowsableState.Never)]");
                sb.AppendLine("\t\t[DebuggerBrowsable(DebuggerBrowsableState.Never)]");
                sb.Append($"\t\tpublic readonly Vector{retrLength}{suffix} {name} => new");
                sb.Append("(");
                for (int i = 0; i < retrLength; i++)
                {
                    sb.Append($"{per[i]}");
                    if (i != retrLength - 1)
                    {
                        sb.Append(", ");
                    }
                }
                sb.AppendLine(");");
                sb.AppendLine();
            }
            sb.AppendLine("\t}");
            sb.AppendLine("}");
            context.AddSource($"Vector{d}{suffix}.Swizzling.generated", SourceText.From(sb.ToString(), Encoding.UTF8));
        }

        public void Initialize(GeneratorInitializationContext context)
        {
        }

        private List<char[]> GetPermutations(int numMembers)
        {
            List<char[]> permutations = new();
            for (int i = 0; i < numMembers; i++)
            {
                for (int j = 0; j < numMembers; j++)
                {
                    permutations.Add(new[] { _members[i], _members[j] });
                }
            }
            for (int i = 0; i < numMembers; i++)
            {
                for (int j = 0; j < numMembers; j++)
                {
                    for (int k = 0; k < numMembers; k++)
                    {
                        permutations.Add(new[] { _members[i], _members[j], _members[k] });
                    }
                }
            }
            for (int i = 0; i < numMembers; i++)
            {
                for (int j = 0; j < numMembers; j++)
                {
                    for (int k = 0; k < numMembers; k++)
                    {
                        for (int l = 0; l < numMembers; l++)
                        {
                            permutations.Add(new[] { _members[i], _members[j], _members[k], _members[l] });
                        }
                    }
                }
            }
            return permutations;
        }
    }
}
