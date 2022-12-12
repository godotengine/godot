using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;

namespace Godot.SourceGenerators
{
    [Generator]
    public class ScriptMethodsGenerator : ISourceGenerator
    {
        public void Initialize(GeneratorInitializationContext context)
        {
        }

        public void Execute(GeneratorExecutionContext context)
        {
            if (context.AreGodotSourceGeneratorsDisabled())
                return;

            INamedTypeSymbol[] godotClasses = context
                .Compilation.SyntaxTrees
                .SelectMany(tree =>
                    tree.GetRoot().DescendantNodes()
                        .OfType<ClassDeclarationSyntax>()
                        .SelectGodotScriptClasses(context.Compilation)
                        // Report and skip non-partial classes
                        .Where(x =>
                        {
                            if (x.cds.IsPartial())
                            {
                                if (x.cds.IsNested() && !x.cds.AreAllOuterTypesPartial(out var typeMissingPartial))
                                {
                                    Common.ReportNonPartialGodotScriptOuterClass(context, typeMissingPartial!);
                                    return false;
                                }

                                return true;
                            }

                            Common.ReportNonPartialGodotScriptClass(context, x.cds, x.symbol);
                            return false;
                        })
                        .Select(x => x.symbol)
                )
                .Distinct<INamedTypeSymbol>(SymbolEqualityComparer.Default)
                .ToArray();

            if (godotClasses.Length > 0)
            {
                var typeCache = new MarshalUtils.TypeCache(context.Compilation);

                foreach (var godotClass in godotClasses)
                {
                    VisitGodotScriptClass(context, typeCache, godotClass);
                }
            }
        }

        private class MethodOverloadEqualityComparer : IEqualityComparer<GodotMethodData>
        {
            public bool Equals(GodotMethodData x, GodotMethodData y)
                => x.ParamTypes.Length == y.ParamTypes.Length && x.Method.Name == y.Method.Name;

            public int GetHashCode(GodotMethodData obj)
            {
                unchecked
                {
                    return (obj.ParamTypes.Length.GetHashCode() * 397) ^ obj.Method.Name.GetHashCode();
                }
            }
        }

        private static void VisitGodotScriptClass(
            GeneratorExecutionContext context,
            MarshalUtils.TypeCache typeCache,
            INamedTypeSymbol symbol
        )
        {
            INamespaceSymbol namespaceSymbol = symbol.ContainingNamespace;
            string classNs = namespaceSymbol != null && !namespaceSymbol.IsGlobalNamespace ?
                namespaceSymbol.FullQualifiedNameOmitGlobal() :
                string.Empty;
            bool hasNamespace = classNs.Length != 0;

            bool isInnerClass = symbol.ContainingType != null;

            string uniqueHint = symbol.FullQualifiedNameOmitGlobal().SanitizeQualifiedNameForUniqueHint()
                                + "_ScriptMethods.generated";

            var source = new StringBuilder();

            source.Append("using Godot;\n");
            source.Append("using Godot.NativeInterop;\n");
            source.Append("\n");

            if (hasNamespace)
            {
                source.Append("namespace ");
                source.Append(classNs);
                source.Append(" {\n\n");
            }

            if (isInnerClass)
            {
                var containingType = symbol.ContainingType;

                while (containingType != null)
                {
                    source.Append("partial ");
                    source.Append(containingType.GetDeclarationKeyword());
                    source.Append(" ");
                    source.Append(containingType.NameWithTypeParameters());
                    source.Append("\n{\n");

                    containingType = containingType.ContainingType;
                }
            }

            source.Append("partial class ");
            source.Append(symbol.NameWithTypeParameters());
            source.Append("\n{\n");

            var members = symbol.GetMembers();

            var methodSymbols = members
                .Where(s => !s.IsStatic && s.Kind == SymbolKind.Method && !s.IsImplicitlyDeclared)
                .Cast<IMethodSymbol>()
                .Where(m => m.MethodKind == MethodKind.Ordinary);

            var godotClassMethods = methodSymbols.WhereHasGodotCompatibleSignature(typeCache)
                .Distinct(new MethodOverloadEqualityComparer())
                .ToArray();

            source.Append("#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword\n");

            source.Append(
                $"    public new class MethodName : {symbol.BaseType.FullQualifiedNameIncludeGlobal()}.MethodName {{\n");

            // Generate cached StringNames for methods and properties, for fast lookup

            var distinctMethodNames = godotClassMethods
                .Select(m => m.Method.Name)
                .Distinct()
                .ToArray();

            foreach (string methodName in distinctMethodNames)
            {
                source.Append("        public new static readonly global::Godot.StringName ");
                source.Append(methodName);
                source.Append(" = \"");
                source.Append(methodName);
                source.Append("\";\n");
            }

            source.Append("    }\n"); // class GodotInternal

            // Generate GetGodotMethodList

            if (godotClassMethods.Length > 0)
            {
                const string listType = "global::System.Collections.Generic.List<global::Godot.Bridge.MethodInfo>";

                source.Append("    internal new static ")
                    .Append(listType)
                    .Append(" GetGodotMethodList()\n    {\n");

                source.Append("        var methods = new ")
                    .Append(listType)
                    .Append("(")
                    .Append(godotClassMethods.Length)
                    .Append(");\n");

                foreach (var method in godotClassMethods)
                {
                    var methodInfo = DetermineMethodInfo(method);
                    AppendMethodInfo(source, methodInfo);
                }

                source.Append("        return methods;\n");
                source.Append("    }\n");
            }

            source.Append("#pragma warning restore CS0109\n");

            // Generate InvokeGodotClassMethod

            if (godotClassMethods.Length > 0)
            {
                source.Append("    protected override bool InvokeGodotClassMethod(in godot_string_name method, ");
                source.Append("NativeVariantPtrArgs args, out godot_variant ret)\n    {\n");

                foreach (var method in godotClassMethods)
                {
                    GenerateMethodInvoker(method, source);
                }

                source.Append("        return base.InvokeGodotClassMethod(method, args, out ret);\n");

                source.Append("    }\n");
            }

            // Generate HasGodotClassMethod

            if (distinctMethodNames.Length > 0)
            {
                source.Append("    protected override bool HasGodotClassMethod(in godot_string_name method)\n    {\n");

                bool isFirstEntry = true;
                foreach (string methodName in distinctMethodNames)
                {
                    GenerateHasMethodEntry(methodName, source, isFirstEntry);
                    isFirstEntry = false;
                }

                source.Append("        return base.HasGodotClassMethod(method);\n");

                source.Append("    }\n");
            }

            source.Append("}\n"); // partial class

            if (isInnerClass)
            {
                var containingType = symbol.ContainingType;

                while (containingType != null)
                {
                    source.Append("}\n"); // outer class

                    containingType = containingType.ContainingType;
                }
            }

            if (hasNamespace)
            {
                source.Append("\n}\n");
            }

            context.AddSource(uniqueHint, SourceText.From(source.ToString(), Encoding.UTF8));
        }

        private static void AppendMethodInfo(StringBuilder source, MethodInfo methodInfo)
        {
            source.Append("        methods.Add(new(name: MethodName.")
                .Append(methodInfo.Name)
                .Append(", returnVal: ");

            AppendPropertyInfo(source, methodInfo.ReturnVal);

            source.Append(", flags: (global::Godot.MethodFlags)")
                .Append((int)methodInfo.Flags)
                .Append(", arguments: ");

            if (methodInfo.Arguments is { Count: > 0 })
            {
                source.Append("new() { ");

                foreach (var param in methodInfo.Arguments)
                {
                    AppendPropertyInfo(source, param);

                    // C# allows colon after the last element
                    source.Append(", ");
                }

                source.Append(" }");
            }
            else
            {
                source.Append("null");
            }

            source.Append(", defaultArguments: null));\n");
        }

        private static void AppendPropertyInfo(StringBuilder source, PropertyInfo propertyInfo)
        {
            source.Append("new(type: (global::Godot.Variant.Type)")
                .Append((int)propertyInfo.Type)
                .Append(", name: \"")
                .Append(propertyInfo.Name)
                .Append("\", hint: (global::Godot.PropertyHint)")
                .Append((int)propertyInfo.Hint)
                .Append(", hintString: \"")
                .Append(propertyInfo.HintString)
                .Append("\", usage: (global::Godot.PropertyUsageFlags)")
                .Append((int)propertyInfo.Usage)
                .Append(", exported: ")
                .Append(propertyInfo.Exported ? "true" : "false")
                .Append(")");
        }

        private static MethodInfo DetermineMethodInfo(GodotMethodData method)
        {
            PropertyInfo returnVal;

            if (method.RetType != null)
            {
                returnVal = DeterminePropertyInfo(method.RetType.Value.MarshalType, name: string.Empty);
            }
            else
            {
                returnVal = new PropertyInfo(VariantType.Nil, string.Empty, PropertyHint.None,
                    hintString: null, PropertyUsageFlags.Default, exported: false);
            }

            int paramCount = method.ParamTypes.Length;

            List<PropertyInfo>? arguments;

            if (paramCount > 0)
            {
                arguments = new(capacity: paramCount);

                for (int i = 0; i < paramCount; i++)
                {
                    arguments.Add(DeterminePropertyInfo(method.ParamTypes[i],
                        name: method.Method.Parameters[i].Name));
                }
            }
            else
            {
                arguments = null;
            }

            return new MethodInfo(method.Method.Name, returnVal, MethodFlags.Default, arguments,
                defaultArguments: null);
        }

        private static PropertyInfo DeterminePropertyInfo(MarshalType marshalType, string name)
        {
            var memberVariantType = MarshalUtils.ConvertMarshalTypeToVariantType(marshalType)!.Value;

            var propUsage = PropertyUsageFlags.Default;

            if (memberVariantType == VariantType.Nil)
                propUsage |= PropertyUsageFlags.NilIsVariant;

            return new PropertyInfo(memberVariantType, name,
                PropertyHint.None, string.Empty, propUsage, exported: false);
        }

        private static void GenerateHasMethodEntry(
            string methodName,
            StringBuilder source,
            bool isFirstEntry
        )
        {
            source.Append("        ");
            if (!isFirstEntry)
                source.Append("else ");
            source.Append("if (method == MethodName.");
            source.Append(methodName);
            source.Append(") {\n           return true;\n        }\n");
        }

        private static void GenerateMethodInvoker(
            GodotMethodData method,
            StringBuilder source
        )
        {
            string methodName = method.Method.Name;

            source.Append("        if (method == MethodName.");
            source.Append(methodName);
            source.Append(" && args.Count == ");
            source.Append(method.ParamTypes.Length);
            source.Append(") {\n");

            if (method.RetType != null)
                source.Append("            var callRet = ");
            else
                source.Append("            ");

            source.Append(methodName);
            source.Append("(");

            for (int i = 0; i < method.ParamTypes.Length; i++)
            {
                if (i != 0)
                    source.Append(", ");

                source.AppendNativeVariantToManagedExpr(string.Concat("args[", i.ToString(), "]"),
                    method.ParamTypeSymbols[i], method.ParamTypes[i]);
            }

            source.Append(");\n");

            if (method.RetType != null)
            {
                source.Append("            ret = ");

                source.AppendManagedToNativeVariantExpr("callRet",
                    method.RetType.Value.TypeSymbol, method.RetType.Value.MarshalType);
                source.Append(";\n");

                source.Append("            return true;\n");
            }
            else
            {
                source.Append("            ret = default;\n");
                source.Append("            return true;\n");
            }

            source.Append("        }\n");
        }
    }
}
