using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;
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
            if (context.IsGodotSourceGeneratorDisabled("ScriptMethods"))
                return;

            bool enableExportNullChecks = context.IsGodotEnableExportNullChecks();

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
                                if (x.cds.IsNested() && !x.cds.AreAllOuterTypesPartial(out _))
                                {
                                    return false;
                                }

                                return true;
                            }
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
                    VisitGodotScriptClass(context, typeCache, godotClass, enableExportNullChecks);
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
            INamedTypeSymbol symbol,
            bool enableExportNullChecks
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
                AppendPartialContainingTypeDeclarations(containingType);

                void AppendPartialContainingTypeDeclarations(INamedTypeSymbol? containingType)
                {
                    if (containingType == null)
                        return;

                    AppendPartialContainingTypeDeclarations(containingType.ContainingType);

                    source.Append("partial ");
                    source.Append(containingType.GetDeclarationKeyword());
                    source.Append(" ");
                    source.Append(containingType.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat));
                    source.Append("\n{\n");
                }
            }

            source.Append("partial class ");
            source.Append(symbol.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat));
            source.Append("\n{\n");

            var members = symbol.GetMembers();

            var methodSymbols = members
                .Where(s => s.Kind == SymbolKind.Method && !s.IsImplicitlyDeclared)
                .Cast<IMethodSymbol>()
                .Where(m => m.MethodKind == MethodKind.Ordinary);

            var godotClassMethods = methodSymbols.WhereHasGodotCompatibleSignature(typeCache)
                .Distinct(new MethodOverloadEqualityComparer())
                .ToArray();

            // Collect exported properties/fields for null checks if feature is enabled
            List<(string Name, ITypeSymbol Type)>? exportedNonNullableGodotTypes = null;
            if (enableExportNullChecks)
            {
                exportedNonNullableGodotTypes = new List<(string, ITypeSymbol)>();

                // Collect exported properties
                var propertySymbols = members
                    .Where(s => !s.IsStatic && s.Kind == SymbolKind.Property)
                    .Cast<IPropertySymbol>();

                foreach (var property in propertySymbols)
                {
                    if (IsExportedNonNullableGodotType(property, property.Type, context.Compilation))
                    {
                        exportedNonNullableGodotTypes.Add((property.Name, property.Type));
                    }
                }

                // Collect exported fields
                var fieldSymbols = members
                    .Where(s => !s.IsStatic && s is { Kind: SymbolKind.Field, IsImplicitlyDeclared: false })
                    .Cast<IFieldSymbol>();

                foreach (var field in fieldSymbols)
                {
                    if (IsExportedNonNullableGodotType(field, field.Type, context.Compilation))
                    {
                        exportedNonNullableGodotTypes.Add((field.Name, field.Type));
                    }
                }
            }

            source.Append("#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword\n");

            source.Append("    /// <summary>\n")
                .Append("    /// Cached StringNames for the methods contained in this class, for fast lookup.\n")
                .Append("    /// </summary>\n");

            source.Append(
                $"    public new class MethodName : {symbol.BaseType!.FullQualifiedNameIncludeGlobal()}.MethodName {{\n");

            // Generate cached StringNames for methods and properties, for fast lookup

            var distinctMethodNames = godotClassMethods
                .Select(m => m.Method.Name)
                .Distinct()
                .ToArray();

            foreach (string methodName in distinctMethodNames)
            {
                source.Append("        /// <summary>\n")
                    .Append("        /// Cached name for the '")
                    .Append(methodName)
                    .Append("' method.\n")
                    .Append("        /// </summary>\n");

                source.Append("        public new static readonly global::Godot.StringName @");
                source.Append(methodName);
                source.Append(" = \"");
                source.Append(methodName);
                source.Append("\";\n");
            }

            source.Append("    }\n"); // class GodotInternal

            // Generate GetGodotMethodList

            if (godotClassMethods.Length > 0)
            {
                const string ListType = "global::System.Collections.Generic.List<global::Godot.Bridge.MethodInfo>";

                source.Append("    /// <summary>\n")
                    .Append("    /// Get the method information for all the methods declared in this class.\n")
                    .Append("    /// This method is used by Godot to register the available methods in the editor.\n")
                    .Append("    /// Do not call this method.\n")
                    .Append("    /// </summary>\n");

                source.Append("    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]\n");

                source.Append("    internal new static ")
                    .Append(ListType)
                    .Append(" GetGodotMethodList()\n    {\n");

                source.Append("        var methods = new ")
                    .Append(ListType)
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

            // Generate ValidateExportedProperties

            if (exportedNonNullableGodotTypes is { Count: > 0 })
            {
                source.Append("    /// <inheritdoc/>\n");
                source.Append("    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]\n");
                source.Append("    protected override void ValidateExportedProperties()\n    {\n");

                GenerateNullChecksForValidation(source, exportedNonNullableGodotTypes);

                source.Append("        base.ValidateExportedProperties();\n");
                source.Append("    }\n");
            }

            // Generate InvokeGodotClassMethod

            if (godotClassMethods.Length > 0)
            {
                source.Append("    /// <inheritdoc/>\n");
                source.Append("    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]\n");
                source.Append("    protected override bool InvokeGodotClassMethod(in godot_string_name method, ");
                source.Append("NativeVariantPtrArgs args, out godot_variant ret)\n    {\n");

                foreach (var method in godotClassMethods)
                {
                    GenerateMethodInvoker(method, source);
                }

                source.Append("        return base.InvokeGodotClassMethod(method, args, out ret);\n");

                source.Append("    }\n");
            }

            // Generate InvokeGodotClassStaticMethod

            var godotClassStaticMethods = godotClassMethods.Where(m => m.Method.IsStatic).ToArray();

            if (godotClassStaticMethods.Length > 0)
            {
                source.Append("#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword\n");
                source.Append("    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]\n");
                source.Append("    internal new static bool InvokeGodotClassStaticMethod(in godot_string_name method, ");
                source.Append("NativeVariantPtrArgs args, out godot_variant ret)\n    {\n");

                foreach (var method in godotClassStaticMethods)
                {
                    GenerateMethodInvoker(method, source);
                }

                source.Append("        ret = default;\n");
                source.Append("        return false;\n");
                source.Append("    }\n");

                source.Append("#pragma warning restore CS0109\n");
            }

            // Generate HasGodotClassMethod

            if (distinctMethodNames.Length > 0)
            {
                source.Append("    /// <inheritdoc/>\n");
                source.Append("    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]\n");
                source.Append("    protected override bool HasGodotClassMethod(in godot_string_name method)\n    {\n");

                foreach (string methodName in distinctMethodNames)
                {
                    GenerateHasMethodEntry(methodName, source);
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
            source.Append("        methods.Add(new(name: MethodName.@")
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
                .Append(propertyInfo.Exported ? "true" : "false");
            if (propertyInfo.ClassName != null)
            {
                source.Append(", className: new global::Godot.StringName(\"")
                    .Append(propertyInfo.ClassName)
                    .Append("\")");
            }
            source.Append(")");
        }

        private static MethodInfo DetermineMethodInfo(GodotMethodData method)
        {
            PropertyInfo returnVal;

            if (method.RetType != null)
            {
                returnVal = DeterminePropertyInfo(method.RetType.Value.MarshalType,
                    method.RetType.Value.TypeSymbol,
                    name: string.Empty);
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
                        method.Method.Parameters[i].Type,
                        name: method.Method.Parameters[i].Name));
                }
            }
            else
            {
                arguments = null;
            }

            MethodFlags flags = MethodFlags.Default;

            if (method.Method.IsStatic)
            {
                flags |= MethodFlags.Static;
            }

            return new MethodInfo(method.Method.Name, returnVal, flags, arguments,
                defaultArguments: null);
        }

        private static PropertyInfo DeterminePropertyInfo(MarshalType marshalType, ITypeSymbol typeSymbol, string name)
        {
            var memberVariantType = MarshalUtils.ConvertMarshalTypeToVariantType(marshalType)!.Value;

            var propUsage = PropertyUsageFlags.Default;

            if (memberVariantType == VariantType.Nil)
                propUsage |= PropertyUsageFlags.NilIsVariant;

            string? className = null;
            if (memberVariantType == VariantType.Object && typeSymbol is INamedTypeSymbol namedTypeSymbol)
            {
                className = namedTypeSymbol.GetGodotScriptNativeClassName();
            }

            return new PropertyInfo(memberVariantType, name,
                PropertyHint.None, string.Empty, propUsage, className, exported: false);
        }

        private static void GenerateHasMethodEntry(
            string methodName,
            StringBuilder source
        )
        {
            source.Append("        ");
            source.Append("if (method == MethodName.@");
            source.Append(methodName);
            source.Append(") {\n           return true;\n        }\n");
        }

        private static void GenerateNullChecksForValidation(
            StringBuilder source,
            List<(string Name, ITypeSymbol Type)> exportedNonNullableGodotTypes
        )
        {
            foreach (var (memberName, memberType) in exportedNonNullableGodotTypes)
            {
                source.Append("        if (this.");
                source.Append(memberName);
                source.Append(" == null) throw new global::System.NullReferenceException(\"The exported property/field '");
                source.Append(memberName);
                source.Append("' of type '");
                source.Append(memberType.ToDisplayString());
                source.Append("' is null.\");\n");
            }
        }

        private static void GenerateMethodInvoker(
            GodotMethodData method,
            StringBuilder source
        )
        {
            string methodName = method.Method.Name;

            source.Append("        if (method == MethodName.@");
            source.Append(methodName);
            source.Append(" && args.Count == ");
            source.Append(method.ParamTypes.Length);
            source.Append(") {\n");

            if (method.RetType != null)
                source.Append("            var callRet = ");
            else
                source.Append("            ");

            source.Append("@");
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

        private static bool IsNullableContextEnabledForSymbol(ISymbol symbol, Compilation compilation)
        {
            // Get the syntax reference for the symbol declaration
            var syntaxReference = symbol.DeclaringSyntaxReferences.FirstOrDefault();
            if (syntaxReference == null)
                return false;

            var syntaxTree = syntaxReference.SyntaxTree;
            var syntaxNode = syntaxReference.GetSyntax();

            // Get the nullable context options at the declaration location
            var semanticModel = compilation.GetSemanticModel(syntaxTree);
            var nullableContext = semanticModel.GetNullableContext(syntaxNode.SpanStart);

            // Check if nullable reference types are enabled (either as warnings or errors)
            return (nullableContext & NullableContext.Enabled) != 0;
        }

        private static bool IsExportedNonNullableGodotType(ISymbol memberSymbol, ITypeSymbol memberType, Compilation compilation)
        {
            // Check if the member has the [Export] attribute
            bool isExported = memberSymbol.GetAttributes()
                .Any(a => a.AttributeClass?.IsGodotExportAttribute() ?? false);

            if (!isExported)
                return false;

            // Check if the type is a Godot compatible class type (Node, Resource, or derived)
            if (memberType is not INamedTypeSymbol namedType)
                return false;

            // Check if it's a reference type and check nullable annotation
            if (!memberType.IsReferenceType)
                return false;

            // Check if member has nullable context enabled (either via #nullable or project settings)
            if (!IsNullableContextEnabledForSymbol(memberSymbol, compilation))
                return false;

            // If the type is nullable annotated (e.g., Node?), skip it
            if (memberType.NullableAnnotation == NullableAnnotation.Annotated)
                return false;

            // Check if the type inherits from Node or Resource
            bool isNodeOrResource = namedType.InheritsFrom("GodotSharp", GodotClasses.Node) ||
                                    namedType.InheritsFrom("GodotSharp", "Godot.Resource");

            return isNodeOrResource;
            if (isNodeOrResource)
                return true;

            // Check if the type is Godot.Collections.Array or Dictionary (including generic variations)
            string fullTypeName = namedType.ConstructedFrom.ToString();
            bool isGodotCollection = fullTypeName == "Godot.Collections.Array" ||
                                     fullTypeName == "Godot.Collections.Array<T>" ||
                                     fullTypeName == "Godot.Collections.Dictionary" ||
                                     fullTypeName == "Godot.Collections.Dictionary<TKey, TValue>";

            return isGodotCollection;
        }
    }

    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public class NullabilitySuppressor : DiagnosticSuppressor
    {
        private static readonly SuppressionDescriptor _suppressionMessage =
            new(
                id: "GDSPR001",
                suppressedDiagnosticId: "CS8618",
                justification: "Member's nullability has been checked by Godot when initialized."
            );

        public override ImmutableArray<SuppressionDescriptor> SupportedSuppressions =>
            ImmutableArray.Create(_suppressionMessage);

        public override void ReportSuppressions(SuppressionAnalysisContext context)
        {
            // Check if the feature is enabled
            var enableExportNullChecks = context.IsGodotEnableExportNullChecks();

            if (!enableExportNullChecks)
                return;

            foreach (var diagnostic in context.ReportedDiagnostics)
            {
                if (diagnostic.Id != "CS8618")
                    continue;

                // Get the symbol that triggered the diagnostic
                var syntaxTree = diagnostic.Location.SourceTree;
                if (syntaxTree == null)
                    continue;

                var semanticModel = context.GetSemanticModel(syntaxTree);
                var root = syntaxTree.GetRoot(context.CancellationToken);
                var node = root.FindNode(diagnostic.Location.SourceSpan);

                // Try to get the member symbol (property or field)
                ISymbol? memberSymbol = null;

                if (node is PropertyDeclarationSyntax propertyDecl)
                {
                    memberSymbol = semanticModel.GetDeclaredSymbol(propertyDecl, context.CancellationToken);
                }
                else if (node is VariableDeclaratorSyntax variableDecl)
                {
                    memberSymbol = semanticModel.GetDeclaredSymbol(variableDecl, context.CancellationToken);
                }

                if (memberSymbol == null)
                    continue;

                // Get the type symbol
                ITypeSymbol? memberType = null;
                if (memberSymbol is IPropertySymbol propertySymbol)
                {
                    memberType = propertySymbol.Type;
                }
                else if (memberSymbol is IFieldSymbol fieldSymbol)
                {
                    memberType = fieldSymbol.Type;
                }

                if (memberType == null)
                    continue;

                // Check if this is an exported non-nullable Godot type
                if (IsExportedNonNullableGodotType(memberSymbol, memberType, context))
                {
                    // Check if the containing type is a Godot script class
                    var containingType = memberSymbol.ContainingType;
                    if (containingType != null && containingType.InheritsFrom("GodotSharp", GodotClasses.GodotObject))
                    {
                        context.ReportSuppression(Suppression.Create(_suppressionMessage, diagnostic));
                    }
                }
            }
        }

        private static bool IsNullableContextEnabledForSymbol(ISymbol symbol, SuppressionAnalysisContext context)
        {
            var syntaxReference = symbol.DeclaringSyntaxReferences.FirstOrDefault();
            if (syntaxReference == null)
                return false;

            var syntaxTree = syntaxReference.SyntaxTree;
            var syntaxNode = syntaxReference.GetSyntax();

            var semanticModel = context.GetSemanticModel(syntaxTree);
            var nullableContext = semanticModel.GetNullableContext(syntaxNode.SpanStart);

            return (nullableContext & NullableContext.Enabled) != 0;
        }

        private static bool IsExportedNonNullableGodotType(ISymbol memberSymbol, ITypeSymbol memberType, SuppressionAnalysisContext context)
        {
            bool isExported = memberSymbol.GetAttributes()
                .Any(a => a.AttributeClass?.IsGodotExportAttribute() ?? false);

            if (!isExported)
                return false;

            if (memberType is not INamedTypeSymbol namedType)
                return false;

            if (!memberType.IsReferenceType)
                return false;

            if (!IsNullableContextEnabledForSymbol(memberSymbol, context))
                return false;

            if (memberType.NullableAnnotation == NullableAnnotation.Annotated)
                return false;

            bool isNodeOrResource = namedType.InheritsFrom("GodotSharp", GodotClasses.Node) ||
                                    namedType.InheritsFrom("GodotSharp", "Godot.Resource");

            return isNodeOrResource;
            if (isNodeOrResource)
                return true;

            // Check if the type is Godot.Collections.Array or Dictionary (including generic variations)
            string fullTypeName = namedType.ConstructedFrom.ToString();
            bool isGodotCollection = fullTypeName == "Godot.Collections.Array" ||
                                     fullTypeName == "Godot.Collections.Array<T>" ||
                                     fullTypeName == "Godot.Collections.Dictionary" ||
                                     fullTypeName == "Godot.Collections.Dictionary<TKey, TValue>";

            return isGodotCollection;
        }
    }
}
