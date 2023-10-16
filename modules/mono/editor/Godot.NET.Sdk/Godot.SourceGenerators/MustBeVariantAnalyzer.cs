using System;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Godot.SourceGenerators
{
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public class MustBeVariantAnalyzer : DiagnosticAnalyzer
    {
        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics
            => ImmutableArray.Create(
                Common.GenericTypeArgumentMustBeVariantRule,
                Common.GenericTypeParameterMustBeVariantAnnotatedRule,
                Common.TypeArgumentParentSymbolUnhandledRule,
                Common.InvalidMustBeGenericTypeParameterUsageRule);

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();

            context.RegisterSyntaxNodeAction(AnalyzeExportedProperties, SyntaxKind.FieldDeclaration,
                SyntaxKind.PropertyDeclaration);
            context.RegisterSyntaxNodeAction(AnalyzeSignals, SyntaxKind.DelegateDeclaration);

            // Has to include IdentifierName because calls to methods with the type arguments inferred aren't 'GenericName' syntax.
            context.RegisterSyntaxNodeAction(AnalyzeGenericTypeUsage, SyntaxKind.GenericName,
                SyntaxKind.IdentifierName);
        }

        /// <summary>
        /// If a generic type is used in an exported field or property, make sure it is [MustBeVariant]
        /// </summary>
        /// <param name="context"></param>
        private void AnalyzeExportedProperties(SyntaxNodeAnalysisContext context)
        {
            SemanticModel sm = context.SemanticModel;
            var member = (MemberDeclarationSyntax)context.Node;

            if (!member.GetAllAttributes().Any(a => a.GetTypeSymbol(sm).IsGodotExportAttribute())) return;

            TypeSyntax typeSyntax = member switch
            {
                FieldDeclarationSyntax field => field.Declaration.Type,
                PropertyDeclarationSyntax property => property.Type,
                _ => throw new ArgumentException() // Shouldn't happen
            };

            ITypeSymbol typeSymbol = (ITypeSymbol)ModelExtensions.GetSymbolInfo(sm, typeSyntax).Symbol!;

            if (typeSymbol is ITypeParameterSymbol typeParam && !typeParam.IsVariantCompatible())
            {
                Common.ReportGenericTypeParameterMustBeVariantAnnotated(context, typeParam);
            }
        }

        /// <summary>
        /// If a generic type is used in a signal delegate, make sure it is [MustBeVariant]
        /// </summary>
        /// <param name="context"></param>
        private void AnalyzeSignals(SyntaxNodeAnalysisContext context)
        {
            SemanticModel sm = context.SemanticModel;
            var delegateSyntax = (DelegateDeclarationSyntax)context.Node;

            if (!delegateSyntax.GetAllAttributes().Any(a => a.GetTypeSymbol(sm).IsGodotSignalAttribute()))
                return;

            foreach (var param in delegateSyntax.ParameterList.Parameters)
            {
                if (param.Type == null) continue;

                var paramType = (ITypeSymbol)ModelExtensions.GetSymbolInfo(sm, param.Type).Symbol!;

                if (paramType is ITypeParameterSymbol typeParam && !typeParam.IsVariantCompatible())
                {
                    Common.ReportGenericTypeParameterMustBeVariantAnnotated(context, typeParam);
                }
            }

            var retType = (ITypeSymbol)ModelExtensions.GetSymbolInfo(sm, delegateSyntax.ReturnType).Symbol!;
            if (retType is ITypeParameterSymbol retTypeParam && !retTypeParam.IsVariantCompatible())
            {
                Common.ReportGenericTypeParameterMustBeVariantAnnotated(context, retTypeParam);
            }
        }

        /// <summary>
        /// Any time a generic name (type, method) is referenced, check to make sure all of the type arguments
        /// that need to be [MustBeVariant] are Variant-compatible.
        /// </summary>
        private void AnalyzeGenericTypeUsage(SyntaxNodeAnalysisContext context)
        {
            SemanticModel sm = context.SemanticModel;
            MarshalUtils.TypeCache typeCache = new MarshalUtils.TypeCache(context.Compilation);

            // Get both the type arguments and type parameters for the generic name
            ISymbol? genericName = sm.GetSymbolInfo(context.Node).Symbol;
            var (typeArguments, typeParameters) = genericName switch
            {
                INamedTypeSymbol typeSymbol => (typeSymbol.TypeArguments, typeSymbol.TypeParameters),
                IMethodSymbol methodSymbol => (methodSymbol.TypeArguments, methodSymbol.TypeParameters),
                _ => (ImmutableArray<ITypeSymbol>.Empty, ImmutableArray<ITypeParameterSymbol>.Empty)
            };

            // Nothing to check.
            if (genericName == null || typeArguments.Length == 0) return;

            // Check each type argument
            for (int i = 0; i < typeParameters.Length; i++)
            {
                // This type parameter isn't [MustBeVariant] so we don't care
                if (!typeParameters[i].IsVariantCompatible()) continue;

                // Check what the provided type argument is
                ITypeSymbol argument = typeArguments[i];
                if (argument is ITypeParameterSymbol typeParam && !typeParam.IsVariantCompatible())
                {
                    // Another type parameter took its place but it isn't also [MustBeVariant]
                    Common.ReportGenericTypeParameterMustBeVariantAnnotated(context, typeParam);
                    continue;
                }

                if (MarshalUtils.ConvertManagedTypeToMarshalType(typeArguments[i], typeCache) == null)
                {
                    // An explicit type was specified but we can't marshal it to a variant type
                    Common.ReportGenericTypeArgumentMustBeVariant(context, context.Node, argument);
                    continue;
                }

                // Now that we know the type arguments used with this generic type we can try to find invalid usage
                CheckForInvalidGenericVariantUsage(context, genericName, typeParameters[i], typeArguments[i],
                    context.Node, typeCache);
            }
        }

        /// <summary>
        /// There are a few edge cases where we might not know for sure if the usage of a [MustBeGeneric] type parameter
        /// is valid until we know the type argument that replaces it. Here we check for that. This isn't perfect, but
        /// hopefully it will catch the majority of invalid uses.
        ///
        /// Consider the class:
        /// <code>
        /// public partial class GenericObject&lt;[MustBeGeneric] T&gt; : Node
        /// {
        ///     [Export] public T SomeField;
        ///     [Export] public T[] SomeArray;
        /// }
        /// </code>
        ///
        /// The above class is perfectly valid provided that both T and T[] are Variant-compatible, which is the case
        /// with several types such as int, float, Vector2, Color, StringName, etc. However, there are also some types
        /// where this isn't the case, such as Rect2, Plane, Callable, and more. A system array of any of these types
        /// is not Variant-compatible and they cannot be marshaled, instead they need to use a Godot.Collections.Array.
        ///
        /// Additionally, any array of a GodotObject-derived type is considered Variant-compatible but it isn't really
        /// as it is always marshalled to/from a Godot array first, and the Variant.From and Variant.ConvertTo methods
        /// don't support them so there's no way to marshal them in an entirely generic context. For these reasons,
        /// this is also considered invalid usage and should also use a Godot.Collections.Array.
        ///
        /// Note on the above; exporting a system array of GodotObject-derived types should still be fine as long as
        /// it isn't generic. There's a special case in the source generator to handle this provided it knows the type
        /// ahead of time, which is not the case when using type parameters.
        /// </summary>
        private void CheckForInvalidGenericVariantUsage(SyntaxNodeAnalysisContext context, ISymbol genericName,
            ITypeParameterSymbol typeParameter, ITypeSymbol typeArgument, SyntaxNode location,
            MarshalUtils.TypeCache typeCache)
        {
            ITypeSymbol? invalidType = null;
            ISymbol? source = null;

            if (typeArgument is not INamedTypeSymbol) return;

            if (genericName is INamedTypeSymbol typeSymbol)
                IsValidType(typeParameter, typeSymbol, typeSymbol.OriginalDefinition, typeCache, out invalidType,
                    ref source);
            else if (genericName is IMethodSymbol methodSymbol)
                CheckMethodSymbol(typeParameter, methodSymbol, methodSymbol.OriginalDefinition, typeCache,
                    out invalidType, out source);

            if (invalidType != null && source != null)
            {
                Common.ReportInvalidMustBeGenericTypeParameterUsage(context, typeParameter, typeArgument,
                    invalidType, source, location);
            }
        }

        private bool IsValidType(ITypeParameterSymbol typeParameter, ITypeSymbol type, ITypeSymbol originalType,
            MarshalUtils.TypeCache typeCache, out ITypeSymbol? invalidType, ref ISymbol? source)
        {
            // If the type itself is not valid then no
            MarshalType? marshalType = MarshalUtils.ConvertManagedTypeToMarshalType(type, typeCache);
            if (marshalType == null || marshalType == MarshalType.GodotObjectOrDerivedArray)
            {
                invalidType = originalType;
                return false;
            }

            // If this type is also generic we need to go deeper still
            if (type is INamedTypeSymbol namedTypeSymbol && namedTypeSymbol.Arity > 0)
            {
                // Check all of the type members to see if any of them are invalid
                INamedTypeSymbol originalNamedType = (INamedTypeSymbol)originalType;
                if (!CheckTypeMembers(typeParameter, namedTypeSymbol, originalNamedType, typeCache, out invalidType,
                        out source))
                    return false;

                INamedTypeSymbol? baseType = type.BaseType;
                if (baseType != null && baseType.Arity > 0)
                {
                    // If the type parameter was passed into a parent generic type, we can check those too
                    INamedTypeSymbol originalBaseType = originalNamedType.BaseType!;
                    for (int i = 0; i < namedTypeSymbol.TypeArguments.Length; i++)
                    {
                        if (!UsesTypeParameter(typeParameter, baseType, originalBaseType))
                            continue;

                        if (!IsValidType(typeParameter, baseType, originalBaseType, typeCache, out invalidType, ref source))
                            return false;
                    }
                }
            }

            invalidType = null;
            return true;
        }

        /// <summary>
        /// Checks a method's argument types and return type for non Variant-compatible types
        /// </summary>
        private bool CheckMethodSymbol(ITypeParameterSymbol typeParameter, IMethodSymbol methodSymbol,
            IMethodSymbol originalMethodSymbol, MarshalUtils.TypeCache typeCache, out ITypeSymbol? invalidType,
            out ISymbol? source)
        {
            source = methodSymbol.OriginalDefinition;

            // Check each parameter and the return type
            for (int i = 0; i < methodSymbol.Parameters.Length; i++)
            {
                // Only check parameters that originally used this specific type parameter
                ITypeSymbol paramType = methodSymbol.Parameters[i].Type;
                ITypeSymbol originalParamType = originalMethodSymbol.Parameters[i].Type;
                if (!UsesTypeParameter(typeParameter, paramType, originalParamType)) continue;

                // Warn if new type is no longer valid
                if (!IsValidType(typeParameter, paramType, originalParamType, typeCache, out invalidType, ref source))
                    return false;
            }

            // Same but for return type
            ITypeSymbol originalRetType = originalMethodSymbol.ReturnType;
            if (UsesTypeParameter(typeParameter, methodSymbol.ReturnType, originalRetType))
            {
                if (!IsValidType(typeParameter, methodSymbol.ReturnType, originalRetType, typeCache,
                        out invalidType, ref source))
                    return false;
            }

            invalidType = null;
            return true;
        }

        /// <summary>
        /// Check a generic type for non Variant-compatible members
        /// </summary>
        private bool CheckTypeMembers(ITypeParameterSymbol typeParameter, INamedTypeSymbol typeSymbol,
            INamedTypeSymbol originalType, MarshalUtils.TypeCache typeCache, out ITypeSymbol? invalidType,
            out ISymbol? source)
        {
            source = originalType;

            ImmutableArray<ISymbol> members = typeSymbol.GetMembers();
            ImmutableArray<ISymbol> originalMembers = originalType.GetMembers();
            for (int i = 0; i < members.Length; i++)
            {
                ISymbol member = members[i];
                if (member is IPropertySymbol property && property.HasGodotExportAttribute())
                {
                    ITypeSymbol originalPropertyType = ((IPropertySymbol)originalMembers[i]).OriginalDefinition.Type;
                    if (!IsValidType(typeParameter, property.Type, originalPropertyType, typeCache, out invalidType,
                            ref source))
                        return false;
                }
                else if (member is IFieldSymbol field && field.HasGodotExportAttribute())
                {
                    ITypeSymbol originalFieldType = ((IFieldSymbol)originalMembers[i]).OriginalDefinition.Type;
                    if (!IsValidType(typeParameter, field.Type, originalFieldType, typeCache, out invalidType,
                            ref source))
                        return false;
                }
                else if (member is IMethodSymbol)
                {
                    // TODO: Do we want to analyze and error on methods...?
                }
                else if (member is INamedTypeSymbol namedType && namedType.HasGodotSignalAttribute())
                {
                    IMethodSymbol? methodSymbol = namedType.DelegateInvokeMethod;
                    if (methodSymbol == null) continue;
                    IMethodSymbol originalMethodSymbol = ((INamedTypeSymbol)originalMembers[i]).DelegateInvokeMethod!;

                    if (!CheckMethodSymbol(typeParameter, methodSymbol, originalMethodSymbol, typeCache,
                            out invalidType, out source))
                    {
                        source = originalMembers[i];
                        return false;
                    }
                }
            }

            invalidType = null;
            return true;
        }

        /// <summary>
        /// Returns true if the given symbol used the given type parameter in any part of itself
        /// </summary>
        private bool UsesTypeParameter(ITypeParameterSymbol typeParamSymbol, ITypeSymbol currentTypeSymbol,
            ITypeSymbol originalTypeSymbol)
        {
            static bool CompareSymbols(ISymbol a, ISymbol b) => SymbolEqualityComparer.Default.Equals(a, b);

            // If it hasn't changed then no
            if (CompareSymbols(originalTypeSymbol, currentTypeSymbol)) return false;

            // If the type previously was just the generic type itself, yes
            if (CompareSymbols(originalTypeSymbol, typeParamSymbol)) return true;

            // If the type was an array of the generic type, yes
            if (originalTypeSymbol is IArrayTypeSymbol array && CompareSymbols(array.ElementType, typeParamSymbol))
                return true;

            // If the symbol used the generic type in another type argument list, yes
            if (currentTypeSymbol is INamedTypeSymbol currentNamedType && currentNamedType.Arity > 0)
            {
                var originalNamedType = (INamedTypeSymbol)originalTypeSymbol;
                for (int i = 0; i < currentNamedType.TypeArguments.Length; i++)
                {
                    // If the nested type used our type parameter, yes.
                    if (UsesTypeParameter(typeParamSymbol, currentNamedType.TypeArguments[i],
                            originalNamedType.TypeArguments[i]))
                        return true;
                }
            }

            // Otherwise probably no? Can't think of anything else.
            return false;
        }
    }
}
