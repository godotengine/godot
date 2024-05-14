using System;
using System.ComponentModel;
using System.Reflection;
using System.Text;
using Godot.NativeInterop;

namespace Godot.Bridge;

#pragma warning disable CS1591 // Missing XML comment for publicly visible type or member

[EditorBrowsable(EditorBrowsableState.Never)]
public class GenericUtils
{
    public static PropertyInfo PropertyInfoFromGenericType<[MustBeVariant] T>(StringName name, PropertyHint hint,
        string hintString, PropertyUsageFlags usage, bool exported)
    {
        Variant.Type variantType = VariantUtils.TypeOf<T>();

        // If there was an explicit hint on the property don't override it.
        if (hint == PropertyHint.None && string.IsNullOrWhiteSpace(hintString))
            GetPropertyHintString(typeof(T), variantType, hint, hintString, out hint, out hintString);

        return new PropertyInfo(variantType, name, hint, hintString, usage, exported);
    }

    /// <summary>
    /// Determines what editor hint and hint string to use for a given managed type.
    /// This function shares its logic with ScriptPropertiesGenerator.TryGetMemberExportHint so if you update
    /// anything here, check if it needs updating over there too!
    /// </summary>
    private static bool GetPropertyHintString(Type type, Variant.Type variantType, PropertyHint exportHint,
        string exportHintString, out PropertyHint hint, out string hintString)
    {
        hint = PropertyHint.None;
        hintString = "";

        if (variantType == Variant.Type.Nil) return true;

        if (variantType == Variant.Type.Int && typeof(Enum).IsAssignableFrom(type))
        {
            hint = type.GetCustomAttribute<FlagsAttribute>() != null ? PropertyHint.Flags : PropertyHint.Enum;

            // Build a string of all the enum names and values, e.g:
            // Foo:0,Bar:1,Baz:2
            StringBuilder sb = new StringBuilder();
            foreach (FieldInfo enumField in type.GetFields(BindingFlags.Public | BindingFlags.Static))
            {
                sb.Append(enumField.Name).Append(':').Append(enumField.GetRawConstantValue()).Append(',');
            }

            // Remove trailing comma
            sb.Length -= 1;
            hintString = sb.ToString();
            return true;
        }

        if (variantType == Variant.Type.Object)
        {
            if (typeof(Resource).IsAssignableFrom(type))
            {
                hint = PropertyHint.ResourceType;
                hintString = GetTypeName(type);
                return true;
            }

            if (typeof(Node).IsAssignableFrom(type))
            {
                hint = PropertyHint.NodeType;
                hintString = GetTypeName(type);
                return true;
            }

            return false;
        }

        if (variantType == Variant.Type.Array)
        {
            // No hint needed for generic arrays
            if (typeof(Godot.Collections.Array) == type)
            {
                return true;
            }

            // Lets find out what the elements should be hinted as
            Type elementType = type!.GetGenericArguments()[0];
            Variant.Type elementVariantType = GetVariantType(elementType);
            bool hasElementHint = GetPropertyHintString(elementType, elementVariantType, exportHint,
                exportHintString, out var elementHint, out var elementHintString);

            // Special case for string arrays
            hint = PropertyHint.TypeString;
            if (!GetStringArrayEnumHint(exportHint, exportHintString, elementVariantType, ref hintString))
            {
                hintString = hasElementHint
                    ? $"{(int)elementVariantType}/{(int)elementHint}:{elementHintString}"
                    : $"{(int)elementVariantType}/{(int)PropertyHint.None}:";
            }

            return true;
        }

        if (variantType == Variant.Type.PackedStringArray)
        {
            if (GetStringArrayEnumHint(exportHint, exportHintString, Variant.Type.String, ref hintString))
            {
                hint = PropertyHint.TypeString;
                return true;
            }

            return false;
        }

        if (variantType == Variant.Type.Dictionary)
        {
            // TODO: Dictionaries are not supported in the editor.
            return false;
        }

        return false;
    }

    private static string GetTypeName(Type type)
    {
        // If this is a global class, we use the class name
        if (type.GetCustomAttribute<GlobalClassAttribute>() != null)
        {
            return type.Name;
        }

        // Otherwise we find the first Godot type and use its name.
        Type nativeType = type;
        while (nativeType != null)
        {
            if (nativeType.Namespace == "Godot")
            {
                var classNameAttribute = nativeType.GetCustomAttribute<GodotClassNameAttribute>(false);
                return classNameAttribute != null ? classNameAttribute.Name : nativeType.Name;
            }

            nativeType = nativeType.BaseType;
        }

        // This shouldn't happen
        return "";
    }

    private static bool GetStringArrayEnumHint(PropertyHint hint, string hintString, Variant.Type elementVariantType,
        ref string newHint)
    {
        if (hint == PropertyHint.Enum)
        {
            newHint = $"{(int)elementVariantType}/{(int)PropertyHint.Enum}:{hintString}";
            return true;
        }

        return false;
    }

    private static Variant.Type GetVariantType(Type type)
    {
        // There's no non-generic overload for VariantUtils.TypeOf
        // But because this is only used for hints and the editor will never be AOT, this is fine.
        return (Variant.Type)typeof(VariantUtils)
            .GetMethod(nameof(VariantUtils.TypeOf),
                BindingFlags.Public | BindingFlags.Static | BindingFlags.DeclaredOnly)!
            .MakeGenericMethod(type).Invoke(null, null)!;
    }
}
