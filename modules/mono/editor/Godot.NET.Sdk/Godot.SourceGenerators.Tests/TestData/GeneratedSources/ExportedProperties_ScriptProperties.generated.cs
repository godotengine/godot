using Godot;
using Godot.NativeInterop;
using Godot.Bridge;
using System.Runtime.CompilerServices;

partial class ExportedProperties
{
#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword
    /// <summary>
    /// Cached StringNames for the properties and fields contained in this class, for fast lookup.
    /// </summary>
    public new class PropertyName : global::Godot.GodotObject.PropertyName {
        /// <summary>
        /// Cached name for the 'NotGenerateComplexLamdaProperty' property.
        /// </summary>
        public new static readonly global::Godot.StringName @NotGenerateComplexLamdaProperty = "NotGenerateComplexLamdaProperty";
        /// <summary>
        /// Cached name for the 'NotGenerateLamdaNoFieldProperty' property.
        /// </summary>
        public new static readonly global::Godot.StringName @NotGenerateLamdaNoFieldProperty = "NotGenerateLamdaNoFieldProperty";
        /// <summary>
        /// Cached name for the 'NotGenerateComplexReturnProperty' property.
        /// </summary>
        public new static readonly global::Godot.StringName @NotGenerateComplexReturnProperty = "NotGenerateComplexReturnProperty";
        /// <summary>
        /// Cached name for the 'NotGenerateReturnsProperty' property.
        /// </summary>
        public new static readonly global::Godot.StringName @NotGenerateReturnsProperty = "NotGenerateReturnsProperty";
        /// <summary>
        /// Cached name for the 'FullPropertyString' property.
        /// </summary>
        public new static readonly global::Godot.StringName @FullPropertyString = "FullPropertyString";
        /// <summary>
        /// Cached name for the 'FullPropertyString_Complex' property.
        /// </summary>
        public new static readonly global::Godot.StringName @FullPropertyString_Complex = "FullPropertyString_Complex";
        /// <summary>
        /// Cached name for the 'FullPropertyStaticImport' property.
        /// </summary>
        public new static readonly global::Godot.StringName @FullPropertyStaticImport = "FullPropertyStaticImport";
        /// <summary>
        /// Cached name for the 'LamdaPropertyString' property.
        /// </summary>
        public new static readonly global::Godot.StringName @LamdaPropertyString = "LamdaPropertyString";
        /// <summary>
        /// Cached name for the 'LambdaPropertyStaticImport' property.
        /// </summary>
        public new static readonly global::Godot.StringName @LambdaPropertyStaticImport = "LambdaPropertyStaticImport";
        /// <summary>
        /// Cached name for the 'PrimaryCtorParameter' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PrimaryCtorParameter = "PrimaryCtorParameter";
        /// <summary>
        /// Cached name for the 'ConstantMath' property.
        /// </summary>
        public new static readonly global::Godot.StringName @ConstantMath = "ConstantMath";
        /// <summary>
        /// Cached name for the 'ConstantMathStaticImport' property.
        /// </summary>
        public new static readonly global::Godot.StringName @ConstantMathStaticImport = "ConstantMathStaticImport";
        /// <summary>
        /// Cached name for the 'StaticStringAddition' property.
        /// </summary>
        public new static readonly global::Godot.StringName @StaticStringAddition = "StaticStringAddition";
        /// <summary>
        /// Cached name for the 'PropertyBoolean' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyBoolean = "PropertyBoolean";
        /// <summary>
        /// Cached name for the 'PropertyChar' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyChar = "PropertyChar";
        /// <summary>
        /// Cached name for the 'PropertySByte' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertySByte = "PropertySByte";
        /// <summary>
        /// Cached name for the 'PropertyInt16' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyInt16 = "PropertyInt16";
        /// <summary>
        /// Cached name for the 'PropertyInt32' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyInt32 = "PropertyInt32";
        /// <summary>
        /// Cached name for the 'PropertyInt64' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyInt64 = "PropertyInt64";
        /// <summary>
        /// Cached name for the 'PropertyByte' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyByte = "PropertyByte";
        /// <summary>
        /// Cached name for the 'PropertyUInt16' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyUInt16 = "PropertyUInt16";
        /// <summary>
        /// Cached name for the 'PropertyUInt32' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyUInt32 = "PropertyUInt32";
        /// <summary>
        /// Cached name for the 'PropertyUInt64' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyUInt64 = "PropertyUInt64";
        /// <summary>
        /// Cached name for the 'PropertySingle' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertySingle = "PropertySingle";
        /// <summary>
        /// Cached name for the 'PropertyDouble' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyDouble = "PropertyDouble";
        /// <summary>
        /// Cached name for the 'PropertyString' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyString = "PropertyString";
        /// <summary>
        /// Cached name for the 'PropertyVector2' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyVector2 = "PropertyVector2";
        /// <summary>
        /// Cached name for the 'PropertyVector2I' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyVector2I = "PropertyVector2I";
        /// <summary>
        /// Cached name for the 'PropertyRect2' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyRect2 = "PropertyRect2";
        /// <summary>
        /// Cached name for the 'PropertyRect2I' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyRect2I = "PropertyRect2I";
        /// <summary>
        /// Cached name for the 'PropertyTransform2D' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyTransform2D = "PropertyTransform2D";
        /// <summary>
        /// Cached name for the 'PropertyVector3' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyVector3 = "PropertyVector3";
        /// <summary>
        /// Cached name for the 'PropertyVector3I' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyVector3I = "PropertyVector3I";
        /// <summary>
        /// Cached name for the 'PropertyBasis' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyBasis = "PropertyBasis";
        /// <summary>
        /// Cached name for the 'PropertyQuaternion' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyQuaternion = "PropertyQuaternion";
        /// <summary>
        /// Cached name for the 'PropertyTransform3D' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyTransform3D = "PropertyTransform3D";
        /// <summary>
        /// Cached name for the 'PropertyVector4' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyVector4 = "PropertyVector4";
        /// <summary>
        /// Cached name for the 'PropertyVector4I' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyVector4I = "PropertyVector4I";
        /// <summary>
        /// Cached name for the 'PropertyProjection' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyProjection = "PropertyProjection";
        /// <summary>
        /// Cached name for the 'PropertyAabb' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyAabb = "PropertyAabb";
        /// <summary>
        /// Cached name for the 'PropertyColor' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyColor = "PropertyColor";
        /// <summary>
        /// Cached name for the 'PropertyPlane' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyPlane = "PropertyPlane";
        /// <summary>
        /// Cached name for the 'PropertyCallable' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyCallable = "PropertyCallable";
        /// <summary>
        /// Cached name for the 'PropertySignal' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertySignal = "PropertySignal";
        /// <summary>
        /// Cached name for the 'PropertyEnum' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyEnum = "PropertyEnum";
        /// <summary>
        /// Cached name for the 'PropertyFlagsEnum' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyFlagsEnum = "PropertyFlagsEnum";
        /// <summary>
        /// Cached name for the 'PropertyByteArray' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyByteArray = "PropertyByteArray";
        /// <summary>
        /// Cached name for the 'PropertyInt32Array' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyInt32Array = "PropertyInt32Array";
        /// <summary>
        /// Cached name for the 'PropertyInt64Array' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyInt64Array = "PropertyInt64Array";
        /// <summary>
        /// Cached name for the 'PropertySingleArray' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertySingleArray = "PropertySingleArray";
        /// <summary>
        /// Cached name for the 'PropertyDoubleArray' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyDoubleArray = "PropertyDoubleArray";
        /// <summary>
        /// Cached name for the 'PropertyStringArray' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyStringArray = "PropertyStringArray";
        /// <summary>
        /// Cached name for the 'PropertyStringArrayEnum' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyStringArrayEnum = "PropertyStringArrayEnum";
        /// <summary>
        /// Cached name for the 'PropertyVector2Array' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyVector2Array = "PropertyVector2Array";
        /// <summary>
        /// Cached name for the 'PropertyVector3Array' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyVector3Array = "PropertyVector3Array";
        /// <summary>
        /// Cached name for the 'PropertyColorArray' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyColorArray = "PropertyColorArray";
        /// <summary>
        /// Cached name for the 'PropertyGodotObjectOrDerivedArray' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyGodotObjectOrDerivedArray = "PropertyGodotObjectOrDerivedArray";
        /// <summary>
        /// Cached name for the 'field_StringNameArray' property.
        /// </summary>
        public new static readonly global::Godot.StringName @field_StringNameArray = "field_StringNameArray";
        /// <summary>
        /// Cached name for the 'field_NodePathArray' property.
        /// </summary>
        public new static readonly global::Godot.StringName @field_NodePathArray = "field_NodePathArray";
        /// <summary>
        /// Cached name for the 'field_RidArray' property.
        /// </summary>
        public new static readonly global::Godot.StringName @field_RidArray = "field_RidArray";
        /// <summary>
        /// Cached name for the 'PropertyVariant' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyVariant = "PropertyVariant";
        /// <summary>
        /// Cached name for the 'PropertyGodotObjectOrDerived' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyGodotObjectOrDerived = "PropertyGodotObjectOrDerived";
        /// <summary>
        /// Cached name for the 'PropertyGodotResourceTexture' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyGodotResourceTexture = "PropertyGodotResourceTexture";
        /// <summary>
        /// Cached name for the 'PropertyGodotResourceTextureWithInitializer' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyGodotResourceTextureWithInitializer = "PropertyGodotResourceTextureWithInitializer";
        /// <summary>
        /// Cached name for the 'PropertyStringName' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyStringName = "PropertyStringName";
        /// <summary>
        /// Cached name for the 'PropertyNodePath' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyNodePath = "PropertyNodePath";
        /// <summary>
        /// Cached name for the 'PropertyRid' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyRid = "PropertyRid";
        /// <summary>
        /// Cached name for the 'PropertyGodotDictionary' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyGodotDictionary = "PropertyGodotDictionary";
        /// <summary>
        /// Cached name for the 'PropertyGodotArray' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyGodotArray = "PropertyGodotArray";
        /// <summary>
        /// Cached name for the 'PropertyGodotGenericDictionary' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyGodotGenericDictionary = "PropertyGodotGenericDictionary";
        /// <summary>
        /// Cached name for the 'PropertyGodotGenericArray' property.
        /// </summary>
        public new static readonly global::Godot.StringName @PropertyGodotGenericArray = "PropertyGodotGenericArray";
        /// <summary>
        /// Cached name for the '_notGeneratePropertyString' field.
        /// </summary>
        public new static readonly global::Godot.StringName @_notGeneratePropertyString = "_notGeneratePropertyString";
        /// <summary>
        /// Cached name for the '_notGeneratePropertyInt' field.
        /// </summary>
        public new static readonly global::Godot.StringName @_notGeneratePropertyInt = "_notGeneratePropertyInt";
        /// <summary>
        /// Cached name for the '_fullPropertyString' field.
        /// </summary>
        public new static readonly global::Godot.StringName @_fullPropertyString = "_fullPropertyString";
        /// <summary>
        /// Cached name for the '_fullPropertyStringComplex' field.
        /// </summary>
        public new static readonly global::Godot.StringName @_fullPropertyStringComplex = "_fullPropertyStringComplex";
        /// <summary>
        /// Cached name for the '_fullPropertyStaticImport' field.
        /// </summary>
        public new static readonly global::Godot.StringName @_fullPropertyStaticImport = "_fullPropertyStaticImport";
        /// <summary>
        /// Cached name for the '_lamdaPropertyString' field.
        /// </summary>
        public new static readonly global::Godot.StringName @_lamdaPropertyString = "_lamdaPropertyString";
        /// <summary>
        /// Cached name for the '_lambdaPropertyStaticImport' field.
        /// </summary>
        public new static readonly global::Godot.StringName @_lambdaPropertyStaticImport = "_lambdaPropertyStaticImport";
    }
#pragma warning restore CS0109 // Disable warning about redundant 'new' keyword

#pragma warning disable CS0618 // Type or member is obsolete
    protected new static readonly ScriptPropertyRegistry<ExportedProperties> PropertyRegistry = new ScriptPropertyRegistry<ExportedProperties>()
        .Register(global::Godot.GodotObject.PropertyRegistry)
        .Register(PropertyName.@NotGenerateComplexLamdaProperty, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@NotGenerateComplexLamdaProperty = global::Godot.NativeInterop.VariantUtils.ConvertTo<string>(value);
                return value;
            })
        .Register(PropertyName.@NotGenerateLamdaNoFieldProperty, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@NotGenerateLamdaNoFieldProperty = global::Godot.NativeInterop.VariantUtils.ConvertTo<string>(value);
                return value;
            })
        .Register(PropertyName.@NotGenerateComplexReturnProperty, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@NotGenerateComplexReturnProperty = global::Godot.NativeInterop.VariantUtils.ConvertTo<string>(value);
                return value;
            })
        .Register(PropertyName.@NotGenerateReturnsProperty, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@NotGenerateReturnsProperty = global::Godot.NativeInterop.VariantUtils.ConvertTo<string>(value);
                return value;
            })
        .Register(PropertyName.@FullPropertyString, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@FullPropertyString = global::Godot.NativeInterop.VariantUtils.ConvertTo<string>(value);
                return value;
            })
        .Register(PropertyName.@FullPropertyString_Complex, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@FullPropertyString_Complex = global::Godot.NativeInterop.VariantUtils.ConvertTo<string>(value);
                return value;
            })
        .Register(PropertyName.@FullPropertyStaticImport, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@FullPropertyStaticImport = global::Godot.NativeInterop.VariantUtils.ConvertTo<float>(value);
                return value;
            })
        .Register(PropertyName.@LamdaPropertyString, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@LamdaPropertyString = global::Godot.NativeInterop.VariantUtils.ConvertTo<string>(value);
                return value;
            })
        .Register(PropertyName.@LambdaPropertyStaticImport, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@LambdaPropertyStaticImport = global::Godot.NativeInterop.VariantUtils.ConvertTo<float>(value);
                return value;
            })
        .Register(PropertyName.@PrimaryCtorParameter, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PrimaryCtorParameter = global::Godot.NativeInterop.VariantUtils.ConvertTo<string>(value);
                return value;
            })
        .Register(PropertyName.@ConstantMath, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@ConstantMath = global::Godot.NativeInterop.VariantUtils.ConvertTo<float>(value);
                return value;
            })
        .Register(PropertyName.@ConstantMathStaticImport, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@ConstantMathStaticImport = global::Godot.NativeInterop.VariantUtils.ConvertTo<float>(value);
                return value;
            })
        .Register(PropertyName.@StaticStringAddition, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@StaticStringAddition = global::Godot.NativeInterop.VariantUtils.ConvertTo<string>(value);
                return value;
            })
        .Register(PropertyName.@PropertyBoolean, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyBoolean = global::Godot.NativeInterop.VariantUtils.ConvertTo<bool>(value);
                return value;
            })
        .Register(PropertyName.@PropertyChar, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyChar = global::Godot.NativeInterop.VariantUtils.ConvertTo<char>(value);
                return value;
            })
        .Register(PropertyName.@PropertySByte, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertySByte = global::Godot.NativeInterop.VariantUtils.ConvertTo<sbyte>(value);
                return value;
            })
        .Register(PropertyName.@PropertyInt16, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyInt16 = global::Godot.NativeInterop.VariantUtils.ConvertTo<short>(value);
                return value;
            })
        .Register(PropertyName.@PropertyInt32, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyInt32 = global::Godot.NativeInterop.VariantUtils.ConvertTo<int>(value);
                return value;
            })
        .Register(PropertyName.@PropertyInt64, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyInt64 = global::Godot.NativeInterop.VariantUtils.ConvertTo<long>(value);
                return value;
            })
        .Register(PropertyName.@PropertyByte, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyByte = global::Godot.NativeInterop.VariantUtils.ConvertTo<byte>(value);
                return value;
            })
        .Register(PropertyName.@PropertyUInt16, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyUInt16 = global::Godot.NativeInterop.VariantUtils.ConvertTo<ushort>(value);
                return value;
            })
        .Register(PropertyName.@PropertyUInt32, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyUInt32 = global::Godot.NativeInterop.VariantUtils.ConvertTo<uint>(value);
                return value;
            })
        .Register(PropertyName.@PropertyUInt64, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyUInt64 = global::Godot.NativeInterop.VariantUtils.ConvertTo<ulong>(value);
                return value;
            })
        .Register(PropertyName.@PropertySingle, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertySingle = global::Godot.NativeInterop.VariantUtils.ConvertTo<float>(value);
                return value;
            })
        .Register(PropertyName.@PropertyDouble, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyDouble = global::Godot.NativeInterop.VariantUtils.ConvertTo<double>(value);
                return value;
            })
        .Register(PropertyName.@PropertyString, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyString = global::Godot.NativeInterop.VariantUtils.ConvertTo<string>(value);
                return value;
            })
        .Register(PropertyName.@PropertyVector2, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyVector2 = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Vector2>(value);
                return value;
            })
        .Register(PropertyName.@PropertyVector2I, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyVector2I = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Vector2I>(value);
                return value;
            })
        .Register(PropertyName.@PropertyRect2, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyRect2 = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Rect2>(value);
                return value;
            })
        .Register(PropertyName.@PropertyRect2I, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyRect2I = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Rect2I>(value);
                return value;
            })
        .Register(PropertyName.@PropertyTransform2D, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyTransform2D = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Transform2D>(value);
                return value;
            })
        .Register(PropertyName.@PropertyVector3, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyVector3 = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Vector3>(value);
                return value;
            })
        .Register(PropertyName.@PropertyVector3I, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyVector3I = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Vector3I>(value);
                return value;
            })
        .Register(PropertyName.@PropertyBasis, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyBasis = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Basis>(value);
                return value;
            })
        .Register(PropertyName.@PropertyQuaternion, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyQuaternion = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Quaternion>(value);
                return value;
            })
        .Register(PropertyName.@PropertyTransform3D, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyTransform3D = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Transform3D>(value);
                return value;
            })
        .Register(PropertyName.@PropertyVector4, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyVector4 = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Vector4>(value);
                return value;
            })
        .Register(PropertyName.@PropertyVector4I, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyVector4I = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Vector4I>(value);
                return value;
            })
        .Register(PropertyName.@PropertyProjection, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyProjection = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Projection>(value);
                return value;
            })
        .Register(PropertyName.@PropertyAabb, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyAabb = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Aabb>(value);
                return value;
            })
        .Register(PropertyName.@PropertyColor, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyColor = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Color>(value);
                return value;
            })
        .Register(PropertyName.@PropertyPlane, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyPlane = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Plane>(value);
                return value;
            })
        .Register(PropertyName.@PropertyCallable, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyCallable = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Callable>(value);
                return value;
            })
        .Register(PropertyName.@PropertySignal, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertySignal = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Signal>(value);
                return value;
            })
        .Register(PropertyName.@PropertyEnum, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyEnum = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::ExportedProperties.MyEnum>(value);
                return value;
            })
        .Register(PropertyName.@PropertyFlagsEnum, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyFlagsEnum = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::ExportedProperties.MyFlagsEnum>(value);
                return value;
            })
        .Register(PropertyName.@PropertyByteArray, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyByteArray = global::Godot.NativeInterop.VariantUtils.ConvertTo<byte[]>(value);
                return value;
            })
        .Register(PropertyName.@PropertyInt32Array, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyInt32Array = global::Godot.NativeInterop.VariantUtils.ConvertTo<int[]>(value);
                return value;
            })
        .Register(PropertyName.@PropertyInt64Array, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyInt64Array = global::Godot.NativeInterop.VariantUtils.ConvertTo<long[]>(value);
                return value;
            })
        .Register(PropertyName.@PropertySingleArray, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertySingleArray = global::Godot.NativeInterop.VariantUtils.ConvertTo<float[]>(value);
                return value;
            })
        .Register(PropertyName.@PropertyDoubleArray, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyDoubleArray = global::Godot.NativeInterop.VariantUtils.ConvertTo<double[]>(value);
                return value;
            })
        .Register(PropertyName.@PropertyStringArray, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyStringArray = global::Godot.NativeInterop.VariantUtils.ConvertTo<string[]>(value);
                return value;
            })
        .Register(PropertyName.@PropertyStringArrayEnum, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyStringArrayEnum = global::Godot.NativeInterop.VariantUtils.ConvertTo<string[]>(value);
                return value;
            })
        .Register(PropertyName.@PropertyVector2Array, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyVector2Array = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Vector2[]>(value);
                return value;
            })
        .Register(PropertyName.@PropertyVector3Array, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyVector3Array = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Vector3[]>(value);
                return value;
            })
        .Register(PropertyName.@PropertyColorArray, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyColorArray = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Color[]>(value);
                return value;
            })
        .Register(PropertyName.@PropertyGodotObjectOrDerivedArray, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyGodotObjectOrDerivedArray = global::Godot.NativeInterop.VariantUtils.ConvertToSystemArrayOfGodotObject<global::Godot.GodotObject>(value);
                return value;
            })
        .Register(PropertyName.@field_StringNameArray, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@field_StringNameArray = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.StringName[]>(value);
                return value;
            })
        .Register(PropertyName.@field_NodePathArray, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@field_NodePathArray = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.NodePath[]>(value);
                return value;
            })
        .Register(PropertyName.@field_RidArray, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@field_RidArray = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Rid[]>(value);
                return value;
            })
        .Register(PropertyName.@PropertyVariant, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyVariant = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Variant>(value);
                return value;
            })
        .Register(PropertyName.@PropertyGodotObjectOrDerived, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyGodotObjectOrDerived = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.GodotObject>(value);
                return value;
            })
        .Register(PropertyName.@PropertyGodotResourceTexture, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyGodotResourceTexture = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Texture>(value);
                return value;
            })
        .Register(PropertyName.@PropertyGodotResourceTextureWithInitializer, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyGodotResourceTextureWithInitializer = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Texture>(value);
                return value;
            })
        .Register(PropertyName.@PropertyStringName, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyStringName = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.StringName>(value);
                return value;
            })
        .Register(PropertyName.@PropertyNodePath, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyNodePath = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.NodePath>(value);
                return value;
            })
        .Register(PropertyName.@PropertyRid, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyRid = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Rid>(value);
                return value;
            })
        .Register(PropertyName.@PropertyGodotDictionary, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyGodotDictionary = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Collections.Dictionary>(value);
                return value;
            })
        .Register(PropertyName.@PropertyGodotArray, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyGodotArray = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Collections.Array>(value);
                return value;
            })
        .Register(PropertyName.@PropertyGodotGenericDictionary, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyGodotGenericDictionary = global::Godot.NativeInterop.VariantUtils.ConvertToDictionary<string, bool>(value);
                return value;
            })
        .Register(PropertyName.@PropertyGodotGenericArray, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyGodotGenericArray = global::Godot.NativeInterop.VariantUtils.ConvertToArray<int>(value);
                return value;
            })
        .Register(PropertyName.@_notGeneratePropertyString, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@_notGeneratePropertyString = global::Godot.NativeInterop.VariantUtils.ConvertTo<string>(value);
                return value;
            })
        .Register(PropertyName.@_notGeneratePropertyInt, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@_notGeneratePropertyInt = global::Godot.NativeInterop.VariantUtils.ConvertTo<int>(value);
                return value;
            })
        .Register(PropertyName.@_fullPropertyString, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@_fullPropertyString = global::Godot.NativeInterop.VariantUtils.ConvertTo<string>(value);
                return value;
            })
        .Register(PropertyName.@_fullPropertyStringComplex, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@_fullPropertyStringComplex = global::Godot.NativeInterop.VariantUtils.ConvertTo<string>(value);
                return value;
            })
        .Register(PropertyName.@_fullPropertyStaticImport, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@_fullPropertyStaticImport = global::Godot.NativeInterop.VariantUtils.ConvertTo<float>(value);
                return value;
            })
        .Register(PropertyName.@_lamdaPropertyString, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@_lamdaPropertyString = global::Godot.NativeInterop.VariantUtils.ConvertTo<string>(value);
                return value;
            })
        .Register(PropertyName.@_lambdaPropertyStaticImport, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@_lambdaPropertyStaticImport = global::Godot.NativeInterop.VariantUtils.ConvertTo<float>(value);
                return value;
            })
        .Register(PropertyName.@NotGenerateComplexLamdaProperty, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@NotGenerateComplexLamdaProperty;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            })
        .Register(PropertyName.@NotGenerateLamdaNoFieldProperty, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@NotGenerateLamdaNoFieldProperty;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            })
        .Register(PropertyName.@NotGenerateComplexReturnProperty, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@NotGenerateComplexReturnProperty;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            })
        .Register(PropertyName.@NotGenerateReturnsProperty, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@NotGenerateReturnsProperty;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            })
        .Register(PropertyName.@FullPropertyString, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@FullPropertyString;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            })
        .Register(PropertyName.@FullPropertyString_Complex, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@FullPropertyString_Complex;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            })
        .Register(PropertyName.@FullPropertyStaticImport, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@FullPropertyStaticImport;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<float>(ret);
            })
        .Register(PropertyName.@LamdaPropertyString, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@LamdaPropertyString;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            })
        .Register(PropertyName.@LambdaPropertyStaticImport, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@LambdaPropertyStaticImport;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<float>(ret);
            })
        .Register(PropertyName.@PrimaryCtorParameter, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PrimaryCtorParameter;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            })
        .Register(PropertyName.@ConstantMath, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@ConstantMath;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<float>(ret);
            })
        .Register(PropertyName.@ConstantMathStaticImport, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@ConstantMathStaticImport;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<float>(ret);
            })
        .Register(PropertyName.@StaticStringAddition, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@StaticStringAddition;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            })
        .Register(PropertyName.@PropertyBoolean, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyBoolean;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<bool>(ret);
            })
        .Register(PropertyName.@PropertyChar, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyChar;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<char>(ret);
            })
        .Register(PropertyName.@PropertySByte, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertySByte;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<sbyte>(ret);
            })
        .Register(PropertyName.@PropertyInt16, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyInt16;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<short>(ret);
            })
        .Register(PropertyName.@PropertyInt32, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyInt32;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<int>(ret);
            })
        .Register(PropertyName.@PropertyInt64, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyInt64;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<long>(ret);
            })
        .Register(PropertyName.@PropertyByte, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyByte;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<byte>(ret);
            })
        .Register(PropertyName.@PropertyUInt16, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyUInt16;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<ushort>(ret);
            })
        .Register(PropertyName.@PropertyUInt32, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyUInt32;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<uint>(ret);
            })
        .Register(PropertyName.@PropertyUInt64, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyUInt64;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<ulong>(ret);
            })
        .Register(PropertyName.@PropertySingle, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertySingle;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<float>(ret);
            })
        .Register(PropertyName.@PropertyDouble, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyDouble;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<double>(ret);
            })
        .Register(PropertyName.@PropertyString, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyString;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            })
        .Register(PropertyName.@PropertyVector2, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyVector2;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Vector2>(ret);
            })
        .Register(PropertyName.@PropertyVector2I, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyVector2I;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Vector2I>(ret);
            })
        .Register(PropertyName.@PropertyRect2, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyRect2;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Rect2>(ret);
            })
        .Register(PropertyName.@PropertyRect2I, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyRect2I;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Rect2I>(ret);
            })
        .Register(PropertyName.@PropertyTransform2D, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyTransform2D;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Transform2D>(ret);
            })
        .Register(PropertyName.@PropertyVector3, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyVector3;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Vector3>(ret);
            })
        .Register(PropertyName.@PropertyVector3I, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyVector3I;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Vector3I>(ret);
            })
        .Register(PropertyName.@PropertyBasis, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyBasis;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Basis>(ret);
            })
        .Register(PropertyName.@PropertyQuaternion, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyQuaternion;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Quaternion>(ret);
            })
        .Register(PropertyName.@PropertyTransform3D, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyTransform3D;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Transform3D>(ret);
            })
        .Register(PropertyName.@PropertyVector4, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyVector4;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Vector4>(ret);
            })
        .Register(PropertyName.@PropertyVector4I, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyVector4I;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Vector4I>(ret);
            })
        .Register(PropertyName.@PropertyProjection, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyProjection;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Projection>(ret);
            })
        .Register(PropertyName.@PropertyAabb, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyAabb;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Aabb>(ret);
            })
        .Register(PropertyName.@PropertyColor, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyColor;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Color>(ret);
            })
        .Register(PropertyName.@PropertyPlane, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyPlane;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Plane>(ret);
            })
        .Register(PropertyName.@PropertyCallable, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyCallable;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Callable>(ret);
            })
        .Register(PropertyName.@PropertySignal, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertySignal;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Signal>(ret);
            })
        .Register(PropertyName.@PropertyEnum, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyEnum;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::ExportedProperties.MyEnum>(ret);
            })
        .Register(PropertyName.@PropertyFlagsEnum, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyFlagsEnum;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::ExportedProperties.MyFlagsEnum>(ret);
            })
        .Register(PropertyName.@PropertyByteArray, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyByteArray;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<byte[]>(ret);
            })
        .Register(PropertyName.@PropertyInt32Array, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyInt32Array;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<int[]>(ret);
            })
        .Register(PropertyName.@PropertyInt64Array, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyInt64Array;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<long[]>(ret);
            })
        .Register(PropertyName.@PropertySingleArray, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertySingleArray;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<float[]>(ret);
            })
        .Register(PropertyName.@PropertyDoubleArray, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyDoubleArray;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<double[]>(ret);
            })
        .Register(PropertyName.@PropertyStringArray, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyStringArray;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string[]>(ret);
            })
        .Register(PropertyName.@PropertyStringArrayEnum, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyStringArrayEnum;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string[]>(ret);
            })
        .Register(PropertyName.@PropertyVector2Array, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyVector2Array;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Vector2[]>(ret);
            })
        .Register(PropertyName.@PropertyVector3Array, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyVector3Array;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Vector3[]>(ret);
            })
        .Register(PropertyName.@PropertyColorArray, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyColorArray;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Color[]>(ret);
            })
        .Register(PropertyName.@PropertyGodotObjectOrDerivedArray, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyGodotObjectOrDerivedArray;
                return global::Godot.NativeInterop.VariantUtils.CreateFromSystemArrayOfGodotObject(ret);
            })
        .Register(PropertyName.@field_StringNameArray, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@field_StringNameArray;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.StringName[]>(ret);
            })
        .Register(PropertyName.@field_NodePathArray, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@field_NodePathArray;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.NodePath[]>(ret);
            })
        .Register(PropertyName.@field_RidArray, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@field_RidArray;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Rid[]>(ret);
            })
        .Register(PropertyName.@PropertyVariant, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyVariant;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Variant>(ret);
            })
        .Register(PropertyName.@PropertyGodotObjectOrDerived, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyGodotObjectOrDerived;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.GodotObject>(ret);
            })
        .Register(PropertyName.@PropertyGodotResourceTexture, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyGodotResourceTexture;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Texture>(ret);
            })
        .Register(PropertyName.@PropertyGodotResourceTextureWithInitializer, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyGodotResourceTextureWithInitializer;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Texture>(ret);
            })
        .Register(PropertyName.@PropertyStringName, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyStringName;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.StringName>(ret);
            })
        .Register(PropertyName.@PropertyNodePath, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyNodePath;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.NodePath>(ret);
            })
        .Register(PropertyName.@PropertyRid, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyRid;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Rid>(ret);
            })
        .Register(PropertyName.@PropertyGodotDictionary, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyGodotDictionary;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Collections.Dictionary>(ret);
            })
        .Register(PropertyName.@PropertyGodotArray, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyGodotArray;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Collections.Array>(ret);
            })
        .Register(PropertyName.@PropertyGodotGenericDictionary, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyGodotGenericDictionary;
                return global::Godot.NativeInterop.VariantUtils.CreateFromDictionary(ret);
            })
        .Register(PropertyName.@PropertyGodotGenericArray, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@PropertyGodotGenericArray;
                return global::Godot.NativeInterop.VariantUtils.CreateFromArray(ret);
            })
        .Register(PropertyName.@_notGeneratePropertyString, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@_notGeneratePropertyString;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            })
        .Register(PropertyName.@_notGeneratePropertyInt, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@_notGeneratePropertyInt;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<int>(ret);
            })
        .Register(PropertyName.@_fullPropertyString, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@_fullPropertyString;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            })
        .Register(PropertyName.@_fullPropertyStringComplex, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@_fullPropertyStringComplex;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            })
        .Register(PropertyName.@_fullPropertyStaticImport, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@_fullPropertyStaticImport;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<float>(ret);
            })
        .Register(PropertyName.@_lamdaPropertyString, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@_lamdaPropertyString;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            })
        .Register(PropertyName.@_lambdaPropertyStaticImport, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportedProperties>(ref scriptInstance).@_lambdaPropertyStaticImport;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<float>(ret);
            })
        .Build();
#pragma warning restore CS0618 // Type or member is obsolete

    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    protected override bool SetGodotClassPropertyValue(in godot_string_name name, in godot_variant value)
    {
        ref readonly var propertySetter = ref PropertyRegistry.GetMethodOrNullRef(in name, 1);
        if (!Unsafe.IsNullRef(in propertySetter))
        {
            propertySetter(this, value);
            return true;
        }
        return false;
    }

    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    protected override bool GetGodotClassPropertyValue(in godot_string_name name, out godot_variant value)
    {
        ref readonly var propertyGetter = ref PropertyRegistry.GetMethodOrNullRef(in name, 0);
        if (!Unsafe.IsNullRef(in propertyGetter))
        {
            value = propertyGetter(this, default);
            return true;
        }
        value = default;
        return false;
    }

#pragma warning disable CS0109 // The member 'member' does not hide an inherited member. The new keyword is not required
    /// <summary>
    /// Get the property information for all the properties declared in this class.
    /// This method is used by Godot to register the available properties in the editor.
    /// Do not call this method.
    /// </summary>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    internal new static global::System.Collections.Generic.List<global::Godot.Bridge.PropertyInfo> GetGodotPropertyList()
    {
        var properties = new global::System.Collections.Generic.List<global::Godot.Bridge.PropertyInfo>();
        properties.Add(new(type: (global::Godot.Variant.Type)4, name: PropertyName.@_notGeneratePropertyString, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        properties.Add(new(type: (global::Godot.Variant.Type)4, name: PropertyName.@NotGenerateComplexLamdaProperty, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)4, name: PropertyName.@NotGenerateLamdaNoFieldProperty, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)4, name: PropertyName.@NotGenerateComplexReturnProperty, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)2, name: PropertyName.@_notGeneratePropertyInt, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        properties.Add(new(type: (global::Godot.Variant.Type)4, name: PropertyName.@NotGenerateReturnsProperty, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)4, name: PropertyName.@_fullPropertyString, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        properties.Add(new(type: (global::Godot.Variant.Type)4, name: PropertyName.@FullPropertyString, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)4, name: PropertyName.@_fullPropertyStringComplex, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        properties.Add(new(type: (global::Godot.Variant.Type)4, name: PropertyName.@FullPropertyString_Complex, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)3, name: PropertyName.@_fullPropertyStaticImport, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        properties.Add(new(type: (global::Godot.Variant.Type)3, name: PropertyName.@FullPropertyStaticImport, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)4, name: PropertyName.@_lamdaPropertyString, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        properties.Add(new(type: (global::Godot.Variant.Type)4, name: PropertyName.@LamdaPropertyString, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)3, name: PropertyName.@_lambdaPropertyStaticImport, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        properties.Add(new(type: (global::Godot.Variant.Type)3, name: PropertyName.@LambdaPropertyStaticImport, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)4, name: PropertyName.@PrimaryCtorParameter, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)3, name: PropertyName.@ConstantMath, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)3, name: PropertyName.@ConstantMathStaticImport, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)4, name: PropertyName.@StaticStringAddition, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)1, name: PropertyName.@PropertyBoolean, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)2, name: PropertyName.@PropertyChar, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)2, name: PropertyName.@PropertySByte, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)2, name: PropertyName.@PropertyInt16, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)2, name: PropertyName.@PropertyInt32, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)2, name: PropertyName.@PropertyInt64, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)2, name: PropertyName.@PropertyByte, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)2, name: PropertyName.@PropertyUInt16, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)2, name: PropertyName.@PropertyUInt32, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)2, name: PropertyName.@PropertyUInt64, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)3, name: PropertyName.@PropertySingle, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)3, name: PropertyName.@PropertyDouble, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)4, name: PropertyName.@PropertyString, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)5, name: PropertyName.@PropertyVector2, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)6, name: PropertyName.@PropertyVector2I, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)7, name: PropertyName.@PropertyRect2, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)8, name: PropertyName.@PropertyRect2I, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)11, name: PropertyName.@PropertyTransform2D, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)9, name: PropertyName.@PropertyVector3, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)10, name: PropertyName.@PropertyVector3I, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)17, name: PropertyName.@PropertyBasis, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)15, name: PropertyName.@PropertyQuaternion, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)18, name: PropertyName.@PropertyTransform3D, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)12, name: PropertyName.@PropertyVector4, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)13, name: PropertyName.@PropertyVector4I, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)19, name: PropertyName.@PropertyProjection, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)16, name: PropertyName.@PropertyAabb, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)20, name: PropertyName.@PropertyColor, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)14, name: PropertyName.@PropertyPlane, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)25, name: PropertyName.@PropertyCallable, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)26, name: PropertyName.@PropertySignal, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)2, name: PropertyName.@PropertyEnum, hint: (global::Godot.PropertyHint)2, hintString: "A,B,C", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)2, name: PropertyName.@PropertyFlagsEnum, hint: (global::Godot.PropertyHint)6, hintString: "A:0,B:1,C:2", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)29, name: PropertyName.@PropertyByteArray, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)30, name: PropertyName.@PropertyInt32Array, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)31, name: PropertyName.@PropertyInt64Array, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)32, name: PropertyName.@PropertySingleArray, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)33, name: PropertyName.@PropertyDoubleArray, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)34, name: PropertyName.@PropertyStringArray, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)34, name: PropertyName.@PropertyStringArrayEnum, hint: (global::Godot.PropertyHint)23, hintString: "4/2:A,B,C", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)35, name: PropertyName.@PropertyVector2Array, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)36, name: PropertyName.@PropertyVector3Array, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)37, name: PropertyName.@PropertyColorArray, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)28, name: PropertyName.@PropertyGodotObjectOrDerivedArray, hint: (global::Godot.PropertyHint)23, hintString: "24/0:", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)28, name: PropertyName.@field_StringNameArray, hint: (global::Godot.PropertyHint)23, hintString: "21/0:", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)28, name: PropertyName.@field_NodePathArray, hint: (global::Godot.PropertyHint)23, hintString: "22/0:", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)28, name: PropertyName.@field_RidArray, hint: (global::Godot.PropertyHint)23, hintString: "23/0:", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)0, name: PropertyName.@PropertyVariant, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)135174, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)24, name: PropertyName.@PropertyGodotObjectOrDerived, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)24, name: PropertyName.@PropertyGodotResourceTexture, hint: (global::Godot.PropertyHint)17, hintString: "Texture", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)24, name: PropertyName.@PropertyGodotResourceTextureWithInitializer, hint: (global::Godot.PropertyHint)17, hintString: "Texture", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)21, name: PropertyName.@PropertyStringName, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)22, name: PropertyName.@PropertyNodePath, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)23, name: PropertyName.@PropertyRid, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)27, name: PropertyName.@PropertyGodotDictionary, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)28, name: PropertyName.@PropertyGodotArray, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)27, name: PropertyName.@PropertyGodotGenericDictionary, hint: (global::Godot.PropertyHint)23, hintString: "4/0:;1/0:", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)28, name: PropertyName.@PropertyGodotGenericArray, hint: (global::Godot.PropertyHint)23, hintString: "2/0:", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        return properties;
    }
#pragma warning restore CS0109 // The member 'member' does not hide an inherited member. The new keyword is not required
}
