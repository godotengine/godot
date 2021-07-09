using System;
using System.Collections.Generic;

namespace Godot
{
    /// <summary>
    /// This class contains color constants created from standardized color names.
    /// The standardized color set is based on the X11 and .NET color names.
    /// </summary>
    public static class Colors
    {
        // Color names and values are derived from core/color_names.inc
        internal static readonly Dictionary<string, Color> NamedColors = new Dictionary<string, Color>
        {
            { "aliceblue", new Color(0.94f, 0.97f, 1.00f) },
            { "antiquewhite", new Color(0.98f, 0.92f, 0.84f) },
            { "aqua", new Color(0.00f, 1.00f, 1.00f) },
            { "aquamarine", new Color(0.50f, 1.00f, 0.83f) },
            { "azure", new Color(0.94f, 1.00f, 1.00f) },
            { "beige", new Color(0.96f, 0.96f, 0.86f) },
            { "bisque", new Color(1.00f, 0.89f, 0.77f) },
            { "black", new Color(0.00f, 0.00f, 0.00f) },
            { "blanchedalmond", new Color(1.00f, 0.92f, 0.80f) },
            { "blue", new Color(0.00f, 0.00f, 1.00f) },
            { "blueviolet", new Color(0.54f, 0.17f, 0.89f) },
            { "brown", new Color(0.65f, 0.16f, 0.16f) },
            { "burlywood", new Color(0.87f, 0.72f, 0.53f) },
            { "cadetblue", new Color(0.37f, 0.62f, 0.63f) },
            { "chartreuse", new Color(0.50f, 1.00f, 0.00f) },
            { "chocolate", new Color(0.82f, 0.41f, 0.12f) },
            { "coral", new Color(1.00f, 0.50f, 0.31f) },
            { "cornflower", new Color(0.39f, 0.58f, 0.93f) },
            { "cornsilk", new Color(1.00f, 0.97f, 0.86f) },
            { "crimson", new Color(0.86f, 0.08f, 0.24f) },
            { "cyan", new Color(0.00f, 1.00f, 1.00f) },
            { "darkblue", new Color(0.00f, 0.00f, 0.55f) },
            { "darkcyan", new Color(0.00f, 0.55f, 0.55f) },
            { "darkgoldenrod", new Color(0.72f, 0.53f, 0.04f) },
            { "darkgray", new Color(0.66f, 0.66f, 0.66f) },
            { "darkgreen", new Color(0.00f, 0.39f, 0.00f) },
            { "darkkhaki", new Color(0.74f, 0.72f, 0.42f) },
            { "darkmagenta", new Color(0.55f, 0.00f, 0.55f) },
            { "darkolivegreen", new Color(0.33f, 0.42f, 0.18f) },
            { "darkorange", new Color(1.00f, 0.55f, 0.00f) },
            { "darkorchid", new Color(0.60f, 0.20f, 0.80f) },
            { "darkred", new Color(0.55f, 0.00f, 0.00f) },
            { "darksalmon", new Color(0.91f, 0.59f, 0.48f) },
            { "darkseagreen", new Color(0.56f, 0.74f, 0.56f) },
            { "darkslateblue", new Color(0.28f, 0.24f, 0.55f) },
            { "darkslategray", new Color(0.18f, 0.31f, 0.31f) },
            { "darkturquoise", new Color(0.00f, 0.81f, 0.82f) },
            { "darkviolet", new Color(0.58f, 0.00f, 0.83f) },
            { "deeppink", new Color(1.00f, 0.08f, 0.58f) },
            { "deepskyblue", new Color(0.00f, 0.75f, 1.00f) },
            { "dimgray", new Color(0.41f, 0.41f, 0.41f) },
            { "dodgerblue", new Color(0.12f, 0.56f, 1.00f) },
            { "firebrick", new Color(0.70f, 0.13f, 0.13f) },
            { "floralwhite", new Color(1.00f, 0.98f, 0.94f) },
            { "forestgreen", new Color(0.13f, 0.55f, 0.13f) },
            { "fuchsia", new Color(1.00f, 0.00f, 1.00f) },
            { "gainsboro", new Color(0.86f, 0.86f, 0.86f) },
            { "ghostwhite", new Color(0.97f, 0.97f, 1.00f) },
            { "gold", new Color(1.00f, 0.84f, 0.00f) },
            { "goldenrod", new Color(0.85f, 0.65f, 0.13f) },
            { "gray", new Color(0.75f, 0.75f, 0.75f) },
            { "green", new Color(0.00f, 1.00f, 0.00f) },
            { "greenyellow", new Color(0.68f, 1.00f, 0.18f) },
            { "honeydew", new Color(0.94f, 1.00f, 0.94f) },
            { "hotpink", new Color(1.00f, 0.41f, 0.71f) },
            { "indianred", new Color(0.80f, 0.36f, 0.36f) },
            { "indigo", new Color(0.29f, 0.00f, 0.51f) },
            { "ivory", new Color(1.00f, 1.00f, 0.94f) },
            { "khaki", new Color(0.94f, 0.90f, 0.55f) },
            { "lavender", new Color(0.90f, 0.90f, 0.98f) },
            { "lavenderblush", new Color(1.00f, 0.94f, 0.96f) },
            { "lawngreen", new Color(0.49f, 0.99f, 0.00f) },
            { "lemonchiffon", new Color(1.00f, 0.98f, 0.80f) },
            { "lightblue", new Color(0.68f, 0.85f, 0.90f) },
            { "lightcoral", new Color(0.94f, 0.50f, 0.50f) },
            { "lightcyan", new Color(0.88f, 1.00f, 1.00f) },
            { "lightgoldenrod", new Color(0.98f, 0.98f, 0.82f) },
            { "lightgray", new Color(0.83f, 0.83f, 0.83f) },
            { "lightgreen", new Color(0.56f, 0.93f, 0.56f) },
            { "lightpink", new Color(1.00f, 0.71f, 0.76f) },
            { "lightsalmon", new Color(1.00f, 0.63f, 0.48f) },
            { "lightseagreen", new Color(0.13f, 0.70f, 0.67f) },
            { "lightskyblue", new Color(0.53f, 0.81f, 0.98f) },
            { "lightslategray", new Color(0.47f, 0.53f, 0.60f) },
            { "lightsteelblue", new Color(0.69f, 0.77f, 0.87f) },
            { "lightyellow", new Color(1.00f, 1.00f, 0.88f) },
            { "lime", new Color(0.00f, 1.00f, 0.00f) },
            { "limegreen", new Color(0.20f, 0.80f, 0.20f) },
            { "linen", new Color(0.98f, 0.94f, 0.90f) },
            { "magenta", new Color(1.00f, 0.00f, 1.00f) },
            { "maroon", new Color(0.69f, 0.19f, 0.38f) },
            { "mediumaquamarine", new Color(0.40f, 0.80f, 0.67f) },
            { "mediumblue", new Color(0.00f, 0.00f, 0.80f) },
            { "mediumorchid", new Color(0.73f, 0.33f, 0.83f) },
            { "mediumpurple", new Color(0.58f, 0.44f, 0.86f) },
            { "mediumseagreen", new Color(0.24f, 0.70f, 0.44f) },
            { "mediumslateblue", new Color(0.48f, 0.41f, 0.93f) },
            { "mediumspringgreen", new Color(0.00f, 0.98f, 0.60f) },
            { "mediumturquoise", new Color(0.28f, 0.82f, 0.80f) },
            { "mediumvioletred", new Color(0.78f, 0.08f, 0.52f) },
            { "midnightblue", new Color(0.10f, 0.10f, 0.44f) },
            { "mintcream", new Color(0.96f, 1.00f, 0.98f) },
            { "mistyrose", new Color(1.00f, 0.89f, 0.88f) },
            { "moccasin", new Color(1.00f, 0.89f, 0.71f) },
            { "navajowhite", new Color(1.00f, 0.87f, 0.68f) },
            { "navyblue", new Color(0.00f, 0.00f, 0.50f) },
            { "oldlace", new Color(0.99f, 0.96f, 0.90f) },
            { "olive", new Color(0.50f, 0.50f, 0.00f) },
            { "olivedrab", new Color(0.42f, 0.56f, 0.14f) },
            { "orange", new Color(1.00f, 0.65f, 0.00f) },
            { "orangered", new Color(1.00f, 0.27f, 0.00f) },
            { "orchid", new Color(0.85f, 0.44f, 0.84f) },
            { "palegoldenrod", new Color(0.93f, 0.91f, 0.67f) },
            { "palegreen", new Color(0.60f, 0.98f, 0.60f) },
            { "paleturquoise", new Color(0.69f, 0.93f, 0.93f) },
            { "palevioletred", new Color(0.86f, 0.44f, 0.58f) },
            { "papayawhip", new Color(1.00f, 0.94f, 0.84f) },
            { "peachpuff", new Color(1.00f, 0.85f, 0.73f) },
            { "peru", new Color(0.80f, 0.52f, 0.25f) },
            { "pink", new Color(1.00f, 0.75f, 0.80f) },
            { "plum", new Color(0.87f, 0.63f, 0.87f) },
            { "powderblue", new Color(0.69f, 0.88f, 0.90f) },
            { "purple", new Color(0.63f, 0.13f, 0.94f) },
            { "rebeccapurple", new Color(0.40f, 0.20f, 0.60f) },
            { "red", new Color(1.00f, 0.00f, 0.00f) },
            { "rosybrown", new Color(0.74f, 0.56f, 0.56f) },
            { "royalblue", new Color(0.25f, 0.41f, 0.88f) },
            { "saddlebrown", new Color(0.55f, 0.27f, 0.07f) },
            { "salmon", new Color(0.98f, 0.50f, 0.45f) },
            { "sandybrown", new Color(0.96f, 0.64f, 0.38f) },
            { "seagreen", new Color(0.18f, 0.55f, 0.34f) },
            { "seashell", new Color(1.00f, 0.96f, 0.93f) },
            { "sienna", new Color(0.63f, 0.32f, 0.18f) },
            { "silver", new Color(0.75f, 0.75f, 0.75f) },
            { "skyblue", new Color(0.53f, 0.81f, 0.92f) },
            { "slateblue", new Color(0.42f, 0.35f, 0.80f) },
            { "slategray", new Color(0.44f, 0.50f, 0.56f) },
            { "snow", new Color(1.00f, 0.98f, 0.98f) },
            { "springgreen", new Color(0.00f, 1.00f, 0.50f) },
            { "steelblue", new Color(0.27f, 0.51f, 0.71f) },
            { "tan", new Color(0.82f, 0.71f, 0.55f) },
            { "teal", new Color(0.00f, 0.50f, 0.50f) },
            { "thistle", new Color(0.85f, 0.75f, 0.85f) },
            { "tomato", new Color(1.00f, 0.39f, 0.28f) },
            { "transparent", new Color(1.00f, 1.00f, 1.00f, 0.00f) },
            { "turquoise", new Color(0.25f, 0.88f, 0.82f) },
            { "violet", new Color(0.93f, 0.51f, 0.93f) },
            { "webgreen", new Color(0.00f, 0.50f, 0.00f) },
            { "webgray", new Color(0.50f, 0.50f, 0.50f) },
            { "webmaroon", new Color(0.50f, 0.00f, 0.00f) },
            { "webpurple", new Color(0.50f, 0.00f, 0.50f) },
            { "wheat", new Color(0.96f, 0.87f, 0.70f) },
            { "white", new Color(1.00f, 1.00f, 1.00f) },
            { "whitesmoke", new Color(0.96f, 0.96f, 0.96f) },
            { "yellow", new Color(1.00f, 1.00f, 0.00f) },
            { "yellowgreen", new Color(0.60f, 0.80f, 0.20f) },
        };

        public static Color AliceBlue => NamedColors["aliceblue"];
        public static Color AntiqueWhite => NamedColors["antiquewhite"];
        public static Color Aqua => NamedColors["aqua"];
        public static Color Aquamarine => NamedColors["aquamarine"];
        public static Color Azure => NamedColors["azure"];
        public static Color Beige => NamedColors["beige"];
        public static Color Bisque => NamedColors["bisque"];
        public static Color Black => NamedColors["black"];
        public static Color BlanchedAlmond => NamedColors["blanchedalmond"];
        public static Color Blue => NamedColors["blue"];
        public static Color BlueViolet => NamedColors["blueviolet"];
        public static Color Brown => NamedColors["brown"];
        public static Color BurlyWood => NamedColors["burlywood"];
        public static Color CadetBlue => NamedColors["cadetblue"];
        public static Color Chartreuse => NamedColors["chartreuse"];
        public static Color Chocolate => NamedColors["chocolate"];
        public static Color Coral => NamedColors["coral"];
        public static Color Cornflower => NamedColors["cornflower"];
        public static Color Cornsilk => NamedColors["cornsilk"];
        public static Color Crimson => NamedColors["crimson"];
        public static Color Cyan => NamedColors["cyan"];
        public static Color DarkBlue => NamedColors["darkblue"];
        public static Color DarkCyan => NamedColors["darkcyan"];
        public static Color DarkGoldenrod => NamedColors["darkgoldenrod"];
        public static Color DarkGray => NamedColors["darkgray"];
        public static Color DarkGreen => NamedColors["darkgreen"];
        public static Color DarkKhaki => NamedColors["darkkhaki"];
        public static Color DarkMagenta => NamedColors["darkmagenta"];
        public static Color DarkOliveGreen => NamedColors["darkolivegreen"];
        public static Color DarkOrange => NamedColors["darkorange"];
        public static Color DarkOrchid => NamedColors["darkorchid"];
        public static Color DarkRed => NamedColors["darkred"];
        public static Color DarkSalmon => NamedColors["darksalmon"];
        public static Color DarkSeaGreen => NamedColors["darkseagreen"];
        public static Color DarkSlateBlue => NamedColors["darkslateblue"];
        public static Color DarkSlateGray => NamedColors["darkslategray"];
        public static Color DarkTurquoise => NamedColors["darkturquoise"];
        public static Color DarkViolet => NamedColors["darkviolet"];
        public static Color DeepPink => NamedColors["deeppink"];
        public static Color DeepSkyBlue => NamedColors["deepskyblue"];
        public static Color DimGray => NamedColors["dimgray"];
        public static Color DodgerBlue => NamedColors["dodgerblue"];
        public static Color Firebrick => NamedColors["firebrick"];
        public static Color FloralWhite => NamedColors["floralwhite"];
        public static Color ForestGreen => NamedColors["forestgreen"];
        public static Color Fuchsia => NamedColors["fuchsia"];
        public static Color Gainsboro => NamedColors["gainsboro"];
        public static Color GhostWhite => NamedColors["ghostwhite"];
        public static Color Gold => NamedColors["gold"];
        public static Color Goldenrod => NamedColors["goldenrod"];
        public static Color Gray => NamedColors["gray"];
        public static Color Green => NamedColors["green"];
        public static Color GreenYellow => NamedColors["greenyellow"];
        public static Color Honeydew => NamedColors["honeydew"];
        public static Color HotPink => NamedColors["hotpink"];
        public static Color IndianRed => NamedColors["indianred"];
        public static Color Indigo => NamedColors["indigo"];
        public static Color Ivory => NamedColors["ivory"];
        public static Color Khaki => NamedColors["khaki"];
        public static Color Lavender => NamedColors["lavender"];
        public static Color LavenderBlush => NamedColors["lavenderblush"];
        public static Color LawnGreen => NamedColors["lawngreen"];
        public static Color LemonChiffon => NamedColors["lemonchiffon"];
        public static Color LightBlue => NamedColors["lightblue"];
        public static Color LightCoral => NamedColors["lightcoral"];
        public static Color LightCyan => NamedColors["lightcyan"];
        public static Color LightGoldenrod => NamedColors["lightgoldenrod"];
        public static Color LightGray => NamedColors["lightgray"];
        public static Color LightGreen => NamedColors["lightgreen"];
        public static Color LightPink => NamedColors["lightpink"];
        public static Color LightSalmon => NamedColors["lightsalmon"];
        public static Color LightSeaGreen => NamedColors["lightseagreen"];
        public static Color LightSkyBlue => NamedColors["lightskyblue"];
        public static Color LightSlateGray => NamedColors["lightslategray"];
        public static Color LightSteelBlue => NamedColors["lightsteelblue"];
        public static Color LightYellow => NamedColors["lightyellow"];
        public static Color Lime => NamedColors["lime"];
        public static Color Limegreen => NamedColors["limegreen"];
        public static Color Linen => NamedColors["linen"];
        public static Color Magenta => NamedColors["magenta"];
        public static Color Maroon => NamedColors["maroon"];
        public static Color MediumAquamarine => NamedColors["mediumaquamarine"];
        public static Color MediumBlue => NamedColors["mediumblue"];
        public static Color MediumOrchid => NamedColors["mediumorchid"];
        public static Color MediumPurple => NamedColors["mediumpurple"];
        public static Color MediumSeaGreen => NamedColors["mediumseagreen"];
        public static Color MediumSlateBlue => NamedColors["mediumslateblue"];
        public static Color MediumSpringGreen => NamedColors["mediumspringgreen"];
        public static Color MediumTurquoise => NamedColors["mediumturquoise"];
        public static Color MediumVioletRed => NamedColors["mediumvioletred"];
        public static Color MidnightBlue => NamedColors["midnightblue"];
        public static Color MintCream => NamedColors["mintcream"];
        public static Color MistyRose => NamedColors["mistyrose"];
        public static Color Moccasin => NamedColors["moccasin"];
        public static Color NavajoWhite => NamedColors["navajowhite"];
        public static Color NavyBlue => NamedColors["navyblue"];
        public static Color OldLace => NamedColors["oldlace"];
        public static Color Olive => NamedColors["olive"];
        public static Color OliveDrab => NamedColors["olivedrab"];
        public static Color Orange => NamedColors["orange"];
        public static Color OrangeRed => NamedColors["orangered"];
        public static Color Orchid => NamedColors["orchid"];
        public static Color PaleGoldenrod => NamedColors["palegoldenrod"];
        public static Color PaleGreen => NamedColors["palegreen"];
        public static Color PaleTurquoise => NamedColors["paleturquoise"];
        public static Color PaleVioletRed => NamedColors["palevioletred"];
        public static Color PapayaWhip => NamedColors["papayawhip"];
        public static Color PeachPuff => NamedColors["peachpuff"];
        public static Color Peru => NamedColors["peru"];
        public static Color Pink => NamedColors["pink"];
        public static Color Plum => NamedColors["plum"];
        public static Color PowderBlue => NamedColors["powderblue"];
        public static Color Purple => NamedColors["purple"];
        public static Color RebeccaPurple => NamedColors["rebeccapurple"];
        public static Color Red => NamedColors["red"];
        public static Color RosyBrown => NamedColors["rosybrown"];
        public static Color RoyalBlue => NamedColors["royalblue"];
        public static Color SaddleBrown => NamedColors["saddlebrown"];
        public static Color Salmon => NamedColors["salmon"];
        public static Color SandyBrown => NamedColors["sandybrown"];
        public static Color SeaGreen => NamedColors["seagreen"];
        public static Color SeaShell => NamedColors["seashell"];
        public static Color Sienna => NamedColors["sienna"];
        public static Color Silver => NamedColors["silver"];
        public static Color SkyBlue => NamedColors["skyblue"];
        public static Color SlateBlue => NamedColors["slateblue"];
        public static Color SlateGray => NamedColors["slategray"];
        public static Color Snow => NamedColors["snow"];
        public static Color SpringGreen => NamedColors["springgreen"];
        public static Color SteelBlue => NamedColors["steelblue"];
        public static Color Tan => NamedColors["tan"];
        public static Color Teal => NamedColors["teal"];
        public static Color Thistle => NamedColors["thistle"];
        public static Color Tomato => NamedColors["tomato"];
        public static Color Transparent => NamedColors["transparent"];
        public static Color Turquoise => NamedColors["turquoise"];
        public static Color Violet => NamedColors["violet"];
        public static Color WebGreen => NamedColors["webgreen"];
        public static Color WebGray => NamedColors["webgray"];
        public static Color WebMaroon => NamedColors["webmaroon"];
        public static Color WebPurple => NamedColors["webpurple"];
        public static Color Wheat => NamedColors["wheat"];
        public static Color White => NamedColors["white"];
        public static Color WhiteSmoke => NamedColors["whitesmoke"];
        public static Color Yellow => NamedColors["yellow"];
        public static Color YellowGreen => NamedColors["yellowgreen"];
    }
}
