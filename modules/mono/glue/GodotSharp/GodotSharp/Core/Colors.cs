using System.Collections.Generic;

namespace Godot
{
    /// <summary>
    /// This class contains color constants created from standardized color names.
    /// The standardized color set is based on the X11 and .NET color names.
    /// </summary>
    public static class Colors
    {
        // Color names and values are derived from core/math/color_names.inc
        internal static readonly Dictionary<string, Color> namedColors = new Dictionary<string, Color> {
            {"ALICEBLUE", new Color(0.94f, 0.97f, 1.00f)},
            {"ANTIQUEWHITE", new Color(0.98f, 0.92f, 0.84f)},
            {"AQUA", new Color(0.00f, 1.00f, 1.00f)},
            {"AQUAMARINE", new Color(0.50f, 1.00f, 0.83f)},
            {"AZURE", new Color(0.94f, 1.00f, 1.00f)},
            {"BEIGE", new Color(0.96f, 0.96f, 0.86f)},
            {"BISQUE", new Color(1.00f, 0.89f, 0.77f)},
            {"BLACK", new Color(0.00f, 0.00f, 0.00f)},
            {"BLANCHEDALMOND", new Color(1.00f, 0.92f, 0.80f)},
            {"BLUE", new Color(0.00f, 0.00f, 1.00f)},
            {"BLUEVIOLET", new Color(0.54f, 0.17f, 0.89f)},
            {"BROWN", new Color(0.65f, 0.16f, 0.16f)},
            {"BURLYWOOD", new Color(0.87f, 0.72f, 0.53f)},
            {"CADETBLUE", new Color(0.37f, 0.62f, 0.63f)},
            {"CHARTREUSE", new Color(0.50f, 1.00f, 0.00f)},
            {"CHOCOLATE", new Color(0.82f, 0.41f, 0.12f)},
            {"CORAL", new Color(1.00f, 0.50f, 0.31f)},
            {"CORNFLOWERBLUE", new Color(0.39f, 0.58f, 0.93f)},
            {"CORNSILK", new Color(1.00f, 0.97f, 0.86f)},
            {"CRIMSON", new Color(0.86f, 0.08f, 0.24f)},
            {"CYAN", new Color(0.00f, 1.00f, 1.00f)},
            {"DARKBLUE", new Color(0.00f, 0.00f, 0.55f)},
            {"DARKCYAN", new Color(0.00f, 0.55f, 0.55f)},
            {"DARKGOLDENROD", new Color(0.72f, 0.53f, 0.04f)},
            {"DARKGRAY", new Color(0.66f, 0.66f, 0.66f)},
            {"DARKGREEN", new Color(0.00f, 0.39f, 0.00f)},
            {"DARKKHAKI", new Color(0.74f, 0.72f, 0.42f)},
            {"DARKMAGENTA", new Color(0.55f, 0.00f, 0.55f)},
            {"DARKOLIVEGREEN", new Color(0.33f, 0.42f, 0.18f)},
            {"DARKORANGE", new Color(1.00f, 0.55f, 0.00f)},
            {"DARKORCHID", new Color(0.60f, 0.20f, 0.80f)},
            {"DARKRED", new Color(0.55f, 0.00f, 0.00f)},
            {"DARKSALMON", new Color(0.91f, 0.59f, 0.48f)},
            {"DARKSEAGREEN", new Color(0.56f, 0.74f, 0.56f)},
            {"DARKSLATEBLUE", new Color(0.28f, 0.24f, 0.55f)},
            {"DARKSLATEGRAY", new Color(0.18f, 0.31f, 0.31f)},
            {"DARKTURQUOISE", new Color(0.00f, 0.81f, 0.82f)},
            {"DARKVIOLET", new Color(0.58f, 0.00f, 0.83f)},
            {"DEEPPINK", new Color(1.00f, 0.08f, 0.58f)},
            {"DEEPSKYBLUE", new Color(0.00f, 0.75f, 1.00f)},
            {"DIMGRAY", new Color(0.41f, 0.41f, 0.41f)},
            {"DODGERBLUE", new Color(0.12f, 0.56f, 1.00f)},
            {"FIREBRICK", new Color(0.70f, 0.13f, 0.13f)},
            {"FLORALWHITE", new Color(1.00f, 0.98f, 0.94f)},
            {"FORESTGREEN", new Color(0.13f, 0.55f, 0.13f)},
            {"FUCHSIA", new Color(1.00f, 0.00f, 1.00f)},
            {"GAINSBORO", new Color(0.86f, 0.86f, 0.86f)},
            {"GHOSTWHITE", new Color(0.97f, 0.97f, 1.00f)},
            {"GOLD", new Color(1.00f, 0.84f, 0.00f)},
            {"GOLDENROD", new Color(0.85f, 0.65f, 0.13f)},
            {"GRAY", new Color(0.75f, 0.75f, 0.75f)},
            {"GREEN", new Color(0.00f, 1.00f, 0.00f)},
            {"GREENYELLOW", new Color(0.68f, 1.00f, 0.18f)},
            {"HONEYDEW", new Color(0.94f, 1.00f, 0.94f)},
            {"HOTPINK", new Color(1.00f, 0.41f, 0.71f)},
            {"INDIANRED", new Color(0.80f, 0.36f, 0.36f)},
            {"INDIGO", new Color(0.29f, 0.00f, 0.51f)},
            {"IVORY", new Color(1.00f, 1.00f, 0.94f)},
            {"KHAKI", new Color(0.94f, 0.90f, 0.55f)},
            {"LAVENDER", new Color(0.90f, 0.90f, 0.98f)},
            {"LAVENDERBLUSH", new Color(1.00f, 0.94f, 0.96f)},
            {"LAWNGREEN", new Color(0.49f, 0.99f, 0.00f)},
            {"LEMONCHIFFON", new Color(1.00f, 0.98f, 0.80f)},
            {"LIGHTBLUE", new Color(0.68f, 0.85f, 0.90f)},
            {"LIGHTCORAL", new Color(0.94f, 0.50f, 0.50f)},
            {"LIGHTCYAN", new Color(0.88f, 1.00f, 1.00f)},
            {"LIGHTGOLDENROD", new Color(0.98f, 0.98f, 0.82f)},
            {"LIGHTGRAY", new Color(0.83f, 0.83f, 0.83f)},
            {"LIGHTGREEN", new Color(0.56f, 0.93f, 0.56f)},
            {"LIGHTPINK", new Color(1.00f, 0.71f, 0.76f)},
            {"LIGHTSALMON", new Color(1.00f, 0.63f, 0.48f)},
            {"LIGHTSEAGREEN", new Color(0.13f, 0.70f, 0.67f)},
            {"LIGHTSKYBLUE", new Color(0.53f, 0.81f, 0.98f)},
            {"LIGHTSLATEGRAY", new Color(0.47f, 0.53f, 0.60f)},
            {"LIGHTSTEELBLUE", new Color(0.69f, 0.77f, 0.87f)},
            {"LIGHTYELLOW", new Color(1.00f, 1.00f, 0.88f)},
            {"LIME", new Color(0.00f, 1.00f, 0.00f)},
            {"LIMEGREEN", new Color(0.20f, 0.80f, 0.20f)},
            {"LINEN", new Color(0.98f, 0.94f, 0.90f)},
            {"MAGENTA", new Color(1.00f, 0.00f, 1.00f)},
            {"MAROON", new Color(0.69f, 0.19f, 0.38f)},
            {"MEDIUMAQUAMARINE", new Color(0.40f, 0.80f, 0.67f)},
            {"MEDIUMBLUE", new Color(0.00f, 0.00f, 0.80f)},
            {"MEDIUMORCHID", new Color(0.73f, 0.33f, 0.83f)},
            {"MEDIUMPURPLE", new Color(0.58f, 0.44f, 0.86f)},
            {"MEDIUMSEAGREEN", new Color(0.24f, 0.70f, 0.44f)},
            {"MEDIUMSLATEBLUE", new Color(0.48f, 0.41f, 0.93f)},
            {"MEDIUMSPRINGGREEN", new Color(0.00f, 0.98f, 0.60f)},
            {"MEDIUMTURQUOISE", new Color(0.28f, 0.82f, 0.80f)},
            {"MEDIUMVIOLETRED", new Color(0.78f, 0.08f, 0.52f)},
            {"MIDNIGHTBLUE", new Color(0.10f, 0.10f, 0.44f)},
            {"MINTCREAM", new Color(0.96f, 1.00f, 0.98f)},
            {"MISTYROSE", new Color(1.00f, 0.89f, 0.88f)},
            {"MOCCASIN", new Color(1.00f, 0.89f, 0.71f)},
            {"NAVAJOWHITE", new Color(1.00f, 0.87f, 0.68f)},
            {"NAVYBLUE", new Color(0.00f, 0.00f, 0.50f)},
            {"OLDLACE", new Color(0.99f, 0.96f, 0.90f)},
            {"OLIVE", new Color(0.50f, 0.50f, 0.00f)},
            {"OLIVEDRAB", new Color(0.42f, 0.56f, 0.14f)},
            {"ORANGE", new Color(1.00f, 0.65f, 0.00f)},
            {"ORANGERED", new Color(1.00f, 0.27f, 0.00f)},
            {"ORCHID", new Color(0.85f, 0.44f, 0.84f)},
            {"PALEGOLDENROD", new Color(0.93f, 0.91f, 0.67f)},
            {"PALEGREEN", new Color(0.60f, 0.98f, 0.60f)},
            {"PALETURQUOISE", new Color(0.69f, 0.93f, 0.93f)},
            {"PALEVIOLETRED", new Color(0.86f, 0.44f, 0.58f)},
            {"PAPAYAWHIP", new Color(1.00f, 0.94f, 0.84f)},
            {"PEACHPUFF", new Color(1.00f, 0.85f, 0.73f)},
            {"PERU", new Color(0.80f, 0.52f, 0.25f)},
            {"PINK", new Color(1.00f, 0.75f, 0.80f)},
            {"PLUM", new Color(0.87f, 0.63f, 0.87f)},
            {"POWDERBLUE", new Color(0.69f, 0.88f, 0.90f)},
            {"PURPLE", new Color(0.63f, 0.13f, 0.94f)},
            {"REBECCAPURPLE", new Color(0.40f, 0.20f, 0.60f)},
            {"RED", new Color(1.00f, 0.00f, 0.00f)},
            {"ROSYBROWN", new Color(0.74f, 0.56f, 0.56f)},
            {"ROYALBLUE", new Color(0.25f, 0.41f, 0.88f)},
            {"SADDLEBROWN", new Color(0.55f, 0.27f, 0.07f)},
            {"SALMON", new Color(0.98f, 0.50f, 0.45f)},
            {"SANDYBROWN", new Color(0.96f, 0.64f, 0.38f)},
            {"SEAGREEN", new Color(0.18f, 0.55f, 0.34f)},
            {"SEASHELL", new Color(1.00f, 0.96f, 0.93f)},
            {"SIENNA", new Color(0.63f, 0.32f, 0.18f)},
            {"SILVER", new Color(0.75f, 0.75f, 0.75f)},
            {"SKYBLUE", new Color(0.53f, 0.81f, 0.92f)},
            {"SLATEBLUE", new Color(0.42f, 0.35f, 0.80f)},
            {"SLATEGRAY", new Color(0.44f, 0.50f, 0.56f)},
            {"SNOW", new Color(1.00f, 0.98f, 0.98f)},
            {"SPRINGGREEN", new Color(0.00f, 1.00f, 0.50f)},
            {"STEELBLUE", new Color(0.27f, 0.51f, 0.71f)},
            {"TAN", new Color(0.82f, 0.71f, 0.55f)},
            {"TEAL", new Color(0.00f, 0.50f, 0.50f)},
            {"THISTLE", new Color(0.85f, 0.75f, 0.85f)},
            {"TOMATO", new Color(1.00f, 0.39f, 0.28f)},
            {"TRANSPARENT", new Color(1.00f, 1.00f, 1.00f, 0.00f)},
            {"TURQUOISE", new Color(0.25f, 0.88f, 0.82f)},
            {"VIOLET", new Color(0.93f, 0.51f, 0.93f)},
            {"WEBGRAY", new Color(0.50f, 0.50f, 0.50f)},
            {"WEBGREEN", new Color(0.00f, 0.50f, 0.00f)},
            {"WEBMAROON", new Color(0.50f, 0.00f, 0.00f)},
            {"WEBPURPLE", new Color(0.50f, 0.00f, 0.50f)},
            {"WHEAT", new Color(0.96f, 0.87f, 0.70f)},
            {"WHITE", new Color(1.00f, 1.00f, 1.00f)},
            {"WHITESMOKE", new Color(0.96f, 0.96f, 0.96f)},
            {"YELLOW", new Color(1.00f, 1.00f, 0.00f)},
            {"YELLOWGREEN", new Color(0.60f, 0.80f, 0.20f)},
        };

#pragma warning disable CS1591 // Disable warning: "Missing XML comment for publicly visible type or member"
        public static Color AliceBlue { get { return namedColors["ALICEBLUE"]; } }
        public static Color AntiqueWhite { get { return namedColors["ANTIQUEWHITE"]; } }
        public static Color Aqua { get { return namedColors["AQUA"]; } }
        public static Color Aquamarine { get { return namedColors["AQUAMARINE"]; } }
        public static Color Azure { get { return namedColors["AZURE"]; } }
        public static Color Beige { get { return namedColors["BEIGE"]; } }
        public static Color Bisque { get { return namedColors["BISQUE"]; } }
        public static Color Black { get { return namedColors["BLACK"]; } }
        public static Color BlanchedAlmond { get { return namedColors["BLANCHEDALMOND"]; } }
        public static Color Blue { get { return namedColors["BLUE"]; } }
        public static Color BlueViolet { get { return namedColors["BLUEVIOLET"]; } }
        public static Color Brown { get { return namedColors["BROWN"]; } }
        public static Color Burlywood { get { return namedColors["BURLYWOOD"]; } }
        public static Color CadetBlue { get { return namedColors["CADETBLUE"]; } }
        public static Color Chartreuse { get { return namedColors["CHARTREUSE"]; } }
        public static Color Chocolate { get { return namedColors["CHOCOLATE"]; } }
        public static Color Coral { get { return namedColors["CORAL"]; } }
        public static Color CornflowerBlue { get { return namedColors["CORNFLOWERBLUE"]; } }
        public static Color Cornsilk { get { return namedColors["CORNSILK"]; } }
        public static Color Crimson { get { return namedColors["CRIMSON"]; } }
        public static Color Cyan { get { return namedColors["CYAN"]; } }
        public static Color DarkBlue { get { return namedColors["DARKBLUE"]; } }
        public static Color DarkCyan { get { return namedColors["DARKCYAN"]; } }
        public static Color DarkGoldenrod { get { return namedColors["DARKGOLDENROD"]; } }
        public static Color DarkGray { get { return namedColors["DARKGRAY"]; } }
        public static Color DarkGreen { get { return namedColors["DARKGREEN"]; } }
        public static Color DarkKhaki { get { return namedColors["DARKKHAKI"]; } }
        public static Color DarkMagenta { get { return namedColors["DARKMAGENTA"]; } }
        public static Color DarkOliveGreen { get { return namedColors["DARKOLIVEGREEN"]; } }
        public static Color DarkOrange { get { return namedColors["DARKORANGE"]; } }
        public static Color DarkOrchid { get { return namedColors["DARKORCHID"]; } }
        public static Color DarkRed { get { return namedColors["DARKRED"]; } }
        public static Color DarkSalmon { get { return namedColors["DARKSALMON"]; } }
        public static Color DarkSeaGreen { get { return namedColors["DARKSEAGREEN"]; } }
        public static Color DarkSlateBlue { get { return namedColors["DARKSLATEBLUE"]; } }
        public static Color DarkSlateGray { get { return namedColors["DARKSLATEGRAY"]; } }
        public static Color DarkTurquoise { get { return namedColors["DARKTURQUOISE"]; } }
        public static Color DarkViolet { get { return namedColors["DARKVIOLET"]; } }
        public static Color DeepPink { get { return namedColors["DEEPPINK"]; } }
        public static Color DeepSkyBlue { get { return namedColors["DEEPSKYBLUE"]; } }
        public static Color DimGray { get { return namedColors["DIMGRAY"]; } }
        public static Color DodgerBlue { get { return namedColors["DODGERBLUE"]; } }
        public static Color Firebrick { get { return namedColors["FIREBRICK"]; } }
        public static Color FloralWhite { get { return namedColors["FLORALWHITE"]; } }
        public static Color ForestGreen { get { return namedColors["FORESTGREEN"]; } }
        public static Color Fuchsia { get { return namedColors["FUCHSIA"]; } }
        public static Color Gainsboro { get { return namedColors["GAINSBORO"]; } }
        public static Color GhostWhite { get { return namedColors["GHOSTWHITE"]; } }
        public static Color Gold { get { return namedColors["GOLD"]; } }
        public static Color Goldenrod { get { return namedColors["GOLDENROD"]; } }
        public static Color Gray { get { return namedColors["GRAY"]; } }
        public static Color Green { get { return namedColors["GREEN"]; } }
        public static Color GreenYellow { get { return namedColors["GREENYELLOW"]; } }
        public static Color Honeydew { get { return namedColors["HONEYDEW"]; } }
        public static Color HotPink { get { return namedColors["HOTPINK"]; } }
        public static Color IndianRed { get { return namedColors["INDIANRED"]; } }
        public static Color Indigo { get { return namedColors["INDIGO"]; } }
        public static Color Ivory { get { return namedColors["IVORY"]; } }
        public static Color Khaki { get { return namedColors["KHAKI"]; } }
        public static Color Lavender { get { return namedColors["LAVENDER"]; } }
        public static Color LavenderBlush { get { return namedColors["LAVENDERBLUSH"]; } }
        public static Color LawnGreen { get { return namedColors["LAWNGREEN"]; } }
        public static Color LemonChiffon { get { return namedColors["LEMONCHIFFON"]; } }
        public static Color LightBlue { get { return namedColors["LIGHTBLUE"]; } }
        public static Color LightCoral { get { return namedColors["LIGHTCORAL"]; } }
        public static Color LightCyan { get { return namedColors["LIGHTCYAN"]; } }
        public static Color LightGoldenrod { get { return namedColors["LIGHTGOLDENROD"]; } }
        public static Color LightGray { get { return namedColors["LIGHTGRAY"]; } }
        public static Color LightGreen { get { return namedColors["LIGHTGREEN"]; } }
        public static Color LightPink { get { return namedColors["LIGHTPINK"]; } }
        public static Color LightSalmon { get { return namedColors["LIGHTSALMON"]; } }
        public static Color LightSeaGreen { get { return namedColors["LIGHTSEAGREEN"]; } }
        public static Color LightSkyBlue { get { return namedColors["LIGHTSKYBLUE"]; } }
        public static Color LightSlateGray { get { return namedColors["LIGHTSLATEGRAY"]; } }
        public static Color LightSteelBlue { get { return namedColors["LIGHTSTEELBLUE"]; } }
        public static Color LightYellow { get { return namedColors["LIGHTYELLOW"]; } }
        public static Color Lime { get { return namedColors["LIME"]; } }
        public static Color LimeGreen { get { return namedColors["LIMEGREEN"]; } }
        public static Color Linen { get { return namedColors["LINEN"]; } }
        public static Color Magenta { get { return namedColors["MAGENTA"]; } }
        public static Color Maroon { get { return namedColors["MAROON"]; } }
        public static Color MediumAquamarine { get { return namedColors["MEDIUMAQUAMARINE"]; } }
        public static Color MediumBlue { get { return namedColors["MEDIUMBLUE"]; } }
        public static Color MediumOrchid { get { return namedColors["MEDIUMORCHID"]; } }
        public static Color MediumPurple { get { return namedColors["MEDIUMPURPLE"]; } }
        public static Color MediumSeaGreen { get { return namedColors["MEDIUMSEAGREEN"]; } }
        public static Color MediumSlateBlue { get { return namedColors["MEDIUMSLATEBLUE"]; } }
        public static Color MediumSpringGreen { get { return namedColors["MEDIUMSPRINGGREEN"]; } }
        public static Color MediumTurquoise { get { return namedColors["MEDIUMTURQUOISE"]; } }
        public static Color MediumVioletRed { get { return namedColors["MEDIUMVIOLETRED"]; } }
        public static Color MidnightBlue { get { return namedColors["MIDNIGHTBLUE"]; } }
        public static Color MintCream { get { return namedColors["MINTCREAM"]; } }
        public static Color MistyRose { get { return namedColors["MISTYROSE"]; } }
        public static Color Moccasin { get { return namedColors["MOCCASIN"]; } }
        public static Color NavajoWhite { get { return namedColors["NAVAJOWHITE"]; } }
        public static Color NavyBlue { get { return namedColors["NAVYBLUE"]; } }
        public static Color OldLace { get { return namedColors["OLDLACE"]; } }
        public static Color Olive { get { return namedColors["OLIVE"]; } }
        public static Color OliveDrab { get { return namedColors["OLIVEDRAB"]; } }
        public static Color Orange { get { return namedColors["ORANGE"]; } }
        public static Color OrangeRed { get { return namedColors["ORANGERED"]; } }
        public static Color Orchid { get { return namedColors["ORCHID"]; } }
        public static Color PaleGoldenrod { get { return namedColors["PALEGOLDENROD"]; } }
        public static Color PaleGreen { get { return namedColors["PALEGREEN"]; } }
        public static Color PaleTurquoise { get { return namedColors["PALETURQUOISE"]; } }
        public static Color PaleVioletRed { get { return namedColors["PALEVIOLETRED"]; } }
        public static Color PapayaWhip { get { return namedColors["PAPAYAWHIP"]; } }
        public static Color PeachPuff { get { return namedColors["PEACHPUFF"]; } }
        public static Color Peru { get { return namedColors["PERU"]; } }
        public static Color Pink { get { return namedColors["PINK"]; } }
        public static Color Plum { get { return namedColors["PLUM"]; } }
        public static Color PowderBlue { get { return namedColors["POWDERBLUE"]; } }
        public static Color Purple { get { return namedColors["PURPLE"]; } }
        public static Color RebeccaPurple { get { return namedColors["REBECCAPURPLE"]; } }
        public static Color Red { get { return namedColors["RED"]; } }
        public static Color RosyBrown { get { return namedColors["ROSYBROWN"]; } }
        public static Color RoyalBlue { get { return namedColors["ROYALBLUE"]; } }
        public static Color SaddleBrown { get { return namedColors["SADDLEBROWN"]; } }
        public static Color Salmon { get { return namedColors["SALMON"]; } }
        public static Color SandyBrown { get { return namedColors["SANDYBROWN"]; } }
        public static Color SeaGreen { get { return namedColors["SEAGREEN"]; } }
        public static Color Seashell { get { return namedColors["SEASHELL"]; } }
        public static Color Sienna { get { return namedColors["SIENNA"]; } }
        public static Color Silver { get { return namedColors["SILVER"]; } }
        public static Color SkyBlue { get { return namedColors["SKYBLUE"]; } }
        public static Color SlateBlue { get { return namedColors["SLATEBLUE"]; } }
        public static Color SlateGray { get { return namedColors["SLATEGRAY"]; } }
        public static Color Snow { get { return namedColors["SNOW"]; } }
        public static Color SpringGreen { get { return namedColors["SPRINGGREEN"]; } }
        public static Color SteelBlue { get { return namedColors["STEELBLUE"]; } }
        public static Color Tan { get { return namedColors["TAN"]; } }
        public static Color Teal { get { return namedColors["TEAL"]; } }
        public static Color Thistle { get { return namedColors["THISTLE"]; } }
        public static Color Tomato { get { return namedColors["TOMATO"]; } }
        public static Color Transparent { get { return namedColors["TRANSPARENT"]; } }
        public static Color Turquoise { get { return namedColors["TURQUOISE"]; } }
        public static Color Violet { get { return namedColors["VIOLET"]; } }
        public static Color WebGray { get { return namedColors["WEBGRAY"]; } }
        public static Color WebGreen { get { return namedColors["WEBGREEN"]; } }
        public static Color WebMaroon { get { return namedColors["WEBMAROON"]; } }
        public static Color WebPurple { get { return namedColors["WEBPURPLE"]; } }
        public static Color Wheat { get { return namedColors["WHEAT"]; } }
        public static Color White { get { return namedColors["WHITE"]; } }
        public static Color WhiteSmoke { get { return namedColors["WHITESMOKE"]; } }
        public static Color Yellow { get { return namedColors["YELLOW"]; } }
        public static Color YellowGreen { get { return namedColors["YELLOWGREEN"]; } }
#pragma warning restore CS1591
    }
}
