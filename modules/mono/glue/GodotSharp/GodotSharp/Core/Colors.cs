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
            { "ALICEBLUE", new Color(0xF0F8FFFF) },
            { "ANTIQUEWHITE", new Color(0xFAEBD7FF) },
            { "AQUA", new Color(0x00FFFFFF) },
            { "AQUAMARINE", new Color(0x7FFFD4FF) },
            { "AZURE", new Color(0xF0FFFFFF) },
            { "BEIGE", new Color(0xF5F5DCFF) },
            { "BISQUE", new Color(0xFFE4C4FF) },
            { "BLACK", new Color(0x000000FF) },
            { "BLANCHEDALMOND", new Color(0xFFEBCDFF) },
            { "BLUE", new Color(0x0000FFFF) },
            { "BLUEVIOLET", new Color(0x8A2BE2FF) },
            { "BROWN", new Color(0xA52A2AFF) },
            { "BURLYWOOD", new Color(0xDEB887FF) },
            { "CADETBLUE", new Color(0x5F9EA0FF) },
            { "CHARTREUSE", new Color(0x7FFF00FF) },
            { "CHOCOLATE", new Color(0xD2691EFF) },
            { "CORAL", new Color(0xFF7F50FF) },
            { "CORNFLOWERBLUE", new Color(0x6495EDFF) },
            { "CORNSILK", new Color(0xFFF8DCFF) },
            { "CRIMSON", new Color(0xDC143CFF) },
            { "CYAN", new Color(0x00FFFFFF) },
            { "DARKBLUE", new Color(0x00008BFF) },
            { "DARKCYAN", new Color(0x008B8BFF) },
            { "DARKGOLDENROD", new Color(0xB8860BFF) },
            { "DARKGRAY", new Color(0xA9A9A9FF) },
            { "DARKGREEN", new Color(0x006400FF) },
            { "DARKKHAKI", new Color(0xBDB76BFF) },
            { "DARKMAGENTA", new Color(0x8B008BFF) },
            { "DARKOLIVEGREEN", new Color(0x556B2FFF) },
            { "DARKORANGE", new Color(0xFF8C00FF) },
            { "DARKORCHID", new Color(0x9932CCFF) },
            { "DARKRED", new Color(0x8B0000FF) },
            { "DARKSALMON", new Color(0xE9967AFF) },
            { "DARKSEAGREEN", new Color(0x8FBC8FFF) },
            { "DARKSLATEBLUE", new Color(0x483D8BFF) },
            { "DARKSLATEGRAY", new Color(0x2F4F4FFF) },
            { "DARKTURQUOISE", new Color(0x00CED1FF) },
            { "DARKVIOLET", new Color(0x9400D3FF) },
            { "DEEPPINK", new Color(0xFF1493FF) },
            { "DEEPSKYBLUE", new Color(0x00BFFFFF) },
            { "DIMGRAY", new Color(0x696969FF) },
            { "DODGERBLUE", new Color(0x1E90FFFF) },
            { "FIREBRICK", new Color(0xB22222FF) },
            { "FLORALWHITE", new Color(0xFFFAF0FF) },
            { "FORESTGREEN", new Color(0x228B22FF) },
            { "FUCHSIA", new Color(0xFF00FFFF) },
            { "GAINSBORO", new Color(0xDCDCDCFF) },
            { "GHOSTWHITE", new Color(0xF8F8FFFF) },
            { "GOLD", new Color(0xFFD700FF) },
            { "GOLDENROD", new Color(0xDAA520FF) },
            { "GRAY", new Color(0xBEBEBEFF) },
            { "GREEN", new Color(0x00FF00FF) },
            { "GREENYELLOW", new Color(0xADFF2FFF) },
            { "HONEYDEW", new Color(0xF0FFF0FF) },
            { "HOTPINK", new Color(0xFF69B4FF) },
            { "INDIANRED", new Color(0xCD5C5CFF) },
            { "INDIGO", new Color(0x4B0082FF) },
            { "IVORY", new Color(0xFFFFF0FF) },
            { "KHAKI", new Color(0xF0E68CFF) },
            { "LAVENDER", new Color(0xE6E6FAFF) },
            { "LAVENDERBLUSH", new Color(0xFFF0F5FF) },
            { "LAWNGREEN", new Color(0x7CFC00FF) },
            { "LEMONCHIFFON", new Color(0xFFFACDFF) },
            { "LIGHTBLUE", new Color(0xADD8E6FF) },
            { "LIGHTCORAL", new Color(0xF08080FF) },
            { "LIGHTCYAN", new Color(0xE0FFFFFF) },
            { "LIGHTGOLDENROD", new Color(0xFAFAD2FF) },
            { "LIGHTGRAY", new Color(0xD3D3D3FF) },
            { "LIGHTGREEN", new Color(0x90EE90FF) },
            { "LIGHTPINK", new Color(0xFFB6C1FF) },
            { "LIGHTSALMON", new Color(0xFFA07AFF) },
            { "LIGHTSEAGREEN", new Color(0x20B2AAFF) },
            { "LIGHTSKYBLUE", new Color(0x87CEFAFF) },
            { "LIGHTSLATEGRAY", new Color(0x778899FF) },
            { "LIGHTSTEELBLUE", new Color(0xB0C4DEFF) },
            { "LIGHTYELLOW", new Color(0xFFFFE0FF) },
            { "LIME", new Color(0x00FF00FF) },
            { "LIMEGREEN", new Color(0x32CD32FF) },
            { "LINEN", new Color(0xFAF0E6FF) },
            { "MAGENTA", new Color(0xFF00FFFF) },
            { "MAROON", new Color(0xB03060FF) },
            { "MEDIUMAQUAMARINE", new Color(0x66CDAAFF) },
            { "MEDIUMBLUE", new Color(0x0000CDFF) },
            { "MEDIUMORCHID", new Color(0xBA55D3FF) },
            { "MEDIUMPURPLE", new Color(0x9370DBFF) },
            { "MEDIUMSEAGREEN", new Color(0x3CB371FF) },
            { "MEDIUMSLATEBLUE", new Color(0x7B68EEFF) },
            { "MEDIUMSPRINGGREEN", new Color(0x00FA9AFF) },
            { "MEDIUMTURQUOISE", new Color(0x48D1CCFF) },
            { "MEDIUMVIOLETRED", new Color(0xC71585FF) },
            { "MIDNIGHTBLUE", new Color(0x191970FF) },
            { "MINTCREAM", new Color(0xF5FFFAFF) },
            { "MISTYROSE", new Color(0xFFE4E1FF) },
            { "MOCCASIN", new Color(0xFFE4B5FF) },
            { "NAVAJOWHITE", new Color(0xFFDEADFF) },
            { "NAVYBLUE", new Color(0x000080FF) },
            { "OLDLACE", new Color(0xFDF5E6FF) },
            { "OLIVE", new Color(0x808000FF) },
            { "OLIVEDRAB", new Color(0x6B8E23FF) },
            { "ORANGE", new Color(0xFFA500FF) },
            { "ORANGERED", new Color(0xFF4500FF) },
            { "ORCHID", new Color(0xDA70D6FF) },
            { "PALEGOLDENROD", new Color(0xEEE8AAFF) },
            { "PALEGREEN", new Color(0x98FB98FF) },
            { "PALETURQUOISE", new Color(0xAFEEEEFF) },
            { "PALEVIOLETRED", new Color(0xDB7093FF) },
            { "PAPAYAWHIP", new Color(0xFFEFD5FF) },
            { "PEACHPUFF", new Color(0xFFDAB9FF) },
            { "PERU", new Color(0xCD853FFF) },
            { "PINK", new Color(0xFFC0CBFF) },
            { "PLUM", new Color(0xDDA0DDFF) },
            { "POWDERBLUE", new Color(0xB0E0E6FF) },
            { "PURPLE", new Color(0xA020F0FF) },
            { "REBECCAPURPLE", new Color(0x663399FF) },
            { "RED", new Color(0xFF0000FF) },
            { "ROSYBROWN", new Color(0xBC8F8FFF) },
            { "ROYALBLUE", new Color(0x4169E1FF) },
            { "SADDLEBROWN", new Color(0x8B4513FF) },
            { "SALMON", new Color(0xFA8072FF) },
            { "SANDYBROWN", new Color(0xF4A460FF) },
            { "SEAGREEN", new Color(0x2E8B57FF) },
            { "SEASHELL", new Color(0xFFF5EEFF) },
            { "SIENNA", new Color(0xA0522DFF) },
            { "SILVER", new Color(0xC0C0C0FF) },
            { "SKYBLUE", new Color(0x87CEEBFF) },
            { "SLATEBLUE", new Color(0x6A5ACDFF) },
            { "SLATEGRAY", new Color(0x708090FF) },
            { "SNOW", new Color(0xFFFAFAFF) },
            { "SPRINGGREEN", new Color(0x00FF7FFF) },
            { "STEELBLUE", new Color(0x4682B4FF) },
            { "TAN", new Color(0xD2B48CFF) },
            { "TEAL", new Color(0x008080FF) },
            { "THISTLE", new Color(0xD8BFD8FF) },
            { "TOMATO", new Color(0xFF6347FF) },
            { "TRANSPARENT", new Color(0xFFFFFF00) },
            { "TURQUOISE", new Color(0x40E0D0FF) },
            { "VIOLET", new Color(0xEE82EEFF) },
            { "WEBGRAY", new Color(0x808080FF) },
            { "WEBGREEN", new Color(0x008000FF) },
            { "WEBMAROON", new Color(0x800000FF) },
            { "WEBPURPLE", new Color(0x800080FF) },
            { "WHEAT", new Color(0xF5DEB3FF) },
            { "WHITE", new Color(0xFFFFFFFF) },
            { "WHITESMOKE", new Color(0xF5F5F5FF) },
            { "YELLOW", new Color(0xFFFF00FF) },
            { "YELLOWGREEN", new Color(0x9ACD32FF) },
        };

        /// <value><c>(0.94, 0.97, 1, 1)</c></value>
        public static Color AliceBlue { get { return namedColors["ALICEBLUE"]; } }
        /// <value><c>(0.98, 0.92, 0.84, 1)</c></value>
        public static Color AntiqueWhite { get { return namedColors["ANTIQUEWHITE"]; } }
        /// <value><c>(0, 1, 1, 1)</c></value>
        public static Color Aqua { get { return namedColors["AQUA"]; } }
        /// <value><c>(0.5, 1, 0.83, 1)</c></value>
        public static Color Aquamarine { get { return namedColors["AQUAMARINE"]; } }
        /// <value><c>(0.94, 1, 1, 1)</c></value>
        public static Color Azure { get { return namedColors["AZURE"]; } }
        /// <value><c>(0.96, 0.96, 0.86, 1)</c></value>
        public static Color Beige { get { return namedColors["BEIGE"]; } }
        /// <value><c>(1, 0.89, 0.77, 1)</c></value>
        public static Color Bisque { get { return namedColors["BISQUE"]; } }
        /// <value><c>(0, 0, 0, 1)</c></value>
        public static Color Black { get { return namedColors["BLACK"]; } }
        /// <value><c>(1, 0.92, 0.8, 1)</c></value>
        public static Color BlanchedAlmond { get { return namedColors["BLANCHEDALMOND"]; } }
        /// <value><c>(0, 0, 1, 1)</c></value>
        public static Color Blue { get { return namedColors["BLUE"]; } }
        /// <value><c>(0.54, 0.17, 0.89, 1)</c></value>
        public static Color BlueViolet { get { return namedColors["BLUEVIOLET"]; } }
        /// <value><c>(0.65, 0.16, 0.16, 1)</c></value>
        public static Color Brown { get { return namedColors["BROWN"]; } }
        /// <value><c>(0.87, 0.72, 0.53, 1)</c></value>
        public static Color Burlywood { get { return namedColors["BURLYWOOD"]; } }
        /// <value><c>(0.37, 0.62, 0.63, 1)</c></value>
        public static Color CadetBlue { get { return namedColors["CADETBLUE"]; } }
        /// <value><c>(0.5, 1, 0, 1)</c></value>
        public static Color Chartreuse { get { return namedColors["CHARTREUSE"]; } }
        /// <value><c>(0.82, 0.41, 0.12, 1)</c></value>
        public static Color Chocolate { get { return namedColors["CHOCOLATE"]; } }
        /// <value><c>(1, 0.5, 0.31, 1)</c></value>
        public static Color Coral { get { return namedColors["CORAL"]; } }
        /// <value><c>(0.39, 0.58, 0.93, 1)</c></value>
        public static Color CornflowerBlue { get { return namedColors["CORNFLOWERBLUE"]; } }
        /// <value><c>(1, 0.97, 0.86, 1)</c></value>
        public static Color Cornsilk { get { return namedColors["CORNSILK"]; } }
        /// <value><c>(0.86, 0.08, 0.24, 1)</c></value>
        public static Color Crimson { get { return namedColors["CRIMSON"]; } }
        /// <value><c>(0, 1, 1, 1)</c></value>
        public static Color Cyan { get { return namedColors["CYAN"]; } }
        /// <value><c>(0, 0, 0.55, 1)</c></value>
        public static Color DarkBlue { get { return namedColors["DARKBLUE"]; } }
        /// <value><c>(0, 0.55, 0.55, 1)</c></value>
        public static Color DarkCyan { get { return namedColors["DARKCYAN"]; } }
        /// <value><c>(0.72, 0.53, 0.04, 1)</c></value>
        public static Color DarkGoldenrod { get { return namedColors["DARKGOLDENROD"]; } }
        /// <value><c>(0.66, 0.66, 0.66, 1)</c></value>
        public static Color DarkGray { get { return namedColors["DARKGRAY"]; } }
        /// <value><c>(0, 0.39, 0, 1)</c></value>
        public static Color DarkGreen { get { return namedColors["DARKGREEN"]; } }
        /// <value><c>(0.74, 0.72, 0.42, 1)</c></value>
        public static Color DarkKhaki { get { return namedColors["DARKKHAKI"]; } }
        /// <value><c>(0.55, 0, 0.55, 1)</c></value>
        public static Color DarkMagenta { get { return namedColors["DARKMAGENTA"]; } }
        /// <value><c>(0.33, 0.42, 0.18, 1)</c></value>
        public static Color DarkOliveGreen { get { return namedColors["DARKOLIVEGREEN"]; } }
        /// <value><c>(1, 0.55, 0, 1)</c></value>
        public static Color DarkOrange { get { return namedColors["DARKORANGE"]; } }
        /// <value><c>(0.6, 0.2, 0.8, 1)</c></value>
        public static Color DarkOrchid { get { return namedColors["DARKORCHID"]; } }
        /// <value><c>(0.55, 0, 0, 1)</c></value>
        public static Color DarkRed { get { return namedColors["DARKRED"]; } }
        /// <value><c>(0.91, 0.59, 0.48, 1)</c></value>
        public static Color DarkSalmon { get { return namedColors["DARKSALMON"]; } }
        /// <value><c>(0.56, 0.74, 0.56, 1)</c></value>
        public static Color DarkSeaGreen { get { return namedColors["DARKSEAGREEN"]; } }
        /// <value><c>(0.28, 0.24, 0.55, 1)</c></value>
        public static Color DarkSlateBlue { get { return namedColors["DARKSLATEBLUE"]; } }
        /// <value><c>(0.18, 0.31, 0.31, 1)</c></value>
        public static Color DarkSlateGray { get { return namedColors["DARKSLATEGRAY"]; } }
        /// <value><c>(0, 0.81, 0.82, 1)</c></value>
        public static Color DarkTurquoise { get { return namedColors["DARKTURQUOISE"]; } }
        /// <value><c>(0.58, 0, 0.83, 1)</c></value>
        public static Color DarkViolet { get { return namedColors["DARKVIOLET"]; } }
        /// <value><c>(1, 0.08, 0.58, 1)</c></value>
        public static Color DeepPink { get { return namedColors["DEEPPINK"]; } }
        /// <value><c>(0, 0.75, 1, 1)</c></value>
        public static Color DeepSkyBlue { get { return namedColors["DEEPSKYBLUE"]; } }
        /// <value><c>(0.41, 0.41, 0.41, 1)</c></value>
        public static Color DimGray { get { return namedColors["DIMGRAY"]; } }
        /// <value><c>(0.12, 0.56, 1, 1)</c></value>
        public static Color DodgerBlue { get { return namedColors["DODGERBLUE"]; } }
        /// <value><c>(0.7, 0.13, 0.13, 1)</c></value>
        public static Color Firebrick { get { return namedColors["FIREBRICK"]; } }
        /// <value><c>(1, 0.98, 0.94, 1)</c></value>
        public static Color FloralWhite { get { return namedColors["FLORALWHITE"]; } }
        /// <value><c>(0.13, 0.55, 0.13, 1)</c></value>
        public static Color ForestGreen { get { return namedColors["FORESTGREEN"]; } }
        /// <value><c>(1, 0, 1, 1)</c></value>
        public static Color Fuchsia { get { return namedColors["FUCHSIA"]; } }
        /// <value><c>(0.86, 0.86, 0.86, 1)</c></value>
        public static Color Gainsboro { get { return namedColors["GAINSBORO"]; } }
        /// <value><c>(0.97, 0.97, 1, 1)</c></value>
        public static Color GhostWhite { get { return namedColors["GHOSTWHITE"]; } }
        /// <value><c>(1, 0.84, 0, 1)</c></value>
        public static Color Gold { get { return namedColors["GOLD"]; } }
        /// <value><c>(0.85, 0.65, 0.13, 1)</c></value>
        public static Color Goldenrod { get { return namedColors["GOLDENROD"]; } }
        /// <value><c>(0.75, 0.75, 0.75, 1)</c></value>
        public static Color Gray { get { return namedColors["GRAY"]; } }
        /// <value><c>(0, 1, 0, 1)</c></value>
        public static Color Green { get { return namedColors["GREEN"]; } }
        /// <value><c>(0.68, 1, 0.18, 1)</c></value>
        public static Color GreenYellow { get { return namedColors["GREENYELLOW"]; } }
        /// <value><c>(0.94, 1, 0.94, 1)</c></value>
        public static Color Honeydew { get { return namedColors["HONEYDEW"]; } }
        /// <value><c>(1, 0.41, 0.71, 1)</c></value>
        public static Color HotPink { get { return namedColors["HOTPINK"]; } }
        /// <value><c>(0.8, 0.36, 0.36, 1)</c></value>
        public static Color IndianRed { get { return namedColors["INDIANRED"]; } }
        /// <value><c>(0.29, 0, 0.51, 1)</c></value>
        public static Color Indigo { get { return namedColors["INDIGO"]; } }
        /// <value><c>(1, 1, 0.94, 1)</c></value>
        public static Color Ivory { get { return namedColors["IVORY"]; } }
        /// <value><c>(0.94, 0.9, 0.55, 1)</c></value>
        public static Color Khaki { get { return namedColors["KHAKI"]; } }
        /// <value><c>(0.9, 0.9, 0.98, 1)</c></value>
        public static Color Lavender { get { return namedColors["LAVENDER"]; } }
        /// <value><c>(1, 0.94, 0.96, 1)</c></value>
        public static Color LavenderBlush { get { return namedColors["LAVENDERBLUSH"]; } }
        /// <value><c>(0.49, 0.99, 0, 1)</c></value>
        public static Color LawnGreen { get { return namedColors["LAWNGREEN"]; } }
        /// <value><c>(1, 0.98, 0.8, 1)</c></value>
        public static Color LemonChiffon { get { return namedColors["LEMONCHIFFON"]; } }
        /// <value><c>(0.68, 0.85, 0.9, 1)</c></value>
        public static Color LightBlue { get { return namedColors["LIGHTBLUE"]; } }
        /// <value><c>(0.94, 0.5, 0.5, 1)</c></value>
        public static Color LightCoral { get { return namedColors["LIGHTCORAL"]; } }
        /// <value><c>(0.88, 1, 1, 1)</c></value>
        public static Color LightCyan { get { return namedColors["LIGHTCYAN"]; } }
        /// <value><c>(0.98, 0.98, 0.82, 1)</c></value>
        public static Color LightGoldenrod { get { return namedColors["LIGHTGOLDENROD"]; } }
        /// <value><c>(0.83, 0.83, 0.83, 1)</c></value>
        public static Color LightGray { get { return namedColors["LIGHTGRAY"]; } }
        /// <value><c>(0.56, 0.93, 0.56, 1)</c></value>
        public static Color LightGreen { get { return namedColors["LIGHTGREEN"]; } }
        /// <value><c>(1, 0.71, 0.76, 1)</c></value>
        public static Color LightPink { get { return namedColors["LIGHTPINK"]; } }
        /// <value><c>(1, 0.63, 0.48, 1)</c></value>
        public static Color LightSalmon { get { return namedColors["LIGHTSALMON"]; } }
        /// <value><c>(0.13, 0.7, 0.67, 1)</c></value>
        public static Color LightSeaGreen { get { return namedColors["LIGHTSEAGREEN"]; } }
        /// <value><c>(0.53, 0.81, 0.98, 1)</c></value>
        public static Color LightSkyBlue { get { return namedColors["LIGHTSKYBLUE"]; } }
        /// <value><c>(0.47, 0.53, 0.6, 1)</c></value>
        public static Color LightSlateGray { get { return namedColors["LIGHTSLATEGRAY"]; } }
        /// <value><c>(0.69, 0.77, 0.87, 1)</c></value>
        public static Color LightSteelBlue { get { return namedColors["LIGHTSTEELBLUE"]; } }
        /// <value><c>(1, 1, 0.88, 1)</c></value>
        public static Color LightYellow { get { return namedColors["LIGHTYELLOW"]; } }
        /// <value><c>(0, 1, 0, 1)</c></value>
        public static Color Lime { get { return namedColors["LIME"]; } }
        /// <value><c>(0.2, 0.8, 0.2, 1)</c></value>
        public static Color LimeGreen { get { return namedColors["LIMEGREEN"]; } }
        /// <value><c>(0.98, 0.94, 0.9, 1)</c></value>
        public static Color Linen { get { return namedColors["LINEN"]; } }
        /// <value><c>(1, 0, 1, 1)</c></value>
        public static Color Magenta { get { return namedColors["MAGENTA"]; } }
        /// <value><c>(0.69, 0.19, 0.38, 1)</c></value>
        public static Color Maroon { get { return namedColors["MAROON"]; } }
        /// <value><c>(0.4, 0.8, 0.67, 1)</c></value>
        public static Color MediumAquamarine { get { return namedColors["MEDIUMAQUAMARINE"]; } }
        /// <value><c>(0, 0, 0.8, 1)</c></value>
        public static Color MediumBlue { get { return namedColors["MEDIUMBLUE"]; } }
        /// <value><c>(0.73, 0.33, 0.83, 1)</c></value>
        public static Color MediumOrchid { get { return namedColors["MEDIUMORCHID"]; } }
        /// <value><c>(0.58, 0.44, 0.86, 1)</c></value>
        public static Color MediumPurple { get { return namedColors["MEDIUMPURPLE"]; } }
        /// <value><c>(0.24, 0.7, 0.44, 1)</c></value>
        public static Color MediumSeaGreen { get { return namedColors["MEDIUMSEAGREEN"]; } }
        /// <value><c>(0.48, 0.41, 0.93, 1)</c></value>
        public static Color MediumSlateBlue { get { return namedColors["MEDIUMSLATEBLUE"]; } }
        /// <value><c>(0, 0.98, 0.6, 1)</c></value>
        public static Color MediumSpringGreen { get { return namedColors["MEDIUMSPRINGGREEN"]; } }
        /// <value><c>(0.28, 0.82, 0.8, 1)</c></value>
        public static Color MediumTurquoise { get { return namedColors["MEDIUMTURQUOISE"]; } }
        /// <value><c>(0.78, 0.08, 0.52, 1)</c></value>
        public static Color MediumVioletRed { get { return namedColors["MEDIUMVIOLETRED"]; } }
        /// <value><c>(0.1, 0.1, 0.44, 1)</c></value>
        public static Color MidnightBlue { get { return namedColors["MIDNIGHTBLUE"]; } }
        /// <value><c>(0.96, 1, 0.98, 1)</c></value>
        public static Color MintCream { get { return namedColors["MINTCREAM"]; } }
        /// <value><c>(1, 0.89, 0.88, 1)</c></value>
        public static Color MistyRose { get { return namedColors["MISTYROSE"]; } }
        /// <value><c>(1, 0.89, 0.71, 1)</c></value>
        public static Color Moccasin { get { return namedColors["MOCCASIN"]; } }
        /// <value><c>(1, 0.87, 0.68, 1)</c></value>
        public static Color NavajoWhite { get { return namedColors["NAVAJOWHITE"]; } }
        /// <value><c>(0, 0, 0.5, 1)</c></value>
        public static Color NavyBlue { get { return namedColors["NAVYBLUE"]; } }
        /// <value><c>(0.99, 0.96, 0.9, 1)</c></value>
        public static Color OldLace { get { return namedColors["OLDLACE"]; } }
        /// <value><c>(0.5, 0.5, 0, 1)</c></value>
        public static Color Olive { get { return namedColors["OLIVE"]; } }
        /// <value><c>(0.42, 0.56, 0.14, 1)</c></value>
        public static Color OliveDrab { get { return namedColors["OLIVEDRAB"]; } }
        /// <value><c>(1, 0.65, 0, 1)</c></value>
        public static Color Orange { get { return namedColors["ORANGE"]; } }
        /// <value><c>(1, 0.27, 0, 1)</c></value>
        public static Color OrangeRed { get { return namedColors["ORANGERED"]; } }
        /// <value><c>(0.85, 0.44, 0.84, 1)</c></value>
        public static Color Orchid { get { return namedColors["ORCHID"]; } }
        /// <value><c>(0.93, 0.91, 0.67, 1)</c></value>
        public static Color PaleGoldenrod { get { return namedColors["PALEGOLDENROD"]; } }
        /// <value><c>(0.6, 0.98, 0.6, 1)</c></value>
        public static Color PaleGreen { get { return namedColors["PALEGREEN"]; } }
        /// <value><c>(0.69, 0.93, 0.93, 1)</c></value>
        public static Color PaleTurquoise { get { return namedColors["PALETURQUOISE"]; } }
        /// <value><c>(0.86, 0.44, 0.58, 1)</c></value>
        public static Color PaleVioletRed { get { return namedColors["PALEVIOLETRED"]; } }
        /// <value><c>(1, 0.94, 0.84, 1)</c></value>
        public static Color PapayaWhip { get { return namedColors["PAPAYAWHIP"]; } }
        /// <value><c>(1, 0.85, 0.73, 1)</c></value>
        public static Color PeachPuff { get { return namedColors["PEACHPUFF"]; } }
        /// <value><c>(0.8, 0.52, 0.25, 1)</c></value>
        public static Color Peru { get { return namedColors["PERU"]; } }
        /// <value><c>(1, 0.75, 0.8, 1)</c></value>
        public static Color Pink { get { return namedColors["PINK"]; } }
        /// <value><c>(0.87, 0.63, 0.87, 1)</c></value>
        public static Color Plum { get { return namedColors["PLUM"]; } }
        /// <value><c>(0.69, 0.88, 0.9, 1)</c></value>
        public static Color PowderBlue { get { return namedColors["POWDERBLUE"]; } }
        /// <value><c>(0.63, 0.13, 0.94, 1)</c></value>
        public static Color Purple { get { return namedColors["PURPLE"]; } }
        /// <value><c>(0.4, 0.2, 0.6, 1)</c></value>
        public static Color RebeccaPurple { get { return namedColors["REBECCAPURPLE"]; } }
        /// <value><c>(1, 0, 0, 1)</c></value>
        public static Color Red { get { return namedColors["RED"]; } }
        /// <value><c>(0.74, 0.56, 0.56, 1)</c></value>
        public static Color RosyBrown { get { return namedColors["ROSYBROWN"]; } }
        /// <value><c>(0.25, 0.41, 0.88, 1)</c></value>
        public static Color RoyalBlue { get { return namedColors["ROYALBLUE"]; } }
        /// <value><c>(0.55, 0.27, 0.07, 1)</c></value>
        public static Color SaddleBrown { get { return namedColors["SADDLEBROWN"]; } }
        /// <value><c>(0.98, 0.5, 0.45, 1)</c></value>
        public static Color Salmon { get { return namedColors["SALMON"]; } }
        /// <value><c>(0.96, 0.64, 0.38, 1)</c></value>
        public static Color SandyBrown { get { return namedColors["SANDYBROWN"]; } }
        /// <value><c>(0.18, 0.55, 0.34, 1)</c></value>
        public static Color SeaGreen { get { return namedColors["SEAGREEN"]; } }
        /// <value><c>(1, 0.96, 0.93, 1)</c></value>
        public static Color Seashell { get { return namedColors["SEASHELL"]; } }
        /// <value><c>(0.63, 0.32, 0.18, 1)</c></value>
        public static Color Sienna { get { return namedColors["SIENNA"]; } }
        /// <value><c>(0.75, 0.75, 0.75, 1)</c></value>
        public static Color Silver { get { return namedColors["SILVER"]; } }
        /// <value><c>(0.53, 0.81, 0.92, 1)</c></value>
        public static Color SkyBlue { get { return namedColors["SKYBLUE"]; } }
        /// <value><c>(0.42, 0.35, 0.8, 1)</c></value>
        public static Color SlateBlue { get { return namedColors["SLATEBLUE"]; } }
        /// <value><c>(0.44, 0.5, 0.56, 1)</c></value>
        public static Color SlateGray { get { return namedColors["SLATEGRAY"]; } }
        /// <value><c>(1, 0.98, 0.98, 1)</c></value>
        public static Color Snow { get { return namedColors["SNOW"]; } }
        /// <value><c>(0, 1, 0.5, 1)</c></value>
        public static Color SpringGreen { get { return namedColors["SPRINGGREEN"]; } }
        /// <value><c>(0.27, 0.51, 0.71, 1)</c></value>
        public static Color SteelBlue { get { return namedColors["STEELBLUE"]; } }
        /// <value><c>(0.82, 0.71, 0.55, 1)</c></value>
        public static Color Tan { get { return namedColors["TAN"]; } }
        /// <value><c>(0, 0.5, 0.5, 1)</c></value>
        public static Color Teal { get { return namedColors["TEAL"]; } }
        /// <value><c>(0.85, 0.75, 0.85, 1)</c></value>
        public static Color Thistle { get { return namedColors["THISTLE"]; } }
        /// <value><c>(1, 0.39, 0.28, 1)</c></value>
        public static Color Tomato { get { return namedColors["TOMATO"]; } }
        /// <value><c>(1, 1, 1, 0)</c></value>
        public static Color Transparent { get { return namedColors["TRANSPARENT"]; } }
        /// <value><c>(0.25, 0.88, 0.82, 1)</c></value>
        public static Color Turquoise { get { return namedColors["TURQUOISE"]; } }
        /// <value><c>(0.93, 0.51, 0.93, 1)</c></value>
        public static Color Violet { get { return namedColors["VIOLET"]; } }
        /// <value><c>(0.5, 0.5, 0.5, 1)</c></value>
        public static Color WebGray { get { return namedColors["WEBGRAY"]; } }
        /// <value><c>(0, 0.5, 0, 1)</c></value>
        public static Color WebGreen { get { return namedColors["WEBGREEN"]; } }
        /// <value><c>(0.5, 0, 0, 1)</c></value>
        public static Color WebMaroon { get { return namedColors["WEBMAROON"]; } }
        /// <value><c>(0.5, 0, 0.5, 1)</c></value>
        public static Color WebPurple { get { return namedColors["WEBPURPLE"]; } }
        /// <value><c>(0.96, 0.87, 0.7, 1)</c></value>
        public static Color Wheat { get { return namedColors["WHEAT"]; } }
        /// <value><c>(1, 1, 1, 1)</c></value>
        public static Color White { get { return namedColors["WHITE"]; } }
        /// <value><c>(0.96, 0.96, 0.96, 1)</c></value>
        public static Color WhiteSmoke { get { return namedColors["WHITESMOKE"]; } }
        /// <value><c>(1, 1, 0, 1)</c></value>
        public static Color Yellow { get { return namedColors["YELLOW"]; } }
        /// <value><c>(0.6, 0.8, 0.2, 1)</c></value>
        public static Color YellowGreen { get { return namedColors["YELLOWGREEN"]; } }
    }
}
