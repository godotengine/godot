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
        internal static readonly Dictionary<string, Color> NamedColors = new Dictionary<string, Color> {
            { "ALICEBLUE", Colors.AliceBlue },
            { "ANTIQUEWHITE", Colors.AntiqueWhite },
            { "AQUA", Colors.Aqua },
            { "AQUAMARINE", Colors.Aquamarine },
            { "AZURE", Colors.Azure },
            { "BEIGE", Colors.Beige },
            { "BISQUE", Colors.Bisque },
            { "BLACK", Colors.Black },
            { "BLANCHEDALMOND", Colors.BlanchedAlmond },
            { "BLUE", Colors.Blue },
            { "BLUEVIOLET", Colors.BlueViolet },
            { "BROWN", Colors.Brown },
            { "BURLYWOOD", Colors.Burlywood },
            { "CADETBLUE", Colors.CadetBlue },
            { "CHARTREUSE", Colors.Chartreuse },
            { "CHOCOLATE", Colors.Chocolate },
            { "CORAL", Colors.Coral },
            { "CORNFLOWERBLUE", Colors.CornflowerBlue },
            { "CORNSILK", Colors.Cornsilk },
            { "CRIMSON", Colors.Crimson },
            { "CYAN", Colors.Cyan },
            { "DARKBLUE", Colors.DarkBlue },
            { "DARKCYAN", Colors.DarkCyan },
            { "DARKGOLDENROD", Colors.DarkGoldenrod },
            { "DARKGRAY", Colors.DarkGray },
            { "DARKGREEN", Colors.DarkGreen },
            { "DARKKHAKI", Colors.DarkKhaki },
            { "DARKMAGENTA", Colors.DarkMagenta },
            { "DARKOLIVEGREEN", Colors.DarkOliveGreen },
            { "DARKORANGE", Colors.DarkOrange },
            { "DARKORCHID", Colors.DarkOrchid },
            { "DARKRED", Colors.DarkRed },
            { "DARKSALMON", Colors.DarkSalmon },
            { "DARKSEAGREEN", Colors.DarkSeaGreen },
            { "DARKSLATEBLUE", Colors.DarkSlateBlue },
            { "DARKSLATEGRAY", Colors.DarkSlateGray },
            { "DARKTURQUOISE", Colors.DarkTurquoise },
            { "DARKVIOLET", Colors.DarkViolet },
            { "DEEPPINK", Colors.DeepPink },
            { "DEEPSKYBLUE", Colors.DeepSkyBlue },
            { "DIMGRAY", Colors.DimGray },
            { "DODGERBLUE", Colors.DodgerBlue },
            { "FIREBRICK", Colors.Firebrick },
            { "FLORALWHITE", Colors.FloralWhite },
            { "FORESTGREEN", Colors.ForestGreen },
            { "FUCHSIA", Colors.Fuchsia },
            { "GAINSBORO", Colors.Gainsboro },
            { "GHOSTWHITE", Colors.GhostWhite },
            { "GOLD", Colors.Gold },
            { "GOLDENROD", Colors.Goldenrod },
            { "GRAY", Colors.Gray },
            { "GREEN", Colors.Green },
            { "GREENYELLOW", Colors.GreenYellow },
            { "HONEYDEW", Colors.Honeydew },
            { "HOTPINK", Colors.HotPink },
            { "INDIANRED", Colors.IndianRed },
            { "INDIGO", Colors.Indigo },
            { "IVORY", Colors.Ivory },
            { "KHAKI", Colors.Khaki },
            { "LAVENDER", Colors.Lavender },
            { "LAVENDERBLUSH", Colors.LavenderBlush },
            { "LAWNGREEN", Colors.LawnGreen },
            { "LEMONCHIFFON", Colors.LemonChiffon },
            { "LIGHTBLUE", Colors.LightBlue },
            { "LIGHTCORAL", Colors.LightCoral },
            { "LIGHTCYAN", Colors.LightCyan },
            { "LIGHTGOLDENROD", Colors.LightGoldenrod },
            { "LIGHTGRAY", Colors.LightGray },
            { "LIGHTGREEN", Colors.LightGreen },
            { "LIGHTPINK", Colors.LightPink },
            { "LIGHTSALMON", Colors.LightSalmon },
            { "LIGHTSEAGREEN", Colors.LightSeaGreen },
            { "LIGHTSKYBLUE", Colors.LightSkyBlue },
            { "LIGHTSLATEGRAY", Colors.LightSlateGray },
            { "LIGHTSTEELBLUE", Colors.LightSteelBlue },
            { "LIGHTYELLOW", Colors.LightYellow },
            { "LIME", Colors.Lime },
            { "LIMEGREEN", Colors.LimeGreen },
            { "LINEN", Colors.Linen },
            { "MAGENTA", Colors.Magenta },
            { "MAROON", Colors.Maroon },
            { "MEDIUMAQUAMARINE", Colors.MediumAquamarine },
            { "MEDIUMBLUE", Colors.MediumBlue },
            { "MEDIUMORCHID", Colors.MediumOrchid },
            { "MEDIUMPURPLE", Colors.MediumPurple },
            { "MEDIUMSEAGREEN", Colors.MediumSeaGreen },
            { "MEDIUMSLATEBLUE", Colors.MediumSlateBlue },
            { "MEDIUMSPRINGGREEN", Colors.MediumSpringGreen },
            { "MEDIUMTURQUOISE", Colors.MediumTurquoise },
            { "MEDIUMVIOLETRED", Colors.MediumVioletRed },
            { "MIDNIGHTBLUE", Colors.MidnightBlue },
            { "MINTCREAM", Colors.MintCream },
            { "MISTYROSE", Colors.MistyRose },
            { "MOCCASIN", Colors.Moccasin },
            { "NAVAJOWHITE", Colors.NavajoWhite },
            { "NAVYBLUE", Colors.NavyBlue },
            { "OLDLACE", Colors.OldLace },
            { "OLIVE", Colors.Olive },
            { "OLIVEDRAB", Colors.OliveDrab },
            { "ORANGE", Colors.Orange },
            { "ORANGERED", Colors.OrangeRed },
            { "ORCHID", Colors.Orchid },
            { "PALEGOLDENROD", Colors.PaleGoldenrod },
            { "PALEGREEN", Colors.PaleGreen },
            { "PALETURQUOISE", Colors.PaleTurquoise },
            { "PALEVIOLETRED", Colors.PaleVioletRed },
            { "PAPAYAWHIP", Colors.PapayaWhip },
            { "PEACHPUFF", Colors.PeachPuff },
            { "PERU", Colors.Peru },
            { "PINK", Colors.Pink },
            { "PLUM", Colors.Plum },
            { "POWDERBLUE", Colors.PowderBlue },
            { "PURPLE", Colors.Purple },
            { "REBECCAPURPLE", Colors.RebeccaPurple },
            { "RED", Colors.Red },
            { "ROSYBROWN", Colors.RosyBrown },
            { "ROYALBLUE", Colors.RoyalBlue },
            { "SADDLEBROWN", Colors.SaddleBrown },
            { "SALMON", Colors.Salmon },
            { "SANDYBROWN", Colors.SandyBrown },
            { "SEAGREEN", Colors.SeaGreen },
            { "SEASHELL", Colors.Seashell },
            { "SIENNA", Colors.Sienna },
            { "SILVER", Colors.Silver },
            { "SKYBLUE", Colors.SkyBlue },
            { "SLATEBLUE", Colors.SlateBlue },
            { "SLATEGRAY", Colors.SlateGray },
            { "SNOW", Colors.Snow },
            { "SPRINGGREEN", Colors.SpringGreen },
            { "STEELBLUE", Colors.SteelBlue },
            { "TAN", Colors.Tan },
            { "TEAL", Colors.Teal },
            { "THISTLE", Colors.Thistle },
            { "TOMATO", Colors.Tomato },
            { "TRANSPARENT", Colors.Transparent },
            { "TURQUOISE", Colors.Turquoise },
            { "VIOLET", Colors.Violet },
            { "WEBGRAY", Colors.WebGray },
            { "WEBGREEN", Colors.WebGreen },
            { "WEBMAROON", Colors.WebMaroon },
            { "WEBPURPLE", Colors.WebPurple },
            { "WHEAT", Colors.Wheat },
            { "WHITE", Colors.White },
            { "WHITESMOKE", Colors.WhiteSmoke },
            { "YELLOW", Colors.Yellow },
            { "YELLOWGREEN", Colors.YellowGreen },
        };

#pragma warning disable CS1591 // Disable warning: "Missing XML comment for publicly visible type or member"
        public static Color AliceBlue => new Color(0xF0F8FFFF);
        public static Color AntiqueWhite => new Color(0xFAEBD7FF);
        public static Color Aqua => new Color(0x00FFFFFF);
        public static Color Aquamarine => new Color(0x7FFFD4FF);
        public static Color Azure => new Color(0xF0FFFFFF);
        public static Color Beige => new Color(0xF5F5DCFF);
        public static Color Bisque => new Color(0xFFE4C4FF);
        public static Color Black => new Color(0x000000FF);
        public static Color BlanchedAlmond => new Color(0xFFEBCDFF);
        public static Color Blue => new Color(0x0000FFFF);
        public static Color BlueViolet => new Color(0x8A2BE2FF);
        public static Color Brown => new Color(0xA52A2AFF);
        public static Color Burlywood => new Color(0xDEB887FF);
        public static Color CadetBlue => new Color(0x5F9EA0FF);
        public static Color Chartreuse => new Color(0x7FFF00FF);
        public static Color Chocolate => new Color(0xD2691EFF);
        public static Color Coral => new Color(0xFF7F50FF);
        public static Color CornflowerBlue => new Color(0x6495EDFF);
        public static Color Cornsilk => new Color(0xFFF8DCFF);
        public static Color Crimson => new Color(0xDC143CFF);
        public static Color Cyan => new Color(0x00FFFFFF);
        public static Color DarkBlue => new Color(0x00008BFF);
        public static Color DarkCyan => new Color(0x008B8BFF);
        public static Color DarkGoldenrod => new Color(0xB8860BFF);
        public static Color DarkGray => new Color(0xA9A9A9FF);
        public static Color DarkGreen => new Color(0x006400FF);
        public static Color DarkKhaki => new Color(0xBDB76BFF);
        public static Color DarkMagenta => new Color(0x8B008BFF);
        public static Color DarkOliveGreen => new Color(0x556B2FFF);
        public static Color DarkOrange => new Color(0xFF8C00FF);
        public static Color DarkOrchid => new Color(0x9932CCFF);
        public static Color DarkRed => new Color(0x8B0000FF);
        public static Color DarkSalmon => new Color(0xE9967AFF);
        public static Color DarkSeaGreen => new Color(0x8FBC8FFF);
        public static Color DarkSlateBlue => new Color(0x483D8BFF);
        public static Color DarkSlateGray => new Color(0x2F4F4FFF);
        public static Color DarkTurquoise => new Color(0x00CED1FF);
        public static Color DarkViolet => new Color(0x9400D3FF);
        public static Color DeepPink => new Color(0xFF1493FF);
        public static Color DeepSkyBlue => new Color(0x00BFFFFF);
        public static Color DimGray => new Color(0x696969FF);
        public static Color DodgerBlue => new Color(0x1E90FFFF);
        public static Color Firebrick => new Color(0xB22222FF);
        public static Color FloralWhite => new Color(0xFFFAF0FF);
        public static Color ForestGreen => new Color(0x228B22FF);
        public static Color Fuchsia => new Color(0xFF00FFFF);
        public static Color Gainsboro => new Color(0xDCDCDCFF);
        public static Color GhostWhite => new Color(0xF8F8FFFF);
        public static Color Gold => new Color(0xFFD700FF);
        public static Color Goldenrod => new Color(0xDAA520FF);
        public static Color Gray => new Color(0xBEBEBEFF);
        public static Color Green => new Color(0x00FF00FF);
        public static Color GreenYellow => new Color(0xADFF2FFF);
        public static Color Honeydew => new Color(0xF0FFF0FF);
        public static Color HotPink => new Color(0xFF69B4FF);
        public static Color IndianRed => new Color(0xCD5C5CFF);
        public static Color Indigo => new Color(0x4B0082FF);
        public static Color Ivory => new Color(0xFFFFF0FF);
        public static Color Khaki => new Color(0xF0E68CFF);
        public static Color Lavender => new Color(0xE6E6FAFF);
        public static Color LavenderBlush => new Color(0xFFF0F5FF);
        public static Color LawnGreen => new Color(0x7CFC00FF);
        public static Color LemonChiffon => new Color(0xFFFACDFF);
        public static Color LightBlue => new Color(0xADD8E6FF);
        public static Color LightCoral => new Color(0xF08080FF);
        public static Color LightCyan => new Color(0xE0FFFFFF);
        public static Color LightGoldenrod => new Color(0xFAFAD2FF);
        public static Color LightGray => new Color(0xD3D3D3FF);
        public static Color LightGreen => new Color(0x90EE90FF);
        public static Color LightPink => new Color(0xFFB6C1FF);
        public static Color LightSalmon => new Color(0xFFA07AFF);
        public static Color LightSeaGreen => new Color(0x20B2AAFF);
        public static Color LightSkyBlue => new Color(0x87CEFAFF);
        public static Color LightSlateGray => new Color(0x778899FF);
        public static Color LightSteelBlue => new Color(0xB0C4DEFF);
        public static Color LightYellow => new Color(0xFFFFE0FF);
        public static Color Lime => new Color(0x00FF00FF);
        public static Color LimeGreen => new Color(0x32CD32FF);
        public static Color Linen => new Color(0xFAF0E6FF);
        public static Color Magenta => new Color(0xFF00FFFF);
        public static Color Maroon => new Color(0xB03060FF);
        public static Color MediumAquamarine => new Color(0x66CDAAFF);
        public static Color MediumBlue => new Color(0x0000CDFF);
        public static Color MediumOrchid => new Color(0xBA55D3FF);
        public static Color MediumPurple => new Color(0x9370DBFF);
        public static Color MediumSeaGreen => new Color(0x3CB371FF);
        public static Color MediumSlateBlue => new Color(0x7B68EEFF);
        public static Color MediumSpringGreen => new Color(0x00FA9AFF);
        public static Color MediumTurquoise => new Color(0x48D1CCFF);
        public static Color MediumVioletRed => new Color(0xC71585FF);
        public static Color MidnightBlue => new Color(0x191970FF);
        public static Color MintCream => new Color(0xF5FFFAFF);
        public static Color MistyRose => new Color(0xFFE4E1FF);
        public static Color Moccasin => new Color(0xFFE4B5FF);
        public static Color NavajoWhite => new Color(0xFFDEADFF);
        public static Color NavyBlue => new Color(0x000080FF);
        public static Color OldLace => new Color(0xFDF5E6FF);
        public static Color Olive => new Color(0x808000FF);
        public static Color OliveDrab => new Color(0x6B8E23FF);
        public static Color Orange => new Color(0xFFA500FF);
        public static Color OrangeRed => new Color(0xFF4500FF);
        public static Color Orchid => new Color(0xDA70D6FF);
        public static Color PaleGoldenrod => new Color(0xEEE8AAFF);
        public static Color PaleGreen => new Color(0x98FB98FF);
        public static Color PaleTurquoise => new Color(0xAFEEEEFF);
        public static Color PaleVioletRed => new Color(0xDB7093FF);
        public static Color PapayaWhip => new Color(0xFFEFD5FF);
        public static Color PeachPuff => new Color(0xFFDAB9FF);
        public static Color Peru => new Color(0xCD853FFF);
        public static Color Pink => new Color(0xFFC0CBFF);
        public static Color Plum => new Color(0xDDA0DDFF);
        public static Color PowderBlue => new Color(0xB0E0E6FF);
        public static Color Purple => new Color(0xA020F0FF);
        public static Color RebeccaPurple => new Color(0x663399FF);
        public static Color Red => new Color(0xFF0000FF);
        public static Color RosyBrown => new Color(0xBC8F8FFF);
        public static Color RoyalBlue => new Color(0x4169E1FF);
        public static Color SaddleBrown => new Color(0x8B4513FF);
        public static Color Salmon => new Color(0xFA8072FF);
        public static Color SandyBrown => new Color(0xF4A460FF);
        public static Color SeaGreen => new Color(0x2E8B57FF);
        public static Color Seashell => new Color(0xFFF5EEFF);
        public static Color Sienna => new Color(0xA0522DFF);
        public static Color Silver => new Color(0xC0C0C0FF);
        public static Color SkyBlue => new Color(0x87CEEBFF);
        public static Color SlateBlue => new Color(0x6A5ACDFF);
        public static Color SlateGray => new Color(0x708090FF);
        public static Color Snow => new Color(0xFFFAFAFF);
        public static Color SpringGreen => new Color(0x00FF7FFF);
        public static Color SteelBlue => new Color(0x4682B4FF);
        public static Color Tan => new Color(0xD2B48CFF);
        public static Color Teal => new Color(0x008080FF);
        public static Color Thistle => new Color(0xD8BFD8FF);
        public static Color Tomato => new Color(0xFF6347FF);
        public static Color Transparent => new Color(0xFFFFFF00);
        public static Color Turquoise => new Color(0x40E0D0FF);
        public static Color Violet => new Color(0xEE82EEFF);
        public static Color WebGray => new Color(0x808080FF);
        public static Color WebGreen => new Color(0x008000FF);
        public static Color WebMaroon => new Color(0x800000FF);
        public static Color WebPurple => new Color(0x800080FF);
        public static Color Wheat => new Color(0xF5DEB3FF);
        public static Color White => new Color(0xFFFFFFFF);
        public static Color WhiteSmoke => new Color(0xF5F5F5FF);
        public static Color Yellow => new Color(0xFFFF00FF);
        public static Color YellowGreen => new Color(0x9ACD32FF);
#pragma warning restore CS1591
    }
}
