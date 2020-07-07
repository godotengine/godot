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
        internal static readonly Dictionary<string, Color> namedColors = new Dictionary<string, Color> {
            {"aliceblue", new Color(0.94f, 0.97f, 1.00f)},
            {"antiquewhite", new Color(0.98f, 0.92f, 0.84f)},
            {"aqua", new Color(0.00f, 1.00f, 1.00f)},
            {"aquamarine", new Color(0.50f, 1.00f, 0.83f)},
            {"azure", new Color(0.94f, 1.00f, 1.00f)},
            {"beige", new Color(0.96f, 0.96f, 0.86f)},
            {"bisque", new Color(1.00f, 0.89f, 0.77f)},
            {"black", new Color(0.00f, 0.00f, 0.00f)},
            {"blanchedalmond", new Color(1.00f, 0.92f, 0.80f)},
            {"blue", new Color(0.00f, 0.00f, 1.00f)},
            {"blueviolet", new Color(0.54f, 0.17f, 0.89f)},
            {"brown", new Color(0.65f, 0.16f, 0.16f)},
            {"burlywood", new Color(0.87f, 0.72f, 0.53f)},
            {"cadetblue", new Color(0.37f, 0.62f, 0.63f)},
            {"chartreuse", new Color(0.50f, 1.00f, 0.00f)},
            {"chocolate", new Color(0.82f, 0.41f, 0.12f)},
            {"coral", new Color(1.00f, 0.50f, 0.31f)},
            {"cornflower", new Color(0.39f, 0.58f, 0.93f)},
            {"cornsilk", new Color(1.00f, 0.97f, 0.86f)},
            {"crimson", new Color(0.86f, 0.08f, 0.24f)},
            {"cyan", new Color(0.00f, 1.00f, 1.00f)},
            {"darkblue", new Color(0.00f, 0.00f, 0.55f)},
            {"darkcyan", new Color(0.00f, 0.55f, 0.55f)},
            {"darkgoldenrod", new Color(0.72f, 0.53f, 0.04f)},
            {"darkgray", new Color(0.66f, 0.66f, 0.66f)},
            {"darkgreen", new Color(0.00f, 0.39f, 0.00f)},
            {"darkkhaki", new Color(0.74f, 0.72f, 0.42f)},
            {"darkmagenta", new Color(0.55f, 0.00f, 0.55f)},
            {"darkolivegreen", new Color(0.33f, 0.42f, 0.18f)},
            {"darkorange", new Color(1.00f, 0.55f, 0.00f)},
            {"darkorchid", new Color(0.60f, 0.20f, 0.80f)},
            {"darkred", new Color(0.55f, 0.00f, 0.00f)},
            {"darksalmon", new Color(0.91f, 0.59f, 0.48f)},
            {"darkseagreen", new Color(0.56f, 0.74f, 0.56f)},
            {"darkslateblue", new Color(0.28f, 0.24f, 0.55f)},
            {"darkslategray", new Color(0.18f, 0.31f, 0.31f)},
            {"darkturquoise", new Color(0.00f, 0.81f, 0.82f)},
            {"darkviolet", new Color(0.58f, 0.00f, 0.83f)},
            {"deeppink", new Color(1.00f, 0.08f, 0.58f)},
            {"deepskyblue", new Color(0.00f, 0.75f, 1.00f)},
            {"dimgray", new Color(0.41f, 0.41f, 0.41f)},
            {"dodgerblue", new Color(0.12f, 0.56f, 1.00f)},
            {"firebrick", new Color(0.70f, 0.13f, 0.13f)},
            {"floralwhite", new Color(1.00f, 0.98f, 0.94f)},
            {"forestgreen", new Color(0.13f, 0.55f, 0.13f)},
            {"fuchsia", new Color(1.00f, 0.00f, 1.00f)},
            {"gainsboro", new Color(0.86f, 0.86f, 0.86f)},
            {"ghostwhite", new Color(0.97f, 0.97f, 1.00f)},
            {"gold", new Color(1.00f, 0.84f, 0.00f)},
            {"goldenrod", new Color(0.85f, 0.65f, 0.13f)},
            {"gray", new Color(0.75f, 0.75f, 0.75f)},
            {"green", new Color(0.00f, 1.00f, 0.00f)},
            {"greenyellow", new Color(0.68f, 1.00f, 0.18f)},
            {"honeydew", new Color(0.94f, 1.00f, 0.94f)},
            {"hotpink", new Color(1.00f, 0.41f, 0.71f)},
            {"indianred", new Color(0.80f, 0.36f, 0.36f)},
            {"indigo", new Color(0.29f, 0.00f, 0.51f)},
            {"ivory", new Color(1.00f, 1.00f, 0.94f)},
            {"khaki", new Color(0.94f, 0.90f, 0.55f)},
            {"lavender", new Color(0.90f, 0.90f, 0.98f)},
            {"lavenderblush", new Color(1.00f, 0.94f, 0.96f)},
            {"lawngreen", new Color(0.49f, 0.99f, 0.00f)},
            {"lemonchiffon", new Color(1.00f, 0.98f, 0.80f)},
            {"lightblue", new Color(0.68f, 0.85f, 0.90f)},
            {"lightcoral", new Color(0.94f, 0.50f, 0.50f)},
            {"lightcyan", new Color(0.88f, 1.00f, 1.00f)},
            {"lightgoldenrod", new Color(0.98f, 0.98f, 0.82f)},
            {"lightgray", new Color(0.83f, 0.83f, 0.83f)},
            {"lightgreen", new Color(0.56f, 0.93f, 0.56f)},
            {"lightpink", new Color(1.00f, 0.71f, 0.76f)},
            {"lightsalmon", new Color(1.00f, 0.63f, 0.48f)},
            {"lightseagreen", new Color(0.13f, 0.70f, 0.67f)},
            {"lightskyblue", new Color(0.53f, 0.81f, 0.98f)},
            {"lightslategray", new Color(0.47f, 0.53f, 0.60f)},
            {"lightsteelblue", new Color(0.69f, 0.77f, 0.87f)},
            {"lightyellow", new Color(1.00f, 1.00f, 0.88f)},
            {"lime", new Color(0.00f, 1.00f, 0.00f)},
            {"limegreen", new Color(0.20f, 0.80f, 0.20f)},
            {"linen", new Color(0.98f, 0.94f, 0.90f)},
            {"magenta", new Color(1.00f, 0.00f, 1.00f)},
            {"maroon", new Color(0.69f, 0.19f, 0.38f)},
            {"mediumaquamarine", new Color(0.40f, 0.80f, 0.67f)},
            {"mediumblue", new Color(0.00f, 0.00f, 0.80f)},
            {"mediumorchid", new Color(0.73f, 0.33f, 0.83f)},
            {"mediumpurple", new Color(0.58f, 0.44f, 0.86f)},
            {"mediumseagreen", new Color(0.24f, 0.70f, 0.44f)},
            {"mediumslateblue", new Color(0.48f, 0.41f, 0.93f)},
            {"mediumspringgreen", new Color(0.00f, 0.98f, 0.60f)},
            {"mediumturquoise", new Color(0.28f, 0.82f, 0.80f)},
            {"mediumvioletred", new Color(0.78f, 0.08f, 0.52f)},
            {"midnightblue", new Color(0.10f, 0.10f, 0.44f)},
            {"mintcream", new Color(0.96f, 1.00f, 0.98f)},
            {"mistyrose", new Color(1.00f, 0.89f, 0.88f)},
            {"moccasin", new Color(1.00f, 0.89f, 0.71f)},
            {"navajowhite", new Color(1.00f, 0.87f, 0.68f)},
            {"navyblue", new Color(0.00f, 0.00f, 0.50f)},
            {"oldlace", new Color(0.99f, 0.96f, 0.90f)},
            {"olive", new Color(0.50f, 0.50f, 0.00f)},
            {"olivedrab", new Color(0.42f, 0.56f, 0.14f)},
            {"orange", new Color(1.00f, 0.65f, 0.00f)},
            {"orangered", new Color(1.00f, 0.27f, 0.00f)},
            {"orchid", new Color(0.85f, 0.44f, 0.84f)},
            {"palegoldenrod", new Color(0.93f, 0.91f, 0.67f)},
            {"palegreen", new Color(0.60f, 0.98f, 0.60f)},
            {"paleturquoise", new Color(0.69f, 0.93f, 0.93f)},
            {"palevioletred", new Color(0.86f, 0.44f, 0.58f)},
            {"papayawhip", new Color(1.00f, 0.94f, 0.84f)},
            {"peachpuff", new Color(1.00f, 0.85f, 0.73f)},
            {"peru", new Color(0.80f, 0.52f, 0.25f)},
            {"pink", new Color(1.00f, 0.75f, 0.80f)},
            {"plum", new Color(0.87f, 0.63f, 0.87f)},
            {"powderblue", new Color(0.69f, 0.88f, 0.90f)},
            {"purple", new Color(0.63f, 0.13f, 0.94f)},
            {"rebeccapurple", new Color(0.40f, 0.20f, 0.60f)},
            {"red", new Color(1.00f, 0.00f, 0.00f)},
            {"rosybrown", new Color(0.74f, 0.56f, 0.56f)},
            {"royalblue", new Color(0.25f, 0.41f, 0.88f)},
            {"saddlebrown", new Color(0.55f, 0.27f, 0.07f)},
            {"salmon", new Color(0.98f, 0.50f, 0.45f)},
            {"sandybrown", new Color(0.96f, 0.64f, 0.38f)},
            {"seagreen", new Color(0.18f, 0.55f, 0.34f)},
            {"seashell", new Color(1.00f, 0.96f, 0.93f)},
            {"sienna", new Color(0.63f, 0.32f, 0.18f)},
            {"silver", new Color(0.75f, 0.75f, 0.75f)},
            {"skyblue", new Color(0.53f, 0.81f, 0.92f)},
            {"slateblue", new Color(0.42f, 0.35f, 0.80f)},
            {"slategray", new Color(0.44f, 0.50f, 0.56f)},
            {"snow", new Color(1.00f, 0.98f, 0.98f)},
            {"springgreen", new Color(0.00f, 1.00f, 0.50f)},
            {"steelblue", new Color(0.27f, 0.51f, 0.71f)},
            {"tan", new Color(0.82f, 0.71f, 0.55f)},
            {"teal", new Color(0.00f, 0.50f, 0.50f)},
            {"thistle", new Color(0.85f, 0.75f, 0.85f)},
            {"tomato", new Color(1.00f, 0.39f, 0.28f)},
            {"transparent", new Color(1.00f, 1.00f, 1.00f, 0.00f)},
            {"turquoise", new Color(0.25f, 0.88f, 0.82f)},
            {"violet", new Color(0.93f, 0.51f, 0.93f)},
            {"webgreen", new Color(0.00f, 0.50f, 0.00f)},
            {"webgray", new Color(0.50f, 0.50f, 0.50f)},
            {"webmaroon", new Color(0.50f, 0.00f, 0.00f)},
            {"webpurple", new Color(0.50f, 0.00f, 0.50f)},
            {"wheat", new Color(0.96f, 0.87f, 0.70f)},
            {"white", new Color(1.00f, 1.00f, 1.00f)},
            {"whitesmoke", new Color(0.96f, 0.96f, 0.96f)},
            {"yellow", new Color(1.00f, 1.00f, 0.00f)},
            {"yellowgreen", new Color(0.60f, 0.80f, 0.20f)},
        };

        public static Color AliceBlue { get { return namedColors["aliceblue"]; } }
        public static Color AntiqueWhite { get { return namedColors["antiquewhite"]; } }
        public static Color Aqua { get { return namedColors["aqua"]; } }
        public static Color Aquamarine { get { return namedColors["aquamarine"]; } }
        public static Color Azure { get { return namedColors["azure"]; } }
        public static Color Beige { get { return namedColors["beige"]; } }
        public static Color Bisque { get { return namedColors["bisque"]; } }
        public static Color Black { get { return namedColors["black"]; } }
        public static Color BlanchedAlmond { get { return namedColors["blanchedalmond"]; } }
        public static Color Blue { get { return namedColors["blue"]; } }
        public static Color BlueViolet { get { return namedColors["blueviolet"]; } }
        public static Color Brown { get { return namedColors["brown"]; } }
        public static Color BurlyWood { get { return namedColors["burlywood"]; } }
        public static Color CadetBlue { get { return namedColors["cadetblue"]; } }
        public static Color Chartreuse { get { return namedColors["chartreuse"]; } }
        public static Color Chocolate { get { return namedColors["chocolate"]; } }
        public static Color Coral { get { return namedColors["coral"]; } }
        public static Color Cornflower { get { return namedColors["cornflower"]; } }
        public static Color Cornsilk { get { return namedColors["cornsilk"]; } }
        public static Color Crimson { get { return namedColors["crimson"]; } }
        public static Color Cyan { get { return namedColors["cyan"]; } }
        public static Color DarkBlue { get { return namedColors["darkblue"]; } }
        public static Color DarkCyan { get { return namedColors["darkcyan"]; } }
        public static Color DarkGoldenrod { get { return namedColors["darkgoldenrod"]; } }
        public static Color DarkGray { get { return namedColors["darkgray"]; } }
        public static Color DarkGreen { get { return namedColors["darkgreen"]; } }
        public static Color DarkKhaki { get { return namedColors["darkkhaki"]; } }
        public static Color DarkMagenta { get { return namedColors["darkmagenta"]; } }
        public static Color DarkOliveGreen { get { return namedColors["darkolivegreen"]; } }
        public static Color DarkOrange { get { return namedColors["darkorange"]; } }
        public static Color DarkOrchid { get { return namedColors["darkorchid"]; } }
        public static Color DarkRed { get { return namedColors["darkred"]; } }
        public static Color DarkSalmon { get { return namedColors["darksalmon"]; } }
        public static Color DarkSeaGreen { get { return namedColors["darkseagreen"]; } }
        public static Color DarkSlateBlue { get { return namedColors["darkslateblue"]; } }
        public static Color DarkSlateGray { get { return namedColors["darkslategray"]; } }
        public static Color DarkTurquoise { get { return namedColors["darkturquoise"]; } }
        public static Color DarkViolet { get { return namedColors["darkviolet"]; } }
        public static Color DeepPink { get { return namedColors["deeppink"]; } }
        public static Color DeepSkyBlue { get { return namedColors["deepskyblue"]; } }
        public static Color DimGray { get { return namedColors["dimgray"]; } }
        public static Color DodgerBlue { get { return namedColors["dodgerblue"]; } }
        public static Color Firebrick { get { return namedColors["firebrick"]; } }
        public static Color FloralWhite { get { return namedColors["floralwhite"]; } }
        public static Color ForestGreen { get { return namedColors["forestgreen"]; } }
        public static Color Fuchsia { get { return namedColors["fuchsia"]; } }
        public static Color Gainsboro { get { return namedColors["gainsboro"]; } }
        public static Color GhostWhite { get { return namedColors["ghostwhite"]; } }
        public static Color Gold { get { return namedColors["gold"]; } }
        public static Color Goldenrod { get { return namedColors["goldenrod"]; } }
        public static Color Gray { get { return namedColors["gray"]; } }
        public static Color Green { get { return namedColors["green"]; } }
        public static Color GreenYellow { get { return namedColors["greenyellow"]; } }
        public static Color Honeydew { get { return namedColors["honeydew"]; } }
        public static Color HotPink { get { return namedColors["hotpink"]; } }
        public static Color IndianRed { get { return namedColors["indianred"]; } }
        public static Color Indigo { get { return namedColors["indigo"]; } }
        public static Color Ivory { get { return namedColors["ivory"]; } }
        public static Color Khaki { get { return namedColors["khaki"]; } }
        public static Color Lavender { get { return namedColors["lavender"]; } }
        public static Color LavenderBlush { get { return namedColors["lavenderblush"]; } }
        public static Color LawnGreen { get { return namedColors["lawngreen"]; } }
        public static Color LemonChiffon { get { return namedColors["lemonchiffon"]; } }
        public static Color LightBlue { get { return namedColors["lightblue"]; } }
        public static Color LightCoral { get { return namedColors["lightcoral"]; } }
        public static Color LightCyan { get { return namedColors["lightcyan"]; } }
        public static Color LightGoldenrod { get { return namedColors["lightgoldenrod"]; } }
        public static Color LightGray { get { return namedColors["lightgray"]; } }
        public static Color LightGreen { get { return namedColors["lightgreen"]; } }
        public static Color LightPink { get { return namedColors["lightpink"]; } }
        public static Color LightSalmon { get { return namedColors["lightsalmon"]; } }
        public static Color LightSeaGreen { get { return namedColors["lightseagreen"]; } }
        public static Color LightSkyBlue { get { return namedColors["lightskyblue"]; } }
        public static Color LightSlateGray { get { return namedColors["lightslategray"]; } }
        public static Color LightSteelBlue { get { return namedColors["lightsteelblue"]; } }
        public static Color LightYellow { get { return namedColors["lightyellow"]; } }
        public static Color Lime { get { return namedColors["lime"]; } }
        public static Color Limegreen { get { return namedColors["limegreen"]; } }
        public static Color Linen { get { return namedColors["linen"]; } }
        public static Color Magenta { get { return namedColors["magenta"]; } }
        public static Color Maroon { get { return namedColors["maroon"]; } }
        public static Color MediumAquamarine { get { return namedColors["mediumaquamarine"]; } }
        public static Color MediumBlue { get { return namedColors["mediumblue"]; } }
        public static Color MediumOrchid { get { return namedColors["mediumorchid"]; } }
        public static Color MediumPurple { get { return namedColors["mediumpurple"]; } }
        public static Color MediumSeaGreen { get { return namedColors["mediumseagreen"]; } }
        public static Color MediumSlateBlue { get { return namedColors["mediumslateblue"]; } }
        public static Color MediumSpringGreen { get { return namedColors["mediumspringgreen"]; } }
        public static Color MediumTurquoise { get { return namedColors["mediumturquoise"]; } }
        public static Color MediumVioletRed { get { return namedColors["mediumvioletred"]; } }
        public static Color MidnightBlue { get { return namedColors["midnightblue"]; } }
        public static Color MintCream { get { return namedColors["mintcream"]; } }
        public static Color MistyRose { get { return namedColors["mistyrose"]; } }
        public static Color Moccasin { get { return namedColors["moccasin"]; } }
        public static Color NavajoWhite { get { return namedColors["navajowhite"]; } }
        public static Color NavyBlue { get { return namedColors["navyblue"]; } }
        public static Color OldLace { get { return namedColors["oldlace"]; } }
        public static Color Olive { get { return namedColors["olive"]; } }
        public static Color OliveDrab { get { return namedColors["olivedrab"]; } }
        public static Color Orange { get { return namedColors["orange"]; } }
        public static Color OrangeRed { get { return namedColors["orangered"]; } }
        public static Color Orchid { get { return namedColors["orchid"]; } }
        public static Color PaleGoldenrod { get { return namedColors["palegoldenrod"]; } }
        public static Color PaleGreen { get { return namedColors["palegreen"]; } }
        public static Color PaleTurquoise { get { return namedColors["paleturquoise"]; } }
        public static Color PaleVioletRed { get { return namedColors["palevioletred"]; } }
        public static Color PapayaWhip { get { return namedColors["papayawhip"]; } }
        public static Color PeachPuff { get { return namedColors["peachpuff"]; } }
        public static Color Peru { get { return namedColors["peru"]; } }
        public static Color Pink { get { return namedColors["pink"]; } }
        public static Color Plum { get { return namedColors["plum"]; } }
        public static Color PowderBlue { get { return namedColors["powderblue"]; } }
        public static Color Purple { get { return namedColors["purple"]; } }
        public static Color RebeccaPurple { get { return namedColors["rebeccapurple"]; } }
        public static Color Red { get { return namedColors["red"]; } }
        public static Color RosyBrown { get { return namedColors["rosybrown"]; } }
        public static Color RoyalBlue { get { return namedColors["royalblue"]; } }
        public static Color SaddleBrown { get { return namedColors["saddlebrown"]; } }
        public static Color Salmon { get { return namedColors["salmon"]; } }
        public static Color SandyBrown { get { return namedColors["sandybrown"]; } }
        public static Color SeaGreen { get { return namedColors["seagreen"]; } }
        public static Color SeaShell { get { return namedColors["seashell"]; } }
        public static Color Sienna { get { return namedColors["sienna"]; } }
        public static Color Silver { get { return namedColors["silver"]; } }
        public static Color SkyBlue { get { return namedColors["skyblue"]; } }
        public static Color SlateBlue { get { return namedColors["slateblue"]; } }
        public static Color SlateGray { get { return namedColors["slategray"]; } }
        public static Color Snow { get { return namedColors["snow"]; } }
        public static Color SpringGreen { get { return namedColors["springgreen"]; } }
        public static Color SteelBlue { get { return namedColors["steelblue"]; } }
        public static Color Tan { get { return namedColors["tan"]; } }
        public static Color Teal { get { return namedColors["teal"]; } }
        public static Color Thistle { get { return namedColors["thistle"]; } }
        public static Color Tomato { get { return namedColors["tomato"]; } }
        public static Color Transparent { get { return namedColors["transparent"]; } }
        public static Color Turquoise { get { return namedColors["turquoise"]; } }
        public static Color Violet { get { return namedColors["violet"]; } }
        public static Color WebGreen { get { return namedColors["webgreen"]; } }
        public static Color WebGray { get { return namedColors["webgray"]; } }
        public static Color WebMaroon { get { return namedColors["webmaroon"]; } }
        public static Color WebPurple { get { return namedColors["webpurple"]; } }
        public static Color Wheat { get { return namedColors["wheat"]; } }
        public static Color White { get { return namedColors["white"]; } }
        public static Color WhiteSmoke { get { return namedColors["whitesmoke"]; } }
        public static Color Yellow { get { return namedColors["yellow"]; } }
        public static Color YellowGreen { get { return namedColors["yellowgreen"]; } }
    }
}
