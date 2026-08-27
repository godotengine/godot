extends RefCounted

# Deterministic styled documents shared by the model and performance checks.
const CJK := "我們正在開設一系列的大學專業課程內容給中小學生讓每一個學生都有機會在將來成為一位真正的軟體工程師從程式設計基礎開始培養資料結構演算法資料庫與網路"
const LATIN := ["editor", "paragraph", "render", "server", "style", "layout", "token", "canvas", "article", "engine"]

var line_count := 0


static func paragraph(p_index: int) -> String:
	var rng := RandomNumberGenerator.new()
	rng.seed = p_index
	var text := ""
	while text.length() < 60:
		if rng.randf() < 0.7:
			var at := rng.randi_range(0, CJK.length() - 8)
			text += CJK.substr(at, rng.randi_range(3, 8))
		else:
			text += " " + LATIN[rng.randi_range(0, LATIN.size() - 1)] + " "
	return text


# Every paragraph has six styled runs; some are list items, centered, quoted
# or carry a line height.
static func styled_paragraph(p_index: int) -> String:
	var text := paragraph(p_index)
	var n := text.length()
	var cut := [n / 7, n * 2 / 7, n * 3 / 7, n * 4 / 7, n * 5 / 7, n * 6 / 7]
	var out := text.substr(0, cut[0]) + "[color=d9480f]" + text.substr(cut[0], cut[1] - cut[0]) + "[/color]"
	out += "[b]" + text.substr(cut[1], cut[2] - cut[1]) + "[/b]"
	out += "[font_size=22]" + text.substr(cut[2], cut[3] - cut[2]) + "[/font_size]"
	out += "[url=https://example.com/" + str(p_index) + "]" + text.substr(cut[3], cut[4] - cut[3]) + "[/url]"
	out += "[bgcolor=fff3bf]" + text.substr(cut[4], cut[5] - cut[4]) + "[/bgcolor]"
	out += "[i][u]" + text.substr(cut[5]) + "[/u][/i]"
	if p_index % 50 == 0:
		out = "[ol]" + out + "[/ol]"
	elif p_index % 70 == 0:
		out = "[center]" + out + "[/center]"
	elif p_index % 90 == 0:
		out = "[quote]" + out + "[/quote]"
	if p_index % 40 == 0:
		out = "[line_height=1.6]" + out + "[/line_height]"
	return out


func make(p_chars: int) -> String:
	var parts := PackedStringArray()
	var total := 0
	line_count = 0
	while total < p_chars:
		parts.append(styled_paragraph(line_count))
		total += paragraph(line_count).length() + 1
		line_count += 1
	return "\n".join(parts)
