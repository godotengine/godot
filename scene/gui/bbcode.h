
#include "core/math/vector2.h"
#include "rich_text_effect.h"
#include "scene/gui/rich_text_label.h"

class BbCodeParser : Control {
	GDCLASS(BbCodeParser, Control);
	//We use the classes from RTL here, to have a quick dependency overview
	//Later, we should move them over here, if we no longer have dependencies in the RTL
	using Line = RichTextLabel::Line;
	using Item = RichTextLabel::Item;
	using ItemFrame = RichTextLabel::ItemFrame;
	using ProcessMode = RichTextLabel::ProcessMode;
	using Align = RichTextLabel::Align;
	using ItemType = RichTextLabel::ItemType;
	using ItemAlign = RichTextLabel::ItemAlign;
	using ItemIndent = RichTextLabel::ItemIndent;
	using ItemFont = RichTextLabel::ItemFont;
	using ItemText = RichTextLabel::ItemText;
	using ItemFade = RichTextLabel::ItemFade;
	using ItemFX = RichTextLabel::ItemFX;
	using ItemMeta = RichTextLabel::ItemMeta;
	using ItemColor = RichTextLabel::ItemColor;
	using Selection = RichTextLabel::Selection;
	using ItemCustomFX = RichTextLabel::ItemCustomFX;
	using ItemShake = RichTextLabel::ItemShake;
	using ItemWave = RichTextLabel::ItemWave;
	using ItemTornado = RichTextLabel::ItemTornado;
	using ItemRainbow = RichTextLabel::ItemRainbow;
	using ItemImage = RichTextLabel::ItemImage;
	using ItemTable = RichTextLabel::ItemTable;

private:
	//localy declared vars in _process_line
	Line &l = Line();
	Item *it = nullptr;
	int line_ofs = 0;
	int margin = 0;
	Align align = Align::ALIGN_LEFT;
	int line = 0;
	int spaces = 0;
	int height = 0;
	int wofs = 0;
	int spaces_size = 0;
	int align_ofs = 0;
	int begin = 0;
	Ref<Font> cfont;

	int line_height = 0;
	int line_ascent = 0;
	int line_descent = 0;

	int backtrack = 0; // for dynamic hidden content.
	int nonblank_line_count = 0; //number of nonblank lines as counted during PROCESS_DRAW
	Variant meta;

	Color selection_fg;
	Color selection_bg;

	int rchar = 0;
	int lh = 0;
	bool line_is_blank = true;
	bool line_wrapped = false;
	int fh = 0;

private:
	//Makros as functions
	bool _new_line();
	bool _ensure_width(int m_width);
	bool _advance(int m_width); //rename?
	void _check_height(int m_height); //rename?
	bool _y_range_visible(int m_top, int m_height);

private:
	//Copied functions
	Item *_get_next_item(Item *p_item, bool p_free = false) const;
	Ref<Font> _find_font(Item *p_item);
	int _find_margin(Item *p_item, const Ref<Font> &p_base_font);
	Align _find_align(Item *p_item);
	void _fetch_item_fx_stack(Item *p_item, Vector<ItemFX *> &r_stack);
	Color _find_color(Item *p_item, const Color &p_default_color);
	bool _find_underline(Item *p_item);
	bool _find_strikethrough(Item *p_item);
	bool _find_meta(Item *p_item, Variant *r_meta, ItemMeta **r_item = nullptr);

	int _process_line(ItemFrame *p_frame, const Vector2 &p_ofs, int &y, int p_width, int p_line, ProcessMode p_mode, const Ref<Font> &p_base_font, const Color &p_base_color, const Color &p_font_color_shadow, bool p_shadow_as_outline, const Point2 &shadow_ofs, const Point2i &p_click_pos = Point2i(), Item **r_click_item = nullptr, int *r_click_char = nullptr, bool *r_outside = nullptr, int p_char_count = 0);

	//TODOS:
	// split process_modes into different functions
	// get rid of the Process Modes

public:
	void start_process(ItemFrame *_p_frame, const Vector2 &_p_ofs, const int _p_height, const int _p_width, const int _p_line, const Ref<Font> &p_base_font);
	void process();
	void process_cache();
	void process_draw();
	void process_pointer();

	int get_data();

public:
	//Duplicated vars
	int tab_size = 4;
	RichTextLabel::Align default_align = RichTextLabel::Align::ALIGN_LEFT;
	bool underline_meta = false;
	bool override_selected_font_color = false;
	Selection selection;
	int visible_characters = -1;

public:
	//LineParser arguments
	ItemFrame *p_frame = nullptr;
	Vector2 &p_ofs = Vector2();
	int p_height = 0; //is altered ONLY when a new line is added //was y
	int p_width = 0;
	int p_line = 0;
	RichTextLabel::ProcessMode p_mode = ProcessMode::PROCESS_DRAW;
	Ref<Font> &p_base_font = Ref<Font>();
	Color &p_base_color = Color();
	Color &p_font_color_shadow = Color();
	bool p_shadow_as_outline = false;
	Point2 &shadow_ofs = Point2();
	Point2i &p_click_pos = Point2i();
	RichTextLabel::Item **r_click_item = nullptr;
	int *r_click_char = nullptr;
	bool *r_outside = nullptr;
	int p_char_count = 0;

	BbCodeParser();
	~BbCodeParser();
};
