#line 1 "hb-ot-shape-complex-use-machine.rl"
/*
* Copyright © 2015  Mozilla Foundation.
* Copyright © 2015  Google, Inc.
*
*  This is part of HarfBuzz, a text shaping library.
*
* Permission is hereby granted, without written agreement and without
* license or royalty fees, to use, copy, modify, and distribute this
* software and its documentation for any purpose, provided that the
* above copyright notice and the following two paragraphs appear in
* all copies of this software.
*
* IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE TO ANY PARTY FOR
* DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
* ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN
* IF THE COPYRIGHT HOLDER HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
* DAMAGE.
*
* THE COPYRIGHT HOLDER SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
* BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
* FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
* ON AN "AS IS" BASIS, AND THE COPYRIGHT HOLDER HAS NO OBLIGATION TO
* PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*
* Mozilla Author(s): Jonathan Kew
* Google Author(s): Behdad Esfahbod
*/

#ifndef HB_OT_SHAPE_COMPLEX_USE_MACHINE_HH
#define HB_OT_SHAPE_COMPLEX_USE_MACHINE_HH

#include "hb.hh"

#include "hb-ot-shape-complex-syllabic.hh"

/* buffer var allocations */
#define use_category() complex_var_u8_category()

#define USE(Cat) use_syllable_machine_ex_##Cat

enum use_syllable_type_t {
	use_independent_cluster,
	use_virama_terminated_cluster,
	use_sakot_terminated_cluster,
	use_standard_cluster,
	use_number_joiner_terminated_cluster,
	use_numeral_cluster,
	use_symbol_cluster,
	use_hieroglyph_cluster,
	use_broken_cluster,
	use_non_cluster,
};


#line 57 "hb-ot-shape-complex-use-machine.hh"
#define use_syllable_machine_ex_B 1u
#define use_syllable_machine_ex_CMAbv 31u
#define use_syllable_machine_ex_CMBlw 32u
#define use_syllable_machine_ex_CS 43u
#define use_syllable_machine_ex_FAbv 24u
#define use_syllable_machine_ex_FBlw 25u
#define use_syllable_machine_ex_FMAbv 45u
#define use_syllable_machine_ex_FMBlw 46u
#define use_syllable_machine_ex_FMPst 47u
#define use_syllable_machine_ex_FPst 26u
#define use_syllable_machine_ex_G 49u
#define use_syllable_machine_ex_GB 5u
#define use_syllable_machine_ex_H 12u
#define use_syllable_machine_ex_HN 13u
#define use_syllable_machine_ex_HVM 44u
#define use_syllable_machine_ex_J 50u
#define use_syllable_machine_ex_MAbv 27u
#define use_syllable_machine_ex_MBlw 28u
#define use_syllable_machine_ex_MPre 30u
#define use_syllable_machine_ex_MPst 29u
#define use_syllable_machine_ex_N 4u
#define use_syllable_machine_ex_O 0u
#define use_syllable_machine_ex_R 18u
#define use_syllable_machine_ex_S 19u
#define use_syllable_machine_ex_SB 51u
#define use_syllable_machine_ex_SE 52u
#define use_syllable_machine_ex_SMAbv 41u
#define use_syllable_machine_ex_SMBlw 42u
#define use_syllable_machine_ex_SUB 11u
#define use_syllable_machine_ex_Sk 48u
#define use_syllable_machine_ex_VAbv 33u
#define use_syllable_machine_ex_VBlw 34u
#define use_syllable_machine_ex_VMAbv 37u
#define use_syllable_machine_ex_VMBlw 38u
#define use_syllable_machine_ex_VMPre 23u
#define use_syllable_machine_ex_VMPst 39u
#define use_syllable_machine_ex_VPre 22u
#define use_syllable_machine_ex_VPst 35u
#define use_syllable_machine_ex_ZWNJ 14u


#line 99 "hb-ot-shape-complex-use-machine.hh"
static const unsigned char _use_syllable_machine_trans_keys[] = {
	1u, 1u, 1u, 1u, 0u, 37u, 5u, 34u,
	5u, 34u, 1u, 1u, 10u, 34u, 11u, 34u,
	12u, 33u, 13u, 33u, 14u, 33u, 31u, 32u,
	32u, 32u, 12u, 34u, 12u, 34u, 12u, 34u,
	1u, 1u, 12u, 34u, 11u, 34u, 11u, 34u,
	11u, 34u, 10u, 34u, 10u, 34u, 10u, 34u,
	5u, 34u, 1u, 34u, 7u, 7u, 3u, 3u,
	5u, 34u, 27u, 28u, 28u, 28u, 5u, 34u,
	10u, 34u, 11u, 34u, 12u, 33u, 13u, 33u,
	14u, 33u, 31u, 32u, 32u, 32u, 12u, 34u,
	12u, 34u, 12u, 34u, 12u, 34u, 11u, 34u,
	11u, 34u, 11u, 34u, 10u, 34u, 10u, 34u,
	10u, 34u, 5u, 34u, 1u, 34u, 1u, 1u,
	3u, 3u, 7u, 7u, 1u, 34u, 5u, 34u,
	27u, 28u, 28u, 28u, 1u, 4u, 36u, 38u,
	35u, 38u, 35u, 37u, 0u
};

static const signed char _use_syllable_machine_char_class[] = {
	0, 1, 2, 2, 3, 4, 2, 2,
	2, 2, 2, 5, 6, 7, 2, 2,
	2, 2, 8, 9, 2, 2, 10, 11,
	12, 13, 14, 15, 16, 17, 18, 19,
	20, 21, 22, 23, 2, 24, 25, 26,
	2, 27, 28, 29, 30, 31, 32, 33,
	34, 35, 36, 37, 38, 0
};

static const short _use_syllable_machine_index_offsets[] = {
	0, 1, 2, 40, 70, 100, 101, 126,
	150, 172, 193, 213, 215, 216, 239, 262,
	285, 286, 309, 333, 357, 381, 406, 431,
	456, 486, 520, 521, 522, 552, 554, 555,
	585, 610, 634, 656, 677, 697, 699, 700,
	723, 746, 769, 792, 816, 840, 864, 889,
	914, 939, 969, 1003, 1004, 1005, 1006, 1040,
	1070, 1072, 1073, 1077, 1080, 1084, 0
};

static const signed char _use_syllable_machine_indicies[] = {
	1, 2, 4, 5, 6, 7, 8, 1,
	9, 10, 11, 12, 13, 14, 15, 16,
	17, 18, 19, 13, 20, 21, 22, 23,
	24, 25, 26, 27, 28, 29, 30, 31,
	32, 33, 34, 35, 9, 36, 6, 37,
	39, 40, 38, 38, 38, 41, 42, 43,
	44, 45, 46, 47, 41, 48, 5, 49,
	50, 51, 52, 53, 54, 55, 38, 38,
	38, 56, 57, 58, 59, 40, 39, 40,
	38, 38, 38, 41, 42, 43, 44, 45,
	46, 47, 41, 48, 49, 49, 50, 51,
	52, 53, 54, 55, 38, 38, 38, 56,
	57, 58, 59, 40, 39, 41, 42, 43,
	44, 45, 38, 38, 38, 38, 38, 38,
	50, 51, 52, 53, 54, 55, 38, 38,
	38, 42, 57, 58, 59, 61, 42, 43,
	44, 45, 38, 38, 38, 38, 38, 38,
	38, 38, 38, 53, 54, 55, 38, 38,
	38, 38, 57, 58, 59, 61, 43, 44,
	45, 38, 38, 38, 38, 38, 38, 38,
	38, 38, 38, 38, 38, 38, 38, 38,
	38, 57, 58, 59, 44, 45, 38, 38,
	38, 38, 38, 38, 38, 38, 38, 38,
	38, 38, 38, 38, 38, 38, 57, 58,
	59, 45, 38, 38, 38, 38, 38, 38,
	38, 38, 38, 38, 38, 38, 38, 38,
	38, 38, 57, 58, 59, 57, 58, 58,
	43, 44, 45, 38, 38, 38, 38, 38,
	38, 38, 38, 38, 53, 54, 55, 38,
	38, 38, 38, 57, 58, 59, 61, 43,
	44, 45, 38, 38, 38, 38, 38, 38,
	38, 38, 38, 38, 54, 55, 38, 38,
	38, 38, 57, 58, 59, 61, 43, 44,
	45, 38, 38, 38, 38, 38, 38, 38,
	38, 38, 38, 38, 55, 38, 38, 38,
	38, 57, 58, 59, 61, 63, 43, 44,
	45, 38, 38, 38, 38, 38, 38, 38,
	38, 38, 38, 38, 38, 38, 38, 38,
	38, 57, 58, 59, 61, 42, 43, 44,
	45, 38, 38, 38, 38, 38, 38, 50,
	51, 52, 53, 54, 55, 38, 38, 38,
	42, 57, 58, 59, 61, 42, 43, 44,
	45, 38, 38, 38, 38, 38, 38, 38,
	51, 52, 53, 54, 55, 38, 38, 38,
	42, 57, 58, 59, 61, 42, 43, 44,
	45, 38, 38, 38, 38, 38, 38, 38,
	38, 52, 53, 54, 55, 38, 38, 38,
	42, 57, 58, 59, 61, 41, 42, 43,
	44, 45, 38, 47, 41, 38, 38, 38,
	50, 51, 52, 53, 54, 55, 38, 38,
	38, 42, 57, 58, 59, 61, 41, 42,
	43, 44, 45, 38, 38, 41, 38, 38,
	38, 50, 51, 52, 53, 54, 55, 38,
	38, 38, 42, 57, 58, 59, 61, 41,
	42, 43, 44, 45, 46, 47, 41, 38,
	38, 38, 50, 51, 52, 53, 54, 55,
	38, 38, 38, 42, 57, 58, 59, 61,
	39, 40, 38, 38, 38, 41, 42, 43,
	44, 45, 46, 47, 41, 48, 38, 49,
	50, 51, 52, 53, 54, 55, 38, 38,
	38, 56, 57, 58, 59, 40, 39, 60,
	60, 60, 60, 60, 60, 60, 60, 60,
	42, 43, 44, 45, 60, 60, 60, 60,
	60, 60, 60, 60, 60, 53, 54, 55,
	60, 60, 60, 60, 57, 58, 59, 61,
	65, 7, 39, 40, 38, 38, 38, 41,
	42, 43, 44, 45, 46, 47, 41, 48,
	5, 49, 50, 51, 52, 53, 54, 55,
	12, 67, 38, 56, 57, 58, 59, 40,
	12, 67, 67, 1, 70, 69, 69, 69,
	13, 14, 15, 16, 17, 18, 19, 13,
	20, 22, 22, 23, 24, 25, 26, 27,
	28, 69, 69, 69, 32, 33, 34, 35,
	70, 13, 14, 15, 16, 17, 69, 69,
	69, 69, 69, 69, 23, 24, 25, 26,
	27, 28, 69, 69, 69, 14, 33, 34,
	35, 71, 14, 15, 16, 17, 69, 69,
	69, 69, 69, 69, 69, 69, 69, 26,
	27, 28, 69, 69, 69, 69, 33, 34,
	35, 71, 15, 16, 17, 69, 69, 69,
	69, 69, 69, 69, 69, 69, 69, 69,
	69, 69, 69, 69, 69, 33, 34, 35,
	16, 17, 69, 69, 69, 69, 69, 69,
	69, 69, 69, 69, 69, 69, 69, 69,
	69, 69, 33, 34, 35, 17, 69, 69,
	69, 69, 69, 69, 69, 69, 69, 69,
	69, 69, 69, 69, 69, 69, 33, 34,
	35, 33, 34, 34, 15, 16, 17, 69,
	69, 69, 69, 69, 69, 69, 69, 69,
	26, 27, 28, 69, 69, 69, 69, 33,
	34, 35, 71, 15, 16, 17, 69, 69,
	69, 69, 69, 69, 69, 69, 69, 69,
	27, 28, 69, 69, 69, 69, 33, 34,
	35, 71, 15, 16, 17, 69, 69, 69,
	69, 69, 69, 69, 69, 69, 69, 69,
	28, 69, 69, 69, 69, 33, 34, 35,
	71, 15, 16, 17, 69, 69, 69, 69,
	69, 69, 69, 69, 69, 69, 69, 69,
	69, 69, 69, 69, 33, 34, 35, 71,
	14, 15, 16, 17, 69, 69, 69, 69,
	69, 69, 23, 24, 25, 26, 27, 28,
	69, 69, 69, 14, 33, 34, 35, 71,
	14, 15, 16, 17, 69, 69, 69, 69,
	69, 69, 69, 24, 25, 26, 27, 28,
	69, 69, 69, 14, 33, 34, 35, 71,
	14, 15, 16, 17, 69, 69, 69, 69,
	69, 69, 69, 69, 25, 26, 27, 28,
	69, 69, 69, 14, 33, 34, 35, 71,
	13, 14, 15, 16, 17, 69, 19, 13,
	69, 69, 69, 23, 24, 25, 26, 27,
	28, 69, 69, 69, 14, 33, 34, 35,
	71, 13, 14, 15, 16, 17, 69, 69,
	13, 69, 69, 69, 23, 24, 25, 26,
	27, 28, 69, 69, 69, 14, 33, 34,
	35, 71, 13, 14, 15, 16, 17, 18,
	19, 13, 69, 69, 69, 23, 24, 25,
	26, 27, 28, 69, 69, 69, 14, 33,
	34, 35, 71, 1, 70, 69, 69, 69,
	13, 14, 15, 16, 17, 18, 19, 13,
	20, 69, 22, 23, 24, 25, 26, 27,
	28, 69, 69, 69, 32, 33, 34, 35,
	70, 1, 69, 69, 69, 69, 69, 69,
	69, 69, 69, 14, 15, 16, 17, 69,
	69, 69, 69, 69, 69, 69, 69, 69,
	26, 27, 28, 69, 69, 69, 69, 33,
	34, 35, 71, 1, 73, 10, 5, 69,
	69, 5, 1, 70, 10, 69, 69, 13,
	14, 15, 16, 17, 18, 19, 13, 20,
	21, 22, 23, 24, 25, 26, 27, 28,
	29, 30, 69, 32, 33, 34, 35, 70,
	1, 70, 69, 69, 69, 13, 14, 15,
	16, 17, 18, 19, 13, 20, 21, 22,
	23, 24, 25, 26, 27, 28, 69, 69,
	69, 32, 33, 34, 35, 70, 29, 30,
	30, 5, 72, 72, 5, 75, 74, 36,
	36, 75, 74, 75, 36, 74, 37, 0
};

static const signed char _use_syllable_machine_index_defaults[] = {
	0, 0, 6, 38, 38, 60, 38, 38,
	38, 38, 38, 38, 38, 38, 38, 38,
	62, 38, 38, 38, 38, 38, 38, 38,
	38, 60, 64, 66, 38, 68, 68, 69,
	69, 69, 69, 69, 69, 69, 69, 69,
	69, 69, 69, 69, 69, 69, 69, 69,
	69, 69, 69, 72, 69, 69, 69, 69,
	69, 69, 72, 74, 74, 74, 0
};

static const signed char _use_syllable_machine_cond_targs[] = {
	2, 31, 42, 2, 2, 3, 2, 26,
	28, 51, 52, 54, 29, 32, 33, 34,
	35, 36, 46, 47, 48, 55, 49, 43,
	44, 45, 39, 40, 41, 56, 57, 58,
	50, 37, 38, 2, 59, 61, 2, 4,
	5, 6, 7, 8, 9, 10, 21, 22,
	23, 24, 18, 19, 20, 13, 14, 15,
	25, 11, 12, 2, 2, 16, 2, 17,
	2, 27, 2, 30, 2, 2, 0, 1,
	2, 53, 2, 60, 0
};

static const signed char _use_syllable_machine_cond_actions[] = {
	1, 2, 2, 0, 5, 0, 6, 0,
	0, 0, 0, 2, 0, 2, 2, 0,
	0, 0, 2, 2, 2, 2, 2, 2,
	2, 2, 2, 2, 2, 0, 0, 0,
	2, 0, 0, 7, 0, 0, 8, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 9, 10, 0, 11, 0,
	12, 0, 13, 0, 14, 15, 0, 0,
	16, 0, 17, 0, 0
};

static const signed char _use_syllable_machine_to_state_actions[] = {
	0, 0, 3, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0
};

static const signed char _use_syllable_machine_from_state_actions[] = {
	0, 0, 4, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0
};

static const signed char _use_syllable_machine_eof_trans[] = {
	1, 1, 4, 39, 39, 61, 39, 39,
	39, 39, 39, 39, 39, 39, 39, 39,
	63, 39, 39, 39, 39, 39, 39, 39,
	39, 61, 65, 67, 39, 69, 69, 70,
	70, 70, 70, 70, 70, 70, 70, 70,
	70, 70, 70, 70, 70, 70, 70, 70,
	70, 70, 70, 73, 70, 70, 70, 70,
	70, 70, 73, 75, 75, 75, 0
};

static const int use_syllable_machine_start = 2;
static const int use_syllable_machine_first_final = 2;
static const int use_syllable_machine_error = -1;

static const int use_syllable_machine_en_main = 2;


#line 59 "hb-ot-shape-complex-use-machine.rl"



#line 176 "hb-ot-shape-complex-use-machine.rl"


#define found_syllable(syllable_type) \
HB_STMT_START { \
	if (0) fprintf (stderr, "syllable %d..%d %s\n", (*ts).second.first, (*te).second.first, #syllable_type); \
		for (unsigned i = (*ts).second.first; i < (*te).second.first; ++i) \
	info[i].syllable() = (syllable_serial << 4) | syllable_type; \
	syllable_serial++; \
	if (unlikely (syllable_serial == 16)) syllable_serial = 1; \
	} HB_STMT_END


template <typename Iter>
struct machine_index_t :
hb_iter_with_fallback_t<machine_index_t<Iter>,
typename Iter::item_t>
{
	machine_index_t (const Iter& it) : it (it) {}
	machine_index_t (const machine_index_t& o) : it (o.it) {}
	
	static constexpr bool is_random_access_iterator = Iter::is_random_access_iterator;
	static constexpr bool is_sorted_iterator = Iter::is_sorted_iterator;
	
	typename Iter::item_t __item__ () const { return *it; }
	typename Iter::item_t __item_at__ (unsigned i) const { return it[i]; }
	unsigned __len__ () const { return it.len (); }
	void __next__ () { ++it; }
	void __forward__ (unsigned n) { it += n; }
	void __prev__ () { --it; }
	void __rewind__ (unsigned n) { it -= n; }
	void operator = (unsigned n)
	{ unsigned index = (*it).first; if (index < n) it += n - index; else if (index > n) it -= index - n; }
	void operator = (const machine_index_t& o) { *this = (*o.it).first; }
	bool operator == (const machine_index_t& o) const { return (*it).first == (*o.it).first; }
	bool operator != (const machine_index_t& o) const { return !(*this == o); }
	
	private:
	Iter it;
};
struct
{
	template <typename Iter,
	hb_requires (hb_is_iterable (Iter))>
	machine_index_t<hb_iter_type<Iter>>
	operator () (Iter&& it) const
	{ return machine_index_t<hb_iter_type<Iter>> (hb_iter (it)); }
}
HB_FUNCOBJ (machine_index);



static bool
not_standard_default_ignorable (const hb_glyph_info_t &i)
{ return !(i.use_category() == USE(O) && _hb_glyph_info_is_default_ignorable (&i)); }

static inline void
find_syllables_use (hb_buffer_t *buffer)
{
	hb_glyph_info_t *info = buffer->info;
	auto p =
	+ hb_iter (info, buffer->len)
	| hb_enumerate
	| hb_filter ([] (const hb_glyph_info_t &i) { return not_standard_default_ignorable (i); },
	hb_second)
	| hb_filter ([&] (const hb_pair_t<unsigned, const hb_glyph_info_t &> p)
	{
		if (p.second.use_category() == USE(ZWNJ))
			for (unsigned i = p.first + 1; i < buffer->len; ++i)
		if (not_standard_default_ignorable (info[i]))
			return !_hb_glyph_info_is_unicode_mark (&info[i]);
		return true;
	})
	| hb_enumerate
	| machine_index
	;
	auto pe = p + p.len ();
	auto eof = +pe;
	auto ts = +p;
	auto te = +p;
	unsigned int act HB_UNUSED;
	int cs;
	
#line 443 "hb-ot-shape-complex-use-machine.hh"
	{
		cs = (int)use_syllable_machine_start;
		ts = 0;
		te = 0;
	}
	
#line 260 "hb-ot-shape-complex-use-machine.rl"
	
	
	unsigned int syllable_serial = 1;
	
#line 455 "hb-ot-shape-complex-use-machine.hh"
	{
		unsigned int _trans = 0;
		const unsigned char * _keys;
		const signed char * _inds;
		int _ic;
		_resume: {}
		if ( p == pe && p != eof )
			goto _out;
		switch ( _use_syllable_machine_from_state_actions[cs] ) {
			case 4:  {
				{
#line 1 "NONE"
					{ts = p;}}
				
#line 470 "hb-ot-shape-complex-use-machine.hh"
				
				
				break; 
			}
		}
		
		if ( p == eof ) {
			if ( _use_syllable_machine_eof_trans[cs] > 0 ) {
				_trans = (unsigned int)_use_syllable_machine_eof_trans[cs] - 1;
			}
		}
		else {
			_keys = ( _use_syllable_machine_trans_keys + ((cs<<1)));
			_inds = ( _use_syllable_machine_indicies + (_use_syllable_machine_index_offsets[cs]));
			
			if ( ((*p).second.second.use_category()) <= 52 ) {
				_ic = (int)_use_syllable_machine_char_class[(int)((*p).second.second.use_category()) - 0];
				if ( _ic <= (int)(*( _keys+1)) && _ic >= (int)(*( _keys)) )
					_trans = (unsigned int)(*( _inds + (int)( _ic - (int)(*( _keys)) ) )); 
				else
					_trans = (unsigned int)_use_syllable_machine_index_defaults[cs];
			}
			else {
				_trans = (unsigned int)_use_syllable_machine_index_defaults[cs];
			}
			
		}
		cs = (int)_use_syllable_machine_cond_targs[_trans];
		
		if ( _use_syllable_machine_cond_actions[_trans] != 0 ) {
			
			switch ( _use_syllable_machine_cond_actions[_trans] ) {
				case 2:  {
					{
#line 1 "NONE"
						{te = p+1;}}
					
#line 508 "hb-ot-shape-complex-use-machine.hh"
					
					
					break; 
				}
				case 5:  {
					{
#line 163 "hb-ot-shape-complex-use-machine.rl"
						{te = p+1;{
#line 163 "hb-ot-shape-complex-use-machine.rl"
								found_syllable (use_independent_cluster); }
						}}
					
#line 521 "hb-ot-shape-complex-use-machine.hh"
					
					
					break; 
				}
				case 9:  {
					{
#line 166 "hb-ot-shape-complex-use-machine.rl"
						{te = p+1;{
#line 166 "hb-ot-shape-complex-use-machine.rl"
								found_syllable (use_standard_cluster); }
						}}
					
#line 534 "hb-ot-shape-complex-use-machine.hh"
					
					
					break; 
				}
				case 7:  {
					{
#line 171 "hb-ot-shape-complex-use-machine.rl"
						{te = p+1;{
#line 171 "hb-ot-shape-complex-use-machine.rl"
								found_syllable (use_broken_cluster); }
						}}
					
#line 547 "hb-ot-shape-complex-use-machine.hh"
					
					
					break; 
				}
				case 6:  {
					{
#line 172 "hb-ot-shape-complex-use-machine.rl"
						{te = p+1;{
#line 172 "hb-ot-shape-complex-use-machine.rl"
								found_syllable (use_non_cluster); }
						}}
					
#line 560 "hb-ot-shape-complex-use-machine.hh"
					
					
					break; 
				}
				case 10:  {
					{
#line 164 "hb-ot-shape-complex-use-machine.rl"
						{te = p;p = p - 1;{
#line 164 "hb-ot-shape-complex-use-machine.rl"
								found_syllable (use_virama_terminated_cluster); }
						}}
					
#line 573 "hb-ot-shape-complex-use-machine.hh"
					
					
					break; 
				}
				case 11:  {
					{
#line 165 "hb-ot-shape-complex-use-machine.rl"
						{te = p;p = p - 1;{
#line 165 "hb-ot-shape-complex-use-machine.rl"
								found_syllable (use_sakot_terminated_cluster); }
						}}
					
#line 586 "hb-ot-shape-complex-use-machine.hh"
					
					
					break; 
				}
				case 8:  {
					{
#line 166 "hb-ot-shape-complex-use-machine.rl"
						{te = p;p = p - 1;{
#line 166 "hb-ot-shape-complex-use-machine.rl"
								found_syllable (use_standard_cluster); }
						}}
					
#line 599 "hb-ot-shape-complex-use-machine.hh"
					
					
					break; 
				}
				case 13:  {
					{
#line 167 "hb-ot-shape-complex-use-machine.rl"
						{te = p;p = p - 1;{
#line 167 "hb-ot-shape-complex-use-machine.rl"
								found_syllable (use_number_joiner_terminated_cluster); }
						}}
					
#line 612 "hb-ot-shape-complex-use-machine.hh"
					
					
					break; 
				}
				case 12:  {
					{
#line 168 "hb-ot-shape-complex-use-machine.rl"
						{te = p;p = p - 1;{
#line 168 "hb-ot-shape-complex-use-machine.rl"
								found_syllable (use_numeral_cluster); }
						}}
					
#line 625 "hb-ot-shape-complex-use-machine.hh"
					
					
					break; 
				}
				case 14:  {
					{
#line 169 "hb-ot-shape-complex-use-machine.rl"
						{te = p;p = p - 1;{
#line 169 "hb-ot-shape-complex-use-machine.rl"
								found_syllable (use_symbol_cluster); }
						}}
					
#line 638 "hb-ot-shape-complex-use-machine.hh"
					
					
					break; 
				}
				case 17:  {
					{
#line 170 "hb-ot-shape-complex-use-machine.rl"
						{te = p;p = p - 1;{
#line 170 "hb-ot-shape-complex-use-machine.rl"
								found_syllable (use_hieroglyph_cluster); }
						}}
					
#line 651 "hb-ot-shape-complex-use-machine.hh"
					
					
					break; 
				}
				case 15:  {
					{
#line 171 "hb-ot-shape-complex-use-machine.rl"
						{te = p;p = p - 1;{
#line 171 "hb-ot-shape-complex-use-machine.rl"
								found_syllable (use_broken_cluster); }
						}}
					
#line 664 "hb-ot-shape-complex-use-machine.hh"
					
					
					break; 
				}
				case 16:  {
					{
#line 172 "hb-ot-shape-complex-use-machine.rl"
						{te = p;p = p - 1;{
#line 172 "hb-ot-shape-complex-use-machine.rl"
								found_syllable (use_non_cluster); }
						}}
					
#line 677 "hb-ot-shape-complex-use-machine.hh"
					
					
					break; 
				}
				case 1:  {
					{
#line 171 "hb-ot-shape-complex-use-machine.rl"
						{p = ((te))-1;
							{
#line 171 "hb-ot-shape-complex-use-machine.rl"
								found_syllable (use_broken_cluster); }
						}}
					
#line 691 "hb-ot-shape-complex-use-machine.hh"
					
					
					break; 
				}
			}
			
		}
		
		if ( p == eof ) {
			if ( cs >= 2 )
				goto _out;
		}
		else {
			switch ( _use_syllable_machine_to_state_actions[cs] ) {
				case 3:  {
					{
#line 1 "NONE"
						{ts = 0;}}
					
#line 711 "hb-ot-shape-complex-use-machine.hh"
					
					
					break; 
				}
			}
			
			p += 1;
			goto _resume;
		}
		_out: {}
	}
	
#line 265 "hb-ot-shape-complex-use-machine.rl"
	
}

#undef found_syllable

#endif /* HB_OT_SHAPE_COMPLEX_USE_MACHINE_HH */
