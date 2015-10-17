#ifndef KEY_MAPPING_HAIKU_H
#define KEY_MAPPING_HAIKU_H

class KeyMappingHaiku
{
	KeyMappingHaiku() {};

public:
	static unsigned int get_keysym(int32 raw_char, int32 key);
	static unsigned int get_modifier_keysym(int32 key);
};

#endif
