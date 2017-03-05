/* Thirdparty code presumably from http://wss.co.uk/pinknoise/yuv2rgb/ */
/* FIXME: Move to thirdparty dir */

#ifndef YUV2RGB_H
#define YUV2RGB_H

#include "typedefs.h"

static const uint32_t tables[256*3] = {
	/* y_table */
	0x7FFFFFEDU,
	0x7FFFFFEFU,
	0x7FFFFFF0U,
	0x7FFFFFF1U,
	0x7FFFFFF2U,
	0x7FFFFFF3U,
	0x7FFFFFF4U,
	0x7FFFFFF6U,
	0x7FFFFFF7U,
	0x7FFFFFF8U,
	0x7FFFFFF9U,
	0x7FFFFFFAU,
	0x7FFFFFFBU,
	0x7FFFFFFDU,
	0x7FFFFFFEU,
	0x7FFFFFFFU,
	0x80000000U,
	0x80400801U,
	0x80A01002U,
	0x80E01803U,
	0x81202805U,
	0x81803006U,
	0x81C03807U,
	0x82004008U,
	0x82604809U,
	0x82A0500AU,
	0x82E0600CU,
	0x8340680DU,
	0x8380700EU,
	0x83C0780FU,
	0x84208010U,
	0x84608811U,
	0x84A09813U,
	0x8500A014U,
	0x8540A815U,
	0x8580B016U,
	0x85E0B817U,
	0x8620C018U,
	0x8660D01AU,
	0x86C0D81BU,
	0x8700E01CU,
	0x8740E81DU,
	0x87A0F01EU,
	0x87E0F81FU,
	0x88210821U,
	0x88811022U,
	0x88C11823U,
	0x89012024U,
	0x89412825U,
	0x89A13026U,
	0x89E14028U,
	0x8A214829U,
	0x8A81502AU,
	0x8AC1582BU,
	0x8B01602CU,
	0x8B61682DU,
	0x8BA1782FU,
	0x8BE18030U,
	0x8C418831U,
	0x8C819032U,
	0x8CC19833U,
	0x8D21A034U,
	0x8D61B036U,
	0x8DA1B837U,
	0x8E01C038U,
	0x8E41C839U,
	0x8E81D03AU,
	0x8EE1D83BU,
	0x8F21E83DU,
	0x8F61F03EU,
	0x8FC1F83FU,
	0x90020040U,
	0x90420841U,
	0x90A21042U,
	0x90E22044U,
	0x91222845U,
	0x91823046U,
	0x91C23847U,
	0x92024048U,
	0x92624849U,
	0x92A2504AU,
	0x92E2604CU,
	0x9342684DU,
	0x9382704EU,
	0x93C2784FU,
	0x94228050U,
	0x94628851U,
	0x94A29853U,
	0x9502A054U,
	0x9542A855U,
	0x9582B056U,
	0x95E2B857U,
	0x9622C058U,
	0x9662D05AU,
	0x96C2D85BU,
	0x9702E05CU,
	0x9742E85DU,
	0x97A2F05EU,
	0x97E2F85FU,
	0x98230861U,
	0x98831062U,
	0x98C31863U,
	0x99032064U,
	0x99632865U,
	0x99A33066U,
	0x99E34068U,
	0x9A434869U,
	0x9A83506AU,
	0x9AC3586BU,
	0x9B23606CU,
	0x9B63686DU,
	0x9BA3786FU,
	0x9BE38070U,
	0x9C438871U,
	0x9C839072U,
	0x9CC39873U,
	0x9D23A074U,
	0x9D63B076U,
	0x9DA3B877U,
	0x9E03C078U,
	0x9E43C879U,
	0x9E83D07AU,
	0x9EE3D87BU,
	0x9F23E87DU,
	0x9F63F07EU,
	0x9FC3F87FU,
	0xA0040080U,
	0xA0440881U,
	0xA0A41082U,
	0xA0E42084U,
	0xA1242885U,
	0xA1843086U,
	0xA1C43887U,
	0xA2044088U,
	0xA2644889U,
	0xA2A4588BU,
	0xA2E4608CU,
	0xA344688DU,
	0xA384708EU,
	0xA3C4788FU,
	0xA4248090U,
	0xA4649092U,
	0xA4A49893U,
	0xA504A094U,
	0xA544A895U,
	0xA584B096U,
	0xA5E4B897U,
	0xA624C098U,
	0xA664D09AU,
	0xA6C4D89BU,
	0xA704E09CU,
	0xA744E89DU,
	0xA7A4F09EU,
	0xA7E4F89FU,
	0xA82508A1U,
	0xA88510A2U,
	0xA8C518A3U,
	0xA90520A4U,
	0xA96528A5U,
	0xA9A530A6U,
	0xA9E540A8U,
	0xAA4548A9U,
	0xAA8550AAU,
	0xAAC558ABU,
	0xAB2560ACU,
	0xAB6568ADU,
	0xABA578AFU,
	0xAC0580B0U,
	0xAC4588B1U,
	0xAC8590B2U,
	0xACE598B3U,
	0xAD25A0B4U,
	0xAD65B0B6U,
	0xADA5B8B7U,
	0xAE05C0B8U,
	0xAE45C8B9U,
	0xAE85D0BAU,
	0xAEE5D8BBU,
	0xAF25E8BDU,
	0xAF65F0BEU,
	0xAFC5F8BFU,
	0xB00600C0U,
	0xB04608C1U,
	0xB0A610C2U,
	0xB0E620C4U,
	0xB12628C5U,
	0xB18630C6U,
	0xB1C638C7U,
	0xB20640C8U,
	0xB26648C9U,
	0xB2A658CBU,
	0xB2E660CCU,
	0xB34668CDU,
	0xB38670CEU,
	0xB3C678CFU,
	0xB42680D0U,
	0xB46690D2U,
	0xB4A698D3U,
	0xB506A0D4U,
	0xB546A8D5U,
	0xB586B0D6U,
	0xB5E6B8D7U,
	0xB626C8D9U,
	0xB666D0DAU,
	0xB6C6D8DBU,
	0xB706E0DCU,
	0xB746E8DDU,
	0xB7A6F0DEU,
	0xB7E6F8DFU,
	0xB82708E1U,
	0xB88710E2U,
	0xB8C718E3U,
	0xB90720E4U,
	0xB96728E5U,
	0xB9A730E6U,
	0xB9E740E8U,
	0xBA4748E9U,
	0xBA8750EAU,
	0xBAC758EBU,
	0xBB2760ECU,
	0xBB6768EDU,
	0xBBA778EFU,
	0xBC0780F0U,
	0xBC4788F1U,
	0xBC8790F2U,
	0xBCE798F3U,
	0xBD27A0F4U,
	0xBD67B0F6U,
	0xBDC7B8F7U,
	0xBE07C0F8U,
	0xBE47C8F9U,
	0xBEA7D0FAU,
	0xBEE7D8FBU,
	0xBF27E8FDU,
	0xBF87F0FEU,
	0xBFC7F8FFU,
	0xC0080100U,
	0xC0480901U,
	0xC0A81102U,
	0xC0E82104U,
	0xC0E82104U,
	0xC0E82104U,
	0xC0E82104U,
	0xC0E82104U,
	0xC0E82104U,
	0xC0E82104U,
	0xC0E82104U,
	0xC0E82104U,
	0xC0E82104U,
	0xC0E82104U,
	0xC0E82104U,
	0xC0E82104U,
	0xC0E82104U,
	0xC0E82104U,
	0xC0E82104U,
	0xC0E82104U,
	/* u_table */
	0x0C400103U,
	0x0C200105U,
	0x0C200107U,
	0x0C000109U,
	0x0BE0010BU,
	0x0BC0010DU,
	0x0BA0010FU,
	0x0BA00111U,
	0x0B800113U,
	0x0B600115U,
	0x0B400117U,
	0x0B400119U,
	0x0B20011BU,
	0x0B00011DU,
	0x0AE0011FU,
	0x0AE00121U,
	0x0AC00123U,
	0x0AA00125U,
	0x0A800127U,
	0x0A600129U,
	0x0A60012BU,
	0x0A40012DU,
	0x0A20012FU,
	0x0A000131U,
	0x0A000132U,
	0x09E00134U,
	0x09C00136U,
	0x09A00138U,
	0x09A0013AU,
	0x0980013CU,
	0x0960013EU,
	0x09400140U,
	0x09400142U,
	0x09200144U,
	0x09000146U,
	0x08E00148U,
	0x08C0014AU,
	0x08C0014CU,
	0x08A0014EU,
	0x08800150U,
	0x08600152U,
	0x08600154U,
	0x08400156U,
	0x08200158U,
	0x0800015AU,
	0x0800015CU,
	0x07E0015EU,
	0x07C00160U,
	0x07A00162U,
	0x07A00164U,
	0x07800166U,
	0x07600168U,
	0x0740016AU,
	0x0720016CU,
	0x0720016EU,
	0x07000170U,
	0x06E00172U,
	0x06C00174U,
	0x06C00176U,
	0x06A00178U,
	0x0680017AU,
	0x0660017CU,
	0x0660017EU,
	0x06400180U,
	0x06200182U,
	0x06000184U,
	0x05E00185U,
	0x05E00187U,
	0x05C00189U,
	0x05A0018BU,
	0x0580018DU,
	0x0580018FU,
	0x05600191U,
	0x05400193U,
	0x05200195U,
	0x05200197U,
	0x05000199U,
	0x04E0019BU,
	0x04C0019DU,
	0x04C0019FU,
	0x04A001A1U,
	0x048001A3U,
	0x046001A5U,
	0x044001A7U,
	0x044001A9U,
	0x042001ABU,
	0x040001ADU,
	0x03E001AFU,
	0x03E001B1U,
	0x03C001B3U,
	0x03A001B5U,
	0x038001B7U,
	0x038001B9U,
	0x036001BBU,
	0x034001BDU,
	0x032001BFU,
	0x032001C1U,
	0x030001C3U,
	0x02E001C5U,
	0x02C001C7U,
	0x02A001C9U,
	0x02A001CBU,
	0x028001CDU,
	0x026001CFU,
	0x024001D1U,
	0x024001D3U,
	0x022001D5U,
	0x020001D7U,
	0x01E001D8U,
	0x01E001DAU,
	0x01C001DCU,
	0x01A001DEU,
	0x018001E0U,
	0x016001E2U,
	0x016001E4U,
	0x014001E6U,
	0x012001E8U,
	0x010001EAU,
	0x010001ECU,
	0x00E001EEU,
	0x00C001F0U,
	0x00A001F2U,
	0x00A001F4U,
	0x008001F6U,
	0x006001F8U,
	0x004001FAU,
	0x004001FCU,
	0x002001FEU,
	0x00000200U,
	0xFFE00202U,
	0xFFC00204U,
	0xFFC00206U,
	0xFFA00208U,
	0xFF80020AU,
	0xFF60020CU,
	0xFF60020EU,
	0xFF400210U,
	0xFF200212U,
	0xFF000214U,
	0xFF000216U,
	0xFEE00218U,
	0xFEC0021AU,
	0xFEA0021CU,
	0xFEA0021EU,
	0xFE800220U,
	0xFE600222U,
	0xFE400224U,
	0xFE200226U,
	0xFE200228U,
	0xFE000229U,
	0xFDE0022BU,
	0xFDC0022DU,
	0xFDC0022FU,
	0xFDA00231U,
	0xFD800233U,
	0xFD600235U,
	0xFD600237U,
	0xFD400239U,
	0xFD20023BU,
	0xFD00023DU,
	0xFCE0023FU,
	0xFCE00241U,
	0xFCC00243U,
	0xFCA00245U,
	0xFC800247U,
	0xFC800249U,
	0xFC60024BU,
	0xFC40024DU,
	0xFC20024FU,
	0xFC200251U,
	0xFC000253U,
	0xFBE00255U,
	0xFBC00257U,
	0xFBC00259U,
	0xFBA0025BU,
	0xFB80025DU,
	0xFB60025FU,
	0xFB400261U,
	0xFB400263U,
	0xFB200265U,
	0xFB000267U,
	0xFAE00269U,
	0xFAE0026BU,
	0xFAC0026DU,
	0xFAA0026FU,
	0xFA800271U,
	0xFA800273U,
	0xFA600275U,
	0xFA400277U,
	0xFA200279U,
	0xFA20027BU,
	0xFA00027CU,
	0xF9E0027EU,
	0xF9C00280U,
	0xF9A00282U,
	0xF9A00284U,
	0xF9800286U,
	0xF9600288U,
	0xF940028AU,
	0xF940028CU,
	0xF920028EU,
	0xF9000290U,
	0xF8E00292U,
	0xF8E00294U,
	0xF8C00296U,
	0xF8A00298U,
	0xF880029AU,
	0xF860029CU,
	0xF860029EU,
	0xF84002A0U,
	0xF82002A2U,
	0xF80002A4U,
	0xF80002A6U,
	0xF7E002A8U,
	0xF7C002AAU,
	0xF7A002ACU,
	0xF7A002AEU,
	0xF78002B0U,
	0xF76002B2U,
	0xF74002B4U,
	0xF74002B6U,
	0xF72002B8U,
	0xF70002BAU,
	0xF6E002BCU,
	0xF6C002BEU,
	0xF6C002C0U,
	0xF6A002C2U,
	0xF68002C4U,
	0xF66002C6U,
	0xF66002C8U,
	0xF64002CAU,
	0xF62002CCU,
	0xF60002CEU,
	0xF60002CFU,
	0xF5E002D1U,
	0xF5C002D3U,
	0xF5A002D5U,
	0xF5A002D7U,
	0xF58002D9U,
	0xF56002DBU,
	0xF54002DDU,
	0xF52002DFU,
	0xF52002E1U,
	0xF50002E3U,
	0xF4E002E5U,
	0xF4C002E7U,
	0xF4C002E9U,
	0xF4A002EBU,
	0xF48002EDU,
	0xF46002EFU,
	0xF46002F1U,
	0xF44002F3U,
	0xF42002F5U,
	0xF40002F7U,
	0xF3E002F9U,
	0xF3E002FBU,
	/* v_table */
	0x1A09A000U,
	0x19E9A800U,
	0x19A9B800U,
	0x1969C800U,
	0x1949D000U,
	0x1909E000U,
	0x18C9E800U,
	0x18A9F800U,
	0x186A0000U,
	0x182A1000U,
	0x180A2000U,
	0x17CA2800U,
	0x17AA3800U,
	0x176A4000U,
	0x172A5000U,
	0x170A6000U,
	0x16CA6800U,
	0x168A7800U,
	0x166A8000U,
	0x162A9000U,
	0x160AA000U,
	0x15CAA800U,
	0x158AB800U,
	0x156AC000U,
	0x152AD000U,
	0x14EAE000U,
	0x14CAE800U,
	0x148AF800U,
	0x146B0000U,
	0x142B1000U,
	0x13EB2000U,
	0x13CB2800U,
	0x138B3800U,
	0x134B4000U,
	0x132B5000U,
	0x12EB6000U,
	0x12CB6800U,
	0x128B7800U,
	0x124B8000U,
	0x122B9000U,
	0x11EBA000U,
	0x11ABA800U,
	0x118BB800U,
	0x114BC000U,
	0x112BD000U,
	0x10EBE000U,
	0x10ABE800U,
	0x108BF800U,
	0x104C0000U,
	0x100C1000U,
	0x0FEC2000U,
	0x0FAC2800U,
	0x0F8C3800U,
	0x0F4C4000U,
	0x0F0C5000U,
	0x0EEC5800U,
	0x0EAC6800U,
	0x0E6C7800U,
	0x0E4C8000U,
	0x0E0C9000U,
	0x0DEC9800U,
	0x0DACA800U,
	0x0D6CB800U,
	0x0D4CC000U,
	0x0D0CD000U,
	0x0CCCD800U,
	0x0CACE800U,
	0x0C6CF800U,
	0x0C4D0000U,
	0x0C0D1000U,
	0x0BCD1800U,
	0x0BAD2800U,
	0x0B6D3800U,
	0x0B2D4000U,
	0x0B0D5000U,
	0x0ACD5800U,
	0x0AAD6800U,
	0x0A6D7800U,
	0x0A2D8000U,
	0x0A0D9000U,
	0x09CD9800U,
	0x098DA800U,
	0x096DB800U,
	0x092DC000U,
	0x090DD000U,
	0x08CDD800U,
	0x088DE800U,
	0x086DF800U,
	0x082E0000U,
	0x07EE1000U,
	0x07CE1800U,
	0x078E2800U,
	0x076E3800U,
	0x072E4000U,
	0x06EE5000U,
	0x06CE5800U,
	0x068E6800U,
	0x064E7800U,
	0x062E8000U,
	0x05EE9000U,
	0x05CE9800U,
	0x058EA800U,
	0x054EB800U,
	0x052EC000U,
	0x04EED000U,
	0x04AED800U,
	0x048EE800U,
	0x044EF000U,
	0x042F0000U,
	0x03EF1000U,
	0x03AF1800U,
	0x038F2800U,
	0x034F3000U,
	0x030F4000U,
	0x02EF5000U,
	0x02AF5800U,
	0x028F6800U,
	0x024F7000U,
	0x020F8000U,
	0x01EF9000U,
	0x01AF9800U,
	0x016FA800U,
	0x014FB000U,
	0x010FC000U,
	0x00EFD000U,
	0x00AFD800U,
	0x006FE800U,
	0x004FF000U,
	0x00100000U,
	0xFFD01000U,
	0xFFB01800U,
	0xFF702800U,
	0xFF303000U,
	0xFF104000U,
	0xFED05000U,
	0xFEB05800U,
	0xFE706800U,
	0xFE307000U,
	0xFE108000U,
	0xFDD09000U,
	0xFD909800U,
	0xFD70A800U,
	0xFD30B000U,
	0xFD10C000U,
	0xFCD0D000U,
	0xFC90D800U,
	0xFC70E800U,
	0xFC30F000U,
	0xFBF10000U,
	0xFBD11000U,
	0xFB911800U,
	0xFB712800U,
	0xFB313000U,
	0xFAF14000U,
	0xFAD14800U,
	0xFA915800U,
	0xFA516800U,
	0xFA317000U,
	0xF9F18000U,
	0xF9D18800U,
	0xF9919800U,
	0xF951A800U,
	0xF931B000U,
	0xF8F1C000U,
	0xF8B1C800U,
	0xF891D800U,
	0xF851E800U,
	0xF831F000U,
	0xF7F20000U,
	0xF7B20800U,
	0xF7921800U,
	0xF7522800U,
	0xF7123000U,
	0xF6F24000U,
	0xF6B24800U,
	0xF6925800U,
	0xF6526800U,
	0xF6127000U,
	0xF5F28000U,
	0xF5B28800U,
	0xF5729800U,
	0xF552A800U,
	0xF512B000U,
	0xF4F2C000U,
	0xF4B2C800U,
	0xF472D800U,
	0xF452E800U,
	0xF412F000U,
	0xF3D30000U,
	0xF3B30800U,
	0xF3731800U,
	0xF3532800U,
	0xF3133000U,
	0xF2D34000U,
	0xF2B34800U,
	0xF2735800U,
	0xF2336800U,
	0xF2137000U,
	0xF1D38000U,
	0xF1B38800U,
	0xF1739800U,
	0xF133A800U,
	0xF113B000U,
	0xF0D3C000U,
	0xF093C800U,
	0xF073D800U,
	0xF033E000U,
	0xF013F000U,
	0xEFD40000U,
	0xEF940800U,
	0xEF741800U,
	0xEF342000U,
	0xEEF43000U,
	0xEED44000U,
	0xEE944800U,
	0xEE745800U,
	0xEE346000U,
	0xEDF47000U,
	0xEDD48000U,
	0xED948800U,
	0xED549800U,
	0xED34A000U,
	0xECF4B000U,
	0xECD4C000U,
	0xEC94C800U,
	0xEC54D800U,
	0xEC34E000U,
	0xEBF4F000U,
	0xEBB50000U,
	0xEB950800U,
	0xEB551800U,
	0xEB352000U,
	0xEAF53000U,
	0xEAB54000U,
	0xEA954800U,
	0xEA555800U,
	0xEA156000U,
	0xE9F57000U,
	0xE9B58000U,
	0xE9958800U,
	0xE9559800U,
	0xE915A000U,
	0xE8F5B000U,
	0xE8B5C000U,
	0xE875C800U,
	0xE855D800U,
	0xE815E000U,
	0xE7F5F000U,
	0xE7B60000U,
	0xE7760800U,
	0xE7561800U,
	0xE7162000U,
	0xE6D63000U,
	0xE6B64000U,
	0xE6764800U,
	0xE6365800U
};

#define FLAGS 0x40080100
#define READUV(U,V) (tables[256 + (U)] + tables[512 + (V)])
#define READY(Y)    tables[Y]
#define FIXUP(Y)                 \
do {                             \
    int tmp = (Y) & FLAGS;       \
    if (tmp != 0)                \
    {                            \
	tmp  -= tmp>>8;          \
	(Y)  |= tmp;             \
	tmp   = FLAGS & ~(Y>>1); \
	(Y)  += tmp>>8;          \
    }                            \
} while (0 == 1)

#define STORE(Y,DSTPTR)         \
do {                            \
    *(DSTPTR)++ = (Y);          \
    *(DSTPTR)++ = (Y)>>22;      \
    *(DSTPTR)++ = (Y)>>11;      \
    *(DSTPTR)++ = 255;            \
} while (0 == 1)

static void yuv422_2_rgb8888(uint8_t  *dst_ptr,
		const uint8_t  *y_ptr,
		const uint8_t  *u_ptr,
		const uint8_t  *v_ptr,
		      int32_t   width,
		      int32_t   height,
		      int32_t   y_span,
		      int32_t   uv_span,
		      int32_t   dst_span,
		      int32_t   dither)
{
    height -= 1;
    while (height > 0)
    {
	height -= width<<16;
	height += 1<<16;
	while (height < 0)
	{
	    /* Do top row pair */
	    uint32_t uv, y0, y1;

	    uv  = READUV(*u_ptr++,*v_ptr++);
	    y0  = uv + READY(*y_ptr++);
	    y1  = uv + READY(*y_ptr++);
	    FIXUP(y0);
	    FIXUP(y1);
	    STORE(y0, dst_ptr);
	    STORE(y1, dst_ptr);
	    height += (2<<16);
	}
	if ((height>>16) == 0)
	{
	    /* Trailing top row pix */
	    uint32_t uv, y0;

	    uv = READUV(*u_ptr,*v_ptr);
	    y0 = uv + READY(*y_ptr++);
	    FIXUP(y0);
	    STORE(y0, dst_ptr);
	}
	dst_ptr += dst_span-width*4;
	y_ptr   += y_span-width;
	u_ptr   += uv_span-(width>>1);
	v_ptr   += uv_span-(width>>1);
	height = (height<<16)>>16;
	height -= 1;
	if (height == 0)
	    break;
	height -= width<<16;
	height += 1<<16;
	while (height < 0)
	{
	    /* Do second row pair */
	    uint32_t uv, y0, y1;

	    uv  = READUV(*u_ptr++,*v_ptr++);
	    y0  = uv + READY(*y_ptr++);
	    y1  = uv + READY(*y_ptr++);
	    FIXUP(y0);
	    FIXUP(y1);
	    STORE(y0, dst_ptr);
	    STORE(y1, dst_ptr);
	    height += (2<<16);
	}
	if ((height>>16) == 0)
	{
	    /* Trailing bottom row pix */
	    uint32_t uv, y0;

	    uv = READUV(*u_ptr,*v_ptr);
	    y0 = uv + READY(*y_ptr++);
	    FIXUP(y0);
	    STORE(y0, dst_ptr);
	}
	dst_ptr += dst_span-width*4;
	y_ptr   += y_span-width;
	u_ptr   += uv_span-(width>>1);
	v_ptr   += uv_span-(width>>1);
	height = (height<<16)>>16;
	height -= 1;
    }
}


#undef FLAGS
#undef READUV
#undef READY
#undef FIXUP
#undef STORE


#define FLAGS 0x40080100
#define READUV(U,V) (tables[256 + (U)] + tables[512 + (V)])
#define READY(Y)    tables[Y]
#define FIXUP(Y)                 \
do {                             \
    int tmp = (Y) & FLAGS;       \
    if (tmp != 0)                \
    {                            \
	tmp  -= tmp>>8;          \
	(Y)  |= tmp;             \
	tmp   = FLAGS & ~(Y>>1); \
	(Y)  += tmp>>8;          \
    }                            \
} while (0 == 1)

#define STORE(Y,DSTPTR)     \
do {                        \
    (DSTPTR) = 0xFF000000 | (Y & 0xFF) | (0xFF00 & (Y>>14)) | (0xFF0000 & (Y<<5));\
} while (0 == 1)

static void yuv420_2_rgb8888(uint8_t  *dst_ptr_,
		const uint8_t  *y_ptr,
		const uint8_t  *u_ptr,
		const uint8_t  *v_ptr,
		      int32_t   width,
		      int32_t   height,
		      int32_t   y_span,
		      int32_t   uv_span,
		      int32_t   dst_span,
		      int32_t   dither)
{
    uint32_t *dst_ptr = (uint32_t *)(void *)dst_ptr_;
    dst_span >>= 2;

    height -= 1;
    while (height > 0)
    {
	height -= width<<16;
	height += 1<<16;
	while (height < 0)
	{
	    /* Do 2 column pairs */
	    uint32_t uv, y0, y1;

	    uv  = READUV(*u_ptr++,*v_ptr++);
	    y1  = uv + READY(y_ptr[y_span]);
	    y0  = uv + READY(*y_ptr++);
	    FIXUP(y1);
	    FIXUP(y0);
	    STORE(y1, dst_ptr[dst_span]);
	    STORE(y0, *dst_ptr++);
	    y1  = uv + READY(y_ptr[y_span]);
	    y0  = uv + READY(*y_ptr++);
	    FIXUP(y1);
	    FIXUP(y0);
	    STORE(y1, dst_ptr[dst_span]);
	    STORE(y0, *dst_ptr++);
	    height += (2<<16);
	}
	if ((height>>16) == 0)
	{
	    /* Trailing column pair */
	    uint32_t uv, y0, y1;

	    uv = READUV(*u_ptr,*v_ptr);
	    y1 = uv + READY(y_ptr[y_span]);
	    y0 = uv + READY(*y_ptr++);
	    FIXUP(y1);
	    FIXUP(y0);
	    STORE(y0, dst_ptr[dst_span]);
	    STORE(y1, *dst_ptr++);
	}
	dst_ptr += dst_span*2-width;
	y_ptr   += y_span*2-width;
	u_ptr   += uv_span-(width>>1);
	v_ptr   += uv_span-(width>>1);
	height = (height<<16)>>16;
	height -= 2;
    }
    if (height == 0)
    {
	/* Trail row */
	height -= width<<16;
	height += 1<<16;
	while (height < 0)
	{
	    /* Do a row pair */
	    uint32_t uv, y0, y1;

	    uv  = READUV(*u_ptr++,*v_ptr++);
	    y1  = uv + READY(*y_ptr++);
	    y0  = uv + READY(*y_ptr++);
	    FIXUP(y1);
	    FIXUP(y0);
	    STORE(y1, *dst_ptr++);
	    STORE(y0, *dst_ptr++);
	    height += (2<<16);
	}
	if ((height>>16) == 0)
	{
	    /* Trailing pix */
	    uint32_t uv, y0;

	    uv = READUV(*u_ptr++,*v_ptr++);
	    y0 = uv + READY(*y_ptr++);
	    FIXUP(y0);
	    STORE(y0, *dst_ptr++);
	}
    }
}



#undef FLAGS
#undef READUV
#undef READY
#undef FIXUP
#undef STORE

#define FLAGS 0x40080100
#define READUV(U,V) (tables[256 + (U)] + tables[512 + (V)])
#define READY(Y)    tables[Y]
#define FIXUP(Y)                 \
do {                             \
    int tmp = (Y) & FLAGS;       \
    if (tmp != 0)                \
    {                            \
	tmp  -= tmp>>8;          \
	(Y)  |= tmp;             \
	tmp   = FLAGS & ~(Y>>1); \
	(Y)  += tmp>>8;          \
    }                            \
} while (0 == 1)

#define STORE(Y,DSTPTR)         \
do {                            \
    *(DSTPTR)++ = (Y);          \
    *(DSTPTR)++ = (Y)>>22;      \
    *(DSTPTR)++ = (Y)>>11;      \
	*(DSTPTR)++ = 255;           \
} while (0 == 1)

static void yuv444_2_rgb8888(uint8_t  *dst_ptr,
		const uint8_t  *y_ptr,
		const uint8_t  *u_ptr,
		const uint8_t  *v_ptr,
		      int32_t   width,
		      int32_t   height,
		      int32_t   y_span,
		      int32_t   uv_span,
		      int32_t   dst_span,
		      int32_t   dither)
{
    height -= 1;
    while (height > 0)
    {
	height -= width<<16;
	height += 1<<16;
	while (height < 0)
	{
	    /* Do top row pair */
	    uint32_t uv, y0, y1;

	    uv  = READUV(*u_ptr++,*v_ptr++);
	    y0  = uv + READY(*y_ptr++);
	    FIXUP(y0);
	    STORE(y0, dst_ptr);
	    uv  = READUV(*u_ptr++,*v_ptr++);
	    y1  = uv + READY(*y_ptr++);
	    FIXUP(y1);
	    STORE(y1, dst_ptr);
	    height += (2<<16);
	}
	if ((height>>16) == 0)
	{
	    /* Trailing top row pix */
	    uint32_t uv, y0;

	    uv = READUV(*u_ptr++,*v_ptr++);
	    y0 = uv + READY(*y_ptr++);
	    FIXUP(y0);
	    STORE(y0, dst_ptr);
	}
	dst_ptr += dst_span-width*4;
	y_ptr   += y_span-width;
	u_ptr   += uv_span-width;
	v_ptr   += uv_span-width;
	height = (height<<16)>>16;
	height -= 1;
	if (height == 0)
	    break;
	height -= width<<16;
	height += 1<<16;
	while (height < 0)
	{
	    /* Do second row pair */
	    uint32_t uv, y0, y1;

	    uv  = READUV(*u_ptr++,*v_ptr++);
	    y0  = uv + READY(*y_ptr++);
	    FIXUP(y0);
	    STORE(y0, dst_ptr);
	    uv  = READUV(*u_ptr++,*v_ptr++);
	    y1  = uv + READY(*y_ptr++);
	    FIXUP(y1);
	    STORE(y1, dst_ptr);
	    height += (2<<16);
	}
	if ((height>>16) == 0)
	{
	    /* Trailing bottom row pix */
	    uint32_t uv, y0;

	    uv = READUV(*u_ptr++,*v_ptr++);
	    y0 = uv + READY(*y_ptr++);
	    FIXUP(y0);
	    STORE(y0, dst_ptr);
	}
	dst_ptr += dst_span-width*4;
	y_ptr   += y_span-width;
	u_ptr   += uv_span-width;
	v_ptr   += uv_span-width;
	height = (height<<16)>>16;
	height -= 1;
    }
}
#endif // YUV2RGB_H
