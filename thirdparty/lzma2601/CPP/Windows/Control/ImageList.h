// Windows/Control/ImageList.h

#ifndef ZIP7_INC_WINDOWS_CONTROL_IMAGE_LIST_H
#define ZIP7_INC_WINDOWS_CONTROL_IMAGE_LIST_H

#include <CommCtrl.h>

#include "../Defs.h"

namespace NWindows {
namespace NControl {

class CImageList
{
  HIMAGELIST m_Object;
public:
  operator HIMAGELIST() const {return m_Object; }
  CImageList(): m_Object(NULL) {}
  bool Attach(HIMAGELIST imageList)
  {
    if (imageList == NULL)
      return false;
    m_Object = imageList;
    return true;
  }

  HIMAGELIST Detach()
  {
    HIMAGELIST imageList = m_Object;
    m_Object = NULL;
    return imageList;
  }
  
  bool Create(int width, int height, UINT flags, int initialNumber, int grow)
  {
    HIMAGELIST a = ImageList_Create(width, height, flags,
      initialNumber, grow);
    if (a == NULL)
      return false;
    return Attach(a);
  }
  
  bool Destroy() // DeleteImageList() in MFC
  {
    if (m_Object == NULL)
      return false;
    return BOOLToBool(ImageList_Destroy(Detach()));
  }

  ~CImageList()
    { Destroy(); }

  int GetImageCount() const
    { return ImageList_GetImageCount(m_Object); }

  bool GetImageInfo(int index, IMAGEINFO* imageInfo) const
    { return BOOLToBool(ImageList_GetImageInfo(m_Object, index, imageInfo)); }

  int Add(HBITMAP hbmImage, HBITMAP hbmMask = NULL)
    { return ImageList_Add(m_Object, hbmImage, hbmMask); }
  int AddMasked(HBITMAP hbmImage, COLORREF mask)
    { return ImageList_AddMasked(m_Object, hbmImage, mask); }
  int AddIcon(HICON icon)
    { return ImageList_AddIcon(m_Object, icon); }
  int Replace(int index, HICON icon)
    { return ImageList_ReplaceIcon(m_Object, index, icon); }

  // If index is -1, the function removes all images.
  bool Remove(int index)
    { return BOOLToBool(ImageList_Remove(m_Object, index)); }
  bool RemoveAll()
    { return BOOLToBool(ImageList_RemoveAll(m_Object)); }

  HICON ExtractIcon(int index)
    { return ImageList_ExtractIcon(NULL, m_Object, index); }
  HICON GetIcon(int index, UINT flags)
    { return ImageList_GetIcon(m_Object, index, flags); }

  bool GetIconSize(int &width, int &height) const
    { return BOOLToBool(ImageList_GetIconSize(m_Object, &width, &height)); }
  bool SetIconSize(int width, int height)
    { return BOOLToBool(ImageList_SetIconSize(m_Object, width, height)); }
};

}}

#endif
