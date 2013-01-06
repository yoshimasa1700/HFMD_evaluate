#ifndef __CPATCH__
#define __CPATCH__

#include "util.h"

class CPatch 
{
 public:
 CPatch(cv::Rect roi, cv::Mat* image)
   : patchRoi(roi)
    {
      //patch = image(cv::Rect(roi));
    }
  
  CPatch(){}

  void setPatch(int x, int y, int w, int h, cv::Mat image){
    //patch = image(cv::Rect(x, y, w, h));
  };

  void setPatch(cv::Rect roi, std::vector<cv::Mat> &image, std::vector<cv::Point> center){
    //patch = image(cv::Rect(roi));
    patchRoi = roi;
    //image(roi);
    //vCenter = center;
    for(int i = 0; i < image.size(); ++i)
      p_image.push_back(&(image.at(i)));
    vCenter = center;
  }
  
  cv::Rect patchRoi;
  std::vector<cv::Point> vCenter;
  //std::vector<cv::Mat *> patch;
  std::vector<cv::Mat *> p_image;
};

#endif
