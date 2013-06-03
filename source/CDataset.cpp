#include "CDataset.h"

class CDataset {
 public:
  CDataset();
  ~CDataset(){}

  std::string rgbImageName, depthImageName, maskImageName, imageFilePath;
  cv::Rect bBox;
  std::vector<std::string> className;
  std::vector<cv::Point> centerPoint;
  std::vector<double> angles;

  void showDataset();
};

