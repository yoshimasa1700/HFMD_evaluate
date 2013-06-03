#ifndef CDATASET_H
#define CDATASET_H

#include "util.h"

// !!caution!!
// this is abstract class you should inherit this class!
class CDataset {
public:
    CDataset();
    virtual ~CDataset(){}

    int loadImage();
    int releaseImage();

    int extractFeatures();
    int releaseFeatures();

private:
    // flag for image or features loaded on memory
    bool imgFlag, featureFlag;

    // image file path
    std::string rgb, depth, mask;

    // loaded images and features
    std::vector<cv::Mat*> img, feature;

    //  cv::Rect bBox;
    //  std::vector<std::string> className;
    //  std::vector<cv::Point> centerPoint;
    //  std::vector<double> angles;

    void showDataset();
};

#endif // CDATASET_H
