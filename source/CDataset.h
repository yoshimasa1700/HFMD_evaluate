#ifndef CDATASET_H
#define CDATASET_H

#include "util.h"
#include "HoG.h"

class CParamset{
public:
    CParamset(){}
    ~CParamset(){}

    int setCenterPoint(cv::Point cp){centerPoint = cp;return 0;}
    cv::Point getCenterPoint(){return centerPoint;}

    int setClassName(std::string name){className = name;return 0;}
    std::string getClassName(){return className;}

    int setAngle(double an){angle = an;return 0;}
    double getAngle(){return angle;}

private:
    // parameters should be estimated
    cv::Point centerPoint;
    std::string className;
    double angle;

    cv::Rect boundingbox;
};

// !!caution!!
// this is abstract class you should inherit this class!
class CDataset {
public:
    CDataset();
    virtual ~CDataset();

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

    // min and max filter
    int minFilter(cv::Mat* src, cv::Mat* des, int fWind);
    int maxFilter(cv::Mat* src, cv::Mat* des, int fWind);

    //  cv::Rect bBox;
    //  std::vector<std::string> className;
    //  std::vector<cv::Point> centerPoint;
    //  std::vector<double> angles;

    void showDataset();
};

class CPosDataset : public CDataset{
public:
    CPosDataset();
    virtual ~CPosDataset();

    CParamset param;
};

class CNegDataset : public CDataset{
public:
    CNegDataset();
    virtual ~CNegDataset();

};

class CTestDataset : public CDataset{
public:
    CTestDataset();
    virtual ~CTestDataset();

    std::vector<CParamset> param;
};

#endif // CDATASET_H
