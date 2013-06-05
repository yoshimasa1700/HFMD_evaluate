#ifndef __CPATCH__
#define __CPATCH__

//#include "util.h"
#include "CDataset.h"

class CPatch
{
public:
    CPatch(CDataset *d, cv::Rect r) : data(d), roi(r){}
    virtual ~CPatch(){}

    void setData(CDataset *d){data = d;}
    CDataset* getData(){return data;}

    void setRoi(cv::Rect r){roi = r;}
    cv::Rect getRoi(){return roi;}

    cv::Mat* getFeature(int featureNum){return data->feature.at(featureNum);}

private:
    cv::Rect roi;
    CDataset *data;
};

class CPosPatch : public CPatch{
public:
    CPosPatch(CPosDataset *pos, cv::Rect r) : pData(pos), CPatch(pos, r){}
    virtual ~CPosPatch();

    std::string getClassName(){return pData->getParam()->getClassName();}
private:
    CPosDataset *pData;
};

class CNegPatch : public CPatch{
public:
    CNegPatch(CNegDataset *neg, cv::Rect r) : nData(neg), CPatch(neg, r){}
    virtual ~CNegPatch();

private:
    CNegDataset *nData;
};

class CTestPatch : public CPatch{
public:
    CTestPatch(CTestDataset *tes, cv::Rect r) : tData(tes), CPatch(tes, r){}
    virtual ~CTestPatch();

private:
    CTestDataset *tData;
};

#endif
