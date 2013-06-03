#ifndef __CPATCH__
#define __CPATCH__

#include "util.h"

class CPatch
{
public:
    CPatch(CDataset *d, cv::Rect r) : data(d), roi(r){}
    virtual ~CPatch(){}

    void setData(CDataset *d){data = d;}
    DATA getData(){return data;}

    void setRoi(cv::Rect r){roi = r;}
    cv::Rect getRoi(){return roi;}

    getFeatureRoi(int featureNum){return &(*(data->feature.at(featureNum)))(roi);}

private:
    cv::Rect roi;
    CDataset *data;
};

class CPosPatch : public CPatch{
    CPosPatch(CPosDataset *pos, cv::Rect r) : CPatch(pos, r){}
    virtual ~CPosPatch();
};

class CPosPatch : public CPatch{
public:
    CPosPatch(CPosDataset *pos, cv::Rect r) : pData(pos), CPatch(pos, r){}
    virtual ~CPosPatch();

    std::string getClassName(){return pData->param.getClassName();}
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
    CNegPatch(CNegDataset *neg, cv::Rect r) : nData(neg), CPatch(neg, r){}
    virtual ~CNegPatch();

private:
    CNegDataset *nData;
};

#endif
