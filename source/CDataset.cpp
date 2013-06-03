#include "CDataset.h"

CDataset::CDataset(){
    imgFlag = 0;
    featureFlag = 0;
    rgb = NULL;
    depth = NULL;
    mask = NULL;
    img.clear();
    feature.clear();
}

int CDataset::loadImage(){

}

int CDataset::releaseImage(){

}

int CDataset::extractFeatures(){

}

int CDataset::releaseFeatures(){

}

