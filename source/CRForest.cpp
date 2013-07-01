#include <boost/timer.hpp>
#include "CRForest.h"

double euclideanDist(cv::Point p, cv::Point q)
{
    cv::Point diff = p - q;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

void CRForest::learning(){
    // grow each tree
    // if you want to fix this program multi thread
    // you should change below
#pragma omp parallel
    {
#pragma omp for
        for(int i = 0;i < conf.ntrees; ++i){
            growATree(i);
        } // end tree loop
    }
}

void CRForest::growATree(const int treeNum){
    // positive, negative dataset
    std::vector<CPosDataset> posSet(0);
    std::vector<CNegDataset> negSet(0);

    // positive, negative patch
    std::vector<CPosPatch> posPatch(0);
    std::vector<CNegPatch> negPatch(0);

    char buffer[256];

    std::cout << "tree number " << treeNum << std::endl;

    // initialize random seed
    boost::mt19937    gen( treeNum * static_cast<unsigned long>(time(NULL)) );
    boost::timer t;

    loadTrainPosFile(conf, posSet);//, gen);
    loadTrainNegFile(conf, negSet);

    std::cout << "dataset loaded" << std::endl;

    // initialize class database
    //classDatabase.clear();

    // extract pos features and register classDatabase
    for(int i = 0; i < posSet.size(); ++i){
        //std::cout << i << std::endl;

        //std::cout << posSet.at(i).rgb << std::endl;
        if(posSet.at(i).loadImage(conf) == -1 && conf.learningMode != 2){
            exit(-1);
        }
	
        posSet.at(i).extractFeatures(conf);

        //std::cout << posSet.size() << std::endl;

        classDatabase.add(posSet.at(i).getParam()->getClassName(),posSet.at(i).img.at(0)->size(),0);
    }

    classDatabase.show();

    // extract neg features
    for(int i = 0; i < negSet.size(); ++i){
        negSet.at(i).loadImage(conf);

        negSet.at(i).extractFeatures(conf);
    }

    CRTree *tree = new CRTree(conf.min_sample, conf.max_depth, classDatabase.vNode.size(),this->classDatabase);
    std::cout << "tree created" << std::endl;

    extractPosPatches(posSet,posPatch,conf,treeNum,this->classDatabase);
    extractNegPatches(negSet,negPatch,conf);

    std::cout << "extracted pathes" << std::endl;
    std::vector<int> patchClassNum(classDatabase.vNode.size(), 0);

    for(int j = 0; j < posPatch.size(); ++j)
        patchClassNum.at(classDatabase.search(posPatch.at(j).getClassName()))++;

    // grow tree
    //vTrees.at(treeNum)->growTree(vPatches, 0,0, (float)(vPatches.at(0).size()) / ((float)(vPatches.at(0).size()) + (float)(vPatches.at(1).size())), conf, gen, patchClassNum);
    tree->growTree(posPatch,negPatch, 0,0, ((float)posPatch.size() / (float)(posPatch.size() + negPatch.size())), conf, patchClassNum);

    // save tree
    sprintf(buffer, "%s%03d.txt",
            conf.treepath.c_str(), treeNum + conf.off_tree);
    std::cout << "tree file name is " << buffer << std::endl;
    tree->saveTree(buffer);

    // save class database
    sprintf(buffer, "%s%s%03d.txt",
            conf.treepath.c_str(),
            conf.classDatabaseName.c_str(), treeNum + conf.off_tree);
    std::cout << "write tree data" << std::endl;
    classDatabase.write(buffer);

    double time = t.elapsed();

    std::cout << "tree " << treeNum << " calicuration time is " << time << std::endl;

    sprintf(buffer, "%s%03d_timeResult.txt",conf.treepath.c_str(), treeNum + conf.off_tree);
    std::fstream lerningResult(buffer, std::ios::out);
    if(lerningResult.fail()){
        std::cout << "can't write result" << std::endl;
    }

    lerningResult << time << std::endl;

    lerningResult.close();

    delete tree;

    posPatch.clear();
    negPatch.clear();

    posSet.clear();
    negSet.clear();
}

void CRForest::loadForest(){
    char buffer[256];
    char buffer2[256];
    std::cout << "loading forest..." << std::endl;
    for(int i = 0; i < vTrees.size(); ++i){
        sprintf(buffer, "%s%03d.txt",conf.treepath.c_str(),i);
        sprintf(buffer2, "%s%s%03d.txt", conf.treepath.c_str(), conf.classDatabaseName.c_str(), i);
        vTrees[i] = new CRTree(buffer, buffer2, conf);

        //std::cout << buffer2 << std::endl;
        classDatabase.read(buffer2);
        pBar(i,vTrees.size(),50);
    }
    std::cout << std::endl;
}

// name   : detect function
// input  : image and dataset
// output : classification result and detect picture
CDetectionResult CRForest::detection(CTestDataset &testSet) const{
    int classNum = classDatabase.vNode.size();//contain class number
    std::vector<CTestPatch> testPatch;
    std::vector<const LeafNode*> result;

    //std::vector<const LeafNode*> storedLN(0);

    std::vector<std::vector<CParamset> > cluster(0);
    std::vector<CParamset> clusterMean(0);

    cv::vector<cv::Mat> outputImage(classNum);
    cv::vector<cv::Mat> outputImageColorOnly(classNum);
    //cv::vector<cv::vector<cv::Mat> > outputImageColorOnlyPerTree(classNum);
    std::vector<int> totalVote(classNum,0);
    boost::timer t;

    testSet.loadImage(conf);

    testSet.extractFeatures(conf);
    //testSet.releaseImage();

    //t.restart();

//#pragma omp parallel
    //{
//#pragma omp for
        for(int i = 0; i < classNum; ++i){
            outputImage.at(i) = testSet.img.at(0)->clone();
            outputImageColorOnly.at(i) = cv::Mat::zeros(testSet.img.at(0)->rows,testSet.img.at(0)->cols,CV_32FC1);
            //        for(int j = 0; j < this->vTrees.size(); ++j)
            //            outputImageColorOnlyPerTree.at(i).push_back()
        }
    //}

    // extract feature from test image
    //features.clear();
    //extractFeatureChannels(image.at(0), features);

    // add depth image to features
    //features.push_back(image.at(1));


    // extract patches from features
    extractTestPatches(testSet,testPatch,this->conf);

    std::cout << "patch num: " << testPatch.size() << std::endl;

    //

    std::cout << "detecting..." << std::endl;
    // regression and vote for every patch

    for(int j = 0; j < testPatch.size(); ++j){
        result.clear();
        this->regression(result, testPatch.at(j));



//#pragma omp parallel
//        {
//#pragma omp for
            for(int m = 0; m < result.size(); ++m){//std::vector<const LeafNode*>::const_iterator itL = result.begin();itL!=result.end(); ++itL) {
//#pragma omp parallel
//                {
//#pragma omp for
                    for(int c = 0; c < classNum; c++){
                        //if(!result.at(m)->param.at(c).empty()){
                        if(result.at(m)->pfg.at(c) > 0.9  ){

                            for(int l = 0; l < result.at(m)->param.at(c).size(); ++l){
                                //storedLN.push_back(result.at(m));

                                cv::Point patchSize(conf.p_height/2,conf.p_width/2);
                                cv::Point pos(testPatch.at(j).getRoi().x + patchSize.x +  result.at(m)->param.at(c).at(l).getCenterPoint().x,
                                              testPatch.at(j).getRoi().y + patchSize.y +  result.at(m)->param.at(c).at(l).getCenterPoint().y);

                                //std::cout << pos << std::endl;

                                int found = 0;
                                for(int p = 0; p < cluster.size(); ++p){
                                    if(clusterMean.at(p).getClassName() == result.at(m)->param.at(c).at(l).getClassName()){
                                        double distanceCtoP = std::sqrt(std::pow((double)(clusterMean.at(p).getCenterPoint().x - pos.x),(int)2) +
                                                                        std::pow((double)(clusterMean.at(p).getCenterPoint().y - pos.y),(int)2));

                                        if(distanceCtoP < 20.0){
                                            //std::cout << "same object?" << std::endl;

                                            found = 1;
                                            CParamset new_param;
                                            new_param = result.at(m)->param.at(c).at(l);
                                            new_param.setCenterPoint(pos);
                                            //std::cout << pos.x << " + " << clusterMean.at(p).getCenterPoint().x << std::endl;
                                            cluster.at(p).push_back(new_param);
                                            //clusterMean.at(p) += new_param;
                                            //clusterMean.at(p) /= 2.0;
                                            //if(p == 0)
                                                //clusterMean.at(0).showParam();
                                        }
                                    }
                                }
                                if(found == 0 && conf.clusterNumLimit > cluster.size() && pos.x > 0 && pos.y > 0 && pos.x < outputImage.at(0).cols && pos.y < outputImage.at(0).rows){
                                    std::vector<CParamset> new_cluster(0);
                                    CParamset new_param;

                                    new_param = result.at(m)->param.at(c).at(l);
                                    new_param.setCenterPoint(pos);
                                    //new_param.setAngle(re
                                    new_cluster.push_back(new_param);
                                    cluster.push_back(new_cluster);
                                    clusterMean.push_back(new_param);

                                    //new_param.showParam();
                                }
                                //std::cout << c << " " << l << " " << result.at(m)->param.at(c).at(l).getClassName() << std::endl;


                                if(pos.x > 0 && pos.y > 0 && pos.x < outputImageColorOnly.at(c).cols && pos.y < outputImageColorOnly.at(c).rows){
                                    outputImageColorOnly.at(c).at<float>(pos.y,pos.x) += result.at(m)->pfg.at(c) / ( result.size() * result.at(m)->param.size());//(result.at(m)->pfg.at(c) - 0.9);// * 100;//weight * 500;
                                    //std::cout << classDatabase.vNode.at(c).name << " " << result.at(m)->pfg.at(c) << std::endl;

                                    totalVote.at(c) += 1;
                                }
                            }
                        }
                        //}
                    }
                //}

            } // for every leaf
       // }
        //pBar(j,testPatch.size(),50);
    } // for every patch
    //  }
    std::cout << std::endl;

    for(int p = 0; p < cluster.size(); ++p){
        //clusterMean.at(p) /= (float)cluster.at(p).size();
        clusterMean.at(p).showParam();
        std::cout << cluster.at(p).size() << std::endl;
    }

    // vote end

//#pragma omp parallel
    //{
//#pragma omp for
        // find balance by mean shift
        for(int i = 0; i < classNum; ++i){
            //        cv::Mat hsv,hue,rgb;
            //        int bins = 256;

            //        double min,max;
            //        cv::Point minLoc,maxLoc;
            //        cv::minMaxLoc(outputImageColorOnly.at(i),&min,&max,&minLoc,&maxLoc);

            cv::GaussianBlur(outputImageColorOnly.at(i),outputImageColorOnly.at(i), cv::Size(21,21),0);

            //        //cv::cvtColor(outputImageColorOnly.at(i), rgb, CV_GRAY2BGR);
            //        //cv::cvtColor(rgb, hsv , CV_BGR2HSV);

            //        hue.create( outputImageColorOnly.at(i).size(), outputImageColorOnly.at(i).depth() );
            //        int ch[] = { 0, 0 };
            //        mixChannels( &outputImageColorOnly.at(i), 1, &hue, 1, ch, 1 );


            //        const int ch_width = 400;
            //        cv::Mat hist;
            //        cv::Mat hist_img(cv::Size(ch_width, 200), CV_8UC3, cv::Scalar::all(255));;
            //        int histSize = MAX( bins, 2 );
            //        float hue_range[] = { 0, 1 };
            //        const float* ranges = { hue_range };
            //        const int hist_size = 256;
            //        double max_val = .0;
            //        double second_val = .0;

            //        /// Get the Histogram and normalize it
            //        cv::calcHist( &outputImageColorOnly.at(i) , 1, 0, cv::Mat(), hist, 1, &histSize, &ranges, true, false );
            //        cv::normalize( hist, hist, 0., 256., cv::NORM_MINMAX, -1, cv::Mat() );

            //        cv::minMaxLoc(hist, 0, &max_val);
            //        hist.at<float>(0) = 0;
            //        cv::minMaxLoc(hist, 0, &second_val);

            //        hist.at<float>(0) = max_val;

            //        // (4)scale and draw the histogram(s)
            //        cv::Scalar color = cv::Scalar::all(100);
            //        //for(int i=0; i<sch; i++) {
            //        //  if(sch==3)
            //        //    color = Scalar((0xaa<<i*8)&0x0000ff,(0xaa<<i*8)&0x00ff00,(0xaa<<i*8)&0xff0000, 0);
            //        hist.convertTo(hist, hist.type(), 200 * 1.0/second_val,0);//?1./max_val:0.,0);
            //        for(int j=0; j<hist_size; ++j) {
            //            int bin_w = cv::saturate_cast<int>((double)ch_width/hist_size);
            //            //std::cout << "draw rect " << bin_w << " " << i << " " << hist.at<float>(j) << " " << max_val << std::endl;
            //            cv::rectangle(hist_img,
            //                          cv::Point( j*bin_w, hist_img.rows),
            //                          cv::Point((j+1)*bin_w, hist_img.rows-cv::saturate_cast<int>(hist.at<float>(j))),
            //                          color, -1);
            //        }


            //        //show and write histgram
            ////        cv::imwrite("test.png",hist_img);

            ////        cv::namedWindow("test");
            ////        cv::imshow("test",hist_img);
            ////        cv::waitKey(0);
            ////        cv::destroyWindow("test");

            //        /// Get Backprojection
            //        cv::Mat backproj;
            //        calcBackProject( &hue, 1, 0, hist, backproj, &ranges, 1, true );

            //        cv::Rect tempRect = cv::Rect(maxLoc.x,maxLoc.y,classDatabase.vNode.at(i).classSize.width,classDatabase.vNode.at(i).classSize.height);//classDatabase.vNode.at(i).classSize.width,classDatabase.vNode.at(i).classSize.height);//outputImageColorOnly.at(i).cols,outputImageColorOnly.at(i).rows);
            //        cv::TermCriteria terminator;
            //        terminator.maxCount = 1000;
            //        terminator.epsilon  = 10;
            //        terminator.type = cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS;
            //        cv::meanShift(backproj,tempRect,terminator);

            //        //cv::Size tempSize = classDatabase.vNode.at(c).classSize;
            //        //cv::Rect_<int> outRect(tempRect.x,maxLoc.y - tempSize.height / 2 , tempSize.width,tempSize.height);
            //        cv::rectangle(outputImage.at(i),tempRect,cv::Scalar(0,200,0),3);
            //        cv::putText(outputImage.at(i),classDatabase.vNode.at(i).name,cv::Point(tempRect.x,tempRect.y),cv::FONT_HERSHEY_SIMPLEX,1.2, cv::Scalar(0,0,0), 2, CV_AA);

        }
    //}



    double time = t.elapsed();
    std::cout << time << "sec" << std::endl;
    std::cout << 1 / (time) << "Hz" << std::endl;

    std::stringstream ss;



    for(int p = 0; p < clusterMean.size(); ++p){
        ss << clusterMean.at(p).getAngle();

        int clustersClass = classDatabase.search(clusterMean.at(p).getClassName());
        cv::circle(outputImage.at(clustersClass), clusterMean.at(p).getCenterPoint(), 10,cv::Scalar(200,200,0));
        //cv::putText(outputImage.at(clustersClass),clusterMean.at(p).getClassName(),clusterMean.at(p).getCenterPoint(),cv::FONT_HERSHEY_SIMPLEX,1.2, cv::Scalar(200,200,0), 2, CV_AA);
        //cv::putText(outputImage.at(clustersClass),ss.str(),clusterMean.at(p).getCenterPoint() + cv::Point(0,-10),cv::FONT_HERSHEY_SIMPLEX,1.2, cv::Scalar(200,200,0), 2, CV_AA);
    }


    std::string opath;
    if(!conf.demoMode){
        //create result directory
        opath = testSet.getRgbImagePath();
        opath.erase(opath.find_last_of(PATH_SEP));
        std::string imageFilename = testSet.getRgbImagePath();
        imageFilename.erase(imageFilename.find_last_of("."));
        //imageFilename.erase(imageFilename.begin(),imageFilename.find_last_of(PATH_SEP));
        imageFilename = imageFilename.substr(imageFilename.rfind(PATH_SEP),imageFilename.length());

        //opath += PATH_SEP;
        opath += imageFilename;
        std::string execstr = "mkdir -p ";
        execstr += opath;
        system( execstr.c_str() );

        for(int c = 0; c < classNum; ++c){
            std::stringstream cToString;
            cToString << c;
            std::string outputName = "output" + cToString.str() + ".png";
            std::string outputName2 = opath + PATH_SEP + "vote_" + classDatabase.vNode.at(c).name + ".png";
            //cv::imwrite(outputName.c_str(),outputImage.at(c));
            //cv::cvtColor(outputImageColorOnly)

            cv::Mat writeImage;
            //hist.convertTo(hist, hist.type(), 200 * 1.0/second_val,0);
            outputImageColorOnly.at(c).convertTo(writeImage, CV_8UC1, 254);
            cv::imwrite(outputName2.c_str(),writeImage);
        }
    }

    //std::cout << "detection result outputed" << std::endl;


    //    cv::cvtColor(outputImageColorOnly.at(0),outputImageC1,CV_8UC1);

    //    cv::namedWindow("test");
    //    cv::imshow("test",outputImageC1);
    //    cv::waitKey(0);
    //    cv::destroyWindow("test");

    //std::vector<detectionResult*> dResult(0);
    CDetectionResult detectResult;

    detectResult.voteImage = outputImageColorOnly;

    std::cout << "show ground truth" << std::endl;
    //    std::cout << dataSet.className.size() << std::endl;
    //    std::cout << dataSet.centerPoint.size() << std::endl;
    for(int i = 0; i < testSet.param.size(); ++i){
        testSet.param.at(i).showParam();
    }

    std::cout << "show result" << std::endl;
    for(int c = 0; c < classNum; ++c){

        double min,max;
        cv::Point minLoc,maxLoc;
        cv::minMaxLoc(outputImageColorOnly.at(c),&min,&max,&minLoc,&maxLoc);
        //double score  = (double)(outputImageColorOnly.at(c).at<int>(maxLoc.y,maxLoc.x) / (double)( conf.stride * conf.stride * outputImage.at(c).cols * outputImage.at(c).rows)) * 1000000;


        //cv::circle(outputImage.at(c),maxLoc,20,cv::Scalar(200,0,0),3);

        //if(score > conf.detectThreshold){
        //        cv::namedWindow("test");
        //        cv::imshow("test",outputImage.at(c));
        //        cv::waitKey(0);
        //        cv::destroyWindow("test");
        cv::Size tempSize = classDatabase.vNode.at(c).classSize;
        cv::Rect_<int> outRect(maxLoc.x - tempSize.width / 2,maxLoc.y - tempSize.height / 2 , tempSize.width,tempSize.height);
        cv::rectangle(outputImage.at(c),outRect,cv::Scalar(0,0,200),3);
        cv::putText(outputImage.at(c),classDatabase.vNode.at(c).name,cv::Point(outRect.x,outRect.y),cv::FONT_HERSHEY_SIMPLEX,1.2, cv::Scalar(0,0,200), 2, CV_AA);
        //}

        //        for(int i = 0; i < storedLN.size(); ++i){

        //        }

        // display grand truth
        if(!conf.demoMode){
            for(int i = 0; i < testSet.param.size(); ++i){
                int tempClassNum = classDatabase.search(testSet.param.at(i).getClassName());
                if(tempClassNum != -1){
                    cv::Size tempSize = classDatabase.vNode.at(tempClassNum).classSize;
                    cv::Rect_<int> outRect(testSet.param.at(i).getCenterPoint().x - tempSize.width / 2,testSet.param.at(i).getCenterPoint().y - tempSize.height / 2 , tempSize.width,tempSize.height);
                    cv::rectangle(outputImage.at(tempClassNum),outRect,cv::Scalar(200,0,0),3);
                    cv::putText(outputImage.at(tempClassNum),classDatabase.vNode.at(c).name,cv::Point(testSet.param.at(i).getCenterPoint().x, testSet.param.at(i).getCenterPoint().y),cv::FONT_HERSHEY_SIMPLEX,1.2, cv::Scalar(200,0,0), 2, CV_AA);
                }
            }
        }

        std::cout << c << " Name : " << classDatabase.vNode.at(c).name << "\tvote : " << totalVote.at(c) << " Score : " << outputImageColorOnly.at(c).at<float>(maxLoc.y, maxLoc.x) << " CenterPoint : " << maxLoc << std::endl;//"\tscore : " << score << std::endl;

        if(!conf.demoMode){
            std::string outputName = opath + PATH_SEP + "detectionResult" + "_" + classDatabase.vNode.at(c).name + ".png";
            cv::imwrite(outputName.c_str(),outputImage.at(c));
        }

        CDetectedClass detectedClass;
        detectedClass.name = classDatabase.vNode.at(c).name;

        double minError = DBL_MAX;
        std::string nearestObject;
        for(int d = 0; d < testSet.param.size(); ++d){
            double tempError = euclideanDist(maxLoc,testSet.param.at(d).getCenterPoint());//= std::sqrt(std::pow((double)(maxLoc.x - testSet.param.at(0).getCenterPoint().x), 2) + std::pow((double)(maxLoc.y - testSet.param.at(0).getCenterPoint().y), 2));
            //std::cout << tempError << std::endl;
            if(tempError < minError){
                minError = tempError;
                nearestObject = testSet.param.at(d).getClassName();
            }
        }
        detectedClass.error = minError;
        detectedClass.nearestClass = nearestObject;
        //        std::cout << "maxLoc x is " << maxLoc.x << " y is " << maxLoc.y << " GT x " << testSet.param.at(0).getCenterPoint().x << " GT y " << testSet.param.at(0).getCenterPoint().y << std::endl;
        //        std::cout << zizyoHeikin << " " << std::sqrt(std::abs(zizyoHeikin)) << std::endl;
        //        detectedClass.error = std::sqrt(abs(zizyoHeikin));
        detectedClass.score = outputImageColorOnly.at(c).at<float>(maxLoc.y, maxLoc.x);
        detectResult.detectedClass.push_back(detectedClass);
    }

    return detectResult;
}

// Regression
void CRForest::regression(std::vector<const LeafNode*>& result, CTestPatch &patch) const{
    result.resize( vTrees.size() );
    //std::cout << "enter regression" << std::endl;
    for(int i=0; i < vTrees.size(); ++i) {
        //std::cout << "regressioning " << i << std::endl;
        result[i] = vTrees[i]->regression(patch);
    }
}

//void CRForest::loadImages(cv::vector<cv::vector<cv::Mat *> > &img, std::vector<CDataset> &dataSet){
//    img.resize(0);

//    cv::Mat* rgb,*depth;//, *mask;
//    cv::vector<cv::Mat*> planes;
//    cv::vector<cv::Mat*> allImages;
//    //cv::vector<cv::Mat> rgbSplited;

//    for(int i = 0;i < dataSet.size(); ++i){
//        rgb = new cv::Mat();
//        depth = new cv::Mat();
//        //mask = new cv::Mat();

//        // load Mask image

//        //*mask = cv::imread(dataSet.at(i).imageFilePath
//        //                   + dataSet.at(i).maskImageName,3).clone();

//        // load RGB image
//        *rgb = cv::imread(dataSet.at(i).imageFilePath
//                          + dataSet.at(i).rgbImageName,3).clone();

//        //std::cout << dataSet.at(i).rgbImageName << " " << rgb->channels() << std::endl;
//        // load Depth image
//        *depth = cv::imread(dataSet.at(i).imageFilePath
//                            + dataSet.at(i).depthImageName,
//                            CV_LOAD_IMAGE_ANYDEPTH).clone();
//        cv::Point tempPoint;
//        tempPoint.x = (*rgb).cols / 2;
//        tempPoint.y = (*rgb).rows / 2;

//        dataSet.at(i).centerPoint.push_back(tempPoint);

//        //        cv::namedWindow("test");
//        //        cv::imshow("test",*rgb);
//        //        cv::waitKey(0);
//        //        cv::destroyWindow("test");



//        //std::cout << depth << std::endl;



//        for(int k = 0;k < rgb->cols; ++k)
//            for(int l = 0;l < rgb->rows; ++l){
//                //std::cout << depth.at<ushort>(l, k) << " " << std::endl;
//                //if(!(bool)mask->at<char>(l, k))
//                //depth->at<ushort>(l, k) = 0;
//                // for(int j = 0;j < 3; ++j)
//                //   if(!(bool)mask.at<char>(l, k))
//                //     rgb.at<cv::Vec3b>(l, k)[j] = 0;
//            }
//        //rgbSplited.resize(rgb.channels());

//        //cv::split(rgb, rgbSplited);

//        allImages.clear();
//        allImages.push_back(rgb);
//        allImages.push_back(depth);
//        img.push_back(allImages);

//        //delete mask;
//    }


//}

