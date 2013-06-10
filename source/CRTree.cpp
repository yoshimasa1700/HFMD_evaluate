#include "CRTree.h"

using namespace std;
boost::lagged_fibonacci1279 CRTree::genTree = boost::lagged_fibonacci1279();

void CRTree::calcHaarLikeFeature(const cv::Mat &patch, const int* test, int &p1, int &p2) const{
    float m[6];

    int angle = test[0];

    cv::Mat showMat(patch.rows, patch.cols, CV_8U);

//    std::cout << angle << std::endl;
//    patch.convertTo(showMat, CV_8U, 255.0/ 1000.0);
//    cv::namedWindow("test");
//    cv::imshow("test", showMat);
//    cv::waitKey(0);

    cv::Mat affine = cv::getRotationMatrix2D(cv::Point(config.p_width / 2 + 1, config.p_height / 2 + 1), (float)test[0] /*/ 180 * CV_PI*/, 1.0);
    cv::Mat rotatedMat(patch.rows, patch.cols, patch.type());

    cv::warpAffine(patch, rotatedMat, affine, cv::Size(patch.cols, patch.rows));

//    rotatedMat.convertTo(showMat, CV_8U, 255.0/ 1000.0);

//    cv::imshow("test", showMat);
//    cv::waitKey(0);
//    cv::destroyAllWindows();

    p1 = 0;
    p2 = 0;
    //std::cout << test[0] << " " << test[1] << " " << test[2] << std::endl;

    switch(test[1]){
    case 0:
        // edge feature
        for(int i = 0; i < rotatedMat.rows; ++i){
            for(int j = 0; j < rotatedMat.cols; ++j)
                if(i < i * test[2] / 100)
                    p1 += rotatedMat.at<ushort>(i, j);
            else
                    p2 += rotatedMat.at<ushort>(i, j);
        }
        break;

    case 1:
        // line feature
        for(int i = 0; i < rotatedMat.rows; ++i){
            for(int j = 0; j < rotatedMat.cols; ++j)
                if(i < rotatedMat.rows * test[2] / 100 / 2 || i > rotatedMat.rows - rotatedMat.rows * test[2] / 100 / 2)
                    p1 += rotatedMat.at<ushort>(i, j);
            else
                    p2 += rotatedMat.at<ushort>(i, j);
        }

        break;

    case 2:
        // center
        for(int i = 0; i < rotatedMat.rows; ++i){
            for(int j = 0; j < rotatedMat.cols; ++j){
                //std::cout << i << " " << j << std::endl;
                //std::cout << rotatedMat.rows * test[2] / 100 / 2 << " " << rotatedMat.rows - rotatedMat.rows * test[2] / 100 / 2 << " " << rotatedMat.cols * test[2] / 100 / 2 << " " << rotatedMat.cols - rotatedMat.cols * test[2] / 100 / 2 << std::endl;
                if(i < rotatedMat.rows * test[2] / 100 / 2 || i > rotatedMat.rows - rotatedMat.rows * test[2] / 100 / 2 || j < rotatedMat.cols * test[2] / 100 / 2 || j > rotatedMat.cols - rotatedMat.cols * test[2] / 100 / 2)
                    p1 += rotatedMat.at<ushort>(i, j);
                else
                    p2 += rotatedMat.at<ushort>(i, j);
            }

        }

        break;
    default:
        break;
    }

    return;
}

const LeafNode* CRTree::regression(CTestPatch &patch) const {
    // pointer to current node
    const int* pnode = &treetable[0];
    int node = 0;
    int p1,p2;

    // Go through tree until one arrives at a leaf, i.e. pnode[0]>=0)
    while(pnode[0]==-1) {
        // binary test 0 - left, 1 - right
        // Note that x, y are changed since the patches are given as matrix and not as image
        // p1 - p2 < t -> left is equal to (p1 - p2 >= t) == false

        // pointer to channel
        //uchar* ptC = ptFCh[pnode[5]];

        cv::Mat tempFeture = *patch.getFeature(pnode[9]);
        cv::Mat ptC = tempFeture(patch.getRoi());

        normarizationByDepth(&patch,ptC);

        //std::cout << ptC << std::endl;

        if(pnode[9] == 32){
//            p1 = 0;
//            p2 = 0;
//            for(int j = 0;j < pnode[3]; ++j){
//                for(int k = 0; k < pnode[4]; ++k)
//                    p1 += (int)ptC.at<uchar>(k + pnode[2],j +  pnode[1]);
//            }

//            for(int j = 0;j < pnode[7]; ++j){
//                for(int k = 0; k < pnode[8]; ++k)
//                    p2 += (int)ptC.at<uchar>(k + pnode[6],j +  pnode[5]);
//            }

            calcHaarLikeFeature(ptC,pnode,p1,p2);

        }else{
            // get pixel values
            p1 = (int)ptC.at<uchar>(pnode[2], pnode[1]);//*(ptC+pnode[1]+pnode[2]*stepImg);
            p2 = (int)ptC.at<uchar>(pnode[6], pnode[5]);//*(ptC+pnode[3]+pnode[4]*stepImg);
        }
        // test
        bool test = ( p1 - p2 ) >= pnode[10];

        // next node: 2*node_id + 1 + test
        // increment node/pointer by node_id + 1 + test
        int incr = node+1+test;
        node += incr;
        pnode += incr*11;
    }

    // return leaf
    return &leaf[pnode[0]];


}

// Read tree from file
CRTree::CRTree(const char* filename, const char* databaseName,CConfig &conf):config(conf) {
    cout << "Load Tree " << filename << endl;

    classDatabase.read(databaseName);

    int dummy;

    ifstream in(filename);
    if(in.is_open()) {
        // allocate memory for tree table
        in >> max_depth;
        std::cout << "max depth: " << max_depth << std::endl;
        num_nodes = (int)pow(2.0,int(max_depth+1))-1;
        // num_nodes x 7 matrix as vector
        treetable = new int[num_nodes * 11];
        int* ptT = &treetable[0];

        // allocate memory for leafs
        in >> num_leaf;
        leaf = new LeafNode[num_leaf];

        std::cout << "number of leaf: " << num_leaf << std::endl;

        // number of center points per patch
        in >> num_cp;

        // read tree nodes
        for(unsigned int n=0; n<num_nodes; ++n) {
            in >> dummy; in >> dummy;
            for(unsigned int i=0; i<11; ++i, ++ptT) {
                in >> *ptT;
            }
        }

        // this is for debug
        std::vector<double> pfgSum(3,0.);
        std::vector<int> voteSum(3,0);

        // read tree leafs
        LeafNode* ptLN = &leaf[0];
        for(unsigned int l=0; l<num_leaf; ++l, ++ptLN) {
            in >> dummy; // leaf node number
            int allClassNum = 0; // class number
            in >> allClassNum;
            int containClassNum = 0;
            in >> containClassNum;

            ptLN->pfg.resize(allClassNum);
            ptLN->vCenter.resize(allClassNum);
            ptLN->param.resize(allClassNum);

            for(int i = 0; i < allClassNum; ++i)
                ptLN->pfg.at(i) = 0.0;

            int cNum;
            int containPoints;
            for(int i = 0; i < containClassNum; ++i){
                in >> cNum;
                std::cout << cNum;// << std::endl;
                std::string tempCName;
                in >> tempCName;

                int collectClassNum = classDatabase.search(tempCName);

                cNum = collectClassNum;

                std::cout << tempCName << std::endl;
                in >> ptLN->pfg.at(cNum);
                in >> containPoints;
                std::cout << " " << ptLN->pfg.at(cNum) << " " << containPoints << std::endl;

                pfgSum.at(cNum) += ptLN->pfg.at(cNum);
                voteSum.at(cNum) += containPoints;

                ptLN->vCenter.at(cNum).resize(containPoints);
                ptLN->param.at(cNum).clear();//resize(containPoints);
                for(int j = 0; j < containPoints; j++){
                    //in >> ptLN->vCenter.at(cNum).at(j).x;

                    // read each parameter
                    //std::string tempClassName;
                    //in >> tempClassName;
                    cv::Point tempPoint;
                    in >> tempPoint.x;
                    in >> tempPoint.y;
                    double tempAngle;
                    in >> tempAngle;

                    CParamset tempParam;
                    //tempParam.setClassName(tempClassName);
                    tempParam.setCenterPoint(tempPoint);
                    tempParam.setAngle(tempAngle);
                    tempParam.setClassName(tempCName);

                    ptLN->param.at(cNum).push_back(tempParam);
                    //in >> ptLN->vCenter.at(cNum).at(j).y;
                    ptLN->param.at(cNum).at(j).showParam();//readParam(in);
                }
            }
        }


        for(int i = 0; i < 3; i++){
            std::cout << pfgSum.at(i) << " " << voteSum.at(i) << std::endl;
        }

    } else {
        cerr << "Could not read tree: " << filename << endl;
    }

    in.close();

    //std::cin >> dummy;

}

bool CRTree::saveTree(const char* filename) const {
    cout << "Save Tree " << filename << endl;

    bool done = false;

    ofstream out(filename);
    if(out.is_open()) {

        out << max_depth << " " << num_leaf << " " << num_cp << endl;

        // save tree nodes
        int* ptT = &treetable[0];
        int depth = 0;
        unsigned int step = 2;
        for(unsigned int n=0; n<num_nodes; ++n) {
            // get depth from node
            if(n==step-1) {
                ++depth;
                step *= 2;
            }

            out << n << " " << depth << " ";
            for(unsigned int i=0; i<11; ++i, ++ptT) {
                out << *ptT << " ";
            }
            out << endl;
        }
        out << endl;

        // save tree leafs
        LeafNode* ptLN = &leaf[0];
        for(unsigned int l=0; l<num_leaf; ++l, ++ptLN) {
            out << l << " ";

            int containClassNum = 0;
            for(int i = 0; i < ptLN->pfg.size(); ++i){
                if(ptLN->pfg.at(i) != 0)
                    containClassNum++;
            }

            out << ptLN->pfg.size() << " " << containClassNum << " "; // class number

            // std::vector<int> classNum(defaultClass.size(), 0);
            // for(int i = 0; i < ptLN->vCenter.size(); ++i)
            // 	classNum(ptLN->vClass.at(i)) += 1;

            for(int j = 0; j < ptLN->pfg.size(); ++j){
                if(ptLN->pfg.at(j) != 0){
                    //std::cout << j << std::endl;
                    out << j << " " << ptLN->param.at(j).at(0).getClassName() << " " << ptLN->pfg.at(j) << " " << ptLN->param.at(j).size() << " ";
                    for(int i = 0; i < ptLN->param.at(j).size(); ++i){
                        out << ptLN->param.at(j).at(i).outputParam();
                    }
                    //out << ptLN->vCenter.at(j).at(i).x << " " << ptLN->vCenter.at(j).at(i).y
                    //  << " ";
                }
            }
            out << endl;
        }

        out.close();

        done = true;
    }
    return done;
}

//void CRTree::growTree(vector<vector<CPatch> > &TrainSet, int node , int depth, float pnratio, CConfig conf, boost::mt19937 gen, const std::vector<int> &defaultClass_){
void CRTree::growTree(std::vector<CPosPatch> &posPatch, std::vector<CNegPatch> &negPatch, int node, int depth, float pnratio, CConfig conf, const std::vector<int> &defaultClass_){


    // for(int i = 0; i < posPatch.size(); ++i){
    //   std::cout << "this is for debug growtree trainset center " << posPatch.at(i).center << std::endl;
    // }]]

    CTrainSet trainSet(posPatch,negPatch);

    config = conf;
    std::cout << "learning mode is " << conf.learningMode << std::endl;

    // int dummy;
    // std::cin >> dummy;

    boost::uniform_int<> zeroOne( 0, 1 );
    boost::variate_generator<boost::lagged_fibonacci1279&,
            boost::uniform_int<> > rand( genTree, zeroOne );

    int totalClassNum = 0;

    defaultClass = defaultClass_;
    nclass = defaultClass.size();

    containClass.clear();
    containClass.resize(nclass);
    for(int i = 0; i < posPatch.size(); ++i)
        containClass.at(classDatabase.search(posPatch.at(i).getClassName()))++;

    int remainClass = 0;
    for(int c = 0; c < nclass; ++c)
        if(containClass.at(c) != 0)
            remainClass++;

    int maxClassNum = 0;

    for(int i = 0; i < containClass.size(); ++i){
        totalClassNum += containClass.at(i);
        if(maxClassNum < containClass.at(i)){
            maxClassNum = containClass.at(i);
        }
    }

    float classRatio  = 0;
    if(maxClassNum != 0 && totalClassNum !=0){
        classRatio = (float)maxClassNum / (float)totalClassNum;
        std::cout << "classRatio is " << classRatio << std::endl;
    }

    float negratio = (float)negPatch.size() / (float)(posPatch.size() + negPatch.size());
    std::cout << "pos patch " << posPatch.size() << " neg patch " << negPatch.size() << " neg ratio " << negratio <<std::endl;

    int measureMode = 1;
    if(classRatio > 0.95)
        measureMode = 0;

    std::cout << "measure mode " << measureMode << std::endl;

    if(depth < max_depth && negratio < 0.95 && posPatch.size() > conf.min_sample) {
        // spilit patches by the binary test
        CTrainSet SetA;
        CTrainSet SetB;
        int test[10];

        // Set measure mode for split: 0 - classification, 1 - regression
        unsigned int measure_mode = 1;
        //if( float(TrainSet[1].size()) / float(TrainSet[0].size()+TrainSet[1].size()) >= 0.05 &&
        if(depth < max_depth-2 )
            measure_mode = rand();

        //cout << "MeasureMode " << measure_mode << "depth " << depth << "Pos patches " << TrainSet[0].size() << " Neg Patches " << TrainSet[1].size() << " pnratio " << pnratio  <<endl;

        cout << "Node num: " << node << endl;
        for(int i = 0; i < nclass; ++i){
            std::cout << "class" << i << " : " << containClass.at(i) << endl;
        }

        // Find optimal test
        if( optimizeTest(SetA, SetB, trainSet, test, 1000, measureMode, depth) ) {

            // Store binary test for current node
            int* ptT = &treetable[node*11];
            std::cout << node << std::endl;
            ptT[0] = -1;
            ++ptT;
            for(int t=0; t<10; ++t)
                ptT[t] = test[t];

            double countA = 0;
            double countB = 0;

            containClassA.clear();
            containClassA.resize(nclass);
            containClassB.clear();
            containClassB.resize(nclass);

            for(int l = 0; l < SetA.posPatch.size(); ++l)
                containClassA.at(classDatabase.search(SetA.posPatch.at(l).getClassName()))++;
            for(int l = 0; l < SetB.posPatch.size(); ++l)
                containClassB.at(classDatabase.search(SetB.posPatch.at(l).getClassName()))++;

            for(int l = 0; l < nclass; ++l)std:cout << "class" << l << " is splitted " << containClassA.at(l) << " " << containClassB.at(l) << std::endl;

            std::cout << "negpatch splitted by " << SetA.negPatch.size() << " " << SetB.negPatch.size() << std::endl;
            // Go left
            // If enough patches are left continue growing else stop
            if(SetA.posPatch.size()+SetA.negPatch.size()>min_samples) {
                growTree(SetA.posPatch,SetA.negPatch, 2*node+1, depth+1, pnratio, conf, defaultClass_);
            } else {
                makeLeaf(SetA, pnratio, 2*node+1);
            }

            // Go right
            // If enough patches are left continue growing else stop
            if(SetB.posPatch.size()+SetB.negPatch.size()>min_samples) {
                growTree(SetB.posPatch,SetB.negPatch, 2*node+2, depth+1, pnratio, conf, defaultClass_);
            } else {
                makeLeaf(SetB, pnratio, 2*node+2);
            }
        } else {
            // Could not find split (only invalid one leave split)
            makeLeaf(trainSet, pnratio, node);
        }

    } else {

        // Only negative patches are left or maximum depth is reached
        makeLeaf(trainSet, pnratio, node);
    }
}

// Create leaf node from patches 
void CRTree::makeLeaf(CTrainSet &trainSet, float pnratio, int node) {
    // Get pointer
    treetable[node*11] = num_leaf;
    LeafNode* ptL = &leaf[num_leaf];

    // divide reached patch to each class
    patchPerClass.clear();
    patchPerClass.resize(classDatabase.vNode.size());
    for(int c = 0; c < nclass; ++c)
        patchPerClass.at(c).clear();

    for(int i = 0; i < trainSet.posPatch.size(); ++i){
        //std::cout << "this is for debug trainset center is " << posPatch.at(i).center << std::endl;
        patchPerClass.at(classDatabase.search(trainSet.posPatch.at(i).getClassName())).push_back(trainSet.posPatch.at(i));
    }

    // calc total default patch num
    int totalPatchNum = 0;
    for(int c = 0; c < nclass; ++c)
        totalPatchNum += defaultClass.at(c);

    ptL->pfg.resize(nclass);
    for(int k = 0; k < nclass; ++k){
        if(patchPerClass.at(k).size() != 0){
            //int totalnum = reachedClass.at(k);

            float maxOtherRatio = (float)defaultClass.at(k) / ((float)(1.0 +  config.pnRatio) * (float)totalPatchNum - (float)defaultClass.at(k));

            // Store data
            ptL->pfg.at(k) = (float)patchPerClass.at(k).size() / (maxOtherRatio * (float)(trainSet.posPatch.size() - patchPerClass.at(k).size() + trainSet.negPatch.size()) + (float)patchPerClass.at(k).size());
        }else{
            ptL->pfg.at(k) = 0;
        }
    }

    // set each center point
    ptL->vCenter.resize(nclass);
    ptL->param.resize(nclass);
    for(int i = 0; i < nclass; ++i){
        ptL->vCenter.at(i).clear();
        ptL->param.at(i).clear();
        for(int j = 0; j < patchPerClass.at(i).size(); ++j){
            //ptL->vCenter.at(i).push_back(patchPerClass.at(i).at(j).getCenterPoint());
            CParamset tParam;
            tParam = patchPerClass.at(i).at(j).getParam();

            cv::Point tCenterPoint = tParam.getCenterPoint();

            tCenterPoint.x -= patchPerClass.at(i).at(j).getRoi().x;
            tCenterPoint.x -= (patchPerClass.at(i).at(j).getRoi().width) / 2;

            tCenterPoint.y -= patchPerClass.at(i).at(j).getRoi().y;
            tCenterPoint.y -= (patchPerClass.at(i).at(j).getRoi().height) / 2;

            tParam.setCenterPoint(tCenterPoint);

            ptL->param.at(i).push_back(tParam);
            //if(!patchPerClass.at(i).empty())
            //std::cout << patchPerClass.at(i).at(0).center << std::endl;
        }
    }

    //for(int i = 0;i < trainSet.posPatch.size(); ++i)
    //  ptL->param.push_back(trainSet.posPatch.at(i).getParam());

    // Increase leaf counter
    ++num_leaf;
}

bool CRTree::optimizeTest(CTrainSet &SetA, CTrainSet &SetB, const CTrainSet &trainSet, int* test, unsigned int iter, unsigned int measure_mode, int depth) {

    bool found = false;

    // temporary data for split into Set A and Set B
    CTrainSet tmpA;
    CTrainSet tmpB;

    //std::cout << "Trainset size" << TrainSet.size() << std::endl;

    // temporary data for finding best test
    std::vector<std::vector<IntIndex> > valSet(2);
    double tmpDist;
    // maximize!!!!
    double bestDist = -DBL_MAX;
    int tmpTest[10];

    boost::uniform_int<> dst( 0, INT_MAX );
    boost::variate_generator<boost::lagged_fibonacci1279&,
            boost::uniform_int<> > rand( genTree, dst );


    std::cout << "finding best test!" << std::endl;

    // Find best test of ITER iterations
    for(unsigned int i =0; i<iter; ++i) {

        // reset temporary data for split
        tmpA.posPatch.clear();
        tmpA.negPatch.clear();
        tmpB.posPatch.clear();
        tmpB.negPatch.clear();

        // generate binary test without threshold

        //std::cout << posPatch.at(0).patch.size() << std::endl;

        //std::cout << SetA.size() << std::endl;

        generateTest(&tmpTest[0],
                config.p_width,//posPatch.at(0).patchRoi.width,
                config.p_height,//posPatch.at(0).patchRoi.height,
                trainSet.posPatch.at(0).getFeatureNum(),
                depth);
        //std::cout << tmpTest[0] << " " << tmpTest[1] << " " << tmpTest[2] << std::endl;


        //for(int q = 0; q < 9; ++q)
        //cout << tmpTest[q] << " ";
        //cout << endl;


        // compute value for each patch
        evaluateTest(valSet, &tmpTest[0], trainSet);

        // for(int l = 0; l < valSet.at(0).size(); ++l){
        //   std::cout << "val = " << valSet.at(0).at(l).val << " index = " << valSet.at(0).at(l).index << std::endl;
        // }

        // int dummy;
        // std::cin >> dummy;


        //std::cout << "evaluation end" << std::endl;

        // find min/max values for threshold
        int vmin = INT_MAX;
        int vmax = INT_MIN;
        for(unsigned int l = 0; l<2; ++l) {
            if(valSet[l].size()>0) {
                if(vmin>valSet[l].front().val)  vmin = valSet[l].front().val;
                if(vmax<valSet[l].back().val )  vmax = valSet[l].back().val;
            }
        }

        int d = vmax-vmin;

        if(d > 0) {
            // Find best threshold
            for(unsigned int j=0; j<10; ++j) {

                // Generate some random thresholds
                int tr = (rand() % (d)) + vmin;

                // // Split training data into two sets A,B accroding to threshold t
                split(tmpA, tmpB, trainSet, valSet, tr);

                // std::cout << "this is for debug " << std::endl;
                // std::cout << "setA----------------------------------------------------" << std::endl;
                // for(int j = 0; j < tmpA.at(0).size(); ++j)
                //   std::cout << tmpA.at(0).at(j).center << std::endl;

                // std::cout << "setB****************************************************" << std::endl;
                // for(int j = 0; j < tmpB.at(0).size(); ++j)
                //   std::cout << tmpB.at(0).at(j).center << std::endl;

                // //int dummy;
                // std::cin >> dummy;

                // Do not allow empty set split (all patches end up in set A or B)
                if( tmpA.posPatch.size()+tmpA.negPatch.size()>0 && tmpB.posPatch.size()+tmpB.negPatch.size()>0 ) {

                    // Measure quality of split with measure_mode 0 - classification, 1 - regression

                    tmpDist = measureSet(tmpA, tmpB, depth, measure_mode);

                    // Take binary test with best split
                    if(tmpDist>bestDist) {

                        found = true;
                        bestDist = tmpDist;
                        for(int t=0; t<10;++t) test[t] = tmpTest[t];
                        test[9] = tr;

                        //cout << "iretayo" << endl;
                        SetA = tmpA;
                        SetB = tmpB;
                    }

                }

            } // end for j

        }

        pBar(i, iter, 50);

    } // end iter

    std::cout << std::endl;

    // this is for debug
//    cv::namedWindow("test");
//    std::cout << test[8] << std::endl;

//    if(test[8] == 32){
    if(found)
        for(int i = 0; i < 8; ++i){
            std::cout << test[i] << " ";
        }
    else
        std::cout << "not found!!!!!!!!!!!!!!!!!!!!Ghaaaaaaaaaaaa!!!!!!!" << std::endl;
        std::cout << std::endl;
//        cv::Rect rect1 = cv::Rect(test[0], test[1], test[2], test[3]);
//        cv::Rect rect2 = cv::Rect(test[4], test[5], test[6], test[7]);
//        std::cout << "A" << std::endl;
//        for(int i = 0; i < SetA.posPatch.size(); ++i){
//            cv::Mat showMat(SetA.posPatch.at(i).getRoi().width, SetA.posPatch.at(i).getRoi().height,CV_8U);
//            (*SetA.posPatch.at(i).getFeature(test[8]))(SetA.posPatch.at(i).getRoi()).convertTo(showMat, CV_8U, 255.0 / 1000);
//            cv::rectangle(showMat,rect1,cv::Scalar(200,200,200));
//            cv::rectangle(showMat,rect2,cv::Scalar(200,200,200));
//            cv::imshow("test", showMat);
//            cv::waitKey(0);
//        }
//        std::cout << "B" << std::endl;
//        for(int i = 0; i < SetB.posPatch.size(); ++i){
//            cv::Mat showMat(SetB.posPatch.at(i).getRoi().width, SetB.posPatch.at(i).getRoi().height,CV_8U);
//            (*SetB.posPatch.at(i).getFeature(test[8]))(SetB.posPatch.at(i).getRoi()).convertTo(showMat, CV_8U, 255.0 / 1000);
//            cv::imshow("test", showMat);
//            cv::waitKey(0);
//        }
//    }else{

//        std::cout << "A" << std::endl;
//        for(int i = 0; i < SetA.posPatch.size(); ++i){
//            cv::imshow("test", *SetA.posPatch.at(i).getFeature(test[8]));
//            cv::waitKey(0);
//        }
//        std::cout << "B" << std::endl;
//        for(int i = 0; i < SetB.posPatch.size(); ++i){
//            cv::imshow("test", *SetB.posPatch.at(i).getFeature(test[8]));
//            cv::waitKey(0);
//        }
//    }
//    cv::destroyAllWindows();

    // return true if a valid test has been found
    // test is invalid if only splits with an empty set A or B has been created
    return found;
}

void CRTree::normarizationByDepth(const CPatch* patch,cv::Mat& rgb)const{//, const CConfig &config)const {
    cv::Mat tempDepth = *patch->getFeature(32);
    cv::Mat depth = tempDepth(patch->getRoi());

//                cv::namedWindow("test");
//                cv::imshow("test",rgb);
//                cv::waitKey(0);
//                cv::destroyAllWindows();

    //std::cout << trainSet.posPatch.at(i).getFeature(32)->at<uchar>(10,10) << std::endl;

    double widthSca, heightSca;
    widthSca = config.widthScale * (double)(depth.at<ushort>(config.p_height / 2 + 1, config.p_width / 2 + 1) + config.mindist) / (double)config.p_width;
    heightSca = config.heightScale * (double)(depth.at<ushort>(config.p_height / 2 + 1, config.p_width / 2 + 1) + config.mindist) / (double)config.p_height;

    //std::cout << "depth : " << depth.at<ushort>(config.p_height / 2 + 1, config.p_width / 2 + 1) << " widht scale : " << widthSca << " height scape : " << heightSca << std::endl;

    double realWindowWidth, realWindowHeight;

    realWindowWidth = (double)config.p_width / widthSca;
    realWindowHeight = (double)config.p_height / heightSca;

    cv::Rect roi;

    roi.x = (int)((config.p_width - realWindowHeight) / 2);
    roi.y = (int)((config.p_height - realWindowWidth) / 2);

    roi.width = (int)realWindowWidth;
    roi.height = (int)realWindowHeight;

    rgb = rgb(roi);
    cv::resize(rgb,rgb,cv::Size(config.p_width,config.p_height));


//                cv::namedWindow("test");
//                cv::imshow("test",rgb);
//                cv::waitKey(0);
//                cv::destroyAllWindows();
}

void CRTree::evaluateTest(std::vector<std::vector<IntIndex> >& valSet, const int* test, const CTrainSet &trainSet) {

    // for(int m = 0; m < 6; ++m)
    //   std::cout << test[m] << ", ";
    // std::cout << std::endl;

    int p1, p2;

    valSet.clear();
    valSet.resize(2);

    //for(unsigned int l = 0; l < TrainSet.size(); ++l)
    {
        valSet[0].resize(trainSet.posPatch.size());
        for(unsigned int i=0;i<trainSet.posPatch.size();++i) {

            // pointer to channel
            cv::Mat tempMat = *trainSet.posPatch.at(i).getFeature(test[8]);
            cv::Mat ptC = tempMat(trainSet.posPatch.at(i).getRoi());//(*(TrainSet[l][i].patch[test[8]]))(TrainSet[l][i].patchRoi);

            normarizationByDepth(&(trainSet.posPatch.at(i)) ,ptC);

            if(test[8] == 32){

                //std::cout << "yobareatyo" << std::endl;
                calcHaarLikeFeature(ptC,test,p1,p2);
                //std::cout << "this is for debug hyahhaaaaaaaaaaaaaaaaaaaa" << std::endl;
                //int dummy;
                //std::cin >> dummy;
//                p1 = 0;
//                p2 = 0;
//                for(int j = 0;j < test[2]; ++j){
//                    for(int k = 0; k < test[3]; ++k)
//                        p1 += (int)ptC.at<uchar>(k + test[1],j +  test[0]);
//                }

//                for(int j = 0;j < test[6]; ++j){
//                    for(int k = 0; k < test[7]; ++k)
//                        p2 += (int)ptC.at<uchar>(k + test[5],j +  test[4]);
//                }

                //std::cout << p1 << " " << p2 << std::endl;

            }else{
                // get pixel values
                p1 = (int)ptC.at<uchar>(test[1], test[0]);
                p2 = (int)ptC.at<uchar>(test[5], test[4]);
            }
            valSet[0][i].val = p1 - p2;
            valSet[0][i].index = i;
        }

        sort( valSet[0].begin(), valSet[0].end() , IntIndex::lessVal);
    }

    {
        valSet[1].resize(trainSet.negPatch.size());
        for(unsigned int i=0;i<trainSet.negPatch.size();++i) {

            // pointer to channel
            cv::Mat tempMat = *trainSet.negPatch.at(i).getFeature(test[8]);
            cv::Mat ptC = tempMat(trainSet.negPatch.at(i).getRoi());

            normarizationByDepth(&(trainSet.negPatch.at(i)) ,ptC);
            //cv::Mat ptC = (*(TrainSet[l][i].patch[test[8]]))(TrainSet[l][i].patchRoi);
            if(test[8] == 32){
                //std::cout << "this is for debug hyahhaaaaaaaaaaaaaaaaaaaa" << std::endl;
                //int dummy;
                //std::cin >> dummy;
//                p1 = 0;
//                p2 = 0;
//                for(int j = 0;j < test[2]; ++j){
//                    for(int k = 0; k < test[3]; ++k)
//                        p1 += (int)ptC.at<uchar>(k + test[1],j +  test[0]);
//                }

//                for(int j = 0;j < test[6]; ++j){
//                    for(int k = 0; k < test[7]; ++k)
//                        p2 += (int)ptC.at<uchar>(k + test[5],j +  test[4]);
//                }

                calcHaarLikeFeature(ptC,test,p1,p2);

            }else{
                // get pixel values
                p1 = (int)ptC.at<uchar>(test[1], test[0]);
                p2 = (int)ptC.at<uchar>(test[5], test[4]);
            }
            valSet[1][i].val = p1 - p2;
            valSet[1][i].index = i;
        }

        sort( valSet[1].begin(), valSet[1].end() , IntIndex::lessVal);
    }
}

void CRTree::split(CTrainSet& SetA, CTrainSet& SetB, const CTrainSet& trainSet, const vector<vector<IntIndex> >& valSet, int t) {
    //for(int j = 0; j < TrainSet.size(); ++j){

    SetA.posPatch.clear();
    SetA.negPatch.clear();

    SetB.posPatch.clear();
    SetB.negPatch.clear();

    // pos patch
    for(int i = 0; i < valSet.at(0).size(); ++i){
        if(valSet.at(0).at(i).val < t){
            SetA.posPatch.push_back(trainSet.posPatch.at(valSet.at(0).at(i).index));
            //SetA.posPatch.back().getCenterPoint() = trainSet.posPatch.at(valSet.at(0).at(i).index).center;
        }else{
            SetB.posPatch.push_back(trainSet.posPatch.at(valSet.at(0).at(i).index));
            //SetB.posPatch.back().center = trainSet.posPatch.at(valSet.at(0).at(i).index).center;
        }
    }

    // neg patch
    for(int i = 0; i < valSet.at(1).size(); ++i){
        if(valSet.at(1).at(i).val < t){
            SetA.negPatch.push_back(trainSet.negPatch.at(valSet.at(1).at(i).index));
            //SetA.negPatch.back().center = trainSet.negPatch.at(valSet.at(1).at(i).index).center;
        }else{
            SetB.negPatch.push_back(trainSet.negPatch.at(valSet.at(1).at(i).index));
            //SetB.negPatch.back().center = trainSet.negPatch.at(valSet.at(1).at(i).index).center;
        }
    }
    //}



    //std::cout << "pos "<< posPatch.size() << " " << SetA.posPatch.size() << " " << SetB.posPatch.size() << std::endl;
    //std::cout << "neg "<< negPatch.size() << " " << SetA.negPatch.size() << " " << SetB.negPatch.size() << std::endl;

    //int dummy;
    //std::cin >> dummy;

    // this is for debug
    // std::cout << "TrainSet-----------------------------------------------------------------------------------------------------------------" << std::endl;
    // for(int i = 0; i < posPatch.size(); ++i)
    //   std::cout << posPatch.at(i).center << std::endl;

    // std::cout << "SetA-----------------------------------------------------------------------------------------------------------------" << std::endl;
    // for(int i = 0; i < SetA.posPatch.size(); ++i)
    //   std::cout << SetA.posPatch.at(i).center << std::endl;

    // std::cout << "SetB-----------------------------------------------------------------------------------------------------------------" << std::endl;
    // for(int i = 0; i < SetB.posPatch.size(); ++i)
    //   std::cout << SetB.posPatch.at(i).center << std::endl;

    // int dummy;

    // std::cin >> dummy;

    // for(unsigned int l = 0; l<TrainSet.size(); ++l) {
    //   // search largest value such that val<t
    //   vector<IntIndex>::const_iterator it = valSet[l].begin();
    //   while(it!=valSet[l].end() && it->val<t) {
    //     ++it;
    //   }

    //   SetA[l].resize(it-valSet[l].begin());
    //   SetB[l].resize(TrainSet[l].size()-SetA[l].size());

    //   it = valSet[l].begin();
    //   for(unsigned int i=0; i<SetA[l].size(); ++i, ++it) {
    //     SetA[l][i] = TrainSet[l][it->index];
    //   }

    //   it = valSet[l].begin()+SetA[l].size();
    //   for(unsigned int i=0; i<SetB[l].size(); ++i, ++it) {
    //     SetB[l][i] = TrainSet[l][it->index];
    //   }

    // }
}

double CRTree::distMean(const std::vector<CPosPatch>& SetA, const std::vector<CPosPatch>& SetB) {
    //std::cout << "callde distMean !!!" << std::endl;

    std::vector<double> meanAx(nclass,0);
    std::vector<double> meanAy(nclass,0);
    //for(std::vector<CPatch>::const_iterator it = SetA.begin(); it != SetA.end(); ++it) {


    for(int i = 0; i < SetA.size(); ++i){
        //for(unsigned int c = 0; c<nclass; ++c) {
        meanAx[classDatabase.search(SetA.at(i).getClassName())] += SetA.at(i).getCenterPoint().x;
        meanAy[classDatabase.search(SetA.at(i).getClassName())] += SetA.at(i).getCenterPoint().y;
        //}
    }

    //for(unsigned int c = 0; c<num_cp; ++c) {
    for(int c = 0; c < nclass; ++c){
        meanAx[c] /= (double)SetA.size();
        meanAy[c] /= (double)SetA.size();
    }
    //}

    vector<double> distA(nclass,0);
    //for(std::vector<CPatch>::const_iterator it = SetA.begin(); it != SetA.end(); ++it) {
    for(int i = 0; i< SetA.size(); ++i){
        //for(unsigned int c = 0; c<num_cp; ++c) {
        double tmp = SetA.at(i).getCenterPoint().x - meanAx[classDatabase.search(SetA.at(i).getClassName())];
        distA[classDatabase.search(SetA.at(i).getClassName())] += tmp*tmp;
        tmp = SetA.at(i).getCenterPoint().y - meanAy[classDatabase.search(SetA.at(i).getClassName())];
        distA[classDatabase.search(SetA.at(i).getClassName())] += tmp*tmp;
        //}
    }

    std::vector<double> meanBx(nclass,0);
    std::vector<double> meanBy(nclass,0);
    //for(std::vector<CPatch>::const_iterator it = SetB.begin(); it != SetB.end(); ++it) {
    for(int i = 0; i < SetB.size(); ++i){
        //for(unsigned int c = 0; c<num_cp; ++c) {
        meanBx[classDatabase.search((SetB.at(i).getClassName()))] += SetB.at(i).getCenterPoint().x;
        meanBy[classDatabase.search((SetB.at(i).getClassName()))] += SetB.at(i).getCenterPoint().y;
        //}
    }

    for(int c = 0; c < nclass; c++){

        //for(unsigned int c = 0; c<num_cp; ++c) {
        meanBx[c] /= (double)SetB.size();
        meanBy[c] /= (double)SetB.size();
        //}
    }

    std::vector<double> distB(nclass,0);
    for(std::vector<CPosPatch>::const_iterator it = SetB.begin(); it != SetB.end(); ++it) {
        //for(unsigned int c = 0; c<num_cp; ++c) {
        double tmp = (*it).getCenterPoint().x - meanBx[classDatabase.search(it->getClassName())];//(*it).classNum];
        distB[classDatabase.search(it->getClassName())] += tmp*tmp;
        tmp = (*it).getCenterPoint().y - meanBy[classDatabase.search(it->getClassName())];
        distB[classDatabase.search(it->getClassName())] += tmp*tmp;
        //}
    }

    double minDist = DBL_MAX;

    for(unsigned int c = 0; c < nclass; ++c) {
        distA[c] += distB[c];
        if(distA[c] < minDist) minDist = distA[c];
    }

    return minDist/double( SetA.size() + SetB.size() );
}

double CRTree::InfGain(const CTrainSet& SetA, const CTrainSet& SetB) {
    CTrainSet set;

    //std::cout << "yobaretayo" << std::endl;

    double entoropyA = 0;
    double entoropyB = 0;
    double entoropy = 0;

    //    int maxClass = 0;
    //    int maxClassNum = 0;
    //    for(int i = 0; i < nclass; ++i)
    //        if(maxClassNum < containClass.at(i)){
    //            maxClassNum = containClass.at(i);
    //            maxClass = i;
    //        }

    for(int i = 0; i < SetA.posPatch.size(); ++i)
        set.posPatch.push_back(SetA.posPatch.at(i));

    for(int i = 0; i < SetB.posPatch.size(); ++i)
        set.posPatch.push_back(SetB.posPatch.at(i));

    for(int i = 0; i < SetA.negPatch.size(); ++i)
        set.negPatch.push_back(SetA.negPatch.at(i));

    for(int i = 0; i < SetB.negPatch.size(); ++i)
        set.negPatch.push_back(SetB.negPatch.at(i));


    //    for(int i = 0; i < nclass; ++i){
    //        entoropyA += calcEntropy(SetA.posPatch, SetA.negPatch.size(),i);
    //        entoropyB += calcEntropy(SetB.posPatch, SetB.negPatch.size(),i);
    //        entoropy += calcEntropy(set, SetA.negPatch.size() + SetB.negPatch.size(), i);
    //    }

    entoropyA = calcEntropy(SetA);
    entoropyB = calcEntropy(SetB);
    entoropy = calcEntropy(set);

    //std::cout << entoropyA << " ";

    double wa = (double)SetA.size() / (double)set.size();
    double wb = (double)SetB.size() / (double)set.size();


    //    std::cout << std::endl << "SetA pos " << SetA.posPatch.size() << " neg " << SetA.negPatch.size() << " entropy A " << entoropyA << std::endl;
    //    std::cout << "SetB pos " << SetB.posPatch.size() << " neg " << SetB.negPatch.size() << " entropy B " << entoropyB << std::endl;
    //    std::cout << "entropy " << entoropy << " IG " << entoropy - (wa * entoropyA + wb * entoropyB) << std::endl;



    //    std::cout << "positive entoropy is caliculated" << std::endl;

    //calc negative entropy
    //    for(int i = 0; i < SetA.size(); ++i){
    //        double pA = (double)SetA.at(i).size() / (double)(SetA.posPatch.size() + SetA.negPatch.size());
    //        double pB = (double)SetB.at(i).size() / (double)(SetB.posPatch.size() + SetB.negPatch.size());
    //        double pT = (double)(SetA.at(i).size() + SetB.at(i).size()) / (double)(set.size() + SetA.negPatch.size() + SetB.negPatch.size());

    //        //std::cout << pA << " " << pB << " " << pT << std::endl;

    //        if(pA != 0 && pB !=0 && pT != 0){
    //            entoropyA += -1 * pA * log(pA);
    //            entoropyB += -1 * pB * log(pB);
    //            entoropy += -1 * pT * log(pT);
    //        }
    //    }

    //std::cout << entoropyA << std::endl;


    //    std::cout << std::endl << "SetA pos " << SetA.posPatch.size() << " neg " << SetA.negPatch.size() << " entropy A " << entoropyA << std::endl;
    //    std::cout << "SetB pos " << SetB.posPatch.size() << " neg " << SetB.negPatch.size() << " entropy B " << entoropyB << std::endl;
    //    std::cout << "entropy " << entoropy << " IG " << entoropy - (wa * entoropyA + wb * entoropyB) << std::endl;

    //    std::cout << "owattayo" << std::endl;

    return entoropy - (wa * entoropyA + wb * entoropyB);
}

double CRTree::calcEntropy(const CTrainSet &set)
{
    double entropy = 0;
    for(int maxClass = 0; maxClass < classDatabase.vNode.size(); ++maxClass){
        int maxClassNum = 0;
        int otherClass = 0;

        for(int i = 0; i < set.posPatch.size(); ++i)
            if(classDatabase.search(set.posPatch.at(i).getClassName()) == maxClass)
                maxClassNum++;
        if(maxClassNum > 0 && maxClassNum != set.size()){
            //std::cout << maxClassNum << std::endl;

            otherClass = set.size() - maxClassNum;

            double p = (double)maxClassNum / (double)set.size();
            entropy += -1 * p * log(p);

            p = (double)otherClass / (double)set.size();
            entropy += -1 * p * log(p);

            //std::cout << "keisan shitayo" << std::endl;
        }
    }

    double p = (double)set.negPatch.size() / (double)set.size();
    entropy += -1 * p * log(p);

    p = (double)set.posPatch.size() / (double)set.size();
    entropy += -1 * p * log(p);

    return entropy;
}
/////////////////////// IO functions /////////////////////////////

void LeafNode::show(int delay, int width, int height) {
    char buffer[200];

    print();



    if(vCenter.size()>0) {
        vector<IplImage*> iShow(vCenter.size());
        for(unsigned int c = 0; c < iShow.size(); ++c) {
            iShow[c] = cvCreateImage( cvSize(width,height), IPL_DEPTH_8U , 1 );
            cvSetZero( iShow[c] );
            for(unsigned int i = 0; i<vCenter[c].size(); ++i) {
                int y = height/2+vCenter[c][i].y;
                int x = width/2+vCenter[c][i].x;

                if(x>=0 && y>=0 && x<width && y<height)
                    cvSetReal2D( iShow[c],  y,  x, 255 );
            }
            sprintf(buffer,"Leaf%d",c);
            cvNamedWindow(buffer,1);
            cvShowImage(buffer, iShow[c]);
        }

        cvWaitKey(delay);

        for(unsigned int c = 0; c<iShow.size(); ++c) {
            sprintf(buffer,"Leaf%d",c);
            cvDestroyWindow(buffer);
            cvReleaseImage(&iShow[c]);
        }
    }
}
