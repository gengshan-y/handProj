#include "caffePose.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Parse pose estimation results */
void PoseMachine::parseRes(const float* data_res, float* res) {
    for (int k = 0; k < 14; k++) {  // 15 is the background
        /*
        if (k != 7 && k != 4) {  // 4 and 7 for l/r wrists
            continue;
        }
        */
        for (int j = 0; j < 46; j++) {
            for (int i = 0; i < 46; i++) {
                res[46*j+i] += data_res[46*46*k+46*j+i];
                // res[46*j+i] = data_res[46*46*k+46*j+i];  // to display parts
            }
        }
        /* 
        std::cout << "k=" << k << std::endl;
        cv::Mat resShow(46, 46, CV_32FC1, res);
        cv::resize(resShow.t(), resShow, cv::Size(368, 368));
        cv::imshow("a", resShow);
        cv::imshow("ori", img);
        cv::waitKey(0);
        */
    }
}

void PoseMachine::EstimateImgPara(vector<cv::Mat>& imgVec, 
                                  vector<cv::Rect> rectVec) {
    unsigned int batchSize = imgVec.size();
    net_->blob_by_name("data")->Reshape(batchSize, 4, 368, 368);  // fit net input
    data_buf = (float*) calloc(368*368*4*batchSize, sizeof(float));
    const float* data_res;  // net output pointer
    float *dataAggre;  // init with 0??
    float *res;  // aggregated net output

    dataAggre = (float*) calloc(batchSize*46*46*15, sizeof(float));
    res = (float*) calloc(batchSize*46*46, sizeof(float));

    cv::Mat img, padImg;  // volatile image
    int top, down, left, right;
    float scale;

    int arraySize = 2;
    float sizeArray[2] = {600, 900};

    //int arraySize = 2;
    //float sizeArray[2] = {200, 500};
    // float sizeArray[5] = {200, 270, 360, 450, 500};

    /* for 1280*720 images
    int arraySize = 10;
    float sizeArray[10] = {270, 370, 470, 570, 670, 770, 870, 970, 1070, 1170};
    */
    for (int scaleIt = 0; scaleIt < arraySize; scaleIt++) {
        for(unsigned int it = 0; it < imgVec.size(); it++) {
            cv::Rect rect = rectVec[it];
            // scaling
            scale = sizeArray[scaleIt] / max(imgVec[0].cols, imgVec[0].rows);
            // std::cout << "using a scale = " << scale << std::endl;
            cv::resize(imgVec[it], img, cv::Size(int(imgVec[0].cols*scale),
                                             int(imgVec[0].rows*scale)));

            // padding 
            int center_x = int((rect.x + rect.width/2) * scale);
            int center_y = int((rect.y + rect.height/2) * scale);
            top = 184 - center_y;
            down = 368 - img.rows - top;
            left = 184 - center_x;
            right = 368 - img.cols - left;
            // std::cout << top << " " << down << " " << left << " " << right << std::endl;
            dataPad(img, padImg, top, down, left, right);
            
            Preprocess(padImg, data_buf+it*4*368*368);
        }
        net_->blob_by_name("data")->set_cpu_data(data_buf);  // should be here
        net_->ForwardFrom(0);
        data_res = net_->blob_by_name("Mconv5_stage6")->cpu_data();
        for (unsigned int it = 0; it < imgVec.size(); it++) {
            dataSum(dataAggre+it*46*46*15, data_res+it*46*46*15);
        }
    }
    /*  other stages
    data_res = net_->blob_by_name("conv7_stage1")->cpu_data();
    data_res = net_->blob_by_name("Mconv5_stage2")->cpu_data();    
    data_res = net_->blob_by_name("Mconv5_stage3")->cpu_data();
    data_res = net_->blob_by_name("Mconv5_stage4")->cpu_data();
    data_res = net_->blob_by_name("Mconv5_stage5")->cpu_data();
    */

    /* average */
    for (unsigned int i = 0; i < batchSize*15*46*46; i++) {
        dataAggre[i] /= arraySize;
    }

    /* get hand position */
    for (unsigned int it = 0; it < batchSize; it++) {
        parseRes(data_res + it *15*46*46, res + it*46*46);
    }

    // filter esitmation points using likelihood
    for (unsigned int i = 0; i < batchSize; i++) {
        unsigned int N = 46*46;
        vector<int> resIdx(N);
        for(unsigned int it = 0; it < N; it++) {
            resIdx[it] = it;
        }
        sort(resIdx.begin(), resIdx.end(),
             [&](int x, int y){return res[x + i*46*46] > res[y+i*46*46];});  // sort in reverse order

        // print out content:
        vector<std::tuple<int, int, float>> respVec;
        for (auto it=resIdx.begin(); it!=resIdx.end() && res[*it] > 0.4; ++it) {
            std::tuple<int, int, float> respPoint((*it)/46, (*it)%46, res[*it + i*46*46]);
            respVec.push_back(respPoint);
        }

        std::cout << "@@pose for image " << i << std::endl;
        for (auto it = respVec.begin(); it != respVec.end(); it++) {
            int xPoint = (-left + 8*std::get<0>(*it)) / scale;
            int yPoint = (-top + 8*std::get<1>(*it)) / scale;
            float pPoint = std::get<2>(*it);
            std::cout << "(" << xPoint << ", " << yPoint << ") p=" << pPoint << std::endl;
            cv::circle(imgVec[i], cv::Point(xPoint, yPoint), 1, cv::Scalar(0, 255, 0),int(pPoint*10));
        }
    }
}

/** Estimate pose using the given model, withing the bounding box.
 */
void PoseMachine::EstimateImg(cv::Mat& cv_img, cv::Rect rect) {
    const float* data_res;  // net output pointer
    float dataAggre[46*46*15] = {0};  // init with 0
    float res[46*46] = {0};  // aggregated net output
    
    cv::Mat img, padImg;  // volatile image
    int top, down, left, right;
    float scale;

    int arraySize = 1;
    float sizeArray[1] = {500};
 
    //int arraySize = 2;
    //float sizeArray[2] = {200, 500};
    // float sizeArray[5] = {200, 270, 360, 450, 500};
 
    /* for 1280*720 images
    int arraySize = 10;
    float sizeArray[10] = {270, 370, 470, 570, 670, 770, 870, 970, 1070, 1170};
    */

    for (int scaleIt = 0; scaleIt < arraySize; scaleIt++) {
        // scaling
        scale = sizeArray[scaleIt] / max(cv_img.cols, cv_img.rows);
        // std::cout << "using a scale = " << scale << std::endl;
        cv::resize(cv_img, img, cv::Size(int(cv_img.cols*scale), 
                                         int(cv_img.rows*scale)));

        // padding 
        int center_x = int((rect.x + rect.width/2) * scale);
        int center_y = int((rect.y + rect.height/2) * scale);
        top = 184 - center_y;
        down = 368 - img.rows - top;
        left = 184 - center_x;
        right = 368 - img.cols - left;
        // std::cout << top << " " << down << " " << left << " " << right << std::endl;
        dataPad(img, padImg, top, down, left, right);

        Preprocess(padImg, data_buf);

        net_->blob_by_name("data")->set_cpu_data(data_buf);  // should be here
        net_->ForwardFrom(0);
        data_res = net_->blob_by_name("Mconv5_stage6")->cpu_data();
        dataSum(dataAggre, data_res);
    }
        
    /*  other stages
    data_res = net_->blob_by_name("conv7_stage1")->cpu_data();
    data_res = net_->blob_by_name("Mconv5_stage2")->cpu_data();    
    data_res = net_->blob_by_name("Mconv5_stage3")->cpu_data();
    data_res = net_->blob_by_name("Mconv5_stage4")->cpu_data();
    data_res = net_->blob_by_name("Mconv5_stage5")->cpu_data();
    */

    /* average */
    for (int i = 0; i < 15*46*46; i++) {
        dataAggre[i] /= arraySize;
    }

    /* get hand position */
    for (int k = 0; k < 14; k++) {  // 15 is the background
        if (k != 7 && k != 4) {  // 4 and 7 for l/r wrists
            continue;
        }
        for (int j = 0; j < 46; j++) {
            for (int i = 0; i < 46; i++) {
                res[46*j+i] += data_res[46*46*k+46*j+i];
                // res[46*j+i] = data_res[46*46*k+46*j+i];  // to display parts
            }
        }
        /* 
        std::cout << "k=" << k << std::endl;
        cv::Mat resShow(46, 46, CV_32FC1, res);
        cv::resize(resShow.t(), resShow, cv::Size(368, 368));
        cv::imshow("a", resShow);
        cv::imshow("ori", img);
        cv::waitKey(0);
        */
    }
 
    // idx = findMax(res)
    unsigned int N = sizeof res / sizeof res[0];
    vector<int> resIdx(N);
    for(unsigned int it = 0; it < N; it++) {
        resIdx[it] = it;
    }
    sort(resIdx.begin(), resIdx.end(), [&](int x, int y){return res[x] > res[y];});  // sort in reverse order
    
    // print out content:
    vector<std::tuple<int, int, float>> respVec;
    for (auto it=resIdx.begin(); it!=resIdx.end() && res[*it] > 0.5; ++it) {
        std::tuple<int, int, float> respPoint((*it)/46, (*it)%46, res[*it]);
        respVec.push_back(respPoint);
    }

    for (auto it = respVec.begin(); it != respVec.end(); it++) {
        int xPoint = (-left + 8*std::get<0>(*it)) / scale;
        int yPoint = (-top + 8*std::get<1>(*it)) / scale;
        float pPoint = std::get<2>(*it);
        std::cout << xPoint << ", " << yPoint << ", " << pPoint << std::endl;
        cv::circle(cv_img, cv::Point(xPoint, yPoint), 1, cv::Scalar(0, 255, 0),int(pPoint*10));
    }
}



/** do preprocessing as in matlab code, include subtract, transpose operation
 */
void PoseMachine::Preprocess(cv::Mat img, float* data_buf) {
    img.convertTo(img, CV_32FC3, 1.0/255);
    cv::subtract(img, 0.5, img);
    
    img = img.t();    

    /*
    cv::imshow("demo", img);
    cv::waitKey(0);
    */

    /*  build center image */
    cv::Mat centerImg(img.rows, img.cols, CV_32FC1, cv::Scalar(0));
    for (int h = 0; h < centerImg.rows; h++) {
        for (int w = 0; w < centerImg.cols; w++) {
            float tmp = std::exp(float((h-img.rows/2)*(h-img.rows/2) + (w-img.cols/2)*(w-img.cols/2)) /-2./21./21.);
            centerImg.at<float>(cv::Point(w, h)) = tmp;
        }
    }
    // std::cout <<centerImg << std::endl;
    // cv::imshow(" ", img);
    // cv::waitKey(0);

    cv::Mat im_out(img.rows, img.cols, CV_32FC4, cv::Scalar(0));
    for (int h = 0; h < im_out.rows; h++) {
        for (int w = 0; w < im_out.cols; w++) {
            data_buf[(0*368+h)*368+w] = img.at<cv::Vec3f>(cv::Point(w, h))[0];
            data_buf[(1*368+h)*368+w] = img.at<cv::Vec3f>(cv::Point(w, h))[1];
            data_buf[(2*368+h)*368+w] = img.at<cv::Vec3f>(cv::Point(w, h))[2];
            data_buf[(3*368+h)*368+w] = centerImg.at<float>(cv::Point(w, h)); 
        }
    }
}


/** Add data_res to dataAggre
 */
void PoseMachine::dataSum(float* dataAggre, const float* data_res) {
    for (int i = 0; i < 15*46*46; i++) {
        dataAggre[i] += data_res[i];
    }
}


/** Pad image tmp with given margins
 */
void PoseMachine::dataPad(cv::Mat tmp, cv::Mat& padImg, int top, int down,
                            int left, int right) {  // Mat& will change outside
    // padding
    if(top < 0) {
        tmp(cv::Range(-top, tmp.rows), cv::Range(0, tmp.cols)).copyTo(tmp); 
        top = 0;
    }
    if(down < 0) {
        tmp(cv::Range(0, tmp.rows+down), cv::Range(0, tmp.cols)).copyTo(tmp);
        down = 0;
    }
    if(left < 0) {
        tmp(cv::Range(0, tmp.rows), cv::Range(-left, tmp.cols)).copyTo(tmp);
        left = 0;
    }   
    if(right < 0) {
        tmp(cv::Range(0, tmp.rows), cv::Range(0, tmp.cols+right)).copyTo(tmp); 
        right = 0;
    }

    cv::copyMakeBorder(tmp, tmp, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));
    tmp.copyTo(padImg);
}


void PoseMachine::Estimate(const string& im_name, cv::Rect rect) {
    cv::Mat cv_img = cv::imread(im_name);
    EstimateImg(cv_img, rect);
    cv::imshow("demo", cv_img);
    cv::waitKey(0);
}
