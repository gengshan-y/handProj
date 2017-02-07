#include "caffePose.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

void PoseMachine::parseRes(const float* source, float* dest, int jointNum) {
  for (int j = 0; j < 46; j++) {
    for (int i = 0; i < 46; i++) {
      dest[46*j+i] += source[46*46*jointNum+46*j+i];
    }
  }
}

void PoseMachine::EstimateImgPara(vector<cv::Mat>& imgVec, 
                                  vector<cv::Rect> rectVec) {
    unsigned int batchSize = imgVec.size();
    net_->blob_by_name("data")->Reshape(batchSize, 4, 368, 368);  // fit net input
    data_buf = (float*) calloc(368*368*4*batchSize, sizeof(float));
    const float* data_res;  // net output pointer
    float *dataAggre;  // init with 0??
    float *resLHand;  // aggregated net output for left hand
    float *resRHand;  // aggregated net output for right hand

    dataAggre = (float*) calloc(batchSize*46*46*15, sizeof(float));
    resLHand = (float*) calloc(batchSize*46*46, sizeof(float));
    resRHand = (float*) calloc(batchSize*46*46, sizeof(float));

    cv::Mat img, padImg;  // volatile image
    vector<int> top, down, left, right;
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

    /* Form a batch */
    for (int scaleIt = 0; scaleIt < arraySize; scaleIt++) {
        top.clear();  // clear for a new scale
        down.clear();
        left.clear();
        right.clear();
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
            top.push_back(184 - center_y);
            down.push_back(368 - img.rows - top[it]);
            left.push_back(184 - center_x);
            right.push_back(368 - img.cols - left[it]);
            // std::cout << top[it] << " " << down[it] << " " << left[it] 
            //           << " " << right[it] << std::endl;
            dataPad(img, padImg, top[it], down[it], left[it], right[it]);
            
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

    
    /* get hand position */  // 15 is the background
    for (unsigned int it = 0; it < batchSize; it++) {
        parseRes(data_res + it *15*46*46, resLHand + it*46*46, 4);
    }

    for (unsigned int it = 0; it < batchSize; it++) {
        parseRes(data_res + it *15*46*46, resRHand + it*46*46, 7);
    }

  /*
  std::cout << "k=" << jointNum << std::endl;
  cv::Mat resShow(46, 46, CV_32FC1, dest);
  cv::resize(resShow.t(), resShow, cv::Size(368, 368));
  cv::imshow("a", resShow);
  cv::imshow("ori", img);
  cv::waitKey(0);
  */

  // filter esitmation points using likelihood
  for (unsigned int it = 0; it < batchSize; it++) {
    std::cout << "@@pose for image " << it << std::endl;
    getJointPos(it, resLHand, left[it], top[it], scale, imgVec[it]);
  }
}

void PoseMachine::getJointPos(int imgNum, float *resJoint, 
                              int left, int top, float scale, cv::Mat& img) {
  // prepare index for sorting
  unsigned int N = 46*46;
  vector<int> resIdx(N);
  for(unsigned int it = 0; it < N; it++) {
    resIdx[it] = it;
  }

  // sort in reverse order
  sort(resIdx.begin(), resIdx.end(), [&](int x, int y){
   return resJoint[x+imgNum*46*46] > resJoint[y+imgNum*46*46];
  });

/* use threshold
  // get position in the heatmap
  vector<std::tuple<int, int, float>> respVec;
  for (auto it=resIdx.begin(); it!=resIdx.end() && resJoint[*it] > 0.5; ++it) {
    std::tuple<int, int, float> respPoint((*it)/46, (*it)%46, resJoint[*it + imgNum*46*46]);
    respVec.push_back(respPoint);
  }

  // get the original mapping
  for (auto it = respVec.begin(); it != respVec.end(); it++) {
    int xPoint = (-left+ 8*std::get<0>(*it)) / scale;
    int yPoint = (-top + 8*std::get<1>(*it)) / scale;
    float pPoint = std::get<2>(*it);
    std::cout << "(" << xPoint << ", " << yPoint << ") p=" << pPoint << std::endl;
    cv::circle(img, cv::Point(xPoint, yPoint), 1, cv::Scalar(0, 255, 0),int(pPoint*10));
  }
*/

  int xPoint = (-left+ 8*(resIdx[0]/46)) / scale;
  int yPoint = (-top + 8*(resIdx[0]%46)) / scale;
  float pPoint = resJoint[resIdx[0] + imgNum*46*46];
  std::cout << "(" << xPoint << ", " << yPoint << ") p=" << pPoint << std::endl;
  // cv::circle(img, cv::Point(xPoint, yPoint), 1, cv::Scalar(0, 255, 0),int(pPoint*10));
  cv::circle(img, cv::Point(xPoint, yPoint), 1, cv::Scalar(0, 255, 0), 5);
}


void PoseMachine::Preprocess(cv::Mat img, float* data_buf) {
    img.convertTo(img, CV_32FC3, 1.0/255);
    cv::subtract(img, 0.5, img);
    
    img = img.t();    

    /* 
    cv::imshow("before pose estim", img);
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


void PoseMachine::dataSum(float* dataAggre, const float* data_res) {
    for (int i = 0; i < 15*46*46; i++) {
        dataAggre[i] += data_res[i];
    }
}


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
