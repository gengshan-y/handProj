#ifndef CAFFE_POSE_HPP
#define CAFFE_POSE_HPP

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#define max(a, b) (((a)>(b)) ? (a) :(b))
#define min(a, b) (((a)<(b)) ? (a) :(b))

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

class PoseMachine {
 public:
    PoseMachine(const string& model_file, const string& trained_file) {
        net_ = boost::shared_ptr<Net<float> >(new Net<float>(model_file, 
                                                             caffe::TEST));
        net_->CopyTrainedLayersFrom(trained_file);
    }
    void Estimate(const string& im_name, cv::Rect rect);
    void EstimateImg(cv::Mat& img, cv::Rect rect);
    void EstimateImgPara(vector<cv::Mat>& imgVec, vector<cv::Rect> rectVec);

 private:
    void parseRes(const float* data_res, float* res);
    boost::shared_ptr<Net<float> > net_;
    float *data_buf;  // net input 
    PoseMachine(){}
    void Preprocess(cv::Mat img, float* data_buf);
    void dataSum(float* dataAggre, const float* data_res);
    void dataPad(cv::Mat cv_img, cv::Mat& padImg, int top, int down, int left, int right);
};

#endif  // CAFFE_POSE_HPP
