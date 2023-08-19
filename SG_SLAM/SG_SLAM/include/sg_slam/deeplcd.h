//
//  Created by Lucifer on 2022/7/19.
//

#ifndef _SG_SLAM_DEEP_LCD_H_
#define _SG_SLAM_DEEP_LCD_H_

#include <Eigen/Core>  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#include <memory>  // 参见 sg_slam/frame.h

#include "caffe/caffe.hpp"

// #include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

namespace sg_slam {

/**
 * @brief Deep Loop Closure Detector
 * 
 */
class DeepLCD
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  // 参见 sg_slam/frame.h
    using this_type = DeepLCD;  // 本类别名;
    using Ptr = std::shared_ptr<DeepLCD>;  // 本类共享/弱指针别名

    using DescrVector = Eigen::Matrix<float, 1064, 1>;

    // N.B.: 改为 std::shared_ptr 会不会更好呢? 
    caffe::Net<float>* autoencoder;  // The deploy(部署) autoencoder
    caffe::Blob<float>* autoencoder_input;  // The encoder's input blob
    caffe::Blob<float>* autoencoder_output;  // The encoder's output blob

    // If gpu_id is -1, the cpu will be used
    // DeepLCD(const std::string& network_definition_file="calc_model/deploy.prototxt", const std::string& pre_trained_model_file="calc_model/calc.caffemodel", int gpu_id=-1);
    DeepLCD(int gpu_id=-1);

    ~DeepLCD() {
        delete autoencoder;
        // delete autoencoder_input;   // N.B.: My add, 若会导致段错误就删去
        // delete autoencoder_output;  // N.B.: My add, 若会导致段错误就删去
    }

    const float score(const DescrVector& d1, const DescrVector& d2);

    DescrVector calcDescrOriginalImg(const cv::Mat& originalImg);
    const DescrVector calcDescr(const cv::Mat& im_);  // make a forward pass through the net, return the descriptor.
};  // end class DeepLCD


}  // namespace sg_slam

#endif  // _SG_SLAM_DEEP_LCD_H_
