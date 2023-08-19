//
//  Created by Lucifer on 2022/7/11.
//

#include "sg_slam/KITTI_dataset.h"

#include <fstream>
#include <iostream>
#include <string>
#include <glog/logging.h>  // 参见 src/sg_slam/config.cpp
#include <boost/format.hpp>  // 格式化字符串

// #include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include "sg_slam/config.h"

namespace sg_slam {

// --------------------------------------------------------------------------------------------------------------
KITTI_Dataset::KITTI_Dataset(const std::string &dataset_path) 
    : dataset_path_(dataset_path) {
    // current_image_index_ = 0;
    // m_scale_ = 1.0;
}

// --------------------------------------------------------------------------------------------------------------
bool KITTI_Dataset::Init(double scale) {
    // read camera intrinsics and extrinsics
    m_scale_ = scale;

    std::ifstream fin(dataset_path_ + "/calib.txt");
    if(!fin) {
        google::InitGoogleLogging("KITTI_Dataset");
        google::SetLogDestination(google::GLOG_ERROR, "./log/KITTI_Dataset/");
        LOG(ERROR) << "can't find" << dataset_path_ << "/calib.txt! ";
        google::ShutdownGoogleLogging();
        return false;
    }

    // google::InitGoogleLogging("KITTI_Dataset");
    // google::SetLogDestination(google::GLOG_INFO, "./log/KITTI_Dataset/");
    for(int i=0; i<4; i++) {
        char camera_name[3]; // 相机名字如"P0:", 为三个字符
        for(int k=0; k<3; k++) {
            fin >> camera_name[k];
        }

        double projection_data[12];
        for(int k=0; k<12; k++) {
            fin >> projection_data[k];
        }

        Mat33 K;
        K << projection_data[0], projection_data[1], projection_data[2],
             projection_data[4], projection_data[5], projection_data[6],
             projection_data[8], projection_data[9], projection_data[10];
        Vec3 t;
        Vec3 k_t;
        k_t << projection_data[3], projection_data[7], projection_data[11];
        // t << projection_data[3], projection_data[7], projection_data[11];
        t = K.inverse() * k_t;
        // K = K * 0.5;
        K = K * m_scale_;
        k_t = k_t * m_scale_;

        Mat34 KT;
        KT << K, k_t;

        // N.B.: My Add 为了读取去畸变参数
        cv::Mat DistCoef(4, 1, CV_32F);
        DistCoef.at<float>(0) = Config::Get<float>(static_cast<std::string>("camera.k1"));
        DistCoef.at<float>(1) = Config::Get<float>(static_cast<std::string>("camera.k2"));
        DistCoef.at<float>(2) = Config::Get<float>(static_cast<std::string>("camera.p1"));
        DistCoef.at<float>(3) = Config::Get<float>(static_cast<std::string>("camera.p2"));
        const float k3 = Config::Get<float>(static_cast<std::string>("camera.k3"));
        if(k3!=0) {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }

        // Camera::Ptr new_camera( new Camera(K(0, 0), K(1, 1), K(0, 2), K(1, 2), t.norm(), SE3(SO3(), t)) );
        Camera::Ptr new_camera( new Camera(K(0, 0), K(1, 1), K(0, 2), K(1, 2), t.norm(), k_t.norm(), SE3(SO3(), t), KT) );
        DistCoef.copyTo(new_camera->m_distcoef_);
        cameras_.push_back(new_camera);
        
        // LOG(INFO) << "Camera " << i << " extinsics: " << k_t.transpose();
        // LOG(INFO) << "Camera " << i << " extinsics: " << t.transpose();
        // LOG(INFO) << "Camera " << i << " discoef: " << new_camera->m_distcoef_;

    }
    // google::ShutdownGoogleLogging();
    fin.close();

    // test 
    /**
    std::cout << std::endl;
    for(size_t j=0; j < cameras_.size(); j++) {
        std::cout << "(KITTI_Dataset::Init): cameras_" << j << " : " << " bf = " << cameras_[j]->baseline_ * cameras_[j]->fx_ << std::endl;
        std::cout << "(KITTI_Dataset::Init): cameras_" << j << " : " << " kf = " << cameras_[j]->K_baseline_ << std::endl;
        std::cout << "(KITTI_Dataset::Init): cameras_" << j << " : " << " bf = " << cameras_[j]->fx_baseline_ << std::endl;
    }
    
    std::cout << std::endl;
    **/

    // timestamps
    std::cout << "vTimestamps_: " << vTimestamps_.size() << std::endl;
    std::ifstream fTimes(dataset_path_ + "/times.txt");
    if(!fTimes) {
        google::InitGoogleLogging("KITTI_Dataset_times");
        google::SetLogDestination(google::GLOG_ERROR, "./log/KITTI_Dataset/");
        LOG(ERROR) << "can't find" << dataset_path_ << "/times.txt ";
        google::ShutdownGoogleLogging();
        return false;
    }
    while (!fTimes.eof()) {
        std::string s;
        std::getline(fTimes, s);
        if(!s.empty()) {
            std::stringstream ss;
            ss << s;
            double t;
            // long double t;
            ss >> t;
            vTimestamps_.push_back(t);
        }
    }
    fTimes.close();
    std::cout << "vTimestamps_: " << vTimestamps_.size() << std::endl;

    current_image_index_ = 0;
    return true;
}

// --------------------------------------------------------------------------------------------------------------
Frame::Ptr KITTI_Dataset::NextFrame() {
    // 用boost::format来格式化字符串
    boost::format boost_fmt("%s/image_%d/%06d.png");
    cv::Mat image_left, image_right;

    // read images
    image_left  = cv::imread((boost_fmt % dataset_path_ % 0 % current_image_index_).str(), cv::IMREAD_GRAYSCALE);
    image_right = cv::imread((boost_fmt % dataset_path_ % 1 % current_image_index_).str(), cv::IMREAD_GRAYSCALE);

    if(image_left.data==nullptr || image_right.data==nullptr) {
        /**
        google::InitGoogleLogging("KITTI_Dataset");
        google::SetLogDestination(google::WARNING, "./log/KITTI_Dataset/");
        LOG(WARNING) << "can't find images at index " << current_image_index_;
        google::ShutdownGoogleLogging();
        **/
        std::cout <<  "can't find images at index " << current_image_index_ << std::endl;
        return nullptr;
    }

    auto new_frame = Frame::CreateFrame();
    if(m_scale_ == 1.0) {
        new_frame->left_img_ = image_left;
        new_frame->right_img_ = image_right;
    } else {
        cv::Mat image_left_resized, image_right_resized;
        cv::resize(image_left, image_left_resized, cv::Size(), m_scale_, m_scale_, cv::INTER_NEAREST);
        cv::resize(image_right, image_right_resized, cv::Size(), m_scale_, m_scale_, cv::INTER_NEAREST);

        new_frame->left_img_ = image_left_resized;
        new_frame->right_img_ = image_right_resized;
    }

    if(cameras_[0]->m_image_cut_flag) {
        new_frame->is_use_cut_image = true;
        cv::Mat left_ROI = new_frame->left_img_(cameras_[0]->m_image_cut_select);
        new_frame->left_cut_img_ = left_ROI;
    } else {
        new_frame->is_use_cut_image = false;
    }
    if(cameras_[1]->m_image_cut_flag) {
        cv::Mat right_ROI = new_frame->right_img_(cameras_[1]->m_image_cut_select);
        new_frame->right_cut_img_ = right_ROI;
    }

    // cv::GaussianBlur(new_frame->left_img_, new_frame->left_img_, cv::Size(7, 7), 0);
    // cv::GaussianBlur(new_frame->right_img_, new_frame->right_img_, cv::Size(7, 7), 0);

    new_frame->time_stamp_ = vTimestamps_[current_image_index_];

    // ++current_image_index_;
    current_image_index_++;

    return new_frame;
}

}  // namespace sg_slam
