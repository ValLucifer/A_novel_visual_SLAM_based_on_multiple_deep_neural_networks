//
//  Created by Lucifer on 2022/10/8.
//

#include "sg_slam/TUM_dataset.h"

#include <fstream>
#include <iostream>

// #include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include "sg_slam/config.h"

namespace sg_slam {

// --------------------------------------------------------------------------------------------------------------
TUM_dataset::TUM_dataset(const std::string &dataset_path) 
    : dataset_path_(dataset_path) {

}

// --------------------------------------------------------------------------------------------------------------
bool TUM_dataset::Init(double scale) {
    m_scale_ = scale;  // 目前没有使用

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

    Camera::Ptr new_camera( new Camera() );
    DistCoef.copyTo(new_camera->m_distcoef_);
    m_camera_ = new_camera;

    // timestameps
    std::cout << "vTimestamps_: " << vTimestamps_.size() << std::endl;
    std::cout << "vDepthTimestamps_: " << vDepthTimestamps_.size() << std::endl;
    std::cout << "vRGB_files_: " << vRGB_files_.size() << std::endl;
    std::cout << "vDepth_files_: " << vDepth_files_.size() << std::endl;

    std::ifstream fin(dataset_path_ + "/associate.txt");

    if(!fin) {
        std::cerr << "can't find " << dataset_path_ << "associate.txt" << std::endl;
        return false;
    }

    while ( !fin.eof() ) {
        std::string rgb_time, rgb_file, depth_time, depth_file;
        fin >> rgb_time >> rgb_file >> depth_time >> depth_file;

        vTimestamps_.push_back( std::atof( rgb_time.c_str() ) );
        vDepthTimestamps_.push_back( std::atof( depth_time.c_str() ) );
        vRGB_files_.push_back( dataset_path_ + "/" + rgb_file );
        vDepth_files_.push_back( dataset_path_ + "/" +  depth_file);

        // fin.good()是判断文件是否打开的，如果返回真的话就是打开了，否则没有打开
        if( fin.good() == false ) {
            break;
        }
    }
    fin.close();
    std::cout << "vTimestamps_: " << vTimestamps_.size() << std::endl;
    std::cout << "vDepthTimestamps_: " << vDepthTimestamps_.size() << std::endl;
    std::cout << "vRGB_files_: " << vRGB_files_.size() << std::endl;
    std::cout << "vDepth_files_: " << vDepth_files_.size() << std::endl;

    current_image_index_ = 0;
    return true;
}

// --------------------------------------------------------------------------------------------------------------
Frame::Ptr TUM_dataset::NextFrame() {
    cv::Mat image_color = cv::imread( vRGB_files_[current_image_index_], CV_LOAD_IMAGE_UNCHANGED );
    cv::Mat image_depth = cv::imread( vDepth_files_[current_image_index_], CV_LOAD_IMAGE_UNCHANGED );
    
    if( image_color.data==nullptr || image_depth.data==nullptr ) {
        std::cout <<  "can't find images at index " << current_image_index_ << std::endl;
        return nullptr;
    }

    if( image_color.channels() == 3 ) {
        cv::cvtColor(image_color, image_color,CV_RGB2GRAY);
    }

    auto new_frame = Frame::CreateFrame();
    new_frame->left_img_ = image_color;
    new_frame->depth_img_ = image_depth;

    if(m_camera_->m_image_cut_flag) {
        new_frame->is_use_cut_image = true;
        cv::Mat left_ROI = new_frame->left_img_(m_camera_->m_image_cut_select);
        cv::Mat depth_ROI = new_frame->depth_img_(m_camera_->m_image_cut_select);
        new_frame->left_cut_img_ = left_ROI;
        new_frame->depth_cut_img_ = depth_ROI;
        // new_frame->left_img_ = left_ROI;
        // new_frame->depth_img_ = depth_ROI;
    }

    new_frame->time_stamp_ = vTimestamps_[current_image_index_];
    new_frame->depth_scale_ = m_camera_->depth_scale_;

    current_image_index_++;

    return new_frame;
}

}  // namespace sg_slam