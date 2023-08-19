//
//  Created by Lucifer on 2022/10/27.
//

#include "sg_slam/EuRoC_dataset.h"

#include "sg_slam/config.h"

namespace sg_slam {

// --------------------------------------------------------------------------------------------------------------
EuRoC_dataset::EuRoC_dataset(const std::string &dataset_path) 
    : dataset_path_(dataset_path) {
    
}

// --------------------------------------------------------------------------------------------------------------
bool EuRoC_dataset::Init(double scale) {
    m_scale_ = scale;

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

    // std::cout << "DistCoef: " << DistCoef << std::endl;

    double fx = Config::Get<double>(static_cast<std::string>("camera.fx"));
    double fy = Config::Get<double>(static_cast<std::string>("camera.fy"));
    double cx = Config::Get<double>(static_cast<std::string>("camera.cx"));
    double cy = Config::Get<double>(static_cast<std::string>("camera.cy"));
    double fx_baseline = Config::Get<double>(static_cast<std::string>("camera.bf"));
    double baseline = fx_baseline / fx;
    
    Mat34 KT_l;
    Mat34 KT_r;
    KT_l <<  fx, 0.0,  cx, 0.0,
            0.0,  fy,  cy, 0.0,
            0.0, 0.0, 1.0, 0.0;
    KT_r <<  fx, 0.0,  cx, -fx_baseline,
            0.0,  fy,  cy, 0.0,
            0.0, 0.0, 1.0, 0.0;

    Vec3 t_l;
    Vec3 t_r;
    t_l << 0.0, 0.0, 0.0;
    t_r << -baseline, 0.0, 0.0;
    Camera::Ptr camera_l( new Camera(fx, fy, cx, cy, 0.0, 0.0, SE3(SO3(), t_l), KT_l) );
    DistCoef.copyTo(camera_l->m_distcoef_);
    cameras_.push_back(camera_l);

    Camera::Ptr camera_r( new Camera(fx, fy, cx, cy, baseline, fx_baseline, SE3(SO3(), t_r), KT_r) );
    DistCoef.copyTo(camera_r->m_distcoef_);
    cameras_.push_back(camera_r);

    /**
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "0: " << std::endl;
    std::cout << cameras_[0]->K() << std::endl;
    std::cout << std::endl;
    std::cout << cameras_[0]->pose().matrix() << std::endl;
    std::cout << std::endl;
    std::cout << (cameras_[0]->K() * cameras_[0]->pose().matrix3x4()) << std::endl;
    std::cout << std::endl;

    std::cout << "1: " << std::endl;
    std::cout << cameras_[1]->K() << std::endl;
    std::cout << std::endl;
    std::cout << cameras_[1]->pose().matrix() << std::endl;
    std::cout << std::endl;
    std::cout << (cameras_[1]->K() * cameras_[1]->pose().matrix3x4()) << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    **/

    cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
    Config::GetCVmat(static_cast<std::string>("LEFT.K")) >> K_l;
    Config::GetCVmat(static_cast<std::string>("RIGHT.K")) >> K_r;

    Config::GetCVmat(static_cast<std::string>("LEFT.P")) >> P_l;
    Config::GetCVmat(static_cast<std::string>("RIGHT.P")) >> P_r;

    Config::GetCVmat(static_cast<std::string>("LEFT.R")) >> R_l;
    Config::GetCVmat(static_cast<std::string>("RIGHT.R")) >> R_r;

    Config::GetCVmat(static_cast<std::string>("LEFT.D")) >> D_l;
    Config::GetCVmat(static_cast<std::string>("RIGHT.D")) >> D_r;

    int rows_l = Config::Get<int>(static_cast<std::string>("LEFT.height"));
    int cols_l = Config::Get<int>(static_cast<std::string>("LEFT.width"));
    int rows_r = Config::Get<int>(static_cast<std::string>("RIGHT.height"));
    int cols_r = Config::Get<int>(static_cast<std::string>("RIGHT.width"));

    if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
            rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0) {
        std::cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << std::endl;
        return false;
    }

    /**
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "K_l = " << std::endl;
    std::cout << K_l << std::endl;
    std::cout << "K_r = " << std::endl;
    std::cout << K_r << std::endl;
    std::cout << "P_l = " << std::endl;
    std::cout << P_l << std::endl;
    std::cout << "P_r = " << std::endl;
    std::cout << P_r << std::endl;
    std::cout << "R_l = " << std::endl;
    std::cout << R_l << std::endl;
    std::cout << "R_r = " << std::endl;
    std::cout << R_r << std::endl;
    std::cout << "D_l = " << std::endl;
    std::cout << D_l << std::endl;
    std::cout << "D_r = " << std::endl;
    std::cout << D_r << std::endl;
    std::cout << "rows_l = " << rows_l << std::endl;
    std::cout << "cols_l = " << cols_l << std::endl;
    std::cout << "rows_r = " << rows_r << std::endl;
    std::cout << "cols_r = " << cols_r << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    **/
    
    cv::initUndistortRectifyMap(K_l, D_l, R_l, P_l.rowRange(0, 3).colRange(0, 3), cv::Size(cols_l, rows_l), CV_32F, m_M1l_, m_M2l_);
    cv::initUndistortRectifyMap(K_r, D_r, R_r, P_r.rowRange(0, 3).colRange(0, 3), cv::Size(cols_r, rows_r), CV_32F, m_M1r_, m_M2r_);
    
    std::string timestampsFile = Config::Get<std::string>(static_cast<std::string>("EuRoc_timestampsfile"));
    std::ifstream fin(timestampsFile);

    if(!fin) {
        std::cerr << "can't find " << timestampsFile << " !" << std::endl;
        return false;
    }

    while (!fin.eof()) {
        std::string s;
        std::getline(fin, s);
        if(!s.empty()) {
            std::stringstream ss;
            ss << s;
            vTimestamps_str_.push_back(ss.str());
            double t;
            // long double t;
            ss >> t;
            vTimestamps_.push_back(t/1e9);
            
        }
    }
    fin.close();
    
    std::cout << "vTimestamps_: " << vTimestamps_.size() << std::endl;
    std::cout << "vTimestamps_str_: " << vTimestamps_str_.size() << std::endl;

    current_image_index_ = 0;

    return true;
}

// --------------------------------------------------------------------------------------------------------------
Frame::Ptr EuRoC_dataset::NextFrame() {
    if(current_image_index_ >= static_cast<int>(vTimestamps_.size())) {
        std::cout << "current_image_index_ = " << current_image_index_ << std::endl;
        std::cout << "Sequence finish! " << std::endl;
        return nullptr;
    }

    std::string strImageLeft = dataset_path_ + "/cam0/data/" + vTimestamps_str_[current_image_index_] + ".png";
    std::string strImageRight = dataset_path_ + "/cam1/data/" + vTimestamps_str_[current_image_index_] + ".png";
    cv::Mat image_left, image_right, image_left_rect, image_right_rect;

    image_left  = cv::imread(strImageLeft, CV_LOAD_IMAGE_UNCHANGED);
    image_right = cv::imread(strImageRight, CV_LOAD_IMAGE_UNCHANGED);

    if(image_left.data==nullptr || image_right.data==nullptr) {
        std::cout <<  "can't find images at index " << current_image_index_ << std::endl;
        return nullptr;
    }

    cv::remap(image_left, image_left_rect, m_M1l_, m_M2l_, cv::INTER_LINEAR);
    cv::remap(image_right, image_right_rect, m_M1r_, m_M2r_, cv::INTER_LINEAR);

    auto new_frame = Frame::CreateFrame();
    new_frame->left_img_ = image_left_rect;
    new_frame->right_img_ = image_right_rect;

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

    new_frame->time_stamp_ = vTimestamps_[current_image_index_];  // 量纲为 s

    current_image_index_++;

    return new_frame;
}

}  // namespace sg_slam
