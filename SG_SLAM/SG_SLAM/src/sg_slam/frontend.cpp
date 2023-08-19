//
//  Created by Lucifer on 2022/8/7.
//

#include "sg_slam/frontend.h"
#include "sg_slam/config.h"

#include <iostream>
#include <vector>

// #include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <torch/torch.h>


#include "sg_slam/sg_detectMatcher.h"

#include "sg_slam/algorithm.h"

#include "sg_slam/mappoint.h"
#include "sg_slam/keyframe.h"
#include "sg_slam/feature.h"

#include "sg_slam/g2o_types.h"

#include <g2o/core/block_solver.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_dogleg.h>  // 2022100401
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>

#include <g2o/types/sba/vertex_se3_expmap.h>
#include <g2o/types/slam3d/se3quat.h>
// #include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/types/sba/edge_project_stereo_xyz_onlypose.h>
#include <g2o/types/sba/edge_project_xyz_onlypose.h>


namespace sg_slam {

// --------------------------------------------------------------------------------------------------------------
Frontend::Frontend() {
    m_num_features_Threshold_init_ = Config::Get<int>(static_cast<std::string>("Frontend_num_features_Threshold_init"));

    m_max_keypoints_init_ = Config::Get<int>(static_cast<std::string>("Frontend_max_keypoints_init"));
    m_max_keypoints_tracking_ = Config::Get<int>(static_cast<std::string>("Frontend_max_keypoints_tracking"));
    m_max_keypoints_key_R_ = Config::Get<int>(static_cast<std::string>("Frontend_max_keypoints_key_R"));

    m_status_ = FrontendStatus::INITIALIZING;  // My add.

    // N.B.: My add.
    Vec6 se3_zero;
    se3_zero.setZero();
    m_relative_motion_ = SE3::exp(se3_zero);

    // N.B.: My add.
    m_sg_Track_match_threshold_ = sg_slam::Config::Get<float>(static_cast<std::string>("sg_Track_match_threshold"));
    m_sg_LR_match_threshold_ = sg_slam::Config::Get<float>(static_cast<std::string>("sg_LR_match_threshold"));

    m_mappoint_camera_depth_max_ = Config::Get<double>(static_cast<std::string>("Frontend_mappoint_camera_depth_max"));
    m_mappoint_camera_depth_min_ = Config::Get<double>(static_cast<std::string>("Frontend_mappoint_camera_depth_min"));

    m_mappoint_camera_x_max_ = Config::Get<double>(static_cast<std::string>("Frontend_mappoint_camera_x_max"));
    m_mappoint_camera_x_min_ = Config::Get<double>(static_cast<std::string>("Frontend_mappoint_camera_x_min"));

    m_mappoint_camera_y_max_ = Config::Get<double>(static_cast<std::string>("Frontend_mappoint_camera_y_max"));
    m_mappoint_camera_y_min_ = Config::Get<double>(static_cast<std::string>("Frontend_mappoint_camera_y_min"));

    m_ratio_features_th_tracking_ = Config::Get<double>(static_cast<std::string>("Frontend_ratio_features_th_tracking"));
    m_ratio_features_th_tracking_bad_ = Config::Get<double>(static_cast<std::string>("Frontend_ratio_features_th_tracking_bad"));
    m_num_fearures_tracking_min_ = Config::Get<int>(static_cast<std::string>("Frontend_num_fearures_tracking_min"));

    // 20221009 改
    // m_num_features_Threshold_tracking_ = Config::Get<int>(static_cast<std::string>("Frontend_num_features_Threshold_tracking"));
    // m_num_features_Threshold_tracking_bad_ = Config::Get<int>(static_cast<std::string>("Frontend_num_features_Threshold_tracking_bad"));
    m_num_features_Threshold_tracking_ = static_cast<int>(m_ratio_features_th_tracking_ * m_max_keypoints_tracking_);
    m_num_features_Threshold_tracking_bad_ = static_cast<int>(m_ratio_features_th_tracking_bad_ * m_max_keypoints_tracking_);

    // 20221009 加
    int camera_type = Config::Get<int>(static_cast<std::string>("camera.type"));
    // std::cout << "********1.5********" << std::endl;
    if( camera_type == 0 ) {
        m_camera_type = CameraType::MONO;
    } else if( camera_type == 1 ) {
        m_camera_type = CameraType::SETERO;
    } else if( camera_type == 2 ) {
        m_camera_type = CameraType::RGBD;
    }

    int constant_speed_flag = Config::Get<int>(static_cast<std::string>("Frontend_constant_speed_flag"));
    if(constant_speed_flag == 1) {
        m_frontend_constant_speed_flag = true;
    } else {
        m_frontend_constant_speed_flag = false;
    }

    // 20221010 加
    m_mappoint_camera_z_track_max_ = Config::Get<double>(static_cast<std::string>("Frontend_mappoint_camera_z_track_max"));
    m_mappoint_camera_z_track_min_ = Config::Get<double>(static_cast<std::string>("Frontend_mappoint_camera_z_track_min"));
    
    m_local_keyframes_deque.clear();
    // m_local_keyframes_deque_length = 2;

    std::cout << std::endl;
    std::cout << "(Frontend::Frontend()): m_num_features_Threshold_init_ = " << m_num_features_Threshold_init_ << std::endl;
    std::cout << "(Frontend::Frontend()): m_num_features_Threshold_tracking_ = " << m_num_features_Threshold_tracking_ << std::endl;
    std::cout << "(Frontend::Frontend()): m_num_features_Threshold_tracking_bad_ = " << m_num_features_Threshold_tracking_bad_ << std::endl;
    std::cout << "(Frontend::Frontend()): m_max_keypoints_init_ = " << m_max_keypoints_init_ << std::endl;
    std::cout << "(Frontend::Frontend()): m_max_keypoints_tracking_ = " << m_max_keypoints_tracking_ << std::endl;
    std::cout << "(Frontend::Frontend()): m_max_keypoints_key_R_ = " << m_max_keypoints_key_R_ << std::endl;
    std::cout << "(Frontend::Frontend()): m_sg_Track_match_threshold_ = " << m_sg_Track_match_threshold_ << std::endl;
    std::cout << "(Frontend::Frontend()): m_sg_LR_match_threshold_ = " << m_sg_LR_match_threshold_ << std::endl;
    std::cout << "(Frontend::Frontend()): m_mappoint_camera_depth_max_ = " << m_mappoint_camera_depth_max_ << std::endl;
    std::cout << "(Frontend::Frontend()): m_mappoint_camera_depth_min_ = " << m_mappoint_camera_depth_min_ << std::endl;
    std::cout << "(Frontend::Frontend()): m_mappoint_camera_x_max_ = " << m_mappoint_camera_x_max_ << std::endl;
    std::cout << "(Frontend::Frontend()): m_mappoint_camera_x_min_ = " << m_mappoint_camera_x_min_ << std::endl;
    std::cout << "(Frontend::Frontend()): m_mappoint_camera_y_max_ = " << m_mappoint_camera_y_max_ << std::endl;
    std::cout << "(Frontend::Frontend()): m_mappoint_camera_y_min_ = " << m_mappoint_camera_y_min_ << std::endl;
    std::cout << "(Frontend::Frontend()): m_ratio_features_th_tracking_ = " << m_ratio_features_th_tracking_ << std::endl;
    std::cout << "(Frontend::Frontend()): m_ratio_features_th_tracking_bad_ = " << m_ratio_features_th_tracking_bad_ << std::endl;
    std::cout << "(Frontend::Frontend()): m_num_fearures_tracking_min_ = " << m_num_fearures_tracking_min_ << std::endl;
    std::cout << "(Frontend::Frontend()): m_camera_type = " << static_cast<int>(m_camera_type) << std::endl;
    std::cout << "(Frontend::Frontend()): m_frontend_constant_speed_flag = " << static_cast<int>(m_frontend_constant_speed_flag) << std::endl;
    std::cout << "(Frontend::Frontend()): m_mappoint_camera_z_track_max_ = " << m_mappoint_camera_z_track_max_ << std::endl;
    std::cout << "(Frontend::Frontend()): m_mappoint_camera_z_track_min_ = " << m_mappoint_camera_z_track_min_ << std::endl;
    std::cout << "(Frontend::Frontend()): m_local_keyframes_deque.size = " << m_local_keyframes_deque.size() << std::endl;
    // std::cout << "(Frontend::Frontend()): local_keyframes_deque_length = " << m_local_keyframes_deque_length << std::endl;
    std::cout << std::endl;
}

// --------------------------------------------------------------------------------------------------------------
bool Frontend::AddFrame(Frame::Ptr frame) {
    m_current_frame_ = frame;
    
    // N.B.: My add
    // 设定激活与否在将关键帧插入地图时会设计
    FrameResult::Ptr current_result = FrameResult::Ptr(new FrameResult(m_current_frame_));
    m_map_->InsertFrameResult(current_result);

    switch( m_status_ ) {
        case FrontendStatus::INITIALIZING:
            // 20221009 改
            // StereoInit();  // 若单目或RGBD换成相应的初始化.
            Init();
            break;
        case FrontendStatus::TRACKING_GOOD:
        case FrontendStatus::TRACKING_BAD:
            // std::cout << "****************Frontend::AddFrame 0.1***************" << std::endl;
            Track();
            // std::cout << "****************Frontend::AddFrame 0.2***************" << std::endl;
            break;
        case FrontendStatus::LOST:
            // Reset();
            break;
    }

    current_result->is_keyframe_ = m_current_frame_->is_keyframe_;
    current_result->SetRelativePose(m_current_frame_->RelativePose());
    current_result->reference_keyframe_ptr_ = m_current_frame_->reference_keyframe_ptr_;

    // std::cout << "****************Frontend::AddFrame 1***************" << std::endl;

    // N.B.: 自己改了
    if(m_viewer_) {
        m_viewer_->AddCurrentFrame(m_current_frame_);
        m_viewer_->UpdateMap();
    }

    // 改 20220917
    if(m_status_ == FrontendStatus::LOST)
    { 
        // My add 20220918
        m_last_lost_number_ += 1;

        m_last_status_ = m_status_;
        m_status_ = FrontendStatus::TRACKING_GOOD;
    } else {
        // My add 20220918
        m_last_lost_number_ = 0;

        m_last_status_ = m_status_;
        m_last_frame_ = m_current_frame_;  // 保存当前帧为下一个时刻的上一帧
    }
    

    // N.B.: 自己加的，为了减轻内存消耗(若显示错误就删除，或段错误也删除).
    m_current_frame_->right_img_.release();
    m_current_frame_->right_cut_img_.release();
    // std::cout << "****************Frontend::AddFrame 2***************" << std::endl;
    // m_current_frame_->left_img_.release();  // N.B.: 加了这句会导致段错误无法正常显示图片!!!!!!

    return true;
}

// --------------------------------------------------------------------------------------------------------------
void Frontend::SetMap(Map::Ptr map) {
    m_map_ = map;
}

// --------------------------------------------------------------------------------------------------------------
void Frontend::SetBackend(Backend::Ptr backend) {
    m_backend_ = backend;
}

// --------------------------------------------------------------------------------------------------------------
void Frontend::SetLoopClosing(LoopClosing::Ptr loopclosing) {
    m_loopclosing_ = loopclosing;
}

// --------------------------------------------------------------------------------------------------------------
void Frontend::SetViewer(Viewer::Ptr viewer) {
    m_viewer_ = viewer;
}

// --------------------------------------------------------------------------------------------------------------
void Frontend::SetCameras(Camera::Ptr left, Camera::Ptr right) {
    m_cam_left_ = left;
    m_cam_right_ = right;
}

// --------------------------------------------------------------------------------------------------------------
void Frontend::SetStatus(FrontendStatus status) {
    m_status_ = status;
}

// --------------------------------------------------------------------------------------------------------------
FrontendStatus Frontend::GetStatus() const {
    return m_status_;
}


// private function
// --------------------------------------------------------------------------------------------------------------
bool Frontend::Track() {
    // 上一帧不为空，即当前帧不为第一帧(第0帧)或上一帧没有丢失(可以跟踪)
    // N.B.: 丢失情况还没写，一个思路是: 可直接设当前帧为关键帧，并和最近的关键帧(地图中ID最大的，或匹配点最多的)计算相对位姿重新定位.
    if(m_last_frame_) {
        if(m_frontend_constant_speed_flag) {
            // 设置当前帧的迭代初值，即估计初值(相对与上一帧的参考帧的)
            m_current_frame_->SetRelativePose(m_relative_motion_ * m_last_frame_->RelativePose());
        } else {
            Vec6 se3_zero;
            se3_zero.setZero();
            m_current_frame_->SetRelativePose(SE3::exp(se3_zero));
        }
        // N.B.: 自己加, 将当前帧的参考关键帧暂时设为上一帧的参考关键帧.
        m_current_frame_->reference_keyframe_ptr_ = m_last_frame_->reference_keyframe_ptr_.lock();
    }

    ExtractCurrent();

    int extract_current_keypoint_number = static_cast<int>(m_current_frame_->mvKeysUn.size());
    m_num_features_Threshold_tracking_ = static_cast<int>(m_ratio_features_th_tracking_ * extract_current_keypoint_number);
    m_num_features_Threshold_tracking_bad_ = static_cast<int>(m_ratio_features_th_tracking_bad_ * extract_current_keypoint_number);

    // N.B.: 在与上一帧联立进行当前帧相对位姿估计之前，先保存优化之前的当前帧相对位姿，
    // 为了作为与参考关键帧联立进行当前帧相对位姿估计的初值, EstimateCurrentRelativePoseForLast()会改变 m_current_frame_ 中的相对位姿.
    SE3 current_RelativePose_before = m_current_frame_->RelativePose();

    // 改 20220919
    // int num_track_reference = TrackReferenceKeyframe();
    int num_track_reference = TrackReferenceKeyframeNew();

    if(m_stereo_dataset_type == 1) {
        int num_track_local_keyframes = TrackLocalKeyframes();
        std::cout << "(Frontend::Track()): num_track_local_keyframes = " << num_track_local_keyframes << std::endl;
    }


    auto key_src_id = m_current_frame_->reference_keyframe_ptr_.lock()->src_frame_id_;

    // int tracking_ref_inliers = EstimateCurrentRelativePoseForReferenceStereo(current_RelativePose_before, 0);
    int tracking_ref_inliers = EstimateCurrentRelativePoseForReferenceNew(current_RelativePose_before);

    int tracking_last3d_inliers = 0;
    int num_track_last = 0;
    num_track_last = TrackLastFrame3D();
    if((m_current_frame_->id_ - m_current_frame_->reference_keyframe_ptr_.lock()->src_frame_id_) > 1) {
        tracking_last3d_inliers = EstimateCurrentRelativePoseForLast3D(current_RelativePose_before, tracking_ref_inliers);
    }
    std::cout << std::endl;
    std::cout << "(Frontend::Track()): " << num_track_last << " in the last image. " << std::endl;
    std::cout << "(Frontend::Track()): m_currentForLast_matches_with_3d_points_unorderedmap_: " << m_currentForLast_matches_with_3d_points_unorderedmap_.size()<< std::endl;
    std::cout << "(Frontend::Track()): tracking_last3d_inliers: " << tracking_last3d_inliers << std::endl;
    std::cout << std::endl;

    std::cout << std::endl;
    std::cout << "(Frontend::Track()): " << num_track_reference << " in the reference image. " << std::endl;
    std::cout << "(Frontend::Track()): tracking_ref_inliers = " << tracking_ref_inliers << std::endl;
    std::cout << "(Frontend::Track()): key_src_id = " << key_src_id << std::endl;
    std::cout << "(Frontend::Track()): m_last_frame_->id_ = " << m_last_frame_->id_ << std::endl;
    std::cout << std::endl;

    if(tracking_ref_inliers > static_cast<int>(m_ratio_features_th_tracking_ * extract_current_keypoint_number)) {
        // tracking good
        m_status_ = FrontendStatus::TRACKING_GOOD;
    } else if(tracking_ref_inliers > static_cast<int>(m_ratio_features_th_tracking_bad_ * extract_current_keypoint_number)) {
        // tracking bad
        m_status_ = FrontendStatus::TRACKING_BAD;
    } else {
        // My add 20220916
        // lost
        m_status_ = FrontendStatus::LOST;
        // m_status_ = FrontendStatus::TRACKING_BAD;
        
    }

    /**
    // 改 20220919
    int num_track_reference = TrackReferenceKeyframe();
    **/

    // 改 20220919
    // LOST 后的处理
    if(m_status_ == FrontendStatus::LOST) {
        // 20221018 改
        if(m_last_lost_number_ > 10) {
            std::cout << std::endl;
            std::cout << "(Frontend::Track()): m_last_lost_number_ = " << m_last_lost_number_ << std::endl;
            std::cout << std::endl;
            m_status_ = FrontendStatus::TRACKING_BAD;
            /**
            m_status_ = FrontendStatus::TRACKING_GOOD;
            // TrackAdditionalLandmarksFromReference();
            // 第二步： 生成新的地图点.
            // StereoCreateNewPoints();
            TrackAdditionalLandmarksFromLast3D();
            if(m_camera_type == CameraType::SETERO) {
                StereoCreateNewPoints();
            } else if (m_camera_type == CameraType::RGBD) {
                RGBDCreateNewPoints();
            }
            return true;
            **/
        }
    } else {
        /**
        auto key_src_id = m_current_frame_->reference_keyframe_ptr_.lock()->src_frame_id_;
        if(key_src_id != m_last_frame_->id_) {
            int tracking_ref_inliers = EstimateCurrentRelativePoseForReferenceStereo(current_RelativePose_before, tracking_last_inliers);
            std::cout << std::endl;
            std::cout << "(Frontend::Track()): " << num_track_reference << " in the reference image. " << std::endl;
            std::cout << "(Frontend::Track()): tracking_ref_inliers = " << tracking_ref_inliers << std::endl;
            std::cout << std::endl;
       }
       std::cout << std::endl;
       std::cout << "(Frontend::Track()): key_src_id = " << key_src_id << std::endl;
       std::cout << "(Frontend::Track()): m_last_frame_->id_ = " << m_last_frame_->id_ << std::endl;
       std::cout << std::endl;
       **/
    }

    // LOST 处理 后还是 LOST
    if(m_status_ == FrontendStatus::LOST) {

        std::cout << std::endl;
        std::cout << "(Frontend::Track()): LOST m_current_frame_->id_: " << m_current_frame_->id_ << ", " << "m_last_frame_->id_: " << m_last_frame_->id_ << std::endl;
        std::cout << "(Frontend::Track()): LOST m_status_: " << static_cast<int>(m_status_)<< ", " << "m_last_status_: " << static_cast<int>(m_last_status_) << std::endl;
        std::cout << "(Frontend::Track()): LOST m_last_lost_number_: " << m_last_lost_number_ << std::endl;
        std::cout << "(Frontend::Track()): LOST " << num_track_reference << " in the reference image. " << std::endl;
        std::cout << "(Frontend::Track()): LOST tracking_ref_inliers = " << tracking_ref_inliers << std::endl;
        std::cout << std::endl;

        Vec6 se3_zero;
        se3_zero.setZero();
        m_relative_motion_ = SE3::exp(se3_zero);
        m_current_frame_->SetRelativePose(m_last_frame_->RelativePose());
        return true;
    }

    
    // 20221030 add
    auto id_key_interval = m_current_frame_->id_ - m_current_frame_->reference_keyframe_ptr_.lock()->src_frame_id_;
    if(id_key_interval > m_key_max_id_interval) {
        std::cout << std::endl;
        std::cout << "(Frontend::Track()): id_key_interval = " << id_key_interval << std::endl;
        std::cout << std::endl;

        m_status_ = FrontendStatus::TRACKING_BAD;
    }

    // N.B.: 更新与上一帧的相对运动，必须放在生成新关键帧之前，
    // 因为生成新关键后当前帧的参考关键帧与上一帧的参考关键帧不同，
    // 无法直接使用相对与同一个参考关键帧的相对位姿计算相对运动. !!!!!
    auto Tcl = m_current_frame_->RelativePose() * m_last_frame_->RelativePose().inverse();
    auto id_interval = m_current_frame_->id_ - m_last_frame_->id_;
    auto id_last = m_last_frame_->id_;
    if(id_interval > 1) {
        double add_factor = 1.0 / static_cast<double>(id_interval);
        auto Tcl_vector = Tcl.log();
        for(unsigned long i = 1; i < id_interval; i++) {
            auto frame_result = m_map_->GetAllFrameResults()[i+id_last];
            // auto frame_result_Tcl = SE3(Tcl_so3, (i*add_factor)*Tcl_t);
            auto frame_result_Tcl = SE3::exp((i*add_factor)*Tcl_vector);
            frame_result->SetRelativePose(frame_result_Tcl*m_last_frame_->RelativePose());
        }
        m_relative_motion_ = SE3::exp((add_factor)*Tcl_vector);
    } else {
        m_relative_motion_ = Tcl;
    }

    // N.B.: 这里采用和清华工程相同的逻辑
    if(m_status_ == FrontendStatus::TRACKING_BAD) {
        // 第一步: 先从与上一帧匹配中补充对应的地图点(只填充没有对应点的位置)
        // TrackAdditionalLandmarksFromReference();
        // 20221105 改
        TrackAdditionalLandmarksFromLast3D();

        // 第二步： 生成新的地图点.
        // 20221009 改
        if(m_camera_type == CameraType::SETERO) {
            /**
            if(m_stereo_dataset_type == 0) {
                StereoCreateNewPoints();
            } else if(m_stereo_dataset_type == 1) {
                Stereo_LR_CreateNewPoints();
            }
            **/
            StereoCreateNewPoints();
        } else if(m_camera_type == CameraType::RGBD) {
            RGBDCreateNewPoints();
        }
        // 第步：插入新关键帧
        InsertKeyFrame();
    }

    // N.B.: 与可视化线程的交互，统一放在 Frontend::AddFrame, 这里不用放.

    return true;
}

// --------------------------------------------------------------------------------------------------------------
// Stereo / RGBD
void Frontend::ExtractCurrent() {
    // auto sg_Track_match_threshold = sg_slam::Config::Get<float>(static_cast<std::string>("sg_Track_match_threshold")); 
    std::vector< cv::KeyPoint > left_keypoints_v;
    std::vector< cv::KeyPoint > left_keypoints_v_un;
    torch::Tensor descL;  // desc0;
    // auto detect_out_tuple = sg_DetectMatcher::detect(m_current_frame_->left_img_, m_max_keypoints_tracking_);
    // 20221011 改
    std::tuple<std::vector< cv::KeyPoint >, torch::Tensor>  detect_out_tuple;
    if(m_current_frame_->is_use_cut_image) {
        detect_out_tuple = sg_DetectMatcher::detect(m_current_frame_->left_cut_img_, m_max_keypoints_tracking_);
    } else {
        detect_out_tuple = sg_DetectMatcher::detect(m_current_frame_->left_img_, m_max_keypoints_tracking_);
    }

    left_keypoints_v = std::get<0>(detect_out_tuple);
    descL = std::get<1>(detect_out_tuple).to(torch::kCPU);
    // 左图去畸变
    if(m_cam_left_->m_distcoef_.at<float>(0) == 0.0) {
        left_keypoints_v_un = left_keypoints_v;
    } else {
        int N_l = left_keypoints_v.size();
        cv::Mat mat(N_l, 2, CV_32F);
        for(int j=0; j<N_l; j++) {
            mat.at<float>(j, 0) = left_keypoints_v[j].pt.x;
            mat.at<float>(j, 1) = left_keypoints_v[j].pt.y;
        }

        cv::Mat mK;
        cv::eigen2cv(m_cam_left_->K(), mK);

        // Undistort points
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, m_cam_left_->m_distcoef_, cv::Mat(), mK);
        mat = mat.reshape(1);

        // Fill undistorted keypoint vector
        left_keypoints_v_un.resize(N_l);
        for(int j=0; j<N_l; j++) {
            cv::KeyPoint kp = left_keypoints_v[j];
            kp.pt.x = mat.at<float>(j, 0);
            kp.pt.y = mat.at<float>(j, 1);
            left_keypoints_v_un[j] = kp;
        }
    }

    // 20221009 改
    if(m_camera_type == CameraType::SETERO) {
        // My add 20220914
        // 在右图中提取关键点
        std::vector< cv::KeyPoint > right_keypoints_v;
        std::vector< cv::KeyPoint > right_keypoints_v_un;
        torch::Tensor descR; // desc1
        // detect_out_tuple = sg_DetectMatcher::detect(m_current_frame_->right_img_, m_max_keypoints_tracking_);

        // 20221011 改
        if(m_current_frame_->is_use_cut_image) {
            detect_out_tuple = sg_DetectMatcher::detect(m_current_frame_->right_cut_img_, m_max_keypoints_tracking_);
        } else {
            detect_out_tuple = sg_DetectMatcher::detect(m_current_frame_->right_img_, m_max_keypoints_tracking_);
        }

        right_keypoints_v = std::get<0>(detect_out_tuple);
        descR = std::get<1>(detect_out_tuple).to(torch::kCPU);
        // 右图去畸变
        if(m_cam_right_->m_distcoef_.at<float>(0) == 0.0) {
            right_keypoints_v_un = right_keypoints_v;
        } else {
            int N_r = right_keypoints_v.size();
            cv::Mat mat(N_r, 2, CV_32F);
            for(int j=0; j<N_r; j++) {
                mat.at<float>(j, 0) = right_keypoints_v[j].pt.x;
                mat.at<float>(j, 1) = right_keypoints_v[j].pt.y;
            }

            cv::Mat mK;
            cv::eigen2cv(m_cam_right_->K(), mK);

            // Undistort points
            mat = mat.reshape(2);
            cv::undistortPoints(mat, mat, mK, m_cam_left_->m_distcoef_, cv::Mat(), mK);
            mat = mat.reshape(1);

            // Fill undistorted keypoint vector
            right_keypoints_v_un.resize(N_r);
            for(int j=0; j<N_r; j++) {
                cv::KeyPoint kp = right_keypoints_v[j];
                kp.pt.x = mat.at<float>(j, 0);
                kp.pt.y = mat.at<float>(j, 1);
                right_keypoints_v_un[j] = kp;
            }
        }

        // My add 20220914
        auto matcher_out_dict = sg_DetectMatcher::descMatch(descL, descR, m_sg_LR_match_threshold_);
        std::vector<cv::DMatch> matchers_v;
        auto matcher0to1_tensor = matcher_out_dict["matches0"][0].to(torch::kCPU);
        for(int64_t i=0; i < matcher0to1_tensor.size(0); i++) {
            int index0to1 = matcher0to1_tensor[i].item<int>();
            if(index0to1 > -1) {
                cv::DMatch match1to2_one;
                match1to2_one.queryIdx = i;
                match1to2_one.trainIdx = index0to1;
                matchers_v.push_back(match1to2_one);
            }
        }
        // N.B.: 关键点集合没有缩小
        auto RR_matchers_v = sg_slam::sg_RANSAC(left_keypoints_v, right_keypoints_v, matchers_v, m_num_features_Threshold_tracking_bad_);

        m_current_frame_->mvKeys = left_keypoints_v;
        m_current_frame_->mvKeysUn = left_keypoints_v_un;
        descL = descL.to(torch::kCPU);
        m_current_frame_->mDescriptors_data = convertDesTensor2Vector(descL);
        m_current_frame_->Descripters_B = descL.size(0);
        m_current_frame_->Descripters_D = descL.size(1);
        m_current_frame_->Descripters_N = descL.size(2);

        // My add 20220914 
        m_current_frame_->mvKeysR_u.resize(m_current_frame_->mvKeysUn.size());
        std::fill(m_current_frame_->mvKeysR_u.begin(), m_current_frame_->mvKeysR_u.end(), -1.0);
        m_current_frame_->mvKeysR_v.resize(m_current_frame_->mvKeysUn.size());
        std::fill(m_current_frame_->mvKeysR_v.begin(), m_current_frame_->mvKeysR_v.end(), -1.0);
        for(size_t i=0; i<RR_matchers_v.size(); i++) {
            int leftIndex = RR_matchers_v[i].queryIdx;
            int rightIndex = RR_matchers_v[i].trainIdx;

            m_current_frame_->mvKeysR_u[leftIndex] = right_keypoints_v_un[rightIndex].pt.x;
            m_current_frame_->mvKeysR_v[leftIndex] = right_keypoints_v_un[rightIndex].pt.y;
        }
    } else if(m_camera_type == CameraType::RGBD) {
        m_current_frame_->mvKeys = left_keypoints_v;
        m_current_frame_->mvKeysUn = left_keypoints_v_un;
        descL = descL.to(torch::kCPU);
        m_current_frame_->mDescriptors_data = convertDesTensor2Vector(descL);
        m_current_frame_->Descripters_B = descL.size(0);
        m_current_frame_->Descripters_D = descL.size(1);
        m_current_frame_->Descripters_N = descL.size(2);

        // My add 20221013 
        double bf = m_cam_left_->fx_baseline_;
        m_current_frame_->mvKeysR_u.resize(m_current_frame_->mvKeysUn.size());
        std::fill(m_current_frame_->mvKeysR_u.begin(), m_current_frame_->mvKeysR_u.end(), -1.0);
        m_current_frame_->mvKeysR_v.resize(m_current_frame_->mvKeysUn.size());
        std::fill(m_current_frame_->mvKeysR_v.begin(), m_current_frame_->mvKeysR_v.end(), -1.0);
        m_current_frame_->mvKeysDepth.resize(m_current_frame_->mvKeysUn.size());
        std::fill(m_current_frame_->mvKeysDepth.begin(), m_current_frame_->mvKeysDepth.end(), -1.0);
        for(size_t i=0; i<m_current_frame_->mvKeysUn.size(); i++) {
            float d = m_current_frame_->findDepth(m_current_frame_->mvKeys[i]);
            if(d > 0.0) {
                auto kpU = m_current_frame_->mvKeysUn[i];
                m_current_frame_->mvKeysDepth[i] = d;
                m_current_frame_->mvKeysR_u[i] = kpU.pt.x - static_cast<float>(bf)/d;
            } else {
                m_current_frame_->mvKeysDepth[i] = -1.0;
                m_current_frame_->mvKeysR_u[i] = -1.0;
            }

            std::cout << "d = " << m_current_frame_->mvKeysDepth[i] << std::endl;
            std::cout << "R_u = " << m_current_frame_->mvKeysR_u[i] << std::endl;
        }
    }

    // 20221014 test 加
    m_current_frame_->mvpMapPoints.resize(m_current_frame_->mvKeysUn.size());
}

// --------------------------------------------------------------------------------------------------------------
int Frontend::TrackLastFrame() {
    // auto sg_Track_match_threshold = sg_slam::Config::Get<float>(static_cast<std::string>("sg_Track_match_threshold"));
    auto desc_current = convertDesVector2Tensor(m_current_frame_->mDescriptors_data, m_current_frame_->Descripters_B, m_current_frame_->Descripters_D, m_current_frame_->Descripters_N);
    auto desc_last = convertDesVector2Tensor(m_last_frame_->mDescriptors_data, m_last_frame_->Descripters_B, m_last_frame_->Descripters_D, m_last_frame_->Descripters_N);

    auto matcher_out_dict_track = sg_DetectMatcher::descMatch(desc_current, desc_last, m_sg_Track_match_threshold_);
    auto matcher0to1_tensor_track = matcher_out_dict_track["matches0"][0].to(torch::kCPU);
    std::vector<cv::DMatch> matchers_v;
    for(int64_t i=0; i<matcher0to1_tensor_track.size(0); i++) {
        int index0to1 = matcher0to1_tensor_track[i].item<int>();

        if(index0to1 > -1) {
            cv::DMatch match1to2_one;
            match1to2_one.queryIdx = i;
            match1to2_one.trainIdx = index0to1;
            matchers_v.push_back(match1to2_one);
        }
    }

    // N.B.: 关键点集合没有缩小
    auto RR_matchers_v = sg_slam::sg_RANSAC(m_current_frame_->mvKeys, m_last_frame_->mvKeys, matchers_v, m_num_features_Threshold_tracking_);
    
    // N.B.: 根据匹配将有对应地图点的上一帧的关键点的对应地图点填充到当前帧的相应位置
    m_current_frame_->mvpMapPoints.resize(m_current_frame_->mvKeysUn.size());
    int cnt_track_last_landmarks = 0;
    auto Tlast_w = m_last_frame_->RelativePose() * m_last_frame_->reference_keyframe_ptr_.lock()->Pose();

    for(size_t i=0; i<RR_matchers_v.size(); i++) {
        auto mp = m_last_frame_->mvpMapPoints[RR_matchers_v[i].trainIdx].lock();
        if(mp) {
            Vec3 posCamera = Tlast_w * mp->GetPos();
            if(posCamera[2] > m_mappoint_camera_z_track_max_) {
                continue;
            }
            if(posCamera[2] < m_mappoint_camera_z_track_min_) {
                continue;
            }

            m_current_frame_->mvpMapPoints[RR_matchers_v[i].queryIdx] = mp;
            cnt_track_last_landmarks++;
        }
    }

    m_current_frame_->keypoints_number = cnt_track_last_landmarks;

    return cnt_track_last_landmarks;
}

// --------------------------------------------------------------------------------------------------------------
int Frontend::TrackLastFrame3D() {
    if(m_last_frame_==nullptr) {
        std::cout << std::endl;
        std::cout << "Frontend::TrackLastFrame3D(): not last frame! " << std::endl;
        std::cout << std::endl;
        return 0;
    }

    auto desc_current = convertDesVector2Tensor(m_current_frame_->mDescriptors_data, m_current_frame_->Descripters_B, m_current_frame_->Descripters_D, m_current_frame_->Descripters_N);
    auto desc_last = convertDesVector2Tensor(m_last_frame_->mDescriptors_data, m_last_frame_->Descripters_B, m_last_frame_->Descripters_D, m_last_frame_->Descripters_N);

    auto matcher_out_dict_track = sg_DetectMatcher::descMatch(desc_current, desc_last, m_sg_Track_match_threshold_);
    auto matcher0to1_tensor_track = matcher_out_dict_track["matches0"][0].to(torch::kCPU);
    std::vector<cv::DMatch> matchers_v;
    for(int64_t i=0; i<matcher0to1_tensor_track.size(0); i++) {
        int index0to1 = matcher0to1_tensor_track[i].item<int>();

        if(index0to1 > -1) {
            cv::DMatch match1to2_one;
            match1to2_one.queryIdx = i;
            match1to2_one.trainIdx = index0to1;
            matchers_v.push_back(match1to2_one);
        }
    }

    // N.B.: 关键点集合没有缩小
    auto RR_matchers_v = sg_slam::sg_RANSAC(m_current_frame_->mvKeys, m_last_frame_->mvKeys, matchers_v, m_num_features_Threshold_tracking_);
    m_currentForLast_matches_with_3d_points_unorderedmap_.clear();
    int cnt_track_last_3dpoints = 0;
    auto last_reference_KF = m_last_frame_->reference_keyframe_ptr_.lock();
    SE3 last_pose_Twc = (m_last_frame_->RelativePose() * last_reference_KF->Pose()).inverse();
    std::vector<SE3> poses{ m_cam_left_->pose(), m_cam_right_->pose() };
    for(size_t i=0; i<RR_matchers_v.size(); i++) {
        int currentKeypointIndex = RR_matchers_v[i].queryIdx;
        int lastKeypointIndex = RR_matchers_v[i].trainIdx;
        if(m_camera_type == CameraType::SETERO) {
            if(m_last_frame_->mvKeysR_u[lastKeypointIndex] <= 0.0) {
                continue;
            }
            if(m_current_frame_->mvKeysR_u[currentKeypointIndex] == -1) {
                continue;
            }

            std::vector<Vec3> points{ 
            m_cam_left_->pixel2camera( 
                Vec2(m_last_frame_->mvKeysUn[lastKeypointIndex].pt.x,
                     m_last_frame_->mvKeysUn[lastKeypointIndex].pt.y)),
            m_cam_right_->pixel2camera( 
                Vec2(m_last_frame_->mvKeysR_u[lastKeypointIndex],
                     m_last_frame_->mvKeysR_v[lastKeypointIndex])) };

            Vec3 pworld = Vec3::Zero();  // N.B.: 这里其实是, p_camera, 即在左相机坐标系下的3D点坐标.
    
            if(triangulation(poses, points, pworld) && pworld[2] > 0.0) {
                // 20221018
                if(pworld.norm() > m_mappoint_camera_depth_max_) {
                    continue;
                }
                if(pworld.norm() < m_mappoint_camera_depth_min_) {
                    continue;
                }
                pworld = last_pose_Twc * pworld;  // N.B: 这里才是真正的世界坐标系下的3D点坐标.
                // 属于同一关键点的匹配不应两次插入到有效匹配中.
                if(m_currentForLast_matches_with_3d_points_unorderedmap_.find(currentKeypointIndex) != m_currentForLast_matches_with_3d_points_unorderedmap_.end()) {
                    continue;
                }
                m_currentForLast_matches_with_3d_points_unorderedmap_.insert( std::make_pair(currentKeypointIndex, pworld) );
                cnt_track_last_3dpoints++;
            }

        } else if(m_camera_type == CameraType::RGBD) {
            // 待加 ......
        }
    }

    return cnt_track_last_3dpoints;
}

int Frontend::EstimateCurrentRelativePoseForLast3D(const SE3 &intial_estimate, int ref_inliers) {
    // 构建图优化，设定g2o
    // setup g2o
    // solver for PnP/3D VO
    // pose dimension is 6 (se3), landmark dimension is 3.
    using BlockSolverType = g2o::BlockSolver_6_3;
    // 线性求解器类型(稠密)
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;

     // use LM (N.B.: 可以换优化搜索算法)
    auto solver = new g2o::OptimizationAlgorithmLevenberg( 
        g2o::make_unique<BlockSolverType>( g2o::make_unique<LinearSolverType>() ));
    
    g2o::SparseOptimizer optimizer;  // N.B.: 图模型优化器，无论稀疏还是稠密都是用这个求解器类型
    optimizer.setAlgorithm( solver );  // 设置求解器
    // optimizer.setVerbose( true );  // 打开调试输出

    // Vertex
    g2o::VertexSE3Expmap *vertex_pose = new g2o::VertexSE3Expmap();  // camera vertex_pose
    auto frame_relative_pose_SE3Quat = g2o::SE3Quat(intial_estimate.rotationMatrix(), intial_estimate.translation());
    vertex_pose->setId(0);  // 前端每次只估计一个 relative pose, 顶点的编号从0开始.
    vertex_pose->setEstimate(frame_relative_pose_SE3Quat);  // 设置相对位姿(相对于参考关键帧)的位姿初值.
    optimizer.addVertex(vertex_pose);

    // K((左)内参) 和 左右外参
    auto left_fx = m_cam_left_->fx_;
    auto left_fy = m_cam_left_->fy_;
    auto left_cx = m_cam_left_->cx_;
    auto left_cy = m_cam_left_->cy_;
    auto rl_bf = m_cam_right_->fx_baseline_;

    std::cout << std::endl;
    std::cout << "Frontend::EstimateCurrentRelativePoseForLast3D(...): rl_bf = " << rl_bf << std::endl;
    std::cout << std::endl;

    // 当前帧的参考关键帧
    auto current_reference_key = m_current_frame_->reference_keyframe_ptr_.lock();
    // N.B.: 当前参考关键帧相对于世界坐标系的位姿.
    SE3 reference_key_pose = current_reference_key->Pose();

    // edges
    int index = 1;  // 边的编号从1开始
    std::vector< g2o::EdgeStereoSE3ProjectXYZOnlyPose* > edges;
    std::vector< bool > vEdgeIsOutlier;
    std::vector< g2o::EdgeSE3ProjectXYZOnlyPose* > edges2d;
    std::vector< bool > vEdgeIsOutlier2d;

    int cnt_inlier = 0;

    const double thHuberStereo = sqrt(7.815); // robust kernel 阈值 (可调)
    const double thHuber2d = sqrt(5.991);
    for(auto iter = m_currentForLast_matches_with_3d_points_unorderedmap_.begin(); iter != m_currentForLast_matches_with_3d_points_unorderedmap_.end(); iter++) {
        int currentKeypointIndex = (*iter).first;
        Vec3 pointFromLast = (*iter).second;

        auto point_kR_u = m_current_frame_->mvKeysR_u[currentKeypointIndex];

        if(point_kR_u > -1.0) {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose *edge = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
            edge->setId(index);
            edge->setVertex(0, vertex_pose);
            Vec3 StereoMeasure;
            auto point2d = m_current_frame_->mvKeysUn[currentKeypointIndex].pt;
            StereoMeasure = toVec3(point2d, point_kR_u);
            edge->setMeasurement( StereoMeasure );
            edge->setInformation( Mat33::Identity() );  // 信息矩阵默认设为单位矩阵, 故测量误差最好白化，或者根据方差设置信息矩阵.
            auto rk = new g2o::RobustKernelHuber();
            rk->setDelta(thHuberStereo);
            edge->setRobustKernel(rk);
            edge->fx = left_fx;
            edge->fy = left_fy;
            edge->cx = left_cx;
            edge->cy = left_cy;
            edge->bf = rl_bf;

            // N.B.: 不同点
            Vec3 posCamera = reference_key_pose * pointFromLast;

            // edge->Xw = posCamera;
            edge->Xw[0] = posCamera[0];
            edge->Xw[1] = posCamera[1];
            edge->Xw[2] = posCamera[2];

            edges.push_back(edge);
            vEdgeIsOutlier.push_back(false);
            optimizer.addEdge(edge);

            cnt_inlier++;
            index++;
        } else {
            g2o::EdgeSE3ProjectXYZOnlyPose *edge = new g2o::EdgeSE3ProjectXYZOnlyPose();
            edge->setId(index);
            edge->setVertex(0, vertex_pose);
            auto point2d = m_current_frame_->mvKeysUn[currentKeypointIndex].pt;
            Vec2 measure2d;
            measure2d = toVec2(point2d);
            edge->setMeasurement(measure2d);
            edge->setInformation( Mat22::Identity() );
            auto rk = new g2o::RobustKernelHuber(); 
            rk->setDelta(thHuber2d);
            edge->setRobustKernel(rk);
            edge->fx = left_fx;
            edge->fy = left_fy;
            edge->cx = left_cx;
            edge->cy = left_cy;

            // N.B.: 不同点
            Vec3 posCamera = reference_key_pose * pointFromLast;

            // edge->Xw = posCamera;
            edge->Xw[0] = posCamera[0];
            edge->Xw[1] = posCamera[1];
            edge->Xw[2] = posCamera[2];

            edges2d.push_back(edge);
            vEdgeIsOutlier2d.push_back(false);
            optimizer.addEdge(edge);

            cnt_inlier++;
            index++;
        }
    }

    if(cnt_inlier < 10) {
        std::cout << std::endl;
        std::cout << "(Frontend::EstimateCurrentRelativePoseForLast3D(...)): Not enough points not to perform optimization " << std::endl;
        std::cout << "(Frontend::EstimateCurrentRelativePoseForLast3D(...)): cnt_inlier = " << cnt_inlier << std::endl;
        std::cout << std::endl;
        return 0;
    }

    const double chi2_th = 7.815;
    const double chi2_2d_th = 5.991;
    int cnt_outlier = 0;
    cnt_inlier = 0;
    // N.B.: 迭代次数(可调), 可转换为外化参数
    for(int iteration = 0; iteration < 8; iteration++) {
        frame_relative_pose_SE3Quat = g2o::SE3Quat(intial_estimate.rotationMatrix(), intial_estimate.translation());
        vertex_pose->setEstimate(frame_relative_pose_SE3Quat);
        optimizer.initializeOptimization();
        optimizer.optimize(10);  // N.B.: 每次优化时，优化器的迭代次数, 可转换为外化参数.
        cnt_outlier = 0;
        cnt_inlier = 0;

        // count the outliers
        for(size_t i=0; i < edges.size(); i++) {
            auto e = edges[i];
            if(vEdgeIsOutlier[i]) {
                // 上一次是外点的，因为这一次迭代没有优化，需要在这一次迭代的基础上重新计算误差，在判断是否还是外点.
                e->computeError();
            }

            // N.B.: 落在拒绝域
            if(e->chi2() > chi2_th) {
                vEdgeIsOutlier[i] = true;
                e->setLevel(1);  // 不优化
                cnt_outlier++;
            } else {
                vEdgeIsOutlier[i] = false;
                e->setLevel(0);  // 优化
                cnt_inlier++;
            }

            // 第三次已经执行完了，外点剔除的差不多了，第四次即最后一次迭代，不使用鲁棒核.
            if(iteration == 6) {
                // 因为剔除了错误的边，所以第三次之后优化不再使用核函数
                e->setRobustKernel(nullptr);
            }
        }

        // count the outliers
        for(size_t i=0; i < edges2d.size(); i++) {
            auto e = edges2d[i];
            if(vEdgeIsOutlier2d[i]) {
                // 上一次是外点的，因为这一次迭代没有优化，需要在这一次迭代的基础上重新计算误差，在判断是否还是外点.
                e->computeError();
            }

            // N.B.: 落在拒绝域
            if(e->chi2() > chi2_2d_th) {
                vEdgeIsOutlier2d[i] = true;
                e->setLevel(1);  // 不优化
                cnt_outlier++;
            } else {
                vEdgeIsOutlier2d[i] = false;
                e->setLevel(0);  // 优化
                cnt_inlier++;
            }

            // 第三次已经执行完了，外点剔除的差不多了，第四次即最后一次迭代，不使用鲁棒核.
            if(iteration == 6) {
                // 因为剔除了错误的边，所以第三次之后优化不再使用核函数
                e->setRobustKernel(nullptr);
            }
        }

        if(cnt_inlier < 10) {
            // N.B.: 下一次没有足够的点优化直接退出，不然系统会崩溃
            std::cout << std::endl;
            std::cout << "(Frontend::EstimateCurrentRelativePoseForLast3D(...)): cnt_inlier < 10 : " << cnt_inlier << std::endl;
            std::cout << std::endl;
            break;
        }
    }

    std::cout << std::endl;
    std::cout << "(Frontend::EstimateCurrentRelativePoseForLast3D(...)): Outlier/Inlier in pose estimating: " 
              << cnt_outlier<< "/" << cnt_inlier<< std::endl;
    std::cout << std::endl;

    if(cnt_inlier < 10) {
        std::cout << std::endl;
        std::cout << "(Frontend::EstimateCurrentRelativePoseForLast3D(...)): Not enough points" << std::endl;
        std::cout << std::endl;
        return 0;
    }

    // // N.B.: 不同点
    Vec6 current_relativePose_before_se3 = m_current_frame_->RelativePose().log();  // 融合前相对位姿李代数
    auto f_est_pose_SE3Quat = vertex_pose->estimate();
    auto f_est_q = f_est_pose_SE3Quat.rotation();
    auto f_est_t = f_est_pose_SE3Quat.translation();
    auto f_est_pose = SE3(f_est_q, f_est_t);
    Vec6 current_last_se3 = f_est_pose.log();

    double fuse_alpha = static_cast<double>(ref_inliers) / static_cast<double>(ref_inliers + cnt_inlier);

    // fuse_alpha = 2.0 * fuse_alpha;  // N.B.: 进行简单的线性缩放(使范围大概在0-1之间)
    // fuse_alpha = 1.6 * fuse_alpha;  // N.B.: 进行简单的线性缩放(使范围大概在0-1之间)
    fuse_alpha = 2.0 * fuse_alpha;  // N.B.: 进行简单的线性缩放(使范围大概在0-1之间)
   
    // 20221105 改
    if(m_stereo_dataset_type == 1) {
        if(fuse_alpha < 0.85) {
            fuse_alpha = 0.85;
        }
    }

    if(fuse_alpha > 1.0) {
        fuse_alpha = 1.0;
    }

    Vec6 relativePose_fuse_se3 = fuse_alpha*current_relativePose_before_se3 + (1.0 - fuse_alpha)*current_last_se3;
    SE3 relativePose_fuse = SE3::exp(relativePose_fuse_se3);

    m_current_frame_->SetRelativePose(relativePose_fuse);

    std::cout << std::endl;
    std::cout << "(Frontend::EstimateCurrentRelativePoseForLast3D(...)): fuse_alpha = " << fuse_alpha << std::endl;
    std::cout << std::endl;

    return cnt_inlier;
}

// --------------------------------------------------------------------------------------------------------------
// 参见 sg_slam/loopclosing.cpp int LoopClosing::OptimizeCurrentPose()
int Frontend::EstimateCurrentRelativePoseForLast() {
    // 构建图优化，设定g2o
    // setup g2o
    // solver for PnP/3D VO
    // pose dimension is 6 (se3), landmark dimension is 3.
    using BlockSolverType = g2o::BlockSolver_6_3;
    // 线性求解器类型(稠密)
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;

    // use LM (N.B.: 可以换优化搜索算法)
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>( g2o::make_unique<LinearSolverType>() ));
    g2o::SparseOptimizer optimizer;  // N.B.: 图模型优化器，无论稀疏还是稠密都是用这个求解器类型
    optimizer.setAlgorithm( solver );  // 设置求解器
    // optimizer.setVerbose( true );  // 打开调试输出

    // Vertex 
    VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
    vertex_pose->setId(0);  // 前端每次只估计一个 relative pose, 顶点的编号从0开始.
    vertex_pose->setEstimate(m_current_frame_->RelativePose());  // 设置相对位姿(相对于参考关键帧)的位姿初值.
    optimizer.addVertex(vertex_pose);

    // K ((左)内参)
    Mat33 K = m_cam_left_->K();

    // N.B.: 当前参考关键帧相对于世界坐标系的位姿.
    SE3 reference_key_pose = m_current_frame_->reference_keyframe_ptr_.lock()->Pose();

    // edges
    int index = 1;  // 边的编号从1开始
    std::vector< EdgeProjectionPoseOnly* > edges;
    std::vector<bool> vEdgeIsOutlier; 
    

    for(size_t i=0; i < m_current_frame_->mvpMapPoints.size(); i++) {
        auto mp = m_current_frame_->mvpMapPoints[i].lock();
        // N.B.: 之后切换看看，是否可以提高精度 !!!!!!
        // if( (mp) && (!mp->is_outlier_) ) {
        if( mp ) {
            Vec3 posCamera = reference_key_pose * mp->GetPos();
            EdgeProjectionPoseOnly* edge = new EdgeProjectionPoseOnly(posCamera, K);
            edge->setId(index);
            // set the ith vertex on hyper-edge to pointer supplied
            // 将超边上的第 i 个顶点设置为提供的指针
            edge->setVertex(0, vertex_pose); 
            edge->setMeasurement( toVec2(m_current_frame_->mvKeysUn[i].pt) );
            // 信息矩阵默认设为单位矩阵, 故测量误差最好白化，或者根据方差设置信息矩阵.
            // 设置信息矩阵为 2x2 单位阵，即 误差向量协方差矩阵的逆为单位阵，亦即协方差矩阵为单位阵(即假设噪声为标准正态分布)
            edge->setInformation( Eigen::Matrix2d::Identity() );
            // edge->setInformation( Mat22::Identity() );
            auto rk = new g2o::RobustKernelHuber();
            // N.B.: delta 默认为 1.
            // N.B.: 可转换为外化参数
            // robust kernel 阈值 (可调)
            // 5.991 自由度为2 (3D-2D, 误差为2D)， 标准差为1个像素(已白化或假定测量特征点位置标准差为1个像素), 置信度为95%(alpha=0.05)的分位数(分位点)
            rk->setDelta(5.991);  // N.B.: 自己加的，估计效果不好再去掉!!!!! (原工程没有)
            edge->setRobustKernel(rk);
            edges.push_back(edge);
            vEdgeIsOutlier.push_back(false);
            optimizer.addEdge(edge);

            
            index++;
        }
    }
    
    // estimate the relative pose and determine the outliers (估计相对位姿并确定异常值)
    // 5.991 自由度为2 (3D-2D, 误差为2D)， 标准差为1个像素(已白化或假定测量特征点位置标准差为1个像素), 置信度为95%(alpha=0.05)的分位数(分位点)
    // N.B.: 可转换为外化参数
    // 阈值(可调) chi2 就是 error*(\Omega)*error, 如果这个数很大，说明此边的值与其他边很不相符, \Omega为信息矩阵
    const double chi2_th = 5.991;
    int cnt_outlier = 0;
    int cnt_inlier = 0;
    // N.B.: 迭代次数(可调), 可转换为外化参数
    // for(int iteration = 0; iteration < 4; iteration++) {
    for(int iteration = 0; iteration < 8; iteration++) {
        vertex_pose->setEstimate(m_current_frame_->RelativePose());  // 设置相对位姿(相对于参考关键帧)的位姿初值.
        optimizer.initializeOptimization();
        optimizer.optimize(10);  // N.B.: 每次优化时，优化器的迭代次数, 可转换为外化参数.
        cnt_outlier = 0;
        cnt_inlier = 0;

        // count the outliers
        for(size_t i=0; i < edges.size(); i++) {
            auto e = edges[i];
            if(vEdgeIsOutlier[i]) {
                // 上一次是外点的，因为这一次迭代没有优化，需要在这一次迭代的基础上重新计算误差，在判断是否还是外点.
                e->computeError();
            }

            // N.B.: 落在拒绝域
            if(e->chi2() > chi2_th) {
                vEdgeIsOutlier[i] = true;
                e->setLevel(1);  // 不优化
                cnt_outlier++;
            } else {
                vEdgeIsOutlier[i] = false;
                e->setLevel(0);  // 优化
                cnt_inlier++;
            }

            // 第三次已经执行完了，外点剔除的差不多了，第四次即最后一次迭代，不使用鲁棒核.
            // if(iteration == 2) {
            if(iteration == 6) {
                // 因为剔除了错误的边，所以第三次之后优化不再使用核函数
                e->setRobustKernel(nullptr);
            }
        }

        if(cnt_inlier < 10) {
            // N.B.: 下一次没有足够的点优化直接退出，不然系统会崩溃
            std::cout << std::endl;
            std::cout << "(Frontend::EstimateCurrentRelativePoseForLast()): cnt_inlier < 10 : " << cnt_inlier << std::endl;
            std::cout << std::endl;
            break;
        }
    }
    
    std::cout << std::endl;
    std::cout << "(Frontend::EstimateCurrentRelativePoseForLast()): Outlier/Inlier in pose estimating: " 
              << cnt_outlier << "/" << cnt_inlier << std::endl;
    std::cout << std::endl;

    // Set pose
    // N.B.: 这里才改变当前帧的位姿
    // vertex_pose->estimate(): return the current estimate of the vertex
    m_current_frame_->SetRelativePose(vertex_pose->estimate());

    return cnt_inlier;  // N.B.: 用于判定是否插入新关键帧
}

// --------------------------------------------------------------------------------------------------------------
int Frontend::EstimateCurrentRelativePoseForLastStereo() {
    // 构建图优化，设定g2o
    // setup g2o
    // solver for PnP/3D VO
    // pose dimension is 6 (se3), landmark dimension is 3.
    using BlockSolverType = g2o::BlockSolver_6_3;
    // 线性求解器类型(稠密)
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;

    // use LM (N.B.: 可以换优化搜索算法)
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>( g2o::make_unique<LinearSolverType>() ));
    
    /**
    // 20221004
    auto solver = new g2o::OptimizationAlgorithmDogleg(
        g2o::make_unique<BlockSolverType>( g2o::make_unique<LinearSolverType>() ));
    **/

    g2o::SparseOptimizer optimizer;  // N.B.: 图模型优化器，无论稀疏还是稠密都是用这个求解器类型
    optimizer.setAlgorithm( solver );  // 设置求解器

    // Vertex
    g2o::VertexSE3Expmap *vertex_pose = new g2o::VertexSE3Expmap();  // camera vertex_pose
    auto frame_relative_pose = m_current_frame_->RelativePose();
    auto frame_relative_pose_SE3Quat = g2o::SE3Quat(frame_relative_pose.rotationMatrix(), frame_relative_pose.translation());
    vertex_pose->setId(0);  // 前端每次只估计一个 relative pose, 顶点的编号从0开始.
    vertex_pose->setEstimate(frame_relative_pose_SE3Quat);  // 设置相对位姿(相对于参考关键帧)的位姿初值.
    optimizer.addVertex(vertex_pose);

    // K((左)内参) 和 左右外参
    auto left_fx = m_cam_left_->fx_;
    auto left_fy = m_cam_left_->fy_;
    auto left_cx = m_cam_left_->cx_;
    auto left_cy = m_cam_left_->cy_;
    auto rl_bf = m_cam_right_->fx_baseline_;

    std::cout << std::endl;
    std::cout << "Frontend::EstimateCurrentRelativePoseForLastStereo(): rl_bf = " << rl_bf << std::endl;
    std::cout << std::endl;

    // N.B.: 当前参考关键帧相对于世界坐标系的位姿.
    SE3 reference_key_pose = m_current_frame_->reference_keyframe_ptr_.lock()->Pose();

    // edges
    int index = 1;  // 边的编号从1开始
    int edges_number = 0;  // 
    std::vector< g2o::EdgeStereoSE3ProjectXYZOnlyPose* > edges;
    std::vector< bool > vEdgeIsOutlier;

    const double thHuberStereo = sqrt(7.815); // robust kernel 阈值 (可调)
    for(size_t i=0; i < m_current_frame_->mvpMapPoints.size(); i++) {
        // 是否同时与右图片有匹配, 否则跳过
        if(m_current_frame_->mvKeysR_u[i] < 0.0) {
            continue;
        }

        auto mp = m_current_frame_->mvpMapPoints[i].lock();

        if( mp ) {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose *edge = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
            edge->setId(index);
            edge->setVertex(0, vertex_pose);
            Vec3 StereoMeasure;
            StereoMeasure = toVec3(m_current_frame_->mvKeysUn[i].pt, m_current_frame_->mvKeysR_u[i]);
            edge->setMeasurement( StereoMeasure );
            edge->setInformation( Mat33::Identity() );  // 信息矩阵默认设为单位矩阵, 故测量误差最好白化，或者根据方差设置信息矩阵.
            auto rk = new g2o::RobustKernelHuber();
            rk->setDelta(thHuberStereo);
            edge->setRobustKernel(rk);

            edge->fx = left_fx;
            edge->fy = left_fy;
            edge->cx = left_cx;
            edge->cy = left_cy;
            edge->bf = rl_bf;

            Vec3 posCamera = reference_key_pose * mp->GetPos();
            // edge->Xw = posCamera;
            edge->Xw[0] = posCamera[0];
            edge->Xw[1] = posCamera[1];
            edge->Xw[2] = posCamera[2];

            edges.push_back(edge);
            vEdgeIsOutlier.push_back(false);
            optimizer.addEdge(edge);

            index++;
            edges_number++;
        }
    }

    if(edges_number < 10) {
        std::cout << std::endl;
        std::cout << "(Frontend::EstimateCurrentRelativePoseForLastStereo): Not enough points not to perform optimization " << std::endl;
        std::cout << "(Frontend::EstimateCurrentRelativePoseForLastStereo): edges_number = " << edges_number << std::endl;
        std::cout << std::endl;
        return 0;
    }

    double chi2_th = 7.815;
    int cnt_outlier = 0;
    int cnt_inlier = 0;
    for(int iteration = 0; iteration < 8; iteration++) {
        auto f_relative_pose = m_current_frame_->RelativePose();
        auto f_relative_pose_SE3Quat = g2o::SE3Quat(f_relative_pose.rotationMatrix(), f_relative_pose.translation());
        vertex_pose->setEstimate(f_relative_pose_SE3Quat);
        optimizer.initializeOptimization();
        optimizer.optimize(10);  // N.B.: 每次优化时，优化器的迭代次数, 可转换为外化参数.
        cnt_outlier = 0;
        cnt_inlier = 0;

        // count the outliers
        for(size_t i=0; i < edges.size(); i++) {
            auto e = edges[i];
            if(vEdgeIsOutlier[i]) {
                // 上一次是外点的，因为这一次迭代没有优化，需要在这一次迭代的基础上重新计算误差，在判断是否还是外点.
                e->computeError();
            }

             // N.B.: 落在拒绝域
            if(e->chi2() > chi2_th) {
                vEdgeIsOutlier[i] = true;
                e->setLevel(1);  // 不优化
                cnt_outlier++;
            } else {
                vEdgeIsOutlier[i] = false;
                e->setLevel(0);  // 优化
                cnt_inlier++;
            }

            // 倒数第二次已经执行完了，外点剔除的差不多了，最后一次迭代，不使用鲁棒核.
            // if(iteration == 2) {
            if(iteration == 6) {
                // 因为剔除了错误的边，所以第三次之后优化不再使用核函数
                e->setRobustKernel(nullptr);
            }
            
        }

        if(cnt_inlier < 10) {
            // N.B.: 下一次没有足够的点优化直接退出，不然系统会崩溃
            std::cout << std::endl;
            std::cout << "(Frontend::EstimateCurrentRelativePoseForLastStereo()): cnt_inlier < 10 : " << cnt_inlier << std::endl;
            std::cout << std::endl;
            break;
        }
    }

    std::cout << std::endl;
    std::cout << "(Frontend::EstimateCurrentRelativePoseForLastStereo()): Outlier/Inlier in pose estimating: " 
              << cnt_outlier << "/" << cnt_inlier << std::endl;
    std::cout << std::endl;

    // Set pose
    // N.B.: 这里才改变当前帧的位姿
    // vertex_pose->estimate(): return the current estimate of the vertex
    auto f_est_pose_SE3Quat = vertex_pose->estimate();
    auto f_est_q = f_est_pose_SE3Quat.rotation();
    auto f_est_t = f_est_pose_SE3Quat.translation();
    auto f_est_pose = SE3(f_est_q, f_est_t);
    m_current_frame_->SetRelativePose(f_est_pose);

    return cnt_inlier;  // N.B.: 用于判定是否插入新关键帧
}

// --------------------------------------------------------------------------------------------------------------
// 参见 sg_slam/frontend.cpp int Frontend::TrackLastFrame()
int Frontend::TrackReferenceKeyframe() {
    auto desc_current = convertDesVector2Tensor(m_current_frame_->mDescriptors_data, m_current_frame_->Descripters_B, m_current_frame_->Descripters_D, m_current_frame_->Descripters_N);
    auto current_reference_key = m_current_frame_->reference_keyframe_ptr_.lock();
    auto desc_reference = convertDesVector2Tensor(current_reference_key->mDescriptors_data, current_reference_key->Descripters_B, current_reference_key->Descripters_D, current_reference_key->Descripters_N);

    auto matcher_out_dict_track = sg_DetectMatcher::descMatch(desc_current, desc_reference, m_sg_Track_match_threshold_);
    auto matcher0to1_tensor_track = matcher_out_dict_track["matches0"][0].to(torch::kCPU);
    std::vector<cv::DMatch> matchers_v;
    for(int64_t i=0; i<matcher0to1_tensor_track.size(0); i++) {
        int index0to1 = matcher0to1_tensor_track[i].item<int>();

        if(index0to1 > -1) {
            cv::DMatch match1to2_one;
            match1to2_one.queryIdx = i;
            match1to2_one.trainIdx = index0to1;
            matchers_v.push_back(match1to2_one);
        }
    }

    // N.B.: 关键点集合没有缩小
    // N.B.: 这里写错了 !!!!!!
    // auto RR_matchers_v = sg_slam::sg_RANSAC(m_current_frame_->mvKeys, m_last_frame_->mvKeys, matchers_v, m_num_features_Threshold_tracking_);
    auto RR_matchers_v = sg_slam::sg_RANSAC(m_current_frame_->mvKeys, current_reference_key->mvKeys, matchers_v, m_num_features_Threshold_tracking_);

    m_currentForReference_matches_with_mappoint_set_.clear();
    int cnt_track_reference_landmarks = 0;
    for(size_t i=0; i<RR_matchers_v.size(); i++) {
    // for(size_t i=0; i<matchers_v.size(); i++) {
        auto feat = current_reference_key->mvpfeatures[RR_matchers_v[i].trainIdx];
        // auto feat = current_reference_key->mvpfeatures[matchers_v[i].trainIdx];
        if(feat) {
            // N.B.: 本系统中特征中一定包含地图点，故本次判断可能多余!!!!!
            // auto mp = feat->map_point_.lock();
            auto mp = feat->GetMapPoint();
            if(mp) {
                int currentKeypointIndex = RR_matchers_v[i].queryIdx;
                int ReferenceKeypointIndex = RR_matchers_v[i].trainIdx;
                // int currentKeypointIndex = matchers_v[i].queryIdx;
                // int ReferenceKeypointIndex = matchers_v[i].trainIdx;

                // 属于同一关键点的匹配不应两次插入到有效匹配中
                if(m_currentForReference_matches_with_mappoint_set_.find({currentKeypointIndex, ReferenceKeypointIndex}) != m_currentForReference_matches_with_mappoint_set_.end()) {
                    continue;
                } 

                m_currentForReference_matches_with_mappoint_set_.insert({currentKeypointIndex, ReferenceKeypointIndex});
                cnt_track_reference_landmarks++;
            }
        }
    }

    return cnt_track_reference_landmarks;
}

// --------------------------------------------------------------------------------------------------------------
int Frontend::TrackReferenceKeyframeNew() {
    auto desc_current = convertDesVector2Tensor(m_current_frame_->mDescriptors_data, m_current_frame_->Descripters_B, m_current_frame_->Descripters_D, m_current_frame_->Descripters_N);
    auto current_reference_key = m_current_frame_->reference_keyframe_ptr_.lock();
    auto desc_reference = convertDesVector2Tensor(current_reference_key->mDescriptors_data, current_reference_key->Descripters_B, current_reference_key->Descripters_D, current_reference_key->Descripters_N);

    auto matcher_out_dict_track = sg_DetectMatcher::descMatch(desc_current, desc_reference, m_sg_Track_match_threshold_);
    auto matcher0to1_tensor_track = matcher_out_dict_track["matches0"][0].to(torch::kCPU);
    std::vector<cv::DMatch> matchers_v;
    for(int64_t i=0; i<matcher0to1_tensor_track.size(0); i++) {
        int index0to1 = matcher0to1_tensor_track[i].item<int>();

        if(index0to1 > -1) {
            cv::DMatch match1to2_one;
            match1to2_one.queryIdx = i;
            match1to2_one.trainIdx = index0to1;
            matchers_v.push_back(match1to2_one);
        }
    }

    // N.B.: 关键点集合没有缩小
    auto RR_matchers_v = sg_slam::sg_RANSAC(m_current_frame_->mvKeys, current_reference_key->mvKeys, matchers_v, m_num_features_Threshold_tracking_);
    // N.B.: 根据匹配将有对应地图点的关键帧的关键点的对应地图点填充到当前帧的相应位置
    // m_current_frame_->mvpMapPoints.resize(m_current_frame_->mvKeysUn.size());
    // m_currentForReference_matches_with_mappoint_set_.clear();
    int cnt_track_reference_landmarks = 0;
    for(size_t i=0; i<RR_matchers_v.size(); i++) {
        auto feat = current_reference_key->mvpfeatures[RR_matchers_v[i].trainIdx];
        if(feat) {
            auto mp = feat->GetMapPoint();
            if(mp) {
                /**
                int currentKeypointIndex = RR_matchers_v[i].queryIdx;
                int ReferenceKeypointIndex = RR_matchers_v[i].trainIdx;

                // 属于同一关键点的匹配不应两次插入到有效匹配中
                if(m_currentForReference_matches_with_mappoint_set_.find({currentKeypointIndex, ReferenceKeypointIndex}) != m_currentForReference_matches_with_mappoint_set_.end()) {
                    continue;
                }
                **/
                // m_currentForReference_matches_with_mappoint_set_.insert({currentKeypointIndex, ReferenceKeypointIndex});
                m_current_frame_->mvpMapPoints[RR_matchers_v[i].queryIdx] = mp;
                cnt_track_reference_landmarks++;
            }
        }
    }

    m_current_frame_->keypoints_number = cnt_track_reference_landmarks;

    return cnt_track_reference_landmarks;
}

// --------------------------------------------------------------------------------------------------------------
int Frontend::TrackLocalKeyframes() {
    auto desc_current = convertDesVector2Tensor(m_current_frame_->mDescriptors_data, m_current_frame_->Descripters_B, m_current_frame_->Descripters_D, m_current_frame_->Descripters_N);
    unsigned long current_ref_key_id = m_current_frame_->reference_keyframe_ptr_.lock()->id_;

    int cnt_track_local_keyframes_landmarks = 0;

    auto active_keyframes_map = m_map_->GetActiveKeyFrames();

    size_t keyframes_number = 0;


    // for(auto it=m_local_keyframes_deque.begin(); it!=m_local_keyframes_deque.end(); it++) {
    for(auto &keyframe : active_keyframes_map) {
        unsigned long kfId = keyframe.first;
        if(kfId == current_ref_key_id) {
            continue;
        }

        // auto current_key = (*it).lock();
        auto current_key = keyframe.second;
        auto desc_key = convertDesVector2Tensor(current_key->mDescriptors_data, current_key->Descripters_B, current_key->Descripters_D, current_key->Descripters_N);
        auto matcher_out_dict_track = sg_DetectMatcher::descMatch(desc_current, desc_key, m_sg_Track_match_threshold_);
        auto matcher0to1_tensor_track = matcher_out_dict_track["matches0"][0].to(torch::kCPU);
        std::vector<cv::DMatch> matchers_v;
        for(int64_t i=0; i<matcher0to1_tensor_track.size(0); i++) {
            int index0to1 = matcher0to1_tensor_track[i].item<int>();

            if(index0to1 > -1) {
                cv::DMatch match1to2_one;
                match1to2_one.queryIdx = i;
                match1to2_one.trainIdx = index0to1;
                matchers_v.push_back(match1to2_one);
            }
        }

        // N.B.: 关键点集合没有缩小
        auto RR_matchers_v = sg_slam::sg_RANSAC(m_current_frame_->mvKeys, current_key->mvKeys, matchers_v, m_num_features_Threshold_tracking_);
        for(size_t i=0; i<RR_matchers_v.size(); i++) {
            /**
            auto current_mp = m_current_frame_->mvpMapPoints[RR_matchers_v[i].queryIdx].lock();
            // N.B.: 已有对应地图点不补.
            if(current_mp) {
                continue;
            }
            **/

            auto feat = current_key->mvpfeatures[RR_matchers_v[i].trainIdx];
            if(feat) {
                auto mp = feat->GetMapPoint();
                if(mp) {
                    m_current_frame_->mvpMapPoints[RR_matchers_v[i].queryIdx] = mp;
                    cnt_track_local_keyframes_landmarks++;
                }
            }
        }

        keyframes_number++;

        if(keyframes_number > m_local_keyframes_deque_length) {
            break;
        }
    }

    int track_ref_landmarks = m_current_frame_->keypoints_number;
    m_current_frame_->keypoints_number = (track_ref_landmarks + cnt_track_local_keyframes_landmarks);

    std::cout << std::endl;
    std::cout << "(Frontend::TrackLocalKeyframes()): track_ref_landmarks = " << track_ref_landmarks << std::endl;
    std::cout << "(Frontend::TrackLocalKeyframes()): cnt_track_local_keyframes_landmarks = " << cnt_track_local_keyframes_landmarks << std::endl;
    std::cout << "(Frontend::TrackLocalKeyframes()): m_current_frame_->keypoints_number = " << m_current_frame_->keypoints_number << std::endl;
    std::cout << "(Frontend::TrackLocalKeyframes()): m_local_keyframes_deque.size() = " << m_local_keyframes_deque.size() << std::endl;
    std::cout << std::endl;

    return cnt_track_local_keyframes_landmarks;
}

// --------------------------------------------------------------------------------------------------------------
// 参见 sg_slam/frontend.cpp int Frontend::EstimateCurrentRelativePoseForLast()
int Frontend::EstimateCurrentRelativePoseForReference(const SE3 &intial_estimate, int last_inliers) {
    // 构建图优化，设定g2o
    // setup g2o
    // solver for PnP/3D VO
    // pose dimension is 6 (se3), landmark dimension is 3.
    using BlockSolverType = g2o::BlockSolver_6_3;
    // 线性求解器类型(稠密)
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;

    // use LM (N.B.: 可以换优化搜索算法)
    auto solver = new g2o::OptimizationAlgorithmLevenberg( 
        g2o::make_unique<BlockSolverType>( g2o::make_unique<LinearSolverType>() ));
    g2o::SparseOptimizer optimizer;  // N.B.: 图模型优化器，无论稀疏还是稠密都是用这个求解器类型
    optimizer.setAlgorithm( solver );  // 设置求解器
    // optimizer.setVerbose( true );  // 打开调试输出

    // Vertex
    VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
    vertex_pose->setId(0);  // 前端每次只估计一个 relative pose, 顶点的编号从0开始.
    // // N.B.: 不同点 (对与last)
    vertex_pose->setEstimate(intial_estimate);  // 设置相对位姿(相对于参考关键帧)的位姿初值.
    optimizer.addVertex(vertex_pose);

    // K ((左)内参)
    Mat33 K = m_cam_left_->K();

    // 当前帧的参考关键帧
    auto current_reference_key = m_current_frame_->reference_keyframe_ptr_.lock();
    // N.B.: 当前参考关键帧相对于世界坐标系的位姿.
    SE3 reference_key_pose = current_reference_key->Pose();

    // edges
    int index = 1; // 边的编号从1开始
    std::vector< EdgeProjectionPoseOnly* > edges;
    std::vector<bool> vEdgeIsOutlier;
    int cnt_inlier = 0;

    for(auto iter = m_currentForReference_matches_with_mappoint_set_.begin(); iter != m_currentForReference_matches_with_mappoint_set_.end(); iter++) {
        int currentKeypointIndex = (*iter).first;
        int ReferenceKeypointIndex = (*iter).second;

        // // N.B.: 不同点 (对与last)
        // auto mp = current_reference_key->mvpfeatures[ReferenceKeypointIndex]->map_point_.lock();
        auto KF_feat = current_reference_key->mvpfeatures[ReferenceKeypointIndex];

        // N.B.: 为空时，若操作会产生段错误 !!!!!!
        if(KF_feat == nullptr) {
            std::cout << std::endl;
            std::cout << "(Frontend::EstimateCurrentRelativePoseForReference()): KF_feat nullptr" << std::endl;
            std::cout << std::endl;
            continue;
        }

        // auto mp = current_reference_key->mvpfeatures[ReferenceKeypointIndex]->GetMapPoint();
        auto mp = KF_feat->GetMapPoint();
        auto point2d = m_current_frame_->mvKeysUn[currentKeypointIndex].pt;

        // assert(mp != nullptr); // N.B.: 经过上面操作，这个断语都会通过.
        if( mp ) {
            Vec3 posCamera = reference_key_pose * mp->GetPos();
            EdgeProjectionPoseOnly* edge = new EdgeProjectionPoseOnly(posCamera, K);
            edge->setId(index);
            // set the ith vertex on hyper-edge to pointer supplied
            // 将超边上的第 i 个顶点设置为提供的指针
            edge->setVertex(0, vertex_pose);
            // // N.B.: 不同点 (对与last)
            edge->setMeasurement( toVec2(point2d) );
            // 信息矩阵默认设为单位矩阵, 故测量误差最好白化，或者根据方差设置信息矩阵.
            // 设置信息矩阵为 2x2 单位阵，即 误差向量协方差矩阵的逆为单位阵，亦即协方差矩阵为单位阵(即假设噪声为标准正态分布)
            edge->setInformation( Eigen::Matrix2d::Identity() );
            // edge->setInformation( Mat22::Identity() );
            auto rk = new g2o::RobustKernelHuber();
            // N.B.: delta 默认为 1.
            // N.B.: 可转换为外化参数
            // robust kernel 阈值 (可调)
            // 5.991 自由度为2 (3D-2D, 误差为2D)， 标准差为1个像素(已白化或假定测量特征点位置标准差为1个像素), 置信度为95%(alpha=0.05)的分位数(分位点)
            rk->setDelta(5.991);  // N.B.: 自己加的，估计效果不好再去掉!!!!! (原工程没有)
            edge->setRobustKernel(rk);
            edges.push_back(edge);
            vEdgeIsOutlier.push_back(false);
            optimizer.addEdge(edge);

            cnt_inlier++;
            index++;
        }
    }

    std::cout << std::endl;
    std::cout << "(Frontend::EstimateCurrentRelativePoseForReference()): LOST mp =  " << cnt_inlier << std::endl;
    // std::cout << "(Frontend::EstimateCurrentRelativePoseForReference()): LOST cnt_mp_kpR =  " << cnt_mp_kpR << std::endl;
    // std::cout << "(Frontend::EstimateCurrentRelativePoseForReference()): LOST cnt_mp_kpR =  " << cnt_matches_with_kpR << std::endl;
    std::cout << std::endl;

    // estimate the relative pose and determine the outliers (估计相对位姿并确定异常值)
    // 5.991 自由度为2 (3D-2D, 误差为2D)， 标准差为1个像素(已白化或假定测量特征点位置标准差为1个像素), 置信度为95%(alpha=0.05)的分位数(分位点)
    // N.B.: 可转换为外化参数
    // 阈值(可调) chi2 就是 error*(\Omega)*error, 如果这个数很大，说明此边的值与其他边很不相符, \Omega为信息矩阵
    
    if(cnt_inlier < 10) {
        std::cout << std::endl;
        std::cout << "(Frontend::EstimateCurrentRelativePoseForReference()): Not enough points not to perform optimization " << std::endl;
        std::cout << "(Frontend::EstimateCurrentRelativePoseForReference()): cnt_inlier = " << cnt_inlier << std::endl;
        std::cout << std::endl;
        return 0;
    }
    
    const double chi2_th = 5.991;
    int cnt_outlier = 0;
    cnt_inlier = 0;
    // N.B.: 迭代次数(可调), 可转换为外化参数
    // for(int iteration = 0; iteration < 4; iteration++) {
    for(int iteration = 0; iteration < 8; iteration++) {
        // // N.B.: 不同点 (对与last)
        vertex_pose->setEstimate(intial_estimate);
        optimizer.initializeOptimization();
        optimizer.optimize(10);  // N.B.: 每次优化时，优化器的迭代次数, 可转换为外化参数.
        cnt_outlier = 0;
        cnt_inlier = 0;

        // count the outliers
        for(size_t i=0; i < edges.size(); i++) {
            auto e = edges[i];
            if(vEdgeIsOutlier[i]) {
                // 上一次是外点的，因为这一次迭代没有优化，需要在这一次迭代的基础上重新计算误差，在判断是否还是外点.
                e->computeError();
            }

            // N.B.: 落在拒绝域
            if(e->chi2() > chi2_th) {
                vEdgeIsOutlier[i] = true;
                e->setLevel(1);  // 不优化
                cnt_outlier++;
            } else {
                vEdgeIsOutlier[i] = false;
                e->setLevel(0);  // 优化
                cnt_inlier++;
            }


            
            // 第三次已经执行完了，外点剔除的差不多了，第四次即最后一次迭代，不使用鲁棒核.
            // if(iteration == 2) {
            if(iteration == 6) {
                // 因为剔除了错误的边，所以第三次之后优化不再使用核函数
                e->setRobustKernel(nullptr);
            }
            
        }
        
        if(cnt_inlier < 10) {
            // N.B.: 下一次没有足够的点优化直接退出，不然系统会崩溃
            std::cout << std::endl;
            std::cout << "(Frontend::EstimateCurrentRelativePoseForReference()): cnt_inlier < 10 : " << cnt_inlier << std::endl;
            std::cout << std::endl;
            break;
        }
    }

    std::cout << std::endl;
    std::cout << "(Frontend::EstimateCurrentRelativePoseForReference()): Outlier/Inlier in pose estimating: " 
              << cnt_outlier<< "/" << cnt_inlier<< std::endl;
    std::cout << std::endl;

    if(cnt_inlier < 5) {
        std::cout << std::endl;
        std::cout << "(Frontend::EstimateCurrentRelativePoseForReference()): Not enough points not to perform fuse " << std::endl;
        std::cout << std::endl;
        return 0;
    }

    // // N.B.: 不同点 (对与last)
    Vec6 current_relativePose_before_se3 = m_current_frame_->RelativePose().log();  // 融合前相对位姿李代数
    Vec6 current_ref_se3 = vertex_pose->estimate().log();

    double fuse_alpha = static_cast<double>(last_inliers) / static_cast<double>(last_inliers + cnt_inlier);

    fuse_alpha = 2.0 * fuse_alpha;  // N.B.: 进行简单的线性缩放(使范围大概在0-1之间)
    if(fuse_alpha > 1.0) {
        fuse_alpha = 1.0;
    }

    Vec6 relativePose_fuse_se3 = fuse_alpha*current_relativePose_before_se3 + (1.0 - fuse_alpha)*current_ref_se3;

    SE3 relativePose_fuse = SE3::exp(relativePose_fuse_se3);

    m_current_frame_->SetRelativePose(relativePose_fuse);


    std::cout << std::endl;
    std::cout << "(Frontend::EstimateCurrentRelativePoseForReference()): fuse_alpha = " << fuse_alpha << std::endl;
    std::cout << std::endl;

    return cnt_inlier;  
}

// --------------------------------------------------------------------------------------------------------------
int Frontend::EstimateCurrentRelativePoseForReferenceStereo(const SE3 &intial_estimate, int last_inliers) {
    // 构建图优化，设定g2o
    // setup g2o
    // solver for PnP/3D VO
    // pose dimension is 6 (se3), landmark dimension is 3.
    using BlockSolverType = g2o::BlockSolver_6_3;
    // 线性求解器类型(稠密)
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;

    // use LM (N.B.: 可以换优化搜索算法)
    auto solver = new g2o::OptimizationAlgorithmLevenberg( 
        g2o::make_unique<BlockSolverType>( g2o::make_unique<LinearSolverType>() ));

    /**
    // 20221004
    auto solver = new g2o::OptimizationAlgorithmDogleg(
        g2o::make_unique<BlockSolverType>( g2o::make_unique<LinearSolverType>() ));
    **/
    
    g2o::SparseOptimizer optimizer;  // N.B.: 图模型优化器，无论稀疏还是稠密都是用这个求解器类型
    optimizer.setAlgorithm( solver );  // 设置求解器
    // optimizer.setVerbose( true );  // 打开调试输出

    // Vertex
    g2o::VertexSE3Expmap *vertex_pose = new g2o::VertexSE3Expmap();  // camera vertex_pose
    // // N.B.: 不同点 (对与last)
    auto frame_relative_pose_SE3Quat = g2o::SE3Quat(intial_estimate.rotationMatrix(), intial_estimate.translation());
    vertex_pose->setId(0);  // 前端每次只估计一个 relative pose, 顶点的编号从0开始.
    vertex_pose->setEstimate(frame_relative_pose_SE3Quat);  // 设置相对位姿(相对于参考关键帧)的位姿初值.
    optimizer.addVertex(vertex_pose);

    // K((左)内参) 和 左右外参
    auto left_fx = m_cam_left_->fx_;
    auto left_fy = m_cam_left_->fy_;
    auto left_cx = m_cam_left_->cx_;
    auto left_cy = m_cam_left_->cy_;
    auto rl_bf = m_cam_right_->fx_baseline_;

    std::cout << std::endl;
    std::cout << "Frontend::EstimateCurrentRelativePoseForReferenceStereo(...): rl_bf = " << rl_bf << std::endl;
    std::cout << std::endl;

    // 当前帧的参考关键帧
    auto current_reference_key = m_current_frame_->reference_keyframe_ptr_.lock();
    // N.B.: 当前参考关键帧相对于世界坐标系的位姿.
    SE3 reference_key_pose = current_reference_key->Pose();

    // edges
    int index = 1;  // 边的编号从1开始
    std::vector< g2o::EdgeStereoSE3ProjectXYZOnlyPose* > edges;
    std::vector< bool > vEdgeIsOutlier;
    std::vector< g2o::EdgeSE3ProjectXYZOnlyPose* > edges2d;
    std::vector< bool > vEdgeIsOutlier2d;

    int cnt_inlier = 0;

    const double thHuberStereo = sqrt(7.815); // robust kernel 阈值 (可调)
    const double thHuber2d = sqrt(5.991);
    for(auto iter = m_currentForReference_matches_with_mappoint_set_.begin(); iter != m_currentForReference_matches_with_mappoint_set_.end(); iter++) {
        int currentKeypointIndex = (*iter).first;
        int ReferenceKeypointIndex = (*iter).second;

        // // N.B.: 不同点 (对与last)
        auto KF_feat = current_reference_key->mvpfeatures[ReferenceKeypointIndex];

        // N.B.: 为空时，若操作会产生段错误 !!!!!!
        if(KF_feat == nullptr) {
            std::cout << std::endl;
            std::cout << "(Frontend::EstimateCurrentRelativePoseForReferenceStereo): KF_feat nullptr" << std::endl;
            std::cout << std::endl;
            continue;
        }

        auto mp = KF_feat->GetMapPoint();

        if(mp == nullptr) {
            continue;
        }

        auto point_kR_u = m_current_frame_->mvKeysR_u[currentKeypointIndex];

        if(point_kR_u > -1.0) {
            
            g2o::EdgeStereoSE3ProjectXYZOnlyPose *edge = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
            edge->setId(index);
            edge->setVertex(0, vertex_pose);
            Vec3 StereoMeasure;
            auto point2d = m_current_frame_->mvKeysUn[currentKeypointIndex].pt;
            StereoMeasure = toVec3(point2d, point_kR_u);
            edge->setMeasurement( StereoMeasure );
            edge->setInformation( Mat33::Identity() );  // 信息矩阵默认设为单位矩阵, 故测量误差最好白化，或者根据方差设置信息矩阵.
            auto rk = new g2o::RobustKernelHuber();
            rk->setDelta(thHuberStereo);
            edge->setRobustKernel(rk);

            edge->fx = left_fx;
            edge->fy = left_fy;
            edge->cx = left_cx;
            edge->cy = left_cy;
            edge->bf = rl_bf;

            Vec3 posCamera = reference_key_pose * mp->GetPos();

            // edge->Xw = posCamera;
            edge->Xw[0] = posCamera[0];
            edge->Xw[1] = posCamera[1];
            edge->Xw[2] = posCamera[2];

            edges.push_back(edge);
            vEdgeIsOutlier.push_back(false);
            optimizer.addEdge(edge);

            cnt_inlier++;
            index++;
        } else {
            g2o::EdgeSE3ProjectXYZOnlyPose *edge = new g2o::EdgeSE3ProjectXYZOnlyPose();
            edge->setId(index);
            edge->setVertex(0, vertex_pose);
            auto point2d = m_current_frame_->mvKeysUn[currentKeypointIndex].pt;
            Vec2 measure2d;
            measure2d = toVec2(point2d);
            edge->setMeasurement(measure2d);
            edge->setInformation( Mat22::Identity() );
            auto rk = new g2o::RobustKernelHuber(); 
            rk->setDelta(thHuber2d);
            edge->setRobustKernel(rk);
            edge->fx = left_fx;
            edge->fy = left_fy;
            edge->cx = left_cx;
            edge->cy = left_cy;

            Vec3 posCamera = reference_key_pose * mp->GetPos();

            // edge->Xw = posCamera;
            edge->Xw[0] = posCamera[0];
            edge->Xw[1] = posCamera[1];
            edge->Xw[2] = posCamera[2];

            edges2d.push_back(edge);
            vEdgeIsOutlier2d.push_back(false);
            optimizer.addEdge(edge);

            cnt_inlier++;
            index++;
        }
    }

    if(cnt_inlier < 10) {
        std::cout << std::endl;
        std::cout << "(Frontend::EstimateCurrentRelativePoseForReferenceStereo): Not enough points not to perform optimization " << std::endl;
        std::cout << "(Frontend::EstimateCurrentRelativePoseForReferenceStereo): cnt_inlier = " << cnt_inlier << std::endl;
        std::cout << std::endl;
        return 0;
    }

    const double chi2_th = 7.815;
    const double chi2_2d_th = 5.991;
    int cnt_outlier = 0;
    cnt_inlier = 0;
    // N.B.: 迭代次数(可调), 可转换为外化参数
    // for(int iteration = 0; iteration < 4; iteration++) {
    for(int iteration = 0; iteration < 8; iteration++) {
        // // N.B.: 不同点 (对与last)
        frame_relative_pose_SE3Quat = g2o::SE3Quat(intial_estimate.rotationMatrix(), intial_estimate.translation());
        vertex_pose->setEstimate(frame_relative_pose_SE3Quat);
        optimizer.initializeOptimization();
        optimizer.optimize(10);  // N.B.: 每次优化时，优化器的迭代次数, 可转换为外化参数.
        cnt_outlier = 0;
        cnt_inlier = 0;

        // count the outliers
        for(size_t i=0; i < edges.size(); i++) {
            auto e = edges[i];
            if(vEdgeIsOutlier[i]) {
                // 上一次是外点的，因为这一次迭代没有优化，需要在这一次迭代的基础上重新计算误差，在判断是否还是外点.
                e->computeError();
            }

            // N.B.: 落在拒绝域
            if(e->chi2() > chi2_th) {
                vEdgeIsOutlier[i] = true;
                e->setLevel(1);  // 不优化
                cnt_outlier++;
            } else {
                vEdgeIsOutlier[i] = false;
                e->setLevel(0);  // 优化
                cnt_inlier++;
            }


            
            // 第三次已经执行完了，外点剔除的差不多了，第四次即最后一次迭代，不使用鲁棒核.
            // if(iteration == 2) {
            if(iteration == 6) {
                // 因为剔除了错误的边，所以第三次之后优化不再使用核函数
                e->setRobustKernel(nullptr);
            }
            
        }

        // count the outliers
        for(size_t i=0; i < edges2d.size(); i++) {
            auto e = edges2d[i];
            if(vEdgeIsOutlier2d[i]) {
                // 上一次是外点的，因为这一次迭代没有优化，需要在这一次迭代的基础上重新计算误差，在判断是否还是外点.
                e->computeError();
            }

            // N.B.: 落在拒绝域
            if(e->chi2() > chi2_2d_th) {
                vEdgeIsOutlier2d[i] = true;
                e->setLevel(1);  // 不优化
                cnt_outlier++;
            } else {
                vEdgeIsOutlier2d[i] = false;
                e->setLevel(0);  // 优化
                cnt_inlier++;
            }


            
            // 第三次已经执行完了，外点剔除的差不多了，第四次即最后一次迭代，不使用鲁棒核.
            // if(iteration == 2) {
            if(iteration == 6) {
                // 因为剔除了错误的边，所以第三次之后优化不再使用核函数
                e->setRobustKernel(nullptr);
            }
            
        }

        if(cnt_inlier < 10) {
            // N.B.: 下一次没有足够的点优化直接退出，不然系统会崩溃
            std::cout << std::endl;
            std::cout << "(Frontend::EstimateCurrentRelativePoseForReferenceStereo): cnt_inlier < 10 : " << cnt_inlier << std::endl;
            std::cout << std::endl;
            break;
        }
    }

    std::cout << std::endl;
    std::cout << "(Frontend::EstimateCurrentRelativePoseForReferenceStereo): Outlier/Inlier in pose estimating: " 
              << cnt_outlier<< "/" << cnt_inlier<< std::endl;
    std::cout << std::endl;

    if(cnt_inlier < 5) {
        std::cout << std::endl;
        std::cout << "(Frontend::EstimateCurrentRelativePoseForReferenceStereo): Not enough points not to perform fuse " << std::endl;
        std::cout << std::endl;
        return 0;
    }

    // // N.B.: 不同点 (对与last)
    Vec6 current_relativePose_before_se3 = m_current_frame_->RelativePose().log();  // 融合前相对位姿李代数
    auto f_est_pose_SE3Quat = vertex_pose->estimate();
    auto f_est_q = f_est_pose_SE3Quat.rotation();
    auto f_est_t = f_est_pose_SE3Quat.translation();
    auto f_est_pose = SE3(f_est_q, f_est_t);
    Vec6 current_ref_se3 = f_est_pose.log();

    double fuse_alpha = static_cast<double>(last_inliers) / static_cast<double>(last_inliers + cnt_inlier);

    fuse_alpha = 2.0 * fuse_alpha;  // N.B.: 进行简单的线性缩放(使范围大概在0-1之间)
    if(fuse_alpha > 1.0) {
        fuse_alpha = 1.0;
    }

    Vec6 relativePose_fuse_se3 = fuse_alpha*current_relativePose_before_se3 + (1.0 - fuse_alpha)*current_ref_se3;
    SE3 relativePose_fuse = SE3::exp(relativePose_fuse_se3);

    m_current_frame_->SetRelativePose(relativePose_fuse);

    std::cout << std::endl;
    std::cout << "(Frontend::EstimateCurrentRelativePoseForReferenceStereo): fuse_alpha = " << fuse_alpha << std::endl;
    std::cout << std::endl;

    return cnt_inlier; 
}

// --------------------------------------------------------------------------------------------------------------
int Frontend::EstimateCurrentRelativePoseForReferenceNew(const SE3 &intial_estimate) {
    // 构建图优化，设定g2o
    // setup g2o
    // solver for PnP/3D VO
    // pose dimension is 6 (se3), landmark dimension is 3.
    using BlockSolverType = g2o::BlockSolver_6_3;
    // 线性求解器类型(稠密)
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;

    // use LM (N.B.: 可以换优化搜索算法)
    auto solver = new g2o::OptimizationAlgorithmLevenberg( 
        g2o::make_unique<BlockSolverType>( g2o::make_unique<LinearSolverType>() ));

    g2o::SparseOptimizer optimizer;  // N.B.: 图模型优化器，无论稀疏还是稠密都是用这个求解器类型
    optimizer.setAlgorithm( solver );  // 设置求解器
    // optimizer.setVerbose( true );  // 打开调试输出

    // Vertex
    g2o::VertexSE3Expmap *vertex_pose = new g2o::VertexSE3Expmap();  // camera vertex_pose
    // // N.B.: 不同点 (对与last)
    auto frame_relative_pose_SE3Quat = g2o::SE3Quat(intial_estimate.rotationMatrix(), intial_estimate.translation());
    vertex_pose->setId(0);  // 前端每次只估计一个 relative pose, 顶点的编号从0开始.
    vertex_pose->setEstimate(frame_relative_pose_SE3Quat);  // 设置相对位姿(相对于参考关键帧)的位姿初值.
    optimizer.addVertex(vertex_pose);

    // K((左)内参) 和 左右外参
    auto left_fx = m_cam_left_->fx_;
    auto left_fy = m_cam_left_->fy_;
    auto left_cx = m_cam_left_->cx_;
    auto left_cy = m_cam_left_->cy_;
    auto rl_bf = m_cam_right_->fx_baseline_;

    std::cout << std::endl;
    std::cout << "Frontend::EstimateCurrentRelativePoseForReferenceNew(...): rl_bf = " << rl_bf << std::endl;
    std::cout << std::endl;

    // 当前帧的参考关键帧
    auto current_reference_key = m_current_frame_->reference_keyframe_ptr_.lock();
    // N.B.: 当前参考关键帧相对于世界坐标系的位姿.
    SE3 reference_key_pose = current_reference_key->Pose();

    // edges
    int index = 1;  // 边的编号从1开始
    std::vector< g2o::EdgeStereoSE3ProjectXYZOnlyPose* > edges;
    std::vector< bool > vEdgeIsOutlier;
    std::vector< g2o::EdgeSE3ProjectXYZOnlyPose* > edges2d;
    std::vector< bool > vEdgeIsOutlier2d;

    int cnt_inlier = 0;

    const double thHuberStereo = sqrt(7.815); // robust kernel 阈值 (可调)
    const double thHuber2d = sqrt(5.991);
    for(size_t i=0; i < m_current_frame_->mvpMapPoints.size(); i++) {
        auto mp = m_current_frame_->mvpMapPoints[i].lock();

        if(mp == nullptr) {
            continue;
        }

        auto point_kR_u = m_current_frame_->mvKeysR_u[i];

        if(point_kR_u > -1.0) {
        // if(false) {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose *edge = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
            edge->setId(index);
            edge->setVertex(0, vertex_pose);
            Vec3 StereoMeasure;
            auto point2d = m_current_frame_->mvKeysUn[i].pt;
            StereoMeasure = toVec3(point2d, point_kR_u);
            edge->setMeasurement( StereoMeasure );
            edge->setInformation( Mat33::Identity() );  // 信息矩阵默认设为单位矩阵, 故测量误差最好白化，或者根据方差设置信息矩阵.
            auto rk = new g2o::RobustKernelHuber();
            rk->setDelta(thHuberStereo);
            edge->setRobustKernel(rk);

            edge->fx = left_fx;
            edge->fy = left_fy;
            edge->cx = left_cx;
            edge->cy = left_cy;
            edge->bf = rl_bf;

            Vec3 posCamera = reference_key_pose * mp->GetPos();

            // edge->Xw = posCamera;
            edge->Xw[0] = posCamera[0];
            edge->Xw[1] = posCamera[1];
            edge->Xw[2] = posCamera[2];

            edges.push_back(edge);
            vEdgeIsOutlier.push_back(false);
            optimizer.addEdge(edge);

            cnt_inlier++;
            index++;
        } else {
            g2o::EdgeSE3ProjectXYZOnlyPose *edge = new g2o::EdgeSE3ProjectXYZOnlyPose();
            edge->setId(index);
            edge->setVertex(0, vertex_pose);
            auto point2d = m_current_frame_->mvKeysUn[i].pt;
            Vec2 measure2d;
            measure2d = toVec2(point2d);
            edge->setMeasurement(measure2d);
            edge->setInformation( Mat22::Identity() );
            auto rk = new g2o::RobustKernelHuber(); 
            rk->setDelta(thHuber2d);
            edge->setRobustKernel(rk);
            edge->fx = left_fx;
            edge->fy = left_fy;
            edge->cx = left_cx;
            edge->cy = left_cy;

            Vec3 posCamera = reference_key_pose * mp->GetPos();

            // edge->Xw = posCamera;
            edge->Xw[0] = posCamera[0];
            edge->Xw[1] = posCamera[1];
            edge->Xw[2] = posCamera[2];

            edges2d.push_back(edge);
            vEdgeIsOutlier2d.push_back(false);
            optimizer.addEdge(edge);

            cnt_inlier++;
            index++;
        }
    }

    if(cnt_inlier < 10) {
        std::cout << std::endl;
        std::cout << "(Frontend::EstimateCurrentRelativePoseForReferenceNew(...)): Not enough points not to perform optimization " << std::endl;
        std::cout << "(Frontend::EstimateCurrentRelativePoseForReferenceNew(...)): cnt_inlier = " << cnt_inlier << std::endl;
        std::cout << std::endl;
        return 0;
    }

    const double chi2_th = 7.815;
    const double chi2_2d_th = 5.991;
    int cnt_outlier = 0;
    cnt_inlier = 0;
    // N.B.: 迭代次数(可调), 可转换为外化参数
    for(int iteration = 0; iteration < 8; iteration++) {
        frame_relative_pose_SE3Quat = g2o::SE3Quat(intial_estimate.rotationMatrix(), intial_estimate.translation());
        vertex_pose->setEstimate(frame_relative_pose_SE3Quat);
        optimizer.initializeOptimization();
        optimizer.optimize(10);  // N.B.: 每次优化时，优化器的迭代次数, 可转换为外化参数.
        cnt_outlier = 0;
        cnt_inlier = 0;

        // count the outliers
        for(size_t i=0; i < edges.size(); i++) {
            auto e = edges[i];
            if(vEdgeIsOutlier[i]) {
                // 上一次是外点的，因为这一次迭代没有优化，需要在这一次迭代的基础上重新计算误差，在判断是否还是外点.
                e->computeError();
            }

            // N.B.: 落在拒绝域
            if(e->chi2() > chi2_th) {
                vEdgeIsOutlier[i] = true;
                e->setLevel(1);  // 不优化
                cnt_outlier++;
            } else {
                vEdgeIsOutlier[i] = false;
                e->setLevel(0);  // 优化
                cnt_inlier++;
            }

            // 第三次已经执行完了，外点剔除的差不多了，第四次即最后一次迭代，不使用鲁棒核.
            if(iteration == 6) {
                // 因为剔除了错误的边，所以第三次之后优化不再使用核函数
                e->setRobustKernel(nullptr);
            }
            
        }

        // count the outliers
        for(size_t i=0; i < edges2d.size(); i++) {
            auto e = edges2d[i];
            if(vEdgeIsOutlier2d[i]) {
                // 上一次是外点的，因为这一次迭代没有优化，需要在这一次迭代的基础上重新计算误差，在判断是否还是外点.
                e->computeError();
            }

            // N.B.: 落在拒绝域
            if(e->chi2() > chi2_2d_th) {
                vEdgeIsOutlier2d[i] = true;
                e->setLevel(1);  // 不优化
                cnt_outlier++;
            } else {
                vEdgeIsOutlier2d[i] = false;
                e->setLevel(0);  // 优化
                cnt_inlier++;
            }

            // 第三次已经执行完了，外点剔除的差不多了，第四次即最后一次迭代，不使用鲁棒核.
            if(iteration == 6) {
                // 因为剔除了错误的边，所以第三次之后优化不再使用核函数
                e->setRobustKernel(nullptr);
            }
            
        }

        if(cnt_inlier < 10) {
            // N.B.: 下一次没有足够的点优化直接退出，不然系统会崩溃
            std::cout << std::endl;
            std::cout << "(Frontend::EstimateCurrentRelativePoseForReferenceNew(...)): cnt_inlier < 10 : " << cnt_inlier << std::endl;
            std::cout << std::endl;
            break;
        }
    }

    std::cout << std::endl;
    std::cout << "(Frontend::EstimateCurrentRelativePoseForReferenceNew(...)): Outlier/Inlier in pose estimating: " 
              << cnt_outlier<< "/" << cnt_inlier<< std::endl;
    std::cout << std::endl;

    if(cnt_inlier < 10) {
        std::cout << std::endl;
        std::cout << "(Frontend::EstimateCurrentRelativePoseForReferenceNew(...)): Not enough points" << std::endl;
        std::cout << std::endl;
        return 0;
    }

    // Set pose
    // N.B.: 这里才改变当前帧的位姿
    // vertex_pose->estimate(): return the current estimate of the vertex
    auto f_est_pose_SE3Quat = vertex_pose->estimate();
    auto f_est_q = f_est_pose_SE3Quat.rotation();
    auto f_est_t = f_est_pose_SE3Quat.translation();
    auto f_est_pose = SE3(f_est_q, f_est_t);
    m_current_frame_->SetRelativePose(f_est_pose);

    return cnt_inlier;
}

// --------------------------------------------------------------------------------------------------------------
void Frontend::TrackAdditionalLandmarksFromReference() {
    // 当前帧的参考关键帧
    auto current_reference_key = m_current_frame_->reference_keyframe_ptr_.lock();
    
    int cnt_additional_landmark = 0;
    for(auto iter = m_currentForReference_matches_with_mappoint_set_.begin(); iter != m_currentForReference_matches_with_mappoint_set_.end(); iter++) {
        int currentKeypointIndex = (*iter).first;
        int ReferenceKeypointIndex = (*iter).second;


        auto current_mp = m_current_frame_->mvpMapPoints[currentKeypointIndex].lock();
        
        // N.B.: 已有对应地图点不补.
        if(current_mp) {
            continue;
        }

        // auto mp = current_reference_key->mvpfeatures[ReferenceKeypointIndex]->map_point_.lock();
        auto KF_feat = current_reference_key->mvpfeatures[ReferenceKeypointIndex];

        // N.B.: 为空时，若操作会产生段错误 !!!!!!
        if(KF_feat == nullptr) {
            continue;
        }

        // auto mp = current_reference_key->mvpfeatures[ReferenceKeypointIndex]->GetMapPoint();
        auto mp = KF_feat->GetMapPoint();
        if( mp ) {
            m_current_frame_->mvpMapPoints[currentKeypointIndex] = mp;
            cnt_additional_landmark++;
        }
    }

    int track_last_landmarks = m_current_frame_->keypoints_number;
    m_current_frame_->keypoints_number = (track_last_landmarks + cnt_additional_landmark);

    std::cout << std::endl;
    std::cout << "(Frontend::TrackAdditionalLandmarksFromReference()): track_last_landmarks = " << track_last_landmarks << std::endl;
    std::cout << "(Frontend::TrackAdditionalLandmarksFromReference()): cnt_additional_landmark = " << cnt_additional_landmark << std::endl;
    std::cout << "(Frontend::TrackAdditionalLandmarksFromReference()): m_current_frame_->keypoints_number = " << m_current_frame_->keypoints_number << std::endl;
    std::cout << std::endl;

}

// --------------------------------------------------------------------------------------------------------------
void Frontend::TrackAdditionalLandmarksFromLast3D() {
    int cnt_additional_landmark = 0;
    for(auto iter = m_currentForLast_matches_with_3d_points_unorderedmap_.begin(); iter != m_currentForLast_matches_with_3d_points_unorderedmap_.end(); iter++) {
        int currentKeypointIndex = (*iter).first;
        Vec3 pointFromLast = (*iter).second;

        auto current_mp = m_current_frame_->mvpMapPoints[currentKeypointIndex].lock();

        // N.B.: 已有对应地图点不补.
        if(current_mp) {
            continue;
        }

        auto new_map_point = MapPoint::CreateNewMappoint();
        new_map_point->SetPos(pointFromLast);
        m_current_frame_->mvpMapPoints[currentKeypointIndex] = new_map_point;
        cnt_additional_landmark++;
    }

    int track_ref_landmarks = m_current_frame_->keypoints_number;
    m_current_frame_->keypoints_number = (track_ref_landmarks + cnt_additional_landmark);

    std::cout << std::endl;
    std::cout << "(Frontend::TrackAdditionalLandmarksFromLast3D()): track_ref_landmarks = " << track_ref_landmarks << std::endl;
    std::cout << "(Frontend::TrackAdditionalLandmarksFromLast3D()): cnt_additional_landmark = " << cnt_additional_landmark << std::endl;
    std::cout << "(Frontend::TrackAdditionalLandmarksFromLast3D()): m_current_frame_->keypoints_number = " << m_current_frame_->keypoints_number << std::endl;
    std::cout << std::endl;
}

// --------------------------------------------------------------------------------------------------------------
// 参见 sg_slam/frontend.cpp Frontend::StereoInit()
void Frontend::StereoCreateNewPoints() {

    // 三角化新的地图点(已有对应地图的位置不产生新的地图点)
    std::vector<SE3> poses{ m_cam_left_->pose(), m_cam_right_->pose() };
    auto current_reference_KF = m_current_frame_->reference_keyframe_ptr_.lock();
    SE3 current_pose_Twc = (m_current_frame_->RelativePose() * current_reference_KF->Pose()).inverse();
    int cnt_trangulated_pts = 0;
    for(size_t i=0; i<m_current_frame_->mvKeysR_u.size(); i++) {
        if(m_current_frame_->mvKeysR_u[i] < 0.0) {
            continue;
        }

        // 若已经有对应地图点，就直接跳过
        auto current_mp = m_current_frame_->mvpMapPoints[i].lock();

        if(current_mp) {
            continue;
        }

        // create map point from triangulation.
        
        std::vector<Vec3> points{ 
            m_cam_left_->pixel2camera( 
                Vec2(m_current_frame_->mvKeysUn[i].pt.x,
                     m_current_frame_->mvKeysUn[i].pt.y)),
            m_cam_right_->pixel2camera( 
                Vec2(m_current_frame_->mvKeysR_u[i],
                     m_current_frame_->mvKeysR_v[i])) };
        Vec3 pworld = Vec3::Zero();  // N.B.: 这里其实是, p_camera, 即在左相机坐标系下的3D点坐标.

        // 先三角化，再判断三角化求得的深度是否为正
        // if(triangulation(poses, points, pworld) && pworld[2] > m_mappoint_camera_depth_min_) {
        // 20221010 改
        if(triangulation(poses, points, pworld) && pworld[2] > 0.0) {
            // N.B.: My add (20220827)
            // 量纲: m
            if(pworld[2] > m_test_pworld_depth_max) {
                m_test_pworld_depth_max = pworld[2];
            }
            if(pworld[2] < m_test_pworld_depth_min) {
                m_test_pworld_depth_min = pworld[2];
            }

            // N.B.: My add (20220829)
            if(pworld[0] > m_test_pworld_x_max) {
                m_test_pworld_x_max = pworld[0];
            }
            if(pworld[0] < m_test_pworld_x_min) {
                m_test_pworld_x_min = pworld[0];
            }
            if(pworld[1] > m_test_pworld_y_max) {
                m_test_pworld_y_max = pworld[1];
            }
            if(pworld[1] < m_test_pworld_y_min) {
                m_test_pworld_y_min = pworld[1];
            }
            
            // 2022092201
            if(pworld.norm() > m_mappoint_camera_depth_max_) {
                continue;
            }
            if(pworld.norm() < m_mappoint_camera_depth_min_) {
                continue;
            }

            auto new_map_point = MapPoint::CreateNewMappoint();
            // // N.B.: 与 StereoInit() 不同的地方!!!!!
            pworld = current_pose_Twc * pworld; // N.B: 这里才是真正的世界坐标系下的3D点坐标.
            new_map_point->SetPos(pworld);
            // 创键关键帧时再添加观测, 现在还没添加观测.
            m_current_frame_->mvpMapPoints[i] = new_map_point;

            m_map_->InsertMapPoint(new_map_point);
            cnt_trangulated_pts++;
        }
    }

    int track_landmarks = m_current_frame_->keypoints_number;
    m_current_frame_->keypoints_number = (track_landmarks + cnt_trangulated_pts);

    std::cout << std::endl;
    std::cout << "(Frontend::StereoCreateNewPoints()): cnt_trangulated_pts = " << cnt_trangulated_pts << std::endl;
    std::cout << "(Frontend::StereoCreateNewPoints()): track_landmarks = " << track_landmarks << std::endl;
    std::cout << "(Frontend::StereoCreateNewPoints()): m_current_frame_->keypoints_number = " << m_current_frame_->keypoints_number << std::endl;
    std::cout << "(Frontend::StereoCreateNewPoints()): m_current_frame_->mvpMapPoints = " << m_current_frame_->mvpMapPoints.size() << std::endl;
    std::cout << std::endl;

}

// --------------------------------------------------------------------------------------------------------------
void Frontend::Stereo_LR_CreateNewPoints() {
    // 三角化新的地图点(已有对应地图的位置不产生新的地图点)
    auto current_reference_KF = m_current_frame_->reference_keyframe_ptr_.lock();
    SE3 current_pose_Tcw = m_current_frame_->RelativePose() * current_reference_KF->Pose();
    int cnt_trangulated_pts = 0;

    const double minZ = m_cam_right_->baseline_;
    const double minD = 0.0;
    const double maxD = m_cam_right_->fx_baseline_ / minZ;
    const double bf = m_cam_right_->fx_baseline_;

    for(size_t i=0; i<m_current_frame_->mvKeysR_u.size(); i++) {
        if(m_current_frame_->mvKeysR_u[i] < 0.0) {
            continue;
        }

        // 若已经有对应地图点，就直接跳过
        auto current_mp = m_current_frame_->mvpMapPoints[i].lock();

        if(current_mp) {
            continue;
        }

        float uL = m_current_frame_->mvKeysUn[i].pt.x;
        float vL = m_current_frame_->mvKeysUn[i].pt.y;
        float uR = m_current_frame_->mvKeysR_u[i];
        float disparity = (uL - uR);

        if((disparity>minD) && (disparity<=maxD)) {
            double depth_camleft = bf / disparity;
            Vec3 pworld = Vec3::Zero();
            pworld = m_cam_left_->pixel2world(
                Vec2(uL, vL),
                current_pose_Tcw,
                depth_camleft
            );

            if(pworld[2] > m_test_pworld_depth_max) {
                m_test_pworld_depth_max = pworld[2];
            }
            if(pworld[2] < m_test_pworld_depth_min) {
                m_test_pworld_depth_min = pworld[2];
            }

            // N.B.: My add (20220829)
            if(pworld[0] > m_test_pworld_x_max) {
                m_test_pworld_x_max = pworld[0];
            }
            if(pworld[0] < m_test_pworld_x_min) {
                m_test_pworld_x_min = pworld[0];
            }
            if(pworld[1] > m_test_pworld_y_max) {
                m_test_pworld_y_max = pworld[1];
            }
            if(pworld[1] < m_test_pworld_y_min) {
                m_test_pworld_y_min = pworld[1];
            }

            auto new_map_point = MapPoint::CreateNewMappoint();
            m_current_frame_->mvpMapPoints[i] = new_map_point;
            m_map_->InsertMapPoint(new_map_point);
            cnt_trangulated_pts++;
        }
    }

    int track_landmarks = m_current_frame_->keypoints_number;
    m_current_frame_->keypoints_number = (track_landmarks + cnt_trangulated_pts);

    std::cout << std::endl;
    std::cout << "(Frontend::Stereo_LR_CreateNewPoints()): cnt_trangulated_pts = " << cnt_trangulated_pts << std::endl;
    std::cout << "(Frontend::Stereo_LR_CreateNewPoints()): track_landmarks = " << track_landmarks << std::endl;
    std::cout << "(Frontend::Stereo_LR_CreateNewPoints()): m_current_frame_->keypoints_number = " << m_current_frame_->keypoints_number << std::endl;
    std::cout << "(Frontend::Stereo_LR_CreateNewPoints()): m_current_frame_->mvpMapPoints = " << m_current_frame_->mvpMapPoints.size() << std::endl;
    std::cout << std::endl;
}

// --------------------------------------------------------------------------------------------------------------
void Frontend::RGBDCreateNewPoints() {
    // Vec6 se3_zero;
    // se3_zero.setZero();
    // SE3 SE3_zero = SE3::exp(se3_zero);
    auto current_reference_KF = m_current_frame_->reference_keyframe_ptr_.lock();
    SE3 current_pose_Twc = (m_current_frame_->RelativePose() * current_reference_KF->Pose()).inverse();
    int cnt_trangulated_pts = 0;
    for(size_t i=0; i<m_current_frame_->mvKeysR_u.size(); i++) {
        if(m_current_frame_->mvKeysR_u[i] < 0.0) {
            continue;
        }

        // 若已经有对应地图点，就直接跳过
        auto current_mp = m_current_frame_->mvpMapPoints[i].lock();

        if(current_mp) {
            continue;
        }

        // create map point
        Vec3 pworld = Vec3::Zero();  // N.B.: 这里其实是, p_camera, 即在左相机坐标系下的3D点坐标.
        /**
        pworld = m_cam_left_->pixel2world(
            Vec2(m_current_frame_->mvKeysUn[i].pt.x, m_current_frame_->mvKeysUn[i].pt.y),
            SE3_zero,
            static_cast<double>(m_current_frame_->mvKeysDepth[i])
        );
        **/

        pworld = m_cam_left_->pixel2camera(
            Vec2(m_current_frame_->mvKeysUn[i].pt.x, m_current_frame_->mvKeysUn[i].pt.y), 
            static_cast<double>(m_current_frame_->mvKeysDepth[i]));

        if(pworld[2] > m_test_pworld_depth_max) {
            m_test_pworld_depth_max = pworld[2];
        }
        if(pworld[2] < m_test_pworld_depth_min) {
            m_test_pworld_depth_min = pworld[2];
        }

        // N.B.: My add (20220829)
        if(pworld[0] > m_test_pworld_x_max) {
            m_test_pworld_x_max = pworld[0];
        }
        if(pworld[0] < m_test_pworld_x_min) {
            m_test_pworld_x_min = pworld[0];
        }
        if(pworld[1] > m_test_pworld_y_max) {
            m_test_pworld_y_max = pworld[1];
        }
        if(pworld[1] < m_test_pworld_y_min) {
            m_test_pworld_y_min = pworld[1];
        }
        if(pworld.norm() > m_mappoint_camera_depth_max_) {
            continue;
        }
        if(pworld.norm() < m_mappoint_camera_depth_min_) {
            continue;
        }

        auto new_map_point = MapPoint::CreateNewMappoint();
        pworld = current_pose_Twc * pworld;
        m_current_frame_->mvpMapPoints[i] = new_map_point;
        m_map_->InsertMapPoint(new_map_point);
        cnt_trangulated_pts++;
    }

    int track_landmarks = m_current_frame_->keypoints_number;
    m_current_frame_->keypoints_number = (track_landmarks + cnt_trangulated_pts);

    std::cout << std::endl;
    std::cout << "(Frontend::RGBDCreateNewPoints()): cnt_trangulated_pts = " << cnt_trangulated_pts << std::endl;
    std::cout << "(Frontend::RGBDCreateNewPoints()): track_landmarks = " << track_landmarks << std::endl;
    std::cout << "(Frontend::RGBDCreateNewPoints()): m_current_frame_->keypoints_number = " << m_current_frame_->keypoints_number << std::endl;
    std::cout << "(Frontend::RGBDCreateNewPoints()): m_current_frame_->mvpMapPoints = " << m_current_frame_->mvpMapPoints.size() << std::endl;
    std::cout << std::endl;
}

// --------------------------------------------------------------------------------------------------------------
bool Frontend::Init() {
    bool outflag = false;
    if(m_camera_type == CameraType::SETERO) {
        /**
        if(m_stereo_dataset_type == 0) {
            outflag = StereoInit();
        } else if(m_stereo_dataset_type == 1) {
            outflag = Stereo_LR_Init();
        }
        **/
        outflag = StereoInit();
    } else if(m_camera_type == CameraType::RGBD) {
        outflag = RGBDInit();
    }

    return outflag;
}

// --------------------------------------------------------------------------------------------------------------
bool Frontend::StereoInit() {
    std::vector< cv::KeyPoint > left_keypoints_v;
    std::vector< cv::KeyPoint > left_keypoints_v_un;
    torch::Tensor descL;  // desc0
    // 20221011 改
    std::tuple<std::vector< cv::KeyPoint >, torch::Tensor>  detect_out_tuple;
    if(m_current_frame_->is_use_cut_image) {
        detect_out_tuple = sg_DetectMatcher::detect(m_current_frame_->left_cut_img_, m_max_keypoints_init_);
    } else {
        detect_out_tuple = sg_DetectMatcher::detect(m_current_frame_->left_img_, m_max_keypoints_init_);
    }
    left_keypoints_v = std::get<0>(detect_out_tuple);
    descL = std::get<1>(detect_out_tuple).to(torch::kCPU);
    // 左图去畸变
    if(m_cam_left_->m_distcoef_.at<float>(0) == 0.0) {
        left_keypoints_v_un = left_keypoints_v;
    } else {
        int N_l = left_keypoints_v.size();
        cv::Mat mat(N_l, 2, CV_32F);
        for(int j=0; j<N_l; j++) {
            mat.at<float>(j, 0) = left_keypoints_v[j].pt.x;
            mat.at<float>(j, 1) = left_keypoints_v[j].pt.y;
        }

        cv::Mat mK;
        cv::eigen2cv(m_cam_left_->K(), mK);
        
        // Undistort points
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, m_cam_left_->m_distcoef_, cv::Mat(), mK);
        mat = mat.reshape(1);

        // Fill undistorted keypoint vector
        left_keypoints_v_un.resize(N_l);
        for(int j=0; j<N_l; j++) {
            cv::KeyPoint kp = left_keypoints_v[j];
            kp.pt.x = mat.at<float>(j, 0);
            kp.pt.y = mat.at<float>(j, 1);
            left_keypoints_v_un[j] = kp;
        }
    }

    std::vector< cv::KeyPoint > right_keypoints_v;
    std::vector< cv::KeyPoint > right_keypoints_v_un;
    torch::Tensor descR;  // desc1
    // detect_out_tuple = sg_DetectMatcher::detect(m_current_frame_->right_img_, m_max_keypoints_init_);
    if(m_current_frame_->is_use_cut_image) {
        detect_out_tuple = sg_DetectMatcher::detect(m_current_frame_->right_cut_img_, m_max_keypoints_init_);
    } else {
        detect_out_tuple = sg_DetectMatcher::detect(m_current_frame_->right_img_, m_max_keypoints_init_);
    }

    right_keypoints_v = std::get<0>(detect_out_tuple);
    descR = std::get<1>(detect_out_tuple).to(torch::kCPU);
    // 右图去畸变
    if(m_cam_right_->m_distcoef_.at<float>(0) == 0.0) {
        right_keypoints_v_un = right_keypoints_v;
    } else {
        int N_r = right_keypoints_v.size();
        cv::Mat mat(N_r, 2, CV_32F);
        for(int j=0; j<N_r; j++) {
            mat.at<float>(j, 0) = right_keypoints_v[j].pt.x;
            mat.at<float>(j, 1) = right_keypoints_v[j].pt.y;
        }

        cv::Mat mK;
        cv::eigen2cv(m_cam_right_->K(), mK);

        // Undistort points
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, m_cam_left_->m_distcoef_, cv::Mat(), mK);
        mat = mat.reshape(1);

        // Fill undistorted keypoint vector
        right_keypoints_v_un.resize(N_r);
        for(int j=0; j<N_r; j++) {
            cv::KeyPoint kp = right_keypoints_v[j];
            kp.pt.x = mat.at<float>(j, 0);
            kp.pt.y = mat.at<float>(j, 1);
            right_keypoints_v_un[j] = kp;
        }
    }

    auto matcher_out_dict = sg_DetectMatcher::descMatch(descL, descR, m_sg_LR_match_threshold_);
    std::vector<cv::DMatch> matchers_v;
    auto matcher0to1_tensor = matcher_out_dict["matches0"][0].to(torch::kCPU);
    for(int64_t i=0; i < matcher0to1_tensor.size(0); i++) {
        int index0to1 = matcher0to1_tensor[i].item<int>();
        if(index0to1 > -1) {
            cv::DMatch match1to2_one;
            match1to2_one.queryIdx = i;
            match1to2_one.trainIdx = index0to1;
            matchers_v.push_back(match1to2_one);
        }
    }

    // N.B.: 关键点集合没有缩小
    auto RR_matchers_v = sg_slam::sg_RANSAC(left_keypoints_v, right_keypoints_v, matchers_v, m_num_features_Threshold_init_);
    int num_correspond_features = RR_matchers_v.size();
    if(num_correspond_features < m_num_features_Threshold_init_) {
        return false;  // 少于初始化所需最少匹配点，故不进行初始化.
    }

    m_current_frame_->mvKeys = left_keypoints_v;
    m_current_frame_->mvKeysUn = left_keypoints_v_un;
    descL = descL.to(torch::kCPU);
    m_current_frame_->mDescriptors_data = convertDesTensor2Vector(descL);
    m_current_frame_->Descripters_B = descL.size(0);
    m_current_frame_->Descripters_D = descL.size(1);
    m_current_frame_->Descripters_N = descL.size(2);

    // 三角化地图点.
    // N.B.: https://blog.csdn.net/wuweiwangyao/article/details/99623866
    // N.B.: resize()有配容器的内存大小并置零的作用, reserve()没有这个功能.
    m_current_frame_->mvpMapPoints.resize(m_current_frame_->mvKeysUn.size());
    
    // My add 20220913
    m_current_frame_->mvKeysR_u.resize(m_current_frame_->mvKeysUn.size());
    std::fill(m_current_frame_->mvKeysR_u.begin(), m_current_frame_->mvKeysR_u.end(), -1.0);
    m_current_frame_->mvKeysR_v.resize(m_current_frame_->mvKeysUn.size());
    std::fill(m_current_frame_->mvKeysR_v.begin(), m_current_frame_->mvKeysR_v.end(), -1.0);

    std::vector<SE3> poses{ m_cam_left_->pose(), m_cam_right_->pose() };
    int cnt_init_landmarks = 0;
    for(size_t i=0; i<RR_matchers_v.size(); i++) {
        // create map point from triangulation
        std::vector<Vec3> points{
            m_cam_left_->pixel2camera(
                Vec2(left_keypoints_v_un[RR_matchers_v[i].queryIdx].pt.x, 
                     left_keypoints_v_un[RR_matchers_v[i].queryIdx].pt.y)),
            m_cam_right_->pixel2camera(
                Vec2(right_keypoints_v_un[RR_matchers_v[i].trainIdx].pt.x,
                     right_keypoints_v_un[RR_matchers_v[i].trainIdx].pt.y)) };
        Vec3 pworld = Vec3::Zero();

        // 先三角化，再判断三角化求得的深度是否为正
        // if(triangulation(poses, points, pworld) && pworld[2] > m_mappoint_camera_depth_min_) {
        // 20221010 改
        if(triangulation(poses, points, pworld) && pworld[2] > 0.0) {
            // N.B.: My add (20220827)
            // 量纲: m
            if(pworld[2] > m_test_pworld_depth_max) {
                m_test_pworld_depth_max = pworld[2];
            }
            if(pworld[2] < m_test_pworld_depth_min) {
                m_test_pworld_depth_min = pworld[2];
            }

            // N.B.: My add (20220829)
            if(pworld[0] > m_test_pworld_x_max) {
                m_test_pworld_x_max = pworld[0];
            }
            if(pworld[0] < m_test_pworld_x_min) {
                m_test_pworld_x_min = pworld[0];
            }
            if(pworld[1] > m_test_pworld_y_max) {
                m_test_pworld_y_max = pworld[1];
            }
            if(pworld[1] < m_test_pworld_y_min) {
                m_test_pworld_y_min = pworld[1];
            }

            // 2022092201
            if(pworld.norm() > m_mappoint_camera_depth_max_) {
                continue;
            }

            if(pworld.norm() < m_mappoint_camera_depth_min_) {
                continue;
            }

            auto new_map_point = MapPoint::CreateNewMappoint();
            new_map_point->SetPos(pworld);
            // 创键关键帧时再添加观测, 现在还没添加观测.
            m_current_frame_->mvpMapPoints[RR_matchers_v[i].queryIdx] = new_map_point;

            // My add 20220913
            m_current_frame_->mvKeysR_u[RR_matchers_v[i].queryIdx] = right_keypoints_v_un[RR_matchers_v[i].trainIdx].pt.x;
            m_current_frame_->mvKeysR_v[RR_matchers_v[i].queryIdx] = right_keypoints_v_un[RR_matchers_v[i].trainIdx].pt.y;
            m_map_->InsertMapPoint(new_map_point);
            cnt_init_landmarks++;
        }
    }

    std::cout << std::endl;
    std::cout << "(Frontend::StereoInit()): cnt_init_landmarks = " << cnt_init_landmarks << std::endl;
    std::cout << "(Frontend::StereoInit()): m_current_frame_->mvpMapPoints = " << m_current_frame_->mvpMapPoints.size() << std::endl;
    std::cout << std::endl;

    m_current_frame_->keypoints_number = cnt_init_landmarks;

    bool insertKey_success = InsertKeyFrame();

    if(insertKey_success) {
        m_status_ = FrontendStatus::TRACKING_GOOD;

        return true;
    }
    
    return false;
}

// --------------------------------------------------------------------------------------------------------------
bool Frontend::Stereo_LR_Init() {
    std::vector< cv::KeyPoint > left_keypoints_v;
    std::vector< cv::KeyPoint > left_keypoints_v_un;
    torch::Tensor descL;  // desc0
    // 20221011 改
    std::tuple<std::vector< cv::KeyPoint >, torch::Tensor>  detect_out_tuple;
    if(m_current_frame_->is_use_cut_image) {
        detect_out_tuple = sg_DetectMatcher::detect(m_current_frame_->left_cut_img_, m_max_keypoints_init_);
    } else {
        detect_out_tuple = sg_DetectMatcher::detect(m_current_frame_->left_img_, m_max_keypoints_init_);
    }
    left_keypoints_v = std::get<0>(detect_out_tuple);
    descL = std::get<1>(detect_out_tuple).to(torch::kCPU);
    // 左图去畸变
    if(m_cam_left_->m_distcoef_.at<float>(0) == 0.0) {
        left_keypoints_v_un = left_keypoints_v;
    } else {
        int N_l = left_keypoints_v.size();
        cv::Mat mat(N_l, 2, CV_32F);
        for(int j=0; j<N_l; j++) {
            mat.at<float>(j, 0) = left_keypoints_v[j].pt.x;
            mat.at<float>(j, 1) = left_keypoints_v[j].pt.y;
        }

        cv::Mat mK;
        cv::eigen2cv(m_cam_left_->K(), mK);
        
        // Undistort points
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, m_cam_left_->m_distcoef_, cv::Mat(), mK);
        mat = mat.reshape(1);

        // Fill undistorted keypoint vector
        left_keypoints_v_un.resize(N_l);
        for(int j=0; j<N_l; j++) {
            cv::KeyPoint kp = left_keypoints_v[j];
            kp.pt.x = mat.at<float>(j, 0);
            kp.pt.y = mat.at<float>(j, 1);
            left_keypoints_v_un[j] = kp;
        }
    }

    std::vector< cv::KeyPoint > right_keypoints_v;
    std::vector< cv::KeyPoint > right_keypoints_v_un;
    torch::Tensor descR;  // desc1
    // detect_out_tuple = sg_DetectMatcher::detect(m_current_frame_->right_img_, m_max_keypoints_init_);
    if(m_current_frame_->is_use_cut_image) {
        detect_out_tuple = sg_DetectMatcher::detect(m_current_frame_->right_cut_img_, m_max_keypoints_init_);
    } else {
        detect_out_tuple = sg_DetectMatcher::detect(m_current_frame_->right_img_, m_max_keypoints_init_);
    }

    right_keypoints_v = std::get<0>(detect_out_tuple);
    descR = std::get<1>(detect_out_tuple).to(torch::kCPU);
    // 右图去畸变
    if(m_cam_right_->m_distcoef_.at<float>(0) == 0.0) {
        right_keypoints_v_un = right_keypoints_v;
    } else {
        int N_r = right_keypoints_v.size();
        cv::Mat mat(N_r, 2, CV_32F);
        for(int j=0; j<N_r; j++) {
            mat.at<float>(j, 0) = right_keypoints_v[j].pt.x;
            mat.at<float>(j, 1) = right_keypoints_v[j].pt.y;
        }

        cv::Mat mK;
        cv::eigen2cv(m_cam_right_->K(), mK);

        // Undistort points
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, m_cam_left_->m_distcoef_, cv::Mat(), mK);
        mat = mat.reshape(1);

        // Fill undistorted keypoint vector
        right_keypoints_v_un.resize(N_r);
        for(int j=0; j<N_r; j++) {
            cv::KeyPoint kp = right_keypoints_v[j];
            kp.pt.x = mat.at<float>(j, 0);
            kp.pt.y = mat.at<float>(j, 1);
            right_keypoints_v_un[j] = kp;
        }
    }

    auto matcher_out_dict = sg_DetectMatcher::descMatch(descL, descR, m_sg_LR_match_threshold_);
    std::vector<cv::DMatch> matchers_v;
    auto matcher0to1_tensor = matcher_out_dict["matches0"][0].to(torch::kCPU);
    for(int64_t i=0; i < matcher0to1_tensor.size(0); i++) {
        int index0to1 = matcher0to1_tensor[i].item<int>();
        if(index0to1 > -1) {
            cv::DMatch match1to2_one;
            match1to2_one.queryIdx = i;
            match1to2_one.trainIdx = index0to1;
            matchers_v.push_back(match1to2_one);
        }
    }

    // N.B.: 关键点集合没有缩小
    auto RR_matchers_v = sg_slam::sg_RANSAC(left_keypoints_v, right_keypoints_v, matchers_v, m_num_features_Threshold_init_);
    int num_correspond_features = RR_matchers_v.size();
    if(num_correspond_features < m_num_features_Threshold_init_) {
        return false;  // 少于初始化所需最少匹配点，故不进行初始化.
    }

    m_current_frame_->mvKeys = left_keypoints_v;
    m_current_frame_->mvKeysUn = left_keypoints_v_un;
    descL = descL.to(torch::kCPU);
    m_current_frame_->mDescriptors_data = convertDesTensor2Vector(descL);
    m_current_frame_->Descripters_B = descL.size(0);
    m_current_frame_->Descripters_D = descL.size(1);
    m_current_frame_->Descripters_N = descL.size(2);

    // 三角化地图点.
    // N.B.: https://blog.csdn.net/wuweiwangyao/article/details/99623866
    // N.B.: resize()有配容器的内存大小并置零的作用, reserve()没有这个功能.
    m_current_frame_->mvpMapPoints.resize(m_current_frame_->mvKeysUn.size());
    
    // My add 20220913
    m_current_frame_->mvKeysR_u.resize(m_current_frame_->mvKeysUn.size());
    std::fill(m_current_frame_->mvKeysR_u.begin(), m_current_frame_->mvKeysR_u.end(), -1.0);
    m_current_frame_->mvKeysR_v.resize(m_current_frame_->mvKeysUn.size());
    std::fill(m_current_frame_->mvKeysR_v.begin(), m_current_frame_->mvKeysR_v.end(), -1.0);
    
    const double minZ = m_cam_right_->baseline_;
    const double minD = 0.0;
    const double maxD = m_cam_right_->fx_baseline_ / minZ;
    const double bf = m_cam_right_->fx_baseline_;

    std::cout << std::endl;
    std::cout << "(Frontend::Stereo_LR_Init()): minZ = " << minZ << std::endl;
    std::cout << "(Frontend::Stereo_LR_Init()): minD = " << minD << std::endl;
    std::cout << "(Frontend::Stereo_LR_Init()): maxD = " << maxD << std::endl;
    std::cout << std::endl;

    int cnt_init_landmarks = 0;
    Vec6 se3_zero;
    se3_zero.setZero();
    SE3 SE3_zero = SE3::exp(se3_zero);
    for(size_t i=0; i<RR_matchers_v.size(); i++) {
        float uL = left_keypoints_v_un[RR_matchers_v[i].queryIdx].pt.x;
        float vL = left_keypoints_v_un[RR_matchers_v[i].queryIdx].pt.y;
        float uR = right_keypoints_v_un[RR_matchers_v[i].trainIdx].pt.x;
        // float vR = right_keypoints_v_un[RR_matchers_v[i].trainIdx].pt.y;
        float disparity = (uL - uR);

        if((disparity>minD) && (disparity<=maxD)) {
            double depth_camleft = bf / disparity;
            Vec3 pworld = Vec3::Zero();
            pworld = m_cam_left_->pixel2world(
                Vec2(uL, vL),
                SE3_zero,
                depth_camleft
            );

            if(pworld[2] > m_test_pworld_depth_max) {
                m_test_pworld_depth_max = pworld[2];
            }
            if(pworld[2] < m_test_pworld_depth_min) {
                m_test_pworld_depth_min = pworld[2];
            }

            // N.B.: My add (20220829)
            if(pworld[0] > m_test_pworld_x_max) {
                m_test_pworld_x_max = pworld[0];
            }
            if(pworld[0] < m_test_pworld_x_min) {
                m_test_pworld_x_min = pworld[0];
            }
            if(pworld[1] > m_test_pworld_y_max) {
                m_test_pworld_y_max = pworld[1];
            }
            if(pworld[1] < m_test_pworld_y_min) {
                m_test_pworld_y_min = pworld[1];
            }

            auto new_map_point = MapPoint::CreateNewMappoint();
            new_map_point->SetPos(pworld);
            // 创键关键帧时再添加观测, 现在还没添加观测.
            m_current_frame_->mvpMapPoints[RR_matchers_v[i].queryIdx] = new_map_point;

            m_current_frame_->mvKeysR_u[RR_matchers_v[i].queryIdx] = right_keypoints_v_un[RR_matchers_v[i].trainIdx].pt.x;
            m_current_frame_->mvKeysR_v[RR_matchers_v[i].queryIdx] = right_keypoints_v_un[RR_matchers_v[i].trainIdx].pt.y;
            m_map_->InsertMapPoint(new_map_point);
            cnt_init_landmarks++;
        }
    }

    std::cout << std::endl;
    std::cout << "(Frontend::Stereo_LR_Init()): cnt_init_landmarks = " << cnt_init_landmarks << std::endl;
    std::cout << "(Frontend::Stereo_LR_Init()): m_current_frame_->mvpMapPoints = " << m_current_frame_->mvpMapPoints.size() << std::endl;
    std::cout << std::endl;

    m_current_frame_->keypoints_number = cnt_init_landmarks;

    bool insertKey_success = InsertKeyFrame();

    if(insertKey_success) {
        m_status_ = FrontendStatus::TRACKING_GOOD;

        return true;
    }

    return false;
}

// --------------------------------------------------------------------------------------------------------------
bool Frontend::RGBDInit() {
    std::vector< cv::KeyPoint > left_keypoints_v;
    std::vector< cv::KeyPoint > left_keypoints_v_un;
    torch::Tensor descL;  // desc0

    std::tuple<std::vector< cv::KeyPoint >, torch::Tensor>  detect_out_tuple;

    if(m_current_frame_->is_use_cut_image) {
        detect_out_tuple = sg_DetectMatcher::detect(m_current_frame_->left_cut_img_, m_max_keypoints_init_);
    } else {
        detect_out_tuple = sg_DetectMatcher::detect(m_current_frame_->left_img_, m_max_keypoints_init_);
    }
    left_keypoints_v = std::get<0>(detect_out_tuple);
    descL = std::get<1>(detect_out_tuple).to(torch::kCPU);
    // 左图去畸变
    if(m_cam_left_->m_distcoef_.at<float>(0) == 0.0) {
        left_keypoints_v_un = left_keypoints_v;
    } else {
        int N_l = left_keypoints_v.size();
        cv::Mat mat(N_l, 2, CV_32F);
        for(int j=0; j<N_l; j++) {
            mat.at<float>(j, 0) = left_keypoints_v[j].pt.x;
            mat.at<float>(j, 1) = left_keypoints_v[j].pt.y;
        }

        cv::Mat mK;
        cv::eigen2cv(m_cam_left_->K(), mK);
        
        // Undistort points
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, m_cam_left_->m_distcoef_, cv::Mat(), mK);
        mat = mat.reshape(1);

        // Fill undistorted keypoint vector
        left_keypoints_v_un.resize(N_l);
        for(int j=0; j<N_l; j++) {
            cv::KeyPoint kp = left_keypoints_v[j];
            kp.pt.x = mat.at<float>(j, 0);
            kp.pt.y = mat.at<float>(j, 1);
            left_keypoints_v_un[j] = kp;
        }
    }

    m_current_frame_->mvKeys = left_keypoints_v;
    m_current_frame_->mvKeysUn = left_keypoints_v_un;
    descL = descL.to(torch::kCPU);
    m_current_frame_->mDescriptors_data = convertDesTensor2Vector(descL);
    m_current_frame_->Descripters_B = descL.size(0);
    m_current_frame_->Descripters_D = descL.size(1);
    m_current_frame_->Descripters_N = descL.size(2);

    // 创建地图点
    m_current_frame_->mvpMapPoints.resize(m_current_frame_->mvKeysUn.size());
    m_current_frame_->mvKeysR_u.resize(m_current_frame_->mvKeysUn.size());
    std::fill(m_current_frame_->mvKeysR_u.begin(), m_current_frame_->mvKeysR_u.end(), -1.0);
    m_current_frame_->mvKeysDepth.resize(m_current_frame_->mvKeysUn.size());
    std::fill(m_current_frame_->mvKeysDepth.begin(), m_current_frame_->mvKeysDepth.end(), -1.0);
    int cnt_init_landmarks = 0;
    double bf = m_cam_left_->fx_baseline_;
    Vec6 se3_zero;
    se3_zero.setZero();
    SE3 SE3_zero = SE3::exp(se3_zero);

    for(size_t i=0; i<m_current_frame_->mvKeysUn.size(); i++) {
        float d = m_current_frame_->findDepth(m_current_frame_->mvKeys[i]);
        if(d > 0.0) {
            Vec3 pworld = Vec3::Zero();
            auto kpU = m_current_frame_->mvKeysUn[i];
            m_current_frame_->mvKeysDepth[i] = d;
            m_current_frame_->mvKeysR_u[i] = kpU.pt.x - static_cast<float>(bf)/d;
            pworld = m_cam_left_->pixel2world(
                Vec2(m_current_frame_->mvKeysUn[i].pt.x, m_current_frame_->mvKeysUn[i].pt.y),
                SE3_zero,
                static_cast<double>(d)
            );

            if(pworld[2] > m_test_pworld_depth_max) {
                m_test_pworld_depth_max = pworld[2];
            }
            if(pworld[2] < m_test_pworld_depth_min) {
                m_test_pworld_depth_min = pworld[2];
            }

            // N.B.: My add (20220829)
            if(pworld[0] > m_test_pworld_x_max) {
                m_test_pworld_x_max = pworld[0];
            }
            if(pworld[0] < m_test_pworld_x_min) {
                m_test_pworld_x_min = pworld[0];
            }
            if(pworld[1] > m_test_pworld_y_max) {
                m_test_pworld_y_max = pworld[1];
            }
            if(pworld[1] < m_test_pworld_y_min) {
                m_test_pworld_y_min = pworld[1];
            }

            if(pworld.norm() > m_mappoint_camera_depth_max_) {
                continue;
            }

            if(pworld.norm() < m_mappoint_camera_depth_min_) {
                continue;
            }

            auto new_map_point = MapPoint::CreateNewMappoint();
            new_map_point->SetPos(pworld);
            m_current_frame_->mvpMapPoints[i] = new_map_point;

            m_map_->InsertMapPoint(new_map_point);
            cnt_init_landmarks++;
        } else {
            m_current_frame_->mvKeysDepth[i] = -1.0;
            m_current_frame_->mvKeysR_u[i] = -1.0;
        }
    }

    std::cout << std::endl;
    std::cout << "(Frontend::RGBDInit()): cnt_init_landmarks = " << cnt_init_landmarks << std::endl;
    std::cout << "(Frontend::RGBDInit()): m_current_frame_->mvpMapPoints = " << m_current_frame_->mvpMapPoints.size() << std::endl;
    std::cout << std::endl;

    if(cnt_init_landmarks < m_num_features_Threshold_init_) {
        return false;  // 少于初始化所需最少地图点，故不进行初始化.
    }

    m_current_frame_->keypoints_number = cnt_init_landmarks;

    bool insertKey_success = InsertKeyFrame();

    if(insertKey_success) {
        m_status_ = FrontendStatus::TRACKING_GOOD;

        return true;
    }
    
    return false;
}

// --------------------------------------------------------------------------------------------------------------
bool Frontend::InsertKeyFrame() {
    Vec6 se3_zero;
    se3_zero.setZero();

    m_current_frame_->SetKeyFrame();

    KeyFrame::Ptr newKF = KeyFrame::CreateKeyFrame();
    newKF->time_stamp_ = m_current_frame_->time_stamp_;
    newKF->src_frame_id_ = m_current_frame_->id_;

    if(m_status_ == FrontendStatus::INITIALIZING) {
        newKF->SetPose(SE3::exp(se3_zero));  // N.B.: 设置参考帧位姿.
    } else {
        newKF->SetPose(m_current_frame_->RelativePose() * m_current_frame_->reference_keyframe_ptr_.lock()->Pose());
        newKF->mpLastKF = m_current_frame_->reference_keyframe_ptr_;  // N.B.: 为了回环修正.
        newKF->m_relative_pose_to_lastKF = m_current_frame_->RelativePose();  // N.B.: 为了回环修正.
        auto lastKF = m_current_frame_->reference_keyframe_ptr_.lock();  // N.B.: 引起循环引用，待回环调试完成再处理!!!!!
        lastKF->mpNextKF = newKF;  // N.B.: 引起循环引用，待回环调试完成再处理!!!!!
        lastKF->m_relative_pose_to_nextKF = m_current_frame_->RelativePose().inverse();  // N.B.: 引起循环引用，待回环调试完成再处理!!!!!

        if(m_stereo_dataset_type == 1) {
            m_local_keyframes_deque.push_front(lastKF);
            if(m_local_keyframes_deque.size() > m_local_keyframes_deque_length) {
                m_local_keyframes_deque.pop_back();
            }
        }
    }

    m_current_frame_->reference_keyframe_ptr_ = newKF;
    m_current_frame_->SetRelativePose(SE3::exp(se3_zero));

    newKF->left_img_ = m_current_frame_->left_img_.clone();
    // 20221011 改
    newKF->left_cut_img_ = m_current_frame_->left_cut_img_.clone();
    newKF->is_use_cut_image = m_current_frame_->is_use_cut_image;

    newKF->mvKeys = m_current_frame_->mvKeys;
    newKF->mvKeysUn = m_current_frame_->mvKeysUn;
    newKF->Descripters_B = m_current_frame_->Descripters_B;
    newKF->Descripters_D = m_current_frame_->Descripters_D;
    newKF->Descripters_N = m_current_frame_->Descripters_N;
    newKF->mDescriptors_data = m_current_frame_->mDescriptors_data;

    int keypoints_N = newKF->mvKeys.size();
    newKF->mvpfeatures.resize(keypoints_N);
    // My add 20220913
    newKF->mvKeysR_u.resize(keypoints_N);
    int cnt_landmarks = 0;
    int cnt_temp = 0;
    for(int i=0; i<keypoints_N; i++) {
        auto mp = m_current_frame_->mvpMapPoints[i].lock();
        if(mp) {
            auto feat = Feature::Ptr(new Feature(newKF, newKF->mvKeysUn[i]));
            // feat->map_point_ = mp;
            
            // My add 20220913
            feat->kpR_u_ = m_current_frame_->mvKeysR_u[i];

            if(feat->kpR_u_ > -1.0) {
                cnt_temp++;
            }

            feat->SetMapPoint(mp);
            newKF->mvpfeatures[i] = feat;

            // My add 20220913
            newKF->mvKeysR_u[i] = m_current_frame_->mvKeysR_u[i];

            // 20221018 改
            if(mp->reference_KeyFrame_.lock() == nullptr) {
                mp->reference_KeyFrame_ = newKF;  // N.B.: 漏加的 !!!!!!
            }
            
            mp->AddObservation(newKF->mvpfeatures[i]);
            mp->AddActiveObservation(newKF->mvpfeatures[i]);
            cnt_landmarks++;
        } else {
            // newKF->mvpfeatures[i].reset();
            newKF->mvpfeatures[i] == nullptr;

            // My add 20220913
            newKF->mvKeysR_u[i] = -1.0;
        }
    }
    newKF->keypoints_number = cnt_landmarks;

    m_map_->InsertKeyFrame(newKF);  // 将新关键帧插入地图

    if(m_backend_) {
        m_backend_->InsertProcessKeyframe();  // 将新关键帧插入后端缓存队列
    }

    if(m_loopclosing_) {
        m_loopclosing_->InsertProcessKeyframe(); // 将新关键帧插入回环缓存队列
    }

    std::cout << std::endl;
    std::cout << "(Frontend::InsertKeyFrame()): newKF->keypoints_number = " <<  newKF->keypoints_number << std::endl;
    std::cout << "(Frontend::InsertKeyFrame()): cnt_temp = " <<  cnt_temp << std::endl;
    std::cout << "(Frontend::InsertKeyFrame()): newKF->mvpfeatures = " << newKF->mvpfeatures.size() << std::endl;
    std::cout << "(Frontend::InsertKeyFrame()): newKF->id_ = " << newKF->id_ << std::endl;
    std::cout << "(Frontend::InsertKeyFrame()): newKF->src_frame_id_ = " << newKF->src_frame_id_ << std::endl;
    std::cout << std::endl;

    return true;
}

// --------------------------------------------------------------------------------------------------------------
// 目前仅为信息提示，之后可添加相关重置操作
bool Frontend::Reset() {
    
    std::cout << "**************************************************" << std::endl;
    std::cout << "(Frontend::Reset()): Reset is not implemented. " << std::endl;
    std::cout << "**************************************************" << std::endl;

    return true;
}

}  // namespace sg_slam
