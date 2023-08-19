//
//  Created by Lucifer on 2022/8/13.
//

#include "sg_slam/system.h"

#include <glog/logging.h>

#include "sg_slam/config.h"
#include "sg_slam/sg_detectMatcher.h"
#include "sg_slam/frame.h"
#include "sg_slam/keyframe.h"
#include "sg_slam/common_include.h"
#include "sg_slam/TUM_dataset.h"  // 20221013
#include "sg_slam/EuRoC_dataset.h"  // 20221028

#include <chrono>  // N.B.: 运行时间测试
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

// #include <map>

#include <Eigen/Geometry>

#include <unistd.h>  // usleep

namespace sg_slam {

// --------------------------------------------------------------------------------------------------------------
System::System(std::string &config_path) 
    : m_config_file_path_(config_path) {
}

// --------------------------------------------------------------------------------------------------------------
bool System::Init() {
    // read from config file
    if(Config::SetParameterFile(m_config_file_path_) == false) {
        return false;
    }

    // 20221009 加
    int camera_type = Config::Get<int>(static_cast<std::string>("camera.type"));
    if( camera_type == 0 ) {
        m_camera_type = CameraType::MONO;
    } else if( camera_type == 1 ) {
        m_camera_type = CameraType::SETERO;
    } else if( camera_type == 2 ) {
        m_camera_type = CameraType::RGBD;
    }

    // 20221012 加
    int use_loop_flag = Config::Get<int>(static_cast<std::string>("LoopClosing_used_flag"));
    if( use_loop_flag == 0 ) {
        m_is_use_loopclosing_flag = false;
    } else {
        m_is_use_loopclosing_flag = true;
    }

    std::cout << std::endl;
    std::cout << "(System::Init()): m_camera_type = " << static_cast<int>(m_camera_type) << std::endl;
    std::cout << "(System::Init()): m_is_use_loopclosing_flag = " << m_is_use_loopclosing_flag << std::endl;
    std::cout << std::endl;

    // 根据配置文件写
    // 20221009 改
    // 20221013 改
    if(m_camera_type == CameraType::SETERO) {
        if(m_stereo_dataset_type == 0) {
            m_dataset_ = KITTI_Dataset::Ptr(new KITTI_Dataset(Config::Get<std::string>(static_cast<std::string>("dataset_dir"))));
        } else if(m_stereo_dataset_type == 1) {
            m_dataset_ = EuRoC_dataset::Ptr(new EuRoC_dataset(Config::Get<std::string>(static_cast<std::string>("dataset_dir"))));
        }
        
    } else if(m_camera_type == CameraType::RGBD) {
        m_dataset_ = TUM_dataset::Ptr(new TUM_dataset(Config::Get<std::string>(static_cast<std::string>("dataset_dir"))));
    }

    CHECK_EQ(m_dataset_->Init(1.0), true);

    // N.B.: 自己加的对提取器和匹配器的初始化.
    auto superpoint_weight = Config::Get<std::string>(static_cast<std::string>("sg_superpoint"));
    auto superglue_weight = Config::Get<std::string>(static_cast<std::string>("supergule"));
    auto sp_gpu_id = Config::Get<int>(static_cast<std::string>("sp_gpu_id"));
    auto sg_gpu_id = Config::Get<int>(static_cast<std::string>("sg_gpu_id"));
    sg_DetectMatcher::set_DetectorAndMatcher(superpoint_weight, superglue_weight, sp_gpu_id, sg_gpu_id);

    // create components 
    m_map_ = Map::Ptr(new Map);
    m_frontend_ = Frontend::Ptr(new Frontend);
    m_backend_ = Backend::Ptr(new Backend);

    // 20221030 add
    m_frontend_->m_key_max_id_interval = m_key_max_id_interval;
    // 20221101 add
    m_frontend_->m_stereo_dataset_type = m_stereo_dataset_type;

    // 20221104 add
    m_frontend_->m_local_keyframes_deque_length = m_local_keyframes_deque_length;

    if(m_is_use_loopclosing_flag) {
        m_loopclosing_ = LoopClosing::Ptr(new LoopClosing);
    }

    m_viewer_ = Viewer::Ptr(new Viewer);

    
    // link components 
    m_frontend_->SetMap( m_map_ );
    m_frontend_->SetBackend( m_backend_ );
    m_frontend_->SetLoopClosing( m_loopclosing_ );
    m_frontend_->SetViewer( m_viewer_ );

    // 20221009 改
    // 20221013 改
    if(m_camera_type == CameraType::SETERO) {
        m_frontend_->SetCameras(m_dataset_->getCamera(0), m_dataset_->getCamera(1));
    } else if(m_camera_type == CameraType::RGBD) {
        m_frontend_->SetCameras(m_dataset_->getCamera(), m_dataset_->getCamera());
    }
    
    m_frontend_->SetStatus( FrontendStatus::INITIALIZING );  // My add.

    // 20221009 改
    // 20221013 改
    if(m_camera_type == CameraType::SETERO) {
        m_backend_->SetCameras(m_dataset_->getCamera(0), m_dataset_->getCamera(1));
    } else if(m_camera_type == CameraType::RGBD) {
        m_backend_->SetCameras(m_dataset_->getCamera(), m_dataset_->getCamera());
    }
    
    m_backend_->SetMap( m_map_ );

    // 20221009 改
    // 20221012 改
    if(m_is_use_loopclosing_flag) {
        if(m_camera_type == CameraType::SETERO) {
            m_loopclosing_->SetCameras(m_dataset_->getCamera(0), m_dataset_->getCamera(1));
        } else if(m_camera_type == CameraType::RGBD) {
            m_loopclosing_->SetCameras(m_dataset_->getCamera(), m_dataset_->getCamera());
        }
        m_loopclosing_->SetMap( m_map_ );
        m_loopclosing_->SetBackend( m_backend_ );
    }

    m_viewer_->SetMap( m_map_ );

    return true;
}

// --------------------------------------------------------------------------------------------------------------
void System::Run() {
    while(1) {
        std::cout << "(System::Run()): SLAM is running ... " << std::endl;

        // 停止条件
        if(Step() == false) {
            break;
        }
    }

    std::cout << std::endl;
    std::cout << "(System::Run()): m_frontend_->m_test_pworld_depth_max: " << m_frontend_->m_test_pworld_depth_max << std::endl;
    std::cout << "(System::Run()): m_frontend_->m_test_pworld_depth_min: " << m_frontend_->m_test_pworld_depth_min << std::endl;
    std::cout << "(System::Run()): m_frontend_->m_test_pworld_x_max: " << m_frontend_->m_test_pworld_x_max << std::endl;
    std::cout << "(System::Run()): m_frontend_->m_test_pworld_x_min: " << m_frontend_->m_test_pworld_x_min << std::endl;
    std::cout << "(System::Run()): m_frontend_->m_test_pworld_y_max: " << m_frontend_->m_test_pworld_y_max << std::endl;
    std::cout << "(System::Run()): m_frontend_->m_test_pworld_y_min: " << m_frontend_->m_test_pworld_y_min << std::endl;
    std::cout << std::endl;

    std::cout << std::endl;
    std::cout << "(System::Run()): m_loopclosing_->m_detect_loop_number : " << m_loopclosing_->m_detect_loop_number << std::endl;
    std::cout << "(System::Run()): m_loopclosing_->m_match_loop_number: " << m_loopclosing_->m_match_loop_number << std::endl;
    std::cout << "(System::Run()): m_loopclosing_->m_correct_number: " << m_loopclosing_->m_correct_number << std::endl;
    std::cout << std::endl;

    /**
    char c = 'A';
    std::cout << " Please enter q to stop: ";
    while(std::cin>>c && c!='q') {

    }
    **/

    std::cout << std::endl;
    std::cout << "(System::Run()): SLAM exit." << std::endl;
    std::cout << std::endl;
}

// --------------------------------------------------------------------------------------------------------------
bool System::Step() {
    Frame::Ptr new_frame = m_dataset_->NextFrame();
    if( new_frame == nullptr ) {
        return false;
    }

    /**
    // test 
    if( new_frame->id_ == 10 ) {
        std::cout << std::endl;
        std::cout << "(System::Step()): test " << std::endl;
        std::cout << std::endl;
        return false;
    }
    **/

    auto t1 = std::chrono::steady_clock::now();
    bool success = m_frontend_->AddFrame( new_frame );
    auto t2 = std::chrono::steady_clock::now();
    auto time_used = std::chrono::duration_cast< std::chrono::duration<double> >(t2 - t1);
    std::cout << std::endl;
    std::cout << "SLAM cost time: " << time_used.count() << " second. " << std::endl;
    std::cout << std::endl;

    return success;
    // return true;
}

// --------------------------------------------------------------------------------------------------------------
void System::Stop() {

    // 20221012 改
    if(m_is_use_loopclosing_flag) {
        m_loopclosing_->Stop();
        while(!m_loopclosing_->m_is_stop_flag.load()) {

        }
    }
    usleep(1000);

    m_backend_->Stop();

    while(!m_backend_->m_is_stop_flag.load()) {

    }

    usleep(1000);

    m_viewer_->Close();
    while(!m_viewer_->m_is_stop_flag.load()) {
        
    }
    usleep(1000);
    
}

// --------------------------------------------------------------------------------------------------------------
// 排序功能函数
bool unorder_key_comp(const std::pair<unsigned long, KeyFrame::Ptr> &a, const std::pair<unsigned long, KeyFrame::Ptr> &b) {
    return a.first < b.first;
}

bool unorder_frame_comp(const std::pair<unsigned long, FrameResult::Ptr> &a, const std::pair<unsigned long, FrameResult::Ptr> &b) {
    return a.first < b.first;
}

bool unorder_landmark_comp(const std::pair<unsigned long, MapPoint::Ptr> &a, const std::pair<unsigned long, MapPoint::Ptr> &b) {
    return a.first < b.first;
}

// --------------------------------------------------------------------------------------------------------------
void System::SaveKeyframeTrajectory(std::string &save_file) {
    std::ofstream outfile;
    outfile.open(save_file, std::ios_base::out | std::ios_base::trunc);
    outfile << std::fixed;
    // std::map<unsigned long, KeyFrame::Ptr> poses_map;
    std::vector< std::pair<unsigned long, KeyFrame::Ptr> > poses_vector;

    auto kfs = m_map_->GetAllKeyFrames();

    // 20221012 改
    SE3 Two = kfs[0]->Pose().inverse(); // 第一帧位姿 (为了将整个结果的参考系变换到第一帧坐标系)

    for(auto &kf : kfs) {
        unsigned long keyframe_id = kf.first;
        KeyFrame::Ptr keyframe = kf.second;
        // poses_map.insert( std::make_pair(keyframe_id, keyframe) );
        poses_vector.push_back( std::make_pair(keyframe_id, keyframe) );
    }

    std::sort(poses_vector.begin(), poses_vector.end(), unorder_key_comp);

    // for(auto &kf : poses_map) {
    for(auto &kf : poses_vector) {
        unsigned long keyframe_id = kf.first;
        KeyFrame::Ptr keyframe = kf.second;
        double timestamp = keyframe->time_stamp_;
        // SE3 frame_pose = keyframe->Pose().inverse();  // Twc
        // 20221012改
        SE3 frame_pose = (keyframe->Pose() * Two).inverse();  // Tcw->Tco -> Toc
        Vec3 pose_t = frame_pose.translation();
        Mat33 pose_R = frame_pose.rotationMatrix();
        Eigen::Quaterniond pose_q = Eigen::Quaterniond(pose_R);

        // https://blog.csdn.net/danshiming/article/details/119202955
        outfile << std::setprecision(6) << keyframe_id << " "<< keyframe->src_frame_id_ << " " << timestamp << " " 
                << pose_t.transpose() << " " << pose_q.coeffs().transpose() << std::endl;
    }

    outfile.close();

}

// --------------------------------------------------------------------------------------------------------------
void System::SaveKeyframeTrajectoryTUM(std::string &save_file) {
    std::ofstream outfile;
    outfile.open(save_file, std::ios_base::out | std::ios_base::trunc);
    outfile << std::fixed;
    // std::map<unsigned long, KeyFrame::Ptr> poses_map;
    std::vector< std::pair<unsigned long, KeyFrame::Ptr> > poses_vector;

    auto kfs = m_map_->GetAllKeyFrames();

    // 20221012 改
    SE3 Two = kfs[0]->Pose().inverse(); // 第一帧位姿 (为了将整个结果的参考系变换到第一帧坐标系)

    for(auto &kf : kfs) {
        unsigned long keyframe_id = kf.first;
        KeyFrame::Ptr keyframe = kf.second;
        // poses_map.insert( std::make_pair(keyframe_id, keyframe) );
        poses_vector.push_back( std::make_pair(keyframe_id, keyframe) );
    }

    std::sort(poses_vector.begin(), poses_vector.end(), unorder_key_comp);

    // for(auto &kf : poses_map) {
    for(auto &kf : poses_vector) {
        // unsigned long keyframe_id = kf.first;
        KeyFrame::Ptr keyframe = kf.second;
        double timestamp = keyframe->time_stamp_;
        // SE3 frame_pose = keyframe->Pose().inverse();  // Twc
        // 20221012改
        SE3 frame_pose = (keyframe->Pose() * Two).inverse();  // Tcw->Tco -> Toc
        Vec3 pose_t = frame_pose.translation();
        Mat33 pose_R = frame_pose.rotationMatrix();
        Eigen::Quaterniond pose_q = Eigen::Quaterniond(pose_R);

        // https://blog.csdn.net/danshiming/article/details/119202955
        outfile << std::setprecision(6) << timestamp << " " 
                << pose_t.transpose() << " " << pose_q.coeffs().transpose() << std::endl;
    }

    outfile.close();
}

// --------------------------------------------------------------------------------------------------------------
void System::SaveLoopEdges(std::string &save_file) {
    std::ofstream outfile;
    outfile.open(save_file, std::ios_base::out | std::ios_base::trunc);
    outfile << std::fixed;
    // std::map<unsigned long, KeyFrame::Ptr> poses_map;
    std::vector< std::pair<unsigned long, KeyFrame::Ptr> > poses_vector;

    auto kfs = m_map_->GetAllKeyFrames();

    // 20221012 改
    SE3 Two = kfs[0]->Pose().inverse(); // 第一帧位姿 (为了将整个结果的参考系变换到第一帧坐标系)

    for(auto &kf : kfs) {
        unsigned long keyframe_id = kf.first;
        KeyFrame::Ptr keyframe = kf.second;
        // poses_map.insert( std::make_pair(keyframe_id, keyframe) );
        poses_vector.push_back( std::make_pair(keyframe_id, keyframe) );
    }

    std::sort(poses_vector.begin(), poses_vector.end(), unorder_key_comp);

    // for(auto &kf : poses_map) {
    for(auto &kf : poses_vector) {
        unsigned long keyframe_id = kf.first;
        KeyFrame::Ptr keyframe = kf.second;
        auto loopKF = keyframe->mpLoopKF.lock();
        if(loopKF) {
            double timestamp_current = keyframe->time_stamp_;
            // SE3 frame_pose_current = keyframe->Pose().inverse();
            SE3 frame_pose_current = (keyframe->Pose() * Two).inverse();
            Vec3 pose_t_current = frame_pose_current.translation();
            Mat33 pose_R_current = frame_pose_current.rotationMatrix();
            Eigen::Quaterniond pose_q_current = Eigen::Quaterniond(pose_R_current);

            outfile << std::setprecision(6) << keyframe_id << " " << keyframe->src_frame_id_ << " " << timestamp_current << " "
                    << pose_t_current.transpose() << " " << pose_q_current.coeffs().transpose() << std::endl;
            
            double timestamp_loop = loopKF->time_stamp_;
            // SE3 frame_pose_loop = loopKF->Pose().inverse();
            SE3 frame_pose_loop = (loopKF->Pose() * Two).inverse();
            Vec3 pose_t_loop = frame_pose_loop.translation();
            Mat33 pose_R_loop = frame_pose_loop.rotationMatrix();
            Eigen::Quaterniond pose_q_loop = Eigen::Quaterniond(pose_R_loop);

            outfile << std::setprecision(6) << loopKF->id_ << " " << loopKF->src_frame_id_ << " " << timestamp_loop << " "
                    << pose_t_loop.transpose() << " " << pose_q_loop.coeffs().transpose() << std::endl;
        }
    }

    outfile.close();
}

// --------------------------------------------------------------------------------------------------------------
void System::SaveFrameTrajectoryTUM(std::string &save_file) {
    std::ofstream outfile;
    outfile.open(save_file, std::ios_base::out | std::ios_base::trunc);
    outfile << std::fixed;
    // std::map<unsigned long, FrameResult::Ptr> poses_map;
    std::vector< std::pair<unsigned long, FrameResult::Ptr> > poses_vector;

    // 20221012 改
    auto kfs = m_map_->GetAllKeyFrames();
    SE3 Two = kfs[0]->Pose().inverse(); // 第一帧位姿 (为了将整个结果的参考系变换到第一帧坐标系)
    
    auto frame_results = m_map_->GetAllFrameResults();
    for(auto fr : frame_results) {
        unsigned long frame_id = fr.first;
        FrameResult::Ptr frameresult = fr.second;
        // poses_map.insert( std::make_pair(frame_id, frameresult) );
        poses_vector.push_back( std::make_pair(frame_id, frameresult) );
    }

    std::sort(poses_vector.begin(), poses_vector.end(), unorder_frame_comp);

    // for(auto &fr : poses_map) {
    for(auto &fr : poses_vector) {
        // unsigned long frame_id = fr.first;
        FrameResult::Ptr frameresult = fr.second;
        double timestamp = frameresult->time_stamp_;
        auto frame_ref_KF = frameresult->reference_keyframe_ptr_.lock();
        // SE3 frame_pose = (frameresult->RelativePose() * frame_ref_KF->Pose()).inverse(); // Twc
        // 20221012 改
        SE3 frame_pose = (frameresult->RelativePose() * frame_ref_KF->Pose() * Two).inverse(); // Twc
        Vec3 pose_t = frame_pose.translation();
        Mat33 pose_R = frame_pose.rotationMatrix();
        Eigen::Quaterniond pose_q = Eigen::Quaterniond(pose_R);

        // https://blog.csdn.net/danshiming/article/details/119202955
        outfile << std::setprecision(6) << timestamp << " " 
                << pose_t.transpose() << " " << pose_q.coeffs().transpose() << std::endl;
    }

    outfile.close();
}

// --------------------------------------------------------------------------------------------------------------
void System::SaveFrameTrajectoryKITTI(std::string &save_file) {
    std::ofstream outfile;
    outfile.open(save_file, std::ios_base::out | std::ios_base::trunc);
    outfile << std::fixed;
    std::vector< std::pair<unsigned long, FrameResult::Ptr> > poses_vector;

    // 20221012 改
    auto kfs = m_map_->GetAllKeyFrames();
    SE3 Two = kfs[0]->Pose().inverse(); // 第一帧位姿 (为了将整个结果的参考系变换到第一帧坐标系)

    auto frame_results = m_map_->GetAllFrameResults();
    for(auto fr : frame_results) {
        unsigned long frame_id = fr.first;
        FrameResult::Ptr frameresult = fr.second;
        poses_vector.push_back( std::make_pair(frame_id, frameresult) );
    }

    std::sort(poses_vector.begin(), poses_vector.end(), unorder_frame_comp);
    for(auto &fr : poses_vector) {
        FrameResult::Ptr frameresult = fr.second;
        // double timestamp = frameresult->time_stamp_;
        auto frame_ref_KF = frameresult->reference_keyframe_ptr_.lock();
        // SE3 frame_pose = (frameresult->RelativePose() * frame_ref_KF->Pose()).inverse(); // Twc
        // 20221012 改
        SE3 frame_pose = (frameresult->RelativePose() * frame_ref_KF->Pose() * Two).inverse(); // Twc
        Vec3 pose_t = frame_pose.translation();
        Mat33 pose_R = frame_pose.rotationMatrix();

        outfile << std::setprecision(9) << pose_R(0, 0) << " " << pose_R(0, 1) << " " << pose_R(0, 2) << " " << pose_t(0) << " "
                                        << pose_R(1, 0) << " " << pose_R(1, 1) << " " << pose_R(1, 2) << " " << pose_t(1) << " "
                                        << pose_R(2, 0) << " " << pose_R(2, 1) << " " << pose_R(2, 2) << " " << pose_t(2) << std::endl;
    }

    outfile.close();
}

// --------------------------------------------------------------------------------------------------------------
void System::SaveFrameTrajectoryKITTI_no_y(std::string &save_file) {
    std::ofstream outfile;
    outfile.open(save_file, std::ios_base::out | std::ios_base::trunc);
    outfile << std::fixed;
    std::vector< std::pair<unsigned long, FrameResult::Ptr> > poses_vector;

    // 20221012 改
    auto kfs = m_map_->GetAllKeyFrames();
    SE3 Two = kfs[0]->Pose().inverse(); // 第一帧位姿 (为了将整个结果的参考系变换到第一帧坐标系)

    auto frame_results = m_map_->GetAllFrameResults();
    for(auto fr : frame_results) {
        unsigned long frame_id = fr.first;
        FrameResult::Ptr frameresult = fr.second;
        poses_vector.push_back( std::make_pair(frame_id, frameresult) );
    }

    std::sort(poses_vector.begin(), poses_vector.end(), unorder_frame_comp);
    for(auto &fr : poses_vector) {
        FrameResult::Ptr frameresult = fr.second;
        // double timestamp = frameresult->time_stamp_;
        auto frame_ref_KF = frameresult->reference_keyframe_ptr_.lock();
        // SE3 frame_pose = (frameresult->RelativePose() * frame_ref_KF->Pose()).inverse(); // Twc
        // 20221012 改
        SE3 frame_pose = (frameresult->RelativePose() * frame_ref_KF->Pose() * Two).inverse(); // Twc
        Vec3 pose_t = frame_pose.translation();
        Mat33 pose_R = frame_pose.rotationMatrix();

        outfile << std::setprecision(9) << pose_R(0, 0) << " " << pose_R(0, 1) << " " << pose_R(0, 2) << " " << pose_t(0) << " "
                                        << pose_R(1, 0) << " " << pose_R(1, 1) << " " << pose_R(1, 2) << " " << 0.0 << " "
                                        << pose_R(2, 0) << " " << pose_R(2, 1) << " " << pose_R(2, 2) << " " << pose_t(2) << std::endl;
    }

    outfile.close();
}

// --------------------------------------------------------------------------------------------------------------
void System::SaveMappointPCD(std::string &save_file) {
    std::ofstream outfile;
    outfile.open(save_file, std::ios_base::out | std::ios_base::trunc);
    outfile << std::fixed;
    std::vector< std::pair<unsigned long, MapPoint::Ptr> > points_vector;

    auto kfs = m_map_->GetAllKeyFrames();
    SE3 Tow = kfs[0]->Pose(); // 第一帧位姿 (为了将整个结果的参考系变换到第一帧坐标系)

    auto points_unorder = m_map_->GetAllMapPoints();
    for(auto point_unorder : points_unorder) {
        unsigned long point_id = point_unorder.first;
        MapPoint::Ptr point_ptr = point_unorder.second;
        points_vector.push_back( std::make_pair(point_id, point_ptr) );
    }

    std::sort(points_vector.begin(), points_vector.end(), unorder_landmark_comp);

    int points_len = points_vector.size();

    outfile << "# .PCD v0.7 - Point Cloud Data file format" << std::endl;
    outfile << "VERSION 0.7" << std::endl;
    outfile << "FIELDS x y z" << std::endl;
    outfile << "SIZE 4 4 4" << std::endl;
    outfile << "TYPE F F F" << std::endl;
    outfile << "COUNT 1 1 1" << std::endl;
    outfile << "WIDTH " << points_len << std::endl;
    outfile << "HEIGHT 1" << std::endl;
    outfile << "VIEWPOINT 0 0 0 1 0 0 0" << std::endl;
    outfile << "POINTS " << points_len << std::endl;
    outfile << "DATA ascii" << std::endl;

    for(auto &point_vector : points_vector) {
        MapPoint::Ptr point_ptr = point_vector.second;
        Vec3 point_pos = Tow * point_ptr->GetPos();
        outfile << point_pos[0] << " " << point_pos[1] << " " << point_pos[2] << std::endl;
    }

    outfile.close();
}

}  // namespace sg_slam
