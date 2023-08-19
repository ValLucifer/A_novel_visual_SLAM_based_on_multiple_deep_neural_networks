//
//  Created by Lucifer on 2022/8/7.
//

#ifndef _SG_SLAM_FRONTEND_H_
#define _SG_SLAM_FRONTEND_H_

#include <Eigen/Core>  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#include <memory>  // 智能指针
#include <unordered_map> // 20221018 add
#include <deque>  // 20221104 add

#include "sg_slam/frame.h"
#include "sg_slam/map.h"
#include "sg_slam/backend.h"
#include "sg_slam/loopclosing.h"
#include "sg_slam/viewer.h"
#include "sg_slam/camera.h"

#include "sg_slam/common_include.h"

namespace sg_slam {

enum class FrontendStatus { INITIALIZING, TRACKING_GOOD, TRACKING_BAD, LOST };

/**
 * @brief 前端
 *  估计当前帧相对参考关键帧的位姿，在满足关键帧条件时向地图加入关键帧
 */
class Frontend 
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  // 参见 sg_slam/frame.h
    using this_type = Frontend;  // 本类别名
    using Ptr = std::shared_ptr<Frontend>;  // 本类共享/弱指针别名

    // 20221009 加
    CameraType m_camera_type = CameraType::SETERO;
    bool m_frontend_constant_speed_flag = true;

    // 20221101 add
    // N.B.: 这种实现不好
    int m_stereo_dataset_type = 0;  // 0: KITTI, 1: EuRoC

    // test
    double m_test_pworld_depth_max = 0.0;
    double m_test_pworld_depth_min = 10000.0;

    double m_test_pworld_x_max = 0.0;
    double m_test_pworld_x_min = 10000.0;

    double m_test_pworld_y_max = 0.0;
    double m_test_pworld_y_min = 10000.0;

    // 20221030 add
    unsigned long m_key_max_id_interval = 10;

    // 20221104 add
    std::deque< std::weak_ptr< KeyFrame > > m_local_keyframes_deque;
    size_t m_local_keyframes_deque_length = 2;

    Frontend();
    ~Frontend() = default;

    // 外部接口，添加一个帧并计算其定位结果
    bool AddFrame(Frame::Ptr frame);

    // 设置地图
    void SetMap(Map::Ptr map);

    // 设置后端
    void SetBackend(Backend::Ptr backend);

    // 设置回环
    void SetLoopClosing(LoopClosing::Ptr loopclosing);

    // 设置可视化器
    void SetViewer(Viewer::Ptr viewer);

    // 设置左右目相机，用于获取相机的内外参
    void SetCameras(Camera::Ptr left, Camera::Ptr right);

    // N.B.: My add 
    // 设置前端的追踪状态
    void SetStatus(FrontendStatus status);

    // 给出当前前端的追踪状态
    FrontendStatus GetStatus() const;

private:
    /**
     * @brief Track in normal mode
     * @return true if success
     * 
     */
    bool Track();

    // N.B.: 自己加的，方便后续处理, 和消融实验
    /**
     * @brief 提取当前帧的关键点和描述子
     * 
     */
    void ExtractCurrent();

    /**
     * @brief Track with last frame
     * @return num of tracked points
     * 
     */
    int TrackLastFrame();

    // 20221018
    /**
     * @brief Track with last frame and create temporary 3D points
     * @return num of tracked points
     * 
     */
    int TrackLastFrame3D();

    // 20221014 20221018
    /**
     * @brief Estimate Relative Pose Current for last
     * @return num of inliers
     */
    int EstimateCurrentRelativePoseForLast3D(const SE3 &intial_estimate, int ref_inliers);

    // N.B.: 自改
    /**
     * @brief Estimate Relative Pose Current for last
     * @return num of inliers
     * N.B.: 实现原工程的 EstimateCurrentPose() 的功能
     */
    int EstimateCurrentRelativePoseForLast();

    // My add 20220915
    int EstimateCurrentRelativePoseForLastStereo();

    // N.B.: My add
    /**
     * @brief Track with reference keyframe
     * @return num of tracked points
     * 
     */
    int TrackReferenceKeyframe();

    // N.B.: 20221014 add
    /**
     * @brief Track with reference keyframe
     * @return num of tracked points
     * 
     */
    int TrackReferenceKeyframeNew();


    // 20221104
    /**
     * @brief Track with local keyframes
     * 
     */
    int TrackLocalKeyframes();

    // N.B.: My add
    /**
     * @brief Estimate Relative Pose Current for Reference
     * @return num of tracked points
     * input: 
     *      intial_estimate: 为迭代计算的初值
     *      last_inliers： 为使用上一帧估计当前帧的相对位姿的最终内点数，用于融合
     * 
     */
    int EstimateCurrentRelativePoseForReference(const SE3 &intial_estimate, int last_inliers);

    // My add 20220919
    int EstimateCurrentRelativePoseForReferenceStereo(const SE3 &intial_estimate, int last_inliers);

    // 20221014 add
    /**
     * @brief Estimate Relative Pose Current for Reference
     * @return num of tracked points
     * input: 
     *      intial_estimate: 为迭代计算的初值
     */
    int EstimateCurrentRelativePoseForReferenceNew(const SE3 &intial_estimate);

    // N.B.: My add
    /**
     * @brief track additional landmarks from reference
     * N.B.: 为当前帧从参考关键帧中补充地图点(只填补没有对应地图点的位置)
     *  
     */
    void TrackAdditionalLandmarksFromReference();

    // 20221018 加
    /**
     * @brief track additional landmarks from last
     * N.B.: 为当前帧从参考关键帧中补充地图点(只填补没有对应地图点的位置)
     *  
     */
    void TrackAdditionalLandmarksFromLast3D();

    // N.B.: My add
    // N.B.: 可以模仿设计单目的地图点生成
    /**
     * @brief triangulate new landmarks in stereo
     * N.B.: 为当前帧补充新地图点，为了生成的新关键帧有足够多的地图点(只填补没有对应地图点的位置, 使得新关键帧的地图点尽可能与之前关键帧有联系).
     * N.B.: 实现原工程的 TriangulateNewPoints() 和 FindFeaturesInRight() 的功能.
     * 
     */
    void StereoCreateNewPoints();

    // 20221101 add
    /**
     * @brief left-right new landmarks in stereo
     * N.B.: 用左右视差计算
     * N.B.: 为当前帧补充新地图点，为了生成的新关键帧有足够多的地图点(只填补没有对应地图点的位置, 使得新关键帧的地图点尽可能与之前关键帧有联系).
     * N.B.: 实现原工程的 TriangulateNewPoints() 和 FindFeaturesInRight() 的功能.
     * 
     */
    void Stereo_LR_CreateNewPoints();

    // 20221013 add
    /**
     * @brief create new landmarks in RGBD
     * N.B.: 为当前帧补充新地图点，为了生成的新关键帧有足够多的地图点(只填补没有对应地图点的位置, 使得新关键帧的地图点尽可能与之前关键帧有联系).
     */
    void RGBDCreateNewPoints();

    // 20221009 加
    /**
     * @brief 总的初始化函数，不同传感器在这里切换
     * 
     */
    bool Init();

    // N.B.: 可以模仿设计单目的初始化
    /**
     * @brief Try init the frontend with stereo images saved in current_frame_
     * @return true if sucess
     */
    bool StereoInit();

    // 20221101 add
    /**
     * @brief Try init the frontend with stereo images saved in current_frame_  (左右视差计算深度)
     * @return true if sucess
     */
    bool Stereo_LR_Init();

    // 20221013 加
    /**
     * @brief Try init the frontend with RGBD images saved in current_frame_
     * @return true if sucess
     */
    bool RGBDInit();

    /**
     * @brief create new keyframe (according to current frame) and do some update work
     * @return true if success
     * 根据清华工程结构实现，十四讲工程没有.
     */
    bool InsertKeyFrame();

    /**
     * @brief Reset when lost
     * @return true if success
     * 
     */
    bool Reset();
    


private:

    // // 原工程的 status_
    FrontendStatus m_status_ = FrontendStatus::INITIALIZING;

    // My add 20220917
    FrontendStatus m_last_status_ = FrontendStatus::INITIALIZING;

    // My add 20220918
    int m_last_lost_number_ = 0;

    // // 原工程的 current_frame_
    Frame::Ptr m_current_frame_ = nullptr;  // 当前帧
    // // 原工程的 last_frame_
    Frame::Ptr m_last_frame_ = nullptr;  // 上一帧
    // // 原工程的 relative_motion_
    // N.B.: 若融合IMU, 可由IMU测量得到, 运动方程
    SE3 m_relative_motion_;  // 当前帧与上一帧的相对运动，用于估计当前帧位姿的初值.


    Map::Ptr m_map_ = nullptr;
    Backend::Ptr m_backend_ = nullptr;
    LoopClosing::Ptr m_loopclosing_ = nullptr;
    Viewer::Ptr m_viewer_ = nullptr;

    Camera::Ptr m_cam_left_ = nullptr;
    Camera::Ptr m_cam_right_ = nullptr;

    // N.B.: My add 
    // 自己加的，用于存储当前帧与参考关键帧的有对应地图点的匹配
    // currentKeypointIndex, ReferenceKeypointIndex
    // 参见 sg_slam/loopclosing.h m_setValidFeatureMatches_
    std::set<std::pair<int, int>> m_currentForReference_matches_with_mappoint_set_;

    // 20221018 add
    // std::set<std::pair<int, Vec3>> m_currentForLast_matches_with_3d_points_set_;
    std::unordered_map<int, Vec3> m_currentForLast_matches_with_3d_points_unorderedmap_;

    // My add 20220921
    // int m_extract_current_keypoint_number_ = 0;

    // N.B.: My Add
    // Settings
    // // 原工程的 num_features_init_
    int m_num_features_Threshold_init_ = 100;  // 初始化所需要的最少特征点数(可以根据匹配器适当放大，更稳定)
    // // 原工程的 num_features_tracking_
    int m_num_features_Threshold_tracking_ = 50;  // 帧间(上一帧和参考关键帧)跟踪(优化后的内点)是好的最少数目(可以根据匹配器适当放大，更稳定)
    // // 原工程的 num_features_tracking_bad_
    int m_num_features_Threshold_tracking_bad_ = 20;  // 帧间(上一帧和参考关键帧)跟踪(优化后的内点)是坏的(但不是丢失)最少数目(不超过上一个阈值)(可以根据匹配器适当放大，更稳定)

    // My add 20220921
    double m_ratio_features_th_tracking_ = 0.14;
    double m_ratio_features_th_tracking_bad_ = 0.06;
    int m_num_fearures_tracking_min_ = 10;
    
    // My add
    int m_max_keypoints_init_ = 2000;  // 初始化时，每一帧提取的最大关键点数量.
    int m_max_keypoints_tracking_ = 1000;  // 跟踪时，每一帧提取的最大关键点数量.
    int m_max_keypoints_key_R_ = 1000; // 生成关键帧时，右图片提取的最大关键点数量.

    // My add
    float m_sg_Track_match_threshold_ = 0.2;
    float m_sg_LR_match_threshold_ = 0.2;

    double m_mappoint_camera_depth_max_ = 7000.0;
    double m_mappoint_camera_depth_min_ = 0.0;

    double m_mappoint_camera_x_max_ = 7000.0;
    double m_mappoint_camera_x_min_ = -7000.0;

    double m_mappoint_camera_y_max_ = 7000.0;
    double m_mappoint_camera_y_min_ = -7000.0;


    double m_mappoint_camera_z_track_max_ = 200.0;
    double m_mappoint_camera_z_track_min_ = 2.0;
    

};  // class Frontend

}  // namespace sg_slam

#endif  // _SG_SLAM_FRONTEND_H_
