//
//  Created by Lucifer on 2022/8/1.
//

#ifndef _SG_SLAM_LOOP_CLOSING_H_
#define _SG_SLAM_LOOP_CLOSING_H_

#include <Eigen/Core>  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
// https://blog.csdn.net/qq_41035283/article/details/122535367
#include <memory>  // 智能指针

#include <thread>  // 多线程 C++11/14 参见 sg_slam/viewer.h

#include <atomic>  // 参见 sg_slam/backend.h

#include <map>
#include <unordered_map>
#include <set>

#include "sg_slam/camera.h"
#include "sg_slam/map.h"
#include "sg_slam/backend.h"
#include "sg_slam/deeplcd.h"
// #include "sg_slam/ORBextractor.h"
#include "sg_slam/keyframe.h"

#include "sg_slam/common_include.h"

namespace sg_slam {

/**
 * @brief loopclosing (回环检测)
 * 单独回环线程
 */
class LoopClosing 
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  // 参见 sg_slam/frame.h
    using this_type = LoopClosing;
    using Ptr = std::shared_ptr<LoopClosing>;

    std::atomic<bool> m_is_stop_flag;  // N.B.: 结束线程时， 其他线程查看，此线程是否已经停止了(自加)!!!!! 防止误操作空指针

    // test 20221019
    int m_detect_loop_number = 0;
    int m_match_loop_number = 0;
    int m_correct_number = 0;

    // 构造函数
    LoopClosing();
    ~LoopClosing() = default;

    // 设置左右目相机，用于获取相机的内外参
    void SetCameras(Camera::Ptr left, Camera::Ptr right);

    // 设置地图
    // void SetMap(std::shared_ptr<Map> map);
    void SetMap(Map::Ptr map);

    // N.B.: 设置ORB提取器，本系统不用, 本系统使用SG提取器及匹配器.
    // void SetORBextractor(std::shared_ptr<ORBextractor> orb){
    //     _mpORBextractor = orb;
    // }

    // 设置后端.
    void SetBackend(Backend::Ptr backend);

    // N.B.: My add
    // 插入缓存数据，关键帧插入地图后插入.
    // 前端调用.
    // N.B.: 实现清华工程的 InsertNewKeyFrame() 函数功能.
    // 参见 sg_slam/backend.h InsertProcessKeyframe()
    void InsertProcessKeyframe();

    // N.B.: My add
    // 检测缓存队列是否不为空，以此来启动优化.
    // N.B.: 实现清华工程的 CheckNewKeyFrames() 函数功能.
    // 参见 sg_slam/backend.h CheckNeedProcessKeyframe()
    bool CheckNeedProcessKeyframe();

    // 关闭回环线程
    void Stop();

private:
    // the main loop of loopclosing thread
    void LoopClosingRun();

    // add new KF(已经执行过回环检测的) to the KF Database (为了检测回环)
    void AddToDatabase();

    // extract one KF from the list(缓冲区)
    // calculate its DeepLCD descriptor vector
    // 从缓存区中提取一个关键帧，计算回环描述符(类词向量)
    void ProcessNewKF();

    /**
     * @brief use DeepLCD to find potential KF candidate for the current KF,
     * return true if successfully detect one.
     * 
     */
    bool DetectLoop();

    // match the sg descriptors (不改变关键帧存储的特征点数量)
    // return true if number of valid matches is enough
    bool MatchFeatures();

    /**
     * @brief 计算当前帧的位姿修正量(与前端类似)(检测到回环后才执行)
     * compute the correct pose of current KF using PnP solver and g2o optimization,
     * return true if number of inliers is enough (优化结果有足够多内点时, 返回 true)(估计结构较好)
     * 
     */
    bool ComputeCorrectPose();

    /**
     * @brief 
     * use g2o to optimize the correct current pose (Tcw), 
     * this function is called in ComputeCorrectPose()
     * return the number of inliers
     */
    int OptimizeCurrentPose();
    
    /**
     * @brief 根据检测到的回环，对当前关键帧和之前的所有关键帧进行修正
     * N.B.: 与清华工程不同点如下:
     * (这里的"所有"，本系统特指回环帧(旧)之后的帧(可能修正到回环帧后一帧，或后几帧，因为越接近回环帧，累积误差越小))
     * (从当前(处理)帧(相对的)开始，最远到回环帧的后一帧，回环帧为这个回环的基准无法用这个回环信息修正其累积误差，修正到那一帧为止可以通过实验确定)
     * (在保证精度的情况下，越短越好，可通过(外化)阈值调节)
     * 
     */
    void LoopCorrect();
    // N.B.: 原清华工程的 LoopLocalFusion();
    void LoopCorrectActive();

    // N.B.: 原清华工程的 PoseGraphOptimization();
    // 即与 LoopCorrectPreviousKFandMappoint() 功能等同.
    void PoseGraphOptimization();



private:
    // // N.B.: 原清华工程的 _mpMap
    Map::Ptr m_map_;  // 为了与地图交互
    // // N.B.: 原清华工程的 _mpBackend
    std::weak_ptr<Backend> m_backend_;  // 为了与后端线程交互 (N.B.: 后端由系统持有)
    // // N.B.: 原清华工程的 _mpDeepLCD
    DeepLCD::Ptr m_deeplcd_;  // 为了检测回环

    // // N.B.: 原清华工程的 _mthreadLoopClosing
    std::thread m_loopclosing_thread_; // N.B.: 多线程处理句柄(handle) 参见 sg_slam/viewer.h
    // // N.B.: 原清华工程的 _mbLoopClosingIsRunning
    std::atomic<bool> m_loopclosing_running_;  // N.B.: 参见 sg_slam/backend.h

    // // N.B.: 原清华工程的 _mpCameraLeft, _mpCameraRight
    Camera::Ptr m_cam_left_ = nullptr;
    Camera::Ptr m_cam_right_ = nullptr;

    // std::shared_ptr<ORBextractor> _mpORBextractor;

    // N.B.: My Add (关键帧和激活区缓存数据库)
    // 待处理关键帧队列
    // // N.B.: 原清华工程的 _mlNewKeyFrames
    std::list< KeyFrame::Ptr > m_process_keyframes_list_;  // 待处理关键帧队列
    std::mutex m_loopclosing_cache_mutex_;  //后端缓存锁. 
    
    // // N.B.: 原清华工程的 _mpLastClosedKF
    KeyFrame::Ptr m_last_closed_KF_ = nullptr;  // 上一次(最近一次)检测到回环的处理帧
    // // N.B.: 原清华工程的 _mpLastKF.
    KeyFrame::Ptr m_last_KF_ = nullptr;  // 上一次加入数据库的处理帧(没有对应回环帧的处理帧).
    // // N.B.: 原清华工程的 _mpCurrentKF.
    KeyFrame::Ptr m_current_KF_ = nullptr;  // 当前处理帧.
    // // N.B.: 原清华工程的 _mpLoopKF.
    KeyFrame::Ptr m_loop_KF_ = nullptr;  // 当前处理帧对应的回环帧.
    // // N.B.: 原清华工程的 _msetValidFeatureMatches. 
    // 当前处理帧与对应的回环帧之间的匹配特征index(索引)对.(类std::vector<cv::DMatch>)
    // 参见 test/test_kitti_sg_key_track.cpp
    // currentFeatureIndex, loopFeatureIndex
    std::set<std::pair<int, int>> m_setValidFeatureMatches_; 

    // N.B.: My add (用于保存已经修正的不参与优化修正的关键帧ID和地图点ID)
    std::set< unsigned long > m_setCorrectedKeyframes_id_;
    std::set< unsigned long > m_setCorrectedMappoints_id_;

    // N.B.: My add (用于保存 PoseGraphOptimization 用到的有回环的之前关键帧的ID)
    std::list< unsigned long > m_last_loopKPs_id_list_;
    // N.B.: My add (用于保存PoseGraphOptimization 保持的最早有回环的之前关键帧的ID)
    std::list< unsigned long > m_most_old_loopKPS_id_list_;

    // N.B.: 仅在回环线程中使用，不用加锁.
    // 键为关键帧ID，值为关键帧(共享)指针
    // 用于搜索回环帧的数据库.
    // // N.B.: 原清华工程的 _mvDatabase
    // https://wenku.baidu.com/view/dbbfc72af211f18583d049649b6648d7c1c70875.html
    // std::map 默认对键进行升序排序(越前面的键越小，即ID越小，亦即越早)
    std::map<unsigned long, KeyFrame::Ptr> m_loopclosing_database_;

    // // N.B.: 原清华工程的 _mseCorrectedCurrentPose
    // 回环修正后的当前处理帧位姿(Tcw).
    // 参见 sg_slam/loopclosing.cpp LoopClosing::ComputeCorrectPose()
    Sophus::SE3d m_corrected_current_pose_;
    // SE3 m_corrected_current_pose_;
    // // N.B.: 原清华工程的 _mbNeedCorrect
    bool m_bNeedCorrect_flag = false;  // 是否进行回环修正标志，程序中根据回环结果自动设置.
    

    // N.B.: My Add
    // Settings
    // // N.B.: 原清华工程的 _similarityThres1
    float m_lcd_similarityScoreThreshold_high_ = 0.94;  
    // // N.B.: 原清华工程的 _similarityThres2
    float m_lcd_similarityScoreThreshold_low_ = 0.92;
    // // N.B.: 原清华工程的 _mnDatabaseMinSize
    // the system won't do loop detection until the number of KFs in database is more than this threshold
    unsigned long m_lcd_nDatabaseMinSize = 50;
    
    // // 是否显示回环检测的匹配结果 
    // // N.B.: 原清华工程的 _mbShowLoopClosingResult
    bool m_show_LoopClosing_result_flag_ = false;

    unsigned long m_insert_loopclosing_interval_ = 5;

    float m_sg_loop_match_threshold_ = 0.2;

    unsigned long m_pose_graphloop_edge_num_ = 3;
    unsigned long m_pose_graph_most_old_loop_edge_num_ = 5;

    int m_valid_feature_matches_th_ = 224;

    // 20220921 
    double m_valid_features_matches_ratio_th_ = 0.12;

    // 20221014 加
    bool m_is_first_loopclosing_ = true;

};  // class LoopClosing

}  // namespace sg_slam

#endif  // _SG_SLAM_LOOP_CLOSING_H_
