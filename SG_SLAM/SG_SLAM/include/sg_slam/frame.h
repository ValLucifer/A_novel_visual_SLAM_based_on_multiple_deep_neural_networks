//
//  Created by Lucifer on 2022/7/10 改.
//

#ifndef _SG_SLAM_FRAME_H_
#define _SG_SLAM_FRAME_H_

#include <Eigen/Core>  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#include <memory>  // 智能指针
#include <mutex>  // C++14多线程中的<mutex>提供了多种互斥操作，可以显式避免数据竞争。
#include <vector>  // vector是C++容器库中非常通用的一种容器

// #include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <torch/torch.h>

#include "sg_slam/common_include.h"

namespace sg_slam {

// forward declare
// class Feature;  // 不需要实现.h/.cpp，及包含.h，只是一个声明, 实际运行时再找实现.
class KeyFrame;  // 不需要实现.h/.cpp，及包含.h，只是一个声明, 实际运行时再找实现.
class MapPoint;

/**
 * @brief Frame(帧), 每一帧分配独立id. 
 */
class Frame
{
public:
    // https://blog.csdn.net/shyjhyp11/article/details/123208279
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  // 用于对齐内存，用于对齐"使用-march=native与不使用-march=native"时的Eigen库, 个人认为最好加上.
    using this_type = Frame;  // 本类别名;
    using Ptr = std::shared_ptr<Frame>; // 本类共享/弱指针别名

    unsigned long id_ = 0;  // id of this frame // 非负
    // unsigned long keyframe_id_ = 0; // keyframe id of this frame, 按关键帧计数与id_不同.
    bool is_keyframe_ = false; // Is this frame a key frame?
    double time_stamp_;  // 时间戳, 用以将结果保存为TUM真值轨迹格式.
    // long double time_stamp_;

    // 2022101101 加
    bool is_use_cut_image = false;

    // 2022100502 加 (与opencv 关键点数据类型float匹配)
    // RGBD 深度尺度因子
    float depth_scale_ = 5000.0;  // (TUM)

    // N.B.: KeyFrame类中有对应锁，不用加锁
    // https://blog.csdn.net/m0_51955470/article/details/118074872
    // 参见 src/sg_slam/mappoint.cpp RemoveObservation.
    std::weak_ptr< KeyFrame > reference_keyframe_ptr_;  // 本帧的参考关键帧 // 弱智能指针不可以赋值为nullptr

    // N.B.: 是关键帧的特性
    // SE3 pose_; // pose with form of Tcw.
    // std::mutex pose_mutex_;  // Data lock of pose
    // N.B.: 需多线程(仅跟踪和可视化线程使用)维护要加互斥锁.
    SE3 relative_pose_;  // pose with form of Tc, keyframe.
    std::mutex relative_pose_mutex_;  // Data lock of pose

    // N.B.: 只在单线程处理.
    cv::Mat left_img_, right_img_;  // stereo images.
    // 2022100502 加
    cv::Mat depth_img_;  // depth image (RGB-D)
    // 2022101101 加
    cv::Mat left_cut_img_, right_cut_img_;
    // 20221012 加
    cv::Mat depth_cut_img_;

    // N.B.: Feature类中有对应锁，不用加锁
    // N.B.: 是关键帧的特性
    // extracted features in left image (已是处理完的特征点不是提取器原对应的特征点, 在双目情况下为已经与右图/之前帧匹配且有对应三维点的特征)
    // std::vector< std::shared_ptr<Feature> > features_left_;
    // corresponding features in right image, set to nullptr if no corresponding
    // std::vector< std::shared_ptr<Feature> > features_right_;  // N.B.: 个人认为可在提取时直接(与左图)匹配，且只在关键帧出提取特征即可.

    // N.B.: Frame 中的特征点和描述子只保存左图的.
    // Superpoint keypoint's number is n
    std::vector< cv::KeyPoint > mvKeys;
    std::vector< cv::KeyPoint > mvKeysUn;

    // My add 20220913
    std::vector< float > mvKeysR_u;  // 与 mvKeys 大小一样，没有匹配的为-1. 非-1项在初始化时与mvpMapPoints非nullptr项对应 (x)
    // My add 20220914
    std::vector< float > mvKeysR_v;  // 与 mvKeys 大小一样，没有匹配的为-1. 非-1项在初始化时与mvpMapPoints非nullptr项对应 (y) 与 mvKeysR_u非-1位置完全一致
    // 2022100502 加
    std::vector< float > mvKeysDepth; //  与 mvKeys 大小一样，没有匹配深度的为-1 (RGBD专用)
    
    // N.B. Superpoint+kenc descriptor [b, d, n]/[c, h, w]
    // N.B.: 使用 std::vector<float> 存储描述子数据，减少内存消耗，需使用时用 sg_slam/sg_detectMatcher.h 中的相应函数转换为torch::Tensor.
    // torch::Tensor mDescriptors_tensor;
    // cv::Mat mDescriptors_mat;
    int64_t Descripters_B = 1;
    int64_t Descripters_D = 256;
    int64_t Descripters_N = 0;
    std::vector<float> mDescriptors_data;

    // MapPoints associated to keypoints, nullptr if no association.
    std::vector< std::weak_ptr<MapPoint> > mvpMapPoints;
    int keypoints_number = 0;  // Number of KeyPoints. // N.B.: 改为有对应地图点的关键点数会更好.

public:
    // data members
    Frame();
    // 形参引用赋值给其他变量后，不与其他变量共享存储空间，与指针不同，其他同类型类变量已有存储空间.
    // Frame(unsigned long id, double time_stamp, const SE3 &pose, const cv::Mat &left, const cv::Mat &right);

    ~Frame() = default;

    // set and get pose, thread safe
    SE3 RelativePose();
    void SetRelativePose(const SE3 &relative_pose);

    void SetKeyFrame();

    /// 工厂构建模式，分配id
    /// create Frame and assign id of frame
    static Frame::Ptr CreateFrame();

    // 2022100502 加
    // find the depth in depth map (RGBD使用)
    float findDepth( const cv::KeyPoint& kp );


};  // class Frame

/**
 * @brief 用于保存 Frame 中得到的结果，主要是为了减少内存消耗.
 * 
 */
class FrameResult
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using this_type = FrameResult;  // 本类别名;
    using Ptr = std::shared_ptr<FrameResult>; // 本类共享/弱指针别名

    unsigned long id_ = 0;
    bool is_keyframe_ = false; // Is this frame a key frame?
    double time_stamp_;  // 时间戳, 用以将结果保存为TUM真值轨迹格式.

    // N.B.: 需要加锁(在地图类中设置)
    bool is_active_key_flag_ = false;  // Is this frame a active key frame?
    std::mutex active_key_flag_mutex_;

    SE3 relative_pose_;  // pose with form of Tc, keyframe.
    std::mutex relative_pose_mutex_;  // Data lock of pose

    // 参见 src/sg_slam/mappoint.cpp RemoveObservation.
    std::weak_ptr< KeyFrame > reference_keyframe_ptr_;  // 本帧的参考关键帧 // 弱智能指针不可以赋值为nullptr

public:
    FrameResult(Frame::Ptr frame_ptr);
    ~FrameResult() = default;

    SE3 RelativePose() {
        std::unique_lock<std::mutex> lck(relative_pose_mutex_); // 离开函数析构，析构释放锁.
        return relative_pose_;
    }

    void SetRelativePose(const SE3 &relative_pose) {
        std::unique_lock<std::mutex> lck(relative_pose_mutex_);
        relative_pose_ = relative_pose;
    }

    bool ActiveKeyFlag() {
        std::unique_lock<std::mutex> lck(active_key_flag_mutex_);
        return is_active_key_flag_;
    }

    void SetActiveKeyFlag(bool flag) {
        std::unique_lock<std::mutex> lck(active_key_flag_mutex_);
        is_active_key_flag_ = flag;
    }

};  // class Frame

}  // namespace sg_slam

#endif  // _SG_SLAM_FRAME_H_
