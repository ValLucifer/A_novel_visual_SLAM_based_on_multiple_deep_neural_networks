//
//  Created by Lucifer on 2022/7/25.
//

#ifndef _SG_SLAM_VIEWER_H_
#define _SG_SLAM_VIEWER_H_

#include <Eigen/Core>  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // 参见 sg_slam/frame.h
#include <memory>  // 参见 sg_slam/frame.h
#include <mutex>  // C++14多线程中的<mutex>提供了多种互斥操作，可以显式避免数据竞争。

// https://wenku.baidu.com/view/acab3e86b3717fd5360cba1aa8114431b90d8e2c.html
// https://blog.csdn.net/shiwujigegea/article/details/119348526
// https://blog.csdn.net/weixin_43836778/article/details/93196507?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-93196507-blog-118102382.pc_relevant_multi_platform_whitelistv1&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-93196507-blog-118102382.pc_relevant_multi_platform_whitelistv1&utm_relevant_index=1
#include <thread>  // 多线程 C++11/14 
#include <unordered_map> 

// #include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <pangolin/pangolin.h>

#include "sg_slam/map.h"
#include "sg_slam/frame.h"
#include "sg_slam/keyframe.h"
#include "sg_slam/mappoint.h"

namespace sg_slam {

/**
 * @brief 可视化
 * 
 */
class Viewer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  // 参见 sg_slam/frame.h
    using this_type = Viewer;  // 本类别名;
    using Ptr = std::shared_ptr<Viewer>;  // 本类共享/弱指针别名

    bool m_viewer_running_ = true;

    std::atomic<bool> m_is_stop_flag;  // N.B.: 结束线程时， 其他线程查看，此线程是否已经停止了(自加)!!!!! 防止误操作空指针

    static float Viewer_sz;  // N.B.: 这样实现不好(但改动量)较小!!!! 20221028

public:
    Viewer();
    ~Viewer() = default;

    void SetMap(Map::Ptr map) {
        map_ = map;
    }

    void Close();

    // 增加一个当前帧(加入关键帧判定, 用不同颜色区分关键帧和普通帧)
    // add the current frame to viewer
    void AddCurrentFrame(Frame::Ptr currentFrame);

    // N.B.: 是否要加如一个加入当前关键帧的多态函数呢? 和一个加入 FrameResult的函数.

    // 更新地图
    void UpdateMap();

private:
    void ThreadLoop();

    // show the current frame's left image and feature points(有对应地图点的)
    // plot the features in current frame into an image
    cv::Mat PlotFrameImage();

    void DrawFrame(Frame::Ptr frame, const float *color);
    // void DrawFrame(KeyFrame::Ptr keyframe, const float *color);
    void DrawFrameResult(FrameResult::Ptr frame_result, const float *color);

    // N.B.: 原 DrawMapPoints 的功能.
    void DrawMapPointsAndFrameResult(const bool menuShowFrames, const bool menuShowPoints);

    void FollowCurrentFrame(pangolin::OpenGlRenderState &vis_camera);

    // void DrawKFsAndMPs(const bool menuShowKeyFrames, const bool menuShowPoints);

private:
    Frame::Ptr current_frame_ = nullptr;

    Map::Ptr map_ = nullptr;  // N.B.: 可视化是处理流程，map是基础数据结构，二者不会存在循环引用，故不需要使用std::weak_ptr!

    std::thread viewer_thread_;  // N.B.: 多线程处理句柄(handle)

    // N.B.: 自己改了
    std::unordered_map<unsigned long, KeyFrame::Ptr> active_keyframes_map_;
    std::unordered_map<unsigned long, MapPoint::Ptr> active_landmarks_map_;
    std::unordered_map<unsigned long, MapPoint::Ptr> landmarks_map_;  // all landmarks 
    std::unordered_map<unsigned long, FrameResult::Ptr> frame_results_map_;
    std::unordered_map<unsigned long, Frame::Ptr> active_frames_;

    // bool map_updated_ = false;

    std::mutex viewer_data_mutex_;

    int delay_mT_ = 1;  // 单位是毫秒 // cv::wait()为了显示.

    // 定义 pangolin 使用的颜色
    const float m_red_[3] = {1.0, 0.0, 0.0};
    const float m_green_[3] = {0.0, 1.0, 0.0};
    const float m_blue_[3] = {0.0, 0.0, 1.0};
    const float m_yellow_[3] = {1.0, 1.0, 0.0};
    const float m_purple_[3] = {1.0, 0.0, 1.0};

    // 20230308
    int m_image_width_ = 1241;
    int m_image_height_ = 376;

};  // class Viewer

}  // namespace sg_slam

#endif  // _SG_SLAM_VIEWER_H_
