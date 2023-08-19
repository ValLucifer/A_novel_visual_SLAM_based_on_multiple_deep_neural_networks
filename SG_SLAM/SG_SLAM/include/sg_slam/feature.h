//
//  Created by Lucifer on 2022/7/20.
//

#ifndef _SG_SLAM_FEATURE_H_
#define _SG_SLAM_FEATURE_H_

#include <Eigen/Core>  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#include <memory>  // 智能指针

#include <mutex>  // C++14多线程中的<mutex>提供了多种互斥操作，可以显式避免数据竞争。

// #include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "sg_slam/mappoint.h"

namespace sg_slam {

// forward declare
class Frame;
class KeyFrame;
// class MapPoint;

/**
 * @brief 2D 特征点, 在三角化之后会被关联一个地图点
 * 可认为是用来整合本关键帧需要处理的2D-3D对应关系，需要大改. N.B.: !!!!!
 * 
 */
class Feature 
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  // 详见 sg_slam/frame.h
    using Ptr = std::shared_ptr<Feature>;

    // unsigned long id_ = 0;  // ID  // 可不用加直接使用 Feature指针判断即可 参见 src/sg_slam/mappoint.cpp RemoveObservation.
    // // c++14 std::weak_ptr 是一种不控制对象生命周期的智能指针, 它指向一个 shared_ptr 管理的对象. 进行该对象的内存管理的是那个强引用的 shared_ptr. weak_ptr只是提供了对管理对象的一个访问手段.
    std::weak_ptr<KeyFrame> KeyFrame_;  // 持有该feature的KeyFrame, 即参考关键帧
    cv::KeyPoint position_;  // 2D 提取位置  // SLAM不会改变为原始数据. (在参考关键帧中关键点的像素位置)
    // 应该加上描述子及匹配置信度(superglue) N.B.:

    // My add 20220913
    float kpR_u_ = -1.0;

    // N.B.: 加锁不然当，map_point_ reset()后，多线程调用会产生段错误.
    std::weak_ptr<MapPoint> map_point_; // 关联地图点, 指针为了共同维护数据.
    std::mutex map_point_mutex_;

    bool is_outlier_ = false; // 是否为异常点(N.B.: 后端使用，回环和前端不使用, 故不用加锁)
    bool is_on_left_image_ = true; // 标识是否提在左图，false为右图, N.B.: 此标识可不要，因为此系统优化只在左图中进行!!

public:
    Feature() { }

    Feature(std::shared_ptr<KeyFrame> keyframe, const cv::KeyPoint &kp);

    ~Feature() = default;

    MapPoint::Ptr GetMapPoint() {
        std::unique_lock<std::mutex> lck(map_point_mutex_);
        /**
        if(map_point_.expired()) {
            return nullptr;
        }
        **/

        return map_point_.lock();
    }

    void EraseMapPoint() {
        std::unique_lock<std::mutex> lck(map_point_mutex_);
        map_point_.reset();
    }

    void SetMapPoint(MapPoint::Ptr mp) {
        std::unique_lock<std::mutex> lck(map_point_mutex_);
        map_point_ = mp;
    }
    
};  // class Feature

}  // namespace sg_slam


#endif  // _SG_SLAM_FEATURE_H_
