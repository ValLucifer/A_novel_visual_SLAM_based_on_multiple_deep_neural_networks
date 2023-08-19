//
//  Created by Lucifer on 2022/7/20.
//

#include "sg_slam/keyframe.h"

namespace sg_slam {

// --------------------------------------------------------------------------------------------------------------
KeyFrame::KeyFrame() {

}

// --------------------------------------------------------------------------------------------------------------
SE3 KeyFrame::Pose() {
    std::unique_lock<std::mutex> lck(pose_mutex_); // 离开函数析构，析构释放锁.
    return pose_;
}

// --------------------------------------------------------------------------------------------------------------
void KeyFrame::SetPose(const SE3 &pose) {
    std::unique_lock<std::mutex> lck(pose_mutex_);
    pose_ = pose;
}

KeyFrame::Ptr KeyFrame::CreateKeyFrame() {
    static unsigned long keyframe_factory_id = 0;
    KeyFrame::Ptr new_keyframe(new KeyFrame); // 使用 KeyFrame() { };构造函数.
    new_keyframe->id_ = keyframe_factory_id++;
    return new_keyframe;
}

}  // namespace sg_slam
