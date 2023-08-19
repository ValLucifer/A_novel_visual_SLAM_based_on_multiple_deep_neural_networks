#include "sg_slam/frame.h"

namespace sg_slam {

// --------------------------------------------------------------------------------------------------------------
/**
Frame::Frame(unsigned long id, double time_stamp, const SE3 &pose, const cv::Mat &left, const cv::Mat &right) 
    : id_(id), time_stamp_(time_stamp), pose_(pose), left_img_(left), right_img_(right) {

}
**/

// --------------------------------------------------------------------------------------------------------------
Frame::Frame() {

}

// --------------------------------------------------------------------------------------------------------------
SE3 Frame::RelativePose() {
    std::unique_lock<std::mutex> lck(relative_pose_mutex_); // 离开函数析构，析构释放锁.
    return relative_pose_;
}

// --------------------------------------------------------------------------------------------------------------
void Frame::SetRelativePose(const SE3 &relative_pose) {
    std::unique_lock<std::mutex> lck(relative_pose_mutex_);
    relative_pose_ = relative_pose;
}

// --------------------------------------------------------------------------------------------------------------
// 关键帧处理应该分离
void Frame::SetKeyFrame() {
    // static long keyframe_factory_id = 0;   
    is_keyframe_ = true;
    // keyframe_id_ = keyframe_factory_id++;
}

// --------------------------------------------------------------------------------------------------------------
Frame::Ptr Frame::CreateFrame() {
    static unsigned long factory_id = 0;
    Frame::Ptr new_frame(new Frame); // 使用 Frame() { };构造函数.
    new_frame->id_ = factory_id++;
    return new_frame;
}

// 2022100502 加
// --------------------------------------------------------------------------------------------------------------
float Frame::findDepth( const cv::KeyPoint& kp ) {
    // https://blog.csdn.net/renyuanxingxing/article/details/99672234
    // 函数的一种，对一个float型的数进行四舍五入。
    int x = cvRound(kp.pt.x);
    int y = cvRound(kp.pt.y);

    ushort d = depth_img_.ptr<ushort>(y)[x];
    if(d != 0) {
        return float(d)/depth_scale_;
    } else {
        /**
        // check the nearby points
        int dx[4] = {-1, 0, 1, 0};
        int dy[4] = {0, -1, 0, 1};
        for( int i=0; i<4; i++ ) {
            d = depth_img_.ptr<ushort>( y+dy[i] )[x+dx[i]];
            if(d != 0) {
                return float(d)/depth_scale_;
            }
        }
        **/
    }

    return -1.0;
}


// --------------------------------------------------------------------------------------------------------------
FrameResult::FrameResult(Frame::Ptr frame_ptr) {
    id_ = frame_ptr->id_;
    // is_keyframe_ = frame_ptr->is_keyframe_;
    time_stamp_ = frame_ptr->time_stamp_;

    // SetRelativePose(frame_ptr->relative_pose_);

    // reference_keyframe_ptr_ = frame_ptr->reference_keyframe_ptr_;
}

}  // // namespace sg_slam
