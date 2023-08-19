//
//  Created by Lucifer on 2022/7/11.
//

#ifndef _SG_SLAM_KITTI_DATASET_H_
#define _SG_SLAM_KITTI_DATASET_H_

#include <Eigen/Core>  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW, 参见sg_slam/frame.h
#include <memory>  // 智能指针
#include <string>
#include <vector>

#include "sg_slam/frame.h" 
#include "sg_slam/camera.h"

namespace sg_slam {

// 20221009 加
/**
 * @brief 数据集基类
 * 
 */
class Dataset
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  // 参见sg_slam/frame.h
    using this_type = Dataset;  // 本类别名
    using Ptr = std::shared_ptr<Dataset>;  // 本类共享/弱指针别名

    Dataset() {

    }

    /**
    Dataset(const std::string &dataset_path) {

    }
    **/

    ~Dataset() = default;

    bool virtual Init(double scale) {
        return false;
    }

    Frame::Ptr virtual NextFrame() {
        return nullptr;
    }

    Camera::Ptr virtual getCamera(int camera_id) const {
        return nullptr;
    }

    Camera::Ptr virtual getCamera() const {
        return nullptr;
    }


};  // class Dataset

/**
 * @brief KITTI数据集读取
 * 构造时传入数据集路径, 配置文件的dataset_dir为数据集路径
 * Init执行之后，才可获得相机和下一帧图像
 * 
 */
class KITTI_Dataset : public Dataset
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  // 参见sg_slam/frame.h
    using this_type = KITTI_Dataset;  // 本类别名
    using Ptr = std::shared_ptr<KITTI_Dataset>;  // 本类共享/弱指针别名

    int current_image_index_ = 0;

    KITTI_Dataset(const std::string &dataset_path);
   
    ~KITTI_Dataset() = default;

    // 初始化, 输入图像缩放比例， 返回是否初始化成功. N.B.: 自改
    bool Init(double scale = 1.0);

    // create and return the next frame containing stereo images
    // 仅填充图像数据(非引用, 但是是浅拷贝)和帧ID没有提取特征
    Frame::Ptr NextFrame();

    // get camera by id
    Camera::Ptr getCamera(int camera_id) const {
        // https://blog.csdn.net/zpznba/article/details/89669888
        return cameras_.at(camera_id);
    }

private:
    std::string dataset_path_;
    // int current_image_index_ = 0;
    double m_scale_ = 1.0;

    std::vector< Camera::Ptr > cameras_;

    std::vector<double> vTimestamps_;
    // std::vector<long double> vTimestamps_;

};  // class KITTI_Dataset

}  // namespace sg_slam

#endif  // _SG_SLAM_KITTI_DATASET_H_
