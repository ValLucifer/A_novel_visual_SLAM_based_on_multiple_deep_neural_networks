//
//  Created by Lucifer on 2022/7/29.
//

#ifndef _SG_SLAM_ALGORITHM_H_
#define _SG_SLAM_ALGORITHM_H_

#include <vector>
#include "sg_slam/common_include.h"

// #include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

namespace sg_slam {

/**
 * @brief linear triangulation with SVD (VIO第六章，多视图几何)
 * 一个地图点，在多个视图的对应归一化平面坐标点.
 * @param poses   poses of cameras (Tcw)
 * @param points  points in normalized plane (of camera)
 * @param pt_world triangulated point in the world (R^(3x1)) (真正的输出)
 * @return true if success
 */
// points 自己改成引用.
inline bool triangulation(const std::vector<SE3> &poses, const std::vector<Vec3> &points, Vec3 &pt_world) {
    MatXX A(2*poses.size(), 4);  // (2nx4)
    VecX b(2*poses.size()); // (2nx1)
    b.setZero();  // 可用于迭代时的初始化, 这里是因为公式中b=0.
    for(size_t i=0; i<poses.size(); i++) {
        Mat34 m = poses[i].matrix3x4();
        // https://blog.csdn.net/Darlingqiang/article/details/124849257
        A.block<1, 4>(2*i, 0) = points[i][0] * m.row(2) - m.row(0);
        A.block<1, 4>(2*i+1, 0) = points[i][1] * m.row(2) - m.row(1);
    }

    // https://zhuanlan.zhihu.com/p/446238693
    // Thin表示奇异值不为零(U, V)
    auto svd = A.bdcSvd( Eigen::ComputeThinU | Eigen::ComputeThinV );

    // N.B.: My add (20220829)
    // N.B.: 太小会导致三角化的地图点坐标大得离谱.
    if(svd.matrixV()(3, 3) < 1e-4) {
        return false;
    }

    // V(矩阵形状4x4), 取最小奇异值对应的特征向量(即最后一列), 转换为齐次坐标，再取前三维(即一般坐标)
    // https://www.cnblogs.com/fuzhuoxin/p/12600532.html.
    pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();

    // ORB 实验得出的条件，其他特征不一定适用
    if( svd.singularValues()[3] / svd.singularValues()[2] < 0.1 ) {
        // N.B.: 解质量较好在使用, 此判断结果根据VIO第六章实验得出 (最小奇异值要足够小)
        return true;
    }

    // 解质量不好放弃
    return false;

    /**
    return true;
    **/
}

/**
 * @brief cv::Point2f convert to Vec2
 * 
 */
inline Vec2 toVec2( const cv::Point2f p ) {
    return Vec2(p.x, p.y);
}

/**
 * @brief cv::Point2f and keyR_u convert to Vec3
 * 
 */
inline Vec3 toVec3( const cv::Point2f p,  const float kpR_u) {
    return Vec3(p.x, p.y, kpR_u);
}

}  // namespace sg_slam

#endif  // _SG_SLAM_ALGORITHM_H_

