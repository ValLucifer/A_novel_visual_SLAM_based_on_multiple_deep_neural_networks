//
//  Created by Lucifer on 2022/7/5.
//

#ifndef _SG_SLAM_COMMON_INCLUDE_H_
#define _SG_SLAM_COMMON_INCLUDE_H_

// Sophus
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <sophus/sim3.hpp>

// // typedefs for Sophus
using SE3 = Sophus::SE3d;
using SO3 = Sophus::SO3d;
using Sim3 = Sophus::Sim3d;


// Eigen
#include <Eigen/Core>

// // typedefs for eigen
// // double vectors
using Vec3 = Eigen::Matrix< double, 3, 1 >;
using Vec2 = Eigen::Matrix< double, 2, 1 >;
using VecX = Eigen::Matrix< double, Eigen::Dynamic, 1 >;
using Vec6 = Eigen::Matrix< double, 6, 1 >;

// // double matricies
using Mat22 = Eigen::Matrix< double, 2, 2 >;
using Mat33 = Eigen::Matrix< double, 3, 3 >;
using Mat34 = Eigen::Matrix< double, 3, 4 >;
using Mat66 = Eigen::Matrix< double, 6, 6 >;
using MatXX = Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic >;


#endif  // _SG_SLAM_COMMON_INCLUDE_H_
