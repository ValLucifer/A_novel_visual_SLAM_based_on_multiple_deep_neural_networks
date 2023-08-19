//
//  Created by Lucifer on 2022/7/29.
//

#ifndef _SG_SLAM_G2O_TYPES_H_
#define _SG_SLAM_G2O_TYPES_H_

#include <Eigen/Core>  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#include <iostream>

#include "sg_slam/common_include.h"
#include "sg_slam/algorithm.h"

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>

// #include <g2o/types/sba/edge_project_stereo_xyz.h>

namespace sg_slam {

/* vertices and edges used in g2o BA */

// --------------------------------------------------------------------------------------------------------------
/**
 * @brief vertices of poses (位姿顶点)
 * 
 */
class VertexPose : public g2o::BaseVertex<6, SE3> 
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  // 参见 sg_slam/frame.h

    // 重置
    // https://blog.csdn.net/clygo9/article/details/114342496
    virtual void setToOriginImpl() override {
        // N.B.: 自改 
        _estimate = SE3();
        // Vec6 zero_eigen = Vec6::Zero();
        // _estimate = SE3().exp(zero_eigen);
    }

    // left multiplication on SE3
    // 参见 virtual void setToOriginImpl() override
    // 更新
    virtual void oplusImpl(const double *update) override {
        Vec6 update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = SE3::exp(update_eigen) * _estimate;
    }

    // 存盘和读盘: 留空(不设计)
    virtual bool read(std::istream &in) override { return true; }
    // N.B.: 被覆盖函数有 const , 覆盖它的函数也要有，即参数表(参数类型,数量,顺序)相同，函数结构必须一致。
    // N.B.: http://www.360doc.com/content/18/0530/09/54097382_758156758.shtml
    virtual bool write(std::ostream &out)  const override { return true; }
};  // class VertexPose

/**
 * @brief vertices of points (路标顶点)
 * 
 */
class VertexXYZ : public g2o::BaseVertex<3, Vec3> 
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  // 参见 sg_slam/frame.h

    // 重置
    virtual void setToOriginImpl() override {
        _estimate = Vec3::Zero();
    }

    // 更新
    virtual void oplusImpl(const double *update) override {
        _estimate[0] += update[0];
        _estimate[1] += update[1];
        _estimate[2] += update[2];
    }

    virtual bool read(std::istream &in) override { return true; }
    virtual bool write(std::ostream &out)  const override { return true; }
};  // class VertexXYZ


// --------------------------------------------------------------------------------------------------------------
/**
 * @brief unary edge used to estimate only poses (前端使用/回环计算当前处理帧时使用)
 * g2o::BaseUnaryEdge(误差模型) 模板参数：边/观测值(_measurement)/误差(_error)类型的维度(Eigen(列)向量为对应维度， 标量为维度1)， 边/观测值/误差类型(观测)，连接顶点类型(优化变量)
 * 7.7.3(十四讲) 最小化重投影误差求解PnP.
 * 
 */
class EdgeProjectionPoseOnly : public g2o::BaseUnaryEdge<2, Vec2, VertexPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  // 参见 sg_slam/frame.h

    EdgeProjectionPoseOnly(const Vec3 &pos, const Mat33 &K) 
    : _pos3d(pos), _K(K) {

    }

    // 计算模型误差
    virtual void computeError() override {
        const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);  // 一元边只有一个顶点.
        SE3 T = v->estimate();  // 在 VertexPose 继承时传入了输出类型，c++泛函/模板编程.
        // N.B.: 这里无法检测Eigen维度错误, 只能在运行中检测，同一模板类的实例都是一个类型， 编译认为Vec2 = Vec3.
        // se3.hpp 第306行, 重载了 operator*.
        Vec3 pose_pixel = _K * (T * _pos3d);  // 十四讲，7.7.3, 公式(7.38)-(7.41)
        pose_pixel /= pose_pixel[2];  // 归一化计算出像素坐标. // 3x1
        // 参见 sg_slam/algorithm.h triangulation 中的 .head<3>().
        _error = _measurement - pose_pixel.head<2>();  // 2x1
    }

    // 计算雅可比 (g2o的使用惯例)
    virtual void linearizeOplus() override {
        const VertexPose *v = static_cast<VertexPose *>(_vertices[0]); // *v 为 v指针指向的内容.
        SE3 T = v->estimate();
        Vec3 pos_cam = T * _pos3d;  // T 为本次迭代的处值，即十四讲，公式(7.37)中的x.
        double fx = _K(0, 0);
        double fy = _K(1, 1);
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Zinv = 1.0 / (Z + 1e-18);  // N.B.: 工程技巧, 防止分母为0.
        double Zinv2 = Zinv * Zinv;

        // 十四讲，公式(7.46)
        _jacobianOplusXi << -fx * Zinv, 0, fx * X * Zinv2, fx * X * Y * Zinv2, -fx - fx*X*X*Zinv2, fx * Y * Zinv, 
                            0, -fy * Zinv, fy * Y * Zinv2, fy + fy*Y*Y*Zinv2, -fy * X * Y * Zinv2, -fy * X * Zinv;
    }

    virtual bool read(std::istream &in) override { return true; }
    virtual bool write(std::ostream &out)  const override { return true; }

private:
    Vec3 _pos3d;  // 点在世界坐标系下的坐标向量
    Mat33 _K;  // 对应相机的内参

};  // class EdgeProjectionPoseOnly

/**
 * @brief binary edge of poses and points (带有地图和位姿的二元边) (后端使用)
 * D: dim(误差维度)(向量), E: edge_type(测量类型), 待优化节点类型0, 待优化节点类型1
 * 9.2.1(十四讲) BA
 */
class EdgeProjection : public g2o::BaseBinaryEdge<2, Vec2, VertexPose, VertexXYZ> 
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  // 参见 sg_slam/frame.h

    /// 构造时传入相机内外参
    EdgeProjection(const Mat33 &K, const SE3 &cam_ext)
        : _K(K) {
        _cam_ext = cam_ext;
    }

    // 计算模型误差
    virtual void computeError() override {
        const VertexPose *v0 = static_cast<VertexPose *>(_vertices[0]);
        const VertexXYZ *v1 = static_cast<VertexXYZ *>(_vertices[1]);
        SE3 T = v0->estimate();

        // 十四讲，9.2.1, 公式(9.36)-(9.31), 但不考虑畸变.
        Vec3 pos_pixel = _K * (_cam_ext * (T * v1->estimate())); 
        pos_pixel /= pos_pixel[2];
        _error = _measurement - pos_pixel.head<2>();  // 十四讲，9.2.1, 公式(9.41)
    }

    // 计算雅可比 (g2o的使用惯例)
    virtual void linearizeOplus() override {
        const VertexPose *v0 = static_cast<VertexPose *>(_vertices[0]);
        const VertexXYZ *v1 = static_cast<VertexXYZ *>(_vertices[1]);
        SE3 T = v0->estimate();
        Vec3 pw = v1->estimate();
        Vec3 pos_cam = _cam_ext * T * pw;
        double fx = _K(0, 0);
        double fy = _K(1, 1);
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Zinv = 1.0 / (Z + 1e-18);
        double Zinv2 = Zinv * Zinv;

        // 十四讲，公式(7.46)
        _jacobianOplusXi << -fx * Zinv, 0, fx * X * Zinv2, fx * X * Y * Zinv2, -fx - fx*X*X*Zinv2, fx * Y * Zinv,
                            0, -fy * Zinv, fy * Y * Zinv2, fy + fy*Y*Y*Zinv2, -fy * X * Y * Zinv2, -fy * X * Zinv;
        
        // 十四讲，公式(7.48)
        _jacobianOplusXj = _jacobianOplusXi.block<2, 3>(0, 0) * _cam_ext.rotationMatrix() * T.rotationMatrix();
    }

    virtual bool read(std::istream &in) override { return true; }
    virtual bool write(std::ostream &out)  const override { return true; }

private:
    Mat33 _K;  // 内参.
    SE3 _cam_ext;  // 外参(相对于左相机), (从左相机变换到对应相机, (此工程仅在左相机做优化，这一项应该恒为I(4x4)))

};  // class EdgeProjection

// --------------------------------------------------------------------------------------------------------------
/**
 * @brief binary edge of poses(位姿间的二元边) (回环修正使用使用)
 * N.B.: 根据清华工程设计(照抄)
 * 
 */
class EdgePoseGraph : public g2o::BaseBinaryEdge<6, SE3, VertexPose, VertexPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  // 参见 sg_slam/frame.h

    // 计算模型误差
    virtual void computeError() override {
        const VertexPose *vertex0 = static_cast<VertexPose *>(_vertices[0]);  
        const VertexPose *vertex1 = static_cast<VertexPose *>(_vertices[1]);
        SE3 v0 = vertex0->estimate();
        SE3 v1 = vertex1->estimate();
        // N.B.: 测量为T01(SE3)
        _error = (_measurement.inverse() * v0 * v1.inverse()).log();  
    }

    virtual bool read(std::istream &in) override { return true; }
    virtual bool write(std::ostream &out) const override { return true; }

};  // class EdgePoseGraph

}  // namespace sg_slam

#endif  // _SG_SLAM_G2O_TYPES_H_
