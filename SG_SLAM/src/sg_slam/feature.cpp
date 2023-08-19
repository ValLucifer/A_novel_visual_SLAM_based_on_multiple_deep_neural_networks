//
//  Created by Lucifer on 2022/7/6.
//

#include "sg_slam/feature.h"

namespace sg_slam {

Feature::Feature(std::shared_ptr<KeyFrame> keyframe, const cv::KeyPoint &kp)
        : KeyFrame_(keyframe), position_(kp) {
    // static unsigned long feature_factory_id = 0;
    // id_ = feature_factory_id++;
}

}  // namespace sg_slam
