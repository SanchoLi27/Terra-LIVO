/*
 * TERRA-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry
 * Copyright (C) 2025 SanchoLi27 <hdalhd1104@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef LIVO_FEATURE_H_
#define LIVO_FEATURE_H_

#include "visual_point.h"

// A salient image region that is tracked across frames.
struct Feature
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum FeatureType
  {
    CORNER,
    EDGELET
  };
  int id_;
  FeatureType type_;     //!< Type can be corner or edgelet.
  cv::Mat img_;          //!< Image associated with the patch feature
  Vector2d px_;          //!< Coordinates in pixels on pyramid level 0.
  Vector3d f_;           //!< Unit-bearing vector of the patch feature.
  int level_;            //!< Image pyramid level where patch feature was extracted.
  VisualPoint *point_;   //!< Pointer to 3D point which corresponds to the patch feature.
  Vector2d grad_;        //!< Dominant gradient direction for edglets, normalized.
  SE3<double> T_f_w_;            //!< Pose of the frame where the patch feature was extracted.
  float *patch_;         //!< Pointer to the image patch data.
  float score_;          //!< Score of the patch feature.
  float mean_;           //!< Mean intensity of the image patch feature, used for normalization.
  double inv_expo_time_; //!< Inverse exposure time of the image where the patch feature was extracted.
  
  Feature(VisualPoint *_point, float *_patch, const Vector2d &_px, const Vector3d &_f, const SE3<double> &_T_f_w, int _level)
      : type_(CORNER), px_(_px), f_(_f), T_f_w_(_T_f_w), mean_(0), score_(0), level_(_level), patch_(_patch), point_(_point)
  {
  }

  inline Vector3d pos() const { return T_f_w_.inverse().translation(); }
  
  ~Feature()
  {
    // ROS_WARN("The feature %d has been destructed.", id_);
    delete[] patch_;
  }
};

#endif // LIVO_FEATURE_H_
