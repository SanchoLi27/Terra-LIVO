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

#ifndef LIVO_POINT_H_
#define LIVO_POINT_H_

#include <boost/noncopyable.hpp>
#include "common_lib.h"
#include "frame.h"

class Feature;

/// A visual map point on the surface of the scene.
class VisualPoint : boost::noncopyable
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Vector3d pos_;                //!< 3d pos of the point in the world coordinate frame.
  Vector3d normal_;             //!< Surface normal at point.
  Matrix3d normal_information_; //!< Inverse covariance matrix of normal estimation.
  Vector3d previous_normal_;    //!< Last updated normal vector.
  list<Feature *> obs_;         //!< Reference patches which observe the point.
  Eigen::Matrix3d covariance_;  //!< Covariance of the point.
  bool is_converged_;           //!< True if the point is converged.
  bool is_normal_initialized_;  //!< True if the normal is initialized.
  bool has_ref_patch_;          //!< True if the point has a reference patch.
  Feature *ref_patch;           //!< Reference patch of the point.

  VisualPoint(const Vector3d &pos);
  ~VisualPoint();
  void findMinScoreFeature(const Vector3d &framepos, Feature *&ftr) const;
  void deleteNonRefPatchFeatures();
  void deleteFeatureRef(Feature *ftr);
  void addFrameRef(Feature *ftr);
  bool getCloseViewObs(const Vector3d &pos, Feature *&obs, const Vector2d &cur_px) const;
};

#endif // LIVO_POINT_H_
