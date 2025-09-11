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

#include <boost/bind/bind.hpp>
#include "feature.h"
#include "frame.h"
#include "visual_point.h"
#include <stdexcept>
#include <vikit/math_utils.h>
#include <vikit/performance_monitor.h>
#include <vikit/vision.h>

int Frame::frame_counter_ = 0;

Frame::Frame(vk::AbstractCamera *cam, const cv::Mat &img)
    : id_(frame_counter_++), 
      cam_(cam)
{
  initFrame(img);
}

Frame::~Frame()
{
  std::for_each(fts_.begin(), fts_.end(), [&](Feature *i) { delete i; });
}

void Frame::initFrame(const cv::Mat &img)
{
  if (img.empty()) { throw std::runtime_error("Frame: provided image is empty"); }

  if (img.cols != cam_->width() || img.rows != cam_->height())
  {
    throw std::runtime_error("Frame: provided image has not the same size as the camera model");
  }

  if (img.type() != CV_8UC1) { throw std::runtime_error("Frame: provided image is not grayscale"); }

  img_ = img;
}

/// Utility functions for the Frame class
namespace frame_utils
{

void createImgPyramid(const cv::Mat &img_level_0, int n_levels, ImgPyr &pyr)
{
  pyr.resize(n_levels);
  pyr[0] = img_level_0;
  for (int i = 1; i < n_levels; ++i)
  {
    pyr[i] = cv::Mat(pyr[i - 1].rows / 2, pyr[i - 1].cols / 2, CV_8U);
    vk::halfSample(pyr[i - 1], pyr[i]);
  }
}

} // namespace frame_utils
