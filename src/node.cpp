/*
 * Copyright 2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Authors: Yukihiro Saito
 *
 */

#include <ros/ros.h>

#include <math.h> // distance 계산
#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_core/geometry/Point.h>
#include <lanelet2_extension/utility/message_conversion.h>
#include <lanelet2_extension/utility/utilities.h>
#include <lanelet2_extension/visualization/visualization.h>
#include <lanelet2_projection/UTM.h>
#include <lanelet2_routing/RoutingGraphContainer.h>
#include <lanelet2_traffic_rules/TrafficRulesFactory.h>

#include <autoware_perception_msgs/TrafficLightRoi.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Transform.h>
#include <traffic_light_map_based_detector/node.hpp>
#define EIGEN_MPL2_ONLY
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"

namespace traffic_light
{
MapBasedDetector::MapBasedDetector() : nh_(""), pnh_("~"), tf_listener_(tf_buffer_)
{
  map_sub_ = pnh_.subscribe("input/vector_map", 1, &MapBasedDetector::mapCallback, this);
  camera_info_sub_ =
    pnh_.subscribe("input/camera_info", 1, &MapBasedDetector::cameraInfoCallback, this);
  route_sub_ = pnh_.subscribe("input/route", 1, &MapBasedDetector::routeCallback, this);
  roi_pub_ = pnh_.advertise<autoware_perception_msgs::TrafficLightRoiArray>("output/rois", 1);
  viz_pub_ = pnh_.advertise<visualization_msgs::MarkerArray>("debug/markers", 1);
  pnh_.getParam("max_vibration_pitch", config_.max_vibration_pitch);
  pnh_.getParam("max_vibration_yaw", config_.max_vibration_yaw);
  pnh_.getParam("max_vibration_height", config_.max_vibration_height);
  pnh_.getParam("max_vibration_width", config_.max_vibration_width);
  pnh_.getParam("max_vibration_depth", config_.max_vibration_depth);
  pnh_.getParam("max_distance", config_.max_distance);
  pnh_.getParam("min_distance", config_.min_distance);
  pnh_.getParam("car_offset", config_.car_offset);
  pnh_.getParam("car_offset_bool", config_.car_offset_bool);
}

void MapBasedDetector::cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr & input_msg)
{
  camera_info_ptr_ = input_msg;
  if (lanelet_map_ptr_ == nullptr || camera_info_ptr_ == nullptr) return;

  autoware_perception_msgs::TrafficLightRoiArray output_msg;
  output_msg.header = camera_info_ptr_->header;
  geometry_msgs::PoseStamped camera_pose_stamped;
  try {
    geometry_msgs::TransformStamped transform;
    transform = tf_buffer_.lookupTransform(
      "map", input_msg->header.frame_id, input_msg->header.stamp, ros::Duration(0.2));
    camera_pose_stamped.header = input_msg->header;
    camera_pose_stamped.pose.position.x = transform.transform.translation.x;
    camera_pose_stamped.pose.position.y = transform.transform.translation.y;
    camera_pose_stamped.pose.position.z = transform.transform.translation.z;
    camera_pose_stamped.pose.orientation.x = transform.transform.rotation.x;
    camera_pose_stamped.pose.orientation.y = transform.transform.rotation.y;
    camera_pose_stamped.pose.orientation.z = transform.transform.rotation.z;
    camera_pose_stamped.pose.orientation.w = transform.transform.rotation.w;
  } catch (tf2::TransformException & ex) {
    ROS_WARN_THROTTLE(5, "cannot get transform from map frame to camera frame");
    return;
  }

  const sensor_msgs::CameraInfo & camera_info = *input_msg;
  std::vector<tls_info> visible_traffic_lights;
  if (route_traffic_lights_ptr_ != nullptr)
    isInVisibility(
      *route_traffic_lights_ptr_, camera_pose_stamped.pose, camera_info, visible_traffic_lights);
  else if (all_traffic_lights_ptr_ != nullptr)
    isInVisibility(
      *all_traffic_lights_ptr_, camera_pose_stamped.pose, camera_info, visible_traffic_lights);
  else
    return;

  for (const auto & traffic_light : visible_traffic_lights) {
    autoware_perception_msgs::TrafficLightRoi tl_roi;
    if (!getTrafficLightRoi(camera_pose_stamped.pose, camera_info, traffic_light, config_, tl_roi))
      continue;

    output_msg.rois.push_back(tl_roi);
  }
  roi_pub_.publish(output_msg);
  publishVisibleTrafficLights(camera_pose_stamped, visible_traffic_lights, viz_pub_);
}

bool MapBasedDetector::getTrafficLightRoi(
  const geometry_msgs::Pose & camera_pose, const sensor_msgs::CameraInfo & camera_info,
  const tls_info traffic_light, const Config & config, autoware_perception_msgs::TrafficLightRoi & tl_roi) {
  cv::Mat cameraMat(3, 3, CV_64FC1, (void *) camera_info.K.data());
  cv::Mat dist(5, 1, CV_64FC1, (void *) camera_info.D.data());
  std::vector<cv::Point2f> undistPts, distPts; // undistorted point, distorted point
  std::vector<cv::Point3f>  xyz; // 3차원 point

  const double tl_height = traffic_light.tls.attributeOr("height", 0.0);
  const double & fx = camera_info.K[(0 * 3) + 0];
  const double & fy = camera_info.K[(1 * 3) + 1];
  const double & cx = camera_info.K[(0 * 3) + 2];
  const double & cy = camera_info.K[(1 * 3) + 2];
  const auto & tl_left_down_point = traffic_light.tls.front();
  const auto & tl_right_down_point = traffic_light.tls.back();
  tf2::Transform tf_map2camera(
    tf2::Quaternion(
      camera_pose.orientation.x, camera_pose.orientation.y, camera_pose.orientation.z,
      camera_pose.orientation.w),
    tf2::Vector3(camera_pose.position.x, camera_pose.position.y, camera_pose.position.z));
  // id
  tl_roi.id = traffic_light.tls.id();

  // 카메라와 신호등의 xy축 간 거리 계산(Top view상 거리)
  geometry_msgs::Point tl_central_point;
  tl_central_point.x = (tl_right_down_point.x() + tl_left_down_point.x()) / 2.0;
  tl_central_point.y = (tl_right_down_point.y() + tl_left_down_point.y()) / 2.0;
  tl_central_point.z = (tl_right_down_point.z() + tl_left_down_point.z() + tl_height) / 2.0;
  tl_roi.distance =  sqrt((tl_central_point.x - camera_pose.position.x) * (tl_central_point.x - camera_pose.position.x) + (tl_central_point.y - camera_pose.position.y) * (tl_central_point.y - camera_pose.position.y));
  tl_roi.dist_cam2stop =  (float)traffic_light.stop_line_distance;

  // for roi.x_offset and roi.y_offset
  {
    tf2::Transform tf_map2tl(
      tf2::Quaternion(0, 0, 0, 1),
      tf2::Vector3(
        tl_left_down_point.x(), tl_left_down_point.y(), tl_left_down_point.z() + tl_height));
    tf2::Transform tf_camera2tl;
    tf_camera2tl = tf_map2camera.inverse() * tf_map2tl;

    const double & camera_x =
      tf_camera2tl.getOrigin().x() -
      (std::sin(config.max_vibration_yaw / 2.0) * tf_camera2tl.getOrigin().z()) -
      config.max_vibration_width / 2.0;
    const double & camera_y =
      tf_camera2tl.getOrigin().y() -
      (std::sin(config.max_vibration_pitch / 2.0) * tf_camera2tl.getOrigin().z()) -
      config.max_vibration_height / 2.0;
    const double & camera_z = tf_camera2tl.getOrigin().z() - config.max_vibration_depth / 2.0;
    if (camera_z <= 0.0) return false;
    const double image_u = (fx * camera_x + cx * camera_z) / camera_z;
    const double image_v = (fy * camera_y + cy * camera_z) / camera_z;

    
    undistPts.push_back(cv::Point2f(image_u, image_v)); // 계산된 undistorted point 저장(Image plane)

    // ref. https://answers.opencv.org/question/98929/trying-to-re-distort-image-points-using-projectpoints/?sort=latest
    // 계산된 undistorted point 저장(Normalizaed image plane)
    cv::undistortPoints(undistPts, undistPts, cameraMat, cv::Mat());
    for (cv::Point2f p : undistPts) xyz.push_back(cv::Point3f(p.x, p.y, 1)); 

    // 재투영을 통한 distorted point 계산
    cv::projectPoints(xyz, cv::Mat::zeros(3, 1, CV_64FC1), cv::Mat::zeros(3, 1, CV_64FC1), cameraMat, dist, distPts);

    //offset이 0보다 작을경우 스킵
    if (distPts[0].x < 0 || distPts[0].y < 0) return false;

    // width 혹은 height 보다 작고, 0보다 큰 값으로 예외처리
    tl_roi.roi.x_offset = int32_t(std::max(std::min((double)distPts[0].x, (double)camera_info.width-1), 0.0));
    tl_roi.roi.y_offset = int32_t(std::max(std::min((double)distPts[0].y, (double)camera_info.height-1), 0.0));
  }

  // for roi.width and roi.height
  {
    tf2::Transform tf_map2tl(
      tf2::Quaternion(0, 0, 0, 1),
      tf2::Vector3(tl_right_down_point.x(), tl_right_down_point.y(), tl_right_down_point.z()));
    tf2::Transform tf_camera2tl;
    tf_camera2tl = tf_map2camera.inverse() * tf_map2tl;

    const double & camera_x =
      tf_camera2tl.getOrigin().x() +
      (std::sin(config.max_vibration_yaw / 2.0) * tf_camera2tl.getOrigin().z()) +
      config.max_vibration_width / 2.0;
    const double & camera_y =
      tf_camera2tl.getOrigin().y() +
      (std::sin(config.max_vibration_pitch / 2.0) * tf_camera2tl.getOrigin().z()) +
      config.max_vibration_height / 2.0;
    const double & camera_z = tf_camera2tl.getOrigin().z() - config.max_vibration_depth / 2.0;
    if (camera_z <= 0.0) return false;
    const double image_u = (fx * camera_x + cx * camera_z) / camera_z;
    const double image_v = (fy * camera_y + cy * camera_z) / camera_z;

    // 새로운 계산을 위해 vector 초기화
    undistPts.pop_back(); 
    xyz.pop_back();

    undistPts.push_back(cv::Point2f(image_u, image_v)); // 계산된 undistorted point 저장(Image plane)

    // 계산된 undistorted point 저장(camera plane)
    cv::undistortPoints(undistPts, undistPts, cameraMat, cv::Mat());
    for (cv::Point2f p : undistPts) xyz.push_back(cv::Point3f(p.x, p.y, 1));

    // 재투영을 통한 distorted point 계산
    cv::projectPoints(xyz, cv::Mat::zeros(3, 1, CV_64FC1), cv::Mat::zeros(3, 1, CV_64FC1), cameraMat, dist, distPts);

    // width 혹은 height 보다 작고, 0보다 큰 값으로 예외처리
    tl_roi.roi.width = int32_t(std::max(std::min((double)distPts[0].x - (double)tl_roi.roi.x_offset, (double)camera_info.width-1), 0.0));
    tl_roi.roi.height = int32_t(std::max(std::min((double)distPts[0].y - (double)tl_roi.roi.y_offset, (double)camera_info.height-1), 0.0));
    if (tl_roi.roi.width <= 0 || tl_roi.roi.height <= 0) return false; // width, height이 0과 같거나 작은 경우 예외처리

    // roi가 이미지 경계를 넘어가는 부분 보정
    if(tl_roi.roi.x_offset + tl_roi.roi.width > camera_info.width){
      int over_size_width = tl_roi.roi.x_offset + tl_roi.roi.width - camera_info.width;
      tl_roi.roi.width = tl_roi.roi.width - over_size_width - 1;
    }
    if(tl_roi.roi.y_offset + tl_roi.roi.height > camera_info.height){
      int over_size_height = tl_roi.roi.y_offset + tl_roi.roi.height - camera_info.height;
      tl_roi.roi.height = tl_roi.roi.height - over_size_height - 1;
    }
  }
  return true;
}

void MapBasedDetector::mapCallback(const autoware_lanelet2_msgs::MapBin & input_msg) {
  lanelet_map_ptr_ = std::make_shared<lanelet::LaneletMap>();
  lanelet::utils::conversion::fromBinMsg(
    input_msg, lanelet_map_ptr_, &traffic_rules_ptr_, &routing_graph_ptr_);
  lanelet::ConstLanelets all_lanelets = lanelet::utils::query::laneletLayer(lanelet_map_ptr_);
  std::vector<lanelet::AutowareTrafficLightConstPtr> all_lanelet_traffic_lights =
    lanelet::utils::query::autowareTrafficLights(all_lanelets);
  all_traffic_lights_ptr_ = std::make_shared<MapBasedDetector::TrafficLightSet>(); // 신호등 1대 단위로 구분된 정보

  // 모든 lanelet 정보 중 정지선/신호등 단위로 나뉜 Iteration에서 정보 추출
  for (auto tl_itr = all_lanelet_traffic_lights.begin(); tl_itr != all_lanelet_traffic_lights.end();
       ++tl_itr) {
    tls_info traffics_info; // Stop line : Stop line 정보가 추가된 struct
    lanelet::AutowareTrafficLightConstPtr tl = *tl_itr;

    /////////////////////////////  Stop line : Stop line 정보 추출  /////////////////////////////
    auto stop_line = *(tl->stopLine());
    if(stop_line.empty()) continue;
    traffics_info.stop_line_id = stop_line.id(); // ID 추출

    traffics_info.stop_line_left.x = stop_line.front().x();
    traffics_info.stop_line_left.y = stop_line.front().y();
    traffics_info.stop_line_left.z = stop_line.front().z();

    traffics_info.stop_line_right.x = stop_line.back().x();
    traffics_info.stop_line_right.y = stop_line.back().y();
    traffics_info.stop_line_right.z = stop_line.back().z();


    ///////////////////////////////////////////////////////////////////////////////////////////
    auto lights = tl->trafficLights();
    for (auto lsp : lights) {
      if (!lsp.isLineString())  // traffic lights must be linestrings
        continue;

      traffics_info.tls = static_cast<lanelet::ConstLineString3d>(lsp); // Stop line : Traffic light 1대에 대한 정보 추가
      all_traffic_lights_ptr_->insert(static_cast<tls_info>(traffics_info)); // 대응되는 정지선 ID와 위치정보 추가
    }
  }
}

void MapBasedDetector::routeCallback(const autoware_planning_msgs::Route::ConstPtr & input_msg)
{
  if (lanelet_map_ptr_ == nullptr) {
    ROS_WARN("cannot set traffic light in route because don't receive map");
    return;
  }
  lanelet::ConstLanelets route_lanelets;
  for (const auto & route_section : input_msg->route_sections) {
    for (const auto & lane_id : route_section.lane_ids) {
      route_lanelets.push_back(lanelet_map_ptr_->laneletLayer.get(lane_id));
    }
  }
  std::vector<lanelet::AutowareTrafficLightConstPtr> route_lanelet_traffic_lights =
    lanelet::utils::query::autowareTrafficLights(route_lanelets);
  route_traffic_lights_ptr_ = std::make_shared<MapBasedDetector::TrafficLightSet>();
  for (auto tl_itr = route_lanelet_traffic_lights.begin();
       tl_itr != route_lanelet_traffic_lights.end(); ++tl_itr) {
    tls_info traffics_info; // Stop line : Stop line 정보가 추가된 struct
    lanelet::AutowareTrafficLightConstPtr tl = *tl_itr;

    /////////////////////////////  Stop line : Stop line 정보 추출  /////////////////////////////
    auto stop_line = *(tl->stopLine());
    if(stop_line.empty()) continue;
    traffics_info.stop_line_id = stop_line.id(); // ID 추출
    traffics_info.stop_line_left.x = stop_line.front().x();
    traffics_info.stop_line_left.y = stop_line.front().y();
    traffics_info.stop_line_left.z = stop_line.front().z();

    traffics_info.stop_line_right.x = stop_line.back().x();
    traffics_info.stop_line_right.y = stop_line.back().y();
    traffics_info.stop_line_right.z = stop_line.back().z();
    ///////////////////////////////////////////////////////////////////////////////////////////
    auto lights = tl->trafficLights();
    for (auto lsp : lights) {
      if (!lsp.isLineString())  // traffic lights must be linestrings
        continue;

      traffics_info.tls = static_cast<lanelet::ConstLineString3d>(lsp); // Stop line : Traffic light 1대에 대한 정보 추가
      route_traffic_lights_ptr_->insert(static_cast<tls_info>(traffics_info));
    }
  }
}

void MapBasedDetector::isInVisibility(
  const MapBasedDetector::TrafficLightSet & all_traffic_lights,
  const geometry_msgs::Pose & camera_pose, const sensor_msgs::CameraInfo & camera_info,
  std::vector<tls_info> & visible_traffic_lights)
{
  // check distance range
  //map -> base_link로 좌표 변환 (정지선 좌, 우)
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tfListener(tf_buffer_);
  geometry_msgs::TransformStamped tfs;
  try {  
    tfs = tf_buffer_.lookupTransform("base_link", "map", ros::Time(0), ros::Duration(1.0));
  }
  catch (tf2::TransformException & ex) {
    ROS_WARN_THROTTLE(5, "cannot get transform");
  }
  tf2::Transform tf_base_link2map(tf2::Quaternion(tfs.transform.rotation.x, tfs.transform.rotation.y, tfs.transform.rotation.z,tfs.transform.rotation.w),
      tf2::Vector3(tfs.transform.translation.x - config_.car_offset, tfs.transform.translation.y, tfs.transform.translation.z));
  tf2::Transform tf_base_link2stop_left;
  tf2::Transform tf_base_link2stop_right;

  for (const auto & traffic_light : all_traffic_lights) {
    const auto & tl_left_down_point = traffic_light.tls.front();
    const auto & tl_right_down_point = traffic_light.tls.back();
    const double tl_height = traffic_light.tls.attributeOr("height", 0.0);

    tf2::Transform tf_map2stop_left(tf2::Quaternion(0,0,0,1),
      tf2::Vector3(traffic_light.stop_line_left.x, traffic_light.stop_line_left.y, traffic_light.stop_line_left.z));
    tf2::Transform tf_map2stop_right(tf2::Quaternion(0,0,0,1),
      tf2::Vector3(traffic_light.stop_line_right.x, traffic_light.stop_line_right.y, traffic_light.stop_line_right.z));
    //map -> base_link로 좌표 변환    
    tf_base_link2stop_left = tf_base_link2map * tf_map2stop_left;
    tf_base_link2stop_right = tf_base_link2map * tf_map2stop_right;
     
    //ref. https://rito15.github.io/posts/foot-of-perpendicular-in-three-vectors/
    //AB 벡터
    tf2::Vector3 stop_line_vector(tf_base_link2stop_right.getOrigin().x() - tf_base_link2stop_left.getOrigin().x(), tf_base_link2stop_right.getOrigin().y() - tf_base_link2stop_left.getOrigin().y(), 0);
    //AC 벡터
    tf2::Vector3 stop_line_left2base_link_vector(-tf_base_link2stop_left.getOrigin().x(), -tf_base_link2stop_left.getOrigin().y(), 0);

    //내적 계산시 분모 0이 되는 부분 예외처리
    double inner_product_result = 0.0;
    if(tf2::tf2Dot(stop_line_vector,stop_line_vector) != 0.0)
      inner_product_result =  tf2::tf2Dot(stop_line_vector, stop_line_left2base_link_vector) / tf2::tf2Dot(stop_line_vector,stop_line_vector);
  
    //수선의 발 좌표
    tf2::Vector3 foot_of_perpendicular(tf_base_link2stop_left.getOrigin().x() + stop_line_vector.getX() * inner_product_result,
    tf_base_link2stop_left.getOrigin().y() + stop_line_vector.getY() * inner_product_result,
    0);
    //정지선 위의 수선의 발과 base_link 간의 거리
    double stop_line_distance = tf2::tf2Distance(foot_of_perpendicular, tf2::Vector3(0, 0, 0));
    
    //정지선 기준100보다 멀리 떨어져있으면 continue
    if(stop_line_distance > config_.max_distance) continue;
    //정지선 filter 적용 0m 넘었으면 continue
    if(config_.car_offset_bool == true && foot_of_perpendicular.getX() < config_.min_distance) continue;
    traffic_light.stop_line_distance = stop_line_distance;
    
    geometry_msgs::Point tl_central_point;
    tl_central_point.x = (tl_right_down_point.x() + tl_left_down_point.x()) / 2.0;
    tl_central_point.y = (tl_right_down_point.y() + tl_left_down_point.y()) / 2.0;
    tl_central_point.z = (tl_right_down_point.z() + tl_left_down_point.z() + tl_height) / 2.0;

    // check angle range
    const double tl_yaw = normalizeAngle(
      std::atan2(
        tl_right_down_point.y() - tl_left_down_point.y(),
        tl_right_down_point.x() - tl_left_down_point.x()) +
      M_PI_2);
    const double max_angele_range = 40.0 / 180.0 * M_PI;

    // get direction of z axis
    tf2::Vector3 camera_z_dir(0, 0, 1);
    tf2::Matrix3x3 camera_rotation_matrix(tf2::Quaternion(
      camera_pose.orientation.x, camera_pose.orientation.y, camera_pose.orientation.z,
      camera_pose.orientation.w));
    camera_z_dir = camera_rotation_matrix * camera_z_dir;
    double camera_yaw = std::atan2(camera_z_dir.y(), camera_z_dir.x());
    camera_yaw = normalizeAngle(camera_yaw);
    if (!isInAngleRange(tl_yaw, camera_yaw, max_angele_range)) continue;

    // check within image frame
    tf2::Transform tf_map2camera(
      tf2::Quaternion(
        camera_pose.orientation.x, camera_pose.orientation.y, camera_pose.orientation.z,
        camera_pose.orientation.w),
      tf2::Vector3(camera_pose.position.x, camera_pose.position.y, camera_pose.position.z));
    tf2::Transform tf_map2tl(
      tf2::Quaternion(0, 0, 0, 1),
      tf2::Vector3(tl_central_point.x, tl_central_point.y, tl_central_point.z));
    tf2::Transform tf_camera2tl;
    tf_camera2tl = tf_map2camera.inverse() * tf_map2tl;

    geometry_msgs::Point camera2tl_point;
    camera2tl_point.x = tf_camera2tl.getOrigin().x();
    camera2tl_point.y = tf_camera2tl.getOrigin().y();
    camera2tl_point.z = tf_camera2tl.getOrigin().z();
    if (!isInImageFrame(camera_info, camera2tl_point)) continue;
    visible_traffic_lights.push_back(traffic_light);
  }
}

bool MapBasedDetector::isInAngleRange(
  const double & tl_yaw, const double & camera_yaw, const double max_angele_range)
{
  Eigen::Vector2d vec1, vec2;
  vec1 << std::cos(tl_yaw), std::sin(tl_yaw);
  vec2 << std::cos(camera_yaw), std::sin(camera_yaw);
  const double diff_angle = std::acos(vec1.dot(vec2));
  return (std::fabs(diff_angle) < max_angele_range);
}

bool MapBasedDetector::isInImageFrame(
  const sensor_msgs::CameraInfo & camera_info, const geometry_msgs::Point & point)
{
  const double & camera_x = point.x;
  const double & camera_y = point.y;
  const double & camera_z = point.z;
  const double & fx = camera_info.K[(0 * 3) + 0];
  const double & fy = camera_info.K[(1 * 3) + 1];
  const double & cx = camera_info.K[(0 * 3) + 2];
  const double & cy = camera_info.K[(1 * 3) + 2];
  if (camera_z <= 0.0) return false;
  const double image_u = (fx * camera_x + cx * camera_z) / camera_z;
  const double image_v = (fy * camera_y + cy * camera_z) / camera_z;
  if (0 <= image_u && image_u < camera_info.width)
    if (0 <= image_v && image_v < camera_info.height) return true;
  return false;
}

void MapBasedDetector::publishVisibleTrafficLights(
  const geometry_msgs::PoseStamped camera_pose_stamped,
  const std::vector<tls_info> & visible_traffic_lights,
  const ros::Publisher & pub)
{
  visualization_msgs::MarkerArray output_msg;
  for (const auto & traffic_light : visible_traffic_lights) {
    const auto & tl_left_down_point = traffic_light.tls.front();
    const auto & tl_right_down_point = traffic_light.tls.back();
    const double tl_height = traffic_light.tls.attributeOr("height", 0.0);
    const int id = traffic_light.tls.id();

    geometry_msgs::Point tl_central_point;
    tl_central_point.x = (tl_right_down_point.x() + tl_left_down_point.x()) / 2.0;
    tl_central_point.y = (tl_right_down_point.y() + tl_left_down_point.y()) / 2.0;
    tl_central_point.z = (tl_right_down_point.z() + tl_left_down_point.z() + tl_height) / 2.0;

    visualization_msgs::Marker marker;

    tf2::Transform tf_map2camera(
      tf2::Quaternion(
        camera_pose_stamped.pose.orientation.x, camera_pose_stamped.pose.orientation.y,
        camera_pose_stamped.pose.orientation.z, camera_pose_stamped.pose.orientation.w),
      tf2::Vector3(
        camera_pose_stamped.pose.position.x, camera_pose_stamped.pose.position.y,
        camera_pose_stamped.pose.position.z));
    tf2::Transform tf_map2tl(
      tf2::Quaternion(0, 0, 0, 1),
      tf2::Vector3(tl_central_point.x, tl_central_point.y, tl_central_point.z));
    tf2::Transform tf_camera2tl;
    tf_camera2tl = tf_map2camera.inverse() * tf_map2tl;

    marker.header = camera_pose_stamped.header;
    marker.id = id;
    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.ns = std::string("beam");
    marker.scale.x = 0.05;
    marker.action = visualization_msgs::Marker::MODIFY;
    marker.pose.position.x = 0.0;
    marker.pose.position.y = 0.0;
    marker.pose.position.z = 0.0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    geometry_msgs::Point point;
    point.x = 0.0;
    point.y = 0.0;
    point.z = 0.0;
    marker.points.push_back(point);
    point.x = tf_camera2tl.getOrigin().x();
    point.y = tf_camera2tl.getOrigin().y();
    point.z = tf_camera2tl.getOrigin().z();
    marker.points.push_back(point);

    marker.lifetime = ros::Duration(0.2);
    marker.color.a = 0.999;  // Don't forget to set the alpha!
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;

    output_msg.markers.push_back(marker);
  }
  pub.publish(output_msg);

  return;
}

double MapBasedDetector::normalizeAngle(const double & angle)
{
  return std::atan2(std::cos(angle), std::sin(angle));
}

}  // namespace traffic_light
