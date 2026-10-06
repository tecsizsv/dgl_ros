#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <dgl_ros_interfaces/action/sample_grasp_poses.hpp>
#include <Eigen/Dense>
#include <pcl_conversions/pcl_conversions.h>
#include <dgl_ros/util/geometry.hpp>
#include <dgl_ros/util/cloud.hpp>
#include <dgl_ros_models/gpd.hpp>
#include <gpd/grasp_detector.h>
#include <pcl/filters/crop_box.h>

using dgl_ros_interfaces::action::SampleGraspPoses;
using sensor_msgs::msg::PointCloud2;

typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudRGB;
typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloudRGBA;

namespace dgl_models
{

Gpd::Gpd(rclcpp::NodeOptions& options) : GpdAgent(options)
{
  tf_lookup_ = std::make_unique<dgl::util::TransformLookup>(this);

  this->declare_parameter("tf_timeout_seconds", 10);
  this->declare_parameter("gpd_config_path", "/simply_ws/src/dgl_ros/dgl_ros_models/config/gpd_config.yaml");

  gpd_grasp_detector_ = std::make_unique<gpd::GraspDetector>(this->get_parameter("gpd_config_path").as_string());

  tf_lookup_->get_tf_isometry(
    this->get_parameter("world_frame").as_string(),
    this->get_parameter("src_frame0").as_string(),
    this->get_parameter("tf_timeout_seconds").as_int(),
    tf_world_src_);
}

SampleGraspPoses::Result::SharedPtr Gpd::actionFromObs(std::shared_ptr<GpdObserver> observer, GoalSharedPtr goal)
{
  const auto& workspace = goal->workspace; // workspace = (xmin xmax ymin ymax zmin zmax)

  RCLCPP_INFO(this->get_logger(), "New action received...");
  RCLCPP_INFO_STREAM(this->get_logger(), "Workspace size: " << workspace.size());
  
  // Get the latest observation
  auto [id, msg] = observer->observe();

  // Convert to PCL.
  PointCloudRGB cloud;
  pcl::fromROSMsg(*msg, cloud);
  pcl::io::savePCDFileASCII ("temp_ros_cloud.pcd", cloud); // Save the point cloud to a PCD file
  
  // Convert to RGBA for cropping and GPD
  auto grasp_cloud = std::make_shared<PointCloudRGBA>();
  pcl::copyPointCloud(cloud, *grasp_cloud);

  if(workspace.size() == 6)
  {
    // Workspace corners in the world frame (from the action goal)
    Eigen::Vector3d ws_min_world(workspace[0], workspace[2], workspace[4]);
    Eigen::Vector3d ws_max_world(workspace[1], workspace[3], workspace[5]);

    // world -> camera transform
    const Eigen::Isometry3d tf_src_world = tf_world_src_.inverse();

    // Corners in the camera frame
    Eigen::Vector3d ws_corner_a_cam = tf_src_world * ws_min_world;
    Eigen::Vector3d ws_corner_b_cam = tf_src_world * ws_max_world;

    // Sort vectors
    Eigen::Vector3d ws_min_cam = ws_corner_a_cam.cwiseMin(ws_corner_b_cam);
    Eigen::Vector3d ws_max_cam = ws_corner_a_cam.cwiseMax(ws_corner_b_cam);
    RCLCPP_INFO_STREAM(this->get_logger(), "ws_min_cam: " << ws_min_cam.transpose() << " | ws_max_cam: " << ws_max_cam.transpose());

    // Crop the point cloud to the workspace
    pcl::CropBox<pcl::PointXYZRGBA> crop_box;
    crop_box.setMin(ws_min_cam.cast<float>().homogeneous());
    crop_box.setMax(ws_max_cam.cast<float>().homogeneous());
    RCLCPP_INFO_STREAM(this->get_logger(), "Cloud size before crop: " << grasp_cloud->size());

    crop_box.setInputCloud(grasp_cloud);
    crop_box.filter(*grasp_cloud);
    RCLCPP_INFO_STREAM(this->get_logger(), "Cloud size after crop: "  << grasp_cloud->size());

    if(grasp_cloud->empty())
    {
      RCLCPP_WARN(this->get_logger(), "No points left after cropping to workspace, skipping grasp detection.");
      return std::make_shared<SampleGraspPoses::Result>();
    }
  }
  else
  {
    RCLCPP_WARN_STREAM(this->get_logger(), "Workspace has " << workspace.size() << " values, expected 6. Skipping crop."); 
  }

  // Convert to GPD.
  RCLCPP_INFO(this->get_logger(), "Preprocess pointcloud...");
  Eigen::Matrix3Xd camera_view_point(3, 1);
  gpd::util::Cloud gpd_cloud(grasp_cloud, 0, camera_view_point);
  gpd_grasp_detector_->preprocessPointCloud(gpd_cloud);

  // Detect grasps
  RCLCPP_INFO(this->get_logger(), "Starting grasp detection...");
  std::vector<std::unique_ptr<gpd::candidate::Hand>> grasps;  
  grasps = gpd_grasp_detector_->detectGrasps(gpd_cloud);  // detect grasp poses in the point cloud
  
  std::vector<unsigned int> grasp_ids;
  for (unsigned int i = 0; i < grasps.size(); i++)
  {
    grasp_ids.push_back(i);
  }
  RCLCPP_INFO_STREAM(this->get_logger(), "Detected " << grasps.size() << " grasps.");

  // Transform grasps to world frame and build result
  auto result = std::make_shared<SampleGraspPoses::Result>();
  for (auto id : grasp_ids)
  {
    // Transform grasp from camera optical link into frame_id
    const Eigen::Isometry3d transform_opt_grasp =
        Eigen::Translation3d(grasps.at(id)->getPosition()) * Eigen::Quaterniond(grasps.at(id)->getOrientation());

    const Eigen::Isometry3d transform_base_grasp = tf_world_src_ * transform_opt_grasp;
    const Eigen::Vector3d trans = transform_base_grasp.translation();
    const Eigen::Quaterniond rot(transform_base_grasp.rotation());

    // Convert back to PoseStamped
    geometry_msgs::msg::PoseStamped grasp_pose;
    grasp_pose.header.frame_id = "world";
    grasp_pose.pose.position.x = trans.x();
    grasp_pose.pose.position.y = trans.y();
    grasp_pose.pose.position.z = trans.z();

    grasp_pose.pose.orientation.w = rot.w();
    grasp_pose.pose.orientation.x = rot.x();
    grasp_pose.pose.orientation.y = rot.y();
    grasp_pose.pose.orientation.z = rot.z();

    result->grasp_candidates.emplace_back(grasp_pose);
    
    // Grasp is selected based on cost not score
    // Invert score to represent grasp with lowest cost
    result->costs.emplace_back(static_cast<double>(1.0 / grasps.at(id)->getScore()));
  }

  RCLCPP_INFO(this->get_logger(), "Finalizing grasp detection.");
  return result;
}

std::unique_ptr<PointCloud2> Gpd::obsFromSrcs(std::shared_ptr<PointCloud2> msg)
{
  // Convert to PCL
  auto cloud = std::make_shared<PointCloudRGB>();
  pcl::fromROSMsg(*msg, *cloud);

  // Segementation works best with XYXRGB
  dgl::util::cloud::removeTable(cloud);

  // Return the final observation
  std::unique_ptr<PointCloud2> cloud_msg = std::make_unique<PointCloud2>();
  pcl::toROSMsg(*cloud, *cloud_msg);
  return cloud_msg;
}
}  // namespace dgl_models

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(true);
  dgl_models::Gpd server(options);
  server.run();
  rclcpp::shutdown();
  return 0;
}