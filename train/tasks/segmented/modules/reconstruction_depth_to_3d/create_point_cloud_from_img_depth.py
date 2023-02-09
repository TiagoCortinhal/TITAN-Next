import numpy as np
import open3d as o3d
import os
from matplotlib import pyplot as plt

vis = o3d.visualization.VisualizerWithEditing()
vis.create_window()
# view_ctl = vis.get_view_control()

if __name__ == "__main__":
    depth_file = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(os.path.join("./depth"))) for f in fn
                  if f.endswith(".png")]
    depth_file.sort()

    rgb_file = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(os.path.join("./image_2"))) for f in fn
                if f.endswith(".png")]
    rgb_file.sort()
    image_rgb = o3d.io.read_image("002059.png")
    image_depth = o3d.io.read_image("depth_002059.png")

    width = int(image_rgb.get_max_bound()[0])
    height = int(image_rgb.get_max_bound()[1])

    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=718, fy=718, cx=width/2,
                                                         cy=height / 2)
    image_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(image_rgb, image_depth,
                                                                    convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image_rgbd, camera_intrinsic)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=100))
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd.transform([[np.cos(0* np.pi / 180),0,np.sin(0* np.pi / 180),0],
         [0,1,0,0],
         [-np.sin(0* np.pi / 180),0,np.cos(0* np.pi / 180),0],
         [0,0,0,1]])
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud("output.ply", pcd)
    # o3d.visualization.draw_geometries([pcd])
    # try:
    #     input("Press enter to continue")
    # except SyntaxError:
    #     pass
    # vis.add_geometry(pcd)
    # # view_ctl = vis.get_view_control()
    # # view_ctl.set_zoom(0.43999999999999972)
    # # view_ctl.set_up([-0.0054435013708127346, 0.99820001246820245, 0.059725232534554057])
    # # view_ctl.set_front([-0.089708290499224561, -0.059972762207610636, 0.99416079705895477])
    # # view_ctl.change_field_of_view(0.60)
    # vis.poll_events()
    # vis.update_renderer()
    # vis.capture_screen_image("test.png")
    # vis.clear_geometries()

    # for i, (d, r) in enumerate(zip(depth_file, rgb_file)):
    #
    #     image_rgb = o3d.io.read_image(r)
    #     image_depth = o3d.io.read_image(d)
    #
    #     width = int(image_rgb.get_max_bound()[0])
    #     height = int(image_rgb.get_max_bound()[1])
    #
    #     camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=718, fy=718, cx=width / 2,
    #                                                          cy=height / 2)
    #     image_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(image_rgb, image_depth,
    #                                                                     convert_rgb_to_intensity=False)
    #
    #     pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image_rgbd, camera_intrinsic, extrinsic=[
    #         [0.99208632491502935,
    #         0.0087690838177547099,
    #         0.12525105622543939,
    #         0.0],
    #         [0.051142835969738426,
    #         -0.93927585713007045,
    #         -0.33933062717877988,
    #         0.0],
    #         [0.11466967448093424,
    #         0.34305096907248184,
    #         -0.93229120899688189,
    #         0.0],
    #         [-3.4427641123257911e-05,
    #         0.00020554840571429447,
    #         0.00043814178840228743,
    #         1.0]
    #     ])  # ,extrinsic=np.array([[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]]))
    #     pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=100))
    #     # Flip it, otherwise the pointcloud will be upside down
    #     #pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    #     # o3d.visualization.draw_geometries([pcd])
    #     # try:
    #     #     input("Press enter to continue")
    #     # except SyntaxError:
    #     #     pass
    #     vis.add_geometry(pcd)
    #     view_ctl = vis.get_view_control()
    #     view_ctl.set_zoom(0.43999999999999972)
    #     view_ctl.set_up([-0.0054435013708127346, 0.99820001246820245, 0.059725232534554057])
    #     view_ctl.set_front([-0.089708290499224561, -0.059972762207610636, 0.99416079705895477])
    #     view_ctl.change_field_of_view(0.60)
    #     vis.poll_events()
    #     vis.update_renderer()
    #     vis.capture_screen_image("./3d/{:06}.png".format(i))
    #     vis.clear_geometries()
    #
    # while True:
    #     continue

