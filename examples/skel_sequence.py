# # Copyright (C) 2024  MPI IS, Marilyn Keller
# import argparse
# import os

# import numpy as np
# import torch

# from aitviewer.viewer import Viewer
# from aitviewer.renderables.skel import SKELSequence
# from aitviewer.renderables.smpl import SMPLSequence
# from aitviewer.configuration import CONFIG as C
# from skel.skel_model import SKEL


# if __name__ == "__main__":
    
#     parser = argparse.ArgumentParser(description='Visualize a SKEL sequence.')
    
#     parser.add_argument('skel_file', type=str, help='Path to the SKEL sequence to visualize.')
#     parser.add_argument('--smpl_seq', type=str, help='The corresponding SMPL sequence', default=None)
#     parser.add_argument('--fps', type=int, help='Fps of the sequence', default=120)
#     parser.add_argument('--fps_out', type=int, help=' Fps at which the sequence will be played back', default=15)
#     parser.add_argument('-z', '--z-up', help='Use Z-up coordinate system. \
#         This is usefull for vizualizing sequences of AMASS that are 90 degree rotated', action='store_true')
#     parser.add_argument('-g', '--gender', type=str, default=None, help='Forces the gender for visualization. By default, the code tries to load the gender from the skel file')
#     parser.add_argument('-e', '--export_mesh', type=str, help='Export the mesh of the skel model to this folder', default=None)
#     parser.add_argument('--offset', help='Offset the SMPL model to display it beside SKEL.', action='store_true') 
                        
#     args = parser.parse_args()
    
#     to_display = []
    
#     fps_in = args.fps # Fps of the sequence
#     fps_out = args.fps_out # Fps at which the sequence will be played back
#     # The skeleton mesh has a lot of vertices, so we don't load all the frames to avoid memory issues
#     if args.smpl_seq is not None:
#         if args.offset:
#             translation = np.array([-1.0, 0.0, 0.0])
#         else:
#             translation = None
            
#         smpl_seq = SMPLSequence.from_amass(
#                         npz_data_path=args.smpl_seq,
#                         fps_out=fps_out,
#                         name="SMPL",
#                         show_joint_angles=True,
#                         position=translation,
#                         z_up=args.z_up
#                         )   
#         to_display.append(smpl_seq)
        

#     skel_seq = SKELSequence.from_file(skel_seq_file = args.skel_file, 
#                                      poses_type='skel', 
#                                      fps_in=fps_in,
#                                      fps_out=fps_out,
#                                      is_rigged=True, 
#                                      show_joint_angles=True, 
#                                      name='SKEL', 
#                                      z_up=args.z_up)
#     to_display.append(skel_seq)

#     v = Viewer()
#     v.playback_fps = fps_out
#     v.scene.add(*to_display)
#     v.scene.camera.position = np.array([-5, 1.7, 0.0])
#     v.lock_to_node(skel_seq, (2, 0.7, 2), smooth_sigma=5.0)
    
#     v.run_animations = True 
#     v.run()
# Copyright (C) 2024  MPI IS, Marilyn Keller
import argparse
import os
os.environ.setdefault("PYGLET_HEADLESS", "1")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np

def build_scene(args):
    # 这里再导入 aitviewer 的可渲染对象，避免过早触发窗口后端加载
    from aitviewer.renderables.skel import SKELSequence
    from aitviewer.renderables.smpl import SMPLSequence

    to_display = []

    fps_in = args.fps          # 输入序列帧率
    fps_out = args.fps_out     # 播放/导出帧率

    # 可选：同时显示 SMPL（来自 AMASS .npz），并可平移到旁边对比
    if args.smpl_seq is not None:
        translation = np.array([-1.0, 0.0, 0.0]) if args.offset else None
        smpl_seq = SMPLSequence.from_amass(
            npz_data_path=args.smpl_seq,
            fps_out=fps_out,
            name="SMPL",
            show_joint_angles=True,
            position=translation,
            z_up=args.z_up,
        )
        to_display.append(smpl_seq)

    # SKEL 序列（.pkl）
    skel_seq = SKELSequence.from_file(
        skel_seq_file=args.skel_file,
        poses_type='skel',
        fps_in=fps_in,
        fps_out=fps_out,
        is_rigged=True,
        show_joint_angles=True,
        name='SKEL',
        z_up=args.z_up
    )
    to_display.append(skel_seq)

    return to_display, skel_seq


def main():
    parser = argparse.ArgumentParser(description='Visualize or render a SKEL sequence (headless-friendly).')
    parser.add_argument('skel_file', type=str, help='Path to the SKEL sequence to visualize (.pkl).')
    parser.add_argument('--smpl_seq', type=str, help='Optional AMASS-style SMPL sequence (.npz).', default=None)
    parser.add_argument('--fps', type=int, default=30, help='Input sequence fps.')
    parser.add_argument('--fps_out', type=int, default=15, help='Playback/export fps.')
    parser.add_argument('-z', '--z-up', action='store_true',
                        help='Use Z-up coordinates (useful for AMASS sequences rotated by 90°).')
    parser.add_argument('-g', '--gender', type=str, default=None,
                        help='Force gender for visualization (by default read from SKEL file).')
    parser.add_argument('-e', '--export_mesh', type=str, default=None,
                        help='Export the SKEL mesh to this folder (optional).')
    parser.add_argument('--offset', action='store_true',
                        help='Offset the SMPL model to display it beside SKEL.')

    # 新增：无头渲染与导出参数
    parser.add_argument('--headless', action='store_true',
                        help='Force headless offscreen rendering (recommended on servers).')
    parser.add_argument('--out', type=str, default=None,
                        help='If set, export an MP4 to this path (e.g., out.mp4).')
    parser.add_argument('--frames', type=str, default=None,
                        help='If set, export per-frame PNGs to this directory.')
    parser.add_argument('--width', type=int, default=1280, help='Render width.')
    parser.add_argument('--height', type=int, default=720, help='Render height.')
    args = parser.parse_args()

    # —— 选择是否无头 —— #
    # 显式指定 --headless 或者没有 DISPLAY 时，进入无头模式
    headless = args.headless or (os.environ.get("DISPLAY") in [None, "", "None"])

    if headless:
        # 在导入任何 aitviewer 代码前设置无头相关环境
        os.environ.setdefault("AITVIEWER_WINDOW", "headless")
        os.environ.setdefault("PYGLET_HEADLESS", "1")
        # 用 EGL（有 NVIDIA 驱动时更稳；没有也可走 Mesa EGL）
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


        from moderngl_window.context import headless as mglw_headless

        _orig_init = mglw_headless.window.Window.init_mgl_context
        def _init_mgl_context_egl(self):
            import moderngl
            # 关键：明确指定 backend='egl'
            self._ctx = moderngl.create_standalone_context(backend='egl')
            self._fbo = self._ctx.simple_framebuffer((self.width, self.height), components=4)
            self._fbo.use()
            self._ctx.viewport = (0, 0, self.width, self.height)
        mglw_headless.window.Window.init_mgl_context = _init_mgl_context_egl
        from aitviewer.headless import HeadlessRenderer
        # 组装场景
        to_display, skel_seq = build_scene(args)

        # 创建无头渲染器并添加对象
        r = HeadlessRenderer(width=args.width, height=args.height)
        r.scene.add(*to_display)

        # 相机与跟随（若接口与 Viewer 一致，这样设置即可；否则可忽略相机设置）
        try:
            r.scene.camera.position = np.array([-5, 1.7, 0.0])
        except Exception:
            pass
        try:
            # 跟随骨架，参数: (distance, height, stiffness) 具体取决于实现
            r.lock_to_node(skel_seq, (2, 0.7, 2), smooth_sigma=5.0)
        except Exception:
            pass

        # —— 导出 —— #
        r.playback_fps = args.fps_out

        # 如果想同时导出逐帧图像，给一个目录；不需要就设为 None
        frames_dir = None  # 比如 "frames_out" 也行

        # 关键：按你的 API 调用 save_video（注意参数名）
        r.save_video(
            frame_dir=frames_dir,
            video_dir=args.out,           # e.g. "skel_out.mp4"
            output_fps=args.fps_out,      # 输出视频帧率
            transparent=False             # 只对 .webm 有效；.mp4 会被忽略
        )
        print("Saved:", args.out)


    else:
        # 有显示：正常 GUI 预览
        #（注意：只有在这里才导入 Viewer，避免在无头时触发窗口依赖）
        from aitviewer.viewer import Viewer
        from aitviewer.configuration import CONFIG as C

        # 如需强制窗口后端，可在这里改：C.update_conf({"window_type": "glfw"})
        to_display, skel_seq = build_scene(args)

        v = Viewer()
        v.playback_fps = args.fps_out
        v.scene.add(*to_display)
        v.scene.camera.position = np.array([-5, 1.7, 0.0])
        v.lock_to_node(skel_seq, (2, 0.7, 2), smooth_sigma=5.0)
        v.run_animations = True
        v.run()


if __name__ == "__main__":
    main()
