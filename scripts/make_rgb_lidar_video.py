#!/usr/bin/env python3
"""
Side-by-side (또는 3분할) 합성 비디오 생성:
- 좌: RGB 프레임(JPG 시퀀스)
- 중: LiDAR 탑뷰(그리드/라벨/축/스케일바/강도 컬러맵 등 옵션)
- 우(선택): YOLO 오버레이 이미지 시퀀스(있으면 동일 인덱스로 매칭)

OpenGL 필요 없음. OpenCV(+선택적 Open3D)만 사용.

예시:
  python3 scripts/make_rgb_lidar_video.py \
    --root "/workspace/media/raw/Validation/01.원천데이터/VS/RB1(4족보행로봇)/PL08(중형식당)/D1(순환주행)/P1(수동조종자A)/SN03/RB1_PL08_D1_P1_SN03_1" \
    --fps 10 \
    --auto-range \
    --grid --grid-step 2 \
    --labels \
    --point-size 3 \
    --intensity --intensity-cmap turbo \
    --view-size 800 800 \
    --out "/workspace/output/RGB_LiDAR_side_by_side.mp4"

YOLO 오버레이 3분할:
    ... --yolo-sub "RGB-D(Image)_overlay"
"""

import os
import re
import glob
import argparse
import math
import cv2
import numpy as np
import datetime

# Open3D는 선택 사항 (없어도 ASCII PCD는 직접 파싱)
try:
    import open3d as o3d  # type: ignore
    HAS_O3D = True
except Exception:
    HAS_O3D = False

# ===== 기본값 =====
DEF_ROOT = (
    "/workspace/media/raw/Validation/01.원천데이터/VS/"
    "RB1(4족보행로봇)/PL08(중형식당)/D1(순환주행)/P1(수동조종자A)/SN03/"
    "RB1_PL08_D1_P1_SN03_1"
)
DEF_RGB_SUB = "RGB-D(Image)"
DEF_PCD_SUB = "LiDAR"
DEF_YOLO_SUB = None  # 기본 비활성
DEF_OUT = "RGB_LiDAR_side_by_side.mp4"

# ===== 유틸 =====
def list_indexed(dir_path: str, exts=(".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG")):
    """디렉토리에서 파일명 끝 _dddd.ext 형태를 찾아 (index, path) 리스트 반환"""
    rx = re.compile(r".*_(\d{4})\.(\w+)$")
    items = []
    for ext in exts:
        for p in glob.glob(os.path.join(dir_path, f"*{ext}")):
            m = rx.match(p)
            if m:
                try:
                    items.append((int(m.group(1)), p))
                except Exception:
                    pass
    items.sort(key=lambda t: t[0])
    return items

def read_pcd_ascii(path: str):
    """
    아주 단순한 ASCII PCD 파서.
    반환: (pts_xyz: [N,3] float32, intensity: [N] float32 or None)
    """
    fields = []
    types = []
    sizes = []
    count = []
    data_mode = None
    header_done = False
    xyz_idx = [-1, -1, -1]
    intens_idx = -1
    values = []
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            s = line.strip().decode("utf-8", "ignore")
            if s == "":
                continue
            if s.lower().startswith("data"):
                data_mode = s.split(None, 1)[1].lower() if len(s.split()) > 1 else ""
                header_done = True
                break
            if s.lower().startswith("fields"):
                fields = s.split()[1:]
            elif s.lower().startswith("type"):
                types = s.split()[1:]
            elif s.lower().startswith("size"):
                sizes = s.split()[1:]
            elif s.lower().startswith("count"):
                count = s.split()[1:]
            # 다른 헤더 항목들(WIDTH/HEIGHT/POINTS 등)은 여기선 생략해도 OK

        if not header_done:
            return np.zeros((0,3), np.float32), None

        # 필드 인덱스 파악
        for i, name in enumerate(fields):
            if name == "x": xyz_idx[0] = i
            elif name == "y": xyz_idx[1] = i
            elif name == "z": xyz_idx[2] = i
            elif name.lower() in ("intensity", "i"):
                intens_idx = i

        lines = f.read().decode("utf-8", "ignore").strip().splitlines()

    xyz_list = []
    inten_list = [] if intens_idx >= 0 else None
    for ln in lines:
        cols = ln.strip().split()
        if len(cols) < max(xyz_idx + [intens_idx if intens_idx>=0 else 0]) + 1:
            continue
        try:
            x = float(cols[xyz_idx[0]]) if xyz_idx[0] >= 0 else 0.0
            y = float(cols[xyz_idx[1]]) if xyz_idx[1] >= 0 else 0.0
            z = float(cols[xyz_idx[2]]) if xyz_idx[2] >= 0 else 0.0
            xyz_list.append((x, y, z))
            if intens_idx >= 0:
                inten_list.append(float(cols[intens_idx]))
        except Exception:
            pass

    if not xyz_list:
        return np.zeros((0,3), np.float32), (None if inten_list is None else np.zeros((0,), np.float32))
    xyz = np.array(xyz_list, dtype=np.float32)
    inten = None
    if inten_list is not None:
        inten = np.array(inten_list, dtype=np.float32)
    return xyz, inten

def load_pcd(path: str):
    """Open3D가 있으면 먼저 시도하고 실패 시 ASCII 파서로."""
    if HAS_O3D:
        try:
            pcd = o3d.io.read_point_cloud(path)
            xyz = np.asarray(pcd.points, dtype=np.float32)
            # intensity는 표준화되어 있지 않아서 보통 없음 → None 처리
            return xyz, None
        except Exception:
            pass
    return read_pcd_ascii(path)

def normalize_to_uint8(vals: np.ndarray, lo=None, hi=None):
    if vals.size == 0:
        return vals.astype(np.uint8)
    v = vals.astype(np.float32)
    if lo is None: lo = np.percentile(v, 1.0)
    if hi is None: hi = np.percentile(v, 99.0)
    if hi <= lo:
        hi = lo + 1e-6
    v = (v - lo) / (hi - lo)
    v = np.clip(v, 0, 1)
    return (v * 255.0 + 0.5).astype(np.uint8)

# ===== LiDAR 렌더링 =====
def draw_grid_and_axes(img, xy_range, grid_step=2.0, labels=False):
    h, w = img.shape[:2]
    xmin, xmax, ymin, ymax = xy_range
    # 축
    cx = int((-xmin) / (xmax - xmin) * (w - 1))
    cy = int((1.0 - (-ymin) / (ymax - ymin)) * (h - 1))
    cv2.line(img, (0, cy), (w-1, cy), (80, 80, 80), 1)
    cv2.line(img, (cx, 0), (cx, h-1), (80, 80, 80), 1)
    # 그리드
    if grid_step > 0:
        # x 방향 수직선
        x = math.ceil(xmin / grid_step) * grid_step
        while x <= xmax:
            px = int((x - xmin) / (xmax - xmin) * (w - 1))
            cv2.line(img, (px, 0), (px, h-1), (40, 40, 40), 1)
            if labels:
                cv2.putText(img, f"{x:.0f}", (px+2, min(h-4, cy-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1, cv2.LINE_AA)
            x += grid_step
        # y 방향 수평선
        y = math.ceil(ymin / grid_step) * grid_step
        while y <= ymax:
            py = int((1.0 - (y - ymin) / (ymax - ymin)) * (h - 1))
            cv2.line(img, (0, py), (w-1, py), (40, 40, 40), 1)
            if labels:
                cv2.putText(img, f"{y:.0f}", (min(w-30, cx+2), py-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1, cv2.LINE_AA)
            y += grid_step

    # 중심 방향 화살표(+X, +Y)
    cv2.arrowedLine(img, (cx, cy), (min(w-1, cx+60), cy), (200, 200, 200), 2, tipLength=0.25)  # +X →
    cv2.putText(img, "+X", (min(w-30, cx+65), cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)
    cv2.arrowedLine(img, (cx, cy), (cx, max(0, cy-60)), (200, 200, 200), 2, tipLength=0.25)  # +Y ↑
    cv2.putText(img, "+Y", (cx+5, max(15, cy-65)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

def draw_scale_bar(img, xy_range, meters=5.0):
    """하단 좌측에 스케일바(meters m) 그리기"""
    h, w = img.shape[:2]
    xmin, xmax, ymin, ymax = xy_range
    px_per_m = (w - 1) / (xmax - xmin)
    bar_px = int(px_per_m * meters)
    x0, y0 = 15, h - 20
    x1 = min(w - 15, x0 + bar_px)
    cv2.line(img, (x0, y0), (x1, y0), (255, 255, 255), 3)
    cv2.putText(img, f"{meters:.0f} m", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)

def color_map(name: str, vals_uint8: np.ndarray):
    """OpenCV 컬러맵 매핑 (uint8 0..255)"""
    # OpenCV 컬러맵 이름 매핑
    name = name.lower()
    cv_map = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
        "inferno": getattr(cv2, "COLORMAP_INFERNO", cv2.COLORMAP_JET),
        "magma": getattr(cv2, "COLORMAP_MAGMA", cv2.COLORMAP_JET),
        "plasma": getattr(cv2, "COLORMAP_PLASMA", cv2.COLORMAP_JET),
    }
    cmap = cv_map.get(name, cv2.COLORMAP_JET)
    col = cv2.applyColorMap(vals_uint8, cmap)
    return col

def lidar_topdown_image(pts_xyz: np.ndarray,
                        intens: np.ndarray | None,
                        size=(640, 480),
                        xy_range=(-10, 10, -10, 10),
                        point_size=2,
                        draw_grid=False,
                        grid_step=2.0,
                        labels=False,
                        draw_axes=True,
                        draw_scalebar=True,
                        use_intensity=False,
                        intensity_cmap="turbo",
                        inten_min=None, inten_max=None):
    w, h = size
    xmin, xmax, ymin, ymax = xy_range
    img = np.zeros((h, w, 3), np.uint8)

    # 배경 그리드/축 먼저
    if draw_grid or draw_axes:
        draw_grid_and_axes(img, xy_range, grid_step=grid_step, labels=labels)

    if pts_xyz.size == 0:
        if draw_scalebar:
            draw_scale_bar(img, xy_range)
        cv2.putText(img, "LiDAR Top-Down", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        return img

    x = pts_xyz[:, 0]
    y = pts_xyz[:, 1]

    mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    if not np.any(mask):
        if draw_scalebar:
            draw_scale_bar(img, xy_range)
        return img

    x = x[mask]
    y = y[mask]

    xs = ((x - xmin) / (xmax - xmin) * (w - 1)).astype(np.int32)
    ys = ((1.0 - (y - ymin) / (ymax - ymin)) * (h - 1)).astype(np.int32)

    if use_intensity and intens is not None and intens.shape[0] >= pts_xyz.shape[0]:
        vi = intens[mask]
        vi8 = normalize_to_uint8(vi, inten_min, inten_max).reshape(-1, 1)
        col = color_map(intensity_cmap, vi8)  # (N,1,3)
        cols = col.reshape(-1, 3).astype(np.uint8)
    else:
        cols = np.full((xs.shape[0], 3), (0, 255, 0), np.uint8)

    # 점 찍기(굵기)
    if point_size <= 1:
        img[ys.clip(0, h - 1), xs.clip(0, w - 1)] = cols
    else:
        ps = int(point_size)
        for i in range(xs.shape[0]):
            cx, cy = int(xs[i]), int(ys[i])
            cv2.circle(img, (cx, cy), ps, tuple(int(v) for v in cols[i]), -1, lineType=cv2.LINE_AA)

    if draw_scalebar:
        draw_scale_bar(img, xy_range)
    return img

# ===== 메인 =====
def main():
    ap = argparse.ArgumentParser(description="Make side-by-side (or triple) RGB + LiDAR ( + YOLO overlay ) MP4")
    ap.add_argument("--root", default=DEF_ROOT, help="Dataset root directory")
    ap.add_argument("--rgb-sub", default=DEF_RGB_SUB, help="RGB subdirectory under root")
    ap.add_argument("--pcd-sub", default=DEF_PCD_SUB, help="PCD subdirectory under root")
    ap.add_argument("--yolo-sub", default=DEF_YOLO_SUB, help="(optional) YOLO overlay image subdirectory under root")
    ap.add_argument("--out", default=DEF_OUT, help="Output MP4 filename (absolute or relative to root)")
    ap.add_argument("--fps", type=float, default=10.0, help="Output FPS")
    ap.add_argument("--view-size", type=int, nargs=2, default=(640, 480), metavar=("W", "H"), help="LiDAR top-down view size (WxH)")
    ap.add_argument("--xy-range", type=float, nargs=4, default=(-10, 10, -10, 10),
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                    help="Top-down range in meters")
    ap.add_argument("--index-digits", type=int, default=4, help="Index digits in filenames (e.g., 4 for 0001)")

    # 새 옵션들
    ap.add_argument("--grid", action="store_true", help="Draw grid")
    ap.add_argument("--grid-step", type=float, default=2.0, help="Grid spacing in meters")
    ap.add_argument("--labels", action="store_true", help="Draw axis tick labels on grid")
    ap.add_argument("--no-axes", action="store_true", help="Hide axes/arrows")
    ap.add_argument("--no-scalebar", action="store_true", help="Hide scale bar")
    ap.add_argument("--point-size", type=int, default=2, help="LiDAR point radius in pixels")

    ap.add_argument("--intensity", action="store_true", help="Use intensity color map if available")
    ap.add_argument("--intensity-cmap", default="turbo", help="Color map name: turbo/jet/viridis/hot/... (OpenCV colormaps)")
    ap.add_argument("--intensity-min", type=float, default=None, help="Fix intensity min (else percentile auto)")
    ap.add_argument("--intensity-max", type=float, default=None, help="Fix intensity max (else percentile auto)")

    ap.add_argument("--auto-range", action="store_true", help="Auto-fit LiDAR XY range per frame (percentile)")
    ap.add_argument("--auto-range-percentile", type=float, nargs=2, default=(1.0, 99.0), metavar=("PLOW", "PHI"),
                    help="Percentile range for auto-fit (e.g., 1 99)")
    ap.add_argument("--range-pad", type=float, default=1.0, help="Padding meters added around auto range")
    ap.add_argument("--range-ema", type=float, default=0.0, help="EMA smoothing for auto range [0..1], 0=off")

    ap.add_argument("--watermark", action="store_true", help="Draw watermark (idx / FPS)")
    ap.add_argument("--title", default="LiDAR Top-Down (x right, y up)", help="Title text on LiDAR pane")

    args = ap.parse_args()

    root = os.path.abspath(args.root)
    rgb_dir = os.path.join(root, args.rgb_sub)
    pcd_dir = os.path.join(root, args.pcd_sub)
    yolo_dir = os.path.join(root, args.yolo_sub) if args.yolo_sub else None
    out_path = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)

    base, ext = os.path.splitext(out_path)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{base}_{ts}{ext}"

    # 파일 목록
    rgb_list = list_indexed(rgb_dir)
    if not rgb_list:
        print("RGB 시퀀스를 찾지 못했습니다.", rgb_dir)
        return
    pcd_list = []
    rx = re.compile(rf".*_(\d{{{args.index_digits}}})\.pcd$")
    for p in glob.glob(os.path.join(pcd_dir, "*.pcd")):
        m = rx.match(p)
        if m:
            pcd_list.append((int(m.group(1)), p))
    pcd_list.sort(key=lambda t: t[0])

    if not pcd_list:
        print("PCD 시퀀스를 찾지 못했습니다.", pcd_dir)
        return

    yolo_list = []
    if yolo_dir and os.path.isdir(yolo_dir):
        yolo_list = list_indexed(yolo_dir)

    rgb_dict = dict(rgb_list)
    pcd_dict = dict(pcd_list)
    yolo_dict = dict(yolo_list) if yolo_list else {}

    common_ids = sorted(set(rgb_dict.keys()) & set(pcd_dict.keys()))
    if not common_ids:
        print("인덱스가 매칭되는 RGB/PCD가 없습니다.")
        return

    # 출력 프레임 크기 결정 (좌 RGB, 중 LiDAR, 우 YOLO(선택))
    sample_rgb = cv2.imread(rgb_dict[common_ids[0]], cv2.IMREAD_COLOR)
    if sample_rgb is None:
        print("RGB 샘플을 읽을 수 없습니다.")
        return
    H_rgb, W_rgb = sample_rgb.shape[:2]
    W_lid, H_lid = args.view_size
    panes = 2 + (1 if yolo_dict else 0)
    out_w = W_rgb + W_lid + (W_rgb if yolo_dict else 0)
    out_h = max(H_rgb, H_lid, H_rgb)  # YOLO도 RGB 사이즈로 가정
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(args.fps), (out_w, out_h), True)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {out_path}")

    # 자동 범위용 EMA 상태
    ema_range = None  # (xmin,xmax,ymin,ymax)

    # 강도 min/max 고정 요청이 아니면 프레임마다 퍼센타일로 설정
    inten_min = args.intensity_min
    inten_max = args.intensity_max

    for idx in common_ids:
        rgb = cv2.imread(rgb_dict[idx], cv2.IMREAD_COLOR)
        if rgb is None:
            continue

        # YOLO 오버레이 (선택)
        yolo_img = None
        if yolo_dict and idx in yolo_dict:
            yolo_img = cv2.imread(yolo_dict[idx], cv2.IMREAD_COLOR)

        # LiDAR 로드
        pts, inten = load_pcd(pcd_dict[idx])

        # XY 범위 결정
        if args.auto_range and pts.size:
            x, y = pts[:,0], pts[:,1]
            plow, phi = args.auto_range_percentile
            xmin = np.percentile(x, plow); xmax = np.percentile(x, phi)
            ymin = np.percentile(y, plow); ymax = np.percentile(y, phi)
            # 폭/높이 0 방지
            if xmax <= xmin: xmax = xmin + 1e-3
            if ymax <= ymin: ymax = ymin + 1e-3
            # 패딩
            xmin -= args.range_pad
            xmax += args.range_pad
            ymin -= args.range_pad
            ymax += args.range_pad
            curr = np.array([xmin, xmax, ymin, ymax], np.float32)
            if ema_range is None or args.range_ema <= 0:
                ema_range = curr
            else:
                alpha = float(args.range_ema)
                ema_range = alpha * ema_range + (1.0 - alpha) * curr
            xy_range = tuple(map(float, ema_range))
        else:
            xy_range = tuple(args.xy_range)

        # LiDAR 이미지 생성
        top = lidar_topdown_image(
            pts_xyz=pts,
            intens=inten,
            size=(W_lid, H_lid),
            xy_range=xy_range,
            point_size=max(1, int(args.point_size)),
            draw_grid=args.grid,
            grid_step=float(args.grid_step),
            labels=args.labels,
            draw_axes=not args.no_axes,
            draw_scalebar=not args.no_scalebar,
            use_intensity=args.intensity,
            intensity_cmap=args.intensity_cmap,
            inten_min=inten_min, inten_max=inten_max,
        )

        # 높이 맞추기
        if rgb.shape[0] != top.shape[0]:
            top = cv2.resize(top, (top.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        # YOLO 패널 정리
        if yolo_img is None and yolo_dict:
            # 프레임 없으면 빈 캔버스
            yolo_img = np.zeros_like(rgb)
        elif yolo_img is not None and yolo_img.shape[0] != rgb.shape[0]:
            yolo_img = cv2.resize(yolo_img, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 합치기
        if yolo_dict:
            combo = cv2.hconcat([rgb, top, yolo_img])
        else:
            combo = cv2.hconcat([rgb, top])

        # 워터마크
        if args.watermark:
            txt = f"idx {idx:0{args.index_digits}d} | FPS {args.fps:.1f}"
            cv2.putText(combo, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(combo, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

        # LiDAR 타이틀
        if args.title:
            # LiDAR 패널 좌상단 위치에 타이틀 (RGB 폭만큼 이동)
            x0 = W_rgb + 10
            cv2.putText(combo, args.title, (x0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2, cv2.LINE_AA)

        writer.write(combo)

    writer.release()
    print("DONE:", out_path)

if __name__ == "__main__":
    main()