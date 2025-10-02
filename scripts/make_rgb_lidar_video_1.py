#!/usr/bin/env python3
"""
Make a side-by-side video:
  - Left  : RGB frame sequence (JPG/PNG). (옵션) YOLO 오버레이 폴더로 대체 가능
  - Right : LiDAR PCD top-down 2D 뷰 (OpenGL 불필요, OpenCV로 렌더)

매칭 규칙: 파일명 끝의 4자리 인덱스(예: *_0037.jpg ↔ *_0037.pcd)

예시:
  python3 scripts/make_rgb_lidar_video.py \
    --root "/workspace/media/raw/Validation/01.원천데이터/VS/RB1(4족보행로봇)/PL08(중형식당)/D1(순환주행)/P1(수동조종자A)/SN03/RB1_PL08_D1_P1_SN03_1" \
    --yolo-sub "RGB-D(Image)_overlay" \
    --auto-range --auto-range-percentile 1 99 --range-ema 0.8 --range-pad 1.0 \
    --grid --grid-step 2 --labels \
    --intensity --intensity-cmap turbo --point-size 3 \
    --view-size 800 800 --fps 10 --watermark \
    --out "/workspace/output/RGB_LiDAR_side_by_side.mp4"
"""

import os
import re
import glob
import argparse
import cv2
import numpy as np
import datetime

# Open3D는 있으면 사용, 없으면 ASCII 파서로 fallback
try:
    import open3d as o3d  # type: ignore
    HAS_O3D = True
except Exception:
    HAS_O3D = False

DEF_ROOT = (
    "/workspace/media/raw/Validation/01.원천데이터/VS/"
    "RB1(4족보행로봇)/PL08(중형식당)/D1(순환주행)/P1(수동조종자A)/SN03/"
    "RB1_PL08_D1_P1_SN03_1"
)
DEF_RGB_SUB = "RGB-D(Image)"   # 기본 RGB 폴더 (YOLO 오버레이 폴더가 따로 있으면 --yolo-sub로 교체)
DEF_PCD_SUB = "LiDAR"
DEF_OUT = "RGB_LiDAR_side_by_side.mp4"

def list_indexed(dir_path: str, exts=(".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG")):
    """Return sorted list of (index, path) where filename ends with _dddd.<ext>"""
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

def read_pcd_points_ascii(path: str):
    """아주 단순한 ASCII PCD 리더: x y z [intensity] 를 읽어서 (Nx3), (N,) 반환."""
    try:
        with open(path, "rb") as f:
            # 헤더 건너뛰고 DATA 줄 이후부터 본문
            while True:
                line = f.readline()
                if not line:
                    break
                if line.strip().lower().startswith(b"data"):
                    break
            lines = f.read().decode("utf-8", errors="ignore").strip().splitlines()
    except Exception:
        return np.zeros((0, 3), np.float32), None

    pts = []
    inten = []
    for ln in lines:
        cols = ln.strip().split()
        if len(cols) >= 3:
            try:
                x, y, z = float(cols[0]), float(cols[1]), float(cols[2])
                pts.append((x, y, z))
                if len(cols) >= 4:
                    inten.append(float(cols[3]))
            except Exception:
                pass
    if not pts:
        return np.zeros((0, 3), np.float32), None
    xyz = np.array(pts, dtype=np.float32)
    if inten and len(inten) == len(pts):
        return xyz, np.array(inten, dtype=np.float32)
    return xyz, None

def load_pcd_xyzI(path: str):
    """PCD에서 (xyz, intensity[선택]) 반환"""
    if HAS_O3D:
        try:
            pcd = o3d.io.read_point_cloud(path)
            xyz = np.asarray(pcd.points, dtype=np.float32)
            # Open3D의 intensity는 대부분 별도 채널로 안 올 때가 많음 → ASCII 파서로 보완
            xyz2, inten2 = read_pcd_points_ascii(path)
            if xyz2.shape[0] == xyz.shape[0] and inten2 is not None:
                return xyz2, inten2
            return xyz, None
        except Exception:
            return read_pcd_points_ascii(path)
    else:
        return read_pcd_points_ascii(path)

def draw_grid(img, xy_range, step=2.0, labels=False):
    """xy_range=(xmin,xmax,ymin,ymax), y up 좌표계 기준 그리드 그리기"""
    h, w = img.shape[:2]
    xmin, xmax, ymin, ymax = xy_range
    # x축(수평), y축(수직)
    cx = int(np.interp(0, [xmin, xmax], [0, w-1]))
    cy = int(np.interp(0, [ymax, ymin], [0, h-1]))  # y up이므로 반전
    cv2.line(img, (0, cy), (w-1, cy), (80, 80, 80), 1)  # x-axis
    cv2.line(img, (cx, 0), (cx, h-1), (80, 80, 80), 1)  # y-axis

    # 눈금선
    if step > 0:
        # 수평선: y = k*step
        k = np.ceil(ymin / step)
        while k*step <= ymax:
            y = k*step
            py = int(np.interp(y, [ymax, ymin], [0, h-1]))
            cv2.line(img, (0, py), (w-1, py), (40, 40, 40), 1)
            if labels:
                cv2.putText(img, f"y={y:.0f}", (5, max(12, py-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160,160,160), 1, cv2.LINE_AA)
            k += 1
        # 수직선: x = k*step
        k = np.ceil(xmin / step)
        while k*step <= xmax:
            x = k*step
            px = int(np.interp(x, [xmin, xmax], [0, w-1]))
            cv2.line(img, (px, 0), (px, h-1), (40, 40, 40), 1)
            if labels:
                cv2.putText(img, f"x={x:.0f}", (px+2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160,160,160), 1, cv2.LINE_AA)
            k += 1

def colormap_from_name(name: str):
    name = (name or "").lower()
    if name == "turbo" and hasattr(cv2, "COLORMAP_TURBO"):
        return cv2.COLORMAP_TURBO
    maps = {
        "jet": cv2.COLORMAP_JET,
        "viridis": getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_JET),
        "hot": cv2.COLORMAP_HOT,
        "gray": cv2.COLORMAP_BONE,
    }
    return maps.get(name, cv2.COLORMAP_JET)


def lidar_topdown_image(
    pts_xyz,
    size=(640, 480),
    xy_range=(-10, 10, -10, 10),
    point_size=1,
    use_intensity=False,
    intensity_vals=None,      # ← 이 이름만 씀 (intens 같은 거 쓰지 말기)
    intensity_cmap='turbo',
    draw_grid=False,
    grid_step=2,
    draw_labels=False,
    auto_range=True,                 # ← 추가
    auto_range_percentile=(1, 99),   # ← 추가
    manual_vmin_vmax=None,           # ← 추가
):
    import numpy as np, cv2, math
    from matplotlib import cm

    W, H = size
    x_min, x_max, y_min, y_max = xy_range
    bev = np.zeros((H, W, 3), dtype=np.uint8)
    if pts_xyz is None or len(pts_xyz) == 0:
        return bev

    x, y, z = pts_xyz[:,0], pts_xyz[:,1], pts_xyz[:,2]
    mask = (x>=x_min)&(x<=x_max)&(y>=y_min)&(y<=y_max)
    if not np.any(mask):
        return bev
    x, y, z = x[mask], y[mask], z[mask]

    # >>> 여기 핵심 패치: intensity를 쓸 때도 변수명은 intensity_vals로 통일
    if use_intensity and intensity_vals is not None and len(intensity_vals) == len(pts_xyz):
        scalars = intensity_vals[mask]
    else:
        scalars = z.copy()  # fallback

    # 색 범위 설정
    if manual_vmin_vmax is not None:
        vmin, vmax = manual_vmin_vmax
    elif auto_range:
        p_low, p_high = auto_range_percentile
        vmin = float(np.percentile(scalars, p_low))
        vmax = float(np.percentile(scalars, p_high))
        if math.isclose(vmax, vmin):
            vmax = vmin + 1e-6
    else:
        vmin, vmax = float(np.min(scalars)), float(np.max(scalars))
        if math.isclose(vmax, vmin):
            vmax = vmin + 1e-6

    # 정규화 → 컬러맵
    scal = np.clip((scalars - vmin) / (vmax - vmin + 1e-12), 0, 1)
    colors = (cm.get_cmap(intensity_cmap)(scal)[..., :3] * 255).astype(np.uint8)

    # 월드→픽셀
    u = ((x - x_min) / (x_max - x_min) * (W - 1)).astype(np.int32)
    v = ((1.0 - (y - y_min) / (y_max - y_min)) * (H - 1)).astype(np.int32)

    # 점 찍기
    if point_size == 1:
        bev[v, u] = colors
    else:
        r = max(1, point_size // 2)
        for uu, vv, c in zip(u, v, colors):
            bev[max(0,vv-r):min(H,vv+r+1), max(0,uu-r):min(W,uu+r+1)] = c

    if draw_grid and grid_step > 0:
        for yt in np.arange(math.ceil(y_min/grid_step)*grid_step, y_max+1e-6, grid_step):
            vv = int((1.0 - (yt - y_min)/(y_max - y_min))*(H-1))
            cv2.line(bev, (0,vv), (W-1,vv), (60,60,60), 1, cv2.LINE_AA)
        for xt in np.arange(math.ceil(x_min/grid_step)*grid_step, x_max+1e-6, grid_step):
            uu = int(((xt - x_min)/(x_max - x_min))*(W-1))
            cv2.line(bev, (uu,0), (uu,H-1), (60,60,60), 1, cv2.LINE_AA)

    if draw_labels:
        txt = f"[{x_min},{x_max}]x,[{y_min},{y_max}]y  {'I' if use_intensity else 'Z'}  v=({vmin:.2f},{vmax:.2f})"
        cv2.putText(bev, txt, (8,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)

    return bev

def place_side_by_side(rgb, lidar, offset_x=50, scale=1.5):
    # lidar 크기 키우기
    H, W = rgb.shape[:2]
    h2 = int(lidar.shape[0] * scale)
    w2 = int(lidar.shape[1] * scale)
    lidar_resized = cv2.resize(lidar, (w2, h2))

    # 캔버스 만들기 (RGB 폭 + lidar 폭 + 여백)
    canvas = np.zeros((max(H, h2), W + w2 + offset_x, 3), dtype=np.uint8)
    canvas[:H, :W] = rgb
    canvas[:h2, W+offset_x:W+offset_x+w2] = lidar_resized
    return canvas

def main():
    ap = argparse.ArgumentParser(description="Make side-by-side RGB + LiDAR top-down MP4")
    ap.add_argument("--root", default=DEF_ROOT, help="Dataset root directory")
    ap.add_argument("--rgb-sub", default=DEF_RGB_SUB, help="RGB subdirectory under root")
    ap.add_argument("--pcd-sub", default=DEF_PCD_SUB, help="PCD subdirectory under root")
    ap.add_argument("--yolo-sub", default="", help="RGB 대신 사용할 오버레이 폴더명(예: 'RGB-D(Image)_overlay')")
    ap.add_argument("--out", default=DEF_OUT, help="Output MP4 filename (relative -> under root)")
    ap.add_argument("--fps", type=float, default=10.0, help="Output FPS")
    ap.add_argument("--view-size", type=int, nargs=2, default=(640, 480), metavar=("W", "H"), help="Top-down view size (right pane)")
    ap.add_argument("--xy-range", type=float, nargs=4, default=(-10, 10, -10, 10), metavar=("XMIN", "XMAX", "YMIN", "YMAX"), help="Manual XY crop range in meters")
    ap.add_argument("--index-digits", type=int, default=4, help="Index digits in filenames (e.g., 4 for 0001)")

    # 자동 범위 / EMA / 패딩
    ap.add_argument("--auto-range", action="store_true", help="LiDAR x/y 분포로 프레임마다 범위를 자동 산정")
    ap.add_argument("--auto-range-percentile", type=float, nargs=2, default=(1.0, 99.0), metavar=("PLOW","PHI"), help="자동 범위 산정시 백분위수")
    ap.add_argument("--range-ema", type=float, default=0.8, help="자동 범위 지수평활 계수(0=즉시 반영, 0.8 권장)")
    ap.add_argument("--range-pad", type=float, default=1.0, help="자동 범위 padding (m)")

    # 렌더 옵션
    ap.add_argument("--grid", action="store_true", help="그리드/축 표시")
    ap.add_argument("--grid-step", type=float, default=2.0, help="그리드 간격(m)")
    ap.add_argument("--labels", action="store_true", help="그리드 라벨 텍스트 표시")
    ap.add_argument("--point-size", type=int, default=1, help="포인트 반지름(픽셀)")
    ap.add_argument("--intensity", action="store_true", help="PCD intensity 컬러맵 사용(가능할 때)")
    ap.add_argument("--intensity-cmap", type=str, default="turbo", help="강도 컬러맵 (turbo/jet/viridis/hot/gray)")
    ap.add_argument("--intensity-percentile", type=float, nargs=2, default=(1.0, 99.0), metavar=("PLOW","PHI"), help="강도 정규화에 사용할 백분위수")
    ap.add_argument("--watermark", action="store_true", help="좌하단 워터마크 표시")
    ap.add_argument("--bev-shift-x", type=float, default=0.0,
                    help="BEV 창을 x축으로 이동(m). +이면 점들이 화면에서 왼쪽으로 보임")
    ap.add_argument("--bev-zoom", type=float, default=1.0,
                    help="BEV 줌 배율(>1이면 확대; 중심 크롭)")

    args = ap.parse_args()

    root = os.path.abspath(args.root)
    rgb_dir = os.path.join(root, (args.yolo_sub or args.rgb_sub))
    pcd_dir = os.path.join(root, args.pcd_sub)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)

    base, ext = os.path.splitext(out_path)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{base}_{ts}{ext}"

    # 입력 수집
    rgb_list = list_indexed(rgb_dir, exts=(".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"))
    pcd_list = []
    rx = re.compile(rf".*_(\d{{{args.index_digits}}})\.pcd$")
    for p in glob.glob(os.path.join(pcd_dir, "*.pcd")):
        m = rx.match(p)
        if m:
            try:
                pidx = int(m.group(1))
                pcd_list.append((pidx, p))
            except Exception:
                pass
    pcd_list.sort(key=lambda t: t[0])

    if not rgb_list or not pcd_list:
        print("RGB 또는 PCD가 없습니다.")
        print(" RGB dir:", rgb_dir)
        print(" PCD dir:", pcd_dir)
        return

    rgb_dict = dict(rgb_list)
    pcd_dict = dict(pcd_list)
    common_ids = sorted(set(rgb_dict.keys()) & set(pcd_dict.keys()))
    if not common_ids:
        print("인덱스가 매칭되는 RGB/PCD가 없습니다.")
        return

    # 출력 준비
    sample = cv2.imread(rgb_dict[common_ids[0]], cv2.IMREAD_COLOR)
    if sample is None:
        print("RGB 샘플을 읽을 수 없습니다.")
        return
    h, w = sample.shape[:2]
    right0 = np.zeros((args.view_size[1], args.view_size[0], 3), np.uint8)
    out_size = (w + right0.shape[1], max(h, right0.shape[0]))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(args.fps), out_size, True)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {out_path}")

    # 자동 범위 EMA 상태
    ema_range = None  # np.array([xmin,xmax,ymin,ymax])

    # 루프
    for idx in common_ids:
        rgb = cv2.imread(rgb_dict[idx], cv2.IMREAD_COLOR)
        if rgb is None:
            continue

        pts, inten = load_pcd_xyzI(pcd_dict[idx])

        # XY 범위 결정
        if args.auto_range and pts.size:
            x, y = pts[:, 0], pts[:, 1]
            plow, phi = args.auto_range_percentile
            # 안전 보정
            lo = float(np.clip(min(plow, phi), 0.0, 100.0))
            hi = float(np.clip(max(plow, phi), 0.0, 100.0))
            if hi <= lo:
                hi = min(100.0, lo + 0.1)

            xmin = float(np.percentile(x, lo)); xmax = float(np.percentile(x, hi))
            ymin = float(np.percentile(y, lo)); ymax = float(np.percentile(y, hi))
            if xmax <= xmin: xmax = xmin + 1e-3
            if ymax <= ymin: ymax = ymin + 1e-3

            pad = float(args.range_pad)
            curr = np.array([xmin - pad, xmax + pad, ymin - pad, ymax + pad], dtype=np.float32)

            a = float(args.range_ema)
            if ema_range is None or a <= 0.0:
                ema_range = curr
            else:
                ema_range = a * ema_range + (1.0 - a) * curr
            xy_range = tuple(map(float, ema_range))
        else:
            xy_range = tuple(args.xy_range)

        # ---- BEV 보정: 왼쪽으로 밀기(shift) + 확대(zoom)
        xmin, xmax, ymin, ymax = xy_range
        # shift: 창을 +dx 만큼 오른쪽으로 옮기면 점들이 화면상 왼쪽으로 이동해 보임
        dx = float(args.bev_shift_x)
        xmin, xmax = xmin + dx, xmax + dx

        # zoom: 중심을 유지한 채 창을 1/zoom 만큼 좁혀 확대 효과
        zoom = max(1e-6, float(args.bev_zoom))
        if zoom != 1.0:
            cx, cy = 0.5*(xmin+xmax), 0.5*(ymin+ymax)
            hx = 0.5*(xmax-xmin)/zoom
            hy = 0.5*(ymax-ymin)/zoom
            xmin, xmax = cx - hx, cx + hx
            ymin, ymax = cy - hy, cy + hy
        # 너무 좁으면 포인트 다 날아가니 최소 폭/높이 보장
        min_span = 2e-1  # 0.2 m
        if (xmax - xmin) < min_span:
            c = 0.5*(xmin+xmax); xmin, xmax = c - min_span/2, c + min_span/2
        if (ymax - ymin) < min_span:
            c = 0.5*(ymin+ymax); ymin, ymax = c - min_span/2, c + min_span/2
        xy_range = (xmin, xmax, ymin, ymax)

        # 라이다 탑뷰
        top = lidar_topdown_image(
            # 라이다 탑뷰  (자동/수동 xy_range 반영)
            pts_xyz=pts,
            size=(args.view_size[0]*2, args.view_size[1]*2),
            xy_range=xy_range,   # ← 위에서 계산한 범위 그대로 사용
            point_size=args.point_size,
            use_intensity=args.intensity,
            intensity_vals=inten,          # ← 실제 intensity 배열 전달
            intensity_cmap=args.intensity_cmap,
            draw_grid=args.grid,
            grid_step=args.grid_step,
            draw_labels=args.labels,
            auto_range=True,
            auto_range_percentile=(
                args.intensity_percentile[0], args.intensity_percentile[1]
            ) if args.intensity else (1.0, 99.0),
        )

        # 높이 맞추고 우측 뷰 폭을 요청 크기로
        if top.shape[0] != rgb.shape[0]:
            scale = rgb.shape[0] / max(1, top.shape[0])
            top = cv2.resize(top, (int(top.shape[1] * scale), rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        desired_w = args.view_size[0]
        if top.shape[1] != desired_w or top.shape[0] != rgb.shape[0]:
            canvas = np.zeros((rgb.shape[0], desired_w, 3), np.uint8)
            tw = min(canvas.shape[1], top.shape[1])
            canvas[:, :tw] = top[:, :tw]
            top = canvas

        # 디버그 텍스트
        n_total = int(pts.shape[0]) if pts is not None else 0
        # lidar_topdown_image 안에서 쓴 것과 동일한 마스크로 개수 구하기
        xmin, xmax, ymin, ymax = xy_range
        if n_total:
            m = (pts[:,0]>=xmin)&(pts[:,0]<=xmax)&(pts[:,1]>=ymin)&(pts[:,1]<=ymax)
            n_in = int(np.count_nonzero(m))
        else:
            n_in = 0
        combo = cv2.hconcat([rgb, top])
        dbg = f"x[{xmin:.2f},{xmax:.2f}] y[{ymin:.2f},{ymax:.2f}]  pts {n_in}/{n_total}"
        cv2.putText(combo, dbg, (combo.shape[1]-top.shape[1]+8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

        # 좌상단 인덱스, (옵션) 워터마크
        cv2.putText(combo, f"idx {idx:0{args.index_digits}d}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        if args.watermark:
            txt = "RGB | LiDAR Top-Down"
            cv2.putText(combo, txt, (10, combo.shape[0]-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)

        writer.write(combo)

    writer.release()
    print("DONE:", out_path)

if __name__ == "__main__":
    main()