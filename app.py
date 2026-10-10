
import io
from collections import defaultdict
from functools import lru_cache

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(
    page_title="PixelTrace — Image Forensics",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# DESIGN SYSTEM — warm white, black, subtle gray
# --------------------------------------------------
st.markdown("""
<style>
:root {
    --paper: #f8f8f6;
    --white: #ffffff;
    --ink: #191919;
    --muted: #777772;
    --line: #e4e4df;
    --soft: #efefeb;
    --accent: #242424;
}
.stApp {
    background: var(--paper);
    color: var(--ink);
    font-family: Inter, -apple-system, BlinkMacSystemFont,
                 "Segoe UI", sans-serif;
}
.block-container {
    max-width: 1440px;
    padding: 1.5rem 2.5rem 3rem;
}
h1, h2, h3, h4 {
    color: var(--ink) !important;
    letter-spacing: -0.045em !important;
}
h1 {
    font-size: clamp(2.2rem, 4vw, 3.7rem) !important;
    font-weight: 650 !important;
    line-height: 1.06 !important;
}
h2 { font-size: 1.65rem !important; font-weight: 620 !important; }
h3 { font-size: 1.1rem !important; font-weight: 620 !important; }
p, label { line-height: 1.65; }
[data-testid="stCaptionContainer"] {
    color: var(--muted);
}
[data-testid="stSidebar"] {
    background: #f2f2ee;
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}
.stButton > button, .stDownloadButton > button {
    min-height: 43px;
    border: 1px solid #d6d6d0;
    border-radius: 5px;
    background: white;
    color: #222;
    font-size: 0.9rem;
    font-weight: 550;
    box-shadow: none;
    transition: background .15s ease, border-color .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #efefeb;
    border-color: #aaa9a2;
    color: #111;
}
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
    background: #202020;
    border-color: #202020;
    color: white;
}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover {
    background: #414141;
    border-color: #414141;
    color: white;
}
[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--line);
    border-radius: 5px;
    padding: 1.1rem;
    box-shadow: none;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: .8rem !important;
}
[data-testid="stMetricValue"] {
    color: var(--ink) !important;
    font-weight: 620 !important;
    letter-spacing: -.04em;
}
[data-testid="stFileUploader"] {
    background: white;
    border: 1px dashed #c9c9c2;
    border-radius: 5px;
    padding: .75rem;
}
[data-testid="stImage"] img {
    border-radius: 3px;
    border: 1px solid var(--line);
}
button[data-baseweb="tab"] {
    font-weight: 550;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #111;
}
[data-testid="stExpander"] {
    background: white;
    border: 1px solid var(--line);
    border-radius: 5px;
}
[data-testid="stAlert"] {
    border-radius: 5px;
}
hr {
    border-color: var(--line);
}
footer { visibility: hidden; }

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 0 18px;
    margin-bottom: 54px;
    border-bottom: 1px solid var(--line);
}
.wordmark {
    font-size: 15px;
    font-weight: 800;
    letter-spacing: -.04em;
    color: #181818;
}
.wordmark span {
    font-weight: 450;
    color: #777772;
}
.topbar-meta {
    color: #777772;
    font-size: 11px;
    letter-spacing: .08em;
    text-transform: uppercase;
}
.hero-label {
    margin-bottom: 17px;
    color: #777772;
    font-size: 11px;
    font-weight: 650;
    letter-spacing: .13em;
    text-transform: uppercase;
}
.hero-copy {
    max-width: 670px;
    margin: 18px 0 26px;
    color: #666660;
    font-size: 16px;
    line-height: 1.8;
}
.hero-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 30px;
}
.hero-meta span {
    padding: 7px 10px;
    border: 1px solid #deded8;
    background: transparent;
    color: #555550;
    font-size: 10px;
    letter-spacing: .055em;
}
.section-label {
    margin: 34px 0 8px;
    color: #777772;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
}
.panel {
    padding: 17px;
    margin-bottom: 10px;
    background: white;
    border: 1px solid var(--line);
    border-radius: 5px;
}
.panel-title {
    color: #202020;
    font-size: 14px;
    font-weight: 650;
}
.panel-copy {
    margin-top: 5px;
    color: #777772;
    font-size: 12px;
    line-height: 1.65;
}
.notice {
    margin: 28px 0;
    padding: 14px 16px;
    border-left: 2px solid #777772;
    background: #f0f0ec;
    color: #5f5f59;
    font-size: 12px;
    line-height: 1.7;
}
.result-banner {
    padding: 15px 17px;
    margin: 18px 0;
    background: white;
    border: 1px solid var(--line);
    border-left: 3px solid #363636;
    color: #393939;
    font-size: 13px;
    line-height: 1.7;
}
.footer-note {
    padding-top: 14px;
    border-top: 1px solid var(--line);
    color: #888880;
    font-size: 10px;
    letter-spacing: .035em;
}
@media (max-width: 700px) {
    .block-container { padding: 1rem .8rem 2rem; }
    .topbar { margin-bottom: 32px; }
    .topbar-meta { display: none; }
    h1 { font-size: 2.35rem !important; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        transition: none !important;
        animation: none !important;
    }
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="topbar">
    <div class="wordmark">PIXELTRACE <span>/ IMAGE FORENSICS</span></div>
    <div class="topbar-meta">Digital Image Processing · Research Tool</div>
</div>
<div class="hero-label">Image integrity / 001</div>
<h1>Look closer.<br>Find what was copied.</h1>
<div class="hero-copy">
    An image-forensics tool for identifying potentially duplicated
    regions in digital images using traditional image processing
    and Discrete Cosine Transform (DCT) block matching.
</div>
<div class="hero-meta">
    <span>TRADITIONAL DIP</span>
    <span>DCT BLOCK MATCHING</span>
    <span>NO AI / ML / DL / CNN</span>
</div>
<div class="notice">
    <strong>About the results.</strong>
    Highlighted areas indicate potential matching regions, not
    definitive proof of manipulation. Natural repetition and textures
    can produce false positives. Results require human interpretation.
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# ORIGINAL DEMO IMAGE GENERATOR — UNCHANGED
# --------------------------------------------------
def make_demo_image():
    """Create a deterministic synthetic scene with a copied patch for repeatable testing."""
    rng = np.random.default_rng(17)
    h, w = 360, 520
    noise = rng.normal(0, 10, (h, w, 3)).astype(np.int16)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.zeros((h, w, 3), dtype=np.int16)
    base[:, :, 0] = 95 + (yy * 35 // h)
    base[:, :, 1] = 145 + (xx * 30 // w)
    base[:, :, 2] = 75 + (yy * 20 // h)
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle(
        (55, 90, 175, 210), radius=8,
        fill=(225, 185, 65), outline=(35, 35, 35), width=4
    )
    d.rectangle(
        (68, 103, 162, 197),
        fill=(40, 90, 155), outline=(245, 245, 245), width=3
    )
    for i in range(5):
        d.line(
            (73, 112 + i * 15, 157, 112 + i * 15),
            fill=(240, 220, 120), width=4
        )
    d.ellipse(
        (91, 130, 139, 178),
        fill=(220, 65, 55), outline=(255, 255, 255), width=3
    )
    d.line((115, 134, 126, 120), fill=(35, 100, 40), width=5)
    for cx, cy, r, color in [
        (260, 100, 24, (235, 220, 120)),
        (350, 260, 32, (70, 120, 60)),
        (450, 100, 18, (190, 90, 60)),
    ]:
        d.ellipse(
            (cx-r, cy-r, cx+r, cy+r),
            fill=color, outline=(35, 35, 35), width=3
        )
    d.rectangle((220, 285, 500, 305), fill=(95, 70, 45))
    patch = im.crop((55, 90, 175, 210))
    im.paste(patch, (311, 114))
    return im


# --------------------------------------------------
# DETECTION ALGORITHM — ADAPTED FROM alperencubuk/copy-move-forgery-detection
# --------------------------------------------------
#
# Source: forgery_detection.py in that repository (DCT block matching with
# lexicographic sorting and shift-vector elimination).
#
# What is kept from the repository (values copied from its source):
#   * grayscale conversion, sliding block scan, orthonormal block DCT
#     (cv2.dct scaling: float32 block / 255 -> DCT -> * 255)
#   * 8-bit conversion of the coefficients, zig-zag scan, first 16
#     coefficients, quantization = floor(value / 16)
#   * lexicographic sort of the 16-value descriptors (ties broken by block
#     row, then column)
#   * each block is compared with the next 9 blocks in sorted order
#   * similarity: Euclidean descriptor distance <= 5
#   * spatial filter: Euclidean block distance >= min distance
#   * shift-vector elimination: keep a pair only if its shift vector occurs
#     at least 20 times
#   * localization: both blocks of each surviving pair are painted
#     (block_size - 1) x (block_size - 1) pixels, as the repository paints
#     7 x 7 pixels for its 8 x 8 blocks
#
# What is adapted / differs (these are deliberate and are documented in the
# technical report):
#   * Input is an in-memory RGB array, not cv2.imread(); no cv2.imshow(),
#     no ground-truth mask and no accuracy computation (an uploaded image
#     has no ground truth).
#   * np.float (removed from NumPy) is not used.
#   * Performance: the repository computes one cv2.dct() per block in Python
#     loops and an O(m^2) elimination loop. Here the same DCT coefficients
#     are obtained for every block position at once by correlating the
#     image with the 16 DCT basis kernels (cv2.filter2D), the descriptor is
#     packed into one 64-bit integer for sorting, neighbour comparison is
#     vectorized, and shift vectors are counted with np.unique. This is
#     mathematically the same pipeline, but floating-point rounding can make
#     a small fraction of quantized coefficients differ by one bin from the
#     repository's per-block cv2.dct() output.
#   * The repository's elimination step deletes rows through stale indices
#     (np.delete inside a loop) and counts vectors with an O(m^2) loop. Here
#     the intended rule is implemented directly ("keep pairs whose shift
#     vector appears >= 20 times"), so results can differ slightly.
#   * The repository's scan loops stop one block position early; here every
#     valid block position is used.
#   * Block size is taken from the sidebar (the repository uses 8). The
#     repository's other constants are kept unchanged.
#   * The sidebar "Block stride" cannot be honoured in a way that keeps the
#     repository's behaviour: the repository scans EVERY pixel position
#     (stride 1). With stride > 1 a copied region is only found when its
#     shift is a multiple of the stride. The detector therefore always scans
#     at stride 1 and the stride value is accepted but not used.
#   * Images with a long side above _CM_MAX_SIDE are downscaled before
#     analysis to keep memory and run time reasonable; resampling can
#     weaken exact-copy matches. The minimum distance is scaled with it.
#   * Very flat blocks (grey-level standard deviation below
#     _CM_MIN_BLOCK_STD) are skipped. The repository has no such filter;
#     without it large uniform areas (sky, walls) match each other.
#
# Only a short rationale is given for each constant; the repository's own
# values are marked "repo".

_CM_QUANTIZATION = 16      # repo: quantization
_CM_FEATURES = 16          # repo: first 16 zig-zag coefficients
_CM_SIMILARITY = 5.0       # repo: tsimilarity
_CM_NEIGHBOURS = 9         # repo: j in range(i + 1, i + 10)
_CM_VECTOR_LIMIT = 20      # repo: vector_limit
_CM_MAX_SIDE = 1024        # adaptation: performance cap on the long side
_CM_MIN_BLOCK_STD = 8.0    # adaptation: skip flat blocks (same threshold as the original PixelTrace detector)


def _zigzag_order(n):
    """Zig-zag scan order exactly as built in the repository."""
    order = []
    for d in range(2 * n - 1):
        cells = [(k, d - k) for k in range(n) if 0 <= d - k < n]
        if d % 2 == 0:
            cells.reverse()
        order.extend(cells)
    return order


@lru_cache(maxsize=None)
def _dct_kernels(n):
    """
    Kernels K[k] such that sum(block * K[k]) equals the k-th zig-zag
    coefficient of cv2.dct(block). Built from cv2.dct itself (the DCT is
    linear), so the scaling is identical to the repository's.
    """
    rows, cols = zip(*_zigzag_order(n)[:_CM_FEATURES])
    kernels = np.empty((_CM_FEATURES, n, n), dtype=np.float32)
    for y in range(n):
        for x in range(n):
            impulse = np.zeros((n, n), dtype=np.float32)
            impulse[y, x] = 1.0
            kernels[:, y, x] = cv2.dct(impulse)[rows, cols]
    return kernels


def _wrap_uint8(values):
    """
    Emulates the repository's np.uint8(float32 array) conversion: truncate
    toward zero, then keep the low 8 bits (what the C cast does on x86).
    Made explicit because the raw cast is undefined for out-of-range
    values and differs between CPU architectures / NumPy versions.
    """
    return (np.trunc(values).astype(np.int32) & 0xFF).astype(np.uint8)


def _packed_descriptors(gray, n):
    """
    One 64-bit key per block position: the 16 quantized coefficients
    (each 0..15) packed as 4-bit fields, first coefficient in the top bits,
    so ordering the integers equals ordering the descriptors
    lexicographically. Returned shape: (H - n + 1, W - n + 1).
    """
    g = gray.astype(np.float32) / 255.0
    oh, ow = g.shape[0] - n + 1, g.shape[1] - n + 1
    kernels = _dct_kernels(n)

    packed = np.zeros((oh, ow), dtype=np.uint64)
    for k in range(_CM_FEATURES):
        response = cv2.filter2D(
            g, cv2.CV_32F, kernels[k],
            anchor=(0, 0), borderType=cv2.BORDER_REFLECT,
        )[:oh, :ow]
        quantized = _wrap_uint8(response * np.float32(255.0)) // _CM_QUANTIZATION
        packed = (packed << np.uint64(4)) | quantized.astype(np.uint64)
    return packed


def _block_std(gray, n):
    """Grey-level standard deviation of every n x n block position."""
    g = gray.astype(np.float64)
    oh, ow = g.shape[0] - n + 1, g.shape[1] - n + 1
    mean = cv2.boxFilter(g, -1, (n, n), anchor=(0, 0),
                         normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(g * g, -1, (n, n), anchor=(0, 0),
                            normalize=True, borderType=cv2.BORDER_REFLECT)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))[:oh, :ow]


def _match_blocks(gray, n, min_distance, min_std=_CM_MIN_BLOCK_STD):
    """
    Core of the repository algorithm on a grayscale array.

    Returns (block_count, candidate_count, selected_count, hit_map) where
    hit_map is a uint8 (H - n + 1, W - n + 1) array marking block positions
    that belong to a selected pair.
    """
    oh, ow = gray.shape[0] - n + 1, gray.shape[1] - n + 1

    packed = _packed_descriptors(gray, n).ravel()
    keep = np.flatnonzero(_block_std(gray, n).ravel() >= min_std)
    block_count = int(keep.size)
    hit_map = np.zeros((oh, ow), dtype=np.uint8)

    if block_count < 2:
        return block_count, 0, 0, hit_map

    # Lexicographic sort. `keep` is in row-major order and the sort is
    # stable, so ties are ordered by row then column (as in the repository,
    # where the coordinates are the last sort keys).
    order = np.argsort(packed[keep], kind="stable")
    position = keep[order]                      # flat block index, sorted
    keys = packed[position]
    rows = (position // ow).astype(np.int32)
    cols = (position % ow).astype(np.int32)

    shifts = np.uint64(60) - np.uint64(4) * np.arange(
        _CM_FEATURES, dtype=np.uint64
    )
    features = ((keys[:, None] >> shifts) & np.uint64(15)).astype(np.int32)

    # Compare every block with the next _CM_NEIGHBOURS blocks.
    min_d2 = int(min_distance) ** 2
    first, second, vec_y, vec_x = [], [], [], []

    for offset in range(1, _CM_NEIGHBOURS + 1):
        if block_count <= offset:
            break
        diff = features[:-offset] - features[offset:]
        similar = np.einsum("ij,ij->i", diff, diff) <= _CM_SIMILARITY ** 2
        i = np.flatnonzero(similar)
        if i.size == 0:
            continue
        j = i + offset
        dy = rows[j] - rows[i]
        dx = cols[j] - cols[i]
        far = dy.astype(np.int64) ** 2 + dx.astype(np.int64) ** 2 >= min_d2
        first.append(i[far])
        second.append(j[far])
        vec_y.append(dy[far])
        vec_x.append(dx[far])

    if not first:
        return block_count, 0, 0, hit_map

    first = np.concatenate(first)
    second = np.concatenate(second)
    vec_y = np.concatenate(vec_y).astype(np.int64)
    vec_x = np.concatenate(vec_x).astype(np.int64)
    candidate_count = int(first.size)

    if candidate_count == 0:
        return block_count, 0, 0, hit_map

    # Shift-vector elimination: keep pairs whose vector occurs at least
    # _CM_VECTOR_LIMIT times (the repository's vector is signed).
    vector_key = (vec_y + oh) * (2 * ow + 1) + (vec_x + ow)
    _, inverse, counts = np.unique(
        vector_key, return_inverse=True, return_counts=True
    )
    selected = counts[inverse.ravel()] >= _CM_VECTOR_LIMIT
    selected_count = int(selected.sum())

    if selected_count:
        flat = hit_map.reshape(-1)
        flat[position[first[selected]]] = 1
        flat[position[second[selected]]] = 1

    return block_count, candidate_count, selected_count, hit_map


def detect_copy_move(
    rgb, block_size=16, stride=8, min_distance=40, threshold=0.06
):
    """
    Copy-move detector adapted from alperencubuk/copy-move-forgery-detection
    (see the notes above for exactly what was kept and what was adapted).

    `stride` and `threshold` are accepted for interface compatibility only:
    the repository scans every pixel position and has no threshold of that
    kind.

    Returns:
        marked, mask, block_count, candidate_count, selected_count
    """
    if rgb is None:
        raise ValueError("No image data was provided.")
    rgb = np.asarray(rgb)
    if rgb.ndim not in (2, 3) or rgb.size == 0:
        raise ValueError("The image is empty or has an unsupported shape.")
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    elif rgb.shape[2] == 1:
        rgb = np.repeat(rgb, 3, axis=2)
    elif rgb.shape[2] >= 4:
        rgb = rgb[:, :, :3]
    elif rgb.shape[2] != 3:
        raise ValueError("The image must have 1, 3 or 4 channels.")
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

    n = int(block_size)
    h0, w0 = rgb.shape[:2]
    if h0 < n or w0 < n:
        raise ValueError(f"Image is too small for {n}x{n} blocks.")

    # Keep processing practical on ordinary laptops.
    scale = min(1.0, float(_CM_MAX_SIDE) / max(h0, w0))
    if scale < 1.0:
        working = cv2.resize(
            rgb,
            (max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        working = rgb

    gray = cv2.cvtColor(working, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    if h < n or w < n:
        raise ValueError(f"Image is too small for {n}x{n} blocks.")

    working_distance = max(1, int(round(min_distance * scale)))
    block_count, candidate_count, selected_count, hit_map = _match_blocks(
        gray, n, working_distance
    )

    # Paint (n - 1) x (n - 1) pixels from every flagged block origin.
    mask_small = np.zeros((h, w), dtype=np.uint8)
    if selected_count:
        canvas = np.zeros((h, w), dtype=np.uint8)
        canvas[:hit_map.shape[0], :hit_map.shape[1]] = hit_map
        k = max(1, n - 1)
        mask_small = cv2.dilate(
            canvas, np.ones((k, k), dtype=np.uint8), anchor=(k - 1, k - 1)
        ) * np.uint8(255)

    # Restore the mask to the original image dimensions.
    if (h, w) != (h0, w0):
        mask = cv2.resize(
            mask_small, (w0, h0), interpolation=cv2.INTER_NEAREST
        )
    else:
        mask = mask_small

    # Preserve PixelTrace's existing red overlay.
    marked = rgb.copy()

    if np.any(mask):
        flagged = mask > 0
        marked[flagged] = (
            0.55 * marked[flagged].astype(np.float32)
            + 0.45 * np.array([255, 0, 0], dtype=np.float32)
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            marked,
            contours,
            -1,
            (255, 30, 30),
            2,
        )

    return (
        marked,
        mask,
        block_count,
        candidate_count,
        selected_count,
    )


# --------------------------------------------------
# SIDEBAR — SAME ANALYSIS CONTROLS
# --------------------------------------------------
with st.sidebar:
    st.markdown("### PixelTrace")
    st.caption("Analysis settings")
    st.divider()

    st.markdown("**01 / Block analysis**")
    block_size = st.select_slider(
        "Block size (pixels)",
        options=[8, 12, 16, 20, 24],
        value=16
    )
    stride = st.select_slider(
        "Block stride (pixels)",
        options=[4, 8, 12, 16],
        value=8
    )
    min_distance = st.slider(
        "Minimum match distance",
        min_value=16,
        max_value=100,
        value=40,
        step=8
    )

    st.divider()
    st.markdown("**02 / Method**")
    st.caption(
        "Grayscale → DCT descriptors → block matching → "
        "spatial localization."
    )
    st.caption(
        "Best suited to copied patches pasted without rotation or resizing."
    )


# --------------------------------------------------
# IMAGE SOURCE
# --------------------------------------------------
st.markdown(
    '<div class="section-label">01 — IMAGE SOURCE</div>',
    unsafe_allow_html=True
)
st.subheader("Choose an image to inspect")

demo_col, upload_col = st.columns([1, 1.5], gap="large")

if "demo_image" not in st.session_state:
    st.session_state.demo_image = None

with demo_col:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">Synthetic test image</div>
        <div class="panel-copy">
            A repeatable sample containing a deliberately copied region.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "Generate test image",
        type="primary",
        use_container_width=True
    ):
        st.session_state.demo_image = make_demo_image()

with upload_col:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">Upload an image</div>
        <div class="panel-copy">
            Choose a JPG, JPEG or PNG image from your computer.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Select image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )


# --------------------------------------------------
# LOAD SELECTED IMAGE
# --------------------------------------------------
rgb = None
source_name = None

if uploaded is not None:
    try:
        rgb = np.array(Image.open(uploaded).convert("RGB"))
        source_name = uploaded.name
    except Exception as e:
        st.error(f"Could not read this image: {e}")
elif st.session_state.demo_image is not None:
    rgb = np.array(st.session_state.demo_image.convert("RGB"))
    source_name = "Synthetic copy-move test image"


# --------------------------------------------------
# FORENSIC WORKSPACE — SAME RESULTS AND FEATURES
# --------------------------------------------------
if rgb is not None:
    st.divider()
    st.markdown(
        '<div class="section-label">02 — FORENSIC WORKSPACE</div>',
        unsafe_allow_html=True
    )
    st.subheader("Image analysis")
    st.caption(
        f"Source: {source_name}  ·  "
        f"Resolution: {rgb.shape[1]:,} × {rgb.shape[0]:,} pixels"
    )

    with st.spinner("Computing DCT descriptors and matching blocks..."):
        try:
            marked, mask, block_count, pair_count, selected_count = (
                detect_copy_move(
                    rgb,
                    block_size=block_size,
                    stride=stride,
                    min_distance=min_distance
                )
            )
        except Exception as e:
            st.error(str(e))
            st.stop()

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("**01 / Original image**")
        st.caption("Unmodified source image")
        st.image(rgb, use_container_width=True)

    with right:
        st.markdown("**02 / Detection overlay**")
        st.caption("Red indicates regions flagged by the detector")
        st.image(marked, use_container_width=True)

    st.markdown(
        '<div class="section-label">03 — ANALYSIS METRICS</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3, gap="medium")
    m1.metric("Blocks analyzed", f"{block_count:,}")
    m2.metric("Candidate matches", f"{pair_count:,}")
    m3.metric("Selected matching pairs", f"{selected_count:,}")

    if mask.any():
        st.markdown("""
        <div class="result-banner">
            <strong>Potential matching regions identified.</strong><br>
            Review the highlighted areas. This result is not definitive
            proof of image forgery.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning(
            "No matching regions were highlighted. Try another image or "
            "adjust the block size, stride, or minimum distance."
        )

    with st.expander("Inspect binary localization mask"):
        st.caption(
            "White pixels represent regions selected by the algorithm; "
            "black pixels are not flagged."
        )
        st.image(mask, clamp=True, use_container_width=True)

    output_buffer = io.BytesIO()
    Image.fromarray(marked).save(output_buffer, format="PNG")

    st.download_button(
        "Download annotated image (PNG)",
        data=output_buffer.getvalue(),
        file_name="copy_move_detection_result.png",
        mime="image/png",
        type="primary",
        use_container_width=True
    )

    st.markdown(
        '<div class="section-label">04 — TECHNICAL REPORT</div>',
        unsafe_allow_html=True
    )

    with st.expander("Algorithm workflow and interpretation"):
        st.markdown("""
**Detection pipeline**

1. **Preprocessing:** Convert RGB pixels to grayscale; downscale very large images (long side above 1,024 pixels).
2. **Block extraction:** Scan every overlapping block position (stride 1, as in the referenced implementation); nearly flat blocks are skipped.
3. **DCT feature extraction:** Compute the block Discrete Cosine Transform, convert to 8-bit values, zig-zag scan, and quantize the first 16 coefficients.
4. **Descriptor matching:** Sort the descriptors lexicographically and compare each block with the next 9 in sorted order; keep pairs with descriptor distance of 5 or less.
5. **Spatial filtering:** Exclude pairs closer than the minimum match distance, then keep only pairs whose displacement vector occurs at least 20 times.
6. **Localization:** Build a binary mask and overlay selected blocks in red.

**Interpretation and limitations**

- This is a traditional image-processing prototype, not an AI or machine-learning system.
- The detector is adapted from the open-source *alperencubuk/copy-move-forgery-detection* project; it is a re-implementation, not an exact reproduction. Differences include vectorized computation, a flat-block filter, a configurable block size and a downscaling limit.
- The block-stride setting has no effect: the referenced method scans every pixel position, and larger strides miss copies whose shift is not a multiple of the stride.
- No accuracy or confidence score is computed; an uploaded image has no ground truth to compare against.
- Repeated natural textures can generate false positives.
- Rotation, resizing, compression, or other transformations can reduce detection reliability.
- A highlighted area is a candidate for human review, not conclusive proof of manipulation.
""")

else:
    st.divider()
    st.markdown(
        '<div class="section-label">02 — AWAITING INPUT</div>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="panel">
        <div class="panel-title">Ready to start an analysis</div>
        <div class="panel-copy">
            Generate the synthetic test image or upload a JPG, JPEG or PNG.
            Start with the synthetic image for a repeatable classroom demonstration.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown("""
<div class="footer-note">
    PIXELTRACE &nbsp;·&nbsp; ACADEMIC DIGITAL IMAGE PROCESSING
    &nbsp;·&nbsp; DCT BLOCK MATCHING
    &nbsp;·&nbsp; RESULTS REQUIRE HUMAN INTERPRETATION
</div>
""", unsafe_allow_html=True)
