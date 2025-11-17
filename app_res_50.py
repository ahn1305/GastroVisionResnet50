# app.py
import streamlit as st
import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import matplotlib.cm as cm
import time

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="GastroVision", layout="wide", page_icon="🩺")

# -----------------------
# Simple auth (demo)
# -----------------------
CREDENTIALS = {"ashwin": "1234", "researcher": "abcd"}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align:center;'>🔐 GastroVision Login</h1>", unsafe_allow_html=True)
    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")
    login_btn = st.button("Login", use_container_width=True)
    if login_btn:
        if username in CREDENTIALS and CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome back, {username.capitalize()} 👋")
            st.rerun()
        else:
            st.error("Invalid credentials. Please try again.")
    st.stop()

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966485.png", width=72)
st.sidebar.title("GastroVision 🩺")
st.sidebar.caption(f"Logged in as: {st.session_state.username}")
if st.sidebar.button("🚪 Logout", use_container_width=True):
    st.session_state.logged_in = False
    st.rerun()

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.write(f"🔧 Device: `{device}`")

# -----------------------
# Class names (22)
# Ensure these match training order
# -----------------------
class_names = [
    'Accessory tools', 'Barrett’s esophagus', 'Blood in lumen', 'Cecum',
    'Colon diverticula', 'Colon polyps', 'Colorectal cancer', 'Duodenal bulb',
    'Dyed-lifted-polyps', 'Dyed-resection-margins', 'Esophagitis',
    'Gastric polyps', 'Gastroesophageal_junction_normal z-line',
    'Ileocecal valve', 'Mucosal inflammation large bowel',
    'Normal esophagus',
    'Normal mucosa and vascular pattern in the large bowel',
    'Normal stomach', 'Pylorus', 'Resected polyps',
    'Retroflex rectum', 'Ulcer'
]


# -----------------------
# Cached model loaders
# -----------------------
@st.cache_resource
def load_classifier(path="./model/best_model_epoch24.pth", device='cpu'):
    n_classes = len(class_names)
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, n_classes),
        nn.LogSoftmax(dim=1)
    )
    state = torch.load(path, map_location=device)
    # handle if checkpoint dict or state_dict
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# UNet class (same as training)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = DoubleConv(256 + 256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(128 + 128, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = DoubleConv(64 + 64, 64)

        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4)
        x = self.conv1(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x = self.conv2(torch.cat([x, x2], dim=1))

        x = self.up3(x)
        x = self.conv3(torch.cat([x, x1], dim=1))

        logits = self.outc(x)
        return torch.sigmoid(logits)

@st.cache_resource
def load_unet(path="./model/best_model.pth", device='cpu'):
    u = UNet(n_channels=3, n_classes=1)
    state = torch.load(path, map_location=device)
    # handle if dict with state key
    if isinstance(state, dict) and 'model_state_dict' in state:
        u.load_state_dict(state['model_state_dict'])
    else:
        u.load_state_dict(state)
    u.to(device)
    u.eval()
    return u

# -----------------------
# Transforms and helpers
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.4762, 0.3054, 0.2368],
                         [0.3345, 0.2407, 0.2164])
])

# UNet preprocessing size
UNET_SIZE = 256

def preprocess_for_unet(np_img):
    resized = cv2.resize(np_img, (UNET_SIZE, UNET_SIZE))
    tensor = torch.from_numpy(resized.transpose(2,0,1)).float().unsqueeze(0) / 255.0
    return resized, tensor

def denormalize_tensor(tensor, mean, std):
    t = tensor.clone()
    for i in range(t.shape[0]):
        t[i] = t[i] * std[i] + mean[i]
    return t

# -----------------------
# Grad-CAM (ResNet50)
# -----------------------
def generate_gradcam(model, img_tensor, target_layer_name='layer4'):
    """
    Input:
      - model: resnet model on device
      - img_tensor: 1xCxHxW tensor on same device with requires_grad False or True
    Returns cam (H,W) normalized 0..1 and predicted class index
    """
    model.eval()
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    # register hooks on last conv block conv3
    target = dict([*model.named_modules()]).get(target_layer_name)
    # for resnet, we want the final block's conv output; default target_layer_name='layer4' (module)
    # attach hooks to the last conv in that block if possible
    # find the deepest conv in target
    if target is None:
        # fallback to layer4[-1].conv3 if exists
        if hasattr(model, 'layer4'):
            target = model.layer4[-1].conv3
        else:
            raise RuntimeError("Could not find target layer for Grad-CAM.")
    # If user passed the block (layer4) we register hooks on its last conv
    # try to find conv inside target if it's a module list
    # If target is a module list, pick its last conv
    # We'll simply register on model.layer4[-1].conv3 which works for ResNet50
    h_fwd = model.layer4[-1].conv3.register_forward_hook(forward_hook)
    h_bwd = model.layer4[-1].conv3.register_full_backward_hook(backward_hook)

    # ensure gradients can be computed on input
    img_tensor = img_tensor.clone().detach().to(next(model.parameters()).device)
    img_tensor.requires_grad_()

    outputs = model(img_tensor)               # forward
    pred_class = int(torch.argmax(outputs, dim=1).item())
    score = outputs[0, pred_class]
    model.zero_grad()
    score.backward(retain_graph=False)

    if not gradients or not activations:
        # cleanup
        h_fwd.remove(); h_bwd.remove()
        raise RuntimeError("Gradients or activations not recorded for Grad-CAM.")

    grads = gradients[0].cpu().numpy()[0]    # C x H' x W'
    acts = activations[0].cpu().numpy()[0]   # C x H' x W'
    weights = np.mean(grads, axis=(1,2))     # C
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    h_fwd.remove(); h_bwd.remove()
    return cam, pred_class

# -----------------------
# Segmentation helpers
# -----------------------
PIXEL_TO_MM_DEFAULT = 0.1  # default calibration - let user adjust in sidebar

def predict_unet_mask(unet, tensor, device='cpu', threshold=0.5):
    tensor = tensor.to(device)
    with torch.no_grad():
        pred = unet(tensor)  # 1 x 1 x H x W
    pred_np = pred.squeeze().cpu().numpy()
    mask = (pred_np > threshold).astype(np.uint8)
    return mask

def mask_to_color_overlay(image_rgb, mask, alpha=0.4, col=(255,0,0)):
    # image_rgb: HxWx3 uint8; mask: HxW uint8 (0/1)
    heat = np.zeros_like(image_rgb, dtype=np.uint8)
    heat[mask == 1] = col
    overlay = cv2.addWeighted(image_rgb, 1-alpha, heat, alpha, 0)
    return overlay

def estimate_area(mask, pixel_to_mm):
    area_px = int(np.sum(mask))
    total_px = mask.size
    rel_pct = area_px / total_px * 100.0
    area_mm2 = area_px * (pixel_to_mm**2)
    return area_px, rel_pct, area_mm2

def classify_severity(area_mm2):
    # thresholds can be tuned
    if area_mm2 < 20:
        return "Normal"
    elif area_mm2 < 80:
        return "Mild"
    elif area_mm2 < 200:
        return "Moderate"
    else:
        return "Severe"

# -----------------------
# Load models (cached)
# -----------------------
RESNET_PATH = "./model/best_model_epoch24.pth"
UNET_PATH = "./model/best_model.pth"

with st.spinner("Loading classifier..."):
    classifier = load_classifier(RESNET_PATH, device=device)
# don't load UNet unless needed; will load lazily when a polyp is predicted

# -----------------------
# UI: main
# -----------------------
st.markdown("<h1 style='text-align:center;'>🩺 GastroVision — Classification + Polyp Segmentation</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload an endoscopic image. Classification runs always. Segmentation (UNet) runs only when the top predicted class name contains 'polyp'.</p>", unsafe_allow_html=True)
st.divider()

col_left, col_right = st.columns([1, 1.1])

with col_left:
    uploaded = st.file_uploader("📤 Upload endoscopic image (jpg/png)", type=["jpg","jpeg","png"])
    st.write("Tip: Try an image from your test set.")
    st.sidebar.markdown("## Settings")
    heat_alpha = st.sidebar.slider("Heatmap opacity (Grad-CAM)", 0.0, 1.0, 0.6, 0.05)
    seg_threshold = st.sidebar.slider("Segmentation threshold", 0.1, 0.9, 0.5, 0.05)
    pixel_to_mm = st.sidebar.number_input("Pixel → mm calibration (pixel size in mm)", value=float(PIXEL_TO_MM_DEFAULT), step=0.01, format="%.3f")

with col_right:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model info**")
    st.sidebar.write(f"Classifier: ResNet50 (fine-tuned), classes: {len(class_names)}")
    st.sidebar.write(f"Segmentation: UNet (loaded on demand)")

if uploaded is None:
    st.info("Please upload an image to start.")
    st.stop()

# Read image bytes to numpy
file_bytes = np.frombuffer(uploaded.read(), np.uint8)
bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if bgr is None:
    st.error("Could not read uploaded image. Try another file.")
    st.stop()
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
orig_h, orig_w = rgb.shape[:2]

# Show original
st.subheader("Input & Preprocessing")
colA, colB = st.columns(2)
with colA:
    st.image(rgb, caption="Original Image", use_container_width=True)

# Preprocess for classifier (224)
pil_img = Image.fromarray(rgb)
input_tensor = transform(pil_img).unsqueeze(0).to(device)  # 1xCxHxW for classifier

# Show preprocessed image (denormalized)
mean = [0.4762, 0.3054, 0.2368]
std = [0.3345, 0.2407, 0.2164]
with colB:
    denorm = input_tensor.clone().cpu().squeeze()
    for c in range(3):
        denorm[c] = denorm[c] * std[c] + mean[c]
    denorm_img = denorm.permute(1,2,0).numpy()
    denorm_img = np.clip(denorm_img * 255.0, 0, 255).astype(np.uint8)
    st.image(denorm_img, caption="Preprocessed Image (model input)", use_container_width=True)

# -----------------------
# Classification (ResNet)
# -----------------------
st.subheader("Classification")

with st.spinner("Running classifier..."):
    classifier.eval()
    with torch.no_grad():
        out = classifier(input_tensor)           # log-softmax
        probs = torch.exp(out).squeeze().cpu().numpy()

top5_idx = probs.argsort()[-5:][::-1]
top5 = [(class_names[i], float(probs[i])) for i in top5_idx]

# Display top results
st.markdown(f"### ✅ Top prediction: **{top5[0][0]}**  ({top5[0][1]*100:.2f}%)")
st.progress(top5[0][1])
df_top5 = pd.DataFrame(top5, columns=["Class","Confidence"])
st.bar_chart(df_top5.set_index("Class")["Confidence"], use_container_width=True)

# Grad-CAM visualization
st.markdown("### 🔥 Grad-CAM (model attention)")
with st.spinner("Generating Grad-CAM..."):
    try:
        cam, pred_class_idx = generate_gradcam(classifier, input_tensor, target_layer_name='layer4')
        # overlay on denorm_img resized to 224
        orig224 = cv2.resize(denorm_img, (224,224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = np.uint8((1-heat_alpha)*orig224 + heat_alpha*heatmap)
        st.image(overlay, caption=f"Grad-CAM overlay (top: {class_names[pred_class_idx]})", use_container_width=True)
    except Exception as e:
        st.error(f"Grad-CAM failed: {e}")

# -----------------------
# If predicted class is a polyp -> run UNet segmentation
# -----------------------
is_polyp = "polyp" in top5[0][0].lower() or "polyps" in top5[0][0].lower()
if is_polyp:
    st.markdown("## 🩺 Polyp segmentation (UNet)")
    # load UNet lazily
    try:
        with st.spinner("Loading UNet..."):
            unet = load_unet(UNET_PATH, device=device)
    except Exception as e:
        st.error(f"Failed to load UNet from `{UNET_PATH}`: {e}")
        st.stop()

    # Preprocess for UNet (256)
    resized_unet, tensor_unet = preprocess_for_unet(rgb)  # resized 256x256, tensor 1x3xHxW
    with st.spinner("Running UNet segmentation..."):
        mask = predict_unet_mask(unet, tensor_unet, device=device, threshold=seg_threshold)  # HxW (256,256)
        # resize mask to original displayed size for overlay (we used resized_unet)
        mask_display = mask.copy().astype(np.uint8)
        overlay_mask = mask_to_color_overlay(resized_unet, mask_display, alpha=0.35, col=(255,0,0))

    # area & severity
    area_px, rel_pct, area_mm2 = estimate_area(mask_display, pixel_to_mm)
    severity = classify_severity(area_mm2)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(resized_unet, caption="UNet input (resized)", use_container_width=True)
    with col2:
        st.image(mask_display*255, caption="Predicted Mask (binary)", use_container_width=True)
    with col3:
        st.image(overlay_mask, caption=f"Segmentation overlay — Severity: {severity}", use_container_width=True)

    st.markdown(f"**Estimated area:** `{area_px}` px  — **{area_mm2:.2f} mm²**  — **{rel_pct:.2f}%** of frame")
    st.info(f"🧠 Predicted severity: **{severity}**")

else:
    st.info("Top prediction isn't a 'polyp' class — segmentation skipped.")

# -----------------------
# Optional: download results summary
# -----------------------
def generate_report(top5, is_polyp, severity=None, area_mm2=None):
    rows = []
    rows.append(["Timestamp", time.strftime("%Y-%m-%d %H:%M:%S")])
    rows.append(["User", st.session_state.username])
    rows.append(["Top prediction", top5[0][0]])
    rows.append(["Top confidence", f"{top5[0][1]:.4f}"])
    rows.append(["All top5", "; ".join([f'{c}:{p:.4f}' for c,p in top5])])
    if is_polyp:
        rows.append(["Segmentation severity", severity])
        rows.append(["Estimated area mm2", f"{area_mm2:.2f}"])
    return rows

if st.button("📥 Download report (CSV)"):
    report = generate_report(top5, is_polyp, severity if is_polyp else None, area_mm2 if is_polyp else None)
    import io, csv
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    for r in report:
        writer.writerow(r)
    st.download_button("Download CSV", buffer.getvalue(), file_name="gastrovision_report.csv", mime="text/csv")

st.caption("© 2025 GastroVision | Built with ❤️ using Streamlit & PyTorch")
