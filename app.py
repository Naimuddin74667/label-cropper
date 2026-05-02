import os, shutil, tempfile, subprocess, uuid
from pathlib import Path
from flask import Flask, request, send_file, jsonify, render_template_string

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

UPLOAD_DIR = Path(tempfile.gettempdir()) / "label_cropper"
UPLOAD_DIR.mkdir(exist_ok=True)

def get_pdftoppm():
    p = shutil.which("pdftoppm")
    if not p:
        raise RuntimeError("pdftoppm not found on server.")
    return p

def _save_pdf(image_paths, output_pdf, dpi=200):
    from PIL import Image
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.utils import ImageReader
    c = None
    for p in image_paths:
        img = Image.open(p)
        W_pt = img.size[0] / dpi * 72
        H_pt = img.size[1] / dpi * 72
        if c is None:
            c = rl_canvas.Canvas(output_pdf, pagesize=(W_pt, H_pt))
        else:
            c.setPageSize((W_pt, H_pt))
        c.drawImage(ImageReader(p), 0, 0, W_pt, H_pt)
        c.showPage()
    if c:
        c.save()

def crop_flipkart_labels(input_pdf, output_pdf):
    from PIL import Image
    import numpy as np
    DPI = 150
    margin_px = int(0.3 / 2.54 * DPI)
    tmp_dir = tempfile.mkdtemp()
    prefix  = os.path.join(tmp_dir, "page")
    subprocess.run([get_pdftoppm(), "-r", str(DPI), input_pdf, prefix], capture_output=True, check=True)
    pages = sorted(Path(tmp_dir).glob("page-*.ppm"))
    if not pages:
        raise RuntimeError("No pages found in PDF.")
    cropped_paths = []
    for ppm_path in pages:
        img = Image.open(str(ppm_path)).convert("RGB")
        arr = np.array(img)
        H, W = arr.shape[:2]
        MIN_COL = max(300, int(H * 0.12))
        MIN_ROW = max(200, int(W * 0.12))
        label_left  = next((x for x in range(W)      if np.sum(arr[:,x,:].min(axis=1)<100) > MIN_COL), 0)
        label_right = next((x for x in range(W-1,0,-1) if np.sum(arr[:,x,:].min(axis=1)<100) > MIN_COL), W-1)
        label_top   = next((y for y in range(H)      if np.sum(arr[y,:,:].min(axis=1)<100) > MIN_ROW), 0)
        span = label_right - label_left
        solid_rows = []
        for y in range(H-1, H//4, -1):
            if np.sum(arr[y, label_left:label_right, :].min(axis=1) < 100) > span * 0.70:
                solid_rows.append(y)
                if len(solid_rows) >= 30: break
        label_bottom = max(solid_rows) if solid_rows else H-1
        arr[label_bottom+2:, :] = 255
        ct = max(0, label_top    - margin_px)
        cl = max(0, label_left   - margin_px)
        cr = min(W, label_right  + margin_px)
        cb = min(H, label_bottom + margin_px)
        out_img  = Image.fromarray(arr[ct:cb, cl:cr])
        out_path = os.path.join(tmp_dir, f"crop_{len(cropped_paths):03d}.png")
        out_img.save(out_path, dpi=(DPI, DPI))
        cropped_paths.append(out_path)
    _save_pdf(cropped_paths, output_pdf, DPI)
    shutil.rmtree(tmp_dir, ignore_errors=True)

def detect_fbf_label_boxes(arr, label_left, label_right):
    """Auto-detect the two label box positions on a FBF page."""
    import numpy as np
    H = arr.shape[0]
    gray = arr.min(axis=2)

    # Find all full-span horizontal borders
    solid_lines = [y for y in range(H)
                   if np.sum(gray[y, label_left:label_right] < 100) > (label_right - label_left) * 0.85]
    if not solid_lines:
        return [(79, 858), (943, 1722)]  # fallback

    # Cluster into border groups
    clusters, cluster = [], [solid_lines[0]]
    for y in solid_lines[1:]:
        if y - cluster[-1] <= 5:
            cluster.append(y)
        else:
            clusters.append(cluster); cluster = [y]
    clusters.append(cluster)
    borders = [min(c) for c in clusters]

    # Find the TWO biggest gaps between borders — these separate inside-label content
    # The INTER-label gap (between label1 bottom and label2 top) is the SMALLEST big gap
    # Strategy: find pairs (top, bottom) where bottom-top > 40% of page height
    label_boxes = []
    i = 0
    while i < len(borders) - 1 and len(label_boxes) < 2:
        top = borders[i]
        # Find the next border that's far enough to be a label bottom
        for j in range(i + 1, len(borders)):
            gap = borders[j] - top
            if gap > H * 0.35:  # label must be at least 35% of page height
                label_boxes.append((top, borders[j]))
                i = j
                break
        else:
            i += 1

    if len(label_boxes) < 2:
        # Fallback: use first and last border positions
        label_boxes = [(borders[0], borders[len(borders)//2]),
                       (borders[len(borders)//2+1], borders[-1])]

    return label_boxes

def crop_fbf_labels(input_pdf, output_pdf):
    from PIL import Image
    import numpy as np
    DPI = 150
    margin_px = int(0.3 / 2.54 * DPI)
    tmp_dir = tempfile.mkdtemp()
    prefix  = os.path.join(tmp_dir, "page")
    subprocess.run([get_pdftoppm(), "-r", str(DPI), input_pdf, prefix], capture_output=True, check=True)
    pages = sorted(Path(tmp_dir).glob("page-*.ppm"))
    if not pages:
        raise RuntimeError("No pages found in PDF.")

    # Detect layout from first page
    img0  = Image.open(str(pages[0])).convert("RGB")
    arr0  = np.array(img0)
    H0, W0 = arr0.shape[:2]
    gray0 = arr0.min(axis=2)
    label_left  = next((x for x in range(W0)      if np.sum(gray0[:, x] < 100) > 200), 0)
    label_right = next((x for x in range(W0-1,0,-1) if np.sum(gray0[:, x] < 100) > 200), W0-1)
    label_boxes = detect_fbf_label_boxes(arr0, label_left, label_right)

    cl = max(0, label_left  - margin_px)
    cr =       label_right  + margin_px

    cropped_paths = []
    for ppm_path in pages:
        img = Image.open(str(ppm_path)).convert("RGB")
        arr = np.array(img)
        H, W = arr.shape[:2]
        for (top, bottom) in label_boxes:
            ct = max(0, top    - margin_px)
            cb = min(H, bottom + margin_px)
            region = arr[ct:cb, cl:cr]
            dark = np.sum(region.min(axis=2) < 150)
            if dark / (region.shape[0] * region.shape[1]) < 0.005:
                continue
            out_img  = Image.fromarray(region)
            out_path = os.path.join(tmp_dir, f"crop_{len(cropped_paths):03d}.png")
            out_img.save(out_path, dpi=(DPI, DPI))
            cropped_paths.append(out_path)

    _save_pdf(cropped_paths, output_pdf, DPI)
    shutil.rmtree(tmp_dir, ignore_errors=True)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Label Cropper</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',sans-serif;background:#F0F2FF;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:24px 16px}
  h1{font-size:22px;font-weight:700;color:#1E1E2E;margin-bottom:4px;text-align:center}
  .subtitle{color:#6B7280;font-size:13px;margin-bottom:24px;text-align:center}
  .tabs{display:flex;gap:10px;margin-bottom:20px;width:100%;max-width:540px}
  .tab{flex:1;padding:11px;border-radius:12px;border:2px solid #E5E7EB;background:#fff;font-size:13px;font-weight:600;color:#6B7280;cursor:pointer;transition:all .2s;text-align:center}
  .tab.active{border-color:#5B6AF0;background:#EEF0FD;color:#5B6AF0}
  .card{background:#fff;border-radius:20px;box-shadow:0 8px 40px rgba(91,106,240,.13);padding:32px 28px;width:100%;max-width:540px}
  .desc{font-size:12px;color:#6B7280;margin-bottom:18px;padding:10px 14px;background:#F9FAFB;border-radius:8px;border-left:3px solid #5B6AF0;line-height:1.6}
  .drop-zone{border:2.5px dashed #5B6AF0;border-radius:14px;background:#F5F6FF;padding:32px 20px;text-align:center;cursor:pointer;transition:all .2s;position:relative;margin-bottom:16px}
  .drop-zone:hover,.drop-zone.dragover{background:#EEF0FD;border-color:#4454D6}
  .drop-zone input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%}
  .drop-zone .dz-icon{font-size:32px;margin-bottom:6px}
  .drop-zone h3{color:#5B6AF0;font-size:14px;font-weight:600}
  .drop-zone p{color:#9CA3AF;font-size:12px;margin-top:3px}
  .file-name{color:#1E1E2E;font-size:12px;margin-top:6px;font-weight:500;display:none}
  .btn{width:100%;padding:13px;background:#5B6AF0;color:#fff;border:none;border-radius:12px;font-size:14px;font-weight:600;cursor:pointer;transition:background .2s;display:flex;align-items:center;justify-content:center;gap:8px}
  .btn:hover{background:#4454D6}
  .btn:disabled{background:#A5B4FC;cursor:not-allowed}
  .progress-wrap{margin:16px 0 0;display:none}
  .progress-bar{height:8px;background:#E5E7EB;border-radius:99px;overflow:hidden}
  .progress-fill{height:100%;background:#5B6AF0;border-radius:99px;width:0%;transition:width .4s ease}
  .status{font-size:12px;color:#6B7280;margin-top:6px;text-align:center}
  .result{display:none;margin-top:16px;background:#F0FDF4;border:1.5px solid #86EFAC;border-radius:12px;padding:14px 18px}
  .result h3{color:#166534;font-size:13px;font-weight:600;margin-bottom:10px}
  .dl-btn{display:flex;align-items:center;justify-content:center;gap:8px;width:100%;padding:11px;background:#22C55E;color:#fff;border:none;border-radius:10px;font-size:13px;font-weight:600;cursor:pointer;text-decoration:none;transition:background .2s}
  .dl-btn:hover{background:#16A34A}
  .another{margin-top:8px;width:100%;padding:9px;background:#fff;color:#5B6AF0;border:1.5px solid #5B6AF0;border-radius:10px;font-size:12px;font-weight:600;cursor:pointer}
  .another:hover{background:#EEF0FD}
  .error{display:none;margin-top:14px;background:#FEF2F2;border:1.5px solid #FCA5A5;border-radius:12px;padding:12px 16px;color:#991B1B;font-size:12px}
</style>
</head>
<body>
<h1>📦 Label Cropper</h1>
<p class="subtitle">Flipkart Shipping & FBF Box Sticker — auto crop & download</p>
<div class="tabs">
  <div class="tab active" onclick="switchTab('flipkart',this)">🏷️ Flipkart Shipping Label</div>
  <div class="tab" onclick="switchTab('fbf',this)">📦 FBF Box Sticker</div>
</div>
<div class="card">
  <div class="desc" id="desc">Upload your Flipkart / E-Kart shipping invoice PDF. Auto-detects and crops each shipping label, removes the invoice section. Equal 0.3cm gap on all sides.</div>
  <div class="drop-zone" id="dropZone">
    <input type="file" id="fileInput" accept=".pdf">
    <div class="dz-icon">📄</div>
    <h3>Click to upload PDF</h3>
    <p>or drag and drop here</p>
    <div class="file-name" id="fileName"></div>
  </div>
  <button class="btn" id="cropBtn" disabled onclick="cropPDF()">✂ Crop Labels</button>
  <div class="progress-wrap" id="progressWrap">
    <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
    <div class="status" id="statusMsg">Processing...</div>
  </div>
  <div class="result" id="resultBox">
    <h3>✅ Done! Your cropped PDF is ready.</h3>
    <a class="dl-btn" id="downloadBtn" href="#" download>⬇ Download Cropped PDF</a>
    <button class="another" onclick="resetApp()">🔄 Crop Another PDF</button>
  </div>
  <div class="error" id="errorBox"></div>
</div>
<script>
  let mode='flipkart',selectedFile=null;
  const descs={
    flipkart:'Upload your Flipkart / E-Kart shipping invoice PDF. Auto-detects and crops each shipping label, removes the invoice section. Equal 0.3cm gap on all sides.',
    fbf:'Upload your FBF Box Sticker PDF (2 labels per page). Each label is split into its own page — blank pages removed automatically. Works with all label sizes. Equal 0.3cm gap on all sides.'
  };
  function switchTab(m,el){mode=m;document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));el.classList.add('active');document.getElementById('desc').textContent=descs[m];resetApp();}
  const fileInput=document.getElementById('fileInput'),dropZone=document.getElementById('dropZone'),cropBtn=document.getElementById('cropBtn'),fileName=document.getElementById('fileName');
  fileInput.addEventListener('change',e=>setFile(e.target.files[0]));
  dropZone.addEventListener('dragover',e=>{e.preventDefault();dropZone.classList.add('dragover');});
  dropZone.addEventListener('dragleave',()=>dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop',e=>{e.preventDefault();dropZone.classList.remove('dragover');setFile(e.dataTransfer.files[0]);});
  function setFile(f){if(!f||!f.name.endsWith('.pdf')){alert('Please select a PDF file.');return;}selectedFile=f;fileName.textContent='📄 '+f.name;fileName.style.display='block';cropBtn.disabled=false;document.getElementById('resultBox').style.display='none';document.getElementById('errorBox').style.display='none';}
  function setProgress(pct,msg){document.getElementById('progressFill').style.width=pct+'%';document.getElementById('statusMsg').textContent=msg;}
  async function cropPDF(){
    if(!selectedFile)return;
    cropBtn.disabled=true;cropBtn.textContent='⏳ Processing...';
    document.getElementById('progressWrap').style.display='block';
    document.getElementById('resultBox').style.display='none';
    document.getElementById('errorBox').style.display='none';
    setProgress(15,'Uploading PDF...');
    const fd=new FormData();fd.append('pdf',selectedFile);fd.append('mode',mode);
    try{
      setProgress(40,'Cropping labels...');
      const res=await fetch('/crop',{method:'POST',body:fd});
      setProgress(85,'Assembling output PDF...');
      const data=await res.json();
      if(data.error)throw new Error(data.error);
      setProgress(100,'Done!');
      const dlBtn=document.getElementById('downloadBtn');
      dlBtn.href='/download/'+data.file_id;dlBtn.download=data.filename;
      document.getElementById('resultBox').style.display='block';
      document.getElementById('progressWrap').style.display='none';
    }catch(err){
      document.getElementById('errorBox').textContent='❌ '+err.message;
      document.getElementById('errorBox').style.display='block';
      document.getElementById('progressWrap').style.display='none';
    }
    cropBtn.disabled=false;cropBtn.innerHTML='✂ Crop Labels';
  }
  function resetApp(){selectedFile=null;fileInput.value='';fileName.style.display='none';cropBtn.disabled=true;document.getElementById('resultBox').style.display='none';document.getElementById('errorBox').style.display='none';document.getElementById('progressWrap').style.display='none';setProgress(0,'');}
</script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/crop', methods=['POST'])
def crop():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f    = request.files['pdf']
    mode = request.form.get('mode', 'flipkart')
    if not f.filename.endswith('.pdf'):
        return jsonify({'error': 'Please upload a PDF file'}), 400
    file_id  = str(uuid.uuid4())
    in_path  = str(UPLOAD_DIR / f"{file_id}_input.pdf")
    out_name = Path(f.filename).stem + "_cropped.pdf"
    out_path = str(UPLOAD_DIR / f"{file_id}_output.pdf")
    f.save(in_path)
    try:
        if mode == 'fbf':
            crop_fbf_labels(in_path, out_path)
        else:
            crop_flipkart_labels(in_path, out_path)
    except Exception as e:
        os.remove(in_path)
        return jsonify({'error': str(e)}), 500
    os.remove(in_path)
    return jsonify({'file_id': file_id, 'filename': out_name})

@app.route('/download/<file_id>')
def download(file_id):
    file_id  = file_id.replace('/', '').replace('..', '')
    out_path = UPLOAD_DIR / f"{file_id}_output.pdf"
    if not out_path.exists():
        return 'File not found', 404
    return send_file(str(out_path), as_attachment=True,
                     download_name=out_path.name.replace(f"{file_id}_", ""))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
